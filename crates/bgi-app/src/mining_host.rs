//! 自动采矿动作闭环（C# `LinneaMiningTask` 的 Rust 移植）。
//!
//! 分两层：
//! - [`run_mining_loop`]：纯流程编排，只通过 [`MiningIo`] 与外界交互，因此可以用脚本化
//!   的假 IO 逐帧验证「进瞄准模式 → 中键元素视野 → 微调 → 挖矿 → 左转扫描 → 收尾释放」。
//! - [`MiningHost`]：真实实现，负责抓图、采矿推理、发送受校验的鼠标/键盘输入。
//!
//! 安全边界：整轮任务持有一份输入暂停租约；所有按下/移动都走 `*_checked` 输入；任何退出
//! 路径都会松开中键、左键并 `release_all_keys`。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bgi_capture::{CapturedFrame, CapturerBackend, FrameBufferPool};
use bgi_control::{CaptureMode, LogLevel, TaskOutcome, TaskReport};
use bgi_task::{
    AimCommand, AimController, AimParams, MiningObservation, PauseController, PauseLease,
};
use tokio::sync::mpsc;

use crate::capture_host::ActiveCapturer;
use crate::cv_service::CvOcrServices;
use crate::pathing_host::PathingEvent;

pub(crate) const MINING_TASK_NAME: &str = "自动采矿（莉奈娅）";
/// 游戏默认「切换瞄准模式」按键（C# `KeyBindingsConfig.SwitchAimingMode = R`）。
const AIMING_MODE_VK: u16 = 0x52;
const AIMING_MODE_HOLD_MS: u64 = 60;
const AIMING_MODE_SETTLE_MS: u64 = 400;
const SIGHT_HOLD_SETTLE_MS: u64 = 1500;
const SIGHT_RELEASE_SETTLE_MS: u64 = 100;
const AIM_STEP_SETTLE_MS: u64 = 150;
const MINE_SETTLE_MS: u64 = 2000;
const RELEASE_SETTLE_MS: u64 = 300;
const RECENTER_SETTLE_MS: u64 = 800;
const LIFT_SETTLE_MS: u64 = 10;
const CAPTURE_RETRY_MS: u64 = 80;
const CAPTURE_RETRIES: usize = 3;

/// 一轮采矿任务的参数（来自 `TaskLimitsConfig` 与路径动作参数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MiningRunConfig {
    pub mine_count: u32,
    pub scan_rounds: u32,
    pub max_shots: u32,
    pub total_timeout_ms: u64,
}

impl MiningRunConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.mine_count == 0 || self.mine_count > 999 {
            return Err("挖矿次数必须在 1..=999".into());
        }
        if self.scan_rounds < self.mine_count || self.scan_rounds > 999 {
            return Err("扫描轮数必须不小于挖矿次数且不超过 999".into());
        }
        if !(1..=20).contains(&self.max_shots) {
            return Err("射击次数上限必须在 1..=20".into());
        }
        if !(1000..=3_600_000).contains(&self.total_timeout_ms) {
            return Err("采矿总超时必须在 1000..=3600000 ms".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MiningOutcome {
    pub mined: u32,
    pub shots: u32,
    pub rounds: u32,
    /// 结束时没有射中任何目标（靠补偿或轮数用尽退出）。
    pub exhausted: bool,
}

/// 闭环所需的外部能力，真实实现见 [`MiningIoAdapter`]。
pub(crate) trait MiningIo {
    /// 任务开始至今的毫秒数（也是元素视野刷新与超时的时钟）。
    fn elapsed_ms(&self) -> u64;
    fn is_cancelled(&self) -> bool;
    fn sleep_ms(&mut self, ms: u64);
    fn enter_aiming_mode(&mut self) -> Result<(), String>;
    fn leave_aiming_mode(&mut self) -> Result<(), String>;
    fn press_sight(&mut self) -> Result<(), String>;
    fn release_sight(&mut self) -> Result<(), String>;
    fn aim_move(&mut self, dx: i32, dy: i32) -> Result<(), String>;
    fn mine_click(&mut self) -> Result<(), String>;
    /// 抓一帧并识别；`Ok(None)` 表示当前帧尚未就绪。
    fn detect(&mut self, prefer_right: bool) -> Result<Option<MiningObservation>, String>;
    fn log(&mut self, level: LogLevel, message: String);
}

/// 采矿主循环。
///
/// 目标对齐后射一枪；未命中则松手中键、按累计增量反向补偿，然后左转继续扫描。
pub(crate) fn run_mining_loop<I: MiningIo + ?Sized>(
    io: &mut I,
    config: MiningRunConfig,
    params: AimParams,
) -> Result<MiningOutcome, String> {
    config.validate()?;
    let prefer_right = config.scan_rounds > 1;
    let mut mined = 0u32;
    let mut shots = 0u32;
    let mut rounds_done = 0u32;
    let mut sight_held = false;
    let mut exhausted = true;

    io.log(
        LogLevel::Info,
        format!(
            "开始采矿：挖矿 {} 次，最多 {} 轮扫描，射击上限 {} 次",
            config.mine_count, config.scan_rounds, config.max_shots
        ),
    );

    let result = (|| -> Result<MiningOutcome, String> {
        io.enter_aiming_mode()?;
        io.sleep_ms(AIMING_MODE_SETTLE_MS);

        for round in 0..config.scan_rounds {
            ensure_running(io, config.total_timeout_ms)?;
            io.press_sight()?;
            sight_held = true;
            io.sleep_ms(SIGHT_HOLD_SETTLE_MS);
            io.log(
                LogLevel::Info,
                format!("第 {}/{} 轮扫描", round + 1, config.scan_rounds),
            );

            let Some(mut observation) = io.detect(prefer_right)? else {
                io.log(LogLevel::Warn, "当前帧未就绪，跳过本轮".into());
                relinquish_sight(io, &mut sight_held)?;
                turn_left(io, params)?;
                rounds_done = round + 1;
                continue;
            };

            let mut aim = AimController::new(params, io.elapsed_ms());
            let mut fired = false;
            let mut lost = false;

            loop {
                ensure_running(io, config.total_timeout_ms)?;
                if aim.needs_refresh(io.elapsed_ms()) {
                    io.release_sight()?;
                    sight_held = false;
                    io.sleep_ms(SIGHT_RELEASE_SETTLE_MS);
                    io.press_sight()?;
                    sight_held = true;
                    io.sleep_ms(SIGHT_HOLD_SETTLE_MS);
                    aim.mark_refreshed(io.elapsed_ms());
                    io.log(LogLevel::Info, "刷新元素视野".into());
                    match io.detect(prefer_right)? {
                        Some(fresh) => {
                            observation = fresh;
                            continue;
                        }
                        None => {
                            lost = true;
                            break;
                        }
                    }
                }

                match aim.decide(&observation, io.elapsed_ms()) {
                    AimCommand::RefreshSight => continue,
                    AimCommand::Aim { dx, dy } => {
                        io.aim_move(dx, dy)?;
                        io.sleep_ms(AIM_STEP_SETTLE_MS);
                        match io.detect(prefer_right)? {
                            Some(fresh) => observation = fresh,
                            None => {
                                lost = true;
                                break;
                            }
                        }
                    }
                    AimCommand::Mine { aligned, lift_px } => {
                        if shots >= config.max_shots {
                            io.log(
                                LogLevel::Warn,
                                format!("已达射击上限 {} 次，提前结束", config.max_shots),
                            );
                            exhausted = false;
                            return Ok(MiningOutcome {
                                mined,
                                shots,
                                rounds: round + 1,
                                exhausted,
                            });
                        }
                        io.release_sight()?;
                        sight_held = false;
                        io.sleep_ms(RELEASE_SETTLE_MS);
                        if lift_px != 0 {
                            io.aim_move(0, lift_px)?;
                            io.sleep_ms(LIFT_SETTLE_MS);
                        }
                        io.mine_click()?;
                        shots += 1;
                        io.sleep_ms(MINE_SETTLE_MS);
                        if aligned {
                            mined += 1;
                            exhausted = false;
                            io.log(
                                LogLevel::Info,
                                format!("已对准并挖矿（{mined}/{}）", config.mine_count),
                            );
                        } else {
                            io.log(
                                LogLevel::Warn,
                                "重试用尽，补射一枪（不计入挖矿次数）".into(),
                            );
                        }
                        fired = true;
                        break;
                    }
                    AimCommand::Lost => {
                        lost = true;
                        break;
                    }
                    AimCommand::Exhausted => {
                        io.log(LogLevel::Warn, "没有剩余重试次数".into());
                        break;
                    }
                }
            }

            rounds_done = round + 1;
            if fired {
                if mined >= config.mine_count {
                    return Ok(MiningOutcome {
                        mined,
                        shots,
                        rounds: rounds_done,
                        exhausted,
                    });
                }
                continue;
            }

            relinquish_sight(io, &mut sight_held)?;
            if lost {
                let progress = aim.progress();
                if progress.total_dx != 0 || progress.total_dy != 0 {
                    io.press_sight()?;
                    sight_held = true;
                    io.sleep_ms(SIGHT_HOLD_SETTLE_MS);
                    io.aim_move(-progress.total_dx, -progress.total_dy)?;
                    io.sleep_ms(RECENTER_SETTLE_MS);
                    relinquish_sight(io, &mut sight_held)?;
                }
            }
            turn_left(io, params)?;
        }

        Ok(MiningOutcome {
            mined,
            shots,
            rounds: rounds_done,
            exhausted,
        })
    })();

    if sight_held {
        let _ = io.release_sight();
    }
    let leave = io.leave_aiming_mode();
    match result {
        Ok(outcome) => {
            leave?;
            Ok(outcome)
        }
        Err(error) => Err(error),
    }
}

fn ensure_running<I: MiningIo + ?Sized>(io: &mut I, total_timeout_ms: u64) -> Result<(), String> {
    if io.is_cancelled() {
        return Err("已手动停止采矿".into());
    }
    if io.elapsed_ms() >= total_timeout_ms {
        return Err(format!("采矿超时（{} 秒）", total_timeout_ms / 1000));
    }
    Ok(())
}

fn relinquish_sight<I: MiningIo + ?Sized>(io: &mut I, sight_held: &mut bool) -> Result<(), String> {
    if *sight_held {
        io.release_sight()?;
        *sight_held = false;
    }
    io.sleep_ms(RELEASE_SETTLE_MS);
    Ok(())
}

fn turn_left<I: MiningIo + ?Sized>(io: &mut I, params: AimParams) -> Result<(), String> {
    let dx = params.scan_turn_dx();
    if dx == 0 {
        return Ok(());
    }
    io.aim_move(dx, 0)?;
    io.sleep_ms(RECENTER_SETTLE_MS);
    Ok(())
}

/// 按窗口客户区与 DPI 推导瞄准参数；读不到尺寸时退回 1920×1080。
pub(crate) fn aim_params_for_window(hwnd: isize) -> AimParams {
    let (width, height) = bgi_input::window_client_size(hwnd).unwrap_or((1920, 1080));
    AimParams::from_frame(width, height, f64::from(bgi_input::window_dpi_scale(hwnd)))
}

/// 真实 IO 适配器：把抓图、推理与受校验输入接到闭环上。
///
/// 两个宿主（采矿任务与路径动作）字段结构相同，共用这一份实现，避免安全校验出现两份。
pub(crate) struct MiningIoAdapter<'a> {
    pub hwnd: isize,
    pub mode: CaptureMode,
    pub cancel: &'a AtomicBool,
    pub capturer: &'a Mutex<ActiveCapturer>,
    pub pool: &'a FrameBufferPool,
    pub cv_ocr: &'a Mutex<CvOcrServices>,
    pub pauses: &'a PauseController,
    pub log: Box<dyn FnMut(LogLevel, String) + 'a>,
    started: Instant,
    pause_lease: Option<PauseLease>,
}

impl<'a> MiningIoAdapter<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        hwnd: isize,
        mode: CaptureMode,
        cancel: &'a AtomicBool,
        capturer: &'a Mutex<ActiveCapturer>,
        pool: &'a FrameBufferPool,
        cv_ocr: &'a Mutex<CvOcrServices>,
        pauses: &'a PauseController,
        log: Box<dyn FnMut(LogLevel, String) + 'a>,
    ) -> Self {
        Self {
            hwnd,
            mode,
            cancel,
            capturer,
            pool,
            cv_ocr,
            pauses,
            log,
            started: Instant::now(),
            pause_lease: None,
        }
    }

    /// 任务结束后的兜底释放：松中键、松左键、清空按键。
    pub(crate) fn release_inputs(&mut self) {
        let _ = bgi_input::mouse_middle_up_checked();
        let _ = bgi_input::mouse_left_up_checked();
        bgi_input::release_all_keys();
        self.pause_lease = None;
    }

    fn frame_size(&self) -> Result<(u32, u32), String> {
        bgi_input::window_client_size(self.hwnd).ok_or_else(|| "读取游戏客户区尺寸失败".to_string())
    }

    fn acquire_pause(&mut self) -> Result<(), String> {
        if self.pause_lease.is_none() {
            self.pause_lease = Some(
                self.pauses
                    .pause()
                    .map_err(|error| format!("获取采矿输入暂停租约失败: {error}"))?,
            );
        }
        Ok(())
    }

    fn capture(&self) -> Result<Option<CapturedFrame>, String> {
        let mut guard = self.capturer.lock().unwrap();
        let need_reopen = match &*guard {
            Some((hwnd, mode, _)) => *hwnd != self.hwnd || *mode != self.mode,
            None => true,
        };
        if need_reopen {
            *guard = None;
            let backend = CapturerBackend::open(self.hwnd, self.mode, 16)
                .map_err(|error| format!("打开抓图会话失败: {error}"))?;
            *guard = Some((self.hwnd, self.mode, backend));
        }
        let (_, _, backend) = guard.as_mut().unwrap();
        match backend.capture(self.pool) {
            Ok(frame) => Ok(Some(frame)),
            Err(bgi_capture::CaptureError::FrameNotReady) => Ok(None),
            Err(error) => {
                *guard = None;
                Err(error.to_string())
            }
        }
    }
}

impl MiningIo for MiningIoAdapter<'_> {
    fn elapsed_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn sleep_ms(&mut self, ms: u64) {
        let mut remaining = ms;
        while remaining > 0 && !self.is_cancelled() {
            let chunk = remaining.min(25);
            std::thread::sleep(Duration::from_millis(chunk));
            remaining -= chunk;
        }
    }

    fn enter_aiming_mode(&mut self) -> Result<(), String> {
        let frame_size = self.frame_size()?;
        bgi_input::key_tap_checked(self.hwnd, AIMING_MODE_VK, AIMING_MODE_HOLD_MS, frame_size)
            .map_err(|error| format!("进入瞄准模式失败: {error}"))
    }

    fn leave_aiming_mode(&mut self) -> Result<(), String> {
        let frame_size = self.frame_size()?;
        bgi_input::key_tap_checked(self.hwnd, AIMING_MODE_VK, AIMING_MODE_HOLD_MS, frame_size)
            .map_err(|error| format!("退出瞄准模式失败: {error}"))
    }

    fn press_sight(&mut self) -> Result<(), String> {
        self.acquire_pause()?;
        let frame_size = self.frame_size()?;
        bgi_input::mouse_middle_down_checked(self.hwnd, frame_size)
            .map_err(|error| format!("按住元素视野失败: {error}"))
    }

    fn release_sight(&mut self) -> Result<(), String> {
        bgi_input::mouse_middle_up_checked().map_err(|error| format!("松开元素视野失败: {error}"))
    }

    fn aim_move(&mut self, dx: i32, dy: i32) -> Result<(), String> {
        self.acquire_pause()?;
        bgi_input::mouse_move_rel_checked(self.hwnd, dx, dy)
            .map_err(|error| format!("瞄准移动失败: {error}"))
    }

    fn mine_click(&mut self) -> Result<(), String> {
        self.acquire_pause()?;
        let frame_size = self.frame_size()?;
        bgi_input::mouse_left_down_checked(self.hwnd, frame_size)
            .map_err(|error| format!("挖矿按下左键失败: {error}"))?;
        std::thread::sleep(Duration::from_millis(40));
        bgi_input::mouse_left_up_checked().map_err(|error| format!("挖矿松开左键失败: {error}"))
    }

    fn detect(&mut self, prefer_right: bool) -> Result<Option<MiningObservation>, String> {
        for attempt in 0..CAPTURE_RETRIES {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(CAPTURE_RETRY_MS));
            }
            let Some(frame) = self.capture()? else {
                continue;
            };
            let mut guard = self.cv_ocr.lock().unwrap();
            let observation = guard
                .detect_mining_observation(
                    frame.data.as_slice(),
                    frame.width,
                    frame.height,
                    frame.format.is_bgr24(),
                    prefer_right,
                )
                .map_err(|error| format!("采矿识别失败: {error}"))?;
            return Ok(Some(observation));
        }
        Ok(None)
    }

    fn log(&mut self, level: LogLevel, message: String) {
        (self.log)(level, message);
    }
}

/// 路径动作 `linnea_mining` 使用的安全上限（来自 `TaskLimitsConfig`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MiningActionLimits {
    pub max_shots: u32,
    pub total_timeout_ms: u64,
}

impl From<&bgi_control::TaskLimitsConfig> for MiningActionLimits {
    fn from(limits: &bgi_control::TaskLimitsConfig) -> Self {
        Self {
            max_shots: u32::from(limits.mining_max_shots),
            total_timeout_ms: limits.mining_total_timeout_ms,
        }
    }
}

/// 采矿任务的真实宿主：抓图 + 推理 + 受校验输入。
pub(crate) struct MiningHost {
    pub task_id: u64,
    pub hwnd: isize,
    pub mode: CaptureMode,
    pub cancel: Arc<AtomicBool>,
    pub capturer: Arc<Mutex<ActiveCapturer>>,
    pub pool: Arc<FrameBufferPool>,
    pub cv_ocr: Arc<Mutex<CvOcrServices>>,
    pub events_tx: mpsc::UnboundedSender<PathingEvent>,
    pub pauses: PauseController,
}

impl MiningHost {
    pub(crate) fn run_mining(&mut self, config: MiningRunConfig) -> Result<TaskReport, String> {
        let started = Instant::now();
        let task_id = self.task_id;
        let events_tx = self.events_tx.clone();
        let mut io = MiningIoAdapter::new(
            self.hwnd,
            self.mode,
            self.cancel.as_ref(),
            self.capturer.as_ref(),
            self.pool.as_ref(),
            self.cv_ocr.as_ref(),
            &self.pauses,
            Box::new(move |level, message| {
                let _ = events_tx.send(PathingEvent::Log {
                    task_id,
                    level,
                    message,
                });
            }),
        );
        let params = self.params();
        let outcome = run_mining_loop(&mut io, config, params);
        io.release_inputs();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let mut report = TaskReport {
            task_id,
            task_name: MINING_TASK_NAME.into(),
            outcome: TaskOutcome::Success,
            runs_completed: 0,
            runs_requested: config.mine_count,
            current_index: None,
            total_points: config.mine_count as usize,
            failure: None,
            stage: "mining".into(),
            elapsed_ms,
        };
        match outcome {
            Ok(done) => {
                report.runs_completed = done.mined;
                report.outcome = if done.mined >= config.mine_count {
                    TaskOutcome::Success
                } else if done.mined > 0 {
                    TaskOutcome::Partial
                } else {
                    TaskOutcome::Failed
                };
                report.failure = if done.mined >= config.mine_count {
                    None
                } else if done.exhausted {
                    Some(format!(
                        "扫描 {} 轮后未再找到可挖矿目标，射击 {} 次",
                        done.rounds, done.shots
                    ))
                } else {
                    Some(format!("射击 {} 次仍未挖到矿", done.shots))
                };
                Ok(report)
            }
            Err(error) => {
                report.outcome = if self.cancel.load(Ordering::Acquire) {
                    TaskOutcome::Cancelled
                } else {
                    TaskOutcome::Failed
                };
                report.failure = Some(error);
                Ok(report)
            }
        }
    }

    fn params(&self) -> AimParams {
        aim_params_for_window(self.hwnd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgi_task::{MiningRect, MiningTarget};

    struct FakeIo {
        elapsed: u64,
        observations: Vec<Option<MiningObservation>>,
        events: Vec<String>,
        cancelled: bool,
        detect_calls: usize,
    }

    impl FakeIo {
        fn new(observations: Vec<Option<MiningObservation>>) -> Self {
            Self {
                elapsed: 0,
                observations,
                events: Vec::new(),
                cancelled: false,
                detect_calls: 0,
            }
        }
    }

    impl MiningIo for FakeIo {
        fn elapsed_ms(&self) -> u64 {
            self.elapsed
        }

        fn is_cancelled(&self) -> bool {
            self.cancelled
        }

        fn sleep_ms(&mut self, ms: u64) {
            self.elapsed += ms;
        }

        fn enter_aiming_mode(&mut self) -> Result<(), String> {
            self.events.push("enter_aim".into());
            Ok(())
        }

        fn leave_aiming_mode(&mut self) -> Result<(), String> {
            self.events.push("leave_aim".into());
            Ok(())
        }

        fn press_sight(&mut self) -> Result<(), String> {
            self.events.push("sight_down".into());
            Ok(())
        }

        fn release_sight(&mut self) -> Result<(), String> {
            self.events.push("sight_up".into());
            Ok(())
        }

        fn aim_move(&mut self, dx: i32, dy: i32) -> Result<(), String> {
            self.events.push(format!("move({dx},{dy})"));
            Ok(())
        }

        fn mine_click(&mut self) -> Result<(), String> {
            self.events.push("click".into());
            Ok(())
        }

        fn detect(&mut self, _prefer_right: bool) -> Result<Option<MiningObservation>, String> {
            self.detect_calls += 1;
            if self.observations.is_empty() {
                return Ok(None);
            }
            let index = (self.detect_calls - 1).min(self.observations.len() - 1);
            Ok(self.observations[index])
        }

        fn log(&mut self, _level: LogLevel, message: String) {
            self.events.push(format!("log:{message}"));
        }
    }

    fn target(cx: f64, cy: f64) -> MiningObservation {
        MiningObservation {
            frame_width: 1920,
            frame_height: 1080,
            cluster_count: 1,
            target: Some(MiningTarget {
                rect: MiningRect {
                    x: (cx - 50.0) as i32,
                    y: (cy - 50.0) as i32,
                    width: 100,
                    height: 100,
                },
                center_x: cx,
                center_y: cy,
            }),
        }
    }

    fn empty() -> MiningObservation {
        MiningObservation {
            frame_width: 1920,
            frame_height: 1080,
            cluster_count: 0,
            target: None,
        }
    }

    fn config() -> MiningRunConfig {
        MiningRunConfig {
            mine_count: 1,
            scan_rounds: 1,
            max_shots: 3,
            total_timeout_ms: 180_000,
        }
    }

    fn params() -> AimParams {
        AimParams::from_frame(1920, 1080, 1.0)
    }

    #[test]
    fn aligned_target_mines_immediately_and_leaves_aiming_mode() {
        let mut io = FakeIo::new(vec![Some(target(980.0, 545.0))]);
        let outcome = run_mining_loop(&mut io, config(), params()).unwrap();
        assert_eq!(outcome.mined, 1);
        assert_eq!(outcome.shots, 1);
        assert!(!outcome.exhausted);
        let events = io.events.join("|");
        assert!(events.starts_with("log:开始采矿"), "{events}");
        assert!(events.contains("enter_aim"), "{events}");
        assert!(events.contains("sight_down"), "{events}");
        assert!(events.contains("click"), "{events}");
        assert!(events.ends_with("leave_aim"), "{events}");
        // 对准即射，不应出现任何微调移动。
        assert!(!events.contains("move("), "{events}");
    }

    #[test]
    fn off_center_target_is_aimed_before_the_shot() {
        let mut io = FakeIo::new(vec![
            Some(target(1200.0, 540.0)),
            Some(target(1100.0, 540.0)),
            Some(target(1000.0, 540.0)),
        ]);
        let outcome = run_mining_loop(&mut io, config(), params()).unwrap();
        assert_eq!((outcome.mined, outcome.shots), (1, 1));
        let events = io.events.join("|");
        assert!(events.contains("move(108,0)"), "{events}");
        assert!(events.contains("move(63,0)"), "{events}");
    }

    #[test]
    fn lost_target_recenters_then_turns_and_reports_partial() {
        let config = MiningRunConfig {
            mine_count: 1,
            scan_rounds: 2,
            ..config()
        };
        let mut io = FakeIo::new(vec![
            Some(target(1200.0, 540.0)),
            Some(empty()),
            Some(empty()),
        ]);
        let outcome = run_mining_loop(&mut io, config, params()).unwrap();
        assert_eq!(outcome.mined, 0);
        assert_eq!(outcome.rounds, 2);
        assert!(outcome.exhausted);
        let events = io.events.join("|");
        assert!(events.contains("move(108,0)"), "{events}");
        // 丢失后的反向补偿：-108 回中，再左转 -250。
        assert!(events.contains("move(-108,0)"), "{events}");
        assert!(events.contains("move(-250,0)"), "{events}");
        assert_eq!(io.events.iter().filter(|e| *e == "click").count(), 0);
    }

    #[test]
    fn shot_cap_stops_the_run_before_over_clicking() {
        let config = MiningRunConfig {
            mine_count: 5,
            scan_rounds: 5,
            max_shots: 1,
            ..config()
        };
        let mut io = FakeIo::new(vec![Some(target(1200.0, 540.0)); 8]);
        let outcome = run_mining_loop(&mut io, config, params()).unwrap();
        assert_eq!(outcome.shots, 1);
        assert_eq!(io.events.iter().filter(|e| *e == "click").count(), 1);
        assert!(io.events.iter().any(|e| e.contains("射击上限")));
    }

    #[test]
    fn cancel_and_timeout_stop_the_loop_but_still_release_sight() {
        let mut cancelled = FakeIo::new(vec![Some(target(1200.0, 540.0)); 8]);
        cancelled.cancelled = true;
        assert!(run_mining_loop(&mut cancelled, config(), params()).is_err());

        let config = MiningRunConfig {
            total_timeout_ms: 1000,
            ..config()
        };
        let mut timed_out = FakeIo::new(vec![Some(target(1200.0, 540.0)); 8]);
        let error = run_mining_loop(&mut timed_out, config, params()).unwrap_err();
        assert!(error.contains("超时"), "{error}");
    }

    #[test]
    fn invalid_config_is_rejected_before_any_input() {
        let mut io = FakeIo::new(vec![]);
        let bad = MiningRunConfig {
            scan_rounds: 0,
            ..config()
        };
        assert!(run_mining_loop(&mut io, bad, params()).is_err());
        assert!(io.events.is_empty(), "配置非法时不得发送任何输入");
    }
}
