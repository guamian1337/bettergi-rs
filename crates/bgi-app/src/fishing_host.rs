use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bgi_capture::{CapturedFrame, CapturerBackend, FrameBufferPool};
use bgi_control::{CaptureMode, LogLevel, TaskOutcome, TaskReport};
use bgi_task::{
    FishingAction, FishingConfig, FishingController, FishingObservation, PauseController,
    PauseLease,
};
use tokio::sync::mpsc;

use crate::capture_host::ActiveCapturer;
use crate::cv_service::CvOcrServices;
use crate::pathing_host::PathingEvent;

const FISHING_TASK_NAME: &str = "自动钓鱼";
const CAPTURE_INTERVAL: Duration = Duration::from_millis(67);
const REPEAT_LOG_INTERVAL: Duration = Duration::from_secs(1);
const HOOK_DELAY: Duration = Duration::from_millis(800);

pub(crate) struct FishingHost {
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

impl FishingHost {
    pub(crate) fn run_fishing_attempt(
        &mut self,
        dry_run: bool,
        config: FishingConfig,
    ) -> Result<TaskReport, String> {
        self.run_fishing_attempt_with(
            dry_run,
            config,
            |host| host.capture_raw_frame(),
            |host, frame| host.detect_fishing(frame),
            FishingHost::validate_mouse_target,
            |host| {
                host.pauses
                    .pause()
                    .map_err(|e| format!("获取钓鱼输入暂停租约失败: {e}"))
            },
            send_mouse_action,
            release_mouse_input,
        )
    }

    // Closures keep the execution loop shared with tests without routing production input through
    // a trait object. The pause lease belongs to this attempt and outlives every held-button phase.
    #[allow(clippy::too_many_arguments)]
    fn run_fishing_attempt_with<C, D, V, P, S, R>(
        &mut self,
        dry_run: bool,
        config: FishingConfig,
        mut capture: C,
        mut detect: D,
        mut validate_target: V,
        mut acquire_pause: P,
        mut send_action: S,
        release_input: R,
    ) -> Result<TaskReport, String>
    where
        C: FnMut(&mut FishingHost) -> Result<Option<CapturedFrame>, String>,
        D: FnMut(&FishingHost, &CapturedFrame) -> Result<(FishingObservation, (u32, u32)), String>,
        V: FnMut(&FishingHost, (u32, u32)) -> Result<(), String>,
        P: FnMut(&FishingHost) -> Result<PauseLease, String>,
        S: FnMut(&FishingHost, FishingAction, (u32, u32)) -> Result<(), String>,
        R: FnMut() -> Result<(), String>,
    {
        let started = Instant::now();
        let mut controller = FishingController::try_new(config).map_err(str::to_string)?;
        let mut report = new_report(self.task_id, dry_run);
        let dry_run_pause = if dry_run {
            Some(
                self.pauses
                    .pause()
                    .map_err(|e| format!("获取钓鱼预演暂停租约失败: {e}"))?,
            )
        } else {
            None
        };
        let mut input_pause: Option<PauseLease> = None;
        let mut release_guard = MouseReleaseGuard::new(release_input);
        let total_deadline = started + config.total_timeout();
        let mut next_capture_at = started;
        let mut hook_seen_at: Option<Instant> = None;
        let mut last_stage_log: Option<(&'static str, Instant)> = None;

        self.log(
            LogLevel::Info,
            if dry_run {
                "自动钓鱼预演已启动：仅记录 MouseDown/MouseUp，不发送输入"
            } else {
                "自动钓鱼已启动：等待鱼儿咬钩"
            },
        );

        let result = (|| {
            loop {
                self.ensure_not_cancelled()?;
                if Instant::now() >= total_deadline {
                    return Ok(failed_report(report, started, timeout_message(config)));
                }

                let now = Instant::now();
                let mut capture_at = next_capture_at;
                if let Some(wait_ms) = controller.wait_ms(now) {
                    capture_at = capture_at.max(now + Duration::from_millis(wait_ms));
                }
                self.wait_until(capture_at.min(total_deadline));
                self.ensure_not_cancelled()?;
                if Instant::now() >= total_deadline {
                    return Ok(failed_report(report, started, timeout_message(config)));
                }

                // Pace capture starts, not completion time: slow OCR must not cause a busy loop.
                next_capture_at = Instant::now() + CAPTURE_INTERVAL;
                let frame = match capture(self) {
                    Ok(Some(frame)) => frame,
                    Ok(None) => continue,
                    Err(error) => return Ok(failed_report(report, started, error)),
                };
                self.ensure_not_cancelled()?;
                if Instant::now() >= total_deadline {
                    return Ok(failed_report(report, started, timeout_message(config)));
                }

                let (observation, frame_size) = match detect(self, &frame) {
                    Ok(detected) => detected,
                    Err(error) => return Ok(failed_report(report, started, error)),
                };
                self.ensure_not_cancelled()?;
                // OCR can be slow. Never act on an observation that arrived after the deadline.
                if Instant::now() >= total_deadline {
                    return Ok(failed_report(report, started, timeout_message(config)));
                }

                // A held button must not keep triggers paused after focus/size changes, even when
                // the controller's next action is only Wait.
                if !dry_run
                    && controller.is_mouse_held()
                    && let Err(error) = validate_target(self, frame_size)
                {
                    self.log(LogLevel::Warn, format!("自动钓鱼输入已失效：{error}"));
                    return Ok(failed_report(report, started, error));
                }

                let now = Instant::now();
                let action = controller.advance_since(observation, frame_size, now, started);
                if matches!(action, FishingAction::MouseDown) && hook_seen_at.is_none() {
                    hook_seen_at = Some(now);
                }
                let stage = fishing_stage(&controller, action, observation, hook_seen_at, now);
                log_stage(self, &mut last_stage_log, stage);

                match action {
                    FishingAction::Wait => {}
                    FishingAction::MouseDown | FishingAction::MouseUp => {
                        if dry_run {
                            self.log(
                                LogLevel::Info,
                                match action {
                                    FishingAction::MouseDown => {
                                        "自动钓鱼预演：MouseDown（未发送输入）"
                                    }
                                    _ => "自动钓鱼预演：MouseUp（未发送输入）",
                                },
                            );
                        } else {
                            let input_result = (|| {
                                self.ensure_not_cancelled()?;
                                validate_target(self, frame_size)?;
                                match action {
                                    FishingAction::MouseDown => {
                                        if input_pause.is_none() {
                                            input_pause = Some(acquire_pause(self)?);
                                        }
                                        // The controller marks the button held before returning the
                                        // action. Arm conservatively before checked input can fail.
                                        release_guard.arm();
                                        send_action(self, action, frame_size)
                                    }
                                    FishingAction::MouseUp => {
                                        send_action(self, action, frame_size)?;
                                        release_guard.disarm();
                                        input_pause = None;
                                        Ok(())
                                    }
                                    _ => unreachable!("only mouse actions reach this branch"),
                                }
                            })();
                            if let Err(error) = input_result {
                                self.log(LogLevel::Warn, format!("自动钓鱼输入失败：{error}"));
                                return Ok(failed_report(report, started, error));
                            }
                        }
                    }
                    FishingAction::Succeeded => {
                        if !controller.succeeded() {
                            return Ok(failed_report(
                                report,
                                started,
                                "自动钓鱼状态机异常结束".into(),
                            ));
                        }
                        report.outcome = TaskOutcome::Success;
                        report.runs_completed = 1;
                        self.log(
                            LogLevel::Info,
                            if dry_run {
                                "自动钓鱼预演完成：状态机判定成功"
                            } else {
                                "自动钓鱼完成：状态机判定成功"
                            },
                        );
                        return Ok(report);
                    }
                    FishingAction::Failed(reason) => {
                        self.log(LogLevel::Warn, format!("自动钓鱼识别失败：{reason}"));
                        return Ok(failed_report(report, started, reason.into()));
                    }
                }

                // A successful release can be the controller's terminal action.
                if controller.succeeded() {
                    report.outcome = TaskOutcome::Success;
                    report.runs_completed = 1;
                    self.log(
                        LogLevel::Info,
                        if dry_run {
                            "自动钓鱼预演完成：状态机判定成功"
                        } else {
                            "自动钓鱼完成：状态机判定成功"
                        },
                    );
                    return Ok(report);
                }
            }
        })();

        let release_result = release_guard.release();
        // Keep triggers paused until LEFTUP (or its release-all fallback) has finished.
        drop(input_pause);
        drop(dry_run_pause);
        match (result, release_result) {
            (Ok(mut report), Ok(())) => {
                report.elapsed_ms = started.elapsed().as_millis() as u64;
                Ok(report)
            }
            (Ok(mut report), Err(error)) => {
                self.log(LogLevel::Error, format!("自动钓鱼输入清理失败：{error}"));
                if report.outcome == TaskOutcome::Success {
                    report.outcome = TaskOutcome::Failed;
                    report.runs_completed = 0;
                    report.failure = Some(error);
                } else if let Some(failure) = report.failure.as_mut() {
                    failure.push('；');
                    failure.push_str(&error);
                }
                report.elapsed_ms = started.elapsed().as_millis() as u64;
                Ok(report)
            }
            (Err(error), _) => Err(error),
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn ensure_not_cancelled(&self) -> Result<(), String> {
        if self.is_cancelled() {
            Err("自动钓鱼已取消".into())
        } else {
            Ok(())
        }
    }

    fn wait_until(&self, deadline: Instant) {
        while !self.is_cancelled() {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(25)));
        }
    }

    fn capture_raw_frame(&self) -> Result<Option<CapturedFrame>, String> {
        let captured = (|| {
            let mut guard = self
                .capturer
                .lock()
                .map_err(|_| "抓图会话锁已中毒".to_string())?;
            let need_reopen = match &*guard {
                Some((hwnd, mode, _)) => *hwnd != self.hwnd || *mode != self.mode,
                None => true,
            };
            if need_reopen {
                *guard = None;
                let backend = CapturerBackend::open(self.hwnd, self.mode, 16)
                    .map_err(|e| format!("打开抓图会话失败: {e}"))?;
                *guard = Some((self.hwnd, self.mode, backend));
            }
            let (_, _, backend) = guard.as_mut().unwrap();
            match backend.capture(&self.pool) {
                Ok(frame) => Ok(Some(frame)),
                Err(bgi_capture::CaptureError::FrameNotReady) => Ok(None),
                Err(error) => {
                    *guard = None;
                    Err(error.to_string())
                }
            }
        })();
        match captured {
            Ok(frame) => Ok(frame),
            Err(error) => {
                self.log(LogLevel::Warn, format!("自动钓鱼抓图失败：{error}"));
                Err(error)
            }
        }
    }

    fn detect_fishing(
        &self,
        frame: &CapturedFrame,
    ) -> Result<(FishingObservation, (u32, u32)), String> {
        let detected = {
            let mut guard = self
                .cv_ocr
                .lock()
                .map_err(|_| "CV/OCR 会话锁已中毒".to_string())?;
            guard.detect_fishing_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        };
        match detected {
            Ok(observation) => Ok((observation, (frame.width, frame.height))),
            Err(error) => {
                self.log(LogLevel::Warn, format!("自动钓鱼识别失败：{error}"));
                Err(error)
            }
        }
    }

    fn validate_mouse_target(&self, frame_size: (u32, u32)) -> Result<(), String> {
        validate_mouse_target(
            self.hwnd,
            frame_size,
            bgi_input::is_foreground_window,
            bgi_input::window_client_size,
        )
    }

    fn log(&self, level: LogLevel, message: impl Into<String>) {
        let _ = self.events_tx.send(PathingEvent::Log {
            task_id: self.task_id,
            level,
            message: message.into(),
        });
    }
}

fn validate_mouse_target(
    hwnd: isize,
    frame_size: (u32, u32),
    is_foreground: impl FnOnce(isize) -> bool,
    client_size: impl FnOnce(isize) -> Option<(u32, u32)>,
) -> Result<(), String> {
    if hwnd == 0 {
        return Err("游戏窗口句柄无效，未发送钓鱼输入".into());
    }
    if !is_foreground(hwnd) {
        return Err("游戏窗口不在前台，未发送钓鱼输入".into());
    }
    if client_size(hwnd) != Some(frame_size) {
        return Err("游戏客户区尺寸与识别帧不一致，未发送钓鱼输入".into());
    }
    Ok(())
}

fn send_mouse_action(
    host: &FishingHost,
    action: FishingAction,
    frame_size: (u32, u32),
) -> Result<(), String> {
    match action {
        FishingAction::MouseDown => {
            bgi_input::mouse_left_down_checked(host.hwnd, frame_size).map_err(str::to_string)
        }
        FishingAction::MouseUp => bgi_input::mouse_left_up_checked().map_err(str::to_string),
        _ => Err("状态机请求了无效的钓鱼鼠标动作".into()),
    }
}

fn release_mouse_input() -> Result<(), String> {
    if let Err(error) = bgi_input::mouse_left_up_checked() {
        bgi_input::release_all_keys();
        Err(format!("{error}；已执行 release_all_keys 兜底"))
    } else {
        Ok(())
    }
}

fn new_report(task_id: u64, dry_run: bool) -> TaskReport {
    TaskReport {
        task_id,
        task_name: if dry_run {
            format!("{FISHING_TASK_NAME}（预演）")
        } else {
            FISHING_TASK_NAME.into()
        },
        outcome: TaskOutcome::Failed,
        runs_completed: 0,
        runs_requested: 1,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: if dry_run { "dry_run" } else { "fishing" }.into(),
        elapsed_ms: 0,
    }
}

fn failed_report(mut report: TaskReport, started: Instant, reason: String) -> TaskReport {
    report.outcome = TaskOutcome::Failed;
    report.runs_completed = 0;
    report.failure = Some(reason);
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

fn timeout_message(config: FishingConfig) -> String {
    format!(
        "自动钓鱼超过 {} 秒，已停止；不会继续发送输入",
        config.total_timeout_ms / 1_000
    )
}

fn fishing_stage(
    controller: &FishingController,
    action: FishingAction,
    observation: FishingObservation,
    hook_seen_at: Option<Instant>,
    now: Instant,
) -> &'static str {
    if controller.succeeded() {
        "completed"
    } else if matches!(action, FishingAction::Failed(_)) {
        "recognition_failed"
    } else if let Some(hooked_at) = hook_seen_at {
        if now.saturating_duration_since(hooked_at) < HOOK_DELAY {
            "hooking"
        } else if observation.bar.is_some() {
            "reeling"
        } else {
            "finding_bar"
        }
    } else {
        "waiting_bite"
    }
}

fn log_stage(host: &FishingHost, last: &mut Option<(&'static str, Instant)>, stage: &'static str) {
    let now = Instant::now();
    let should_log = last.as_ref().is_none_or(|(previous, logged_at)| {
        *previous != stage || now.saturating_duration_since(*logged_at) >= REPEAT_LOG_INTERVAL
    });
    if !should_log {
        return;
    }
    let message = match stage {
        "waiting_bite" => "自动钓鱼阶段：等待鱼儿咬钩",
        "hooking" => "自动钓鱼阶段：已咬钩，MouseDown 后等待鱼条",
        "finding_bar" => "自动钓鱼阶段：等待有效鱼条出现",
        "reeling" => "自动钓鱼阶段：根据鱼条位置控制按住/松开",
        "completed" => "自动钓鱼阶段：鱼条消失，状态机判定完成",
        "recognition_failed" => "自动钓鱼阶段：识别失败，准备停止",
        _ => "自动钓鱼阶段：未知",
    };
    host.log(
        if stage == "recognition_failed" {
            LogLevel::Warn
        } else {
            LogLevel::Info
        },
        message,
    );
    *last = Some((stage, now));
}

/// Releases only after a checked down was attempted, including during unwinding.
struct MouseReleaseGuard<F: FnMut() -> Result<(), String>> {
    active: bool,
    release_input: F,
}

impl<F: FnMut() -> Result<(), String>> MouseReleaseGuard<F> {
    fn new(release_input: F) -> Self {
        Self {
            active: false,
            release_input,
        }
    }

    fn arm(&mut self) {
        self.active = true;
    }

    fn disarm(&mut self) {
        self.active = false;
    }

    fn release(&mut self) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        (self.release_input)()
    }
}

impl<F: FnMut() -> Result<(), String>> Drop for MouseReleaseGuard<F> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgi_capture::PixelFormat;
    use bgi_control::{UiSnapshot, create_control_bus};
    use bgi_task::{FishingBar, FishingRect};
    use std::collections::VecDeque;

    const FRAME: (u32, u32) = (1_920, 1_080);

    fn observation(in_fishing: bool, bite: bool, bar: Option<FishingBar>) -> FishingObservation {
        FishingObservation {
            in_fishing,
            bite,
            bar,
        }
    }

    fn waiting() -> FishingObservation {
        observation(true, false, None)
    }

    fn bite() -> FishingObservation {
        observation(true, true, None)
    }

    fn reeling_bar() -> FishingObservation {
        let rect = |x| FishingRect {
            x,
            y: 100,
            width: if x == 100 { 10 } else { 100 },
            height: 20,
        };
        observation(
            true,
            false,
            Some(FishingBar {
                cursor: rect(100),
                target: rect(500),
            }),
        )
    }

    fn release_bar() -> FishingObservation {
        let rect = |x, width| FishingRect {
            x,
            y: 100,
            width,
            height: 20,
        };
        observation(
            true,
            false,
            Some(FishingBar {
                cursor: rect(600, 10),
                target: rect(100, 100),
            }),
        )
    }

    fn successful_sequence() -> VecDeque<FishingObservation> {
        let mut sequence: VecDeque<FishingObservation> =
            [waiting(), bite(), reeling_bar(), release_bar()].into();
        // The controller waits for the bar to disappear before declaring success.
        sequence.extend((0..20).map(|_| waiting()));
        sequence
    }

    #[derive(Default)]
    struct TestInputState {
        hwnd: isize,
        foreground: bool,
        client_size: Option<(u32, u32)>,
        pause_error: Option<&'static str>,
        down_error: Option<&'static str>,
        fail_up: bool,
        cancel_after_down: bool,
        lose_focus_after_down: bool,
        panic_after_down: bool,
        pause_acquires: usize,
        cleanup_count: usize,
        paused_during_cleanup: Vec<bool>,
        actions: Vec<FishingAction>,
        paused_during_detection: Vec<bool>,
        paused_during_send: Vec<bool>,
    }

    struct TestAttempt {
        host: FishingHost,
        state: Arc<Mutex<TestInputState>>,
        observations: VecDeque<FishingObservation>,
        fallback: FishingObservation,
    }

    impl TestAttempt {
        fn new(observations: impl IntoIterator<Item = FishingObservation>) -> Self {
            let (_, _, core) = create_control_bus(UiSnapshot::default());
            let (events_tx, _events_rx) = mpsc::unbounded_channel();
            let pool = Arc::new(FrameBufferPool::new(1));
            Self {
                host: FishingHost {
                    task_id: 7,
                    hwnd: 1,
                    mode: CaptureMode::default(),
                    cancel: Arc::new(AtomicBool::new(false)),
                    capturer: Arc::new(Mutex::new(None)),
                    pool: pool.clone(),
                    cv_ocr: Arc::new(Mutex::new(CvOcrServices::new(
                        std::path::PathBuf::from("."),
                        core.log_publisher(),
                    ))),
                    events_tx,
                    pauses: PauseController::default(),
                },
                state: Arc::new(Mutex::new(TestInputState {
                    hwnd: 1,
                    foreground: true,
                    client_size: Some(FRAME),
                    ..TestInputState::default()
                })),
                observations: observations.into_iter().collect(),
                fallback: observation(false, false, None),
            }
        }

        fn run(&mut self, dry_run: bool) -> Result<TaskReport, String> {
            let state = self.state.clone();
            let pool = self.host.pool.clone();
            let pauses = self.host.pauses.clone();
            let observations = &mut self.observations;
            let fallback = self.fallback;
            self.host.run_fishing_attempt_with(
                dry_run,
                FishingConfig::default(),
                |_host| {
                    Ok(Some(CapturedFrame {
                        width: FRAME.0,
                        height: FRAME.1,
                        format: PixelFormat::Bgr24,
                        data: pool.checkout(1),
                    }))
                },
                |host, _frame| {
                    state
                        .lock()
                        .unwrap()
                        .paused_during_detection
                        .push(host.pauses.is_paused());
                    Ok((observations.pop_front().unwrap_or(fallback), FRAME))
                },
                |_host, frame_size| {
                    let state = state.lock().unwrap();
                    validate_mouse_target(
                        state.hwnd,
                        frame_size,
                        |_| state.foreground,
                        |_| state.client_size,
                    )
                },
                |host| {
                    let mut state = state.lock().unwrap();
                    state.pause_acquires += 1;
                    if let Some(error) = state.pause_error {
                        Err(error.into())
                    } else {
                        host.pauses
                            .pause()
                            .map_err(|e| format!("获取钓鱼输入暂停租约失败: {e}"))
                    }
                },
                |host, action, _frame_size| {
                    let (down_error, fail_up, cancel, lose_focus, panic_after_down) = {
                        let mut state = state.lock().unwrap();
                        state.actions.push(action);
                        state.paused_during_send.push(host.pauses.is_paused());
                        (
                            state.down_error,
                            state.fail_up,
                            state.cancel_after_down,
                            state.lose_focus_after_down,
                            state.panic_after_down,
                        )
                    };
                    match action {
                        FishingAction::MouseDown => {
                            if let Some(error) = down_error {
                                return Err(error.into());
                            }
                            if cancel {
                                host.cancel.store(true, Ordering::Release);
                            }
                            if lose_focus {
                                state.lock().unwrap().foreground = false;
                            }
                            if panic_after_down {
                                panic!("injected panic after checked down");
                            }
                            Ok(())
                        }
                        FishingAction::MouseUp if fail_up => Err("模拟 MouseUp 失败".into()),
                        FishingAction::MouseUp => Ok(()),
                        _ => unreachable!("test only injects mouse actions"),
                    }
                },
                || {
                    let mut state = state.lock().unwrap();
                    state.cleanup_count += 1;
                    state.actions.push(FishingAction::MouseUp);
                    state.paused_during_cleanup.push(pauses.is_paused());
                    Ok(())
                },
            )
        }

        fn assert_input_idle(&self) {
            assert!(!self.host.pauses.is_paused());
            assert!(
                self.state
                    .lock()
                    .unwrap()
                    .paused_during_send
                    .iter()
                    .all(|paused| *paused)
            );
            assert!(
                self.state
                    .lock()
                    .unwrap()
                    .paused_during_cleanup
                    .iter()
                    .all(|paused| *paused)
            );
        }
    }

    #[test]
    fn reports_are_one_attempt_and_use_mode_specific_stages() {
        for (dry_run, expected_stage) in [(false, "fishing"), (true, "dry_run")] {
            let report = new_report(42, dry_run);
            assert_eq!(report.task_id, 42);
            assert_eq!(report.stage, expected_stage);
            assert_eq!(report.runs_requested, 1);
            assert_eq!(report.runs_completed, 0);
            assert_eq!(report.outcome, TaskOutcome::Failed);
        }
    }

    #[test]
    fn cancelled_dry_run_never_opens_a_capture_session() {
        let cancel = Arc::new(AtomicBool::new(true));
        let capturer = Arc::new(Mutex::new(None));
        let (_, _, core) = create_control_bus(UiSnapshot::default());
        let (events_tx, _events_rx) = mpsc::unbounded_channel();
        let mut host = FishingHost {
            task_id: 7,
            hwnd: 0,
            mode: CaptureMode::default(),
            cancel,
            capturer: capturer.clone(),
            pool: Arc::new(FrameBufferPool::new(1)),
            cv_ocr: Arc::new(Mutex::new(CvOcrServices::new(
                std::path::PathBuf::from("."),
                core.log_publisher(),
            ))),
            events_tx,
            pauses: PauseController::default(),
        };

        assert_eq!(
            host.run_fishing_attempt(true, FishingConfig::default()),
            Err("自动钓鱼已取消".into())
        );
        assert!(capturer.lock().unwrap().is_none());
    }

    #[test]
    fn complete_dry_run_success_sends_zero_input() {
        let mut attempt = TestAttempt::new(successful_sequence());
        attempt.state.lock().unwrap().pause_error = Some("模拟暂停租约耗尽");
        let report = attempt.run(true).unwrap();

        assert_eq!(report.outcome, TaskOutcome::Success, "{report:?}");
        assert_eq!(report.runs_completed, 1);
        let state = attempt.state.lock().unwrap();
        assert!(state.actions.is_empty());
        assert_eq!(state.cleanup_count, 0);
        assert_eq!(state.pause_acquires, 0);
        assert!(state.paused_during_detection.iter().all(|paused| *paused));
        assert!(!attempt.host.pauses.is_paused());
    }

    #[test]
    fn cancelled_real_run_before_input_sends_zero_input() {
        let mut attempt = TestAttempt::new(successful_sequence());
        attempt.host.cancel.store(true, Ordering::Release);

        assert_eq!(attempt.run(false), Err("自动钓鱼已取消".into()));
        let state = attempt.state.lock().unwrap();
        assert!(state.actions.is_empty());
        assert_eq!(state.cleanup_count, 0);
        assert_eq!(state.pause_acquires, 0);
        assert!(state.paused_during_detection.is_empty());
        assert!(!attempt.host.pauses.is_paused());
    }

    #[test]
    fn invalid_input_targets_block_down_without_spurious_up() {
        let cases = [
            (0, true, Some(FRAME), "游戏窗口句柄无效"),
            (1, false, Some(FRAME), "游戏窗口不在前台"),
            (1, true, Some((1_280, 720)), "游戏客户区尺寸与识别帧不一致"),
        ];
        for (hwnd, foreground, client_size, expected) in cases {
            let mut attempt = TestAttempt::new([waiting(), bite()]);
            {
                let mut state = attempt.state.lock().unwrap();
                state.hwnd = hwnd;
                state.foreground = foreground;
                state.client_size = client_size;
            }

            let report = attempt.run(false).unwrap();
            assert_eq!(report.outcome, TaskOutcome::Failed);
            assert!(report.failure.as_deref().unwrap().contains(expected));
            let state = attempt.state.lock().unwrap();
            assert!(state.actions.is_empty());
            assert_eq!(state.cleanup_count, 0);
            assert_eq!(state.pause_acquires, 0);
            assert!(!attempt.host.pauses.is_paused());
        }
    }

    #[test]
    fn pause_lease_exhaustion_blocks_down_without_spurious_up() {
        let mut attempt = TestAttempt::new([waiting(), bite()]);
        attempt.state.lock().unwrap().pause_error = Some("模拟暂停租约耗尽");

        let report = attempt.run(false).unwrap();
        assert_eq!(report.outcome, TaskOutcome::Failed);
        assert!(
            report
                .failure
                .as_deref()
                .unwrap()
                .contains("模拟暂停租约耗尽")
        );
        let state = attempt.state.lock().unwrap();
        assert_eq!(state.pause_acquires, 1);
        assert!(state.actions.is_empty());
        assert_eq!(state.cleanup_count, 0);
        assert!(!attempt.host.pauses.is_paused());
    }

    #[test]
    fn failed_checked_down_leads_to_cleanup_and_drops_pause() {
        let mut attempt = TestAttempt::new([waiting(), bite()]);
        attempt.state.lock().unwrap().down_error = Some("模拟 checked down 失败");

        let report = attempt.run(false).unwrap();
        assert_eq!(report.outcome, TaskOutcome::Failed);
        assert!(
            report
                .failure
                .as_deref()
                .unwrap()
                .contains("模拟 checked down 失败")
        );
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [FishingAction::MouseDown, FishingAction::MouseUp]
        );
        assert_eq!(state.cleanup_count, 1);
        assert_eq!(state.pause_acquires, 1);
        drop(state);
        attempt.assert_input_idle();
    }

    #[test]
    fn cancel_after_successful_down_releases_and_drops_pause() {
        let mut attempt = TestAttempt::new([waiting(), bite()]);
        attempt.state.lock().unwrap().cancel_after_down = true;

        assert_eq!(attempt.run(false), Err("自动钓鱼已取消".into()));
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [FishingAction::MouseDown, FishingAction::MouseUp]
        );
        assert_eq!(state.cleanup_count, 1);
        assert_eq!(state.paused_during_cleanup, [true]);
        drop(state);
        attempt.assert_input_idle();
    }

    #[test]
    fn recognition_failure_while_held_releases_and_drops_pause() {
        let mut attempt = TestAttempt::new([waiting(), bite(), observation(false, false, None)]);

        let report = attempt.run(false).unwrap();
        assert_eq!(report.outcome, TaskOutcome::Failed);
        assert_eq!(report.failure.as_deref(), Some("当前不在钓鱼状态"));
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [FishingAction::MouseDown, FishingAction::MouseUp]
        );
        assert_eq!(state.cleanup_count, 1);
        assert!(state.paused_during_detection[2]);
        drop(state);
        attempt.assert_input_idle();
    }

    #[test]
    fn focus_loss_while_held_releases_without_foreground_check() {
        // The third observation would otherwise be only Wait during the hook delay.
        let mut attempt = TestAttempt::new([waiting(), bite(), waiting()]);
        attempt.state.lock().unwrap().lose_focus_after_down = true;

        let report = attempt.run(false).unwrap();
        assert_eq!(report.outcome, TaskOutcome::Failed);
        assert!(report.failure.as_deref().unwrap().contains("不在前台"));
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [FishingAction::MouseDown, FishingAction::MouseUp]
        );
        assert_eq!(state.cleanup_count, 1);
        assert!(state.paused_during_detection[2]);
        drop(state);
        attempt.assert_input_idle();
    }

    #[test]
    fn failed_mouse_up_is_cleaned_up_and_drops_pause() {
        let mut attempt = TestAttempt::new(successful_sequence());
        attempt.state.lock().unwrap().fail_up = true;

        let report = attempt.run(false).unwrap();
        assert_eq!(report.outcome, TaskOutcome::Failed);
        assert_eq!(report.failure.as_deref(), Some("模拟 MouseUp 失败"));
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [
                FishingAction::MouseDown,
                FishingAction::MouseUp,
                FishingAction::MouseUp,
            ]
        );
        assert_eq!(state.cleanup_count, 1);
        drop(state);
        attempt.assert_input_idle();
    }

    #[test]
    fn normal_mouse_up_keeps_pause_until_up_then_releases_it() {
        let mut sequence: VecDeque<FishingObservation> = [
            waiting(),
            bite(),
            reeling_bar(),
            release_bar(),
            reeling_bar(),
            release_bar(),
        ]
        .into();
        sequence.extend((0..20).map(|_| waiting()));
        let mut attempt = TestAttempt::new(sequence);

        let report = attempt.run(false).unwrap();
        assert_eq!(report.outcome, TaskOutcome::Success, "{report:?}");
        assert_eq!(report.runs_completed, 1);
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [
                FishingAction::MouseDown,
                FishingAction::MouseUp,
                FishingAction::MouseDown,
                FishingAction::MouseUp,
            ]
        );
        assert_eq!(state.cleanup_count, 0);
        assert_eq!(state.pause_acquires, 2);
        assert_eq!(state.paused_during_send, [true, true, true, true]);
        assert!(state.paused_during_detection[2]);
        assert!(!state.paused_during_detection[4]);
        assert!(state.paused_during_detection[5]);
        assert!(!state.paused_during_detection[6]);
        drop(state);
        attempt.assert_input_idle();
    }

    #[test]
    fn panic_after_checked_down_unwinds_through_cleanup() {
        let mut attempt = TestAttempt::new([waiting(), bite()]);
        attempt.state.lock().unwrap().panic_after_down = true;

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = attempt.run(false);
        }));
        assert!(panic.is_err());
        let state = attempt.state.lock().unwrap();
        assert_eq!(
            state.actions,
            [FishingAction::MouseDown, FishingAction::MouseUp]
        );
        assert_eq!(state.cleanup_count, 1);
        drop(state);
        attempt.assert_input_idle();
    }
}
