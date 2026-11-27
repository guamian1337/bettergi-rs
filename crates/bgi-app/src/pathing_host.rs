use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bgi_capture::{CapturedFrame, CapturerBackend, FrameBufferPool};
use bgi_control::{CaptureMode, LogLevel, TaskReport};
use bgi_pathing::{
    AutoPickFrame, BigMapLayerButtons, BigMapObservation, BigMapProjection, CookingObservation,
    MinimapObservation, MotionStatus, OcrHit, PathingHost,
};
use bgi_task::{AUTO_PICK_MAX_OBSERVATION_AGE, PauseController, PauseLease};
use tokio::sync::mpsc;

use crate::capture_host::ActiveCapturer;
use crate::cv_service::{ArtifactSalvageStep, CvOcrServices};
use crate::mining_host::{
    MiningActionLimits, MiningIoAdapter, MiningRunConfig, aim_params_for_window, run_mining_loop,
};

pub(crate) enum PathingEvent {
    Progress {
        task_id: u64,
        current_index: usize,
        total_points: usize,
    },
    Log {
        task_id: u64,
        level: LogLevel,
        message: String,
    },
    Finished {
        task_id: u64,
        result: Result<TaskReport, String>,
    },
    SalvagePreviewFinished {
        task_id: u64,
        result: Result<TaskReport, String>,
        preview: Option<bgi_control::SalvagePreviewReport>,
    },
}

pub(crate) struct CorePathingHost {
    pub task_id: u64,
    pub hwnd: isize,
    pub mode: CaptureMode,
    pub cancel: Arc<AtomicBool>,
    pub capturer: Arc<Mutex<ActiveCapturer>>,
    pub pool: Arc<FrameBufferPool>,
    pub cv_ocr: Arc<Mutex<CvOcrServices>>,
    pub events_tx: mpsc::UnboundedSender<PathingEvent>,
    pub current_index: Option<usize>,
    pub pauses: PauseController,
    pub(crate) input_pause: Option<PauseLease>,
    pub(crate) held_keys: BTreeSet<u16>,
    pub(crate) right_mouse_held: bool,
    pub(crate) mining_limits: MiningActionLimits,
}

impl CorePathingHost {
    fn acquire_input_pause(&mut self) {
        if self.input_pause.is_none() {
            self.input_pause = Some(self.pauses.pause().expect("pause owner overflow"));
        }
    }

    fn release_input_pause_if_idle(&mut self) {
        if self.held_keys.is_empty() && !self.right_mouse_held {
            self.input_pause = None;
        }
    }

    fn capture_raw_frame(&self) -> Result<Option<CapturedFrame>, String> {
        let mut guard = self.capturer.lock().unwrap();
        let need_reopen = match &*guard {
            Some((h, m, _)) => *h != self.hwnd || *m != self.mode,
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
            Ok(res) => Ok(Some(res)),
            Err(bgi_capture::CaptureError::FrameNotReady) => Ok(None),
            Err(e) => {
                *guard = None;
                Err(e.to_string())
            }
        }
    }

    pub(crate) fn click_client_guarded(&mut self, x: i32, y: i32) -> bool {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::mouse_left_click_client_checked(self.hwnd, x, y)
    }

    pub(crate) fn detect_artifact_salvage(&self) -> Result<Option<ArtifactSalvageStep>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard
            .detect_artifact_salvage_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
            .map(Some)
    }

    /// The salvage preview bridge can only emit wheel input. It has no click/key action.
    pub(crate) fn scroll_artifact_salvage_preview(
        &mut self,
        center: (i32, i32),
        notches: i32,
        frame_size: (u32, u32),
    ) -> Result<(), String> {
        if self.is_cancelled() {
            return Err("分解预览已取消".into());
        }
        if !bgi_input::is_foreground_window(self.hwnd) {
            return Err("游戏窗口不在前台，已停止分解预览".into());
        }
        if !(-3..=-1).contains(&notches)
            || center.0 < 0
            || center.1 < 0
            || center.0 >= frame_size.0 as i32
            || center.1 >= frame_size.1 as i32
        {
            return Err("分解预览滚动参数越界，已停止".into());
        }
        let _pause = self.pauses.pause().expect("pause owner overflow");
        let direction = if notches > 0 { 1 } else { -1 };
        for _ in 0..notches.abs() {
            if self.is_cancelled() {
                return Err("分解预览已取消".into());
            }
            bgi_input::mouse_wheel_notch_checked(
                self.hwnd, center.0, center.1, frame_size, direction,
            )
            .map_err(str::to_string)?;
        }
        Ok(())
    }

    /// Grabs the current frame and dumps it under `log/failed` so a failed run can be
    /// inspected afterwards. Retries briefly because the failing tick may arrive while a
    /// frame is still in flight. `Ok(None)` means every attempt found no frame.
    pub(crate) fn save_failure_frame(
        &self,
        tag: &str,
    ) -> Result<Option<std::path::PathBuf>, String> {
        for attempt in 0..3 {
            let Some(frame) = self.capture_raw_frame()? else {
                if attempt < 2 {
                    std::thread::sleep(Duration::from_millis(40));
                }
                continue;
            };
            let mut guard = self.cv_ocr.lock().unwrap();
            return guard.save_failure_frame(
                tag,
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            );
        }
        Ok(None)
    }

    pub(crate) fn detect_auto_wood(
        &self,
    ) -> Result<Option<(bgi_cv::auto_wood_templates::AutoWoodObservation, bool)>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard
            .detect_auto_wood_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
            .map(Some)
    }

    pub(crate) fn detect_expedition(
        &self,
    ) -> Result<Option<bgi_cv::expedition::ExpeditionObservation>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard
            .detect_expedition_observation(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
            .map(Some)
    }

    pub(crate) fn detect_chest(
        &self,
    ) -> Result<Option<(bgi_cv::chest_templates::ChestObservation, bool)>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard
            .detect_chest_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
            .map(Some)
    }

    pub(crate) fn detect_rewards(
        &self,
    ) -> Result<Option<(Vec<bgi_task::RewardCandidate>, bool)>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard
            .detect_rewards_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
            .map(Some)
    }
}

impl PathingHost for CorePathingHost {
    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn current_waypoint_index(&self) -> Option<usize> {
        self.current_index
    }

    fn detect_loading(&mut self) -> Result<bool, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(true);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        Ok(guard
            .locate_minimap_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
                None,
            )?
            .is_none())
    }

    fn dpi_scale(&self) -> f32 {
        bgi_input::window_dpi_scale(self.hwnd)
    }

    fn focus_game_window(&mut self) {
        bgi_input::focus_window(self.hwnd);
        std::thread::sleep(Duration::from_millis(120));
    }

    fn capture_minimap(
        &mut self,
        prev_pos: Option<(f32, f32)>,
    ) -> Result<Option<MinimapObservation>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard.locate_minimap_step(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
            prev_pos,
        )
    }

    fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard
            .locate_bigmap_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
            .map(|opt| {
                opt.map(|(proj, inliers)| BigMapObservation {
                    projection: proj,
                    inliers,
                })
            })
    }

    fn capture_ocr_roi(
        &mut self,
        norm_x0: f32,
        norm_y0: f32,
        norm_x1: f32,
        norm_y1: f32,
    ) -> Result<Vec<OcrHit>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(Vec::new());
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard.ocr_roi_step(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
            norm_x0,
            norm_y0,
            norm_x1,
            norm_y1,
        )
    }

    fn capture_auto_pick(&mut self) -> Result<Option<AutoPickFrame>, String> {
        let observed_at = std::time::Instant::now();
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        let (observation, ocr_hits) = guard.detect_auto_pick_step(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
        )?;
        Ok(Some(AutoPickFrame {
            width: observation.width,
            height: observation.height,
            prompt: observation.prompt,
            ocr_hits,
            observed_at,
        }))
    }

    fn key_tap_checked(
        &mut self,
        vk: u16,
        hold_ms: u64,
        frame_size: (u32, u32),
        observed_at: Instant,
    ) -> Result<(), String> {
        if self.is_cancelled() {
            return Err("自动拾取已取消".into());
        }
        let Some(_pause) = self.pauses.try_pause().map_err(|e| e.to_string())? else {
            return Err("自动拾取已暂停".into());
        };
        bgi_input::key_tap_checked_if(self.hwnd, vk, hold_ms, frame_size, || {
            !self.is_cancelled()
                && observed_at.elapsed() <= AUTO_PICK_MAX_OBSERVATION_AGE
                && _pause.is_active()
        })
        .map_err(str::to_string)
    }

    fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
        self.cv_ocr
            .lock()
            .unwrap()
            .set_last_minimap_pos(Some((x, y)));
    }

    fn reset_minimap_prev_pos(&mut self) {
        self.cv_ocr.lock().unwrap().set_last_minimap_pos(None);
    }

    fn sleep_ms(&mut self, ms: u64) {
        let slice = 25u64;
        let mut rem = ms;
        while rem > 0 && !self.is_cancelled() {
            let chunk = rem.min(slice);
            std::thread::sleep(Duration::from_millis(chunk));
            rem -= chunk;
        }
    }

    fn key_down(&mut self, vk: u16) {
        self.acquire_input_pause();
        self.held_keys.insert(vk);
        bgi_input::key_down(vk);
    }

    fn key_up(&mut self, vk: u16) {
        bgi_input::key_up(vk);
        self.held_keys.remove(&vk);
        self.release_input_pause_if_idle();
    }

    fn key_tap(&mut self, vk: u16, hold_ms: u64) {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::key_tap(vk, hold_ms);
    }

    fn mouse_move_rel(&mut self, dx: i32, dy: i32) {
        bgi_input::mouse_move_rel(dx, dy);
    }

    fn mouse_move_client(&mut self, x: i32, y: i32) {
        bgi_input::mouse_move_to_client(self.hwnd, x, y);
    }

    fn mouse_left_click(&mut self) {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::mouse_left_click();
    }

    fn mouse_right_click(&mut self) {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::mouse_right_click();
    }

    fn mouse_right_down(&mut self) {
        self.acquire_input_pause();
        self.right_mouse_held = true;
        bgi_input::mouse_right_down();
    }

    fn mouse_right_up(&mut self) {
        bgi_input::mouse_right_up();
        self.right_mouse_held = false;
        self.release_input_pause_if_idle();
    }

    fn mouse_left_click_client(&mut self, x: i32, y: i32) {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::mouse_left_click_client(self.hwnd, x, y);
    }

    fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::mouse_drag_client(self.hwnd, x1, y1, x2, y2);
    }

    fn mouse_wheel(&mut self, notches: i32) {
        let _pause = self.pauses.pause().expect("pause owner overflow");
        bgi_input::mouse_wheel(notches);
    }

    fn detect_bigmap_layer_buttons(&mut self) -> Result<BigMapLayerButtons, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(BigMapLayerButtons::default());
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard.detect_bigmap_layer_buttons_step(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
        )
    }

    fn detect_motion_status(&mut self) -> Result<MotionStatus, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Err("当前帧尚未就绪，无法判断运动状态".to_string());
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        guard.detect_motion_status_step(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
        )
    }
    fn capture_cooking_bar(&mut self) -> Result<Option<CookingObservation>, String> {
        let Some(frame) = self.capture_raw_frame()? else {
            return Ok(None);
        };
        let mut guard = self.cv_ocr.lock().unwrap();
        let ui = guard.detect_cooking_ui_step(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
        )?;
        let (page_visible, white_confirm, recover_visible) =
            (ui.page_visible, ui.white_confirm, ui.recover_visible);
        let scale = f64::from(frame.height) / 1080.0;
        let left = (600.0 * scale).round().max(0.0) as u32;
        let top = (660.0 * scale).round().max(0.0) as u32;
        let width = (730.0 * scale).round().max(0.0) as u32;
        let height = (190.0 * scale).round().max(0.0) as u32;
        if left >= frame.width || top >= frame.height {
            return Ok(Some(CookingObservation {
                color_count: 0,
                scale,
                page_visible,
                white_confirm,
                recover_visible,
            }));
        }
        let right = left.saturating_add(width).min(frame.width);
        let bottom = top.saturating_add(height).min(frame.height);
        let channels = if frame.format.is_bgr24() { 3 } else { 4 };
        let mut count = 0u32;
        for y in top..bottom {
            let start = (y * frame.width + left) as usize * channels;
            let end = (y * frame.width + right) as usize * channels;
            for pixel in frame.data.as_slice()[start..end].chunks_exact(channels) {
                if pixel[0] == 64 && pixel[1] == 192 && pixel[2] == 255 {
                    count = count.saturating_add(1);
                }
            }
        }
        Ok(Some(CookingObservation {
            color_count: count,
            scale,
            page_visible,
            white_confirm,
            recover_visible,
        }))
    }

    /// 路径动作 `linnea_mining`：复用采矿任务的瞄准闭环。
    ///
    /// 角色切换不在本层：路径里请先自行切到莉奈娅，否则识别不到 ore 会按"未找到目标"返回错误。
    fn run_linnea_mining(&mut self, params: Option<&str>) -> Result<(), String> {
        let (mine_count, scan_rounds) = bgi_task::parse_linnea_mining_params(params);
        let config = MiningRunConfig {
            mine_count,
            scan_rounds,
            max_shots: self.mining_limits.max_shots,
            total_timeout_ms: self.mining_limits.total_timeout_ms,
        };
        let aim = aim_params_for_window(self.hwnd);
        let task_id = self.task_id;
        let events_tx = self.events_tx.clone();
        let outcome = {
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
            let outcome = run_mining_loop(&mut io, config, aim);
            io.release_inputs();
            outcome
        }?;
        self.on_log(
            LogLevel::Info,
            format!(
                "莉奈娅挖矿结束：挖 {} 次、射击 {} 次、{} 轮扫描",
                outcome.mined, outcome.shots, outcome.rounds
            ),
        );
        if outcome.mined == 0 {
            // 与 C# 处理器一致：单个挖矿点没挖到不算整条路线失败，只记警告继续跑图。
            self.on_log(
                LogLevel::Warn,
                "莉奈娅挖矿未挖到任何矿（目标丢失或元素视野内无 ore），继续执行后续点位".into(),
            );
        }
        Ok(())
    }

    fn release_all_keys(&mut self) {
        bgi_input::release_all_keys();
        self.held_keys.clear();
        self.right_mouse_held = false;
        self.input_pause = None;
    }

    fn on_progress(&mut self, current_index: usize, total_points: usize) {
        self.current_index = Some(current_index);
        let _ = self.events_tx.send(PathingEvent::Progress {
            task_id: self.task_id,
            current_index,
            total_points,
        });
    }

    fn on_log(&mut self, level: LogLevel, message: String) {
        let _ = self.events_tx.send(PathingEvent::Log {
            task_id: self.task_id,
            level,
            message,
        });
    }
}

pub(crate) fn to_projection(m: bgi_cv::sift_map::BigMapRectMatch) -> (BigMapProjection, usize) {
    (
        BigMapProjection {
            game_tl_x: m.game_tl_x,
            game_tl_y: m.game_tl_y,
            game_br_x: m.game_br_x,
            game_br_y: m.game_br_y,
            frame_width: m.frame_width,
            frame_height: m.frame_height,
        },
        m.inliers,
    )
}
