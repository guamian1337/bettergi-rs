use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bgi_capture::{CapturedFrame, CapturerBackend, FrameBufferPool};
use bgi_control::{CaptureMode, LogLevel};
use bgi_pathing::{
    BigMapObservation, BigMapProjection, MinimapObservation, OcrHit, PathingHost,
};
use tokio::sync::mpsc;

use crate::core::CvOcrServices;

pub(crate) enum PathingEvent {
    Progress {
        current_index: usize,
        total_points: usize,
    },
    Log {
        level: LogLevel,
        message: String,
    },
    Finished(Result<String, String>),
}

pub(crate) struct CorePathingHost {
    pub hwnd: isize,
    pub mode: CaptureMode,
    pub cancel: Arc<AtomicBool>,
    pub capturer: Arc<Mutex<Option<(isize, CaptureMode, CapturerBackend)>>>,
    pub pool: Arc<FrameBufferPool>,
    pub cv_ocr: Arc<Mutex<CvOcrServices>>,
    pub events_tx: mpsc::UnboundedSender<PathingEvent>,
}

impl CorePathingHost {
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
}

impl PathingHost for CorePathingHost {
    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
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
            .map(|opt| opt.map(|(proj, inliers)| BigMapObservation { projection: proj, inliers }))
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

    fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
        self.cv_ocr.lock().unwrap().set_last_minimap_pos(Some((x, y)));
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
        bgi_input::key_down(vk);
    }

    fn key_up(&mut self, vk: u16) {
        bgi_input::key_up(vk);
    }

    fn key_tap(&mut self, vk: u16, hold_ms: u64) {
        bgi_input::key_tap(vk, hold_ms);
    }

    fn mouse_move_rel(&mut self, dx: i32, dy: i32) {
        bgi_input::mouse_move_rel(dx, dy);
    }

    fn mouse_left_click(&mut self) {
        bgi_input::mouse_left_click();
    }

    fn mouse_left_click_client(&mut self, x: i32, y: i32) {
        bgi_input::mouse_left_click_client(self.hwnd, x, y);
    }

    fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        bgi_input::mouse_drag_client(self.hwnd, x1, y1, x2, y2);
    }

    fn mouse_wheel(&mut self, notches: i32) {
        bgi_input::mouse_wheel(notches);
    }

    fn release_all_keys(&mut self) {
        bgi_input::release_all_keys();
    }

    fn on_progress(&mut self, current_index: usize, total_points: usize) {
        let _ = self.events_tx.send(PathingEvent::Progress {
            current_index,
            total_points,
        });
    }

    fn on_log(&mut self, level: LogLevel, message: String) {
        let _ = self.events_tx.send(PathingEvent::Log { level, message });
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
