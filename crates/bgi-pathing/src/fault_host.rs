use bgi_control::LogLevel;
use bgi_task::{FaultBoundary, FaultInjector};

use crate::executor::{
    BigMapLayerButtons, BigMapObservation, MinimapObservation, MotionStatus, OcrHit,
    PathingCheckpoint, PathingHost,
};

pub struct FaultHost<H> {
    pub inner: H,
    pub injector: FaultInjector,
}

impl<H> FaultHost<H> {
    pub fn new(inner: H) -> Self {
        Self {
            inner,
            injector: FaultInjector::new(),
        }
    }
}

impl<H: PathingHost> PathingHost for FaultHost<H> {
    fn is_cancelled(&self) -> bool {
        self.injector.check(FaultBoundary::Cancellation).is_err() || self.inner.is_cancelled()
    }

    fn dpi_scale(&self) -> f32 {
        self.inner.dpi_scale()
    }
    fn focus_game_window(&mut self) {
        self.inner.focus_game_window();
    }

    fn capture_minimap(
        &mut self,
        prev: Option<(f32, f32)>,
    ) -> Result<Option<MinimapObservation>, String> {
        self.injector.check(FaultBoundary::Capture)?;
        self.injector.check(FaultBoundary::Minimap)?;
        self.inner.capture_minimap(prev)
    }

    fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
        self.injector.check(FaultBoundary::Capture)?;
        self.injector.check(FaultBoundary::BigMap)?;
        self.inner.capture_bigmap()
    }

    fn capture_ocr_roi(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) -> Result<Vec<OcrHit>, String> {
        self.injector.check(FaultBoundary::Capture)?;
        self.injector.check(FaultBoundary::Ocr)?;
        self.inner.capture_ocr_roi(x0, y0, x1, y1)
    }

    fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
        self.inner.set_minimap_prev_pos(x, y);
    }
    fn reset_minimap_prev_pos(&mut self) {
        self.inner.reset_minimap_prev_pos();
    }
    fn sleep_ms(&mut self, ms: u64) {
        self.inner.sleep_ms(ms);
    }

    fn key_down(&mut self, vk: u16) {
        let _ = self.injector.check(FaultBoundary::Input);
        self.inner.key_down(vk);
    }

    fn key_up(&mut self, vk: u16) {
        let _ = self.injector.check(FaultBoundary::Input);
        self.inner.key_up(vk);
    }

    fn key_tap(&mut self, vk: u16, hold_ms: u64) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.key_tap(vk, hold_ms);
        }
    }

    fn mouse_move_rel(&mut self, dx: i32, dy: i32) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_move_rel(dx, dy);
        }
    }

    fn mouse_move_client(&mut self, x: i32, y: i32) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_move_client(x, y);
        }
    }

    fn mouse_left_click(&mut self) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_left_click();
        }
    }

    fn mouse_right_click(&mut self) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_right_click();
        }
    }

    fn mouse_right_down(&mut self) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_right_down();
        }
    }

    fn mouse_right_up(&mut self) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_right_up();
        }
    }

    fn mouse_left_click_client(&mut self, x: i32, y: i32) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_left_click_client(x, y);
        }
    }

    fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_drag_client(x1, y1, x2, y2);
        }
    }

    fn mouse_wheel(&mut self, notches: i32) {
        if self.injector.check(FaultBoundary::Input).is_ok() {
            self.inner.mouse_wheel(notches);
        }
    }

    fn detect_bigmap_layer_buttons(&mut self) -> Result<BigMapLayerButtons, String> {
        self.injector.check(FaultBoundary::Capture)?;
        self.inner.detect_bigmap_layer_buttons()
    }

    fn detect_motion_status(&mut self) -> Result<MotionStatus, String> {
        self.injector.check(FaultBoundary::Capture)?;
        self.injector.check(FaultBoundary::Motion)?;
        self.inner.detect_motion_status()
    }

    fn release_all_keys(&mut self) {
        self.inner.release_all_keys();
    }
    fn on_progress(&mut self, index: usize, total: usize) {
        self.inner.on_progress(index, total);
    }
    fn on_checkpoint(&mut self, checkpoint: &PathingCheckpoint) {
        self.inner.on_checkpoint(checkpoint);
    }
    fn on_log(&mut self, level: LogLevel, message: String) {
        self.inner.on_log(level, message);
    }
}
