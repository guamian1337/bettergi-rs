use std::sync::Arc;
use std::time::{Duration, Instant};

use bgi_capture::{CapturedFrame, CapturerBackend, FrameBufferPool, frame_to_preview};
use bgi_control::{CaptureMode, CoreEndpoint, LogLevel, UiSnapshot};

pub(crate) type ActiveCapturer = Option<(isize, CaptureMode, CapturerBackend)>;

pub(crate) struct CapturePaceState {
    window_start: Instant,
    window_frames: u64,
    pub(crate) last_stats_publish: Instant,
}

impl CapturePaceState {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            window_start: now,
            window_frames: 0,
            last_stats_publish: now,
        }
    }

    pub(crate) fn reset(&mut self) {
        let now = Instant::now();
        self.window_start = now;
        self.window_frames = 0;
        self.last_stats_publish = now;
    }
}

pub(crate) fn close_capturer(capturer: &mut ActiveCapturer, core: &CoreEndpoint) {
    if let Some((_, _, backend)) = capturer.take()
        && let Some(diag) = backend.diagnostics()
    {
        core.emit_log(LogLevel::Info, "Capture", format!("会话关闭: {diag}"));
    }
}

pub(crate) fn release_idle_capturer(
    snapshot: &UiSnapshot,
    capturer: &mut ActiveCapturer,
    core: &CoreEndpoint,
) {
    if !snapshot.preview_enabled
        && !snapshot.triggers_active
        && !snapshot.task_state.is_task_active()
    {
        close_capturer(capturer, core);
    }
}

fn ensure_capturer<'a>(
    snapshot: &UiSnapshot,
    capturer: &'a mut ActiveCapturer,
    core: &CoreEndpoint,
) -> Result<&'a mut CapturerBackend, String> {
    let Some(win) = &snapshot.selected_window else {
        return Err("尚未选择目标窗口".to_string());
    };
    let mode = snapshot.config.capture_mode;

    let need_reopen = match capturer {
        Some((h, m, _)) => *h != win.hwnd || *m != mode,
        None => true,
    };

    if need_reopen {
        close_capturer(capturer, core);
        let interval_ms = (1000 / snapshot.config.preview_fps.clamp(1, 60)).max(16);
        let backend =
            CapturerBackend::open(win.hwnd, mode, interval_ms).map_err(|e| e.to_string())?;
        if let Some(diag) = backend.diagnostics() {
            core.emit_log(LogLevel::Info, "Capture", format!("会话启动: {diag}"));
        }
        *capturer = Some((win.hwnd, mode, backend));
    }

    Ok(&mut capturer.as_mut().unwrap().2)
}

pub(crate) fn capture_raw_frame(
    snapshot: &UiSnapshot,
    capturer: &mut ActiveCapturer,
    pool: &FrameBufferPool,
    core: &CoreEndpoint,
) -> Result<Option<CapturedFrame>, String> {
    let backend = ensure_capturer(snapshot, capturer, core)?;
    match backend.capture(pool) {
        Ok(res) => Ok(Some(res)),
        Err(bgi_capture::CaptureError::FrameNotReady) => Ok(None),
        Err(e) => {
            close_capturer(capturer, core);
            Err(e.to_string())
        }
    }
}

fn record_capture_stats(
    snapshot: &mut UiSnapshot,
    pace: &mut CapturePaceState,
    sw: u32,
    sh: u32,
    capture_ms: f32,
) {
    pace.window_frames += 1;
    let elapsed = pace.window_start.elapsed();
    if elapsed >= Duration::from_millis(500) {
        snapshot.capture_stats.measured_fps = (pace.window_frames as f32) / elapsed.as_secs_f32();
        pace.window_start = Instant::now();
        pace.window_frames = 0;
    }
    snapshot.capture_stats.captured_frames += 1;
    snapshot.capture_stats.source_width = sw;
    snapshot.capture_stats.source_height = sh;
    snapshot.capture_stats.last_capture_ms = capture_ms;
}

pub(crate) fn capture_frame(
    snapshot: &mut UiSnapshot,
    capturer: &mut ActiveCapturer,
    pool: &FrameBufferPool,
    frame_seq: &mut u64,
    pace: &mut CapturePaceState,
    core: &CoreEndpoint,
) -> Result<Option<(u64, CapturedFrame)>, String> {
    let started = Instant::now();
    let Some(frame) = capture_raw_frame(snapshot, capturer, pool, core)? else {
        return Ok(None);
    };
    let capture_ms = started.elapsed().as_secs_f32() * 1000.0;
    record_capture_stats(snapshot, pace, frame.width, frame.height, capture_ms);
    *frame_seq += 1;
    Ok(Some((*frame_seq, frame)))
}

pub(crate) fn capture_once(
    snapshot: &mut UiSnapshot,
    capturer: &mut ActiveCapturer,
    pool: &FrameBufferPool,
    frame_seq: &mut u64,
    pace: &mut CapturePaceState,
    core: &CoreEndpoint,
) -> Result<(), String> {
    let capture_ms = snapshot.capture_stats.last_capture_ms;
    let Some((seq, frame)) = capture_frame(snapshot, capturer, pool, frame_seq, pace, core)? else {
        return Ok(());
    };
    if snapshot.config.capture_only_no_render {
        drop(frame);
    } else {
        let preview = frame_to_preview(seq, &frame, 960, 540, capture_ms);
        core.publish_preview(preview);
    }
    if pace.last_stats_publish.elapsed() >= Duration::from_secs(1) {
        pace.last_stats_publish = Instant::now();
        core.publish_snapshot(Arc::new(snapshot.clone()));
    }
    Ok(())
}
