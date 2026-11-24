pub mod bitblt;
pub mod pool;
pub mod wgc;
pub mod windows_enum;

use bgi_control::{CaptureMode, FramePreview};
use thiserror::Error;

pub use bitblt::BitBltCapturer;
pub use pool::{FrameBufferPool, PooledBuffer};
pub use wgc::WgcCapturer;
pub use windows_enum::enumerate_candidate_windows;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("目标窗口无效或已关闭")]
    InvalidWindow,
    #[error("目标窗口处于最小化状态")]
    WindowMinimized,
    #[error("当前尚未收到新帧")]
    FrameNotReady,
    #[error("截图接口调用失败: {0}")]
    Api(String),
}

pub enum CapturerBackend {
    BitBlt(BitBltCapturer),
    Wgc(WgcCapturer),
}

impl CapturerBackend {
    pub fn open(hwnd: isize, mode: CaptureMode, min_update_interval_ms: u32) -> Result<Self, CaptureError> {
        match mode {
            CaptureMode::BitBlt => Ok(Self::BitBlt(BitBltCapturer::new(hwnd)?)),
            CaptureMode::Wgc => Ok(Self::Wgc(WgcCapturer::new(hwnd, min_update_interval_ms)?)),
        }
    }

    pub fn set_min_update_interval_ms(&self, min_update_interval_ms: u32) {
        if let Self::Wgc(wgc) = self {
            wgc.set_min_update_interval_ms(min_update_interval_ms);
        }
    }

    pub fn needs_reopen_for_interval(&self, min_update_interval_ms: u32) -> bool {
        match self {
            Self::BitBlt(_) => false,
            Self::Wgc(wgc) => wgc.needs_reopen_for_interval(min_update_interval_ms),
        }
    }

    pub fn wgc_diagnostics(&self) -> Option<String> {
        match self {
            Self::BitBlt(_) => None,
            Self::Wgc(wgc) => Some(wgc.diagnostics_summary()),
        }
    }

    pub fn capture(&mut self, pool: &FrameBufferPool) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        match self {
            Self::BitBlt(c) => c.capture(pool),
            Self::Wgc(c) => c.capture(pool),
        }
    }
}

/// Converts BGRA buffer into downsampled RGBA `FramePreview` on the background thread.
pub fn bgra_to_preview(
    seq: u64,
    src_w: u32,
    src_h: u32,
    bgra: &[u8],
    max_w: u32,
    max_h: u32,
    capture_ms: f32,
) -> FramePreview {
    if src_w == 0 || src_h == 0 || bgra.len() < (src_w as usize) * (src_h as usize) * 4 {
        return FramePreview {
            seq,
            width: 1,
            height: 1,
            source_width: src_w,
            source_height: src_h,
            capture_ms,
            rgba: vec![0, 0, 0, 255],
        };
    }

    let scale_w = (max_w as f32) / (src_w as f32);
    let scale_h = (max_h as f32) / (src_h as f32);
    let scale = scale_w.min(scale_h).min(1.0);

    let dst_w = ((src_w as f32) * scale).round().max(1.0) as u32;
    let dst_h = ((src_h as f32) * scale).round().max(1.0) as u32;

    let sx_lut: Vec<usize> = (0..dst_w)
        .map(|dx| (((dx as u64) * (src_w as u64) / (dst_w as u64)).min((src_w - 1) as u64) as usize) * 4)
        .collect();
    let src_stride = (src_w as usize) * 4;
    let dst_stride = (dst_w as usize) * 4;
    let mut rgba = vec![0u8; dst_stride * (dst_h as usize)];
    for dy in 0..dst_h {
        let sy = ((dy as u64) * (src_h as u64) / (dst_h as u64)).min((src_h - 1) as u64) as usize;
        let src_row = &bgra[sy * src_stride..(sy + 1) * src_stride];
        let dst_row = &mut rgba[(dy as usize) * dst_stride..((dy as usize) + 1) * dst_stride];
        for (dx, &sx_off) in sx_lut.iter().enumerate() {
            let d_off = dx * 4;
            dst_row[d_off] = src_row[sx_off + 2];
            dst_row[d_off + 1] = src_row[sx_off + 1];
            dst_row[d_off + 2] = src_row[sx_off];
            dst_row[d_off + 3] = 255;
        }
    }

    FramePreview {
        seq,
        width: dst_w,
        height: dst_h,
        source_width: src_w,
        source_height: src_h,
        capture_ms,
        rgba,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_preview_swaps_channels_and_downsamples() {
        let mut bgra = vec![0u8; 4 * 4 * 4];
        // Pixel (0, 0): B=10, G=20, R=30, A=0 -> RGBA (30, 20, 10, 255)
        bgra[0] = 10;
        bgra[1] = 20;
        bgra[2] = 30;
        bgra[3] = 0;

        let preview = bgra_to_preview(7, 4, 4, &bgra, 2, 2, 1.5);
        assert_eq!(preview.width, 2);
        assert_eq!(preview.height, 2);
        assert_eq!(&preview.rgba[0..4], &[30, 20, 10, 255]);
    }
}
