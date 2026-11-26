pub mod bitblt;
pub mod dwm_shared;
pub mod gpu_bgr;
pub mod pool;
pub mod replay;
pub mod wgc;
pub mod windows_enum;

use bgi_control::{CaptureMode, FramePreview};
use thiserror::Error;

pub use bitblt::BitBltCapturer;
pub use dwm_shared::DwmSharedCapturer;
pub use pool::{FrameBufferPool, PooledBuffer};
pub use replay::{FrameRecord, FrameRecorder, FrameReplay, ReplayError};
pub use wgc::WgcCapturer;
pub use windows_enum::enumerate_candidate_windows;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Bgra32,
    Bgr24,
}

impl PixelFormat {
    pub fn is_bgr24(self) -> bool {
        matches!(self, Self::Bgr24)
    }
}

pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: PooledBuffer,
}

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
    DwmShared(DwmSharedCapturer),
}

impl CapturerBackend {
    pub fn open(
        hwnd: isize,
        mode: CaptureMode,
        min_update_interval_ms: u32,
    ) -> Result<Self, CaptureError> {
        match mode {
            CaptureMode::BitBlt => Ok(Self::BitBlt(BitBltCapturer::new(hwnd)?)),
            CaptureMode::Wgc => Ok(Self::Wgc(WgcCapturer::new(hwnd, min_update_interval_ms)?)),
            CaptureMode::DwmSharedSurface => Ok(Self::DwmShared(DwmSharedCapturer::new(hwnd)?)),
        }
    }

    pub fn set_min_update_interval_ms(&self, min_update_interval_ms: u32) {
        if let Self::Wgc(wgc) = self {
            wgc.set_min_update_interval_ms(min_update_interval_ms);
        }
    }

    pub fn needs_reopen_for_interval(&self, min_update_interval_ms: u32) -> bool {
        match self {
            Self::Wgc(wgc) => wgc.needs_reopen_for_interval(min_update_interval_ms),
            Self::BitBlt(_) | Self::DwmShared(_) => false,
        }
    }

    pub fn diagnostics(&self) -> Option<String> {
        match self {
            Self::BitBlt(_) => None,
            Self::Wgc(wgc) => Some(wgc.diagnostics_summary()),
            Self::DwmShared(d) => Some(d.diagnostics_summary()),
        }
    }

    pub fn capture(&mut self, pool: &FrameBufferPool) -> Result<CapturedFrame, CaptureError> {
        match self {
            Self::BitBlt(c) => {
                let (width, height, data) = c.capture(pool)?;
                Ok(CapturedFrame {
                    width,
                    height,
                    format: PixelFormat::Bgra32,
                    data,
                })
            }
            Self::Wgc(c) => {
                let (width, height, data) = c.capture_bgr(pool)?;
                Ok(CapturedFrame {
                    width,
                    height,
                    format: PixelFormat::Bgr24,
                    data,
                })
            }
            Self::DwmShared(d) => {
                let (width, height, data) = d.capture_bgr(pool)?;
                Ok(CapturedFrame {
                    width,
                    height,
                    format: PixelFormat::Bgr24,
                    data,
                })
            }
        }
    }

    pub fn capture_roi(
        &mut self,
        pool: &FrameBufferPool,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<CapturedFrame, CaptureError> {
        match self {
            Self::Wgc(c) => {
                let (rw, rh, data) = c.capture_bgr_roi(pool, x, y, width, height)?;
                Ok(CapturedFrame {
                    width: rw,
                    height: rh,
                    format: PixelFormat::Bgr24,
                    data,
                })
            }
            Self::DwmShared(d) => {
                let (rw, rh, data) = d.capture_bgr_roi(pool, x, y, width, height)?;
                Ok(CapturedFrame {
                    width: rw,
                    height: rh,
                    format: PixelFormat::Bgr24,
                    data,
                })
            }
            Self::BitBlt(c) => {
                let (fw, fh, full) = c.capture(pool)?;
                if fw == 0 || fh == 0 || width == 0 || height == 0 {
                    return Err(CaptureError::Api("ROI 宽高不能为 0".into()));
                }
                let rx = x.min(fw - 1);
                let ry = y.min(fh - 1);
                let rw = width.min(fw - rx).max(1);
                let rh = height.min(fh - ry).max(1);
                let mut out = pool.checkout((rw as usize) * (rh as usize) * 4);
                let src = full.as_slice();
                let dst = out.as_mut_slice();
                let row_bytes = (rw as usize) * 4;
                let src_stride = (fw as usize) * 4;
                for row in 0..(rh as usize) {
                    let s_off = ((ry as usize) + row) * src_stride + (rx as usize) * 4;
                    let d_off = row * row_bytes;
                    dst[d_off..d_off + row_bytes].copy_from_slice(&src[s_off..s_off + row_bytes]);
                }
                Ok(CapturedFrame {
                    width: rw,
                    height: rh,
                    format: PixelFormat::Bgra32,
                    data: out,
                })
            }
        }
    }
}

struct PreviewPixels<'a> {
    width: u32,
    height: u32,
    pixels: &'a [u8],
    format: PixelFormat,
}

/// Converts a captured `BGR24` or `BGRA32` buffer into a downsampled `RGBA` `FramePreview`.
pub fn frame_to_preview(
    seq: u64,
    frame: &CapturedFrame,
    max_w: u32,
    max_h: u32,
    capture_ms: f32,
) -> FramePreview {
    pixels_to_preview(
        seq,
        PreviewPixels {
            width: frame.width,
            height: frame.height,
            pixels: frame.data.as_slice(),
            format: frame.format,
        },
        max_w,
        max_h,
        capture_ms,
    )
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
    pixels_to_preview(
        seq,
        PreviewPixels {
            width: src_w,
            height: src_h,
            pixels: bgra,
            format: PixelFormat::Bgra32,
        },
        max_w,
        max_h,
        capture_ms,
    )
}

fn pixels_to_preview(
    seq: u64,
    source: PreviewPixels<'_>,
    max_w: u32,
    max_h: u32,
    capture_ms: f32,
) -> FramePreview {
    let PreviewPixels {
        width: src_w,
        height: src_h,
        pixels,
        format,
    } = source;
    let bpp = match format {
        PixelFormat::Bgra32 => 4usize,
        PixelFormat::Bgr24 => 3usize,
    };
    if src_w == 0 || src_h == 0 || pixels.len() < (src_w as usize) * (src_h as usize) * bpp {
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
        .map(|dx| {
            (((dx as u64) * (src_w as u64) / (dst_w as u64)).min((src_w - 1) as u64) as usize) * bpp
        })
        .collect();
    let src_stride = (src_w as usize) * bpp;
    let dst_stride = (dst_w as usize) * 4;
    let mut rgba = vec![0u8; dst_stride * (dst_h as usize)];
    for dy in 0..dst_h {
        let sy = ((dy as u64) * (src_h as u64) / (dst_h as u64)).min((src_h - 1) as u64) as usize;
        let src_row = &pixels[sy * src_stride..(sy + 1) * src_stride];
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
        let bgra = vec![
            10, 20, 30, 0, // (0,0) B=10,G=20,R=30
            40, 50, 60, 0, // (1,0)
            70, 80, 90, 0, // (0,1)
            100, 110, 120, 0, // (1,1)
        ];
        let preview = bgra_to_preview(7, 2, 2, &bgra, 2, 2, 1.5);
        assert_eq!(preview.seq, 7);
        assert_eq!(preview.width, 2);
        assert_eq!(preview.height, 2);
        assert_eq!(&preview.rgba[0..4], &[30, 20, 10, 255]);
    }

    #[test]
    fn bgr24_preview_swaps_channels_without_alpha() {
        let bgr = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let pool = FrameBufferPool::new(1);
        let mut buf = pool.checkout(bgr.len());
        buf.as_mut_slice().copy_from_slice(&bgr);
        let frame = CapturedFrame {
            width: 2,
            height: 2,
            format: PixelFormat::Bgr24,
            data: buf,
        };
        let preview = frame_to_preview(3, &frame, 2, 2, 0.5);
        assert_eq!(&preview.rgba[0..4], &[30, 20, 10, 255]);
    }
}
