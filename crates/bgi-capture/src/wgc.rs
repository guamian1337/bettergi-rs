use std::ptr::copy_nonoverlapping;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_WAIT_TIMEOUT, IDXGIAdapter1, IDXGIDevice};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::WindowsAndMessaging::{
    GWL_EXSTYLE, GetClientRect, GetWindowLongW, GetWindowRect, IsIconic, IsWindow, WS_EX_TOPMOST,
};
use windows::core::Interface;
use windows_capture::capture::{CaptureControl, Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window as WcWindow;

use crate::CaptureError;
use crate::pool::{FrameBufferPool, PooledBuffer};

struct SharedLatestFrame {
    is_capturing: bool,
    seq: u64,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
}

struct WgcTelemetry {
    interval_ms: AtomicU32,
    arrivals: AtomicU64,
    copied_frames: AtomicU64,
    throttled_skips: AtomicU64,
    adapter_name: Mutex<String>,
}

impl WgcTelemetry {
    fn new(interval_ms: u32) -> Self {
        Self {
            interval_ms: AtomicU32::new(interval_ms),
            arrivals: AtomicU64::new(0),
            copied_frames: AtomicU64::new(0),
            throttled_skips: AtomicU64::new(0),
            adapter_name: Mutex::new(String::from("Unknown")),
        }
    }
}

struct WgcHandlerFlags {
    hwnd_raw: isize,
    shared: Arc<(Mutex<SharedLatestFrame>, Condvar)>,
    telemetry: Arc<WgcTelemetry>,
}

struct WgcHandler {
    hwnd: HWND,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    shared: Arc<(Mutex<SharedLatestFrame>, Condvar)>,
    telemetry: Arc<WgcTelemetry>,
    staging: Option<(u32, u32, ID3D11Texture2D)>,
    scratch_bgra: Vec<u8>,
    last_ts: i64,
    last_at: Option<Instant>,
}

// Safety: WgcHandler and all its D3D11 objects stay exclusively on windows-capture's single capture thread.
unsafe impl Send for WgcHandler {}

impl GraphicsCaptureApiHandler for WgcHandler {
    type Flags = WgcHandlerFlags;
    type Error = CaptureError;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        if let Ok(dxgi_dev) = ctx.device.cast::<IDXGIDevice>() {
            let name = query_adapter_name(&dxgi_dev);
            if let Ok(mut guard) = ctx.flags.telemetry.adapter_name.lock() {
                *guard = name;
            }
        }

        Ok(Self {
            hwnd: HWND(ctx.flags.hwnd_raw as *mut _),
            device: ctx.device,
            context: ctx.device_context,
            shared: ctx.flags.shared,
            telemetry: ctx.flags.telemetry,
            staging: None,
            scratch_bgra: Vec::new(),
            last_ts: 0,
            last_at: None,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        self.telemetry.arrivals.fetch_add(1, Ordering::Relaxed);
        if unsafe { IsIconic(self.hwnd).as_bool() } {
            return Ok(());
        }

        let frame_ts = frame.timestamp().map(|t| t.Duration).unwrap_or(0);
        if frame_ts != 0 && frame_ts == self.last_ts {
            return Ok(());
        }

        let min_interval_ms = self.telemetry.interval_ms.load(Ordering::Relaxed);
        let now = Instant::now();
        if should_throttle_capture(self.last_at, !self.scratch_bgra.is_empty(), min_interval_ms, now)
        {
            self.telemetry.throttled_skips.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let src_w = frame.width();
        let src_h = frame.height();
        if src_w == 0 || src_h == 0 {
            return Ok(());
        }

        let (crop_x, crop_y, out_w, out_h) = compute_client_crop(self.hwnd, src_w, src_h);
        ensure_staging(&self.device, &mut self.staging, out_w, out_h)?;
        let staging_tex = self.staging.as_ref().unwrap().2.clone();

        let src_box = D3D11_BOX {
            left: crop_x,
            top: crop_y,
            front: 0,
            right: crop_x + out_w,
            bottom: crop_y + out_h,
            back: 1,
        };
        unsafe {
            self.context.CopySubresourceRegion(
                &staging_tex,
                0,
                0,
                0,
                0,
                frame.as_raw_texture(),
                0,
                Some(&src_box),
            );
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(&staging_tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| {
                    if e.code() == DXGI_ERROR_WAIT_TIMEOUT {
                        CaptureError::FrameNotReady
                    } else {
                        CaptureError::Api(format!("Map staging texture 失败: {e}"))
                    }
                })?;
        }

        let row_bytes = (out_w as usize) * 4;
        let total_bytes = row_bytes * (out_h as usize);
        self.scratch_bgra.resize(total_bytes, 0);
        let src_pitch = mapped.RowPitch as usize;
        let src_base = mapped.pData as *const u8;
        for y in 0..(out_h as usize) {
            unsafe {
                copy_nonoverlapping(
                    src_base.add(y * src_pitch),
                    self.scratch_bgra.as_mut_ptr().add(y * row_bytes),
                    row_bytes,
                );
            }
        }
        unsafe {
            self.context.Unmap(&staging_tex, 0);
        }

        self.last_ts = frame_ts;
        self.last_at = Some(now);
        self.telemetry.copied_frames.fetch_add(1, Ordering::Relaxed);

        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.width = out_w;
            guard.height = out_h;
            guard.seq = guard.seq.wrapping_add(1).max(1);
            std::mem::swap(&mut guard.bgra, &mut self.scratch_bgra);
            cv.notify_all();
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        let (lock, cv) = &*self.shared;
        if let Ok(mut guard) = lock.lock() {
            guard.is_capturing = false;
            cv.notify_all();
        }
        self.staging = None;
        Ok(())
    }
}

pub struct WgcCapturer {
    hwnd: HWND,
    shared: Arc<(Mutex<SharedLatestFrame>, Condvar)>,
    telemetry: Arc<WgcTelemetry>,
    control: Option<CaptureControl<WgcHandler, CaptureError>>,
    os_min_interval_ms: u32,
    os_interval_desc: String,
}

unsafe impl Send for WgcCapturer {}

impl WgcCapturer {
    pub fn new(hwnd: isize, min_update_interval_ms: u32) -> Result<Self, CaptureError> {
        let hwnd_win = HWND(hwnd as *mut _);
        if unsafe { !IsWindow(Some(hwnd_win)).as_bool() } {
            return Err(CaptureError::InvalidWindow);
        }
        if unsafe { IsIconic(hwnd_win).as_bool() } {
            return Err(CaptureError::WindowMinimized);
        }

        let cursor_setting = if GraphicsCaptureApi::is_cursor_settings_supported().unwrap_or(false) {
            CursorCaptureSettings::WithoutCursor
        } else {
            CursorCaptureSettings::Default
        };
        let border_setting = if GraphicsCaptureApi::is_border_settings_supported().unwrap_or(false) {
            DrawBorderSettings::WithoutBorder
        } else {
            DrawBorderSettings::Default
        };
        let (min_interval_setting, os_interval_desc) =
            if min_update_interval_ms > 0
                && GraphicsCaptureApi::is_minimum_update_interval_supported().unwrap_or(false)
            {
                (
                    MinimumUpdateIntervalSettings::Custom(Duration::from_millis(
                        min_update_interval_ms as u64,
                    )),
                    format!("Custom({min_update_interval_ms}ms)"),
                )
            } else {
                (
                    MinimumUpdateIntervalSettings::Default,
                    "Default".to_string(),
                )
            };

        let shared = Arc::new((
            Mutex::new(SharedLatestFrame {
                is_capturing: true,
                seq: 0,
                width: 0,
                height: 0,
                bgra: Vec::new(),
            }),
            Condvar::new(),
        ));
        let telemetry = Arc::new(WgcTelemetry::new(min_update_interval_ms));

        let flags = WgcHandlerFlags {
            hwnd_raw: hwnd,
            shared: shared.clone(),
            telemetry: telemetry.clone(),
        };
        let target = WcWindow::from_raw_hwnd(hwnd as *mut _);
        let settings = Settings::new(
            target,
            cursor_setting,
            border_setting,
            SecondaryWindowSettings::Default,
            min_interval_setting,
            DirtyRegionSettings::Default,
            ColorFormat::Bgra8,
            flags,
        );

        let control = WgcHandler::start_free_threaded(settings)
            .map_err(|e| CaptureError::Api(format!("windows-capture 启动失败: {e}")))?;

        Ok(Self {
            hwnd: hwnd_win,
            shared,
            telemetry,
            control: Some(control),
            os_min_interval_ms: min_update_interval_ms,
            os_interval_desc,
        })
    }

    pub fn set_min_update_interval_ms(&self, min_update_interval_ms: u32) {
        self.telemetry
            .interval_ms
            .store(min_update_interval_ms, Ordering::Relaxed);
    }

    pub fn needs_reopen_for_interval(&self, interval_ms: u32) -> bool {
        self.os_min_interval_ms > 0 && interval_ms < self.os_min_interval_ms
    }

    pub fn diagnostics_summary(&self) -> String {
        let adapter = self
            .telemetry
            .adapter_name
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "Unknown".to_string());
        format!(
            "windows-capture(CopySubresource+SingleThread), adapter={}, min_interval={} (soft={}ms), arrivals={}, copied={}, throttled_skips={}",
            adapter,
            self.os_interval_desc,
            self.telemetry.interval_ms.load(Ordering::Relaxed),
            self.telemetry.arrivals.load(Ordering::Relaxed),
            self.telemetry.copied_frames.load(Ordering::Relaxed),
            self.telemetry.throttled_skips.load(Ordering::Relaxed),
        )
    }

    pub fn capture(&mut self, pool: &FrameBufferPool) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        if unsafe { !IsWindow(Some(self.hwnd)).as_bool() } {
            return Err(CaptureError::InvalidWindow);
        }
        if unsafe { IsIconic(self.hwnd).as_bool() } {
            return Err(CaptureError::WindowMinimized);
        }

        let (lock, cv) = &*self.shared;
        let mut guard = lock.lock().unwrap();
        if !guard.is_capturing {
            return Err(CaptureError::InvalidWindow);
        }
        if guard.seq == 0 {
            let (g, _) = cv
                .wait_timeout_while(guard, Duration::from_millis(200), |s| {
                    s.is_capturing && s.seq == 0
                })
                .unwrap();
            guard = g;
            if guard.seq == 0 {
                return Err(CaptureError::FrameNotReady);
            }
        }

        let mut out = pool.checkout(guard.bgra.len());
        out.as_mut_slice().copy_from_slice(&guard.bgra);
        Ok((guard.width, guard.height, out))
    }
}

impl Drop for WgcCapturer {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.shared.0.lock() {
            guard.is_capturing = false;
            self.shared.1.notify_all();
        }
        if let Some(ctrl) = self.control.take() {
            let _ = ctrl.stop();
        }
    }
}

fn should_throttle_capture(
    last_at: Option<Instant>,
    has_frame: bool,
    min_update_interval_ms: u32,
    now: Instant,
) -> bool {
    if !has_frame || min_update_interval_ms == 0 {
        return false;
    }
    let Some(prev) = last_at else {
        return false;
    };
    now.saturating_duration_since(prev) < Duration::from_millis(min_update_interval_ms as u64)
}

fn query_adapter_name(dxgi_device: &IDXGIDevice) -> String {
    unsafe {
        let Ok(adapter) = dxgi_device.GetAdapter() else {
            return "Unknown".to_string();
        };
        let Ok(adapter1) = adapter.cast::<IDXGIAdapter1>() else {
            return "Unknown".to_string();
        };
        let Ok(desc) = adapter1.GetDesc1() else {
            return "Unknown".to_string();
        };
        let end = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        String::from_utf16_lossy(&desc.Description[..end])
    }
}

fn ensure_staging(
    device: &ID3D11Device,
    slot: &mut Option<(u32, u32, ID3D11Texture2D)>,
    width: u32,
    height: u32,
) -> Result<(), CaptureError> {
    if let Some((w, h, _)) = slot
        && *w == width
        && *h == height
    {
        return Ok(());
    }

    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
    };

    let mut tex = None;
    unsafe {
        device
            .CreateTexture2D(&desc, None, Some(&mut tex))
            .map_err(|e| CaptureError::Api(format!("CreateTexture2D staging 失败: {e}")))?;
    }
    let tex = tex.ok_or_else(|| CaptureError::Api("staging texture 为空".into()))?;
    *slot = Some((width, height, tex));
    Ok(())
}

fn compute_client_crop(hwnd: HWND, tex_w: u32, tex_h: u32) -> (u32, u32, u32, u32) {
    let ex_style = unsafe { GetWindowLongW(hwnd, GWL_EXSTYLE) };
    if (ex_style & (WS_EX_TOPMOST.0 as i32)) != 0 {
        return (0, 0, tex_w, tex_h);
    }

    let mut frame_rect = RECT::default();
    let dwm_ok = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &mut frame_rect as *mut RECT as *mut _,
            size_of::<RECT>() as u32,
        )
    }
    .is_ok();
    if !dwm_ok && unsafe { GetWindowRect(hwnd, &mut frame_rect) }.is_err() {
        return (0, 0, tex_w, tex_h);
    }

    let mut cli_rect = RECT::default();
    let mut cli_origin = POINT::default();
    let ok = unsafe {
        GetClientRect(hwnd, &mut cli_rect).is_ok() && ClientToScreen(hwnd, &mut cli_origin).as_bool()
    };
    if !ok {
        return (0, 0, tex_w, tex_h);
    }

    let left = (cli_origin.x - frame_rect.left).max(0) as u32;
    let top = (cli_origin.y - frame_rect.top).max(0) as u32;
    let cw = (cli_rect.right - cli_rect.left).max(1) as u32;
    let ch = (cli_rect.bottom - cli_rect.top).max(1) as u32;

    if left >= tex_w || top >= tex_h {
        return (0, 0, tex_w, tex_h);
    }

    let out_w = cw.min(tex_w - left).max(1);
    let out_h = ch.min(tex_h - top).max(1);
    (left, top, out_w, out_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_throttle_gates_overdelivery_without_blocking_first_or_unthrottled() {
        let t0 = Instant::now();
        assert!(!should_throttle_capture(None, false, 100, t0));
        assert!(!should_throttle_capture(Some(t0), false, 100, t0 + Duration::from_millis(5)));
        assert!(!should_throttle_capture(Some(t0), true, 0, t0 + Duration::from_millis(5)));
        assert!(should_throttle_capture(Some(t0), true, 100, t0 + Duration::from_millis(16)));
        assert!(!should_throttle_capture(Some(t0), true, 100, t0 + Duration::from_millis(100)));
    }
}
