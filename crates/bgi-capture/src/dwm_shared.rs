use std::mem::{size_of, transmute};
use std::sync::OnceLock;

use windows::Win32::Foundation::{HANDLE, HMODULE, HWND, POINT, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_FEATURE_LEVEL_11_0,
    D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11CreateDevice,
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
use windows::Win32::Graphics::Gdi::MapWindowPoints;
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetWindowRect, IsIconic, IsWindow};
use windows::core::{BOOL, s, w};

use crate::CaptureError;
use crate::gpu_bgr::GpuBgr24Packer;
use crate::pool::{FrameBufferPool, PooledBuffer};

type DwmGetDxSharedSurfaceFn = unsafe extern "system" fn(
    hwnd: HWND,
    ph_surface: *mut HANDLE,
    p_adapter_luid: *mut i64,
    p_fmt_window: *mut u32,
    p_present_flags: *mut u32,
    p_win32k_update_id: *mut u64,
) -> BOOL;

static DWM_SHARED_FN: OnceLock<Option<DwmGetDxSharedSurfaceFn>> = OnceLock::new();

fn resolve_dwm_get_dx_shared_surface() -> Option<DwmGetDxSharedSurfaceFn> {
    *DWM_SHARED_FN.get_or_init(|| unsafe {
        let user32 = GetModuleHandleW(w!("user32.dll")).ok()?;
        let proc_ptr = GetProcAddress(user32, s!("DwmGetDxSharedSurface"))?;
        Some(transmute::<_, DwmGetDxSharedSurfaceFn>(proc_ptr))
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct ClientOffsets {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

/// Captures DWM's already-composited shared window surface directly without creating a WGC session.
/// Useful on Intel Arc GPUs where `GraphicsCaptureSession` causes periodic `dwm.exe` spikes.
pub struct DwmSharedCapturer {
    hwnd: HWND,
    fetch_fn: DwmGetDxSharedSurfaceFn,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    packer: GpuBgr24Packer,
    surface_w: u32,
    surface_h: u32,
    offsets: ClientOffsets,
    last_update_id: u64,
    cached_bgr: Vec<u8>,
    cached_w: u32,
    cached_h: u32,
    scratch_roi: Vec<u8>,
}

unsafe impl Send for DwmSharedCapturer {}

impl DwmSharedCapturer {
    pub fn new(hwnd: isize) -> Result<Self, CaptureError> {
        let hwnd_win = HWND(hwnd as *mut _);
        if unsafe { !IsWindow(Some(hwnd_win)).as_bool() } {
            return Err(CaptureError::InvalidWindow);
        }
        if unsafe { IsIconic(hwnd_win).as_bool() } {
            return Err(CaptureError::WindowMinimized);
        }
        let fetch_fn = resolve_dwm_get_dx_shared_surface()
            .ok_or_else(|| CaptureError::Api("user32!DwmGetDxSharedSurface 不可用".into()))?;
        let (device, context) = create_d3d11_device()?;
        let packer = GpuBgr24Packer::new(&device);
        Ok(Self {
            hwnd: hwnd_win,
            fetch_fn,
            device,
            context,
            packer,
            surface_w: 0,
            surface_h: 0,
            offsets: ClientOffsets::default(),
            last_update_id: 0,
            cached_bgr: Vec::new(),
            cached_w: 0,
            cached_h: 0,
            scratch_roi: Vec::new(),
        })
    }

    pub fn diagnostics_summary(&self) -> String {
        format!(
            "DwmGetDxSharedSurface(Zero-WGC-Session, gpu_cs={})",
            self.packer.is_gpu_cs_active()
        )
    }

    pub fn capture_bgr(&mut self, pool: &FrameBufferPool) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        self.capture_region(pool, None)
    }

    pub fn capture_bgr_roi(
        &mut self,
        pool: &FrameBufferPool,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        if width == 0 || height == 0 {
            return Err(CaptureError::Api("ROI 宽高不能为 0".into()));
        }
        self.capture_region(pool, Some((x, y, width, height)))
    }

    fn capture_region(
        &mut self,
        pool: &FrameBufferPool,
        roi: Option<(u32, u32, u32, u32)>,
    ) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        if unsafe { !IsWindow(Some(self.hwnd)).as_bool() } {
            return Err(CaptureError::InvalidWindow);
        }
        if unsafe { IsIconic(self.hwnd).as_bool() } {
            return Err(CaptureError::WindowMinimized);
        }

        let mut ph_surface = HANDLE::default();
        let mut adapter_luid = 0i64;
        let mut fmt = 0u32;
        let mut flags = 0u32;
        let mut update_id = 0u64;
        let ok = unsafe {
            (self.fetch_fn)(
                self.hwnd,
                &mut ph_surface,
                &mut adapter_luid,
                &mut fmt,
                &mut flags,
                &mut update_id,
            )
        }
        .as_bool();
        if !ok || ph_surface.is_invalid() {
            return Err(CaptureError::FrameNotReady);
        }

        if update_id != 0 && update_id == self.last_update_id && self.cached_w > 0 && roi.is_none() {
            return Ok(self.checkout_cached_full(pool));
        }

        let mut src_tex: Option<ID3D11Texture2D> = None;
        unsafe { self.device.OpenSharedResource(ph_surface, &mut src_tex) }
            .map_err(|e| CaptureError::Api(format!("OpenSharedResource 失败: {e}")))?;
        let src_tex = src_tex.ok_or_else(|| CaptureError::Api("OpenSharedResource 返回空".into()))?;

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { src_tex.GetDesc(&mut desc) };
        if desc.Width == 0 || desc.Height == 0 {
            return Err(CaptureError::FrameNotReady);
        }
        if desc.Width != self.surface_w || desc.Height != self.surface_h {
            self.surface_w = desc.Width;
            self.surface_h = desc.Height;
            self.offsets = calc_client_offsets(self.hwnd, desc.Width, desc.Height);
            self.packer.reset_buffers();
            self.cached_bgr.clear();
            self.cached_w = 0;
            self.cached_h = 0;
        }

        let left = self.offsets.left.min(desc.Width.saturating_sub(1));
        let top = self.offsets.top.min(desc.Height.saturating_sub(1));
        let client_w = (desc.Width - left).saturating_sub(self.offsets.right).max(1);
        let client_h = (desc.Height - top).saturating_sub(self.offsets.bottom).max(1);

        let (crop_x, crop_y, out_w, out_h) = if let Some((rx, ry, rw, rh)) = roi {
            let cx = left + rx.min(client_w.saturating_sub(1));
            let cy = top + ry.min(client_h.saturating_sub(1));
            let cw = rw.min(left + client_w - cx).max(1);
            let ch = rh.min(top + client_h - cy).max(1);
            (cx, cy, cw, ch)
        } else {
            (left, top, client_w, client_h)
        };

        let used_cs = self.packer.stage_cropped_bgr24(
            &self.device,
            &self.context,
            &src_tex,
            desc.Format,
            crop_x,
            crop_y,
            out_w,
            out_h,
        )?;
        drop(src_tex);

        if roi.is_some() {
            self.packer
                .map_staged_bgr24(&self.context, used_cs, out_w, out_h, &mut self.scratch_roi)?;
            let mut out = pool.checkout(self.scratch_roi.len());
            out.as_mut_slice().copy_from_slice(&self.scratch_roi);
            return Ok((out_w, out_h, out));
        }

        self.packer
            .map_staged_bgr24(&self.context, used_cs, out_w, out_h, &mut self.cached_bgr)?;
        self.cached_w = out_w;
        self.cached_h = out_h;
        self.last_update_id = update_id;
        Ok(self.checkout_cached_full(pool))
    }

    fn checkout_cached_full(&self, pool: &FrameBufferPool) -> (u32, u32, PooledBuffer) {
        let mut out = pool.checkout(self.cached_bgr.len());
        out.as_mut_slice().copy_from_slice(&self.cached_bgr);
        (self.cached_w, self.cached_h, out)
    }
}

fn create_d3d11_device() -> Result<(ID3D11Device, ID3D11DeviceContext), CaptureError> {
    let feature_levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    for driver_type in [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP] {
        let hr = unsafe {
            D3D11CreateDevice(
                None,
                driver_type,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&feature_levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
        if hr.is_ok()
            && let (Some(d), Some(c)) = (device.take(), context.take())
        {
            return Ok((d, c));
        }
    }
    Err(CaptureError::Api("D3D11CreateDevice 初始化失败".into()))
}

fn calc_client_offsets(hwnd: HWND, tex_w: u32, tex_h: u32) -> ClientOffsets {
    if unsafe { IsIconic(hwnd).as_bool() } {
        return ClientOffsets::default();
    }
    let mut client_rect = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut client_rect) }.is_err() {
        return ClientOffsets::default();
    }
    let mut pts = [
        POINT { x: client_rect.left, y: client_rect.top },
        POINT { x: client_rect.right, y: client_rect.bottom },
    ];
    unsafe {
        if MapWindowPoints(Some(hwnd), None, &mut pts) == 0 {
            return ClientOffsets::default();
        }
    }
    let screen_client = RECT {
        left: pts[0].x,
        top: pts[0].y,
        right: pts[1].x,
        bottom: pts[1].y,
    };
    let mut frame_bounds = RECT::default();
    let dwm_ok = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &mut frame_bounds as *mut RECT as *mut _,
            size_of::<RECT>() as u32,
        )
    }
    .is_ok();
    let mut window_rect = RECT::default();
    let win_ok = unsafe { GetWindowRect(hwnd, &mut window_rect) }.is_ok();

    let chosen = if dwm_ok
        && (tex_w as i32 == frame_bounds.right - frame_bounds.left
            || tex_h as i32 == frame_bounds.bottom - frame_bounds.top)
    {
        frame_bounds
    } else if win_ok
        && (tex_w as i32 == window_rect.right - window_rect.left
            || tex_h as i32 == window_rect.bottom - window_rect.top)
    {
        window_rect
    } else if dwm_ok {
        frame_bounds
    } else if win_ok {
        window_rect
    } else {
        return ClientOffsets::default();
    };

    ClientOffsets {
        left: (screen_client.left - chosen.left).max(0) as u32,
        top: (screen_client.top - chosen.top).max(0) as u32,
        right: (chosen.right - screen_client.right).max(0) as u32,
        bottom: (chosen.bottom - screen_client.bottom).max(0) as u32,
    }
}
