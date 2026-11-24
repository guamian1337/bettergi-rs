use std::collections::VecDeque;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use windows::Foundation::TypedEventHandler;
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem,
    GraphicsCaptureSession,
};
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::{HMODULE, HWND, LPARAM, POINT, RECT, RPC_E_CHANGED_MODE, WPARAM};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_FEATURE_LEVEL_11_0,
    D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11CreateDevice,
    ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice};
use windows::Win32::Graphics::Gdi::MapWindowPoints;
use windows::Win32::System::Com::CoIncrementMTAUsage;
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize, RoUninitialize};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
    SetThreadDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetClientRect, GetMessageW, GetWindowRect, IsIconic, IsWindow, MSG,
    PM_NOREMOVE, PeekMessageW, PostThreadMessageW, TranslateMessage, WM_APP,
};
use windows::core::{IInspectable, Interface};

use crate::CaptureError;
use crate::gpu_bgr::GpuBgr24Packer;
use crate::pool::{FrameBufferPool, PooledBuffer};

const CAPTURE_POOL_SIZE: i32 = 2;
const FRAME_WAIT_TIMEOUT: Duration = Duration::from_millis(200);
const CTRL_WAIT_TIMEOUT: Duration = Duration::from_millis(1000);
const WM_WGC_EXEC: u32 = WM_APP + 101;
const WM_WGC_SHUTDOWN: u32 = WM_APP + 102;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ClientAreaOffsets {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

struct WgcMailbox {
    is_capturing: bool,
    pool_generation: u64,
    produced: Option<(Direct3D11CaptureFrame, u64)>,
}

// Safety: WinRT Direct3D11CaptureFrame from CreateFreeThreaded is MTA agile.
unsafe impl Send for WgcMailbox {}

struct WgcTelemetry {
    interval_ms: AtomicU32,
    arrivals: AtomicU64,
    copied_frames: AtomicU64,
    throttled_skips: AtomicU64,
    recreates: AtomicU64,
    min_interval_supported: AtomicBool,
    gpu_bgr_cs: AtomicBool,
    adapter_name: Mutex<String>,
}

impl WgcTelemetry {
    fn new(interval_ms: u32) -> Self {
        Self {
            interval_ms: AtomicU32::new(interval_ms),
            arrivals: AtomicU64::new(0),
            copied_frames: AtomicU64::new(0),
            throttled_skips: AtomicU64::new(0),
            recreates: AtomicU64::new(0),
            min_interval_supported: AtomicBool::new(false),
            gpu_bgr_cs: AtomicBool::new(false),
            adapter_name: Mutex::new(String::from("Unknown")),
        }
    }
}

struct WgcShared {
    mailbox: Mutex<WgcMailbox>,
    cv: Condvar,
    telemetry: WgcTelemetry,
}

type WgcTaskFn = Box<dyn FnOnce(&mut WgcSessionOwner) + Send>;

struct WgcSessionOwner {
    hwnd: HWND,
    winrt_device: IDirect3DDevice,
    item: GraphicsCaptureItem,
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    cap_size: SizeInt32,
    client_offsets: ClientAreaOffsets,
    closed_token: i64,
    frame_arrived_token: i64,
}

impl WgcSessionOwner {
    fn close(self) {
        let _ = self.item.RemoveClosed(self.closed_token);
        let _ = self.frame_pool.RemoveFrameArrived(self.frame_arrived_token);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}

static WINRT_MTA_ANCHOR: OnceLock<usize> = OnceLock::new();

/// Keeps process-wide WinRT MTA alive so `windows-rs` static `FactoryCache` pointers never dangle.
fn ensure_winrt_mta_anchor() {
    let _ = WINRT_MTA_ANCHOR.get_or_init(|| {
        unsafe { CoIncrementMTAUsage() }
            .map(|c| c.0 as usize)
            .unwrap_or(0)
    });
}

struct RoInitGuard {
    should_uninit: bool,
}

impl RoInitGuard {
    fn init_mta() -> Result<Self, CaptureError> {
        ensure_winrt_mta_anchor();
        match unsafe { RoInitialize(RO_INIT_MULTITHREADED) } {
            Ok(()) => Ok(Self { should_uninit: true }),
            Err(e) if e.code() == RPC_E_CHANGED_MODE => Ok(Self { should_uninit: false }),
            Err(e) => Err(CaptureError::Api(format!("RoInitialize(MTA) 失败: {e}"))),
        }
    }
}

impl Drop for RoInitGuard {
    fn drop(&mut self) {
        if self.should_uninit {
            unsafe { RoUninitialize() };
        }
    }
}

struct WgcInitPayload {
    thread_id: u32,
    d3d_device: ID3D11Device,
    d3d_context: ID3D11DeviceContext,
    cap_size: SizeInt32,
    client_offsets: ClientAreaOffsets,
}

unsafe impl Send for WgcInitPayload {}

/// Self-written WGC backend: FFmpeg lifecycle/thread-affinity + Sunshine 2-slot zero-D3D arrival + GPU BGR24 packing.
pub struct WgcCapturer {
    hwnd: HWND,
    d3d_device: ID3D11Device,
    d3d_context: ID3D11DeviceContext,
    shared: Arc<WgcShared>,
    wgc_thread_id: u32,
    task_queue: Arc<Mutex<VecDeque<WgcTaskFn>>>,
    shutting_down: Arc<AtomicBool>,
    thread_handle: Option<JoinHandle<()>>,
    cap_size: SizeInt32,
    client_offsets: ClientAreaOffsets,
    current_gen: u64,
    packer: GpuBgr24Packer,
    scratch_roi: Vec<u8>,
    cached_bgr: Vec<u8>,
    cached_w: u32,
    cached_h: u32,
    last_ts: i64,
    last_capture_at: Option<Instant>,
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

        let shared = Arc::new(WgcShared {
            mailbox: Mutex::new(WgcMailbox {
                is_capturing: true,
                pool_generation: 1,
                produced: None,
            }),
            cv: Condvar::new(),
            telemetry: WgcTelemetry::new(min_update_interval_ms),
        });
        let task_queue: Arc<Mutex<VecDeque<WgcTaskFn>>> = Arc::new(Mutex::new(VecDeque::new()));
        let shutting_down = Arc::new(AtomicBool::new(false));
        let (init_tx, init_rx) = mpsc::channel::<Result<WgcInitPayload, CaptureError>>();

        let shared_thread = shared.clone();
        let queue_thread = task_queue.clone();
        let shut_thread = shutting_down.clone();

        let thread_handle = thread::Builder::new()
            .name("bgi-wgc-ctrl".to_string())
            .spawn(move || {
                run_wgc_thread(
                    hwnd,
                    min_update_interval_ms,
                    shared_thread,
                    queue_thread,
                    shut_thread,
                    init_tx,
                );
            })
            .map_err(|e| CaptureError::Api(format!("启动 WGC 控制线程失败: {e}")))?;

        let init = match init_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(payload)) => payload,
            Ok(Err(e)) => {
                shutting_down.store(true, Ordering::Release);
                let _ = thread_handle.join();
                return Err(e);
            }
            Err(_) => {
                shutting_down.store(true, Ordering::Release);
                return Err(CaptureError::Api("WGC 初始化超时".into()));
            }
        };

        let packer = GpuBgr24Packer::new(&init.d3d_device);
        shared
            .telemetry
            .gpu_bgr_cs
            .store(packer.is_gpu_cs_active(), Ordering::Relaxed);

        let effective_ms = effective_min_interval_ms(min_update_interval_ms);
        let os_interval_desc = if shared.telemetry.min_interval_supported.load(Ordering::Acquire) {
            format!("{effective_ms}ms")
        } else {
            "SoftOnly".to_string()
        };

        Ok(Self {
            hwnd: hwnd_win,
            d3d_device: init.d3d_device,
            d3d_context: init.d3d_context,
            shared,
            wgc_thread_id: init.thread_id,
            task_queue,
            shutting_down,
            thread_handle: Some(thread_handle),
            cap_size: init.cap_size,
            client_offsets: init.client_offsets,
            current_gen: 1,
            packer,
            scratch_roi: Vec::new(),
            cached_bgr: Vec::new(),
            cached_w: 0,
            cached_h: 0,
            last_ts: 0,
            last_capture_at: None,
            os_min_interval_ms: min_update_interval_ms,
            os_interval_desc,
        })
    }

    pub fn set_min_update_interval_ms(&self, min_update_interval_ms: u32) {
        let prev = self
            .shared
            .telemetry
            .interval_ms
            .swap(min_update_interval_ms, Ordering::Relaxed);
        if prev != min_update_interval_ms
            && self
                .shared
                .telemetry
                .min_interval_supported
                .load(Ordering::Acquire)
        {
            let target_ms = effective_min_interval_ms(min_update_interval_ms);
            let _ = self.dispatch_on_wgc_thread(move |owner| {
                let _ = owner
                    .session
                    .SetMinUpdateInterval(Duration::from_millis(target_ms as u64).into());
            });
        }
    }

    pub fn needs_reopen_for_interval(&self, interval_ms: u32) -> bool {
        !self
            .shared
            .telemetry
            .min_interval_supported
            .load(Ordering::Acquire)
            && self.os_min_interval_ms > 0
            && interval_ms < self.os_min_interval_ms
    }

    pub fn diagnostics_summary(&self) -> String {
        let adapter = self
            .shared
            .telemetry
            .adapter_name
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "Unknown".to_string());
        let bgr_mode = if self.shared.telemetry.gpu_bgr_cs.load(Ordering::Relaxed) {
            "GPU-CS-BGR24"
        } else {
            "CPU-BGR24"
        };
        format!(
            "self-wgc(Pool=2, {bgr_mode}), adapter={}, min_interval={} (soft={}ms), arrivals={}, copied={}, recreates={}, throttled_skips={}",
            adapter,
            self.os_interval_desc,
            self.shared.telemetry.interval_ms.load(Ordering::Relaxed),
            self.shared.telemetry.arrivals.load(Ordering::Relaxed),
            self.shared.telemetry.copied_frames.load(Ordering::Relaxed),
            self.shared.telemetry.recreates.load(Ordering::Relaxed),
            self.shared.telemetry.throttled_skips.load(Ordering::Relaxed),
        )
    }

    /// Captures cropped client frame packed directly as tight `BGR24` bytes.
    pub fn capture_bgr(&mut self, pool: &FrameBufferPool) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        self.capture_region_bgr(pool, None)
    }

    /// Captures a cropped client sub-rectangle `(x, y, width, height)` packed directly as `BGR24`.
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
        self.capture_region_bgr(pool, Some((x, y, width, height)))
    }

    fn capture_region_bgr(
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

        let soft_interval = self.shared.telemetry.interval_ms.load(Ordering::Relaxed);
        let now = Instant::now();
        if roi.is_none()
            && self.cached_w > 0
            && should_throttle_capture(self.last_capture_at, true, soft_interval, now)
        {
            self.shared
                .telemetry
                .throttled_skips
                .fetch_add(1, Ordering::Relaxed);
            return Ok(self.checkout_cached_full(pool));
        }

        let deadline = Instant::now() + FRAME_WAIT_TIMEOUT;
        let mut waited_recreate = false;

        loop {
            let need_wait = (roi.is_none() && self.cached_w == 0) || waited_recreate;
            let frame_opt = self.take_produced_frame(need_wait || roi.is_some() && self.cached_w == 0, deadline)?;

            let Some((frame, frame_gen)) = frame_opt else {
                if let Some((rx, ry, rw, rh)) = roi {
                    if self.cached_w > 0 {
                        return Ok(self.checkout_cached_roi(pool, rx, ry, rw, rh));
                    }
                    return Err(CaptureError::FrameNotReady);
                }
                if self.cached_w > 0 {
                    self.last_capture_at = Some(now);
                    return Ok(self.checkout_cached_full(pool));
                }
                return Err(CaptureError::FrameNotReady);
            };

            if frame_gen != self.current_gen {
                let _ = frame.Close();
                if Instant::now() < deadline {
                    waited_recreate = true;
                    continue;
                }
                return Err(CaptureError::FrameNotReady);
            }

            let Ok(content_size) = frame.ContentSize() else {
                let _ = frame.Close();
                return Err(CaptureError::FrameNotReady);
            };
            if content_size.Width <= 0 || content_size.Height <= 0 {
                let _ = frame.Close();
                return Err(CaptureError::FrameNotReady);
            }

            if content_size.Width != self.cap_size.Width
                || content_size.Height != self.cap_size.Height
            {
                let _ = frame.Close();
                if unsafe { IsIconic(self.hwnd).as_bool() } {
                    return Err(CaptureError::WindowMinimized);
                }
                self.recreate_pool_for_size(content_size)?;
                waited_recreate = true;
                if Instant::now() < deadline {
                    continue;
                }
                return Err(CaptureError::FrameNotReady);
            }

            let frame_ts = frame.SystemRelativeTime().map(|t| t.Duration).unwrap_or(0);
            if frame_ts != 0 && frame_ts == self.last_ts && self.cached_w > 0 {
                let _ = frame.Close();
                if let Some((rx, ry, rw, rh)) = roi {
                    return Ok(self.checkout_cached_roi(pool, rx, ry, rw, rh));
                }
                self.last_capture_at = Some(now);
                return Ok(self.checkout_cached_full(pool));
            }

            // Stage 1: GPU compute pack `src_tex -> staging`, flush queue, then CLOSE WGC frame immediately!
            let staged = match extract_frame_texture(&frame) {
                Ok(src_tex) => {
                    let mut src_desc = D3D11_TEXTURE2D_DESC::default();
                    unsafe { src_tex.GetDesc(&mut src_desc) };
                    if src_desc.Width == 0 || src_desc.Height == 0 {
                        Err(CaptureError::FrameNotReady)
                    } else {
                        let (left, top, client_w, client_h) =
                            clamp_client_crop(self.client_offsets, src_desc.Width, src_desc.Height);
                        let (crop_x, crop_y, out_w, out_h) = if let Some((rx, ry, rw, rh)) = roi {
                            let cx = left + rx.min(client_w.saturating_sub(1));
                            let cy = top + ry.min(client_h.saturating_sub(1));
                            let cw = rw.min(left + client_w - cx).max(1);
                            let ch = rh.min(top + client_h - cy).max(1);
                            (cx, cy, cw, ch)
                        } else {
                            (left, top, client_w, client_h)
                        };
                        self.packer
                            .stage_cropped_bgr24(
                                &self.d3d_device,
                                &self.d3d_context,
                                &src_tex,
                                src_desc.Format,
                                crop_x,
                                crop_y,
                                out_w,
                                out_h,
                            )
                            .map(|used_cs| (used_cs, out_w, out_h))
                    }
                }
                Err(e) => Err(e),
            };
            let _ = frame.Close();
            drop(frame);

            let (used_cs, out_w, out_h) = staged?;
            self.last_capture_at = Some(now);
            self.shared
                .telemetry
                .copied_frames
                .fetch_add(1, Ordering::Relaxed);

            if roi.is_some() {
                self.packer.map_staged_bgr24(
                    &self.d3d_context,
                    used_cs,
                    out_w,
                    out_h,
                    &mut self.scratch_roi,
                )?;
                let mut out = pool.checkout(self.scratch_roi.len());
                out.as_mut_slice().copy_from_slice(&self.scratch_roi);
                return Ok((out_w, out_h, out));
            }

            self.packer.map_staged_bgr24(
                &self.d3d_context,
                used_cs,
                out_w,
                out_h,
                &mut self.cached_bgr,
            )?;
            self.cached_w = out_w;
            self.cached_h = out_h;
            self.last_ts = frame_ts;
            return Ok(self.checkout_cached_full(pool));
        }
    }

    fn take_produced_frame(
        &self,
        wait_if_empty: bool,
        deadline: Instant,
    ) -> Result<Option<(Direct3D11CaptureFrame, u64)>, CaptureError> {
        let mut guard = self.shared.mailbox.lock().unwrap();
        if !guard.is_capturing {
            return Err(CaptureError::InvalidWindow);
        }
        if guard.produced.is_none() && wait_if_empty {
            while guard.is_capturing && guard.produced.is_none() {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                let (g, _) = self
                    .shared
                    .cv
                    .wait_timeout(guard, deadline.saturating_duration_since(now))
                    .unwrap();
                guard = g;
            }
            if !guard.is_capturing {
                return Err(CaptureError::InvalidWindow);
            }
        }
        Ok(guard.produced.take())
    }

    fn recreate_pool_for_size(&mut self, new_size: SizeInt32) -> Result<(), CaptureError> {
        self.shared.telemetry.recreates.fetch_add(1, Ordering::Relaxed);
        let shared = self.shared.clone();
        let (done_tx, done_rx) =
            mpsc::channel::<Result<(SizeInt32, ClientAreaOffsets, u64), CaptureError>>();

        self.dispatch_on_wgc_thread(move |owner| {
            let res = (|| {
                owner
                    .frame_pool
                    .Recreate(
                        &owner.winrt_device,
                        DirectXPixelFormat::B8G8R8A8UIntNormalized,
                        CAPTURE_POOL_SIZE,
                        new_size,
                    )
                    .map_err(|e| CaptureError::Api(format!("WGC FramePool::Recreate 失败: {e}")))?;

                let mut guard = shared.mailbox.lock().unwrap();
                guard.pool_generation = guard.pool_generation.wrapping_add(1);
                let pool_gen = guard.pool_generation;
                if let Some((old, _)) = guard.produced.take() {
                    let _ = old.Close();
                }
                drop(guard);

                owner.cap_size = new_size;
                owner.client_offsets = calculate_client_area(owner.hwnd, new_size);
                Ok((owner.cap_size, owner.client_offsets, pool_gen))
            })();
            let _ = done_tx.send(res);
        })?;

        let (cap_size, offsets, new_gen) = done_rx
            .recv_timeout(CTRL_WAIT_TIMEOUT)
            .map_err(|_| CaptureError::Api("等待 WGC Recreate 超时".into()))??;

        self.cap_size = cap_size;
        self.client_offsets = offsets;
        self.current_gen = new_gen;
        self.packer.reset_buffers();
        self.cached_bgr.clear();
        self.cached_w = 0;
        self.cached_h = 0;
        self.last_ts = 0;
        Ok(())
    }

    fn checkout_cached_full(&self, pool: &FrameBufferPool) -> (u32, u32, PooledBuffer) {
        let mut out = pool.checkout(self.cached_bgr.len());
        out.as_mut_slice().copy_from_slice(&self.cached_bgr);
        (self.cached_w, self.cached_h, out)
    }

    fn checkout_cached_roi(
        &self,
        pool: &FrameBufferPool,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> (u32, u32, PooledBuffer) {
        let rx = x.min(self.cached_w.saturating_sub(1));
        let ry = y.min(self.cached_h.saturating_sub(1));
        let rw = w.min(self.cached_w - rx).max(1);
        let rh = h.min(self.cached_h - ry).max(1);
        let row_bytes = (rw as usize) * 3;
        let src_stride = (self.cached_w as usize) * 3;
        let mut out = pool.checkout(row_bytes * (rh as usize));
        let dst = out.as_mut_slice();
        for row in 0..(rh as usize) {
            let s_off = ((ry as usize) + row) * src_stride + (rx as usize) * 3;
            let d_off = row * row_bytes;
            dst[d_off..d_off + row_bytes].copy_from_slice(&self.cached_bgr[s_off..s_off + row_bytes]);
        }
        (rw, rh, out)
    }

    fn dispatch_on_wgc_thread<F>(&self, f: F) -> Result<(), CaptureError>
    where
        F: FnOnce(&mut WgcSessionOwner) + Send + 'static,
    {
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(CaptureError::InvalidWindow);
        }
        self.task_queue.lock().unwrap().push_back(Box::new(f));
        unsafe {
            PostThreadMessageW(self.wgc_thread_id, WM_WGC_EXEC, WPARAM(0), LPARAM(0))
                .map_err(|e| CaptureError::Api(format!("PostThreadMessageW(WM_WGC_EXEC) 失败: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for WgcCapturer {
    fn drop(&mut self) {
        self.shutting_down.store(true, Ordering::Release);
        {
            let mut guard = self.shared.mailbox.lock().unwrap();
            guard.is_capturing = false;
            if let Some((old, _)) = guard.produced.take() {
                let _ = old.Close();
            }
            self.shared.cv.notify_all();
        }
        if self.wgc_thread_id != 0 {
            unsafe {
                let _ = PostThreadMessageW(self.wgc_thread_id, WM_WGC_SHUTDOWN, WPARAM(0), LPARAM(0));
            }
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        self.packer.reset_buffers();
    }
}

fn run_wgc_thread(
    hwnd_raw: isize,
    min_update_interval_ms: u32,
    shared: Arc<WgcShared>,
    task_queue: Arc<Mutex<VecDeque<WgcTaskFn>>>,
    shutting_down: Arc<AtomicBool>,
    init_tx: mpsc::Sender<Result<WgcInitPayload, CaptureError>>,
) {
    let _prev_dpi: DPI_AWARENESS_CONTEXT =
        unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    let _ro = match RoInitGuard::init_mta() {
        Ok(g) => g,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };

    let mut msg = MSG::default();
    unsafe {
        let _ = PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE);
    }

    let mut owner = match setup_wgc_session(hwnd_raw, min_update_interval_ms, &shared) {
        Ok((owner, d3d_device, d3d_context)) => {
            let payload = WgcInitPayload {
                thread_id: unsafe { GetCurrentThreadId() },
                d3d_device,
                d3d_context,
                cap_size: owner.cap_size,
                client_offsets: owner.client_offsets,
            };
            let _ = init_tx.send(Ok(payload));
            Some(owner)
        }
        Err(e) => {
            let _ = init_tx.send(Err(e));
            None
        }
    };

    if owner.is_some() {
        while !shutting_down.load(Ordering::Acquire) {
            let ret = unsafe { GetMessageW(&mut msg, None, 0, 0).0 };
            if ret <= 0 {
                break;
            }
            if msg.hwnd.is_invalid() && msg.message == WM_WGC_SHUTDOWN {
                break;
            }
            if msg.hwnd.is_invalid() && msg.message == WM_WGC_EXEC {
                drain_wgc_tasks(&task_queue, owner.as_mut());
                continue;
            }
            unsafe {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            drain_wgc_tasks(&task_queue, owner.as_mut());
        }
    }

    task_queue.lock().unwrap().clear();
    if let Some(o) = owner.take() {
        o.close();
    }
}

fn drain_wgc_tasks(
    task_queue: &Mutex<VecDeque<WgcTaskFn>>,
    mut owner: Option<&mut WgcSessionOwner>,
) {
    loop {
        let Some(task) = task_queue.lock().unwrap().pop_front() else {
            break;
        };
        if let Some(ref mut o) = owner {
            task(o);
        }
    }
}

fn setup_wgc_session(
    hwnd_raw: isize,
    min_update_interval_ms: u32,
    shared: &Arc<WgcShared>,
) -> Result<(WgcSessionOwner, ID3D11Device, ID3D11DeviceContext), CaptureError> {
    if !GraphicsCaptureSession::IsSupported().unwrap_or(false) {
        return Err(CaptureError::Api("当前系统不支持 Windows Graphics Capture".into()));
    }

    let (d3d_device, d3d_context) = create_d3d11_device()?;
    if let Ok(mt) = d3d_device.cast::<ID3D11Multithread>() {
        unsafe {
            let _ = mt.SetMultithreadProtected(true);
        }
    }
    if let Ok(dxgi_dev) = d3d_device.cast::<IDXGIDevice>() {
        let name = query_adapter_name(&dxgi_dev);
        if let Ok(mut guard) = shared.telemetry.adapter_name.lock() {
            *guard = name;
        }
    }

    let dxgi_dev: IDXGIDevice = d3d_device
        .cast()
        .map_err(|e| CaptureError::Api(format!("查询 IDXGIDevice 失败: {e}")))?;
    let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_dev) }
        .map_err(|e| CaptureError::Api(format!("CreateDirect3D11DeviceFromDXGIDevice 失败: {e}")))?;
    let winrt_device: IDirect3DDevice = inspectable
        .cast()
        .map_err(|e| CaptureError::Api(format!("转换 WinRT IDirect3DDevice 失败: {e}")))?;

    let hwnd = HWND(hwnd_raw as *mut _);
    let interop = windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
        .map_err(|e| CaptureError::Api(format!("获取 IGraphicsCaptureItemInterop 失败: {e}")))?;
    let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(hwnd) }
        .map_err(|e| CaptureError::Api(format!("CreateForWindow 失败: {e}")))?;

    let cap_size = item
        .Size()
        .map_err(|e| CaptureError::Api(format!("读取 CaptureItem::Size 失败: {e}")))?;
    if cap_size.Width <= 0 || cap_size.Height <= 0 {
        return Err(CaptureError::Api(format!(
            "窗口捕获尺寸无效: {}x{}",
            cap_size.Width, cap_size.Height
        )));
    }
    let client_offsets = calculate_client_area(hwnd, cap_size);

    let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        &winrt_device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        CAPTURE_POOL_SIZE,
        cap_size,
    )
    .map_err(|e| CaptureError::Api(format!("CreateFreeThreaded 失败: {e}")))?;

    let session = frame_pool
        .CreateCaptureSession(&item)
        .map_err(|e| CaptureError::Api(format!("CreateCaptureSession 失败: {e}")))?;

    let _ = session.SetIsCursorCaptureEnabled(false);
    let _ = session.SetIsBorderRequired(false);

    let target_ms = effective_min_interval_ms(min_update_interval_ms);
    if session
        .SetMinUpdateInterval(Duration::from_millis(target_ms as u64).into())
        .is_ok()
    {
        shared
            .telemetry
            .min_interval_supported
            .store(true, Ordering::Release);
    }

    let shared_closed = shared.clone();
    let closed_token = item
        .Closed(&TypedEventHandler::<GraphicsCaptureItem, IInspectable>::new(
            move |_, _| {
                let mut guard = shared_closed.mailbox.lock().unwrap();
                guard.is_capturing = false;
                if let Some((old, _)) = guard.produced.take() {
                    let _ = old.Close();
                }
                shared_closed.cv.notify_all();
                Ok(())
            },
        ))
        .map_err(|e| CaptureError::Api(format!("注册 Item.Closed 失败: {e}")))?;

    // Sunshine low-latency FrameArrived: drain pool to newest frame, drop unconsumed old frame, ZERO D3D!
    let shared_arrived = shared.clone();
    let frame_arrived_token = frame_pool
        .FrameArrived(&TypedEventHandler::<
            Direct3D11CaptureFramePool,
            IInspectable,
        >::new(move |sender, _| {
            let Some(pool) = sender.as_ref() else {
                return Ok(());
            };
            let mut latest: Option<Direct3D11CaptureFrame> = None;
            while let Ok(next) = pool.TryGetNextFrame() {
                if let Some(prev) = latest.replace(next) {
                    let _ = prev.Close();
                }
            }
            let Some(frame) = latest else {
                return Ok(());
            };

            let replaced = {
                let mut guard = shared_arrived.mailbox.lock().unwrap();
                if !guard.is_capturing {
                    let _ = frame.Close();
                    return Ok(());
                }
                let pool_gen = guard.pool_generation;
                let old = guard.produced.replace((frame, pool_gen));
                shared_arrived.telemetry.arrivals.fetch_add(1, Ordering::Relaxed);
                shared_arrived.cv.notify_all();
                old
            };
            if let Some((old_frame, _)) = replaced {
                let _ = old_frame.Close();
            }
            Ok(())
        }))
        .map_err(|e| CaptureError::Api(format!("注册 FrameArrived 失败: {e}")))?;

    session
        .StartCapture()
        .map_err(|e| CaptureError::Api(format!("StartCapture 失败: {e}")))?;

    Ok((
        WgcSessionOwner {
            hwnd,
            winrt_device,
            item,
            frame_pool,
            session,
            cap_size,
            client_offsets,
            closed_token,
            frame_arrived_token,
        },
        d3d_device,
        d3d_context,
    ))
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

fn extract_frame_texture(frame: &Direct3D11CaptureFrame) -> Result<ID3D11Texture2D, CaptureError> {
    let surface = frame
        .Surface()
        .map_err(|e| CaptureError::Api(format!("获取 Frame::Surface 失败: {e}")))?;
    let access: IDirect3DDxgiInterfaceAccess = surface
        .cast()
        .map_err(|e| CaptureError::Api(format!("查询 IDirect3DDxgiInterfaceAccess 失败: {e}")))?;
    unsafe { access.GetInterface::<ID3D11Texture2D>() }
        .map_err(|e| CaptureError::Api(format!("提取 ID3D11Texture2D 失败: {e}")))
}

fn calculate_client_area(hwnd: HWND, cap_size: SizeInt32) -> ClientAreaOffsets {
    if unsafe { IsIconic(hwnd).as_bool() } {
        return ClientAreaOffsets::default();
    }
    let mut client_rect = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut client_rect) }.is_err() {
        return ClientAreaOffsets::default();
    }
    let mut pts = [
        POINT { x: client_rect.left, y: client_rect.top },
        POINT { x: client_rect.right, y: client_rect.bottom },
    ];
    unsafe {
        if MapWindowPoints(Some(hwnd), None, &mut pts) == 0 {
            return ClientAreaOffsets::default();
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
        && (cap_size.Width == frame_bounds.right - frame_bounds.left
            || cap_size.Height == frame_bounds.bottom - frame_bounds.top)
    {
        frame_bounds
    } else if win_ok
        && (cap_size.Width == window_rect.right - window_rect.left
            || cap_size.Height == window_rect.bottom - window_rect.top)
    {
        window_rect
    } else if dwm_ok {
        frame_bounds
    } else if win_ok {
        window_rect
    } else {
        return ClientAreaOffsets::default();
    };

    ClientAreaOffsets {
        left: (screen_client.left - chosen.left).max(0) as u32,
        top: (screen_client.top - chosen.top).max(0) as u32,
        right: (chosen.right - screen_client.right).max(0) as u32,
        bottom: (chosen.bottom - screen_client.bottom).max(0) as u32,
    }
}

fn clamp_client_crop(offsets: ClientAreaOffsets, src_w: u32, src_h: u32) -> (u32, u32, u32, u32) {
    let left = offsets.left.min(src_w.saturating_sub(1));
    let top = offsets.top.min(src_h.saturating_sub(1));
    let max_w = src_w - left;
    let max_h = src_h - top;
    let out_w = max_w.saturating_sub(offsets.right).max(1);
    let out_h = max_h.saturating_sub(offsets.bottom).max(1);
    (left, top, out_w, out_h)
}

fn effective_min_interval_ms(requested_ms: u32) -> u32 {
    if requested_ms == 0 { 16 } else { requested_ms }
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

    #[test]
    fn client_crop_clamps_safely_on_transient_shrink() {
        let offsets = ClientAreaOffsets {
            left: 8,
            top: 31,
            right: 8,
            bottom: 8,
        };
        assert_eq!(clamp_client_crop(offsets, 1936, 1119), (8, 31, 1920, 1080));
        assert_eq!(clamp_client_crop(offsets, 20, 20), (8, 19, 4, 1));
    }
}
