use std::num::NonZeroIsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bgi_control::ControlClient;
use bgi_ui::{BgiAppUi, UiShellAction};
use egui::{
    Event, Key, Modifiers, MouseWheelSource, MouseWheelUnit, PointerButton, Pos2, RawInput,
    ViewportId,
};
use egui_wgpu::wgpu;
use egui_wgpu::wgpu::rwh::{
    RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock};
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetKeyState, ReleaseCapture, SetCapture, VK_CONTROL, VK_MENU, VK_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CREATESTRUCTW, CS_HREDRAW, CS_OWNDC, CS_VREDRAW, CW_USEDEFAULT, CreateWindowExW,
    DefWindowProcW, DestroyWindow, GWLP_USERDATA, GetClientRect, GetWindowLongPtrW, IDC_ARROW,
    KillTimer, LoadCursorW, PostMessageW, RegisterClassW, SW_SHOW, SetTimer, SetWindowLongPtrW,
    ShowWindow, WM_APP, WM_CHAR, WM_CLOSE, WM_DESTROY, WM_DPICHANGED, WM_ERASEBKGND, WM_KEYDOWN,
    WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL,
    WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_NCCREATE, WM_PAINT, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SIZE,
    WM_SYSKEYDOWN, WM_SYSKEYUP, WM_TIMER, WNDCLASSW, WS_OVERLAPPEDWINDOW,
};
use windows::core::{PCWSTR, w};

use crate::tray::TrayHost;

pub const WM_APP_REPAINT: u32 = WM_APP + 20;
pub const WM_APP_CLOSE_UI: u32 = WM_APP + 21;
pub const WM_APP_QUIT_APP: u32 = WM_APP + 22;
const REPAINT_TIMER_ID: usize = 1;

static CLASS_REGISTERED: std::sync::Once = std::sync::Once::new();

struct WindowInputState {
    events: Vec<Event>,
    width_px: u32,
    height_px: u32,
    dpi_scale: f32,
    needs_resize: bool,
    needs_repaint: bool,
    close_requested: bool,
}

impl WindowInputState {
    fn logical_pos(&self, lparam: LPARAM) -> Pos2 {
        let x = (lparam.0 & 0xFFFF) as i16 as f32;
        let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as f32;
        Pos2::new(x / self.dpi_scale, y / self.dpi_scale)
    }
}

pub struct UiHost {
    hwnd: HWND,
    input_state: Box<WindowInputState>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    egui_ctx: egui::Context,
    app_ui: BgiAppUi,
    start_time: Instant,
    tray_hwnd: HWND,
}

impl UiHost {
    pub fn open(tray: &TrayHost, client: &ControlClient) -> Result<Self, String> {
        let hinstance: HINSTANCE = unsafe { GetModuleHandleW(None) }
            .map_err(|e| e.to_string())?
            .into();
        let class_name: PCWSTR = w!("BetterGI_RS_MainWindow");

        CLASS_REGISTERED.call_once(|| {
            let wc = WNDCLASSW {
                style: CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
                lpfnWndProc: Some(ui_wnd_proc),
                hInstance: hinstance,
                hCursor: unsafe { LoadCursorW(None, IDC_ARROW).unwrap_or_default() },
                hIcon: tray.hicon,
                lpszClassName: class_name,
                ..Default::default()
            };
            unsafe {
                RegisterClassW(&wc);
            }
        });

        let mut input_state = Box::new(WindowInputState {
            events: Vec::with_capacity(32),
            width_px: 1120,
            height_px: 740,
            dpi_scale: 1.0,
            needs_resize: false,
            needs_repaint: true,
            close_requested: false,
        });
        let input_ptr: *mut WindowInputState = &mut *input_state;

        let hwnd = unsafe {
            CreateWindowExW(
                Default::default(),
                class_name,
                w!("BetterGI-RS（基础寻路 MVP · 关窗退托盘释放渲染上下文）"),
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                1120,
                740,
                None,
                None,
                Some(hinstance),
                Some(input_ptr as *const _),
            )
        }
        .map_err(|e| format!("创建主窗口失败: {e}"))?;

        tray.apply_window_icons(hwnd);

        let dpi = unsafe { GetDpiForWindow(hwnd) };
        input_state.dpi_scale = if dpi == 0 { 1.0 } else { (dpi as f32) / 96.0 };
        let mut rect = RECT::default();
        unsafe {
            let _ = GetClientRect(hwnd, &mut rect);
        }
        input_state.width_px = (rect.right - rect.left).max(1) as u32;
        input_state.height_px = (rect.bottom - rect.top).max(1) as u32;

        let raw_window =
            Win32WindowHandle::new(NonZeroIsize::new(hwnd.0 as isize).ok_or("HWND 为空")?);
        let make_target = || wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(RawDisplayHandle::Windows(WindowsDisplayHandle::new())),
            raw_window_handle: RawWindowHandle::Win32(raw_window),
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = unsafe { instance.create_surface_unsafe(make_target()) }
            .map_err(|e| format!("创建 wgpu Surface 失败: {e}"))?;

        let init_rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| e.to_string())?;
        let adapter = init_rt
            .block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            }))
            .map_err(|e| format!("找不到兼容 GPU/WARP 适配器: {e}"))?;
        let adapter_info = adapter.get_info();
        tracing::info!(
            "egui-wgpu 渲染后端: {:?}, 适配器: {}",
            adapter_info.backend,
            adapter_info.name
        );

        let (device, queue) = init_rt
            .block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|e| format!("创建 wgpu Device 失败: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: input_state.width_px,
            height: input_state.height_px,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new(&device, surface_format, RendererOptions::default());
        let egui_ctx = egui::Context::default();
        BgiAppUi::configure_context(&egui_ctx);

        let hwnd_raw = hwnd.0 as isize;
        client.waker_slot().set(Some(Arc::new(move || unsafe {
            let _ = PostMessageW(
                Some(HWND(hwnd_raw as *mut _)),
                WM_APP_REPAINT,
                WPARAM(0),
                LPARAM(0),
            );
        })));
        egui_ctx.set_request_repaint_callback(move |info| unsafe {
            let ms = info.delay.as_millis() as u32;
            let target = HWND(hwnd_raw as *mut _);
            if ms == 0 {
                let _ = PostMessageW(Some(target), WM_APP_REPAINT, WPARAM(0), LPARAM(0));
            } else {
                let _ = SetTimer(Some(target), REPAINT_TIMER_ID, ms.max(10), None);
            }
        });

        unsafe {
            let _ = ShowWindow(hwnd, SW_SHOW);
            let _ = PostMessageW(Some(hwnd), WM_APP_REPAINT, WPARAM(0), LPARAM(0));
        }

        Ok(Self {
            hwnd,
            input_state,
            _instance: instance,
            surface,
            device,
            queue,
            surface_config,
            renderer,
            egui_ctx,
            app_ui: BgiAppUi::new(),
            start_time: Instant::now(),
            tray_hwnd: tray.hwnd,
        })
    }

    fn discard_unrendered_textures(&mut self, delta: &mut egui::TexturesDelta) {
        for id in std::mem::take(&mut delta.free) {
            self.renderer.free_texture(&id);
        }
        delta.clear();
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_REPAINT, WPARAM(0), LPARAM(0));
        }
    }

    pub fn should_repaint(&self) -> bool {
        self.input_state.needs_repaint
            || self.input_state.needs_resize
            || !self.input_state.events.is_empty()
    }

    pub fn close_requested(&self) -> bool {
        self.input_state.close_requested
    }

    pub fn render_if_needed(&mut self, client: &mut ControlClient) {
        if client.has_snapshot_update() {
            self.input_state.needs_repaint = true;
        }
        if !self.should_repaint() {
            return;
        }
        self.input_state.needs_repaint = false;

        if self.input_state.needs_resize {
            self.input_state.needs_resize = false;
            let w = self.input_state.width_px.max(1);
            let h = self.input_state.height_px.max(1);
            self.surface_config.width = w;
            self.surface_config.height = h;
            self.surface.configure(&self.device, &self.surface_config);
        }

        let ppp = self.input_state.dpi_scale;
        let w_pt = (self.surface_config.width as f32) / ppp;
        let h_pt = (self.surface_config.height as f32) / ppp;
        let events = std::mem::take(&mut self.input_state.events);

        let mut viewports = egui::ViewportIdMap::default();
        viewports.insert(
            ViewportId::ROOT,
            egui::ViewportInfo {
                native_pixels_per_point: Some(ppp),
                focused: Some(true),
                ..Default::default()
            },
        );

        let raw_input = RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(w_pt, h_pt),
            )),
            viewport_id: ViewportId::ROOT,
            viewports,
            time: Some(self.start_time.elapsed().as_secs_f64()),
            predicted_dt: 1.0 / 60.0,
            events,
            ..Default::default()
        };

        let mut shell_action = UiShellAction::None;
        let mut full_output = self.egui_ctx.run_ui(raw_input, |ui| {
            shell_action = self.app_ui.render(ui, client);
        });

        for cmd in &full_output.platform_output.commands {
            if let egui::OutputCommand::CopyText(text) = cmd {
                copy_to_clipboard(self.hwnd, text);
            }
        }

        if let Some(vp) = full_output.viewport_output.get(&ViewportId::ROOT) {
            if vp.repaint_delay == Duration::ZERO {
                unsafe {
                    let _ = PostMessageW(Some(self.hwnd), WM_APP_REPAINT, WPARAM(0), LPARAM(0));
                }
            } else if vp.repaint_delay < Duration::from_secs(60) {
                let ms = (vp.repaint_delay.as_millis() as u32).max(10);
                unsafe {
                    let _ = SetTimer(Some(self.hwnd), REPAINT_TIMER_ID, ms, None);
                }
            }
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                self.discard_unrendered_textures(&mut full_output.textures_delta);
                return;
            }
            _ => {
                self.discard_unrendered_textures(&mut full_output.textures_delta);
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let clipped = self.egui_ctx.tessellate(full_output.shapes, ppp);
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: ppp,
        };

        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("egui_encoder"),
            });
        let user_cmds = self.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped,
            &screen_desc,
        );

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.09,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.renderer.render(&mut pass, &clipped, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        full_output.textures_delta.clear();

        let all_cmds = user_cmds
            .into_iter()
            .chain(std::iter::once(encoder.finish()));
        self.queue.submit(all_cmds);
        self.queue.present(frame);

        match shell_action {
            UiShellAction::None => {}
            UiShellAction::HideToTray => {
                self.input_state.close_requested = true;
                unsafe {
                    let _ =
                        PostMessageW(Some(self.tray_hwnd), WM_APP_CLOSE_UI, WPARAM(0), LPARAM(0));
                }
            }
            UiShellAction::QuitApp => unsafe {
                let _ = PostMessageW(Some(self.tray_hwnd), WM_APP_QUIT_APP, WPARAM(0), LPARAM(0));
            },
        }
    }

    pub fn teardown(self, client: &ControlClient) {
        client.waker_slot().set(None);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let hwnd = self.hwnd;
        let _input_state = self.input_state;
        drop(self.renderer);
        drop(self.surface);
        drop(self.device);
        drop(self.queue);
        drop(self._instance);
        unsafe {
            let _ = KillTimer(Some(hwnd), REPAINT_TIMER_ID);
            let _ = DestroyWindow(hwnd);
        }
        drop(_input_state);
    }
}

unsafe extern "system" fn ui_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_NCCREATE {
        let cs = unsafe { &*(lparam.0 as *const CREATESTRUCTW) };
        unsafe {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, cs.lpCreateParams as isize);
        }
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    }

    let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowInputState;
    if ptr.is_null() {
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    }
    let state = unsafe { &mut *ptr };

    match msg {
        WM_CLOSE => {
            state.close_requested = true;
            return LRESULT(0);
        }
        WM_DESTROY => {
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            }
            return LRESULT(0);
        }
        WM_ERASEBKGND => return LRESULT(1),
        WM_PAINT | WM_APP_REPAINT => {
            state.needs_repaint = true;
        }
        WM_TIMER => {
            if wparam.0 == REPAINT_TIMER_ID {
                unsafe {
                    let _ = KillTimer(Some(hwnd), REPAINT_TIMER_ID);
                }
                state.needs_repaint = true;
            }
            return LRESULT(0);
        }
        WM_SIZE => {
            let w = (lparam.0 & 0xFFFF) as u32;
            let h = ((lparam.0 >> 16) & 0xFFFF) as u32;
            if w > 0 && h > 0 {
                state.width_px = w;
                state.height_px = h;
                state.needs_resize = true;
                state.needs_repaint = true;
            }
            return LRESULT(0);
        }
        WM_DPICHANGED => {
            let new_dpi = (wparam.0 & 0xFFFF) as f32;
            if new_dpi > 0.0 {
                state.dpi_scale = new_dpi / 96.0;
                state.needs_repaint = true;
            }
            return LRESULT(0);
        }
        WM_MOUSEMOVE => {
            state
                .events
                .push(Event::PointerMoved(state.logical_pos(lparam)));
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_LBUTTONDOWN => {
            unsafe { SetCapture(hwnd) };
            state.events.push(Event::PointerButton {
                pos: state.logical_pos(lparam),
                button: PointerButton::Primary,
                pressed: true,
                modifiers: current_modifiers(),
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_LBUTTONUP => {
            unsafe {
                let _ = ReleaseCapture();
            }
            state.events.push(Event::PointerButton {
                pos: state.logical_pos(lparam),
                button: PointerButton::Primary,
                pressed: false,
                modifiers: current_modifiers(),
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_RBUTTONDOWN => {
            state.events.push(Event::PointerButton {
                pos: state.logical_pos(lparam),
                button: PointerButton::Secondary,
                pressed: true,
                modifiers: current_modifiers(),
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_RBUTTONUP => {
            state.events.push(Event::PointerButton {
                pos: state.logical_pos(lparam),
                button: PointerButton::Secondary,
                pressed: false,
                modifiers: current_modifiers(),
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_MBUTTONDOWN => {
            state.events.push(Event::PointerButton {
                pos: state.logical_pos(lparam),
                button: PointerButton::Middle,
                pressed: true,
                modifiers: current_modifiers(),
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_MBUTTONUP => {
            state.events.push(Event::PointerButton {
                pos: state.logical_pos(lparam),
                button: PointerButton::Middle,
                pressed: false,
                modifiers: current_modifiers(),
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_MOUSEWHEEL => {
            let delta = ((wparam.0 >> 16) as i16 as f32) / 120.0 * 28.0;
            state.events.push(Event::MouseWheel {
                unit: MouseWheelUnit::Point,
                delta: egui::vec2(0.0, delta),
                modifiers: current_modifiers(),
                phase: egui::TouchPhase::Move,
                source: MouseWheelSource::Unknown,
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_MOUSEHWHEEL => {
            let delta = ((wparam.0 >> 16) as i16 as f32) / 120.0 * 28.0;
            state.events.push(Event::MouseWheel {
                unit: MouseWheelUnit::Point,
                delta: egui::vec2(-delta, 0.0),
                modifiers: current_modifiers(),
                phase: egui::TouchPhase::Move,
                source: MouseWheelSource::Unknown,
            });
            state.needs_repaint = true;
            return LRESULT(0);
        }
        WM_CHAR => {
            if let Some(ch) = char::from_u32(wparam.0 as u32)
                && !ch.is_control()
            {
                state.events.push(Event::Text(ch.to_string()));
                state.needs_repaint = true;
            }
            return LRESULT(0);
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            if let Some(key) = map_vk_to_egui(wparam.0 as u16) {
                state.events.push(Event::Key {
                    key,
                    physical_key: Some(key),
                    pressed: true,
                    repeat: (lparam.0 & (1 << 30)) != 0,
                    modifiers: current_modifiers(),
                });
                state.needs_repaint = true;
            }
            return LRESULT(0);
        }
        WM_KEYUP | WM_SYSKEYUP => {
            if let Some(key) = map_vk_to_egui(wparam.0 as u16) {
                state.events.push(Event::Key {
                    key,
                    physical_key: Some(key),
                    pressed: false,
                    repeat: false,
                    modifiers: current_modifiers(),
                });
                state.needs_repaint = true;
            }
            return LRESULT(0);
        }
        _ => {}
    }

    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

fn current_modifiers() -> Modifiers {
    let ctrl = unsafe { GetKeyState(VK_CONTROL.0 as i32) } < 0;
    let shift = unsafe { GetKeyState(VK_SHIFT.0 as i32) } < 0;
    let alt = unsafe { GetKeyState(VK_MENU.0 as i32) } < 0;
    Modifiers {
        alt,
        ctrl,
        shift,
        mac_cmd: false,
        command: ctrl,
    }
}

fn map_vk_to_egui(vk: u16) -> Option<Key> {
    Some(match vk {
        0x08 => Key::Backspace,
        0x09 => Key::Tab,
        0x0D => Key::Enter,
        0x1B => Key::Escape,
        0x20 => Key::Space,
        0x21 => Key::PageUp,
        0x22 => Key::PageDown,
        0x23 => Key::End,
        0x24 => Key::Home,
        0x25 => Key::ArrowLeft,
        0x26 => Key::ArrowUp,
        0x27 => Key::ArrowRight,
        0x28 => Key::ArrowDown,
        0x2D => Key::Insert,
        0x2E => Key::Delete,
        0x30 => Key::Num0,
        0x31 => Key::Num1,
        0x32 => Key::Num2,
        0x33 => Key::Num3,
        0x34 => Key::Num4,
        0x35 => Key::Num5,
        0x36 => Key::Num6,
        0x37 => Key::Num7,
        0x38 => Key::Num8,
        0x39 => Key::Num9,
        0x41 => Key::A,
        0x42 => Key::B,
        0x43 => Key::C,
        0x44 => Key::D,
        0x45 => Key::E,
        0x46 => Key::F,
        0x47 => Key::G,
        0x48 => Key::H,
        0x49 => Key::I,
        0x4A => Key::J,
        0x4B => Key::K,
        0x4C => Key::L,
        0x4D => Key::M,
        0x4E => Key::N,
        0x4F => Key::O,
        0x50 => Key::P,
        0x51 => Key::Q,
        0x52 => Key::R,
        0x53 => Key::S,
        0x54 => Key::T,
        0x55 => Key::U,
        0x56 => Key::V,
        0x57 => Key::W,
        0x58 => Key::X,
        0x59 => Key::Y,
        0x5A => Key::Z,
        0x70 => Key::F1,
        0x71 => Key::F2,
        0x72 => Key::F3,
        0x73 => Key::F4,
        0x74 => Key::F5,
        0x75 => Key::F6,
        0x76 => Key::F7,
        0x77 => Key::F8,
        0x78 => Key::F9,
        0x79 => Key::F10,
        0x7A => Key::F11,
        0x7B => Key::F12,
        _ => return None,
    })
}

fn copy_to_clipboard(hwnd: HWND, text: &str) {
    let utf16: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    let bytes = utf16.len() * size_of::<u16>();
    unsafe {
        if OpenClipboard(Some(hwnd)).is_err() {
            return;
        }
        let _ = EmptyClipboard();
        if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, bytes) {
            let ptr = GlobalLock(hmem) as *mut u16;
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(utf16.as_ptr(), ptr, utf16.len());
                let _ = GlobalUnlock(hmem);
                let _ = SetClipboardData(13, Some(windows::Win32::Foundation::HANDLE(hmem.0)));
            }
        }
        let _ = CloseClipboard();
    }
}
