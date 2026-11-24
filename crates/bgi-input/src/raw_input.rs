use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::thread::JoinHandle;

use bgi_control::{HotkeyAction, HotkeyBinding, ModifierMask, format_hotkey};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::{
    GetRawInputData, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RID_INPUT,
    RIDEV_INPUTSINK, RIDEV_REMOVE, RegisterRawInputDevices,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW, HWND_MESSAGE,
    MSG, PostMessageW, PostQuitMessage, RegisterClassW, RI_KEY_BREAK, RI_KEY_E0, RI_KEY_E1,
    TranslateMessage, WM_APP, WM_CLOSE, WM_DESTROY, WM_INPUT, WNDCLASSW,
};
use windows::core::{PCWSTR, w};

use crate::InputError;

const GENERIC_DESKTOP_PAGE: u16 = 0x01;
const KEYBOARD_USAGE: u16 = 0x06;
const INVALID_VKEY: u16 = 0x00FF;
const WM_APP_SHUTDOWN: u32 = WM_APP + 1;

static SINGLETON_RUNNING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub enum RawInputSignal {
    KeyObserved {
        vk: u16,
        scan_code: u16,
        extended: bool,
        pressed: bool,
        combo: String,
    },
    HotkeyTriggered {
        action: HotkeyAction,
        combo: String,
    },
    HotkeyCaptured(HotkeyBinding),
}

#[derive(Default)]
struct SharedInputState {
    bindings: Vec<HotkeyBinding>,
    capture_target: Option<HotkeyAction>,
    modifiers: PhysicalModifiers,
    pressed_vks: HashSet<u16>,
}

#[derive(Default, Clone, Copy)]
struct PhysicalModifiers {
    lctrl: bool,
    rctrl: bool,
    lshift: bool,
    rshift: bool,
    lalt: bool,
    ralt: bool,
    lwin: bool,
    rwin: bool,
}

impl PhysicalModifiers {
    fn as_mask(self) -> ModifierMask {
        ModifierMask {
            ctrl: self.lctrl || self.rctrl,
            shift: self.lshift || self.rshift,
            alt: self.lalt || self.ralt,
            win: self.lwin || self.rwin,
        }
    }

    fn update(&mut self, norm_vk: u16, pressed: bool) -> bool {
        match norm_vk {
            0xA2 => self.lctrl = pressed,
            0xA3 => self.rctrl = pressed,
            0xA0 => self.lshift = pressed,
            0xA1 => self.rshift = pressed,
            0xA4 => self.lalt = pressed,
            0xA5 => self.ralt = pressed,
            0x5B => self.lwin = pressed,
            0x5C => self.rwin = pressed,
            _ => return false,
        }
        true
    }
}

/// Resolves left/right physical modifier virtual keys using E0/make code.
pub(crate) fn normalize_raw_vkey(vkey: u16, make_code: u16, flags: u16) -> u16 {
    let e0 = (flags & RI_KEY_E0 as u16) != 0;
    match vkey {
        0x10 => {
            // VK_SHIFT: 0x36 is Right Shift scan code
            if make_code == 0x36 { 0xA1 } else { 0xA0 }
        }
        0x11 => {
            // VK_CONTROL
            if e0 { 0xA3 } else { 0xA2 }
        }
        0x12 => {
            // VK_MENU (Alt)
            if e0 { 0xA5 } else { 0xA4 }
        }
        other => other,
    }
}

struct WindowUserData {
    state: Arc<Mutex<SharedInputState>>,
    signal_cb: Arc<dyn Fn(RawInputSignal) + Send + Sync>,
    raw_buf: Vec<u8>,
}

impl WindowUserData {
    fn process_raw_keyboard(
        &mut self,
        hdevice_is_null: bool,
        vkey: u16,
        make_code: u16,
        flags: u16,
    ) {
        // Guard 1: Filter software-injected SendInput packets (`hDevice == NULL`).
        if hdevice_is_null || vkey == 0 || vkey == INVALID_VKEY {
            return;
        }

        let pressed = (flags & RI_KEY_BREAK as u16) == 0;
        let extended = (flags & (RI_KEY_E0 | RI_KEY_E1) as u16) != 0;
        let norm_vk = normalize_raw_vkey(vkey, make_code, flags);

        let mut emitted = Vec::new();
        {
            let mut guard = self.state.lock().unwrap();
            let is_mod = guard.modifiers.update(norm_vk, pressed);
            let mask = guard.modifiers.as_mask();

            let is_new_press = if pressed {
                guard.pressed_vks.insert(norm_vk)
            } else {
                guard.pressed_vks.remove(&norm_vk);
                false
            };

            let combo = if is_mod {
                bgi_control::vk_name(norm_vk)
            } else {
                format_hotkey(mask, norm_vk)
            };

            if is_new_press || !pressed {
                emitted.push(RawInputSignal::KeyObserved {
                    vk: norm_vk,
                    scan_code: make_code,
                    extended,
                    pressed,
                    combo: combo.clone(),
                });
            }

            if is_new_press && !is_mod {
                if let Some(target_action) = guard.capture_target.take() {
                    let binding = HotkeyBinding {
                        action: target_action,
                        vk: norm_vk,
                        modifiers: mask,
                    };
                    if let Some(slot) = guard.bindings.iter_mut().find(|b| b.action == target_action) {
                        *slot = binding.clone();
                    } else {
                        guard.bindings.push(binding.clone());
                    }
                    emitted.push(RawInputSignal::HotkeyCaptured(binding));
                } else {
                    for binding in &guard.bindings {
                        if binding.vk == norm_vk && binding.modifiers == mask {
                            emitted.push(RawInputSignal::HotkeyTriggered {
                                action: binding.action,
                                combo: binding.display_combo(),
                            });
                        }
                    }
                }
            }
        }

        for sig in emitted {
            (self.signal_cb)(sig);
        }
    }
}

/// Process-singleton Raw Input listener on a dedicated `HWND_MESSAGE` thread.
pub struct RawInputHub {
    hwnd_raw: isize,
    state: Arc<Mutex<SharedInputState>>,
    thread: Option<JoinHandle<()>>,
}

impl RawInputHub {
    pub fn start<F>(initial_bindings: Vec<HotkeyBinding>, on_signal: F) -> Result<Self, InputError>
    where
        F: Fn(RawInputSignal) + Send + Sync + 'static,
    {
        if SINGLETON_RUNNING
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(InputError::AlreadyRunning);
        }

        let state = Arc::new(Mutex::new(SharedInputState {
            bindings: initial_bindings,
            ..Default::default()
        }));
        let state_clone = state.clone();
        let cb: Arc<dyn Fn(RawInputSignal) + Send + Sync> = Arc::new(on_signal);
        let (init_tx, init_rx) = std_mpsc::sync_channel::<Result<isize, String>>(1);

        let handle = std::thread::Builder::new()
            .name("bgi-rawinput-hub".to_string())
            .spawn(move || {
                run_message_loop(state_clone, cb, init_tx);
                SINGLETON_RUNNING.store(false, Ordering::SeqCst);
            })
            .map_err(|e| {
                SINGLETON_RUNNING.store(false, Ordering::SeqCst);
                InputError::Init(format!("启动 RawInput 线程失败: {e}"))
            })?;

        match init_rx.recv() {
            Ok(Ok(hwnd_raw)) => Ok(Self {
                hwnd_raw,
                state,
                thread: Some(handle),
            }),
            Ok(Err(msg)) => {
                let _ = handle.join();
                Err(InputError::Init(msg))
            }
            Err(_) => {
                let _ = handle.join();
                Err(InputError::Init("RawInput 初始化通道意外关闭".into()))
            }
        }
    }

    pub fn set_bindings(&self, bindings: Vec<HotkeyBinding>) {
        let mut guard = self.state.lock().unwrap();
        guard.bindings = bindings;
    }

    pub fn begin_capture(&self, target: Option<HotkeyAction>) {
        let mut guard = self.state.lock().unwrap();
        guard.capture_target = target;
    }

    pub fn shutdown(&mut self) {
        if self.hwnd_raw != 0 {
            unsafe {
                let _ = PostMessageW(
                    Some(HWND(self.hwnd_raw as *mut _)),
                    WM_APP_SHUTDOWN,
                    WPARAM(0),
                    LPARAM(0),
                );
            }
            self.hwnd_raw = 0;
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for RawInputHub {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_message_loop(
    state: Arc<Mutex<SharedInputState>>,
    signal_cb: Arc<dyn Fn(RawInputSignal) + Send + Sync>,
    init_tx: std_mpsc::SyncSender<Result<isize, String>>,
) {
    let class_name: PCWSTR = w!("BetterGI_Rust_RawInput_Sink");
    let hinstance = unsafe { GetModuleHandleW(None) }.unwrap_or_default();

    let wc = WNDCLASSW {
        lpfnWndProc: Some(raw_wnd_proc),
        hInstance: hinstance.into(),
        lpszClassName: class_name,
        ..Default::default()
    };
    unsafe {
        RegisterClassW(&wc);
    }

    let user_data = Box::new(WindowUserData {
        state,
        signal_cb,
        raw_buf: Vec::with_capacity(64),
    });
    let user_ptr = Box::into_raw(user_data);

    let hwnd = match unsafe {
        CreateWindowExW(
            Default::default(),
            class_name,
            w!("BetterGI RawInput Sink"),
            Default::default(),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            Some(hinstance.into()),
            Some(user_ptr as *const _),
        )
    } {
        Ok(h) if !h.is_invalid() => h,
        Ok(_) | Err(_) => {
            unsafe {
                drop(Box::from_raw(user_ptr));
            }
            let _ = init_tx.send(Err("CreateWindowExW(HWND_MESSAGE) 失败".to_string()));
            return;
        }
    };

    let dev = [RAWINPUTDEVICE {
        usUsagePage: GENERIC_DESKTOP_PAGE,
        usUsage: KEYBOARD_USAGE,
        dwFlags: RIDEV_INPUTSINK,
        hwndTarget: hwnd,
    }];
    if unsafe { RegisterRawInputDevices(&dev, size_of::<RAWINPUTDEVICE>() as u32) }.is_err() {
        unsafe {
            let _ = DestroyWindow(hwnd);
        }
        let _ = init_tx.send(Err("RegisterRawInputDevices 失败".to_string()));
        return;
    }

    let _ = init_tx.send(Ok(hwnd.0 as isize));

    let mut msg = MSG::default();
    while unsafe { GetMessageW(&mut msg, None, 0, 0) }.as_bool() {
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    let remove_dev = [RAWINPUTDEVICE {
        usUsagePage: GENERIC_DESKTOP_PAGE,
        usUsage: KEYBOARD_USAGE,
        dwFlags: RIDEV_REMOVE,
        hwndTarget: HWND::default(),
    }];
    unsafe {
        let _ = RegisterRawInputDevices(&remove_dev, size_of::<RAWINPUTDEVICE>() as u32);
    }
}

unsafe extern "system" fn raw_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    use windows::Win32::UI::WindowsAndMessaging::{
        CREATESTRUCTW, GWLP_USERDATA, GetWindowLongPtrW, SetWindowLongPtrW, WM_NCCREATE,
    };

    if msg == WM_NCCREATE {
        let cs = unsafe { &*(lparam.0 as *const CREATESTRUCTW) };
        unsafe {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, cs.lpCreateParams as isize);
        }
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    }

    let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowUserData;
    if msg == WM_DESTROY {
        if !ptr.is_null() {
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                drop(Box::from_raw(ptr));
            }
        }
        unsafe { PostQuitMessage(0) };
        return LRESULT(0);
    }

    if msg == WM_APP_SHUTDOWN || msg == WM_CLOSE {
        unsafe {
            let _ = DestroyWindow(hwnd);
        }
        return LRESULT(0);
    }

    if msg == WM_INPUT && !ptr.is_null() {
        let data = unsafe { &mut *ptr };
        let hraw = HRAWINPUT(lparam.0 as *mut _);
        let mut size = 0u32;
        let header_size = size_of::<RAWINPUTHEADER>() as u32;
        if unsafe { GetRawInputData(hraw, RID_INPUT, None, &mut size, header_size) } == 0
            && size > 0
        {
            if data.raw_buf.len() < size as usize {
                data.raw_buf.resize(size as usize, 0);
            }
            let copied = unsafe {
                GetRawInputData(
                    hraw,
                    RID_INPUT,
                    Some(data.raw_buf.as_mut_ptr() as *mut _),
                    &mut size,
                    header_size,
                )
            };
            if copied != u32::MAX && copied as usize >= size_of::<RAWINPUT>() {
                let raw = unsafe { &*(data.raw_buf.as_ptr() as *const RAWINPUT) };
                if raw.header.dwType == 1 {
                    // RIM_TYPEKEYBOARD == 1
                    let kb = unsafe { raw.data.keyboard };
                    data.process_raw_keyboard(
                        raw.header.hDevice.is_invalid(),
                        kb.VKey,
                        kb.MakeCode,
                        kb.Flags,
                    );
                }
            }
        }
        return LRESULT(0);
    }

    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sendinput_null_device_is_filtered_and_modifiers_match_hotkeys() {
        let signals = Arc::new(Mutex::new(Vec::new()));
        let signals_cb = signals.clone();
        let mut ud = WindowUserData {
            state: Arc::new(Mutex::new(SharedInputState {
                bindings: vec![HotkeyBinding {
                    action: HotkeyAction::EmergencyStop,
                    vk: 0x78, // F9
                    modifiers: ModifierMask {
                        ctrl: true,
                        ..Default::default()
                    },
                }],
                ..Default::default()
            })),
            signal_cb: Arc::new(move |sig| signals_cb.lock().unwrap().push(sig)),
            raw_buf: Vec::new(),
        };

        // Simulated SendInput event (hDevice == NULL) must be ignored completely.
        ud.process_raw_keyboard(true, 0x11, 0x1D, 0);
        ud.process_raw_keyboard(true, 0x78, 0x43, 0);
        assert!(signals.lock().unwrap().is_empty());

        // Real hardware Ctrl + F9 triggers EmergencyStop once (autorepeat ignored).
        ud.process_raw_keyboard(false, 0x11, 0x1D, 0); // LCtrl down
        ud.process_raw_keyboard(false, 0x78, 0x43, 0); // F9 down
        ud.process_raw_keyboard(false, 0x78, 0x43, 0); // F9 autorepeat

        let captured = signals.lock().unwrap();
        let triggered = captured
            .iter()
            .filter(|s| matches!(s, RawInputSignal::HotkeyTriggered { .. }))
            .count();
        assert_eq!(triggered, 1);
    }
}
