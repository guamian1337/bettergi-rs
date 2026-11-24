use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MAPVK_VK_TO_VSC, MOUSE_EVENT_FLAGS,
    MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE,
    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_WHEEL,
    MOUSEINPUT, MapVirtualKeyW, SendInput, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, IsIconic, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN, SW_RESTORE, SetForegroundWindow, ShowWindow,
};

const WHEEL_DELTA_NOTCH: i32 = 120;

fn send_mouse(dw_flags: MOUSE_EVENT_FLAGS, dx: i32, dy: i32, mouse_data: u32) {
    let input = INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx,
                dy,
                mouseData: mouse_data,
                dwFlags: dw_flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    unsafe {
        let _ = SendInput(&[input], size_of::<INPUT>() as i32);
    }
}

pub fn focus_window(hwnd: isize) {
    if hwnd == 0 {
        return;
    }
    let h = HWND(hwnd as *mut _);
    unsafe {
        if IsIconic(h).as_bool() {
            let _ = ShowWindow(h, SW_RESTORE);
        }
        let _ = SetForegroundWindow(h);
    }
}

pub fn window_dpi_scale(hwnd: isize) -> f32 {
    if hwnd == 0 {
        return 1.0;
    }
    let dpi = unsafe { GetDpiForWindow(HWND(hwnd as *mut _)) };
    if dpi == 0 { 1.0 } else { (dpi as f32) / 96.0 }
}

/// Matches `BetterGenshinImpact/Core/Simulator/ExtendedKeys.cs` hardware behavior.
pub fn is_extended_key(vk: u16) -> bool {
    matches!(
        vk,
        0xA3 // VK_RCONTROL
        | 0xA5 // VK_RMENU
        | 0x2D // VK_INSERT
        | 0x2E // VK_DELETE
        | 0x24 // VK_HOME
        | 0x23 // VK_END
        | 0x21 // VK_PRIOR
        | 0x22 // VK_NEXT
        | 0x25 // VK_LEFT
        | 0x26 // VK_UP
        | 0x27 // VK_RIGHT
        | 0x28 // VK_DOWN
        | 0x6F // VK_DIVIDE
        | 0x90 // VK_NUMLOCK
        | 0x03 // VK_CANCEL
        | 0x2C // VK_SNAPSHOT
        | 0x5B // VK_LWIN
        | 0x5C // VK_RWIN
        | 0x5D // VK_APPS
    )
}

pub fn key_down(vk: u16) {
    send_key(vk, false);
}

pub fn key_up(vk: u16) {
    send_key(vk, true);
}

pub fn key_tap(vk: u16, hold_ms: u64) {
    key_down(vk);
    if hold_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(hold_ms));
    }
    key_up(vk);
}

pub fn mouse_move_rel(dx: i32, dy: i32) {
    send_mouse(MOUSEEVENTF_MOVE, dx, dy, 0);
}

pub fn mouse_move_to_client(hwnd: isize, client_x: i32, client_y: i32) {
    if hwnd == 0 {
        return;
    }
    let mut pt = POINT {
        x: client_x,
        y: client_y,
    };
    if unsafe { !ClientToScreen(HWND(hwnd as *mut _), &mut pt).as_bool() } {
        return;
    }
    let vx = unsafe { GetSystemMetrics(SM_XVIRTUALSCREEN) };
    let vy = unsafe { GetSystemMetrics(SM_YVIRTUALSCREEN) };
    let vw = unsafe { GetSystemMetrics(SM_CXVIRTUALSCREEN) }.max(1);
    let vh = unsafe { GetSystemMetrics(SM_CYVIRTUALSCREEN) }.max(1);

    let norm_x = ((((pt.x - vx) as f64) * 65535.0) / ((vw - 1).max(1) as f64))
        .round()
        .clamp(0.0, 65535.0) as i32;
    let norm_y = ((((pt.y - vy) as f64) * 65535.0) / ((vh - 1).max(1) as f64))
        .round()
        .clamp(0.0, 65535.0) as i32;
    send_mouse(
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        norm_x,
        norm_y,
        0,
    );
}

pub fn mouse_left_down() {
    send_mouse(MOUSEEVENTF_LEFTDOWN, 0, 0, 0);
}

pub fn mouse_left_up() {
    send_mouse(MOUSEEVENTF_LEFTUP, 0, 0, 0);
}

pub fn mouse_left_click() {
    mouse_left_down();
    mouse_left_up();
}

pub fn mouse_left_click_client(hwnd: isize, client_x: i32, client_y: i32) {
    mouse_move_to_client(hwnd, client_x, client_y);
    std::thread::sleep(std::time::Duration::from_millis(40));
    mouse_left_click();
}

pub fn mouse_drag_client(hwnd: isize, x1: i32, y1: i32, x2: i32, y2: i32) {
    mouse_move_to_client(hwnd, x1, y1);
    std::thread::sleep(std::time::Duration::from_millis(50));
    mouse_left_down();
    std::thread::sleep(std::time::Duration::from_millis(50));
    const STEPS: i32 = 4;
    for step in 1..=STEPS {
        let sx = x1 + ((x2 - x1) * step) / STEPS;
        let sy = y1 + ((y2 - y1) * step) / STEPS;
        mouse_move_to_client(hwnd, sx, sy);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    mouse_left_up();
    std::thread::sleep(std::time::Duration::from_millis(60));
}

pub fn mouse_wheel(notches: i32) {
    if notches == 0 {
        return;
    }
    send_mouse(
        MOUSEEVENTF_WHEEL,
        0,
        0,
        (notches.saturating_mul(WHEEL_DELTA_NOTCH)) as u32,
    );
}

pub fn mouse_right_click() {
    send_mouse(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0);
    send_mouse(MOUSEEVENTF_RIGHTUP, 0, 0, 0);
}

/// Releases common movement/modifier keys and mouse buttons on emergency stop.
pub fn release_all_keys() {
    for vk in 1u16..255u16 {
        let state = unsafe { GetAsyncKeyState(vk as i32) };
        if (state as u16 & 0x8000) != 0 {
            key_up(vk);
        }
    }
    let up_buttons = [
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dwFlags: MOUSEEVENTF_LEFTUP,
                    ..Default::default()
                },
            },
        },
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dwFlags: MOUSEEVENTF_RIGHTUP,
                    ..Default::default()
                },
            },
        },
    ];
    unsafe {
        let _ = SendInput(&up_buttons, size_of::<INPUT>() as i32);
    }
}

fn send_key(vk: u16, up: bool) {
    let scan = unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) } as u16;
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if scan != 0 {
        flags |= KEYEVENTF_SCANCODE;
    }
    if is_extended_key(vk) {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    if up {
        flags |= KEYEVENTF_KEYUP;
    }

    let input = INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    unsafe {
        let _ = SendInput(&[input], size_of::<INPUT>() as i32);
    }
}
