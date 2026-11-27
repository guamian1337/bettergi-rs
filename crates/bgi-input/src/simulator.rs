use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::Graphics::Gdi::ClientToScreen;
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MAPVK_VK_TO_VSC, MOUSE_EVENT_FLAGS,
    MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN,
    MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
    MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_WHEEL, MOUSEINPUT, MapVirtualKeyW, SendInput, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetClientRect, GetForegroundWindow, GetSystemMetrics, IsIconic, SM_CXVIRTUALSCREEN,
    SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SW_RESTORE, SetForegroundWindow,
    ShowWindow,
};

const WHEEL_DELTA_NOTCH: i32 = 120;

fn send_mouse(dw_flags: MOUSE_EVENT_FLAGS, dx: i32, dy: i32, mouse_data: u32) -> bool {
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
    unsafe { SendInput(&[input], size_of::<INPUT>() as i32) == 1 }
}

fn has_nonzero_frame_size(frame_size: (u32, u32)) -> bool {
    frame_size.0 > 0 && frame_size.1 > 0
}

fn window_has_expected_client_size(hwnd: isize, frame_size: (u32, u32)) -> bool {
    window_client_size(hwnd) == Some(frame_size)
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

pub fn window_client_size(hwnd: isize) -> Option<(u32, u32)> {
    if hwnd == 0 {
        return None;
    }
    let mut rect = RECT::default();
    if !unsafe { GetClientRect(HWND(hwnd as *mut _), &mut rect) }.is_ok() {
        return None;
    }
    let width = rect.right.saturating_sub(rect.left);
    let height = rect.bottom.saturating_sub(rect.top);
    (width > 0 && height > 0).then_some((width as u32, height as u32))
}

pub fn is_foreground_window(hwnd: isize) -> bool {
    hwnd != 0 && unsafe { GetForegroundWindow() == HWND(hwnd as *mut _) }
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

pub fn key_tap_checked(
    hwnd: isize,
    vk: u16,
    hold_ms: u64,
    frame_size: (u32, u32),
) -> Result<(), &'static str> {
    key_tap_checked_if(hwnd, vk, hold_ms, frame_size, || true)
}

pub fn key_tap_checked_if(
    hwnd: isize,
    vk: u16,
    hold_ms: u64,
    frame_size: (u32, u32),
    mut can_input: impl FnMut() -> bool,
) -> Result<(), &'static str> {
    if hwnd == 0 {
        return Err("游戏窗口句柄无效，未按下按键");
    }
    if !has_nonzero_frame_size(frame_size) {
        return Err("游戏客户区尺寸无效，未按下按键");
    }
    if !window_has_expected_client_size(hwnd, frame_size) {
        return Err("游戏客户区尺寸已变化，未按下按键");
    }
    if !is_foreground_window(hwnd) {
        return Err("游戏窗口不在前台，未按下按键");
    }
    if !can_input() {
        return Err("输入安全条件已变化，未按下按键");
    }
    if !window_has_expected_client_size(hwnd, frame_size) {
        return Err("发送前游戏客户区尺寸已变化，未按下按键");
    }
    if !is_foreground_window(hwnd) {
        return Err("发送前游戏窗口失焦，未按下按键");
    }
    if !can_input() {
        return Err("发送前输入安全条件已变化，未按下按键");
    }
    let pressed = send_key(vk, false);
    if !pressed {
        return Err("按键按下输入发送失败");
    }
    if hold_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(hold_ms));
    }
    if !send_key(vk, true) {
        release_all_keys();
        return Err("按键释放输入发送失败");
    }
    Ok(())
}

pub fn mouse_move_rel(dx: i32, dy: i32) {
    send_mouse(MOUSEEVENTF_MOVE, dx, dy, 0);
}

/// Moves the pointer relative to its current position, refusing to move a window that lost focus.
///
/// Aiming feeds raw pixel deltas straight into the camera, so a background window would turn the
/// player's view; this variant is the one the mining loop must use.
pub fn mouse_move_rel_checked(hwnd: isize, dx: i32, dy: i32) -> Result<(), &'static str> {
    if hwnd == 0 {
        return Err("游戏窗口句柄无效，未移动鼠标");
    }
    if !is_foreground_window(hwnd) {
        return Err("游戏窗口不在前台，未移动鼠标");
    }
    if !send_mouse(MOUSEEVENTF_MOVE, dx, dy, 0) {
        return Err("鼠标相对移动发送失败");
    }
    Ok(())
}

fn mouse_move_to_client_checked(hwnd: isize, client_x: i32, client_y: i32) -> bool {
    if hwnd == 0 {
        return false;
    }
    let mut pt = POINT {
        x: client_x,
        y: client_y,
    };
    if unsafe { !ClientToScreen(HWND(hwnd as *mut _), &mut pt).as_bool() } {
        return false;
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
    )
}

pub fn mouse_move_to_client(hwnd: isize, client_x: i32, client_y: i32) {
    let _ = mouse_move_to_client_checked(hwnd, client_x, client_y);
}

pub fn mouse_wheel_notch_checked(
    hwnd: isize,
    client_x: i32,
    client_y: i32,
    frame_size: (u32, u32),
    notches: i32,
) -> Result<(), &'static str> {
    if notches != -1 && notches != 1 {
        return Err("滚轮步数必须为 -1 或 1");
    }
    if window_client_size(hwnd) != Some(frame_size) {
        return Err("游戏客户区尺寸已变化，未滚动");
    }
    if !is_foreground_window(hwnd) {
        return Err("游戏窗口不在前台，未滚动");
    }
    if client_x < 0
        || client_y < 0
        || client_x >= frame_size.0 as i32
        || client_y >= frame_size.1 as i32
    {
        return Err("滚动坐标越界，未滚动");
    }
    if !mouse_move_to_client_checked(hwnd, client_x, client_y) {
        return Err("移动鼠标失败，未滚动");
    }
    std::thread::sleep(std::time::Duration::from_millis(20));
    if window_client_size(hwnd) != Some(frame_size) {
        return Err("移动后游戏客户区尺寸已变化，未滚动");
    }
    if !is_foreground_window(hwnd) {
        return Err("移动后游戏窗口失焦，未滚动");
    }
    let delta = if notches > 0 {
        WHEEL_DELTA_NOTCH as u32
    } else {
        (-WHEEL_DELTA_NOTCH) as u32
    };
    if !send_mouse(MOUSEEVENTF_WHEEL, 0, 0, delta) {
        return Err("滚轮输入发送失败");
    }
    Ok(())
}

/// Blocks a held-button press when the target window or frame is no longer trustworthy.
///
/// Read twice: once before the press and once after the size/focus recheck, so a window that
/// lost focus between the two probes still aborts before any input is sent.
fn held_button_blocker(hwnd: isize, frame_size: (u32, u32)) -> Option<&'static str> {
    if hwnd == 0 {
        return Some("游戏窗口句柄无效，未按下鼠标键");
    }
    if !has_nonzero_frame_size(frame_size) {
        return Some("游戏客户区尺寸无效，未按下鼠标键");
    }
    if !window_has_expected_client_size(hwnd, frame_size) {
        return Some("游戏客户区尺寸已变化，未按下鼠标键");
    }
    if !is_foreground_window(hwnd) {
        return Some("游戏窗口不在前台，未按下鼠标键");
    }
    if !window_has_expected_client_size(hwnd, frame_size) {
        return Some("发送前游戏客户区尺寸已变化，未按下鼠标键");
    }
    if !is_foreground_window(hwnd) {
        return Some("发送前游戏窗口失焦，未按下鼠标键");
    }
    None
}

pub fn mouse_left_down() {
    send_mouse(MOUSEEVENTF_LEFTDOWN, 0, 0, 0);
}

/// Presses the left mouse button after validating and rechecking the target window.
pub fn mouse_left_down_checked(hwnd: isize, frame_size: (u32, u32)) -> Result<(), &'static str> {
    if let Some(reason) = held_button_blocker(hwnd, frame_size) {
        return Err(reason);
    }
    if !send_mouse(MOUSEEVENTF_LEFTDOWN, 0, 0, 0) {
        return Err("鼠标左键按下输入发送失败");
    }
    Ok(())
}

pub fn mouse_left_up() {
    send_mouse(MOUSEEVENTF_LEFTUP, 0, 0, 0);
}

/// Releases the left mouse button regardless of the target window's current state.
pub fn mouse_left_up_checked() -> Result<(), &'static str> {
    if send_mouse(MOUSEEVENTF_LEFTUP, 0, 0, 0) {
        Ok(())
    } else {
        Err("鼠标左键松开输入发送失败")
    }
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

/// Moves to a client-space target, then rechecks foreground immediately before clicking.
pub fn mouse_left_click_client_checked(hwnd: isize, client_x: i32, client_y: i32) -> bool {
    mouse_move_to_client(hwnd, client_x, client_y);
    std::thread::sleep(std::time::Duration::from_millis(40));
    if !is_foreground_window(hwnd) {
        return false;
    }
    mouse_left_click();
    true
}

pub fn mouse_drag_client(hwnd: isize, x1: i32, y1: i32, x2: i32, y2: i32) {
    mouse_move_to_client(hwnd, x1, y1);
    std::thread::sleep(std::time::Duration::from_millis(45));
    mouse_left_down();
    std::thread::sleep(std::time::Duration::from_millis(45));

    let dx = (x2 - x1) as f64;
    let dy = (y2 - y1) as f64;
    let dist = (dx * dx + dy * dy).sqrt();
    let steps = ((dist / 42.0).ceil() as i32).clamp(6, 36);

    for step in 1..=steps {
        let t = (step as f64) / (steps as f64);
        // Quadratic ease-out slows near release to suppress map flick inertia.
        let eased = 1.0 - (1.0 - t) * (1.0 - t);
        let sx = (x1 as f64 + dx * eased).round() as i32;
        let sy = (y1 as f64 + dy * eased).round() as i32;
        mouse_move_to_client(hwnd, sx, sy);
        std::thread::sleep(std::time::Duration::from_millis(12));
    }

    // Hold briefly at destination before releasing so Genshin cancels map inertia.
    std::thread::sleep(std::time::Duration::from_millis(65));
    mouse_left_up();
    std::thread::sleep(std::time::Duration::from_millis(50));
}

pub fn mouse_wheel(notches: i32) {
    if notches == 0 {
        return;
    }
    let count = notches.abs().min(20);
    let delta = if notches > 0 {
        WHEEL_DELTA_NOTCH
    } else {
        -WHEEL_DELTA_NOTCH
    } as u32;
    for i in 0..count {
        send_mouse(MOUSEEVENTF_WHEEL, 0, 0, delta);
        if i + 1 < count {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

pub fn mouse_right_down() {
    send_mouse(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0);
}

pub fn mouse_right_up() {
    send_mouse(MOUSEEVENTF_RIGHTUP, 0, 0, 0);
}

pub fn mouse_right_click() {
    mouse_right_down();
    std::thread::sleep(std::time::Duration::from_millis(25));
    mouse_right_up();
}

pub fn mouse_middle_down() {
    send_mouse(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0);
}

/// Presses the middle mouse button after validating and rechecking the target window.
///
/// Element sight is held with this button, so a stale window must never receive the press.
pub fn mouse_middle_down_checked(hwnd: isize, frame_size: (u32, u32)) -> Result<(), &'static str> {
    if let Some(reason) = held_button_blocker(hwnd, frame_size) {
        return Err(reason);
    }
    if !send_mouse(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0) {
        return Err("鼠标中键按下输入发送失败");
    }
    Ok(())
}

pub fn mouse_middle_up() {
    send_mouse(MOUSEEVENTF_MIDDLEUP, 0, 0, 0);
}

/// Releases the middle mouse button regardless of the target window's current state.
pub fn mouse_middle_up_checked() -> Result<(), &'static str> {
    if send_mouse(MOUSEEVENTF_MIDDLEUP, 0, 0, 0) {
        Ok(())
    } else {
        Err("鼠标中键松开输入发送失败")
    }
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
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dwFlags: MOUSEEVENTF_MIDDLEUP,
                    ..Default::default()
                },
            },
        },
    ];
    unsafe {
        let _ = SendInput(&up_buttons, size_of::<INPUT>() as i32);
    }
}

fn send_key(vk: u16, up: bool) -> bool {
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
    unsafe { SendInput(&[input], size_of::<INPUT>() as i32) == 1 }
}

#[cfg(test)]
mod tests {
    use super::{
        has_nonzero_frame_size, key_tap_checked_if, mouse_left_down_checked,
        mouse_middle_down_checked, mouse_move_rel_checked,
    };

    #[test]
    fn checked_key_tap_rejects_invalid_targets_and_failed_guard() {
        assert!(key_tap_checked_if(0, 0x46, 0, (1920, 1080), || panic!("guard called")).is_err());
        assert!(key_tap_checked_if(0, 0x46, 0, (0, 1080), || true).is_err());
    }

    #[test]
    fn frame_size_requires_both_dimensions_to_be_nonzero() {
        assert!(has_nonzero_frame_size((1920, 1080)));
        assert!(has_nonzero_frame_size((1, 1)));
        assert!(!has_nonzero_frame_size((0, 1080)));
        assert!(!has_nonzero_frame_size((1920, 0)));
        assert!(!has_nonzero_frame_size((0, 0)));
    }

    #[test]
    fn invalid_window_and_frame_reject_mouse_down_without_os_input() {
        assert!(mouse_left_down_checked(0, (1920, 1080)).is_err());
        assert!(mouse_left_down_checked(0, (0, 1080)).is_err());
        assert!(mouse_left_down_checked(0, (1920, 0)).is_err());
    }

    #[test]
    fn middle_button_and_relative_move_refuse_invalid_windows() {
        assert!(mouse_middle_down_checked(0, (1920, 1080)).is_err());
        assert!(mouse_middle_down_checked(0, (0, 1080)).is_err());
        assert!(mouse_move_rel_checked(0, 10, 10).is_err());
        // 非零但并非真实窗口的句柄也必须先过前台检查，不能落到 SendInput。
        assert!(mouse_move_rel_checked(0x1234, 10, 10).is_err());
    }
}
