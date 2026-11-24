use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MAPVK_VK_TO_VSC, MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEINPUT,
    MapVirtualKeyW, SendInput, VIRTUAL_KEY,
};

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
    let input = INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx,
                dy,
                mouseData: 0,
                dwFlags: MOUSEEVENTF_MOVE,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    unsafe {
        let _ = SendInput(&[input], size_of::<INPUT>() as i32);
    }
}

pub fn mouse_left_click() {
    let inputs = [
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dwFlags: MOUSEEVENTF_LEFTDOWN,
                    ..Default::default()
                },
            },
        },
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dwFlags: MOUSEEVENTF_LEFTUP,
                    ..Default::default()
                },
            },
        },
    ];
    unsafe {
        let _ = SendInput(&inputs, size_of::<INPUT>() as i32);
    }
}

pub fn mouse_right_click() {
    let inputs = [
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dwFlags: MOUSEEVENTF_RIGHTDOWN,
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
        let _ = SendInput(&inputs, size_of::<INPUT>() as i32);
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
