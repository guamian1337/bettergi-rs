use std::path::Path;

use bgi_control::WindowCandidate;
use windows::Win32::Foundation::{HWND, LPARAM, RECT};
use windows::core::BOOL;
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetClientRect, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
    IsIconic, IsWindowVisible,
};

struct EnumCtx {
    preferred_process: String,
    items: Vec<WindowCandidate>,
}

/// Enumerate visible non-minimized desktop windows suitable for capture.
pub fn enumerate_candidate_windows(preferred_process: &str) -> Vec<WindowCandidate> {
    let mut ctx = EnumCtx {
        preferred_process: preferred_process.to_ascii_lowercase(),
        items: Vec::new(),
    };

    unsafe {
        let _ = EnumWindows(Some(enum_window_proc), LPARAM(&mut ctx as *mut EnumCtx as isize));
    }

    ctx.items.sort_by(|a, b| {
        let a_pref = a.process_name.eq_ignore_ascii_case(&ctx.preferred_process);
        let b_pref = b.process_name.eq_ignore_ascii_case(&ctx.preferred_process);
        b_pref
            .cmp(&a_pref)
            .then_with(|| (b.width * b.height).cmp(&(a.width * a.height)))
            .then_with(|| a.title.cmp(&b.title))
    });

    ctx.items
}

unsafe extern "system" fn enum_window_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let ctx = unsafe { &mut *(lparam.0 as *mut EnumCtx) };

    if unsafe { !IsWindowVisible(hwnd).as_bool() } {
        return BOOL(1);
    }
    if unsafe { IsIconic(hwnd).as_bool() } {
        return BOOL(1);
    }

    let title_len = unsafe { GetWindowTextLengthW(hwnd) };
    if title_len <= 0 {
        return BOOL(1);
    }

    let mut rect = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut rect) }.is_err() {
        return BOOL(1);
    }

    let width = (rect.right - rect.left).max(0) as u32;
    let height = (rect.bottom - rect.top).max(0) as u32;
    if width < 200 || height < 200 {
        return BOOL(1);
    }

    let mut buf = vec![0u16; (title_len as usize) + 1];
    let copied = unsafe { GetWindowTextW(hwnd, &mut buf) };
    if copied <= 0 {
        return BOOL(1);
    }
    let title = String::from_utf16_lossy(&buf[..copied as usize]);

    let process_name = query_process_name(hwnd).unwrap_or_else(|| "<受限/未知进程>".to_string());

    ctx.items.push(WindowCandidate {
        hwnd: hwnd.0 as isize,
        title,
        process_name,
        width,
        height,
    });

    BOOL(1)
}

fn query_process_name(hwnd: HWND) -> Option<String> {
    let mut pid = 0u32;
    unsafe {
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
    }
    if pid == 0 {
        return None;
    }

    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut path_buf = vec![0u16; 512];
    let mut len = path_buf.len() as u32;
    let ok = unsafe {
        QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(path_buf.as_mut_ptr()),
            &mut len,
        )
    };
    let _ = unsafe { windows::Win32::Foundation::CloseHandle(handle) };
    ok.ok()?;

    let full = String::from_utf16_lossy(&path_buf[..len as usize]);
    Path::new(&full)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
}
