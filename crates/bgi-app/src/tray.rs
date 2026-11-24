use std::num::NonZeroIsize;

use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, CreateBitmap, CreateCompatibleDC, CreateDIBSection,
    DIB_RGB_COLORS, DeleteDC, DeleteObject, HBITMAP,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Shell::{
    NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD, NIM_DELETE, NOTIFYICONDATAW, Shell_NotifyIconW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreateIconIndirect, CreatePopupMenu, CreateWindowExW, DefWindowProcW,
    DestroyIcon, DestroyMenu, DestroyWindow, GetCursorPos, HICON, ICON_BIG, ICON_SMALL, ICONINFO,
    IDI_APPLICATION, LoadIconW, MF_SEPARATOR, MF_STRING, PostMessageW, PostQuitMessage,
    RegisterClassW, SetForegroundWindow, TPM_BOTTOMALIGN, TPM_LEFTALIGN, TPM_RIGHTBUTTON,
    TrackPopupMenu, WM_APP, WM_COMMAND, WM_DESTROY, WM_LBUTTONDBLCLK, WM_RBUTTONUP, WNDCLASSW,
};
use windows::core::{PCWSTR, w};

const LOGO_PNG: &[u8] = include_bytes!("../../../assets/logo.png");

pub const WM_APP_TRAY: u32 = WM_APP + 10;
pub const WM_APP_OPEN_UI: u32 = WM_APP + 11;
pub const WM_APP_CAPTURE_ONCE: u32 = WM_APP + 12;
pub const WM_APP_EMERGENCY_STOP: u32 = WM_APP + 13;

const CMD_OPEN_UI: usize = 1001;
const CMD_CAPTURE_ONCE: usize = 1002;
const CMD_EMERGENCY_STOP: usize = 1003;
const CMD_QUIT: usize = 1004;

pub struct TrayHost {
    pub hwnd: HWND,
    pub hicon: HICON,
    owns_icon: bool,
}

impl TrayHost {
    pub fn create() -> windows::core::Result<Self> {
        let hinstance: HINSTANCE = unsafe { GetModuleHandleW(None) }?.into();
        let class_name: PCWSTR = w!("BetterGI_RS_TrayHost");

        let (hicon, owns_icon) = create_logo_hicon()
            .map(|h| (h, true))
            .unwrap_or_else(|| unsafe {
                (
                    LoadIconW(None, IDI_APPLICATION).unwrap_or_default(),
                    false,
                )
            });

        let wc = WNDCLASSW {
            lpfnWndProc: Some(tray_wnd_proc),
            hInstance: hinstance,
            hIcon: hicon,
            lpszClassName: class_name,
            ..Default::default()
        };
        unsafe {
            RegisterClassW(&wc);
        }

        let hwnd = unsafe {
            CreateWindowExW(
                Default::default(),
                class_name,
                w!("BetterGI-RS Tray"),
                Default::default(),
                0,
                0,
                0,
                0,
                None,
                None,
                Some(hinstance),
                None,
            )
        }?;

        let mut nid = NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: hwnd,
            uID: 1,
            uFlags: NIF_MESSAGE | NIF_ICON | NIF_TIP,
            uCallbackMessage: WM_APP_TRAY,
            hIcon: hicon,
            ..Default::default()
        };

        let tip: Vec<u16> = "BetterGI-RS (双击打开主界面)"
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let copy_len = tip.len().min(nid.szTip.len());
        nid.szTip[..copy_len].copy_from_slice(&tip[..copy_len]);

        unsafe {
            let _ = Shell_NotifyIconW(NIM_ADD, &nid);
        }

        Ok(Self {
            hwnd,
            hicon,
            owns_icon,
        })
    }

    pub fn apply_window_icons(&self, target_hwnd: HWND) {
        if !self.hicon.is_invalid() {
            unsafe {
                let _ = PostMessageW(
                    Some(target_hwnd),
                    windows::Win32::UI::WindowsAndMessaging::WM_SETICON,
                    WPARAM(ICON_BIG as usize),
                    LPARAM(self.hicon.0 as isize),
                );
                let _ = PostMessageW(
                    Some(target_hwnd),
                    windows::Win32::UI::WindowsAndMessaging::WM_SETICON,
                    WPARAM(ICON_SMALL as usize),
                    LPARAM(self.hicon.0 as isize),
                );
            }
        }
    }
}

impl Drop for TrayHost {
    fn drop(&mut self) {
        let nid = NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: self.hwnd,
            uID: 1,
            ..Default::default()
        };
        unsafe {
            let _ = Shell_NotifyIconW(NIM_DELETE, &nid);
            if !self.hwnd.is_invalid() {
                let _ = DestroyWindow(self.hwnd);
            }
            if self.owns_icon && !self.hicon.is_invalid() {
                let _ = DestroyIcon(self.hicon);
            }
        }
    }
}

unsafe extern "system" fn tray_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_APP_TRAY => {
            let event = (lparam.0 & 0xFFFF) as u32;
            if event == WM_LBUTTONDBLCLK {
                unsafe {
                    let _ = PostMessageW(Some(hwnd), WM_APP_OPEN_UI, WPARAM(0), LPARAM(0));
                }
            } else if event == WM_RBUTTONUP {
                show_context_menu(hwnd);
            }
            LRESULT(0)
        }
        WM_COMMAND => {
            let cmd = wparam.0 & 0xFFFF;
            match cmd {
                CMD_OPEN_UI => unsafe {
                    let _ = PostMessageW(Some(hwnd), WM_APP_OPEN_UI, WPARAM(0), LPARAM(0));
                },
                CMD_CAPTURE_ONCE => unsafe {
                    let _ = PostMessageW(Some(hwnd), WM_APP_CAPTURE_ONCE, WPARAM(0), LPARAM(0));
                },
                CMD_EMERGENCY_STOP => unsafe {
                    let _ = PostMessageW(Some(hwnd), WM_APP_EMERGENCY_STOP, WPARAM(0), LPARAM(0));
                },
                CMD_QUIT => unsafe {
                    PostQuitMessage(0);
                },
                _ => {}
            }
            LRESULT(0)
        }
        WM_DESTROY => LRESULT(0),
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn show_context_menu(hwnd: HWND) {
    let Ok(menu) = (unsafe { CreatePopupMenu() }) else {
        return;
    };
    unsafe {
        let _ = AppendMenuW(menu, MF_STRING, CMD_OPEN_UI, w!("打开主界面"));
        let _ = AppendMenuW(menu, MF_STRING, CMD_CAPTURE_ONCE, w!("单帧抓图测试"));
        let _ = AppendMenuW(menu, MF_STRING, CMD_EMERGENCY_STOP, w!("急停 (释放按键)"));
        let _ = AppendMenuW(menu, MF_SEPARATOR, 0, None);
        let _ = AppendMenuW(menu, MF_STRING, CMD_QUIT, w!("彻底退出 BetterGI"));

        let mut pt = POINT::default();
        let _ = GetCursorPos(&mut pt);
        let _ = SetForegroundWindow(hwnd);
        let _ = TrackPopupMenu(
            menu,
            TPM_LEFTALIGN | TPM_BOTTOMALIGN | TPM_RIGHTBUTTON,
            pt.x,
            pt.y,
            Some(0),
            hwnd,
            None,
        );
        let _ = DestroyMenu(menu);
    }
}

fn create_logo_hicon() -> Option<HICON> {
    let img = image::load_from_memory(LOGO_PNG).ok()?;
    let resized = img
        .resize_exact(32, 32, image::imageops::FilterType::Triangle)
        .to_rgba8();

    let hdc = unsafe { CreateCompatibleDC(None) };
    if hdc.is_invalid() {
        return None;
    }

    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: 32,
            biHeight: -32,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        bmiColors: [Default::default()],
    };

    let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
    let color_bmp: HBITMAP =
        unsafe { CreateDIBSection(Some(hdc), &bmi, DIB_RGB_COLORS, &mut bits, None, 0) }.ok()?;
    if bits.is_null() {
        unsafe {
            let _ = DeleteObject(color_bmp.into());
            let _ = DeleteDC(hdc);
        }
        return None;
    }

    let dst = unsafe { std::slice::from_raw_parts_mut(bits as *mut u8, 32 * 32 * 4) };
    for (i, px) in resized.pixels().enumerate() {
        let base = i * 4;
        dst[base] = px[2];
        dst[base + 1] = px[1];
        dst[base + 2] = px[0];
        dst[base + 3] = px[3];
    }

    let mask_bytes = [0u8; 32 * 4];
    let mask_bmp = unsafe { CreateBitmap(32, 32, 1, 1, Some(mask_bytes.as_ptr() as *const _)) };

    let info = ICONINFO {
        fIcon: true.into(),
        xHotspot: 0,
        yHotspot: 0,
        hbmMask: mask_bmp,
        hbmColor: color_bmp,
    };

    let icon = unsafe { CreateIconIndirect(&info) }.ok();
    unsafe {
        if !mask_bmp.is_invalid() {
            let _ = DeleteObject(mask_bmp.into());
        }
        let _ = DeleteObject(color_bmp.into());
        let _ = DeleteDC(hdc);
    }
    let _ = NonZeroIsize::new(icon.map(|i| i.0 as isize).unwrap_or(0))?;
    icon
}
