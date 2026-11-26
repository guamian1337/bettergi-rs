use std::ptr::{copy_nonoverlapping, null_mut};

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleDC, CreateDIBSection,
    DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, HBITMAP, HDC, HGDIOBJ, ReleaseDC, SRCCOPY,
    SelectObject,
};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, IsIconic, IsWindow};

use crate::CaptureError;
use crate::pool::{FrameBufferPool, PooledBuffer};

pub struct BitBltCapturer {
    hwnd: HWND,
    width: i32,
    height: i32,
    hdc_src: HDC,
    hdc_dest: HDC,
    hbitmap: HBITMAP,
    old_obj: HGDIOBJ,
    bits_ptr: *const u8,
}

// Safety: Owned GDI handles are exclusively accessed through `&mut self`.
unsafe impl Send for BitBltCapturer {}

impl BitBltCapturer {
    pub fn new(hwnd: isize) -> Result<Self, CaptureError> {
        let hwnd = HWND(hwnd as *mut _);
        if unsafe { !IsWindow(Some(hwnd)).as_bool() } {
            return Err(CaptureError::InvalidWindow);
        }
        if unsafe { IsIconic(hwnd).as_bool() } {
            return Err(CaptureError::WindowMinimized);
        }

        let (width, height) = client_size(hwnd)?;
        Self::create_session(hwnd, width, height)
    }

    fn create_session(hwnd: HWND, width: i32, height: i32) -> Result<Self, CaptureError> {
        let hdc_src = unsafe { GetDC(Some(hwnd)) };
        if hdc_src.is_invalid() {
            return Err(CaptureError::Api("GetDC 失败".to_string()));
        }

        let hdc_dest = unsafe { CreateCompatibleDC(Some(hdc_src)) };
        if hdc_dest.is_invalid() {
            unsafe {
                let _ = ReleaseDC(Some(hwnd), hdc_src);
            }
            return Err(CaptureError::Api("CreateCompatibleDC 失败".to_string()));
        }

        let bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                // Negative height guarantees top-down BGRA layout.
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [Default::default()],
        };

        let mut bits: *mut std::ffi::c_void = null_mut();
        let hbitmap = match unsafe {
            CreateDIBSection(Some(hdc_dest), &bmi, DIB_RGB_COLORS, &mut bits, None, 0)
        } {
            Ok(bmp) if !bmp.is_invalid() && !bits.is_null() => bmp,
            Ok(_) | Err(_) => {
                unsafe {
                    let _ = DeleteDC(hdc_dest);
                    let _ = ReleaseDC(Some(hwnd), hdc_src);
                }
                return Err(CaptureError::Api("CreateDIBSection 失败".to_string()));
            }
        };

        let old_obj = unsafe { SelectObject(hdc_dest, hbitmap.into()) };

        Ok(Self {
            hwnd,
            width,
            height,
            hdc_src,
            hdc_dest,
            hbitmap,
            old_obj,
            bits_ptr: bits as *const u8,
        })
    }

    pub fn capture(
        &mut self,
        pool: &FrameBufferPool,
    ) -> Result<(u32, u32, PooledBuffer), CaptureError> {
        if unsafe { !IsWindow(Some(self.hwnd)).as_bool() } {
            return Err(CaptureError::InvalidWindow);
        }
        if unsafe { IsIconic(self.hwnd).as_bool() } {
            return Err(CaptureError::WindowMinimized);
        }

        let (cur_w, cur_h) = client_size(self.hwnd)?;
        if cur_w != self.width || cur_h != self.height {
            let new_sess = Self::create_session(self.hwnd, cur_w, cur_h)?;
            *self = new_sess;
        }

        unsafe {
            BitBlt(
                self.hdc_dest,
                0,
                0,
                self.width,
                self.height,
                Some(self.hdc_src),
                0,
                0,
                SRCCOPY | CAPTUREBLT,
            )
            .map_err(|e| CaptureError::Api(format!("BitBlt 失败: {e}")))?;
        }

        let byte_len = (self.width as usize) * (self.height as usize) * 4;
        let mut out = pool.checkout(byte_len);
        unsafe {
            copy_nonoverlapping(self.bits_ptr, out.as_mut_slice().as_mut_ptr(), byte_len);
        }

        Ok((self.width as u32, self.height as u32, out))
    }
}

impl Drop for BitBltCapturer {
    fn drop(&mut self) {
        unsafe {
            if !self.old_obj.is_invalid() {
                let _ = SelectObject(self.hdc_dest, self.old_obj);
            }
            if !self.hbitmap.is_invalid() {
                let _ = DeleteObject(self.hbitmap.into());
            }
            if !self.hdc_dest.is_invalid() {
                let _ = DeleteDC(self.hdc_dest);
            }
            if !self.hdc_src.is_invalid() {
                let _ = ReleaseDC(Some(self.hwnd), self.hdc_src);
            }
        }
    }
}

fn client_size(hwnd: HWND) -> Result<(i32, i32), CaptureError> {
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }
        .map_err(|e| CaptureError::Api(format!("GetClientRect 失败: {e}")))?;
    let w = rect.right - rect.left;
    let h = rect.bottom - rect.top;
    if w <= 0 || h <= 0 {
        return Err(CaptureError::WindowMinimized);
    }
    Ok((w, h))
}
