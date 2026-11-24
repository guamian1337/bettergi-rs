use std::ptr::copy_nonoverlapping;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D11_SRV_DIMENSION_TEXTURE2D, ID3DBlob};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_UNORDERED_ACCESS, D3D11_BOX, D3D11_BUFFER_DESC,
    D3D11_BUFFER_UAV, D3D11_CPU_ACCESS_READ, D3D11_CPU_ACCESS_WRITE, D3D11_MAP_READ,
    D3D11_MAP_WRITE_DISCARD, D3D11_MAPPED_SUBRESOURCE, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
    D3D11_SHADER_RESOURCE_VIEW_DESC, D3D11_SHADER_RESOURCE_VIEW_DESC_0, D3D11_TEX2D_SRV,
    D3D11_TEXTURE2D_DESC, D3D11_UAV_DIMENSION_BUFFER, D3D11_UNORDERED_ACCESS_VIEW_DESC,
    D3D11_UNORDERED_ACCESS_VIEW_DESC_0, D3D11_USAGE_DEFAULT, D3D11_USAGE_DYNAMIC,
    D3D11_USAGE_STAGING, ID3D11Buffer, ID3D11ComputeShader, ID3D11Device, ID3D11DeviceContext,
    ID3D11ShaderResourceView, ID3D11Texture2D, ID3D11UnorderedAccessView,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_WAIT_TIMEOUT;
use windows::core::s;

use crate::CaptureError;

/// Compute Shader packing cropped BGRA8 pixels directly into tightly packed BGR24 (12 bytes / 4 pixels).
/// Reduces PCIe readback traffic by 25% and eliminates CPU `COLOR_BGRA2BGR` conversion.
const PACK_BGR24_CS_SRC: &str = r#"
cbuffer cb : register(b0) { uint4 params; } // x=crop_x, y=crop_y, z=out_w, w=total_pixels
Texture2D<float4> src : register(t0);
RWStructuredBuffer<uint> outBuf : register(u0);

[numthreads(64, 1, 1)]
void main_cs(uint3 dt : SV_DispatchThreadID)
{
    uint p = dt.x * 4;
    if (p >= params.w) return;

    uint bc[4];
    uint gc[4];
    uint rc[4];
    [unroll]
    for (uint k = 0; k < 4; ++k)
    {
        uint idx = min(p + k, params.w - 1);
        uint x = params.x + (idx % params.z);
        uint y = params.y + (idx / params.z);
        float4 px = src.Load(int3(x, y, 0));
        bc[k] = (uint)(saturate(px.z) * 255.0f + 0.5f);
        gc[k] = (uint)(saturate(px.y) * 255.0f + 0.5f);
        rc[k] = (uint)(saturate(px.x) * 255.0f + 0.5f);
    }

    uint o = (p / 4) * 3;
    outBuf[o + 0] = bc[0] | (gc[0] << 8) | (rc[0] << 16) | (bc[1] << 24);
    outBuf[o + 1] = gc[1] | (rc[1] << 8) | (bc[2] << 16) | (gc[2] << 24);
    outBuf[o + 2] = rc[2] | (bc[3] << 8) | (gc[3] << 16) | (rc[3] << 24);
}
"#;

pub(crate) struct GpuBgr24Packer {
    cs: Option<(ID3D11ComputeShader, ID3D11Buffer)>,
    pack_gpu: Option<ID3D11Buffer>,
    pack_uav: Option<ID3D11UnorderedAccessView>,
    pack_staging: Option<ID3D11Buffer>,
    pack_bytes: u32,
    fallback_staging: Option<(u32, u32, ID3D11Texture2D)>,
}

impl GpuBgr24Packer {
    pub fn new(device: &ID3D11Device) -> Self {
        let cs = compile_pack_cs(device).ok();
        Self {
            cs,
            pack_gpu: None,
            pack_uav: None,
            pack_staging: None,
            pack_bytes: 0,
            fallback_staging: None,
        }
    }

    pub fn is_gpu_cs_active(&self) -> bool {
        self.cs.is_some()
    }

    pub fn reset_buffers(&mut self) {
        self.pack_uav = None;
        self.pack_gpu = None;
        self.pack_staging = None;
        self.pack_bytes = 0;
        self.fallback_staging = None;
    }

    /// Stage 1 (GPU side): encodes `(crop_x, crop_y, out_w, out_h)` from `src_tex` to a staging buffer
    /// and flushes the D3D11 queue so the caller can immediately release/close `src_tex`.
    pub fn stage_cropped_bgr24(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        src_tex: &ID3D11Texture2D,
        src_format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        crop_x: u32,
        crop_y: u32,
        out_w: u32,
        out_h: u32,
    ) -> Result<bool, CaptureError> {
        if let Some((cs, cb)) = self.cs.clone() {
            let total_pixels = out_w.saturating_mul(out_h).max(1);
            let aligned_bytes = ((total_pixels + 3) / 4) * 12;
            self.ensure_pack_buffers(device, aligned_bytes)?;

            let mut srv_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: src_format,
                ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                    },
                },
            };
            let mut srv: Option<ID3D11ShaderResourceView> = None;
            unsafe {
                device
                    .CreateShaderResourceView(src_tex, Some(&mut srv_desc as *mut _), Some(&mut srv))
                    .map_err(|e| CaptureError::Api(format!("CreateShaderResourceView(BGR24) 失败: {e}")))?;
            }
            let srv = srv.ok_or_else(|| CaptureError::Api("BGR24 SRV 为空".into()))?;

            let params: [u32; 4] = [crop_x, crop_y, out_w, total_pixels];
            let mut cb_map = D3D11_MAPPED_SUBRESOURCE::default();
            unsafe {
                context
                    .Map(&cb, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut cb_map))
                    .map_err(|e| CaptureError::Api(format!("Map BGR24 cb 失败: {e}")))?;
                copy_nonoverlapping(params.as_ptr() as *const u8, cb_map.pData as *mut u8, 16);
                context.Unmap(&cb, 0);
            }

            let pixel_groups = (total_pixels + 3) / 4;
            let thread_groups = (pixel_groups + 63) / 64;
            let gpu_buf = self.pack_gpu.as_ref().unwrap();
            let uav = self.pack_uav.as_ref().unwrap();
            let staging_buf = self.pack_staging.as_ref().unwrap();

            unsafe {
                context.CSSetShader(&cs, None);
                context.CSSetShaderResources(0, Some(&[Some(srv)]));
                context.CSSetUnorderedAccessViews(0, 1, Some(&Some(uav.clone()) as *const _), None);
                context.CSSetConstantBuffers(0, Some(&[Some(cb)]));
                context.Dispatch(thread_groups, 1, 1);
                context.CSSetShaderResources(0, Some(&[None]));
                context.CSSetUnorderedAccessViews(0, 1, Some(&None as *const _), None);
                context.CopyResource(staging_buf, gpu_buf);
                // Ensure GPU commands reading WGC/DWM shared texture are submitted before closing frame.
                context.Flush();
            }
            return Ok(true);
        }

        // Fallback: CopySubresourceRegion BGRA to staging texture, then Flush before closing frame.
        ensure_bgra_staging(device, &mut self.fallback_staging, out_w, out_h)?;
        let staging_tex = &self.fallback_staging.as_ref().unwrap().2;
        let src_box = D3D11_BOX {
            left: crop_x,
            top: crop_y,
            front: 0,
            right: crop_x + out_w,
            bottom: crop_y + out_h,
            back: 1,
        };
        unsafe {
            context.CopySubresourceRegion(staging_tex, 0, 0, 0, 0, src_tex, 0, Some(&src_box));
            context.Flush();
        }
        Ok(false)
    }

    /// Stage 2 (CPU side): maps the staging resource after `src_tex` has already been released.
    pub fn map_staged_bgr24(
        &self,
        context: &ID3D11DeviceContext,
        used_cs: bool,
        out_w: u32,
        out_h: u32,
        dst: &mut Vec<u8>,
    ) -> Result<(), CaptureError> {
        let exact_bytes = (out_w as usize) * (out_h as usize) * 3;
        dst.resize(exact_bytes, 0);

        if used_cs {
            let staging_buf = self
                .pack_staging
                .as_ref()
                .ok_or_else(|| CaptureError::Api("BGR24 staging buffer 未初始化".into()))?;
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            unsafe {
                context
                    .Map(staging_buf, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                    .map_err(map_err_to_capture)?;
                copy_nonoverlapping(mapped.pData as *const u8, dst.as_mut_ptr(), exact_bytes);
                context.Unmap(staging_buf, 0);
            }
            return Ok(());
        }

        let staging_tex = &self
            .fallback_staging
            .as_ref()
            .ok_or_else(|| CaptureError::Api("BGRA fallback staging 未初始化".into()))?
            .2;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            context
                .Map(staging_tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(map_err_to_capture)?;
        }
        let src_pitch = mapped.RowPitch as usize;
        let src_base = mapped.pData as *const u8;
        let row_bgr_bytes = (out_w as usize) * 3;
        for y in 0..(out_h as usize) {
            let src_row = unsafe { std::slice::from_raw_parts(src_base.add(y * src_pitch), (out_w as usize) * 4) };
            let dst_row = &mut dst[y * row_bgr_bytes..(y + 1) * row_bgr_bytes];
            for x in 0..(out_w as usize) {
                let s = x * 4;
                let d = x * 3;
                dst_row[d] = src_row[s];
                dst_row[d + 1] = src_row[s + 1];
                dst_row[d + 2] = src_row[s + 2];
            }
        }
        unsafe {
            context.Unmap(staging_tex, 0);
        }
        Ok(())
    }

    fn ensure_pack_buffers(&mut self, device: &ID3D11Device, bytes: u32) -> Result<(), CaptureError> {
        if self.pack_gpu.is_some() && self.pack_bytes == bytes {
            return Ok(());
        }
        self.pack_uav = None;
        self.pack_gpu = None;
        self.pack_staging = None;

        let gpu_desc = D3D11_BUFFER_DESC {
            ByteWidth: bytes,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
            StructureByteStride: 4,
        };
        let mut gpu_buf: Option<ID3D11Buffer> = None;
        unsafe {
            device
                .CreateBuffer(&gpu_desc, None, Some(&mut gpu_buf))
                .map_err(|e| CaptureError::Api(format!("CreateBuffer(BGR24 GPU) 失败: {e}")))?;
        }
        let gpu_buf = gpu_buf.unwrap();

        let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
            Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D11_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: bytes / 4,
                    Flags: 0,
                },
            },
        };
        let mut uav: Option<ID3D11UnorderedAccessView> = None;
        unsafe {
            device
                .CreateUnorderedAccessView(&gpu_buf, Some(&uav_desc), Some(&mut uav))
                .map_err(|e| CaptureError::Api(format!("CreateUnorderedAccessView(BGR24) 失败: {e}")))?;
        }

        let staging_desc = D3D11_BUFFER_DESC {
            ByteWidth: bytes,
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
            StructureByteStride: 4,
        };
        let mut staging_buf: Option<ID3D11Buffer> = None;
        unsafe {
            device
                .CreateBuffer(&staging_desc, None, Some(&mut staging_buf))
                .map_err(|e| CaptureError::Api(format!("CreateBuffer(BGR24 Staging) 失败: {e}")))?;
        }

        self.pack_gpu = Some(gpu_buf);
        self.pack_uav = uav;
        self.pack_staging = staging_buf;
        self.pack_bytes = bytes;
        Ok(())
    }
}

fn compile_pack_cs(
    device: &ID3D11Device,
) -> Result<(ID3D11ComputeShader, ID3D11Buffer), CaptureError> {
    let mut blob: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            PACK_BGR24_CS_SRC.as_ptr() as *const _,
            PACK_BGR24_CS_SRC.len(),
            None,
            None,
            None,
            s!("main_cs"),
            s!("cs_5_0"),
            0,
            0,
            &mut blob,
            None,
        )
        .map_err(|e| CaptureError::Api(format!("D3DCompile(BGR24 CS) 失败: {e}")))?;
    }
    let blob = blob.ok_or_else(|| CaptureError::Api("D3DCompile(BGR24 CS) 返回空".into()))?;
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())
    };
    let mut cs: Option<ID3D11ComputeShader> = None;
    unsafe {
        device
            .CreateComputeShader(bytes, None, Some(&mut cs))
            .map_err(|e| CaptureError::Api(format!("CreateComputeShader(BGR24) 失败: {e}")))?;
    }

    let cb_desc = D3D11_BUFFER_DESC {
        ByteWidth: 16,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut cb: Option<ID3D11Buffer> = None;
    unsafe {
        device
            .CreateBuffer(&cb_desc, None, Some(&mut cb))
            .map_err(|e| CaptureError::Api(format!("CreateBuffer(BGR24 cb) 失败: {e}")))?;
    }
    Ok((cs.unwrap(), cb.unwrap()))
}

fn ensure_bgra_staging(
    device: &ID3D11Device,
    slot: &mut Option<(u32, u32, ID3D11Texture2D)>,
    width: u32,
    height: u32,
) -> Result<(), CaptureError> {
    if let Some((w, h, _)) = slot
        && *w == width
        && *h == height
    {
        return Ok(());
    }
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
    };
    let mut tex = None;
    unsafe {
        device
            .CreateTexture2D(&desc, None, Some(&mut tex))
            .map_err(|e| CaptureError::Api(format!("CreateTexture2D(staging) 失败: {e}")))?;
    }
    *slot = Some((width, height, tex.unwrap()));
    Ok(())
}

fn map_err_to_capture(e: windows::core::Error) -> CaptureError {
    if e.code() == DXGI_ERROR_WAIT_TIMEOUT {
        CaptureError::FrameNotReady
    } else {
        CaptureError::Api(format!("Map staging 失败: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::{
        D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_1,
    };
    use windows::Win32::Graphics::Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
        D3D11_SUBRESOURCE_DATA, D3D11CreateDevice,
    };

    #[test]
    fn gpu_cs_packs_bgra_texture_crop_to_tight_bgr24() {
        let mut dev = None;
        let mut ctx = None;
        if unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_1]),
                D3D11_SDK_VERSION,
                Some(&mut dev),
                None,
                Some(&mut ctx),
            )
        }
        .is_err()
        {
            return;
        }
        let dev = dev.unwrap();
        let ctx = ctx.unwrap();
        let mut packer = GpuBgr24Packer::new(&dev);
        assert!(packer.is_gpu_cs_active());

        // 4x4 BGRA source texture; crop 3x2 starting at (1, 1) (tests non-multiple-of-4 pixel count = 6).
        let mut pixels = Vec::with_capacity(4 * 4 * 4);
        for y in 0u8..4 {
            for x in 0u8..4 {
                pixels.extend_from_slice(&[x * 30 + y * 5, y * 40 + x * 7, 200 - x * 20 - y * 10, 128]);
            }
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 4,
            Height: 4,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr() as *const _,
            SysMemPitch: 16,
            SysMemSlicePitch: 0,
        };
        let mut tex = None;
        unsafe {
            dev.CreateTexture2D(&desc, Some(&init_data), Some(&mut tex))
                .unwrap();
        }
        let tex = tex.unwrap();

        let used_cs = packer
            .stage_cropped_bgr24(&dev, &ctx, &tex, DXGI_FORMAT_B8G8R8A8_UNORM, 1, 1, 3, 2)
            .unwrap();
        assert!(used_cs);

        let mut out_bgr = Vec::new();
        packer
            .map_staged_bgr24(&ctx, used_cs, 3, 2, &mut out_bgr)
            .unwrap();
        assert_eq!(out_bgr.len(), 3 * 2 * 3);

        let mut expected = Vec::new();
        for y in 1..3 {
            for x in 1..4 {
                let s = (y * 4 + x) * 4;
                expected.extend_from_slice(&pixels[s..s + 3]);
            }
        }
        assert_eq!(out_bgr, expected);
    }
}
