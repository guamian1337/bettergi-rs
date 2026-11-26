use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use crate::{CapturedFrame, FrameBufferPool, PixelFormat};

const MAGIC: &[u8; 8] = b"BGIRPLY1";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 48;
const MAX_FRAME_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("回放 I/O 失败: {0}")]
    Io(#[from] io::Error),
    #[error("回放文件头无效")]
    InvalidHeader,
    #[error("不支持的像素格式: {0}")]
    UnsupportedFormat(u16),
    #[error("回放帧尺寸无效: {0}x{0}", .width)]
    InvalidDimensions { width: u32, height: u32 },
    #[error("回放文件被截断")]
    Truncated,
    #[error("回放帧校验失败")]
    ChecksumMismatch,
    #[error("像素长度与图像尺寸不匹配")]
    InvalidPixelLength,
    #[error("帧过大")]
    FrameTooLarge,
    #[error("时间戳无效")]
    InvalidTimestamp,
}

#[derive(Debug, Clone)]
pub struct FrameRecord {
    pub seq: u64,
    pub captured_at: SystemTime,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub pixels: Vec<u8>,
}

impl FrameRecord {
    pub fn from_frame(
        seq: u64,
        captured_at: SystemTime,
        frame: &CapturedFrame,
    ) -> Result<Self, ReplayError> {
        let bpp = if frame.format.is_bgr24() { 3 } else { 4 };
        let expected = (frame.width as usize)
            .checked_mul(frame.height as usize)
            .and_then(|v| v.checked_mul(bpp))
            .ok_or(ReplayError::FrameTooLarge)?;
        if expected != frame.data.len() {
            return Err(ReplayError::InvalidPixelLength);
        }
        validate_dimensions(frame.width, frame.height, frame.format, expected)?;
        Ok(Self {
            seq,
            captured_at,
            width: frame.width,
            height: frame.height,
            format: frame.format,
            pixels: frame.data.as_slice().to_vec(),
        })
    }

    pub fn into_frame(self, pool: &FrameBufferPool) -> CapturedFrame {
        let mut data = pool.checkout(self.pixels.len());
        data.as_mut_slice().copy_from_slice(&self.pixels);
        CapturedFrame {
            width: self.width,
            height: self.height,
            format: self.format,
            data,
        }
    }
}

pub struct FrameRecorder {
    writer: BufWriter<File>,
    path: PathBuf,
    temporary: PathBuf,
    frames: u64,
    finished: bool,
}

impl FrameRecorder {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let path = path.as_ref().to_path_buf();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let temporary = path.with_extension(format!(
            "{}.tmp",
            path.extension().and_then(|e| e.to_str()).unwrap_or("bgi")
        ));
        Ok(Self {
            writer: BufWriter::new(File::create(&temporary)?),
            path,
            temporary,
            frames: 0,
            finished: false,
        })
    }

    pub fn record(&mut self, record: &FrameRecord) -> Result<(), ReplayError> {
        validate_dimensions(
            record.width,
            record.height,
            record.format,
            record.pixels.len(),
        )?;
        let captured_at = record
            .captured_at
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ReplayError::InvalidTimestamp)?;
        let nanos =
            u64::try_from(captured_at.as_nanos()).map_err(|_| ReplayError::InvalidTimestamp)?;
        let format = match record.format {
            PixelFormat::Bgr24 => 1u16,
            PixelFormat::Bgra32 => 2,
        };
        let mut header = [0u8; HEADER_LEN];
        header[..8].copy_from_slice(MAGIC);
        header[8..10].copy_from_slice(&VERSION.to_le_bytes());
        header[10..12].copy_from_slice(&format.to_le_bytes());
        header[12..16].copy_from_slice(&record.width.to_le_bytes());
        header[16..20].copy_from_slice(&record.height.to_le_bytes());
        header[20..24].copy_from_slice(&(record.pixels.len() as u32).to_le_bytes());
        header[24..28].copy_from_slice(&checksum(&record.pixels).to_le_bytes());
        header[28..36].copy_from_slice(&nanos.to_le_bytes());
        header[36..44].copy_from_slice(&record.seq.to_le_bytes());
        header[44..48].copy_from_slice(&0u32.to_le_bytes());
        self.writer.write_all(&header)?;
        self.writer.write_all(&record.pixels)?;
        self.frames += 1;
        Ok(())
    }

    pub fn record_frame(
        &mut self,
        seq: u64,
        captured_at: SystemTime,
        frame: &CapturedFrame,
    ) -> Result<(), ReplayError> {
        self.record(&FrameRecord::from_frame(seq, captured_at, frame)?)
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn finish(mut self) -> Result<PathBuf, ReplayError> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        replace_file(&self.temporary, &self.path)?;
        self.finished = true;
        Ok(self.path.clone())
    }
}

impl Drop for FrameRecorder {
    fn drop(&mut self) {
        if !self.finished {
            let _ = fs::remove_file(&self.temporary);
        }
    }
}

pub struct FrameReplay {
    reader: BufReader<File>,
    path: PathBuf,
    remaining_bytes: u64,
    next_seq: u64,
    finished: bool,
}

impl FrameReplay {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let remaining_bytes = file.metadata()?.len();
        Ok(Self {
            reader: BufReader::new(file),
            path,
            remaining_bytes,
            next_seq: 0,
            finished: false,
        })
    }

    pub fn next_frame(&mut self) -> Result<Option<FrameRecord>, ReplayError> {
        if self.finished {
            return Ok(None);
        }
        if self.remaining_bytes == 0 {
            self.finished = true;
            return Ok(None);
        }
        if self.remaining_bytes < HEADER_LEN as u64 {
            return Err(ReplayError::Truncated);
        }
        let mut header = [0u8; HEADER_LEN];
        self.reader.read_exact(&mut header)?;
        self.remaining_bytes -= HEADER_LEN as u64;
        if &header[..8] != MAGIC
            || u16::from_le_bytes([header[8], header[9]]) != VERSION
            || header[44..48] != [0; 4]
        {
            return Err(ReplayError::InvalidHeader);
        }
        let format_id = u16::from_le_bytes([header[10], header[11]]);
        let format = match format_id {
            1 => PixelFormat::Bgr24,
            2 => PixelFormat::Bgra32,
            other => return Err(ReplayError::UnsupportedFormat(other)),
        };
        let width = u32::from_le_bytes(header[12..16].try_into().unwrap());
        let height = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let len = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;
        let expected_checksum = u32::from_le_bytes(header[24..28].try_into().unwrap());
        let nanos = u64::from_le_bytes(header[28..36].try_into().unwrap());
        let seq = u64::from_le_bytes(header[36..44].try_into().unwrap());
        if seq != self.next_seq {
            return Err(ReplayError::InvalidHeader);
        }
        validate_dimensions(width, height, format, len)?;
        if len as u64 > self.remaining_bytes {
            return Err(ReplayError::Truncated);
        }
        let mut pixels = vec![0u8; len];
        self.reader.read_exact(&mut pixels)?;
        self.remaining_bytes -= len as u64;
        self.next_seq = seq.checked_add(1).ok_or(ReplayError::InvalidHeader)?;
        if checksum(&pixels) != expected_checksum {
            return Err(ReplayError::ChecksumMismatch);
        }
        Ok(Some(FrameRecord {
            seq,
            captured_at: UNIX_EPOCH
                .checked_add(std::time::Duration::from_nanos(nanos))
                .ok_or(ReplayError::InvalidTimestamp)?,
            width,
            height,
            format,
            pixels,
        }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn validate_dimensions(
    width: u32,
    height: u32,
    format: PixelFormat,
    len: usize,
) -> Result<(), ReplayError> {
    if width == 0 || height == 0 {
        return Err(ReplayError::InvalidDimensions { width, height });
    }
    if width > 32_768 || height > 32_768 || len > MAX_FRAME_BYTES {
        return Err(ReplayError::FrameTooLarge);
    }
    let bpp = if format.is_bgr24() { 3 } else { 4 };
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(bpp))
        .ok_or(ReplayError::FrameTooLarge)?;
    if len != expected {
        return Err(ReplayError::InvalidPixelLength);
    }
    Ok(())
}

fn checksum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c9dc5u32, |hash, byte| {
        (hash ^ u32::from(*byte)).wrapping_mul(0x01000193)
    })
}

fn replace_file(from: &Path, to: &Path) -> Result<(), ReplayError> {
    if let Err(error) = fs::rename(from, to) {
        if !to.exists() {
            return Err(error.into());
        }
        fs::remove_file(to)?;
        fs::rename(from, to)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(seq: u64) -> FrameRecord {
        FrameRecord {
            seq,
            captured_at: UNIX_EPOCH + std::time::Duration::from_secs(3),
            width: 2,
            height: 1,
            format: PixelFormat::Bgr24,
            pixels: vec![1, 2, 3, 4, 5, 6],
        }
    }

    #[test]
    fn records_and_replays_multiple_formats() {
        let path = std::env::temp_dir().join(format!("bgi-replay-{}.bgi", std::process::id()));
        let mut writer = FrameRecorder::create(&path).unwrap();
        let mut bgra = record(1);
        bgra.format = PixelFormat::Bgra32;
        bgra.pixels.extend_from_slice(&[7, 8]);
        writer.record(&record(0)).unwrap();
        writer.record(&bgra).unwrap();
        assert_eq!(writer.frames(), 2);
        writer.finish().unwrap();
        let mut reader = FrameReplay::open(&path).unwrap();
        let first = reader.next_frame().unwrap().unwrap();
        assert_eq!(first.pixels, record(0).pixels);
        let second = reader.next_frame().unwrap().unwrap();
        assert_eq!(second.format, PixelFormat::Bgra32);
        assert!(reader.next_frame().unwrap().is_none());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn detects_truncation_and_corruption() {
        let path =
            std::env::temp_dir().join(format!("bgi-replay-corrupt-{}.bgi", std::process::id()));
        let mut writer = FrameRecorder::create(&path).unwrap();
        writer.record(&record(0)).unwrap();
        let output = writer.finish().unwrap();
        let bytes = fs::read(&output).unwrap();
        fs::write(&output, &bytes[..bytes.len() - 1]).unwrap();
        assert!(matches!(
            FrameReplay::open(&output).unwrap().next_frame(),
            Err(ReplayError::Truncated)
        ));
        let mut corrupted = bytes;
        corrupted[HEADER_LEN] ^= 1;
        fs::write(&output, corrupted).unwrap();
        assert!(matches!(
            FrameReplay::open(&output).unwrap().next_frame(),
            Err(ReplayError::ChecksumMismatch)
        ));
        fs::remove_file(output).unwrap();
    }

    #[test]
    fn rejects_invalid_pixel_lengths() {
        let mut invalid = record(0);
        invalid.pixels.pop();
        assert!(matches!(
            validate_dimensions(
                invalid.width,
                invalid.height,
                invalid.format,
                invalid.pixels.len()
            ),
            Err(ReplayError::InvalidPixelLength)
        ));
        assert!(validate_dimensions(2, 1, PixelFormat::Bgr24, 6).is_ok());
        assert!(validate_dimensions(0, 1, PixelFormat::Bgr24, 0).is_err());
    }
}
