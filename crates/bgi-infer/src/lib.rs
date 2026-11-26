#[cfg(feature = "paddle-ocr")]
pub mod item_icon;
#[cfg(feature = "paddle-ocr")]
pub use item_icon::{ArtifactIconMatch, ItemIconMatch, ItemIconPaths, ItemIconRecognizer};
#[cfg(feature = "paddle-ocr")]
pub mod mining;
#[cfg(feature = "paddle-ocr")]
pub mod paddle_ocr;
#[cfg(feature = "paddle-ocr")]
pub use mining::{MiningDetection, MiningDetector};

#[cfg(feature = "paddle-ocr")]
pub mod weapon_name;
#[cfg(feature = "paddle-ocr")]
pub use weapon_name::{WeaponNameCatalog, WeaponNameMatch, normalize_weapon_name};

use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("ONNX Runtime 后端未启用或 DLL 加载失败: {0}")]
    Unavailable(String),
    #[error("推理失败: {0}")]
    Runtime(String),
    #[error("I/O 失败: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "paddle-ocr")]
    #[error("OpenCV 前后处理失败: {0}")]
    Cv(#[from] bgi_cv::CvError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleUnloadPolicy {
    pub idle_timeout: Duration,
}

impl Default for IdleUnloadPolicy {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrRegion {
    pub text: String,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub regions: Vec<OcrRegion>,
}

/// Phase 2 trait boundary for PaddleOCR with idle-unload support.
pub trait OcrService: Send + Sync {
    fn recognize_bgra(
        &mut self,
        bgra: &[u8],
        width: u32,
        height: u32,
    ) -> Result<OcrResult, InferError>;
    fn unload_if_idle(&mut self);
}
