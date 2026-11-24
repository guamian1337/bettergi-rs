#[cfg(feature = "opencv411")]
pub mod matchers;
#[cfg(feature = "opencv411")]
pub mod orientation;
#[cfg(feature = "opencv411")]
pub mod sift_map;
#[cfg(feature = "opencv411")]
pub mod teyvat_minimap;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CvError {
    #[error("OpenCV 4.11 后端未启用（请使用 --features opencv411）")]
    BackendDisabled,
    #[error("图像参数无效: {0}")]
    InvalidImage(String),
    #[error("视觉匹配失败: {0}")]
    MatchFailed(String),
    #[error("I/O 失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON 解析失败: {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "opencv411")]
    #[error("OpenCV 错误: {0}")]
    OpenCv(String),
}

#[cfg(feature = "opencv411")]
impl From<opencv::Error> for CvError {
    fn from(value: opencv::Error) -> Self {
        Self::OpenCv(value.message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchPoint {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

pub fn opencv_feature_enabled() -> bool {
    cfg!(feature = "opencv411")
}
