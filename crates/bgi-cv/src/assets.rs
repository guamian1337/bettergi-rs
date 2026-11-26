use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AssetError {
    #[error("资源索引读取失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("资源索引解析失败: {0}")]
    Json(#[from] serde_json::Error),
    #[error("资源路径不安全: {0}")]
    UnsafePath(String),
    #[error("资源不存在: {0}")]
    Missing(PathBuf),
    #[error("资源 {0:?} 不是文件")]
    NotFile(PathBuf),
    #[error("索引结构无效: {0}")]
    InvalidIndex(String),
    #[error("资源 {0:?} 不是普通文件")]
    ReparsePoint(PathBuf),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "snake_case")]
pub struct LocalePack {
    pub templates: BTreeMap<String, String>,
    pub ocr: OcrAssets,
    pub models: BTreeMap<String, String>,
    pub text: BTreeMap<String, Vec<String>>,
    pub maps: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "snake_case")]
pub struct OcrAssets {
    pub det: String,
    pub rec: String,
    pub dict: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct NormRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl NormRect {
    pub fn pixels(self, image_width: u32, image_height: u32) -> Option<(u32, u32, u32, u32)> {
        if !self.x.is_finite()
            || !self.y.is_finite()
            || !self.width.is_finite()
            || !self.height.is_finite()
            || self.x < 0.0
            || self.y < 0.0
            || self.width <= 0.0
            || self.height <= 0.0
            || self.x + self.width > 1.000_001
            || self.y + self.height > 1.000_001
        {
            return None;
        }
        let right = ((self.x + self.width) * image_width as f64).round() as u32;
        let bottom = ((self.y + self.height) * image_height as f64).round() as u32;
        let x = (self.x * image_width as f64).round() as u32;
        let y = (self.y * image_height as f64).round() as u32;
        let width = right.saturating_sub(x);
        let height = bottom.saturating_sub(y);
        (width > 0 && height > 0).then_some((x, y, width, height))
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Layout {
    pub base_width: u32,
    pub base_height: u32,
    pub rois: BTreeMap<String, NormRect>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, rename_all = "snake_case")]
pub struct AssetIndex {
    pub schema: u32,
    pub default_locale: String,
    pub locales: BTreeMap<String, LocalePack>,
    pub layouts: BTreeMap<String, Layout>,
}

impl AssetIndex {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AssetError> {
        let index: Self = serde_json::from_slice(&fs::read(path)?)?;
        index.validate()?;
        Ok(index)
    }

    pub fn validate(&self) -> Result<(), AssetError> {
        if self.schema != 1 {
            return Err(AssetError::InvalidIndex(format!(
                "unsupported schema {}",
                self.schema
            )));
        }
        if self.locales.is_empty() || !self.locales.contains_key(&self.default_locale) {
            return Err(AssetError::InvalidIndex("default locale is missing".into()));
        }
        if self.default_locale.trim().is_empty() {
            return Err(AssetError::InvalidIndex("default_locale is empty".into()));
        }
        for (locale, pack) in &self.locales {
            if !pack.ocr.det.is_empty() {
                validate_relative_path(&pack.ocr.det)?;
                validate_relative_path(&pack.ocr.rec)?;
                validate_relative_path(&pack.ocr.dict)?;
            }
            for path in pack.templates.values().chain(pack.maps.values()) {
                validate_relative_path(path)?;
            }
            for path in pack.models.values() {
                validate_relative_path(path)?;
            }
            for aliases in pack.text.values() {
                if aliases.is_empty() || aliases.iter().any(|s| s.trim().is_empty()) {
                    return Err(AssetError::InvalidIndex(format!(
                        "empty text alias in {locale}"
                    )));
                }
            }
        }
        for (name, layout) in &self.layouts {
            if layout.base_width == 0 || layout.base_height == 0 {
                return Err(AssetError::InvalidIndex(format!(
                    "invalid base size for {name}"
                )));
            }
        }
        Ok(())
    }

    fn locale<'a>(&'a self, requested: &str) -> &'a LocalePack {
        let mut current = requested.replace('_', "-");
        loop {
            if let Some(pack) = self.locales.get(&current) {
                return pack;
            }
            let Some((head, _)) = current.split_once('-') else {
                break;
            };
            current = head.to_string();
        }
        self.locales.get(&self.default_locale).unwrap_or(
            self.locales
                .values()
                .next()
                .expect("validated index has a locale"),
        )
    }
}

#[derive(Debug, Clone)]
pub struct AssetCatalog {
    root: PathBuf,
    index: AssetIndex,
    locale: String,
}

impl AssetCatalog {
    pub fn load(root: impl AsRef<Path>, locale: impl Into<String>) -> Result<Self, AssetError> {
        let root = fs::canonicalize(root)?;
        let index = AssetIndex::load(root.join("index.json"))?;
        if index.locales.is_empty() {
            return Err(AssetError::InvalidIndex(
                "at least one locale is required".into(),
            ));
        }
        let locale = locale.into();
        Ok(Self {
            root,
            index,
            locale,
        })
    }

    pub fn embedded_fallback(root: impl AsRef<Path>) -> Result<Self, AssetError> {
        let root = fs::canonicalize(root)?;
        let pack = LocalePack {
            ocr: OcrAssets {
                det: r"Model\PaddleOCR\Det\V5\PP-OCRv5_mobile_det_infer\slim.onnx".into(),
                rec: r"Model\PaddleOCR\Rec\V5\PP-OCRv5_mobile_rec_infer\slim.onnx".into(),
                dict: r"Model\PaddleOCR\Rec\V5\PP-OCRv5_mobile_rec_infer\inference.yml".into(),
            },
            maps: BTreeMap::from([("Teyvat".into(), r"Map\Teyvat".into())]),
            models: BTreeMap::from([("mine".into(), r"Model\Mine\bgi_mine.onnx".into())]),
            ..Default::default()
        };
        let mut locales = BTreeMap::new();
        locales.insert("zh-CN".into(), pack);
        let index = AssetIndex {
            schema: 1,
            default_locale: "zh-CN".into(),
            locales,
            layouts: BTreeMap::new(),
        };
        Ok(Self {
            root,
            index,
            locale: "zh-CN".into(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn locale_pack(&self) -> &LocalePack {
        self.index.locale(&self.locale)
    }

    pub fn resolve(&self, relative: impl AsRef<Path>) -> Result<PathBuf, AssetError> {
        let relative = relative.as_ref();
        validate_relative_path(relative)?;
        let resolved = self.root.join(relative);
        ensure_contained_file(&resolved, &self.root)?;
        Ok(resolved)
    }

    pub fn template(&self, id: &str) -> Result<PathBuf, AssetError> {
        let relative =
            self.locale_pack().templates.get(id).ok_or_else(|| {
                AssetError::InvalidIndex(format!("template {id:?} is not indexed"))
            })?;
        self.resolve(relative)
    }

    pub fn map(&self, id: &str) -> Result<PathBuf, AssetError> {
        let relative = self
            .locale_pack()
            .maps
            .get(id)
            .ok_or_else(|| AssetError::InvalidIndex(format!("map {id:?} is not indexed")))?;
        validate_relative_path(relative)?;
        let path = self.root.join(relative);
        ensure_contained(&path, false, &self.root)?;
        Ok(path)
    }

    pub fn model(&self, id: &str) -> Result<PathBuf, AssetError> {
        let relative = self
            .locale_pack()
            .models
            .get(id)
            .ok_or_else(|| AssetError::InvalidIndex(format!("model {id:?} is not indexed")))?;
        self.resolve(relative)
    }

    pub fn ocr_paths(&self) -> Result<(PathBuf, PathBuf, PathBuf), AssetError> {
        let ocr = &self.locale_pack().ocr;
        if ocr.det.is_empty() || ocr.rec.is_empty() || ocr.dict.is_empty() {
            return Err(AssetError::InvalidIndex("OCR paths are incomplete".into()));
        }
        Ok((
            self.resolve(&ocr.det)?,
            self.resolve(&ocr.rec)?,
            self.resolve(&ocr.dict)?,
        ))
    }

    pub fn text_aliases(&self, id: &str) -> &[String] {
        self.locale_pack()
            .text
            .get(id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn roi(
        &self,
        layout: &str,
        id: &str,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32, u32, u32)> {
        self.index
            .layouts
            .get(layout)?
            .rois
            .get(id)?
            .pixels(width, height)
    }
}

pub fn validate_relative_path(path: impl AsRef<Path>) -> Result<(), AssetError> {
    let path = path.as_ref();
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(AssetError::UnsafePath(path.display().to_string()));
    }
    Ok(())
}

fn ensure_contained_file(path: &Path, root: &Path) -> Result<(), AssetError> {
    ensure_contained(path, true, root)
}

fn ensure_contained(path: &Path, require_file: bool, root: &Path) -> Result<(), AssetError> {
    let expected_type = require_file;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        let metadata =
            fs::symlink_metadata(path).map_err(|_| AssetError::Missing(path.to_path_buf()))?;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(AssetError::ReparsePoint(path.to_path_buf()));
        }
        if (expected_type && !metadata.is_file()) || (!expected_type && !metadata.is_dir()) {
            return Err(AssetError::NotFile(path.to_path_buf()));
        }
    }
    #[cfg(not(windows))]
    {
        let metadata =
            fs::symlink_metadata(path).map_err(|_| AssetError::Missing(path.to_path_buf()))?;
        if metadata.file_type().is_symlink()
            || (expected_type && !metadata.is_file())
            || (!expected_type && !metadata.is_dir())
        {
            return Err(AssetError::NotFile(path.to_path_buf()));
        }
    }
    let mut current = path;
    while let Some(parent) = current.parent() {
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            let metadata = fs::symlink_metadata(current)
                .map_err(|_| AssetError::Missing(path.to_path_buf()))?;
            if metadata.file_attributes() & 0x400 != 0 {
                return Err(AssetError::ReparsePoint(path.to_path_buf()));
            }
        }
        #[cfg(not(windows))]
        if fs::symlink_metadata(current)
            .map_err(|_| AssetError::Missing(path.to_path_buf()))?
            .file_type()
            .is_symlink()
        {
            return Err(AssetError::ReparsePoint(path.to_path_buf()));
        }
        if current == root {
            break;
        }
        current = parent;
    }
    let canonical_root = fs::canonicalize(root)?;
    let canonical = fs::canonicalize(path)?;
    if !canonical.starts_with(canonical_root) {
        return Err(AssetError::UnsafePath(path.display().to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> AssetIndex {
        AssetIndex {
            schema: 1,
            default_locale: "zh-CN".into(),
            locales: BTreeMap::from([
                (
                    "zh-CN".into(),
                    LocalePack {
                        templates: BTreeMap::from([("ui".into(), "zh/ui.png".into())]),
                        models: BTreeMap::from([("mine".into(), "zh/mine.onnx".into())]),
                        text: BTreeMap::from([("start".into(), vec!["开始".into()])]),
                        ..Default::default()
                    },
                ),
                (
                    "en".into(),
                    LocalePack {
                        templates: BTreeMap::from([("ui".into(), "en/ui.png".into())]),
                        ..Default::default()
                    },
                ),
            ]),
            layouts: BTreeMap::from([(
                "default".into(),
                Layout {
                    base_width: 1920,
                    base_height: 1080,
                    rois: BTreeMap::from([(
                        "minimap".into(),
                        NormRect {
                            x: 0.4,
                            y: 0.4,
                            width: 0.2,
                            height: 0.2,
                        },
                    )]),
                },
            )]),
        }
    }

    #[test]
    fn locale_falls_back_by_language_and_default() {
        let index = index();
        assert_eq!(index.locale("en-GB").templates["ui"], "en/ui.png");
        assert_eq!(index.locale("fr-FR").templates["ui"], "zh/ui.png");
    }

    #[test]
    fn normalized_roi_scales_and_rejects_invalid_values() {
        let rect = NormRect {
            x: 0.4,
            y: 0.4,
            width: 0.2,
            height: 0.2,
        };
        assert_eq!(rect.pixels(1280, 720), Some((512, 288, 256, 144)));
        assert_eq!(NormRect { width: 0.0, ..rect }.pixels(1920, 1080), None);
        assert_eq!(NormRect { x: 0.9, ..rect }.pixels(1920, 1080), None);
    }

    #[test]
    fn unsafe_relative_paths_are_rejected() {
        for path in ["../x", "/x", "a/../../x", "", r"C:\x"] {
            assert!(validate_relative_path(Path::new(path)).is_err(), "{path}");
        }
        assert!(validate_relative_path(Path::new(r"Model\ocr\det.onnx")).is_ok());
    }

    #[test]
    fn catalog_loads_assets_and_rejects_escape() {
        let dir = std::env::temp_dir().join(format!("bgi-assets-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("en")).unwrap();
        fs::write(dir.join("en/ui.png"), b"png").unwrap();
        fs::write(dir.join("en/mine.onnx"), b"model").unwrap();
        let mut value = index();
        value
            .locales
            .get_mut("en")
            .unwrap()
            .templates
            .insert("ui".into(), "en/ui.png".into());
        value
            .locales
            .get_mut("en")
            .unwrap()
            .models
            .insert("mine".into(), "en/mine.onnx".into());
        fs::write(dir.join("index.json"), serde_json::to_vec(&value).unwrap()).unwrap();
        let catalog = AssetCatalog::load(&dir, "en-US").unwrap();
        assert_eq!(
            catalog.template("ui").unwrap(),
            fs::canonicalize(dir.join("en/ui.png")).unwrap()
        );
        assert_eq!(
            catalog.model("mine").unwrap(),
            fs::canonicalize(dir.join("en/mine.onnx")).unwrap()
        );
        assert!(catalog.resolve("../outside").is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn empty_locale_set_is_rejected() {
        let mut value = index();
        value.locales.clear();
        assert!(matches!(value.validate(), Err(AssetError::InvalidIndex(_))));
    }
}
