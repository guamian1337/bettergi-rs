use std::path::Path;

use opencv::core::{self, Mat, MatTraitConst, Point, Rect, Size, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{self, COLOR_BGR2GRAY, INTER_AREA, INTER_CUBIC, TM_CCOEFF_NORMED};

use crate::CvError;

const CHEST_PNG: &[u8] = include_bytes!("../assets/ui/chest.png");
const CHEST_F_ICON_PNG: &[u8] = include_bytes!("../assets/ui/chest_F_icon.png");
const MATCH_THRESHOLD: f64 = 0.80;
const BASE_WIDTH: i32 = 1920;
const BASE_HEIGHT: i32 = 1080;
const CHEST_ROI: (i32, i32, i32, i32) = (330, 130, 1250, 840);
const INTERACT_ROI: (i32, i32, i32, i32) = (1150, 450, 100, 300);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChestObservation {
    pub chest: Option<(i32, i32, f64)>,
    pub interact: bool,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct ChestMatcher {
    chest: Mat,
    chest_f_icon: Mat,
}

impl ChestMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            chest: decode_png(CHEST_PNG)?,
            chest_f_icon: decode_png(CHEST_F_ICON_PNG)?,
        })
    }

    pub fn from_assets(
        chest: impl AsRef<Path>,
        chest_f_icon: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            chest: decode_file(chest.as_ref())?,
            chest_f_icon: decode_file(chest_f_icon.as_ref())?,
        })
    }

    pub fn detect(&self, full_bgr: &Mat) -> Result<ChestObservation, CvError> {
        if full_bgr.empty() {
            return Ok(ChestObservation::default());
        }
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        let mut gray = Mat::default();
        imgproc::cvt_color(
            full_bgr,
            &mut gray,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let scale = (f64::from(width) / f64::from(BASE_WIDTH))
            .min(f64::from(height) / f64::from(BASE_HEIGHT));
        let chest = match scaled_roi(CHEST_ROI, scale, width, height) {
            Some(roi) => {
                let image = Mat::roi(&gray, roi)?;
                match_scaled_template(&image, &self.chest, scale, roi, MATCH_THRESHOLD)?
            }
            None => None,
        };
        let interact = match scaled_roi(INTERACT_ROI, scale, width, height) {
            Some(roi) => {
                let image = Mat::roi(&gray, roi)?;
                match_scaled_template(&image, &self.chest_f_icon, scale, roi, MATCH_THRESHOLD)?
                    .is_some()
            }
            None => false,
        };
        Ok(ChestObservation {
            chest,
            interact,
            width: width as u32,
            height: height as u32,
        })
    }
}

fn scaled_roi(roi: (i32, i32, i32, i32), scale: f64, width: i32, height: i32) -> Option<Rect> {
    let (x, y, w, h) = roi;
    let right = ((f64::from(x + w) * scale).round() as i32).clamp(0, width);
    let bottom = ((f64::from(y + h) * scale).round() as i32).clamp(0, height);
    let x = (f64::from(x) * scale).round() as i32;
    let y = (f64::from(y) * scale).round() as i32;
    let x = x.clamp(0, width);
    let y = y.clamp(0, height);
    let roi_width = right - x;
    let roi_height = bottom - y;
    (roi_width > 0 && roi_height > 0).then_some(Rect::new(x, y, roi_width, roi_height))
}

fn match_scaled_template(
    image: &impl core::ToInputArray,
    template: &Mat,
    scale: f64,
    roi: Rect,
    threshold: f64,
) -> Result<Option<(i32, i32, f64)>, CvError> {
    let scaled_storage: Mat;
    let template = if (scale - 1.0).abs() > 0.02 {
        let size = Size::new(
            ((f64::from(template.cols()) * scale).round().max(4.0)) as i32,
            ((f64::from(template.rows()) * scale).round().max(4.0)) as i32,
        );
        let interpolation = if scale < 1.0 { INTER_AREA } else { INTER_CUBIC };
        let mut resized = Mat::default();
        imgproc::resize(template, &mut resized, size, 0.0, 0.0, interpolation)?;
        scaled_storage = resized;
        &scaled_storage
    } else {
        template
    };
    if template.cols() > roi.width || template.rows() > roi.height {
        return Ok(None);
    }

    let mut result = Mat::default();
    imgproc::match_template(
        image,
        template,
        &mut result,
        TM_CCOEFF_NORMED,
        &core::no_array(),
    )?;
    let mut score = 0.0;
    let mut location = Point::default();
    core::min_max_loc(
        &result,
        None,
        Some(&mut score),
        None,
        Some(&mut location),
        &core::no_array(),
    )?;
    if score.is_finite() && score >= threshold {
        Ok(Some((
            roi.x + location.x + template.cols() / 2,
            roi.y + location.y + template.rows() / 2,
            score,
        )))
    } else {
        Ok(None)
    }
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("内置宝箱模板解码失败".into()));
    }
    Ok(decoded)
}

fn decode_file(path: &Path) -> Result<Mat, CvError> {
    decode_png(&std::fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use opencv::core::CV_8UC3;
    use opencv::prelude::MatExprTraitConst;

    use super::*;

    fn paste_scaled(mut image: Mat, template: &Mat, x: i32, y: i32, scale: f64) -> Mat {
        let mut gray = Mat::default();
        imgproc::resize(
            template,
            &mut gray,
            Size::new(
                (f64::from(template.cols()) * scale).round() as i32,
                (f64::from(template.rows()) * scale).round() as i32,
            ),
            0.0,
            0.0,
            INTER_AREA,
        )
        .unwrap();
        let mut bgr = Mat::default();
        imgproc::cvt_color(
            &gray,
            &mut bgr,
            imgproc::COLOR_GRAY2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        let mut target = Mat::roi_mut(&mut image, Rect::new(x, y, bgr.cols(), bgr.rows())).unwrap();
        bgr.copy_to(&mut target).unwrap();
        image
    }

    fn screen(scale: f64, prompt_in_roi: bool, extra_prompt_outside_roi: bool) -> Mat {
        let image = Mat::zeros(
            (f64::from(BASE_HEIGHT) * scale).round() as i32,
            (f64::from(BASE_WIDTH) * scale).round() as i32,
            CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap();
        let image = paste_scaled(
            image,
            &decode_png(CHEST_PNG).unwrap(),
            (500.0 * scale).round() as i32,
            (400.0 * scale).round() as i32,
            scale,
        );
        let image = if prompt_in_roi {
            paste_scaled(
                image,
                &decode_png(CHEST_F_ICON_PNG).unwrap(),
                (1180.0 * scale).round() as i32,
                (500.0 * scale).round() as i32,
                scale,
            )
        } else {
            image
        };
        if extra_prompt_outside_roi {
            paste_scaled(
                image,
                &decode_png(CHEST_F_ICON_PNG).unwrap(),
                (1300.0 * scale).round() as i32,
                (800.0 * scale).round() as i32,
                scale,
            )
        } else {
            image
        }
    }

    #[test]
    fn detects_chest_and_prompt_at_reference_and_reduced_resolution() {
        let matcher = ChestMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let observation = matcher.detect(&screen(scale, true, false)).unwrap();
            assert!(observation.interact, "scale={scale}: {observation:?}");
            let (x, y, score) = observation.chest.expect("chest should be detected");
            assert!((x - (518.0 * scale).round() as i32).abs() <= 2);
            assert!((y - (415.0 * scale).round() as i32).abs() <= 2);
            assert!(score >= MATCH_THRESHOLD);
            assert_eq!(observation.width, (1920.0 * scale).round() as u32);
            assert_eq!(observation.height, (1080.0 * scale).round() as u32);
        }
    }

    #[test]
    fn identical_prompt_outside_interaction_roi_is_ignored() {
        let matcher = ChestMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let observation = matcher.detect(&screen(scale, false, true)).unwrap();
            assert!(!observation.interact, "scale={scale}: {observation:?}");
            assert!(observation.chest.is_some());
        }
    }

    #[test]
    fn external_assets_use_the_same_chest_templates() {
        let templates = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/Templates");
        let assets =
            crate::assets::AssetCatalog::load(templates.parent().unwrap(), "zh-CN").unwrap();
        let matcher = ChestMatcher::from_assets(
            assets.template("ui.chest").unwrap(),
            assets.template("ui.chest_f_icon").unwrap(),
        )
        .unwrap();
        let observation = matcher.detect(&screen(1.0, true, false)).unwrap();
        assert!(observation.interact);
        assert!(observation.chest.is_some());
    }
}
