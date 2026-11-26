use std::path::Path;

use bgi_task::ExpeditionPageState;
use opencv::core::{self, Mat, MatTraitConst, Rect, Size, ToInputArray, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{self, COLOR_BGR2GRAY, INTER_AREA, INTER_CUBIC, TM_CCOEFF_NORMED};

use crate::{CvError, MatchPoint};

const COLLECT_PNG: &[u8] = include_bytes!("../assets/ui/expedition_collect.png");
const REDISPATCH_PNG: &[u8] = include_bytes!("../assets/ui/expedition_redispatch.png");
const BASE_WIDTH: i32 = 1920;
const BASE_HEIGHT: i32 = 1080;
const MATCH_THRESHOLD: f64 = 0.8;

pub struct ExpeditionMatcher {
    collect: Mat,
    redispatch: Mat,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpeditionObservation {
    pub width: u32,
    pub height: u32,
    pub collect: Option<MatchPoint>,
    pub redispatch: Option<MatchPoint>,
}

impl ExpeditionObservation {
    pub fn state(self) -> Option<ExpeditionPageState> {
        match (self.collect, self.redispatch) {
            (Some(_), None) => Some(ExpeditionPageState::CollectAvailable),
            (None, Some(_)) => Some(ExpeditionPageState::RedispatchAvailable),
            _ => None,
        }
    }
}

impl ExpeditionMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            collect: decode_png(COLLECT_PNG)?,
            redispatch: decode_png(REDISPATCH_PNG)?,
        })
    }

    pub fn from_assets(
        collect: impl AsRef<Path>,
        redispatch: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            collect: decode_png(&std::fs::read(collect)?)?,
            redispatch: decode_png(&std::fs::read(redispatch)?)?,
        })
    }

    pub fn observe(&self, full_bgr: &Mat) -> Result<ExpeditionObservation, CvError> {
        if full_bgr.empty() || full_bgr.channels() != 3 {
            return Err(CvError::InvalidImage(
                "派遣状态识别需要非空三通道 BGR 图像".into(),
            ));
        }
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        if width < 320 || height < 240 {
            return Ok(ExpeditionObservation {
                width: width.max(0) as u32,
                height: height.max(0) as u32,
                collect: None,
                redispatch: None,
            });
        }
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
        let collect_roi = Rect::new(0, height - height / 3, width / 4, height / 3);
        let redispatch_roi = Rect::new(width / 2, height - height / 4, width / 4, height / 4);
        Ok(ExpeditionObservation {
            width: width as u32,
            height: height as u32,
            collect: matches_in_roi(&gray, &self.collect, collect_roi, scale)?,
            redispatch: matches_in_roi(&gray, &self.redispatch, redispatch_roi, scale)?,
        })
    }

    pub fn detect(&self, full_bgr: &Mat) -> Result<Option<ExpeditionPageState>, CvError> {
        Ok(self.observe(full_bgr)?.state())
    }
}

fn matches_in_roi(
    gray: &Mat,
    template: &Mat,
    roi: Rect,
    scale: f64,
) -> Result<Option<MatchPoint>, CvError> {
    let image = Mat::roi(gray, roi)?;
    match_scaled_template(&image, template, roi, scale)
}

fn match_scaled_template(
    image: &(impl MatTraitConst + ToInputArray),
    template: &Mat,
    roi: Rect,
    scale: f64,
) -> Result<Option<MatchPoint>, CvError> {
    let scaled_storage: Mat;
    let template = if (scale - 1.0).abs() > 0.02 {
        let size = Size::new(
            (f64::from(template.cols()) * scale).round().max(4.0) as i32,
            (f64::from(template.rows()) * scale).round().max(4.0) as i32,
        );
        let interpolation = if scale < 1.0 { INTER_AREA } else { INTER_CUBIC };
        let mut resized = Mat::default();
        imgproc::resize(template, &mut resized, size, 0.0, 0.0, interpolation)?;
        scaled_storage = resized;
        &scaled_storage
    } else {
        template
    };
    if template.cols() > image.cols() || template.rows() > image.rows() {
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
    let mut max_loc = core::Point::new(0, 0);
    core::min_max_loc(
        &result,
        None,
        Some(&mut score),
        None,
        Some(&mut max_loc),
        &core::no_array(),
    )?;
    Ok(
        (score.is_finite() && score >= MATCH_THRESHOLD).then_some(MatchPoint {
            x: f64::from(roi.x + max_loc.x + template.cols() / 2),
            y: f64::from(roi.y + max_loc.y + template.rows() / 2),
            score,
        }),
    )
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("内置派遣模板解码失败".into()));
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use opencv::core::{CV_8UC3, Scalar};
    use opencv::imgproc::resize;
    use opencv::prelude::{MatExprTraitConst, MatTrait};

    use super::*;

    fn frame(width: i32, height: i32) -> Mat {
        Mat::zeros(height, width, CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap()
    }

    fn paste(frame: &mut Mat, template: &Mat, x: i32, y: i32, scale: f64) {
        let mut scaled = Mat::default();
        resize(
            template,
            &mut scaled,
            Size::new(
                (f64::from(template.cols()) * scale).round() as i32,
                (f64::from(template.rows()) * scale).round() as i32,
            ),
            0.0,
            0.0,
            INTER_AREA,
        )
        .unwrap();
        let rect = Rect::new(x, y, scaled.cols(), scaled.rows());
        let mut target = Mat::roi_mut(frame, rect).unwrap();
        let mut color = Mat::default();
        opencv::imgproc::cvt_color(
            &scaled,
            &mut color,
            opencv::imgproc::COLOR_GRAY2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        color.copy_to(&mut target).unwrap();
    }

    #[test]
    fn detects_collect_and_redispatch_in_their_reference_rois() {
        let matcher = ExpeditionMatcher::new().unwrap();
        let mut image = frame(1920, 1080);
        paste(&mut image, &matcher.collect, 100, 800, 1.0);
        let observation = matcher.observe(&image).unwrap();
        assert_eq!(
            observation.state(),
            Some(ExpeditionPageState::CollectAvailable)
        );
        let collect = observation.collect.unwrap();
        assert!((collect.x - (100.0 + f64::from(matcher.collect.cols()) / 2.0)).abs() <= 1.0);
        assert!((collect.y - (800.0 + f64::from(matcher.collect.rows()) / 2.0)).abs() <= 1.0);
        assert!(collect.score >= MATCH_THRESHOLD);
        let mut image = frame(1920, 1080);
        paste(&mut image, &matcher.redispatch, 1100, 900, 1.0);
        assert_eq!(
            matcher.detect(&image).unwrap(),
            Some(ExpeditionPageState::RedispatchAvailable)
        );
    }

    #[test]
    fn scales_templates_and_rois_for_half_resolution() {
        let matcher = ExpeditionMatcher::new().unwrap();
        let mut image = frame(1280, 720);
        paste(&mut image, &matcher.collect, 67, 533, 2.0 / 3.0);
        assert_eq!(
            matcher.detect(&image).unwrap(),
            Some(ExpeditionPageState::CollectAvailable)
        );
        let mut image = frame(1280, 720);
        paste(&mut image, &matcher.redispatch, 733, 600, 2.0 / 3.0);
        assert_eq!(
            matcher.detect(&image).unwrap(),
            Some(ExpeditionPageState::RedispatchAvailable)
        );
    }

    #[test]
    fn ignores_icons_outside_their_rois() {
        let matcher = ExpeditionMatcher::new().unwrap();
        let mut image = frame(1920, 1080);
        image
            .set_to(&Scalar::all(127.0), &core::no_array())
            .unwrap();
        assert_eq!(matcher.detect(&image).unwrap(), None);
        paste(&mut image, &matcher.collect, 600, 100, 1.0);
        paste(&mut image, &matcher.redispatch, 100, 100, 1.0);
        assert_eq!(matcher.detect(&image).unwrap(), None);
    }

    #[test]
    fn rejects_frames_with_both_available_icons() {
        let matcher = ExpeditionMatcher::new().unwrap();
        let mut image = frame(1920, 1080);
        paste(&mut image, &matcher.collect, 100, 800, 1.0);
        paste(&mut image, &matcher.redispatch, 1100, 900, 1.0);
        assert_eq!(matcher.detect(&image).unwrap(), None);
    }

    #[test]
    fn rejects_empty_and_non_bgr_frames() {
        let matcher = ExpeditionMatcher::new().unwrap();
        assert!(matcher.detect(&Mat::default()).is_err());
        let gray = Mat::zeros(1080, 1920, opencv::core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        assert!(matcher.detect(&gray).is_err());
    }
}
