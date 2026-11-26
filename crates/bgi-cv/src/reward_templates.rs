use std::path::Path;

use bgi_task::{RewardCandidate, RewardKind};
use opencv::core::{self, Mat, MatTraitConst, Point, Size, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{self, INTER_AREA, INTER_CUBIC, TM_CCOEFF_NORMED};
use opencv::prelude::MatTrait;

use crate::CvError;

const CLAIM_TEXT_PNG: &[u8] = include_bytes!("../assets/ui/claim_text.png");
const CLAIM_GIFT_PNG: &[u8] = include_bytes!("../assets/ui/claim_gift.png");
const CONTINUE_TIP_PNG: &[u8] = include_bytes!("../assets/ui/click_blank_continue.png");
const REWARD_THRESHOLD: f64 = 0.86;
const CONTINUE_THRESHOLD: f64 = 0.82;
const MAX_MATCHES: usize = 20;

type TemplateMatch = ((i32, i32), f64);

#[derive(Debug, Clone)]
pub struct RewardTemplateMatcher {
    claim_text: Mat,
    claim_gift: Mat,
    continue_tip: Mat,
}

impl RewardTemplateMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            claim_text: decode_png(CLAIM_TEXT_PNG)?,
            claim_gift: decode_png(CLAIM_GIFT_PNG)?,
            continue_tip: decode_png(CONTINUE_TIP_PNG)?,
        })
    }

    pub fn from_assets(
        claim_text: impl AsRef<Path>,
        claim_gift: impl AsRef<Path>,
        continue_tip: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            claim_text: decode_file(claim_text.as_ref())?,
            claim_gift: decode_file(claim_gift.as_ref())?,
            continue_tip: decode_file(continue_tip.as_ref())?,
        })
    }

    pub fn detect(&self, full_bgr: &Mat) -> Result<(Vec<RewardCandidate>, bool), CvError> {
        if full_bgr.empty() {
            return Ok((Vec::new(), false));
        }
        let mut gray = Mat::default();
        imgproc::cvt_color(
            full_bgr,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let scale = (gray.cols() as f64 / 1920.0).min(gray.rows() as f64 / 1080.0);
        let mut candidates = find_multi(&gray, &self.claim_text, scale, REWARD_THRESHOLD)?
            .into_iter()
            .map(|(center, score)| RewardCandidate {
                kind: RewardKind::ClaimText,
                center,
                score,
            })
            .collect::<Vec<_>>();
        candidates.extend(
            find_multi(&gray, &self.claim_gift, scale, REWARD_THRESHOLD)?
                .into_iter()
                .map(|(center, score)| RewardCandidate {
                    kind: RewardKind::ClaimGift,
                    center,
                    score,
                }),
        );
        candidates.sort_by(|a, b| {
            a.center
                .1
                .cmp(&b.center.1)
                .then_with(|| a.center.0.cmp(&b.center.0))
                .then_with(|| b.score.total_cmp(&a.score))
        });
        let continue_tip = find_multi(&gray, &self.continue_tip, scale, CONTINUE_THRESHOLD)?
            .into_iter()
            .any(|_| true);
        Ok((candidates, continue_tip))
    }
}

fn find_multi(
    image: &Mat,
    template: &Mat,
    scale: f64,
    threshold: f64,
) -> Result<Vec<TemplateMatch>, CvError> {
    let scaled_storage: Mat;
    let template = if (scale - 1.0).abs() > 0.02 {
        let width = ((template.cols() as f64) * scale).round().max(4.0) as i32;
        let height = ((template.rows() as f64) * scale).round().max(4.0) as i32;
        let interp = if scale < 1.0 { INTER_AREA } else { INTER_CUBIC };
        let mut resized = Mat::default();
        imgproc::resize(
            template,
            &mut resized,
            Size::new(width, height),
            0.0,
            0.0,
            interp,
        )?;
        scaled_storage = resized;
        &scaled_storage
    } else {
        template
    };
    if template.cols() > image.cols() || template.rows() > image.rows() {
        return Ok(Vec::new());
    }
    let mut result = Mat::default();
    imgproc::match_template(
        image,
        template,
        &mut result,
        TM_CCOEFF_NORMED,
        &core::no_array(),
    )?;
    let mut matches = Vec::new();
    loop {
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
        if !score.is_finite() || score < threshold || matches.len() >= MAX_MATCHES {
            break;
        }
        matches.push((
            (
                location.x + template.cols() / 2,
                location.y + template.rows() / 2,
            ),
            score,
        ));
        let max_delta_x = template.cols() / 3 + 1;
        let max_delta_y = template.rows() / 3 + 1;
        for y in
            (location.y - max_delta_y).max(0)..=(location.y + max_delta_y).min(result.rows() - 1)
        {
            for x in (location.x - max_delta_x).max(0)
                ..=(location.x + max_delta_x).min(result.cols() - 1)
            {
                let intersection = (template.cols() - (x - location.x).abs()).max(0)
                    * (template.rows() - (y - location.y).abs()).max(0);
                let union = 2 * template.cols() * template.rows() - intersection;
                if intersection * 2 >= union {
                    *result.at_2d_mut::<f32>(y, x)? = -1.0;
                }
            }
        }
    }
    Ok(matches)
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("内置领取模板解码失败".into()));
    }
    Ok(decoded)
}

fn decode_file(path: &Path) -> Result<Mat, CvError> {
    decode_png(&std::fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use opencv::core::{CV_8UC3, Rect, Size};
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
        let mut roi = Mat::roi_mut(&mut image, Rect::new(x, y, bgr.cols(), bgr.rows())).unwrap();
        bgr.copy_to(&mut roi).unwrap();
        image
    }

    fn screen(scale: f64) -> Mat {
        let text = decode_png(CLAIM_TEXT_PNG).unwrap();
        let gift = decode_png(CLAIM_GIFT_PNG).unwrap();
        let image = Mat::zeros(
            (1080.0 * scale).round() as i32,
            (1920.0 * scale).round() as i32,
            CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap();
        let image = paste_scaled(
            image,
            &text,
            (100.0 * scale) as i32,
            (200.0 * scale) as i32,
            scale,
        );
        let image = paste_scaled(
            image,
            &gift,
            (500.0 * scale) as i32,
            (100.0 * scale) as i32,
            scale,
        );
        let image = paste_scaled(
            image,
            &gift,
            (600.0 * scale) as i32,
            (100.0 * scale) as i32,
            scale,
        );
        paste_scaled(
            image,
            &decode_png(CONTINUE_TIP_PNG).unwrap(),
            (800.0 * scale) as i32,
            (700.0 * scale) as i32,
            scale,
        )
    }

    #[test]
    fn rewards_are_sorted_and_scaled_without_duplicate_gift_matches() {
        let matcher = RewardTemplateMatcher::new().unwrap();
        for scale in [1.0, 0.5] {
            let (candidates, continue_tip) = matcher.detect(&screen(scale)).unwrap();
            assert!(continue_tip);
            assert_eq!(candidates.len(), 3, "scale={scale}: {candidates:?}");
            assert_eq!(candidates[0].kind, RewardKind::ClaimGift);
            assert_eq!(candidates[1].kind, RewardKind::ClaimGift);
            assert_eq!(candidates[2].kind, RewardKind::ClaimText);
        }
    }
}
