use std::path::Path;

use opencv::core::{self, Mat, MatTraitConst, Point, Rect, Size, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{self, COLOR_BGR2GRAY, INTER_AREA, INTER_CUBIC, TM_CCOEFF_NORMED};

use crate::CvError;

const GADGET_PNG: &[u8] = include_bytes!("../assets/ui/TheBoonOfTheElderTree.png");
const MENU_PNG: &[u8] = include_bytes!("../assets/ui/menu_bag.png");
const CONFIRM_PNG: &[u8] = include_bytes!("../assets/ui/confirm.png");
const ENTER_GAME_PNG: &[u8] = include_bytes!("../assets/ui/exit_welcome.png");
const MATCH_THRESHOLD: f64 = 0.80;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AutoWoodObservation {
    pub gadget: Option<(i32, i32, f64)>,
    pub menu: Option<(i32, i32, f64)>,
    pub confirm: Option<(i32, i32, f64)>,
    pub enter_game: Option<(i32, i32, f64)>,
    pub width: u32,
    pub height: u32,
}

/// Score of one auto-wood template: `best` is the raw peak correlation
/// (0.0 when the template does not fit its ROI), `hit` is the accepted match.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WoodProbe {
    pub best: f64,
    pub hit: Option<(i32, i32, f64)>,
}

/// Per-template scores for the four auto-wood ROIs, kept for diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AutoWoodProbes {
    pub gadget: WoodProbe,
    pub menu: WoodProbe,
    pub confirm: WoodProbe,
    pub enter_game: WoodProbe,
}

impl AutoWoodProbes {
    pub fn observation(&self, width: u32, height: u32) -> AutoWoodObservation {
        AutoWoodObservation {
            gadget: self.gadget.hit,
            menu: self.menu.hit,
            confirm: self.confirm.hit,
            enter_game: self.enter_game.hit,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoWoodMatcher {
    gadget: Mat,
    menu: Mat,
    confirm: Mat,
    enter_game: Mat,
}

impl AutoWoodMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            gadget: decode_png(GADGET_PNG)?,
            menu: decode_png(MENU_PNG)?,
            confirm: decode_png(CONFIRM_PNG)?,
            enter_game: decode_png(ENTER_GAME_PNG)?,
        })
    }

    pub fn from_assets(
        gadget: impl AsRef<Path>,
        menu: impl AsRef<Path>,
        confirm: impl AsRef<Path>,
        enter_game: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            gadget: decode_file(gadget.as_ref())?,
            menu: decode_file(menu.as_ref())?,
            confirm: decode_file(confirm.as_ref())?,
            enter_game: decode_file(enter_game.as_ref())?,
        })
    }

    pub fn detect(&self, full_bgr: &Mat) -> Result<AutoWoodObservation, CvError> {
        if full_bgr.empty() {
            return Ok(AutoWoodObservation::default());
        }
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        if width < 320 || height < 240 {
            return Ok(AutoWoodObservation::default());
        }
        Ok(self
            .probe(full_bgr)?
            .observation(width as u32, height as u32))
    }

    /// Same ROIs as [`Self::detect`], but keeps the peak score of every template
    /// so callers can log near-misses as well as accepted hits.
    pub fn probe(&self, full_bgr: &Mat) -> Result<AutoWoodProbes, CvError> {
        if full_bgr.empty() {
            return Ok(AutoWoodProbes::default());
        }
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        if width < 320 || height < 240 {
            return Ok(AutoWoodProbes::default());
        }
        let mut gray = Mat::default();
        imgproc::cvt_color(
            full_bgr,
            &mut gray,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let scale = (f64::from(width) / 1920.0).min(f64::from(height) / 1080.0);

        // These are the four independent ROIs from AutoWood/Assets/Recognition.json.
        let gadget_roi = Rect::new(
            width * 3 / 4,
            height / 2,
            width - width * 3 / 4,
            height - height / 2,
        );
        let menu_roi = Rect::new(0, 0, width / 2, height);
        let confirm_roi = Rect::new(0, 0, width, height);
        let enter_game_roi = Rect::new(0, height / 2, width, height - height / 2);

        Ok(AutoWoodProbes {
            gadget: probe_in_roi(&gray, gadget_roi, &self.gadget, scale)?,
            menu: probe_in_roi(&gray, menu_roi, &self.menu, scale)?,
            confirm: probe_in_roi(&gray, confirm_roi, &self.confirm, scale)?,
            enter_game: probe_in_roi(&gray, enter_game_roi, &self.enter_game, scale)?,
        })
    }
}

fn probe_in_roi(gray: &Mat, roi: Rect, template: &Mat, scale: f64) -> Result<WoodProbe, CvError> {
    let image = Mat::roi(gray, roi)?;
    let scaled_storage: Mat;
    let template = if (scale - 1.0).abs() > 0.02 {
        let mut resized = Mat::default();
        imgproc::resize(
            template,
            &mut resized,
            Size::new(
                (f64::from(template.cols()) * scale).round().max(4.0) as i32,
                (f64::from(template.rows()) * scale).round().max(4.0) as i32,
            ),
            0.0,
            0.0,
            if scale < 1.0 { INTER_AREA } else { INTER_CUBIC },
        )?;
        scaled_storage = resized;
        &scaled_storage
    } else {
        template
    };
    if template.cols() > roi.width || template.rows() > roi.height {
        return Ok(WoodProbe::default());
    }
    let mut result = Mat::default();
    imgproc::match_template(
        &image,
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
    // OpenCV yields NaN for constant (zero-variance) regions; report them as 0.
    let best = if score.is_finite() { score } else { 0.0 };
    let hit = (best >= MATCH_THRESHOLD).then_some((
        roi.x + location.x + template.cols() / 2,
        roi.y + location.y + template.rows() / 2,
        best,
    ));
    Ok(WoodProbe { best, hit })
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("自动砍树模板解码失败".into()));
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

    fn screen_with(template: &[u8], scale: f64, base: (f64, f64)) -> Mat {
        let mut screen = Mat::zeros(
            (1080.0 * scale).round() as i32,
            (1920.0 * scale).round() as i32,
            CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap();
        let template = decode_png(template).unwrap();
        let mut gray = Mat::default();
        imgproc::resize(
            &template,
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
        let mut target = Mat::roi_mut(
            &mut screen,
            Rect::new(
                (base.0 * scale).round() as i32,
                (base.1 * scale).round() as i32,
                bgr.cols(),
                bgr.rows(),
            ),
        )
        .unwrap();
        bgr.copy_to(&mut target).unwrap();
        screen
    }

    #[test]
    fn auto_wood_templates_match_at_reference_and_half_resolution() {
        let matcher = AutoWoodMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            for (template, base, expected) in [
                (GADGET_PNG, (1500.0, 650.0), 0),
                (MENU_PNG, (20.0, 20.0), 1),
                (CONFIRM_PNG, (80.0, 950.0), 2),
                (ENTER_GAME_PNG, (900.0, 700.0), 3),
            ] {
                let observation = matcher.detect(&screen_with(template, scale, base)).unwrap();
                let hits = [
                    observation.gadget,
                    observation.menu,
                    observation.confirm,
                    observation.enter_game,
                ];
                for (index, hit) in hits.into_iter().enumerate() {
                    assert_eq!(
                        hit.is_some(),
                        index == expected,
                        "scale={scale}, index={index}: {observation:?}"
                    );
                }
                assert_eq!(observation.width, (1920.0 * scale).round() as u32);
                assert_eq!(observation.height, (1080.0 * scale).round() as u32);
            }
        }
    }

    #[test]
    fn probe_scores_agree_with_detect_decisions() {
        let matcher = AutoWoodMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            for (template, base, expected) in [
                (GADGET_PNG, (1500.0, 650.0), 0usize),
                (MENU_PNG, (20.0, 20.0), 1),
                (CONFIRM_PNG, (80.0, 950.0), 2),
                (ENTER_GAME_PNG, (900.0, 700.0), 3),
            ] {
                let screen = screen_with(template, scale, base);
                let probes = matcher.probe(&screen).unwrap();
                let observation = matcher.detect(&screen).unwrap();
                assert_eq!(
                    probes.observation(observation.width, observation.height),
                    observation
                );
                let picked = [
                    probes.gadget,
                    probes.menu,
                    probes.confirm,
                    probes.enter_game,
                ][expected];
                assert!(
                    picked.hit.is_some(),
                    "scale={scale} expected={expected}: {probes:?}"
                );
                assert!(picked.best >= MATCH_THRESHOLD, "{picked:?}");
            }
        }
    }

    #[test]
    fn menu_and_login_templates_outside_their_rois_are_ignored() {
        let matcher = AutoWoodMatcher::new().unwrap();
        let menu = matcher
            .detect(&screen_with(MENU_PNG, 1.0, (1200.0, 100.0)))
            .unwrap();
        assert!(menu.menu.is_none());
        let login = matcher
            .detect(&screen_with(ENTER_GAME_PNG, 1.0, (900.0, 100.0)))
            .unwrap();
        assert!(login.enter_game.is_none());
    }

    #[test]
    fn external_assets_load_the_same_auto_wood_templates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let assets = crate::assets::AssetCatalog::load(root, "zh-CN").unwrap();
        let matcher = AutoWoodMatcher::from_assets(
            assets.template("ui.auto_wood.gadget").unwrap(),
            assets.template("ui.auto_wood.menu").unwrap(),
            assets.template("ui.auto_wood.confirm").unwrap(),
            assets.template("ui.auto_wood.enter_game").unwrap(),
        )
        .unwrap();
        assert!(
            matcher
                .detect(&screen_with(GADGET_PNG, 1.0, (1500.0, 650.0)))
                .unwrap()
                .gadget
                .is_some()
        );
    }
}
