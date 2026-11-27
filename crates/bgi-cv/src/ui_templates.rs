use std::path::Path;

use opencv::core::{self, Mat, MatTraitConst, Rect, Size, Vector};
use opencv::imgcodecs::{IMREAD_COLOR, IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{self, COLOR_BGR2GRAY, INTER_AREA, INTER_CUBIC, TM_CCOEFF_NORMED};

use crate::CvError;

const UNDERGROUND_SWITCH_PNG: &[u8] = include_bytes!("../assets/ui/MapUndergroundSwitchButton.png");
const TO_GROUND_PNG: &[u8] = include_bytes!("../assets/ui/MapUndergroundToGroundButton.png");
const KEY_SPACE_PNG: &[u8] = include_bytes!("../assets/ui/key_space.png");
const KEY_X_PNG: &[u8] = include_bytes!("../assets/ui/key_x.png");
const PAIMON_MENU_PNG: &[u8] = include_bytes!("../assets/ui/paimon_menu.png");
const INVENTORY_PNG: &[u8] = include_bytes!("../assets/ui/inventory.png");
const IN_DOMAIN_PNG: &[u8] = include_bytes!("../assets/ui/in_domain.png");
const COOK_ICON_PNG: &[u8] = include_bytes!("../assets/ui/ui_left_top_cook_icon.png");
const WHITE_CONFIRM_PNG: &[u8] = include_bytes!("../assets/ui/btn_white_confirm.png");
const WHITE_RECOVER_PNG: &[u8] = include_bytes!("../assets/ui/btn_white_recover.png");
const TALK_DISABLED_UI_PNG: &[u8] = include_bytes!("../assets/ui/disabled_ui.png");
const RECOVERY_PNG: &[u8] = include_bytes!("../assets/ui/Recovery.png");
const RESURRECTION_PNG: &[u8] = include_bytes!("../assets/ui/Resurrection.png");

const TO_GROUND_THRESHOLD: f64 = 0.92;
const UNDERGROUND_SWITCH_THRESHOLD: f64 = 0.80;
const KEY_TIP_THRESHOLD: f64 = 0.78;
/// Shared acceptance threshold for the five blocking-screen templates.
const SCREEN_THRESHOLD: f64 = 0.82;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionStatus {
    #[default]
    Normal,
    Fly,
    Climb,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BigMapLayerButtons {
    /// Center `(x, y)` in client pixels of the active cyan underground/layered toggle icon.
    pub underground_switch: Option<(i32, i32, f64)>,
    /// Center `(x, y)` in client pixels of the ground/surface layer button.
    pub to_ground: Option<(i32, i32, f64)>,
}

/// Template matcher for Genshin UI chrome buttons (`MapUnderground*`) and bottom-right `@keyTip` indicators.
pub struct UiTemplateMatcher {
    underground_switch_bgr: Mat,
    to_ground_bgr: Mat,
    key_space_gray: Mat,
    key_x_gray: Mat,
}

impl UiTemplateMatcher {
    pub fn new() -> Result<Self, CvError> {
        let underground_switch_bgr = decode_png(UNDERGROUND_SWITCH_PNG, IMREAD_COLOR)?;
        let to_ground_bgr = decode_png(TO_GROUND_PNG, IMREAD_COLOR)?;
        let key_space_gray = decode_png(KEY_SPACE_PNG, IMREAD_GRAYSCALE)?;
        let key_x_gray = decode_png(KEY_X_PNG, IMREAD_GRAYSCALE)?;
        Ok(Self {
            underground_switch_bgr,
            to_ground_bgr,
            key_space_gray,
            key_x_gray,
        })
    }

    pub fn from_assets(
        underground_switch: impl AsRef<Path>,
        to_ground: impl AsRef<Path>,
        key_space: impl AsRef<Path>,
        key_x: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            underground_switch_bgr: decode_png_file(underground_switch.as_ref(), IMREAD_COLOR)?,
            to_ground_bgr: decode_png_file(to_ground.as_ref(), IMREAD_COLOR)?,
            key_space_gray: decode_png_file(key_space.as_ref(), IMREAD_GRAYSCALE)?,
            key_x_gray: decode_png_file(key_x.as_ref(), IMREAD_GRAYSCALE)?,
        })
    }

    pub fn detect_bigmap_layer_buttons(
        &self,
        full_bgr: &Mat,
    ) -> Result<BigMapLayerButtons, CvError> {
        let cw = full_bgr.cols();
        let ch = full_bgr.rows();
        if cw < 320 || ch < 240 {
            return Ok(BigMapLayerButtons::default());
        }
        let scale = (cw as f64 / 1920.0).min(ch as f64 / 1080.0);
        let roi = clamp_roi(
            ((cw as f64) - 120.0 * scale).round() as i32,
            (250.0 * scale).round() as i32,
            (90.0 * scale).round() as i32,
            (570.0 * scale).round() as i32,
            cw,
            ch,
        );
        let Some(roi) = roi else {
            return Ok(BigMapLayerButtons::default());
        };

        let roi_bgr = Mat::roi(full_bgr, roi)?;
        let underground_switch = match_scaled_template(
            &roi_bgr,
            &self.underground_switch_bgr,
            roi,
            scale,
            UNDERGROUND_SWITCH_THRESHOLD,
        )?;
        let to_ground = match_scaled_template(
            &roi_bgr,
            &self.to_ground_bgr,
            roi,
            scale,
            TO_GROUND_THRESHOLD,
        )?;

        Ok(BigMapLayerButtons {
            underground_switch,
            to_ground,
        })
    }

    pub fn detect_motion_status(&self, full_bgr: &Mat) -> Result<MotionStatus, CvError> {
        let cw = full_bgr.cols();
        let ch = full_bgr.rows();
        if cw < 320 || ch < 240 {
            return Ok(MotionStatus::Normal);
        }
        let scale = (cw as f64 / 1920.0).min(ch as f64 / 1080.0);
        // Matches Element/Assets/Recognition.json `@keyTip`: rect(cw - 350*s, ch - 70*s, 200*s, 70*s).
        let roi = clamp_roi(
            ((cw as f64) - 350.0 * scale).round() as i32,
            ((ch as f64) - 70.0 * scale).round() as i32,
            (200.0 * scale).round() as i32,
            (70.0 * scale).round() as i32,
            cw,
            ch,
        );
        let Some(roi) = roi else {
            return Ok(MotionStatus::Normal);
        };

        let roi_bgr = Mat::roi(full_bgr, roi)?;
        let mut roi_gray = Mat::default();
        imgproc::cvt_color(
            &roi_bgr,
            &mut roi_gray,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let has_space = match_scaled_template(
            &roi_gray,
            &self.key_space_gray,
            roi,
            scale,
            KEY_TIP_THRESHOLD,
        )?
        .is_some();
        if !has_space {
            return Ok(MotionStatus::Normal);
        }

        let has_x =
            match_scaled_template(&roi_gray, &self.key_x_gray, roi, scale, KEY_TIP_THRESHOLD)?
                .is_some();
        Ok(if has_x {
            MotionStatus::Climb
        } else {
            MotionStatus::Fly
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiScreens {
    pub paimon_menu: bool,
    pub inventory: bool,
    pub in_domain: bool,
    pub cooking_page: bool,
    pub talk: bool,
}

impl UiScreens {
    /// True when the in-game HUD is on screen.
    ///
    /// The Paimon button sits in the top-left corner of the normal HUD and is covered by
    /// any full-screen UI (Paimon menu, dialogue, loading, cutscene), so its presence is a
    /// positive signal for gameplay — matching C# `Bv.IsInMainUi`, which reports "main UI"
    /// when the Paimon template is found. The remaining screens are screens *inside* the
    /// main UI and still exclude it.
    pub fn is_gameplay(&self) -> bool {
        self.paimon_menu && !self.talk && !self.in_domain && !self.cooking_page
    }
}

/// Score of a single blocking-screen template: `best` is the raw peak correlation
/// (0.0 when the template does not fit its ROI), `hit` is the accepted match with
/// its client-space center and score.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScreenProbe {
    pub best: f64,
    pub hit: Option<(i32, i32, f64)>,
}

impl ScreenProbe {
    fn at(best: f64, max_loc: core::Point, template: &Mat, roi: Rect, threshold: f64) -> Self {
        // OpenCV yields NaN for constant (zero-variance) regions; report them as 0.
        let best = if best.is_finite() { best } else { 0.0 };
        let hit = (best >= threshold).then_some((
            roi.x + max_loc.x + template.cols() / 2,
            roi.y + max_loc.y + template.rows() / 2,
            best,
        ));
        Self { best, hit }
    }

    pub fn hit_at(&self) -> Option<(i32, i32)> {
        self.hit.map(|(x, y, _)| (x, y))
    }
}

/// Per-template scores for the five screens that must all be absent to call a frame gameplay.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UiScreenProbes {
    pub paimon_menu: ScreenProbe,
    pub inventory: ScreenProbe,
    pub in_domain: ScreenProbe,
    pub cooking_page: ScreenProbe,
    pub talk: ScreenProbe,
}

impl UiScreenProbes {
    pub fn screens(&self) -> UiScreens {
        UiScreens {
            paimon_menu: self.paimon_menu.hit.is_some(),
            inventory: self.inventory.hit.is_some(),
            in_domain: self.in_domain.hit.is_some(),
            cooking_page: self.cooking_page.hit.is_some(),
            talk: self.talk.hit.is_some(),
        }
    }

    pub fn is_gameplay(&self) -> bool {
        self.screens().is_gameplay()
    }
}

pub struct UiScreenMatcher {
    paimon_menu: Mat,
    inventory: Mat,
    in_domain: Mat,
    cook_icon: Mat,
    white_confirm: Mat,
    white_recover: Mat,
    talk: Mat,
    recovery: Mat,
    resurrection: Mat,
}

impl UiScreenMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            paimon_menu: decode_png(PAIMON_MENU_PNG, IMREAD_GRAYSCALE)?,
            inventory: decode_png(INVENTORY_PNG, IMREAD_GRAYSCALE)?,
            in_domain: decode_png(IN_DOMAIN_PNG, IMREAD_GRAYSCALE)?,
            cook_icon: decode_png(COOK_ICON_PNG, IMREAD_GRAYSCALE)?,
            white_confirm: decode_png(WHITE_CONFIRM_PNG, IMREAD_COLOR)?,
            white_recover: decode_png(WHITE_RECOVER_PNG, IMREAD_COLOR)?,
            talk: decode_png(TALK_DISABLED_UI_PNG, IMREAD_GRAYSCALE)?,
            recovery: decode_png(RECOVERY_PNG, IMREAD_GRAYSCALE)?,
            resurrection: decode_png(RESURRECTION_PNG, IMREAD_GRAYSCALE)?,
        })
    }

    // Keep one positional entry point for the eight independent screen templates.
    #[allow(clippy::too_many_arguments)]
    pub fn from_assets(
        paimon_menu: impl AsRef<Path>,
        inventory: impl AsRef<Path>,
        in_domain: impl AsRef<Path>,
        cook_icon: impl AsRef<Path>,
        white_confirm: impl AsRef<Path>,
        white_recover: impl AsRef<Path>,
        talk: impl AsRef<Path>,
        recovery: impl AsRef<Path>,
        resurrection: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            paimon_menu: decode_png_file(paimon_menu.as_ref(), IMREAD_GRAYSCALE)?,
            inventory: decode_png_file(inventory.as_ref(), IMREAD_GRAYSCALE)?,
            in_domain: decode_png_file(in_domain.as_ref(), IMREAD_GRAYSCALE)?,
            cook_icon: decode_png_file(cook_icon.as_ref(), IMREAD_GRAYSCALE)?,
            white_confirm: decode_png_file(white_confirm.as_ref(), IMREAD_COLOR)?,
            white_recover: decode_png_file(white_recover.as_ref(), IMREAD_COLOR)?,
            talk: decode_png_file(talk.as_ref(), IMREAD_GRAYSCALE)?,
            recovery: decode_png_file(recovery.as_ref(), IMREAD_GRAYSCALE)?,
            resurrection: decode_png_file(resurrection.as_ref(), IMREAD_GRAYSCALE)?,
        })
    }

    pub fn detect(&self, full_bgr: &Mat) -> Result<UiScreens, CvError> {
        Ok(self.probe(full_bgr)?.screens())
    }

    /// Same ROIs as [`Self::detect`], but keeps the peak score of every template
    /// so callers can log why a frame was (not) classified as gameplay.
    pub fn probe(&self, full_bgr: &Mat) -> Result<UiScreenProbes, CvError> {
        let cw = full_bgr.cols();
        let ch = full_bgr.rows();
        if cw < 320 || ch < 240 {
            return Ok(UiScreenProbes::default());
        }
        let scale = (cw as f64 / 1920.0).min(ch as f64 / 1080.0);
        let top_left = clamp_roi(0, 0, cw / 4, ch / 4, cw, ch);
        let top_right = clamp_roi(
            cw * 3 / 4,
            0,
            cw / 4,
            (100.0 * scale).round() as i32,
            cw,
            ch,
        );
        let cook_roi = clamp_roi(
            0,
            0,
            (150.0 * scale).round() as i32,
            (120.0 * scale).round() as i32,
            cw,
            ch,
        );
        // One grayscale pass for all five ROIs: every screen template is decoded as
        // grayscale, so converting each ROI separately only burns CPU.
        let mut gray = Mat::default();
        imgproc::cvt_color(
            full_bgr,
            &mut gray,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut probes = UiScreenProbes::default();
        if let Some(roi) = top_left {
            probes.paimon_menu =
                probe_in_gray(&gray, roi, &self.paimon_menu, scale, SCREEN_THRESHOLD)?;
            probes.in_domain = probe_in_gray(&gray, roi, &self.in_domain, scale, SCREEN_THRESHOLD)?;
        }
        if let Some(roi) = top_right {
            probes.inventory = probe_in_gray(&gray, roi, &self.inventory, scale, SCREEN_THRESHOLD)?;
        }
        if let Some(roi) = cook_roi {
            probes.cooking_page =
                probe_in_gray(&gray, roi, &self.cook_icon, scale, SCREEN_THRESHOLD)?;
        }
        if let Some(roi) = clamp_roi(0, 0, cw / 3, ch / 8, cw, ch) {
            probes.talk = probe_in_gray(&gray, roi, &self.talk, scale, SCREEN_THRESHOLD)?;
        }
        Ok(probes)
    }

    pub fn detect_auto_eat(&self, full_bgr: &Mat) -> Result<(bool, bool, bool), CvError> {
        let cw = full_bgr.cols();
        let ch = full_bgr.rows();
        if cw < 320 || ch < 240 {
            return Ok((false, false, false));
        }
        let scale = (cw as f64 / 1920.0).min(ch as f64 / 1080.0);
        let x = (808.0 * scale).round() as i32;
        let y = (1010.0 * scale).round() as i32;
        let low_hp = if x < cw && y < ch {
            let pixel = full_bgr.at_2d::<core::Vec3b>(y, x)?;
            pixel[0] == 90 && pixel[1] == 90 && pixel[2] == 255
        } else {
            false
        };
        // C# `AutoEat/Assets/Recognition.json`: RecoveryIcon is rect(1810s,778s,23s,23s)
        // and ResurrectionIcon is rect(1810s,778s,18s,19s) — same origin, different sizes.
        let mut gray = Mat::default();
        imgproc::cvt_color(
            full_bgr,
            &mut gray,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let probe = |template: &Mat, w: f64, h: f64| -> Result<bool, CvError> {
            let Some(roi) = clamp_roi(
                (1810.0 * scale).round() as i32,
                (778.0 * scale).round() as i32,
                (w * scale).round() as i32,
                (h * scale).round() as i32,
                cw,
                ch,
            ) else {
                return Ok(false);
            };
            let roi_img = Mat::roi(&gray, roi)?;
            Ok(match_scaled_template(&roi_img, template, roi, scale, 0.80)?.is_some())
        };
        let recovery = probe(&self.recovery, 23.0, 23.0)?;
        let resurrection = probe(&self.resurrection, 18.0, 19.0)?;
        Ok((low_hp, recovery, resurrection))
    }

    pub fn find_white_recover(&self, full_bgr: &Mat) -> Result<Option<(i32, i32)>, CvError> {
        let cw = full_bgr.cols();
        let ch = full_bgr.rows();
        if cw < 320 || ch < 240 {
            return Ok(None);
        }
        let scale = (cw as f64 / 1920.0).min(ch as f64 / 1080.0);
        detect_color_in_roi(
            full_bgr,
            Rect::new(0, 0, cw, ch),
            &self.white_recover,
            scale,
            0.95,
        )
    }

    pub fn find_white_confirm(&self, full_bgr: &Mat) -> Result<Option<(i32, i32)>, CvError> {
        let cw = full_bgr.cols();
        let ch = full_bgr.rows();
        if cw < 320 || ch < 240 {
            return Ok(None);
        }
        let scale = (cw as f64 / 1920.0).min(ch as f64 / 1080.0);
        detect_color_in_roi(
            full_bgr,
            Rect::new(0, 0, cw, ch),
            &self.white_confirm,
            scale,
            0.95,
        )
    }
}

fn probe_in_gray(
    gray: &Mat,
    roi: Rect,
    template: &Mat,
    scale: f64,
    threshold: f64,
) -> Result<ScreenProbe, CvError> {
    let roi_gray = Mat::roi(gray, roi)?;
    let Some((best, max_loc, tpl)) = match_scaled_template_raw(&roi_gray, template, roi, scale)?
    else {
        return Ok(ScreenProbe::default());
    };
    Ok(ScreenProbe::at(best, max_loc, &tpl, roi, threshold))
}

fn detect_color_in_roi(
    full_bgr: &Mat,
    roi: Rect,
    template: &Mat,
    scale: f64,
    threshold: f64,
) -> Result<Option<(i32, i32)>, CvError> {
    let roi_bgr = Mat::roi(full_bgr, roi)?;
    match_scaled_template(&roi_bgr, template, roi, scale, threshold)
        .map(|hit| hit.map(|(x, y, _)| (x, y)))
}

fn decode_png(bytes: &[u8], flags: i32) -> Result<Mat, CvError> {
    let vec = Vector::<u8>::from_slice(bytes);
    let mat = imdecode(&vec, flags)?;
    if mat.empty() {
        return Err(CvError::InvalidImage("内置 UI 模板解码失败".into()));
    }
    Ok(mat)
}

fn decode_png_file(path: &Path, flags: i32) -> Result<Mat, CvError> {
    let bytes = std::fs::read(path).map_err(CvError::Io)?;
    decode_png(&bytes, flags)
}

fn clamp_roi(x: i32, y: i32, w: i32, h: i32, max_w: i32, max_h: i32) -> Option<Rect> {
    let x0 = x.clamp(0, max_w);
    let y0 = y.clamp(0, max_h);
    let x1 = (x + w).clamp(0, max_w);
    let y1 = (y + h).clamp(0, max_h);
    let rw = x1 - x0;
    let rh = y1 - y0;
    if rw <= 0 || rh <= 0 {
        None
    } else {
        Some(Rect::new(x0, y0, rw, rh))
    }
}

/// Peak template correlation inside `roi`, without applying a threshold.
/// Returns `None` when the (scaled) template does not fit the ROI.
fn match_scaled_template_raw(
    roi_img: &impl core::ToInputArray,
    tpl: &Mat,
    roi: Rect,
    scale: f64,
) -> Result<Option<(f64, core::Point, Mat)>, CvError> {
    let scaled_storage: Mat;
    let tpl_ref: &Mat = if (scale - 1.0).abs() > 0.02 {
        let tw = ((tpl.cols() as f64) * scale).round().max(4.0) as i32;
        let th = ((tpl.rows() as f64) * scale).round().max(4.0) as i32;
        let interp = if scale < 1.0 { INTER_AREA } else { INTER_CUBIC };
        let mut out = Mat::default();
        imgproc::resize(tpl, &mut out, Size::new(tw, th), 0.0, 0.0, interp)?;
        scaled_storage = out;
        &scaled_storage
    } else {
        tpl
    };

    if tpl_ref.cols() > roi.width || tpl_ref.rows() > roi.height {
        return Ok(None);
    }

    let mut res = Mat::default();
    imgproc::match_template(
        roi_img,
        tpl_ref,
        &mut res,
        TM_CCOEFF_NORMED,
        &core::no_array(),
    )?;
    let mut max_val = 0.0f64;
    let mut max_loc = core::Point::default();
    core::min_max_loc(
        &res,
        None,
        Some(&mut max_val),
        None,
        Some(&mut max_loc),
        &core::no_array(),
    )?;
    Ok(Some((max_val, max_loc, tpl_ref.try_clone()?)))
}

fn match_scaled_template(
    roi_img: &impl core::ToInputArray,
    tpl: &Mat,
    roi: Rect,
    scale: f64,
    threshold: f64,
) -> Result<Option<(i32, i32, f64)>, CvError> {
    let Some((max_val, max_loc, tpl_ref)) = match_scaled_template_raw(roi_img, tpl, roi, scale)?
    else {
        return Ok(None);
    };
    if max_val.is_finite() && max_val >= threshold {
        let cx = roi.x + max_loc.x + tpl_ref.cols() / 2;
        let cy = roi.y + max_loc.y + tpl_ref.rows() / 2;
        Ok(Some((cx, cy, max_val)))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use opencv::core::Rect;
    use opencv::imgcodecs::imread;
    use opencv::prelude::{MatExprTraitConst, MatTrait};

    use super::*;

    fn screen_with(template: &[u8], scale: f64) -> Mat {
        let mut screen = Mat::zeros(
            (1080.0 * scale).round() as i32,
            (1920.0 * scale).round() as i32,
            core::CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap();
        let template = decode_png(template, IMREAD_COLOR).unwrap();
        let mut scaled = Mat::default();
        imgproc::resize(
            &template,
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
        let mut target = Mat::roi_mut(
            &mut screen,
            Rect::new(
                (20.0 * scale).round() as i32,
                (20.0 * scale).round() as i32,
                scaled.cols(),
                scaled.rows(),
            ),
        )
        .unwrap();
        scaled.copy_to(&mut target).unwrap();
        screen
    }

    #[test]
    fn gameplay_follows_the_paimon_hud_button() {
        // The Paimon corner button is what C# `Bv.IsInMainUi` looks for.
        assert!(
            UiScreens {
                paimon_menu: true,
                ..Default::default()
            }
            .is_gameplay()
        );
        // HUD button plus inventory is still the main UI: both are常驻 elements.
        assert!(
            UiScreens {
                paimon_menu: true,
                inventory: true,
                ..Default::default()
            }
            .is_gameplay()
        );
        // Without it, nothing on screen proves we are in the game world.
        assert!(!UiScreens::default().is_gameplay());
        assert!(
            !UiScreens {
                inventory: true,
                ..Default::default()
            }
            .is_gameplay()
        );
        // Screens layered on top of the HUD still exclude gameplay.
        for screens in [
            UiScreens {
                paimon_menu: true,
                in_domain: true,
                ..Default::default()
            },
            UiScreens {
                paimon_menu: true,
                cooking_page: true,
                ..Default::default()
            },
            UiScreens {
                paimon_menu: true,
                talk: true,
                ..Default::default()
            },
        ] {
            assert!(!screens.is_gameplay(), "{screens:?}");
        }
    }

    #[test]
    fn screen_templates_match_at_reference_and_reduced_resolution() {
        let matcher = UiScreenMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut screen = Mat::zeros(
                (1080.0 * scale).round() as i32,
                (1920.0 * scale).round() as i32,
                core::CV_8UC3,
            )
            .unwrap()
            .to_mat()
            .unwrap();
            let mut target = Mat::roi_mut(
                &mut screen,
                Rect::new(
                    (1440.0 * scale) as i32,
                    0,
                    (f64::from(matcher.inventory.cols()) * scale).round() as i32,
                    (f64::from(matcher.inventory.rows()) * scale).round() as i32,
                ),
            )
            .unwrap();
            let mut inventory = Mat::default();
            let inventory_bgr = decode_png(INVENTORY_PNG, IMREAD_COLOR).unwrap();
            imgproc::resize(
                &inventory_bgr,
                &mut inventory,
                Size::new(target.cols(), target.rows()),
                0.0,
                0.0,
                INTER_AREA,
            )
            .unwrap();
            inventory.copy_to(&mut target).unwrap();
            let screens = matcher.detect(&screen).unwrap();
            assert!(screens.inventory, "scale={scale}: {screens:?}");
            assert!(!screens.paimon_menu && !screens.in_domain && !screens.cooking_page);
        }
    }

    #[test]
    fn menu_and_domain_templates_match_only_their_own_screen() {
        let matcher = UiScreenMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let menu = matcher
                .detect(&screen_with(PAIMON_MENU_PNG, scale))
                .unwrap();
            assert!(menu.paimon_menu, "scale={scale}: {menu:?}");
            assert!(!menu.inventory && !menu.in_domain && !menu.cooking_page);

            let domain = matcher.detect(&screen_with(IN_DOMAIN_PNG, scale)).unwrap();
            assert!(domain.in_domain, "scale={scale}: {domain:?}");
            assert!(!domain.paimon_menu && !domain.inventory && !domain.cooking_page);
        }
    }

    #[test]
    fn cooking_icon_and_white_confirmation_are_detected_independently() {
        let matcher = UiScreenMatcher::new().unwrap();
        let cooking = screen_with(COOK_ICON_PNG, 1.0);
        assert!(matcher.detect(&cooking).unwrap().cooking_page);
        assert!(matcher.find_white_confirm(&cooking).unwrap().is_none());

        for scale in [1.0_f64, 0.5] {
            let confirmation = screen_with(WHITE_CONFIRM_PNG, scale);
            assert!(matcher.find_white_confirm(&confirmation).unwrap().is_some());
            assert!(matcher.find_white_recover(&confirmation).unwrap().is_none());

            let recover = screen_with(WHITE_RECOVER_PNG, scale);
            assert!(matcher.find_white_recover(&recover).unwrap().is_some());
            assert!(matcher.find_white_confirm(&recover).unwrap().is_none());
        }
    }

    #[test]
    fn talk_template_is_detected_in_its_reference_region() {
        let matcher = UiScreenMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let screens = matcher
                .detect(&screen_with(TALK_DISABLED_UI_PNG, scale))
                .unwrap();
            assert!(screens.talk, "scale={scale}: {screens:?}");
            assert!(!screens.cooking_page && !screens.inventory);
        }
    }

    #[test]
    fn probe_scores_agree_with_detect_decisions() {
        let matcher = UiScreenMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            for (template, pick) in [
                (PAIMON_MENU_PNG, 0usize),
                (IN_DOMAIN_PNG, 1),
                (COOK_ICON_PNG, 2),
                (TALK_DISABLED_UI_PNG, 3),
            ] {
                let screen = screen_with(template, scale);
                let probes = matcher.probe(&screen).unwrap();
                assert_eq!(probes.screens(), matcher.detect(&screen).unwrap());
                assert_eq!(probes.is_gameplay(), probes.screens().is_gameplay());
                let picked = [
                    probes.paimon_menu,
                    probes.in_domain,
                    probes.cooking_page,
                    probes.talk,
                ][pick];
                assert!(
                    picked.hit.is_some(),
                    "scale={scale} pick={pick}: {probes:?}"
                );
                assert!(picked.best >= SCREEN_THRESHOLD, "{picked:?}");
            }
        }
    }

    #[test]
    fn blank_frame_reports_low_scores_and_is_not_gameplay() {
        let matcher = UiScreenMatcher::new().unwrap();
        let screen = Mat::zeros(1080, 1920, core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let probes = matcher.probe(&screen).unwrap();
        // No Paimon corner button means no evidence of the in-game HUD.
        assert!(!probes.is_gameplay(), "{probes:?}");
        for probe in [
            probes.paimon_menu,
            probes.inventory,
            probes.in_domain,
            probes.cooking_page,
            probes.talk,
        ] {
            assert!(probe.hit.is_none(), "{probe:?}");
            assert!(probe.best < SCREEN_THRESHOLD, "{probe:?}");
        }
    }

    #[test]
    fn auto_eat_detects_low_hp_and_recovery_or_resurrection_icons() {
        let matcher = UiScreenMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            for (template, expected) in [
                (RECOVERY_PNG, (true, false)),
                (RESURRECTION_PNG, (false, true)),
            ] {
                let mut screen = Mat::zeros(
                    (1080.0 * scale).round() as i32,
                    (1920.0 * scale).round() as i32,
                    core::CV_8UC3,
                )
                .unwrap()
                .to_mat()
                .unwrap();
                let pixel = screen
                    .at_2d_mut::<core::Vec3b>(
                        (1010.0 * scale).round() as i32,
                        (808.0 * scale).round() as i32,
                    )
                    .unwrap();
                *pixel = core::Vec3b::from([90, 90, 255]);
                let icon = decode_png(template, IMREAD_COLOR).unwrap();
                let mut scaled = Mat::default();
                imgproc::resize(
                    &icon,
                    &mut scaled,
                    Size::new(
                        (f64::from(icon.cols()) * scale).round() as i32,
                        (f64::from(icon.rows()) * scale).round() as i32,
                    ),
                    0.0,
                    0.0,
                    INTER_AREA,
                )
                .unwrap();
                let mut target = Mat::roi_mut(
                    &mut screen,
                    Rect::new(
                        (1810.0 * scale).round() as i32,
                        (778.0 * scale).round() as i32,
                        scaled.cols(),
                        scaled.rows(),
                    ),
                )
                .unwrap();
                scaled.copy_to(&mut target).unwrap();
                assert_eq!(
                    matcher.detect_auto_eat(&screen).unwrap(),
                    (true, expected.0, expected.1)
                );
            }
        }
    }

    #[test]
    fn detects_underground_layered_map_and_ignores_surface_map() {
        let matcher = UiTemplateMatcher::new().unwrap();

        let underground_img = Path::new(
            r"d:\bgiprro\bettergi-rs\target\release\log\failed\20261125-105423_157-bigmap_low_inliers-3.png",
        );
        if underground_img.exists() {
            let mat = imread(&*underground_img.to_string_lossy(), IMREAD_COLOR).unwrap();
            let btns = matcher.detect_bigmap_layer_buttons(&mat).unwrap();
            assert!(
                btns.to_ground.is_some(),
                "expected to-ground button on 105423 underground screenshot: {btns:?}"
            );
        }

        let surface_img = Path::new(
            r"d:\bgiprro\bettergi-rs\target\release\log\failed\20261125-103243_283-bigmap_sift_fail-3.png",
        );
        if surface_img.exists() {
            let mat = imread(&*surface_img.to_string_lossy(), IMREAD_COLOR).unwrap();
            let btns = matcher.detect_bigmap_layer_buttons(&mat).unwrap();
            assert!(
                btns.to_ground.is_none() && btns.underground_switch.is_none(),
                "surface bigmap should not trigger layered-map switch: {btns:?}"
            );
        }
    }
}
