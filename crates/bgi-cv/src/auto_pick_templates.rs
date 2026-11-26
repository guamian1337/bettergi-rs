use std::path::Path;

use bgi_task::{AutoPickPrompt, AutoPickTextRoi};
use opencv::core::{self, Mat, MatTraitConst, Point, Rect, Size, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{self, COLOR_BGR2GRAY, INTER_AREA, INTER_CUBIC, TM_CCOEFF_NORMED};

use crate::CvError;

const PICK_PNG: &[u8] = include_bytes!("../assets/ui/auto_pick_F.png");
const L_PNG: &[u8] = include_bytes!("../assets/ui/auto_pick_L.png");
const SETTINGS_PNG: &[u8] = include_bytes!("../assets/ui/auto_pick_settings.png");
const CHAT_PNG: &[u8] = include_bytes!("../assets/ui/auto_pick_chat.png");
const BASE_WIDTH: i32 = 1920;
const BASE_HEIGHT: i32 = 1080;
const MATCH_THRESHOLD: f64 = 0.80;
const MAX_ASPECT_ERROR: f64 = 0.01;
const PICK_ROI: (i32, i32, i32, i32) = (1090, 330, 60, 420);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AutoPickObservation {
    pub prompt: Option<AutoPickPrompt>,
    pub blocked: bool,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct AutoPickPromptMatcher {
    pick: Mat,
    l: Mat,
    settings: Mat,
    chat: Mat,
}

impl AutoPickPromptMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            pick: decode_png(PICK_PNG)?,
            l: decode_png(L_PNG)?,
            settings: decode_png(SETTINGS_PNG)?,
            chat: decode_png(CHAT_PNG)?,
        })
    }

    pub fn from_assets(
        pick: impl AsRef<Path>,
        l: impl AsRef<Path>,
        settings: impl AsRef<Path>,
        chat: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            pick: decode_file(pick.as_ref())?,
            l: decode_file(l.as_ref())?,
            settings: decode_file(settings.as_ref())?,
            chat: decode_file(chat.as_ref())?,
        })
    }

    pub fn observe(&self, full_bgr: &Mat) -> Result<AutoPickObservation, CvError> {
        if full_bgr.empty() {
            return Ok(AutoPickObservation::default());
        }
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        let mut observation = AutoPickObservation {
            width: width as u32,
            height: height as u32,
            ..AutoPickObservation::default()
        };
        if width < 320 || height < 240 {
            return Ok(observation);
        }
        let aspect = f64::from(width) / f64::from(height);
        if (aspect - f64::from(BASE_WIDTH) / f64::from(BASE_HEIGHT)).abs() > MAX_ASPECT_ERROR {
            return Ok(observation);
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
        let Some(pick_roi) = scaled_roi(PICK_ROI, scale, width, height) else {
            return Ok(observation);
        };
        let pick_image = Mat::roi(&gray, pick_roi)?;
        let Some(pick) =
            match_scaled_template(&pick_image, &self.pick, scale, pick_roi, MATCH_THRESHOLD)?
        else {
            return Ok(observation);
        };

        let l_roi = scaled_roi((BASE_WIDTH - 110, 550, 70, 100), scale, width, height);
        if let Some(roi) = l_roi {
            let image = Mat::roi(&gray, roi)?;
            if match_scaled_template(&image, &self.l, scale, roi, MATCH_THRESHOLD)?.is_some() {
                observation.blocked = true;
                return Ok(observation);
            }
        }

        let icon_roi = relative_roi(
            pick.rect,
            (60, 0),
            (55, self.pick.rows()),
            scale,
            (width, height),
        );
        let Some(icon_roi) = icon_roi else {
            observation.blocked = true;
            return Ok(observation);
        };
        let icon_image = Mat::roi(&gray, icon_roi)?;
        for template in [&self.chat, &self.settings] {
            if match_scaled_template(&icon_image, template, scale, icon_roi, MATCH_THRESHOLD)?
                .is_some()
            {
                observation.blocked = true;
                return Ok(observation);
            }
        }

        let Some(text_rect) = relative_roi(
            pick.rect,
            (115, 0),
            (285, self.pick.rows()),
            scale,
            (width, height),
        ) else {
            observation.blocked = true;
            return Ok(observation);
        };
        observation.prompt = Some(AutoPickPrompt {
            identity: prompt_identity(width, height, text_rect),
            text_roi: AutoPickTextRoi {
                x0: normalized_lower(text_rect.x, width),
                y0: normalized_lower(text_rect.y, height),
                x1: normalized_upper(text_rect.x + text_rect.width, width),
                y1: normalized_upper(text_rect.y + text_rect.height, height),
            },
        });
        Ok(observation)
    }
}

#[derive(Debug, Clone, Copy)]
struct TemplateMatch {
    rect: Rect,
}

fn scaled_roi(roi: (i32, i32, i32, i32), scale: f64, width: i32, height: i32) -> Option<Rect> {
    let (x, y, w, h) = roi;
    let right = (f64::from(x + w) * scale).round() as i32;
    let bottom = (f64::from(y + h) * scale).round() as i32;
    let x = (f64::from(x) * scale).round() as i32;
    let y = (f64::from(y) * scale).round() as i32;
    rect(
        x,
        y,
        right.saturating_sub(x),
        bottom.saturating_sub(y),
        width,
        height,
    )
}

fn relative_roi(
    prompt: Rect,
    offset: (i32, i32),
    size: (i32, i32),
    scale: f64,
    frame_size: (i32, i32),
) -> Option<Rect> {
    let x = prompt.x + (f64::from(offset.0) * scale).round() as i32;
    let y = prompt.y + (f64::from(offset.1) * scale).round() as i32;
    let width = (f64::from(size.0) * scale).round().max(1.0) as i32;
    let height = (f64::from(size.1) * scale).round().max(1.0) as i32;
    rect(x, y, width, height, frame_size.0, frame_size.1)
}

fn rect(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    frame_width: i32,
    frame_height: i32,
) -> Option<Rect> {
    if width <= 0 || height <= 0 || x >= frame_width || y >= frame_height {
        return None;
    }
    let x = x.max(0);
    let y = y.max(0);
    let width = width.min(frame_width - x);
    let height = height.min(frame_height - y);
    (width > 0 && height > 0).then_some(Rect::new(x, y, width, height))
}

fn match_scaled_template(
    image: &impl core::ToInputArray,
    template: &Mat,
    scale: f64,
    roi: Rect,
    threshold: f64,
) -> Result<Option<TemplateMatch>, CvError> {
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
    Ok(
        (score.is_finite() && score >= threshold).then_some(TemplateMatch {
            rect: Rect::new(
                roi.x + location.x,
                roi.y + location.y,
                template.cols(),
                template.rows(),
            ),
        }),
    )
}

fn normalized_lower(value: i32, size: i32) -> f32 {
    let mut normalized = value as f32 / size as f32;
    for _ in 0..2 {
        if (normalized * size as f32).floor() >= value as f32 {
            break;
        }
        normalized = normalized.next_up();
    }
    normalized
}

fn normalized_upper(value: i32, size: i32) -> f32 {
    let mut normalized = value as f32 / size as f32;
    for _ in 0..2 {
        if (normalized * size as f32).ceil() <= value as f32 {
            break;
        }
        normalized = normalized.next_down();
    }
    normalized
}

fn prompt_identity(width: i32, height: i32, text_rect: Rect) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in [
        width as u64,
        height as u64,
        text_rect.x as u64,
        text_rect.y as u64,
        text_rect.width as u64,
        text_rect.height as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("自动拾取模板解码失败".into()));
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
    use opencv::prelude::{MatExprTraitConst, MatTrait};

    use super::*;

    fn paste_scaled(mut image: Mat, template: &Mat, x: i32, y: i32, scale: f64) -> Mat {
        let mut resized = Mat::default();
        imgproc::resize(
            template,
            &mut resized,
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
            &resized,
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

    fn screen(scale: f64, prompt: bool, l: bool, unsafe_icon: Option<&[u8]>) -> Mat {
        let mut image = Mat::zeros(
            (f64::from(BASE_HEIGHT) * scale).round() as i32,
            (f64::from(BASE_WIDTH) * scale).round() as i32,
            CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap();
        if prompt {
            image = paste_scaled(
                image,
                &decode_png(PICK_PNG).unwrap(),
                (1110.0 * scale).round() as i32,
                (500.0 * scale).round() as i32,
                scale,
            );
        }
        if l {
            image = paste_scaled(
                image,
                &decode_png(L_PNG).unwrap(),
                (1820.0 * scale).round() as i32,
                (560.0 * scale).round() as i32,
                scale,
            );
        }
        if let Some(icon) = unsafe_icon {
            image = paste_scaled(
                image,
                &decode_png(icon).unwrap(),
                (1170.0 * scale).round() as i32,
                (500.0 * scale).round() as i32,
                scale,
            );
        }
        image
    }

    #[test]
    fn detects_prompt_and_anchors_text_roi_at_reference_and_reduced_resolution() {
        let matcher = AutoPickPromptMatcher::new().unwrap();
        for scale in [1.0_f64, 2.0 / 3.0] {
            let observation = matcher.observe(&screen(scale, true, false, None)).unwrap();
            let prompt = observation
                .prompt
                .unwrap_or_else(|| panic!("scale={scale}: {observation:?}"));
            assert!(!observation.blocked, "scale={scale}: {observation:?}");
            let width = (f64::from(BASE_WIDTH) * scale).round() as i32;
            let height = (f64::from(BASE_HEIGHT) * scale).round() as i32;
            assert_eq!(
                (prompt.text_roi.x0 * width as f32).floor() as i32,
                (1225.0 * scale).round() as i32
            );
            assert_eq!(
                (prompt.text_roi.x1 * width as f32).ceil() as i32,
                (1510.0 * scale).round() as i32
            );
            assert_eq!(
                (prompt.text_roi.y0 * height as f32).floor() as i32,
                (500.0 * scale).round() as i32
            );
            assert_eq!(
                (prompt.text_roi.y1 * height as f32).ceil() as i32,
                (500.0 * scale).round() as i32 + (32.0 * scale).round() as i32,
                "scale={scale}: {prompt:?}"
            );
        }
    }

    #[test]
    fn missing_prompt_or_prompt_outside_roi_never_produces_candidate() {
        let matcher = AutoPickPromptMatcher::new().unwrap();
        assert!(
            matcher
                .observe(&screen(1.0, false, false, None))
                .unwrap()
                .prompt
                .is_none()
        );
        let mut outside = screen(1.0, false, false, None);
        outside = paste_scaled(outside, &decode_png(PICK_PNG).unwrap(), 1300, 500, 1.0);
        assert!(matcher.observe(&outside).unwrap().prompt.is_none());
    }

    #[test]
    fn non_widescreen_frames_fail_closed_before_roi_matching() {
        let matcher = AutoPickPromptMatcher::new().unwrap();
        let mut image = Mat::zeros(1200, 1600, CV_8UC3).unwrap().to_mat().unwrap();
        image = paste_scaled(image, &decode_png(PICK_PNG).unwrap(), 925, 566, 5.0 / 6.0);
        image = paste_scaled(image, &decode_png(L_PNG).unwrap(), 1517, 617, 5.0 / 6.0);
        let observation = matcher.observe(&image).unwrap();
        assert_eq!((observation.width, observation.height), (1600, 1200));
        assert!(observation.prompt.is_none());
    }

    #[test]
    fn l_key_chat_and_settings_icons_block_pickup() {
        let matcher = AutoPickPromptMatcher::new().unwrap();
        for (l, icon) in [
            (true, None),
            (false, Some(CHAT_PNG)),
            (false, Some(SETTINGS_PNG)),
        ] {
            let observation = matcher.observe(&screen(1.0, true, l, icon)).unwrap();
            assert!(observation.blocked);
            assert!(observation.prompt.is_none());
        }
    }

    #[test]
    fn empty_and_non_bgr_frames_fail_closed() {
        let matcher = AutoPickPromptMatcher::new().unwrap();
        assert_eq!(
            matcher.observe(&Mat::default()).unwrap(),
            AutoPickObservation::default()
        );
        let mut gray = Mat::zeros(1080, 1920, CV_8UC3).unwrap().to_mat().unwrap();
        let mut single = Mat::default();
        imgproc::cvt_color(
            &gray,
            &mut single,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        assert!(matcher.observe(&single).is_err());
        gray.set_to(&core::Scalar::all(0.0), &core::no_array())
            .unwrap();
        assert!(matcher.observe(&gray).unwrap().prompt.is_none());
    }

    #[test]
    fn external_assets_use_the_same_prompt_templates() -> Result<(), CvError> {
        let templates = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/Templates");
        let assets =
            crate::assets::AssetCatalog::load(templates.parent().unwrap(), "zh-CN").unwrap();
        let matcher = AutoPickPromptMatcher::from_assets(
            assets
                .template("ui.auto_pick.pick")
                .map_err(|e| CvError::InvalidImage(e.to_string()))?,
            assets
                .template("ui.auto_pick.l")
                .map_err(|e| CvError::InvalidImage(e.to_string()))?,
            assets
                .template("ui.auto_pick.settings")
                .map_err(|e| CvError::InvalidImage(e.to_string()))?,
            assets
                .template("ui.auto_pick.chat")
                .map_err(|e| CvError::InvalidImage(e.to_string()))?,
        )
        .unwrap();
        assert!(
            matcher
                .observe(&screen(1.0, true, false, None))?
                .prompt
                .is_some()
        );
        Ok(())
    }
}
