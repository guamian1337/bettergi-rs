use std::path::Path;

use bgi_task::fishing::{FishingBar, FishingObservation, FishingRect};
use opencv::core::{self, Mat, MatTraitConst, Point, Rect, Scalar, Size, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{
    self, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, COLOR_BGR2HSV_FULL, INTER_AREA, INTER_CUBIC,
    RETR_EXTERNAL, TM_CCOEFF_NORMED,
};

use crate::CvError;

const WAIT_BITE_PNG: &[u8] = include_bytes!("../assets/ui/fishing_wait_bite.png");
const LIFT_ROD_PNG: &[u8] = include_bytes!("../assets/ui/fishing_lift_rod.png");
const EXIT_FISHING_PNG: &[u8] = include_bytes!("../assets/ui/fishing_exit_fishing.png");
const BASE_WIDTH: i32 = 1920;
const BASE_HEIGHT: i32 = 1080;
const PROMPT_ROI: (i32, i32, i32, i32) = (1440, 540, 480, 540);
const EXIT_ROI: (i32, i32, i32, i32) = (1780, 930, 140, 150);
const PROMPT_THRESHOLD: f64 = 0.7;
const EXIT_THRESHOLD: f64 = 0.8;
const MAX_ANGLE_ERROR: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct FishingMatcher {
    exit_fishing: Mat,
    wait_bite: Mat,
    lift_rod: Mat,
}

impl FishingMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            exit_fishing: decode_png(EXIT_FISHING_PNG)?,
            wait_bite: decode_png(WAIT_BITE_PNG)?,
            lift_rod: decode_png(LIFT_ROD_PNG)?,
        })
    }

    pub fn from_assets(
        exit_fishing: impl AsRef<Path>,
        wait_bite: impl AsRef<Path>,
        lift_rod: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            exit_fishing: decode_file(exit_fishing.as_ref())?,
            wait_bite: decode_file(wait_bite.as_ref())?,
            lift_rod: decode_file(lift_rod.as_ref())?,
        })
    }

    pub fn detect(
        &self,
        full_bgr: &Mat,
        bite_text: Option<&str>,
    ) -> Result<FishingObservation, CvError> {
        if full_bgr.empty() {
            return Ok(FishingObservation {
                in_fishing: false,
                bite: false,
                bar: None,
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
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        let scale = (f64::from(width) / f64::from(BASE_WIDTH))
            .min(f64::from(height) / f64::from(BASE_HEIGHT));

        let in_fishing =
            matches_template_in_roi(&gray, &self.exit_fishing, EXIT_ROI, scale, EXIT_THRESHOLD)?;
        // Keep the wait-bite check alongside lift-rod for diagnostics. It is not
        // a bite signal: only the lift prompt or OCR advances the state machine.
        let _wait_bite =
            matches_template_in_roi(&gray, &self.wait_bite, PROMPT_ROI, scale, PROMPT_THRESHOLD)?;
        let lift_rod =
            matches_template_in_roi(&gray, &self.lift_rod, PROMPT_ROI, scale, PROMPT_THRESHOLD)?;
        let bar = detect_fishing_bar(full_bgr)?;

        Ok(FishingObservation {
            in_fishing,
            bite: lift_rod || bite_text.is_some_and(ocr_has_bite),
            bar,
        })
    }
}

fn matches_template_in_roi(
    gray: &Mat,
    template: &Mat,
    roi: (i32, i32, i32, i32),
    scale: f64,
    threshold: f64,
) -> Result<bool, CvError> {
    let Some(roi) = scaled_roi(roi, scale, gray.cols(), gray.rows()) else {
        return Ok(false);
    };
    let image = Mat::roi(gray, roi)?;
    Ok(match_scaled_template(&image, template, scale, threshold)?.is_some())
}

fn scaled_roi(roi: (i32, i32, i32, i32), scale: f64, width: i32, height: i32) -> Option<Rect> {
    let (x, y, w, h) = roi;
    let right = (f64::from(x + w) * scale).round() as i32;
    let bottom = (f64::from(y + h) * scale).round() as i32;
    let x = (f64::from(x) * scale).round() as i32;
    let y = (f64::from(y) * scale).round() as i32;
    let x = x.clamp(0, width);
    let y = y.clamp(0, height);
    let right = right.clamp(0, width);
    let bottom = bottom.clamp(0, height);
    let roi_width = right - x;
    let roi_height = bottom - y;
    (roi_width > 0 && roi_height > 0).then_some(Rect::new(x, y, roi_width, roi_height))
}

fn match_scaled_template(
    image: &(impl MatTraitConst + core::ToInputArray),
    template: &Mat,
    scale: f64,
    threshold: f64,
) -> Result<Option<f64>, CvError> {
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
    core::min_max_loc(
        &result,
        None,
        Some(&mut score),
        None,
        None,
        &core::no_array(),
    )?;
    Ok((score.is_finite() && score >= threshold).then_some(score))
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("内置钓鱼模板解码失败".into()));
    }
    Ok(decoded)
}

fn decode_file(path: &Path) -> Result<Mat, CvError> {
    decode_png(&std::fs::read(path)?)
}

fn ocr_has_bite(text: &str) -> bool {
    let normalized = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    ["上钩", "上鉤", "bite", "mordu", "bboccato"]
        .iter()
        .any(|keyword| normalized.contains(keyword))
}

fn detect_fishing_bar(full_bgr: &Mat) -> Result<Option<FishingBar>, CvError> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        full_bgr,
        &mut hsv,
        COLOR_BGR2HSV_FULL,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut mask = Mat::default();
    // C# `AutoFishingImageRecognition` derives the hue window from RGB(255,255,192)
    // ("拉条框的黄色") via CommonHSV2OpenCVHSVFull(new Scalar(60, 0.25, 1.00)) and then
    // widens it by -3/+3.5 on H, -20/+40 on S, -10/0 on V, in HSV_FULL units.
    core::in_range(
        &hsv,
        &Scalar::new(39.5, 43.75, 245.0, 0.0),
        &Scalar::new(46.0, 103.75, 255.0, 0.0),
        &mut mask,
    )?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::default(),
    )?;
    let mut boxes = Vec::new();
    for contour in contours {
        let rotated = imgproc::min_area_rect(&contour)?;
        let angle = rotated.angle.rem_euclid(45.0);
        if angle.min(45.0 - angle) > MAX_ANGLE_ERROR {
            continue;
        }
        let bounds = imgproc::bounding_rect(&contour)?;
        if bounds.width > 0 && bounds.height > 0 {
            boxes.push(bounds);
        }
    }

    let Some(baseline) = boxes.iter().max_by_key(|rect| rect.width) else {
        return Ok(None);
    };
    let baseline_height = i64::from(baseline.height);
    let baseline_center_y = i64::from(baseline.y) * 2 + i64::from(baseline.height);
    boxes.retain(|rect| {
        let center_y = i64::from(rect.y) * 2 + i64::from(rect.height);
        (center_y - baseline_center_y).abs() * 5 < baseline_height
            && (i64::from(rect.height) - baseline_height).abs() * 3 < baseline_height
            && i64::from(rect.width) * 4 > baseline_height
    });
    if boxes.len() > 3 {
        boxes.sort_by(|left, right| {
            right
                .height
                .cmp(&left.height)
                .then_with(|| left.x.cmp(&right.x))
                .then_with(|| left.y.cmp(&right.y))
        });
        boxes.truncate(3);
    }
    boxes.sort_by_key(|rect| rect.x);
    if boxes.len() < 2 {
        return Ok(None);
    }

    let (cursor, target) = if boxes.len() == 2 {
        let [cursor, target] = boxes.as_slice() else {
            unreachable!("length checked above");
        };
        (*cursor, *target)
    } else {
        let [left, cursor, right] = boxes.as_slice() else {
            unreachable!("more than three boxes were truncated");
        };
        (
            *cursor,
            Rect::new(left.x, left.y, right.x + right.width - left.x, left.height),
        )
    };
    Ok(validate_bar(
        cursor,
        target,
        full_bgr.cols(),
        full_bgr.rows(),
    ))
}

fn validate_bar(cursor: Rect, target: Rect, width: i32, height: i32) -> Option<FishingBar> {
    if cursor.width <= 0
        || cursor.height <= 0
        || target.width <= 0
        || target.height <= 0
        || cursor.width > target.width
        || i64::from(target.width) < 10 * i64::from(cursor.width)
        || i64::from(cursor.x) * 2 + i64::from(cursor.width)
            >= i64::from(target.x) * 2 + i64::from(target.width)
        || cursor.x < 0
        || cursor.y < 0
        || target.x < 0
        || target.y < 0
        || cursor.x + cursor.width > width
        || cursor.y + cursor.height > height
        || target.x + target.width > width
        || target.y + target.height > height
    {
        return None;
    }
    Some(FishingBar {
        cursor: FishingRect {
            x: cursor.x,
            y: cursor.y,
            width: cursor.width,
            height: cursor.height,
        },
        target: FishingRect {
            x: target.x,
            y: target.y,
            width: target.width,
            height: target.height,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use opencv::core::{CV_8UC3, Vec3b};
    use opencv::prelude::{MatExprTraitConst, MatTrait};

    use super::*;

    fn screen(scale: f64) -> Mat {
        Mat::zeros(
            (f64::from(BASE_HEIGHT) * scale).round() as i32,
            (f64::from(BASE_WIDTH) * scale).round() as i32,
            CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap()
    }

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

    fn draw_rect(image: &mut Mat, x: i32, y: i32, width: i32, height: i32, color: Vec3b) {
        for row in y..y + height {
            for column in x..x + width {
                *image.at_2d_mut::<Vec3b>(row, column).unwrap() = color;
            }
        }
    }

    fn draw_bar(image: &mut Mat, x: i32, y: i32, cursor_offset: i32, scale: f64) {
        // BGR (192, 255, 255) is RGB(255, 255, 192): the yellow C# samples for the bar
        // ("拉条框的黄色是：RGB 255, 255, 192").
        let color = Vec3b::from([192, 255, 255]);
        let target_width = (240.0 * scale).round() as i32;
        let target_height = (20.0 * scale).round() as i32;
        draw_rect(image, x, y, target_width, target_height, color);
        let cursor_x = x + cursor_offset;
        let cursor_y = y + (3.0 * scale).round() as i32;
        let cursor_width = (8.0 * scale).round() as i32;
        let cursor_height = (14.0 * scale).round() as i32;
        draw_rect(
            image,
            cursor_x,
            cursor_y,
            cursor_width,
            cursor_height,
            color,
        );
    }

    fn draw_split_bar(image: &mut Mat, x: i32, y: i32, cursor_offset: i32, scale: f64) {
        let color = Vec3b::from([192, 255, 255]);
        let segment_width = (80.0 * scale).round() as i32;
        let target_height = (20.0 * scale).round() as i32;
        draw_rect(image, x, y, segment_width, target_height, color);
        draw_rect(
            image,
            x + cursor_offset,
            y + (3.0 * scale).round() as i32,
            (8.0 * scale).round() as i32,
            (14.0 * scale).round() as i32,
            color,
        );
        draw_rect(
            image,
            x + (200.0 * scale).round() as i32,
            y,
            segment_width,
            target_height,
            color,
        );
    }

    #[test]
    fn templates_match_at_reference_and_half_resolution() {
        let matcher = FishingMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut image = screen(scale);
            image = paste_scaled(
                image,
                &decode_png(WAIT_BITE_PNG).unwrap(),
                (1500.0 * scale).round() as i32,
                (600.0 * scale).round() as i32,
                scale,
            );
            image = paste_scaled(
                image,
                &decode_png(EXIT_FISHING_PNG).unwrap(),
                (1820.0 * scale).round() as i32,
                (960.0 * scale).round() as i32,
                scale,
            );
            let observation = matcher.detect(&image, None).unwrap();
            assert!(observation.in_fishing, "scale={scale}: {observation:?}");
            assert!(
                !observation.bite,
                "wait-bite is not a bite: {observation:?}"
            );

            image = paste_scaled(
                image,
                &decode_png(LIFT_ROD_PNG).unwrap(),
                (1560.0 * scale).round() as i32,
                (620.0 * scale).round() as i32,
                scale,
            );
            let observation = matcher.detect(&image, None).unwrap();
            assert!(observation.in_fishing, "scale={scale}: {observation:?}");
            assert!(observation.bite, "scale={scale}: {observation:?}");
        }
    }

    #[test]
    fn templates_outside_their_rois_are_ignored() {
        let matcher = FishingMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut image = screen(scale);
            image = paste_scaled(
                image,
                &decode_png(LIFT_ROD_PNG).unwrap(),
                (1200.0 * scale).round() as i32,
                (600.0 * scale).round() as i32,
                scale,
            );
            image = paste_scaled(
                image,
                &decode_png(EXIT_FISHING_PNG).unwrap(),
                (1700.0 * scale).round() as i32,
                (960.0 * scale).round() as i32,
                scale,
            );
            let observation = matcher.detect(&image, None).unwrap();
            assert!(!observation.in_fishing, "scale={scale}: {observation:?}");
            assert!(!observation.bite, "scale={scale}: {observation:?}");
        }
    }

    #[test]
    fn external_assets_match_the_builtin_templates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/ui");
        let external = FishingMatcher::from_assets(
            root.join("fishing_exit_fishing.png"),
            root.join("fishing_wait_bite.png"),
            root.join("fishing_lift_rod.png"),
        )
        .unwrap();
        let builtin = FishingMatcher::new().unwrap();
        let mut image = screen(1.0);
        for (bytes, x, y) in [
            (EXIT_FISHING_PNG, 1820, 960),
            (WAIT_BITE_PNG, 1500, 600),
            (LIFT_ROD_PNG, 1560, 620),
        ] {
            image = paste_scaled(image, &decode_png(bytes).unwrap(), x, y, 1.0);
        }
        let expected = builtin.detect(&image, None).unwrap();
        let actual = external.detect(&image, None).unwrap();
        assert_eq!(actual.in_fishing, expected.in_fishing);
        assert_eq!(actual.bite, expected.bite);
        assert_eq!(actual.bar.is_some(), expected.bar.is_some());
    }

    #[test]
    fn detects_two_box_bar_at_reference_and_half_resolution() {
        let matcher = FishingMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut image = screen(scale);
            draw_bar(
                &mut image,
                (600.0 * scale).round() as i32,
                (400.0 * scale).round() as i32,
                (-20.0 * scale).round() as i32,
                scale,
            );
            let bar = matcher.detect(&image, None).unwrap().bar.unwrap();
            assert_eq!(bar.cursor.x, (580.0 * scale).round() as i32);
            assert_eq!(bar.cursor.y, (403.0 * scale).round() as i32);
            assert_eq!(bar.cursor.width, (8.0 * scale).round() as i32);
            assert_eq!(bar.cursor.height, (14.0 * scale).round() as i32);
            assert_eq!(bar.target.x, (600.0 * scale).round() as i32);
            assert_eq!(bar.target.y, (400.0 * scale).round() as i32);
            assert_eq!(bar.target.width, (240.0 * scale).round() as i32);
            assert_eq!(bar.target.height, (20.0 * scale).round() as i32);
        }
    }

    #[test]
    fn detects_split_three_box_bar_at_reference_and_half_resolution() {
        let matcher = FishingMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut image = screen(scale);
            draw_split_bar(
                &mut image,
                (600.0 * scale).round() as i32,
                (400.0 * scale).round() as i32,
                (100.0 * scale).round() as i32,
                scale,
            );
            let bar = matcher.detect(&image, None).unwrap().bar.unwrap();
            assert_eq!(bar.cursor.x, (700.0 * scale).round() as i32);
            assert_eq!(bar.cursor.width, (8.0 * scale).round() as i32);
            assert_eq!(bar.target.x, (600.0 * scale).round() as i32);
            assert_eq!(bar.target.y, (400.0 * scale).round() as i32);
            assert_eq!(bar.target.width, (280.0 * scale).round() as i32);
            assert_eq!(bar.target.height, (20.0 * scale).round() as i32);
        }
    }

    #[test]
    fn more_than_three_boxes_keep_the_three_tallest() {
        let matcher = FishingMatcher::new().unwrap();
        let mut image = screen(1.0);
        let color = Vec3b::from([192, 255, 255]);
        draw_rect(&mut image, 600, 400, 80, 24, color);
        draw_rect(&mut image, 700, 403, 8, 18, color);
        draw_rect(&mut image, 800, 400, 80, 24, color);
        draw_rect(&mut image, 920, 403, 80, 17, color);
        let bar = matcher.detect(&image, None).unwrap().bar.unwrap();
        assert_eq!(bar.cursor.x, 700);
        assert_eq!(bar.cursor.height, 18);
        assert_eq!(bar.target.x, 600);
        assert_eq!(bar.target.width, 280);
        assert_eq!(bar.target.height, 24);
    }

    #[test]
    fn unrelated_green_is_rejected() {
        let matcher = FishingMatcher::new().unwrap();
        let mut unrelated = screen(1.0);
        draw_rect(&mut unrelated, 600, 400, 240, 20, Vec3b::from([0, 255, 0]));
        draw_rect(&mut unrelated, 700, 403, 8, 14, Vec3b::from([0, 255, 0]));
        assert!(matcher.detect(&unrelated, None).unwrap().bar.is_none());

        let mut valid = unrelated;
        draw_bar(&mut valid, 600, 400, -20, 1.0);
        let bar = matcher.detect(&valid, None).unwrap().bar.unwrap();
        assert_eq!(bar.target.width, 240);
        assert_eq!(bar.cursor.width, 8);
    }

    #[test]
    fn ambiguous_or_invalid_bar_geometry_is_rejected() {
        let matcher = FishingMatcher::new().unwrap();
        let mut equal_widths = screen(1.0);
        draw_rect(
            &mut equal_widths,
            600,
            400,
            40,
            20,
            Vec3b::from([192, 255, 255]),
        );
        draw_rect(
            &mut equal_widths,
            700,
            403,
            40,
            14,
            Vec3b::from([192, 255, 255]),
        );
        assert!(matcher.detect(&equal_widths, None).unwrap().bar.is_none());

        let mut small_target = screen(1.0);
        draw_rect(
            &mut small_target,
            600,
            400,
            60,
            20,
            Vec3b::from([192, 255, 255]),
        );
        draw_rect(
            &mut small_target,
            580,
            403,
            8,
            14,
            Vec3b::from([192, 255, 255]),
        );
        assert!(matcher.detect(&small_target, None).unwrap().bar.is_none());

        let mut cursor_right = screen(1.0);
        draw_rect(
            &mut cursor_right,
            600,
            400,
            320,
            20,
            Vec3b::from([192, 255, 255]),
        );
        draw_rect(
            &mut cursor_right,
            800,
            403,
            8,
            14,
            Vec3b::from([192, 255, 255]),
        );
        assert!(matcher.detect(&cursor_right, None).unwrap().bar.is_none());
    }

    #[test]
    fn ocr_bite_matching_is_localized_case_insensitive_and_whitespace_free() {
        let matcher = FishingMatcher::new().unwrap();
        let image = screen(1.0);
        for text in ["鱼 上 钩 了", "上 鉤", "B I T E!", "MoRdU", " BBOCCATO "] {
            assert!(matcher.detect(&image, Some(text)).unwrap().bite, "{text:?}");
        }
        for text in ["等待鱼儿", "拉竿", "not yet", ""] {
            assert!(
                !matcher.detect(&image, Some(text)).unwrap().bite,
                "{text:?}"
            );
        }
    }

    #[test]
    fn empty_frame_has_no_fishing_observation() {
        let observation = FishingMatcher::new()
            .unwrap()
            .detect(&Mat::default(), Some("bite"))
            .unwrap();
        assert!(!observation.in_fishing);
        assert!(!observation.bite);
        assert!(observation.bar.is_none());
    }
}
