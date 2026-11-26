use std::path::Path;

use opencv::core::{self, Mat, MatTrait, MatTraitConst, Point, Rect, Scalar, Size, Vector};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imdecode};
use opencv::imgproc::{
    self, COLOR_BGR2GRAY, COLOR_BGR2HSV_FULL, INTER_AREA, INTER_CUBIC, INTER_LINEAR, RETR_EXTERNAL,
    TM_CCOEFF_NORMED, find_contours,
};
use opencv::prelude::MatExprTraitConst;

use crate::CvError;

const SALVAGE_BUTTON_PNG: &[u8] = include_bytes!("../assets/ui/btn_artifact_salvage.png");
const SALVAGE_CONFIRM_PNG: &[u8] = include_bytes!("../assets/ui/btn_artifact_salvage_confirm.png");
const BLACK_CONFIRM_PNG: &[u8] = include_bytes!("../assets/ui/btn_black_confirm.png");

const REFERENCE_WIDTH: i32 = 1920;
const REFERENCE_HEIGHT: i32 = 1080;
const GRID_COLUMNS: i32 = 9;
const GRID_ROWS: i32 = 4;
const MATCH_THRESHOLD: f64 = 0.86;

/// C# `GridParams.ArtifactSalvage`: 1080p ROI `(48, 106, 1267, 768)`, 9 columns.
pub const SALVAGE_GRID_ROI_1080P: (i32, i32, i32, i32) = (48, 106, 1267, 768);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl PixelRect {
    pub fn center(self) -> (i32, i32) {
        (
            self.x.saturating_add(self.width / 2),
            self.y.saturating_add(self.height / 2),
        )
    }

    fn to_opencv(self) -> Rect {
        Rect::new(self.x, self.y, self.width, self.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactSalvageCardStatus {
    Unlocked,
    Locked,
    Selected,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArtifactSalvageCell {
    pub index: u8,
    pub row: u8,
    pub column: u8,
    pub rect: PixelRect,
    pub center: (i32, i32),
    pub present: bool,
    pub presence_confidence: f64,
    pub status: ArtifactSalvageCardStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArtifactSalvageGrid {
    pub roi: PixelRect,
    pub scroll_center: (i32, i32),
    pub fingerprint: u64,
    pub cells: [ArtifactSalvageCell; GRID_COLUMNS as usize * GRID_ROWS as usize],
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArtifactSalvageObservation {
    pub width: u32,
    pub height: u32,
    pub inventory_salvage_button: Option<(i32, i32, f64)>,
    pub salvage_confirm_button: Option<(i32, i32, f64)>,
    pub black_confirm_button: Option<(i32, i32, f64)>,
    pub grid: Option<ArtifactSalvageGrid>,
}

impl ArtifactSalvageObservation {
    pub fn salvage_page_visible(self) -> bool {
        self.grid.is_some()
            && self.salvage_confirm_button.is_some()
            && self.black_confirm_button.is_none()
    }
}

#[derive(Debug, Clone)]
pub struct ArtifactSalvageMatcher {
    salvage_button: Mat,
    salvage_confirm: Mat,
    black_confirm: Mat,
}

impl ArtifactSalvageMatcher {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            salvage_button: decode_png(SALVAGE_BUTTON_PNG)?,
            salvage_confirm: decode_png(SALVAGE_CONFIRM_PNG)?,
            black_confirm: decode_png(BLACK_CONFIRM_PNG)?,
        })
    }

    pub fn from_assets(
        salvage_button: impl AsRef<Path>,
        salvage_confirm: impl AsRef<Path>,
        black_confirm: impl AsRef<Path>,
    ) -> Result<Self, CvError> {
        Ok(Self {
            salvage_button: decode_file(salvage_button.as_ref())?,
            salvage_confirm: decode_file(salvage_confirm.as_ref())?,
            black_confirm: decode_file(black_confirm.as_ref())?,
        })
    }

    pub fn detect(&self, full_bgr: &Mat) -> Result<ArtifactSalvageObservation, CvError> {
        if full_bgr.empty() {
            return Ok(ArtifactSalvageObservation::default());
        }
        let width = full_bgr.cols();
        let height = full_bgr.rows();
        let mut observation = ArtifactSalvageObservation {
            width: width as u32,
            height: height as u32,
            ..Default::default()
        };
        if width < 320 || height < 240 {
            return Ok(observation);
        }
        let scale = (f64::from(width) / f64::from(REFERENCE_WIDTH))
            .min(f64::from(height) / f64::from(REFERENCE_HEIGHT));
        let mut gray = Mat::default();
        imgproc::cvt_color(
            full_bgr,
            &mut gray,
            COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Recognition.json: @artifactSalvageBottom = cutBottom(0.1).
        let salvage_roi = Rect::new(0, height * 9 / 10, width, height - height * 9 / 10);
        // Recognition.json: @artifactSalvageConfirm = cutRightBottom(0.3, 0.1).
        let confirm_roi = Rect::new(
            width * 7 / 10,
            height * 9 / 10,
            width - width * 7 / 10,
            height - height * 9 / 10,
        );
        // Bv.ClickBlackConfirmButton passes no searchRect, so C# uses @full.
        observation.inventory_salvage_button =
            detect_in_roi(&gray, salvage_roi, &self.salvage_button, scale)?;
        observation.salvage_confirm_button =
            detect_in_roi(&gray, confirm_roi, &self.salvage_confirm, scale)?;
        observation.black_confirm_button = detect_in_roi(
            &gray,
            Rect::new(0, 0, width, height),
            &self.black_confirm,
            scale,
        )?;

        if aspect_ratio_is_16_9(width, height) {
            observation.grid = Some(detect_grid(full_bgr, scale)?);
        }
        Ok(observation)
    }
}

fn aspect_ratio_is_16_9(width: i32, height: i32) -> bool {
    // Cross multiplication avoids float drift at reduced resolutions.
    i64::from(width) * 9 == i64::from(height) * 16
}

fn scaled_rect(x: f64, y: f64, width: f64, height: f64, scale: f64) -> PixelRect {
    PixelRect {
        x: (x * scale).round() as i32,
        y: (y * scale).round() as i32,
        width: (width * scale).round() as i32,
        height: (height * scale).round() as i32,
    }
}

fn detect_grid(full_bgr: &Mat, scale: f64) -> Result<ArtifactSalvageGrid, CvError> {
    let (x, y, width, height) = SALVAGE_GRID_ROI_1080P;
    let roi = scaled_rect(
        f64::from(x),
        f64::from(y),
        f64::from(width),
        f64::from(height),
        scale,
    );
    let grid = Mat::roi(full_bgr, roi.to_opencv())?;
    let fingerprint = grid_fingerprint(&grid)?;
    let scroll_center = roi.center();
    let mut cells = [empty_cell(); GRID_COLUMNS as usize * GRID_ROWS as usize];
    for row in 0..GRID_ROWS {
        for column in 0..GRID_COLUMNS {
            let index = row * GRID_COLUMNS + column;
            // Use floating-point boundaries so half-resolution rounding does not accumulate drift.
            let left = (f64::from(roi.x)
                + f64::from(roi.width) * f64::from(column) / f64::from(GRID_COLUMNS))
            .round() as i32;
            let right = (f64::from(roi.x)
                + f64::from(roi.width) * f64::from(column + 1) / f64::from(GRID_COLUMNS))
            .round() as i32;
            let top = (f64::from(roi.y)
                + f64::from(roi.height) * f64::from(row) / f64::from(GRID_ROWS))
            .round() as i32;
            let bottom = (f64::from(roi.y)
                + f64::from(roi.height) * f64::from(row + 1) / f64::from(GRID_ROWS))
            .round() as i32;
            let rect = PixelRect {
                x: left,
                y: top,
                width: right - left,
                height: bottom - top,
            };
            let cell = Mat::roi(full_bgr, rect.to_opencv())?;
            let (present, presence_confidence) = card_presence(&cell)?;
            let status = if present {
                detect_card_status(&cell)?
            } else {
                ArtifactSalvageCardStatus::Unlocked
            };
            cells[index as usize] = ArtifactSalvageCell {
                index: index as u8,
                row: row as u8,
                column: column as u8,
                rect,
                center: rect.center(),
                present,
                presence_confidence,
                status,
            };
        }
    }
    Ok(ArtifactSalvageGrid {
        roi,
        scroll_center,
        fingerprint,
        cells,
    })
}

const fn empty_cell() -> ArtifactSalvageCell {
    ArtifactSalvageCell {
        index: 0,
        row: 0,
        column: 0,
        rect: PixelRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        },
        center: (0, 0),
        present: false,
        presence_confidence: 0.0,
        status: ArtifactSalvageCardStatus::Unlocked,
    }
}

fn card_presence(cell: &impl MatTraitConst) -> Result<(bool, f64), CvError> {
    let resized = resize_card(cell)?;
    // Match GridScreenExtensions.GetGridBottom + IsCorrectBottomColor exactly:
    // resize to 125x153, inspect rows 126..153, ignore the item-name area.
    let bottom = Mat::roi(&resized, Rect::new(0, 126, 125, 27))?;
    let mut mask = Mat::zeros(27, 125, core::CV_8UC1)?.to_mat()?;
    mask.set_to(&Scalar::new(255.0, 0.0, 0.0, 0.0), &core::no_array())?;
    Mat::roi_mut(&mut mask, Rect::new(6, 0, 113, 21))?
        .set_to(&Scalar::new(0.0, 0.0, 0.0, 0.0), &core::no_array())?;
    let mean = core::mean(&bottom, &mask)?;
    let difference: f64 =
        (mean[0] - 220.0).abs() + (mean[1] - 229.0).abs() + (mean[2] - 233.0).abs();
    let confidence = (1.0 - difference / 180.0).clamp(0.0, 1.0);
    Ok((difference <= 90.0, confidence))
}

/// Matches C# `GetGridIcon`: resize the whole card to 125x153, then keep its upper 125x125 icon.
pub fn resize_card_for_icon(full_bgr: &Mat, rect: PixelRect) -> Result<Mat, CvError> {
    let resized = resize_card(&Mat::roi(full_bgr, rect.to_opencv())?)?;
    Ok(Mat::roi(&resized, Rect::new(0, 0, 125, 125))?.try_clone()?)
}

fn resize_card(cell: &impl MatTraitConst) -> Result<Mat, CvError> {
    let mut cell_copy = Mat::default();
    cell.copy_to(&mut cell_copy)?;
    let mut resized = Mat::default();
    imgproc::resize(
        &cell_copy,
        &mut resized,
        Size::new(125, 153),
        0.0,
        0.0,
        INTER_LINEAR,
    )?;
    Ok(resized)
}

fn detect_card_status(cell: &impl MatTraitConst) -> Result<ArtifactSalvageCardStatus, CvError> {
    // C# AutoArtifactSalvageTask.GetArtifactStatus inspects the upper 19%.
    let upper_height = (cell.rows() as f64 * 0.19).round() as i32;
    let upper = Mat::roi(cell, Rect::new(0, 0, cell.cols(), upper_height.max(1)))?;
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &upper,
        &mut hsv,
        COLOR_BGR2HSV_FULL,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // CommonHSV (9, .54, 1) -> HSV-full hue 6.35, saturation 137.7, value 255.
    if has_colored_contour(
        &hsv,
        Scalar::new(3.35, 112.7, 230.0, 0.0),
        Scalar::new(9.35, 162.7, 255.0, 0.0),
        true,
    )? {
        return Ok(ArtifactSalvageCardStatus::Locked);
    }
    // CommonHSV (80, .76, 1) -> HSV-full hue 56.47, saturation 193.8, value 255.
    if has_colored_contour(
        &hsv,
        Scalar::new(53.47, 183.8, 250.0, 0.0),
        Scalar::new(59.47, 203.8, 255.0, 0.0),
        false,
    )? {
        return Ok(ArtifactSalvageCardStatus::Selected);
    }
    Ok(ArtifactSalvageCardStatus::Unlocked)
}

fn has_colored_contour(
    hsv: &Mat,
    lower: Scalar,
    upper: Scalar,
    left_side_only: bool,
) -> Result<bool, CvError> {
    let mut mask = Mat::default();
    core::in_range(&hsv, &lower, &upper, &mut mask)?;
    let mut contours = Vector::<Vector<Point>>::new();
    find_contours(
        &mask,
        &mut contours,
        RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::default(),
    )?;
    // C# concatenates all qualifying contour points before measuring one bounding box.
    let mut points = Vec::new();
    for contour in contours {
        let contour_points: Vec<Point> = contour.iter().collect();
        if left_side_only
            && contour_points
                .iter()
                .any(|point| point.x >= (hsv.cols() as f64 * 0.2) as i32)
        {
            continue;
        }
        points.extend(contour_points);
    }
    if points.is_empty() {
        return Ok(false);
    }
    let bounds = imgproc::bounding_rect(&Vector::<Point>::from_iter(points))?;
    let (width_ratio, height_ratio) = if left_side_only {
        (0.07, 0.3)
    } else {
        (0.2, 0.8)
    };
    Ok(bounds.width > (hsv.cols() as f64 * width_ratio) as i32
        && bounds.height > (hsv.rows() as f64 * height_ratio) as i32)
}

fn grid_fingerprint(grid: &impl MatTraitConst) -> Result<u64, CvError> {
    // Stable FNV-1a over a sparse grid. It is only a pagination progress signal, not an item ID.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for y in (0..grid.rows()).step_by(8) {
        for x in (0..grid.cols()).step_by(8) {
            let pixel = grid.at_2d::<core::Vec3b>(y, x)?;
            for byte in pixel.iter() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    Ok(hash)
}

fn detect_in_roi(
    gray: &Mat,
    roi: Rect,
    template: &Mat,
    scale: f64,
) -> Result<Option<(i32, i32, f64)>, CvError> {
    let image = Mat::roi(gray, roi)?;
    let scaled_storage: Mat;
    let template_ref: &Mat = if (scale - 1.0).abs() > 0.02 {
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
    if template_ref.cols() > roi.width || template_ref.rows() > roi.height {
        return Ok(None);
    }
    let mut result = Mat::default();
    imgproc::match_template(
        &image,
        template_ref,
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
    if score.is_finite() && score >= MATCH_THRESHOLD {
        Ok(Some((
            roi.x + location.x + template_ref.cols() / 2,
            roi.y + location.y + template_ref.rows() / 2,
            score,
        )))
    } else {
        Ok(None)
    }
}

fn decode_png(bytes: &[u8]) -> Result<Mat, CvError> {
    let decoded = imdecode(&Vector::<u8>::from_slice(bytes), IMREAD_GRAYSCALE)?;
    if decoded.empty() {
        return Err(CvError::InvalidImage("圣遗物分解模板解码失败".into()));
    }
    Ok(decoded)
}

fn decode_file(path: &Path) -> Result<Mat, CvError> {
    decode_png(&std::fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_8UC3, Vec3b};
    use opencv::imgproc::COLOR_GRAY2BGR;

    fn marker_color(hsv_hue: f64, common_s: f64, common_v: f64) -> Vec3b {
        let hue = hsv_hue / 255.0 * 360.0;
        let value = common_v * 255.0;
        let chroma = value * common_s;
        let sector = hue / 60.0;
        let (red, green, blue) = match sector.floor() as u8 {
            0 => (chroma, chroma * sector, 0.0),
            1 => (chroma * (2.0 - sector), chroma, 0.0),
            2 => (0.0, chroma, chroma * (sector - 2.0)),
            3 => (0.0, chroma * (4.0 - sector), chroma),
            4 => (chroma * (sector - 4.0), 0.0, chroma),
            _ => (chroma, 0.0, chroma * (6.0 - sector)),
        };
        let m = value - chroma;
        Vec3b::from([
            (blue + m).round() as u8,
            (green + m).round() as u8,
            (red + m).round() as u8,
        ])
    }

    fn blank_screen(scale: f64) -> Mat {
        Mat::zeros(
            (f64::from(REFERENCE_HEIGHT) * scale).round() as i32,
            (f64::from(REFERENCE_WIDTH) * scale).round() as i32,
            CV_8UC3,
        )
        .unwrap()
        .to_mat()
        .unwrap()
    }

    fn paste_template(screen: &mut Mat, bytes: &[u8], scale: f64, base: (i32, i32)) {
        let source = decode_png(bytes).unwrap();
        let mut resized = Mat::default();
        imgproc::resize(
            &source,
            &mut resized,
            Size::new(
                (f64::from(source.cols()) * scale).round() as i32,
                (f64::from(source.rows()) * scale).round() as i32,
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
            COLOR_GRAY2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        let mut target = Mat::roi_mut(
            screen,
            Rect::new(
                (f64::from(base.0) * scale).round() as i32,
                (f64::from(base.1) * scale).round() as i32,
                bgr.cols(),
                bgr.rows(),
            ),
        )
        .unwrap();
        bgr.copy_to(&mut target).unwrap();
    }

    fn draw_cards(screen: &mut Mat, scale: f64) {
        let roi = scaled_rect(48.0, 106.0, 1267.0, 768.0, scale);
        for row in 0..GRID_ROWS {
            for column in 0..GRID_COLUMNS {
                let left = (f64::from(roi.x)
                    + f64::from(roi.width) * f64::from(column) / f64::from(GRID_COLUMNS))
                .round() as i32;
                let right = (f64::from(roi.x)
                    + f64::from(roi.width) * f64::from(column + 1) / f64::from(GRID_COLUMNS))
                .round() as i32;
                let top = (f64::from(roi.y)
                    + f64::from(roi.height) * f64::from(row) / f64::from(GRID_ROWS))
                .round() as i32;
                let bottom = (f64::from(roi.y)
                    + f64::from(roi.height) * f64::from(row + 1) / f64::from(GRID_ROWS))
                .round() as i32;
                let rect = Rect::new(left, top, right - left, bottom - top);
                let mut target = Mat::roi_mut(screen, rect).unwrap();
                target
                    .set_to(&Scalar::new(60.0, 55.0, 50.0, 0.0), &core::no_array())
                    .unwrap();
                let band_top = target.rows() * 88 / 100;
                let band_height = target.rows() - band_top;
                let band_width = target.cols();
                let mut band =
                    Mat::roi_mut(&mut target, Rect::new(0, band_top, band_width, band_height))
                        .unwrap();
                band.set_to(&Scalar::new(220.0, 229.0, 233.0, 0.0), &core::no_array())
                    .unwrap();
            }
        }
    }

    #[test]
    fn grid_icon_matches_itemv2_125_square_input() {
        let mut parent = Mat::zeros(160, 130, CV_8UC3).unwrap().to_mat().unwrap();
        let mut card = Mat::roi_mut(&mut parent, Rect::new(2, 3, 125, 153)).unwrap();
        card.set_to(&Scalar::new(10.0, 20.0, 30.0, 0.0), &core::no_array())
            .unwrap();
        Mat::roi_mut(&mut card, Rect::new(0, 125, 125, 28))
            .unwrap()
            .set_to(&Scalar::new(220.0, 229.0, 233.0, 0.0), &core::no_array())
            .unwrap();
        let icon = resize_card_for_icon(
            &parent,
            PixelRect {
                x: 2,
                y: 3,
                width: 125,
                height: 153,
            },
        )
        .unwrap();
        assert_eq!((icon.rows(), icon.cols()), (125, 125));
        assert!(icon.is_continuous());
        assert_eq!(
            *icon.at_2d::<Vec3b>(124, 124).unwrap(),
            Vec3b::from([10, 20, 30])
        );
    }

    #[test]
    fn salvage_templates_and_grid_match_reference_and_half_resolution() {
        let matcher = ArtifactSalvageMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut screen = blank_screen(scale);
            draw_cards(&mut screen, scale);
            paste_template(&mut screen, SALVAGE_CONFIRM_PNG, scale, (1780, 1000));
            let observation = matcher.detect(&screen).unwrap();
            assert!(
                observation.salvage_confirm_button.is_some(),
                "scale={scale}: {observation:?}"
            );
            assert!(observation.salvage_page_visible());
            let grid = observation.grid.unwrap();
            assert_eq!(grid.cells.len(), 36);
            assert!(grid.cells.iter().all(|cell| cell.present));
            assert_eq!(grid.roi.width, (1267.0 * scale).round() as i32);
            assert_eq!(grid.roi.height, (768.0 * scale).round() as i32);
            let roi = scaled_rect(48.0, 106.0, 1267.0, 768.0, scale);
            let first_right = (roi.x as f64 + roi.width as f64 / 9.0).round() as i32;
            let first_bottom = (roi.y as f64 + roi.height as f64 / 4.0).round() as i32;
            let last_left = (roi.x as f64 + roi.width as f64 * 8.0 / 9.0).round() as i32;
            let last_top = (roi.y as f64 + roi.height as f64 * 3.0 / 4.0).round() as i32;
            assert_eq!(
                grid.cells[0].center,
                ((roi.x + first_right) / 2, (roi.y + first_bottom) / 2)
            );
            assert_eq!(
                grid.cells[35].center,
                (
                    (last_left + roi.x + roi.width) / 2,
                    (last_top + roi.y + roi.height) / 2
                )
            );
        }
    }

    #[test]
    fn csharp_rois_reject_misplaced_buttons_and_full_black_confirmation_wins() {
        let matcher = ArtifactSalvageMatcher::new().unwrap();
        let mut misplaced = blank_screen(1.0);
        paste_template(&mut misplaced, SALVAGE_CONFIRM_PNG, 1.0, (100, 100));
        paste_template(&mut misplaced, SALVAGE_BUTTON_PNG, 1.0, (100, 500));
        let observation = matcher.detect(&misplaced).unwrap();
        assert!(observation.salvage_confirm_button.is_none());
        assert!(observation.inventory_salvage_button.is_none());
        assert!(!observation.salvage_page_visible());

        let mut popup = blank_screen(1.0);
        draw_cards(&mut popup, 1.0);
        paste_template(&mut popup, SALVAGE_CONFIRM_PNG, 1.0, (1780, 1000));
        paste_template(&mut popup, BLACK_CONFIRM_PNG, 1.0, (100, 100));
        let observation = matcher.detect(&popup).unwrap();
        assert!(observation.black_confirm_button.is_some());
        assert!(!observation.salvage_page_visible());
    }

    fn fill_rect(screen: &mut Mat, rect: Rect, color: Vec3b) {
        let mut target = Mat::roi_mut(screen, rect).unwrap();
        target
            .set_to(
                &Scalar::new(
                    f64::from(color[0]),
                    f64::from(color[1]),
                    f64::from(color[2]),
                    0.0,
                ),
                &core::no_array(),
            )
            .unwrap();
    }

    #[test]
    fn locked_and_selected_statuses_follow_csharp_upper_card_colors() {
        let matcher = ArtifactSalvageMatcher::new().unwrap();
        for scale in [1.0_f64, 0.5] {
            let mut screen = blank_screen(scale);
            draw_cards(&mut screen, scale);
            let first = scaled_rect(48.0, 106.0, 1267.0 / 9.0, 192.0, scale);
            let upper_height = first.height * 19 / 100;
            fill_rect(
                &mut screen,
                Rect::new(
                    first.x + first.width / 20,
                    first.y + 1,
                    first.width / 8,
                    upper_height * 3 / 4,
                ),
                marker_color(6.35, 0.54, 1.0),
            );
            let second_x = first.x + first.width;
            fill_rect(
                &mut screen,
                Rect::new(
                    second_x + first.width * 3 / 5,
                    first.y + 1,
                    first.width / 4,
                    upper_height * 9 / 10,
                ),
                marker_color(56.47, 0.76, 1.0),
            );
            paste_template(&mut screen, SALVAGE_CONFIRM_PNG, scale, (1780, 1000));
            let observation = matcher.detect(&screen).unwrap();
            let grid = observation.grid.unwrap();
            assert!(grid.cells[0].present, "scale={scale}");
            assert!(grid.cells[1].present, "scale={scale}");
            assert_eq!(
                grid.cells[0].status,
                ArtifactSalvageCardStatus::Locked,
                "scale={scale}"
            );
            assert_eq!(
                grid.cells[1].status,
                ArtifactSalvageCardStatus::Selected,
                "scale={scale}"
            );
        }
    }

    #[test]
    fn non_16_9_frames_never_expose_a_grid() {
        let matcher = ArtifactSalvageMatcher::new().unwrap();
        let screen = Mat::zeros(800, 1600, CV_8UC3).unwrap().to_mat().unwrap();
        let observation = matcher.detect(&screen).unwrap();
        assert!(observation.grid.is_none());
        assert!(!observation.salvage_page_visible());
    }
}
