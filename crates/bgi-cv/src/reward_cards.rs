use bgi_task::RewardCardRect;
use opencv::core::{
    self, CV_8UC1, CV_32S, Mat, MatTraitConst, Point, Rect, Scalar, Size, ToInputArray, Vector,
};
use opencv::imgproc::{
    self, CHAIN_APPROX_SIMPLE, COLOR_BGR2HSV, INTER_LINEAR, MORPH_CLOSE, MORPH_OPEN, RETR_EXTERNAL,
    bounding_rect, find_contours, morphology_ex, resize,
};
use opencv::prelude::{MatExprTraitConst, MatTrait};

use crate::CvError;

pub const REWARD_BAND_ROI_1080P: (i32, i32, i32, i32) = (220, 444, 1480, 220);
pub const REWARD_CARD_WIDTH: i32 = 125;
pub const REWARD_CARD_HEIGHT: i32 = 153;

#[derive(Debug, Clone)]
pub struct RewardCountPreparation {
    pub normalized: Mat,
    pub component_count: usize,
    pub foreground_bounds: Option<RewardCardRect>,
}

#[derive(Debug, Clone)]
pub struct RewardCardPreparation {
    pub icon: Mat,
    pub count: Option<RewardCountPreparation>,
}

pub fn detect_reward_card_rects(
    band: &(impl MatTraitConst + ToInputArray),
) -> Result<Vec<RewardCardRect>, CvError> {
    if band.empty() || band.channels() != 3 {
        return Err(CvError::InvalidImage(
            "奖励卡片检测需要非空三通道 BGR 条带".into(),
        ));
    }
    let mask = create_reward_band_mask(band)?;
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let count = imgproc::connected_components_with_stats(
        &mask,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        CV_32S,
    )?;
    let min_width = f64::from(REWARD_CARD_WIDTH) * 0.55;
    let max_width = f64::from(REWARD_CARD_WIDTH) * 1.35;
    let min_height = f64::from(REWARD_CARD_HEIGHT) * 0.10;
    let max_height = f64::from(REWARD_CARD_HEIGHT) * 0.50;
    let min_center_y = f64::from(band.rows()) * 0.40;
    let mut strips = Vec::new();
    for index in 1..count {
        let x = *stats.at_2d::<i32>(index, 0)?;
        let y = *stats.at_2d::<i32>(index, 1)?;
        let width = *stats.at_2d::<i32>(index, 2)?;
        let height = *stats.at_2d::<i32>(index, 3)?;
        let area = *stats.at_2d::<i32>(index, 4)?;
        if (f64::from(width) < min_width || f64::from(width) > max_width)
            || (f64::from(height) < min_height || f64::from(height) > max_height)
            || area < 1200
            || f64::from(y) + f64::from(height) / 2.0 < min_center_y
        {
            continue;
        }
        strips.push(RewardCardRect {
            x,
            y,
            width,
            height,
        });
    }
    if strips.is_empty() {
        return Ok(Vec::new());
    }
    strips.sort_by_key(|strip| strip.y + strip.height);
    let median_bottom = strips[strips.len() / 2].y + strips[strips.len() / 2].height;
    strips.retain(|strip| {
        f64::from((strip.y + strip.height - median_bottom).abs())
            <= f64::from(REWARD_CARD_HEIGHT) * 0.5
    });
    strips.sort_by_key(|strip| strip.y + strip.height);
    let row_bottom = strips[strips.len() / 2].y + strips[strips.len() / 2].height;
    let mut cards = Vec::new();
    for strip in strips {
        let card_x = strip.x + strip.width / 2 - REWARD_CARD_WIDTH / 2;
        let card_y = row_bottom - REWARD_CARD_HEIGHT;
        if card_x < 0
            || card_y < 0
            || card_x + REWARD_CARD_WIDTH > band.cols()
            || card_y + REWARD_CARD_HEIGHT > band.rows()
        {
            continue;
        }
        cards.push(RewardCardRect {
            x: card_x,
            y: card_y,
            width: REWARD_CARD_WIDTH,
            height: REWARD_CARD_HEIGHT,
        });
    }
    cards.sort_by_key(|card| card.x);
    Ok(cards)
}

fn create_reward_band_mask(band: &(impl MatTraitConst + ToInputArray)) -> Result<Mat, CvError> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        band,
        &mut hsv,
        COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut mask = Mat::default();
    core::in_range(
        &hsv,
        &Scalar::new(0.0, 0.0, 190.0, 0.0),
        &Scalar::new(179.0, 20.0, 249.0, 0.0),
        &mut mask,
    )?;
    let mut opened = Mat::default();
    let mut closed = Mat::default();
    let kernel = Mat::ones(3, 3, CV_8UC1)?.to_mat()?;
    morphology_ex(
        &mask,
        &mut opened,
        MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    morphology_ex(
        &opened,
        &mut closed,
        MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(closed)
}

pub fn prepare_reward_card(
    card: &(impl MatTraitConst + ToInputArray),
) -> Result<RewardCardPreparation, CvError> {
    if card.empty() || card.channels() != 3 {
        return Err(CvError::InvalidImage(
            "奖励卡片预处理需要非空三通道 BGR 图像".into(),
        ));
    }
    let mut normalized_card = Mat::default();
    resize(
        card,
        &mut normalized_card,
        Size::new(REWARD_CARD_WIDTH, REWARD_CARD_HEIGHT),
        0.0,
        0.0,
        INTER_LINEAR,
    )?;
    let icon_view = Mat::roi(
        &normalized_card,
        Rect::new(0, 0, REWARD_CARD_WIDTH, REWARD_CARD_WIDTH),
    )?;
    let icon = icon_view.try_clone()?;
    let count = prepare_count(&normalized_card)?;
    Ok(RewardCardPreparation { icon, count })
}

fn prepare_count(card: &Mat) -> Result<Option<RewardCountPreparation>, CvError> {
    let count_view = Mat::roi(
        card,
        Rect::new(
            card.cols() * 5 / REWARD_CARD_WIDTH,
            card.rows() * 128 / REWARD_CARD_HEIGHT,
            card.cols() * 115 / REWARD_CARD_WIDTH,
            card.rows() * 22 / REWARD_CARD_HEIGHT,
        ),
    )?;
    let mut resized = Mat::default();
    resize(
        &count_view,
        &mut resized,
        Size::new(count_view.cols() * 3, count_view.rows() * 3),
        0.0,
        0.0,
        INTER_LINEAR,
    )?;
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &resized,
        &mut hsv,
        COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut mask = Mat::default();
    core::in_range(
        &hsv,
        &Scalar::new(0.0, 0.0, 0.0, 0.0),
        &Scalar::new(180.0, 140.0, 210.0, 0.0),
        &mut mask,
    )?;
    let mut contours = Vector::<Vector<Point>>::new();
    find_contours(
        &mask,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;
    let mut components = contours
        .iter()
        .map(|contour| bounding_rect(&contour))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|rect| {
            rect.width > 1
                && rect.height >= 12
                && rect.width * rect.height >= 20
                && f64::from(rect.y + rect.height) >= f64::from(mask.rows()) * 0.55
        })
        .collect::<Vec<_>>();
    if components.is_empty() {
        return Ok(None);
    }
    components.sort_by_key(|rect| rect.x);
    let left = components.iter().map(|rect| rect.x).min().unwrap();
    let top = components.iter().map(|rect| rect.y).min().unwrap();
    let right = components
        .iter()
        .map(|rect| rect.x + rect.width)
        .max()
        .unwrap();
    let bottom = components
        .iter()
        .map(|rect| rect.y + rect.height)
        .max()
        .unwrap();
    let bounds = Rect::new(left, top, right - left, bottom - top);
    let foreground = Mat::roi(&mask, bounds)?;
    let mut inverted = Mat::default();
    core::bitwise_not(&foreground, &mut inverted, &core::no_array())?;
    let scaled_width = ((f64::from(inverted.cols()) * 48.0 / f64::from(inverted.rows()))
        .round_ties_even() as i32)
        .max(1);
    let mut scaled = Mat::default();
    resize(
        &inverted,
        &mut scaled,
        Size::new(scaled_width, 48),
        0.0,
        0.0,
        opencv::imgproc::INTER_NEAREST,
    )?;
    let mut normalized = Mat::zeros(60, scaled_width + 48, CV_8UC1)?.to_mat()?;
    normalized.set_to(&Scalar::all(255.0), &core::no_array())?;
    let mut target = Mat::roi_mut(&mut normalized, Rect::new(24, 6, scaled_width, 48))?;
    scaled.copy_to(&mut target)?;
    Ok(Some(RewardCountPreparation {
        normalized,
        component_count: components.len(),
        foreground_bounds: Some(RewardCardRect {
            x: left,
            y: top,
            width: bounds.width,
            height: bounds.height,
        }),
    }))
}

#[cfg(test)]
mod tests {
    use opencv::core::CV_8UC3;
    use opencv::imgproc::rectangle;

    use super::*;

    fn band() -> Mat {
        Mat::zeros(220, 1480, CV_8UC3).unwrap().to_mat().unwrap()
    }

    fn strip(band: &mut Mat, x: i32, y: i32) {
        rectangle(band, Rect::new(x, y, 100, 20), Scalar::all(246.0), -1, 8, 0).unwrap();
    }

    #[test]
    fn card_detection_sorts_cards_and_discards_out_of_bounds_strips() {
        let mut image = band();
        strip(&mut image, 500, 180);
        strip(&mut image, 100, 180);
        strip(&mut image, 300, 180);
        let cards = detect_reward_card_rects(&image).unwrap();
        assert_eq!(
            cards,
            vec![
                RewardCardRect {
                    x: 88,
                    y: 47,
                    width: 125,
                    height: 153
                },
                RewardCardRect {
                    x: 288,
                    y: 47,
                    width: 125,
                    height: 153
                },
                RewardCardRect {
                    x: 488,
                    y: 47,
                    width: 125,
                    height: 153
                },
            ]
        );
        let mut clipped = band();
        rectangle(
            &mut clipped,
            Rect::new(0, 180, 69, 20),
            Scalar::all(246.0),
            -1,
            8,
            0,
        )
        .unwrap();
        assert!(detect_reward_card_rects(&clipped).unwrap().is_empty());
    }
    #[test]
    fn card_detection_rejects_empty_noise_and_off_row_components() {
        assert!(detect_reward_card_rects(&band()).unwrap().is_empty());
        let mut image = band();
        strip(&mut image, 100, 180);
        strip(&mut image, 300, 180);
        strip(&mut image, 500, 180);
        strip(&mut image, 700, 80);
        let cards = detect_reward_card_rects(&image).unwrap();
        assert_eq!(cards.len(), 3);
        assert!(cards.iter().all(|card| card.y == 47));
        assert!(detect_reward_card_rects(&Mat::default()).is_err());
    }

    #[test]
    fn count_preparation_extracts_only_filtered_bottom_foreground() {
        let mut card = Mat::zeros(REWARD_CARD_HEIGHT, REWARD_CARD_WIDTH, CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        card.set_to(&Scalar::all(255.0), &core::no_array()).unwrap();
        rectangle(
            &mut card,
            Rect::new(25, 137, 8, 10),
            Scalar::all(20.0),
            -1,
            8,
            0,
        )
        .unwrap();
        rectangle(
            &mut card,
            Rect::new(40, 136, 8, 11),
            Scalar::all(20.0),
            -1,
            8,
            0,
        )
        .unwrap();
        rectangle(
            &mut card,
            Rect::new(55, 130, 8, 5),
            Scalar::all(20.0),
            -1,
            8,
            0,
        )
        .unwrap();
        let prepared = prepare_reward_card(&card).unwrap();
        assert_eq!(prepared.icon.rows(), 125);
        assert_eq!(prepared.icon.cols(), 125);
        let count = prepared.count.unwrap();
        assert_eq!(count.component_count, 2);
        assert_eq!(count.normalized.rows(), 60);
        assert!(count.normalized.cols() > 48);
        let bounds = count.foreground_bounds.unwrap();
        assert!(bounds.width > 0 && bounds.height > 0);
    }
}
