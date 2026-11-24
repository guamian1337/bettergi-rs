use std::fs;
use std::path::Path;

use opencv::calib3d::{RANSAC, find_homography};
use opencv::core::{
    self, CV_32FC1, DMatch, KeyPoint, Mat, MatTraitConst, Point2f, Size, Vector,
};
use opencv::features2d::{Feature2DTrait, FlannBasedMatcher, SIFT};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imread};
use opencv::imgproc::{self, INTER_AREA};
use opencv::prelude::{DescriptorMatcherTraitConst, KeyPointTraitConst, MatTraitConstManual};

use crate::CvError;

/// Teyvat map layout constants matching `BetterGenshinImpact/GameTask/Common/Map/Maps/TeyvatMap.cs`.
pub const TEYVAT_MAP_ROWS: i32 = 19;
pub const TEYVAT_MAP_COLS: i32 = 22;
pub const TEYVAT_MAP_UP_ROWS: i32 = 11;
pub const TEYVAT_MAP_LEFT_COLS: i32 = 15;
pub const TEYVAT_BLOCK_WIDTH_2048: i32 = 2048;
pub const BIGMAP_256_SCALE_TO_2048: f32 = 8.0;

#[derive(Debug, Clone, Copy)]
pub struct TeyvatCoordTransform {
    pub origin_x: f32,
    pub origin_y: f32,
    pub block_scale_to_1024: f32,
}

impl Default for TeyvatCoordTransform {
    fn default() -> Self {
        Self {
            origin_x: ((TEYVAT_MAP_LEFT_COLS + 1) * TEYVAT_BLOCK_WIDTH_2048) as f32,
            origin_y: ((TEYVAT_MAP_UP_ROWS + 1) * TEYVAT_BLOCK_WIDTH_2048) as f32,
            block_scale_to_1024: (TEYVAT_BLOCK_WIDTH_2048 as f32) / 1024.0,
        }
    }
}

impl TeyvatCoordTransform {
    pub fn image_to_game(&self, img_x: f32, img_y: f32) -> (f32, f32) {
        (
            (self.origin_x - img_x) / self.block_scale_to_1024,
            (self.origin_y - img_y) / self.block_scale_to_1024,
        )
    }

    pub fn game_to_image(&self, game_x: f32, game_y: f32) -> (f32, f32) {
        (
            self.origin_x - game_x * self.block_scale_to_1024,
            self.origin_y - game_y * self.block_scale_to_1024,
        )
    }
}

/// Loads BetterGI's raw binary `*_SIFT.kp.bin` (28-byte `cv::KeyPoint` records) safely.
pub fn load_keypoint_points_bin(kp_path: impl AsRef<Path>) -> Result<Vec<Point2f>, CvError> {
    const KP_RECORD_BYTES: usize = 28;
    let bytes = fs::read(kp_path)?;
    if bytes.is_empty() || bytes.len() % KP_RECORD_BYTES != 0 {
        return Err(CvError::InvalidImage(format!(
            "KeyPoint 二进制大小 {} 不是 {KP_RECORD_BYTES} 的整数倍",
            bytes.len(),
        )));
    }
    let mut pts = Vec::with_capacity(bytes.len() / KP_RECORD_BYTES);
    for chunk in bytes.chunks_exact(KP_RECORD_BYTES) {
        let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        pts.push(Point2f::new(x, y));
    }
    Ok(pts)
}

pub fn load_descriptors_png(mat_path: impl AsRef<Path>) -> Result<Mat, CvError> {
    let path_str = mat_path.as_ref().to_string_lossy();
    let img = imread(&*path_str, IMREAD_GRAYSCALE)?;
    if img.empty() {
        return Err(CvError::InvalidImage(format!("无法加载 SIFT 描述符图像: {path_str}")));
    }
    let mut f32_mat = Mat::default();
    img.convert_to(&mut f32_mat, CV_32FC1, 1.0, 0.0)?;
    Ok(f32_mat)
}

pub struct BigMapSiftMatcher {
    sift: opencv::core::Ptr<SIFT>,
    matcher: opencv::core::Ptr<FlannBasedMatcher>,
    train_pts: Vec<Point2f>,
    train_desc: Mat,
    coord: TeyvatCoordTransform,
}

impl BigMapSiftMatcher {
    pub fn load_teyvat_256(map_dir: impl AsRef<Path>) -> Result<Self, CvError> {
        let dir = map_dir.as_ref();
        let kp_path = dir.join("Teyvat_0_256_SIFT.kp.bin");
        let desc_path = dir.join("Teyvat_0_256_SIFT.mat.png");
        let train_pts = load_keypoint_points_bin(kp_path)?;
        let train_desc = load_descriptors_png(desc_path)?;

        let sift = SIFT::create(0, 3, 0.04, 10.0, 1.6, false)?;
        let matcher = FlannBasedMatcher::create()?;

        Ok(Self {
            sift,
            matcher,
            train_pts,
            train_desc,
            coord: TeyvatCoordTransform::default(),
        })
    }

    pub fn keypoint_count(&self) -> usize {
        self.train_pts.len()
    }

    /// Big-map SIFT match with projected viewport bounds in Genshin world coordinates.
    pub fn locate_big_map_rect(&mut self, frame_bgr_or_gray: &Mat) -> Result<BigMapRectMatch, CvError> {
        let frame_width = frame_bgr_or_gray.cols().max(1) as u32;
        let frame_height = frame_bgr_or_gray.rows().max(1) as u32;

        let mut gray = Mat::default();
        if frame_bgr_or_gray.channels() == 3 {
            imgproc::cvt_color(
                frame_bgr_or_gray,
                &mut gray,
                imgproc::COLOR_BGR2GRAY,
                0,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
        } else {
            gray = frame_bgr_or_gray.try_clone()?;
        }

        // Mirrors TeyvatMap.GetBigMapPosition scaling (1/4 for 256-level SIFT map)
        let mut resized = Mat::default();
        imgproc::resize(
            &gray,
            &mut resized,
            Size::new(0, 0),
            0.25,
            0.25,
            INTER_AREA,
        )?;

        let mut query_kps = Vector::<KeyPoint>::new();
        let mut query_desc = Mat::default();
        self.sift.detect_and_compute(&resized, &core::no_array(), &mut query_kps, &mut query_desc, false)?;
        if query_kps.is_empty() || query_desc.empty() {
            return Err(CvError::MatchFailed("大地图未提取到 SIFT 特征点".into()));
        }

        let mut knn_matches = Vector::<Vector<DMatch>>::new();
        self.matcher.knn_train_match_def(
            &query_desc,
            &self.train_desc,
            &mut knn_matches,
            2,
        )?;

        let mut src_pts = Vector::<Point2f>::new();
        let mut dst_pts = Vector::<Point2f>::new();
        for pair in knn_matches {
            if pair.len() == 2 {
                let m0 = pair.get(0)?;
                let m1 = pair.get(1)?;
                if m0.distance < 0.75 * m1.distance {
                    src_pts.push(query_kps.get(m0.query_idx as usize)?.pt());
                    dst_pts.push(self.train_pts[m0.train_idx as usize]);
                }
            }
        }

        if src_pts.len() < 7 {
            return Err(CvError::MatchFailed(format!(
                "有效 SIFT 匹配点不足 ({} < 7)",
                src_pts.len()
            )));
        }

        let mut mask = Mat::default();
        let h_mat = find_homography(&src_pts, &dst_pts, &mut mask, RANSAC, 3.0)?;
        if h_mat.empty() {
            return Err(CvError::MatchFailed("RANSAC 单应矩阵计算失败".into()));
        }

        let mut inliers = 0usize;
        if let Ok(mask_bytes) = mask.data_bytes() {
            inliers = mask_bytes.iter().filter(|&&b| b != 0).count();
        }

        let mut pts_in = Vector::<Point2f>::new();
        pts_in.push(Point2f::new(0.0, 0.0));
        pts_in.push(Point2f::new(resized.cols() as f32 / 2.0, resized.rows() as f32 / 2.0));
        pts_in.push(Point2f::new(resized.cols() as f32, resized.rows() as f32));
        let mut pts_out = Vector::<Point2f>::new();
        core::perspective_transform(&pts_in, &mut pts_out, &h_mat)?;

        let to_game = |p256: Point2f| -> (f32, f32) {
            self.coord.image_to_game(
                p256.x * BIGMAP_256_SCALE_TO_2048,
                p256.y * BIGMAP_256_SCALE_TO_2048,
            )
        };
        let (game_tl_x, game_tl_y) = to_game(pts_out.get(0)?);
        let (game_center_x, game_center_y) = to_game(pts_out.get(1)?);
        let (game_br_x, game_br_y) = to_game(pts_out.get(2)?);

        Ok(BigMapRectMatch {
            game_center_x,
            game_center_y,
            game_tl_x,
            game_tl_y,
            game_br_x,
            game_br_y,
            frame_width,
            frame_height,
            inliers,
        })
    }

    /// Locates big-map screenshot (grayscale or BGR) and returns `(game_x, game_y)` plus `inliers`.
    pub fn locate_big_map(&mut self, frame_bgr_or_gray: &Mat) -> Result<(f32, f32, usize), CvError> {
        let m = self.locate_big_map_rect(frame_bgr_or_gray)?;
        Ok((m.game_center_x, m.game_center_y, m.inliers))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigMapRectMatch {
    pub game_center_x: f32,
    pub game_center_y: f32,
    pub game_tl_x: f32,
    pub game_tl_y: f32,
    pub game_br_x: f32,
    pub game_br_y: f32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub inliers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_real_teyvat_256_sift_features_if_present() {
        let dir = Path::new(r"d:\bgiprro\bgi_publish_new\Assets\Map\Teyvat");
        if !dir.join("Teyvat_0_256_SIFT.kp.bin").exists() {
            return;
        }
        let mut matcher = BigMapSiftMatcher::load_teyvat_256(dir).unwrap();
        assert!(matcher.keypoint_count() > 1000);
        assert_eq!(matcher.train_desc.rows() as usize, matcher.keypoint_count());
        assert_eq!(matcher.train_desc.cols(), 128);

        // Self-localization regression test against a crop of Teyvat_0_256.png
        let map_256_path = dir.join("Teyvat_0_256.png");
        if map_256_path.exists() {
            let full_256 = imread(&*map_256_path.to_string_lossy(), IMREAD_GRAYSCALE).unwrap();
            let crop_rect = core::Rect::new(1800, 1200, 420, 260);
            let crop = Mat::roi(&full_256, crop_rect).unwrap().clone_pointee();
            // Upscale 4x to mimic 1080p game screenshot before locate_big_map's 0.25x resize
            let mut upscaled = Mat::default();
            imgproc::resize(&crop, &mut upscaled, Size::new(1680, 1040), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

            let (gx, gy, inliers) = matcher.locate_big_map(&upscaled).unwrap();
            let expected_center_256_x = crop_rect.x as f32 + crop_rect.width as f32 / 2.0;
            let expected_center_256_y = crop_rect.y as f32 + crop_rect.height as f32 / 2.0;
            let (ex, ey) = matcher.coord.image_to_game(
                expected_center_256_x * BIGMAP_256_SCALE_TO_2048,
                expected_center_256_y * BIGMAP_256_SCALE_TO_2048,
            );
            assert!(inliers >= 20, "expected >=20 RANSAC inliers, got {inliers}");
            assert!((gx - ex).abs() < 4.0 && (gy - ey).abs() < 4.0, "got ({gx}, {gy}) vs ({ex}, {ey})");
        }
    }
}
