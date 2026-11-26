use std::fs;
use std::path::Path;

use opencv::calib3d::{RANSAC, find_homography};
use opencv::core::{self, CV_32FC1, DMatch, KeyPoint, Mat, MatTraitConst, Point2f, Size, Vector};
use opencv::features2d::{Feature2DTrait, FlannBasedMatcher, SIFT};
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imread};
use opencv::imgproc::{self, INTER_AREA};
use opencv::prelude::{DescriptorMatcherTraitConst, KeyPointTraitConst, MatTraitConstManual};

use crate::CvError;

/// Teyvat map layout defaults (`BetterGenshinImpact/GameTask/Common/Map/Maps/TeyvatMap.cs`).
/// Note: 15-row (`3840px`) 256-level assets use `TEYVAT_MAP_UP_ROWS_15 = 7`;
/// newer 19-row (`4864px`) Snezhnaya assets use `TEYVAT_MAP_UP_ROWS_19 = 11`.
pub const TEYVAT_MAP_ROWS: i32 = 19;
pub const TEYVAT_MAP_COLS: i32 = 22;
pub const TEYVAT_MAP_UP_ROWS_15: i32 = 7;
pub const TEYVAT_MAP_UP_ROWS_19: i32 = 11;
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
        Self::for_up_rows(TEYVAT_MAP_UP_ROWS_19)
    }
}

impl TeyvatCoordTransform {
    pub fn for_up_rows(up_rows: i32) -> Self {
        Self {
            origin_x: ((TEYVAT_MAP_LEFT_COLS + 1) * TEYVAT_BLOCK_WIDTH_2048) as f32,
            origin_y: ((up_rows + 1) * TEYVAT_BLOCK_WIDTH_2048) as f32,
            block_scale_to_1024: (TEYVAT_BLOCK_WIDTH_2048 as f32) / 1024.0,
        }
    }

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
    for chunk in bytes.as_chunks::<KP_RECORD_BYTES>().0 {
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
        return Err(CvError::InvalidImage(format!(
            "无法加载 SIFT 描述符图像: {path_str}"
        )));
    }
    let mut f32_mat = Mat::default();
    img.convert_to(&mut f32_mat, CV_32FC1, 1.0, 0.0)?;
    Ok(f32_mat)
}

fn detect_png_dimensions(path: &Path) -> Option<(u32, u32)> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 24 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((w, h))
}

fn infer_teyvat_coord_transform(map_dir: &Path, train_pts: &[Point2f]) -> TeyvatCoordTransform {
    let map_256_path = map_dir.join("Teyvat_0_256.png");
    let map_rows = if let Some((_, h)) = detect_png_dimensions(&map_256_path) {
        ((h as i32) + 128) / 256
    } else {
        let max_y = train_pts.iter().map(|p| p.y).fold(0.0f32, f32::max);
        if max_y <= 3900.0 { 15 } else { TEYVAT_MAP_ROWS }
    };

    let up_rows = if map_rows <= 15 {
        TEYVAT_MAP_UP_ROWS_15
    } else if map_rows >= 19 {
        TEYVAT_MAP_UP_ROWS_19
    } else {
        // Fallback linear shift (256px row = 1 row above/below): keep anchor consistent.
        TEYVAT_MAP_UP_ROWS_15 + (map_rows - 15)
    };
    TeyvatCoordTransform::for_up_rows(up_rows)
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
        let coord = infer_teyvat_coord_transform(dir, &train_pts);

        let sift = SIFT::create(0, 3, 0.04, 10.0, 1.6, false)?;
        let matcher = FlannBasedMatcher::create()?;

        Ok(Self {
            sift,
            matcher,
            train_pts,
            train_desc,
            coord,
        })
    }

    pub fn keypoint_count(&self) -> usize {
        self.train_pts.len()
    }

    pub fn coord_transform(&self) -> TeyvatCoordTransform {
        self.coord
    }

    /// Big-map SIFT match with projected viewport bounds in Genshin world coordinates.
    pub fn locate_big_map_rect(
        &mut self,
        frame_bgr_or_gray: &Mat,
    ) -> Result<BigMapRectMatch, CvError> {
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

        // Mirrors C# CaptureContent.DeriveTo1080P() + TeyvatMap.GetBigMapPosition (1/4 for 256-level SIFT map)
        let resize_scale = if frame_height >= 720 {
            (1080.0 / frame_height as f64) * 0.25
        } else {
            0.25
        };
        let mut resized = Mat::default();
        imgproc::resize(
            &gray,
            &mut resized,
            Size::new(0, 0),
            resize_scale,
            resize_scale,
            INTER_AREA,
        )?;
        let res_w = resized.cols();
        let res_h = resized.rows();

        let mut query_kps = Vector::<KeyPoint>::new();
        let mut query_desc = Mat::default();
        self.sift.detect_and_compute(
            &resized,
            &core::no_array(),
            &mut query_kps,
            &mut query_desc,
            false,
        )?;
        let kp_count = query_kps.len();
        if kp_count == 0 || query_desc.empty() {
            return Err(CvError::MatchFailed(format!(
                "大地图未提取到 SIFT 特征点 (src={frame_width}x{frame_height}, resized={res_w}x{res_h})"
            )));
        }

        let mut knn_matches = Vector::<Vector<DMatch>>::new();
        self.matcher
            .knn_train_match_def(&query_desc, &self.train_desc, &mut knn_matches, 2)?;

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

        let good_count = src_pts.len();
        if good_count < 7 {
            return Err(CvError::MatchFailed(format!(
                "有效 SIFT 匹配点不足 ({good_count} < 7, query_kps={kp_count}, src={frame_width}x{frame_height}, resized={res_w}x{res_h})"
            )));
        }

        let mut mask = Mat::default();
        let h_mat = find_homography(&src_pts, &dst_pts, &mut mask, RANSAC, 3.0)?;
        if h_mat.empty() {
            return Err(CvError::MatchFailed(format!(
                "RANSAC 单应矩阵计算失败 (good={good_count}, query_kps={kp_count}, src={frame_width}x{frame_height})"
            )));
        }

        let mut inliers = 0usize;
        if let Ok(mask_bytes) = mask.data_bytes() {
            inliers = mask_bytes.iter().filter(|&&b| b != 0).count();
        }
        if inliers < 4 {
            return Err(CvError::MatchFailed(format!(
                "RANSAC 内点数过少 ({inliers} < 4, good={good_count}, query_kps={kp_count}, src={frame_width}x{frame_height})"
            )));
        }

        let mut pts_in = Vector::<Point2f>::new();
        pts_in.push(Point2f::new(0.0, 0.0));
        pts_in.push(Point2f::new(res_w as f32 / 2.0, res_h as f32 / 2.0));
        pts_in.push(Point2f::new(res_w as f32, res_h as f32));
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

        if !game_center_x.is_finite()
            || !game_center_y.is_finite()
            || !game_tl_x.is_finite()
            || !game_tl_y.is_finite()
            || !game_br_x.is_finite()
            || !game_br_y.is_finite()
        {
            return Err(CvError::MatchFailed(format!(
                "单应变换坐标无效 ({game_center_x}, {game_center_y}), inliers={inliers}"
            )));
        }

        let span_x = game_tl_x - game_br_x;
        let span_y = game_tl_y - game_br_y;
        if span_x <= 100.0 || span_y <= 60.0 || span_x > 30000.0 || span_y > 20000.0 {
            return Err(CvError::MatchFailed(format!(
                "单应视口跨度异常 (span=({span_x:.1}, {span_y:.1}), inliers={inliers})"
            )));
        }

        let screen_aspect = frame_width as f32 / frame_height as f32;
        let map_aspect = span_x / span_y;
        if (map_aspect / screen_aspect - 1.0).abs() > 0.35 {
            return Err(CvError::MatchFailed(format!(
                "单应纵横比失真 (map={map_aspect:.2} vs screen={screen_aspect:.2}, inliers={inliers})"
            )));
        }

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
            query_kps: kp_count,
            good_matches: good_count,
        })
    }

    /// Locates big-map screenshot (grayscale or BGR) and returns `(game_x, game_y)` plus `inliers`.
    pub fn locate_big_map(
        &mut self,
        frame_bgr_or_gray: &Mat,
    ) -> Result<(f32, f32, usize), CvError> {
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
    pub query_kps: usize,
    pub good_matches: usize,
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

        // Self-localization regression test against Mondstadt region of Teyvat_0_256.png
        // Verifies both SIFT matching and UpRows origin alignment ((-876, 2278), not +4096 off).
        let map_256_path = dir.join("Teyvat_0_256.png");
        if map_256_path.exists() {
            let full_256 = imread(&*map_256_path.to_string_lossy(), IMREAD_GRAYSCALE).unwrap();
            let crop_rect = core::Rect::new(4105, 1348, 420, 260);
            let crop = Mat::roi(&full_256, crop_rect).unwrap().clone_pointee();
            // Upscale 4x to mimic 1080p game screenshot before locate_big_map's 0.25x resize
            let mut upscaled = Mat::default();
            imgproc::resize(
                &crop,
                &mut upscaled,
                Size::new(1680, 1040),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .unwrap();

            let (gx, gy, inliers) = matcher.locate_big_map(&upscaled).unwrap();
            assert!(inliers >= 20, "expected >=20 RANSAC inliers, got {inliers}");
            assert!(
                (gx - (-876.0)).abs() < 4.0 && (gy - 2278.0).abs() < 4.0,
                "expected Mondstadt (-876, 2278), got ({gx}, {gy})"
            );
        }
    }
}
