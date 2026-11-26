use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use opencv::core::{self, Mat, MatTraitConst, Rect, Size};
use opencv::imgcodecs::{IMREAD_COLOR, IMREAD_GRAYSCALE, imread};
use opencv::imgproc::{self, INTER_AREA, INTER_CUBIC, INTER_NEAREST, TM_SQDIFF};
use serde::Deserialize;

use crate::CvError;
use crate::matchers::{FastSqDiffMatcher, TemplateMatchNormalizer, match_template_subpix};
use crate::orientation::MiniMapPreprocessor;

const ROUGH_SIZE: i32 = 52;
const EXACT_SIZE: i32 = 260;
const ROUGH_ZOOM: f32 = 5.0;
const EXACT_ZOOM: f32 = 1.0;
const ROUGH_SEARCH_RADIUS: f32 = 50.0;
const EXACT_SEARCH_RADIUS: i32 = 20;
const CONFIDENCE_THRESHOLDS: [f64; 3] = [0.99, 0.97, 0.95];

type ScoredWorldPosition = ((f32, f32), f64);

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct LayerManifestItem {
    layer_id: String,
    #[serde(default)]
    floor: i32,
    #[serde(default = "default_scale")]
    scale: f32,
    #[serde(default)]
    name: String,
    #[serde(default)]
    left: f32,
    #[serde(default)]
    top: f32,
    #[serde(default)]
    is_over_size: bool,
}

fn default_scale() -> f32 {
    1.0
}

pub struct TemplateMapLayer {
    pub layer_id: String,
    pub floor: i32,
    pub scale: f32,
    pub name: String,
    pub left: f32,
    pub top: f32,
    coarse_matcher: FastSqDiffMatcher,
    fine_gray: Mat,
}

impl TemplateMapLayer {
    fn world_to_map(&self, world_x: f32, world_y: f32, zoom: f32) -> (i32, i32) {
        (
            ((self.left - world_x) * self.scale / zoom).round() as i32,
            ((self.top - world_y) * self.scale / zoom).round() as i32,
        )
    }

    fn map_to_world(&self, map_x: f32, map_y: f32, zoom: f32, mini_size: i32) -> (f32, f32) {
        let half = (mini_size as f32) / 2.0;
        (
            self.left - (map_x + half) * zoom / self.scale,
            self.top - (map_y + half) * zoom / self.scale,
        )
    }

    fn search_rect(center_x: i32, center_y: i32, half_side: i32, mini_size: i32) -> Rect {
        Rect::new(
            center_x - half_side - mini_size / 2,
            center_y - half_side - mini_size / 2,
            half_side * 2 + mini_size,
            half_side * 2 + mini_size,
        )
    }

    fn rough_match_global(
        &self,
        masked: &[Mat],
        mask_f: &Mat,
    ) -> Result<((f32, f32), f64), CvError> {
        let (loc, val) = self.coarse_matcher.match_global(masked, mask_f)?;
        Ok((
            self.map_to_world(loc.x as f32, loc.y as f32, ROUGH_ZOOM, ROUGH_SIZE),
            val,
        ))
    }

    fn rough_match_local(
        &self,
        masked: &[Mat],
        mask_f: &Mat,
        prev_world: (f32, f32),
    ) -> Result<Option<ScoredWorldPosition>, CvError> {
        let (rx, ry) = self.world_to_map(prev_world.0, prev_world.1, ROUGH_ZOOM);
        let desired = Self::search_rect(
            rx,
            ry,
            (ROUGH_SEARCH_RADIUS * self.scale) as i32,
            ROUGH_SIZE,
        );
        let bounds = Rect::new(
            0,
            0,
            self.coarse_matcher.width(),
            self.coarse_matcher.height(),
        );
        let roi = desired & bounds;
        if roi.width < ROUGH_SIZE || roi.height < ROUGH_SIZE {
            return Ok(None);
        }
        let (loc, val) = self.coarse_matcher.match_roi(masked, mask_f, roi, None)?;
        Ok(Some((
            self.map_to_world(
                (roi.x + loc.x) as f32,
                (roi.y + loc.y) as f32,
                ROUGH_ZOOM,
                ROUGH_SIZE,
            ),
            val,
        )))
    }

    fn exact_match(
        &self,
        mini_exact: &Mat,
        mask_exact: &Mat,
        rough_world: (f32, f32),
    ) -> Result<Option<ScoredWorldPosition>, CvError> {
        let (ex, ey) = self.world_to_map(rough_world.0, rough_world.1, EXACT_ZOOM);
        let desired = Self::search_rect(ex, ey, EXACT_SEARCH_RADIUS, EXACT_SIZE);
        let bounds = Rect::new(0, 0, self.fine_gray.cols(), self.fine_gray.rows());
        let roi = desired & bounds;
        if roi.width < EXACT_SIZE || roi.height < EXACT_SIZE {
            return Ok(None);
        }
        let big_roi = Mat::roi(&self.fine_gray, roi)?.clone_pointee();
        let (loc, val) = match_template_subpix(&big_roi, mini_exact, TM_SQDIFF, Some(mask_exact))?;
        Ok(Some((
            self.map_to_world(
                (roi.x as f32) + loc.x,
                (roi.y as f32) + loc.y,
                EXACT_ZOOM,
                EXACT_SIZE,
            ),
            val,
        )))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MiniMapMatchOutput {
    pub game_x: f32,
    pub game_y: f32,
    pub confidence: f64,
    pub floor: i32,
    pub camera_angle: f32,
    pub camera_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerLoadScope {
    /// Loads surface, terrain-patch, and city layers (`mapback*`, `path_map*`, `city*`), skipping underground `layer_map*`.
    SurfaceOnly,
    /// Loads every layer JSON in the directory (including underground `layer_map*`).
    AllLayers,
}

/// Template-match Teyvat minimap localizer with idle layer unloading.
pub struct TeyvatMiniMapLocalizer {
    map_dir: PathBuf,
    load_scope: LayerLoadScope,
    preprocessor: MiniMapPreprocessor,
    layers: Vec<TemplateMapLayer>,
    prev_layer_idx: Option<usize>,
    last_used_at: Instant,
    idle_timeout: Duration,
}

impl TeyvatMiniMapLocalizer {
    pub fn new(
        map_dir: impl AsRef<Path>,
        load_scope: LayerLoadScope,
        idle_timeout: Duration,
    ) -> Result<Self, CvError> {
        Ok(Self {
            map_dir: map_dir.as_ref().to_path_buf(),
            load_scope,
            preprocessor: MiniMapPreprocessor::new()?,
            layers: Vec::new(),
            prev_layer_idx: None,
            last_used_at: Instant::now(),
            idle_timeout,
        })
    }

    pub fn loaded_layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn unload_if_idle(&mut self) -> bool {
        if !self.layers.is_empty() && self.last_used_at.elapsed() >= self.idle_timeout {
            self.layers.clear();
            self.prev_layer_idx = None;
            return true;
        }
        false
    }

    /// Loads `mapback_info*.json` layers (plus `city_info*.json` if present) on demand.
    pub fn ensure_loaded(&mut self) -> Result<(), CvError> {
        self.last_used_at = Instant::now();
        if !self.layers.is_empty() {
            return Ok(());
        }

        let mut loaded = Vec::new();
        let entries = fs::read_dir(&self.map_dir)?;
        let mut manifests: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                    && match self.load_scope {
                        LayerLoadScope::SurfaceOnly => {
                            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                                n.starts_with("mapback")
                                    || n.starts_with("path_map")
                                    || n.starts_with("city")
                            })
                        }
                        LayerLoadScope::AllLayers => {
                            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                                n.starts_with("mapback")
                                    || n.starts_with("path_map")
                                    || n.starts_with("city")
                                    || n.starts_with("layer_map")
                            })
                        }
                    }
            })
            .collect();
        manifests.sort();

        for json_path in manifests {
            let text = fs::read_to_string(&json_path)?;
            let Ok(items) = serde_json::from_str::<Vec<LayerManifestItem>>(&text) else {
                continue;
            };
            for item in items {
                let color_name = format!("{}_color.webp", item.layer_id);
                let gray_name = format!(
                    "{}_gray.{}",
                    item.layer_id,
                    if item.is_over_size { "png" } else { "webp" }
                );
                crate::assets::validate_relative_path(&color_name)
                    .map_err(|e| CvError::InvalidImage(e.to_string()))?;
                crate::assets::validate_relative_path(&gray_name)
                    .map_err(|e| CvError::InvalidImage(e.to_string()))?;
                let color_path = self.map_dir.join(&color_name);
                let gray_path = self.map_dir.join(&gray_name);
                if !color_path.exists() || !gray_path.exists() {
                    continue;
                }
                let color_mat = imread(&*color_path.to_string_lossy(), IMREAD_COLOR)?;
                let gray_mat = imread(&*gray_path.to_string_lossy(), IMREAD_GRAYSCALE)?;
                if color_mat.empty() || gray_mat.empty() {
                    continue;
                }
                let coarse_matcher =
                    FastSqDiffMatcher::new(&color_mat, Size::new(ROUGH_SIZE, ROUGH_SIZE))?;
                loaded.push(TemplateMapLayer {
                    layer_id: item.layer_id,
                    floor: item.floor,
                    scale: item.scale,
                    name: item.name,
                    left: item.left,
                    top: item.top,
                    coarse_matcher,
                    fine_gray: gray_mat,
                });
            }
        }

        if loaded.is_empty() {
            return Err(CvError::InvalidImage(format!(
                "目录 {:?} 中未找到有效地图模板资源",
                self.map_dir
            )));
        }
        self.layers = loaded;
        Ok(())
    }

    pub fn locate(
        &mut self,
        minimap_bgr: &Mat,
        prev_game_pos: Option<(f32, f32)>,
        rank: usize,
    ) -> Result<Option<MiniMapMatchOutput>, CvError> {
        self.ensure_loaded()?;
        let (mini_bgr, mask, cam_angle, cam_conf) =
            self.preprocessor.get_minimap_and_mask(minimap_bgr)?;

        // Reject solid black/white/loading frames whose minimap circle lacks terrain contrast;
        // otherwise TM_SQDIFF can falsely match uniform ocean/cloud patches at high confidence.
        let mut mean = core::Scalar::default();
        let mut stddev = core::Scalar::default();
        core::mean_std_dev(&mini_bgr, &mut mean, &mut stddev, &mask)?;
        let max_stddev = stddev[0].max(stddev[1]).max(stddev[2]);
        if max_stddev < 4.0 {
            return Ok(None);
        }

        let mut mini_rough = Mat::default();
        let mut mask_rough = Mat::default();
        imgproc::resize(
            &mini_bgr,
            &mut mini_rough,
            Size::new(ROUGH_SIZE, ROUGH_SIZE),
            0.0,
            0.0,
            INTER_AREA,
        )?;
        imgproc::resize(
            &mask,
            &mut mask_rough,
            Size::new(ROUGH_SIZE, ROUGH_SIZE),
            0.0,
            0.0,
            INTER_NEAREST,
        )?;
        let (masked_roughs, mask_rough_f) =
            FastSqDiffMatcher::pre_process(&mini_rough, &mask_rough)?;
        let tpl_sum_sq = FastSqDiffMatcher::get_tpl_sum_sq(&masked_roughs, None)?;
        let mut norm_rough = TemplateMatchNormalizer::from_templates(
            &masked_roughs,
            Some(&mask_rough),
            TM_SQDIFF,
            None,
        )?;

        let mut mini_gray = Mat::default();
        imgproc::cvt_color(
            &mini_bgr,
            &mut mini_gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut mini_exact = Mat::default();
        let mut mask_exact = Mat::default();
        imgproc::resize(
            &mini_gray,
            &mut mini_exact,
            Size::new(EXACT_SIZE, EXACT_SIZE),
            0.0,
            0.0,
            INTER_CUBIC,
        )?;
        imgproc::resize(
            &mask,
            &mut mask_exact,
            Size::new(EXACT_SIZE, EXACT_SIZE),
            0.0,
            0.0,
            INTER_NEAREST,
        )?;
        let norm_exact =
            TemplateMatchNormalizer::from_template(&mini_exact, Some(&mask_exact), TM_SQDIFF)?;

        let mut best_layer_idx = None;
        let mut best_pos = (0.0f32, 0.0f32);
        let mut conf = 0.0f64;
        // Track top rough candidates so overlapping underground/surface layers don't fail
        // exact_match solely because a coarser 52x52 overlapping layer scored slightly higher.
        let rough_base = norm_rough.clone();
        let mut candidates: Vec<(usize, (f32, f32), f64)> = Vec::new();
        let mut record_candidate = |idx: usize, pos: (f32, f32), raw_val: f64| {
            let mut probe = rough_base.clone();
            if probe.update(raw_val + tpl_sum_sq) {
                let c = probe.confidence();
                candidates.push((idx, pos, c));
            }
        };

        match prev_game_pos {
            Some(prev) => {
                if let Some(prev_idx) = self.prev_layer_idx
                    && let Some(layer) = self.layers.get(prev_idx)
                    && let Some((pos, val)) =
                        layer.rough_match_local(&masked_roughs, &mask_rough_f, prev)?
                {
                    record_candidate(prev_idx, pos, val);
                    if norm_rough.update(val + tpl_sum_sq) {
                        best_layer_idx = Some(prev_idx);
                        best_pos = pos;
                        conf = norm_rough.confidence();
                    }
                }

                if conf < CONFIDENCE_THRESHOLDS[0] {
                    for (idx, layer) in self.layers.iter().enumerate() {
                        if Some(idx) == self.prev_layer_idx {
                            continue;
                        }
                        if let Some((pos, val)) =
                            layer.rough_match_local(&masked_roughs, &mask_rough_f, prev)?
                        {
                            record_candidate(idx, pos, val);
                            if norm_rough.update(val + tpl_sum_sq) {
                                best_layer_idx = Some(idx);
                                best_pos = pos;
                                conf = norm_rough.confidence();
                            }
                        }
                    }
                }
            }
            None => {
                for (idx, layer) in self.layers.iter().enumerate() {
                    let (pos, val) = layer.rough_match_global(&masked_roughs, &mask_rough_f)?;
                    record_candidate(idx, pos, val);
                    if norm_rough.update(val + tpl_sum_sq) {
                        best_layer_idx = Some(idx);
                        best_pos = pos;
                    }
                }
                conf = norm_rough.confidence();
            }
        }

        if best_layer_idx.is_none() || conf < CONFIDENCE_THRESHOLDS[2] {
            return Ok(None);
        }

        // Order unique layer candidates by rough confidence descending, keeping up to top 3.
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut tried_layers = Vec::with_capacity(3);
        let mut ordered_candidates = Vec::with_capacity(3);
        if let Some(idx) = best_layer_idx {
            tried_layers.push(idx);
            ordered_candidates.push((idx, best_pos));
        }
        for (idx, pos, c) in candidates {
            if ordered_candidates.len() >= 3 {
                break;
            }
            if c >= CONFIDENCE_THRESHOLDS[2] && !tried_layers.contains(&idx) {
                tried_layers.push(idx);
                ordered_candidates.push((idx, pos));
            }
        }

        let threshold = CONFIDENCE_THRESHOLDS[rank.min(CONFIDENCE_THRESHOLDS.len() - 1)];
        let mut best_exact: Option<(usize, (f32, f32), f64)> = None;
        for (layer_idx, rough_pos) in ordered_candidates {
            let layer = &self.layers[layer_idx];
            if let Some((exact_pos, exact_val)) =
                layer.exact_match(&mini_exact, &mask_exact, rough_pos)?
            {
                let mut exact_probe = norm_exact.clone();
                if exact_probe.update(exact_val) {
                    let c = exact_probe.confidence();
                    if c >= threshold && c <= 1.0 && best_exact.is_none_or(|(_, _, bc)| c > bc) {
                        best_exact = Some((layer_idx, exact_pos, c));
                    }
                }
            }
        }

        let Some((layer_idx, exact_pos, final_conf)) = best_exact else {
            return Ok(None);
        };
        let layer = &self.layers[layer_idx];

        self.prev_layer_idx = Some(layer_idx);
        Ok(Some(MiniMapMatchOutput {
            game_x: exact_pos.0,
            game_y: exact_pos.1,
            confidence: final_conf,
            floor: layer.floor,
            camera_angle: cam_angle,
            camera_confidence: cam_conf,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_surface_layers_and_unloads_on_idle() {
        let dir = Path::new(r"d:\bgiprro\bgi_publish_new\Assets\Map\Teyvat");
        if !dir.join("mapback_info.json").exists() {
            return;
        }
        let mut localizer =
            TeyvatMiniMapLocalizer::new(dir, LayerLoadScope::SurfaceOnly, Duration::from_millis(1))
                .unwrap();
        localizer.ensure_loaded().unwrap();
        assert!(localizer.loaded_layer_count() >= 5);

        std::thread::sleep(Duration::from_millis(5));
        assert!(localizer.unload_if_idle());
        assert_eq!(localizer.loaded_layer_count(), 0);
    }

    #[test]
    fn locates_real_world_minimap_captures_with_surface_and_city_layers() {
        let dir = Path::new(r"d:\bgiprro\bgi_publish_new\Assets\Map\Teyvat");
        if !dir.join("mapback_info.json").exists() {
            return;
        }
        let samples = [
            (
                r"d:\bgiprro\bettergi-rs\target\release\log\failed\20261125-102225_221-bigmap_sift_fail-4.png",
                9685.8f32,
                -1621.2f32,
            ),
            (
                r"d:\bgiprro\bettergi-rs\target\release\log\failed\20261125-102148_054-bigmap_sift_fail-3.png",
                4645.5f32,
                3467.9f32,
            ),
        ];
        let mut localizer =
            TeyvatMiniMapLocalizer::new(dir, LayerLoadScope::SurfaceOnly, Duration::from_secs(60))
                .unwrap();
        for (path, ex, ey) in samples {
            if !Path::new(path).exists() {
                continue;
            }
            let full = imread(path, opencv::imgcodecs::IMREAD_COLOR).unwrap();
            let mm = crate::matchers::extract_minimap_bgr(&full).unwrap();
            let out = localizer
                .locate(&mm, None, 2)
                .unwrap()
                .expect("expected surface+city+patch minimap match");
            assert!(
                (out.game_x - ex).abs() < 5.0 && (out.game_y - ey).abs() < 5.0,
                "expected near ({ex}, {ey}), got ({}, {})",
                out.game_x,
                out.game_y
            );
        }
    }

    #[test]
    fn rejects_uniform_loading_screen_frames() {
        let dir = Path::new(r"d:\bgiprro\bgi_publish_new\Assets\Map\Teyvat");
        if !dir.join("mapback_info.json").exists() {
            return;
        }
        let mut localizer =
            TeyvatMiniMapLocalizer::new(dir, LayerLoadScope::SurfaceOnly, Duration::from_secs(60))
                .unwrap();
        let white = Mat::new_rows_cols_with_default(
            212,
            212,
            core::CV_8UC3,
            opencv::core::Scalar::all(255.0),
        )
        .unwrap();
        let black = Mat::new_rows_cols_with_default(
            212,
            212,
            core::CV_8UC3,
            opencv::core::Scalar::all(0.0),
        )
        .unwrap();
        assert!(localizer.locate(&white, None, 2).unwrap().is_none());
        assert!(localizer.locate(&black, None, 2).unwrap().is_none());
    }

    #[test]
    fn loads_all_layers_including_layered_maps_and_measures_perf() {
        let dir = Path::new(
            r"d:\bgiprro\better-genshin-impact\BetterGenshinImpact\bin\Debug\net8.0-windows10.0.22621.0\Assets\Map\Teyvat",
        );
        if !dir.join("layer_map_info.json").exists() {
            return;
        }
        let t0 = Instant::now();
        let mut localizer =
            TeyvatMiniMapLocalizer::new(dir, LayerLoadScope::AllLayers, Duration::from_secs(180))
                .unwrap();
        localizer.ensure_loaded().unwrap();
        let load_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let count = localizer.loaded_layer_count();
        println!("AllLayers loaded {count} layers in {load_ms:.1}ms");
        assert!(
            count > 100,
            "expected > 100 layers with AllLayers, got {count}"
        );

        let sample = Path::new(
            r"d:\bgiprro\bettergi-rs\target\release\log\failed\20261125-102148_054-bigmap_sift_fail-3.png",
        );
        if sample.exists() {
            let full = imread(&*sample.to_string_lossy(), IMREAD_COLOR).unwrap();
            let mm = crate::matchers::extract_minimap_bgr(&full).unwrap();
            let t1 = Instant::now();
            let out_global = localizer.locate(&mm, None, 2).unwrap().unwrap();
            let g_ms = t1.elapsed().as_secs_f32() * 1000.0;
            let t2 = Instant::now();
            let out_local = localizer
                .locate(&mm, Some((4645.5, 3467.9)), 2)
                .unwrap()
                .unwrap();
            let l_ms = t2.elapsed().as_secs_f32() * 1000.0;
            println!(
                "AllLayers locate: global={g_ms:.1}ms ({:.1},{:.1}) local={l_ms:.1}ms ({:.1},{:.1})",
                out_global.game_x, out_global.game_y, out_local.game_x, out_local.game_y
            );
        }
    }
}
