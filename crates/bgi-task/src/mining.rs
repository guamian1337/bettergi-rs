//! Read-only mineral detection and target selection for Linnea's mining task.
//!
//! The constants and algorithms mirror `LinneaMiningTask` and `MineralCluster`.
//! This module deliberately performs no capture, inference, logging, or input.

pub const MINING_REQUIRED_STABLE_SAMPLES: usize = 3;
pub const MINING_MAX_ATTEMPTS: usize = 10;

const CONFIDENCE_THRESHOLD: f32 = 0.70;
const BASE_CLUSTER_DISTANCE: f64 = 400.0;
const BASE_CLUSTER_AREA: f64 = 1800.0;
const BASE_EDGE_IGNORE: f64 = 200.0;
const AREA_RATIO_THRESHOLD: f64 = 4.0;
const REFERENCE_FRAME_WIDTH: f64 = 1920.0;

const UNKNOWN_MISSING_OBSERVATION: &str = "未获得有效的采矿观察";
const UNKNOWN_INVALID_FRAME: &str = "采矿观察的画面尺寸无效";
const UNKNOWN_ATTEMPTS_EXHAUSTED: &str = "采矿观察未在次数上限内达到稳定";

/// An `ore` detection in floating-point frame coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiningDetection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

/// A detection box after the C# `(int)` conversion (truncation toward zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiningRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl MiningRect {
    fn center(self) -> (f64, f64) {
        (
            f64::from(self.x) + f64::from(self.width) / 2.0,
            f64::from(self.y) + f64::from(self.height) / 2.0,
        )
    }

    fn area(self) -> f64 {
        f64::from(self.width) * f64::from(self.height)
    }
}

/// The individual box selected for aiming, not the cluster's bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiningTarget {
    pub rect: MiningRect,
    pub center_x: f64,
    pub center_y: f64,
}

/// A complete recognition result. `target: None` is a valid empty detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiningObservation {
    pub frame_width: u32,
    pub frame_height: u32,
    pub cluster_count: usize,
    pub target: Option<MiningTarget>,
}

/// Keep ore boxes at or above the model confidence threshold.
///
/// Class filtering belongs to the inference adapter; every supplied detection
/// is already an `ore` result. Coordinates are converted only after filtering.
pub fn filter_ore_detections(detections: &[MiningDetection]) -> Vec<MiningRect> {
    detections
        .iter()
        .filter(|detection| detection.confidence >= CONFIDENCE_THRESHOLD)
        .map(|detection| MiningRect {
            x: detection.x as i32,
            y: detection.y as i32,
            width: detection.width as i32,
            height: detection.height as i32,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct MineralCluster {
    rects: Vec<MiningRect>,
    area_sum: f64,
    center_x: f64,
    center_y: f64,
    target_index: usize,
    prefer_right: bool,
}

impl MineralCluster {
    fn new(rect: MiningRect, prefer_right: bool) -> Self {
        let mut cluster = Self {
            rects: vec![rect],
            area_sum: rect.area(),
            center_x: 0.0,
            center_y: 0.0,
            target_index: 0,
            prefer_right,
        };
        cluster.recalculate();
        cluster
    }

    fn average_area(&self) -> f64 {
        self.area_sum / self.rects.len() as f64
    }

    fn try_add(&mut self, rect: MiningRect) -> bool {
        let average_area = self.average_area();
        let new_area = rect.area();
        if new_area > average_area * AREA_RATIO_THRESHOLD
            || new_area < average_area / AREA_RATIO_THRESHOLD
        {
            return false;
        }

        self.rects.push(rect);
        self.area_sum += new_area;
        self.recalculate();
        true
    }

    fn recalculate(&mut self) {
        let count = self.rects.len() as f64;
        self.center_x = self.rects.iter().map(|rect| rect.center().0).sum::<f64>() / count;
        self.center_y = self.rects.iter().map(|rect| rect.center().1).sum::<f64>() / count;

        let mut indices: Vec<usize> = (0..self.rects.len()).collect();
        indices.sort_by(|left, right| {
            squared_distance(self.rects[*left].center(), (self.center_x, self.center_y)).total_cmp(
                &squared_distance(self.rects[*right].center(), (self.center_x, self.center_y)),
            )
        });

        // `sort_by` is stable, so equal-distance boxes retain detection order,
        // matching C# `OrderBy(...).First()`. Multi-round scans choose the
        // rightmost of the nearest two; ProbeMining always passes `false`.
        self.target_index = if self.prefer_right && indices.len() >= 2 {
            if self.rects[indices[1]].center().0 > self.rects[indices[0]].center().0 {
                indices[1]
            } else {
                indices[0]
            }
        } else {
            indices[0]
        };
    }

    fn target(&self) -> MiningTarget {
        let rect = self.rects[self.target_index];
        let (center_x, center_y) = rect.center();
        MiningTarget {
            rect,
            center_x,
            center_y,
        }
    }
}

fn squared_distance(left: (f64, f64), right: (f64, f64)) -> f64 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    dx * dx + dy * dy
}

fn cluster_minerals(
    rects: &[MiningRect],
    width_scale: f64,
    prefer_right: bool,
) -> Vec<MineralCluster> {
    let reference_area = BASE_CLUSTER_AREA * width_scale * width_scale;
    let distance_threshold = BASE_CLUSTER_DISTANCE * width_scale;
    let mut clusters: Vec<MineralCluster> = Vec::new();

    for &rect in rects {
        let center = rect.center();
        let mut nearest_index = None;
        let mut nearest_distance = f64::MAX;
        for (index, cluster) in clusters.iter().enumerate() {
            let distance = squared_distance(center, (cluster.center_x, cluster.center_y)).sqrt();
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_index = Some(index);
            }
        }

        if let Some(index) = nearest_index {
            let nearest = &clusters[index];
            let combined_average_area = (nearest.average_area() * nearest.rects.len() as f64
                + rect.area())
                / (nearest.rects.len() + 1) as f64;
            let effective_distance =
                distance_threshold * (combined_average_area / reference_area.max(1.0)).sqrt();
            if nearest_distance < effective_distance && clusters[index].try_add(rect) {
                continue;
            }
        }

        clusters.push(MineralCluster::new(rect, prefer_right));
    }

    clusters
}

/// Cluster ore boxes and choose the eligible cluster nearest to frame center.
///
/// Centers must be within `200 * width_scale` of every frame edge. If any
/// center candidate exists, edge clusters are ignored; otherwise all clusters
/// remain eligible. The returned observation includes frame dimensions so a
/// resolution change cannot compare equal during stability gating.
///
/// Pass `prefer_right: false` for ProbeMining's single-round scan.
pub fn select_nearest_mining_target(
    rects: &[MiningRect],
    frame_width: u32,
    frame_height: u32,
    prefer_right: bool,
) -> MiningObservation {
    let width_scale = f64::from(frame_width) / REFERENCE_FRAME_WIDTH;
    let clusters = cluster_minerals(rects, width_scale, prefer_right);
    let cluster_count = clusters.len();
    let edge_ignore = BASE_EDGE_IGNORE * width_scale;
    let frame_width_f64 = f64::from(frame_width);
    let frame_height_f64 = f64::from(frame_height);

    let centered = clusters
        .iter()
        .filter(|cluster| {
            cluster.center_x >= edge_ignore
                && cluster.center_x <= frame_width_f64 - edge_ignore
                && cluster.center_y >= edge_ignore
                && cluster.center_y <= frame_height_f64 - edge_ignore
        })
        .collect::<Vec<_>>();
    let candidates = if centered.is_empty() {
        clusters.iter().collect::<Vec<_>>()
    } else {
        centered
    };

    let frame_center = (frame_width_f64 / 2.0, frame_height_f64 / 2.0);
    let mut nearest = None;
    let mut nearest_distance = f64::MAX;
    for cluster in candidates {
        let distance = squared_distance((cluster.center_x, cluster.center_y), frame_center);
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = Some(cluster);
        }
    }

    MiningObservation {
        frame_width,
        frame_height,
        cluster_count,
        target: nearest.map(MineralCluster::target),
    }
}

/// A successful stable observation, or an explicit unknown recognition state.
///
/// There is intentionally no "no ore" state based on missing inference. Only
/// [`MiningObservation`] with `target: None` can establish a stable empty frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MiningRecognitionState {
    Pending,
    Ready(MiningObservation),
    Unknown(&'static str),
}

/// Gates observations on exact equality and bounds the recognition attempt.
///
/// Once ready or terminally unknown, later samples cannot change the result.
/// Create a new gate for a new scan. `None` consumes an attempt and clears the
/// candidate/streak, but does not erase the attempt limit.
#[derive(Debug, Clone)]
pub struct MiningRecognitionStability {
    required: usize,
    max_attempts: usize,
    attempts: usize,
    consecutive: usize,
    candidate: Option<MiningObservation>,
    terminal: Option<MiningRecognitionState>,
}

impl Default for MiningRecognitionStability {
    fn default() -> Self {
        Self::new(MINING_REQUIRED_STABLE_SAMPLES, MINING_MAX_ATTEMPTS)
    }
}

impl MiningRecognitionStability {
    pub fn new(required: usize, max_attempts: usize) -> Self {
        assert!(required > 0, "required stable samples must be positive");
        assert!(max_attempts > 0, "maximum attempts must be positive");
        Self {
            required,
            max_attempts,
            attempts: 0,
            consecutive: 0,
            candidate: None,
            terminal: None,
        }
    }

    /// Record a complete observation, or clear stability when inference had no
    /// result. Empty detections must be passed as `Some(observation)`.
    pub fn observe(&mut self, sample: Option<MiningObservation>) -> MiningRecognitionState {
        if let Some(state) = self.terminal {
            return state;
        }

        self.attempts = self.attempts.saturating_add(1);
        let state = match sample {
            Some(sample) if sample.frame_width == 0 || sample.frame_height == 0 => {
                self.clear_candidate();
                if self.attempts >= self.max_attempts {
                    MiningRecognitionState::Unknown(UNKNOWN_ATTEMPTS_EXHAUSTED)
                } else {
                    MiningRecognitionState::Unknown(UNKNOWN_INVALID_FRAME)
                }
            }
            Some(sample) => {
                if self.candidate == Some(sample) {
                    self.consecutive += 1;
                } else {
                    self.candidate = Some(sample);
                    self.consecutive = 1;
                }

                if self.consecutive >= self.required {
                    MiningRecognitionState::Ready(sample)
                } else if self.attempts >= self.max_attempts {
                    MiningRecognitionState::Unknown(UNKNOWN_ATTEMPTS_EXHAUSTED)
                } else {
                    MiningRecognitionState::Pending
                }
            }
            None => {
                self.clear_candidate();
                if self.attempts >= self.max_attempts {
                    MiningRecognitionState::Unknown(UNKNOWN_ATTEMPTS_EXHAUSTED)
                } else {
                    MiningRecognitionState::Unknown(UNKNOWN_MISSING_OBSERVATION)
                }
            }
        };

        if matches!(state, MiningRecognitionState::Ready(_)) || self.attempts >= self.max_attempts {
            self.terminal = Some(state);
        }
        state
    }

    /// Immediately report a terminal unknown result, e.g. a missing model or
    /// failed inference. The caller-provided reason is preserved verbatim.
    pub fn reject(&mut self, reason: &'static str) -> MiningRecognitionState {
        if let Some(state) = self.terminal {
            return state;
        }
        self.clear_candidate();
        let state = MiningRecognitionState::Unknown(reason);
        self.terminal = Some(state);
        state
    }

    fn clear_candidate(&mut self) {
        self.candidate = None;
        self.consecutive = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(x: f32, y: f32, width: f32, height: f32, confidence: f32) -> MiningDetection {
        MiningDetection {
            x,
            y,
            width,
            height,
            confidence,
        }
    }

    fn rect(x: i32, y: i32, width: i32, height: i32) -> MiningRect {
        MiningRect {
            x,
            y,
            width,
            height,
        }
    }

    fn centered_rect(center_x: i32, center_y: i32, area: i32) -> MiningRect {
        debug_assert_eq!(area % 30, 0);
        let width = area / 30;
        rect(center_x - width / 2, center_y - 15, width, 30)
    }

    fn observation(frame_width: u32, frame_height: u32, rects: &[MiningRect]) -> MiningObservation {
        select_nearest_mining_target(rects, frame_width, frame_height, false)
    }

    fn empty_observation(frame_width: u32, frame_height: u32) -> MiningObservation {
        observation(frame_width, frame_height, &[])
    }

    #[test]
    fn confidence_threshold_is_inclusive_and_rejects_uncertain_detections() {
        let boxes = filter_ore_detections(&[
            detection(1.0, 2.0, 3.0, 4.0, 0.70),
            detection(5.0, 6.0, 7.0, 8.0, 0.91),
            detection(9.0, 10.0, 11.0, 12.0, 0.699_999_9),
            detection(13.0, 14.0, 15.0, 16.0, f32::NAN),
        ]);
        assert_eq!(
            boxes,
            vec![rect(1, 2, 3, 4), rect(5, 6, 7, 8)],
            "0.70f must be accepted"
        );
    }

    #[test]
    fn floating_point_box_coordinates_truncate_toward_zero() {
        assert_eq!(
            filter_ore_detections(&[detection(-10.9, -20.9, 30.9, 40.9, 0.8)]),
            vec![rect(-10, -20, 30, 40)]
        );
        assert_eq!(
            filter_ore_detections(&[detection(10.9, 20.9, 30.9, 40.9, 0.8)]),
            vec![rect(10, 20, 30, 40)]
        );
    }

    #[test]
    fn clustering_uses_scaled_strict_distance_threshold() {
        for (frame_width, frame_height, distance) in [(1920, 1080, 400), (960, 540, 200)] {
            let area = (1800.0 * (f64::from(frame_width) / 1920.0).powi(2)) as i32;
            let first = centered_rect(300, 300, area);
            let inside = centered_rect(300 + distance - 1, 300, area);
            let boundary = centered_rect(300 + distance, 300, area);

            assert_eq!(
                observation(frame_width, frame_height, &[first, inside]).cluster_count,
                1
            );
            assert_eq!(
                observation(frame_width, frame_height, &[first, boundary]).cluster_count,
                2,
                "distance == effective threshold must start a new cluster"
            );
        }
    }

    #[test]
    fn clustering_distance_uses_combined_average_area() {
        let small = centered_rect(300, 300, 1800);
        let large = centered_rect(1000, 300, 7200);
        let result = observation(1920, 1080, &[small, large]);
        assert_eq!(
            result.cluster_count, 2,
            "combined area gives a threshold of about 632, not the incoming box's 800"
        );
    }

    #[test]
    fn clustering_scales_distance_for_large_average_area() {
        let first = centered_rect(300, 300, 7200);
        let nearby = centered_rect(1000, 300, 7200);
        assert_eq!(
            observation(1920, 1080, &[first, nearby]).cluster_count,
            1,
            "4x reference area increases the distance threshold to 800"
        );
    }

    #[test]
    fn clustering_enforces_four_to_one_average_area_ratio() {
        let small = centered_rect(300, 300, 1800);
        let four_times = centered_rect(400, 300, 7200);
        let too_large = centered_rect(500, 300, 7230);
        let too_small = centered_rect(400, 300, 420);

        assert_eq!(
            observation(1920, 1080, &[small, four_times]).cluster_count,
            1,
            "exactly 4:1 is accepted"
        );
        assert_eq!(
            observation(1920, 1080, &[small, too_large]).cluster_count,
            2
        );
        assert_eq!(
            observation(1920, 1080, &[small, too_small]).cluster_count,
            2
        );
    }

    #[test]
    fn greedy_clustering_only_tries_the_nearest_existing_cluster() {
        let rects = [
            centered_rect(300, 300, 1800),
            centered_rect(3000, 300, 1800),
            centered_rect(320, 300, 14400),
        ];
        assert_eq!(
            observation(1920, 1080, &rects).cluster_count,
            3,
            "the nearby 1800-area cluster rejects the 8x box; the farther cluster is not tried"
        );
    }

    #[test]
    fn single_round_target_is_the_box_nearest_to_the_centroid() {
        let rects = [
            centered_rect(700, 500, 1800),
            centered_rect(900, 500, 1800),
            centered_rect(1100, 500, 1800),
        ];
        let result = observation(1920, 1080, &rects);
        assert_eq!(result.cluster_count, 1);
        let target = result.target.unwrap();
        assert_eq!(target.rect, rects[1]);
        assert_eq!((target.center_x, target.center_y), (900.0, 500.0));
    }

    #[test]
    fn equal_distance_target_ties_keep_detection_order_unless_prefer_right() {
        let rects = [
            centered_rect(800, 500, 1800),
            centered_rect(1100, 500, 1800),
        ];
        let single = observation(1920, 1080, &rects);
        assert_eq!(single.target.unwrap().rect, rects[0]);

        let multi = select_nearest_mining_target(&rects, 1920, 1080, true);
        assert_eq!(multi.target.unwrap().rect, rects[1]);
    }

    #[test]
    fn center_candidates_exclude_edge_clusters_on_both_axes() {
        let edge = centered_rect(199, 540, 1800);
        let center = centered_rect(1719, 879, 1800);
        let x_filtered = observation(1920, 1080, &[edge, center]);
        assert_eq!(x_filtered.cluster_count, 2);
        assert_eq!(x_filtered.target.unwrap().rect, center);

        let edge = centered_rect(960, 199, 1800);
        let center = centered_rect(1719, 540, 1800);
        let y_filtered = observation(1920, 1080, &[edge, center]);
        assert_eq!(y_filtered.target.unwrap().rect, center);
    }

    #[test]
    fn center_boundaries_are_inclusive_and_scale_with_frame_width() {
        let edge = centered_rect(199, 100, 1800);
        let left_boundary = centered_rect(200, 880, 1800);
        let right_boundary = centered_rect(1720, 880, 1800);
        assert_eq!(
            observation(1920, 1080, &[edge, left_boundary])
                .target
                .unwrap()
                .rect,
            left_boundary
        );
        assert_eq!(
            observation(1920, 1080, &[edge, right_boundary])
                .target
                .unwrap()
                .rect,
            right_boundary
        );

        let edge = centered_rect(99, 270, 600);
        let center = centered_rect(860, 440, 600);
        let scaled = observation(960, 540, &[edge, center]);
        assert_eq!(scaled.target.unwrap().rect, center);
    }

    #[test]
    fn target_falls_back_to_all_clusters_when_no_center_candidate_exists() {
        let left = centered_rect(100, 540, 1800);
        let right = centered_rect(1720, 540, 1800);
        let result = observation(1920, 1080, &[left, right]);
        assert_eq!(result.cluster_count, 2);
        assert_eq!(result.target.unwrap().rect, right);
    }

    #[test]
    fn final_target_minimizes_squared_distance_to_frame_center() {
        let farther = centered_rect(700, 500, 1800);
        let nearer = centered_rect(950, 550, 1800);
        let result = observation(1920, 1080, &[nearer, farther]);
        assert_eq!(result.target.unwrap().rect, nearer);

        let left = centered_rect(660, 540, 1800);
        let right = centered_rect(1260, 540, 1800);
        let tied = observation(1920, 1080, &[left, right]);
        assert_eq!(tied.target.unwrap().rect, left);
    }

    #[test]
    fn empty_observation_becomes_ready_after_three_identical_frames() {
        let mut stability = MiningRecognitionStability::default();
        let empty = empty_observation(1920, 1080);
        assert_eq!(
            stability.observe(Some(empty)),
            MiningRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(empty)),
            MiningRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(empty)),
            MiningRecognitionState::Ready(empty)
        );
        assert_eq!(empty.cluster_count, 0);
        assert_eq!(empty.target, None);
    }

    #[test]
    fn changed_observation_resets_the_consecutive_streak() {
        let mut stability = MiningRecognitionStability::default();
        let empty = empty_observation(1920, 1080);
        let target = observation(1920, 1080, &[centered_rect(960, 540, 1800)]);
        stability.observe(Some(empty));
        stability.observe(Some(empty));
        assert_eq!(
            stability.observe(Some(target)),
            MiningRecognitionState::Pending
        );
        stability.observe(Some(target));
        assert_eq!(
            stability.observe(Some(target)),
            MiningRecognitionState::Ready(target)
        );
    }

    #[test]
    fn missing_observation_clears_stability_without_erasing_attempt_limit() {
        let mut stability = MiningRecognitionStability::default();
        let empty = empty_observation(1920, 1080);
        stability.observe(Some(empty));
        stability.observe(Some(empty));
        assert_eq!(
            stability.observe(None),
            MiningRecognitionState::Unknown(UNKNOWN_MISSING_OBSERVATION)
        );
        stability.observe(Some(empty));
        assert_eq!(
            stability.observe(Some(empty)),
            MiningRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(empty)),
            MiningRecognitionState::Ready(empty)
        );

        let mut missing = MiningRecognitionStability::default();
        for _ in 0..MINING_MAX_ATTEMPTS - 1 {
            assert_eq!(
                missing.observe(None),
                MiningRecognitionState::Unknown(UNKNOWN_MISSING_OBSERVATION)
            );
        }
        assert_eq!(
            missing.observe(None),
            MiningRecognitionState::Unknown(UNKNOWN_ATTEMPTS_EXHAUSTED)
        );
    }

    #[test]
    fn frame_size_changes_make_empty_observations_inconsistent() {
        let mut stability = MiningRecognitionStability::default();
        let original = empty_observation(1920, 1080);
        let resized = empty_observation(1920, 1081);
        assert_ne!(original, resized);
        stability.observe(Some(original));
        stability.observe(Some(original));
        assert_eq!(
            stability.observe(Some(resized)),
            MiningRecognitionState::Pending
        );
        stability.observe(Some(resized));
        assert_eq!(
            stability.observe(Some(resized)),
            MiningRecognitionState::Ready(resized)
        );
    }

    #[test]
    fn jitter_becomes_unknown_at_the_attempt_limit() {
        let mut stability = MiningRecognitionStability::default();
        for attempt in 0..MINING_MAX_ATTEMPTS - 1 {
            let width = if attempt % 2 == 0 { 1920 } else { 1921 };
            assert_eq!(
                stability.observe(Some(empty_observation(width, 1080))),
                MiningRecognitionState::Pending
            );
        }
        assert_eq!(
            stability.observe(Some(empty_observation(1921, 1080))),
            MiningRecognitionState::Unknown(UNKNOWN_ATTEMPTS_EXHAUSTED)
        );
        assert_eq!(
            stability.observe(Some(empty_observation(1921, 1080))),
            MiningRecognitionState::Unknown(UNKNOWN_ATTEMPTS_EXHAUSTED)
        );
    }

    #[test]
    fn model_error_or_missing_model_is_immediately_and_permanently_unknown() {
        for reason in ["模型缺失", "模型推理失败", "识别结果不确定"] {
            let mut stability = MiningRecognitionStability::default();
            let empty = empty_observation(1920, 1080);
            stability.observe(Some(empty));
            stability.observe(Some(empty));
            assert_eq!(
                stability.reject(reason),
                MiningRecognitionState::Unknown(reason)
            );
            assert_eq!(
                stability.observe(Some(empty)),
                MiningRecognitionState::Unknown(reason)
            );
        }
    }

    #[test]
    fn invalid_frame_cannot_become_a_stable_empty_observation() {
        let mut stability = MiningRecognitionStability::default();
        for _ in 0..MINING_REQUIRED_STABLE_SAMPLES {
            assert_eq!(
                stability.observe(Some(empty_observation(0, 1080))),
                MiningRecognitionState::Unknown(UNKNOWN_INVALID_FRAME)
            );
        }
        assert!(!matches!(
            stability.observe(Some(empty_observation(0, 1080))),
            MiningRecognitionState::Ready(_)
        ));
    }

    #[test]
    fn ready_and_rejected_gates_are_terminal() {
        let mut ready = MiningRecognitionStability::default();
        let empty = empty_observation(1920, 1080);
        for _ in 0..MINING_REQUIRED_STABLE_SAMPLES {
            ready.observe(Some(empty));
        }
        assert_eq!(ready.observe(None), MiningRecognitionState::Ready(empty));
        assert_eq!(
            ready.reject("后续推理失败"),
            MiningRecognitionState::Ready(empty),
            "a new rejection belongs to a new scan"
        );
    }
}
