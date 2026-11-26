use std::time::{Duration, Instant};

use crate::{Waypoint, is_basic_action_supported};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigMapProjection {
    pub game_tl_x: f32,
    pub game_tl_y: f32,
    pub game_br_x: f32,
    pub game_br_y: f32,
    pub frame_width: u32,
    pub frame_height: u32,
}

#[derive(Debug, Clone)]
pub struct PlannedWaypoint {
    pub index: usize,
    pub waypoint: Waypoint,
    pub basic_action_supported: bool,
}

pub fn split_segments_by_teleport(positions: &[Waypoint]) -> Vec<Vec<PlannedWaypoint>> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    for (index, wp) in positions.iter().enumerate() {
        if wp.waypoint_type == "teleport" && !current.is_empty() {
            segments.push(current);
            current = Vec::new();
        }
        current.push(PlannedWaypoint {
            index,
            waypoint: wp.clone(),
            basic_action_supported: is_basic_action_supported(wp.action.as_deref()),
        });
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

pub fn distance_2d(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
}

/// Computes target camera heading in degrees `[0, 360)` matching `CameraOrientation` (`0°=East(-x), 90°=South(-y), 180°=West(+x), 270°=North(+y)`).
pub fn target_orientation_deg(cur_x: f64, cur_y: f64, target_x: f64, target_y: f64) -> f32 {
    // Convert world delta (+x=West, +y=North) to minimap image delta (+x=East, +y=South).
    let img_dx = -(target_x - cur_x);
    let img_dy = -(target_y - cur_y);
    if img_dx.abs() < 1e-6 && img_dy.abs() < 1e-6 {
        return 0.0;
    }
    (img_dy.atan2(img_dx).to_degrees().rem_euclid(360.0)) as f32
}

/// Signed shortest delta `(cur - target)` in `[-180, 180)`, matching `CameraRotateTask.cs`.
pub fn signed_angle_diff_deg(current_deg: f32, target_deg: f32) -> f32 {
    (current_deg - target_deg + 540.0).rem_euclid(360.0) - 180.0
}

/// Converts heading error to horizontal mouse delta matching `CameraRotateTask.cs`.
pub fn turn_mouse_dx(diff_deg: f32, dpi_scale: f32) -> i32 {
    let abs = diff_deg.abs();
    if abs < 0.4 {
        return 0;
    }
    let ratio = if abs > 90.0 {
        4.0
    } else if abs > 30.0 {
        3.0
    } else if abs > 5.0 {
        2.0
    } else {
        1.0
    };
    (-ratio * diff_deg * dpi_scale.max(0.5)).round() as i32
}

/// Projects a Genshin world `(target_x, target_y)` to client pixel coordinates `(x, y)` inside `proj`.
pub fn project_bigmap_click(
    proj: &BigMapProjection,
    target_x: f64,
    target_y: f64,
) -> Option<(f32, f32)> {
    let span_x = (proj.game_br_x - proj.game_tl_x) as f64;
    let span_y = (proj.game_br_y - proj.game_tl_y) as f64;
    if span_x.abs() < 1.0 || span_y.abs() < 1.0 || proj.frame_width == 0 || proj.frame_height == 0 {
        return None;
    }
    let u = (target_x - proj.game_tl_x as f64) / span_x;
    let v = (target_y - proj.game_tl_y as f64) / span_y;
    Some((
        (u * proj.frame_width as f64) as f32,
        (v * proj.frame_height as f64) as f32,
    ))
}

pub fn is_bigmap_click_in_safe_zone(x: f32, y: f32, width: u32, height: u32) -> bool {
    let w = width as f32;
    let h = height as f32;
    x >= w * 0.18 && x <= w * 0.72 && y >= h * 0.14 && y <= h * 0.84
}

/// Estimates Genshin big-map zoom level (~1.0 zoomed in .. ~6.0 zoomed out)
/// using the 1080p baseline `MapScaleFactor = 2.361` (`span_y ≈ 1080 * zoom / 2.361`).
pub fn estimate_bigmap_zoom_level(proj: &BigMapProjection) -> f32 {
    let span_y = (proj.game_tl_y - proj.game_br_y).abs();
    let span_x = (proj.game_tl_x - proj.game_br_x).abs();
    if span_y > 1.0 {
        span_y * 2.361 / 1080.0
    } else {
        span_x * 2.361 / 1920.0
    }
}

/// Predicts where `(click_x, click_y)` will move on screen when zooming centered at `(width/2, height/2)`
/// from `current_zoom` to `target_zoom`.
pub fn predict_post_zoom_click(
    click_x: f32,
    click_y: f32,
    width: u32,
    height: u32,
    current_zoom: f32,
    target_zoom: f32,
) -> (f32, f32) {
    if !current_zoom.is_finite() || !target_zoom.is_finite() || target_zoom <= 0.05 {
        return (click_x, click_y);
    }
    let scale = current_zoom / target_zoom;
    let half_w = width as f32 * 0.5;
    let half_h = height as f32 * 0.5;
    (
        half_w + (click_x - half_w) * scale,
        half_h + (click_y - half_h) * scale,
    )
}

/// Returns `(x1, y1, x2, y2)` client drag coordinates to pull `(click_x, click_y)` toward screen center.
pub fn compute_bigmap_drag(
    click_x: f32,
    click_y: f32,
    width: u32,
    height: u32,
) -> (i32, i32, i32, i32) {
    let w = (width.max(320)) as f32;
    let h = (height.max(240)) as f32;
    let scale = (w / 1920.0).min(h / 1080.0);

    let min_x = (125.0 * scale).clamp(40.0, w * 0.25);
    let max_x = (w - 150.0 * scale).clamp(w * 0.75, w - 40.0);
    let min_y = (115.0 * scale).clamp(36.0, h * 0.25);
    let max_y = (h - 115.0 * scale).clamp(h * 0.75, h - 36.0);

    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    let dx = cx - click_x;
    let dy = cy - click_y;
    let dist = (dx * dx + dy * dy).sqrt().max(1.0);

    // Uniformly scale drag vector so neither axis hits edge clamp independently and tilts the angle.
    let max_drag = w.min(h) * 0.45;
    let max_span_x = (max_x - min_x) * 0.90;
    let max_span_y = (max_y - min_y) * 0.90;
    let mut scale_factor = (max_drag / dist).min(1.0);
    if dx.abs() > 1e-3 {
        scale_factor = scale_factor.min(max_span_x / dx.abs());
    }
    if dy.abs() > 1e-3 {
        scale_factor = scale_factor.min(max_span_y / dy.abs());
    }

    let step_x = dx * scale_factor;
    let step_y = dy * scale_factor;
    let mut x1 = cx - step_x * 0.5;
    let mut y1 = cy - step_y * 0.5;
    let mut x2 = cx + step_x * 0.5;
    let mut y2 = cy + step_y * 0.5;

    // Avoid top-left map UI overlay (`< 390x220` at 1080p) by shifting downward if room permits.
    let forbid_x = 390.0 * scale;
    let forbid_y = 220.0 * scale;
    let hits_forbid = (x1 < forbid_x && y1 < forbid_y) || (x2 < forbid_x && y2 < forbid_y);
    if hits_forbid {
        let need_dy = (forbid_y + 12.0 - y1.min(y2)).max(0.0);
        let room_dy = (max_y - y1.max(y2)).max(0.0);
        let shift_y = need_dy.min(room_dy);
        y1 += shift_y;
        y2 += shift_y;

        if (x1 < forbid_x && y1 < forbid_y) || (x2 < forbid_x && y2 < forbid_y) {
            let need_dx = (forbid_x + 12.0 - x1.min(x2)).max(0.0);
            let room_dx = (max_x - x1.max(x2)).max(0.0);
            let shift_x = need_dx.min(room_dx);
            x1 += shift_x;
            x2 += shift_x;
        }
    }

    (
        x1.round() as i32,
        y1.round() as i32,
        x2.round() as i32,
        y2.round() as i32,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CountryAnchor {
    pub name: &'static str,
    pub x: f64,
    pub y: f64,
    pub aliases: &'static [&'static str],
}

/// Center positions matching `MapLazyAssets.CountryPositions` in C# BetterGI.
pub const TEYVAT_COUNTRY_ANCHORS: &[CountryAnchor] = &[
    CountryAnchor {
        name: "蒙德",
        x: -876.0,
        y: 2278.0,
        aliases: &["蒙德", "蒙德城"],
    },
    CountryAnchor {
        name: "璃月",
        x: 270.0,
        y: -666.0,
        aliases: &["璃月", "璃月港"],
    },
    CountryAnchor {
        name: "稻妻",
        x: -4400.0,
        y: -3050.0,
        aliases: &["稻妻", "稻妻城"],
    },
    CountryAnchor {
        name: "须弥",
        x: 2877.0,
        y: -374.0,
        aliases: &["须弥", "須彌", "须弥城"],
    },
    CountryAnchor {
        name: "枫丹",
        x: 4515.0,
        y: 3631.0,
        aliases: &["枫丹", "楓丹", "枫丹廷"],
    },
    CountryAnchor {
        name: "纳塔",
        x: 8973.5,
        y: -1879.1,
        aliases: &["纳塔", "納塔"],
    },
    CountryAnchor {
        name: "挪德卡莱",
        x: 9542.25,
        y: 1661.84,
        aliases: &["挪德卡莱", "挪德卡萊"],
    },
    CountryAnchor {
        name: "至冬",
        x: 6755.051,
        y: 9480.659,
        aliases: &["至冬"],
    },
];

pub fn should_switch_country(
    current_center: Option<(f64, f64)>,
    target_x: f64,
    target_y: f64,
) -> Option<&'static CountryAnchor> {
    let best = TEYVAT_COUNTRY_ANCHORS
        .iter()
        .map(|a| (a, distance_2d(a.x, a.y, target_x, target_y)))
        .min_by(|a, b| a.1.total_cmp(&b.1));

    let (best_anchor, best_dist) = best?;
    match current_center {
        None => Some(best_anchor),
        Some((cx, cy)) => {
            let cur_dist = distance_2d(cx, cy, target_x, target_y);
            if cur_dist > 900.0 && best_dist + 450.0 < cur_dist {
                Some(best_anchor)
            } else {
                None
            }
        }
    }
}

pub fn normalize_region_ocr_text(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .map(|c| match c {
            '須' => '须',
            '彌' => '弥',
            '楓' => '枫',
            '納' => '纳',
            '萊' => '莱',
            '宮' => '宫',
            _ => c,
        })
        .collect::<String>()
        .to_ascii_lowercase()
}

pub struct StuckMonitor {
    window: Duration,
    min_displacement: f64,
    anchor_at: Instant,
    anchor_pos: Option<(f64, f64)>,
}

impl StuckMonitor {
    pub fn new(window: Duration, min_displacement: f64) -> Self {
        Self {
            window,
            min_displacement,
            anchor_at: Instant::now(),
            anchor_pos: None,
        }
    }

    pub fn reset(&mut self, pos: Option<(f64, f64)>) {
        self.anchor_at = Instant::now();
        self.anchor_pos = pos;
    }

    pub fn observe(&mut self, pos: (f64, f64)) -> bool {
        self.observe_at(pos, Instant::now())
    }

    pub fn observe_at(&mut self, pos: (f64, f64), now: Instant) -> bool {
        let Some(anchor) = self.anchor_pos else {
            self.anchor_pos = Some(pos);
            self.anchor_at = now;
            return false;
        };
        if now.saturating_duration_since(self.anchor_at) < self.window {
            return false;
        }
        let moved = distance_2d(anchor.0, anchor.1, pos.0, pos.1);
        self.anchor_pos = Some(pos);
        self.anchor_at = now;
        moved < self.min_displacement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_and_signed_turn_match_cardinal_directions() {
        // Genshin world axes: +x = West (180°), -x = East (0°), +y = North (270°), -y = South (90°).
        assert!((target_orientation_deg(0.0, 0.0, -10.0, 0.0) - 0.0).abs() < 1e-3);
        assert!((target_orientation_deg(0.0, 0.0, 0.0, -10.0) - 90.0).abs() < 1e-3);
        assert!((target_orientation_deg(0.0, 0.0, 10.0, 0.0) - 180.0).abs() < 1e-3);
        assert!((target_orientation_deg(0.0, 0.0, 0.0, 10.0) - 270.0).abs() < 1e-3);

        assert!((signed_angle_diff_deg(10.0, 350.0) - 20.0).abs() < 1e-3);
        assert!((signed_angle_diff_deg(350.0, 10.0) + 20.0).abs() < 1e-3);
        assert!(turn_mouse_dx(20.0, 1.0) < 0);
        assert!(turn_mouse_dx(-20.0, 1.0) > 0);
    }

    #[test]
    fn projects_bigmap_coordinates_even_when_world_axes_flip() {
        let proj = BigMapProjection {
            game_tl_x: 1000.0,
            game_tl_y: 2000.0,
            game_br_x: -920.0,
            game_br_y: 920.0,
            frame_width: 1920,
            frame_height: 1080,
        };
        let (cx, cy) = project_bigmap_click(&proj, 40.0, 1460.0).unwrap();
        assert!((cx - 960.0).abs() < 1.0);
        assert!((cy - 540.0).abs() < 1.0);
        assert!(is_bigmap_click_in_safe_zone(cx, cy, 1920, 1080));
        assert!(!is_bigmap_click_in_safe_zone(1850.0, 980.0, 1920, 1080));
    }

    #[test]
    fn stuck_monitor_flags_when_displacement_below_threshold() {
        let mut mon = StuckMonitor::new(Duration::from_secs(2), 2.5);
        let t0 = Instant::now();
        assert!(!mon.observe_at((100.0, 200.0), t0));
        assert!(!mon.observe_at((100.5, 200.5), t0 + Duration::from_millis(1500)));
        assert!(mon.observe_at((101.0, 200.5), t0 + Duration::from_millis(2200)));
        assert!(!mon.observe_at((106.0, 205.0), t0 + Duration::from_millis(4500)));
    }

    #[test]
    fn bigmap_drag_pulls_target_toward_center_without_inversion() {
        // Target far to the right (+x screen): cursor must drag right -> left (x2 < x1).
        let (x1, _, x2, _) = compute_bigmap_drag(2800.0, 540.0, 1920, 1080);
        assert!(x2 < x1, "expected drag left, got {x1} -> {x2}");

        // Target far to the left (-x screen): cursor must drag left -> right (x2 > x1).
        let (x1, _, x2, _) = compute_bigmap_drag(-1000.0, 540.0, 1920, 1080);
        assert!(x2 > x1, "expected drag right, got {x1} -> {x2}");

        // Target far below (+y screen): cursor must drag down -> up (y2 < y1).
        let (_, y1, _, y2) = compute_bigmap_drag(960.0, 2400.0, 1920, 1080);
        assert!(y2 < y1, "expected drag up, got {y1} -> {y2}");

        // Target far above (-y screen): cursor must drag up -> down (y2 > y1).
        let (_, y1, _, y2) = compute_bigmap_drag(960.0, -1200.0, 1920, 1080);
        assert!(y2 > y1, "expected drag down, got {y1} -> {y2}");
    }

    #[test]
    fn country_switch_triggers_only_when_closer_across_regions() {
        // Nearby inside Liyue -> stay put.
        assert!(should_switch_country(Some((270.0, -666.0)), 450.0, -800.0).is_none());

        // From Liyue to Inazuma -> switch to 稻妻.
        let picked = should_switch_country(Some((270.0, -666.0)), -4350.0, -3010.0).unwrap();
        assert_eq!(picked.name, "稻妻");

        // Traditional OCR variant normalization works cleanly.
        assert_eq!(normalize_region_ocr_text(" 須 彌 "), "须弥");
        assert_eq!(normalize_region_ocr_text("楓丹"), "枫丹");
    }

    #[test]
    fn bigmap_zoom_estimation_and_prediction_match_1080p_scale() {
        let zoomed_out = BigMapProjection {
            game_tl_x: 9514.3 + 2354.8,
            game_tl_y: 4664.8 + 1327.6,
            game_br_x: 9514.3 - 2354.8,
            game_br_y: 4664.8 - 1327.6,
            frame_width: 1920,
            frame_height: 1080,
        };
        let z = estimate_bigmap_zoom_level(&zoomed_out);
        assert!((z - 5.8).abs() < 0.1, "expected zoom ~5.8, got {z}");

        // Edge click (944, 275) at zoom 5.8 would fly out of safe zone when zooming to 2.4.
        let (px, py) = predict_post_zoom_click(944.0, 275.0, 1920, 1080, z, 2.4);
        assert!(!is_bigmap_click_in_safe_zone(px, py, 1920, 1080));

        // Centered click stays in safe zone when zooming in.
        let (cx, cy) = predict_post_zoom_click(944.0, 540.0, 1920, 1080, z, 2.4);
        assert!(is_bigmap_click_in_safe_zone(cx, cy, 1920, 1080));
    }
}
