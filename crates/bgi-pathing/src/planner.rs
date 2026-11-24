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

/// Computes target counterclockwise heading in degrees `[0, 360)` matching `Navigation.GetTargetOrientation`.
pub fn target_orientation_deg(cur_x: f64, cur_y: f64, target_x: f64, target_y: f64) -> f32 {
    let dx = target_x - cur_x;
    let dy = target_y - cur_y;
    if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
        return 0.0;
    }
    (dy.atan2(dx).to_degrees().rem_euclid(360.0)) as f32
}

/// Signed shortest delta `(cur - target)` in `[-180, 180)`, matching `CameraRotateTask.cs`.
pub fn signed_angle_diff_deg(current_deg: f32, target_deg: f32) -> f32 {
    let raw = (current_deg - target_deg + 540.0).rem_euclid(360.0) - 180.0;
    raw
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
pub fn project_bigmap_click(proj: &BigMapProjection, target_x: f64, target_y: f64) -> Option<(f32, f32)> {
    let span_x = (proj.game_br_x - proj.game_tl_x) as f64;
    let span_y = (proj.game_br_y - proj.game_tl_y) as f64;
    if span_x.abs() < 1.0 || span_y.abs() < 1.0 || proj.frame_width == 0 || proj.frame_height == 0 {
        return None;
    }
    let u = (target_x - proj.game_tl_x as f64) / span_x;
    let v = (target_y - proj.game_tl_y as f64) / span_y;
    Some(((u * proj.frame_width as f64) as f32, (v * proj.frame_height as f64) as f32))
}

pub fn is_bigmap_click_in_safe_zone(x: f32, y: f32, width: u32, height: u32) -> bool {
    let w = width as f32;
    let h = height as f32;
    x >= w * 0.18 && x <= w * 0.72 && y >= h * 0.14 && y <= h * 0.84
}

/// Returns `(x1, y1, x2, y2)` client drag coordinates to pull `(click_x, click_y)` toward screen center.
pub fn compute_bigmap_drag(click_x: f32, click_y: f32, width: u32, height: u32) -> (i32, i32, i32, i32) {
    let w = width as f32;
    let h = height as f32;
    let cx = w * 0.48;
    let cy = h * 0.50;
    let dx = cx - click_x;
    let dy = cy - click_y;
    let dist = (dx * dx + dy * dy).sqrt().max(1.0);
    let max_drag = w.min(h) * 0.42;
    let scale = (max_drag / dist).min(1.0);
    let step_x = dx * scale;
    let step_y = dy * scale;
    let x1 = (cx - step_x * 0.5).clamp(w * 0.15, w * 0.85).round() as i32;
    let y1 = (cy - step_y * 0.5).clamp(h * 0.15, h * 0.85).round() as i32;
    let x2 = (cx + step_x * 0.5).clamp(w * 0.15, w * 0.85).round() as i32;
    let y2 = (cy + step_y * 0.5).clamp(h * 0.15, h * 0.85).round() as i32;
    (x1, y1, x2, y2)
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
        assert!((target_orientation_deg(0.0, 0.0, 10.0, 0.0) - 0.0).abs() < 1e-3);
        assert!((target_orientation_deg(0.0, 0.0, 0.0, 10.0) - 90.0).abs() < 1e-3);
        assert!((target_orientation_deg(0.0, 0.0, -10.0, 0.0) - 180.0).abs() < 1e-3);
        assert!((target_orientation_deg(0.0, 0.0, 0.0, -10.0) - 270.0).abs() < 1e-3);

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
}
