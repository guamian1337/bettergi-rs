use std::time::{Duration, Instant};

use bgi_control::LogLevel;

use crate::PathingTaskFile;
use crate::planner::{
    BigMapProjection, PlannedWaypoint, StuckMonitor, compute_bigmap_drag, distance_2d,
    is_bigmap_click_in_safe_zone, project_bigmap_click, signed_angle_diff_deg,
    split_segments_by_teleport, target_orientation_deg, turn_mouse_dx,
};

pub const VK_W: u16 = 0x57;
pub const VK_A: u16 = 0x41;
pub const VK_D: u16 = 0x44;
pub const VK_F: u16 = 0x46;
pub const VK_M: u16 = 0x4D;
pub const VK_SPACE: u16 = 0x20;
pub const VK_LSHIFT: u16 = 0xA0;
pub const VK_ESCAPE: u16 = 0x1B;

#[derive(Debug, Clone, Copy)]
pub struct MinimapObservation {
    pub game_x: f32,
    pub game_y: f32,
    pub confidence: f64,
    pub camera_angle: f32,
    pub camera_confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct BigMapObservation {
    pub projection: BigMapProjection,
    pub inliers: usize,
}

#[derive(Debug, Clone)]
pub struct OcrHit {
    pub text: String,
    pub center_x: i32,
    pub center_y: i32,
}

pub trait PathingHost {
    fn is_cancelled(&self) -> bool;
    fn dpi_scale(&self) -> f32;
    fn focus_game_window(&mut self);
    fn capture_minimap(&mut self, prev_pos: Option<(f32, f32)>) -> Result<Option<MinimapObservation>, String>;
    fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String>;
    fn capture_ocr_roi(
        &mut self,
        norm_x0: f32,
        norm_y0: f32,
        norm_x1: f32,
        norm_y1: f32,
    ) -> Result<Vec<OcrHit>, String>;
    fn set_minimap_prev_pos(&mut self, x: f32, y: f32);
    fn reset_minimap_prev_pos(&mut self);
    fn sleep_ms(&mut self, ms: u64);
    fn key_down(&mut self, vk: u16);
    fn key_up(&mut self, vk: u16);
    fn key_tap(&mut self, vk: u16, hold_ms: u64);
    fn mouse_move_rel(&mut self, dx: i32, dy: i32);
    fn mouse_left_click(&mut self);
    fn mouse_left_click_client(&mut self, x: i32, y: i32);
    fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32);
    fn mouse_wheel(&mut self, notches: i32);
    fn release_all_keys(&mut self);
    fn on_progress(&mut self, current_index: usize, total_points: usize);
    fn on_log(&mut self, level: LogLevel, message: String);
}

struct KeyReleaseGuard<'a, H: PathingHost + ?Sized> {
    host: &'a mut H,
}

impl<'a, H: PathingHost + ?Sized> KeyReleaseGuard<'a, H> {
    fn new(host: &'a mut H) -> Self {
        Self { host }
    }
}

impl<H: PathingHost + ?Sized> Drop for KeyReleaseGuard<'_, H> {
    fn drop(&mut self) {
        self.host.release_all_keys();
    }
}

pub struct TpTask;

impl TpTask {
    pub fn teleport<H: PathingHost + ?Sized>(
        host: &mut H,
        tp_x: f64,
        tp_y: f64,
    ) -> Result<(), String> {
        host.on_log(
            LogLevel::Info,
            format!("TpTask: 开始传送至 ({tp_x:.1}, {tp_y:.1})"),
        );
        host.focus_game_window();

        let mut bigmap = Self::ensure_bigmap_open(host)?;

        let mut selected = false;
        for step in 0..18 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let Some((cx, cy)) = project_bigmap_click(&bigmap.projection, tp_x, tp_y) else {
                return Err("大地图投影无效".to_string());
            };
            let w = bigmap.projection.frame_width;
            let h = bigmap.projection.frame_height;
            if is_bigmap_click_in_safe_zone(cx, cy, w, h) {
                host.on_log(
                    LogLevel::Info,
                    format!("TpTask: 目标点位于屏幕 ({cx:.0}, {cy:.0})，点击选中 (第 {} 步)", step + 1),
                );
                host.mouse_left_click_client(cx.round() as i32, cy.round() as i32);
                host.sleep_ms(450);
                selected = true;
                break;
            }

            let (x1, y1, x2, y2) = compute_bigmap_drag(cx, cy, w, h);
            host.mouse_drag_client(x1, y1, x2, y2);
            host.sleep_ms(180);

            bigmap = match host.capture_bigmap()? {
                Some(bm) if bm.inliers >= 12 => bm,
                _ => {
                    host.mouse_wheel(-5);
                    host.sleep_ms(220);
                    host.capture_bigmap()?.ok_or_else(|| "拖拽大地图后 SIFT 定位失效".to_string())?
                }
            };
        }

        if !selected {
            host.key_tap(VK_ESCAPE, 40);
            return Err(format!("拖拽超过上限仍未接近传送点 ({tp_x:.1}, {tp_y:.1})"));
        }

        Self::confirm_teleport_panel(host)?;

        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(25) {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            if let Ok(Some(obs)) = host.capture_minimap(Some((tp_x as f32, tp_y as f32)))
                && obs.confidence >= 0.72
            {
                host.set_minimap_prev_pos(tp_x as f32, tp_y as f32);
                host.sleep_ms(600);
                host.on_log(LogLevel::Info, format!("TpTask: 传送完成 ({tp_x:.1}, {tp_y:.1})"));
                return Ok(());
            }
            host.sleep_ms(350);
        }

        Err("TpTask: 等待传送加载回到大世界超时 (25s)".to_string())
    }

    fn ensure_bigmap_open<H: PathingHost + ?Sized>(host: &mut H) -> Result<BigMapObservation, String> {
        for attempt in 0..3 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            if let Ok(Some(bm)) = host.capture_bigmap()
                && bm.inliers >= 12
            {
                return Ok(bm);
            }
            if attempt == 0 {
                host.key_tap(VK_M, 50);
            } else {
                host.mouse_wheel(-5);
            }
            host.sleep_ms(850);
        }
        Err("TpTask: 打开大地图后仍无法通过 SIFT 定位当前视图".to_string())
    }

    fn confirm_teleport_panel<H: PathingHost + ?Sized>(host: &mut H) -> Result<(), String> {
        for attempt in 0..3 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let hits = host.capture_ocr_roi(0.55, 0.25, 0.99, 0.98)?;
            if let Some(tp_btn) = hits
                .iter()
                .find(|h| h.text.contains("传送") && !h.text.contains("锚点"))
                .or_else(|| hits.iter().find(|h| h.text.eq_ignore_ascii_case("Teleport")))
            {
                host.mouse_left_click_client(tp_btn.center_x, tp_btn.center_y);
                host.sleep_ms(200);
                return Ok(());
            }

            if attempt < 2
                && let Some(cand) = hits.iter().find(|h| {
                    h.text.contains("传送锚点") || h.text.contains("七天神像") || h.text.contains("秘境")
                })
            {
                host.on_log(
                    LogLevel::Info,
                    format!("TpTask: 命中重叠候选「{}」，尝试展开", cand.text),
                );
                host.mouse_left_click_client(cand.center_x, cand.center_y);
                host.sleep_ms(350);
                continue;
            }
            host.sleep_ms(250);
        }

        host.key_tap(VK_ESCAPE, 40);
        Err("TpTask: 点击地图点位后未识别到「传送」确认按钮".to_string())
    }
}

pub struct PathExecutor;

impl PathExecutor {
    pub fn run<H: PathingHost + ?Sized>(
        host: &mut H,
        task_name: &str,
        task: &PathingTaskFile,
    ) -> Result<(), String> {
        let total = task.positions.len();
        if total == 0 {
            return Err("路径点位为空".to_string());
        }

        host.on_log(
            LogLevel::Info,
            format!("PathExecutor: 开始执行「{task_name}」，共 {total} 个点位"),
        );
        host.focus_game_window();
        host.reset_minimap_prev_pos();

        let guard = KeyReleaseGuard::new(host);
        let segments = split_segments_by_teleport(&task.positions);

        for segment in segments {
            if let Some(first) = segment.first()
                && first.waypoint.waypoint_type != "teleport"
            {
                guard.host.set_minimap_prev_pos(first.waypoint.x as f32, first.waypoint.y as f32);
            }

            for item in segment {
                if guard.host.is_cancelled() {
                    return Err("已手动停止寻路".to_string());
                }
                guard.host.on_progress(item.index + 1, total);
                Self::execute_waypoint(guard.host, &item, total)?;
            }
        }

        guard.host.on_log(LogLevel::Info, format!("PathExecutor:「{task_name}」全部执行完成"));
        Ok(())
    }

    fn execute_waypoint<H: PathingHost + ?Sized>(
        host: &mut H,
        item: &PlannedWaypoint,
        total: usize,
    ) -> Result<(), String> {
        let wp = &item.waypoint;
        let idx1 = item.index + 1;
        if wp.waypoint_type == "teleport" {
            host.on_log(
                LogLevel::Info,
                format!("[{idx1}/{total}] 传送点 ({:.1}, {:.1})", wp.x, wp.y),
            );
            return TpTask::teleport(host, wp.x, wp.y);
        }

        if wp.waypoint_type == "orientation" {
            if let Some(obs) = Self::locate_minimap_retry(host, None, 5)? {
                let target_angle = target_orientation_deg(obs.game_x as f64, obs.game_y as f64, wp.x, wp.y);
                Self::wait_until_rotated_to(host, target_angle, 3.0, 25)?;
            }
            return Ok(());
        }

        if !item.basic_action_supported
            && let Some(act) = wp.action.as_deref()
        {
            host.on_log(
                LogLevel::Warn,
                format!("[{idx1}/{total}] 复杂动作「{act}」暂未实现，按仅跑图经过"),
            );
        } else if wp.action.as_deref() == Some("log_output")
            && let Some(msg) = wp.action_params.as_deref()
        {
            host.on_log(LogLevel::Info, format!("[{idx1}/{total}] {msg}"));
        }

        Self::move_to(host, item, total)?;

        if wp.action.as_deref() == Some("stop_flying") {
            host.on_log(LogLevel::Info, format!("[{idx1}/{total}] 下落攻击 (stop_flying)"));
            host.mouse_left_click();
            host.sleep_ms(850);
        }

        if wp.waypoint_type == "target" {
            Self::move_close_to(host, wp.x, wp.y)?;
            if item.basic_action_supported {
                host.on_log(LogLevel::Info, format!("[{idx1}/{total}] 目标点交互 (按 F)"));
                for _ in 0..3 {
                    if host.is_cancelled() {
                        break;
                    }
                    host.key_tap(VK_F, 40);
                    host.sleep_ms(160);
                }
            }
        }
        Ok(())
    }

    fn locate_minimap_retry<H: PathingHost + ?Sized>(
        host: &mut H,
        prev: Option<(f32, f32)>,
        retries: usize,
    ) -> Result<Option<MinimapObservation>, String> {
        for _ in 0..retries {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            if let Some(obs) = host.capture_minimap(prev)? {
                return Ok(Some(obs));
            }
            host.sleep_ms(120);
        }
        Ok(None)
    }

    fn wait_until_rotated_to<H: PathingHost + ?Sized>(
        host: &mut H,
        target_angle: f32,
        max_diff: f32,
        max_tries: usize,
    ) -> Result<(), String> {
        let dpi = host.dpi_scale();
        for _ in 0..max_tries {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let Some(obs) = host.capture_minimap(None)? else {
                host.sleep_ms(50);
                continue;
            };
            let diff = signed_angle_diff_deg(obs.camera_angle, target_angle);
            if diff.abs() < max_diff {
                return Ok(());
            }
            let dx = turn_mouse_dx(diff, dpi);
            if dx != 0 {
                host.mouse_move_rel(dx, 0);
            }
            host.sleep_ms(45);
        }
        Ok(())
    }

    fn move_to<H: PathingHost + ?Sized>(
        host: &mut H,
        item: &PlannedWaypoint,
        total: usize,
    ) -> Result<(), String> {
        let wp = &item.waypoint;
        let idx1 = item.index + 1;
        let initial = Self::locate_minimap_retry(host, Some((wp.x as f32, wp.y as f32)), 6)?
            .ok_or_else(|| format!("[{idx1}/{total}] 小地图初始定位失败"))?;

        let init_dist = distance_2d(initial.game_x as f64, initial.game_y as f64, wp.x, wp.y);
        let init_angle = target_orientation_deg(initial.game_x as f64, initial.game_y as f64, wp.x, wp.y);
        host.on_log(
            LogLevel::Info,
            format!(
                "[{idx1}/{total}] 前往 ({:.1}, {:.1}), 模式={}, 初始距离={init_dist:.1}",
                wp.x, wp.y, wp.move_mode
            ),
        );
        Self::wait_until_rotated_to(host, init_angle, 5.0, 30)?;

        let dpi = host.dpi_scale();
        let arrive_dist = if wp.waypoint_type == "target" { 4.0 } else { 4.5 };
        let start = Instant::now();
        let mut last_dash = Instant::now() - Duration::from_secs(2);
        let mut last_jump = Instant::now();
        let mut sprint_held = false;
        let mut miss_count = 0usize;
        let mut stuck = StuckMonitor::new(Duration::from_millis(2200), 2.5);
        stuck.reset(Some((initial.game_x as f64, initial.game_y as f64)));
        let mut stuck_recoveries = 0usize;
        let mut angle_offset = 0.0f32;

        host.key_down(VK_W);

        loop {
            if host.is_cancelled() {
                host.key_up(VK_W);
                if sprint_held {
                    host.key_up(VK_LSHIFT);
                }
                return Err("已取消".to_string());
            }
            if start.elapsed() > Duration::from_secs(180) {
                host.key_up(VK_W);
                if sprint_held {
                    host.key_up(VK_LSHIFT);
                }
                return Err(format!("[{idx1}/{total}] 移动超时 (180s)"));
            }

            let obs_opt = host.capture_minimap(None)?;
            let Some(obs) = obs_opt else {
                miss_count += 1;
                if miss_count > 24 {
                    host.key_up(VK_W);
                    if sprint_held {
                        host.key_up(VK_LSHIFT);
                    }
                    return Err(format!("[{idx1}/{total}] 连续 24 帧小地图丢失"));
                }
                host.sleep_ms(75);
                continue;
            };
            miss_count = 0;

            let dist = distance_2d(obs.game_x as f64, obs.game_y as f64, wp.x, wp.y);
            if dist <= arrive_dist {
                break;
            }
            if dist > 600.0 {
                host.key_up(VK_W);
                if sprint_held {
                    host.key_up(VK_LSHIFT);
                }
                return Err(format!("[{idx1}/{total}] 距离异常过远 ({dist:.1})"));
            }

            if stuck.observe((obs.game_x as f64, obs.game_y as f64)) {
                stuck_recoveries += 1;
                if stuck_recoveries > 4 {
                    host.on_log(
                        LogLevel::Warn,
                        format!("[{idx1}/{total}] 脱困次数超限，跳过继续"),
                    );
                    break;
                }
                host.on_log(
                    LogLevel::Warn,
                    format!("[{idx1}/{total}] 疑似卡住，执行脱困 #{stuck_recoveries}"),
                );
                host.key_tap(VK_SPACE, 50);
                let strafe = if stuck_recoveries % 2 == 1 { VK_A } else { VK_D };
                host.key_down(strafe);
                host.sleep_ms(280);
                host.key_up(strafe);
                angle_offset = if stuck_recoveries % 2 == 1 { 35.0 } else { -35.0 };
            } else if angle_offset.abs() > 0.1 && dist > 12.0 {
                angle_offset *= 0.85;
            } else {
                angle_offset = 0.0;
            }

            let target_angle = (target_orientation_deg(obs.game_x as f64, obs.game_y as f64, wp.x, wp.y)
                + angle_offset)
                .rem_euclid(360.0);
            let diff = signed_angle_diff_deg(obs.camera_angle, target_angle);
            if diff.abs() > 38.0 {
                host.key_up(VK_W);
                Self::wait_until_rotated_to(host, target_angle, 5.0, 20)?;
                host.key_down(VK_W);
            } else {
                let dx = turn_mouse_dx(diff, dpi);
                if dx != 0 {
                    host.mouse_move_rel(dx, 0);
                }
            }

            match wp.move_mode.as_str() {
                "run" => {
                    let want = dist > 18.0;
                    if want && !sprint_held {
                        host.key_down(VK_LSHIFT);
                        sprint_held = true;
                    } else if !want && sprint_held {
                        host.key_up(VK_LSHIFT);
                        sprint_held = false;
                    }
                }
                "dash" => {
                    if dist > 18.0 && last_dash.elapsed() >= Duration::from_millis(1100) {
                        last_dash = Instant::now();
                        host.key_tap(VK_LSHIFT, 35);
                    }
                }
                "jump" => {
                    if last_jump.elapsed() >= Duration::from_millis(480) {
                        last_jump = Instant::now();
                        host.key_tap(VK_SPACE, 35);
                    }
                }
                _ => {}
            }

            host.sleep_ms(75);
        }

        host.key_up(VK_W);
        if sprint_held {
            host.key_up(VK_LSHIFT);
        }
        Ok(())
    }

    fn move_close_to<H: PathingHost + ?Sized>(
        host: &mut H,
        target_x: f64,
        target_y: f64,
    ) -> Result<(), String> {
        for _ in 0..16 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let Some(obs) = host.capture_minimap(None)? else {
                host.sleep_ms(60);
                continue;
            };
            let dist = distance_2d(obs.game_x as f64, obs.game_y as f64, target_x, target_y);
            if dist < 2.0 {
                break;
            }
            let target_angle = target_orientation_deg(obs.game_x as f64, obs.game_y as f64, target_x, target_y);
            Self::wait_until_rotated_to(host, target_angle, 4.0, 12)?;
            host.key_down(VK_W);
            host.sleep_ms(65);
            host.key_up(VK_W);
            host.sleep_ms(25);
        }
        host.key_up(VK_W);
        host.sleep_ms(200);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PathingInfo, Waypoint};

    struct MockHost {
        pos: (f32, f32),
        angle: f32,
        events: Vec<String>,
        progress: Vec<(usize, usize)>,
        bigmap_open: bool,
        clicked_tp_btn: bool,
        w_down: bool,
    }

    impl MockHost {
        fn new(start: (f32, f32)) -> Self {
            Self {
                pos: start,
                angle: 0.0,
                events: Vec::new(),
                progress: Vec::new(),
                bigmap_open: false,
                clicked_tp_btn: false,
                w_down: false,
            }
        }
    }

    impl PathingHost for MockHost {
        fn is_cancelled(&self) -> bool {
            false
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {
            self.events.push("focus".into());
        }
        fn capture_minimap(&mut self, prev: Option<(f32, f32)>) -> Result<Option<MinimapObservation>, String> {
            if self.clicked_tp_btn {
                self.pos = prev.unwrap_or(self.pos);
                self.clicked_tp_btn = false;
                self.bigmap_open = false;
            }
            Ok(Some(MinimapObservation {
                game_x: self.pos.0,
                game_y: self.pos.1,
                confidence: 0.99,
                camera_angle: self.angle,
                camera_confidence: 0.9,
            }))
        }
        fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
            if !self.bigmap_open {
                return Ok(None);
            }
            Ok(Some(BigMapObservation {
                projection: BigMapProjection {
                    game_tl_x: 100.0,
                    game_tl_y: 100.0,
                    game_br_x: -100.0,
                    game_br_y: -100.0,
                    frame_width: 1920,
                    frame_height: 1080,
                },
                inliers: 40,
            }))
        }
        fn capture_ocr_roi(&mut self, _: f32, _: f32, _: f32, _: f32) -> Result<Vec<OcrHit>, String> {
            Ok(vec![OcrHit {
                text: "传送".into(),
                center_x: 1600,
                center_y: 960,
            }])
        }
        fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
            self.pos = (x, y);
        }
        fn reset_minimap_prev_pos(&mut self) {}
        fn sleep_ms(&mut self, _: u64) {
            if self.w_down {
                let rad = self.angle.to_radians();
                self.pos.0 += rad.cos() * 2.5;
                self.pos.1 += rad.sin() * 2.5;
            }
        }
        fn key_down(&mut self, vk: u16) {
            if vk == VK_W {
                self.w_down = true;
            }
        }
        fn key_up(&mut self, vk: u16) {
            if vk == VK_W {
                self.w_down = false;
            }
        }
        fn key_tap(&mut self, vk: u16, _: u64) {
            if vk == VK_M {
                self.bigmap_open = true;
            } else if vk == VK_F {
                self.events.push("press_F".into());
            }
        }
        fn mouse_move_rel(&mut self, dx: i32, _: i32) {
            self.angle = (self.angle - dx as f32).rem_euclid(360.0);
        }
        fn mouse_left_click(&mut self) {}
        fn mouse_left_click_client(&mut self, x: i32, y: i32) {
            self.events.push(format!("click({x},{y})"));
            if x > 1400 && y > 800 {
                self.clicked_tp_btn = true;
            }
        }
        fn mouse_drag_client(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn mouse_wheel(&mut self, _: i32) {}
        fn release_all_keys(&mut self) {
            self.w_down = false;
            self.events.push("release_all".into());
        }
        fn on_progress(&mut self, current_index: usize, total_points: usize) {
            self.progress.push((current_index, total_points));
        }
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
    }

    #[test]
    fn mock_path_executor_teleports_walks_and_presses_f() {
        let task = PathingTaskFile {
            info: PathingInfo {
                name: "MockRoute".into(),
                ..Default::default()
            },
            positions: vec![
                Waypoint {
                    x: 10.0,
                    y: 20.0,
                    waypoint_type: "teleport".into(),
                    move_mode: "walk".into(),
                    action: None,
                    action_params: None,
                },
                Waypoint {
                    x: 20.0,
                    y: 20.0,
                    waypoint_type: "target".into(),
                    move_mode: "walk".into(),
                    action: None,
                    action_params: None,
                },
            ],
        };

        let mut host = MockHost::new((-500.0, -500.0));
        PathExecutor::run(&mut host, "MockRoute", &task).unwrap();
        assert_eq!(host.progress, vec![(1, 2), (2, 2)]);
        assert!(host.events.iter().any(|e| e == "press_F"));
        assert_eq!(host.events.last().map(String::as_str), Some("release_all"));
    }
}
