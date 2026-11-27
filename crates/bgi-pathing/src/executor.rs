use std::time::{Duration, Instant};

use bgi_control::LogLevel;
use serde::{Deserialize, Serialize};

use crate::planner::{
    BigMapProjection, CountryAnchor, PlannedWaypoint, StuckMonitor, compute_bigmap_drag,
    distance_2d, estimate_bigmap_zoom_level, is_bigmap_click_in_safe_zone,
    normalize_region_ocr_text, predict_post_zoom_click, project_bigmap_click,
    should_switch_country, signed_angle_diff_deg, split_segments_by_teleport,
    target_orientation_deg, turn_mouse_dx,
};
use crate::{PathingTaskFile, Waypoint};

const BIGMAP_CLICK_MAX_ZOOM: f32 = 3.4;
const BIGMAP_CLICK_TARGET_ZOOM: f32 = 2.4;
const BIGMAP_ZOOM_PER_NOTCH: f32 = 0.085;

pub const VK_W: u16 = 0x57;
pub const VK_A: u16 = 0x41;
pub const VK_D: u16 = 0x44;
pub const VK_F: u16 = 0x46;
pub const VK_M: u16 = 0x4D;
pub const VK_Z: u16 = 0x5A;
pub const VK_SPACE: u16 = 0x20;
pub const VK_LSHIFT: u16 = 0xA0;
pub const VK_ESCAPE: u16 = 0x1B;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionStatus {
    #[default]
    Normal,
    Fly,
    Climb,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BigMapLayerButtons {
    /// Center `(x, y)` of active cyan layered-map toggle icon (`MapUndergroundSwitchButton`).
    pub underground_switch: Option<(i32, i32)>,
    /// Center `(x, y)` of ground/surface layer button (`MapUndergroundToGroundButton`).
    pub to_ground: Option<(i32, i32)>,
}

#[derive(Debug, Clone, Copy)]
pub struct MinimapObservation {
    pub game_x: f32,
    pub game_y: f32,
    pub confidence: f64,
    pub camera_angle: f32,
    pub camera_confidence: f32,
    pub motion: MotionStatus,
}

#[derive(Debug, Clone, Copy)]
pub struct BigMapObservation {
    pub projection: BigMapProjection,
    pub inliers: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CookingObservation {
    pub color_count: u32,
    pub scale: f64,
    pub page_visible: bool,
    pub white_confirm: Option<(i32, i32)>,
    pub recover_visible: bool,
}

#[derive(Debug, Clone)]
pub struct OcrHit {
    pub text: String,
    pub center_x: i32,
    pub center_y: i32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathingCheckpoint {
    pub path_fingerprint: u64,
    pub current_index: usize,
}

impl PathingCheckpoint {
    pub fn for_task(task: &PathingTaskFile, current_index: usize) -> Result<Self, String> {
        Self::for_positions(&task.positions, current_index)
    }

    pub fn for_positions(positions: &[Waypoint], current_index: usize) -> Result<Self, String> {
        if current_index >= positions.len() {
            return Err("恢复点位超出路径范围".into());
        }
        let bytes = serde_json::to_vec(positions).map_err(|error| error.to_string())?;
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(Self {
            path_fingerprint: hash,
            current_index,
        })
    }
}

pub trait PathingHost {
    fn is_cancelled(&self) -> bool;
    fn current_waypoint_index(&self) -> Option<usize> {
        None
    }
    fn detect_loading(&mut self) -> Result<bool, String> {
        Err("加载检测未实现".into())
    }
    fn dpi_scale(&self) -> f32;
    fn focus_game_window(&mut self);
    fn capture_minimap(
        &mut self,
        prev_pos: Option<(f32, f32)>,
    ) -> Result<Option<MinimapObservation>, String>;
    fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String>;
    fn capture_ocr_roi(
        &mut self,
        norm_x0: f32,
        norm_y0: f32,
        norm_x1: f32,
        norm_y1: f32,
    ) -> Result<Vec<OcrHit>, String>;
    fn capture_auto_pick(&mut self) -> Result<Option<crate::auto_pick::AutoPickFrame>, String> {
        Err("自动拾取正例识别未实现".into())
    }
    fn key_tap_checked(
        &mut self,
        _vk: u16,
        _hold_ms: u64,
        _frame_size: (u32, u32),
        _observed_at: Instant,
    ) -> Result<(), String> {
        Err("自动拾取安全输入未实现".into())
    }
    fn set_minimap_prev_pos(&mut self, x: f32, y: f32);
    fn reset_minimap_prev_pos(&mut self);
    fn sleep_ms(&mut self, ms: u64);
    fn key_down(&mut self, vk: u16);
    fn key_up(&mut self, vk: u16);
    fn key_tap(&mut self, vk: u16, hold_ms: u64);
    fn mouse_move_rel(&mut self, dx: i32, dy: i32);
    fn mouse_move_client(&mut self, _x: i32, _y: i32) {}
    fn mouse_left_click(&mut self);
    fn mouse_right_click(&mut self) {}
    fn mouse_right_down(&mut self) {}
    fn mouse_right_up(&mut self) {}
    fn mouse_left_click_client(&mut self, x: i32, y: i32);
    fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32);
    fn mouse_wheel(&mut self, notches: i32);
    fn detect_bigmap_layer_buttons(&mut self) -> Result<BigMapLayerButtons, String> {
        Ok(BigMapLayerButtons::default())
    }
    fn detect_motion_status(&mut self) -> Result<MotionStatus, String> {
        Ok(MotionStatus::Normal)
    }
    fn capture_cooking_bar(&mut self) -> Result<Option<CookingObservation>, String> {
        Err("烹饪条识别未实现".into())
    }
    /// 莉奈娅挖矿动作：参数形如 `"3"`、`"3,10"`、`"mines=3,rounds=10"`。
    fn run_linnea_mining(&mut self, _params: Option<&str>) -> Result<(), String> {
        Err("莉奈娅挖矿动作需要采矿运行时，当前 host 未接入".into())
    }
    fn release_all_keys(&mut self);
    fn on_progress(&mut self, current_index: usize, total_points: usize);
    fn on_checkpoint(&mut self, _checkpoint: &PathingCheckpoint) {}
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

        let bigmap = Self::ensure_bigmap_open(host)?;
        if let Err(err) = Self::select_and_confirm_teleport(host, bigmap, tp_x, tp_y) {
            host.key_tap(VK_ESCAPE, 40);
            host.sleep_ms(300);
            return Err(err);
        }

        // Wait briefly for the teleport transition/loading screen to begin so we do not
        // mistake the pre-teleport world or early fade frame for arrival.
        host.sleep_ms(900);
        host.reset_minimap_prev_pos();

        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(25) {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            if let Ok(Some(obs)) = host.capture_minimap(Some((tp_x as f32, tp_y as f32))) {
                let arrived_dist = distance_2d(obs.game_x as f64, obs.game_y as f64, tp_x, tp_y);
                if obs.confidence >= 0.72 && arrived_dist <= 160.0 {
                    host.set_minimap_prev_pos(obs.game_x, obs.game_y);
                    host.sleep_ms(600);
                    host.on_log(
                        LogLevel::Info,
                        format!(
                            "TpTask: 传送完成 pos=({:.1}, {:.1}) -> target=({tp_x:.1}, {tp_y:.1})",
                            obs.game_x, obs.game_y
                        ),
                    );
                    return Ok(());
                }
                host.on_log(
                    LogLevel::Debug,
                    format!(
                        "TpTask: 忽略加载期远距离匹配 pos=({:.1}, {:.1}) dist={arrived_dist:.1} conf={:.3}",
                        obs.game_x, obs.game_y, obs.confidence
                    ),
                );
                host.reset_minimap_prev_pos();
            }
            host.sleep_ms(350);
        }

        Err("TpTask: 等待传送加载回到大世界超时 (25s)".to_string())
    }

    fn zoom_bigmap_centered<H: PathingHost + ?Sized>(
        host: &mut H,
        frame_width: u32,
        frame_height: u32,
        notches: i32,
    ) {
        let cx = (frame_width.max(320) / 2) as i32;
        let cy = (frame_height.max(240) / 2) as i32;
        host.mouse_move_client(cx, cy);
        host.sleep_ms(20);
        host.mouse_wheel(notches);
    }

    fn capture_bigmap_after_zoom<H: PathingHost + ?Sized>(
        host: &mut H,
        frame_width: u32,
        frame_height: u32,
        notches: i32,
        delay_ms: u64,
    ) -> Result<Option<BigMapObservation>, String> {
        Self::zoom_bigmap_centered(host, frame_width, frame_height, notches);
        host.sleep_ms(delay_ms);
        host.capture_bigmap()
    }

    fn select_and_confirm_teleport<H: PathingHost + ?Sized>(
        host: &mut H,
        mut bigmap: BigMapObservation,
        tp_x: f64,
        tp_y: f64,
    ) -> Result<(), String> {
        let cur_x = ((bigmap.projection.game_tl_x + bigmap.projection.game_br_x) * 0.5) as f64;
        let cur_y = ((bigmap.projection.game_tl_y + bigmap.projection.game_br_y) * 0.5) as f64;
        if let Some(anchor) = should_switch_country(Some((cur_x, cur_y)), tp_x, tp_y) {
            let w = bigmap.projection.frame_width;
            let h = bigmap.projection.frame_height;
            host.on_log(
                LogLevel::Info,
                format!(
                    "TpTask: 当前中心 ({cur_x:.0}, {cur_y:.0}) 距目标 ({tp_x:.0}, {tp_y:.0}) 较远，尝试切换地区 -> {}",
                    anchor.name
                ),
            );
            if Self::switch_country_map(host, anchor, w, h)? {
                bigmap = match host.capture_bigmap()? {
                    Some(bm) if bm.inliers >= 12 => bm,
                    _ => match Self::capture_bigmap_after_zoom(host, w, h, -5, 220)? {
                        Some(bm) if bm.inliers >= 12 => bm,
                        _ => match Self::capture_bigmap_after_zoom(host, w, h, 3, 220)? {
                            Some(bm) if bm.inliers >= 12 => bm,
                            _ => {
                                return Err(format!(
                                    "切换地区 [{}] 后大地图 SIFT 定位失败，请检查 Teyvat 地图资源版本",
                                    anchor.name
                                ));
                            }
                        },
                    },
                };
            }
        }

        let mut selected = false;
        let mut zoom_outs = 0usize;
        let mut zoom_ins = 0usize;
        let mut stall_count = 0usize;
        let mut last_center: Option<(f64, f64)> = None;

        for step in 0..22 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let Some((cx, cy)) = project_bigmap_click(&bigmap.projection, tp_x, tp_y) else {
                return Err("大地图投影无效".to_string());
            };
            let w = bigmap.projection.frame_width;
            let h = bigmap.projection.frame_height;
            let est_zoom = estimate_bigmap_zoom_level(&bigmap.projection);

            let in_safe_zone = is_bigmap_click_in_safe_zone(cx, cy, w, h);
            if in_safe_zone {
                if est_zoom > BIGMAP_CLICK_MAX_ZOOM && zoom_ins < 4 {
                    let (pred_x, pred_y) =
                        predict_post_zoom_click(cx, cy, w, h, est_zoom, BIGMAP_CLICK_TARGET_ZOOM);
                    if is_bigmap_click_in_safe_zone(pred_x, pred_y, w, h) {
                        let notches = ((est_zoom - BIGMAP_CLICK_TARGET_ZOOM)
                            / BIGMAP_ZOOM_PER_NOTCH)
                            .round()
                            .clamp(2.0, 12.0) as i32;
                        host.on_log(
                            LogLevel::Info,
                            format!(
                                "TpTask: 当前大地图过小 (zoom≈{est_zoom:.2} > {BIGMAP_CLICK_MAX_ZOOM:.1})，滚轮放大 (+{notches}) 后再点击"
                            ),
                        );
                        zoom_ins += 1;
                        last_center = None;
                        bigmap = match Self::capture_bigmap_after_zoom(host, w, h, notches, 240)? {
                            Some(bm) if bm.inliers >= 12 => bm,
                            _ => {
                                zoom_ins = 4;
                                match Self::capture_bigmap_after_zoom(host, w, h, -3, 240)? {
                                    Some(bm) if bm.inliers >= 12 => bm,
                                    _ => {
                                        return Err("滚轮放大后大地图 SIFT 定位失效".to_string());
                                    }
                                }
                            }
                        };
                        continue;
                    }
                    host.on_log(
                        LogLevel::Debug,
                        format!(
                            "TpTask: 目标 ({cx:.0}, {cy:.0}) 放大后可能出界 ({pred_x:.0}, {pred_y:.0})，先向中心拖动"
                        ),
                    );
                } else {
                    host.on_log(
                        LogLevel::Info,
                        format!(
                            "TpTask: 目标点位于屏幕 ({cx:.0}, {cy:.0}, zoom≈{est_zoom:.2})，点击选中 (第 {} 步)",
                            step + 1
                        ),
                    );
                    host.mouse_left_click_client(cx.round() as i32, cy.round() as i32);
                    host.sleep_ms(450);
                    selected = true;
                    break;
                }
            }

            let off_dist = ((cx - w as f32 * 0.5).powi(2) + (cy - h as f32 * 0.5).powi(2)).sqrt();
            if !in_safe_zone && zoom_outs < 2 && off_dist > (w.max(h) as f32) * 1.15 {
                zoom_outs += 1;
                last_center = None;
                host.on_log(
                    LogLevel::Debug,
                    format!(
                        "TpTask: 目标偏离屏外较远 ({off_dist:.0}px)，滚轮缩小大地图 ({zoom_outs}/2)"
                    ),
                );
                if let Some(bm) = Self::capture_bigmap_after_zoom(host, w, h, -5, 220)?
                    && bm.inliers >= 12
                {
                    bigmap = bm;
                    continue;
                }
            }

            let cur_x = ((bigmap.projection.game_tl_x + bigmap.projection.game_br_x) * 0.5) as f64;
            let cur_y = ((bigmap.projection.game_tl_y + bigmap.projection.game_br_y) * 0.5) as f64;
            if let Some((px, py)) = last_center {
                let moved = distance_2d(px, py, cur_x, cur_y);
                if moved < 25.0 {
                    stall_count += 1;
                    host.on_log(
                        LogLevel::Warn,
                        format!(
                            "TpTask: 上次拖拽位移过小 ({moved:.1})，第 {stall_count} 次自动偏移拖拽区域"
                        ),
                    );
                } else {
                    stall_count = 0;
                }
            }
            last_center = Some((cur_x, cur_y));

            let (mut x1, mut y1, mut x2, mut y2) = compute_bigmap_drag(cx, cy, w, h);
            if stall_count > 0 {
                let scale = ((w as f32 / 1920.0).min(h as f32 / 1080.0)).max(0.5);
                let offsets: &[(f32, f32)] = &[
                    (180.0, 0.0),
                    (-180.0, 0.0),
                    (0.0, 140.0),
                    (0.0, -140.0),
                    (220.0, 120.0),
                    (-220.0, -120.0),
                ];
                let (ox, oy) = offsets[(stall_count - 1) % offsets.len()];
                let ox_i = (ox * scale).round() as i32;
                let oy_i = (oy * scale).round() as i32;
                let pad_x = (140.0 * scale).round() as i32;
                let pad_y = (120.0 * scale).round() as i32;
                let max_x = (w as i32) - pad_x;
                let max_y = (h as i32) - pad_y;
                if x1 + ox_i >= pad_x
                    && x1 + ox_i <= max_x
                    && x2 + ox_i >= pad_x
                    && x2 + ox_i <= max_x
                {
                    x1 += ox_i;
                    x2 += ox_i;
                }
                if y1 + oy_i >= pad_y
                    && y1 + oy_i <= max_y
                    && y2 + oy_i >= pad_y
                    && y2 + oy_i <= max_y
                {
                    y1 += oy_i;
                    y2 += oy_i;
                }
            }

            host.on_log(
                LogLevel::Info,
                format!(
                    "TpTask: 拖拽大地图 step={} zoom≈{est_zoom:.2} cur=({cur_x:.0},{cur_y:.0}) tp=({tp_x:.0},{tp_y:.0}) click=({cx:.0},{cy:.0}) drag=({x1},{y1})->({x2},{y2}) inliers={}",
                    step + 1,
                    bigmap.inliers
                ),
            );
            host.mouse_drag_client(x1, y1, x2, y2);
            host.sleep_ms(200);

            bigmap = match host.capture_bigmap()? {
                Some(bm) if bm.inliers >= 12 => bm,
                _ => Self::capture_bigmap_after_zoom(host, w, h, -5, 240)?
                    .ok_or_else(|| "拖拽大地图后 SIFT 定位失效".to_string())?,
            };
        }

        if !selected {
            return Err(format!("拖拽超过上限仍未接近传送点 ({tp_x:.1}, {tp_y:.1})"));
        }

        Self::confirm_teleport_panel(host)
    }

    fn switch_to_ground_map_layer_if_needed<H: PathingHost + ?Sized>(
        host: &mut H,
    ) -> Result<bool, String> {
        let mut clicked_switch = false;
        let mut clicked_ground = false;

        for attempt in 0..10 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let buttons = host.detect_bigmap_layer_buttons()?;
            if let Some((gx, gy)) = buttons.to_ground {
                if !clicked_ground {
                    host.on_log(
                        LogLevel::Info,
                        format!("TpTask: 检测到分层地图界面，点击切回地表图层 ({gx}, {gy})"),
                    );
                    host.mouse_left_click_client(gx, gy);
                    host.sleep_ms(260);
                    clicked_ground = true;
                    continue;
                }
                if buttons.underground_switch.is_none() && attempt >= 1 {
                    return Ok(true);
                }
            } else if let Some((sx, sy)) = buttons.underground_switch {
                if !clicked_switch {
                    host.on_log(
                        LogLevel::Info,
                        format!("TpTask: 检测到分层地图开关 ({sx}, {sy})，点击展开图层列表"),
                    );
                    host.mouse_left_click_client(sx, sy);
                    host.sleep_ms(260);
                    clicked_switch = true;
                    continue;
                }
            } else {
                return Ok(clicked_ground || clicked_switch);
            }

            host.sleep_ms(140);
        }

        Ok(clicked_ground || clicked_switch)
    }

    fn ensure_bigmap_open<H: PathingHost + ?Sized>(
        host: &mut H,
    ) -> Result<BigMapObservation, String> {
        for attempt in 0..4 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            match host.capture_bigmap() {
                Ok(Some(bm)) if bm.inliers >= 12 => {
                    let cx = (bm.projection.game_tl_x + bm.projection.game_br_x) * 0.5;
                    let cy = (bm.projection.game_tl_y + bm.projection.game_br_y) * 0.5;
                    host.on_log(
                        LogLevel::Info,
                        format!(
                            "TpTask: 大地图 SIFT 定位成功 (attempt={}, inliers={}, center=({cx:.1}, {cy:.1}))",
                            attempt + 1,
                            bm.inliers,
                        ),
                    );
                    return Ok(bm);
                }
                Ok(Some(bm)) => {
                    host.on_log(
                        LogLevel::Warn,
                        format!(
                            "TpTask: 第 {} 次大地图定位内点不足 (inliers={} < 12)，检查分层图层或调整视图",
                            attempt + 1,
                            bm.inliers
                        ),
                    );
                }
                Ok(None) => {
                    host.on_log(
                        LogLevel::Debug,
                        format!(
                            "TpTask: 第 {} 次检测大地图未匹配，尝试打开、切回地表图层或缩放地图",
                            attempt + 1
                        ),
                    );
                }
                Err(err) => return Err(format!("TpTask: 大地图识别出错: {err}")),
            }

            if Self::switch_to_ground_map_layer_if_needed(host)? {
                host.sleep_ms(320);
                continue;
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

    fn switch_country_map<H: PathingHost + ?Sized>(
        host: &mut H,
        anchor: &CountryAnchor,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<bool, String> {
        let w = frame_width.max(640) as f32;
        let h = frame_height.max(360) as f32;
        let scale = (w / 1920.0).min(h / 1080.0);

        // Mirrors C# TpTask.TrySwitchArea: click bottom-right region switcher button.
        let btn_x = (w - 160.0 * scale).round() as i32;
        let btn_y = (h - 60.0 * scale).round() as i32;
        host.on_log(
            LogLevel::Info,
            format!(
                "TpTask: 点击右下角地区按钮 ({btn_x},{btn_y}) (视口 {frame_width}x{frame_height}, scale={scale:.2}) 寻找 [{}]",
                anchor.name
            ),
        );
        host.mouse_left_click_client(btn_x, btn_y);
        host.sleep_ms(120);

        let aliases: Vec<String> = anchor
            .aliases
            .iter()
            .map(|a| normalize_region_ocr_text(a))
            .collect();
        let mut last_candidates = Vec::new();

        for _attempt in 0..16 {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let hits = host.capture_ocr_roi(0.66, 0.0, 1.0, 1.0)?;
            last_candidates = hits.iter().map(|h| h.text.clone()).collect();

            let matched = hits
                .iter()
                .filter(|hit| {
                    let norm = normalize_region_ocr_text(&hit.text);
                    !norm.is_empty() && aliases.iter().any(|alias| norm.contains(alias))
                })
                .max_by_key(|hit| hit.center_y);

            if let Some(hit) = matched {
                host.on_log(
                    LogLevel::Info,
                    format!(
                        "TpTask: OCR 命中地区 [{}] -> '{}' ({}, {})，点击切换",
                        anchor.name, hit.text, hit.center_x, hit.center_y
                    ),
                );
                host.mouse_left_click_client(hit.center_x, hit.center_y);
                host.sleep_ms(480);
                return Ok(true);
            }
            host.sleep_ms(80);
        }

        host.on_log(
            LogLevel::Warn,
            format!(
                "TpTask: 切换地区 [{}] 未在右侧菜单识别到匹配项，OCR候选: {:?}，将回退直接拖拽地图",
                anchor.name, last_candidates
            ),
        );
        // Close region drawer if still open before falling back to dragging.
        host.mouse_left_click_client((w * 0.45).round() as i32, (h * 0.50).round() as i32);
        host.sleep_ms(150);
        Ok(false)
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
                .or_else(|| {
                    hits.iter()
                        .find(|h| h.text.eq_ignore_ascii_case("Teleport"))
                })
            {
                host.on_log(
                    LogLevel::Info,
                    format!(
                        "TpTask: 确认传送按钮 '{}' @ ({},{}) (第 {} 次)",
                        tp_btn.text,
                        tp_btn.center_x,
                        tp_btn.center_y,
                        attempt + 1
                    ),
                );
                host.mouse_left_click_client(tp_btn.center_x, tp_btn.center_y);
                host.sleep_ms(200);
                return Ok(());
            }

            if attempt < 2
                && let Some(cand) = hits.iter().find(|h| {
                    h.text.contains("传送锚点")
                        || h.text.contains("七天神像")
                        || h.text.contains("秘境")
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
        Self::run_from(host, task_name, task, None)
    }

    pub fn run_from<H: PathingHost + ?Sized>(
        host: &mut H,
        task_name: &str,
        task: &PathingTaskFile,
        checkpoint: Option<&PathingCheckpoint>,
    ) -> Result<(), String> {
        let total = task.positions.len();
        if total == 0 {
            return Err("路径点位为空".into());
        }
        let resume_index = if let Some(checkpoint) = checkpoint {
            let current = PathingCheckpoint::for_task(task, checkpoint.current_index)?;
            if current.path_fingerprint != checkpoint.path_fingerprint {
                return Err("恢复点位与当前路径内容不匹配".into());
            }
            checkpoint.current_index
        } else {
            0
        };
        let path_fingerprint =
            PathingCheckpoint::for_positions(&task.positions, resume_index)?.path_fingerprint;

        host.on_log(
            LogLevel::Info,
            format!("PathExecutor: 开始执行「{task_name}」，共 {total} 个点位"),
        );
        host.focus_game_window();
        host.reset_minimap_prev_pos();

        let guard = KeyReleaseGuard::new(host);
        let segments = split_segments_by_teleport(&task.positions);

        for segment in segments {
            if let Some(first) = segment.iter().find(|item| item.index >= resume_index)
                && first.waypoint.waypoint_type != "teleport"
            {
                guard
                    .host
                    .set_minimap_prev_pos(first.waypoint.x as f32, first.waypoint.y as f32);
            }

            for item in segment {
                if item.index < resume_index {
                    continue;
                }
                if guard.host.is_cancelled() {
                    return Err("已手动停止寻路".into());
                }
                guard.host.on_progress(item.index + 1, total);
                let checkpoint = PathingCheckpoint {
                    path_fingerprint,
                    current_index: item.index,
                };
                guard.host.on_checkpoint(&checkpoint);
                Self::execute_waypoint(guard.host, &item, total)?;
            }
        }

        guard.host.on_log(
            LogLevel::Info,
            format!("PathExecutor:「{task_name}」全部执行完成"),
        );
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
            let obs = Self::locate_minimap_retry(host, None, 5)?
                .ok_or_else(|| format!("[{idx1}/{total}] 方向点初始定位失败"))?;
            let target_angle =
                target_orientation_deg(obs.game_x as f64, obs.game_y as f64, wp.x, wp.y);
            Self::wait_until_rotated_to(host, target_angle, 3.0, 25)?;
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

        match wp.action.as_deref().map(str::trim) {
            Some("stop_flying") => {
                if let Some(wait_ms) = wp
                    .action_params
                    .as_deref()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .filter(|&ms| ms > 0)
                {
                    host.key_tap(VK_SPACE, 40);
                    host.sleep_ms(wait_ms.min(10_000));
                    host.key_tap(VK_SPACE, 40);
                    host.sleep_ms(300);
                }
                host.on_log(
                    LogLevel::Info,
                    format!("[{idx1}/{total}] 下落攻击 (stop_flying)"),
                );
                host.mouse_left_click();
                host.sleep_ms(300);
                let mut landed = false;
                for _ in 0..45 {
                    if host.is_cancelled() {
                        break;
                    }
                    if host.detect_motion_status()? != MotionStatus::Fly {
                        landed = true;
                        break;
                    }
                    host.sleep_ms(300);
                }
                if !landed && !host.is_cancelled() {
                    return Err(format!("[{idx1}/{total}] 下落攻击后仍处于飞行状态"));
                }
            }
            Some("use_gadget") => {
                host.on_log(
                    LogLevel::Info,
                    format!("[{idx1}/{total}] 使用小道具 (按 Z)"),
                );
                host.key_tap(VK_Z, 50);
                host.sleep_ms(250);
            }
            Some("linnea_mining") => {
                host.on_log(
                    LogLevel::Info,
                    format!(
                        "[{idx1}/{total}] 莉奈娅挖矿（参数 {}）",
                        wp.action_params.as_deref().unwrap_or("默认")
                    ),
                );
                host.run_linnea_mining(wp.action_params.as_deref())?;
            }
            _ => {}
        }

        if wp.waypoint_type == "target" {
            Self::move_close_to(host, wp.x, wp.y)?;
            if item.basic_action_supported {
                host.on_log(
                    LogLevel::Info,
                    format!("[{idx1}/{total}] 目标点交互 (按 F)"),
                );
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
        let mut last_diff = None;
        for _ in 0..max_tries {
            if host.is_cancelled() {
                return Err("已取消".to_string());
            }
            let Some(obs) = host.capture_minimap(None)? else {
                host.sleep_ms(50);
                continue;
            };
            let diff = signed_angle_diff_deg(obs.camera_angle, target_angle);
            last_diff = Some(diff);
            if diff.abs() < max_diff {
                return Ok(());
            }
            let dx = turn_mouse_dx(diff, dpi);
            if dx != 0 {
                host.mouse_move_rel(dx, 0);
            }
            host.sleep_ms(45);
        }
        Err(match last_diff {
            Some(diff) => format!("转向失败: 目标 {target_angle:.1}°，最后偏差 {diff:.1}°"),
            None => "转向失败: 未能获取小地图朝向".to_string(),
        })
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
        let init_angle =
            target_orientation_deg(initial.game_x as f64, initial.game_y as f64, wp.x, wp.y);
        host.on_log(
            LogLevel::Info,
            format!(
                "[{idx1}/{total}] 起始=({:.1},{:.1}) 前往=({:.1},{:.1}), 模式={}, 初始距离={init_dist:.1}, cam={:.1}° -> target={init_angle:.1}°",
                initial.game_x, initial.game_y, wp.x, wp.y, wp.move_mode, initial.camera_angle
            ),
        );
        Self::wait_until_rotated_to(host, init_angle, 5.0, 30)?;

        let dpi = host.dpi_scale();
        let arrive_dist = if wp.waypoint_type == "target" {
            4.0
        } else {
            4.5
        };
        let start = Instant::now();
        let mut last_dash = Instant::now() - Duration::from_secs(3);
        let mut last_jump = Instant::now() - Duration::from_secs(2);
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
                    host.mouse_right_up();
                    host.key_up(VK_LSHIFT);
                }
                return Err("已取消".to_string());
            }
            if start.elapsed() > Duration::from_secs(180) {
                host.key_up(VK_W);
                if sprint_held {
                    host.mouse_right_up();
                    host.key_up(VK_LSHIFT);
                }
                return Err(format!("[{idx1}/{total}] 移动超时 (180s)"));
            }

            let obs_opt = host.capture_minimap(None)?;
            let Some(obs) = obs_opt else {
                miss_count += 1;
                if miss_count.is_multiple_of(5) {
                    host.reset_minimap_prev_pos();
                    host.on_log(
                        LogLevel::Debug,
                        format!(
                            "[{idx1}/{total}] 连续 {miss_count} 帧小地图丢失，重置局部定位提示"
                        ),
                    );
                }
                if miss_count > 24 {
                    host.key_up(VK_W);
                    if sprint_held {
                        host.mouse_right_up();
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
                    host.mouse_right_up();
                    host.key_up(VK_LSHIFT);
                }
                return Err(format!("[{idx1}/{total}] 距离异常过远 ({dist:.1})"));
            }

            if stuck.observe((obs.game_x as f64, obs.game_y as f64)) {
                stuck_recoveries += 1;
                if stuck_recoveries > 4 {
                    host.key_up(VK_W);
                    if sprint_held {
                        host.mouse_right_up();
                        host.key_up(VK_LSHIFT);
                    }
                    return Err(format!("[{idx1}/{total}] 脱困次数超限"));
                }
                host.on_log(
                    LogLevel::Warn,
                    format!("[{idx1}/{total}] 疑似卡住，执行脱困 #{stuck_recoveries}"),
                );
                if obs.motion != MotionStatus::Fly {
                    host.key_tap(VK_SPACE, 50);
                }
                let strafe = if stuck_recoveries % 2 == 1 {
                    VK_A
                } else {
                    VK_D
                };
                host.key_down(strafe);
                host.sleep_ms(280);
                host.key_up(strafe);
                angle_offset = if stuck_recoveries % 2 == 1 {
                    35.0
                } else {
                    -35.0
                };
            } else if angle_offset.abs() > 0.1 && dist > 12.0 {
                angle_offset *= 0.85;
            } else {
                angle_offset = 0.0;
            }

            let target_angle =
                (target_orientation_deg(obs.game_x as f64, obs.game_y as f64, wp.x, wp.y)
                    + angle_offset)
                    .rem_euclid(360.0);
            let diff = signed_angle_diff_deg(obs.camera_angle, target_angle);
            if diff.abs() > 38.0 {
                host.key_up(VK_W);
                if sprint_held {
                    host.mouse_right_up();
                    host.key_up(VK_LSHIFT);
                    sprint_held = false;
                }
                Self::wait_until_rotated_to(host, target_angle, 5.0, 20)?;
                host.key_down(VK_W);
            } else {
                let dx = turn_mouse_dx(diff, dpi);
                if dx != 0 {
                    host.mouse_move_rel(dx, 0);
                }
            }

            match wp.move_mode.as_str() {
                "fly" => {
                    if sprint_held {
                        host.mouse_right_up();
                        host.key_up(VK_LSHIFT);
                        sprint_held = false;
                    }
                    if obs.motion != MotionStatus::Fly
                        && last_jump.elapsed() >= Duration::from_millis(250)
                    {
                        last_jump = Instant::now();
                        host.key_tap(VK_SPACE, 45);
                    }
                }
                "jump" => {
                    if sprint_held {
                        host.mouse_right_up();
                        host.key_up(VK_LSHIFT);
                        sprint_held = false;
                    }
                    if obs.motion != MotionStatus::Fly
                        && last_jump.elapsed() >= Duration::from_millis(300)
                    {
                        last_jump = Instant::now();
                        host.key_tap(VK_SPACE, 40);
                    }
                }
                "run" => {
                    let want = dist > 15.0 && obs.motion == MotionStatus::Normal;
                    if want && !sprint_held {
                        host.mouse_right_down();
                        sprint_held = true;
                    } else if !want && sprint_held {
                        host.mouse_right_up();
                        sprint_held = false;
                    }
                }
                "dash" => {
                    if sprint_held {
                        host.mouse_right_up();
                        sprint_held = false;
                    }
                    if dist > 6.0
                        && obs.motion == MotionStatus::Normal
                        && last_dash.elapsed() >= Duration::from_millis(950)
                    {
                        last_dash = Instant::now();
                        host.mouse_right_click();
                    }
                }
                "climb" => {
                    if sprint_held {
                        host.mouse_right_up();
                        sprint_held = false;
                    }
                }
                // `walk` / `swim` / unknown: periodic short dash over longer straight spans (AutoRun).
                _ => {
                    if sprint_held {
                        host.mouse_right_up();
                        sprint_held = false;
                    }
                    if dist > 20.0
                        && obs.motion == MotionStatus::Normal
                        && last_dash.elapsed() >= Duration::from_millis(2500)
                    {
                        last_dash = Instant::now();
                        host.mouse_right_click();
                    }
                }
            }

            host.sleep_ms(75);
        }

        host.key_up(VK_W);
        if sprint_held {
            host.mouse_right_up();
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
            let target_angle =
                target_orientation_deg(obs.game_x as f64, obs.game_y as f64, target_x, target_y);
            Self::wait_until_rotated_to(host, target_angle, 4.0, 12)?;
            host.key_down(VK_W);
            host.sleep_ms(65);
            host.key_up(VK_W);
            host.sleep_ms(25);
        }
        host.key_up(VK_W);
        host.sleep_ms(200);
        if host.is_cancelled() {
            return Err("已取消".to_string());
        }
        let obs = host
            .capture_minimap(None)?
            .ok_or_else(|| "接近目标失败: 最终小地图定位失败".to_string())?;
        let dist = distance_2d(obs.game_x as f64, obs.game_y as f64, target_x, target_y);
        if dist >= 2.0 {
            return Err(format!("接近目标失败: 最终距离 {dist:.1}"));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::fault_host::FaultHost;
    use crate::{PathingInfo, Waypoint};
    use bgi_task::FaultBoundary;

    #[test]
    fn injected_input_failure_suppresses_action() {
        let mut host = FaultHost::new(MockHost::new((0.0, 0.0)));
        host.injector
            .add(FaultBoundary::Input, 1, "input unavailable");
        host.key_tap(VK_W, 30);
        assert!(host.inner.events.is_empty());
    }

    #[test]
    fn injected_recognition_failure_propagates() {
        let mut host = FaultHost::new(MockHost::new((0.0, 0.0)));
        host.injector
            .add(FaultBoundary::Minimap, 1, "minimap unavailable");
        let result = host.capture_minimap(None);
        assert_eq!(result.unwrap_err(), "minimap unavailable");
    }

    #[test]
    fn injected_cancellation_is_observed() {
        let host = FaultHost::new(MockHost::new((0.0, 0.0)));
        host.injector.add(FaultBoundary::Cancellation, 1, "cancel");
        assert!(host.is_cancelled());
    }

    pub(crate) struct MockHost {
        pub(crate) cancelled: bool,
        pub(crate) capture_failures: usize,
        pub(crate) fail_on_focus: usize,
        fail_next_capture: bool,
        focus_calls: usize,
        pub(crate) capture_calls: usize,
        pub(crate) loading: bool,
        pub(crate) ocr_hits: Vec<OcrHit>,
        pub(crate) auto_pick_frame: Option<crate::auto_pick::AutoPickFrame>,
        pub(crate) checked_input_error: Option<String>,
        pos: (f32, f32),
        angle: f32,
        pub(crate) events: Vec<String>,
        progress: Vec<(usize, usize)>,
        checkpoints: Vec<PathingCheckpoint>,
        bigmap_open: bool,
        clicked_tp_btn: bool,
        w_down: bool,
        turn_enabled: bool,
        move_step: f32,
    }

    impl MockHost {
        pub(crate) fn new(start: (f32, f32)) -> Self {
            Self {
                cancelled: false,
                capture_failures: 0,
                fail_on_focus: 0,
                fail_next_capture: false,
                focus_calls: 0,
                capture_calls: 0,
                loading: false,
                ocr_hits: Vec::new(),
                auto_pick_frame: None,
                checked_input_error: None,
                pos: start,
                angle: 0.0,
                events: Vec::new(),
                progress: Vec::new(),
                checkpoints: Vec::new(),
                bigmap_open: false,
                clicked_tp_btn: false,
                w_down: false,
                turn_enabled: true,
                move_step: 2.5,
            }
        }
    }

    impl PathingHost for MockHost {
        fn is_cancelled(&self) -> bool {
            self.cancelled
        }
        fn current_waypoint_index(&self) -> Option<usize> {
            self.progress.last().map(|(index, _)| *index)
        }
        fn detect_loading(&mut self) -> Result<bool, String> {
            let loading = self.loading;
            self.loading = false;
            Ok(loading)
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {
            self.events.push("focus".into());
            self.focus_calls += 1;
            if self.fail_on_focus == self.focus_calls {
                self.fail_next_capture = true;
            }
        }
        fn capture_minimap(
            &mut self,
            prev: Option<(f32, f32)>,
        ) -> Result<Option<MinimapObservation>, String> {
            if self.fail_next_capture {
                self.fail_next_capture = false;
                return Err("injected capture failure".into());
            }
            if self.loading {
                self.loading = false;
                return Ok(None);
            }
            self.capture_calls += 1;
            if self.capture_failures == self.capture_calls {
                return Err("injected capture failure".into());
            }
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
                motion: MotionStatus::Normal,
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
        fn capture_ocr_roi(
            &mut self,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Vec<OcrHit>, String> {
            if self.ocr_hits.is_empty() {
                Ok(vec![OcrHit {
                    text: "传送".into(),
                    center_x: 1600,
                    center_y: 960,
                    confidence: 1.0,
                }])
            } else {
                Ok(self.ocr_hits.clone())
            }
        }
        fn capture_auto_pick(&mut self) -> Result<Option<crate::auto_pick::AutoPickFrame>, String> {
            Ok(self.auto_pick_frame.clone())
        }
        fn key_tap_checked(
            &mut self,
            vk: u16,
            _: u64,
            frame_size: (u32, u32),
            observed_at: Instant,
        ) -> Result<(), String> {
            if observed_at.elapsed() > bgi_task::AUTO_PICK_MAX_OBSERVATION_AGE {
                return Err("自动拾取帧已过期".into());
            }
            if let Some(error) = &self.checked_input_error {
                return Err(error.clone());
            }
            assert_eq!(frame_size, (1280, 720));
            if vk == VK_F {
                self.events.push("press_F".into());
            }
            Ok(())
        }
        fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
            self.pos = (x, y);
        }
        fn reset_minimap_prev_pos(&mut self) {}
        fn sleep_ms(&mut self, _: u64) {
            if self.w_down {
                let rad = self.angle.to_radians();
                // 0° points East (-game_x), 90° points South (-game_y).
                self.pos.0 -= rad.cos() * self.move_step;
                self.pos.1 -= rad.sin() * self.move_step;
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
            if self.turn_enabled {
                // Positive horizontal mouse movement turns clockwise (+degree in image coords).
                self.angle = (self.angle + (dx as f32) * 0.25).rem_euclid(360.0);
            }
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
        fn on_checkpoint(&mut self, checkpoint: &PathingCheckpoint) {
            self.checkpoints.push(*checkpoint);
        }
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
        fn run_linnea_mining(&mut self, params: Option<&str>) -> Result<(), String> {
            self.events
                .push(format!("linnea_mining:{}", params.unwrap_or("")));
            Ok(())
        }
    }

    #[test]
    fn mock_path_executor_runs_linnea_mining_action() {
        let task = PathingTaskFile {
            info: PathingInfo {
                name: "MiningAction".into(),
                ..Default::default()
            },
            positions: vec![Waypoint {
                x: 0.0,
                y: 0.0,
                waypoint_type: "path".into(),
                move_mode: "walk".into(),
                action: Some("linnea_mining".into()),
                action_params: Some("mines=2,rounds=3".into()),
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut host = MockHost::new((0.0, 0.0));
        PathExecutor::run(&mut host, "MiningAction", &task).unwrap();
        assert!(
            host.events
                .iter()
                .any(|event| event == "linnea_mining:mines=2,rounds=3"),
            "{:?}",
            host.events
        );
        assert!(
            !host.events.iter().any(|event| event.contains("暂未实现")),
            "linnea_mining 已支持，不应再报未实现: {:?}",
            host.events
        );
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
                    ..Default::default()
                },
                Waypoint {
                    x: 20.0,
                    y: 20.0,
                    waypoint_type: "target".into(),
                    move_mode: "walk".into(),
                    action: None,
                    action_params: None,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let mut host = MockHost::new((-500.0, -500.0));
        PathExecutor::run(&mut host, "MockRoute", &task).unwrap();
        assert_eq!(host.progress, vec![(1, 2), (2, 2)]);
        assert!(host.events.iter().any(|e| e == "press_F"));
        assert_eq!(host.events.last().map(String::as_str), Some("release_all"));
    }

    #[test]
    fn resumes_after_teleport_without_replaying_completed_waypoints() {
        let task = PathingTaskFile {
            info: PathingInfo {
                name: "ResumeRoute".into(),
                ..Default::default()
            },
            positions: vec![
                Waypoint {
                    x: 10.0,
                    y: 20.0,
                    waypoint_type: "teleport".into(),
                    ..Default::default()
                },
                Waypoint {
                    x: 20.0,
                    y: 20.0,
                    waypoint_type: "path".into(),
                    ..Default::default()
                },
                Waypoint {
                    x: 30.0,
                    y: 20.0,
                    waypoint_type: "target".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let checkpoint = PathingCheckpoint::for_task(&task, 2).unwrap();
        let mut host = MockHost::new((29.0, 20.0));
        PathExecutor::run_from(&mut host, "ResumeRoute", &task, Some(&checkpoint)).unwrap();
        assert_eq!(host.progress, [(3, 3)]);
        assert_eq!(host.checkpoints, [checkpoint]);
        assert!(host.events.iter().any(|event| event == "press_F"));
    }

    #[test]
    fn rejects_checkpoint_from_a_different_path() {
        let task = PathingTaskFile {
            positions: vec![Waypoint {
                x: 1.0,
                y: 2.0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let checkpoint = PathingCheckpoint {
            path_fingerprint: 0,
            current_index: 0,
        };
        let mut host = MockHost::new((1.0, 2.0));
        let error =
            PathExecutor::run_from(&mut host, "route", &task, Some(&checkpoint)).unwrap_err();
        assert!(error.contains("不匹配"), "{error}");
        assert!(host.progress.is_empty());
    }

    #[test]
    fn rotation_timeout_is_an_error() {
        let mut host = MockHost::new((0.0, 0.0));
        host.turn_enabled = false;
        let err = PathExecutor::wait_until_rotated_to(&mut host, 90.0, 1.0, 3).unwrap_err();
        assert!(err.contains("转向失败"), "{err}");
    }

    #[test]
    fn close_approach_timeout_is_an_error() {
        let mut host = MockHost::new((0.0, 0.0));
        host.move_step = 0.0;
        let err = PathExecutor::move_close_to(&mut host, 20.0, 0.0).unwrap_err();
        assert!(err.contains("接近目标失败"), "{err}");
        assert!(!host.w_down);
    }

    struct CrossRegionMockHost {
        auto_pick_frame: Option<crate::auto_pick::AutoPickFrame>,
        checked_input_error: Option<String>,
        pos: (f32, f32),
        view_center: (f32, f32),
        bigmap_open: bool,
        region_drawer_open: bool,
        clicked_tp_btn: bool,
        events: Vec<String>,
    }

    impl PathingHost for CrossRegionMockHost {
        fn is_cancelled(&self) -> bool {
            false
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {}
        fn capture_minimap(
            &mut self,
            prev: Option<(f32, f32)>,
        ) -> Result<Option<MinimapObservation>, String> {
            if self.clicked_tp_btn {
                self.pos = prev.unwrap_or(self.pos);
                self.clicked_tp_btn = false;
                self.bigmap_open = false;
            }
            Ok(Some(MinimapObservation {
                game_x: self.pos.0,
                game_y: self.pos.1,
                confidence: 0.99,
                camera_angle: 0.0,
                camera_confidence: 0.9,
                motion: MotionStatus::Normal,
            }))
        }
        fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
            if !self.bigmap_open {
                return Ok(None);
            }
            Ok(Some(BigMapObservation {
                projection: BigMapProjection {
                    game_tl_x: self.view_center.0 + 400.0,
                    game_tl_y: self.view_center.1 + 225.0,
                    game_br_x: self.view_center.0 - 400.0,
                    game_br_y: self.view_center.1 - 225.0,
                    frame_width: 1920,
                    frame_height: 1080,
                },
                inliers: 35,
            }))
        }
        fn capture_ocr_roi(
            &mut self,
            norm_x0: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Vec<OcrHit>, String> {
            if self.region_drawer_open && norm_x0 >= 0.60 {
                return Ok(vec![
                    OcrHit {
                        text: "璃月".into(),
                        center_x: 1620,
                        center_y: 380,
                        confidence: 1.0,
                    },
                    OcrHit {
                        text: "稻妻".into(),
                        center_x: 1620,
                        center_y: 460,
                        confidence: 1.0,
                    },
                ]);
            }
            Ok(vec![OcrHit {
                text: "传送".into(),
                center_x: 1600,
                center_y: 960,
                confidence: 1.0,
            }])
        }
        fn capture_auto_pick(&mut self) -> Result<Option<crate::auto_pick::AutoPickFrame>, String> {
            Ok(self.auto_pick_frame.clone())
        }
        fn key_tap_checked(
            &mut self,
            vk: u16,
            _: u64,
            frame_size: (u32, u32),
            observed_at: Instant,
        ) -> Result<(), String> {
            if observed_at.elapsed() > bgi_task::AUTO_PICK_MAX_OBSERVATION_AGE {
                return Err("自动拾取帧已过期".into());
            }
            if let Some(error) = &self.checked_input_error {
                return Err(error.clone());
            }
            assert_eq!(frame_size, (1280, 720));
            if vk == VK_F {
                self.events.push("press_F".into());
            }
            Ok(())
        }
        fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
            self.pos = (x, y);
        }
        fn reset_minimap_prev_pos(&mut self) {}
        fn sleep_ms(&mut self, _: u64) {}
        fn key_down(&mut self, _: u16) {}
        fn key_up(&mut self, _: u16) {}
        fn key_tap(&mut self, vk: u16, _: u64) {
            if vk == VK_M {
                self.bigmap_open = true;
            }
        }
        fn mouse_move_rel(&mut self, _: i32, _: i32) {}
        fn mouse_left_click(&mut self) {}
        fn mouse_left_click_client(&mut self, x: i32, y: i32) {
            self.events.push(format!("click({x},{y})"));
            if x >= 1700 && y >= 980 {
                self.region_drawer_open = true;
            } else if self.region_drawer_open && x == 1620 && y == 460 {
                self.region_drawer_open = false;
                self.view_center = (-4400.0, -3050.0);
                self.events.push("switched_to_inazuma".into());
            } else if x > 1400 && y > 800 {
                self.clicked_tp_btn = true;
            }
        }
        fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
            self.events.push(format!("drag({x1},{y1}->{x2},{y2})"));
        }
        fn mouse_wheel(&mut self, _: i32) {}
        fn release_all_keys(&mut self) {}
        fn on_progress(&mut self, _: usize, _: usize) {}
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
    }

    #[test]
    fn mock_cross_region_teleport_switches_country_before_panning() {
        let mut host = CrossRegionMockHost {
            auto_pick_frame: None,
            checked_input_error: None,
            pos: (270.0, -666.0),
            view_center: (270.0, -666.0),
            bigmap_open: false,
            region_drawer_open: false,
            clicked_tp_btn: false,
            events: Vec::new(),
        };

        TpTask::teleport(&mut host, -4350.0, -3010.0).unwrap();
        assert!(host.events.iter().any(|e| e == "switched_to_inazuma"));
        assert!((host.pos.0 - (-4350.0)).abs() < 1.0);
        assert!((host.pos.1 - (-3010.0)).abs() < 1.0);
    }

    struct ZoomInMockHost {
        auto_pick_frame: Option<crate::auto_pick::AutoPickFrame>,
        checked_input_error: Option<String>,
        pos: (f32, f32),
        view_center: (f64, f64),
        zoom: f32,
        bigmap_open: bool,
        clicked_tp_pin_zoomed_in: bool,
        clicked_tp_btn: bool,
        events: Vec<String>,
    }

    impl PathingHost for ZoomInMockHost {
        fn is_cancelled(&self) -> bool {
            false
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {}
        fn capture_minimap(
            &mut self,
            prev: Option<(f32, f32)>,
        ) -> Result<Option<MinimapObservation>, String> {
            if self.clicked_tp_btn {
                self.pos = prev.unwrap_or(self.pos);
                self.clicked_tp_btn = false;
                self.bigmap_open = false;
            }
            Ok(Some(MinimapObservation {
                game_x: self.pos.0,
                game_y: self.pos.1,
                confidence: 0.99,
                camera_angle: 0.0,
                camera_confidence: 0.9,
                motion: MotionStatus::Normal,
            }))
        }
        fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
            if !self.bigmap_open {
                return Ok(None);
            }
            let half_x = (1920.0 * self.zoom as f64 / 2.361) * 0.5;
            let half_y = (1080.0 * self.zoom as f64 / 2.361) * 0.5;
            Ok(Some(BigMapObservation {
                projection: BigMapProjection {
                    game_tl_x: (self.view_center.0 + half_x) as f32,
                    game_tl_y: (self.view_center.1 + half_y) as f32,
                    game_br_x: (self.view_center.0 - half_x) as f32,
                    game_br_y: (self.view_center.1 - half_y) as f32,
                    frame_width: 1920,
                    frame_height: 1080,
                },
                inliers: 75,
            }))
        }
        fn capture_ocr_roi(
            &mut self,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Vec<OcrHit>, String> {
            if self.clicked_tp_pin_zoomed_in {
                Ok(vec![OcrHit {
                    text: "传送".into(),
                    center_x: 1600,
                    center_y: 960,
                    confidence: 1.0,
                }])
            } else {
                Ok(vec![OcrHit {
                    text: "八酝岛".into(),
                    center_x: 1600,
                    center_y: 960,
                    confidence: 1.0,
                }])
            }
        }
        fn capture_auto_pick(&mut self) -> Result<Option<crate::auto_pick::AutoPickFrame>, String> {
            Ok(self.auto_pick_frame.clone())
        }
        fn key_tap_checked(
            &mut self,
            vk: u16,
            _: u64,
            frame_size: (u32, u32),
            observed_at: Instant,
        ) -> Result<(), String> {
            if observed_at.elapsed() > bgi_task::AUTO_PICK_MAX_OBSERVATION_AGE {
                return Err("自动拾取帧已过期".into());
            }
            if let Some(error) = &self.checked_input_error {
                return Err(error.clone());
            }
            assert_eq!(frame_size, (1280, 720));
            if vk == VK_F {
                self.events.push("press_F".into());
            }
            Ok(())
        }
        fn set_minimap_prev_pos(&mut self, x: f32, y: f32) {
            self.pos = (x, y);
        }
        fn reset_minimap_prev_pos(&mut self) {}
        fn sleep_ms(&mut self, _: u64) {}
        fn key_down(&mut self, _: u16) {}
        fn key_up(&mut self, _: u16) {}
        fn key_tap(&mut self, vk: u16, _: u64) {
            if vk == VK_M {
                self.bigmap_open = true;
            } else if vk == VK_ESCAPE {
                self.bigmap_open = false;
            }
        }
        fn mouse_move_rel(&mut self, _: i32, _: i32) {}
        fn mouse_move_client(&mut self, x: i32, y: i32) {
            self.events.push(format!("move_client({x},{y})"));
        }
        fn mouse_left_click(&mut self) {}
        fn mouse_left_click_client(&mut self, x: i32, y: i32) {
            self.events.push(format!("click({x},{y})"));
            if x > 1400 && y > 800 {
                self.clicked_tp_btn = true;
            } else if self.zoom <= BIGMAP_CLICK_MAX_ZOOM {
                self.clicked_tp_pin_zoomed_in = true;
            }
        }
        fn mouse_drag_client(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
            self.events.push(format!("drag({x1},{y1}->{x2},{y2})"));
            let scale = self.zoom as f64 / 2.361;
            self.view_center.0 += (x2 - x1) as f64 * scale;
            self.view_center.1 += (y2 - y1) as f64 * scale;
        }
        fn mouse_wheel(&mut self, notches: i32) {
            self.events.push(format!("wheel({notches})"));
            self.zoom = (self.zoom - notches as f32 * BIGMAP_ZOOM_PER_NOTCH).clamp(1.0, 6.0);
        }
        fn release_all_keys(&mut self) {}
        fn on_progress(&mut self, _: usize, _: usize) {}
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
    }

    #[test]
    fn mock_zooms_in_and_centers_before_clicking_teleport() {
        // Reproduces log scenario: map zoomed way out (zoom=5.8, span~4709x2655) with target near top edge.
        let mut host = ZoomInMockHost {
            auto_pick_frame: None,
            checked_input_error: None,
            pos: (9561.8, 5321.7),
            view_center: (9514.3, 4664.8),
            zoom: 5.8,
            bigmap_open: false,
            clicked_tp_pin_zoomed_in: false,
            clicked_tp_btn: false,
            events: Vec::new(),
        };

        TpTask::teleport(&mut host, 9561.8, 5321.7).unwrap();
        assert!(
            host.zoom <= BIGMAP_CLICK_MAX_ZOOM,
            "expected zoom <= 3.4, got {}",
            host.zoom
        );
        assert!(host.clicked_tp_pin_zoomed_in);
        assert!(
            host.events
                .iter()
                .any(|e| e.starts_with("move_client(960,540)"))
        );
        assert!(
            host.events
                .iter()
                .any(|e| e.starts_with("wheel(") && !e.contains('-'))
        );
    }

    struct LoadingFalsePositiveHost {
        poll_count: usize,
        clicked_tp_btn: bool,
        events: Vec<String>,
    }

    impl PathingHost for LoadingFalsePositiveHost {
        fn is_cancelled(&self) -> bool {
            false
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {}
        fn capture_minimap(
            &mut self,
            _: Option<(f32, f32)>,
        ) -> Result<Option<MinimapObservation>, String> {
            self.poll_count += 1;
            let pos = if self.poll_count <= 2 {
                (-5228.0, -6149.4)
            } else {
                (3656.2, 2433.5)
            };
            Ok(Some(MinimapObservation {
                game_x: pos.0,
                game_y: pos.1,
                confidence: 0.96,
                camera_angle: 0.0,
                camera_confidence: 0.9,
                motion: MotionStatus::Normal,
            }))
        }
        fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
            Ok(Some(BigMapObservation {
                projection: BigMapProjection {
                    game_tl_x: 3656.2 + 950.0,
                    game_tl_y: 2433.5 + 540.0,
                    game_br_x: 3656.2 - 950.0,
                    game_br_y: 2433.5 - 540.0,
                    frame_width: 1920,
                    frame_height: 1080,
                },
                inliers: 60,
            }))
        }
        fn capture_ocr_roi(
            &mut self,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Vec<OcrHit>, String> {
            Ok(vec![OcrHit {
                text: "传送".into(),
                center_x: 1691,
                center_y: 1007,
                confidence: 1.0,
            }])
        }
        fn set_minimap_prev_pos(&mut self, _: f32, _: f32) {}
        fn reset_minimap_prev_pos(&mut self) {
            self.events.push("reset_prev".into());
        }
        fn sleep_ms(&mut self, _: u64) {}
        fn key_down(&mut self, _: u16) {}
        fn key_up(&mut self, _: u16) {}
        fn key_tap(&mut self, _: u16, _: u64) {}
        fn mouse_move_rel(&mut self, _: i32, _: i32) {}
        fn mouse_left_click(&mut self) {}
        fn mouse_left_click_client(&mut self, x: i32, y: i32) {
            if x > 1400 && y > 800 {
                self.clicked_tp_btn = true;
            }
        }
        fn mouse_drag_client(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn mouse_wheel(&mut self, _: i32) {}
        fn release_all_keys(&mut self) {}
        fn on_progress(&mut self, _: usize, _: usize) {}
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
    }

    #[test]
    fn mock_teleport_ignores_distant_loading_false_positive() {
        let mut host = LoadingFalsePositiveHost {
            poll_count: 0,
            clicked_tp_btn: false,
            events: Vec::new(),
        };
        TpTask::teleport(&mut host, 3656.2, 2433.5).unwrap();
        assert!(
            host.poll_count >= 3,
            "should ignore first two far false positives"
        );
        assert!(host.events.iter().any(|e| e == "reset_prev"));
    }

    struct LayeredMapMockHost {
        pos: (f32, f32),
        bigmap_open: bool,
        on_underground_layer: bool,
        clicked_tp_btn: bool,
        events: Vec<String>,
    }

    impl PathingHost for LayeredMapMockHost {
        fn is_cancelled(&self) -> bool {
            false
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {}
        fn capture_minimap(
            &mut self,
            prev: Option<(f32, f32)>,
        ) -> Result<Option<MinimapObservation>, String> {
            if self.clicked_tp_btn {
                self.pos = prev.unwrap_or(self.pos);
                self.clicked_tp_btn = false;
                self.bigmap_open = false;
            }
            Ok(Some(MinimapObservation {
                game_x: self.pos.0,
                game_y: self.pos.1,
                confidence: 0.99,
                camera_angle: 0.0,
                camera_confidence: 0.9,
                motion: MotionStatus::Normal,
            }))
        }
        fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
            if !self.bigmap_open {
                return Ok(None);
            }
            let inliers = if self.on_underground_layer { 4 } else { 72 };
            Ok(Some(BigMapObservation {
                projection: BigMapProjection {
                    game_tl_x: 960.0,
                    game_tl_y: 540.0,
                    game_br_x: -960.0,
                    game_br_y: -540.0,
                    frame_width: 1920,
                    frame_height: 1080,
                },
                inliers,
            }))
        }
        fn capture_ocr_roi(
            &mut self,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Vec<OcrHit>, String> {
            Ok(vec![OcrHit {
                text: "传送".into(),
                center_x: 1691,
                center_y: 1007,
                confidence: 1.0,
            }])
        }
        fn set_minimap_prev_pos(&mut self, _: f32, _: f32) {}
        fn reset_minimap_prev_pos(&mut self) {}
        fn sleep_ms(&mut self, _: u64) {}
        fn key_down(&mut self, _: u16) {}
        fn key_up(&mut self, _: u16) {}
        fn key_tap(&mut self, vk: u16, _: u64) {
            if vk == VK_M {
                self.bigmap_open = true;
            }
        }
        fn mouse_move_rel(&mut self, _: i32, _: i32) {}
        fn mouse_left_click(&mut self) {}
        fn mouse_left_click_client(&mut self, x: i32, y: i32) {
            self.events.push(format!("click({x},{y})"));
            if x == 1843 && y == 503 {
                self.on_underground_layer = false;
                self.events.push("switched_to_ground_layer".into());
            } else if x > 1400 && y > 800 {
                self.clicked_tp_btn = true;
            }
        }
        fn mouse_drag_client(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn mouse_wheel(&mut self, _: i32) {}
        fn detect_bigmap_layer_buttons(&mut self) -> Result<BigMapLayerButtons, String> {
            if self.bigmap_open && self.on_underground_layer {
                Ok(BigMapLayerButtons {
                    underground_switch: Some((1841, 577)),
                    to_ground: Some((1843, 503)),
                })
            } else {
                Ok(BigMapLayerButtons::default())
            }
        }
        fn release_all_keys(&mut self) {}
        fn on_progress(&mut self, _: usize, _: usize) {}
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
    }

    #[test]
    fn mock_switches_from_layered_map_to_ground_before_teleport() {
        let mut host = LayeredMapMockHost {
            pos: (0.0, 0.0),
            bigmap_open: true,
            on_underground_layer: true,
            clicked_tp_btn: false,
            events: Vec::new(),
        };
        TpTask::teleport(&mut host, 25.0, -30.0).unwrap();
        assert!(!host.on_underground_layer);
        assert!(host.events.iter().any(|e| e == "switched_to_ground_layer"));
    }

    struct ActionMoveMockHost {
        pos: (f32, f32),
        motion: MotionStatus,
        w_down: bool,
        events: Vec<String>,
    }

    impl PathingHost for ActionMoveMockHost {
        fn is_cancelled(&self) -> bool {
            false
        }
        fn dpi_scale(&self) -> f32 {
            1.0
        }
        fn focus_game_window(&mut self) {}
        fn capture_minimap(
            &mut self,
            _: Option<(f32, f32)>,
        ) -> Result<Option<MinimapObservation>, String> {
            Ok(Some(MinimapObservation {
                game_x: self.pos.0,
                game_y: self.pos.1,
                confidence: 0.99,
                camera_angle: 180.0,
                camera_confidence: 0.9,
                motion: self.motion,
            }))
        }
        fn capture_bigmap(&mut self) -> Result<Option<BigMapObservation>, String> {
            Ok(None)
        }
        fn capture_ocr_roi(
            &mut self,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Vec<OcrHit>, String> {
            Ok(Vec::new())
        }
        fn set_minimap_prev_pos(&mut self, _: f32, _: f32) {}
        fn reset_minimap_prev_pos(&mut self) {}
        fn sleep_ms(&mut self, _: u64) {
            if self.w_down {
                self.pos.0 += 4.0;
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
            self.events.push(format!("key_tap({vk:#x})"));
            if vk == VK_SPACE {
                self.motion = MotionStatus::Fly;
            }
        }
        fn mouse_move_rel(&mut self, _: i32, _: i32) {}
        fn mouse_left_click(&mut self) {
            self.events.push("left_click".into());
            self.motion = MotionStatus::Normal;
        }
        fn mouse_right_click(&mut self) {
            self.events.push("right_dash".into());
        }
        fn mouse_left_click_client(&mut self, _: i32, _: i32) {}
        fn mouse_drag_client(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn mouse_wheel(&mut self, _: i32) {}
        fn detect_motion_status(&mut self) -> Result<MotionStatus, String> {
            Ok(self.motion)
        }
        fn release_all_keys(&mut self) {}
        fn on_progress(&mut self, _: usize, _: usize) {}
        fn on_log(&mut self, _: LogLevel, msg: String) {
            self.events.push(msg);
        }
    }

    #[test]
    fn mock_dash_jump_and_fly_move_modes() {
        let task = PathingTaskFile {
            info: PathingInfo {
                name: "ActionTest".into(),
                ..Default::default()
            },
            positions: vec![
                Waypoint {
                    x: 12.0,
                    y: 0.0,
                    waypoint_type: "path".into(),
                    move_mode: "dash".into(),
                    action: None,
                    action_params: None,
                    ..Default::default()
                },
                Waypoint {
                    x: 24.0,
                    y: 0.0,
                    waypoint_type: "path".into(),
                    move_mode: "jump".into(),
                    action: None,
                    action_params: None,
                    ..Default::default()
                },
                Waypoint {
                    x: 36.0,
                    y: 0.0,
                    waypoint_type: "path".into(),
                    move_mode: "fly".into(),
                    action: Some("stop_flying".into()),
                    action_params: None,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut host = ActionMoveMockHost {
            pos: (0.0, 0.0),
            motion: MotionStatus::Normal,
            w_down: false,
            events: Vec::new(),
        };
        PathExecutor::run(&mut host, "ActionTest", &task).unwrap();
        assert!(host.events.iter().any(|e| e == "right_dash"));
        assert!(host.events.iter().any(|e| e == "key_tap(0x20)"));
        assert!(host.events.iter().any(|e| e == "left_click"));
    }
}
