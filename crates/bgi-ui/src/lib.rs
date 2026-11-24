use std::collections::VecDeque;
use std::sync::Arc;

use bgi_control::{
    CaptureMode, ControlClient, CoreCommand, CoreEvent, LogEntry, LogLevel, PathSummary,
    TaskState, UiSnapshot,
};
use egui::{Color32, RichText, TextureHandle, TextureOptions};
use tokio::sync::oneshot;

const MISANS_REGULAR: &[u8] = include_bytes!("../../../assets/MiSans-Regular.ttf");
const MAX_LOGS: usize = 500;
const MAX_EVENTS_PER_FRAME: usize = 64;
const MAX_RAW_KEY_HISTORY: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavTab {
    Pathing,
    Capture,
    CvDebug,
    Input,
    Logs,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiShellAction {
    None,
    HideToTray,
    QuitApp,
}

pub struct BgiAppUi {
    active_tab: NavTab,
    logs: VecDeque<LogEntry>,
    raw_keys: VecDeque<String>,
    pending_replies: Vec<(String, oneshot::Receiver<Result<(), String>>)>,
    status_toast: Option<(String, bool)>,
    preview_texture: Option<TextureHandle>,
    last_preview_seq: u64,
    path_filter: String,
    pathing_dir_draft: String,
    assets_dir_draft: String,
    preferred_proc_draft: String,
    fps_draft: i32,
    drafts_initialized: bool,
    test_vk: u16,
}

impl Default for BgiAppUi {
    fn default() -> Self {
        Self::new()
    }
}

impl BgiAppUi {
    pub fn new() -> Self {
        Self {
            active_tab: NavTab::Pathing,
            logs: VecDeque::with_capacity(MAX_LOGS),
            raw_keys: VecDeque::with_capacity(MAX_RAW_KEY_HISTORY),
            pending_replies: Vec::new(),
            status_toast: None,
            preview_texture: None,
            last_preview_seq: 0,
            path_filter: String::new(),
            pathing_dir_draft: String::new(),
            assets_dir_draft: String::new(),
            preferred_proc_draft: String::new(),
            fps_draft: 10,
            drafts_initialized: false,
            test_vk: 0x7B, // F12
        }
    }

    /// Configures embedded MiSans font and desktop dark visuals once per created `egui::Context`.
    pub fn configure_context(ctx: &egui::Context) {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "misans".to_owned(),
            Arc::new(egui::FontData::from_static(MISANS_REGULAR)),
        );
        if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
            family.insert(0, "misans".to_owned());
        }
        if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
            family.insert(0, "misans".to_owned());
        }
        ctx.set_fonts(fonts);

        let mut visuals = egui::Visuals::dark();
        visuals.window_corner_radius = egui::CornerRadius::same(8);
        ctx.set_visuals(visuals);
    }

    pub fn render(&mut self, ui: &mut egui::Ui, client: &mut ControlClient) -> UiShellAction {
        let ctx = ui.ctx().clone();

        // 1. Read watch snapshot and drop lock guard immediately.
        let snapshot = client.snapshot();
        if !self.drafts_initialized {
            self.pathing_dir_draft = snapshot.config.pathing_dir.clone();
            self.assets_dir_draft = snapshot.config.assets_dir.clone();
            self.preferred_proc_draft = snapshot.config.preferred_process.clone();
            self.fps_draft = snapshot.config.preview_fps as i32;
            self.drafts_initialized = true;
        }

        // 2. Drain bounded CoreEvents non-blockingly with per-frame cap.
        self.drain_events(client);

        // 3. Poll in-flight command replies with try_recv (never blocking).
        self.poll_replies();

        // 4. Refresh single-slot preview texture only when frame seq advances.
        self.sync_preview_texture(&ctx, client);

        let mut shell_action = UiShellAction::None;

        egui::Panel::top("bgi_top_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("BetterGI-RS");
                ui.separator();

                let state_text = match &snapshot.task_state {
                    TaskState::Idle => RichText::new("空闲").color(Color32::LIGHT_GREEN),
                    TaskState::Previewing => {
                        if snapshot.config.capture_only_no_render {
                            RichText::new("仅抓图 (不渲染)").color(Color32::LIGHT_BLUE)
                        } else {
                            RichText::new("预览抓图中").color(Color32::LIGHT_BLUE)
                        }
                    }
                    TaskState::Pathing {
                        task_name,
                        current_index,
                        total_points,
                    } => RichText::new(format!("寻路中: {task_name} ({current_index}/{total_points})"))
                        .color(Color32::GOLD),
                    TaskState::Error(e) => RichText::new(format!("异常: {e}")).color(Color32::LIGHT_RED),
                };
                ui.label(state_text);

                ui.separator();
                let win_text = snapshot
                    .selected_window
                    .as_ref()
                    .map(|w| format!("{} ({}x{})", w.process_name, w.width, w.height))
                    .unwrap_or_else(|| "未绑定游戏窗口".to_string());
                ui.label(format!("窗口: {win_text}"));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("彻底退出").clicked() {
                        shell_action = UiShellAction::QuitApp;
                    }
                    if ui.button("退回托盘").clicked() {
                        shell_action = UiShellAction::HideToTray;
                    }
                    if ui
                        .add(
                            egui::Button::new(RichText::new("急停").color(Color32::WHITE))
                                .fill(Color32::from_rgb(180, 45, 45)),
                        )
                        .clicked()
                    {
                        self.send_cmd(client, "急停", CoreCommand::EmergencyStop);
                    }
                });
            });
        });

        egui::Panel::left("bgi_nav")
            .resizable(false)
            .default_size(148.0)
            .show(ui, |ui| {
                ui.add_space(6.0);
                let tabs = [
                    (NavTab::Pathing, "基础寻路"),
                    (NavTab::Capture, "抓图预览"),
                    (NavTab::CvDebug, "视觉与 OCR"),
                    (NavTab::Input, "热键与输入"),
                    (NavTab::Logs, "运行日志"),
                    (NavTab::Settings, "基础设置"),
                ];
                for (tab, label) in tabs {
                    let selected = self.active_tab == tab;
                    if ui
                        .add_sized([132.0, 32.0], egui::Button::selectable(selected, label))
                        .clicked()
                    {
                        self.active_tab = tab;
                    }
                }

                ui.separator();
                ui.label(
                    RichText::new("MVP 边界:\n仅基础寻路\n无遮罩 / 无多开")
                        .small()
                        .color(Color32::GRAY),
                );
            });

        if let Some((msg, is_err)) = &self.status_toast {
            egui::Panel::bottom("bgi_status_footer").show(ui, |ui| {
                let color = if *is_err {
                    Color32::LIGHT_RED
                } else {
                    Color32::LIGHT_GREEN
                };
                ui.label(RichText::new(msg).color(color));
            });
        }

        egui::CentralPanel::default().show(ui, |ui| match self.active_tab {
            NavTab::Pathing => self.render_pathing_tab(ui, client, &snapshot),
            NavTab::Capture => self.render_capture_tab(ui, client, &snapshot),
            NavTab::CvDebug => self.render_cv_debug_tab(ui, client, &snapshot),
            NavTab::Input => self.render_input_tab(ui, client, &snapshot),
            NavTab::Logs => self.render_logs_tab(ui),
            NavTab::Settings => self.render_settings_tab(ui, client, &snapshot),
        });

        shell_action
    }

    fn drain_events(&mut self, client: &mut ControlClient) {
        for _ in 0..MAX_EVENTS_PER_FRAME {
            let Some(ev) = client.try_recv_event() else {
                break;
            };
            match ev {
                CoreEvent::Log(entry) => {
                    if self.logs.len() >= MAX_LOGS {
                        self.logs.pop_front();
                    }
                    self.logs.push_back(entry);
                }
                CoreEvent::HotkeyTriggered { action, combo } => {
                    self.status_toast = Some((
                        format!("触发热键 [{combo}] -> {}", action.label()),
                        false,
                    ));
                }
                CoreEvent::RawKeyObserved {
                    vk,
                    scan_code,
                    extended,
                    pressed,
                    combo,
                } => {
                    if self.raw_keys.len() >= MAX_RAW_KEY_HISTORY {
                        self.raw_keys.pop_front();
                    }
                    let dir = if pressed { "DOWN" } else { "UP" };
                    let ext = if extended { " E0" } else { "" };
                    self.raw_keys.push_back(format!(
                        "{dir:<4} {combo:<16} VK=0x{vk:02X} Scan=0x{scan_code:02X}{ext}"
                    ));
                }
            }
        }
    }

    fn poll_replies(&mut self) {
        let mut i = 0;
        while i < self.pending_replies.len() {
            match self.pending_replies[i].1.try_recv() {
                Ok(res) => {
                    let (label, _) = self.pending_replies.swap_remove(i);
                    match res {
                        Ok(()) => self.status_toast = Some((format!("{label}: 完成"), false)),
                        Err(e) => self.status_toast = Some((format!("{label}失败: {e}"), true)),
                    }
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    i += 1;
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    let (label, _) = self.pending_replies.swap_remove(i);
                    self.status_toast = Some((format!("{label}: 核心未响应"), true));
                }
            }
        }
    }

    fn sync_preview_texture(&mut self, ctx: &egui::Context, client: &ControlClient) {
        let Some(frame) = client.latest_preview() else {
            self.preview_texture = None;
            self.last_preview_seq = 0;
            return;
        };
        if frame.seq == self.last_preview_seq {
            return;
        }
        let img = egui::ColorImage::from_rgba_unmultiplied(
            [frame.width as usize, frame.height as usize],
            &frame.rgba,
        );
        match &mut self.preview_texture {
            Some(tex) => tex.set(img, TextureOptions::LINEAR),
            None => {
                self.preview_texture = Some(ctx.load_texture("capture_preview", img, TextureOptions::LINEAR));
            }
        }
        self.last_preview_seq = frame.seq;
    }

    fn send_cmd(&mut self, client: &ControlClient, label: &str, cmd: CoreCommand) {
        match client.send_cmd(cmd) {
            Ok(rx) => self.pending_replies.push((label.to_string(), rx)),
            Err(e) => self.status_toast = Some((format!("{label}: {e}"), true)),
        }
    }

    fn render_pathing_tab(&mut self, ui: &mut egui::Ui, client: &ControlClient, snapshot: &UiSnapshot) {
        ui.heading("最基础寻路 (AutoPathing MVP)");
        ui.label("支持: 大地图传送 (teleport) + 小地图移动纠偏 (walk/run/dash/fly/jump/swim) + stop_flying / target 点按 F；复杂战斗/采集动作将自动跳过。");
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label("路径目录:");
            ui.text_edit_singleline(&mut self.pathing_dir_draft);
            if ui.button("应用并扫描").clicked() {
                let dir = self.pathing_dir_draft.clone();
                self.send_cmd(client, "设置路径目录", CoreCommand::SetPathingDir(dir));
            }
            if ui.button("刷新列表").clicked() {
                self.send_cmd(client, "刷新路径", CoreCommand::RefreshPaths);
            }
        });

        ui.horizontal(|ui| {
            ui.label("筛选:");
            ui.text_edit_singleline(&mut self.path_filter);
            if ui.button("启动基础寻路").clicked() {
                self.send_cmd(client, "启动基础寻路", CoreCommand::StartBasicPathing);
            }
        });

        ui.separator();
        ui.label(format!("已发现 {} 条路径:", snapshot.paths.len()));

        let filter_lower = self.path_filter.to_ascii_lowercase();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for path in &snapshot.paths {
                if !filter_lower.is_empty()
                    && !path.name.to_ascii_lowercase().contains(&filter_lower)
                    && !path.file_path.to_ascii_lowercase().contains(&filter_lower)
                {
                    continue;
                }
                self.render_path_row(ui, client, snapshot, path);
            }
        });
    }

    fn render_path_row(
        &mut self,
        ui: &mut egui::Ui,
        client: &ControlClient,
        snapshot: &UiSnapshot,
        path: &PathSummary,
    ) {
        let selected = snapshot
            .active_path_file
            .as_deref()
            .is_some_and(|p| p == path.file_path);
        ui.group(|ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(selected, &path.name).clicked() {
                    self.send_cmd(
                        client,
                        "选择路径",
                        CoreCommand::SelectPath {
                            file_path: path.file_path.clone(),
                        },
                    );
                }
                ui.label(format!(
                    "点位: {} | 基础可执行: {}",
                    path.waypoint_count, path.supported_count
                ));
                if !path.skipped_actions.is_empty() {
                    ui.label(
                        RichText::new(format!("跳过复杂动作: {}", path.skipped_actions.join(", ")))
                            .small()
                            .color(Color32::KHAKI),
                    );
                }
            });
            ui.label(RichText::new(&path.file_path).small().color(Color32::GRAY));
        });
    }

    fn render_capture_tab(&mut self, ui: &mut egui::Ui, client: &ControlClient, snapshot: &UiSnapshot) {
        ui.heading("窗口抓图预览与帧池调试");

        ui.horizontal(|ui| {
            if ui.button("刷新窗口").clicked() {
                self.send_cmd(client, "刷新窗口", CoreCommand::RefreshWindows);
            }

            let cur_title = snapshot
                .selected_window
                .as_ref()
                .map(|w| format!("{} | {} ({}x{})", w.process_name, w.title, w.width, w.height))
                .unwrap_or_else(|| "请选择目标窗口...".to_string());

            egui::ComboBox::from_id_salt("window_selector")
                .width(420.0)
                .selected_text(cur_title)
                .show_ui(ui, |ui| {
                    for w in &snapshot.available_windows {
                        let is_sel = snapshot
                            .selected_window
                            .as_ref()
                            .is_some_and(|s| s.hwnd == w.hwnd);
                        let label = format!("{} | {} ({}x{})", w.process_name, w.title, w.width, w.height);
                        if ui.selectable_label(is_sel, label).clicked() {
                            self.send_cmd(
                                client,
                                "绑定窗口",
                                CoreCommand::SelectWindow { hwnd: Some(w.hwnd) },
                            );
                        }
                    }
                });
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("截图模式:");
            for mode in [CaptureMode::BitBlt, CaptureMode::Wgc] {
                let checked = snapshot.config.capture_mode == mode;
                if ui.selectable_label(checked, mode.label()).clicked() {
                    self.send_cmd(client, "切换截图模式", CoreCommand::SetCaptureMode(mode));
                }
            }

            ui.separator();
            if ui.button("单帧抓图测试").clicked() {
                self.send_cmd(client, "单帧抓图", CoreCommand::CaptureSingleFrame);
            }
            let preview_label = if snapshot.preview_enabled {
                if snapshot.config.capture_only_no_render {
                    "停止持续抓图"
                } else {
                    "停止实时预览"
                }
            } else if snapshot.config.capture_only_no_render {
                "开启持续抓图 (不渲染)"
            } else {
                "开启实时预览"
            };
            if ui.button(preview_label).clicked() {
                self.send_cmd(
                    client,
                    "切换抓图",
                    CoreCommand::SetPreviewEnabled(!snapshot.preview_enabled),
                );
            }

            ui.separator();
            let mut no_render = snapshot.config.capture_only_no_render;
            if ui
                .checkbox(&mut no_render, "只捕获不渲染")
                .on_hover_text("开启后只执行抓图取帧与计数统计，不做 BGRA->RGBA 缩放、纹理上传与预览绘制")
                .changed()
            {
                self.send_cmd(
                    client,
                    "只捕获不渲染",
                    CoreCommand::SetCaptureNoRender(no_render),
                );
            }

            ui.label("目标 FPS:");
            let resp = ui.add(egui::Slider::new(&mut self.fps_draft, 1..=60));
            if !resp.dragged() && !resp.has_focus() {
                self.fps_draft = self.fps_draft.clamp(1, 60);
            }
            if resp.drag_stopped() || (resp.changed() && !resp.dragged()) {
                let target_fps = self.fps_draft.clamp(1, 60) as u32;
                self.send_cmd(client, "设置抓图 FPS", CoreCommand::SetPreviewFps(target_fps));
            }
        });

        ui.separator();
        if snapshot.config.capture_only_no_render {
            let st = &snapshot.capture_stats;
            ui.label(
                RichText::new(format!(
                    "只捕获不渲染模式 | 累计抓帧 #{frames} | 实测 {fps:.1} FPS | 分辨率 {w}x{h} | 单帧耗时 {ms:.2} ms（每秒刷新一次统计）",
                    frames = st.captured_frames,
                    fps = st.measured_fps,
                    w = st.source_width,
                    h = st.source_height,
                    ms = st.last_capture_ms,
                ))
                .color(Color32::LIGHT_BLUE),
            );
        } else if let Some(frame) = client.latest_preview() {
            ui.label(format!(
                "帧 #{seq} | 原始尺寸 {sw}x{sh} -> 预览 {pw}x{ph} | 耗时 {ms:.2} ms",
                seq = frame.seq,
                sw = frame.source_width,
                sh = frame.source_height,
                pw = frame.width,
                ph = frame.height,
                ms = frame.capture_ms
            ));
        } else {
            ui.label("暂无抓图帧。选择任意窗口后点击「单帧抓图测试」或「开启实时预览」。");
        }

        if !snapshot.config.capture_only_no_render
            && let Some(tex) = &self.preview_texture
        {
            let avail = ui.available_size();
            let tex_size = tex.size_vec2();
            if tex_size.x > 0.0 && tex_size.y > 0.0 {
                let scale = (avail.x / tex_size.x).min((avail.y - 16.0).max(120.0) / tex_size.y).min(1.0);
                ui.image((tex.id(), tex_size * scale));
            }
        }
    }

    fn render_cv_debug_tab(&mut self, ui: &mut egui::Ui, client: &ControlClient, snapshot: &UiSnapshot) {
        ui.heading("Phase 2: CV 定位、朝向与 PaddleOCR 联调台");
        ui.horizontal(|ui| {
            ui.label("Assets 根目录:");
            ui.text_edit_singleline(&mut self.assets_dir_draft);
            if ui.button("应用 Assets").clicked() {
                let dir = self.assets_dir_draft.clone();
                self.send_cmd(client, "设置 Assets 目录", CoreCommand::SetAssetsDir(dir));
            }
        });

        ui.horizontal(|ui| {
            if ui.button("小地图定位 + 朝向测试").clicked() {
                self.send_cmd(client, "小地图定位", CoreCommand::ProbeMiniMapCv);
            }
            if ui.button("大地图 SIFT 定位测试").clicked() {
                self.send_cmd(client, "大地图 SIFT 定位", CoreCommand::ProbeBigMapSift);
            }
            if ui.button("全屏 PaddleOCR 识别测试").clicked() {
                self.send_cmd(client, "PaddleOCR 识别", CoreCommand::ProbeOcrScreen);
            }
        });

        ui.separator();
        let dbg = &snapshot.cv_debug;
        if dbg.updated_seq == 0 {
            ui.label("尚未执行识别测试。请在「抓图预览」绑定原神窗口后，点击上方任意测试按钮。");
        } else {
            ui.label(format!(
                "结果 #{} [{}] | 耗时 {:.2} ms",
                dbg.updated_seq,
                dbg.source_label,
                dbg.elapsed_ms.unwrap_or(0.0)
            ));
            egui::Grid::new("cv_dbg_grid").striped(true).show(ui, |ui| {
                ui.label("游戏世界坐标 (X, Y):");
                let coord_str = match (dbg.game_x, dbg.game_y) {
                    (Some(x), Some(y)) => format!("({x:.2}, {y:.2})"),
                    _ => "未匹配".to_string(),
                };
                ui.label(RichText::new(coord_str).monospace());
                ui.end_row();

                ui.label("匹配置信度 / 层级:");
                ui.label(format!(
                    "{:.4} / Floor {:?}",
                    dbg.match_confidence.unwrap_or(0.0),
                    dbg.floor
                ));
                ui.end_row();

                ui.label("摄像机朝向 (置信度):");
                let cam_str = match (dbg.camera_angle, dbg.camera_confidence) {
                    (Some(a), Some(c)) => format!("{a:.1}° (conf={c:.3})"),
                    _ => "-".to_string(),
                };
                ui.label(RichText::new(cam_str).monospace());
                ui.end_row();

                ui.label("角色箭头朝向:");
                ui.label(
                    dbg.character_angle
                        .map(|a| format!("{a}°"))
                        .unwrap_or_else(|| "-".to_string()),
                );
                ui.end_row();
            });

            if let Some(text) = &dbg.ocr_text {
                ui.separator();
                ui.label("OCR 文本输出:");
                egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                    ui.label(RichText::new(text).monospace());
                });
            }
        }
    }

    fn render_input_tab(&mut self, ui: &mut egui::Ui, client: &ControlClient, snapshot: &UiSnapshot) {
        ui.heading("RawInputHub 全局热键与 SendInput 自检");
        if snapshot.raw_input_active {
            ui.label(RichText::new("RawInputHub 运行中（已过滤 hDevice == NULL 模拟输入）").color(Color32::LIGHT_GREEN));
        } else {
            let err = snapshot
                .raw_input_error
                .as_deref()
                .unwrap_or("未启动");
            ui.label(RichText::new(format!("RawInputHub 未就绪: {err}")).color(Color32::LIGHT_RED));
        }

        if let Some(action) = snapshot.capturing_hotkey_for {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("正在录制「{}」的新热键，请按下任意组合键...", action.label()))
                        .color(Color32::GOLD),
                );
                if ui.button("取消录制").clicked() {
                    self.send_cmd(client, "取消录制", CoreCommand::BeginHotkeyCapture(None));
                }
            });
        }

        ui.separator();
        egui::Grid::new("hotkey_grid").striped(true).show(ui, |ui| {
            ui.label(RichText::new("动作").strong());
            ui.label(RichText::new("当前绑定").strong());
            ui.label(RichText::new("操作").strong());
            ui.end_row();

            for hk in &snapshot.config.hotkeys {
                ui.label(hk.action.label());
                ui.label(RichText::new(hk.display_combo()).monospace());
                if ui.button("修改热键").clicked() {
                    self.send_cmd(
                        client,
                        "开始录制热键",
                        CoreCommand::BeginHotkeyCapture(Some(hk.action)),
                    );
                }
                ui.end_row();
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("SendInput 回环过滤测试:");
            if ui.button("发送模拟按键 F12").clicked() {
                self.send_cmd(
                    client,
                    "发送测试按键",
                    CoreCommand::TestInputKey { vk: self.test_vk },
                );
            }
            ui.label(
                RichText::new("（正常情况下下方物理按键流水不会出现 F12，说明已过滤自身注入）")
                    .small()
                    .color(Color32::GRAY),
            );
        });

        ui.separator();
        ui.label("最近物理按键事件（Raw Input）:");
        egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
            for line in self.raw_keys.iter().rev() {
                ui.label(RichText::new(line).monospace().small());
            }
        });
    }

    fn render_logs_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("实时运行日志");
            if ui.button("清空").clicked() {
                self.logs.clear();
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
            for log in &self.logs {
                let color = match log.level {
                    LogLevel::Debug => Color32::GRAY,
                    LogLevel::Info => Color32::LIGHT_GRAY,
                    LogLevel::Warn => Color32::GOLD,
                    LogLevel::Error => Color32::LIGHT_RED,
                };
                ui.label(
                    RichText::new(format!(
                        "[{:>7}ms][{}][{}] {}",
                        log.elapsed_ms,
                        log.level.tag(),
                        log.target,
                        log.message
                    ))
                    .monospace()
                    .color(color),
                );
            }
        });
    }

    fn render_settings_tab(&mut self, ui: &mut egui::Ui, client: &ControlClient, _snapshot: &UiSnapshot) {
        ui.heading("基础设置与内存设计说明");
        ui.horizontal(|ui| {
            ui.label("优先匹配进程名:");
            ui.text_edit_singleline(&mut self.preferred_proc_draft);
        });

        ui.horizontal(|ui| {
            ui.label("抓图预览帧率 (FPS):");
            let resp = ui.add(egui::Slider::new(&mut self.fps_draft, 1..=60));
            if resp.drag_stopped() || (resp.changed() && !resp.dragged()) {
                let target_fps = self.fps_draft.clamp(1, 60) as u32;
                self.send_cmd(client, "设置预览 FPS", CoreCommand::SetPreviewFps(target_fps));
            }
        });

        ui.separator();
        ui.label("架构约束状态:");
        ui.label("• bgi-ui 仅依赖 bgi-control，完全独立于 OpenCV / ORT / Win32 截图。");
        ui.label("• 关闭主窗口退回系统托盘时，将销毁 Win32 HWND 与 wgpu 渲染上下文以降低待机内存。");
        ui.label("• 与管理员权限游戏联动时，若热键无响应请以管理员身份运行本程序 (UIPI)。");
    }
}
