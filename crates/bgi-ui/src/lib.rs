use std::collections::VecDeque;
use std::sync::Arc;

use bgi_control::{
    CaptureMode, ControlClient, CoreCommand, CoreEvent, LogEntry, LogLevel, MAX_LOG_HISTORY,
    PathSummary, TaskLimitsConfig, TaskState, UiSnapshot,
};
use egui::{Color32, RichText, TextureHandle, TextureOptions};
use tokio::sync::oneshot;

const MISANS_REGULAR: &[u8] = include_bytes!("../../../assets/MiSans-Regular.ttf");
const MAX_LOGS: usize = MAX_LOG_HISTORY;
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
    log_min_level: LogLevel,
    log_filter: String,
    log_auto_scroll: bool,
    logs_hydrated: bool,
    last_log_seq: u64,
    raw_keys: VecDeque<String>,
    pending_replies: Vec<(String, oneshot::Receiver<Result<(), String>>)>,
    status_toast: Option<(String, bool)>,
    preview_texture: Option<TextureHandle>,
    last_preview_seq: u64,
    path_filter: String,
    pathing_dir_draft: String,
    pathing_runs_draft: u32,
    pathing_loop_delay_ms_draft: u64,
    pathing_wait_for_loading_draft: bool,
    pathing_auto_pick_draft: bool,
    auto_skip_draft: bool,
    auto_eat_draft: bool,
    auto_cook_interval_ms_draft: u64,
    auto_cook_stop_on_recover_draft: bool,
    auto_cook_timeout_ms_draft: u64,
    task_limits_draft: TaskLimitsConfig,
    fishing_dry_run_draft: bool,
    auto_wood_verify_gain: bool,
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
            log_min_level: LogLevel::Info,
            log_filter: String::new(),
            log_auto_scroll: true,
            logs_hydrated: false,
            last_log_seq: 0,
            raw_keys: VecDeque::with_capacity(MAX_RAW_KEY_HISTORY),
            pending_replies: Vec::new(),
            status_toast: None,
            preview_texture: None,
            last_preview_seq: 0,
            path_filter: String::new(),
            pathing_dir_draft: String::new(),
            pathing_runs_draft: 1,
            pathing_loop_delay_ms_draft: 1000,
            pathing_wait_for_loading_draft: false,
            pathing_auto_pick_draft: false,
            auto_skip_draft: false,
            auto_eat_draft: false,
            auto_cook_interval_ms_draft: 10,
            auto_cook_stop_on_recover_draft: true,
            auto_cook_timeout_ms_draft: 120_000,
            task_limits_draft: TaskLimitsConfig::default(),
            fishing_dry_run_draft: true,
            auto_wood_verify_gain: false,
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
            self.pathing_runs_draft = snapshot.config.pathing_runs;
            self.pathing_loop_delay_ms_draft = snapshot.config.pathing_loop_delay_ms;
            self.pathing_wait_for_loading_draft = snapshot.config.pathing_wait_for_loading;
            self.pathing_auto_pick_draft = snapshot.config.pathing_auto_pick;
            self.auto_skip_draft = snapshot.config.auto_skip;
            self.auto_eat_draft = snapshot.config.auto_eat;
            self.auto_cook_interval_ms_draft = snapshot.config.auto_cook_interval_ms;
            self.auto_cook_stop_on_recover_draft = snapshot.config.auto_cook_stop_on_recover;
            self.auto_cook_timeout_ms_draft = snapshot.config.auto_cook_timeout_ms;
            self.task_limits_draft = snapshot.config.task_limits.clone();
            self.fishing_dry_run_draft = snapshot.config.fishing_dry_run;
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
                        ..
                    } => RichText::new(format!(
                        "寻路中: {task_name} ({current_index}/{total_points})"
                    ))
                    .color(Color32::GOLD),
                    TaskState::Cooking { task_name, .. } => {
                        RichText::new(format!("烹饪中: {task_name}")).color(Color32::GOLD)
                    }
                    TaskState::ClaimingRewards { task_name, .. } => {
                        RichText::new(format!("领取中: {task_name}")).color(Color32::GOLD)
                    }
                    TaskState::OpeningChest { task_name, .. } => {
                        RichText::new(format!("开箱中: {task_name}")).color(Color32::GOLD)
                    }
                    TaskState::AutoWood { task_name, .. } => {
                        RichText::new(format!("砍树中: {task_name}")).color(Color32::GOLD)
                    }
                    TaskState::OneKeyExpedition { task_name, .. } => {
                        RichText::new(format!("派遣中: {task_name}")).color(Color32::GOLD)
                    }
                    TaskState::SalvagePreview { task_name, .. } => {
                        RichText::new(format!("只读预览中: {task_name}")).color(Color32::LIGHT_BLUE)
                    }
                    TaskState::Fishing { task_name, .. } => {
                        RichText::new(format!("钓鱼中: {task_name}")).color(Color32::GOLD)
                    }
                    TaskState::Stopping {
                        task_name,
                        current_index,
                        total_points,
                        ..
                    } => RichText::new(format!(
                        "停止中: {task_name} ({current_index}/{total_points})"
                    ))
                    .color(Color32::LIGHT_RED),
                    TaskState::Error(e) => {
                        RichText::new(format!("异常: {e}")).color(Color32::LIGHT_RED)
                    }
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
            NavTab::Logs => self.render_logs_tab(ui, client),
            NavTab::Settings => self.render_settings_tab(ui, client, &snapshot),
        });

        shell_action
    }

    fn push_log_entry(&mut self, entry: LogEntry) {
        if entry.seq <= self.last_log_seq {
            return;
        }
        self.last_log_seq = entry.seq;
        if self.logs.len() >= MAX_LOGS {
            self.logs.pop_front();
        }
        self.logs.push_back(entry);
    }

    fn drain_events(&mut self, client: &mut ControlClient) {
        if !self.logs_hydrated {
            self.logs_hydrated = true;
            for entry in client.recent_logs() {
                self.push_log_entry(entry);
            }
        }

        for _ in 0..MAX_EVENTS_PER_FRAME {
            let Some(ev) = client.try_recv_event() else {
                break;
            };
            match ev {
                CoreEvent::Log(entry) => self.push_log_entry(entry),
                CoreEvent::HotkeyTriggered { action, combo } => {
                    self.status_toast =
                        Some((format!("触发热键 [{combo}] -> {}", action.label()), false));
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
                self.preview_texture =
                    Some(ctx.load_texture("capture_preview", img, TextureOptions::LINEAR));
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

    fn render_pathing_tab(
        &mut self,
        ui: &mut egui::Ui,
        client: &ControlClient,
        snapshot: &UiSnapshot,
    ) {
        ui.heading("最基础寻路 (AutoPathing MVP)");
        ui.label("支持: 大地图传送 (teleport/分层自动切地表) + 小地图移动纠偏 (walk/run/dash/fly/jump/climb/swim) + stop_flying/use_gadget/pick_around/target 按 F；其余复杂战斗动作自动跳过。");
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

        let task_active = snapshot.task_state.is_task_active();
        ui.group(|ui| {
            ui.label(RichText::new("循环与执行策略").strong());
            ui.horizontal_wrapped(|ui| {
                ui.label("连续轮数:");
                let runs_resp = ui.add_enabled(
                    !task_active,
                    egui::DragValue::new(&mut self.pathing_runs_draft)
                        .range(0..=u32::MAX)
                        .speed(0.25)
                        .custom_formatter(|value, _| {
                            if value == 0.0 {
                                "无限".to_owned()
                            } else {
                                format!("{value:.0}")
                            }
                        }),
                );
                if runs_resp.drag_stopped() || (runs_resp.changed() && !runs_resp.dragged()) {
                    self.send_cmd(
                        client,
                        "设置连续轮数",
                        CoreCommand::SetPathingRuns(self.pathing_runs_draft),
                    );
                }

                ui.label("轮间延迟:");
                let delay_resp = ui.add_enabled(
                    !task_active,
                    egui::DragValue::new(&mut self.pathing_loop_delay_ms_draft)
                        .range(0..=u64::MAX)
                        .speed(25.0)
                        .suffix(" ms"),
                );
                if delay_resp.drag_stopped() || (delay_resp.changed() && !delay_resp.dragged()) {
                    self.send_cmd(
                        client,
                        "设置轮间延迟",
                        CoreCommand::SetPathingLoopDelayMs(self.pathing_loop_delay_ms_draft),
                    );
                }
            });

            ui.horizontal_wrapped(|ui| {
                let mut wait_for_loading = self.pathing_wait_for_loading_draft;
                if ui
                    .add_enabled(
                        !task_active,
                        egui::Checkbox::new(&mut wait_for_loading, "等待加载"),
                    )
                    .changed()
                {
                    self.pathing_wait_for_loading_draft = wait_for_loading;
                    self.send_cmd(
                        client,
                        "设置等待加载",
                        CoreCommand::SetPathingWaitForLoading(wait_for_loading),
                    );
                }

                let mut auto_pick = self.pathing_auto_pick_draft;
                if ui
                    .add_enabled(
                        !task_active,
                        egui::Checkbox::new(&mut auto_pick, "自动拾取"),
                    )
                    .changed()
                {
                    self.pathing_auto_pick_draft = auto_pick;
                    self.send_cmd(
                        client,
                        "设置自动拾取",
                        CoreCommand::SetPathingAutoPick(auto_pick),
                    );
                }
                let mut auto_skip = self.auto_skip_draft;
                if ui
                    .add(egui::Checkbox::new(&mut auto_skip, "自动剧情推进"))
                    .changed()
                {
                    self.auto_skip_draft = auto_skip;
                    self.send_cmd(
                        client,
                        "设置自动剧情推进",
                        CoreCommand::SetAutoSkipEnabled(auto_skip),
                    );
                }
                let mut auto_eat = self.auto_eat_draft;
                if ui
                    .add(egui::Checkbox::new(&mut auto_eat, "自动进食/复活"))
                    .changed()
                {
                    self.auto_eat_draft = auto_eat;
                    self.send_cmd(
                        client,
                        "设置自动进食/复活",
                        CoreCommand::SetAutoEatEnabled(auto_eat),
                    );
                }
            });
            if task_active {
                ui.label(
                    RichText::new("任务活跃时不可修改循环与执行策略")
                        .small()
                        .color(Color32::GRAY),
                );
            } else {
                ui.label(
                    RichText::new("连续轮数设为 0 时无限循环")
                        .small()
                        .color(Color32::GRAY),
                );
            }
        });
        ui.add_space(6.0);

        ui.horizontal_wrapped(|ui| {
            ui.label("筛选:");
            ui.text_edit_singleline(&mut self.path_filter);
            let can_stop = snapshot.task_state.is_task_active();
            if ui
                .add_enabled(!task_active, egui::Button::new("启动基础寻路 (F8)"))
                .clicked()
            {
                self.send_cmd(client, "启动基础寻路", CoreCommand::StartBasicPathing);
            }
            if ui
                .add_enabled(can_stop, egui::Button::new("停止当前任务"))
                .clicked()
            {
                self.send_cmd(client, "停止任务", CoreCommand::StopPathing);
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("一键领取当前页"))
                .clicked()
            {
                self.send_cmd(
                    client,
                    "一键领取当前页",
                    CoreCommand::ClaimCurrentPageRewards,
                );
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("持续领取当前可见页"))
                .clicked()
            {
                self.send_cmd(
                    client,
                    "持续领取当前可见页",
                    CoreCommand::ClaimVisiblePageRewards,
                );
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("启动自动烹饪"))
                .clicked()
            {
                self.send_cmd(client, "启动自动烹饪", CoreCommand::StartAutoCook);
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("自动开箱当前目标"))
                .clicked()
            {
                self.send_cmd(client, "自动开箱当前目标", CoreCommand::OpenCurrentChest);
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("自动砍树（王树瑞佑）"))
                .clicked()
            {
                self.send_cmd(
                    client,
                    "自动砍树（王树瑞佑）",
                    CoreCommand::StartAutoWood {
                        verify_gain: self.auto_wood_verify_gain,
                    },
                );
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("一键派遣（领取并再次派遣）"))
                .clicked()
            {
                self.send_cmd(client, "一键派遣", CoreCommand::StartOneKeyExpedition);
            }
            ui.label("一键派遣仅点击明确识别的全部领取和再次派遣；最多各一次，窗口失焦或取消立即停止，派遣图标未消失时不会按 ESC。");
            if ui
                .add_enabled(!task_active, egui::Button::new("自动采矿目标只读识别"))
                .clicked()
            {
                self.send_cmd(client, "自动采矿目标只读识别", CoreCommand::ProbeMining);
            }
            ui.label("自动采矿仅识别 ore、聚类并选择最近目标；最多 10 帧、间隔 200ms，连续 3 帧一致才报告。模型缺失或结果不确定时报告未知；不聚焦、不移动鼠标、不点击、不切换角色、不发送按键，也不支持 mining/linnea_mining 路径动作。");
            if snapshot.cv_debug.source_label.starts_with("自动采矿") {
                ui.separator();
                ui.label(format!(
                    "{} | 耗时 {:.2} ms",
                    snapshot.cv_debug.source_label,
                    snapshot.cv_debug.elapsed_ms.unwrap_or(0.0)
                ));
                if let Some(text) = &snapshot.cv_debug.ocr_text {
                    ui.label(RichText::new(text).monospace());
                }
            }
            if ui
                .add_enabled(!task_active, egui::Button::new("分解预览（不分解）"))
                .clicked()
            {
                self.send_cmd(
                    client,
                    "分解预览（不分解）",
                    CoreCommand::StartArtifactSalvagePreview,
                );
            }
        });
        let mut dry_run = self.fishing_dry_run_draft;
        if ui
            .add_enabled(
                !task_active,
                egui::Checkbox::new(&mut dry_run, "钓鱼预演（不发送输入）"),
            )
            .changed()
        {
            self.send_cmd(
                client,
                "钓鱼预演设置",
                CoreCommand::SetFishingDryRun(dry_run),
            );
            self.fishing_dry_run_draft = dry_run;
        }
        if ui
            .add_enabled(!task_active, egui::Button::new("手动入竿后尝试一次钓鱼"))
            .clicked()
        {
            self.send_cmd(
                client,
                "手动入竿后尝试一次钓鱼",
                CoreCommand::StartFishing {
                    dry_run: self.fishing_dry_run_draft,
                },
            );
        }
        ui.label("自动钓鱼不找鱼、不选饵、不自动抛竿；请在游戏内手动进入钓鱼并抛竿。预演仅识别咬钩和鱼条并记录动作。");
        let mut verify_gain = self.auto_wood_verify_gain;
        if ui
            .add_enabled(
                !task_active,
                egui::Checkbox::new(&mut verify_gain, "验证木材收益后重登刷新"),
            )
            .changed()
        {
            self.auto_wood_verify_gain = verify_gain;
        }
        egui::CollapsingHeader::new("任务安全边界（保存后重启仍生效）")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("仅限制领取、开箱、砍树和钓鱼；非法值会被核心拒绝，不影响当前配置。");
                egui::Grid::new("task_limits_grid").show(ui, |ui| {
                    let limits = &mut self.task_limits_draft;
                    ui.label("领取点击上限:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.reward_claim_max_clicks)
                            .range(1..=100)
                            .suffix(" 次"),
                    );
                    ui.end_row();
                    ui.label("领取总超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.reward_claim_timeout_ms)
                            .range(1000..=600_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("开箱总超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_open_chest_timeout_ms)
                            .range(1000..=3_600_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("开箱目标丢失超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_open_chest_target_lost_timeout_ms)
                            .range(500..=10_000)
                            .speed(100)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("开箱验证超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_open_chest_verify_timeout_ms)
                            .range(500..=30_000)
                            .speed(100)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("砍树总超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_wood_total_timeout_ms)
                            .range(1000..=3_600_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("砍树收益验证超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_wood_gain_timeout_ms)
                            .range(1000..=600_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("砍树登录超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_wood_login_timeout_ms)
                            .range(1000..=600_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("砍树菜单超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_wood_menu_timeout_ms)
                            .range(1000..=60_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("砍树退出确认超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_wood_logout_confirm_timeout_ms)
                            .range(1000..=60_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                    ui.label("砍树空 OCR 上限:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.auto_wood_empty_ocr_limit)
                            .range(1..=10)
                            .suffix(" 次"),
                    );
                    ui.end_row();
                    ui.label("钓鱼总超时:");
                    ui.add_enabled(
                        !task_active,
                        egui::DragValue::new(&mut limits.fishing_total_timeout_ms)
                            .range(1000..=3_600_000)
                            .speed(1000)
                            .suffix(" ms"),
                    );
                    ui.end_row();
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!task_active, egui::Button::new("应用并保存任务边界"))
                        .clicked()
                    {
                        self.send_cmd(
                            client,
                            "应用任务安全边界",
                            CoreCommand::SetTaskLimits(self.task_limits_draft.clone()),
                        );
                    }
                    if ui.button("恢复默认草稿").clicked() {
                        self.task_limits_draft = TaskLimitsConfig::default();
                    }
                });
            });
        ui.label(format!(
            "一键领取当前页：按从上到下、从左到右点击领取图标，单次最多 {} 次、总超时 {} 秒；检测到点击空白继续提示时按 ESC，不滚屏。",
            snapshot.config.task_limits.reward_claim_max_clicks,
            snapshot.config.task_limits.reward_claim_timeout_ms / 1000
        ));
        ui.label("持续领取当前可见页：空扫描间隔 180ms，连续 3 次无候选后结束；候选出现会重新计数。与一键领取共用点击次数和总超时上限，不滚屏、不持续按住鼠标。");
        ui.label("自动进食/复活仅在游戏画面且前台时按 Z；营养袋确认缓存 30 秒，进食冷却 1 秒，复活冷却 2 秒。");
        ui.label("自动剧情推进与自动进食/复活可在任务运行期间开关；任务模式按当前开关创建独立租约。剧情推进仅在对话 UI 且前台时按空格，每 200ms 最多一次；进食/复活仅在游戏画面且前台时按 Z。");
        ui.label("自动拾取默认关闭。仅在两个连续样本确认 F 拾取正例后，对提示相对区域做 OCR 并按白名单按 F；任一缺失或样本间隔超过 500ms 都重新计数。同一提示在本任务/配置周期最多一次，重置配置后才重新武装，不再盲按 F。发送瞬间重新校验任务、取消、排他暂停、前台、客户区尺寸和 750ms 帧新鲜度；名单缺失时停用。");
        egui::CollapsingHeader::new(format!(
            "任务期触发器租约 ({})",
            snapshot.trigger_leases.len()
        ))
        .default_open(true)
        .show(ui, |ui| {
            if snapshot.trigger_leases.is_empty() {
                ui.label("当前任务模式没有活动租约。");
            } else {
                for lease in &snapshot.trigger_leases {
                    ui.label(format!("#{} {}", lease.id, lease.name));
                    ui.label(
                        RichText::new(lease.options.to_string())
                            .small()
                            .color(Color32::GRAY),
                    );
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("烹饪检查:");
            let response = ui.add_enabled(
                !task_active,
                egui::DragValue::new(&mut self.auto_cook_interval_ms_draft)
                    .range(1..=10_000)
                    .speed(5.0)
                    .suffix(" ms"),
            );
            if response.drag_stopped() || (response.changed() && !response.dragged()) {
                self.send_cmd(
                    client,
                    "设置烹饪检查间隔",
                    CoreCommand::SetAutoCookIntervalMs(self.auto_cook_interval_ms_draft),
                );
            }
            ui.label("总超时:");
            let response = ui.add_enabled(
                !task_active,
                egui::DragValue::new(&mut self.auto_cook_timeout_ms_draft)
                    .range(1..=3_600_000)
                    .speed(1000.0)
                    .suffix(" ms"),
            );
            if response.drag_stopped() || (response.changed() && !response.dragged()) {
                self.send_cmd(
                    client,
                    "设置烹饪总超时",
                    CoreCommand::SetAutoCookTimeoutMs(self.auto_cook_timeout_ms_draft),
                );
            }
            let mut stop_on_recover = self.auto_cook_stop_on_recover_draft;
            if ui
                .add_enabled(
                    !task_active,
                    egui::Checkbox::new(&mut stop_on_recover, "再次烹饪按钮出现后结束"),
                )
                .changed()
            {
                self.auto_cook_stop_on_recover_draft = stop_on_recover;
                self.send_cmd(
                    client,
                    "设置烹饪结束条件",
                    CoreCommand::SetAutoCookStopOnRecover(stop_on_recover),
                );
            }
        });
        ui.label("自动烹饪：先手动开始烹饪，程序识别最佳熟度条并按空格完成；检测到再次烹饪按钮时默认结束，绝不把该按钮当普通确认点击。总超时默认 120 秒，超时后停止输入；关闭结束条件时仍不点击再次按钮。");
        ui.label(format!(
            "自动开箱当前目标：仅开启当前视野中的单个箱子，总超时 {} 秒；窗口失焦或检测到菜单/对话时立即停止。",
            snapshot.config.task_limits.auto_open_chest_timeout_ms / 1000
        ));
        ui.label("分解预览（不分解）：请手动打开圣遗物分解界面；使用 ItemV2 识别图标星级并仅滚屏，绝不选择卡片、分解或确认。模型缺失、未命中或置信度不足时 fail-closed；实机识别结果需人工核对。");
        ui.label(format!(
            "自动砍树：固定 1 轮，必须先检测到已装备的王树瑞佑才按 Z；关闭收益验证时等待 8 秒后结束，不 OCR、不退出。开启时验证木材收益后才检测菜单并退出重登，总超时 {} 秒。",
            snapshot.config.task_limits.auto_wood_total_timeout_ms / 1000
        ));

        if let TaskState::Pathing {
            task_name,
            current_index,
            total_points,
            ..
        } = &snapshot.task_state
        {
            let total = (*total_points).max(1);
            let frac = (*current_index as f32 / total as f32).clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!(
                        "正在执行「{task_name}」: 点位 {current_index}/{total_points}"
                    ))
                    .animate(true),
            );
        }

        if let Some(report) = &snapshot.last_task_report {
            ui.group(|ui| {
                ui.label(format!(
                    "上次结果: {:?} | 「{}」| 计数 {} | {} ms | 阶段 {}",
                    report.outcome,
                    report.task_name,
                    report.runs_completed,
                    report.elapsed_ms,
                    report.stage
                ));
                if let Some(index) = report.current_index {
                    ui.label(format!("停止/失败点位: {}/{}", index, report.total_points));
                }
                if let Some(failure) = &report.failure {
                    ui.label(RichText::new(failure).color(Color32::LIGHT_RED));
                }
            });
        }

        if let Some(preview) = &snapshot.last_salvage_preview_report {
            ui.group(|ui| {
                ui.label(RichText::new("分解预览报告（不分解）").strong());
                ui.label(format!(
                    "状态: {:?} | 页数: {} | 滚屏: {} | 无进展: {} | 候选: {}",
                    preview.stop_reason,
                    preview.pages_seen,
                    preview.interactions_used,
                    preview.no_progress,
                    preview.candidates.len()
                ));
                ui.label(format!(
                    "跳过: 空 {} / 锁定 {} / 已选 {} / 5星 {} / 重叠 {}",
                    preview.skipped.empty,
                    preview.skipped.locked,
                    preview.skipped.selected,
                    preview.skipped.five_star,
                    preview.skipped.overlapping
                ));
                ui.label(format!(
                    "选择点击: {} | 分解点击: {} | 确认点击: {} | 最终确认授权: {}",
                    preview.selection_clicks,
                    preview.salvage_clicks,
                    preview.confirmation_clicks,
                    preview.final_confirmation_authorized
                ));
                if let Some(failure) = &preview.failure {
                    ui.label(RichText::new(failure).color(Color32::LIGHT_RED));
                }
                egui::ScrollArea::vertical()
                    .id_salt("salvage_preview_candidates")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for candidate in &preview.candidates {
                            ui.label(format!(
                                "报告内编号={} 页{} 格{} 行{} 列{} 客户区({},{},{}x{}) {}星 置信度={:.3}",
                                candidate.id,
                                candidate.position.page + 1,
                                candidate.position.slot,
                                candidate.position.row + 1,
                                candidate.position.column + 1,
                                candidate.position.x,
                                candidate.position.y,
                                candidate.position.width,
                                candidate.position.height,
                                candidate.star,
                                candidate.confidence
                            ));
                        }
                    });
            });
        }

        if matches!(snapshot.task_state, TaskState::Stopping { .. }) {
            ui.label(RichText::new("正在等待执行器退出并释放按键…").color(Color32::LIGHT_RED));
        }

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

    fn render_capture_tab(
        &mut self,
        ui: &mut egui::Ui,
        client: &ControlClient,
        snapshot: &UiSnapshot,
    ) {
        ui.heading("窗口抓图预览与帧池调试");

        ui.horizontal(|ui| {
            if ui.button("刷新窗口").clicked() {
                self.send_cmd(client, "刷新窗口", CoreCommand::RefreshWindows);
            }

            let cur_title = snapshot
                .selected_window
                .as_ref()
                .map(|w| {
                    format!(
                        "{} | {} ({}x{})",
                        w.process_name, w.title, w.width, w.height
                    )
                })
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
                        let label = format!(
                            "{} | {} ({}x{})",
                            w.process_name, w.title, w.width, w.height
                        );
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
            for mode in [
                CaptureMode::BitBlt,
                CaptureMode::Wgc,
                CaptureMode::DwmSharedSurface,
            ] {
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
                .on_hover_text(
                    "开启后只执行抓图取帧与计数统计，不做 BGRA->RGBA 缩放、纹理上传与预览绘制",
                )
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
                self.send_cmd(
                    client,
                    "设置抓图 FPS",
                    CoreCommand::SetPreviewFps(target_fps),
                );
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
                let scale = (avail.x / tex_size.x)
                    .min((avail.y - 16.0).max(120.0) / tex_size.y)
                    .min(1.0);
                ui.image((tex.id(), tex_size * scale));
            }
        }
    }

    fn render_cv_debug_tab(
        &mut self,
        ui: &mut egui::Ui,
        client: &ControlClient,
        snapshot: &UiSnapshot,
    ) {
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

        ui.horizontal(|ui| {
            if ui.button("武器详情只读识别测试").clicked() {
                self.send_cmd(client, "武器详情只读识别", CoreCommand::ProbeWeaponDetails);
            }
        });
        ui.label(
            "请手动停在武器详情页：仅识别名称、当前等级、等级上限；最多 10 帧、间隔 200ms，连续 3 帧一致才成功。不点击、不升级、不分解；无页面模板，不保证拒绝其他界面。",
        );

        ui.horizontal(|ui| {
            if ui.button("天赋详情只读识别测试").clicked() {
                self.send_cmd(client, "天赋详情只读识别", CoreCommand::ProbeTalentDetails);
            }
        });
        ui.label(
            "请手动停在天赋详情页：识别“天赋介绍”、类型、显示等级和“天赋等级+3”；最多 10 帧、间隔 200ms，完整三元组连续 3 次一致才成功。加成区域为空表示无加成，异常非空文本会拒绝；不点击、不切换天赋、不升级。",
        );

        ui.horizontal(|ui| {
            if ui.button("角色属性等级只读识别测试").clicked() {
                self.send_cmd(
                    client,
                    "角色属性等级只读识别",
                    CoreCommand::ProbeCharacterAttributes,
                );
            }
        });
        ui.label(
            "请手动停在角色属性页：仅识别当前等级和等级上限；最多 10 帧、间隔 200ms，等级对连续 3 次一致才成功。需通过左侧“属性”页门控；不点击、不切换角色、不升级。",
        );

        ui.horizontal(|ui| {
            if ui.button("奖励结果页卡片只读识别测试").clicked() {
                self.send_cmd(
                    client,
                    "奖励结果页卡片只读识别",
                    CoreCommand::ProbeRewardResultCards,
                );
            }
        });
        ui.label(
            "请手动停在 1920×1080 奖励结果页：识别卡片名称与数量；最多 10 帧、间隔 200ms，完整卡片连续 3 帧一致才成功。数量失败显示“未知”，不默认按 1 累计；不点击、不翻页、不领取、不按 ESC。",
        );

        ui.horizontal(|ui| {
            if ui.button("派遣页面状态只读识别测试").clicked() {
                self.send_cmd(
                    client,
                    "派遣页面状态只读识别",
                    CoreCommand::ProbeExpeditionStatus,
                );
            }
        });
        ui.label(
            "请手动停在派遣页面：仅识别“全部领取”或“再次派遣”图标；最多 10 帧、间隔 200ms，同一状态连续 3 帧一致才成功。两者同时出现、均未出现或页面不确定时不作结论，不等于无奖励或零完成；不点击、不领取、不翻页、不按 ESC。",
        );

        ui.separator();
        let metrics = &snapshot.runtime_metrics;
        ui.label(format!(
            "触发调度 | 提交 {} / 完成 {} / 丢弃 {} | 准点率 {:.1}% | 平均排队 {} µs | 阶段均值 抓图 {:.2} / 识别 {:.2} / 决策 {:.2} / 输入 {:.2} ms",
            metrics.submitted,
            metrics.completed,
            metrics.dropped,
            metrics.deadline_hit_rate * 100.0,
            metrics.average_queue_delay_us,
            metrics.capture.total_ms / metrics.capture.samples.max(1) as f32,
            metrics.recognition.total_ms / metrics.recognition.samples.max(1) as f32,
            metrics.decision.total_ms / metrics.decision.samples.max(1) as f32,
            metrics.input.total_ms / metrics.input.samples.max(1) as f32,
        ));
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
                egui::ScrollArea::vertical()
                    .max_height(240.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new(text).monospace());
                    });
            }
        }
    }

    fn render_input_tab(
        &mut self,
        ui: &mut egui::Ui,
        client: &ControlClient,
        snapshot: &UiSnapshot,
    ) {
        ui.heading("RawInputHub 全局热键与 SendInput 自检");
        if snapshot.raw_input_active {
            ui.label(
                RichText::new("RawInputHub 运行中（已过滤 hDevice == NULL 模拟输入）")
                    .color(Color32::LIGHT_GREEN),
            );
        } else {
            let err = snapshot.raw_input_error.as_deref().unwrap_or("未启动");
            ui.label(RichText::new(format!("RawInputHub 未就绪: {err}")).color(Color32::LIGHT_RED));
        }

        if let Some(action) = snapshot.capturing_hotkey_for {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "正在录制「{}」的新热键，请按下任意组合键...",
                        action.label()
                    ))
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
        egui::ScrollArea::vertical()
            .max_height(220.0)
            .show(ui, |ui| {
                for line in self.raw_keys.iter().rev() {
                    ui.label(RichText::new(line).monospace().small());
                }
            });
    }

    fn render_logs_tab(&mut self, ui: &mut egui::Ui, client: &ControlClient) {
        let needle = self.log_filter.trim().to_ascii_lowercase();
        let min_rank = self.log_min_level.rank();

        let format_entry = |log: &LogEntry| {
            let ts = if log.timestamp.is_empty() {
                format!("{:>7}ms", log.elapsed_ms)
            } else {
                log.timestamp.clone()
            };
            format!(
                "[{ts}][{}][{}] {}",
                log.level.tag(),
                log.target,
                log.message
            )
        };

        let visible_lines: Vec<(LogLevel, String)> = self
            .logs
            .iter()
            .filter(|log| {
                log.level.rank() >= min_rank
                    && (needle.is_empty()
                        || log.target.to_ascii_lowercase().contains(&needle)
                        || log.message.to_ascii_lowercase().contains(&needle))
            })
            .map(|log| (log.level, format_entry(log)))
            .collect();

        ui.horizontal_wrapped(|ui| {
            ui.heading("实时运行日志");
            ui.separator();
            ui.label("级别:");
            ui.selectable_value(&mut self.log_min_level, LogLevel::Debug, "Debug+");
            ui.selectable_value(&mut self.log_min_level, LogLevel::Info, "Info+");
            ui.selectable_value(&mut self.log_min_level, LogLevel::Warn, "Warn+");
            ui.selectable_value(&mut self.log_min_level, LogLevel::Error, "Error");
            ui.separator();
            ui.label("搜索:");
            ui.add(
                egui::TextEdit::singleline(&mut self.log_filter)
                    .desired_width(160.0)
                    .hint_text("模块或关键字"),
            );
            ui.checkbox(&mut self.log_auto_scroll, "自动滚动");

            if ui.button("复制日志").clicked() {
                let text = visible_lines
                    .iter()
                    .map(|(_, line)| line.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let count = visible_lines.len();
                ui.ctx().copy_text(text);
                self.status_toast = Some((format!("已复制 {count} 条日志到剪贴板"), false));
            }
            if ui.button("打开日志目录").clicked() {
                self.send_cmd(client, "打开日志目录", CoreCommand::OpenLogDir);
            }
            if ui.button("清空列表").clicked() {
                self.logs.clear();
            }
        });

        ui.label(
            RichText::new(format!(
                "显示 {} / 共 {} 条（单次上限 {} 条，磁盘文件自动按天保存）",
                visible_lines.len(),
                self.logs.len(),
                MAX_LOGS
            ))
            .small()
            .color(Color32::GRAY),
        );
        ui.separator();

        egui::ScrollArea::vertical()
            .stick_to_bottom(self.log_auto_scroll)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (level, line) in &visible_lines {
                    let color = match level {
                        LogLevel::Debug => Color32::GRAY,
                        LogLevel::Info => Color32::LIGHT_GRAY,
                        LogLevel::Warn => Color32::GOLD,
                        LogLevel::Error => Color32::LIGHT_RED,
                    };
                    ui.label(RichText::new(line).monospace().color(color));
                }
            });
    }

    fn render_settings_tab(
        &mut self,
        ui: &mut egui::Ui,
        client: &ControlClient,
        _snapshot: &UiSnapshot,
    ) {
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
                self.send_cmd(
                    client,
                    "设置预览 FPS",
                    CoreCommand::SetPreviewFps(target_fps),
                );
            }
        });

        ui.separator();
        ui.label("架构约束状态:");
        ui.label("• bgi-ui 仅依赖 bgi-control，完全独立于 OpenCV / ORT / Win32 截图。");
        ui.label(
            "• 关闭主窗口退回系统托盘时，将销毁 Win32 HWND 与 wgpu 渲染上下文以降低待机内存。",
        );
        ui.label("• 与管理员权限游戏联动时，若热键无响应请以管理员身份运行本程序 (UIPI)。");
    }
}
