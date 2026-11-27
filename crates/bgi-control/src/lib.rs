use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, watch};

/// Capture backend mode exposed cleanly in settings and UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CaptureMode {
    #[default]
    BitBlt,
    Wgc,
    DwmSharedSurface,
}

impl CaptureMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::BitBlt => "BitBlt",
            Self::Wgc => "WGC (自研双缓冲)",
            Self::DwmSharedSurface => "DwmSharedSurface (无WGC)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskOutcome {
    Success,
    Partial,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskReport {
    pub task_id: u64,
    pub task_name: String,
    pub outcome: TaskOutcome,
    pub runs_completed: u32,
    pub runs_requested: u32,
    pub current_index: Option<usize>,
    pub total_points: usize,
    pub failure: Option<String>,
    pub stage: String,
    pub elapsed_ms: u64,
}

/// Logical task state tracked by core and rendered in UI snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TaskState {
    #[default]
    Idle,
    Previewing,
    Pathing {
        task_id: u64,
        task_name: String,
        current_index: usize,
        total_points: usize,
    },
    Cooking {
        task_id: u64,
        task_name: String,
    },
    ClaimingRewards {
        task_id: u64,
        task_name: String,
    },
    OpeningChest {
        task_id: u64,
        task_name: String,
    },
    AutoWood {
        task_id: u64,
        task_name: String,
    },
    OneKeyExpedition {
        task_id: u64,
        task_name: String,
    },
    SalvagePreview {
        task_id: u64,
        task_name: String,
    },
    Fishing {
        task_id: u64,
        task_name: String,
    },
    Mining {
        task_id: u64,
        task_name: String,
    },
    Stopping {
        task_id: u64,
        task_name: String,
        current_index: usize,
        total_points: usize,
    },
    Error(String),
}

impl TaskState {
    pub fn is_task_active(&self) -> bool {
        matches!(
            self,
            Self::Pathing { .. }
                | Self::Cooking { .. }
                | Self::ClaimingRewards { .. }
                | Self::OpeningChest { .. }
                | Self::AutoWood { .. }
                | Self::OneKeyExpedition { .. }
                | Self::SalvagePreview { .. }
                | Self::Fishing { .. }
                | Self::Mining { .. }
                | Self::Stopping { .. }
        )
    }
}

/// Supported global hotkey actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HotkeyAction {
    StartStopPathing,
    EmergencyStop,
    TogglePreview,
    TestPing,
}

impl HotkeyAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::StartStopPathing => "启动 / 停止基础寻路",
            Self::EmergencyStop => "全局急停 (释放按键)",
            Self::TogglePreview => "开关抓图预览",
            Self::TestPing => "热键测试信号",
        }
    }
}

/// Physical modifier keys tracked by `RawInputHub`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ModifierMask {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub win: bool,
}

impl ModifierMask {
    pub fn format_prefix(self) -> String {
        let mut parts = Vec::with_capacity(4);
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.alt {
            parts.push("Alt");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.win {
            parts.push("Win");
        }
        parts.join("+")
    }
}

/// Format a human-readable Win32 virtual-key + modifier string.
pub fn format_hotkey(modifiers: ModifierMask, vk: u16) -> String {
    let key_name = vk_name(vk);
    let prefix = modifiers.format_prefix();
    if prefix.is_empty() {
        key_name
    } else {
        format!("{prefix}+{key_name}")
    }
}

/// Minimal readable Win32 virtual key naming table.
pub fn vk_name(vk: u16) -> String {
    match vk {
        0 => "未绑定".to_string(),
        0x08 => "Backspace".to_string(),
        0x09 => "Tab".to_string(),
        0x0D => "Enter".to_string(),
        0x13 => "Pause".to_string(),
        0x1B => "Esc".to_string(),
        0x20 => "Space".to_string(),
        0x21 => "PageUp".to_string(),
        0x22 => "PageDown".to_string(),
        0x23 => "End".to_string(),
        0x24 => "Home".to_string(),
        0x25 => "Left".to_string(),
        0x26 => "Up".to_string(),
        0x27 => "Right".to_string(),
        0x28 => "Down".to_string(),
        0x2C => "PrintScreen".to_string(),
        0x2D => "Insert".to_string(),
        0x2E => "Delete".to_string(),
        0x30..=0x39 => format!("{}", (vk as u8) as char),
        0x41..=0x5A => format!("{}", (vk as u8) as char),
        0x60..=0x69 => format!("Num{}", vk - 0x60),
        0x70..=0x87 => format!("F{}", vk - 0x6F),
        0xA0 => "LShift".to_string(),
        0xA1 => "RShift".to_string(),
        0xA2 => "LCtrl".to_string(),
        0xA3 => "RCtrl".to_string(),
        0xA4 => "LAlt".to_string(),
        0xA5 => "RAlt".to_string(),
        0xBA => ";".to_string(),
        0xBB => "=".to_string(),
        0xBC => ",".to_string(),
        0xBD => "-".to_string(),
        0xBE => ".".to_string(),
        0xBF => "/".to_string(),
        0xC0 => "`".to_string(),
        0xDB => "[".to_string(),
        0xDC => "\\".to_string(),
        0xDD => "]".to_string(),
        0xDE => "'".to_string(),
        _ => format!("VK(0x{vk:02X})"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HotkeyBinding {
    pub action: HotkeyAction,
    pub vk: u16,
    pub modifiers: ModifierMask,
}

impl HotkeyBinding {
    pub fn display_combo(&self) -> String {
        format_hotkey(self.modifiers, self.vk)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowCandidate {
    pub hwnd: isize,
    pub title: String,
    pub process_name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathSummary {
    pub name: String,
    pub file_path: String,
    pub waypoint_count: usize,
    pub supported_count: usize,
    pub skipped_actions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskLimitsConfig {
    pub reward_claim_max_clicks: usize,
    pub reward_claim_timeout_ms: u64,
    pub auto_open_chest_timeout_ms: u64,
    pub auto_open_chest_target_lost_timeout_ms: u64,
    pub auto_open_chest_verify_timeout_ms: u64,
    pub auto_wood_total_timeout_ms: u64,
    pub auto_wood_gain_timeout_ms: u64,
    pub auto_wood_login_timeout_ms: u64,
    pub auto_wood_menu_timeout_ms: u64,
    pub auto_wood_logout_confirm_timeout_ms: u64,
    pub auto_wood_empty_ocr_limit: u8,
    #[serde(default = "default_fishing_timeout")]
    pub fishing_total_timeout_ms: u64,
    #[serde(default = "default_mining_max_shots")]
    pub mining_max_shots: u8,
    #[serde(default = "default_mining_timeout")]
    pub mining_total_timeout_ms: u64,
}

impl Default for TaskLimitsConfig {
    fn default() -> Self {
        Self {
            reward_claim_max_clicks: 30,
            reward_claim_timeout_ms: 30_000,
            auto_open_chest_timeout_ms: 60_000,
            auto_open_chest_target_lost_timeout_ms: 1500,
            auto_open_chest_verify_timeout_ms: 2000,
            auto_wood_total_timeout_ms: 120_000,
            auto_wood_gain_timeout_ms: 30_000,
            auto_wood_login_timeout_ms: 90_000,
            auto_wood_menu_timeout_ms: 10_000,
            auto_wood_logout_confirm_timeout_ms: 15_000,
            auto_wood_empty_ocr_limit: 3,
            fishing_total_timeout_ms: default_fishing_timeout(),
            mining_max_shots: default_mining_max_shots(),
            mining_total_timeout_ms: default_mining_timeout(),
        }
    }
}

impl TaskLimitsConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=100).contains(&self.reward_claim_max_clicks) {
            return Err("单页领取点击上限必须在 1..=100".into());
        }
        for (name, value, min, max) in [
            ("领取总超时", self.reward_claim_timeout_ms, 1000, 600_000),
            (
                "开箱总超时",
                self.auto_open_chest_timeout_ms,
                1000,
                3_600_000,
            ),
            (
                "开箱目标丢失超时",
                self.auto_open_chest_target_lost_timeout_ms,
                500,
                10_000,
            ),
            (
                "开箱验证超时",
                self.auto_open_chest_verify_timeout_ms,
                500,
                30_000,
            ),
            (
                "砍树总超时",
                self.auto_wood_total_timeout_ms,
                1000,
                3_600_000,
            ),
            (
                "砍树收益超时",
                self.auto_wood_gain_timeout_ms,
                1000,
                600_000,
            ),
            (
                "砍树登录超时",
                self.auto_wood_login_timeout_ms,
                1000,
                600_000,
            ),
            ("砍树菜单超时", self.auto_wood_menu_timeout_ms, 1000, 60_000),
            (
                "砍树退出确认超时",
                self.auto_wood_logout_confirm_timeout_ms,
                1000,
                60_000,
            ),
        ] {
            if !(min..=max).contains(&value) {
                return Err(format!("{name}必须在 {min}..={max} ms"));
            }
        }
        if !(1..=10).contains(&self.auto_wood_empty_ocr_limit) {
            return Err("砍树空 OCR 上限必须在 1..=10".into());
        }
        if !(1000..=3_600_000).contains(&self.fishing_total_timeout_ms) {
            return Err("钓鱼总超时必须在 1000..=3600000 ms".into());
        }
        if !(1..=20).contains(&self.mining_max_shots) {
            return Err("采矿射击次数上限必须在 1..=20".into());
        }
        if !(1000..=3_600_000).contains(&self.mining_total_timeout_ms) {
            return Err("采矿总超时必须在 1000..=3600000 ms".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreConfig {
    pub capture_mode: CaptureMode,
    pub preferred_process: String,
    pub preview_fps: u32,
    #[serde(default)]
    pub capture_only_no_render: bool,
    pub pathing_dir: String,
    pub assets_dir: String,
    #[serde(default = "default_pathing_runs")]
    pub pathing_runs: u32,
    #[serde(default = "default_pathing_loop_delay")]
    pub pathing_loop_delay_ms: u64,
    #[serde(default)]
    pub pathing_wait_for_loading: bool,
    #[serde(default)]
    pub pathing_auto_pick: bool,
    #[serde(default)]
    pub auto_skip: bool,
    #[serde(default)]
    pub auto_eat: bool,
    #[serde(default = "default_auto_cook_interval")]
    pub auto_cook_interval_ms: u64,
    #[serde(default = "default_true")]
    pub auto_cook_stop_on_recover: bool,
    #[serde(default = "default_auto_cook_timeout")]
    pub auto_cook_timeout_ms: u64,
    #[serde(default)]
    pub task_limits: TaskLimitsConfig,
    #[serde(default = "default_true")]
    pub fishing_dry_run: bool,
    pub hotkeys: Vec<HotkeyBinding>,
}

fn default_true() -> bool {
    true
}

fn default_fishing_timeout() -> u64 {
    // C# `AutoFishingConfig.WholeProcessTimeoutSeconds` default = 300.
    300_000
}

fn default_mining_max_shots() -> u8 {
    3
}

fn default_mining_timeout() -> u64 {
    180_000
}

fn default_pathing_runs() -> u32 {
    1
}

fn default_pathing_loop_delay() -> u64 {
    1000
}

fn default_auto_cook_interval() -> u64 {
    10
}

fn default_auto_cook_timeout() -> u64 {
    120_000
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            capture_mode: CaptureMode::BitBlt,
            preferred_process: "YuanShen.exe".to_string(),
            preview_fps: 10,
            capture_only_no_render: false,
            pathing_dir: String::new(),
            assets_dir: String::new(),
            pathing_runs: 1,
            pathing_loop_delay_ms: 1000,
            pathing_wait_for_loading: false,
            pathing_auto_pick: false,
            auto_skip: false,
            auto_eat: false,
            auto_cook_interval_ms: default_auto_cook_interval(),
            auto_cook_stop_on_recover: true,
            auto_cook_timeout_ms: default_auto_cook_timeout(),
            task_limits: TaskLimitsConfig::default(),
            fishing_dry_run: true,
            hotkeys: vec![
                HotkeyBinding {
                    action: HotkeyAction::StartStopPathing,
                    vk: 0x77, // F8
                    modifiers: ModifierMask::default(),
                },
                HotkeyBinding {
                    action: HotkeyAction::EmergencyStop,
                    vk: 0x78, // F9
                    modifiers: ModifierMask::default(),
                },
                HotkeyBinding {
                    action: HotkeyAction::TogglePreview,
                    vk: 0x79, // F10
                    modifiers: ModifierMask::default(),
                },
                HotkeyBinding {
                    action: HotkeyAction::TestPing,
                    vk: 0x50, // Ctrl+Shift+P
                    modifiers: ModifierMask {
                        ctrl: true,
                        shift: true,
                        ..Default::default()
                    },
                },
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CvDebugResult {
    pub updated_seq: u64,
    pub source_label: String,
    pub game_x: Option<f32>,
    pub game_y: Option<f32>,
    pub match_confidence: Option<f64>,
    pub floor: Option<i32>,
    pub camera_angle: Option<f32>,
    pub camera_confidence: Option<f32>,
    pub character_angle: Option<i32>,
    pub ocr_text: Option<String>,
    pub elapsed_ms: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CaptureStats {
    pub captured_frames: u64,
    pub source_width: u32,
    pub source_height: u32,
    pub last_capture_ms: f32,
    pub measured_fps: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct StageMetricsSnapshot {
    pub samples: u64,
    pub total_ms: f32,
    pub max_ms: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeMetricsSnapshot {
    pub submitted: u64,
    pub completed: u64,
    pub dropped: u64,
    pub deadline_hits: u64,
    pub deadline_misses: u64,
    pub deadline_hit_rate: f32,
    pub average_queue_delay_us: u64,
    pub max_queue_delay_us: u64,
    pub capture: StageMetricsSnapshot,
    pub recognition: StageMetricsSnapshot,
    pub decision: StageMetricsSnapshot,
    pub input: StageMetricsSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SalvagePreviewPosition {
    pub page: u8,
    pub slot: usize,
    pub row: u8,
    pub column: u8,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SalvagePreviewCandidate {
    /// Observation ID scoped to one preview session, not a persistent game item ID.
    pub id: u64,
    pub position: SalvagePreviewPosition,
    pub star: u8,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SalvagePreviewSkipStats {
    pub empty: usize,
    pub locked: usize,
    pub selected: usize,
    pub five_star: usize,
    pub overlapping: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SalvagePreviewStopReason {
    Complete,
    EmptyPage,
    PageLimit,
    NoProgress,
    InteractionBudget,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SalvagePreviewReport {
    pub candidates: Vec<SalvagePreviewCandidate>,
    pub skipped: SalvagePreviewSkipStats,
    pub pages_seen: u8,
    pub interactions_used: u16,
    pub no_progress: u8,
    pub selection_clicks: u16,
    pub salvage_clicks: u16,
    pub confirmation_clicks: u16,
    pub stop_reason: Option<SalvagePreviewStopReason>,
    pub failure: Option<String>,
    /// Always false. A preview report never authorizes destructive confirmation.
    pub final_confirmation_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerLeaseStatus {
    pub id: u64,
    pub name: String,
    pub options: serde_json::Value,
}

/// 配置组摘要（对应 C# `ScriptGroup` 的 UI 展示字段）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ScriptGroupSummary {
    pub name: String,
    pub index: i32,
    pub project_count: usize,
    pub enabled_count: usize,
    /// 项目明细（序号 / 名称 / 类型 / 是否启用），供 UI 展示与删除。
    #[serde(default)]
    pub projects: Vec<ScriptGroupProjectSummary>,
}

/// 配置组项目摘要。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ScriptGroupProjectSummary {
    pub index: i32,
    pub name: String,
    pub project_type: String,
    pub folder_name: String,
    pub enabled: bool,
}

/// 配置组当前执行进度。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ScriptGroupProgress {
    pub group_name: String,
    pub project_name: String,
    pub project_index: usize,
    pub project_total: usize,
    pub run_num: i32,
}

/// Read-only state snapshot published by `bgi-core` via `watch`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct UiSnapshot {
    pub task_state: TaskState,
    pub selected_window: Option<WindowCandidate>,
    pub available_windows: Vec<WindowCandidate>,
    pub raw_input_active: bool,
    pub raw_input_error: Option<String>,
    pub last_raw_key: Option<String>,
    pub capturing_hotkey_for: Option<HotkeyAction>,
    pub active_path_file: Option<String>,
    pub paths: Vec<PathSummary>,
    pub config: CoreConfig,
    pub preview_enabled: bool,
    #[serde(default)]
    pub triggers_active: bool,
    #[serde(default)]
    pub trigger_leases: Vec<TriggerLeaseStatus>,
    #[serde(default)]
    pub capture_stats: CaptureStats,
    #[serde(default)]
    pub runtime_metrics: RuntimeMetricsSnapshot,
    pub cv_debug: CvDebugResult,
    #[serde(default)]
    pub last_task_report: Option<TaskReport>,
    #[serde(default)]
    pub last_salvage_preview_report: Option<SalvagePreviewReport>,
    #[serde(default)]
    pub script_groups: Vec<ScriptGroupSummary>,
    #[serde(default)]
    pub script_group_progress: Option<ScriptGroupProgress>,
}

/// Separate image data plane (`ArcSwap` single-slot, newest frame wins).
#[derive(Debug, Clone)]
pub struct FramePreview {
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub capture_ms: f32,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Debug => "DBG",
            Self::Info => "INF",
            Self::Warn => "WRN",
            Self::Error => "ERR",
        }
    }

    pub fn rank(self) -> u8 {
        match self {
            Self::Debug => 0,
            Self::Info => 1,
            Self::Warn => 2,
            Self::Error => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    pub seq: u64,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub timestamp: String,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
}

/// Events sent over bounded mpsc from core to UI (`try_send` on core side).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreEvent {
    Log(LogEntry),
    HotkeyTriggered {
        action: HotkeyAction,
        combo: String,
    },
    RawKeyObserved {
        vk: u16,
        scan_code: u16,
        extended: bool,
        pressed: bool,
        combo: String,
    },
}

/// Commands sent from UI/Tray to core with optional completion receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreCommand {
    RefreshWindows,
    SelectWindow {
        hwnd: Option<isize>,
    },
    SetCaptureMode(CaptureMode),
    SetPreviewEnabled(bool),
    SetCaptureNoRender(bool),
    SetPreviewFps(u32),
    CaptureSingleFrame,
    StartFrameRecording(String),
    StopFrameRecording,
    StartFrameReplay(String),
    StopFrameReplay,
    SetPathingDir(String),
    SetPathingRuns(u32),
    SetPathingLoopDelayMs(u64),
    SetPathingWaitForLoading(bool),
    SetPathingAutoPick(bool),
    SetAutoSkipEnabled(bool),
    SetAutoEatEnabled(bool),
    SetAutoCookIntervalMs(u64),
    SetAutoCookStopOnRecover(bool),
    SetAutoCookTimeoutMs(u64),
    SetTaskLimits(TaskLimitsConfig),
    SetAssetsDir(String),
    RefreshPaths,
    SelectPath {
        file_path: String,
    },
    ProbeMiniMapCv,
    ProbeBigMapSift,
    ProbeOcrScreen,
    ProbeWeaponDetails,
    ProbeTalentDetails,
    ProbeCharacterAttributes,
    ProbeRewardResultCards,
    ProbeExpeditionStatus,
    ProbeMining,
    StartMining,
    BeginHotkeyCapture(Option<HotkeyAction>),
    UpdateHotkey(HotkeyBinding),
    TestInputKey {
        vk: u16,
    },
    StartBasicPathing,
    StartAutoCook,
    ClaimCurrentPageRewards,
    ClaimVisiblePageRewards,
    OpenCurrentChest,
    StartAutoWood {
        verify_gain: bool,
    },
    StartOneKeyExpedition,
    StartArtifactSalvagePreview,
    StartFishing {
        dry_run: bool,
    },
    SetFishingDryRun(bool),
    /// 配置组：刷新列表（从 User/ScriptGroup 重新加载）。
    RefreshScriptGroups,
    /// 配置组：按名称启动，可选从断点项目索引开始。
    StartScriptGroup {
        name: String,
        from_index: Option<i32>,
    },
    /// 配置组：停止并保留断点。
    StopScriptGroup,
    /// 配置组：新建空组。
    CreateScriptGroup {
        name: String,
    },
    /// 配置组：删除组（同时删除其 JSON 文件）。
    DeleteScriptGroup {
        name: String,
    },
    /// 配置组：新增一个项目。
    AddScriptGroupProject {
        group: String,
        project_type: String,
        folder_name: String,
        name: String,
    },
    /// 配置组：删除一个项目（按索引）。
    RemoveScriptGroupProject {
        group: String,
        index: i32,
    },
    StopPathing,
    OpenLogDir,
    EmergencyStop,
    ShutdownCore,
}

pub struct CommandEnvelope {
    pub cmd: CoreCommand,
    pub reply: oneshot::Sender<Result<(), String>>,
}

pub const MAX_LOG_HISTORY: usize = 1000;

pub type SharedPreview = Arc<ArcSwap<Option<Arc<FramePreview>>>>;
pub type UiWakeCallback = Arc<dyn Fn() + Send + Sync>;
pub type TimestampProvider = Arc<dyn Fn() -> String + Send + Sync>;
pub type LogMirrorCallback = Arc<dyn Fn(&LogEntry) + Send + Sync>;

/// Shared log sink bridging `CoreEndpoint::emit_log`, `tracing` layers, disk persistence, and UI.
#[derive(Clone)]
pub struct LogPublisher {
    event_tx: mpsc::Sender<CoreEvent>,
    waker_slot: UiWakerSlot,
    log_seq: Arc<AtomicU64>,
    start_instant: std::time::Instant,
    history: Arc<Mutex<VecDeque<LogEntry>>>,
    timestamp_fn: Arc<ArcSwap<Option<TimestampProvider>>>,
    mirror_slot: Arc<ArcSwap<Option<LogMirrorCallback>>>,
}

impl LogPublisher {
    /// Installs an optional wall-clock timestamp formatter (e.g. `HH:MM:SS.mmm`).
    pub fn set_timestamp_provider(&self, provider: Option<TimestampProvider>) {
        self.timestamp_fn.store(Arc::new(provider));
    }

    /// Installs an optional synchronous sink (such as rolling file + stderr mirror).
    /// Implementations must NOT call `tracing::*!` inside this callback.
    pub fn set_mirror(&self, mirror: Option<LogMirrorCallback>) {
        self.mirror_slot.store(Arc::new(mirror));
    }

    pub fn emit(&self, level: LogLevel, target: impl Into<String>, message: impl Into<String>) {
        let seq = self.log_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed_ms = self.start_instant.elapsed().as_millis() as u64;
        let timestamp = self
            .timestamp_fn
            .load()
            .as_ref()
            .as_ref()
            .map(|cb| cb())
            .unwrap_or_else(|| format!("{:>7}ms", elapsed_ms));
        let entry = LogEntry {
            seq,
            elapsed_ms,
            timestamp,
            level,
            target: target.into(),
            message: message.into(),
        };

        {
            let mut guard = self.history.lock().unwrap_or_else(|e| e.into_inner());
            if guard.len() >= MAX_LOG_HISTORY {
                guard.pop_front();
            }
            guard.push_back(entry.clone());
        }

        if let Some(mirror) = self.mirror_slot.load().as_ref() {
            mirror(&entry);
        }

        let _ = self.event_tx.try_send(CoreEvent::Log(entry));
        self.waker_slot.wake();
    }
}

/// Thread-safe UI repaint hook registered only while an `egui` window is open.
#[derive(Clone, Default)]
pub struct UiWakerSlot {
    inner: Arc<ArcSwap<Option<UiWakeCallback>>>,
}

impl UiWakerSlot {
    pub fn set(&self, waker: Option<UiWakeCallback>) {
        self.inner.store(Arc::new(waker));
    }

    pub fn wake(&self) {
        if let Some(cb) = self.inner.load().as_ref() {
            cb();
        }
    }
}

pub struct ControlClient {
    cmd_tx: mpsc::Sender<CommandEnvelope>,
    snapshot_rx: watch::Receiver<Arc<UiSnapshot>>,
    event_rx: mpsc::Receiver<CoreEvent>,
    preview_slot: SharedPreview,
    waker_slot: UiWakerSlot,
    log_history: Arc<Mutex<VecDeque<LogEntry>>>,
}

impl ControlClient {
    /// Non-blocking command dispatch so UI thread never stalls.
    pub fn send_cmd(
        &self,
        cmd: CoreCommand,
    ) -> Result<oneshot::Receiver<Result<(), String>>, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .try_send(CommandEnvelope {
                cmd,
                reply: reply_tx,
            })
            .map_err(|err| format!("控制通道拥塞或已关闭: {err}"))?;
        Ok(reply_rx)
    }

    /// Reads the latest snapshot and immediately drops the watch lock guard.
    pub fn snapshot(&mut self) -> Arc<UiSnapshot> {
        self.snapshot_rx.borrow_and_update().clone()
    }

    pub fn has_snapshot_update(&self) -> bool {
        self.snapshot_rx.has_changed().unwrap_or(false)
    }

    pub fn try_recv_event(&mut self) -> Option<CoreEvent> {
        self.event_rx.try_recv().ok()
    }

    pub fn recent_logs(&self) -> Vec<LogEntry> {
        self.log_history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    pub fn latest_preview(&self) -> Option<Arc<FramePreview>> {
        self.preview_slot.load_full().as_ref().clone()
    }

    pub fn waker_slot(&self) -> &UiWakerSlot {
        &self.waker_slot
    }
}

/// Lightweight command handle suitable for tray menus or non-UI callers.
#[derive(Clone)]
pub struct CommandDispatcher {
    cmd_tx: mpsc::Sender<CommandEnvelope>,
}

impl CommandDispatcher {
    pub fn try_send(
        &self,
        cmd: CoreCommand,
    ) -> Result<oneshot::Receiver<Result<(), String>>, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .try_send(CommandEnvelope {
                cmd,
                reply: reply_tx,
            })
            .map_err(|err| format!("控制通道拥塞或已关闭: {err}"))?;
        Ok(reply_rx)
    }
}

pub struct CoreEndpoint {
    pub cmd_rx: mpsc::Receiver<CommandEnvelope>,
    snapshot_tx: watch::Sender<Arc<UiSnapshot>>,
    event_tx: mpsc::Sender<CoreEvent>,
    preview_slot: SharedPreview,
    waker_slot: UiWakerSlot,
    log_publisher: LogPublisher,
}

impl CoreEndpoint {
    pub fn publish_snapshot(&self, snapshot: Arc<UiSnapshot>) {
        let _ = self.snapshot_tx.send(snapshot);
        self.waker_slot.wake();
    }

    pub fn publish_preview(&self, preview: FramePreview) {
        self.preview_slot.store(Arc::new(Some(Arc::new(preview))));
        self.waker_slot.wake();
    }

    pub fn clear_preview(&self) {
        self.preview_slot.store(Arc::new(None));
        self.waker_slot.wake();
    }

    pub fn log_publisher(&self) -> LogPublisher {
        self.log_publisher.clone()
    }

    /// Bounded non-blocking event emission (`try_send` never back-pressures core tasks).
    pub fn emit_event(&self, event: CoreEvent) {
        let _ = self.event_tx.try_send(event);
        self.waker_slot.wake();
    }

    pub fn emit_log(&self, level: LogLevel, target: impl Into<String>, message: impl Into<String>) {
        self.log_publisher.emit(level, target, message);
    }
}

pub fn create_control_bus(
    initial_snapshot: UiSnapshot,
) -> (ControlClient, CommandDispatcher, CoreEndpoint) {
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (snapshot_tx, snapshot_rx) = watch::channel(Arc::new(initial_snapshot));
    let (event_tx, event_rx) = mpsc::channel(512);
    let preview_slot: SharedPreview = Arc::new(ArcSwap::from_pointee(None));
    let waker_slot = UiWakerSlot::default();
    let log_history = Arc::new(Mutex::new(VecDeque::with_capacity(MAX_LOG_HISTORY)));
    let log_publisher = LogPublisher {
        event_tx: event_tx.clone(),
        waker_slot: waker_slot.clone(),
        log_seq: Arc::new(AtomicU64::new(0)),
        start_instant: std::time::Instant::now(),
        history: log_history.clone(),
        timestamp_fn: Arc::new(ArcSwap::from_pointee(None)),
        mirror_slot: Arc::new(ArcSwap::from_pointee(None)),
    };

    let dispatcher = CommandDispatcher {
        cmd_tx: cmd_tx.clone(),
    };
    let client = ControlClient {
        cmd_tx,
        snapshot_rx,
        event_rx,
        preview_slot: preview_slot.clone(),
        waker_slot: waker_slot.clone(),
        log_history,
    };
    let core = CoreEndpoint {
        cmd_rx,
        snapshot_tx,
        event_tx,
        preview_slot,
        waker_slot,
        log_publisher,
    };

    (client, dispatcher, core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_and_bounded_log_do_not_block() {
        let (mut client, dispatcher, core) = create_control_bus(UiSnapshot::default());

        // Overfill bounded event channel (>512) without reading; must not panic or block.
        for i in 0..600 {
            core.emit_log(LogLevel::Info, "test", format!("msg {i}"));
        }

        let mut received = 0;
        while client.try_recv_event().is_some() {
            received += 1;
        }
        assert_eq!(received, 512);

        // History retains all 600 entries even after channel overflow, and caps at MAX_LOG_HISTORY.
        let recent = client.recent_logs();
        assert_eq!(recent.len(), 600);
        assert_eq!(recent.last().map(|e| e.seq), Some(600));

        for i in 600..1200 {
            core.emit_log(LogLevel::Debug, "test", format!("msg {i}"));
        }
        let capped = client.recent_logs();
        assert_eq!(capped.len(), MAX_LOG_HISTORY);
        assert_eq!(capped.first().map(|e| e.seq), Some(201));
        assert_eq!(capped.last().map(|e| e.seq), Some(1200));

        let rx = dispatcher.try_send(CoreCommand::RefreshWindows).unwrap();
        drop(rx);
    }

    #[test]
    fn legacy_config_defaults_auto_cook_bounds() {
        let mut value = serde_json::to_value(CoreConfig::default()).unwrap();
        for key in [
            "auto_cook_interval_ms",
            "auto_cook_stop_on_recover",
            "auto_cook_timeout_ms",
        ] {
            value.as_object_mut().unwrap().remove(key);
        }
        value.as_object_mut().unwrap().remove("task_limits");
        let config: CoreConfig = serde_json::from_value(value).unwrap();
        assert_eq!(config.task_limits, TaskLimitsConfig::default());
        assert_eq!(config.auto_cook_interval_ms, 10);
        assert!(config.auto_cook_stop_on_recover);
        assert_eq!(config.auto_cook_timeout_ms, 120_000);
    }

    #[test]
    fn legacy_config_defaults_fishing_to_dry_run() {
        let mut value = serde_json::to_value(CoreConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("fishing_dry_run");
        value["task_limits"]
            .as_object_mut()
            .unwrap()
            .remove("fishing_total_timeout_ms");
        let config: CoreConfig = serde_json::from_value(value).unwrap();
        assert!(config.fishing_dry_run);
        assert_eq!(config.task_limits.fishing_total_timeout_ms, 300_000);
    }

    #[test]
    fn legacy_config_defaults_auto_pick_to_off() {
        assert!(!CoreConfig::default().pathing_auto_pick);
        let mut value = serde_json::to_value(CoreConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("pathing_auto_pick");
        let config: CoreConfig = serde_json::from_value(value).unwrap();
        assert!(!config.pathing_auto_pick);
    }

    #[test]
    fn task_limits_round_trip_and_reject_out_of_range_values() {
        let config = TaskLimitsConfig {
            reward_claim_max_clicks: 17,
            reward_claim_timeout_ms: 45_000,
            ..TaskLimitsConfig::default()
        };
        let restored: TaskLimitsConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(restored, config);

        for invalid in [
            TaskLimitsConfig {
                reward_claim_max_clicks: 0,
                ..Default::default()
            },
            TaskLimitsConfig {
                reward_claim_max_clicks: 101,
                ..Default::default()
            },
            TaskLimitsConfig {
                reward_claim_timeout_ms: 999,
                ..Default::default()
            },
            TaskLimitsConfig {
                auto_open_chest_target_lost_timeout_ms: 499,
                ..Default::default()
            },
            TaskLimitsConfig {
                auto_open_chest_verify_timeout_ms: 30_001,
                ..Default::default()
            },
            TaskLimitsConfig {
                auto_wood_menu_timeout_ms: 60_001,
                ..Default::default()
            },
            TaskLimitsConfig {
                auto_wood_empty_ocr_limit: 0,
                ..Default::default()
            },
            TaskLimitsConfig {
                auto_wood_empty_ocr_limit: 11,
                ..Default::default()
            },
            TaskLimitsConfig {
                fishing_total_timeout_ms: 999,
                ..Default::default()
            },
            TaskLimitsConfig {
                fishing_total_timeout_ms: 3_600_001,
                ..Default::default()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn trigger_lease_status_round_trips_and_legacy_snapshots_default_empty() {
        let snapshot = UiSnapshot {
            trigger_leases: vec![TriggerLeaseStatus {
                id: 7,
                name: "PathRequest".into(),
                options: serde_json::json!({"task_name": "Daily route", "request_every_n": 3}),
            }],
            ..Default::default()
        };
        let restored: UiSnapshot =
            serde_json::from_value(serde_json::to_value(&snapshot).unwrap()).unwrap();
        assert_eq!(restored.trigger_leases, snapshot.trigger_leases);

        let mut legacy = serde_json::to_value(snapshot).unwrap();
        legacy.as_object_mut().unwrap().remove("trigger_leases");
        let restored: UiSnapshot = serde_json::from_value(legacy).unwrap();
        assert!(restored.trigger_leases.is_empty());
    }

    #[test]
    fn running_and_stopping_states_keep_the_task_lease() {
        let running = TaskState::Pathing {
            task_id: 1,
            task_name: "route".into(),
            current_index: 1,
            total_points: 2,
        };
        let stopping = TaskState::Stopping {
            task_id: 1,
            task_name: "route".into(),
            current_index: 1,
            total_points: 2,
        };
        assert!(running.is_task_active());
        assert!(stopping.is_task_active());
        assert!(
            TaskState::OpeningChest {
                task_id: 2,
                task_name: "自动开箱当前目标".into(),
            }
            .is_task_active()
        );
        assert!(
            TaskState::AutoWood {
                task_id: 3,
                task_name: "自动砍树（王树瑞佑）".into(),
            }
            .is_task_active()
        );
        assert!(
            TaskState::AutoWood {
                task_id: 1,
                task_name: "wood".into(),
            }
            .is_task_active()
        );
        assert!(
            TaskState::OneKeyExpedition {
                task_id: 1,
                task_name: "expedition".into(),
            }
            .is_task_active()
        );
        assert!(
            TaskState::Fishing {
                task_id: 4,
                task_name: "自动钓鱼".into(),
            }
            .is_task_active()
        );
        assert!(!TaskState::Idle.is_task_active());
        assert!(!TaskState::Error("failed".into()).is_task_active());
    }

    #[test]
    fn hotkey_formatting_matches_expected_labels() {
        let combo = format_hotkey(
            ModifierMask {
                ctrl: true,
                shift: true,
                alt: false,
                win: false,
            },
            0x77,
        );
        assert_eq!(combo, "Ctrl+Shift+F8");
    }
}
