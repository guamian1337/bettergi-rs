use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Logical task state tracked by core and rendered in UI snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TaskState {
    #[default]
    Idle,
    Previewing,
    Pathing {
        task_name: String,
        current_index: usize,
        total_points: usize,
    },
    Error(String),
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
pub struct CoreConfig {
    pub capture_mode: CaptureMode,
    pub preferred_process: String,
    pub preview_fps: u32,
    #[serde(default)]
    pub capture_only_no_render: bool,
    pub pathing_dir: String,
    pub assets_dir: String,
    pub hotkeys: Vec<HotkeyBinding>,
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
    pub capture_stats: CaptureStats,
    pub cv_debug: CvDebugResult,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    pub seq: u64,
    pub elapsed_ms: u64,
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
    SelectWindow { hwnd: Option<isize> },
    SetCaptureMode(CaptureMode),
    SetPreviewEnabled(bool),
    SetCaptureNoRender(bool),
    SetPreviewFps(u32),
    CaptureSingleFrame,
    SetPathingDir(String),
    SetAssetsDir(String),
    RefreshPaths,
    SelectPath { file_path: String },
    ProbeMiniMapCv,
    ProbeBigMapSift,
    ProbeOcrScreen,
    BeginHotkeyCapture(Option<HotkeyAction>),
    UpdateHotkey(HotkeyBinding),
    TestInputKey { vk: u16 },
    StartBasicPathing,
    StopPathing,
    EmergencyStop,
    ShutdownCore,
}

pub struct CommandEnvelope {
    pub cmd: CoreCommand,
    pub reply: oneshot::Sender<Result<(), String>>,
}

pub type SharedPreview = Arc<ArcSwap<Option<Arc<FramePreview>>>>;
pub type UiWakeCallback = Arc<dyn Fn() + Send + Sync>;

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
}

impl ControlClient {
    /// Non-blocking command dispatch so UI thread never stalls.
    pub fn send_cmd(&self, cmd: CoreCommand) -> Result<oneshot::Receiver<Result<(), String>>, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .try_send(CommandEnvelope { cmd, reply: reply_tx })
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
    pub fn try_send(&self, cmd: CoreCommand) -> Result<oneshot::Receiver<Result<(), String>>, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .try_send(CommandEnvelope { cmd, reply: reply_tx })
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
    log_seq: AtomicU64,
    start_instant: std::time::Instant,
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

    /// Bounded non-blocking event emission (`try_send` never back-pressures core tasks).
    pub fn emit_event(&self, event: CoreEvent) {
        let _ = self.event_tx.try_send(event);
        self.waker_slot.wake();
    }

    pub fn emit_log(&self, level: LogLevel, target: impl Into<String>, message: impl Into<String>) {
        let seq = self.log_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed_ms = self.start_instant.elapsed().as_millis() as u64;
        self.emit_event(CoreEvent::Log(LogEntry {
            seq,
            elapsed_ms,
            level,
            target: target.into(),
            message: message.into(),
        }));
    }
}

pub fn create_control_bus(initial_snapshot: UiSnapshot) -> (ControlClient, CommandDispatcher, CoreEndpoint) {
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (snapshot_tx, snapshot_rx) = watch::channel(Arc::new(initial_snapshot));
    let (event_tx, event_rx) = mpsc::channel(512);
    let preview_slot: SharedPreview = Arc::new(ArcSwap::from_pointee(None));
    let waker_slot = UiWakerSlot::default();

    let dispatcher = CommandDispatcher {
        cmd_tx: cmd_tx.clone(),
    };
    let client = ControlClient {
        cmd_tx,
        snapshot_rx,
        event_rx,
        preview_slot: preview_slot.clone(),
        waker_slot: waker_slot.clone(),
    };
    let core = CoreEndpoint {
        cmd_rx,
        snapshot_tx,
        event_tx,
        preview_slot,
        waker_slot,
        log_seq: AtomicU64::new(0),
        start_instant: std::time::Instant::now(),
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

        let rx = dispatcher.try_send(CoreCommand::RefreshWindows).unwrap();
        drop(rx);
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
