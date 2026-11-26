use std::path::Path;
use std::time::{Instant, SystemTime};

use bgi_capture::{CapturedFrame, FrameBufferPool, FrameRecorder, FrameReplay};
use bgi_control::{CoreEndpoint, LogLevel, TriggerLeaseStatus, UiSnapshot};
use bgi_pathing::{OcrHit, PathingTaskConfig};
use bgi_task::{
    AutoPickConfig, AutoPickGateState, AutoPickMatcher, AutoPickPrompt, AutoPickPromptGate,
    Decision, DecisionAction, FrameContext, OcrCandidate, PauseController, RoiCenter, RuntimeError,
    Trigger, TriggerDispatcher, TriggerLease, TriggerLeaseRegistry, TriggerMode, UiCategory,
};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

#[path = "builtin_triggers.rs"]
mod builtin_triggers;

use builtin_triggers::install_builtin_triggers;

pub(crate) struct AutoPickExecutionContext {
    pub(crate) prompt: AutoPickPrompt,
    pub(crate) frame_size: (u32, u32),
    pub(crate) hwnd: isize,
    pub(crate) task_id: u64,
    pub(crate) observed_at: Instant,
}

pub(crate) struct TriggerRuntime {
    dispatcher: TriggerDispatcher,
    pool: FrameBufferPool,
    recorder: Option<FrameRecorder>,
    replay: Option<FrameReplay>,
    pending_pathing: Option<(String, String)>,
    pending_auto_pick: bool,
    auto_pick_config: Option<AutoPickConfig>,
    auto_pick_matcher: AutoPickMatcher,
    auto_pick_gate: AutoPickPromptGate,
    auto_skip: bool,
    auto_eat: bool,
    task_skip_lease: Option<TriggerLease>,
    task_eat_lease: Option<TriggerLease>,
}

impl TriggerRuntime {
    pub(crate) fn new(auto_skip: bool, auto_eat: bool) -> Self {
        let mut runtime = Self {
            dispatcher: TriggerDispatcher::new(TriggerMode::Config),
            pool: FrameBufferPool::new(2),
            recorder: None,
            replay: None,
            pending_pathing: None,
            pending_auto_pick: false,
            auto_pick_config: None,
            auto_pick_matcher: AutoPickMatcher::new(),
            auto_pick_gate: AutoPickPromptGate::default(),
            auto_skip,
            auto_eat,
            task_skip_lease: None,
            task_eat_lease: None,
        };
        runtime
            .install(install_builtin_triggers(auto_skip, auto_eat))
            .expect("安装内置触发器失败");
        runtime
    }

    pub(crate) fn has_triggers(&self) -> bool {
        self.dispatcher.has_triggers()
    }

    pub(crate) fn lease_status(&self) -> Vec<TriggerLeaseStatus> {
        self.dispatcher
            .leases()
            .snapshot()
            .into_iter()
            .map(|lease| TriggerLeaseStatus {
                id: lease.id,
                name: lease.name,
                options: lease.options,
            })
            .collect()
    }

    pub(crate) fn task_leases(&self) -> TriggerLeaseRegistry {
        self.dispatcher.leases().clone()
    }

    pub(crate) fn create_pathing_trigger_leases(
        &self,
        config: &PathingTaskConfig,
        auto_pick_available: bool,
    ) -> Result<Vec<TriggerLease>, String> {
        let mut leases = Vec::new();
        let path_request_options = if config
            .realtime_triggers
            .get(builtin_triggers::PATH_REQUEST_TRIGGER_NAME)
            .copied()
            .unwrap_or(false)
        {
            let options = config
                .realtime_trigger_options
                .get(builtin_triggers::PATH_REQUEST_TRIGGER_NAME)
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            builtin_triggers::validate_options(&options)
                .map_err(|e| format!("PathRequest 配置无效: {e}"))?;
            Some(options)
        } else {
            None
        };
        if auto_pick_available
            && config
                .realtime_triggers
                .get(builtin_triggers::AUTO_PICK_TRIGGER_NAME)
                .copied()
                .unwrap_or(true)
        {
            leases.push(
                self.task_leases()
                    .lease(
                        builtin_triggers::AUTO_PICK_TRIGGER_NAME,
                        serde_json::json!({"request_every_n": 4}),
                    )
                    .map_err(|e| format!("创建自动拾取触发器租约失败: {e}"))?,
            );
        }
        if let Some(options) = path_request_options {
            leases.push(
                self.task_leases()
                    .lease(builtin_triggers::PATH_REQUEST_TRIGGER_NAME, options)
                    .map_err(|e| format!("创建路径请求触发器租约失败: {e}"))?,
            );
        }
        Ok(leases)
    }

    pub(crate) fn has_pending_pathing(&self) -> bool {
        self.pending_pathing.is_some()
    }

    pub(crate) fn task_pauses(&self) -> PauseController {
        self.dispatcher.pauses().clone()
    }

    pub(crate) fn set_auto_pick_config(&mut self, config: Option<AutoPickConfig>) {
        self.auto_pick_config = config;
        self.auto_pick_matcher = AutoPickMatcher::new();
        self.auto_pick_gate.reset();
        self.pending_auto_pick = false;
    }

    pub(crate) fn take_auto_pick_request(&mut self) -> bool {
        std::mem::take(&mut self.pending_auto_pick)
    }

    pub(crate) fn observe_auto_pick_prompt(&mut self, identity: Option<u64>) -> AutoPickGateState {
        self.auto_pick_gate.observe(identity, Instant::now())
    }

    pub(crate) fn apply_auto_pick_hits(
        &mut self,
        context: &AutoPickExecutionContext,
        active_task_id: Option<u64>,
        selected_hwnd: Option<isize>,
        hits: Vec<OcrHit>,
        cancelled: bool,
        core: &CoreEndpoint,
    ) -> usize {
        use bgi_task::{AUTO_PICK_MAX_OBSERVATION_AGE, AutoPickGateState};

        if context.task_id == 0
            || active_task_id != Some(context.task_id)
            || selected_hwnd != Some(context.hwnd)
            || context.frame_size.0 == 0
            || context.frame_size.1 == 0
            || cancelled
            || self.dispatcher.pauses().is_paused()
            || context.observed_at.elapsed() > AUTO_PICK_MAX_OBSERVATION_AGE
            || self.observe_auto_pick_prompt(Some(context.prompt.identity))
                != AutoPickGateState::Ready(context.prompt.identity)
        {
            return 0;
        }
        let Some(config) = &self.auto_pick_config else {
            return 0;
        };
        let accepted = hits.into_iter().any(|hit| {
            self.auto_pick_matcher
                .decide(
                    &OcrCandidate {
                        text: hit.text,
                        confidence: f64::from(hit.confidence),
                        roi_center: RoiCenter {
                            x: f64::from(hit.center_x) / f64::from(context.frame_size.0),
                            y: f64::from(hit.center_y) / f64::from(context.frame_size.1),
                        },
                    },
                    config,
                    Instant::now(),
                )
                .should_interact()
        });
        if !accepted {
            return 0;
        }
        let Ok(Some(_pause)) = self.dispatcher.pauses().try_pause() else {
            return 0;
        };
        if bgi_input::key_tap_checked_if(context.hwnd, 0x46, 80, context.frame_size, || {
            active_task_id == Some(context.task_id)
                && selected_hwnd == Some(context.hwnd)
                && !cancelled
                && context.observed_at.elapsed() <= AUTO_PICK_MAX_OBSERVATION_AGE
                && _pause.is_active()
        })
        .is_err()
        {
            return 0;
        }
        self.auto_pick_gate.mark_serviced(context.prompt.identity);
        core.emit_log(
            LogLevel::Debug,
            "AutoPick",
            "实时触发器完成 1 次正例白名单拾取",
        );
        1
    }

    pub(crate) fn set_builtin_config(&mut self, auto_skip: bool, auto_eat: bool) {
        let skip_changed = self.auto_skip != auto_skip;
        let eat_changed = self.auto_eat != auto_eat;
        self.auto_skip = auto_skip;
        self.auto_eat = auto_eat;
        self.dispatcher
            .set_enabled(&[builtin_triggers::AUTO_SKIP_TRIGGER_NAME], auto_skip);
        self.dispatcher
            .set_enabled(&[builtin_triggers::AUTO_EAT_TRIGGER_NAME], auto_eat);
        if self.dispatcher.mode() == TriggerMode::Task {
            if skip_changed {
                self.set_task_builtin_lease(builtin_triggers::AUTO_SKIP_TRIGGER_NAME, auto_skip);
            }
            if eat_changed {
                self.set_task_builtin_lease(builtin_triggers::AUTO_EAT_TRIGGER_NAME, auto_eat);
            }
        }
    }

    fn set_task_builtin_lease(&mut self, name: &str, enabled: bool) {
        let slot = if name == builtin_triggers::AUTO_SKIP_TRIGGER_NAME {
            &mut self.task_skip_lease
        } else {
            &mut self.task_eat_lease
        };
        slot.take();
        if enabled {
            *slot = Some(
                self.dispatcher
                    .leases()
                    .lease(name, serde_json::Value::Null)
                    .expect("task lease overflow"),
            );
        }
    }

    pub(crate) fn set_auto_eat_observation(&mut self, observation: bgi_task::AutoEatObservation) {
        self.dispatcher.set_auto_eat_observation(observation);
    }

    pub(crate) fn set_task_mode(&mut self, enabled: bool) {
        let mode = if enabled {
            TriggerMode::Task
        } else {
            TriggerMode::Config
        };
        if self.dispatcher.mode() == mode {
            return;
        }
        self.dispatcher.set_mode(mode);
        self.set_task_builtin_lease(
            builtin_triggers::AUTO_SKIP_TRIGGER_NAME,
            enabled && self.auto_skip,
        );
        self.set_task_builtin_lease(
            builtin_triggers::AUTO_EAT_TRIGGER_NAME,
            enabled && self.auto_eat,
        );
    }

    pub(crate) fn request_pathing(
        &mut self,
        name: &str,
        snapshot: &UiSnapshot,
    ) -> Result<(), String> {
        if self.pending_pathing.is_some() {
            return Err("已有寻路请求待处理，拒绝重复请求".into());
        }

        let path = snapshot
            .paths
            .iter()
            .find(|path| path.name == name)
            .ok_or_else(|| format!("未找到名称精确匹配的寻路路径: {name}"))?;
        if path.file_path.trim().is_empty() {
            return Err(format!("寻路路径 {name} 的文件路径为空"));
        }
        self.pending_pathing = Some((name.to_string(), path.file_path.clone()));
        Ok(())
    }

    pub(crate) fn take_pathing_request(&mut self) -> Option<(String, String)> {
        self.pending_pathing.take()
    }

    pub(crate) fn clear_pending_pathing(&mut self) {
        self.pending_pathing = None;
    }

    pub(crate) fn install(&mut self, triggers: Vec<Box<dyn Trigger>>) -> Result<(), RuntimeError> {
        self.dispatcher.replace_triggers(triggers)
    }

    pub(crate) fn start_recording(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        if self.recorder.is_some() {
            return Err("帧录制已在进行".into());
        }
        self.recorder = Some(FrameRecorder::create(path).map_err(|e| e.to_string())?);
        Ok(())
    }

    pub(crate) fn stop_recording(&mut self) -> Result<(), String> {
        let Some(recorder) = self.recorder.take() else {
            return Ok(());
        };
        recorder.finish().map_err(|e| e.to_string()).map(|_| ())
    }

    pub(crate) fn set_replay(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        self.replay = Some(FrameReplay::open(path).map_err(|e| e.to_string())?);
        Ok(())
    }

    pub(crate) fn clear_replay(&mut self) {
        self.replay = None;
    }

    pub(crate) fn is_replaying(&self) -> bool {
        self.replay.is_some()
    }

    pub(crate) fn next_frame(&mut self) -> Result<Option<(u64, CapturedFrame)>, String> {
        let Some(replay) = self.replay.as_mut() else {
            return Ok(None);
        };
        replay
            .next_frame()
            .map_err(|e| e.to_string())
            .map(|record| {
                record.map(|record| {
                    let seq = record.seq;
                    let frame = record.into_frame(&self.pool);
                    (seq, frame)
                })
            })
    }

    pub(crate) fn dispatch(
        &mut self,
        seq: u64,
        frame: &CapturedFrame,
        snapshot: &UiSnapshot,
        ui: UiCategory,
        core: &CoreEndpoint,
    ) -> Result<Vec<Decision>, RuntimeError> {
        if let Some(recorder) = self.recorder.as_mut() {
            recorder
                .record_frame(seq, SystemTime::now(), frame)
                .map_err(|e| RuntimeError::Trigger {
                    name: "FrameRecorder".into(),
                    callback: "record",
                    message: e.to_string(),
                })?;
        }
        let foreground = snapshot.selected_window.as_ref().is_some_and(|window| {
            let hwnd = HWND(window.hwnd as *mut _);
            let active = unsafe { GetForegroundWindow() };
            !active.is_invalid() && active == hwnd
        });
        let context = FrameContext {
            frame_id: seq,
            captured_at: Instant::now(),
            pixels: frame.data.as_slice(),
            width: frame.width,
            height: frame.height,
            bgr24: frame.format.is_bgr24(),
            ui,
            foreground,
            requires_foreground: true,
        };
        let decisions = self.dispatcher.tick(&context, Instant::now())?;
        for failure in self.dispatcher.take_failures() {
            core.emit_log(
                LogLevel::Warn,
                "Trigger",
                format!(
                    "{} {} 失败: {}",
                    failure.name, failure.callback, failure.message
                ),
            );
        }
        Ok(decisions)
    }

    pub(crate) fn process_decisions(
        &mut self,
        decisions: Vec<Decision>,
        snapshot: &UiSnapshot,
        core: &CoreEndpoint,
    ) {
        for decision in decisions {
            match decision.action {
                DecisionAction::Log(message) => core.emit_log(LogLevel::Info, "Trigger", message),
                DecisionAction::RequestTask(name) => match self.request_pathing(&name, snapshot) {
                    Ok(()) => core.emit_log(
                        LogLevel::Info,
                        "Trigger",
                        format!("寻路任务请求已接受: {name}"),
                    ),
                    Err(error) => core.emit_log(
                        LogLevel::Warn,
                        "Trigger",
                        format!("寻路任务请求 {name} 失败: {error}"),
                    ),
                },
                DecisionAction::Custom(name)
                    if name == "AutoEatRecover" || name == "AutoEatRevive" =>
                {
                    let _pause = self
                        .dispatcher
                        .pauses()
                        .pause()
                        .expect("pause owner overflow");
                    bgi_input::key_tap(0x5A, 80);
                    std::thread::sleep(std::time::Duration::from_millis(120));
                    let message = if name == "AutoEatRevive" {
                        "检测到复活图标，按 Z 复活"
                    } else {
                        "检测到低血与营养袋，按 Z 进食"
                    };
                    core.emit_log(LogLevel::Info, "AutoEat", message);
                }
                DecisionAction::Custom(name)
                    if name == builtin_triggers::AUTO_SKIP_TRIGGER_NAME =>
                {
                    let _pause = self
                        .dispatcher
                        .pauses()
                        .pause()
                        .expect("pause owner overflow");
                    bgi_input::key_tap(0x20, 80);
                }
                DecisionAction::Custom(name)
                    if name == builtin_triggers::AUTO_PICK_TRIGGER_NAME =>
                {
                    self.pending_auto_pick = true;
                }
                DecisionAction::Custom(name) => core.emit_log(
                    LogLevel::Debug,
                    "Trigger",
                    format!("自定义触发器动作: {name}"),
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use bgi_control::{PathSummary, TaskState, create_control_bus};
    use builtin_triggers::{PATH_REQUEST_TRIGGER_NAME, PathRequestTrigger};
    use serde_json::{Value, json};

    fn frame(frame_id: u64, foreground: bool) -> FrameContext<'static> {
        FrameContext {
            frame_id,
            captured_at: Instant::now(),
            pixels: &[],
            width: 1920,
            height: 1080,
            bgr24: true,
            ui: UiCategory::Unknown,
            foreground,
            requires_foreground: true,
        }
    }

    fn tick(runtime: &mut TriggerRuntime, frame_id: u64, foreground: bool) -> Vec<Decision> {
        runtime
            .dispatcher
            .tick(&frame(frame_id, foreground), Instant::now())
            .unwrap()
    }

    fn path_snapshot() -> UiSnapshot {
        UiSnapshot {
            paths: vec![PathSummary {
                name: "Daily route".into(),
                file_path: r"D:\paths\daily.json".into(),
                waypoint_count: 2,
                supported_count: 2,
                skipped_actions: Vec::new(),
            }],
            ..UiSnapshot::default()
        }
    }

    #[test]
    fn builtin_switches_preserve_state_and_follow_task_mode() {
        use builtin_triggers::{AUTO_EAT_TRIGGER_NAME, AUTO_SKIP_TRIGGER_NAME};
        let mut runtime = TriggerRuntime::new(true, true);
        let mut context = frame(1, true);
        context.ui = UiCategory::Talk;
        assert_eq!(
            runtime
                .dispatcher
                .tick(&context, Instant::now())
                .unwrap()
                .len(),
            1
        );
        runtime.set_builtin_config(true, false);
        assert!(
            runtime
                .dispatcher
                .tick(&context, Instant::now())
                .unwrap()
                .is_empty()
        );
        runtime.set_builtin_config(true, true);
        context.captured_at += std::time::Duration::from_millis(250);
        assert_eq!(
            runtime
                .dispatcher
                .tick(&context, Instant::now())
                .unwrap()
                .len(),
            1
        );

        runtime.set_task_mode(true);
        let names: Vec<_> = runtime
            .task_leases()
            .snapshot()
            .into_iter()
            .map(|lease| lease.name)
            .collect();
        assert_eq!(names, [AUTO_SKIP_TRIGGER_NAME, AUTO_EAT_TRIGGER_NAME]);
        assert!(runtime.has_triggers());
        let eat_lease_id = runtime
            .task_leases()
            .snapshot()
            .into_iter()
            .find(|lease| lease.name == AUTO_EAT_TRIGGER_NAME)
            .unwrap()
            .id;
        runtime.set_builtin_config(false, true);
        let names: Vec<_> = runtime
            .task_leases()
            .snapshot()
            .into_iter()
            .map(|lease| lease.name)
            .collect();
        assert_eq!(names, [AUTO_EAT_TRIGGER_NAME]);
        assert_eq!(
            runtime
                .task_leases()
                .snapshot()
                .into_iter()
                .find(|lease| lease.name == AUTO_EAT_TRIGGER_NAME)
                .unwrap()
                .id,
            eat_lease_id
        );
        runtime.set_task_mode(false);
        assert!(runtime.task_leases().snapshot().is_empty());
    }

    #[test]
    fn auto_eat_requires_gameplay_foreground_and_observed_recovery() {
        let mut runtime = TriggerRuntime::new(false, true);
        let mut context = frame(1, true);
        context.ui = UiCategory::Gameplay;
        assert!(
            runtime
                .dispatcher
                .tick(&context, Instant::now())
                .unwrap()
                .is_empty()
        );
        runtime.set_auto_eat_observation(bgi_task::AutoEatObservation {
            low_hp: true,
            recovery_icon: true,
            resurrection_icon: false,
        });
        let decisions = runtime.dispatcher.tick(&context, Instant::now()).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].action,
            DecisionAction::Custom("AutoEatRecover".into())
        );
        context.foreground = false;
        assert!(
            runtime
                .dispatcher
                .tick(&context, context.captured_at)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn auto_skip_is_talk_only_foreground_only_and_rate_limited() {
        use builtin_triggers::AUTO_SKIP_TRIGGER_NAME;
        let mut runtime = TriggerRuntime::new(true, false);
        assert!(runtime.has_triggers());
        let mut context = frame(1, true);
        context.ui = UiCategory::Gameplay;
        let now = Instant::now();
        assert!(runtime.dispatcher.tick(&context, now).unwrap().is_empty());
        context.ui = UiCategory::Talk;
        context.foreground = false;
        assert!(runtime.dispatcher.tick(&context, now).unwrap().is_empty());
        context.foreground = true;
        let decisions = runtime.dispatcher.tick(&context, now).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].trigger, AUTO_SKIP_TRIGGER_NAME);
        assert!(runtime.dispatcher.tick(&context, now).unwrap().is_empty());
        runtime.set_builtin_config(false, false);
        assert!(!runtime.has_triggers());
    }

    #[test]
    fn builtin_path_request_is_installed_and_requires_task_mode_and_lease() {
        let mut runtime = TriggerRuntime::new(false, false);
        assert!(!runtime.has_triggers());
        assert_eq!(runtime.dispatcher.mode(), TriggerMode::Config);

        let leases = runtime.task_leases();
        let lease = leases
            .lease(
                PATH_REQUEST_TRIGGER_NAME,
                json!({"task_name": "Daily route", "request_every_n": 3}),
            )
            .unwrap();
        assert!(tick(&mut runtime, 6, true).is_empty());

        runtime.set_task_mode(true);
        assert_eq!(
            tick(&mut runtime, 6, true),
            vec![Decision {
                trigger: PATH_REQUEST_TRIGGER_NAME.into(),
                action: DecisionAction::RequestTask("Daily route".into()),
            }]
        );
        assert!(tick(&mut runtime, 7, true).is_empty());
        assert!(tick(&mut runtime, 6, false).is_empty());

        let pauses = runtime.task_pauses();
        let pause = pauses.pause().unwrap();
        assert!(tick(&mut runtime, 9, true).is_empty());
        drop(pause);
        assert_eq!(tick(&mut runtime, 9, true).len(), 1);

        drop(lease);
        assert!(tick(&mut runtime, 12, true).is_empty());
        runtime.set_task_mode(false);
        assert_eq!(runtime.dispatcher.mode(), TriggerMode::Config);
    }

    #[test]
    fn pathing_trigger_leases_are_explicit_and_preserve_auto_pick() {
        let mut runtime = TriggerRuntime::new(false, false);
        let mut config = bgi_pathing::PathingTaskConfig::default();
        assert!(
            runtime
                .create_pathing_trigger_leases(&config, false)
                .unwrap()
                .is_empty()
        );

        config
            .realtime_triggers
            .insert(PATH_REQUEST_TRIGGER_NAME.into(), true);
        config.realtime_trigger_options.insert(
            PATH_REQUEST_TRIGGER_NAME.into(),
            json!({"task_name": "Daily route", "request_every_n": 3}),
        );
        let mut leases = runtime
            .create_pathing_trigger_leases(&config, true)
            .unwrap();
        assert_eq!(leases.len(), 2);
        assert_eq!(leases[0].name(), builtin_triggers::AUTO_PICK_TRIGGER_NAME);
        assert_eq!(leases[1].name(), PATH_REQUEST_TRIGGER_NAME);
        let status = runtime.lease_status();
        assert_eq!(status.len(), 2);
        assert_eq!(status[1].name, PATH_REQUEST_TRIGGER_NAME);
        assert_eq!(status[1].options["task_name"], "Daily route");
        assert_eq!(status[1].options["request_every_n"], 3);

        runtime.set_task_mode(true);
        let decisions = tick(&mut runtime, 12, true);
        assert!(decisions.iter().any(|decision| {
            decision.action == DecisionAction::RequestTask("Daily route".into())
        }));
        assert!(decisions.iter().any(|decision| {
            decision.action
                == DecisionAction::Custom(builtin_triggers::AUTO_PICK_TRIGGER_NAME.into())
        }));

        leases.pop();
        assert_eq!(runtime.lease_status().len(), 1);
        let decisions = tick(&mut runtime, 12, true);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].trigger,
            builtin_triggers::AUTO_PICK_TRIGGER_NAME
        );
        leases.clear();
        assert!(!runtime.has_triggers());
        assert!(runtime.lease_status().is_empty());
    }

    #[test]
    fn path_request_lease_rejects_invalid_options_without_side_effects() {
        let runtime = TriggerRuntime::new(false, false);
        let mut config = bgi_pathing::PathingTaskConfig::default();
        config
            .realtime_triggers
            .insert(PATH_REQUEST_TRIGGER_NAME.into(), true);
        for options in [
            Value::Null,
            json!({"task_name": "Daily route"}),
            json!({"task_name": " ", "request_every_n": 3}),
            json!({"task_name": "Daily route", "request_every_n": 0}),
            json!({"task_name": "Daily route", "request_every_n": -1}),
            json!({"task_name": "Daily route", "request_every_n": 3, "extra": true}),
        ] {
            config
                .realtime_trigger_options
                .insert(PATH_REQUEST_TRIGGER_NAME.into(), options);
            assert!(
                runtime
                    .create_pathing_trigger_leases(&config, true)
                    .is_err()
            );
            assert!(runtime.dispatcher.leases().snapshot().is_empty());
        }
    }

    #[test]
    fn path_request_lease_is_released_during_unwind() {
        let runtime = TriggerRuntime::new(false, false);
        let mut config = bgi_pathing::PathingTaskConfig::default();
        config
            .realtime_triggers
            .insert(PATH_REQUEST_TRIGGER_NAME.into(), true);
        config.realtime_trigger_options.insert(
            PATH_REQUEST_TRIGGER_NAME.into(),
            json!({"task_name": "Daily route", "request_every_n": 3}),
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = runtime
                .create_pathing_trigger_leases(&config, false)
                .unwrap();
            panic!("injected pathing panic");
        }));
        assert!(result.is_err());
        assert!(runtime.dispatcher.leases().snapshot().is_empty());
    }

    #[test]
    fn task_lease_activates_auto_pick_and_releases_capture_demand() {
        let mut runtime = TriggerRuntime::new(false, false);
        let leases = runtime.task_leases();
        runtime.set_task_mode(true);
        assert!(!runtime.has_triggers());
        let lease = leases
            .lease(
                builtin_triggers::AUTO_PICK_TRIGGER_NAME,
                json!({"request_every_n": 4}),
            )
            .unwrap();
        assert!(runtime.has_triggers());
        let (_, _, core) = create_control_bus(UiSnapshot::default());
        let decisions = tick(&mut runtime, 4, true);
        runtime.process_decisions(decisions, &UiSnapshot::default(), &core);
        assert!(runtime.take_auto_pick_request());
        assert!(!runtime.take_auto_pick_request());
        drop(lease);
        assert!(!runtime.has_triggers());
    }

    #[test]
    fn path_request_options_are_strictly_validated() {
        let mut trigger = PathRequestTrigger::new();
        let invalid = vec![
            Value::Null,
            json!({}),
            json!({"task_name": "Daily route"}),
            json!({"request_every_n": 3}),
            json!({"task_name": 1, "request_every_n": 3}),
            json!({"task_name": "   ", "request_every_n": 3}),
            json!({"task_name": "Daily route", "request_every_n": 0}),
            json!({"task_name": "Daily route", "request_every_n": -1}),
            json!({"task_name": "Daily route", "request_every_n": 1.0}),
            json!({"task_name": "Daily route", "request_every_n": "3"}),
            json!({"task_name": "Daily route", "request_every_n": 3, "extra": true}),
        ];
        for options in invalid {
            assert!(trigger.on_enable(&options).is_err(), "{options}");
            assert!(trigger.on_capture(&frame(3, true)).unwrap().is_empty());
        }

        trigger
            .on_enable(&json!({"task_name": "Daily route", "request_every_n": 3}))
            .unwrap();
        assert_eq!(trigger.on_capture(&frame(6, true)).unwrap().len(), 1);
        assert!(trigger.on_capture(&frame(7, true)).unwrap().is_empty());
        assert!(trigger.on_capture(&frame(6, false)).unwrap().is_empty());
    }

    fn auto_pick_context() -> AutoPickExecutionContext {
        AutoPickExecutionContext {
            prompt: AutoPickPrompt {
                identity: 7,
                text_roi: bgi_task::AutoPickTextRoi {
                    x0: 0.1,
                    y0: 0.1,
                    x1: 0.2,
                    y1: 0.2,
                },
            },
            frame_size: (1920, 1080),
            hwnd: 0,
            task_id: 9,
            observed_at: Instant::now(),
        }
    }

    fn auto_pick_hits() -> Vec<OcrHit> {
        vec![OcrHit {
            text: "矿".into(),
            center_x: 200,
            center_y: 200,
            confidence: 0.99,
        }]
    }

    fn auto_pick_runtime() -> TriggerRuntime {
        let mut runtime = TriggerRuntime::new(false, false);
        let mut config = AutoPickConfig::default();
        config.whitelist.push("矿".into());
        runtime.set_auto_pick_config(Some(config));
        runtime
    }

    #[test]
    fn auto_pick_completion_rejects_invalid_or_unstable_context() {
        for case in [
            "task",
            "window",
            "cancelled",
            "stale",
            "paused",
            "unstable",
            "serviced",
            "no-config",
            "no-hits",
            "rejected-hits",
        ] {
            let mut runtime = auto_pick_runtime();
            let mut context = auto_pick_context();
            let (_, _, core) = create_control_bus(UiSnapshot::default());
            let mut task_id = Some(context.task_id);
            let mut selected_hwnd = Some(context.hwnd);
            let mut cancelled = false;
            let mut hits = auto_pick_hits();
            let mut _pause = None;
            match case {
                "task" => task_id = Some(context.task_id + 1),
                "window" => selected_hwnd = Some(context.hwnd + 1),
                "cancelled" => cancelled = true,
                "stale" => {
                    context.observed_at -=
                        bgi_task::AUTO_PICK_MAX_OBSERVATION_AGE + Duration::from_millis(1);
                }
                "paused" => {
                    _pause = Some(runtime.task_pauses().pause().unwrap());
                }
                "unstable" => {
                    assert_eq!(
                        runtime.observe_auto_pick_prompt(Some(context.prompt.identity)),
                        AutoPickGateState::Idle
                    );
                }
                "serviced" => {
                    for _ in 0..2 {
                        runtime.observe_auto_pick_prompt(Some(context.prompt.identity));
                    }
                    assert_eq!(
                        runtime.observe_auto_pick_prompt(Some(context.prompt.identity)),
                        AutoPickGateState::Ready(context.prompt.identity)
                    );
                    runtime
                        .auto_pick_gate
                        .mark_serviced(context.prompt.identity);
                }
                "no-config" => runtime.set_auto_pick_config(None),
                "no-hits" => hits.clear(),
                "rejected-hits" => hits[0].text = "聊天".into(),
                _ => unreachable!(),
            }
            if !matches!(case, "unstable" | "serviced") {
                for _ in 0..2 {
                    runtime.observe_auto_pick_prompt(Some(context.prompt.identity));
                }
            }
            assert_eq!(
                runtime.apply_auto_pick_hits(
                    &context,
                    task_id,
                    selected_hwnd,
                    hits,
                    cancelled,
                    &core,
                ),
                0,
                "case={case}"
            );
            assert_eq!(
                runtime.observe_auto_pick_prompt(Some(context.prompt.identity))
                    == AutoPickGateState::Serviced(context.prompt.identity),
                case == "serviced",
                "case={case}"
            );
        }
    }

    #[test]
    fn failed_checked_input_does_not_service_auto_pick_prompt() {
        let mut runtime = auto_pick_runtime();
        let context = auto_pick_context();
        let (_, _, core) = create_control_bus(UiSnapshot::default());
        for _ in 0..2 {
            runtime.observe_auto_pick_prompt(Some(context.prompt.identity));
        }
        assert_eq!(
            runtime.apply_auto_pick_hits(
                &context,
                Some(context.task_id),
                Some(context.hwnd),
                auto_pick_hits(),
                false,
                &core,
            ),
            0
        );
        assert_eq!(
            runtime.observe_auto_pick_prompt(Some(context.prompt.identity)),
            AutoPickGateState::Ready(context.prompt.identity)
        );
    }

    #[test]
    fn auto_pick_config_reset_rearms_only_at_task_boundary() {
        let mut runtime = auto_pick_runtime();
        let identity = 7;
        for _ in 0..2 {
            runtime.observe_auto_pick_prompt(Some(identity));
        }
        runtime.auto_pick_gate.mark_serviced(identity);
        assert_eq!(
            runtime.observe_auto_pick_prompt(Some(identity)),
            AutoPickGateState::Serviced(identity)
        );
        runtime.set_auto_pick_config(runtime.auto_pick_config.clone());
        assert_eq!(
            runtime.observe_auto_pick_prompt(Some(identity)),
            AutoPickGateState::Idle
        );
    }

    #[test]
    fn pathing_request_resolves_an_exact_name() {
        let mut runtime = TriggerRuntime::new(false, false);
        let snapshot = path_snapshot();

        runtime.request_pathing("Daily route", &snapshot).unwrap();
        assert_eq!(
            runtime.take_pathing_request(),
            Some(("Daily route".into(), r"D:\paths\daily.json".into()))
        );
        assert_eq!(runtime.take_pathing_request(), None);
        assert!(runtime.request_pathing("daily route", &snapshot).is_err());
        assert!(runtime.request_pathing("Daily", &snapshot).is_err());
        assert!(runtime.request_pathing("missing", &snapshot).is_err());

        let mut empty_file = snapshot.clone();
        empty_file.paths[0].file_path.clear();
        assert!(runtime.request_pathing("Daily route", &empty_file).is_err());
    }

    #[test]
    fn pathing_request_queues_during_active_and_stopping_tasks() {
        let mut runtime = TriggerRuntime::new(false, false);
        let mut snapshot = path_snapshot();

        for task_state in [
            TaskState::Pathing {
                task_id: 7,
                task_name: "Daily route".into(),
                current_index: 0,
                total_points: 2,
            },
            TaskState::Stopping {
                task_id: 7,
                task_name: "Daily route".into(),
                current_index: 0,
                total_points: 2,
            },
        ] {
            snapshot.task_state = task_state;
            runtime.request_pathing("Daily route", &snapshot).unwrap();
            assert!(runtime.has_pending_pathing());
            assert!(runtime.take_pathing_request().is_some());
            assert!(!runtime.has_pending_pathing());
        }

        runtime.request_pathing("Daily route", &snapshot).unwrap();
        let error = runtime
            .request_pathing("Daily route", &snapshot)
            .unwrap_err();
        assert!(error.contains("待处理"));
        assert!(runtime.take_pathing_request().is_some());
    }

    #[test]
    fn request_task_decisions_report_success_and_failure() {
        let mut runtime = TriggerRuntime::new(false, false);
        let snapshot = path_snapshot();
        let (client, _, core) = create_control_bus(snapshot.clone());

        runtime.process_decisions(
            vec![
                Decision {
                    trigger: "test".into(),
                    action: DecisionAction::Log("log".into()),
                },
                Decision {
                    trigger: PATH_REQUEST_TRIGGER_NAME.into(),
                    action: DecisionAction::RequestTask("Daily route".into()),
                },
                Decision {
                    trigger: PATH_REQUEST_TRIGGER_NAME.into(),
                    action: DecisionAction::RequestTask("missing".into()),
                },
                Decision {
                    trigger: "test".into(),
                    action: DecisionAction::Custom("custom".into()),
                },
            ],
            &snapshot,
            &core,
        );

        let logs = client.recent_logs();
        assert_eq!(logs.len(), 4);
        assert_eq!(logs[0].level, LogLevel::Info);
        assert_eq!(logs[0].message, "log");
        assert_eq!(logs[1].level, LogLevel::Info);
        assert!(logs[1].message.contains("Daily route"));
        assert_eq!(logs[2].level, LogLevel::Warn);
        assert!(logs[2].message.contains("missing"));
        assert_eq!(logs[3].level, LogLevel::Debug);
        assert_eq!(logs[3].message, "自定义触发器动作: custom");
        assert_eq!(
            runtime.take_pathing_request(),
            Some(("Daily route".into(), r"D:\paths\daily.json".into()))
        );
    }
}
