use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{PauseController, TriggerLeaseRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UiCategory {
    Unknown,
    Gameplay,
    Loading,
    Menu,
    Inventory,
    Domain,
    Talk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriggerSpec {
    pub name: String,
    pub priority: i32,
    pub enabled: bool,
    pub exclusive: bool,
    pub background: bool,
    pub supported_ui: UiCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerMode {
    Config,
    Task,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameContext<'a> {
    pub frame_id: u64,
    pub captured_at: Instant,
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub bgr24: bool,
    pub ui: UiCategory,
    pub foreground: bool,
    pub requires_foreground: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionAction {
    Log(String),
    RequestTask(String),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub trigger: String,
    pub action: DecisionAction,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("任务已取消")]
    Cancelled,
    #[error("请求重试")]
    Retry,
    #[error("请求不计数重试")]
    RetryNoCount,
    #[error("任务正常结束")]
    NormalEnd,
    #[error("任务已处理该情况")]
    Handled,
    #[error("触发器 {name} 执行 {callback} 失败: {message}")]
    Trigger {
        name: String,
        callback: &'static str,
        message: String,
    },
    #[error("触发器租约名称不能为空")]
    InvalidLeaseName,
    #[error("租约编号溢出")]
    LeaseOverflow,
}

impl RuntimeError {
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Self::Cancelled | Self::Retry | Self::RetryNoCount | Self::NormalEnd | Self::Handled
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriggerFailure {
    pub name: String,
    pub callback: &'static str,
    pub message: String,
}

pub trait Trigger: Send {
    fn spec(&self) -> TriggerSpec;

    fn on_enable(&mut self, _options: &Value) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn on_disable(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn observe(&mut self, _observation: crate::AutoEatObservation) {}

    fn set_enabled(&mut self, _enabled: bool) {}

    fn on_capture(&mut self, _frame: &FrameContext<'_>) -> Result<Vec<Decision>, RuntimeError> {
        Ok(Vec::new())
    }
}

struct TriggerSlot {
    trigger: Box<dyn Trigger>,
    spec: TriggerSpec,
    active: bool,
    options: Value,
    lease_id: Option<u64>,
}

pub struct TriggerDispatcher {
    slots: Vec<TriggerSlot>,
    mode: TriggerMode,
    leases: TriggerLeaseRegistry,
    pauses: PauseController,
    ui: UiCategory,
    ui_changed_at: Option<Instant>,
    ui_grace: Duration,
    failures: Vec<TriggerFailure>,
}

impl TriggerDispatcher {
    pub fn new(mode: TriggerMode) -> Self {
        Self {
            slots: Vec::new(),
            mode,
            leases: TriggerLeaseRegistry::default(),
            pauses: PauseController::default(),
            ui: UiCategory::Unknown,
            ui_changed_at: None,
            ui_grace: Duration::from_secs(30),
            failures: Vec::new(),
        }
    }

    pub fn with_shared_state(
        mut self,
        leases: TriggerLeaseRegistry,
        pauses: PauseController,
    ) -> Self {
        self.leases = leases;
        self.pauses = pauses;
        self
    }

    pub fn mode(&self) -> TriggerMode {
        self.mode
    }

    pub fn has_triggers(&self) -> bool {
        match self.mode {
            TriggerMode::Config => self.slots.iter().any(|slot| slot.spec.enabled),
            TriggerMode::Task => !self.leases.snapshot().is_empty(),
        }
    }

    pub fn set_mode(&mut self, mode: TriggerMode) {
        if self.mode != mode {
            self.mode = mode;
        }
    }

    pub fn leases(&self) -> &TriggerLeaseRegistry {
        &self.leases
    }

    pub fn pauses(&self) -> &PauseController {
        &self.pauses
    }

    pub fn set_enabled(&mut self, names: &[&str], enabled: bool) {
        for slot in &mut self.slots {
            if names.iter().any(|name| *name == slot.spec.name) {
                slot.trigger.set_enabled(enabled);
                slot.spec = slot.trigger.spec();
            }
        }
    }

    pub fn replace_triggers(
        &mut self,
        triggers: Vec<Box<dyn Trigger>>,
    ) -> Result<(), RuntimeError> {
        let mut signal = None;
        for slot in &mut self.slots {
            if slot.active {
                let result = slot.trigger.on_disable();
                if let Err(error) = result {
                    if error.is_control_flow() {
                        signal = Some(error);
                    } else {
                        Self::record_error(
                            &mut self.failures,
                            &slot.spec.name,
                            "on_disable",
                            error,
                        );
                    }
                }
            }
        }
        self.slots = triggers
            .into_iter()
            .map(|trigger| {
                let spec = trigger.spec();
                TriggerSlot {
                    trigger,
                    spec,
                    active: false,
                    options: Value::Null,
                    lease_id: None,
                }
            })
            .collect();
        if let Some(signal) = signal {
            Err(signal)
        } else {
            Ok(())
        }
    }

    pub fn tick(
        &mut self,
        frame: &FrameContext<'_>,
        now: Instant,
    ) -> Result<Vec<Decision>, RuntimeError> {
        if self.slots.is_empty() {
            return Ok(Vec::new());
        }
        self.sync_states()?;
        if frame.ui != self.ui {
            self.ui = frame.ui;
            self.ui_changed_at = Some(now);
        }
        if self.pauses.is_paused() {
            return Ok(Vec::new());
        }

        let mut runnable: Vec<usize> = (0..self.slots.len())
            .filter(|&index| self.slots[index].active)
            .collect();
        runnable.sort_by(|&left, &right| {
            self.slots[right]
                .spec
                .priority
                .cmp(&self.slots[left].spec.priority)
                .then(left.cmp(&right))
        });

        if !frame.foreground && frame.requires_foreground {
            if let Some(&exclusive) = runnable
                .iter()
                .find(|&&index| self.slots[index].spec.exclusive)
            {
                if !self.slots[exclusive].spec.background {
                    return Ok(Vec::new());
                }
                runnable = vec![exclusive];
            } else {
                runnable.retain(|&index| self.slots[index].spec.background);
            }
        } else if let Some(&exclusive) = runnable
            .iter()
            .find(|&&index| self.slots[index].spec.exclusive)
        {
            runnable = vec![exclusive];
        }

        let ui_relaxed = self.ui != UiCategory::Unknown
            && self
                .ui_changed_at
                .is_some_and(|changed| now.saturating_duration_since(changed) <= self.ui_grace);
        let mut decisions = Vec::new();
        for index in runnable {
            let slot = &mut self.slots[index];
            if !ui_relaxed && slot.spec.supported_ui != frame.ui {
                continue;
            }
            match slot.trigger.on_capture(frame) {
                Ok(mut current) => decisions.append(&mut current),
                Err(error) if error.is_control_flow() => return Err(error),
                Err(error) => {
                    Self::record_error(&mut self.failures, &slot.spec.name, "on_capture", error);
                }
            }
        }
        Ok(decisions)
    }

    pub fn set_auto_eat_observation(&mut self, observation: crate::AutoEatObservation) {
        for slot in &mut self.slots {
            slot.trigger.observe(observation);
        }
    }

    pub fn failures(&self) -> &[TriggerFailure] {
        &self.failures
    }

    pub fn take_failures(&mut self) -> Vec<TriggerFailure> {
        std::mem::take(&mut self.failures)
    }

    fn sync_states(&mut self) -> Result<(), RuntimeError> {
        let leases = if self.mode == TriggerMode::Task {
            self.leases.snapshot()
        } else {
            Vec::new()
        };
        let mut signal = None;
        for slot in &mut self.slots {
            slot.spec = slot.trigger.spec();
            let (should_be_active, options, lease_id) = match self.mode {
                TriggerMode::Config => (slot.spec.enabled, Value::Null, None),
                TriggerMode::Task => leases
                    .iter()
                    .rev()
                    .find(|lease| lease.name == slot.spec.name)
                    .map(|lease| (true, lease.options.clone(), Some(lease.id)))
                    .unwrap_or((false, Value::Null, None)),
            };
            if should_be_active != slot.active || lease_id != slot.lease_id {
                if slot.active {
                    let result = slot.trigger.on_disable();
                    if let Err(error) = result {
                        if error.is_control_flow() {
                            signal = Some(error);
                        } else {
                            Self::record_error(
                                &mut self.failures,
                                &slot.spec.name,
                                "on_disable",
                                error,
                            );
                        }
                    }
                }
                slot.active = should_be_active;
                slot.options = options;
                slot.lease_id = lease_id;
                if slot.active {
                    let result = slot.trigger.on_enable(&slot.options);
                    match result {
                        Ok(()) => {}
                        Err(error) if error.is_control_flow() => signal = Some(error),
                        Err(error) => {
                            slot.active = false;
                            Self::record_error(
                                &mut self.failures,
                                &slot.spec.name,
                                "on_enable",
                                error,
                            );
                        }
                    }
                }
            }
        }
        if let Some(signal) = signal {
            Err(signal)
        } else {
            Ok(())
        }
    }

    fn record_error(
        failures: &mut Vec<TriggerFailure>,
        name: &str,
        callback: &'static str,
        error: RuntimeError,
    ) {
        let message = error.to_string();
        failures.push(TriggerFailure {
            name: name.to_string(),
            callback,
            message: message.clone(),
        });
        tracing::warn!(trigger = name, callback, error = %message, "触发器普通异常已隔离");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn unknown_ui_does_not_relax_supported_ui_gate() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(RecordingTrigger {
                spec: TriggerSpec {
                    name: "gameplay-only".into(),
                    priority: 0,
                    enabled: true,
                    exclusive: false,
                    background: false,
                    supported_ui: UiCategory::Gameplay,
                },
                events: events.clone(),
                behavior: Behavior::Record,
            })])
            .unwrap();
        let frame = FrameContext {
            frame_id: 1,
            captured_at: Instant::now(),
            pixels: &[],
            width: 1,
            height: 1,
            bgr24: true,
            ui: UiCategory::Unknown,
            foreground: true,
            requires_foreground: false,
        };
        assert!(dispatcher.tick(&frame, Instant::now()).unwrap().is_empty());
        assert!(
            !events
                .lock()
                .unwrap()
                .iter()
                .any(|event| event.starts_with("run:"))
        );
    }

    #[test]
    fn failed_enable_never_receives_capture() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(RecordingTrigger {
                spec: TriggerSpec {
                    name: "broken".into(),
                    priority: 0,
                    enabled: true,
                    exclusive: false,
                    background: false,
                    supported_ui: UiCategory::Gameplay,
                },
                events: events.clone(),
                behavior: Behavior::Fail,
            })])
            .unwrap();
        let frame = FrameContext {
            frame_id: 1,
            captured_at: Instant::now(),
            pixels: &[],
            width: 1,
            height: 1,
            bgr24: true,
            ui: UiCategory::Gameplay,
            foreground: true,
            requires_foreground: false,
        };
        let _ = dispatcher.tick(&frame, Instant::now());
        assert_eq!(
            dispatcher
                .take_failures()
                .iter()
                .filter(|f| f.callback == "on_enable")
                .count(),
            1
        );
        assert!(
            !events
                .lock()
                .unwrap()
                .iter()
                .any(|event| event == "capture")
        );
    }

    #[test]
    fn updating_config_preserves_unrelated_trigger_state() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![
                Box::new(RecordingTrigger::new("AutoEat", 10, &events)),
                Box::new(RecordingTrigger::new("AutoSkip", 20, &events)),
            ])
            .unwrap();
        let frame = FrameContext {
            frame_id: 1,
            captured_at: Instant::now(),
            pixels: &[],
            width: 1,
            height: 1,
            bgr24: true,
            ui: UiCategory::Gameplay,
            foreground: true,
            requires_foreground: false,
        };
        dispatcher.tick(&frame, Instant::now()).unwrap();
        events.lock().unwrap().clear();
        dispatcher.set_enabled(&["AutoSkip"], false);
        dispatcher.tick(&frame, Instant::now()).unwrap();
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| *event == "run:AutoEat")
                .count(),
            1
        );
        assert!(!events.iter().any(|event| *event == "enable:AutoEat"));
        assert!(!events.iter().any(|event| *event == "disable:AutoEat"));
    }

    #[derive(Debug, Clone, Copy)]
    enum Behavior {
        Record,
        Fail,
        Cancel,
    }

    #[derive(Debug)]
    struct RecordingTrigger {
        spec: TriggerSpec,
        events: Arc<Mutex<Vec<String>>>,
        behavior: Behavior,
    }

    impl RecordingTrigger {
        fn new(name: &str, priority: i32, events: &Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                spec: TriggerSpec {
                    name: name.into(),
                    priority,
                    enabled: true,
                    exclusive: false,
                    background: false,
                    supported_ui: UiCategory::Gameplay,
                },
                events: events.clone(),
                behavior: Behavior::Record,
            }
        }
    }

    impl Trigger for RecordingTrigger {
        fn spec(&self) -> TriggerSpec {
            self.spec.clone()
        }

        fn set_enabled(&mut self, enabled: bool) {
            self.spec.enabled = enabled;
        }

        fn on_enable(&mut self, _: &Value) -> Result<(), RuntimeError> {
            self.events
                .lock()
                .unwrap()
                .push(format!("enable:{}", self.spec.name));
            if matches!(self.behavior, Behavior::Fail) {
                return Err(RuntimeError::Trigger {
                    name: self.spec.name.clone(),
                    callback: "on_enable",
                    message: "injected enable failure".into(),
                });
            }
            Ok(())
        }

        fn on_disable(&mut self) -> Result<(), RuntimeError> {
            self.events
                .lock()
                .unwrap()
                .push(format!("disable:{}", self.spec.name));
            Ok(())
        }

        fn on_capture(&mut self, _: &FrameContext) -> Result<Vec<Decision>, RuntimeError> {
            self.events
                .lock()
                .unwrap()
                .push(format!("run:{}", self.spec.name));
            match self.behavior {
                Behavior::Record => Ok(vec![Decision {
                    trigger: self.spec.name.clone(),
                    action: DecisionAction::Log("ok".into()),
                }]),
                Behavior::Fail => Err(RuntimeError::Trigger {
                    name: self.spec.name.clone(),
                    callback: "on_capture",
                    message: "injected".into(),
                }),
                Behavior::Cancel => Err(RuntimeError::Cancelled),
            }
        }
    }

    fn frame(foreground: bool) -> FrameContext<'static> {
        FrameContext {
            frame_id: 1,
            captured_at: Instant::now(),
            pixels: &[],
            width: 1920,
            height: 1080,
            bgr24: true,
            ui: UiCategory::Gameplay,
            foreground,
            requires_foreground: true,
        }
    }

    #[test]
    fn exclusive_trigger_preempts_lower_priority_triggers() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let high = RecordingTrigger::new("high", 100, &events);
        let mut exclusive = RecordingTrigger::new("exclusive", 10, &events);
        exclusive.spec.exclusive = true;
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(high), Box::new(exclusive)])
            .unwrap();

        let decisions = dispatcher.tick(&frame(true), Instant::now()).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].trigger, "exclusive");
        assert_eq!(
            events
                .lock()
                .unwrap()
                .iter()
                .filter(|event| event.starts_with("run:"))
                .collect::<Vec<_>>(),
            vec!["run:exclusive"]
        );
    }

    #[test]
    fn task_mode_switch_and_lease_apply_on_the_next_tick() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let trigger = RecordingTrigger::new("pick", 10, &events);
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(trigger)])
            .unwrap();
        dispatcher.tick(&frame(true), Instant::now()).unwrap();

        dispatcher.set_mode(TriggerMode::Task);
        assert!(
            dispatcher
                .tick(&frame(true), Instant::now())
                .unwrap()
                .is_empty()
        );
        let lease = dispatcher
            .leases()
            .lease("pick", serde_json::json!({"x": 1}))
            .unwrap();
        assert_eq!(
            dispatcher.tick(&frame(true), Instant::now()).unwrap().len(),
            1
        );

        let before = events.lock().unwrap().len();
        let replacement = dispatcher
            .leases()
            .lease("pick", serde_json::json!({"x": 1}))
            .unwrap();
        assert_eq!(
            dispatcher.tick(&frame(true), Instant::now()).unwrap().len(),
            1
        );
        assert_eq!(
            &events.lock().unwrap()[before..],
            ["disable:pick", "enable:pick", "run:pick"]
        );

        drop(replacement);
        assert_eq!(
            dispatcher.tick(&frame(true), Instant::now()).unwrap().len(),
            1
        );
        drop(lease);
        assert!(
            dispatcher
                .tick(&frame(true), Instant::now())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn ordinary_failure_is_isolated_but_cancellation_propagates() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut failing = RecordingTrigger::new("failing", 20, &events);
        failing.behavior = Behavior::Fail;
        let healthy = RecordingTrigger::new("healthy", 10, &events);
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(failing), Box::new(healthy)])
            .unwrap();

        assert_eq!(
            dispatcher.tick(&frame(true), Instant::now()).unwrap().len(),
            1
        );
        assert_eq!(dispatcher.failures().len(), 1);
        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .any(|event| event == "run:healthy")
        );

        let mut canceling = RecordingTrigger::new("canceling", 20, &events);
        canceling.behavior = Behavior::Cancel;
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(canceling)])
            .unwrap();
        assert!(matches!(
            dispatcher.tick(&frame(true), Instant::now()),
            Err(RuntimeError::Cancelled)
        ));
    }

    #[test]
    fn lost_foreground_runs_only_background_triggers() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let normal = RecordingTrigger::new("normal", 20, &events);
        let mut background = RecordingTrigger::new("background", 10, &events);
        background.spec.background = true;
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(normal), Box::new(background)])
            .unwrap();

        let decisions = dispatcher.tick(&frame(false), Instant::now()).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].trigger, "background");
    }

    #[test]
    fn ui_filter_relaxes_for_thirty_seconds_after_change() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut trigger = RecordingTrigger::new("gameplay", 10, &events);
        trigger.spec.supported_ui = UiCategory::Inventory;
        let mut dispatcher = TriggerDispatcher::new(TriggerMode::Config);
        dispatcher
            .replace_triggers(vec![Box::new(trigger)])
            .unwrap();
        let now = Instant::now();

        assert_eq!(dispatcher.tick(&frame(true), now).unwrap().len(), 1);
        assert!(
            dispatcher
                .tick(&frame(true), now + Duration::from_secs(31))
                .unwrap()
                .is_empty()
        );
    }
}
