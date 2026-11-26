use bgi_task::{
    AutoEatAction, AutoEatController, AutoEatObservation, AutoSkipController, Decision,
    DecisionAction, FrameContext, RuntimeError, Trigger, TriggerSpec, UiCategory,
};
use serde_json::Value;

pub(super) const PATH_REQUEST_TRIGGER_NAME: &str = "PathRequest";
pub(super) const AUTO_PICK_TRIGGER_NAME: &str = "AutoPick";
pub(super) const AUTO_SKIP_TRIGGER_NAME: &str = "AutoSkip";
pub(super) const AUTO_EAT_TRIGGER_NAME: &str = "AutoEat";

#[derive(Debug, Default)]
pub(super) struct PathRequestTrigger {
    options: Option<PathRequestOptions>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PathRequestOptions {
    task_name: String,
    request_every_n: u64,
}

impl PathRequestTrigger {
    pub(super) fn new() -> Self {
        Self::default()
    }
}

impl Trigger for PathRequestTrigger {
    fn spec(&self) -> TriggerSpec {
        TriggerSpec {
            name: PATH_REQUEST_TRIGGER_NAME.into(),
            priority: 0,
            enabled: false,
            exclusive: false,
            background: false,
            supported_ui: UiCategory::Unknown,
        }
    }

    fn on_enable(&mut self, options: &Value) -> Result<(), RuntimeError> {
        self.options = None;
        self.options = Some(parse_options(options)?);
        Ok(())
    }

    fn on_disable(&mut self) -> Result<(), RuntimeError> {
        self.options = None;
        Ok(())
    }

    fn on_capture(&mut self, frame: &FrameContext<'_>) -> Result<Vec<Decision>, RuntimeError> {
        let Some(options) = &self.options else {
            return Ok(Vec::new());
        };
        if !frame.foreground || !frame.frame_id.is_multiple_of(options.request_every_n) {
            return Ok(Vec::new());
        }

        Ok(vec![Decision {
            trigger: PATH_REQUEST_TRIGGER_NAME.into(),
            action: DecisionAction::RequestTask(options.task_name.clone()),
        }])
    }
}

#[derive(Debug, Default)]
pub(super) struct AutoPickTrigger {
    request_every_n: Option<u64>,
}

impl Trigger for AutoPickTrigger {
    fn spec(&self) -> TriggerSpec {
        TriggerSpec {
            name: AUTO_PICK_TRIGGER_NAME.into(),
            priority: -10,
            enabled: false,
            exclusive: false,
            background: false,
            supported_ui: UiCategory::Unknown,
        }
    }

    fn on_enable(&mut self, options: &Value) -> Result<(), RuntimeError> {
        let object = options
            .as_object()
            .ok_or_else(|| auto_pick_error("选项必须是 JSON 对象"))?;
        let mut unknown: Vec<_> = object
            .keys()
            .filter(|key| key.as_str() != "request_every_n")
            .cloned()
            .collect();
        unknown.sort_unstable();
        if !unknown.is_empty() {
            return Err(auto_pick_error(format!(
                "包含未知选项: {}",
                unknown.join(", ")
            )));
        }
        let interval = object
            .get("request_every_n")
            .and_then(Value::as_u64)
            .filter(|interval| *interval > 0)
            .ok_or_else(|| auto_pick_error("request_every_n 必须是大于 0 的整数"))?;
        self.request_every_n = Some(interval);
        Ok(())
    }

    fn on_disable(&mut self) -> Result<(), RuntimeError> {
        self.request_every_n = None;
        Ok(())
    }

    fn on_capture(&mut self, frame: &FrameContext<'_>) -> Result<Vec<Decision>, RuntimeError> {
        if !frame.foreground
            || !frame
                .frame_id
                .is_multiple_of(self.request_every_n.unwrap_or(4))
        {
            return Ok(Vec::new());
        }
        Ok(vec![Decision {
            trigger: AUTO_PICK_TRIGGER_NAME.into(),
            action: DecisionAction::Custom(AUTO_PICK_TRIGGER_NAME.into()),
        }])
    }
}

#[derive(Debug, Default)]
pub(super) struct AutoSkipTrigger {
    enabled: bool,
    controller: AutoSkipController,
}

impl AutoSkipTrigger {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            controller: AutoSkipController::default(),
        }
    }
}

impl Trigger for AutoSkipTrigger {
    fn spec(&self) -> TriggerSpec {
        TriggerSpec {
            name: AUTO_SKIP_TRIGGER_NAME.into(),
            priority: 20,
            enabled: self.enabled,
            exclusive: false,
            background: false,
            supported_ui: UiCategory::Talk,
        }
    }

    fn on_disable(&mut self) -> Result<(), RuntimeError> {
        self.controller.reset();
        Ok(())
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn on_capture(&mut self, frame: &FrameContext<'_>) -> Result<Vec<Decision>, RuntimeError> {
        if self.controller.observe(
            frame.ui == UiCategory::Talk,
            frame.foreground,
            frame.captured_at,
        ) != bgi_task::AutoSkipEvent::Advance
        {
            return Ok(Vec::new());
        }
        Ok(vec![Decision {
            trigger: AUTO_SKIP_TRIGGER_NAME.into(),
            action: DecisionAction::Custom(AUTO_SKIP_TRIGGER_NAME.into()),
        }])
    }
}

#[derive(Debug, Default)]
pub(super) struct AutoEatTrigger {
    enabled: bool,
    controller: AutoEatController,
    observation: AutoEatObservation,
}

impl AutoEatTrigger {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }
}

impl Trigger for AutoEatTrigger {
    fn spec(&self) -> TriggerSpec {
        TriggerSpec {
            name: AUTO_EAT_TRIGGER_NAME.into(),
            priority: 25,
            enabled: self.enabled,
            exclusive: false,
            background: false,
            supported_ui: UiCategory::Gameplay,
        }
    }

    fn observe(&mut self, observation: AutoEatObservation) {
        self.observation = observation;
    }

    fn on_disable(&mut self) -> Result<(), RuntimeError> {
        self.controller.reset();
        self.observation = AutoEatObservation::default();
        Ok(())
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn on_capture(&mut self, frame: &FrameContext<'_>) -> Result<Vec<Decision>, RuntimeError> {
        let Some(action) = self.controller.observe(
            self.observation,
            frame.ui == UiCategory::Gameplay,
            frame.foreground,
            frame.captured_at,
        ) else {
            return Ok(Vec::new());
        };
        let name = match action {
            AutoEatAction::Recover => "AutoEatRecover",
            AutoEatAction::Revive => "AutoEatRevive",
        };
        Ok(vec![Decision {
            trigger: AUTO_EAT_TRIGGER_NAME.into(),
            action: DecisionAction::Custom(name.into()),
        }])
    }
}

fn auto_pick_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Trigger {
        name: AUTO_PICK_TRIGGER_NAME.into(),
        callback: "on_enable",
        message: message.into(),
    }
}

pub(super) fn install_builtin_triggers(auto_skip: bool, auto_eat: bool) -> Vec<Box<dyn Trigger>> {
    vec![
        Box::new(PathRequestTrigger::new()),
        Box::new(AutoPickTrigger::default()),
        Box::new(AutoSkipTrigger::new(auto_skip)),
        Box::new(AutoEatTrigger::new(auto_eat)),
    ]
}

pub(crate) fn validate_options(options: &Value) -> Result<(), RuntimeError> {
    parse_options(options).map(|_| ())
}

fn parse_options(options: &Value) -> Result<PathRequestOptions, RuntimeError> {
    let object = options
        .as_object()
        .ok_or_else(|| options_error("选项必须是 JSON 对象"))?;

    let mut unknown: Vec<_> = object
        .keys()
        .filter(|key| !matches!(key.as_str(), "task_name" | "request_every_n"))
        .cloned()
        .collect();
    unknown.sort_unstable();
    if !unknown.is_empty() {
        return Err(options_error(format!(
            "包含未知选项: {}",
            unknown.join(", ")
        )));
    }

    let task_name = object
        .get("task_name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| options_error("task_name 必须是非空字符串"))?;
    let request_every_n = object
        .get("request_every_n")
        .and_then(Value::as_u64)
        .filter(|interval| *interval > 0)
        .ok_or_else(|| options_error("request_every_n 必须是大于 0 的整数"))?;

    Ok(PathRequestOptions {
        task_name: task_name.to_string(),
        request_every_n,
    })
}

fn options_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Trigger {
        name: PATH_REQUEST_TRIGGER_NAME.into(),
        callback: "on_enable",
        message: message.into(),
    }
}
