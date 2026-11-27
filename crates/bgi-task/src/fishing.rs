use std::time::{Duration, Instant};

// C# `AutoFishingConfig.WholeProcessTimeoutSeconds` default = 300.
const DEFAULT_TOTAL_TIMEOUT_MS: u64 = 300_000;
const HOOK_DELAY: Duration = Duration::from_millis(800);
const FIND_BAR_TIMEOUT: Duration = Duration::from_secs(5);
const BAR_LOST_TIMEOUT: Duration = Duration::from_secs(1);
const MIN_TARGET_WIDTH_RATIO: i64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishingConfig {
    pub total_timeout_ms: u64,
}

impl Default for FishingConfig {
    fn default() -> Self {
        Self {
            total_timeout_ms: DEFAULT_TOTAL_TIMEOUT_MS,
        }
    }
}

impl FishingConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1_000..=3_600_000).contains(&self.total_timeout_ms) {
            return Err("自动钓鱼总超时必须位于 1000..=3600000 ms");
        }
        Ok(())
    }

    pub const fn total_timeout(&self) -> Duration {
        Duration::from_millis(self.total_timeout_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishingRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl FishingRect {
    pub fn right(self) -> i64 {
        i64::from(self.x) + i64::from(self.width)
    }

    pub fn center_x(self) -> i64 {
        i64::from(self.x) + i64::from(self.width / 2)
    }

    pub fn center_y(self) -> i64 {
        i64::from(self.y) + i64::from(self.height / 2)
    }

    pub fn validate(self, frame_size: (u32, u32)) -> Result<(), &'static str> {
        if frame_size.0 == 0 || frame_size.1 == 0 {
            return Err("钓鱼画面尺寸无效");
        }
        if self.width <= 0 || self.height <= 0 {
            return Err("钓鱼鱼条矩形宽高必须为正");
        }
        if self.x < 0
            || self.y < 0
            || self.right() > i64::from(frame_size.0)
            || i64::from(self.y) + i64::from(self.height) > i64::from(frame_size.1)
        {
            return Err("钓鱼鱼条矩形必须位于画面内");
        }
        Ok(())
    }

    fn is_valid(self, frame_size: (u32, u32)) -> bool {
        self.validate(frame_size).is_ok()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishingBar {
    pub cursor: FishingRect,
    pub target: FishingRect,
}

impl FishingBar {
    fn is_valid(self, frame_size: (u32, u32)) -> bool {
        self.cursor.is_valid(frame_size) && self.target.is_valid(frame_size)
    }

    fn is_valid_for_acquisition(self, frame_size: (u32, u32)) -> bool {
        self.is_valid(frame_size)
            && self.cursor.center_x() < self.target.center_x()
            && i64::from(self.cursor.width) <= i64::from(self.target.width)
            && i64::from(self.target.width) >= MIN_TARGET_WIDTH_RATIO * i64::from(self.cursor.width)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishingObservation {
    pub in_fishing: bool,
    pub bite: bool,
    pub bar: Option<FishingBar>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishingAction {
    MouseDown,
    MouseUp,
    Wait,
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    AwaitBite,
    HookDelay { ready_at: Instant },
    FindBar { started_at: Instant },
    Reeling { bar_lost_since: Option<Instant> },
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone)]
pub struct FishingController {
    config: FishingConfig,
    phase: Phase,
    started_at: Option<Instant>,
    mouse_held: bool,
}

impl Default for FishingController {
    fn default() -> Self {
        Self::new(FishingConfig::default())
    }
}

impl FishingController {
    pub fn new(config: FishingConfig) -> Self {
        Self::try_new(config).expect("invalid fishing config")
    }

    pub fn try_new(config: FishingConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            phase: Phase::AwaitBite,
            started_at: None,
            mouse_held: false,
        })
    }

    pub const fn config(&self) -> FishingConfig {
        self.config
    }

    pub const fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Succeeded | Phase::Failed(_))
    }

    pub const fn succeeded(&self) -> bool {
        matches!(self.phase, Phase::Succeeded)
    }

    pub const fn is_mouse_held(&self) -> bool {
        self.mouse_held
    }

    pub fn wait_ms(&self, now: Instant) -> Option<u64> {
        if self.is_finished() {
            return None;
        }
        let Phase::HookDelay { ready_at } = self.phase else {
            return None;
        };
        let started_at = self.started_at?;
        let deadline = started_at + self.config.total_timeout();
        Some(
            ready_at
                .min(deadline)
                .saturating_duration_since(now)
                .as_millis() as u64
                + 1,
        )
    }

    pub fn advance(
        &mut self,
        observation: FishingObservation,
        frame_size: (u32, u32),
        now: Instant,
    ) -> FishingAction {
        if self.is_finished() {
            return self.terminal_action();
        }
        let started_at = *self.started_at.get_or_insert(now);
        self.advance_since(observation, frame_size, now, started_at)
    }

    pub fn advance_since(
        &mut self,
        observation: FishingObservation,
        frame_size: (u32, u32),
        now: Instant,
        started_at: Instant,
    ) -> FishingAction {
        if self.is_finished() {
            return self.terminal_action();
        }
        let started_at = self
            .started_at
            .map_or(started_at, |current| current.min(started_at));
        self.started_at = Some(started_at);
        if now.saturating_duration_since(started_at) >= self.config.total_timeout() {
            return self.fail("自动钓鱼总超时");
        }
        if !observation.in_fishing {
            return self.fail("当前不在钓鱼状态");
        }
        if frame_size.0 == 0 || frame_size.1 == 0 {
            return self.fail("钓鱼画面尺寸无效");
        }

        match self.phase {
            Phase::AwaitBite => {
                if observation.bite {
                    self.mouse_held = true;
                    self.phase = Phase::HookDelay {
                        ready_at: now + HOOK_DELAY,
                    };
                    FishingAction::MouseDown
                } else {
                    FishingAction::Wait
                }
            }
            Phase::HookDelay { ready_at } => {
                if now < ready_at {
                    FishingAction::Wait
                } else {
                    self.phase = Phase::FindBar {
                        started_at: ready_at,
                    };
                    self.advance_find_bar(observation, frame_size, now)
                }
            }
            Phase::FindBar { .. } => self.advance_find_bar(observation, frame_size, now),
            Phase::Reeling { .. } => self.advance_reeling(observation, frame_size, now),
            Phase::Succeeded | Phase::Failed(_) => {
                unreachable!("terminal state handled above")
            }
        }
    }

    fn advance_find_bar(
        &mut self,
        observation: FishingObservation,
        frame_size: (u32, u32),
        now: Instant,
    ) -> FishingAction {
        let Phase::FindBar { started_at } = self.phase else {
            unreachable!("advance_find_bar called outside FindBar")
        };
        if now.saturating_duration_since(started_at) >= FIND_BAR_TIMEOUT {
            return self.fail("未检测到有效的钓鱼鱼条");
        }
        if observation
            .bar
            .is_some_and(|bar| bar.is_valid_for_acquisition(frame_size))
        {
            self.phase = Phase::Reeling {
                bar_lost_since: None,
            };
            return self.advance_reeling(observation, frame_size, now);
        }
        FishingAction::Wait
    }

    fn advance_reeling(
        &mut self,
        observation: FishingObservation,
        frame_size: (u32, u32),
        now: Instant,
    ) -> FishingAction {
        let Phase::Reeling { bar_lost_since } = self.phase else {
            unreachable!("advance_reeling called outside Reeling")
        };
        if let Some(bar) = observation.bar.filter(|bar| bar.is_valid(frame_size)) {
            self.phase = Phase::Reeling {
                bar_lost_since: None,
            };
            return self.set_mouse_held(bar.cursor.center_x() < bar.target.center_x());
        }

        let bar_lost_since = bar_lost_since.unwrap_or(now);
        self.phase = Phase::Reeling {
            bar_lost_since: Some(bar_lost_since),
        };
        if now.saturating_duration_since(bar_lost_since) < BAR_LOST_TIMEOUT {
            return FishingAction::Wait;
        }
        self.phase = Phase::Succeeded;
        if self.mouse_held {
            self.mouse_held = false;
            FishingAction::MouseUp
        } else {
            FishingAction::Succeeded
        }
    }

    fn set_mouse_held(&mut self, held: bool) -> FishingAction {
        if held == self.mouse_held {
            return FishingAction::Wait;
        }
        self.mouse_held = held;
        if held {
            FishingAction::MouseDown
        } else {
            FishingAction::MouseUp
        }
    }

    fn fail(&mut self, reason: &'static str) -> FishingAction {
        self.phase = Phase::Failed(reason);
        FishingAction::Failed(reason)
    }

    fn terminal_action(&self) -> FishingAction {
        match self.phase {
            Phase::Succeeded => FishingAction::Succeeded,
            Phase::Failed(reason) => FishingAction::Failed(reason),
            _ => unreachable!("terminal_action called for a non-terminal state"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: (u32, u32) = (1_920, 1_080);

    fn at(milliseconds: u64) -> Instant {
        static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *BASE.get_or_init(Instant::now) + Duration::from_millis(milliseconds)
    }

    fn rect(x: i32, width: i32) -> FishingRect {
        FishingRect {
            x,
            y: 100,
            width,
            height: 20,
        }
    }

    fn bar(cursor_x: i32, target_x: i32) -> FishingBar {
        FishingBar {
            cursor: rect(cursor_x, 10),
            target: rect(target_x, 100),
        }
    }

    fn observation(in_fishing: bool, bite: bool, bar: Option<FishingBar>) -> FishingObservation {
        FishingObservation {
            in_fishing,
            bite,
            bar,
        }
    }

    fn start_reeling(controller: &mut FishingController, now: Instant) {
        assert_eq!(
            controller.advance(observation(true, true, None), FRAME, now),
            FishingAction::MouseDown
        );
        assert_eq!(
            controller.advance(
                observation(true, false, Some(bar(100, 500))),
                FRAME,
                now + HOOK_DELAY,
            ),
            FishingAction::Wait
        );
    }

    #[test]
    fn config_defaults_and_boundaries_are_validated() {
        let default = FishingConfig::default();
        assert_eq!(default.total_timeout_ms, 300_000);
        assert_eq!(default.total_timeout(), Duration::from_secs(300));
        assert_eq!(default.validate(), Ok(()));
        assert_eq!(FishingController::default().config(), default);

        for total_timeout_ms in [1_000, 3_600_000] {
            let config = FishingConfig { total_timeout_ms };
            assert_eq!(config.validate(), Ok(()));
            assert!(FishingController::try_new(config).is_ok());
        }
        for total_timeout_ms in [0, 999, 3_600_001, u64::MAX] {
            let config = FishingConfig { total_timeout_ms };
            assert!(config.validate().is_err());
            assert!(FishingController::try_new(config).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "invalid fishing config")]
    fn invalid_config_is_rejected_by_the_infallible_constructor() {
        FishingController::new(FishingConfig {
            total_timeout_ms: 999,
        });
    }

    #[test]
    fn rectangle_geometry_and_bounds_are_validated_without_overflow() {
        let rectangle = FishingRect {
            x: 10,
            y: 20,
            width: 21,
            height: 11,
        };
        assert_eq!(rectangle.right(), 31);
        assert_eq!(rectangle.center_x(), 20);
        assert_eq!(rectangle.center_y(), 25);
        assert_eq!(rectangle.validate(FRAME), Ok(()));
        assert_eq!(
            FishingRect {
                x: 1_900,
                y: 1_060,
                width: 20,
                height: 20,
            }
            .validate(FRAME),
            Ok(())
        );

        for invalid in [
            FishingRect {
                x: 0,
                y: 0,
                width: 0,
                height: 10,
            },
            FishingRect {
                x: 0,
                y: 0,
                width: -1,
                height: 10,
            },
            FishingRect {
                x: 0,
                y: 0,
                width: 10,
                height: 0,
            },
            FishingRect {
                x: -1,
                y: 0,
                width: 10,
                height: 10,
            },
            FishingRect {
                x: 0,
                y: -1,
                width: 10,
                height: 10,
            },
            FishingRect {
                x: 1_920,
                y: 0,
                width: 1,
                height: 10,
            },
            FishingRect {
                x: 0,
                y: 1_080,
                width: 10,
                height: 1,
            },
        ] {
            assert!(invalid.validate(FRAME).is_err());
        }
        assert!(rectangle.validate((0, FRAME.1)).is_err());
        assert!(rectangle.validate((FRAME.0, 0)).is_err());

        let extreme = FishingRect {
            x: i32::MAX,
            y: i32::MAX,
            width: i32::MAX,
            height: i32::MAX,
        };
        assert_eq!(extreme.right(), i64::from(i32::MAX) * 2);
        assert_eq!(
            extreme.center_x(),
            i64::from(i32::MAX) + i64::from(i32::MAX / 2)
        );
        assert_eq!(extreme.validate((u32::MAX, u32::MAX)), Ok(()));
    }

    #[test]
    fn every_nonterminal_phase_requires_fishing_entry() {
        let mut awaiting = FishingController::default();
        assert_eq!(
            awaiting.advance(observation(false, true, None), FRAME, at(0)),
            FishingAction::Failed("当前不在钓鱼状态")
        );
        assert!(awaiting.is_finished());
        assert!(!awaiting.is_mouse_held());

        let mut hook_delay = FishingController::default();
        assert_eq!(
            hook_delay.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(
            hook_delay.advance(observation(false, false, None), FRAME, at(1)),
            FishingAction::Failed("当前不在钓鱼状态")
        );

        let mut finding = FishingController::default();
        assert_eq!(
            finding.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(
            finding.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            finding.advance(observation(false, false, None), FRAME, at(801)),
            FishingAction::Failed("当前不在钓鱼状态")
        );

        let mut reeling = FishingController::default();
        start_reeling(&mut reeling, at(0));
        assert_eq!(
            reeling.advance(observation(false, false, None), FRAME, at(800)),
            FishingAction::Failed("当前不在钓鱼状态")
        );
    }

    #[test]
    fn invalid_frame_fails_closed_without_clicking() {
        let mut controller = FishingController::default();
        assert_eq!(
            controller.advance(observation(true, true, None), (0, FRAME.1), at(0)),
            FishingAction::Failed("钓鱼画面尺寸无效")
        );
        assert!(!controller.is_mouse_held());
    }

    #[test]
    fn bite_emits_one_mouse_down_and_waits_800_ms() {
        let mut controller = FishingController::default();
        assert_eq!(controller.wait_ms(at(0)), None);
        assert_eq!(
            controller.advance(observation(true, true, Some(bar(100, 500))), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert!(controller.is_mouse_held());
        assert_eq!(controller.wait_ms(at(0)), Some(801));
        assert_eq!(controller.wait_ms(at(799)), Some(2));

        for elapsed in [1, 400, 799] {
            assert_eq!(
                controller.advance(observation(true, true, None), FRAME, at(elapsed)),
                FishingAction::Wait
            );
        }
        assert_eq!(
            controller.advance(observation(true, true, Some(bar(100, 500))), FRAME, at(800),),
            FishingAction::Wait
        );
        assert_eq!(controller.wait_ms(at(800)), None);
        for elapsed in [801, 1_600, 10_000] {
            assert_eq!(
                controller.advance(
                    observation(true, true, Some(bar(100, 500))),
                    FRAME,
                    at(elapsed),
                ),
                FishingAction::Wait
            );
        }
        assert!(controller.is_mouse_held());
    }

    #[test]
    fn missing_bar_times_out_five_seconds_after_hook_delay() {
        let mut controller = FishingController::default();
        assert_eq!(
            controller.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(5_799)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(5_800)),
            FishingAction::Failed("未检测到有效的钓鱼鱼条")
        );
        assert!(controller.is_finished());
        assert!(!controller.succeeded());
    }

    #[test]
    fn invalid_acquisition_bars_never_click_or_enter_reeling() {
        let valid = bar(100, 500);
        let invalid_bars = [
            FishingBar {
                cursor: rect(550, 10),
                target: rect(500, 100),
            },
            FishingBar {
                cursor: rect(100, 101),
                target: rect(500, 100),
            },
            FishingBar {
                cursor: rect(100, 11),
                target: rect(500, 100),
            },
            FishingBar {
                cursor: rect(100, 10),
                target: rect(500, 90),
            },
            FishingBar {
                cursor: rect(-1, 10),
                target: rect(500, 100),
            },
            FishingBar {
                cursor: rect(100, 0),
                target: rect(500, 100),
            },
            FishingBar {
                cursor: rect(100, 10),
                target: FishingRect {
                    x: 1_900,
                    y: 100,
                    width: 100,
                    height: 20,
                },
            },
            FishingBar {
                cursor: rect(100, 10),
                target: FishingRect {
                    x: 500,
                    y: 1_061,
                    width: 100,
                    height: 20,
                },
            },
        ];

        for invalid in invalid_bars {
            let mut controller = FishingController::default();
            assert_eq!(
                controller.advance(observation(true, true, None), FRAME, at(0)),
                FishingAction::MouseDown
            );
            assert_eq!(
                controller.advance(observation(true, false, Some(invalid)), FRAME, at(800),),
                FishingAction::Wait
            );
            assert!(matches!(controller.phase, Phase::FindBar { .. }));
            assert!(controller.is_mouse_held());
        }

        let mut controller = FishingController::default();
        assert_eq!(
            controller.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(
            controller.advance(observation(true, false, Some(valid)), FRAME, at(800),),
            FishingAction::Wait
        );
        assert!(matches!(controller.phase, Phase::Reeling { .. }));
    }

    #[test]
    fn reeling_emits_directional_mouse_actions_only_on_transitions() {
        let mut controller = FishingController::default();
        start_reeling(&mut controller, at(0));
        assert!(controller.is_mouse_held());

        for elapsed in [800, 801, 900] {
            assert_eq!(
                controller.advance(
                    observation(true, false, Some(bar(100, 500))),
                    FRAME,
                    at(elapsed),
                ),
                FishingAction::Wait
            );
        }
        for elapsed in [1_000, 1_001, 1_100] {
            assert_eq!(
                controller.advance(
                    observation(true, false, Some(bar(600, 500))),
                    FRAME,
                    at(elapsed),
                ),
                if elapsed == 1_000 {
                    FishingAction::MouseUp
                } else {
                    FishingAction::Wait
                }
            );
        }
        assert!(!controller.is_mouse_held());

        assert_eq!(
            controller.advance(
                observation(true, false, Some(bar(100, 500))),
                FRAME,
                at(1_200),
            ),
            FishingAction::MouseDown
        );
        assert!(controller.is_mouse_held());
        assert_eq!(
            controller.advance(
                observation(true, false, Some(bar(100, 500))),
                FRAME,
                at(1_201),
            ),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(
                observation(true, false, Some(bar(545, 500))),
                FRAME,
                at(1_202),
            ),
            FishingAction::MouseUp
        );
        assert!(!controller.is_mouse_held());
    }

    #[test]
    fn one_continuous_second_without_bar_succeeds_and_releases_mouse() {
        let mut controller = FishingController::default();
        start_reeling(&mut controller, at(0));
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_799)),
            FishingAction::Wait
        );
        assert!(controller.is_mouse_held());
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_800)),
            FishingAction::MouseUp
        );
        assert!(controller.is_finished());
        assert!(controller.succeeded());
        assert!(!controller.is_mouse_held());
    }

    #[test]
    fn disappearance_success_without_held_mouse_emits_no_mouse_up() {
        let mut controller = FishingController::default();
        start_reeling(&mut controller, at(0));
        assert_eq!(
            controller.advance(
                observation(true, false, Some(bar(600, 500))),
                FRAME,
                at(800),
            ),
            FishingAction::MouseUp
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_800)),
            FishingAction::Succeeded
        );
        assert!(!controller.is_mouse_held());
    }

    #[test]
    fn bar_reappearance_resets_the_continuous_loss_timer() {
        let mut controller = FishingController::default();
        start_reeling(&mut controller, at(0));
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_799)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(
                observation(true, false, Some(bar(100, 500))),
                FRAME,
                at(1_800),
            ),
            FishingAction::Wait
        );
        assert!(!controller.is_finished());
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_800)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(2_799)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(2_800)),
            FishingAction::MouseUp
        );
        assert!(controller.succeeded());
    }

    #[test]
    fn invalid_reeling_bar_is_treated_as_lost_without_input() {
        let mut controller = FishingController::default();
        start_reeling(&mut controller, at(0));
        let invalid = FishingBar {
            cursor: rect(100, 0),
            target: rect(500, 100),
        };
        assert_eq!(
            controller.advance(observation(true, false, Some(invalid)), FRAME, at(800),),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_800)),
            FishingAction::MouseUp
        );
        assert!(controller.succeeded());
    }

    #[test]
    fn total_timeout_is_enforced_in_every_nonterminal_phase() {
        let config = FishingConfig {
            total_timeout_ms: 1_000,
        };

        let mut awaiting = FishingController::new(config);
        assert_eq!(
            awaiting.advance(observation(true, false, None), FRAME, at(0)),
            FishingAction::Wait
        );
        assert_eq!(
            awaiting.advance(observation(true, false, None), FRAME, at(999)),
            FishingAction::Wait
        );
        assert_eq!(
            awaiting.advance(observation(true, true, None), FRAME, at(1_000)),
            FishingAction::Failed("自动钓鱼总超时")
        );
        assert!(!awaiting.is_mouse_held());

        let mut hook_delay = FishingController::new(config);
        assert_eq!(
            hook_delay.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(hook_delay.wait_ms(at(0)), Some(801));
        assert_eq!(
            hook_delay.advance(observation(true, false, None), FRAME, at(999)),
            FishingAction::Wait
        );
        assert_eq!(
            hook_delay.advance(observation(true, false, None), FRAME, at(1_000)),
            FishingAction::Failed("自动钓鱼总超时")
        );

        let mut finding = FishingController::new(config);
        assert_eq!(
            finding.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(
            finding.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            finding.advance(observation(true, false, None), FRAME, at(999)),
            FishingAction::Wait
        );
        assert_eq!(
            finding.advance(observation(true, false, None), FRAME, at(1_000)),
            FishingAction::Failed("自动钓鱼总超时")
        );

        let mut reeling = FishingController::new(config);
        assert_eq!(
            reeling.advance(observation(true, true, None), FRAME, at(0)),
            FishingAction::MouseDown
        );
        assert_eq!(
            reeling.advance(
                observation(true, false, Some(bar(100, 500))),
                FRAME,
                at(800),
            ),
            FishingAction::Wait
        );
        assert_eq!(
            reeling.advance(
                observation(true, false, Some(bar(100, 500))),
                FRAME,
                at(999),
            ),
            FishingAction::Wait
        );
        assert_eq!(
            reeling.advance(
                observation(true, false, Some(bar(600, 500))),
                FRAME,
                at(1_000),
            ),
            FishingAction::Failed("自动钓鱼总超时")
        );
    }

    #[test]
    fn hook_wait_is_capped_by_the_total_deadline() {
        let mut controller = FishingController::new(FishingConfig {
            total_timeout_ms: 1_000,
        });
        assert_eq!(
            controller.advance_since(observation(true, true, None), FRAME, at(900), at(0),),
            FishingAction::MouseDown
        );
        assert_eq!(controller.wait_ms(at(900)), Some(101));
        assert_eq!(controller.wait_ms(at(999)), Some(2));
    }

    #[test]
    fn advance_since_preserves_the_earliest_start_time() {
        let config = FishingConfig {
            total_timeout_ms: 2_000,
        };
        let mut controller = FishingController::new(config);
        assert_eq!(
            controller.advance_since(observation(true, false, None), FRAME, at(1_500), at(1_000),),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance_since(observation(true, true, None), FRAME, at(3_000), at(2_900),),
            FishingAction::Failed("自动钓鱼总超时")
        );

        let mut controller = FishingController::new(config);
        assert_eq!(
            controller.advance_since(observation(true, false, None), FRAME, at(1_000), at(1_000),),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance_since(observation(true, false, None), FRAME, at(2_500), at(500),),
            FishingAction::Failed("自动钓鱼总超时")
        );
    }

    #[test]
    fn delayed_hook_observation_does_not_extend_find_bar_timeout() {
        let mut controller = FishingController::default();
        assert_eq!(
            controller.advance_since(observation(true, true, None), FRAME, at(1_000), at(0),),
            FishingAction::MouseDown
        );
        assert_eq!(
            controller.advance_since(observation(true, false, None), FRAME, at(2_000), at(900),),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance_since(observation(true, false, None), FRAME, at(6_799), at(6_800),),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance_since(observation(true, false, None), FRAME, at(6_800), at(6_700),),
            FishingAction::Failed("未检测到有效的钓鱼鱼条")
        );
    }

    #[test]
    fn terminal_success_is_idempotent() {
        let mut controller = FishingController::default();
        start_reeling(&mut controller, at(0));
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(800)),
            FishingAction::Wait
        );
        assert_eq!(
            controller.advance(observation(true, false, None), FRAME, at(1_800)),
            FishingAction::MouseUp
        );
        for (in_fishing, frame_size, now) in [
            (true, FRAME, at(1_801)),
            (false, (0, 0), at(10_000)),
            (true, FRAME, at(1_700)),
        ] {
            assert_eq!(
                controller.advance(
                    observation(in_fishing, true, Some(bar(100, 500))),
                    frame_size,
                    now,
                ),
                FishingAction::Succeeded
            );
            assert_eq!(controller.wait_ms(now), None);
            assert!(controller.is_finished());
            assert!(controller.succeeded());
            assert!(!controller.is_mouse_held());
        }
    }

    #[test]
    fn terminal_failure_is_idempotent() {
        let mut controller = FishingController::default();
        assert_eq!(
            controller.advance(observation(false, true, None), FRAME, at(0)),
            FishingAction::Failed("当前不在钓鱼状态")
        );
        for (in_fishing, frame_size, now) in [
            (true, FRAME, at(1)),
            (false, (0, 0), at(120_000)),
            (true, FRAME, at(0)),
        ] {
            assert_eq!(
                controller.advance_since(
                    observation(in_fishing, true, Some(bar(100, 500))),
                    frame_size,
                    now,
                    at(0),
                ),
                FishingAction::Failed("当前不在钓鱼状态")
            );
            assert!(controller.is_finished());
            assert!(!controller.succeeded());
            assert!(!controller.is_mouse_held());
        }
    }
}
