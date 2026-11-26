use std::time::{Duration, Instant};

pub const AUTO_OPEN_CHEST_TICK: Duration = Duration::from_millis(80);
pub const AUTO_OPEN_CHEST_TIMEOUT: Duration = Duration::from_secs(60);
pub const AUTO_OPEN_CHEST_INTERACT_DELAY: Duration = Duration::from_millis(180);
pub const AUTO_OPEN_CHEST_VERIFY_TIMEOUT: Duration = Duration::from_secs(2);
pub const AUTO_OPEN_CHEST_COOLDOWN: Duration = Duration::from_millis(500);

const INITIAL_TARGET_GRACE: Duration = Duration::from_millis(800);
const HORIZONTAL_DEAD_ZONE: f64 = 0.12;
const VERTICAL_DEAD_ZONE: f64 = 0.12;
const TURN_INTERVAL: Duration = Duration::from_millis(300);
const TURN_DIVISOR: i64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoOpenChestConfig {
    pub total_timeout_ms: u64,
    pub target_lost_timeout_ms: u64,
    pub verify_timeout_ms: u64,
}

impl Default for AutoOpenChestConfig {
    fn default() -> Self {
        Self {
            total_timeout_ms: AUTO_OPEN_CHEST_TIMEOUT.as_millis() as u64,
            target_lost_timeout_ms: 1_500,
            verify_timeout_ms: AUTO_OPEN_CHEST_VERIFY_TIMEOUT.as_millis() as u64,
        }
    }
}

impl AutoOpenChestConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1_000..=3_600_000).contains(&self.total_timeout_ms) {
            return Err("自动开箱总超时必须位于 1000..=3600000 ms");
        }
        if !(500..=10_000).contains(&self.target_lost_timeout_ms) {
            return Err("连续目标丢失超时必须位于 500..=10000 ms");
        }
        if !(500..=30_000).contains(&self.verify_timeout_ms) {
            return Err("交互确认超时必须位于 500..=30000 ms");
        }
        Ok(())
    }

    pub const fn total_timeout(&self) -> Duration {
        Duration::from_millis(self.total_timeout_ms)
    }

    pub const fn target_lost_timeout(&self) -> Duration {
        Duration::from_millis(self.target_lost_timeout_ms)
    }

    pub const fn verify_timeout(&self) -> Duration {
        Duration::from_millis(self.verify_timeout_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoOpenChestAction {
    MoveForward,
    MoveBackward,
    Turn(i32),
    Interact,
    Wait,
    Stop,
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone)]
enum Phase {
    Approach,
    StopBeforeInteract,
    InteractDelay { next_at: Instant },
    Verify { started_at: Instant },
    Cooldown { next_at: Instant },
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone)]
pub struct AutoOpenChestController {
    config: AutoOpenChestConfig,
    phase: Phase,
    started_at: Option<Instant>,
    target_lost_since: Option<Instant>,
    turn_next: bool,
    next_turn_at: Option<Instant>,
    movement_active: bool,
}

impl Default for AutoOpenChestController {
    fn default() -> Self {
        Self::new(AutoOpenChestConfig::default())
    }
}

impl AutoOpenChestController {
    pub fn new(config: AutoOpenChestConfig) -> Self {
        Self::try_new(config).expect("invalid auto open chest config")
    }

    pub fn try_new(config: AutoOpenChestConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            phase: Phase::Approach,
            started_at: None,
            target_lost_since: None,
            turn_next: true,
            next_turn_at: None,
            movement_active: false,
        })
    }

    pub const fn config(&self) -> AutoOpenChestConfig {
        self.config
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Succeeded | Phase::Failed(_))
    }

    pub fn succeeded(&self) -> bool {
        matches!(self.phase, Phase::Succeeded)
    }

    pub fn advance(
        &mut self,
        chest: Option<(i32, i32, f64)>,
        interact: bool,
        frame_size: (u32, u32),
        now: Instant,
    ) -> AutoOpenChestAction {
        if self.is_finished() {
            return self.terminal_action();
        }
        let started_at = *self.started_at.get_or_insert(now);
        if frame_size.0 == 0 || frame_size.1 == 0 {
            return self.fail("画面尺寸无效");
        }
        if now.saturating_duration_since(started_at) >= self.config.total_timeout() {
            return self.fail("自动开箱总超时");
        }

        match self.phase {
            Phase::Succeeded | Phase::Failed(_) => unreachable!("terminal state handled above"),
            Phase::Cooldown { next_at } => {
                if now >= next_at {
                    self.phase = Phase::Succeeded;
                    AutoOpenChestAction::Succeeded
                } else {
                    AutoOpenChestAction::Wait
                }
            }
            Phase::StopBeforeInteract => {
                self.phase = Phase::InteractDelay {
                    next_at: now + AUTO_OPEN_CHEST_INTERACT_DELAY,
                };
                AutoOpenChestAction::Interact
            }
            Phase::InteractDelay { next_at } => {
                if now < next_at {
                    AutoOpenChestAction::Wait
                } else {
                    self.phase = Phase::Verify { started_at: now };
                    AutoOpenChestAction::Wait
                }
            }
            Phase::Verify { started_at } => {
                if !interact && chest.is_none() {
                    self.phase = Phase::Cooldown {
                        next_at: now + AUTO_OPEN_CHEST_COOLDOWN,
                    };
                    AutoOpenChestAction::Wait
                } else if now.saturating_duration_since(started_at) >= self.config.verify_timeout()
                {
                    self.fail("交互后未确认宝箱消失")
                } else {
                    AutoOpenChestAction::Wait
                }
            }
            Phase::Approach => self.advance_approach(chest, interact, frame_size, now),
        }
    }

    fn advance_approach(
        &mut self,
        chest: Option<(i32, i32, f64)>,
        interact: bool,
        frame_size: (u32, u32),
        now: Instant,
    ) -> AutoOpenChestAction {
        if interact {
            self.target_lost_since = None;
            self.movement_active = false;
            self.phase = Phase::StopBeforeInteract;
            return AutoOpenChestAction::Stop;
        }
        let Some((x, y, _)) = chest else {
            self.movement_active = false;
            let started_at = self.started_at.expect("start time initialized by advance");
            if now.saturating_duration_since(started_at) < INITIAL_TARGET_GRACE {
                self.target_lost_since = None;
            } else {
                let lost_since = *self.target_lost_since.get_or_insert(now);
                if now.saturating_duration_since(lost_since) >= self.config.target_lost_timeout() {
                    return self.fail("连续目标丢失超时");
                }
            }
            return AutoOpenChestAction::Stop;
        };
        self.target_lost_since = None;

        let dx = i64::from(x) - i64::from(frame_size.0 / 2);
        let dy = i64::from(y) - i64::from(frame_size.1 / 2);
        let horizontal_limit = f64::from(frame_size.0) * HORIZONTAL_DEAD_ZONE;
        let vertical_limit = f64::from(frame_size.1) * VERTICAL_DEAD_ZONE;
        let should_turn = (dx as f64).abs() > horizontal_limit;
        let movement = if (dy as f64).abs() > vertical_limit {
            if dy > 0 {
                Some(AutoOpenChestAction::MoveBackward)
            } else {
                Some(AutoOpenChestAction::MoveForward)
            }
        } else {
            None
        };

        match (should_turn, movement) {
            (true, Some(movement)) => {
                let action = if self.turn_next {
                    self.turn(dx, now)
                } else {
                    self.movement_active = true;
                    movement
                };
                if !matches!(action, AutoOpenChestAction::Wait) {
                    self.turn_next = !self.turn_next;
                }
                action
            }
            (true, None) if self.movement_active => {
                self.movement_active = false;
                AutoOpenChestAction::Stop
            }
            (true, None) => self.turn(dx, now),
            (false, Some(movement)) => {
                self.movement_active = true;
                movement
            }
            (false, None) => {
                self.movement_active = false;
                AutoOpenChestAction::Stop
            }
        }
    }

    fn turn(&mut self, dx: i64, now: Instant) -> AutoOpenChestAction {
        if self.next_turn_at.is_some_and(|next_at| now < next_at) {
            return AutoOpenChestAction::Wait;
        }
        self.next_turn_at = Some(now + TURN_INTERVAL);
        AutoOpenChestAction::Turn((dx / TURN_DIVISOR).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    fn fail(&mut self, reason: &'static str) -> AutoOpenChestAction {
        self.phase = Phase::Failed(reason);
        AutoOpenChestAction::Failed(reason)
    }

    fn terminal_action(&self) -> AutoOpenChestAction {
        match self.phase {
            Phase::Succeeded => AutoOpenChestAction::Succeeded,
            Phase::Failed(reason) => AutoOpenChestAction::Failed(reason),
            _ => unreachable!("terminal_action called for a non-terminal state"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;

    fn at(milliseconds: u64) -> Instant {
        static BASE: OnceLock<Instant> = OnceLock::new();
        *BASE.get_or_init(Instant::now) + Duration::from_millis(milliseconds)
    }

    #[test]
    fn config_defaults_and_boundaries_are_validated() {
        let default = AutoOpenChestConfig::default();
        assert_eq!(default.total_timeout(), AUTO_OPEN_CHEST_TIMEOUT);
        assert_eq!(default.target_lost_timeout_ms, 1_500);
        assert_eq!(default.verify_timeout(), AUTO_OPEN_CHEST_VERIFY_TIMEOUT);
        assert_eq!(default.validate(), Ok(()));
        assert_eq!(AutoOpenChestController::default().config(), default);

        for config in [
            AutoOpenChestConfig {
                total_timeout_ms: 1_000,
                target_lost_timeout_ms: 500,
                verify_timeout_ms: 500,
            },
            AutoOpenChestConfig {
                total_timeout_ms: 3_600_000,
                target_lost_timeout_ms: 10_000,
                verify_timeout_ms: 30_000,
            },
        ] {
            assert_eq!(config.validate(), Ok(()));
            assert!(AutoOpenChestController::try_new(config).is_ok());
        }
        for (total, lost, verify) in [
            (999, 1_500, 2_000),
            (3_600_001, 1_500, 2_000),
            (60_000, 499, 2_000),
            (60_000, 10_001, 2_000),
            (60_000, 1_500, 499),
            (60_000, 1_500, 30_001),
        ] {
            let config = AutoOpenChestConfig {
                total_timeout_ms: total,
                target_lost_timeout_ms: lost,
                verify_timeout_ms: verify,
            };
            assert!(config.validate().is_err());
            assert!(AutoOpenChestController::try_new(config).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "invalid auto open chest config")]
    fn invalid_config_is_rejected_by_the_infallible_constructor() {
        AutoOpenChestController::new(AutoOpenChestConfig {
            total_timeout_ms: 0,
            target_lost_timeout_ms: 1_500,
            verify_timeout_ms: 2_000,
        });
    }

    #[test]
    fn custom_total_timeout_fails_before_any_input() {
        let mut controller = AutoOpenChestController::new(AutoOpenChestConfig {
            total_timeout_ms: 2_000,
            target_lost_timeout_ms: 10_000,
            verify_timeout_ms: 30_000,
        });
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), at(0)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(1_999)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(2_000)),
            AutoOpenChestAction::Failed("自动开箱总超时")
        );
    }

    #[test]
    fn custom_target_loss_and_verification_timeouts_are_enforced() {
        let mut target = AutoOpenChestController::new(AutoOpenChestConfig {
            total_timeout_ms: 5_000,
            target_lost_timeout_ms: 500,
            verify_timeout_ms: 30_000,
        });
        assert_eq!(
            target.advance(None, false, (1920, 1080), at(0)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            target.advance(None, false, (1920, 1080), at(800)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            target.advance(None, false, (1920, 1080), at(1_299)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            target.advance(None, false, (1920, 1080), at(1_300)),
            AutoOpenChestAction::Failed("连续目标丢失超时")
        );

        let mut verify = AutoOpenChestController::new(AutoOpenChestConfig {
            total_timeout_ms: 5_000,
            target_lost_timeout_ms: 10_000,
            verify_timeout_ms: 500,
        });
        verify.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(0));
        verify.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(0));
        assert_eq!(
            verify.advance(None, true, (1920, 1080), at(180)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            verify.advance(None, true, (1920, 1080), at(679)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            verify.advance(None, true, (1920, 1080), at(680)),
            AutoOpenChestAction::Failed("交互后未确认宝箱消失")
        );
    }

    #[test]
    fn interaction_prompt_stops_then_interacts_on_the_next_tick() {
        let now = at(0);
        let mut controller = AutoOpenChestController::default();
        assert_eq!(
            controller.advance(Some((700, 700, 0.9)), true, (1920, 1080), now),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(Some((700, 700, 0.9)), true, (1920, 1080), now),
            AutoOpenChestAction::Interact
        );
        assert_eq!(
            controller.advance(Some((700, 700, 0.9)), true, (1920, 1080), now),
            AutoOpenChestAction::Wait
        );
    }

    #[test]
    fn centered_chest_stops_and_vertical_offset_controls_movement() {
        let now = at(0);
        let mut controller = AutoOpenChestController::default();
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(Some((960, 700, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::MoveBackward
        );
        assert_eq!(
            controller.advance(Some((960, 380, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::MoveForward
        );
    }

    #[test]
    fn horizontal_offset_turns_left_or_right_while_movement_continues() {
        let now = at(0);
        let mut right = AutoOpenChestController::default();
        assert_eq!(
            right.advance(Some((1400, 700, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::Turn(220)
        );
        assert_eq!(
            right.advance(Some((1400, 700, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::MoveBackward
        );

        let mut left = AutoOpenChestController::default();
        assert_eq!(
            left.advance(Some((500, 380, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::Turn(-230)
        );
        assert_eq!(
            left.advance(Some((500, 380, 0.9)), false, (1920, 1080), now),
            AutoOpenChestAction::MoveForward
        );
    }

    #[test]
    fn entering_horizontal_only_dead_zone_releases_movement_before_turning() {
        let mut controller = AutoOpenChestController::default();
        controller.advance(Some((1400, 700, 0.9)), false, (1920, 1080), at(0));
        assert_eq!(
            controller.advance(Some((1400, 700, 0.9)), false, (1920, 1080), at(80)),
            AutoOpenChestAction::MoveBackward
        );
        assert_eq!(
            controller.advance(Some((1400, 540, 0.9)), false, (1920, 1080), at(160)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(Some((1400, 540, 0.9)), false, (1920, 1080), at(240)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            controller.advance(Some((1400, 540, 0.9)), false, (1920, 1080), at(300)),
            AutoOpenChestAction::Turn(220)
        );
    }

    #[test]
    fn camera_turns_are_rate_limited_while_movement_ticks_continue() {
        let mut controller = AutoOpenChestController::default();
        assert_eq!(
            controller.advance(Some((1400, 700, 0.9)), false, (1920, 1080), at(0)),
            AutoOpenChestAction::Turn(220)
        );
        assert_eq!(
            controller.advance(Some((1400, 700, 0.9)), false, (1920, 1080), at(80)),
            AutoOpenChestAction::MoveBackward
        );
        for elapsed in [160, 240] {
            assert_eq!(
                controller.advance(Some((1400, 700, 0.9)), false, (1920, 1080), at(elapsed)),
                AutoOpenChestAction::Wait
            );
        }
        assert_eq!(
            controller.advance(Some((1400, 700, 0.9)), false, (1920, 1080), at(320)),
            AutoOpenChestAction::Turn(220)
        );
    }

    #[test]
    fn target_loss_has_initial_grace_and_then_fails_after_continuous_timeout() {
        let mut controller = AutoOpenChestController::default();
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(0)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(800)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(2_299)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(2_300)),
            AutoOpenChestAction::Failed("连续目标丢失超时")
        );
        assert!(controller.is_finished());
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(2_301)),
            AutoOpenChestAction::Failed("连续目标丢失超时")
        );
    }

    #[test]
    fn target_reappearance_resets_the_loss_timer() {
        let mut controller = AutoOpenChestController::default();
        controller.advance(None, false, (1920, 1080), at(0));
        controller.advance(None, false, (1920, 1080), at(1_500));
        controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), at(1_600));
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(2_000)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(3_499)),
            AutoOpenChestAction::Stop
        );
        assert!(matches!(
            controller.advance(None, false, (1920, 1080), at(3_500)),
            AutoOpenChestAction::Failed(_)
        ));
    }

    #[test]
    fn total_timeout_is_enforced() {
        let mut controller = AutoOpenChestController::default();
        controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), at(0));
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), at(59_999)),
            AutoOpenChestAction::Stop
        );
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), at(60_000)),
            AutoOpenChestAction::Failed("自动开箱总超时")
        );
    }

    #[test]
    fn total_timeout_also_interrupts_pending_interaction() {
        let mut controller = AutoOpenChestController::default();
        controller.advance(Some((960, 540, 0.9)), false, (1920, 1080), at(0));
        controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(59_900));
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(59_900)),
            AutoOpenChestAction::Interact
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(60_000)),
            AutoOpenChestAction::Failed("自动开箱总超时")
        );
    }

    #[test]
    fn total_timeout_also_interrupts_cooldown_before_success() {
        let mut controller = AutoOpenChestController::new(AutoOpenChestConfig {
            total_timeout_ms: 1_000,
            ..AutoOpenChestConfig::default()
        });
        controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(0));
        controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), at(180));
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(680)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(681)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), at(1_000)),
            AutoOpenChestAction::Failed("自动开箱总超时")
        );
        assert!(!controller.succeeded());
    }

    #[test]
    fn disappeared_prompt_and_chest_verify_success_then_cooldown() {
        let now = at(0);
        let mut controller = AutoOpenChestController::default();
        controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), now);
        controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), now);
        assert_eq!(
            controller.advance(None, false, (1920, 1080), now + Duration::from_millis(180)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), now + Duration::from_millis(181)),
            AutoOpenChestAction::Wait
        );
        assert!(!controller.is_finished());
        assert_eq!(
            controller.advance(None, false, (1920, 1080), now + Duration::from_millis(680)),
            AutoOpenChestAction::Wait
        );
        assert_eq!(
            controller.advance(None, false, (1920, 1080), now + Duration::from_millis(681)),
            AutoOpenChestAction::Succeeded
        );
        assert!(controller.succeeded());
        assert_eq!(
            controller.advance(Some((960, 540, 0.9)), true, (1920, 1080), now),
            AutoOpenChestAction::Succeeded
        );
    }

    #[test]
    fn verification_fails_if_either_the_prompt_or_chest_remains() {
        for (chest, interact) in [(None, true), (Some((960, 540, 0.9)), false)] {
            let now = at(0);
            let mut controller = AutoOpenChestController::default();
            controller.advance(chest, true, (1920, 1080), now);
            controller.advance(chest, true, (1920, 1080), now);
            assert_eq!(
                controller.advance(
                    chest,
                    interact,
                    (1920, 1080),
                    now + Duration::from_millis(180)
                ),
                AutoOpenChestAction::Wait
            );
            assert_eq!(
                controller.advance(
                    chest,
                    interact,
                    (1920, 1080),
                    now + Duration::from_millis(2_180)
                ),
                AutoOpenChestAction::Failed("交互后未确认宝箱消失")
            );
            assert!(!controller.succeeded());
        }
    }
}
