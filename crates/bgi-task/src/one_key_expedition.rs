use std::time::{Duration, Instant};

use crate::expedition_recognition::ExpeditionPageState;

pub const ONE_KEY_EXPEDITION_TICK: Duration = Duration::from_millis(100);
pub const ONE_KEY_EXPEDITION_TIMEOUT: Duration = Duration::from_secs(20);
pub const ONE_KEY_EXPEDITION_COLLECT_DELAY: Duration = Duration::from_millis(1100);
pub const ONE_KEY_EXPEDITION_RESPONSE_DELAY: Duration = Duration::from_millis(500);
pub const ONE_KEY_EXPEDITION_VERIFY_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OneKeyExpeditionConfig {
    pub total_timeout_ms: u64,
    pub collect_delay_ms: u64,
    pub response_delay_ms: u64,
    pub verify_timeout_ms: u64,
}

impl Default for OneKeyExpeditionConfig {
    fn default() -> Self {
        Self {
            total_timeout_ms: ONE_KEY_EXPEDITION_TIMEOUT.as_millis() as u64,
            collect_delay_ms: ONE_KEY_EXPEDITION_COLLECT_DELAY.as_millis() as u64,
            response_delay_ms: ONE_KEY_EXPEDITION_RESPONSE_DELAY.as_millis() as u64,
            verify_timeout_ms: ONE_KEY_EXPEDITION_VERIFY_TIMEOUT.as_millis() as u64,
        }
    }
}

impl OneKeyExpeditionConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(5_000..=120_000).contains(&self.total_timeout_ms) {
            return Err("一键派遣总超时必须位于 5000..=120000 ms");
        }
        if !(500..=10_000).contains(&self.collect_delay_ms) {
            return Err("领取后的等待时间必须位于 500..=10000 ms");
        }
        if !(200..=5_000).contains(&self.response_delay_ms) {
            return Err("派遣后的响应等待必须位于 200..=5000 ms");
        }
        if !(1_000..=15_000).contains(&self.verify_timeout_ms) {
            return Err("再次派遣确认超时必须位于 1000..=15000 ms");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OneKeyExpeditionTarget {
    pub state: ExpeditionPageState,
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneKeyExpeditionAction {
    ClickCollect { x: i32, y: i32 },
    ClickRedispatch { x: i32, y: i32 },
    PressEscape,
    Wait,
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone)]
enum Phase {
    WaitingPage,
    CollectSettling { next_at: Instant },
    WaitingRedispatch,
    VerifyingRedispatch { started_at: Instant },
    Exiting { next_at: Instant },
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone)]
pub struct OneKeyExpeditionController {
    config: OneKeyExpeditionConfig,
    phase: Phase,
    started_at: Option<Instant>,
}

impl OneKeyExpeditionController {
    pub fn new(config: OneKeyExpeditionConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            phase: Phase::WaitingPage,
            started_at: None,
        })
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Succeeded | Phase::Failed(_))
    }

    pub fn succeeded(&self) -> bool {
        matches!(self.phase, Phase::Succeeded)
    }

    pub fn advance(
        &mut self,
        target: Option<OneKeyExpeditionTarget>,
        frame_size: (u32, u32),
        now: Instant,
    ) -> OneKeyExpeditionAction {
        if self.is_finished() {
            return self.terminal_action();
        }
        let started_at = *self.started_at.get_or_insert(now);
        if frame_size.0 == 0 || frame_size.1 == 0 {
            return self.fail("画面尺寸无效");
        }
        if now.saturating_duration_since(started_at)
            >= Duration::from_millis(self.config.total_timeout_ms)
        {
            return self.fail("一键派遣总超时");
        }
        if let Some(target) = target
            && (target.x < 0
                || target.y < 0
                || target.x >= frame_size.0 as i32
                || target.y >= frame_size.1 as i32)
        {
            return self.fail("派遣模板点击坐标越界");
        }

        match self.phase {
            Phase::Succeeded | Phase::Failed(_) => unreachable!("terminal state handled above"),
            Phase::WaitingPage => match target {
                Some(target) if target.state == ExpeditionPageState::CollectAvailable => {
                    self.phase = Phase::CollectSettling {
                        next_at: now + Duration::from_millis(self.config.collect_delay_ms),
                    };
                    OneKeyExpeditionAction::ClickCollect {
                        x: target.x,
                        y: target.y,
                    }
                }
                Some(target) if target.state == ExpeditionPageState::RedispatchAvailable => {
                    self.phase = Phase::VerifyingRedispatch { started_at: now };
                    OneKeyExpeditionAction::ClickRedispatch {
                        x: target.x,
                        y: target.y,
                    }
                }
                _ => OneKeyExpeditionAction::Wait,
            },
            Phase::CollectSettling { next_at } => {
                if now >= next_at {
                    self.phase = Phase::WaitingRedispatch;
                    return self.advance(target, frame_size, now);
                }
                OneKeyExpeditionAction::Wait
            }
            Phase::WaitingRedispatch => match target {
                Some(target) if target.state == ExpeditionPageState::RedispatchAvailable => {
                    self.phase = Phase::VerifyingRedispatch { started_at: now };
                    OneKeyExpeditionAction::ClickRedispatch {
                        x: target.x,
                        y: target.y,
                    }
                }
                _ => OneKeyExpeditionAction::Wait,
            },
            Phase::VerifyingRedispatch { started_at } => {
                if target
                    .is_some_and(|target| target.state == ExpeditionPageState::RedispatchAvailable)
                {
                    if now.saturating_duration_since(started_at)
                        >= Duration::from_millis(self.config.verify_timeout_ms)
                    {
                        return self.fail("再次派遣后图标仍未消失，未确认派遣成功");
                    }
                    OneKeyExpeditionAction::Wait
                } else if now.saturating_duration_since(started_at)
                    >= Duration::from_millis(self.config.response_delay_ms)
                {
                    self.phase = Phase::Exiting {
                        next_at: now + Duration::from_millis(self.config.response_delay_ms),
                    };
                    OneKeyExpeditionAction::PressEscape
                } else {
                    OneKeyExpeditionAction::Wait
                }
            }
            Phase::Exiting { next_at } => {
                if now >= next_at {
                    self.phase = Phase::Succeeded;
                    OneKeyExpeditionAction::Succeeded
                } else {
                    OneKeyExpeditionAction::Wait
                }
            }
        }
    }

    fn terminal_action(&self) -> OneKeyExpeditionAction {
        match self.phase {
            Phase::Succeeded => OneKeyExpeditionAction::Succeeded,
            Phase::Failed(reason) => OneKeyExpeditionAction::Failed(reason),
            _ => unreachable!("nonterminal phase"),
        }
    }

    fn fail(&mut self, reason: &'static str) -> OneKeyExpeditionAction {
        self.phase = Phase::Failed(reason);
        OneKeyExpeditionAction::Failed(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(state: ExpeditionPageState) -> OneKeyExpeditionTarget {
        OneKeyExpeditionTarget {
            state,
            x: 100,
            y: 200,
        }
    }

    fn at(ms: u64) -> Instant {
        Instant::now() + Duration::from_millis(ms)
    }

    #[test]
    fn collects_once_waits_then_redispatches_once_and_exits() {
        let mut controller =
            OneKeyExpeditionController::new(OneKeyExpeditionConfig::default()).unwrap();
        let now = at(0);
        assert_eq!(
            controller.advance(
                Some(target(ExpeditionPageState::CollectAvailable)),
                (1920, 1080),
                now
            ),
            OneKeyExpeditionAction::ClickCollect { x: 100, y: 200 }
        );
        assert_eq!(
            controller.advance(
                Some(target(ExpeditionPageState::CollectAvailable)),
                (1920, 1080),
                at(500)
            ),
            OneKeyExpeditionAction::Wait
        );
        assert_eq!(
            controller.advance(
                Some(target(ExpeditionPageState::RedispatchAvailable)),
                (1920, 1080),
                at(1200)
            ),
            OneKeyExpeditionAction::ClickRedispatch { x: 100, y: 200 }
        );
        assert_eq!(
            controller.advance(
                Some(target(ExpeditionPageState::RedispatchAvailable)),
                (1920, 1080),
                at(1500)
            ),
            OneKeyExpeditionAction::Wait
        );
        assert_eq!(
            controller.advance(None, (1920, 1080), at(1800)),
            OneKeyExpeditionAction::PressEscape
        );
        assert_eq!(
            controller.advance(None, (1920, 1080), at(2200)),
            OneKeyExpeditionAction::Wait
        );
        assert_eq!(
            controller.advance(None, (1920, 1080), at(2400)),
            OneKeyExpeditionAction::Succeeded
        );
    }

    #[test]
    fn unknown_or_ambiguous_page_never_clicks() {
        let mut controller =
            OneKeyExpeditionController::new(OneKeyExpeditionConfig::default()).unwrap();
        for ms in 0..5000 {
            assert_eq!(
                controller.advance(None, (1920, 1080), at(ms)),
                OneKeyExpeditionAction::Wait
            );
        }
        assert_eq!(
            controller.advance(None, (1920, 1080), at(20_000)),
            OneKeyExpeditionAction::Failed("一键派遣总超时")
        );
    }

    #[test]
    fn redispatch_can_run_without_collect_and_requires_icon_clear() {
        let mut controller =
            OneKeyExpeditionController::new(OneKeyExpeditionConfig::default()).unwrap();
        assert_eq!(
            controller.advance(
                Some(target(ExpeditionPageState::RedispatchAvailable)),
                (1920, 1080),
                at(0)
            ),
            OneKeyExpeditionAction::ClickRedispatch { x: 100, y: 200 }
        );
        assert_eq!(
            controller.advance(
                Some(target(ExpeditionPageState::RedispatchAvailable)),
                (1920, 1080),
                at(3000)
            ),
            OneKeyExpeditionAction::Failed("再次派遣后图标仍未消失，未确认派遣成功")
        );
        assert_eq!(
            controller.advance(None, (1920, 1080), at(4000)),
            OneKeyExpeditionAction::Failed("再次派遣后图标仍未消失，未确认派遣成功")
        );
    }

    #[test]
    fn invalid_target_bounds_fail_before_any_click() {
        let mut controller =
            OneKeyExpeditionController::new(OneKeyExpeditionConfig::default()).unwrap();
        let invalid = OneKeyExpeditionTarget {
            state: ExpeditionPageState::CollectAvailable,
            x: 1920,
            y: 100,
        };
        assert_eq!(
            controller.advance(Some(invalid), (1920, 1080), at(0)),
            OneKeyExpeditionAction::Failed("派遣模板点击坐标越界")
        );
    }
}
