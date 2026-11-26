use std::time::{Duration, Instant};

pub const MAX_CLICKS_PER_RUN: usize = 30;
pub const MAX_EMPTY_SCANS: usize = 3;
const CLICK_TO_CONTINUE_CHECK: Duration = Duration::from_millis(160);
const EMPTY_RESCAN_DELAY: Duration = Duration::from_millis(180);
const AFTER_ESCAPE: Duration = Duration::from_millis(220);
const MAX_CONTINUE_CHECKS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewardClaimConfig {
    pub max_clicks: usize,
    pub total_timeout_ms: u64,
}

impl Default for RewardClaimConfig {
    fn default() -> Self {
        Self {
            max_clicks: MAX_CLICKS_PER_RUN,
            total_timeout_ms: 30_000,
        }
    }
}

impl RewardClaimConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1..=100).contains(&self.max_clicks) {
            return Err("领奖最大点击次数必须位于 1..=100");
        }
        if !(1_000..=600_000).contains(&self.total_timeout_ms) {
            return Err("领奖总超时必须位于 1000..=600000 ms");
        }
        Ok(())
    }

    pub const fn total_timeout(&self) -> Duration {
        Duration::from_millis(self.total_timeout_ms)
    }

    pub fn timed_out(&self, started_at: Instant, now: Instant) -> bool {
        now.saturating_duration_since(started_at) >= self.total_timeout()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardKind {
    ClaimText,
    ClaimGift,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RewardCandidate {
    pub kind: RewardKind,
    pub center: (i32, i32),
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RewardClaimAction {
    Click(RewardCandidate),
    PressEscape,
    Wait,
    Finished,
}

#[derive(Debug, Clone)]
enum Phase {
    Scanning,
    ContinueChecks { remaining: usize, next_at: Instant },
    EscapeDelay { next_at: Instant },
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardClaimMode {
    CurrentPage,
    VisiblePageContinuous,
}

impl RewardClaimMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CurrentPage => "一键领取当前页",
            Self::VisiblePageContinuous => "持续领取当前可见页",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardClaimFinish {
    EmptyPage,
    EmptyScans,
    ClickLimit,
    Timeout,
}

#[derive(Debug, Clone)]
pub struct RewardClaimController {
    config: RewardClaimConfig,
    mode: RewardClaimMode,
    phase: Phase,
    clicked: usize,
    started_at: Option<Instant>,
    empty_scans: usize,
    rescan_at: Option<Instant>,
    finish_reason: Option<RewardClaimFinish>,
}

impl Default for RewardClaimController {
    fn default() -> Self {
        Self::new(RewardClaimConfig::default())
    }
}

impl RewardClaimController {
    pub fn new(config: RewardClaimConfig) -> Self {
        Self::try_new(config).expect("invalid reward claim config")
    }

    pub fn try_new(config: RewardClaimConfig) -> Result<Self, &'static str> {
        Self::try_new_with_mode(config, RewardClaimMode::CurrentPage)
    }

    pub fn try_new_with_mode(
        config: RewardClaimConfig,
        mode: RewardClaimMode,
    ) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            mode,
            phase: Phase::Scanning,
            clicked: 0,
            started_at: None,
            empty_scans: 0,
            rescan_at: None,
            finish_reason: None,
        })
    }

    pub const fn mode(&self) -> RewardClaimMode {
        self.mode
    }

    pub const fn config(&self) -> RewardClaimConfig {
        self.config
    }

    pub fn clicked_count(&self) -> usize {
        self.clicked
    }

    pub const fn finish_reason(&self) -> Option<RewardClaimFinish> {
        self.finish_reason
    }

    pub fn wait_ms(&self, now: Instant) -> Option<u64> {
        let next_at = match self.phase {
            Phase::ContinueChecks { next_at, .. } | Phase::EscapeDelay { next_at } => next_at,
            Phase::Scanning => self.rescan_at?,
            Phase::Finished => return None,
        };
        Some(next_at.saturating_duration_since(now).as_millis() as u64 + 1)
    }

    pub fn advance(
        &mut self,
        candidates: &[RewardCandidate],
        continue_tip: bool,
        now: Instant,
    ) -> RewardClaimAction {
        let started_at = *self.started_at.get_or_insert(now);
        self.advance_since(candidates, continue_tip, now, started_at)
    }

    pub fn advance_since(
        &mut self,
        candidates: &[RewardCandidate],
        continue_tip: bool,
        now: Instant,
        started_at: Instant,
    ) -> RewardClaimAction {
        let started_at = *self.started_at.get_or_insert(started_at);
        if matches!(self.phase, Phase::Finished) {
            return RewardClaimAction::Finished;
        }
        if self.config.timed_out(started_at, now) {
            self.phase = Phase::Finished;
            self.finish_reason = Some(RewardClaimFinish::Timeout);
            return RewardClaimAction::Finished;
        }

        match &self.phase {
            Phase::Finished => return RewardClaimAction::Finished,
            Phase::ContinueChecks { remaining, next_at } if now < *next_at => {
                return RewardClaimAction::Wait;
            }
            Phase::EscapeDelay { next_at } if now < *next_at => {
                return RewardClaimAction::Wait;
            }
            Phase::Scanning if self.rescan_at.is_some_and(|at| now < at) => {
                return RewardClaimAction::Wait;
            }
            Phase::Scanning => {}
            Phase::ContinueChecks { remaining, .. } if continue_tip => {
                self.phase = Phase::EscapeDelay {
                    next_at: now + AFTER_ESCAPE,
                };
                return RewardClaimAction::PressEscape;
            }
            Phase::ContinueChecks { remaining, .. } => {
                let remaining = *remaining - 1;
                self.phase = if remaining == 0 {
                    Phase::Scanning
                } else {
                    Phase::ContinueChecks {
                        remaining,
                        next_at: now + CLICK_TO_CONTINUE_CHECK,
                    }
                };
                return RewardClaimAction::Wait;
            }
            Phase::EscapeDelay { .. } => {
                self.phase = Phase::Scanning;
                return RewardClaimAction::Wait;
            }
        }

        if self.clicked >= self.config.max_clicks {
            self.phase = Phase::Finished;
            self.finish_reason = Some(RewardClaimFinish::ClickLimit);
            return RewardClaimAction::Finished;
        }
        let Some(candidate) = candidates.first() else {
            if self.mode == RewardClaimMode::VisiblePageContinuous
                && self.empty_scans + 1 < MAX_EMPTY_SCANS
            {
                self.empty_scans += 1;
                self.rescan_at = Some(now + EMPTY_RESCAN_DELAY);
                return RewardClaimAction::Wait;
            }
            self.phase = Phase::Finished;
            self.finish_reason = Some(match self.mode {
                RewardClaimMode::CurrentPage => RewardClaimFinish::EmptyPage,
                RewardClaimMode::VisiblePageContinuous => RewardClaimFinish::EmptyScans,
            });
            return RewardClaimAction::Finished;
        };
        self.empty_scans = 0;
        self.rescan_at = None;
        self.clicked += 1;
        self.phase = Phase::ContinueChecks {
            remaining: MAX_CONTINUE_CHECKS,
            next_at: now + CLICK_TO_CONTINUE_CHECK,
        };
        RewardClaimAction::Click(candidate.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn candidate(kind: RewardKind, x: i32, y: i32) -> RewardCandidate {
        RewardCandidate {
            kind,
            center: (x, y),
            score: 0.9,
        }
    }

    #[test]
    fn config_defaults_and_boundaries_are_validated() {
        let default = RewardClaimConfig::default();
        assert_eq!(default.max_clicks, MAX_CLICKS_PER_RUN);
        assert_eq!(default.total_timeout_ms, 30_000);
        assert_eq!(default.validate(), Ok(()));
        assert_eq!(RewardClaimController::default().config(), default);

        for config in [
            RewardClaimConfig {
                max_clicks: 1,
                total_timeout_ms: 1_000,
            },
            RewardClaimConfig {
                max_clicks: 100,
                total_timeout_ms: 600_000,
            },
        ] {
            assert_eq!(config.validate(), Ok(()));
            assert!(RewardClaimController::try_new(config).is_ok());
        }
        for (max_clicks, total_timeout_ms) in [(0, 30_000), (101, 30_000), (30, 999), (30, 600_001)]
        {
            let config = RewardClaimConfig {
                max_clicks,
                total_timeout_ms,
            };
            assert!(config.validate().is_err());
            assert!(RewardClaimController::try_new(config).is_err());
        }

        let now = Instant::now();
        assert!(!default.timed_out(now, now + Duration::from_millis(29_999)));
        assert!(default.timed_out(now, now + Duration::from_millis(30_000)));
        assert!(!default.timed_out(now, now - Duration::from_millis(1)));
    }

    #[test]
    #[should_panic(expected = "invalid reward claim config")]
    fn invalid_config_is_rejected_by_the_infallible_constructor() {
        RewardClaimController::new(RewardClaimConfig {
            max_clicks: 0,
            total_timeout_ms: 30_000,
        });
    }

    #[test]
    fn custom_click_limit_is_enforced_without_extra_input() {
        let now = Instant::now();
        let config = RewardClaimConfig {
            max_clicks: 1,
            total_timeout_ms: 1_000,
        };
        let mut controller = RewardClaimController::new(config);
        let candidates = [candidate(RewardKind::ClaimGift, 50, 50)];
        assert_eq!(
            controller.advance(&candidates, false, now),
            RewardClaimAction::Click(candidates[0].clone())
        );
        for elapsed in [160, 320, 480] {
            assert_eq!(
                controller.advance(&candidates, false, now + Duration::from_millis(elapsed)),
                RewardClaimAction::Wait
            );
        }
        assert_eq!(
            controller.advance(&candidates, true, now + Duration::from_millis(480)),
            RewardClaimAction::Finished
        );
        assert_eq!(controller.clicked_count(), 1);
    }

    #[test]
    fn total_timeout_finishes_without_clicking_or_pressing_escape() {
        let now = Instant::now();
        let config = RewardClaimConfig {
            max_clicks: 100,
            total_timeout_ms: 1_000,
        };
        let candidates = [candidate(RewardKind::ClaimText, 10, 10)];

        let mut before_escape = RewardClaimController::new(config);
        assert_eq!(
            before_escape.advance(&candidates, false, now),
            RewardClaimAction::Click(candidates[0].clone())
        );
        assert_eq!(
            before_escape.advance(&candidates, true, now + config.total_timeout()),
            RewardClaimAction::Finished
        );
        assert_eq!(before_escape.clicked_count(), 1);
        assert_eq!(
            before_escape.advance(&candidates, true, now + Duration::from_secs(1)),
            RewardClaimAction::Finished
        );

        let mut before_click = RewardClaimController::new(config);
        assert_eq!(
            before_click.advance(&candidates, false, now),
            RewardClaimAction::Click(candidates[0].clone())
        );
        for elapsed in [160, 320, 480] {
            assert_eq!(
                before_click.advance(&[], false, now + Duration::from_millis(elapsed)),
                RewardClaimAction::Wait
            );
        }
        assert_eq!(
            before_click.advance(&candidates, false, now + config.total_timeout()),
            RewardClaimAction::Finished
        );
        assert_eq!(before_click.clicked_count(), 1);
    }

    #[test]
    fn advance_since_keeps_the_original_deadline_across_slow_observations() {
        let now = Instant::now();
        let mut controller = RewardClaimController::new(RewardClaimConfig {
            max_clicks: 100,
            total_timeout_ms: 1_000,
        });
        let candidates = [candidate(RewardKind::ClaimGift, 50, 50)];
        assert_eq!(
            controller.advance_since(&candidates, false, now, now),
            RewardClaimAction::Click(candidates[0].clone())
        );
        assert_eq!(
            controller.advance_since(
                &candidates,
                true,
                now + Duration::from_millis(1_000),
                now + Duration::from_millis(900),
            ),
            RewardClaimAction::Finished
        );
        assert_eq!(controller.clicked_count(), 1);
    }

    #[test]
    fn clicks_first_candidate_then_checks_continue_tip_three_times() {
        let now = Instant::now();
        let mut controller = RewardClaimController::default();
        let candidates = [candidate(RewardKind::ClaimText, 100, 100)];
        assert_eq!(
            controller.advance(&candidates, false, now),
            RewardClaimAction::Click(candidates[0].clone())
        );
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(159)),
            RewardClaimAction::Wait
        );
        for elapsed in [160, 320] {
            assert_eq!(
                controller.advance(&[], false, now + Duration::from_millis(elapsed)),
                RewardClaimAction::Wait
            );
        }
        assert_eq!(
            controller.advance(&candidates, false, now + Duration::from_millis(480)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.advance(&candidates, false, now + Duration::from_millis(480)),
            RewardClaimAction::Click(candidates[0].clone())
        );
        assert_eq!(controller.clicked_count(), 2);
    }

    #[test]
    fn continue_tip_presses_escape_and_waits_before_next_scan() {
        let now = Instant::now();
        let mut controller = RewardClaimController::default();
        let candidates = [candidate(RewardKind::ClaimGift, 50, 50)];
        controller.advance(&candidates, false, now);
        assert_eq!(
            controller.advance(&[], true, now + Duration::from_millis(160)),
            RewardClaimAction::PressEscape
        );
        assert_eq!(
            controller.advance(&candidates, false, now + Duration::from_millis(379)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.advance(&candidates, false, now + Duration::from_millis(380)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.advance(&candidates, false, now + Duration::from_millis(380)),
            RewardClaimAction::Click(candidates[0].clone())
        );
    }

    #[test]
    fn continuous_mode_waits_for_three_separated_empty_scans() {
        let now = Instant::now();
        let config = RewardClaimConfig::default();
        let mut controller = RewardClaimController::try_new_with_mode(
            config,
            RewardClaimMode::VisiblePageContinuous,
        )
        .unwrap();

        assert_eq!(controller.advance(&[], false, now), RewardClaimAction::Wait);
        assert_eq!(controller.wait_ms(now), Some(181));
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(179)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(180)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.wait_ms(now + Duration::from_millis(180)),
            Some(181)
        );
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(360)),
            RewardClaimAction::Finished
        );
        assert_eq!(
            controller.finish_reason(),
            Some(RewardClaimFinish::EmptyScans)
        );
        assert_eq!(controller.clicked_count(), 0);
        assert_eq!(controller.mode(), RewardClaimMode::VisiblePageContinuous);
    }

    #[test]
    fn continuous_mode_resets_empty_scans_when_a_candidate_returns() {
        let now = Instant::now();
        let mut controller = RewardClaimController::try_new_with_mode(
            RewardClaimConfig::default(),
            RewardClaimMode::VisiblePageContinuous,
        )
        .unwrap();
        let candidates = [candidate(RewardKind::ClaimText, 20, 30)];

        assert_eq!(controller.advance(&[], false, now), RewardClaimAction::Wait);
        assert_eq!(
            controller.advance(&candidates, false, now + Duration::from_millis(180)),
            RewardClaimAction::Click(candidates[0].clone())
        );
        for elapsed in [340, 500, 660] {
            assert_eq!(
                controller.advance(&[], false, now + Duration::from_millis(elapsed)),
                RewardClaimAction::Wait
            );
        }
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(660)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.wait_ms(now + Duration::from_millis(660)),
            Some(181)
        );
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(840)),
            RewardClaimAction::Wait
        );
        assert_eq!(
            controller.advance(&[], false, now + Duration::from_millis(1_020)),
            RewardClaimAction::Finished
        );
        assert_eq!(controller.clicked_count(), 1);
        assert_eq!(
            controller.finish_reason(),
            Some(RewardClaimFinish::EmptyScans)
        );
    }

    #[test]
    fn continuous_mode_keeps_click_limit_and_timeout_guards() {
        let now = Instant::now();
        let candidates = [candidate(RewardKind::ClaimGift, 20, 30)];
        let config = RewardClaimConfig {
            max_clicks: 1,
            total_timeout_ms: 2_000,
        };
        let mut limited = RewardClaimController::try_new_with_mode(
            config,
            RewardClaimMode::VisiblePageContinuous,
        )
        .unwrap();
        limited.advance(&candidates, false, now);
        for elapsed in [160, 320, 480] {
            limited.advance(&[], false, now + Duration::from_millis(elapsed));
        }
        assert_eq!(
            limited.advance(&candidates, false, now + Duration::from_millis(480)),
            RewardClaimAction::Finished
        );
        assert_eq!(limited.finish_reason(), Some(RewardClaimFinish::ClickLimit));
        assert_eq!(limited.clicked_count(), 1);

        let mut timed_out = RewardClaimController::try_new_with_mode(
            config,
            RewardClaimMode::VisiblePageContinuous,
        )
        .unwrap();
        timed_out.advance(&[], false, now);
        assert_eq!(
            timed_out.advance_since(
                &candidates,
                false,
                now + config.total_timeout(),
                now + Duration::from_millis(1_900)
            ),
            RewardClaimAction::Finished
        );
        assert_eq!(timed_out.finish_reason(), Some(RewardClaimFinish::Timeout));
        assert_eq!(timed_out.clicked_count(), 0);
    }

    #[test]
    fn empty_page_finishes_and_click_limit_is_enforced() {
        let now = Instant::now();
        let mut empty = RewardClaimController::default();
        assert_eq!(empty.advance(&[], false, now), RewardClaimAction::Finished);
        assert_eq!(empty.advance(&[], false, now), RewardClaimAction::Finished);

        let mut limited = RewardClaimController::default();
        let candidates = [candidate(RewardKind::ClaimText, 10, 10)];
        for index in 0..MAX_CLICKS_PER_RUN {
            let at = now + Duration::from_millis(index as u64 * 500);
            limited.advance(&candidates, false, at);
            limited.advance(&[], false, at + Duration::from_millis(160));
            limited.advance(&[], false, at + Duration::from_millis(320));
            assert_eq!(
                limited.advance(&candidates, false, at + Duration::from_millis(480)),
                RewardClaimAction::Wait
            );
            if index + 1 == MAX_CLICKS_PER_RUN {
                assert_eq!(
                    limited.advance(&candidates, false, at + Duration::from_millis(480)),
                    RewardClaimAction::Finished
                );
            } else {
                assert_eq!(
                    limited.advance(&candidates, false, at + Duration::from_millis(480)),
                    RewardClaimAction::Click(candidates[0].clone())
                );
            }
        }
        assert_eq!(
            limited.advance(&candidates, false, now + Duration::from_secs(20)),
            RewardClaimAction::Finished
        );
    }
}
