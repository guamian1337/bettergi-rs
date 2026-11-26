//! Pure, resolution-independent state machine for automatic cooking.
//!
//! The input adapter owns capture, cooking-bar recognition, timing, and input
//! dispatch. It feeds one normalized progress sample per frame and executes the
//! returned [`CookingEvent::CompletionTriggered`] action.

use std::ops::RangeInclusive;
use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoCookTaskConfig {
    pub check_interval_ms: u64,
    pub stop_on_recover: bool,
    pub total_timeout_ms: u64,
}

impl Default for AutoCookTaskConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 10,
            stop_on_recover: true,
            total_timeout_ms: 120_000,
        }
    }
}

impl AutoCookTaskConfig {
    pub fn check_interval(&self) -> Result<Duration, CookingConfigError> {
        self.validate()?;
        Ok(Duration::from_millis(self.check_interval_ms))
    }

    pub fn validate(&self) -> Result<(), CookingConfigError> {
        if !(1..=10_000).contains(&self.check_interval_ms) {
            return Err(CookingConfigError::InvalidCheckInterval);
        }
        if !(1..=3_600_000).contains(&self.total_timeout_ms) {
            return Err(CookingConfigError::InvalidTotalTimeout);
        }
        Ok(())
    }

    pub fn timed_out(&self, started: std::time::Instant, now: std::time::Instant) -> bool {
        now.duration_since(started) >= Duration::from_millis(self.total_timeout_ms)
    }
}

/// Configuration expressed entirely in normalized progress, never pixels or
/// screen coordinates, so the same configuration applies at 1080p and 1440p.
#[derive(Debug, Clone, PartialEq)]
pub struct CookingConfig {
    optimal_range: RangeInclusive<f64>,
    stable_frames: u32,
}

impl CookingConfig {
    /// Creates a configuration. Both range bounds must be finite and in
    /// `0..=1`, and at least one stable frame is required.
    pub fn new(
        optimal_min: f64,
        optimal_max: f64,
        stable_frames: u32,
    ) -> Result<Self, CookingConfigError> {
        if !optimal_min.is_finite() || !optimal_max.is_finite() {
            return Err(CookingConfigError::NonFiniteBound);
        }
        if !(0.0..=1.0).contains(&optimal_min) || !(0.0..=1.0).contains(&optimal_max) {
            return Err(CookingConfigError::BoundOutOfRange);
        }
        if optimal_min > optimal_max {
            return Err(CookingConfigError::InvertedRange);
        }
        if stable_frames == 0 {
            return Err(CookingConfigError::ZeroStableFrames);
        }
        Ok(Self {
            optimal_range: optimal_min..=optimal_max,
            stable_frames,
        })
    }

    pub fn optimal_range(&self) -> RangeInclusive<f64> {
        self.optimal_range.clone()
    }

    pub fn stable_frames(&self) -> u32 {
        self.stable_frames
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CookingConfigError {
    #[error("最佳熟度区间端点必须为有限数")]
    NonFiniteBound,
    #[error("最佳熟度区间端点必须位于 0..=1")]
    BoundOutOfRange,
    #[error("最佳熟度区间下界不能大于上界")]
    InvertedRange,
    #[error("连续稳定帧数必须大于 0")]
    ZeroStableFrames,
    #[error("烹饪检查间隔必须位于 1..=10000 ms")]
    InvalidCheckInterval,
    #[error("烹饪总超时必须位于 1..=3600000 ms")]
    InvalidTotalTimeout,
}

/// One frame from the future capture/recognition adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CookingFrame {
    /// Normalized cooking progress in `0..=1`.
    pub progress: f64,
    pub page_visible: bool,
    pub cancelled: bool,
}

impl CookingFrame {
    pub const fn new(progress: f64, page_visible: bool, cancelled: bool) -> Self {
        Self {
            progress,
            page_visible,
            cancelled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CookingState {
    /// The cooking page is absent, or the machine has just been reset.
    Inactive,
    /// The page is visible but progress is outside the optimal range.
    Observing,
    /// Progress is in range, with fewer than the required consecutive frames.
    Accumulating,
    /// Completion was emitted and is latched for this visible page session.
    Completed,
    /// Terminal until an explicit [`CookingStateMachine::reset`].
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CookingEvent {
    PageEntered,
    /// An in-range streak was interrupted; accumulation starts over.
    StabilityLost,
    /// Request exactly one completion action from the input adapter.
    CompletionTriggered {
        progress: f64,
    },
    PageLeft,
    Cancelled,
    Reset,
}

/// Returns events rather than performing input, keeping the core deterministic
/// and directly testable. Events are ordered in the order they occurred.
pub type CookingEvents = Vec<CookingEvent>;

#[derive(Debug, Clone)]
pub struct CookingStateMachine {
    config: CookingConfig,
    state: CookingState,
    stable_frames: u32,
}

impl CookingStateMachine {
    pub fn new(config: CookingConfig) -> Self {
        Self {
            config,
            state: CookingState::Inactive,
            stable_frames: 0,
        }
    }

    pub const fn state(&self) -> CookingState {
        self.state
    }

    pub const fn stable_frames(&self) -> u32 {
        self.stable_frames
    }

    pub const fn config(&self) -> &CookingConfig {
        &self.config
    }

    /// Validates and consumes exactly one frame. Invalid progress never mutates
    /// the machine. Cancellation takes precedence over all frame observations.
    pub fn on_frame(&mut self, frame: CookingFrame) -> Result<CookingEvents, CookingInputError> {
        Self::validate_progress(frame.progress)?;

        if frame.cancelled {
            if self.state == CookingState::Cancelled {
                return Ok(CookingEvents::new());
            }
            self.clear_runtime_state();
            self.state = CookingState::Cancelled;
            return Ok(vec![CookingEvent::Cancelled]);
        }

        // Cancellation remains terminal until reset, even if later frames say
        // the page is visible or the cancellation flag has cleared.
        if self.state == CookingState::Cancelled {
            return Ok(CookingEvents::new());
        }

        if !frame.page_visible {
            return Ok(self.leave_page());
        }

        // A completion latch lasts for the whole visible page session. Repeated
        // captures of the same cooking bar cannot emit another action.
        if self.state == CookingState::Completed {
            return Ok(CookingEvents::new());
        }

        let mut events = CookingEvents::new();
        if self.state == CookingState::Inactive {
            self.state = CookingState::Observing;
            events.push(CookingEvent::PageEntered);
        }

        if self.config.optimal_range.contains(&frame.progress) {
            self.stable_frames = self.stable_frames.saturating_add(1);
            if self.state == CookingState::Observing {
                self.state = CookingState::Accumulating;
            }
            if self.stable_frames >= self.config.stable_frames {
                self.clear_runtime_state();
                self.state = CookingState::Completed;
                events.push(CookingEvent::CompletionTriggered {
                    progress: frame.progress,
                });
            }
        } else {
            if self.state == CookingState::Accumulating {
                events.push(CookingEvent::StabilityLost);
            }
            self.clear_runtime_state();
            self.state = CookingState::Observing;
        }

        Ok(events)
    }

    /// Clears both streak and completion/cancellation latches. The next visible
    /// frame is treated as a new cooking session.
    pub fn reset(&mut self) -> CookingEvent {
        self.clear_runtime_state();
        self.state = CookingState::Inactive;
        CookingEvent::Reset
    }

    fn validate_progress(progress: f64) -> Result<(), CookingInputError> {
        if !progress.is_finite() {
            return Err(CookingInputError::NonFiniteProgress(progress));
        }
        if !(0.0..=1.0).contains(&progress) {
            return Err(CookingInputError::ProgressOutOfRange(progress));
        }
        Ok(())
    }

    fn leave_page(&mut self) -> CookingEvents {
        if self.state == CookingState::Inactive {
            self.clear_runtime_state();
            return CookingEvents::new();
        }
        self.clear_runtime_state();
        self.state = CookingState::Inactive;
        vec![CookingEvent::PageLeft]
    }

    fn clear_runtime_state(&mut self) {
        self.stable_frames = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum CookingInputError {
    #[error("熟度进度必须为有限数，实际为 {0}")]
    NonFiniteProgress(f64),
    #[error("熟度进度必须位于 0..=1，实际为 {0}")]
    ProgressOutOfRange(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> CookingStateMachine {
        CookingStateMachine::new(CookingConfig::new(0.80, 0.90, 3).unwrap())
    }

    fn visible(progress: f64) -> CookingFrame {
        CookingFrame::new(progress, true, false)
    }

    fn completion_events(events: &CookingEvents) -> Vec<f64> {
        events
            .iter()
            .filter_map(|event| match event {
                CookingEvent::CompletionTriggered { progress } => Some(*progress),
                _ => None,
            })
            .collect()
    }

    fn assert_no_completion(events: CookingEvents) {
        assert!(
            completion_events(&events).is_empty(),
            "unexpected events: {events:?}"
        );
    }

    #[test]
    fn peak_below_optimal_range_never_completes() {
        let mut machine = machine();
        for _ in 0..10 {
            assert_no_completion(machine.on_frame(visible(0.79)).unwrap());
        }
        assert_eq!(machine.state(), CookingState::Observing);
        assert_eq!(machine.stable_frames(), 0);
    }

    #[test]
    fn completion_occurs_on_exactly_the_third_stable_frame() {
        let mut machine = machine();
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_eq!(machine.stable_frames(), 1);
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_eq!(machine.stable_frames(), 2);
        let events = machine.on_frame(visible(0.85)).unwrap();
        assert_eq!(
            events,
            vec![CookingEvent::CompletionTriggered { progress: 0.85 }]
        );
        assert_eq!(machine.state(), CookingState::Completed);
        assert_eq!(machine.stable_frames(), 0);
    }

    #[test]
    fn repeated_frames_after_completion_do_not_trigger_again() {
        let mut machine = machine();
        for _ in 0..3 {
            let _ = machine.on_frame(visible(0.85)).unwrap();
        }
        for _ in 0..10 {
            assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        }
        assert_eq!(machine.state(), CookingState::Completed);
        assert_eq!(machine.stable_frames(), 0);
    }

    #[test]
    fn progress_leaving_range_resets_consecutive_streak() {
        let mut machine = machine();
        machine.on_frame(visible(0.85)).unwrap();
        machine.on_frame(visible(0.86)).unwrap();
        assert_eq!(
            machine.on_frame(visible(0.95)).unwrap(),
            vec![CookingEvent::StabilityLost]
        );
        assert_eq!(machine.stable_frames(), 0);
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_eq!(
            completion_events(&machine.on_frame(visible(0.85)).unwrap()),
            [0.85]
        );
    }

    #[test]
    fn leaving_page_clears_an_incomplete_streak() {
        let mut machine = machine();
        machine.on_frame(visible(0.85)).unwrap();
        machine.on_frame(visible(0.85)).unwrap();
        assert_eq!(
            machine
                .on_frame(CookingFrame::new(0.85, false, false))
                .unwrap(),
            vec![CookingEvent::PageLeft]
        );
        assert_eq!(machine.state(), CookingState::Inactive);
        assert_eq!(machine.stable_frames(), 0);
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_eq!(machine.stable_frames(), 2);
    }

    #[test]
    fn leaving_page_clears_completion_latch_and_reentry_needs_new_streak() {
        let mut machine = machine();
        for _ in 0..3 {
            let _ = machine.on_frame(visible(0.85)).unwrap();
        }
        assert_eq!(
            machine
                .on_frame(CookingFrame::new(0.85, false, false))
                .unwrap(),
            vec![CookingEvent::PageLeft]
        );
        let events = machine.on_frame(visible(0.85)).unwrap();
        assert_eq!(events, vec![CookingEvent::PageEntered]);
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_eq!(
            completion_events(&machine.on_frame(visible(0.85)).unwrap()),
            [0.85]
        );
    }

    #[test]
    fn invalid_progress_is_rejected_without_mutating_state() {
        let mut machine = machine();
        machine.on_frame(visible(0.85)).unwrap();
        for progress in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            let result = machine.on_frame(visible(progress));
            assert!(result.is_err(), "accepted invalid progress {progress}");
            assert_eq!(machine.state(), CookingState::Accumulating);
            assert_eq!(machine.stable_frames(), 1);
        }
    }

    #[test]
    fn hidden_and_cancelled_frames_still_validate_progress() {
        let mut machine = machine();
        assert!(
            machine
                .on_frame(CookingFrame::new(f64::NAN, false, false))
                .is_err()
        );
        assert!(
            machine
                .on_frame(CookingFrame::new(2.0, true, true))
                .is_err()
        );
        assert_eq!(machine.state(), CookingState::Inactive);
    }

    #[test]
    fn cancellation_discards_streak_and_is_terminal_until_reset() {
        let mut machine = machine();
        machine.on_frame(visible(0.85)).unwrap();
        machine.on_frame(visible(0.85)).unwrap();
        assert_eq!(
            machine
                .on_frame(CookingFrame::new(0.85, true, true))
                .unwrap(),
            vec![CookingEvent::Cancelled]
        );
        assert_eq!(machine.state(), CookingState::Cancelled);
        assert_eq!(machine.stable_frames(), 0);
        for _ in 0..5 {
            assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        }
        assert_eq!(machine.reset(), CookingEvent::Reset);
        assert_eq!(machine.state(), CookingState::Inactive);
        assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        assert_eq!(machine.stable_frames(), 1);
    }

    #[test]
    fn cancellation_cannot_race_a_completion_on_the_same_frame() {
        let mut machine = machine();
        machine.on_frame(visible(0.85)).unwrap();
        machine.on_frame(visible(0.85)).unwrap();
        assert_eq!(
            machine
                .on_frame(CookingFrame::new(0.85, true, true))
                .unwrap(),
            vec![CookingEvent::Cancelled]
        );
        assert_eq!(machine.state(), CookingState::Cancelled);
    }

    #[test]
    fn explicit_reset_clears_completion_latch() {
        let mut machine = machine();
        for _ in 0..3 {
            let _ = machine.on_frame(visible(0.85)).unwrap();
        }
        assert_eq!(machine.reset(), CookingEvent::Reset);
        assert_eq!(machine.state(), CookingState::Inactive);
        for _ in 0..2 {
            assert_no_completion(machine.on_frame(visible(0.85)).unwrap());
        }
        assert_eq!(
            completion_events(&machine.on_frame(visible(0.85)).unwrap()),
            [0.85]
        );
    }

    #[test]
    fn optimal_range_bounds_are_inclusive_and_above_peak_does_not_complete() {
        let mut machine = machine();
        machine.on_frame(visible(0.80)).unwrap();
        machine.on_frame(visible(0.90)).unwrap();
        assert_eq!(
            completion_events(&machine.on_frame(visible(0.85)).unwrap()),
            [0.85]
        );

        machine.reset();
        for _ in 0..5 {
            assert_no_completion(machine.on_frame(visible(1.0)).unwrap());
        }
        assert_eq!(machine.state(), CookingState::Observing);
    }

    #[test]
    fn progress_zero_and_one_are_valid_inputs() {
        let mut machine = machine();
        assert!(machine.on_frame(visible(0.0)).is_ok());
        assert!(machine.on_frame(visible(1.0)).is_ok());
    }

    #[test]
    fn normalized_progress_and_configuration_are_resolution_independent() {
        let config = CookingConfig::new(0.80, 0.90, 3).unwrap();
        let run = |width, height| {
            assert_eq!(width * 1080, height * 1920);
            let mut machine = CookingStateMachine::new(config.clone());
            // Example adapter-side normalization of the 1080p reference bar.
            let scale = height as f64 / 1080.0;
            let bar_left = 600.0 * scale;
            let bar_width = 730.0 * scale;
            let normalized_samples = [0.70, 0.84, 0.85, 0.86];
            let mut completions = 0;
            for progress in normalized_samples {
                let pixel_x = bar_left + progress * bar_width;
                let adapter_progress = (pixel_x - bar_left) / bar_width;
                completions += usize::from(
                    !completion_events(&machine.on_frame(visible(adapter_progress)).unwrap())
                        .is_empty(),
                );
            }
            (completions, machine.state())
        };
        assert_eq!(run(1920, 1080), run(2560, 1440));
        assert_eq!(run(1920, 1080), (1, CookingState::Completed));
    }

    #[test]
    fn task_configuration_defaults_and_rejects_unbounded_inputs() {
        let config = AutoCookTaskConfig::default();
        assert_eq!(config.check_interval().unwrap(), Duration::from_millis(10));
        assert!(config.stop_on_recover);
        let started = std::time::Instant::now();
        assert!(!config.timed_out(started, started + Duration::from_millis(119_999)));
        assert!(config.timed_out(started, started + Duration::from_millis(120_000)));
        for invalid in [
            AutoCookTaskConfig {
                check_interval_ms: 0,
                ..config.clone()
            },
            AutoCookTaskConfig {
                check_interval_ms: 10_001,
                ..config.clone()
            },
            AutoCookTaskConfig {
                total_timeout_ms: 0,
                ..config.clone()
            },
            AutoCookTaskConfig {
                total_timeout_ms: 3_600_001,
                ..config
            },
        ] {
            assert!(invalid.validate().is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        for result in [
            CookingConfig::new(f64::NAN, 0.9, 3),
            CookingConfig::new(0.8, f64::INFINITY, 3),
            CookingConfig::new(-0.1, 0.9, 3),
            CookingConfig::new(0.8, 1.1, 3),
            CookingConfig::new(0.9, 0.8, 3),
            CookingConfig::new(0.8, 0.9, 0),
        ] {
            assert!(result.is_err());
        }
        assert!(CookingConfig::new(0.0, 1.0, 1).is_ok());
    }
}
