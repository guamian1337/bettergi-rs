pub const EXPEDITION_REQUIRED_STABLE_SAMPLES: usize = 3;
pub const EXPEDITION_MAX_ATTEMPTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpeditionPageState {
    CollectAvailable,
    RedispatchAvailable,
}

impl ExpeditionPageState {
    pub fn label(self) -> &'static str {
        match self {
            Self::CollectAvailable => "检测到全部领取图标",
            Self::RedispatchAvailable => "检测到再次派遣图标",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpeditionRecognitionState {
    Pending,
    Ready(ExpeditionPageState),
    Failed,
}

#[derive(Debug, Clone)]
pub struct ExpeditionRecognitionStability {
    required: usize,
    max_attempts: usize,
    attempts: usize,
    consecutive: usize,
    candidate: Option<ExpeditionPageState>,
    state: ExpeditionRecognitionState,
}

impl Default for ExpeditionRecognitionStability {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpeditionRecognitionStability {
    pub fn new() -> Self {
        Self {
            required: EXPEDITION_REQUIRED_STABLE_SAMPLES,
            max_attempts: EXPEDITION_MAX_ATTEMPTS,
            attempts: 0,
            consecutive: 0,
            candidate: None,
            state: ExpeditionRecognitionState::Pending,
        }
    }

    pub fn observe(&mut self, sample: Option<ExpeditionPageState>) -> ExpeditionRecognitionState {
        if self.state != ExpeditionRecognitionState::Pending {
            return self.state.clone();
        }
        self.attempts += 1;
        match sample {
            Some(sample) if self.candidate == Some(sample) => self.consecutive += 1,
            Some(sample) => {
                self.candidate = Some(sample);
                self.consecutive = 1;
            }
            None => {
                self.candidate = None;
                self.consecutive = 0;
            }
        }
        if self.consecutive >= self.required {
            self.state = ExpeditionRecognitionState::Ready(self.candidate.unwrap());
        } else if self.attempts >= self.max_attempts {
            self.state = ExpeditionRecognitionState::Failed;
        }
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stability_requires_three_identical_unambiguous_states() {
        let mut stability = ExpeditionRecognitionStability::new();
        let state = ExpeditionPageState::CollectAvailable;
        assert_eq!(
            stability.observe(Some(state)),
            ExpeditionRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(state)),
            ExpeditionRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(state)),
            ExpeditionRecognitionState::Ready(state)
        );
        assert_eq!(
            stability.observe(Some(ExpeditionPageState::RedispatchAvailable)),
            ExpeditionRecognitionState::Ready(state)
        );
    }

    #[test]
    fn missing_or_changed_state_resets_consecutive_samples() {
        let mut stability = ExpeditionRecognitionStability::new();
        let collect = ExpeditionPageState::CollectAvailable;
        let redispatch = ExpeditionPageState::RedispatchAvailable;
        stability.observe(Some(collect));
        stability.observe(Some(collect));
        assert_eq!(stability.observe(None), ExpeditionRecognitionState::Pending);
        assert_eq!(
            stability.observe(Some(collect)),
            ExpeditionRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(redispatch)),
            ExpeditionRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(redispatch)),
            ExpeditionRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(Some(redispatch)),
            ExpeditionRecognitionState::Ready(redispatch)
        );
    }

    #[test]
    fn no_detection_fails_after_ten_attempts_without_inventing_an_empty_page() {
        let mut stability = ExpeditionRecognitionStability::new();
        for _ in 0..EXPEDITION_MAX_ATTEMPTS - 1 {
            assert_eq!(stability.observe(None), ExpeditionRecognitionState::Pending);
        }
        assert_eq!(stability.observe(None), ExpeditionRecognitionState::Failed);
    }
}
