use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoSkipEvent {
    Advance,
    Waiting,
    NotInDialogue,
}

#[derive(Debug, Default)]
pub struct AutoSkipController {
    last_advance: Option<Instant>,
}

impl AutoSkipController {
    pub fn observe(&mut self, in_dialogue: bool, foreground: bool, now: Instant) -> AutoSkipEvent {
        if !in_dialogue || !foreground {
            return AutoSkipEvent::NotInDialogue;
        }
        if self
            .last_advance
            .is_some_and(|last| now.duration_since(last) < Duration::from_millis(200))
        {
            return AutoSkipEvent::Waiting;
        }
        self.last_advance = Some(now);
        AutoSkipEvent::Advance
    }

    pub fn reset(&mut self) {
        self.last_advance = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_dialogue_at_most_once_per_two_hundred_ms() {
        let now = Instant::now();
        let mut controller = AutoSkipController::default();
        assert_eq!(controller.observe(true, true, now), AutoSkipEvent::Advance);
        assert_eq!(
            controller.observe(true, true, now + Duration::from_millis(199)),
            AutoSkipEvent::Waiting
        );
        assert_eq!(
            controller.observe(true, true, now + Duration::from_millis(200)),
            AutoSkipEvent::Advance
        );
    }

    #[test]
    fn ignores_non_dialogue_and_background_frames() {
        let now = Instant::now();
        let mut controller = AutoSkipController::default();
        assert_eq!(
            controller.observe(false, true, now),
            AutoSkipEvent::NotInDialogue
        );
        assert_eq!(
            controller.observe(true, false, now),
            AutoSkipEvent::NotInDialogue
        );
        controller.reset();
        assert_eq!(controller.observe(true, true, now), AutoSkipEvent::Advance);
    }
}
