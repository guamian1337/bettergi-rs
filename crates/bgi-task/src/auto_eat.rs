use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoEatAction {
    Recover,
    Revive,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AutoEatObservation {
    pub low_hp: bool,
    pub recovery_icon: bool,
    pub resurrection_icon: bool,
}

#[derive(Debug, Default)]
pub struct AutoEatController {
    last_recovery_check: Option<Instant>,
    recovery_detected: bool,
    last_recover: Option<Instant>,
    last_revive: Option<Instant>,
}

impl AutoEatController {
    pub fn observe(
        &mut self,
        observation: AutoEatObservation,
        gameplay: bool,
        foreground: bool,
        now: Instant,
    ) -> Option<AutoEatAction> {
        if !gameplay || !foreground {
            return None;
        }
        if observation.low_hp {
            if self
                .last_recovery_check
                .is_none_or(|last| now.saturating_duration_since(last) >= Duration::from_secs(30))
            {
                self.recovery_detected = false;
                self.last_recovery_check = Some(now);
            }
            if observation.recovery_icon {
                self.recovery_detected = true;
                self.last_recovery_check = Some(now);
            }
            if self.recovery_detected
                && self.last_recover.is_none_or(|last| {
                    now.saturating_duration_since(last) >= Duration::from_secs(1)
                })
            {
                self.last_recover = Some(now);
                return Some(AutoEatAction::Recover);
            }
        }
        if observation.resurrection_icon
            && self
                .last_revive
                .is_none_or(|last| now.saturating_duration_since(last) >= Duration::from_secs(2))
        {
            self.last_revive = Some(now);
            return Some(AutoEatAction::Revive);
        }
        None
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_low_hp_and_recovery_icon_before_first_use() {
        let now = Instant::now();
        let mut controller = AutoEatController::default();
        assert_eq!(
            controller.observe(
                AutoEatObservation {
                    low_hp: true,
                    ..Default::default()
                },
                true,
                true,
                now,
            ),
            None
        );
        assert_eq!(
            controller.observe(
                AutoEatObservation {
                    low_hp: true,
                    recovery_icon: true,
                    ..Default::default()
                },
                true,
                true,
                now,
            ),
            Some(AutoEatAction::Recover)
        );
        assert_eq!(
            controller.observe(
                AutoEatObservation {
                    low_hp: true,
                    ..Default::default()
                },
                true,
                true,
                now + Duration::from_secs(1),
            ),
            Some(AutoEatAction::Recover)
        );
    }

    #[test]
    fn recovery_cache_expires_and_revive_has_two_second_cooldown() {
        let now = Instant::now();
        let mut controller = AutoEatController::default();
        controller.observe(
            AutoEatObservation {
                low_hp: true,
                recovery_icon: true,
                ..Default::default()
            },
            true,
            true,
            now,
        );
        assert_eq!(
            controller.observe(
                AutoEatObservation {
                    low_hp: true,
                    ..Default::default()
                },
                true,
                true,
                now + Duration::from_secs(30),
            ),
            None
        );
        let revive = AutoEatObservation {
            resurrection_icon: true,
            ..Default::default()
        };
        assert_eq!(
            controller.observe(revive, true, true, now),
            Some(AutoEatAction::Revive)
        );
        assert_eq!(
            controller.observe(revive, true, true, now + Duration::from_secs(1)),
            None
        );
        assert_eq!(
            controller.observe(revive, true, true, now + Duration::from_secs(2)),
            Some(AutoEatAction::Revive)
        );
    }

    #[test]
    fn never_acts_in_menus_or_background() {
        let now = Instant::now();
        let mut controller = AutoEatController::default();
        let observation = AutoEatObservation {
            low_hp: true,
            recovery_icon: true,
            resurrection_icon: true,
        };
        assert_eq!(controller.observe(observation, false, true, now), None);
        assert_eq!(controller.observe(observation, true, false, now), None);
    }
}
