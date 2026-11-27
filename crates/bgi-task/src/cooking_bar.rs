#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CookingBarEvent {
    PeakDetected { color_count: u32 },
    CompletionTriggered { peak: u32, current: u32 },
}

#[derive(Debug, Clone)]
pub struct CookingBarConfig {
    pub minimum_count: f64,
    pub tolerance: f64,
    pub stable_frames: u32,
    pub drop_count: f64,
}

impl Default for CookingBarConfig {
    fn default() -> Self {
        Self {
            minimum_count: 600.0,
            tolerance: 20.0,
            stable_frames: 3,
            drop_count: 300.0,
        }
    }
}

impl CookingBarConfig {
    pub fn scaled(&self, scale: f64) -> Result<Self, &'static str> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("烹饪识别缩放必须为正的有限数");
        }
        if !self.minimum_count.is_finite()
            || !self.tolerance.is_finite()
            || !self.drop_count.is_finite()
            || self.minimum_count < 0.0
            || self.tolerance < 0.0
            || self.drop_count < 0.0
            || self.stable_frames == 0
        {
            return Err("烹饪识别配置无效");
        }
        Ok(Self {
            minimum_count: self.minimum_count * scale,
            // C# scales only `peakMinCount` and `triggerDropCount` (`AutoCookTask.cs:34-35`)
            // and compares the tolerance unscaled (`:129`), so this must not follow `scale`.
            tolerance: self.tolerance,
            stable_frames: self.stable_frames,
            drop_count: self.drop_count * scale,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct CookingBarDetector {
    candidate: Option<u32>,
    stable_frames: u32,
    peak: Option<u32>,
}

impl CookingBarDetector {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn observe(
        &mut self,
        color_count: u32,
        page_visible: bool,
        config: &CookingBarConfig,
    ) -> CookingBarEvent {
        if !page_visible {
            self.reset();
            return CookingBarEvent::PeakDetected { color_count };
        }
        let current = f64::from(color_count);
        if let Some(peak) = self.peak
            && current <= f64::from(peak) - config.drop_count
        {
            self.reset();
            return CookingBarEvent::CompletionTriggered {
                peak,
                current: color_count,
            };
        }
        if self.peak.is_some() {
            return CookingBarEvent::PeakDetected { color_count };
        }
        if current <= config.minimum_count {
            self.candidate = None;
            self.stable_frames = 0;
            return CookingBarEvent::PeakDetected { color_count };
        }
        match self.candidate {
            Some(candidate) if (current - f64::from(candidate)).abs() <= config.tolerance => {
                self.candidate = Some(candidate.max(color_count));
                self.stable_frames = self.stable_frames.saturating_add(1);
            }
            _ => {
                self.candidate = Some(color_count);
                self.stable_frames = 1;
            }
        }
        if self.stable_frames >= config.stable_frames
            && let Some(candidate) = self.candidate
            && f64::from(candidate) > config.minimum_count
        {
            self.candidate = None;
            self.stable_frames = 0;
            self.peak = Some(candidate);
        }
        CookingBarEvent::PeakDetected { color_count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_peak_followed_by_drop_triggers_once() {
        let mut detector = CookingBarDetector::default();
        let config = CookingBarConfig::default();
        for _ in 0..3 {
            assert_eq!(
                detector.observe(1000, true, &config),
                CookingBarEvent::PeakDetected { color_count: 1000 }
            );
        }
        assert_eq!(
            detector.observe(699, true, &config),
            CookingBarEvent::CompletionTriggered {
                peak: 1000,
                current: 699
            }
        );
        assert_eq!(
            detector.observe(0, true, &config),
            CookingBarEvent::PeakDetected { color_count: 0 }
        );
    }

    #[test]
    fn page_loss_and_small_counts_reset_candidate() {
        let config = CookingBarConfig::default();
        let mut detector = CookingBarDetector::default();
        detector.observe(1000, true, &config);
        detector.observe(1000, false, &config);
        detector.observe(1000, true, &config);
        assert_eq!(detector.stable_frames, 1);
        detector.observe(500, true, &config);
        assert_eq!(detector.stable_frames, 0);
    }

    #[test]
    fn configuration_scaling_preserves_reference_behavior() {
        let base = CookingBarConfig::default().scaled(2.0).unwrap();
        assert_eq!(base.minimum_count, 1200.0);
        assert_eq!(base.drop_count, 600.0);
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(CookingBarConfig::default().scaled(scale).is_err());
        }
    }
}
