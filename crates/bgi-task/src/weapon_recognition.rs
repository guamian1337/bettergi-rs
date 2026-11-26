pub const WEAPON_REQUIRED_STABLE_SAMPLES: usize = 3;
pub const WEAPON_MAX_OCR_ATTEMPTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeaponPixelRoi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeaponRois {
    pub name: WeaponPixelRoi,
    pub level: WeaponPixelRoi,
}

pub fn weapon_rois(width: u32, height: u32) -> Option<WeaponRois> {
    if width == 0 || height == 0 || u64::from(width) * 1080 != u64::from(height) * 1920 {
        return None;
    }
    let scale = width as f64 / 1920.0;
    let rect = |x: u32, y: u32, roi_width: u32, roi_height: u32| -> Option<WeaponPixelRoi> {
        let x = (x as f64 * scale).round() as u32;
        let y = (y as f64 * scale).round() as u32;
        let roi_width = (roi_width as f64 * scale).round() as u32;
        let roi_height = (roi_height as f64 * scale).round() as u32;
        (roi_width > 0 && roi_height > 0 && x + roi_width <= width && y + roi_height <= height)
            .then_some(WeaponPixelRoi {
                x,
                y,
                width: roi_width,
                height: roi_height,
            })
    };
    Some(WeaponRois {
        name: rect(1465, 132, 346, 42)?,
        level: rect(1464, 319, 147, 73)?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeaponLevelPair {
    pub level: u8,
    pub limit: u8,
}

pub fn parse_weapon_level_pair(text: &str) -> Option<WeaponLevelPair> {
    let mut values = Vec::with_capacity(2);
    let mut digits = String::new();
    for ch in text.chars().chain(['\n']) {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            values.push(digits.parse::<u8>().ok()?);
            digits.clear();
            if values.len() > 2 {
                return None;
            }
        }
    }
    if values.len() != 2 {
        return None;
    }
    let [level, limit] = values.try_into().ok()?;
    (level > 0 && level <= 90 && limit >= level && limit <= 90)
        .then_some(WeaponLevelPair { level, limit })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeaponDetails {
    pub name: String,
    pub level: u8,
    pub level_limit: u8,
}

impl WeaponDetails {
    pub fn try_new(name: String, level: u8, level_limit: u8) -> Result<Self, &'static str> {
        if name.trim().is_empty() {
            return Err("武器名称不能为空");
        }
        if level == 0 || level > 90 || level_limit < level || level_limit > 90 {
            return Err("武器等级必须满足 1 <= 当前等级 <= 等级上限 <= 90");
        }
        Ok(Self {
            name,
            level,
            level_limit,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeaponRecognitionState {
    Pending,
    Ready(WeaponDetails),
    Failed,
}

#[derive(Debug, Clone)]
pub struct WeaponRecognitionStability {
    required: usize,
    max_attempts: usize,
    attempts: usize,
    consecutive: usize,
    candidate: Option<WeaponDetails>,
    state: WeaponRecognitionState,
}

impl Default for WeaponRecognitionStability {
    fn default() -> Self {
        Self::new(WEAPON_REQUIRED_STABLE_SAMPLES, WEAPON_MAX_OCR_ATTEMPTS)
    }
}

impl WeaponRecognitionStability {
    pub fn new(required: usize, max_attempts: usize) -> Self {
        Self {
            required,
            max_attempts,
            attempts: 0,
            consecutive: 0,
            candidate: None,
            state: WeaponRecognitionState::Pending,
        }
    }

    pub fn observe(&mut self, sample: Option<WeaponDetails>) -> WeaponRecognitionState {
        if self.state != WeaponRecognitionState::Pending {
            return self.state.clone();
        }
        self.attempts += 1;
        match sample {
            Some(sample) if self.candidate.as_ref() == Some(&sample) => {
                self.consecutive += 1;
            }
            Some(sample) => {
                self.candidate = Some(sample);
                self.consecutive = 1;
            }
            None => {
                self.candidate = None;
                self.consecutive = 0;
            }
        }
        if self.consecutive >= self.required
            && let Some(candidate) = self.candidate.clone()
        {
            self.state = WeaponRecognitionState::Ready(candidate);
        } else if self.attempts >= self.max_attempts {
            self.state = WeaponRecognitionState::Failed;
        }
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn details(name: &str, level: u8, limit: u8) -> Option<WeaponDetails> {
        WeaponDetails::try_new(name.into(), level, limit).ok()
    }

    #[test]
    fn reference_rois_scale_exactly_and_other_aspect_ratios_fail_closed() {
        let full = weapon_rois(1920, 1080).unwrap();
        assert_eq!(
            full.name,
            WeaponPixelRoi {
                x: 1465,
                y: 132,
                width: 346,
                height: 42
            }
        );
        assert_eq!(
            full.level,
            WeaponPixelRoi {
                x: 1464,
                y: 319,
                width: 147,
                height: 73
            }
        );
        let half = weapon_rois(1280, 720).unwrap();
        assert_eq!(
            half.name,
            WeaponPixelRoi {
                x: 977,
                y: 88,
                width: 231,
                height: 28
            }
        );
        assert_eq!(
            half.level,
            WeaponPixelRoi {
                x: 976,
                y: 213,
                width: 98,
                height: 49
            }
        );
        let large = weapon_rois(2560, 1440).unwrap();
        assert_eq!(large.name.x, 1953);
        assert_eq!(large.name.width, 461);
        for size in [(0, 1080), (1920, 0), (1919, 1080), (1920, 1079), (16, 9)] {
            assert!(weapon_rois(size.0, size.1).is_none());
        }
    }

    #[test]
    fn level_parser_accepts_only_two_valid_ordered_levels() {
        assert_eq!(
            parse_weapon_level_pair("Lv.20\n90"),
            Some(WeaponLevelPair {
                level: 20,
                limit: 90
            })
        );
        assert_eq!(
            parse_weapon_level_pair("1/1"),
            Some(WeaponLevelPair { level: 1, limit: 1 })
        );
        assert_eq!(
            parse_weapon_level_pair("90/90"),
            Some(WeaponLevelPair {
                level: 90,
                limit: 90
            })
        );
        for text in [
            "",
            "20",
            "0/90",
            "91/90",
            "20/91",
            "90/20",
            "20/90/80",
            "二十/九十",
        ] {
            assert!(parse_weapon_level_pair(text).is_none(), "{text}");
        }
    }

    #[test]
    fn stability_requires_three_consecutive_identical_complete_samples() {
        let mut stable = WeaponRecognitionStability::default();
        let expected = details("西风剑", 20, 90).unwrap();
        assert_eq!(
            stable.observe(details("西风剑", 20, 90)),
            WeaponRecognitionState::Pending
        );
        assert_eq!(stable.observe(None), WeaponRecognitionState::Pending);
        assert_eq!(
            stable.observe(details("西风剑", 20, 90)),
            WeaponRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(details("西风剑", 21, 90)),
            WeaponRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected.clone())),
            WeaponRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected.clone())),
            WeaponRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected.clone())),
            WeaponRecognitionState::Ready(expected.clone())
        );
        assert_eq!(
            stable.observe(None),
            WeaponRecognitionState::Ready(expected)
        );
    }

    #[test]
    fn stability_fails_after_ten_attempts_without_inventing_partial_results() {
        let mut stable = WeaponRecognitionStability::default();
        for _ in 0..9 {
            assert_eq!(stable.observe(None), WeaponRecognitionState::Pending);
        }
        assert_eq!(stable.observe(None), WeaponRecognitionState::Failed);
        assert_eq!(
            stable.observe(details("西风剑", 1, 90)),
            WeaponRecognitionState::Failed
        );
        assert!(WeaponDetails::try_new("西风剑".into(), 0, 90).is_err());
        assert!(WeaponDetails::try_new(" ".into(), 1, 90).is_err());
    }
}
