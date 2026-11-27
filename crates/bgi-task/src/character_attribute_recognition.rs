pub const CHARACTER_ATTRIBUTE_REQUIRED_STABLE_SAMPLES: usize = 3;
pub const CHARACTER_ATTRIBUTE_MAX_OCR_ATTEMPTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterAttributePixelRoi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn character_attribute_roi(width: u32, height: u32) -> Option<CharacterAttributePixelRoi> {
    if width == 0 || height == 0 || u64::from(width) * 1080 != u64::from(height) * 1920 {
        return None;
    }
    let scale = width as f64 / 1920.0;
    let x = (1467.0 * scale).round_ties_even() as u32;
    let y = (207.0 * scale).round_ties_even() as u32;
    let roi_width = (172.0 * scale).round_ties_even() as u32;
    let roi_height = (35.0 * scale).round_ties_even() as u32;
    (roi_width > 0 && roi_height > 0 && x + roi_width <= width && y + roi_height <= height)
        .then_some(CharacterAttributePixelRoi {
            x,
            y,
            width: roi_width,
            height: roi_height,
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CharacterAttributeDetails {
    pub level: u32,
    pub level_limit: u32,
}

impl CharacterAttributeDetails {
    pub fn try_new(level: u32, level_limit: u32) -> Result<Self, &'static str> {
        if level == 0 || level_limit < level {
            return Err("角色等级必须大于 0，且等级上限不得小于当前等级");
        }
        Ok(Self { level, level_limit })
    }
}

/// Takes the first two integers as `(level, level_limit)`.
///
/// Mirrors C# `CharacterDevelopmentTask.TryParseLevelPair`: `NumberRegex.Matches` keeps
/// every number and only requires `Count >= 2`, so trailing noise such as `90/90/1` still
/// yields `90/90`. Reliability comes from requiring three identical samples, not from
/// rejecting extra digits here.
pub fn parse_character_level_pair(text: &str) -> Option<CharacterAttributeDetails> {
    let mut values = Vec::with_capacity(2);
    let mut digits = String::new();
    for ch in text.chars().chain(['\n']) {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            values.push(digits.parse::<u32>().ok()?);
            digits.clear();
            if values.len() == 2 {
                break;
            }
        }
    }
    if values.len() != 2 {
        return None;
    }
    let [level, level_limit] = values.try_into().ok()?;
    CharacterAttributeDetails::try_new(level, level_limit).ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharacterAttributeRecognitionState {
    Pending,
    Ready(CharacterAttributeDetails),
    Failed,
}

#[derive(Debug, Clone)]
pub struct CharacterAttributeRecognitionStability {
    required: usize,
    max_attempts: usize,
    attempts: usize,
    consecutive: usize,
    candidate: Option<CharacterAttributeDetails>,
    state: CharacterAttributeRecognitionState,
}

impl Default for CharacterAttributeRecognitionStability {
    fn default() -> Self {
        Self::new(
            CHARACTER_ATTRIBUTE_REQUIRED_STABLE_SAMPLES,
            CHARACTER_ATTRIBUTE_MAX_OCR_ATTEMPTS,
        )
    }
}

impl CharacterAttributeRecognitionStability {
    pub fn new(required: usize, max_attempts: usize) -> Self {
        Self {
            required,
            max_attempts,
            attempts: 0,
            consecutive: 0,
            candidate: None,
            state: CharacterAttributeRecognitionState::Pending,
        }
    }

    pub fn observe(
        &mut self,
        sample: Option<CharacterAttributeDetails>,
    ) -> CharacterAttributeRecognitionState {
        if self.state != CharacterAttributeRecognitionState::Pending {
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
            && let Some(candidate) = self.candidate
        {
            self.state = CharacterAttributeRecognitionState::Ready(candidate);
        } else if self.attempts >= self.max_attempts {
            self.state = CharacterAttributeRecognitionState::Failed;
        }
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn details(level: u32, level_limit: u32) -> Option<CharacterAttributeDetails> {
        CharacterAttributeDetails::try_new(level, level_limit).ok()
    }

    #[test]
    fn reference_roi_scales_with_csharp_rounding_and_rejects_invalid_frames() {
        let full = character_attribute_roi(1920, 1080).unwrap();
        assert_eq!(
            (full.x, full.y, full.width, full.height),
            (1467, 207, 172, 35)
        );
        let half = character_attribute_roi(1280, 720).unwrap();
        assert_eq!(
            (half.x, half.y, half.width, half.height),
            (978, 138, 115, 23)
        );
        let large = character_attribute_roi(2560, 1440).unwrap();
        assert_eq!(
            (large.x, large.y, large.width, large.height),
            (1956, 276, 229, 47)
        );
        for size in [(0, 1080), (1920, 0), (1919, 1080), (1920, 1079), (16, 9)] {
            assert!(character_attribute_roi(size.0, size.1).is_none());
        }
    }

    #[test]
    fn level_pair_parser_takes_first_two_positive_ordered_values() {
        for (text, level, limit) in [
            ("90/90", 90, 90),
            ("Lv.80\n90", 80, 90),
            // Extra trailing numbers are ignored, matching C# `NumberRegex.Matches[0..2]`.
            ("90/90/90", 90, 90),
            ("90/90/1", 90, 90),
        ] {
            let parsed = parse_character_level_pair(text).unwrap();
            assert_eq!((parsed.level, parsed.level_limit), (level, limit));
        }
        for text in ["", "90", "0/90", "90/89", "999999999999999999/90"] {
            assert_eq!(parse_character_level_pair(text), None, "{text}");
        }
    }

    #[test]
    fn stability_requires_three_identical_complete_samples_and_resets_on_failure() {
        let mut stable = CharacterAttributeRecognitionStability::default();
        let expected = details(90, 90).unwrap();
        assert_eq!(
            stable.observe(details(90, 90)),
            CharacterAttributeRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(None),
            CharacterAttributeRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(details(90, 90)),
            CharacterAttributeRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(details(80, 90)),
            CharacterAttributeRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected)),
            CharacterAttributeRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected)),
            CharacterAttributeRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected)),
            CharacterAttributeRecognitionState::Ready(expected)
        );
        assert_eq!(
            stable.observe(None),
            CharacterAttributeRecognitionState::Ready(expected)
        );
    }

    #[test]
    fn stability_fails_after_ten_attempts_without_partial_results() {
        let mut stable = CharacterAttributeRecognitionStability::default();
        for _ in 0..9 {
            assert_eq!(
                stable.observe(None),
                CharacterAttributeRecognitionState::Pending
            );
        }
        assert_eq!(
            stable.observe(None),
            CharacterAttributeRecognitionState::Failed
        );
        assert_eq!(
            stable.observe(details(90, 90)),
            CharacterAttributeRecognitionState::Failed
        );
    }
}
