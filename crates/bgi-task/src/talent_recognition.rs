pub const TALENT_REQUIRED_STABLE_SAMPLES: usize = 3;
pub const TALENT_MAX_OCR_ATTEMPTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TalentType {
    Attack,
    Skill,
    Burst,
}

impl TalentType {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Attack => "普通攻击",
            Self::Skill => "元素战技",
            Self::Burst => "元素爆发",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TalentPixelRoi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TalentRois {
    pub page_gate: TalentPixelRoi,
    pub talent_type: TalentPixelRoi,
    pub level: TalentPixelRoi,
    pub bonus: TalentPixelRoi,
}

pub fn talent_rois(width: u32, height: u32) -> Option<TalentRois> {
    if width == 0 || height == 0 || u64::from(width) * 1080 != u64::from(height) * 1920 {
        return None;
    }
    let scale = width as f64 / 1920.0;
    let rect = |x: u32, y: u32, roi_width: u32, roi_height: u32| -> Option<TalentPixelRoi> {
        let x = (x as f64 * scale).round_ties_even() as u32;
        let y = (y as f64 * scale).round_ties_even() as u32;
        let roi_width = (roi_width as f64 * scale).round_ties_even() as u32;
        let roi_height = (roi_height as f64 * scale).round_ties_even() as u32;
        (roi_width > 0 && roi_height > 0 && x + roi_width <= width && y + roi_height <= height)
            .then_some(TalentPixelRoi {
                x,
                y,
                width: roi_width,
                height: roi_height,
            })
    };
    Some(TalentRois {
        page_gate: rect(100, 219, 125, 33)?,
        talent_type: rect(242, 13, 98, 36)?,
        level: rect(256, 168, 66, 25)?,
        bonus: rect(35, 285, 146, 30)?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TalentDetails {
    pub talent_type: TalentType,
    pub level: u32,
    pub has_bonus: bool,
}

impl TalentDetails {
    pub fn try_new(
        talent_type: TalentType,
        level: u32,
        has_bonus: bool,
    ) -> Result<Self, &'static str> {
        if level == 0 {
            return Err("天赋显示等级必须大于 0");
        }
        Ok(Self {
            talent_type,
            level,
            has_bonus,
        })
    }
}

pub fn is_talent_detail_text(text: &str) -> bool {
    text.contains("天赋介绍")
}

pub fn parse_talent_type(text: &str) -> Option<TalentType> {
    let mut matches = [
        (TalentType::Attack, text.contains("普通攻击")),
        (TalentType::Skill, text.contains("元素战技")),
        (TalentType::Burst, text.contains("元素爆发")),
    ]
    .into_iter()
    .filter_map(|(talent_type, matched)| matched.then_some(talent_type));
    let talent_type = matches.next()?;
    matches.next().is_none().then_some(talent_type)
}

pub fn parse_talent_level(text: &str) -> Option<u32> {
    let start = text
        .char_indices()
        .find_map(|(index, ch)| ch.is_ascii_digit().then_some(index))?;
    let digits = &text[start..];
    let end = digits
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_ascii_digit()).then_some(index))
        .unwrap_or(digits.len());
    digits[..end].parse::<u32>().ok().filter(|level| *level > 0)
}

pub fn parse_talent_bonus(text: &str) -> Option<bool> {
    let compact: Vec<char> = text.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact.is_empty() {
        return Some(false);
    }
    let marker = ['天', '赋', '等', '级', '+', '3'];
    let full_width_marker = ['天', '赋', '等', '级', '＋', '3'];
    let has_bonus = compact
        .windows(marker.len())
        .any(|window| window == marker || window == full_width_marker);
    has_bonus.then_some(true)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TalentRecognitionState {
    Pending,
    Ready(TalentDetails),
    Failed,
}

#[derive(Debug, Clone)]
pub struct TalentRecognitionStability {
    required: usize,
    max_attempts: usize,
    attempts: usize,
    consecutive: usize,
    candidate: Option<TalentDetails>,
    state: TalentRecognitionState,
}

impl Default for TalentRecognitionStability {
    fn default() -> Self {
        Self::new(TALENT_REQUIRED_STABLE_SAMPLES, TALENT_MAX_OCR_ATTEMPTS)
    }
}

impl TalentRecognitionStability {
    pub fn new(required: usize, max_attempts: usize) -> Self {
        Self {
            required,
            max_attempts,
            attempts: 0,
            consecutive: 0,
            candidate: None,
            state: TalentRecognitionState::Pending,
        }
    }

    pub fn observe(&mut self, sample: Option<TalentDetails>) -> TalentRecognitionState {
        if self.state != TalentRecognitionState::Pending {
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
            self.state = TalentRecognitionState::Ready(candidate);
        } else if self.attempts >= self.max_attempts {
            self.state = TalentRecognitionState::Failed;
        }
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn details(talent_type: TalentType, level: u32, has_bonus: bool) -> Option<TalentDetails> {
        TalentDetails::try_new(talent_type, level, has_bonus).ok()
    }

    #[test]
    fn reference_rois_match_csharp_rounding_and_reject_invalid_frames() {
        let full = talent_rois(1920, 1080).unwrap();
        assert_eq!(full.page_gate.x, 100);
        assert_eq!(full.page_gate.y, 219);
        assert_eq!(full.talent_type.x, 242);
        assert_eq!(full.talent_type.y, 13);
        assert_eq!(full.level.x, 256);
        assert_eq!(full.level.y, 168);
        assert_eq!(full.bonus.x, 35);
        assert_eq!(full.bonus.y, 285);

        let half = talent_rois(1280, 720).unwrap();
        assert_eq!(
            half.page_gate,
            TalentPixelRoi {
                x: 67,
                y: 146,
                width: 83,
                height: 22
            }
        );
        assert_eq!(
            half.talent_type,
            TalentPixelRoi {
                x: 161,
                y: 9,
                width: 65,
                height: 24
            }
        );
        assert_eq!(
            half.level,
            TalentPixelRoi {
                x: 171,
                y: 112,
                width: 44,
                height: 17
            }
        );
        assert_eq!(
            half.bonus,
            TalentPixelRoi {
                x: 23,
                y: 190,
                width: 97,
                height: 20
            }
        );
        for size in [(0, 1080), (1920, 0), (1919, 1080), (1920, 1079), (16, 9)] {
            assert!(talent_rois(size.0, size.1).is_none());
        }
    }

    #[test]
    fn parsers_require_known_page_type_level_and_bonus_markers() {
        assert!(is_talent_detail_text("天赋介绍"));
        assert!(!is_talent_detail_text("普通攻击"));
        assert_eq!(parse_talent_type("普通攻击"), Some(TalentType::Attack));
        assert_eq!(parse_talent_type("元素战技"), Some(TalentType::Skill));
        assert_eq!(parse_talent_type("元素爆发"), Some(TalentType::Burst));
        for text in ["", "普通攻击 元素战技", "治疗"] {
            assert_eq!(parse_talent_type(text), None, "{text}");
        }
        for (text, expected) in [("Lv.10", 10), ("等级 15", 15), ("1/20", 1)] {
            assert_eq!(parse_talent_level(text), Some(expected));
        }
        for text in ["", "Lv.", "0", "999999999999999999999"] {
            assert_eq!(parse_talent_level(text), None, "{text}");
        }
        assert_eq!(parse_talent_bonus(" \n"), Some(false));
        assert_eq!(parse_talent_bonus("天赋等级+3"), Some(true));
        assert_eq!(parse_talent_bonus("天赋 等级 ＋ 3"), Some(true));
        assert_eq!(parse_talent_bonus("命之座"), None);
        assert_eq!(parse_talent_bonus("天赋等级+2"), None);
        assert!(TalentDetails::try_new(TalentType::Attack, 0, false).is_err());
    }

    #[test]
    fn stability_requires_three_identical_complete_samples_and_resets_on_failure() {
        let mut stable = TalentRecognitionStability::default();
        let expected = details(TalentType::Skill, 10, true).unwrap();
        assert_eq!(
            stable.observe(details(TalentType::Skill, 10, true)),
            TalentRecognitionState::Pending
        );
        assert_eq!(stable.observe(None), TalentRecognitionState::Pending);
        assert_eq!(
            stable.observe(details(TalentType::Skill, 10, true)),
            TalentRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(details(TalentType::Skill, 10, false)),
            TalentRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected.clone())),
            TalentRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected.clone())),
            TalentRecognitionState::Pending
        );
        assert_eq!(
            stable.observe(Some(expected.clone())),
            TalentRecognitionState::Ready(expected.clone())
        );
        assert_eq!(
            stable.observe(None),
            TalentRecognitionState::Ready(expected)
        );
    }

    #[test]
    fn stability_fails_after_ten_attempts_without_partial_results() {
        let mut stable = TalentRecognitionStability::default();
        for _ in 0..9 {
            assert_eq!(stable.observe(None), TalentRecognitionState::Pending);
        }
        assert_eq!(stable.observe(None), TalentRecognitionState::Failed);
        assert_eq!(
            stable.observe(details(TalentType::Burst, 8, false)),
            TalentRecognitionState::Failed
        );
    }
}
