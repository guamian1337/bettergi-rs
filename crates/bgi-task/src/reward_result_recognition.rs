use std::collections::BTreeMap;

pub const REWARD_RESULT_REQUIRED_STABLE_SAMPLES: usize = 3;
pub const REWARD_RESULT_MAX_ATTEMPTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewardCardRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewardIdentity {
    pub name: String,
    pub quality_level: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewardCardObservation {
    pub rect: RewardCardRect,
    pub identity: Option<RewardIdentity>,
    pub count: Option<u32>,
}

impl RewardCardObservation {
    pub fn reliable_identity(&self) -> bool {
        self.identity.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewardPageObservation(pub Vec<RewardCardObservation>);

impl RewardPageObservation {
    pub fn is_complete(&self) -> bool {
        !self.0.is_empty() && self.0.iter().all(RewardCardObservation::reliable_identity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewardResultRecognitionState {
    Pending,
    Ready(RewardPageObservation),
    Failed,
}

#[derive(Debug, Clone)]
pub struct RewardResultRecognitionStability {
    required: usize,
    max_attempts: usize,
    attempts: usize,
    consecutive: usize,
    candidate: Option<RewardPageObservation>,
    state: RewardResultRecognitionState,
}

impl Default for RewardResultRecognitionStability {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardResultRecognitionStability {
    pub fn new() -> Self {
        Self {
            required: REWARD_RESULT_REQUIRED_STABLE_SAMPLES,
            max_attempts: REWARD_RESULT_MAX_ATTEMPTS,
            attempts: 0,
            consecutive: 0,
            candidate: None,
            state: RewardResultRecognitionState::Pending,
        }
    }

    pub fn observe(&mut self, sample: RewardPageObservation) -> RewardResultRecognitionState {
        if self.state != RewardResultRecognitionState::Pending {
            return self.state.clone();
        }
        self.attempts += 1;
        if !sample.is_complete() {
            self.candidate = None;
            self.consecutive = 0;
        } else if self.candidate.as_ref() == Some(&sample) {
            self.consecutive += 1;
        } else {
            self.candidate = Some(sample);
            self.consecutive = 1;
        }
        if self.consecutive >= self.required {
            self.state = RewardResultRecognitionState::Ready(self.candidate.clone().unwrap());
        } else if self.attempts >= self.max_attempts {
            self.state = RewardResultRecognitionState::Failed;
        }
        self.state.clone()
    }
}

pub fn parse_reward_count(text: &str) -> Option<u32> {
    let normalized = text
        .trim()
        .chars()
        .map(|ch| match ch {
            '０'..='９' => char::from_u32(ch as u32 - '０' as u32 + '0' as u32).unwrap(),
            _ => ch,
        })
        .collect::<String>();
    if normalized.is_empty() || !normalized.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    normalized.parse::<u32>().ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewardPageMerge {
    pub new_cards: Vec<RewardCardObservation>,
    pub duplicate_count: usize,
    pub unchanged_page: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardPageMergeError {
    SamePageCountConflict,
    OverlapCountConflict,
    UnknownOverlapCount,
}

pub fn merge_reward_pages(
    previous: &RewardPageObservation,
    current: &RewardPageObservation,
) -> Result<RewardPageMerge, RewardPageMergeError> {
    if previous.0.is_empty() || current.0.is_empty() || !current.is_complete() {
        return Ok(RewardPageMerge {
            new_cards: Vec::new(),
            duplicate_count: 0,
            unchanged_page: false,
        });
    }
    if previous
        .0
        .iter()
        .map(|card| &card.identity)
        .eq(current.0.iter().map(|card| &card.identity))
    {
        for (before, after) in previous.0.iter().zip(&current.0) {
            if before.count != after.count {
                return Err(RewardPageMergeError::SamePageCountConflict);
            }
        }
        return Ok(RewardPageMerge {
            new_cards: Vec::new(),
            duplicate_count: current.0.len(),
            unchanged_page: true,
        });
    }
    let duplicate_count = (1..=previous.0.len().min(current.0.len()))
        .rev()
        .find(|count| {
            previous.0[previous.0.len() - count..]
                .iter()
                .zip(&current.0[..*count])
                .all(|(before, after)| before.identity == after.identity)
        })
        .unwrap_or(0);
    if duplicate_count > 0 {
        for (before, after) in previous.0[previous.0.len() - duplicate_count..]
            .iter()
            .zip(&current.0[..duplicate_count])
        {
            match (before.count, after.count) {
                (Some(left), Some(right)) if left != right => {
                    return Err(RewardPageMergeError::OverlapCountConflict);
                }
                (None, _) | (_, None) => return Err(RewardPageMergeError::UnknownOverlapCount),
                _ => {}
            }
        }
    }
    Ok(RewardPageMerge {
        new_cards: current.0[duplicate_count..].to_vec(),
        duplicate_count,
        unchanged_page: false,
    })
}

pub fn summarize_reward_cards<'a>(
    cards: impl IntoIterator<Item = &'a RewardCardObservation>,
) -> BTreeMap<String, Option<u32>> {
    let mut summary = BTreeMap::<String, Option<u32>>::new();
    for card in cards {
        let Some(identity) = &card.identity else {
            continue;
        };
        let total = summary.entry(identity.name.clone()).or_insert(None);
        *total = match (*total, card.count) {
            (Some(existing), Some(count)) => Some(existing.saturating_add(count)),
            (None, Some(count)) => Some(count),
            (_, None) => None,
        };
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(name: &str, count: Option<u32>) -> RewardCardObservation {
        RewardCardObservation {
            rect: RewardCardRect {
                x: 0,
                y: 0,
                width: 125,
                height: 153,
            },
            identity: Some(RewardIdentity {
                name: name.into(),
                quality_level: None,
            }),
            count,
        }
    }

    fn page(cards: Vec<RewardCardObservation>) -> RewardPageObservation {
        RewardPageObservation(cards)
    }

    #[test]
    fn count_parser_accepts_only_ascii_or_full_width_digits() {
        assert_eq!(parse_reward_count(" １２３ "), Some(123));
        assert_eq!(parse_reward_count("007"), Some(7));
        for text in ["", "-1", "+1", "1.0", "1 个", "十二", "4294967296"] {
            assert_eq!(parse_reward_count(text), None, "{text:?}");
        }
    }

    #[test]
    fn stability_requires_three_identical_complete_pages_and_resets_on_invalid_samples() {
        let mut stability = RewardResultRecognitionStability::new();
        let sample = page(vec![card("原石", Some(60))]);
        assert_eq!(
            stability.observe(sample.clone()),
            RewardResultRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(sample.clone()),
            RewardResultRecognitionState::Pending
        );
        assert_eq!(
            stability.observe(page(vec![RewardCardObservation {
                identity: None,
                ..sample.0[0].clone()
            }])),
            RewardResultRecognitionState::Pending
        );
        for _ in 0..2 {
            assert_eq!(
                stability.observe(sample.clone()),
                RewardResultRecognitionState::Pending
            );
        }
        assert_eq!(
            stability.observe(sample.clone()),
            RewardResultRecognitionState::Ready(sample.clone())
        );
        assert_eq!(
            stability.observe(page(vec![card("原石", Some(30))])),
            RewardResultRecognitionState::Ready(sample)
        );
    }

    #[test]
    fn unknown_counts_are_preserved_without_defaulting_or_blocking_stable_names() {
        let sample = page(vec![card("材料", None)]);
        let mut stability = RewardResultRecognitionStability::new();
        for _ in 0..3 {
            stability.observe(sample.clone());
        }
        assert_eq!(
            stability.observe(sample.clone()),
            RewardResultRecognitionState::Ready(sample)
        );
        assert_eq!(
            summarize_reward_cards([card("材料", None)].iter()),
            BTreeMap::from([("材料".into(), None)])
        );
    }

    #[test]
    fn stability_fails_after_ten_attempts_without_partial_results() {
        let mut stability = RewardResultRecognitionStability::new();
        for _ in 0..10 {
            stability.observe(page(Vec::new()));
        }
        assert_eq!(
            stability.observe(page(Vec::new())),
            RewardResultRecognitionState::Failed
        );
    }

    #[test]
    fn page_merge_rejects_same_page_and_overlap_count_conflicts() {
        let previous = page(vec![card("A", Some(1)), card("B", Some(2))]);
        assert_eq!(
            merge_reward_pages(
                &previous,
                &page(vec![card("A", Some(3)), card("B", Some(2))])
            ),
            Err(RewardPageMergeError::SamePageCountConflict)
        );
        let current = page(vec![card("B", Some(4)), card("C", Some(5))]);
        assert_eq!(
            merge_reward_pages(&previous, &current),
            Err(RewardPageMergeError::OverlapCountConflict)
        );
        let unknown = page(vec![card("B", None), card("C", Some(5))]);
        assert_eq!(
            merge_reward_pages(&previous, &unknown),
            Err(RewardPageMergeError::UnknownOverlapCount)
        );
    }

    #[test]
    fn page_merge_removes_longest_suffix_prefix_overlap_and_never_duplicates_unchanged_page() {
        let previous = page(vec![
            card("A", Some(1)),
            card("B", Some(2)),
            card("C", Some(3)),
        ]);
        let current = page(vec![
            card("B", Some(2)),
            card("C", Some(3)),
            card("D", Some(4)),
        ]);
        let merged = merge_reward_pages(&previous, &current).unwrap();
        assert_eq!(merged.duplicate_count, 2);
        assert_eq!(merged.new_cards, vec![card("D", Some(4))]);
        assert!(!merged.unchanged_page);
        assert_eq!(
            merge_reward_pages(&previous, &previous).unwrap(),
            RewardPageMerge {
                new_cards: Vec::new(),
                duplicate_count: 3,
                unchanged_page: true,
            }
        );
        let cards = [card("A", Some(2)), card("A", Some(3))];
        assert_eq!(
            summarize_reward_cards(cards.iter()),
            BTreeMap::from([("A".into(), Some(5))])
        );
    }
}
