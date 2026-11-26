use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::artifact_salvage::MAX_SALVAGE_ITEMS;

pub const MAX_SALVAGE_PREVIEW_PAGES: u8 = 20;
pub const MAX_SALVAGE_CELLS_PER_PAGE: usize = 36;
pub const MAX_SALVAGE_INTERACTIONS: u16 = 40;
pub const MAX_SALVAGE_NO_PROGRESS: u8 = 5;
const EMPTY_PAGE_CONFIRMATIONS: u8 = 2;
const MIN_PRESENT_CONFIDENCE: f64 = 0.5;
const MAX_ABSENT_CONFIDENCE: f64 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SalvagePreviewConfig {
    pub max_pages: u8,
    pub max_cells_per_page: usize,
    /// Input budget for pagination only. Card-selection and salvage clicks do not exist in this API.
    pub interaction_budget: u16,
    pub max_no_progress: u8,
    pub min_confidence: f64,
}

impl Default for SalvagePreviewConfig {
    fn default() -> Self {
        Self {
            max_pages: 8,
            max_cells_per_page: MAX_SALVAGE_CELLS_PER_PAGE,
            interaction_budget: 24,
            max_no_progress: 3,
            min_confidence: 0.90,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SalvagePreviewError {
    #[error("分解预览页数上限必须为 1 到 {MAX_SALVAGE_PREVIEW_PAGES}，实际为 {0}")]
    InvalidPageLimit(u8),
    #[error("每页格子数上限必须为 1 到 {MAX_SALVAGE_CELLS_PER_PAGE}，实际为 {0}")]
    InvalidCellLimit(usize),
    #[error("分页输入预算必须为 1 到 {MAX_SALVAGE_INTERACTIONS}，实际为 {0}")]
    InvalidInteractionBudget(u16),
    #[error("无进展停止阈值必须为 1 到 {MAX_SALVAGE_NO_PROGRESS}，实际为 {0}")]
    InvalidNoProgressLimit(u8),
    #[error("置信度阈值必须为 0.5 到 1.0，实际为 {0:?}")]
    InvalidConfidenceThreshold(f64),
    #[error("只支持 16:9 帧，实际为 {width}x{height}")]
    UnsupportedAspectRatio { width: u32, height: u32 },
    #[error("未确认位于圣遗物分解界面")]
    SalvagePageNotVisible,
    #[error("检测到最终破坏性确认弹窗，预览立即停止")]
    DestructiveConfirmationVisible,
    #[error("本页格子数 {observed} 超过上限 {limit}")]
    TooManyCells { observed: usize, limit: usize },
    #[error("本页必须恰好包含 {MAX_SALVAGE_CELLS_PER_PAGE} 个格子，实际为 {0}")]
    InvalidCellCount(usize),
    #[error("本页格子位置 {0} 重复")]
    DuplicateSlot(usize),
    #[error("格子位置 {slot} 的占用状态不确定：{confidence:?}")]
    UncertainPresence { slot: usize, confidence: f64 },
    #[error("格子位置 {slot} 缺少星级或置信度不足：{confidence:?}")]
    LowConfidence {
        slot: usize,
        confidence: Option<f64>,
    },
    #[error("格子位置 {slot} 的星级无效：{star}")]
    InvalidStar { slot: usize, star: u8 },
    #[error("候选数量 {eligible} 超过上限 {limit}")]
    CandidateBudgetExceeded { eligible: usize, limit: u16 },
    #[error("预览状态机已结束：{0}")]
    AlreadyFinished(&'static str),
}

impl SalvagePreviewConfig {
    pub fn validate(self) -> Result<(), SalvagePreviewError> {
        if self.max_pages == 0 || self.max_pages > MAX_SALVAGE_PREVIEW_PAGES {
            return Err(SalvagePreviewError::InvalidPageLimit(self.max_pages));
        }
        if self.max_cells_per_page == 0 || self.max_cells_per_page > MAX_SALVAGE_CELLS_PER_PAGE {
            return Err(SalvagePreviewError::InvalidCellLimit(
                self.max_cells_per_page,
            ));
        }
        if self.interaction_budget == 0 || self.interaction_budget > MAX_SALVAGE_INTERACTIONS {
            return Err(SalvagePreviewError::InvalidInteractionBudget(
                self.interaction_budget,
            ));
        }
        if self.max_no_progress == 0 || self.max_no_progress > MAX_SALVAGE_NO_PROGRESS {
            return Err(SalvagePreviewError::InvalidNoProgressLimit(
                self.max_no_progress,
            ));
        }
        if !self.min_confidence.is_finite() || !(0.5..=1.0).contains(&self.min_confidence) {
            return Err(SalvagePreviewError::InvalidConfidenceThreshold(
                self.min_confidence,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SalvageCellRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SalvageCardObservation {
    pub slot: usize,
    pub rect: SalvageCellRect,
    pub star: Option<u8>,
    pub confidence: f64,
    pub presence_confidence: f64,
    pub name: String,
    pub visual_id: u64,
    pub present: bool,
    pub locked: bool,
    pub selected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SalvagePreviewFrame {
    pub width: u32,
    pub height: u32,
    pub salvage_page_visible: bool,
    pub final_confirmation_visible: bool,
    /// CV page fingerprint used only to detect scrolling progress and overlapping pages.
    pub fingerprint: u64,
    pub scroll_center: (i32, i32),
    pub cards: Vec<SalvageCardObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SalvagePosition {
    pub page: u8,
    pub slot: usize,
    pub row: u8,
    pub column: u8,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SalvageCandidate {
    /// Preview-session observation ID (`page << 32 | slot`), not a persistent game item ID.
    pub id: u64,
    pub position: SalvagePosition,
    pub star: u8,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SalvageItemKey {
    name: String,
    star: u8,
    visual_id: u64,
}

impl SalvageItemKey {
    fn from_card(card: &SalvageCardObservation) -> Self {
        Self {
            name: card.name.clone(),
            star: card.star.unwrap_or(0),
            visual_id: card.visual_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SalvageScrollRequest {
    pub center: (i32, i32),
    pub notches: i32,
}

/// The only action this state machine can emit is a wheel scroll over the grid.
/// There is deliberately no select, salvage, confirm, or final-confirmation action variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SalvagePreviewAction {
    Scroll(SalvageScrollRequest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SalvagePreviewStopReason {
    Complete,
    EmptyPage,
    PageLimit,
    NoProgress,
    InteractionBudget,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SalvageSkipStats {
    pub empty: usize,
    pub locked: usize,
    pub selected: usize,
    pub five_star: usize,
    pub overlapping: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SalvagePreviewReport {
    pub candidates: Vec<SalvageCandidate>,
    pub skipped: SalvageSkipStats,
    pub pages_seen: u8,
    pub interactions_used: u16,
    pub no_progress: u8,
    pub selection_clicks: u16,
    pub salvage_clicks: u16,
    pub confirmation_clicks: u16,
    pub stop_reason: Option<SalvagePreviewStopReason>,
    pub failure: Option<String>,
    /// Always false in this read-only phase; no report can authorize destruction.
    pub final_confirmation_authorized: bool,
}

impl SalvagePreviewReport {
    pub fn to_json(&self) -> String {
        // The report contains only primitive values, so serialization cannot fail.
        serde_json::to_string(self).expect("salvage preview report is serializable")
    }
}

#[derive(Debug, Clone)]
pub struct SalvagePreviewStateMachine {
    config: SalvagePreviewConfig,
    report: SalvagePreviewReport,
    previous_page_items: std::collections::HashMap<SalvageItemKey, usize>,
    visited_fingerprints: std::collections::HashSet<u64>,
    empty_fingerprint: Option<u64>,
    empty_confirmations: u8,
}

impl SalvagePreviewStateMachine {
    pub fn new(config: SalvagePreviewConfig) -> Result<Self, SalvagePreviewError> {
        config.validate()?;
        Ok(Self {
            config,
            report: SalvagePreviewReport {
                candidates: Vec::new(),
                skipped: SalvageSkipStats::default(),
                pages_seen: 0,
                interactions_used: 0,
                no_progress: 0,
                selection_clicks: 0,
                salvage_clicks: 0,
                confirmation_clicks: 0,
                stop_reason: None,
                failure: None,
                final_confirmation_authorized: false,
            },
            previous_page_items: std::collections::HashMap::new(),
            visited_fingerprints: std::collections::HashSet::new(),
            empty_fingerprint: None,
            empty_confirmations: 0,
        })
    }

    pub fn report(&self) -> &SalvagePreviewReport {
        &self.report
    }

    pub fn is_finished(&self) -> bool {
        self.report.stop_reason.is_some()
    }

    pub fn cancel(&mut self) {
        if !self.is_finished() {
            self.report.failure = Some("分解预览已取消".into());
            self.finish(SalvagePreviewStopReason::Cancelled);
        }
    }

    pub fn fail_closed(&mut self, message: String) {
        if !self.is_finished() {
            self.fail(message);
        }
    }

    pub fn observe(
        &mut self,
        frame: SalvagePreviewFrame,
    ) -> Result<Option<SalvagePreviewAction>, SalvagePreviewError> {
        if self.is_finished() {
            return Err(SalvagePreviewError::AlreadyFinished("preview ended"));
        }
        if let Err(error) = self.validate_frame(&frame) {
            self.fail(error.to_string());
            return Err(error);
        }
        if frame.cards.iter().all(|card| !card.present) {
            if self.empty_fingerprint == Some(frame.fingerprint) {
                self.empty_confirmations = self.empty_confirmations.saturating_add(1);
            } else {
                self.empty_fingerprint = Some(frame.fingerprint);
                self.empty_confirmations = 1;
            }
            if self.empty_confirmations >= EMPTY_PAGE_CONFIRMATIONS {
                self.finish(SalvagePreviewStopReason::EmptyPage);
            }
            return Ok(None);
        }
        self.empty_fingerprint = None;
        self.empty_confirmations = 0;
        if self.report.pages_seen >= self.config.max_pages {
            self.finish(SalvagePreviewStopReason::PageLimit);
            return Ok(None);
        }

        let page = self.report.pages_seen;
        self.report.pages_seen += 1;
        let page_revisited = !self.visited_fingerprints.insert(frame.fingerprint);
        if page_revisited {
            self.report.no_progress = self.report.no_progress.saturating_add(1);
        } else {
            self.report.no_progress = 0;
        }

        let result = self.collect_page(page, &frame, page_revisited);
        if let Err(error) = result {
            self.fail(error.to_string());
            return Err(error);
        }

        if self.report.no_progress >= self.config.max_no_progress {
            self.finish(SalvagePreviewStopReason::NoProgress);
            return Ok(None);
        }
        if self.report.pages_seen >= self.config.max_pages {
            self.finish(SalvagePreviewStopReason::PageLimit);
            return Ok(None);
        }
        if self.report.interactions_used >= self.config.interaction_budget {
            self.finish(SalvagePreviewStopReason::InteractionBudget);
            return Ok(None);
        }
        self.report.interactions_used += 1;
        Ok(Some(SalvagePreviewAction::Scroll(SalvageScrollRequest {
            center: frame.scroll_center,
            notches: -3,
        })))
    }

    fn validate_frame(&self, frame: &SalvagePreviewFrame) -> Result<(), SalvagePreviewError> {
        if u64::from(frame.width) * 9 != u64::from(frame.height) * 16 {
            return Err(SalvagePreviewError::UnsupportedAspectRatio {
                width: frame.width,
                height: frame.height,
            });
        }
        if frame.final_confirmation_visible {
            return Err(SalvagePreviewError::DestructiveConfirmationVisible);
        }
        if !frame.salvage_page_visible {
            return Err(SalvagePreviewError::SalvagePageNotVisible);
        }
        if frame.cards.len() > self.config.max_cells_per_page {
            return Err(SalvagePreviewError::TooManyCells {
                observed: frame.cards.len(),
                limit: self.config.max_cells_per_page,
            });
        }
        if frame.cards.len() != MAX_SALVAGE_CELLS_PER_PAGE {
            return Err(SalvagePreviewError::InvalidCellCount(frame.cards.len()));
        }
        if frame.scroll_center.0 < 0
            || frame.scroll_center.1 < 0
            || frame.scroll_center.0 >= frame.width as i32
            || frame.scroll_center.1 >= frame.height as i32
        {
            return Err(SalvagePreviewError::SalvagePageNotVisible);
        }
        let mut slots = std::collections::HashSet::with_capacity(frame.cards.len());
        for card in &frame.cards {
            if card.slot >= MAX_SALVAGE_CELLS_PER_PAGE || !slots.insert(card.slot) {
                return Err(SalvagePreviewError::DuplicateSlot(card.slot));
            }
            if card.rect.width <= 0
                || card.rect.height <= 0
                || card.rect.x < 0
                || card.rect.y < 0
                || i64::from(card.rect.x) + i64::from(card.rect.width) > i64::from(frame.width)
                || i64::from(card.rect.y) + i64::from(card.rect.height) > i64::from(frame.height)
            {
                return Err(SalvagePreviewError::SalvagePageNotVisible);
            }
            if !card.presence_confidence.is_finite()
                || (card.present && card.presence_confidence < MIN_PRESENT_CONFIDENCE)
                || (!card.present && card.presence_confidence > MAX_ABSENT_CONFIDENCE)
            {
                return Err(SalvagePreviewError::UncertainPresence {
                    slot: card.slot,
                    confidence: card.presence_confidence,
                });
            }
            if card.present {
                let Some(star) = card.star else {
                    return Err(SalvagePreviewError::LowConfidence {
                        slot: card.slot,
                        confidence: None,
                    });
                };
                if !card.confidence.is_finite() || card.confidence < self.config.min_confidence {
                    return Err(SalvagePreviewError::LowConfidence {
                        slot: card.slot,
                        confidence: Some(card.confidence),
                    });
                }
                if !(1..=5).contains(&star) {
                    return Err(SalvagePreviewError::InvalidStar {
                        slot: card.slot,
                        star,
                    });
                }
                if card.name.trim().is_empty() {
                    return Err(SalvagePreviewError::LowConfidence {
                        slot: card.slot,
                        confidence: Some(card.confidence),
                    });
                }
            }
        }
        Ok(())
    }

    fn collect_page(
        &mut self,
        page: u8,
        frame: &SalvagePreviewFrame,
        page_revisited: bool,
    ) -> Result<(), SalvagePreviewError> {
        let mut current_items = std::collections::HashMap::<SalvageItemKey, usize>::new();
        for card in frame.cards.iter().filter(|card| card.present) {
            *current_items
                .entry(SalvageItemKey::from_card(card))
                .or_default() += 1;
        }
        let overlaps_previous = current_items
            .keys()
            .any(|key| self.previous_page_items.contains_key(key));
        let mut carryover = std::collections::HashMap::<SalvageItemKey, usize>::new();
        for (key, count) in &current_items {
            let previous_count = if page_revisited {
                *count
            } else if overlaps_previous {
                self.previous_page_items.get(key).copied().unwrap_or(0)
            } else {
                0
            };
            carryover.insert(key.clone(), (*count).min(previous_count));
        }

        let mut ordered: Vec<_> = frame.cards.iter().collect();
        ordered.sort_by_key(|card| card.slot);
        for card in ordered {
            if !card.present {
                self.report.skipped.empty += 1;
                continue;
            }
            let key = SalvageItemKey::from_card(card);
            if carryover.get_mut(&key).is_some_and(|remaining| {
                if *remaining > 0 {
                    *remaining -= 1;
                    true
                } else {
                    false
                }
            }) {
                self.report.skipped.overlapping += 1;
                continue;
            }
            let Some(star) = card.star else {
                return Err(SalvagePreviewError::LowConfidence {
                    slot: card.slot,
                    confidence: None,
                });
            };
            if card.locked {
                self.report.skipped.locked += 1;
                continue;
            }
            if card.selected {
                self.report.skipped.selected += 1;
                continue;
            }
            if star == 5 {
                self.report.skipped.five_star += 1;
                continue;
            }
            if self.report.candidates.len() >= MAX_SALVAGE_ITEMS as usize {
                return Err(SalvagePreviewError::CandidateBudgetExceeded {
                    eligible: self.report.candidates.len() + 1,
                    limit: MAX_SALVAGE_ITEMS,
                });
            }
            self.report.candidates.push(SalvageCandidate {
                id: (u64::from(page) << 32) | card.slot as u64,
                position: SalvagePosition {
                    page,
                    slot: card.slot,
                    row: (card.slot / 9) as u8,
                    column: (card.slot % 9) as u8,
                    x: card.rect.x,
                    y: card.rect.y,
                    width: card.rect.width,
                    height: card.rect.height,
                },
                star,
                confidence: card.confidence,
            });
        }
        self.previous_page_items = current_items;
        Ok(())
    }

    fn finish(&mut self, reason: SalvagePreviewStopReason) {
        self.report.stop_reason = Some(reason);
    }

    fn fail(&mut self, message: String) {
        self.report.candidates.clear();
        self.report.failure = Some(message);
        self.finish(SalvagePreviewStopReason::Failed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(slot: usize, star: u8, confidence: f64) -> SalvageCardObservation {
        SalvageCardObservation {
            slot,
            rect: SalvageCellRect {
                x: 48 + (slot % 9) as i32 * 141,
                y: 106 + (slot / 9) as i32 * 192,
                width: 141,
                height: 192,
            },
            star: Some(star),
            confidence,
            presence_confidence: confidence,
            name: "测试圣遗物".into(),
            visual_id: 1,
            present: true,
            locked: false,
            selected: false,
        }
    }

    fn frame(fingerprint: u64, cards: Vec<SalvageCardObservation>) -> SalvagePreviewFrame {
        let mut all_cards: Vec<_> = (0..MAX_SALVAGE_CELLS_PER_PAGE)
            .map(|slot| SalvageCardObservation {
                slot,
                rect: SalvageCellRect {
                    x: 48 + (slot % 9) as i32 * 141,
                    y: 106 + (slot / 9) as i32 * 192,
                    width: 141,
                    height: 192,
                },
                star: None,
                confidence: 0.0,
                presence_confidence: 0.0,
                name: String::new(),
                visual_id: 0,
                present: false,
                locked: false,
                selected: false,
            })
            .collect();
        for card in cards {
            let slot = card.slot;
            all_cards[slot] = card;
        }
        SalvagePreviewFrame {
            width: 1920,
            height: 1080,
            salvage_page_visible: true,
            final_confirmation_visible: false,
            fingerprint,
            scroll_center: (680, 490),
            cards: all_cards,
        }
    }

    #[test]
    fn pagination_is_bounded_and_only_requests_scrolls() {
        let config = SalvagePreviewConfig {
            max_pages: 3,
            interaction_budget: 2,
            ..Default::default()
        };
        let mut state = SalvagePreviewStateMachine::new(config).unwrap();
        for page in 1..=3 {
            let mut observed = card(0, 4, 0.99);
            observed.visual_id = page;
            let action = state.observe(frame(page, vec![observed])).unwrap();
            if page < 3 {
                assert!(matches!(
                    action,
                    Some(SalvagePreviewAction::Scroll(SalvageScrollRequest {
                        notches: -3,
                        ..
                    }))
                ));
            } else {
                assert_eq!(action, None);
            }
        }
        let report = state.report();
        assert_eq!(report.pages_seen, 3);
        assert_eq!(report.interactions_used, 2);
        assert_eq!(report.candidates.len(), 3);
        assert_eq!(
            report.stop_reason,
            Some(SalvagePreviewStopReason::PageLimit)
        );
        assert_eq!(report.selection_clicks, 0);
        assert_eq!(report.salvage_clicks, 0);
        assert_eq!(report.confirmation_clicks, 0);
        assert!(!report.final_confirmation_authorized);
    }

    #[test]
    fn scrolling_without_progress_stops_at_the_configured_limit() {
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig {
            max_no_progress: 2,
            ..Default::default()
        })
        .unwrap();
        state.observe(frame(7, vec![card(0, 3, 0.95)])).unwrap();
        state.observe(frame(7, vec![card(0, 3, 0.95)])).unwrap();
        assert_eq!(state.observe(frame(7, vec![card(0, 3, 0.95)])), Ok(None));
        assert_eq!(state.report().pages_seen, 3);
        assert_eq!(state.report().interactions_used, 2);
        assert_eq!(state.report().skipped.overlapping, 2);
        assert_eq!(
            state.report().stop_reason,
            Some(SalvagePreviewStopReason::NoProgress)
        );
    }

    #[test]
    fn overlapping_pages_deduplicate_by_item_identity_and_multiplicity() {
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
        state
            .observe(frame(1, vec![card(0, 4, 0.99), card(1, 4, 0.99)]))
            .unwrap();
        assert_eq!(state.report().candidates.len(), 2);

        let mut new_item = card(2, 3, 0.99);
        new_item.visual_id = 2;
        state
            .observe(frame(2, vec![card(0, 4, 0.99), card(1, 4, 0.99), new_item]))
            .unwrap();
        assert_eq!(state.report().candidates.len(), 3);
        assert_eq!(state.report().skipped.overlapping, 2);
    }

    #[test]
    fn any_failure_discards_all_partial_candidates() {
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
        let placeholder = SalvageCandidate {
            id: 0,
            position: SalvagePosition {
                page: 0,
                slot: 0,
                row: 0,
                column: 0,
                x: 48,
                y: 106,
                width: 141,
                height: 192,
            },
            star: 4,
            confidence: 0.99,
        };
        state.report.candidates.resize(199, placeholder);
        let mut second = card(1, 4, 0.99);
        second.visual_id = 2;
        assert!(matches!(
            state.observe(frame(1, vec![card(0, 4, 0.99), second])),
            Err(SalvagePreviewError::CandidateBudgetExceeded { .. })
        ));
        assert!(state.report().candidates.is_empty());
        assert_eq!(
            state.report().stop_reason,
            Some(SalvagePreviewStopReason::Failed)
        );
        assert!(!state.report().final_confirmation_authorized);
    }

    #[test]
    fn empty_page_requires_two_stable_high_confidence_observations() {
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
        assert_eq!(state.observe(frame(7, vec![])), Ok(None));
        assert!(!state.is_finished());
        assert_eq!(state.report().pages_seen, 0);
        assert_eq!(state.observe(frame(8, vec![])), Ok(None));
        assert!(!state.is_finished());
        assert_eq!(state.observe(frame(8, vec![])), Ok(None));
        assert!(state.is_finished());
        assert_eq!(
            state.report().stop_reason,
            Some(SalvagePreviewStopReason::EmptyPage)
        );
        assert_eq!(state.report().interactions_used, 0);
    }

    #[test]
    fn uncertain_presence_never_becomes_an_empty_page() {
        for (present, confidence) in [(false, 0.5), (true, 0.49), (false, f64::NAN)] {
            let mut state =
                SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
            let mut observation = frame(1, vec![]);
            observation.cards[0].present = present;
            observation.cards[0].presence_confidence = confidence;
            assert!(matches!(
                state.observe(observation),
                Err(SalvagePreviewError::UncertainPresence { .. })
            ));
            assert_eq!(
                state.report().stop_reason,
                Some(SalvagePreviewStopReason::Failed)
            );
        }
    }

    #[test]
    fn alternating_page_fingerprints_are_detected_as_no_progress() {
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig {
            max_no_progress: 2,
            ..Default::default()
        })
        .unwrap();
        state.observe(frame(1, vec![card(0, 3, 0.95)])).unwrap();
        state.observe(frame(2, vec![card(0, 3, 0.95)])).unwrap();
        state.observe(frame(1, vec![card(0, 3, 0.95)])).unwrap();
        assert_eq!(
            state.observe(frame(2, vec![card(0, 3, 0.95)])).unwrap(),
            None
        );
        assert_eq!(
            state.report().stop_reason,
            Some(SalvagePreviewStopReason::NoProgress)
        );
    }

    #[test]
    fn five_star_locked_selected_and_empty_cells_are_never_candidates() {
        let mut locked = card(1, 5, 1.0);
        locked.locked = true;
        let mut selected = card(2, 4, 1.0);
        selected.selected = true;
        let mut empty = card(3, 4, 1.0);
        empty.present = false;
        empty.confidence = 0.0;
        empty.presence_confidence = 0.0;
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
        state
            .observe(frame(1, vec![card(0, 5, 1.0), locked, selected, empty]))
            .unwrap();
        assert!(state.report().candidates.is_empty());
        assert_eq!(state.report().skipped.five_star, 1);
        assert_eq!(state.report().skipped.locked, 1);
        assert_eq!(state.report().skipped.selected, 1);
        assert_eq!(state.report().skipped.empty, MAX_SALVAGE_CELLS_PER_PAGE - 3);
    }

    #[test]
    fn missing_star_and_low_or_non_finite_confidence_fail_closed() {
        for (star, confidence) in [(None, 1.0), (Some(4), 0.89), (Some(4), f64::NAN)] {
            let mut state =
                SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
            let mut observation = frame(1, vec![card(0, 4, confidence)]);
            observation.cards[0].star = star;
            observation.cards[0].presence_confidence = 1.0;
            assert!(matches!(
                state.observe(observation),
                Err(SalvagePreviewError::LowConfidence { .. })
            ));
            assert!(state.report().candidates.is_empty());
            assert_eq!(
                state.report().stop_reason,
                Some(SalvagePreviewStopReason::Failed)
            );
            assert!(!state.report().final_confirmation_authorized);
        }
    }

    #[test]
    fn invalid_page_popup_aspect_and_cell_limits_fail_closed() {
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig {
            max_cells_per_page: 1,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            state.observe(frame(1, vec![card(0, 4, 1.0), card(1, 4, 1.0)])),
            Err(SalvagePreviewError::TooManyCells { .. })
        ));
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
        let mut popup = frame(1, vec![card(0, 4, 1.0)]);
        popup.final_confirmation_visible = true;
        assert!(matches!(
            state.observe(popup),
            Err(SalvagePreviewError::DestructiveConfirmationVisible)
        ));
        let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default()).unwrap();
        let mut wrong_aspect = frame(1, vec![card(0, 4, 1.0)]);
        wrong_aspect.width = 1600;
        wrong_aspect.height = 1000;
        assert!(matches!(
            state.observe(wrong_aspect),
            Err(SalvagePreviewError::UnsupportedAspectRatio { .. })
        ));
    }

    #[test]
    fn configuration_limits_are_enforced() {
        for config in [
            SalvagePreviewConfig {
                max_pages: 0,
                ..Default::default()
            },
            SalvagePreviewConfig {
                max_pages: MAX_SALVAGE_PREVIEW_PAGES + 1,
                ..Default::default()
            },
            SalvagePreviewConfig {
                max_cells_per_page: 0,
                ..Default::default()
            },
            SalvagePreviewConfig {
                max_cells_per_page: MAX_SALVAGE_CELLS_PER_PAGE + 1,
                ..Default::default()
            },
            SalvagePreviewConfig {
                interaction_budget: 0,
                ..Default::default()
            },
            SalvagePreviewConfig {
                max_no_progress: 0,
                ..Default::default()
            },
            SalvagePreviewConfig {
                min_confidence: 0.49,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }
}
