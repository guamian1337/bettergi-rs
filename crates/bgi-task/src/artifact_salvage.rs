use std::collections::HashSet;
use thiserror::Error;

pub const MAX_SALVAGE_ITEMS: u16 = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SalvageConfig {
    pub max_star: u8,
    pub max_items: u16,
}

impl SalvageConfig {
    pub fn validate(self) -> Result<(), SalvageError> {
        if !(1..=4).contains(&self.max_star) {
            return Err(SalvageError::InvalidMaxStar(self.max_star));
        }
        if self.max_items == 0 || self.max_items > MAX_SALVAGE_ITEMS {
            return Err(SalvageError::InvalidItemLimit(self.max_items));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactCard {
    pub id: u64,
    pub star: u8,
    pub level: u8,
    pub locked: bool,
    pub selected: bool,
    pub recognized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SalvagePreview {
    pub selected_ids: Vec<u64>,
    pub skipped_locked: usize,
    pub skipped_selected: usize,
    pub skipped_five_star: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SalvageError {
    #[error("自动分解星级上限必须为 1 到 4，实际为 {0}")]
    InvalidMaxStar(u8),
    #[error("单批数量上限必须为 1 到 {MAX_SALVAGE_ITEMS}，实际为 {0}")]
    InvalidItemLimit(u16),
    #[error("圣遗物编号 {0} 重复")]
    DuplicateId(u64),
    #[error("圣遗物编号 {id} 的星级无效")]
    InvalidStar { id: u64, star: u8 },
    #[error("圣遗物编号 {id} 的等级超出 0 到 20")]
    InvalidLevel { id: u64, level: u8 },
    #[error("圣遗物编号 {0} 的识别置信度不足")]
    UnrecognizedCard(u64),
    #[error("符合条件数量 {eligible} 超过本批上限 {limit}")]
    BudgetExceeded { eligible: usize, limit: u16 },
    #[error("没有可预览的圣遗物")]
    EmptySelection,
    #[error("选中数量不一致：预期 {expected}，实际 {observed}")]
    SelectionCountMismatch { expected: u16, observed: u16 },
    #[error("未识别到局部破坏性确认按钮")]
    ConfirmationNotRecognized,
    #[error("需要人工最终确认")]
    HumanApprovalRequired,
}

pub fn plan_salvage_preview(
    cards: &[ArtifactCard],
    config: SalvageConfig,
) -> Result<SalvagePreview, SalvageError> {
    config.validate()?;
    let mut seen = HashSet::with_capacity(cards.len());
    let mut preview = SalvagePreview {
        selected_ids: Vec::new(),
        skipped_locked: 0,
        skipped_selected: 0,
        skipped_five_star: 0,
    };

    for card in cards {
        if !seen.insert(card.id) {
            return Err(SalvageError::DuplicateId(card.id));
        }
        if !(1..=5).contains(&card.star) {
            return Err(SalvageError::InvalidStar {
                id: card.id,
                star: card.star,
            });
        }
        if card.level > 20 {
            return Err(SalvageError::InvalidLevel {
                id: card.id,
                level: card.level,
            });
        }
        if !card.recognized {
            return Err(SalvageError::UnrecognizedCard(card.id));
        }
        if card.locked {
            preview.skipped_locked += 1;
            continue;
        }
        if card.selected {
            preview.skipped_selected += 1;
            continue;
        }
        if card.star == 5 {
            preview.skipped_five_star += 1;
            continue;
        }
        if card.star <= config.max_star {
            preview.selected_ids.push(card.id);
            if preview.selected_ids.len() > config.max_items as usize {
                return Err(SalvageError::BudgetExceeded {
                    eligible: preview.selected_ids.len(),
                    limit: config.max_items,
                });
            }
        }
    }

    if preview.selected_ids.is_empty() {
        Err(SalvageError::EmptySelection)
    } else {
        Ok(preview)
    }
}

pub fn authorize_final_confirmation(
    expected_count: u16,
    observed_count: u16,
    confirmation_button_recognized: bool,
    human_approved: bool,
) -> Result<(), SalvageError> {
    if expected_count == 0 {
        return Err(SalvageError::EmptySelection);
    }
    if expected_count != observed_count {
        return Err(SalvageError::SelectionCountMismatch {
            expected: expected_count,
            observed: observed_count,
        });
    }
    if !confirmation_button_recognized {
        return Err(SalvageError::ConfirmationNotRecognized);
    }
    if !human_approved {
        return Err(SalvageError::HumanApprovalRequired);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(id: u64, star: u8) -> ArtifactCard {
        ArtifactCard {
            id,
            star,
            level: 0,
            locked: false,
            selected: false,
            recognized: true,
        }
    }

    #[test]
    fn preview_selects_only_unlocked_unselected_cards_at_or_below_star_limit() {
        let mut locked = card(2, 3);
        locked.locked = true;
        let mut selected = card(3, 2);
        selected.selected = true;
        let preview = plan_salvage_preview(
            &[card(1, 1), locked, selected, card(4, 4), card(5, 5)],
            SalvageConfig {
                max_star: 4,
                max_items: 10,
            },
        )
        .unwrap();
        assert_eq!(preview.selected_ids, [1, 4]);
        assert_eq!(preview.skipped_locked, 1);
        assert_eq!(preview.skipped_selected, 1);
        assert_eq!(preview.skipped_five_star, 1);
    }

    #[test]
    fn all_non_five_star_cards_are_eligible_at_four_star_limit() {
        let cards: Vec<_> = (1..=4).map(|star| card(u64::from(star), star)).collect();
        let preview = plan_salvage_preview(
            &cards,
            SalvageConfig {
                max_star: 4,
                max_items: 4,
            },
        )
        .unwrap();
        assert_eq!(preview.selected_ids, [1, 2, 3, 4]);
    }

    #[test]
    fn malformed_or_low_confidence_cards_fail_closed() {
        let config = SalvageConfig {
            max_star: 4,
            max_items: 10,
        };
        let mut unknown = card(1, 2);
        unknown.recognized = false;
        assert_eq!(
            plan_salvage_preview(&[unknown], config),
            Err(SalvageError::UnrecognizedCard(1))
        );
        assert_eq!(
            plan_salvage_preview(&[card(1, 0)], config),
            Err(SalvageError::InvalidStar { id: 1, star: 0 })
        );
        assert_eq!(
            plan_salvage_preview(&[card(1, 6)], config),
            Err(SalvageError::InvalidStar { id: 1, star: 6 })
        );
        let mut invalid_level = card(1, 2);
        invalid_level.level = 21;
        assert_eq!(
            plan_salvage_preview(&[invalid_level], config),
            Err(SalvageError::InvalidLevel { id: 1, level: 21 })
        );
        assert_eq!(
            plan_salvage_preview(&[card(1, 2), card(1, 3)], config),
            Err(SalvageError::DuplicateId(1))
        );
    }

    #[test]
    fn invalid_limits_and_budget_overflow_are_rejected() {
        assert_eq!(
            SalvageConfig {
                max_star: 5,
                max_items: 1,
            }
            .validate(),
            Err(SalvageError::InvalidMaxStar(5))
        );
        assert_eq!(
            SalvageConfig {
                max_star: 4,
                max_items: 0,
            }
            .validate(),
            Err(SalvageError::InvalidItemLimit(0))
        );
        assert_eq!(
            SalvageConfig {
                max_star: 4,
                max_items: MAX_SALVAGE_ITEMS + 1,
            }
            .validate(),
            Err(SalvageError::InvalidItemLimit(MAX_SALVAGE_ITEMS + 1))
        );
        assert_eq!(
            plan_salvage_preview(
                &[card(1, 2), card(2, 3)],
                SalvageConfig {
                    max_star: 4,
                    max_items: 1,
                }
            ),
            Err(SalvageError::BudgetExceeded {
                eligible: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn empty_or_only_protected_cards_never_authorize_an_action() {
        let config = SalvageConfig {
            max_star: 4,
            max_items: 10,
        };
        assert_eq!(
            plan_salvage_preview(&[], config),
            Err(SalvageError::EmptySelection)
        );
        assert_eq!(
            plan_salvage_preview(&[card(1, 5)], config),
            Err(SalvageError::EmptySelection)
        );
    }

    #[test]
    fn final_confirmation_requires_matching_counts_visible_button_and_human_approval() {
        assert_eq!(
            authorize_final_confirmation(0, 0, true, true),
            Err(SalvageError::EmptySelection)
        );
        assert_eq!(
            authorize_final_confirmation(3, 2, true, true),
            Err(SalvageError::SelectionCountMismatch {
                expected: 3,
                observed: 2,
            })
        );
        assert_eq!(
            authorize_final_confirmation(3, 3, false, true),
            Err(SalvageError::ConfirmationNotRecognized)
        );
        assert_eq!(
            authorize_final_confirmation(3, 3, true, false),
            Err(SalvageError::HumanApprovalRequired)
        );
        assert_eq!(authorize_final_confirmation(3, 3, true, true), Ok(()));
    }
}
