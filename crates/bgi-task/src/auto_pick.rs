//! M3 自动拾取的纯 OCR 识别与决策核心。
//!
//! 本模块只接收已经识别出的单行文字，不包含模板检测、截图或输入模拟。

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// OCR 文字在画面中的归一化中心位置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoiCenter {
    pub x: f64,
    pub y: f64,
}

/// 单行 OCR 候选。`confidence` 与 `roi_center` 均使用 `0.0..=1.0`。
#[derive(Debug, Clone, PartialEq)]
pub struct OcrCandidate {
    pub text: String,
    pub roi_center: RoiCenter,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoPickMode {
    /// 只拾取白名单精确命中或足够相似的文字。
    #[default]
    Whitelist,
    /// 拾取未命中黑名单的文字；聊天、设置等安全排除始终优先。
    Blacklist,
}

/// 自动拾取决策配置。
///
/// 默认采用白名单模式，空名单不会触发交互。黑名单同时约束白名单的精确与模糊结果。
#[derive(Debug, Clone, PartialEq)]
pub struct AutoPickConfig {
    pub mode: AutoPickMode,
    pub whitelist: Vec<String>,
    pub blacklist: Vec<String>,
    /// 全局交互冷却；拒绝候选不会刷新冷却时间。
    pub cooldown: Duration,
    pub allow_fuzzy: bool,
    /// 模糊相似度阈值，取值范围 `0.0..=1.0`。
    pub fuzzy_threshold: f64,
    /// OCR 最低置信度，取值范围 `0.0..=1.0`。
    pub min_confidence: f64,
    /// 额外的排除关键词；内置的聊天、设置与 L 键排除不可被配置覆盖。
    pub excluded_keywords: Vec<String>,
}

impl Default for AutoPickConfig {
    fn default() -> Self {
        Self {
            mode: AutoPickMode::Whitelist,
            whitelist: Vec::new(),
            blacklist: Vec::new(),
            cooldown: Duration::from_millis(500),
            allow_fuzzy: false,
            fuzzy_threshold: 0.75,
            min_confidence: 0.8,
            excluded_keywords: Vec::new(),
        }
    }
}

/// 交互决定及其可审计原因。
#[derive(Debug, Clone, PartialEq)]
pub enum AutoPickReason {
    WhitelistExact {
        item: String,
    },
    BlacklistAllowed {
        item: String,
    },
    WhitelistFuzzy {
        item: String,
        similarity: f64,
    },
    EmptyText,
    MultilineText,
    InvalidRoi,
    InvalidConfidence,
    InvalidMinConfidence,
    BelowMinConfidence {
        confidence: f64,
        minimum: f64,
    },
    ExcludedKeyword,
    SingleCharacter,
    Blacklisted {
        item: String,
    },
    FuzzyDisabled,
    InvalidFuzzyThreshold,
    FuzzyBelowThreshold {
        similarity: Option<f64>,
        threshold: f64,
    },
    Cooldown {
        remaining: Duration,
    },
}

impl AutoPickReason {
    pub fn is_interact(&self) -> bool {
        matches!(
            self,
            Self::WhitelistExact { .. }
                | Self::BlacklistAllowed { .. }
                | Self::WhitelistFuzzy { .. }
        )
    }
}

/// 纯决策结果。构造器私有，避免“是否交互”与原因不一致。
#[derive(Debug, Clone, PartialEq)]
pub struct AutoPickDecision {
    reason: AutoPickReason,
}

impl AutoPickDecision {
    pub fn should_interact(&self) -> bool {
        self.reason.is_interact()
    }

    pub fn reason(&self) -> &AutoPickReason {
        &self.reason
    }

    fn reject(reason: AutoPickReason) -> Self {
        Self { reason }
    }

    fn accept(reason: AutoPickReason) -> Self {
        Self { reason }
    }
}

// C# `AutoPickTrigger` presses the pick key on the first frame that matches, with no
// stability window (`:301-323`), so a single sample is the equivalent behaviour.
pub const AUTO_PICK_REQUIRED_STABLE_SAMPLES: usize = 1;
pub const AUTO_PICK_MAX_OBSERVATION_AGE: Duration = Duration::from_millis(750);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoPickTextRoi {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoPickPrompt {
    pub identity: u64,
    pub text_roi: AutoPickTextRoi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoPickGateState {
    Idle,
    Ready(u64),
    Serviced(u64),
}

#[derive(Debug, Clone, Default)]
pub struct AutoPickPromptGate {
    candidate: Option<(u64, usize)>,
    last_seen_at: Option<Instant>,
    serviced: BTreeSet<u64>,
}

impl AutoPickPromptGate {
    pub fn observe(&mut self, identity: Option<u64>, now: Instant) -> AutoPickGateState {
        let Some(identity) = identity else {
            self.candidate = None;
            self.last_seen_at = None;
            return AutoPickGateState::Idle;
        };
        if self.serviced.contains(&identity) {
            self.candidate = None;
            self.last_seen_at = Some(now);
            return AutoPickGateState::Serviced(identity);
        }
        let stale = self
            .last_seen_at
            .is_some_and(|last| now.saturating_duration_since(last) > Duration::from_millis(500));
        if stale {
            self.candidate = None;
        }
        let count = match self.candidate {
            Some((current, count)) if current == identity => count + 1,
            _ => 1,
        };
        self.candidate = Some((identity, count));
        self.last_seen_at = Some(now);
        if count >= AUTO_PICK_REQUIRED_STABLE_SAMPLES {
            AutoPickGateState::Ready(identity)
        } else {
            AutoPickGateState::Idle
        }
    }

    pub fn mark_serviced(&mut self, identity: u64) {
        if self
            .candidate
            .is_some_and(|(current, _)| current == identity)
        {
            self.serviced.insert(identity);
            self.candidate = None;
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// 可复用的自动拾取匹配器，仅保存最近一次允许交互的时间。
#[derive(Debug, Clone, Default)]
pub struct AutoPickMatcher {
    last_interact_at: Option<Instant>,
}

impl AutoPickMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// 判断一个 OCR 候选是否允许交互。
    ///
    /// 处理顺序为：输入校验、安全排除、精确黑名单、精确白名单、受阈值约束的模糊白名单、
    /// 冷却检查。只有最终允许交互时才刷新冷却。
    pub fn decide(
        &mut self,
        candidate: &OcrCandidate,
        config: &AutoPickConfig,
        now: Instant,
    ) -> AutoPickDecision {
        let text = match normalize_text(&candidate.text) {
            Ok(text) if text.is_empty() => {
                return AutoPickDecision::reject(AutoPickReason::EmptyText);
            }
            Ok(text) => text,
            Err(()) => {
                return AutoPickDecision::reject(AutoPickReason::MultilineText);
            }
        };

        let center = candidate.roi_center;
        if !center.x.is_finite()
            || !center.y.is_finite()
            || !(0.0..=1.0).contains(&center.x)
            || !(0.0..=1.0).contains(&center.y)
        {
            return AutoPickDecision::reject(AutoPickReason::InvalidRoi);
        }
        if !candidate.confidence.is_finite() || !(0.0..=1.0).contains(&candidate.confidence) {
            return AutoPickDecision::reject(AutoPickReason::InvalidConfidence);
        }
        if !config.min_confidence.is_finite() || !(0.0..=1.0).contains(&config.min_confidence) {
            return AutoPickDecision::reject(AutoPickReason::InvalidMinConfidence);
        }
        if candidate.confidence < config.min_confidence {
            return AutoPickDecision::reject(AutoPickReason::BelowMinConfidence {
                confidence: candidate.confidence,
                minimum: config.min_confidence,
            });
        }

        if is_excluded(&text, config) {
            return AutoPickDecision::reject(AutoPickReason::ExcludedKeyword);
        }
        if text.chars().count() == 1 && text != "盐" && text != "糖" {
            return AutoPickDecision::reject(AutoPickReason::SingleCharacter);
        }

        let whitelist = normalized_entries(&config.whitelist);
        let blacklist = normalized_entries(&config.blacklist);
        if blacklist.iter().any(|item| item == &text) {
            return AutoPickDecision::reject(AutoPickReason::Blacklisted { item: text });
        }

        let reason = match config.mode {
            AutoPickMode::Blacklist => AutoPickReason::BlacklistAllowed { item: text },
            AutoPickMode::Whitelist => match whitelist.iter().find(|item| *item == &text) {
                Some(item) => AutoPickReason::WhitelistExact { item: item.clone() },
                None => fuzzy_match(&text, &whitelist, &blacklist, config),
            },
        };
        if !reason.is_interact() {
            return AutoPickDecision { reason };
        }

        if let Some(last) = self.last_interact_at {
            let elapsed = now.saturating_duration_since(last);
            if elapsed < config.cooldown {
                return AutoPickDecision::reject(AutoPickReason::Cooldown {
                    remaining: config.cooldown - elapsed,
                });
            }
        }
        self.last_interact_at = Some(now);
        AutoPickDecision::accept(reason)
    }

    /// 清除交互冷却，例如任务停用或切换场景时调用。
    pub fn reset_cooldown(&mut self) {
        self.last_interact_at = None;
    }
}

fn normalize_text(text: &str) -> Result<String, ()> {
    if text.chars().any(|c| {
        matches!(
            c,
            '\r' | '\n' | '\u{000b}' | '\u{000c}' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
    }) {
        return Err(());
    }
    let mut normalized: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| match c {
            '[' => '「',
            ']' => '」',
            other => other,
        })
        .collect();
    if normalized.contains('「') && !normalized.contains('」') {
        normalized.push('」');
    } else if normalized.contains('」') && !normalized.contains('「') {
        normalized.insert(0, '「');
    }
    Ok(normalized)
}

fn normalized_entries(entries: &[String]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|entry| normalize_text(entry).ok())
        .filter(|entry| !entry.is_empty())
        .collect()
}

fn is_excluded(text: &str, config: &AutoPickConfig) -> bool {
    if text.eq_ignore_ascii_case("l") || text.eq_ignore_ascii_case("l键") {
        return true;
    }
    const REQUIRED_KEYWORDS: &[&str] = &["聊天", "设置"];
    REQUIRED_KEYWORDS
        .iter()
        .any(|keyword| text.contains(keyword))
        || normalized_entries(&config.excluded_keywords)
            .iter()
            .any(|keyword| text.contains(keyword))
}

/// 与 C# 单字通配符规则保持一致：只比较相同字数、最多一处替换的 OCR 结果。
/// 返回最高相似度；同分时保留白名单顺序。
fn fuzzy_match(
    text: &str,
    whitelist: &[String],
    blacklist: &[String],
    config: &AutoPickConfig,
) -> AutoPickReason {
    if !config.allow_fuzzy {
        return AutoPickReason::FuzzyDisabled;
    }
    if !config.fuzzy_threshold.is_finite() || !(0.0..=1.0).contains(&config.fuzzy_threshold) {
        return AutoPickReason::InvalidFuzzyThreshold;
    }

    let text_chars: Vec<char> = text.chars().collect();
    let mut best: Option<(usize, f64)> = None;
    for (index, item) in whitelist.iter().enumerate() {
        if blacklist.iter().any(|blocked| blocked == item) || is_excluded(item, config) {
            continue;
        }
        let item_chars: Vec<char> = item.chars().collect();
        if item_chars.len() != text_chars.len() || item_chars.len() < 2 {
            continue;
        }
        let differences = text_chars
            .iter()
            .zip(&item_chars)
            .filter(|(left, right)| left != right)
            .count();
        if differences > 1 {
            continue;
        }
        let similarity = 1.0 - differences as f64 / item_chars.len() as f64;
        if best.as_ref().is_none_or(|(best_index, best_score)| {
            similarity > *best_score || (similarity == *best_score && index < *best_index)
        }) {
            best = Some((index, similarity));
        }
    }

    match best {
        Some((index, similarity)) if similarity + f64::EPSILON >= config.fuzzy_threshold => {
            AutoPickReason::WhitelistFuzzy {
                item: whitelist[index].clone(),
                similarity,
            }
        }
        _ => AutoPickReason::FuzzyBelowThreshold {
            similarity: best.map(|(_, similarity)| similarity),
            threshold: config.fuzzy_threshold,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(text: &str) -> OcrCandidate {
        OcrCandidate {
            text: text.into(),
            roi_center: RoiCenter { x: 0.5, y: 0.2 },
            confidence: 0.95,
        }
    }

    fn config() -> AutoPickConfig {
        AutoPickConfig {
            mode: AutoPickMode::Whitelist,
            whitelist: vec![
                "甜甜花朵".into(),
                "甜甜菊朵".into(),
                "盐".into(),
                "糖".into(),
            ],
            blacklist: vec!["甜甜菊朵".into(), "陷阱".into()],
            cooldown: Duration::ZERO,
            allow_fuzzy: true,
            fuzzy_threshold: 0.75,
            min_confidence: 0.8,
            excluded_keywords: vec!["活动".into()],
        }
    }

    #[test]
    fn exact_whitelist_normalizes_whitespace_and_brackets() {
        let mut matcher = AutoPickMatcher::new();
        let mut config = config();
        config.whitelist.push("「琉璃百合」".into());
        let decision = matcher.decide(&candidate(" [琉璃百合] "), &config, Instant::now());
        assert!(decision.should_interact());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::WhitelistExact {
                item: "「琉璃百合」".into()
            }
        );
    }

    #[test]
    fn exact_whitelist_wins_over_an_earlier_fuzzy_candidate() {
        let mut matcher = AutoPickMatcher::new();
        let decision = matcher.decide(&candidate("甜甜菊朵"), &config(), Instant::now());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::Blacklisted {
                item: "甜甜菊朵".into()
            }
        );

        let mut exact_config = config();
        exact_config.blacklist.clear();
        let decision = matcher.decide(&candidate("甜甜菊朵"), &exact_config, Instant::now());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::WhitelistExact {
                item: "甜甜菊朵".into()
            }
        );
    }

    #[test]
    fn fuzzy_matching_requires_opt_in() {
        let mut matcher = AutoPickMatcher::new();
        let mut disabled = config();
        disabled.allow_fuzzy = false;
        let decision = matcher.decide(&candidate("甜甜桃朵"), &disabled, Instant::now());
        assert!(!decision.should_interact());
        assert_eq!(decision.reason(), &AutoPickReason::FuzzyDisabled);
    }

    #[test]
    fn fuzzy_match_uses_threshold_and_returns_canonical_name() {
        let mut matcher = AutoPickMatcher::new();
        let decision = matcher.decide(&candidate("甜甜桃朵"), &config(), Instant::now());
        assert!(decision.should_interact());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::WhitelistFuzzy {
                item: "甜甜花朵".into(),
                similarity: 0.75,
            }
        );

        let mut strict = config();
        strict.fuzzy_threshold = 0.8;
        let decision = matcher.decide(&candidate("甜甜桃朵"), &strict, Instant::now());
        assert!(!decision.should_interact());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::FuzzyBelowThreshold {
                similarity: Some(0.75),
                threshold: 0.8,
            }
        );
    }

    #[test]
    fn fuzzy_blacklist_and_dangerous_canonical_names_cannot_be_picked() {
        let mut matcher = AutoPickMatcher::new();
        let mut unsafe_config = config();
        unsafe_config.whitelist = vec!["甜甜菊朵".into(), "聊天工具".into(), "糖".into()];
        let decision = matcher.decide(&candidate("甜甜桃朵"), &unsafe_config, Instant::now());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::FuzzyBelowThreshold {
                similarity: None,
                threshold: 0.75,
            }
        );
        assert_eq!(
            matcher
                .decide(&candidate("甜甜菊朵"), &unsafe_config, Instant::now())
                .reason(),
            &AutoPickReason::Blacklisted {
                item: "甜甜菊朵".into()
            }
        );

        unsafe_config.blacklist.clear();
        let decision = matcher.decide(&candidate("聊天工具"), &unsafe_config, Instant::now());
        assert_eq!(decision.reason(), &AutoPickReason::ExcludedKeyword);
    }

    #[test]
    fn invalid_fuzzy_threshold_fails_closed_but_does_not_block_exact_match() {
        let mut matcher = AutoPickMatcher::new();
        for threshold in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut invalid = config();
            invalid.fuzzy_threshold = threshold;
            let decision = matcher.decide(&candidate("甜甜桃朵"), &invalid, Instant::now());
            assert!(!decision.should_interact());
            assert_eq!(decision.reason(), &AutoPickReason::InvalidFuzzyThreshold);

            let exact = matcher.decide(&candidate("甜甜花朵"), &invalid, Instant::now());
            assert!(exact.should_interact());
            assert!(matches!(
                exact.reason(),
                AutoPickReason::WhitelistExact { .. }
            ));
        }
    }

    #[test]
    fn empty_multiline_and_whitespace_text_are_rejected() {
        let mut matcher = AutoPickMatcher::new();
        for text in ["", " \t\u{3000}", "甜甜\n花", "甜甜\r花", "甜甜\u{2028}花"] {
            let decision = matcher.decide(&candidate(text), &config(), Instant::now());
            assert!(!decision.should_interact(), "accepted {text:?}");
        }
        assert_eq!(
            matcher
                .decide(&candidate(""), &config(), Instant::now())
                .reason(),
            &AutoPickReason::EmptyText
        );
        assert_eq!(
            matcher
                .decide(&candidate("甜甜\n花"), &config(), Instant::now())
                .reason(),
            &AutoPickReason::MultilineText
        );
    }

    #[test]
    fn non_finite_or_out_of_range_roi_is_rejected() {
        let mut matcher = AutoPickMatcher::new();
        for (x, y) in [
            (-0.01, 0.5),
            (1.01, 0.5),
            (0.5, -0.01),
            (0.5, 1.01),
            (f64::NAN, 0.5),
            (0.5, f64::INFINITY),
        ] {
            let mut invalid = candidate("甜甜花朵");
            invalid.roi_center = RoiCenter { x, y };
            assert_eq!(
                matcher.decide(&invalid, &config(), Instant::now()).reason(),
                &AutoPickReason::InvalidRoi
            );
        }
        let mut boundary = candidate("甜甜花朵");
        boundary.roi_center = RoiCenter { x: 0.0, y: 1.0 };
        assert!(
            matcher
                .decide(&boundary, &config(), Instant::now())
                .should_interact()
        );
    }

    #[test]
    fn non_finite_out_of_range_and_low_confidence_are_rejected() {
        let mut matcher = AutoPickMatcher::new();
        for confidence in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
            let mut invalid = candidate("甜甜花朵");
            invalid.confidence = confidence;
            assert_eq!(
                matcher.decide(&invalid, &config(), Instant::now()).reason(),
                &AutoPickReason::InvalidConfidence
            );
        }
        let mut low = candidate("甜甜花朵");
        low.confidence = 0.79;
        assert_eq!(
            matcher.decide(&low, &config(), Instant::now()).reason(),
            &AutoPickReason::BelowMinConfidence {
                confidence: 0.79,
                minimum: 0.8,
            }
        );

        let mut invalid_config = config();
        invalid_config.min_confidence = f64::NAN;
        assert_eq!(
            matcher
                .decide(&candidate("甜甜花朵"), &invalid_config, Instant::now())
                .reason(),
            &AutoPickReason::InvalidMinConfidence
        );
    }

    #[test]
    fn chat_settings_l_and_custom_keywords_are_always_excluded() {
        let mut matcher = AutoPickMatcher::new();
        let mut unsafe_config = config();
        unsafe_config.whitelist = vec!["聊天".into(), "设置".into(), "L".into(), "周年活动".into()];
        for text in ["聊天", "打开设置", "L", " l ", "L键", "周年活动"] {
            let decision = matcher.decide(&candidate(text), &unsafe_config, Instant::now());
            assert!(!decision.should_interact(), "accepted {text:?}");
            assert_eq!(decision.reason(), &AutoPickReason::ExcludedKeyword);
        }
    }

    #[test]
    fn single_character_guard_only_allows_salt_and_sugar() {
        let mut matcher = AutoPickMatcher::new();
        let mut single_config = config();
        single_config.whitelist.push("花".into());
        for text in ["花", "A", "🍇"] {
            assert_eq!(
                matcher
                    .decide(&candidate(text), &single_config, Instant::now())
                    .reason(),
                &AutoPickReason::SingleCharacter
            );
        }
        for text in ["盐", "糖"] {
            assert!(
                matcher
                    .decide(&candidate(text), &single_config, Instant::now())
                    .should_interact()
            );
        }
    }

    #[test]
    fn explicit_blacklist_mode_allows_unknown_items_but_not_exclusions() {
        let mut matcher = AutoPickMatcher::new();
        let mut blacklist_mode = config();
        blacklist_mode.mode = AutoPickMode::Blacklist;
        let allowed = matcher.decide(&candidate("未知材料"), &blacklist_mode, Instant::now());
        assert_eq!(
            allowed.reason(),
            &AutoPickReason::BlacklistAllowed {
                item: "未知材料".into()
            }
        );
        assert!(allowed.should_interact());
        assert_eq!(
            matcher
                .decide(&candidate("陷阱"), &blacklist_mode, Instant::now())
                .reason(),
            &AutoPickReason::Blacklisted {
                item: "陷阱".into()
            }
        );
    }

    #[test]
    fn empty_whitelist_default_is_safe_and_does_not_blindly_interact() {
        let mut matcher = AutoPickMatcher::new();
        let decision = matcher.decide(
            &candidate("陌生物品"),
            &AutoPickConfig::default(),
            Instant::now(),
        );
        assert!(!decision.should_interact());
        assert_eq!(decision.reason(), &AutoPickReason::FuzzyDisabled);
    }

    #[test]
    fn cooldown_blocks_repeats_until_boundary_and_rejections_do_not_extend_it() {
        let mut matcher = AutoPickMatcher::new();
        let mut cooling = config();
        cooling.cooldown = Duration::from_millis(500);
        let start = Instant::now();
        assert!(
            matcher
                .decide(&candidate("甜甜花朵"), &cooling, start)
                .should_interact()
        );

        let blocked = matcher.decide(
            &candidate("甜甜花朵"),
            &cooling,
            start + Duration::from_millis(200),
        );
        assert_eq!(
            blocked.reason(),
            &AutoPickReason::Cooldown {
                remaining: Duration::from_millis(300),
            }
        );
        let rejected = matcher.decide(
            &candidate("陷阱"),
            &cooling,
            start + Duration::from_millis(400),
        );
        assert!(!rejected.should_interact());
        assert!(
            matcher
                .decide(
                    &candidate("甜甜花朵"),
                    &cooling,
                    start + Duration::from_millis(500)
                )
                .should_interact()
        );

        matcher.reset_cooldown();
        assert!(
            matcher
                .decide(
                    &candidate("甜甜花朵"),
                    &cooling,
                    start + Duration::from_millis(500)
                )
                .should_interact()
        );
    }

    #[test]
    fn fuzzy_matching_counts_unicode_characters_not_bytes() {
        let mut matcher = AutoPickMatcher::new();
        let mut unicode = config();
        unicode.whitelist = vec!["琉璃百合".into()];
        let decision = matcher.decide(&candidate("琉璃百含"), &unicode, Instant::now());
        assert_eq!(
            decision.reason(),
            &AutoPickReason::WhitelistFuzzy {
                item: "琉璃百合".into(),
                similarity: 0.75,
            }
        );
    }

    #[test]
    fn prompt_gate_readies_on_the_first_positive_observation() {
        let mut gate = AutoPickPromptGate::default();
        let now = Instant::now();
        assert_eq!(gate.observe(None, now), AutoPickGateState::Idle);
        // C# acts on the first matching frame, so no second sample is required.
        assert_eq!(gate.observe(Some(7), now), AutoPickGateState::Ready(7));
    }

    #[test]
    fn prompt_gate_services_each_prompt_at_most_once() {
        let mut gate = AutoPickPromptGate::default();
        let now = Instant::now();
        assert_eq!(gate.observe(Some(7), now), AutoPickGateState::Ready(7));
        gate.mark_serviced(7);
        assert_eq!(
            gate.observe(Some(7), now + Duration::from_millis(100)),
            AutoPickGateState::Serviced(7)
        );
        assert_eq!(
            gate.observe(None, now + Duration::from_millis(200)),
            AutoPickGateState::Idle
        );
        // Clearing the prompt does not un-service an identity.
        assert_eq!(
            gate.observe(Some(7), now + Duration::from_millis(300)),
            AutoPickGateState::Serviced(7)
        );
        gate.reset();
        assert_eq!(
            gate.observe(Some(7), now + Duration::from_millis(400)),
            AutoPickGateState::Ready(7)
        );
    }

    #[test]
    fn servicing_another_prompt_does_not_rearm_an_earlier_identity() {
        let mut gate = AutoPickPromptGate::default();
        let now = Instant::now();
        assert_eq!(gate.observe(Some(7), now), AutoPickGateState::Ready(7));
        gate.mark_serviced(7);
        assert_eq!(
            gate.observe(Some(8), now + Duration::from_millis(200)),
            AutoPickGateState::Ready(8)
        );
        gate.mark_serviced(8);
        assert_eq!(
            gate.observe(Some(7), now + Duration::from_millis(400)),
            AutoPickGateState::Serviced(7)
        );
    }

    #[test]
    fn stale_gap_does_not_block_a_single_frame_hit() {
        let mut gate = AutoPickPromptGate::default();
        let now = Instant::now();
        assert_eq!(gate.observe(Some(7), now), AutoPickGateState::Ready(7));
        // A gap longer than the 500 ms stale window clears the candidate, but one frame is
        // already enough, so the next sighting is Ready again.
        assert_eq!(
            gate.observe(Some(7), now + Duration::from_secs(2)),
            AutoPickGateState::Ready(7)
        );
    }

    #[test]
    fn prompt_gate_reset_clears_serviced_identities() {
        let mut gate = AutoPickPromptGate::default();
        let now = Instant::now();
        assert_eq!(gate.observe(Some(1), now), AutoPickGateState::Ready(1));
        gate.mark_serviced(1);
        assert_eq!(
            gate.observe(Some(1), now + Duration::from_secs(1)),
            AutoPickGateState::Serviced(1)
        );
        gate.reset();
        assert_eq!(
            gate.observe(Some(1), now + Duration::from_secs(2)),
            AutoPickGateState::Ready(1)
        );
    }
}
