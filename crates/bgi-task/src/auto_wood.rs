use std::collections::HashSet;
use std::time::{Duration, Instant};

pub const AUTO_WOOD_TICK: Duration = Duration::from_millis(100);
// C# presses Z then waits 300 ms plus `AfterZSleepDelay` (default 0) before OCR (`:481`).
pub const AUTO_WOOD_AFTER_USE_WAIT: Duration = Duration::from_millis(300);
pub const AUTO_WOOD_GAIN_TIMEOUT: Duration = Duration::from_secs(30);
pub const AUTO_WOOD_TOTAL_TIMEOUT: Duration = Duration::from_secs(120);
pub const AUTO_WOOD_LOGIN_TIMEOUT: Duration = Duration::from_secs(90);
pub const AUTO_WOOD_MENU_TIMEOUT: Duration = Duration::from_secs(10);
pub const AUTO_WOOD_LOGOUT_CONFIRM_TIMEOUT: Duration = Duration::from_secs(15);
pub const AUTO_WOOD_EMPTY_OCR_LIMIT: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoWoodConfig {
    pub total_timeout_ms: u64,
    pub gain_timeout_ms: u64,
    pub login_timeout_ms: u64,
    pub menu_timeout_ms: u64,
    pub logout_confirm_timeout_ms: u64,
    pub empty_ocr_limit: u8,
}

impl Default for AutoWoodConfig {
    fn default() -> Self {
        Self {
            total_timeout_ms: AUTO_WOOD_TOTAL_TIMEOUT.as_millis() as u64,
            gain_timeout_ms: AUTO_WOOD_GAIN_TIMEOUT.as_millis() as u64,
            login_timeout_ms: AUTO_WOOD_LOGIN_TIMEOUT.as_millis() as u64,
            menu_timeout_ms: AUTO_WOOD_MENU_TIMEOUT.as_millis() as u64,
            logout_confirm_timeout_ms: AUTO_WOOD_LOGOUT_CONFIRM_TIMEOUT.as_millis() as u64,
            empty_ocr_limit: AUTO_WOOD_EMPTY_OCR_LIMIT,
        }
    }
}

impl AutoWoodConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1_000..=3_600_000).contains(&self.total_timeout_ms) {
            return Err("自动砍树总超时必须位于 1000..=3600000 ms");
        }
        if !(1_000..=600_000).contains(&self.gain_timeout_ms) {
            return Err("木材收益验证超时必须位于 1000..=600000 ms");
        }
        if !(1_000..=600_000).contains(&self.login_timeout_ms) {
            return Err("登录超时必须位于 1000..=600000 ms");
        }
        if !(1_000..=60_000).contains(&self.menu_timeout_ms) {
            return Err("菜单等待超时必须位于 1000..=60000 ms");
        }
        if !(1_000..=60_000).contains(&self.logout_confirm_timeout_ms) {
            return Err("退出确认超时必须位于 1000..=60000 ms");
        }
        if !(1..=10).contains(&self.empty_ocr_limit) {
            return Err("连续空 OCR 次数上限必须位于 1..=10");
        }
        Ok(())
    }

    pub const fn total_timeout(&self) -> Duration {
        Duration::from_millis(self.total_timeout_ms)
    }

    pub const fn gain_timeout(&self) -> Duration {
        Duration::from_millis(self.gain_timeout_ms)
    }

    pub const fn login_timeout(&self) -> Duration {
        Duration::from_millis(self.login_timeout_ms)
    }

    pub const fn menu_timeout(&self) -> Duration {
        Duration::from_millis(self.menu_timeout_ms)
    }

    pub const fn logout_confirm_timeout(&self) -> Duration {
        Duration::from_millis(self.logout_confirm_timeout_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoWoodAction {
    UseGadget,
    Wait,
    OpenMenu,
    ConfirmLogout((i32, i32)),
    Confirm((i32, i32)),
    WaitLogin,
    EnterGame((i32, i32)),
    FinishUsed,
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone, Default)]
pub struct AutoWoodOcrSample {
    /// Full OCR text is retained so the same reward popup is not counted twice.
    pub text: String,
    pub quantity: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct AutoWoodObservation {
    pub gadget: Option<(i32, i32, f64)>,
    pub menu: Option<(i32, i32, f64)>,
    pub confirm: Option<(i32, i32, f64)>,
    pub enter_game: Option<(i32, i32, f64)>,
    pub gameplay: bool,
    /// `None`: no OCR poll this tick; `Some(None)`: OCR returned no wood gain.
    pub ocr: Option<AutoWoodOcrSample>,
}

#[derive(Debug, Clone)]
enum Phase {
    CheckGadget,
    AfterUse {
        ready_at: Instant,
    },
    VerifyGain {
        started_at: Instant,
        empty_results: u8,
    },
    FinishUsed {
        ready_at: Instant,
    },
    WaitMenu {
        started_at: Instant,
    },
    WaitLogoutConfirm {
        started_at: Instant,
    },
    WaitLogin {
        started_at: Instant,
    },
    WaitGameplay {
        started_at: Instant,
    },
    Succeeded,
    Failed(&'static str),
}

#[derive(Debug, Clone)]
pub struct AutoWoodController {
    verify_gain: bool,
    config: AutoWoodConfig,
    phase: Phase,
    started_at: Option<Instant>,
    seen_rewards: HashSet<String>,
    wood_gain: u32,
}

impl AutoWoodController {
    pub fn new(verify_gain: bool) -> Self {
        Self::new_with_config(verify_gain, AutoWoodConfig::default())
    }

    pub fn new_with_config(verify_gain: bool, config: AutoWoodConfig) -> Self {
        Self::try_new(verify_gain, config).expect("invalid auto wood config")
    }

    pub fn try_new(verify_gain: bool, config: AutoWoodConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            verify_gain,
            config,
            phase: Phase::CheckGadget,
            started_at: None,
            seen_rewards: HashSet::new(),
            wood_gain: 0,
        })
    }

    pub const fn config(&self) -> AutoWoodConfig {
        self.config
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.phase, Phase::Succeeded | Phase::Failed(_))
    }

    pub fn succeeded(&self) -> bool {
        matches!(self.phase, Phase::Succeeded)
    }

    pub fn wood_gain(&self) -> u32 {
        self.wood_gain
    }

    pub fn stage(&self) -> &'static str {
        match self.phase {
            Phase::CheckGadget => "check_gadget",
            Phase::AfterUse { .. } => "after_use",
            Phase::VerifyGain { .. } => "verify_gain",
            Phase::FinishUsed { .. } => "finish_used",
            Phase::WaitMenu { .. } => "wait_menu",
            Phase::WaitLogoutConfirm { .. } => "wait_logout_confirm",
            Phase::WaitLogin { .. } => "wait_login",
            Phase::WaitGameplay { .. } => "wait_gameplay",
            Phase::Succeeded => "completed",
            Phase::Failed(_) => "failed",
        }
    }

    pub fn advance(&mut self, observation: AutoWoodObservation, now: Instant) -> AutoWoodAction {
        if self.is_finished() {
            return self.terminal_action();
        }
        let started_at = *self.started_at.get_or_insert(now);
        if now.saturating_duration_since(started_at) >= self.config.total_timeout() {
            return self.fail("自动砍树总超时");
        }

        match self.phase.clone() {
            Phase::CheckGadget => {
                if !observation.gameplay || observation.menu.is_some() {
                    return self.fail("开始前必须处于游戏画面，不能打开菜单或对话");
                }
                if observation.gadget.is_none() {
                    return self.fail("未检测到已装备的王树瑞佑，已停止且未按 Z");
                }
                self.phase = Phase::AfterUse {
                    ready_at: now + AUTO_WOOD_AFTER_USE_WAIT,
                };
                AutoWoodAction::UseGadget
            }
            Phase::AfterUse { ready_at } => {
                if !observation.gameplay || observation.menu.is_some() {
                    return self.fail("使用王树瑞佑后出现意外菜单或非游戏画面");
                }
                if now < ready_at {
                    return AutoWoodAction::Wait;
                }
                if self.verify_gain {
                    self.phase = Phase::VerifyGain {
                        started_at: now,
                        empty_results: 0,
                    };
                    AutoWoodAction::Wait
                } else {
                    self.phase = Phase::Succeeded;
                    AutoWoodAction::Succeeded
                }
            }
            Phase::VerifyGain {
                started_at,
                mut empty_results,
            } => {
                if !observation.gameplay || observation.menu.is_some() {
                    return self.fail("验证木材收益时出现意外菜单或非游戏画面");
                }
                if let Some(sample) = observation.ocr {
                    if sample.quantity.is_some() {
                        empty_results = 0;
                        match self.record_gain(sample) {
                            Ok(true) => {
                                self.phase = Phase::FinishUsed {
                                    ready_at: now + Duration::from_secs(1),
                                };
                                return AutoWoodAction::FinishUsed;
                            }
                            Ok(false) => {}
                            Err(reason) => return self.fail(reason),
                        }
                    } else {
                        empty_results = empty_results.saturating_add(1);
                        if empty_results >= self.config.empty_ocr_limit {
                            return self.fail("连续空 OCR 结果已达上限，未识别到木材收益");
                        }
                    }
                    self.phase = Phase::VerifyGain {
                        started_at,
                        empty_results,
                    };
                }
                if now.saturating_duration_since(started_at) >= self.config.gain_timeout() {
                    self.fail("木材收益验证超时")
                } else {
                    AutoWoodAction::Wait
                }
            }
            Phase::FinishUsed { ready_at } => {
                if !observation.gameplay || observation.menu.is_some() {
                    return self.fail("收益验证后出现意外菜单或非游戏画面");
                }
                if let Some(sample) = observation.ocr
                    && sample.quantity.is_some()
                {
                    match self.record_gain(sample) {
                        Ok(true) => {
                            self.phase = Phase::FinishUsed {
                                ready_at: now + Duration::from_secs(1),
                            };
                            return AutoWoodAction::FinishUsed;
                        }
                        Ok(false) => {}
                        Err(reason) => return self.fail(reason),
                    }
                }
                if now < ready_at {
                    AutoWoodAction::Wait
                } else {
                    self.phase = Phase::WaitMenu { started_at: now };
                    AutoWoodAction::OpenMenu
                }
            }
            Phase::WaitMenu { started_at } => {
                if let Some(confirm) = observation.confirm
                    && observation.menu.is_some()
                {
                    self.phase = Phase::WaitLogoutConfirm { started_at: now };
                    return AutoWoodAction::ConfirmLogout(center(confirm));
                }
                if now.saturating_duration_since(started_at) >= self.config.menu_timeout() {
                    self.fail("ESC 后未同时检测到背包菜单和退出按钮")
                } else {
                    AutoWoodAction::Wait
                }
            }
            Phase::WaitLogoutConfirm { started_at } => {
                if let Some(confirm) = observation.confirm
                    && observation.menu.is_none()
                {
                    self.phase = Phase::WaitLogin { started_at: now };
                    return AutoWoodAction::Confirm(center(confirm));
                }
                if now.saturating_duration_since(started_at) >= self.config.logout_confirm_timeout()
                {
                    self.fail("点击退出后未检测到退出确认按钮")
                } else {
                    AutoWoodAction::Wait
                }
            }
            Phase::WaitLogin { started_at } => {
                if observation.menu.is_some() {
                    return self.fail("登录等待期间出现意外菜单");
                }
                if let Some(enter_game) = observation.enter_game {
                    self.phase = Phase::WaitGameplay { started_at: now };
                    return AutoWoodAction::EnterGame(center(enter_game));
                }
                if now.saturating_duration_since(started_at) >= self.config.login_timeout() {
                    self.fail("登录等待超时，未检测到进入游戏按钮")
                } else {
                    AutoWoodAction::WaitLogin
                }
            }
            Phase::WaitGameplay { started_at } => {
                if observation.menu.is_some() {
                    return self.fail("重新进入游戏时出现意外菜单");
                }
                if observation.gameplay
                    && observation.gadget.is_some()
                    && observation.enter_game.is_none()
                {
                    self.phase = Phase::Succeeded;
                    AutoWoodAction::Succeeded
                } else if now.saturating_duration_since(started_at) >= self.config.login_timeout() {
                    self.fail("点击进入游戏后未回到游戏画面")
                } else {
                    AutoWoodAction::Wait
                }
            }
            Phase::Succeeded | Phase::Failed(_) => unreachable!("terminal state handled above"),
        }
    }

    fn record_gain(&mut self, sample: AutoWoodOcrSample) -> Result<bool, &'static str> {
        let Some(quantity) = sample.quantity else {
            return Ok(false);
        };
        let key = normalize_reward_text(&sample.text);
        if key.is_empty() || self.seen_rewards.contains(&key) {
            return Ok(false);
        }
        self.wood_gain = self
            .wood_gain
            .checked_add(quantity)
            .ok_or("木材收益累计数量溢出")?;
        self.seen_rewards.insert(key);
        Ok(true)
    }

    fn fail(&mut self, reason: &'static str) -> AutoWoodAction {
        self.phase = Phase::Failed(reason);
        AutoWoodAction::Failed(reason)
    }

    fn terminal_action(&self) -> AutoWoodAction {
        match self.phase {
            Phase::Succeeded => AutoWoodAction::Succeeded,
            Phase::Failed(reason) => AutoWoodAction::Failed(reason),
            _ => unreachable!("terminal_action called for a non-terminal state"),
        }
    }
}

fn center(hit: (i32, i32, f64)) -> (i32, i32) {
    (hit.0, hit.1)
}

/// Parses only the literal whitelisted item `木材`, never arbitrary material names.
/// Accepts `木材×N`, `木材xN`, `木材X N`, full-width digits/spaces, and multiple lines.
pub fn parse_wood_gain(text: &str) -> Option<u32> {
    let normalized: Vec<char> = text.chars().map(normalize_char).collect();
    let needle = ['木', '材'];
    let mut total: Option<u32> = None;
    let mut seen_lines = HashSet::new();
    let mut index = 0;
    while index + needle.len() <= normalized.len() {
        if normalized[index..index + needle.len()] != needle {
            index += 1;
            continue;
        }
        let reward_start = index;
        index += needle.len();
        while index < normalized.len() && normalized[index].is_whitespace() {
            index += 1;
        }
        if index >= normalized.len() || !matches!(normalized[index], '×' | 'x' | 'X' | ':' | '：')
        {
            continue;
        }
        index += 1;
        while index < normalized.len() && is_space(normalized[index]) {
            index += 1;
        }
        let mut quantity = 0u32;
        let mut digits = 0;
        while index < normalized.len() && normalized[index].is_ascii_digit() {
            quantity = quantity
                .checked_mul(10)?
                .checked_add(normalized[index].to_digit(10)?)?;
            index += 1;
            digits += 1;
        }
        if digits > 0 && quantity > 0 {
            let key: String = normalized[reward_start..index]
                .iter()
                .filter(|c| {
                    let c = normalize_char(**c);
                    !is_space(c) && !matches!(c, ':' | '：' | '·' | '×' | 'x')
                })
                .collect();
            if seen_lines.insert(key) {
                total = Some(total.unwrap_or(0).checked_add(quantity)?);
            }
        }
    }
    total
}

fn normalize_char(c: char) -> char {
    match c {
        '０'..='９' => char::from(b'0' + (c as u32 - '０' as u32) as u8),
        '\u{3000}' | '\u{00a0}' => ' ',
        'X' => 'x',
        other => other,
    }
}

fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, ':' | '：' | '·')
}

fn normalize_reward_text(text: &str) -> String {
    text.chars()
        .map(normalize_char)
        .filter(|c| !is_space(*c) && !matches!(*c, ':' | '：' | '·' | '×' | 'x'))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;

    fn at(milliseconds: u64) -> Instant {
        static BASE: OnceLock<Instant> = OnceLock::new();
        *BASE.get_or_init(Instant::now) + Duration::from_millis(milliseconds)
    }

    fn observation() -> AutoWoodObservation {
        AutoWoodObservation {
            gadget: Some((1600, 700, 0.9)),
            gameplay: true,
            ..Default::default()
        }
    }

    fn sample(text: &str, quantity: Option<u32>) -> AutoWoodObservation {
        AutoWoodObservation {
            ocr: Some(AutoWoodOcrSample {
                text: text.into(),
                quantity,
            }),
            ..observation()
        }
    }

    #[test]
    fn config_defaults_and_boundaries_are_validated() {
        let default = AutoWoodConfig::default();
        assert_eq!(default.total_timeout(), AUTO_WOOD_TOTAL_TIMEOUT);
        assert_eq!(default.gain_timeout(), AUTO_WOOD_GAIN_TIMEOUT);
        assert_eq!(default.login_timeout(), AUTO_WOOD_LOGIN_TIMEOUT);
        assert_eq!(default.menu_timeout(), AUTO_WOOD_MENU_TIMEOUT);
        assert_eq!(
            default.logout_confirm_timeout(),
            AUTO_WOOD_LOGOUT_CONFIRM_TIMEOUT
        );
        assert_eq!(default.empty_ocr_limit, AUTO_WOOD_EMPTY_OCR_LIMIT);
        assert_eq!(default.validate(), Ok(()));
        assert_eq!(AutoWoodController::new(true).config(), default);
        assert_eq!(AutoWoodController::new(false).config(), default);

        for config in [
            AutoWoodConfig {
                total_timeout_ms: 1_000,
                gain_timeout_ms: 1_000,
                login_timeout_ms: 1_000,
                menu_timeout_ms: 1_000,
                logout_confirm_timeout_ms: 1_000,
                empty_ocr_limit: 1,
            },
            AutoWoodConfig {
                total_timeout_ms: 3_600_000,
                gain_timeout_ms: 600_000,
                login_timeout_ms: 600_000,
                menu_timeout_ms: 60_000,
                logout_confirm_timeout_ms: 60_000,
                empty_ocr_limit: 10,
            },
        ] {
            assert_eq!(config.validate(), Ok(()));
            assert!(AutoWoodController::try_new(true, config).is_ok());
            assert_eq!(
                AutoWoodController::new_with_config(false, config).config(),
                config
            );
        }

        let invalid = [
            AutoWoodConfig {
                total_timeout_ms: 999,
                ..default
            },
            AutoWoodConfig {
                total_timeout_ms: 3_600_001,
                ..default
            },
            AutoWoodConfig {
                gain_timeout_ms: 999,
                ..default
            },
            AutoWoodConfig {
                gain_timeout_ms: 600_001,
                ..default
            },
            AutoWoodConfig {
                login_timeout_ms: 999,
                ..default
            },
            AutoWoodConfig {
                login_timeout_ms: 600_001,
                ..default
            },
            AutoWoodConfig {
                menu_timeout_ms: 999,
                ..default
            },
            AutoWoodConfig {
                menu_timeout_ms: 60_001,
                ..default
            },
            AutoWoodConfig {
                logout_confirm_timeout_ms: 999,
                ..default
            },
            AutoWoodConfig {
                logout_confirm_timeout_ms: 60_001,
                ..default
            },
            AutoWoodConfig {
                empty_ocr_limit: 0,
                ..default
            },
            AutoWoodConfig {
                empty_ocr_limit: 11,
                ..default
            },
        ];
        for config in invalid {
            assert!(config.validate().is_err());
            assert!(AutoWoodController::try_new(true, config).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "invalid auto wood config")]
    fn invalid_config_is_rejected_by_the_infallible_constructor() {
        AutoWoodController::new_with_config(
            true,
            AutoWoodConfig {
                empty_ocr_limit: 0,
                ..AutoWoodConfig::default()
            },
        );
    }

    #[test]
    fn custom_total_timeout_prevents_further_input() {
        let config = AutoWoodConfig {
            total_timeout_ms: 1_000,
            ..AutoWoodConfig::default()
        };
        let mut controller = AutoWoodController::new_with_config(true, config);
        assert_eq!(
            controller.advance(observation(), at(999)),
            AutoWoodAction::UseGadget
        );
        assert_eq!(
            controller.advance(observation(), at(1_999)),
            AutoWoodAction::Failed("自动砍树总超时")
        );
        assert_eq!(
            controller.advance(observation(), at(2_000)),
            AutoWoodAction::Failed("自动砍树总超时")
        );
    }

    #[test]
    fn custom_gain_timeout_and_empty_ocr_limit_are_enforced() {
        let config = AutoWoodConfig {
            gain_timeout_ms: 1_000,
            empty_ocr_limit: 1,
            ..AutoWoodConfig::default()
        };
        let mut gain_timeout = AutoWoodController::new_with_config(true, config);
        gain_timeout.advance(observation(), at(0));
        gain_timeout.advance(observation(), at(8_000));
        assert_eq!(
            gain_timeout.advance(observation(), at(8_999)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            gain_timeout.advance(observation(), at(9_000)),
            AutoWoodAction::Failed("木材收益验证超时")
        );

        let mut empty_ocr = AutoWoodController::new_with_config(true, config);
        empty_ocr.advance(observation(), at(0));
        empty_ocr.advance(observation(), at(8_000));
        assert_eq!(
            empty_ocr.advance(sample("", None), at(8_100)),
            AutoWoodAction::Failed("连续空 OCR 结果已达上限，未识别到木材收益")
        );
    }

    #[test]
    fn parses_only_whitelisted_wood_and_common_separators() {
        for text in [
            "获得\n木材×30",
            "木材x12",
            "木材X 7",
            "获得 木材 ： ５",
            "木材×2\n木材x3",
        ] {
            assert!(parse_wood_gain(text).is_some(), "{text}");
        }
        assert_eq!(parse_wood_gain("获得\n木材×30"), Some(30));
        assert_eq!(parse_wood_gain("获得 木材 ： ５"), Some(5));
        assert_eq!(parse_wood_gain("木材×2\n木材x3"), Some(5));
        assert_eq!(parse_wood_gain("木材×2\n木材 X 2"), Some(2));
        assert_eq!(parse_wood_gain("铁矿×99\n木材×4"), Some(4));
        for text in [
            "",
            "获得",
            "木材",
            "木材×",
            "木材×0",
            "竹节×30",
            "木材x999999999999",
        ] {
            assert_eq!(parse_wood_gain(text), None, "{text}");
        }
    }

    #[test]
    fn missing_gadget_fails_without_use_and_unexpected_menu_is_rejected() {
        let mut controller = AutoWoodController::new(true);
        let mut missing = observation();
        missing.gadget = None;
        assert_eq!(
            controller.advance(missing, at(0)),
            AutoWoodAction::Failed("未检测到已装备的王树瑞佑，已停止且未按 Z")
        );
        let mut controller = AutoWoodController::new(true);
        let mut menu = observation();
        menu.gameplay = false;
        menu.menu = Some((50, 50, 0.9));
        assert!(matches!(
            controller.advance(menu, at(0)),
            AutoWoodAction::Failed(_)
        ));
    }

    #[test]
    fn disabled_verification_waits_briefly_and_never_logs_out() {
        let mut controller = AutoWoodController::new(false);
        assert_eq!(
            controller.advance(observation(), at(0)),
            AutoWoodAction::UseGadget
        );
        assert_eq!(
            controller.advance(observation(), at(299)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(sample("木材×3", Some(3)), at(300)),
            AutoWoodAction::Succeeded
        );
        assert_eq!(controller.wood_gain(), 0);
        assert!(controller.succeeded());
        assert_eq!(
            controller.advance(observation(), at(120_000)),
            AutoWoodAction::Succeeded
        );
    }

    #[test]
    fn verification_deduplicates_repeated_reward_text_and_accumulates_distinct_gains() {
        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        assert_eq!(
            controller.advance(sample("获得 木材×3", Some(3)), at(8_100)),
            AutoWoodAction::FinishUsed
        );
        assert_eq!(controller.wood_gain(), 3);
        assert_eq!(
            controller.advance(sample("获得木材×3", Some(3)), at(8_200)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(sample("获得 木材×2", Some(2)), at(8_300)),
            AutoWoodAction::FinishUsed
        );
        assert_eq!(controller.wood_gain(), 5);
        // A short settling window captures distinct popup lines, then logout starts.
        assert_eq!(
            controller.advance(observation(), at(9_299)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(observation(), at(9_300)),
            AutoWoodAction::OpenMenu
        );
    }

    #[test]
    fn three_empty_ocr_results_fail_but_skipped_ticks_do_not_count() {
        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        for count in 1..3 {
            assert_eq!(
                controller.advance(sample("", None), at(8_000 + count * 100)),
                AutoWoodAction::Wait
            );
        }
        assert_eq!(
            controller.advance(sample("", None), at(8_300)),
            AutoWoodAction::Failed("连续空 OCR 结果已达上限，未识别到木材收益")
        );
        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        for elapsed in [8_100, 8_200, 8_300, 8_400] {
            assert_eq!(
                controller.advance(observation(), at(elapsed)),
                AutoWoodAction::Wait
            );
        }
    }

    #[test]
    fn verified_run_requires_menu_before_logout_confirmation_before_login_click() {
        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        controller.advance(sample("木材×4", Some(4)), at(8_100));
        assert_eq!(
            controller.advance(observation(), at(9_100)),
            AutoWoodAction::OpenMenu
        );
        let confirm_only = AutoWoodObservation {
            confirm: Some((100, 100, 0.9)),
            ..observation()
        };
        assert_eq!(
            controller.advance(confirm_only, at(9_200)),
            AutoWoodAction::Wait
        );
        let menu_and_confirm = AutoWoodObservation {
            menu: Some((100, 100, 0.9)),
            confirm: Some((80, 1000, 0.9)),
            gameplay: false,
            ..Default::default()
        };
        assert_eq!(
            controller.advance(menu_and_confirm, at(9_300)),
            AutoWoodAction::ConfirmLogout((80, 1000))
        );
        let dialog = AutoWoodObservation {
            confirm: Some((960, 600, 0.9)),
            gameplay: false,
            ..Default::default()
        };
        assert_eq!(
            controller.advance(dialog, at(9_400)),
            AutoWoodAction::Confirm((960, 600))
        );
        assert_eq!(
            controller.advance(AutoWoodObservation::default(), at(9_500)),
            AutoWoodAction::WaitLogin
        );
        let login = AutoWoodObservation {
            enter_game: Some((960, 800, 0.9)),
            ..Default::default()
        };
        assert_eq!(
            controller.advance(login, at(9_600)),
            AutoWoodAction::EnterGame((960, 800))
        );
        assert_eq!(
            controller.advance(AutoWoodObservation::default(), at(99_599)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(observation(), at(99_600)),
            AutoWoodAction::Succeeded
        );
        assert_eq!(controller.wood_gain(), 4);
    }

    #[test]
    fn phase_timeouts_and_total_timeout_cannot_be_overridden_by_success() {
        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        assert!(matches!(
            controller.advance(sample("", None), at(38_000)),
            AutoWoodAction::Failed("木材收益验证超时")
        ));
        assert_eq!(
            controller.advance(sample("木材×9", Some(9)), at(38_001)),
            AutoWoodAction::Failed("木材收益验证超时")
        );

        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        controller.advance(sample("木材×1", Some(1)), at(8_100));
        controller.advance(observation(), at(9_100));
        assert_eq!(
            controller.advance(observation(), at(120_000)),
            AutoWoodAction::Failed("自动砍树总超时")
        );
    }

    fn verified_wait_menu() -> AutoWoodController {
        verified_wait_menu_with_config(AutoWoodConfig::default())
    }

    fn verified_wait_menu_with_config(config: AutoWoodConfig) -> AutoWoodController {
        let mut controller = AutoWoodController::new_with_config(true, config);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        controller.advance(sample("木材×1", Some(1)), at(8_100));
        assert_eq!(
            controller.advance(observation(), at(9_100)),
            AutoWoodAction::OpenMenu
        );
        controller
    }

    #[test]
    fn custom_menu_logout_and_login_timeouts_are_independent() {
        let config = AutoWoodConfig {
            total_timeout_ms: 60_000,
            gain_timeout_ms: 30_000,
            login_timeout_ms: 2_000,
            menu_timeout_ms: 1_000,
            logout_confirm_timeout_ms: 1_000,
            empty_ocr_limit: 3,
        };

        let mut menu = verified_wait_menu_with_config(config);
        assert_eq!(
            menu.advance(observation(), at(10_099)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            menu.advance(observation(), at(10_100)),
            AutoWoodAction::Failed("ESC 后未同时检测到背包菜单和退出按钮")
        );

        let menu_and_confirm = AutoWoodObservation {
            menu: Some((1, 1, 0.9)),
            confirm: Some((2, 2, 0.9)),
            ..Default::default()
        };
        let mut logout = verified_wait_menu_with_config(config);
        assert_eq!(
            logout.advance(menu_and_confirm.clone(), at(9_200)),
            AutoWoodAction::ConfirmLogout((2, 2))
        );
        assert_eq!(
            logout.advance(menu_and_confirm.clone(), at(10_199)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            logout.advance(menu_and_confirm.clone(), at(10_200)),
            AutoWoodAction::Failed("点击退出后未检测到退出确认按钮")
        );

        let dialog = AutoWoodObservation {
            confirm: Some((2, 2, 0.9)),
            ..Default::default()
        };
        let mut login = verified_wait_menu_with_config(config);
        login.advance(menu_and_confirm.clone(), at(9_200));
        assert_eq!(
            login.advance(dialog.clone(), at(9_300)),
            AutoWoodAction::Confirm((2, 2))
        );
        assert_eq!(
            login.advance(AutoWoodObservation::default(), at(11_299)),
            AutoWoodAction::WaitLogin
        );
        assert_eq!(
            login.advance(AutoWoodObservation::default(), at(11_300)),
            AutoWoodAction::Failed("登录等待超时，未检测到进入游戏按钮")
        );

        let mut gameplay = verified_wait_menu_with_config(config);
        gameplay.advance(menu_and_confirm, at(9_200));
        gameplay.advance(dialog, at(9_300));
        assert_eq!(
            gameplay.advance(
                AutoWoodObservation {
                    enter_game: Some((960, 800, 0.9)),
                    ..Default::default()
                },
                at(9_400)
            ),
            AutoWoodAction::EnterGame((960, 800))
        );
        let loading = AutoWoodObservation {
            gameplay: true,
            ..Default::default()
        };
        assert_eq!(
            gameplay.advance(loading.clone(), at(11_399)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            gameplay.advance(loading, at(11_400)),
            AutoWoodAction::Failed("点击进入游戏后未回到游戏画面")
        );
    }

    #[test]
    fn menu_and_logout_confirmation_have_independent_timeouts() {
        let mut controller = verified_wait_menu();
        assert_eq!(
            controller.advance(observation(), at(19_099)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(observation(), at(19_100)),
            AutoWoodAction::Failed("ESC 后未同时检测到背包菜单和退出按钮")
        );

        let mut controller = verified_wait_menu();
        controller.advance(
            AutoWoodObservation {
                menu: Some((1, 1, 0.9)),
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_200),
        );
        let menu_still_visible = AutoWoodObservation {
            menu: Some((1, 1, 0.9)),
            confirm: Some((2, 2, 0.9)),
            ..Default::default()
        };
        assert_eq!(
            controller.advance(menu_still_visible.clone(), at(24_199)),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(menu_still_visible, at(24_200)),
            AutoWoodAction::Failed("点击退出后未检测到退出确认按钮")
        );
    }

    #[test]
    fn relogin_requires_positive_gameplay_evidence_and_times_out_independently() {
        let mut controller = verified_wait_menu();
        controller.advance(
            AutoWoodObservation {
                menu: Some((1, 1, 0.9)),
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_200),
        );
        controller.advance(
            AutoWoodObservation {
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_300),
        );
        controller.advance(
            AutoWoodObservation {
                enter_game: Some((960, 800, 0.9)),
                ..Default::default()
            },
            at(9_400),
        );
        // UiScreens alone treats an unknown loading screen as gameplay; the gadget icon is positive evidence.
        assert_eq!(
            controller.advance(observation(), at(99_399)),
            AutoWoodAction::Succeeded
        );
        let mut controller = verified_wait_menu();
        controller.advance(
            AutoWoodObservation {
                menu: Some((1, 1, 0.9)),
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_200),
        );
        controller.advance(
            AutoWoodObservation {
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_300),
        );
        controller.advance(
            AutoWoodObservation {
                enter_game: Some((960, 800, 0.9)),
                ..Default::default()
            },
            at(9_400),
        );
        assert_eq!(
            controller.advance(
                AutoWoodObservation {
                    gameplay: true,
                    ..Default::default()
                },
                at(99_399)
            ),
            AutoWoodAction::Wait
        );
        assert_eq!(
            controller.advance(
                AutoWoodObservation {
                    gameplay: true,
                    ..Default::default()
                },
                at(99_400)
            ),
            AutoWoodAction::Failed("点击进入游戏后未回到游戏画面")
        );
    }

    #[test]
    fn login_button_timeout_is_independent_and_bounded() {
        let mut controller = AutoWoodController::new(true);
        controller.advance(observation(), at(0));
        controller.advance(observation(), at(8_000));
        controller.advance(sample("木材×1", Some(1)), at(8_100));
        controller.advance(observation(), at(9_100));
        controller.advance(
            AutoWoodObservation {
                menu: Some((1, 1, 0.9)),
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_200),
        );
        controller.advance(
            AutoWoodObservation {
                confirm: Some((2, 2, 0.9)),
                ..Default::default()
            },
            at(9_300),
        );
        assert_eq!(
            controller.advance(AutoWoodObservation::default(), at(99_299)),
            AutoWoodAction::WaitLogin
        );
        assert_eq!(
            controller.advance(AutoWoodObservation::default(), at(99_300)),
            AutoWoodAction::Failed("登录等待超时，未检测到进入游戏按钮")
        );
    }
}
