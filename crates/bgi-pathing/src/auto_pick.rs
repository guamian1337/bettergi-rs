use std::time::Instant;

use bgi_task::{
    AUTO_PICK_MAX_OBSERVATION_AGE, AutoPickConfig, AutoPickGateState, AutoPickMatcher,
    AutoPickPrompt, AutoPickPromptGate, OcrCandidate, RoiCenter,
};

use crate::OcrHit;
use crate::executor::PathingHost;

#[derive(Debug, Clone)]
pub struct AutoPickFrame {
    pub width: u32,
    pub height: u32,
    pub prompt: Option<AutoPickPrompt>,
    pub ocr_hits: Vec<OcrHit>,
    pub observed_at: Instant,
}

pub fn pick_once<H: PathingHost + ?Sized>(
    host: &mut H,
    matcher: &mut AutoPickMatcher,
    gate: &mut AutoPickPromptGate,
    config: &AutoPickConfig,
) -> Result<usize, String> {
    let Some(frame) = host.capture_auto_pick()? else {
        gate.observe(None, Instant::now());
        return Ok(0);
    };
    let now = Instant::now();
    if frame.observed_at.elapsed() > AUTO_PICK_MAX_OBSERVATION_AGE
        || frame.width == 0
        || frame.height == 0
    {
        gate.observe(None, now);
        return Ok(0);
    }
    let identity = frame.prompt.as_ref().map(|prompt| prompt.identity);
    if !matches!(gate.observe(identity, now), AutoPickGateState::Ready(_)) {
        return Ok(0);
    }
    for hit in frame.ocr_hits {
        if host.is_cancelled() {
            break;
        }
        let candidate = OcrCandidate {
            text: hit.text,
            confidence: f64::from(hit.confidence),
            roi_center: RoiCenter {
                x: f64::from(hit.center_x.clamp(0, frame.width as i32 - 1))
                    / f64::from(frame.width),
                y: f64::from(hit.center_y.clamp(0, frame.height as i32 - 1))
                    / f64::from(frame.height),
            },
        };
        if matcher
            .decide(&candidate, config, Instant::now())
            .should_interact()
        {
            if let Err(error) =
                host.key_tap_checked(0x46, 80, (frame.width, frame.height), frame.observed_at)
            {
                host.on_log(
                    bgi_control::LogLevel::Debug,
                    format!("自动拾取未输入: {error}"),
                );
                return Ok(0);
            }
            if let Some(identity) = identity {
                gate.mark_serviced(identity);
            }
            host.sleep_ms(160);
            return Ok(1);
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::tests::MockHost;
    use bgi_task::{AutoPickMode, AutoPickTextRoi};
    use std::time::Duration;

    fn frame(prompt: bool, hits: Vec<OcrHit>) -> AutoPickFrame {
        AutoPickFrame {
            width: 1280,
            height: 720,
            prompt: prompt.then_some(AutoPickPrompt {
                identity: 7,
                text_roi: AutoPickTextRoi {
                    x0: 0.6,
                    y0: 0.3,
                    x1: 0.8,
                    y1: 0.4,
                },
            }),
            ocr_hits: hits,
            observed_at: Instant::now(),
        }
    }

    fn hit(text: &str) -> OcrHit {
        OcrHit {
            text: text.into(),
            center_x: 800,
            center_y: 250,
            confidence: 1.0,
        }
    }

    #[test]
    fn only_two_stable_positive_prompts_and_whitelisted_text_cause_input() {
        let mut host = MockHost::new((0.0, 0.0));
        let mut gate = AutoPickPromptGate::default();
        let mut matcher = AutoPickMatcher::new();
        let config = AutoPickConfig {
            whitelist: vec!["甜甜花".into()],
            cooldown: Duration::ZERO,
            ..Default::default()
        };
        host.auto_pick_frame = Some(frame(true, vec![hit("聊天"), hit("甜甜花")]));
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            0
        );
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            1
        );
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            0
        );
        assert_eq!(
            host.events
                .iter()
                .filter(|event| event.as_str() == "press_F")
                .count(),
            1
        );
    }

    #[test]
    fn missing_prompt_never_uses_ocr_hits_as_positive_evidence() {
        let mut host = MockHost::new((0.0, 0.0));
        let mut gate = AutoPickPromptGate::default();
        let mut matcher = AutoPickMatcher::new();
        let config = AutoPickConfig {
            whitelist: vec!["甜甜花".into()],
            cooldown: Duration::ZERO,
            ..Default::default()
        };
        host.auto_pick_frame = Some(frame(false, vec![hit("甜甜花")]));
        for _ in 0..3 {
            assert_eq!(
                pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
                0
            );
        }
        assert!(!host.events.iter().any(|event| event == "press_F"));
    }

    #[test]
    fn blacklist_mode_still_excludes_chat_after_prompt_gate() {
        let mut host = MockHost::new((0.0, 0.0));
        let mut gate = AutoPickPromptGate::default();
        let mut matcher = AutoPickMatcher::new();
        let config = AutoPickConfig {
            mode: AutoPickMode::Blacklist,
            cooldown: Duration::ZERO,
            ..Default::default()
        };
        host.auto_pick_frame = Some(frame(true, vec![hit("聊天"), hit("未知材料")]));
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            0
        );
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            1
        );
    }

    #[test]
    fn stale_observation_and_failed_checked_input_never_press() {
        let mut host = MockHost::new((0.0, 0.0));
        host.checked_input_error = Some("窗口失焦".into());
        let mut gate = AutoPickPromptGate::default();
        let mut matcher = AutoPickMatcher::new();
        let config = AutoPickConfig::default();
        host.auto_pick_frame = Some(frame(true, vec![hit("甜甜花")]));
        let config = AutoPickConfig {
            whitelist: vec!["甜甜花".into()],
            ..config
        };
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            0
        );
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            0
        );
        assert!(!host.events.iter().any(|event| event == "press_F"));
        host.checked_input_error = None;
        host.auto_pick_frame.as_mut().unwrap().observed_at =
            Instant::now() - AUTO_PICK_MAX_OBSERVATION_AGE - Duration::from_millis(1);
        assert_eq!(
            pick_once(&mut host, &mut matcher, &mut gate, &config).unwrap(),
            0
        );
        assert!(!host.events.iter().any(|event| event == "press_F"));
    }
}
