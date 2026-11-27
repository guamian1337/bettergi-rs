use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use bgi_capture::{FrameBufferPool, enumerate_candidate_windows};
use bgi_control::{
    CoreCommand, CoreEndpoint, CoreEvent, CvDebugResult, HotkeyAction, LogLevel, TaskLimitsConfig,
    TaskOutcome, TaskReport, TaskState, UiSnapshot,
};
use bgi_input::{RawInputHub, RawInputSignal};
use bgi_pathing::{LoopPolicy, LoopRunner, PathingHost};
use bgi_task::{
    AUTO_OPEN_CHEST_TICK, AUTO_WOOD_TICK, AutoCookTaskConfig, AutoOpenChestAction,
    AutoOpenChestConfig, AutoOpenChestController, AutoWoodAction, AutoWoodConfig,
    AutoWoodController, AutoWoodObservation as AutoWoodTaskObservation, AutoWoodOcrSample,
    CHARACTER_ATTRIBUTE_MAX_OCR_ATTEMPTS, CharacterAttributeRecognitionStability,
    CharacterAttributeRecognitionState, CookingBarConfig, CookingBarDetector, CookingBarEvent,
    EXPEDITION_MAX_ATTEMPTS, ExpeditionRecognitionStability, ExpeditionRecognitionState,
    FishingConfig, MINING_MAX_ATTEMPTS, MiningRecognitionStability, MiningRecognitionState,
    ONE_KEY_EXPEDITION_TICK, OneKeyExpeditionAction, OneKeyExpeditionConfig,
    OneKeyExpeditionController, OneKeyExpeditionTarget, REWARD_RESULT_MAX_ATTEMPTS,
    RewardClaimAction, RewardClaimConfig, RewardClaimController, RewardClaimFinish,
    RewardClaimMode, RewardKind, RewardResultRecognitionStability, RewardResultRecognitionState,
    TALENT_MAX_OCR_ATTEMPTS, TalentRecognitionStability, TalentRecognitionState,
    WEAPON_MAX_OCR_ATTEMPTS, WeaponRecognitionStability, WeaponRecognitionState, parse_wood_gain,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::capture_host::{
    ActiveCapturer, CapturePaceState, capture_once, capture_raw_frame, close_capturer,
    release_idle_capturer,
};
use crate::cv_service::CvOcrServices;
use crate::fishing_host::FishingHost;
use crate::mining_host::{MINING_TASK_NAME, MiningHost, MiningRunConfig};
use crate::pathing_host::{CorePathingHost, PathingEvent};
use crate::task_runtime::{ActiveTask, allocate_task_id};

fn ensure_task_inactive(snapshot: &UiSnapshot) -> Result<(), String> {
    if snapshot.task_state.is_task_active() {
        return Err("寻路任务运行或停止期间不能更改抓图/识别会话".to_string());
    }
    Ok(())
}

async fn run_cv_probe<F>(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
    op: F,
) -> Result<CvDebugResult, String>
where
    F: FnOnce(&mut CvOcrServices, &[u8], u32, u32, bool, u64) -> Result<CvDebugResult, String>
        + Send
        + 'static,
{
    ensure_task_inactive(snapshot)?;
    let captured = {
        let mut cap_guard = capturer.lock().unwrap();
        let res = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
        release_idle_capturer(snapshot, &mut cap_guard, core);
        res
    };
    let Some(frame) = captured? else {
        return Err("当前帧尚未就绪".to_string());
    };
    *dbg_seq += 1;
    let seq = *dbg_seq;
    let cv = cv_ocr.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = cv.lock().unwrap();
        op(
            &mut guard,
            frame.data.as_slice(),
            frame.width,
            frame.height,
            frame.format.is_bgr24(),
            seq,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

const WEAPON_RETRY_DELAY: Duration = Duration::from_millis(200);
const TALENT_RETRY_DELAY: Duration = Duration::from_millis(200);
const CHARACTER_ATTRIBUTE_RETRY_DELAY: Duration = Duration::from_millis(200);
const REWARD_RESULT_RETRY_DELAY: Duration = Duration::from_millis(200);
const EXPEDITION_RETRY_DELAY: Duration = Duration::from_millis(200);
const MINING_RETRY_DELAY: Duration = Duration::from_millis(200);

fn weapon_probe_result(
    seq: u64,
    attempts: usize,
    state: WeaponRecognitionState,
    elapsed_ms: f32,
) -> CvDebugResult {
    let (source_label, ocr_text) = match state {
        WeaponRecognitionState::Ready(details) => (
            "武器详情（三次稳定识别）".to_string(),
            format!(
                "名称：{}\n当前等级：{}\n等级上限：{}\n采样次数：{attempts}",
                details.name, details.level, details.level_limit
            ),
        ),
        WeaponRecognitionState::Pending | WeaponRecognitionState::Failed => (
            format!("武器详情（{attempts} 次采样未稳定）"),
            "未获得稳定结果。\n请手动停在武器详情页；名称和等级区域的所有 OCR 置信度必须 ≥ 0.90。\n本测试不点击、不升级、不分解。".to_string(),
        ),
    };
    CvDebugResult {
        updated_seq: seq,
        source_label,
        game_x: None,
        game_y: None,
        match_confidence: None,
        floor: None,
        camera_angle: None,
        camera_confidence: None,
        character_angle: None,
        ocr_text: Some(ocr_text),
        elapsed_ms: Some(elapsed_ms),
    }
}

async fn run_weapon_probe(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<CvDebugResult, String> {
    ensure_task_inactive(snapshot)?;
    let started = Instant::now();
    let mut stability = WeaponRecognitionStability::default();
    let mut attempts = 0;
    let (seq, state) = loop {
        let captured = {
            let mut cap_guard = capturer.lock().unwrap();
            let result = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            result?
        };
        let Some(frame) = captured else {
            return Err("当前帧尚未就绪，请重试武器详情识别".to_string());
        };
        *dbg_seq += 1;
        attempts += 1;
        let cv = cv_ocr.clone();
        let sample = tokio::task::spawn_blocking(move || {
            let mut guard = cv.lock().unwrap();
            guard.detect_weapon_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        })
        .await
        .map_err(|error| error.to_string())??;
        let state = stability.observe(sample);
        if state != WeaponRecognitionState::Pending {
            break (*dbg_seq, state);
        }
        if attempts < WEAPON_MAX_OCR_ATTEMPTS {
            tokio::time::sleep(WEAPON_RETRY_DELAY).await;
        }
    };
    Ok(weapon_probe_result(
        seq,
        attempts,
        state,
        started.elapsed().as_secs_f32() * 1000.0,
    ))
}

fn talent_probe_result(
    seq: u64,
    attempts: usize,
    state: TalentRecognitionState,
    elapsed_ms: f32,
) -> CvDebugResult {
    let (source_label, ocr_text) = match state {
        TalentRecognitionState::Ready(details) => (
            "天赋详情（三次稳定识别）".to_string(),
            format!(
                "天赋类型：{}\n显示等级：{}\n天赋等级+3：{}\n采样次数：{attempts}",
                details.talent_type.label(),
                details.level,
                if details.has_bonus { "是" } else { "否" }
            ),
        ),
        TalentRecognitionState::Pending | TalentRecognitionState::Failed => (
            format!("天赋详情（{attempts} 次采样未稳定）"),
            "未获得稳定结果。\n请手动停在天赋详情页；“天赋介绍”、类型和等级区域的所有 OCR 置信度必须 ≥ 0.90。\n加成区域应为空，或明确显示“天赋等级+3”；异常非空文本会拒绝结果。\n本测试不点击、不切换天赋、不升级。".to_string(),
        ),
    };
    CvDebugResult {
        updated_seq: seq,
        source_label,
        game_x: None,
        game_y: None,
        match_confidence: None,
        floor: None,
        camera_angle: None,
        camera_confidence: None,
        character_angle: None,
        ocr_text: Some(ocr_text),
        elapsed_ms: Some(elapsed_ms),
    }
}

async fn run_talent_probe(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<CvDebugResult, String> {
    ensure_task_inactive(snapshot)?;
    let started = Instant::now();
    let mut stability = TalentRecognitionStability::default();
    let mut attempts = 0;
    let (seq, state) = loop {
        let captured = {
            let mut cap_guard = capturer.lock().unwrap();
            let result = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            result?
        };
        let Some(frame) = captured else {
            return Err("当前帧尚未就绪，请重试天赋详情识别".to_string());
        };
        *dbg_seq += 1;
        attempts += 1;
        let cv = cv_ocr.clone();
        let sample = tokio::task::spawn_blocking(move || {
            let mut guard = cv.lock().unwrap();
            guard.detect_talent_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        })
        .await
        .map_err(|error| error.to_string())??;
        let state = stability.observe(sample);
        if state != TalentRecognitionState::Pending {
            break (*dbg_seq, state);
        }
        if attempts < TALENT_MAX_OCR_ATTEMPTS {
            tokio::time::sleep(TALENT_RETRY_DELAY).await;
        }
    };
    Ok(talent_probe_result(
        seq,
        attempts,
        state,
        started.elapsed().as_secs_f32() * 1000.0,
    ))
}

fn character_attribute_probe_result(
    seq: u64,
    attempts: usize,
    state: CharacterAttributeRecognitionState,
    elapsed_ms: f32,
) -> CvDebugResult {
    let (source_label, ocr_text) = match state {
        CharacterAttributeRecognitionState::Ready(details) => (
            "角色属性（三次稳定识别）".to_string(),
            format!(
                "当前等级：{}\n等级上限：{}\n采样次数：{attempts}",
                details.level, details.level_limit
            ),
        ),
        CharacterAttributeRecognitionState::Pending | CharacterAttributeRecognitionState::Failed => (
            format!("角色属性（{attempts} 次采样未稳定）"),
            "未获得稳定结果。\n请手动停在角色属性页；等级区域需连续 3 次识别出相同的「当前等级/等级上限」。\n本测试不点击、不切换角色、不升级。".to_string(),
        ),
    };
    CvDebugResult {
        updated_seq: seq,
        source_label,
        game_x: None,
        game_y: None,
        match_confidence: None,
        floor: None,
        camera_angle: None,
        camera_confidence: None,
        character_angle: None,
        ocr_text: Some(ocr_text),
        elapsed_ms: Some(elapsed_ms),
    }
}

async fn run_character_attribute_probe(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<CvDebugResult, String> {
    ensure_task_inactive(snapshot)?;
    let started = Instant::now();
    let mut stability = CharacterAttributeRecognitionStability::default();
    let mut attempts = 0;
    let (seq, state) = loop {
        let captured = {
            let mut cap_guard = capturer.lock().unwrap();
            let result = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            result?
        };
        let Some(frame) = captured else {
            return Err("当前帧尚未就绪，请重试角色属性识别".to_string());
        };
        *dbg_seq += 1;
        attempts += 1;
        let cv = cv_ocr.clone();
        let sample = tokio::task::spawn_blocking(move || {
            let mut guard = cv.lock().unwrap();
            guard.detect_character_attribute_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        })
        .await
        .map_err(|error| error.to_string())??;
        let state = stability.observe(sample);
        if state != CharacterAttributeRecognitionState::Pending {
            break (*dbg_seq, state);
        }
        if attempts < CHARACTER_ATTRIBUTE_MAX_OCR_ATTEMPTS {
            tokio::time::sleep(CHARACTER_ATTRIBUTE_RETRY_DELAY).await;
        }
    };
    Ok(character_attribute_probe_result(
        seq,
        attempts,
        state,
        started.elapsed().as_secs_f32() * 1000.0,
    ))
}

fn reward_result_probe_result(
    seq: u64,
    attempts: usize,
    state: RewardResultRecognitionState,
    elapsed_ms: f32,
) -> CvDebugResult {
    let (source_label, ocr_text) = match state {
        RewardResultRecognitionState::Ready(page) => {
            let mut lines = vec![format!("卡片数量：{}", page.0.len()), format!("采样次数：{attempts}")];
            for (index, card) in page.0.iter().enumerate() {
                lines.push(format!(
                    "#{}：{} × {}",
                    index + 1,
                    card.identity.as_ref().map_or("未知名称", |item| item.name.as_str()),
                    card.count.map_or_else(|| "未知".into(), |count| count.to_string())
                ));
            }
            ("奖励结果页（三次稳定识别）".into(), lines.join("\n"))
        }
        RewardResultRecognitionState::Pending | RewardResultRecognitionState::Failed => (
            format!("奖励结果页（{attempts} 次采样未稳定）"),
            "未获得稳定结果。\n请手动停在 1920×1080 奖励结果页；卡片名称必须由 ItemV2 可靠识别，数量 OCR 置信度必须 ≥ 0.90。\n数量无法严格识别时显示“未知”，绝不默认按 1 累计。\n本测试不点击、不翻页、不领取、不按 ESC。".into(),
        ),
    };
    CvDebugResult {
        updated_seq: seq,
        source_label,
        game_x: None,
        game_y: None,
        match_confidence: None,
        floor: None,
        camera_angle: None,
        camera_confidence: None,
        character_angle: None,
        ocr_text: Some(ocr_text),
        elapsed_ms: Some(elapsed_ms),
    }
}

async fn run_reward_result_probe(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<CvDebugResult, String> {
    ensure_task_inactive(snapshot)?;
    let started = Instant::now();
    let mut stability = RewardResultRecognitionStability::default();
    let mut attempts = 0;
    let (seq, state) = loop {
        let captured = {
            let mut cap_guard = capturer.lock().unwrap();
            let result = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            result?
        };
        let Some(frame) = captured else {
            return Err("当前帧尚未就绪，请重试奖励结果页识别".into());
        };
        *dbg_seq += 1;
        attempts += 1;
        let cv = cv_ocr.clone();
        let sample = tokio::task::spawn_blocking(move || {
            let mut guard = cv.lock().unwrap();
            guard.detect_reward_result_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        })
        .await
        .map_err(|error| error.to_string())??;
        let state = stability.observe(sample);
        if state != RewardResultRecognitionState::Pending {
            break (*dbg_seq, state);
        }
        if attempts < REWARD_RESULT_MAX_ATTEMPTS {
            tokio::time::sleep(REWARD_RESULT_RETRY_DELAY).await;
        }
    };
    Ok(reward_result_probe_result(
        seq,
        attempts,
        state,
        started.elapsed().as_secs_f32() * 1000.0,
    ))
}

fn expedition_probe_result(
    seq: u64,
    attempts: usize,
    state: ExpeditionRecognitionState,
    elapsed_ms: f32,
) -> CvDebugResult {
    let (source_label, text) = match state {
        ExpeditionRecognitionState::Ready(page_state) => (
            "派遣页面（三次稳定识别）".into(),
            format!("{}\n采样次数：{attempts}", page_state.label()),
        ),
        ExpeditionRecognitionState::Pending | ExpeditionRecognitionState::Failed => (
            format!("派遣页面（{attempts} 次采样未获得明确结论）"),
            "未检测到唯一明确的“全部领取”或“再次派遣”图标。\n这不等于“无奖励”或“零完成”；未知页面一律不推断。\n本测试不点击、不领取、不翻页、不按 ESC。".into(),
        ),
    };
    CvDebugResult {
        updated_seq: seq,
        source_label,
        game_x: None,
        game_y: None,
        match_confidence: None,
        floor: None,
        camera_angle: None,
        camera_confidence: None,
        character_angle: None,
        ocr_text: Some(text),
        elapsed_ms: Some(elapsed_ms),
    }
}

async fn run_expedition_probe(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<CvDebugResult, String> {
    ensure_task_inactive(snapshot)?;
    let started = Instant::now();
    let mut stability = ExpeditionRecognitionStability::default();
    let mut attempts = 0;
    let (seq, state) = loop {
        let captured = {
            let mut cap_guard = capturer.lock().unwrap();
            let result = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            result?
        };
        let Some(frame) = captured else {
            return Err("当前帧尚未就绪，请重试派遣页面识别".into());
        };
        *dbg_seq += 1;
        attempts += 1;
        let cv = cv_ocr.clone();
        let sample = tokio::task::spawn_blocking(move || {
            let mut guard = cv.lock().unwrap();
            guard.detect_expedition_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        })
        .await
        .map_err(|error| error.to_string())??;
        let state = stability.observe(sample);
        if state != ExpeditionRecognitionState::Pending {
            break (*dbg_seq, state);
        }
        if attempts < EXPEDITION_MAX_ATTEMPTS {
            tokio::time::sleep(EXPEDITION_RETRY_DELAY).await;
        }
    };
    Ok(expedition_probe_result(
        seq,
        attempts,
        state,
        started.elapsed().as_secs_f32() * 1000.0,
    ))
}

fn mining_probe_result(
    seq: u64,
    attempts: usize,
    state: MiningRecognitionState,
    elapsed_ms: f32,
) -> CvDebugResult {
    let (source_label, text) = match state {
        MiningRecognitionState::Ready(observation) => match observation.target {
            Some(target) => (
                "自动采矿（三次稳定识别）".into(),
                format!(
                    "检测到矿石目标（仅识别）\n聚类数：{}\n目标框：x={}, y={}, {}×{}\n目标中心：({:.1}, {:.1})\n采样次数：{attempts}",
                    observation.cluster_count,
                    target.rect.x,
                    target.rect.y,
                    target.rect.width,
                    target.rect.height,
                    target.center_x,
                    target.center_y
                ),
            ),
            None => (
                "自动采矿（三次稳定识别）".into(),
                format!(
                    "连续 3 帧未检测到置信度 ≥ 0.70 的 ore。\n仅代表当前稳定画面，不保证镜头外没有矿石。\n采样次数：{attempts}"
                ),
            ),
        },
        MiningRecognitionState::Pending => (
            format!("自动采矿（{attempts} 次采样未获得明确结论）"),
            "尚未获得连续一致的采矿观察。\n未知不等于无矿。\n本测试不聚焦、不移动鼠标、不点击、不切换角色、不发送按键。".into(),
        ),
        MiningRecognitionState::Unknown(reason) => (
            format!("自动采矿（{attempts} 次采样未获得明确结论）"),
            format!(
                "{reason}\n未知不等于无矿。\n请确认 Assets 中 Model/Mine/bgi_mine.onnx 存在且 ONNX Runtime 可加载。\n本测试不聚焦、不移动鼠标、不点击、不切换角色、不发送按键。"
            ),
        ),
    };
    CvDebugResult {
        updated_seq: seq,
        source_label,
        game_x: None,
        game_y: None,
        match_confidence: None,
        floor: None,
        camera_angle: None,
        camera_confidence: None,
        character_angle: None,
        ocr_text: Some(text),
        elapsed_ms: Some(elapsed_ms),
    }
}

async fn run_mining_probe(
    snapshot: &UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    dbg_seq: &mut u64,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<CvDebugResult, String> {
    ensure_task_inactive(snapshot)?;
    let started = Instant::now();
    let mut stability = MiningRecognitionStability::default();
    let mut attempts = 0;
    let (seq, state) = loop {
        let captured = {
            let mut cap_guard = capturer.lock().unwrap();
            let result = capture_raw_frame(snapshot, &mut cap_guard, pool, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            result?
        };
        let Some(frame) = captured else {
            return Err("当前帧尚未就绪，请重试自动采矿识别".into());
        };
        *dbg_seq += 1;
        attempts += 1;
        let cv = cv_ocr.clone();
        let sample = tokio::task::spawn_blocking(move || {
            let mut guard = cv.lock().unwrap();
            guard.detect_mining_step(
                frame.data.as_slice(),
                frame.width,
                frame.height,
                frame.format.is_bgr24(),
            )
        })
        .await
        .map_err(|error| error.to_string())?;
        let state = match sample {
            Ok(observation) => stability.observe(Some(observation)),
            Err(error) => {
                core.emit_log(
                    LogLevel::Warn,
                    "CV",
                    format!("[自动采矿只读识别] 推理失败，按未知处理: {error}"),
                );
                stability.reject("模型加载或推理失败，结果未知")
            }
        };
        if state != MiningRecognitionState::Pending {
            break (*dbg_seq, state);
        }
        if attempts < MINING_MAX_ATTEMPTS {
            tokio::time::sleep(MINING_RETRY_DELAY).await;
        }
    };
    Ok(mining_probe_result(
        seq,
        attempts,
        state,
        started.elapsed().as_secs_f32() * 1000.0,
    ))
}

fn stop_active_pathing(
    snapshot: &mut UiSnapshot,
    active_pathing: &mut Option<ActiveTask>,
    core: &CoreEndpoint,
    reason: &str,
) {
    if let Some(active) = active_pathing.as_ref() {
        active.request_stop();
        if matches!(snapshot.task_state, TaskState::Stopping { .. }) {
            return;
        }
        match &snapshot.task_state {
            TaskState::Pathing {
                task_id,
                task_name,
                current_index,
                total_points,
            } => {
                snapshot.task_state = TaskState::Stopping {
                    task_id: *task_id,
                    task_name: task_name.clone(),
                    current_index: *current_index,
                    total_points: *total_points,
                };
            }
            TaskState::Cooking { task_id, task_name }
            | TaskState::ClaimingRewards { task_id, task_name }
            | TaskState::OpeningChest { task_id, task_name }
            | TaskState::AutoWood { task_id, task_name }
            | TaskState::OneKeyExpedition { task_id, task_name }
            | TaskState::SalvagePreview { task_id, task_name }
            | TaskState::Fishing { task_id, task_name } => {
                snapshot.task_state = TaskState::Stopping {
                    task_id: *task_id,
                    task_name: task_name.clone(),
                    current_index: 0,
                    total_points: 1,
                };
            }
            _ => {}
        }
        let release_inputs = active_pathing
            .as_ref()
            .is_none_or(|task| task.releases_inputs_on_stop());
        if release_inputs {
            std::thread::spawn(bgi_input::release_all_keys);
        }
        core.emit_log(LogLevel::Warn, "Pathing", reason.to_string());
    }
}

async fn stop_and_join_active_pathing(
    snapshot: &mut UiSnapshot,
    active_pathing: &mut Option<ActiveTask>,
    core: &CoreEndpoint,
    reason: &str,
) -> Result<(), String> {
    stop_active_pathing(snapshot, active_pathing, core, reason);
    let join_result = if let Some(mut active) = active_pathing.take() {
        if let Some(join) = active.take_join() {
            tokio::task::spawn_blocking(move || join.join())
                .await
                .map_err(|e| format!("等待寻路线程退出失败: {e}"))
                .and_then(|r| r.map_err(|_| "寻路线程异常退出".to_string()))
        } else {
            Ok(())
        }
    } else {
        Ok(())
    };
    bgi_input::release_all_keys();
    snapshot.task_state = if snapshot.preview_enabled {
        TaskState::Previewing
    } else {
        TaskState::Idle
    };
    join_result
}

pub(crate) fn load_auto_pick_config(assets_dir: &str) -> Result<bgi_task::AutoPickConfig, String> {
    let assets = PathBuf::from(assets_dir);
    let whitelist_path = assets.join("Config/Pick/default_pick_white_lists.json");
    let mut whitelist: Vec<String> = serde_json::from_slice(
        &std::fs::read(&whitelist_path)
            .map_err(|e| format!("读取默认拾取白名单失败 {}: {e}", whitelist_path.display()))?,
    )
    .map_err(|e| format!("解析默认拾取白名单失败: {e}"))?;
    let user_dir = assets
        .parent()
        .map(|path| path.join("User"))
        .unwrap_or_else(|| PathBuf::from("User"));
    for (file_name, remove) in [
        ("pick_whitelist_mode_pick_lists.txt", false),
        ("pick_whitelist_mode_do_not_pick_lists.txt", true),
    ] {
        let path = user_dir.join(file_name);
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("读取拾取名单失败 {}: {e}", path.display()))?;
        for item in text.lines().map(str::trim).filter(|item| !item.is_empty()) {
            if remove {
                whitelist.retain(|existing| existing != item);
            } else if !whitelist.iter().any(|existing| existing == item) {
                whitelist.push(item.to_string());
            }
        }
    }
    Ok(bgi_task::AutoPickConfig {
        whitelist,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn start_basic_pathing_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有正在执行或正在停止的寻路任务".to_string());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先在「抓图预览」中选择目标游戏窗口".to_string());
    };
    let Some(file) = snapshot.active_path_file.clone() else {
        return Err("尚未选择路径文件".to_string());
    };
    let (summary, task) = bgi_pathing::load_path_file(&file).map_err(|e| e.to_string())?;
    let total_points = task.positions.len();
    let task_name = summary.name.clone();

    core.emit_log(
        LogLevel::Info,
        "Pathing",
        format!(
            "启动基础寻路「{}」：共 {} 个点位，基础可执行 {} 个，跳过复杂动作 [{}]",
            summary.name,
            summary.waypoint_count,
            summary.supported_count,
            if summary.skipped_actions.is_empty() {
                "无".to_string()
            } else {
                summary.skipped_actions.join(", ")
            }
        ),
    );

    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };

    let worker_name = task_name.clone();
    let auto_pick_config = if snapshot.config.pathing_auto_pick {
        match load_auto_pick_config(&snapshot.config.assets_dir) {
            Ok(config) => Some(config),
            Err(error) => {
                core.emit_log(
                    LogLevel::Warn,
                    "AutoPick",
                    format!("自动拾取已停用: {error}"),
                );
                None
            }
        }
    } else {
        None
    };
    let trigger_leases =
        trigger_runtime.create_pathing_trigger_leases(&task.config, auto_pick_config.is_some())?;
    let realtime_auto_pick = trigger_leases
        .iter()
        .any(|lease| lease.name() == "AutoPick");
    trigger_runtime.set_auto_pick_config(auto_pick_config.clone());
    let policy = LoopPolicy {
        runs: snapshot.config.pathing_runs,
        delay: Duration::from_millis(snapshot.config.pathing_loop_delay_ms),
        wait_for_loading: snapshot.config.pathing_wait_for_loading,
        auto_pick: snapshot.config.pathing_auto_pick && !realtime_auto_pick,
        auto_pick_config,
        ..Default::default()
    };
    let runs_requested = policy.runs;
    let join = std::thread::Builder::new()
        .name(format!("bgi-pathing-exec-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                LoopRunner::run(&mut host, task_id, &worker_name, &task, policy)
            }))
            .unwrap_or_else(|_| bgi_control::TaskReport {
                task_id,
                task_name: worker_name.clone(),
                outcome: bgi_control::TaskOutcome::Failed,
                runs_completed: 0,
                runs_requested,
                current_index: None,
                total_points: task.positions.len(),
                failure: Some("寻路线程发生 panic".to_string()),
                stage: "panic".into(),
                elapsed_ms: 0,
            });
            bgi_input::release_all_keys();
            drop(trigger_leases);
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(res),
            });
        })
        .map_err(|e| format!("启动寻路线程失败: {e}"))?;

    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::Pathing {
        task_id,
        task_name: task_name.clone(),
        current_index: 1,
        total_points,
    };
    Ok(())
}

fn auto_cook_task_config(snapshot: &UiSnapshot) -> AutoCookTaskConfig {
    AutoCookTaskConfig {
        check_interval_ms: snapshot.config.auto_cook_interval_ms,
        stop_on_recover: snapshot.config.auto_cook_stop_on_recover,
        total_timeout_ms: snapshot.config.auto_cook_timeout_ms,
    }
}

fn reward_claim_task_config(limits: &TaskLimitsConfig) -> RewardClaimConfig {
    RewardClaimConfig {
        max_clicks: limits.reward_claim_max_clicks,
        total_timeout_ms: limits.reward_claim_timeout_ms,
    }
}

fn open_chest_task_config(limits: &TaskLimitsConfig) -> AutoOpenChestConfig {
    AutoOpenChestConfig {
        total_timeout_ms: limits.auto_open_chest_timeout_ms,
        target_lost_timeout_ms: limits.auto_open_chest_target_lost_timeout_ms,
        verify_timeout_ms: limits.auto_open_chest_verify_timeout_ms,
    }
}

fn auto_wood_task_config(limits: &TaskLimitsConfig) -> AutoWoodConfig {
    AutoWoodConfig {
        total_timeout_ms: limits.auto_wood_total_timeout_ms,
        gain_timeout_ms: limits.auto_wood_gain_timeout_ms,
        login_timeout_ms: limits.auto_wood_login_timeout_ms,
        menu_timeout_ms: limits.auto_wood_menu_timeout_ms,
        logout_confirm_timeout_ms: limits.auto_wood_logout_confirm_timeout_ms,
        empty_ocr_limit: limits.auto_wood_empty_ocr_limit,
    }
}

fn run_auto_cook(host: &mut CorePathingHost, task_config: AutoCookTaskConfig) -> TaskReport {
    let started = std::time::Instant::now();
    let check_interval = task_config
        .check_interval()
        .unwrap_or(Duration::from_millis(10));
    let mut report = TaskReport {
        task_id: host.task_id,
        task_name: "自动烹饪".into(),
        outcome: TaskOutcome::Success,
        runs_completed: 0,
        runs_requested: 0,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: "waiting_for_cooking".into(),
        elapsed_ms: 0,
    };
    let mut detector = CookingBarDetector::default();
    loop {
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            break;
        }
        if task_config.timed_out(started, std::time::Instant::now()) {
            report.outcome = TaskOutcome::Failed;
            report.stage = "timeout".into();
            report.failure = Some(format!(
                "自动烹饪超过 {} 秒，已停止；不会继续点击或按键",
                task_config.total_timeout_ms / 1000
            ));
            break;
        }
        let observation = match host.capture_cooking_bar() {
            Ok(Some(observation)) => observation,
            Ok(None) => {
                detector.reset();
                host.sleep_ms(check_interval.as_millis() as u64);
                continue;
            }
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "capture".into();
                report.failure = Some(error);
                break;
            }
        };
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            break;
        }
        if task_config.timed_out(started, std::time::Instant::now()) {
            report.outcome = TaskOutcome::Failed;
            report.stage = "timeout".into();
            report.failure = Some(format!(
                "自动烹饪超过 {} 秒，已停止；不会继续点击或按键",
                task_config.total_timeout_ms / 1000
            ));
            break;
        }
        if !observation.page_visible {
            detector.reset();
            host.sleep_ms(check_interval.as_millis() as u64);
            continue;
        }
        if observation.recover_visible {
            detector.reset();
            if task_config.stop_on_recover {
                report.stage = "completed".into();
                host.on_log(
                    LogLevel::Info,
                    "自动烹饪：检测到再次烹饪按钮，任务结束".into(),
                );
                break;
            }
            host.sleep_ms(check_interval.as_millis() as u64);
            continue;
        }
        if let Some((x, y)) = observation.white_confirm {
            host.mouse_left_click_client(x, y);
            detector.reset();
            host.on_log(LogLevel::Info, "自动烹饪：已点击白色确认按钮".into());
            host.sleep_ms(400.max(check_interval.as_millis() as u64));
            continue;
        }
        let config = match CookingBarConfig::default().scaled(observation.scale) {
            Ok(config) => config,
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "configuration".into();
                report.failure = Some(error.into());
                break;
            }
        };
        let event = detector.observe(observation.color_count, true, &config);
        if let CookingBarEvent::CompletionTriggered { peak, current } = event {
            host.key_tap(0x20, 80);
            report.runs_completed = report.runs_completed.saturating_add(1);
            host.on_log(
                LogLevel::Info,
                format!("自动烹饪完成: 峰值 {peak}，当前 {current}"),
            );
        }
        host.sleep_ms(check_interval.as_millis() as u64);
    }
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

#[allow(clippy::too_many_arguments)]
fn start_auto_cook_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let task_config = auto_cook_task_config(snapshot);
    task_config.validate().map_err(|error| error.to_string())?;
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };
    trigger_runtime.set_auto_pick_config(None);
    if let Ok(mut guard) = cv_ocr.try_lock() {
        guard.reset_cooking_ui_cache();
    }
    let join = std::thread::Builder::new()
        .name(format!("bgi-cooking-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_auto_cook(&mut host, task_config)
            }))
            .unwrap_or_else(|_| TaskReport {
                task_id,
                task_name: "自动烹饪".into(),
                outcome: TaskOutcome::Failed,
                runs_completed: 0,
                runs_requested: 0,
                current_index: None,
                total_points: 1,
                failure: Some("自动烹饪线程发生 panic".into()),
                stage: "panic".into(),
                elapsed_ms: 0,
            });
            bgi_input::release_all_keys();
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(result),
            });
        })
        .map_err(|e| format!("启动自动烹饪线程失败: {e}"))?;
    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::Cooking {
        task_id,
        task_name: "自动烹饪".into(),
    };
    core.emit_log(
        LogLevel::Info,
        "Cooking",
        "自动烹饪已启动，请先在游戏内开始烹饪",
    );
    Ok(())
}

fn run_reward_claim(
    host: &mut CorePathingHost,
    task_config: RewardClaimConfig,
    mode: RewardClaimMode,
) -> TaskReport {
    let started = std::time::Instant::now();
    let task_name = mode.label();
    let mut report = TaskReport {
        task_id: host.task_id,
        task_name: task_name.into(),
        outcome: TaskOutcome::Success,
        runs_completed: 0,
        runs_requested: 1,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: "scanning".into(),
        elapsed_ms: 0,
    };
    let mut controller = RewardClaimController::try_new_with_mode(task_config, mode)
        .expect("validated reward config");
    let total_timeout = Duration::from_millis(task_config.total_timeout_ms);
    host.focus_game_window();
    loop {
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            break;
        }
        if started.elapsed() >= total_timeout {
            report.outcome = TaskOutcome::Failed;
            report.stage = "timeout".into();
            report.failure = Some(format!(
                "{task_name}超过 {} 秒，已停止；不会继续点击或按 ESC",
                task_config.total_timeout_ms / 1000
            ));
            break;
        }
        let now = std::time::Instant::now();
        if let Some(wait_ms) = controller.wait_ms(now) {
            host.sleep_ms(wait_ms);
            continue;
        }
        let (candidates, continue_tip) = match host.detect_rewards() {
            Ok(Some(observation)) => observation,
            Ok(None) => {
                host.sleep_ms(30);
                continue;
            }
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "capture".into();
                report.failure = Some(error);
                break;
            }
        };
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            break;
        }
        if started.elapsed() >= total_timeout {
            report.outcome = TaskOutcome::Failed;
            report.stage = "timeout".into();
            report.failure = Some(format!(
                "{task_name}超过 {} 秒，已停止；不会继续点击或按 ESC",
                task_config.total_timeout_ms / 1000
            ));
            break;
        }
        match controller.advance_since(
            &candidates,
            continue_tip,
            std::time::Instant::now(),
            started,
        ) {
            RewardClaimAction::Finished => {
                if task_config.timed_out(started, std::time::Instant::now()) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "timeout".into();
                    report.failure = Some(format!(
                        "{task_name}超过 {} 秒，已停止；不会继续点击或按 ESC",
                        task_config.total_timeout_ms / 1000
                    ));
                }
                break;
            }
            RewardClaimAction::Wait => {}
            RewardClaimAction::Click(candidate) => {
                if !bgi_input::is_foreground_window(host.hwnd) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "foreground".into();
                    report.failure = Some(format!("游戏窗口不在前台，已停止{task_name}"));
                    break;
                }
                host.mouse_left_click_client(candidate.center.0, candidate.center.1);
                let name = match candidate.kind {
                    RewardKind::ClaimText => "领取",
                    RewardKind::ClaimGift => "礼物领取",
                };
                report.runs_completed = controller.clicked_count() as u32;
                host.on_log(
                    LogLevel::Info,
                    format!(
                        "{task_name}：点击{name}图标 ({}/{})",
                        report.runs_completed, task_config.max_clicks
                    ),
                );
            }
            RewardClaimAction::PressEscape => {
                if !bgi_input::is_foreground_window(host.hwnd) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "foreground".into();
                    report.failure = Some(format!("游戏窗口不在前台，已停止{task_name}"));
                    break;
                }
                host.key_tap(0x1B, 60);
                host.on_log(
                    LogLevel::Info,
                    format!("{task_name}：检测到点击空白继续提示，已按 ESC"),
                );
            }
        }
    }
    if report.outcome == TaskOutcome::Success {
        match controller.finish_reason() {
            Some(RewardClaimFinish::ClickLimit) => {
                report.stage = "click_limit".into();
                host.on_log(
                    LogLevel::Warn,
                    format!(
                        "{task_name}：已达到单次 {} 次点击上限，请检查当前页面",
                        task_config.max_clicks
                    ),
                );
            }
            Some(RewardClaimFinish::Timeout) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "timeout".into();
                report.failure = Some(format!(
                    "{task_name}超过 {} 秒，已停止；不会继续点击或按 ESC",
                    task_config.total_timeout_ms / 1000
                ));
            }
            Some(RewardClaimFinish::EmptyPage | RewardClaimFinish::EmptyScans) | None => {
                report.stage = "completed".into();
            }
        }
    }
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

#[allow(clippy::too_many_arguments)]
fn start_reward_claim_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
    mode: RewardClaimMode,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    let task_config = reward_claim_task_config(&snapshot.config.task_limits);
    task_config.validate().map_err(|error| error.to_string())?;
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-claim-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_reward_claim(&mut host, task_config, mode)
            }))
            .unwrap_or_else(|_| TaskReport {
                task_id,
                task_name: mode.label().into(),
                outcome: TaskOutcome::Failed,
                runs_completed: 0,
                runs_requested: 1,
                current_index: None,
                total_points: 1,
                failure: Some(format!("{}线程发生 panic", mode.label())),
                stage: "panic".into(),
                elapsed_ms: 0,
            });
            bgi_input::release_all_keys();
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(result),
            });
        })
        .map_err(|e| format!("启动{}线程失败: {e}", mode.label()))?;
    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::ClaimingRewards {
        task_id,
        task_name: mode.label().into(),
    };
    core.emit_log(
        LogLevel::Info,
        "RewardClaim",
        format!("{}已启动，将点击当前页面上的领取图标", mode.label()),
    );
    Ok(())
}

const AUTO_WOOD_TASK_NAME: &str = "自动砍树（王树瑞佑）";
const OPEN_CHEST_TASK_NAME: &str = "自动开箱当前目标";
const ONE_KEY_EXPEDITION_TASK_NAME: &str = "一键派遣";
const VK_W: u16 = 0x57;
const VK_S: u16 = 0x53;
const VK_F: u16 = 0x46;

fn stop_chest_movement(host: &mut CorePathingHost) {
    for vk in [VK_W, VK_S] {
        if host.held_keys.contains(&vk) {
            host.key_up(vk);
        }
    }
}

fn set_chest_movement(host: &mut CorePathingHost, forward: bool) {
    let (wanted, opposite) = if forward { (VK_W, VK_S) } else { (VK_S, VK_W) };
    if host.held_keys.contains(&opposite) {
        host.key_up(opposite);
    }
    if !host.held_keys.contains(&wanted) {
        host.key_down(wanted);
    }
}

fn expedition_automation_target(
    observation: bgi_cv::expedition::ExpeditionObservation,
) -> Option<OneKeyExpeditionTarget> {
    let state = observation.state()?;
    let point = match state {
        bgi_task::ExpeditionPageState::CollectAvailable => observation.collect?,
        bgi_task::ExpeditionPageState::RedispatchAvailable => observation.redispatch?,
    };
    Some(OneKeyExpeditionTarget {
        state,
        x: point.x.round() as i32,
        y: point.y.round() as i32,
    })
}

fn run_one_key_expedition(host: &mut CorePathingHost) -> TaskReport {
    let started = std::time::Instant::now();
    let mut report = TaskReport {
        task_id: host.task_id,
        task_name: ONE_KEY_EXPEDITION_TASK_NAME.into(),
        outcome: TaskOutcome::Success,
        runs_completed: 0,
        runs_requested: 1,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: "waiting_page".into(),
        elapsed_ms: 0,
    };
    let mut controller =
        OneKeyExpeditionController::new(OneKeyExpeditionConfig::default()).unwrap();
    let mut frame_size = (1920_u32, 1080_u32);
    let mut redispatched = false;
    host.focus_game_window();
    loop {
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            report.failure = Some("一键派遣已取消".into());
            break;
        }
        if !bgi_input::is_foreground_window(host.hwnd) {
            report.outcome = TaskOutcome::Failed;
            report.stage = "foreground".into();
            report.failure = Some("游戏窗口不在前台，已停止一键派遣".into());
            break;
        }
        let observation = match host.detect_expedition() {
            Ok(Some(observation)) => {
                frame_size = (observation.width, observation.height);
                Some(observation)
            }
            Ok(None) => None,
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "capture".into();
                report.failure = Some(error);
                break;
            }
        };
        let target = observation.and_then(expedition_automation_target);
        let action = controller.advance(target, frame_size, std::time::Instant::now());
        match action {
            OneKeyExpeditionAction::ClickCollect { x, y } => {
                if !host.click_client_guarded(x, y) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "collect_click".into();
                    report.failure = Some("点击全部领取时窗口失焦或点击失败".into());
                    break;
                }
                report.stage = "collect_settling".into();
                host.on_log(
                    LogLevel::Info,
                    "一键派遣：已点击全部领取，等待页面响应".into(),
                );
            }
            OneKeyExpeditionAction::ClickRedispatch { x, y } => {
                if !host.click_client_guarded(x, y) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "redispatch_click".into();
                    report.failure = Some("点击再次派遣时窗口失焦或点击失败".into());
                    break;
                }
                redispatched = true;
                report.stage = "verifying_redispatch".into();
                host.on_log(
                    LogLevel::Info,
                    "一键派遣：已点击再次派遣，等待图标消失".into(),
                );
            }
            OneKeyExpeditionAction::PressEscape => {
                host.key_tap(0x1b, 80);
                report.stage = "exiting".into();
            }
            OneKeyExpeditionAction::Wait => {}
            OneKeyExpeditionAction::Succeeded => {
                report.outcome = TaskOutcome::Success;
                report.runs_completed = u32::from(redispatched);
                report.stage = "completed".into();
                report.failure = None;
                host.on_log(
                    LogLevel::Info,
                    "一键派遣：再次派遣图标已消失，已发送 ESC 退出派遣页".into(),
                );
                break;
            }
            OneKeyExpeditionAction::Failed(reason) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "one_key_expedition".into();
                report.failure = Some(reason.into());
                break;
            }
        }
        host.sleep_ms(ONE_KEY_EXPEDITION_TICK.as_millis() as u64);
    }
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

#[allow(clippy::too_many_arguments)]
fn start_one_key_expedition_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    OneKeyExpeditionConfig::default()
        .validate()
        .map_err(|error| error.to_string())?;
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-expedition-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let events_tx = host.events_tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_one_key_expedition(&mut host)
            }))
            .unwrap_or_else(|_| TaskReport {
                task_id,
                task_name: ONE_KEY_EXPEDITION_TASK_NAME.into(),
                outcome: TaskOutcome::Failed,
                runs_completed: 0,
                runs_requested: 1,
                current_index: None,
                total_points: 1,
                failure: Some("一键派遣线程发生 panic".into()),
                stage: "panic".into(),
                elapsed_ms: 0,
            });
            bgi_input::release_all_keys();
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(result),
            });
        })
        .map_err(|e| format!("启动一键派遣线程失败: {e}"))?;
    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::OneKeyExpedition {
        task_id,
        task_name: ONE_KEY_EXPEDITION_TASK_NAME.into(),
    };
    core.emit_log(
        LogLevel::Info,
        "Expedition",
        "一键派遣已启动：最多领取一次、再次派遣一次；未知或歧义页面不点击，失焦立即停止"
            .to_string(),
    );
    Ok(())
}

fn run_open_current_chest(
    host: &mut CorePathingHost,
    task_config: AutoOpenChestConfig,
) -> TaskReport {
    let started = std::time::Instant::now();
    let mut report = TaskReport {
        task_id: host.task_id,
        task_name: OPEN_CHEST_TASK_NAME.into(),
        outcome: TaskOutcome::Success,
        runs_completed: 0,
        runs_requested: 1,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: "approaching".into(),
        elapsed_ms: 0,
    };
    let mut controller = AutoOpenChestController::new(task_config);
    let mut frame_size = (1920_u32, 1080_u32);
    host.focus_game_window();
    loop {
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            report.failure = Some("自动开箱已取消".into());
            break;
        }
        if !bgi_input::is_foreground_window(host.hwnd) {
            report.outcome = TaskOutcome::Failed;
            report.stage = "foreground".into();
            report.failure = Some("游戏窗口不在前台，已停止自动开箱".into());
            break;
        }

        let (chest, interact, gameplay) = match host.detect_chest() {
            Ok(Some((observation, gameplay))) => {
                frame_size = (observation.width, observation.height);
                (observation.chest, observation.interact, gameplay)
            }
            Ok(None) => (None, false, true),
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "capture".into();
                report.failure = Some(error);
                break;
            }
        };
        if !gameplay {
            report.outcome = TaskOutcome::Failed;
            report.stage = "gameplay".into();
            report.failure = Some("检测到菜单、对话或非游戏画面，已停止自动开箱".into());
            break;
        }

        let action = controller.advance(chest, interact, frame_size, std::time::Instant::now());
        match action {
            AutoOpenChestAction::MoveForward => set_chest_movement(host, true),
            AutoOpenChestAction::MoveBackward => set_chest_movement(host, false),
            AutoOpenChestAction::Turn(dx) => host.mouse_move_rel(dx, 0),
            AutoOpenChestAction::Interact => {
                stop_chest_movement(host);
                host.key_tap(VK_F, 80);
                report.stage = "verifying".into();
            }
            AutoOpenChestAction::Wait => {}
            AutoOpenChestAction::Stop => {
                stop_chest_movement(host);
                report.stage = "approaching".into();
            }
            AutoOpenChestAction::Succeeded => {
                report.outcome = TaskOutcome::Success;
                report.runs_completed = 1;
                report.stage = "completed".into();
                host.on_log(
                    LogLevel::Info,
                    "自动开箱：已确认当前宝箱消失，本次成功开启 1 个箱子".into(),
                );
                break;
            }
            AutoOpenChestAction::Failed(reason) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "auto_open_chest".into();
                report.failure = Some(reason.into());
                break;
            }
        }
        host.sleep_ms(AUTO_OPEN_CHEST_TICK.as_millis() as u64);
    }
    stop_chest_movement(host);
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

fn wood_ocr_text(hits: &[bgi_pathing::OcrHit]) -> String {
    let mut ordered: Vec<&bgi_pathing::OcrHit> = hits.iter().collect();
    ordered.sort_by_key(|hit| (hit.center_y, hit.center_x));
    ordered
        .into_iter()
        .map(|hit| hit.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn wood_ocr_roi(width: u32, height: u32) -> (f32, f32, f32, f32) {
    // C# AutoWood uses s=min(cw/1920,1), rect(100s,450s,300s,250s).
    let scale = (f64::from(width) / 1920.0).clamp(0.0, 1.0);
    let x0 = 100.0 * scale;
    let y0 = 450.0 * scale;
    let x1 = (x0 + 300.0 * scale).min(f64::from(width));
    let y1 = (y0 + 250.0 * scale).min(f64::from(height));
    (
        (x0 / f64::from(width.max(1))) as f32,
        (y0 / f64::from(height.max(1))) as f32,
        (x1 / f64::from(width.max(1))) as f32,
        (y1 / f64::from(height.max(1))) as f32,
    )
}

fn run_auto_wood(
    host: &mut CorePathingHost,
    verify_gain: bool,
    task_config: AutoWoodConfig,
) -> TaskReport {
    let started = std::time::Instant::now();
    let mut report = TaskReport {
        task_id: host.task_id,
        task_name: AUTO_WOOD_TASK_NAME.into(),
        outcome: TaskOutcome::Success,
        runs_completed: 0,
        runs_requested: 1,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: "focusing".into(),
        elapsed_ms: 0,
    };
    let mut controller = AutoWoodController::new_with_config(verify_gain, task_config);
    let mut next_ocr_at = std::time::Instant::now();
    let mut last_stage = "";
    host.focus_game_window();
    loop {
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            report.failure = Some("自动砍树已取消".into());
            break;
        }
        if !bgi_input::is_foreground_window(host.hwnd) {
            report.outcome = TaskOutcome::Failed;
            report.stage = "foreground".into();
            report.failure = Some("游戏窗口不在前台，已停止自动砍树".into());
            break;
        }

        let stage_before = controller.stage();
        let (mut recognized, mut gameplay, frame_size) = match host.detect_auto_wood() {
            Ok(Some((recognized, gameplay))) => {
                (recognized, gameplay, (recognized.width, recognized.height))
            }
            Ok(None) => {
                host.sleep_ms(AUTO_WOOD_TICK.as_millis() as u64);
                continue;
            }
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = stage_before.into();
                report.failure = Some(error);
                break;
            }
        };
        let mut observation = AutoWoodTaskObservation {
            gadget: recognized.gadget,
            menu: recognized.menu,
            confirm: recognized.confirm,
            enter_game: recognized.enter_game,
            gameplay,
            ocr: None,
        };
        let now = std::time::Instant::now();
        if verify_gain
            && matches!(stage_before, "verify_gain" | "finish_used")
            && now >= next_ocr_at
        {
            let (x0, y0, x1, y1) = wood_ocr_roi(frame_size.0, frame_size.1);
            match host.capture_ocr_roi(x0, y0, x1, y1) {
                Ok(hits) => {
                    let text = wood_ocr_text(&hits);
                    observation.ocr = Some(AutoWoodOcrSample {
                        quantity: parse_wood_gain(&text),
                        text,
                    });
                    next_ocr_at = std::time::Instant::now() + Duration::from_millis(500);
                }
                Err(error) => {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "ocr".into();
                    report.failure = Some(format!("木材收益 OCR 失败: {error}"));
                    break;
                }
            }
            // OCR can be slow: never act on the pre-OCR foreground state.
            if host.is_cancelled() {
                report.outcome = TaskOutcome::Cancelled;
                report.stage = "cancelled".into();
                report.failure = Some("自动砍树已取消".into());
                break;
            }
            if !bgi_input::is_foreground_window(host.hwnd) {
                report.outcome = TaskOutcome::Failed;
                report.stage = "foreground".into();
                report.failure = Some("OCR 期间游戏窗口失焦，已停止自动砍树".into());
                break;
            }
        }

        if observation.ocr.is_some() {
            // Refresh recognition after OCR so no click uses a pre-OCR frame.
            match host.detect_auto_wood() {
                Ok(Some((fresh, fresh_gameplay))) => {
                    recognized = fresh;
                    gameplay = fresh_gameplay;
                    observation.gadget = recognized.gadget;
                    observation.menu = recognized.menu;
                    observation.confirm = recognized.confirm;
                    observation.enter_game = recognized.enter_game;
                    observation.gameplay = gameplay;
                }
                Ok(None) => {
                    host.sleep_ms(AUTO_WOOD_TICK.as_millis() as u64);
                    continue;
                }
                Err(error) => {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "capture_after_ocr".into();
                    report.failure = Some(error);
                    break;
                }
            }
        }
        let action = controller.advance(observation, std::time::Instant::now());
        if host.is_cancelled() {
            report.outcome = TaskOutcome::Cancelled;
            report.stage = "cancelled".into();
            report.failure = Some("自动砍树已取消".into());
            break;
        }
        let stage = controller.stage();
        if stage != last_stage {
            host.on_log(
                LogLevel::Info,
                format!("自动砍树阶段: {stage} (验证收益={verify_gain})"),
            );
            last_stage = stage;
        }
        report.stage = stage.into();
        match action {
            AutoWoodAction::UseGadget => {
                if !gameplay
                    || recognized.gadget.is_none()
                    || recognized.menu.is_some()
                    || !bgi_input::is_foreground_window(host.hwnd)
                {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "gameplay".into();
                    report.failure = Some("按 Z 前游戏画面或前台状态已变化".into());
                    break;
                }
                host.key_tap(0x5A, 80);
            }
            AutoWoodAction::OpenMenu => {
                if !gameplay
                    || recognized.menu.is_some()
                    || !bgi_input::is_foreground_window(host.hwnd)
                {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "gameplay".into();
                    report.failure = Some("按 ESC 前游戏画面或前台状态已变化".into());
                    break;
                }
                host.key_tap(0x1B, 80);
            }
            AutoWoodAction::ConfirmLogout((x, y)) => {
                // The state machine only emits this action when this frame has BOTH menu and exit icon.
                if recognized.menu.is_none()
                    || recognized.confirm.is_none()
                    || !bgi_input::is_foreground_window(host.hwnd)
                {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "logout_guard".into();
                    report.failure = Some("退出前未同时确认背包菜单和退出按钮或窗口已失焦".into());
                    break;
                }
                if !host.click_client_guarded(x, y) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "foreground".into();
                    report.failure = Some("移动至退出按钮后窗口失焦，未点击".into());
                    break;
                }
            }
            AutoWoodAction::Confirm((x, y)) => {
                if gameplay
                    || recognized.menu.is_some()
                    || recognized.confirm.is_none()
                    || !bgi_input::is_foreground_window(host.hwnd)
                {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "logout_confirm_guard".into();
                    report.failure = Some("确认退出前未确认退出弹窗且已离开游戏画面".into());
                    break;
                }
                if !host.click_client_guarded(x, y) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "foreground".into();
                    report.failure = Some("移动至退出确认按钮后窗口失焦，未点击".into());
                    break;
                }
            }
            AutoWoodAction::EnterGame((x, y)) => {
                if gameplay
                    || recognized.menu.is_some()
                    || recognized.enter_game.is_none()
                    || !bgi_input::is_foreground_window(host.hwnd)
                {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "enter_game_guard".into();
                    report.failure = Some("进入游戏前未确认登录界面和进入按钮或窗口已失焦".into());
                    break;
                }
                if !host.click_client_guarded(x, y) {
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "foreground".into();
                    report.failure = Some("移动至进入游戏按钮后窗口失焦，未点击".into());
                    break;
                }
            }
            AutoWoodAction::FinishUsed => {
                report.runs_completed = controller.wood_gain();
                host.on_log(
                    LogLevel::Info,
                    format!("自动砍树已验证木材收益，累计 {} 个", controller.wood_gain()),
                );
            }
            AutoWoodAction::Wait | AutoWoodAction::WaitLogin => {}
            AutoWoodAction::Succeeded => {
                if report.outcome != TaskOutcome::Success {
                    break;
                }
                report.runs_completed = controller.wood_gain();
                report.stage = "completed".into();
                report.failure = None;
                host.on_log(
                    LogLevel::Info,
                    if verify_gain {
                        format!(
                            "自动砍树完成：验证木材 {} 个，已重新进入游戏",
                            controller.wood_gain()
                        )
                    } else {
                        "自动砍树完成：未验证收益，未执行退出或重登".into()
                    },
                );
                break;
            }
            AutoWoodAction::Failed(reason) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = stage_before.into();
                report.failure = Some(reason.into());
                break;
            }
        }
        host.sleep_ms(AUTO_WOOD_TICK.as_millis() as u64);
    }
    // Every failing exit path keeps the frame that triggered it; the log line and the
    // task error both point at the dump so a post-mortem needs no second run.
    if matches!(report.outcome, TaskOutcome::Failed) {
        let reason = report
            .failure
            .clone()
            .unwrap_or_else(|| "自动砍树失败".into());
        match host.save_failure_frame("auto_wood_failed") {
            Ok(Some(path)) => {
                report.failure = Some(format!("{reason}（截图: {}）", path.display()));
            }
            Ok(None) => {}
            Err(error) => host.on_log(LogLevel::Warn, format!("自动砍树失败截图保存失败: {error}")),
        }
    }
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

#[allow(clippy::too_many_arguments)]
fn start_auto_wood_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
    verify_gain: bool,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    let task_config = auto_wood_task_config(&snapshot.config.task_limits);
    task_config.validate().map_err(|error| error.to_string())?;
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-auto-wood-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let events_tx = host.events_tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_auto_wood(&mut host, verify_gain, task_config)
            }))
            .unwrap_or_else(|_| TaskReport {
                task_id,
                task_name: AUTO_WOOD_TASK_NAME.into(),
                outcome: TaskOutcome::Failed,
                runs_completed: 0,
                runs_requested: 1,
                current_index: None,
                total_points: 1,
                failure: Some("自动砍树线程发生 panic".into()),
                stage: "panic".into(),
                elapsed_ms: 0,
            });
            bgi_input::release_all_keys();
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(result),
            });
        })
        .map_err(|e| format!("启动自动砍树线程失败: {e}"))?;
    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::AutoWood {
        task_id,
        task_name: AUTO_WOOD_TASK_NAME.into(),
    };
    core.emit_log(
        LogLevel::Info,
        "AutoWood",
        format!(
            "自动砍树已启动：固定 1 轮，总超时 {} 秒，验证收益={verify_gain}；失焦或意外菜单立即停止",
            task_config.total_timeout_ms / 1000
        ),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn start_open_current_chest_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    let task_config = open_chest_task_config(&snapshot.config.task_limits);
    task_config.validate().map_err(|error| error.to_string())?;
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-open-chest-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_open_current_chest(&mut host, task_config)
            }))
            .unwrap_or_else(|_| TaskReport {
                task_id,
                task_name: OPEN_CHEST_TASK_NAME.into(),
                outcome: TaskOutcome::Failed,
                runs_completed: 0,
                runs_requested: 1,
                current_index: None,
                total_points: 1,
                failure: Some("自动开箱线程发生 panic".into()),
                stage: "panic".into(),
                elapsed_ms: 0,
            });
            bgi_input::release_all_keys();
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(result),
            });
        })
        .map_err(|e| format!("启动自动开箱线程失败: {e}"))?;
    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::OpeningChest {
        task_id,
        task_name: OPEN_CHEST_TASK_NAME.into(),
    };
    core.emit_log(
        LogLevel::Info,
        "OpenChest",
        format!(
            "自动开箱当前目标已启动：仅处理当前视野单个箱子，{} 秒超时，失焦或菜单立即停止",
            task_config.total_timeout_ms / 1000
        ),
    );
    Ok(())
}

const FISHING_TASK_NAME: &str = "自动钓鱼";

#[allow(clippy::too_many_arguments)]
fn start_fishing_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
    dry_run: bool,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    snapshot.config.task_limits.validate()?;
    let task_config = FishingConfig {
        total_timeout_ms: snapshot.config.task_limits.fishing_total_timeout_ms,
    };
    task_config.validate().map_err(|error| error.to_string())?;
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = FishingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        pauses: trigger_runtime.task_pauses(),
    };
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-fishing-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                host.run_fishing_attempt(dry_run, task_config)
            }))
            .unwrap_or_else(|_| {
                Ok(TaskReport {
                    task_id,
                    task_name: FISHING_TASK_NAME.into(),
                    outcome: TaskOutcome::Failed,
                    runs_completed: 0,
                    runs_requested: 1,
                    current_index: None,
                    total_points: 1,
                    failure: Some("自动钓鱼线程发生 panic".into()),
                    stage: "panic".into(),
                    elapsed_ms: 0,
                })
            })
            .unwrap_or_else(|error| TaskReport {
                task_id,
                task_name: FISHING_TASK_NAME.into(),
                outcome: TaskOutcome::Cancelled,
                runs_completed: 0,
                runs_requested: 1,
                current_index: None,
                total_points: 1,
                failure: Some(error),
                stage: "cancelled".into(),
                elapsed_ms: 0,
            });
            if !dry_run {
                bgi_input::release_all_keys();
            }
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(result),
            });
        })
        .map_err(|e| format!("启动自动钓鱼线程失败: {e}"))?;
    *active_pathing = Some(ActiveTask::with_input_policy(
        task_id, cancel, join, !dry_run,
    ));
    trigger_runtime.set_task_mode(true);
    snapshot.config.fishing_dry_run = dry_run;
    snapshot.task_state = TaskState::Fishing {
        task_id,
        task_name: if dry_run {
            format!("{FISHING_TASK_NAME}（预演）")
        } else {
            FISHING_TASK_NAME.into()
        },
    };
    core.emit_log(
        LogLevel::Info,
        "Fishing",
        format!(
            "{}已启动：仅处理手动入竿后的单次尝试，总超时 {} 秒",
            if dry_run {
                "自动钓鱼预演"
            } else {
                "自动钓鱼"
            },
            task_config.total_timeout_ms / 1000
        ),
    );
    Ok(())
}

/// 启动自动采矿（莉奈娅）。识别与瞄准在 `mining_host` 内闭环，这里只做接线与状态发布。
#[allow(clippy::too_many_arguments)]
fn start_mining_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    ensure_task_inactive(snapshot)?;
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    snapshot.config.task_limits.validate()?;
    let limits = &snapshot.config.task_limits;
    let config = MiningRunConfig {
        mine_count: bgi_task::DEFAULT_MINE_COUNT,
        scan_rounds: bgi_task::DEFAULT_SCAN_ROUNDS,
        max_shots: u32::from(limits.mining_max_shots),
        total_timeout_ms: limits.mining_total_timeout_ms,
    };
    config.validate()?;

    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host = MiningHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        pauses: trigger_runtime.task_pauses(),
    };
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-mining-{task_id}"))
        .spawn(move || {
            let mut host = host;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let report =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| host.run_mining(config)))
                    .unwrap_or_else(|_| {
                        Ok(TaskReport {
                            task_id,
                            task_name: MINING_TASK_NAME.into(),
                            outcome: TaskOutcome::Failed,
                            runs_completed: 0,
                            runs_requested: config.mine_count,
                            current_index: None,
                            total_points: config.mine_count as usize,
                            failure: Some("自动采矿线程发生 panic".into()),
                            stage: "panic".into(),
                            elapsed_ms: 0,
                        })
                    })
                    .unwrap_or_else(|error| TaskReport {
                        task_id,
                        task_name: MINING_TASK_NAME.into(),
                        outcome: TaskOutcome::Cancelled,
                        runs_completed: 0,
                        runs_requested: config.mine_count,
                        current_index: None,
                        total_points: config.mine_count as usize,
                        failure: Some(error),
                        stage: "cancelled".into(),
                        elapsed_ms: 0,
                    });
            bgi_input::release_all_keys();
            let _ = events_tx.send(PathingEvent::Finished {
                task_id,
                result: Ok(report),
            });
        })
        .map_err(|e| format!("启动自动采矿线程失败: {e}"))?;

    *active_pathing = Some(ActiveTask::with_input_policy(task_id, cancel, join, true));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::Mining {
        task_id,
        task_name: MINING_TASK_NAME.into(),
    };
    core.emit_log(
        LogLevel::Info,
        "Mining",
        format!(
            "自动采矿已启动：挖矿 {} 次、射击上限 {} 次、总超时 {} 秒。\
             需要队伍中当前是莉奈娅并处在可瞄准状态；急停或切走窗口会立刻让步并松手。",
            config.mine_count,
            config.max_shots,
            config.total_timeout_ms / 1000
        ),
    );
    Ok(())
}

const SALVAGE_PREVIEW_TASK_NAME: &str = "分解预览（不分解）";

fn to_control_salvage_report(
    report: &bgi_task::SalvagePreviewReport,
) -> bgi_control::SalvagePreviewReport {
    use bgi_control::{
        SalvagePreviewCandidate, SalvagePreviewPosition, SalvagePreviewReport,
        SalvagePreviewSkipStats, SalvagePreviewStopReason,
    };
    SalvagePreviewReport {
        candidates: report
            .candidates
            .iter()
            .map(|candidate| SalvagePreviewCandidate {
                id: candidate.id,
                position: SalvagePreviewPosition {
                    page: candidate.position.page,
                    slot: candidate.position.slot,
                    row: candidate.position.row,
                    column: candidate.position.column,
                    x: candidate.position.x,
                    y: candidate.position.y,
                    width: candidate.position.width,
                    height: candidate.position.height,
                },
                star: candidate.star,
                confidence: candidate.confidence,
            })
            .collect(),
        skipped: SalvagePreviewSkipStats {
            empty: report.skipped.empty,
            locked: report.skipped.locked,
            selected: report.skipped.selected,
            five_star: report.skipped.five_star,
            overlapping: report.skipped.overlapping,
        },
        pages_seen: report.pages_seen,
        interactions_used: report.interactions_used,
        no_progress: report.no_progress,
        selection_clicks: report.selection_clicks,
        salvage_clicks: report.salvage_clicks,
        confirmation_clicks: report.confirmation_clicks,
        stop_reason: report.stop_reason.map(|reason| match reason {
            bgi_task::SalvagePreviewStopReason::Complete => SalvagePreviewStopReason::Complete,
            bgi_task::SalvagePreviewStopReason::EmptyPage => SalvagePreviewStopReason::EmptyPage,
            bgi_task::SalvagePreviewStopReason::PageLimit => SalvagePreviewStopReason::PageLimit,
            bgi_task::SalvagePreviewStopReason::NoProgress => SalvagePreviewStopReason::NoProgress,
            bgi_task::SalvagePreviewStopReason::InteractionBudget => {
                SalvagePreviewStopReason::InteractionBudget
            }
            bgi_task::SalvagePreviewStopReason::Failed => SalvagePreviewStopReason::Failed,
            bgi_task::SalvagePreviewStopReason::Cancelled => SalvagePreviewStopReason::Cancelled,
        }),
        failure: report.failure.clone(),
        final_confirmation_authorized: false,
    }
}

fn run_artifact_salvage_preview(
    host: &mut CorePathingHost,
) -> (TaskReport, bgi_control::SalvagePreviewReport) {
    use bgi_cv::artifact_salvage::ArtifactSalvageCardStatus;
    use bgi_task::{
        SalvageCardObservation, SalvagePreviewAction, SalvagePreviewConfig, SalvagePreviewFrame,
        SalvagePreviewStateMachine, SalvagePreviewStopReason,
    };
    let started = std::time::Instant::now();
    let mut state = SalvagePreviewStateMachine::new(SalvagePreviewConfig::default())
        .expect("default salvage preview configuration is valid");
    let mut report = TaskReport {
        task_id: host.task_id,
        task_name: SALVAGE_PREVIEW_TASK_NAME.into(),
        outcome: TaskOutcome::Success,
        runs_completed: 0,
        runs_requested: 1,
        current_index: None,
        total_points: 1,
        failure: None,
        stage: "waiting_for_salvage_page".into(),
        elapsed_ms: 0,
    };
    loop {
        if host.is_cancelled() {
            state.cancel();
            break;
        }
        let step = match host.detect_artifact_salvage() {
            Ok(Some(step)) => step,
            Ok(None) => {
                host.sleep_ms(50);
                continue;
            }
            Err(error) => {
                state.fail_closed(error.clone());
                report.outcome = TaskOutcome::Failed;
                report.stage = "capture".into();
                report.failure = Some(error);
                break;
            }
        };
        let observation = step.observation;
        let frame_size = (observation.width, observation.height);
        let Some(grid) = observation.grid else {
            report.outcome = TaskOutcome::Failed;
            report.stage = "aspect_ratio".into();
            report.failure = Some("分解预览仅支持 16:9 帧".into());
            state.fail_closed(report.failure.clone().unwrap());
            break;
        };
        let frame = SalvagePreviewFrame {
            width: observation.width,
            height: observation.height,
            salvage_page_visible: observation.salvage_page_visible(),
            final_confirmation_visible: observation.black_confirm_button.is_some(),
            fingerprint: grid.fingerprint,
            scroll_center: grid.scroll_center,
            cards: grid
                .cells
                .iter()
                .map(|cell| {
                    let matched = step.icon_matches[cell.index as usize].as_ref();
                    SalvageCardObservation {
                        slot: cell.index as usize,
                        rect: bgi_task::SalvageCellRect {
                            x: cell.rect.x,
                            y: cell.rect.y,
                            width: cell.rect.width,
                            height: cell.rect.height,
                        },
                        star: matched.map(|matched| matched.quality_level),
                        confidence: matched.map_or(0.0, |matched| f64::from(matched.score)),
                        presence_confidence: cell.presence_confidence,
                        name: matched.map_or_else(String::new, |matched| matched.name.clone()),
                        visual_id: matched.map_or(0, |matched| matched.visual_id),
                        present: cell.present,
                        locked: cell.status == ArtifactSalvageCardStatus::Locked,
                        selected: cell.status == ArtifactSalvageCardStatus::Selected,
                    }
                })
                .collect(),
        };
        report.stage = "observing".into();
        match state.observe(frame) {
            Ok(Some(SalvagePreviewAction::Scroll(scroll))) => {
                report.runs_completed = u32::from(state.report().pages_seen);
                if let Err(error) =
                    host.scroll_artifact_salvage_preview(scroll.center, scroll.notches, frame_size)
                {
                    state.fail_closed(error.clone());
                    report.outcome = TaskOutcome::Failed;
                    report.stage = "scroll_guard".into();
                    report.failure = Some(error);
                    break;
                }
                report.stage = "scrolling".into();
                host.sleep_ms(250);
            }
            Ok(None) if state.is_finished() => break,
            Ok(None) => {
                host.sleep_ms(50);
            }
            Err(error) => {
                report.outcome = TaskOutcome::Failed;
                report.stage = "fail_closed".into();
                report.failure = Some(match error {
                    bgi_task::SalvagePreviewError::LowConfidence { .. }
                    | bgi_task::SalvagePreviewError::UncertainPresence { .. } => {
                        format!("{error}；已停止，不会选择或分解任何卡片")
                    }
                    _ => error.to_string(),
                });
                break;
            }
        }
    }
    let preview = state.report();
    report.runs_completed = u32::from(preview.pages_seen);
    if report.failure.is_none() {
        match preview.stop_reason {
            Some(SalvagePreviewStopReason::Failed) => {
                report.outcome = TaskOutcome::Failed;
                report.failure = preview.failure.clone();
            }
            Some(SalvagePreviewStopReason::Cancelled) => {
                report.outcome = TaskOutcome::Cancelled;
                report.failure = preview.failure.clone();
            }
            Some(SalvagePreviewStopReason::NoProgress) => {
                report.outcome = TaskOutcome::Partial;
                report.stage = "no_progress".into();
            }
            Some(
                SalvagePreviewStopReason::PageLimit | SalvagePreviewStopReason::InteractionBudget,
            ) => {
                report.outcome = TaskOutcome::Partial;
                report.stage = "bounded_stop".into();
            }
            Some(SalvagePreviewStopReason::Complete | SalvagePreviewStopReason::EmptyPage) => {
                report.stage = "completed".into();
            }
            None => {
                report.outcome = TaskOutcome::Failed;
                report.failure = Some("分解预览异常结束".into());
            }
        }
    }
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    let preview = to_control_salvage_report(preview);
    (report, preview)
}

#[allow(clippy::too_many_arguments)]
fn start_artifact_salvage_preview_task(
    snapshot: &mut UiSnapshot,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    if active_pathing.is_some() || snapshot.task_state.is_task_active() {
        return Err("已有任务正在执行或停止".into());
    }
    let Some(win) = &snapshot.selected_window else {
        return Err("请先选择目标游戏窗口".into());
    };
    let task_id = allocate_task_id(next_task_id);
    let cancel = Arc::new(AtomicBool::new(false));
    let host_args = CorePathingHost {
        task_id,
        hwnd: win.hwnd,
        mode: snapshot.config.capture_mode,
        cancel: cancel.clone(),
        capturer: capturer.clone(),
        pool: pool.clone(),
        cv_ocr: cv_ocr.clone(),
        events_tx: pathing_tx.clone(),
        current_index: None,
        pauses: trigger_runtime.task_pauses(),
        input_pause: None,
        held_keys: Default::default(),
        right_mouse_held: false,
        mining_limits: crate::mining_host::MiningActionLimits::from(&snapshot.config.task_limits),
    };
    snapshot.last_salvage_preview_report = None;
    trigger_runtime.set_auto_pick_config(None);
    let join = std::thread::Builder::new()
        .name(format!("bgi-salvage-preview-{task_id}"))
        .spawn(move || {
            let mut host = host_args;
            let task_id = host.task_id;
            let events_tx = host.events_tx.clone();
            let (result, preview) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_artifact_salvage_preview(&mut host)
            }))
            .unwrap_or_else(|_| {
                let mut state = bgi_task::SalvagePreviewStateMachine::new(Default::default())
                    .expect("default salvage preview configuration is valid");
                state.fail_closed("分解预览线程发生 panic".into());
                (
                    TaskReport {
                        task_id,
                        task_name: SALVAGE_PREVIEW_TASK_NAME.into(),
                        outcome: TaskOutcome::Failed,
                        runs_completed: 0,
                        runs_requested: 1,
                        current_index: None,
                        total_points: 1,
                        failure: Some("分解预览线程发生 panic".into()),
                        stage: "panic".into(),
                        elapsed_ms: 0,
                    },
                    to_control_salvage_report(state.report()),
                )
            });
            let _ = events_tx.send(PathingEvent::SalvagePreviewFinished {
                task_id,
                result: Ok(result),
                preview: Some(preview),
            });
        })
        .map_err(|e| format!("启动分解预览线程失败: {e}"))?;
    *active_pathing = Some(ActiveTask::new(task_id, cancel, join));
    trigger_runtime.set_task_mode(true);
    snapshot.task_state = TaskState::SalvagePreview {
        task_id,
        task_name: SALVAGE_PREVIEW_TASK_NAME.into(),
    };
    core.emit_log(
        LogLevel::Info,
        "ArtifactSalvage",
        "分解预览已启动：仅观察与滚屏，不选择、不分解、不确认；请手动打开分解界面",
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_command(
    cmd: CoreCommand,
    snapshot: &mut UiSnapshot,
    raw_hub: &Option<RawInputHub>,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    frame_seq: &mut u64,
    dbg_seq: &mut u64,
    pace: &mut CapturePaceState,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    match cmd {
        CoreCommand::RefreshWindows => {
            ensure_task_inactive(snapshot)?;
            let pref = snapshot.config.preferred_process.clone();
            let list = tokio::task::spawn_blocking(move || enumerate_candidate_windows(&pref))
                .await
                .map_err(|e| e.to_string())?;
            if let Some(sel) = &snapshot.selected_window {
                snapshot.selected_window = list.iter().find(|w| w.hwnd == sel.hwnd).cloned();
            }
            if snapshot.selected_window.is_none() {
                snapshot.selected_window = list
                    .iter()
                    .find(|w| {
                        w.process_name
                            .eq_ignore_ascii_case(&snapshot.config.preferred_process)
                    })
                    .cloned();
            }
            snapshot.available_windows = list;
            core.emit_log(
                LogLevel::Info,
                "Capture",
                format!("已刷新桌面候选窗口 {} 个", snapshot.available_windows.len()),
            );
            Ok(())
        }
        CoreCommand::SelectWindow { hwnd } => {
            ensure_task_inactive(snapshot)?;
            {
                let mut cap_guard = capturer.lock().unwrap();
                close_capturer(&mut cap_guard, core);
            }
            snapshot.selected_window = hwnd.and_then(|id| {
                snapshot
                    .available_windows
                    .iter()
                    .find(|w| w.hwnd == id)
                    .cloned()
            });
            if let Some(w) = &snapshot.selected_window {
                core.emit_log(
                    LogLevel::Info,
                    "Capture",
                    format!("已绑定窗口: {} ({}x{})", w.title, w.width, w.height),
                );
            }
            Ok(())
        }
        CoreCommand::SetCaptureMode(mode) => {
            ensure_task_inactive(snapshot)?;
            {
                let mut cap_guard = capturer.lock().unwrap();
                close_capturer(&mut cap_guard, core);
            }
            snapshot.config.capture_mode = mode;
            core.emit_log(
                LogLevel::Info,
                "Capture",
                format!("切换截图后端 -> {}", mode.label()),
            );
            Ok(())
        }
        CoreCommand::SetPreviewEnabled(enabled) => {
            if enabled && snapshot.selected_window.is_none() {
                return Err("请先在「抓图预览」中选择一个目标窗口".to_string());
            }
            snapshot.preview_enabled = enabled;
            if !enabled {
                let mut cap_guard = capturer.lock().unwrap();
                release_idle_capturer(snapshot, &mut cap_guard, core);
            }
            if !snapshot.task_state.is_task_active() {
                snapshot.task_state = if enabled {
                    TaskState::Previewing
                } else {
                    TaskState::Idle
                };
            }
            Ok(())
        }
        CoreCommand::SetCaptureNoRender(enabled) => {
            snapshot.config.capture_only_no_render = enabled;
            if enabled {
                core.clear_preview();
                core.emit_log(
                    LogLevel::Info,
                    "Capture",
                    "已开启「只捕获不渲染」：跳过 BGRA->RGBA 缩放、纹理上传与预览绘制",
                );
            } else {
                core.emit_log(LogLevel::Info, "Capture", "已恢复实时预览画面渲染");
            }
            Ok(())
        }
        CoreCommand::SetPreviewFps(fps) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.preview_fps = fps.clamp(1, 60);
            let interval_ms = (1000 / snapshot.config.preview_fps).max(16);
            let mut cap_guard = capturer.lock().unwrap();
            let reopen = cap_guard
                .as_ref()
                .is_some_and(|(_, _, backend)| backend.needs_reopen_for_interval(interval_ms));
            if reopen {
                close_capturer(&mut cap_guard, core);
            } else if let Some((_, _, backend)) = cap_guard.as_ref() {
                backend.set_min_update_interval_ms(interval_ms);
            }
            Ok(())
        }
        CoreCommand::CaptureSingleFrame => {
            ensure_task_inactive(snapshot)?;
            let mut cap_guard = capturer.lock().unwrap();
            let res = capture_once(snapshot, &mut cap_guard, pool, frame_seq, pace, core);
            release_idle_capturer(snapshot, &mut cap_guard, core);
            res
        }
        CoreCommand::StartFrameRecording(path) => {
            ensure_task_inactive(snapshot)?;
            trigger_runtime.start_recording(path)
        }
        CoreCommand::StopFrameRecording => {
            ensure_task_inactive(snapshot)?;
            trigger_runtime.stop_recording()
        }
        CoreCommand::StartFrameReplay(path) => {
            ensure_task_inactive(snapshot)?;
            trigger_runtime.set_replay(path)
        }
        CoreCommand::StopFrameReplay => {
            trigger_runtime.clear_replay();
            Ok(())
        }
        CoreCommand::SetPathingDir(dir) => {
            snapshot.config.pathing_dir = dir.clone();
            let paths =
                tokio::task::spawn_blocking(move || bgi_pathing::scan_path_directory(dir, 500))
                    .await
                    .map_err(|e| e.to_string())?;
            snapshot.paths = paths;
            snapshot.active_path_file = snapshot.paths.first().map(|p| p.file_path.clone());
            core.emit_log(
                LogLevel::Info,
                "Pathing",
                format!("已扫描 {} 条 .json 路径", snapshot.paths.len()),
            );
            Ok(())
        }
        CoreCommand::SetPathingRuns(runs) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.pathing_runs = runs;
            Ok(())
        }
        CoreCommand::SetPathingLoopDelayMs(delay_ms) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.pathing_loop_delay_ms = delay_ms;
            Ok(())
        }
        CoreCommand::SetPathingWaitForLoading(enabled) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.pathing_wait_for_loading = enabled;
            Ok(())
        }
        CoreCommand::SetPathingAutoPick(enabled) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.pathing_auto_pick = enabled;
            Ok(())
        }
        CoreCommand::SetAutoSkipEnabled(enabled) => {
            snapshot.config.auto_skip = enabled;
            trigger_runtime.set_builtin_config(snapshot.config.auto_skip, snapshot.config.auto_eat);
            Ok(())
        }
        CoreCommand::SetAutoEatEnabled(enabled) => {
            snapshot.config.auto_eat = enabled;
            trigger_runtime.set_builtin_config(snapshot.config.auto_skip, snapshot.config.auto_eat);
            Ok(())
        }
        CoreCommand::SetAutoCookIntervalMs(interval_ms) => {
            ensure_task_inactive(snapshot)?;
            let config = AutoCookTaskConfig {
                check_interval_ms: interval_ms,
                ..auto_cook_task_config(snapshot)
            };
            config.validate().map_err(|error| error.to_string())?;
            snapshot.config.auto_cook_interval_ms = interval_ms;
            Ok(())
        }
        CoreCommand::SetAutoCookStopOnRecover(enabled) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.auto_cook_stop_on_recover = enabled;
            Ok(())
        }
        CoreCommand::SetAutoCookTimeoutMs(timeout_ms) => {
            ensure_task_inactive(snapshot)?;
            let config = AutoCookTaskConfig {
                total_timeout_ms: timeout_ms,
                ..auto_cook_task_config(snapshot)
            };
            config.validate().map_err(|error| error.to_string())?;
            snapshot.config.auto_cook_timeout_ms = timeout_ms;
            Ok(())
        }
        CoreCommand::SetTaskLimits(limits) => {
            ensure_task_inactive(snapshot)?;
            limits.validate()?;
            snapshot.config.task_limits = limits;
            Ok(())
        }
        CoreCommand::SetFishingDryRun(enabled) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.fishing_dry_run = enabled;
            Ok(())
        }
        CoreCommand::SetAssetsDir(dir) => {
            ensure_task_inactive(snapshot)?;
            snapshot.config.assets_dir = dir.clone();
            cv_ocr.lock().unwrap().set_assets_dir(PathBuf::from(&dir));
            core.emit_log(
                LogLevel::Info,
                "CV/OCR",
                format!("Assets 目录已切换至: {dir}"),
            );
            Ok(())
        }
        CoreCommand::RefreshPaths => {
            let dir = snapshot.config.pathing_dir.clone();
            if dir.trim().is_empty() {
                return Err("请先填写路径目录".to_string());
            }
            let paths =
                tokio::task::spawn_blocking(move || bgi_pathing::scan_path_directory(dir, 500))
                    .await
                    .map_err(|e| e.to_string())?;
            snapshot.paths = paths;
            Ok(())
        }
        CoreCommand::SelectPath { file_path } => {
            snapshot.active_path_file = Some(file_path);
            Ok(())
        }
        CoreCommand::ProbeMiniMapCv => {
            let res = run_cv_probe(
                snapshot,
                capturer,
                pool,
                dbg_seq,
                cv_ocr,
                core,
                CvOcrServices::probe_minimap,
            )
            .await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[小地图测试] pos=({:?}, {:?}) cam={:?} char={:?}",
                    res.game_x, res.game_y, res.camera_angle, res.character_angle
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeBigMapSift => {
            let res = run_cv_probe(
                snapshot,
                capturer,
                pool,
                dbg_seq,
                cv_ocr,
                core,
                CvOcrServices::probe_bigmap_sift,
            )
            .await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[大地图 SIFT 测试] pos=({:?}, {:?}) {} elapsed={:.1}ms",
                    res.game_x,
                    res.game_y,
                    res.source_label,
                    res.elapsed_ms.unwrap_or(0.0)
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeOcrScreen => {
            let res = run_cv_probe(
                snapshot,
                capturer,
                pool,
                dbg_seq,
                cv_ocr,
                core,
                CvOcrServices::probe_ocr,
            )
            .await?;
            core.emit_log(
                LogLevel::Info,
                "OCR",
                format!(
                    "[PaddleOCR] {} 耗时 {:.1}ms",
                    res.source_label,
                    res.elapsed_ms.unwrap_or(0.0)
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeWeaponDetails => {
            let res = run_weapon_probe(snapshot, capturer, pool, dbg_seq, cv_ocr, core).await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[武器详情只读识别] {} | {}",
                    res.source_label,
                    res.ocr_text.as_deref().unwrap_or("-")
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeTalentDetails => {
            let res = run_talent_probe(snapshot, capturer, pool, dbg_seq, cv_ocr, core).await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[天赋详情只读识别] {} | {}",
                    res.source_label,
                    res.ocr_text.as_deref().unwrap_or("-")
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeCharacterAttributes => {
            let res =
                run_character_attribute_probe(snapshot, capturer, pool, dbg_seq, cv_ocr, core)
                    .await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[角色属性只读识别] {} | {}",
                    res.source_label,
                    res.ocr_text.as_deref().unwrap_or("-")
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeRewardResultCards => {
            let res =
                run_reward_result_probe(snapshot, capturer, pool, dbg_seq, cv_ocr, core).await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[奖励结果页只读识别] {} | {}",
                    res.source_label,
                    res.ocr_text.as_deref().unwrap_or("-")
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeExpeditionStatus => {
            let res = run_expedition_probe(snapshot, capturer, pool, dbg_seq, cv_ocr, core).await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[派遣页面只读识别] {} | {}",
                    res.source_label,
                    res.ocr_text.as_deref().unwrap_or("-")
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeMining => {
            let res = run_mining_probe(snapshot, capturer, pool, dbg_seq, cv_ocr, core).await?;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!(
                    "[自动采矿只读识别] {} | {}",
                    res.source_label,
                    res.ocr_text.as_deref().unwrap_or("-")
                ),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::BeginHotkeyCapture(action) => {
            snapshot.capturing_hotkey_for = action;
            if let Some(hub) = raw_hub {
                hub.begin_capture(action);
            }
            Ok(())
        }
        CoreCommand::UpdateHotkey(binding) => {
            if let Some(slot) = snapshot
                .config
                .hotkeys
                .iter_mut()
                .find(|b| b.action == binding.action)
            {
                *slot = binding;
            } else {
                snapshot.config.hotkeys.push(binding);
            }
            if let Some(hub) = raw_hub {
                hub.set_bindings(snapshot.config.hotkeys.clone());
            }
            Ok(())
        }
        CoreCommand::TestInputKey { vk } => {
            ensure_task_inactive(snapshot)?;
            tokio::task::spawn_blocking(move || bgi_input::key_tap(vk, 25))
                .await
                .map_err(|e| e.to_string())?;
            core.emit_log(
                LogLevel::Info,
                "Input",
                format!("已通过 SendInput 发送测试按键 {}", bgi_control::vk_name(vk)),
            );
            Ok(())
        }
        CoreCommand::StartBasicPathing => start_basic_pathing_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
        ),
        CoreCommand::StartAutoCook => start_auto_cook_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
        ),
        CoreCommand::ClaimCurrentPageRewards => start_reward_claim_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
            RewardClaimMode::CurrentPage,
        ),
        CoreCommand::ClaimVisiblePageRewards => start_reward_claim_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
            RewardClaimMode::VisiblePageContinuous,
        ),
        CoreCommand::StartOneKeyExpedition => start_one_key_expedition_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
        ),
        CoreCommand::OpenCurrentChest => start_open_current_chest_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
        ),
        CoreCommand::StartAutoWood { verify_gain } => start_auto_wood_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
            verify_gain,
        ),
        CoreCommand::StartArtifactSalvagePreview => start_artifact_salvage_preview_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
        ),
        CoreCommand::StartFishing { dry_run } => start_fishing_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
            dry_run,
        ),
        CoreCommand::StartMining => start_mining_task(
            snapshot,
            capturer,
            pool,
            cv_ocr,
            trigger_runtime,
            active_pathing,
            next_task_id,
            pathing_tx,
            core,
        ),
        CoreCommand::StopPathing => {
            stop_and_join_active_pathing(snapshot, active_pathing, core, "已请求停止当前任务")
                .await?;
            trigger_runtime.set_task_mode(false);
            let mut cap_guard = capturer.lock().unwrap();
            release_idle_capturer(snapshot, &mut cap_guard, core);
            Ok(())
        }
        CoreCommand::EmergencyStop => {
            snapshot.preview_enabled = false;
            stop_and_join_active_pathing(
                snapshot,
                active_pathing,
                core,
                "急停中断正在执行的寻路任务",
            )
            .await?;
            trigger_runtime.set_task_mode(false);
            {
                let mut cap_guard = capturer.lock().unwrap();
                close_capturer(&mut cap_guard, core);
            }
            tokio::task::spawn_blocking(bgi_input::release_all_keys)
                .await
                .map_err(|e| e.to_string())?;
            core.emit_log(LogLevel::Warn, "Core", "已执行急停并释放所有键盘/鼠标按键");
            Ok(())
        }
        CoreCommand::OpenLogDir => {
            let dir = crate::logging::open_log_dir()?;
            core.emit_log(
                LogLevel::Info,
                "Core",
                format!("已打开日志目录: {}", dir.display()),
            );
            Ok(())
        }
        CoreCommand::ShutdownCore => {
            snapshot.preview_enabled = false;
            stop_and_join_active_pathing(snapshot, active_pathing, core, "核心退出，终止寻路任务")
                .await?;
            trigger_runtime.set_task_mode(false);
            core.emit_log(LogLevel::Info, "Core", "正在关闭后台核心...");
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_raw_signal(
    sig: RawInputSignal,
    snapshot: &mut UiSnapshot,
    raw_hub: &Option<RawInputHub>,
    capturer: &Arc<Mutex<ActiveCapturer>>,
    pool: &Arc<FrameBufferPool>,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    trigger_runtime: &mut crate::trigger_runtime::TriggerRuntime,
    active_pathing: &mut Option<ActiveTask>,
    next_task_id: &mut u64,
    pathing_tx: &mpsc::UnboundedSender<PathingEvent>,
    core: &CoreEndpoint,
) {
    match sig {
        RawInputSignal::KeyObserved {
            vk,
            scan_code,
            extended,
            pressed,
            combo,
        } => {
            if pressed {
                snapshot.last_raw_key = Some(combo.clone());
            }
            core.emit_event(CoreEvent::RawKeyObserved {
                vk,
                scan_code,
                extended,
                pressed,
                combo,
            });
        }
        RawInputSignal::HotkeyCaptured(binding) => {
            snapshot.capturing_hotkey_for = None;
            if let Some(slot) = snapshot
                .config
                .hotkeys
                .iter_mut()
                .find(|b| b.action == binding.action)
            {
                *slot = binding.clone();
            } else {
                snapshot.config.hotkeys.push(binding.clone());
            }
            if let Some(hub) = raw_hub {
                hub.set_bindings(snapshot.config.hotkeys.clone());
            }
            core.emit_log(
                LogLevel::Info,
                "RawInputHub",
                format!(
                    "已绑定热键「{}」-> {}",
                    binding.action.label(),
                    binding.display_combo()
                ),
            );
        }
        RawInputSignal::HotkeyTriggered { action, combo } => {
            core.emit_event(CoreEvent::HotkeyTriggered {
                action,
                combo: combo.clone(),
            });
            core.emit_log(
                LogLevel::Info,
                "Hotkey",
                format!("命中全局热键 [{combo}] -> {}", action.label()),
            );
            match action {
                HotkeyAction::EmergencyStop => {
                    snapshot.preview_enabled = false;
                    stop_active_pathing(snapshot, active_pathing, core, "全局急停中断寻路");
                    std::thread::spawn(bgi_input::release_all_keys);
                }
                HotkeyAction::TogglePreview => {
                    if snapshot.selected_window.is_some() {
                        snapshot.preview_enabled = !snapshot.preview_enabled;
                        if !snapshot.preview_enabled {
                            let mut cap_guard = capturer.lock().unwrap();
                            release_idle_capturer(snapshot, &mut cap_guard, core);
                        }
                        if !snapshot.task_state.is_task_active() {
                            snapshot.task_state = if snapshot.preview_enabled {
                                TaskState::Previewing
                            } else {
                                TaskState::Idle
                            };
                        }
                    }
                }
                HotkeyAction::StartStopPathing => {
                    if active_pathing.is_some() {
                        stop_active_pathing(snapshot, active_pathing, core, "热键触发停止寻路");
                    } else if let Err(err) = start_basic_pathing_task(
                        snapshot,
                        capturer,
                        pool,
                        cv_ocr,
                        trigger_runtime,
                        active_pathing,
                        next_task_id,
                        pathing_tx,
                        core,
                    ) {
                        core.emit_log(LogLevel::Warn, "Pathing", format!("无法启动寻路: {err}"));
                    }
                }
                HotkeyAction::TestPing => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgi_task::MiningObservation;

    fn hit(text: &str, x: i32, y: i32) -> bgi_pathing::OcrHit {
        bgi_pathing::OcrHit {
            text: text.into(),
            center_x: x,
            center_y: y,
            confidence: 0.9,
        }
    }

    #[test]
    fn weapon_probe_reports_only_stable_complete_results() {
        let details = bgi_task::WeaponDetails::try_new("西风剑".into(), 20, 90).unwrap();
        let ready =
            weapon_probe_result(7, 3, WeaponRecognitionState::Ready(details.clone()), 321.0);
        assert_eq!(ready.updated_seq, 7);
        assert_eq!(ready.source_label, "武器详情（三次稳定识别）");
        assert_eq!(
            ready.ocr_text.as_deref(),
            Some("名称：西风剑\n当前等级：20\n等级上限：90\n采样次数：3")
        );
        assert_eq!(ready.elapsed_ms, Some(321.0));

        let failed = weapon_probe_result(8, 10, WeaponRecognitionState::Failed, 2000.0);
        assert_eq!(failed.updated_seq, 8);
        assert!(failed.source_label.contains("10 次采样未稳定"));
        let text = failed.ocr_text.unwrap();
        assert!(text.contains("未获得稳定结果"));
        assert!(!text.contains("西风剑"));
        assert!(text.contains("不点击、不升级、不分解"));
    }

    #[test]
    fn weapon_probe_rejects_active_and_stopping_tasks() {
        for task_state in [
            TaskState::ClaimingRewards {
                task_id: 1,
                task_name: "领取".into(),
            },
            TaskState::AutoWood {
                task_id: 1,
                task_name: "砍树".into(),
            },
            TaskState::OneKeyExpedition {
                task_id: 1,
                task_name: "派遣".into(),
            },
            TaskState::Stopping {
                task_id: 2,
                task_name: "停止中".into(),
                current_index: 0,
                total_points: 1,
            },
        ] {
            let snapshot = UiSnapshot {
                task_state,
                ..Default::default()
            };
            assert!(ensure_task_inactive(&snapshot).is_err());
        }
    }

    #[test]
    fn talent_probe_reports_only_stable_complete_results() {
        let details =
            bgi_task::TalentDetails::try_new(bgi_task::TalentType::Skill, 10, true).unwrap();
        let ready = talent_probe_result(9, 3, TalentRecognitionState::Ready(details), 456.0);
        assert_eq!(ready.updated_seq, 9);
        assert_eq!(ready.source_label, "天赋详情（三次稳定识别）");
        assert_eq!(
            ready.ocr_text.as_deref(),
            Some("天赋类型：元素战技\n显示等级：10\n天赋等级+3：是\n采样次数：3")
        );
        assert_eq!(ready.elapsed_ms, Some(456.0));

        let failed = talent_probe_result(10, 10, TalentRecognitionState::Failed, 2000.0);
        assert!(failed.source_label.contains("10 次采样未稳定"));
        let text = failed.ocr_text.unwrap();
        assert!(text.contains("未获得稳定结果"));
        assert!(text.contains("异常非空文本会拒绝结果"));
        assert!(!text.contains("元素战技"));
        assert!(text.contains("不点击、不切换天赋、不升级"));
    }

    #[test]
    fn character_attribute_probe_reports_only_stable_complete_results() {
        let details = bgi_task::CharacterAttributeDetails::try_new(90, 90).unwrap();
        let ready = character_attribute_probe_result(
            11,
            3,
            CharacterAttributeRecognitionState::Ready(details),
            500.0,
        );
        assert_eq!(ready.updated_seq, 11);
        assert_eq!(ready.source_label, "角色属性（三次稳定识别）");
        assert_eq!(
            ready.ocr_text.as_deref(),
            Some("当前等级：90\n等级上限：90\n采样次数：3")
        );
        assert_eq!(ready.elapsed_ms, Some(500.0));

        let failed = character_attribute_probe_result(
            12,
            10,
            CharacterAttributeRecognitionState::Failed,
            2000.0,
        );
        assert!(failed.source_label.contains("10 次采样未稳定"));
        let text = failed.ocr_text.unwrap();
        assert!(text.contains("未获得稳定结果"));
        assert!(text.contains("连续 3 次"));
        assert!(!text.contains("当前等级：90"));
        assert!(text.contains("不点击、不切换角色、不升级"));
    }

    #[test]
    fn reward_result_probe_displays_unknown_counts_without_guessing() {
        let page = bgi_task::RewardPageObservation(vec![bgi_task::RewardCardObservation {
            rect: bgi_task::RewardCardRect {
                x: 220,
                y: 444,
                width: 125,
                height: 153,
            },
            identity: Some(bgi_task::RewardIdentity {
                name: "材料".into(),
                quality_level: None,
            }),
            count: None,
        }]);
        let ready =
            reward_result_probe_result(13, 3, RewardResultRecognitionState::Ready(page), 600.0);
        assert_eq!(
            ready.ocr_text.as_deref(),
            Some("卡片数量：1\n采样次数：3\n#1：材料 × 未知")
        );
        let failed =
            reward_result_probe_result(14, 10, RewardResultRecognitionState::Failed, 2000.0);
        let text = failed.ocr_text.unwrap();
        assert!(text.contains("绝不默认按 1 累计"));
        assert!(text.contains("不点击、不翻页、不领取、不按 ESC"));
    }

    #[test]
    fn expedition_probe_reports_only_explicit_stable_icons() {
        let ready = expedition_probe_result(
            15,
            3,
            ExpeditionRecognitionState::Ready(bgi_task::ExpeditionPageState::CollectAvailable),
            600.0,
        );
        assert_eq!(
            ready.ocr_text.as_deref(),
            Some("检测到全部领取图标\n采样次数：3")
        );
        let failed = expedition_probe_result(16, 10, ExpeditionRecognitionState::Failed, 2000.0);
        let text = failed.ocr_text.unwrap();
        assert!(text.contains("不等于“无奖励”或“零完成”"));
        assert!(text.contains("不点击、不领取、不翻页、不按 ESC"));
    }

    #[test]
    fn mining_probe_distinguishes_stable_target_empty_frame_and_unknown() {
        let target = bgi_task::MiningTarget {
            rect: bgi_task::MiningRect {
                x: 900,
                y: 400,
                width: 40,
                height: 30,
            },
            center_x: 920.0,
            center_y: 415.0,
        };
        let ready = mining_probe_result(
            17,
            3,
            MiningRecognitionState::Ready(MiningObservation {
                frame_width: 1920,
                frame_height: 1080,
                cluster_count: 1,
                target: Some(target),
            }),
            750.0,
        );
        assert_eq!(ready.source_label, "自动采矿（三次稳定识别）");
        let text = ready.ocr_text.unwrap();
        assert!(text.contains("检测到矿石目标（仅识别）"));
        assert!(text.contains("目标框：x=900, y=400, 40×30"));
        assert!(text.contains("目标中心：(920.0, 415.0)"));

        let empty = mining_probe_result(
            18,
            3,
            MiningRecognitionState::Ready(MiningObservation {
                frame_width: 1920,
                frame_height: 1080,
                cluster_count: 0,
                target: None,
            }),
            700.0,
        );
        let text = empty.ocr_text.unwrap();
        assert!(text.contains("连续 3 帧未检测到"));
        assert!(text.contains("不保证镜头外没有矿石"));

        let unknown = mining_probe_result(
            19,
            1,
            MiningRecognitionState::Unknown("模型加载或推理失败，结果未知"),
            20.0,
        );
        let text = unknown.ocr_text.unwrap();
        assert!(text.contains("未知不等于无矿"));
        assert!(text.contains("bgi_mine.onnx"));
        assert!(text.contains("不聚焦、不移动鼠标、不点击、不切换角色、不发送按键"));
    }

    #[test]
    fn expedition_automation_never_converts_ambiguous_icons_to_a_click() {
        let collect = bgi_cv::MatchPoint {
            x: 120.4,
            y: 800.6,
            score: 0.95,
        };
        let redispatch = bgi_cv::MatchPoint {
            x: 1200.4,
            y: 930.6,
            score: 0.95,
        };
        let observation = bgi_cv::expedition::ExpeditionObservation {
            width: 1920,
            height: 1080,
            collect: Some(collect),
            redispatch: None,
        };
        assert_eq!(
            expedition_automation_target(observation),
            Some(OneKeyExpeditionTarget {
                state: bgi_task::ExpeditionPageState::CollectAvailable,
                x: 120,
                y: 801,
            })
        );
        let ambiguous = bgi_cv::expedition::ExpeditionObservation {
            collect: Some(collect),
            redispatch: Some(redispatch),
            ..observation
        };
        assert_eq!(expedition_automation_target(ambiguous), None);
        let unknown = bgi_cv::expedition::ExpeditionObservation {
            collect: None,
            redispatch: None,
            ..observation
        };
        assert_eq!(expedition_automation_target(unknown), None);
    }

    #[test]
    fn wood_ocr_roi_matches_csharp_scaling_and_clamps_to_frame() {
        let (x0, y0, x1, y1) = wood_ocr_roi(1920, 1080);
        assert!((x0 - 100.0 / 1920.0).abs() < f32::EPSILON);
        assert!((y0 - 450.0 / 1080.0).abs() < f32::EPSILON);
        assert!((x1 - 400.0 / 1920.0).abs() < f32::EPSILON);
        assert!((y1 - 700.0 / 1080.0).abs() < f32::EPSILON);
        let (x0, y0, x1, y1) = wood_ocr_roi(960, 540);
        assert!((x0 - 50.0 / 960.0).abs() < f32::EPSILON);
        assert!((y0 - 225.0 / 540.0).abs() < f32::EPSILON);
        assert!((x1 - 200.0 / 960.0).abs() < f32::EPSILON);
        assert!((y1 - 350.0 / 540.0).abs() < f32::EPSILON);
        let (x0, y0, x1, y1) = wood_ocr_roi(2560, 1440);
        assert!((x0 - 100.0 / 2560.0).abs() < f32::EPSILON);
        assert!((y0 - 450.0 / 1440.0).abs() < f32::EPSILON);
        assert!((x1 - 400.0 / 2560.0).abs() < f32::EPSILON);
        assert!((y1 - 700.0 / 1440.0).abs() < f32::EPSILON);
    }

    #[test]
    fn wood_ocr_text_is_reading_order_and_whitelist_parser_rejects_other_items() {
        let hits = vec![
            hit("木材x2", 120, 510),
            hit("获得", 120, 480),
            hit("铁矿×9", 120, 550),
        ];
        let text = wood_ocr_text(&hits);
        assert_eq!(text, "获得\n木材x2\n铁矿×9");
        assert_eq!(parse_wood_gain(&text), Some(2));
        assert_eq!(parse_wood_gain("铁矿×9"), None);
    }
}
