use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bgi_capture::{CapturedFrame, FrameBufferPool, enumerate_candidate_windows, frame_to_preview};
use bgi_control::{
    CoreCommand, CoreConfig, CoreEndpoint, LogLevel, RuntimeMetricsSnapshot, StageMetricsSnapshot,
    TaskOutcome, TaskState, UiSnapshot,
};
use bgi_input::{RawInputHub, RawInputSignal};
use tokio::sync::mpsc;

use crate::capture_host::{ActiveCapturer, CapturePaceState, capture_frame, release_idle_capturer};
use crate::commands::{handle_command, handle_raw_signal};
use crate::cv_service::CvOcrServices;
use crate::pathing_host::PathingEvent;
use crate::task_runtime::{ActiveTask, accepts_event};
use crate::trigger_runtime::{AutoPickExecutionContext, TriggerRuntime};
use bgi_task::{
    AUTO_PICK_MAX_OBSERVATION_AGE, AutoPickGateState, FrameScheduler, PipelineMetrics,
    PipelineStage, UiCategory,
};

struct PendingAutoPick {
    context: AutoPickExecutionContext,
    join: tokio::task::JoinHandle<Result<Vec<bgi_pathing::OcrHit>, String>>,
    started: std::time::Instant,
}

pub struct CoreHandle {
    join: Option<std::thread::JoinHandle<()>>,
}

impl CoreHandle {
    pub fn join(mut self) {
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

pub fn spawn_core_actor(
    core: CoreEndpoint,
    initial_snapshot: UiSnapshot,
    persisted_config_path: PathBuf,
) -> CoreHandle {
    let join = std::thread::Builder::new()
        .name("bgi-core-rt".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(run_core_loop(core, initial_snapshot, persisted_config_path));
        })
        .expect("spawn core thread");

    CoreHandle { join: Some(join) }
}

fn idle_or_previewing(preview_enabled: bool) -> TaskState {
    if preview_enabled {
        TaskState::Previewing
    } else {
        TaskState::Idle
    }
}

fn stage_snapshot(stats: &bgi_task::scheduler::StageStats) -> StageMetricsSnapshot {
    StageMetricsSnapshot {
        samples: stats.samples,
        total_ms: stats.total.as_secs_f32() * 1000.0,
        max_ms: stats.max.as_secs_f32() * 1000.0,
    }
}

fn metrics_snapshot(
    scheduler: &FrameScheduler,
    pipeline: &PipelineMetrics,
) -> RuntimeMetricsSnapshot {
    let stats = scheduler.stats();
    RuntimeMetricsSnapshot {
        submitted: stats.submitted,
        completed: stats.completed,
        dropped: stats.dropped,
        deadline_hits: stats.deadline_hits,
        deadline_misses: stats.deadline_misses,
        deadline_hit_rate: stats.deadline_hit_rate(),
        average_queue_delay_us: stats.average_queue_delay().as_micros() as u64,
        max_queue_delay_us: stats.queue_delay_max_us as u64,
        capture: stage_snapshot(pipeline.stage(PipelineStage::Capture)),
        recognition: stage_snapshot(pipeline.stage(PipelineStage::Recognition)),
        decision: stage_snapshot(pipeline.stage(PipelineStage::Decision)),
        input: stage_snapshot(pipeline.stage(PipelineStage::Input)),
    }
}

fn persist_config(
    path: &Path,
    previous: &CoreConfig,
    current: &CoreConfig,
    core: &CoreEndpoint,
) -> Result<(), String> {
    if previous == current {
        return Ok(());
    }
    crate::config_store::save_config(path, current).map_err(|error| {
        core.emit_log(
            LogLevel::Error,
            "Config",
            format!("设置已在本次运行生效，但保存失败: {error}"),
        );
        error
    })
}

async fn run_core_loop(
    mut core: CoreEndpoint,
    initial_snapshot: UiSnapshot,
    persisted_config_path: PathBuf,
) {
    let mut snapshot = initial_snapshot;

    if snapshot.config.pathing_dir.is_empty() {
        for candidate in [
            r"d:\bgiprro\bettergi-scripts-list\repo\pathing",
            r"d:\bgiprro\better-genshin-impact\BetterGenshinImpact\User\AutoPathing",
        ] {
            if PathBuf::from(candidate).is_dir() {
                snapshot.config.pathing_dir = candidate.to_string();
                break;
            }
        }
    }

    if snapshot.config.assets_dir.is_empty() {
        for candidate in [
            r"d:\bgiprro\better-genshin-impact\BetterGenshinImpact\bin\Debug\net8.0-windows10.0.22621.0\Assets",
            r"d:\bgiprro\bgi_publish_new\Assets",
            r"Assets",
        ] {
            if PathBuf::from(candidate).is_dir() {
                snapshot.config.assets_dir = candidate.to_string();
                break;
            }
        }
    }

    let cv_ocr = Arc::new(Mutex::new(CvOcrServices::new(
        PathBuf::from(&snapshot.config.assets_dir),
        core.log_publisher(),
    )));

    let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<RawInputSignal>();
    let raw_hub = match RawInputHub::start(snapshot.config.hotkeys.clone(), move |sig| {
        let _ = raw_tx.send(sig);
    }) {
        Ok(hub) => {
            snapshot.raw_input_active = true;
            core.emit_log(
                LogLevel::Info,
                "RawInputHub",
                "进程级单例 RawInputHub 已就绪 (已启用 hDevice==NULL 回环过滤)",
            );
            Some(hub)
        }
        Err(e) => {
            snapshot.raw_input_active = false;
            snapshot.raw_input_error = Some(e.to_string());
            core.emit_log(LogLevel::Error, "RawInputHub", e.to_string());
            None
        }
    };

    snapshot.available_windows = enumerate_candidate_windows(&snapshot.config.preferred_process);
    if let Some(pref) = snapshot
        .available_windows
        .iter()
        .find(|w| {
            w.process_name
                .eq_ignore_ascii_case(&snapshot.config.preferred_process)
        })
        .cloned()
    {
        snapshot.selected_window = Some(pref);
    }

    if !snapshot.config.pathing_dir.is_empty() {
        let dir = snapshot.config.pathing_dir.clone();
        snapshot.paths =
            tokio::task::spawn_blocking(move || bgi_pathing::scan_path_directory(dir, 500))
                .await
                .unwrap_or_default();
        if let Some(first) = snapshot.paths.first() {
            snapshot.active_path_file = Some(first.file_path.clone());
        }
    }

    core.publish_snapshot(Arc::new(snapshot.clone()));
    core.emit_log(
        LogLevel::Info,
        "Core",
        format!(
            "BetterGI-RS 核心启动完成：候选窗口 {} 个，路径 {} 条，Assets={}",
            snapshot.available_windows.len(),
            snapshot.paths.len(),
            snapshot.config.assets_dir
        ),
    );

    let pool = Arc::new(FrameBufferPool::new(4));
    let mut trigger_runtime =
        TriggerRuntime::new(snapshot.config.auto_skip, snapshot.config.auto_eat);
    let capturer: Arc<Mutex<ActiveCapturer>> = Arc::new(Mutex::new(None));
    let (pathing_tx, mut pathing_rx) = mpsc::unbounded_channel::<PathingEvent>();
    let mut active_pathing: Option<ActiveTask> = None;
    let mut next_task_id = 1u64;
    let mut frame_seq = 0u64;
    let mut dbg_seq = 0u64;
    let mut pace = CapturePaceState::new();
    let mut next_capture_at = tokio::time::Instant::now();
    let mut idle_gc_interval = tokio::time::interval(Duration::from_secs(5));
    let mut scheduler = FrameScheduler::new(Duration::from_millis(50), 4);
    let mut pending_frames: VecDeque<(u64, Option<u64>, CapturedFrame, std::time::Instant)> =
        VecDeque::new();
    let mut pipeline = PipelineMetrics::default();
    let mut last_ui = UiCategory::Unknown;
    let mut last_ui_classification = tokio::time::Instant::now() - Duration::from_secs(1);
    let mut pending_auto_pick: Option<PendingAutoPick> = None;

    loop {
        if active_pathing.is_none()
            && !snapshot.task_state.is_task_active()
            && trigger_runtime.has_pending_pathing()
        {
            let (name, file_path) = trigger_runtime
                .take_pathing_request()
                .expect("pending pathing checked");
            snapshot.active_path_file = Some(file_path);
            let result = handle_command(
                CoreCommand::StartBasicPathing,
                &mut snapshot,
                &raw_hub,
                &capturer,
                &pool,
                &mut frame_seq,
                &mut dbg_seq,
                &mut pace,
                &cv_ocr,
                &mut trigger_runtime,
                &mut active_pathing,
                &mut next_task_id,
                &pathing_tx,
                &core,
            )
            .await;
            if let Err(error) = result {
                core.emit_log(
                    LogLevel::Warn,
                    "Trigger",
                    format!("启动请求任务 {name} 失败: {error}"),
                );
            }
        }
        let triggers_active = trigger_runtime.has_triggers();
        let trigger_leases = trigger_runtime.lease_status();
        if triggers_active != snapshot.triggers_active || trigger_leases != snapshot.trigger_leases
        {
            snapshot.triggers_active = triggers_active;
            snapshot.trigger_leases = trigger_leases;
            core.publish_snapshot(Arc::new(snapshot.clone()));
        }
        let capture_needed =
            snapshot.preview_enabled || snapshot.triggers_active || trigger_runtime.is_replaying();
        let preview_interval = if snapshot.triggers_active || trigger_runtime.is_replaying() {
            Duration::from_millis(50)
        } else {
            Duration::from_millis((1000 / snapshot.config.preview_fps.clamp(1, 60)).max(16) as u64)
        };

        let next_recognition_at = scheduler
            .next_deadline()
            .map(tokio::time::Instant::from_std);

        tokio::select! {
            maybe_env = core.cmd_rx.recv() => {
                let Some(env) = maybe_env else {
                    break;
                };
                let was_preview = snapshot.preview_enabled;
                let is_shutdown = matches!(env.cmd, CoreCommand::ShutdownCore);
                let previous_config = snapshot.config.clone();
                let mut result = handle_command(
                    env.cmd,
                    &mut snapshot,
                    &raw_hub,
                    &capturer,
                    &pool,
                    &mut frame_seq,
                    &mut dbg_seq,
                    &mut pace,
                    &cv_ocr,
                    &mut trigger_runtime,
                    &mut active_pathing,
                    &mut next_task_id,
                    &pathing_tx,
                    &core,
                ).await;
                if result.is_ok() {
                    result = persist_config(
                        &persisted_config_path,
                        &previous_config,
                        &snapshot.config,
                        &core,
                    );
                }
                if !was_preview && snapshot.preview_enabled {
                    pace.reset();
                    next_capture_at = tokio::time::Instant::now();
                }
                if let Err(ref err) = result {
                    core.emit_log(LogLevel::Warn, "Core", format!("指令执行失败: {err}"));
                }
                let _ = env.reply.send(result);
                core.publish_snapshot(Arc::new(snapshot.clone()));
                if is_shutdown {
                    break;
                }
            }
            Some(sig) = raw_rx.recv() => {
                let was_preview = snapshot.preview_enabled;
                let previous_config = snapshot.config.clone();
                handle_raw_signal(
                    sig,
                    &mut snapshot,
                    &raw_hub,
                    &capturer,
                    &pool,
                    &cv_ocr,
                    &mut trigger_runtime,
                    &mut active_pathing,
                    &mut next_task_id,
                    &pathing_tx,
                    &core,
                );
                let _ = persist_config(
                    &persisted_config_path,
                    &previous_config,
                    &snapshot.config,
                    &core,
                );
                if !was_preview && snapshot.preview_enabled {
                    pace.reset();
                    next_capture_at = tokio::time::Instant::now();
                }
                core.publish_snapshot(Arc::new(snapshot.clone()));
            }
            Some(pev) = pathing_rx.recv() => {
                match pev {
                    PathingEvent::Progress { task_id, current_index, total_points } => {
                        if accepts_event(&active_pathing, task_id)
                            && let TaskState::Pathing { task_id: state_id, task_name, .. } = &snapshot.task_state
                            && *state_id == task_id
                        {
                            snapshot.task_state = TaskState::Pathing {
                                task_id,
                                task_name: task_name.clone(),
                                current_index,
                                total_points,
                            };
                            core.publish_snapshot(Arc::new(snapshot.clone()));
                        }
                    }
                    PathingEvent::Log { task_id, level, message } => {
                        if accepts_event(&active_pathing, task_id) {
                            core.emit_log(level, "Pathing", message);
                        }
                    }
                    PathingEvent::SalvagePreviewFinished { task_id, result, preview } => {
                        if !accepts_event(&active_pathing, task_id) {
                            continue;
                        }
                        if let Some(mut active) = active_pathing.take()
                            && let Some(join) = active.take_join()
                        {
                            let _ = join.join();
                        }
                        snapshot.last_salvage_preview_report = preview;
                        match result {
                            Ok(report) => {
                                let outcome = report.outcome;
                                let failure = report.failure.clone();
                                snapshot.last_task_report = Some(report);
                                match outcome {
                                    TaskOutcome::Success | TaskOutcome::Partial => {
                                        core.emit_log(LogLevel::Info, "ArtifactSalvage", format!("分解预览结束: {outcome:?}（未选择、未分解、未确认）"));
                                        snapshot.task_state = idle_or_previewing(snapshot.preview_enabled);
                                    }
                                    TaskOutcome::Cancelled => {
                                        core.emit_log(LogLevel::Info, "ArtifactSalvage", "分解预览已取消（未选择、未分解、未确认）");
                                        snapshot.task_state = idle_or_previewing(snapshot.preview_enabled);
                                    }
                                    TaskOutcome::Failed => {
                                        let err = failure.unwrap_or_else(|| "分解预览失败".to_string());
                                        core.emit_log(LogLevel::Warn, "ArtifactSalvage", format!("分解预览已停止: {err}"));
                                        snapshot.task_state = TaskState::Error(err);
                                    }
                                }
                            }
                            Err(err) => {
                                core.emit_log(LogLevel::Warn, "ArtifactSalvage", format!("分解预览结束: {err}"));
                                snapshot.task_state = TaskState::Error(err);
                            }
                        }
                        trigger_runtime.set_task_mode(false);
                        {
                            let mut cap_guard = capturer.lock().unwrap();
                            release_idle_capturer(&snapshot, &mut cap_guard, &core);
                        }
                        core.publish_snapshot(Arc::new(snapshot.clone()));
                    }
                    PathingEvent::Finished { task_id, result } => {
                        if !accepts_event(&active_pathing, task_id) {
                            continue;
                        }
                        let stopping = matches!(snapshot.task_state, TaskState::Stopping { .. });
                        if let Some(mut active) = active_pathing.take()
                            && let Some(join) = active.take_join()
                        {
                            let _ = join.join();
                        }
                        match result {
                            Ok(report) => {
                                let outcome = report.outcome;
                                let failure = report.failure.clone();
                                let runs = report.runs_completed;
                                snapshot.last_task_report = Some(report);
                                match outcome {
                                    TaskOutcome::Success | TaskOutcome::Partial => {
                                        core.emit_log(LogLevel::Info, "Pathing", format!("寻路完成: {outcome:?}, 完成 {runs} 轮"));
                                        snapshot.task_state = idle_or_previewing(snapshot.preview_enabled);
                                    }
                                    TaskOutcome::Cancelled => {
                                        trigger_runtime.clear_pending_pathing();
                                        core.emit_log(LogLevel::Info, "Pathing", "寻路已取消");
                                        snapshot.task_state = idle_or_previewing(snapshot.preview_enabled);
                                    }
                                    TaskOutcome::Failed => {
                                        trigger_runtime.clear_pending_pathing();
                                        let err = failure.unwrap_or_else(|| "寻路失败".to_string());
                                        core.emit_log(LogLevel::Warn, "Pathing", format!("寻路结束: {err}"));
                                        snapshot.task_state = if stopping { idle_or_previewing(snapshot.preview_enabled) } else { TaskState::Error(err) };
                                    }
                                }
                            }
                            Err(err) => {
                                trigger_runtime.clear_pending_pathing();
                                if stopping {
                                    snapshot.task_state = idle_or_previewing(snapshot.preview_enabled);
                                } else {
                                    core.emit_log(LogLevel::Warn, "Pathing", format!("寻路结束: {err}"));
                                    snapshot.task_state = TaskState::Error(err);
                                }
                            }
                        }
                        trigger_runtime.set_task_mode(false);
                        {
                            let mut cap_guard = capturer.lock().unwrap();
                            release_idle_capturer(&snapshot, &mut cap_guard, &core);
                        }
                        core.publish_snapshot(Arc::new(snapshot.clone()));
                    }
                }
            }
            result = async {
                let join = &mut pending_auto_pick.as_mut().expect("guarded job").join;
                std::future::poll_fn(|cx| std::pin::Pin::new(&mut *join).poll(cx)).await
            }, if pending_auto_pick.is_some() => {
                let job = pending_auto_pick.take().expect("guarded job");
                pipeline.observe(PipelineStage::Recognition, job.started.elapsed());
                let input_started = std::time::Instant::now();
                match result {
                    Ok(Ok(hits)) if accepts_event(&active_pathing, job.context.task_id) => {
                        let cancelled = active_pathing.as_ref().is_some_and(|task| {
                            task.cancel.load(std::sync::atomic::Ordering::Acquire)
                        });
                        trigger_runtime.apply_auto_pick_hits(
                            &job.context,
                            active_pathing.as_ref().map(|task| task.id),
                            snapshot.selected_window.as_ref().map(|window| window.hwnd),
                            hits,
                            cancelled,
                            &core,
                        );
                    }
                    Ok(Ok(_)) => core.emit_log(
                        LogLevel::Debug,
                        "AutoPick",
                        "丢弃已结束任务的实时拾取结果",
                    ),
                    Ok(Err(error)) => core.emit_log(
                        LogLevel::Warn,
                        "AutoPick",
                        format!("实时拾取 OCR 失败: {error}"),
                    ),
                    Err(error) => core.emit_log(
                        LogLevel::Warn,
                        "AutoPick",
                        format!("实时拾取 OCR 线程失败: {error}"),
                    ),
                }
                pipeline.observe(PipelineStage::Input, input_started.elapsed());
                core.publish_snapshot(Arc::new(snapshot.clone()));
            }
            _ = tokio::time::sleep_until(next_recognition_at.unwrap_or(tokio::time::Instant::now())), if next_recognition_at.is_some() => {
                let now = std::time::Instant::now();
                if let Some(ticket) = scheduler.poll(now) {
                    let position = pending_frames
                        .iter()
                        .position(|(frame_id, _, _, _)| *frame_id == ticket.frame_id)
                        .expect("scheduled frame is retained");
                    let (_, captured_task_id, frame, captured_at) =
                        pending_frames.remove(position).unwrap();
                    let recognition_started = std::time::Instant::now();
                    if last_ui_classification.elapsed() >= Duration::from_millis(250)
                        && let Ok(mut guard) = cv_ocr.try_lock()
                    {
                        last_ui = guard.classify_ui_step(
                            frame.data.as_slice(),
                            frame.width,
                            frame.height,
                            frame.format.is_bgr24(),
                        );
                        last_ui_classification = tokio::time::Instant::now();
                    }
                    pipeline.observe(
                        PipelineStage::Recognition,
                        recognition_started.elapsed(),
                    );
                    let decision_started = std::time::Instant::now();
                    if snapshot.config.auto_eat
                        && let Ok(mut guard) = cv_ocr.try_lock()
                    {
                        let observation = guard.detect_auto_eat_step(
                            frame.data.as_slice(),
                            frame.width,
                            frame.height,
                            frame.format.is_bgr24(),
                        );
                        trigger_runtime.set_auto_eat_observation(observation);
                    }
                    match trigger_runtime.dispatch(
                        ticket.frame_id,
                        &frame,
                        &snapshot,
                        last_ui,
                        &core,
                    ) {
                        Ok(decisions) => {
                            trigger_runtime.process_decisions(decisions, &snapshot, &core);
                        }
                        Err(error) => {
                            let level = if error.is_control_flow() {
                                LogLevel::Debug
                            } else {
                                LogLevel::Warn
                            };
                            core.emit_log(
                                level,
                                "Trigger",
                                format!("触发器控制流: {error}"),
                            );
                        }
                    }
                    if trigger_runtime.take_auto_pick_request() && pending_auto_pick.is_none() {
                        let mut prompt = None;
                        if captured_at.elapsed() <= AUTO_PICK_MAX_OBSERVATION_AGE
                            && let Ok(mut guard) = cv_ocr.try_lock()
                        {
                            match guard.detect_auto_pick_prompt(
                                frame.data.as_slice(),
                                frame.width,
                                frame.height,
                                frame.format.is_bgr24(),
                            ) {
                                Ok(observation) => prompt = observation.prompt,
                                Err(error) => core.emit_log(
                                    LogLevel::Warn,
                                    "AutoPick",
                                    format!("实时拾取正例识别失败: {error}"),
                                ),
                            }
                        }
                        let state = trigger_runtime
                            .observe_auto_pick_prompt(prompt.map(|prompt| prompt.identity));
                        if let (
                            AutoPickGateState::Ready(identity),
                            Some(prompt),
                            Some(task_id),
                            Some(hwnd),
                        ) = (
                            state,
                            prompt,
                            captured_task_id,
                            snapshot.selected_window.as_ref().map(|window| window.hwnd),
                        ) && prompt.identity == identity
                        {
                            let cv = cv_ocr.clone();
                            let width = frame.width;
                            let height = frame.height;
                            let is_bgr24 = frame.format.is_bgr24();
                            let data = frame.data.as_slice().to_vec();
                            let roi = prompt.text_roi;
                            let started = std::time::Instant::now();
                            let join = tokio::spawn(async move {
                                tokio::task::spawn_blocking(move || {
                                    let mut guard = cv.lock().unwrap();
                                    guard.ocr_roi_step(
                                        &data,
                                        width,
                                        height,
                                        is_bgr24,
                                        roi.x0,
                                        roi.y0,
                                        roi.x1,
                                        roi.y1,
                                    )
                                })
                                .await
                                .map_err(|e| format!("实时拾取 OCR 线程失败: {e}"))?
                            });
                            pending_auto_pick = Some(PendingAutoPick {
                                context: AutoPickExecutionContext {
                                    prompt,
                                    frame_size: (width, height),
                                    hwnd,
                                    task_id,
                                    observed_at: captured_at,
                                },
                                join,
                                started,
                            });
                        }
                    }
                    pipeline.observe(PipelineStage::Decision, decision_started.elapsed());
                    scheduler.complete();
                }
            }
            _ = tokio::time::sleep_until(next_capture_at), if capture_needed => {
                let capture_started = std::time::Instant::now();
                let now = tokio::time::Instant::now();
                next_capture_at = if now.duration_since(next_capture_at) > preview_interval.saturating_mul(2) {
                    now + preview_interval
                } else {
                    next_capture_at + preview_interval
                };
                let captured = if trigger_runtime.is_replaying() {
                    trigger_runtime.next_frame()
                } else if let Ok(mut cap_guard) = capturer.try_lock() {
                    capture_frame(
                        &mut snapshot,
                        &mut cap_guard,
                        &pool,
                        &mut frame_seq,
                        &mut pace,
                        &core,
                    )
                } else {
                    Ok(None)
                };
                match captured {
                    Ok(Some((seq, frame))) => {
                        pipeline.observe(PipelineStage::Capture, capture_started.elapsed());
                        if snapshot.preview_enabled && !snapshot.config.capture_only_no_render {
                            core.publish_preview(frame_to_preview(
                                seq,
                                &frame,
                                960,
                                540,
                                snapshot.capture_stats.last_capture_ms,
                            ));
                        }
                        if snapshot.triggers_active {
                            let enqueued_at = std::time::Instant::now();
                            let deadline = enqueued_at + scheduler.interval();
                            if let Err(error) = scheduler.submit(seq, deadline, enqueued_at) {
                                core.emit_log(
                                    LogLevel::Debug,
                                    "Scheduler",
                                    format!("丢弃触发帧 {seq}: {error}"),
                                );
                            } else {
                                let task_id = active_pathing.as_ref().map(|task| task.id);
                                pending_frames.push_back((seq, task_id, frame, capture_started));
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(err) => {
                        core.emit_log(LogLevel::Warn, "Capture", format!("抓图停止: {err}"));
                        snapshot.preview_enabled = false;
                        if trigger_runtime.is_replaying() {
                            trigger_runtime.clear_replay();
                        } else if let Ok(mut cap_guard) = capturer.try_lock() {
                            *cap_guard = None;
                        }
                        if matches!(snapshot.task_state, TaskState::Previewing) {
                            snapshot.task_state = TaskState::Idle;
                        }
                        core.publish_snapshot(Arc::new(snapshot.clone()));
                    }
                }
            }
            _ = idle_gc_interval.tick() => {
                snapshot.runtime_metrics = metrics_snapshot(&scheduler, &pipeline);
                core.publish_snapshot(Arc::new(snapshot.clone()));
                let unloaded = {
                    let mut guard = cv_ocr.lock().unwrap();
                    guard.unload_idle()
                };
                if !unloaded.is_empty() {
                    core.emit_log(
                        LogLevel::Info,
                        "IdleUnload",
                        format!("空闲超时已自动卸载缓存: {}", unloaded.join(", ")),
                    );
                }
            }
        }
    }

    if let Some(mut active) = active_pathing.take() {
        active.request_stop();
        if let Some(join) = active.take_join() {
            let _ = join.join();
        }
    }
    if let Err(err) = trigger_runtime.stop_recording() {
        core.emit_log(LogLevel::Warn, "Replay", format!("保存帧录制失败: {err}"));
    }
    if let Ok(mut guard) = capturer.lock() {
        *guard = None;
    }
    drop(raw_hub);
    bgi_input::release_all_keys();
}
