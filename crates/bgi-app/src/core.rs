use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bgi_capture::{
    CapturerBackend, FrameBufferPool, PooledBuffer, bgra_to_preview, enumerate_candidate_windows,
};
use bgi_control::{
    CaptureMode, CoreCommand, CoreEndpoint, CoreEvent, CvDebugResult, HotkeyAction, LogLevel,
    TaskState, UiSnapshot,
};
use bgi_cv::matchers::{bgra_to_bgr_mat, extract_minimap_bgr};
use bgi_cv::orientation::{MiniMapPreprocessor, compute_character_orientation};
use bgi_cv::sift_map::BigMapSiftMatcher;
use bgi_cv::teyvat_minimap::{LayerLoadScope, TeyvatMiniMapLocalizer};
use bgi_infer::paddle_ocr::{PaddleOcrEngine, PaddleOcrPaths};
use bgi_infer::{IdleUnloadPolicy, OcrService};
use bgi_input::{RawInputHub, RawInputSignal};
use tokio::sync::mpsc;

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

struct CvOcrServices {
    assets_dir: PathBuf,
    sift: Option<BigMapSiftMatcher>,
    minimap: Option<TeyvatMiniMapLocalizer>,
    orient: Option<MiniMapPreprocessor>,
    ocr: Option<PaddleOcrEngine>,
    last_minimap_pos: Option<(f32, f32)>,
}

impl CvOcrServices {
    fn new(assets_dir: PathBuf) -> Self {
        Self {
            assets_dir,
            sift: None,
            minimap: None,
            orient: None,
            ocr: None,
            last_minimap_pos: None,
        }
    }

    fn set_assets_dir(&mut self, dir: PathBuf) {
        self.assets_dir = dir;
        self.sift = None;
        self.minimap = None;
        self.ocr = None;
        self.last_minimap_pos = None;
    }

    fn unload_idle(&mut self) -> Vec<&'static str> {
        let mut unloaded = Vec::new();
        if let Some(mm) = self.minimap.as_mut()
            && mm.unload_if_idle()
        {
            unloaded.push("TeyvatMiniMap");
        }
        if let Some(ocr) = self.ocr.as_mut() {
            let was_loaded = ocr.is_loaded();
            ocr.unload_if_idle();
            if was_loaded && !ocr.is_loaded() {
                unloaded.push("PaddleOCR");
            }
        }
        unloaded
    }

    fn probe_minimap(&mut self, bgra: &[u8], width: u32, height: u32, seq: u64) -> Result<CvDebugResult, String> {
        let started = Instant::now();
        let full_bgr = bgra_to_bgr_mat(bgra, width, height).map_err(|e| e.to_string())?;
        let minimap_bgr = extract_minimap_bgr(&full_bgr).map_err(|e| e.to_string())?;

        if self.orient.is_none() {
            self.orient = Some(MiniMapPreprocessor::new().map_err(|e| e.to_string())?);
        }
        let (cam_angle, cam_conf) = self
            .orient
            .as_mut()
            .unwrap()
            .compute_camera_orientation(&minimap_bgr)
            .map_err(|e| e.to_string())?;
        let char_angle = compute_character_orientation(&minimap_bgr).map_err(|e| e.to_string())?;

        if self.minimap.is_none() {
            let teyvat_dir = self.assets_dir.join("Map").join("Teyvat");
            let localizer = TeyvatMiniMapLocalizer::new(
                teyvat_dir,
                LayerLoadScope::SurfaceOnly,
                Duration::from_secs(60),
            )
            .map_err(|e| e.to_string())?;
            self.minimap = Some(localizer);
        }

        let located = self
            .minimap
            .as_mut()
            .unwrap()
            .locate(&minimap_bgr, self.last_minimap_pos, 2)
            .map_err(|e| e.to_string())?;

        if let Some(m) = located {
            self.last_minimap_pos = Some((m.game_x, m.game_y));
            Ok(CvDebugResult {
                updated_seq: seq,
                source_label: "小地图模板匹配 + 朝向".to_string(),
                game_x: Some(m.game_x),
                game_y: Some(m.game_y),
                match_confidence: Some(m.confidence),
                floor: Some(m.floor),
                camera_angle: Some(m.camera_angle),
                camera_confidence: Some(m.camera_confidence),
                character_angle: char_angle,
                ocr_text: None,
                elapsed_ms: Some(started.elapsed().as_secs_f32() * 1000.0),
            })
        } else {
            self.last_minimap_pos = None;
            Ok(CvDebugResult {
                updated_seq: seq,
                source_label: "小地图模板匹配 (未命中)".to_string(),
                game_x: None,
                game_y: None,
                match_confidence: None,
                floor: None,
                camera_angle: Some(cam_angle),
                camera_confidence: Some(cam_conf),
                character_angle: char_angle,
                ocr_text: None,
                elapsed_ms: Some(started.elapsed().as_secs_f32() * 1000.0),
            })
        }
    }

    fn probe_bigmap_sift(&mut self, bgra: &[u8], width: u32, height: u32, seq: u64) -> Result<CvDebugResult, String> {
        let started = Instant::now();
        let full_bgr = bgra_to_bgr_mat(bgra, width, height).map_err(|e| e.to_string())?;
        if self.sift.is_none() {
            let teyvat_dir = self.assets_dir.join("Map").join("Teyvat");
            let sift = BigMapSiftMatcher::load_teyvat_256(teyvat_dir).map_err(|e| e.to_string())?;
            self.sift = Some(sift);
        }
        let (gx, gy, inliers) = self
            .sift
            .as_mut()
            .unwrap()
            .locate_big_map(&full_bgr)
            .map_err(|e| e.to_string())?;

        Ok(CvDebugResult {
            updated_seq: seq,
            source_label: format!("大地图 SIFT (inliers={inliers})"),
            game_x: Some(gx),
            game_y: Some(gy),
            match_confidence: None,
            floor: Some(0),
            camera_angle: None,
            camera_confidence: None,
            character_angle: None,
            ocr_text: None,
            elapsed_ms: Some(started.elapsed().as_secs_f32() * 1000.0),
        })
    }

    fn probe_ocr(&mut self, bgra: &[u8], width: u32, height: u32, seq: u64) -> Result<CvDebugResult, String> {
        let started = Instant::now();
        if self.ocr.is_none() {
            let paddle_dir = self.assets_dir.join("Model").join("PaddleOCR");
            let paths = PaddleOcrPaths::from_model_root(&paddle_dir, None).ok_or_else(|| {
                format!("在 {paddle_dir:?} 下未找到 PP-OCRv5/V4 Det/Rec 模型与 inference.yml")
            })?;
            self.ocr = Some(PaddleOcrEngine::new(
                paths,
                IdleUnloadPolicy {
                    idle_timeout: Duration::from_secs(60),
                },
            ));
        }

        let res = self
            .ocr
            .as_mut()
            .unwrap()
            .recognize_bgra(bgra, width, height)
            .map_err(|e| e.to_string())?;

        Ok(CvDebugResult {
            updated_seq: seq,
            source_label: format!("PaddleOCR ({} 区域)", res.regions.len()),
            game_x: None,
            game_y: None,
            match_confidence: None,
            floor: None,
            camera_angle: None,
            camera_confidence: None,
            character_angle: None,
            ocr_text: Some(res.text),
            elapsed_ms: Some(started.elapsed().as_secs_f32() * 1000.0),
        })
    }
}

pub fn spawn_core_actor(core: CoreEndpoint) -> CoreHandle {
    let join = std::thread::Builder::new()
        .name("bgi-core-rt".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("build tokio runtime");
            rt.block_on(run_core_loop(core));
        })
        .expect("spawn core thread");

    CoreHandle { join: Some(join) }
}

async fn run_core_loop(mut core: CoreEndpoint) {
    let mut snapshot = UiSnapshot::default();

    for candidate in [
        r"d:\bgiprro\bettergi-scripts-list\repo\pathing",
        r"d:\bgiprro\better-genshin-impact\BetterGenshinImpact\User\AutoPathing",
    ] {
        if PathBuf::from(candidate).is_dir() {
            snapshot.config.pathing_dir = candidate.to_string();
            break;
        }
    }

    for candidate in [
        r"d:\bgiprro\bgi_publish_new\Assets",
        r"Assets",
    ] {
        if PathBuf::from(candidate).is_dir() {
            snapshot.config.assets_dir = candidate.to_string();
            break;
        }
    }

    let cv_ocr = Arc::new(Mutex::new(CvOcrServices::new(PathBuf::from(
        &snapshot.config.assets_dir,
    ))));

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
        .find(|w| w.process_name.eq_ignore_ascii_case(&snapshot.config.preferred_process))
        .cloned()
    {
        snapshot.selected_window = Some(pref);
    }

    if !snapshot.config.pathing_dir.is_empty() {
        let dir = snapshot.config.pathing_dir.clone();
        snapshot.paths = tokio::task::spawn_blocking(move || bgi_pathing::scan_path_directory(dir, 500))
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

    let pool = FrameBufferPool::new(4);
    let mut capturer: Option<(isize, CaptureMode, CapturerBackend)> = None;
    let mut frame_seq = 0u64;
    let mut dbg_seq = 0u64;
    let mut pace = CapturePaceState::new();
    let mut next_capture_at = tokio::time::Instant::now();
    let mut idle_gc_interval = tokio::time::interval(Duration::from_secs(5));

    loop {
        let preview_interval =
            Duration::from_millis((1000 / snapshot.config.preview_fps.clamp(1, 60)).max(16) as u64);

        tokio::select! {
            maybe_env = core.cmd_rx.recv() => {
                let Some(env) = maybe_env else {
                    break;
                };
                let was_preview = snapshot.preview_enabled;
                let is_shutdown = matches!(env.cmd, CoreCommand::ShutdownCore);
                let result = handle_command(
                    env.cmd,
                    &mut snapshot,
                    &raw_hub,
                    &mut capturer,
                    &pool,
                    &mut frame_seq,
                    &mut dbg_seq,
                    &mut pace,
                    &cv_ocr,
                    &core,
                ).await;
                if !was_preview && snapshot.preview_enabled {
                    pace.reset();
                    next_capture_at = tokio::time::Instant::now();
                }
                let _ = env.reply.send(result);
                core.publish_snapshot(Arc::new(snapshot.clone()));
                if is_shutdown {
                    break;
                }
            }
            Some(sig) = raw_rx.recv() => {
                let was_preview = snapshot.preview_enabled;
                handle_raw_signal(sig, &mut snapshot, &raw_hub, &mut capturer, &core);
                if !was_preview && snapshot.preview_enabled {
                    pace.reset();
                    next_capture_at = tokio::time::Instant::now();
                }
                core.publish_snapshot(Arc::new(snapshot.clone()));
            }
            _ = tokio::time::sleep_until(next_capture_at), if snapshot.preview_enabled => {
                let now = tokio::time::Instant::now();
                next_capture_at = if now.duration_since(next_capture_at) > preview_interval.saturating_mul(2) {
                    now + preview_interval
                } else {
                    next_capture_at + preview_interval
                };
                if let Err(err) = capture_once(
                    &mut snapshot,
                    &mut capturer,
                    &pool,
                    &mut frame_seq,
                    &mut pace,
                    &core,
                ) {
                    core.emit_log(LogLevel::Warn, "Capture", format!("抓图停止: {err}"));
                    snapshot.preview_enabled = false;
                    if matches!(snapshot.task_state, TaskState::Previewing) {
                        snapshot.task_state = TaskState::Idle;
                    }
                    core.publish_snapshot(Arc::new(snapshot.clone()));
                }
            }
            _ = idle_gc_interval.tick() => {
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

    drop(capturer);
    drop(raw_hub);
    bgi_input::release_all_keys();
}

#[allow(clippy::too_many_arguments)]
async fn handle_command(
    cmd: CoreCommand,
    snapshot: &mut UiSnapshot,
    raw_hub: &Option<RawInputHub>,
    capturer: &mut Option<(isize, CaptureMode, CapturerBackend)>,
    pool: &FrameBufferPool,
    frame_seq: &mut u64,
    dbg_seq: &mut u64,
    pace: &mut CapturePaceState,
    cv_ocr: &Arc<Mutex<CvOcrServices>>,
    core: &CoreEndpoint,
) -> Result<(), String> {
    match cmd {
        CoreCommand::RefreshWindows => {
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
                    .find(|w| w.process_name.eq_ignore_ascii_case(&snapshot.config.preferred_process))
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
            close_capturer(capturer, core);
            snapshot.selected_window = hwnd.and_then(|id| {
                snapshot.available_windows.iter().find(|w| w.hwnd == id).cloned()
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
            close_capturer(capturer, core);
            snapshot.config.capture_mode = mode;
            core.emit_log(LogLevel::Info, "Capture", format!("切换截图后端 -> {}", mode.label()));
            Ok(())
        }
        CoreCommand::SetPreviewEnabled(enabled) => {
            if enabled && snapshot.selected_window.is_none() {
                return Err("请先在「抓图预览」中选择一个目标窗口".to_string());
            }
            snapshot.preview_enabled = enabled;
            if !enabled {
                // Close active capture session cleanly when preview stops so DWM stops pushing frames.
                close_capturer(capturer, core);
            }
            snapshot.task_state = if enabled {
                TaskState::Previewing
            } else {
                TaskState::Idle
            };
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
            snapshot.config.preview_fps = fps.clamp(1, 60);
            let interval_ms = (1000 / snapshot.config.preview_fps).max(16);
            let reopen = capturer
                .as_ref()
                .is_some_and(|(_, _, backend)| backend.needs_reopen_for_interval(interval_ms));
            if reopen {
                close_capturer(capturer, core);
            } else if let Some((_, _, backend)) = capturer.as_ref() {
                backend.set_min_update_interval_ms(interval_ms);
            }
            Ok(())
        }
        CoreCommand::CaptureSingleFrame => {
            let res = capture_once(snapshot, capturer, pool, frame_seq, pace, core);
            release_idle_capturer(snapshot, capturer, core);
            res
        }
        CoreCommand::SetPathingDir(dir) => {
            snapshot.config.pathing_dir = dir.clone();
            let paths = tokio::task::spawn_blocking(move || bgi_pathing::scan_path_directory(dir, 500))
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
        CoreCommand::SetAssetsDir(dir) => {
            snapshot.config.assets_dir = dir.clone();
            cv_ocr.lock().unwrap().set_assets_dir(PathBuf::from(&dir));
            core.emit_log(LogLevel::Info, "CV/OCR", format!("Assets 目录已切换至: {dir}"));
            Ok(())
        }
        CoreCommand::RefreshPaths => {
            let dir = snapshot.config.pathing_dir.clone();
            if dir.trim().is_empty() {
                return Err("请先填写路径目录".to_string());
            }
            let paths = tokio::task::spawn_blocking(move || bgi_pathing::scan_path_directory(dir, 500))
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
            let captured = capture_raw_bgra(snapshot, capturer, pool, core);
            release_idle_capturer(snapshot, capturer, core);
            let Some((w, h, buf)) = captured? else {
                return Err("当前帧尚未就绪".to_string());
            };
            *dbg_seq += 1;
            let seq = *dbg_seq;
            let cv = cv_ocr.clone();
            let res = tokio::task::spawn_blocking(move || {
                cv.lock().unwrap().probe_minimap(buf.as_slice(), w, h, seq)
            })
            .await
            .map_err(|e| e.to_string())??;
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
            let captured = capture_raw_bgra(snapshot, capturer, pool, core);
            release_idle_capturer(snapshot, capturer, core);
            let Some((w, h, buf)) = captured? else {
                return Err("当前帧尚未就绪".to_string());
            };
            *dbg_seq += 1;
            let seq = *dbg_seq;
            let cv = cv_ocr.clone();
            let res = tokio::task::spawn_blocking(move || {
                cv.lock().unwrap().probe_bigmap_sift(buf.as_slice(), w, h, seq)
            })
            .await
            .map_err(|e| e.to_string())??;
            core.emit_log(
                LogLevel::Info,
                "CV",
                format!("[大地图 SIFT] pos=({:?}, {:?}) {}", res.game_x, res.game_y, res.source_label),
            );
            snapshot.cv_debug = res;
            Ok(())
        }
        CoreCommand::ProbeOcrScreen => {
            let captured = capture_raw_bgra(snapshot, capturer, pool, core);
            release_idle_capturer(snapshot, capturer, core);
            let Some((w, h, buf)) = captured? else {
                return Err("当前帧尚未就绪".to_string());
            };
            *dbg_seq += 1;
            let seq = *dbg_seq;
            let cv = cv_ocr.clone();
            let res = tokio::task::spawn_blocking(move || {
                cv.lock().unwrap().probe_ocr(buf.as_slice(), w, h, seq)
            })
            .await
            .map_err(|e| e.to_string())??;
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
        CoreCommand::StartBasicPathing => {
            let Some(file) = snapshot.active_path_file.clone() else {
                return Err("尚未选择路径文件".to_string());
            };
            let (summary, _task) = bgi_pathing::load_path_file(&file).map_err(|e| e.to_string())?;
            core.emit_log(
                LogLevel::Info,
                "Pathing",
                format!(
                    "已加载基础路径「{}」：共 {} 个点位，基础可执行 {} 个，将跳过复杂动作 [{}]",
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
            core.emit_log(
                LogLevel::Warn,
                "Pathing",
                "Phase 2 CV/OCR 已接通；Phase 3 将接入 TpTask 传送与 PathExecutor 实机闭环移动控制。",
            );
            Ok(())
        }
        CoreCommand::EmergencyStop => {
            snapshot.preview_enabled = false;
            close_capturer(capturer, core);
            snapshot.task_state = TaskState::Idle;
            tokio::task::spawn_blocking(bgi_input::release_all_keys)
                .await
                .map_err(|e| e.to_string())?;
            core.emit_log(LogLevel::Warn, "Core", "已执行急停并释放所有键盘/鼠标按键");
            Ok(())
        }
        CoreCommand::ShutdownCore => {
            snapshot.preview_enabled = false;
            snapshot.task_state = TaskState::Idle;
            core.emit_log(LogLevel::Info, "Core", "正在关闭后台核心...");
            Ok(())
        }
    }
}

fn handle_raw_signal(
    sig: RawInputSignal,
    snapshot: &mut UiSnapshot,
    raw_hub: &Option<RawInputHub>,
    capturer: &mut Option<(isize, CaptureMode, CapturerBackend)>,
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
                    close_capturer(capturer, core);
                    snapshot.task_state = TaskState::Idle;
                    std::thread::spawn(bgi_input::release_all_keys);
                }
                HotkeyAction::TogglePreview => {
                    if snapshot.selected_window.is_some() {
                        snapshot.preview_enabled = !snapshot.preview_enabled;
                        if !snapshot.preview_enabled {
                            close_capturer(capturer, core);
                        }
                        snapshot.task_state = if snapshot.preview_enabled {
                            TaskState::Previewing
                        } else {
                            TaskState::Idle
                        };
                    }
                }
                HotkeyAction::StartStopPathing | HotkeyAction::TestPing => {}
            }
        }
    }
}

fn close_capturer(
    capturer: &mut Option<(isize, CaptureMode, CapturerBackend)>,
    core: &CoreEndpoint,
) {
    if let Some((_, _, backend)) = capturer.take()
        && let Some(diag) = backend.wgc_diagnostics()
    {
        core.emit_log(LogLevel::Info, "WGC", format!("会话关闭: {diag}"));
    }
}

fn release_idle_capturer(
    snapshot: &UiSnapshot,
    capturer: &mut Option<(isize, CaptureMode, CapturerBackend)>,
    core: &CoreEndpoint,
) {
    if !snapshot.preview_enabled && !matches!(snapshot.task_state, TaskState::Pathing { .. }) {
        close_capturer(capturer, core);
    }
}

fn capture_raw_bgra(
    snapshot: &UiSnapshot,
    capturer: &mut Option<(isize, CaptureMode, CapturerBackend)>,
    pool: &FrameBufferPool,
    core: &CoreEndpoint,
) -> Result<Option<(u32, u32, PooledBuffer)>, String> {
    let Some(win) = &snapshot.selected_window else {
        return Err("尚未选择目标窗口".to_string());
    };
    let mode = snapshot.config.capture_mode;

    let need_reopen = match capturer {
        Some((h, m, _)) => *h != win.hwnd || *m != mode,
        None => true,
    };

    if need_reopen {
        close_capturer(capturer, core);
        let interval_ms = (1000 / snapshot.config.preview_fps.clamp(1, 60)).max(16);
        let backend = CapturerBackend::open(win.hwnd, mode, interval_ms).map_err(|e| e.to_string())?;
        if let Some(diag) = backend.wgc_diagnostics() {
            core.emit_log(LogLevel::Info, "WGC", format!("会话启动: {diag}"));
        }
        *capturer = Some((win.hwnd, mode, backend));
    }

    let (_, _, backend) = capturer.as_mut().unwrap();
    match backend.capture(pool) {
        Ok(res) => Ok(Some(res)),
        Err(bgi_capture::CaptureError::FrameNotReady) => Ok(None),
        Err(e) => {
            close_capturer(capturer, core);
            Err(e.to_string())
        }
    }
}

struct CapturePaceState {
    window_start: Instant,
    window_frames: u64,
    last_stats_publish: Instant,
}

impl CapturePaceState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            window_start: now,
            window_frames: 0,
            last_stats_publish: now,
        }
    }

    fn reset(&mut self) {
        let now = Instant::now();
        self.window_start = now;
        self.window_frames = 0;
        self.last_stats_publish = now;
    }
}

fn capture_once(
    snapshot: &mut UiSnapshot,
    capturer: &mut Option<(isize, CaptureMode, CapturerBackend)>,
    pool: &FrameBufferPool,
    frame_seq: &mut u64,
    pace: &mut CapturePaceState,
    core: &CoreEndpoint,
) -> Result<(), String> {
    let started = Instant::now();
    let Some((w, h, buf)) = capture_raw_bgra(snapshot, capturer, pool, core)? else {
        return Ok(());
    };
    let capture_ms = started.elapsed().as_secs_f32() * 1000.0;

    pace.window_frames += 1;
    let elapsed = pace.window_start.elapsed();
    if elapsed >= Duration::from_millis(500) {
        snapshot.capture_stats.measured_fps = (pace.window_frames as f32) / elapsed.as_secs_f32();
        pace.window_start = Instant::now();
        pace.window_frames = 0;
    }
    snapshot.capture_stats.captured_frames += 1;
    snapshot.capture_stats.source_width = w;
    snapshot.capture_stats.source_height = h;
    snapshot.capture_stats.last_capture_ms = capture_ms;

    if snapshot.config.capture_only_no_render {
        // Return buffer to pool immediately; skip CPU scaling and UI texture upload.
        drop(buf);
        if pace.last_stats_publish.elapsed() >= Duration::from_secs(1) {
            pace.last_stats_publish = Instant::now();
            core.publish_snapshot(Arc::new(snapshot.clone()));
        }
        return Ok(());
    }

    *frame_seq += 1;
    let preview = bgra_to_preview(*frame_seq, w, h, buf.as_slice(), 960, 540, capture_ms);
    core.publish_preview(preview);
    Ok(())
}
