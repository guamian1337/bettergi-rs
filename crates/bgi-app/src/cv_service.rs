use std::path::PathBuf;
use std::time::{Duration, Instant};

use bgi_control::{CvDebugResult, LogLevel, LogPublisher};
use bgi_cv::artifact_salvage::{
    ArtifactSalvageMatcher, ArtifactSalvageObservation, resize_card_for_icon,
};
use bgi_cv::assets::AssetCatalog;
use bgi_cv::attribute_page::attribute_tab_ocr_image;
use bgi_cv::auto_pick_templates::{AutoPickObservation, AutoPickPromptMatcher};
use bgi_cv::auto_wood_templates::{AutoWoodMatcher, AutoWoodObservation};
use bgi_cv::chest_templates::{ChestMatcher, ChestObservation};
use bgi_cv::expedition::{ExpeditionMatcher, ExpeditionObservation};
use bgi_cv::fishing_templates::FishingMatcher;
use bgi_cv::matchers::{extract_minimap_bgr, frame_to_bgr_mat};
use bgi_cv::orientation::{MiniMapPreprocessor, compute_character_orientation};
use bgi_cv::reward_cards::{
    REWARD_BAND_ROI_1080P, RewardCountPreparation, detect_reward_card_rects, prepare_reward_card,
};
use bgi_cv::reward_templates::RewardTemplateMatcher;
use bgi_cv::sift_map::BigMapSiftMatcher;
use bgi_cv::teyvat_minimap::{LayerLoadScope, TeyvatMiniMapLocalizer};
use bgi_cv::ui_templates::{MotionStatus as CvMotionStatus, UiScreenMatcher, UiTemplateMatcher};
use bgi_infer::paddle_ocr::{NormalizedRoi, PaddleOcrEngine, PaddleOcrPaths};
use bgi_infer::{
    IdleUnloadPolicy, ItemIconMatch, ItemIconRecognizer, MiningDetector, OcrResult, OcrService,
    WeaponNameCatalog,
};
use bgi_pathing::{BigMapLayerButtons, MinimapObservation, MotionStatus, OcrHit};
use bgi_task::{
    CharacterAttributeDetails, CharacterAttributePixelRoi, ExpeditionPageState, FishingObservation,
    MAX_SALVAGE_CELLS_PER_PAGE, MiningDetection, MiningObservation, RewardCardObservation,
    RewardCardRect, RewardIdentity, RewardPageObservation, TalentDetails, TalentPixelRoi,
    UiCategory, WeaponDetails, WeaponPixelRoi, character_attribute_roi, filter_ore_detections,
    is_talent_detail_text, parse_character_level_pair, parse_reward_count, parse_talent_bonus,
    parse_talent_level, parse_talent_type, parse_weapon_level_pair, select_nearest_mining_target,
    talent_rois, weapon_rois,
};

use crate::pathing_host::to_projection;

pub(crate) struct CookingUiState {
    pub(crate) page_visible: bool,
    pub(crate) white_confirm: Option<(i32, i32)>,
    pub(crate) recover_visible: bool,
}

pub(crate) struct ArtifactSalvageStep {
    pub(crate) observation: ArtifactSalvageObservation,
    pub(crate) icon_matches: Vec<Option<ItemIconMatch>>,
}

/// Manages lazy-loaded CV/OCR engines, pathing localization steps, debug probes, and failure dumps.
pub(crate) struct CvOcrServices {
    assets: AssetCatalog,
    sift: Option<BigMapSiftMatcher>,
    minimap: Option<TeyvatMiniMapLocalizer>,
    orient: Option<MiniMapPreprocessor>,
    ui_matcher: Option<UiTemplateMatcher>,
    ui_screens: Option<UiScreenMatcher>,
    reward_matcher: Option<RewardTemplateMatcher>,
    chest_matcher: Option<ChestMatcher>,
    auto_pick_matcher: Option<AutoPickPromptMatcher>,
    fishing_matcher: Option<FishingMatcher>,
    expedition_matcher: Option<ExpeditionMatcher>,
    auto_wood_matcher: Option<AutoWoodMatcher>,
    artifact_salvage_matcher: Option<ArtifactSalvageMatcher>,
    item_icon_recognizer: Option<ItemIconRecognizer>,
    mining_detector: Option<MiningDetector>,
    weapon_name_catalog: Option<WeaponNameCatalog>,
    ocr: Option<PaddleOcrEngine>,
    last_minimap_pos: Option<(f32, f32)>,
    cooking_ui_checked: Option<Instant>,
    cooking_page: bool,
    cooking_confirm: Option<(i32, i32)>,
    cooking_recover: Option<(i32, i32)>,
    log_pub: LogPublisher,
    last_failed_save: Option<Instant>,
}

impl CvOcrServices {
    pub(crate) fn new(assets_dir: PathBuf, log_pub: LogPublisher) -> Self {
        let assets = AssetCatalog::load(&assets_dir, "zh-CN")
            .or_else(|_| AssetCatalog::embedded_fallback(&assets_dir))
            .unwrap_or_else(|_| {
                AssetCatalog::embedded_fallback(".").expect("working directory exists")
            });
        Self {
            assets,
            sift: None,
            minimap: None,
            orient: None,
            ui_matcher: None,
            ui_screens: None,
            reward_matcher: None,
            chest_matcher: None,
            auto_pick_matcher: None,
            fishing_matcher: None,
            expedition_matcher: None,
            auto_wood_matcher: None,
            artifact_salvage_matcher: None,
            item_icon_recognizer: None,
            mining_detector: None,
            weapon_name_catalog: None,
            ocr: None,
            last_minimap_pos: None,
            cooking_ui_checked: None,
            cooking_page: false,
            cooking_confirm: None,
            cooking_recover: None,
            log_pub,
            last_failed_save: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_failed_capture(
        &mut self,
        tag: &str,
        reason: &str,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        throttle: bool,
    ) -> Option<PathBuf> {
        let should_save = !throttle
            || self
                .last_failed_save
                .is_none_or(|t| t.elapsed() >= Duration::from_secs(1));

        let saved_path = if should_save {
            self.last_failed_save = Some(Instant::now());
            match crate::logging::save_failed_image(tag, width, height, is_bgr24, bytes) {
                Ok(p) => Some(p),
                Err(err) => {
                    self.log_pub.emit(
                        LogLevel::Warn,
                        "CV",
                        format!("{reason} (保存失败截图出错: {err})"),
                    );
                    return None;
                }
            }
        } else {
            None
        };

        let msg = if let Some(ref p) = saved_path {
            format!("{reason}，已保存截图: {}", p.display())
        } else {
            reason.to_string()
        };
        self.log_pub.emit(LogLevel::Warn, "CV", msg);
        saved_path
    }

    pub(crate) fn set_last_minimap_pos(&mut self, pos: Option<(f32, f32)>) {
        self.last_minimap_pos = pos;
    }

    fn ensure_minimap(&mut self) -> Result<&mut TeyvatMiniMapLocalizer, String> {
        if self.minimap.is_none() {
            let teyvat_dir = self
                .assets
                .map("Teyvat")
                .unwrap_or_else(|_| self.assets_root().join("Map").join("Teyvat"));
            let started = Instant::now();
            let mut localizer = TeyvatMiniMapLocalizer::new(
                &teyvat_dir,
                LayerLoadScope::AllLayers,
                Duration::from_secs(180),
            )
            .map_err(|e| e.to_string())?;
            localizer.ensure_loaded().map_err(|e| e.to_string())?;
            self.log_pub.emit(
                LogLevel::Info,
                "CV",
                format!(
                    "[小地图] 已加载全量地表与分层模板 ({} 层, {:.0}ms)",
                    localizer.loaded_layer_count(),
                    started.elapsed().as_secs_f32() * 1000.0
                ),
            );
            self.minimap = Some(localizer);
        }
        Ok(self.minimap.as_mut().unwrap())
    }

    fn ensure_sift(&mut self) -> Result<&mut BigMapSiftMatcher, String> {
        if self.sift.is_none() {
            let teyvat_dir = self
                .assets
                .map("Teyvat")
                .unwrap_or_else(|_| self.assets_root().join("Map").join("Teyvat"));
            let sift =
                BigMapSiftMatcher::load_teyvat_256(&teyvat_dir).map_err(|e| e.to_string())?;
            let ct = sift.coord_transform();
            self.log_pub.emit(
                LogLevel::Info,
                "CV",
                format!(
                    "[大地图 SIFT] 已加载特征库 ({} kps, origin=({:.0}, {:.0}), dir={})",
                    sift.keypoint_count(),
                    ct.origin_x,
                    ct.origin_y,
                    teyvat_dir.display()
                ),
            );
            self.sift = Some(sift);
        }
        Ok(self.sift.as_mut().unwrap())
    }

    fn ensure_ocr(&mut self) -> Result<&mut PaddleOcrEngine, String> {
        if self.ocr.is_none() {
            let paddle_dir = self.assets.root().join("Model").join("PaddleOCR");
            let paths = match self.assets.ocr_paths() {
                Ok((det, rec, dict)) => Some(PaddleOcrPaths {
                    ort_dll_path: None,
                    det_model_path: det,
                    rec_model_path: rec,
                    rec_yml_path: dict,
                }),
                Err(_) => PaddleOcrPaths::from_model_root(&paddle_dir, None),
            }
            .ok_or_else(|| {
                format!("在 {paddle_dir:?} 下未找到 PP-OCRv5/V4 Det/Rec 模型与 inference.yml")
            })?;
            self.ocr = Some(PaddleOcrEngine::new(
                paths,
                IdleUnloadPolicy {
                    idle_timeout: Duration::from_secs(60),
                },
            ));
        }
        Ok(self.ocr.as_mut().unwrap())
    }

    fn ensure_ui_matcher(&mut self) -> Result<&UiTemplateMatcher, String> {
        if self.ui_matcher.is_none() {
            let external = (|| {
                UiTemplateMatcher::from_assets(
                    self.assets
                        .template("ui.map_underground_switch")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.map_underground_to_ground")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.key_space")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.key_x")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.ui_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => UiTemplateMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.ui_matcher.as_ref().unwrap())
    }

    fn ensure_ui_screens(&mut self) -> Result<&UiScreenMatcher, String> {
        if self.ui_screens.is_none() {
            let external = (|| {
                UiScreenMatcher::from_assets(
                    self.assets
                        .template("ui.paimon_menu")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.inventory")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.in_domain")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.cook_icon")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.white_confirm")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.white_recover")
                        .map_err(|e| e.to_string())?,
                    self.assets.template("ui.talk").map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.recovery")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.resurrection")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.ui_screens = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => UiScreenMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.ui_screens.as_ref().unwrap())
    }

    fn ensure_reward_matcher(&mut self) -> Result<&RewardTemplateMatcher, String> {
        if self.reward_matcher.is_none() {
            let external = (|| {
                RewardTemplateMatcher::from_assets(
                    self.assets
                        .template("ui.claim_text")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.claim_gift")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.click_blank_continue")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.reward_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => RewardTemplateMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.reward_matcher.as_ref().unwrap())
    }

    fn ensure_chest_matcher(&mut self) -> Result<&ChestMatcher, String> {
        if self.chest_matcher.is_none() {
            let external = (|| {
                ChestMatcher::from_assets(
                    self.assets
                        .template("ui.chest")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.chest_f_icon")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.chest_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => ChestMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.chest_matcher.as_ref().unwrap())
    }

    fn ensure_auto_pick_matcher(&mut self) -> Result<&AutoPickPromptMatcher, String> {
        if self.auto_pick_matcher.is_none() {
            let external = (|| {
                AutoPickPromptMatcher::from_assets(
                    self.assets
                        .template("ui.auto_pick.pick")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.auto_pick.l")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.auto_pick.settings")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.auto_pick.chat")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.auto_pick_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => AutoPickPromptMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.auto_pick_matcher.as_ref().unwrap())
    }

    fn ensure_fishing_matcher(&mut self) -> Result<&FishingMatcher, String> {
        if self.fishing_matcher.is_none() {
            let external = (|| {
                FishingMatcher::from_assets(
                    self.assets
                        .template("ui.fishing.exit_fishing")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.fishing.wait_bite")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.fishing.lift_rod")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.fishing_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => FishingMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.fishing_matcher.as_ref().unwrap())
    }

    fn ensure_expedition_matcher(&mut self) -> Result<&ExpeditionMatcher, String> {
        if self.expedition_matcher.is_none() {
            self.expedition_matcher = Some(ExpeditionMatcher::new().map_err(|e| e.to_string())?);
        }
        Ok(self.expedition_matcher.as_ref().unwrap())
    }

    fn ensure_auto_wood_matcher(&mut self) -> Result<&mut AutoWoodMatcher, String> {
        if self.auto_wood_matcher.is_none() {
            let external = (|| {
                AutoWoodMatcher::from_assets(
                    self.assets
                        .template("ui.auto_wood.gadget")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.auto_wood.menu")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.auto_wood.confirm")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.auto_wood.enter_game")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.auto_wood_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => AutoWoodMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.auto_wood_matcher.as_mut().unwrap())
    }

    fn ensure_artifact_salvage_matcher(&mut self) -> Result<&mut ArtifactSalvageMatcher, String> {
        if self.artifact_salvage_matcher.is_none() {
            let external = (|| {
                ArtifactSalvageMatcher::from_assets(
                    self.assets
                        .template("ui.artifact_salvage.inventory")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.artifact_salvage.confirm")
                        .map_err(|e| e.to_string())?,
                    self.assets
                        .template("ui.artifact_salvage.black_confirm")
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            })();
            self.artifact_salvage_matcher = Some(match external {
                Ok(matcher) => matcher,
                Err(_) => ArtifactSalvageMatcher::new().map_err(|e| e.to_string())?,
            });
        }
        Ok(self.artifact_salvage_matcher.as_mut().unwrap())
    }

    fn ensure_item_icon_recognizer(&mut self) -> Result<&mut ItemIconRecognizer, String> {
        if self.item_icon_recognizer.is_none() {
            let model_dir = self.assets_root().join("Model").join("ItemV2");
            let started = Instant::now();
            let recognizer = ItemIconRecognizer::from_paths(
                model_dir.join("item.onnx"),
                model_dir.join("item.csv"),
                None,
            )
            .map_err(|error| format!("加载 ItemV2 圣遗物图标模型失败: {error}"))?;
            self.log_pub.emit(
                LogLevel::Info,
                "CV",
                format!(
                    "[圣遗物图标] ItemV2 已加载 (model={}, {:.0}ms)",
                    recognizer.paths().model_path.display(),
                    started.elapsed().as_secs_f32() * 1000.0
                ),
            );
            self.item_icon_recognizer = Some(recognizer);
        }
        Ok(self.item_icon_recognizer.as_mut().unwrap())
    }

    fn ensure_mining_detector(&mut self) -> Result<&mut MiningDetector, String> {
        if self.mining_detector.is_none() {
            let model_path = self
                .assets
                .model("mine")
                .map_err(|error| format!("采矿模型资源不可用: {error}"))?;
            let started = Instant::now();
            let detector = MiningDetector::from_paths(&model_path, None)
                .map_err(|error| format!("加载采矿模型失败: {error}"))?;
            self.log_pub.emit(
                LogLevel::Info,
                "CV",
                format!(
                    "[自动采矿] bgi_mine 已加载 (model={}, {:.0}ms)",
                    detector.model_path().display(),
                    started.elapsed().as_secs_f32() * 1000.0
                ),
            );
            self.mining_detector = Some(detector);
        }
        Ok(self.mining_detector.as_mut().unwrap())
    }

    fn ensure_weapon_name_catalog(&mut self) -> Result<&mut WeaponNameCatalog, String> {
        if self.weapon_name_catalog.is_none() {
            let path = self
                .assets_root()
                .join("Model")
                .join("ItemV2")
                .join("item.csv");
            let catalog = WeaponNameCatalog::from_path(&path)
                .map_err(|error| format!("加载武器名称表 {} 失败: {error}", path.display()))?;
            self.weapon_name_catalog = Some(catalog);
        }
        Ok(self.weapon_name_catalog.as_mut().unwrap())
    }

    pub(crate) fn detect_weapon_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<Option<WeaponDetails>, String> {
        let rois = weapon_rois(width, height)
            .ok_or_else(|| "武器详情识别仅支持严格 16:9 的抓图分辨率".to_string())?;
        let name_roi = weapon_roi_to_normalized(rois.name, width, height);
        let name_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            name_roi.x0,
            name_roi.y0,
            name_roi.x1,
            name_roi.y1,
        )?;
        let level_roi = weapon_roi_to_normalized(rois.level, width, height);
        let level_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            level_roi.x0,
            level_roi.y0,
            level_roi.x1,
            level_roi.y1,
        )?;
        let catalog = self.ensure_weapon_name_catalog()?;
        Ok(weapon_details_from_hits(&name_hits, &level_hits, catalog))
    }

    pub(crate) fn detect_talent_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<Option<TalentDetails>, String> {
        let rois = talent_rois(width, height)
            .ok_or_else(|| "天赋详情识别仅支持严格 16:9 的抓图分辨率".to_string())?;
        let page_gate_roi = talent_roi_to_normalized(rois.page_gate, width, height);
        let page_gate_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            page_gate_roi.x0,
            page_gate_roi.y0,
            page_gate_roi.x1,
            page_gate_roi.y1,
        )?;
        if page_gate_hits.is_empty() || !is_talent_detail_text(&join_ocr_hits(&page_gate_hits)) {
            return Ok(None);
        }
        let type_roi = talent_roi_to_normalized(rois.talent_type, width, height);
        let type_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            type_roi.x0,
            type_roi.y0,
            type_roi.x1,
            type_roi.y1,
        )?;
        let level_roi = talent_roi_to_normalized(rois.level, width, height);
        let level_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            level_roi.x0,
            level_roi.y0,
            level_roi.x1,
            level_roi.y1,
        )?;
        let bonus_roi = talent_roi_to_normalized(rois.bonus, width, height);
        let bonus_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            bonus_roi.x0,
            bonus_roi.y0,
            bonus_roi.x1,
            bonus_roi.y1,
        )?;
        Ok(talent_details_from_hits(
            &page_gate_hits,
            &type_hits,
            &level_hits,
            &bonus_hits,
        ))
    }

    pub(crate) fn detect_character_attribute_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<Option<CharacterAttributeDetails>, String> {
        let roi = character_attribute_roi(width, height)
            .ok_or_else(|| "角色属性识别仅支持严格 16:9 的抓图分辨率".to_string())?;
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|error| error.to_string())?;
        let gate_image = attribute_tab_ocr_image(&full_bgr).map_err(|error| error.to_string())?;
        let gate_started = Instant::now();
        let gate_result = self
            .ensure_ocr()?
            .recognize_bgr_mat(&gate_image)
            .map_err(|error| error.to_string())?;
        let gate_hits: Vec<OcrHit> = gate_result
            .regions
            .iter()
            .map(|region| OcrHit {
                text: region.text.clone(),
                center_x: (region.x + region.width * 0.5).round() as i32,
                center_y: (region.y + region.height * 0.5).round() as i32,
                confidence: region.score,
            })
            .collect();
        self.log_pub.emit(
            LogLevel::Info,
            "OCR",
            format!(
                "[角色属性页门控] 左侧 20% HSV 阈值 OCR 命中 {} 条 ({:.1}ms)",
                gate_hits.len(),
                gate_started.elapsed().as_secs_f32() * 1000.0
            ),
        );
        if !attribute_page_from_hits(&gate_hits) {
            return Ok(None);
        }
        let level_roi = character_attribute_roi_to_normalized(roi, width, height);
        let level_hits = self.ocr_roi_step(
            bytes,
            width,
            height,
            is_bgr24,
            level_roi.x0,
            level_roi.y0,
            level_roi.x1,
            level_roi.y1,
        )?;
        Ok(character_attribute_details_from_hits(&level_hits))
    }

    pub(crate) fn detect_reward_result_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<RewardPageObservation, String> {
        if width != 1920 || height != 1080 {
            return Err("奖励结果页识别仅支持 1920×1080 抓图分辨率".into());
        }
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|error| error.to_string())?;
        let (band_x, band_y, band_width, band_height) = REWARD_BAND_ROI_1080P;
        let band = opencv::core::Mat::roi(
            &full_bgr,
            opencv::core::Rect::new(band_x, band_y, band_width, band_height),
        )
        .map_err(|error| error.to_string())?;
        let rects = detect_reward_card_rects(&band).map_err(|error| error.to_string())?;
        let mut cards = Vec::with_capacity(rects.len());
        for (index, rect) in rects.into_iter().enumerate() {
            let card = opencv::core::Mat::roi(
                &band,
                opencv::core::Rect::new(rect.x, rect.y, rect.width, rect.height),
            )
            .map_err(|error| error.to_string())?;
            let prepared = prepare_reward_card(&card).map_err(|error| error.to_string())?;
            let icon_match = self
                .ensure_item_icon_recognizer()?
                .recognize_item_bgr_125(&prepared.icon)
                .map_err(|error| format!("奖励卡片 {} 的 ItemV2 识别失败: {error}", index + 1))?;
            let identity = icon_match.map(|matched| RewardIdentity {
                name: matched.name,
                quality_level: (matched.quality_level > 0).then_some(matched.quality_level),
            });
            let count = if let Some(count_image) = &prepared.count {
                let result = self
                    .ensure_ocr()?
                    .recognize_cropped_bgr_mat(&count_image.normalized)
                    .map_err(|error| format!("奖励卡片 {} 的数量 OCR 失败: {error}", index + 1))?;
                reward_count_from_ocr(&result, count_image)
            } else {
                None
            };
            self.log_pub.emit(
                LogLevel::Info,
                "CV",
                format!(
                    "[奖励卡片] #{} 名称={} 数量={}",
                    index + 1,
                    identity.as_ref().map_or("未知", |item| item.name.as_str()),
                    count.map_or_else(|| "未知".into(), |value| value.to_string())
                ),
            );
            cards.push(RewardCardObservation {
                rect: RewardCardRect {
                    x: rect.x + band_x,
                    y: rect.y + band_y,
                    width: rect.width,
                    height: rect.height,
                },
                identity,
                count,
            });
        }
        Ok(RewardPageObservation(cards))
    }

    pub(crate) fn detect_artifact_salvage_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<ArtifactSalvageStep, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let observation = self
            .ensure_artifact_salvage_matcher()?
            .detect(&full_bgr)
            .map_err(|e| e.to_string())?;
        let mut icon_matches = vec![None; MAX_SALVAGE_CELLS_PER_PAGE];
        if observation.salvage_page_visible()
            && let Some(grid) = observation.grid
            && grid.cells.iter().any(|cell| cell.present)
        {
            let recognizer = self.ensure_item_icon_recognizer()?;
            for cell in grid.cells {
                if !cell.present {
                    continue;
                }
                let icon = resize_card_for_icon(&full_bgr, cell.rect).map_err(|e| e.to_string())?;
                icon_matches[cell.index as usize] =
                    recognizer.recognize_bgr_125(&icon).map_err(|error| {
                        format!("圣遗物格子 {} 的 ItemV2 识别失败: {error}", cell.index)
                    })?;
            }
        }
        Ok(ArtifactSalvageStep {
            observation,
            icon_matches,
        })
    }

    pub(crate) fn detect_auto_wood_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<(AutoWoodObservation, bool), String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let observation = self
            .ensure_auto_wood_matcher()?
            .detect(&full_bgr)
            .map_err(|e| e.to_string())?;
        let gameplay = self
            .ensure_ui_screens()?
            .detect(&full_bgr)
            .map_err(|e| e.to_string())?
            .is_gameplay();
        Ok((observation, gameplay))
    }

    pub(crate) fn detect_fishing_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<FishingObservation, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let (_, _, ocr) = self
            .ensure_ocr()?
            .recognize_frame_roi(
                bytes,
                width,
                height,
                is_bgr24,
                fishing_bite_roi(width, height),
            )
            .map_err(|e| e.to_string())?;
        let bite_text = (!ocr.text.trim().is_empty()).then_some(ocr.text.as_str());
        self.ensure_fishing_matcher()?
            .detect(&full_bgr, bite_text)
            .map_err(|e| e.to_string())
    }

    pub(crate) fn detect_expedition_observation(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<ExpeditionObservation, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        self.ensure_expedition_matcher()?
            .observe(&full_bgr)
            .map_err(|e| e.to_string())
    }

    pub(crate) fn detect_expedition_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<Option<ExpeditionPageState>, String> {
        let observation = self.detect_expedition_observation(bytes, width, height, is_bgr24)?;
        let state = observation.state();
        self.log_pub.emit(
            LogLevel::Info,
            "CV",
            format!(
                "[派遣页面只读识别] {}",
                state.map_or_else(
                    || "未检测到唯一明确图标；不能推断为无奖励或零完成".to_string(),
                    |state| state.label().to_string(),
                )
            ),
        );
        Ok(state)
    }

    pub(crate) fn detect_mining_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<MiningObservation, String> {
        let started = Instant::now();
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|error| error.to_string())?;
        let detections = self
            .ensure_mining_detector()?
            .detect_bgr(&full_bgr)
            .map_err(|error| error.to_string())?;
        let detections = detections
            .into_iter()
            .map(|detection| MiningDetection {
                x: detection.x,
                y: detection.y,
                width: detection.width,
                height: detection.height,
                confidence: detection.confidence,
            })
            .collect::<Vec<_>>();
        let rects = filter_ore_detections(&detections);
        let observation = select_nearest_mining_target(&rects, width, height, false);
        self.log_pub.emit(
            LogLevel::Info,
            "CV",
            format!(
                "[自动采矿只读识别] ore框={} 聚类={} 目标={} ({:.1}ms)",
                rects.len(),
                observation.cluster_count,
                observation.target.map_or_else(
                    || "无".into(),
                    |target| format!("({}, {})", target.center_x, target.center_y)
                ),
                started.elapsed().as_secs_f32() * 1000.0
            ),
        );
        Ok(observation)
    }

    pub(crate) fn detect_auto_pick_prompt(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<AutoPickObservation, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|error| error.to_string())?;
        self.ensure_auto_pick_matcher()?
            .observe(&full_bgr)
            .map_err(|error| error.to_string())
    }

    pub(crate) fn detect_auto_pick_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<(AutoPickObservation, Vec<OcrHit>), String> {
        let observation = self.detect_auto_pick_prompt(bytes, width, height, is_bgr24)?;
        let hits = if let Some(prompt) = observation.prompt {
            let roi = prompt.text_roi;
            self.ocr_roi_step(
                bytes, width, height, is_bgr24, roi.x0, roi.y0, roi.x1, roi.y1,
            )?
        } else {
            Vec::new()
        };
        Ok((observation, hits))
    }

    pub(crate) fn detect_chest_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<(ChestObservation, bool), String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let chest = self
            .ensure_chest_matcher()?
            .detect(&full_bgr)
            .map_err(|e| e.to_string())?;
        let gameplay = self
            .ensure_ui_screens()?
            .detect(&full_bgr)
            .map_err(|e| e.to_string())?
            .is_gameplay();
        Ok((chest, gameplay))
    }

    pub(crate) fn detect_rewards_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<(Vec<bgi_task::RewardCandidate>, bool), String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        self.ensure_reward_matcher()?
            .detect(&full_bgr)
            .map_err(|e| e.to_string())
    }

    fn assets_root(&self) -> &std::path::Path {
        self.assets.root()
    }

    pub(crate) fn detect_bigmap_layer_buttons_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<BigMapLayerButtons, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let btns = self
            .ensure_ui_matcher()?
            .detect_bigmap_layer_buttons(&full_bgr)
            .map_err(|e| e.to_string())?;
        Ok(BigMapLayerButtons {
            underground_switch: btns.underground_switch.map(|(x, y, _)| (x, y)),
            to_ground: btns.to_ground.map(|(x, y, _)| (x, y)),
        })
    }

    pub(crate) fn detect_motion_status_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<MotionStatus, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let status = self
            .ensure_ui_matcher()?
            .detect_motion_status(&full_bgr)
            .map_err(|e| e.to_string())?;
        Ok(map_motion_status(status))
    }

    pub(crate) fn classify_ui_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> UiCategory {
        let full_bgr = match frame_to_bgr_mat(bytes, width, height, is_bgr24) {
            Ok(frame) => frame,
            Err(_) => return UiCategory::Unknown,
        };
        if let Ok(screens) = self
            .ensure_ui_screens()
            .and_then(|matcher| matcher.detect(&full_bgr).map_err(|e| e.to_string()))
        {
            let category = classify_screens(&screens);
            if category != UiCategory::Unknown {
                return category;
            }
        }
        let previous = self.last_minimap_pos;
        let result = self.locate_minimap_step(bytes, width, height, is_bgr24, None);
        self.last_minimap_pos = previous;
        match result {
            Ok(Some(_)) => UiCategory::Gameplay,
            Ok(None) => UiCategory::Loading,
            Err(_) => UiCategory::Unknown,
        }
    }

    pub(crate) fn detect_auto_eat_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> bgi_task::AutoEatObservation {
        let detected = frame_to_bgr_mat(bytes, width, height, is_bgr24)
            .ok()
            .and_then(|frame| self.ensure_ui_screens().ok()?.detect_auto_eat(&frame).ok());
        match detected {
            Some((low_hp, recovery_icon, resurrection_icon)) => bgi_task::AutoEatObservation {
                low_hp,
                recovery_icon,
                resurrection_icon,
            },
            None => bgi_task::AutoEatObservation::default(),
        }
    }

    pub(crate) fn reset_cooking_ui_cache(&mut self) {
        self.cooking_ui_checked = None;
        self.cooking_page = false;
        self.cooking_confirm = None;
        self.cooking_recover = None;
    }

    pub(crate) fn detect_cooking_ui_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<CookingUiState, String> {
        if self
            .cooking_ui_checked
            .is_some_and(|checked| checked.elapsed() < Duration::from_millis(400))
        {
            return Ok(CookingUiState {
                page_visible: self.cooking_page,
                white_confirm: self.cooking_confirm,
                recover_visible: self.cooking_recover.is_some(),
            });
        }
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let matcher = self.ensure_ui_screens()?;
        let screens = matcher.detect(&full_bgr).map_err(|e| e.to_string())?;
        let (confirm, recover) = if screens.cooking_page {
            (
                matcher
                    .find_white_confirm(&full_bgr)
                    .map_err(|e| e.to_string())?,
                matcher
                    .find_white_recover(&full_bgr)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            (None, None)
        };
        self.cooking_page = screens.cooking_page;
        self.cooking_confirm = confirm;
        self.cooking_recover = recover;
        self.cooking_ui_checked = Some(Instant::now());
        Ok(CookingUiState {
            page_visible: self.cooking_page,
            white_confirm: self.cooking_confirm,
            recover_visible: self.cooking_recover.is_some(),
        })
    }

    pub(crate) fn locate_minimap_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        prev_override: Option<(f32, f32)>,
    ) -> Result<Option<MinimapObservation>, String> {
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let minimap_bgr = extract_minimap_bgr(&full_bgr).map_err(|e| e.to_string())?;
        let hint = prev_override.or(self.last_minimap_pos);
        let mut located = self
            .ensure_minimap()?
            .locate(&minimap_bgr, hint, 2)
            .map_err(|e| e.to_string())?;
        if located.is_none() && hint.is_some() && prev_override.is_some() {
            self.log_pub.emit(
                LogLevel::Debug,
                "CV",
                format!("[小地图] 局部提示 {hint:?} 未命中，回退全图匹配"),
            );
            located = self
                .ensure_minimap()?
                .locate(&minimap_bgr, None, 2)
                .map_err(|e| e.to_string())?;
        }
        if let Some(m) = located {
            let motion = self
                .ensure_ui_matcher()?
                .detect_motion_status(&full_bgr)
                .map(map_motion_status)
                .unwrap_or(MotionStatus::Normal);
            self.last_minimap_pos = Some((m.game_x, m.game_y));
            Ok(Some(MinimapObservation {
                game_x: m.game_x,
                game_y: m.game_y,
                confidence: m.confidence,
                camera_angle: m.camera_angle,
                camera_confidence: m.camera_confidence,
                motion,
            }))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn locate_bigmap_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
    ) -> Result<Option<(bgi_pathing::BigMapProjection, usize)>, String> {
        let started = Instant::now();
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let res = self.ensure_sift()?.locate_big_map_rect(&full_bgr);
        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;
        match res {
            Ok(m) => {
                let span_x = m.game_tl_x - m.game_br_x;
                let span_y = m.game_tl_y - m.game_br_y;
                if m.inliers < 12 {
                    self.record_failed_capture(
                        "bigmap_low_inliers",
                        &format!(
                            "[大地图 SIFT] 内点数偏低 center=({:.1},{:.1}) tl=({:.1},{:.1}) br=({:.1},{:.1}) span=({span_x:.0},{span_y:.0}) kps={} good={} inliers={} src={width}x{height} ({elapsed_ms:.1}ms)",
                            m.game_center_x,
                            m.game_center_y,
                            m.game_tl_x,
                            m.game_tl_y,
                            m.game_br_x,
                            m.game_br_y,
                            m.query_kps,
                            m.good_matches,
                            m.inliers,
                        ),
                        bytes,
                        width,
                        height,
                        is_bgr24,
                        true,
                    );
                } else {
                    self.log_pub.emit(
                        LogLevel::Info,
                        "CV",
                        format!(
                            "[大地图 SIFT] 命中 center=({:.1},{:.1}) tl=({:.1},{:.1}) br=({:.1},{:.1}) span=({span_x:.0},{span_y:.0}) kps={} good={} inliers={} src={width}x{height} ({elapsed_ms:.1}ms)",
                            m.game_center_x,
                            m.game_center_y,
                            m.game_tl_x,
                            m.game_tl_y,
                            m.game_br_x,
                            m.game_br_y,
                            m.query_kps,
                            m.good_matches,
                            m.inliers,
                        ),
                    );
                }
                Ok(Some(to_projection(m)))
            }
            Err(err) => {
                self.record_failed_capture(
                    "bigmap_sift_fail",
                    &format!("[大地图 SIFT 匹配失败] ({elapsed_ms:.1}ms) {err}"),
                    bytes,
                    width,
                    height,
                    is_bgr24,
                    true,
                );
                Ok(None)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ocr_roi_step(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        norm_x0: f32,
        norm_y0: f32,
        norm_x1: f32,
        norm_y1: f32,
    ) -> Result<Vec<OcrHit>, String> {
        let started = Instant::now();
        let (ox, oy, res) = self
            .ensure_ocr()?
            .recognize_frame_roi(
                bytes,
                width,
                height,
                is_bgr24,
                NormalizedRoi {
                    x0: norm_x0,
                    y0: norm_y0,
                    x1: norm_x1,
                    y1: norm_y1,
                },
            )
            .map_err(|e| e.to_string())?;
        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;
        let hits: Vec<OcrHit> = res
            .regions
            .into_iter()
            .map(|r| OcrHit {
                text: r.text,
                center_x: ox + (r.x + r.width * 0.5).round() as i32,
                center_y: oy + (r.y + r.height * 0.5).round() as i32,
                confidence: r.score,
            })
            .collect();
        let summary: Vec<String> = hits
            .iter()
            .map(|h| format!("'{}'@({},{})", h.text, h.center_x, h.center_y))
            .collect();
        self.log_pub.emit(
            LogLevel::Info,
            "OCR",
            format!(
                "[OCR ROI] rect=({norm_x0:.2},{norm_y0:.2})->({norm_x1:.2},{norm_y1:.2}) src={width}x{height} 命中 {} 条: [{}] ({elapsed_ms:.1}ms)",
                hits.len(),
                summary.join(", "),
            ),
        );
        Ok(hits)
    }

    pub(crate) fn set_assets_dir(&mut self, dir: PathBuf) {
        self.assets = AssetCatalog::load(&dir, "zh-CN")
            .or_else(|_| AssetCatalog::embedded_fallback(&dir))
            .unwrap_or_else(|_| {
                AssetCatalog::embedded_fallback(".").expect("working directory exists")
            });
        self.sift = None;
        self.minimap = None;
        self.ui_matcher = None;
        self.chest_matcher = None;
        self.auto_pick_matcher = None;
        self.fishing_matcher = None;
        self.expedition_matcher = None;
        self.auto_wood_matcher = None;
        self.artifact_salvage_matcher = None;
        self.item_icon_recognizer = None;
        self.mining_detector = None;
        self.weapon_name_catalog = None;
        self.ocr = None;
        self.last_minimap_pos = None;
    }

    pub(crate) fn unload_idle(&mut self) -> Vec<&'static str> {
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

    pub(crate) fn probe_minimap(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        seq: u64,
    ) -> Result<CvDebugResult, String> {
        let started = Instant::now();
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
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

        let located = match self.ensure_minimap()?.locate(&minimap_bgr, None, 2) {
            Ok(opt) => opt,
            Err(err) => {
                self.record_failed_capture(
                    "minimap_probe_err",
                    &format!("[小地图测试异常] {err}"),
                    bytes,
                    width,
                    height,
                    is_bgr24,
                    false,
                );
                return Err(err.to_string());
            }
        };

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
            self.record_failed_capture(
                "minimap_probe_miss",
                &format!(
                    "[小地图测试未命中] cam={cam_angle:.1} (conf={cam_conf:.2}), src={width}x{height}"
                ),
                bytes,
                width,
                height,
                is_bgr24,
                false,
            );
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

    pub(crate) fn probe_bigmap_sift(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        seq: u64,
    ) -> Result<CvDebugResult, String> {
        let started = Instant::now();
        let full_bgr =
            frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let res = self.ensure_sift()?.locate_big_map_rect(&full_bgr);
        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;
        let m = match res {
            Ok(v) => v,
            Err(err) => {
                let err_str = err.to_string();
                let saved = self.record_failed_capture(
                    "bigmap_sift_probe_fail",
                    &format!("[大地图 SIFT 测试失败] ({elapsed_ms:.1}ms) {err_str}"),
                    bytes,
                    width,
                    height,
                    is_bgr24,
                    false,
                );
                let suffix = saved
                    .map(|p| format!(" (已保存截图: {})", p.display()))
                    .unwrap_or_default();
                return Err(format!("{err_str}{suffix}"));
            }
        };

        let span_x = m.game_tl_x - m.game_br_x;
        let span_y = m.game_tl_y - m.game_br_y;
        if m.inliers < 12 {
            self.record_failed_capture(
                "bigmap_sift_low_inliers",
                &format!(
                    "[大地图 SIFT 测试] 内点数偏低 center=({:.1},{:.1}) tl=({:.1},{:.1}) br=({:.1},{:.1}) span=({span_x:.0},{span_y:.0}) kps={} good={} inliers={} src={width}x{height} ({elapsed_ms:.1}ms)",
                    m.game_center_x,
                    m.game_center_y,
                    m.game_tl_x,
                    m.game_tl_y,
                    m.game_br_x,
                    m.game_br_y,
                    m.query_kps,
                    m.good_matches,
                    m.inliers,
                ),
                bytes,
                width,
                height,
                is_bgr24,
                false,
            );
        }

        Ok(CvDebugResult {
            updated_seq: seq,
            source_label: format!(
                "大地图 SIFT (kps={}, good={}, inliers={}, span={span_x:.0}x{span_y:.0})",
                m.query_kps, m.good_matches, m.inliers
            ),
            game_x: Some(m.game_center_x),
            game_y: Some(m.game_center_y),
            match_confidence: None,
            floor: Some(0),
            camera_angle: None,
            camera_confidence: None,
            character_angle: None,
            ocr_text: None,
            elapsed_ms: Some(elapsed_ms),
        })
    }

    pub(crate) fn probe_ocr(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        seq: u64,
    ) -> Result<CvDebugResult, String> {
        let started = Instant::now();
        let bgr = frame_to_bgr_mat(bytes, width, height, is_bgr24).map_err(|e| e.to_string())?;
        let res = self
            .ensure_ocr()?
            .recognize_bgr_mat(&bgr)
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

const WEAPON_MIN_OCR_CONFIDENCE: f32 = 0.9;
const TALENT_MIN_OCR_CONFIDENCE: f32 = 0.9;
const CHARACTER_ATTRIBUTE_MIN_OCR_CONFIDENCE: f32 = 0.9;
const REWARD_COUNT_MIN_OCR_CONFIDENCE: f32 = 0.9;

fn weapon_roi_to_normalized(roi: WeaponPixelRoi, width: u32, height: u32) -> NormalizedRoi {
    let x0 = roi.x as f32 / width as f32;
    let y0 = roi.y as f32 / height as f32;
    let x1 = (roi.x + roi.width) as f32 / width as f32;
    let y1 = (roi.y + roi.height) as f32 / height as f32;
    NormalizedRoi {
        x0: x0.next_up(),
        y0: y0.next_up(),
        x1: x1.next_down(),
        y1: y1.next_down(),
    }
}

fn weapon_details_from_hits(
    name_hits: &[OcrHit],
    level_hits: &[OcrHit],
    catalog: &WeaponNameCatalog,
) -> Option<WeaponDetails> {
    if name_hits.is_empty()
        || level_hits.is_empty()
        || name_hits
            .iter()
            .chain(level_hits)
            .any(|hit| !hit.confidence.is_finite() || hit.confidence < WEAPON_MIN_OCR_CONFIDENCE)
    {
        return None;
    }
    let name_text = name_hits
        .iter()
        .map(|hit| hit.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let level_text = level_hits
        .iter()
        .map(|hit| hit.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let matched = catalog.match_name(&name_text).ok()?;
    if !matched.reliable {
        return None;
    }
    let levels = parse_weapon_level_pair(&level_text)?;
    WeaponDetails::try_new(matched.name, levels.level, levels.limit).ok()
}

fn talent_roi_to_normalized(roi: TalentPixelRoi, width: u32, height: u32) -> NormalizedRoi {
    let x0 = roi.x as f32 / width as f32;
    let y0 = roi.y as f32 / height as f32;
    let x1 = (roi.x + roi.width) as f32 / width as f32;
    let y1 = (roi.y + roi.height) as f32 / height as f32;
    NormalizedRoi {
        x0: x0.next_up(),
        y0: y0.next_up(),
        x1: x1.next_down(),
        y1: y1.next_down(),
    }
}

fn join_ocr_hits(hits: &[OcrHit]) -> String {
    hits.iter()
        .map(|hit| hit.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn talent_details_from_hits(
    page_gate_hits: &[OcrHit],
    type_hits: &[OcrHit],
    level_hits: &[OcrHit],
    bonus_hits: &[OcrHit],
) -> Option<TalentDetails> {
    if page_gate_hits.is_empty()
        || type_hits.is_empty()
        || level_hits.is_empty()
        || page_gate_hits
            .iter()
            .chain(type_hits)
            .chain(level_hits)
            .chain(bonus_hits)
            .any(|hit| !hit.confidence.is_finite() || hit.confidence < TALENT_MIN_OCR_CONFIDENCE)
    {
        return None;
    }
    if !is_talent_detail_text(&join_ocr_hits(page_gate_hits)) {
        return None;
    }
    let talent_type = parse_talent_type(&join_ocr_hits(type_hits))?;
    let level = parse_talent_level(&join_ocr_hits(level_hits))?;
    let has_bonus = parse_talent_bonus(&join_ocr_hits(bonus_hits))?;
    TalentDetails::try_new(talent_type, level, has_bonus).ok()
}

fn character_attribute_roi_to_normalized(
    roi: CharacterAttributePixelRoi,
    width: u32,
    height: u32,
) -> NormalizedRoi {
    let x0 = roi.x as f32 / width as f32;
    let y0 = roi.y as f32 / height as f32;
    let x1 = (roi.x + roi.width) as f32 / width as f32;
    let y1 = (roi.y + roi.height) as f32 / height as f32;
    NormalizedRoi {
        x0: x0.next_up(),
        y0: y0.next_up(),
        x1: x1.next_down(),
        y1: y1.next_down(),
    }
}

fn attribute_page_from_hits(hits: &[OcrHit]) -> bool {
    hits.iter().any(|hit| {
        hit.text.contains("属性")
            && hit.confidence.is_finite()
            && hit.confidence >= CHARACTER_ATTRIBUTE_MIN_OCR_CONFIDENCE
    })
}

fn character_attribute_details_from_hits(hits: &[OcrHit]) -> Option<CharacterAttributeDetails> {
    if hits.is_empty()
        || hits.iter().any(|hit| {
            !hit.confidence.is_finite() || hit.confidence < CHARACTER_ATTRIBUTE_MIN_OCR_CONFIDENCE
        })
    {
        return None;
    }
    parse_character_level_pair(&join_ocr_hits(hits))
}

fn reward_count_from_ocr(result: &OcrResult, prepared: &RewardCountPreparation) -> Option<u32> {
    if result.regions.len() != 1
        || result.regions.iter().any(|region| {
            !region.score.is_finite() || region.score < REWARD_COUNT_MIN_OCR_CONFIDENCE
        })
    {
        return None;
    }
    let bounds = prepared.foreground_bounds?;
    let aspect = f64::from(bounds.width) / f64::from(bounds.height.max(1));
    let parsed = parse_reward_count(&result.text);
    if prepared.component_count == 1 && aspect <= 0.45 && parsed != Some(1) {
        return Some(1);
    }
    if prepared.component_count == 1 && parsed == Some(1) && aspect >= 0.65 {
        return Some(7);
    }
    parsed
}

fn fishing_bite_roi(width: u32, height: u32) -> NormalizedRoi {
    let width = width.max(1);
    let height = height.max(1);
    let x = width / 3;
    let roi_width = width / 3;
    // PaddleOCR floors the start and ceils the end. Bias normalized endpoints
    // inward so f32 rounding cannot expand or shift the requested pixel ROI.
    let x0 = (x as f32 / width as f32).next_up();
    let x1 = ((x + roi_width) as f32 / width as f32).next_down();
    NormalizedRoi {
        x0,
        y0: 0.0,
        x1,
        y1: ((height / 2) as f32 / height as f32).next_down(),
    }
}

fn classify_screens(screens: &bgi_cv::ui_templates::UiScreens) -> UiCategory {
    if screens.talk {
        UiCategory::Talk
    } else if screens.in_domain {
        UiCategory::Domain
    } else if screens.inventory {
        UiCategory::Inventory
    } else if screens.paimon_menu {
        UiCategory::Menu
    } else {
        UiCategory::Unknown
    }
}

fn map_motion_status(s: CvMotionStatus) -> MotionStatus {
    match s {
        CvMotionStatus::Normal => MotionStatus::Normal,
        CvMotionStatus::Fly => MotionStatus::Fly,
        CvMotionStatus::Climb => MotionStatus::Climb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgi_control::{UiSnapshot, create_control_bus};
    use bgi_cv::ui_templates::UiScreens;

    #[test]
    fn reward_count_ocr_rejects_low_confidence_and_preserves_structural_corrections() {
        let preparation = |width: i32, components: usize| RewardCountPreparation {
            normalized: opencv::core::Mat::default(),
            component_count: components,
            foreground_bounds: Some(bgi_task::RewardCardRect {
                x: 0,
                y: 0,
                width,
                height: 20,
            }),
        };
        let ocr = |text: &str, confidence: f32| OcrResult {
            text: text.into(),
            regions: vec![bgi_infer::OcrRegion {
                text: text.into(),
                score: confidence,
                x: 0.0,
                y: 0.0,
                width: 20.0,
                height: 20.0,
            }],
        };
        assert_eq!(
            reward_count_from_ocr(&ocr("１２", 0.9), &preparation(20, 2)),
            Some(12)
        );
        assert_eq!(
            reward_count_from_ocr(&ocr("12", 0.89), &preparation(20, 2)),
            None
        );
        assert_eq!(
            reward_count_from_ocr(&ocr("x1", 0.99), &preparation(8, 1)),
            Some(1)
        );
        assert_eq!(
            reward_count_from_ocr(&ocr("1", 0.99), &preparation(14, 1)),
            Some(7)
        );
        assert_eq!(
            reward_count_from_ocr(&ocr("1个", 0.99), &preparation(20, 2)),
            None
        );
        assert_eq!(
            reward_count_from_ocr(&ocr("", 0.99), &preparation(20, 2)),
            None
        );
    }

    #[test]
    fn weapon_name_catalog_resets_with_assets() {
        let (_, _, core) = create_control_bus(UiSnapshot::default());
        let mut services = CvOcrServices::new(PathBuf::from("."), core.log_publisher());
        services.weapon_name_catalog = Some(
            WeaponNameCatalog::from_csv("item_name,item_class_id\n西风剑,weapon:sword\n").unwrap(),
        );
        services.set_assets_dir(PathBuf::from("."));
        assert!(services.weapon_name_catalog.is_none());
    }

    #[test]
    fn fishing_matcher_is_lazy_builtin_and_resets_with_assets() {
        let (_, _, core) = create_control_bus(UiSnapshot::default());
        let mut services = CvOcrServices::new(PathBuf::from("."), core.log_publisher());
        assert!(services.fishing_matcher.is_none());

        services.ensure_fishing_matcher().unwrap();
        assert!(services.fishing_matcher.is_some());

        services.set_assets_dir(PathBuf::from("."));
        assert!(services.fishing_matcher.is_none());
    }

    #[test]
    fn expedition_matcher_is_lazy_builtin_and_never_invents_completion() {
        let (_, _, core) = create_control_bus(UiSnapshot::default());
        let mut services = CvOcrServices::new(PathBuf::from("."), core.log_publisher());
        assert!(services.expedition_matcher.is_none());

        let blank = vec![0_u8; 1920 * 1080 * 3].into_boxed_slice();
        assert!(
            services
                .detect_expedition_step(&blank, 1920, 1080, true)
                .unwrap()
                .is_none()
        );
        services.set_assets_dir(PathBuf::from("."));
        assert!(services.expedition_matcher.is_none());
    }

    #[test]
    fn weapon_ocr_requires_complete_confident_and_unambiguous_fields() {
        let catalog =
            WeaponNameCatalog::from_csv("item_name,item_class_id\n西风剑,weapon:sword\n").unwrap();
        let hit = |text: &str, confidence: f32| OcrHit {
            text: text.into(),
            center_x: 0,
            center_y: 0,
            confidence,
        };
        let name = [hit("西风剑", 0.9)];
        let level = [hit("Lv.20", 0.95), hit("90", 0.99)];
        let details = weapon_details_from_hits(&name, &level, &catalog).unwrap();
        assert_eq!(details.name, "西风剑");
        assert_eq!((details.level, details.level_limit), (20, 90));
        assert_eq!(
            weapon_details_from_hits(&[hit("西风创", 0.99)], &level, &catalog)
                .unwrap()
                .name,
            "西风剑"
        );

        for invalid_name in [
            vec![],
            vec![hit("西风剑", 0.89)],
            vec![hit("西风剑", f32::NAN)],
            vec![hit("赤凰剑", 0.99)],
            vec![hit("完全不相关的名字", 0.99)],
        ] {
            assert!(weapon_details_from_hits(&invalid_name, &level, &catalog).is_none());
        }
        let ambiguous = WeaponNameCatalog::from_csv(
            "item_name,item_class_id\n西风剑,weapon:sword\n西风创,weapon:sword\n",
        )
        .unwrap();
        assert!(weapon_details_from_hits(&[hit("西风锻", 0.99)], &level, &ambiguous).is_none());
        for invalid_level in [
            vec![],
            vec![hit("20", 0.99)],
            vec![hit("90/20", 0.99)],
            vec![hit("20", 0.99), hit("90", 0.89)],
        ] {
            assert!(weapon_details_from_hits(&name, &invalid_level, &catalog).is_none());
        }
    }

    #[test]
    fn weapon_normalized_rois_preserve_scaled_pixel_bounds() {
        for (width, height) in [(1920, 1080), (1280, 720), (2560, 1440)] {
            let rois = weapon_rois(width, height).unwrap();
            for pixel in [rois.name, rois.level] {
                let normalized = weapon_roi_to_normalized(pixel, width, height);
                assert_eq!((normalized.x0 * width as f32).floor() as u32, pixel.x);
                assert_eq!((normalized.y0 * height as f32).floor() as u32, pixel.y);
                assert_eq!(
                    (normalized.x1 * width as f32).ceil() as u32,
                    pixel.x + pixel.width
                );
                assert_eq!(
                    (normalized.y1 * height as f32).ceil() as u32,
                    pixel.y + pixel.height
                );
            }
        }
    }

    #[test]
    fn talent_ocr_requires_page_gate_and_complete_confident_fields() {
        let hit = |text: &str, confidence: f32| OcrHit {
            text: text.into(),
            center_x: 0,
            center_y: 0,
            confidence,
        };
        let page = [hit("天赋介绍", 0.95)];
        let talent_type = [hit("元素战技", 0.99)];
        let level = [hit("Lv.10", 0.99)];
        let details = talent_details_from_hits(&page, &talent_type, &level, &[]).unwrap();
        assert_eq!(details.talent_type, bgi_task::TalentType::Skill);
        assert_eq!(details.level, 10);
        assert!(!details.has_bonus);
        let bonus =
            talent_details_from_hits(&page, &talent_type, &level, &[hit("天赋等级+3", 0.9)])
                .unwrap();
        assert!(bonus.has_bonus);

        for invalid_page in [
            vec![],
            vec![hit("普通攻击", 0.99)],
            vec![hit("天赋介绍", 0.89)],
            vec![hit("天赋介绍", f32::INFINITY)],
        ] {
            assert!(talent_details_from_hits(&invalid_page, &talent_type, &level, &[]).is_none());
        }
        for invalid_type in [
            vec![],
            vec![hit("普通攻击 元素战技", 0.99)],
            vec![hit("元素战技", 0.89)],
        ] {
            assert!(talent_details_from_hits(&page, &invalid_type, &level, &[]).is_none());
        }
        for invalid_level in [
            vec![],
            vec![hit("Lv.", 0.99)],
            vec![hit("0", 0.99)],
            vec![hit("10", 0.89)],
        ] {
            assert!(talent_details_from_hits(&page, &talent_type, &invalid_level, &[]).is_none());
        }
        for invalid_bonus in [
            vec![hit("命之座", 0.99)],
            vec![hit("天赋等级+2", 0.99)],
            vec![hit("天赋等级+3", 0.89)],
            vec![hit("天赋等级+3", f32::NAN)],
        ] {
            assert!(
                talent_details_from_hits(&page, &talent_type, &level, &invalid_bonus).is_none()
            );
        }
    }

    #[test]
    fn talent_normalized_rois_preserve_scaled_pixel_bounds() {
        for (width, height) in [(1920, 1080), (1280, 720), (2560, 1440)] {
            let rois = talent_rois(width, height).unwrap();
            for pixel in [rois.page_gate, rois.talent_type, rois.level, rois.bonus] {
                let normalized = talent_roi_to_normalized(pixel, width, height);
                assert_eq!((normalized.x0 * width as f32).floor() as u32, pixel.x);
                assert_eq!((normalized.y0 * height as f32).floor() as u32, pixel.y);
                assert_eq!(
                    (normalized.x1 * width as f32).ceil() as u32,
                    pixel.x + pixel.width
                );
                assert_eq!(
                    (normalized.y1 * height as f32).ceil() as u32,
                    pixel.y + pixel.height
                );
            }
        }
    }

    #[test]
    fn character_attribute_ocr_requires_confident_page_gate_and_complete_level_pair() {
        let hit = |text: &str, confidence: f32| OcrHit {
            text: text.into(),
            center_x: 0,
            center_y: 0,
            confidence,
        };
        assert!(attribute_page_from_hits(&[hit("属性", 0.9)]));
        assert!(attribute_page_from_hits(&[
            hit("武器", 0.1),
            hit("属性", 0.99),
        ]));
        for invalid_gate in [
            vec![],
            vec![hit("武器", 0.99)],
            vec![hit("属性", 0.89)],
            vec![hit("属性", f32::NAN)],
        ] {
            assert!(!attribute_page_from_hits(&invalid_gate));
        }

        let details =
            character_attribute_details_from_hits(&[hit("Lv.90", 0.95), hit("90", 0.99)]).unwrap();
        assert_eq!((details.level, details.level_limit), (90, 90));
        for invalid_level in [
            vec![],
            vec![hit("90", 0.99)],
            vec![hit("90/89", 0.99)],
            vec![hit("90/90/90", 0.99)],
            vec![hit("90", 0.99), hit("90", 0.89)],
            vec![hit("90/90", f32::INFINITY)],
        ] {
            assert!(character_attribute_details_from_hits(&invalid_level).is_none());
        }
    }

    #[test]
    fn character_attribute_normalized_roi_preserves_scaled_pixel_bounds() {
        for (width, height) in [(1920, 1080), (1280, 720), (2560, 1440)] {
            let pixel = character_attribute_roi(width, height).unwrap();
            let normalized = character_attribute_roi_to_normalized(pixel, width, height);
            assert_eq!((normalized.x0 * width as f32).floor() as u32, pixel.x);
            assert_eq!((normalized.y0 * height as f32).floor() as u32, pixel.y);
            assert_eq!(
                (normalized.x1 * width as f32).ceil() as u32,
                pixel.x + pixel.width
            );
            assert_eq!(
                (normalized.y1 * height as f32).ceil() as u32,
                pixel.y + pixel.height
            );
        }
    }

    #[test]
    fn fishing_ocr_roi_is_exact_top_center_third() {
        for (width, height, expected) in [
            (1_920, 1_080, (640, 0, 1_280, 540)),
            (1_919, 1_079, (639, 0, 1_278, 539)),
            (1_280, 720, (426, 0, 852, 360)),
            (3, 3, (1, 0, 2, 1)),
        ] {
            let roi = fishing_bite_roi(width, height);
            // Match recognize_frame_roi's floor/ceil conversion to pixels.
            let actual = [
                (roi.x0 * width as f32).floor() as u32,
                (roi.y0 * height as f32).floor() as u32,
                (roi.x1 * width as f32).ceil() as u32,
                (roi.y1 * height as f32).ceil() as u32,
            ];
            let expected = [expected.0, expected.1, expected.2, expected.3];
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn screen_classification_uses_specific_ui_precedence() {
        assert_eq!(classify_screens(&UiScreens::default()), UiCategory::Unknown);
        assert_eq!(
            classify_screens(&UiScreens {
                talk: true,
                paimon_menu: true,
                ..Default::default()
            }),
            UiCategory::Talk
        );
        assert_eq!(
            classify_screens(&UiScreens {
                paimon_menu: true,
                ..Default::default()
            }),
            UiCategory::Menu
        );
        assert_eq!(
            classify_screens(&UiScreens {
                paimon_menu: true,
                inventory: true,
                ..Default::default()
            }),
            UiCategory::Inventory
        );
        assert_eq!(
            classify_screens(&UiScreens {
                paimon_menu: true,
                inventory: true,
                in_domain: true,
                ..Default::default()
            }),
            UiCategory::Domain
        );
    }
}
