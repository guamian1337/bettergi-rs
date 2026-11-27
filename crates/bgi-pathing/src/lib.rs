pub mod auto_pick;
pub mod executor;
pub mod fault_host;
pub mod merger;
pub mod planner;
pub mod recorder;
pub mod runner;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub use auto_pick::{AutoPickFrame, pick_once};
use bgi_control::PathSummary;
pub use executor::{
    BigMapLayerButtons, BigMapObservation, CookingObservation, MinimapObservation, MotionStatus,
    OcrHit, PathExecutor, PathingCheckpoint, PathingHost, TpTask,
};
pub use planner::BigMapProjection;
pub use recorder::PathRecorder;
pub use runner::{LoopPolicy, LoopRunner};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PathingError {
    #[error("读取路径文件失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("解析路径 JSON 失败: {0}")]
    Json(#[from] serde_json::Error),
    #[error("解析路径 JSON5 失败: {0}")]
    Json5(#[from] json5::Error),
    #[error("路径中没有任何有效点位")]
    EmptyPositions,
    #[error("路径要求 BetterGI {required}，当前版本为 {current}")]
    RequiredVersion { required: String, current: String },
    #[error("路径要求的版本号无效: {0}")]
    InvalidRequiredVersion(String),
    #[error("路径合并失败: {0}")]
    Merge(String),
    #[error("当前 BetterGI 版本号无效: {0}")]
    InvalidCurrentVersion(String),
    #[error("录制坐标必须是有限数值: ({x}, {y})")]
    InvalidRecordedPosition { x: f64, y: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PathingTaskFile {
    #[serde(default)]
    pub info: PathingInfo,
    #[serde(default)]
    pub config: PathingTaskConfig,
    #[serde(default)]
    pub farming_info: FarmingInfo,
    #[serde(default)]
    pub positions: Vec<Waypoint>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathingInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default, rename = "bgi_version")]
    pub bgi_version: Option<String>,
    #[serde(default, rename = "type")]
    pub task_type: String,
    #[serde(default)]
    pub order: i64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub enable_monster_loot_split: bool,
    #[serde(default = "default_map_name")]
    pub map_name: String,
    #[serde(default)]
    pub map_match_method: Option<String>,
    #[serde(default)]
    pub items: Vec<MaterialInfo>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

impl Default for PathingInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: None,
            author: None,
            version: None,
            bgi_version: None,
            task_type: String::new(),
            order: 0,
            tags: Vec::new(),
            enable_monster_loot_split: false,
            map_name: default_map_name(),
            map_match_method: None,
            items: Vec::new(),
            unknown: BTreeMap::new(),
        }
    }
}

fn default_map_name() -> String {
    "Teyvat".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathingTaskConfig {
    #[serde(default)]
    pub realtime_triggers: BTreeMap<String, bool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub realtime_trigger_options: BTreeMap<String, Value>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FarmingInfo {
    #[serde(default)]
    pub allow_farming_count: bool,
    #[serde(default)]
    pub normal_mob_count: f64,
    #[serde(default)]
    pub elite_mob_count: f64,
    #[serde(default)]
    pub primary_target: String,
    #[serde(default)]
    pub duration_seconds: f64,
    #[serde(default)]
    pub elite_details: String,
    #[serde(default)]
    pub total_mora: f64,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialInfo {
    pub monster: Option<String>,
    pub material: Option<String>,
    pub count: Option<String>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Misidentification {
    #[serde(rename = "type", default = "default_misidentification_types")]
    pub types: Vec<String>,
    #[serde(default = "default_misidentification_mode")]
    pub handling_mode: String,
    #[serde(default)]
    pub arrival_time: i64,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

fn default_misidentification_types() -> Vec<String> {
    vec!["unrecognized".to_string()]
}

fn default_misidentification_mode() -> String {
    "previousDetectedPoint".to_string()
}

impl Default for Misidentification {
    fn default() -> Self {
        Self {
            types: default_misidentification_types(),
            handling_mode: default_misidentification_mode(),
            arrival_time: 0,
            unknown: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PointExtParams {
    #[serde(default)]
    pub misidentification: Misidentification,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub monster_tag: Option<String>,
    #[serde(default)]
    pub enable_monster_loot_split: bool,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Waypoint {
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
    #[serde(default)]
    pub point_ext_params: PointExtParams,
    #[serde(default = "default_waypoint_type", rename = "type")]
    pub waypoint_type: String,
    #[serde(default = "default_move_mode")]
    pub move_mode: String,
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub action_params: Option<String>,
    #[serde(default)]
    pub items: Vec<MaterialInfo>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

fn default_waypoint_type() -> String {
    "path".to_string()
}

fn default_move_mode() -> String {
    "walk".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSupport {
    Implemented,
    Partial,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPhase {
    Before,
    After,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaypointConstraint {
    Any,
    Path,
    Target,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionCapability {
    pub code: &'static str,
    pub phase: ActionPhase,
    pub support: ActionSupport,
    pub waypoint: WaypointConstraint,
    pub parameters: &'static str,
}

const ACTION_CAPABILITIES: &[ActionCapability] = &[
    ActionCapability {
        code: "stop_flying",
        phase: ActionPhase::Before,
        support: ActionSupport::Partial,
        waypoint: WaypointConstraint::Any,
        parameters: "optional wait milliseconds",
    },
    ActionCapability {
        code: "log_output",
        phase: ActionPhase::After,
        support: ActionSupport::Implemented,
        waypoint: WaypointConstraint::Any,
        parameters: "required text",
    },
    ActionCapability {
        code: "use_gadget",
        phase: ActionPhase::After,
        support: ActionSupport::Partial,
        waypoint: WaypointConstraint::Any,
        parameters: "gadget name; currently key Z only",
    },
    ActionCapability {
        code: "pick_around",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "requires prompt-confirmed AutoPick; blind F fallback removed",
    },
    ActionCapability {
        code: "force_tp",
        phase: ActionPhase::Before,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "none",
    },
    ActionCapability {
        code: "nahida_collect",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "none",
    },
    ActionCapability {
        code: "fight",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Path,
        parameters: "combat configuration",
    },
    ActionCapability {
        code: "normal_attack",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "attack configuration",
    },
    ActionCapability {
        code: "elemental_skill",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "skill configuration",
    },
    ActionCapability {
        code: "up_down_grab_leaf",
        phase: ActionPhase::Before,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "none",
    },
    ActionCapability {
        code: "hydro_collect",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Target,
        parameters: "none",
    },
    ActionCapability {
        code: "electro_collect",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Target,
        parameters: "none",
    },
    ActionCapability {
        code: "anemo_collect",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Target,
        parameters: "none",
    },
    ActionCapability {
        code: "pyro_collect",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Target,
        parameters: "none",
    },
    ActionCapability {
        code: "combat_script",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "required script",
    },
    ActionCapability {
        code: "mining",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "character-specific attack combo; needs the combat/party module",
    },
    ActionCapability {
        code: "linnea_mining",
        phase: ActionPhase::After,
        support: ActionSupport::Partial,
        waypoint: WaypointConstraint::Any,
        parameters: "mines=N,rounds=M (mines do not switch characters yet)",
    },
    ActionCapability {
        code: "fishing",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "fishing configuration",
    },
    ActionCapability {
        code: "exit_and_relogin",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "none",
    },
    ActionCapability {
        code: "wonderland_cycle",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "none",
    },
    ActionCapability {
        code: "set_time",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "required time",
    },
    ActionCapability {
        code: "pick_up_collect",
        phase: ActionPhase::After,
        support: ActionSupport::Unsupported,
        waypoint: WaypointConstraint::Any,
        parameters: "pickup configuration",
    },
];

pub fn action_capabilities() -> &'static [ActionCapability] {
    ACTION_CAPABILITIES
}

pub fn action_capability(action: Option<&str>) -> Option<&'static ActionCapability> {
    let action = action?.trim();
    ACTION_CAPABILITIES.iter().find(|item| item.code == action)
}

pub fn is_basic_action_supported(action: Option<&str>) -> bool {
    action.map(str::trim).is_none_or(|action| {
        action.is_empty()
            || action_capability(Some(action))
                .is_some_and(|item| item.support != ActionSupport::Unsupported)
    })
}

pub fn load_path_file(
    path: impl AsRef<Path>,
) -> Result<(PathSummary, PathingTaskFile), PathingError> {
    load_path_file_for_version(path, env!("CARGO_PKG_VERSION"))
}

pub fn load_path_file_for_version(
    path: impl AsRef<Path>,
    current_version: &str,
) -> Result<(PathSummary, PathingTaskFile), PathingError> {
    let path_ref = path.as_ref();
    let merged = merger::merge_pathing_json(path_ref)?;
    let mut parsed: PathingTaskFile = serde_json::from_value(merged)?;
    validate_required_version(parsed.info.bgi_version.as_deref(), current_version)?;
    if parsed.positions.is_empty() {
        return Err(PathingError::EmptyPositions);
    }
    for waypoint in &mut parsed.positions {
        waypoint.point_ext_params.enable_monster_loot_split = parsed.info.enable_monster_loot_split;
    }

    let mut supported_count = 0usize;
    let mut skipped = BTreeSet::new();
    for wp in &parsed.positions {
        if is_basic_action_supported(wp.action.as_deref()) {
            supported_count += 1;
        } else if let Some(act) = wp.action.as_deref() {
            skipped.insert(act.to_string());
        }
    }

    let name = if parsed.info.name.trim().is_empty() {
        path_ref
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "未命名路径".to_string())
    } else {
        parsed.info.name.clone()
    };

    let summary = PathSummary {
        name,
        file_path: path_ref.to_string_lossy().into_owned(),
        waypoint_count: parsed.positions.len(),
        supported_count,
        skipped_actions: skipped.into_iter().collect(),
    };

    Ok((summary, parsed))
}

pub fn validate_required_version(
    required: Option<&str>,
    current: &str,
) -> Result<(), PathingError> {
    let Some(required) = required.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(());
    };
    let required = semver::Version::parse(required.strip_prefix('v').unwrap_or(required))
        .map_err(|_| PathingError::InvalidRequiredVersion(required.to_string()))?;
    let current = semver::Version::parse(current)
        .map_err(|_| PathingError::InvalidCurrentVersion(current.to_string()))?;
    if required > current {
        return Err(PathingError::RequiredVersion {
            required: required.to_string(),
            current: current.to_string(),
        });
    }
    Ok(())
}

impl PathingTaskFile {
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), PathingError> {
        let path = path.as_ref();
        let temporary = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&temporary, json)?;
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        Ok(())
    }
}

/// Scans up to `limit` `.json` path files recursively under `root`.
pub fn scan_path_directory(root: impl AsRef<Path>, limit: usize) -> Vec<PathSummary> {
    let mut files = Vec::new();
    collect_json_files(root.as_ref(), &mut files, limit);
    files
        .into_iter()
        .filter_map(|p| load_path_file(p).ok().map(|(s, _)| s))
        .collect()
}

fn collect_json_files(dir: &Path, out: &mut Vec<PathBuf>, limit: usize) {
    if out.len() >= limit || !dir.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= limit {
            break;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out, limit);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legacy_json_and_marks_complex_actions_for_skip() {
        let json = r#"{
            "info": { "name": "风起地测试", "map_name": "Teyvat" },
            "positions": [
                { "x": -1200.5, "y": 2300.2, "type": "teleport", "move_mode": "walk" },
                { "x": -1210.0, "y": 2310.0, "type": "path", "move_mode": "dash" },
                { "x": -1220.0, "y": 2320.0, "type": "target", "move_mode": "fly", "action": "stop_flying" },
                { "x": -1230.0, "y": 2330.0, "type": "target", "move_mode": "walk", "action": "nahida_collect" }
            ]
        }"#;
        let parsed: PathingTaskFile = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.positions.len(), 4);
        assert!(is_basic_action_supported(
            parsed.positions[2].action.as_deref()
        ));
        assert!(!is_basic_action_supported(
            parsed.positions[3].action.as_deref()
        ));
        assert!(!is_basic_action_supported(Some("force_tp")));
        assert!(is_basic_action_supported(Some("linnea_mining")));
        assert!(!is_basic_action_supported(Some("mining")));
        // 动作码按 C# 路径文件的大小写精确匹配。
        assert!(!is_basic_action_supported(Some("LINNEA_MINING")));
    }

    #[test]
    fn csharp_fields_and_unknown_keys_round_trip() {
        let json = r#"{
            "info": {
                "name": "route", "description": null, "author": "author", "version": "1",
                "bgi_version": "1.2.3", "type": "Normal", "order": 4,
                "tags": ["daily"], "enable_monster_loot_split": true, "map_name": "Teyvat",
                "map_match_method": "SIFT", "items": [{"monster":"m","material":"x","count":"2"}],
                "future_info": {"enabled":true}
            },
            "config": {"realtime_triggers":{"AutoPick":false,"PathRequest":true},"realtime_trigger_options":{"PathRequest":{"task_name":"Daily route","request_every_n":3}},"future_config":1},
            "farming_info": {"allow_farming_count":true,"normal_mob_count":2.5,"future_farming":[1]},
            "positions": [{
                "x": 1, "y": 2,
                "point_ext_params": {
                    "misidentification":{"type":["pathTooFar"],"handling_mode":"mapRecognition","arrival_time":9},
                    "description":"point", "monster_tag":"elite", "enable_monster_loot_split":false,
                    "future_ext":"kept"
                },
                "type":"target", "move_mode":"walk", "action":"log_output","action_params":"hello",
                "items":[{"monster":null,"material":"flower","count":"1","future_item":true}],
                "future_waypoint":[1,2]
            }],
            "future_root":"kept"
        }"#;
        let mut parsed: PathingTaskFile = serde_json::from_str(json).unwrap();
        parsed.positions[0]
            .point_ext_params
            .enable_monster_loot_split = true;
        let serialized = serde_json::to_value(&parsed).unwrap();
        let restored: PathingTaskFile = serde_json::from_value(serialized).unwrap();

        assert_eq!(restored.info.author.as_deref(), Some("author"));
        assert_eq!(restored.info.order, 4);
        assert_eq!(
            restored.config.realtime_triggers.get("AutoPick"),
            Some(&false)
        );
        assert_eq!(
            restored.config.realtime_triggers.get("PathRequest"),
            Some(&true)
        );
        assert_eq!(
            restored.config.realtime_trigger_options["PathRequest"]["task_name"],
            "Daily route"
        );
        assert_eq!(
            restored.config.realtime_trigger_options["PathRequest"]["request_every_n"],
            3
        );
        assert_eq!(restored.farming_info.normal_mob_count, 2.5);
        assert_eq!(
            restored.positions[0]
                .point_ext_params
                .misidentification
                .types,
            ["pathTooFar"]
        );
        assert_eq!(
            restored.positions[0].items[0].material.as_deref(),
            Some("flower")
        );
        assert_eq!(restored.unknown["future_root"], "kept");
        assert_eq!(restored.info.unknown["future_info"]["enabled"], true);
        assert_eq!(restored.config.unknown["future_config"], 1);
        assert_eq!(
            restored.farming_info.unknown["future_farming"],
            serde_json::json!([1])
        );
        assert_eq!(
            restored.positions[0].unknown["future_waypoint"],
            serde_json::json!([1, 2])
        );
        assert_eq!(
            restored.positions[0].point_ext_params.unknown["future_ext"],
            "kept"
        );
        assert_eq!(restored.positions[0].items[0].unknown["future_item"], true);
    }

    #[test]
    fn missing_waypoint_fields_use_csharp_defaults() {
        let parsed: Waypoint = serde_json::from_str(r#"{"x":1,"y":2}"#).unwrap();
        assert_eq!(parsed.waypoint_type, "path");
        assert_eq!(parsed.move_mode, "walk");
        assert_eq!(
            parsed.point_ext_params.misidentification.types,
            ["unrecognized"]
        );
        assert_eq!(
            parsed.point_ext_params.misidentification.handling_mode,
            "previousDetectedPoint"
        );
        assert_eq!(PathingInfo::default().map_name, "Teyvat");
        assert_eq!(
            PathingTaskConfig::default()
                .realtime_triggers
                .get("AutoPick"),
            None
        );
    }

    #[test]
    fn legacy_realtime_trigger_config_has_no_request_options() {
        let config: PathingTaskConfig =
            serde_json::from_str(r#"{"realtime_triggers":{"AutoPick":false}}"#).unwrap();
        assert_eq!(config.realtime_triggers.get("AutoPick"), Some(&false));
        assert!(config.realtime_trigger_options.is_empty());
    }

    #[test]
    fn action_table_distinguishes_implemented_partial_and_unsupported() {
        assert_eq!(
            action_capability(Some("log_output")).unwrap().support,
            ActionSupport::Implemented
        );
        assert_eq!(
            action_capability(Some("stop_flying")).unwrap().phase,
            ActionPhase::Before
        );
        assert_eq!(
            action_capability(Some("use_gadget")).unwrap().support,
            ActionSupport::Partial
        );
        assert_eq!(
            action_capability(Some("fight")).unwrap().waypoint,
            WaypointConstraint::Path
        );
        assert_eq!(
            action_capability(Some("hydro_collect")).unwrap().waypoint,
            WaypointConstraint::Target
        );
        assert_eq!(action_capability(Some("future_action")), None);
        assert!(!is_basic_action_supported(Some("pick_around")));
        assert!(!is_basic_action_supported(Some("combat_script")));
    }

    #[test]
    fn required_version_gate_rejects_newer_and_malformed_versions() {
        assert!(validate_required_version(None, "1.2.3").is_ok());
        assert!(validate_required_version(Some("1.2.3"), "1.2.3").is_ok());
        assert!(validate_required_version(Some("1.2.2"), "1.2.3").is_ok());
        assert!(matches!(
            validate_required_version(Some("1.2.4"), "1.2.3"),
            Err(PathingError::RequiredVersion { .. })
        ));
        assert!(matches!(
            validate_required_version(Some("latest"), "1.2.3"),
            Err(PathingError::InvalidRequiredVersion(_))
        ));
        assert!(matches!(
            validate_required_version(Some("1.0.0"), "bad"),
            Err(PathingError::InvalidCurrentVersion(_))
        ));
    }

    #[test]
    fn save_to_file_preserves_unknown_fields() {
        let mut task: PathingTaskFile = serde_json::from_str(
            r#"{
            "info":{"name":"saved"},
            "positions":[{"x":1,"y":2,"future":true}],
            "future_root":{"value":3}
        }"#,
        )
        .unwrap();
        task.positions[0].action = Some("log_output".into());
        let path = std::env::temp_dir().join(format!("bgi-path-save-{}.json", std::process::id()));
        task.save_to_file(&path).unwrap();
        let (_, restored) = load_path_file_for_version(&path, "1.0.0").unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(restored.unknown["future_root"]["value"], 3);
        assert_eq!(restored.positions[0].unknown["future"], true);
        assert_eq!(restored.positions[0].action.as_deref(), Some("log_output"));
    }
}
