pub mod executor;
pub mod planner;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use bgi_control::PathSummary;
pub use executor::{BigMapObservation, MinimapObservation, OcrHit, PathExecutor, PathingHost, TpTask};
pub use planner::BigMapProjection;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PathingError {
    #[error("读取路径文件失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("解析路径 JSON 失败: {0}")]
    Json(#[from] serde_json::Error),
    #[error("路径中没有任何有效点位")]
    EmptyPositions,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PathingTaskFile {
    #[serde(default)]
    pub info: PathingInfo,
    #[serde(default)]
    pub positions: Vec<Waypoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PathingInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub map_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waypoint {
    pub x: f64,
    pub y: f64,
    #[serde(default = "default_waypoint_type", rename = "type")]
    pub waypoint_type: String,
    #[serde(default = "default_move_mode")]
    pub move_mode: String,
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub action_params: Option<String>,
}

fn default_waypoint_type() -> String {
    "path".to_string()
}

fn default_move_mode() -> String {
    "walk".to_string()
}

/// MVP basic action classification: complex combat/mining/fishing/elemental actions are skipped.
pub fn is_basic_action_supported(action: Option<&str>) -> bool {
    match action.map(str::trim) {
        None | Some("") | Some("stop_flying") | Some("force_tp") | Some("log_output") => true,
        Some(_) => false,
    }
}

pub fn load_path_file(path: impl AsRef<Path>) -> Result<(PathSummary, PathingTaskFile), PathingError> {
    let path_ref = path.as_ref();
    let content = fs::read_to_string(path_ref)?;
    let parsed: PathingTaskFile = serde_json::from_str(&content)?;
    if parsed.positions.is_empty() {
        return Err(PathingError::EmptyPositions);
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
        assert!(is_basic_action_supported(parsed.positions[2].action.as_deref()));
        assert!(!is_basic_action_supported(parsed.positions[3].action.as_deref()));
    }
}
