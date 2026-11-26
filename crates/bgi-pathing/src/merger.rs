use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::PathingError;

const CONTROL_FILE: &str = "control.json5";
const MAX_REF_DEPTH: usize = 32;

pub fn merge_pathing_json(path: &Path) -> Result<Value, PathingError> {
    let original = read_json5(path)?;
    match merge_pathing_json_strict(path) {
        Ok(merged) => Ok(merged),
        Err(_) => Ok(original),
    }
}

pub fn merge_pathing_json_strict(path: &Path) -> Result<Value, PathingError> {
    let original = read_json5(path)?;
    let Some(parent) = path.parent() else {
        return Ok(original);
    };
    let control_path = parent.join(CONTROL_FILE);
    if !control_path.is_file() {
        return Ok(original);
    }
    let control = resolve_control(&control_path, 0, &mut BTreeSet::new())?;
    let name = path
        .file_stem()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(apply_control(&control, original, &name))
}

fn read_json5(path: &Path) -> Result<Value, PathingError> {
    let content = fs::read_to_string(path)?;
    Ok(json5::from_str(&content)?)
}

fn resolve_control(
    path: &Path,
    depth: usize,
    seen: &mut BTreeSet<PathBuf>,
) -> Result<Value, PathingError> {
    if depth >= MAX_REF_DEPTH {
        return Err(PathingError::Merge("control.json5 ref 链过深".into()));
    }
    let path = if path.is_dir() {
        path.join(CONTROL_FILE)
    } else {
        path.to_path_buf()
    };
    let canonical = fs::canonicalize(&path).map_err(|error| {
        PathingError::Merge(format!("无法解析控制文件 {}: {error}", path.display()))
    })?;
    if !seen.insert(canonical.clone()) {
        return Err(PathingError::Merge(format!(
            "控制文件 ref 链存在循环: {}",
            canonical.display()
        )));
    }
    let value = read_json5(&canonical)?;
    let Some(object) = value.as_object() else {
        return Err(PathingError::Merge("控制文件根节点必须是对象".into()));
    };
    let Some(reference) = object
        .get("ref")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return Ok(value);
    };
    let reference_path = PathBuf::from(reference);
    let resolved = if reference_path.is_absolute() {
        reference_path
    } else {
        canonical
            .parent()
            .unwrap_or(Path::new("."))
            .join(reference_path)
    };
    resolve_control(&resolved, depth + 1, seen)
}

fn apply_control(control: &Value, mut original: Value, name: &str) -> Value {
    let Some(control_object) = control.as_object() else {
        return original;
    };
    let Some(original_object) = original.as_object_mut() else {
        return original;
    };
    if let Some(global_cover) = control_object
        .get("global_cover")
        .and_then(Value::as_object)
    {
        merge_object(global_cover, original_object);
    }
    if let Some(entries) = control_object.get("json_list").and_then(Value::as_array) {
        for entry in entries {
            let matches_name = entry
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|entry_name| entry_name == name);
            if matches_name && let Some(cover) = entry.get("cover").and_then(Value::as_object) {
                merge_object(cover, original_object);
                break;
            }
        }
    }
    original
}

fn merge_object(control: &Map<String, Value>, target: &mut Map<String, Value>) {
    let mut skipped = BTreeSet::new();
    if let Some(keys) = control.get("_obj_cover").and_then(Value::as_array) {
        for key in keys.iter().filter_map(Value::as_str) {
            if let Some(value) = control.get(key) {
                target.insert(key.to_string(), value.clone());
                skipped.insert(key.to_string());
            }
        }
        skipped.insert("_obj_cover".to_string());
    }
    if let Some(keys) = control.get("_arr_add").and_then(Value::as_array) {
        for key in keys.iter().filter_map(Value::as_str) {
            if let Some(Value::Array(source)) = control.get(key) {
                let merged = match target.remove(key) {
                    Some(Value::Array(existing)) => merge_arrays(&existing, source),
                    _ => source.clone(),
                };
                target.insert(key.to_string(), Value::Array(merged));
                skipped.insert(key.to_string());
            }
        }
        skipped.insert("_arr_add".to_string());
    }

    for (key, control_value) in control {
        if skipped.contains(key) {
            continue;
        }
        match control_value {
            Value::Object(source) => match target.get_mut(key) {
                Some(Value::Object(existing)) => merge_object(source, existing),
                _ => {
                    target.insert(key.clone(), Value::Object(source.clone()));
                }
            },
            _ => {
                target.insert(key.clone(), control_value.clone());
            }
        }
    }
}

fn merge_arrays(target: &[Value], source: &[Value]) -> Vec<Value> {
    let mut result = Vec::with_capacity(target.len() + source.len());
    let mut seen = BTreeSet::new();
    for value in target.iter().chain(source) {
        if let Ok(serialized) = serde_json::to_string(value)
            && seen.insert(serialized)
        {
            result.push(value.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn applies_global_named_and_special_merge_rules() {
        let control = json!({
            "global_cover": {
                "info": { "author": "global", "tags": ["a"] },
                "nested": { "left": 1 },
                "recursive": { "left": 1 },
                "items": [2, 3],
                "_obj_cover": ["nested"],
                "_arr_add": ["items"]
            },
            "json_list": [{
                "name": "route",
                "cover": { "info": { "author": "named" } }
            }]
        });
        let original = json!({
            "info": { "name": "route", "author": "original" },
            "nested": { "right": 2 },
            "recursive": { "right": 2 },
            "items": [1, 2, 2]
        });
        let merged = apply_control(&control, original, "route");
        assert_eq!(merged["info"]["author"], "named");
        assert_eq!(merged["info"]["tags"], json!(["a"]));
        assert_eq!(merged["nested"], json!({"left": 1}));
        assert_eq!(merged["recursive"], json!({"left": 1, "right": 2}));
        assert_eq!(merged["items"], json!([1, 2, 3]));
    }

    #[test]
    fn ref_cycles_are_rejected() {
        let root =
            std::env::temp_dir().join(format!("bgi-json5-cycle-{}-cycle", std::process::id(),));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.json5"), r#"{"ref":"b.json5"}"#).unwrap();
        fs::write(root.join("b.json5"), r#"{"ref":"a.json5"}"#).unwrap();
        let result = resolve_control(&root.join("a.json5"), 0, &mut BTreeSet::new());
        assert!(matches!(result, Err(PathingError::Merge(_))));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_json5_control_refs_and_applies_version_gate() {
        let root =
            std::env::temp_dir().join(format!("bgi-json5-load-{}-load", std::process::id(),));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("base.json5"),
            r#"{
                // JSON5 comment
                "global_cover": {
                    "info": { "author": "merged", "bgi_version": "1.5.0", },
                    "config": { "realtime_triggers": { "AutoPick": false, }, },
                },
            }"#,
        )
        .unwrap();
        fs::write(root.join("control.json5"), r#"{"ref":"base.json5"}"#).unwrap();
        fs::write(
            root.join("route.json"),
            r#"{
                "info": { "name": "route", },
                "positions": [{ "x": 1, "y": 2, "future": true, },],
                "future_root": 9,
            }"#,
        )
        .unwrap();

        let (summary, task) =
            crate::load_path_file_for_version(root.join("route.json"), "1.5.0").unwrap();
        assert_eq!(summary.name, "route");
        assert_eq!(task.info.author.as_deref(), Some("merged"));
        assert_eq!(task.config.realtime_triggers.get("AutoPick"), Some(&false));
        assert_eq!(task.unknown["future_root"], 9);
        assert_eq!(task.positions[0].unknown["future"], true);
        assert!(matches!(
            crate::load_path_file_for_version(root.join("route.json"), "1.4.9"),
            Err(crate::PathingError::RequiredVersion { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_control_falls_back_to_original() {
        let root = std::env::temp_dir().join(format!(
            "bgi-json5-fallback-{}-fallback",
            std::process::id(),
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("control.json5"), "not json").unwrap();
        fs::write(
            root.join("route.json"),
            r#"{"info":{"name":"original"},"positions":[{"x":1,"y":2}]}"#,
        )
        .unwrap();
        assert!(merge_pathing_json_strict(&root.join("route.json")).is_err());
        let merged = merge_pathing_json(&root.join("route.json")).unwrap();
        assert_eq!(merged["info"]["name"], "original");
        fs::remove_dir_all(root).unwrap();
    }
}
