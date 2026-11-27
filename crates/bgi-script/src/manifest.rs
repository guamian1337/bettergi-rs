//! `manifest.json` 与 `settings.json` 解析。
//!
//! 键名为 snake_case（与真实脚本一致：`main` / `settings_ui` / `bgi_version` / …），
//! 未知字段容忍，以便兼容将来新增的键。

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::error::ManifestError;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ManifestAuthor {
    pub name: String,
    #[serde(default)]
    pub links: String,
}

/// 脚本项目元数据，对应 C# `Core/Script/Project/Manifest.cs`。
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct Manifest {
    pub manifest_version: u32,
    pub name: String,
    pub version: String,
    pub description: String,
    pub authors: Vec<ManifestAuthor>,
    pub tags: Vec<String>,
    pub main: String,
    pub settings_ui: Option<String>,
    pub bgi_version: Option<String>,
    pub saved_files: Vec<String>,
    pub allow_js_notification: bool,
    /// 以下三项 C# 定义里有，真实脚本暂未使用，保留以兼容。
    pub library: Vec<String>,
    pub scripts: Vec<String>,
    pub http_allowed_urls: Vec<String>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            manifest_version: 1,
            name: String::new(),
            version: String::new(),
            description: String::new(),
            authors: Vec::new(),
            tags: Vec::new(),
            main: String::new(),
            settings_ui: None,
            bgi_version: None,
            saved_files: Vec::new(),
            allow_js_notification: false,
            library: Vec::new(),
            scripts: Vec::new(),
            http_allowed_urls: Vec::new(),
        }
    }
}

impl Manifest {
    pub fn parse(json: &str) -> Result<Self, ManifestError> {
        let manifest: Manifest = serde_json::from_str(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.main.trim().is_empty() {
            return Err(ManifestError::Invalid("main 不能为空".into()));
        }
        if safe_relative(&self.main).is_none() {
            return Err(ManifestError::Invalid(format!(
                "main 必须是项目内的相对路径: {}",
                self.main
            )));
        }
        if let Some(ui) = &self.settings_ui
            && safe_relative(ui).is_none()
        {
            return Err(ManifestError::Invalid(format!(
                "settings_ui 必须是项目内的相对路径: {ui}"
            )));
        }
        if let Some(version) = &self.bgi_version
            && semver::Version::parse(version).is_err()
            && semver::VersionReq::parse(version).is_err()
        {
            return Err(ManifestError::Invalid(format!(
                "bgi_version 不是合法的语义化版本: {version}"
            )));
        }
        Ok(())
    }

    /// 宿主最低版本要求。本移植版**不强制**这个约束：Rust 侧版本号为 0.1.0，
    /// 若照搬 C# 的「脚本 bgi_version ≤ 宿主版本」判断，所有现有脚本都会被拒。
    /// 只解析、只提示，由上层决定是否提示用户。
    pub fn required_bgi_version(&self) -> Option<&str> {
        self.bgi_version.as_deref()
    }
}

/// 把 `/` 或 `\` 分隔的相对路径安全地拼到 `root` 下。
/// 拒绝绝对路径、盘符、UNC、`..`、`.` 与空段；返回规范化后的路径。
pub fn safe_join(root: &Path, relative: &str) -> Option<PathBuf> {
    let normalized = safe_relative(relative)?;
    Some(root.join(normalized))
}

/// 只做校验与分隔符归一，不拼接。`\\?\` 与盘符前缀一律拒绝。
pub fn safe_relative(relative: &str) -> Option<PathBuf> {
    if relative.is_empty() {
        return None;
    }
    let trimmed = relative.trim();
    if trimmed.starts_with('\\') || trimmed.starts_with('/') || trimmed.contains(':') {
        return None;
    }
    let mut out = PathBuf::new();
    for segment in trimmed.split(['/', '\\']) {
        match segment {
            "" | "." | ".." => return None,
            _ => out.push(segment),
        }
    }
    if out.components().any(|c| !matches!(c, Component::Normal(_))) {
        return None;
    }
    Some(out)
}

/// 一个 `settings.json` 控件。
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, rename_all = "snake_case")]
#[derive(Default)]
pub struct SettingsControl {
    pub name: String,
    #[serde(rename = "type")]
    pub control_type: String,
    pub label: String,
    pub default: Option<Value>,
    pub options: Vec<String>,
}

impl SettingsControl {
    /// 按 C# 的语义推导默认值：
    /// `input-text` → `default` 或空串；`select` → `default`（须在 options 内）或第一项；
    /// `checkbox` → `default` 或 false；`multi-checkbox` → `default` 或空数组；`separator` 无值。
    pub fn default_value(&self) -> Option<Value> {
        match self.control_type.as_str() {
            "input-text" => Some(self.default.clone().unwrap_or(Value::String(String::new()))),
            "select" => {
                let fallback = self.options.first().cloned().unwrap_or_default();
                Some(match &self.default {
                    Some(Value::String(s))
                        if self.options.is_empty() || self.options.contains(s) =>
                    {
                        Value::String(s.clone())
                    }
                    Some(other) if self.options.is_empty() => other.clone(),
                    _ => Value::String(fallback),
                })
            }
            "checkbox" => Some(
                self.default
                    .as_ref()
                    .and_then(Value::as_bool)
                    .map(Value::Bool)
                    .unwrap_or(Value::Bool(false)),
            ),
            "multi-checkbox" => Some(
                self.default
                    .clone()
                    .filter(|v| v.is_array())
                    .unwrap_or_else(|| Value::Array(Vec::new())),
            ),
            "separator" => None,
            other => {
                tracing::warn!(
                    "未知的 settings 控件类型「{other}」（{}），跳过默认值",
                    self.name
                );
                None
            }
        }
    }
}

/// 解析 `settings.json`（控件数组）。
pub fn parse_settings_ui(json: &str) -> Result<Vec<SettingsControl>, ManifestError> {
    Ok(serde_json::from_str(json)?)
}

/// 由控件定义推导全部默认值，键按名称排序以便输出稳定。
pub fn settings_defaults(controls: &[SettingsControl]) -> BTreeMap<String, Value> {
    controls
        .iter()
        .filter(|c| !c.name.is_empty())
        .filter_map(|c| c.default_value().map(|v| (c.name.clone(), v)))
        .collect()
}

/// 把用户保存的设置合并到默认值上：用户值覆盖同名键，默认值补齐缺失键。
/// 不在控件定义里的用户键保留（对齐 C# 只清理 `multi-checkbox` 的行为，这里更宽松，
/// 不丢用户数据）。
pub fn merge_settings(
    defaults: &BTreeMap<String, Value>,
    saved: Option<&Value>,
) -> BTreeMap<String, Value> {
    let mut merged = defaults.clone();
    if let Some(Value::Object(saved)) = saved {
        for (key, value) in saved {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_JS_ROOT: &str = "../../../../better-genshin-impact/BetterGenshinImpact/bin/Debug/net8.0-windows10.0.22621.0/User/JsScript";

    #[test]
    fn parses_real_manifest_shape() {
        let json = r#"{
  "manifest_version": 1,
  "name": "性能测试",
  "version": "2.0",
  "tags": [],
  "bgi_version": "0.46.0",
  "description": "测测你的",
  "authors": [{"name": "mno", "links": "https://github.com/Bedrockx"}],
  "settings_ui": "settings.json",
  "main": "main.js",
  "future_field": {"ignored": true}
}"#;
        let manifest = Manifest::parse(json).unwrap();
        assert_eq!(manifest.name, "性能测试");
        assert_eq!(manifest.main, "main.js");
        assert_eq!(manifest.settings_ui.as_deref(), Some("settings.json"));
        assert_eq!(manifest.required_bgi_version(), Some("0.46.0"));
        assert!(!manifest.allow_js_notification);
        assert_eq!(manifest.authors[0].name, "mno");
    }

    #[test]
    fn rejects_missing_or_escaping_main() {
        assert!(Manifest::parse(r#"{"name":"x"}"#).is_err(), "main 缺失");
        for main in [
            "../evil.js",
            "a/../../b.js",
            "/abs/main.js",
            "C://x//main.js",
            "",
        ] {
            let json = serde_json::json!({ "name": "x", "main": main });
            assert!(
                Manifest::parse(&json.to_string()).is_err(),
                "应拒绝 main={main:?}"
            );
        }
    }

    #[test]
    fn rejects_bad_version_but_accepts_partial() {
        let bad = serde_json::json!({ "name": "x", "main": "main.js", "bgi_version": "banana" });
        assert!(Manifest::parse(&bad.to_string()).is_err());
        let ok = serde_json::json!({ "name": "x", "main": "main.js", "bgi_version": "0.55.1" });
        assert!(Manifest::parse(&ok.to_string()).is_ok());
    }

    #[test]
    fn safe_relative_normalizes_separators_and_rejects_traversal() {
        assert_eq!(
            safe_relative("assets\\imgs\\a.png"),
            Some(PathBuf::from("assets").join("imgs").join("a.png"))
        );
        assert_eq!(safe_relative("main.js"), Some(PathBuf::from("main.js")));
        for bad in [
            "..",
            "a/../b",
            "./a",
            "a//b",
            "\\\\server\\share",
            "D:/x",
            "a:b",
        ] {
            assert_eq!(safe_relative(bad), None, "应拒绝 {bad:?}");
        }
    }

    #[test]
    fn settings_defaults_follow_control_types() {
        let controls = vec![
            SettingsControl {
                name: "t".into(),
                control_type: "input-text".into(),
                ..Default::default()
            },
            SettingsControl {
                name: "n".into(),
                control_type: "input-text".into(),
                default: Some(Value::String("1".into())),
                ..Default::default()
            },
            SettingsControl {
                name: "s".into(),
                control_type: "select".into(),
                options: vec!["a".into(), "b".into()],
                ..Default::default()
            },
            SettingsControl {
                name: "sd".into(),
                control_type: "select".into(),
                options: vec!["a".into(), "b".into()],
                default: Some(Value::String("b".into())),
                ..Default::default()
            },
            SettingsControl {
                name: "sx".into(),
                control_type: "select".into(),
                options: vec!["a".into()],
                default: Some(Value::String("zzz".into())),
                ..Default::default()
            },
            SettingsControl {
                name: "c".into(),
                control_type: "checkbox".into(),
                ..Default::default()
            },
            SettingsControl {
                name: "cd".into(),
                control_type: "checkbox".into(),
                default: Some(Value::Bool(true)),
                ..Default::default()
            },
            SettingsControl {
                name: "m".into(),
                control_type: "multi-checkbox".into(),
                ..Default::default()
            },
            SettingsControl {
                name: "-".into(),
                control_type: "separator".into(),
                ..Default::default()
            },
        ];
        let d = settings_defaults(&controls);
        assert_eq!(d["t"], Value::String(String::new()));
        assert_eq!(d["n"], Value::String("1".into()));
        assert_eq!(
            d["s"],
            Value::String("a".into()),
            "select 无 default 取第一项"
        );
        assert_eq!(d["sd"], Value::String("b".into()));
        assert_eq!(
            d["sx"],
            Value::String("a".into()),
            "default 不在 options 内时回退第一项"
        );
        assert_eq!(d["c"], Value::Bool(false));
        assert_eq!(d["cd"], Value::Bool(true));
        assert_eq!(d["m"], Value::Array(Vec::new()));
        assert!(!d.contains_key("-"), "separator 不产生值");
    }

    #[test]
    fn merge_settings_overrides_and_fills() {
        let controls = vec![
            SettingsControl {
                name: "a".into(),
                control_type: "input-text".into(),
                default: Some(Value::String("def".into())),
                ..Default::default()
            },
            SettingsControl {
                name: "b".into(),
                control_type: "checkbox".into(),
                ..Default::default()
            },
        ];
        let defaults = settings_defaults(&controls);
        let saved = serde_json::json!({ "a": "user", "extra": 1 });
        let merged = merge_settings(&defaults, Some(&saved));
        assert_eq!(merged["a"], Value::String("user".into()));
        assert_eq!(merged["b"], Value::Bool(false), "用户没写的键用默认值补齐");
        assert_eq!(merged["extra"], 1, "不在控件定义里的用户键保留");
        assert_eq!(merge_settings(&defaults, None), defaults);
        assert_eq!(merge_settings(&defaults, Some(&Value::Null)), defaults);
    }

    #[test]
    fn loads_every_real_manifest_and_settings_file() {
        let root = Path::new(REAL_JS_ROOT);
        if !root.is_dir() {
            return;
        }
        let mut count = 0;
        let mut with_settings = 0;
        for entry in std::fs::read_dir(root).unwrap().flatten() {
            let manifest_path = entry.path().join("manifest.json");
            if !manifest_path.is_file() {
                continue;
            }
            let manifest = Manifest::parse(&std::fs::read_to_string(&manifest_path).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e}", entry.path().display()));
            let main = safe_join(&entry.path(), &manifest.main).unwrap();
            assert!(main.is_file(), "main 不存在: {}", main.display());
            if let Some(ui) = &manifest.settings_ui {
                let ui_path = safe_join(&entry.path(), ui).unwrap();
                let text = std::fs::read_to_string(&ui_path).unwrap();
                let controls = parse_settings_ui(&text)
                    .unwrap_or_else(|e| panic!("{}: {e}", ui_path.display()));
                assert!(!controls.is_empty());
                with_settings += 1;
            }
            count += 1;
        }
        assert!(count >= 10, "真实项目数 {count}");
        assert!(
            with_settings >= 8,
            "带 settings_ui 的项目数 {with_settings}"
        );
    }
}
