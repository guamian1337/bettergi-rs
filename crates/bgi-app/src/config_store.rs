use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bgi_control::CoreConfig;

pub fn config_path() -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    base.join("User/core_config.json")
}

pub fn load_config(path: &Path) -> Result<CoreConfig, String> {
    if !path.exists() {
        return Ok(CoreConfig::default());
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("读取配置文件失败 ({}): {error}", path.display()))?;
    let config: CoreConfig = serde_json::from_slice(&bytes)
        .map_err(|error| format!("解析配置文件失败 ({}): {error}", path.display()))?;
    config
        .task_limits
        .validate()
        .map_err(|error| format!("配置文件任务限制无效: {error}"))?;
    Ok(config)
}

pub fn load_initial_config(path: &Path) -> (CoreConfig, Option<String>) {
    match load_config(path) {
        Ok(config) => (config, None),
        Err(error) => (CoreConfig::default(), Some(error)),
    }
}

pub fn save_config(path: &Path, config: &CoreConfig) -> Result<(), String> {
    config
        .task_limits
        .validate()
        .map_err(|error| format!("拒绝保存无效任务限制: {error}"))?;
    let parent = path
        .parent()
        .ok_or_else(|| "配置文件路径没有父目录".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("创建配置目录失败 ({}): {error}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = parent.join(format!(
        ".{}.{nonce}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("config"),
        std::process::id()
    ));
    let result = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|error| format!("创建临时配置失败: {error}"))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, config)
            .map_err(|error| format!("序列化配置失败: {error}"))?;
        writer
            .flush()
            .map_err(|error| format!("写入配置失败: {error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("同步配置失败: {error}"))?;
        drop(writer);
        fs::rename(&temp_path, path)
            .map_err(|error| format!("替换配置失败 ({}): {error}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(tag: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "bettergi-rs-config-{tag}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_legacy_and_corrupt_configs_are_handled_explicitly() {
        let dir = test_dir("load");
        let path = dir.join("core_config.json");
        assert_eq!(load_config(&path).unwrap(), CoreConfig::default());

        let mut legacy = serde_json::to_value(CoreConfig::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("task_limits");
        fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert_eq!(
            load_config(&path).unwrap().task_limits,
            bgi_control::TaskLimitsConfig::default()
        );
        let (loaded, error) = load_initial_config(&path);
        assert_eq!(loaded, load_config(&path).unwrap());
        assert_eq!(error, None);

        fs::write(&path, b"{broken").unwrap();
        assert!(load_config(&path).is_err());
        let (fallback, error) = load_initial_config(&path);
        assert_eq!(fallback, CoreConfig::default());
        assert!(error.is_some());
        assert_eq!(fs::read(&path).unwrap(), b"{broken");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn save_is_atomic_round_trip_and_rejects_invalid_limits() {
        let dir = test_dir("save");
        let path = dir.join("core_config.json");
        let mut config = CoreConfig {
            preferred_process: "YuanShen.exe".into(),
            task_limits: bgi_control::TaskLimitsConfig {
                reward_claim_max_clicks: 12,
                ..bgi_control::TaskLimitsConfig::default()
            },
            ..CoreConfig::default()
        };
        save_config(&path, &config).unwrap();
        assert_eq!(load_config(&path).unwrap(), config);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);

        config.preferred_process = "YuanShen.exe".into();
        config.task_limits.reward_claim_timeout_ms = 45_000;
        save_config(&path, &config).unwrap();
        assert_eq!(load_config(&path).unwrap(), config);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);

        config.task_limits.reward_claim_max_clicks = 0;
        assert!(save_config(&path, &config).is_err());
        assert_eq!(
            load_config(&path)
                .unwrap()
                .task_limits
                .reward_claim_max_clicks,
            12
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn legacy_limits_without_mining_fields_fall_back_to_defaults() {
        let dir = test_dir("legacy_mining");
        let path = dir.join("core_config.json");
        let mut value = serde_json::to_value(CoreConfig::default()).unwrap();
        let limits = value
            .get_mut("task_limits")
            .unwrap()
            .as_object_mut()
            .unwrap();
        limits.remove("mining_max_shots");
        limits.remove("mining_total_timeout_ms");
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();

        let defaults = bgi_control::TaskLimitsConfig::default();
        let loaded = load_config(&path).unwrap().task_limits;
        assert_eq!(loaded.mining_max_shots, defaults.mining_max_shots);
        assert_eq!(
            loaded.mining_total_timeout_ms,
            defaults.mining_total_timeout_ms
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
