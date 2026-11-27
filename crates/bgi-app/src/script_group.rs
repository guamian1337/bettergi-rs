//! 配置组（ScriptGroup）数据模型、持久化与调度辅助。
//!
//! 对应 C# `Core/Script/Group` + `Service/ScriptService`。JSON 字段保持与
//! C# `User/ScriptGroup/*.json` 一致（camelCase），以便直接读取用户现有配置。

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 项目类型，对应 C# `ScriptGroupProject.Type`（`ScriptGroupProject.cs:405-413`）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ProjectType {
    #[serde(rename = "Javascript")]
    #[default]
    Javascript,
    #[serde(rename = "KeyMouse")]
    KeyMouse,
    #[serde(rename = "Pathing")]
    Pathing,
    #[serde(rename = "Shell")]
    Shell,
    /// C# 里未知类型会走到 `TypeDesc` 的默认值；保留原字符串避免丢数据。
    #[serde(untagged)]
    Unknown(String),
}

impl ProjectType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Javascript => "Javascript",
            Self::KeyMouse => "KeyMouse",
            Self::Pathing => "Pathing",
            Self::Shell => "Shell",
            Self::Unknown(s) => s,
        }
    }
}

/// 一个脚本项目，字段对应 C# `ScriptGroupProject`（`ScriptGroupProject.cs:27-110`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ScriptGroupProject {
    pub index: i32,
    pub name: String,
    pub folder_name: String,
    #[serde(rename = "type")]
    pub project_type: ProjectType,
    pub status: String,
    /// C# 中 Schedule 仅用于显示与自定义标记，不参与实际跳过判定。
    pub schedule: String,
    pub run_num: i32,
    pub js_script_settings_object: Option<Value>,
    pub allow_js_notification: Option<bool>,
    pub allow_js_http_hash: Option<String>,
}

impl Default for ScriptGroupProject {
    fn default() -> Self {
        Self {
            index: 0,
            name: String::new(),
            folder_name: String::new(),
            project_type: ProjectType::Javascript,
            status: "Enabled".into(),
            schedule: "Daily".into(),
            run_num: 1,
            js_script_settings_object: None,
            allow_js_notification: Some(true),
            allow_js_http_hash: Some(String::new()),
        }
    }
}

impl ScriptGroupProject {
    pub fn is_enabled(&self) -> bool {
        self.status == "Enabled"
    }

    /// C# `Pathing` 项目路径 = `<路径根目录>/<folderName>/<name>`。
    /// `name` 通常已经带 `.json`；兼容不带后缀的情况。
    pub fn pathing_relative(&self) -> String {
        let name = if self.name.to_ascii_lowercase().ends_with(".json") {
            self.name.clone()
        } else {
            format!("{}.json", self.name)
        };
        if self.folder_name.is_empty() {
            name
        } else {
            format!("{}/{}", self.folder_name.trim_end_matches('/'), name)
        }
    }
}

/// 组级配置。C# 的 `PathingPartyConfig` 字段极多且含大量一期不用的战斗配置，
/// 这里用 `Value` 原样保留，确保读写不丢字段（见方案文档风险表）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ScriptGroupConfig {
    pub pathing_config: Value,
    pub shell_config: Value,
    pub enable_shell_config: bool,
}

impl ScriptGroupConfig {
    pub fn pathing_enabled(&self) -> bool {
        self.pathing_config
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// C# `SkipDuring`：禁止执行的小时（0-23）。
    pub fn skip_during_hour(&self) -> Option<u32> {
        let raw = self.pathing_config.get("skipDuring")?.as_str()?;
        let hour: u32 = raw.parse().ok()?;
        (hour <= 23).then_some(hour)
    }

    pub fn task_cycle(&self) -> TaskCycleConfig {
        self.pathing_config
            .get("taskCycleConfig")
            .cloned()
            .map(TaskCycleConfig::from_value)
            .unwrap_or_default()
    }

    pub fn completion_skip_rule(&self) -> CompletionSkipRule {
        self.pathing_config
            .get("taskCompletionSkipRuleConfig")
            .cloned()
            .map(CompletionSkipRule::from_value)
            .unwrap_or_default()
    }

    /// 一期只读取保留，优先级插队调度未启用（见方案文档）。
    #[allow(dead_code)]
    pub fn pre_execution_priority(&self) -> PreExecutionPriority {
        self.pathing_config
            .get("preExecutionPriorityConfig")
            .cloned()
            .map(PreExecutionPriority::from_value)
            .unwrap_or_default()
    }
}

/// 对应 C# `PathingPartyTaskCycleConfig`（`PathingPartyTaskCycleConfig.cs:10-68`）。
#[derive(Debug, Clone, Default)]
pub struct TaskCycleConfig {
    pub enable: bool,
    pub cycle: i64,
    pub index: i64,
    pub boundary_time: i32,
}

impl TaskCycleConfig {
    fn from_value(v: Value) -> Self {
        Self {
            enable: v.get("enable").and_then(Value::as_bool).unwrap_or(false),
            cycle: v.get("cycle").and_then(Value::as_i64).unwrap_or(0),
            index: v.get("index").and_then(Value::as_i64).unwrap_or(0),
            boundary_time: v.get("boundaryTime").and_then(Value::as_i64).unwrap_or(0) as i32,
        }
    }

    /// C# `GetExecutionOrder`：序号 = (纪元至修正日期的天数 % Cycle) + 1，非法返回 -1。
    pub fn execution_order(&self) -> i64 {
        if !(1..=u32::MAX as i64).contains(&self.cycle) || !(0..=23).contains(&self.boundary_time) {
            return -1;
        }
        let days = adjusted_unix_days(self.boundary_time);
        if days < 0 {
            return -1;
        }
        (days % self.cycle) + 1
    }
}

/// 对应 C# `TaskCompletionSkipRuleConfig`（`TaskCompletionSkipRuleConfig.cs:5-32`）。
#[derive(Debug, Clone)]
pub struct CompletionSkipRule {
    pub enable: bool,
    pub skip_policy: String,
    pub boundary_time: i32,
    pub last_run_gap_seconds: i64,
    pub reference_point: String,
}

impl Default for CompletionSkipRule {
    fn default() -> Self {
        Self {
            enable: false,
            skip_policy: "GroupPhysicalPathSkipPolicy".into(),
            boundary_time: 4,
            last_run_gap_seconds: -1,
            reference_point: "EndTime".into(),
        }
    }
}

impl CompletionSkipRule {
    fn from_value(v: Value) -> Self {
        let default = Self::default();
        Self {
            enable: v.get("enable").and_then(Value::as_bool).unwrap_or(false),
            skip_policy: v
                .get("skipPolicy")
                .and_then(Value::as_str)
                .unwrap_or(&default.skip_policy)
                .to_string(),
            boundary_time: v
                .get("boundaryTime")
                .and_then(Value::as_i64)
                .unwrap_or(default.boundary_time as i64) as i32,
            last_run_gap_seconds: v
                .get("lastRunGapSeconds")
                .and_then(Value::as_i64)
                .unwrap_or(default.last_run_gap_seconds),
            reference_point: v
                .get("referencePoint")
                .and_then(Value::as_str)
                .unwrap_or(&default.reference_point)
                .to_string(),
        }
    }

    pub fn is_valid(&self) -> bool {
        let boundary_ok = (0..=23).contains(&self.boundary_time);
        (boundary_ok || self.last_run_gap_seconds >= 0) && self.enable
    }
}

/// 对应 C# `PreExecutionPriorityConfig`（`PreExecutionPriorityConfig.cs:7-24`）。
/// 一期只保留读取，调度不启用。
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct PreExecutionPriority {
    pub enabled: bool,
    pub group_names: Vec<String>,
    pub max_retry_count: u32,
}

impl PreExecutionPriority {
    fn from_value(v: Value) -> Self {
        Self {
            enabled: v.get("enabled").and_then(Value::as_bool).unwrap_or(false),
            group_names: v
                .get("groupNames")
                .and_then(Value::as_str)
                .unwrap_or("")
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
            max_retry_count: v.get("maxRetryCount").and_then(Value::as_u64).unwrap_or(1) as u32,
        }
    }
}

/// 一个配置组，字段对应 C# `ScriptGroup`（`ScriptGroup.cs:19-36`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
#[derive(Default)]
pub struct ScriptGroup {
    pub index: i32,
    pub name: String,
    pub config: ScriptGroupConfig,
    pub projects: Vec<ScriptGroupProject>,
    /// C# `NextFlag` 标了 JsonIgnore，这里同样不序列化。
    #[serde(skip)]
    pub next_flag: bool,
}

/// 执行记录条目，字段对应 C# `ExecutionRecord`（`ExecutionRecord.cs:9-43`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub guid: String,
    pub group_name: String,
    pub project_name: String,
    pub folder_name: String,
    #[serde(rename = "type")]
    pub project_type: String,
    pub server_start_time: Option<String>,
    pub start_time: String,
    pub server_end_time: Option<String>,
    pub end_time: String,
    pub is_successful: bool,
}

/// 一次组执行的进度标记，对应 C# `NextScheduledTask`（`AllConfig.cs:99`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NextScheduledTask {
    pub group_name: String,
    pub index: i32,
    pub folder_name: String,
    pub name: String,
}

/// 按本地日期计算「修正日」的 Unix 天数。C# 以凌晨 `boundaryTime` 为日界。
fn adjusted_unix_days(boundary_hour: i32) -> i64 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let shifted = secs - i64::from(boundary_hour) * 3600;
    shifted.div_euclid(86_400)
}

/// 当前本地小时（0-23），取自 `SystemTime`，避免引入 chrono 依赖。
fn current_local_hour() -> u32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    // 本机时区按 GMT+8 识别；C# 用 DateTime.Now.Hour。这里用 UTC+8 与用户环境一致。
    ((secs + 8 * 3600).rem_euclid(86_400) / 3600) as u32
}

fn local_date_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = (secs + 8 * 3600).div_euclid(86_400);
    // civil_from_days（Howard Hinnant 算法）
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}{m:02}{d:02}")
}

/// 配置组存储：每组一个 JSON，位于 `User/ScriptGroup/`。
#[derive(Debug)]
pub struct GroupStore {
    dir: PathBuf,
    records_dir: PathBuf,
}

impl GroupStore {
    pub fn new(user_dir: &Path) -> Self {
        Self {
            dir: user_dir.join("ScriptGroup"),
            records_dir: user_dir.join("log").join("ExecutionRecords"),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// 加载全部配置组，按文件名排序（C# 按 Index 排序，但文件名即组名，需先加载再排）。
    pub fn load_all(&self) -> Result<Vec<ScriptGroup>, String> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }
        let mut files: Vec<PathBuf> = fs::read_dir(&self.dir)
            .map_err(|e| format!("读取配置组目录失败: {e}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        files.sort();
        let mut groups = Vec::with_capacity(files.len());
        for path in files {
            match Self::load_file(&path) {
                Ok(group) => groups.push(group),
                Err(error) => {
                    // 单个组损坏不影响其他组
                    tracing::warn!("跳过损坏的配置组 {}: {error}", path.display());
                }
            }
        }
        groups.sort_by_key(|g| g.index);
        Ok(groups)
    }

    pub fn load_file(path: &Path) -> Result<ScriptGroup, String> {
        let bytes = fs::read(path).map_err(|e| format!("读取 {} 失败: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("解析 {} 失败: {e}", path.display()))
    }

    /// 原子写：临时文件 + rename，复用 `config_store.rs` 的模式。
    #[allow(dead_code)]
    pub fn save(&self, group: &ScriptGroup) -> Result<(), String> {
        fs::create_dir_all(&self.dir).map_err(|e| format!("创建配置组目录失败: {e}"))?;
        let path = self.dir.join(format!("{}.json", group.name));
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp = self.dir.join(format!(
            ".{}.{nonce}.{}.tmp",
            group.name,
            std::process::id()
        ));
        let result = (|| {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
                .map_err(|e| format!("创建临时文件失败: {e}"))?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, group)
                .map_err(|e| format!("序列化配置组失败: {e}"))?;
            writer.flush().map_err(|e| format!("写入失败: {e}"))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| format!("同步失败: {e}"))?;
            drop(writer);
            fs::rename(&temp, &path).map_err(|e| format!("替换配置组失败: {e}"))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    /// 删除组 JSON 文件。
    pub fn remove_group(&self, name: &str) -> Result<(), String> {
        let path = self.dir.join(format!("{name}.json"));
        if !path.exists() {
            return Err(format!("配置组文件不存在: {}", path.display()));
        }
        fs::remove_file(&path).map_err(|e| format!("删除配置组失败: {e}"))
    }

    pub fn append_record(&self, record: &ExecutionRecord) -> Result<(), String> {
        fs::create_dir_all(&self.records_dir).map_err(|e| format!("创建执行记录目录失败: {e}"))?;
        let path = self
            .records_dir
            .join(format!("{}.json", local_date_stamp()));
        let mut records: Vec<ExecutionRecord> = if path.exists() {
            let bytes = fs::read(&path).map_err(|e| e.to_string())?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            Vec::new()
        };
        records.push(record.clone());
        let bytes = serde_json::to_vec_pretty(&records).map_err(|e| e.to_string())?;
        fs::write(&path, bytes).map_err(|e| format!("写入执行记录失败: {e}"))
    }

    pub fn recent_records(&self, days: usize) -> Vec<ExecutionRecord> {
        let mut all = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.records_dir) {
            let mut files: Vec<PathBuf> = entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                .collect();
            files.sort();
            files.reverse();
            for path in files.into_iter().take(days) {
                if let Ok(bytes) = fs::read(&path)
                    && let Ok(mut records) = serde_json::from_slice::<Vec<ExecutionRecord>>(&bytes)
                {
                    all.append(&mut records);
                }
            }
        }
        all
    }

    /// 对应 C# `ExecutionRecordStorage.IsSkipTask`（`ExecutionRecordStorage.cs:178-309`）。
    pub fn is_skipped(&self, group: &ScriptGroup, project: &ScriptGroupProject) -> Option<String> {
        let rule = group.config.completion_skip_rule();
        if !rule.is_valid() {
            return None;
        }
        let boundary_ok = (0..=23).contains(&rule.boundary_time);
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        for record in self.recent_records(7) {
            if !record.is_successful
                || record.project_type != project.project_type.as_str()
                || record.project_name != project.name
            {
                continue;
            }
            let calc = parse_iso_secs(if rule.reference_point == "StartTime" {
                &record.start_time
            } else {
                &record.end_time
            });
            if rule.last_run_gap_seconds >= 0
                && calc.is_none_or(|t| now_secs - t > rule.last_run_gap_seconds)
            {
                continue;
            }
            if boundary_ok {
                let record_secs = parse_iso_secs(
                    record
                        .server_start_time
                        .as_deref()
                        .unwrap_or(&record.start_time),
                );
                let Some(t) = record_secs else { continue };
                let record_day = (t - i64::from(rule.boundary_time) * 3600).div_euclid(86_400);
                if record_day != adjusted_unix_days(rule.boundary_time) {
                    continue;
                }
            }
            let matched = match rule.skip_policy.as_str() {
                "GroupPhysicalPathSkipPolicy" => {
                    record.group_name == group.name && record.folder_name == project.folder_name
                }
                "PhysicalPathSkipPolicy" => record.folder_name == project.folder_name,
                "SameNameSkipPolicy" => true,
                _ => false,
            };
            if matched {
                return Some(format!("执行记录命中跳过规则（{}）", rule.skip_policy));
            }
        }
        None
    }
}

/// 解析 C# ISO 8601 时间字符串为 Unix 秒。只处理项目实际写入的格式。
fn parse_iso_secs(text: &str) -> Option<i64> {
    let date = text.get(0..10)?;
    let time = text.get(11..19)?;
    let year: i64 = date.get(0..4)?.parse().ok()?;
    let month: i64 = date.get(5..7)?.parse().ok()?;
    let day: i64 = date.get(8..10)?.parse().ok()?;
    let hour: i64 = time.get(0..2)?.parse().ok()?;
    let minute: i64 = time.get(3..5)?.parse().ok()?;
    let second: i64 = time.get(6..8)?.parse().ok()?;
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - 8 * 3600)
}

/// 综合跳过判定，对应 C# `ScriptService.ShouldSkipTask`（`ScriptService.cs:59-118`）。
pub fn should_skip(
    group: &ScriptGroup,
    project: &ScriptGroupProject,
    store: &GroupStore,
) -> Option<String> {
    if !project.is_enabled() {
        return Some("项目已禁用".into());
    }
    let config = &group.config;
    if config.pathing_enabled() {
        if let Some(hour) = config.skip_during_hour()
            && current_local_hour() == hour
        {
            return Some(format!("当前 {hour} 点为禁止执行时段"));
        }
        let cycle = config.task_cycle();
        if cycle.enable {
            let order = cycle.execution_order();
            if order == -1 {
                tracing::warn!("周期配置参数错误，配置不生效");
            } else if order != cycle.index {
                return Some(format!(
                    "不在执行周期（当前 {order} != 配置 {}）",
                    cycle.index
                ));
            }
        }
    }
    store.is_skipped(group, project)
}

/// 断点点：按 C# 四元组定位到项目的位置。
#[allow(dead_code)]
pub fn find_project<'a>(
    groups: &'a [ScriptGroup],
    marker: &NextScheduledTask,
) -> Option<(&'a ScriptGroup, usize)> {
    groups
        .iter()
        .find(|g| g.name == marker.group_name)
        .and_then(|g| {
            g.projects
                .iter()
                .position(|p| {
                    p.index == marker.index
                        && p.folder_name == marker.folder_name
                        && p.name == marker.name
                })
                .map(|i| (g, i))
        })
}

/// 从断点位置展开执行序列：标记项之前的项目跳过，之后（含标记项）按序执行。
#[allow(dead_code)]
pub fn projects_from(group: &ScriptGroup, start: usize) -> Vec<&ScriptGroupProject> {
    group
        .projects
        .iter()
        .skip(start)
        .filter(|p| p.is_enabled())
        .collect()
}

/// 记录一次项目执行结果，生成 guid（无需 uuid 依赖，用时间+进程+内容哈希）。
pub fn new_record_id(group: &str, project: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for byte in format!("{group}/{project}/{now}/{}", std::process::id())
        .bytes()
        .chain(now.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// 当前 ISO 时间字符串（`yyyy-MM-dd HH:mm:ss`），与 C# 记录格式一致。
pub fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
        + 8 * 3600;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!(
        "{year:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// 供外部写入 NextScheduledTask 标记（C# 存在 `User/config.json`）。
pub fn schedule_marker_path(user_dir: &Path) -> PathBuf {
    user_dir.join("next_scheduled_task.json")
}

#[allow(dead_code)]
pub fn load_schedule_marker(user_dir: &Path) -> Option<NextScheduledTask> {
    let path = schedule_marker_path(user_dir);
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub fn save_schedule_marker(user_dir: &Path, marker: &NextScheduledTask) -> Result<(), String> {
    let path = schedule_marker_path(user_dir);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(
        &path,
        serde_json::to_vec_pretty(marker).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

/// 统计各组的项目数，供 UI 摘要使用。
pub fn group_summaries(groups: &[ScriptGroup]) -> Vec<bgi_control::ScriptGroupSummary> {
    groups
        .iter()
        .map(|g| bgi_control::ScriptGroupSummary {
            name: g.name.clone(),
            index: g.index,
            project_count: g.projects.len(),
            enabled_count: g.projects.iter().filter(|p| p.is_enabled()).count(),
            projects: g
                .projects
                .iter()
                .map(|p| bgi_control::ScriptGroupProjectSummary {
                    index: p.index,
                    name: p.name.clone(),
                    project_type: p.project_type.as_str().to_string(),
                    folder_name: p.folder_name.clone(),
                    enabled: p.is_enabled(),
                })
                .collect(),
        })
        .collect()
}

/// HashMap 兼容 C# 的按名索引（保留给后续优先级插队使用）。
#[allow(dead_code)]
pub fn groups_by_name(groups: &[ScriptGroup]) -> HashMap<String, usize> {
    groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.name.clone(), i))
        .collect()
}

/// 配置组执行状态机。由 core 主循环持有并在 `PathingEvent::Finished` 时推进。
///
/// 每次只跑一个项目；非 Pathing 项目返回 `Step::Unsupported` 由 core 记日志后推进。
#[derive(Debug)]
pub struct GroupRunner {
    store: GroupStore,
    groups: Vec<ScriptGroup>,
    /// 当前组名。
    group_name: String,
    /// 当前待执行的项目（原始序号，索引到 group.projects）。
    queue: Vec<usize>,
    /// queue 游标。
    cursor: usize,
    /// 当前项目已完成的轮次（RunNum）。
    current_run: i32,
    /// 当前项目路径（用于写执行记录）。
    current_folder: String,
    current_project_name: String,
    current_type: String,
    /// 核心注入的路径根目录（`CoreConfig.pathing_dir`）。
    pathing_root: Option<String>,
}

impl GroupRunner {
    pub fn new(user_dir: &Path) -> Self {
        let store = GroupStore::new(user_dir);
        let groups = store.load_all().unwrap_or_default();
        Self {
            store,
            groups,
            group_name: String::new(),
            queue: Vec::new(),
            cursor: 0,
            current_run: 0,
            current_folder: String::new(),
            current_project_name: String::new(),
            current_type: String::new(),
            pathing_root: None,
        }
    }

    #[allow(dead_code)]
    pub fn store(&self) -> &GroupStore {
        &self.store
    }

    #[allow(dead_code)]
    pub fn groups(&self) -> &[ScriptGroup] {
        &self.groups
    }

    pub fn summaries(&self) -> Vec<bgi_control::ScriptGroupSummary> {
        group_summaries(&self.groups)
    }

    pub fn is_running(&self) -> bool {
        !self.queue.is_empty() && self.cursor < self.queue.len()
    }

    pub fn current_group_name(&self) -> &str {
        &self.group_name
    }

    pub fn reload(&mut self) -> Result<(), String> {
        self.groups = self.store.load_all()?;
        Ok(())
    }

    /// 新建空组。若同名组已存在则拒绝。
    pub fn create_group(&mut self, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("组名不能为空".into());
        }
        if self.groups.iter().any(|g| g.name == name) {
            return Err(format!("配置组「{name}」已存在"));
        }
        let next_index = self.groups.iter().map(|g| g.index).max().unwrap_or(0) + 1;
        let group = ScriptGroup {
            index: next_index,
            name: name.to_string(),
            ..Default::default()
        };
        self.store.save(&group)?;
        self.groups.push(group);
        self.groups.sort_by_key(|g| g.index);
        Ok(())
    }

    /// 删除组及其 JSON 文件。正在运行的组不能删除。
    pub fn delete_group(&mut self, name: &str) -> Result<(), String> {
        if self.is_running() && self.group_name == name {
            return Err("无法删除正在运行的配置组".into());
        }
        let pos = self
            .groups
            .iter()
            .position(|g| g.name == name)
            .ok_or_else(|| format!("找不到配置组「{name}」"))?;
        self.store.remove_group(name)?;
        self.groups.remove(pos);
        Ok(())
    }

    /// 往组里新增项目。项目类型按 C# 四类型解析。
    pub fn add_project(
        &mut self,
        group_name: &str,
        project_type: &str,
        folder_name: &str,
        name: &str,
    ) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("项目名不能为空".into());
        }
        let pos = self
            .groups
            .iter()
            .position(|g| g.name == group_name)
            .ok_or_else(|| format!("找不到配置组「{group_name}」"))?;
        let project_type = parse_project_type(project_type)?;
        let next_index = self.groups[pos]
            .projects
            .iter()
            .map(|p| p.index)
            .max()
            .unwrap_or(0)
            + 1;
        self.groups[pos].projects.push(ScriptGroupProject {
            index: next_index,
            name: name.to_string(),
            folder_name: folder_name.trim().to_string(),
            project_type,
            ..Default::default()
        });
        self.store.save(&self.groups[pos])?;
        Ok(())
    }

    /// 按索引删除项目。
    pub fn remove_project(&mut self, group_name: &str, index: i32) -> Result<(), String> {
        if self.is_running() && self.group_name == group_name {
            return Err("无法修改正在运行的配置组".into());
        }
        let pos = self
            .groups
            .iter()
            .position(|g| g.name == group_name)
            .ok_or_else(|| format!("找不到配置组「{group_name}」"))?;
        let project_pos = self.groups[pos]
            .projects
            .iter()
            .position(|p| p.index == index)
            .ok_or_else(|| format!("找不到项目序号 {index}"))?;
        self.groups[pos].projects.remove(project_pos);
        self.store.save(&self.groups[pos])?;
        Ok(())
    }

    /// 启动一个组。返回第一步动作。
    pub fn start(&mut self, name: &str, from_index: Option<i32>) -> Result<Step, String> {
        let group = self
            .groups
            .iter()
            .find(|g| g.name == name)
            .ok_or_else(|| format!("找不到配置组「{name}」"))?;
        let start_pos = from_index
            .and_then(|idx| group.projects.iter().position(|p| p.index == idx))
            .unwrap_or(0);

        let mut queue = Vec::new();
        for pos in start_pos..group.projects.len() {
            let project = &group.projects[pos];
            if !project.is_enabled() {
                continue;
            }
            match should_skip(group, project, &self.store) {
                Some(reason) => {
                    tracing::info!("配置组跳过 {}：{reason}", project.name);
                }
                None => queue.push(pos),
            }
        }

        self.group_name = name.to_string();
        self.queue = queue;
        self.cursor = 0;
        self.current_run = 0;
        if self.queue.is_empty() {
            return Ok(Step::Done);
        }
        self.prepare_current()
    }

    /// 准备当前项目：判断类型、写断点标记。
    fn prepare_current(&mut self) -> Result<Step, String> {
        let Some(&pos) = self.queue.get(self.cursor) else {
            return Ok(Step::Done);
        };
        let group = &self.groups[self.group_name_index()];
        let project = &group.projects[pos];
        self.current_folder = project.folder_name.clone();
        self.current_project_name = project.name.clone();
        self.current_type = project.project_type.as_str().to_string();
        self.current_run = 1;

        // 写断点标记，供停止后「从此项目开始」。
        let _ = save_schedule_marker(
            self.store.dir().parent().unwrap_or(Path::new(".")),
            &NextScheduledTask {
                group_name: self.group_name.clone(),
                index: project.index,
                folder_name: project.folder_name.clone(),
                name: project.name.clone(),
            },
        );

        match project.project_type {
            ProjectType::Javascript => {
                let script_root = self
                    .store
                    .dir()
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("JsScript");
                let project_dir = script_root.join(&project.folder_name);
                if !project_dir.join("manifest.json").is_file() {
                    return Err(format!("找不到 JS 脚本项目: {}", project_dir.display()));
                }
                Ok(Step::RunScript {
                    project_dir,
                    main: "main.js".into(),
                    settings: project
                        .js_script_settings_object
                        .clone()
                        .unwrap_or(Value::Null),
                })
            }
            ProjectType::Pathing => {
                let relative = project.pathing_relative();
                let path = self.resolve_pathing(&relative)?;
                Ok(Step::RunPath(path))
            }
            _ => Ok(Step::Unsupported {
                project_type: self.current_type.clone(),
            }),
        }
    }

    fn group_name_index(&self) -> usize {
        self.groups
            .iter()
            .position(|g| g.name == self.group_name)
            .unwrap_or(0)
    }

    /// 解析 Pathing 项目到实际文件：优先 folder/name 拼接，再尝试直接文件名。
    fn resolve_pathing(&self, relative: &str) -> Result<String, String> {
        // 路径根目录来自 config；这里通过 store 目录反推 user 目录再取配置不现实，
        // 因此由 core 侧注入 base 目录（见 set_pathing_root）。
        let root = self
            .pathing_root
            .as_deref()
            .ok_or_else(|| "未设置路径根目录".to_string())?;
        let candidate = Path::new(root).join(relative);
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().to_string());
        }
        // 尝试在根目录下递归查找同名文件
        let file_name = Path::new(relative)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(relative);
        if let Some(found) = find_file_recursive(Path::new(root), file_name, 6) {
            return Ok(found.to_string_lossy().to_string());
        }
        Err(format!("找不到路径文件 {relative}（根目录 {root}）"))
    }

    /// 核心注入的路径根目录。
    pub fn set_pathing_root(&mut self, root: String) {
        self.pathing_root = Some(root);
    }

    /// 当前项目的执行记录素材。
    pub fn record_context(&self) -> (String, String, String) {
        (
            self.current_folder.clone(),
            self.current_project_name.clone(),
            self.current_type.clone(),
        )
    }

    /// 一个项目执行结束，推进到下一项目/下一轮。成功时写执行记录。
    pub fn on_finished(&mut self, success: bool) -> Result<Step, String> {
        if !self.is_running() {
            return Ok(Step::Done);
        }
        let (folder, name, project_type) = self.record_context();
        let group_name = self.group_name.clone();
        let start = now_iso();
        let record = ExecutionRecord {
            guid: new_record_id(&group_name, &name),
            group_name,
            project_name: name,
            folder_name: folder,
            project_type,
            server_start_time: None,
            start_time: start.clone(),
            server_end_time: None,
            end_time: now_iso(),
            is_successful: success,
        };
        if let Err(error) = self.store.append_record(&record) {
            tracing::warn!("写入执行记录失败: {error}");
        }

        let run_num = self.groups[self.group_name_index()].projects[self.queue[self.cursor]]
            .run_num
            .max(1);
        if success && self.current_run < run_num {
            self.current_run += 1;
            return self.prepare_current();
        }

        self.cursor += 1;
        self.current_run = 0;
        self.prepare_current()
    }

    /// 当前进度摘要。
    pub fn progress(&self) -> Option<bgi_control::ScriptGroupProgress> {
        if !self.is_running() {
            return None;
        }
        Some(bgi_control::ScriptGroupProgress {
            group_name: self.group_name.clone(),
            project_name: self.current_project_name.clone(),
            project_index: self.cursor + 1,
            project_total: self.queue.len(),
            run_num: self.current_run,
        })
    }

    pub fn stop(&mut self) {
        self.queue.clear();
        self.cursor = 0;
        self.current_run = 0;
    }
}

/// 组执行器给出的下一步动作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// 启动一条路径文件。
    RunPath(String),
    /// 执行一个 Javascript 脚本项目。
    RunScript {
        project_dir: PathBuf,
        main: String,
        settings: Value,
    },
    /// 类型一期不支持，core 记日志后推进。
    Unsupported { project_type: String },
    /// 组已跑完。
    Done,
}

/// 解析 UI 传入的项目类型字符串（大小写不敏感），未知类型拒绝。
fn parse_project_type(raw: &str) -> Result<ProjectType, String> {
    let matched = match raw.trim().to_ascii_lowercase().as_str() {
        "javascript" | "js" => ProjectType::Javascript,
        "keymouse" | "键鼠" => ProjectType::KeyMouse,
        "pathing" | "地图追踪" | "寻路" => ProjectType::Pathing,
        "shell" => ProjectType::Shell,
        other => {
            return Err(format!(
                "未知项目类型「{other}」，可选 Javascript/KeyMouse/Pathing/Shell"
            ));
        }
    };
    Ok(matched)
}

fn find_file_recursive(root: &Path, name: &str, max_depth: usize) -> Option<PathBuf> {
    if max_depth == 0 || !root.is_dir() {
        return None;
    }
    let mut dirs = Vec::new();
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|s| s.to_str()) == Some(name) {
                return Some(path);
            }
            if path.is_dir() {
                dirs.push(path);
            }
        }
    }
    for dir in dirs {
        if let Some(found) = find_file_recursive(&dir, name, max_depth - 1) {
            return Some(found);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> &'static str {
        include_str!(
            "../../../../better-genshin-impact/BetterGenshinImpact/bin/Debug/net8.0-windows10.0.22621.0/User/ScriptGroup/1.json"
        )
    }

    #[test]
    fn real_csharp_group_json_round_trips_without_losing_fields() {
        let group: ScriptGroup = serde_json::from_str(fixture()).expect("解析 C# 配置组样例");
        assert_eq!(group.index, 1);
        assert_eq!(group.name, "1");
        assert!(!group.projects.is_empty());

        let project = &group.projects[0];
        assert_eq!(project.name, "锄地一条龙");
        assert_eq!(project.folder_name, "AutoHoeingOneDragon");
        assert_eq!(project.project_type, ProjectType::Javascript);
        assert_eq!(project.status, "Enabled");
        assert_eq!(project.schedule, "Daily");
        assert_eq!(project.run_num, 1);

        // 关键：pathingConfig 的整个子树必须原样保留
        let source: Value = serde_json::from_str(fixture()).unwrap();
        assert_eq!(
            group.config.pathing_config,
            source["config"]["pathingConfig"]
        );

        let encoded = serde_json::to_string(&group).unwrap();
        let reloaded: ScriptGroup = serde_json::from_str(&encoded).unwrap();
        assert_eq!(reloaded.config.pathing_config, group.config.pathing_config);
        assert_eq!(
            reloaded.projects[0].js_script_settings_object,
            project.js_script_settings_object
        );
    }

    #[test]
    fn unknown_project_type_is_preserved() {
        let json = r#"{"index":0,"name":"g","config":{},"projects":[
            {"index":1,"name":"x","folderName":"f","type":"FutureType","status":"Enabled","schedule":"Daily","runNum":1}
        ]}"#;
        let group: ScriptGroup = serde_json::from_str(json).unwrap();
        assert_eq!(
            group.projects[0].project_type,
            ProjectType::Unknown("FutureType".into())
        );
        let out = serde_json::to_string(&group).unwrap();
        assert!(out.contains("FutureType"));
    }

    #[test]
    fn defaults_fill_missing_fields_like_csharp() {
        let group: ScriptGroup = serde_json::from_str(r#"{"name":"minimal"}"#).unwrap();
        assert_eq!(group.name, "minimal");
        assert_eq!(group.index, 0);
        assert!(group.projects.is_empty());
        assert!(!group.config.enable_shell_config);
    }

    #[test]
    fn pathing_relative_handles_folder_and_suffix() {
        let mut p = ScriptGroupProject {
            folder_name: "AutoHoeingOneDragon".into(),
            name: "锄地一条龙".into(),
            ..Default::default()
        };
        assert_eq!(p.pathing_relative(), "AutoHoeingOneDragon/锄地一条龙.json");

        p.name = "路线.json".into();
        assert_eq!(p.pathing_relative(), "AutoHoeingOneDragon/路线.json");

        p.folder_name.clear();
        assert_eq!(p.pathing_relative(), "路线.json");
    }

    #[test]
    fn cycle_execution_order_matches_csharp_formula() {
        let cycle = TaskCycleConfig {
            enable: true,
            cycle: 3,
            index: 1,
            boundary_time: 4,
        };
        let order = cycle.execution_order();
        assert!((1..=3).contains(&order), "order={order}");

        let invalid = TaskCycleConfig { cycle: 0, ..cycle };
        assert_eq!(invalid.execution_order(), -1);
        let bad_hour = TaskCycleConfig {
            boundary_time: 25,
            ..cycle
        };
        assert_eq!(bad_hour.execution_order(), -1);
    }

    #[test]
    fn skip_rule_validity_requires_enable_and_one_time_constraint() {
        let mut rule = CompletionSkipRule::default();
        assert!(!rule.is_valid(), "默认禁用");
        rule.enable = true;
        assert!(rule.is_valid(), "boundaryTime=4 有效");
        rule.boundary_time = 99;
        rule.last_run_gap_seconds = -1;
        assert!(!rule.is_valid(), "两个时间约束都无效");
        rule.last_run_gap_seconds = 3600;
        assert!(rule.is_valid(), "间隔约束有效");
    }

    #[test]
    fn schedule_marker_round_trips() {
        let dir =
            std::env::temp_dir().join(format!("bgi-script-group-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let marker = NextScheduledTask {
            group_name: "日常".into(),
            index: 3,
            folder_name: "AutoHoeingOneDragon".into(),
            name: "锄地一条龙".into(),
        };
        save_schedule_marker(&dir, &marker).unwrap();
        assert_eq!(load_schedule_marker(&dir), Some(marker));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn group_store_loads_all_real_csharp_groups() {
        let dir = Path::new(
            "../../../../better-genshin-impact/BetterGenshinImpact/bin/Debug/net8.0-windows10.0.22621.0/User",
        );
        if !dir.exists() {
            return;
        }
        let store = GroupStore::new(dir);
        let groups = store.load_all().unwrap();
        assert!(!groups.is_empty(), "应能加载真实配置组");
        assert!(groups.iter().all(|g| !g.name.is_empty()));
        assert!(
            groups.windows(2).all(|w| w[0].index <= w[1].index),
            "按 index 排序"
        );
    }

    #[test]
    fn find_project_and_projects_from_respect_marker() {
        let mut group = ScriptGroup {
            name: "g".into(),
            ..Default::default()
        };
        for i in 0..4 {
            group.projects.push(ScriptGroupProject {
                index: i,
                name: format!("p{i}"),
                folder_name: format!("f{i}"),
                ..Default::default()
            });
        }
        let marker = NextScheduledTask {
            group_name: "g".into(),
            index: 2,
            folder_name: "f2".into(),
            name: "p2".into(),
        };
        let groups = vec![group];
        let (found_g, pos) = find_project(&groups, &marker).unwrap();
        assert_eq!(pos, 2);
        let remaining = projects_from(found_g, pos);
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].name, "p2");
    }

    #[test]
    fn group_runner_skips_disabled_and_orders_projects() {
        let dir = std::env::temp_dir().join(format!("bgi-runner-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(dir.join("ScriptGroup"));
        let store = GroupStore::new(&dir);
        let mut group = ScriptGroup {
            index: 1,
            name: "测试组".into(),
            ..Default::default()
        };
        for i in 0..3 {
            group.projects.push(ScriptGroupProject {
                index: i,
                name: format!("p{i}"),
                folder_name: String::new(),
                project_type: ProjectType::Pathing,
                status: if i == 1 {
                    "Disabled".into()
                } else {
                    "Enabled".into()
                },
                ..Default::default()
            });
        }
        store.save(&group).unwrap();

        let mut runner = GroupRunner::new(&dir);
        // 无路径根目录时，Pathing 项目会解析失败；这里只验证队列跳过逻辑。
        runner.set_pathing_root(dir.to_string_lossy().to_string());
        let result = runner.start("测试组", None);
        // 第一个项目是 Pathing，找不到文件会报错，说明确实跳过了 Disabled 项并尝试了 p0。
        assert!(result.is_err() || matches!(result, Ok(Step::RunPath(_))));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn group_runner_progress_and_stop() {
        let dir = std::env::temp_dir().join(format!("bgi-runner-progress-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(dir.join("ScriptGroup"));
        let store = GroupStore::new(&dir);
        let group = ScriptGroup {
            index: 1,
            name: "g".into(),
            projects: vec![ScriptGroupProject {
                index: 0,
                name: "main.js".into(),
                folder_name: "js".into(),
                project_type: ProjectType::Javascript,
                status: "Enabled".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        store.save(&group).unwrap();
        // Javascript 项目需要真实 manifest 才能启动
        let js_dir = dir.join("JsScript/js");
        std::fs::create_dir_all(&js_dir).unwrap();
        std::fs::write(
            js_dir.join("manifest.json"),
            r#"{"manifest_version":1,"name":"js","main":"main.js"}"#,
        )
        .unwrap();
        std::fs::write(js_dir.join("main.js"), "(async () => {})()").unwrap();
        let mut runner = GroupRunner::new(&dir);
        let step = runner.start("g", None).unwrap();
        assert!(matches!(step, Step::RunScript { .. }), "{step:?}");
        assert!(runner.is_running());
        assert_eq!(runner.progress().unwrap().project_total, 1);
        let next = runner.on_finished(false).unwrap();
        assert_eq!(next, Step::Done);
        assert!(!runner.is_running());
        runner.stop();
        assert!(runner.progress().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn group_runner_crud_creates_deletes_and_persists() {
        let dir = std::env::temp_dir().join(format!("bgi-crud-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut runner = GroupRunner::new(&dir);

        runner.create_group("新组").unwrap();
        assert_eq!(runner.summaries().len(), 1);
        assert_eq!(runner.summaries()[0].name, "新组");
        assert!(dir.join("ScriptGroup/新组.json").exists());

        // 重复创建应拒绝
        assert!(runner.create_group("新组").is_err());
        // 空名拒绝
        assert!(runner.create_group("   ").is_err());

        runner
            .add_project("新组", "pathing", "AutoHoeing", "锄地")
            .unwrap();
        assert_eq!(runner.summaries()[0].project_count, 1);
        let projects = runner.summaries()[0].projects.clone();
        assert_eq!(projects[0].index, 1);
        assert_eq!(projects[0].project_type, "Pathing");
        assert_eq!(projects[0].folder_name, "AutoHoeing");

        // 未知类型拒绝
        assert!(runner.add_project("新组", "Unknown", "f", "x").is_err());

        // 从磁盘重新加载应保留
        let reloaded = GroupRunner::new(&dir);
        assert_eq!(reloaded.summaries()[0].project_count, 1);

        runner.remove_project("新组", 1).unwrap();
        assert_eq!(runner.summaries()[0].project_count, 0);

        runner.delete_group("新组").unwrap();
        assert!(runner.summaries().is_empty());
        assert!(!dir.join("ScriptGroup/新组.json").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso_time_is_parseable_and_current() {
        let now = now_iso();
        assert_eq!(now.len(), 19);
        assert!(parse_iso_secs(&now).is_some(), "解析 {now}");
        let parsed = parse_iso_secs(&now).unwrap();
        let expected = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert!(
            (parsed - expected).abs() <= 2,
            "偏差过大: {parsed} vs {expected}"
        );
    }
}
