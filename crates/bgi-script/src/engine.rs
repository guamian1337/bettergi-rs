//! 引擎无关的执行接口与请求/结果类型。
//!
//! 两个引擎实现（rquickjs / v8-builtin）都遵守同一份契约：
//! 1. 先装载 `compat-shim.js`，再执行用户脚本；
//! 2. 宿主 API 只经 `__host.{sync,async,release}` 三个通道；
//! 3. 宿主错误以**纯数据**返回（`{"$hostError": kind, "message": ...}`），
//!    由 shim 在 JS 侧 throw —— 绝不在 Rust 侧构造 JS 异常对象
//!    （见 analysis/js-script-layer-plan-20261127.md 3.3 的断言根因）；
//! 4. 拆卸顺序必须正确，否则 QuickJS / V8 会在释放时中止进程。

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;

use crate::error::ScriptError;

/// 一次脚本运行的输入。
#[derive(Debug, Clone)]
pub struct ScriptRun {
    /// 脚本项目根目录（含 manifest.json）。
    pub project_dir: PathBuf,
    /// 入口文件相对路径（manifest.main）。
    pub main: String,
    /// 宿主注入的 `settings` 对象（已合并默认值与用户保存值）。
    pub settings: BTreeMap<String, Value>,
    /// 取消标志。
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// 支持的宿主方法白名单（未映射的 API 抛未实现错误）。
    pub allowed_methods: Vec<String>,
}

impl ScriptRun {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// 一次脚本运行的输出。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptOutcome {
    /// 脚本正常结束（async IIFE 的 Promise 兑现）。
    pub completed: bool,
    /// 未捕获异常 / 宿主错误文本。
    pub error: Option<String>,
    /// 运行期间调用 `log.*` 的条数（指标）。
    pub log_calls: usize,
    /// 运行结束时仍未释放的宿主句柄数（正常应为 0）。
    pub leaked_handles: usize,
}

impl ScriptOutcome {
    pub fn ok() -> Self {
        Self {
            completed: true,
            error: None,
            log_calls: 0,
            leaked_handles: 0,
        }
    }
}

/// 引擎接口。实现必须可以被多次 `run`，每次运行有独立的 context 与句柄作用域。
pub trait Engine: Send {
    /// 引擎名，用于日志与一致性测试。
    fn name(&self) -> &'static str;

    /// 执行一次脚本，返回运行结果。
    ///
    /// 这个方法在 tokio 异步上下文中调用；引擎内部要么用 async runtime
    /// （rquickjs），要么把 isolate 放到专用线程并用通道驱动（v8）。
    fn run(
        &self,
        run: &ScriptRun,
        kernel: std::sync::Arc<dyn crate::kernel::HostKernel>,
    ) -> impl std::future::Future<Output = Result<ScriptOutcome, ScriptError>> + Send;
}

/// 把宿主错误编码成纯数据（供 shim 在 JS 侧解码后 throw）。
pub fn encode_host_error(error: &crate::error::HostError) -> Value {
    let (kind, message) = match error {
        crate::error::HostError::Unimplemented(m) => ("Unimplemented", m.clone()),
        crate::error::HostError::InvalidArgs(m, d) => ("InvalidArgs", format!("{m}: {d}")),
        crate::error::HostError::Cancelled => ("Cancelled", "任务已取消".into()),
        crate::error::HostError::Runtime(m, d) => ("Runtime", format!("{m}: {d}")),
        crate::error::HostError::Io(d) => ("Io", d.clone()),
        crate::error::HostError::Timeout => ("Timeout", "脚本执行超时".into()),
    };
    serde_json::json!({ "$hostError": kind, "message": message })
}

/// 从 JS 侧拿到的值中解出宿主错误。仅识别 `$hostError` 标记，避免误判用户数据。
pub fn decode_host_error(value: &Value) -> Option<crate::error::HostError> {
    let marker = value.get("$hostError")?.as_str()?;
    let msg = value
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Some(match marker {
        "Unimplemented" => crate::error::HostError::Unimplemented(msg),
        "InvalidArgs" => crate::error::HostError::InvalidArgs(msg, String::new()),
        "Cancelled" => crate::error::HostError::Cancelled,
        "Runtime" => crate::error::HostError::Runtime(msg, String::new()),
        "Io" => crate::error::HostError::Io(msg),
        "Timeout" => crate::error::HostError::Timeout,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_error_round_trips_through_plain_data() {
        for error in [
            crate::error::HostError::Unimplemented("genshin.foo".into()),
            crate::error::HostError::Cancelled,
            crate::error::HostError::Io("denied".into()),
            crate::error::HostError::Timeout,
        ] {
            let encoded = encode_host_error(&error);
            let decoded = decode_host_error(&encoded).unwrap();
            match (&error, &decoded) {
                (
                    crate::error::HostError::Unimplemented(a),
                    crate::error::HostError::Unimplemented(b),
                ) => assert_eq!(a, b),
                (crate::error::HostError::Cancelled, crate::error::HostError::Cancelled) => {}
                (crate::error::HostError::Io(a), crate::error::HostError::Io(b)) => {
                    assert_eq!(a, b)
                }
                (crate::error::HostError::Timeout, crate::error::HostError::Timeout) => {}
                _ => panic!("编码/解码不一致: {error:?} -> {decoded:?}"),
            }
        }
    }

    #[test]
    fn plain_user_data_is_not_mistaken_for_host_error() {
        assert!(decode_host_error(&serde_json::json!({"name": "x"})).is_none());
        assert!(decode_host_error(&serde_json::json!({"$hostError": 1})).is_none());
        assert!(decode_host_error(&serde_json::json!({"$hostError": "Unknown"})).is_none());
    }

    #[test]
    fn script_run_reports_cancellation() {
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let run = ScriptRun {
            project_dir: PathBuf::new(),
            main: "main.js".into(),
            settings: BTreeMap::new(),
            cancel: cancel.clone(),
            allowed_methods: Vec::new(),
        };
        assert!(!run.is_cancelled());
        cancel.store(true, std::sync::atomic::Ordering::Release);
        assert!(run.is_cancelled());
    }
}
