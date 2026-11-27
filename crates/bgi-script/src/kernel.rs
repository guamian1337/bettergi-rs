//! `HostKernel` 三原语：脚本访问宿主能力的唯一通道。
//!
//! 设计约束：
//! - 引擎无关：参数与返回值都是 `serde_json::Value`，两个引擎实现共用同一份 kernel。
//! - 规范名：`method` 用点分小驼峰（`log.info`、`genshin.returnMainUi`）。
//!   C# ClearScript 的成员绑定大小写不敏感，脚本里拼写混用；`canonical_method`
//!   负责在进入 kernel 前归一化，kernel 自身只认规范名。
//! - 资源用句柄：截图区域、识别对象等由 kernel 持有，脚本侧只拿到 `{$handle, $kind}`，
//!   通过 `drop_handle` 归还，脚本漏写 `dispose()` 时由运行结束时的统一回收兜底。

use std::future::Future;
use std::pin::Pin;

use serde_json::{Value, json};

use crate::error::HostError;

pub type HostValue = Value;

/// `async_call` 返回的 future。要求 `Send`，以便在 tokio 线程池上推进而不阻塞引擎线程。
pub type HostFuture = Pin<Box<dyn Future<Output = Result<HostValue, HostError>> + Send>>;

/// 宿主资源句柄。`kind` 对应 C# 侧类型名（`ImageRegion`、`RecognitionObject`、`Mat`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct HostHandle {
    pub id: u64,
    #[serde(rename = "$kind")]
    pub kind: &'static str,
}

impl HostHandle {
    /// 脚本侧看到的表示：`{"$handle": id, "$kind": kind}`。
    pub fn to_value(self) -> HostValue {
        json!({ "$handle": self.id, "$kind": self.kind })
    }

    pub fn from_value(value: &HostValue) -> Option<Self> {
        Some(Self {
            id: value.get("$handle")?.as_u64()?,
            kind: match value.get("$kind")?.as_str()? {
                "ImageRegion" => "ImageRegion",
                "RecognitionObject" => "RecognitionObject",
                "Mat" => "Mat",
                _ => return None,
            },
        })
    }
}

/// 三原语接口。由 `bgi-app` 实现，`bgi-script` 不依赖任何游戏侧类型。
pub trait HostKernel: Send + Sync + 'static {
    /// sync：立即返回。键鼠、截图、识别、文件、日志等。
    fn sync_call(&self, method: &str, args: Vec<HostValue>) -> Result<HostValue, HostError>;

    /// async：引擎在 await 点挂起，future 完成后继续。sleep、传送、寻路、宏回放等。
    fn async_call(&self, method: &str, args: Vec<HostValue>) -> HostFuture;

    /// drop：释放脚本持有的宿主资源。
    fn drop_handle(&self, handle: HostHandle);

    /// 取消标志。kernel 在 `async_call` 开始与等待期间检查。
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// 把大小写不敏感的调用名归一化为规范名。
///
/// 规则：按 `.` 切段，对象段保持原样（已是小驼峰），成员段首字母小写。
/// `genshin.ReturnMainUi` → `genshin.returnMainUi`；`file.ReadTextSync` → `file.readTextSync`；
/// `dispatcher.AddTrigger` → `dispatcher.addTrigger`。
///
/// C# 里注册的对象名本身就是小驼峰（`genshin`/`file`/`log`），脚本侧不会出现 `Genshin`，
/// 但为了稳妥，对象段也做首字母小写。
pub fn canonical_method(method: &str) -> String {
    method
        .split('.')
        .map(|segment| {
            let mut chars = segment.chars();
            match chars.next() {
                Some(first) => first.to_lowercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// 一次脚本运行里由脚本持有的句柄集合。脚本结束/取消/异常时统一 `drop_handle`，
/// 脚本显式 `dispose()` 时即时移除。
#[derive(Debug, Default)]
pub struct HandleScope {
    live: std::collections::HashSet<HostHandle>,
}

impl HandleScope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, handle: HostHandle) -> HostValue {
        self.live.insert(handle);
        handle.to_value()
    }

    /// 显式释放。重复 dispose 返回 false 但不报错，对齐 C# `Dispose` 可重复调用的行为。
    pub fn release(&mut self, handle: HostHandle) -> bool {
        self.live.remove(&handle)
    }

    pub fn release_value(&mut self, value: &HostValue) -> Option<bool> {
        HostHandle::from_value(value).map(|h| self.release(h))
    }

    pub fn is_live(&self, handle: HostHandle) -> bool {
        self.live.contains(&handle)
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// 回收全部未释放句柄，返回回收数量。
    pub fn drain(&mut self, kernel: &dyn HostKernel) -> usize {
        let count = self.live.len();
        for handle in self.live.drain() {
            kernel.drop_handle(handle);
        }
        count
    }
}

/// 测试与探针用的最小 kernel：记录调用，`sync` 原样回显，`async` 立即完成。
#[derive(Debug, Default)]
pub struct EchoKernel {
    pub cancelled: std::sync::atomic::AtomicBool,
    pub dropped: std::sync::Mutex<Vec<HostHandle>>,
}

impl HostKernel for EchoKernel {
    fn sync_call(&self, method: &str, args: Vec<HostValue>) -> Result<HostValue, HostError> {
        Ok(json!({ "method": method, "args": args }))
    }

    fn async_call(&self, method: &str, args: Vec<HostValue>) -> HostFuture {
        let value = json!({ "method": method, "args": args });
        Box::pin(std::future::ready(Ok(value)))
    }

    fn drop_handle(&self, handle: HostHandle) {
        self.dropped.lock().unwrap().push(handle);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_method_lowercases_member_segments() {
        assert_eq!(
            canonical_method("genshin.ReturnMainUi"),
            "genshin.returnMainUi"
        );
        assert_eq!(canonical_method("file.ReadTextSync"), "file.readTextSync");
        assert_eq!(
            canonical_method("dispatcher.AddTrigger"),
            "dispatcher.addTrigger"
        );
        assert_eq!(canonical_method("genshin.tp"), "genshin.tp");
        assert_eq!(canonical_method("log.info"), "log.info");
        assert_eq!(canonical_method("Genshin.Tp"), "genshin.tp");
        assert_eq!(canonical_method("sleep"), "sleep");
        assert_eq!(canonical_method(""), "");
    }

    #[test]
    fn handle_round_trips_through_json_and_rejects_unknown_kinds() {
        let handle = HostHandle {
            id: 7,
            kind: "ImageRegion",
        };
        let value = handle.to_value();
        assert_eq!(HostHandle::from_value(&value), Some(handle));
        assert_eq!(
            HostHandle::from_value(&json!({ "$handle": 1, "$kind": "Nope" })),
            None
        );
        assert_eq!(HostHandle::from_value(&json!({ "id": 1 })), None);
    }

    #[test]
    fn handle_scope_releases_once_and_drains_rest() {
        let kernel = EchoKernel::default();
        let mut scope = HandleScope::new();
        let a = HostHandle {
            id: 1,
            kind: "ImageRegion",
        };
        let b = HostHandle { id: 2, kind: "Mat" };
        assert_eq!(scope.register(a), a.to_value());
        scope.register(b);
        assert_eq!(scope.live_count(), 2);
        assert!(scope.release(a));
        assert!(!scope.release(a), "重复 dispose 不报错但返回 false");
        assert!(!scope.is_live(a));
        assert_eq!(scope.drain(&kernel), 1, "剩余句柄统一回收");
        assert_eq!(scope.live_count(), 0);
        let dropped = kernel.dropped.lock().unwrap().clone();
        assert_eq!(dropped, vec![b]);
    }

    #[test]
    fn echo_kernel_reports_cancellation() {
        let kernel = EchoKernel::default();
        assert!(!kernel.is_cancelled());
        kernel
            .cancelled
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(kernel.is_cancelled());
    }
}
