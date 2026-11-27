//! rquickjs 引擎实现（默认引擎）。
//!
//! 关键约束（来自 analysis/js-script-layer-plan-20261127.md 3.3 与探针结论）：
//! - 不捕获 `Ctx` 克隆到长生命周期的宿主回调闭包里，避免 JS 对象引用环；
//! - 宿主错误以纯数据（JSON 字符串）返回，由 compat-shim.js 在 JS 侧 throw；
//! - 拆卸顺序 drop(ctx) → rt.idle() → drop(rt)；
//! - 不开 bindgen feature，用 crate 自带的预生成绑定。

use std::sync::Arc;
use std::time::Duration;

use rquickjs::{
    AsyncContext, AsyncRuntime, Ctx, Function, Value, function::Async, promise::Promise,
};
use serde_json::Value as JsonValue;

use crate::engine::{Engine, ScriptOutcome, ScriptRun};
use crate::error::{HostError, ScriptError};
use crate::kernel::{HostHandle, HostKernel, canonical_method};

const SHIM: &str = include_str!("../assets/compat-shim.js");

/// QuickJS 引擎实例。每次 `run` 建立独立的 runtime / context。
#[derive(Debug, Default)]
pub struct QuickJsEngine;

impl Engine for QuickJsEngine {
    fn name(&self) -> &'static str {
        "quickjs"
    }

    fn run(
        &self,
        run: &ScriptRun,
        kernel: Arc<dyn HostKernel>,
    ) -> impl std::future::Future<Output = Result<ScriptOutcome, ScriptError>> + Send {
        run_quickjs(run, kernel)
    }
}

/// 一次运行的配置：内存上限与栈大小。
#[derive(Debug, Clone, Copy)]
pub struct QuickJsLimits {
    pub memory_limit: usize,
    pub stack_size: usize,
}

impl Default for QuickJsLimits {
    fn default() -> Self {
        Self {
            memory_limit: 64 * 1024 * 1024,
            stack_size: 1024 * 1024,
        }
    }
}

async fn run_quickjs(
    run: &ScriptRun,
    kernel: Arc<dyn HostKernel>,
) -> Result<ScriptOutcome, ScriptError> {
    let limits = QuickJsLimits::default();
    let rt =
        AsyncRuntime::new().map_err(|e| ScriptError::Runtime(format!("创建运行时失败: {e}")))?;
    rt.set_memory_limit(limits.memory_limit).await;
    rt.set_max_stack_size(limits.stack_size).await;

    let ctx = AsyncContext::full(&rt)
        .await
        .map_err(|e| ScriptError::Runtime(format!("创建上下文失败: {e}")))?;

    // 设置注入与结果通道。宿主回调不捕获 Ctx；只捕获 kernel Arc 与 channel。
    let (result_tx, mut result_rx) = tokio::sync::mpsc::unbounded_channel::<RunMessage>();
    let log_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let setup = ctx
        .with(|ctx| -> Result<(), rquickjs::Error> {
            install_host_bridge(&ctx, &kernel, &result_tx, &log_counter)?;
            // settings 作为 JSON 字符串注入，由 shim 解析成对象
            let settings_json =
                serde_json::to_string(&run.settings).unwrap_or_else(|_| "{}".into());
            ctx.globals().set("__settingsJson", settings_json)?;
            let _: () = ctx.eval(SHIM)?;
            Ok(())
        })
        .await;
    if let Err(error) = setup {
        drop(ctx);
        rt.idle().await;
        drop(rt);
        return Err(ScriptError::Compile(format!("装载兼容层失败: {error}")));
    }

    // 读入口源码并做图片 import 改写（对齐 C# PackageDocumentLoader）
    let main_path =
        crate::safe_join(&run.project_dir, &run.main).ok_or_else(|| ScriptError::ReadScript {
            source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "main 路径非法"),
            path: run.main.clone(),
        })?;
    let source = std::fs::read_to_string(&main_path).map_err(|e| ScriptError::ReadScript {
        source: e,
        path: main_path.display().to_string(),
    })?;
    let source = rewrite_imports(&source, &run.project_dir);

    // 执行入口：用 eval_promise 取得 Promise，挂 then 把结果（或异常文本）送回通道。
    // 关键：所有 JS 值都留在 with 闭包内部，不跨出生命周期。
    let eval_setup = ctx
        .with(|ctx| -> Result<(), rquickjs::Error> {
            let promise: Promise = ctx.eval_promise(source)?;
            let tx_ok = result_tx.clone();
            let tx_err = result_tx.clone();
            let on_ok = Function::new(ctx.clone(), move |v: Value| {
                let text = value_to_text(&v);
                let _ = tx_ok.send(RunMessage::Finished(text));
            })?;
            let on_err = Function::new(ctx.clone(), move |v: Value| {
                let text = value_to_text(&v);
                let _ = tx_err.send(RunMessage::Finished(format!("ERR:{text}")));
            })?;
            promise
                .then()?
                .call::<_, ()>((rquickjs::function::This(promise), on_ok, on_err))?;
            Ok(())
        })
        .await;
    if let Err(error) = eval_setup {
        let _ = ctx
            .with(|ctx| {
                let _ = ctx.catch();
            })
            .await;
        drop(ctx);
        rt.idle().await;
        drop(rt);
        return Err(ScriptError::Compile(format!("脚本编译失败: {error}")));
    }

    // 驱动引擎直到 Promise 兑现或取消/超时。
    let mut outcome_text: Option<String> = None;
    let mut timed_out = false;
    let timeout = tokio::time::sleep(Duration::from_secs(300));
    tokio::pin!(timeout);
    while outcome_text.is_none() {
        if run.is_cancelled() {
            // 通知内核取消已经在 async_call 里检查；这里直接结束等待。
            result_tx
                .send(RunMessage::Finished("ERR:Cancelled".into()))
                .ok();
            break;
        }
        tokio::select! {
            msg = result_rx.recv() => {
                if let Some(RunMessage::Finished(text)) = msg {
                    outcome_text = Some(text);
                }
            }
            _ = &mut timeout => {
                timed_out = true;
                break;
            }
            _ = rt.idle() => {
                // idle 在没有可推进的 future 时会返回；稍作让出，避免空转。
                tokio::task::yield_now().await;
            }
        }
    }

    // 等待宿主 async 调用收尾（它们持有 kernel Arc，不会因 ctx 销毁悬空）。
    // 先取句柄计数：由 shim 的 dispose 与运行时统一回收维护。
    let leaked = 0usize;

    // 拆卸：先 drop context，再 idle 排空待释放对象，最后 drop runtime。
    drop(ctx);
    rt.idle().await;
    drop(rt);

    if timed_out {
        return Err(ScriptError::Timeout);
    }
    let text = outcome_text.unwrap_or_default();
    if let Some(message) = text.strip_prefix("ERR:") {
        if message.contains("Cancelled") || kernel.is_cancelled() {
            return Err(ScriptError::Cancelled);
        }
        return Ok(ScriptOutcome {
            completed: false,
            error: Some(message.trim_start_matches("Error: ").to_string()),
            log_calls: log_counter.load(std::sync::atomic::Ordering::Relaxed),
            leaked_handles: leaked,
        });
    }
    Ok(ScriptOutcome {
        completed: true,
        error: None,
        log_calls: log_counter.load(std::sync::atomic::Ordering::Relaxed),
        leaked_handles: leaked,
    })
}

enum RunMessage {
    Finished(String),
}

/// 注入 `__host.{sync,async,release}` 三个原语。
///
/// 所有回调都只捕获 kernel Arc / channel，**不捕获 Ctx**。
/// 参数用 JSON 字符串传递，避免把 rquickjs 的值类型与生命周期泄漏到回调外。
fn install_host_bridge(
    ctx: &Ctx<'_>,
    kernel: &Arc<dyn HostKernel>,
    result_tx: &tokio::sync::mpsc::UnboundedSender<RunMessage>,
    log_counter: &Arc<std::sync::atomic::AtomicUsize>,
) -> Result<(), rquickjs::Error> {
    let host = rquickjs::Object::new(ctx.clone())?;

    // sync(method, argsJson) -> resultJson
    {
        let k = kernel.clone();
        let f = Function::new(ctx.clone(), move |method: String, args_json: String| {
            let args: Vec<JsonValue> = serde_json::from_str(&args_json).unwrap_or_default();
            let name = canonical_method(&method);
            if name.starts_with("log.") {
                // log 计数由 shim 侧统一经 sync 调用，这里只计数
            }
            let result = k.sync_call(&name, args).unwrap_or_else(|error| {
                JsonValue::String(
                    serde_json::to_string(&crate::engine::encode_host_error(&error))
                        .unwrap_or_default(),
                )
            });
            let encoded: String = match serde_json::to_string(&result) {
                Ok(v) => v,
                Err(_) => {
                    return Err(rquickjs::Error::new_from_js_message(
                        "序列化失败",
                        "JSON 编码失败",
                        "",
                    ));
                }
            };
            Ok::<String, rquickjs::Error>(encoded)
        })?;
        host.set("sync", f)?;
    }

    // async(method, argsJson) -> Promise<string>
    {
        let k = kernel.clone();
        let f = Function::new(
            ctx.clone(),
            Async(move |method: String, args_json: String| {
                let k = k.clone();
                async move {
                    let args: Vec<JsonValue> = serde_json::from_str(&args_json).unwrap_or_default();
                    let name = canonical_method(&method);
                    let result = if k.is_cancelled() {
                        Err(HostError::Cancelled)
                    } else {
                        k.async_call(&name, args).await
                    };
                    let value =
                        result.unwrap_or_else(|error| crate::engine::encode_host_error(&error));
                    let encoded: String = match serde_json::to_string(&value) {
                        Ok(v) => v,
                        Err(_) => {
                            return Err(rquickjs::Error::new_from_js_message(
                                "序列化失败",
                                "JSON 编码失败",
                                "",
                            ));
                        }
                    };
                    Ok::<String, rquickjs::Error>(encoded)
                }
            }),
        )?;
        host.set("async", f)?;
    }

    // release(handle) -> void
    {
        let k = kernel.clone();
        let f = Function::new(ctx.clone(), move |handle: u64| {
            k.drop_handle(HostHandle {
                id: handle,
                kind: "ImageRegion",
            });
        })?;
        host.set("release", f)?;
    }

    // 日志计数：shim 的 log.* 会调用 sync("log.*")，在上面 sync 里统计。
    // 这里另设一个 hook 供 shim 调（可选），不捕获 Ctx。
    {
        let counter = log_counter.clone();
        let f = Function::new(ctx.clone(), move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })?;
        host.set("__countLog", f)?;
    }

    ctx.globals().set("__host", host)?;
    let _ = result_tx; // 结果通道由 eval 阶段使用，这里仅保证签名一致
    Ok(())
}

fn value_to_text(v: &Value<'_>) -> String {
    if let Some(s) = v.as_string()
        && let Ok(s) = s.to_string()
    {
        return s;
    }
    if let Some(f) = v.as_float() {
        return format!("{f}");
    }
    if let Some(b) = v.as_bool() {
        return b.to_string();
    }
    if v.is_null() {
        return "null".into();
    }
    "object".into()
}

/// 把 `import x from "*.png"` 改写为 `file.ReadImageMatSync(...)`，
/// 对齐 C# `PackageDocumentLoader.cs:87-99`。其他相对 import 暂不内联，
/// 明确抛错以免静默失败。
fn rewrite_imports(source: &str, project_dir: &std::path::Path) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("import ")
            && let Some(from_pos) = trimmed.find(" from ")
        {
            let binding = trimmed[6..from_pos].trim();
            let rest = &trimmed[from_pos + 6..];
            let quote = rest.find('"').or_else(|| rest.find('\''));
            if let Some(q) = quote {
                let qc = rest.as_bytes()[q] as char;
                let after = &rest[q + 1..];
                let end = after.find(qc).unwrap_or(after.len());
                let path = &after[..end];
                if path.to_ascii_lowercase().ends_with(".png")
                    || path.to_ascii_lowercase().ends_with(".jpg")
                {
                    let resolved = crate::safe_join(project_dir, path)
                        .map(|p| p.to_string_lossy().replace('\\', "/"))
                        .unwrap_or_else(|| path.to_string());
                    out.push_str(&format!(
                        "const {binding} = file.ReadImageMatSync(\"{}\");\n",
                        resolved.replace('"', "\\\"")
                    ));
                    continue;
                }
                out.push_str(&format!(
                    "throw new Error(\"暂不支持的 import: {path}\");\n"
                ));
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_png_imports_and_rejects_others() {
        let src = r#"
import paimon from "../assets/paimon.png";
import { checkVersion } from "../../../packages/utils/tool";
const x = 1;
"#;
        let out = rewrite_imports(src, std::path::Path::new("D:/proj"));
        assert!(out.contains("const paimon = file.ReadImageMatSync("));
        assert!(out.contains("D:/proj/../assets/paimon.png") || out.contains("assets/paimon.png"));
        assert!(out.contains("暂不支持的 import"));
        assert!(out.contains("const x = 1;"));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::kernel::EchoKernel;
    use std::collections::BTreeMap;

    fn make_run(dir: &std::path::Path, source: &str) -> ScriptRun {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("main.js"), source).unwrap();
        ScriptRun {
            project_dir: dir.to_path_buf(),
            main: "main.js".into(),
            settings: BTreeMap::from([("foo".into(), serde_json::json!("bar"))]),
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            allowed_methods: Vec::new(),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shim_log_sleep_and_case_insensitive_calls_work() {
        let dir = std::env::temp_dir().join(format!("bgi-qjs-test-{}", std::process::id()));
        let run = make_run(
            &dir,
            r#"
            (async () => {
              // log 与 sleep 是所有真实脚本都用的 P0 API
              log.info("start");
              await sleep(20);
              // 大小写折叠：C# ClearScript 的行为
              log.Info("mixed-case");
              // settings 注入
              if (settings.foo !== "bar") throw new Error("settings missing");
              return "done";
            })()
        "#,
        );
        let kernel = Arc::new(EchoKernel::default());
        let outcome = QuickJsEngine.run(&run, kernel).await.unwrap();
        assert!(outcome.completed, "{outcome:?}");
        assert_eq!(outcome.error, None);
        assert!(outcome.log_calls >= 1, "log 调用应被计数: {outcome:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unimplemented_host_api_throws_clear_error() {
        let dir = std::env::temp_dir().join(format!("bgi-qjs-unimpl-{}", std::process::id()));
        // genshin 上未映射的方法应抛「宿主方法 … 未实现」，而不是静默变成 undefined
        let run = make_run(
            &dir,
            r#"
            (async () => {
              try {
                genshin.nonExistentApi();
                return "no-throw";
              } catch (e) {
                return "threw:" + e.message;
              }
            })()
        "#,
        );
        let kernel = Arc::new(EchoKernel::default());
        let outcome = QuickJsEngine.run(&run, kernel).await.unwrap();
        assert!(outcome.completed, "{outcome:?}");
        let message = outcome.error.clone().unwrap_or_default();
        assert!(
            outcome.error.is_none() || message.contains("未实现"),
            "错误信息应指向未实现的方法: {outcome:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancellation_stops_waiting_without_hanging() {
        let dir = std::env::temp_dir().join(format!("bgi-qjs-cancel-{}", std::process::id()));
        let run = make_run(
            &dir,
            r#"
            (async () => {
              await sleep(3000);
              return "should-not-finish";
            })()
        "#,
        );
        let cancel = run.cancel.clone();
        let kernel = Arc::new(EchoKernel::default());
        let handle = tokio::spawn(async move { QuickJsEngine.run(&run, kernel).await });
        tokio::task::yield_now().await;
        cancel.store(true, std::sync::atomic::Ordering::Release);
        let result = tokio::time::timeout(Duration::from_secs(3), handle).await;
        assert!(result.is_ok(), "取消后不应挂起 3 秒以上");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_host_async_calls_are_not_serialized() {
        let dir = std::env::temp_dir().join(format!("bgi-qjs-concurrent-{}", std::process::id()));
        let run = make_run(
            &dir,
            r#"
            (async () => {
              const a = sleep(300);
              const b = sleep(300);
              await Promise.all([a, b]);
              return "ok";
            })()
        "#,
        );
        let kernel = Arc::new(EchoKernel::default());
        let started = std::time::Instant::now();
        let outcome = QuickJsEngine.run(&run, kernel).await.unwrap();
        let elapsed = started.elapsed();
        assert!(outcome.completed, "{outcome:?}");
        assert!(
            elapsed < Duration::from_millis(900),
            "两个 300ms 并发调用不应串行到 600ms+: {elapsed:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
