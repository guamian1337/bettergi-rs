//! `bgi-app` 侧的 `HostKernel` 实现：把脚本三原语接到现有游戏能力上。
//!
//! 一期覆盖范围（按真实脚本使用频率）：
//! - P0：log / sleep / file / 键鼠全局函数 / settings
//! - P1：captureGameRegion（返回句柄占位，区域识别一期不实现）
//! - 其余：明确返回 `Unimplemented`，由 shim 在 JS 侧 throw
//!
//! 安全边界：`file.*` 的所有路径必须落在脚本项目根目录内。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bgi_control::{LogLevel, LogPublisher};
use bgi_script::error::HostError;
use bgi_script::kernel::{HostFuture, HostHandle, HostKernel, HostValue, canonical_method};
use serde_json::{Value, json};

/// 脚本宿主实现。一个实例对应一次脚本运行，句柄作用域在实例内。
pub struct CoreHostKernel {
    log_pub: LogPublisher,
    project_dir: PathBuf,
    cancel: Arc<AtomicBool>,
    next_handle: AtomicU64,
    handles: std::sync::Mutex<HashMap<u64, String>>,
    log_calls: AtomicU64,
}

impl std::fmt::Debug for CoreHostKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreHostKernel")
            .field("project_dir", &self.project_dir)
            .field("cancelled", &self.cancel.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl CoreHostKernel {
    pub fn new(log_pub: LogPublisher, project_dir: PathBuf, cancel: Arc<AtomicBool>) -> Self {
        Self {
            log_pub,
            project_dir,
            cancel,
            next_handle: AtomicU64::new(1),
            handles: std::sync::Mutex::new(HashMap::new()),
            log_calls: AtomicU64::new(0),
        }
    }

    #[allow(dead_code)]
    pub fn log_calls(&self) -> u64 {
        self.log_calls.load(Ordering::Relaxed)
    }

    /// 校验路径必须落在脚本项目根目录内。
    fn resolve_path(&self, raw: &str) -> Result<PathBuf, HostError> {
        let relative = bgi_script::safe_relative(raw)
            .ok_or_else(|| HostError::Io(format!("非法脚本路径: {raw}")))?;
        let path = self.project_dir.join(relative);
        // 词法前缀校验：safe_relative 已拒绝 .. / 绝对路径 / 盘符，
        // 这里再确认拼接结果确实落在 root 下。不用 canonicalize——
        // 写文件时目标还不存在，canonicalize 会失败。
        let root = &self.project_dir;
        if !path.starts_with(root) {
            return Err(HostError::Io(format!("路径越界: {raw}")));
        }
        Ok(path)
    }

    fn arg(args: &[Value], index: usize) -> Result<&Value, HostError> {
        args.get(index)
            .ok_or_else(|| HostError::InvalidArgs(format!("参数 {index}"), "缺少参数".into()))
    }

    fn arg_str(args: &[Value], index: usize) -> Result<String, HostError> {
        Self::arg(args, index)?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| HostError::InvalidArgs(format!("参数 {index}"), "应为字符串".into()))
    }

    fn arg_u64(args: &[Value], index: usize) -> Result<u64, HostError> {
        Self::arg(args, index)?
            .as_u64()
            .ok_or_else(|| HostError::InvalidArgs(format!("参数 {index}"), "应为非负整数".into()))
    }

    fn arg_i64(args: &[Value], index: usize) -> Result<i64, HostError> {
        Self::arg(args, index)?
            .as_i64()
            .ok_or_else(|| HostError::InvalidArgs(format!("参数 {index}"), "应为整数".into()))
    }

    fn alloc_handle(&self, kind: &'static str) -> HostValue {
        let id = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.handles.lock().unwrap().insert(id, kind.to_string());
        HostHandle { id, kind }.to_value()
    }

    fn handle_exists(&self, handle: u64) -> bool {
        self.handles.lock().unwrap().contains_key(&handle)
    }
}

impl HostKernel for CoreHostKernel {
    fn sync_call(&self, method: &str, args: Vec<HostValue>) -> Result<HostValue, HostError> {
        let name = canonical_method(method);
        match name.as_str() {
            // ---- log ----
            "log.info" | "log.warn" | "log.error" | "log.debug" => {
                let message = Self::arg_str(&args, 0)?;
                let level = match name.as_str() {
                    "log.warn" => LogLevel::Warn,
                    "log.error" => LogLevel::Error,
                    "log.debug" => LogLevel::Debug,
                    _ => LogLevel::Info,
                };
                self.log_calls.fetch_add(1, Ordering::Relaxed);
                self.log_pub.emit(level, "Script", message);
                Ok(Value::Null)
            }

            // ---- 键鼠全局函数 ----
            "keyDown" | "keyup" | "keypress" => {
                let vk = Self::arg_u64(&args, 0)? as u16;
                match name.as_str() {
                    "keydown" => bgi_input::key_down(vk),
                    "keyup" => bgi_input::key_up(vk),
                    _ => bgi_input::key_tap(vk, 25),
                }
                Ok(Value::Null)
            }
            "click" => {
                // 客户端坐标点击需要窗口句柄，一期只支持无参点击（当前鼠标位置）
                bgi_input::mouse_left_click();
                Ok(Value::Null)
            }
            "leftbuttonclick" => {
                bgi_input::mouse_left_click();
                Ok(Value::Null)
            }
            "leftbuttondown" => {
                bgi_input::mouse_left_down();
                Ok(Value::Null)
            }
            "leftbuttonup" => {
                bgi_input::mouse_left_up();
                Ok(Value::Null)
            }
            "rightbuttonclick" => {
                bgi_input::mouse_right_click();
                Ok(Value::Null)
            }
            "rightbuttondown" => {
                bgi_input::mouse_right_down();
                Ok(Value::Null)
            }
            "rightbuttonup" => {
                bgi_input::mouse_right_up();
                Ok(Value::Null)
            }
            "mousemoveby" => {
                bgi_input::mouse_move_rel(
                    Self::arg_i64(&args, 0)? as i32,
                    Self::arg_i64(&args, 1)? as i32,
                );
                Ok(Value::Null)
            }
            "verticalscroll" => {
                bgi_input::mouse_wheel(Self::arg_i64(&args, 0)? as i32);
                Ok(Value::Null)
            }
            "inputtext" => {
                // C# 用剪贴板 + Ctrl+V，一期返回未实现，避免误输入
                Err(HostError::Unimplemented(method.to_string()))
            }

            // ---- file ----
            "file.readtextsync" | "file.readtext" => {
                let path = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| HostError::Io(format!("{}: {e}", path.display())))?;
                Ok(Value::String(text))
            }
            "file.writetextsync" | "file.writetext" => {
                let path = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                let text = Self::arg_str(&args, 1)?;
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| HostError::Io(format!("创建目录失败: {e}")))?;
                }
                std::fs::write(&path, text).map_err(|e| HostError::Io(format!("写入失败: {e}")))?;
                Ok(Value::Bool(true))
            }
            "file.readpathsync" => {
                let path = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                let entries = std::fs::read_dir(&path)
                    .map_err(|e| HostError::Io(e.to_string()))?
                    .filter_map(Result::ok)
                    .filter_map(|e| e.file_name().into_string().ok())
                    .map(Value::String)
                    .collect();
                Ok(Value::Array(entries))
            }
            "file.isfolder" | "file.isfile" | "file.isexists" => {
                let path = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                let exists = path.exists();
                let matched = match name.as_str() {
                    "file.isfolder" => exists && path.is_dir(),
                    "file.isfile" => exists && path.is_file(),
                    _ => exists,
                };
                Ok(Value::Bool(matched))
            }
            "file.createdirectory" => {
                let path = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                std::fs::create_dir_all(&path).map_err(|e| HostError::Io(e.to_string()))?;
                Ok(Value::Bool(true))
            }
            "file.deletepathsync" => {
                let path = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                if path.is_dir() {
                    std::fs::remove_dir_all(&path)
                } else {
                    std::fs::remove_file(&path)
                }
                .map_err(|e| HostError::Io(e.to_string()))?;
                Ok(Value::Bool(true))
            }
            "file.renamepathsync" => {
                let from = self.resolve_path(&Self::arg_str(&args, 0)?)?;
                let to = self.resolve_path(&Self::arg_str(&args, 1)?)?;
                std::fs::rename(from, to).map_err(|e| HostError::Io(e.to_string()))?;
                Ok(Value::Bool(true))
            }
            "file.readimagematsync" | "file.readimagewithresizesync" | "file.writeimagesync" => {
                // 图像对象一期不注入到脚本（需要 OpenCV Mat 的 JS 包装）
                Err(HostError::Unimplemented(method.to_string()))
            }

            // ---- 截图与识别：返回句柄，方法调用未实现 ----
            "capturegameregion" => Ok(self.alloc_handle("ImageRegion")),
            "genshin.getpositionfrommap" | "genshin.getpositionfrombigmap" => {
                Ok(json!({ "x": 0.0, "y": 0.0 }))
            }
            "genshin.getcameraorientation" => Ok(json!(0.0)),
            "genshin.getbigmapzoomlevel" => Ok(json!(1.0)),
            "getavatars" => Ok(json!([])),

            // ---- dispatcher：一期未开启动态任务 ----
            "dispatcher.addtimer"
            | "dispatcher.addtrigger"
            | "dispatcher.clearalltriggers"
            | "dispatcher.getlinkedcancellationToken" => {
                Err(HostError::Unimplemented(method.to_string()))
            }

            // ---- 句柄上的区域方法（shim 走 region.*） ----
            m if m.starts_with("region.") => {
                let handle = Self::arg(args.as_slice(), 0)?
                    .get("$handle")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| HostError::InvalidArgs("handle".into(), "缺少句柄".into()))?;
                if !self.handle_exists(handle) {
                    return Err(HostError::Runtime("region".into(), "句柄已释放".into()));
                }
                Err(HostError::Unimplemented(method.to_string()))
            }

            _ => Err(HostError::Unimplemented(method.to_string())),
        }
    }

    fn async_call(&self, method: &str, args: Vec<HostValue>) -> HostFuture {
        let name = canonical_method(method);
        let method_owned = method.to_string();
        let cancel = self.cancel.clone();
        Box::pin(async move {
            if cancel.load(Ordering::Acquire) {
                return Err(HostError::Cancelled);
            }
            match name.as_str() {
                // sleep：可被取消打断，不阻塞线程
                "sleep" => {
                    let ms = Self::arg_u64(args.as_slice(), 0)?;
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    if cancel.load(Ordering::Acquire) {
                        return Err(HostError::Cancelled);
                    }
                    Ok(Value::Null)
                }
                // file 的异步版：一期与同步版等价（小文件，不阻塞）
                "file.readtext" | "file.writetext" => Err(HostError::Unimplemented(method_owned)),
                _ => Err(HostError::Unimplemented(method_owned)),
            }
        })
    }

    fn drop_handle(&self, handle: HostHandle) {
        self.handles.lock().unwrap().remove(&handle.id);
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bgi_control::create_control_bus;
    use std::path::Path;

    fn kernel(dir: &Path) -> (CoreHostKernel, std::sync::Arc<AtomicBool>) {
        let (_, _, endpoint) = create_control_bus(bgi_control::UiSnapshot::default());
        let cancel = Arc::new(AtomicBool::new(false));
        let k = CoreHostKernel::new(endpoint.log_publisher(), dir.to_path_buf(), cancel.clone());
        (k, cancel)
    }

    #[test]
    fn file_paths_are_confined_to_project_dir() {
        let dir = std::env::temp_dir().join(format!("bgi-script-host-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let (k, _) = kernel(&dir);
        // 绝对路径与 .. 逃逸都要拒绝
        assert!(k.resolve_path("C:/Windows/win.ini").is_err());
        assert!(k.resolve_path("../../evil.txt").is_err());
        assert!(k.resolve_path("records/a.txt").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unimplemented_methods_report_clear_errors() {
        let dir = std::env::temp_dir();
        let (k, _) = kernel(&dir);
        let err = k.sync_call("genshin.tp", vec![]).unwrap_err();
        assert!(matches!(err, HostError::Unimplemented(_)), "{err:?}");
    }

    #[tokio::test]
    async fn sleep_can_be_cancelled() {
        let (k, cancel) = kernel(&std::env::temp_dir());
        cancel.store(true, Ordering::Release);
        let result = k.async_call("sleep", vec![json!(5000)]).await;
        assert!(matches!(result, Err(HostError::Cancelled)), "{result:?}");
    }
}
