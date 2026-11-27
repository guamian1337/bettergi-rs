//! 脚本层错误类型。引擎无关，两个引擎实现都映射到这里。

use thiserror::Error;

/// 宿主方法调用失败。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostError {
    /// 宿主方法尚未实现。`compat-shim` 对未映射 API 统一报这个，绝不静默返回 undefined。
    #[error("宿主方法 {0} 未实现")]
    Unimplemented(String),
    #[error("宿主方法 {0} 参数无效: {1}")]
    InvalidArgs(String, String),
    #[error("任务已取消")]
    Cancelled,
    #[error("宿主方法 {0} 执行失败: {1}")]
    Runtime(String, String),
    #[error("文件操作失败: {0}")]
    Io(String),
    #[error("脚本执行超时")]
    Timeout,
}

/// 脚本运行请求级错误（加载、语法、未捕获异常等）。
#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("manifest 解析失败 ({path}): {source}")]
    Manifest {
        path: String,
        #[source]
        source: ManifestError,
    },
    #[error("读取脚本文件失败 ({path}): {source}")]
    ReadScript {
        #[source]
        source: std::io::Error,
        path: String,
    },
    #[error("脚本编译失败: {0}")]
    Compile(String),
    #[error("脚本运行时异常: {0}")]
    Runtime(String),
    #[error("脚本已取消")]
    Cancelled,
    #[error("脚本执行超时")]
    Timeout,
    #[error("宿主资源已失效: {0}")]
    StaleHandle(u64),
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}
