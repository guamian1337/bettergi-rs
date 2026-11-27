//! BetterGI 脚本层：`HostKernel` 三原语 + 引擎适配 + manifest 加载。
//!
//! 依赖方向：`bgi-script` 不依赖任何游戏侧 crate；`bgi-app` 实现 [`kernel::HostKernel`]
//! 并注入。引擎实现放在各自 feature 之后，接口见 `engine::Engine`。
//!
//! 设计文档：`analysis/js-script-layer-plan-20261127.md`。

pub mod error;
pub mod kernel;
pub mod manifest;

pub use error::{HostError, ManifestError, ScriptError};
pub use kernel::{HostFuture, HostHandle, HostKernel, HostValue, canonical_method};
pub use manifest::{
    Manifest, SettingsControl, merge_settings, parse_settings_ui, safe_join, safe_relative,
    settings_defaults,
};
pub mod engine;
pub use engine::{Engine, ScriptOutcome, ScriptRun, decode_host_error, encode_host_error};
pub mod quickjs;
pub use quickjs::{QuickJsEngine, QuickJsLimits};
