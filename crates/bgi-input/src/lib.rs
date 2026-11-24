pub mod raw_input;
pub mod simulator;

use thiserror::Error;

pub use raw_input::{RawInputHub, RawInputSignal};
pub use simulator::{
    is_extended_key, key_down, key_tap, key_up, mouse_left_click, mouse_move_rel,
    mouse_right_click, release_all_keys,
};

#[derive(Debug, Error)]
pub enum InputError {
    #[error("RawInputHub 已处于运行状态（进程内只允许单一监听实例）")]
    AlreadyRunning,
    #[error("RawInputHub 初始化失败: {0}")]
    Init(String),
}
