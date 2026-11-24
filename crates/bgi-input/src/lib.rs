pub mod raw_input;
pub mod simulator;

use thiserror::Error;

pub use raw_input::{RawInputHub, RawInputSignal};
pub use simulator::{
    focus_window, is_extended_key, key_down, key_tap, key_up, mouse_drag_client, mouse_left_click,
    mouse_left_click_client, mouse_left_down, mouse_left_up, mouse_move_rel, mouse_move_to_client,
    mouse_right_click, mouse_wheel, release_all_keys, window_dpi_scale,
};

#[derive(Debug, Error)]
pub enum InputError {
    #[error("RawInputHub 已处于运行状态（进程内只允许单一监听实例）")]
    AlreadyRunning,
    #[error("RawInputHub 初始化失败: {0}")]
    Init(String),
}
