pub mod raw_input;
pub mod simulator;

use thiserror::Error;

pub use raw_input::{RawInputHub, RawInputSignal};
pub use simulator::{
    focus_window, is_extended_key, is_foreground_window, key_down, key_tap, key_tap_checked,
    key_tap_checked_if, key_up, mouse_drag_client, mouse_left_click, mouse_left_click_client,
    mouse_left_click_client_checked, mouse_left_down, mouse_left_down_checked, mouse_left_up,
    mouse_left_up_checked, mouse_middle_down, mouse_middle_down_checked, mouse_middle_up,
    mouse_middle_up_checked, mouse_move_rel, mouse_move_rel_checked, mouse_move_to_client,
    mouse_right_click, mouse_right_down, mouse_right_up, mouse_wheel, mouse_wheel_notch_checked,
    release_all_keys, window_client_size, window_dpi_scale,
};

#[derive(Debug, Error)]
pub enum InputError {
    #[error("RawInputHub 已处于运行状态（进程内只允许单一监听实例）")]
    AlreadyRunning,
    #[error("RawInputHub 初始化失败: {0}")]
    Init(String),
}
