mod core;
mod pathing_host;
mod tray;
mod ui_host;

use bgi_control::{CoreCommand, UiSnapshot, create_control_bus};
use mimalloc::MiMalloc;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, MSG, PM_REMOVE, PeekMessageW, PostQuitMessage,
    TranslateMessage, WM_QUIT,
};

use crate::core::spawn_core_actor;
use crate::tray::{
    TrayHost, WM_APP_CAPTURE_ONCE, WM_APP_EMERGENCY_STOP, WM_APP_OPEN_UI,
};
use crate::ui_host::{UiHost, WM_APP_CLOSE_UI, WM_APP_QUIT_APP};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }

    let (mut client, dispatcher, core_endpoint) = create_control_bus(UiSnapshot::default());
    let core_handle = spawn_core_actor(core_endpoint);

    let tray = TrayHost::create().expect("创建系统托盘失败");
    let start_minimized = std::env::args().any(|a| a == "--minimized");

    let mut ui_host: Option<UiHost> = if start_minimized {
        None
    } else {
        match UiHost::open(&tray, &client) {
            Ok(host) => Some(host),
            Err(err) => {
                tracing::error!("打开主窗口失败: {err}");
                None
            }
        }
    };

    let mut msg = MSG::default();
    let mut quit_loop = false;
    while !quit_loop && unsafe { GetMessageW(&mut msg, None, 0, 0) }.as_bool() {
        let mut batch_msg = msg;
        loop {
            match batch_msg.message {
                WM_QUIT => {
                    quit_loop = true;
                    break;
                }
                WM_APP_OPEN_UI => {
                    if ui_host.is_none() {
                        match UiHost::open(&tray, &client) {
                            Ok(host) => ui_host = Some(host),
                            Err(err) => tracing::error!("重建主窗口失败: {err}"),
                        }
                    }
                }
                WM_APP_CLOSE_UI => {
                    if let Some(host) = ui_host.take() {
                        host.teardown(&client);
                    }
                }
                WM_APP_CAPTURE_ONCE => {
                    let _ = dispatcher.try_send(CoreCommand::CaptureSingleFrame);
                }
                WM_APP_EMERGENCY_STOP => {
                    let _ = dispatcher.try_send(CoreCommand::EmergencyStop);
                }
                WM_APP_QUIT_APP => unsafe {
                    PostQuitMessage(0);
                },
                _ => {}
            }

            unsafe {
                let _ = TranslateMessage(&batch_msg);
                DispatchMessageW(&batch_msg);
            }

            if !unsafe { PeekMessageW(&mut batch_msg, None, 0, 0, PM_REMOVE) }.as_bool() {
                break;
            }
        }

        if quit_loop {
            break;
        }

        if let Some(host) = ui_host.as_mut() {
            if host.close_requested() {
                if let Some(taken) = ui_host.take() {
                    taken.teardown(&client);
                }
            } else {
                host.render_if_needed(&mut client);
                if host.close_requested()
                    && let Some(taken) = ui_host.take()
                {
                    taken.teardown(&client);
                }
            }
        }
    }

    if let Some(host) = ui_host.take() {
        host.teardown(&client);
    }

    drop(tray);
    let _ = dispatcher.try_send(CoreCommand::ShutdownCore);
    core_handle.join();
}
