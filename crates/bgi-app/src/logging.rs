use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bgi_control::{LogEntry, LogLevel, LogPublisher};
use image::RgbImage;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, Layer};
use windows::Win32::System::SystemInformation::GetLocalTime;
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
use windows::core::{PCWSTR, w};

const MAX_LOG_FILES: usize = 14;
const MAX_FAILED_IMAGES: usize = 50;
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);

static FAILED_IMG_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn log_dir_path() -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    base.join("log")
}

pub fn local_date_and_time() -> (String, String) {
    let st = unsafe { GetLocalTime() };
    let date = format!("{:04}-{:02}-{:02}", st.wYear, st.wMonth, st.wDay);
    let time = format!(
        "{:02}:{:02}:{:02}.{:03}",
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds
    );
    (date, time)
}

pub fn open_log_dir() -> Result<PathBuf, String> {
    let dir = log_dir_path();
    fs::create_dir_all(&dir).map_err(|e| format!("创建日志目录失败 ({}): {e}", dir.display()))?;

    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);

    let result = unsafe {
        ShellExecuteW(
            None,
            w!("open"),
            PCWSTR(wide.as_ptr()),
            None,
            None,
            SW_SHOWNORMAL,
        )
    };
    let code = result.0 as isize;
    if code <= 32 {
        return Err(format!("ShellExecuteW 打开日志目录失败 (code={code})"));
    }
    Ok(dir)
}

pub fn failed_dir_path() -> PathBuf {
    log_dir_path().join("failed")
}

/// Saves a failed CV/capture frame as RGB PNG into `<exe_dir>/log/failed/` and prunes old snapshots.
pub fn save_failed_image(
    tag: &str,
    width: u32,
    height: u32,
    is_bgr24: bool,
    bytes: &[u8],
) -> Result<PathBuf, String> {
    save_failed_image_in_dir(&failed_dir_path(), tag, width, height, is_bgr24, bytes)
}

fn save_failed_image_in_dir(
    dir: &Path,
    tag: &str,
    width: u32,
    height: u32,
    is_bgr24: bool,
    bytes: &[u8],
) -> Result<PathBuf, String> {
    if width == 0 || height == 0 {
        return Err(format!("无效分辨率 {width}x{height}，跳过保存"));
    }
    let bpp = if is_bgr24 { 3 } else { 4 };
    let expected = (width as usize) * (height as usize) * bpp;
    if bytes.len() < expected {
        return Err(format!(
            "像素缓冲不足 ({width}x{height} bpp={bpp}, 期望 {expected}, 实际 {})",
            bytes.len()
        ));
    }

    fs::create_dir_all(dir)
        .map_err(|e| format!("创建失败截图目录失败 ({}): {e}", dir.display()))?;

    let st = unsafe { GetLocalTime() };
    let seq = FAILED_IMG_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let safe_tag: String = tag
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let filename = format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}_{:03}-{safe_tag}-{seq}.png",
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds
    );
    let path = dir.join(filename);

    let px_count = (width as usize) * (height as usize);
    let mut rgb = Vec::with_capacity(px_count * 3);
    for px in bytes[..expected].chunks_exact(bpp) {
        rgb.push(px[2]); // R
        rgb.push(px[1]); // G
        rgb.push(px[0]); // B
    }
    let img = RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| format!("构建 PNG 图像失败 ({width}x{height})"))?;
    img.save(&path)
        .map_err(|e| format!("保存失败截图出错 ({}): {e}", path.display()))?;

    prune_old_failed_images(dir, MAX_FAILED_IMAGES);
    Ok(path)
}

fn prune_old_failed_images(dir: &Path, keep: usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(String, PathBuf)> = entries
        .filter_map(|res| res.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".png") {
                Some((name, entry.path()))
            } else {
                None
            }
        })
        .collect();

    if files.len() <= keep {
        return;
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    let remove_count = files.len().saturating_sub(keep);
    for (_, path) in files.into_iter().take(remove_count) {
        let _ = fs::remove_file(path);
    }
}

struct RollingFileLogger {
    log_dir: PathBuf,
    current_date: String,
    writer: Option<BufWriter<File>>,
    last_flush: Instant,
}

impl RollingFileLogger {
    fn new(log_dir: PathBuf) -> Self {
        let mut logger = Self {
            log_dir,
            current_date: String::new(),
            writer: None,
            last_flush: Instant::now(),
        };
        let (date, _) = local_date_and_time();
        let _ = logger.ensure_writer(&date);
        logger
    }

    fn ensure_writer(&mut self, date: &str) -> io::Result<()> {
        if self.writer.is_some() && self.current_date == date {
            return Ok(());
        }

        if let Some(mut old) = self.writer.take() {
            let _ = old.flush();
        }

        fs::create_dir_all(&self.log_dir)?;
        let file_path = self.log_dir.join(format!("bettergi-{date}.log"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?;

        self.current_date = date.to_owned();
        self.writer = Some(BufWriter::new(file));
        self.last_flush = Instant::now();
        cleanup_old_logs(&self.log_dir, MAX_LOG_FILES);
        Ok(())
    }

    fn write_entry(&mut self, date: &str, line: &str, force_flush: bool) {
        if self.ensure_writer(date).is_err() {
            return;
        }
        if let Some(writer) = self.writer.as_mut() {
            let _ = writeln!(writer, "{line}");
            if force_flush || self.last_flush.elapsed() >= FLUSH_INTERVAL {
                let _ = writer.flush();
                self.last_flush = Instant::now();
            }
        }
    }

    fn flush(&mut self) {
        if let Some(writer) = self.writer.as_mut() {
            let _ = writer.flush();
            self.last_flush = Instant::now();
        }
    }
}

fn cleanup_old_logs(log_dir: &Path, keep: usize) {
    let Ok(entries) = fs::read_dir(log_dir) else {
        return;
    };
    let mut files: Vec<(String, PathBuf)> = entries
        .filter_map(|res| res.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("bettergi-") && name.ends_with(".log") {
                Some((name, entry.path()))
            } else {
                None
            }
        })
        .collect();

    if files.len() <= keep {
        return;
    }

    // Filenames use YYYY-MM-DD so lexicographical order matches chronological order.
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let remove_count = files.len().saturating_sub(keep);
    for (_, path) in files.into_iter().take(remove_count) {
        let _ = fs::remove_file(path);
    }
}

fn format_log_line(date: &str, entry: &LogEntry) -> String {
    format!(
        "[{date} {}][{}][{}] {}",
        entry.timestamp,
        entry.level.tag(),
        entry.target,
        entry.message
    )
}

#[derive(Default)]
struct MessageFieldVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl MessageFieldVisitor {
    fn record_pair(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            self.fields.push(format!("{}={}", field.name(), value));
        }
    }

    fn finish(self) -> String {
        match (self.message, self.fields.is_empty()) {
            (Some(msg), true) => msg,
            (Some(msg), false) => format!("{msg} ({})", self.fields.join(", ")),
            (None, false) => self.fields.join(", "),
            (None, true) => String::new(),
        }
    }
}

impl Visit for MessageFieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_pair(field, value.to_owned());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let rendered = format!("{value:?}");
        // Tracing sometimes records `message` through `record_debug` with surrounding quotes.
        let trimmed = if field.name() == "message"
            && rendered.starts_with('"')
            && rendered.ends_with('"')
            && rendered.len() >= 2
        {
            rendered[1..rendered.len() - 1].to_owned()
        } else {
            rendered
        };
        self.record_pair(field, trimmed);
    }
}

struct UiAndFileTracingLayer {
    publisher: LogPublisher,
}

impl<S> Layer<S> for UiAndFileTracingLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let level = match *event.metadata().level() {
            tracing::Level::ERROR => LogLevel::Error,
            tracing::Level::WARN => LogLevel::Warn,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::DEBUG | tracing::Level::TRACE => LogLevel::Debug,
        };

        let mut visitor = MessageFieldVisitor::default();
        event.record(&mut visitor);
        let msg = visitor.finish();
        self.publisher.emit(level, event.metadata().target(), msg);
    }
}

pub struct LoggingGuard {
    file_logger: Arc<Mutex<RollingFileLogger>>,
}

impl Drop for LoggingGuard {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.file_logger.lock() {
            guard.flush();
        }
    }
}

pub fn init_logging(publisher: LogPublisher) -> LoggingGuard {
    let log_dir = log_dir_path();
    let file_logger = Arc::new(Mutex::new(RollingFileLogger::new(log_dir)));

    publisher.set_timestamp_provider(Some(Arc::new(|| local_date_and_time().1)));

    let logger_for_mirror = file_logger.clone();
    publisher.set_mirror(Some(Arc::new(move |entry: &LogEntry| {
        let (date, _) = local_date_and_time();
        let line = format_log_line(&date, entry);
        eprintln!("{line}");
        let force_flush = matches!(entry.level, LogLevel::Warn | LogLevel::Error);
        let mut guard = logger_for_mirror
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        guard.write_entry(&date, &line, force_flush);
    })));

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,wgpu=warn,wgpu_core=warn,wgpu_hal=warn,naga=warn,egui_wgpu=warn,egui_glow=warn",
        )
    });

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(UiAndFileTracingLayer {
            publisher: publisher.clone(),
        })
        .try_init();

    install_panic_hook(publisher, file_logger.clone());

    LoggingGuard { file_logger }
}

fn install_panic_hook(publisher: LogPublisher, file_logger: Arc<Mutex<RollingFileLogger>>) {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let payload = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            (*s).to_owned()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_owned()
        };

        let location = panic_info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "<unknown location>".to_owned());

        let backtrace = std::backtrace::Backtrace::force_capture();
        publisher.emit(
            LogLevel::Error,
            "panic",
            format!("thread panicked at {location}: {payload}\n{backtrace}"),
        );
        if let Ok(mut guard) = file_logger.lock() {
            guard.flush();
        }

        prev_hook(panic_info);
    }));
}

#[cfg(test)]
mod tests {
    use bgi_control::{UiSnapshot, create_control_bus};

    use super::*;

    #[test]
    fn rolling_file_logger_writes_and_prunes_old_days() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("bettergi-log-test-{unique}"));
        let _ = fs::remove_dir_all(&temp_dir);

        {
            let mut logger = RollingFileLogger {
                log_dir: temp_dir.clone(),
                current_date: String::new(),
                writer: None,
                last_flush: Instant::now(),
            };
            for day in 1..=16 {
                let date = format!("2026-11-{day:02}");
                logger.write_entry(
                    &date,
                    &format!("[{date} 12:00:00.000][INF][test] day {day}"),
                    true,
                );
            }
            logger.flush();
        }

        let mut names: Vec<String> = fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(|r| r.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("bettergi-") && n.ends_with(".log"))
            .collect();
        names.sort();

        assert_eq!(names.len(), MAX_LOG_FILES);
        assert_eq!(
            names.first().map(String::as_str),
            Some("bettergi-2026-11-03.log")
        );
        assert_eq!(
            names.last().map(String::as_str),
            Some("bettergi-2026-11-16.log")
        );

        let latest = fs::read_to_string(temp_dir.join("bettergi-2026-11-16.log")).unwrap();
        assert!(latest.contains("[INF][test] day 16"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn tracing_layer_and_emit_log_share_publisher_cleanly() {
        let (client, _dispatcher, core) = create_control_bus(UiSnapshot::default());
        let publisher = core.log_publisher();
        publisher.set_timestamp_provider(Some(Arc::new(|| "09:30:00.123".to_owned())));

        let subscriber = tracing_subscriber::registry().with(UiAndFileTracingLayer {
            publisher: publisher.clone(),
        });

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(step = 3, "pathing ready");
            tracing::warn!("low hp");
        });
        core.emit_log(LogLevel::Error, "Core", "manual error");

        let logs = client.recent_logs();
        assert_eq!(logs.len(), 3);
        assert_eq!(logs[0].timestamp, "09:30:00.123");
        assert_eq!(logs[0].level, LogLevel::Info);
        assert!(logs[0].message.contains("pathing ready"));
        assert!(logs[0].message.contains("step=3"));

        assert_eq!(logs[1].level, LogLevel::Warn);
        assert_eq!(logs[1].message, "low hp");

        assert_eq!(logs[2].level, LogLevel::Error);
        assert_eq!(logs[2].target, "Core");
        assert_eq!(logs[2].message, "manual error");
    }

    #[test]
    fn save_failed_image_writes_png_and_prunes_excess() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("bettergi-failed-img-test-{unique}"));
        let _ = fs::remove_dir_all(&temp_dir);

        // 4x2 BGR24 frame
        let bgr = vec![
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12,
            13,
        ];
        for i in 0..53 {
            let saved = save_failed_image_in_dir(&temp_dir, &format!("sift_{i}"), 4, 2, true, &bgr)
                .unwrap();
            assert!(saved.exists());
        }

        let png_count = fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(|r| r.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".png"))
            .count();
        assert_eq!(png_count, MAX_FAILED_IMAGES);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
