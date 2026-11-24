# bettergi-rs

BetterGI Rust 原生重构仓库（独立 Cargo Workspace，优先 GUI 解耦控制面 + 最基础地图寻路闭环）。

## Workspace 结构

- `crates/bgi-control`: UI 与后台 Core 的纯数据控制/事件协议（无 UI/Win32/OpenCV 依赖）
- `crates/bgi-capture`: Win32 窗口枚举、`BitBlt` / `WGC` 抓图与固定容量帧池
- `crates/bgi-input`: `SendInput` 键鼠模拟 + 进程级单例 `RawInputHub`
- `crates/bgi-cv`: OpenCV 视觉匹配与定位（Phase 2，可选 `--features opencv411`）
- `crates/bgi-infer`: `ort` v2.0 动态加载 PaddleOCR 与空闲卸载（Phase 2）
- `crates/bgi-pathing`: 路径 `.json` 兼容解析与最基础寻路执行器（Phase 3）
- `crates/bgi-ui`: 纯 `egui` 界面实现（仅依赖 `bgi-control`）
- `crates/bgi-app`: `bettergi.exe` 入口（`mimalloc`、Win32 托盘生命周期、按需创建/销毁 `egui-wgpu` 渲染上下文）

## 构建

```powershell
# Phase 0 / 1 编译与运行（无需提前安装 OpenCV SDK）
cargo build --workspace
cargo run -p bgi-app

# 拷贝 onnxruntime.dll / 可选下载 OpenCV 4.11 SDK
.\scripts\setup_native_deps.ps1
```
