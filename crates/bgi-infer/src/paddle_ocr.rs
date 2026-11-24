use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bgi_cv::matchers::frame_to_bgr_mat;
use opencv::core::{
    self, BORDER_REPLICATE, CMP_GT, CV_8UC1, Mat, MatTraitConst, MatTraitConstManual, Point,
    Point2f, Rect, RotatedRect, Scalar, Size, Vector,
};
use opencv::imgproc::{
    self, CHAIN_APPROX_SIMPLE, INTER_CUBIC, INTER_LINEAR, RETR_LIST, fill_poly,
};
use opencv::prelude::MatExprTraitConst;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use crate::{IdleUnloadPolicy, InferError, OcrRegion, OcrResult, OcrService};

static ORT_ENV: OnceLock<Arc<Environment>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct PaddleOcrPaths {
    pub ort_dll_path: Option<PathBuf>,
    pub det_model_path: PathBuf,
    pub rec_model_path: PathBuf,
    pub rec_yml_path: PathBuf,
}

impl PaddleOcrPaths {
    /// Resolves PP-OCRv5 (falling back to V4 if V5 absent) under an `Assets/Model/PaddleOCR` folder.
    pub fn from_model_root(paddle_dir: impl AsRef<Path>, ort_dll_path: Option<PathBuf>) -> Option<Self> {
        let root = paddle_dir.as_ref();
        let candidates = [
            (
                root.join(r"Det\V5\PP-OCRv5_mobile_det_infer\slim.onnx"),
                root.join(r"Rec\V5\PP-OCRv5_mobile_rec_infer\slim.onnx"),
                root.join(r"Rec\V5\PP-OCRv5_mobile_rec_infer\inference.yml"),
            ),
            (
                root.join(r"Det\V4\PP-OCRv4_mobile_det_infer\slim.onnx"),
                root.join(r"Rec\V4\PP-OCRv4_mobile_rec_infer\slim.onnx"),
                root.join(r"Rec\V4\PP-OCRv4_mobile_rec_infer\inference.yml"),
            ),
        ];

        for (det, rec, yml) in candidates {
            if det.exists() && rec.exists() && yml.exists() {
                return Some(Self {
                    ort_dll_path,
                    det_model_path: det,
                    rec_model_path: rec,
                    rec_yml_path: yml,
                });
            }
        }
        None
    }
}

fn find_ort_dll(preferred: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = preferred
        && p.exists()
    {
        return Some(p.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let beside = dir.join("onnxruntime.dll");
        if beside.exists() {
            return Some(beside);
        }
    }
    for fallback in [
        r"d:\bgiprro\bettergi-rs\vendor\ort\onnxruntime.dll",
        r"d:\bgiprro\bgi_publish_new\runtimes\win-x64\native\onnxruntime.dll",
    ] {
        let p = Path::new(fallback);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

fn get_or_init_ort_env(preferred_dll: Option<&Path>) -> Result<Arc<Environment>, InferError> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(Arc::clone(env));
    }
    let dll = find_ort_dll(preferred_dll).ok_or_else(|| {
        InferError::Unavailable("未找到 onnxruntime.dll，请先运行 scripts/setup_native_deps.ps1".into())
    })?;
    let env = ort::init_from(&dll)
        .map_err(|e| InferError::Unavailable(format!("加载 {dll:?} 失败: {e}")))?
        .with_name("bgi-infer")
        .build()
        .map_err(|e| InferError::Unavailable(format!("初始化 ORT Environment 失败: {e}")))?;
    Ok(Arc::clone(ORT_ENV.get_or_init(|| Arc::new(env))))
}

/// Parses `PostProcess -> character_dict` list from PaddleOCR `inference.yml`.
pub fn parse_character_dict(yml_path: impl AsRef<Path>) -> Result<Vec<String>, InferError> {
    let content = fs::read_to_string(yml_path)?;
    let mut in_post = false;
    let mut in_dict = false;
    let mut dict = Vec::new();

    for raw_line in content.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = raw_line.len() - raw_line.trim_start().len();
        if indent == 0 {
            in_post = trimmed.starts_with("PostProcess:");
            in_dict = false;
            continue;
        }
        if !in_post {
            continue;
        }
        if !in_dict {
            if trimmed.starts_with("character_dict:") {
                in_dict = true;
            }
            continue;
        }
        if let Some(item) = trimmed.strip_prefix("- ") {
            dict.push(unescape_yaml_scalar(item));
        } else if trimmed == "-" {
            dict.push(String::new());
        } else if !raw_line.trim_start().starts_with('-') {
            break;
        }
    }

    if dict.is_empty() {
        return Err(InferError::Runtime("YAML 中未解析到 PostProcess.character_dict".into()));
    }
    Ok(dict)
}

fn unescape_yaml_scalar(raw: &str) -> String {
    let s = raw.trim_end_matches('\r');
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        return s[1..s.len() - 1].replace("''", "'");
    }
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        return s[1..s.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\");
    }
    s.to_string()
}

struct LoadedModels {
    det: Session,
    rec: Session,
    labels: Vec<String>,
}

pub struct PaddleOcrEngine {
    paths: PaddleOcrPaths,
    idle_policy: IdleUnloadPolicy,
    models: Option<LoadedModels>,
    last_used_at: Instant,
}

impl PaddleOcrEngine {
    pub fn new(paths: PaddleOcrPaths, idle_policy: IdleUnloadPolicy) -> Self {
        Self {
            paths,
            idle_policy,
            models: None,
            last_used_at: Instant::now(),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.models.is_some()
    }

    pub fn ensure_loaded(&mut self) -> Result<(), InferError> {
        self.last_used_at = Instant::now();
        if self.models.is_some() {
            return Ok(());
        }

        let env = get_or_init_ort_env(self.paths.ort_dll_path.as_deref())?;
        let labels = parse_character_dict(&self.paths.rec_yml_path)?;
        let det = Session::builder(&env)
            .and_then(|b| b.with_intra_threads(2)?.commit_from_file(&self.paths.det_model_path))
            .map_err(|e| InferError::Runtime(format!("加载 Det 模型失败: {e}")))?;
        let rec = Session::builder(&env)
            .and_then(|b| b.with_intra_threads(2)?.commit_from_file(&self.paths.rec_model_path))
            .map_err(|e| InferError::Runtime(format!("加载 Rec 模型失败: {e}")))?;

        self.models = Some(LoadedModels { det, rec, labels });
        Ok(())
    }

    pub fn unload_now(&mut self) {
        self.models = None;
    }

    pub fn recognize_bgr_mat(&mut self, bgr: &Mat) -> Result<OcrResult, InferError> {
        self.ensure_loaded()?;
        let loaded = self.models.as_mut().unwrap();
        let boxes = run_det(&mut loaded.det, bgr)?;
        if boxes.is_empty() {
            return Ok(OcrResult::default());
        }

        let mut regions = Vec::with_capacity(boxes.len());
        let mut texts = Vec::with_capacity(boxes.len());
        for b in boxes {
            let crop = get_rotate_crop_image(bgr, &b.points)?;
            let (text, score) = run_rec(&mut loaded.rec, &crop, &loaded.labels)?;
            if score >= 0.5 && !text.is_empty() {
                let rr = imgproc::min_area_rect(&Vector::<Point2f>::from_slice(&b.points))
                    .map_err(|e| InferError::Runtime(e.message))?;
                let br = rr.bounding_rect().map_err(|e| InferError::Runtime(e.message))?;
                texts.push(text.clone());
                regions.push(OcrRegion {
                    text,
                    score,
                    x: br.x as f32,
                    y: br.y as f32,
                    width: br.width as f32,
                    height: br.height as f32,
                });
            }
        }

        Ok(OcrResult {
            text: texts.join("\n"),
            regions,
        })
    }

    /// Runs OCR inside a normalized `[0..1]` rectangle `(norm_x0, norm_y0)..(norm_x1, norm_y1)`
    /// and returns `(origin_x, origin_y, OcrResult)` in full-frame coordinates.
    pub fn recognize_frame_roi(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        is_bgr24: bool,
        norm_x0: f32,
        norm_y0: f32,
        norm_x1: f32,
        norm_y1: f32,
    ) -> Result<(i32, i32, OcrResult), InferError> {
        let full_bgr = frame_to_bgr_mat(bytes, width, height, is_bgr24)?;
        let iw = full_bgr.cols().max(1);
        let ih = full_bgr.rows().max(1);
        let x0 = ((norm_x0.clamp(0.0, 1.0) * iw as f32).floor() as i32).clamp(0, iw - 1);
        let y0 = ((norm_y0.clamp(0.0, 1.0) * ih as f32).floor() as i32).clamp(0, ih - 1);
        let x1 = ((norm_x1.clamp(0.0, 1.0) * iw as f32).ceil() as i32).clamp(x0 + 1, iw);
        let y1 = ((norm_y1.clamp(0.0, 1.0) * ih as f32).ceil() as i32).clamp(y0 + 1, ih);

        let view = Mat::roi(&full_bgr, Rect::new(x0, y0, x1 - x0, y1 - y0))
            .map_err(|e| InferError::Runtime(e.message))?;
        let contiguous = view.try_clone().map_err(|e| InferError::Runtime(e.message))?;
        let res = self.recognize_bgr_mat(&contiguous)?;
        Ok((x0, y0, res))
    }

    pub fn recognize_bgra_roi(
        &mut self,
        bgra: &[u8],
        width: u32,
        height: u32,
        norm_x0: f32,
        norm_y0: f32,
        norm_x1: f32,
        norm_y1: f32,
    ) -> Result<(i32, i32, OcrResult), InferError> {
        self.recognize_frame_roi(bgra, width, height, false, norm_x0, norm_y0, norm_x1, norm_y1)
    }
}

impl OcrService for PaddleOcrEngine {
    fn recognize_bgra(&mut self, bgra: &[u8], width: u32, height: u32) -> Result<OcrResult, InferError> {
        let bgr = frame_to_bgr_mat(bgra, width, height, false)?;
        self.recognize_bgr_mat(&bgr)
    }

    fn unload_if_idle(&mut self) {
        if self.models.is_some()
            && self.idle_policy.idle_timeout > Duration::ZERO
            && self.last_used_at.elapsed() >= self.idle_policy.idle_timeout
        {
            self.models = None;
        }
    }
}

struct DetBox {
    points: [Point2f; 4],
}

fn det_resize_for_test(src: &Mat, limit_side_len: i32) -> Result<Mat, InferError> {
    let padded = if src.cols() + src.rows() < 64 {
        let nw = src.cols().max(32);
        let nh = src.rows().max(32);
        let mut canvas = Mat::zeros(nh, nw, src.typ())
            .map_err(|e| InferError::Runtime(e.message))?
            .to_mat()
            .map_err(|e| InferError::Runtime(e.message))?;
        let mut roi = Mat::roi_mut(&mut canvas, opencv::core::Rect::new(0, 0, src.cols(), src.rows()))
            .map_err(|e| InferError::Runtime(e.message))?;
        src.copy_to(&mut roi).map_err(|e| InferError::Runtime(e.message))?;
        canvas
    } else {
        src.try_clone().map_err(|e| InferError::Runtime(e.message))?
    };

    let w = padded.cols();
    let h = padded.rows();
    let max_side = w.max(h);
    let ratio = if max_side > limit_side_len {
        (limit_side_len as f64) / (max_side as f64)
    } else {
        1.0
    };

    let mut rh = ((h as f64) * ratio).round() as i32;
    let mut rw = ((w as f64) * ratio).round() as i32;
    rh = (((rh as f64) / 32.0).round() as i32 * 32).max(32);
    rw = (((rw as f64) / 32.0).round() as i32 * 32).max(32);

    if rw == w && rh == h {
        return Ok(padded);
    }
    let mut out = Mat::default();
    imgproc::resize(&padded, &mut out, Size::new(rw, rh), 0.0, 0.0, INTER_LINEAR)
        .map_err(|e| InferError::Runtime(e.message))?;
    drop(padded);
    Ok(out)
}

fn mat_bgr_to_nchw_norm(bgr: &Mat, mean: [f32; 3], std: [f32; 3], scale: f32) -> Result<Vec<f32>, InferError> {
    let h = bgr.rows() as usize;
    let w = bgr.cols() as usize;
    let plane = h * w;
    let mut out = vec![0.0f32; 3 * plane];
    let bytes = bgr.data_bytes().map_err(|e| InferError::Runtime(e.message))?;

    for idx in 0..plane {
        let base = idx * 3;
        out[idx] = ((bytes[base] as f32) * scale - mean[0]) / std[0];
        out[plane + idx] = ((bytes[base + 1] as f32) * scale - mean[1]) / std[1];
        out[2 * plane + idx] = ((bytes[base + 2] as f32) * scale - mean[2]) / std[2];
    }
    Ok(out)
}

fn run_det(session: &mut Session, src_bgr: &Mat) -> Result<Vec<DetBox>, InferError> {
    let resized = det_resize_for_test(src_bgr, 960)?;
    let rw = resized.cols();
    let rh = resized.rows();

    let nchw = mat_bgr_to_nchw_norm(
        &resized,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        1.0 / 255.0,
    )?;
    let input_tensor = Tensor::from_array(([1usize, 3, rh as usize, rw as usize], nchw))
        .map_err(|e| InferError::Runtime(format!("Det Tensor 构造失败: {e}")))?;

    let input_name = session.inputs()[0].name().to_string();
    let outputs = session
        .run(ort::inputs![input_name => input_tensor])
        .map_err(|e| InferError::Runtime(format!("Det 推理失败: {e}")))?;

    let (shape, pred_slice) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferError::Runtime(format!("读取 Det 输出失败: {e}")))?;
    if shape.len() != 4 {
        return Err(InferError::Runtime(format!("Det 输出维度异常: {shape:?}")));
    }
    let out_h = shape[2] as i32;
    let _out_w = shape[3] as i32;

    let pred_base = Mat::from_slice(pred_slice).map_err(|e| InferError::Runtime(e.message))?;
    let pred = pred_base
        .reshape(1, out_h)
        .map_err(|e| InferError::Runtime(e.message))?
        .clone_pointee();
    let mut boxes = db_post_process(&pred, src_bgr.cols(), src_bgr.rows())?;
    sort_boxes(&mut boxes);
    Ok(boxes)
}

fn db_post_process(pred: &Mat, src_w: i32, src_h: i32) -> Result<Vec<DetBox>, InferError> {
    const BOX_THRESH: f32 = 0.3;
    const BOX_SCORE_THRESH: f32 = 0.6;
    const UNCLIP_RATIO: f32 = 1.5;
    const MIN_SIZE: f32 = 3.0;
    const MAX_CANDIDATES: usize = 1000;

    let mut bitmap = Mat::default();
    core::compare(pred, &Scalar::all(BOX_THRESH as f64), &mut bitmap, CMP_GT)
        .map_err(|e| InferError::Runtime(e.message))?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(&bitmap, &mut contours, RETR_LIST, CHAIN_APPROX_SIMPLE, Point::default())
        .map_err(|e| InferError::Runtime(e.message))?;

    let pred_w = pred.cols();
    let pred_h = pred.rows();
    let mut out = Vec::new();

    for c_idx in 0..contours.len().min(MAX_CANDIDATES) {
        let contour = contours.get(c_idx).map_err(|e| InferError::Runtime(e.message))?;
        let mut pts_f = Vector::<Point2f>::with_capacity(contour.len());
        for p_idx in 0..contour.len() {
            let p = contour.get(p_idx).map_err(|e| InferError::Runtime(e.message))?;
            pts_f.push(Point2f::new(p.x as f32, p.y as f32));
        }

        let (mini_box, short_side) = min_box_from_points(&pts_f)?;
        if short_side < MIN_SIZE {
            continue;
        }

        let score = box_score(&mini_box, pred)?;
        if score < BOX_SCORE_THRESH {
            continue;
        }

        // Unclip rotated box cleanly: inflate width & height by 2 * distance along box axes
        let (area, perim) = box_area_and_perimeter(&mini_box);
        if area <= 0.0 || perim <= 0.0 {
            continue;
        }
        let dist = (area * UNCLIP_RATIO) / perim;
        let inflated = inflate_rect_box(&pts_f, dist)?;
        let (expanded_box, expanded_short) = min_box_from_rect(&inflated);
        if expanded_short < MIN_SIZE + 2.0 {
            continue;
        }

        let mapped = map_to_source(&expanded_box, pred_w, pred_h, src_w, src_h);
        let bw = dist_2f(mapped[0], mapped[1]) as i32;
        let bh = dist_2f(mapped[0], mapped[3]) as i32;
        if bw > MIN_SIZE as i32 && bh > MIN_SIZE as i32 {
            out.push(DetBox { points: mapped });
        }
    }

    Ok(out)
}

fn min_box_from_points(pts: &Vector<Point2f>) -> Result<([Point2f; 4], f32), InferError> {
    let rect = imgproc::min_area_rect(pts).map_err(|e| InferError::Runtime(e.message))?;
    Ok(min_box_from_rect(&rect))
}

fn min_box_from_rect(rect: &RotatedRect) -> ([Point2f; 4], f32) {
    let mut corners = [Point2f::default(); 4];
    rect.points(&mut corners).unwrap_or_default();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));

    let (idx1, idx4) = if corners[1].y > corners[0].y { (0, 1) } else { (1, 0) };
    let (idx2, idx3) = if corners[3].y > corners[2].y { (2, 3) } else { (3, 2) };
    let short = rect.size.width.min(rect.size.height);
    ([corners[idx1], corners[idx2], corners[idx3], corners[idx4]], short)
}

fn inflate_rect_box(pts: &Vector<Point2f>, distance: f32) -> Result<RotatedRect, InferError> {
    let mut rr = imgproc::min_area_rect(pts).map_err(|e| InferError::Runtime(e.message))?;
    rr.size.width += 2.0 * distance;
    rr.size.height += 2.0 * distance;
    Ok(rr)
}

fn box_score(box_pts: &[Point2f; 4], pred: &Mat) -> Result<f32, InferError> {
    let xmin = box_pts.iter().map(|p| p.x.floor() as i32).min().unwrap_or(0).clamp(0, pred.cols() - 1);
    let xmax = box_pts.iter().map(|p| p.x.ceil() as i32).max().unwrap_or(0).clamp(0, pred.cols() - 1);
    let ymin = box_pts.iter().map(|p| p.y.floor() as i32).min().unwrap_or(0).clamp(0, pred.rows() - 1);
    let ymax = box_pts.iter().map(|p| p.y.ceil() as i32).max().unwrap_or(0).clamp(0, pred.rows() - 1);

    let w = (xmax - xmin + 1).max(1);
    let h = (ymax - ymin + 1).max(1);
    let mut mask = Mat::zeros(h, w, CV_8UC1)
        .map_err(|e| InferError::Runtime(e.message))?
        .to_mat()
        .map_err(|e| InferError::Runtime(e.message))?;

    let mut poly = Vector::<Point>::with_capacity(4);
    for p in box_pts {
        poly.push(Point::new((p.x as i32) - xmin, (p.y as i32) - ymin));
    }
    let mut polys = Vector::<Vector<Point>>::new();
    polys.push(poly);
    fill_poly(&mut mask, &polys, Scalar::all(255.0), imgproc::LINE_8, 0, Point::default())
        .map_err(|e| InferError::Runtime(e.message))?;

    let crop = Mat::roi(pred, opencv::core::Rect::new(xmin, ymin, w, h))
        .map_err(|e| InferError::Runtime(e.message))?;
    let mean = core::mean(&crop, &mask).map_err(|e| InferError::Runtime(e.message))?;
    Ok(mean[0] as f32)
}

fn box_area_and_perimeter(pts: &[Point2f; 4]) -> (f32, f32) {
    let mut area = 0.0f32;
    let mut perim = 0.0f32;
    for i in 0..4 {
        let next = pts[(i + 1) % 4];
        area += pts[i].x * next.y - next.x * pts[i].y;
        perim += dist_2f(pts[i], next);
    }
    (area.abs() / 2.0, perim)
}

fn dist_2f(a: Point2f, b: Point2f) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn map_to_source(pts: &[Point2f; 4], pred_w: i32, pred_h: i32, src_w: i32, src_h: i32) -> [Point2f; 4] {
    let sx = (src_w as f32) / (pred_w as f32);
    let sy = (src_h as f32) / (pred_h as f32);
    let mut out = [Point2f::default(); 4];
    for i in 0..4 {
        out[i] = Point2f::new(
            (pts[i].x * sx).round().clamp(0.0, (src_w - 1) as f32),
            (pts[i].y * sy).round().clamp(0.0, (src_h - 1) as f32),
        );
    }
    out
}

fn sort_boxes(boxes: &mut [DetBox]) {
    boxes.sort_by(|a, b| {
        a.points[0]
            .y
            .total_cmp(&b.points[0].y)
            .then_with(|| a.points[0].x.total_cmp(&b.points[0].x))
    });
    for i in 0..boxes.len().saturating_sub(1) {
        let mut j = i as isize;
        while j >= 0 {
            let idx = j as usize;
            if (boxes[idx + 1].points[0].y - boxes[idx].points[0].y).abs() < 10.0
                && boxes[idx + 1].points[0].x < boxes[idx].points[0].x
            {
                boxes.swap(idx, idx + 1);
                j -= 1;
            } else {
                break;
            }
        }
    }
}

fn get_rotate_crop_image(src: &Mat, pts: &[Point2f; 4]) -> Result<Mat, InferError> {
    let crop_w = (dist_2f(pts[0], pts[1]).max(dist_2f(pts[2], pts[3])) as i32).max(1);
    let crop_h = (dist_2f(pts[0], pts[3]).max(dist_2f(pts[1], pts[2])) as i32).max(1);

    let src_arr = [pts[0], pts[1], pts[2], pts[3]];
    let dst_arr = [
        Point2f::new(0.0, 0.0),
        Point2f::new(crop_w as f32, 0.0),
        Point2f::new(crop_w as f32, crop_h as f32),
        Point2f::new(0.0, crop_h as f32),
    ];

    let src_m = Mat::from_slice(&src_arr).map_err(|e| InferError::Runtime(e.message))?;
    let dst_m = Mat::from_slice(&dst_arr).map_err(|e| InferError::Runtime(e.message))?;
    let transform = imgproc::get_perspective_transform(&src_m, &dst_m, core::DECOMP_LU)
        .map_err(|e| InferError::Runtime(e.message))?;

    let mut cropped = Mat::default();
    imgproc::warp_perspective(
        src,
        &mut cropped,
        &transform,
        Size::new(crop_w, crop_h),
        INTER_CUBIC,
        BORDER_REPLICATE,
        Scalar::default(),
    )
    .map_err(|e| InferError::Runtime(e.message))?;

    if (cropped.rows() as f64) / (cropped.cols().max(1) as f64) >= 1.5 {
        let mut rotated = Mat::default();
        core::rotate(&cropped, &mut rotated, core::ROTATE_90_COUNTERCLOCKWISE)
            .map_err(|e| InferError::Runtime(e.message))?;
        return Ok(rotated);
    }
    Ok(cropped)
}

fn run_rec(session: &mut Session, crop_bgr: &Mat, labels: &[String]) -> Result<(String, f32), InferError> {
    const MODEL_H: i32 = 48;
    let ratio = (crop_bgr.cols() as f64) / (crop_bgr.rows().max(1) as f64);
    let target_w = ((MODEL_H as f64) * ratio).ceil().max(1.0) as i32;

    let mut resized = Mat::default();
    imgproc::resize(crop_bgr, &mut resized, Size::new(target_w, MODEL_H), 0.0, 0.0, INTER_LINEAR)
        .map_err(|e| InferError::Runtime(e.message))?;

    let nchw = mat_bgr_to_nchw_norm(&resized, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1.0 / 255.0)?;
    let tensor = Tensor::from_array(([1usize, 3, MODEL_H as usize, target_w as usize], nchw))
        .map_err(|e| InferError::Runtime(format!("Rec Tensor 构造失败: {e}")))?;

    let input_name = session.inputs()[0].name().to_string();
    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .map_err(|e| InferError::Runtime(format!("Rec 推理失败: {e}")))?;

    let (shape, arr) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| InferError::Runtime(format!("读取 Rec 输出失败: {e}")))?;
    if shape.len() != 3 {
        return Err(InferError::Runtime(format!("Rec 输出维度异常: {shape:?}")));
    }

    let char_count = shape[1] as usize;
    let label_count = shape[2] as usize;

    let mut out_text = String::new();
    let mut last_idx = 0usize;
    let mut score_sum = 0.0f32;
    let mut token_count = 0usize;

    for n in 0..char_count {
        let row = &arr[n * label_count..(n + 1) * label_count];
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (c, &v) in row.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = c;
            }
        }
        if max_idx > 0 && !(n > 0 && max_idx == last_idx) {
            if max_idx <= labels.len() {
                out_text.push_str(&labels[max_idx - 1]);
            } else if max_idx == labels.len() + 1 {
                out_text.push(' ');
            }
            score_sum += max_val;
            token_count += 1;
        }
        last_idx = max_idx;
    }

    let avg_score = if token_count > 0 {
        score_sum / (token_count as f32)
    } else {
        0.0
    };
    Ok((out_text, avg_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs::{IMREAD_COLOR, imread};

    #[test]
    fn recognizes_paddle_ocr_test_image_and_unloads_idle() {
        let root = Path::new(r"d:\bgiprro\bgi_publish_new\Assets\Model\PaddleOCR");
        let Some(paths) = PaddleOcrPaths::from_model_root(root, None) else {
            return;
        };
        let test_img = root.join("test_pp_ocr.png");
        if !test_img.exists() {
            return;
        }

        let mat = imread(&*test_img.to_string_lossy(), IMREAD_COLOR).unwrap();
        let mut engine = PaddleOcrEngine::new(
            paths,
            IdleUnloadPolicy {
                idle_timeout: Duration::from_millis(1),
            },
        );
        let res = engine.recognize_bgr_mat(&mat).unwrap();
        println!("OCR output: {:?}", res.text);
        assert!(!res.text.trim().is_empty(), "OCR output should not be empty");
        assert!(engine.is_loaded());

        std::thread::sleep(Duration::from_millis(5));
        engine.unload_if_idle();
        assert!(!engine.is_loaded());
    }
}
