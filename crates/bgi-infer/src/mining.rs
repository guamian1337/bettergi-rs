use std::path::{Path, PathBuf};

use opencv::core::{
    BORDER_CONSTANT, CV_8UC3, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size,
};
use opencv::imgproc::{INTER_NEAREST, resize};
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::InferError;
use crate::paddle_ocr::get_or_init_ort_env;

const MODEL_SIZE: usize = 640;
const ANCHOR_COUNT: usize = 8400;
const DEFAULT_CONFIDENCE: f32 = 0.3;
const ORE_CONFIDENCE: f32 = 0.70;
const NMS_IOU: f32 = 0.45;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MiningDetection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

pub struct MiningDetector {
    model_path: PathBuf,
    session: Session,
}

impl MiningDetector {
    pub fn from_paths(
        model_path: impl AsRef<Path>,
        ort_dll_path: Option<PathBuf>,
    ) -> Result<Self, InferError> {
        let model_path = model_path.as_ref().to_path_buf();
        let env = get_or_init_ort_env(ort_dll_path.as_deref())?;
        let session = Session::builder(&env)
            .and_then(|builder| builder.with_intra_threads(2)?.commit_from_file(&model_path))
            .map_err(|error| InferError::Runtime(format!("加载 bgi_mine 模型失败: {error}")))?;
        validate_session(&session)?;
        Ok(Self {
            model_path,
            session,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn detect_bgr(&mut self, bgr: &Mat) -> Result<Vec<MiningDetection>, InferError> {
        if bgr.empty() || bgr.typ() != CV_8UC3 {
            return Err(InferError::Runtime(
                "采矿输入必须是非空 CV_8UC3 BGR 图像".into(),
            ));
        }
        let (input, layout) = prepare_input(bgr)?;
        let tensor = Tensor::from_array(([1usize, 3, MODEL_SIZE, MODEL_SIZE], input))
            .map_err(|error| InferError::Runtime(format!("构造采矿输入 Tensor 失败: {error}")))?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => tensor])
            .map_err(|error| InferError::Runtime(format!("采矿模型推理失败: {error}")))?;
        let output = outputs
            .get("output0")
            .ok_or_else(|| InferError::Runtime("采矿模型没有返回 output0".into()))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| InferError::Runtime(format!("读取采矿模型输出失败: {error}")))?;
        decode_output(shape, values, layout)
    }
}

fn validate_session(session: &Session) -> Result<(), InferError> {
    let valid_io = |outlet: &ort::value::Outlet, name: &str, expected: &[i64]| {
        if outlet.name() != name {
            return false;
        }
        matches!(
            outlet.dtype(),
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape,
                ..
            } if shape.iter().copied().eq(expected.iter().copied())
        )
    };
    if session.inputs().len() != 1 || !valid_io(&session.inputs()[0], "images", &[1, 3, 640, 640]) {
        return Err(InferError::Runtime(
            "采矿模型输入必须为 images: Float32 [1,3,640,640]".into(),
        ));
    }
    if session.outputs().len() != 1 || !valid_io(&session.outputs()[0], "output0", &[1, 5, 8400]) {
        return Err(InferError::Runtime(
            "采矿模型输出必须为 output0: Float32 [1,5,8400]".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Letterbox {
    source_width: u32,
    source_height: u32,
    resized_width: i32,
    resized_height: i32,
    padding_x: i32,
    padding_y: i32,
    /// 反向比例（模型像素 → 原图像素）：`max(origW, origH) / 640`。
    ///
    /// YoloSharp 的 `KeepAspectRatio` 只认这一个等比比例，x / y / w / h **统一**乘它，
    /// 不是按轴各自的 `origW / 640`、`origH / 640`（后者会把正方形框还原成非正方形）。
    scale: f32,
}

impl Letterbox {
    fn new(width: u32, height: u32) -> Result<Self, InferError> {
        if width == 0 || height == 0 {
            return Err(InferError::Runtime("采矿输入尺寸不能为零".into()));
        }
        // 等比缩放到 640×640 内：长边贴合 640，因此正向缩放系数是 640 / max(W,H)，
        // 反向比例 scale 就是它的倒数。
        let scale = width.max(height) as f32 / MODEL_SIZE as f32;
        let fit = MODEL_SIZE as f32 / width.max(height) as f32;
        let resized_width = ((width as f32 * fit).round() as i32).clamp(1, MODEL_SIZE as i32);
        let resized_height = ((height as f32 * fit).round() as i32).clamp(1, MODEL_SIZE as i32);
        Ok(Self {
            source_width: width,
            source_height: height,
            resized_width,
            resized_height,
            padding_x: (MODEL_SIZE as i32 - resized_width) / 2,
            padding_y: (MODEL_SIZE as i32 - resized_height) / 2,
            scale,
        })
    }

    fn restore(self, x: f32, y: f32, width: f32, height: f32) -> (f32, f32, f32, f32) {
        (
            (x - self.padding_x as f32) * self.scale,
            (y - self.padding_y as f32) * self.scale,
            width * self.scale,
            height * self.scale,
        )
    }
}

fn prepare_input(bgr: &Mat) -> Result<(Vec<f32>, Letterbox), InferError> {
    let layout = Letterbox::new(bgr.cols() as u32, bgr.rows() as u32)?;
    let mut resized = Mat::default();
    resize(
        bgr,
        &mut resized,
        Size::new(layout.resized_width, layout.resized_height),
        0.0,
        0.0,
        INTER_NEAREST,
    )
    .map_err(|error| InferError::Runtime(format!("采矿图像缩放失败: {error}")))?;
    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &resized,
        &mut padded,
        layout.padding_y,
        MODEL_SIZE as i32 - layout.resized_height - layout.padding_y,
        layout.padding_x,
        MODEL_SIZE as i32 - layout.resized_width - layout.padding_x,
        BORDER_CONSTANT,
        Scalar::all(0.0),
    )
    .map_err(|error| InferError::Runtime(format!("采矿图像填充失败: {error}")))?;
    if padded.rows() != MODEL_SIZE as i32
        || padded.cols() != MODEL_SIZE as i32
        || padded.typ() != CV_8UC3
        || !padded.is_continuous()
    {
        return Err(InferError::Runtime(
            "采矿预处理输出不是连续 640×640 BGR 图像".into(),
        ));
    }
    let bytes = padded
        .data_bytes()
        .map_err(|error| InferError::Runtime(format!("读取采矿预处理像素失败: {error}")))?;
    Ok((bgr_to_rgb_nchw(bytes), layout))
}

fn bgr_to_rgb_nchw(bytes: &[u8]) -> Vec<f32> {
    let plane = MODEL_SIZE * MODEL_SIZE;
    let mut input = vec![0.0f32; 3 * plane];
    for index in 0..plane {
        let base = index * 3;
        input[index] = f32::from(bytes[base + 2]) / 255.0;
        input[plane + index] = f32::from(bytes[base + 1]) / 255.0;
        input[2 * plane + index] = f32::from(bytes[base]) / 255.0;
    }
    input
}

fn decode_output(
    shape: &[i64],
    values: &[f32],
    layout: Letterbox,
) -> Result<Vec<MiningDetection>, InferError> {
    if shape != [1, 5, ANCHOR_COUNT as i64] || values.len() != 5 * ANCHOR_COUNT {
        return Err(InferError::Runtime(format!(
            "采矿输出必须为 Float32 [1,5,{ANCHOR_COUNT}]，实际 shape={shape:?}, len={}",
            values.len()
        )));
    }
    let mut candidates = Vec::new();
    for anchor in 0..ANCHOR_COUNT {
        let confidence = values[4 * ANCHOR_COUNT + anchor];
        if !confidence.is_finite()
            || confidence <= DEFAULT_CONFIDENCE
            || confidence < ORE_CONFIDENCE
        {
            continue;
        }
        let cx = values[anchor];
        let cy = values[ANCHOR_COUNT + anchor];
        let width = values[2 * ANCHOR_COUNT + anchor];
        let height = values[3 * ANCHOR_COUNT + anchor];
        if ![cx, cy, width, height]
            .iter()
            .all(|value| value.is_finite())
            || width <= 0.0
            || height <= 0.0
        {
            continue;
        }
        let (x, y, width, height) =
            layout.restore(cx - width / 2.0, cy - height / 2.0, width, height);
        candidates.push(MiningDetection {
            x,
            y,
            width,
            height,
            confidence,
        });
    }
    Ok(non_max_suppression(candidates))
}

fn non_max_suppression(mut boxes: Vec<MiningDetection>) -> Vec<MiningDetection> {
    boxes.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut kept: Vec<MiningDetection> = Vec::new();
    for candidate in boxes {
        if kept
            .iter()
            .all(|selected| iou(candidate, *selected) <= NMS_IOU)
        {
            kept.push(candidate);
        }
    }
    kept
}

fn iou(left: MiningDetection, right: MiningDetection) -> f32 {
    let left_area = left.width * left.height;
    let right_area = right.width * right.height;
    if left_area <= 0.0 || right_area <= 0.0 {
        return 0.0;
    }
    let width = (left.x + left.width).min(right.x + right.width) - left.x.max(right.x);
    let height = (left.y + left.height).min(right.y + right.height) - left.y.max(right.y);
    let intersection = width.max(0.0) * height.max(0.0);
    intersection / (left_area + right_area - intersection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::MatTrait;

    fn detection(x: f32, y: f32, width: f32, height: f32, confidence: f32) -> MiningDetection {
        MiningDetection {
            x,
            y,
            width,
            height,
            confidence,
        }
    }

    fn output(candidates: &[(usize, f32, f32, f32, f32, f32)]) -> Vec<f32> {
        let mut values = vec![0.0; 5 * ANCHOR_COUNT];
        for &(anchor, cx, cy, width, height, confidence) in candidates {
            values[anchor] = cx;
            values[ANCHOR_COUNT + anchor] = cy;
            values[2 * ANCHOR_COUNT + anchor] = width;
            values[3 * ANCHOR_COUNT + anchor] = height;
            values[4 * ANCHOR_COUNT + anchor] = confidence;
        }
        values
    }

    #[test]
    fn preprocessing_uses_rgb_normalization_and_centered_black_padding() {
        let mut image = Mat::new_rows_cols_with_default(
            1080,
            1920,
            CV_8UC3,
            Scalar::new(10.0, 20.0, 30.0, 0.0),
        )
        .unwrap();
        image.at_2d_mut::<opencv::core::Vec3b>(0, 0).unwrap()[0] = 40;
        let (input, layout) = prepare_input(&image).unwrap();
        assert_eq!(layout.resized_width, 640);
        assert_eq!(layout.resized_height, 360);
        assert_eq!((layout.padding_x, layout.padding_y), (0, 140));
        let plane = MODEL_SIZE * MODEL_SIZE;
        assert_eq!(input[140 * MODEL_SIZE], 30.0 / 255.0);
        assert_eq!(input[plane + 140 * MODEL_SIZE], 20.0 / 255.0);
        assert_eq!(input[2 * plane + 140 * MODEL_SIZE], 40.0 / 255.0);
        assert_eq!(input[0], 0.0);
        assert_eq!(input[plane], 0.0);
        assert_eq!(input[2 * plane], 0.0);
    }

    #[test]
    fn letterbox_restores_model_coordinates_to_original_frame() {
        // YoloSharp `KeepAspectRatio`：反向比例恒为 max(origW, origH) / 640，
        // x / y / w / h 统一乘它。
        let landscape = Letterbox::new(1920, 1080).unwrap();
        assert_eq!(
            (landscape.resized_width, landscape.resized_height),
            (640, 360)
        );
        assert_eq!((landscape.padding_x, landscape.padding_y), (0, 140));
        assert_eq!(landscape.scale, 3.0);
        assert_eq!(
            landscape.restore(320.0, 320.0, 20.0, 10.0),
            (960.0, 540.0, 60.0, 30.0)
        );

        // 竖屏：长边是原来的高，比例仍是 1920 / 640 = 3，padding 落在左右。
        let portrait = Letterbox::new(1080, 1920).unwrap();
        assert_eq!(
            (portrait.resized_width, portrait.resized_height),
            (360, 640)
        );
        assert_eq!((portrait.padding_x, portrait.padding_y), (140, 0));
        assert_eq!(portrait.scale, 3.0);
        assert_eq!(
            portrait.restore(320.0, 320.0, 20.0, 10.0),
            (540.0, 960.0, 60.0, 30.0)
        );
    }

    #[test]
    fn letterbox_restore_keeps_a_square_box_square() {
        // 4:3 画面下按轴比例（1440/640 = 2.25、1080/640 = 1.6875）会把方框还原成
        // 225×168.75；统一比例必须仍是正方形，这是该修正最容易漏掉的回归。
        let layout = Letterbox::new(1440, 1080).unwrap();
        assert_eq!((layout.resized_width, layout.resized_height), (640, 480));
        assert_eq!((layout.padding_x, layout.padding_y), (0, 80));
        assert_eq!(layout.scale, 2.25);
        assert_eq!(
            layout.restore(320.0, 320.0, 40.0, 40.0),
            (720.0, 540.0, 90.0, 90.0)
        );
    }

    #[test]
    fn letterbox_restore_inverts_preprocessing_within_one_scaled_pixel() {
        for (width, height) in [
            (1920u32, 1080u32),
            (1080, 1920),
            (2560, 1080),
            (1000, 1000),
            (1280, 720),
        ] {
            let layout = Letterbox::new(width, height).unwrap();
            assert_eq!(
                layout.scale,
                width.max(height) as f32 / MODEL_SIZE as f32,
                "{width}x{height}"
            );
            // 缩放后的边长取整会带来最多半个模型像素的偏差。
            let tolerance = layout.scale * 0.5 + 1e-4;
            for (source_x, source_y) in [
                (0.0f32, 0.0f32),
                ((width as f32 - 1.0) / 2.0, (height as f32 - 1.0) / 2.0),
                (width as f32 - 1.0, height as f32 - 1.0),
            ] {
                let model_x = source_x / layout.scale + layout.padding_x as f32;
                let model_y = source_y / layout.scale + layout.padding_y as f32;
                let (x, y, _, _) = layout.restore(model_x, model_y, 0.0, 0.0);
                assert!(
                    (x - source_x).abs() <= tolerance && (y - source_y).abs() <= tolerance,
                    "{width}x{height}: ({source_x}, {source_y}) -> ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn decoder_keeps_inclusive_ore_threshold_and_restores_boxes() {
        let values = output(&[
            (0, 320.0, 320.0, 20.0, 10.0, 0.70),
            (1, 100.0, 100.0, 20.0, 10.0, 0.699_999),
            (2, 500.0, 500.0, 0.0, 10.0, 0.90),
            (3, f32::NAN, 500.0, 20.0, 10.0, 0.90),
        ]);
        let result = decode_output(
            &[1, 5, ANCHOR_COUNT as i64],
            &values,
            Letterbox::new(1920, 1080).unwrap(),
        )
        .unwrap();
        assert_eq!(result, vec![detection(930.0, 525.0, 60.0, 30.0, 0.70)]);
    }

    #[test]
    fn nms_sorts_confidence_and_suppresses_only_above_iou_threshold() {
        let result = non_max_suppression(vec![
            detection(0.0, 0.0, 10.0, 10.0, 0.80),
            detection(0.0, 0.0, 10.0, 10.0, 0.90),
            detection(6.0, 0.0, 10.0, 10.0, 0.75),
            detection(20.0, 20.0, 10.0, 10.0, 0.72),
        ]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].confidence, 0.90);
        assert_eq!(result[1].x, 6.0);
        assert_eq!(result[2].x, 20.0);
        assert!(
            iou(
                detection(0.0, 0.0, 10.0, 10.0, 0.9),
                detection(6.0, 0.0, 10.0, 10.0, 0.8)
            ) < NMS_IOU
        );
    }

    #[test]
    fn decoder_rejects_wrong_shape_or_buffer_length() {
        let layout = Letterbox::new(1920, 1080).unwrap();
        assert!(decode_output(&[1, 84, 8400], &vec![0.0; 84 * ANCHOR_COUNT], layout).is_err());
        assert!(decode_output(&[1, 5, 8400], &[0.0], layout).is_err());
        assert!(Letterbox::new(0, 1080).is_err());
    }

    #[test]
    #[ignore = "requires BGI_MINE_MODEL and native ONNX Runtime"]
    fn real_mining_model_accepts_black_frame() {
        let path =
            std::env::var("BGI_MINE_MODEL").expect("BGI_MINE_MODEL must point to bgi_mine.onnx");
        let mut detector = MiningDetector::from_paths(path, None).unwrap();
        let image = Mat::new_rows_cols_with_default(1080, 1920, CV_8UC3, Scalar::all(0.0)).unwrap();
        assert!(detector.detect_bgr(&image).unwrap().is_empty());
    }
}
