use std::fs;
use std::path::{Path, PathBuf};

use opencv::core::{CV_8UC3, Mat, MatTraitConst, MatTraitConstManual};
use ort::session::Session;
use ort::value::Tensor;

use crate::InferError;
use crate::paddle_ocr::get_or_init_ort_env;

const INPUT_SIZE: usize = 125;
const EMBEDDING_DIM: usize = 256;
const MATCH_THRESHOLD: f64 = 0.75;

/// Paths for the ItemV2 icon model, its prototype CSV, and an optional ONNX Runtime DLL.
#[derive(Debug, Clone)]
pub struct ItemIconPaths {
    pub ort_dll_path: Option<PathBuf>,
    pub model_path: PathBuf,
    pub csv_path: PathBuf,
}

/// A matched relic icon. Different star levels of the same icon remain distinct candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactIconMatch {
    pub name: String,
    pub quality_level: u8,
    pub score: f32,
    pub visual_id: u64,
}

/// Alias using the item-recognition terminology of the underlying ItemV2 model.
pub type ItemIconMatch = ArtifactIconMatch;

/// Standalone ItemV2 icon recognizer. The caller must supply a 125x125 BGR icon.
pub struct ItemIconRecognizer {
    paths: ItemIconPaths,
    session: Session,
    prototypes: Vec<IconPrototype>,
}

#[derive(Debug, Clone)]
struct IconPrototype {
    name: String,
    quality_level: u8,
    embedding: Vec<f32>,
}

impl ItemIconRecognizer {
    /// Loads and validates the prototype CSV before creating the CPU ONNX session.
    pub fn from_paths(
        model_path: impl AsRef<Path>,
        csv_path: impl AsRef<Path>,
        ort_dll_path: Option<PathBuf>,
    ) -> Result<Self, InferError> {
        let paths = ItemIconPaths {
            ort_dll_path,
            model_path: model_path.as_ref().to_path_buf(),
            csv_path: csv_path.as_ref().to_path_buf(),
        };
        let csv = fs::read_to_string(&paths.csv_path)?;
        let prototypes = parse_prototypes(&csv)?;
        let env = get_or_init_ort_env(paths.ort_dll_path.as_deref())?;
        let session = Session::builder(&env)
            .and_then(|builder| {
                builder
                    .with_intra_threads(2)?
                    .commit_from_file(&paths.model_path)
            })
            .map_err(|e| InferError::Runtime(format!("加载 ItemV2 模型失败: {e}")))?;

        Ok(Self {
            paths,
            session,
            prototypes,
        })
    }

    /// Recognizes any ItemV2 icon at the C# match threshold.
    /// Non-relic matches have `quality_level == 0`.
    pub fn recognize_item_bgr_125(
        &mut self,
        bgr: &Mat,
    ) -> Result<Option<ItemIconMatch>, InferError> {
        if bgr.rows() as usize != INPUT_SIZE
            || bgr.cols() as usize != INPUT_SIZE
            || bgr.typ() != CV_8UC3
        {
            return Err(InferError::Runtime(format!(
                "ItemV2 输入必须为 {INPUT_SIZE}x{INPUT_SIZE}、CV_8UC3 的 BGR 图像，实际为 {}x{}、类型 {}",
                bgr.cols(),
                bgr.rows(),
                bgr.typ()
            )));
        }

        // OpenCV permits strided ROIs; make them contiguous before reading pixel bytes.
        let contiguous;
        let bgr = if bgr.is_continuous() {
            bgr
        } else {
            contiguous = bgr
                .try_clone()
                .map_err(|e| InferError::Runtime(format!("复制 ItemV2 输入失败: {e}")))?;
            &contiguous
        };
        let bytes = bgr
            .data_bytes()
            .map_err(|e| InferError::Runtime(format!("读取 ItemV2 输入失败: {e}")))?;
        let input = bgr_to_rgb_nchw(bytes);
        let visual_id = icon_visual_hash(bgr);
        let tensor = Tensor::from_array(([1usize, 3, INPUT_SIZE, INPUT_SIZE], input))
            .map_err(|e| InferError::Runtime(format!("构造 ItemV2 输入 Tensor 失败: {e}")))?;
        let input_name = self.session.inputs()[0].name().to_string();
        let outputs = self
            .session
            .run(ort::inputs![input_name => tensor])
            .map_err(|e| InferError::Runtime(format!("ItemV2 推理失败: {e}")))?;
        let output = outputs
            .get("embedding")
            .ok_or_else(|| InferError::Runtime("ItemV2 模型没有返回 embedding 输出".into()))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| InferError::Runtime(format!("读取 ItemV2 embedding 失败: {e}")))?;
        let embedding = normalize_model_output(shape, values)?;
        best_match(&embedding, &self.prototypes, visual_id)
    }

    /// Recognizes only relic icons; non-relic matches are negative evidence.
    pub fn recognize_bgr_125(&mut self, bgr: &Mat) -> Result<Option<ItemIconMatch>, InferError> {
        Ok(self
            .recognize_item_bgr_125(bgr)?
            .filter(|matched| matched.quality_level > 0))
    }

    /// Returns the validated asset paths held by this recognizer.
    pub fn paths(&self) -> &ItemIconPaths {
        &self.paths
    }
}

fn bgr_to_rgb_nchw(bytes: &[u8]) -> Vec<f32> {
    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut input = vec![0.0f32; 3 * plane];
    for index in 0..plane {
        let base = index * 3;
        // ItemV2 expects RGB planar data, normalized to [-1, 1].
        input[index] = (f32::from(bytes[base + 2]) / 255.0 - 0.5) / 0.5;
        input[plane + index] = (f32::from(bytes[base + 1]) / 255.0 - 0.5) / 0.5;
        input[2 * plane + index] = (f32::from(bytes[base]) / 255.0 - 0.5) / 0.5;
    }
    input
}

fn icon_visual_hash(bgr: &Mat) -> u64 {
    let Ok(bytes) = bgr.data_bytes() else {
        return 0;
    };
    let mut hash = 0_u64;
    for row in 0..8 {
        let y = (row * 2 + 1) * INPUT_SIZE / 16;
        for column in 0..8 {
            let left_x = (column * 2 + 1) * INPUT_SIZE / 16;
            let right_x = (column * 2 + 3) * INPUT_SIZE / 16;
            let left = (left_x + y * INPUT_SIZE) * 3;
            let right = (right_x + y * INPUT_SIZE) * 3;
            let left_luma = (19 * u32::from(bytes[left + 2])
                + 183 * u32::from(bytes[left + 1])
                + 54 * u32::from(bytes[left]))
                >> 8;
            let right_luma = (19 * u32::from(bytes[right + 2])
                + 183 * u32::from(bytes[right + 1])
                + 54 * u32::from(bytes[right]))
                >> 8;
            hash = (hash << 1) | u64::from(left_luma < right_luma);
        }
    }
    hash
}

fn normalize_model_output(shape: &[i64], values: &[f32]) -> Result<Vec<f32>, InferError> {
    if shape != [1, EMBEDDING_DIM as i64] || values.len() != EMBEDDING_DIM {
        return Err(InferError::Runtime(format!(
            "ItemV2 模型输出维度异常: shape={shape:?}, 元素数={}，期望 [1, {EMBEDDING_DIM}]",
            values.len()
        )));
    }
    let mut embedding = values.to_vec();
    normalize_l2(&mut embedding)
        .map_err(|message| InferError::Runtime(format!("ItemV2 模型输出无效: {message}")))?;
    Ok(embedding)
}

fn best_match(
    embedding: &[f32],
    prototypes: &[IconPrototype],
    visual_id: u64,
) -> Result<Option<ArtifactIconMatch>, InferError> {
    let mut best: Option<(&IconPrototype, f64)> = None;
    for prototype in prototypes {
        let score = cosine_similarity(embedding, &prototype.embedding);
        if !score.is_finite() {
            return Err(InferError::Runtime(format!(
                "ItemV2 原型 {} 的余弦分数不是有限值",
                prototype.name
            )));
        }
        if best
            .as_ref()
            .is_none_or(|(_, best_score)| score > *best_score)
        {
            best = Some((prototype, score));
        }
    }
    let Some((prototype, score)) = best else {
        return Ok(None);
    };
    if score < MATCH_THRESHOLD {
        return Ok(None);
    }
    Ok(Some(ArtifactIconMatch {
        name: prototype.name.clone(),
        quality_level: prototype.quality_level,
        score: score as f32,
        visual_id,
    }))
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(0.0_f64, |sum, (&a, &b)| sum + f64::from(a) * f64::from(b))
}

fn normalize_l2(vector: &mut [f32]) -> Result<(), String> {
    if vector.is_empty() {
        return Err("向量为空".into());
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("向量包含 NaN 或无穷值".into());
    }
    let norm2 = vector
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    if !norm2.is_finite() {
        return Err("向量 L2 范数不是有限值".into());
    }
    let norm = norm2.sqrt();
    if norm <= 1e-12 {
        return Err("向量 L2 范数接近 0".into());
    }
    for value in &mut *vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("归一化后包含 NaN 或无穷值".into());
    }
    Ok(())
}

fn parse_prototypes(csv: &str) -> Result<Vec<IconPrototype>, InferError> {
    let mut records = parse_csv(csv)?
        .into_iter()
        .filter(|(_, fields)| !(fields.len() == 1 && fields[0].is_empty()));
    let (header_line, headers) = records
        .next()
        .ok_or_else(|| InferError::Runtime("ItemV2 CSV 为空或缺少表头行".into()))?;
    let column = |name: &str| -> Result<usize, InferError> {
        let matches: Vec<_> = headers
            .iter()
            .enumerate()
            .filter(|(_, header)| header.trim().eq_ignore_ascii_case(name))
            .map(|(index, _)| index)
            .collect();
        if matches.len() != 1 {
            return Err(InferError::Runtime(format!(
                "ItemV2 CSV 第 {header_line} 行的字段 {name:?} 必须恰好出现一次，实际 {} 次",
                matches.len()
            )));
        }
        Ok(matches[0])
    };
    let name_index = column("item_name")?;
    let class_index = column("item_class_id")?;
    let quality_index = column("quality_level")?;
    let embedding_index = column("embedding")?;
    let required_fields = name_index
        .max(class_index)
        .max(quality_index)
        .max(embedding_index);

    let mut prototypes = Vec::new();
    for (line, fields) in records {
        if fields.len() <= required_fields {
            return Err(InferError::Runtime(format!(
                "ItemV2 CSV 第 {line} 行字段数量不足: {}，至少需要 {}",
                fields.len(),
                required_fields + 1
            )));
        }
        let name = fields[name_index].trim();
        if name.is_empty() {
            return Err(InferError::Runtime(format!(
                "ItemV2 CSV 第 {line} 行的 item_name 为空"
            )));
        }
        let class_id = fields[class_index].trim();
        let is_relic = class_id
            .get(..6)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("relic:"));
        let quality_level = if is_relic {
            let raw = fields[quality_index].trim();
            let quality = raw
                .parse::<u8>()
                .ok()
                .filter(|value| (1..=5).contains(value));
            quality.ok_or_else(|| {
                InferError::Runtime(format!(
                    "ItemV2 CSV 第 {line} 行圣遗物 {name:?} 的 quality_level 必须为 1..5，实际 {raw:?}"
                ))
            })?
        } else {
            0
        };
        let encoded = fields[embedding_index].trim();
        let bytes = decode_base64(encoded).map_err(|message| {
            InferError::Runtime(format!(
                "ItemV2 CSV 第 {line} 行 {name:?} 的 embedding 无效: {message}"
            ))
        })?;
        if bytes.len() != EMBEDDING_DIM * 4 {
            return Err(InferError::Runtime(format!(
                "ItemV2 CSV 第 {line} 行 {name:?} 的 embedding 维度异常: {}，期望 {EMBEDDING_DIM}",
                bytes.len() / 4
            )));
        }
        let mut embedding = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        normalize_l2(&mut embedding).map_err(|message| {
            InferError::Runtime(format!(
                "ItemV2 CSV 第 {line} 行原型 {name:?} 无效: {message}"
            ))
        })?;
        prototypes.push(IconPrototype {
            name: name.to_string(),
            quality_level,
            embedding,
        });
    }
    if prototypes.is_empty() {
        return Err(InferError::Runtime("ItemV2 CSV 未包含有效原型".into()));
    }
    Ok(prototypes)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CsvState {
    Start,
    Unquoted,
    Quoted,
    AfterQuote,
}

pub(crate) fn parse_csv(input: &str) -> Result<Vec<(usize, Vec<String>)>, InferError> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut records = Vec::new();
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut state = CsvState::Start;
    let mut chars = input.chars().peekable();
    let mut line = 1usize;
    let mut record_line = 1usize;

    while let Some(ch) = chars.next() {
        match state {
            CsvState::Quoted => match ch {
                '"' => state = CsvState::AfterQuote,
                '\n' => {
                    field.push(ch);
                    line += 1;
                }
                _ => field.push(ch),
            },
            CsvState::AfterQuote => match ch {
                '"' => {
                    field.push('"');
                    state = CsvState::Quoted;
                }
                ',' => {
                    fields.push(std::mem::take(&mut field));
                    state = CsvState::Start;
                }
                '\n' | '\r' => {
                    fields.push(std::mem::take(&mut field));
                    records.push((record_line, std::mem::take(&mut fields)));
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    line += 1;
                    record_line = line;
                    state = CsvState::Start;
                }
                _ => {
                    return Err(InferError::Runtime(format!(
                        "ItemV2 CSV 第 {line} 行引号字段结束后出现非法字符 {ch:?}"
                    )));
                }
            },
            CsvState::Start | CsvState::Unquoted => match ch {
                '"' if state == CsvState::Start => state = CsvState::Quoted,
                '"' => {
                    return Err(InferError::Runtime(format!(
                        "ItemV2 CSV 第 {line} 行未加引号的字段中包含引号"
                    )));
                }
                ',' => {
                    fields.push(std::mem::take(&mut field));
                    state = CsvState::Start;
                }
                '\n' | '\r' => {
                    fields.push(std::mem::take(&mut field));
                    records.push((record_line, std::mem::take(&mut fields)));
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    line += 1;
                    record_line = line;
                    state = CsvState::Start;
                }
                _ => {
                    field.push(ch);
                    state = CsvState::Unquoted;
                }
            },
        }
    }
    if state == CsvState::Quoted {
        return Err(InferError::Runtime(format!(
            "ItemV2 CSV 第 {record_line} 行存在未闭合的引号字段"
        )));
    }
    if !fields.is_empty() || !field.is_empty() || state == CsvState::AfterQuote {
        fields.push(field);
        records.push((record_line, fields));
    }
    Ok(records)
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let bytes = input.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err("base64 长度必须为非零的 4 的倍数".into());
    }
    let mut output = Vec::with_capacity(bytes.len() / 4 * 3);
    let (quartets, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    for (index, quartet) in quartets.iter().enumerate() {
        let error = || format!("base64 第 {} 个四字节组无效", index + 1);
        let a = base64_value(quartet[0]).ok_or_else(error)?;
        let b = base64_value(quartet[1]).ok_or_else(error)?;
        let c_padding = quartet[2] == b'=';
        let d_padding = quartet[3] == b'=';
        output.push((a << 2) | (b >> 4));
        if c_padding {
            if !d_padding || b & 0x0f != 0 {
                return Err(error());
            }
            continue;
        }
        let c = base64_value(quartet[2]).ok_or_else(error)?;
        output.push((b << 4) | (c >> 2));
        if d_padding {
            if c & 0x03 != 0 {
                return Err(error());
            }
            continue;
        }
        let d = base64_value(quartet[3]).ok_or_else(error)?;
        output.push((c << 6) | d);
    }
    Ok(output)
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "item_name,item_class_id,quality_level,embedding";

    fn encode_floats(values: &[f32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut output = String::new();
        for chunk in bytes.chunks(3) {
            let a = chunk[0];
            let b = chunk.get(1).copied().unwrap_or(0);
            let c = chunk.get(2).copied().unwrap_or(0);
            output.push(ALPHABET[(a >> 2) as usize] as char);
            output.push(ALPHABET[(((a & 3) << 4) | (b >> 4)) as usize] as char);
            output.push(if chunk.len() > 1 {
                ALPHABET[(((b & 15) << 2) | (c >> 6)) as usize] as char
            } else {
                '='
            });
            output.push(if chunk.len() > 2 {
                ALPHABET[(c & 63) as usize] as char
            } else {
                '='
            });
        }
        output
    }

    fn unit_embedding() -> Vec<f32> {
        let mut embedding = vec![0.0; EMBEDDING_DIM];
        embedding[0] = 1.0;
        embedding
    }

    #[test]
    fn l2_normalization_validates_and_normalizes() {
        let mut vector = vec![3.0, 4.0, 0.0];
        normalize_l2(&mut vector).unwrap();
        assert_eq!(vector, vec![0.6, 0.8, 0.0]);
        assert!((cosine_similarity(&vector, &vector) - 1.0).abs() < 1e-6);

        for invalid in [
            vec![0.0, 0.0],
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, 1.0],
        ] {
            assert!(normalize_l2(&mut invalid.clone()).is_err());
        }
    }

    #[test]
    fn csv_handles_bom_crlf_quotes_commas_and_newlines() {
        let embedding = encode_floats(&unit_embedding());
        let csv = format!(
            "\u{feff}\r\n\" ITEM_NAME \",\"item_class_id\",quality_level,embedding\r\n\"花, \"\"羽毛\"\"\n套装\",\"ReLiC:flower\",5,\"{embedding}\"\r\n"
        );
        let prototypes = parse_prototypes(&csv).unwrap();
        assert_eq!(prototypes.len(), 1);
        assert_eq!(prototypes[0].name, "花, \"羽毛\"\n套装");
        assert_eq!(prototypes[0].quality_level, 5);
        assert_eq!(prototypes[0].embedding, unit_embedding());
    }

    #[test]
    fn csv_rejects_empty_missing_headers_and_unclosed_quotes() {
        assert!(parse_prototypes("").is_err());
        assert!(parse_prototypes("item_name,item_class_id\n").is_err());
        let embedding = encode_floats(&unit_embedding());
        assert!(
            parse_prototypes(&format!(
                "{HEADER}\n\"unterminated,relic:flower,5,{embedding}"
            ))
            .is_err()
        );
    }

    #[test]
    fn matching_uses_inclusive_cosine_threshold() {
        let query = unit_embedding();
        let mut at_threshold = unit_embedding();
        at_threshold[0] = MATCH_THRESHOLD as f32;
        at_threshold[1] = (1.0 - MATCH_THRESHOLD * MATCH_THRESHOLD).sqrt() as f32;
        let prototype = IconPrototype {
            name: "threshold".into(),
            quality_level: 4,
            embedding: at_threshold,
        };
        let matched = best_match(&query, std::slice::from_ref(&prototype), 42)
            .unwrap()
            .unwrap();
        assert_eq!(matched.name, "threshold");
        assert_eq!(matched.quality_level, 4);
        assert_eq!(matched.visual_id, 42);
        assert!(
            (f64::from(matched.score) - MATCH_THRESHOLD).abs() < 1e-6,
            "score={} threshold={MATCH_THRESHOLD}",
            matched.score
        );

        let mut below_threshold = unit_embedding();
        below_threshold[0] = 0.74;
        below_threshold[1] = (1.0 - 0.74_f64.powi(2)).sqrt() as f32;
        let below_prototype = IconPrototype {
            name: "below".into(),
            quality_level: 4,
            embedding: below_threshold,
        };
        assert!(best_match(&query, &[below_prototype], 0).unwrap().is_none());
    }

    #[test]
    fn overall_non_relic_match_is_returned_and_blocks_lower_scoring_relics() {
        let exact = encode_floats(&unit_embedding());
        let mut weaker = unit_embedding();
        weaker[0] = 0.8;
        weaker[1] = 0.6;
        let weaker = encode_floats(&weaker);
        let csv =
            format!("{HEADER}\nweapon,weapon:sword,5,{exact}\nflower,relic:flower,5,{weaker}");
        let prototypes = parse_prototypes(&csv).unwrap();
        assert_eq!(prototypes.len(), 2);
        assert_eq!(prototypes[0].quality_level, 0);
        assert_eq!(prototypes[1].quality_level, 5);
        let matched = best_match(&unit_embedding(), &prototypes, 7)
            .unwrap()
            .unwrap();
        assert_eq!(matched.name, "weapon");
        assert_eq!(matched.quality_level, 0);
        assert!(matched.score > 0.99);
        assert_eq!(matched.visual_id, 7);

        let csv =
            format!("{HEADER}\nweapon,weapon:sword,5,{weaker}\nflower,relic:flower,5,{exact}");
        let matched = best_match(&unit_embedding(), &parse_prototypes(&csv).unwrap(), 7)
            .unwrap()
            .unwrap();
        assert_eq!(matched.name, "flower");
        assert_eq!(matched.visual_id, 7);

        for quality in ["0", "6", "-1", "", "five"] {
            assert!(
                parse_prototypes(&format!("{HEADER}\nflower,relic:flower,{quality},{exact}"))
                    .is_err()
            );
        }
    }

    #[test]
    fn base64_is_validated_and_requires_256_little_endian_floats() {
        assert_eq!(decode_base64("AA==").unwrap(), [0]);
        assert_eq!(decode_base64("/w==").unwrap(), [255]);
        for invalid in ["", "AAA", "AA=A", "AB==", "AAB=", "AA==\n", "!!!!"] {
            assert!(decode_base64(invalid).is_err(), "{invalid:?}");
        }

        for values in [vec![1.0; 255], vec![1.0; 257]] {
            let embedding = encode_floats(&values);
            assert!(
                parse_prototypes(&format!("{HEADER}\nflower,relic:flower,5,{embedding}")).is_err()
            );
        }
        let little_endian = encode_floats(&[1.0, -2.0]);
        assert_eq!(
            decode_base64(&little_endian).unwrap()[..4],
            1.0f32.to_le_bytes()
        );
        assert_eq!(
            decode_base64(&little_endian).unwrap()[4..],
            (-2.0f32).to_le_bytes()
        );
    }

    #[test]
    fn model_output_requires_exact_shape_finite_values_and_nonzero_norm() {
        let query = unit_embedding();
        let normalized = normalize_model_output(&[1, EMBEDDING_DIM as i64], &query).unwrap();
        assert_eq!(normalized, query);
        for shape in [vec![1, 255], vec![256], vec![2, 256]] {
            assert!(normalize_model_output(&shape, &query).is_err());
        }
        let mut nonfinite = query.clone();
        nonfinite[0] = f32::NAN;
        assert!(normalize_model_output(&[1, EMBEDDING_DIM as i64], &nonfinite).is_err());
        assert!(
            normalize_model_output(&[1, EMBEDDING_DIM as i64], &vec![0.0; EMBEDDING_DIM]).is_err()
        );
    }

    #[test]
    fn preprocessing_converts_bgr_to_rgb_and_normalizes() {
        let mut bgr = vec![0u8; INPUT_SIZE * INPUT_SIZE * 3];
        bgr[..3].copy_from_slice(&[0, 128, 255]);
        let input = bgr_to_rgb_nchw(&bgr);
        let plane = INPUT_SIZE * INPUT_SIZE;
        assert_eq!(input[0], 1.0);
        assert!((input[plane] - 1.0 / 255.0).abs() < 1e-6);
        assert_eq!(input[2 * plane], -1.0);
        assert_eq!(input[1], -1.0);
        assert_eq!(input.len(), 3 * plane);
    }

    /// Requires BGI_ITEM_ICON_MODEL and BGI_ITEM_ICON_CSV. BGI_ITEM_ICON_ORT_DLL is
    /// optional. Supplying both BGI_ITEM_ICON_TEST_IMAGE and BGI_ITEM_ICON_EXPECTED_NAME
    /// additionally checks recognition of a known 125x125 relic icon.
    #[test]
    #[ignore = "requires local ItemV2 model and CSV assets"]
    fn loads_and_runs_real_item_model() {
        use opencv::imgcodecs::{IMREAD_COLOR, imread};
        use opencv::prelude::MatExprTraitConst;

        let model =
            std::env::var_os("BGI_ITEM_ICON_MODEL").expect("BGI_ITEM_ICON_MODEL is required");
        let csv = std::env::var_os("BGI_ITEM_ICON_CSV").expect("BGI_ITEM_ICON_CSV is required");
        let dll = std::env::var_os("BGI_ITEM_ICON_ORT_DLL").map(PathBuf::from);
        let mut recognizer = ItemIconRecognizer::from_paths(model, csv, dll).unwrap();

        // Exercise the real output shape and finite-value checks without needing a fixture.
        let blank = Mat::zeros(INPUT_SIZE as i32, INPUT_SIZE as i32, CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        if let Some(matched) = recognizer.recognize_bgr_125(&blank).unwrap() {
            assert!((1..=5).contains(&matched.quality_level));
            assert!(matched.score >= MATCH_THRESHOLD as f32);
        }

        if let Some(image) = std::env::var_os("BGI_ITEM_ICON_TEST_IMAGE") {
            let expected = std::env::var("BGI_ITEM_ICON_EXPECTED_NAME")
                .expect("BGI_ITEM_ICON_EXPECTED_NAME is required with a test image");
            let icon = imread(PathBuf::from(image), IMREAD_COLOR).unwrap();
            assert_eq!(
                (icon.cols(), icon.rows()),
                (INPUT_SIZE as i32, INPUT_SIZE as i32)
            );
            let matched = recognizer.recognize_bgr_125(&icon).unwrap().unwrap();
            assert_eq!(matched.name, expected);
            assert!((1..=5).contains(&matched.quality_level));
            assert!(matched.score >= MATCH_THRESHOLD as f32);
        }
    }
}
