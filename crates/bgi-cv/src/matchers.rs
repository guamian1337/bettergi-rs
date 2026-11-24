use opencv::core::{
    self, CV_32F, CV_64F, CV_64FC1, CV_8UC1, Mat, MatTraitConst, MatTraitConstManual,
    NORM_L2SQR, Point, Point2f, Rect, Scalar, Size, Vector,
};
use opencv::imgproc::{self, TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED};
use opencv::prelude::MatExprTraitConst;

use crate::CvError;

/// Wraps tightly packed `BGR24` bytes directly into an OpenCV `CV_8UC3` `Mat` without color conversion.
pub fn bgr_to_mat(bgr: &[u8], width: u32, height: u32) -> Result<Mat, CvError> {
    let expected = (width as usize) * (height as usize) * 3;
    if width == 0 || height == 0 || bgr.len() < expected {
        return Err(CvError::InvalidImage(format!(
            "期望至少 {expected} 字节 BGR24 ({width}x{height})，实际 {}",
            bgr.len()
        )));
    }
    let view = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            height as i32,
            width as i32,
            core::CV_8UC3,
            bgr.as_ptr() as *mut std::ffi::c_void,
            (width as usize) * 3,
        )
    }?;
    Ok(view.try_clone()?)
}

pub fn frame_to_bgr_mat(
    bytes: &[u8],
    width: u32,
    height: u32,
    is_bgr24: bool,
) -> Result<Mat, CvError> {
    if is_bgr24 {
        bgr_to_mat(bytes, width, height)
    } else {
        bgra_to_bgr_mat(bytes, width, height)
    }
}

pub fn bgra_to_bgr_mat(bgra: &[u8], width: u32, height: u32) -> Result<Mat, CvError> {
    let expected = (width as usize) * (height as usize) * 4;
    if width == 0 || height == 0 || bgra.len() < expected {
        return Err(CvError::InvalidImage(format!(
            "期望至少 {expected} 字节 BGRA ({width}x{height})，实际 {}",
            bgra.len()
        )));
    }
    let src_view = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            height as i32,
            width as i32,
            core::CV_8UC4,
            bgra.as_ptr() as *mut std::ffi::c_void,
            (width as usize) * 4,
        )
    }?;
    let mut bgr = Mat::default();
    imgproc::cvt_color(&src_view, &mut bgr, imgproc::COLOR_BGRA2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(bgr)
}

pub fn crop_minimap_rect(width: i32, height: i32) -> Rect {
    let scale = (width as f64 / 1920.0).min(height as f64 / 1080.0);
    let x = (62.0 * scale).round() as i32;
    let y = (19.0 * scale).round() as i32;
    let w = ((212.0 * scale).round() as i32).min(width - x).max(1);
    let h = ((212.0 * scale).round() as i32).min(height - y).max(1);
    Rect::new(x.max(0), y.max(0), w, h)
}

/// Crops the minimap region and resizes it to the canonical 1080p minimap size (`212x212`) when needed.
pub fn extract_minimap_bgr(full_bgr: &Mat) -> Result<Mat, CvError> {
    let rect = crop_minimap_rect(full_bgr.cols(), full_bgr.rows());
    let roi = Mat::roi(full_bgr, rect)?;
    if roi.cols() == 212 && roi.rows() == 212 {
        Ok(roi.try_clone()?)
    } else {
        let mut out = Mat::default();
        let interp = if roi.cols() > 212 {
            imgproc::INTER_AREA
        } else {
            imgproc::INTER_CUBIC
        };
        imgproc::resize(&roi, &mut out, Size::new(212, 212), 0.0, 0.0, interp)?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct TemplateMatchNormalizer {
    pub mode: i32,
    pub value: f64,
    pub sign: f64,
    best: f64,
    worst: f64,
}

impl TemplateMatchNormalizer {
    pub fn from_template(template: &Mat, mask: Option<&Mat>, mode: i32) -> Result<Self, CvError> {
        let (best, worst) = match mode {
            TM_SQDIFF => sq_diff_match_value(template, mask)?,
            TM_SQDIFF_NORMED => (0.0, 1.0),
            TM_CCORR => ccorr_match_value(template, mask)?,
            TM_CCORR_NORMED => (1.0, 0.0),
            TM_CCOEFF => ccoeff_match_value(template, mask)?,
            TM_CCOEFF_NORMED => (1.0, -1.0),
            _ => return Err(CvError::OpenCv(format!("不支持的模板匹配模式: {mode}"))),
        };
        Ok(Self::new(best, worst, mode))
    }

    pub fn from_templates(
        templates: &[Mat],
        mask: Option<&Mat>,
        mode: i32,
        channels: Option<&[usize]>,
    ) -> Result<Self, CvError> {
        let mut best_sum = 0.0;
        let mut worst_sum = 0.0;
        let selected: Vec<&Mat> = match channels {
            Some(chs) => chs.iter().filter_map(|&idx| templates.get(idx)).collect(),
            None => templates.iter().collect(),
        };
        match mode {
            TM_SQDIFF => {
                for t in selected {
                    let (b, w) = sq_diff_match_value(t, mask)?;
                    best_sum += b;
                    worst_sum += w;
                }
            }
            TM_SQDIFF_NORMED => (best_sum, worst_sum) = (0.0, 1.0),
            TM_CCORR => {
                for t in selected {
                    let (b, w) = ccorr_match_value(t, mask)?;
                    best_sum += b;
                    worst_sum += w;
                }
            }
            TM_CCORR_NORMED => (best_sum, worst_sum) = (1.0, 0.0),
            TM_CCOEFF => {
                for t in selected {
                    let (b, w) = ccoeff_match_value(t, mask)?;
                    best_sum += b;
                    worst_sum += w;
                }
            }
            TM_CCOEFF_NORMED => (best_sum, worst_sum) = (1.0, -1.0),
            _ => return Err(CvError::OpenCv(format!("不支持的模板匹配模式: {mode}"))),
        }
        Ok(Self::new(best_sum, worst_sum, mode))
    }

    pub fn new(best: f64, worst: f64, mode: i32) -> Self {
        let sign = if mode == TM_SQDIFF || mode == TM_SQDIFF_NORMED {
            -1.0
        } else {
            1.0
        };
        let value = if sign > 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
        Self {
            mode,
            value,
            sign,
            best,
            worst,
        }
    }

    pub fn update(&mut self, val: f64) -> bool {
        if !val.is_finite() {
            return false;
        }
        if val > self.best.max(self.worst) || val < self.best.min(self.worst) {
            return false;
        }
        if self.sign * (self.value - val) >= 0.0 {
            return false;
        }
        self.value = val;
        true
    }

    pub fn confidence(&self) -> f64 {
        if (self.best - self.worst).abs() < 1e-12 {
            0.0
        } else {
            (self.value - self.worst) / (self.best - self.worst)
        }
    }
}

fn sq_diff_match_value(template: &Mat, mask: Option<&Mat>) -> Result<(f64, f64), CvError> {
    let mut inverted = Mat::default();
    core::subtract(&Scalar::all(255.0), template, &mut inverted, &core::no_array(), -1)?;
    let mut worst_mat = Mat::default();
    core::max(template, &inverted, &mut worst_mat)?;
    let worst = match mask {
        Some(m) => core::norm(&worst_mat, NORM_L2SQR, m)?,
        None => core::norm(&worst_mat, NORM_L2SQR, &core::no_array())?,
    };
    Ok((0.0, worst))
}

fn ccorr_match_value(template: &Mat, mask: Option<&Mat>) -> Result<(f64, f64), CvError> {
    let best = match mask {
        Some(m) => core::norm(template, NORM_L2SQR, m)?,
        None => core::norm(template, NORM_L2SQR, &core::no_array())?,
    };
    Ok((best, 0.0))
}

fn ccoeff_match_value(template: &Mat, mask: Option<&Mat>) -> Result<(f64, f64), CvError> {
    let mut res = Mat::default();
    match mask {
        Some(m) => imgproc::match_template(template, template, &mut res, TM_CCOEFF, m)?,
        None => imgproc::match_template(template, template, &mut res, TM_CCOEFF, &core::no_array())?,
    }
    let best = *res.at_2d::<f32>(0, 0)? as f64;
    Ok((best, -best))
}

/// Fast weighted-mask SqDiff matcher mirroring `BetterGenshinImpact`'s `FastSqDiffMatcher.cs`.
pub struct FastSqDiffMatcher {
    pub source: Vec<Mat>,
    source_sq: Vec<Mat>,
}

impl FastSqDiffMatcher {
    pub fn new(source: &Mat, template_size: Size) -> Result<Self, CvError> {
        if source.empty() {
            return Err(CvError::InvalidImage("源图像为空".into()));
        }
        if template_size.width > source.cols() || template_size.height > source.rows() {
            return Err(CvError::InvalidImage("模板图尺寸超过源图像尺寸".into()));
        }

        let mut split_src = Vector::<Mat>::new();
        core::split(source, &mut split_src)?;

        let mut src_f = Mat::default();
        source.convert_to(&mut src_f, CV_32F, 1.0, 0.0)?;
        let mut src_sq_full = Mat::default();
        core::multiply(&src_f, &src_f, &mut src_sq_full, 1.0, -1)?;

        let mut split_sq = Vector::<Mat>::new();
        core::split(&src_sq_full, &mut split_sq)?;

        Ok(Self {
            source: split_src.into_iter().collect(),
            source_sq: split_sq.into_iter().collect(),
        })
    }

    pub fn width(&self) -> i32 {
        self.source.first().map_or(0, |m| m.cols())
    }

    pub fn height(&self) -> i32 {
        self.source.first().map_or(0, |m| m.rows())
    }

    pub fn pre_process(template: &Mat, mask: &Mat) -> Result<(Vec<Mat>, Mat), CvError> {
        if mask.typ() != CV_8UC1 {
            return Err(CvError::InvalidImage("遮罩必须为 CV_8UC1".into()));
        }
        if template.size()? != mask.size()? {
            return Err(CvError::InvalidImage("模板图与遮罩尺寸不匹配".into()));
        }

        let mut mask_norm = Mat::default();
        mask.convert_to(&mut mask_norm, CV_32F, 1.0 / 255.0, 0.0)?;

        let mut masked_tpl = Mat::new_size_with_default(template.size()?, template.typ(), Scalar::all(0.0))?;
        template.copy_to_masked(&mut masked_tpl, mask)?;

        let mut split = Vector::<Mat>::new();
        core::split(&masked_tpl, &mut split)?;
        Ok((split.into_iter().collect(), mask_norm))
    }

    pub fn get_tpl_sum_sq(masked_templates: &[Mat], channels: Option<&[usize]>) -> Result<f64, CvError> {
        let mut sum = 0.0;
        match channels {
            Some(chs) => {
                for &idx in chs {
                    if let Some(m) = masked_templates.get(idx) {
                        sum += core::norm(m, NORM_L2SQR, &core::no_array())?;
                    }
                }
            }
            None => {
                for m in masked_templates {
                    sum += core::norm(m, NORM_L2SQR, &core::no_array())?;
                }
            }
        }
        Ok(sum)
    }

    pub fn match_global(&self, masked_templates: &[Mat], mask_f: &Mat) -> Result<(Point, f64), CvError> {
        self.match_slices(&self.source, &self.source_sq, masked_templates, mask_f)
    }

    pub fn match_roi(
        &self,
        masked_templates: &[Mat],
        mask_f: &Mat,
        rect: Rect,
        channels: Option<&[usize]>,
    ) -> Result<(Point, f64), CvError> {
        let mut src_views = Vec::with_capacity(self.source.len());
        let mut sq_views = Vec::with_capacity(self.source_sq.len());
        let mut tpl_views = Vec::with_capacity(masked_templates.len());

        let indices: Vec<usize> = match channels {
            Some(chs) => chs.iter().copied().filter(|&i| i < self.source.len()).collect(),
            None => (0..self.source.len()).collect(),
        };

        for idx in indices {
            src_views.push(Mat::roi(&self.source[idx], rect)?.clone_pointee());
            sq_views.push(Mat::roi(&self.source_sq[idx], rect)?.clone_pointee());
            tpl_views.push(masked_templates[idx].clone());
        }

        self.match_slices(&src_views, &sq_views, &tpl_views, mask_f)
    }

    fn match_slices(
        &self,
        source: &[Mat],
        source_sq: &[Mat],
        masked_templates: &[Mat],
        mask_f: &Mat,
    ) -> Result<(Point, f64), CvError> {
        let n = source.len();
        if masked_templates.len() != n || n == 0 {
            return Err(CvError::InvalidImage(format!(
                "通道数不匹配: source={n}, tpl={}",
                masked_templates.len()
            )));
        }

        let mut cross_corr = Mat::default();
        let mut temp = Mat::default();
        imgproc::match_template(&source[0], &masked_templates[0], &mut cross_corr, TM_CCORR, &core::no_array())?;
        for i in 1..n {
            imgproc::match_template(&source[i], &masked_templates[i], &mut temp, TM_CCORR, &core::no_array())?;
            let mut sum = Mat::default();
            core::add(&cross_corr, &temp, &mut sum, &core::no_array(), -1)?;
            cross_corr = sum;
        }

        let mut scaled = Mat::default();
        cross_corr.convert_to(&mut scaled, -1, -2.0, 0.0)?;
        cross_corr = scaled;

        for sq in source_sq {
            imgproc::match_template(sq, mask_f, &mut temp, TM_CCORR, &core::no_array())?;
            let mut sum = Mat::default();
            core::add(&cross_corr, &temp, &mut sum, &core::no_array(), -1)?;
            cross_corr = sum;
        }

        let mut min_val = 0.0;
        let mut min_loc = Point::default();
        core::min_max_loc(
            &cross_corr,
            Some(&mut min_val),
            None,
            Some(&mut min_loc),
            None,
            &core::no_array(),
        )?;

        Ok((min_loc, min_val))
    }
}

/// 3x3 quadratic subpixel peak/valley fitting mirroring `SubPixMatch.cs`.
pub fn subpix_fit(src: &Mat, loc: Point) -> Result<Point2f, CvError> {
    if src.empty() || src.cols() < 3 || src.rows() < 3 {
        return Err(CvError::InvalidImage("SubPixMatch 输入矩阵为空或过小".into()));
    }
    if loc.x < 0 || loc.y < 0 || loc.x >= src.cols() || loc.y >= src.rows() {
        return Err(CvError::InvalidImage("SubPixMatch 输入坐标越界".into()));
    }

    let clamped_x = loc.x.clamp(1, src.cols() - 2);
    let clamped_y = loc.y.clamp(1, src.rows() - 2);
    let roi = Mat::roi(src, Rect::new(clamped_x - 1, clamped_y - 1, 3, 3))?;
    let cloned = roi.try_clone()?;
    let reshaped = cloned.reshape(0, 9)?;
    let mut neigh_f64 = Mat::default();
    reshaped.convert_to(&mut neigh_f64, CV_64FC1, 1.0, 0.0)?;

    #[rustfmt::skip]
    const FEATURES_DATA: [f64; 54] = [
        1.0/6.0, -1.0/3.0, 1.0/6.0, 1.0/6.0, -1.0/3.0, 1.0/6.0, 1.0/6.0, -1.0/3.0, 1.0/6.0,
        1.0/6.0,  1.0/6.0, 1.0/6.0,-1.0/3.0, -1.0/3.0,-1.0/3.0, 1.0/6.0,  1.0/6.0, 1.0/6.0,
        1.0/4.0,  0.0,    -1.0/4.0, 0.0,      0.0,     0.0,    -1.0/4.0,  0.0,     1.0/4.0,
       -1.0/6.0,  0.0,     1.0/6.0,-1.0/6.0,  0.0,     1.0/6.0,-1.0/6.0,  0.0,     1.0/6.0,
       -1.0/6.0, -1.0/6.0,-1.0/6.0, 0.0,      0.0,     0.0,     1.0/6.0,  1.0/6.0, 1.0/6.0,
       -1.0/9.0,  2.0/9.0,-1.0/9.0, 2.0/9.0,  5.0/9.0, 2.0/9.0,-1.0/9.0,  2.0/9.0,-1.0/9.0,
    ];
    let features = Mat::from_slice_2d(&[
        &FEATURES_DATA[0..9],
        &FEATURES_DATA[9..18],
        &FEATURES_DATA[18..27],
        &FEATURES_DATA[27..36],
        &FEATURES_DATA[36..45],
        &FEATURES_DATA[45..54],
    ])?;
    let mut coeff_mat = Mat::zeros(6, 1, CV_64F)?.to_mat()?;
    core::gemm(&features, &neigh_f64, 1.0, &core::no_array(), 0.0, &mut coeff_mat, 0)?;

    let c = coeff_mat.data_typed::<f64>()?;
    let discriminant = c[2] * c[2] - 4.0 * c[0] * c[1];
    if discriminant.abs() < 1e-20 {
        return Ok(Point2f::new(loc.x as f32, loc.y as f32));
    }

    let offset_x: f64 = ((2.0 * c[1] * c[3] - c[2] * c[4]) / discriminant).clamp(-1.0, 1.0);
    let offset_y: f64 = ((2.0 * c[0] * c[4] - c[2] * c[3]) / discriminant).clamp(-1.0, 1.0);
    Ok(Point2f::new(
        (offset_x + clamped_x as f64) as f32,
        (offset_y + clamped_y as f64) as f32,
    ))
}

pub fn match_template_subpix(
    big_mat: &Mat,
    small_mat: &Mat,
    mode: i32,
    mask: Option<&Mat>,
) -> Result<(Point2f, f64), CvError> {
    let mut result = Mat::default();
    match mask {
        Some(m) => imgproc::match_template(big_mat, small_mat, &mut result, mode, m)?,
        None => imgproc::match_template(big_mat, small_mat, &mut result, mode, &core::no_array())?,
    }

    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    core::min_max_loc(
        &result,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &core::no_array(),
    )?;

    let (best_loc, best_val) = if mode == TM_SQDIFF || mode == TM_SQDIFF_NORMED {
        (min_loc, min_val)
    } else {
        (max_loc, max_val)
    };

    let sub_loc = if result.cols() >= 3 && result.rows() >= 3 {
        subpix_fit(&result, best_loc)?
    } else {
        Point2f::new(best_loc.x as f32, best_loc.y as f32)
    };
    Ok((sub_loc, best_val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sq_diff_locates_embedded_patch() {
        let mut big = Mat::zeros(120, 120, core::CV_8UC3).unwrap().to_mat().unwrap();
        let mut tpl = Mat::zeros(52, 52, core::CV_8UC3).unwrap().to_mat().unwrap();
        imgproc::rectangle(
            &mut tpl,
            Rect::new(10, 10, 20, 20),
            Scalar::new(200.0, 120.0, 60.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )
        .unwrap();

        let mut dst_roi = Mat::roi_mut(&mut big, Rect::new(34, 41, 52, 52)).unwrap();
        tpl.copy_to(&mut dst_roi).unwrap();

        let mask = Mat::new_rows_cols_with_default(52, 52, CV_8UC1, Scalar::all(255.0)).unwrap();
        let (masked, mask_f) = FastSqDiffMatcher::pre_process(&tpl, &mask).unwrap();
        let matcher = FastSqDiffMatcher::new(&big, Size::new(52, 52)).unwrap();
        let (loc, val) = matcher.match_global(&masked, &mask_f).unwrap();
        let sum_sq = FastSqDiffMatcher::get_tpl_sum_sq(&masked, None).unwrap();

        assert_eq!(loc, Point::new(34, 41));
        assert!((val + sum_sq).abs() / sum_sq.max(1.0) < 1e-5);
    }

    #[test]
    fn bgr24_mat_matches_bgra_conversion() {
        let bgra = vec![
            11, 22, 33, 255, 44, 55, 66, 128, 77, 88, 99, 0, 10, 20, 30, 200,
        ];
        let bgr = vec![
            11, 22, 33, 44, 55, 66, 77, 88, 99, 10, 20, 30,
        ];
        let m_from_bgra = frame_to_bgr_mat(&bgra, 2, 2, false).unwrap();
        let m_from_bgr = frame_to_bgr_mat(&bgr, 2, 2, true).unwrap();
        assert_eq!(m_from_bgra.data_bytes().unwrap(), m_from_bgr.data_bytes().unwrap());
    }
}
