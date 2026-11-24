use opencv::core::{
    self, BORDER_CONSTANT, CMP_EQ, CMP_LT, CV_32F, CV_32FC1, CV_8UC1, CV_8UC3, Mat,
    MatTraitConst, MatTraitConstManual, Point, Point2f, Rect, Scalar, Size, Vector,
};
use opencv::imgproc::{
    self, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, COLOR_BGR2HLS_FULL, COLOR_BGR2HSV, COLOR_GRAY2BGR,
    INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, MORPH_CLOSE, MORPH_ELLIPSE, MORPH_OPEN, RETR_EXTERNAL,
    THRESH_BINARY, WARP_POLAR_LINEAR,
};
use opencv::prelude::{MatExprTraitConst, MatTrait};

use crate::CvError;

pub const MINIMAP_SIZE: i32 = 156;
const TPL_OUT_RAD: f32 = 78.0;
const TPL_INN_RAD: f32 = 19.0;
const R_LENGTH: usize = 60;
const THETA_LENGTH: usize = 360;
const F_LENGTH: usize = 256;
const ORIENT_SCALE: usize = 2;
const PEAK_WIDTH: i32 = (THETA_LENGTH as i32 / 4) * (ORIENT_SCALE as i32) + 1;

const ALPHA_PARAMS1: [f32; 13] = [
    18.632, 20.157, 24.093, 34.617, 38.566, 41.94, 47.654, 51.087, 58.561, 63.925, 67.759, 71.77,
    75.214,
];

fn linear_spaced(a: f32, b: f32, n: usize, endpoint: bool) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let denom = if endpoint && n > 1 { (n - 1) as f32 } else { n as f32 };
    (0..n).map(|i| a + (b - a) * (i as f32) / denom).collect()
}

fn alpha1_lut_value(v: f32) -> f32 {
    let idx = match ALPHA_PARAMS1.binary_search_by(|p| p.total_cmp(&v)) {
        Ok(i) | Err(i) => i,
    };
    (229 + idx).min(255) as f32
}

fn apply_mask_inplace(input_f32: &mut Mat, mask: &Mat, bkg: Scalar) -> Result<(), CvError> {
    let mut sub = Mat::default();
    core::subtract(input_f32, &bkg, &mut sub, &core::no_array(), -1)?;
    let mut div = Mat::default();
    core::divide2(&sub, mask, &mut div, 255.0, CV_32F)?;
    core::add(&div, &bkg, input_f32, &core::no_array(), -1)?;
    Ok(())
}

pub struct MaskCalculator {
    alpha_mask1: Mat,
    alpha_mask2: Mat,
    radius_u8: Mat,
    angle_u8: Mat,
    circle_mask: Mat,
    kernel5: Mat,
    out_mat_f: Mat,
    out_mask: Mat,
}

impl MaskCalculator {
    pub fn new() -> Result<Self, CvError> {
        let x_arr = linear_spaced(-(MINIMAP_SIZE as f32) / 2.0, (MINIMAP_SIZE as f32) / 2.0, MINIMAP_SIZE as usize, false);
        let x_base = Mat::from_slice(&x_arr)?;
        let x_row = x_base.reshape(1, 1)?;
        let x_mat = core::repeat(&x_row, MINIMAP_SIZE, 1)?;
        let mut y_mat = Mat::default();
        core::transpose(&x_mat, &mut y_mat)?;
        let mut radius_f = Mat::default();
        let mut angle_f = Mat::default();
        core::cart_to_polar(&x_mat, &y_mat, &mut radius_f, &mut angle_f, true)?;

        let mut radius_u8 = Mat::default();
        radius_f.convert_to(&mut radius_u8, CV_8UC1, 1.0, 0.0)?;

        let mut angle_half = Mat::default();
        core::divide2(&angle_f, &Scalar::all(2.0), &mut angle_half, 1.0, -1)?;
        let mut angle_u8 = Mat::default();
        angle_half.convert_to(&mut angle_u8, CV_8UC1, 1.0, 0.0)?;

        let lut_data1: Vec<u8> = (0..256).map(|v| alpha1_lut_value(v as f32) as u8).collect();
        let lut_data2: Vec<u8> = (0..256)
            .map(|v| (137.0 + 1.43 * (v as f64)).min(255.0) as u8)
            .collect();
        let lut1_base = Mat::from_slice(&lut_data1)?;
        let lut1 = lut1_base.reshape(1, 1)?;
        let lut2_base = Mat::from_slice(&lut_data2)?;
        let lut2 = lut2_base.reshape(1, 1)?;

        let mut alpha1_gray = Mat::default();
        let mut alpha2_gray = Mat::default();
        core::lut(&radius_u8, &lut1, &mut alpha1_gray)?;
        core::lut(&radius_u8, &lut2, &mut alpha2_gray)?;

        let mut alpha_mask1 = Mat::default();
        let mut alpha_mask2 = Mat::default();
        imgproc::cvt_color(&alpha1_gray, &mut alpha_mask1, COLOR_GRAY2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        imgproc::cvt_color(&alpha2_gray, &mut alpha_mask2, COLOR_GRAY2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        let mut circle_mask = Mat::zeros(MINIMAP_SIZE, MINIMAP_SIZE, CV_8UC1)?.to_mat()?;
        imgproc::circle(
            &mut circle_mask,
            Point::new(MINIMAP_SIZE / 2, MINIMAP_SIZE / 2),
            MINIMAP_SIZE / 2,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        let kernel5 = imgproc::get_structuring_element(MORPH_ELLIPSE, Size::new(5, 5), Point::new(-1, -1))?;
        let out_mat_f = Mat::zeros(MINIMAP_SIZE, MINIMAP_SIZE, core::CV_32FC3)?.to_mat()?;
        let out_mask = Mat::zeros(MINIMAP_SIZE, MINIMAP_SIZE, CV_8UC1)?.to_mat()?;

        Ok(Self {
            alpha_mask1,
            alpha_mask2,
            radius_u8,
            angle_u8,
            circle_mask,
            kernel5,
            out_mat_f,
            out_mask,
        })
    }

    pub fn process1(&mut self, bgr: &Mat) -> Result<(Mat, Mat), CvError> {
        let off = (bgr.cols() - MINIMAP_SIZE).max(0) / 2;
        let roi = Mat::roi(bgr, Rect::new(off, off, MINIMAP_SIZE, MINIMAP_SIZE))?;
        let src = roi.try_clone()?;
        src.convert_to(&mut self.out_mat_f, core::CV_32FC3, 1.0, 0.0)?;
        self.create_icon_mask(&src)?;
        Ok((src, self.out_mask.try_clone()?))
    }

    pub fn process2(&mut self, angle: f32) -> Result<(Mat, Mat), CvError> {
        let mut sector_mask = self.alpha_mask2.try_clone()?;
        let a = angle.round() as f64;
        imgproc::ellipse(
            &mut sector_mask,
            Point::new(MINIMAP_SIZE / 2, MINIMAP_SIZE / 2),
            Size::new(MINIMAP_SIZE, MINIMAP_SIZE),
            0.0,
            a + 45.5,
            a + 314.5,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;

        apply_mask_inplace(&mut self.out_mat_f, &sector_mask, Scalar::all(255.0))?;
        apply_mask_inplace(&mut self.out_mat_f, &self.alpha_mask1, Scalar::all(0.0))?;

        let mut out_mat = Mat::default();
        self.out_mat_f.convert_to(&mut out_mat, CV_8UC3, 1.0, 0.0)?;

        let mut inv_mask = Mat::default();
        core::bitwise_not(&self.out_mask, &mut inv_mask, &core::no_array())?;
        self.out_mask = inv_mask;

        self.create_bg_mask(&out_mat)?;

        let mut final_mask = Mat::default();
        core::bitwise_and(&self.circle_mask, &self.out_mask, &mut final_mask, &core::no_array())?;
        self.out_mask = final_mask.try_clone()?;
        Ok((out_mat, final_mask))
    }

    fn create_icon_mask(&mut self, bgr: &Mat) -> Result<(), CvError> {
        let mut ch = Vector::<Mat>::new();
        core::split(bgr, &mut ch)?;
        let (b, g, r) = (ch.get(0)?, ch.get(1)?, ch.get(2)?);

        let mut rg_max = Mat::default();
        let mut cmax = Mat::default();
        core::max(&r, &g, &mut rg_max)?;
        core::max(&rg_max, &b, &mut cmax)?;

        let mut rg_min = Mat::default();
        let mut cmin = Mat::default();
        core::min(&r, &g, &mut rg_min)?;
        core::min(&rg_min, &b, &mut cmin)?;

        let mut eq_mask = Mat::default();
        core::compare(&cmax, &cmin, &mut eq_mask, CMP_EQ)?;
        let mut range_mask = Mat::default();
        core::in_range(&cmax, &Scalar::all(50.0), &Scalar::all(127.0), &mut range_mask)?;
        core::bitwise_and(&range_mask, &eq_mask, &mut self.out_mask, &core::no_array())?;

        let mut diff = Mat::default();
        core::subtract(&cmax, &cmin, &mut diff, &core::no_array(), -1)?;
        let mut inv_cmax = Mat::default();
        core::subtract(&Scalar::all(255.0), &cmax, &mut inv_cmax, &core::no_array(), -1)?;
        let mut inv_div = Mat::default();
        core::divide2(&inv_cmax, &Scalar::all(6.0), &mut inv_div, 1.0, -1)?;
        let mut min_diff = Mat::default();
        core::min(&inv_div, &diff, &mut min_diff)?;
        let mut denom = Mat::default();
        core::add(&min_diff, &Scalar::all(10.0), &mut denom, &core::no_array(), -1)?;

        cmax.set_to(&Scalar::all(255.0), &self.out_mask)?;
        let mut ratio = Mat::default();
        core::divide2(&cmax, &denom, &mut ratio, 10.0, -1)?;

        let mut thresh = Mat::default();
        imgproc::threshold(&ratio, &mut thresh, 200.0, 255.0, THRESH_BINARY)?;
        let mut dilated = Mat::default();
        imgproc::dilate(&thresh, &mut dilated, &self.kernel5, Point::new(-1, -1), 1, BORDER_CONSTANT, Scalar::default())?;
        imgproc::morphology_ex(
            &dilated,
            &mut self.out_mask,
            MORPH_CLOSE,
            &self.kernel5,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            Scalar::default(),
        )?;
        Ok(())
    }

    fn create_bg_mask(&mut self, bgr: &Mat) -> Result<(), CvError> {
        let mut temp = Mat::default();
        core::in_range(
            bgr,
            &Scalar::new(165.0, 165.0, 55.0, 0.0),
            &Scalar::new(180.0, 180.0, 75.0, 0.0),
            &mut temp,
        )?;
        let kernel2 = Mat::ones(2, 2, CV_8UC1)?.to_mat()?;
        let mut opened = Mat::default();
        imgproc::morphology_ex(
            &temp,
            &mut opened,
            MORPH_OPEN,
            &kernel2,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            Scalar::default(),
        )?;
        if core::count_non_zero(&opened)? == 0 {
            return Ok(());
        }

        let mut min_dist = [255u8; 256];
        let mask_bytes = opened.data_bytes()?;
        let angle_bytes = self.angle_u8.data_bytes()?;
        let radius_bytes = self.radius_u8.data_bytes()?;
        let total = (MINIMAP_SIZE * MINIMAP_SIZE) as usize;
        for i in 0..total {
            if mask_bytes[i] == 255 {
                let a = angle_bytes[i] as usize;
                let r = radius_bytes[i];
                if min_dist[a] > r {
                    min_dist[a] = r;
                }
            }
        }

        let lut_base = Mat::from_slice(&min_dist)?;
        let lut = lut_base.reshape(1, 1)?;
        let mut lut_out = Mat::default();
        core::lut(&self.angle_u8, &lut, &mut lut_out)?;
        let mut radius_lt = Mat::default();
        core::compare(&self.radius_u8, &lut_out, &mut radius_lt, CMP_LT)?;

        let mut temp1 = Mat::default();
        core::in_range(bgr, &Scalar::all(100.0), &Scalar::all(255.0), &mut temp1)?;
        let mut combined = self.out_mask.try_clone()?;
        core::bitwise_or(&temp1, &radius_lt, &mut combined, &self.out_mask)?;
        self.out_mask = combined;
        Ok(())
    }
}

pub struct CameraOrientationCalculator {
    remap_x: Mat,
    remap_y: Mat,
    alpha_mask1_remap: Mat,
    alpha_mask2_remap: Mat,
}

impl CameraOrientationCalculator {
    pub fn new() -> Result<Self, CvError> {
        let r_arr = linear_spaced(TPL_INN_RAD, TPL_OUT_RAD, R_LENGTH, true);
        let theta_arr = linear_spaced(0.0, 360.0, THETA_LENGTH, false);

        let r_base = Mat::from_slice(&r_arr)?;
        let r_row = r_base.reshape(1, 1)?;
        let theta_base = Mat::from_slice(&theta_arr)?;
        let theta_col = theta_base.reshape(1, THETA_LENGTH as i32)?;
        let r_mat = core::repeat(&r_row, THETA_LENGTH as i32, 1)?;
        let theta_mat = core::repeat(&theta_col, 1, R_LENGTH as i32)?;

        let mut remap_x = Mat::default();
        let mut remap_y = Mat::default();
        core::polar_to_cart(&r_mat, &theta_mat, &mut remap_x, &mut remap_y, true)?;
        let mut rx_shift = Mat::default();
        let mut ry_shift = Mat::default();
        let half = Scalar::all((MINIMAP_SIZE as f64) / 2.0);
        core::add(&remap_x, &half, &mut rx_shift, &core::no_array(), -1)?;
        core::add(&remap_y, &half, &mut ry_shift, &core::no_array(), -1)?;

        let a2_vals: Vec<f32> = r_arr.iter().map(|&v| 137.0 + 1.43 * v).collect();
        let a1_vals: Vec<f32> = r_arr.iter().map(|&v| alpha1_lut_value(v)).collect();
        let a2_base = Mat::from_slice(&a2_vals)?;
        let a2_row = a2_base.reshape(1, 1)?;
        let a1_base = Mat::from_slice(&a1_vals)?;
        let a1_row = a1_base.reshape(1, 1)?;
        let alpha_mask2_remap = core::repeat(&a2_row, THETA_LENGTH as i32, 1)?;
        let alpha_mask1_remap = core::repeat(&a1_row, THETA_LENGTH as i32, 1)?;

        Ok(Self {
            remap_x: rx_shift,
            remap_y: ry_shift,
            alpha_mask1_remap,
            alpha_mask2_remap,
        })
    }

    pub fn predict_rotation(&self, src: &Mat, mask: &Mat) -> Result<(f32, f32), CvError> {
        let mut remap_src = Mat::default();
        let mut remap_mask = Mat::default();
        imgproc::remap(src, &mut remap_src, &self.remap_x, &self.remap_y, INTER_LINEAR, BORDER_CONSTANT, Scalar::default())?;
        imgproc::remap(mask, &mut remap_mask, &self.remap_x, &self.remap_y, INTER_NEAREST, BORDER_CONSTANT, Scalar::default())?;

        let mut hls = Mat::default();
        let mut gray = Mat::default();
        imgproc::cvt_color(&remap_src, &mut hls, COLOR_BGR2HLS_FULL, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        imgproc::cvt_color(&remap_src, &mut gray, COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let mut h_u8 = Mat::default();
        core::extract_channel(&hls, &mut h_u8, 0)?;

        let mut h_img = Mat::default();
        let mut fa_img = Mat::default();
        h_u8.convert_to(&mut h_img, CV_32FC1, 1.0, 0.0)?;
        gray.convert_to(&mut fa_img, CV_32FC1, 1.0, 0.0)?;

        let temp = fa_img.try_clone()?;
        apply_mask_inplace(&mut fa_img, &self.alpha_mask1_remap, Scalar::all(0.0))?;
        temp.copy_to_masked(&mut fa_img, &remap_mask)?;

        let mut temp_b = temp.try_clone()?;
        apply_mask_inplace(&mut temp_b, &self.alpha_mask2_remap, Scalar::all(255.0))?;
        let mut fb_img = temp_b.try_clone()?;
        apply_mask_inplace(&mut fb_img, &self.alpha_mask1_remap, Scalar::all(0.0))?;
        temp_b.copy_to_masked(&mut fb_img, &remap_mask)?;

        let mut imgs_a = Vector::<Mat>::new();
        imgs_a.push(h_img.clone());
        imgs_a.push(fa_img.clone());
        let mut imgs_b = Vector::<Mat>::new();
        imgs_b.push(h_img.clone());
        imgs_b.push(fb_img.clone());

        let channels = Vector::<i32>::from_slice(&[0, 1]);
        let hist_size = Vector::<i32>::from_slice(&[F_LENGTH as i32, F_LENGTH as i32]);
        let ranges = Vector::<f32>::from_slice(&[0.0, 256.0, 0.0, 256.0]);

        let mut hist_a = Mat::default();
        let mut hist_b = Mat::default();
        imgproc::calc_hist(&imgs_a, &channels, &core::no_array(), &mut hist_a, &hist_size, &ranges, false)?;
        imgproc::calc_hist(&imgs_b, &channels, &core::no_array(), &mut hist_b, &hist_size, &ranges, false)?;

        let h_data = h_img.data_typed::<f32>()?;
        let fa_data = fa_img.data_typed::<f32>()?;
        let fb_data = fb_img.data_typed::<f32>()?;
        let ha_data = hist_a.data_typed::<f32>()?;
        let hb_data = hist_b.data_typed::<f32>()?;

        let mut vote_row = vec![0.0f32; THETA_LENGTH];
        for i in 0..(THETA_LENGTH * R_LENGTH) {
            let h = h_data[i];
            let fa = fa_data[i];
            let fb = fb_data[i];
            if !(0.0..256.0).contains(&h) || fa < 0.0 || fb >= 256.0 {
                continue;
            }
            let row_idx = i / R_LENGTH;
            if fa >= 256.0 {
                vote_row[row_idx] += 255.0;
            } else if fb < 0.0 {
                vote_row[row_idx] += -255.0;
            } else {
                let hx = ((h / 256.0) * (F_LENGTH as f32)) as usize;
                let ay = ((fa / 256.0) * (F_LENGTH as f32)) as usize;
                let by = ((fb / 256.0) * (F_LENGTH as f32)) as usize;
                let ha = ha_data[hx * F_LENGTH + ay];
                let hb = hb_data[hx * F_LENGTH + by];
                vote_row[row_idx] += if ha > hb {
                    0.0
                } else if (ha - hb).abs() < 0.0001 {
                    100.0
                } else {
                    255.0
                };
            }
        }

        let vote_base = Mat::from_slice(&vote_row)?;
        let vote_mat = vote_base.reshape(1, 1)?;
        let mut upscaled = Mat::default();
        imgproc::resize(
            &vote_mat,
            &mut upscaled,
            Size::new((THETA_LENGTH * ORIENT_SCALE) as i32, 1),
            0.0,
            0.0,
            INTER_CUBIC,
        )?;

        let cols = upscaled.cols();
        let part1 = Mat::roi(&upscaled, Rect::new(cols - PEAK_WIDTH, 0, PEAK_WIDTH, 1))?;
        let part2 = Mat::roi(&upscaled, Rect::new(0, 0, cols - PEAK_WIDTH, 1))?;
        let mut hconcat_in = Vector::<Mat>::new();
        hconcat_in.push(part1.clone_pointee());
        hconcat_in.push(part2.clone_pointee());
        let mut result_shift = Mat::default();
        core::hconcat(&hconcat_in, &mut result_shift)?;

        let peak_region = Mat::roi(&result_shift, Rect::new(0, 0, PEAK_WIDTH, 1))?;
        let peak_region_sum = core::sum_elems(&peak_region)?[0];

        let mut diff = Mat::default();
        core::subtract(&upscaled, &result_shift, &mut diff, &core::no_array(), -1)?;
        let mut integral_mat = Mat::default();
        imgproc::integral(&diff, &mut integral_mat, -1)?;

        let mut max_val = 0.0;
        let mut max_loc = Point::default();
        core::min_max_loc(
            &integral_mat,
            None,
            Some(&mut max_val),
            None,
            Some(&mut max_loc),
            &core::no_array(),
        )?;

        let mut degree =
            ((max_loc.x - 1) as f32) / (THETA_LENGTH as f32) * 360.0 / (ORIENT_SCALE as f32) - 45.0;
        if degree < 0.0 {
            degree += 360.0;
        }
        let confidence =
            ((max_val + peak_region_sum) / ((PEAK_WIDTH as f64) * (R_LENGTH as f64) * 255.0)) as f32;
        Ok((degree, confidence))
    }
}

/// Fallback orientation estimator mirroring `CameraOrientationFromGia.cs`.
pub fn compute_camera_gia_fallback(minimap_bgr: &Mat) -> Result<f32, CvError> {
    let mut gray = Mat::default();
    if minimap_bgr.channels() == 3 {
        imgproc::cvt_color(minimap_bgr, &mut gray, COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    } else {
        gray = minimap_bgr.try_clone()?;
    }

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(&gray, &mut blurred, Size::new(3, 3), 0.0, 0.0, BORDER_CONSTANT, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let center = Point2f::new(blurred.cols() as f32 / 2.0, blurred.rows() as f32 / 2.0);
    let mut polar = Mat::default();
    imgproc::warp_polar(
        &blurred,
        &mut polar,
        Size::new(360, 360),
        center,
        360.0,
        INTER_LINEAR | WARP_POLAR_LINEAR,
    )?;

    let polar_roi = Mat::roi(&polar, Rect::new(10, 0, 70, polar.rows()))?;
    let mut rot = Mat::default();
    core::rotate(&polar_roi, &mut rot, core::ROTATE_90_COUNTERCLOCKWISE)?;

    let mut scharr = Mat::default();
    imgproc::scharr(&rot, &mut scharr, CV_32F, 1, 0, 1.0, 0.0, BORDER_CONSTANT)?;
    let arr = scharr.data_typed::<f32>()?;

    let mut left = [0i32; 360];
    let mut right = [0i32; 360];
    for idx in find_peaks(arr, false) {
        left[idx % 360] += 1;
    }
    for idx in find_peaks(arr, true) {
        right[idx % 360] += 1;
    }

    let mut left2 = [0i32; 360];
    let mut right2 = [0i32; 360];
    for i in 0..360 {
        left2[i] = (left[i] - right[i]).max(0);
        right2[i] = (right[i] - left[i]).max(0);
    }

    let mut sum = [0i32; 360];
    for shift in -2i32..=2 {
        let w = 3 - shift.abs();
        for i in 0..360 {
            let ri = ((i as i32 - (-90 + shift)).rem_euclid(360)) as usize;
            sum[i] += left2[i] * right2[ri] * w / 3;
        }
    }

    let mut result = [0i32; 360];
    for shift in -2i32..=2 {
        let w = 3 - shift.abs();
        for i in 0..360 {
            let si = ((i as i32 - shift).rem_euclid(360)) as usize;
            result[i] += sum[si] * w / 3;
        }
    }

    let max_index = result
        .iter()
        .enumerate()
        .max_by_key(|&(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut angle = (max_index + 45) as f32;
    if angle > 360.0 {
        angle -= 360.0;
    }
    Ok(angle)
}

fn find_peaks(arr: &[f32], negate: bool) -> Vec<usize> {
    let mut peaks = Vec::new();
    for i in 1..arr.len().saturating_sub(1) {
        let prev = if negate { -arr[i - 1] } else { arr[i - 1] };
        let cur = if negate { -arr[i] } else { arr[i] };
        let next = if negate { -arr[i + 1] } else { arr[i + 1] };
        if cur > prev && cur > next && cur > 100.0 {
            peaks.push(i);
        }
    }
    peaks
}

/// Computes player character triangle arrow direction in degrees (mirroring `CharacterOrientation.cs`).
pub fn compute_character_orientation(minimap_bgr: &Mat) -> Result<Option<i32>, CvError> {
    let mut ch = Vector::<Mat>::new();
    core::split(minimap_bgr, &mut ch)?;
    let b = ch.get(0)?;
    let r = ch.get(2)?;

    let mut mask_b = Mat::default();
    let mut mask_r = Mat::default();
    core::in_range(&b, &Scalar::all(250.0), &Scalar::all(255.0), &mut mask_b)?;
    core::in_range(&r, &Scalar::all(0.0), &Scalar::all(10.0), &mut mask_r)?;
    let mut and_mat = Mat::default();
    core::bitwise_and(&mask_b, &mask_r, &mut and_mat, &core::no_array())?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &and_mat,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::default(),
    )?;
    if contours.is_empty() {
        return Ok(None);
    }

    let mut max_rect = Rect::default();
    let mut max_idx = 0usize;
    for i in 0..contours.len() {
        let box_r = imgproc::bounding_rect(&contours.get(i)?)?;
        if box_r.width * box_r.height > max_rect.width * max_rect.height {
            max_rect = box_r;
            max_idx = i;
        }
    }
    if max_rect.width <= 0 || max_rect.height <= 0 {
        return Ok(None);
    }

    let max_contour = contours.get(max_idx)?;
    let perimeter = imgproc::arc_length(&max_contour, true)?;
    let mut approx = Vector::<Point>::new();
    imgproc::approx_poly_dp(&max_contour, &mut approx, 0.08 * perimeter, true)?;
    if approx.len() != 3 {
        return Ok(None);
    }

    let crop = Mat::roi(minimap_bgr, max_rect)?;
    let mut hsv = Mat::default();
    imgproc::cvt_color(&crop, &mut hsv, COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut hsv_thresh = Mat::default();
    core::in_range(
        &hsv,
        &Scalar::new(93.0, 155.0, 170.0, 0.0),
        &Scalar::new(255.0, 255.0, 255.0, 0.0),
        &mut hsv_thresh,
    )?;

    let mut max_black = 0;
    let mut best_angle = None;
    for i in 0..3 {
        let p0 = approx.get(i)?;
        let p1 = approx.get((i + 1) % 3)?;
        let target = approx.get((i + 2) % 3)?;
        let mid = Point::new((p0.x + p1.x) / 2, (p0.y + p1.y) / 2);

        let local_mid = Point::new(mid.x - max_rect.x, mid.y - max_rect.y);
        let local_target = Point::new(target.x - max_rect.x, target.y - max_rect.y);
        let black_count = count_dark_pixels_on_line(&hsv_thresh, local_mid, local_target)?;
        if black_count > max_black {
            max_black = black_count;
            let rad = ((target.y - mid.y) as f64).atan2((target.x - mid.x) as f64);
            best_angle = Some((rad * 180.0 / std::f64::consts::PI) as i32);
        }
    }

    Ok(best_angle)
}

fn count_dark_pixels_on_line(thresh_u8: &Mat, p0: Point, p1: Point) -> Result<i32, CvError> {
    let steps = (p1.x - p0.x).abs().max((p1.y - p0.y).abs()).max(1);
    let mut count = 0;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = ((p0.x as f32) + (p1.x - p0.x) as f32 * t).round() as i32;
        let y = ((p0.y as f32) + (p1.y - p0.y) as f32 * t).round() as i32;
        if x < 0 || y < 0 || x >= thresh_u8.cols() || y >= thresh_u8.rows() {
            break;
        }
        if *thresh_u8.at_2d::<u8>(y, x)? == 255 {
            break;
        }
        count += 1;
    }
    Ok(count)
}

pub struct MiniMapPreprocessor {
    mask_calc: MaskCalculator,
    co_calc: CameraOrientationCalculator,
}

impl MiniMapPreprocessor {
    pub fn new() -> Result<Self, CvError> {
        Ok(Self {
            mask_calc: MaskCalculator::new()?,
            co_calc: CameraOrientationCalculator::new()?,
        })
    }

    pub fn predict_rotation_with_confidence(&mut self, minimap_bgr: &Mat) -> Result<(f32, f32), CvError> {
        let (src, mask) = self.mask_calc.process1(minimap_bgr)?;
        self.co_calc.predict_rotation(&src, &mask)
    }

    pub fn compute_camera_orientation(&mut self, minimap_bgr: &Mat) -> Result<(f32, f32), CvError> {
        let (angle, conf) = self.predict_rotation_with_confidence(minimap_bgr)?;
        if conf < 0.2 {
            let fallback = compute_camera_gia_fallback(minimap_bgr)?;
            Ok((fallback, conf))
        } else {
            Ok((angle, conf))
        }
    }

    pub fn get_minimap_and_mask(&mut self, minimap_bgr: &Mat) -> Result<(Mat, Mat, f32, f32), CvError> {
        let (src, mask) = self.mask_calc.process1(minimap_bgr)?;
        let (raw_angle, conf) = self.co_calc.predict_rotation(&src, &mask)?;
        let (out_mat, out_mask) = self.mask_calc.process2(raw_angle)?;
        Ok((out_mat, out_mask, raw_angle, conf))
    }
}
