use opencv::{
    core::{self, Rect, Scalar},
    imgproc,
    prelude::*,
};

use crate::CvError;

pub fn attribute_tab_ocr_image(frame: &Mat) -> Result<Mat, CvError> {
    let width = frame.cols();
    let height = frame.rows();
    let left_width = width / 5;
    if width <= 0 || height <= 0 || left_width == 0 {
        return Err(CvError::InvalidImage(
            "属性页门控需要宽度至少为 5 的有效图像".into(),
        ));
    }
    let left = Mat::roi(frame, Rect::new(0, 0, left_width, height))?;
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &left,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut mask = Mat::default();
    core::in_range(
        &hsv,
        &Scalar::new(0.0, 0.0, 243.0, 0.0),
        &Scalar::new(30.0, 19.0, 249.0, 0.0),
        &mut mask,
    )?;
    let mut binary = Mat::default();
    imgproc::cvt_color(
        &mask,
        &mut binary,
        imgproc::COLOR_GRAY2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(binary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::CV_8UC3;

    #[test]
    fn attribute_gate_uses_only_left_fifth_and_csharp_hsv_range() {
        let mut frame = Mat::zeros(50, 100, CV_8UC3).unwrap().to_mat().unwrap();
        let mut left = Mat::roi_mut(&mut frame, Rect::new(0, 0, 20, 50)).unwrap();
        left.set_to(&Scalar::all(246.0), &core::no_array()).unwrap();
        let binary = attribute_tab_ocr_image(&frame).unwrap();
        assert_eq!((binary.cols(), binary.rows()), (20, 50));
        assert_eq!(core::mean(&binary, &core::no_array()).unwrap()[0], 255.0);

        let mut frame = Mat::zeros(50, 100, CV_8UC3).unwrap().to_mat().unwrap();
        let mut left = Mat::roi_mut(&mut frame, Rect::new(0, 0, 20, 50)).unwrap();
        left.set_to(&Scalar::new(255.0, 0.0, 0.0, 0.0), &core::no_array())
            .unwrap();
        let binary = attribute_tab_ocr_image(&frame).unwrap();
        assert_eq!(core::mean(&binary, &core::no_array()).unwrap()[0], 0.0);
    }

    #[test]
    fn attribute_gate_rejects_empty_or_too_narrow_images() {
        let empty = Mat::default();
        assert!(attribute_tab_ocr_image(&empty).is_err());
        let narrow = Mat::zeros(10, 4, CV_8UC3).unwrap().to_mat().unwrap();
        assert!(attribute_tab_ocr_image(&narrow).is_err());
    }
}
