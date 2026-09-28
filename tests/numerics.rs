use control_math::lstsq::DampedLstsq;
use control_math::mat::Mat;
use control_math::qp::{box_qp, qp_ineq};

fn close(got: &[f64], want: &[f64], tol: f64) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

#[test]
fn solve_on_an_identity_map_is_the_scaled_error() {
    let j = Mat::eye(3);
    let e = [1.0, 2.0, 3.0];
    assert!(close(&DampedLstsq::new(3, 0.0).solve(&j, &e), &e, 1e-12));
    let damped = DampedLstsq::new(3, 0.5).solve(&j, &e);
    assert!(close(&damped, &[1.0 / 1.5, 2.0 / 1.5, 3.0 / 1.5], 1e-12));
}

#[test]
fn the_left_and_right_inverse_forms_agree() {
    let j = Mat::from_rows(&[vec![1.0, 0.5, 0.0], vec![0.0, 0.25, 1.0]]);
    let e = [0.3, -0.2];
    let b = DampedLstsq::new(3, 1e-3);
    assert!(close(&b.solve(&j, &e), &b.solve_right(&j, &e), 1e-9));
}

#[test]
fn box_qp_clips_the_unconstrained_minimum() {
    let h = Mat::eye(2);
    let inside = box_qp(&h, &[-0.5, -0.5], &[-1.0, -1.0], &[1.0, 1.0]);
    assert!(close(&inside, &[0.5, 0.5], 1e-12));
    let clipped = box_qp(&h, &[-2.0, -2.0], &[-1.0, -1.0], &[1.0, 1.0]);
    assert!(close(&clipped, &[1.0, 1.0], 1e-12));
}

#[test]
fn qp_ineq_lands_on_the_row_it_must_respect() {
    let h = Mat::eye(2);
    let a = Mat::from_rows(&[vec![1.0, 0.0]]);
    let u = qp_ineq(
        &h,
        &[-1.0, 0.0],
        &[-10.0, -10.0],
        &[10.0, 10.0],
        &a,
        &[0.25],
    );
    assert!(close(&u, &[0.25, 0.0], 1e-6));
}
