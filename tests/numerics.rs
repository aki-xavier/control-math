// numerics.rs — the QP solvers and the task-space bridge, this crate's own slice of them: each
// assertion is an arithmetic property their callers depend on (the box projection, the
// left/right-inverse identity, the weighted ridge's fallbacks), not a pinned consumer reading.

use control_math::mat::Mat;
use control_math::qp::{box_qp, qp_ineq};
use control_math::task_space_bridge::TaskSpaceBridge;

fn close(got: &[f64], want: &[f64], tol: f64) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

#[test]
fn the_bridge_on_an_identity_map_is_the_scaled_error() {
    let j = Mat::eye(3);
    let e = [1.0, 2.0, 3.0];
    // no ridge: dq = e
    assert!(close(&TaskSpaceBridge::new(3, 0.0).step(&j, &e), &e, 1e-12));
    // ridge: (I + lam I)^-1 e = e / (1 + lam)
    let damped = TaskSpaceBridge::new(3, 0.5).step(&j, &e);
    assert!(close(&damped, &[1.0 / 1.5, 2.0 / 1.5, 3.0 / 1.5], 1e-12));
}

#[test]
fn the_left_and_right_inverse_forms_agree() {
    // J' (JJ' + lam I)^-1 = (J'J + lam I)^-1 J': the bridge's two spellings of one damped inverse,
    // on a map where the two systems have different sizes.
    let j = Mat::from_rows(&[vec![1.0, 0.5, 0.0], vec![0.0, 0.25, 1.0]]);
    let e = [0.3, -0.2];
    let b = TaskSpaceBridge::new(3, 1e-3);
    assert!(close(&b.step(&j, &e), &b.solve_right(&j, &e), 1e-9));
}

#[test]
fn a_weight_of_ones_is_the_plain_ridge() {
    let j = Mat::from_rows(&[vec![1.0, 2.0], vec![3.0, 1.0]]);
    let e = [1.0, -1.0];
    let b = TaskSpaceBridge::new(2, 1e-6);
    assert!(close(
        &b.step(&j, &e),
        &b.step_weighted(&j, &e, &[1.0, 1.0]),
        1e-12
    ));
}

#[test]
fn a_non_finite_or_non_positive_weight_answers_one() {
    let j = Mat::from_rows(&[vec![1.0, 2.0], vec![3.0, 1.0]]);
    let e = [1.0, -1.0];
    let b = TaskSpaceBridge::new(2, 1e-6);
    let plain = b.step(&j, &e);
    for w in [
        [f64::NAN, 1.0],
        [0.0, 1.0],
        [-2.0, 1.0],
        [f64::INFINITY, 1.0],
    ] {
        assert!(close(&plain, &b.step_weighted(&j, &e, &w), 1e-12));
    }
}

#[test]
fn box_qp_clips_the_unconstrained_minimum() {
    let h = Mat::eye(2);
    // inside the box: u* = -H^-1 f = [0.5, 0.5]
    let inside = box_qp(&h, &[-0.5, -0.5], &[-1.0, -1.0], &[1.0, 1.0]);
    assert!(close(&inside, &[0.5, 0.5], 1e-12));
    // against the upper bound: u* = [2, 2] cut back to [1, 1]
    let clipped = box_qp(&h, &[-2.0, -2.0], &[-1.0, -1.0], &[1.0, 1.0]);
    assert!(close(&clipped, &[1.0, 1.0], 1e-12));
}

#[test]
fn qp_ineq_lands_on_the_row_it_must_respect() {
    // min 0.5 u'u - u0 s.t. u0 <= 0.25: the unconstrained u0 = 1 is cut back onto the row.
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
