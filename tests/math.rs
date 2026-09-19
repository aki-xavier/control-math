// math.rs — this crate's own tests, with their own tolerances, plus the cross-checks its
// consumers depend on.
//
// These assertions are the oracle: they are its numbers, so an expression that
// is merely "close" or "equivalent" fails here.

use control_math::mat::Mat;
use control_math::quat::Quat;
use control_math::rng::Mt19937;
use control_math::vec3::Vec3;

// ---- core arithmetic --------------------------------------------------------

#[test]
fn solve_and_inv() {
    let a = Mat::from_rows(&[vec![2.0, 1.0], vec![1.0, 3.0]]);
    let b = [1.0, 2.0];
    let x = a.solve(&b);
    assert!((x[0] - 0.2).abs() < 1e-12);
    assert!((x[1] - 0.6).abs() < 1e-12);
    let ai = a.inv();
    let eye = a.mul(&ai);
    assert!((eye.at(0, 0) - 1.0).abs() < 1e-12);
    assert!(eye.at(0, 1).abs() < 1e-12);
    assert!((eye.at(1, 1) - 1.0).abs() < 1e-12);
}

#[test]
fn mul_and_vec() {
    let a = Mat::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let x = [1.0, 1.0, 1.0];
    let y = a.mul_vec(&x);
    assert_eq!(y.len(), 2);
    assert!((y[0] - 6.0).abs() < 1e-12);
    assert!((y[1] - 15.0).abs() < 1e-12);
    let at = a.transposed();
    assert_eq!(at.rows, 3);
    assert_eq!(at.cols, 2);
}

#[test]
fn quat_to_mat3_roundtrip() {
    let q = Quat {
        w: 0.3f64.cos(),
        x: 0.0,
        y: 0.0,
        z: 0.3f64.sin(),
    };
    let r = q.to_mat3();
    let q2 = Quat::from_mat3(&r);
    // sign-normalized w>0 equivalence (identical rotation up to sign)
    assert!((q.w - q2.w).abs() < 1e-10);
    assert!((q.z - q2.z).abs() < 1e-10);
}

#[test]
fn quat_rotvec_between_pi() {
    // rotation of pi about z maps to rotation vector pi*z.
    let q = Quat {
        w: 0.0,
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    let w = Quat::rotvec_between(q, Quat::IDENTITY);
    assert!((w.z - std::f64::consts::PI).abs() < 1e-9);
    assert!(w.x.abs() < 1e-12);
    assert!(w.y.abs() < 1e-12);
}

// ---- the cross-checks the dynamics layer relies on ---------------------------

#[test]
fn from_axis_angle_agrees_with_the_quaternion_path() {
    // the dynamics derive joint frames from Mat::from_axis_angle while the plant's base
    // pose travels as a quaternion: the two must be the same rotation, or every
    // frame in between is wrong by the difference
    let axis = Vec3 {
        x: 1.0,
        y: -2.0,
        z: 0.5,
    }
    .normalized();
    for angle in [-2.5, -0.7, 0.0, 0.3, 1.9] {
        let r = Mat::from_axis_angle(axis, angle);
        let half = angle / 2.0;
        let q = Quat {
            w: half.cos(),
            x: axis.x * half.sin(),
            y: axis.y * half.sin(),
            z: axis.z * half.sin(),
        };
        let rq = q.to_mat3();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r.at(i, j) - rq.at(i, j)).abs() < 1e-12,
                    "Mat::from_axis_angle vs quat to_mat3 at ({i},{j}) for angle {angle}"
                );
            }
        }
    }
}

#[test]
fn to_rotvec_is_the_inverse_of_the_exponential_map() {
    let axis = Vec3 {
        x: 0.2,
        y: 1.0,
        z: -0.4,
    }
    .normalized();
    for angle in [0.0, 1e-6, 0.25, 3.0] {
        let r = Mat::from_axis_angle(axis, angle);
        let w = r.to_rotvec();
        assert!(
            (w.norm() - angle).abs() < 1e-6,
            "rotvec length {} vs angle {angle}",
            w.norm()
        );
        if angle > 1e-3 {
            // direction is the axis (for angles below pi, where the map is injective)
            let u = w.normalized();
            assert!((u.dot(axis) - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn eye_and_at_out_of_range_follow_the_v_behaviour() {
    let e = Mat::eye(3);
    assert_eq!(e.diag(), vec![1.0, 1.0, 1.0]);
    // at() outside the matrix answers 0.0 rather than panicking: the control
    // layer reads optional rows through it
    assert_eq!(e.at(9, 9), 0.0);
    assert_eq!(e.at(3, 0), 0.0);
}

#[test]
fn mt19937_stream_is_deterministic_and_standard_normal() {
    let mut a = Mt19937::new(0);
    let mut b = Mt19937::new(0);
    for _ in 0..1000 {
        assert_eq!(a.next_f64(), b.next_f64());
    }
    // the noise-driven estimator benchmarks need a unit-variance source
    let mut r = Mt19937::new(0);
    let n = 200_000;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for _ in 0..n {
        let x = r.randn();
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / f64::from(n);
    let var = sum2 / f64::from(n) - mean * mean;
    assert!(mean.abs() < 0.02, "randn mean {mean}");
    assert!((var - 1.0).abs() < 0.02, "randn variance {var}");
    // uniform draws stay in [0, 1)
    let mut u = Mt19937::new(7);
    for _ in 0..1000 {
        let x = u.next_f64();
        assert!((0.0..1.0).contains(&x), "next_f64 out of range: {x}");
    }
}
