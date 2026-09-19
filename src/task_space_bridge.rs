// task_space_bridge.rs — the task-space GA-PID bridge: maps a task-space error to a joint-space
// increment through the damped least-squares pseudo-inverse, (J'J + lambda I) dq = J' e. It moved
// here from simu: it is arithmetic over `Mat` and nothing else, and it was the last edge from the
// legged stack (synergy) into the arm core.

use crate::mat::Mat;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TaskSpaceBridge {
    pub n: usize,
    pub lam: f64,
}

impl TaskSpaceBridge {
    pub fn new(n: usize, lam: f64) -> TaskSpaceBridge {
        TaskSpaceBridge { n, lam }
    }

    /// step maps an m-DOF task error over an m x n Jacobian through the left-inverse form (J'J + lam I)^-1 J' e.
    pub fn step(&self, j: &Mat, e: &[f64]) -> Vec<f64> {
        self.solve(j, e)
    }

    /// solve_right is the right-inverse form J' (J J' + lam I)^-1 e; cheaper than step when m < n.
    pub fn solve_right(&self, j: &Mat, e: &[f64]) -> Vec<f64> {
        let jt = j.transposed();
        let mut a = j.mul(&jt);
        for i in 0..a.rows {
            a.set(i, i, a.at(i, i) + self.lam);
        }
        jt.mul_vec(&a.solve(e))
    }

    fn solve(&self, j: &Mat, e: &[f64]) -> Vec<f64> {
        let jt = j.transposed();
        let mut a = jt.mul(j);
        for i in 0..self.n {
            a.set(i, i, a.at(i, i) + self.lam);
        }
        a.solve(&jt.mul_vec(e))
    }

    /// step_weighted is the same solve with a WEIGHTED ridge, `(J'J + lam W^-1) dq = J' e`: a large
    /// weight is spent first, non-finite or non-positive weights answer 1.0, a weight vector of ones
    /// reproduces `solve`; on a tall map (`m > n`) only the ridge moves with them.
    pub fn step_weighted(&self, j: &Mat, e: &[f64], w: &[f64]) -> Vec<f64> {
        let jt = j.transposed();
        let mut a = jt.mul(j);
        for i in 0..self.n {
            let wi = w
                .get(i)
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(1.0);
            a.set(i, i, a.at(i, i) + self.lam / wi);
        }
        a.solve(&jt.mul_vec(e))
    }
}
