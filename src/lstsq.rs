use crate::mat::Mat;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DampedLstsq {
    pub n: usize,
    pub lam: f64,
}

impl DampedLstsq {
    pub fn new(n: usize, lam: f64) -> DampedLstsq {
        DampedLstsq { n, lam }
    }

    pub fn solve(&self, j: &Mat, e: &[f64]) -> Vec<f64> {
        let jt = j.transposed();
        let mut a = jt.mul(j);
        for i in 0..self.n {
            a.set(i, i, a.at(i, i) + self.lam);
        }
        a.solve(&jt.mul_vec(e))
    }

    pub fn solve_right(&self, j: &Mat, e: &[f64]) -> Vec<f64> {
        let jt = j.transposed();
        let mut a = j.mul(&jt);
        for i in 0..a.rows {
            a.set(i, i, a.at(i, i) + self.lam);
        }
        jt.mul_vec(&a.solve(e))
    }
}
