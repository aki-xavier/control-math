// mat.rs — two deliberate behaviours hold wherever they are met: `at` answers 0.0 out of range
// instead of panicking (an optional row reads as zeros), and `set` re-allocates when the storage
// length does not match rows * cols (a Mat built with a mismatched literal stays usable).

use crate::vec3::Vec3;

/// Row-major — the layout every index in this file assumes.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn eye(n: usize) -> Mat {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// The scaled identity a covariance is assembled from.
    pub fn eye_scaled(s: f64, n: usize) -> Mat {
        let mut r = Mat::zeros(n, n);
        for i in 0..n {
            r.set(i, i, s);
        }
        r
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Mat {
        if rows.is_empty() {
            return Mat::zeros(0, 0);
        }
        let mut m = Mat::zeros(rows.len(), rows[0].len());
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                m.set(i, j, *v);
            }
        }
        m
    }

    /// The allocating counterpart of set_block, for a system matrix written as four quadrants.
    pub fn from_blocks(a: &Mat, b: &Mat, c: &Mat, d: &Mat) -> Mat {
        let mut r = Mat::zeros(a.rows + c.rows, a.cols + b.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                r.set(i, j, a.at(i, j));
            }
        }
        for i in 0..b.rows {
            for j in 0..b.cols {
                r.set(i, a.cols + j, b.at(i, j));
            }
        }
        for i in 0..c.rows {
            for j in 0..c.cols {
                r.set(a.rows + i, j, c.at(i, j));
            }
        }
        for i in 0..d.rows {
            for j in 0..d.cols {
                r.set(a.rows + i, a.cols + j, d.at(i, j));
            }
        }
        r
    }

    /// copy_from copies src into self, resizing only when the data length differs: a loop that
    /// copies one matrix per step would otherwise allocate one per step.
    pub fn copy_from(&mut self, src: &Mat) {
        self.rows = src.rows;
        self.cols = src.cols;
        if self.data.len() != src.data.len() {
            self.data.resize(src.data.len(), 0.0);
        }
        self.data.copy_from_slice(&src.data);
    }

    pub fn skew(v: Vec3) -> Mat {
        let mut m = Mat::zeros(3, 3);
        m.set(0, 1, -v.z);
        m.set(0, 2, v.y);
        m.set(1, 0, v.z);
        m.set(1, 2, -v.x);
        m.set(2, 0, -v.y);
        m.set(2, 1, v.x);
        m
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Mat {
        let mut r = Mat::zeros(3, 3);
        Mat::from_axis_angle_into(axis, angle, &mut r);
        r
    }

    /// Into `out`, with skew(k) written inline: both the result and that skew were allocations
    /// otherwise.
    pub fn from_axis_angle_into(axis: Vec3, angle: f64, out: &mut Mat) {
        let k = axis.normalized();
        let c = angle.cos();
        let sn = angle.sin();
        if out.rows != 3 || out.cols != 3 {
            *out = Mat::zeros(3, 3);
        }
        let kv = k.to_array();
        // skew(k) in Mat::skew's own layout
        let kx = [
            [0.0, -kv[2], kv[1]],
            [kv[2], 0.0, -kv[0]],
            [-kv[1], kv[0], 0.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                out.set(
                    i,
                    j,
                    c * (if i == j { 1.0 } else { 0.0 })
                        + sn * kx[i][j]
                        + (1.0 - c) * kv[i] * kv[j],
                );
            }
        }
    }

    /// Axis-angle, world frame.
    pub fn to_rotvec(&self) -> Vec3 {
        let c = ((self.at(0, 0) + self.at(1, 1) + self.at(2, 2) - 1.0) / 2.0).clamp(-1.0, 1.0);
        let th = c.acos();
        if th < 1e-9 {
            return Vec3::ZERO;
        }
        let mut ax = Vec3 {
            x: self.at(2, 1) - self.at(1, 2),
            y: self.at(0, 2) - self.at(2, 0),
            z: self.at(1, 0) - self.at(0, 1),
        };
        let mut n = ax.norm();
        if n < 1e-12 {
            let v = Vec3 {
                x: (self.at(0, 0) + 1.0).max(0.0).sqrt(),
                y: (self.at(1, 1) + 1.0).max(0.0).sqrt(),
                z: (self.at(2, 2) + 1.0).max(0.0).sqrt(),
            };
            ax = v.normalized();
            n = 1.0;
        }
        ax.scale(th / n)
    }

    /// The guard against a covariance that has drifted asymmetric: (M + M')/2.
    pub fn symmetrized(&self) -> Mat {
        let mut r = Mat::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r.set(i, j, self.at(i, j).midpoint(self.at(j, i)));
            }
        }
        r
    }

    /// Out-of-range reads answer 0.0 rather than panicking, so an optional row reads as zeros.
    pub fn at(&self, i: usize, j: usize) -> f64 {
        if i >= self.rows || j >= self.cols {
            return 0.0;
        }
        if self.data.is_empty() {
            return 0.0;
        }
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i >= self.rows || j >= self.cols {
            return;
        }
        if self.data.len() != self.rows * self.cols {
            self.data = vec![0.0; self.rows * self.cols];
        }
        self.data[i * self.cols + j] = v;
    }

    pub fn transposed(&self) -> Mat {
        let mut r = Mat::zeros(self.cols, self.rows);
        self.transposed_into(&mut r);
        r
    }

    /// Into `out`, for mul_into's reason.
    pub fn transposed_into(&self, out: &mut Mat) {
        if out.rows != self.cols || out.cols != self.rows {
            *out = Mat::zeros(self.cols, self.rows);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.at(i, j));
            }
        }
    }

    // A method rather than `impl Add`, as Vec3::add records: the expressions are written out and
    // the tests measure them.
    #[allow(clippy::should_implement_trait)]
    pub fn add(&self, o: &Mat) -> Mat {
        let mut r = Mat::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            r.data[i] = self.data[i] + o.data[i];
        }
        r
    }

    pub fn scale(&self, s: f64) -> Mat {
        let mut r = Mat::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            r.data[i] = self.data[i] * s;
        }
        r
    }

    pub fn max_abs(&self) -> f64 {
        let mut mx = 0.0;
        for v in &self.data {
            if v.abs() > mx {
                mx = v.abs();
            }
        }
        mx
    }

    pub fn set_block(&mut self, a: &Mat, off_r: usize, off_c: usize) {
        for i in 0..a.rows {
            for j in 0..a.cols {
                self.set(off_r + i, off_c + j, a.at(i, j));
            }
        }
    }

    pub fn mul(&self, o: &Mat) -> Mat {
        let mut r = Mat::zeros(self.rows, o.cols);
        self.mul_into(o, &mut r);
        r
    }

    /// Into `out`, resized on demand and reused after that: otherwise a tight loop allocates a
    /// fresh `Mat` per step.
    pub fn mul_into(&self, o: &Mat, out: &mut Mat) {
        if out.rows != self.rows || out.cols != o.cols {
            *out = Mat::zeros(self.rows, o.cols);
        } else {
            for v in out.data.iter_mut() {
                *v = 0.0;
            }
        }
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.at(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..o.cols {
                    let cur = out.data[i * out.cols + j];
                    out.data[i * out.cols + j] = cur + a * o.at(k, j);
                }
            }
        }
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..x.len() {
                y[i] += self.at(i, j) * x[j];
            }
        }
        y
    }

    pub fn mul_vec3(&self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.at(0, 0) * v.x + self.at(0, 1) * v.y + self.at(0, 2) * v.z,
            y: self.at(1, 0) * v.x + self.at(1, 1) * v.y + self.at(1, 2) * v.z,
            z: self.at(2, 0) * v.x + self.at(2, 1) * v.y + self.at(2, 2) * v.z,
        }
    }

    /// Partial-pivoting Gaussian elimination. A singular pivot is skipped and the result is
    /// best-effort: a solve that can meet a singular matrix argues about conditioning at its own
    /// level.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.rows;
        let mut a = vec![0.0; self.data.len()];
        a[..n * self.cols].copy_from_slice(&self.data[..n * self.cols]);
        let mut rhs = vec![0.0; b.len()];
        rhs[..b.len()].copy_from_slice(b);
        for col in 0..n {
            let mut piv = col;
            let mut best = a[col * self.cols + col].abs();
            for i in col + 1..n {
                let cand = a[i * self.cols + col].abs();
                if cand > best {
                    best = cand;
                    piv = i;
                }
            }
            if best < 1e-14 {
                continue; // singular; keep going, result is best-effort
            }
            if piv != col {
                for j in 0..self.cols {
                    a.swap(col * self.cols + j, piv * self.cols + j);
                }
                rhs.swap(col, piv);
            }
            let div = a[col * self.cols + col];
            for j in 0..self.cols {
                a[col * self.cols + j] /= div;
            }
            rhs[col] /= div;
            for i in 0..n {
                if i == col {
                    continue;
                }
                let f = a[i * self.cols + col];
                if f == 0.0 {
                    continue;
                }
                for j in 0..self.cols {
                    a[i * self.cols + j] -= f * a[col * self.cols + j];
                }
                rhs[i] -= f * rhs[col];
            }
        }
        rhs
    }

    /// One solve per column; inv() is this against the identity.
    pub fn solve_mat(&self, b: &Mat) -> Mat {
        let mut out = Mat::zeros(b.rows, b.cols);
        for j in 0..b.cols {
            let mut col = vec![0.0; b.rows];
            for i in 0..b.rows {
                col[i] = b.data[i * b.cols + j];
            }
            let sol = self.solve(&col);
            for (i, v) in sol.iter().enumerate() {
                out.set(i, j, *v);
            }
        }
        out
    }

    pub fn inv(&self) -> Mat {
        self.solve_mat(&Mat::eye(self.rows))
    }

    /// Nested rows, for JSON-friendly output.
    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        let mut rows = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                rows[i][j] = self.at(i, j);
            }
        }
        rows
    }

    pub fn diag(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.rows];
        for i in 0..self.rows {
            d[i] = self.at(i, i);
        }
        d
    }
}
