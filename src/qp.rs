use crate::mat::Mat;

fn clamp_box(v: f64, lb: f64, ub: f64) -> f64 {
    lb.max(ub.min(v))
}

pub fn box_qp(h: &Mat, f_vec: &[f64], lb: &[f64], ub: &[f64]) -> Vec<f64> {
    let n = h.rows;
    let mut u = vec![0.0; n];
    for i in 0..n {
        u[i] = clamp_box(0.0, lb[i], ub[i]);
    }
    let mut active = vec![false; n];
    for i in 0..n {
        active[i] = u[i] <= lb[i] + 1e-14 || u[i] >= ub[i] - 1e-14;
    }
    for _ in 0..400 {
        let mut g = h.mul_vec(u.as_slice());
        for i in 0..n {
            g[i] += f_vec[i];
        }
        let mut worst = -1.0;
        let mut rel: i32 = -1;
        for i in 0..n {
            if active[i] {
                if u[i] <= lb[i] + 1e-14 {
                    if g[i] < -1e-9 && -g[i] > worst {
                        worst = -g[i];
                        rel = i as i32;
                    }
                } else if g[i] > 1e-9 && g[i] > worst {
                    worst = g[i];
                    rel = i as i32;
                }
            }
        }
        if rel >= 0 {
            let rel = rel as usize;
            active[rel] = false;
            u[rel] = lb[rel].midpoint(ub[rel]);
            continue;
        }
        let mut free_idx: Vec<usize> = Vec::new();
        for (i, a) in active.iter().enumerate() {
            if !a {
                free_idx.push(i);
            }
        }
        if free_idx.is_empty() {
            return u;
        }
        let mut hf = Mat::zeros(free_idx.len(), free_idx.len());
        let mut rhs = vec![0.0; free_idx.len()];
        for a in 0..free_idx.len() {
            for bb in 0..free_idx.len() {
                hf.set(a, bb, h.at(free_idx[a], free_idx[bb]));
            }
            let mut s = f_vec[free_idx[a]];
            for i2 in 0..n {
                if active[i2] {
                    s += h.at(free_idx[a], i2) * u[i2];
                }
            }
            rhs[a] = -s;
        }
        let uf = hf.solve(&rhs);
        let mut maxd = 0.0;
        for a in 0..free_idx.len() {
            let d = (uf[a] - u[free_idx[a]]).abs();
            if d > maxd {
                maxd = d;
            }
        }
        if maxd < 1e-11 {
            return u;
        }
        let mut alpha = 1.0;
        let mut block: i32 = -1;
        for a in 0..free_idx.len() {
            let i = free_idx[a];
            let d = uf[a] - u[i];
            if d > 0.0 {
                let step = (ub[i] - u[i]) / d;
                if step < alpha {
                    alpha = step;
                    block = i as i32;
                }
            } else if d < 0.0 {
                let step = (lb[i] - u[i]) / d;
                if step < alpha {
                    alpha = step;
                    block = i as i32;
                }
            }
        }
        for a in 0..free_idx.len() {
            let i = free_idx[a];
            u[i] += alpha * (uf[a] - u[i]);
            if i as i32 == block {
                active[i] = true;
            }
        }
    }
    u
}

pub fn qp_ineq(
    h: &Mat,
    f_vec: &[f64],
    lb: &[f64],
    ub: &[f64],
    a_mat: &Mat,
    b_vec: &[f64],
) -> Vec<f64> {
    qp_ineq_warm(h, f_vec, lb, ub, a_mat, b_vec, &[])
}

pub fn qp_ineq_warm(
    h: &Mat,
    f_vec: &[f64],
    lb: &[f64],
    ub: &[f64],
    a_mat: &Mat,
    b_vec: &[f64],
    u0: &[f64],
) -> Vec<f64> {
    let n = h.rows;
    let m = b_vec.len();
    let mut u = vec![0.0; n];
    for i in 0..n {
        let mut init = 0.0;
        if u0.len() == n {
            init = u0[i];
        }
        u[i] = clamp_box(init, lb[i], ub[i]);
    }
    if m > 0 {
        for _ in 0..800 {
            let av = a_mat.mul_vec(&u);
            let mut viol = vec![0.0; m];
            let mut vc = 0.0;
            for i in 0..m {
                viol[i] = 0.0f64.max(av[i] - b_vec[i]);
                vc += viol[i] * viol[i];
            }
            if vc < 1e-18 {
                break;
            }
            let grad = a_mat.transposed().mul_vec(&viol);
            let mut alpha = 1.0;
            for _ in 0..24 {
                let mut un = vec![0.0; n];
                for i in 0..n {
                    un[i] = clamp_box(u[i] - alpha * grad[i], lb[i], ub[i]);
                }
                let avn = a_mat.mul_vec(&un);
                let mut vn = 0.0;
                for i in 0..m {
                    let d = 0.0f64.max(avn[i] - b_vec[i]);
                    vn += d * d;
                }
                if vn < vc * 0.999 {
                    u = un.clone();
                    break;
                }
                alpha *= 0.5;
            }
        }
    }
    let mut w_list: Vec<usize> = Vec::new();
    for _it in 0..300 {
        let mut free_idx: Vec<usize> = Vec::new();
        for i in 0..n {
            if u[i] > lb[i] + 1e-12 && u[i] < ub[i] - 1e-12 {
                free_idx.push(i);
            }
        }
        let nf = free_idx.len();
        let nw = w_list.len();
        let ns = nf + nw;
        if ns == 0 {
            return u;
        }
        let mut k = Mat::zeros(ns, ns);
        let mut rhs = vec![0.0; ns];
        for a in 0..nf {
            rhs[a] = -f_vec[free_idx[a]];
            for p in 0..n {
                if u[p] <= lb[p] + 1e-12 || u[p] >= ub[p] - 1e-12 {
                    rhs[a] -= h.at(free_idx[a], p) * u[p];
                }
            }
            for b2 in 0..nf {
                k.set(a, b2, h.at(free_idx[a], free_idx[b2]));
            }
        }
        let mut singular = false;
        for r in 0..nw {
            let row = w_list[r];
            rhs[nf + r] = b_vec[row];
            for p in 0..n {
                if u[p] <= lb[p] + 1e-12 || u[p] >= ub[p] - 1e-12 {
                    rhs[nf + r] -= a_mat.at(row, p) * u[p];
                }
            }
            for b2 in 0..nf {
                k.set(nf + r, b2, a_mat.at(row, free_idx[b2]));
                k.set(b2, nf + r, a_mat.at(row, free_idx[b2]));
            }
        }
        let sol = k.solve(&rhs);
        for &v in &sol {
            if v.is_nan() {
                singular = true;
                break;
            }
        }
        if singular {
            if nw > 0 {
                w_list.pop();
            }
            continue;
        }
        let mut alpha = 1.0;
        let mut block: i32 = -1;
        for a in 0..nf {
            let i = free_idx[a];
            let d = sol[a] - u[i];
            if d > 0.0 {
                let step = (ub[i] - u[i]) / d;
                if step < alpha {
                    alpha = step;
                    block = i as i32;
                }
            } else if d < 0.0 {
                let step = (lb[i] - u[i]) / d;
                if step < alpha {
                    alpha = step;
                    block = i as i32;
                }
            }
        }
        if alpha < 1.0 {
            for a in 0..nf {
                let i = free_idx[a];
                u[i] += alpha * (sol[a] - u[i]);
            }
            let block = block as usize;
            u[block] = if u[block] > lb[block].midpoint(ub[block]) {
                ub[block]
            } else {
                lb[block]
            };
            continue;
        }
        for a in 0..nf {
            u[free_idx[a]] = sol[a];
        }
        let mut worst_v = 0.0;
        let mut worst_row: i32 = -1;
        for r in 0..m {
            if !w_list.contains(&r) {
                let mut s = -b_vec[r];
                for i in 0..n {
                    s += a_mat.at(r, i) * u[i];
                }
                if s > worst_v {
                    worst_v = s;
                    worst_row = r as i32;
                }
            }
        }
        if worst_row >= 0 {
            w_list.push(worst_row as usize);
            continue;
        }
        let mut worst_l = -1e-30;
        let mut rel: i32 = -1;
        for r in 0..nw {
            let l = sol[nf + r];
            if l < worst_l {
                worst_l = l;
                rel = r as i32;
            }
        }
        if rel >= 0 {
            w_list.remove(rel as usize);
            continue;
        }
        let mut lam = vec![0.0; nw];
        lam[..nw].copy_from_slice(&sol[nf..nf + nw]);
        let mut geff = h.mul_vec(&u);
        for i in 0..n {
            geff[i] += f_vec[i];
        }
        for r in 0..nw {
            let row = w_list[r];
            for i in 0..n {
                geff[i] += a_mat.at(row, i) * lam[r];
            }
        }
        let mut rel_b: i32 = -1;
        let mut rel_worst = 0.0;
        for i in 0..n {
            if u[i] <= lb[i] + 1e-12 {
                if -geff[i] > rel_worst {
                    rel_worst = -geff[i];
                    rel_b = i as i32;
                }
            } else if u[i] >= ub[i] - 1e-12 {
                if geff[i] > rel_worst {
                    rel_worst = geff[i];
                    rel_b = i as i32;
                }
            } else if geff[i].abs() > rel_worst {
                rel_worst = geff[i].abs();
                rel_b = i as i32;
            }
        }
        if rel_b >= 0 && rel_worst > 1e-7 {
            let rel_b = rel_b as usize;
            u[rel_b] = lb[rel_b].midpoint(ub[rel_b]);
            continue;
        }
        return u;
    }
    u
}

pub struct ConstHessianQp {
    pub n: usize,
    pub h: Mat,
    pub g_inv: Mat,
    pub u_prev: Vec<f64>,
    pub pins_prev: Vec<usize>,
    pub wrows_prev: Vec<usize>,
    pub iters: usize,
    pub phase1_iter: usize,
    pub n_row_add: usize,
    pub n_row_rel: usize,
    pub n_pin_add: usize,
    pub n_pin_rel: usize,
    pub n_eq_fail: usize,
}

impl ConstHessianQp {
    pub fn new(h: Mat) -> ConstHessianQp {
        let n = h.rows;
        let mut l = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut s = h.at(i, j);
                for k in 0..j {
                    s -= l.at(i, k) * l.at(j, k);
                }
                if i == j {
                    l.set(i, j, s.max(1e-300).sqrt());
                } else {
                    l.set(i, j, s / l.at(j, j));
                }
            }
        }
        let mut g = Mat::zeros(n, n);
        for j in 0..n {
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut s = if i == j { 1.0 } else { 0.0 };
                for k in 0..i {
                    s -= l.at(i, k) * y[k];
                }
                y[i] = s / l.at(i, i);
            }
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in i + 1..n {
                    s -= l.at(k, i) * g.at(k, j);
                }
                g.set(i, j, s / l.at(i, i));
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                let avg = g.at(i, j).midpoint(g.at(j, i));
                g.set(i, j, avg);
                g.set(j, i, avg);
            }
        }
        ConstHessianQp {
            n,
            h,
            g_inv: g,
            u_prev: Vec::new(),
            pins_prev: Vec::new(),
            wrows_prev: Vec::new(),
            iters: 0,
            phase1_iter: 0,
            n_row_add: 0,
            n_row_rel: 0,
            n_pin_add: 0,
            n_pin_rel: 0,
            n_eq_fail: 0,
        }
    }

    fn warm_u(&self, lb: &[f64], ub: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut u = vec![0.0; n];
        for i in 0..n {
            let mut init = 0.0;
            if self.u_prev.len() == n {
                init = self.u_prev[i];
            }
            u[i] = clamp_box(init, lb[i], ub[i]);
        }
        u
    }

    fn restore_pins(&self, lb: &[f64], ub: &[f64]) -> (Vec<bool>, Vec<usize>, Vec<f64>) {
        let n = self.n;
        let mut pinned = vec![false; n];
        let mut pins: Vec<usize> = Vec::new();
        let mut pv: Vec<f64> = Vec::new();
        for &i in &self.pins_prev {
            if self.u_prev.len() != n || i >= n {
                continue;
            }
            if (self.u_prev[i] - lb[i]).abs() < 1e-9 {
                pinned[i] = true;
                pins.push(i);
                pv.push(lb[i]);
            } else if (self.u_prev[i] - ub[i]).abs() < 1e-9 {
                pinned[i] = true;
                pins.push(i);
                pv.push(ub[i]);
            }
        }
        (pinned, pins, pv)
    }

    fn g_matvec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut v = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            let base = i * n;
            for j in 0..n {
                s += self.g_inv.data[base + j] * x[j];
            }
            v[i] = s;
        }
        v
    }

    fn g_col(&self, i: usize) -> Vec<f64> {
        let n = self.n;
        let mut col = vec![0.0; n];
        for r in 0..n {
            col[r] = self.g_inv.data[r * n + i];
        }
        col
    }

    #[allow(clippy::too_many_arguments)]
    fn eq_solve(
        &self,
        f: &[f64],
        pins: &[usize],
        pv: &[f64],
        a: &Mat,
        wrows: &[usize],
        b: &[f64],
        pcol: &mut [Vec<f64>],
        gcol_cache: &mut [Vec<f64>],
    ) -> (Vec<f64>, Vec<f64>, bool) {
        let n = self.n;
        let np = pins.len();
        let nw = wrows.len();
        let m = np + nw;
        let gf = self.g_matvec(f);
        let mut gcols: Vec<Vec<f64>> = vec![Vec::new(); m];
        let mut cdots = vec![0.0; m];
        let mut dvals = vec![0.0; m];
        for j in 0..np {
            let ci = pins[j];
            if pcol[ci].len() != n {
                pcol[ci] = self.g_col(ci);
            }
            gcols[j] = pcol[ci].clone();
            cdots[j] = gf[ci];
            dvals[j] = pv[j];
        }
        for r in 0..nw {
            let row = wrows[r];
            if gcol_cache[row].len() != n {
                let mut c = vec![0.0; n];
                for i in 0..n {
                    c[i] = a.at(row, i);
                }
                gcol_cache[row] = self.g_matvec(&c);
            }
            gcols[np + r] = gcol_cache[row].clone();
            let mut s = 0.0;
            for i in 0..n {
                s += a.at(row, i) * gf[i];
            }
            cdots[np + r] = s;
            dvals[np + r] = b[row];
        }
        let mut mu: Vec<f64> = Vec::new();
        if m > 0 {
            let mut k = Mat::zeros(m, m);
            for i in 0..m {
                for j in 0..m {
                    let mut s = 0.0;
                    if i < np {
                        s = gcols[j][pins[i]];
                    } else {
                        let row = wrows[i - np];
                        for l in 0..n {
                            s += a.at(row, l) * gcols[j][l];
                        }
                    }
                    k.set(i, j, s);
                }
            }
            let mut rhs = vec![0.0; m];
            for i in 0..m {
                rhs[i] = -(cdots[i] + dvals[i]);
            }
            mu = k.solve(&rhs);
            for &v in &mu {
                if v.is_nan() || v.abs() > 1e300 {
                    return (Vec::new(), Vec::new(), false);
                }
            }
        }
        let mut u = vec![0.0; n];
        for i in 0..n {
            let mut s = -gf[i];
            for j in 0..m {
                s -= mu[j] * gcols[j][i];
            }
            u[i] = s;
        }
        (u, mu, true)
    }

    pub fn solve_box(&mut self, f: &[f64], lb: &[f64], ub: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut u = self.warm_u(lb, ub);
        let (mut pinned, mut pins, mut pv) = self.restore_pins(lb, ub);
        let mut pcol: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut gcol_cache: Vec<Vec<f64>> = Vec::new();
        let empty_a = Mat::zeros(0, 0);
        self.iters = 0;
        for _ in 0..400 {
            self.iters += 1;
            let (uf, mu, ok) = self.eq_solve(
                f,
                &pins,
                &pv,
                &empty_a,
                &[],
                &[],
                &mut pcol,
                &mut gcol_cache,
            );
            if !ok {
                if pins.is_empty() {
                    self.u_prev = u.clone();
                    self.pins_prev = pins.clone();
                    return u;
                }
                let last = pins[pins.len() - 1];
                pinned[last] = false;
                pins.pop();
                pv.pop();
                continue;
            }
            let mut alpha = 1.0;
            let mut block: i32 = -1;
            let mut bval = 0.0;
            for i in 0..n {
                if pinned[i] {
                    continue;
                }
                let d = uf[i] - u[i];
                if d > 0.0 && uf[i] > ub[i] + 1e-12 {
                    let step = (ub[i] - u[i]) / d;
                    if step < alpha {
                        alpha = step;
                        block = i as i32;
                        bval = ub[i];
                    }
                } else if d < 0.0 && uf[i] < lb[i] - 1e-12 {
                    let step = (lb[i] - u[i]) / d;
                    if step < alpha {
                        alpha = step;
                        block = i as i32;
                        bval = lb[i];
                    }
                }
            }
            if block >= 0 {
                let block = block as usize;
                for i in 0..n {
                    if !pinned[i] {
                        u[i] += alpha * (uf[i] - u[i]);
                    }
                }
                u[block] = bval;
                pinned[block] = true;
                pins.push(block);
                pv.push(bval);
                self.n_pin_add += 1;
                continue;
            }
            let mut worst = 0.0;
            let mut rel: i32 = -1;
            for j in 0..pins.len() {
                let i = pins[j];
                let viol = if pv[j] <= lb[i] + 1e-12 {
                    mu[j]
                } else {
                    -mu[j]
                };
                if viol > worst {
                    worst = viol;
                    rel = j as i32;
                }
            }
            if rel >= 0 && worst > 1e-9 {
                pinned[pins[rel as usize]] = false;
                pins.remove(rel as usize);
                pv.remove(rel as usize);
                continue;
            }
            self.u_prev = uf.clone();
            self.pins_prev = pins.clone();
            return uf;
        }
        self.u_prev = u.clone();
        self.pins_prev = pins.clone();
        u
    }

    pub fn solve_ineq(
        &mut self,
        f: &[f64],
        lb: &[f64],
        ub: &[f64],
        a: &Mat,
        b: &[f64],
    ) -> Vec<f64> {
        let n = self.n;
        let m = b.len();
        let mut u = self.warm_u(lb, ub);
        self.phase1_iter = 0;
        if m > 0 {
            for _ in 0..800 {
                self.phase1_iter += 1;
                let av = a.mul_vec(&u);
                let mut viol = vec![0.0; m];
                let mut vc = 0.0;
                for i in 0..m {
                    viol[i] = 0.0f64.max(av[i] - b[i]);
                    vc += viol[i] * viol[i];
                }
                if vc < 1e-18 {
                    break;
                }
                let grad = a.transposed().mul_vec(&viol);
                let mut alpha = 1.0;
                let mut progressed = false;
                for _ in 0..24 {
                    let mut un = vec![0.0; n];
                    for i in 0..n {
                        un[i] = clamp_box(u[i] - alpha * grad[i], lb[i], ub[i]);
                    }
                    let avn = a.mul_vec(&un);
                    let mut vn = 0.0;
                    for i in 0..m {
                        let d = 0.0f64.max(avn[i] - b[i]);
                        vn += d * d;
                    }
                    if vn < vc * 0.999 {
                        u = un.clone();
                        progressed = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if !progressed {
                    break;
                }
            }
        }
        let (mut pinned, mut pins, mut pv) = self.restore_pins(lb, ub);
        let mut wrows: Vec<usize> = Vec::new();
        for r in 0..self.wrows_prev.len() {
            let r = self.wrows_prev[r];
            if r >= m {
                continue;
            }
            let mut s = -b[r];
            for i in 0..n {
                s += a.at(r, i) * u[i];
            }
            if s.abs() < 1e-6 {
                wrows.push(r);
            }
        }
        let mut pcol: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut gcol_cache: Vec<Vec<f64>> = vec![Vec::new(); m];
        self.iters = 0;
        self.n_row_add = 0;
        self.n_row_rel = 0;
        self.n_pin_add = 0;
        self.n_pin_rel = 0;
        self.n_eq_fail = 0;
        for _ in 0..300 {
            self.iters += 1;
            let (uf, mu, ok) =
                self.eq_solve(f, &pins, &pv, a, &wrows, b, &mut pcol, &mut gcol_cache);
            if !ok {
                self.n_eq_fail += 1;
                if !wrows.is_empty() {
                    self.n_row_rel += 1;
                    wrows.pop();
                    continue;
                }
                if !pins.is_empty() {
                    self.n_pin_rel += 1;
                    let last = pins[pins.len() - 1];
                    pinned[last] = false;
                    pins.pop();
                    pv.pop();
                    continue;
                }
                self.u_prev = u.clone();
                self.pins_prev = pins.clone();
                self.wrows_prev = wrows.clone();
                return u;
            }
            let mut alpha = 1.0;
            let mut block: i32 = -1;
            let mut bval = 0.0;
            for i in 0..n {
                if pinned[i] {
                    continue;
                }
                let d = uf[i] - u[i];
                if d > 0.0 && uf[i] > ub[i] + 1e-12 {
                    let step = (ub[i] - u[i]) / d;
                    if step < alpha {
                        alpha = step;
                        block = i as i32;
                        bval = ub[i];
                    }
                } else if d < 0.0 && uf[i] < lb[i] - 1e-12 {
                    let step = (lb[i] - u[i]) / d;
                    if step < alpha {
                        alpha = step;
                        block = i as i32;
                        bval = lb[i];
                    }
                }
            }
            if block >= 0 {
                let block = block as usize;
                for i in 0..n {
                    if !pinned[i] {
                        u[i] += alpha * (uf[i] - u[i]);
                    }
                }
                u[block] = bval;
                pinned[block] = true;
                pins.push(block);
                pv.push(bval);
                self.n_pin_add += 1;
                continue;
            }
            u = uf.clone();
            let av = a.mul_vec(&u);
            let mut worst_r: i32 = -1;
            let mut worst_v = 1e-9;
            for r in 0..m {
                if !wrows.contains(&r) && av[r] - b[r] > worst_v {
                    worst_v = av[r] - b[r];
                    worst_r = r as i32;
                }
            }
            if worst_r >= 0 {
                wrows.push(worst_r as usize);
                self.n_row_add += 1;
                continue;
            }
            let mut rel_any = false;
            let mut keep: Vec<usize> = Vec::new();
            for r in 0..wrows.len() {
                if mu[pins.len() + r] < -1e-9 {
                    rel_any = true;
                } else {
                    keep.push(wrows[r]);
                }
            }
            if rel_any {
                self.n_row_rel += wrows.len() - keep.len();
                wrows = keep.clone();
                continue;
            }
            let mut worst_p = 0.0;
            let mut rel_p: i32 = -1;
            for j in 0..pins.len() {
                let i = pins[j];
                let viol = if pv[j] <= lb[i] + 1e-12 {
                    mu[j]
                } else {
                    -mu[j]
                };
                if viol > worst_p {
                    worst_p = viol;
                    rel_p = j as i32;
                }
            }
            if rel_p >= 0 && worst_p > 1e-9 {
                pinned[pins[rel_p as usize]] = false;
                pins.remove(rel_p as usize);
                pv.remove(rel_p as usize);
                self.n_pin_rel += 1;
                continue;
            }
            self.u_prev = u.clone();
            self.pins_prev = pins.clone();
            self.wrows_prev = wrows.clone();
            return u;
        }
        self.u_prev = u.clone();
        self.pins_prev = pins.clone();
        self.wrows_prev = wrows.clone();
        u
    }
}
