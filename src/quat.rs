use crate::mat::Mat;
use crate::vec3::Vec3;

/// wxyz — w is the scalar part, the ordering this API fixes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Shepperd's method, with the 180-degree fallback the trace branch needs: its divisor
    /// 2*sqrt(1 + trace) vanishes at pi.
    pub fn from_mat3(r: &Mat) -> Quat {
        let tr = r.at(0, 0) + r.at(1, 1) + r.at(2, 2);
        let mut q;
        if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            q = Quat {
                w: 0.25 * s,
                x: (r.at(2, 1) - r.at(1, 2)) / s,
                y: (r.at(0, 2) - r.at(2, 0)) / s,
                z: (r.at(1, 0) - r.at(0, 1)) / s,
            };
        } else if r.at(0, 0) > r.at(1, 1) && r.at(0, 0) > r.at(2, 2) {
            let s = (1.0 + r.at(0, 0) - r.at(1, 1) - r.at(2, 2)).max(0.0).sqrt() * 2.0;
            q = Quat {
                w: (r.at(2, 1) - r.at(1, 2)) / s,
                x: 0.25 * s,
                y: (r.at(0, 1) + r.at(1, 0)) / s,
                z: (r.at(0, 2) + r.at(2, 0)) / s,
            };
        } else if r.at(1, 1) > r.at(2, 2) {
            let s = (1.0 + r.at(1, 1) - r.at(0, 0) - r.at(2, 2)).max(0.0).sqrt() * 2.0;
            q = Quat {
                w: (r.at(0, 2) - r.at(2, 0)) / s,
                x: (r.at(0, 1) + r.at(1, 0)) / s,
                y: 0.25 * s,
                z: (r.at(1, 2) + r.at(2, 1)) / s,
            };
        } else {
            let s = (1.0 + r.at(2, 2) - r.at(0, 0) - r.at(1, 1)).max(0.0).sqrt() * 2.0;
            q = Quat {
                w: (r.at(1, 0) - r.at(0, 1)) / s,
                x: (r.at(0, 2) + r.at(2, 0)) / s,
                y: (r.at(1, 2) + r.at(2, 1)) / s,
                z: 0.25 * s,
            };
        }
        if q.w < 0.0 {
            q = Quat {
                w: -q.w,
                x: -q.x,
                y: -q.y,
                z: -q.z,
            };
        }
        q
    }

    /// World frame.
    pub fn rotvec_between(target: Quat, current: Quat) -> Vec3 {
        let rt = target.to_mat3();
        let rc = current.to_mat3();
        rt.mul(&rc.transposed()).to_rotvec()
    }

    /// A method rather than `impl Mul`, for the reason Vec3::add records.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    /// Local -> world, the direction mul and rotvec_between are written against.
    pub fn to_mat3(self) -> Mat {
        let mut m = Mat::zeros(3, 3);
        self.to_mat3_into(&mut m);
        m
    }

    /// To keep a 3 x 3 allocation out of a per-call loop.
    pub fn to_mat3_into(self, out: &mut Mat) {
        if out.rows != 3 || out.cols != 3 {
            *out = Mat::zeros(3, 3);
        }
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        out.set(0, 0, 1.0 - 2.0 * (y * y + z * z));
        out.set(0, 1, 2.0 * (x * y - z * w));
        out.set(0, 2, 2.0 * (x * z + y * w));
        out.set(1, 0, 2.0 * (x * y + z * w));
        out.set(1, 1, 1.0 - 2.0 * (x * x + z * z));
        out.set(1, 2, 2.0 * (y * z - x * w));
        out.set(2, 0, 2.0 * (x * z - y * w));
        out.set(2, 1, 2.0 * (y * z + x * w));
        out.set(2, 2, 1.0 - 2.0 * (x * x + y * y));
    }
}

impl Default for Quat {
    /// IDENTITY, because a zeroed quaternion is not a rotation; this also lets `#[derive(Default)]`
    /// reach a struct that carries a Quat.
    fn default() -> Self {
        Self::IDENTITY
    }
}
