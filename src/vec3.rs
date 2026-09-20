#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn from_slice(a: &[f64]) -> Vec3 {
        let mut v = Vec3::ZERO;
        if !a.is_empty() {
            v.x = a[0];
        }
        if a.len() > 1 {
            v.y = a[1];
        }
        if a.len() > 2 {
            v.z = a[2];
        }
        v
    }

    // A method rather than `impl Add`: the arithmetic is written out expression by expression.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }

    // A method, for the reason add records.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }

    pub fn scale(self, s: f64) -> Vec3 {
        Vec3 {
            x: s * self.x,
            y: s * self.y,
            z: s * self.z,
        }
    }

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n < 1e-12 {
            return Vec3::ZERO;
        }
        self.scale(1.0 / n)
    }

    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Deterministic, so a run reproduces its basis.
    pub fn perp(self) -> Vec3 {
        let a = Vec3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        };
        if a.x <= a.y && a.x <= a.z {
            return self.cross(Vec3::new(1.0, 0.0, 0.0)).normalized();
        }
        if a.y <= a.z {
            return self.cross(Vec3::new(0.0, 1.0, 0.0)).normalized();
        }
        self.cross(Vec3::new(0.0, 0.0, 1.0)).normalized()
    }
}
