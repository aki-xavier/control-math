// vec3.rs — Vec3, a 3D Euclidean vector, and its operations.
//
// All state is carried by the struct; operations are methods, so the arithmetic reads as
// object-oriented vector math. Constructors are associated functions and constants
// (`Vec3::new`, `Vec3::ZERO`), the way the rest of the crate's types read.

/// Vec3 is a 3D Euclidean vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// ZERO is the additive identity.
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// from_slice reads up to three leading components, the rest staying zero.
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

    // add is self + o.
    //
    // A method rather than `impl Add`: every arithmetic expression in this crate is written out
    // expression by expression, and the suite's numbers are those expressions' results. Operators
    // would rewrite ~1,000 call sites for a spelling change.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }

    // sub is self - o; a method for the reason add records.
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

    /// to_array returns the components as an array.
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// perp returns an arbitrary unit vector perpendicular to self (deterministic;
    /// smallest-component pivot, sign fixed by the cross product).
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
