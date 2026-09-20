// `rand_mt`'s `Mt` is used unchanged, so the stream agrees with the reference MT19937 and a draw
// reproduces byte for byte; `next_f64` and `randn` are the only additions.

use rand_mt::Mt;

pub struct Mt19937 {
    inner: Mt,
    has_spare: bool,
    spare: f64,
}

impl Mt19937 {
    /// The seed alone fixes the stream: the same seed replays byte for byte.
    pub fn new(seed: u32) -> Mt19937 {
        Mt19937 {
            inner: Mt::new(seed),
            has_spare: false,
            spare: 0.0,
        }
    }

    /// 53 bits, so the granularity matches an f64 mantissa exactly.
    pub fn next_f64(&mut self) -> f64 {
        let a = f64::from(self.inner.next_u32() >> 5);
        let b = f64::from(self.inner.next_u32() >> 6);
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// The Box-Muller pair's second value is kept rather than thrown away.
    pub fn randn(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let rad = (-2.0 * u1.ln()).sqrt();
        let ang = 2.0 * std::f64::consts::PI * u2;
        self.spare = rad * ang.sin();
        self.has_spare = true;
        rad * ang.cos()
    }
}
