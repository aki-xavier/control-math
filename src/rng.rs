// rng.rs — deterministic Mersenne Twister (MT19937, seed 0 default) with a
// Box-Muller Gaussian pair for the noise-driven estimator benchmarks.
//
// The stream is deterministic for a given seed, which is what makes the
// benchmarks reproducible.
//
// The generator is `rand_mt`'s `Mt`, the reference MT19937 algorithm (init_genrand's seeding, the
// reference twist including its last-element read of the already-updated state[0], and the
// reference tempering), which is why the streams agree: the seeded figures and every test that
// draws from this are unchanged, byte for byte. What stays here is the two MAPPINGS on top of the
// raw 32-bit draws, `next_f64` and `randn`, because the pinned numbers are made of those.

use rand_mt::Mt;

/// Mt19937 is the generator state plus the Gaussian spare.
pub struct Mt19937 {
    inner: Mt,
    has_spare: bool,
    spare: f64,
}

impl Mt19937 {
    /// new seeds the generator (custom seed convention; the stream is
    /// deterministic for a given seed, which is all the benchmarks need).
    pub fn new(seed: u32) -> Mt19937 {
        Mt19937 {
            inner: Mt::new(seed),
            has_spare: false,
            spare: 0.0,
        }
    }

    /// next_f64 returns a uniform double in [0, 1) with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        let a = f64::from(self.inner.next_u32() >> 5);
        let b = f64::from(self.inner.next_u32() >> 6);
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// randn returns a standard normal draw (Box-Muller, spare value caching).
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
