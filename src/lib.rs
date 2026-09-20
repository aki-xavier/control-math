// Imports run one way and no further (`mat` on `vec3`, `quat` on both, `qp` and `lstsq` on `mat`,
// `rng` on `rand_mt` alone), so no module needs one above it.

pub mod lstsq;
pub mod mat;
pub mod qp;
pub mod quat;
pub mod rng;
pub mod vec3;
