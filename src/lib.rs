// control-math — a self-contained library of arithmetic, and nothing else.
//
// Six modules: `vec3` and `quat` (the geometric primitives), `mat` (a dense row-major matrix with
// its constructors, decompositions and block helpers), `rng` (a deterministic MT19937 with the
// Box-Muller Gaussian pair), `qp` (the constant-Hessian QP solvers — box_qp / qp_ineq, the
// cold-start active-set reference, and ConstHessianQp, the warm-started working-set fast path) and
// `lstsq` (damped least squares over a Jacobian: the left- and right-inverse ridge solves of
// (J'J + lambda I) x = J' e).
//
// It carries no model, no plant and no engine. The imports run one way and no further: `mat` on
// `vec3`, `quat` on both, `qp` and `lstsq` on `mat`, and `rng` on `rand_mt` alone.

pub mod lstsq;
pub mod mat;
pub mod qp;
pub mod quat;
pub mod rng;
pub mod vec3;
