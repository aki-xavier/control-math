// control-math — the math basis of the control stack, as a project of its own.
//
// Four modules and nothing else: `vec3` and `quat` (the geometric primitives), `mat` (a dense
// row-major matrix with the constructors, decompositions and block helpers the control core and the
// benches share), and `rng` (a deterministic MT19937 with the Box-Muller pair the noise-driven
// estimator benches draw from). It carries no model, no plant and no engine: every consumer in the
// stack — kinematics, the observers, the task loops, the legged stack, the benches — sits above it.
//
// It was extracted from the simu crate's `src/` (where the four modules lived beside the code that
// consumes them) once the dependency graph made the order obvious: this cluster has NO internal
// dependencies at all, which is why it can leave first and why everything else can follow it.
// simu consumes it as a sibling path dependency (`{ path = "../control-math" }`).

pub mod mat;
pub mod quat;
pub mod rng;
pub mod vec3;
