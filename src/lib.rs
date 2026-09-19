// control-math — the math basis of the control stack, as a project of its own.
//
// Six modules and nothing else: `vec3` and `quat` (the geometric primitives), `mat` (a dense
// row-major matrix with the constructors, decompositions and block helpers the control core and the
// benches share), `rng` (a deterministic MT19937 with the Box-Muller pair the noise-driven estimator
// benches draw from), `qp` (the constant-Hessian QP solvers — box_qp/qp_ineq, the cold-start
// active-set reference, and ConstHessianQp, the warm-started MPC fast path) and `task_space_bridge`
// (the damped least-squares task-to-joint map, (J'J + lambda I) dq = J' e). It carries no model, no
// plant and no engine: every consumer in the stack — kinematics, the observers, the task loops, the
// legged stack, the benches — sits above it.
//
// It was extracted from the simu crate's `src/` (where the modules lived beside the code that
// consumes them) once the dependency graph made the order obvious: nothing here reaches above
// itself, which is why it can leave first and why everything else can follow it. simu consumes it as
// a sibling path dependency (`{ path = "../control-math" }`).
//
// `qp` and `task_space_bridge` came in a second pass, from the same tree and for the same reason:
// each is pure arithmetic over `mat`, and each was the LAST edge from the legged stack into the arm
// core (`synergy -> task_space_bridge`) and into the walk's own base layer (`whole_body_filter ->
// qp`). With them here, those two product lines share no code above this crate.

pub mod mat;
pub mod qp;
pub mod quat;
pub mod rng;
pub mod task_space_bridge;
pub mod vec3;
