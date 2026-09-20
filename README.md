# control-math

A small, self-contained library of arithmetic, and nothing else: vectors,
quaternions, a dense matrix, a deterministic random stream, and two solver
families. MIT-licensed (see `LICENSE`).

Six modules:

```text
vec3    the geometric primitive       (add/sub/scale, dot, cross, norms, perp)
quat    rotation as a unit quaternion (from_mat3, to_mat3(_into),
                                       rotvec_between, mul)
mat     a dense row-major matrix      (eye, from_rows/from_blocks, mul(_vec),
                                       solve(_mat), inv, skew, from_axis_angle,
                                       to_rotvec, symmetrized, diag)
rng     MT19937                       (the reference seeding/twist/tempering,
                                       next_f64, Box-Muller randn)
qp      constant-Hessian QP solvers   (box_qp, qp_ineq, ConstHessianQp)
lstsq   damped least squares          (the left- and right-inverse ridge forms of
                                       (J'J + lambda I) x = J' e)
```

The modules' own docs in `src/` carry the derivations and the conventions each
one holds to. The imports run one way and no further — `mat` on `vec3`, `quat`
on both, `qp` and `lstsq` on `mat` — so no module needs one above it.

`rng` is the one module with a dependency: `rand_mt` supplies the MT19937 core,
built without its `rand-traits` feature so `rand_core` stays out of the tree.

## Tests

`tests/math.rs` and `tests/numerics.rs` are this crate's own oracle — its
numbers, asserted with their own tolerances: the 2×2 solve and its inverse, the
quaternion/matrix round trip, `rotvec_between` at π, the axis-angle path against
the quaternion one, `to_rotvec` as the inverse of the exponential map, the
out-of-range behaviour of `eye`/`at`, the MT19937 stream's determinism and
standard normal, the box projection, and the left/right-inverse identity.

```sh
make test                                                   # 12 tests, then the comment rules
mbx clippy --all-targets -- -D warnings
```

`make test` is `mbx test` and then the comment rules over the tree, so the lint is not a step anyone has
to remember. The rules are `../comment-why`, a sibling project that reads text and asks the compiler for
nothing — which is what lets the same rules also be an ordinary test here, `tests/comment_why.rs`, with
no nightly and no plugin. They decide three shapes: process narration and filler, a comment line whose
content words are all in the code below it, and a short doc comment that re-says the item's own name.
The rest is a reader's call, and `make comments` prints that crate's local approximation — long,
marker-free and mostly the code's own words — as advice it never fails on. The rules are a
DEV-dependency: nothing that depends on this crate links them.
