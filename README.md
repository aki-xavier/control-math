# control-math — the math basis of the control stack

A project of its own: `../z1-arm` and `../g1-biped` (its consumers) depend on it as
sibling path dependencies, so no arithmetic lives in either tree and this crate can
be built, tested and released alone. MIT-licensed (see `LICENSE`).

Four modules, no model, no plant, no engine:

```text
vec3   the geometric primitive       (add/sub/scale, dot, cross, norms, perp)
quat   rotation as a unit quaternion (from_mat3, to_mat3(_into),
                                      rotvec_between, mul)
mat    a dense row-major matrix      (eye, from_rows/from_blocks, mul(_vec),
                                      solve(_mat), inv, skew, from_axis_angle,
                                      to_rotvec, symmetrized, diag)
rng    MT19937                       (the reference seeding/twist/tempering,
                                      next_f64, Box-Muller randn)
```

The modules' own docs in `src/` carry the derivations and the conventions each
one holds to. `rng` is the one module with a dependency: `rand_mt` supplies the
MT19937 core, built without its `rand-traits` feature so `rand_core` stays out of
the tree.

## Provenance

Extracted from `simu` at commit `2570eb1`, where the four files lived in
`simu`'s `src/`. It left first because the dependency graph made the order
obvious: the cluster has no internal dependencies at all, and no `crate::`
reference from inside it reaches the code that consumes it. The algebra crate
[`pga`](../pga) left before it for the same reason. The history of the files
stays readable in `simu` (`git log --follow -- src/mat.rs`).

## Tests

`tests/math.rs` is the cluster's own oracle: the numbers the rest of the stack
calibrates against, asserted here with their own tolerances — the 2×2 solve and
its inverse, the quaternion/matrix round trip, `rotvec_between` at π, the
axis-angle path against the quaternion one, `to_rotvec` as the inverse of the
exponential map, the out-of-range behaviour of `eye`/`at`, and the MT19937
stream's determinism and standard normal.

```sh
mbx test                                                    # 8 tests
mbx clippy --all-targets -- -D warnings
```

(`mbx` is the build-cache wrapper the sibling projects use for Cargo; plain
`cargo` works the same way.)
