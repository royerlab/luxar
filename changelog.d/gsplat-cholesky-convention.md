#### The hand-authoring docs said `1/σ` where the format means `σ`

`cholesky_factors` is the packed lower-triangular factor **L of the covariance**
(Σ = L·Lᵀ), so an isotropic Gaussian of standard deviation σ is
`[σ, 0, σ, 0, 0, σ]` and the diagonal is scale-like — bigger numbers, bigger
splats. Three agent-skill files and `gsplats_basic_example.py` documented the
inverse reading, L as the factor of the *precision* matrix Σ⁻¹, and told the
reader to pack `1/σ`.

Nothing raised. An inverted diagonal is still positive, so the writer's Cholesky
gate passes and a perfectly valid store lands whose splats are wrong by `1/σ²`
in linear extent — a screen-filling wash for the sub-unit σ of a normalized
scene, an invisible scene for large-σ voxel-space data. Both read as "the viewer
is broken". The example was the concrete casualty: its "elongated along Y" splat
was authored with `sy=1.6` and stored `0.625`, so the splat was actually
*squashed* along Y and contradicted its own on-screen explainer text.

The example's three helpers now pack the covariance factor directly, and its
`tilted_cholesky` docstring states which way the off-diagonal moves the extent
(`Σ[1,1] = L[1,0]² + L[1,1]²`, so a coupling *widens* Y rather than narrowing
it).

Nothing tied the documented convention to the implementation, which is why the
drift survived. Round-trip tests assert that stored bytes decode back to the
input array — convention-blind, since they round-trip `1/σ` just as faithfully —
and the WASM/TypeScript parity tests feed a synthetic L and check both backends
agree on Σ = L·Lᵀ, which says nothing about what the author meant by σ. Two new
test files close the gap from both ends: `test_cholesky_convention.py` authors σ
through `GSplatData` and a full scene round-trip and asserts σ comes back, and
`test_gsplats_basic_example_convention.py` asserts the example's own helpers
against Σ = L·Lᵀ. Every fixture uses σ ≠ 1 deliberately, because σ == 1/σ at
1.0: a unit-σ smoke test passes under either convention and is exactly what let
this go unnoticed.
