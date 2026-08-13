#### The volumetric line primitive is deleted

`?linePrimitive=volumetric` — the closed-form segment-⊛-Gaussian line model
that shipped during the #1352 campaign (exact in its core ray integral and
bisector-cut algebra, with documented bounded approximations at chain
ends, near-clip, and sharp bends) — is gone, together with its GLSL and TSL
shader pairs, the pick pair, the quadrature-validated CPU reference, the
Abel-transform sharpness LUT, and the `ray-integral` shared-math module
(about 4,600 lines plus 21 parity fixtures). Since the capsule became the
default, an A/B on the QA grid found the capsule matching or beating it
visually — including near-axial, the case the volumetric model existed to
fix — at several times less frame cost, so there is no scene left that
benefits from carrying a third primitive through every cross-cutting
material change. `linePrimitive` now accepts `capsule` (default) and
`screen-space`; an explicit `?linePrimitive=volumetric` is treated like any
other unrecognised value and falls back to the default. The full
implementation remains in git history at the deletion's branch point.
