#### Preserve shear in node transforms

The viewer decomposed each node's authored 4×4 into position / quaternion /
scale. That factorisation cannot represent shear, so an ordinary
composition that applies a non-uniform scale after rotation — which
`luxar.transforms` supports and `examples/transform_example.py` authors — was
silently rendered as different geometry. The full affine matrix is now installed
directly.

The end-to-end coverage for this was not evidence. Every assertion in
`transform-hierarchy.spec.ts` was a "differs from the identity" check, all of
which stay green against the decomposing implementation — confirmed by
reverting the fix and watching all nine tests pass. The spec now decomposes the
delivered matrix and recomposes it: for the authored non-uniform-scale-after-
rotation matrix that round trip must fail, and against the old code it succeeded
to 1.1e-16.

Two supporting fixes came out of running that suite. Playwright's example
fixture status is tri-state but reached the workers as a single stale/not-stale
boolean, so a checkout with no `datasets/examples` at all looked healthy and
every spec that needed one died on a bare 45 s readiness timeout with no stated
cause; `missing` is now its own state carrying the directory and the remedy.
And `applyTransform` now leaves `matrixWorldNeedsUpdate` set after its synchronous
creation-time refresh, so on-demand bounds and audio updates re-resolve the world
matrix when the loader parents the node afterward. That path now has unit
coverage, since the Playwright suite that would notice does not run in CI.
