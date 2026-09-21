#### Preserve shear in node transforms

The viewer decomposed each node's authored 4×4 into position / quaternion /
scale. That factorisation cannot represent shear, so an ordinary
`compose(rotate, non_uniform_scale)` — which `luxar.transforms` advertises and
`examples/transform_example.py` authors — was silently rendered as different
geometry. The full affine matrix is now installed directly.

The end-to-end coverage for this was not evidence. Every assertion in
`transform-hierarchy.spec.ts` was a "differs from the identity" check, all of
which stay green against the decomposing implementation — confirmed by
reverting the fix and watching all nine tests pass. The spec now decomposes the
delivered matrix and recomposes it: for an authored `rotate ∘ non-uniform
scale` that round trip must fail, and against the old code it succeeded to
1.1e-16.

Two supporting fixes came out of running that suite. Playwright's example
fixture status is tri-state but reached the workers as a single stale/not-stale
boolean, so a checkout with no `datasets/examples` at all looked healthy and
every spec that needed one died on a bare 45 s readiness timeout with no stated
cause; `missing` is now its own state carrying the directory and the remedy.
And `applyTransform`'s reliance on the root scene's own `matrixAutoUpdate` to
re-propagate world matrices — it turns `matrixAutoUpdate` off on the node, and
applies transforms before parenting — now has unit coverage, since the
Playwright suite that would notice does not run in CI.
