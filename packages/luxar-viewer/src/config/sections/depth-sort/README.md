# Depth-Sort Config Section

Scheduling knobs for camera-triggered gsplat depth re-sorts (depth-sorting
Phase 3, `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §6).

Gaussian splats in the order-dependent `normal` blending mode are sorted
back-to-front by the async SortWorker at commit time (Phase 2). This section
tunes the per-frame scheduler (`rendering/depth-sort-coordinator.ts` →
`evaluateDepthSortPerFrame`) that keeps the ordering tracking the camera:

| Knob                  | Default | Meaning                                                                                                         |
| --------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `enabled`             | `true`  | Master switch; `false` pins the identity (storage) order. URL escape hatch: `?depthSort=0`.                     |
| `angleThresholdDeg`   | `3`     | Re-sort when the node-relative view axis rotates past this angle.                                               |
| `translationFraction` | `0.05`  | Re-sort when the camera translates along the view axis past this fraction of the node's bounding-sphere radius. |

The sort kernel orders by view-space z, so the permutation depends only on
the model-space view axis direction and its offset: rotation changes the
order, view-axis translation changes the behind-camera set, and orthogonal
translation cannot change either — the scheduler deliberately ignores it.

Files follow the standard section shape: `types.ts` (interface), `data.ts`
(defaults), `validate.ts` (startup validation), registered in
`config/index.ts`, `config/types.ts`, and `config/validation.ts`.
