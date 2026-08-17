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
| `workerInitTimeoutMs` | `30000` | Deadline for the SortWorker's one-time `initialize()` (WASM load + instantiate). `0` disables the guard.        |

`workerInitTimeoutMs` is the odd one out — it is a startup deadline, not a
re-sort trigger, and it bounds ONE attempt rather than the worker's life. A miss
TERMINATES that worker; because a deadline miss says nothing about the worker's
health (the reply has to be dispatched on a main thread that a large scene keeps
busy), the bounded per-frame retry then spawns a FRESH worker after a backoff,
and a retry that lands forces a re-registration sweep. Shortening it therefore
buys faster failure detection at the cost of extra worker spawns + WASM
instantiations and that forced re-commit of every sorted node, which is why
it is deliberately more generous than the data pool's
`dataLoading.performance.workerInitTimeoutMs` (10 s) — erring long costs only a
later first sort. `0` installs no timer at all, so a worker that neither answers
nor errors leaves `initialize()` pending and every order-dependent commit parks
another continuation on it: a debugging escape hatch, not a tuning option. See
the failure taxonomy in `rendering/depth-sort-coordinator/README.md`.

The sort kernel orders by view-space z, so the permutation depends only on
the model-space view axis direction and its offset: rotation changes the
order, view-axis translation changes the behind-camera set, and orthogonal
translation cannot change either — the scheduler deliberately ignores it.

Files follow the standard section shape: `types.ts` (interface), `data.ts`
(defaults), `validate.ts` (startup validation), registered in
`config/index.ts`, `config/types.ts`, and `config/validation.ts`.
