# PickingSystem Helper Modules

> Pure helpers split from `../picking-system.ts` so the orchestrator stays focused on lifecycle (RAF scheduling, renderer binding, event wiring) and the load-bearing math/policy is unit-testable in isolation.

Each module owns one concern. None of them hold a back-reference to the
orchestrator — they take their state as plain arguments (maps, scratch
buffers, timestamps) so they can be exercised directly from unit tests
without spinning up a renderer.

## Module map

| File                  | Role                                                                                                                                                                                                                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `registration.ts`     | `PickNodeEntry` (the `{ main, pick }` pair the orchestrator tracks), `disposePickMaterial(mesh)`, `unregisterAllPickMaterials(nodeMap)` — drops pick materials from `materialManager` without disposing the GPU object, for the post-context-loss rebuild path                                       |
| `ray-aabb.ts`         | World-space AABB cache + `rayHitsAnyNode(ray, nodeMap, cache)` early-out. `getOrComputeWorldBox` lazily projects a node's local `boundingBox` through `matrixWorld`; `invalidateBoxCache(cache, pickId?)` drops one or all entries                                                                   |
| `pick-render.ts`      | `voteWinner(pixels, pickSize, votesScratch)` — brightness-weighted majority vote across the 5×5 pixel readback. Returns the `(nodeId, elementId, accumulated brightness)` triple or `null`                                                                                                           |
| `settle-loop.ts`      | `HOVER_SETTLE_MS = 120` plus `evaluateSettle({ now, lastMouseMoveTime, lastDirtyTime, lastPickFiredTime })` returning `{ action: 'fire' \| 'wait' \| 'idle' }`                                                                                                                                       |
| `settle-scheduler.ts` | `SettleScheduler` — owns the rAF lifecycle and the mouse/dirty timestamps. Forwards each tick's decision to `evaluateSettle` and invokes `ctx.firePick` when both axes settle. The orchestrator hands it a narrow `SettleSchedulerCtx` (no `this` back-pointer)                                      |
| `lens-distortion.ts`  | `applyLensDistortion(u, v, params, out)` — TypeScript port of the green-channel Brown–Conrady distortion from `post-processing/mega/shader.glsl.ts::applyDistortion`. Lets the picking system map mouse coords into the undistorted pick buffer when the mega-shader is distorting the visible frame |

## Why this split exists

`picking-system.ts` was carrying four unrelated responsibilities at once —
material lifecycle, ray culling, pixel voting, settle-window decisioning,
and lens-distortion math — interleaved with the RAF state machine that
drives all of them. Pulling the pure pieces out lets each be tested
against a fixture without a `WebGLRenderer`, and keeps the orchestrator
short enough to read end-to-end. The orchestrator still owns every piece
of mutable state; the helpers are stateless except for the maps that the
orchestrator owns and passes in.

## Pick-material registration

`unregisterAllPickMaterials` is the load-bearing piece of `registration.ts`:
after a WebGL context loss, `MaterialManager` performs a "soft dispose"
(see `SOFT_DISPOSE_FLAG` in `../../material-manager/lifecycle.ts`) so the
visual side can rebuild without the registry getting double-released. The
picking side mirrors that contract — it drops every pick material from
`materialManager`'s `CameraAwareMaterial` broadcast list **without**
calling `.dispose()`, because the underlying GPU object is about to be
rebuilt and resubscribed. `disposePickMaterial` is the symmetric helper
for the normal `unregisterNode` path, where the GPU object really is
going away. The pair keeps the "should this dispose?" decision in one
place per code path.

## Ray-AABB world-box cache

The cursor spends most of its time over empty canvas. The cheap pre-check
in `ray-aabb.ts` projects each registered node's local `boundingBox`
through its current `matrixWorld` once and caches the resulting
`THREE.Box3`; subsequent picks reuse the cached box until the orchestrator
explicitly invalidates it (on `registerNode`, `unregisterNode`, or a
geometry commit). Camera motion does **not** invalidate the cache — the
boxes live in world space and only depend on the _object's_ transform.
`rayHitsAnyNode` short-circuits at the first intersection so the cost is
O(N) only when the ray genuinely misses every node.

## Brightness-weighted majority vote

`voteWinner` reads a 5×5 block from the float32 pick target where each
pixel encodes `(nodeId, elementId-low16, brightness, elementId-high16)`.
Background pixels (`r < 0.5`) are skipped; the rest are tallied into the
caller-supplied `votesScratch` map, accumulating brightness as the vote
weight. The brightest total wins. The 5×5 footprint is deliberate: it gives a
tiny amount of slack so a single-pixel-wide point or line edge still picks
reliably, but it's small enough that the brightness weighting still resolves
overlapping splats by intensity rather than by
which-fragment-was-drawn-last.

The map key is `nodeId * 2^27 + elementId` — a stride that both exceeds every
reachable element index (44,728,319 at max capacity) and keeps the product
exactly representable for any `nodeId` the pick buffer can resolve (see
`VOTE_KEY_STRIDE` and `MAX_PICK_NODE_ID`, pinned against the live layout
maxima by a unit test).

## Hover settle decision

The two-axis settle is the whole reason picking feels "non-twitchy".
`evaluateSettle` reports `wait` while either the mouse or the camera /
geometry has moved within the last `HOVER_SETTLE_MS` (120 ms); once both
axes are quiet it reports `fire` if anything has changed since the last
pick, or `idle` if nothing has changed. The orchestrator uses `idle` as
its cue to stop scheduling RAF ticks — a future `markDirty` or
`mousemove` rearms the loop. The 120 ms window matches the standard
tooltip-appearance delay used by browsers and IDEs, so the latency lands
in the "intentional" rather than "laggy" range. The threshold is the only
tunable in the file; everything else falls out of the four-timestamp
input.

## Lens-distortion parity

`lens-distortion.ts` exists to keep screen-space picking aligned with the
mega-shader's Brown–Conrady distortion. The TS port mirrors the GLSL
green-channel formula in `post-processing/mega/shader.glsl.ts` (`applyDistortion`; note the GLSL negates skew/principalPoint.y — the flip conjugation for its bottom-up uv)
(and the TSL counterpart in `post-processing/mega/shader.tsl.ts`); the orchestrator runs it on
the raw mouse UV before computing the pick-target read coords so the
pixel sampled matches what the user sees under the distorted frame. The
function takes an explicit `UVScratch` so the hot path is allocation-free.

## See Also

- `../picking-system.ts` — orchestrator that imports these five modules
- `../../material-manager/lifecycle.ts` — `SOFT_DISPOSE_FLAG` symbol that
  `unregisterAllPickMaterials` cooperates with on context-loss rebuild
- `../../post-processing/mega/shader.glsl.ts` — GLSL `applyDistortion`
  that `lens-distortion.ts` must stay byte-for-byte equivalent to
- `../../materials/_shared/camera-uniforms.ts` — sibling parity case
  (picking math centralised so screen-space hit tests match rendering)
- `../README.md` — picking subsystem overview (the orchestrator + the
  per-geometry material trios that wrap these helpers)
- `../../README.md` — rendering package overview (picking is one
  component in the larger pipeline)
