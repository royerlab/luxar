# GPU Picking Subsystem

> Cached half-res RGBA32F pick buffer + per-geometry pick materials + a two-axis hover-settle scheduler that fires async pixel readbacks only when the world is quiet.

## Overview

Picking answers "which `(nodeId, elementId)` is under the cursor?" by rendering a parallel scene to an offscreen RGBA32F target where each fragment encodes `(nodeId, elementId-low, brightness, elementId-high)`. A 5×5 readback at the cursor is then resolved by brightness-weighted majority vote (see `picking-system/pick-render.ts`).

The orchestrator (`picking-system.ts`) owns lifecycle, the cached pick target, and the renderer-state save/restore around `renderPickBuffer()`. The rAF settle scheduler lives in `picking-system/settle-scheduler.ts` and the pure decision in `picking-system/settle-loop.ts`. Each geometry kind contributes a matched `(GLSL, TSL)` material pair plus a TSL factory that re-uses the wrapper-owned `UniformNode` references — no `.onUpdate('render')` bridge. The four files per geometry (`material.ts`, `material-tsl.ts`, `pick.tsl.ts`, `shaders.ts`) live together in a per-geometry subfolder. Both backends are kept in lock-step by `tsl-shader-parity.spec.ts`.

The pick buffer is **cached**: it re-renders only when `markDirty()` is called (camera move, geometry commit, resize). Hover motion over a static scene costs one async readback per settle event, no GPU rasterisation. Cursor-over-empty-canvas is short-circuited by a ray vs cached-world-AABB pre-check.

## Architecture

```
picking/
├── picking-system.ts            # Orchestrator: cached pick target, renderer-state
│                                #   save/restore, async readback, ray-AABB cull,
│                                #   vote dispatch, dirty tracking
│
├── picking-system/              # Pure helpers split out of picking-system.ts —
│                                #   ray-AABB cache, vote, settle decision +
│                                #   scheduler, registration, lens-distortion.
│                                #   See child README.
│
├── point/                       # Per-geometry picking sources for points
│   ├── material.ts              #   GLSL ShaderMaterial wrapper (80%-radius truncation)
│   ├── material-tsl.ts          #   WebGPU NodeMaterial counterpart
│   ├── pick.tsl.ts              #   TSL node factory
│   └── shaders.ts               #   GLSL3 vertex/fragment source + ShaderSource record
│
├── line/                        # idem for lines (sharpness fast-path define mirrored,
│   ├── material.ts              #   full pick width, isOrtho-aware webgpu factory)
│   ├── material-tsl.ts
│   ├── pick.tsl.ts
│   └── shaders.ts
│
├── gsplat/                      # idem for gsplats (1.5σ truncation, max-projection,
│   ├── material.ts              #   invalidCov2D parity guards)
│   ├── material-tsl.ts
│   ├── pick.tsl.ts
│   └── shaders.ts
│
├── mesh/                        # idem for meshes, plus two files no sibling needs:
│   ├── material.ts              #   NOT camera-aware; `side` synced from the visual
│   ├── material-tsl.ts          #   material; element id from gl_VertexID
│   ├── pick.tsl.ts
│   ├── shaders.ts
│   ├── pick-mode.ts             #   MeshPickAwareMaterial + the mode → (cutout, depth) map
│   └── provoking-vertex.ts      #   aligns WebGL's flat provoking vertex with WebGPU's
│
└── PICKING_DESIGN.md            # Backend readback strategy + 1-frame-latency rationale
```

## Per-geometry picking parity

Each geometry has one GLSL wrapper and one TSL wrapper. The first three implement the shared `CameraAwareMaterial` contract from `../materials/_shared/camera-aware-material.ts`; **mesh deliberately does not** — it has no screen-space footprint to size, so there is nothing for a camera broadcast to update (`material-manager.ts` routes it to `staticMaterials` instead, exactly as it does the visual mesh material). The TSL wrapper owns the `UniformNode`s and exposes them through `proxyIUniform` so `material.uniforms.uX.value = …` writes land directly on the node — symmetric with the visual `PointTSLMaterial` / `LineTSLMaterial` / `GSplatTSLMaterial` plumbing.

| Geometry | GLSL wrapper            | TSL wrapper                | TSL factory          | Pick footprint vs visual                                    |
| -------- | ----------------------- | -------------------------- | -------------------- | ----------------------------------------------------------- |
| Points   | `PointPickingMaterial`  | `PointPickingTSLMaterial`  | `point/pick.tsl.ts`  | **80% radius** (biased toward the bright core)              |
| Lines    | `LinePickingMaterial`   | `LinePickingTSLMaterial`   | `line/pick.tsl.ts`   | **Full width** (thin lines, super-Gaussian profile)         |
| GSplats  | `GSplatPickingMaterial` | `GSplatPickingTSLMaterial` | `gsplat/pick.tsl.ts` | **1.5σ** truncation (vs the visual default 2.75σ), max-proj |
| Mesh     | `MeshPickingMaterial`   | `MeshPickingTSLMaterial`   | `mesh/pick.tsl.ts`   | **Identical** — the same triangles, the same alpha cutout   |

All four fragment shaders write `vec4(vNodeId, vElementId.x, brightness, vElementId.y)` — where `vNodeId` is the `uNodeId` uniform and `vElementId` is the ordering index split into two 16-bit halves by `luxarElementIdParts()` (low in `.x`, high in `.y`), both carried as `flat` varyings — and set `gl_FragDepth = 1.0 - clamp(brightness, 0, 1)` — brightness-as-depth, so the brightest overlapping fragment wins the depth test for hover-through-translucent stacks. The split exists because float32 has a 24-bit mantissa while a node's capacity reaches 2^25 on a 32768-texel device, so one channel could not represent large indices exactly; both halves are <= 65535 and the decoder recombines them. Vertex shaders mirror visual-side sanitization (`sanitizePositive` / `sanitizeNonNegative`) and near-plane culling so the pick footprint cannot diverge from the visible footprint.

**Mesh diverges on three of those points, and each is documented where it happens** (`mesh/README.md`, spec §6.5): its `vElementId` is a per-VERTEX ordinal from `gl_VertexID` rather than an ordering index (there is no depth sort to indirect through, and a face ordinal would be renumbered on every slice change), it writes REAL projected depth under the `opaque`/`normal` surface modes instead of brightness-as-depth, and in `opaque` it applies the visual shader's alpha cutout so a hole you can see through is neither pickable nor depth-occluding. The 16-bit split itself is shared: `luxarElementIdParts()` is now a one-line wrapper over `luxarElementIdSplit(uint)`, which mesh calls directly.

**Surface-mode exception (gsplats).** Brightness-as-depth is right for the commutative blending modes (additive/max/luminous), but under the depth-ordered surface modes — depth-sorted alpha-over (`normal`) and depth-written `opaque` — the user sees an occluding surface: brightest-wins could pick a brighter splat _behind_ that surface. The gsplat pick shaders therefore carry a `uSurfaceDepth` uniform: when 1, the fragment writes the real projected depth (`gl_FragCoord.z` / the TSL `depth` builtin) so the **front-most** splat wins. `PickingSystem.renderPickBuffer()` syncs the flag per node from the main material's `userData.blendingMode` via `setSurfacePickDepth(isNormalMode(mode) || isOpaqueMode(mode))` — only the gsplat pick wrappers implement the method; points/lines are unaffected.

**Mixed-mode limitation.** All effectively visible registered nodes (hidden/demoted LOD levels are skipped) render into ONE shared pick depth buffer (pre-existing single-buffer design). A scene mixing `normal`-mode gsplats with additive/max/luminous nodes therefore mixes two depth conventions in that buffer: where footprints from both conventions overlap, the depth comparison is between a real projected depth and a `1 - brightness` pseudo-depth, so cross-mode boundary pixels resolve arbitrarily. Within a single convention (all-normal or all-commutative overlap) picking stays well-defined.

## Cached pick buffer + two-axis settle

`PickingSystem` keeps one `WebGLRenderTarget(RGBA32F, NearestFilter)` sized at `min(drawBuf / 2, MAX_PICK_BUFFER_DIM=1024)` per axis. `_dirty` gates re-rendering — only a `markDirty()` (camera move, geometry commit, resize, context restore) or a setSize triggers a fresh `renderPickBuffer()`. Hover motion alone just rereads the cached target.

A pick fires from the rAF loop only when **both** axes have been still for `HOVER_SETTLE_MS = 120 ms` and at least one axis changed since the last pick. See `picking-system/settle-loop.ts` for the decision table. The orchestrator wires `markDirty` to camera controllers, the geometry-commit pipeline, and `ResizeObserver`. `suppress(true)` cancels picking entirely during orbit/pan/zoom; releasing it re-arms the loop so a camera-settle re-pick fires naturally without requiring a mouse wiggle.

Cursor-over-empty-canvas is rejected before readback by `rayHitsAnyNode` against per-node cached world-space AABBs (`picking-system/ray-aabb.ts`). The box cache survives camera motion — only `registerNode` / `unregisterNode` / `invalidateBoxes(id)` drops entries.

## Async readback (both backends)

Pixel readback runs through `../post-processing/hdr/pixel-utils.ts::readPixelsCompactAsync`, which:

- Uses `readRenderTargetPixelsAsync` on both `WebGLRenderer` and `WebGPURenderer` (r184).
- Compacts WebGPU's padded row layout transparently.
- Returns top-down rows on both backends (the orchestrator computes `_lastReadX/Y` in top-down coords and passes them through unchanged).

The 1-frame latency this incurs is documented in `PICKING_DESIGN.md`; the settle delay absorbs it completely for the common hover case. Pre-allocated `_readDst` / `_readFlipped` buffers and a reused `_votes` map keep the hot path allocation-free.

## Lens-distortion parity

When the mega-shader is applying Brown–Conrady distortion, the visible frame and the (undistorted) pick buffer diverge. `picking-system/lens-distortion.ts::applyLensDistortion` is a TS port of the GLSL green-channel formula in `../post-processing/mega/shader.glsl.ts`; the orchestrator runs it on the raw mouse UV before computing pick-target read coords so the pixel sampled matches what the user sees. The two implementations are kept byte-for-byte equivalent.

## Context loss

On WebGL `webglcontextrestored`, `NodeFactory.rebuildAfterContextRestore` calls `PickingSystem.clearRegistrationsForRebuild()`. That helper drops every pick material from `materialManager`'s broadcast list **without** calling `.dispose()` (the GPU object is already invalid), matching the `SOFT_DISPOSE_FLAG` contract in `../material-manager/lifecycle.ts`. The caller then re-registers every node, producing fresh pick materials against the new context.

WebGPU device loss is currently treated as unrecoverable — see `scene-manager.ts::setupContextLossHandling`.

## Subpackages

- [`picking-system/`](./picking-system/README.md) — Pure helpers split from the orchestrator: `registration.ts`, `ray-aabb.ts`, `pick-render.ts` (vote), `settle-loop.ts` (pure decision), `settle-scheduler.ts` (rAF lifecycle), `lens-distortion.ts`. Each is independently unit-testable without a renderer.
- [`point/`](./point/README.md), [`line/`](./line/README.md), [`gsplat/`](./gsplat/README.md), [`mesh/`](./mesh/README.md) — Per-geometry picking sources. Each folder ships the same four files: `material.ts` (GLSL3), `material-tsl.ts` (WebGPU), `pick.tsl.ts` (TSL factory), `shaders.ts` (GLSL3 source + `ShaderSource`). `mesh/` adds two: `pick-mode.ts` and `provoking-vertex.ts`.

## See Also

- `../README.md` — Rendering package overview (picking sits alongside the visual material stacks in the larger pipeline)
- `../materials/_shared/camera-aware-material.ts` — Shared `CameraAwareMaterial` interface that all six picking materials implement
- `../materials/_shared/shader-source.ts` — `ShaderSource` shape used by each `<geometry>/shaders.ts` to ship a GLSL+TSL pair
- `../materials/_shared/glsl-lib.ts` — `GLSL_SANITIZE_FUNCTIONS` used in all four pick vertex shaders for parity with their visual counterparts, and `GLSL_ELEMENT_ID_SPLIT`, the single source of the 16-bit id split (mesh injects it alone; the other three get it inside `GLSL_SORTED_INDEX`)
- `../post-processing/hdr/pixel-utils.ts` — `readPixelsCompactAsync` (unified WebGL2/WebGPU readback)
- `../post-processing/mega/shader.glsl.ts` — GLSL `applyDistortion` that `picking-system/lens-distortion.ts` must stay byte-for-byte equivalent to
- `../../tests/e2e/harnesses/tsl-harness.ts` — Imports `POINT_PICK_SOURCE` / `LINE_PICK_SOURCE` / `GSPLAT_PICK_SOURCE` directly from each `<geometry>/shaders.ts` to drive the GLSL/TSL parity tests
- `PICKING_DESIGN.md` — Full backend rationale for the 1-frame async-readback latency
