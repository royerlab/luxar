# Line Material

> Thick-line material stack — instanced-quad geometry, semicircle-kernel soft falloff, and seamless additive joints — paired across the GLSL `ShaderMaterial` and TSL `NodeMaterial` backends.

This folder holds the four-file material stack that renders one of Luxar's
three first-class geometry types. Each line segment is drawn as an instanced
screen-space quad expanded perpendicular to its pixel-space direction; the
fragment stage shades a parabolic `(1 − p²)^sharpness` profile that sums to
flat full intensity at joints under additive blending. Both backends share
the same `LineMaterialConfig` shape and the same update / clone / blending
semantics — `MaterialManager.getLineMaterial` dispatches on
`RendererCapabilities.apiSurface`, so call sites never see the divergence.

## Module map

| File               | Role                                                                                                                                                                                                                                   |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material-glsl.ts` | `LineMaterial extends THREE.ShaderMaterial` — wraps the GLSL3 vertex/fragment pair, owns `uniforms`, manages variant `defines`, applies the canonical blending state. WebGL2 path.                                                     |
| `material-tsl.ts`  | `LineTSLMaterial extends NodeMaterial` — same constructor + update API, but owns persistent `UniformNode`s and rebuilds its TSL graph (`rebuildGraph`) when graph-specialized defines or projection mode flip. WebGPU path.            |
| `shader-glsl.ts`   | `LINE_VERTEX_SHADER` + `LINE_FRAGMENT_SHADER` GLSL3 source strings and the `LINE_SOURCE: ShaderSource` registry entry. The `webgpu` field re-enters `lineWebGPUFactory` so the parity harness can drive both backends from one symbol. |
| `shader-tsl.ts`    | `lineWebGPUFactory(nodes, config, outMaterial?)` — TSL counterpart to the GLSL strings. Reads pre-created `UniformNode`s from a `LineTSLNodes` table and emits the NodeMaterial graph.                                                 |

## Rendering model in one paragraph

A line is a list of segments. Each segment becomes one **instance** of a
unit quad with corners `aQuadCorner ∈ {(±1, ±1)}`: `x` selects the
interpolation parameter `t` along the segment (`0` at start, `1` at end),
`y` selects the perpendicular offset (`−1` bottom edge, `+1` top edge). The
vertex stage projects both endpoints to view space, runs near-plane safety
(degenerate-quad when both endpoints fail `uNearCull`; pathological-quad
discard when both are near AND the raw pixel width blows past
`uMaxLinePixelWidth × 2`), converts world-space width to pixel width using
either `uPerspectiveLineScale = resY / tan(fov/2)` or
`uOrthoLineScale = 2·resY / frustumHeight` (precomputed CPU-side so the
shader has no `tan()` or projection-mode divide), clamps to
`[1.5 px, uMaxLinePixelWidth]` with an intensity-fading `vWidthFade`, then
offsets `clipPos.xy` by `perpendicular × aQuadCorner.y × clampedPixelWidth`.
The fragment stage shades
`capFactor × (1 − p²)^sharpness × edgeAA × widthScale × vWidthFade`,
applies the per-node GOG (`color × uIntensity + uOffset`, clamped) and the
`pow(·, uInvGamma)` gamma curve, and writes `vec4(rgb, intensity × uOpacity)`.

## The semicircle-kernel joint trick

Without compensation, two adjacent segments sharing an endpoint would each
draw a full-intensity quad up to that endpoint, summing to **2.0** under
additive blending — a visible bright nub at every joint. The fix lives in
the fragment shader: each segment fades to `0.5` at its true endpoints
(`baseCap = 0.5 + 0.5 × distToNearest / vWidthAtT`), and joints add to
`0.5 + 0.5 = 1.0` — the documented full body intensity. The ramp is
overridden to `1.0` when the nearest endpoint is **clipped** (the slice
boundary cut the polyline mid-segment; the real endpoint is outside the
slice so no neighbour will arrive to sum with). The cap is computed
fragment-side rather than vertex-side because with only 4 vertices per
quad, a vertex-side `min(t, 1−t) × segLen / width` collapses to `0.5`
everywhere — there's no vertex at the body midpoint to interpolate from.

## Geometry and attribute layout

Lines use `THREE.Mesh` with `InstancedBufferGeometry` — **not**
`THREE.InstancedMesh`. The per-instance attributes (start/end position,
colour-or-scalar, width, sharpness, segment length, clipped flags) are
packed into a single `InstancedInterleavedBuffer` so the geometry reports
one vertex buffer slot. This both fits within WebGL2's 16-attribute-location
limit and stays under WebGPU's `maxVertexBuffers` ceiling on
compat-mode adapters (Luxar requests the higher real limits at adapter
init; see `scene-manager.ts::setupWebGPURenderer`). See
`../../line-geometry.ts` for the buffer construction.

## Variant defines (fast paths)

Both backends share the same five `#define`s, set by the wrapper and
either gated via `#ifdef` (GLSL) or read at TSL build time
(`rebuildGraph` re-runs the factory):

| Define                       | Effect                                                                                                                                                                | Set by                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `USE_COLORMAP`               | Replaces `aStartColor`/`aEndColor` per-vertex RGB with `aStartScalar`/`aEndScalar` + LUT lookup                                                                       | `setColormapTexture(texture)` / `updateColormapTexture`          |
| `LUXAR_GAMMA_ONE`            | Skips three per-fragment `pow()` calls when `gamma == 1.0 ± 1e-4` (the default)                                                                                       | `updateGamma` when crossing the threshold                        |
| `LUXAR_NO_GOG`               | Skips the `vColor × uIntensity + uOffset` chain and its `max(·, 0)` clamp when `intensity==1 && offset==0`                                                            | `updateIntensity` / `updateOffset` via `_refreshNoGOGDefine`     |
| `LUXAR_SHARPNESS_TWO`        | Replaces `pow((1 − p²), max(vSharpness, 1e-4))` with `(1 − p²)²` when every per-vertex sharpness is 2.0                                                               | `setSharpnessAllTwo(true)` — node-factory inspects upload arrays |
| `LUXAR_MAX_RGB_CONTRIBUTION` | Premultiplies `rgb *= intensity × opacity` so `CustomBlending + MaxEquation + OneFactor/OneFactor` captures contribution-weighted colour rather than flat full-bright | `applyBlendingMode('max')`                                       |

The GLSL wrapper toggles `this.needsUpdate = true` when a define changes
so THREE's program cache recompiles; the TSL wrapper calls
`rebuildGraph()` and re-runs the factory. Projection mode (`uIsOrtho`) is
**not** a uniform branch in TSL — it's read from `nodes.uIsOrtho.value`
at build time and emits a single-branch graph, so a mode flip in
`updateCameraParams` triggers an explicit rebuild.

## Shared helpers from `_shared/`

| Symbol                                           | Used for                                                                                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `clampGamma(g)`                                  | `Math.max(0.001, g ?? 1.0)` guard before `1 / gamma` (shared across all six material constructors).                                                          |
| `CameraAwareMaterial` interface                  | Implemented so `MaterialManager.updateCameraParams(fov, resolution, isOrtho?)` reaches this material.                                                        |
| `ColormapAwareMaterial` interface                | Implemented so `material-colormap-helpers.ts` sets the LUT texture and scalar range through setters.                                                         |
| `GLSL_SANITIZE_FUNCTIONS`                        | Prepended to the GLSL vertex shader; gives `sanitizePositive` / `sanitizeNonNegative` to clean width/sharpness inputs against NaN/Inf/negative.              |
| `sanitizePositive` / `sanitizeNonNegative` (TSL) | TSL counterparts of the GLSL sanitisers — same contract, called inline in the factory.                                                                       |
| `proxyIUniform(node)`                            | Wraps each TSL `UniformNode` in an `IUniform`-shaped getter/setter so `material.uniforms.uX.value = Y` lands on `node.value`. No per-render callback bridge. |

## `isGammaOne` / `isNoGOG` cross-export

`isGammaOne` now lives in `../_shared/uniform-helpers.ts` (shared by all
three geometry types — Point/Line/GSplat each gate `LUXAR_GAMMA_ONE` on
it). `material-glsl.ts` re-exports it alongside the line-local `isNoGOG`,
and `material-tsl.ts` imports both from `./material-glsl`. The intent is
a single source of truth for the `±1e-4` epsilon — both backends decide
to flip the `LUXAR_GAMMA_ONE` / `LUXAR_NO_GOG` defines at the same
numeric boundary so a value that's a fast-path on WebGL2 is also a
fast-path on WebGPU.

## GLSL/TSL parity invariant

The GLSL strings are the authoritative spec for the rendering math; the
TSL factory must produce a graph that emits the same per-pixel result.
`tsl-shader-parity.spec.ts` (e2e) renders identical scenes through both
backends and pixel-compares. GLSL3 sources are **never** deleted from
this folder (see `feedback_keep_glsl_reference.md` in project memory) —
they remain the readable reference even after the TSL path stabilises.

## See Also

- `../_shared/README.md` — shared infrastructure, `ShaderSource` pattern, `buildMaterial` dispatch
- `../../README.md` — Rendering package overview and where line materials sit in the pipeline
- `../../line-geometry.ts` — `InstancedBufferGeometry` builder and the interleaved-attribute layout this shader binds
- `../../material-manager.ts` — owns the `getLineMaterial` cache and the camera-broadcast loop
- `../../picking/line/material.ts` / `material-tsl.ts` — picking counterparts; share the vertex-stage screen-space expansion math
- `../../../tests/e2e/tsl-shader-parity.spec.ts` — GLSL ↔ TSL parity harness
