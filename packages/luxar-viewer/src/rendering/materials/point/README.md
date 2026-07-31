# Point Material

> Soft-edged sprite shader for Luxar's first-class **Points** geometry — one instanced unit-quad per point, world-space sizing, per-point sharpness, optional colormap, per-node Gain/Offset/Gamma. Ships as a GLSL3 `ShaderMaterial` + TSL `NodeMaterial` pair behind a single `ShaderSource`.

This folder is the Point half of the per-geometry material stack
(`materials/point/`, `materials/line/`, `materials/gsplat/`). It mirrors the
shape of its siblings one-for-one: the four-file layout, the wrapper-class
surface (`updateOpacity`, `updateGamma`, `applyBlendingMode`, `clone`, …), the
`CameraAwareMaterial` / `ColormapAwareMaterial` interface implementations, and
the GLSL ↔ TSL parity contract — see the
[three-geometry symmetry note](../../README.md) in the rendering README and
the [shared infrastructure README](../_shared/README.md).

## Module map

| File               | Role                                                                                                                                                                                                                                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shader-glsl.ts`   | `POINT_VERTEX_SHADER` + `POINT_FRAGMENT_SHADER` GLSL3 strings, plus the `POINT_SOURCE: ShaderSource` that pairs them with the TSL factory.                                                                                                                                                                                       |
| `shader-tsl.ts`    | `pointWebGPUFactory(nodes, config, outMaterial?)` — TSL counterpart to the GLSL shaders. Consumes wrapper-owned `PointTSLNodes` (see `buildPointTSLNodesFromUniforms` for the harness/ShaderSource path), builds `vertexNode` + `colorNode`, and wires blending via `getCompleteBlendingState` + `applyBlendingStateToMaterial`. |
| `material-glsl.ts` | `PointMaterial extends THREE.ShaderMaterial` — the default WebGL2 wrapper. Owns the IUniform table, the `applyBlendingMode` state machine, `clone()`, and the `ColormapAwareMaterial` setters.                                                                                                                                   |
| `material-tsl.ts`  | `PointTSLMaterial extends NodeMaterial` — the WebGPU counterpart. Same public surface as `PointMaterial`; owns persistent `UniformNode`s exposed as `proxyIUniform` bridges and calls `pointWebGPUFactory(..., this)` to attach the TSL graph in place.                                                                          |

`MaterialManager.getPointMaterial` dispatches on `caps.apiSurface` so callers
(`NodeFactory.createPointsMaterial`, `LayersPanel`, …) never see the
divergence.

## The point sprite

Each point is one instance of a 4-vertex unit-quad (`aQuadCorner ∈ [-1, 1]²`,
the base geometry from `../../point-geometry.ts`).

Per-point data (`center`, `radius`, `color`, `sharpness`, `scalar`) does
**not** live in vertex attributes. It lives in an RGBA32F **point texture**
(`uPointTex`, 3 texels/point — layout authority in
`../../element-texture-layout.ts`; per-texel map in
`../../point-geometry.ts`), fetched in the vertex stage via `texelFetch`
(GLSL) / `textureLoad` (TSL). The only per-instance data is the double-buffered ordering pair
`aSortedIndex` / `aSortedIndexB` (Uint32): the draw-slot → storage-slot map, written as
identity by every commit and permuted by the sort worker; a `uSortedIndexSlot`
uniform selects the buffer the shaders read. Consequences (mirroring the gsplat stack):

- **Materials are per node.** Each point material binds its node's texture,
  so the material-manager LRU is bypassed for points (`getPointMaterial`
  always creates). The commit rebinds `uPointTex` on render + pick materials
  via `syncPointMaterialWithGeometry` (pool acquire may hand the node a
  different geometry+texture pair on growth/reuse).
- **Texture lifetime = geometry lifetime.** `attachPointStorage` registers a
  geometry-`dispose` listener; every pool/fallback dispose site frees the
  texture with its geometry.
- **TSL texture-node lifecycle.** The TSL `texture()` node is factory-time
  bound, so `updatePointTexture` rebuilds the graph on an identity change
  (exact mirror of the colormap-texture lifecycle) and no-ops otherwise.
- **GLSL fallback trap (load-bearing `int()`).** TSL types `textureSize()`
  as `uint` (WGSL convention) but GLSL's `textureSize` returns `int` — the
  width read is wrapped in `int(...)` or the generated GLSL fails to compile
  on the `forceWebGL` backend.

The texel fetch prologue reconstructs the historical local names, so the
math below it is unchanged:

| Texel slot | Local        | Meaning                                                                                                                                                                                                                                                                                                                          |
| ---------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| texel0.xyz | `aCenter`    | World-space centre position                                                                                                                                                                                                                                                                                                      |
| texel0.w   | `aRadius`    | Per-point radius (multiplied by `radiusScale` for dtype normalisation; e.g. `1/255` for `uint8` storage)                                                                                                                                                                                                                         |
| texel1.rgb | `aColor`     | Per-point colour (HDR); the writer fills white when the dataset has none                                                                                                                                                                                                                                                         |
| texel1.w   | `aSharpness` | Per-point sharpness — a normalised `[0, 1]` knob (no scale; `uint8/255` already lands in range). Maps in-shader to the super-Gaussian exponent `β = 2^(6s − 2)`                                                                                                                                                                  |
| texel2.x   | `aScalar`    | `USE_COLORMAP` only — replaces `aColor` via LUT lookup (fetched only in colormap builds; 0.0 identity fill when the dataset has no scalars)                                                                                                                                                                                      |
| texel2.y   | `vAlpha`     | Per-point opacity alpha — the RGBA color column when the dataset carries one, else the 1.0 opaque identity fill. Read through `sanitizeAlpha` (NaN/Inf → opaque 1.0; finite clamped to `[0, 1]`). Linear contribution scale in every mode; volumetric maps it into optical depth `w(a) = −ln(1 − a)` gated by `uHasElementAlpha` |

Scalar PRESENCE is not knowable from the fixed layout, so the texel writers
stamp `geometry.userData.hasScalars` and the fail-closed colormap guard
(`supportsScalarColormap('points', …)`) reads that stamp.

The vertex shader projects `aCenter` to clip space, computes a world-space
`pointSize` in pixels, then expands the unit quad by
`aQuadCorner * pointSize / uResolution * projCenter.w` (the `* projCenter.w`
converts the NDC delta into a clip-space delta that survives the upcoming
perspective divide). The interpolated sprite UV `vSpriteCoord ∈ [0, 1]²`
replaces `gl_PointCoord`, which is unavailable under `THREE.Mesh`.

Points behind the camera (perspective only — `mvPosition.z >= 0`, where
`projCenter.w <= 0` would flip/degenerate the sprite) are rejected to an
off-screen clip position so they produce no fragments, mirroring the gsplat
shader's behind-camera guard.

## World-space sizing

Sizing is FOV-independent and matches `..` siblings' implementation: per-frame
camera changes update **only** the precomputed scalar uniforms, not the shader.

- `pointSizeFactor = 2 · resolution.y / tan(fov/2)` (perspective) or
  `4 · resolution.y / frustumHeight` (ortho) — computed by `computePointSizeFactor`
  in `../_shared/camera-uniforms.ts`, the shared math module that both Point and
  GSplat materials and their picking counterparts pull from.
- `maxPointSize = resolution.y · 0.5` — `computeMaxPointSize`, same module.
- `uIsOrtho` (`int`) branches the inverse-distance term: `1.0` for ortho,
  `1.0 / max(-mvPosition.z, 1e-20)` for perspective (the floor is a pure INF guard, not a scale floor) — VIEW-SPACE DEPTH, matching
  the line + gsplat shaders. (Euclidean camera distance shrank edge-of-screen
  points by `cos θ` relative to identical centered points.)
- `pointSize = clamp(basePointSize, 1.5, maxPointSize)`. There is **no
  sharpness size compensation** — the shifted-truncated super-Gaussian falloff
  truncates to zero exactly at the sprite edge (`ρ = 1`), so `basePointSize`
  already is the visible extent. The `1.5` px floor matches the line shader
  (thinner quads cause rasterization gaps); the raw pre-clamp size travels to
  the fragment as `vPointSize`, where sub-pixel sprites are energy-compensated
  by `sizeScale² = min(vPointSize/1.5, 1)²` on alpha (squared because both
  sprite dimensions clamp — energy ∝ area; the line shader's `widthScale` is
  linear because only width clamps). Zero-radius filtering happens in the
  fragment shader (see "nD slicing").
- `uNearCull` + the shared `perspectiveNearFade` helper: behind-camera fade 0,
  smooth `[nearCull, 2·nearCull]` fade under perspective (reject < 0.01,
  `vNearFade` multiplied into alpha), fade ≡ 1 under ortho where NDC clipping
  is the sole cull authority — unified with the line + gsplat shaders.

`MaterialManager.updateCameraParams(fov, resolution, isOrtho?)` broadcasts to
every registered material via the `CameraAwareMaterial` interface, so a single
camera-change call updates every Point material in the scene.

## Sharpness → super-Gaussian exponent

The fragment falloff is a **shifted-truncated super-Gaussian** (see the next
section). `sharpness` is a normalised `[0, 1]` knob; the vertex stage maps it to
the exponent `β`:

```
s    = clamp(sanitizeNonNegative(aSharpness, 0.5), 0.0, 1.0)
vBeta = exp2(6·s − 2)        // β = 2^(6s − 2)
```

So `s = 0.5 → β = 2` (a true Gaussian — the default, and exactly the GSplat
kernel's shape); higher `s → β` up to 16 (a harder, crisper edge); lower
`s → β` down to 0.25 (a peakier cusp). Because the kernel truncates at the
sprite edge, the visible extent no longer depends on `β` — there is **no
sharpness size compensation** (the old polynomial `(1 − r)^s` kernel needed it;
this kernel does not).

`sanitizeNonNegative` keeps a valid `s = 0` (→ β = 0.25) and routes
NaN/Inf/negative to the `0.5` default; the subsequent `clamp` bounds `[0, 1]`.
It comes from `../_shared/glsl-lib.ts` (GLSL) / `../_shared/tsl-helpers.ts`
(TSL); `aRadius` is sanitised the same way to prevent malformed-data poisoning.

## Fragment stage: falloff + GOG + max-mode

The fragment shader runs in this order:

1. **Zero-radius discard** — points clipped by nD slicing arrive with
   `vRadius ≈ 0` from the loader; `discard` on `vRadius <= 0.0` (exact zero — scale-free) short-circuits
   the rest. (`vRadius` uses `highp` precision specifically for this check.)
2. **Inscribed-circle discard** — `centered = vSpriteCoord - 0.5`,
   `r2 = dot(centered, centered)`, `discard` if `r2 > 0.25`. Comparing squared
   distance avoids a `sqrt` on the discard path.
3. **Falloff** — `normalizedR = sqrt(4 · r2)`, then the shifted-truncated
   super-Gaussian `falloff = max(exp(−K·ρ^β) − C, 0) / (1 − C)` with
   `K = ln(100) ≈ 4.605`, `C = exp(−K) = 0.01` (the 1% iso-contour floor). It is
   C⁰-continuous at the edge (`falloff(0)=1`, `falloff(1)=0`, no hard ring), and
   `β = 2` reproduces the GSplat Gaussian exactly.
4. **Per-node GOG (Gain/Offset/Gamma)** —
   `adjusted = vColor * uIntensity + uOffset` (clamped non-negative),
   then `finalColor = pow(adjusted, vec3(uInvGamma))`. Pre-computed `uInvGamma`
   moves the division out of the per-fragment path. A second discard culls
   sub-1e-4 fragments to skip cost on offset-zeroed pixels. GOG is **per-node**;
   global EOG (exposure) lives in the mega-shader post-processing pass.
   The `LUXAR_GAMMA_ONE` define (set by `updateGamma` when `gamma == 1.0 ±
1e-4`, the default) skips this `pow()` — `pow(x, 1) == x` — and also the
   pre-LUT value `pow()` in colormap mode. The `LUXAR_NO_GOG` define (set by
   `updateIntensity`/`updateOffset` when `intensity == 1 && offset == 0`, the
   default) skips the gain/offset mul/add/clamp chain the same way. Both
   mirror the Line/GSplat fast paths (`isGammaOne` / `isNoGOG` in
   `../_shared/uniform-helpers`).
   In `USE_COLORMAP` mode only the gamma `pow()` is skipped: gamma and the
   display range already shaped the scalar **value** before the LUT lookup in
   the vertex stage, so warping the mapped LUT colour again would be wrong.
   `uIntensity`/`uOffset` still apply post-LUT to the mapped colour (matching
   the GSplat shader) so the layer intensity/offset controls work on
   colormapped nodes too. See the colormap section.
5. **Alpha** — `alpha = falloff * uOpacity`.
6. **Max-mode RGB premultiplication** —
   `#ifdef LUXAR_MAX_RGB_CONTRIBUTION` returns `vec4(finalColor * alpha, alpha)`;
   the default path returns `vec4(finalColor, alpha)` (for `AdditiveBlending`'s
   `SrcAlpha, One`). See the next section.

## Blending modes

`PointMaterial`/`PointTSLMaterial` accept the canonical Luxar `BlendingMode`
(`'additive' | 'volumetric' | 'normal' | 'opaque' | 'luminous' | 'max'`; `volumetric` is the real emission–absorption math since phase 3 — `LUXAR_VOLUMETRIC` output branch, τ = κ·density·chord with the isotropic chord scale from `./math.ts`, over the premultiplied One/OneMinusSrcAlpha state) and route everything
through `applyBlendingMode(mode)`, which is the **single source of truth** for
both creation (called from the constructor) and runtime UI transitions (called
from `LayersPanel`). The method:

- Pulls the canonical THREE state from `getCompleteBlendingState` and applies
  it via `applyBlendingStateToMaterial` (`../../blending-state.ts`) — covers
  `blending`, `blendEquation`, `blendSrc`/`blendDst`, `depthTest`, etc.
- Adds/removes the `LUXAR_MAX_RGB_CONTRIBUTION` shader define. In `max` mode
  the framebuffer uses `CustomBlending + MaxEquation + OneFactor/OneFactor`,
  which does **not** multiply source RGB by alpha at composite time. Without
  premultiplication a soft point with `alpha = 0.1` would still write its full
  bright RGB → max captures a flat coloured disk instead of the intended soft
  contribution. The define toggles a shader recompile and forces the
  premultiplied output.
- Idempotent: identical state is a no-op via `userData.blendingMode` /
  `defines.LUXAR_MAX_RGB_CONTRIBUTION` early exits.

This is why the constructor's blending-mode wiring isn't done inline — the
LayersPanel runtime-transition path needs the exact same code, and a previous
generic UI path forgot `blendSrc`/`blendDst` and stranded SrcAlpha factors on
an `additive → max` switch.

## Colormap branch (`USE_COLORMAP`)

Optional per-point scalar colouring. When enabled, the **vertex** shader
reconstructs `aScalar` from texel2.x of the point texture and samples a
256×1 LUT texture
(`uColormapTex`) at `t = clamp((aScalar - uScalarMin) * uScalarScale, 0, 1)`
instead of reading `aColor`. The LUT lookup runs in the vertex stage and the
mapped colour is carried to the fragment as `vColor`.

Gamma and the display range act on the scalar **value** pre-LUT, not on the
resulting colour: `t = pow(t, uInvGamma)` (skipped under `LUXAR_GAMMA_ONE`) is
applied to `t` before the texture read, and the fragment gamma `pow()` is
skipped (see step 4 above). Intensity/offset apply **post-LUT** to the mapped
colour, matching the GSplat shader, so the layer gain/offset controls work on
colormapped nodes. This keeps gamma controls meaningful on the data axis
while leaving the chosen LUT palette undistorted.

- **GLSL** path: a `#ifdef USE_COLORMAP` block. `updateColormapTexture` flips
  the define and sets `material.needsUpdate = true` to trigger recompilation.
- **TSL** path: the colormap branch is a JS-side `if` in the factory, plus a
  `texture()` node that captures the `THREE.Texture` by reference at
  factory-call time. `PointTSLMaterial.updateColormapTexture` calls
  `rebuildGraph()` whenever the on/off state **or** texture identity changes —
  the GLSL `ShaderMaterial` path gets away with a single uniform write because
  it reads the IUniform by reference every frame; TSL doesn't.

Both materials implement the `ColormapAwareMaterial` interface
(`setColormapTexture`, `setScalarRange`) so `material-colormap-helpers.ts` is
their only external writer — nothing outside reaches into
`material.uniforms.uColormapTex` directly.

## nD slicing

Points with zero effective radius arrive from `data/points/projection.ts`
when their nD position doesn't intersect the current display hyperplane.
Rather than culling on the CPU side, the encoder ships `aRadius = 0` and the
fragment shader's `if (vRadius <= 0.0) discard` (exact zero — scale-free) short-circuits them.
Performance-wise this is acceptable because the vertex stage still runs, but
the more expensive falloff/GOG/colormap fragment work is skipped.

## Uniforms (reference)

| Name              | Type      | Source                                        | Notes                                                                                        |
| ----------------- | --------- | --------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `uOpacity`        | float     | `updateOpacity`                               | Multiplied into final alpha (historically the un-prefixed `opacity`; renamed for symmetry)   |
| `uInvGamma`       | float     | `updateGamma` (pre-computed `1/γ`)            | Per-node gamma; `userData.gamma` carries the original value for `clone()`                    |
| `uIntensity`      | float     | `updateIntensity`                             | Per-node GOG gain                                                                            |
| `uOffset`         | float     | `updateOffset`                                | Per-node GOG offset                                                                          |
| `pointSizeFactor` | float     | `updateCameraParams` (camera math)            | Pre-computed `2·resY/tan(fov/2)` (or ortho form)                                             |
| `maxPointSize`    | float     | `updateCameraParams`                          | Pre-computed `resY · 0.5`                                                                    |
| `uIsOrtho`        | int       | `updateCameraParams`                          | `0` = perspective, `1` = ortho                                                               |
| `uNearCull`       | float     | `updateCameraParams`                          | Near-fade start (world units, scene-bounds-scaled); shader floors at 1e-20 (zero-guard only) |
| `uResolution`     | vec2      | `updateCameraParams` (mutates same Vector2)   | Physical framebuffer pixels; vertex uses for `pixel → NDC` conversion                        |
| `radiusScale`     | float     | `updateRadiusScale`                           | Dtype normalisation (e.g. `1/255` for uint8 radii)                                           |
| `uPointTex`       | sampler2D | `updatePointTexture` (commit sync)            | RGBA32F point texture, 3 texels/point — the per-node data store                              |
| `uColormapTex`    | sampler2D | `setColormapTexture`                          | 256×1 LUT; `USE_COLORMAP` only                                                               |
| `uScalarMin`      | float     | `setScalarRange`                              | LUT normalisation min                                                                        |
| `uScalarScale`    | float     | `setScalarRange` (pre-computed `1/(max-min)`) | LUT normalisation scale                                                                      |

`clampGamma` (`../_shared/uniform-helpers.ts`) is the single source of truth
for the `Math.max(0.001, γ ?? 1.0)` clamp — the GLSL `pow(color, 1/γ)` divides
by zero at `γ == 0`, so every gamma write goes through this helper.

## `clone()`

Both wrappers override `clone()` to return their concrete type (`PointMaterial`
or `PointTSLMaterial` — not the base `THREE.ShaderMaterial`/`NodeMaterial`).
The clone path:

1. Constructs a new instance with the original `PointMaterialConfig` derived
   from `this.uniforms.*.value`, `this.userData.{gamma,depthTest,blendingMode,scalarRange}`,
   and `this.uniforms.uColormapTex?.value`. The constructor's
   `applyBlendingMode` re-establishes blending state and shader defines.
2. Copies the runtime-only camera uniforms (`pointSizeFactor`, `maxPointSize`,
   `uInvGamma`, `radiusScale`) verbatim so the clone starts at
   the current camera frame, not the default.
3. For `THREE.CustomBlending` (`max` mode), copies `blendEquation/Src/Dst` from
   the source — the constructor would set canonical defaults, but if the source
   had any post-construction overrides, copy carries them through.

Disposal is inherited from the base material class; `MaterialManager`
subscribes to the synchronous `dispose` event and cleans up its registry +
cache automatically, so no explicit unregister hook lives here. (This is why
this file stays out of the manager's import graph.)

## Related

- `../_shared/README.md` — `ShaderSource`, `buildMaterial`, `CameraAwareMaterial`,
  `ColormapAwareMaterial`, `clampGamma`, the shared GLSL/TSL sanitisers, the
  shared `pointSizeFactor` / `maxPointSize` / `focalLength` math.
- `../line/`, `../gsplat/` — sibling stacks with the same four-file layout and
  the same public surface (three-geometry symmetry).
- `../../point-geometry.ts` — the 4-vertex unit-quad base geometry the
  instancing builds on.
- `../../node-factory/create-points-node.ts` — `createPointsGeometry` (sets up
  the `InstancedBufferAttribute`s), `createPointsMaterial` (the entry point
  for callers), and `createPointsNode` (full mesh + picking assembly).
- `../../material-manager.ts` — `getPointMaterial(config)` is the dispatch
  entry that selects `PointMaterial` vs `PointTSLMaterial`.
- `../../picking/point/material.ts` / `material-tsl.ts` — picking counterpart;
  reuses the same vertex math via `camera-uniforms.ts` so screen-space hit
  tests match what the user sees.
- `../../../tests/e2e/tsl-shader-parity.spec.ts` — GLSL ↔ TSL parity harness
  that pairs `POINT_SOURCE.webgl` and `POINT_SOURCE.webgpu`.
