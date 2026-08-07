# Line Material

> Thick-line material stack — instanced-quad geometry, shifted-truncated super-Gaussian soft falloff, and continuous polyline joints — paired across the GLSL `ShaderMaterial` and TSL `NodeMaterial` backends.

This folder holds the four-file material stack that renders one of Luxar's
four first-class geometry types. Each line segment is drawn as an instanced
screen-space quad expanded perpendicular to its pixel-space direction; the
fragment stage shades a shifted-truncated super-Gaussian perpendicular
cross-section, with a per-endpoint joint code keeping interior polyline joints
continuous and driving the screen-space miter join. Both backends share
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
(degenerate-quad when both endpoints fail `uNearCull`; **segment clipping**
when exactly ONE endpoint is closer than `uNearCull` or behind the camera —
that endpoint is moved along the segment onto the nearCull plane before any
screen-space math and `t` is remapped (`tEff`) so per-endpoint attributes
and the cap math keep the original parameterization; without it the
behind-camera endpoint's `clip.w ≤ 0` wraps the quad into an external
primitive whose visible half cuts a bright razor edge through the
cross-profile at close zoom, and the cut lands exactly where the
per-fragment near fade reaches zero so no seam is visible; pathological-quad
discard when both are near AND the per-segment MAX raw pixel width — evaluated
at both clipped endpoints, so the whole quad takes one branch — blows past
`uMaxLinePixelWidth × 2`), converts world-space width to pixel width using
either `uPerspectiveLineScale = resY / tan(fov/2)` or
`uOrthoLineScale = 2·resY / frustumHeight` (precomputed CPU-side so the
shader has no `tan()` or projection-mode divide), clamps to
`[1.5 px, uMaxLinePixelWidth]` with an intensity-fading `vWidthFade`, then
offsets `clipPos.xy` by `perpendicular × aQuadCorner.y × startEndPixelWidth` /
`endEndPixelWidth` — the clamped pixel half-width of the END this corner sits at
(from the shared `luxarLineEndPixelWidth` / `tslLineEndPixelWidth` helper),
which is segment-constant and equals the per-vertex clamp exactly at the corner
it is consumed at (the join below needs it segment-constant; see there).
The fragment stage shades
`capFactor × perpFalloff × edgeAA × widthScale × vWidthFade × nearFade` (the near
fade is computed PER-FRAGMENT from the interpolated view depth `vViewZ` — a
fade varying would mis-interpolate on long segments), where the
perpendicular cross-section `perpFalloff(p) = max(exp(−K·p^β) − C, 0)/(1 − C)`
is a **shifted-truncated super-Gaussian** (`p = |vPerpNorm| ∈ [0, 1]` from the
centerline, `K = ln(100) ≈ 4.605`, `C = exp(−K) = 0.01`, the 1% iso-contour
floor). The per-vertex `sharpness` is a normalised `[0, 1]` knob mapping to the
super-Gaussian exponent `β = 2^(6s − 2)`: `s = 0.5 → β = 2` (a truncated
Gaussian, the default — identical to the GSplat kernel's shape), `s = 1 → β = 16`
(hard edge), `s = 0 → β = 0.25` (cusp). It is C⁰-continuous at the line edge
(`perpFalloff(0) = 1`, `perpFalloff(1) = 0`, no hard ring). The stage then
applies the per-node GOG (`color × uIntensity + uOffset`, clamped) and the
`pow(·, uInvGamma)` gamma curve — under `USE_COLORMAP` only the post-LUT
gamma is skipped (gamma + display range shape the scalar pre-LUT in the
vertex stage); intensity/offset still apply post-LUT to the mapped colour,
matching the GSplat shader, so the layer gain/offset controls work on
colormapped nodes — and writes `vec4(rgb, intensity × vAlpha × uOpacity)`
(`vAlpha` is the per-endpoint opacity, `1.0` for RGB data). Under
`LUXAR_VOLUMETRIC` the output switches to the emission–absorption branch
described below.
The `capFactor` joint machinery (next section) is **independent** of the
perpendicular falloff — only `perpFalloff` changed when the kernel was swapped
to the super-Gaussian, and the miter join leaves it untouched too (the mitred
trapezoid's edges stay on the segment's own ±R offset lines).

## The endpoint cap and the joint code

Each segment fades to `0.5` at its true endpoints (a per-endpoint ramp
`0.5 + 0.5 × dist / vWidthAtT`), giving a soft cap at a free polyline end
rather than a hard flat cut. The cap is computed fragment-side rather than
vertex-side because with only 4 vertices per quad, a vertex-side
`min(t, 1−t) × segLen / width` collapses to `0.5` everywhere — there's no
vertex at the body midpoint to interpolate from.

**That dimming is only correct where a neighbouring quad meets the
endpoint.** The quad spans exactly `[start, end]` — there is no longitudinal
extension — so two collinear segments _tile_ rather than overlap. A fragment
just inside segment A gets `0.5 + 0.5·d/w` from A and nothing at all from B
(it is outside B's quad), so the two halves never sum back to 1.0 and every
interior joint became a dark notch of axial length `2 × width` bottoming out
at 50% — thick polylines rendered as bead chains (issue #780).

So the endpoint dimming is gated by a per-endpoint **joint code** (texel4.yz).
Each endpoint's ramp is lifted by its own value and the two caps combine with
`min()`:
`capFactor = min(mix(startRamp, 1.0, suppressStart), mix(endRamp, 1.0, suppressEnd))`
— evaluated independently per endpoint (not keyed on the nearest one), which
makes the cap field continuous **within** each segment: the nearest-endpoint
pick used to jump at the midpoint of segments shorter than `2 × width` when
the two values differ, the routine case for a polyline's first/last segment
(issue #796). The old form was exactly continuous **across** the joint seam,
so the `min()` form _relocates_ the discontinuity rather than leaving one
behind: a strictly smaller step at the seam, appearing only when a segment is
shorter than one width, and zero for `L ≥ width`.

The code is a small exact integer, not a scalar — it says what KIND of
endpoint this is, and at an ordinary two-segment joint, WHICH segment it
joins:

| Code          | Endpoint                  | Cap  | Why                                                              |
| ------------- | ------------------------- | ---- | ---------------------------------------------------------------- |
| `0`           | Free polyline end         | kept | nothing meets it; the soft cap is the point                      |
| `-1`          | Slice-clipped             | none | the real endpoint is outside the slice; no neighbour will arrive |
| `-2`          | Degree-≥3 hub             | kept | several quads already stack here; suppressing stacks them bright |
| `+(slot + 1)` | Joins `slot` at its START | none | a neighbouring quad meets this endpoint                          |
| `-(slot + 3)` | Joins `slot` at its END   | none | same, with the partner's other endpoint shared                   |

`slot` is the partner's index in the visible stream, which is exactly its
line-texture **storage** slot. `aSortedIndex` maps draw→storage and the
partner is read directly rather than through that permutation, so a
depth-sort re-ordering needs no bookkeeping. The sign carries which of the
partner's endpoints is shared rather than a packed `(slot << 1) | bit`,
because a bare slot stays inside float32's 2²⁴ exact-integer range at the
11.17M per-node segment ceiling while the packed form reaches 22.35M and
would silently lose precision on a 16384-class device.

**A slot-bearing code suppresses the cap.** Defaulting it the other way is
the #780 bead chain again, and not only under join style `none`: the join
block is also skipped for every line below the rendered-width gate, so
thin-line scenes — the million-segment ones — would lose the fix entirely.
Measured, an interior joint bottoms out at 0.5 instead of 1.0 and a dense
polyline loses ~40% of its total brightness.

Joints are matched by vertex **index**, not by position: a chain whose
segments each carry their own duplicate copy of the shared point (what
`line_type="segments"` emits for abutting segments) has no shared index, so
it keeps the cap at every joint and still shows the notch. Author connected
geometry as `line_type="polyline"` to get continuous joints — position
matching would also fuse two unrelated lines that merely touch.

The code is computed once per commit, off the main thread, by
`compute_joint_codes` (`wasm/rust/src/lines_clipping.rs`, with the uncapped
TypeScript reference in `wasm/typescript/lines-clipping.ts` — that mirror is
the production backend above 16 dimensions, not just a fallback). It only
pairs endpoints that both actually _reach_ the shared vertex, so a culled or
slice-trimmed neighbour does not anchor a joint. It is purely topological: it
reads connectivity and the clip parameters, never positions, so the bend
angle is the shader's business.

## Join geometry (`uLineJoin`)

At a turn of angle θ two quads leave an uncovered circular sector of that
angle on the OUTSIDE of the bend and double-cover a lens on the inside: dark
ticks along the convex edge of a thick curve, bright ticks along the concave
one (issue #790). No per-endpoint intensity scalar can close the outer wedge
— nothing rasterises there to shade — so it needs geometry.

`uLineJoin` selects the strategy (`types/line-join.ts`; precedence is
`?lineJoin=` > the node's authored `join` attribute > `miter`):

| Style   | Per-vertex cost           | Wedge     | Blending modes           |
| ------- | ------------------------- | --------- | ------------------------ |
| `none`  | zero                      | left open | n/a                      |
| `miter` | +1 texel fetch, 1 project | **exact** | all six, by construction |

`miter` rotates the quad's end edge onto the shared miter edge, so the two
quads TILE: coverage becomes a partition, and with nothing to sum there is no
axial profile and no per-mode special case. The miter point is the
intersection of the two segments' `+R` offset lines,
`M = R·(perpIn + perpOut) / (1 + turn)`, which reduces to `R·perp` at a
collinear joint — so straight polylines are untouched — and lies ON this
segment's own `±R` offset line, so `vPerpNorm` stays an exact perpendicular
coordinate and the super-Gaussian cross-section is unchanged.

Both sides of a joint must take the same branch, or one rotated edge has
nothing to tile against and rasterises as a flap. The guards are therefore
computed from operands that are identical on either side — the shared
vertex's width, the depths of the joint's two FAR endpoints (NOT the shared
vertex's own; see the bullet below), and `min()` over the two lengths — with
the directions read in a canonical order (incoming edge first):

- miter limit `grow = sqrt(2/(1+turn)) ≤ 2` (θ ≤ 120°)
- overshoot on the **axial** reach `R·tan(θ/2) ≤ ½·min(pixelLen, partnerLen)`
  — not on `|M|`, which is ≈R always and would disable the join on every
  polyline whose segments are shorter than twice the tube radius, i.e.
  exactly the dense-curve case
- BOTH far endpoints of the joint — the partner's and this segment's own —
  must be in front of the near plane (testing only the partner's has each side
  testing a different point, so one side can miter alone against nothing), and
  this endpoint must actually reach its source vertex (`tA ≤ 0` / `tB ≥ 1`)
- a **rendered-width gate** of 2 px: the wedge has area ~θ·R²/2, so below
  that it is sub-pixel and the line is already pinned to the 1.5 px floor
  with its intensity faded. The cost then lands only where the benefit is —
  million-segment scenes are thin-line scenes and skip the block entirely.

Where the block runs it also DERIVES the endpoint cap, as
`clamp(dot(lineDir, partnerDir), 0, 1)` — algebraically the same quantity the
kernel used to store, but measured in SCREEN space, per frame. That is what
retires the old "the angle is data-space, the overlap is screen-space"
limitation (#795): a gentle 3D bend that projects sharp is now seen as sharp.
Where the block is skipped, the code-implied cap above applies instead, which
is exact for the straight and gentle joints that dominate real polyline data
and are projection-invariant anyway.

All four stages build the join from one source: the visual and pick GLSL vertex
shaders share `GLSL_LINE_JOIN`'s `luxarLineJoin`, and the visual and pick TSL
factories share its twin `tslLineJoin` (`_shared/tsl-helpers.ts`). Parity tests
assert the two backends agree on the VISUAL join — end→start, END–END, and a
tapered perspective joint (`line-join-*` in the TSL harness) — so a mitred
joint is not a WebGL2/WebGPU difference. The pick stages run the same helper by
construction, but no pick fixture carries a slot-bearing joint code, so a
mitred corner's pick footprint is not pixel-pinned on either backend.

## Geometry and storage layout

Lines use `THREE.Mesh` with `InstancedBufferGeometry` — **not**
`THREE.InstancedMesh`. Per-segment data lives in a per-node RGBA32F
**line texture** (`uLineTex`, 6 texels/segment — layout authority in
`../../element-texture-layout.ts`; per-texel map in
`../../line-geometry.ts`), fetched in the vertex stage via `texelFetch`
and indexed by the double-buffered ordering pair `aSortedIndex` /
`aSortedIndexB` (Uint32,
the draw-slot → storage-slot mapping the depth-sort worker permutes).
The texel fetch prologue reconstructs the historical local names
(`aStartPos`, `aEndWidth`, …), so the expansion math below is unchanged
from the interleaved era. See `../../line-geometry.ts` for the storage
construction and the fused texel writer.

Texel5 carries the colormap scalars in `.xy` and the **per-endpoint
opacity alphas** in `.zw` — the alpha column of an RGBA color dataset
(`(N, 4)` colors; the writer fills the `1.0` opaque identity for RGB
data). The vertex stage reads both slots through `sanitizeAlpha`
(NaN/Inf → opaque `1.0`; finite clamped to `[0, 1]`) and interpolates
them along the segment parameter `t` into the `vAlpha` varying. In every
non-volumetric mode `vAlpha` is a plain linear contribution scale
(identity for RGB data, no gate needed); under `LUXAR_VOLUMETRIC` it
maps into optical depth `w(a) = −ln(1 − a)`, gated by
`uHasElementAlpha` (next-but-one section).

## Volumetric emission–absorption branch (`LUXAR_VOLUMETRIC`)

Since volumetric Phase 4 (VOLUMETRIC_BLENDING_SPEC.md §7) lines render
the REAL `volumetric` blending math on both backends — the former
additive-state fallback (and the `effectiveGeometryMode` helper that
encoded it) is gone. Since the 2026-08-02 ray-mass unification
(VOLUMETRIC_BLENDING_SPEC.md status banner) the ray mass is the SAME
quantity the additive branch emits — no world-thickness factor (the
former `× vWidthAtT × LINE_CHORD_SCALE` chord factor and its `math.ts`
module are deleted), keeping the point/line/gsplat κ scales aligned.
The optical depth the shader computes is `τ = uAbsorption × alpha`,
where `alpha` is the additive-mode screen density — the full intensity
chain (`capFactor · perpFalloff · edgeAA · widthScale · vWidthFade ·
nearFade`, so `perpFalloff` enters τ exactly once) times `uOpacity`,
times the `w(vAlpha)` map when `uHasElementAlpha` is set. Every "how
much of this line is there" factor scales emission and absorption
together, so fades leave no ghost fog. Emission is `gammaColor × alpha × S(τ)` with the shared
self-screening series `S(τ) = (1 − e^(−τ))/τ` (constants from
`../_shared/volumetric.ts`, shared with the point/gsplat twins), and
the output alpha is the physical absorption `1 − e^(−τ)` over the
premultiplied `One / OneMinusSrcAlpha` state. `κ = 0` reproduces
`additive` exactly. The color early-discard is bypassed while `τ` is
significant — a black line still absorbs (a pure-ink occluder keeps its
optical depth).

Plumbing: both wrappers accept `absorption` in `LineMaterialConfig`,
own the `uAbsorption` (composed node κ) and `uHasElementAlpha` uniforms
(plain uniforms — no rebuild on toggle), and expose `updateAbsorption` /
`updateHasElementAlpha`; both are carried through `clone()`. The
layers-panel κ slider shows for volumetric lines layers. Because
`volumetric` is order-dependent, line meshes in this mode are
back-to-front depth-sorted by segment midpoint via `needsDepthSort(mode)`
and the existing lazy midpoint provider (see
`GSPLAT_DEPTH_SORTING_SPEC.md` §8).

## Variant defines (fast paths)

Both backends share the same five `#define`s, set by the wrapper and
either gated via `#ifdef` (GLSL) or read at TSL build time
(`rebuildGraph` re-runs the factory):

| Define                       | Effect                                                                                                                                                                | Set by                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `USE_COLORMAP`               | Replaces the texel2/3 per-endpoint RGB with the texel5 scalars + LUT lookup (presence rides the geometry's `userData.hasScalars` stamp)                               | `setColormapTexture(texture)` / `updateColormapTexture`      |
| `LUXAR_GAMMA_ONE`            | Skips three per-fragment `pow()` calls when `gamma == 1.0 ± 1e-4` (the default)                                                                                       | `updateGamma` when crossing the threshold                    |
| `LUXAR_NO_GOG`               | Skips the `vColor × uIntensity + uOffset` chain and its `max(·, 0)` clamp when `intensity==1 && offset==0`                                                            | `updateIntensity` / `updateOffset` via `_refreshNoGOGDefine` |
| `LUXAR_MAX_RGB_CONTRIBUTION` | Premultiplies `rgb *= intensity × opacity` so `CustomBlending + MaxEquation + OneFactor/OneFactor` captures contribution-weighted colour rather than flat full-bright | `applyBlendingMode('max')`                                   |
| `LUXAR_VOLUMETRIC`           | Switches the fragment output to the emission–absorption branch (τ = κ·alpha, `S(τ)` screening, `1 − e^(−τ)` alpha — see the volumetric section above)                 | `applyBlendingMode('volumetric')`                            |

The perpendicular falloff is **not** a define-gated fast path: the
super-Gaussian `max(exp(−K·p^β) − C, 0)/(1 − C)` is computed unconditionally
from the interpolated `[0, 1]` sharpness knob (`β = 2^(6s − 2)`), so there is
no `LUXAR_SHARPNESS_TWO` / `setSharpnessAllTwo` analogue any more — the old
`(1 − p²)^sharpness` polynomial that needed an `x·x` shortcut for `sharpness == 2`
is gone.

The GLSL wrapper toggles `this.needsUpdate = true` when a define changes
so THREE's program cache recompiles; the TSL wrapper calls
`rebuildGraph()` and re-runs the factory. Projection mode (`uIsOrtho`) is
**not** a uniform branch in TSL — it's read from `nodes.uIsOrtho.value`
at build time and emits a single-branch graph, so a mode flip in
`updateCameraParams` triggers an explicit rebuild.

## Shared helpers from `_shared/`

| Symbol                                                             | Used for                                                                                                                                                                                        |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clampGamma(g)`                                                    | `Math.max(0.001, g ?? 1.0)` guard before `1 / gamma` (shared across all eight material constructors).                                                                                           |
| `CameraAwareMaterial` interface                                    | Implemented so `MaterialManager.updateCameraParams(fov, resolution, isOrtho?)` reaches this material.                                                                                           |
| `ColormapAwareMaterial` interface                                  | Implemented so `material-colormap-helpers.ts` sets the LUT texture and scalar range through setters.                                                                                            |
| `GLSL_SANITIZE_FUNCTIONS`                                          | Prepended to the GLSL vertex shader; gives `sanitizePositive` / `sanitizeNonNegative` / `sanitizeAlpha` to clean width/sharpness/alpha inputs against NaN/Inf/out-of-range.                     |
| `sanitizePositive` / `sanitizeNonNegative` / `sanitizeAlpha` (TSL) | TSL counterparts of the GLSL sanitisers — same contract, called inline in the factory.                                                                                                          |
| `volumetric.ts` constants                                          | `ALPHA_CLAMP` (the `1 − 1/512` cap of the `w(a)` map) + the `S(τ)` series thresholds/coefficients — shared with the point/gsplat volumetric branches so all three geometries agree numerically. |
| `proxyIUniform(node)`                                              | Wraps each TSL `UniformNode` in an `IUniform`-shaped getter/setter so `material.uniforms.uX.value = Y` lands on `node.value`. No per-render callback bridge.                                    |

## `isGammaOne` / `isNoGOG` cross-export

`isGammaOne` now lives in `../_shared/uniform-helpers.ts` (shared by all
four geometry types — Point/Line/GSplat/Mesh each gate `LUXAR_GAMMA_ONE`
on it). `material-glsl.ts` re-exports it alongside the line-local `isNoGOG`,
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
- `../../line-geometry.ts` — `InstancedBufferGeometry` builder, the 6-texel layout, and the fused texel writer this shader reads
- `../../material-manager.ts` — creates the per-node line materials and owns the camera-broadcast loop
- `../../picking/line/material.ts` / `material-tsl.ts` — picking counterparts; share the vertex-stage screen-space expansion math
- `../../../tests/e2e/tsl-shader-parity.spec.ts` — GLSL ↔ TSL parity harness
