# Line Material

> Thick-line material stack — instanced-quad geometry, shifted-truncated super-Gaussian soft falloff, and continuous polyline joints — paired across the GLSL `ShaderMaterial` and TSL `NodeMaterial` backends.

This folder holds the six-file material stack that renders one of Luxar's
four first-class geometry types, with two selectable primitives: the
classic screen-space quad described in the next sections, and the capsule
(the default since the #1352 flip — see its own section below). Under the
quad, each line segment is drawn as an instanced
screen-space quad expanded perpendicular to its pixel-space direction; the
fragment stage shades a shifted-truncated super-Gaussian perpendicular
cross-section, with a per-endpoint joint code keeping interior polyline joints
continuous and driving the screen-space miter join. Both backends share
the same `LineMaterialConfig` shape and the same update / clone / blending
semantics — `MaterialManager.getLineMaterial` dispatches on
`RendererCapabilities.apiSurface`, so call sites never see the divergence.

## Module map

| File                     | Role                                                                                                                                                                                                                                   |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material-glsl.ts`       | `LineMaterial extends THREE.ShaderMaterial` — wraps the GLSL3 vertex/fragment pair, owns `uniforms`, manages variant `defines`, applies the canonical blending state. WebGL2 path.                                                     |
| `material-tsl.ts`        | `LineTSLMaterial extends NodeMaterial` — same constructor + update API, but owns persistent `UniformNode`s and rebuilds its TSL graph (`rebuildGraph`) when graph-specialized defines or projection mode flip. WebGPU path.            |
| `shader-glsl.ts`         | `LINE_VERTEX_SHADER` + `LINE_FRAGMENT_SHADER` GLSL3 source strings and the `LINE_SOURCE: ShaderSource` registry entry. The `webgpu` field re-enters `lineWebGPUFactory` so the parity harness can drive both backends from one symbol. |
| `shader-tsl.ts`          | `lineWebGPUFactory(nodes, config, outMaterial?)` — TSL counterpart to the GLSL strings. Reads pre-created `UniformNode`s from a `LineTSLNodes` table and emits the NodeMaterial graph.                                                 |
| `shader-glsl-capsule.ts` | `CAPSULE_LINE_VERTEX_SHADER` + `CAPSULE_LINE_FRAGMENT_SHADER` + `CAPSULE_LINE_SOURCE` — the capsule primitive's GLSL pair (see the section below).                                                                                     |
| `shader-tsl-capsule.ts`  | `capsuleLineWebGPUFactory(nodes, config, outMaterial?)` — TSL twin of the capsule pair; same `LineTSLNodes`/`LineTSLConfig` contract as the screen-space factory.                                                                      |

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
`[1.5 px × max(render-target scale, 1), uMaxLinePixelWidth]` with an intensity-fading `vWidthFade` (CSS-invariant above 1×; the historical framebuffer floor below 1×), then
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
- a **rendered-HALF-width gate** of 2 CSS px (`LINE_JOIN_MIN_HALF_WIDTH`), i.e.
  4 px of rendered width: the wedge has area ~θ·R²/2, so below
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
assert the two backends agree on the VISUAL join — end→start, END–END, a tapered
perspective joint, a joint across the near plane (the only rendered coverage of
the two-sided guard: both sides decline, so the mitred render IS the unmitred
one), and its lowered-cull-plane control, which must differ (`line-join-*` in
the TSL harness) — so a mitred joint is not a WebGL2/WebGPU difference. The pick
stages run the same helper by construction, but no pick fixture carries a
slot-bearing joint code, so a mitred corner's pick footprint is not pixel-pinned
on either backend.

The wedge was given an automated acceptance measurement before it was
closed, and that harness stays. `../../../tests/e2e/line-join-artifact.spec.ts`
renders the `test_line_joins` fixture (five joint cases, one per horizontal
band — smooth curve, 90° zigzag, thin and thick straights, and a nine-ray
hub) and scores every band on one frame with the pure metrics in
`../../../tests/helpers/line-join-metrics.ts`. There are **two** metrics
because each is blind to what the other catches: a local-median outlier
count sees the narrow one-to-two-pixel wedge tick but tracks any smooth
variation invisibly, while an axial flux profile (cross-section sum along
the tube, normalised by its own median) sees exactly the smooth
per-joint dip that was the #780 bead chain and would score zero on the
outlier metric. The spec asserts a gapless flux profile on all five bands,
zero dark and zero bright outliers plus a flat flux profile on both
straight bands — and, since the miter landed, at most two outliers of each
kind on the two bend bands, which both measure zero, so a regression to
unmitred rendering fails it by a wide margin. (The bend ceiling is two rather
than zero only to absorb a seam pixel the float32 operand order can cost;
the spec header quantifies it.)

The before/after on that harness:

| Band                 | Unmitred                  | Mitred          |
| -------------------- | ------------------------- | --------------- |
| `curve_smooth`       | 4.94% dark / 3.52% bright | 0 / 0           |
| `zigzag_right_angle` | flux p05 0.780            | flux p05 0.985  |
| `straight_thin`      | 0 / 0, flat profile       | unchanged       |
| `straight_thick`     | 0 / 0, flat profile       | unchanged       |
| `hub_9ray` (control) | 0.157% / 0.114%           | 0.157% / 0.114% |

Read the metrics module header before quoting one of its numbers. The
local-median count is non-monotone in defect width (a wedge three or more
pixels across poisons its own median and scores zero), which is why the
gentle `curve_smooth` band measured 4.94% dark unmitred while the 90°
`zigzag_right_angle`, whose wedge is far worse but far wider, measured
only 0.076% — and why the zigzag is gated on its flux profile instead.
(Both columns measured in headless Chromium with `dpr=1` pinned on
2026-08-07, the unmitred one via `&lineJoin=none`. The E2E job is not part of the
per-PR CI run; the spec runs under `make test-e2e`.)

## The deleted volumetric primitive (#1352, historical)

A third primitive, `volumetric` (`?linePrimitive=volumetric`), shipped
during the #1352 campaign: each segment drawn as its true 3D density —
the segment convolved with an isotropic 3D Gaussian, solved per fragment
against the camera-space segment in closed form (Gaussian×erf identity),
with bisector-cut joints whose CUT MATH was exact at any bend angle. The
core model and cut algebra were exact; the shipped implementation carried
documented bounded approximations around them (sign-selected mixed-lane
splits at chain ends, a soft/soft near-clip exemption, a Gaussian-only
axial window under non-Gaussian sharpness, stencil truncation past sharp
bends). It measured 2.7–5× the quad's frame cost, and after the capsule
flip a visual A/B found the capsule matched or beat it — including
near-axial, its signature case — so it was deleted rather than maintained
as a third parity surface. The full implementation (GLSL + TSL twins, the
quadrature-validated CPU reference, the Abel-transform sharpness LUT, the
ray-integral shared-math module, and the pick pair) lives in git history
at the deletion's branch point, `1481995d9`.

## Primitive selection: the auto policy

Which primitive a lines node builds is decided ONCE, at material
construction, by `types/line-primitive.ts` — every consumer (visual +
picking, both backends) resolves through the same seam, so the pick
footprint always rasterizes the stencil the eye sees. Precedence:
`?linePrimitive=` (session override, the A/B escape hatch) > the
`Settings → Advanced → Line primitive` policy (`capsule` / `quad`
force one primitive) > the `auto` rule. Auto keeps the capsule default
but builds the cheaper quad when the scene's aggregate concurrent
effective segment load — authored `n_segments` × a rendered-width
factor normalized by each node's authored extent — reaches 2 M. The
threshold is a measured budget choice, not a crossover: on a discrete
NVIDIA GPU the capsule's GPU pass costs ~1.5× the quad at every
thin-line count and 3.16–3.38× on wide lines, while an Apple
GPU barely registers the difference (1.04–1.11×); past ~2 M thin
segments the capsule's GPU pass alone costs over a quarter of a
60 fps frame on the NVIDIA class (4.6 ms of 16.7 ms, versus the
quad's 3.0 ms). Both backends stamp the RESOLVED primitive as
`userData.linePrimitive` and carry it through `clone()`, the
node-factory retro picking pass, and TSL graph rebuilds; sizing never
re-runs after first build. Plain groups and `kind=partition` groups
sum their children; `kind=lod` groups take the maximum because
levels are substitutive. The resulting scene load is installed
before any line material is built, so sibling line nodes resolve
uniformly. Runtime visibility is deliberately not an input: the
authored all-layers-on total is the frozen worst-case budget.

## Capsule primitive (the DEFAULT since the #1352 flip)

THE default line primitive (flipped from `screen-space` after the #1352
re-gate), built after the G1 gate measured the exact volumetric primitive
(the deleted one, section above) at 2.7–5× the quad's frame cost: a
deliberately relaxed model that keeps its two behavioural wins — direction-stable
near-axial rendering (an end-on segment is a round disc, never a flickering
sliver) and seamless bisector-cut joins — at quad-class cost (measured
1.04–1.11× the quad on the 10M-segment worst case, parity at vsync
elsewhere).

The model: each fragment shades a gaussian-like profile of the **2D
point-to-segment distance in pixel space**. The vertex stage emits
stencil-LOCAL coordinates (`vLocal = (axial px, perpendicular px)` plus the
segment's pixel length), so the fragment's distance² is
`y² + max(0, −x, x−L)²` — no projection, no sqrt, no transcendentals. The
profile is the compact quartic bump `(1 − p²)^n` with `p = distance/radius`
and the sharpness map `n = 2^(3 − 4·sharpness)` (smaller exponent = boxier);
the drawn radius is the 2σ support of the quad's Gaussian-equivalent σ
(`CAPSULE_RADIUS_PER_QUAD_HALFWIDTH` — all constants single-sourced in
`_shared/line-capsule.ts`, which also carries the CPU reference profile and
the joint-composition model the unit tests pin — the latter mirrors the
vertex stage's STENCIL as well as the fragment math, so a reach shortfall
chops the model exactly as it would chop the rasterized image, #1488).

Near-plane handling shares the quad's segment cull when both endpoints are
inside `uNearCull` and its clip onto the `nearCull` plane when only one is,
but not the pathological-wide discard: the capsule only clamps its radii to
`uMaxLinePixelWidth`. The shared `perspectiveNearFade` is evaluated PER
CORNER — at the clamped span parameter `tc` of each stencil vertex — and
interpolated as the `fade` factor of `vFade` (which also carries
`widthScale`), not per fragment from `vViewZ` as the quad does. The fade values
are exact at the stencil's axial extremes, not at the drawn endpoints: the
varying spans the cap extensions too, as `_shared/line-capsule.ts` notes.
Even without that extra stretch, interpolation replaces the smoothstep along
the axial span with its chord. A span matching `[nearCull, 2 * nearCull]`
deviates by at most 0.096, while a span clipped at `nearCull` and ending at
`3 * nearCull` under-fades by 0.5 at mid-span, and the maximum error approaches
the full ramp as the far endpoint recedes (the long-segment case called out in
`shader-glsl.ts`). Two legs sharing a vertex also disagree on the fade away
from it (see the `fade` note in `_shared/line-capsule.ts`). Both are further
#1352-licensed approximations alongside the three below. The clip keeps
corners out of the behind-eye hard-zero branch; the ramp reaches zero
continuously at `nearCull`, and the both-near cull only removes a span already
in that zero region, so there is no pop.

Three exactness relaxations are deliberate, licensed by the #1352 relaxed
spec ("not physics-exact; no pathological near-axial drawing; gaussian-like
profile; approximate math fine"):

1. **2D, not 3D.** The distance is measured in screen space, so the profile
   is a screen-space quantity like the quad's — there is no view-ray
   integral and no per-fragment camera-space solve. End-on stability comes
   from the distance field itself (a point's field is radial).
2. **One profile for all blending modes.** The capsule is peak-shaped by
   construction; there is no peak/sum lane split (the deleted volumetric
   primitive's `LUXAR_PEAK_PROJECTION` define went with it). The
   blending-mode tails (volumetric τ map, max premultiply, colormap,
   NO_GOG, gamma) are cribbed from the quad fragment unchanged.
3. **Compact support.** The quartic hits exact zero at the rim (the 2σ
   trim), so there is no truncation constant and no shifted-Gaussian
   renormalization.

Interior polyline joints reuse the joint-code partner machinery
(the per-endpoint partner slot; `compute_joint_codes` stays load-bearing),
including its cap rule: which ends cut at all comes from the shared
`luxarLineJointCapSuppression` / `tslLineJointCapSuppression`, so a free end
(code `0`) and a degree-≥3 hub (code `-2`) keep the whole round cap — a hub
has no single partner to tile against — while a slice-clipped end (`-1`) is
butt-cut at the slice plane.
**Every end is a round cap**; a partner-bearing interior end keeps its HALF
of the joint disc — the cap region (beyond the endpoint) is partitioned along the joint
bisector, the line through the shared vertex with 2D normal
`normalize(q̂ − m̂)` in pixel space (the partner's normal is the exact
negation, so the two half-discs tile the disc exactly at ANY bend angle —
no notch, no chopped miter tip, no double-bright overlap). The cut spans the
full joint plane (cap and body) and composes by the DEFICIT rule over a 1 px
AA ramp: on the partner's side each leg renders `max(mine − partner, 0)`, so
the additive pair composes to `max(mine, partner)`. The normal is kept
unquantised on purpose: each leg builds it in its own local basis, so any
per-leg rounding never cancels, and the disagreement grows with distance
from the joint (#1502). Congruent legs turning 120° or less (the common case)
gate to an exact zero-double-count partition — the same domain partition the
deleted volumetric primitive integrated per ray — while tapered or perspective-diverged
partners get exactly the light a pure partition would chop (a fat vertex's disc
keeps the half a thin neighbour cannot render; the numeric composition sweep in
`line-capsule.test.ts` pins the three reconstruction errors of
#1494/#1488/#1490, and two source locks additionally hold the shader text
across all four surfaces:
`tests/unit/rendering/materials/line/capsule-partner-radius.test.ts` for the
shared-vertex base radius of #1494, and
`tests/unit/rendering/materials/line/capsule-joint-packet-source-lock.test.ts`
for the rest of the deficit packet — the far-capped rod of #1490, both packet
lanes and the packing that transports them, and the gate clauses of
#1495/#1501). Two documented exceptions to that last sentence, both in
`CAPSULE_JOINT_PACKET_MIN_RADIUS_PX`: the deficit packet is skipped below a
4 CSS px stencil half-width, so a GENTLE joint thinner than that keeps the plain
cut — within 0.03 of peak of what a congruent joint at the same angle and
radius costs anyway, though in absolute terms that reaches −0.09 at 90° and
−0.19…−0.39 at 120° — and the sharp-turn exception that overrides the skip
stops at the AA radius floor, because below it the two legs' `widthScale`
factors disagree, so the PAIR no longer composes to `max` even though each
deficit stays bounded by its own unscaled profile (#1495).
Note also the one exception to "keeps its HALF" above: the STENCIL a cut end
reserves is the full disc, not the half, whenever a deficit packet exists —
the deficit term is bounded by the leg's own profile, so nothing shorter
covers what it draws (#1488).
The partner's far endpoint is near-plane-clipped
toward the joint vertex before projecting (a behind-eye projection flips
and would poison the cut normal), with a joint vertex behind the near
plane keeping the perpendicular butt. The per-fragment radius is computed EXACTLY from the
endpoint radii (`mix(rA, rB, clamp(x/L, 0, 1))` — a linear varying cannot
represent this, since its interpolation spans the cap extensions).

Picking follows the toggle: the capsule pick
shaders run the same stencil and joint partition as the visual pair so the
pick footprint tracks the pixels exactly (see `../../picking/line/README.md`).
The primitive is BUILD-time, like the screen-space quad: it selects the source pair /
TSL factory at material construction and never changes on a live material.

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
construction and the fused texel writer. The texture width used by the
prologue's `%`/int-div addressing is a **baked compile-time constant**
(`LUXAR_LINE_TEX_W` define / TSL literal from `getElementTextureWidth`),
not a per-vertex `textureSize` query — the constant lets the shader
compiler strength-reduce the integer division, measured −7% on the
quad's whole GPU pass at 4 M segments (a uniform recovered almost none
of it; the width is a session constant capped at 4096, so baking is
safe).

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

Both backends share the same six `#define`s, set by the wrapper and
either gated via `#ifdef` (GLSL) or read at TSL build time
(`rebuildGraph` re-runs the factory):

| Define                          | Effect                                                                                                                                                                | Set by                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `USE_COLORMAP`                  | Replaces the texel2/3 per-endpoint RGB with the texel5 scalars + LUT lookup (presence rides the geometry's `userData.hasScalars` stamp)                               | `setColormapTexture(texture)` / `updateColormapTexture`      |
| `LUXAR_GAMMA_ONE`               | Skips three per-fragment `pow()` calls when `gamma == 1.0 ± 1e-4` (the default)                                                                                       | `updateGamma` when crossing the threshold                    |
| `LUXAR_NO_GOG`                  | Skips the `vColor × uIntensity + uOffset` chain and its `max(·, 0)` clamp when `intensity==1 && offset==0`                                                            | `updateIntensity` / `updateOffset` via `_refreshNoGOGDefine` |
| `LUXAR_MAX_RGB_CONTRIBUTION`    | Premultiplies `rgb *= intensity × opacity` so `CustomBlending + MaxEquation + OneFactor/OneFactor` captures contribution-weighted colour rather than flat full-bright | `applyBlendingMode('max')`                                   |
| `LUXAR_OPAQUE_RGB_CONTRIBUTION` | Discards sub-1e-4 alpha-weighted RGB contributions before an `opaque` fragment can write depth                                                                        | `applyBlendingMode('opaque')`                                |
| `LUXAR_VOLUMETRIC`              | Switches the fragment output to the emission–absorption branch (τ = κ·alpha, `S(τ)` screening, `1 − e^(−τ)` alpha — see the volumetric section above)                 | `applyBlendingMode('volumetric')`                            |

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

| Symbol                                        | Used for                                                                                                                                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clampGamma(g)`                               | `Math.max(0.001, g ?? 1.0)` guard before `1 / gamma` (shared across all eight material constructors).                                                                                           |
| `CameraAwareMaterial` interface               | Implemented so `MaterialManager.updateCameraParams(fov, resolution, isOrtho?)` reaches this material.                                                                                           |
| `ColormapAwareMaterial` interface             | Implemented so `material-colormap-helpers.ts` sets the LUT texture and scalar range through setters.                                                                                            |
| `GLSL_SANITIZE_FUNCTIONS`                     | Prepended to the GLSL vertex shader; gives `sanitizePositive` / `sanitizeNonNegative` / `sanitizeAlpha` to clean width/sharpness/alpha inputs against NaN/Inf/out-of-range.                     |
| `sanitizeNonNegative` / `sanitizeAlpha` (TSL) | TSL counterparts of those two GLSL sanitisers — same contract, called inline in the factory. `sanitizePositive` has no TSL twin: no TSL shader calls it.                                        |
| `volumetric.ts` constants                     | `ALPHA_CLAMP` (the `1 − 1/512` cap of the `w(a)` map) + the `S(τ)` series thresholds/coefficients — shared with the point/gsplat volumetric branches so all three geometries agree numerically. |
| `proxyIUniform(node)`                         | Wraps each TSL `UniformNode` in an `IUniform`-shaped getter/setter so `material.uniforms.uX.value = Y` lands on `node.value`. No per-render callback bridge.                                    |

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
backends and pixel-compares. GLSL3 sources of the shipping primitives are
**never** deleted from this folder — they remain the readable reference
even after the TSL path stabilises.

## See Also

- `../_shared/README.md` — shared infrastructure, `ShaderSource` pattern, `buildMaterial` dispatch
- `../../README.md` — Rendering package overview and where line materials sit in the pipeline
- `../../line-geometry.ts` — `InstancedBufferGeometry` builder, the 6-texel layout, and the fused texel writer this shader reads
- `../../material-manager.ts` — creates the per-node line materials and owns the camera-broadcast loop
- `../../picking/line/material.ts` / `material-tsl.ts` — picking counterparts; share the vertex-stage expansion math of whichever primitive the session resolves (screen-space quad or capsule)
- `../../../tests/e2e/tsl-shader-parity.spec.ts` — GLSL ↔ TSL parity harness
- `../../../tests/e2e/line-join-artifact.spec.ts` / `../../../tests/helpers/line-join-metrics.ts` — the joint-artifact acceptance measurement described above (#780 / #785 / #790)
