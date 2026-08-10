/**
 * The #1352 volumetric-line primitive's ray integral — the single source
 * of truth for its math, CPU and GPU.
 *
 * Lives under `line/` rather than `_shared/` because it is line-specific
 * by name, by every export, and by its only plausible consumer; the
 * folder it sits in already holds both backends' shader sources, so a
 * module shared *between backends* but not *between geometries* belongs
 * here. Only `erf.ts` — genuinely geometry-agnostic — is imported from
 * `../_shared/`.
 *
 * ## The primitive
 *
 * A line segment is modelled as a *cylindrically symmetric* density: the
 * segment convolved with an isotropic 3D Gaussian of width `sigma`,
 *
 * ```text
 *   (segment ⊛ G_3D)(x) = σ√(2π) · G_2D(r) · W(s)
 *   G_2D(r) = exp(-r² / (2σ²))                              (peak 1)
 *   W(s)    = ½[ erf(s / (σ√2)) − erf((s − L) / (σ√2)) ]
 * ```
 *
 * with `r` the radial distance from the segment's *infinite axis*, `s`
 * the axial coordinate measured from the segment start, and `L` the
 * segment length. Folding the `σ√(2π)` into the per-element amplitude,
 * the density this module integrates is
 *
 * ```text
 *   rho(x) = a · G_2D(r) · W(s)
 * ```
 *
 * i.e. `a` is the peak density mid-segment on the axis. Unlike the
 * screen-space quad it replaces, this object has no preferred viewing
 * direction — which is the point: the quad degenerates to a zero-area
 * sliver when the segment points at the camera, and that degeneracy is
 * the artifact #1352 is about.
 *
 * ## The closed-form ray integral (sum-blend modes)
 *
 * Swapping the order of integration — first integrate `G_3D` along the
 * view ray (giving a 2D Gaussian of the point-to-ray distance), then
 * integrate along the segment — turns the double integral into a single
 * erf window. For a unit-direction ray with
 *
 *   - `|u|`   = magnitude of the ray direction's component perpendicular
 *               to the axis (= sin of the ray/axis angle), in [0, 1],
 *   - `D`     = closest-approach distance from the ray *line* to the
 *               segment's *infinite axis*,
 *   - `s*`    = axial coordinate of that closest-approach foot,
 *
 * the integral is
 *
 * ```text
 *   I = a · exp(-D²/(2σ²)) · (σ√(2π) / |u|) · ½[ erf(A) − erf(B) ]
 *   A = s*·|u| / (σ√2),   B = (s* − L)·|u| / (σ√2)
 * ```
 *
 * Two limits are exact and both matter:
 *
 *   - **SIDE-ON** (`|u| → 1`, `L ≫ σ`, foot mid-segment): the erf window
 *     saturates to 1 and `I → a · σ√(2π) · exp(-D²/(2σ²))` — the
 *     *unshifted, untruncated* Gaussian cross-section. See "Relation to
 *     what the current shader draws" below: this is close to, but
 *     deliberately not identical to, today's stroke.
 *   - **END-ON** (`|u| → 0`): the `1/|u|` cancels analytically against a
 *     vanishing erf window and `I → a · L · exp(-D²/(2σ²))` — path length
 *     times the radial profile. Finite, orientation-free, no sliver.
 *
 * ## Normalization
 *
 * **The two sum-mode entry points ({@link lineRayIntegralRef} and
 * {@link lineRayIntegralPoly}) return `I / (a · σ√(2π))`.** So:
 *
 *   - side-on through the middle of a long segment at `D = 0` is exactly
 *     **1.0** (matching the current renderer's per-fragment peak),
 *   - end-on is `L / (σ√(2π)) · exp(-D²/(2σ²))` — brighter than side-on
 *     by the honest chord/path factor, which is the visible payoff.
 *
 * The caller multiplies by the per-element amplitude `a`. Because the
 * normalization is a constant for a given `sigma`, every linearity
 * property (see *additivity* below) holds for the normalized values too.
 * The other exports are not integrals and carry no such factor:
 * {@link lineCapsuleProfileRef} is a peak-1 profile,
 * {@link raySegmentDistance} / {@link raySegmentGeometry} return world
 * lengths, and {@link lineSigmaFromAuthoredWidth} returns a sigma.
 *
 * ## Peak (max-blend) modes use a different evaluation
 *
 * A max-composited mode must not integrate — summing along the ray and
 * then taking a max across segments double-counts depth. The peak family
 * instead evaluates the *Gaussian-shouldered capsule*: the radial profile
 * of the ray-to-**finite-segment** distance,
 * `exp(-dist(ray, segment)² / (2σ²))`. Its max-composition across
 * segments is seamless because the distance field itself is continuous
 * across a shared joint. See {@link lineCapsuleProfileRef}.
 *
 * ## Sigma calibration — and the factor of 2 that is easy to get wrong
 *
 * `sigma = 2 · width / GAUSSIAN_EQUIVALENT_TRUNCATION`
 * (see {@link lineSigmaFromAuthoredWidth}). The `2` is **not** a fudge:
 * the authored per-vertex `width` attribute is documented as a
 * half-width (`rendering/line-geometry.ts` ~:460, "widths in the codebase
 * are half-widths per the rendering spec"), but the vertex stage doubles
 * it on the way to the quad edge:
 *
 *   - `luxarLineEndPixelWidth = width · uPerspectiveLineScale / z` with
 *     `uPerspectiveLineScale = resY / tan(fov/2)`
 *     (`_shared/glsl-lib.ts` :286-290, `line/shader-glsl.ts` :100),
 *   - whereas pixels-per-world-unit at depth `z` is
 *     `resY / (2·z·tan(fov/2))` — so that expression is
 *     `width · 2 · pxPerUnit`,
 *   - and it is then applied as `cornerOffset · aQuadCorner.y` with
 *     `aQuadCorner.y ∈ {−1, +1}` (`line/shader-glsl.ts` :505-511), so the
 *     world-space radius at `|vPerpNorm| = 1` is `2 · width`.
 *
 * Corroborated twice in-repo: `tests/e2e/line-perf-bench.spec.ts` :106
 * ("drawn half-width = 2 × width texel = 6 world units" for width 3.0),
 * and Python's `luxar/gsplats/lift.py` :365 (`sigma = 2.0 · wbar ·
 * radius_scale / T`). **Do not "simplify" the 2 away.**
 *
 * `falloff.ts` establishes that the sprite profile `exp(-K·p²)` on the
 * normalized coordinate `p = r / (2·width)` *is* a Gaussian truncated at
 * `GAUSSIAN_EQUIVALENT_TRUNCATION` sigmas, which is why reusing that
 * constant makes the side-on limit land on the current renderer's shape
 * rather than merely near it. There is no second truncation constant **in
 * the viewer**; Python's `lift.py` :99 deliberately uses
 * `LIFT_TRUNCATION_RADIUS = 3.0` instead of 3.034854, a 1.15% smaller
 * sigma, and `falloff.ts` :58-59 documents why that divergence is
 * intentional.
 *
 * ## Relation to what the current line shader draws — NOT identical
 *
 * The side-on limit is the *bare* `exp(-K·p²)`. Today's fragment
 * (`line/shader-glsl.ts` :578) draws the **shifted, truncated,
 * renormalized** super-Gaussian
 * `max(exp(-K·p^β) − C, 0) · INV_ONE_MINUS_FALLOFF_FLOOR`, and discards
 * at `p ≥ 1` (:564). Measured divergence at `β = 2`, as a fraction of the
 * bare value: **0.34% at p = 0.25, 2.18% at p = 0.5, 12.5% at p = 0.75,
 * 41.1% at p = 0.9**, and total at `p = 1` (0.01 vs exactly 0). The 1D
 * cross-section carries **+1.46%** more mass inside `p ≤ 1` and
 * **+1.71%** counting the tail the quad discards.
 *
 * That is a feature, not an oversight: the shift-and-truncate exists so a
 * *rasterized quad* can end without a visible ring, and it is exactly
 * what breaks additivity (a truncated profile does not telescope across a
 * split). The volumetric primitive un-truncates the profile, which is
 * what buys the exact seams below. Broadside rendering therefore changes
 * slightly — it is not "pixel-neutral" — and the change is a soft ~1.7%
 * brighter halo, concentrated in the outer third of the stroke.
 *
 * ## Sharpness / `β` is NOT covered by this slice
 *
 * Lines carry `sharpness` as a first-class attribute, and the fragment
 * maps it to `β = 2^(6s − 2)`. Bundled demos author `s` of 0.45, 0.5,
 * 0.55, 0.8 and 0.85 — i.e. `β` of 1.63, 2 (the default), 2.46, 6.96 and
 * 8.57. **Everything in this module is the `β = 2` Gaussian only.** There
 * is no closed-form Abel transform of `exp(-K·p^β)` for general `β`, and
 * at `β = 8.57` the current stroke is a near-hard-edged flat-top that no
 * Gaussian approximates. This mirrors `lift.py` :49-52, which states the
 * same restriction for the point→gsplat lift and its rationale.
 *
 * The wiring slice must decide what to do here. Issue #1352's decision 1
 * proposes a `(d/σ, β)` Abel-transform LUT for the sum modes and
 * pointwise evaluation for the peak modes; nothing in this module
 * prejudges that.
 *
 * ## The numerical lanes
 *
 * Write `Δ = A − B = L·|u| / (σ√2)` (the erf window's argument gap) and
 * `M = (A + B)/2 = (s* − L/2)·|u| / (σ√2)` (its midpoint). Then
 *
 * ```text
 *   I_normalized = exp(-D²/(2σ²)) · [L/(σ√2)] · ½·DQ,
 *   DQ = (erf(A) − erf(B)) / Δ
 * ```
 *
 * because `L/(σ√2) = Δ/|u|`. **This reformulation removes `1/|u|` from
 * the code entirely** — nothing here ever divides by `|u|`.
 *
 * ### Lane (a) `|u| → 0` and lane (b) near-coincident erf arguments are
 * ### the SAME lane
 *
 * The `|u| → 0` end-on limit is a `0/0`, and `erf.ts` separately warns
 * that `erfPoly`'s difference quotient is only trustworthy for arguments
 * ≥ 0.5 apart. These look like two hazards but `Δ = L·|u|/(σ√2)` ties
 * them together: small `|u|` *is* small `Δ`, and a segment short relative
 * to `σ` is small `Δ` too. Once the integral is written as
 * `[L/(σ√2)]·½·DQ`, both collapse into one question — *is the gap `Δ`
 * large enough to difference two erfs?* — and one branch handles both.
 * The end-on limit falls out for free: `DQ → erf'(0) = 2/√π` as
 * `Δ, M → 0`, giving `I → L/(σ√(2π))`, exactly the analytic answer. So
 * this module implements **one lane split, keyed on `Δ`**:
 *
 *   - **Closed-form lane** (`Δ > {@link LINE_WINDOW_GAP_THRESHOLD}`):
 *     `DQ = (erf(A) − erf(B)) / Δ` directly.
 *   - **Derivative (Taylor) lane** (`Δ ≤ threshold`): the central
 *     expansion of the difference quotient about `M`, with `h = Δ/2`,
 *     ```text
 *       DQ ≈ erf'(M) · [ 1 + h²(2M² − 1)/3 + h⁴(4M⁴ − 12M² + 3)/30 ]
 *     ```
 *     (from `erf(M±h)` Taylor series; the odd orders cancel, so the
 *     surviving terms are `h²·erf'''/6` and `h⁴·erf⁽⁵⁾/120`).
 *
 * ### Why the threshold is 0.5
 *
 * Measured, not chosen by taste. Three curves meet here:
 *
 *   1. **`erfPoly` fidelity sets the floor.** `erf.ts` documents that the
 *      polynomial's difference quotient is only trustworthy for gaps
 *      ≥ 0.5, and that near saturation the raw window can come out
 *      *negative*. Measured worst negative window quotient: **−6.3e-4 at
 *      Δ = 0.5**, degrading to **−2.5e-3 at Δ = 0.3** and **−6.4e-3 at
 *      Δ = 0.1**. Measured closed-form-lane error vs the exact erf:
 *      **1.44e-3 at Δ = 0.5** → **3.00e-3 at Δ = 0.3**. Going below 0.5
 *      degrades fast.
 *   2. **Taylor truncation sets the ceiling.** The 3-term expansion's
 *      worst absolute error in `DQ` (whose peak is `2/√π ≈ 1.128`) is
 *      **7.0e-6 at Δ = 0.5**, rising to **1.07e-4 at Δ = 0.8** and
 *      **4.0e-4 at Δ = 1.0**. Going above 0.5 degrades fast too.
 *   3. **float32 cancellation is NOT the binding constraint** — the usual
 *      suspect, and it is three orders of magnitude away. Differencing
 *      two O(1) erfs in float32 costs ~1.2e-7 absolute; divided by Δ that
 *      is a measured **1.7e-7 at Δ = 0.5**, and it only reaches
 *      `erfPoly`'s own 5.4e-4 error floor near **Δ ≈ 2.1e-4**. The
 *      polynomial-fidelity floor is ~2400× more restrictive, so it is
 *      what picks the number.
 *
 * At Δ = 0.5 the two lanes agree to **7.0e-6 absolute** in `DQ`
 * (**4.5e-5 relative** wherever `DQ > 0.01`) when both use the accurate
 * `erfRef`; on the shader path the crossing step is **1.45e-3**, which is
 * `erfPoly`'s own error rather than anything the lane split adds. Both
 * numbers are pinned by `ray-integral.test.ts`.
 *
 * ### Why the derivative lane uses `exp`, not the polynomial's derivative
 *
 * The tempting shader-cheap move — differentiate `ERF_POLY_COEFFS`
 * analytically and get `erf'` with no `exp` — does not survive
 * measurement. `erfPoly` is a *constrained least-squares fit of erf*, not
 * of `erf'`: `max |P'(x) − erf'(x)|` is **2.15e-2 on [0, 3]** (at the
 * constrained endpoint `x = 3`, where `P'(3) = 2.16e-2` against the true
 * `erf'(3) = 1.39e-4` — off by 155×), and still **2.13e-3 on [0, 2]**,
 * four times worse than `erfPoly` itself. So the derivative lane
 * evaluates the exact `erf'(M) = (2/√π)·exp(-M²)` on both CPU and GPU.
 * That costs one extra `exp` in the fragment — a second SFU op next to
 * the `exp(-D²/(2σ²))` the fragment already computes, not a new class of
 * cost — and it keeps the CPU reference and the shader mirror
 * *structurally identical* in that lane.
 *
 * ### Non-negativity, division guards, and the float32 overflow guards
 *
 * The erf window is clamped at zero ({@link erfPoly} can return a
 * slightly negative window near saturation — `erf.ts` documents −8.8e-4).
 * A coverage weight must never go negative, and a negative sum-mode
 * contribution would *darken* a pixel another segment already lit. The
 * derivative lane is non-negative by construction: its bracket is
 * ≥ 0.9796 for every `h ≤ threshold/2` (minimum at `h = 0.25, M = 0`).
 *
 * Both lanes are evaluated unconditionally and combined with
 * `mix(closed, taylor, step(gap, threshold))` — GPUs evaluate both sides
 * of a select anyway, and a `mix` generates flat, branch-free code on
 * both backends. Because `mix(x, y, a) = x·(1−a) + y·a`, **an `Inf` in
 * the UNSELECTED arm becomes `Inf·0 = NaN` and destroys a perfectly good
 * result in the other one.** Three guards make both arms finite:
 *
 *   - the closed lane divides by `max(Δ, threshold)` — identical inside
 *     the selected regime, finite outside it (the `volumetric.ts`
 *     `max(τ, eps)` pattern);
 *   - the derivative lane's half-gap is `0.5·min(Δ, threshold)`, so its
 *     `q = h²` is bounded by 0.0625 instead of growing like `Δ²/4`;
 *   - the derivative lane's `M²` is clamped to
 *     {@link LINE_TAYLOR_MM_CLAMP} = 104, because `exp(-M²)` underflows
 *     to exactly 0 in float32 above 104 while the polynomial `series`
 *     keeps growing — `0 · Inf = NaN` again.
 *
 * The two min-guards protect **different** regimes and each is pinned by
 * its own reproducer, because either one alone rescues the headline case
 * (`σ = 0.0033`, `L = 1000`, `s* = 817`, `|u| = 1` — a *broadside*
 * fragment on a thin, long segment, exact answer 1.0, fp32 `NaN` with
 * both guards gone):
 *
 *   - **half-gap** — needed when `Δ` itself is astronomical. Measured
 *     onset with only this guard removed: clean to `L/σ = 1e9`, `NaN` at
 *     `1e10` (statement-granularity fp32 model; a per-operation model
 *     puts it far lower, near `L/σ ≈ 1.6e5`, because it rounds the
 *     `q²·4M⁴` product as it forms).
 *   - **`M²`** — needed when `|s*| ≫ L`, and this one bites in the
 *     **SELECTED** lane, where nothing else can help: at `σ = 1`,
 *     `L = 1`, `s* = 1e11`, `|u| = 0.7` (gap 0.495, derivative lane) the
 *     shipped shader returns `2.8e-45` and the mm-inert one `NaN`.
 *
 * 104 is the float32 underflow point itself (`fround(exp(-103))` =
 * 1.4e-45 > 0, `fround(exp(-104))` = 0), so **in float32** the clamp is
 * a no-op wherever the lane's value is nonzero. It is NOT a no-op in the
 * float64 CPU reference, which shares the code path and has ~1000× more
 * exponent range: at `σ = 1, L = 1, s* = 100, |u| = 0.7` the reference
 * returns `2.85e-45` where the unclamped math is exactly 0. Negligible
 * in magnitude, but it means the reference no longer returns *exactly*
 * zero far past the segment ends inside the derivative lane (the closed
 * lane still does — its window saturates to 0). Pinned as behaviour, not
 * papered over.
 *
 * Both min-guards are **bit-neutral** where they are selected, because
 * `min(x, c) === x` throughout the selected regime; measured difference
 * over the test's own 50,000-sample float64 sweep and an 8,100-point
 * float32 grid: exactly 0, with the fp32 grid going from 565 NaN/negative
 * samples to none. The one precondition left is that `L/(σ√2)` itself be
 * finite in the target precision.
 *
 * ### erfPoly argument range
 *
 * `A` and `B` are unbounded (a distant ray gives huge `|s*|`), but both
 * `erfPoly` and its GLSL twin `luxarErf` clamp `|x|` to
 * `ERF_POLY_CLAMP = 3` internally, so the [-3, 3] validity range is
 * respected without a second clamp here. Nothing in this module defeats
 * that clamp, and `A`/`B` are handed to the erf unmodified.
 *
 * ## HAZARD: `D` and `s*` are individually discontinuous at `|u| = 0`
 *
 * As the ray swings through parallel-to-the-axis, the mutual
 * perpendicular's foot escapes to infinity: `s* → ±∞` while `|u| → 0`,
 * and `D` (a line-to-line distance) can jump — a ray that *crosses* the
 * axis at any nonzero angle has `D = 0` right up to the moment it becomes
 * parallel and its `D` becomes the full offset. The integral is
 * nevertheless continuous, because
 * `exp(-D²/(2σ²))·exp(-M²) = exp(-dist(midpoint, ray)²/(2σ²))` and the
 * products `s*·|u|` stay bounded.
 *
 * {@link raySegmentGeometry} derives the triple stably **in float64, on
 * the CPU**. The GPU-side derivation is deliberately **out of scope for
 * this slice** and is the wiring slice's genuinely hard problem: this is
 * where the real `1/|u|` divergence lives, `RAY_SEGMENT_PARALLEL_EPS`'s
 * value is float64-specific (see its docblock — a float32 consumer needs
 * {@link RAY_SEGMENT_PARALLEL_EPS_F32} or larger), and a ray through a
 * pixel on a segment aimed at the camera has `|u| = 0` *exactly*.
 *
 * ## Additivity — why every joint is seamless
 *
 * Splitting a segment at an interior point telescopes exactly:
 * `[erf(A) − erf(A_c)] + [erf(A_c) − erf(B)] = erf(A) − erf(B)`. The
 * un-normalized integrals of two collinear halves sum to the whole's, for
 * *every* ray direction. That is the design's central claim and it is
 * pinned by the additivity test (residual: float64-exact on the `erfRef`
 * path within one lane, 5.2e-4 absolute on the `erfPoly` path).
 *
 * ## What the WIRING slice will have to retire — the blast radius
 *
 * Additivity does not merely make the following redundant, it makes them
 * **double-count**. Named here so the next slice does not inherit them by
 * default; **nothing in that code is changed by this slice**.
 *
 *   - The **joint-code / cap-suppression subsystem** —
 *     `capFactor = min(startCap, endCap)`, `GLSL_LINE_JOIN` /
 *     `tslLineJoin`, `luxarLineJointCapSuppression`, the texel4.yz joint
 *     codes — was built across #780/#785/#790/#1342 purely to stop two
 *     rasterized quads from double-covering (or under-covering) a shared
 *     joint. An additive world-space integral already sums to the exact
 *     union there, so keeping the suppression would now make joints
 *     *dark*: it would UNDER-count.
 *   - **`widthScale = min(vPixelWidth / 1.5, 1.0)`** and the **`edgeAA`
 *     smoothstep** are screen-space coverage compensations for a
 *     rasterized quad of finite pixel width. Multiplying a world-space
 *     line integral by either double-counts coverage that the integral
 *     already accounts for.
 *
 * ## What this module does NOT fix on its own
 *
 * The header frames the end-on sliver as the artifact — but a fragment
 * shader only runs where the **quad rasterizes**, and end-on the quad IS
 * the sliver. Swapping the fragment math alone therefore changes nothing
 * in the very configuration that motivates the change. The wiring slice
 * needs a conservative screen-space footprint as well: roughly the
 * projection of a disc of radius `T·σ` about the segment (a
 * capsule-shaped or billboarded-bounding-quad expansion), so there are
 * fragments to evaluate this integral at. Read this module as the
 * *shading model*, not as the whole fix.
 *
 * ## Single-source pattern
 *
 * `GLSL_LINE_RAY_INTEGRAL_FUNCTIONS` and the TSL builders are generated
 * from the same exported constants, which are themselves stored
 * pre-rounded to their `toFixed(9)` serialization (the `falloff.ts`
 * trick) so the CPU mirror, the GLSL literal and the TSL `float()` carry
 * bit-identical values. As in `erf.ts`, the guarantee is **value-level,
 * not textual**: TSL literal formatting belongs to Three's code
 * generator.
 *
 * Sharing the constants is necessary but **not sufficient** — it does
 * not constrain a hand-written expression tree, and the TS/GLSL unit
 * tests never touch the node graph. What backs the TSL half is the
 * `erf.ts` apparatus, replicated here:
 *
 *   - `tests/e2e/harnesses/tsl-harness/line-ray-integral.ts` renders the
 *     GLSL block and the TSL builders side by side over a 2D sweep that
 *     crosses the lane threshold, and `tsl-shader-parity.spec.ts`
 *     pixel-compares them;
 *   - `tests/__codegen__/line-ray-integral.fragment.glsl.txt` pins the
 *     generated code textually, so a sign flip, an inverted `mix`, a
 *     reversed `step`, a dropped `1/√2` or a missing guard shows up as a
 *     snapshot diff.
 *
 * Both are Playwright specs and therefore run under `make test-e2e`,
 * not in the per-PR unit run — the same coverage boundary `erf.ts` has.
 *
 * No production shader consumes this module yet — the renderer wiring is
 * the next step of #1352. Until then its consumers are the CPU reference
 * and the unit tests, exactly as `erf.ts`'s polynomial was staged.
 *
 * @module rendering/materials/line/ray-integral
 */

import { exp, float, max, min, mix, sqrt, step } from 'three/tsl';
import { erfPoly, erfPolyTSL, erfRef } from '../_shared/erf';
import { GAUSSIAN_EQUIVALENT_TRUNCATION } from '../_shared/falloff';
import type { TSLNode } from '../_shared/tsl-helpers';

/**
 * Round a constant to the decimal form the GLSL literal will carry, so
 * the CPU mirror, the GLSL string and the TSL `float()` all evaluate the
 * *same* number. Same rationale as `falloff.ts`'s `FALLOFF_K`: the
 * rounding is not cosmetic, it is what makes value-level parity exact
 * rather than approximate. 9 decimals is `erf.ts`'s serialization width.
 */
function fixed9(v: number): number {
  return Number(v.toFixed(9));
}

/** Serialize a constant into GLSL exactly as `erf.ts` does. */
function glslNum(v: number): string {
  return v.toFixed(9);
}

/**
 * `1/√2`, the scale that turns an axial/radial distance in world units
 * into an erf argument once divided by `sigma`. Pre-rounded — see
 * `fixed9`.
 */
export const LINE_INV_SQRT2 = fixed9(Math.SQRT1_2);

/**
 * `erf'(0) = 2/√π`, the peak of the difference quotient and the value the
 * end-on limit reduces to. Pre-rounded — see `fixed9`.
 */
export const LINE_TWO_OVER_SQRT_PI = fixed9(2 / Math.sqrt(Math.PI));

/**
 * The erf-window gap `Δ = A − B` below which the difference quotient is
 * replaced by its central derivative expansion. **0.5**; the three
 * measurements that pick it are in the module header (`erfPoly` gap
 * fidelity floor, Taylor truncation ceiling, and the float32 cancellation
 * floor that turns out to be ~2400× looser).
 */
export const LINE_WINDOW_GAP_THRESHOLD = fixed9(0.5);

/**
 * Coefficient of the `h²(2M² − 1)` term of the difference quotient's
 * central expansion: `h²·erf'''(M)/6` divided through by `erf'(M)`.
 */
export const LINE_WINDOW_TAYLOR_C2 = fixed9(1 / 3);

/**
 * Coefficient of the `h⁴(4M⁴ − 12M² + 3)` term: `h⁴·erf⁽⁵⁾(M)/120`
 * divided through by `erf'(M)`. Dropping it would cost a factor ~60 in
 * lane accuracy at the threshold (7.0e-6 → 4.3e-4).
 */
export const LINE_WINDOW_TAYLOR_C4 = fixed9(1 / 30);

/**
 * Upper clamp on `M²` inside the derivative lane. It stops the
 * polynomial `series`, which keeps growing in `M²`, from reaching `Inf`
 * and turning `0 · Inf` into `NaN`.
 *
 * 104 is exactly the float32 underflow point of `exp(-M²)`
 * (`fround(exp(-103)) = 1.4e-45 > 0`, `fround(exp(-104)) = 0`), so **in
 * float32** the clamp is a no-op wherever the lane's value is nonzero.
 * In the float64 CPU reference it is not: it makes the reference return
 * ~1e-45 instead of exactly 0 far past the segment ends inside the
 * derivative lane. Both halves of that are pinned by the tests — the
 * underflow property from both sides, and the float64 residue as
 * behaviour. See the module header.
 */
export const LINE_TAYLOR_MM_CLAMP = fixed9(104);

/**
 * The factor by which the vertex stage scales the authored per-vertex
 * `width` on its way to the quad edge, i.e. the world-space radius at
 * `|vPerpNorm| = 1` is `LINE_RENDERED_HALF_WIDTH_FACTOR · width`. The
 * derivation and its three in-repo corroborations are in the module
 * header's calibration section — read it before touching this.
 */
export const LINE_RENDERED_HALF_WIDTH_FACTOR = 2;

/**
 * Below this `|u|` the ray is treated as exactly parallel to the axis by
 * {@link raySegmentGeometry}. Purely a *geometry-derivation* guard, not a
 * math lane: the ray integral itself needs no such cutoff (its single
 * lane split is on `Δ`, and `Δ = 0` is a perfectly good input).
 *
 * **This value is float64-specific.** The cross product `V × d̂` loses
 * relative accuracy like `ε/|u|`, and 1e-7 is chosen so a float64 helper
 * keeps `D` and `s*` to ~1e-9 relative. That is the accuracy of the
 * *inputs*, not of the *result*: switching lanes at `|u| = ε` moves `M`
 * by roughly `lever · offset · k² · ε` (with `lever` the axial distance
 * from the ray origin to the segment and `k = 1/(σ√2)`), so the relative
 * jump in the integral is about `2 · lever · offset · k² · ε` and grows
 * without bound as the lever arm grows or `σ` shrinks. Measured at
 * `L = 10`, offset `0.3σ`: **1.7e-6** at lever 50 / σ = 1, **3.0e-4** at
 * lever 1e4 / σ = 1, **3.0e-2** at lever 1e6 / σ = 1, and **6.0e-3** /
 * **4.6e-1** for the same two levers at σ = 0.05 (the predictor is within
 * 30% of every one of those).
 *
 * A float32 consumer must NOT copy this number:
 * {@link RAY_SEGMENT_PARALLEL_EPS_F32}.
 */
export const RAY_SEGMENT_PARALLEL_EPS = 1e-7;

/**
 * The float32 sibling of {@link RAY_SEGMENT_PARALLEL_EPS}. 1e-7 is BELOW
 * float32 machine epsilon (1.19e-7), so a shader that copies it gets no
 * guard at all — every `|u|` it can distinguish from zero is already
 * above the threshold, and `|u| = 0` exactly (a ray through a pixel on a
 * segment aimed at the camera) then divides by zero. 1e-3 keeps the
 * `ε/|u|` amplification of a float32 cross product near 1e-4 relative,
 * which is the loosest a shader can afford; a consumer with a large axial
 * lever arm needs more, per the sensitivity bound above.
 */
export const RAY_SEGMENT_PARALLEL_EPS_F32 = 1e-3;

/** A plain 3-vector; keeps this module free of a THREE dependency. */
export type Vec3 = readonly [number, number, number];

/**
 * Convert a line's **authored** per-vertex `width` attribute to the
 * Gaussian sigma of the volumetric primitive:
 * `sigma = 2 · width / GAUSSIAN_EQUIVALENT_TRUNCATION`.
 *
 * The name says *authored* because the factor of 2 is the whole trap:
 * `width` is a half-width in the data, and the vertex stage doubles it,
 * so the drawn half-width is `2 · width`. The module header derives this
 * from `glsl-lib.ts` :286-290 and `shader-glsl.ts` :100 / :505-511 and
 * lists two independent in-repo corroborations. Pass the value straight
 * out of the width attribute; do not pre-double it.
 */
export function lineSigmaFromAuthoredWidth(authoredWidth: number): number {
  return (LINE_RENDERED_HALF_WIDTH_FACTOR * authoredWidth) / GAUSSIAN_EQUIVALENT_TRUNCATION;
}

/** Ray/segment geometry, in the parameterization the integral consumes. */
export interface RaySegmentGeometry {
  /** Segment length `L` (world units). */
  length: number;
  /**
   * `|u|` — magnitude of the ray direction's component perpendicular to
   * the segment axis, i.e. `|V × d̂|` = sin of the ray/axis angle. 0 when
   * end-on, 1 when broadside.
   */
  absU: number;
  /**
   * `D` — closest-approach distance from the ray **line** to the
   * segment's **infinite axis**. Discontinuous at `|u| = 0`; see the
   * module header's hazard note.
   */
  distanceToAxis: number;
  /**
   * `s*` — axial coordinate (measured from the segment start) of that
   * closest-approach foot. Diverges like `1/|u|` as the ray goes end-on;
   * the products `s*·|u|` that the integral actually uses stay bounded.
   */
  sStar: number;
  /**
   * Distance from the ray **line** to the **finite segment** — the input
   * to the peak-mode capsule profile. Continuous everywhere, including
   * through end-on.
   */
  distanceToSegment: number;
}

/** Vector helpers, local so the module stays dependency-free. */
function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] as const;
}

function scale3(a: Vec3, k: number): Vec3 {
  return [a[0] * k, a[1] * k, a[2] * k] as const;
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]] as const;
}

function len3(a: Vec3): number {
  return Math.sqrt(dot3(a, a));
}

/**
 * Distance from the ray line to the **finite** segment, derived from the
 * same `(L, D, s*, |u|)` triple the integral uses.
 *
 * For a fixed axial coordinate `s` the point `P0 + s·d̂` sits at
 * `√(D² + |u|²(s − s*)²)` from the ray line — the same quadratic the
 * integral's erf window comes from. Minimising it over `s ∈ [0, L]` is
 * therefore just clamping `s*` into the segment. That shared derivation
 * is why the sum-mode integral and the peak-mode capsule cannot disagree
 * about where a segment is.
 *
 * The `(clamp(s*) − s*)·|u|` product is written as a product on purpose:
 * near end-on `s*` is huge and `|u|` is tiny, and multiplying them is
 * stable where subtracting the corresponding distances is not.
 */
export function raySegmentDistance(
  length: number,
  distanceToAxis: number,
  sStar: number,
  absU: number
): number {
  const overshoot = (Math.min(Math.max(sStar, 0), length) - sStar) * absU;
  return Math.sqrt(distanceToAxis * distanceToAxis + overshoot * overshoot);
}

/**
 * Derive `(L, |u|, D, s*, dist-to-segment)` for a ray and a segment, in
 * **float64, on the CPU**.
 *
 * Shared by the CPU references and the unit tests so there is exactly one
 * geometry derivation to get wrong. `rayDirection` need not be
 * normalized; the segment must have nonzero length.
 *
 * The formulation is chosen for stability near end-on:
 *
 * ```text
 *   m = V × d̂,  |u| = |m|,  m̂ = m/|u|,  û = d̂ × m̂
 *   D  = |w0 · m̂|                       (no cancellation)
 *   s* = (w0 · d̂) − (V·d̂)(w0 · û)/|u|   (the 1/|u| divergence is real)
 * ```
 *
 * with `w0 = rayOrigin − segmentStart`. The naive alternative,
 * `D² = |w_perp|² − (w_perp·û)²`, is a difference of two nearly equal
 * numbers exactly when the ray goes end-on, and loses the answer there.
 *
 * Below {@link RAY_SEGMENT_PARALLEL_EPS} the ray is treated as exactly
 * parallel: `D` becomes the constant perpendicular offset and `s*` the
 * (then irrelevant — it enters only through `s*·|u|`) foot of `w0`.
 *
 * This is **not** a shader-ready derivation, and porting it is out of
 * scope for this slice — see the module header's hazard section for the
 * float32 problems (`RAY_SEGMENT_PARALLEL_EPS` in particular) the wiring
 * slice has to solve.
 */
export function raySegmentGeometry(
  rayOrigin: Vec3,
  rayDirection: Vec3,
  segmentStart: Vec3,
  segmentEnd: Vec3
): RaySegmentGeometry {
  const seg = sub3(segmentEnd, segmentStart);
  const length = len3(seg);
  if (!(length > 0)) {
    throw new Error('raySegmentGeometry: segment must have nonzero length');
  }
  const dirLen = len3(rayDirection);
  if (!(dirLen > 0)) {
    throw new Error('raySegmentGeometry: ray direction must be nonzero');
  }
  const d = scale3(seg, 1 / length);
  const v = scale3(rayDirection, 1 / dirLen);
  const w0 = sub3(rayOrigin, segmentStart);

  const m = cross3(v, d);
  const absU = Math.min(len3(m), 1);

  let distanceToAxis: number;
  let sStar: number;
  if (absU < RAY_SEGMENT_PARALLEL_EPS) {
    const wd = dot3(w0, d);
    const wPerp = sub3(w0, scale3(d, wd));
    distanceToAxis = len3(wPerp);
    sStar = wd;
  } else {
    const mHat = scale3(m, 1 / absU);
    const uHat = cross3(d, mHat);
    distanceToAxis = Math.abs(dot3(w0, mHat));
    sStar = dot3(w0, d) - (dot3(v, d) * dot3(w0, uHat)) / absU;
  }

  return {
    length,
    absU,
    distanceToAxis,
    sStar,
    distanceToSegment: raySegmentDistance(length, distanceToAxis, sStar, absU),
  };
}

/** Inputs to the sum-mode ray integral. */
export interface LineRayIntegralParams {
  /** Gaussian width — see {@link lineSigmaFromAuthoredWidth}. Must be > 0. */
  sigma: number;
  /** Segment length `L`. */
  length: number;
  /** `D` — ray-line to segment-axis closest approach. */
  distanceToAxis: number;
  /** `s*` — axial coordinate of that closest approach. */
  sStar: number;
  /** `|u|` — perpendicular fraction of the ray direction, in [0, 1]. */
  absU: number;
}

/** GLSL-mirroring `mix(a, b, t)`; exact for `t` of 0 or 1. */
function mixScalar(a: number, b: number, t: number): number {
  return a * (1 - t) + b * t;
}

/** GLSL-mirroring `step(edge, x)`. */
function stepScalar(edge: number, x: number): number {
  return x < edge ? 0 : 1;
}

/**
 * The shared body of both CPU entry points, written statement-for-
 * statement against `GLSL_LINE_RAY_INTEGRAL_FUNCTIONS` so the transpiled
 * GLSL and this mirror are bit-identical (pinned by the unit test).
 */
function lineRayIntegral(p: LineRayIntegralParams, erf: (x: number) => number): number {
  const k = LINE_INV_SQRT2 / p.sigma;
  const axialScale = p.length * k;
  const gap = axialScale * p.absU;
  const halfGap = 0.5 * gap;
  const m = (p.sStar - 0.5 * p.length) * p.absU * k;
  const dr = p.distanceToAxis * k;
  const radial = Math.exp(-dr * dr);

  // Closed-form lane. The divisor is guarded because both lanes are
  // evaluated before the mix (see the module header).
  const window = Math.max(0, 0.5 * (erf(m + halfGap) - erf(m - halfGap)));
  const closed = window / Math.max(gap, LINE_WINDOW_GAP_THRESHOLD);

  // Derivative lane: half the central expansion of the difference
  // quotient about M (`closed` and `taylor` both represent ½·DQ). Its
  // two inputs are clamped to the regime where this arm is SELECTED —
  // without that the UNSELECTED arm overflows float32 and `Inf * 0`
  // poisons the mix. Both clamps are no-ops where the arm is selected.
  const taylorHalfGap = 0.5 * Math.min(gap, LINE_WINDOW_GAP_THRESHOLD);
  const q = taylorHalfGap * taylorHalfGap;
  const mm = Math.min(m * m, LINE_TAYLOR_MM_CLAMP);
  const series =
    1 +
    q * (2 * mm - 1) * LINE_WINDOW_TAYLOR_C2 +
    q * q * (4 * mm * mm - 12 * mm + 3) * LINE_WINDOW_TAYLOR_C4;
  const taylor = 0.5 * LINE_TWO_OVER_SQRT_PI * Math.exp(-mm) * series;

  const halfDq = mixScalar(closed, taylor, stepScalar(gap, LINE_WINDOW_GAP_THRESHOLD));
  return radial * axialScale * halfDq;
}

/**
 * CPU **reference** for the sum-mode ray integral, normalized by
 * `σ√(2π)` (so side-on mid-segment at `D = 0` is exactly 1.0 — see the
 * module header). Uses `erfRef`, the accurate A&S erf.
 *
 * Accuracy, measured against an independent high-precision oracle over
 * 400k random samples (`erfc` continued fraction, ~1e-15): worst
 * **5.0e-4 relative** among samples whose normalized value exceeds 1e-3,
 * **1.1e-4** above 0.1. The worst cases are in the CLOSED lane and the
 * cause is `erfRef`'s 1.5e-7 *absolute* error divided by an
 * exponentially small window, not Taylor truncation — the derivative
 * lane's own worst relative error is 6.7e-6. Deep in the saturated tail
 * (window ~1e-15) the relative error is unbounded; measured 1.7e-2 at a
 * window of 2.1e-15, where the absolute value is ~1e-16 and irrelevant.
 *
 * Excludes the per-element amplitude `a`; multiply it in at the call
 * site. Never returns a negative value.
 */
export function lineRayIntegralRef(params: LineRayIntegralParams): number {
  return lineRayIntegral(params, erfRef);
}

/**
 * TS mirror of the GPU evaluation: identical lanes and thresholds, but
 * with `erfPoly` in place of `erfRef`. Use it to predict what the shader
 * will produce (to within accumulated float32 rounding); use
 * {@link lineRayIntegralRef} when accuracy matters. Exactly the
 * `erfRef`/`erfPoly` split `erf.ts` established.
 *
 * NOTE this mirror is float64. It reproduces the shader's *lanes and
 * literals*, not its *precision*; the float32 overflow guards documented
 * in the module header are invisible here and are pinned by the test's
 * explicit `Math.fround`-per-operation harness instead.
 */
export function lineRayIntegralPoly(params: LineRayIntegralParams): number {
  return lineRayIntegral(params, erfPoly);
}

/**
 * CPU reference for the **peak (max-blend) mode** profile: the
 * Gaussian-shouldered capsule `exp(-dist² / (2σ²))` evaluated on the
 * ray-to-**finite-segment** distance from {@link raySegmentDistance}.
 *
 * Mid-segment and perpendicular this is the same radial Gaussian the sum
 * mode's side-on limit gives; past an endpoint the clamp turns it into a
 * spherical cap, which is what makes a max composition across a polyline
 * joint seamless.
 */
export function lineCapsuleProfileRef(sigma: number, distanceToSegment: number): number {
  const z = distanceToSegment * (LINE_INV_SQRT2 / sigma);
  return Math.exp(-z * z);
}

/**
 * GLSL implementations, generated from the constants above. Inject once
 * per shader, **after** `GLSL_ERF_FUNCTIONS` — `luxarLineRayIntegral`
 * calls `luxarErf` and this block deliberately does not redeclare it
 * (a shader may already inject erf for another reason).
 *
 * Cost per fragment for the integral: two `luxarErf` (pure ALU), two
 * `exp`, two divides (`1/sigma` and the guarded window quotient). The
 * `mix`/`step` pair is the lane select, written branch-free on purpose;
 * see the module header, including why the two `min` guards are load-
 * bearing in float32.
 */
export const GLSL_LINE_RAY_INTEGRAL_FUNCTIONS = `
float luxarRaySegmentDistance(float segLength, float distanceToAxis, float sStar, float absU) {
  float overshoot = (min(max(sStar, 0.0), segLength) - sStar) * absU;
  return sqrt(distanceToAxis * distanceToAxis + overshoot * overshoot);
}

float luxarLineCapsuleProfile(float sigma, float distanceToSegment) {
  float z = distanceToSegment * (${glslNum(LINE_INV_SQRT2)} / sigma);
  return exp(-z * z);
}

float luxarLineRayIntegral(float sigma, float segLength, float distanceToAxis, float sStar, float absU) {
  float k = ${glslNum(LINE_INV_SQRT2)} / sigma;
  float axialScale = segLength * k;
  float gap = axialScale * absU;
  float halfGap = 0.5 * gap;
  float m = (sStar - 0.5 * segLength) * absU * k;
  float dr = distanceToAxis * k;
  float radial = exp(-dr * dr);
  float window = max(0.0, 0.5 * (luxarErf(m + halfGap) - luxarErf(m - halfGap)));
  float closed = window / max(gap, ${glslNum(LINE_WINDOW_GAP_THRESHOLD)});
  float taylorHalfGap = 0.5 * min(gap, ${glslNum(LINE_WINDOW_GAP_THRESHOLD)});
  float q = taylorHalfGap * taylorHalfGap;
  float mm = min(m * m, ${glslNum(LINE_TAYLOR_MM_CLAMP)});
  float series = 1.0 + q * (2.0 * mm - 1.0) * ${glslNum(LINE_WINDOW_TAYLOR_C2)}
               + q * q * (4.0 * mm * mm - 12.0 * mm + 3.0) * ${glslNum(LINE_WINDOW_TAYLOR_C4)};
  float taylor = 0.5 * ${glslNum(LINE_TWO_OVER_SQRT_PI)} * exp(-mm) * series;
  float halfDq = mix(closed, taylor, step(gap, ${glslNum(LINE_WINDOW_GAP_THRESHOLD)}));
  return radial * axialScale * halfDq;
}
`;

/** TSL node inputs mirroring {@link LineRayIntegralParams}. */
export interface LineRayIntegralNodes {
  sigma: TSLNode;
  length: TSLNode;
  distanceToAxis: TSLNode;
  sStar: TSLNode;
  absU: TSLNode;
}

/**
 * TSL twin of `luxarRaySegmentDistance`, built from the same expression
 * tree as the GLSL body.
 */
export function raySegmentDistanceTSL(
  length: TSLNode,
  distanceToAxis: TSLNode,
  sStar: TSLNode,
  absU: TSLNode
): TSLNode {
  const overshoot = min(max(sStar, float(0)), length)
    .sub(sStar)
    .mul(absU);
  return sqrt(distanceToAxis.mul(distanceToAxis).add(overshoot.mul(overshoot)));
}

/** TSL twin of `luxarLineCapsuleProfile`. */
export function lineCapsuleProfileTSL(sigma: TSLNode, distanceToSegment: TSLNode): TSLNode {
  const z = distanceToSegment.mul(float(LINE_INV_SQRT2).div(sigma));
  return exp(z.mul(z).negate());
}

/**
 * TSL twin of `luxarLineRayIntegral`, built from the SAME constants (the
 * code generator owns literal formatting — see the module header). The
 * lane select is a `mix`/`step` rather than a `select` for the reason
 * `erf.ts` gives: a `select` lowers to an `if`/`else` that duplicates
 * both arms in the generated code, and here both arms are large.
 */
export function lineRayIntegralTSL(p: LineRayIntegralNodes): TSLNode {
  const k = float(LINE_INV_SQRT2).div(p.sigma);
  const axialScale = p.length.mul(k);
  const gap = axialScale.mul(p.absU);
  const halfGap = gap.mul(0.5);
  const m = p.sStar.sub(p.length.mul(0.5)).mul(p.absU).mul(k);
  const dr = p.distanceToAxis.mul(k);
  const radial = exp(dr.mul(dr).negate());

  const window = max(
    float(0),
    erfPolyTSL(m.add(halfGap))
      .sub(erfPolyTSL(m.sub(halfGap)))
      .mul(0.5)
  );
  const closed = window.div(max(gap, float(LINE_WINDOW_GAP_THRESHOLD)));

  const taylorHalfGap = min(gap, float(LINE_WINDOW_GAP_THRESHOLD)).mul(0.5);
  const q = taylorHalfGap.mul(taylorHalfGap);
  const mm = min(m.mul(m), float(LINE_TAYLOR_MM_CLAMP));
  const series = float(1)
    .add(q.mul(mm.mul(2).sub(1)).mul(float(LINE_WINDOW_TAYLOR_C2)))
    .add(
      q
        .mul(q)
        .mul(mm.mul(mm).mul(4).sub(mm.mul(12)).add(3))
        .mul(float(LINE_WINDOW_TAYLOR_C4))
    );
  const taylor = float(0.5).mul(float(LINE_TWO_OVER_SQRT_PI)).mul(exp(mm.negate())).mul(series);

  const halfDq = mix(closed, taylor, step(gap, float(LINE_WINDOW_GAP_THRESHOLD)));
  return radial.mul(axialScale).mul(halfDq);
}
