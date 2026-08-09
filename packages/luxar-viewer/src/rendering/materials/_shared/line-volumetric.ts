/**
 * Volumetric line primitive — shared calibration constants and the CPU
 * reference implementation of its ray integral (issue #1352).
 *
 * The volumetric primitive draws a line segment as a true 3D density: the
 * segment convolved with an isotropic 3D Gaussian,
 *
 *   ρ(x) = a · exp(−r² / 2σ²) · W(s)
 *
 * with `r` the distance from the segment's infinite axis, `s` the axial
 * coordinate, and `W` an end-treatment window. Sum-family blending modes
 * (additive, luminous, volumetric) integrate ρ along the view ray in closed
 * form; peak-family modes (max, normal, opaque) take the max along the ray,
 * which is today's profile at the ray→segment distance.
 *
 * ## End treatments (the four sum-family lanes)
 *
 * Each segment end is either SOFT (a free polyline end, a hub, or a
 * slice-clip boundary): the erf cap of the exact convolution — or HARD (an
 * interior polyline joint): the infinite-rod density clipped by the bisector
 * plane between the segment and its partner. Reflection across the bisector
 * plane swaps the two rod axes, so the two densities agree pointwise ON the
 * plane and the pair sums to a seamless, single-covered miter at any bend
 * angle.
 *
 * - **soft/soft** — W(s) = ½[erf(s·c) − erf((s−L)·c)], c = 1/(σ√2): the
 *   exact segment ⊛ Gaussian. Ray integral: one erf difference (the
 *   Gaussian×erf identity below), with a midpoint-Taylor lane when the two
 *   erf arguments are close.
 * - **hard/hard** — infinite rod ∩ the two half-spaces. The ray integral is
 *   an erf difference over the ξ-interval between the ray's two plane
 *   crossings. Exact at any bend.
 * - **mixed** (one hard, one soft — the FIRST and LAST segment of every
 *   polyline) — ρ = rod · one-sided erf cap · half-space. The ray integral
 *   has no elementary closed form (it is an Owen-T-class integral), but
 *   TWO inclusion–exclusion splits exist with COMPLEMENTARY error domains,
 *   selected per ray by one sign (see the lane body):
 *
 *     J1 = ∫_halfspace G − ∫_FULL-LINE G·(1 − W_cap)   (plane-clip primary)
 *     J2 = ∫_FULL-LINE G·W_cap                          (cap primary)
 *
 *   Every term is erf-only (half-line: plain erf; full-line: the identity).
 *   J1 drops the cap-COMPLEMENT's mass on the ray's excluded side of the
 *   plane, J2 the CAP's — for axis-dominant rays exactly one is
 *   exponentially exact. A THIRD split covers the saturated-cap regime:
 *
 *     J0 = [plane bracket alone, cap ignored]
 *
 *   selected when the cap's pointwise ramp lies wholly outside the bracket
 *   on its saturated side — then the window is ≡1 over every unit of
 *   bracket mass and J0 is exact. This is what carries a biting near-plane
 *   clip (camera inside a chain-end segment): J1 would double-count the
 *   exclusion, J2 would keep the clipped mass; J2 is additionally min()ed
 *   with the bracket, both being upper bounds of the exact integral. The
 *   residual survives only where a plane sits within ~3σ of the cap's
 *   support (short chain-end segments; sharp bends, where the binding
 *   metric is PERPENDICULAR distance to the near-axial plane, which longer
 *   segments do not grow): a measured, bounded UNDERestimate (clamped at
 *   0), pinned in the unit tests.
 *
 * ## Near plane (review finding 2, PR #1426)
 *
 * Sum lanes integrate the ray closed-form, so a segment straddling the
 * perspective near plane would otherwise contribute light from BEHIND the
 * eye. The near plane is treated as ONE MORE PLANE CLIP where that stays
 * closed-form: the structural-parallel lane maps it to an s-bound, the
 * general plane lane to a ξ-bound (see {@link SumIntegralOptions.tMin}).
 * The soft/soft general lane keeps the documented full-line + near-fade
 * convention — its behind-eye mass is exponentially small outside the
 * few-degrees-of-axial cone the structural lane owns.
 *
 * ## The load-bearing identity
 *
 *   ∫ e^{−ξ²} erf(αξ + β) dξ  =  √π · erf(β / √(1+α²))     (full line)
 *
 * Every closed form above is this identity (the soft/soft lane applies it
 * twice). The √(1+α²) contraction is what the shader's `kk` constant folds.
 *
 * ## Calibration
 *
 * σ = drawnHalfWidth / T with T = {@link GAUSSIAN_EQUIVALENT_TRUNCATION}:
 * today's screen-space profile exp(−K·p²) IS that truncated Gaussian, so the
 * volumetric primitive matches the quad side-on by construction. The drawn
 * half-width is 2 width-texel units, hence
 * {@link LINE_SIGMA_PER_WIDTH} = 2/T. Ray integrals are normalized by
 * σ·√(2π), which pins the side-on peak of a long segment to exactly the
 * quad's core intensity.
 *
 * This module is the SINGLE SOURCE for those constants (GLSL interpolates
 * them, TSL passes them to `float()`) and the CPU reference the unit tests
 * hold both shader backends' lane math against — the same pattern as
 * `erf.ts` / `falloff.ts` / `volumetric.ts`.
 *
 * @module rendering/materials/_shared/line-volumetric
 */

import { erfRef } from './erf';
import { GAUSSIAN_EQUIVALENT_TRUNCATION } from './falloff';

/**
 * World-space Gaussian σ per width-texel unit: drawn half-width (2 width
 * units) divided by the Gaussian-equivalent truncation T ≈ 3.035. Value
 * ≈ 0.659.
 */
export const LINE_SIGMA_PER_WIDTH = 2 / GAUSSIAN_EQUIVALENT_TRUNCATION;

/**
 * Screen-variance dilation floor in px², added to the projected σ² so a
 * sub-pixel segment still covers ~a pixel (with matching energy
 * compensation, so total flux is conserved). Same calibration as the gsplat
 * `cov2DDilation` default; T²·dilation is the matching STENCIL radius floor
 * the vertex stage folds in.
 */
export const LINE_STENCIL_DILATION = 0.3;

/**
 * Relative A/n² threshold below which the sum-family fragment lane treats
 * the ray as STRUCTURALLY parallel to the segment axis and switches to the
 * exact axial closed forms. This is not merely a precision guard: the
 * general mixed-end lane is a difference of two erf brackets whose values
 * both approach saturation as A → 0, so its float32 cancellation error grows
 * like 1/√A while the true answer stays finite. At A/n² = 1e-5 the axial
 * formula's own error (it ignores the ray's perpendicular drift across the
 * window) is O(A/n²) — far below float32 noise — while the subtraction still
 * has ~2.5 significant digits of headroom. Verified in the unit tests with
 * float32-emulated lane evaluation.
 */
export const LINE_PARALLEL_LANE_THRESHOLD = 1e-5;

const INV_SQRT_PI = 1 / Math.sqrt(Math.PI);
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

/** Minimal vec3 for the reference implementation. */
export type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function scale(a: Vec3, k: number): Vec3 {
  return [a[0] * k, a[1] * k, a[2] * k];
}
function norm(a: Vec3): number {
  return Math.sqrt(dot(a, a));
}

/**
 * A volumetric line segment in camera space. `cutA`/`cutB` are UNIT bisector
 * plane normals through the respective endpoint with the segment's own
 * material on the NEGATIVE side; `null`/`undefined` marks a soft (erf-cap)
 * end.
 */
export interface VolumetricSegment {
  readonly a: Vec3;
  readonly b: Vec3;
  /** Effective Gaussian σ (variance floor already folded in, if any). */
  readonly sigma: number;
  readonly cutA?: Vec3 | null;
  readonly cutB?: Vec3 | null;
}

/**
 * The MODEL density ρ(x) (unit amplitude, no display floor) — what the
 * closed-form lanes must integrate. This is the quadrature oracle's
 * integrand, deliberately written from the definition rather than the lane
 * math.
 */
export function lineVolumetricModelDensity(p: Vec3, seg: VolumetricSegment): number {
  const w = sub(seg.b, seg.a);
  const L = norm(w);
  const axis = L > 0 ? scale(w, 1 / L) : ([1, 0, 0] as Vec3);
  const rel = sub(p, seg.a);
  const s = dot(rel, axis);
  const perp = sub(rel, scale(axis, s));
  const r2 = dot(perp, perp);
  const c = 1 / (seg.sigma * Math.SQRT2);

  let window: number;
  const hardA = seg.cutA != null;
  const hardB = seg.cutB != null;
  if (hardA && hardB) {
    window = 1;
  } else if (hardA) {
    window = 0.5 * (1 - erfRef((s - L) * c));
  } else if (hardB) {
    window = 0.5 * (1 + erfRef(s * c));
  } else {
    window = 0.5 * (erfRef(s * c) - erfRef((s - L) * c));
  }
  if (hardA && dot(sub(p, seg.a), seg.cutA as Vec3) > 0) return 0;
  if (hardB && dot(sub(p, seg.b), seg.cutB as Vec3) > 0) return 0;

  return Math.exp(-r2 / (2 * seg.sigma * seg.sigma)) * window;
}

/** Options for the reference lane evaluation. */
export interface SumIntegralOptions {
  /** erf implementation (default `erfRef`; pass `erfPoly` to measure the shader polynomial's error). */
  readonly erf?: (x: number) => number;
  /** Round every intermediate through float32 (`Math.fround`) to emulate the GPU. */
  readonly f32?: boolean;
  /**
   * Ray-domain lower bound, in RAW-ray parameter units (the same `t` that
   * multiplies `dRaw`): material only at t ≥ tMin. This is the perspective
   * near plane — with the shader's ray convention (rayO.z = 0, dRaw.z = −1)
   * the crossing sits at exactly t = nearCull — treated as ONE MORE PLANE
   * CLIP in the lanes where it stays closed-form: the structural-parallel
   * lane (mapped to an s-bound) and the general plane lane (a ξ-bound).
   * The soft/soft general lane deliberately IGNORES it and keeps the
   * documented full-line + near-fade convention: its behind-eye mass is
   * exponentially small except within a few degrees of axial, where the
   * structural lane takes over. Undefined = whole line (the ortho path).
   */
  readonly tMin?: number;
}

/**
 * Ψ(x) = ∫ₓ^∞ ½(1 − erf(u)) du = ½[e^{−x²}/√π − x·(1 − erf(x))] — the
 * axial cap remainder used by the structurally-parallel mixed lane.
 *
 * Saturation guards are LOAD-BEARING, not a nicety: with the shader's
 * polynomial erf, `1 − erfPoly(x)` saturates to a residual of ~1e-7 rather
 * than decaying, so a large `x` (an unbounded plane crossing) turns
 * `x·(1−erf(x))` into garbage of magnitude x·1e-7. Beyond |x| = 6 the true
 * Ψ is 0 (x → +∞) or −x + 0 (x → −∞) to under 1e-17.
 */
export function erfCapRemainder(x: number, erf: (v: number) => number = erfRef): number {
  if (x > 6) return 0;
  if (x < -6) return -x;
  return 0.5 * (Math.exp(-x * x) * INV_SQRT_PI - x * (1 - erf(x)));
}

/**
 * CPU reference of the sum-family fragment lanes: the normalized ray
 * integral
 *
 *   I = (1 / σ√2π) · ∫ ρ(rayO + u·d̂) du        (u = arc length, full line)
 *
 * evaluated with the SAME lane structure the shaders use (structural
 * parallel / soft-soft / hard-hard / mixed, including the Taylor branches
 * and the mixed-lane inclusion–exclusion), so the unit tests validate the
 * lane math itself against numerical quadrature of
 * {@link lineVolumetricModelDensity}. No display floor and no amplitude —
 * pure geometry.
 *
 * Returns 0 for rays whose integral the shader would `discard` (dead
 * half-space, empty ξ-interval).
 */
export function lineVolumetricSumIntegral(
  rayO: Vec3,
  dRaw: Vec3,
  seg: VolumetricSegment,
  opts: SumIntegralOptions = {}
): number {
  const erf = opts.erf ?? erfRef;
  const fr = opts.f32 ? Math.fround : (x: number) => x;
  const tMin = opts.tMin;

  const segVec = sub(seg.b, seg.a);
  const L = fr(norm(segVec));
  const w: Vec3 = L > 0 ? scale(segVec, 1 / L) : [1, 0, 0];
  const M = add(seg.a, scale(w, 0.5 * L));
  const b = sub(rayO, M);
  const n2 = fr(dot(dRaw, dRaw));
  const rn = fr(Math.sqrt(n2));
  const dw = fr(dot(dRaw, w));
  const bdr = fr(dot(b, dRaw));
  const bw = fr(dot(b, w));
  const A = fr(n2 - dw * dw);
  const invSE = fr(1 / seg.sigma);

  const hardA = seg.cutA != null;
  const hardB = seg.cutB != null;
  const parallel = A < LINE_PARALLEL_LANE_THRESHOLD * n2;

  // Closest approach (midpoint-relative axial coord sM, ray parameter
  // tCenter) and the radial factor.
  let sM: number;
  let D2: number;
  let tCenter: number;
  if (parallel) {
    const invN2 = 1 / n2;
    const r0 = sub(b, scale(dRaw, bdr * invN2));
    const rw = dot(r0, w);
    D2 = fr(Math.max(dot(r0, r0) - rw * rw, 0));
    sM = 0;
    tCenter = fr(-bdr * invN2);
  } else {
    const invA = 1 / A;
    sM = fr((bw * n2 - dw * bdr) * invA);
    tCenter = fr((dw * bw - bdr) * invA);
    const pv = sub(add(b, scale(dRaw, tCenter)), scale(w, sM));
    D2 = fr(dot(pv, pv));
  }
  const r2n = fr(D2 * invSE * invSE);
  const radial = fr(Math.exp(-0.5 * r2n));

  // ξ mapping for plane crossings: ξ = (t − tCenter) · kxi. `sRay(t)` is the
  // ENDPOINT-coords axial position of the ray point at parameter t.
  const kxi = fr(Math.sqrt(A) * invSE * Math.SQRT1_2);
  const sAtCenter = fr(sM + 0.5 * L); // endpoint coords of the closest approach

  // Fold one bisector plane into the ξ-interval. Returns null when the ray
  // is (near-)parallel to the plane: `dead` = true kills the fragment,
  // false leaves the interval untouched.
  interface PlaneClip {
    xi: number;
    tightensHi: boolean;
  }
  const clipPlane = (normal: Vec3, through: Vec3): PlaneClip | { dead: boolean } => {
    const dn = fr(dot(dRaw, normal));
    const sn = fr(dot(sub(through, rayO), normal));
    if (Math.abs(dn) <= 1e-7 * rn) {
      return { dead: sn < 0 };
    }
    const tX = sn / dn;
    const xi = fr((tX - tCenter) * kxi);
    return { xi, tightensHi: dn > 0 };
  };

  if (parallel) {
    // STRUCTURAL PARALLEL: the ray runs along the axis, so the radial factor
    // is constant and the integral reduces to the axial window's length,
    // measured in s. s(t) = sAtCenter + (t − tCenter)·dw, so plane crossings
    // map to s-bounds directly; a soft cap contributes the Ψ remainder.
    const c = fr(invSE * Math.SQRT1_2);
    let sLo = Number.NEGATIVE_INFINITY;
    let sHi = Number.POSITIVE_INFINITY;
    const applyParallel = (normal: Vec3, through: Vec3): boolean => {
      const dn = dot(dRaw, normal);
      const sn = dot(sub(through, rayO), normal);
      if (Math.abs(dn) <= 1e-7 * rn) return sn >= 0;
      // s at the plane crossing: s(t) = sAtCenter + (t − tCenter)·dw with
      // the crossing at t = sn/dn. dn > 0 means the (negative-side) material
      // lies BEFORE the crossing in t; convert to an s-bound via sign(dw).
      const sX = sAtCenter + (sn / dn - tCenter) * dw;
      const materialBeforeInT = dn > 0;
      const sIncreasesWithT = dw > 0;
      if (materialBeforeInT === sIncreasesWithT) {
        sHi = Math.min(sHi, sX);
      } else {
        sLo = Math.max(sLo, sX);
      }
      return true;
    };
    if (hardA && !applyParallel(seg.cutA as Vec3, seg.a)) return 0;
    if (hardB && !applyParallel(seg.cutB as Vec3, seg.b)) return 0;

    // Near-plane ray-domain bound: one more plane clip, mapped to an
    // s-bound via s(t) = sAtCenter + (t − tCenter)·dw. Material at
    // t > tMin, and |dw| ≈ rn > 0 structurally in this lane.
    if (tMin !== undefined) {
      const sX = fr(sAtCenter + (tMin - tCenter) * dw);
      if (dw > 0) sLo = Math.max(sLo, sX);
      else sHi = Math.min(sHi, sX);
    }

    // Soft caps as axial windows; hard bounds as sharp limits. Effective
    // windowed length G, then I = radial · G · invSE / √(2π).
    let G: number;
    if (!hardA && !hardB) {
      // Only the near bound can clip here. H(x) = ∫ₓ^∞ W ds =
      // (Ψ((x−L)c) − Ψ(x·c))/c; bounds clamped into the window's support
      // keep the Ψ difference well-conditioned (H(−7/c) = L, H(L+7/c) = 0
      // exactly through the saturation guards, no 1e30-scale cancellation).
      const lo = Math.max(sLo, -7 / c);
      const hi = Math.min(sHi, L + 7 / c);
      const H = (x: number) =>
        (erfCapRemainder((x - L) * c, erf) - erfCapRemainder(x * c, erf)) / c;
      G = hi > lo ? H(lo) - H(hi) : 0;
    } else if (hardA && hardB) {
      const lo = Math.max(sLo, -1e30);
      const hi = Math.min(sHi, 1e30);
      G = Math.max(hi - lo, 0);
    } else if (hardA) {
      // Hard bound(s) from the plane, soft cap at B: ∫ ½(1−erf((s−L)c)) ds
      // over [sLo, sHi] = (Ψ((sLo−L)c) − Ψ((sHi−L)c)) / c.
      const lo = Number.isFinite(sLo) ? sLo : -1e30;
      const hi = Number.isFinite(sHi) ? sHi : 1e30;
      G =
        hi > lo ? (erfCapRemainder((lo - L) * c, erf) - erfCapRemainder((hi - L) * c, erf)) / c : 0;
    } else {
      // Soft cap at A: window ½(1+erf(s·c)) = ½(1−erf(−s·c)); mirror s.
      const lo = Number.isFinite(sLo) ? sLo : -1e30;
      const hi = Number.isFinite(sHi) ? sHi : 1e30;
      G = hi > lo ? (erfCapRemainder(-hi * c, erf) - erfCapRemainder(-lo * c, erf)) / c : 0;
    }
    return radial * Math.max(G, 0) * invSE * INV_SQRT_2PI;
  }

  if (hardA || hardB) {
    // GENERAL PLANE LANE (hard/hard exact; mixed via inclusion–exclusion).
    let xiLo = -4;
    let xiHi = 4;
    const applyGeneral = (normal: Vec3, through: Vec3): boolean => {
      const r = clipPlane(normal, through);
      if ('dead' in r) return !r.dead;
      const xi = fr(Math.min(Math.max(r.xi, -4), 4));
      if (r.tightensHi) xiHi = Math.min(xiHi, xi);
      else xiLo = Math.max(xiLo, xi);
      return true;
    };
    if (hardA && !applyGeneral(seg.cutA as Vec3, seg.a)) return 0;
    if (hardB && !applyGeneral(seg.cutB as Vec3, seg.b)) return 0;
    // Near-plane ray-domain bound: ξ increases with t (kxi > 0), so
    // material at t > tMin tightens the LOWER edge of the bracket.
    if (tMin !== undefined) {
      xiLo = Math.max(xiLo, fr(Math.min(Math.max((tMin - tCenter) * kxi, -4), 4)));
    }
    if (xiLo >= xiHi) return 0;

    const pref = fr((0.5 * rn) / Math.sqrt(Math.max(A, 1e-12 * n2)));

    // erf(ξHi) − erf(ξLo), through the same Taylor lane as the shader when
    // the interval is narrow (hard/hard wedges).
    const dxi = fr(xiHi - xiLo);
    const xim = fr(0.5 * (xiHi + xiLo));
    let bracket: number;
    if (dxi < 0.5) {
      const xim2 = Math.min(xim * xim, 80);
      bracket =
        fr(2 * INV_SQRT_PI * Math.exp(-xim2) * (1 + (dxi * dxi * (4 * xim2 - 2)) / 24)) * dxi;
    } else {
      bracket = fr(erf(xiHi) - erf(xiLo));
    }

    // Mixed: two inclusion–exclusion splits are available and their error
    // domains are COMPLEMENTARY, selected by one sign.
    //
    //   J1 = [plane-clipped rod] − [full-line cap-complement]   (the identity)
    //   J2 = [cap-only, plane ignored]                          (the identity)
    //
    // J1's dropped term is the cap-COMPLEMENT's mass on the ray's excluded
    // side of the plane; J2's is the CAP's mass there. When the ray is
    // axis-dominant (|α| = |dw|/√A > 1) the excluded side points along ±s,
    // and whichever split has its small window on that side is exponentially
    // exact: excluded side toward the complement ⇒ J2, else J1. When the
    // ray is perpendicular-dominant, s barely varies over the excluded side
    // and J1's subtraction is the right shape. The only regime with real
    // error left is plane-within-~3σ-of-the-cap (short chain-end segments,
    // extreme bends) — measured and pinned in the unit tests.
    if (hardA !== hardB) {
      // PRECISION NOTE: the mixed lane's erf terms are multiplied by
      // pref ∝ 1/sin(ray, axis), so near-axial rays amplify erf error
      // unboundedly. The shader therefore uses the A&S exponential erf in
      // THIS lane (its fragments are only the two chain-end segments of
      // each polyline — the cost is irrelevant), keeping the cheap
      // polynomial for the hot soft/soft and hard/hard lanes. The reference
      // mirrors that split: `opts.erf` applies to the other lanes, A&S here.
      const erfMixed = (x: number) => fr(erfRef(fr(x)));
      const kk = fr(kxi / rn);
      const hardNormal = (hardA ? seg.cutA : seg.cutB) as Vec3;
      const dn = fr(dot(dRaw, hardNormal));
      const axialDominant = dw * dw > A;
      // Excluded t-side is t > tX when dn > 0 puts material BEFORE the
      // crossing... material (negative side) sits at t < tX for dn > 0, so
      // the excluded side is t > tX; it maps toward s → −∞ exactly when
      // dn·dw < 0. The complement of a cap at A lives at s → −∞; of a cap
      // at B, at s → +∞.
      const excludedTowardMinusS = dn * dw < 0;
      const complementOnExcludedSide = hardA
        ? !excludedTowardMinusS // soft cap at B ⇒ complement at +∞
        : excludedTowardMinusS; // soft cap at A ⇒ complement at −∞
      // Re-evaluate the plane bracket with the precise erf too (the poly
      // bracket is fine for hard/hard where the Taylor lane covers narrow
      // intervals, but here pref amplification demands the exact form).
      const preciseBracket = dxi < 0.5 ? bracket : fr(erfMixed(xiHi) - erfMixed(xiLo));
      // SATURATED-CAP SHORTCUTS (the J0 split): when the cap's pointwise
      // ramp (s-width 3σ√2, mapped to ξ through ds/dξ = dw/kxi) lies
      // entirely outside the bracket on its SATURATED side, the window is
      // ≡1 across every unit of bracket mass and the pure plane bracket is
      // the exact form; entirely outside on the dead side, the integral is
      // 0. This is what carries a near-plane clip whose crossing has moved
      // past the cap (camera inside a chain-end segment): J1 would
      // subtract the full-line cap complement even though the bracket
      // already excludes it — double-counting the exclusion — and J2 would
      // keep the behind-camera cap mass the clip removed.
      if (axialDominant) {
        const capAtA = hardB; // soft cap at the A end (hard cut at B)
        const sEdge = capAtA ? 0 : L;
        const xiEdge = fr(((sEdge - sAtCenter) * kxi) / dw);
        const halfRamp = fr((3 * Math.sqrt(A)) / Math.abs(dw));
        const satHigh = capAtA === dw > 0; // cap saturates toward high ξ
        if (satHigh ? xiEdge + halfRamp <= xiLo : xiEdge - halfRamp >= xiHi) {
          return radial * pref * Math.max(preciseBracket, 0);
        }
        if (satHigh ? xiEdge - halfRamp >= xiHi : xiEdge + halfRamp <= xiLo) {
          return 0;
        }
      }
      if (axialDominant && complementOnExcludedSide) {
        // Cap-only split: the bisector's clip removes only cap-tail mass.
        // Both capOnly (full-line cap integral) and the bracket (interval
        // integral, cap ignored) are UPPER BOUNDS of the exact clipped
        // integral, so min() is at least as good as either — and it is
        // what carries a biting near-plane clip: once the near crossing
        // moves past the cap's ramp the bracket becomes the exact form,
        // while capOnly (which ignores plane clips) would keep the full
        // behind-camera cap mass.
        const capOnly = hardA
          ? fr(1 - erfMixed((sAtCenter - L) * kk))
          : fr(1 + erfMixed(sAtCenter * kk));
        return radial * pref * Math.max(Math.min(capOnly, preciseBracket), 0);
      }
      const capTerm = hardA
        ? fr(1 + erfMixed((sAtCenter - L) * kk)) // complement of the B cap
        : fr(1 - erfMixed(sAtCenter * kk)); // complement of the A cap
      return radial * pref * Math.max(preciseBracket - capTerm, 0);
    }
    const F = pref * Math.max(bracket, 0);
    return radial * F;
  }

  // SOFT/SOFT: the erf-difference closed form with the midpoint-Taylor lane.
  const kk = fr(kxi / rn);
  const xm = fr(sM * kk);
  const dx = fr(L * kk);
  const x0 = fr(xm - 0.5 * dx);
  const x1 = fr(xm + 0.5 * dx);
  if (x0 > 3 || x1 < -3) return 0;
  let E: number;
  if (dx < 0.5) {
    const xm2 = Math.min(xm * xm, 80);
    E = fr(2 * INV_SQRT_PI * Math.exp(-xm2) * (1 + (dx * dx * (4 * xm2 - 2)) / 24));
  } else {
    E = fr((erf(x1) - erf(x0)) / dx);
  }
  const F = fr(Math.max(E, 0) * L * invSE * (1 / (2 * Math.SQRT2)));
  return radial * F;
}
