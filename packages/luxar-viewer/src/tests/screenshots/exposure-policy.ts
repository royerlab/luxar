/**
 * Gallery auto-exposure policy — the pure DECISION half of the capture
 * harness's exposure calibration.
 *
 * `generate-gallery.spec.ts` owns the IO (drive the viewer's exposure, take a
 * screenshot, measure it); this module owns the *policy* (given luminance
 * stats, which exposure to try next). Splitting them lets the policy be
 * unit-tested against a simulated capture with no browser, dataset or GPU —
 * see `src/tests/unit/gallery-exposure-policy.test.ts`.
 *
 * The tuning constants live here too, so the harness's measurement code and
 * this policy share one definition of each threshold. `scripts/gallery/
 * score_exposure.py` (the offline scorer) keeps HAND-SYNCED Python copies of
 * several of them (including the module-private MIN_LIT_FRACTION) — nothing
 * asserts the two stay in agreement, so change both.
 */

/**
 * Auto-exposure targets (on the 0..1 tone-mapped canvas). We drive the HIGH
 * PERCENTILE (not the mean): highlights should sit just below clipping so the
 * subject reads bright without blowing out. Mean-chasing runs away on additive
 * / bloom-heavy scenes (the faint haze keeps the mean low while the core
 * saturates), so a percentile target is far more robust across geometries.
 */
export const TARGET_HI = 0.9; // desired p99 luminance of the lit foreground
export const HI_PERCENTILE = 0.99;
export const LIT_THRESHOLD = 0.04; // pixels dimmer than this are "background"
export const EXPOSURE_MIN = -6;
export const EXPOSURE_MAX = 8;
export const AUTO_EXPOSURE_ITERS = 3;

/**
 * Flat-subject (mid-tone) pass. A SHADED subject lit by the view-anchored
 * headlight has N·L ≈ 1 on every visible facet, so its lit-luminance histogram
 * is intrinsically NARROW — `mesh_isosurface_cells3d` measured p10 0.840 /
 * p50 0.873 / p99 0.903 at the auto-chosen exposure. Driving p99 to 0.9 parks
 * that whole 0.063-wide distribution deep in the ACES highlight rolloff, which
 * desaturates toward white by design (authored saturations of 0.85/0.70
 * rendered as 0.28/0.14). The clip guard cannot see it: that is a TAIL test,
 * and a 0.063-wide distribution at 0.87 has no tail above CLIP_LUMA. So when
 * the spread is small, p99 is a meaningless anchor and we re-target the MEDIAN.
 *
 * THE THRESHOLD IS EMPIRICAL, and deliberately conservative:
 *   - The one measured flat case has spread 0.063 — a ≈2× margin below 0.12.
 *   - Re-exposing the committed gallery media to the post-phase-1 state as a
 *     proxy, the TIGHTEST emissive tile measured (`gsplats_4d_celegans_tracking`,
 *     a README pick with auto-exposure on) came out at spread ≈0.167, only
 *     ≈40% above 0.12 — and its histogram is unimodal, so the comfortable
 *     "an emissive cloud always has masses of dim halo pixels" story does NOT
 *     hold for every tile. Other tiles sat at ≈1.8× the threshold, not 10×.
 *   - The error is therefore made ASYMMETRIC on purpose. A false fire silently
 *     recalibrates a committed tile by ~1.5-2 stops; a miss just leaves today's
 *     behaviour, with the per-demo `exposure` override as the escape hatch. So
 *     the gate is tight rather than generous.
 *   - Acceptance check is a manifest-wide before/after sweep with
 *     `scripts/gallery/score_exposure.py`, not this comment.
 *
 * KNOWN OPPOSITE FAILURE MODE: with HI_PERCENTILE 0.99 and a p10 floor, the
 * gate reduces to "fewer than ~10% of the lit pixels are dim", so a
 * high-perimeter mesh whose antialiased silhouette covers more than ~10% of its
 * lit pixels will NOT trigger it and will stay over-exposed. It does trigger on
 * the measured case (p10 = 0.840 there, i.e. the silhouette is well under 10%),
 * but a lacier surface would need the per-demo override.
 */
export const NARROW_SPREAD_MAX = 0.12; // p99 − p10 below this ⇒ "flat" subject
/** Desired p50 luminance of a flat subject once the mid-tone pass converges. */
export const TARGET_MID = 0.5;

/**
 * Mid-tone iteration budget. A tone curve's shoulder is compressive, so a plain
 * `log2(target / current)` step UNDER-corrects there and the error contracts
 * only partially per iteration — the issue's own two data points bracket it
 * (+0.97 stops → p50 0.873, −1.50 stops → p50 0.466: 2.47 stops of exposure
 * moved log2 p50 by only ≈0.91, i.e. a display gain of ≈0.37 per stop). Hence
 * many more iterations than the p99 pass, with an early exit once the step is
 * negligible. Do NOT over-relax: the tone curve's log-log slope is ≤ 1 in the
 * shoulder so a plain step never overshoots there, while over-stepping would
 * oscillate in the linear region.
 *
 * 10, not 8 — but the headroom is thin and the budget is NOT guaranteed to be
 * enough. Swept over display bands on the simulated headlit subject under a
 * Reinhard `x/(1+x)` curve: the 0.835-0.905 band early-exits on the 8th
 * iteration and a narrower 0.90-0.93 band needs 9, so the margin over the
 * measured worst case is ONE iteration. Bands starting higher still consume
 * most or all of the budget, and the tightest (0.99-0.999) exhausts it and hits
 * the EXPOSURE_MIN clamp. Those runs still land at p50 0.505-0.586 — slightly
 * high, but well under
 * FLAT_MID_MIN, so an exhausted budget degrades gracefully instead of producing
 * a wrong exposure. The extra screenshots are only ever spent on a flat subject.
 */
export const MID_EXPOSURE_ITERS = 10;
/** Early-exit threshold for the mid-tone pass, in stops. */
export const MID_EXPOSURE_TOL = 0.05;

/**
 * Clipping guard: after the p99 pass, if more than this fraction of the lit
 * subject is blown to near-white (bright AND desaturated), step exposure down
 * until it isn't. This is what actually kills the additive/HDR white-out that
 * a percentile target alone misses (the subject can have a huge blown core yet
 * a below-target p99). `scripts/gallery/score_exposure.py` re-derives the same
 * three numbers offline (hand-synced, and off a lossless PNG rather than this
 * pass's JPEG-q70 screenshot, so the two can disagree at the margin).
 */
export const CLIP_LUMA = 0.95; // luma above this is "bright"
export const CLIP_SAT_MAX = 0.15; // saturation below this is "near-white"
export const CLIP_FRAC_MAX = 0.05; // > this blown fraction → step exposure down (user: <5%)
export const BG_LUMA_MAX = 0.1; // background (frame p10) must stay near-black; else too bright
export const CLIP_GUARD_ITERS = 8;
export const CLIP_GUARD_STEP = 0.5; // stops per guard step

/** Luminance statistics of one captured frame, as the policy consumes them. */
export interface LumaStats {
  /** HI_PERCENTILE (p99) luma of the lit foreground. */
  hiLuma: number;
  /** 10th-percentile luma of the lit foreground. */
  loLuma: number;
  /** Median luma of the lit foreground. */
  midLuma: number;
  /** Fraction of the whole frame that is lit (luma > LIT_THRESHOLD). */
  litFraction: number;
  /** Fraction of the LIT pixels blown to near-white (bright + desaturated). */
  clippedFrac: number;
  /** 10th-percentile luma of the WHOLE frame — the background level. */
  bgLuma: number;
}

/** The two side-effecting operations the policy needs from the harness. */
export interface ExposureIO {
  /** Apply an exposure (log2 stops) to the live viewer. */
  apply(stops: number): Promise<void>;
  /** Capture + measure the current frame. */
  measure(): Promise<LumaStats>;
}

/** Outcome of {@link computeAutoExposure}. */
export interface AutoExposureResult {
  /** The chosen exposure, in log2 stops. */
  stops: number;
  /**
   * Whether the flat-subject (mid-tone) pass fired. Logged by the harness so a
   * manifest-wide sweep can tell which tiles the new gate actually touched —
   * a false fire is the main risk of the narrow-spread heuristic.
   */
  flatSubject: boolean;
}

/** Nothing lit below this frame fraction — treat the measurement as empty. */
const MIN_LIT_FRACTION = 0.0005;

const clampStops = (s: number): number => Math.max(EXPOSURE_MIN, Math.min(EXPOSURE_MAX, s));

/**
 * Pick an exposure in three phases:
 *   1. Percentile pass — converge so the lit foreground's high percentile hits
 *      TARGET_HI (bright but not clipped). Exposure is ~log-linear in
 *      luminance, so a few log2 corrections converge.
 *   2. Flat-subject pass — if the lit histogram's spread (p99 − p10) is under
 *      NARROW_SPREAD_MAX the subject has essentially no internal dynamic range
 *      (a headlit shaded surface), so p99 is a meaningless anchor: re-target
 *      the MEDIAN to TARGET_MID instead. The spread is evaluated AFTER phase 1,
 *      i.e. at the exposure where p99 ≈ 0.9 for every subject, which is what
 *      makes one absolute threshold comparable across geometries. It is an
 *      EMPIRICAL gate, not a proof — see NARROW_SPREAD_MAX for the measured
 *      margins on both sides and the false-fire risk. `flatSubject` in the
 *      result records whether it fired.
 *   3. Guard — step exposure DOWN while EITHER >5% of the subject is blown to
 *      white OR the background is lifted to grey (frame p10 above near-black).
 *      Phase 1 (p99 target) over-boosts sparse/bloomy scenes into a grey wash;
 *      the background term is what pulls those back to a black background.
 *
 * NOTE (preserved quirk): when a pass's correction is already negligible it
 * records the corrected value and breaks WITHOUT applying it, so the returned
 * stops can differ from the applied exposure by up to the pass's tolerance, and
 * the next phase's first measurement is taken at the old applied exposure.
 * Deliberate — "fixing" it would shift every existing tile.
 */
export async function computeAutoExposure(io: ExposureIO): Promise<AutoExposureResult> {
  let stops = 1.0;
  await io.apply(stops);
  // Phase 1: percentile.
  for (let iter = 0; iter < AUTO_EXPOSURE_ITERS; iter++) {
    const { hiLuma, litFraction } = await io.measure();
    if (litFraction < MIN_LIT_FRACTION || hiLuma <= 0) {
      stops = Math.min(EXPOSURE_MAX, stops + 2); // almost nothing lit — brighten
      await io.apply(stops);
      continue;
    }
    const correction = Math.log2(TARGET_HI / hiLuma);
    const next = clampStops(stops + correction);
    if (Math.abs(next - stops) < 0.1) {
      stops = next;
      break;
    }
    stops = next;
    await io.apply(stops);
  }
  // Phase 2: flat subject — re-anchor on the median when the lit histogram has
  // no spread. The measurement that decides this is CARRIED into phase 3 when
  // the branch is not taken, so a wide-spread subject performs exactly the same
  // applies and measurements (in the same order) as before this phase existed.
  let carried: LumaStats | null = await io.measure();
  const spread = carried.hiLuma - carried.loLuma;
  const flatSubject =
    carried.litFraction >= MIN_LIT_FRACTION && carried.midLuma > 0 && spread < NARROW_SPREAD_MAX;
  if (flatSubject) {
    for (let iter = 0; iter < MID_EXPOSURE_ITERS; iter++) {
      const stats = carried ?? (await io.measure());
      carried = null;
      if (stats.litFraction < MIN_LIT_FRACTION || stats.midLuma <= 0) break;
      const correction = Math.log2(TARGET_MID / stats.midLuma);
      const next = clampStops(stops + correction);
      if (Math.abs(next - stops) < MID_EXPOSURE_TOL) {
        stops = next;
        break;
      }
      stops = next;
      await io.apply(stops);
    }
  }
  // Phase 3: guard — pull exposure DOWN while EITHER the subject is blowing out
  // (>5% white-blown) OR the background is lifted to grey (frame p10 not black).
  // The background term is what actually fixes the "too bright / grey bg" cases
  // that a white-blown check alone misses (bloom haze sits well below clip luma).
  for (let iter = 0; iter < CLIP_GUARD_ITERS; iter++) {
    const { clippedFrac, bgLuma } = carried ?? (await io.measure());
    carried = null;
    const tooBlown = clippedFrac > CLIP_FRAC_MAX;
    const bgTooBright = bgLuma > BG_LUMA_MAX;
    if ((!tooBlown && !bgTooBright) || stops <= EXPOSURE_MIN) break;
    stops = Math.max(EXPOSURE_MIN, stops - CLIP_GUARD_STEP);
    await io.apply(stops);
  }
  return { stops, flatSubject };
}
