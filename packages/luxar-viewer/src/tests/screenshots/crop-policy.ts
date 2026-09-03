/**
 * Gallery framing policy — the pure DECISION half of the capture harness's
 * under-fill and border-lit (cropped-subject) checks.
 *
 * A **border-lit pixel** is a lit pixel (luma above `LIT_THRESHOLD`) on the
 * outermost row or column of the captured frame — row 0, row H−1, column 0 or
 * column W−1. Lit content on the frame edge means the subject is running off the
 * frame: what we see is a crop, not a fit.
 *
 * WHY A SEPARATE CHECK. The framing loop (`fillToScreen` → `measureCoverage`)
 * closes on an OUTLIER-ROBUST percentile bounding box of the lit pixels (3rd–97th
 * percentile), which is right for DRIVING the dolly — a stray speck must not
 * dominate the framing (the Gaia/asteroid failure mode). But what touches the
 * frame edge IS the outliers: a detached fragment, or the far corners of an
 * elongated subject. So the robust bbox can sit comfortably inside the frame,
 * report a good fit, and be cropped anyway. The loop is blind to this BY
 * CONSTRUCTION; no percentile tuning does both jobs. As in `./exposure-policy`,
 * the decision is split from the IO so it is unit-testable
 * (`src/tests/unit/gallery-crop-policy.test.ts`).
 *
 * The measured ladder on `mesh_isosurface_cells3d` (border-lit pixels SUMMED
 * over all 120 orbit frames):
 *
 * | fillTarget | coverage reported | border-lit pixels     |
 * |------------|-------------------|-----------------------|
 * | 0.84       | "82%" — a fit     | 3389 (worst frame 218)|
 * | 0.75       |                   | 1256                  |
 * | 0.69       | 69%               | 0                     |
 *
 * At 0.84 the harness reported a good fit; the issue author describes watching a
 * nucleus run off the right edge in about a third of the frames (his observation,
 * not one of the figures above) — and nothing in the capture said so.
 *
 * WHY IT WARNS AND NEVER FAILS. A non-zero count looks like the norm rather than
 * the exception: of the 28 committed README gallery tiles, 25 are non-zero at
 * `LIT_THRESHOLD` (0.04). Their shape:
 *   - **A whole-perimeter faint wash** — 4 tiles light every perimeter pixel, at
 *     border median luma 0.043–0.090. An even wash is not a crop.
 *   - **Bright at the edge, and unhinted** — at least thirteen of the 25 have
 *     hundreds to thousands of border pixels above luma 0.5 and NO framing hint
 *     at all in the manifest, so nobody authored that framing: the default fill
 *     at 0.95 chose it (e.g. `rainbow_sphere`, 2335 border pixels, 1877 above
 *     0.5). They read as full-bleed compositions, but that is also verbatim the
 *     next class's signature — presumed intentional, UNVERIFIED, and possibly
 *     true positives no one has inspected.
 *   - **A genuine crop** — a bright, spatially localized run of edge pixels: the
 *     `mesh_isosurface_cells3d` case above.
 * Luma alone does NOT separate these classes and this module does not try to: the
 * localization that would discriminate a crop from a wash is measured nowhere,
 * and one tile's own manifest `note` calls its bright border a background
 * artifact. Hence the honest output is the NUMBER (a per-tile regression signal:
 * it should not grow between captures), not the boolean: always log the count,
 * warn on the floor, never fail — which is also the issue author's explicit
 * instruction, cropping being a judgement call at the margin.
 *
 * AUDIT BASIS. The audit read the committed, lossy 900×900 VP9 `.webm` masters
 * (`-crf 24`, `yuv420p`; hence the 3596-px perimeters above), worst of every 8th
 * frame; the check reads the lossless 1080×1080 PNG screenshot, perimeter
 * **4316**. So the audit sizes the expected noise level; it does not predict live
 * counts.
 */

/**
 * Warning floor: border-lit pixels in ANY sampled pose above this count trigger
 * the warning. Default 0, i.e. any border-lit pixel warns, per the issue's spec
 * ("non-zero → warn") — which on the audit above means most tiles will warn. That
 * is tolerable only because this warns rather than fails and because the logged
 * COUNT is the actual signal. If the noise proves unhelpful, raise this: a floor
 * of ~1% of the measured `borderPixels` (≈43 px at the 1080×1080 capture
 * geometry) would still have caught the measured 218-px worst frame while
 * silencing the speck-level cases. The whole-perimeter wash class would need a
 * brightness- or localization-aware measurement instead, not a bigger count.
 */
export const BORDER_LIT_MAX = 0;

/**
 * Warn-only under-fill floors. The snapshot measured frame 0 of the 29 committed
 * 340 px animated WebP tiles, a mixed-vintage set rendered from 2026-07-15 through
 * 2026-08-26; its minima were 51.0% span and 12.7% lit area. The runtime check
 * instead reads the settled PNG. WebP and PNG agree closely for the same current
 * render, but an older committed tile can drift enough to warn; that warning is
 * the intended staleness signal, not a reason to suppress it. Re-derive these
 * snapshot floors from a full run's `final coverage=` / `final lit=` lines when
 * the tile set or renderer changes. The two signals are deliberately independent:
 * coverage is the wider percentile-bbox axis, while lit fraction catches a long,
 * thin subject whose span can look full despite occupying little screen area.
 */
export const COVERAGE_MIN = 0.5;
export const LIT_FRACTION_MIN = 0.1;

export interface CoverageMeasurement {
  /** Wider 3rd–97th-percentile bounding-box axis as a fraction of the frame. */
  coverage: number;
  /** Fraction of frame pixels whose luma exceeds `LIT_THRESHOLD`. */
  litFraction: number;
}

export interface UnderfillVerdict {
  /** Whether either under-fill signal is below its warning floor. */
  underfilled: boolean;
  /** Human-readable warning, non-null iff `underfilled` is true. */
  message: string | null;
}

/** Decide whether the final still looks under-filled. This verdict never fails a capture. */
export function evaluateUnderfill(args: {
  demoId: string;
  measurement: CoverageMeasurement;
}): UnderfillVerdict {
  const { demoId, measurement } = args;
  const lowSpan = measurement.coverage < COVERAGE_MIN;
  const lowArea = measurement.litFraction < LIT_FRACTION_MIN;
  if (!lowSpan && !lowArea) return { underfilled: false, message: null };

  const span = `${(measurement.coverage * 100).toFixed(1)}% span`;
  const area = `${(measurement.litFraction * 100).toFixed(1)}% lit area`;
  const failures: string[] = [];
  if (lowSpan) failures.push(`span is below minimum ${(COVERAGE_MIN * 100).toFixed(1)}%`);
  if (lowArea) failures.push(`lit area is below minimum ${(LIT_FRACTION_MIN * 100).toFixed(1)}%`);
  return {
    underfilled: true,
    message:
      `[${demoId}] under-filled? ${span}, ${area}; ${failures.join('; ')}. ` +
      'This is a warning only: inspect the tile before changing its framing.',
  };
}

/**
 * How much lower a `fillTarget` to suggest when a crop is detected. The ladder
 * above took ~0.15 of fill target to undo its crop, so one 0.1 step is a starting
 * point, not a solution — and deliberately not the full 0.15, since overshooting
 * the fill costs screen area on every tile and the crop threshold is
 * subject-dependent.
 */
export const FILL_TARGET_SUGGEST_STEP = 0.1;

/**
 * Floor for a suggested `fillTarget`. Below roughly a third of the frame the
 * subject stops reading as a hero shot at all, so no suggestion is made that
 * would land under it: at that point the framing is not the problem (the subject
 * probably has a detached far-flung fragment) and a human has to look. Without
 * this the message would emit a no-op ("from 0.3 to 0.3") or, lower still, an
 * INVERTED suggestion that tightens the crop.
 */
export const FILL_TARGET_MIN = 0.3;

/**
 * Angular spacing, in degrees, of the orbit poses the harness measures — see
 * {@link borderSampleFrames}.
 *
 * WHY A GRID AND NOT JUST THE TWO EXTREMES. The rock's endpoints are the poses
 * FURTHEST from the framed still, but they are not necessarily the poses of
 * greatest projected extent: a feature sitting ~90° round from the screen-x axis
 * reaches its maximum projected |x| at an INTERMEDIATE rock angle and comes back
 * in by the endpoint, so an endpoints-only check can read zero while the frames in
 * between visibly crop.
 *
 * 5° is fine enough to be equivalent to measuring all 120 frames. The worst pose
 * is either an endpoint (sampled exactly) or a stationary point of the projected
 * extent, and at a stationary point the extent falls off only as 1 − cos δ: with
 * the true worst pose at most `step`/2 = 2.5° from a sampled one, we see it to
 * within 0.1% of its extent. Halving the step would buy 0.03%; measuring every
 * frame costs 120 in-page PNG decodes per tile to buy the same.
 */
export const BORDER_SAMPLE_STEP_DEG = 5;

/**
 * The orbit frame indices to measure for border-lit content: for each target
 * angle on a {@link BORDER_SAMPLE_STEP_DEG} grid spanning the WHOLE ±`amplitudeDeg`
 * rock, the captured frame closest to it. Returned sorted, without duplicates.
 *
 * Frames are captured at `amplitudeDeg·sin(2π·i/orbitFrames)`, i.e. bunched near
 * the endpoints, so a stride over `i` would sample angles very unevenly; picking
 * the nearest frame per target angle spreads the poses over the sweep instead.
 * Ties go to the lower index, which makes the choice deterministic.
 *
 * The grid is capped at one target per captured frame — that keeps a smoke run
 * (`GALLERY_ORBIT_FRAMES=4`) from asking for more distinct poses than exist, and
 * makes an absurd `stepDeg` degrade to "measure every frame" rather than spin.
 * `orbitFrames <= 0` yields no poses.
 */
export function borderSampleFrames(
  orbitFrames: number,
  amplitudeDeg: number,
  stepDeg: number = BORDER_SAMPLE_STEP_DEG
): number[] {
  if (!(orbitFrames > 0)) return [];
  const angleAt = (i: number): number => amplitudeDeg * Math.sin((i / orbitFrames) * Math.PI * 2);
  const steps = Math.max(1, Math.min(orbitFrames, Math.round((2 * amplitudeDeg) / stepDeg)));
  const picked = new Set<number>();
  for (let k = 0; k <= steps; k++) {
    const target = -amplitudeDeg + (2 * amplitudeDeg * k) / steps;
    let best = 0;
    let bestErr = Infinity;
    for (let i = 0; i < orbitFrames; i++) {
      const err = Math.abs(angleAt(i) - target);
      if (err < bestErr) {
        bestErr = err;
        best = i;
      }
    }
    picked.add(best);
  }
  return [...picked].sort((a, b) => a - b);
}

/** One measured pose: how much lit content sat on the frame's outermost ring. */
export interface BorderSample {
  /** Human label for the pose, e.g. `'still'` or `'rock -20°'`. */
  label: string;
  /** Lit pixels (luma > LIT_THRESHOLD) on the outermost row/column. */
  borderLit: number;
  /**
   * Perimeter pixel count they were counted out of (2·w + 2·h − 4 for any real
   * frame; `w·h` for a degenerate 1-px-wide/tall one, where that formula
   * over-counts).
   */
  borderPixels: number;
}

/**
 * Select the poses that can affect the published media's crop verdict.
 * No-orbit tiles publish only the curated still, while the caller may continue
 * measuring and reporting orbit poses as diagnostics.
 */
export function selectCropVerdictSamples(
  stillSample: BorderSample | null,
  orbitSamples: BorderSample[],
  noOrbitVideo: boolean
): BorderSample[] {
  return [...(stillSample ? [stillSample] : []), ...(noOrbitVideo ? [] : orbitSamples)];
}

/**
 * Border-lit pixels as a PERCENTAGE of the frame perimeter — the
 * resolution-independent reading, used by the warning and by the harness's log
 * line so the one division has a single guarded home. Zero when `borderPixels`
 * is 0.
 */
export function borderLitPercent(sample: BorderSample): number {
  return sample.borderPixels > 0 ? (sample.borderLit / sample.borderPixels) * 100 : 0;
}

/**
 * The framing knobs in effect for this demo, mirroring the manifest fields of the
 * same names (see `DemoEntry` in `generate-gallery.spec.ts`). They decide WHICH
 * knob the warning points at: the three framing PATHS (`distance`, baked,
 * closed-loop fill) are mutually exclusive in the harness, but the `zoom` nudge
 * is applied after ALL of them, so it can compound any of the three.
 */
export interface CropFraming {
  /**
   * The fill target the closed-loop fill used — or would have used, since the
   * harness fills this in (`demo.fillTarget ?? FILL_TARGET`) on every path,
   * including the ones that bypass the fill. Already defaulted by the caller, so
   * this module needs no copy of the harness's `FILL_TARGET`.
   */
  fillTarget: number;
  /** Per-demo zoom nudge applied AFTER the framing path (>1 = closer). */
  zoom?: number;
  /** Absolute camera distance; when set, the closed-loop fill was bypassed. */
  distance?: number;
  /**
   * `false` means the fill loop was skipped and framing is whatever `F` gave:
   * the scene's authored `viewer_config` camera when it has one, else a plain
   * bounds fit. UNDEFINED is the common case (most manifest demos omit it) and
   * means the closed-loop fill DID run — so this must be tested against `false`
   * explicitly, never for falsiness.
   */
  autoFrame?: boolean;
}

/** Outcome of {@link evaluateBorderLit}. */
export interface CropVerdict {
  /** Whether any sampled pose exceeded the floor. */
  cropped: boolean;
  /** The worst sampled pose, or null when no poses were measured. */
  worst: BorderSample | null;
  /** One-line warning, non-null iff `cropped`. */
  message: string | null;
}

/** Round to 2 decimals — suggestions are manifest values, not full precision. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/**
 * Decide whether a capture looks cropped, and say what to do about it.
 *
 * Pure and synchronous: it logs nothing (the harness owns the console), and the
 * caller decides whether to warn. `worst` is the pose with the highest border-lit
 * COUNT — ties keep the FIRST, so the `'still'` sample wins over an orbit pose
 * with the same reading and the message names the pose easiest to reproduce.
 * Ranking by raw count is sound because every sample is a screenshot of the same
 * Playwright viewport, so count and perimeter fraction rank identically; it also
 * keeps `cropped` (a count against `floor`) consistent with `worst`. If poses are
 * ever captured at differing resolutions, BOTH this ranking and the count-based
 * floor need revisiting.
 *
 * The advice follows whichever framing path the demo actually took, because
 * suggesting a knob that is not in play would be worse than saying nothing:
 *   - `distance` set → the closed-loop fill was bypassed; RAISE the distance.
 *   - `autoFrame: false` → the fill was skipped, so framing is whatever `F`
 *     produced (an authored `viewer_config` camera, or a plain bounds fit when
 *     the scene authors none); the crop has to be fixed there.
 *   - otherwise → lower `fillTarget` by {@link FILL_TARGET_SUGGEST_STEP}, unless
 *     that would land under {@link FILL_TARGET_MIN}.
 * A `zoom` > 1 is then appended on ALL THREE paths, because the harness applies
 * that dolly last on all three — and being last makes it the likeliest culprit.
 *
 * `floor` exists so the documented escape hatch (raising
 * {@link BORDER_LIT_MAX} when the warning proves noisy) is a tested contract
 * rather than an untested claim. It is NOT a per-call knob for the harness,
 * which always takes the default.
 */
export function evaluateBorderLit(
  args: {
    demoId: string;
    samples: BorderSample[];
    framing: CropFraming;
  },
  floor: number = BORDER_LIT_MAX
): CropVerdict {
  const { demoId, samples, framing } = args;
  let worst: BorderSample | null = null;
  for (const s of samples) {
    if (worst === null || s.borderLit > worst.borderLit) worst = s;
  }
  const cropped = worst !== null && worst.borderLit > floor;
  if (!cropped || worst === null) return { cropped: false, worst, message: null };

  const head =
    `[${demoId}] cropped? ${worst.borderLit}/${worst.borderPixels} border pixels ` +
    `(${borderLitPercent(worst).toFixed(1)}% of the frame perimeter) are lit at pose ` +
    `"${worst.label}" — the subject touches the frame edge, which the coverage metric ` +
    'cannot see (its percentile bbox ignores exactly these outliers).';

  let advice: string;
  if (typeof framing.distance === 'number') {
    advice =
      ` Framing came from an absolute distance=${framing.distance} (fill bypassed), ` +
      'so RAISE that distance to pull the camera back.';
  } else if (framing.autoFrame === false) {
    advice =
      " Framing is whatever F produced (autoFrame off): the scene's authored viewer_config " +
      'camera, or a plain bounds fit if it authors none. Fix it there, or give the demo a ' +
      '"distance" to override it.';
  } else {
    // Decide on the value that would actually be PRINTED, so the branch cannot
    // be settled by float noise in `effective − STEP`.
    const suggested = round2(framing.fillTarget - FILL_TARGET_SUGGEST_STEP);
    advice =
      suggested < FILL_TARGET_MIN
        ? ` One step below fillTarget=${framing.fillTarget} would go under the framing floor ` +
          `(FILL_TARGET_MIN ${FILL_TARGET_MIN}), so lowering it is not the answer — the ` +
          'subject probably has a detached far-flung fragment; inspect it by hand.'
        : ` Try lowering fillTarget from ${framing.fillTarget} to ${suggested}.`;
  }
  // The zoom dolly runs after EVERY framing path, so this compounds whichever
  // advice was chosen above rather than replacing it.
  if (typeof framing.zoom === 'number' && framing.zoom > 1) {
    advice +=
      ` The zoom=${framing.zoom} nudge is applied AFTER the framing and is the more ` +
      'likely culprit — lower it first.';
  }
  return { cropped: true, worst, message: head + advice };
}
