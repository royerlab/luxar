/**
 * Line-join artifact metrics — the acceptance harness for issue #790.
 *
 * ## What this measures
 *
 * Consecutive line segments are drawn as independent screen-space quads
 * with no join geometry between them, so at every bend the two quads leave
 * an uncovered wedge on the OUTSIDE of the turn and double-cover a lens on
 * the INSIDE. On a thick curved line that reads as near-black ticks along
 * the outer edge and fainter bright ticks on the inner edge under additive
 * blending. This module turns "reads as ticks" into two numbers so the
 * miter-join series can be judged, and so the already-fixed joint defects
 * (#780, #785) cannot silently regress while the vertex stage is rewritten.
 *
 * ## Why there are two metrics, not one
 *
 * They are blind to different failures, and neither subsumes the other:
 *
 * 1. {@link measureLocalMedianOutliers} compares each pixel to the median of
 *    its own small neighbourhood. It is exquisitely sensitive to the narrow
 *    one-to-two-pixel wedge tick of #790 — a lone dark pixel in a bright
 *    neighbourhood — and completely blind to anything smooth, because a
 *    local median tracks a smooth signal by construction.
 * 2. {@link measureAxialFlux} sums luminance across the tube's cross-section
 *    at each position ALONG its axis and normalises by the profile's own
 *    median. This catches exactly the class the median metric cannot see: a
 *    smooth 50% flux ramp at every interior joint — the #780 bead-chain
 *    failure — scores **zero** local-median outliers while the flux
 *    profile's p05 collapses from 1.0 toward 0.7.
 *
 * A regression in either direction therefore has to move at least one of
 * the two numbers.
 *
 * ## Sensitivity envelope of the local-median metric (READ THIS)
 *
 * A local median is a small-scale detector and nothing else. Because the
 * defect sits inside the very window the median is taken over, a wide defect
 * poisons its own reference: once a dark run is at least `(window + 1) / 2`
 * pixels across, the local median goes dark too, the pixel then fails the
 * background mask, and the defect scores **zero**. Measured on a 21-pixel
 * band at the default `window = 5`:
 *
 * | Notch width | Dark outliers |
 * | ----------- | ------------- |
 * | 1 px        | 19            |
 * | 2 px        | 34            |
 * | 3 px        | 0             |
 * | 5 px        | 0             |
 * | 7 px        | 0             |
 *
 * So `darkFraction` is a **detector, not a severity measure**, it is
 * non-monotone in defect width, and it is **not comparable between bands of
 * different turn angle**. The live #790 baseline shows this plainly: the
 * GENTLE `curve_smooth` band (~10.5° turns, ~1.3 px wedge chord) scores
 * ~5.1% while the 90° `zigzag_right_angle` band — whose wedge is a 7.2
 * px-radius quarter disc, a far worse defect — scores ~0.12%, because its
 * wedge is too wide for the window to see. For wide defects read
 * {@link measureAxialFlux} instead: on that same frame the zigzag's flux p05
 * is 0.714 against 1.000 on the straight bands. The unit tests pin this
 * envelope so a change to `window` that silently alters sensitivity fails.
 *
 * ## Display encoding
 *
 * Both metrics operate on DISPLAY-encoded (sRGB) luminance, because that is
 * what a composited screenshot carries. The encoding is monotone, so a
 * defect always registers — but it is also compressive near white: around a
 * typical mid-bright tube value a small relative deviation measures about
 * **2.3x smaller** than it is in linear light (a 4.0% linear ripple reads as
 * ~1.76%). Any future change to the absolute `threshold` must be chosen with
 * that factor in mind.
 *
 * ## Contract
 *
 * Everything here is a pure function over plain data (a luminance array plus
 * a width/height and a rectangular region of interest), so it is unit-tested
 * without a browser (`src/tests/unit/rendering/line-join-metrics.test.ts`)
 * and reused verbatim by the E2E acceptance spec
 * (`src/tests/e2e/line-join-artifact.spec.ts`).
 *
 * Degenerate inputs are reported, never papered over: an empty region, a
 * region containing only background, and a region wholly outside the image
 * all return `insidePixels: 0` rather than a zero-outlier "clean" verdict.
 * Callers are expected to assert `insidePixels` is non-trivial so a blank
 * frame or a mislocated rectangle fails loudly.
 */

/** A rectangular region of interest in pixel coordinates (top-left origin). */
export interface PixelRect {
  /** Left edge, in pixels. May be negative or beyond the image; it is clipped. */
  x: number;
  /** Top edge, in pixels. May be negative or beyond the image; it is clipped. */
  y: number;
  /** Width in pixels. Zero or negative means an empty region. */
  width: number;
  /** Height in pixels. Zero or negative means an empty region. */
  height: number;
}

/** A single-channel luminance image, one value per pixel, row-major. */
export type LuminanceArray = Float32Array | Uint8ClampedArray;

/** Tuning knobs for {@link measureLocalMedianOutliers}. */
export interface LocalMedianOptions {
  /**
   * Side length of the square neighbourhood the median is taken over. Must
   * be an odd integer `>= 3` — at `1` the "median" is the pixel itself and
   * the metric can never report an outlier. Default `5` — wide enough to
   * step over a one-to-two-pixel wedge tick, narrow enough to track a smooth
   * ramp. Changing it moves the sensitivity envelope documented in the
   * module header; the unit tests pin that envelope deliberately.
   */
  window?: number;
  /**
   * Absolute luminance deviation (0-255 scale) beyond which a pixel counts
   * as an outlier. Default `25`.
   */
  threshold?: number;
  /**
   * Luminance below which a pixel counts as background and is excluded from
   * "inside". The test is applied to the pixel's LOCAL MEDIAN, not to the
   * pixel itself: a #790 wedge tick is background-dark on its own yet sits
   * in a bright neighbourhood, and it is precisely the pixel that must be
   * counted. Default `12`.
   */
  backgroundCutoff?: number;
}

/** Outcome of {@link measureLocalMedianOutliers}. */
export interface LocalMedianOutlierResult {
  /** Pixels of the clipped region that passed the background mask. */
  insidePixels: number;
  /** Inside pixels dimmer than their local median by more than the threshold. */
  darkOutliers: number;
  /** Inside pixels brighter than their local median by more than the threshold. */
  brightOutliers: number;
  /** `darkOutliers / insidePixels`, or `0` when there are no inside pixels. */
  darkFraction: number;
  /** `brightOutliers / insidePixels`, or `0` when there are no inside pixels. */
  brightFraction: number;
  /** Largest `median - value` seen over the dark outliers (`0` if none). */
  worstDeficit: number;
  /** Largest `value - median` seen over the bright outliers (`0` if none). */
  worstExcess: number;
}

/** Tuning knobs for {@link measureAxialFlux}. */
export interface AxialFluxOptions {
  /**
   * Luminance below which a pixel is treated as background and left out of
   * the cross-section sum. Default `12`.
   */
  backgroundCutoff?: number;
}

/** Outcome of {@link measureAxialFlux}. */
export interface AxialFluxResult {
  /** Above-cutoff pixels that contributed to the profile. */
  insidePixels: number;
  /**
   * Axial positions in the profile, i.e. the region's extent along the axis
   * minus the leading and trailing all-background runs. Compare it against
   * the region's own width/height: a tube that is continuous by construction
   * must give `samples === rect.width` (or `rect.height` for `'y'`).
   */
  samples: number;
  /**
   * Positions INSIDE the profile whose cross-section was entirely
   * background, i.e. holes in the tube. Recorded as `0` in
   * {@link profile} rather than dropped — see the function docs.
   */
  emptySamples: number;
  /** Raw (un-normalised) median cross-section sum, in luminance units. */
  medianFlux: number;
  /** Cross-section sums divided by {@link medianFlux}, in axial order. */
  profile: number[];
  /** 5th percentile of {@link profile} — the joint-dip detector. */
  p05: number;
  /**
   * 50th percentile of {@link profile}. Exactly `1` whenever the profile is
   * non-empty, because the normaliser uses the same rank — so it carries no
   * information and asserting on it would be vacuous.
   */
  p50: number;
  /** 95th percentile of {@link profile}. */
  p95: number;
  /** Smallest value of {@link profile}. */
  min: number;
}

/** Rec.709 luma weights — the standard perceptual RGB → luminance mixdown. */
const REC709_R = 0.2126;
const REC709_G = 0.7152;
const REC709_B = 0.0722;

const DEFAULT_WINDOW = 5;
const DEFAULT_THRESHOLD = 25;
const DEFAULT_BACKGROUND_CUTOFF = 12;

/**
 * Convert an interleaved RGBA byte buffer — the layout `getImageData()`
 * returns — to a one-value-per-pixel luminance array using the Rec.709
 * weights. Alpha is ignored: the canvas is already composited.
 *
 * @param rgba Interleaved RGBA bytes, at least `width * height * 4` long.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @returns A `Float32Array` of `width * height` luminance values in `[0, 255]`.
 * @throws If the dimensions are not positive or the buffer is too short.
 */
export function rgbaToLuminance(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number
): Float32Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`rgbaToLuminance: invalid image size ${width}x${height}`);
  }
  const needed = width * height * 4;
  if (rgba.length < needed) {
    throw new Error(
      `rgbaToLuminance: buffer holds ${rgba.length} bytes, ${needed} needed for ${width}x${height}`
    );
  }
  const out = new Float32Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = REC709_R * rgba[p] + REC709_G * rgba[p + 1] + REC709_B * rgba[p + 2];
  }
  return out;
}

/**
 * Intersect a region with the image bounds.
 *
 * @returns The clipped rectangle, or `null` when nothing survives.
 */
function clipRect(region: PixelRect, width: number, height: number): PixelRect | null {
  const x0 = Math.max(0, Math.floor(region.x));
  const y0 = Math.max(0, Math.floor(region.y));
  const x1 = Math.min(width, Math.ceil(region.x + region.width));
  const y1 = Math.min(height, Math.ceil(region.y + region.height));
  if (x1 <= x0 || y1 <= y0) return null;
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

/**
 * Nearest-rank percentile of an ascending-sorted array.
 *
 * Using the same rank rule for the normalising median and for `p50` is what
 * makes {@link AxialFluxResult.p50} exactly `1`.
 */
function percentileSorted(sorted: readonly number[], fraction: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const idx = Math.min(n - 1, Math.max(0, Math.round(fraction * (n - 1))));
  return sorted[idx];
}

/**
 * Median of the `window x window` neighbourhood centred on `(cx, cy)`.
 *
 * The neighbourhood is clipped to the IMAGE, not to the region of interest,
 * so a tightly-cropped band still sees its own surroundings; near an image
 * border the window is simply truncated.
 *
 * @param scratch Reused buffer, at least `window * window` long.
 */
function neighbourhoodMedian(
  luminance: LuminanceArray,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
  scratch: Float64Array
): number {
  const yStart = Math.max(0, cy - radius);
  const yEnd = Math.min(height - 1, cy + radius);
  const xStart = Math.max(0, cx - radius);
  const xEnd = Math.min(width - 1, cx + radius);
  let n = 0;
  for (let y = yStart; y <= yEnd; y++) {
    const row = y * width;
    for (let x = xStart; x <= xEnd; x++) {
      scratch[n++] = luminance[row + x];
    }
  }
  const slice = scratch.subarray(0, n);
  slice.sort();
  const mid = n >> 1;
  return n % 2 === 1 ? slice[mid] : 0.5 * (slice[mid - 1] + slice[mid]);
}

/**
 * Count pixels that deviate from their own local median — the narrow-tick
 * detector.
 *
 * Every pixel of the clipped region whose local median clears
 * `backgroundCutoff` is "inside"; each inside pixel is then compared to that
 * median and counted as a dark or bright outlier when it deviates by more
 * than `threshold`. This is the metric that sees the #790 outer-side miter
 * wedge (a one-to-two-pixel dark tick crossing an otherwise uniform tube),
 * and the metric that is deliberately blind to any smooth variation — see
 * {@link measureAxialFlux} for that half.
 *
 * It is ALSO blind to any defect wider than about `(window + 1) / 2` pixels,
 * which makes the returned fractions a detector rather than a severity
 * measure and makes them incomparable across bands of different turn angle.
 * Read the "Sensitivity envelope" section of the module header before
 * interpreting a number from this function.
 *
 * @param luminance One value per pixel, row-major, on a 0-255 scale.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param region Region of interest; clipped to the image.
 * @param options See {@link LocalMedianOptions}.
 * @returns Counts, fractions and worst deviations. `insidePixels` is `0` for
 *   an empty, out-of-image or all-background region — assert on it rather
 *   than reading a zero outlier count as "clean".
 * @throws If the image size is invalid, the buffer is too short, or `window`
 *   is not an odd integer `>= 3`.
 */
export function measureLocalMedianOutliers(
  luminance: LuminanceArray,
  width: number,
  height: number,
  region: PixelRect,
  options: LocalMedianOptions = {}
): LocalMedianOutlierResult {
  const window = options.window ?? DEFAULT_WINDOW;
  const threshold = options.threshold ?? DEFAULT_THRESHOLD;
  const cutoff = options.backgroundCutoff ?? DEFAULT_BACKGROUND_CUTOFF;

  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`measureLocalMedianOutliers: invalid image size ${width}x${height}`);
  }
  if (luminance.length < width * height) {
    throw new Error(
      `measureLocalMedianOutliers: buffer holds ${luminance.length} values, ` +
        `${width * height} needed for ${width}x${height}`
    );
  }
  if (!Number.isInteger(window) || window < 3 || window % 2 === 0) {
    // `window: 1` makes the median the pixel itself, so the metric would
    // report zero outliers forever — reject it rather than lie.
    throw new Error(
      `measureLocalMedianOutliers: window must be an odd integer >= 3, got ${window}`
    );
  }
  if (!(threshold >= 0)) {
    throw new Error(`measureLocalMedianOutliers: threshold must be >= 0, got ${threshold}`);
  }

  const empty: LocalMedianOutlierResult = {
    insidePixels: 0,
    darkOutliers: 0,
    brightOutliers: 0,
    darkFraction: 0,
    brightFraction: 0,
    worstDeficit: 0,
    worstExcess: 0,
  };

  const rect = clipRect(region, width, height);
  if (!rect) return empty;

  const radius = (window - 1) / 2;
  const scratch = new Float64Array(window * window);

  let insidePixels = 0;
  let darkOutliers = 0;
  let brightOutliers = 0;
  let worstDeficit = 0;
  let worstExcess = 0;

  for (let y = rect.y; y < rect.y + rect.height; y++) {
    const row = y * width;
    for (let x = rect.x; x < rect.x + rect.width; x++) {
      const median = neighbourhoodMedian(luminance, width, height, x, y, radius, scratch);
      if (median <= cutoff) continue;
      insidePixels++;
      const deviation = median - luminance[row + x];
      if (deviation > threshold) {
        darkOutliers++;
        if (deviation > worstDeficit) worstDeficit = deviation;
      } else if (-deviation > threshold) {
        brightOutliers++;
        if (-deviation > worstExcess) worstExcess = -deviation;
      }
    }
  }

  return {
    insidePixels,
    darkOutliers,
    brightOutliers,
    darkFraction: insidePixels > 0 ? darkOutliers / insidePixels : 0,
    brightFraction: insidePixels > 0 ? brightOutliers / insidePixels : 0,
    worstDeficit,
    worstExcess,
  };
}

/**
 * Profile the tube's total flux along its axis — the smooth-dip detector.
 *
 * For every position along `axis` the above-cutoff luminance of the
 * perpendicular cross-section is summed, then the whole profile is divided
 * by its own median so the result is a relative "how much of the tube is
 * here" curve centred on `1.0`. A tube that renders at uniform brightness
 * gives a dead-flat profile; the #780 bead chain — every interior joint
 * dimmed to 50% over a width-sized, perfectly smooth ramp — leaves
 * {@link measureLocalMedianOutliers} at zero outliers while dragging `p05`
 * from 1.0 down toward 0.7. That asymmetry is the entire reason this second
 * metric exists.
 *
 * Only the LEADING and TRAILING all-background runs are trimmed — the case
 * where the region overhangs the ends of the tube. An all-background
 * position in the INTERIOR is a hole in the tube and is recorded as `0`,
 * counted in {@link AxialFluxResult.emptySamples}. Skipping interior holes
 * instead (the obvious implementation) is catastrophic: a tube missing every
 * other 8-pixel run — half the line gone — closes back up into a perfectly
 * flat profile of `p05 = p50 = p95 = 1` and zero median outliers, so every
 * assertion an acceptance spec could make would pass on a renderer that lost
 * half the geometry.
 *
 * @param luminance One value per pixel, row-major, on a 0-255 scale.
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param region Region of interest; clipped to the image.
 * @param axis Which image axis the tube runs along: `'x'` sums columns,
 *   `'y'` sums rows.
 * @param options See {@link AxialFluxOptions}.
 * @returns The normalised profile and its percentiles. `insidePixels` and
 *   `samples` are `0` for an empty, out-of-image or all-background region,
 *   and the percentiles are then `0` — a value that fails a `> 0.9` guard
 *   loudly instead of reading as clean. The same holds when more than half
 *   the profile is empty, which leaves the median at zero and the profile
 *   unnormalisable.
 * @throws If the image size is invalid or the buffer is too short.
 */
export function measureAxialFlux(
  luminance: LuminanceArray,
  width: number,
  height: number,
  region: PixelRect,
  axis: 'x' | 'y',
  options: AxialFluxOptions = {}
): AxialFluxResult {
  const cutoff = options.backgroundCutoff ?? DEFAULT_BACKGROUND_CUTOFF;

  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`measureAxialFlux: invalid image size ${width}x${height}`);
  }
  if (luminance.length < width * height) {
    throw new Error(
      `measureAxialFlux: buffer holds ${luminance.length} values, ` +
        `${width * height} needed for ${width}x${height}`
    );
  }

  const empty: AxialFluxResult = {
    insidePixels: 0,
    samples: 0,
    emptySamples: 0,
    medianFlux: 0,
    profile: [],
    p05: 0,
    p50: 0,
    p95: 0,
    min: 0,
  };

  const rect = clipRect(region, width, height);
  if (!rect) return empty;

  const alongCount = axis === 'x' ? rect.width : rect.height;
  const acrossCount = axis === 'x' ? rect.height : rect.width;
  const raw: number[] = [];
  let insidePixels = 0;

  for (let a = 0; a < alongCount; a++) {
    let sum = 0;
    let count = 0;
    for (let c = 0; c < acrossCount; c++) {
      const x = axis === 'x' ? rect.x + a : rect.x + c;
      const y = axis === 'x' ? rect.y + c : rect.y + a;
      const value = luminance[y * width + x];
      if (value > cutoff) {
        sum += value;
        count++;
      }
    }
    raw.push(count > 0 ? sum : 0);
    insidePixels += count;
  }

  // Trim the leading/trailing overhang only — interior zeros are holes in
  // the tube and MUST survive into the profile.
  let first = 0;
  while (first < raw.length && raw[first] === 0) first++;
  let last = raw.length - 1;
  while (last >= first && raw[last] === 0) last--;
  if (first > last) return empty;

  const flux = raw.slice(first, last + 1);
  const emptySamples = flux.reduce((n, f) => (f === 0 ? n + 1 : n), 0);

  const sorted = [...flux].sort((p, q) => p - q);
  const medianFlux = percentileSorted(sorted, 0.5);
  if (medianFlux <= 0) {
    // More than half the tube is missing — there is no sane normaliser, so
    // report the counts and let the zeroed percentiles fail the caller's
    // guard rather than inventing a scale.
    return { ...empty, insidePixels, samples: flux.length, emptySamples };
  }

  const profile = flux.map((f) => f / medianFlux);
  const sortedProfile = sorted.map((f) => f / medianFlux);

  return {
    insidePixels,
    samples: flux.length,
    emptySamples,
    medianFlux,
    profile,
    p05: percentileSorted(sortedProfile, 0.05),
    p50: percentileSorted(sortedProfile, 0.5),
    p95: percentileSorted(sortedProfile, 0.95),
    min: sortedProfile[0],
  };
}
