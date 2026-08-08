/**
 * Frame similarity — the pure COMPARISON half of the capture harness's
 * still-vs-orbit consistency check.
 *
 * `generate-gallery.spec.ts` owns the IO (read two PNGs, decode and downscale
 * them inside the page, which is the only place a browser decoder exists); this
 * module owns the *arithmetic*. Splitting them lets the maths be unit-tested
 * against synthetic pixel buffers with no browser, dataset or GPU — the same
 * split, for the same reason, as `exposure-policy.ts` and its
 * `src/tests/unit/gallery-exposure-policy.test.ts`.
 *
 * That split is worth making here specifically because this code is a
 * *detector*: it exists to notice a defect (#1377) that already shipped once.
 * A detector whose arithmetic is untested can silently answer "fine" forever,
 * and the absence of a warning is indistinguishable from a passing check.
 */

/**
 * Side length of the greyscale buffers compared. Both frames are downscaled to
 * this before comparison, so a still captured at full resolution and an orbit
 * frame captured at `ORBIT_CAPTURE_PX` are directly comparable.
 *
 * 256² = 65,536 samples: far more than enough to separate "same pose" from
 * "rolled 90°" (measured separation is ~0.45 of correlation), while small
 * enough that the buffers cross the CDP boundary as compact integer JSON.
 */
export const COMPARE_SIZE = 256;

/**
 * Below this correlation, the still and orbit frame 0 are reported as
 * disagreeing.
 *
 * Calibrated on real captures, not chosen for roundness: a consistent pair
 * measures 0.98–1.00 (asteroids 1.000, cosmicflows 0.981) and a rolled pair
 * 0.52–0.56 (asteroids 0.552/0.563, cosmicflows 0.522). 0.85 sits clear of both
 * clusters with ~0.13 of margin to the nearest true positive.
 */
export const DISAGREEMENT_THRESHOLD = 0.85;

/**
 * Rec. 601 luma from 8-bit RGB, rounded to an integer.
 *
 * Rounded deliberately: the result crosses the CDP boundary as JSON, where an
 * integer costs ~4 characters and an unrounded double costs ~20. The lost
 * precision is under half a grey level, which is four orders of magnitude below
 * the ~0.45 correlation gap this feeds — see `COMPARE_SIZE`.
 */
export function luma8(r: number, g: number, b: number): number {
  return Math.round(0.299 * r + 0.587 * g + 0.114 * b);
}

/**
 * Normalised cross-correlation of two equal-length sample buffers.
 *
 * Each buffer is mean-centred and divided by its standard deviation, then the
 * products are averaged — so the result is invariant to overall brightness and
 * contrast, and depends only on *structure*. That invariance is the point: two
 * captures of the same pose can differ in exposure (auto-exposure re-runs) or
 * in how much of a progressively-streamed scene has arrived, and neither should
 * read as "different pose".
 *
 * Returns a value in [-1, 1]:
 *   +1  identical structure
 *    0  unrelated — and also what a FLAT buffer yields, see below
 *   -1  inverted
 *
 * A flat buffer (all pixels equal, e.g. an all-black failed capture) has zero
 * standard deviation and no structure to correlate. Rather than dividing by
 * zero and returning NaN — which would compare false against any threshold and
 * so silently disable the check — it is left mean-centred at all zeros, giving
 * a correlation of 0. That reads as "disagree", which is the right answer: an
 * all-black gallery frame is a failed capture whether or not its counterpart
 * matches it.
 *
 * @param a First buffer.
 * @param b Second buffer, same length as `a`.
 * @returns Correlation in [-1, 1], or 0 when either buffer is flat.
 * @throws If the buffers have different lengths, or are empty — both indicate a
 *   caller bug (mismatched downscale, or a decode that produced nothing) that
 *   would otherwise surface as a meaningless number.
 */
export function normalizedCrossCorrelation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`normalizedCrossCorrelation: length mismatch (${a.length} vs ${b.length})`);
  }
  if (a.length === 0) {
    throw new Error('normalizedCrossCorrelation: empty buffers');
  }
  const za = standardize(a);
  const zb = standardize(b);
  let acc = 0;
  for (let i = 0; i < za.length; i++) acc += za[i] * zb[i];
  return acc / za.length;
}

/** Mean-centre and scale to unit standard deviation; all-zeros when flat. */
function standardize(src: ArrayLike<number>): Float64Array {
  const out = new Float64Array(src.length);
  let sum = 0;
  for (let i = 0; i < src.length; i++) sum += src[i];
  const mean = sum / src.length;
  let ss = 0;
  for (let i = 0; i < src.length; i++) {
    const d = src[i] - mean;
    out[i] = d;
    ss += d * d;
  }
  const sd = Math.sqrt(ss / src.length);
  if (sd > 0) for (let i = 0; i < out.length; i++) out[i] /= sd;
  return out;
}
