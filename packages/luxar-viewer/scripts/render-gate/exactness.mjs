/**
 * Exactness scoring for the render gate: compare two float captures of the
 * same frame (baseline build vs candidate build) and classify the difference.
 *
 * The viewer's HDR and LDR targets are HalfFloat, so a difference is measured
 * in HALF-FLOAT STEPS of the pixel's own value (ULP16), not in absolute units:
 * one step is ~1e-3 of a bright pixel and far less of a dim one, and a scorer
 * with an absolute tolerance would be either blind on dim pixels or
 * impossibly strict on bright ones.
 *
 * Two populations of differing pixels behave differently and are reported
 * separately:
 *   - "drift": a pixel whose value moved by a couple of ULP16 because the
 *     arithmetic producing it changed (reordered, fused, derived elsewhere);
 *   - "flips": a pixel whose DISCRETE decision changed (a fragment discarded
 *     or not at a Gaussian cutoff, a coverage edge, a depth tie). Its value
 *     jumps by far more than a few ULP16, but only at isolated pixels.
 * A math-moving change is expected to show a little drift and very few flips;
 * a real regression shows many flips or large drift.
 *
 * @module scripts/render-gate/exactness
 */

/** Smallest positive half-float subnormal, 2^-24: the ULP16 of values below 2^-14. */
const HALF_MIN_SUBNORMAL = 2 ** -24;
/** Smallest positive normal half-float, 2^-14. */
const HALF_MIN_NORMAL = 2 ** -14;

/** A difference larger than this many ULP16 counts as a flip rather than drift. */
export const FLIP_ULP = 2;

/**
 * Distance between adjacent half-floats around `value` (the half-float ULP).
 *
 * @param {number} value Any finite number.
 * @returns {number} The ULP16 at |value|; 2^-24 in the subnormal range.
 */
export function ulp16(value) {
  const a = Math.abs(value);
  if (a < HALF_MIN_NORMAL) return HALF_MIN_SUBNORMAL;
  return 2 ** (Math.floor(Math.log2(a)) - 10);
}

/**
 * Difference between two channel values in ULP16 of the larger magnitude.
 * Two NaNs agree; one NaN (or one infinity) is an infinite difference.
 *
 * @param {number} a Baseline value.
 * @param {number} b Candidate value.
 * @returns {number} Non-negative ULP16 distance (Infinity for a NaN mismatch).
 */
export function ulpDistance(a, b) {
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.isNaN(a) && Number.isNaN(b) ? 0 : Infinity;
  if (a === b) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return Infinity;
  return Math.abs(a - b) / ulp16(Math.max(Math.abs(a), Math.abs(b)));
}

/**
 * Value at quantile `q` of an unsorted numeric array (nearest-rank).
 *
 * @param {Float64Array|number[]} values Samples; not modified.
 * @param {number} q Quantile in [0, 1].
 * @returns {number} The sample at that rank, or 0 for an empty array.
 */
export function quantile(values, q) {
  if (values.length === 0) return 0;
  const sorted = Float64Array.from(values).sort();
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
  return sorted[rank];
}

/**
 * Scale-aware comparison: does the candidate move LIGHT, or only jitter it?
 *
 * A per-pixel score cannot tell a systematic error (a line 1e-4 too wide
 * carries 1e-4 more light and brightens every neighbourhood it crosses) from
 * rounding jitter (a vertex position one ulp off moves light between two
 * neighbouring pixels and changes nothing in their sum). Box-filtering both
 * frames into `block`x`block` tiles cancels the jitter and keeps the
 * systematic part, so the worst tile change measures what a viewer could
 * actually see.
 *
 * @param {Float32Array} base Baseline RGBA pixels.
 * @param {Float32Array} cand Candidate RGBA pixels.
 * @param {number} width Frame width in pixels.
 * @param {number} height Frame height in pixels.
 * @param {number} [block=4] Tile edge in pixels.
 * @returns {{ energyRel: number, blockMaxRel: number, blockP999Rel: number }}
 *   `energyRel`: signed relative change of the frame's total RGB energy.
 *   `blockMaxRel` / `blockP999Rel`: max / p99.9 over tiles of |ΔE_tile|,
 *   relative to the brightest baseline tile.
 */
export function scoreBlocks(base, cand, width, height, block = 4) {
  const bw = Math.ceil(width / block);
  const bh = Math.ceil(height / block);
  const eb = new Float64Array(bw * bh);
  const ec = new Float64Array(bw * bh);
  let totalB = 0;
  let totalC = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      let sb = 0;
      let sc = 0;
      for (let c = 0; c < 3; c++) {
        const a = base[i + c];
        const b = cand[i + c];
        if (Number.isFinite(a)) sb += a;
        if (Number.isFinite(b)) sc += b;
      }
      const t = Math.floor(y / block) * bw + Math.floor(x / block);
      eb[t] += sb;
      ec[t] += sc;
      totalB += sb;
      totalC += sc;
    }
  }
  let peakTile = 0;
  for (let t = 0; t < eb.length; t++) peakTile = Math.max(peakTile, eb[t]);
  const rel = new Float64Array(eb.length);
  let maxRel = 0;
  for (let t = 0; t < eb.length; t++) {
    rel[t] = peakTile > 0 ? Math.abs(ec[t] - eb[t]) / peakTile : 0;
    if (rel[t] > maxRel) maxRel = rel[t];
  }
  return {
    energyRel: totalB > 0 ? (totalC - totalB) / totalB : 0,
    blockMaxRel: maxRel,
    blockP999Rel: quantile(rel, 0.999),
  };
}

/**
 * Score two RGBA float captures of identical size.
 *
 * Per pixel the distance is the maximum over its four channels. `peak` is the
 * largest finite RGB magnitude in the BASELINE, so a flip's size is reported
 * relative to the brightest thing in the frame.
 *
 * @param {Float32Array} base Baseline RGBA pixels.
 * @param {Float32Array} cand Candidate RGBA pixels.
 * @returns {{
 *   pixels: number, differing: number, maxUlp: number, maxDriftUlp: number,
 *   p9999Ulp: number, flips: number, flipFraction: number, maxFlipRel: number,
 *   peak: number, perPixelUlp: Float64Array
 * }} Summary statistics plus the per-pixel ULP16 map (for heatmaps).
 */
export function scoreFloatBuffers(base, cand) {
  if (base.length !== cand.length || base.length % 4 !== 0) {
    throw new Error(`capture sizes differ or are not RGBA: ${base.length} vs ${cand.length}`);
  }
  const pixels = base.length / 4;
  const perPixelUlp = new Float64Array(pixels);
  let peak = 0;
  for (let i = 0; i < base.length; i++) {
    if (i % 4 !== 3 && Number.isFinite(base[i])) peak = Math.max(peak, Math.abs(base[i]));
  }
  let differing = 0;
  let flips = 0;
  let maxUlp = 0;
  let maxDriftUlp = 0;
  let maxFlipAbs = 0;
  for (let p = 0; p < pixels; p++) {
    let worst = 0;
    let worstAbs = 0;
    for (let c = 0; c < 4; c++) {
      const a = base[p * 4 + c];
      const b = cand[p * 4 + c];
      const d = ulpDistance(a, b);
      if (d > worst) worst = d;
      const abs = Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) : Infinity;
      if (abs > worstAbs) worstAbs = abs;
    }
    perPixelUlp[p] = worst;
    if (worst > 0) differing++;
    if (worst > maxUlp) maxUlp = worst;
    if (worst > FLIP_ULP) {
      flips++;
      if (worstAbs > maxFlipAbs) maxFlipAbs = worstAbs;
    } else if (worst > maxDriftUlp) {
      maxDriftUlp = worst;
    }
  }
  return {
    pixels,
    differing,
    maxUlp,
    maxDriftUlp,
    p9999Ulp: quantile(perPixelUlp, 0.9999),
    flips,
    flipFraction: pixels > 0 ? flips / pixels : 0,
    maxFlipRel: peak > 0 ? maxFlipAbs / peak : maxFlipAbs > 0 ? Infinity : 0,
    peak,
    perPixelUlp,
  };
}

/**
 * Compare two pick-ID captures. A pick pixel encodes an identity, so there is
 * no drift for it — but there are two levels of identity:
 *   - the NODE (channel R): which object the pixel belongs to;
 *   - the ELEMENT and depth/brightness payload (G, B, A): which of that
 *     node's elements won the pixel.
 * Where elements of one node overlap, the winner is a near-tie that ANY
 * rounding change flips: nudging the line width scale by one float32 ulp
 * (the calibration arm, `calib-lines-1ulp`) changed the element at up to 35%
 * of a dense line scene's pick pixels while not one pixel changed node. So a
 * math-moving change is judged on NODE identity; element changes are
 * reported. `IDENTICAL` still requires both to be zero.
 *
 * `hits` counts baseline pixels that carry an id at all: a pick buffer that is
 * empty in both builds "matches" trivially and certifies nothing, so the
 * harness treats `hits === 0` on a pickable case as an error.
 *
 * @param {Float32Array} base Baseline RGBA pick buffer.
 * @param {Float32Array} cand Candidate RGBA pick buffer.
 * @returns {{ pixels: number, hits: number, mismatches: number, mismatchFraction: number,
 *   nodeMismatches: number, nodeMismatchFraction: number }}
 */
export function scorePickBuffers(base, cand) {
  if (base.length !== cand.length || base.length % 4 !== 0) {
    throw new Error(`pick buffer sizes differ or are not RGBA: ${base.length} vs ${cand.length}`);
  }
  const pixels = base.length / 4;
  let mismatches = 0;
  let nodeMismatches = 0;
  let hits = 0;
  for (let p = 0; p < pixels; p++) {
    let hit = false;
    let differs = false;
    for (let c = 0; c < 4; c++) {
      if (base[p * 4 + c] !== 0) hit = true;
      if (!Object.is(base[p * 4 + c], cand[p * 4 + c])) differs = true;
    }
    if (hit) hits++;
    if (differs) mismatches++;
    if (!Object.is(base[p * 4], cand[p * 4])) nodeMismatches++;
  }
  const frac = (n) => (pixels > 0 ? n / pixels : 0);
  return {
    pixels,
    hits,
    mismatches,
    mismatchFraction: frac(mismatches),
    nodeMismatches,
    nodeMismatchFraction: frac(nodeMismatches),
  };
}

/**
 * Tolerances per verdict class (see RENDER_GATE.md).
 *
 * `IDENTICAL` admits no difference at all: 0 ULP16 in every buffer, 0 pick
 * mismatches.
 *
 * `ULP` is judged on WHAT A VIEWER COULD SEE, not on per-pixel counts: the
 * frame's energy change and the p99.9 change of 4x4 box-filtered tiles
 * (`scoreBlocks`). Per-pixel flips overstate sub-pixel jitter — reading
 * projectionMatrix[1][1] in a line shader changed ANGLE/Metal codegen enough
 * to flip 2.7e-3 of a dense line scene's pixels while its energy moved 1e-7,
 * exactly like a 1-ulp nudge. The limits are CALIBRATED on arms that scale a
 * derived quantity (point size factor, line width scale, splat focal length)
 * by (1 + eps), plus the real commits, on points / lines / splats in every
 * blending mode, both backends (Apple M4 Max, 2026-09-25). Worst case per view:
 *
 *   arm                        |energy|      tile p99.9    (tile max overlaps)
 *   rounding: 1/8 ulp32, commits   <= 2.2e-6     <= 9.4e-5
 *   real error: eps = 1e-4         >= 4.4e-6 *   >= 1.7e-4
 *
 *   * max / normal splats at the near pose barely change energy (those modes
 *     do not add light); tile p99.9 catches them at >= 2.2e-3.
 *
 * So `ULP` fails a view on |energy| > 1e-5 OR tile p99.9 > 1.3e-4 (margins
 * ~1.4x on the tile metric, ~5x on energy where energy separates at all), and
 * on pick NODE mismatches > 3e-3 (element-level pick ties are reported only).
 * Per-pixel statistics stay in the report for reading.
 */
export const CLASS_LIMITS = Object.freeze({
  IDENTICAL: { exact: true },
  ULP: { energyRel: 1e-5, blockP999Rel: 1.3e-4, pickNodeMismatchFraction: 3e-3 },
});

function exactFailures(name, score, failures) {
  if (score && score.differing > 0) {
    failures.push(
      `${name}: ${score.differing} px differ (drift ${score.maxDriftUlp.toFixed(1)} ULP16, ${score.flips} flips)`
    );
  }
}

/**
 * Judge one view against a verdict class.
 *
 * @param {'IDENTICAL'|'ULP'} cls Declared class.
 * @param {ReturnType<typeof scoreFloatBuffers>} hdr Raw-scene HDR score.
 * @param {ReturnType<typeof scoreFloatBuffers>|null} ldr Visible-LDR score, when captured.
 * @param {ReturnType<typeof scorePickBuffers>|null} pick Pick score, when captured.
 * @param {ReturnType<typeof scoreBlocks>|null} blocks Scale-aware HDR score (required for ULP).
 * @returns {{ pass: boolean, failures: string[] }} Verdict and the violated limits.
 */
export function judge(cls, hdr, ldr, pick, blocks = null) {
  const limits = CLASS_LIMITS[cls];
  if (!limits) throw new Error(`unknown verdict class ${cls}`);
  const failures = [];
  if (limits.exact) {
    exactFailures('hdr', hdr, failures);
    exactFailures('ldr', ldr, failures);
    if (pick && pick.mismatches > 0) failures.push(`pick: ${pick.mismatches} px differ`);
    return { pass: failures.length === 0, failures };
  }
  if (!blocks) throw new Error('the ULP class needs the scale-aware block score');
  if (Math.abs(blocks.energyRel) > limits.energyRel) {
    failures.push(`energy ${blocks.energyRel.toExponential(2)} beyond ±${limits.energyRel}`);
  }
  if (blocks.blockP999Rel > limits.blockP999Rel) {
    failures.push(`tile p99.9 ${blocks.blockP999Rel.toExponential(2)} > ${limits.blockP999Rel}`);
  }
  if (pick && pick.nodeMismatchFraction > limits.pickNodeMismatchFraction) {
    failures.push(
      `pick node mismatches ${pick.nodeMismatches} (${pick.nodeMismatchFraction.toExponential(2)}) > ${limits.pickNodeMismatchFraction}`
    );
  }
  return { pass: failures.length === 0, failures };
}
