/**
 * Gallery auto-exposure policy — unit tests against a SIMULATED capture.
 *
 * The real harness needs a dataset, a dev server and a GPU, so the policy is
 * exercised here against a synthetic "subject": a linear lit-luminance
 * distribution pushed through a monotone compressive tone curve. That is enough
 * to reproduce the failure the flat-subject pass fixes — a flat shaded mesh
 * whose lit histogram is only a few hundredths wide, parked deep in the tone
 * curve's shoulder, which the tail-based clip guard structurally cannot see.
 *
 * The simulation is NOT a model of ACES. To keep the assertions from encoding
 * one particular curve's geometry, the flat-subject case is run through TWO curves
 * (each with its profile calibrated through its own inverse) and only
 * curve-independent claims are asserted: the median lands on TARGET_MID, and
 * exposure moves down substantially. The absolute size of that move is a
 * property of the curve, not of the algorithm, so it is only lower-bounded.
 *
 * The pre-fix algorithm (p99 pass + clip guard, no mid-tone phase) is
 * reimplemented here as `referenceAutoExposure` so the assertions are genuine
 * regression guards: the emissive profile must land on EXACTLY the pre-fix
 * exposure with the same applies/measurements, and the flat-subject profile must
 * land somewhere clearly else.
 */

import { describe, it, expect } from 'vitest';
import {
  computeAutoExposure,
  type ExposureIO,
  type LumaStats,
  AUTO_EXPOSURE_ITERS,
  BG_LUMA_MAX,
  CLIP_FRAC_MAX,
  CLIP_GUARD_STEP,
  CLIP_LUMA,
  CLIP_SAT_MAX,
  EXPOSURE_MAX,
  EXPOSURE_MIN,
  HI_PERCENTILE,
  LIT_THRESHOLD,
  MID_EXPOSURE_ITERS,
  MID_EXPOSURE_TOL,
  NARROW_SPREAD_MAX,
  TARGET_HI,
  TARGET_MID,
} from '../screenshots/exposure-policy';

// ---------------------------------------------------------------------------
// Tone curves
// ---------------------------------------------------------------------------

interface ToneCurve {
  name: string;
  /** Linear luminance → display value in [0,1). Monotone and compressive. */
  map(x: number): number;
  /** Inverse of `map`, used to calibrate a profile to a target display histogram. */
  inv(d: number): number;
}

/** `1 − (1+x)⁻²` — unit log-log slope at the bottom, a fairly hard shoulder. */
const SHOULDER: ToneCurve = {
  name: '1-(1+x)^-2',
  map: (x) => 1 - 1 / ((1 + x) * (1 + x)),
  inv: (d) => Math.pow(1 - d, -0.5) - 1,
};

/** `x/(1+x)` — the classic Reinhard curve; noticeably more compressive up high. */
const REINHARD: ToneCurve = {
  name: 'x/(1+x)',
  map: (x) => x / (1 + x),
  inv: (d) => d / (1 - d),
};

const CURVES = [SHOULDER, REINHARD];

/**
 * A crude monotone stand-in for "the shoulder washes colour out": saturation
 * falls linearly to 0 as the display value approaches white. This is NOT a fit
 * to the values in the issue (it would predict 0.85 → 0.11 and 0.70 → 0.09,
 * where 0.28 and 0.14 were measured; no purely multiplicative model can even
 * reproduce that pair's 2.0 ratio). It exists solely so the blown-pixel check
 * has a saturation term to evaluate.
 */
function displaySaturation(baseSat: number, display: number): number {
  return baseSat * (1 - display);
}

// ---------------------------------------------------------------------------
// Simulated capture
// ---------------------------------------------------------------------------

interface SubjectProfile {
  curve: ToneCurve;
  /** Linear (pre-exposure, pre-tone-map) luminance of each lit sample. */
  linear: number[];
  /** Fraction of the whole frame the subject occupies. */
  litFraction: number;
  /** Authored saturation of the subject's colour. */
  saturation: number;
  /** Linear luminance of the background. */
  background: number;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const rank = Math.min(sorted.length, Math.max(1, Math.ceil(p * sorted.length)));
  return sorted[rank - 1];
}

/** Render a profile at `stops` and measure it exactly as the harness would. */
function capture(profile: SubjectProfile, stops: number): LumaStats {
  const gain = Math.pow(2, stops);
  const displayed = profile.linear.map((v) => profile.curve.map(v * gain));
  const lit = displayed.filter((v) => v > LIT_THRESHOLD).sort((a, b) => a - b);
  // Frame p10 = the background level: every gallery subject covers well under
  // 90% of the frame, so the 10th percentile of the whole frame is background.
  const bgLuma = profile.curve.map(profile.background * gain);
  if (lit.length === 0) {
    return { hiLuma: 0, loLuma: 0, midLuma: 0, litFraction: 0, clippedFrac: 0, bgLuma };
  }
  const blown = lit.filter(
    (v) => v > CLIP_LUMA && displaySaturation(profile.saturation, v) < CLIP_SAT_MAX
  );
  return {
    hiLuma: percentile(lit, HI_PERCENTILE),
    loLuma: percentile(lit, 0.1),
    midLuma: percentile(lit, 0.5),
    litFraction: (profile.litFraction * lit.length) / profile.linear.length,
    clippedFrac: blown.length / lit.length,
    bgLuma,
  };
}

interface RecordingIO extends ExposureIO {
  applied: number[];
  measures: number;
}

/** An `ExposureIO` over a simulated subject, recording every applied exposure. */
function makeIO(profile: SubjectProfile): RecordingIO {
  let current = 0;
  const io: RecordingIO = {
    applied: [],
    measures: 0,
    async apply(stops: number) {
      current = stops;
      io.applied.push(stops);
    },
    async measure() {
      io.measures++;
      return capture(profile, current);
    },
  };
  return io;
}

// ---------------------------------------------------------------------------
// Pre-fix reference algorithm (p99 pass + clip guard only, no mid-tone phase)
// ---------------------------------------------------------------------------

// Freeze the historical flat guard budget now that production derives its cap.
const REFERENCE_CLIP_GUARD_ITERS = 8;

async function referenceAutoExposure(io: ExposureIO): Promise<number> {
  let stops = 1.0;
  await io.apply(stops);
  for (let iter = 0; iter < AUTO_EXPOSURE_ITERS; iter++) {
    const { hiLuma, litFraction } = await io.measure();
    if (litFraction < 0.0005 || hiLuma <= 0) {
      stops = Math.min(EXPOSURE_MAX, stops + 2);
      await io.apply(stops);
      continue;
    }
    const correction = Math.log2(TARGET_HI / hiLuma);
    const next = Math.max(EXPOSURE_MIN, Math.min(EXPOSURE_MAX, stops + correction));
    if (Math.abs(next - stops) < 0.1) {
      stops = next;
      break;
    }
    stops = next;
    await io.apply(stops);
  }
  for (let iter = 0; iter < REFERENCE_CLIP_GUARD_ITERS; iter++) {
    const { clippedFrac, bgLuma } = await io.measure();
    const tooBlown = clippedFrac > CLIP_FRAC_MAX;
    const bgTooBright = bgLuma > BG_LUMA_MAX;
    if ((!tooBlown && !bgTooBright) || stops <= EXPOSURE_MIN) break;
    stops = Math.max(EXPOSURE_MIN, stops - CLIP_GUARD_STEP);
    await io.apply(stops);
  }
  return stops;
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/**
 * Synthetic flat shaded surface. This deliberately preserves the narrow
 * pre-fix failure distribution so the policy branch remains exercised across
 * two tone curves. The current offset-key `mesh_isosurface_cells3d` capture
 * instead measures p10 0.378 / p50 0.506 / p99 0.597 (spread 0.219) after the
 * median pass; its p50 confirms the same policy still lands on TARGET_MID.
 */
const flatShaded = (curve: ToneCurve): SubjectProfile => ({
  curve,
  linear: Array.from({ length: 401 }, (_, i) => curve.inv(0.835 + (0.07 * i) / 400) / 2),
  litFraction: 0.35,
  saturation: 0.85,
  background: 0.0,
});

/**
 * EMISSIVE additive cloud: a power-law distribution — a large population of dim
 * halo pixels just above LIT_THRESHOLD and a thin bright core. This is the EASY
 * side of the gate (it measures a spread of ~0.77 under SHOULDER, ~6× the
 * threshold); the tightest REAL emissive tile measured on a proxy re-exposure
 * was ~0.167, only ~40% above it. See NARROW_SPREAD_MAX.
 */
const emissive = (curve: ToneCurve): SubjectProfile => ({
  curve,
  linear: Array.from({ length: 2000 }, (_, i) => 0.02 * Math.pow((i + 1) / 2000, -0.8)),
  litFraction: 0.2,
  saturation: 0.6,
  background: 0.0,
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('gallery auto-exposure policy', () => {
  describe('flat shaded subject', () => {
    it('the pre-fix algorithm parks the whole subject in the tone-curve shoulder', async () => {
      const profile = flatShaded(SHOULDER);
      const shot = capture(profile, await referenceAutoExposure(makeIO(profile)));
      expect(shot.midLuma).toBeGreaterThan(0.8);
      expect(shot.hiLuma - shot.loLuma).toBeLessThan(NARROW_SPREAD_MAX);
    });

    it('documents the profile: the failure mode leaves no tail for the clip guard', async () => {
      // NOTE: this asserts properties of the PROFILE, not of the guard — the
      // background is black by construction and the brightest display value is
      // below CLIP_LUMA by construction. That is exactly the point: the guard
      // is a tail test, and this failure mode has no tail, so no plausible
      // tuning of CLIP_FRAC_MAX / BG_LUMA_MAX would have caught it.
      const profile = flatShaded(SHOULDER);
      const shot = capture(profile, await referenceAutoExposure(makeIO(profile)));
      expect(shot.hiLuma).toBeLessThan(CLIP_LUMA);
      expect(shot.clippedFrac).toBeLessThan(CLIP_FRAC_MAX);
      expect(shot.bgLuma).toBeLessThanOrEqual(BG_LUMA_MAX);
    });

    for (const curve of CURVES) {
      it(`lands the lit median on TARGET_MID under ${curve.name}`, async () => {
        const profile = flatShaded(curve);
        const io = makeIO(profile);
        const { stops, flatSubject } = await computeAutoExposure(io);
        const reference = await referenceAutoExposure(makeIO(profile));

        expect(flatSubject).toBe(true);
        // The curve-independent claim: the median actually reaches the target.
        expect(Math.abs(capture(profile, stops).midLuma - TARGET_MID)).toBeLessThan(0.1);
        // ...and that same claim genuinely FAILS for the pre-fix algorithm.
        expect(Math.abs(capture(profile, reference).midLuma - TARGET_MID)).toBeGreaterThan(0.1);
        // Exposure moves down substantially. Lower bound only: HOW far is a
        // property of the curve's compressiveness (2.06 stops under SHOULDER,
        // 2.70 under REINHARD), not of the policy.
        expect(reference - stops).toBeGreaterThan(1.5);

        // Pin the two tuning knobs. The residual is inside the pass's own
        // early-exit tolerance...
        const residual = Math.abs(Math.log2(TARGET_MID / capture(profile, stops).midLuma));
        expect(residual).toBeLessThan(MID_EXPOSURE_TOL);
        // ...and it got there with iterations to spare (phase 1 and the guard
        // apply nothing for this profile, so every apply after the initial one
        // is a mid-tone step). 5 of 10 used under SHOULDER, 7 under REINHARD —
        // real headroom, but see MID_EXPOSURE_ITERS: it is only one iteration
        // over the measured worst case.
        const midApplies = io.applied.length - 1;
        expect(midApplies).toBeGreaterThan(0);
        expect(midApplies).toBeLessThan(MID_EXPOSURE_ITERS);
      });
    }
  });

  describe('emissive (wide-histogram) subject', () => {
    for (const curve of CURVES) {
      it(`chooses EXACTLY the pre-fix exposure under ${curve.name}`, async () => {
        const profile = emissive(curve);
        const { stops, flatSubject, guardExhausted } = await computeAutoExposure(makeIO(profile));
        expect(stops).toBe(await referenceAutoExposure(makeIO(profile)));
        expect(flatSubject).toBe(false);
        expect(guardExhausted).toBe(false);
      });
    }

    it('performs the same applies and measurements as the pre-fix algorithm', async () => {
      const fixedIO = makeIO(emissive(SHOULDER));
      await computeAutoExposure(fixedIO);
      const refIO = makeIO(emissive(SHOULDER));
      await referenceAutoExposure(refIO);
      expect(fixedIO.applied).toEqual(refIO.applied);
      expect(fixedIO.measures).toBe(refIO.measures);
    });

    it('preserves a converging trajectory that takes guard steps', async () => {
      const guardedIO = (): RecordingIO => {
        let current = 0;
        const io: RecordingIO = {
          applied: [],
          measures: 0,
          async apply(stops) {
            current = stops;
            io.applied.push(stops);
          },
          async measure(): Promise<LumaStats> {
            io.measures++;
            const guarded = current > -1.5;
            return {
              hiLuma: TARGET_HI,
              loLuma: 0.2,
              midLuma: 0.6,
              litFraction: 0.5,
              clippedFrac: guarded ? 1 : 0,
              bgLuma: guarded ? 0.5 : 0,
            };
          },
        };
        return io;
      };

      const fixedIO = guardedIO();
      const result = await computeAutoExposure(fixedIO);
      const refIO = guardedIO();
      const reference = await referenceAutoExposure(refIO);
      expect(result.stops).toBe(reference);
      expect(result.guardExhausted).toBe(false);
      expect(fixedIO.applied).toEqual(refIO.applied);
      expect(fixedIO.measures).toBe(refIO.measures);
      expect(fixedIO.applied).toEqual([1, 0.5, 0, -0.5, -1, -1.5]);
      expect(fixedIO.measures).toBe(7);
    });

    it('stays well clear of the narrow-spread gate', async () => {
      const profile = emissive(SHOULDER);
      const shot = capture(profile, await referenceAutoExposure(makeIO(profile)));
      // Measures ≈0.77 here, ≈6× NARROW_SPREAD_MAX. A real tile can be much
      // closer to the gate (≈0.167 was measured) — this profile is the easy case.
      expect(shot.hiLuma - shot.loLuma).toBeGreaterThan(0.5);
    });
  });

  describe('spread criterion boundary', () => {
    /**
     * A stub whose lit histogram has a FIXED width regardless of exposure, so
     * only the spread test decides whether the mid-tone pass runs.
     */
    const fixedSpread = (spread: number): ExposureIO => ({
      async apply() {},
      async measure(): Promise<LumaStats> {
        return {
          hiLuma: 0.87,
          loLuma: 0.87 - spread,
          midLuma: 0.87 - spread / 2,
          litFraction: 0.3,
          clippedFrac: 0,
          bgLuma: 0,
        };
      },
    });

    it('fires just below NARROW_SPREAD_MAX and not just above it', async () => {
      const p99Only = 1.0 + Math.log2(TARGET_HI / 0.87);
      const narrow = await computeAutoExposure(fixedSpread(NARROW_SPREAD_MAX - 0.01));
      const wide = await computeAutoExposure(fixedSpread(NARROW_SPREAD_MAX + 0.01));
      expect(wide.flatSubject).toBe(false);
      expect(wide.stops).toBeCloseTo(p99Only, 6); // untouched by the new phase
      expect(narrow.flatSubject).toBe(true);
      expect(narrow.stops).toBeLessThan(wide.stops);
    });
  });

  describe('degenerate subjects', () => {
    it('terminates and returns a clamped value when nothing is lit', async () => {
      const io = makeIO({
        curve: SHOULDER,
        linear: [0],
        litFraction: 0,
        saturation: 0,
        background: 0,
      });
      const { stops, flatSubject } = await computeAutoExposure(io);
      expect(Number.isFinite(stops)).toBe(true);
      expect(stops).toBeGreaterThanOrEqual(EXPOSURE_MIN);
      expect(stops).toBeLessThanOrEqual(EXPOSURE_MAX);
      expect(flatSubject).toBe(false); // gated on there being lit pixels
      // Phase 1 brightens by 2 stops per empty measurement, then both later
      // phases no-op.
      expect(stops).toBe(1.0 + 2 * AUTO_EXPOSURE_ITERS);
    });

    it('continues past the old guard budget until the frame satisfies the predicate', async () => {
      let applied = 0;
      const measured: number[] = [];
      const io: ExposureIO = {
        async apply(stops) {
          applied = stops;
        },
        async measure(): Promise<LumaStats> {
          measured.push(applied);
          const guarded = applied > -5;
          return {
            hiLuma: TARGET_HI,
            loLuma: 0.2,
            midLuma: 0.6,
            litFraction: 0.5,
            clippedFrac: guarded ? 1 : 0,
            bgLuma: guarded ? 0.5 : 0,
          };
        },
      };

      const { stops, guardExhausted } = await computeAutoExposure(io);
      expect(stops).toBe(-5);
      expect(measured.at(-1)).toBe(-5);
      expect(guardExhausted).toBe(false);
    });

    it('measures the floor and reports when a permanently blown subject exhausts the guard', async () => {
      // Permanently blown + grey background, and a WIDE histogram so the
      // mid-tone pass stays out of it. The guard must use its full available
      // range, then measure the applied floor before declaring exhaustion.
      let applied = 0;
      const measured: number[] = [];
      const io: ExposureIO = {
        async apply(stops) {
          applied = stops;
        },
        async measure(): Promise<LumaStats> {
          measured.push(applied);
          return {
            hiLuma: 0.99,
            loLuma: 0.2,
            midLuma: 0.9,
            litFraction: 0.5,
            clippedFrac: 1,
            bgLuma: 0.5,
          };
        },
      };
      const { stops, flatSubject, guardExhausted } = await computeAutoExposure(io);
      expect(flatSubject).toBe(false);
      expect(stops).toBe(-10);
      expect(measured.at(-1)).toBe(-10);
      expect(guardExhausted).toBe(true);
    });

    it('clamps a fractional guard trajectory and measures the floor', async () => {
      let applied = 0;
      let measures = 0;
      const measured: number[] = [];
      const io: ExposureIO = {
        async apply(stops) {
          applied = stops;
        },
        async measure(): Promise<LumaStats> {
          measured.push(applied);
          const hiLuma = measures++ === 0 ? TARGET_HI * Math.pow(2, 0.45) : TARGET_HI;
          return {
            hiLuma,
            loLuma: 0.2,
            midLuma: 0.6,
            litFraction: 0.5,
            clippedFrac: 1,
            bgLuma: 0.5,
          };
        },
      };

      const { stops, guardExhausted } = await computeAutoExposure(io);
      expect(stops).toBe(EXPOSURE_MIN);
      expect(measured.at(-1)).toBe(EXPOSURE_MIN);
      expect(guardExhausted).toBe(true);
    });

    it('uses the guard to reach EXPOSURE_MIN after the mid-tone budget is exhausted', async () => {
      // Narrow AND pinned near white, and the subject never responds to
      // exposure, so the mid-tone pass spends its whole budget stepping down.
      let applied = 0;
      const measured: number[] = [];
      const io: ExposureIO = {
        async apply(stops) {
          applied = stops;
        },
        async measure(): Promise<LumaStats> {
          measured.push(applied);
          return {
            hiLuma: 0.99,
            loLuma: 0.97,
            midLuma: 0.99,
            litFraction: 0.5,
            clippedFrac: 1,
            bgLuma: 0.5,
          };
        },
      };
      // The widened floor is below what phase 2 can reach with valid 0..1 luma,
      // so the median pass must exhaust its budget above the floor and phase 3
      // must finish the descent.
      const afterPhase1 = 1.0 + AUTO_EXPOSURE_ITERS * Math.log2(TARGET_HI / 0.99);
      const unclamped = afterPhase1 + MID_EXPOSURE_ITERS * Math.log2(TARGET_MID / 0.99);
      expect(unclamped).toBeGreaterThan(EXPOSURE_MIN);

      const { stops, flatSubject, guardExhausted } = await computeAutoExposure(io);
      expect(flatSubject).toBe(true);
      expect(stops).toBe(EXPOSURE_MIN);
      expect(measured.at(-1)).toBe(EXPOSURE_MIN);
      expect(guardExhausted).toBe(true);
    });
  });
});
