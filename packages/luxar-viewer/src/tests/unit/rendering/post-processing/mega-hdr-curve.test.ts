import * as fc from 'fast-check';
import { describe, expect, it } from 'vitest';

import { HDR_SLOPE_STEP, hdrKnee } from '../../../../rendering/post-processing/mega/shader.tsl';
import type { LuxarToneMappingMode } from '../../../../rendering/post-processing/tone-mapping';

// Grey-axis reductions of Three r185's ToneMappingFunctions. The shader calls
// those functions through applyToneMapping; the WGSL test checks each call is
// emitted. The ACES matrices preserve grey; AgX's final red row sums to 1.0001.
function sdrGrey(mode: LuxarToneMappingMode, level: number): number {
  switch (mode) {
    case 1:
      return Math.min(1, Math.max(0, level));
    case 2:
      return Math.min(1, level / (level + 1));
    case 3: {
      const x = Math.max(0, level - 0.004);
      return Math.pow((x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06), 2.2);
    }
    case 4: {
      const x = level / 0.6;
      return Math.min(
        1,
        Math.max(
          0,
          (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * (x + 0.432951)) + 0.238081)
        )
      );
    }
    case 5: {
      const ev = Math.log2(Math.max(level, 1e-10));
      const x = Math.min(1, Math.max(0, (ev + 12.47393) / (4.026069 + 12.47393)));
      const x2 = x * x;
      const x4 = x2 * x2;
      const contrast =
        15.5 * x4 * x2 -
        40.14 * x4 * x +
        31.96 * x4 -
        6.868 * x2 * x +
        0.4298 * x2 +
        0.1191 * x -
        0.00232;
      // The final Rec2020 → sRGB red-row sum in Three is 1.0001.
      return Math.min(1, Math.max(0, 1.0001 * Math.pow(Math.max(0, contrast), 2.2)));
    }
    case 6: {
      const offset = level < 0.08 ? level - 6.25 * level * level : 0.04;
      const x = level - offset;
      return x < 0.76 ? x : 1 - (0.24 * 0.24) / (x + 0.24 - 0.76);
    }
  }
}

function hdrGrey(mode: LuxarToneMappingMode, level: number, peak: number): number {
  if (mode === 1) return Math.min(peak, Math.max(0, level));
  const sdr = sdrGrey(mode, level);
  const knee = hdrKnee(mode);
  if (level <= knee || peak <= 1) return sdr;
  const yKnee = sdrGrey(mode, knee);
  const slope =
    (sdrGrey(mode, knee + HDR_SLOPE_STEP) - sdrGrey(mode, knee - HDR_SLOPE_STEP)) /
    (2 * HDR_SLOPE_STEP);
  const room = Math.max(peak - yKnee, 1e-4);
  const shoulder = peak - room * Math.exp((-slope * (level - knee)) / room);
  return Math.min((sdr * shoulder) / Math.max(sdrGrey(mode, level), 1e-6), peak);
}

const MODES: LuxarToneMappingMode[] = [1, 2, 3, 4, 5, 6];
const PEAKS = [1, 1.25, 2, 4, 8];

describe('HDR tone curve on grey', () => {
  it.each(MODES)('mode %i is exactly SDR below its knee and at peak 1', (mode) => {
    const knee = hdrKnee(mode);
    for (const level of [0, knee / 2, knee]) {
      for (const peak of PEAKS) expect(hdrGrey(mode, level, peak)).toBe(sdrGrey(mode, level));
    }
    if (mode !== 1) {
      for (const level of [knee + 0.01, 1, 4, 100]) {
        expect(hdrGrey(mode, level, 1)).toBe(sdrGrey(mode, level));
      }
    }
  });

  it.each(MODES.flatMap((mode) => PEAKS.map((peak) => [mode, peak] as const)))(
    'mode %i at peak %f stays monotone, >= SDR and <= peak',
    (mode, peak) => {
      fc.assert(
        fc.property(fc.double({ min: 0, max: 200, noNaN: true }), (level) => {
          const value = hdrGrey(mode, level, peak);
          expect(value).toBeGreaterThanOrEqual(sdrGrey(mode, level) - 1e-6);
          expect(value).toBeLessThanOrEqual(peak + 1e-6);
          expect(hdrGrey(mode, level + 1e-3, peak)).toBeGreaterThanOrEqual(value - 1e-6);
        }),
        { numRuns: 200 }
      );
    }
  );

  it.each([2, 3, 4, 5, 6] as LuxarToneMappingMode[])(
    'mode %i joins the SDR slope continuously at the knee',
    (mode) => {
      const knee = hdrKnee(mode);
      const h = 1e-5;
      for (const peak of PEAKS.slice(1)) {
        const left = (hdrGrey(mode, knee, peak) - hdrGrey(mode, knee - h, peak)) / h;
        const right = (hdrGrey(mode, knee + h, peak) - hdrGrey(mode, knee, peak)) / h;
        expect(Math.abs(left - right)).toBeLessThan(0.01);
      }
    }
  );

  it('Linear and None are exact peak clamps', () => {
    for (const peak of PEAKS) {
      for (const level of [0, 0.25, 1, peak, peak + 1, 100]) {
        expect(hdrGrey(1, level, peak)).toBe(Math.min(level, peak));
      }
    }
  });
});
