/**
 * Unit tests for the line-join artifact metrics (issue #790 acceptance
 * harness — `src/tests/helpers/line-join-metrics.ts`).
 *
 * Every case builds a synthetic luminance image by hand so the expected
 * answer is known exactly, including the two edge rows the background mask
 * legitimately drops. The fourth case is the load-bearing one: it proves the
 * two metrics are genuinely complementary, because the #780 bead-chain
 * failure it models is invisible to the local-median metric by construction.
 */
import { describe, it, expect } from 'vitest';

import {
  measureAxialFlux,
  measureLocalMedianOutliers,
  rgbaToLuminance,
  type PixelRect,
} from '../../helpers/line-join-metrics';

/** Image geometry shared by the band fixtures below. */
const BAND_TOP = 5;
const BAND_HEIGHT = 11;
const BAND_BOTTOM = BAND_TOP + BAND_HEIGHT - 1; // inclusive
const BAND_VALUE = 200;

/**
 * Build a black image with one horizontal band whose brightness is a
 * function of the column, i.e. a perfectly straight "tube" running along the
 * image x axis.
 */
function makeBand(
  width: number,
  height: number,
  valueAt: (x: number) => number = () => BAND_VALUE
): Float32Array {
  const image = new Float32Array(width * height);
  for (let y = BAND_TOP; y <= BAND_BOTTOM; y++) {
    for (let x = 0; x < width; x++) {
      image[y * width + x] = valueAt(x);
    }
  }
  return image;
}

/** Overwrite one full-height column of the band with `value`. */
function punchColumn(
  image: Float32Array,
  width: number,
  column: number,
  value: number
): Float32Array {
  for (let y = BAND_TOP; y <= BAND_BOTTOM; y++) {
    image[y * width + column] = value;
  }
  return image;
}

const WHOLE = (width: number, height: number): PixelRect => ({ x: 0, y: 0, width, height });

describe('rgbaToLuminance', () => {
  it('applies the Rec.709 weights per pixel', () => {
    const rgba = new Uint8ClampedArray([
      255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ]);
    const lum = rgbaToLuminance(rgba, 4, 1);
    expect(lum).toHaveLength(4);
    expect(lum[0]).toBeCloseTo(0.2126 * 255, 3);
    expect(lum[1]).toBeCloseTo(0.7152 * 255, 3);
    expect(lum[2]).toBeCloseTo(0.0722 * 255, 3);
    expect(lum[3]).toBeCloseTo(255, 3);
  });

  it('rejects a buffer too short for the declared size', () => {
    expect(() => rgbaToLuminance(new Uint8ClampedArray(8), 4, 1)).toThrow(/needed for 4x1/);
  });
});

describe('measureLocalMedianOutliers', () => {
  it('reports no outliers on a clean uniform band', () => {
    const width = 40;
    const height = 20;
    const image = makeBand(width, height);

    const m = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));

    // Only the band itself clears the background mask: 11 rows x 40 columns.
    expect(m.insidePixels).toBe(width * BAND_HEIGHT);
    expect(m.darkOutliers).toBe(0);
    expect(m.brightOutliers).toBe(0);
    expect(m.worstDeficit).toBe(0);
    expect(m.worstExcess).toBe(0);
  });

  it('counts a one-pixel-wide black tick through the band (the #790 wedge in miniature)', () => {
    const width = 40;
    const height = 20;
    const tick = 20;
    const image = punchColumn(makeBand(width, height), width, tick, 0);

    const m = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));

    // The tick is 11 pixels tall, but at the band's first and last row a
    // 5x5 window straddles the background: 10 background + 3 tick = 13 of
    // 25 samples are zero, so the local median is 0 and those two pixels
    // fall out of the mask along with their four horizontal neighbours.
    expect(m.darkOutliers).toBe(BAND_HEIGHT - 2);
    expect(m.worstDeficit).toBeCloseTo(BAND_VALUE, 6);
    expect(m.brightOutliers).toBe(0);
    expect(m.insidePixels).toBe(width * BAND_HEIGHT - 10);
    expect(m.darkFraction).toBeCloseTo((BAND_HEIGHT - 2) / m.insidePixels, 10);
  });

  it('counts a one-pixel-wide overbright tick through the band', () => {
    const width = 40;
    const height = 20;
    const tick = 20;
    const overbright = 255;
    const image = punchColumn(makeBand(width, height), width, tick, overbright);

    const m = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));

    // A bright tick never pushes the local median below the cutoff, so every
    // one of the band's 11 rows is measured.
    expect(m.brightOutliers).toBe(BAND_HEIGHT);
    expect(m.worstExcess).toBeCloseTo(overbright - BAND_VALUE, 6);
    expect(m.darkOutliers).toBe(0);
    expect(m.insidePixels).toBe(width * BAND_HEIGHT);
  });

  it('rejects an even window', () => {
    const image = makeBand(20, 20);
    expect(() => measureLocalMedianOutliers(image, 20, 20, WHOLE(20, 20), { window: 4 })).toThrow(
      /odd positive integer/
    );
  });
});

describe('measureAxialFlux', () => {
  it('is dead flat on a clean uniform band', () => {
    const width = 40;
    const height = 20;
    const image = makeBand(width, height);

    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    expect(a.samples).toBe(width);
    expect(a.insidePixels).toBe(width * BAND_HEIGHT);
    expect(a.medianFlux).toBeCloseTo(BAND_HEIGHT * BAND_VALUE, 4);
    expect(a.p05).toBeCloseTo(1, 10);
    expect(a.p50).toBeCloseTo(1, 10);
    expect(a.p95).toBeCloseTo(1, 10);
    expect(a.min).toBeCloseTo(1, 10);
  });

  it('sums rows instead of columns for a vertical tube', () => {
    // Transpose the clean band: a vertical tube measured along 'y' must give
    // the same flat profile as the horizontal one measured along 'x'.
    const width = 20;
    const height = 40;
    const image = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = BAND_TOP; x <= BAND_BOTTOM; x++) {
        image[y * width + x] = BAND_VALUE;
      }
    }

    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'y');

    expect(a.samples).toBe(height);
    expect(a.insidePixels).toBe(height * BAND_HEIGHT);
    expect(a.min).toBeCloseTo(1, 10);
    expect(a.p95).toBeCloseTo(1, 10);
  });
});

describe('the two metrics are complementary (#780 vs #790)', () => {
  it('a broad smooth 50% flux dip scores zero median outliers but collapses the axial p05', () => {
    // A tube whose flux swings smoothly between 100% and 50% of peak with a
    // 40-pixel period. This is the shape of the #780 bead chain: every
    // interior joint dimmed over a width-sized ramp. A local median tracks a
    // ramp this gentle exactly, so the tick detector is blind to it — which
    // is precisely why the axial flux profile exists.
    const width = 120;
    const height = 20;
    const period = 40;
    const image = makeBand(
      width,
      height,
      (x) => BAND_VALUE * (0.75 + 0.25 * Math.cos((2 * Math.PI * x) / period))
    );

    const m = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));
    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    // Half one: the median metric sees nothing at all.
    expect(m.insidePixels).toBe(width * BAND_HEIGHT);
    expect(m.darkOutliers).toBe(0);
    expect(m.brightOutliers).toBe(0);

    // Half two: the flux profile sees it plainly. Peak-to-trough is 2:1 and
    // the profile median sits at 0.75 of peak, so the trough normalises to
    // ~0.67 — far below the 1.0 a healthy tube holds.
    expect(a.samples).toBe(width);
    expect(a.p50).toBeCloseTo(1, 10);
    expect(a.p05).toBeLessThan(0.8);
    expect(a.min).toBeLessThan(0.72);
    expect(a.min).toBeGreaterThan(0.6);
  });
});

describe('degenerate regions are reported, not silently zeroed', () => {
  const width = 40;
  const height = 20;
  const image = makeBand(width, height);

  it('an empty region yields insidePixels 0', () => {
    const region: PixelRect = { x: 5, y: 5, width: 0, height: 0 };
    expect(measureLocalMedianOutliers(image, width, height, region).insidePixels).toBe(0);
    const a = measureAxialFlux(image, width, height, region, 'x');
    expect(a.insidePixels).toBe(0);
    expect(a.samples).toBe(0);
    expect(a.p05).toBe(0);
  });

  it('a region wholly outside the image yields insidePixels 0', () => {
    const region: PixelRect = { x: width + 10, y: 0, width: 10, height: height };
    expect(measureLocalMedianOutliers(image, width, height, region).insidePixels).toBe(0);
    expect(measureAxialFlux(image, width, height, region, 'x').samples).toBe(0);
  });

  it('a background-only region yields insidePixels 0 rather than a clean verdict', () => {
    // Rows 0..2 are pure background, and far enough from the band that no
    // 5x5 window reaches it.
    const region: PixelRect = { x: 0, y: 0, width, height: 3 };
    const m = measureLocalMedianOutliers(image, width, height, region);
    expect(m.insidePixels).toBe(0);
    expect(m.darkOutliers).toBe(0);
    const a = measureAxialFlux(image, width, height, region, 'x');
    expect(a.insidePixels).toBe(0);
    expect(a.min).toBe(0);
  });

  it('a region clipped by the image edge measures only the visible part', () => {
    // Starts 10 columns from the right edge and asks for 50 — 10 survive.
    const region: PixelRect = { x: width - 10, y: 0, width: 50, height };
    const m = measureLocalMedianOutliers(image, width, height, region);
    expect(m.insidePixels).toBe(10 * BAND_HEIGHT);
    expect(m.darkOutliers).toBe(0);
    const a = measureAxialFlux(image, width, height, region, 'x');
    expect(a.samples).toBe(10);
    expect(a.min).toBeCloseTo(1, 10);
  });
});
