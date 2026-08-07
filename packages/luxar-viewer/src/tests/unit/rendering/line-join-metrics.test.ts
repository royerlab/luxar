/**
 * Unit tests for the line-join artifact metrics (issue #790 acceptance
 * harness — `src/tests/helpers/line-join-metrics.ts`).
 *
 * Every case builds a synthetic luminance image by hand so the expected
 * answer is known exactly, including the two edge rows the background mask
 * legitimately drops. Three groups are load-bearing beyond simple coverage:
 *
 *   - "the two metrics are complementary" proves the #780 bead-chain failure
 *     is invisible to the local-median metric by construction, which is why
 *     the axial flux metric exists at all.
 *   - "sensitivity envelope" pins the local-median metric's non-monotone
 *     response to defect width (1 px counted, 2 px counted, >= 3 px invisible
 *     because the defect poisons its own median). A change to the default
 *     window that silently moves that envelope must fail here, because
 *     nothing downstream would notice.
 *   - "percentile convention" pins which sample p05/p95/median actually pick,
 *     at a length where the nearest-sample and nearest-rank conventions
 *     disagree, and pins the half-empty boundary that follows from it.
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
  valueAt: (x: number) => number = () => BAND_VALUE,
  bandHeight: number = BAND_HEIGHT
): Float32Array {
  const image = new Float32Array(width * height);
  for (let y = BAND_TOP; y < BAND_TOP + bandHeight; y++) {
    for (let x = 0; x < width; x++) {
      image[y * width + x] = valueAt(x);
    }
  }
  return image;
}

/** Overwrite `count` adjacent full-height columns of the band with `value`. */
function punchColumns(
  image: Float32Array,
  width: number,
  column: number,
  value: number,
  count = 1,
  bandHeight: number = BAND_HEIGHT
): Float32Array {
  for (let y = BAND_TOP; y < BAND_TOP + bandHeight; y++) {
    for (let x = column; x < column + count; x++) {
      image[y * width + x] = value;
    }
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
    const image = punchColumns(makeBand(width, height), width, tick, 0);

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
    const image = punchColumns(makeBand(width, height), width, tick, overbright);

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
      /odd integer >= 3/
    );
  });

  it('rejects window 1, where the "median" is the pixel itself', () => {
    // With a 1x1 window every pixel equals its own median, so the metric
    // would report zero outliers on ANY image — a silent always-pass.
    const image = punchColumns(makeBand(40, 20), 40, 20, 0);
    expect(() => measureLocalMedianOutliers(image, 40, 20, WHOLE(40, 20), { window: 1 })).toThrow(
      /odd integer >= 3/
    );
  });
});

describe('sensitivity envelope of the local-median metric', () => {
  // A 21-pixel-tall band, matching the measured table in the module header.
  const ENVELOPE_BAND_HEIGHT = 21;
  const width = 60;
  const height = 40;

  /** Dark outliers for a `notchWidth`-pixel black notch across the band. */
  function darkOutliersForNotch(notchWidth: number): number {
    const image = punchColumns(
      makeBand(width, height, () => BAND_VALUE, ENVELOPE_BAND_HEIGHT),
      width,
      30,
      0,
      notchWidth,
      ENVELOPE_BAND_HEIGHT
    );
    return measureLocalMedianOutliers(image, width, height, WHOLE(width, height)).darkOutliers;
  }

  it('counts a 1 px notch', () => {
    // Every band row except the two at the band edges, where the window
    // straddles the background and the median drops out of the mask.
    expect(darkOutliersForNotch(1)).toBe(ENVELOPE_BAND_HEIGHT - 2);
  });

  it('counts a 2 px notch, at roughly double the 1 px score', () => {
    // Two columns x (21 - 4) rows: the mask loses two rows per band edge
    // now, because the notch contributes more zeros to the window.
    expect(darkOutliersForNotch(2)).toBe(2 * (ENVELOPE_BAND_HEIGHT - 4));
    expect(darkOutliersForNotch(2)).toBeGreaterThan(darkOutliersForNotch(1));
  });

  it('is blind to a 3 px notch, because the defect poisons its own median', () => {
    // 3 of the 5 window columns are dark, so the local median goes dark,
    // the mask rejects the pixel, and the defect scores zero. This is the
    // documented non-monotonicity, pinned here on purpose.
    expect(darkOutliersForNotch(3)).toBe(0);
  });

  it('is blind to 5 px and 7 px notches — much worse defects, still zero', () => {
    expect(darkOutliersForNotch(5)).toBe(0);
    expect(darkOutliersForNotch(7)).toBe(0);
    // Stated as an ordering so the intent survives a refactor: a wider
    // defect scoring less than a narrower one is the whole warning.
    expect(darkOutliersForNotch(7)).toBeLessThan(darkOutliersForNotch(1));
  });
});

describe('non-default option values', () => {
  const width = 40;
  const height = 20;

  it('honours a raised outlier threshold', () => {
    // A 30-unit dip clears the default threshold of 25 but not a raised 40.
    const image = punchColumns(makeBand(width, height), width, 20, BAND_VALUE - 30);
    const atDefault = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));
    const raised = measureLocalMedianOutliers(image, width, height, WHOLE(width, height), {
      threshold: 40,
    });
    expect(atDefault.darkOutliers).toBeGreaterThan(0);
    expect(raised.darkOutliers).toBe(0);
    expect(raised.insidePixels).toBe(atDefault.insidePixels);
  });

  it('honours a raised background cutoff', () => {
    // A dim band at luminance 20 is "inside" at the default cutoff of 12
    // and background at a cutoff of 50.
    const image = makeBand(width, height, () => 20);
    const atDefault = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));
    const raised = measureLocalMedianOutliers(image, width, height, WHOLE(width, height), {
      backgroundCutoff: 50,
    });
    expect(atDefault.insidePixels).toBe(width * BAND_HEIGHT);
    expect(raised.insidePixels).toBe(0);
  });

  it('honours a raised background cutoff in the flux profile too', () => {
    const image = makeBand(width, height, () => 20);
    expect(measureAxialFlux(image, width, height, WHOLE(width, height), 'x').samples).toBe(width);
    expect(
      measureAxialFlux(image, width, height, WHOLE(width, height), 'x', {
        backgroundCutoff: 50,
      }).samples
    ).toBe(0);
  });

  it('rejects a negative background cutoff in both metrics', () => {
    // A negative cutoff admits every background pixel as "inside" and
    // inflates the counts past any floor a caller could set.
    const image = makeBand(width, height);
    expect(() =>
      measureLocalMedianOutliers(image, width, height, WHOLE(width, height), {
        backgroundCutoff: -1,
      })
    ).toThrow(/backgroundCutoff must be >= 0/);
    expect(() =>
      measureAxialFlux(image, width, height, WHOLE(width, height), 'x', {
        backgroundCutoff: -1,
      })
    ).toThrow(/backgroundCutoff must be >= 0/);
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

  it('records an INTERIOR all-background column as a hole, never skips it', () => {
    // Punch four adjacent columns of the tube out entirely. If interior
    // dropouts were skipped instead of recorded, the profile would close up
    // and report a perfectly flat, perfectly clean tube.
    const width = 40;
    const height = 20;
    const image = punchColumns(makeBand(width, height), width, 18, 0, 4);

    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    expect(a.samples).toBe(width);
    expect(a.emptySamples).toBe(4);
    expect(a.min).toBe(0);
    expect(a.p05).toBe(0);
    expect(a.profile[18]).toBe(0);
    expect(a.profile[21]).toBe(0);
    expect(a.profile[17]).toBeCloseTo(1, 10);
  });

  it('does not report a tube with every other 8 px run missing as clean', () => {
    // The exact regression this behaviour exists for: half the line gone.
    // Skipping interior dropouts closed the profile back up into
    // samples=328, p05=p95=min=1 with zero median outliers, so every
    // assertion an acceptance spec can make passed on a broken renderer.
    const width = 128;
    const height = 20;
    const image = makeBand(width, height);
    for (let x = 8; x + 8 <= width; x += 16) punchColumns(image, width, x, 0, 8);

    const m = measureLocalMedianOutliers(image, width, height, WHOLE(width, height));
    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    // The median metric still sees nothing — an 8 px hole is far outside
    // its envelope. The flux profile is what catches this.
    expect(m.darkOutliers).toBe(0);
    expect(a.emptySamples).toBeGreaterThan(50);
    expect(a.min).toBe(0);
    expect(a.p05).toBe(0);
  });

  it('trims only the LEADING and TRAILING overhang', () => {
    // The band spans the full width, so overhang has to come from the
    // region: ask for 6 columns of pure background on each side by moving
    // the band inward instead.
    const width = 40;
    const height = 20;
    const image = makeBand(width, height);
    // Blank the first 6 and last 6 columns — pure overhang, not holes.
    punchColumns(image, width, 0, 0, 6);
    punchColumns(image, width, width - 6, 0, 6);

    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    expect(a.samples).toBe(width - 12);
    expect(a.emptySamples).toBe(0);
    expect(a.min).toBeCloseTo(1, 10);
  });

  it('refuses to normalise when more than half the tube is missing', () => {
    const width = 40;
    const height = 20;
    const image = makeBand(width, height);
    // Blank 24 of the 40 columns, interleaved so they stay interior.
    for (let x = 2; x < 38; x += 3) punchColumns(image, width, x, 0, 2);

    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    expect(a.emptySamples).toBeGreaterThan(a.samples / 2);
    expect(a.medianFlux).toBe(0);
    expect(a.profile).toHaveLength(0);
    expect(a.p05).toBe(0);
  });
});

describe('percentile convention', () => {
  // The percentile helper takes the observed sample NEAREST the interpolated
  // rank — `round(fraction * (n - 1))`, numpy's `method='nearest'` — not the
  // nearest-RANK `ceil(fraction * n) - 1`. The two disagree at n = 20, which
  // is what makes these pins real rather than tautological.
  const width = 20;
  const height = 20;

  it('selects the sample nearest the interpolated rank', () => {
    // Column x carries luminance 100 + x, so the 20 cross-section sums are
    // strictly increasing and already in ascending order.
    const image = makeBand(width, height, (x) => 100 + x);

    const a = measureAxialFlux(image, width, height, WHOLE(width, height), 'x');

    expect(a.samples).toBe(20);
    // Median: index round(0.5 * 19) = 10 -> 110. Nearest-rank would say 109.
    expect(a.medianFlux).toBeCloseTo(BAND_HEIGHT * 110, 4);
    // p05: index round(0.05 * 19) = 1 -> 101. Nearest-rank would say 100.
    expect(a.p05).toBeCloseTo(101 / 110, 10);
    // p95: index round(0.95 * 19) = 18 -> 118. Both conventions agree here.
    expect(a.p95).toBeCloseTo(118 / 110, 10);
    expect(a.min).toBeCloseTo(100 / 110, 10);
  });

  it('normalises at exactly half the profile empty, and gives up past it', () => {
    // The documented boundary: the normaliser survives half a missing tube and
    // collapses only past it. Column 0 and the tail stay lit so nothing is
    // trimmed as overhang and the profile keeps all 20 samples.
    const half = punchColumns(makeBand(width, height), width, 1, 0, 10);
    const atHalf = measureAxialFlux(half, width, height, WHOLE(width, height), 'x');
    expect(atHalf.samples).toBe(20);
    expect(atHalf.emptySamples).toBe(10);
    expect(atHalf.medianFlux).toBeCloseTo(BAND_HEIGHT * BAND_VALUE, 4);

    const past = punchColumns(makeBand(width, height), width, 1, 0, 11);
    const pastHalf = measureAxialFlux(past, width, height, WHOLE(width, height), 'x');
    expect(pastHalf.samples).toBe(20);
    expect(pastHalf.emptySamples).toBe(11);
    expect(pastHalf.medianFlux).toBe(0);
    expect(pastHalf.profile).toHaveLength(0);
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
    expect(a.emptySamples).toBe(0);
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
