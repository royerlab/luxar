/**
 * Unit tests for the gallery harness's still-vs-orbit comparison maths.
 *
 * The capture side of this check needs a browser, a dataset and a GPU; the
 * arithmetic does not. Sibling of `gallery-exposure-policy.test.ts`, which
 * splits the same way for the same reason.
 *
 * These matter more than a typical helper's tests: this code is a DETECTOR for
 * a defect that already shipped once (#1377). If its arithmetic silently
 * returned a high correlation for everything, the harness would print no
 * warning and the absence of a warning is indistinguishable from a passing
 * check — so the interesting assertions here are the ones that pin the
 * detector's *sensitivity*, not just its happy path.
 */

import { describe, expect, it } from 'vitest';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  COMPARE_SIZE,
  DISAGREEMENT_THRESHOLD,
  luma8,
  normalizedCrossCorrelation,
} from '../screenshots/frame-similarity';

/** Deterministic LCG — no `Math.random`, so a failure always reproduces. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return (s & 0x7fffffff) / 0x7fffffff;
  };
}

function noiseImage(seed: number, n = 4096): number[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => Math.round(r() * 255));
}

describe('normalizedCrossCorrelation', () => {
  it('is exactly 1 for an identical buffer', () => {
    const a = noiseImage(1);
    expect(normalizedCrossCorrelation(a, a)).toBeCloseTo(1, 12);
  });

  it('is exactly -1 for a photometric inversion', () => {
    const a = noiseImage(2);
    const inverted = a.map((v) => 255 - v);
    expect(normalizedCrossCorrelation(a, inverted)).toBeCloseTo(-1, 12);
  });

  it('is near 0 for independent images', () => {
    expect(Math.abs(normalizedCrossCorrelation(noiseImage(3), noiseImage(4)))).toBeLessThan(0.05);
  });

  it('is invariant to brightness and contrast', () => {
    // The whole reason for normalising. Auto-exposure re-runs between the still
    // and the orbit, and a progressively-streamed scene fills in over time;
    // neither is a change of POSE and neither may read as one.
    const a = noiseImage(5);
    const brighter = a.map((v) => v * 0.6 + 40);
    expect(normalizedCrossCorrelation(a, brighter)).toBeCloseTo(1, 10);
  });

  it('survives a flat buffer with 0 rather than NaN', () => {
    // A NaN would compare false against every threshold and so silently
    // DISABLE the check — the failure mode this detector exists to avoid.
    const flat = new Array(4096).fill(128);
    const black = new Array(4096).fill(0);
    for (const [x, y] of [
      [flat, flat],
      [black, black],
      [noiseImage(6), black],
    ] as number[][][]) {
      const c = normalizedCrossCorrelation(x, y);
      expect(Number.isNaN(c)).toBe(false);
      expect(c).toBe(0);
    }
  });

  it('rejects mismatched or empty buffers instead of returning a number', () => {
    expect(() => normalizedCrossCorrelation([1, 2, 3], [1, 2])).toThrow(/length mismatch/);
    expect(() => normalizedCrossCorrelation([], [])).toThrow(/empty/);
  });

  it('stays within [-1, 1] across many random pairs', () => {
    for (let s = 0; s < 50; s++) {
      const c = normalizedCrossCorrelation(noiseImage(100 + s, 512), noiseImage(900 + s, 512));
      expect(c).toBeGreaterThanOrEqual(-1.000001);
      expect(c).toBeLessThanOrEqual(1.000001);
    }
  });

  it('is symmetric', () => {
    const [a, b] = [noiseImage(7, 1024), noiseImage(8, 1024)];
    expect(normalizedCrossCorrelation(a, b)).toBeCloseTo(normalizedCrossCorrelation(b, a), 12);
  });
});

describe('the disagreement threshold, against measured captures', () => {
  // The numbers in the second column are real, from A/B runs of the capture
  // harness (see #1377). They are what the threshold was calibrated on, so
  // pinning them here is what stops a future tweak quietly reclassifying a
  // known-good or known-bad tile.
  const MEASURED: Array<[string, number, 'agree' | 'disagree']> = [
    ['asteroids, axis derived', 1.0, 'agree'],
    ['cosmicflows, axis derived', 0.981, 'agree'],
    ['asteroids, rolled (world-Y)', 0.552, 'disagree'],
    ['asteroids, rolled (re-measured)', 0.563, 'disagree'],
    ['cosmicflows, rolled (world-Y)', 0.522, 'disagree'],
  ];

  it.each(MEASURED)('%s (corr %f) classifies as %s', (_name, corr, verdict) => {
    expect(corr < DISAGREEMENT_THRESHOLD).toBe(verdict === 'disagree');
  });

  it('keeps clear margin on both sides of the real clusters', () => {
    const agree = MEASURED.filter(([, , v]) => v === 'agree').map(([, c]) => c);
    const disagree = MEASURED.filter(([, , v]) => v === 'disagree').map(([, c]) => c);
    // A threshold wedged against either cluster would flip on noise.
    expect(Math.min(...agree) - DISAGREEMENT_THRESHOLD).toBeGreaterThan(0.1);
    expect(DISAGREEMENT_THRESHOLD - Math.max(...disagree)).toBeGreaterThan(0.1);
  });
});

describe('luma8', () => {
  it('maps greys to themselves and is Rec. 601 weighted', () => {
    expect(luma8(0, 0, 0)).toBe(0);
    expect(luma8(255, 255, 255)).toBe(255);
    expect(luma8(128, 128, 128)).toBe(128);
    expect(luma8(255, 0, 0)).toBe(76); // 0.299 * 255
    expect(luma8(0, 255, 0)).toBe(150); // 0.587 * 255
    expect(luma8(0, 0, 255)).toBe(29); // 0.114 * 255
  });

  it('returns integers, so the buffer serialises compactly', () => {
    for (const v of [
      [1, 2, 3],
      [200, 100, 50],
      [7, 7, 7],
    ] as number[][]) {
      expect(Number.isInteger(luma8(v[0], v[1], v[2]))).toBe(true);
    }
  });

  it('rounding costs far less precision than the decision needs', () => {
    // Rounding to integers is what keeps the CDP payload small. Confirm it does
    // not perturb a correlation anywhere near the ~0.13 threshold margin.
    const r = rng(11);
    const exact: number[] = [];
    const rounded: number[] = [];
    for (let i = 0; i < 4096; i++) {
      const [red, g, b] = [r() * 255, r() * 255, r() * 255];
      exact.push(0.299 * red + 0.587 * g + 0.114 * b);
      rounded.push(luma8(red, g, b));
    }
    // Bound tied to the decision, not picked for roundness: the correlation lost
    // to rounding must sit at least two orders of magnitude below the ~0.13
    // margin between the threshold and the nearest real cluster. Measured ~2e-5.
    const lost = 1 - normalizedCrossCorrelation(exact, rounded);
    const margin = 0.981 - DISAGREEMENT_THRESHOLD;
    expect(lost).toBeLessThan(margin / 100);
  });
});

describe('the capture spec and this module agree', () => {
  const specPath = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    '../screenshots/generate-gallery.spec.ts'
  );

  it('the in-page luma expression matches luma8', () => {
    // The page-side decode CANNOT import this module — its body is serialised
    // into the browser — so the coefficients are duplicated by necessity. This
    // pins the duplicate: if either side is retuned alone, this fails.
    const src = fs.readFileSync(specPath, 'utf-8');
    expect(src).toContain('Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2])');
  });

  it('the page downscales to COMPARE_SIZE rather than a hardcoded literal', () => {
    const src = fs.readFileSync(specPath, 'utf-8');
    expect(src).toContain('[a, b, COMPARE_SIZE]');
    expect(COMPARE_SIZE).toBe(256);
  });

  it('freezes auto-dolly before still framing begins', () => {
    const src = fs.readFileSync(specPath, 'utf-8');
    const freeze = src.indexOf('controls?.setAutoDolly?.(false)');
    const frame = src.indexOf('await centerCamera(page);', freeze);

    expect(freeze).toBeGreaterThan(src.indexOf('await waitForDataLoaded(page'));
    expect(frame).toBeGreaterThan(freeze);
  });
});
