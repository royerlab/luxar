/**
 * Unit tests for the hdr-capture pure helpers.
 *
 * The actual `composer.render()` and `gl.readPixels()` calls live in
 * the manager and need a real WebGL context — out of scope for unit
 * tests. The deterministic book-keeping (which buffer the result
 * lands in, the memory estimate, the EXR log line) is pure and tested
 * here.
 */

import { describe, it, expect } from 'vitest';
import {
  HDR_BYTES_PER_PIXEL,
  countSwapPasses,
  estimatePostProcMemoryMB,
  formatHDRExrLogLine,
  pickResultBuffer,
} from '../../../../rendering/post-processing/hdr-capture';

describe('countSwapPasses', () => {
  it('returns 0 for an empty pass list', () => {
    expect(countSwapPasses([])).toBe(0);
  });

  it('counts only enabled + needsSwap passes', () => {
    expect(
      countSwapPasses([
        { enabled: true, needsSwap: true },
        { enabled: true, needsSwap: false }, // ClearPass-like
        { enabled: false, needsSwap: true }, // disabled, ignored
        { enabled: true, needsSwap: true },
        { enabled: false, needsSwap: false },
      ])
    ).toBe(2);
  });
});

describe('pickResultBuffer', () => {
  it('returns inputBuffer when an even number of swaps occurred', () => {
    const input = { tag: 'A' };
    const output = { tag: 'B' };
    expect(pickResultBuffer([], input, output)).toBe(input);
    expect(
      pickResultBuffer(
        [
          { enabled: true, needsSwap: true },
          { enabled: true, needsSwap: true },
        ],
        input,
        output
      )
    ).toBe(input);
  });

  it('returns outputBuffer when an odd number of swaps occurred', () => {
    const input = { tag: 'A' };
    const output = { tag: 'B' };
    expect(pickResultBuffer([{ enabled: true, needsSwap: true }], input, output)).toBe(output);
    expect(
      pickResultBuffer(
        [
          { enabled: true, needsSwap: true },
          { enabled: true, needsSwap: true },
          { enabled: true, needsSwap: true },
        ],
        input,
        output
      )
    ).toBe(output);
  });

  it('ignores disabled passes and non-swap passes when picking', () => {
    const input = { tag: 'A' };
    const output = { tag: 'B' };
    expect(
      pickResultBuffer(
        [
          { enabled: true, needsSwap: true },
          { enabled: false, needsSwap: true }, // disabled, no swap
          { enabled: true, needsSwap: false }, // no swap
        ],
        input,
        output
      )
    ).toBe(output); // 1 effective swap → outputBuffer
  });
});

describe('HDR_BYTES_PER_PIXEL', () => {
  it('is 8 (half-float RGBA: 2 bytes × 4 channels)', () => {
    expect(HDR_BYTES_PER_PIXEL).toBe(8);
  });
});

describe('estimatePostProcMemoryMB', () => {
  it('base case: 3 buffers, 1MP, no SSAA/MSAA, no bloom/AO', () => {
    const mb = estimatePostProcMemoryMB({
      pixelCount: 1024 * 1024,
      ssaaEnabled: false,
      ssaaMultiplier: 1,
      msaaEnabled: false,
      msaaSamples: 0,
      hasBloom: false,
      hasAO: false,
    });
    // 1 MP × 8 B × 3 buffers / (1024×1024) = 24
    expect(mb).toBeCloseTo(24, 6);
  });

  it('+2 buffers when bloom is on, +1 when AO is on', () => {
    const baseInputs = {
      pixelCount: 1024 * 1024,
      ssaaEnabled: false,
      ssaaMultiplier: 1,
      msaaEnabled: false,
      msaaSamples: 0,
      hasBloom: false,
      hasAO: false,
    };
    const base = estimatePostProcMemoryMB(baseInputs);
    const withBloom = estimatePostProcMemoryMB({ ...baseInputs, hasBloom: true });
    const withAO = estimatePostProcMemoryMB({ ...baseInputs, hasAO: true });
    expect(withBloom / base).toBeCloseTo(5 / 3, 6);
    expect(withAO / base).toBeCloseTo(4 / 3, 6);
  });

  it('SSAA multiplies by m²; MSAA multiplies by sample count', () => {
    const inputs = {
      pixelCount: 1000,
      ssaaEnabled: true,
      ssaaMultiplier: 2,
      msaaEnabled: true,
      msaaSamples: 4,
      hasBloom: false,
      hasAO: false,
    };
    // 1000 × 4 (ssaa²) × 4 (msaa) × 8 B × 3 buffers
    const expected = (1000 * 4 * 4 * 8 * 3) / (1024 * 1024);
    expect(estimatePostProcMemoryMB(inputs)).toBeCloseTo(expected, 6);
  });

  it('disabled SSAA/MSAA contribute multiplier 1, regardless of multiplier value', () => {
    const inputs = {
      pixelCount: 1024 * 1024,
      ssaaEnabled: false,
      ssaaMultiplier: 4, // ignored
      msaaEnabled: false,
      msaaSamples: 16, // ignored
      hasBloom: false,
      hasAO: false,
    };
    expect(estimatePostProcMemoryMB(inputs)).toBeCloseTo(24, 6);
  });
});

describe('formatHDRExrLogLine', () => {
  it('formats half-float capture metadata', () => {
    const line = formatHDRExrLogLine(1920, 1080, true, 4 * 1024 * 1024);
    expect(line).toBe('HDR EXR captured: 1920x1080, half-float, 4.0 MB');
  });

  it('formats full-float capture metadata', () => {
    const line = formatHDRExrLogLine(640, 480, false, 1.5 * 1024 * 1024);
    expect(line).toBe('HDR EXR captured: 640x480, float, 1.5 MB');
  });

  it('uses one decimal place for MB', () => {
    const line = formatHDRExrLogLine(100, 100, true, 1234567);
    expect(line).toMatch(/\d+\.\d MB$/);
  });
});
