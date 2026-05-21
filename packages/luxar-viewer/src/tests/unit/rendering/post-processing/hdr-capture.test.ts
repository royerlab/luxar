/**
 * Pure-helper tests for the `hdr-capture` module. Covers the EXR
 * log-line formatter.
 */

import { describe, it, expect } from 'vitest';
import { formatHDRExrLogLine } from '../../../../rendering/post-processing/hdr/capture';

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
