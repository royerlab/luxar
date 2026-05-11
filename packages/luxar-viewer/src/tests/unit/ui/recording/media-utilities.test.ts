/**
 * Unit tests for the recording-panel media-utility helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  type MediaRecorderLike,
  anchorOffset,
  computeVideoBitrate,
  generateFfmpegScript,
  generateFilename,
  generateTimestampSuffix,
  getSupportedMimeType,
} from '../../../../ui/recording/media-utilities';

describe('computeVideoBitrate', () => {
  it('uses bpp from the quality preset', () => {
    // 1920×1080 × 30 × 0.15 (high) = 9_331_200
    expect(computeVideoBitrate(1920, 1080, 30, 'high')).toBe(9_331_200);
  });

  it('low quality is 4× smaller than max', () => {
    const low = computeVideoBitrate(1920, 1080, 30, 'low');
    const max = computeVideoBitrate(1920, 1080, 30, 'max');
    expect(max / low).toBeCloseTo(7.5, 6); // 0.3 / 0.04 = 7.5
  });

  it('scales linearly with each input', () => {
    expect(computeVideoBitrate(100, 100, 30, 'high')).toBe(Math.round(100 * 100 * 30 * 0.15));
    expect(computeVideoBitrate(100, 200, 30, 'high')).toBe(Math.round(100 * 200 * 30 * 0.15));
    expect(computeVideoBitrate(100, 100, 60, 'high')).toBe(Math.round(100 * 100 * 60 * 0.15));
  });

  it('falls back to high when given an unknown quality', () => {
    const unknown = computeVideoBitrate(1, 1, 1, 'unknown' as 'low');
    expect(unknown).toBe(Math.round(0.15));
  });
});

describe('getSupportedMimeType', () => {
  function makeRecorderStub(supported: ReadonlyArray<string>): MediaRecorderLike {
    return { isTypeSupported: (t) => supported.includes(t) };
  }

  it('returns the highest-quality codec available (VP9 preferred)', () => {
    expect(
      getSupportedMimeType(
        makeRecorderStub(['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'])
      )
    ).toBe('video/webm;codecs=vp9');
  });

  it('falls back to VP8 when VP9 unsupported', () => {
    expect(getSupportedMimeType(makeRecorderStub(['video/webm;codecs=vp8', 'video/webm']))).toBe(
      'video/webm;codecs=vp8'
    );
  });

  it('falls back to bare video/webm', () => {
    expect(getSupportedMimeType(makeRecorderStub(['video/webm']))).toBe('video/webm');
  });

  it('returns null when no codec is supported', () => {
    expect(getSupportedMimeType(makeRecorderStub([]))).toBeNull();
  });

  it('returns null when MediaRecorder is unavailable', () => {
    expect(getSupportedMimeType(undefined)).toBeNull();
  });
});

describe('generateTimestampSuffix', () => {
  it('formats YYYY-MM-DD-hhmmss with zero-padding', () => {
    // Mar 5 2026 at 09:08:07 (Jan = 0)
    expect(generateTimestampSuffix(new Date(2026, 2, 5, 9, 8, 7))).toBe('2026-03-05-090807');
  });

  it('handles end-of-year correctly', () => {
    expect(generateTimestampSuffix(new Date(2025, 11, 31, 23, 59, 59))).toBe('2025-12-31-235959');
  });
});

describe('generateFilename', () => {
  it('appends timestamp + extension to the luxar-capture prefix', () => {
    const name = generateFilename('png', new Date(2026, 4, 5, 10, 30, 0));
    expect(name).toBe('luxar-capture-2026-05-05-103000.png');
  });

  it('respects arbitrary extension', () => {
    const name = generateFilename('exr', new Date(2026, 4, 5, 10, 30, 0));
    expect(name).toBe('luxar-capture-2026-05-05-103000.exr');
  });
});

describe('generateFfmpegScript', () => {
  it('includes frame count, fps, computed duration, and ext-aware input pattern', () => {
    const script = generateFfmpegScript(30, 600, 'png');
    expect(script).toContain('600 PNG frames at 30 FPS (20.0s)');
    expect(script).toContain("'frame_%06d.png'");
    expect(script).toContain('-framerate 30');
    expect(script).toContain('"turntable.mp4"');
  });

  it('appends an HDR encoding stanza ONLY for EXR sequences', () => {
    const exr = generateFfmpegScript(30, 60, 'exr');
    const png = generateFfmpegScript(30, 60, 'png');
    expect(exr).toContain('HDR MP4');
    expect(exr).toContain('yuv420p10le');
    expect(png).not.toContain('HDR MP4');
    expect(png).not.toContain('yuv420p10le');
  });

  it('defaults to exr when ext omitted', () => {
    expect(generateFfmpegScript(30, 60)).toContain('HDR MP4');
  });

  it('starts with shebang and includes the chmod usage hint', () => {
    expect(generateFfmpegScript(30, 30, 'jpg').startsWith('#!/bin/bash')).toBe(true);
    expect(generateFfmpegScript(30, 30, 'jpg')).toContain('chmod +x encode_video.sh');
  });
});

describe('anchorOffset', () => {
  it('returns [0, 0] for top-left and unknown anchors', () => {
    expect(anchorOffset('top-left', 100, 80)).toEqual([0, 0]);
    expect(anchorOffset('garbage', 100, 80)).toEqual([0, 0]);
  });

  it('maps the 9 anchor names to documented translation offsets', () => {
    expect(anchorOffset('top-center', 100, 80)).toEqual([-50, 0]);
    expect(anchorOffset('top-right', 100, 80)).toEqual([-100, 0]);
    expect(anchorOffset('center-left', 100, 80)).toEqual([0, -40]);
    expect(anchorOffset('center', 100, 80)).toEqual([-50, -40]);
    expect(anchorOffset('center-right', 100, 80)).toEqual([-100, -40]);
    expect(anchorOffset('bottom-left', 100, 80)).toEqual([0, -80]);
    expect(anchorOffset('bottom-center', 100, 80)).toEqual([-50, -80]);
    expect(anchorOffset('bottom-right', 100, 80)).toEqual([-100, -80]);
  });
});
