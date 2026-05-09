/**
 * Unit tests for the video codec selection helper.
 *
 * The original logic was inline in `recording-panel.ts`'s 432-LOC
 * runOfflineCaptureLoop and untestable in isolation. Now lives in
 * `recording/video-codec-selection.ts`.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  selectVideoCodec,
  type MediabunnyCodec,
} from '../../../../ui/recording/video-codec-selection';

const ENC_OPTS = { width: 1920, height: 1080, bitrate: 8_000_000 };

describe('selectVideoCodec', () => {
  it('returns the preferred codec when canEncodeVideo accepts it', async () => {
    const can = vi.fn(async () => true);
    const result = await selectVideoCodec({
      preferredCodec: 'h265',
      containerMode: 'mp4',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBe('hevc');
    expect(result.isPreferred).toBe(true);
    expect(result.fallbackFrom).toBeUndefined();
    expect(can).toHaveBeenCalledTimes(1);
    expect(can).toHaveBeenCalledWith('hevc', ENC_OPTS);
  });

  it('downgrades h265 → vp9 in WebM container before any encode test', async () => {
    const can = vi.fn(async () => true);
    const result = await selectVideoCodec({
      preferredCodec: 'h265',
      containerMode: 'webm',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBe('vp9');
    expect(result.isPreferred).toBe(false);
    // The first probe is for the downgraded codec, not hevc.
    expect(can).toHaveBeenCalledWith('vp9', ENC_OPTS);
  });

  it('walks fallback chain when preferred codec is unsupported', async () => {
    const supported: ReadonlySet<MediabunnyCodec> = new Set(['vp9']);
    const can = vi.fn(async (codec: MediabunnyCodec) => supported.has(codec));
    const result = await selectVideoCodec({
      preferredCodec: 'h265',
      containerMode: 'mp4',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBe('vp9');
    expect(result.isPreferred).toBe(false);
    expect(result.fallbackFrom).toBe('hevc');
  });

  it('HEVC fallback chain prefers AVC first', async () => {
    const supported: ReadonlySet<MediabunnyCodec> = new Set(['avc']);
    const can = vi.fn(async (codec: MediabunnyCodec) => supported.has(codec));
    const result = await selectVideoCodec({
      preferredCodec: 'h265',
      containerMode: 'mp4',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBe('avc');
    expect(result.fallbackFrom).toBe('hevc');
  });

  it('non-HEVC fallback chain prefers VP9 first', async () => {
    const supported: ReadonlySet<MediabunnyCodec> = new Set(['vp9', 'av1']);
    const can = vi.fn(async (codec: MediabunnyCodec) => supported.has(codec));
    const result = await selectVideoCodec({
      preferredCodec: 'h264',
      containerMode: 'mp4',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBe('vp9');
  });

  it('WebM filters fallback list to compatible codecs', async () => {
    // Only AVC is supported, but AVC is not WebM-compatible — should fail.
    const supported: ReadonlySet<MediabunnyCodec> = new Set(['avc']);
    const can = vi.fn(async (codec: MediabunnyCodec) => supported.has(codec));
    const result = await selectVideoCodec({
      preferredCodec: 'h265',
      containerMode: 'webm',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBeNull();
  });

  it('returns null when nothing in the fallback chain is supported', async () => {
    const can = vi.fn(async () => false);
    const result = await selectVideoCodec({
      preferredCodec: 'vp9',
      containerMode: 'webm',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBeNull();
    expect(result.isPreferred).toBe(false);
    expect(result.fallbackFrom).toBe('vp9');
  });

  it('user h264 → avc; supported: returns avc as preferred', async () => {
    const can = vi.fn(async (c: MediabunnyCodec) => c === 'avc');
    const result = await selectVideoCodec({
      preferredCodec: 'h264',
      containerMode: 'mp4',
      encOpts: ENC_OPTS,
      canEncodeVideo: can,
    });
    expect(result.codec).toBe('avc');
    expect(result.isPreferred).toBe(true);
  });
});
