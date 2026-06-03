/**
 * Unit tests for the GUI / control-visibility format helpers.
 *
 * The visibility logic in `recording-panel.ts` previously hard-coded
 * three identical lookup tables (mode → formats, format → predicates,
 * GUI label → value). These tests pin every branch of those rules so
 * future format additions land in one place.
 */

import { describe, it, expect } from 'vitest';
import {
  getValidFormatsForMode,
  getDefaultFormatForMode,
  isImageSequenceFormat,
  isVideoContainerFormat,
  computeControlVisibility,
  FORMAT_LABEL_TO_VALUE,
  CODEC_LABEL_TO_VALUE,
} from '../../../../ui/recording-panel/gui-builder';

describe('getValidFormatsForMode', () => {
  it('returns the four image formats for image mode', () => {
    expect(getValidFormatsForMode('image').sort()).toEqual(['exr', 'jpeg', 'png', 'webp']);
  });

  it('returns only WebM for video mode', () => {
    expect(getValidFormatsForMode('video')).toEqual(['webm']);
  });

  it('returns the union of image + video formats for turntable mode', () => {
    const formats = getValidFormatsForMode('turntable');
    for (const expected of ['png', 'webp', 'jpeg', 'exr', 'mp4', 'webm', 'mkv']) {
      expect(formats).toContain(expected);
    }
    expect(formats).toHaveLength(7);
  });
});

describe('getDefaultFormatForMode', () => {
  it('defaults to webm for video mode', () => {
    expect(getDefaultFormatForMode('video')).toBe('webm');
  });

  it('defaults to webp for image mode', () => {
    expect(getDefaultFormatForMode('image')).toBe('webp');
  });

  it('defaults to mp4 for turntable mode', () => {
    expect(getDefaultFormatForMode('turntable')).toBe('mp4');
  });
});

describe('isImageSequenceFormat', () => {
  it('returns true for image-per-frame formats', () => {
    expect(isImageSequenceFormat('png')).toBe(true);
    expect(isImageSequenceFormat('webp')).toBe(true);
    expect(isImageSequenceFormat('jpeg')).toBe(true);
    expect(isImageSequenceFormat('exr')).toBe(true);
  });

  it('returns false for video container formats', () => {
    expect(isImageSequenceFormat('mp4')).toBe(false);
    expect(isImageSequenceFormat('webm')).toBe(false);
    expect(isImageSequenceFormat('mkv')).toBe(false);
  });
});

describe('isVideoContainerFormat', () => {
  it('returns true only for mp4 / webm / mkv', () => {
    expect(isVideoContainerFormat('mp4')).toBe(true);
    expect(isVideoContainerFormat('webm')).toBe(true);
    expect(isVideoContainerFormat('mkv')).toBe(true);
    expect(isVideoContainerFormat('png')).toBe(false);
    expect(isVideoContainerFormat('webp')).toBe(false);
    expect(isVideoContainerFormat('jpeg')).toBe(false);
    expect(isVideoContainerFormat('exr')).toBe(false);
  });
});

describe('FORMAT_LABEL_TO_VALUE', () => {
  it('maps each GUI label to the right internal value', () => {
    expect(FORMAT_LABEL_TO_VALUE.PNG).toBe('png');
    expect(FORMAT_LABEL_TO_VALUE.WebP).toBe('webp');
    expect(FORMAT_LABEL_TO_VALUE.JPEG).toBe('jpeg');
    expect(FORMAT_LABEL_TO_VALUE.EXR).toBe('exr');
    expect(FORMAT_LABEL_TO_VALUE.MP4).toBe('mp4');
    expect(FORMAT_LABEL_TO_VALUE.WebM).toBe('webm');
    expect(FORMAT_LABEL_TO_VALUE.MKV).toBe('mkv');
  });
});

describe('CODEC_LABEL_TO_VALUE', () => {
  it('maps each codec label to the right internal value', () => {
    expect(CODEC_LABEL_TO_VALUE['H.265']).toBe('h265');
    expect(CODEC_LABEL_TO_VALUE.VP9).toBe('vp9');
    expect(CODEC_LABEL_TO_VALUE['H.264']).toBe('h264');
    expect(CODEC_LABEL_TO_VALUE.VP8).toBe('vp8');
  });
});

describe('computeControlVisibility', () => {
  it('image mode + png: shows image group, hides quality + transparent (lossless)', () => {
    const d = computeControlVisibility('image', { outputFormat: 'png', syncToSlider: false });
    expect(d.showImageGroup).toBe(true);
    expect(d.showVideoGroup).toBe(false);
    expect(d.showTurntableGroup).toBe(false);
    expect(d.showImageQuality).toBe(false); // png is lossless
    expect(d.showImageTransparent).toBe(true); // png has alpha
    expect(d.correctedFormat).toBeNull();
  });

  it('image mode + jpeg: quality slider visible, transparent hidden when format is exr-equivalent', () => {
    const d = computeControlVisibility('image', { outputFormat: 'jpeg', syncToSlider: false });
    expect(d.showImageQuality).toBe(true);
    expect(d.showImageTransparent).toBe(true);
  });

  it('image mode + exr: quality hidden (lossless), transparent hidden (always alpha)', () => {
    const d = computeControlVisibility('image', { outputFormat: 'exr', syncToSlider: false });
    expect(d.showImageQuality).toBe(false);
    expect(d.showImageTransparent).toBe(false);
  });

  it('image mode + invalid mp4: auto-corrects to webp', () => {
    const d = computeControlVisibility('image', { outputFormat: 'mp4', syncToSlider: false });
    expect(d.correctedFormat).toBe('webp');
  });

  it('video mode + webm: shows video group but hides codec + format (WebM-only)', () => {
    const d = computeControlVisibility('video', { outputFormat: 'webm', syncToSlider: false });
    expect(d.showVideoGroup).toBe(true);
    expect(d.showImageGroup).toBe(false);
    // Real-time MediaRecorder picks the codec itself — no codec dropdown.
    expect(d.showVideoCodec).toBe(false);
    expect(d.visibleVideoCodecs).toBeNull();
    // WebM is the only valid video-mode format, so the dropdown is hidden.
    expect(d.showFormat).toBe(false);
    expect(d.showVideoDuration).toBe(true);
    expect(d.showSyncToggle).toBe(true);
    expect(d.showSyncDimension).toBe(false); // sync is off
  });

  it('video mode + syncToSlider: shows sync dimension dropdown', () => {
    const d = computeControlVisibility('video', { outputFormat: 'webm', syncToSlider: true });
    expect(d.showSyncDimension).toBe(true);
  });

  it('turntable mode + mp4: shows video + turntable groups, all codecs visible, format shown', () => {
    const d = computeControlVisibility('turntable', { outputFormat: 'mp4', syncToSlider: false });
    expect(d.showVideoGroup).toBe(true);
    expect(d.showTurntableGroup).toBe(true);
    expect(d.showVideoCodec).toBe(true);
    // Turntable has 7 formats, so the dropdown is a real choice.
    expect(d.showFormat).toBe(true);
    // Turntable uses mediabunny; all codecs visible.
    expect(d.visibleVideoCodecs).toEqual(['h265', 'vp9', 'h264', 'vp8']);
    // Turntable computes duration from speed; sync irrelevant.
    expect(d.showVideoDuration).toBe(false);
    expect(d.showSyncToggle).toBe(false);
    expect(d.showSyncDimension).toBe(false);
  });

  it('turntable mode + png (image sequence): video quality hidden, codec hidden', () => {
    const d = computeControlVisibility('turntable', { outputFormat: 'png', syncToSlider: false });
    expect(d.showVideoQuality).toBe(false);
    expect(d.showVideoCodec).toBe(false);
    expect(d.visibleVideoCodecs).toBeNull();
    // png is lossless — no quality slider.
    expect(d.showImageQuality).toBe(false);
  });

  it('turntable mode + jpeg/webp (lossy sequence): image quality slider visible', () => {
    for (const fmt of ['jpeg', 'webp'] as const) {
      const d = computeControlVisibility('turntable', { outputFormat: fmt, syncToSlider: false });
      expect(d.showImageQuality).toBe(true);
      // It's the image-quality control, not the video-bitrate one.
      expect(d.showVideoQuality).toBe(false);
    }
  });

  it('keeps the format dropdown visible when the mode offers multiple formats', () => {
    // Image (4 formats) and turntable (7) show the dropdown; only video
    // (WebM-only) hides it — asserted in the video-mode test above.
    expect(
      computeControlVisibility('image', { outputFormat: 'png', syncToSlider: false }).showFormat
    ).toBe(true);
    expect(
      computeControlVisibility('turntable', { outputFormat: 'mp4', syncToSlider: false }).showFormat
    ).toBe(true);
  });

  it('valid formats per mode match getValidFormatsForMode', () => {
    expect(
      computeControlVisibility('image', { outputFormat: 'png', syncToSlider: false }).validFormats
    ).toEqual(getValidFormatsForMode('image'));
    expect(
      computeControlVisibility('video', { outputFormat: 'webm', syncToSlider: false }).validFormats
    ).toEqual(getValidFormatsForMode('video'));
    expect(
      computeControlVisibility('turntable', { outputFormat: 'mp4', syncToSlider: false })
        .validFormats
    ).toEqual(getValidFormatsForMode('turntable'));
  });
});
