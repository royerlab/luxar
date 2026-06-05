/**
 * Unit tests for the GUI / control-visibility format helpers.
 *
 * The visibility logic in `recording-panel.ts` previously hard-coded
 * three identical lookup tables (mode → formats, format → predicates,
 * GUI label → value). These tests pin every branch of those rules so
 * future format additions land in one place.
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  getValidFormatsForMode,
  getDefaultFormatForMode,
  isImageSequenceFormat,
  isVideoContainerFormat,
  computeControlVisibility,
  FORMAT_LABEL_TO_VALUE,
  CODEC_LABEL_TO_VALUE,
} from '../../../../ui/recording-panel/gui-builder';
import type { RecordingMode, OutputFormat } from '../../../../ui/recording-panel/types';

describe('getValidFormatsForMode', () => {
  it('returns the four image formats for image mode', () => {
    // [P1] Copy before sorting: getValidFormatsForMode returns the live
    // module-level array, so an in-place .sort() here would corrupt the
    // shared constant and leak ordering into every later test.
    expect([...getValidFormatsForMode('image')].sort()).toEqual(['exr', 'jpeg', 'png', 'webp']);
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
  it('maps the complete label set to internal values, with no missing or extra keys', () => {
    // [P2/P11] Assert the whole map (not key-by-key): a mutation that
    // deletes, renames, reorders, or adds a label is caught here. The two
    // GUI-side maps and the dropdown options must stay in lockstep.
    expect(FORMAT_LABEL_TO_VALUE).toEqual({
      PNG: 'png',
      WebP: 'webp',
      JPEG: 'jpeg',
      EXR: 'exr',
      MP4: 'mp4',
      WebM: 'webm',
      MKV: 'mkv',
    });
    expect(Object.keys(FORMAT_LABEL_TO_VALUE)).toEqual([
      'PNG',
      'WebP',
      'JPEG',
      'EXR',
      'MP4',
      'WebM',
      'MKV',
    ]);
  });
});

describe('CODEC_LABEL_TO_VALUE', () => {
  it('maps the complete codec-label set to internal values, with no missing or extra keys', () => {
    // [P2/P11] Full-map assertion: the turntable codec dropdown derives its
    // visible options from Object.values(CODEC_LABEL_TO_VALUE), so a dropped
    // or reordered codec would silently change the dropdown.
    expect(CODEC_LABEL_TO_VALUE).toEqual({
      'H.265': 'h265',
      VP9: 'vp9',
      'H.264': 'h264',
      VP8: 'vp8',
    });
    expect(Object.keys(CODEC_LABEL_TO_VALUE)).toEqual(['H.265', 'VP9', 'H.264', 'VP8']);
  });
});

describe('computeControlVisibility', () => {
  it('image mode + png: shows image group, hides quality + transparent (lossless)', () => {
    const d = computeControlVisibility('image', { outputFormat: 'png', syncToSlider: false });
    // [P2] Pin the ENTIRE decision object for this representative case so a
    // mutation that flips any single field (or returns the wrong shape) is
    // caught — partial assertions let most-of-the-object mutations survive.
    expect(d).toEqual({
      showImageGroup: true,
      showVideoGroup: false,
      showTurntableGroup: false,
      showFormat: true, // image offers 4 formats → real choice
      showImageQuality: false, // png is lossless
      showImageTransparent: true, // png has alpha
      showVideoCodec: false,
      showVideoQuality: false,
      showVideoDuration: false,
      showSyncToggle: false,
      showSyncDimension: false,
      validFormats: ['png', 'webp', 'jpeg', 'exr'],
      correctedFormat: null,
      visibleVideoCodecs: null,
    });
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

  it('[property] invariants hold across every mode × format × sync combination', () => {
    // [P12] The example cases above are specific; these are the algebraic
    // invariants that must hold for ALL inputs — including invalid mode/format
    // pairs that trigger auto-correction. fast-check enumerates the full grid.
    const modes: RecordingMode[] = ['image', 'video', 'turntable'];
    const formats: OutputFormat[] = ['png', 'webp', 'jpeg', 'exr', 'mp4', 'webm', 'mkv'];
    fc.assert(
      fc.property(
        fc.constantFrom(...modes),
        fc.constantFrom(...formats),
        fc.boolean(),
        (mode, outputFormat, syncToSlider) => {
          const d = computeControlVisibility(mode, { outputFormat, syncToSlider });

          // Group visibility is a pure function of mode.
          expect(d.showImageGroup).toBe(mode === 'image');
          expect(d.showVideoGroup).toBe(mode !== 'image');
          expect(d.showTurntableGroup).toBe(mode === 'turntable');

          // validFormats always equals the mode's canonical list.
          expect(d.validFormats).toEqual(getValidFormatsForMode(mode));

          // correctedFormat is non-null EXACTLY when the chosen format is
          // invalid for the mode, and the correction is itself valid.
          if (d.correctedFormat === null) {
            expect(d.validFormats).toContain(outputFormat);
          } else {
            expect(d.validFormats).not.toContain(outputFormat);
            expect(d.validFormats).toContain(d.correctedFormat);
          }

          // A codec list exists exactly when the codec dropdown is shown.
          expect(d.visibleVideoCodecs !== null).toBe(d.showVideoCodec);

          // The sync-dimension dropdown can only show when the sync toggle does.
          if (d.showSyncDimension) expect(d.showSyncToggle).toBe(true);
        }
      )
    );
  });
});
