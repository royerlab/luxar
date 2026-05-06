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
  getMediaRecorderCodecs,
  FORMAT_LABEL_TO_VALUE,
  CODEC_LABEL_TO_VALUE,
} from '../../../../ui/recording/gui-builder';

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

describe('getMediaRecorderCodecs', () => {
  it('lists vp9 and vp8 — the two codecs the browser MediaRecorder supports', () => {
    expect(getMediaRecorderCodecs()).toEqual(['vp9', 'vp8']);
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
