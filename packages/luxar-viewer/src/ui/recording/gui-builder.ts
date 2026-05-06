/**
 * GUI / control-visibility helpers extracted from `recording-panel.ts`.
 *
 * The `buildGUI` and `updateControlVisibility` methods stay in the
 * panel — they touch a dozen lil-gui controller instances by name and
 * parameterizing them would just move the coupling around. What this
 * module owns is the *pure* mode → format and format → predicate logic
 * that the visibility rule checks against.
 *
 * Pulling these out makes the rules testable without instantiating a
 * RecordingPanel and lets future format additions land in one place.
 *
 * @module ui/recording/gui-builder
 */

import type { RecordingMode, OutputFormat } from '../recording-panel';

/**
 * The format options shown in the dropdown for each recording mode.
 *
 *   - `image`: still-image formats (PNG, WebP, JPEG) plus the HDR EXR.
 *   - `video`: WebM only — real-time MediaRecorder always emits WebM
 *     (the codec dropdown picks vp8 / vp9 inside the WebM container).
 *   - `turntable`: every offline-encoded format — image sequences end
 *     up as ZIPs, video formats go through mediabunny.
 */
const VALID_FORMATS_BY_MODE: Record<RecordingMode, OutputFormat[]> = {
  image: ['png', 'webp', 'jpeg', 'exr'],
  video: ['webm'],
  turntable: ['png', 'webp', 'jpeg', 'exr', 'mp4', 'webm', 'mkv'],
};

/** Return the formats valid for `mode`. */
export function getValidFormatsForMode(mode: RecordingMode): OutputFormat[] {
  return VALID_FORMATS_BY_MODE[mode];
}

/**
 * The format auto-correction default when the current selection is
 * invalid for the new mode. Matches the inline rule in the panel:
 * video → webm, image → webp, turntable → mp4.
 */
export function getDefaultFormatForMode(mode: RecordingMode): OutputFormat {
  if (mode === 'video') return 'webm';
  if (mode === 'image') return 'webp';
  return 'mp4';
}

/**
 * True for formats that produce one image per frame (so the encoder
 * doesn't care about bitrate). The Video-quality slider is hidden
 * for these.
 */
export function isImageSequenceFormat(fmt: OutputFormat): boolean {
  return fmt === 'exr' || fmt === 'png' || fmt === 'webp' || fmt === 'jpeg';
}

/** True for formats that need a video codec selector. */
export function isVideoContainerFormat(fmt: OutputFormat): boolean {
  return fmt === 'mp4' || fmt === 'webm' || fmt === 'mkv';
}

/**
 * Codecs supported by the browser's MediaRecorder API (used in real-
 * time Video mode). Turntable / offline modes use mediabunny which
 * supports all four codecs (h265 / vp9 / h264 / vp8).
 */
export function getMediaRecorderCodecs(): readonly string[] {
  return ['vp9', 'vp8'];
}

/** Mapping of GUI display labels to internal format values. */
export const FORMAT_LABEL_TO_VALUE: Readonly<Record<string, OutputFormat>> = {
  PNG: 'png',
  WebP: 'webp',
  JPEG: 'jpeg',
  EXR: 'exr',
  MP4: 'mp4',
  WebM: 'webm',
  MKV: 'mkv',
};

/** Mapping of GUI codec labels to internal codec values. */
export const CODEC_LABEL_TO_VALUE: Readonly<Record<string, string>> = {
  'H.265': 'h265',
  VP9: 'vp9',
  'H.264': 'h264',
  VP8: 'vp8',
};
