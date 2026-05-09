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

import type { RecordingMode, OutputFormat } from './types';

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

/**
 * Per-control visibility decision returned by
 * {@link computeControlVisibility}. Phase 19B (r5).
 *
 * The recording-panel's `updateControlVisibility` method previously
 * mixed the "should X show?" decisions with the lil-gui controller
 * `.show()` / `.hide()` calls. This object captures the pure-logic
 * answers so the panel only has to apply them, and so the rules
 * are testable without lil-gui in the loop.
 */
export interface ControlVisibilityDecision {
  /** Image-only group (quality, max DPR, transparent BG). */
  showImageGroup: boolean;
  /** Video / turntable shared group (resolution, duration, codec, …). */
  showVideoGroup: boolean;
  /** Turntable-only group (rotation speed, axis, …). */
  showTurntableGroup: boolean;

  /** Image quality slider — hidden for png / exr (lossless). */
  showImageQuality: boolean;
  /** Transparent-BG checkbox — hidden for exr (always has alpha). */
  showImageTransparent: boolean;

  /** Video codec dropdown — only video container formats. */
  showVideoCodec: boolean;
  /** Video quality slider — hidden for image-sequence formats. */
  showVideoQuality: boolean;
  /** Video duration limit — hidden for turntable (computed from speed). */
  showVideoDuration: boolean;
  /** Sync-to-slider toggle — hidden for turntable. */
  showSyncToggle: boolean;
  /** Sync-dimension dropdown — hidden when sync-toggle is off or for turntable. */
  showSyncDimension: boolean;

  /** Format dropdown options that should be visible for the current mode. */
  validFormats: readonly OutputFormat[];

  /**
   * If the current `outputFormat` is invalid for the new mode, this
   * holds the auto-corrected default. `null` if no correction
   * needed.
   */
  correctedFormat: OutputFormat | null;

  /**
   * For video container formats only: codec values that should be
   * visible in the codec dropdown. Video mode (MediaRecorder) limits
   * to the MediaRecorder-supported codecs; turntable uses mediabunny
   * which supports all. `null` for non-video formats.
   */
  visibleVideoCodecs: readonly string[] | null;
}

/**
 * Compute the per-control visibility decision for the recording
 * panel given the current mode + options. Pure function of its
 * inputs (no DOM, no controller access).
 */
export function computeControlVisibility(
  mode: RecordingMode,
  options: { outputFormat: OutputFormat; syncToSlider: boolean }
): ControlVisibilityDecision {
  const isImage = mode === 'image';
  const isVideo = mode === 'video';
  const isTurntable = mode === 'turntable';

  const validFormats = getValidFormatsForMode(mode);
  const correctedFormat = validFormats.includes(options.outputFormat)
    ? null
    : getDefaultFormatForMode(mode);
  const fmt = correctedFormat ?? options.outputFormat;

  const isVideoFormat = isVideoContainerFormat(fmt);
  const isImgSeq = isImageSequenceFormat(fmt);

  // Image quality slider visible only in image mode and only for
  // lossy formats (jpeg / webp). Hidden for png / exr.
  const showImageQuality = isImage && fmt !== 'png' && fmt !== 'exr';
  // Transparent-BG hidden for exr (always has alpha).
  const showImageTransparent = isImage && fmt !== 'exr';

  const showVideoGroup = isVideo || isTurntable;
  const showVideoCodec = showVideoGroup && isVideoFormat;
  // Video quality irrelevant for image-sequence formats.
  const showVideoQuality = showVideoGroup && !isImgSeq;
  // Turntable computes duration from speed; sync-toggle is irrelevant.
  const showVideoDuration = isVideo;
  const showSyncToggle = isVideo;
  const showSyncDimension = isVideo && options.syncToSlider;

  let visibleVideoCodecs: readonly string[] | null = null;
  if (showVideoCodec) {
    if (isVideo) {
      visibleVideoCodecs = getMediaRecorderCodecs();
    } else {
      // Turntable: mediabunny supports all codecs; pass through the
      // full label-to-value map's value set.
      visibleVideoCodecs = Object.values(CODEC_LABEL_TO_VALUE);
    }
  }

  return {
    showImageGroup: isImage,
    showVideoGroup,
    showTurntableGroup: isTurntable,
    showImageQuality,
    showImageTransparent,
    showVideoCodec,
    showVideoQuality,
    showVideoDuration,
    showSyncToggle,
    showSyncDimension,
    validFormats,
    correctedFormat,
    visibleVideoCodecs,
  };
}
