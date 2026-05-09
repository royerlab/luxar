/**
 * Phase r8 §A5: shared recording types extracted from
 * `recording-panel.ts` so the per-mode drivers and helpers don't
 * need to import types from their parent orchestrator (which
 * inverts the intended dependency direction).
 *
 * `recording-panel.ts` re-exports each of these for back-compat
 * with existing external consumers.
 */

import type { VideoQuality } from './media-utilities';

export type RecordingMode = 'image' | 'video' | 'turntable';

/** Video resolution presets — 0 means native canvas size */
export type VideoResolution = 0 | 1080 | 1440 | 2160;

/** Output format — what file type you get */
export type OutputFormat = 'png' | 'webp' | 'jpeg' | 'exr' | 'mp4' | 'webm' | 'mkv';

/** Video codec for MP4/WebM encoding (ordered by modernity) */
export type VideoCodecOption = 'h265' | 'vp9' | 'h264' | 'vp8';

/** Re-export VideoQuality so consumers have a single import surface. */
export type { VideoQuality };

/** Panel visibility state snapshot for hide/restore */
export type PanelStates = Map<string, boolean>;

/** Recording options for image and video capture */
export interface RecordingOptions {
  // Image options
  outputFormat: OutputFormat;
  imageQuality: number;
  maxDPR: boolean;
  transparentBackground: boolean;
  // Video options
  videoDurationLimit: number; // 0 = unlimited, else seconds
  videoFPS: number;
  videoCodec: VideoCodecOption;
  videoQuality: VideoQuality;
  videoResolution: VideoResolution; // target height in pixels, 0 = native
  syncToSlider: boolean;
  syncDimensionIndex: number; // -1 = none
  // Turntable options
  turntableSpeed: number; // degrees per second
  frameByFrame: boolean; // offline frame-by-frame capture (smooth but slow)
  // General
  showPanels: boolean;
  includeOverlays: boolean;
}
