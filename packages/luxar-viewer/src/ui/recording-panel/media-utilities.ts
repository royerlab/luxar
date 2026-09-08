/**
 * Media-utility helpers for the recording panel.
 *
 * Pure functions extracted from `ui/recording-panel.ts` so the
 * arithmetic + string-formatting paths are unit-testable without a
 * real `MediaRecorder` or DOM. The browser-coupled
 * {@link getSupportedMimeType} accepts an explicit MediaRecorder-like
 * argument so it can also be tested with a stub.
 *
 * @module ui/recording-panel/media-utilities
 */

/** Quality presets for video recording. */
export type VideoQuality = 'low' | 'medium' | 'high' | 'max';

/**
 * Bits-per-pixel multipliers for single-pass VP9 encoding. The default
 * MediaRecorder bitrate (~2.5 Mbps) is far too low for HD content.
 */
const BPP_BY_QUALITY: Record<VideoQuality, number> = {
  low: 0.04, // ~2.5 Mbps at 1080p30 (comparable to default)
  medium: 0.08, // ~5 Mbps at 1080p30
  high: 0.15, // ~9.3 Mbps at 1080p30 — good quality
  max: 0.3, // ~18.7 Mbps at 1080p30 — near-lossless
};

/**
 * Compute the recommended video bitrate (bits per second) for a given
 * canvas size, FPS, and quality preset.
 *
 * `width × height × fps × bpp`, rounded.
 */
export function computeVideoBitrate(
  width: number,
  height: number,
  fps: number,
  quality: VideoQuality
): number {
  const bpp = BPP_BY_QUALITY[quality] ?? BPP_BY_QUALITY.high;
  return Math.round(width * height * fps * bpp);
}

/**
 * Minimal contract we need from `MediaRecorder` to probe codec
 * support — just the static `isTypeSupported(string)` method.
 */
export interface MediaRecorderLike {
  isTypeSupported(type: string): boolean;
}

/**
 * Return the highest-quality WebM codec MediaRecorder can record with,
 * or `null` if MediaRecorder is unavailable / no codec is supported.
 *
 * MediaRecorder only supports WebM with VP9 or VP8 — H.264/H.265 are
 * NOT valid WebM codecs (the videoCodec option there only applies to
 * the offline mediabunny path, not real-time recording).
 *
 * Pass an explicit `recorder` for testing; default uses the global.
 */
export function getSupportedMimeType(
  recorder: MediaRecorderLike | undefined = typeof MediaRecorder !== 'undefined'
    ? (MediaRecorder as unknown as MediaRecorderLike)
    : undefined,
  withAudio = false
): string | null {
  if (!recorder) return null;
  // With an audio track the container needs an audio codec too; Opus is the
  // only one WebM carries. A recorder that cannot name it still records the
  // audio through the plain candidates below (the browser picks the codec).
  const candidates = [
    ...(withAudio ? ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus'] : []),
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  for (const type of candidates) {
    if (recorder.isTypeSupported(type)) {
      return type;
    }
  }
  return null;
}

/** Padding-2 helper used by {@link generateTimestampSuffix}. */
function pad2(n: number): string {
  return n.toString().padStart(2, '0');
}

/**
 * Format a Date as `YYYY-MM-DD-hhmmss`. Pure given an explicit Date.
 */
export function generateTimestampSuffix(now: Date): string {
  return (
    `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}-` +
    `${pad2(now.getHours())}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`
  );
}

/**
 * Build a Luxar capture filename with the given extension and a timestamp
 * suffix. Defaults `now` to `new Date()` for production callers; tests
 * can pin a specific Date.
 */
export function generateFilename(ext: string, now: Date = new Date()): string {
  return `luxar-capture-${generateTimestampSuffix(now)}.${ext}`;
}

/**
 * Anchor offset (same semantics as CSS `transform: translate`). Pure.
 */
export type AnchorName =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'center-left'
  | 'center'
  | 'center-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right';

export function anchorOffset(
  anchor: AnchorName | string,
  width: number,
  height: number
): [number, number] {
  switch (anchor) {
    case 'top-left':
      return [0, 0];
    case 'top-center':
      return [-width / 2, 0];
    case 'top-right':
      return [-width, 0];
    case 'center-left':
      return [0, -height / 2];
    case 'center':
      return [-width / 2, -height / 2];
    case 'center-right':
      return [-width, -height / 2];
    case 'bottom-left':
      return [0, -height];
    case 'bottom-center':
      return [-width / 2, -height];
    case 'bottom-right':
      return [-width, -height];
    default:
      return [0, 0];
  }
}
