/**
 * HDR-capture log helper.
 *
 * Centralized so the unit test pins the format. Used by
 * `captureHDRAsEXR` after a successful EXR encode.
 *
 * @module rendering/post-processing/hdr/capture
 */

/**
 * Format the size-tagged log line printed after a successful EXR
 * capture.
 */
export function formatHDRExrLogLine(
  width: number,
  height: number,
  isHalfFloat: boolean,
  byteLength: number
): string {
  return (
    `HDR EXR captured: ${width}x${height}, ` +
    `${isHalfFloat ? 'half-float' : 'float'}, ` +
    `${(byteLength / (1024 * 1024)).toFixed(1)} MB`
  );
}
