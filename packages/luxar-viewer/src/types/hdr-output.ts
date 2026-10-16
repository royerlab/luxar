/**
 * HDR output (PROTOTYPE, `?hdr`): the headroom bounds and the state a
 * WebGPU renderer carries when it presents to an extended-range canvas.
 *
 * Headroom is how far above SDR white the display can go, as a multiple of
 * SDR white. Shared by `config/url-params` (which clamps `?hdr=<n>`), the
 * renderer setup, post-processing and the Rendering panel.
 *
 * @module types/hdr-output
 */

/** Headroom range, also the Rendering panel slider's range. */
export const HDR_HEADROOM_MIN = 1;
export const HDR_HEADROOM_MAX = 16;

/**
 * Fallback headroom when nothing identifies the display: 2x SDR white (one
 * stop). Deliberately conservative — overshooting the real headroom lets the
 * OS clip the brightest detail, undershooting only leaves range unused.
 * DisplayHDR 400 monitors at a ~200-240 nit SDR white have about 1.7-2x.
 */
export const DEFAULT_HDR_HEADROOM = 2;

/** Clamp a headroom into `[HDR_HEADROOM_MIN, HDR_HEADROOM_MAX]`. */
export function clampHdrHeadroom(value: number): number {
  return Math.min(HDR_HEADROOM_MAX, Math.max(HDR_HEADROOM_MIN, value));
}

/**
 * Present on `RendererCapabilities` only while the canvas really is an
 * extended-range (`rgba16float`, tone-mapping `extended`) WebGPU canvas.
 */
export interface HdrOutputState {
  /** Headroom the renderer started with (pinned, remembered, or guessed). */
  readonly headroom: number;
  /** True when `?hdr=<n>` / `LuxarAppOptions.hdrHeadroom` pinned it. */
  readonly pinned: boolean;
  /**
   * WebGL unmasked renderer string (names the Apple chip), probed once at
   * renderer setup for display identification; null when unavailable.
   */
  readonly gpuRenderer: string | null;
}
