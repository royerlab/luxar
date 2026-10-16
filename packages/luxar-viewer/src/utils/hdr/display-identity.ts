/**
 * Which display is the viewer on, and how much HDR headroom does it have?
 * (PROTOTYPE, `?hdr`.)
 *
 * Two sources, best first:
 *
 * 1. **Measured.** Chromium's Window Management API carries the screen's live
 *    headroom on `ScreenDetailed` — `hdrHeadroom` in STOPS (the CSS HDR model)
 *    and the older `highDynamicRangeHeadroom` as a plain MULTIPLE (measured on
 *    a MacBook Pro: 2.667 and 6.35, and log2(6.35) = 2.667). Behind
 *    `--enable-experimental-web-platform-features` as of Chrome 155, and only
 *    after the `window-management` permission, which prompts — so it is used
 *    when already granted, or after an explicit "Detect Display" click.
 * 2. **Model knowledge.** Without a measurement, identify the panel from what
 *    the page can see: the display `label` (the monitor's EDID model name,
 *    e.g. `M28U`; permission-gated too), the GPU (WebGL's unmasked renderer
 *    names the Apple chip), the platform and the screen geometry, then look up
 *    a CONSERVATIVE headroom. Conservative because overshooting the real
 *    headroom lets the OS clip the brightest detail, while undershooting only
 *    leaves range unused — and no table knows the brightness setting, which
 *    moves macOS EDR headroom several-fold.
 *
 * @module utils/hdr/display-identity
 */

import { platformName } from '../platform';

/** What the page can observe about the display the window is on. */
export interface DisplaySignals {
  /** `'macOS'`, `'Windows'`, … (UA client hints, else `navigator.platform`). */
  readonly platform: string;
  /** WebGL unmasked renderer, e.g. `ANGLE (Apple, ANGLE Metal Renderer: Apple M4 Max, …)`. */
  readonly gpuRenderer: string | null;
  /** Screen size in CSS px and the device pixel ratio (current screen). */
  readonly width: number;
  readonly height: number;
  readonly dpr: number;
  /** `ScreenDetailed.label` / `isInternal` — only with the permission. */
  readonly label: string | null;
  readonly isInternal: boolean | null;
}

/** A model-table identification. */
export interface DisplayGuess {
  /** Human-readable name of what was identified. */
  readonly name: string;
  /** Conservative headroom (multiple of SDR white). */
  readonly headroom: number;
}

/**
 * Liquid Retina XDR panels (every Apple-silicon 14"/16" MacBook Pro): 1000
 * nits sustained / 1600 peak HDR. SDR white sits at 500-1000 nits depending
 * on brightness, so the true headroom ranges from ~1.6x at full brightness to
 * well over 6x when dimmed (6.35x measured at ordinary indoor brightness).
 * 3x keeps small highlights (the 1600-nit peak) unclipped at a 500-nit SDR
 * white without wasting most of the range.
 */
const MACBOOK_PRO_XDR: DisplayGuess = { name: 'MacBook Pro Liquid Retina XDR', headroom: 3 };

/**
 * Known external monitors by EDID model name (`ScreenDetailed.label`).
 * Seed entries only; extend as displays are verified. Values are
 * conservative for the panel's VESA tier at a ~200-nit SDR white.
 */
const EXTERNAL_DISPLAYS: Readonly<Record<string, DisplayGuess>> = {
  // 1600-nit peak, 500-nit SDR reference.
  'Pro Display XDR': { name: 'Apple Pro Display XDR', headroom: 3 },
  // DisplayHDR 400.
  M28U: { name: 'Gigabyte M28U (DisplayHDR 400)', headroom: 1.5 },
};

/**
 * macOS "looks like" sizes of the 14" MacBook Pro (3024x1964 panel at 2x),
 * from the scaled-mode list macOS ships for it. Needed only for base-chip
 * Macs: a Pro/Max chip already rules the MacBook Air out.
 */
const MACBOOK_PRO_14_SIZES: ReadonlyArray<readonly [number, number]> = [
  [1800, 1169],
  [1512, 982],
  [1352, 878],
  [1147, 745],
  [1024, 665],
];

/**
 * True when the screen has the MacBook panel shape: Retina-class, aspect ~1.54-1.55
 * (14"/16" Pro, 13"/15" Air). External monitors are almost all 16:9, 16:10
 * or ultrawide, which this excludes; a notched MacBook in a non-default
 * scaled mode keeps the panel aspect.
 */
function hasMacBookPanelShape(s: DisplaySignals): boolean {
  // `>= 1.5`, not `=== 2`: devicePixelRatio includes the page zoom (2.2 at
  // 110%), while the aspect ratio below is zoom-invariant.
  if (s.dpr < 1.5 || s.height <= 0) return false;
  const aspect = s.width / s.height;
  return aspect >= 1.535 && aspect <= 1.55;
}

function isAppleSilicon(s: DisplaySignals): boolean {
  return s.platform === 'macOS' && /\bApple M\d/.test(s.gpuRenderer ?? '');
}

/** Apple M-series Pro / Max / Ultra: never in a MacBook Air. */
function isProClassChip(s: DisplaySignals): boolean {
  return /\bApple M\d+ (Pro|Max|Ultra)\b/.test(s.gpuRenderer ?? '');
}

function isMacBookProXdr(s: DisplaySignals): boolean {
  if (!isAppleSilicon(s)) return false;
  // With the permission an internal panel is known outright; without it the
  // panel shape stands in (an external monitor on a MacBook fails it).
  if (!(s.isInternal ?? hasMacBookPanelShape(s))) return false;
  if (isProClassChip(s)) return true;
  return MACBOOK_PRO_14_SIZES.some(([w, h]) => w === s.width && h === s.height);
}

/** Identify the display from the model table, or null when unknown. */
export function identifyDisplay(s: DisplaySignals): DisplayGuess | null {
  if (s.label !== null && Object.hasOwn(EXTERNAL_DISPLAYS, s.label)) {
    return EXTERNAL_DISPLAYS[s.label];
  }
  return isMacBookProXdr(s) ? MACBOOK_PRO_XDR : null;
}

/**
 * The live headroom a `ScreenDetailed` reports, as a multiple of SDR white,
 * or null when the browser exposes none. Prefers `hdrHeadroom` (stops).
 */
export function measuredHeadroom(screen: unknown): number | null {
  if (screen === null || typeof screen !== 'object') return null;
  const s = screen as { hdrHeadroom?: unknown; highDynamicRangeHeadroom?: unknown };
  if (typeof s.hdrHeadroom === 'number' && Number.isFinite(s.hdrHeadroom)) {
    return 2 ** Math.max(0, s.hdrHeadroom);
  }
  if (typeof s.highDynamicRangeHeadroom === 'number' && s.highDynamicRangeHeadroom > 0) {
    return Math.max(1, s.highDynamicRangeHeadroom);
  }
  return null;
}

/**
 * Signals for the window's current screen. `gpuRenderer` comes from the
 * renderer setup (`probeUnmaskedGpuRenderer`); `detailed` (a `ScreenDetailed`)
 * adds label/isInternal.
 */
export function collectDisplaySignals(
  gpuRenderer: string | null,
  detailed: { label?: unknown; isInternal?: unknown } | null = null
): DisplaySignals {
  return {
    platform: platformName(),
    gpuRenderer,
    width: window.screen.width,
    height: window.screen.height,
    dpr: window.devicePixelRatio,
    label: typeof detailed?.label === 'string' && detailed.label !== '' ? detailed.label : null,
    isInternal: typeof detailed?.isInternal === 'boolean' ? detailed.isInternal : null,
  };
}

/** Stable per-display key: the label when known, else the screen geometry. */
export function displayKey(s: DisplaySignals): string {
  // No devicePixelRatio in the key: it moves with the page zoom.
  return s.label !== null ? `label:${s.label}` : `screen:${s.width}x${s.height}`;
}
