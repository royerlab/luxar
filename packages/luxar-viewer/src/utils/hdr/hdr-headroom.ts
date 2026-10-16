/**
 * HDR output headroom (PROTOTYPE, `?hdr`): how far above SDR white the
 * extended-range canvas may go, as a multiple of SDR white.
 *
 * Resolved per DISPLAY, best source first:
 *
 *   url       `?hdr=<n>` pins it for the session
 *   manual    a value set with the Headroom slider, remembered per display
 *   measured  the browser's live `ScreenDetailed` headroom (see display-identity)
 *   model     a conservative guess from the identified display model
 *   default   {@link DEFAULT_HDR_HEADROOM}
 *
 * plus `sdr` (1x) while the window sits on a display that does not report
 * `(dynamic-range: high)` — e.g. dragged onto an external monitor whose HDR
 * mode is off. The {@link HdrHeadroomController} re-resolves when the window
 * changes display or the measured headroom changes (brightness).
 *
 * @module utils/hdr/hdr-headroom
 */

import { getParam, URL_PARAM_KEYS } from '../../config/url-params';
import { StorageKeys } from '../storage-keys';
import {
  collectDisplaySignals,
  displayKey,
  identifyDisplay,
  measuredHeadroom,
  type DisplaySignals,
} from './display-identity';

/**
 * Fallback headroom when nothing identifies the display: 2x SDR white (one
 * stop). Deliberately conservative — overshooting makes the OS clip the
 * brightest detail, undershooting only leaves headroom unused. DisplayHDR 400
 * monitors at a ~200-240 nit SDR white have about 1.7-2x.
 */
export const DEFAULT_HDR_HEADROOM = 2;

/** Slider range for the headroom control. */
export const HDR_HEADROOM_MIN = 1;
export const HDR_HEADROOM_MAX = 16;

/** Where a resolved headroom came from. */
export type HdrHeadroomSource = 'url' | 'manual' | 'measured' | 'model' | 'default' | 'sdr';

/** A resolved headroom and its provenance. */
export interface HdrHeadroomResolution {
  readonly value: number;
  readonly source: HdrHeadroomSource;
  /** What the value describes, e.g. `MacBook Pro Liquid Retina XDR`. */
  readonly display: string;
}

function clampHeadroom(value: number): number {
  return Math.min(HDR_HEADROOM_MAX, Math.max(HDR_HEADROOM_MIN, value));
}

// ---------------------------------------------------------------------------
// Per-display manual values: localStorage JSON `{ [displayKey]: number }`.
// ---------------------------------------------------------------------------

/** Keep only well-formed `{ key: positive number }` entries. */
function sanitizeManualMap(parsed: unknown): Record<string, number> {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (typeof v === 'number' && Number.isFinite(v) && v > 0) out[k] = clampHeadroom(v);
  }
  return out;
}

function loadManualMap(): Record<string, number> {
  try {
    const raw = globalThis.localStorage?.getItem(StorageKeys.hdrHeadroom);
    return raw ? sanitizeManualMap(JSON.parse(raw)) : {};
  } catch {
    return {};
  }
}

function saveManualMap(map: Record<string, number>): void {
  try {
    globalThis.localStorage?.setItem(StorageKeys.hdrHeadroom, JSON.stringify(map));
  } catch {
    // Storage disabled (private mode, sandboxed iframe): per-session only.
  }
}

/** The headroom the user set by hand for display `key`, or null. */
export function loadManualHeadroom(key: string): number | null {
  return loadManualMap()[key] ?? null;
}

/** Remember (or with null, forget) the hand-set headroom for display `key`. */
export function storeManualHeadroom(key: string, value: number | null): void {
  const map = loadManualMap();
  if (value === null) delete map[key];
  else map[key] = clampHeadroom(value);
  saveManualMap(map);
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/** The explicit `?hdr=<n>` value, or null for a bare `?hdr` / no flag. */
export function urlHdrHeadroom(search?: string): number | null {
  const raw = search ?? (typeof window !== 'undefined' ? window.location.search : '');
  const value = getParam(new URLSearchParams(raw), URL_PARAM_KEYS.hdr);
  if (value === null) return null;
  const n = Number.parseFloat(value);
  return Number.isFinite(n) && n > 0 ? clampHeadroom(n) : null;
}

function displayReportsHdr(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(dynamic-range: high)').matches;
}

/**
 * Resolve the headroom for the display described by `signals`.
 * `measured` is the live `ScreenDetailed` headroom when available.
 */
export function resolveHdrHeadroomFor(
  signals: DisplaySignals,
  measured: number | null,
  urlValue: number | null
): HdrHeadroomResolution {
  const guess = identifyDisplay(signals);
  const display = guess?.name ?? signals.label ?? 'unidentified display';
  if (urlValue !== null) return { value: urlValue, source: 'url', display };
  if (!displayReportsHdr()) return { value: 1, source: 'sdr', display };
  const manual = loadManualHeadroom(displayKey(signals));
  if (manual !== null) return { value: manual, source: 'manual', display };
  if (measured !== null) return { value: clampHeadroom(measured), source: 'measured', display };
  if (guess !== null) return { value: guess.headroom, source: 'model', display };
  return { value: DEFAULT_HDR_HEADROOM, source: 'default', display };
}

/**
 * Synchronous best effort for renderer construction (no permission query):
 * URL, then the hand-set value for this screen's geometry, then the model
 * guess, then the default. The controller refines it once the UI is up.
 */
export function resolveHdrHeadroom(urlValue: number | null): number {
  try {
    return resolveHdrHeadroomFor(collectDisplaySignals(), null, urlValue).value;
  } catch {
    return urlValue ?? DEFAULT_HDR_HEADROOM;
  }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/** The Window Management API surface used here (Chromium). */
interface ScreenDetailedLike extends EventTarget {
  readonly label?: string;
  readonly isInternal?: boolean;
}
interface ScreenDetailsLike extends EventTarget {
  readonly currentScreen: ScreenDetailedLike;
}
type GetScreenDetails = () => Promise<ScreenDetailsLike>;

function getScreenDetailsFn(): GetScreenDetails | null {
  const fn = (window as Window & { getScreenDetails?: GetScreenDetails }).getScreenDetails;
  return typeof fn === 'function' ? fn.bind(window) : null;
}

/**
 * `window.screen` as an event target. Chromium fires `change` on it when the
 * window moves to another display (part of the Window Management API); the
 * DOM lib types `Screen` without EventTarget.
 */
function screenEvents(): EventTarget | null {
  const target = window.screen as unknown as Partial<EventTarget> | undefined;
  return typeof target?.addEventListener === 'function' ? (target as EventTarget) : null;
}

async function windowManagementGranted(): Promise<boolean> {
  try {
    const status = await navigator.permissions.query({
      name: 'window-management' as PermissionName,
    });
    return status.state === 'granted';
  } catch {
    return false;
  }
}

/**
 * Keeps the HDR headroom right for whatever display the window is on.
 * `apply` receives every changed resolution (value or provenance).
 */
export class HdrHeadroomController {
  private details: ScreenDetailsLike | null = null;
  private watchedScreen: ScreenDetailedLike | null = null;
  private last: HdrHeadroomResolution | null = null;
  private disposed = false;
  private readonly urlValue: number | null;
  private readonly hdrQuery: MediaQueryList;
  private readonly onChange = (): void => {
    this.resolve();
  };
  private readonly onScreenSwitch = (): void => {
    this.watchCurrentScreen();
    this.resolve();
  };

  constructor(
    private readonly apply: (r: HdrHeadroomResolution) => void,
    urlValue: number | null = urlHdrHeadroom()
  ) {
    this.urlValue = urlValue;
    this.hdrQuery = window.matchMedia('(dynamic-range: high)');
  }

  /** Resolve now, then attach to screen details if the permission is already granted. */
  async start(): Promise<void> {
    screenEvents()?.addEventListener('change', this.onChange);
    this.hdrQuery.addEventListener('change', this.onChange);
    this.resolve();
    if (getScreenDetailsFn() !== null && (await windowManagementGranted())) {
      await this.attachScreenDetails();
    }
  }

  /**
   * User-initiated (may show the permission prompt): measure the display,
   * forget any hand-set value for it, and re-resolve.
   */
  async detect(): Promise<HdrHeadroomResolution> {
    if (this.details === null) await this.attachScreenDetails();
    storeManualHeadroom(displayKey(this.signals()), null);
    this.last = null;
    return this.resolve();
  }

  /** The user moved the slider: remember it for the current display. */
  setManual(value: number): void {
    storeManualHeadroom(displayKey(this.signals()), value);
    this.resolve();
  }

  /** True when the browser can measure headroom (API present; may need permission). */
  canMeasure(): boolean {
    return getScreenDetailsFn() !== null;
  }

  dispose(): void {
    this.disposed = true;
    screenEvents()?.removeEventListener('change', this.onChange);
    this.hdrQuery.removeEventListener('change', this.onChange);
    this.details?.removeEventListener('currentscreenchange', this.onScreenSwitch);
    this.watchedScreen?.removeEventListener('hdrheadroomchange', this.onChange);
  }

  private async attachScreenDetails(): Promise<void> {
    const getDetails = getScreenDetailsFn();
    if (getDetails === null) return;
    try {
      this.details = await getDetails();
    } catch {
      return; // Permission denied: stay on the model guess.
    }
    if (this.disposed) return;
    this.details.addEventListener('currentscreenchange', this.onScreenSwitch);
    this.watchCurrentScreen();
    this.resolve();
  }

  private watchCurrentScreen(): void {
    this.watchedScreen?.removeEventListener('hdrheadroomchange', this.onChange);
    this.watchedScreen = this.details?.currentScreen ?? null;
    this.watchedScreen?.addEventListener('hdrheadroomchange', this.onChange);
  }

  private signals(): DisplaySignals {
    return collectDisplaySignals(this.details?.currentScreen ?? null);
  }

  private resolve(): HdrHeadroomResolution {
    const measured = measuredHeadroom(this.details?.currentScreen ?? null);
    const r = resolveHdrHeadroomFor(this.signals(), measured, this.urlValue);
    const changed =
      this.last === null ||
      Math.abs(this.last.value - r.value) > 1e-3 ||
      this.last.source !== r.source ||
      this.last.display !== r.display;
    if (changed && !this.disposed) {
      this.last = r;
      this.apply(r);
    }
    return r;
  }
}
