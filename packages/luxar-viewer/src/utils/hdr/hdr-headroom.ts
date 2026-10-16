/**
 * HDR output headroom (PROTOTYPE, `?hdr`): how far above SDR white the
 * extended-range canvas may go, as a multiple of SDR white.
 *
 * Resolved per DISPLAY, best source first:
 *
 *   url       pinned by `?hdr=<n>` / `LuxarAppOptions.hdrHeadroom`
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

import { DEFAULT_HDR_HEADROOM, clampHdrHeadroom } from '../../types/hdr-output';
import { EventGroup } from '../cross-layer/event-group';
import { log, Modules } from '../log';
import { StorageKeys } from '../storage-keys';
import {
  collectDisplaySignals,
  displayKey,
  identifyDisplay,
  measuredHeadroom,
  type DisplaySignals,
} from './display-identity';
import { HDR_DISPLAY_QUERY } from './hdr-detection';

/** Where a resolved headroom came from. */
export type HdrHeadroomSource = 'url' | 'manual' | 'measured' | 'model' | 'default' | 'sdr';

/** A resolved headroom and its provenance. */
export interface HdrHeadroomResolution {
  readonly value: number;
  readonly source: HdrHeadroomSource;
  /** What the value describes, e.g. `MacBook Pro Liquid Retina XDR`. */
  readonly display: string;
}

/** True when the slider must not edit the value (pinned, or an SDR display). */
export function isHeadroomLocked(source: HdrHeadroomSource): boolean {
  return source === 'url' || source === 'sdr';
}

// ---------------------------------------------------------------------------
// Per-display manual values: localStorage JSON `{ [displayKey]: number }`.
// ---------------------------------------------------------------------------

/** Keep only well-formed `{ key: positive number }` entries. */
function sanitizeManualMap(parsed: unknown): Record<string, number> {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (typeof v === 'number' && Number.isFinite(v) && v > 0) out[k] = clampHdrHeadroom(v);
  }
  return out;
}

function loadManualMap(): Record<string, number> {
  try {
    if (typeof localStorage === 'undefined') return {};
    const raw = localStorage.getItem(StorageKeys.hdrHeadroom);
    return raw ? sanitizeManualMap(JSON.parse(raw)) : {};
  } catch {
    return {};
  }
}

function saveManualMap(map: Record<string, number>): void {
  try {
    if (typeof localStorage === 'undefined') return;
    localStorage.setItem(StorageKeys.hdrHeadroom, JSON.stringify(map));
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
  else map[key] = clampHdrHeadroom(value);
  saveManualMap(map);
}

/**
 * Once the display label is known, carry a value hand-set under the
 * geometry key (before the permission) over to the label key, so granting
 * the permission does not forget it.
 */
function migrateManualHeadroom(fromKey: string, toKey: string): void {
  if (fromKey === toKey) return;
  const map = loadManualMap();
  if (map[fromKey] === undefined || map[toKey] !== undefined) return;
  map[toKey] = map[fromKey];
  delete map[fromKey];
  saveManualMap(map);
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

function displayReportsHdr(): boolean {
  return typeof window !== 'undefined' && window.matchMedia(HDR_DISPLAY_QUERY).matches;
}

/**
 * Resolve the headroom for the display described by `signals`.
 * `measured` is the live `ScreenDetailed` headroom when available; `pinned`
 * the `?hdr=<n>` / `LuxarAppOptions.hdrHeadroom` value, already clamped.
 */
export function resolveHdrHeadroomFor(
  signals: DisplaySignals,
  measured: number | null,
  pinned: number | null
): HdrHeadroomResolution {
  const guess = identifyDisplay(signals);
  const display = guess?.name ?? signals.label ?? 'unidentified display';
  if (pinned !== null) return { value: pinned, source: 'url', display };
  if (!displayReportsHdr()) return { value: 1, source: 'sdr', display };
  const manual = loadManualHeadroom(displayKey(signals));
  if (manual !== null) return { value: manual, source: 'manual', display };
  if (measured !== null) {
    return { value: clampHdrHeadroom(measured), source: 'measured', display };
  }
  if (guess !== null) return { value: guess.headroom, source: 'model', display };
  return { value: DEFAULT_HDR_HEADROOM, source: 'default', display };
}

/**
 * Synchronous best effort for renderer construction (no permission query):
 * the hand-set value for this screen's geometry, then the model guess, then
 * the default. The controller refines it once the UI is up.
 */
export function resolveHdrHeadroom(gpuRenderer: string | null): number {
  try {
    return resolveHdrHeadroomFor(collectDisplaySignals(gpuRenderer), null, null).value;
  } catch {
    return DEFAULT_HDR_HEADROOM;
  }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/** The slice of the Window Management API's `ScreenDetailed` used here (Chromium). */
export interface ScreenDetailedLike extends EventTarget {
  readonly label?: string;
  readonly isInternal?: boolean;
}

/** The slice of the Window Management API's `ScreenDetails` used here (Chromium). */
export interface ScreenDetailsLike extends EventTarget {
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
 * DOM lib types `Screen` without EventTarget, and jsdom's is not one.
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

/** What the renderer setup decided, handed to the controller explicitly. */
export interface HdrHeadroomControllerOptions {
  /** Pinned headroom (`?hdr=<n>` / `LuxarAppOptions.hdrHeadroom`), or null. */
  readonly pinned: number | null;
  /** WebGL unmasked renderer string from the renderer setup, or null. */
  readonly gpuRenderer: string | null;
}

/** Outcome of an explicit "Detect Display" request. */
export type HdrDetectOutcome = 'measured' | 'unmeasurable' | 'denied';

/**
 * Keeps the HDR headroom right for whatever display the window is on.
 * `apply` receives every changed resolution (value or provenance).
 */
export class HdrHeadroomController {
  private readonly events = new EventGroup();
  private details: ScreenDetailsLike | null = null;
  private stopWatchingScreen: (() => void) | null = null;
  private last: HdrHeadroomResolution | null = null;
  private disposed = false;
  private readonly onChange = (): void => {
    this.resolve();
  };
  private readonly onScreenSwitch = (): void => {
    this.watchCurrentScreen();
    this.resolve();
  };

  constructor(
    private readonly apply: (r: HdrHeadroomResolution) => void,
    private readonly options: HdrHeadroomControllerOptions
  ) {}

  /** Resolve now, then attach to screen details if the permission is already granted. */
  async start(): Promise<void> {
    const screenTarget = screenEvents();
    if (screenTarget) this.events.on(screenTarget, 'change', this.onChange);
    this.events.on(window.matchMedia(HDR_DISPLAY_QUERY), 'change', this.onChange);
    this.resolve();
    if (getScreenDetailsFn() !== null && (await windowManagementGranted())) {
      await this.attachScreenDetails();
    }
  }

  /**
   * User-initiated (may show the permission prompt): measure the display
   * and, only when a measurement is available, forget any hand-set value for
   * it so the measurement takes over.
   */
  async detect(): Promise<HdrDetectOutcome> {
    if (this.details === null) await this.attachScreenDetails();
    if (this.details === null) return 'denied';
    if (measuredHeadroom(this.details.currentScreen) === null) return 'unmeasurable';
    storeManualHeadroom(displayKey(this.signals()), null);
    this.resolve();
    return 'measured';
  }

  /** The user moved the slider: remember it for the current display. */
  setManual(value: number): void {
    if (this.last !== null && isHeadroomLocked(this.last.source)) return;
    storeManualHeadroom(displayKey(this.signals()), value);
    this.resolve();
  }

  /** The current resolution (null before {@link start}). */
  current(): HdrHeadroomResolution | null {
    return this.last;
  }

  /** True when the browser can measure headroom (API present; may need permission). */
  canMeasure(): boolean {
    return getScreenDetailsFn() !== null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.events.dispose();
  }

  private async attachScreenDetails(): Promise<void> {
    const getDetails = getScreenDetailsFn();
    if (getDetails === null) return;
    const geometryKey = displayKey(this.signals());
    try {
      this.details = await getDetails();
    } catch {
      log.info(Modules.HDR, 'Display details unavailable (permission not granted)');
      return;
    }
    if (this.disposed) return;
    migrateManualHeadroom(geometryKey, displayKey(this.signals()));
    this.events.on(this.details, 'currentscreenchange', this.onScreenSwitch);
    this.watchCurrentScreen();
    this.resolve();
  }

  private watchCurrentScreen(): void {
    this.stopWatchingScreen?.();
    const screen = this.details?.currentScreen;
    this.stopWatchingScreen = screen
      ? this.events.on(screen, 'hdrheadroomchange', this.onChange)
      : null;
  }

  private signals(): DisplaySignals {
    return collectDisplaySignals(this.options.gpuRenderer, this.details?.currentScreen ?? null);
  }

  private resolve(): HdrHeadroomResolution {
    const measured = measuredHeadroom(this.details?.currentScreen ?? null);
    const r = resolveHdrHeadroomFor(this.signals(), measured, this.options.pinned);
    const changed =
      this.last === null ||
      Math.abs(this.last.value - r.value) > 1e-3 ||
      this.last.source !== r.source ||
      this.last.display !== r.display;
    if (changed && !this.disposed) {
      this.last = r;
      log.info(Modules.HDR, `Headroom ${r.value.toFixed(2)}x (${r.source}) for ${r.display}`);
      this.apply(r);
    }
    return r;
  }
}
