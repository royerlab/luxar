/**
 * Layer-settings document — the per-layer appearance fields a user changed
 * away from the scene's authored defaults, as JSON.
 *
 * One shape serves three routes: the `#layers=<url-encoded JSON>` fragment
 * the viewer keeps up to date as the user edits (so the address bar is a
 * share link), "Copy / Download layer settings" and "Load layer settings…"
 * in the Layers header menu. Only differences are recorded, so a scene with
 * hundreds of untouched layers encodes to nothing. The published schema is
 * `schemas/layer-settings.v1.schema.json`; a unit test pins it to
 * {@link LAYER_PATCH_FIELDS}.
 *
 * Per-layer values are {@link LayerPatch} — the same fields, same key, same
 * apply route as `LuxarApp.setLayer()`.
 *
 * @module ui/layers/layer-settings
 */

import type { LayerPatch, LayerSummary } from '../../core/app/embedder/events';
import { BLENDING_MODES } from '../../types/blending';

export const LAYER_SETTINGS_VERSION = 1;

/** Hash-fragment key: `#layers=<encoded JSON>`. */
const HASH_KEY = 'layers';

export interface LayerSettingsDoc {
  version: typeof LAYER_SETTINGS_VERSION;
  /** Keyed by scene-graph path (`LayerSummary.path`). */
  layers: Record<string, LayerPatch>;
}

const EMPTY: LayerSettingsDoc = { version: LAYER_SETTINGS_VERSION, layers: {} };

type FieldCheck = (value: unknown) => boolean;
const finite: FieldCheck = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * One value check per {@link LayerPatch} field. Typed exhaustively so adding
 * a field to `LayerPatch` fails to compile until it is described here — and
 * the schema-parity test then fails until the JSON schema lists it too.
 */
export const LAYER_PATCH_FIELDS: Record<keyof LayerPatch, FieldCheck> = {
  visible: (v) => typeof v === 'boolean',
  opacity: finite,
  gamma: finite,
  absorption: finite,
  gain: finite,
  displayRange: (v) => Array.isArray(v) && v.length === 2 && v.every(finite),
  colormap: (v) => v === null || typeof v === 'string',
  blendingMode: (v) => (BLENDING_MODES as readonly unknown[]).includes(v),
  layerOrder: (v) => v === null || Number.isInteger(v),
};

const FIELD_KEYS = Object.keys(LAYER_PATCH_FIELDS) as (keyof LayerPatch)[];

/**
 * The fields of `current` that differ from `authored`, per layer. Layers the
 * authored scene lacks are ignored; identical layers are omitted entirely.
 */
export function diffLayerSettings(
  current: LayerSummary[],
  authored: LayerSummary[]
): LayerSettingsDoc {
  const base = new Map(authored.map((l) => [l.path, l]));
  const layers: Record<string, LayerPatch> = {};
  for (const live of current) {
    const fresh = base.get(live.path);
    if (!fresh) continue;
    const patch: Record<string, unknown> = {};
    for (const key of FIELD_KEYS) {
      const value = live[key];
      if (value !== undefined && JSON.stringify(value) !== JSON.stringify(fresh[key])) {
        patch[key] = value;
      }
    }
    if (Object.keys(patch).length > 0) layers[live.path] = patch as LayerPatch;
  }
  return { version: LAYER_SETTINGS_VERSION, layers };
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Parse and validate a document. Throws an `Error` whose message names the
 * offending layer and field — it is shown to the user as-is.
 */
export function parseLayerSettings(text: string): LayerSettingsDoc {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error('Layer settings: not valid JSON');
  }
  if (!isRecord(raw)) throw new Error('Layer settings: not an object');
  if (raw.version !== LAYER_SETTINGS_VERSION) {
    throw new Error(`Layer settings: unsupported version ${String(raw.version)}`);
  }
  if (!isRecord(raw.layers)) throw new Error('Layer settings: missing "layers" object');
  const layers: Record<string, LayerPatch> = {};
  for (const [path, patch] of Object.entries(raw.layers)) {
    layers[path] = validateLayerPatch(path, patch);
  }
  return { version: LAYER_SETTINGS_VERSION, layers };
}

function validateLayerPatch(path: string, patch: unknown): LayerPatch {
  if (!isRecord(patch)) throw new Error(`Layer settings: layer '${path}' is not an object`);
  for (const [key, value] of Object.entries(patch)) {
    const check = FIELD_KEYS.includes(key as keyof LayerPatch)
      ? LAYER_PATCH_FIELDS[key as keyof LayerPatch]
      : undefined;
    if (!check) throw new Error(`Layer settings: '${path}'.${key} is not a layer setting`);
    if (!check(value)) throw new Error(`Layer settings: '${path}'.${key} has an invalid value`);
  }
  return patch as LayerPatch;
}

/** The document carried in a `location.hash`, or `null` if absent or unreadable. */
export function readLayerSettingsHash(hash: string): LayerSettingsDoc | null {
  const text = new URLSearchParams(hash.replace(/^#/, '')).get(HASH_KEY);
  if (text === null) return null;
  try {
    return parseLayerSettings(text);
  } catch {
    return null;
  }
}

/**
 * `hash` with the `layers` key set to `doc` (or removed when `doc` records no
 * changes). Other hash keys are preserved. Returns `''` for an empty hash.
 */
export function writeLayerSettingsHash(hash: string, doc: LayerSettingsDoc): string {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  if (Object.keys(doc.layers).length > 0) params.set(HASH_KEY, JSON.stringify(doc));
  else params.delete(HASH_KEY);
  const out = params.toString();
  return out ? `#${out}` : '';
}

/** The slice of `window` the URL writer needs; injectable for tests. */
export interface LayerSettingsUrlWindow {
  location: { pathname: string; search: string; hash?: string };
  history: { replaceState(data: unknown, unused: string, url?: string | null): void };
}

/** What the writer observes: change notifications and the current document. */
interface LayerSettingsSource {
  onChange(listener: () => void): () => void;
  getLayerSettings(): LayerSettingsDoc;
}

function replaceHash(win: LayerSettingsUrlWindow, nextHash: string): void {
  const { pathname, search } = win.location;
  if (nextHash === currentHash(win)) return;
  try {
    win.history.replaceState(null, '', `${pathname}${search}${nextHash}`);
  } catch {
    // Sandboxed iframe: the URL is a convenience, never a requirement.
  }
}

const currentHash = (win: LayerSettingsUrlWindow): string => win.location.hash ?? '';

/** Drop the `layers` key (a dataset switch must not carry settings across). */
export function clearLayerSettingsHash(win: LayerSettingsUrlWindow): void {
  replaceHash(win, writeLayerSettingsHash(currentHash(win), EMPTY));
}

/**
 * Keep `#layers=` in step with the source. Writes `delayMs` after the last
 * change (a slider drag notifies per tick, and browsers rate-limit
 * `replaceState`) and skips writes that would not change the URL. Returns a
 * stop function.
 */
export function startLayerSettingsUrlSync(
  source: LayerSettingsSource,
  win: LayerSettingsUrlWindow,
  delayMs = 300
): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const write = (): void => {
    timer = null;
    replaceHash(win, writeLayerSettingsHash(currentHash(win), source.getLayerSettings()));
  };
  const off = source.onChange(() => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(write, delayMs);
  });
  return () => {
    off();
    if (timer !== null) clearTimeout(timer);
  };
}
