/**
 * View-state document — what a share link needs to reproduce the screen: the
 * per-layer edits (`layers`, see `layers/layer-settings.ts`) and the camera
 * pose (`camera`, the same block `LuxarApp.getCameraPose()` returns). One JSON
 * object serves three routes, Neuroglancer-style: the URL fragment
 * `#!<url-encoded JSON>` the viewer rewrites as the user works and reads back
 * on load, and "Copy / Download view state" and "Load view state…" in the
 * Layers header menu. Only layer fields that differ from the authored scene
 * are recorded; the camera is always recorded once a scene is up, because the
 * pose is what a "looks the same on their screen" link is for.
 *
 * The published schema is `schemas/view-state.v1.schema.json`; a unit test
 * pins its layer field list to {@link LAYER_PATCH_FIELDS}.
 *
 * @module ui/view-state
 */

import type { LayerPatch } from '../core/app/embedder/events';
import type { CameraSnapshot } from '../core/app/snapshot/viewer-snapshot';
import { validateLayerPatch } from './layers/layer-settings';

export const VIEW_STATE_VERSION = 1;

/** The fragment is the whole document: `#!` + `encodeURIComponent(JSON)`. */
const HASH_PREFIX = '#!';

export interface ViewStateDoc {
  version: typeof VIEW_STATE_VERSION;
  /** Keyed by scene-graph path (`LayerSummary.path`). Absent when nothing differs. */
  layers?: Record<string, LayerPatch>;
  /** Camera pose; absent before a scene is loaded. */
  camera?: CameraSnapshot;
}

/** `true` when the document records nothing worth a fragment. */
export function isEmptyViewState(doc: ViewStateDoc): boolean {
  return doc.camera === undefined && Object.keys(doc.layers ?? {}).length === 0;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const vec3 = (v: unknown): v is [number, number, number] =>
  Array.isArray(v) && v.length === 3 && v.every(finite);

function validateCamera(cam: unknown): CameraSnapshot {
  if (!isRecord(cam)) throw new Error('View state: "camera" is not an object');
  for (const key of ['position', 'target', 'up'] as const) {
    if (!vec3(cam[key])) throw new Error(`View state: camera.${key} is not a 3-vector`);
  }
  if (typeof cam.isOrtho !== 'boolean') throw new Error('View state: camera.isOrtho is not a boolean');
  for (const key of ['near', 'far'] as const) {
    if (!finite(cam[key])) throw new Error(`View state: camera.${key} is not a number`);
  }
  for (const key of ['fov', 'zoom'] as const) {
    if (cam[key] !== undefined && !finite(cam[key])) {
      throw new Error(`View state: camera.${key} is not a number`);
    }
  }
  return cam as unknown as CameraSnapshot;
}

/**
 * Parse and validate a document. Throws an `Error` whose message names the
 * offending field — it is shown to the user as-is. A layer-settings-only file
 * (`{ version, layers }`) is a valid view state.
 */
export function parseViewState(text: string): ViewStateDoc {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error('View state: not valid JSON');
  }
  if (!isRecord(raw)) throw new Error('View state: not an object');
  if (raw.version !== VIEW_STATE_VERSION) {
    throw new Error(`View state: unsupported version ${String(raw.version)}`);
  }
  const doc: ViewStateDoc = { version: VIEW_STATE_VERSION };
  if (raw.layers !== undefined) {
    if (!isRecord(raw.layers)) throw new Error('View state: "layers" is not an object');
    const layers: Record<string, LayerPatch> = {};
    for (const [path, patch] of Object.entries(raw.layers)) {
      layers[path] = validateLayerPatch(path, patch);
    }
    doc.layers = layers;
  }
  if (raw.camera !== undefined) doc.camera = validateCamera(raw.camera);
  return doc;
}

/** The document carried in a `location.hash`, or `null` if absent or unreadable. */
export function readViewStateHash(hash: string | undefined | null): ViewStateDoc | null {
  if (!hash?.startsWith(HASH_PREFIX)) return null;
  try {
    return parseViewState(decodeURIComponent(hash.slice(HASH_PREFIX.length)));
  } catch {
    return null;
  }
}

/** The fragment for `doc`, or `''` when it records nothing. */
export function writeViewStateHash(doc: ViewStateDoc): string {
  return isEmptyViewState(doc) ? '' : `${HASH_PREFIX}${encodeURIComponent(JSON.stringify(doc))}`;
}

/** The slice of `window` the URL writer needs; injectable for tests. */
export interface ViewStateUrlWindow {
  location: { pathname: string; search: string; hash?: string };
  history: { replaceState(data: unknown, unused: string, url?: string | null): void };
}

/** What the writer observes: change notifications and the current document. */
interface ViewStateSource {
  onChange(listener: () => void): () => void;
  getViewState(): ViewStateDoc;
}

function replaceHash(win: ViewStateUrlWindow, nextHash: string): void {
  const { pathname, search } = win.location;
  if (nextHash === (win.location.hash ?? '')) return;
  try {
    win.history.replaceState(null, '', `${pathname}${search}${nextHash}`);
  } catch {
    // Sandboxed iframe: the URL is a convenience, never a requirement.
  }
}

/** Drop the fragment (a dataset switch must not carry state across). */
export function clearViewStateHash(win: ViewStateUrlWindow): void {
  replaceHash(win, '');
}

/**
 * Keep the fragment in step with the source. Writes `delayMs` after the last
 * change (a slider drag or an orbit notifies per tick, and browsers rate-limit
 * `replaceState`) and skips writes that would not change the URL. Returns a
 * stop function.
 */
export function startViewStateUrlSync(
  source: ViewStateSource,
  win: ViewStateUrlWindow,
  delayMs = 300
): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const write = (): void => {
    timer = null;
    replaceHash(win, writeViewStateHash(source.getViewState()));
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
