/** Capture and restore a shareable view using the public app control surface. */
import type { LuxarApp } from '../app';
import type { ViewerSnapshot } from './snapshot/viewer-snapshot';
import type { LayerPatch } from './embedder/events';
import type { RenderingSettings } from '../../config';
import { buildViewBookmarkUrl, normalizeDataSourceUrl } from '../../config/url-params';

export interface ViewBookmark {
  version: 1;
  src: string;
  snapshot: ViewerSnapshot;
  rendering: Partial<RenderingSettings>;
  layers: Array<{ path: string; appearance: LayerPatch }>;
}

const MAX_BOOKMARK_LENGTH = 100_000;
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const vector3 = (value: unknown): boolean =>
  Array.isArray(value) && value.length === 3 && value.every(finite);
const positive = (value: unknown): boolean => finite(value) && value > 0;
const validRange = (value: unknown): boolean =>
  Array.isArray(value) && value.length === 2 && value.every(finite);

function validCamera(value: unknown): boolean {
  if (!record(value)) return false;
  return [
    [value.position, value.target, value.up].every(vector3),
    typeof value.isOrtho === 'boolean',
    validProjection(value),
  ].every(Boolean);
}

function validProjection(camera: Record<string, unknown>): boolean {
  if (!positive(camera.near) || !finite(camera.far)) return false;
  if (camera.far <= (camera.near as number)) return false;
  return camera.isOrtho === true
    ? positive(camera.zoom)
    : finite(camera.fov) && camera.fov > 0 && camera.fov < 180;
}

function validDims(value: unknown): boolean {
  if (!record(value)) return false;
  if (!Number.isSafeInteger(value.ndim) || (value.ndim as number) < 1) return false;
  if (!Array.isArray(value.displayed) || !Array.isArray(value.currentStep)) return false;
  return (
    value.currentStep.length === value.ndim &&
    value.currentStep.every(finite) &&
    value.displayed.every((i: unknown) => Number.isSafeInteger(i))
  );
}

function validAppearance(value: unknown): boolean {
  if (!record(value) || !finiteTree(value)) return false;
  return [
    typeof value.visible === 'boolean',
    [value.opacity, value.gamma, value.absorption].every(finite),
    validRange(value.displayRange),
    value.colormap === null || typeof value.colormap === 'string',
    typeof value.blendingMode === 'string',
    value.layerOrder === null || Number.isSafeInteger(value.layerOrder),
    value.gain === undefined || finite(value.gain),
  ].every(Boolean);
}

function validLayer(value: unknown): boolean {
  return record(value) && typeof value.path === 'string' && validAppearance(value.appearance);
}

function validBookmark(value: unknown): value is ViewBookmark {
  if (!record(value) || value.version !== 1) return false;
  if (typeof value.src !== 'string') return false;
  if (normalizeDataSourceUrl(value.src) !== value.src) return false;
  return [
    validSnapshot(value.snapshot),
    record(value.rendering) && finiteTree(value.rendering),
    Array.isArray(value.layers) && value.layers.every(validLayer),
  ].every(Boolean);
}

function validSnapshot(value: unknown): boolean {
  if (!record(value) || value.version !== 1) return false;
  if (!validCamera(value.camera)) return false;
  return value.dims === undefined || validDims(value.dims);
}

/** Snapshot only editable layer fields; diagnostics and authored bounds are not state. */
export function captureBookmark(app: LuxarApp): ViewBookmark {
  const src = normalizeDataSourceUrl(app.getViewerState().src);
  if (!src) throw new Error('Load a dataset before adding a bookmark');
  const snapshot = app.captureSnapshot();
  if (!snapshot.dims) throw new Error('Load a dataset before adding a bookmark');
  const rendering = app.getRenderingSettings();
  // With dynamic clipping, these are live readouts from the current pose,
  // not settings the user chose. Restoring them would briefly override the
  // recomputed planes and warn on every shared link.
  if (rendering.dynamicClippingEnabled) {
    delete (rendering as Partial<RenderingSettings>).near;
    delete (rendering as Partial<RenderingSettings>).far;
  }
  return {
    version: 1,
    src,
    snapshot,
    rendering,
    layers: app.getLayers().map((layer) => ({
      path: layer.path,
      appearance: {
        visible: layer.visible,
        opacity: layer.opacity,
        gamma: layer.gamma,
        displayRange: layer.displayRange,
        colormap: layer.colormap,
        blendingMode: layer.blendingMode,
        absorption: layer.absorption,
        layerOrder: layer.layerOrder,
        ...(layer.gain !== undefined ? { gain: layer.gain } : {}),
      },
    })),
  };
}

/** Validate URL input before it reaches camera, dimension, or layer setters. */
export function parseBookmark(raw: string | null): ViewBookmark | null {
  if (!raw || raw.length > MAX_BOOKMARK_LENGTH) return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  return validBookmark(value) ? value : null;
}

function finiteTree(value: unknown): boolean {
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(finiteTree);
  if (record(value)) return Object.values(value).every(finiteTree);
  return value === null || typeof value === 'string' || typeof value === 'boolean';
}

/** Preserve ordinary viewer options while replacing the dataset and view. */
export function buildBookmarkUrl(base: string, bookmark: ViewBookmark): string {
  const encoded = JSON.stringify(bookmark);
  if (encoded.length > MAX_BOOKMARK_LENGTH) throw new Error('Bookmark is too large to share');
  return buildViewBookmarkUrl(base, bookmark.src, encoded);
}

/** Dataset switches finish before the saved appearance and camera are applied. */
export async function restoreBookmark(app: LuxarApp, bookmark: ViewBookmark): Promise<void> {
  if (normalizeDataSourceUrl(app.getViewerState().src) !== bookmark.src) {
    await app.switchDataset(bookmark.src);
  }
  app.setRenderingSettings(bookmark.rendering);
  for (const layer of bookmark.layers) {
    if (app.getLayers().some((current) => current.path === layer.path)) {
      app.setLayer(layer.path, layer.appearance);
    }
  }
  app.restoreSnapshot(bookmark.snapshot);
}
