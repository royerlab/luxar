/**
 * Layer-settings document — the per-layer appearance fields a user changed
 * away from the scene's authored defaults, as JSON.
 *
 * The `layers` block of the view-state document (`../view-state.ts`), which
 * is what the `#!` URL fragment and "Copy / Download / Load view state" carry.
 * Only differences are recorded, so a scene with hundreds of untouched layers
 * encodes to nothing. The published schema is
 * `schemas/view-state.v1.schema.json`; a unit test pins its layer fields to
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

export interface LayerSettingsDoc {
  version: typeof LAYER_SETTINGS_VERSION;
  /** Keyed by scene-graph path (`LayerSummary.path`). */
  layers: Record<string, LayerPatch>;
}

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
 * Validate one layer's patch. Throws an `Error` whose message names the
 * offending layer and field — it is shown to the user as-is.
 */
export function validateLayerPatch(path: string, patch: unknown): LayerPatch {
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
