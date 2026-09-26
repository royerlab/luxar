/**
 * Unit tests for the layer-settings block of the view state: the "only what
 * the user changed" diff and parity between the runtime field table and the
 * published JSON schema. The document codec and URL writer are tested in
 * `../view-state.test.ts`.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import type { LayerSummary } from '../../../../core/app/embedder/events';
import { BLENDING_MODES } from '../../../../types/blending';
import {
  LAYER_PATCH_FIELDS,
  LAYER_SETTINGS_VERSION,
  diffLayerSettings,
} from '../../../../ui/layers/layer-settings';

function summary(path: string, over: Partial<LayerSummary> = {}): LayerSummary {
  return {
    path,
    name: path.split('/').pop() ?? path,
    type: 'gsplats',
    visible: true,
    opacity: 1,
    gamma: 1,
    displayRange: [0, 1000],
    dataRange: [0, 1000],
    colormap: null,
    supportsColormap: true,
    blendingMode: 'additive',
    absorption: 0,
    layerOrder: null,
    ...over,
  };
}

const AUTHORED = [summary('c0'), summary('c1'), summary('/story/hum', { type: 'sound', gain: 1 })];

describe('diffLayerSettings', () => {
  it('is empty when nothing differs from the authored state', () => {
    expect(diffLayerSettings(AUTHORED, AUTHORED)).toEqual({ version: 1, layers: {} });
  });

  it('emits only the changed fields, keyed by path', () => {
    const current = [
      summary('c0', { gamma: 1.4, displayRange: [10, 500] }),
      summary('c1'),
      summary('/story/hum', { type: 'sound', gain: 0.5 }),
    ];
    expect(diffLayerSettings(current, AUTHORED)).toEqual({
      version: 1,
      layers: {
        c0: { gamma: 1.4, displayRange: [10, 500] },
        '/story/hum': { gain: 0.5 },
      },
    });
  });

  it('records a released colormap / order as null and an added colormap as its name', () => {
    const authored = [summary('c0', { colormap: 'viridis', layerOrder: 10 })];
    const current = [summary('c0', { colormap: null, layerOrder: null })];
    expect(diffLayerSettings(current, authored).layers).toEqual({
      c0: { colormap: null, layerOrder: null },
    });
    expect(diffLayerSettings(authored, current).layers).toEqual({
      c0: { colormap: 'viridis', layerOrder: 10 },
    });
  });

  it('ignores a layer the authored scene does not have', () => {
    expect(diffLayerSettings([summary('ghost', { gamma: 2 })], AUTHORED).layers).toEqual({});
  });
});

describe('JSON schema parity', () => {
  const schema = JSON.parse(
    readFileSync(resolve(__dirname, '../../../../../schemas/view-state.v1.schema.json'), 'utf8')
  );

  it('describes exactly the fields the runtime validator accepts', () => {
    const layer = schema.$defs.layer;
    expect(Object.keys(layer.properties).sort()).toEqual(Object.keys(LAYER_PATCH_FIELDS).sort());
    expect(layer.additionalProperties).toBe(false);
    expect(layer.properties.blendingMode.enum).toEqual([...BLENDING_MODES]);
    expect(schema.properties.version.const).toBe(LAYER_SETTINGS_VERSION);
    expect(schema.required).toEqual(['version']);
    expect(Object.keys(schema.properties).sort()).toEqual(['camera', 'layers', 'version']);
  });
});
