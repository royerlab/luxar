/**
 * Unit tests for the layer-settings document: the "only what the user
 * changed" diff, its JSON parse/validate, the `#layers=` hash codec, the
 * debounced URL writer, and parity between the runtime field table and the
 * published JSON schema.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect, vi } from 'vitest';
import type { LayerSummary } from '../../../../core/app/embedder/events';
import { BLENDING_MODES } from '../../../../types/blending';
import {
  LAYER_PATCH_FIELDS,
  LAYER_SETTINGS_VERSION,
  diffLayerSettings,
  parseLayerSettings,
  readLayerSettingsHash,
  writeLayerSettingsHash,
  startLayerSettingsUrlSync,
  clearLayerSettingsHash,
  type LayerSettingsDoc,
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

describe('parseLayerSettings', () => {
  const doc: LayerSettingsDoc = {
    version: 1,
    layers: { c0: { visible: false, blendingMode: 'max', colormap: 'viridis', layerOrder: null } },
  };

  it('round-trips a document through JSON', () => {
    expect(parseLayerSettings(JSON.stringify(doc))).toEqual(doc);
  });

  it.each([
    ['not json', 'not valid JSON'],
    ['[]', 'not an object'],
    ['{"version":2,"layers":{}}', 'version 2'],
    ['{"version":1}', '"layers"'],
    ['{"version":1,"layers":{"c0":3}}', "'c0'"],
    ['{"version":1,"layers":{"c0":{"gamma":"1"}}}', "'c0'.gamma"],
    ['{"version":1,"layers":{"c0":{"displayRange":[1]}}}', 'displayRange'],
    ['{"version":1,"layers":{"c0":{"blendingMode":"foo"}}}', 'blendingMode'],
    ['{"version":1,"layers":{"c0":{"colormap":3}}}', 'colormap'],
    ['{"version":1,"layers":{"c0":{"bogus":1}}}', "'c0'.bogus"],
  ])('rejects %s', (text, fragment) => {
    expect(() => parseLayerSettings(text)).toThrow(fragment);
  });
});

describe('hash codec', () => {
  const doc: LayerSettingsDoc = { version: 1, layers: { c0: { gamma: 1.4 } } };

  it('writes next to other hash keys and reads back the same document', () => {
    const hash = writeLayerSettingsHash('#foo=1', doc);
    expect(hash.startsWith('#')).toBe(true);
    expect(hash).toContain('foo=1');
    expect(readLayerSettingsHash(hash)).toEqual(doc);
  });

  it('removes the key for an empty document, leaving the rest of the hash', () => {
    const hash = writeLayerSettingsHash(writeLayerSettingsHash('#foo=1', doc), {
      version: 1,
      layers: {},
    });
    expect(hash).toBe('#foo=1');
    expect(
      writeLayerSettingsHash(writeLayerSettingsHash('', doc), { version: 1, layers: {} })
    ).toBe('');
  });

  it('returns null for an absent or malformed key', () => {
    expect(readLayerSettingsHash('')).toBeNull();
    expect(readLayerSettingsHash('#foo=1')).toBeNull();
    expect(readLayerSettingsHash('#layers=%7Bnope')).toBeNull();
    expect(readLayerSettingsHash('#layers=%7B%22version%22%3A9%7D')).toBeNull();
  });
});

describe('clearLayerSettingsHash', () => {
  it('drops only the layers key and does not touch an already-clean URL', () => {
    const win = {
      location: { pathname: '/v/', search: '?src=a', hash: '#foo=1&layers=%7B%7D' },
      history: { replaceState: vi.fn() },
    };
    clearLayerSettingsHash(win);
    expect(win.history.replaceState).toHaveBeenCalledWith(null, '', '/v/?src=a#foo=1');
    win.location.hash = '#foo=1';
    clearLayerSettingsHash(win);
    expect(win.history.replaceState).toHaveBeenCalledOnce();
  });
});

describe('startLayerSettingsUrlSync', () => {
  it('writes the diff into the hash once, after the changes settle', () => {
    vi.useFakeTimers();
    try {
      const listeners = new Set<() => void>();
      let current = [summary('c0')];
      const win = {
        location: { pathname: '/v/', search: '?src=a', hash: '' },
        history: {
          replaceState: vi.fn((_s: unknown, _t: string, url: string) => {
            win.location.hash = url.slice(url.indexOf('#'));
          }),
        },
      };
      const stop = startLayerSettingsUrlSync(
        {
          onChange: (l) => {
            listeners.add(l);
            return () => listeners.delete(l);
          },
          getLayerSettings: () => diffLayerSettings(current, AUTHORED),
        },
        win,
        300
      );
      current = [summary('c0', { gamma: 1.2 })];
      listeners.forEach((l) => l());
      current = [summary('c0', { gamma: 1.4 })];
      listeners.forEach((l) => l());
      vi.advanceTimersByTime(299);
      expect(win.history.replaceState).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(win.history.replaceState).toHaveBeenCalledOnce();
      expect(win.history.replaceState.mock.calls[0][2]).toMatch(/^\/v\/\?src=a#layers=/);
      expect(readLayerSettingsHash(win.location.hash)).toEqual({
        version: 1,
        layers: { c0: { gamma: 1.4 } },
      });

      // A change that leaves the document identical does not rewrite the URL.
      listeners.forEach((l) => l());
      vi.advanceTimersByTime(300);
      expect(win.history.replaceState).toHaveBeenCalledOnce();

      stop();
      current = [summary('c0')];
      listeners.forEach((l) => l());
      vi.advanceTimersByTime(300);
      expect(win.history.replaceState).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('JSON schema parity', () => {
  const schema = JSON.parse(
    readFileSync(resolve(__dirname, '../../../../../schemas/layer-settings.v1.schema.json'), 'utf8')
  );

  it('describes exactly the fields the runtime validator accepts', () => {
    const layer = schema.$defs.layer;
    expect(Object.keys(layer.properties).sort()).toEqual(Object.keys(LAYER_PATCH_FIELDS).sort());
    expect(layer.additionalProperties).toBe(false);
    expect(layer.properties.blendingMode.enum).toEqual([...BLENDING_MODES]);
    expect(schema.properties.version.const).toBe(LAYER_SETTINGS_VERSION);
    expect(schema.required).toEqual(['version', 'layers']);
  });
});
