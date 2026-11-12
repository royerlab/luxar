/**
 * The gsplats data-unit display gain (`display_gain`, the Layers panel's
 * Brightness) and the data-unit labels of the display window (WP16).
 *
 * A converted scene keeps splat amplitudes in source data units and carries its
 * brightness as a separate per-node factor; the panel must show the window in
 * those units (with the unit and the background floor) and edit the gain as an
 * independent control. Old scenes, which carry none of the attrs, see the
 * identity everywhere.
 */
import { describe, expect, it, vi } from 'vitest';

import { composeAttrs } from '../../../../data/attrs-composer';
import type { SceneNode } from '../../../../data/data-loader-types';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { syncPickFromVisual } from '../../../../rendering/picking/picking-system/visibility-sync';
import { resolveColormapWindow } from '../../../../rendering/display-range';
import { liveLayerAttrs } from '../../../../ui/layers/attrs-utils';
import {
  BRIGHTNESS_FLOOR,
  brightnessSliderRange,
  brightnessTooltip,
  colourRangeTooltip,
  dataRangeLine,
  formatBrightness,
  formatDataValue,
  formatLegacyRangeValue,
  relativeBrightness,
  remapWindowForLayer,
  scalarRangeLabel,
} from '../../../../ui/layers/brightness-range';
import { LayerStateManager, computeUniforms } from '../../../../ui/layers/layer-state';
import { isControlVisible } from '../../../../ui/layers/layer-control-rules';
import {
  parseBookmark,
  captureBookmark,
  restoreBookmark,
} from '../../../../core/app/bookmark-state';
import type { LuxarApp } from '../../../../core/app';

const node = (
  path: string,
  type: string,
  attrs: Record<string, unknown>,
  children: SceneNode[] = []
): SceneNode =>
  ({ name: path.split('/').pop(), path, type, attrs, children }) as unknown as SceneNode;

const root = (...children: SceneNode[]): SceneNode => node('/', 'scene', {}, children);

function layersOf(graph: SceneNode): LayerStateManager {
  const state = new LayerStateManager();
  state.initFromSceneGraph(graph);
  return state;
}

/** A converted 12-bit fit: data units, a gain, a unit and a floor. */
const CONVERTED = {
  layer: true,
  colormap: 'gray',
  amplitude_data_range: [0, 4200],
  display_gain: 2.5e-4,
  amplitude_unit: 'counts above background',
  amplitude_offset: 112,
  source_data_range: [0, 12000],
};

describe('brightness-range helpers', () => {
  it('anchors a seven-decade log track on the decade of the live gain', () => {
    expect(brightnessSliderRange(2.5e-4)).toEqual({ min: 1e-7, max: 1 });
    const one = brightnessSliderRange(1);
    expect(one.min).toBeCloseTo(1e-3);
    expect(one.max).toBeCloseTo(1e4);
    // A junk value falls back to the identity's track rather than NaN bounds.
    expect(brightnessSliderRange(Number.NaN)).toEqual(brightnessSliderRange(1));
    expect(brightnessSliderRange(0)).toEqual(brightnessSliderRange(1));
  });

  it('formats the gain as a multiplier with three significant digits', () => {
    expect(formatBrightness(1)).toBe('×1');
    expect(formatBrightness(0.25)).toBe('×0.25');
    expect(formatBrightness(2.5e-4)).toBe('×2.50e-4');
    expect(formatBrightness(12345)).toBe('×1.23e+4');
    expect(formatBrightness(0)).toBe('×0');
  });

  it('formats data units with four significant digits and separators', () => {
    expect(formatDataValue(0.5)).toBe('0.5');
    expect(formatDataValue(0)).toBe('0');
    expect(formatDataValue(5.25212621689)).toBe('5.252');
    // The header value and the end label read the same, however the input snaps.
    expect(formatDataValue(208.912979126)).toBe('208.9');
    expect(formatDataValue(208.912)).toBe('208.9');
    expect(formatDataValue(4200)).toBe('4,200');
    expect(formatDataValue(65535.4)).toBe('65,535');
    expect(formatDataValue(2.5e-5)).toBe('2.50e-5');
    expect(formatDataValue(3.2e7)).toBe('3.20e+7');
  });

  it('puts the unit, data range and peak note in the label tooltip', () => {
    const labelled = scalarRangeLabel({
      amplitudeUnit: 'counts above background',
      amplitudeOffset: 405,
      sourceDataRange: [0, 1597.2],
    });
    expect(labelled.label).toBe('Splat amplitude');
    expect(labelled.tooltip.split('\n')).toEqual([
      'Unit: counts above background',
      'Data range: 0 \u2013 1,597 (floor 405)',
      "Each value is one splat's peak. Overlapping splats sum to the data value, " +
        'and the view also sums along each ray.',
    ]);
    // Floors are rounded with the same formatter as every other readout.
    expect(dataRangeLine({ sourceDataRange: [0, 12000.39], amplitudeOffset: 99.6083984375 })).toBe(
      'Data range: 0 \u2013 12,000 (floor 99.61)'
    );
    expect(dataRangeLine({ amplitudeOffset: 1 })).toBeUndefined();
    // No data range recorded: unit and note only.
    expect(scalarRangeLabel({ amplitudeUnit: 'a.u.' }).tooltip.split('\n')).toHaveLength(2);
    // No unit (every pre-WP16 scene): the historical label and text, untouched.
    expect(scalarRangeLabel({}).label).toBe('Display range');
    expect(scalarRangeLabel({}).tooltip).toBe(
      'Scalar data values in this range are mapped across the colormap.'
    );
  });

  it('appends the data range to a direct-colour tooltip only when recorded', () => {
    expect(colourRangeTooltip('base', {})).toBe('base');
    expect(colourRangeTooltip('base', { sourceDataRange: [0, 10], amplitudeUnit: 'counts' })).toBe(
      'base\nData range: 0 \u2013 10 counts'
    );
    // The unit sits with the numbers, before the floor's parenthesis.
    expect(
      colourRangeTooltip('base', {
        sourceDataRange: [0, 10],
        amplitudeUnit: 'counts',
        amplitudeOffset: 5,
      })
    ).toBe('base\nData range: 0 \u2013 10 counts (floor 5)');
  });
});

describe('relative controls', () => {
  it('shows Brightness relative to the authored gain, absolute in the tooltip', () => {
    const layer = { brightness: 5e-4, authoredBrightness: 2.5e-4 };
    expect(relativeBrightness(layer)).toBeCloseTo(2, 12);
    expect(brightnessTooltip(layer)).toContain('Absolute display gain: 0.0005');
    expect(relativeBrightness({ brightness: 1, authoredBrightness: 1 })).toBe(1);
  });

  it('re-expresses a multi-select window as fractions of each layer\u2019s range', () => {
    const primary = { scalarWindow: true, scalarDataRange: [0, 4000] as [number, number] };
    const other = { scalarWindow: true, scalarDataRange: [10, 20] as [number, number] };
    expect(remapWindowForLayer(primary, other, 1000, 3000)).toEqual([12.5, 17.5]);
    const colour = { scalarWindow: false, scalarDataRange: undefined };
    expect(remapWindowForLayer(primary, colour, 0, 2000)).toEqual([0, 0.5]);
    const degenerate = { scalarWindow: true, scalarDataRange: [5, 5] as [number, number] };
    expect(remapWindowForLayer(primary, degenerate, 1, 2)).toEqual([1, 2]);
  });

  it('keeps the historical readout for scenes without a unit', () => {
    expect(formatLegacyRangeValue(2.5e-5)).toBe('0');
    expect(formatLegacyRangeValue(4200.1234)).toBe('4200.123');
  });
});

describe('layer state', () => {
  it('opens a converted gsplats layer on its data-unit window, gain and unit', () => {
    const layer = layersOf(root(node('/g', 'gsplats', CONVERTED))).getLayer('/g')!;
    expect(layer.brightness).toBe(2.5e-4);
    expect(layer.authoredBrightness).toBe(2.5e-4);
    expect(layer.sourceDataRange).toEqual([0, 12000]);
    expect(layer.amplitudeUnit).toBe('counts above background');
    expect(layer.amplitudeOffset).toBe(112);
    expect([layer.displayMin, layer.displayMax]).toEqual([0, 4200]);
    expect(isControlVisible('brightness', layer)).toBe(true);
  });

  it('reads the gain off a partition wrapper and the unit off its parts', () => {
    const graph = root(
      node(
        '/tiles',
        'group',
        { layer: true, kind: 'partition', display_type: 'gsplats', display_gain: 0.01 },
        [
          node('/tiles/part_0', 'gsplats', {
            amplitude_data_range: [0, 900],
            amplitude_unit: 'counts',
            n_splats: 10,
          }),
          node('/tiles/part_1', 'gsplats', { amplitude_data_range: [0, 800], n_splats: 5 }),
        ]
      )
    );
    const layer = layersOf(graph).getLayer('/tiles')!;
    expect(layer.brightness).toBe(0.01);
    expect(layer.amplitudeUnit).toBe('counts');
    expect(layer.amplitudeOffset).toBeUndefined();
    expect(isControlVisible('brightness', layer)).toBe(true);
  });

  it('gives every other layer (and an old scene) the identity, hidden', () => {
    const state = layersOf(
      root(
        node('/old', 'gsplats', { layer: true, amplitude_data_range: [0, 1] }),
        node('/pts', 'points', { layer: true })
      )
    );
    const old = state.getLayer('/old')!;
    expect(old.brightness).toBe(1);
    expect(old.amplitudeUnit).toBeUndefined();
    expect(isControlVisible('brightness', old)).toBe(true);
    expect(isControlVisible('brightness', state.getLayer('/pts')!)).toBe(false);
  });

  it('ignores a non-positive or non-finite Brightness', () => {
    const state = layersOf(root(node('/g', 'gsplats', CONVERTED)));
    state.setBrightness('/g', 0);
    state.setBrightness('/g', -1);
    state.setBrightness('/g', Number.NaN);
    expect(state.getLayer('/g')!.brightness).toBe(2.5e-4);
    state.setBrightness('/g', 1e-3);
    expect(state.getLayer('/g')!.brightness).toBe(1e-3);
    expect(BRIGHTNESS_FLOOR).toBeGreaterThan(0);
  });

  it('emits the live gain as the layer’s own display_gain', () => {
    const layer = layersOf(root(node('/g', 'gsplats', CONVERTED))).getLayer('/g')!;
    expect(liveLayerAttrs(layer).display_gain).toBe(2.5e-4);
  });
});

describe('window in data units', () => {
  it('windows a 0–4200 amplitude range exactly, with no [0, 1] assumption', () => {
    const { intensity, offset } = computeUniforms(100, 4200);
    expect(intensity).toBeCloseTo(1 / 4100, 12);
    expect(offset).toBeCloseTo(-100 / 4100, 12);
    // Identity leaf/owner gain: the colormap window IS the data-unit range.
    expect(
      resolveColormapWindow([0, 4200], { intensity: 1, offset: 0 }, { intensity: 1, offset: 0 })
    ).toEqual([0, 4200]);
    // An authored data-unit window (convert restates --intensity as I·gain).
    const [lo, hi] = resolveColormapWindow(
      [0, 4200],
      { intensity: 1 / 2000, offset: 0 },
      { intensity: 1 / 2000, offset: 0 }
    );
    expect(lo).toBeCloseTo(0, 9);
    expect(hi).toBeCloseTo(2000, 6);
  });
});

describe('composition', () => {
  it('multiplies display_gain along the ancestry, identity 1', () => {
    expect(composeAttrs([]).display_gain).toBe(1);
    expect(composeAttrs([{ display_gain: 0.5 }, {}, { display_gain: 4 }]).display_gain).toBe(2);
    // A hand-edited non-finite gain must not NaN every amplitude.
    expect(composeAttrs([{ display_gain: Number.NaN }]).display_gain).toBe(1);
  });
});

describe('material plumbing', () => {
  it.each([
    ['GLSL', (g?: number) => new GSplatMaterial(g === undefined ? {} : { displayGain: g })],
    ['TSL', (g?: number) => new GSplatTSLMaterial(g === undefined ? {} : { displayGain: g })],
  ])('%s: uDisplayGain defaults to 1, takes the config, updates and clones', (_n, make) => {
    expect(make().uniforms.uDisplayGain.value).toBe(1);
    const material = make(2.5e-4);
    expect(material.getDisplayGain()).toBe(2.5e-4);
    material.updateDisplayGain(0.125);
    expect(material.uniforms.uDisplayGain.value).toBe(0.125);
    expect(material.clone().uniforms.uDisplayGain.value).toBe(0.125);
  });

  it('syncs the live gain onto the pick material', () => {
    const visual = new GSplatMaterial({ displayGain: 3e-3 });
    const pick = new GSplatPickingMaterial({ nodeId: 1 });
    expect(pick.uniforms.uDisplayGain.value).toBe(1);
    syncPickFromVisual(pick, visual);
    expect(pick.uniforms.uDisplayGain.value).toBe(3e-3);
  });
});

describe('bookmarks', () => {
  const app = (brightness: number, authoredBrightness = 2.5e-4) =>
    ({
      getViewerState: () => ({ src: '/s.zarr' }),
      captureSnapshot: () => ({
        version: 1,
        camera: {
          position: [1, 2, 3],
          target: [0, 0, 0],
          up: [0, 1, 0],
          isOrtho: false,
          fov: 45,
          near: 0.1,
          far: 100,
        },
        dims: { ndim: 3, displayed: [0, 1, 2], currentStep: [0, 0, 0] },
      }),
      getRenderingSettings: () => ({ dynamicClippingEnabled: false }),
      getLayers: () => [
        {
          path: 'g',
          name: 'g',
          type: 'gsplats',
          visible: true,
          opacity: 1,
          gamma: 1,
          displayRange: [0, 4200],
          dataRange: [0, 4200],
          colormap: 'gray',
          supportsColormap: true,
          blendingMode: 'additive',
          absorption: 1,
          brightness,
          authoredBrightness,
          layerOrder: null,
        },
      ],
    }) as unknown as LuxarApp;

  it('carries an edited Brightness through a share link', () => {
    const bookmark = captureBookmark(app(5e-4));
    expect(bookmark.layers[0].appearance.brightness).toBe(5e-4);
    expect(parseBookmark(JSON.stringify(bookmark))?.layers[0].appearance.brightness).toBe(5e-4);
  });

  it('keeps an edit back to exactly 1 on a converted layer (authored 2.5e-4)', () => {
    expect(captureBookmark(app(1)).layers[0].appearance.brightness).toBe(1);
  });

  it('omits the authored gain and refuses a non-positive gain', () => {
    const bookmark = captureBookmark(app(2.5e-4));
    expect(bookmark.layers[0].appearance).not.toHaveProperty('brightness');
    const bad = {
      ...bookmark,
      layers: [
        { ...bookmark.layers[0], appearance: { ...bookmark.layers[0].appearance, brightness: 0 } },
      ],
    };
    expect(parseBookmark(JSON.stringify(bad))).toBeNull();
  });

  it('restores an untouched bookmark after Brightness changes', async () => {
    const state = layersOf(root(node('/g', 'gsplats', CONVERTED)));
    const viewer = {
      ...app(2.5e-4),
      getLayers: () => [{ ...state.getLayer('/g')!, path: 'g' }],
      restoreSnapshot: vi.fn(),
      setRenderingSettings: vi.fn(),
      setLayer: vi.fn((_path: string, patch: { brightness?: number }) => {
        if (patch.brightness !== undefined) state.setBrightness('/g', patch.brightness);
      }),
      flyTo: vi.fn().mockResolvedValue({ completed: true }),
    } as unknown as LuxarApp;
    const bookmark = captureBookmark(viewer);
    expect(bookmark.layers[0].appearance).not.toHaveProperty('brightness');
    state.setBrightness('/g', 2.5e-3);
    await restoreBookmark(viewer, bookmark);
    expect(state.getLayer('/g')!.brightness).toBe(2.5e-4);
  });

  describe('content_hash', () => {
    const viewerOn = (
      hash: string | undefined,
      layer: Record<string, unknown> = {}
    ): { viewer: LuxarApp; setLayer: ReturnType<typeof vi.fn> } => {
      const setLayer = vi.fn();
      const base = app(2.5e-4);
      const viewer = {
        ...base,
        getViewerState: () => ({ src: '/s.zarr', contentHash: hash }),
        getLayers: () => [{ ...base.getLayers()[0], ...layer }],
        restoreSnapshot: vi.fn(),
        setRenderingSettings: vi.fn(),
        setLayer,
        flyTo: vi.fn().mockResolvedValue({ completed: true }),
      } as unknown as LuxarApp;
      return { viewer, setLayer };
    };

    it('stamps the scene hash and restores the window on the same build', async () => {
      const { viewer, setLayer } = viewerOn('abc', { brightness: 5e-4 });
      const bookmark = captureBookmark(viewer);
      expect(bookmark.contentHash).toBe('abc');
      await restoreBookmark(viewer, bookmark);
      expect(setLayer.mock.calls[0][1]).toMatchObject({
        displayRange: [0, 4200],
        brightness: 5e-4,
      });
    });

    it('skips the window and gain on another build of the scene', async () => {
      const { viewer: before } = viewerOn('old', { brightness: 5e-4 });
      const bookmark = captureBookmark(before);
      const { viewer: after, setLayer } = viewerOn('new');
      await restoreBookmark(after, bookmark);
      const patch = setLayer.mock.calls[0][1];
      expect(patch).not.toHaveProperty('displayRange');
      expect(patch.brightness).toBe(2.5e-4); // this build's own opening exposure
      expect(patch.opacity).toBe(1);
    });

    it('treats an unstamped bookmark on a data-unit layer as another build', async () => {
      const old = { ...captureBookmark(viewerOn(undefined).viewer) };
      delete (old as { contentHash?: string }).contentHash;
      const { viewer, setLayer } = viewerOn('new', { amplitudeUnit: 'counts' });
      await restoreBookmark(viewer, old);
      expect(setLayer.mock.calls[0][1]).not.toHaveProperty('displayRange');
    });

    it('restores an unstamped bookmark on an old scene exactly as before', async () => {
      const old = captureBookmark(viewerOn(undefined).viewer);
      const { viewer, setLayer } = viewerOn(undefined);
      await restoreBookmark(viewer, old);
      expect(setLayer.mock.calls[0][1]).toMatchObject({ displayRange: [0, 4200] });
    });

    it('refuses a non-string hash', () => {
      const bookmark = captureBookmark(viewerOn('abc').viewer);
      expect(parseBookmark(JSON.stringify({ ...bookmark, contentHash: 3 }))).toBeNull();
    });
  });
});
