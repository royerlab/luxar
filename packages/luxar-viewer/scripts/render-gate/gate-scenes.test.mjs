import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import { judgeMetric, METRIC_DIRECTIONS, validateMetricDirections } from './suites.mjs';

const manifest = JSON.parse(readFileSync(new URL('./gate-scenes.json', import.meta.url), 'utf8'));
const generator = readFileSync(new URL('./generate_gate_scenes.py', import.meta.url), 'utf8');

describe('render-gate scene manifest', () => {
  it('references only generated stores with the correct archive extension', () => {
    const sceneLists = [
      ...generator.matchAll(/^(?:SCENE_NAMES|HEAVY_SCENE_NAMES) = \(\n([\s\S]*?)\n\)/gm),
    ];
    expect(sceneLists).toHaveLength(2);
    const generated = sceneLists.flatMap((list) =>
      [...list[1].matchAll(/^\s+"([^"]+)",?$/gm)].map((match) => match[1])
    );
    expect(generated.length).toBeGreaterThan(0);
    expect(new Set(generated).size).toBe(generated.length);
    const zipScenes = generator.match(/^ZIP_SCENES = frozenset\(\{(.+)\}\)$/m)?.[1];
    expect(zipScenes).toBeDefined();
    const zipped = new Set([...zipScenes.matchAll(/"([^"]+)"/g)].map((match) => match[1]));
    const generatedStores = new Set(
      generated.map((name) => `gate/${name}.luxar.zarr${zipped.has(name) ? '.zip' : ''}`)
    );

    const directStores = [...manifest.exact, ...manifest.perf]
      .filter((scene) => scene.store !== undefined)
      .map((scene) => scene.store);
    const suiteStores = Object.values(manifest.suites ?? {}).flatMap((suite) =>
      (suite.cases ?? []).filter((scene) => scene.store !== undefined).map((scene) => scene.store)
    );
    const stores = [...directStores, ...suiteStores];
    expect(stores.length).toBeGreaterThan(0);
    for (const store of stores) expect(generatedStores.has(store)).toBe(true);
    // The six original exact/perf stores stay referenced directly.
    for (const name of [
      'mixed',
      'env_splats',
      'partition_normal',
      'glass',
      'lod_ladder',
      'tiny_units_ortho',
    ])
      expect(directStores).toContain(`gate/${name}.luxar.zarr`);
  });

  it('has a unique id for every case', () => {
    const ids = [...manifest.exact, ...manifest.perf].map((scene) => scene.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('judges dragFrames as better: higher (more rAF callbacks = a freer main thread)', () => {
    // `dragFrames` counts the animation frames that fired during the scrub
    // drag; a build that frees the main thread fires MORE of them. Judged
    // lower-is-better, an improvement from 39 to 63 frames read as a FAIL.
    const drag = manifest.suites.playback.cases.find((c) => c.id === 'scrub_tp50_drag');
    const m = drag.metrics.find((x) => x.name === 'dragFrames');
    expect(m).toMatchObject({ better: 'higher' });
    const arm = (...frames) => frames.map((dragFrames) => ({ dragFrames }));
    const judged = judgeMetric(m, {
      base: arm(39, 40, 38),
      base2: arm(39, 38, 40),
      cand: arm(63, 62, 64),
    });
    expect(judged.verdict).toBe('win');
  });

  it('declares every suite metric in its known direction', () => {
    expect(() => validateMetricDirections(manifest.suites)).not.toThrow();
    // Every metric the shipped manifest gates on has its direction pinned, so
    // a new metric must state which way is better before it can be declared.
    for (const suite of Object.values(manifest.suites)) {
      for (const c of suite.cases) {
        for (const m of c.metrics) expect(METRIC_DIRECTIONS).toHaveProperty([m.of ?? m.name]);
      }
    }
  });

  it('covers the authored opening camera with a cold load that sets no harness pose', () => {
    // `cold_sp64_closeup` applies its pose from the harness AFTER navigation,
    // so the initial load still sees the default camera. The authored case's
    // close-up lives in the STORE (viewer_config.camera), which the viewer
    // frames before any node loads; a harness pose would mask exactly that.
    const hosted = manifest.suites.hosted.cases;
    const c = hosted.find((x) => x.id === 'cold_sp64_closeup_authored');
    expect(c).toMatchObject({
      store: 'gate/sp64_closeup_authored.luxar.zarr',
      workload: { kind: 'coldLoad' },
    });
    expect(c.pose).toBeUndefined();
    const names = c.metrics.filter((m) => !m.report).map((m) => m.name);
    expect(names).toEqual(
      expect.arrayContaining(['partition.partsInitialised', 'requestsToFirstFrame'])
    );
    for (const name of ['partition.partsInitialised', 'requestsToFirstFrame'])
      expect(c.metrics.find((m) => m.name === name)).toMatchObject({ better: 'lower' });
  });
});
