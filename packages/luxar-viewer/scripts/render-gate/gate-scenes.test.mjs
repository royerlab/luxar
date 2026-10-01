import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const manifest = JSON.parse(readFileSync(new URL('./gate-scenes.json', import.meta.url), 'utf8'));
const generator = readFileSync(new URL('./generate_gate_scenes.py', import.meta.url), 'utf8');

describe('render-gate scene manifest', () => {
  it('runs every default exact case on all three renderer paths', () => {
    expect(manifest.defaults.backends).toEqual(['webgl', 'webgpu', 'webgpu-gl']);
    for (const scene of manifest.exact) {
      expect(scene.backends ?? manifest.defaults.backends).toContain('webgpu-gl');
    }
    expect(manifest.perfDefaults.backends).toEqual(['webgl', 'webgpu']);
  });

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
});
