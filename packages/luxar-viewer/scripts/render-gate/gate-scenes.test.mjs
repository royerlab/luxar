import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

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
    const zipScenes = generator.match(/^ZIP_SCENES = frozenset\(\{(.+)\}\)$/m)?.[1];
    expect(zipScenes).toBeDefined();
    const zipped = new Set([...zipScenes.matchAll(/"([^"]+)"/g)].map((match) => match[1]));
    const generatedStores = new Set(
      generated.map((name) => `gate/${name}.luxar.zarr${zipped.has(name) ? '.zip' : ''}`)
    );

    const suites = Object.values(manifest.suites ?? {}).flatMap((suite) => suite.cases ?? []);
    const stores = [...manifest.exact, ...manifest.perf, ...suites]
      .filter((scene) => scene.store !== undefined)
      .map((scene) => scene.store);
    expect(stores.length).toBeGreaterThan(0);
    for (const store of stores) expect(generatedStores.has(store)).toBe(true);
  });

  it('has a unique id for every case', () => {
    const ids = [...manifest.exact, ...manifest.perf].map((scene) => scene.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
