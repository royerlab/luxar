import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const manifest = JSON.parse(readFileSync(new URL('./gate-scenes.json', import.meta.url), 'utf8'));
const generator = readFileSync(new URL('./generate_gate_scenes.py', import.meta.url), 'utf8');

describe('render-gate scene manifest', () => {
  it('references only generated stores across all suites', () => {
    const sceneNames = generator.match(/SCENE_NAMES = \(\n([\s\S]*?)\n\)/)?.[1];
    expect(sceneNames).toBeDefined();
    const heavySceneNames = generator.match(/HEAVY_SCENE_NAMES = \(\n([\s\S]*?)\n\)/)?.[1];
    expect(heavySceneNames).toBeDefined();
    const generated = [sceneNames, heavySceneNames].flatMap((names) =>
      [...names.matchAll(/^\s+"([^"]+)",?$/gm)].map((match) => match[1])
    );
    expect(generated.length).toBeGreaterThan(0);

    const stores = [
      ...manifest.exact,
      ...manifest.perf,
      ...Object.values(manifest.suites).flatMap((suite) => suite.cases),
    ]
      .filter((scene) => scene.store !== undefined)
      .map((scene) => scene.store);
    expect(stores.length).toBeGreaterThan(0);
    for (const store of stores) {
      expect(store).toMatch(/^gate\/.+\.luxar\.zarr(\.zip)?$/);
      expect(generated).toContain(store.slice(5).replace(/\.luxar\.zarr(\.zip)?$/, ''));
    }
  });

  it('has a unique id for every case', () => {
    const ids = [...manifest.exact, ...manifest.perf].map((scene) => scene.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
