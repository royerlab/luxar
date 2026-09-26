import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const manifest = JSON.parse(readFileSync(new URL('./gate-scenes.json', import.meta.url), 'utf8'));
const generator = readFileSync(new URL('./generate_gate_scenes.py', import.meta.url), 'utf8');

describe('render-gate scene manifest', () => {
  it('names every generated store and no others', () => {
    const sceneNames = generator.match(/SCENE_NAMES = \(\n([\s\S]*?)\n\)/)?.[1];
    expect(sceneNames).toBeDefined();
    const generated = [...sceneNames.matchAll(/^\s+"([^"]+)",?$/gm)].map((match) => match[1]);
    expect(generated.length).toBeGreaterThan(0);

    const stores = [...manifest.exact, ...manifest.perf]
      .filter((scene) => scene.store !== undefined)
      .map((scene) => scene.store);
    expect(stores.length).toBeGreaterThan(0);
    for (const store of stores) expect(store).toMatch(/^gate\/.+\.luxar\.zarr$/);
    expect(new Set(stores.map((store) => store.slice(5, -'.luxar.zarr'.length)))).toEqual(
      new Set(generated)
    );
  });

  it('has a unique id for every case', () => {
    const ids = [...manifest.exact, ...manifest.perf].map((scene) => scene.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
