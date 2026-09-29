import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const manifest = JSON.parse(readFileSync(new URL('./gate-scenes.json', import.meta.url), 'utf8'));
const generator = readFileSync(new URL('./generate_gate_scenes.py', import.meta.url), 'utf8');

describe('render-gate scene manifest', () => {
  it('has a writer for every referenced store, including suite and heavy cases', () => {
    const names = ['SCENE_NAMES', 'HEAVY_SCENE_NAMES'].flatMap((group) => {
      const entries = generator.match(new RegExp(`^${group} = \\(\\n([\\s\\S]*?)\\n\\)`, 'm'))?.[1];
      expect(entries).toBeDefined();
      return [...entries.matchAll(/^\s+"([^"]+)",?$/gm)].map((match) => match[1]);
    });
    expect(names.length).toBeGreaterThan(0);
    expect(new Set(names).size).toBe(names.length);
    const writtenStores = new Set(
      names.map((name) => `gate/${name}.luxar.zarr${name.startsWith('zip_') ? '.zip' : ''}`)
    );
    const originalStores = [
      'mixed',
      'env_splats',
      'partition_normal',
      'glass',
      'lod_ladder',
      'tiny_units_ortho',
    ];
    const directStores = [...manifest.exact, ...manifest.perf]
      .filter((scene) => scene.store !== undefined)
      .map((scene) => scene.store);
    const suiteStores = Object.values(manifest.suites).flatMap((suite) =>
      suite.cases.filter((scene) => scene.store !== undefined).map((scene) => scene.store)
    );
    for (const store of [...directStores, ...suiteStores])
      expect(writtenStores.has(store)).toBe(true);
    for (const name of originalStores) expect(directStores).toContain(`gate/${name}.luxar.zarr`);
  });

  it('has a unique id for every case', () => {
    const ids = [...manifest.exact, ...manifest.perf].map((scene) => scene.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
