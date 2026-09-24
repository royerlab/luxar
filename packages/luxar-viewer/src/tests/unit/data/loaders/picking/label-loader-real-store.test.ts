/** LabelLoader producer/consumer coverage against a real zarr fixture. */

import { FileSystemStore } from '@zarrita/storage';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { describe, expect, it } from 'vitest';
import { LabelLoader } from '../../../../../data/loaders';
import * as zarr from '../../../../../data/zarr';

const FIXTURES_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../../../../tests/fixtures'
);

async function loadLabels(fixture: string, nodePath: string, count: number) {
  const store = await zarr.openStore(new FileSystemStore(path.join(FIXTURES_DIR, fixture)));
  const loader = new LabelLoader(zarr.root(store));
  return Promise.all(Array.from({ length: count }, (_, index) => loader.getLabel(nodePath, index)));
}

describe('LabelLoader real-store CSR reads', () => {
  it('reads the first and last valid elements and rejects the first invalid index', async () => {
    const fixture = 'test_labelled_points.luxar.zarr';
    const labels = await loadLabels(fixture, '/labelled_points', 8);
    const store = await zarr.openStore(new FileSystemStore(path.join(FIXTURES_DIR, fixture)));
    const loader = new LabelLoader(zarr.root(store));

    expect(new Set(labels)).toEqual(new Set(Array.from({ length: 8 }, (_, index) => `Point ${index}`)));
    await expect(loader.getLabel('/labelled_points', 8)).resolves.toBeNull();
  });

  it('reads an empty label from the writer fixture as null', async () => {
    const labels = await loadLabels('test_linked_points.luxar.zarr', '/linked_points', 5);

    expect(labels.filter((label) => label === null)).toHaveLength(4);
    expect(labels.filter((label) => label !== null)).toEqual(['Linked point']);
  });
});
