import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { discoverExampleDatasets } from '../../../../tools/example-smoke-inventory';

const temporaryDirectories: string[] = [];

function examplesRoot(): string {
  const root = mkdtempSync(join(tmpdir(), 'luxar-example-smoke-'));
  temporaryDirectories.push(root);
  return root;
}

function addDataset(root: string, name: string): void {
  mkdirSync(join(root, name), { recursive: true });
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe('discoverExampleDatasets', () => {
  it('includes every generated dataset except explicit exclusions', () => {
    const root = examplesRoot();
    addDataset(root, 'alpha_example.luxar.zarr');
    addDataset(root, 'beta_example.luxar.zarr');
    addDataset(root, 'ignored-directory');

    expect(
      discoverExampleDatasets(root, {
        'beta_example.luxar.zarr': 'Too large for the parallel smoke worker pool.',
      })
    ).toEqual(['alpha_example.luxar.zarr']);
  });

  it('rejects exclusions without a generated dataset or reason', () => {
    const root = examplesRoot();
    addDataset(root, 'alpha_example.luxar.zarr');

    expect(() =>
      discoverExampleDatasets(root, {
        'missing_example.luxar.zarr': 'Known upstream failure.',
      })
    ).toThrow(/missing_example.*not present/);
    expect(() =>
      discoverExampleDatasets(root, {
        'alpha_example.luxar.zarr': '   ',
      })
    ).toThrow(/alpha_example.*reason/);
  });
});
