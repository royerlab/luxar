import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  discoverExampleDatasets,
  validateExampleDatasetReferences,
} from '../../../../tools/example-smoke-inventory';

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
    writeFileSync(join(root, 'stray_example.luxar.zarr'), 'not a dataset');

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

  it('explains how to generate examples when the directory is missing', () => {
    const missingRoot = examplesRoot();
    rmSync(missingRoot, { recursive: true });

    expect(() => discoverExampleDatasets(missingRoot, {})).toThrow(
      /Examples directory not found.*make run-examples/
    );
  });
});

describe('validateExampleDatasetReferences', () => {
  it('accepts covered and excluded datasets but rejects stale references', () => {
    const exclusions = {
      'excluded_example.luxar.zarr': 'Too large for the parallel smoke worker pool.',
    };

    expect(() =>
      validateExampleDatasetReferences(
        ['covered_example.luxar.zarr'],
        exclusions,
        ['covered_example.luxar.zarr', 'excluded_example.luxar.zarr'],
        'zero-points allowance'
      )
    ).not.toThrow();

    expect(() =>
      validateExampleDatasetReferences(
        ['covered_example.luxar.zarr'],
        exclusions,
        ['missing_example.luxar.zarr'],
        'zero-points allowance'
      )
    ).toThrow(/zero-points allowance.*missing_example.*not present/);
  });
});
