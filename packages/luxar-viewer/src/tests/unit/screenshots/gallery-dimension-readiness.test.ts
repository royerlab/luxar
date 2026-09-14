import * as fs from 'fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  hasNonZeroDimensionStep,
  isGalleryDataReady,
  readBakedDimensionStep,
} from '../../../tests/screenshots/gallery-dimension-readiness';

function response(ok: boolean, body: unknown) {
  return { ok, json: async () => body };
}

afterEach(() => {
  delete (globalThis as any).__luxarDebug;
});

describe('gallery baked dimension metadata', () => {
  it('reads the authored step from format-3 root attributes', async () => {
    const fetchRoot = vi.fn().mockResolvedValue(
      response(true, {
        zarr_format: 3,
        attributes: { viewer_config: { dimensions: { current_step: [0, 0, 0, 4] } } },
      })
    );

    await expect(readBakedDimensionStep('http://data/scene.zarr/', fetchRoot)).resolves.toEqual([
      0, 0, 0, 4,
    ]);
    expect(fetchRoot).toHaveBeenCalledWith('http://data/scene.zarr/zarr.json');
  });

  it('falls back to format-2 attributes', async () => {
    const fetchRoot = vi
      .fn()
      .mockResolvedValueOnce(response(false, null))
      .mockResolvedValueOnce(
        response(true, { viewer_config: { dimensions: { current_step: [0, 3] } } })
      );

    await expect(readBakedDimensionStep('http://data/scene.zarr', fetchRoot)).resolves.toEqual([
      0, 3,
    ]);
    expect(fetchRoot.mock.calls.map(([url]) => url)).toEqual([
      'http://data/scene.zarr/zarr.json',
      'http://data/scene.zarr/.zattrs',
    ]);
  });

  it('rejects a malformed authored step instead of silently skipping the gate', async () => {
    const fetchRoot = vi.fn().mockResolvedValue(
      response(true, {
        zarr_format: 3,
        attributes: { viewer_config: { dimensions: { current_step: [0, Number.NaN] } } },
      })
    );

    await expect(readBakedDimensionStep('http://data/scene.zarr', fetchRoot)).rejects.toThrow(
      'current_step must be an array of finite numbers'
    );
  });
});

describe('gallery data readiness', () => {
  function setState(state: unknown): void {
    (globalThis as any).__luxarDebug = { getState: () => state };
  }

  it('does not accept visible geometry from the wrong slice', () => {
    setState({ isLoading: false, totalElements: 100, dimensions: { currentStep: [0, 0] } });

    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(
      false
    );
  });

  it('accepts an idle matching slice with content', () => {
    setState({ isLoading: false, totalElements: 100, dimensions: { currentStep: [0, 4] } });

    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(true);
  });

  it('allows an empty matching opening slice when navigation will populate it', () => {
    setState({ isLoading: false, totalElements: 0, dimensions: { currentStep: [0, 4] } });

    expect(isGalleryDataReady({ requireElements: false, expectedDimensionStep: [0, 4] })).toBe(
      true
    );
  });

  it('still waits for loader idle after the step matches', () => {
    setState({ isLoading: true, totalElements: 100, dimensions: { currentStep: [0, 4] } });

    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(
      false
    );
  });

  it('detects whether the scene opens away from the all-zero default', () => {
    expect(hasNonZeroDimensionStep(null)).toBe(false);
    expect(hasNonZeroDimensionStep([0, 0, 0])).toBe(false);
    expect(hasNonZeroDimensionStep([0, 0, 4])).toBe(true);
  });
});

describe('gallery dimension readiness wiring', () => {
  const specPath = 'src/tests/screenshots/generate-gallery.spec.ts';

  it('gates the initial load on the scene-authored opening step', () => {
    const source = fs.readFileSync(specPath, 'utf-8');

    const readStep = source.indexOf('await readBakedDimensionStep(dataUrl)');
    const firstWait = source.indexOf('await waitForDataLoaded(', readStep);
    const navigation = source.indexOf('if (demo.dimensionNav)', firstWait);

    expect(readStep).toBeGreaterThan(-1);
    expect(firstWait).toBeGreaterThan(readStep);
    expect(navigation).toBeGreaterThan(firstWait);
    expect(source.slice(readStep, firstWait)).toContain(
      'expectedDimensionStep: bakedDimensionStep'
    );
    expect(
      source.slice(firstWait, navigation).match(/waitForDataLoaded\(page, initialLoadOptions\)/g)
    ).toHaveLength(2);
  });
});
