import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  describeGalleryDataState,
  isGalleryDataReady,
  readBakedDimensionStep,
} from '../../../tests/screenshots/gallery-dimension-readiness';

function response(ok: boolean, body: unknown) {
  return { ok, json: async () => body };
}

afterEach(() => {
  delete (globalThis as any).__luxarDebug;
  delete (globalThis as any).__luxarGalleryDimensionStepReached;
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

  it('returns no expected step when the scene does not author one', async () => {
    const fetchRoot = vi.fn().mockResolvedValue(
      response(true, {
        zarr_format: 3,
        attributes: { viewer_config: { camera: { fov: 50 } } },
      })
    );

    await expect(readBakedDimensionStep('http://data/scene.zarr', fetchRoot)).resolves.toBeNull();
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
    setState({
      isLoading: false,
      totalElements: 100,
      dimensions: { currentStep: [0, 4], displayed: [0] },
    });

    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(true);
  });

  it('stays ready after an animated scene advances past the authored step', () => {
    const state = {
      isLoading: true,
      totalElements: 100,
      dimensions: { currentStep: [0, 4], displayed: [0] },
    };
    setState(state);

    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(
      false
    );
    state.dimensions.currentStep = [0, 5];
    state.isLoading = false;
    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(true);
  });

  it('ignores displayed coordinates that the viewer clamps on load', () => {
    setState({
      isLoading: false,
      totalElements: 100,
      dimensions: { currentStep: [2, 4], displayed: [0] },
    });

    expect(isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(true);
  });

  it('matches only the dimension prefix authored by the scene', () => {
    setState({ isLoading: false, totalElements: 100, dimensions: { currentStep: [0, 4, 9] } });

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

  it('fails fast when the authored step has more dimensions than the scene', () => {
    setState({
      isLoading: false,
      totalElements: 100,
      dimensions: { currentStep: [0, 4], displayed: [0] },
    });

    expect(() =>
      isGalleryDataReady({ requireElements: true, expectedDimensionStep: [0, 4, 9] })
    ).toThrow('3 values, but the scene exposes 2 dimensions');
  });

  it('reports the expected and live state for timeout diagnostics', () => {
    setState({
      isLoading: true,
      totalElements: 0,
      dimensions: { currentStep: [0, 2], displayed: [0] },
    });

    expect(describeGalleryDataState({ requireElements: true, expectedDimensionStep: [0, 4] })).toBe(
      'expectedDimensionStep=[0,4], currentStep=[0,2], displayed=[0], isLoading=true, totalElements=0, expectedStepReached=false'
    );
  });
});
