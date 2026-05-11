/**
 * Unit tests for `simpleDimsToViewState`.
 *
 * Pure helper extracted from `zarr-loader.ts::updateSceneForDimensions`.
 * No mocks needed — the function only sees the SimpleDims input + the
 * supplied tolerance options. The previous integration test exercised
 * this same logic through the full SceneLoaderManager → SceneLoader stack
 * with a 110-line THREE mock and a 50-line manager mock; testing the
 * extracted helper directly is sharper and proves the contract without
 * any of the orchestration plumbing.
 */

import { describe, it, expect } from 'vitest';
import { simpleDimsToViewState } from '../../../../data/utils/dims-to-view-state';
import type { SimpleDims } from '../../../../types/dims';

const baseDims = (overrides: Partial<SimpleDims> = {}): SimpleDims => ({
  ndim: 4,
  displayed: [0, 1, 2],
  currentStep: [0, 0, 0, 5],
  metadata: [
    { name: 'x', unit: 'um', scale: 1, display: true },
    { name: 'y', unit: 'um', scale: 1, display: true },
    { name: 'z', unit: 'um', scale: 1, display: true },
    { name: 'time', unit: 's', scale: 1, display: false },
  ],
  ...overrides,
});

describe('simpleDimsToViewState', () => {
  it('copies displayed and currentStep arrays so callers cannot mutate the source', () => {
    const dims = baseDims();
    const vs = simpleDimsToViewState(dims, { maxRadius: 0.5, defaultTolerance: 0.1 });

    expect(vs.displayDims).toEqual([0, 1, 2]);
    expect(vs.slicePosition).toEqual([0, 0, 0, 5]);
    // Mutating the result must not bleed into the input.
    (vs.displayDims as number[]).push(99);
    (vs.slicePosition as number[]).push(99);
    expect(dims.displayed).toEqual([0, 1, 2]);
    expect(dims.currentStep).toEqual([0, 0, 0, 5]);
  });

  it('zeroes tolerance on displayed dimensions and uses maxRadius on continuous non-displayed', () => {
    const vs = simpleDimsToViewState(baseDims(), { maxRadius: 0.7, defaultTolerance: 0.1 });

    expect(vs.tolerance).toEqual([0, 0, 0, 0.7]);
  });

  it('uses 0.5 (not maxRadius) for discrete non-spatial non-displayed dims', () => {
    const dims: SimpleDims = {
      ndim: 5,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 5, 2],
      metadata: [
        { name: 'x', unit: 'um', scale: 1, display: true },
        { name: 'y', unit: 'um', scale: 1, display: true },
        { name: 'z', unit: 'um', scale: 1, display: true },
        { name: 'time', unit: 's', scale: 1, display: false },
        {
          name: 'channel',
          unit: '',
          scale: 1,
          display: false,
          discrete: true,
          spatial: false,
        },
      ],
    };

    const vs = simpleDimsToViewState(dims, { maxRadius: 2.0, defaultTolerance: 0.1 });

    expect(vs.tolerance[3]).toBe(2.0); // continuous → maxRadius
    expect(vs.tolerance[4]).toBe(0.5); // discrete + non-spatial → 0.5
  });

  it('treats discrete + spatial as continuous (maxRadius, not 0.5)', () => {
    const dims: SimpleDims = {
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 1],
      metadata: [
        { name: 'x', unit: 'um', scale: 1, display: true },
        { name: 'y', unit: 'um', scale: 1, display: true },
        { name: 'z', unit: 'um', scale: 1, display: true },
        // A spatial-discrete dim (e.g. integer-quantised z stack) keeps the
        // continuous treatment because geometry can still cross the slice.
        {
          name: 'z-stack',
          unit: 'um',
          scale: 1,
          display: false,
          discrete: true,
          spatial: true,
        },
      ],
    };

    const vs = simpleDimsToViewState(dims, { maxRadius: 0.3, defaultTolerance: 0.1 });
    expect(vs.tolerance[3]).toBe(0.3);
  });

  it('falls back to maxRadius when metadata is missing for a non-displayed dim', () => {
    const dims: SimpleDims = {
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 5],
      // metadata array shorter than ndim — dim 3 has no entry.
      metadata: [
        { name: 'x', unit: 'um', scale: 1, display: true },
        { name: 'y', unit: 'um', scale: 1, display: true },
        { name: 'z', unit: 'um', scale: 1, display: true },
      ],
    };

    const vs = simpleDimsToViewState(dims, { maxRadius: 0.4, defaultTolerance: 0.1 });
    expect(vs.tolerance[3]).toBe(0.4);
  });

  it('passes the SimpleDims metadata array through as ViewState.dimensions', () => {
    const dims = baseDims();
    const vs = simpleDimsToViewState(dims, { maxRadius: 0.5, defaultTolerance: 0.1 });
    expect(vs.dimensions).toBe(dims.metadata);
  });

  it('returns the same length tolerance array as ndim regardless of metadata size', () => {
    const dims: SimpleDims = {
      ndim: 6,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 0, 0, 0],
      metadata: [],
    };

    const vs = simpleDimsToViewState(dims, { maxRadius: 0.5, defaultTolerance: 0.1 });
    expect(vs.tolerance).toHaveLength(6);
    expect(vs.tolerance.slice(0, 3)).toEqual([0, 0, 0]);
    expect(vs.tolerance.slice(3)).toEqual([0.5, 0.5, 0.5]);
  });
});
