import { describe, expect, it } from 'vitest';
import { planChunkBoundaryViewStates } from '../../../../../data/loaders';
import type { ViewState } from '../../../../../data/data-loader-types';

function view(time: number): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, time],
    tolerance: [0, 0, 0, 0],
  };
}

function index() {
  const bounds = new Float32Array(8 * 4 * 2);
  for (let atom = 0; atom < 8; atom++) {
    for (let dim = 0; dim < 4; dim++) {
      const offset = atom * 8 + dim * 2;
      bounds[offset] = dim === 3 ? atom : 0;
      bounds[offset + 1] = dim === 3 ? atom : 1;
    }
  }
  return {
    chunkBounds: bounds,
    chunkCount: 8,
    metadata: { ndim: 4, chunk_size: 100 },
  };
}

describe('planChunkBoundaryViewStates', () => {
  it('adds only the nearest future slice that enters a next zarr chunk', () => {
    const views = planChunkBoundaryViewStates(
      view(1),
      view(2),
      [{ start: 100, end: 200 }],
      index(),
      [
        { shape: [800, 3], chunks: [400, 3] },
        { shape: [800], chunks: [600] },
      ]
    );

    expect(views.map((candidate) => candidate.slicePosition[3])).toEqual([2, 4]);
  });

  it('finds previous chunk boundaries during reverse playback', () => {
    const views = planChunkBoundaryViewStates(
      view(6),
      view(5),
      [{ start: 600, end: 700 }],
      index(),
      [{ shape: [800], chunks: [400] }]
    );

    expect(views.map((candidate) => candidate.slicePosition[3])).toEqual([5, 3]);
  });

  it('searches beyond the predicted slice after a multi-step scrub', () => {
    const views = planChunkBoundaryViewStates(view(0), view(5), [{ start: 0, end: 100 }], index(), [
      { shape: [800], chunks: [200] },
    ]);

    expect(views.map((candidate) => candidate.slicePosition[3])).toEqual([5, 6]);
  });

  it('deduplicates a chunk boundary that equals the predicted slice', () => {
    const views = planChunkBoundaryViewStates(view(0), view(2), [{ start: 0, end: 100 }], index(), [
      { shape: [800], chunks: [200] },
    ]);

    expect(views.map((candidate) => candidate.slicePosition[3])).toEqual([2]);
  });

  it('searches behind the predicted slice after a reverse multi-step scrub', () => {
    const views = planChunkBoundaryViewStates(
      view(6),
      view(2),
      [{ start: 600, end: 700 }],
      index(),
      [{ shape: [800], chunks: [200] }]
    );

    expect(views.map((candidate) => candidate.slicePosition[3])).toEqual([2, 1]);
  });

  it('keeps one-step prefetch when several hidden axes move', () => {
    const current = { ...view(1), displayDims: [0, 1], slicePosition: [0, 0, 1, 1] };
    const predicted = { ...view(2), displayDims: [0, 1], slicePosition: [0, 0, 2, 2] };
    expect(
      planChunkBoundaryViewStates(current, predicted, [{ start: 100, end: 200 }], index(), [
        { shape: [800], chunks: [400] },
      ])
    ).toEqual([predicted]);
  });
});
