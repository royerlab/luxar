/**
 * B6: `chunk_bounds` ordering in the progressive gsplats loader.
 *
 * Rung k+1's spatial index (its `chunk_bounds`, fetched by
 * `ensureInitialized`) must already be in flight while rung k's data is still
 * loading — requested concurrently, at `refinement` fetch priority — instead of
 * waiting for the whole first pass to finish.
 */

import { describe, expect, it, vi } from 'vitest';

import { GSplatsProgressiveLoader } from '../../../data/gsplats/gsplats-progressive-loader';
import type { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import type { GSplatsViewState, LoadedGSplatsData } from '../../../types/gsplats';

const VIEW: GSplatsViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

function lodData(splatCount: number): LoadedGSplatsData {
  return {
    positions: new Float32Array(splatCount * 3),
    amplitudes: new Float32Array(splatCount),
    choleskyFactors: new Float32Array(splatCount * 6),
    colors: null,
    splatCount,
    ndim: 3,
  } as LoadedGSplatsData;
}

/** A sub-loader whose `updateViewWithResidency` resolves only when released. */
function gatedSubLoader() {
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  const stub = {
    ensureInitialized: vi.fn().mockResolvedValue(undefined),
    updateViewWithResidency: vi.fn(async () => {
      await gate;
      return { data: lodData(10), allResident: true };
    }),
    updateView: vi.fn(),
    prefetchChunks: vi.fn().mockResolvedValue(undefined),
    planPrefetch: vi.fn().mockResolvedValue({ bytes: 0, ranges: [] }),
    getPrefetchCacheStats: vi.fn(() => null),
    releaseAccumulator: vi.fn(),
    dispose: vi.fn(),
    getMetrics: vi.fn(() => ({})),
    getActiveQueries: vi.fn(() => []),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
  return { stub, release: () => release() };
}

describe('GSplatsProgressiveLoader — chunk_bounds ordering (B6)', () => {
  it("requests rung k+1's chunk_bounds before rung k's data completes", async () => {
    const rungs = [gatedSubLoader(), gatedSubLoader(), gatedSubLoader()];
    const loader = new GSplatsProgressiveLoader(
      rungs.map((r) => r.stub) as unknown as GSplatsSpatialIndexLoader[],
      rungs.length,
      '/splats'
    );

    const pass = loader.loadGSplats(VIEW);
    await vi.waitFor(() => expect(rungs[0].stub.updateViewWithResidency).toHaveBeenCalled());

    // Rung 0 is still loading: rung 1's index is already requested, at
    // refinement priority; rung 2's is not (it is warmed behind rung 1).
    expect(rungs[1].stub.ensureInitialized).toHaveBeenCalledWith('refinement');
    expect(rungs[2].stub.ensureInitialized).not.toHaveBeenCalled();

    rungs.forEach((r) => r.release());
    await pass;
  });
});
