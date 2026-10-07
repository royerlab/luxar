// @vitest-environment jsdom
/**
 * A loader's worker projections race ITS OWN dataset signal.
 *
 * The data-worker pool is a page singleton shared by every host (a LuxarApp
 * and any number of LuxarLayers), so it carries no dataset signal of its own:
 * each SceneLoader threads its dataset signal into the worker calls it makes.
 * A call made without a per-update signal (the initial node build, a retry)
 * must still settle when its dataset is disposed; one made with a per-update
 * signal settles when either aborts.
 */

import { describe, expect, it, vi } from 'vitest';

const processors = vi.hoisted(() => ({
  lines: vi.fn(async (..._args: unknown[]) => null),
  gsplats: vi.fn(async (..._args: unknown[]) => null),
}));

vi.mock(
  import('../../../data/scene-loader/process/data-processor-lines'),
  async (importOriginal) => ({
    ...(await importOriginal()),
    processLinesData: processors.lines,
  })
);
vi.mock(
  import('../../../data/scene-loader/process/data-processor-gsplats'),
  async (importOriginal) => ({
    ...(await importOriginal()),
    processGSplatsData: processors.gsplats,
  })
);

import { SceneLoader } from '../../../data';

type Process = (
  path: string,
  data: unknown,
  viewState: unknown,
  session?: unknown,
  signal?: AbortSignal
) => Promise<unknown>;
type Internals = {
  _datasetAbortController: AbortController | null;
  processLinesData: Process;
  processGSplatsData: Process;
  makeNodeBuildCtx(): { processLinesData: Process; processGSplatsData: Process };
};

/** The signal argument (7th positional) of the helper's last call. */
const lastSignal = (helper: typeof processors.lines): AbortSignal | undefined =>
  helper.mock.lastCall?.[6] as AbortSignal | undefined;

function loaderWithDataset(): {
  loader: SceneLoader;
  internals: Internals;
  dataset: AbortController;
} {
  const loader = new SceneLoader();
  const internals = loader as unknown as Internals;
  const dataset = new AbortController();
  internals._datasetAbortController = dataset;
  return { loader, internals, dataset };
}

describe('worker projections race the loader’s own dataset signal', () => {
  // geometry-subset: only lines and gsplats project on a worker; points arrive display-ready and mesh projects on the main thread
  it.each([
    ['lines', 'processLinesData', processors.lines],
    ['gsplats', 'processGSplatsData', processors.gsplats],
  ] as const)('%s: initial build, retry and update paths', async (_label, method, helper) => {
    const a = loaderWithDataset();
    const b = loaderWithDataset();

    // Initial node build (no per-update signal): the dataset signal itself.
    await a.internals.makeNodeBuildCtx()[method]('/node', {}, {});
    expect(lastSignal(helper)).toBe(a.dataset.signal);
    await b.internals.makeNodeBuildCtx()[method]('/node', {}, {});
    expect(lastSignal(helper)).toBe(b.dataset.signal);

    // Retry / refinement entry (no per-update signal).
    await a.internals[method]('/node', {}, {});
    expect(lastSignal(helper)).toBe(a.dataset.signal);

    // A per-update signal is raced together with the dataset's.
    const update = new AbortController();
    await b.internals[method]('/node', {}, {}, undefined, update.signal);
    const raced = lastSignal(helper);
    expect(raced?.aborted).toBe(false);
    a.dataset.abort();
    expect(raced?.aborted).toBe(false);
    b.dataset.abort();
    expect(raced?.aborted).toBe(true);

    await a.loader.dispose();
    await b.loader.dispose();
  });
});
