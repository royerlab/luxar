// @vitest-environment jsdom
/**
 * The per-geometry processors gate their "first update" info logs on a PASS
 * counter. The view version only advances on a view CHANGE, so on a static
 * scene it stays at 1 forever — handing it to the processors kept those logs
 * on for every refinement rung, retry and lazy commit. Every processor entry
 * point must read the pass counter, not just the main sweep's.
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

type Internals = {
  _updateVersion: number;
  _passCount: number;
  processLinesData(path: string, data: unknown, viewState: unknown): Promise<unknown>;
  processGSplatsData(path: string, data: unknown, viewState: unknown): Promise<unknown>;
};

describe('processor first-update log gate reads the pass counter', () => {
  it.each([
    ['lines', 'processLinesData', processors.lines],
    ['gsplats', 'processGSplatsData', processors.gsplats],
  ] as const)('%s', async (_label, method, helper) => {
    const loader = new SceneLoader();
    const internals = loader as unknown as Internals;
    // A static scene four passes in: one view change, several same-view passes.
    internals._updateVersion = 1;
    internals._passCount = 4;

    await internals[method]('/node', {}, {});

    // The gate argument is the fifth positional parameter of both helpers.
    expect(helper).toHaveBeenCalledOnce();
    expect(helper.mock.calls[0][4]).toBe(4);
    await loader.dispose();
  });
});
