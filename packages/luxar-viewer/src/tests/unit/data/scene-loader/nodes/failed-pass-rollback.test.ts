/**
 * A failed initial load unwinds the ladder pass it started, for every geometry.
 *
 * A progressive loader advances its ladder cursor as each rung arrives, BEFORE
 * the concat and commit that can still throw. A catch that only records the
 * failure leaves the cursor advanced, so the retry resumes PAST the prefix that
 * failed and attempts a strictly larger allocation — escalation instead of
 * backoff, and at the last rung `hasMoreLODs` goes false and the node is
 * stranded (#2426, `loaders/progressive/pass-rollback.ts`). The update sweep and
 * the refinement wrapper already unwind; these pin the initial-load halves (the
 * lazy-level path runs exactly these) and the retry path in `retry.test.ts`.
 */

import { describe, it, expect, vi } from 'vitest';

vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => ({}));

import { loadPointsNodeExpensive } from '../../../../../data/scene-loader/nodes/load-points-node';
import { loadLinesNodeExpensive } from '../../../../../data/scene-loader/nodes/load-lines-node';
import { loadGSplatsNodeExpensive } from '../../../../../data/scene-loader/nodes/load-gsplats-node';
import { loadMeshNodeExpensive } from '../../../../../data/scene-loader/nodes/load-mesh-node';
import { LoaderError } from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

const PATH = '/scene/ladder';

/** The initial-load entry point each kind's loader is driven through. */
const CASES = [
  ['points', 'loadPoints', loadPointsNodeExpensive],
  ['lines', 'loadLines', loadLinesNodeExpensive],
  ['gsplats', 'loadGSplats', loadGSplatsNodeExpensive],
  ['mesh', 'loadMesh', loadMeshNodeExpensive],
] as const;

describe('initial-load failure unwinds the ladder pass', () => {
  it.each(CASES)('%s: rolls the loader back to its pass start', async (type, method, load) => {
    const rollbackToPassStart = vi.fn(() => 2);
    // The pass appended rungs, then its concat threw (an allocation failure).
    const loader = {
      [method]: vi.fn().mockRejectedValue(new RangeError('Array buffer allocation failed')),
      rollbackToPassStart,
    };
    const ctx = makeTestNodeBuildCtx({});
    const node: SceneNode = {
      path: PATH,
      type,
      attrs: { type },
      hasSpatialIndex: type !== 'mesh',
      children: [],
    };

    await expect(load(node, ctx, loader as never)).rejects.toBeInstanceOf(LoaderError);

    expect(rollbackToPassStart).toHaveBeenCalledTimes(1);
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(true);
  });

  it('a loader without a ladder (no rollbackToPassStart) still records the failure', async () => {
    const ctx = makeTestNodeBuildCtx({});
    const loader = { loadPoints: vi.fn().mockRejectedValue(new Error('decode failed')) };
    const node: SceneNode = {
      path: PATH,
      type: 'points',
      attrs: {},
      hasSpatialIndex: true,
      children: [],
    };

    await expect(loadPointsNodeExpensive(node, ctx, loader as never)).rejects.toBeInstanceOf(
      LoaderError
    );
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(true);
  });
});
