/**
 * Unit tests for `deriveNodeViewState` — the single source of truth that
 * folds `extend_to_all` tolerance overrides and the inverse world
 * `nd_transform` into a per-node ViewState.
 *
 * Pure function: no DOM, no I/O. We build real ViewState / SceneNode /
 * Dimensions fixtures (validated against the imported types) and assert
 * the three branches:
 *   1. full-extend skip
 *   2. partial-extend tolerance override
 *   3. nd_transform inverse-query mapping (affine + permutation)
 */

import { describe, it, expect } from 'vitest';
import { deriveNodeViewState } from '../../../../../data/scene-loader/view-state/derive-node-view-state';
import { EXTEND_TO_ALL_TOLERANCE } from '../../../../../data/scene-loader/view-state/extend-tolerance';
import type { ViewState, SceneNode } from '../../../../../data/data-loader-types';
import type { DimensionMetadata } from '../../../../../types/dims';

// 4D scene: time, channel are non-displayed; z,y,x displayed (indices 2,3,4
// in a 5D layout would be unwieldy — keep it to 4D: dims = [time, channel,
// z, y] with displayDims = [2, 3]).
function dims(): DimensionMetadata[] {
  return [
    { name: 'time', unit: '', scale: 1.0 },
    { name: 'channel', unit: '', scale: 1.0 },
    { name: 'z', unit: 'um', scale: 1.0 },
    { name: 'y', unit: 'um', scale: 1.0 },
  ];
}

function baseViewState(overrides: Partial<ViewState> = {}): ViewState {
  return {
    displayDims: [2, 3], // z, y are displayed; time + channel are non-displayed
    slicePosition: [5, 1, 0, 0],
    tolerance: [0.5, 0.5, 0, 0],
    dimensions: dims(),
    ...overrides,
  };
}

function node(path: string, attrs: SceneNode['attrs'] = {}, children?: SceneNode[]): SceneNode {
  return { path, type: 'group', attrs, hasSpatialIndex: false, children };
}

const noTransformOpts = { applyPartialExtendTolerance: false };

describe('deriveNodeViewState — no extend, no transform', () => {
  it('returns a derived viewState mirroring the base when nothing applies', () => {
    const base = baseViewState();
    const result = deriveNodeViewState('points', undefined, base, null, noTransformOpts);
    expect(result.skip).toBe(false);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.displayDims).toEqual([2, 3]);
    expect(result.viewState.slicePosition).toEqual([5, 1, 0, 0]);
    expect(result.viewState.tolerance).toEqual([0.5, 0.5, 0, 0]);
    expect(result.viewState.dimensions).toBe(base.dimensions);
  });

  it('treats an empty extend_to_all list as no extension', () => {
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: [] },
      baseViewState(),
      null,
      noTransformOpts
    );
    expect(result.skip).toBe(false);
  });
});

describe('deriveNodeViewState — full-extend skip', () => {
  it('returns skip "extend_to_all" WITH a derived view state when ALL non-displayed dims are extended', () => {
    // Non-displayed dims are time + channel; extending both => full skip hint.
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time', 'channel'] },
      baseViewState(),
      null,
      { applyPartialExtendTolerance: true }
    );
    expect(result.skip).toBe('extend_to_all');
    // #1157: `skip` is only an optimization hint now — a fully-extended node
    // still carries its derived view state (the base slice with the 1e10
    // sentinels applied to the extended, hidden dims), so consumers query the
    // "ignore these dims" region instead of falling back to the raw live slice.
    expect(result.viewState).toBeDefined();
    expect(result.viewState.tolerance).toEqual([
      EXTEND_TO_ALL_TOLERANCE,
      EXTEND_TO_ALL_TOLERANCE,
      0,
      0,
    ]);
  });

  it('does NOT skip when only some non-displayed dims are extended', () => {
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time'] },
      baseViewState(),
      null,
      { applyPartialExtendTolerance: false }
    );
    expect(result.skip).toBe(false);
  });

  it('skips (with a derived view state) even if a displayed dim is also (redundantly) listed', () => {
    // Listing 'z' (displayed) alongside both non-displayed dims still
    // fully covers the non-displayed set -> skip hint, plus the derived state.
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time', 'channel', 'z'] },
      baseViewState(),
      null,
      { applyPartialExtendTolerance: true }
    );
    expect(result.skip).toBe('extend_to_all');
    expect(result.viewState).toBeDefined();
    // 'z' (index 2) is a displayed dim but is redundantly extended, so its
    // tolerance slot also gets the sentinel — harmless, as a displayed dim's
    // tolerance is ignored by the query. The two hidden dims (time, channel)
    // carry the sentinel as the #1157 fix requires.
    expect(result.viewState.tolerance).toEqual([
      EXTEND_TO_ALL_TOLERANCE,
      EXTEND_TO_ALL_TOLERANCE,
      EXTEND_TO_ALL_TOLERANCE,
      0,
    ]);
  });

  it('throws via validateExtendDims when an extend dim name is unknown', () => {
    expect(() =>
      deriveNodeViewState(
        'points',
        { extend_to_all: ['bogus'] },
        baseViewState(),
        null,
        noTransformOpts
      )
    ).toThrow(/Invalid extend_to_all/);
  });

  it('skips the full-extend check entirely when dimensions is undefined', () => {
    const base = baseViewState({ dimensions: undefined });
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time', 'channel'] },
      base,
      null,
      noTransformOpts
    );
    // No dimensions => Step 1 short-circuits, no skip.
    expect(result.skip).toBe(false);
  });

  it('carries the 1e10 sentinels on the fully-extended case (issue #1157 regression)', () => {
    // The core #1157 regression. Before the fix, a fully-extended node
    // early-returned a bare `{ skip: 'extend_to_all' }` with NO view state,
    // BEFORE the tolerance override ran. Consumers then either short-circuited
    // (freezing the additive ladder) or fell back to the raw live slice —
    // slicing points/lines away and filtering gsplats out, because
    // data-processor-gsplats derives its extended-dims set from exactly these
    // 1e10 sentinels, which were never applied. Now the fully-extended case
    // ALSO gets the override, so the derived view state carries the sentinels
    // that feed the initial-load / refinement / retry / prefetch paths.
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time', 'channel'] },
      baseViewState(),
      null,
      { applyPartialExtendTolerance: true }
    );
    expect(result.skip).toBe('extend_to_all');
    // Hidden dims (time = 0, channel = 1) carry the sentinel; displayed dims stay 0.
    expect(result.viewState.tolerance).toEqual([
      EXTEND_TO_ALL_TOLERANCE,
      EXTEND_TO_ALL_TOLERANCE,
      0,
      0,
    ]);
    // Only the tolerance changed — slice position and display dims are the base ones.
    expect(result.viewState.displayDims).toEqual([2, 3]);
    expect(result.viewState.slicePosition).toEqual([5, 1, 0, 0]);
  });
});

describe('deriveNodeViewState — partial-extend tolerance override', () => {
  it('replaces the extended dim tolerance with EXTEND_TO_ALL_TOLERANCE when enabled', () => {
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time'] },
      baseViewState(),
      null,
      { applyPartialExtendTolerance: true }
    );
    expect(result.skip).toBe(false);
    if (result.skip !== false) throw new Error('expected non-skip');
    // 'time' is index 0 -> set to EXTEND_TO_ALL_TOLERANCE, channel untouched.
    expect(result.viewState.tolerance).toEqual([EXTEND_TO_ALL_TOLERANCE, 0.5, 0, 0]);
  });

  it('does NOT override tolerance when applyPartialExtendTolerance is false', () => {
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time'] },
      baseViewState(),
      null,
      { applyPartialExtendTolerance: false }
    );
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.tolerance).toEqual([0.5, 0.5, 0, 0]);
  });

  it('uses the provided extendedToleranceCache when present', () => {
    const cache = new Map<string, number[]>();
    const r1 = deriveNodeViewState('points', { extend_to_all: ['time'] }, baseViewState(), null, {
      applyPartialExtendTolerance: true,
      extendedToleranceCache: cache,
    });
    if (r1.skip !== false) throw new Error('expected non-skip');
    // The cache now holds the computed tolerance keyed by the sorted set.
    expect(cache.has('time')).toBe(true);
    // A second call with the same set reuses the cached array reference.
    const r2 = deriveNodeViewState('points', { extend_to_all: ['time'] }, baseViewState(), null, {
      applyPartialExtendTolerance: true,
      extendedToleranceCache: cache,
    });
    if (r2.skip !== false) throw new Error('expected non-skip');
    expect(r2.viewState.tolerance).toBe(r1.viewState.tolerance);
  });
});

describe('deriveNodeViewState — nd_transform inverse-query mapping', () => {
  it('inverts an affine transform on a non-displayed dim (world->local)', () => {
    // nd_transform on 'time': effective = scale*raw + offset, scale=2, offset=10.
    // Inverse query: local = (world - offset)/scale = (5 - 10)/2 = -2.5
    //                localTol = worldTol / |scale| = 0.5 / 2 = 0.25
    const root = node('', {}, [
      node('points', { nd_transform: { time: { scale: 2, offset: 10 } } }),
    ]);
    const result = deriveNodeViewState('points', undefined, baseViewState(), root, noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.slicePosition[0]).toBeCloseTo(-2.5, 10);
    expect(result.viewState.tolerance[0]).toBeCloseTo(0.25, 10);
    // Other dims unchanged.
    expect(result.viewState.slicePosition[1]).toBe(1);
    expect(result.viewState.tolerance[1]).toBe(0.5);
  });

  it('leaves displayed dims untouched even if the transform names them', () => {
    // 'z' is a displayed dim (index 2). invertNdTransformForQuery skips
    // displayDims, so the transform on 'z' must have no effect.
    const root = node('', {}, [node('points', { nd_transform: { z: { scale: 3, offset: 1 } } })]);
    const result = deriveNodeViewState('points', undefined, baseViewState(), root, noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.slicePosition[2]).toBe(0); // unchanged display dim
    expect(result.viewState.tolerance[2]).toBe(0);
  });

  it('inverts a permutation transform on a categorical non-displayed dim', () => {
    // permutation[old] = new. perm = [2,0,1] means:
    //   inversePerm[new]=old -> inversePerm[2]=0, inversePerm[0]=1, inversePerm[1]=2
    // base channel slice (world index) = 1 -> local = inversePerm[1] = 2.
    const base = baseViewState({ slicePosition: [5, 1, 0, 0] });
    const root = node('', {}, [
      node('points', { nd_transform: { channel: { permutation: [2, 0, 1] } } }),
    ]);
    const result = deriveNodeViewState('points', undefined, base, root, noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.slicePosition[1]).toBe(2); // remapped channel
    // Tolerance for categorical dims is unchanged.
    expect(result.viewState.tolerance[1]).toBe(0.5);
  });

  it('composes nd_transforms along the path (parent over child) before inverting', () => {
    // Parent affine on 'time': scale=2, offset=0; child affine scale=1, offset=4.
    // Composed effective = 2*(1*raw + 4) + 0 = 2*raw + 8.
    // Inverse: local = (world - 8)/2 = (5 - 8)/2 = -1.5; tol = 0.5/2 = 0.25.
    const root = node('root', { nd_transform: { time: { scale: 2 } } }, [
      node('root/points', { nd_transform: { time: { offset: 4 } } }),
    ]);
    const result = deriveNodeViewState(
      'root/points',
      undefined,
      baseViewState(),
      root,
      noTransformOpts
    );
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.slicePosition[0]).toBeCloseTo(-1.5, 10);
    expect(result.viewState.tolerance[0]).toBeCloseTo(0.25, 10);
  });

  it('is identity when the node has no nd_transform in its path', () => {
    const root = node('', {}, [node('points', {})]);
    const result = deriveNodeViewState('points', undefined, baseViewState(), root, noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.slicePosition).toEqual([5, 1, 0, 0]);
    expect(result.viewState.tolerance).toEqual([0.5, 0.5, 0, 0]);
  });

  it('skips the inverse step when dimensions is undefined', () => {
    const base = baseViewState({ dimensions: undefined });
    const root = node('', {}, [
      node('points', { nd_transform: { time: { scale: 2, offset: 10 } } }),
    ]);
    const result = deriveNodeViewState('points', undefined, base, root, noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    // No dimensions => Step 3 short-circuits, slicePosition unchanged.
    expect(result.viewState.slicePosition).toEqual([5, 1, 0, 0]);
  });
});

describe('deriveNodeViewState — combined partial-extend + nd_transform', () => {
  it('applies the tolerance override AND the inverse transform together', () => {
    // Extend 'time' (index 0) -> tolerance[0] = EXTEND_TO_ALL_TOLERANCE.
    // Then invert affine on 'channel' (index 1): scale=2, offset=0.
    //   local channel = (1 - 0)/2 = 0.5; tol = 0.5/2 = 0.25.
    const root = node('', {}, [node('points', { nd_transform: { channel: { scale: 2 } } })]);
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time'] },
      baseViewState(),
      root,
      { applyPartialExtendTolerance: true }
    );
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.tolerance[0]).toBe(EXTEND_TO_ALL_TOLERANCE);
    expect(result.viewState.slicePosition[1]).toBeCloseTo(0.5, 10);
    expect(result.viewState.tolerance[1]).toBeCloseTo(0.25, 10);
  });
});

describe('deriveNodeViewState — nd_transform no-preimage propagation', () => {
  /** Same 4D layout, but `time` is a discrete ordinal on the integer grid. */
  function discreteDims(): DimensionMetadata[] {
    return [
      { name: 'time', unit: '', scale: 1.0, discrete: true, step: 1 },
      { name: 'channel', unit: '', scale: 1.0 },
      { name: 'z', unit: 'um', scale: 1.0 },
      { name: 'y', unit: 'um', scale: 1.0 },
    ];
  }

  const scaledNode = (scale: number) =>
    node('', {}, [node('points', { nd_transform: { time: { scale } } })]);

  it('flags noPreimage when the inverse query falls between discrete categories', () => {
    // world time 5, scale 2 → local 2.5: no local category maps to world 5.
    const base = baseViewState({ dimensions: discreteDims(), tolerance: [0, 0.5, 0, 0] });
    const result = deriveNodeViewState('points', undefined, base, scaledNode(2), noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.noPreimage).toBe(true);
    // The position is still inverted — the flag is what suppresses the render.
    expect(result.viewState.slicePosition[0]).toBeCloseTo(2.5, 10);
  });

  it('leaves noPreimage unset when the inverse query lands on a category', () => {
    const base = baseViewState({
      dimensions: discreteDims(),
      slicePosition: [4, 1, 0, 0],
      tolerance: [0, 0.5, 0, 0],
    });
    const result = deriveNodeViewState('points', undefined, base, scaledNode(2), noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.noPreimage).toBeUndefined();
    expect(result.viewState.slicePosition[0]).toBeCloseTo(2, 10);
  });

  it('leaves noPreimage unset for a plain offset transform', () => {
    const root = node('', {}, [node('points', { nd_transform: { time: { offset: 3 } } })]);
    const base = baseViewState({ dimensions: discreteDims(), tolerance: [0, 0.5, 0, 0] });
    const result = deriveNodeViewState('points', undefined, base, root, noTransformOpts);
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.noPreimage).toBeUndefined();
  });

  it('exempts an extended dim on the LINES path, which never sets the sentinel', () => {
    // Lines derive with applyPartialExtendTolerance: false, so an extended dim
    // carries its ORDINARY tolerance here — the 1e10 sentinel never appears.
    // The exemption must therefore come from the extend_to_all NAMES, or a lines
    // node with a partial extend on a scaled discrete dim goes dark while its
    // points/gsplats siblings render.
    const root = node('', {}, [
      node('lines', { extend_to_all: ['time'], nd_transform: { time: { scale: 2 } } }),
    ]);
    const base = baseViewState({ dimensions: discreteDims(), tolerance: [0, 0.5, 0, 0] });
    const result = deriveNodeViewState(
      'lines',
      { extend_to_all: ['time'] },
      base,
      root,
      { applyPartialExtendTolerance: false } // the lines contract
    );
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.tolerance[0]).toBeLessThan(1e9); // no sentinel, as expected
    expect(result.viewState.noPreimage).toBeUndefined(); // ...but still exempt
  });

  it('still flags a NON-extended scaled dim on the lines path', () => {
    // Same lines contract, but the extended dim is a different one — the
    // exemption is per-dimension, so `time` must still be flagged.
    const root = node('', {}, [
      node('lines', { extend_to_all: ['channel'], nd_transform: { time: { scale: 2 } } }),
    ]);
    const base = baseViewState({ dimensions: discreteDims(), tolerance: [0, 0.5, 0, 0] });
    const result = deriveNodeViewState('lines', { extend_to_all: ['channel'] }, base, root, {
      applyPartialExtendTolerance: false,
    });
    if (result.skip !== false) throw new Error('expected non-skip');
    expect(result.viewState.noPreimage).toBe(true);
  });
});

describe('deriveNodeViewState — fully-extended + nd_transform (issue #1157)', () => {
  it('keeps the 1e10 sentinel UNSCALED on extended dims under an affine nd_transform and exempts them from no-preimage', () => {
    // Step 3 (nd_transform inverse) is newly reachable for a FULLY-extended node
    // after the #1157 fix. `time` is discrete AND scaled: without the extend
    // exemption the inverse (world 5, scale 2 -> local 2.5, off-grid) would flag
    // noPreimage and blank the node, and the sentinel would be divided below the
    // 1e9 floor and silently un-extend the dim. Both must be prevented.
    const discreteTimeDims: DimensionMetadata[] = [
      { name: 'time', unit: '', scale: 1.0, discrete: true, step: 1 },
      { name: 'channel', unit: '', scale: 1.0 },
      { name: 'z', unit: 'um', scale: 1.0 },
      { name: 'y', unit: 'um', scale: 1.0 },
    ];
    const root = node('', {}, [node('points', { nd_transform: { time: { scale: 2 } } })]);
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time', 'channel'] },
      baseViewState({ dimensions: discreteTimeDims }),
      root,
      { applyPartialExtendTolerance: true }
    );
    expect(result.skip).toBe('extend_to_all');
    // Both extended dims keep the sentinel UNSCALED (not divided by |scale|).
    expect(result.viewState.tolerance[0]).toBe(EXTEND_TO_ALL_TOLERANCE);
    expect(result.viewState.tolerance[1]).toBe(EXTEND_TO_ALL_TOLERANCE);
    // Extended dims are exempt from the no-preimage rule, even discrete+scaled.
    expect(result.viewState.noPreimage).toBeUndefined();
    // The position is still inverted: (5 - 0) / 2 = 2.5.
    expect(result.viewState.slicePosition[0]).toBeCloseTo(2.5, 10);
  });

  it('returns the skip hint vacuously (with a defined view state) when there are zero non-displayed dims', () => {
    // Every dim is displayed, so nonDisplayedDims is empty and `[].every(...)`
    // is vacuously true -> fully-extended skip hint, without throwing. Vacuous
    // coverage for that branch.
    const result = deriveNodeViewState(
      'points',
      { extend_to_all: ['time'] },
      baseViewState({ displayDims: [0, 1, 2, 3] }),
      null,
      { applyPartialExtendTolerance: true }
    );
    expect(result.skip).toBe('extend_to_all');
    expect(result.viewState).toBeDefined();
  });
});
