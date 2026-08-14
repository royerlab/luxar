/**
 * Unit tests for core/app/debug/capture-readiness.ts.
 *
 * Pure function over a plain object — no THREE, no jsdom globals needed. The
 * point of the module is that the capture tool's readiness verdict is testable
 * at all: the previous inline `page.evaluate` version read the totals from a
 * `state.performance` sub-object that `DebugState` never had, and reported
 * `ok: false` for every scene (royerlab/luxar#1579).
 */

import { describe, it, expect } from 'vitest';
import {
  summarizeCaptureReadiness,
  type CaptureReadinessSummary,
} from '../../../../../core/app/debug/capture-readiness';
import type { DebugState } from '../../../../../core/app/debug/debug-state';

/**
 * Build a FLAT `DebugState`-shaped snapshot (the real `getState()` shape) with
 * every total zeroed and every per-node array empty, overridden by `overrides`.
 */
function makeState(overrides: Partial<DebugState> = {}): Partial<DebugState> {
  return {
    totalPoints: 0,
    totalGSplats: 0,
    totalLines: 0,
    totalTriangles: 0,
    totalElements: 0,
    pointClouds: [],
    gsplatMeshes: [],
    lineMeshes: [],
    meshNodes: [],
    ...overrides,
  };
}

/** A minimal mesh-node entry — only the counts this helper reads matter. */
function meshNode(name: string, triangleCount: number): DebugState['meshNodes'][number] {
  return {
    name,
    triangleCount,
    vertexCount: triangleCount * 3,
    visible: true,
    flatNormal: false,
    alphaCutout: false,
    hasColormap: false,
  };
}

/** Assert no summary field leaked a NaN / undefined across the boundary. */
function expectNoNaN(summary: CaptureReadinessSummary): void {
  for (const key of [
    'totalPoints',
    'totalGSplats',
    'totalLines',
    'totalTriangles',
    'totalElements',
    'pointCloudCount',
    'gsplatCount',
    'lineCount',
    'meshNodeCount',
  ] as const) {
    expect(Number.isFinite(summary[key]), `${key} must be a finite number`).toBe(true);
  }
  // JSON round-trip is what the tool actually prints: `undefined` values vanish
  // silently, which is how #1579 stayed invisible. `toStrictEqual`, not
  // `toEqual` — the latter treats a missing key and an `undefined` one as
  // equal, so it could never catch a dropped total.
  expect(JSON.parse(JSON.stringify(summary))).toStrictEqual(summary);
}

describe('summarizeCaptureReadiness', () => {
  it('reports a MESH-ONLY scene as ready, with its mesh node count', () => {
    const summary = summarizeCaptureReadiness(
      makeState({
        totalTriangles: 1200,
        totalElements: 1200,
        meshNodes: [meshNode('surface', 800), meshNode('cap', 400)],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.totalTriangles).toBe(1200);
    expect(summary.totalElements).toBe(1200);
    expect(summary.meshNodeCount).toBe(2);
    expect(summary.pointCloudCount).toBe(0);
    expect(summary.gsplatCount).toBe(0);
    expect(summary.reason).toBeUndefined();
    expectNoNaN(summary);
  });

  it('reports a LINES-ONLY scene as ready, with its line node count', () => {
    const summary = summarizeCaptureReadiness(
      makeState({
        totalLines: 5000,
        totalElements: 5000,
        lineMeshes: [{ name: 'tracks', segmentCount: 5000, visible: true, hasColormap: true }],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.totalLines).toBe(5000);
    expect(summary.lineCount).toBe(1);
    expect(summary.meshNodeCount).toBe(0);
    expectNoNaN(summary);
  });

  it('reports a points + gsplats scene as ready, with both node counts', () => {
    const summary = summarizeCaptureReadiness(
      makeState({
        totalPoints: 100,
        totalGSplats: 250,
        totalElements: 350,
        pointClouds: [
          {
            name: 'cloud',
            pointCount: 100,
            visible: true,
            hasColors: true,
            hasRadii: false,
            hasSharpness: false,
          },
        ],
        gsplatMeshes: [
          { name: 'splats_a', splatCount: 150, visible: true },
          { name: 'splats_b', splatCount: 100, visible: true },
        ],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.totalPoints).toBe(100);
    expect(summary.totalGSplats).toBe(250);
    expect(summary.totalElements).toBe(350);
    expect(summary.pointCloudCount).toBe(1);
    expect(summary.gsplatCount).toBe(2);
    expectNoNaN(summary);
  });

  it('reports an empty scene as not ready, naming all four types', () => {
    const summary = summarizeCaptureReadiness(makeState());

    expect(summary.ok).toBe(false);
    expect(summary.reason).toMatch(/zero points, gsplats, lines and triangles/);
    expect(summary.totalElements).toBe(0);
    expectNoNaN(summary);
  });

  it('reports a null / missing state as not ready with a reason and no NaN', () => {
    for (const input of [null, undefined]) {
      const summary = summarizeCaptureReadiness(input);
      expect(summary.ok).toBe(false);
      expect(summary.reason).toMatch(/no debug state/);
      expect(summary.totalElements).toBe(0);
      expectNoNaN(summary);
    }
  });

  it('derives totalElements from the per-type totals when the field is absent', () => {
    const partial = {
      totalTriangles: 42,
      meshNodes: [meshNode('surface', 42)],
    } as Partial<DebugState>;

    const summary = summarizeCaptureReadiness(partial);
    expect(summary.ok).toBe(true);
    expect(summary.totalElements).toBe(42);
    expectNoNaN(summary);
  });

  it('does not let a STALE totalElements contradict the per-type totals', () => {
    // A snapshot that carries `totalElements` but disagrees with its own
    // per-type totals must not resurrect the #1579 symptom: trusting the
    // snapshot's field outright made this mesh-only scene read as EMPTY while
    // simultaneously reporting 1200 triangles. `max(field, sum)` means a
    // present total can only under-claim, never contradict.
    const summary = summarizeCaptureReadiness(
      makeState({
        totalTriangles: 1200,
        totalElements: 0,
        meshNodes: [meshNode('surface', 1200)],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.reason).toBeUndefined();
    expect(summary.totalTriangles).toBe(1200);
    expect(summary.totalElements).toBe(1200);
    expectNoNaN(summary);
  });

  it('calls a NON-FINITE total present-but-unusable, not a shape mismatch', () => {
    // THREE defaults `InstancedBufferGeometry.instanceCount` to `Infinity`, and
    // debug-state's gsplat / lines arms read it without a finite guard — so an
    // `Infinity` total is reachable. The totals ARE present, so "unexpected
    // getState() shape" would be the wrong diagnosis; report the real one.
    // Deliberately NOT built with `makeState()`: the only totals present are
    // the non-finite ones, so nothing finite can mask the diagnosis.
    const nonFiniteOnly = {
      totalGSplats: Infinity,
      totalElements: Infinity,
      gsplatMeshes: [{ name: 'splats', splatCount: 0, visible: true }],
    } as Partial<DebugState>;

    const summary = summarizeCaptureReadiness(nonFiniteOnly);
    expect(summary.ok).toBe(false);
    expect(summary.reason).toMatch(/not finite/);
    expect(summary.reason).not.toMatch(/unexpected getState\(\) shape/);
    expect(summary.totalGSplats).toBe(0);
    expect(summary.totalElements).toBe(0);
    expect(summary.gsplatCount).toBe(1);
    expectNoNaN(summary);

    // Same honest reason when finite zeros sit alongside the non-finite total.
    const mixed = summarizeCaptureReadiness(makeState({ totalGSplats: Infinity }));
    expect(mixed.ok).toBe(false);
    expect(mixed.reason).toMatch(/not finite/);
    expectNoNaN(mixed);
  });

  it('flags a non-finite total as a CAVEAT even when the scene is otherwise ready', () => {
    // The mixed case: real points AND an unreadable gsplat total. The verdict is
    // legitimately `ok: true` (100 points are there to capture), but
    // `totalGSplats` prints as 0 — a count we could NOT read, indistinguishable
    // from a genuinely splat-free scene unless the summary says so. A silent 0
    // on the ready path is the same failure class as #1579; `reason` therefore
    // doubles as a caveat channel and must name the offending field.
    const summary = summarizeCaptureReadiness(
      makeState({
        totalPoints: 100,
        totalGSplats: Infinity,
        totalElements: Infinity,
        pointClouds: [
          {
            name: 'cloud',
            pointCount: 100,
            visible: true,
            hasColors: true,
            hasRadii: false,
            hasSharpness: false,
          },
        ],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.reason).toBeDefined();
    expect(summary.reason).toMatch(/not finite/);
    expect(summary.reason).toMatch(/totalGSplats/);
    expect(summary.totalGSplats).toBe(0);
    expect(summary.totalPoints).toBe(100);
    expect(summary.totalElements).toBe(100);
    expectNoNaN(summary);
  });

  it('clamps NEGATIVE totals at zero so they cannot cancel a real positive', () => {
    // An element count cannot be negative. Unclamped, this sums to exactly 0 and
    // reports "nothing loaded" over a scene that carries 1200 triangles.
    const cancelling = summarizeCaptureReadiness(
      makeState({
        totalPoints: -1200,
        totalTriangles: 1200,
        meshNodes: [meshNode('surface', 1200)],
      })
    );

    expect(cancelling.ok).toBe(true);
    expect(cancelling.totalPoints).toBe(0);
    expect(cancelling.totalTriangles).toBe(1200);
    expect(cancelling.totalElements).toBe(1200);
    expectNoNaN(cancelling);

    // And a lone negative must not be PRINTED, next to a reason that says every
    // total is zero.
    const lone = summarizeCaptureReadiness(makeState({ totalPoints: -5 }));
    expect(lone.ok).toBe(false);
    expect(lone.reason).toMatch(/zero points, gsplats, lines and triangles/);
    expect(lone.totalPoints).toBe(0);
    expect(lone.totalElements).toBe(0);
    expectNoNaN(lone);
  });

  it('survives a non-object and an array crossing the page.evaluate boundary', () => {
    // Both guards on the input boundary are pinned here, because a serialisation
    // boundary can hand back anything and neither guard is otherwise exercised.
    //
    // A non-object is truthy, so only the `typeof state !== 'object'` half of
    // the null guard catches it — without that half these fall through and get
    // mis-diagnosed as a shape mismatch instead of "no debug state".
    for (const input of ['not a state', 42, true] as unknown as Partial<DebugState>[]) {
      const summary = summarizeCaptureReadiness(input);
      expect(summary.ok).toBe(false);
      expect(summary.reason).toMatch(/no debug state/);
      expect(summary.totalElements).toBe(0);
      expectNoNaN(summary);
    }

    // An ARRAY is truthy and `typeof 'object'`, so it passes the null guard and
    // reaches the field reads with every field absent — the array-length helper
    // must return 0 for a non-array (here `undefined`) rather than dereferencing
    // `.length` on it and throwing.
    const asArray = summarizeCaptureReadiness([] as unknown as Partial<DebugState>);
    expect(asArray.ok).toBe(false);
    expect(asArray.reason).toMatch(/no element totals/);
    expect(asArray.pointCloudCount).toBe(0);
    expect(asArray.meshNodeCount).toBe(0);
    expect(asArray.totalElements).toBe(0);
    expectNoNaN(asArray);

    // ...and a non-array that HAS a `length` must read as 0 nodes rather than as
    // its character count, which is why the helper tests `Array.isArray` instead
    // of just reaching for `?.length ?? 0`.
    const bogusArrays = {
      totalTriangles: 5,
      pointClouds: 'three',
      meshNodes: { length: 9 },
    } as unknown as Partial<DebugState>;

    const bogus = summarizeCaptureReadiness(bogusArrays);
    expect(bogus.ok).toBe(true);
    expect(bogus.pointCloudCount).toBe(0);
    expect(bogus.meshNodeCount).toBe(0);
    expectNoNaN(bogus);
  });

  it('KNOWN AND INTENDED: an all-hidden scene still reports ok (graph, not pixels)', () => {
    // The verdict deliberately mirrors `debug-state.ts`'s aggregate contract,
    // where hidden nodes still count towards the totals (and every level of a
    // substitutive kind=lod group counts, not just the active one). Filtering
    // on `visible` here would make the two disagree, so `ok` answers "the
    // scene graph carries drawable elements", NOT "the screenshot will have
    // pixels". Pinned so the limitation is visible rather than a surprise.
    const summary = summarizeCaptureReadiness(
      makeState({
        totalTriangles: 900,
        totalElements: 900,
        meshNodes: [{ ...meshNode('hidden-surface', 900), visible: false }],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.totalTriangles).toBe(900);
    expect(summary.meshNodeCount).toBe(1);
    expectNoNaN(summary);
  });

  describe('REGRESSION #1579: totals are read from the FLAT DebugState', () => {
    it('does NOT accept totals nested under a `performance` sub-object', () => {
      // The exact shape the old inline read ASSUMED. `getState()` has never
      // returned it, so a helper that trusts it would be blind to the real
      // flat fields — pin that this shape reads as "unexpected shape", not
      // as a ready scene.
      const nested = {
        performance: {
          totalPoints: 0,
          totalGSplats: 0,
          totalLines: 0,
          totalTriangles: 5,
          totalElements: 5,
          pointClouds: [],
          gsplatMeshes: [],
        },
      } as unknown as Partial<DebugState>;

      const summary = summarizeCaptureReadiness(nested);
      expect(summary.ok).toBe(false);
      expect(summary.reason).toMatch(/no element totals/);
      expect(summary.totalElements).toBe(0);
      expectNoNaN(summary);
    });

    it('DOES accept the flat shape `getState()` actually returns', () => {
      // The mirror of the case above: the same 5 triangles, flat. This is the
      // assertion the old code failed for every geometry type.
      const summary = summarizeCaptureReadiness(
        makeState({
          totalTriangles: 5,
          totalElements: 5,
          meshNodes: [meshNode('surface', 5)],
        })
      );

      expect(summary.ok).toBe(true);
      expect(summary.totalElements).toBe(5);
      expect(summary.meshNodeCount).toBe(1);
    });
  });
});
