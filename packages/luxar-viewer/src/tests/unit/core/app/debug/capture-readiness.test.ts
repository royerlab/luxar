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
import type { DebugState, GPUPoolDebugStats } from '../../../../../core/app/debug/debug-state';
import type { RefinementResidencyStop } from '../../../../../data/scene-loader/progressive/residency-budget';

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
    totalDroppedElements: 0,
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
    'totalDroppedElements',
    'refinementDeclinedPathCount',
    'gpuByteBudgetEvictions',
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
  it('rejects a capture when the renderer dropped elements', () => {
    const summary = summarizeCaptureReadiness(
      makeState({ totalPoints: 80, totalElements: 80, totalDroppedElements: 20 })
    );

    expect(summary.ok).toBe(false);
    expect(summary.totalDroppedElements).toBe(20);
    expect(summary.reason).toContain('20 elements were dropped by renderer capacity limits');
    expectNoNaN(summary);
  });

  it('reports dropped elements together with unreadable totals', () => {
    const summary = summarizeCaptureReadiness(
      makeState({ totalPoints: Infinity, totalDroppedElements: 20 })
    );

    expect(summary.ok).toBe(false);
    expect(summary.reason).toContain('20 elements were dropped by renderer capacity limits');
    expect(summary.reason).toContain('totalPoints present but not finite');
    expectNoNaN(summary);
  });

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
        lineMeshes: [
          {
            name: 'tracks',
            segmentCount: 5000,
            visible: true,
            hasColormap: true,
            requestedElementCount: 5000,
            grantedElementCount: 5000,
            droppedElementCount: 0,
          },
        ],
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
            requestedElementCount: 100,
            grantedElementCount: 100,
            droppedElementCount: 0,
          },
        ],
        gsplatMeshes: [
          {
            name: 'splats_a',
            splatCount: 150,
            visible: true,
            requestedElementCount: 150,
            grantedElementCount: 150,
            droppedElementCount: 0,
          },
          {
            name: 'splats_b',
            splatCount: 100,
            visible: true,
            requestedElementCount: 100,
            grantedElementCount: 100,
            droppedElementCount: 0,
          },
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
    // An absent `totalElements` costs nothing — it is re-derived from the
    // per-type totals — so it must NOT be named as a count that was lost.
    expect(summary.reason).not.toMatch(/totalElements/);
    expectNoNaN(summary);
  });

  it('names the per-type totals a PARTIAL snapshot never carried', () => {
    // The version-skew case the `Math.max` exists for: a snapshot from another
    // viewer build that carries the aggregate but not (all of) the per-type
    // breakdown. Those three print as 0 like any other unreadable count, and an
    // unannounced 0 is indistinguishable from a genuinely empty geometry type —
    // the silent-zero failure #1579 was. So they get named, and the verdict
    // stays `ok` because there are 100 elements to capture.
    const summary = summarizeCaptureReadiness({ totalElements: 100 } as Partial<DebugState>);

    expect(summary.ok).toBe(true);
    expect(summary.totalElements).toBe(100);
    expect(summary.reason).toMatch(/absent from the snapshot/);
    expect(summary.reason).toMatch(/totalPoints/);
    expect(summary.reason).toMatch(/totalGSplats/);
    expect(summary.reason).toMatch(/totalLines/);
    expect(summary.reason).toMatch(/totalTriangles/);
    expect(summary.reason).toMatch(/under-state the scene/);
    expectNoNaN(summary);

    // A snapshot missing only ONE of them names only that one, and says nothing
    // about the four counts it did read.
    const oneMissing = summarizeCaptureReadiness(
      makeState({ totalPoints: 100, totalElements: 100, totalTriangles: undefined })
    );
    expect(oneMissing.ok).toBe(true);
    expect(oneMissing.reason).toMatch(/^totalTriangles absent from the snapshot/);
    expect(oneMissing.reason).not.toMatch(/totalPoints/);
    expectNoNaN(oneMissing);

    // Both flavours of unreadable at once are reported together, not one
    // silently shadowing the other.
    const both = summarizeCaptureReadiness({
      totalPoints: 100,
      totalGSplats: Infinity,
      totalElements: 100,
    } as Partial<DebugState>);
    expect(both.ok).toBe(true);
    expect(both.reason).toMatch(/totalGSplats present but not finite/);
    expect(both.reason).toMatch(/totalLines, totalTriangles absent from the snapshot/);
    expectNoNaN(both);
  });

  it('keeps BOTH unreadable halves inside ONE `; `-split clause', () => {
    // `reason` documents that splitting on `'; '` recovers exactly the causes.
    // The unreadable-totals caveat is ONE cause, but it has two halves — the
    // non-finite totals and the absent ones — and joining them with the cause
    // separator broke that guarantee for any snapshot carrying both, which is
    // reachable on any version-skewed build. They are comma-joined instead.
    const summary = summarizeCaptureReadiness({
      totalPoints: Infinity,
      totalLines: 100,
      totalDroppedElements: 0,
    } as Partial<DebugState>);

    expect(summary.ok).toBe(true);
    // One caveat in, ONE fragment out — not two.
    expect(summary.reason!.split('; ')).toHaveLength(1);
    expect(summary.reason).toBe(
      'totalPoints present but not finite (Infinity/NaN), totalGSplats, totalTriangles ' +
        'absent from the snapshot — counted as zero, so the reported counts under-state the scene'
    );
    expectNoNaN(summary);

    // And beside a real cause it is still exactly one extra fragment, so a
    // split-based caller counts two causes rather than three.
    const withCap = summarizeCaptureReadiness({
      totalPoints: Infinity,
      totalLines: 100,
      totalDroppedElements: 20,
    } as Partial<DebugState>);
    expect(withCap.ok).toBe(false);
    const clauses = withCap.reason!.split('; ');
    expect(clauses).toHaveLength(2);
    expect(clauses[0]).toBe('20 elements were dropped by renderer capacity limits');
    expect(clauses[1]).toMatch(/^totalPoints present but not finite/);
    expectNoNaN(withCap);
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
            requestedElementCount: 100,
            grantedElementCount: 100,
            droppedElementCount: 0,
          },
        ],
      })
    );

    expect(summary.ok).toBe(true);
    expect(summary.reason).toBeDefined();
    expect(summary.reason).toMatch(/not finite/);
    expect(summary.reason).toMatch(/totalGSplats/);
    // The caveat names only the PER-TYPE fields it had to zero. `totalElements`
    // is re-derived from them, and it prints as 100 right below — saying it was
    // "counted as zero" would contradict the number in the same object.
    expect(summary.reason).not.toMatch(/totalElements/);
    expect(summary.totalGSplats).toBe(0);
    expect(summary.totalPoints).toBe(100);
    expect(summary.totalElements).toBe(100);
    expectNoNaN(summary);
  });

  it('says totalElements was RE-DERIVED when only that field is non-finite', () => {
    // Nothing per-type was lost here, so the caveat must not claim an
    // under-stated count: the per-type totals are intact and `totalElements`
    // came from their sum.
    const summary = summarizeCaptureReadiness(
      makeState({ totalPoints: 100, totalElements: Infinity })
    );

    expect(summary.ok).toBe(true);
    expect(summary.reason).toMatch(/re-derived/);
    expect(summary.reason).not.toMatch(/under-state/);
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

  /**
   * #2508: the counts must be a property of the STORE, not of the run. Two
   * caps can break that without dropping a single element — refinement stopping
   * scene-wide at the residency byte ceiling, and the GPU pool reclaiming
   * committed geometry to stay under its VRAM budget. `desi_galaxies` authors
   * 9.75M points, of which one layer's finest level alone is 446 MiB of the
   * 512 MiB ceiling, and the scene settles at ~348k elements with `ok: true`
   * under the pre-fix verdict.
   */
  describe('run-dependent caps', () => {
    const MIB = 1024 * 1024;

    /** A stop record shaped exactly as `RefinementResidencyReporter.snapshot()` emits it. */
    function residencyStop(
      overrides: Partial<RefinementResidencyStop> = {}
    ): RefinementResidencyStop {
      return {
        reason: 'over-budget',
        residentBytes: 517 * MIB,
        budgetBytes: 512 * MIB,
        firstPath: '/galaxies/bright',
        declinedPathCount: 6,
        declinedPaths: ['/galaxies/bright', '/galaxies/faint'],
        ...overrides,
      };
    }

    /** Pool stats with NO byte-budget pressure — the baseline the guards below vary. */
    function pool(overrides: Partial<GPUPoolDebugStats> = {}): GPUPoolDebugStats {
      return {
        activeBuffers: 4,
        pooledBuffers: 2,
        activeBytes: 40 * MIB,
        pooledBytes: 8 * MIB,
        totalBytes: 48 * MIB,
        largestPooledBytes: 6 * MIB,
        evictions: 0,
        byteBudgetEvictions: 0,
        ...overrides,
      };
    }

    it('refuses a scene that stopped at the residency ceiling, naming the figures', () => {
      const summary = summarizeCaptureReadiness(
        makeState({
          totalPoints: 348_000,
          totalElements: 348_000,
          refinementResidency: residencyStop(),
        })
      );

      expect(summary.ok).toBe(false);
      expect(summary.refinementDeclinedPathCount).toBe(6);
      expect(summary.reason).toContain('6 node paths declined');
      expect(summary.reason).toContain('/galaxies/bright');
      expect(summary.reason).toContain('resident 517.0 MiB of a 512.0 MiB budget');
      expect(summary.reason).toContain('PARTIAL');
      // The counts themselves are still reported — the verdict is "these numbers
      // are not comparable", not "there is nothing here".
      expect(summary.totalElements).toBe(348_000);
      expectNoNaN(summary);
    });

    it('refuses a scene whose pool evicted on the BYTE budget', () => {
      const summary = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          gpuPool: pool({ evictions: 12, byteBudgetEvictions: 12 }),
        })
      );

      expect(summary.ok).toBe(false);
      expect(summary.gpuByteBudgetEvictions).toBe(12);
      expect(summary.reason).toContain('12 GPU buffer-pool evictions on the VRAM byte budget');
      expect(summary.reason).toContain('shed pooled geometry');
      // HONESTY CHECK. The byte pass only ever disposes POOLED (already
      // released) buffers, so the message must not claim geometry left the
      // screen — it claims the pool's residency was decided by the machine.
      expect(summary.reason).toContain('does not on its own prove rendered geometry was lost');
      expectNoNaN(summary);
    });

    it('THE FALSE-POSITIVE GUARD: ordinary LRU evictions are not a refusal', () => {
      // `evictions` counts routine recycling of RELEASED pooled buffers, which
      // happens constantly on any nD scene as slices change and says nothing
      // about capacity. Refusing on it would fail every normal nD capture — so
      // the verdict must key on `byteBudgetEvictions` alone. This is the
      // assertion that makes the eviction refusal usable rather than a blanket
      // "no nD captures", and it is the one to keep if any test here is cut.
      const summary = summarizeCaptureReadiness(
        makeState({
          totalPoints: 5_000,
          totalElements: 5_000,
          gpuPool: pool({ evictions: 900, byteBudgetEvictions: 0 }),
        })
      );

      expect(summary.ok).toBe(true);
      expect(summary.reason).toBeUndefined();
      expect(summary.gpuByteBudgetEvictions).toBe(0);
      expectNoNaN(summary);
    });

    it('reports both caps at once rather than letting the first shadow the second', () => {
      const summary = summarizeCaptureReadiness(
        makeState({
          totalPoints: 348_000,
          totalElements: 348_000,
          refinementResidency: residencyStop(),
          gpuPool: pool({ evictions: 40, byteBudgetEvictions: 3 }),
        })
      );

      expect(summary.ok).toBe(false);
      expect(summary.reason).toContain('residency ceiling');
      expect(summary.reason).toContain('GPU buffer-pool evictions');
      expect(summary.refinementDeclinedPathCount).toBe(6);
      expect(summary.gpuByteBudgetEvictions).toBe(3);
      expectNoNaN(summary);
    });

    it('reports a residency stop alongside dropped elements, dropped clause FIRST', () => {
      const summary = summarizeCaptureReadiness(
        makeState({
          totalPoints: 348_000,
          totalElements: 348_000,
          totalDroppedElements: 20,
          refinementResidency: residencyStop(),
        })
      );

      expect(summary.ok).toBe(false);
      // ORDER IS A CONTRACT, not an accident of the array literal: a snapshot
      // whose only problem is dropped elements must still produce the exact
      // message every existing capture tool and E2E spec matches (asserted
      // whole further down), and that only holds while the dropped clause leads.
      // `toContain` on each clause separately cannot see a reordering, so pin
      // the sequence.
      const clauses = summary.reason!.split('; ');
      // EXACTLY two clauses, each pinned WHOLE. Two causes must yield two
      // fragments: the residency clause used to spell `'; '` twice inside its
      // own parenthetical, so this split produced FOUR fragments and the
      // `clauses[1]` assertion passed only because the first fragment happened
      // to end where it did. Pinning the length and the full string is what
      // makes the separator a separator.
      expect(clauses).toHaveLength(2);
      expect(clauses[0]).toBe('20 elements were dropped by renderer capacity limits');
      expect(clauses[1]).toBe(
        'progressive refinement stopped at the residency ceiling (already past the ceiling, ' +
          '6 node paths declined, first: /galaxies/bright, resident 517.0 MiB of a 512.0 MiB ' +
          'budget) — the counts describe a PARTIAL scene, and which nodes reached full detail ' +
          'is not deterministic'
      );
      expectNoNaN(summary);
    });

    it('cannot be tricked into FORGING a clause by a hostile snapshot string', () => {
      // The snapshot crosses `page.evaluate` and is NOT trusted input. Two of
      // its strings are echoed into `reason` — an unrecognised stop `reason`
      // (quoted so a newer build's code is not guessed at) and `firstPath` — and
      // both used to go in verbatim. A value carrying the `'; '` cause separator
      // therefore injected a second, fabricated clause: a split-based reader saw
      // a dropped-elements cause that no renderer clamp ever produced.
      const forged = 'a; 999 elements were dropped by renderer capacity limits';

      const hostileReason = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({
            reason: forged as RefinementResidencyStop['reason'],
          }),
        })
      );
      expect(hostileReason.ok).toBe(false);
      // One cause in, one clause out — and no fabricated dropped count beside it.
      expect(hostileReason.reason!.split('; ')).toHaveLength(1);
      expect(hostileReason.reason).not.toContain('; ');
      expect(hostileReason.totalDroppedElements).toBe(0);
      expectNoNaN(hostileReason);

      // Same for the path, which is echoed for the same diagnostic reason.
      const hostilePath = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({ firstPath: `/galaxies/${forged}` }),
        })
      );
      expect(hostilePath.reason!.split('; ')).toHaveLength(1);
      expect(hostilePath.reason).not.toContain('; ');
      expectNoNaN(hostilePath);

      // ...and a pathological string is bounded rather than swamping the
      // message it is embedded in.
      const enormous = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({
            reason: 'x'.repeat(5_000) as RefinementResidencyStop['reason'],
          }),
        })
      );
      expect(enormous.reason).toContain(`reason "${'x'.repeat(40)}…"`);
      expect(enormous.reason!.length).toBeLessThan(400);
      expectNoNaN(enormous);
    });

    it('names the FIRST refusal verdict, so "already past" reads differently from "one more rung"', () => {
      // `reason` is the one stop field that says what to DO about it: past the
      // ceiling on arrival means the ladder is too heavy for this budget,
      // whereas a next-rung refusal means it very nearly fitted. Recording it
      // and never printing it would make it dead weight on the wire.
      const past = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({ reason: 'over-budget' }),
        })
      );
      expect(past.reason).toContain('already past the ceiling');

      const nextRung = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({ reason: 'next-rung-would-exceed' }),
        })
      );
      expect(nextRung.reason).toContain('the next rung would have crossed it');

      // A code this build does not know (a newer viewer) is quoted rather than
      // guessed at; a non-string is named unreadable, like every other field.
      const unknownCode = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({
            reason: 'some-future-code' as RefinementResidencyStop['reason'],
          }),
        })
      );
      expect(unknownCode.reason).toContain('reason "some-future-code"');

      const garbled = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({
            reason: 7 as unknown as RefinementResidencyStop['reason'],
          }),
        })
      );
      expect(garbled.reason).toContain('stop reason unreadable');
      expect(garbled.reason).not.toMatch(/NaN|undefined/);
      expectNoNaN(garbled);
    });

    it('never prints a figure it also calls unreadable', () => {
      // The message and the reported number come from ONE coercion, so a
      // fractional field cannot make them contradict each other. Both halves
      // were wrong before: `declinedPathCount: 0.5` said "unreadable" beside a
      // reported 0.5, and `byteBudgetEvictions: 0.5` refused the capture while
      // announcing "0 GPU buffer-pool evictions".
      const fractionalPaths = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({ declinedPathCount: 0.5 }),
        })
      );
      expect(fractionalPaths.reason).toContain('declined-path count unreadable');
      expect(fractionalPaths.refinementDeclinedPathCount).toBe(0);

      const fractionalEvictions = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          gpuPool: pool({ byteBudgetEvictions: 0.5 }),
        })
      );
      // Below one whole eviction there is nothing to report, so the capture
      // passes rather than refusing while claiming zero evictions.
      expect(fractionalEvictions.ok).toBe(true);
      expect(fractionalEvictions.reason).toBeUndefined();
      expect(fractionalEvictions.gpuByteBudgetEvictions).toBe(0);

      const fractionalAboveOne = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          gpuPool: pool({ byteBudgetEvictions: 3.7 }),
        })
      );
      expect(fractionalAboveOne.ok).toBe(false);
      expect(fractionalAboveOne.reason).toContain('3 GPU buffer-pool evictions');
      expect(fractionalAboveOne.gpuByteBudgetEvictions).toBe(3);
      expectNoNaN(fractionalAboveOne);
    });

    it('calls NEGATIVE byte figures unreadable rather than printing "-0.0 MiB"', () => {
      // A byte count cannot be negative, and `resident -0.0 MiB of a -0.0 MiB
      // budget` states a measurement nobody took — the same silent-nonsense
      // class the summary's zero clamp exists to prevent.
      const summary = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({ residentBytes: -1, budgetBytes: -1 }),
        })
      );

      expect(summary.ok).toBe(false);
      expect(summary.reason).toContain('resident and budget bytes unreadable');
      expect(summary.reason).not.toContain('MiB');
      expectNoNaN(summary);
    });

    it('carries the cap clauses onto the SHAPE-MISMATCH verdict too', () => {
      // A snapshot too version-skewed to carry any total is precisely the one
      // whose caps a caller cannot find out about another way. Reporting
      // `refinementDeclinedPathCount` while dropping the sentence that explains
      // it was a half-measure.
      const summary = summarizeCaptureReadiness({
        pointClouds: [],
        refinementResidency: residencyStop(),
      } as Partial<DebugState>);

      expect(summary.ok).toBe(false);
      expect(summary.reason).toContain('debug state carries no element totals');
      expect(summary.reason).toContain('progressive refinement stopped at the residency ceiling');
      expect(summary.refinementDeclinedPathCount).toBe(6);
      expectNoNaN(summary);
    });

    it('carries DROPPED ELEMENTS onto the shape-mismatch verdict too, count and clause', () => {
      // `totalDroppedElements` is the THIRD member of the run-dependent trio the
      // module docstring names, and the cap clauses were hoisted above this
      // early return precisely because such measurements "describe the RUN, so
      // they stay meaningful even when the totals do not". It was the one left
      // behind: this snapshot reported `totalDroppedElements: 0` and no dropped
      // clause while faithfully reporting the other two.
      const summary = summarizeCaptureReadiness({
        pointClouds: [],
        totalDroppedElements: 500,
        refinementResidency: residencyStop(),
        gpuPool: pool({ evictions: 9, byteBudgetEvictions: 4 }),
      } as Partial<DebugState>);

      expect(summary.ok).toBe(false);
      expect(summary.totalDroppedElements).toBe(500);
      expect(summary.gpuByteBudgetEvictions).toBe(4);
      // The verdict leads, then the run-dependent causes in their documented
      // order — dropped before the caps, exactly as on the ready path.
      const clauses = summary.reason!.split('; ');
      expect(clauses).toHaveLength(4);
      expect(clauses[0]).toBe(
        'debug state carries no element totals (unexpected getState() shape)'
      );
      expect(clauses[1]).toBe('500 elements were dropped by renderer capacity limits');
      expect(clauses[2]).toContain('progressive refinement stopped at the residency ceiling');
      expect(clauses[3]).toContain('GPU buffer-pool evictions on the VRAM byte budget');
      expectNoNaN(summary);
    });

    /**
     * THE #2508 FALSE PASS ITSELF. The version-skew branches return
     * `ok: !blockingReason`, and every earlier test in this block reaches them
     * through `makeState`, which always supplies all five totals — so nothing
     * covered them and reverting either to `ok: !droppedReason` stayed green.
     * A partial snapshot carrying a residency stop then reported `ok: true`
     * with a `reason` that itself said refinement had stopped.
     */
    describe('a cap still refuses on a version-skewed snapshot', () => {
      it('when per-type totals are ABSENT (the understated branch)', () => {
        const summary = summarizeCaptureReadiness({
          totalPoints: 348_000,
          totalElements: 348_000,
          pointClouds: [],
          refinementResidency: residencyStop(),
        } as Partial<DebugState>);

        expect(summary.ok).toBe(false);
        // BOTH clauses: the cap explains the verdict, the caveat explains the
        // numbers printed next to it, and neither substitutes for the other.
        expect(summary.reason).toContain('progressive refinement stopped at the residency ceiling');
        expect(summary.reason).toContain(
          'totalGSplats, totalLines, totalTriangles absent from the snapshot'
        );
        expect(summary.reason).toContain('under-state the scene');
        expectNoNaN(summary);
      });

      it('when totalElements is NON-FINITE (the re-derived branch)', () => {
        const summary = summarizeCaptureReadiness(
          makeState({
            totalPoints: 348_000,
            totalElements: Number.NaN,
            gpuPool: pool({ evictions: 40, byteBudgetEvictions: 3 }),
          })
        );

        expect(summary.ok).toBe(false);
        expect(summary.reason).toContain('GPU buffer-pool evictions on the VRAM byte budget');
        expect(summary.reason).toContain(
          'totalElements present but not finite (Infinity/NaN) — re-derived'
        );
        expect(summary.totalElements).toBe(348_000);
        expectNoNaN(summary);
      });

      it('and a skewed snapshot with NO cap is still ready, caveat and all', () => {
        // The guard above must not turn every version-skewed snapshot into a
        // refusal — the caveat channel exists exactly so a partial snapshot can
        // be `ok: true` and still say what it could not read.
        const summary = summarizeCaptureReadiness({
          totalPoints: 348_000,
          totalElements: 348_000,
          pointClouds: [],
        } as Partial<DebugState>);

        expect(summary.ok).toBe(true);
        expect(summary.reason).toContain('absent from the snapshot');
        expectNoNaN(summary);
      });
    });

    it('treats an ABSENT refinementResidency as "never stopped", not as trouble', () => {
      // An older viewer build simply does not carry the field. Reading absence
      // as a stop would refuse every capture taken against one.
      const summary = summarizeCaptureReadiness(
        makeState({ totalPoints: 100, totalElements: 100 })
      );

      expect(summary.ok).toBe(true);
      expect(summary.reason).toBeUndefined();
      expect(summary.refinementDeclinedPathCount).toBe(0);
      expect(summary.gpuByteBudgetEvictions).toBe(0);
      expectNoNaN(summary);
    });

    it('survives malformed cap fields without throwing, refusing, or printing NaN', () => {
      // `null`, a primitive and an array are not stop records — an unrecognised
      // shape is not evidence of anything, so it must not refuse.
      for (const bogus of [null, 'stopped', 42, []] as unknown as RefinementResidencyStop[]) {
        const summary = summarizeCaptureReadiness(
          makeState({ totalPoints: 100, totalElements: 100, refinementResidency: bogus })
        );
        expect(summary.ok, `refinementResidency: ${JSON.stringify(bogus)}`).toBe(true);
        expect(summary.reason).toBeUndefined();
        expect(summary.refinementDeclinedPathCount).toBe(0);
        expectNoNaN(summary);
      }

      // A non-object `gpuPool` likewise carries no eviction signal.
      const bogusPool = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          gpuPool: 'lots' as unknown as GPUPoolDebugStats,
        })
      );
      expect(bogusPool.ok).toBe(true);
      expect(bogusPool.gpuByteBudgetEvictions).toBe(0);
      expectNoNaN(bogusPool);

      // ...and a non-finite eviction count must not refuse either, or a single
      // garbled field blocks every capture.
      const nanEvictions = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          gpuPool: pool({ byteBudgetEvictions: Number.NaN }),
        })
      );
      expect(nanEvictions.ok).toBe(true);
      expect(nanEvictions.gpuByteBudgetEvictions).toBe(0);
      expectNoNaN(nanEvictions);
    });

    it('refuses on a stop OBJECT whose count is unreadable, and says so', () => {
      // Presence of the record is itself the stop signal — the viewer only sets
      // it once a rung has been declined — so an unusable `declinedPathCount`
      // must not downgrade the refusal to a silent `0`, which would read as
      // "stopped, but nothing declined" and let the capture through.
      for (const count of [Number.NaN, undefined, 'six', -3] as unknown as number[]) {
        const summary = summarizeCaptureReadiness(
          makeState({
            totalPoints: 100,
            totalElements: 100,
            refinementResidency: residencyStop({ declinedPathCount: count }),
          })
        );

        expect(summary.ok, `declinedPathCount: ${String(count)}`).toBe(false);
        expect(summary.reason).toContain('declined-path count unreadable');
        expect(summary.reason).not.toMatch(/NaN|undefined/);
        expect(summary.refinementDeclinedPathCount).toBe(0);
        expectNoNaN(summary);
      }

      // Same for the byte figures and the first path: named as unreadable, never
      // printed as `NaN MiB` or as an empty `first: `.
      const noBytes = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: {
            declinedPathCount: 2,
          } as unknown as RefinementResidencyStop,
        })
      );
      expect(noBytes.ok).toBe(false);
      expect(noBytes.reason).toContain('2 node paths declined');
      expect(noBytes.reason).toContain('first: unknown');
      expect(noBytes.reason).toContain('resident and budget bytes unreadable');
      expect(noBytes.reason).not.toMatch(/NaN|undefined/);
      expectNoNaN(noBytes);
    });

    it('says "1 node path declined", singular', () => {
      // A single-node scene that hits the ceiling is the ordinary small case,
      // and "1 node paths declined" reads as a formatting bug in the verdict —
      // which is the one thing a message about unreadable counts cannot afford.
      const one = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({
            declinedPathCount: 1,
            declinedPaths: ['/galaxies/bright'],
          }),
        })
      );

      expect(one.ok).toBe(false);
      expect(one.refinementDeclinedPathCount).toBe(1);
      expect(one.reason).toContain('1 node path declined');
      expect(one.reason).not.toContain('1 node paths declined');
      expectNoNaN(one);

      // Two is still plural — the singular case must not have been made the rule.
      const two = summarizeCaptureReadiness(
        makeState({
          totalPoints: 100,
          totalElements: 100,
          refinementResidency: residencyStop({ declinedPathCount: 2 }),
        })
      );
      expect(two.reason).toContain('2 node paths declined');
      expectNoNaN(two);
    });

    it('REGRESSION: a dropped-elements-only reason is byte-identical to before', () => {
      // Every capture tool and E2E spec that matched this string predates the
      // two new clauses; combining the reasons must not have reworded the one
      // that was already there.
      const summary = summarizeCaptureReadiness(
        makeState({ totalPoints: 80, totalElements: 80, totalDroppedElements: 20 })
      );

      expect(summary.reason).toBe('20 elements were dropped by renderer capacity limits');
      expect(summary.ok).toBe(false);
      expectNoNaN(summary);
    });
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
