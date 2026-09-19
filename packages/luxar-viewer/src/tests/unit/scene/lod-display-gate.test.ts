/**
 * Unit tests for the pure never-downgrade display gate
 * (`shouldHoldPreviousDisplay`) and its subtree aggregation
 * (`subtreeDisplayProgress`) — no THREE, structural node stubs only.
 */

import { describe, expect, it } from 'vitest';

import {
  displayedGeometricErrorFraction,
  displayedQualityFraction,
  shouldHoldPreviousDisplay,
  subtreeDisplayProgress,
  type HoldCandidate,
  type ProgressNode,
} from '../../../scene/lod-display-gate';
import type { FreshnessChild } from '../../../scene/lod-freshness';

/** A stamped leaf node (the shape the commit helpers produce). */
function leaf(
  nodeType: string,
  count: number,
  opts: { version?: number; complete?: boolean; visible?: boolean } = {}
): ProgressNode {
  const countKey =
    nodeType === 'points'
      ? 'visiblePointCount'
      : nodeType === 'lines'
        ? 'visibleSegmentCount'
        : 'visibleSplatCount';
  return {
    visible: opts.visible ?? true,
    userData: {
      nodeType,
      loadedViewVersion: opts.version ?? 2,
      [countKey]: count,
      committedLadderComplete: opts.complete ?? true,
    },
  };
}

/** An unstamped group node with children (partition / lod wrapper shape). */
function group(children: ProgressNode[], visible = true): ProgressNode {
  return { visible, children };
}

// ────────────────────────────────────────────────────────────────────────
// subtreeDisplayProgress — the aggregate fold
// ────────────────────────────────────────────────────────────────────────

describe('subtreeDisplayProgress', () => {
  it('sums counts over a partition of stamped leaves', () => {
    const root = group([leaf('gsplats', 10), leaf('gsplats', 20), leaf('gsplats', 30)]);
    expect(subtreeDisplayProgress(root, 2)).toEqual({
      count: 60,
      complete: true,
      fresh: true,
      nodeType: 'gsplats',
      energy: null,
      quality: null,
    });
  });

  it('is incomplete while ANY visible leaf ladder is still streaming', () => {
    const root = group([leaf('gsplats', 10), leaf('gsplats', 20, { complete: false })]);
    expect(subtreeDisplayProgress(root, 2)!.complete).toBe(false);
  });

  it('is stale while ANY visible leaf is stamped for an older version', () => {
    const root = group([leaf('gsplats', 10), leaf('gsplats', 20, { version: 1 })]);
    expect(subtreeDisplayProgress(root, 2)!.fresh).toBe(false);
    // Freshness untracked (version null) → vacuously fresh.
    expect(subtreeDisplayProgress(root, null)!.fresh).toBe(true);
  });

  it('ignores the ROOT visible flag but respects descendant visibility (inner lod_group toggling)', () => {
    // Hidden root (the registry hides the placeholder of a held child) with
    // an inner lod_group that shows only ONE of its levels.
    const innerLodGroup = group([
      leaf('gsplats', 100, { visible: false }), // hidden coarse level
      leaf('gsplats', 400, { complete: false }), // displayed streaming level
    ]);
    const root = group([innerLodGroup], /* visible= */ false);
    expect(subtreeDisplayProgress(root, 2)).toEqual({
      count: 400,
      complete: false,
      fresh: true,
      nodeType: 'gsplats',
      energy: null,
      quality: null,
    });
  });

  it('skips an entire invisible branch (its leaves neither count nor block completeness)', () => {
    const hiddenBranch = group([leaf('gsplats', 999, { complete: false, version: 1 })], false);
    const root = group([hiddenBranch, leaf('gsplats', 10)]);
    expect(subtreeDisplayProgress(root, 2)).toEqual({
      count: 10,
      complete: true,
      fresh: true,
      nodeType: 'gsplats',
      energy: null,
      quality: null,
    });
  });

  it('handles arbitrary nesting depth (lod → partition → lod → leaves)', () => {
    const root = group([group([group([leaf('gsplats', 5), group([leaf('gsplats', 7)])])])]);
    expect(subtreeDisplayProgress(root, 2)!.count).toBe(12);
  });

  it('returns null when no stamped leaf exists (nothing committed / not a geometry subtree)', () => {
    expect(subtreeDisplayProgress(group([group([])]), 2)).toBeNull();
    // A leaf missing its count stamp contributes nothing.
    const unstamped: ProgressNode = { visible: true, userData: { nodeType: 'gsplats' } };
    expect(subtreeDisplayProgress(group([unstamped]), 2)).toBeNull();
  });

  it("reports 'mixed' when leaf geometry types differ", () => {
    const root = group([leaf('gsplats', 10), leaf('points', 20)]);
    expect(subtreeDisplayProgress(root, 2)!.nodeType).toBe('mixed');
  });

  it('unstamped intermediate leaves never block completeness or freshness', () => {
    const partial: ProgressNode = { visible: true, userData: { nodeType: 'gsplats' } };
    const root = group([partial, leaf('gsplats', 10)]);
    expect(subtreeDisplayProgress(root, 2)).toEqual({
      count: 10,
      complete: true,
      fresh: true,
      nodeType: 'gsplats',
      energy: null,
      quality: null,
    });
  });
});

// ────────────────────────────────────────────────────────────────────────
// shouldHoldPreviousDisplay — leaf semantics (moved from lod-freshness.test.ts)
// ────────────────────────────────────────────────────────────────────────

describe('shouldHoldPreviousDisplay (leaf levels)', () => {
  /** A gsplats level with committed count + version + ladder-complete stamps. */
  function shown(visibleSplatCount: number, loadedViewVersion = 2): FreshnessChild {
    return {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion,
          visibleSplatCount,
          committedLadderComplete: true,
        },
      },
    };
  }

  /** A streaming gsplats aspiration: committed stamp says ladder incomplete. */
  function streamingAsp(
    visibleSplatCount: number,
    overrides: Partial<HoldCandidate> = {}
  ): HoldCandidate {
    return {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion: 2,
          visibleSplatCount,
          committedLadderComplete: false,
        },
      },
      ...overrides,
    };
  }

  it('holds while the committed ladder stamp is incomplete and the count is below prev', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100), 2)).toBe(true);
  });

  it('releases on committed ladder completion (stamp flips at the final COMMIT)', () => {
    // The stamp — not the loader's live hasMoreLODs — drives the release: it
    // is written in the same synchronous commit as the final count, so there
    // is no fetch-resolved-but-not-committed window in which the gate could
    // re-show a partial level (the premature-release race, by construction).
    const asp = streamingAsp(50);
    asp.object.userData!.committedLadderComplete = true;
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(false);
  });

  it('releases at the committed-count crossover (aspiration caught up)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(100), shown(100), 2)).toBe(false);
    expect(shouldHoldPreviousDisplay(streamingAsp(150), shown(100), 2)).toBe(false);
  });

  it('releases when the aspiration ladder failed (degrade to ungated behavior)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10, { failed: true }), shown(100), 2)).toBe(
      false
    );
  });

  it('never holds a stale prev (staleness beats quality)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100, 1), 2)).toBe(false);
  });

  it('never holds an empty or count-untracked prev (fast first paint)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(0), 2)).toBe(false);
    const untrackedPrev: FreshnessChild = {
      ready: true,
      object: { userData: { nodeType: 'gsplats', loadedViewVersion: 2 } },
    };
    expect(shouldHoldPreviousDisplay(streamingAsp(10), untrackedPrev, 2)).toBe(false);
    expect(shouldHoldPreviousDisplay(streamingAsp(10), undefined, 2)).toBe(false);
  });

  it('does not hold against an aspiration with an untracked count', () => {
    const asp = streamingAsp(0);
    delete asp.object.userData!.visibleSplatCount;
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(false);
  });

  it('does not hold when the ladder stamp is missing (single-LOD / unstamped level)', () => {
    const asp = streamingAsp(10);
    delete asp.object.userData!.committedLadderComplete;
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(false);
  });

  it('does not compare counts across geometry types (mixed-type ladder)', () => {
    const linesPrev: FreshnessChild = {
      ready: true,
      object: { userData: { nodeType: 'lines', loadedViewVersion: 2, visibleSegmentCount: 100 } },
    };
    expect(shouldHoldPreviousDisplay(streamingAsp(10), linesPrev, 2)).toBe(false);
  });

  it('falls back to readiness for prev when freshness is untracked (version null)', () => {
    // No version wiring: a READY prev can be held...
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100), null)).toBe(true);
    // ...a not-ready prev cannot.
    const notReady = shown(100);
    notReady.ready = false;
    expect(shouldHoldPreviousDisplay(streamingAsp(10), notReady, null)).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// shouldHoldPreviousDisplay — nested-group sides (subtree aggregation)
// ────────────────────────────────────────────────────────────────────────

describe('shouldHoldPreviousDisplay (nested-group levels)', () => {
  /** A registry child whose object is a group SUBTREE (deferred-GROUP shape). */
  function groupChild(subtree: ProgressNode, ready = true): HoldCandidate {
    return { ready, object: subtree as FreshnessChild['object'] };
  }

  /** A complete leaf prev (the overview coarse cap shape). */
  function leafPrev(count: number, version = 2): FreshnessChild {
    return {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion: version,
          visibleSplatCount: count,
          committedLadderComplete: true,
        },
      },
    };
  }

  it('holds a leaf prev over a streaming group aspiration until the aggregate crosses', () => {
    // The overview shape: coarse cap (leaf, complete, 100) vs the fine
    // kind=partition branch whose parts are mid-ladder.
    const streamingBranch = group([
      leaf('gsplats', 10, { complete: false }),
      leaf('gsplats', 20, { complete: false }),
    ]);
    expect(shouldHoldPreviousDisplay(groupChild(streamingBranch), leafPrev(100), 2)).toBe(true);

    // Aggregate count crossover → release (the ladders keep streaming visibly).
    const crossed = group([
      leaf('gsplats', 60, { complete: false }),
      leaf('gsplats', 50, { complete: false }),
    ]);
    expect(shouldHoldPreviousDisplay(groupChild(crossed), leafPrev(100), 2)).toBe(false);
  });

  it('releases when every part ladder committed complete (even below the prev count)', () => {
    const completeBranch = group([leaf('gsplats', 30), leaf('gsplats', 40)]);
    expect(shouldHoldPreviousDisplay(groupChild(completeBranch), leafPrev(100), 2)).toBe(false);
  });

  it('holds a GROUP prev over a streaming leaf aspiration (aggregate on the prev side)', () => {
    const fullBranch = group([leaf('gsplats', 500), leaf('gsplats', 500)]);
    const streamingLeaf: HoldCandidate = {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion: 2,
          visibleSplatCount: 10,
          committedLadderComplete: false,
        },
      },
    };
    expect(shouldHoldPreviousDisplay(streamingLeaf, groupChild(fullBranch), 2)).toBe(true);
  });

  it('never holds a group prev whose visible leaves are stale (aggregate freshness)', () => {
    // Closes the vacuous-fresh blind spot: a group child has no direct
    // freshness stamp, but its leaves do — an old-slice subtree must not be
    // held over a fresh streaming aspiration.
    const staleBranch = group([leaf('gsplats', 500, { version: 1 })]);
    const streamingLeaf: HoldCandidate = {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion: 2,
          visibleSplatCount: 10,
          committedLadderComplete: false,
        },
      },
    };
    expect(shouldHoldPreviousDisplay(streamingLeaf, groupChild(staleBranch), 2)).toBe(false);
  });

  it('group-vs-group: holds while the streaming subtree is behind the displayed one', () => {
    const shownBranch = group([leaf('gsplats', 300), leaf('gsplats', 300)]);
    const streamingBranch = group([leaf('gsplats', 50, { complete: false })]);
    expect(shouldHoldPreviousDisplay(groupChild(streamingBranch), groupChild(shownBranch), 2)).toBe(
      true
    );
    expect(shouldHoldPreviousDisplay(groupChild(shownBranch), groupChild(streamingBranch), 2)).toBe(
      false // complete aspiration never held back
    );
  });

  it("refuses to compare a 'mixed' subtree (counts across types are meaningless)", () => {
    const mixedStreaming = group([
      leaf('gsplats', 10, { complete: false }),
      leaf('points', 10, { complete: false }),
    ]);
    expect(shouldHoldPreviousDisplay(groupChild(mixedStreaming), leafPrev(100), 2)).toBe(false);
  });

  it("refuses even when BOTH sides are 'mixed' (the asp.nodeType==='mixed' guard, not just type mismatch)", () => {
    // Here asp.nodeType === prev.nodeType === 'mixed', so the type-MISMATCH
    // half of the guard is satisfied; only the explicit `=== 'mixed'` disjunct
    // blocks the hold. A mutant dropping that disjunct would wrongly hold.
    const mixedStreaming = group([
      leaf('gsplats', 10, { complete: false }),
      leaf('points', 10, { complete: false }),
    ]);
    const mixedPrev = group([leaf('gsplats', 500), leaf('points', 500)]);
    expect(shouldHoldPreviousDisplay(groupChild(mixedStreaming), groupChild(mixedPrev), 2)).toBe(
      false
    );
  });

  it('is inert for a group with no stamped leaves (unknown ⇒ swap, today’s behavior)', () => {
    expect(shouldHoldPreviousDisplay(groupChild(group([group([])])), leafPrev(100), 2)).toBe(false);
  });

  it('inner lod_group toggling shapes the aggregate (only the displayed level counts)', () => {
    // adaptive-under-lod shape: the aspiration subtree contains an inner
    // lod_group whose hidden fine level would dwarf the prev count — only the
    // VISIBLE inner level participates.
    const innerLodGroup = group([
      leaf('gsplats', 40, { complete: false }), // displayed inner level, streaming
      leaf('gsplats', 4000, { visible: false }), // hidden inner level
    ]);
    expect(shouldHoldPreviousDisplay(groupChild(group([innerLodGroup])), leafPrev(100), 2)).toBe(
      true // 40 < 100 and streaming → hold
    );
  });
});

// ────────────────────────────────────────────────────────────────────────
// The committed-energy release (Q·e quality stamps)
// ────────────────────────────────────────────────────────────────────────

describe('committed-energy release', () => {
  /** A complete gsplats prev with a committed count. */
  function shown(visibleSplatCount: number): FreshnessChild {
    return {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion: 2,
          visibleSplatCount,
          committedLadderComplete: true,
        },
      },
    };
  }

  /** A streaming gsplats aspiration with an optional committed-energy stamp. */
  function streamingAsp(visibleSplatCount: number, energy?: number): HoldCandidate {
    return {
      ready: true,
      object: {
        userData: {
          nodeType: 'gsplats',
          loadedViewVersion: 2,
          visibleSplatCount,
          committedLadderComplete: false,
          ...(energy !== undefined ? { committedEnergyFraction: energy } : {}),
        },
      },
    };
  }

  /** A stamped leaf carrying both the dynamic e stamp and the static w. */
  function energyLeaf(
    count: number,
    energy: number | undefined,
    weight: number | undefined,
    opts: { complete?: boolean } = {}
  ): ProgressNode {
    return {
      visible: true,
      userData: {
        nodeType: 'gsplats',
        loadedViewVersion: 2,
        visibleSplatCount: count,
        committedLadderComplete: opts.complete ?? false,
        ...(energy !== undefined ? { committedEnergyFraction: energy } : {}),
        ...(weight !== undefined ? { attrs: { level_stats: { reference_energy: weight } } } : {}),
      },
    };
  }

  it('releases a stamped streaming aspiration at the energy threshold, far below the count crossover (the headline)', () => {
    // 10 vs 100 committed: the count rule would hold, but the committed
    // prefix already carries 70% of the level's energy — swap now, the rest
    // of the ladder streams visibly.
    expect(shouldHoldPreviousDisplay(streamingAsp(10, 0.7), shown(100), 2)).toBe(false);
  });

  it('releases exactly AT the threshold (>=, not >)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10, 0.6), shown(100), 2)).toBe(false);
  });

  it('keeps holding while the committed energy is below the threshold', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10, 0.3), shown(100), 2)).toBe(true);
    expect(shouldHoldPreviousDisplay(streamingAsp(10, 0.59), shown(100), 2)).toBe(true);
  });

  it('falls back to the count rule on an unstamped aspiration (legacy dataset)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100), 2)).toBe(true);
    expect(shouldHoldPreviousDisplay(streamingAsp(100), shown(100), 2)).toBe(false);
  });

  it('low-energy stamped aspiration still releases at the count crossover (energy never delays release)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(100, 0.3), shown(100), 2)).toBe(false);
  });

  it('does NOT energy-release on a DOWNGRADE (zoom-out): holds the finer prev until the coarse ladder completes', () => {
    // Direction matters. On zoom-out the aspiration is the COARSER level and
    // `prev` is the finer level that was on screen. Releasing at 60% of the
    // coarse level's OWN energy would show fine-complete -> coarse-60% ->
    // coarse-100% — a transient dip BELOW both, violating the never-downgrade
    // contract. The energy short-circuit is upgrade-only; on a downgrade the
    // gate falls through to the count rule (holds until the coarse ladder
    // completes). isUpgrade=false models the coarser-aspiration case.
    expect(shouldHoldPreviousDisplay(streamingAsp(10, 0.7), shown(100), 2, false)).toBe(true);
    // Same inputs as an UPGRADE still release at the threshold (unchanged).
    expect(shouldHoldPreviousDisplay(streamingAsp(10, 0.7), shown(100), 2, true)).toBe(false);
    // Downgrade releases once the coarse ladder is committed-complete.
    const complete = streamingAsp(10, 1.0);
    complete.object.userData!.committedLadderComplete = true;
    expect(shouldHoldPreviousDisplay(complete, shown(100), 2, false)).toBe(false);
  });

  it('aggregates a group aspiration as the w-weighted mean of its leaves', () => {
    // Heavy leaf nearly done, light leaf barely started → mean 0.82 → release.
    const heavyDone = group([energyLeaf(10, 0.9, 90), energyLeaf(5, 0.1, 10)]);
    expect(subtreeDisplayProgress(heavyDone, 2)!.energy).toBeCloseTo(0.82, 10);
    const aspRelease: HoldCandidate = { ready: true, object: heavyDone as never };
    expect(shouldHoldPreviousDisplay(aspRelease, shown(1000), 2)).toBe(false);

    // Swap the weights: the barely-started leaf dominates → mean 0.18 → hold.
    const heavyStarting = group([energyLeaf(10, 0.9, 10), energyLeaf(5, 0.1, 90)]);
    expect(subtreeDisplayProgress(heavyStarting, 2)!.energy).toBeCloseTo(0.18, 10);
    const aspHold: HoldCandidate = { ready: true, object: heavyStarting as never };
    expect(shouldHoldPreviousDisplay(aspHold, shown(1000), 2)).toBe(true);
  });

  it('a partially stamped subtree falls back WHOLE to the count rule (never blend measured and guessed)', () => {
    // One leaf stamped high, one missing its e stamp → aggregate null.
    const mixed = group([energyLeaf(10, 0.95, 90), energyLeaf(5, undefined, 10)]);
    expect(subtreeDisplayProgress(mixed, 2)!.energy).toBeNull();
    const asp: HoldCandidate = { ready: true, object: mixed as never };
    expect(shouldHoldPreviousDisplay(asp, shown(1000), 2)).toBe(true);
    // A missing static w poisons the aggregate the same way.
    const noW = group([energyLeaf(10, 0.95, 90), energyLeaf(5, 0.5, undefined)]);
    expect(subtreeDisplayProgress(noW, 2)!.energy).toBeNull();
  });

  it('excludes known-empty leaves from the weighted mean (mirrors the empty-level display guard)', () => {
    // The empty part carries no stamps at this slice; the stamped non-empty
    // part alone decides the aggregate.
    const withEmpty = group([energyLeaf(0, undefined, undefined), energyLeaf(10, 0.7, 50)]);
    expect(subtreeDisplayProgress(withEmpty, 2)!.energy).toBeCloseTo(0.7, 10);
    const asp: HoldCandidate = { ready: true, object: withEmpty as never };
    expect(shouldHoldPreviousDisplay(asp, shown(1000), 2)).toBe(false);
  });

  it('an all-empty subtree reports unknown energy (null), not 0 or 1', () => {
    const allEmpty = group([energyLeaf(0, undefined, undefined)]);
    expect(subtreeDisplayProgress(allEmpty, 2)!.energy).toBeNull();
  });
});
describe('displayedQualityFraction (Q·e readout)', () => {
  it('multiplies the measured level quality Q into the committed energy e', () => {
    const node: ProgressNode = {
      visible: true,
      userData: {
        nodeType: 'gsplats',
        visibleSplatCount: 10,
        committedLadderComplete: false,
        committedEnergyFraction: 0.8,
        attrs: { level_stats: { quality: 0.75, reference_energy: 100 } },
      },
    };
    expect(displayedQualityFraction(node)).toBeCloseTo(0.6, 10);
  });

  it('reports unmeasured when the level has energy but no quality stamp', () => {
    const node: ProgressNode = {
      visible: true,
      userData: {
        nodeType: 'gsplats',
        visibleSplatCount: 10,
        committedLadderComplete: false,
        committedEnergyFraction: 0.8,
        attrs: { level_stats: { reference_energy: 100 } },
      },
    };
    expect(displayedQualityFraction(node)).toBeNull();
  });

  it('reports a partially quality-stamped subtree as unmeasured', () => {
    const measured: ProgressNode = {
      userData: {
        nodeType: 'gsplats',
        visibleSplatCount: 10,
        committedEnergyFraction: 0.8,
        attrs: { level_stats: { quality: 0.75, reference_energy: 100 } },
      },
    };
    const unmeasured: ProgressNode = {
      userData: {
        nodeType: 'gsplats',
        visibleSplatCount: 10,
        committedEnergyFraction: 0.9,
        attrs: { level_stats: { reference_energy: 100 } },
      },
    };
    expect(displayedQualityFraction(group([measured, unmeasured]))).toBeNull();
  });

  it('returns null on an unstamped dataset or an empty subtree', () => {
    const unstamped: ProgressNode = {
      visible: true,
      userData: { nodeType: 'gsplats', visibleSplatCount: 10, committedLadderComplete: false },
    };
    expect(displayedQualityFraction(unstamped)).toBeNull();
    expect(displayedQualityFraction({ visible: true, children: [] })).toBeNull();
  });
});

describe('displayedGeometricErrorFraction (mesh error readout)', () => {
  it('reports mesh error without an energy pair and keeps it out of mixture Q', () => {
    const mesh: ProgressNode = {
      userData: {
        nodeType: 'mesh',
        visibleTriangleCount: 20,
        attrs: { level_stats: { geometric_error: 0.125 } },
      },
    };
    expect(displayedGeometricErrorFraction(mesh)).toBeCloseTo(0.125, 10);
    expect(displayedQualityFraction(mesh)).toBeNull();
  });

  it('takes the worst mesh error and ignores non-mesh quality currencies', () => {
    const root = group([
      {
        userData: {
          nodeType: 'mesh',
          visibleTriangleCount: 20,
          attrs: { level_stats: { geometric_error: 0.05 } },
        },
      },
      {
        userData: {
          nodeType: 'mesh',
          visibleTriangleCount: 30,
          attrs: { level_stats: { geometric_error: 0.12 } },
        },
      },
      {
        userData: {
          nodeType: 'gsplats',
          visibleSplatCount: 10,
          attrs: { level_stats: { quality: 0.8, reference_energy: 2 } },
        },
      },
    ]);
    expect(displayedGeometricErrorFraction(root)).toBeCloseTo(0.12, 10);
  });

  it('returns null when any visible non-empty mesh leaf is unstamped', () => {
    const stamped: ProgressNode = {
      userData: {
        nodeType: 'mesh',
        visibleTriangleCount: 20,
        attrs: { level_stats: { geometric_error: 0.05 } },
      },
    };
    const legacy: ProgressNode = {
      userData: { nodeType: 'mesh', visibleTriangleCount: 10 },
    };
    expect(displayedGeometricErrorFraction(group([stamped, legacy]))).toBeNull();
  });
});
