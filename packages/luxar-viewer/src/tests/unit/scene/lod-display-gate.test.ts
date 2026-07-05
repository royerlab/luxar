/**
 * Unit tests for the pure never-downgrade display gate
 * (`shouldHoldPreviousDisplay`) and its subtree aggregation
 * (`subtreeDisplayProgress`) — no THREE, structural node stubs only.
 */

import { describe, expect, it } from 'vitest';

import {
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
