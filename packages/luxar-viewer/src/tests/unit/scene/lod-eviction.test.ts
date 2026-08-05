/**
 * Direct tests for the resident-byte eviction policy (`scene/lod-eviction.ts`).
 *
 * The policy is otherwise exercised through the LODGroupRegistry suite
 * (`lod-group-registry.test.ts` — frustum-aware selection & eviction, blend
 * partner protection); these tests pin the module's own candidate filter in
 * isolation, in particular the ON-SCREEN protection: eviction must never
 * release a level that is rendering this frame (`object.visible === true`),
 * which includes the cross-fade blend partner — a level that is visible but
 * is NOT the entry's `displayedChildIndex` (campaign-4 iter-7, bug B).
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import {
  enforceResidentByteBudget,
  type EvictableChild,
  type EvictableEntry,
} from '../../../scene/lod-eviction';
import type { VisibilityNode } from '../../../utils/object-visibility';
import type { BoundingBox } from '../../../scene/scene-manager/clipping/bounds-math';

/** A ready, previously-shown child with a release spy and a visibility flag. */
function child(visible: boolean, lastVisibleTick = 1): EvictableChild & { release: () => void } {
  return {
    ready: true,
    release: vi.fn() as unknown as () => void,
    lastVisibleTick,
    object: { visible },
  };
}

function run(entry: EvictableEntry, bytes: () => number, budget = 100): void {
  enforceResidentByteBudget({
    entries: [entry],
    camera: new THREE.Camera(),
    frustum: new THREE.Frustum(),
    getResidentByteBudget: () => budget,
    getResidentBytes: bytes,
    // World box irrelevant to the candidate filter under test; null keeps the
    // spatial ranking neutral (on-screen, distance 0).
    computeWorldBox: () => null,
  });
}

describe('enforceResidentByteBudget — on-screen protection', () => {
  it('never releases a VISIBLE child that is not the displayed index (blend partner)', () => {
    // displayed = 2; child 1 is visible too (mid cross-fade). Pre-fix only the
    // displayed index was protected and the visible partner was the coldest
    // ordinary candidate. Child 0 (hidden) must be evicted instead.
    const hidden = child(false, 1);
    const partner = child(true, 2);
    const displayed = child(true, 3);
    const entry: EvictableEntry = {
      children: [hidden, partner, displayed],
      activeChildIndex: 2,
      displayedChildIndex: 2,
    };
    let resident = 300;
    (hidden.release as ReturnType<typeof vi.fn>).mockImplementation(() => {
      resident -= 100;
    });
    run(entry, () => resident);
    expect(hidden.release).toHaveBeenCalledTimes(1); // hidden level gives back VRAM
    expect(partner.release).not.toHaveBeenCalled(); // on-screen partner protected
    expect(displayed.release).not.toHaveBeenCalled();
  });

  it('releases NOTHING (stays over budget) when every candidate is on screen', () => {
    // Both releasable levels are rendering (a two-level dissolve): blanking the
    // screen is worse than briefly exceeding the byte budget.
    const partner = child(true, 1);
    const displayed = child(true, 2);
    const entry: EvictableEntry = {
      children: [partner, displayed],
      activeChildIndex: 1,
      displayedChildIndex: 1,
    };
    run(entry, () => 300);
    expect(partner.release).not.toHaveBeenCalled();
    expect(displayed.release).not.toHaveBeenCalled();
  });

  it('still evicts hidden levels normally (protection is visibility-scoped)', () => {
    const hiddenCold = child(false, 1);
    const hiddenWarm = child(false, 5);
    const displayed = child(true, 9);
    const entry: EvictableEntry = {
      children: [hiddenCold, hiddenWarm, displayed],
      activeChildIndex: 2,
      displayedChildIndex: 2,
    };
    let resident = 300;
    const drop = () => {
      resident -= 100;
    };
    (hiddenCold.release as ReturnType<typeof vi.fn>).mockImplementation(drop);
    (hiddenWarm.release as ReturnType<typeof vi.fn>).mockImplementation(drop);
    run(entry, () => resident, 100);
    // Coldest hidden first; frees until back under budget (2 releases needed).
    expect(hiddenCold.release).toHaveBeenCalledTimes(1);
    expect(hiddenWarm.release).toHaveBeenCalledTimes(1);
    expect(displayed.release).not.toHaveBeenCalled();
  });

  it('treats a child without an object shape as not-on-screen (structural compat)', () => {
    // EvictableChild.object is optional (minimal structural shapes); a child
    // that omits it keeps the pre-existing candidate behaviour.
    const bare: EvictableChild = { ready: true, release: vi.fn(), lastVisibleTick: 1 };
    const displayed = child(true, 2);
    const entry: EvictableEntry = {
      children: [bare, displayed],
      activeChildIndex: 1,
      displayedChildIndex: 1,
    };
    let resident = 200;
    (bare.release as ReturnType<typeof vi.fn>).mockImplementation(() => {
      resident -= 100;
    });
    run(entry, () => resident);
    expect(bare.release).toHaveBeenCalledTimes(1);
  });
});

describe('enforceResidentByteBudget — hidden-layer (effective visibility)', () => {
  /**
   * A ready, previously-shown child sitting under `layer` — the scene-graph
   * shape a layer authored `visible=false` (or toggled off in the layers panel)
   * produces: the LAYER carries the hidden flag while the level's own `visible`
   * stays whatever the registry last set it to.
   */
  function childUnder(
    layer: VisibilityNode,
    visible: boolean,
    lastVisibleTick: number
  ): EvictableChild & { release: () => void } {
    const c = child(visible, lastVisibleTick);
    c.object = { visible, parent: layer };
    return c;
  }

  /** Sum of un-released 100-byte levels, so releases actually free bytes. */
  function residentModel(children: readonly (EvictableChild & { release: () => void })[]) {
    return () =>
      children.reduce((sum, c) => {
        const fired = (c.release as ReturnType<typeof vi.fn>).mock.calls.length > 0;
        return sum + (fired ? 0 : 100);
      }, 0);
  }

  it('evicts every ready level of a hidden layer, including the displayed one', () => {
    // Nothing of a hidden layer is on screen, so its levels are ordinary cold
    // candidates: neither the `object.visible === true` protection (the flag is
    // stale — the ancestor is what hides it) nor the displayed-index exemption
    // applies. Pre-fix only the flag-hidden level 0 could be reclaimed, leaving
    // the bulk (a hidden layer's finest level) resident forever.
    const layer: VisibilityNode = { visible: false };
    const cold = childUnder(layer, false, 1);
    const partner = childUnder(layer, true, 2);
    const displayed = childUnder(layer, true, 3);
    const children = [cold, partner, displayed];
    const entry: EvictableEntry = {
      children,
      activeChildIndex: 2,
      displayedChildIndex: 2,
    };
    run(entry, residentModel(children), 50); // budget forces everything out
    expect(cold.release).toHaveBeenCalledTimes(1);
    expect(partner.release).toHaveBeenCalledTimes(1);
    expect(displayed.release).toHaveBeenCalledTimes(1);
  });

  it('walks the WHOLE ancestor chain (a hidden grandparent hides the level)', () => {
    const hiddenRoot: VisibilityNode = { visible: false };
    const group: VisibilityNode = { visible: true, parent: hiddenRoot };
    const displayed = childUnder(group, true, 1);
    const entry: EvictableEntry = {
      children: [displayed],
      activeChildIndex: 0,
      displayedChildIndex: 0,
    };
    run(entry, residentModel([displayed]), 50);
    expect(displayed.release).toHaveBeenCalledTimes(1);
  });

  it('leaves a VISIBLE layer untouched: on-screen and displayed levels protected', () => {
    // Regression guard — with every ancestor visible the candidate filter is
    // byte-identical to before: only the flag-hidden cold level is reclaimed,
    // and the pass stops over budget rather than blanking the screen.
    const layer: VisibilityNode = { visible: true };
    const cold = childUnder(layer, false, 1);
    const partner = childUnder(layer, true, 2);
    const displayed = childUnder(layer, true, 3);
    const children = [cold, partner, displayed];
    const entry: EvictableEntry = {
      children,
      activeChildIndex: 2,
      displayedChildIndex: 2,
    };
    run(entry, residentModel(children), 50);
    expect(cold.release).toHaveBeenCalledTimes(1);
    expect(partner.release).not.toHaveBeenCalled();
    expect(displayed.release).not.toHaveBeenCalled();
  });

  it('clears object.visible when it releases a hidden-layer level (no stale-visible window)', () => {
    // A hidden layer's level can carry a stale `visible === true` (the ancestor,
    // not the level's own flag, hides it). Once demoted the buffer is released
    // and no longer ready, so the flag MUST be cleared — otherwise a same-frame
    // re-show of the layer would expose the stale geometry for one frame before
    // the gated reload commits.
    const layer: VisibilityNode = { visible: false };
    const staleVisible = childUnder(layer, true, 1);
    const children = [staleVisible];
    const entry: EvictableEntry = {
      children,
      activeChildIndex: 0,
      displayedChildIndex: 0,
    };
    expect(staleVisible.object?.visible).toBe(true);
    run(entry, residentModel(children), 50);
    expect(staleVisible.release).toHaveBeenCalledTimes(1);
    expect(staleVisible.object?.visible).toBe(false);
  });
});

describe('enforceResidentByteBudget — hidden-first ranking', () => {
  /**
   * A frustum that keeps only boxes reaching `x <= 0` in view: all six planes
   * share normal (-1, 0, 0) with constant 0, so `intersectsBox` is true iff the
   * box's `min.x <= 0`. Lets a test place one group in-frustum and another
   * off-screen without wiring a real projection matrix.
   */
  function halfSpaceFrustum(): THREE.Frustum {
    const f = new THREE.Frustum();
    for (const p of f.planes) p.set(new THREE.Vector3(-1, 0, 0), 0);
    return f;
  }

  /** A ready, previously-shown child under `layer` (same shape the panel toggle produces). */
  function childUnder(
    layer: VisibilityNode,
    visible: boolean,
    lastVisibleTick: number
  ): EvictableChild & { release: () => void } {
    return {
      ready: true,
      release: vi.fn() as unknown as () => void,
      lastVisibleTick,
      object: { visible, parent: layer },
    };
  }

  function residentModel(children: readonly (EvictableChild & { release: () => void })[]) {
    return () =>
      children.reduce((sum, c) => {
        const fired = (c.release as ReturnType<typeof vi.fn>).mock.calls.length > 0;
        return sum + (fired ? 0 : 100);
      }, 0);
  }

  it('evicts an in-frustum HIDDEN level before a VISIBLE layer’s off-screen level', () => {
    // Pre-fix, off-screen was the primary key: the visible layer's off-screen
    // fine level ranked FIRST and was reclaimed while the hidden (undrawable)
    // in-frustum level — which the camera cannot draw at all — was spared.
    // Post-fix, hidden is the primary key, so the undrawable data goes first.
    const hiddenLayer: VisibilityNode = { visible: false };
    const visibleLayer: VisibilityNode = { visible: true };

    // Hidden layer, IN frustum (min.x <= 0). Its displayed level is the HOTTEST
    // in the LRU (lastVisibleTick 5, re-stamped every frame while hidden), so
    // only the hidden-first primary key — not the tick tiebreak — evicts it.
    const hiddenNear = childUnder(hiddenLayer, true, 5);
    const hiddenEntry: EvictableEntry = {
      children: [hiddenNear],
      activeChildIndex: 0,
      displayedChildIndex: 0,
    };
    // Visible layer, OFF screen (min.x > 0): a cold non-displayed level (index 0,
    // own flag false ⇒ evictable) plus the on-screen displayed level (protected).
    const visibleCold = childUnder(visibleLayer, false, 1);
    const visibleDisplayed = childUnder(visibleLayer, true, 2);
    const visibleEntry: EvictableEntry = {
      children: [visibleCold, visibleDisplayed],
      activeChildIndex: 1,
      displayedChildIndex: 1,
    };

    const boxes = new Map<EvictableEntry, BoundingBox>([
      [hiddenEntry, { min: { x: -10, y: -1, z: -1 }, max: { x: -5, y: 1, z: 1 } }],
      [visibleEntry, { min: { x: 5, y: -1, z: -1 }, max: { x: 10, y: 1, z: 1 } }],
    ]);

    // Two evictable 100-byte levels; budget frees exactly one.
    const bytes = residentModel([hiddenNear, visibleCold]);
    enforceResidentByteBudget({
      entries: [hiddenEntry, visibleEntry],
      camera: new THREE.Camera(),
      frustum: halfSpaceFrustum(),
      getResidentByteBudget: () => 100,
      getResidentBytes: bytes,
      computeWorldBox: (e) => boxes.get(e) ?? null,
    });

    expect(hiddenNear.release).toHaveBeenCalledTimes(1); // undrawable → first out
    expect(visibleCold.release).not.toHaveBeenCalled(); // visible layer spared
    expect(visibleDisplayed.release).not.toHaveBeenCalled();
  });
});

describe('enforceResidentByteBudget — pure-retention early-outs', () => {
  // The three guards at the top of the function: with no budget, a
  // non-positive budget, or no residency measurement wired, the policy is
  // PURE RETENTION — nothing may be released even when candidates exist.
  function coldEntry(): {
    entry: EvictableEntry;
    release: ReturnType<typeof vi.fn>;
  } {
    const cold = child(false, 1);
    const displayed = child(true, 2);
    return {
      entry: {
        children: [cold, displayed],
        activeChildIndex: 1,
        displayedChildIndex: 1,
      },
      release: cold.release as ReturnType<typeof vi.fn>,
    };
  }

  it('releases nothing when no budget getter is wired (budget == null)', () => {
    const { entry, release } = coldEntry();
    enforceResidentByteBudget({
      entries: [entry],
      camera: new THREE.Camera(),
      frustum: new THREE.Frustum(),
      // no getResidentByteBudget at all
      getResidentBytes: () => 1e12,
      computeWorldBox: () => null,
    });
    expect(release).not.toHaveBeenCalled();
  });

  it('releases nothing for a non-positive budget (0 disables eviction)', () => {
    for (const budget of [0, -1]) {
      const { entry, release } = coldEntry();
      run(entry, () => 1e12, budget);
      expect(release).not.toHaveBeenCalled();
    }
  });

  it('releases nothing when no residency measurement is wired', () => {
    const { entry, release } = coldEntry();
    enforceResidentByteBudget({
      entries: [entry],
      camera: new THREE.Camera(),
      frustum: new THREE.Frustum(),
      getResidentByteBudget: () => 100,
      // no getResidentBytes
      computeWorldBox: () => null,
    });
    expect(release).not.toHaveBeenCalled();
  });

  it('releases nothing while under budget', () => {
    const { entry, release } = coldEntry();
    run(entry, () => 50, 100);
    expect(release).not.toHaveBeenCalled();
  });
});
