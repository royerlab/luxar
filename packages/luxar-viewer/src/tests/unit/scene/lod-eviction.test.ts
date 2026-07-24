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
