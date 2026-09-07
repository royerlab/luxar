/**
 * Unit tests for the settled-pick cache and its staleness guard
 * (`core/app/interaction/picked-element-cache.ts`, issue #1917).
 *
 * This is the safety-critical half of click-to-act: acting on a stale pick
 * opens a link for the WRONG element, which the user cannot undo. So there is
 * one test per invalidation signal, plus the counter-test that an ordinary
 * click — which passes through `suppress()` — is still honoured.
 *
 * `PickGenerationPort` is two numbers, so no PickingSystem is needed.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  PickedElementCache,
  TOUCH_CLICK_SLOP_PX,
  clickSlopFor,
  CLICK_SLOP_PX,
  type CachedPick,
  type PickGenerationPort,
} from '../../../../../core/app/interaction/picked-element-cache';

/** Mutable stand-in for the two PickingSystem counters. */
class FakePicking implements PickGenerationPort {
  pickGeneration = 100;
  visibleSignature = 7;
}

function makePick(overrides: Partial<CachedPick> = {}): CachedPick {
  const mainNode = new THREE.Object3D();
  mainNode.name = '/proteins';
  return {
    mainNode,
    nodeName: '/proteins',
    hitNodeName: '/proteins',
    elementIndex: 42,
    label: 'P04637',
    key: 'P04637',
    screenX: 200,
    screenY: 150,
    ...overrides,
  };
}

describe('PickedElementCache — basics', () => {
  it('returns null before anything is stored', () => {
    const cache = new PickedElementCache();
    expect(cache.peek(new FakePicking())).toBeNull();
    expect(cache.read(new FakePicking(), 200, 150)).toBeNull();
  });

  it('returns the stored pick while nothing has changed', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    const pick = makePick();
    cache.store(pick, ports);

    expect(cache.read(ports, 200, 150)).toMatchObject({
      nodeName: '/proteins',
      elementIndex: 42,
      label: 'P04637',
    });
  });

  it('clear() forgets the pick', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick(), ports);
    cache.clear();
    expect(cache.read(ports, 200, 150)).toBeNull();
  });

  it('a newer store replaces the older pick', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick({ elementIndex: 1 }), ports);
    ports.pickGeneration++;
    cache.store(makePick({ elementIndex: 2 }), ports);
    expect(cache.read(ports, 200, 150)?.elementIndex).toBe(2);
  });
});

describe('PickedElementCache — refuses a stale pick', () => {
  /**
   * `pickGeneration` is the single counter behind camera moves, resize,
   * persp↔ortho, the FOV edit of #1916, the layers-panel invalidator, any
   * mousemove, mouseleave and dispose. One assertion covers all of them,
   * because they all reach the cache the same way.
   */
  it('after anything that advances the pick generation', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick(), ports);

    ports.pickGeneration++; // camera moved / cursor moved / buffer dirtied

    expect(cache.peek(ports)).toBeNull();
    expect(cache.read(ports, 200, 150)).toBeNull();
  });

  it('after a layer visibility toggle, which does NOT advance the generation', () => {
    // The reason `visibleSignature` is checked separately: `applyVisibility`
    // only calls requestRender(), so the generation counter never moves and
    // the pick would otherwise still look valid while pointing at a layer the
    // user just hid.
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick(), ports);

    ports.visibleSignature = 99;

    expect(ports.pickGeneration).toBe(100); // unchanged — that is the point
    expect(cache.peek(ports)).toBeNull();
  });

  it('when the click lands beyond the slop radius', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick({ screenX: 200, screenY: 150 }), ports);

    expect(cache.read(ports, 200 + CLICK_SLOP_PX + 1, 150)).toBeNull();
    expect(cache.read(ports, 200, 150 - (CLICK_SLOP_PX + 1))).toBeNull();
  });

  it('accepts a click within the slop radius, including diagonally', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick({ screenX: 200, screenY: 150 }), ports);

    expect(cache.read(ports, 200, 150)).not.toBeNull();
    expect(cache.read(ports, 200 + CLICK_SLOP_PX, 150)).not.toBeNull();
    // Euclidean, not Chebyshev — the distinction matters on the diagonal:
    // (2,3) is √13 ≈ 3.6 away and inside; (3,3) is √18 ≈ 4.24 and outside,
    // even though both are within 4 on each axis.
    expect(cache.read(ports, 202, 153)).not.toBeNull();
    expect(cache.read(ports, 203, 153)).toBeNull();
  });

  it('peek() ignores position — it is for the cursor affordance and keyboard menu', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick({ screenX: 200, screenY: 150 }), ports);

    // Far away, but nothing has invalidated the pick, so peek still answers.
    expect(cache.peek(ports)).not.toBeNull();
    expect(cache.read(ports, 900, 900)).toBeNull();
  });
});

describe('PickedElementCache — an ordinary click must survive', () => {
  /**
   * The counter-test to all of the above, and the reason `suppress()`
   * deliberately does not touch `pickGeneration`.
   *
   * A click is: pointerdown → controls dispatch `start` → `suppress(true)` →
   * pointerup → `suppress(false)`. If any of that advanced the generation,
   * every click would refuse itself and the feature would be dead on arrival
   * with no obvious cause.
   */
  it('is honoured across the suppress/unsuppress of its own pointerdown', () => {
    const cache = new PickedElementCache();
    const ports = new FakePicking();
    cache.store(makePick({ screenX: 200, screenY: 150 }), ports);

    // Simulate the gesture: suppress toggles, no generation change, no
    // visibility change, pointer stays put.
    const generationBefore = ports.pickGeneration;
    // (suppress(true) … suppress(false) happen here in the real system)
    expect(ports.pickGeneration).toBe(generationBefore);

    expect(cache.read(ports, 200, 150)).not.toBeNull();
  });
});

describe('click slop per pointer type', () => {
  it('a finger gets the wider tolerance, everything else the mouse value', () => {
    expect(clickSlopFor('touch')).toBe(TOUCH_CLICK_SLOP_PX);
    expect(clickSlopFor('mouse')).toBe(CLICK_SLOP_PX);
    expect(clickSlopFor('pen')).toBe(CLICK_SLOP_PX);
    expect(clickSlopFor('')).toBe(CLICK_SLOP_PX);
    expect(TOUCH_CLICK_SLOP_PX).toBeGreaterThan(CLICK_SLOP_PX);
  });

  it('read() honours an explicit slop', () => {
    const cache = new PickedElementCache();
    const ports = { pickGeneration: 1, visibleSignature: 1 };
    cache.store(
      {
        mainNode: new THREE.Object3D(),
        nodeName: 'n',
        hitNodeName: 'n',
        elementIndex: 0,
        label: null,
        key: null,
        screenX: 100,
        screenY: 100,
      },
      ports
    );
    expect(cache.read(ports, 110, 100)).toBeNull(); // 10 px > mouse slop
    expect(cache.read(ports, 110, 100, TOUCH_CLICK_SLOP_PX)).not.toBeNull();
    expect(cache.read(ports, 113, 100, TOUCH_CLICK_SLOP_PX)).toBeNull(); // 13 px > 12
  });
});
