/**
 * Tests for the LODGroupRegistry selector logic.
 *
 * Covers two independent surfaces:
 *
 *   - The pure pick-with-hysteresis function (no THREE / camera): given
 *     thresholds + a current active index + a screen-space diagonal in
 *     pixels, the right child is selected. Hysteresis suppresses
 *     downgrade flicker but doesn't impede upgrades.
 *
 *   - The full registry: register/unregister, lock override, visibility
 *     swap on auto-mode evaluation, and the "keep old until new is
 *     ready" contract.
 *
 * The bbox-projection helper (``projectBoxDiagonalPx``) is exercised
 * indirectly through end-to-end registry evaluation against a
 * deterministic mock camera that maps world coords identity to NDC.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  LODGroupRegistry,
  pickChildWithHysteresis,
  projectBoxDiagonalPx,
  type LODGroupChild,
  type LODGroupEntry,
} from '../../../scene/lod-group-registry';
import type { BoundingBox } from '../../../scene/scene-manager/clipping/bounds-math';

// ────────────────────────────────────────────────────────────────────────
// pickChildWithHysteresis — pure selector math
// ────────────────────────────────────────────────────────────────────────

describe('pickChildWithHysteresis', () => {
  const thresholds = [0, 100, 500];

  it('picks the coarsest child below the first threshold', () => {
    expect(pickChildWithHysteresis(thresholds, 0, 0)).toBe(0);
    expect(pickChildWithHysteresis(thresholds, 0, 50)).toBe(0);
  });

  it('picks finer child once the diagonal reaches its threshold (upgrade)', () => {
    // From child 0 → child 1 when diagonalPx ≥ 100 (literal threshold).
    expect(pickChildWithHysteresis(thresholds, 0, 100)).toBe(1);
    expect(pickChildWithHysteresis(thresholds, 1, 500)).toBe(2);
  });

  it('downgrade requires falling below threshold * (1 - 0.1)', () => {
    // From child 1 with threshold 100: downgrade only when diagonalPx < 90.
    // At 95 (still 10% within), we stay on 1 even though natural is 0.
    expect(pickChildWithHysteresis(thresholds, 1, 95)).toBe(1);
    expect(pickChildWithHysteresis(thresholds, 1, 89)).toBe(0);
  });

  it('returns -1 for an empty thresholds list', () => {
    expect(pickChildWithHysteresis([], 0, 100)).toBe(-1);
  });

  it('handles a current index that no longer satisfies its threshold', () => {
    // current=2 (threshold 500) but diagonal is only 50 → expected
    // downgrade to 0 (well below 500 * 0.9).
    expect(pickChildWithHysteresis(thresholds, 2, 50)).toBe(0);
  });
});

// ────────────────────────────────────────────────────────────────────────
// projectBoxDiagonalPx — sanity-check the screen-space conversion
// ────────────────────────────────────────────────────────────────────────

describe('projectBoxDiagonalPx', () => {
  /**
   * Build a minimal mock camera that behaves as an identity-NDC
   * transform for world coordinates inside [-1, 1]. Vector3.project()
   * applies `matrixWorldInverse * projectionMatrix` to the vector,
   * so we just leave both as identity → NDC == world.
   */
  function identityCamera(): THREE.Camera {
    const cam = new THREE.Camera();
    cam.matrixWorldInverse.identity();
    cam.projectionMatrix.identity();
    cam.matrixWorld.identity();
    return cam;
  }

  it('returns 0 for a degenerate box', () => {
    const box: BoundingBox = {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
    };
    expect(
      projectBoxDiagonalPx(box, identityCamera(), { width: 800, height: 600 })
    ).toBeCloseTo(0, 5);
  });

  it('returns the screen-space diagonal for a unit NDC box (identity camera)', () => {
    // Box that spans the full NDC cube [-1, 1] on x/y. Projects to the
    // full viewport. Diagonal of an 800x600 viewport is √(800² + 600²)
    // = 1000.
    const box: BoundingBox = {
      min: { x: -1, y: -1, z: 0 },
      max: { x: 1, y: 1, z: 0 },
    };
    const diagonal = projectBoxDiagonalPx(box, identityCamera(), {
      width: 800,
      height: 600,
    });
    expect(diagonal).toBeCloseTo(1000, 1);
  });

  it('scales with viewport size', () => {
    const box: BoundingBox = {
      min: { x: -1, y: -1, z: 0 },
      max: { x: 1, y: 1, z: 0 },
    };
    const small = projectBoxDiagonalPx(box, identityCamera(), {
      width: 100,
      height: 100,
    });
    const large = projectBoxDiagonalPx(box, identityCamera(), {
      width: 200,
      height: 200,
    });
    // Doubling both width and height should double the diagonal.
    expect(large).toBeCloseTo(small * 2, 5);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Registry — registration, lock override, swap behaviour
// ────────────────────────────────────────────────────────────────────────

function makeChild(minPixelSize: number, ready: boolean = true): LODGroupChild {
  return {
    object: new THREE.Group(),
    minPixelSize,
    positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
    ready,
  };
}

function makeEntry(
  children: LODGroupChild[],
  activeIndex: number = 0,
  path: string = '/lod'
): LODGroupEntry {
  return {
    path,
    groupObject: new THREE.Group(),
    children,
    selectorMode: 'auto',
    defaultLevel: activeIndex,
    activeChildIndex: activeIndex,
  };
}

function makeRegistry(displayDims: readonly number[] = [0, 1, 2]) {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  return new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => displayDims,
  });
}

describe('LODGroupRegistry — registration', () => {
  it('hides all children except the active one on register', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100), makeChild(500)];
    reg.register(makeEntry(children, 1));
    expect(children[0].object.visible).toBe(false);
    expect(children[1].object.visible).toBe(true);
    expect(children[2].object.visible).toBe(false);
  });

  it('unregister removes the entry without disturbing visibility', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100)];
    reg.register(makeEntry(children, 0, '/g'));
    expect(reg.size()).toBe(1);
    reg.unregister('/g');
    expect(reg.size()).toBe(0);
  });

  it('list returns all registered entries', () => {
    const reg = makeRegistry();
    reg.register(makeEntry([makeChild(0), makeChild(100)], 0, '/g0'));
    reg.register(makeEntry([makeChild(0), makeChild(100)], 0, '/g1'));
    expect(reg.list().map((e) => e.path).sort()).toEqual(['/g0', '/g1']);
  });
});

describe('LODGroupRegistry — selector mode', () => {
  it('setSelectorMode rejects out-of-range lockLevel', () => {
    const reg = makeRegistry();
    reg.register(makeEntry([makeChild(0), makeChild(100)], 0, '/g'));
    expect(() => reg.setSelectorMode('/g', { lockLevel: 5 })).toThrow(/lockLevel/);
  });

  it('lock override swaps to the locked child on the next evaluation', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100), makeChild(500)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 2 });
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(false);
    expect(children[2].object.visible).toBe(true);
  });

  it('lock override is no-op for unknown paths', () => {
    const reg = makeRegistry();
    // Should not throw.
    reg.setSelectorMode('/missing', { lockLevel: 0 });
  });
});

describe('LODGroupRegistry — readiness gate', () => {
  it('does not swap to a child whose ready flag is false', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100, false /* unready */)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    // child 1 is unready → active stays on 0.
    expect(children[0].object.visible).toBe(true);
    expect(children[1].object.visible).toBe(false);
  });

  it('swaps once markChildReady flips the flag', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100, false)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(false);
    reg.markChildReady('/g', 1);
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
  });
});

describe('LODGroupRegistry — auto evaluation', () => {
  it('picks the finest child whose threshold is satisfied by the projected diagonal', () => {
    // bbox spans the full NDC cube → identity camera projects to a
    // 1000 px diagonal on an 800x600 viewport. With thresholds
    // [0, 100, 500], the finest applicable is child 2.
    const reg = makeRegistry();
    const children = [
      { ...makeChild(0), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
      { ...makeChild(100), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
      { ...makeChild(500), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
    ];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[2].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
    expect(children[1].object.visible).toBe(false);
  });

  it('skips evaluation when the viewport has zero size', () => {
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 0, height: 0 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const children = [makeChild(0), makeChild(100)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    // Active stayed on 0 because evaluation early-returned.
    expect(children[0].object.visible).toBe(true);
  });
});
