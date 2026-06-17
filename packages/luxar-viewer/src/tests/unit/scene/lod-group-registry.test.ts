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
 *     swap on auto-mode evaluation, and the clamp-on-out-of-range lock
 *     behaviour.
 *
 * The bbox-projection helper (``projectBoxDiagonalPx``) is exercised
 * indirectly through end-to-end registry evaluation against a
 * deterministic mock camera that maps world coords identity to NDC.
 */

import { describe, expect, it, vi } from 'vitest';
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

  it('uses a gap-relative downgrade band for tightly-spaced levels', () => {
    // Levels 10 and 11 are within 10% of each other (e.g. from the
    // derive_min_pixel_sizes ×1.1 nudge). The spacing-aware band is
    // ratio * gap = 0.1 * (11 - 10) = 0.1, so the deadband is [10.9, 11):
    // level 1 still renders on the way down instead of being skipped.
    const tight = [0, 10, 11];
    expect(pickChildWithHysteresis(tight, 2, 10.95)).toBe(2); // deadband → stay
    expect(pickChildWithHysteresis(tight, 2, 10.5)).toBe(1); // one clean level down
    expect(pickChildWithHysteresis(tight, 2, 9.9)).toBe(0); // into level 0's range
  });

  it('reduces to the threshold-fraction band for the bottom level', () => {
    // prev threshold is 0, so gap == currentThreshold and the band equals
    // the original currentThreshold * (1 - ratio) behaviour.
    expect(pickChildWithHysteresis(thresholds, 1, 91)).toBe(1); // 91 ≥ 90 → stay
    expect(pickChildWithHysteresis(thresholds, 1, 89)).toBe(0); // 89 < 90 → down
  });

  it('handles a current index that no longer satisfies its threshold', () => {
    // current=2 (threshold 500) but diagonal is only 50 → expected
    // downgrade to 0 (well below 500 * 0.9).
    expect(pickChildWithHysteresis(thresholds, 2, 50)).toBe(0);
  });

  // C2: pin the downgrade band edge EXACTLY. margin = 0.1*(100-0) = 10, so the
  // boundary is 90: `diagonalPx < 90` downgrades, `>= 90` stays. A mutant that
  // changed `<` to `<=` (or the ratio) would shift this edge.
  it('treats the downgrade boundary as exclusive (exactly 90 stays on level 1)', () => {
    expect(pickChildWithHysteresis(thresholds, 1, 90)).toBe(1); // 90 not < 90 → stay
    expect(pickChildWithHysteresis(thresholds, 1, 89.999)).toBe(0); // just below → down
  });

  // M1: pin the upgrade boundary edge. The natural pick uses `threshold <=
  // diagonalPx`, so exactly 100 upgrades to level 1 and 99.999 stays at 0.
  it('treats the upgrade threshold as inclusive (exactly 100 reaches level 1)', () => {
    expect(pickChildWithHysteresis(thresholds, 0, 100)).toBe(1);
    expect(pickChildWithHysteresis(thresholds, 0, 99.999)).toBe(0);
  });

  // projectBoxDiagonalPx returns +Infinity when the camera is inside/straddling
  // the bbox. That must select the finest child from any current index, and
  // must never produce NaN through the downgrade arithmetic.
  it('saturates to the finest child for an infinite diagonal (camera-inside)', () => {
    expect(pickChildWithHysteresis(thresholds, 0, Number.POSITIVE_INFINITY)).toBe(2);
    expect(pickChildWithHysteresis(thresholds, 1, Number.POSITIVE_INFINITY)).toBe(2);
    expect(pickChildWithHysteresis(thresholds, 2, Number.POSITIVE_INFINITY)).toBe(2);
    expect(Number.isNaN(pickChildWithHysteresis(thresholds, 2, Number.POSITIVE_INFINITY))).toBe(
      false
    );
  });

  // A multi-level downgrade snaps straight to the natural level: the metric fell
  // well past the adjacent band, so the single-gap hysteresis cannot suppress it.
  it('snaps multiple levels down at once when the metric drops far', () => {
    // current=2 (threshold 500), metric 95 → natural is 0 (95 < 100, so only
    // threshold 0 qualifies); well under 500 - 0.1*(500-100)=460 → straight to 0.
    expect(pickChildWithHysteresis(thresholds, 2, 95)).toBe(0);
    // current=2, metric 150 → natural=1 (100≤150<500); below 500-0.1*(500-100)=460 → down to 1.
    expect(pickChildWithHysteresis(thresholds, 2, 150)).toBe(1);
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

  /** Perspective camera at the origin looking down −Z (near 0.1, far 1000). */
  function perspectiveAtOrigin(): THREE.PerspectiveCamera {
    const cam = new THREE.PerspectiveCamera(60, 800 / 600, 0.1, 1000);
    cam.position.set(0, 0, 0);
    cam.lookAt(0, 0, -1);
    cam.updateMatrixWorld(true);
    cam.updateProjectionMatrix();
    return cam;
  }

  /** Orthographic camera at the origin looking down −Z. Keeps w == 1. */
  function orthoAtOrigin(): THREE.OrthographicCamera {
    const cam = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
    cam.position.set(0, 0, 0);
    cam.lookAt(0, 0, -1);
    cam.updateMatrixWorld(true);
    cam.updateProjectionMatrix();
    return cam;
  }

  it('returns 0 for a degenerate box', () => {
    const box: BoundingBox = {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
    };
    expect(projectBoxDiagonalPx(box, identityCamera(), { width: 800, height: 600 })).toBeCloseTo(
      0,
      5
    );
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

  // M2: symmetric scaling alone can't distinguish hypot(w,h) from a buggy
  // w+h (both double when w,h double). Use an ASYMMETRIC viewport where the
  // two formulas diverge: for a full-NDC box, hypot(800,600)=1000 but
  // 800+600=1400. (The unit-box test above already pins 1000; this makes the
  // diagonal-not-sum contract explicit alongside the scaling test.)
  it('uses the Euclidean diagonal, not the width+height sum (asymmetric viewport)', () => {
    const box: BoundingBox = { min: { x: -1, y: -1, z: 0 }, max: { x: 1, y: 1, z: 0 } };
    const diagonal = projectBoxDiagonalPx(box, identityCamera(), { width: 800, height: 600 });
    expect(diagonal).toBeCloseTo(1000, 1); // hypot, not 1400 (sum)
    expect(diagonal).not.toBeCloseTo(1400, 0);
  });

  // Near-plane saturation: the camera being inside or straddling the box must
  // yield +Infinity (→ finest level), not a collapsed/garbage diagonal.
  it('returns the correct finite diagonal when the box is fully in front (perspective)', () => {
    // Small box well in front of the camera (negative z, all corners have w>0).
    const box: BoundingBox = { min: { x: -1, y: -1, z: -11 }, max: { x: 1, y: 1, z: -9 } };
    const diagonal = projectBoxDiagonalPx(box, perspectiveAtOrigin(), { width: 800, height: 600 });
    expect(Number.isFinite(diagonal)).toBe(true);
    // Analytic value pins the perspective divide + NDC→px scaling (not just
    // finite>0, which a divide-omitting mutant would also pass). fovY=60 →
    // f=1/tan30=1.73205; m[0]=f/aspect=1.29904, w=-z. Nearest corners (z=-9)
    // dominate the AABB: Δndc_x=2·1.29904/9, Δndc_y=2·1.73205/9 → widthPx=
    // heightPx=115.47 → hypot=163.30.
    expect(diagonal).toBeCloseTo(163.3, 1);
  });

  it('saturates to +Infinity when the camera is inside the box (perspective)', () => {
    // Camera at origin sits inside this box → some corners are behind it (w<=0).
    const box: BoundingBox = { min: { x: -5, y: -5, z: -5 }, max: { x: 5, y: 5, z: 5 } };
    expect(projectBoxDiagonalPx(box, perspectiveAtOrigin(), { width: 800, height: 600 })).toBe(
      Number.POSITIVE_INFINITY
    );
  });

  it('saturates to +Infinity when the box straddles the near plane (perspective)', () => {
    // Box spans z=-2..+2; the +z corners are behind a camera looking down −Z.
    const box: BoundingBox = { min: { x: -1, y: -1, z: -2 }, max: { x: 1, y: 1, z: 2 } };
    expect(projectBoxDiagonalPx(box, perspectiveAtOrigin(), { width: 800, height: 600 })).toBe(
      Number.POSITIVE_INFINITY
    );
  });

  it('never saturates for an orthographic camera (w stays 1)', () => {
    // Camera "inside" the box, but ortho has no perspective divide → w==1, so
    // the diagonal stays finite rather than tripping the near-plane guard.
    const box: BoundingBox = { min: { x: -5, y: -5, z: -5 }, max: { x: 5, y: 5, z: 5 } };
    const diagonal = projectBoxDiagonalPx(box, orthoAtOrigin(), { width: 800, height: 600 });
    expect(Number.isFinite(diagonal)).toBe(true);
    // Pin the ortho projection math: left/right=±10 maps x=±5 → NDC ±0.5;
    // top/bottom=±10 maps y=±5 → NDC ±0.5. widthPx=1·0.5·800=400,
    // heightPx=1·0.5·600=300 → hypot=500. Confirms w stays 1 (no saturation)
    // AND the manual column-major NDC computation is correct for ortho.
    expect(diagonal).toBeCloseTo(500, 1);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Registry — registration, lock override, swap behaviour
// ────────────────────────────────────────────────────────────────────────

function makeChild(minPixelSize: number): LODGroupChild {
  return {
    object: new THREE.Group(),
    minPixelSize,
    positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
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

/**
 * A lazily-loaded child: geometry not committed (``ready: false``) and a
 * deferred loader thunk. Mirrors what ``loadLodGroupNode`` builds for
 * non-default substitutive levels.
 */
function makeLazyChild(minPixelSize: number, ensureLoaded: () => void): LODGroupChild {
  return {
    object: new THREE.Group(),
    minPixelSize,
    positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
    ready: false,
    ensureLoaded,
  };
}

function makeRegistry(
  displayDims: readonly number[] = [0, 1, 2],
  residentByteBudget?: number,
  getResidentBytes?: () => number
) {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  return new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => displayDims,
    ...(residentByteBudget != null
      ? {
          getResidentByteBudget: () => residentByteBudget,
          // The registry no longer tracks per-level bytes; it queries the
          // pool's resident total. Default model: 0 (never over budget)
          // unless the test supplies one.
          getResidentBytes: getResidentBytes ?? (() => 0),
        }
      : {}),
  });
}

/**
 * Model the pool's resident-byte total for eviction tests: `perLevel`
 * bytes for every ready child whose `release` spy has not yet fired
 * (release moves the level's geometry out of residency). Mirrors the real
 * pool, whose `getResidentBytes` drops as the registry demotes cold levels.
 */
function residentModel(children: readonly LODGroupChild[], perLevel = 100): () => number {
  return () =>
    children.reduce((sum, c) => {
      if (c.ready !== true) return sum;
      const releaseFired =
        ((c.release as unknown as { mock?: { calls: unknown[] } })?.mock?.calls.length ?? 0) > 0;
      return sum + (releaseFired ? 0 : perLevel);
    }, 0);
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
    expect(
      reg
        .list()
        .map((e) => e.path)
        .sort()
    ).toEqual(['/g0', '/g1']);
  });
});

describe('LODGroupRegistry — selector mode', () => {
  it('setSelectorMode clamps out-of-range lockLevel and warns', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100)];
    reg.register(makeEntry(children, 0, '/g'));
    // Out-of-range lockLevel must not throw — UI state can drift from
    // registry state if it does. The registry clamps into [0, n-1]
    // and re-emits a warning so the bug surfaces in the log.
    reg.setSelectorMode('/g', { lockLevel: 5 });
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(false);
    expect(children[1].object.visible).toBe(true);
  });

  it('setSelectorMode clamps negative lockLevel to 0', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100)];
    reg.register(makeEntry(children, 1, '/g'));
    reg.setSelectorMode('/g', { lockLevel: -3 });
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true);
    expect(children[1].object.visible).toBe(false);
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

  it('evaluatePerFrame returns true on a level swap, false on a no-op frame', () => {
    // The per-frame callback uses this signal to refresh the monitor's
    // visible-element tally only when the rendered level actually changes
    // — otherwise the count would stay pinned to the default level.
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100), makeChild(500)];
    reg.register(makeEntry(children, 0, '/g'));

    reg.setSelectorMode('/g', { lockLevel: 2 });
    expect(reg.evaluatePerFrame()).toBe(true); // 0 → 2 swap
    expect(reg.evaluatePerFrame()).toBe(false); // already at 2, no change
  });

  it('evaluatePerFrame returns false when there are no entries', () => {
    expect(makeRegistry().evaluatePerFrame()).toBe(false);
  });

  it('swaps atomically when the locked level changes', () => {
    // Atomic swap supersedes the old per-child readiness gate: each
    // child is already visible-false until the registry picks it
    // (loadLodGroupNode ensures that on initial load via the
    // sequential-await + visible=false-after-attach pattern).
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(100)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
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

describe('LODGroupRegistry — lazy children', () => {
  it('register leaves a not-ready non-active child hidden', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(100, () => {})];
    reg.register(makeEntry(children, 0, '/g'));
    expect(children[0].object.visible).toBe(true); // active + ready
    expect(children[1].object.visible).toBe(false); // not ready → hidden
  });

  it('does not swap to a not-ready child; fires ensureLoaded exactly once across frames', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(100, ensureLoaded)];
    reg.register(makeEntry(children, 0, '/g'));
    // Drive desired→1 via lock so the test doesn't depend on projection math.
    reg.setSelectorMode('/g', { lockLevel: 1 });

    // Target not ready → no swap, current level stays visible.
    expect(reg.evaluatePerFrame()).toBe(false);
    expect(children[0].object.visible).toBe(true);
    expect(children[1].object.visible).toBe(false);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);

    // The registry set loading=true; subsequent frames must not refire
    // (the real thunk clears loading; the mock leaves it set).
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
  });

  it('swaps to the lazy child on the next frame after it becomes ready', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(100, () => {})];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame(); // fires ensureLoaded, no swap yet

    // Simulate the thunk completing.
    children[1].ready = true;
    children[1].loading = false;

    expect(reg.evaluatePerFrame()).toBe(true); // now ready → swap
    expect(children[0].object.visible).toBe(false);
    expect(children[1].object.visible).toBe(true);
  });

  it('does not refire ensureLoaded for a child that previously failed', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(100, ensureLoaded)];
    children[1].failed = true; // a prior load attempt failed
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  /** A loaded lazy level with a release spy + a prior-visible tick. */
  function readyLazy(
    minPixelSize: number,
    opts: { release?: ReturnType<typeof vi.fn>; tick?: number } = {}
  ): LODGroupChild {
    return {
      ...makeLazyChild(minPixelSize, () => {}),
      ready: true,
      release: (opts.release ?? vi.fn()) as () => void,
      lastVisibleTick: opts.tick,
    };
  }

  it('does not release on swap (pure retention when no byte budget is wired)', () => {
    const reg = makeRegistry(); // no budget → no eviction
    const rel0 = vi.fn();
    const children = [readyLazy(0, { release: rel0 }), readyLazy(100), readyLazy(500)];
    // Active on a far level; swap all the way to level 0.
    reg.register(makeEntry(children, 2, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 });

    expect(reg.evaluatePerFrame()).toBe(true); // swap 2 → 0
    // The level we left (2) and its release thunks are NOT called — retention
    // keeps every loaded level resident so swapping back is a free toggle.
    expect(children[2].release).not.toHaveBeenCalled();
    expect(rel0).not.toHaveBeenCalled();
    expect(children[0].object.visible).toBe(true);
  });

  it('evicts the coldest hidden level when over the byte budget', () => {
    // Budget 250 bytes; three 100-byte levels → 300 resident → one must go.
    const rel0 = vi.fn();
    const rel1 = vi.fn();
    const children = [
      readyLazy(0, { release: rel0, tick: 1 }), // coldest (lowest tick)
      readyLazy(100, { release: rel1, tick: 5 }),
      readyLazy(500), // will be the locked/active level
    ];
    const reg = makeRegistry([0, 1, 2], 250, residentModel(children));
    reg.register(makeEntry(children, 2, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 2 });

    reg.evaluatePerFrame();
    // Active (2) exempt; among hidden {0,1}, level 0 is coldest → demoting it
    // drops resident 300 → 200 ≤ 250, so the loop stops. Level 1 retained.
    expect(rel0).toHaveBeenCalledTimes(1);
    expect(rel1).not.toHaveBeenCalled();
  });

  it('never evicts the currently-visible level even when over budget', () => {
    // Tiny budget so eviction definitely triggers; only the active level
    // is a (would-be) candidate — but the active is always exempt.
    const relActive = vi.fn();
    const children = [
      readyLazy(0), // not released, but no tick → not a candidate
      readyLazy(100, { release: relActive, tick: 3 }),
    ];
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children));
    reg.register(makeEntry(children, 1, '/g')); // active = level 1
    reg.setSelectorMode('/g', { lockLevel: 1 });

    reg.evaluatePerFrame();
    expect(relActive).not.toHaveBeenCalled(); // visible level never freed
    expect(children[1].object.visible).toBe(true);
  });

  it('never evicts a still-loading (not-ready) level — preserves the 1-frame swap gap', () => {
    // The genuine 1-frame gap: a level whose ensureLoaded is in flight
    // (ready still false) must never be evicted. The ``isReady`` filter
    // excludes it regardless of budget pressure.
    const relLoading = vi.fn();
    const loading = makeLazyChild(100, () => {});
    loading.loading = true; // in flight, ready still false
    loading.release = relLoading;
    const children = [readyLazy(0, { tick: 9 }), loading];
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children));
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 });

    reg.evaluatePerFrame();
    expect(relLoading).not.toHaveBeenCalled();
  });

  it('evicts a loaded level that became ready but was never shown (Fix 2 — no VRAM leak)', () => {
    // A lazy level that finished loading but was never swapped-to (camera
    // moved away mid-load) used to keep lastVisibleTick == null forever and
    // was permanently exempt from eviction — a slow VRAM leak. It now gets
    // stamped "became ready" and ages into the eviction LRU like any other
    // cold level.
    const relStranded = vi.fn();
    const children = [
      readyLazy(0, { tick: 9 }), // active
      readyLazy(100, { release: relStranded }), // loaded, never shown
    ];
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children));
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 });

    reg.evaluatePerFrame();
    expect(relStranded).toHaveBeenCalledTimes(1);
  });

  it('self-heals a not-ready active child: kicks its load, then shows it once ready (Fix 1)', () => {
    // The blank-render bug: a fallback can pin a not-ready lazy level as the
    // active child (e.g. the eager default failed to attach). With
    // desired === active the swap block never runs, so without self-heal the
    // group would render blank forever. The registry must kick the active
    // child's load and reveal it the frame it becomes ready.
    const ensureLoaded = vi.fn();
    const lazyActive = makeLazyChild(0, ensureLoaded); // not ready
    const reg = makeRegistry();
    reg.register(makeEntry([lazyActive], 0, '/g')); // single child, active = it
    expect(lazyActive.object.visible).toBe(false); // hidden: not ready

    reg.evaluatePerFrame(); // desired === active (0) — self-heal kicks load
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    expect(lazyActive.object.visible).toBe(false); // still loading

    // Thunk completes.
    lazyActive.ready = true;
    lazyActive.loading = false;
    expect(reg.evaluatePerFrame()).toBe(true); // becomes visible → changed
    expect(lazyActive.object.visible).toBe(true);
  });

  it('retries a transiently-failed level after the cooldown elapses (Fix 3)', () => {
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(100, ensureLoaded)];
    children[1].failed = true; // a prior reload attempt failed
    const reg = makeRegistry();
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 }); // desired = the failed level

    // Within the cooldown window the failed level is not retried.
    reg.evaluatePerFrame(); // stamps failedTick
    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();

    // Advance past FAILED_RETRY_FRAMES (120) of frames; the cooldown clears
    // and the next selection retries the load exactly once.
    for (let i = 0; i < 121; i++) reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    expect(children[1].failed).not.toBe(true);
  });

  it('clear() resets the monotonic tick', () => {
    const reg = makeRegistry();
    reg.register(makeEntry([makeChild(0), makeChild(100)], 0, '/g'));
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    reg.clear();
    // After clear, a freshly-registered entry's first stamp should start
    // from a reset clock. We can't read tick directly, but a re-registered
    // child must not inherit a stale high tick — exercise the path.
    const children = [makeChild(0), makeChild(100)];
    reg.register(makeEntry(children, 0, '/g2'));
    reg.evaluatePerFrame();
    expect(children[0].lastVisibleTick).toBe(1); // tick reset → first frame == 1
  });
});

// ────────────────────────────────────────────────────────────────────────
// Frustum-aware selection (off-screen gate) + frustum/distance-aware eviction
//
// These need a *real* PerspectiveCamera (the identity mock above can't tell
// on- from off-screen by position): the camera sits at the origin looking
// down −Z, so a group placed in front (negative z, near the axis) is inside
// the frustum and a group offset far along +x is outside it.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — frustum-aware selection & eviction', () => {
  /** Camera at the origin looking down −Z (matrixWorldInverse refreshed). */
  function cameraLookingDownNegZ(): THREE.PerspectiveCamera {
    const cam = new THREE.PerspectiveCamera(60, 800 / 600, 0.1, 1000);
    cam.position.set(0, 0, 0);
    cam.lookAt(0, 0, -1);
    cam.updateMatrixWorld(true); // THREE.Camera also refreshes matrixWorldInverse
    cam.updateProjectionMatrix();
    return cam;
  }

  function registryWith(camera: THREE.Camera, budget?: number, getResidentBytes?: () => number) {
    return new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      ...(budget != null
        ? {
            getResidentByteBudget: () => budget,
            getResidentBytes: getResidentBytes ?? (() => 0),
          }
        : {}),
    });
  }

  /** A loaded (ready) level with a release spy + an optional prior-visible tick. */
  function readyLevel(
    minPixelSize: number,
    opts: { release?: ReturnType<typeof vi.fn>; tick?: number } = {}
  ): LODGroupChild {
    return {
      ...makeLazyChild(minPixelSize, () => {}),
      ready: true,
      release: (opts.release ?? vi.fn()) as () => void,
      lastVisibleTick: opts.tick,
    };
  }

  /** Place an entry's group in world space, then refresh its matrix. */
  function placeAt(entry: LODGroupEntry, x: number, y: number, z: number): LODGroupEntry {
    entry.groupObject.position.set(x, y, z);
    entry.groupObject.updateMatrixWorld(true);
    return entry;
  }

  it('auto: holds an off-screen group at the coarsest ready level (drops a fine level)', () => {
    const reg = registryWith(cameraLookingDownNegZ());
    const children = [makeChild(0), makeChild(100)]; // both ready (non-lazy)
    // Active on the *fine* level, but the group is offset far along +x → outside
    // the frustum. The gate must drop it back to the coarsest ready level.
    const entry = placeAt(makeEntry(children, 1, '/g'), 100, 0, -5);
    reg.register(entry);
    expect(children[1].object.visible).toBe(true); // fine active on register

    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true); // dropped to coarsest
    expect(children[1].object.visible).toBe(false);
    expect(entry.offScreen).toBe(true); // surfaced in the layers-panel readout
  });

  it('auto: does not kick a lazy load for an off-screen group', () => {
    const reg = registryWith(cameraLookingDownNegZ());
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(100, ensureLoaded)];
    reg.register(placeAt(makeEntry(children, 0, '/g'), 100, 0, -5));

    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    // The fine lazy level is never fetched while the group is frustum-culled.
    expect(ensureLoaded).not.toHaveBeenCalled();
    expect(children[0].object.visible).toBe(true); // coarsest stays shown
  });

  it('auto: an on-screen group still selects the fine level by projected diagonal', () => {
    // Regression guard: the gate is transparent when the group is in view.
    const reg = registryWith(cameraLookingDownNegZ());
    const children = [
      { ...makeChild(0), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
      { ...makeChild(10), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
    ];
    // Centered in front at depth 3 → inside the frustum, projects to a large
    // (>> 10 px) diagonal, so the diagonal selector upgrades 0 → 1.
    const entry = placeAt(makeEntry(children, 0, '/g'), 0, 0, -3);
    reg.register(entry);

    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
    expect(entry.offScreen).toBe(false); // on-screen → no "(off-screen)" hint
  });

  it('eviction: demotes an off-screen group before an on-screen colder level', () => {
    // On-screen entry: in front, hidden level is the *coldest* by tick (1).
    const relOn = vi.fn();
    const onChildren = [
      readyLevel(0, { tick: 50 }), // active — exempt
      readyLevel(100, { release: relOn, tick: 1 }), // hidden, coldest by tick
    ];
    // Off-screen entry: far +x, hidden level is *warmer* by tick (40).
    const relOff = vi.fn();
    const offChildren = [
      readyLevel(0, { tick: 60 }), // active — exempt
      readyLevel(100, { release: relOff, tick: 40 }), // hidden, warmer by tick
    ];
    const all = [...onChildren, ...offChildren];
    // 4 ready levels × 100 = 400 resident; budget 350 → exactly one demotion.
    const reg = registryWith(cameraLookingDownNegZ(), 350, residentModel(all));
    reg.register(placeAt(makeEntry(onChildren, 0, '/on'), 0, 0, -5));
    reg.register(placeAt(makeEntry(offChildren, 0, '/off'), 100, 0, -5));
    // Lock both to level 0 so the active level is stable and the auto gate
    // doesn't reshuffle visibility — isolates the eviction comparator.
    reg.setSelectorMode('/on', { lockLevel: 0 });
    reg.setSelectorMode('/off', { lockLevel: 0 });

    reg.evaluatePerFrame();
    // Off-screen-first dominates the tick LRU: the warmer off-screen level is
    // evicted; the colder on-screen level survives.
    expect(relOff).toHaveBeenCalledTimes(1);
    expect(relOn).not.toHaveBeenCalled();
  });

  it('eviction: among off-screen groups, demotes the furthest from the camera first', () => {
    const relNear = vi.fn();
    const relFar = vi.fn();
    const nearChildren = [
      readyLevel(0, { tick: 10 }),
      readyLevel(100, { release: relNear, tick: 5 }),
    ];
    const farChildren = [
      readyLevel(0, { tick: 10 }),
      readyLevel(100, { release: relFar, tick: 5 }),
    ];
    const all = [...nearChildren, ...farChildren];
    const reg = registryWith(cameraLookingDownNegZ(), 350, residentModel(all));
    reg.register(placeAt(makeEntry(nearChildren, 0, '/near'), 50, 0, -5)); // off-screen, nearer
    reg.register(placeAt(makeEntry(farChildren, 0, '/far'), 500, 0, -5)); // off-screen, further
    reg.setSelectorMode('/near', { lockLevel: 0 });
    reg.setSelectorMode('/far', { lockLevel: 0 });

    reg.evaluatePerFrame();
    // Both off-screen, equal tick → distance breaks the tie: furthest first.
    expect(relFar).toHaveBeenCalledTimes(1);
    expect(relNear).not.toHaveBeenCalled();
  });
});
