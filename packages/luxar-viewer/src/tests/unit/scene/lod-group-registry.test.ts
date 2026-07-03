/**
 * Tests for the LODGroupRegistry selector logic.
 *
 * Covers two independent surfaces:
 *
 *   - The pure pick-with-hysteresis function (no THREE / camera): given
 *     coverage_fraction thresholds + a current active index + a coverage
 *     metric (both dimensionless fractions), the right child is selected.
 *     Hysteresis suppresses downgrade flicker but doesn't impede upgrades.
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
  // Coverage-fraction thresholds (dimensionless, in [0,1], coarsest→finest).
  // The `metric` argument is the coverage metric (projected diagonal ÷
  // FILL_FACTOR·viewportDiagonal), also in fraction space.
  const thresholds = [0, 0.1, 0.5];

  it('picks the coarsest child below the first threshold', () => {
    expect(pickChildWithHysteresis(thresholds, 0, 0)).toBe(0);
    expect(pickChildWithHysteresis(thresholds, 0, 0.05)).toBe(0);
  });

  it('picks finer child once the coverage metric reaches its threshold (upgrade)', () => {
    // From child 0 → child 1 when the coverage metric ≥ 0.1 (literal threshold).
    expect(pickChildWithHysteresis(thresholds, 0, 0.1)).toBe(1);
    expect(pickChildWithHysteresis(thresholds, 1, 0.5)).toBe(2);
  });

  it('downgrade requires falling below threshold * (1 - 0.1)', () => {
    // From child 1 with threshold 0.1: downgrade only when the metric < 0.09.
    // At 0.095 (still 10% within), we stay on 1 even though natural is 0.
    expect(pickChildWithHysteresis(thresholds, 1, 0.095)).toBe(1);
    expect(pickChildWithHysteresis(thresholds, 1, 0.089)).toBe(0);
  });

  it('returns -1 for an empty thresholds list', () => {
    expect(pickChildWithHysteresis([], 0, 0.1)).toBe(-1);
  });

  it('uses a gap-relative downgrade band for tightly-spaced levels', () => {
    // Levels 0.01 and 0.011 are within 10% of each other (e.g. from the
    // coverage_fractions ×1.1 monotonicity nudge). The spacing-aware band is
    // ratio * gap = 0.1 * (0.011 - 0.01) = 0.0001, so the deadband is
    // [0.0109, 0.011): level 1 still renders on the way down instead of skipped.
    const tight = [0, 0.01, 0.011];
    expect(pickChildWithHysteresis(tight, 2, 0.01095)).toBe(2); // deadband → stay
    expect(pickChildWithHysteresis(tight, 2, 0.0105)).toBe(1); // one clean level down
    expect(pickChildWithHysteresis(tight, 2, 0.0099)).toBe(0); // into level 0's range
  });

  it('reduces to the threshold-fraction band for the bottom level', () => {
    // prev threshold is 0, so gap == currentThreshold and the band equals
    // the original currentThreshold * (1 - ratio) behaviour.
    expect(pickChildWithHysteresis(thresholds, 1, 0.091)).toBe(1); // 0.091 ≥ 0.09 → stay
    expect(pickChildWithHysteresis(thresholds, 1, 0.089)).toBe(0); // 0.089 < 0.09 → down
  });

  it('handles a current index that no longer satisfies its threshold', () => {
    // current=2 (threshold 0.5) but the metric is only 0.05 → expected
    // downgrade to 0 (well below 0.5 * 0.9).
    expect(pickChildWithHysteresis(thresholds, 2, 0.05)).toBe(0);
  });

  // C2: pin the downgrade band edge EXACTLY. margin = 0.1*(0.1-0) = 0.01, so the
  // boundary is 0.09: `metric < 0.09` downgrades, `>= 0.09` stays. A mutant that
  // changed `<` to `<=` (or the ratio) would shift this edge.
  it('treats the downgrade boundary as exclusive (exactly 0.09 stays on level 1)', () => {
    expect(pickChildWithHysteresis(thresholds, 1, 0.09)).toBe(1); // 0.09 not < 0.09 → stay
    expect(pickChildWithHysteresis(thresholds, 1, 0.089999)).toBe(0); // just below → down
  });

  // M1: pin the upgrade boundary edge. The natural pick uses `threshold <=
  // metric`, so exactly 0.1 upgrades to level 1 and 0.099999 stays at 0.
  it('treats the upgrade threshold as inclusive (exactly 0.1 reaches level 1)', () => {
    expect(pickChildWithHysteresis(thresholds, 0, 0.1)).toBe(1);
    expect(pickChildWithHysteresis(thresholds, 0, 0.099999)).toBe(0);
  });

  // projectBoxDiagonalPx returns +Infinity when the camera is inside/straddling
  // the bbox (→ infinite coverage metric). That must select the finest child
  // from any current index, and must never produce NaN through the downgrade
  // arithmetic.
  it('saturates to the finest child for an infinite metric (camera-inside)', () => {
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
    // current=2 (threshold 0.5), metric 0.095 → natural is 0 (0.095 < 0.1, so
    // only threshold 0 qualifies); well under 0.5 - 0.1*(0.5-0.1)=0.46 → to 0.
    expect(pickChildWithHysteresis(thresholds, 2, 0.095)).toBe(0);
    // current=2, metric 0.15 → natural=1 (0.1≤0.15<0.5); below 0.46 → down to 1.
    expect(pickChildWithHysteresis(thresholds, 2, 0.15)).toBe(1);
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

function makeChild(coverageFraction: number): LODGroupChild {
  return {
    object: new THREE.Group(),
    coverageFraction,
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
function makeLazyChild(coverageFraction: number, ensureLoaded: () => void): LODGroupChild {
  return {
    object: new THREE.Group(),
    coverageFraction,
    positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
    ready: false,
    ensureLoaded,
  };
}

function makeRegistry(
  displayDims: readonly number[] = [0, 1, 2],
  residentByteBudget?: number,
  getResidentBytes?: () => number,
  getViewVersion?: () => number,
  requestRender?: () => void
) {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  return new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => displayDims,
    ...(getViewVersion != null ? { getViewVersion } : {}),
    ...(requestRender != null ? { requestRender } : {}),
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
 * A gsplats LOD child stamped with the view-version its (ready) geometry was
 * committed for — what ``commitGSplatsGeometry`` writes. The registry's
 * slice-aware fallback reads ``object.userData.{nodeType,loadedViewVersion}``;
 * a plain ``makeChild`` (a bare ``THREE.Group`` with no ``nodeType``) is always
 * treated as fresh, so the fallback only engages for gsplats children.
 */
function makeGsplatChild(coverageFraction: number, loadedViewVersion: number): LODGroupChild {
  const child = makeChild(coverageFraction);
  child.object.userData = { nodeType: 'gsplats', loadedViewVersion };
  return child;
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
    const children = [makeChild(0), makeChild(0.5), makeChild(1.0)];
    reg.register(makeEntry(children, 1));
    expect(children[0].object.visible).toBe(false);
    expect(children[1].object.visible).toBe(true);
    expect(children[2].object.visible).toBe(false);
  });

  it('unregister removes the entry without disturbing visibility', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(0.5)];
    reg.register(makeEntry(children, 0, '/g'));
    expect(reg.size()).toBe(1);
    reg.unregister('/g');
    expect(reg.size()).toBe(0);
  });

  it('list returns all registered entries', () => {
    const reg = makeRegistry();
    reg.register(makeEntry([makeChild(0), makeChild(0.5)], 0, '/g0'));
    reg.register(makeEntry([makeChild(0), makeChild(0.5)], 0, '/g1'));
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
    const children = [makeChild(0), makeChild(0.5)];
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
    const children = [makeChild(0), makeChild(0.5)];
    reg.register(makeEntry(children, 1, '/g'));
    reg.setSelectorMode('/g', { lockLevel: -3 });
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true);
    expect(children[1].object.visible).toBe(false);
  });

  it('lock override swaps to the locked child on the next evaluation', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(0.5), makeChild(1.0)];
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
    const children = [makeChild(0), makeChild(0.5), makeChild(1.0)];
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
    const children = [makeChild(0), makeChild(0.5)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
  });
});

describe('LODGroupRegistry — auto evaluation', () => {
  it('picks the finest child whose coverage_fraction is satisfied by the projected diagonal', () => {
    // bbox spans the full NDC cube → identity camera projects to a 1000 px
    // diagonal on an 800x600 viewport, whose own diagonal is hypot(800,600)=1000
    // → coverage metric = 1000/1000 = 1.0 (the group fills the screen). With
    // coverage_fraction thresholds [0, 0.5, 1.0], the finest applicable (1.0) is
    // child 2.
    const reg = makeRegistry();
    const children = [
      { ...makeChild(0), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
      { ...makeChild(0.5), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
      { ...makeChild(1.0), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
    ];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[2].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
    expect(children[1].object.visible).toBe(false);
  });

  it('normalizes coverage by viewport size: a full-viewport box picks the finest (coverage=1.0) child on ANY viewport', () => {
    // The coverage metric is diagonalPx / (FILL_FACTOR·viewportDiagonal), so a
    // box that fills the NDC cube projects to a diagonal equal to the viewport
    // diagonal on ANY viewport size → coverage metric == 1.0 regardless. Both a
    // small and a large viewport must therefore pick the finest coverage=1.0
    // child. This pins the viewport-relative normalization (the whole point of
    // switching from absolute pixels to a coverage fraction).
    const fullBox = { min: [-1, -1, -1], max: [1, 1, 1] };
    for (const viewport of [
      { width: 400, height: 300 }, // small
      { width: 3840, height: 2160 }, // large (4K)
    ]) {
      const camera = new THREE.Camera();
      camera.matrixWorldInverse.identity();
      camera.projectionMatrix.identity();
      const reg = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => viewport,
        getDisplayDims: () => [0, 1, 2],
      });
      const children = [
        { ...makeChild(0), positionBounds: fullBox },
        { ...makeChild(0.5), positionBounds: fullBox },
        { ...makeChild(1.0), positionBounds: fullBox },
      ];
      reg.register(makeEntry(children, 0, '/g'));
      reg.evaluatePerFrame();
      expect(children[2].object.visible, `finest at viewport ${viewport.width}x${viewport.height}`).toBe(true);
      expect(children[0].object.visible).toBe(false);
      expect(children[1].object.visible).toBe(false);
    }
  });

  it('divides by the viewport diagonal: a HALF-viewport box picks the middle child, not the finest, on ANY viewport', () => {
    // Discriminating test for the normalization (the full-viewport test above
    // can't: it gives metric 1.0 whether or not you divide). A box spanning
    // NDC [-0.5, 0.5] projects to a diagonal of 0.5·hypot(w,h) → coverage
    // metric = 0.5·hypot / hypot = 0.5 on ANY viewport. With thresholds
    // [0, 0.5, 1.0] the finest applicable to 0.5 is the MIDDLE child (index 1).
    // A buggy selector that skipped the ÷viewportDiagonal step would compare the
    // raw pixel diagonal (250 px on 400×300, 2203 px on 4K — both ≫ 1.0) and
    // wrongly pick the finest (index 2) on both. So this pins the division AND
    // its viewport-independence.
    const halfBox = { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] };
    for (const viewport of [
      { width: 400, height: 300 },
      { width: 3840, height: 2160 },
    ]) {
      const camera = new THREE.Camera();
      camera.matrixWorldInverse.identity();
      camera.projectionMatrix.identity();
      const reg = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => viewport,
        getDisplayDims: () => [0, 1, 2],
      });
      const children = [
        { ...makeChild(0), positionBounds: halfBox },
        { ...makeChild(0.5), positionBounds: halfBox },
        { ...makeChild(1.0), positionBounds: halfBox },
      ];
      reg.register(makeEntry(children, 0, '/g'));
      reg.evaluatePerFrame();
      expect(children[1].object.visible, `middle at ${viewport.width}x${viewport.height}`).toBe(true);
      expect(children[2].object.visible, `NOT finest at ${viewport.width}x${viewport.height}`).toBe(false);
      expect(children[0].object.visible).toBe(false);
    }
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
    const children = [makeChild(0), makeChild(0.5)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    // Active stayed on 0 because evaluation early-returned.
    expect(children[0].object.visible).toBe(true);
  });
});

describe('LODGroupRegistry — lazy children', () => {
  it('register leaves a not-ready non-active child hidden', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
    reg.register(makeEntry(children, 0, '/g'));
    expect(children[0].object.visible).toBe(true); // active + ready
    expect(children[1].object.visible).toBe(false); // not ready → hidden
  });

  it('does not swap to a not-ready child; fires ensureLoaded exactly once across frames', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
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
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
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
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    children[1].failed = true; // a prior load attempt failed
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  /** A loaded lazy level with a release spy + a prior-visible tick. */
  function readyLazy(
    coverageFraction: number,
    opts: { release?: ReturnType<typeof vi.fn>; tick?: number } = {}
  ): LODGroupChild {
    return {
      ...makeLazyChild(coverageFraction, () => {}),
      ready: true,
      release: (opts.release ?? vi.fn()) as () => void,
      lastVisibleTick: opts.tick,
    };
  }

  it('does not release on swap (pure retention when no byte budget is wired)', () => {
    const reg = makeRegistry(); // no budget → no eviction
    const rel0 = vi.fn();
    const children = [readyLazy(0, { release: rel0 }), readyLazy(0.5), readyLazy(1.0)];
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
      readyLazy(0.5, { release: rel1, tick: 5 }),
      readyLazy(1.0), // will be the locked/active level
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
      readyLazy(0.5, { release: relActive, tick: 3 }),
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
    const loading = makeLazyChild(0.5, () => {});
    loading.loading = true; // in flight, ready still false
    loading.release = relLoading;
    const children = [readyLazy(0, { tick: 9 }), loading];
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children));
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 });

    reg.evaluatePerFrame();
    expect(relLoading).not.toHaveBeenCalled();
  });

  it('never evicts a READY level whose deferred RELOAD is in flight (B2 finding 7)', () => {
    // Distinct from the not-ready case above: a stale fine level being reloaded
    // keeps ready=true (prior geometry committed) while loading=true. Evicting
    // it mid-reload would reset ready/loading via release(), and the next frame
    // the registry would kick a SECOND concurrent ensureLoaded on the same
    // loader. The eviction filter's `!child.loading` clause is the only thing
    // preventing that — this test fails if that clause is removed.
    const relReloading = vi.fn();
    const reloading = readyLazy(0.5, { release: relReloading, tick: 1 });
    reloading.loading = true; // ready=true AND loading=true → stale reload in flight
    const children = [readyLazy(0, { tick: 9 }), reloading];
    // residentModel counts both ready children (200 bytes) > 50 budget → the
    // eviction pass runs and would target `reloading` (hidden, has release/tick)
    // if not for the loading guard.
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children));
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 });

    reg.evaluatePerFrame();
    expect(relReloading).not.toHaveBeenCalled();
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
      readyLazy(0.5, { release: relStranded }), // loaded, never shown
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
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
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
    reg.register(makeEntry([makeChild(0), makeChild(0.5)], 0, '/g'));
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    reg.clear();
    // After clear, a freshly-registered entry's first stamp should start
    // from a reset clock. We can't read tick directly, but a re-registered
    // child must not inherit a stale high tick — exercise the path.
    const children = [makeChild(0), makeChild(0.5)];
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
    coverageFraction: number,
    opts: { release?: ReturnType<typeof vi.fn>; tick?: number } = {}
  ): LODGroupChild {
    return {
      ...makeLazyChild(coverageFraction, () => {}),
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
    const children = [makeChild(0), makeChild(0.5)]; // both ready (non-lazy)
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
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
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
      { ...makeChild(0.1), positionBounds: { min: [-1, -1, -1], max: [1, 1, 1] } },
    ];
    // Centered in front at depth 3 → inside the frustum, projects to a large
    // diagonal → coverage metric ≈ 0.73 (well above child 1's 0.1 threshold),
    // so the coverage selector upgrades 0 → 1.
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
      readyLevel(0.5, { release: relOn, tick: 1 }), // hidden, coldest by tick
    ];
    // Off-screen entry: far +x, hidden level is *warmer* by tick (40).
    const relOff = vi.fn();
    const offChildren = [
      readyLevel(0, { tick: 60 }), // active — exempt
      readyLevel(0.5, { release: relOff, tick: 40 }), // hidden, warmer by tick
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
      readyLevel(0.5, { release: relNear, tick: 5 }),
    ];
    const farChildren = [
      readyLevel(0, { tick: 10 }),
      readyLevel(0.5, { release: relFar, tick: 5 }),
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

// ────────────────────────────────────────────────────────────────────────
// Slice-aware freshness fallback — show the coarsest level that is fresh for
// the current view-version while the screen-desired (fine) level reloads after
// a slice / displayDims change, then swap up once it recommits.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — slice-aware freshness fallback', () => {
  it('displays the coarsest FRESH level while the screen-desired level is stale', () => {
    // coarse fresh@2, fine stale@1; current version 2. Identity camera → the
    // huge on-screen diagonal makes the finest level the aspiration, but it's
    // stale → display falls back to the fresh coarse level.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const children = [makeGsplatChild(0, 2), makeGsplatChild(0.5, 1)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true);
    expect(children[1].object.visible).toBe(false);
    // Aspiration still advanced to the finest level (it's what we want on screen).
    expect(reg.list()[0].activeChildIndex).toBe(1);
  });

  it('swaps up to the fine level once it commits for the current version', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const children = [makeGsplatChild(0, 2), makeGsplatChild(0.5, 1)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame(); // fine stale → coarse shown
    expect(children[0].object.visible).toBe(true);
    // Fine recommits for V2 (the re-slice load landed).
    (children[1].object.userData as { loadedViewVersion: number }).loadedViewVersion = 2;
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
  });

  it('returns changed=true on the fallback (fine→coarse) and swap-up (coarse→fine) frames', () => {
    let version = 1;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => version);
    const children = [makeGsplatChild(0, 1), makeGsplatChild(0.5, 1)]; // both fresh@1
    reg.register(makeEntry(children, 0, '/g'));
    expect(reg.evaluatePerFrame()).toBe(true); // swap up 0→1 (fine fresh)
    expect(children[1].object.visible).toBe(true);
    // Scrub to a new view-version: both stamps now lag → stale.
    version = 2;
    expect(reg.evaluatePerFrame()).toBe(true); // display drops fine→coarse
    expect(children[0].object.visible).toBe(true);
    // Fine recommits for V2 → swap back up.
    (children[1].object.userData as { loadedViewVersion: number }).loadedViewVersion = 2;
    expect(reg.evaluatePerFrame()).toBe(true); // coarse→fine
    expect(children[1].object.visible).toBe(true);
  });

  it('falls back to the coarsest READY level (never blank) when no level is fresh', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 9);
    const children = [makeGsplatChild(0, 1), makeGsplatChild(0.5, 1)]; // both stale for V9
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true); // coarsest ready shown
    expect(children.filter((c) => c.object.visible).length).toBe(1); // exactly one, never blank
  });

  it('never evicts the DISPLAYED coarse-fresh level even though the aspiration is the fine level', () => {
    // Pins the §4 fix: the eviction guard must protect ``displayedChildIndex``,
    // not ``activeChildIndex``. Here display=coarse(0), aspiration=fine(1).
    // Guarding the aspiration instead would free the on-screen coarse → blank.
    const relCoarse = vi.fn();
    const relFine = vi.fn();
    const coarse = {
      ...makeGsplatChild(0, 2),
      ready: true,
      release: relCoarse as () => void,
      lastVisibleTick: 5,
    };
    const fine = {
      ...makeGsplatChild(0.5, 1),
      ready: true,
      release: relFine as () => void,
      lastVisibleTick: 5,
    };
    const children = [coarse, fine];
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children), () => 2);
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 }); // aspiration = fine
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true); // coarse-fresh displayed
    expect(relCoarse).not.toHaveBeenCalled(); // displayed level never freed
    expect(relFine).toHaveBeenCalledTimes(1); // hidden aspiration is the candidate
  });

  it('a locked level that is stale still shows the coarse-fresh level until it commits', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const children = [makeGsplatChild(0, 2), makeGsplatChild(0.5, 1)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true); // coarse-fresh while locked-fine reloads
    (children[1].object.userData as { loadedViewVersion: number }).loadedViewVersion = 2;
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true); // locked fine shown once fresh
  });

  it('no-op when the desired level is already fresh (display == aspiration)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const children = [makeGsplatChild(0, 2), makeGsplatChild(0.5, 2)]; // both fresh@2
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true); // finest shown, no fallback
  });

  it('no-op for non-gsplats LOD children (freshness tracked only for gsplats)', () => {
    // Plain children (no nodeType:'gsplats') are always fresh even with a
    // version wired — points/lines LOD groups are unaffected.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 999);
    const children = [makeChild(0), makeChild(0.5)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
  });

  it('treats every ready level as fresh when getViewVersion is not wired (pre-feature parity)', () => {
    const reg = makeRegistry(); // no getViewVersion dep
    const children = [makeGsplatChild(0, 1), makeGsplatChild(0.5, 1)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true); // finest shown (no freshness gating)
  });
});

// ────────────────────────────────────────────────────────────────────────
// Settle-gated reload of a stale fine level (B2 decoupling). A lazy fine level
// that has left the per-slice sweep is reloaded by the REGISTRY — but only once
// the scrub has settled (the view version held steady for FINE_RELOAD_SETTLE_TICKS
// frames), so active scrubbing shows only the cheap coarse level.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — settle-gated fine reload', () => {
  /** A ready-but-stale lazy fine gsplats child with an ensureLoaded spy. */
  function makeStaleLazyFine(coverageFraction: number, staleVersion: number, ensureLoaded: () => void) {
    const child = makeGsplatChild(coverageFraction, staleVersion);
    child.ready = true; // committed (just stale for the current version)
    child.ensureLoaded = ensureLoaded;
    return child;
  }

  it('does NOT reload the stale fine level while the version is still changing (scrubbing)', () => {
    let version = 1;
    const ensureLoaded = vi.fn();
    const children = [makeGsplatChild(0, 1), makeStaleLazyFine(0.5, 1, ensureLoaded)];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => version);
    reg.register(makeEntry(children, 0, '/g'));
    // Scrub every frame: the version keeps changing so it never settles.
    for (let i = 0; i < 12; i++) {
      version += 1;
      reg.evaluatePerFrame();
    }
    expect(ensureLoaded).not.toHaveBeenCalled();
    // ...and the coarse fallback is what's displayed meanwhile.
    expect(children[0].object.visible).toBe(true);
    expect(children[1].object.visible).toBe(false);
  });

  it('reloads the stale fine level exactly once after the scrub settles', () => {
    const ensureLoaded = vi.fn();
    const children = [makeGsplatChild(0, 2), makeStaleLazyFine(0.5, 1, ensureLoaded)];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2); // version fixed at 2
    reg.register(makeEntry(children, 0, '/g'));
    // A few frames: not yet settled → no reload.
    for (let i = 0; i < 4; i++) reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
    // Hold steady long enough to settle (FINE_RELOAD_SETTLE_TICKS ~ 8).
    for (let i = 0; i < 10; i++) reg.evaluatePerFrame();
    // Fired once; the loading guard prevents re-firing every subsequent frame.
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    // The coarse-fresh level stays displayed while the fine reloads.
    expect(children[0].object.visible).toBe(true);
  });

  it('never reloads an eager (sweep-driven) coarse level — it has no ensureLoaded', () => {
    // Both children stale + ready but NEITHER has ensureLoaded (eager levels):
    // the registry must not attempt a reload (that would throw on undefined).
    const children = [makeGsplatChild(0, 1), makeGsplatChild(0.5, 1)];
    children.forEach((c) => (c.ready = true));
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    reg.register(makeEntry(children, 0, '/g'));
    expect(() => {
      for (let i = 0; i < 12; i++) reg.evaluatePerFrame();
    }).not.toThrow();
  });
});

// ────────────────────────────────────────────────────────────────────────
// Render-loop keep-alive while a lazy level loads (B2). The viewer is
// on-demand and idles after ~2s; a deferred fine reload commits OUTSIDE the
// per-slice sweep and can outlast the idle timeout, so the registry must keep
// requesting renders while any child is loading or the swap-up never fires.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — render keep-alive while loading', () => {
  it('requests a render every frame while a child is loading', () => {
    const requestRender = vi.fn();
    const loading = makeLazyChild(0.5, () => {});
    loading.loading = true; // a deferred (re)load in flight
    const children = [makeGsplatChild(0, 1), loading];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 1, requestRender);
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(requestRender).toHaveBeenCalledTimes(2); // kept alive each frame
  });

  it('does NOT request renders when nothing is loading (power-saving preserved)', () => {
    const requestRender = vi.fn();
    const children = [makeGsplatChild(0, 1), makeGsplatChild(0.5, 1)];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 1, requestRender);
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(requestRender).not.toHaveBeenCalled();
  });
});

// ────────────────────────────────────────────────────────────────────────
// Registry-driven progressive refinement (B2 finding 6). A lazy level backed
// by a progressive (additive-laddered) loader is fresh but reports hasMoreLODs;
// since lazy levels no longer ride the per-slice sweep, the registry must keep
// re-firing ensureLoaded (settle-gated) to advance the ladder to completion.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — progressive refinement of a lazy level', () => {
  it('re-fires ensureLoaded while hasMoreLODs (fresh but incomplete), and stops when complete', () => {
    let more = true;
    const fine = makeGsplatChild(0.5, 2); // ready & fresh for version 2
    fine.ready = true;
    // Spy simulates a completed progressive pass (clears loading so the next
    // settled frame can advance the ladder again).
    const ensureLoaded = vi.fn(() => {
      fine.loading = false;
    });
    fine.ensureLoaded = ensureLoaded;
    fine.hasMoreLODs = () => more;
    const children = [makeGsplatChild(0, 2), fine];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    reg.register(makeEntry(children, 0, '/g'));

    // Settle (~8 frames) then several refinement passes.
    for (let i = 0; i < 16; i++) reg.evaluatePerFrame();
    expect(ensureLoaded.mock.calls.length).toBeGreaterThan(1); // ladder advancing
    expect(fine.object.visible).toBe(true); // fresh progressive level stays displayed

    // Ladder complete → no further passes.
    more = false;
    const before = ensureLoaded.mock.calls.length;
    for (let i = 0; i < 6; i++) reg.evaluatePerFrame();
    expect(ensureLoaded.mock.calls.length).toBe(before);
  });
});
