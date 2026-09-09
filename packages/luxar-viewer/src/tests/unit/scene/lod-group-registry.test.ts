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
  FILL_FACTOR,
  LODGroupRegistry,
  pickChildWithHysteresis,
  projectBoxAreaFraction,
  projectBoxDiagonalPx,
  SCREEN_FILL_DIAGONAL_RATIO,
  type LODGroupChild,
  type LODGroupEntry,
  type PartitionGroupEntry,
} from '../../../scene/lod-group-registry';
import {
  computeEntryWorldBox,
  DEGENERATE_RECT_HALF_EXTENT,
} from '../../../scene/lod-selector-math';
import {
  calculateCameraDistance,
  type BoundingBox,
} from '../../../scene/scene-manager/clipping/bounds-math';
import { updateCameraAspect } from '../../../utils/camera-utils';
import { log } from '../../../utils/log';

describe('computeEntryWorldBox', () => {
  it('reuses the caller-owned world box', () => {
    const groupObject = new THREE.Group();
    groupObject.position.set(5, 10, 15);
    const localBoxScratch: BoundingBox = {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
    };
    const worldBoxScratch: BoundingBox = {
      min: { x: 99, y: 99, z: 99 },
      max: { x: 99, y: 99, z: 99 },
    };

    const result = computeEntryWorldBox(
      {
        groupObject,
        children: [{ positionBounds: { min: [0, 0, 0], max: [1, 1, 1] } }],
      },
      [0, 1, 2],
      localBoxScratch,
      new Array<number>(16),
      { worldBoxScratch }
    );

    expect(result).toBe(worldBoxScratch);
    expect(result).toEqual({
      min: { x: 5, y: 10, z: 15 },
      max: { x: 6, y: 11, z: 16 },
    });
  });
});

// ────────────────────────────────────────────────────────────────────────
// pickChildWithHysteresis — pure selector math
// ────────────────────────────────────────────────────────────────────────

describe('pickChildWithHysteresis', () => {
  // Coverage-fraction thresholds (dimensionless, in [0,1], coarsest→finest).
  // The `metric` argument is the coverage metric (projected diagonal ÷
  // FILL_FACTOR·fittedAxisPx), also in fraction space.
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

  // ── projectBoxAreaFraction — the selector='screen-area' metric ──────────
  describe('projectBoxAreaFraction', () => {
    it('returns 1.0 for a screen-filling box (full NDC extent)', () => {
      const box: BoundingBox = { min: { x: -1, y: -1, z: 0 }, max: { x: 1, y: 1, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBeCloseTo(1.0, 6);
    });

    it('returns the covered AREA fraction, not an extent: half-NDC box → 1/4', () => {
      // Half the screen along EACH axis covers a quarter of its area — the
      // property the whole selector is named for. An extent-shaped mutant
      // (returning halfW, or hypot-like math) would give 0.5/0.7 here.
      const box: BoundingBox = { min: { x: -0.5, y: -0.5, z: 0 }, max: { x: 0.5, y: 0.5, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBeCloseTo(0.25, 6);
    });

    it('multiplies the two axis fractions (asymmetric rect)', () => {
      // x spans the full screen (fraction 1), y a quarter of it (0.25) → 0.25.
      // Pins the PRODUCT against a max/min/hypot-of-axes mutant.
      const box: BoundingBox = { min: { x: -1, y: -0.25, z: 0 }, max: { x: 1, y: 0.25, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBeCloseTo(0.25, 6);
    });

    it('clips to the viewport: past full-screen the metric tops out at exactly 1.0', () => {
      // NDC extent 4 per axis, but only the [-1,1]² viewport is VISIBLE →
      // occupancy 1.0 exactly. The natural pick uses `threshold <= metric`,
      // so the fills-screen partition threshold (1.0) is satisfied the moment
      // coverage is complete and stays satisfied while zoomed past it.
      const box: BoundingBox = { min: { x: -2, y: -2, z: 0 }, max: { x: 2, y: 2, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBeCloseTo(1.0, 6);
    });

    it('clips to the viewport: a huge rect intersecting only a screen corner reads its small VISIBLE fraction', () => {
      // Rect spans NDC [0.5, 10] on both axes — enormous unclipped (~22.5 area
      // units) but only the [0.5, 1]² corner is on screen: visible fraction =
      // (0.25)·(0.25) = 0.0625. The review-caught failure: unclipped, this
      // read arbitrarily large occupancy and pinned the finest level while
      // panning across partition tiles.
      const box: BoundingBox = { min: { x: 0.5, y: 0.5, z: 0 }, max: { x: 10, y: 10, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBeCloseTo(0.0625, 6);
    });

    it('saturates to +Infinity when the camera is inside the box (perspective)', () => {
      const box: BoundingBox = { min: { x: -5, y: -5, z: -5 }, max: { x: 5, y: 5, z: 5 } };
      expect(projectBoxAreaFraction(box, perspectiveAtOrigin())).toBe(Number.POSITIVE_INFINITY);
    });

    it('a fully off-screen DEGENERATE rect reads 0, not the other axis span', () => {
      // The review-caught interaction bug between clipping and the degenerate
      // ramp: after clamping, "no viewport overlap on Y" and "zero-thickness
      // visible line" both produced a zero clipped half-extent, so this
      // full-width rect entirely above the viewport (y in [2, 3]) returned
      // 1.0 (finest) instead of 0 — selecting expensive levels for geometry
      // not on screen at all whenever the conservative frustum gate let it
      // through near a corner.
      const box: BoundingBox = { min: { x: -1, y: 2, z: 0 }, max: { x: 1, y: 3, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBe(0);
      // Off-screen zero-thickness line (raw-degenerate AND off-screen) too.
      const line: BoundingBox = { min: { x: -1, y: 2, z: 0 }, max: { x: 1, y: 2, z: 0 } };
      expect(projectBoxAreaFraction(line, identityCamera())).toBe(0);
      // And plain off-screen non-degenerate.
      const sq: BoundingBox = { min: { x: 2, y: 2, z: 0 }, max: { x: 4, y: 4, z: 0 } };
      expect(projectBoxAreaFraction(sq, identityCamera())).toBe(0);
    });

    it('a wide 2D node panned to a thin visible sliver reads its tiny area, not a linear span', () => {
      // The ramp is gated on RAW (pre-clip) thinness: this square is 2 NDC
      // units tall (not thin content) but only a 0.001-half sliver remains on
      // screen — the honest visible occupancy is ~0.001, and inflating it to
      // the full-width linear span (1.0 → finest) would resurrect the
      // unclipped-corner cost while panning.
      const box: BoundingBox = { min: { x: -1, y: 0.998, z: 0 }, max: { x: 1, y: 3, z: 0 } };
      expect(projectBoxAreaFraction(box, identityCamera())).toBeCloseTo(0.001, 5);
    });

    it('degenerate rect (zero thickness) falls back to the LINEAR span, not area 0', () => {
      // An axis-aligned straight polyline: full-width, zero-height projected
      // bounds. The raw area product is exactly 0, which would pin the node to
      // the coarsest level forever no matter how much screen it spans (the
      // review-caught regression: the legacy diagonal metric read the long
      // extent for these shapes). The metric must instead read the linear
      // span: full width → 1.0.
      const line: BoundingBox = { min: { x: -1, y: 0, z: 0 }, max: { x: 1, y: 0, z: 0 } };
      expect(projectBoxAreaFraction(line, identityCamera())).toBeCloseTo(1.0, 6);
      // Same for the vertical orientation (branch must take max, not halfW).
      const vline: BoundingBox = { min: { x: 0, y: -0.5, z: 0 }, max: { x: 0, y: 0.5, z: 0 } };
      expect(projectBoxAreaFraction(vline, identityCamera())).toBeCloseTo(0.5, 6);
    });

    it('a point (both axes degenerate) still reads ~0 → coarsest', () => {
      const point: BoundingBox = { min: { x: 0.2, y: 0.2, z: 0 }, max: { x: 0.2, y: 0.2, z: 0 } };
      expect(projectBoxAreaFraction(point, identityCamera())).toBeCloseTo(0, 6);
    });

    it('barely-non-degenerate rects stay on the area product (no early fallback)', () => {
      // Thickness just ABOVE the sub-pixel degeneracy floor must NOT take the
      // linear-span ramp — otherwise every thin-but-real object would jump
      // to a wildly finer level. halfH = 0.01 (≈10px on 1080p) → area path:
      // 1.0 × 0.01 = 0.01, NOT the linear 1.0.
      const thin: BoundingBox = { min: { x: -1, y: -0.01, z: 0 }, max: { x: 1, y: 0.01, z: 0 } };
      expect(projectBoxAreaFraction(thin, identityCamera())).toBeCloseTo(0.01, 6);
    });

    it('the degenerate fallback is a CONTINUOUS ramp, not a cliff at the floor', () => {
      // Review-caught oscillation hazard: a hard cutover at the degeneracy
      // floor meant halfH=0.001 → 1.0 (finest) vs halfH=0.001001 → ~0.001
      // (coarsest) — a three-orders jump no hysteresis can absorb when an
      // edge-on plane rotates across it. The metric is now
      // max(area, span·(1 − thin/floor)):
      //   thin = floor/2 (0.0005): max(0.0005, 1·0.5) = 0.5 — midway;
      //   thin = floor exactly:    max(0.001, 1·0)   = 0.001 — meets the
      //     area product with NO jump (the ramp has decayed to zero);
      //   and values sampled across the floor differ smoothly.
      const floor = DEGENERATE_RECT_HALF_EXTENT;
      const mk = (halfH: number): BoundingBox => ({
        min: { x: -1, y: -halfH, z: 0 },
        max: { x: 1, y: halfH, z: 0 },
      });
      expect(projectBoxAreaFraction(mk(floor / 2), identityCamera())).toBeCloseTo(0.5, 6);
      expect(projectBoxAreaFraction(mk(floor), identityCamera())).toBeCloseTo(floor, 6);
      // Just below vs just above the floor: both ~the area product — smooth.
      const below = projectBoxAreaFraction(mk(floor * 0.99), identityCamera());
      const above = projectBoxAreaFraction(mk(floor * 1.01), identityCamera());
      expect(Math.abs(below - above)).toBeLessThan(0.02);
    });

    it('never saturates for an orthographic camera (w stays 1) — the projection stays well-defined', () => {
      // INTENDED, and identical to the legacy diagonal metric's contract (the
      // ortho pin above): saturation is the PERSPECTIVE near-plane guard,
      // where the homogeneous divide degenerates. Ortho never degenerates, so
      // the plain clipped metric applies — here the camera is inside the box
      // and the box's rect covers NDC ±0.5 per axis → 0.5 × 0.5 = 0.25 (a
      // camera inside a LARGE node reads full coverage naturally instead;
      // this box simply doesn't span the viewport). See the v3.4 spec's
      // normative metric rules.
      const box: BoundingBox = { min: { x: -5, y: -5, z: -5 }, max: { x: 5, y: 5, z: 5 } };
      expect(projectBoxAreaFraction(box, orthoAtOrigin())).toBeCloseTo(0.25, 6);
    });
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

function withLodBounds(
  child: LODGroupChild,
  lodBounds: { min: readonly number[]; max: readonly number[] }
): LODGroupChild {
  child.lodBounds = lodBounds;
  return child;
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
  requestRender?: () => void,
  now?: () => number,
  hasArchiveFault?: () => boolean,
  requestReprocess?: (paths: readonly string[]) => void,
  isUpdateInProgress?: () => boolean
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
    ...(now != null ? { now } : {}),
    ...(hasArchiveFault != null ? { hasArchiveFault } : {}),
    ...(requestReprocess != null ? { requestReprocess } : {}),
    ...(isUpdateInProgress != null ? { isUpdateInProgress } : {}),
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

describe('LODGroupRegistry — partition frustum selection', () => {
  it('gates and restores every object emitted by a part', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const first = new THREE.Group();
    const second = new THREE.Group();
    groupObject.add(first, second);
    const entry: PartitionGroupEntry = {
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [first, second],
          positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
        },
      ],
    };

    reg.registerPartition(entry);
    expect(reg.evaluatePerFrame()).toBe(true);
    for (const object of [first, second]) {
      expect(object.visible).toBe(false);
      expect(object.userData.partitionFrustumVisible).toBe(false);
    }

    reg.unregister(entry.path);
    for (const object of [first, second]) {
      expect(object.visible).toBe(true);
      expect(object.userData.partitionFrustumVisible).toBeUndefined();
    }
  });

  it('hides only parts outside the frustum using mapped 2D display dimensions', () => {
    const reg = makeRegistry([1, 3]);
    const groupObject = new THREE.Group();
    const visible = new THREE.Group();
    const culled = new THREE.Group();
    groupObject.add(visible, culled);
    const entry: PartitionGroupEntry = {
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [visible],
          positionBounds: { min: [20, -0.5, 30, -0.5], max: [21, 0.5, 31, 0.5] },
        },
        {
          path: '/partition/part_1',
          objects: [culled],
          positionBounds: { min: [-0.5, 2, -0.5, 2], max: [0.5, 3, 0.5, 3] },
        },
      ],
    };

    reg.registerPartition(entry);
    expect(reg.evaluatePerFrame()).toBe(true);

    expect(visible.visible).toBe(true);
    expect(visible.userData.partitionFrustumVisible).toBe(true);
    expect(culled.visible).toBe(false);
    expect(culled.userData.partitionFrustumVisible).toBe(false);
  });

  it('keeps a part visible when its bounds cannot be projected safely', () => {
    const reg = makeRegistry([0, 1]);
    const groupObject = new THREE.Group();
    const culled = new THREE.Group();
    const unprojectable = new THREE.Group();
    groupObject.add(culled, unprojectable);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [culled],
          positionBounds: { min: [2, 2], max: [3, 3] },
        },
        {
          path: '/partition/part_1',
          objects: [unprojectable],
          positionBounds: { min: [], max: [] },
        },
      ],
    });

    expect(reg.evaluatePerFrame()).toBe(true);
    expect(culled.visible).toBe(false);
    expect(culled.userData.partitionFrustumVisible).toBe(false);
    expect(unprojectable.visible).toBe(true);
    expect(unprojectable.userData.partitionFrustumVisible).toBe(true);
  });

  it('keeps a part visible when its rendered footprint intersects the frustum', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const footprintGeometry = new THREE.BufferGeometry();
    footprintGeometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(0.9, -0.1, -0.1),
      new THREE.Vector3(1.3, 0.1, 0.1)
    );
    const footprintVisible = new THREE.Mesh(footprintGeometry);
    const culled = new THREE.Group();
    groupObject.add(footprintVisible, culled);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [footprintVisible],
          positionBounds: { min: [1.2, -0.1, -0.1], max: [1.3, 0.1, 0.1] },
        },
        {
          path: '/partition/part_1',
          objects: [culled],
          positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
        },
      ],
    });

    expect(reg.evaluatePerFrame()).toBe(true);
    expect(footprintVisible.visible).toBe(true);
    expect(footprintVisible.userData.partitionFrustumVisible).toBe(true);
    expect(culled.visible).toBe(false);
  });

  it('reuses a static part footprint across frames', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const geometry = new THREE.BufferGeometry();
    geometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(-0.5, -0.5, -0.5),
      new THREE.Vector3(0.5, 0.5, 0.5)
    );
    const child = new THREE.Mesh(geometry);
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
        },
      ],
    });
    const setFromObject = vi.spyOn(THREE.Box3.prototype, 'setFromObject');

    reg.evaluatePerFrame();
    reg.evaluatePerFrame();

    // Restore before asserting so a failure can't leak the spy into later tests.
    const captures = setFromObject.mock.calls.length;
    setFromObject.mockRestore();
    expect(captures).toBe(1);
  });

  it('captures every object of a part once and reuses them across frames', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const near = new THREE.Mesh(
      Object.assign(new THREE.BufferGeometry(), {
        boundingBox: new THREE.Box3(
          new THREE.Vector3(-0.5, -0.5, -0.5),
          new THREE.Vector3(0.5, 0.5, 0.5)
        ),
      })
    );
    const far = new THREE.Mesh(
      Object.assign(new THREE.BufferGeometry(), {
        boundingBox: new THREE.Box3(new THREE.Vector3(2, 2, 2), new THREE.Vector3(3, 3, 3)),
      })
    );
    groupObject.add(near, far);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [near, far],
          positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
        },
      ],
    });
    const setFromObject = vi.spyOn(THREE.Box3.prototype, 'setFromObject');

    reg.evaluatePerFrame();
    reg.evaluatePerFrame();

    // One capture per object on the first frame, nothing on the second.
    const captures = setFromObject.mock.calls.length;
    setFromObject.mockRestore();
    expect(captures).toBe(2);
    // The union still reaches the near object, so the part stays visible.
    expect(near.visible).toBe(true);
    expect(far.visible).toBe(true);
  });

  it('refreshes a cached footprint after the partition transform changes', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const geometry = new THREE.BufferGeometry();
    geometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(-2, -0.05, -0.05),
      new THREE.Vector3(-1.9, 0.05, 0.05)
    );
    const part = new THREE.Mesh(geometry);
    groupObject.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [-3.4, -0.05, -0.05], max: [-3.3, 0.05, 0.05] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(part.visible).toBe(false);

    groupObject.position.x = 2;
    groupObject.updateMatrixWorld(true);
    expect(new THREE.Box3().setFromObject(part).min.x).toBeCloseTo(0);

    reg.evaluatePerFrame();
    expect(part.visible).toBe(true);
  });

  it('refreshes a cached footprint when only the second object of a part moves', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const anchor = new THREE.Group();
    const geometry = new THREE.BufferGeometry();
    geometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(-2, -0.05, -0.05),
      new THREE.Vector3(-1.9, 0.05, 0.05)
    );
    const mover = new THREE.Mesh(geometry);
    groupObject.add(anchor, mover);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [anchor, mover],
          positionBounds: { min: [-3.4, -0.05, -0.05], max: [-3.3, 0.05, 0.05] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(mover.visible).toBe(false);

    // Only the local transform is touched: the gate is responsible for
    // refreshing each object's world matrix before comparing it.
    mover.position.x = 2;

    reg.evaluatePerFrame();
    expect(mover.visible).toBe(true);
  });

  it('re-culls a part whose refreshed footprint no longer reaches the frustum', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const geometry = new THREE.BufferGeometry();
    geometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(-0.5, -0.5, -0.5),
      new THREE.Vector3(0.5, 0.5, 0.5)
    );
    const part = new THREE.Mesh(geometry);
    groupObject.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      // Off-screen position bounds, so only the rendered footprint can keep it visible.
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
        },
      ],
    });
    reg.evaluatePerFrame();
    expect(part.visible).toBe(true);

    geometry.boundingBox.set(new THREE.Vector3(2, 2, 2), new THREE.Vector3(3, 3, 3));
    reg.invalidatePartitionFootprint('/partition/part_0');
    reg.evaluatePerFrame();

    // A recapture REPLACES the cached box; a box that only ever grew would
    // still carry the old on-screen extent and never cull.
    expect(part.visible).toBe(false);
  });

  /**
   * Two off-screen parts whose rendered footprints both move on screen without
   * a registry notification. Used to show which parts an invalidation reaches:
   * everything stays culled until the owning part is dirtied.
   */
  function makeTwoStalePartitionParts(reg: ReturnType<typeof makeRegistry>) {
    const groupObject = new THREE.Group();
    const parts = ['/partition/part_0', '/partition/part_1'].map((path) => {
      const part = new THREE.Group();
      const geometry = new THREE.BufferGeometry();
      geometry.boundingBox = new THREE.Box3(new THREE.Vector3(2, 2, 2), new THREE.Vector3(3, 3, 3));
      part.add(new THREE.Mesh(geometry));
      groupObject.add(part);
      return { path, part, geometry };
    });
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: parts.map(({ path, part }) => ({
        path,
        objects: [part],
        positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
      })),
    });
    reg.evaluatePerFrame();
    for (const { part } of parts) expect(part.visible).toBe(false);

    for (const { geometry } of parts) {
      geometry.boundingBox!.set(
        new THREE.Vector3(-0.5, -0.5, -0.5),
        new THREE.Vector3(0.5, 0.5, 0.5)
      );
    }
    return parts;
  }

  it('refreshes only the owning part after a nested geometry commit', () => {
    const reg = makeRegistry();
    const [owner, sibling] = makeTwoStalePartitionParts(reg);

    reg.evaluatePerFrame();
    expect(owner.part.visible).toBe(false);

    reg.invalidatePartitionFootprint('/partition/part_0/level_1');
    reg.evaluatePerFrame();

    expect(owner.part.visible).toBe(true);
    // The commit path is under part_0 only, so part_1 keeps its cached box.
    expect(sibling.part.visible).toBe(false);
  });

  it('refreshes every cached part when a partition commit path matches no child', () => {
    const reg = makeRegistry();
    const [first, second] = makeTwoStalePartitionParts(reg);

    reg.invalidatePartitionFootprint('/partition/unregistered-child');
    reg.evaluatePerFrame();

    expect(first.part.visible).toBe(true);
    expect(second.part.visible).toBe(true);
  });

  it('unions the rendered footprints of every object in a part', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const outside = new THREE.Group();
    const footprintGeometry = new THREE.BufferGeometry();
    footprintGeometry.boundingBox = new THREE.Box3(
      new THREE.Vector3(0.9, -0.1, -0.1),
      new THREE.Vector3(1.3, 0.1, 0.1)
    );
    const footprintVisible = new THREE.Mesh(footprintGeometry);
    groupObject.add(outside, footprintVisible);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [outside, footprintVisible],
          positionBounds: { min: [1.2, 2, 2], max: [1.3, 3, 3] },
        },
      ],
    });

    expect(reg.evaluatePerFrame()).toBe(false);
    for (const object of [outside, footprintVisible]) {
      expect(object.visible).toBe(true);
      expect(object.userData.partitionFrustumVisible).toBe(true);
    }
  });

  it('preloads a cold part just outside the exact frustum', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const child = new THREE.Group();
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [1.05, -0.1, -0.1], max: [1.2, 0.1, 0.1] },
        },
      ],
    });

    expect(reg.evaluatePerFrame()).toBe(false);
    expect(child.visible).toBe(true);
    expect(child.userData.partitionFrustumVisible).toBe(true);
  });

  it('requests a view resync when a culled part becomes visible again', () => {
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      requestReprocess
    );
    const groupObject = new THREE.Group();
    const child = new THREE.Group();
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    expect(reg.evaluatePerFrame()).toBe(true);
    expect(child.visible).toBe(false);
    expect(requestReprocess).not.toHaveBeenCalled();

    groupObject.position.x = -2.5;
    expect(reg.evaluatePerFrame()).toBe(true);
    expect(child.visible).toBe(true);
    expect(requestReprocess).toHaveBeenCalledOnce();
    // The part carries a registered node path, so the resync targets it rather
    // than re-sweeping the whole partition.
    expect(requestReprocess).toHaveBeenCalledWith(['/partition/part_0']);
  });

  it('passes the re-entering part path on the rising edge', () => {
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      requestReprocess
    );
    const groupObject = new THREE.Group();
    const returning = new THREE.Group();
    const staying = new THREE.Group();
    groupObject.add(returning, staying);
    // The RETURNING part is registered FIRST so a rising edge that leaked into
    // later children (e.g. testing the accumulated flags instead of this
    // child's) would name ``part_1`` too and fail the assertion below.
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [returning],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
        {
          path: '/partition/part_1',
          objects: [staying],
          positionBounds: { min: [-0.5, 0, 0], max: [0.5, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(staying.visible).toBe(true);
    expect(returning.visible).toBe(false);
    expect(requestReprocess).not.toHaveBeenCalled();

    // Bring ONLY the culled part back: the resync names it, not the wrapper and
    // not the part that never left.
    groupObject.position.x = -2.5;
    reg.evaluatePerFrame();
    expect(returning.visible).toBe(true);
    expect(requestReprocess).toHaveBeenCalledOnce();
    expect(requestReprocess).toHaveBeenCalledWith(['/partition/part_0']);
  });

  it('coalesces the union of re-entering parts across frames while an update is in flight', () => {
    let updateInProgress = true;
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      requestReprocess,
      () => updateInProgress
    );
    const groupObject = new THREE.Group();
    const first = new THREE.Group();
    const second = new THREE.Group();
    groupObject.add(first, second);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [first],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
        {
          path: '/partition/part_1',
          objects: [second],
          positionBounds: { min: [4, 0, 0], max: [5, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    groupObject.position.x = -2.5; // part_0 re-enters
    reg.evaluatePerFrame();
    groupObject.position.x = -4.5; // part_1 re-enters
    reg.evaluatePerFrame();
    expect(requestReprocess).not.toHaveBeenCalled();

    updateInProgress = false;
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledOnce();
    const paths = requestReprocess.mock.calls[0][0] as string[];
    expect([...paths].sort()).toEqual(['/partition/part_0', '/partition/part_1']);
  });

  it('falls back to the whole wrapper when a re-entering part has no path', () => {
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      requestReprocess
    );
    const groupObject = new THREE.Group();
    const pathed = new THREE.Group();
    const unpathed = new THREE.Group();
    groupObject.add(pathed, unpathed);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [pathed],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
        { path: '', objects: [unpathed], positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] } },
      ],
    });

    reg.evaluatePerFrame();
    groupObject.position.x = -2.5; // both re-enter
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledOnce();
    // The pathless part widens the request to the whole partition, which already
    // covers the pathed one — exactly one path, the wrapper.
    expect(requestReprocess).toHaveBeenCalledWith(['/partition']);
  });

  it('requests a view resync when any object in a part was culled', () => {
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      requestReprocess
    );
    const groupObject = new THREE.Group();
    const first = new THREE.Group();
    const second = new THREE.Group();
    groupObject.add(first, second);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [first, second],
          positionBounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
        },
      ],
    });
    first.userData.partitionFrustumVisible = false;

    expect(reg.evaluatePerFrame()).toBe(false);
    expect(first.userData.partitionFrustumVisible).toBe(true);
    expect(second.userData.partitionFrustumVisible).toBe(true);
    expect(requestReprocess).toHaveBeenCalledOnce();
  });

  it('buckets rising parts per wrapper and flushes each wrapper on its own', () => {
    let updateInProgress = true;
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      requestReprocess,
      () => updateInProgress
    );
    const firstGroup = new THREE.Group();
    const firstPart = new THREE.Group();
    firstGroup.add(firstPart);
    const secondGroup = new THREE.Group();
    const secondPart = new THREE.Group();
    secondGroup.add(secondPart);
    reg.registerPartition({
      path: '/first',
      groupObject: firstGroup,
      children: [
        {
          path: '/first/part_0',
          objects: [firstPart],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });
    reg.registerPartition({
      path: '/second',
      groupObject: secondGroup,
      children: [
        {
          path: '/second/part_0',
          objects: [secondPart],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(firstPart.visible).toBe(false);
    expect(secondPart.visible).toBe(false);

    // Both wrappers re-enter on the SAME frame, then only the first is
    // flushable: each wrapper carries ITS OWN parts, so the request names
    // ``/first``'s part alone and ``/second`` stays pending.
    firstGroup.position.x = -2.5;
    secondGroup.position.x = -2.5;
    reg.evaluatePerFrame();
    expect(requestReprocess).not.toHaveBeenCalled();

    secondGroup.visible = false;
    updateInProgress = false;
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledTimes(1);
    expect(requestReprocess).toHaveBeenLastCalledWith(['/first/part_0']);

    secondGroup.visible = true;
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledTimes(2);
    expect(requestReprocess).toHaveBeenLastCalledWith(['/second/part_0']);
  });

  it('keeps requesting frames while a rising-edge resync is pending', () => {
    const requestRender = vi.fn();
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      requestRender,
      undefined,
      undefined,
      requestReprocess,
      () => true
    );
    const groupObject = new THREE.Group();
    const child = new THREE.Group();
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    groupObject.position.x = -2.5;
    reg.evaluatePerFrame();
    reg.evaluatePerFrame();

    expect(requestReprocess).not.toHaveBeenCalled();
    expect(requestRender).toHaveBeenCalledTimes(2);
  });

  it('defers a pending resync while its partition wrapper is hidden', () => {
    let updateInProgress = true;
    const requestRender = vi.fn();
    const requestReprocess = vi.fn();
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      requestRender,
      undefined,
      undefined,
      requestReprocess,
      () => updateInProgress
    );
    const groupObject = new THREE.Group();
    const child = new THREE.Group();
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(child.visible).toBe(false);
    groupObject.position.x = -2.5;
    expect(reg.evaluatePerFrame()).toBe(true);
    expect(requestReprocess).not.toHaveBeenCalled();
    expect(requestRender).toHaveBeenCalledOnce();

    groupObject.visible = false;
    updateInProgress = false;
    requestRender.mockClear();
    expect(reg.evaluatePerFrame()).toBe(false);
    expect(requestReprocess).not.toHaveBeenCalled();
    expect(requestRender).not.toHaveBeenCalled();

    groupObject.visible = true;
    expect(reg.evaluatePerFrame()).toBe(false);
    expect(requestReprocess).toHaveBeenCalledOnce();
    expect(requestReprocess).toHaveBeenCalledWith(['/partition/part_0']);
  });

  it('unregister removes partition entries and their per-frame visibility writes', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const child = new THREE.Group();
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(child.visible).toBe(false);
    reg.unregister('/partition');
    expect(reg.evaluatePerFrame()).toBe(false);
    expect(child.visible).toBe(true);
    expect(child.userData.partitionFrustumVisible).toBeUndefined();
  });

  it('clear restores culled partition children before dropping registry state', () => {
    const reg = makeRegistry();
    const groupObject = new THREE.Group();
    const child = new THREE.Group();
    groupObject.add(child);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [child],
          positionBounds: { min: [2, 2, 2], max: [3, 3, 3] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(child.visible).toBe(false);
    reg.clear();
    expect(child.visible).toBe(true);
    expect(child.userData.partitionFrustumVisible).toBeUndefined();
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

  it('a lock on an entry with no children is inert (no per-frame throw)', () => {
    // `loadLodGroupNode` registers an empty entry on purpose, and
    // `setSelectorMode` skips clamping when there is nothing to clamp to — so
    // `children[lockLevel]` is undefined and must not be dereferenced.
    const reg = makeRegistry();
    reg.register(makeEntry([], 0, '/empty'));
    reg.setSelectorMode('/empty', { lockLevel: 2 });
    expect(() => {
      reg.evaluatePerFrame();
      reg.evaluatePerFrame();
    }).not.toThrow();
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
    // diagonal on an 800x600 viewport, whose fitted axis is min(800,600)=600
    // → coverage metric = 1000/(FILL_FACTOR·600) = 1000/300 ≈ 3.33 (the group
    // fills the screen, well past the half-fitted-axis anchor). With
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
    // The coverage metric is diagonalPx / (FILL_FACTOR·fittedAxisPx). A box that
    // fills the NDC cube projects to a diagonal of hypot(width,height) on ANY
    // viewport, so the metric is hypot(width,height)/(FILL_FACTOR·min(width,height))
    // — no longer a single aspect-independent constant (unlike the old diagonal
    // normalization, where numerator and denominator were both the viewport
    // diagonal and cancelled exactly), but it comfortably clears 1.0 on both a
    // small 4:3 viewport (measured metric ≈3.33) and a large 16:9 one (≈4.08).
    // Both a small and a large viewport must therefore pick the
    // finest coverage=1.0 child. This pins the viewport-relative normalization
    // (the whole point of switching from absolute pixels to a coverage fraction).
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
      expect(
        children[2].object.visible,
        `finest at viewport ${viewport.width}x${viewport.height}`
      ).toBe(true);
      expect(children[0].object.visible).toBe(false);
      expect(children[1].object.visible).toBe(false);
    }
  });

  it('divides by the fitted screen axis: a small box picks the middle child, not the finest, on same-aspect viewports of any size', () => {
    // Discriminating test for the normalization (the full-viewport test above
    // can't: it clears 1.0 whether or not you divide). A box spanning NDC
    // [-0.15, 0.15] on a 4:3 viewport projects to widthPx = 0.15·width,
    // heightPx = 0.15·height, so diagonalPx = 0.15·hypot(width,height); the
    // fitted axis is height (min for a 4:3, aspect >= 1, viewport), and
    // hypot(width,height)/height = hypot(4,3)/3 = 5/3 for ANY 4:3 viewport
    // (scale-invariant — depends only on aspect, not absolute size). So the
    // coverage metric = 0.15·(5/3) / FILL_FACTOR = 0.25/0.5 = 0.5 on a 4:3
    // viewport of any size (measured via projectBoxDiagonalPx below, not just
    // asserted). With thresholds [0, 0.5, 1.0] the finest applicable to 0.5 is
    // the MIDDLE child (index 1).
    // A buggy selector that skipped the ÷fittedAxisPx step would compare the
    // raw pixel diagonal (75 px on 400×300, 750 px on 4000×3000 — both ≫ 1.0)
    // and wrongly pick the finest (index 2) on both. So this pins the division
    // AND its independence from absolute viewport size (aspect-independence at
    // a FIXED aspect is exact for a fixed NDC box; independence ACROSS aspect
    // ratios is a property of real camera framing, not of an arbitrary NDC box,
    // and is pinned separately by the opening-framing invariance test below).
    const smallBox = { min: [-0.15, -0.15, -0.15], max: [0.15, 0.15, 0.15] };
    for (const viewport of [
      { width: 400, height: 300 }, // 4:3, small
      { width: 4000, height: 3000 }, // 4:3, 10x larger
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
        { ...makeChild(0), positionBounds: smallBox },
        { ...makeChild(0.5), positionBounds: smallBox },
        { ...makeChild(1.0), positionBounds: smallBox },
      ];
      reg.register(makeEntry(children, 0, '/g'));
      reg.evaluatePerFrame();
      expect(children[1].object.visible, `middle at ${viewport.width}x${viewport.height}`).toBe(
        true
      );
      expect(children[2].object.visible, `NOT finest at ${viewport.width}x${viewport.height}`).toBe(
        false
      );
      expect(children[0].object.visible).toBe(false);
    }
  });

  it("selector='screen-area': picks by the fraction of the viewport AREA occupied", () => {
    // Derived screen-area ladder [0, 1/8, 1/4, 1/2]. A box spanning NDC
    // [-0.74, 0.74] × [-0.5, 0.5] covers 0.74 × 0.5 = 37% of the screen — the
    // measured zebrahub "clearly zoomed out" pose that motivated this selector
    // (the legacy diagonal metric still read 2.37/4 there and held the finest
    // level, i.e. index 3 under these thresholds). Under screen-area, 1/4 ≤
    // 0.37 < 1/2 → level 2 of 4: one step coarser, as the user expects.
    const zebraBox = { min: [-0.74, -0.5, -0.5], max: [0.74, 0.5, 0.5] };
    const reg = makeRegistry();
    const children = [0, 0.125, 0.25, 0.5].map((t) => ({
      ...makeChild(t),
      positionBounds: zebraBox,
    }));
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(children[2].object.visible).toBe(true);
    expect(children[3].object.visible, 'finest must NOT hold at 37% occupancy').toBe(false);
  });

  it("selector='screen-area': the finest level holds while the node occupies at least half the screen", () => {
    // Area = 0.8 × 0.8 = 64% ≥ 1/2 → finest. The literal statement of the
    // occupancy-halving rule's anchor.
    const bigBox = { min: [-0.8, -0.8, -0.5], max: [0.8, 0.8, 0.5] };
    const reg = makeRegistry();
    const children = [0, 0.125, 0.25, 0.5].map((t) => ({
      ...makeChild(t),
      positionBounds: bigBox,
    }));
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(children[3].object.visible).toBe(true);
  });

  it("selector='screen-area': sizes the node from lod_bounds instead of an outlier-dominated raw AABB", () => {
    const rawBounds = { min: [-4, -4, -0.5], max: [4, 4, 0.5] };
    const robustBounds = { min: [-0.4, -0.4, -0.5], max: [0.4, 0.4, 0.5] };
    const rawReg = makeRegistry();
    const rawChildren = [0, 0.25, 0.5].map((threshold) => ({
      ...makeChild(threshold),
      positionBounds: rawBounds,
    }));
    const rawEntry = makeEntry(rawChildren, 0, '/raw');
    rawEntry.selector = 'screen-area';
    rawReg.register(rawEntry);
    rawReg.evaluatePerFrame();
    expect(rawChildren.map((child) => child.object.visible)).toEqual([false, false, true]);

    const reg = makeRegistry();
    const children = [0, 0.25, 0.5].map((threshold) =>
      withLodBounds({ ...makeChild(threshold), positionBounds: rawBounds }, robustBounds)
    );
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);

    reg.evaluatePerFrame();

    expect(children[0].object.visible, 'robust box occupies only 16% of the viewport').toBe(true);
    expect(children[2].object.visible, 'raw AABB would clip to full-screen and pick finest').toBe(
      false
    );
  });

  it("selector='screen-area': contained lod_bounds cannot select finer than position_bounds", () => {
    const rawBounds = { min: [-0.5, -0.0021, -0.5], max: [0.5, 0.0021, 0.5] };
    const robustBounds = { min: [-0.4, -0.0004, -0.5], max: [0.4, 0.0004, 0.5] };
    const reg = makeRegistry();
    const children = [0, 0.1].map((threshold) =>
      withLodBounds({ ...makeChild(threshold), positionBounds: rawBounds }, robustBounds)
    );
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);

    reg.evaluatePerFrame();

    expect(children[0].object.visible, 'raw metric is 0.00105, below the fine threshold').toBe(
      true
    );
    expect(
      children[1].object.visible,
      'the thin-box ramp makes the robust metric 0.24 unless it is clamped to raw'
    ).toBe(false);
  });

  it("selector='coverage': sizes the node from lod_bounds instead of an outlier-dominated raw AABB", () => {
    const rawBounds = { min: [-4, -4, -0.5], max: [4, 4, 0.5] };
    const robustBounds = { min: [-0.05, -0.05, -0.5], max: [0.05, 0.05, 0.5] };
    const rawReg = makeRegistry();
    const rawChildren = [0, 0.5].map((threshold) => ({
      ...makeChild(threshold),
      positionBounds: rawBounds,
    }));
    rawReg.register(makeEntry(rawChildren, 0, '/raw'));
    rawReg.evaluatePerFrame();
    expect(rawChildren.map((child) => child.object.visible)).toEqual([false, true]);

    const reg = makeRegistry();
    const children = [0, 0.5].map((threshold) =>
      withLodBounds({ ...makeChild(threshold), positionBounds: rawBounds }, robustBounds)
    );
    reg.register(makeEntry(children, 0, '/g'));

    reg.evaluatePerFrame();

    expect(children[0].object.visible, 'robust diagonal stays below the fine threshold').toBe(true);
    expect(children[1].object.visible, 'raw AABB would saturate the legacy metric').toBe(false);
  });

  it('does not fold a second world box when no child publishes lod_bounds', () => {
    const reg = makeRegistry();
    const entry = makeEntry([makeChild(0), makeChild(0.5)], 0, '/g');
    const updateWorldMatrix = vi.spyOn(entry.groupObject, 'updateWorldMatrix');
    reg.register(entry);

    reg.evaluatePerFrame();

    expect(updateWorldMatrix).toHaveBeenCalledTimes(1);
  });

  it("selector='screen-area' is viewport-size independent (same pick on any monitor)", () => {
    // The metric is built from NDC fractions, so pixel dimensions must not
    // matter. Same 37%-occupancy box, tiny and 4K viewports → same level.
    const zebraBox = { min: [-0.74, -0.5, -0.5], max: [0.74, 0.5, 0.5] };
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
      const children = [0, 0.125, 0.25, 0.5].map((t) => ({
        ...makeChild(t),
        positionBounds: zebraBox,
      }));
      const entry = makeEntry(children, 0, '/g');
      entry.selector = 'screen-area';
      reg.register(entry);
      reg.evaluatePerFrame();
      expect(children[2].object.visible, `level 2 at ${viewport.width}x${viewport.height}`).toBe(
        true
      );
    }
  });

  it("selector='screen-area': a degenerate (zero-thickness) node spanning the screen picks the finest, not the coarsest", () => {
    // Regression for the review-caught failure: an axis-aligned straight
    // polyline projects to a zero-height rect, whose raw area product is 0 —
    // permanently coarsest under a naive area metric even at full screen
    // width. The degenerate fallback reads the linear span (1.0 here ≥ the
    // 0.5 finest threshold) → finest.
    const lineBox = { min: [-1, 0, 0], max: [1, 0, 0] };
    const reg = makeRegistry();
    const children = [0, 0.125, 0.25, 0.5].map((t) => ({
      ...makeChild(t),
      positionBounds: lineBox,
    }));
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(children[3].object.visible).toBe(true);
    expect(children[0].object.visible, 'must NOT be pinned to the coarsest').toBe(false);
  });

  it("selector='screen-area': a fitted high-aspect object reads its literal occupancy (one level below finest) — BY DESIGN", () => {
    // The occupancy rule applied verbatim, pinning the DELIBERATE revision of
    // the old diagonal-anchored opening-framing guarantee: a fitted full-width
    // but quarter-height object occupies 25% of the screen, so on the standard
    // 4-level derived ladder [0, ⅛, ¼, ½] it opens at the SECOND-FINEST level
    // (0.25 sits exactly on that threshold — thresholds are inclusive) with
    // full detail one modest zoom away. Under the retired diagonal metric the
    // same rod read ≈ its LENGTH and pinned the finest level — exactly how
    // dense sub-pixel elongated content rendered its most expensive level
    // across the whole zoom range. NOT the degenerate ramp's territory: the
    // rod is 0.25 half-extents thick, far above the sub-pixel floor.
    const rodBox = { min: [-1, -0.25, -0.1], max: [1, 0.25, 0.1] };
    const reg = makeRegistry();
    const children = [0, 0.125, 0.25, 0.5].map((t) => ({
      ...makeChild(t),
      positionBounds: rodBox,
    }));
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(children[2].object.visible, 'second-finest at 25% occupancy').toBe(true);
    expect(children[3].object.visible, 'finest requires ≥ half-screen occupancy').toBe(false);
    expect(children[0].object.visible, 'NOT pinned to the coarsest').toBe(false);
  });

  it("selector='screen-area': a two-level ladder holds the coarsest for a fitted 25%-occupancy object (the documented degenerate-config case)", () => {
    // With only [0, 0.5] there is no intermediate level for the rod's 25%
    // occupancy to land on — it stays coarse until half-screen. Pinned as
    // INTENDED: two-level ladders trade granularity away everywhere, and the
    // per-level halving that would catch this needs levels to halve onto.
    const rodBox = { min: [-1, -0.25, -0.1], max: [1, 0.25, 0.1] };
    const reg = makeRegistry();
    const children = [0, 0.5].map((t) => ({ ...makeChild(t), positionBounds: rodBox }));
    const entry = makeEntry(children, 0, '/g');
    entry.selector = 'screen-area';
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true);
  });

  it('an entry WITHOUT a selector keeps the legacy diagonal metric', () => {
    // The zebrahub-pose box under the LEGACY metric: projected diagonal =
    // hypot(0.74·800, 0.5·600) = hypot(592, 300) ≈ 663.7 px on 800×600
    // (viewport diagonal 1000) → metric ≈ 663.7/250 ≈ 2.65 ≥ threshold 0.5·…
    // — with the same [0, 0.125, 0.25, 0.5] thresholds every level qualifies,
    // so the FINEST is picked. This is exactly the mis-selection the
    // screen-area selector fixes; pinning it here proves the two entries
    // genuinely take different code paths (a selector-ignoring mutant would
    // make this and the 37% test disagree).
    const zebraBox = { min: [-0.74, -0.5, -0.5], max: [0.74, 0.5, 0.5] };
    const reg = makeRegistry();
    const children = [0, 0.125, 0.25, 0.5].map((t) => ({
      ...makeChild(t),
      positionBounds: zebraBox,
    }));
    reg.register(makeEntry(children, 0, '/g')); // no entry.selector
    reg.evaluatePerFrame();
    expect(children[3].object.visible).toBe(true);
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

// ────────────────────────────────────────────────────────────────────────
// Opening-framing anchor (FILL_FACTOR) — issues #1361 and #1410
//
// The finest level must already be selected at the DEFAULT opening framing,
// not only once the object overfills the screen (#1361), and that must hold at
// EVERY viewport aspect ratio, not just near-square ones (#1410). These tests
// drive the real fit math end to end — `calculateCameraDistance` (fitRatio
// 0.75, exact near-face depth term) at the default fov 47, then the real
// `projectBoxDiagonalPx` — instead of hard-coding a metric, so they pin the
// anchor against the framing code that actually produces it.
//
// Every "finest at the opening framing" assertion below FAILS at FILL_FACTOR
// 1.0: the raw `diagonalPx / fittedAxisPx` ratio of a default framing is only
// 0.750–1.061 across `SHAPES` × `VIEWPORTS` (asserted explicitly, which is also
// the guard that the projection is NOT saturating to +Infinity), so the
// WORST-case shape's finest threshold of 1.0 is not reached without the ÷0.5.
// It would ALSO fail under the pre-#1410 diagonal normalisation at aspect
// ratios beyond 16:9-ish (measured there: 0.31–0.86 near 1:1/16:9/9:16, but
// falling toward 0 as the canvas widens — see the invariance test below for
// why the fitted-axis denominator fixes that).
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — opening-framing anchor (FILL_FACTOR)', () => {
  /** The viewer's default field of view (`config.renderingControls.defaults.fov`). */
  const DEFAULT_FOV = 47;

  /**
   * A stamped 4-level substitutive ladder (values as a pre-halving LEGACY
   * store wrote them, consumed under `selector: 'coverage'` — the registry
   * uses whatever thresholds are stamped, so this fixture stays valid for
   * old datasets; new stores derive the screen-area occupancy-halving
   * whole-object ladder [0, 0.125, 0.25, 0.5] under `selector:
   * 'screen-area'`).
   */
  const SUBSTITUTIVE_LADDER = [0, 0.125, 0.35355, 1.0];

  /** Axis-aligned bounds of a `w × h × d` box centred on the origin. */
  function centredBounds(w: number, h: number, d: number) {
    return { min: [-w / 2, -h / 2, -d / 2], max: [w / 2, h / 2, d / 2] };
  }

  function toBox(bounds: { min: number[]; max: number[] }): BoundingBox {
    return {
      min: { x: bounds.min[0], y: bounds.min[1], z: bounds.min[2] },
      max: { x: bounds.max[0], y: bounds.max[1], z: bounds.max[2] },
    };
  }

  /**
   * The camera the viewer itself would place for a default "fit to bounds"
   * framing: distance straight from `calculateCameraDistance`, positioned on
   * +Z of the box centre and looking at it. `distanceScale` pulls the camera
   * further back (1 = the opening framing).
   */
  function framedCamera(
    bounds: { min: number[]; max: number[] },
    viewport: { width: number; height: number },
    distanceScale = 1
  ): THREE.PerspectiveCamera {
    const aspect = viewport.width / viewport.height;
    const box = toBox(bounds);
    const near = 0.1;
    const far = 1e6;
    const distance =
      calculateCameraDistance(box, { fov: DEFAULT_FOV, aspect, near, far }) * distanceScale;
    const cam = new THREE.PerspectiveCamera(DEFAULT_FOV, aspect, near, far);
    const cx = (box.min.x + box.max.x) / 2;
    const cy = (box.min.y + box.max.y) / 2;
    const cz = (box.min.z + box.max.z) / 2;
    cam.position.set(cx, cy, cz + distance);
    cam.lookAt(cx, cy, cz);
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true); // THREE.Camera also refreshes matrixWorldInverse
    return cam;
  }

  function registryFor(camera: THREE.Camera, viewport: { width: number; height: number }) {
    return new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => viewport,
      getDisplayDims: () => [0, 1, 2],
    });
  }

  /** One child per rung of `SUBSTITUTIVE_LADDER`, all sharing the group bounds. */
  function ladderChildren(bounds: { min: number[]; max: number[] }): LODGroupChild[] {
    return SUBSTITUTIVE_LADDER.map((cf) => ({ ...makeChild(cf), positionBounds: bounds }));
  }

  /**
   * `diagonalPx / fittedAxisPx` — the coverage metric at FILL_FACTOR 1.0.
   * `fittedAxisPx` is `min(viewport.width, viewport.height)`, the extent
   * `calculateCameraDistance` actually fits (see the `FILL_FACTOR` doc in
   * `lod-group-registry.ts`) — the #1410 fix's normalisation.
   */
  function rawCoverageRatio(
    bounds: { min: number[]; max: number[] },
    camera: THREE.Camera,
    viewport: { width: number; height: number }
  ): number {
    const diagonalPx = projectBoxDiagonalPx(toBox(bounds), camera, viewport);
    return diagonalPx / Math.min(viewport.width, viewport.height);
  }

  // Shapes that bracket what real scenes look like, including the #1543
  // regression: a cloud elongated along the view axis.
  const SHAPES: ReadonlyArray<{ name: string; bounds: { min: number[]; max: number[] } }> = [
    { name: 'cube 100×100×100', bounds: centredBounds(100, 100, 100) },
    { name: 'pancake 100×100×1', bounds: centredBounds(100, 100, 1) },
    { name: 'in-plane elongated 100×1×1', bounds: centredBounds(100, 1, 1) },
    { name: 'umap-ish 100×80×60', bounds: centredBounds(100, 80, 60) },
    { name: 'view-axis elongated 1×1×100', bounds: centredBounds(1, 1, 100) },
  ];

  // The full aspect matrix from issue #1410: 1:1, 16:9, 9:16, 21:9 and 32:9.
  // Under the OLD (viewport-diagonal) normalisation these last two ("ultrawide")
  // aspects used to live in a separate table because the anchor drifted there
  // (a shape's raw ratio fell as `raw(1)·√2 / hypot(aspect, 1)` for aspect >= 1)
  // — see the invariance test below for the fix's root-cause pin. They are
  // folded into the same matrix here because the fitted-axis normalisation no
  // longer treats them specially: every shape reaches the finest level at every
  // aspect below.
  const VIEWPORTS: ReadonlyArray<{ name: string; width: number; height: number }> = [
    { name: 'square 1:1 1200×1200', width: 1200, height: 1200 },
    { name: 'landscape 16:9 1600×900', width: 1600, height: 900 },
    { name: 'portrait 9:16 900×1600', width: 900, height: 1600 },
    { name: 'ultrawide 21:9 2560×1080', width: 2560, height: 1080 },
    { name: 'super-ultrawide 32:9 3840×1080', width: 3840, height: 1080 },
    // Extreme-portrait rows prove the exact fitted-axis invariant well beyond
    // ordinary browser shapes.
    { name: 'extreme-portrait 9:32 900×3200', width: 900, height: 3200 },
    { name: 'extreme-portrait 1:4 800×3200', width: 800, height: 3200 },
  ];

  for (const shape of SHAPES) {
    for (const viewport of VIEWPORTS) {
      it(`selects the FINEST level at the default opening framing (${shape.name}, ${viewport.name})`, () => {
        const camera = framedCamera(shape.bounds, viewport);
        const reg = registryFor(camera, viewport);
        const children = ladderChildren(shape.bounds);
        reg.register(makeEntry(children, 0, '/g')); // starts on the coarsest
        reg.evaluatePerFrame();
        expect(
          children[children.length - 1].object.visible,
          `finest for ${shape.name} on ${viewport.name}`
        ).toBe(true);
        for (let i = 0; i < children.length - 1; i++) {
          expect(children[i].object.visible, `coarse level ${i} hidden`).toBe(false);
        }
      });
    }
  }

  it('the opening framing is a FINITE sub-viewport projection, not a saturated +Infinity', () => {
    // Guard against passing for the wrong reason: `projectBoxDiagonalPx` returns
    // +Infinity when the camera is inside/straddling the box, which would select
    // the finest level at ANY fill factor. The fit distance always puts the
    // camera outside, so every raw ratio here is finite. Unlike the OLD
    // diagonal-normalised ratio (always < 1 by construction, since a sub-viewport
    // box can never project past the viewport's own diagonal), the fitted-axis
    // ratio can itself exceed 1 for a shape whose cross-section isn't much
    // smaller than the fitted axis — measured range across all 5 shapes and all
    // 7 aspects in `VIEWPORTS` (including the two extreme-portrait rows):
    // 0.750 (in-plane rod, worst case) … 1.061 (cube, pancake, and view-axis
    // rod, best case). The finest-level anchor (FILL_FACTOR)
    // still does real work for the worst case — see the next test.
    for (const shape of SHAPES) {
      for (const viewport of VIEWPORTS) {
        const camera = framedCamera(shape.bounds, viewport);
        const ratio = rawCoverageRatio(shape.bounds, camera, viewport);
        const label = `${shape.name} on ${viewport.name}`;
        expect(Number.isFinite(ratio), `finite for ${label}`).toBe(true);
        expect(ratio, `at least the measured worst case for ${label}`).toBeGreaterThanOrEqual(0.75);
        expect(ratio, `at most the measured best case for ${label}`).toBeLessThan(1.07);
      }
    }
  });

  it('FILL_FACTOR is exactly what turns those sub-viewport ratios into a finest-level metric', () => {
    // Reads the REAL exported constant rather than hard-coding 0.5, so this
    // fails if the anchor moves: at 1.0, the measured opening ratios (see above)
    // would leave the WORST case (the in-plane rod, ratio 0.750) short of the
    // finest threshold of 1.0 — reproducing #1361's blur. FILL_FACTOR = 0.5 is
    // what turns that same worst-case ratio into a metric of 1.50, clearing the
    // rung with 50% headroom.
    const finestThreshold = SUBSTITUTIVE_LADDER[SUBSTITUTIVE_LADDER.length - 1];
    let worstCaseRatio = Infinity;
    for (const shape of SHAPES) {
      for (const viewport of VIEWPORTS) {
        const camera = framedCamera(shape.bounds, viewport);
        const ratio = rawCoverageRatio(shape.bounds, camera, viewport);
        const label = `${shape.name} on ${viewport.name}`;
        expect(ratio / FILL_FACTOR, `metric clears the finest rung for ${label}`).toBeGreaterThan(
          finestThreshold
        );
        worstCaseRatio = Math.min(worstCaseRatio, ratio);
      }
    }
    // The division is doing real work for the worst case specifically — at
    // FILL_FACTOR 1.0 the worst-case ratio alone would be short of the rung
    // (this is the assertion FILL_FACTOR = 1.0 fails, unlike the metric above).
    expect(worstCaseRatio, 'the worst-case raw ratio alone is short of the rung').toBeLessThan(
      finestThreshold
    );
  });

  it('MAX_COVERAGE_FRACTION (Python) equals SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR — the screen-filling metric', () => {
    // The cross-language coupling, pinned from this side too (a Python test reads
    // this file and asserts the same relation). Unlike the pre-#1410 scheme, a
    // screen-filling object's raw ratio is no longer exactly 1.0 (that identity
    // only held for the old diagonal normalisation) — SCREEN_FILL_DIAGONAL_RATIO
    // is the multiple of the fitted axis a screen-filling diagonal actually
    // measures. That multiple is aspect-dependent (1.41 at 1:1, 2.57 at 21:9);
    // the constant is anchored at 16:9, where it is 2.04. See the
    // `FILL_FACTOR` doc.
    expect(SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR).toBeCloseTo(4.0, 10);
  });

  it('the raw fitted-axis ratio is invariant across viewport aspect ratio', () => {
    // THE root-cause fix. Under the OLD (viewport-diagonal) normalisation a
    // shape's raw ratio fell off sharply with aspect
    // (`raw(aspect) = raw(1)·√2 / hypot(aspect, 1)` for aspect >= 1), so a shape
    // could drop below the finest threshold on a sufficiently wide monitor
    // (#1361's blur, returning at wide aspects). The fitted-axis ratio does not
    // have that problem:
    //
    // The exact near-face fit added for #1543 makes the portrait branch exact
    // too: the half-depth term no longer changes relative to the projected
    // screen-plane extent as aspect changes.
    expect(VIEWPORTS.length).toBeGreaterThanOrEqual(7);
    expect(
      VIEWPORTS.filter((viewport) => viewport.width < viewport.height).length
    ).toBeGreaterThanOrEqual(3);
    for (const shape of SHAPES) {
      const ratios = VIEWPORTS.map((viewport) =>
        rawCoverageRatio(shape.bounds, framedCamera(shape.bounds, viewport), viewport)
      );
      const reference = ratios[0];
      for (let i = 1; i < ratios.length; i++) {
        expect(
          ratios[i],
          `${shape.name}: ${VIEWPORTS[i].name} vs ${VIEWPORTS[0].name}`
        ).toBeCloseTo(reference, 9);
      }
    }
  });

  it('frames a view-axis rod on the finest rung instead of the coarsest', () => {
    const viewport = { width: 1600, height: 900 };
    const bounds = centredBounds(1, 1, 100);
    const camera = framedCamera(bounds, viewport);
    const metric = rawCoverageRatio(bounds, camera, viewport) / FILL_FACTOR;

    expect(metric).toBeCloseTo(2.1213, 4);
    expect(metric).toBeGreaterThan(SUBSTITUTIVE_LADDER.at(-1)!);
  });

  it('pins the resize-without-a-re-fit asymmetry documented on FILL_FACTOR', () => {
    // The flip side of the fix, and the one case where the new denominator can
    // move MORE than the old one. `updateCameraAspect` (utils/camera-utils.ts)
    // only touches `camera.aspect` on a window resize — it preserves the
    // vertical fov and never re-fits the distance — so with HEIGHT held fixed
    // the projected pixel diagonal is completely unchanged (the viewport-width
    // term cancels out of the NDC→pixel conversion), while the denominator
    // keeps shrinking once width < height.
    //
    // This test exists so those claims in the `FILL_FACTOR` doc stay honest:
    // adding a camera re-fit on resize (the real fix, deliberately out of scope
    // here) is supposed to break it, at which point the doc gets updated too.
    const height = 900;
    const baseline = { width: 1600, height };
    const bounds = centredBounds(100, 100, 100);
    // ONE camera, framed for the baseline viewport, reused at every width —
    // this is a resize, not a re-fit.
    const camera = framedCamera(bounds, baseline);
    const box = toBox(bounds);

    /** What the viewer itself does on a resize — and all it does. */
    const diagonalAt = (width: number) => {
      updateCameraAspect(camera, width, height);
      return projectBoxDiagonalPx(box, camera, { width, height });
    };
    const newMetric = (width: number) =>
      diagonalAt(width) / (FILL_FACTOR * Math.min(width, height));
    // The pre-#1410 denominator, for the comparison the doc block records.
    const oldMetric = (width: number) => diagonalAt(width) / (0.25 * Math.hypot(width, height));

    const baseDiagonal = diagonalAt(baseline.width);
    const inflation = (width: number) => ({
      now: newMetric(width) / newMetric(baseline.width),
      before: oldMetric(width) / oldMetric(baseline.width),
    });

    // (1) Narrowing the window does not change the projected diagonal at all.
    for (const width of [900, 506, 500, 200]) {
      expect(diagonalAt(width), `diagonal unchanged at ${width}×${height}`).toBeCloseTo(
        baseDiagonal,
        9
      );
    }

    // (2) Down to the crossover at `height² / width0` (506px here, aspect
    // ≈0.56) the NEW normalisation is the better-behaved of the two: while the
    // viewport is still landscape the metric does not move at all, where the
    // diagonal denominator already inflated it.
    expect(inflation(900).now).toBeCloseTo(1.0, 9); // square: exactly stable now…
    expect(inflation(900).before).toBeCloseTo(1.442, 3); // …but ×1.44 before
    const crossover = inflation(506);
    expect(crossover.now).toBeCloseTo(crossover.before, 2);

    // (3) Past it, narrowing inflates faster than it used to — `min(w, h)` is
    // unbounded below where `hypot(w, h)` floors at `height`.
    expect(inflation(500).now).toBeCloseTo(1.8, 3);
    expect(inflation(500).before).toBeCloseTo(1.783, 3);
    expect(inflation(200).now).toBeCloseTo(4.5, 3);
    expect(inflation(200).before).toBeCloseTo(1.991, 3);

    // (4) …and what that means through the real selector, not just the
    // arithmetic above: a zoomed-out cube sitting on level 2 at the baseline
    // (metric ≈0.47) is pushed onto the finest level by an extreme narrowing
    // alone (≈2.13), with the camera never moving.
    const zoomedOut = framedCamera(bounds, baseline, 4);
    for (const [viewport, expected] of [
      [baseline, 2],
      [{ width: 200, height }, 3],
    ] as const) {
      updateCameraAspect(zoomedOut, viewport.width, viewport.height);
      const reg = registryFor(zoomedOut, viewport);
      const children = ladderChildren(bounds);
      reg.register(makeEntry(children, 3, '/g'));
      reg.evaluatePerFrame();
      expect(
        children[expected].object.visible,
        `level ${expected} at ${viewport.width}×${viewport.height}`
      ).toBe(true);
    }
  });

  it('a substantially zoomed-out view falls back to a coarser level', () => {
    // Cube at 4× the opening distance. Projected diagonal shrinks roughly as
    // 1/distance, so the raw ratio drops 1.061 → ≈0.213 and the coverage metric
    // 2.12 → ≈0.426: past level 3's downgrade band (hysteresis edge at
    // 1.0 − 0.1·(1.0 − 0.354) = 0.935, so ≈0.426 is comfortably clear) and into
    // level 2's range [0.354, 1.0).
    const viewport = { width: 1600, height: 900 };
    const bounds = centredBounds(100, 100, 100);
    const camera = framedCamera(bounds, viewport, 4);
    const metric = rawCoverageRatio(bounds, camera, viewport) / FILL_FACTOR;
    expect(metric).toBeGreaterThan(SUBSTITUTIVE_LADDER[2]);
    expect(metric).toBeLessThan(0.9); // clear of the downgrade hysteresis band
    const reg = registryFor(camera, viewport);
    const children = ladderChildren(bounds);
    reg.register(makeEntry(children, 3, '/g')); // was on the finest
    reg.evaluatePerFrame();
    expect(children[2].object.visible).toBe(true);
    expect(children[3].object.visible).toBe(false);
  });

  it('a far-away view falls all the way back to the coarsest level', () => {
    // 40× the opening distance → metric ≈ 0.0402, below every threshold but 0.
    const viewport = { width: 1600, height: 900 };
    const bounds = centredBounds(100, 100, 100);
    const camera = framedCamera(bounds, viewport, 40);
    const metric = rawCoverageRatio(bounds, camera, viewport) / FILL_FACTOR;
    expect(metric).toBeLessThan(SUBSTITUTIVE_LADDER[1]);
    const reg = registryFor(camera, viewport);
    const children = ladderChildren(bounds);
    reg.register(makeEntry(children, 3, '/g'));
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true);
    expect(children[3].object.visible).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// coverage_fraction thresholds ABOVE 1.0 (the tiled-layer escape hatch)
//
// 1.0 is only the AUTO-DERIVED ladder's finest anchor (half the fitted screen
// axis). An explicitly authored ladder may go up to
// `SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR` == 4.0
// — roughly the metric a screen-filling object produces (exactly so only near
// aspect ratio √3) — so a spatially tiled layer,
// whose tiles each project to a fraction of the viewport, can still hold a
// coarse level at whole-scene framing. Python's `MAX_COVERAGE_FRACTION` bounds
// the authored list; the viewer deliberately enforces NO upper bound, and these
// tests pin that the selector honours such thresholds rather than saturating.
// (This is what `demo_biodiversity_planetary_scale`'s OCCURRENCE_COVERAGE
// = (0.0, 3.52, 3.84, 4.0) relies on.)
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — coverage_fraction thresholds above 1.0', () => {
  // The demo's authored ladder, in coverage-METRIC space: its diagonal-era raw
  // fractions 0.0/0.88/0.96/1.0 taken ×4 (÷ the 0.25 `FILL_FACTOR` of the day).
  // Today's two-step conversion of the same fractions is ×2.0398/0.5 = ×4.08,
  // so these still land within ~2% of their intended switch points.
  const TILED_LADDER = [0.0, 3.52, 3.84, 4.0];

  /**
   * Identity-NDC camera (as in the other registry tests) plus a box spanning
   * NDC ±`half`. On an 800×600 viewport the projected diagonal is
   * `2·half·hypot(800,600)/2 = half·1000` px and the fitted axis is
   * `min(800,600) = 600`, so the RAW fraction is `half·1000/600 = half·(5/3)`
   * and the coverage metric is `half·(5/3)/FILL_FACTOR = half·(10/3)` (verified
   * against the real `projectBoxDiagonalPx`, not just hand-derived).
   */
  function registryAtRawFraction(half: number, children: LODGroupChild[]) {
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    const bounds = { min: [-half, -half, -half], max: [half, half, half] };
    for (const c of children) c.positionBounds = bounds;
    return new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
    });
  }

  it('holds a tiled layer at its coarsest level when the tile is only ~0.6 of the viewport', () => {
    // NOT a re-derivation of the demo's own measurement — `half` here is a
    // synthetic NDC half-extent on THIS test's 800×600 (4:3) viewport (see
    // `registryAtRawFraction`'s docstring), unrelated in units to the demo's
    // raw diagonal fraction. half=0.6 → RAW fraction 0.6·(5/3) = 1.0 → metric
    // 0.6·(10/3) = 2.0, comfortably below the 3.52 threshold, so the coarsest
    // level renders — pinning the >1.0-ceiling machinery in isolation. (For
    // scale, not equivalence: the demo's own measured whole-globe framing was
    // ~0.60 OF THE VIEWPORT DIAGONAL — pre-#1410 units — which converts to a
    // metric of ~2.45 at a 16:9 viewport, a different number on a different
    // viewport shape than this synthetic test's 2.0.) Under the old [0, 1]
    // ceiling the ladder could not have expressed values above 1.0 at all,
    // and every tile would have jumped to its finest level (the
    // 18.1M-resident blow-up).
    const children = TILED_LADDER.map((cf) => makeChild(cf));
    const reg = registryAtRawFraction(0.6, children);
    reg.register(makeEntry(children, 3, '/tile')); // start on the finest
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true);
    for (let i = 1; i < children.length; i++) {
      expect(children[i].object.visible, `level ${i} hidden`).toBe(false);
    }
  });

  it('still reaches the finest level once the tile substantially overfills the viewport', () => {
    // On THIS 800×600 (4:3) viewport, half=1.0 is what actually "fills the
    // viewport" (the box spans NDC ±1, matching the screen edges exactly):
    // metric = 1.0·(10/3) ≈ 3.33 — matching the real 4:3 screen-fill metric
    // `2·hypot(4/3, 1)/min(4/3, 1)` ≈ 3.33 computed independently in the
    // `FILL_FACTOR` doc — which only reaches TILED_LADDER's level 2 (3.84
    // threshold), not the top. half=1.2 here is a further 20% LINEAR
    // OVERFILL beyond that: metric 1.2·(10/3) = 4.0, exactly the finest
    // threshold (inclusive — the natural pick uses `threshold <= metric`).
    // The >1 thresholds must be reachable, not dead weight.
    const children = TILED_LADDER.map((cf) => makeChild(cf));
    const reg = registryAtRawFraction(1.2, children);
    reg.register(makeEntry(children, 0, '/tile'));
    reg.evaluatePerFrame();
    expect(children[3].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
  });

  it('walks the intermediate levels of an above-1.0 ladder', () => {
    // half=1.1 → metric 1.1·(10/3) ≈ 3.667: past 3.52, short of 3.84 → level 1.
    const children = TILED_LADDER.map((cf) => makeChild(cf));
    const reg = registryAtRawFraction(1.1, children);
    reg.register(makeEntry(children, 0, '/tile'));
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[2].object.visible).toBe(false);
    expect(children[3].object.visible).toBe(false);
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

  it('a never-shown level ages as the COLDEST in its group, not the hottest', () => {
    // Back-filling lastVisibleTick with the CURRENT tick made the one level the
    // user never saw the most-recently-used: under budget pressure the registry
    // evicted a level shown moments ago and kept the stranded one (#2634).
    const relShown = vi.fn();
    const relStranded = vi.fn();
    const children = [
      readyLazy(0, { tick: 1 }), // locked + displayed → never evicted
      readyLazy(0.5, { release: relShown, tick: 15 }), // shown at tick 15
      { ...readyLazy(1.0, { release: relStranded }), ready: false }, // still loading
    ];
    // Two resident levels fit the budget, three do not (100 per ready level).
    const reg = makeRegistry([0, 1, 2], 250, residentModel(children));
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 });
    for (let i = 0; i < 20; i++) reg.evaluatePerFrame(); // tick → 20, under budget
    expect(relShown).not.toHaveBeenCalled();

    children[2].ready = true; // the stranded load lands at tick 21, never displayed
    reg.evaluatePerFrame();
    expect(children[2].lastVisibleTick).toBe(0);
    expect(relStranded).toHaveBeenCalledTimes(1); // coldest → evicted first
    expect(relShown).not.toHaveBeenCalled();
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

  it('pauses automatic lazy loads during an archive fault and resumes on the next frame', () => {
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    children[1].nodePath = '/g/child_1';
    let hasArchiveFault = true;
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      () => hasArchiveFault
    );
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });

    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();

    hasArchiveFault = false;
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
  });

  it('does not auto-retry a permanently failed child after the loader fault clears', () => {
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    children[1].failed = true;
    children[1].permanentlyFailed = true;
    const reg = makeRegistry();
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });

    for (let i = 0; i < 300; i++) reg.evaluatePerFrame();

    expect(ensureLoaded).not.toHaveBeenCalled();
    expect(children[1].failed).toBe(true);
    expect(children[1].failedTick).toBeUndefined();
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
    // The DISPLAYED level is stamped with the tick (reset → first frame == 1);
    // the level that was never shown gets the coldest sentinel, 0.
    const shown = children.find((c) => c.object.visible)!;
    const hidden = children.find((c) => !c.object.visible)!;
    expect(shown.lastVisibleTick).toBe(1);
    expect(hidden.lastVisibleTick).toBe(0);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Hidden-layer load gate (effective, ancestor-aware visibility).
//
// A scene can author a layer `visible=false` (the layers panel applies the
// flag after load, and the eye toggle flips it live). That hides the LAYER
// object; the lod_group and its levels underneath keep their own `visible`
// flags, so the selector used to keep aspiring to — and lazily loading — fine
// levels that cannot be drawn, competing for the shared fetch gate, the worker
// pool and VRAM with the layer the user is looking at. The gate must stop
// STARTING those loads while any ancestor is hidden, and resume on the very
// next frame once it is shown.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — hidden-layer load gate', () => {
  /**
   * Put `entry.groupObject` under a layer group (what the scene graph looks
   * like: the hidden flag sits on an ANCESTOR, not on the lod_group itself)
   * and return the layer so a test can toggle it.
   */
  function withLayerParent(entry: LODGroupEntry, layerVisible: boolean): THREE.Group {
    const layer = new THREE.Group();
    layer.visible = layerVisible;
    layer.add(entry.groupObject);
    return layer;
  }

  it('does not fire ensureLoaded for a level under a hidden ancestor layer', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    const entry = makeEntry(children, 0, '/g');
    withLayerParent(entry, false); // layer authored visible=false
    reg.register(entry);
    reg.setSelectorMode('/g', { lockLevel: 1 }); // desired = the lazy fine level

    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
    // The gate must not strand `loading` either — that flag is what would
    // block the load forever once the layer is shown again.
    expect(children[1].loading).not.toBe(true);
  });

  it('fires ensureLoaded on the next frame once the ancestor is made visible', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    const entry = makeEntry(children, 0, '/g');
    const layer = withLayerParent(entry, false);
    reg.register(entry);
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();

    // What the layers panel's eye toggle does (LayerApplyEngine.applyVisibility),
    // followed by its requestRender → next per-frame evaluation.
    layer.visible = true;
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
  });

  it('gates a directly-hidden lod_group node too (not only an ancestor)', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    const entry = makeEntry(children, 0, '/g');
    entry.groupObject.visible = false; // the lod_group itself is the hidden node
    reg.register(entry);
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  it('loads normally under a VISIBLE layer (regression guard)', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const children = [makeChild(0), makeLazyChild(0.5, ensureLoaded)];
    const entry = makeEntry(children, 0, '/g');
    withLayerParent(entry, true); // ordinary visible layer
    reg.register(entry);
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
  });

  it('does not reload a stale fine level under a hidden layer (settled scrub)', () => {
    // The reload path (maybeKickReload) shares the gate: a hidden layer must
    // not re-fetch its fine level for every settled slice change either.
    const ensureLoaded = vi.fn();
    const stale = makeGsplatChild(0.5, 1); // committed for version 1
    stale.ready = true;
    stale.ensureLoaded = ensureLoaded;
    const children = [makeGsplatChild(0, 2), stale];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2); // version fixed at 2
    const entry = makeEntry(children, 1, '/g'); // aspiration = the stale fine level
    withLayerParent(entry, false);
    reg.register(entry);

    for (let i = 0; i < 14; i++) reg.evaluatePerFrame(); // well past the settle window
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  it('still honours an explicit retry of a failed level under a hidden layer', () => {
    // retryLazyChildByNodePath is a user-driven action (the error toast's retry)
    // and deliberately bypasses the visibility gate.
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const child = makeLazyChild(0.5, ensureLoaded);
    child.nodePath = '/g/level1';
    child.failed = true;
    const entry = makeEntry([makeChild(0), child], 0, '/g');
    withLayerParent(entry, false);
    reg.register(entry);

    expect(reg.retryLazyChildByNodePath('/g/level1')).toBe(true);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
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
    // diagonal → coverage metric ≈ 2.9 (well above child 1's 0.1 threshold),
    // so the coverage selector upgrades 0 → 1.
    const entry = placeAt(makeEntry(children, 0, '/g'), 0, 0, -3);
    reg.register(entry);

    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
    expect(entry.offScreen).toBe(false); // on-screen → no "(off-screen)" hint
  });

  it('auto: keeps raw position_bounds for the frustum gate when lod_bounds are off-screen', () => {
    const reg = registryWith(cameraLookingDownNegZ());
    const positionBounds = { min: [-1, -1, -1], max: [100, 1, 1] };
    const lodBounds = { min: [99, -1, -1], max: [100, 1, 1] };
    const children = [0, 0.5].map((threshold) =>
      withLodBounds({ ...makeChild(threshold), positionBounds }, lodBounds)
    );
    const entry = placeAt(makeEntry(children, 1, '/g'), 0, 0, -5);
    entry.selector = 'screen-area';
    reg.register(entry);

    reg.evaluatePerFrame();

    expect(entry.offScreen).toBe(false);
    expect(children[0].object.visible, 'off-screen robust core still drives a coarse metric').toBe(
      true
    );
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

// ────────────────────────────────────────────────────────────────────────
// Fresh-but-empty display guard — a fresh level that committed 0 elements
// while another fresh level has visible geometry signals inconsistent data;
// the registry must show the populated level instead of blanking the group.
// ────────────────────────────────────────────────────────────────────────

/** A gsplats child stamped fresh with an explicit committed splat count. */
function makeCountedChild(
  coverageFraction: number,
  loadedViewVersion: number,
  visibleSplatCount: number
): LODGroupChild {
  const child = makeChild(coverageFraction);
  child.object.userData = { nodeType: 'gsplats', loadedViewVersion, visibleSplatCount };
  return child;
}

describe('LODGroupRegistry — retryLazyChildByNodePath', () => {
  it('clears the failure cooldown and re-kicks ensureLoaded for a named lazy leaf', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const child = makeLazyChild(0.5, ensureLoaded);
    child.nodePath = '/g/child_1';
    child.failed = true;
    child.failedTick = 42;
    reg.register(makeEntry([makeChild(0), child], 0, '/g'));

    expect(reg.retryLazyChildByNodePath('/g/child_1')).toBe(true);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    expect(child.failed).toBe(false);
    expect(child.failedTick).toBeUndefined();
    // kickDeferredLoad owns the loading flag (the thunk never sets it).
    expect(child.loading).toBe(true);
  });

  it('returns true WITHOUT re-firing when a load is already in flight', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const child = makeLazyChild(0.5, ensureLoaded);
    child.nodePath = '/g/child_1';
    child.loading = true;
    reg.register(makeEntry([makeChild(0), child], 0, '/g'));

    expect(reg.retryLazyChildByNodePath('/g/child_1')).toBe(true);
    expect(ensureLoaded).not.toHaveBeenCalled();
  });

  it('clears a permanent failure latch for an explicit retry', () => {
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      () => true
    );
    const ensureLoaded = vi.fn();
    const child = makeLazyChild(0.5, ensureLoaded);
    child.nodePath = '/g/child_1';
    child.failed = true;
    child.failedTick = 42;
    child.permanentlyFailed = true;
    reg.register(makeEntry([makeChild(0), child], 0, '/g'));

    expect(reg.retryLazyChildByNodePath('/g/child_1')).toBe(true);
    expect(ensureLoaded).toHaveBeenCalledTimes(1);
    expect(child.failed).toBe(false);
    expect(child.failedTick).toBeUndefined();
    expect(child.permanentlyFailed).toBe(false);
    expect(child.loading).toBe(true);
  });

  it('retries an anonymous deferred-group placeholder by its authored node path', () => {
    const reg = makeRegistry();
    const ensureLoaded = vi.fn();
    const groupChild = makeLazyChild(0.5, ensureLoaded);
    groupChild.nodePath = '/g/nested';
    groupChild.failed = true;
    groupChild.permanentlyFailed = true;
    reg.register(makeEntry([makeChild(0), groupChild], 0, '/g'));

    expect(groupChild.object.name).toBe('');
    expect(reg.retryLazyChildByNodePath('/g/nested')).toBe(true);
    expect(ensureLoaded).toHaveBeenCalledOnce();
    expect(groupChild.permanentlyFailed).toBe(false);
    expect(reg.getFailedLazyChildPaths()).toEqual([]);
    expect(reg.retryLazyChildByNodePath('/nope')).toBe(false);
    expect(reg.retryLazyChildByNodePath('')).toBe(false);
  });

  it('eager children (no ensureLoaded) never match even when named', () => {
    const reg = makeRegistry();
    const eager = makeChild(0);
    eager.object.name = '/g/child_0';
    reg.register(makeEntry([eager], 0, '/g'));
    expect(reg.retryLazyChildByNodePath('/g/child_0')).toBe(false);
  });
});

describe('LODGroupRegistry — fresh-but-empty display guard', () => {
  it('redirects display to the coarsest fresh NON-empty level when the chosen level is empty', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    // Identity camera → aspiration is the finest level; it is FRESH but
    // committed 0 splats (the stale-cache / corrupt-data scenario), while the
    // coarse level holds 100 fresh splats.
    const children = [makeCountedChild(0, 2, 100), makeCountedChild(0.5, 2, 0)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true); // populated coarse shown
    expect(children[1].object.visible).toBe(false); // empty fine hidden
  });

  it('directs network-backed empty levels to Retry instead of clear-cache', () => {
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getViewVersion: () => 2,
      hasNetworkFailureUnder: (path) => path === '/g',
    });
    const children = [makeCountedChild(0, 2, 100), makeCountedChild(0.5, 2, 0)];
    reg.register(makeEntry(children, 0, '/g'));

    reg.evaluatePerFrame();

    expect(warning).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('use the monitor Retry action')
    );
    expect(warning.mock.calls[0]?.[1]).not.toContain('?clear-cache');
    warning.mockRestore();
  });

  it('leaves a genuinely empty slice unchanged (every fresh level empty)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const children = [makeCountedChild(0, 2, 0), makeCountedChild(0.5, 2, 0)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    // No non-empty fallback exists → the aspiration (fine, fresh) displays
    // as before; exactly one level visible.
    expect(children.filter((c) => c.object.visible).length).toBe(1);
    expect(children[1].object.visible).toBe(true);
  });

  it('is inert for untracked counts (no commit stamp)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    // Fine level fresh but WITHOUT a visibleSplatCount stamp → count is
    // unknown (null), not known-empty → no redirect.
    const children = [makeCountedChild(0, 2, 100), makeGsplatChild(0.5, 2)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
  });

  it('recovers once the empty level recommits with visible elements', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const children = [makeCountedChild(0, 2, 100), makeCountedChild(0.5, 2, 0)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(children[0].object.visible).toBe(true); // guard active
    (children[1].object.userData as { visibleSplatCount: number }).visibleSplatCount = 5000;
    reg.evaluatePerFrame();
    expect(children[1].object.visible).toBe(true); // healed → fine shows
    expect(children[0].object.visible).toBe(false);
  });

  it('protects the redirected (displayed) level from eviction', () => {
    const relCoarse = vi.fn();
    const relFine = vi.fn();
    const coarse = {
      ...makeCountedChild(0, 2, 100),
      ready: true,
      release: relCoarse as () => void,
      lastVisibleTick: 5,
    };
    const fine = {
      ...makeCountedChild(0.5, 2, 0),
      ready: true,
      release: relFine as () => void,
      lastVisibleTick: 5,
    };
    const children = [coarse, fine];
    const reg = makeRegistry([0, 1, 2], 50, residentModel(children), () => 2);
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // guard redirected display here
    expect(relCoarse).not.toHaveBeenCalled(); // displayed level never freed
    expect(relFine).toHaveBeenCalledTimes(1); // hidden empty level is evictable
  });

  it('never redirects to a NOT-READY deferred-group placeholder (keeps the fresh-but-empty level)', () => {
    // Property-harness repro (campaign-4 iter-7, bug A): the fresh level
    // committed 0 elements (poisoned cache) and the ONLY other level is a
    // deferred kind=partition placeholder — a bare group, ready:false, nothing
    // committed. Pre-fix ``childFreshAndCount`` reported that placeholder
    // fresh-with-unknown-count (its ``!aggregate`` branch never checked
    // ``isReady``), the guard redirected display onto it, and the ready-gated
    // visibility pass then showed NOTHING — a permanently blank group. The
    // guard must only redirect to a level that can actually draw; with no such
    // level, an empty-but-real level beats a blank placeholder.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const empty = makeCountedChild(0, 2, 0); // fresh@2, committed 0 splats
    const placeholder: LODGroupChild = {
      object: new THREE.Group(), // no leaf nodeType, no stamped leaves
      coverageFraction: 1,
      positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
      ready: false,
      ensureLoaded: vi.fn(),
    };
    reg.register(makeEntry([empty, placeholder], 0, '/g'));
    for (let i = 0; i < 5; i++) reg.evaluatePerFrame();
    expect(empty.object.visible).toBe(true); // fresh-but-empty level kept on screen
    expect(placeholder.object.visible).toBe(false); // never the blank placeholder
  });
});

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
// Group-typed LOD child freshness/emptiness (overview / nested lod). A group
// LOD child (a deferred kind=partition or nested lod subtree) carries no leaf
// nodeType on its own node, so leaf-only isFresh would call it unconditionally
// fresh and visibleElementCount would return null. The registry must fold the
// subtree's visible stamped leaves instead (matches the never-downgrade gate).
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — group-typed LOD child freshness', () => {
  /**
   * A group LOD child (bare THREE.Group placeholder, no leaf nodeType) with one
   * visible stamped gsplats leaf under it — what the overview recipe's deferred
   * kind=partition branch looks like once activated.
   */
  function makeGroupChildWithLeaf(
    coverageFraction: number,
    leafVersion: number,
    count: number
  ): { child: LODGroupChild; leaf: THREE.Object3D } {
    const group = new THREE.Group();
    const leaf = new THREE.Group();
    leaf.visible = true;
    leaf.userData = {
      nodeType: 'gsplats',
      loadedViewVersion: leafVersion,
      visibleSplatCount: count,
      committedLadderComplete: true,
    };
    group.add(leaf);
    return {
      child: {
        object: group,
        coverageFraction,
        positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
        ready: true,
      },
      leaf,
    };
  }

  function makeCoarseLeaf(count: number, version = 2): LODGroupChild {
    return makeLeafAt(0, count, version);
  }

  function makeLeafAt(coverageFraction: number, count: number, version = 2): LODGroupChild {
    const child = { ...makeGsplatChild(coverageFraction, version), ready: true };
    child.object.userData = {
      nodeType: 'gsplats',
      loadedViewVersion: version,
      visibleSplatCount: count,
    };
    return child;
  }

  it('folds a STALE group subtree → shows the coarse leaf, then swaps up when the subtree recommits', () => {
    // Pins F3: a group aspiration whose inner leaves are stale for the current
    // version must NOT be treated as fresh (leaf-only isFresh returns true for a
    // group nodeType). Pre-fix the stale group displays the previous slice with
    // no coarse fallback.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCoarseLeaf(50);
    const { child: fineGroup, leaf } = makeGroupChildWithLeaf(0.5, 1, 100); // subtree stale@1
    reg.register(makeEntry([coarse, fineGroup], 0, '/ov'));

    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // fresh coarse shown while the group is stale
    expect(fineGroup.object.visible).toBe(false);
    expect(reg.list()[0].activeChildIndex).toBe(1); // aspiration still the finest (the group)

    // Inner leaf recommits for v2 → the group folds fresh → swap up.
    (leaf.userData as { loadedViewVersion: number }).loadedViewVersion = 2;
    reg.evaluatePerFrame();
    expect(fineGroup.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
  });

  it('redirects when a FRESH group subtree committed all-empty leaves (fresh-but-empty guard)', () => {
    // Pins F4: a fresh group whose visible leaves all committed 0 elements
    // (poisoned/stale cache) would blank the group; leaf-only visibleElementCount
    // returns null for a group so the guard never fired. Now it folds to count 0
    // and redirects to the coarse non-empty leaf.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCoarseLeaf(100);
    const { child: fineGroup } = makeGroupChildWithLeaf(0.5, 2, 0); // fresh@2 but empty
    reg.register(makeEntry([coarse, fineGroup], 0, '/ov'));

    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // redirected to the non-empty coarse leaf
    expect(fineGroup.object.visible).toBe(false);
  });

  it('empty-guard fallback SKIPS a fresh-but-empty GROUP child (group-aware scan)', () => {
    // coarsest→finest: [empty GROUP, non-empty leaf, empty leaf]. The finest
    // (empty leaf) is chosen then redirected by the fresh-but-empty guard; the
    // fallback scan must SKIP the empty GROUP at index 0 (a leaf-only element
    // count reads null for a group and would wrongly accept it as non-empty,
    // redirecting onto a blank group) and land on the non-empty leaf at index 1.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const { child: emptyGroup } = makeGroupChildWithLeaf(0, 2, 0); // fresh@2, subtree empty
    const nonEmptyLeaf = makeLeafAt(0.5, 100);
    const emptyFineLeaf = makeLeafAt(1.0, 0); // finest, fresh, empty → chosen then redirected
    reg.register(makeEntry([emptyGroup, nonEmptyLeaf, emptyFineLeaf], 0, '/ov'));

    reg.evaluatePerFrame();
    expect(nonEmptyLeaf.object.visible).toBe(true); // group-aware fallback landed here
    expect(emptyGroup.object.visible).toBe(false); // NOT redirected onto the empty group
    expect(emptyFineLeaf.object.visible).toBe(false);
  });

  it('slice-aware fallback skips a READY group child with a STALE subtree (shows the fresh level)', () => {
    // Property-harness repro (campaign-4 iter-7, bug D): after a re-slice the
    // fallback picker ``coarsestFreshOrReadyIndex`` still used the leaf-only
    // ``coarsestFreshIndex``/``isFresh`` (non-leaf ⇒ unconditionally fresh)
    // while its sibling paths were converted to the group-aware
    // ``childFreshAndCount`` — so it displayed the OLD slice from a stale
    // partition branch even though a genuinely fresh level was resident.
    const TINY = { min: [0, 0, 0], max: [1e-4, 1e-4, 1e-4] }; // → aspiration = level 0
    let version = 0;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => version);
    const coarseLeaf = makeLeafAt(0, 50, 0); // stamped v0
    coarseLeaf.positionBounds = TINY;
    const { child: staleGroup } = makeGroupChildWithLeaf(0.5, 0, 200); // leaves stamped v0
    staleGroup.positionBounds = TINY;
    const fineLeaf = makeLeafAt(1.0, 1600, 0);
    fineLeaf.positionBounds = TINY;
    reg.register(makeEntry([coarseLeaf, staleGroup, fineLeaf], 0, '/ov'));
    reg.evaluatePerFrame(); // v0 steady state: coarse aspiration displayed

    // Re-slice: bump the view version; ONLY the finest leaf recommits fresh
    // (the eager coarse sweep + the deferred group reload have not landed yet).
    version = 1;
    (fineLeaf.object.userData as { loadedViewVersion: number }).loadedViewVersion = 1;
    for (let f = 0; f < 3; f++) reg.evaluatePerFrame();
    expect(fineLeaf.object.visible).toBe(true); // the only level fresh for v1
    expect(staleGroup.object.visible).toBe(false); // stale subtree = old slice, skipped
    expect(coarseLeaf.object.visible).toBe(false);
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
  function makeStaleLazyFine(
    coverageFraction: number,
    staleVersion: number,
    ensureLoaded: () => void
  ) {
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

  it('never re-activates a stale deferred GROUP child — its leaves are sweep-driven (#2632)', () => {
    // The overview recipe's fine branch: a READY group child (a bare Group, no
    // nodeType) whose stamped leaf is stale for the current version. Its
    // `ensureLoaded` is `loadChildren`, which attaches a fresh subtree every
    // time — re-firing it for staleness hung a second full copy of the branch
    // under the placeholder. The leaves are sweep-registered and re-stamp
    // themselves, so staleness must show the coarse fallback and kick nothing.
    const ensureLoaded = vi.fn();
    const group = makeChild(0.5);
    group.ready = true;
    group.ensureLoaded = ensureLoaded;
    group.deferredGroup = true; // what load-lod-group-node stamps on the group path
    const leaf = new THREE.Group();
    leaf.userData = { nodeType: 'gsplats', loadedViewVersion: 1, visibleSplatCount: 10 };
    group.object.add(leaf);
    const children = [makeGsplatChild(0, 2), group];
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2); // leaf stale at v2
    reg.register(makeEntry(children, 0, '/g'));
    for (let i = 0; i < 14; i++) reg.evaluatePerFrame(); // well past the settle window

    expect(ensureLoaded).not.toHaveBeenCalled();
    // Staleness WAS detected (not a vacuous pass): the group is the aspiration
    // yet the coarse fresh level is what is displayed.
    expect(children[0].object.visible).toBe(true);
    expect(group.object.visible).toBe(false);
  });

  it('clear() resets the settle clock so a reused registry reloads promptly after a dataset switch', () => {
    const first = vi.fn();
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    reg.register(makeEntry([makeGsplatChild(0, 2), makeStaleLazyFine(0.5, 1, first)], 0, '/a'));
    for (let i = 0; i < 40; i++) reg.evaluatePerFrame(); // settle clock seeded at tick 1, tick now 40
    expect(first).toHaveBeenCalledTimes(1);

    // Dataset switch on a reused registry: tick restarts at 0. The SAME version
    // (2) is observed again — without the reset the clock still points at the
    // old scene's tick and "settled" would need ~40 more frames.
    reg.clear();
    const second = vi.fn();
    reg.register(makeEntry([makeGsplatChild(0, 2), makeStaleLazyFine(0.5, 1, second)], 0, '/b'));
    for (let i = 0; i < 12; i++) reg.evaluatePerFrame();
    expect(second).toHaveBeenCalledTimes(1);
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

// ────────────────────────────────────────────────────────────────────────
// Never-downgrade display gate. A lazy level flips ready after its FIRST
// additive chunk commits; ungated, the swap to a fresh-but-still-streaming
// aspiration pops displayed quality down to chunk-1 (zoom in, zoom out, or
// after a scrub settles) and climbs back. The registry must hold the
// previously-displayed level while the streaming aspiration is strictly
// worse, and release on completion / count crossover / failure.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — never-downgrade display gate', () => {
  /**
   * A fresh counted gsplats child whose additive ladder is still streaming:
   * the committed stamp says incomplete (what the display gate reads) and the
   * live `hasMoreLODs` thunk agrees (what the registry's refinement-kick
   * logic reads) — the consistent state of a real mid-ladder level.
   */
  function makeStreamingChild(
    coverageFraction: number,
    loadedViewVersion: number,
    visibleSplatCount: number
  ): LODGroupChild {
    const child = makeCountedChild(coverageFraction, loadedViewVersion, visibleSplatCount);
    child.object.userData.committedLadderComplete = false;
    child.hasMoreLODs = () => true;
    return child;
  }

  /** Mark a streaming child's ladder committed-complete (final commit landed). */
  function completeLadder(child: LODGroupChild): void {
    child.object.userData.committedLadderComplete = true;
    child.hasMoreLODs = () => false;
  }

  it('zoom in: holds the full coarse level until the streaming fine level crosses its count', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100); // displayed, complete
    const fine = makeStreamingChild(0.5, 2, 10); // fresh chunk-1, ladder streaming
    reg.register(makeEntry([coarse, fine], 0, '/g'));

    reg.evaluatePerFrame(); // identity camera → aspiration = fine
    expect(reg.list()[0].activeChildIndex).toBe(1); // aspiration advanced
    expect(coarse.object.visible).toBe(true); // ...but display held on coarse
    expect(fine.object.visible).toBe(false);

    // Ladder streams: count climbs but stays below prev → still held.
    (fine.object.userData as { visibleSplatCount: number }).visibleSplatCount = 60;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);

    // Count crossover → swap; the rest of the ladder streams visibly.
    (fine.object.userData as { visibleSplatCount: number }).visibleSplatCount = 100;
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
  });

  it('zoom out: holds the full fine level until the streaming coarse level completes', () => {
    // Tiny bounds → tiny coverage metric → the COARSE level is desired.
    const tinyBounds = { min: [0, 0, 0], max: [0.001, 0.001, 0.001] };
    const coarse = makeStreamingChild(0, 2, 5); // cold, chunk-1 committed
    const fine = makeCountedChild(0.5, 2, 1000); // fully-laddered, displayed
    coarse.positionBounds = tinyBounds;
    fine.positionBounds = tinyBounds;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    reg.register(makeEntry([coarse, fine], 1, '/g')); // fine active + displayed

    reg.evaluatePerFrame();
    expect(reg.list()[0].activeChildIndex).toBe(0); // aspiration moved coarse
    expect(fine.object.visible).toBe(true); // display held on the better fine
    expect(coarse.object.visible).toBe(false);

    // Crossover is unreachable (coarse total < fine total) — completion releases.
    completeLadder(coarse);
    (coarse.object.userData as { visibleSplatCount: number }).visibleSplatCount = 50;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(false);
  });

  it('holds through the fetch-resolved-but-not-committed window (stamp beats live loader state)', () => {
    // The loader's live hasMoreLODs flips false at fetch-resolve, BEFORE the
    // final chunk's processing + commit. The gate reads the COMMITTED stamp,
    // which only flips in the same synchronous commit as the final count —
    // so the hold persists through that window by construction.
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    const fine = makeStreamingChild(0.5, 2, 10);
    fine.hasMoreLODs = () => false; // final fetch resolved; commit not landed
    reg.register(makeEntry([coarse, fine], 0, '/g'));

    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // held through the commit window

    // The final commit lands: count + ladder stamp written together → release.
    completeLadder(fine);
    (fine.object.userData as { visibleSplatCount: number }).visibleSplatCount = 500;
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
  });

  it('scrub-settle: the complete coarse fallback is held over the fine chunk-1 recommit', () => {
    let version = 1;
    let clock = 0;
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      () => version,
      undefined,
      () => clock
    );
    const coarse = makeCountedChild(0, 1, 100);
    const fine = makeCountedChild(0.5, 1, 1000); // complete for v1
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true); // steady state: fine shown

    // Scrub to v2: both stale. The stale-hold keeps the fine level's previous
    // slice up first (it is 10x the fallback — see the stale-hold suite); this
    // test is about what happens AFTER that budget is spent, so run the clock
    // past it to reach the staleness fallback.
    version = 2;
    reg.evaluatePerFrame();
    clock += 1000;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);

    // The eager coarse recommits fresh + complete for v2 via the sweep.
    Object.assign(coarse.object.userData!, { loadedViewVersion: 2, visibleSplatCount: 100 });
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);

    // The fine reload commits chunk-1 fresh — WORSE than the coarse on
    // screen. Pre-gate this swapped immediately (the post-scrub pop).
    Object.assign(fine.object.userData!, {
      loadedViewVersion: 2,
      visibleSplatCount: 10,
      committedLadderComplete: false,
    });
    fine.hasMoreLODs = () => true;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // held
    expect(fine.object.visible).toBe(false);

    // Ladder catches up → crossover → fine shows again.
    (fine.object.userData as { visibleSplatCount: number }).visibleSplatCount = 400;
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
  });

  it('releases the hold when the aspiration ladder failed (degrade to ungated behavior)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    const fine = makeStreamingChild(0.5, 2, 10);
    fine.failed = true;
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true); // no hold behind a failing ladder
  });

  it('is bypassed by an explicit level lock (the user wants that level now)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    const fine = makeStreamingChild(0.5, 2, 10);
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true); // locked level shows while streaming
  });

  it('is bypassed while the group is off-screen (frustum-culled: no visual pop)', () => {
    // Bounds far outside the identity-camera frustum → off-screen gate holds
    // the group at the coarsest ready level; the display gate must not pin
    // the fine level's VRAM for an invisible group.
    const farBounds = { min: [100, 100, 100], max: [110, 110, 110] };
    const coarse = makeStreamingChild(0, 2, 5);
    const fine = makeCountedChild(0.5, 2, 1000);
    coarse.positionBounds = farBounds;
    fine.positionBounds = farBounds;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    reg.register(makeEntry([coarse, fine], 1, '/g')); // fine displayed
    reg.evaluatePerFrame();
    expect(reg.list()[0].offScreen).toBe(true);
    expect(coarse.object.visible).toBe(true); // no hold off-screen
    expect(fine.object.visible).toBe(false);
  });

  it('preserves the hold across an off-screen excursion (look away and back does not re-pop)', () => {
    // Zoom-out hold: displaying the full fine level while the coarser
    // streaming aspiration catches up. A camera look-away (no view-version
    // change) must NOT clobber the gate's memory: on return the finer level
    // is still held instead of popping to the partial coarse aspiration.
    const tiny = { min: [0, 0, 0], max: [0.001, 0.001, 0.001] }; // on-screen, coarse desired
    const far = { min: [100, 100, 100], max: [110, 110, 110] }; // frustum-culled
    const coarse = makeStreamingChild(0, 2, 5); // cold, still streaming
    const fine = makeCountedChild(0.5, 2, 1000); // full, displayed
    const setBounds = (b: { min: number[]; max: number[] }) => {
      coarse.positionBounds = b;
      fine.positionBounds = b;
    };
    setBounds(tiny);
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    reg.register(makeEntry([coarse, fine], 1, '/g')); // fine active + displayed

    reg.evaluatePerFrame(); // on-screen: gate holds the fine level
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);

    setBounds(far);
    reg.evaluatePerFrame(); // off-screen: coarse shown, displayedChildIndex clobbered
    expect(reg.list()[0].offScreen).toBe(true);
    expect(coarse.object.visible).toBe(true);

    setBounds(tiny);
    reg.evaluatePerFrame(); // back on-screen: the finer level must be re-held
    expect(fine.object.visible).toBe(true); // pre-fix this popped to the partial coarse
    expect(coarse.object.visible).toBe(false);
  });

  it('keeps the held aspiration warm in the eviction LRU (lastVisibleTick re-stamped)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    const fine = makeStreamingChild(0.5, 2, 10);
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    const afterFirst = fine.lastVisibleTick;
    reg.evaluatePerFrame();
    // The never-shown stamp fires only once; the hold must keep re-stamping.
    expect(fine.lastVisibleTick).toBeGreaterThan(afterFirst!);
  });

  it('is inert for a deferred-group aspiration (untracked count, no hasMoreLODs)', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    const groupFine = makeChild(0.5); // nested-group level: no nodeType/count
    reg.register(makeEntry([coarse, groupFine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(groupFine.object.visible).toBe(true); // current behavior preserved
  });

  it('nested-group aspiration (partition subtree): holds the coarse leaf until the aggregate crosses or completes', () => {
    // The `overview` shape: an eager coarse cap leaf vs a deferred
    // kind=partition branch whose parts are streaming their ladders. The
    // gate reads the SUBTREE aggregate (visible stamped part meshes).
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    coarse.object.userData.committedLadderComplete = true;

    const placeholder = new THREE.Group(); // anonymous deferred-GROUP wrapper
    const partition = new THREE.Group();
    placeholder.add(partition);
    const makePart = (count: number, complete: boolean): THREE.Mesh => {
      const m = new THREE.Mesh();
      m.userData = {
        nodeType: 'gsplats',
        loadedViewVersion: 2,
        visibleSplatCount: count,
        committedLadderComplete: complete,
      };
      return m;
    };
    const part1 = makePart(10, false);
    const part2 = makePart(20, false);
    partition.add(part1);
    partition.add(part2);
    const fine: LODGroupChild = {
      object: placeholder,
      coverageFraction: 0.5,
      positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
      ready: true, // activation (loadChildren) settled
    };
    reg.register(makeEntry([coarse, fine], 0, '/g'));

    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // held: aggregate 30 < 100
    expect(placeholder.visible).toBe(false);

    // Parts stream past the cap → aggregate crossover → swap.
    (part1.userData as { visibleSplatCount: number }).visibleSplatCount = 80;
    (part2.userData as { visibleSplatCount: number }).visibleSplatCount = 40;
    reg.evaluatePerFrame();
    expect(placeholder.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
  });

  it('nested-group aspiration releases on aggregate completion even below the prev count', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 1000);
    coarse.object.userData.committedLadderComplete = true;
    const placeholder = new THREE.Group();
    const part = new THREE.Mesh();
    part.userData = {
      nodeType: 'gsplats',
      loadedViewVersion: 2,
      visibleSplatCount: 50,
      committedLadderComplete: true, // every part ladder committed complete
    };
    placeholder.add(part);
    const fine: LODGroupChild = {
      object: placeholder,
      coverageFraction: 0.5,
      positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
      ready: true,
    };
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(placeholder.visible).toBe(true); // complete → shown despite 50 < 1000
    expect(coarse.object.visible).toBe(false);
  });

  it('keeps advancing the held aspiration ladder while the previous level stays visible', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const coarse = makeCountedChild(0, 2, 100);
    const fine = makeStreamingChild(0.5, 2, 10);
    fine.ensureLoaded = vi.fn(() => {
      fine.loading = false; // simulate a completed progressive pass
    });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    for (let i = 0; i < 16; i++) reg.evaluatePerFrame(); // settle (~8) + passes
    expect((fine.ensureLoaded as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThan(1);
    expect(coarse.object.visible).toBe(true); // held throughout
    expect(fine.object.visible).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Coverage-band cross-fade (on by default; ?no-lod-fade disables) — two adjacent
// blendable (additive/luminous/volumetric) levels
// render with complementary opacity as the DISTANCE (coverage metric) crosses
// their boundary. Distance-driven, independent of streaming. Off / non-blendable
// / off-screen ⇒ the byte-identical hard swap. A small tile
// ([0,0,0]–[0.3,0.3,0.3]) under the identity test camera (800×600 viewport,
// fitted axis 600) projects to a coverage metric of exactly 0.5
// (150 px diagonal ÷ (FILL_FACTOR·600) = 150/300), so placing
// the finer level's threshold at/near 0.5 lands the metric in its blend band.
// ────────────────────────────────────────────────────────────────────────

describe('LODGroupRegistry — coverage-band cross-fade', () => {
  interface FadeMatStub {
    userData: { blendingMode: string };
    _op: number;
    updateOpacity(v: number): void;
    getOpacity(): number;
    clone(): FadeMatStub;
  }
  function fadeMat(blendingMode = 'additive'): FadeMatStub {
    return {
      userData: { blendingMode },
      _op: 1,
      updateOpacity(v: number) {
        this._op = v;
      },
      getOpacity() {
        return this._op;
      },
      clone() {
        return fadeMat(blendingMode);
      },
    };
  }
  // A gsplats leaf whose object is a real Mesh (so .material / .traverse work),
  // with position bounds [0,0,0]-[0.3,0.3,0.3] → projects to coverage metric 0.5
  // on the 800×600 test viewport below (fitted axis 600, FILL_FACTOR 0.5).
  function fadeChild(
    coverageFraction: number,
    opts: { mode?: string; ready?: boolean } = {}
  ): LODGroupChild {
    const mesh = new THREE.Mesh();
    mesh.material = fadeMat(opts.mode ?? 'additive') as unknown as THREE.Material;
    mesh.userData = {
      nodeType: 'gsplats',
      loadedViewVersion: 2,
      visibleSplatCount: 100,
      committedLadderComplete: true, // complete single-set level (no streaming)
    };
    return {
      object: mesh,
      coverageFraction,
      positionBounds: { min: [0, 0, 0], max: [0.3, 0.3, 0.3] },
      ready: opts.ready ?? true,
    };
  }
  function makeReg(crossFade: boolean, getViewVersion = () => 2, now?: () => number) {
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    return new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getViewVersion,
      getCrossFadeEnabled: () => crossFade,
      now,
    });
  }
  const liveOpacity = (c: LODGroupChild): number =>
    ((c.object as THREE.Mesh).material as unknown as FadeMatStub).getOpacity();

  it('blends the two levels 50/50 exactly at the boundary (metric == threshold)', () => {
    const reg = makeReg(true);
    const coarse = fadeChild(0); // threshold 0
    const fine = fadeChild(0.5); // boundary at 0.5 == the metric
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(true);
    expect(liveOpacity(fine)).toBeCloseTo(0.5, 6);
    expect(liveOpacity(coarse)).toBeCloseTo(0.5, 6);
    expect(coarse.lastVisibleTick).toBeGreaterThan(0);
    expect(fine.lastVisibleTick).toBeGreaterThan(0);
  });

  it('does not blend a held-stale aspiration with a fresh partner', () => {
    const state = { version: 2, clock: 0 };
    const reg = makeReg(
      true,
      () => state.version,
      () => state.clock
    );
    const coarse = fadeChild(0);
    const fine = fadeChild(0.5);
    (coarse.object.userData as { visibleSplatCount: number }).visibleSplatCount = 10;
    reg.register(makeEntry([coarse, fine], 0, '/g'));

    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(true);
    expect(liveOpacity(coarse)).toBeCloseTo(0.5, 6);
    expect(liveOpacity(fine)).toBeCloseTo(0.5, 6);

    state.version = 3;
    Object.assign(coarse.object.userData, {
      loadedViewVersion: 3,
      visibleSplatCount: 10,
    });
    reg.evaluatePerFrame();

    expect(reg.get('/g')!.displayedChildIndex).toBe(1);
    expect(coarse.object.visible).toBe(false);
    expect(fine.object.visible).toBe(true);
    expect(liveOpacity(fine)).toBe(1);
  });

  it('weights shift with the metric position in the band (finer boundary just above the metric ⇒ coarse dominant)', () => {
    // Proportional band: boundary 0.625, gap 0.625 → half-width 0.4·0.625=0.25,
    // band [0.375,0.875]. metric 0.5 → finer weight smoothstep(0.375,0.875,0.5)=0.15625.
    const reg = makeReg(true);
    const coarse = fadeChild(0);
    const fine = fadeChild(0.625);
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(true);
    expect(liveOpacity(fine)).toBeCloseTo(0.15625, 5);
    expect(liveOpacity(coarse)).toBeCloseTo(0.84375, 5);
    expect(liveOpacity(fine) + liveOpacity(coarse)).toBeCloseTo(1, 6);
  });

  it('shows a single level (no partner) when the metric is outside every band', () => {
    // boundary 0.9, gap 0.9 → band [0.54,1.26]; metric 0.5 < 0.54 ⇒ coarse alone.
    const reg = makeReg(true);
    const coarse = fadeChild(0);
    const fine = fadeChild(0.9);
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(false);
    expect(liveOpacity(coarse)).toBe(1); // untouched
  });

  it('flag OFF ⇒ hard swap, one level visible, opacity untouched (byte-identical)', () => {
    const reg = makeReg(false);
    const coarse = fadeChild(0);
    const fine = fadeChild(0.5);
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    // metric 0.5 ≥ threshold 0.5 ⇒ finest selected, shown alone; no opacity writes.
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
    expect(liveOpacity(fine)).toBe(1);
    expect(liveOpacity(coarse)).toBe(1);
  });

  it('non-blendable mode (max) ⇒ no blend even with the flag on', () => {
    const reg = makeReg(true);
    const coarse = fadeChild(0, { mode: 'max' });
    const fine = fadeChild(0.5, { mode: 'max' });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false); // hard swap to finest
    expect(liveOpacity(fine)).toBe(1);
    expect(liveOpacity(coarse)).toBe(1);
  });

  it('volumetric mode ⇒ blends 50/50 at the boundary (opacity scales τ, so the fade is well-behaved)', () => {
    const reg = makeReg(true);
    const coarse = fadeChild(0, { mode: 'volumetric' });
    const fine = fadeChild(0.5, { mode: 'volumetric' });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(true);
    expect(liveOpacity(fine)).toBeCloseTo(0.5, 6);
    expect(liveOpacity(coarse)).toBeCloseTo(0.5, 6);
  });

  it('mixed volumetric + non-blendable subtree ⇒ hard swap (uniformity requirement)', () => {
    const reg = makeReg(true);
    const coarse = fadeChild(0, { mode: 'volumetric' });
    const fine = fadeChild(0.5, { mode: 'max' });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false); // hard swap to finest
    expect(liveOpacity(fine)).toBe(1);
    expect(liveOpacity(coarse)).toBe(1);
  });

  it('brightness invariance: the two levels’ opacities always sum to 1 across the band', () => {
    // The physics guarantee (additive shader = energy·opacity, mass-conserved
    // levels ⇒ equal integrated E): blendedDC = E·(1−w) + E·w = E for all w. The
    // JS-side invariant underwriting it is exactly-complementary opacities.
    // (For volumetric the relevant quantity is per-ray optical depth τ, not
    // summed energy. τ is linear in opacity, so the same complementary weights
    // give 1−exp(−(w·τ_fine + (1−w)·τ_coarse)) — a monotone interpolation
    // between the two levels' absorptions, EXACT only where both present the
    // same per-ray τ. See the volumetric-math suite for the general case; the
    // JS-side complementary-weight invariant is what this test pins.)
    for (const boundary of [0.4, 0.45, 0.5, 0.55, 0.6]) {
      const reg = makeReg(true);
      const coarse = fadeChild(0);
      const fine = fadeChild(boundary);
      reg.register(makeEntry([coarse, fine], 0, '/g'));
      reg.evaluatePerFrame();
      if (coarse.object.visible && fine.object.visible) {
        expect(liveOpacity(coarse) + liveOpacity(fine)).toBeCloseTo(1, 6);
      }
    }
  });

  it('kicks the finer level’s load (no blend yet) when it is not resident, so the next crossing blends', () => {
    const reg = makeReg(true);
    const coarse = fadeChild(0);
    let loaded = false;
    const fine = fadeChild(0.5, { ready: false });
    fine.ensureLoaded = () => {
      loaded = true;
    };
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    // Partner (finer) not resident → no two-level blend this frame, but its load
    // is kicked so a subsequent crossing can fade against it.
    expect(loaded).toBe(true);
    expect(fine.object.visible).toBe(false);
  });

  it('byte-budget eviction never releases the ON-SCREEN cross-fade blend partner', () => {
    // Property-harness repro (campaign-4 iter-7, bug B): levels 1↔2 blend at
    // ~50/50 when VRAM pressure hits. The eviction pass protected only
    // ``displayedChildIndex`` (level 2); the blend partner (level 1) was an
    // ordinary candidate and got released MID-FADE — half the dissolve
    // vanished and a visible-but-not-ready level was left behind. The evictor
    // must never release anything on screen (``object.visible === true``).
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    let budget = 1e15;
    const relMid = vi.fn(() => {
      mid.ready = false;
    });
    const relFine = vi.fn(() => {
      fine.ready = false;
    });
    // fadeChild's 0.3-box bounds → coverage metric 0.5 = the 1↔2 boundary of
    // thresholds [0, 0.25, 0.5] (gap 0.25 → band [0.4, 0.6]) → exact 50/50 blend.
    const coarse = fadeChild(0); // eager fallback: no release, never evictable
    const mid = fadeChild(0.25);
    mid.release = relMid as () => void;
    const fine = fadeChild(0.5);
    fine.release = relFine as () => void;
    const children = [coarse, mid, fine];
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getViewVersion: () => 2,
      getCrossFadeEnabled: () => true,
      getResidentByteBudget: () => budget,
      getResidentBytes: () => children.reduce((s, c) => s + (c.ready !== false ? 100 : 0), 0),
    });
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame(); // steady blend: [mid, fine] visible at 50/50
    expect(mid.object.visible).toBe(true);
    expect(fine.object.visible).toBe(true);

    budget = 250; // 300 resident > 250 → eviction pass fires
    reg.evaluatePerFrame();
    expect(relMid).not.toHaveBeenCalled(); // the on-screen partner is protected
    expect(relFine).not.toHaveBeenCalled(); // the displayed level is protected
    expect(mid.object.visible).toBe(true); // the dissolve survives the pressure
    expect(fine.object.visible).toBe(true);
  });

  it('restores authored opacity once when fade management toggles OFF mid-fade', () => {
    // Property-harness repro (campaign-4 iter-7, observation E): with a 50/50
    // cross-fade in flight, turning BOTH anti-popping flags off skipped the
    // restore branch (``manageFade === false``) and stranded opacity 0.5
    // forever. The falling-edge restore must return every faded leaf to its
    // authored opacity on the next frame.
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    let crossFade = true;
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getViewVersion: () => 2,
      getCrossFadeEnabled: () => crossFade,
    });
    const coarse = fadeChild(0);
    const fine = fadeChild(0.5); // boundary at fadeChild's metric 0.5 → 50/50
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBeCloseTo(0.5, 6); // mid-fade

    crossFade = false; // both anti-popping flags now off
    for (let f = 0; f < 3; f++) reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBe(1); // authored opacity restored, not stranded
    expect(liveOpacity(coarse)).toBe(1);
    expect(fine.object.visible).toBe(true); // hard swap to the finest level
    expect(coarse.object.visible).toBe(false);
  });
});

// Streaming energy compensation (ON by default; ?no-lod-energy disables) — as a
// blendable (additive/luminous/volumetric) leaf's additive ladder streams in,
// its committed prefix carries only
// e(k) of the leaf's full energy, so it renders at e·E and brightens toward E as
// chunks arrive (a pop). Scaling the leaf's opacity by 1/e(k) holds the rendered
// energy at E throughout. Time-axis and PER-LEAF, orthogonal to the distance-
// driven cross-fade; the two compose multiplicatively. ENERGY_FLOOR=0.1 caps the
// boost at 10×. Off / non-blendable / complete / unstamped ⇒ byte-identical.
// ────────────────────────────────────────────────────────────────────────
describe('LODGroupRegistry — streaming energy compensation', () => {
  interface FadeMatStub {
    userData: { blendingMode: string };
    _op: number;
    updateOpacity(v: number): void;
    getOpacity(): number;
    clone(): FadeMatStub;
  }
  function fadeMat(blendingMode = 'additive'): FadeMatStub {
    return {
      userData: { blendingMode },
      _op: 1,
      updateOpacity(v: number) {
        this._op = v;
      },
      getOpacity() {
        return this._op;
      },
      clone() {
        return fadeMat(blendingMode);
      },
    };
  }
  // A gsplats leaf (0.3-box bounds → coverage metric 0.5). `energy` stamps the
  // committed prefix's energy fraction e(k); omitted ⇒ unstamped (no field).
  // `committedLadderComplete: true` keeps the never-downgrade gate out of the way
  // so these tests isolate the opacity math from the (orthogonal) hold logic.
  function fadeChild(
    coverageFraction: number,
    opts: { mode?: string; energy?: number } = {}
  ): LODGroupChild {
    const mesh = new THREE.Mesh();
    mesh.material = fadeMat(opts.mode ?? 'additive') as unknown as THREE.Material;
    mesh.userData = {
      nodeType: 'gsplats',
      loadedViewVersion: 2,
      visibleSplatCount: 100,
      committedLadderComplete: true,
      ...(opts.energy != null ? { committedEnergyFraction: opts.energy } : {}),
    };
    return {
      object: mesh,
      coverageFraction,
      positionBounds: { min: [0, 0, 0], max: [0.3, 0.3, 0.3] },
      ready: true,
    };
  }
  function makeReg(crossFade: boolean, energyComp: boolean) {
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    return new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getViewVersion: () => 2,
      getCrossFadeEnabled: () => crossFade,
      getEnergyCompEnabled: () => energyComp,
    });
  }
  const liveOpacity = (c: LODGroupChild): number =>
    ((c.object as THREE.Mesh).material as unknown as FadeMatStub).getOpacity();
  // Returned untyped: only used for referential-identity (toBe) clone checks.
  const liveMaterial = (c: LODGroupChild): unknown => (c.object as THREE.Mesh).material;

  it('boosts a streaming leaf’s opacity by 1/e (energyComp on, no cross-fade)', () => {
    // Finest (threshold 0.5) is selected alone at metric 0.5; e=0.5 ⇒ opacity ×2.
    const reg = makeReg(false, true);
    const coarse = fadeChild(0);
    const fine = fadeChild(0.5, { energy: 0.5 });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
    expect(liveOpacity(fine)).toBeCloseTo(2, 6);
    expect(liveOpacity(coarse)).toBe(1); // hidden, untouched
  });

  it('caps the boost at 1/ENERGY_FLOOR (10×) for a tiny early prefix', () => {
    const reg = makeReg(false, true);
    const fine = fadeChild(0.5, { energy: 0.02 });
    reg.register(makeEntry([fadeChild(0), fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBeCloseTo(10, 6); // 1/0.1, not 1/0.02 = 50
  });

  it('leaves a complete/unstamped leaf byte-identical (no compensation, no clone)', () => {
    const reg = makeReg(false, true);
    const fine = fadeChild(0.5); // no energy stamp ⇒ factor 1
    const matBefore = liveMaterial(fine);
    reg.register(makeEntry([fadeChild(0), fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBe(1);
    expect(liveMaterial(fine)).toBe(matBefore); // never cloned
  });

  it('both anti-popping flags off ⇒ byte-identical even for a streaming leaf', () => {
    const reg = makeReg(false, false);
    const fine = fadeChild(0.5, { energy: 0.5 });
    const matBefore = liveMaterial(fine);
    reg.register(makeEntry([fadeChild(0), fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBe(1);
    expect(liveMaterial(fine)).toBe(matBefore); // no writes, no clone
  });

  it('does not compensate a non-blendable (max) streaming leaf', () => {
    const reg = makeReg(false, true);
    const fine = fadeChild(0.5, { mode: 'max', energy: 0.5 });
    reg.register(makeEntry([fadeChild(0, { mode: 'max' }), fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBe(1); // max doesn't sum energy → no 1/e
  });

  it('compensates a volumetric streaming leaf by 1/e (opacity linearly scales τ)', () => {
    const reg = makeReg(false, true);
    const fine = fadeChild(0.5, { mode: 'volumetric', energy: 0.5 });
    reg.register(makeEntry([fadeChild(0, { mode: 'volumetric' }), fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBeCloseTo(2, 6); // τ restored to the full ladder's
  });

  it('composes with the cross-fade so rendered energy stays E (opacity·e sums to 1)', () => {
    // Both flags on. Boundary 0.5, metric 0.5 ⇒ 50/50 coverage. The finer level's
    // prefix carries e=0.5, so its opacity = 0.5 (coverage) × 2 (1/e) = 1.0; the
    // complete coarse = 0.5 × 1 = 0.5. Rendered energy 1.0·0.5 + 0.5·1.0 = 1.0 = E.
    const reg = makeReg(true, true);
    const coarse = fadeChild(0); // complete (e = 1)
    const fine = fadeChild(0.5, { energy: 0.5 });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(true);
    expect(liveOpacity(fine)).toBeCloseTo(1.0, 6);
    expect(liveOpacity(coarse)).toBeCloseTo(0.5, 6);
    const renderedEnergy = liveOpacity(fine) * 0.5 + liveOpacity(coarse) * 1.0;
    expect(renderedEnergy).toBeCloseTo(1, 6);
  });

  it('relaxes to the authored opacity as the ladder completes', () => {
    const reg = makeReg(false, true);
    const coarse = fadeChild(0);
    const fine = fadeChild(0.5, { energy: 0.5 });
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBeCloseTo(2, 6);
    // Ladder finishes: e → 1. The (now cloned) material restores to its base.
    (fine.object.userData as { committedEnergyFraction?: number }).committedEnergyFraction = 1;
    reg.evaluatePerFrame();
    expect(liveOpacity(fine)).toBe(1);
  });
});

describe('LODGroupRegistry — force-finest capture override (?lod-finest / LuxarAppOptions.lodFinest)', () => {
  function makeForceFinestRegistry(force: boolean): LODGroupRegistry {
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    return new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getForceFinestLOD: () => force,
    });
  }

  /** A child whose bounds project to ~zero screen coverage (coarse pick). */
  function tinyChild(coverageFraction: number): LODGroupChild {
    const child = makeChild(coverageFraction);
    child.positionBounds = { min: [0, 0, 0], max: [1e-4, 1e-4, 1e-4] };
    return child;
  }

  /** A child whose bounds sit entirely outside the identity frustum. */
  function offScreenChild(coverageFraction: number): LODGroupChild {
    const child = makeChild(coverageFraction);
    child.positionBounds = { min: [100, 100, 100], max: [101, 101, 101] };
    return child;
  }

  it('does not fold lod_bounds when force-finest bypasses the selector metric', () => {
    const reg = makeForceFinestRegistry(true);
    const children = [0, 0.5].map((threshold) =>
      withLodBounds(makeChild(threshold), { min: [0, 0, 0], max: [1, 1, 1] })
    );
    const entry = makeEntry(children, 0, '/g');
    const updateWorldMatrix = vi.spyOn(entry.groupObject, 'updateWorldMatrix');
    reg.register(entry);

    reg.evaluatePerFrame();

    expect(updateWorldMatrix).toHaveBeenCalledTimes(1);
    expect(children[1].object.visible).toBe(true);
  });

  function settle(reg: LODGroupRegistry): void {
    for (let i = 0; i < 5; i++) reg.evaluatePerFrame();
  }

  it('selects the FINEST level despite near-zero screen coverage (capture-quality override)', () => {
    const reg = makeForceFinestRegistry(true);
    const children = [tinyChild(0), tinyChild(0.5), tinyChild(1.0)];
    reg.register(makeEntry(children, 0, '/g'));
    settle(reg);
    expect(children[2].object.visible).toBe(true);
    expect(children[0].object.visible).toBe(false);
  });

  it('control: without the flag the same tiny group stays at the coarsest level', () => {
    const reg = makeForceFinestRegistry(false);
    const children = [tinyChild(0), tinyChild(0.5), tinyChild(1.0)];
    reg.register(makeEntry(children, 0, '/g'));
    settle(reg);
    expect(children[0].object.visible).toBe(true);
    expect(children[2].object.visible).toBe(false);
  });

  it('bypasses the off-screen coarsening gate (never coarsen during capture)', () => {
    const reg = makeForceFinestRegistry(true);
    const children = [offScreenChild(0), offScreenChild(0.5), offScreenChild(1.0)];
    reg.register(makeEntry(children, 0, '/g'));
    settle(reg);
    expect(children[2].object.visible).toBe(true);
  });
});

describe('LODGroupRegistry — blending-mode-switch stamp-clear recovery (depth-sort integration)', () => {
  it('a hidden-or-shown resident lazy level whose stamps were cleared reloads via the ready-but-stale kick and never blanks the display', () => {
    // The depth-sort coordinator's switch-TO-sorted hook clears BOTH the
    // committedData noop stamp AND the loadedViewVersion freshness stamp
    // on every affected level (layer-apply reaches all lod_group leaves).
    // Lazy levels are outside the reprocess sweep, so their ONLY recovery
    // is this registry's settle-gated ready-but-stale reload — and the
    // display must never blank meanwhile (a brief coarse fallback is the
    // documented staleness behavior, same as any re-slice).
    const version = 1;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => version);
    const coarse = makeCountedChild(0, version, 100); // eager, complete
    const ensureLoaded = vi.fn();
    const fine = makeCountedChild(0.5, version, 1000); // resident lazy level
    fine.ensureLoaded = ensureLoaded; // lazy: registry-driven reloads
    reg.register(makeEntry([coarse, fine], 0, '/g'));

    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true); // steady state: fine displayed

    // === The mode switch fires: both levels' stamps cleared. ===
    delete (coarse.object.userData as { loadedViewVersion?: number }).loadedViewVersion;
    delete (fine.object.userData as { loadedViewVersion?: number }).loadedViewVersion;

    // Settle-gated: evaluate several frames at a CONSTANT version. The
    // display must show SOMETHING every frame (ready-but-stale content),
    // and the stale lazy aspiration must get exactly one reload kick.
    for (let f = 0; f < 10; f++) {
      reg.evaluatePerFrame();
      expect(coarse.object.visible || fine.object.visible, `frame ${f}: display blanked`).toBe(
        true
      );
    }
    expect(ensureLoaded).toHaveBeenCalled();

    // The reload's re-commit lands (fresh stamp + count) → fine displays.
    fine.loading = false; // commit path's stamp; kickDeferredLoad set it true
    (fine.object.userData as { loadedViewVersion?: number }).loadedViewVersion = version;
    (coarse.object.userData as { loadedViewVersion?: number }).loadedViewVersion = version;
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
  });
});

describe('LODGroupRegistry — capture quiescence (isCaptureQuiescent)', () => {
  /** Bounds that fall entirely outside the identity camera's NDC frustum. */
  const FAR_BOUNDS = { min: [100, 100, 100], max: [101, 101, 101] };

  it('reports quiescent for an empty registry (nothing to wait for)', () => {
    expect(makeRegistry().isCaptureQuiescent()).toBe(true);
  });

  it('counts both lod groups and partitions as capture work', () => {
    const reg = makeRegistry();
    expect(reg.captureSize()).toBe(0);

    reg.register(makeEntry([makeChild(0)], 0, '/lod'));
    expect(reg.captureSize()).toBe(1);

    const partition = new THREE.Group();
    const part = new THREE.Group();
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [0, 0, 0], max: [0.5, 0.5, 0.5] },
        },
      ],
    });
    expect(reg.captureSize()).toBe(2);
  });

  it('waits across a partition rising edge until its targeted resync commits', () => {
    let loadPassInProgress = true;
    const requestReprocess = vi.fn(() => {
      loadPassInProgress = true;
    });
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      () => 2,
      undefined,
      undefined,
      undefined,
      requestReprocess,
      () => loadPassInProgress
    );
    const partition = new THREE.Group();
    const part = new THREE.Group();
    part.userData = {
      nodeType: 'points',
      visiblePointCount: 10,
      loadedViewVersion: 1,
      committedLadderComplete: true,
    };
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    expect(part.visible).toBe(false);
    expect(reg.isCaptureQuiescent()).toBe(true);

    partition.position.x = -2.5;
    reg.evaluatePerFrame();
    expect(part.visible).toBe(true);
    expect(requestReprocess).not.toHaveBeenCalled();
    expect(reg.isCaptureQuiescent()).toBe(false);

    loadPassInProgress = false;
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledWith(['/partition/part_0']);

    part.userData.loadedViewVersion = 2;
    expect(reg.isCaptureQuiescent()).toBe(false);

    loadPassInProgress = false;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('does not block forever on a pending partition resync after an archive fault', () => {
    let loadPassInProgress = true;
    let archiveFault = false;
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      () => archiveFault,
      vi.fn(),
      () => loadPassInProgress
    );
    const partition = new THREE.Group();
    const part = new THREE.Group();
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    partition.position.x = -2.5;
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(false);

    loadPassInProgress = false;
    expect(reg.isCaptureQuiescent()).toBe(false);
    archiveFault = true;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('does not wait on a pending partition resync while its wrapper is hidden', () => {
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      vi.fn(),
      () => true
    );
    const partition = new THREE.Group();
    const part = new THREE.Group();
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    partition.position.x = -2.5;
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(false);

    partition.visible = false;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('does not wait on a pending resync after its partition is removed', () => {
    let loadPassInProgress = true;
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      vi.fn(),
      () => loadPassInProgress
    );
    const partition = new THREE.Group();
    const part = new THREE.Group();
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [2, 0, 0], max: [3, 0.5, 0.5] },
        },
      ],
    });

    reg.evaluatePerFrame();
    partition.position.x = -2.5;
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(false);

    reg.unregister('/partition');
    loadPassInProgress = false;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('waits for a visible bare partition leaf to finish its progressive ladder', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 1);
    const partition = new THREE.Group();
    const part = new THREE.Group();
    part.userData = {
      nodeType: 'gsplats',
      visibleSplatCount: 100,
      loadedViewVersion: 1,
      committedLadderComplete: false,
    };
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [0, 0, 0], max: [0.5, 0.5, 0.5] },
        },
      ],
    });
    reg.evaluatePerFrame();

    expect(reg.isCaptureQuiescent()).toBe(false);
    part.userData.committedLadderComplete = true;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('does not wait on a stale partition leaf while its wrapper is hidden', () => {
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => 2);
    const partition = new THREE.Group();
    const part = new THREE.Group();
    part.userData = {
      nodeType: 'points',
      visiblePointCount: 10,
      loadedViewVersion: 1,
      committedLadderComplete: true,
    };
    partition.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject: partition,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [0, 0, 0], max: [0.5, 0.5, 0.5] },
        },
      ],
    });
    reg.evaluatePerFrame();

    expect(reg.isCaptureQuiescent()).toBe(false);
    partition.visible = false;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('reports quiescent for a settled single-level group', () => {
    const reg = makeRegistry();
    const children = [makeChild(0)];
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('reports quiescent for a multi-level group whose every level is already resident', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(0.5), makeChild(1.0)];
    reg.register(makeEntry(children, 0, '/g'));
    // Several passes so the selector reaches its steady state (a non-lazy
    // target swaps on the very frame it is desired).
    for (let f = 0; f < 3; f++) reg.evaluatePerFrame();
    const entry = reg.get('/g')!;
    expect(entry.desiredChildIndex).toBe(entry.activeChildIndex);
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  // ── The regression that motivates the whole predicate ──
  it('is NOT quiescent in the one-frame window after a lazy level loads but before the swap', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
    reg.register(makeEntry(children, 0, '/g'));
    // Lock so the test does not depend on projection math; the lock branch
    // writes ``desiredChildIndex`` exactly like the auto branches do.
    reg.setSelectorMode('/g', { lockLevel: 1 });

    reg.evaluatePerFrame(); // desired 1, not ready → kicks the load, no swap
    const entry = reg.get('/g')!;
    expect(entry.desiredChildIndex).toBe(1); // the selector WANTS the fine level
    expect(entry.activeChildIndex).toBe(0); // …but the aspiration only moves onto a READY level
    expect(reg.isCaptureQuiescent()).toBe(false); // in flight

    // The thunk lands: ready flips true and loading clears. The registry has
    // NOT swapped — that happens on the next selector pass.
    children[1].ready = true;
    children[1].loading = false;

    // Nothing is `loading`, level 0 is ready, and displayed === active === 0,
    // so a predicate reading only ready/loading/displayed would call this
    // "settled" and the capture would film the coarse level one frame before
    // the swap. `desiredChildIndex` is what closes that hole.
    expect(reg.isCaptureQuiescent()).toBe(false);

    reg.evaluatePerFrame(); // the swap
    expect(reg.get('/g')!.activeChildIndex).toBe(1);
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('is NOT quiescent while any child of an in-frame group is loading', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 0 }); // pin the coarse level: nothing to load
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(true); // coarse selected, nothing in flight

    // A load kicked for a level the selector is not (yet) aspiring to still
    // means the frame can change under the capture.
    children[1].loading = true;
    expect(reg.isCaptureQuiescent()).toBe(false);
  });

  it('is NOT quiescent when a fallback level is displayed instead of the aspiration', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeChild(0.5)];
    reg.register(makeEntry(children, 1, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(true);

    // Isolate the clause: everything else about the entry is settled, only the
    // DISPLAYED level differs from the aspiration — what a slice fallback or a
    // never-downgrade hold leaves on screen.
    reg.get('/g')!.displayedChildIndex = 0;
    expect(reg.isCaptureQuiescent()).toBe(false);
  });

  it('is NOT quiescent while a re-slice shows the coarse fallback for a stale aspiration', () => {
    // The same clause reached through a real evaluation pass: bump the view
    // version so the fine level's committed geometry is stale and the
    // slice-aware fallback displays the fresh coarse level instead.
    let version = 1;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => version);
    const coarse = makeGsplatChild(0, 1);
    const fine = makeGsplatChild(0.5, 1);
    reg.register(makeEntry([coarse, fine], 1, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(true);

    version = 2;
    (coarse.object.userData as { loadedViewVersion?: number }).loadedViewVersion = 2;
    reg.evaluatePerFrame();
    const entry = reg.get('/g')!;
    expect(entry.activeChildIndex).toBe(1);
    expect(entry.displayedChildIndex).toBe(0);
    expect(reg.isCaptureQuiescent()).toBe(false);
  });

  it('is NOT quiescent while the displayed level still has additive LODs to stream', () => {
    const reg = makeRegistry();
    const children = [makeChild(0)];
    children[0].hasMoreLODs = () => true;
    reg.register(makeEntry(children, 0, '/g'));
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(false); // only a prefix has committed

    children[0].hasMoreLODs = () => false; // ladder completes
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('is NOT quiescent while an EAGER leaf aspiration is still climbing its own ladder', () => {
    // `load-lod-group-node` attaches `hasMoreLODs` only on the DEFERRED path,
    // so the eagerly-loaded default level never has one — its ladder is
    // advanced by the sweep-driven background refinement loop instead. Reading
    // the thunk alone therefore declared a still-streaming coarse level
    // complete, and a Capture pressed before the initial load finished
    // exported its first frames at chunk-1. The commit stamp is the signal
    // that exists on this shape.
    const reg = makeRegistry();
    const leaf = makeChild(0);
    leaf.object.userData = {
      nodeType: 'gsplats',
      visibleSplatCount: 1000,
      committedLadderComplete: false,
    };
    expect(leaf.hasMoreLODs).toBeUndefined();
    reg.register(makeEntry([leaf], 0, '/g'));
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(false);

    leaf.object.userData.committedLadderComplete = true; // final chunk commits
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('does not block on a leaf that carries no ladder stamp at all', () => {
    // A non-progressive loader stamps nothing (`stamp-view-version` only writes
    // `committedLadderComplete` when it has a loader to ask), and an unstamped
    // leaf must read as complete rather than wedging the drain.
    const reg = makeRegistry();
    const leaf = makeChild(0);
    leaf.object.userData = { nodeType: 'points', visiblePointCount: 500 };
    reg.register(makeEntry([leaf], 0, '/g'));
    reg.evaluatePerFrame();
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  /**
   * A deferred-GROUP LOD child, as ``loadLodGroupNode``'s ``canDeferGroup``
   * path builds it: a placeholder ``THREE.Group`` holding a loaded subtree, and
   * — decisively — NO ``hasMoreLODs`` thunk (that call site passes only five
   * arguments). Its one stamped part leaf carries the ladder-completeness stamp
   * the fold reads.
   */
  function makeDeferredGroupChild(
    coverageFraction: number,
    ladderComplete: boolean
  ): LODGroupChild {
    const group = new THREE.Group();
    const partLeaf = new THREE.Group();
    partLeaf.userData = {
      nodeType: 'gsplats',
      visibleSplatCount: 1000,
      committedLadderComplete: ladderComplete,
    };
    group.add(partLeaf);
    return {
      object: group,
      coverageFraction,
      positionBounds: { min: [0, 0, 0], max: [10, 10, 10] },
      ready: true,
    };
  }

  it('is NOT quiescent while a deferred GROUP aspiration sits at chunk-1 of its parts ladders', () => {
    // The `overview` shape: the fine `kind=partition` branch is a GROUP child
    // with no `hasMoreLODs` thunk, so the leaf-only completeness clause is a
    // no-op for it — yet its part leaves are at chunk-1 by construction the
    // moment the branch loads (the deferred-group call site kicks refinement
    // precisely because of that). Ready, fresh, nothing loading: without the
    // subtree fold the predicate calls this settled and the frame is exported
    // at the first additive chunk, refining in over the next seconds.
    const reg = makeRegistry();
    const child = makeDeferredGroupChild(0, false);
    reg.register(makeEntry([child], 0, '/ov'));
    reg.evaluatePerFrame();
    // Documents the shape this case is about — a deferred GROUP child carries
    // no ladder thunk — but only for the literal built above. It pins nothing
    // about production: the deferred-group call site in the node factory is
    // free to start passing a `hasMoreLODs`, and this assertion would stay
    // green while the case below stopped exercising the subtree fold.
    expect(child.hasMoreLODs).toBeUndefined();
    expect(reg.isCaptureQuiescent()).toBe(false);

    // Control: the same subtree with its ladders complete IS quiescent, so the
    // false above is the completeness fold and not some unrelated blocker.
    (
      child.object.children[0].userData as { committedLadderComplete?: boolean }
    ).committedLadderComplete = true;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('treats an entry with no child at the aspiration index as nothing to wait for', () => {
    // `children` is legitimately EMPTY when every level failed its
    // `getObjectByName` attach in `load-lod-group-node` (it warns and carries
    // on). Nothing at a missing index can ever become ready, so blocking would
    // make the predicate permanently false: each frame burns the whole drain
    // budget (2 s / 120 rAFs) until three in a row latch draining off — ~6 s
    // spent, and the run then reports frames filmed at a coarse LOD that no
    // level was ever going to improve.
    const reg = makeRegistry();
    reg.register(makeEntry([], 0, '/broken'));
    expect(reg.isCaptureQuiescent()).toBe(true);

    // The same trap through an out-of-range aspiration on a populated entry.
    const reg2 = makeRegistry();
    const children = [makeChild(0)];
    const entry = makeEntry(children, 0, '/g');
    reg2.register(entry);
    reg2.evaluatePerFrame();
    expect(reg2.isCaptureQuiescent()).toBe(true);
    entry.activeChildIndex = 7;
    entry.displayedChildIndex = 7;
    expect(reg2.isCaptureQuiescent()).toBe(true);

    // …and through an out-of-range DESIRED index, which is the same shape one
    // clause earlier: no child there either, so nothing to block on.
    entry.activeChildIndex = 0;
    entry.displayedChildIndex = 0;
    entry.desiredChildIndex = 7;
    expect(reg2.isCaptureQuiescent()).toBe(true);
  });

  it('excludes an OFF-SCREEN group that is deliberately held at its coarse level', () => {
    const reg = makeRegistry();
    const coarse = { ...makeChild(0), positionBounds: FAR_BOUNDS };
    const fine = { ...makeLazyChild(0.5, () => {}), positionBounds: FAR_BOUNDS };
    fine.loading = true; // a load kicked before the tile left the frustum
    const entry = makeEntry([coarse, fine], 0, '/g');
    reg.register(entry);

    reg.evaluatePerFrame();
    expect(entry.offScreen).toBe(true);
    // Skipped entirely: an off-screen group draws nothing this frame and is
    // held coarse on purpose, so waiting for its fine level only times out.
    expect(reg.isCaptureQuiescent()).toBe(true);

    // Control — the exclusion is what makes it quiescent, not the entry being
    // vacuously settled: the same entry on screen would block on the load.
    entry.offScreen = false;
    expect(reg.isCaptureQuiescent()).toBe(false);
  });

  it('excludes a group hidden by an ancestor, whose desired level can never load', () => {
    // The permanently-false trap. `kickDeferredLoadIfVisible` refuses to START
    // a deferred load while the group is effectively hidden, but the
    // selector's frustum test is pure geometry and still records a fine
    // `desiredChildIndex`. So `desired !== active` with nothing loading,
    // nothing failing and nothing ever becoming ready: without the
    // visibility skip the predicate could never be satisfied again, and every
    // capture frame would burn its whole drain budget before giving up.
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
    const entry = makeEntry(children, 0, '/g');
    // The hidden flag sits on an ANCESTOR (a layer toggled off in the panel),
    // not on the lod_group itself — which is why the check has to be the
    // ancestor-aware one.
    const layer = new THREE.Group();
    layer.visible = false;
    layer.add(entry.groupObject);
    reg.register(entry);
    reg.setSelectorMode('/g', { lockLevel: 1 });

    reg.evaluatePerFrame();
    expect(entry.desiredChildIndex).toBe(1); // the selector wants the fine level
    expect(entry.activeChildIndex).toBe(0); // …which never loaded
    expect(children[1].loading).toBeFalsy(); // the load gate refused to start it
    expect(children[1].failed).toBeFalsy(); // …so it cannot fail out either
    expect(reg.isCaptureQuiescent()).toBe(true);

    // Control — the exclusion is what makes it quiescent, not the entry being
    // vacuously settled: the same entry visible blocks on the pending level.
    layer.visible = true;
    expect(reg.isCaptureQuiescent()).toBe(false);
  });

  it('treats an archive-faulted scene as settled until automatic loading resumes', () => {
    let hasArchiveFault = true;
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      () => hasArchiveFault
    );
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
    const entry = makeEntry(children, 0, '/g');
    reg.register(entry);
    reg.setSelectorMode('/g', { lockLevel: 1 });

    reg.evaluatePerFrame();
    expect(entry.desiredChildIndex).toBe(1);
    expect(entry.activeChildIndex).toBe(0);
    expect(children[1].loading).toBeFalsy();
    expect(reg.isCaptureQuiescent()).toBe(true);

    children[1].loading = true;
    expect(reg.isCaptureQuiescent()).toBe(false);

    children[1].loading = false;
    hasArchiveFault = false;
    expect(reg.isCaptureQuiescent()).toBe(false);
  });

  it('is NOT quiescent while the DISPLAYED aspiration is stale for the current view version', () => {
    // Isolates the freshness clause: unlike the re-slice test above (which
    // returns false two clauses earlier, at displayed !== active), this group
    // has a single level, so the slice-aware fallback has nothing coarser to
    // fall back to and keeps displaying the aspiration. Only the stale stamp
    // distinguishes the two answers — delete the freshness check and this
    // test goes green on a group that is showing the previous slice.
    let version = 1;
    const reg = makeRegistry([0, 1, 2], undefined, undefined, () => version);
    const only = makeGsplatChild(0, 1);
    const entry = makeEntry([only], 0, '/g');
    reg.register(entry);
    reg.evaluatePerFrame();
    expect(entry.displayedChildIndex).toBe(entry.activeChildIndex);
    expect(reg.isCaptureQuiescent()).toBe(true);

    // A slice/displayDims scrub: the committed geometry now describes the
    // previous view version, and nothing has re-committed it yet.
    version = 2;
    reg.evaluatePerFrame();
    expect(entry.displayedChildIndex).toBe(entry.activeChildIndex); // same clause NOT hit
    expect(entry.desiredChildIndex).toBe(entry.activeChildIndex); // …nor this one
    expect(only.loading).toBeFalsy(); // …nor the loading one
    expect(reg.isCaptureQuiescent()).toBe(false);

    // The re-commit lands with a fresh stamp → settled again.
    (only.object.userData as { loadedViewVersion?: number }).loadedViewVersion = 2;
    expect(reg.isCaptureQuiescent()).toBe(true);
  });

  it('does not block forever on a desired level that FAILED to load', () => {
    const reg = makeRegistry();
    const children = [makeChild(0), makeLazyChild(0.5, () => {})];
    reg.register(makeEntry(children, 0, '/g'));
    reg.setSelectorMode('/g', { lockLevel: 1 });
    reg.evaluatePerFrame(); // kicks the load
    expect(reg.isCaptureQuiescent()).toBe(false);

    // The thunk fails: ready never flips, loading clears, failed is set.
    children[1].failed = true;
    children[1].loading = false;
    reg.evaluatePerFrame();

    const entry = reg.get('/g')!;
    expect(entry.desiredChildIndex).toBe(1); // the selector still WANTS it
    expect(entry.activeChildIndex).toBe(0);
    // …but a failed level can never become ready this frame, so blocking on it
    // would buy a timeout on every remaining capture frame.
    expect(reg.isCaptureQuiescent()).toBe(true);
  });
});

describe('LODGroupRegistry — stale-hold over a far coarser fallback', () => {
  /** A registry whose clock and view version the test drives. */
  function makeScrubRig(coarseCount: number, fineCount: number) {
    const state = { version: 1, clock: 0 };
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      () => state.version,
      undefined,
      () => state.clock
    );
    const coarse = makeCountedChild(0, 1, coarseCount);
    const fine = makeCountedChild(0.5, 1, fineCount);
    reg.register(makeEntry([coarse, fine], 0, '/g'));
    reg.evaluatePerFrame();
    return { reg, coarse, fine, state };
  }

  /** The sweep recommitting a child for `version`, keeping its count. */
  function recommit(
    child: ReturnType<typeof makeCountedChild>,
    version: number,
    count: number,
    complete = true
  ) {
    Object.assign(child.object.userData!, {
      loadedViewVersion: version,
      visibleSplatCount: count,
      committedLadderComplete: complete,
    });
  }

  it('holds the previous slice instead of flashing down to a token of it', () => {
    // The reported defect: stepping the Time slider on a 4D timelapse dropped
    // the finest level (6,900 splats) to the coarsest (108) for ~70 ms and back
    // — every step — while the fine level's data was already cached.
    const { reg, coarse, fine, state } = makeScrubRig(100, 1000);
    expect(fine.object.visible).toBe(true);

    state.version = 2; // scrub: both stale, coarse is 10% of fine
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);

    // Even once the cheap coarse level recommits fresh, the far better stale
    // fine level stays up — it is the previous FRAME, not a previous view.
    recommit(coarse, 2, 100);
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);

    // ...until the fine level lands for the new slice, which ends the hold.
    recommit(fine, 2, 1000);
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
  });

  it('shows the fresh fallback when the remembered level is no longer ready', () => {
    // A held level can be evicted while it is remembered but off-screen. Its
    // commit stamps survive release, so readiness — not the count ratio — must
    // prevent the registry from selecting geometry that cannot be drawn.
    const { reg, coarse, fine, state } = makeScrubRig(100, 1000);
    fine.ready = false;
    state.version = 2;
    recommit(coarse, 2, 100);

    reg.evaluatePerFrame();

    expect(reg.get('/g')!.displayedChildIndex).toBe(0);
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(false);
  });

  it('hands a partial aspiration back to the never-downgrade gate after a hold', () => {
    const { reg, coarse, fine, state } = makeScrubRig(400, 1000);

    state.version = 2;
    reg.evaluatePerFrame(); // hold the stale fine level
    recommit(coarse, 2, 400);
    reg.evaluatePerFrame(); // keep holding over the fresh coarse fallback

    // The fine aspiration ends the stale hold with a chunk-1 prefix smaller
    // than the complete coarse fallback. The ordinary never-downgrade gate
    // must still hold coarse until the fine ladder catches up.
    recommit(fine, 2, 100, false);
    reg.evaluatePerFrame();

    expect(reg.get('/g')!.displayedChildIndex).toBe(0);
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(false);

    recommit(fine, 2, 400, false); // count crossover releases the ordinary gate
    reg.evaluatePerFrame();
    expect(reg.get('/g')!.displayedChildIndex).toBe(1);
    expect(coarse.object.visible).toBe(false);
    expect(fine.object.visible).toBe(true);
  });

  it('takes the fresh fallback immediately when it is comparably good', () => {
    // Freshness normally wins: only a SEVERE downgrade justifies showing the
    // wrong slice. At 60% of the fine level the fallback is taken at once.
    const { reg, coarse, fine, state } = makeScrubRig(600, 1000);
    state.version = 2;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(false);
  });

  it('gives up after the budget so a long scrub shows live coarse geometry', () => {
    const { reg, coarse, fine, state } = makeScrubRig(100, 1000);
    state.version = 2;
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true); // holding

    state.clock += 260; // past STALE_HOLD_MS
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);
    expect(fine.object.visible).toBe(false);
  });

  it('does not re-arm the hold on every version bump during a drag', () => {
    // The budget is spent from when the hold STARTS, not from the last version
    // change — otherwise a drag (a new version every frame) would re-hold
    // forever and freeze the display on one stale frame.
    const { reg, coarse, fine, state } = makeScrubRig(100, 1000);
    state.version = 2;
    reg.evaluatePerFrame();
    state.clock += 260;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // gave up

    for (let i = 0; i < 5; i++) {
      state.version += 1; // keep scrubbing
      state.clock += 16;
      reg.evaluatePerFrame();
      expect(coarse.object.visible).toBe(true);
      expect(fine.object.visible).toBe(false);
    }
  });

  it('re-arms once the aspiration lands, so the NEXT step is held again', () => {
    const { reg, coarse, fine, state } = makeScrubRig(100, 1000);
    state.version = 2;
    reg.evaluatePerFrame();
    state.clock += 260;
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true); // budget spent

    recommit(fine, 2, 1000); // the wait ends
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);

    state.version = 3; // next step is held again, not flashed
    state.clock += 16;
    reg.evaluatePerFrame();
    expect(fine.object.visible).toBe(true);
    expect(coarse.object.visible).toBe(false);
  });

  it('never holds a level COARSER than the fallback, even with inverted counts', () => {
    // Holding something coarser than the fallback would be a downgrade — the
    // opposite of the point. With CONSISTENT data the count ratio already
    // declines that (a coarser level has fewer elements), so this pins the
    // ordering guard on its own by inverting the counts: the sort of
    // inconsistent store the fresh-but-empty guard above also exists for.
    const state = { version: 1, clock: 0 };
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      () => state.version,
      undefined,
      () => state.clock
    );
    const coarse = makeCountedChild(0, 1, 5000); // MORE than the fine level
    // A threshold this camera's coverage never reaches, so the selector really
    // settles on index 0 rather than promoting the finer level.
    const fine = makeCountedChild(50, 1, 1000);
    reg.register(makeEntry([coarse, fine], 0, '/g')); // index 0 displayed
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(true);

    // The FINER level is the fresh fallback. Its count (1000) is far below the
    // displayed coarse level's (5000), so the ratio test alone would hold —
    // only the index ordering stops it.
    state.version = 2;
    recommit(fine, 2, 1000);
    reg.evaluatePerFrame();
    expect(coarse.object.visible).toBe(false);
    expect(fine.object.visible).toBe(true);
  });

  it('stays given up when the fallback moves COARSER mid-scrub', () => {
    // Why the exhausted latch is not redundant with the index ordering. Once
    // the budget is spent the display drops to the fallback and the gate memory
    // follows it, so ordering alone blocks a re-hold — until the fallback moves
    // to a COARSER level (levels refreshing out of order during a drag), at
    // which point the memory is finer than the fallback again and the hold
    // would re-arm, freezing the display for another budget.
    const state = { version: 1, clock: 0 };
    const reg = makeRegistry(
      [0, 1, 2],
      undefined,
      undefined,
      () => state.version,
      undefined,
      () => state.clock
    );
    const l0 = makeCountedChild(0, 1, 100);
    const l1 = makeCountedChild(0.25, 1, 500);
    const l2 = makeCountedChild(0.5, 1, 5000);
    reg.register(makeEntry([l0, l1, l2], 2, '/g'));
    reg.evaluatePerFrame();
    expect(l2.object.visible).toBe(true);

    // Scrub: everything stale → hold l2, then spend the budget. l1 is the
    // coarsest READY level here, so it becomes the fallback and the memory.
    state.version = 2;
    reg.evaluatePerFrame();
    recommit(l1, 2, 500);
    state.clock += 260;
    reg.evaluatePerFrame();
    expect(l1.object.visible).toBe(true);

    // Now l0 refreshes for a newer version and l1 falls behind: the fallback
    // moves from index 1 to index 0, so the memory (1) is finer than it again.
    state.version = 3;
    recommit(l0, 3, 100);
    state.clock += 16;
    reg.evaluatePerFrame();
    expect(l0.object.visible).toBe(true); // still given up, not re-held
    expect(l1.object.visible).toBe(false);
  });
});
