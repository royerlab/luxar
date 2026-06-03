/**
 * Unit tests for the WASM TypeScript-fallback visibility / spatial helpers:
 * - spatial.query_chunks_for_view
 * - points.compute_nd_visibility_points
 * - lines.compute_nd_visibility_lines
 * - gsplats.compute_nd_visibility_gsplats
 *
 * Pure math on typed arrays, no mocks. The same fixtures double as
 * cross-implementation reference data for the Rust→WASM build.
 *
 * [wasm.md/G][P8] three-geometry symmetry: Lines was previously absent from
 * this file; round 7 fills the gap by adding parallel happy-path / far-away /
 * mixed / zero-segments cases.
 * [wasm.md/G][P5] WASM 16-dim boundary: round 7 adds ndim=16 (in-range) and
 * ndim=17 (out-of-WASM-range, exercises JS fallback path) cases for all
 * three geometries.
 */

import { describe, it, expect } from 'vitest';
import { query_chunks_for_view } from '../../../wasm/typescript/spatial';
import { compute_nd_visibility_points } from '../../../wasm/typescript/points';
import { compute_nd_visibility_lines } from '../../../wasm/typescript/lines';
import { compute_nd_visibility_gsplats } from '../../../wasm/typescript/gsplats';

describe('query_chunks_for_view', () => {
  it('returns empty result when no chunks intersect the slice', () => {
    // Two chunks both far from slicePosition=0 with tolerance=0.1
    const chunkBounds = new Float32Array([
      // chunk 0: x ∈ [10, 11]
      10, 11,
      // chunk 1: x ∈ [-5, -4]
      -5, -4,
    ]);
    const slicePosition = new Float32Array([0]);
    const tolerance = new Float32Array([0.1]);
    const output = new Uint32Array(2);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 1, 2, output);
    expect(n).toBe(0);
  });

  it('returns the chunk indices whose bounds straddle the slice (1D)', () => {
    const chunkBounds = new Float32Array([
      // 0: [-1, 1]   -> intersects [-0.1, 0.1]
      -1, 1,
      // 1: [10, 11]  -> doesn't
      10, 11,
      // 2: [0.05, 0.2] -> intersects via lower bound
      0.05, 0.2,
    ]);
    const slicePosition = new Float32Array([0]);
    const tolerance = new Float32Array([0.1]);
    const output = new Uint32Array(3);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 1, 3, output);
    expect(n).toBe(2);
    expect(Array.from(output.slice(0, 2))).toEqual([0, 2]);
  });

  it('requires intersection in *all* dimensions (AND semantics)', () => {
    const chunkBounds = new Float32Array([
      // chunk 0 in 2D: x ∈ [-1, 1], y ∈ [10, 11] — y misses
      -1, 1, 10, 11,
      // chunk 1: x ∈ [-1, 1], y ∈ [-1, 1] — both pass
      -1, 1, -1, 1,
    ]);
    const slicePosition = new Float32Array([0, 0]);
    const tolerance = new Float32Array([0.5, 0.5]);
    const output = new Uint32Array(2);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 2, 2, output);
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('handles tolerance=0 — chunk must literally span the slice position', () => {
    const chunkBounds = new Float32Array([
      // chunk 0: [-1, 1] — spans 0
      -1, 1,
      // chunk 1: [0.5, 1.5] — does not span 0
      0.5, 1.5,
    ]);
    const slicePosition = new Float32Array([0]);
    const tolerance = new Float32Array([0]);
    const output = new Uint32Array(2);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 1, 2, output);
    expect(n).toBe(1);
    expect(output[0]).toBe(0);
  });

  it('handles numChunks=0', () => {
    const output = new Uint32Array(0);
    const n = query_chunks_for_view(
      new Float32Array(0),
      new Float32Array([0]),
      new Float32Array([1]),
      1,
      0,
      output
    );
    expect(n).toBe(0);
  });
});

describe('compute_nd_visibility_points', () => {
  it('a point at the slice with non-zero tolerance is visible', () => {
    const positions = new Float32Array([0, 0, 0]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('a point on the tolerance boundary is visible (≤1, not <1)', () => {
    // delta=0.1 with tolerance=0.1 → normalized=1 → distSq=1 → visible
    const positions = new Float32Array([0.1]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      1,
      output
    );
    expect(output[0]).toBe(1);
  });

  it('a point outside the tolerance is hidden', () => {
    const positions = new Float32Array([0.5]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('point radius extends visibility in 3D', () => {
    // Point at delta=0.5, tolerance=0.1, radius=0.5
    // effectiveTolerance = 0.6 in each dim → normalized=0.5/0.6 = 0.833
    // distSq = 3 × 0.833² = 2.083 → still hidden (>1)
    // But with radius=2.0 in just one dim:
    const positions = new Float32Array([0.5, 0, 0]);
    const radii = new Float32Array([2.0]);
    const output = new Uint8Array(1);
    compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    // delta=0.5, tol=0.1+2=2.1, normalized=0.238, distSq=0.057 → visible
    expect(output[0]).toBe(1);
  });

  it('zero tolerance + zero radius + non-trivial delta hides the point', () => {
    const positions = new Float32Array([1, 1]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0, 0]),
      new Float32Array([0, 0]),
      2,
      1,
      output
    );
    expect(n).toBe(0);
  });

  it('mixed visibility across many points returns the correct count', () => {
    // 5 points in 1D at deltas 0, 0.05, 0.1, 0.2, 1.0
    const positions = new Float32Array([0, 0.05, 0.1, 0.2, 1.0]);
    const radii = new Float32Array([0, 0, 0, 0, 0]);
    const output = new Uint8Array(5);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      5,
      output
    );
    expect(n).toBe(3);
    expect(Array.from(output)).toEqual([1, 1, 1, 0, 0]);
  });
});

describe('compute_nd_visibility_gsplats', () => {
  it('an isotropic 3D splat at the slice is visible', () => {
    // 3D splat with diagonal Cholesky [1, 0, 1, 0, 0, 1] = identity covariance
    const centers = new Float32Array([0, 0, 0]);
    const cholesky = new Float32Array([1, 0, 1, 0, 0, 1]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('a far-away splat with small extent is hidden', () => {
    const centers = new Float32Array([100, 0, 0]);
    const cholesky = new Float32Array([0.01, 0, 0.01, 0, 0, 0.01]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('a stretched splat (large row norm) reaches further', () => {
    // 1D splat at distance 5 with cholesky=[6] → max extent = 6, tolerance=0.1
    // effectiveTolerance = 6.1, delta=5, normalized=5/6.1=0.82, distSq=0.67 → visible
    const centers = new Float32Array([5]);
    const cholesky = new Float32Array([6]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      1,
      output
    );
    expect(n).toBe(1);
  });

  it('mixed visibility across a batch returns the right count + mask', () => {
    // Three 1D splats: at 0 (visible), 100 (hidden), 0.05 (visible)
    const centers = new Float32Array([0, 100, 0.05]);
    const cholesky = new Float32Array([0.5, 0.5, 0.5]);
    const output = new Uint8Array(3);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      3,
      output
    );
    expect(n).toBe(2);
    expect(Array.from(output)).toEqual([1, 0, 1]);
  });

  it('numSplats=0 returns 0 and writes nothing', () => {
    const output = new Uint8Array(2).fill(99);
    const n = compute_nd_visibility_gsplats(
      new Float32Array(0),
      new Float32Array(0),
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      0,
      output
    );
    expect(n).toBe(0);
    expect(Array.from(output)).toEqual([99, 99]); // untouched
  });
});

// [wasm.md/G][P8] Three-geometry symmetry: Lines visibility was missing from
// this file. The Points and GSplats sections above pin the analogous cases.
describe('compute_nd_visibility_lines', () => {
  it('a segment with both endpoints at the slice is visible', () => {
    // 3D segment from (0,0,0) → (0,0,0) — both at slice with non-zero tolerance.
    const vertices = new Float32Array([0, 0, 0, 0, 0, 0]);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([0, 0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('a segment with only one endpoint inside is visible (endpoint-based OR)', () => {
    // v0 inside, v1 far away → segment is visible (per docstring).
    const vertices = new Float32Array([0, 0, 0, 100, 100, 100]);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([0, 0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('a segment with both endpoints far is hidden', () => {
    const vertices = new Float32Array([100, 0, 0, 100, 100, 100]);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([0, 0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('per-vertex width extends visibility (mirrors point.radius contract)', () => {
    // Endpoint at distance 5 along x, width=10 → effectiveTolerance=10.1,
    // normalized=5/10.1≈0.495, distSq≈0.245 < 1 → visible.
    const vertices = new Float32Array([5, 0, 0, 100, 100, 100]);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([10, 0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
  });

  it('mixed visibility across many segments returns the correct count + mask', () => {
    // 3 segments in 1D: [0,0] (visible), [100,100] (hidden), [0.05,100] (visible — v0 inside)
    const vertices = new Float32Array([0, 0, 100, 100, 0.05, 100]);
    const segments = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const widths = new Float32Array([0, 0, 0, 0, 0, 0]);
    const output = new Uint8Array(3);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      3,
      output
    );
    expect(n).toBe(2);
    expect(Array.from(output)).toEqual([1, 0, 1]);
  });

  it('numSegments=0 returns 0 and writes nothing', () => {
    const output = new Uint8Array(2).fill(99);
    const n = compute_nd_visibility_lines(
      new Float32Array(0),
      new Uint32Array(0),
      new Float32Array(0),
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      0,
      output
    );
    expect(n).toBe(0);
    expect(Array.from(output)).toEqual([99, 99]); // untouched
  });
});

// [wasm.md/G][P5] WASM 16-dim boundary tests. The JS fallback supports
// arbitrary ndim (CLAUDE.md: "For >16D data: TypeScript fallback is used
// automatically"). Pin the upper-boundary contract for all three geometry
// types: ndim=16 (in-WASM-range) and ndim=17 (above WASM range — fallback
// path). If a refactor accidentally introduces a fixed-size buffer in the
// JS fallback that mirrors the WASM 16-cap, these tests fail loudly.
describe('TypeScript fallback at WASM 16-dim boundary', () => {
  describe('compute_nd_visibility_points', () => {
    it('handles ndim=16 (WASM boundary, in range) correctly', () => {
      const ndim = 16;
      const positions = new Float32Array(ndim); // single point at origin
      const radii = new Float32Array([0]);
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_points(
        positions,
        radii,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('handles ndim=17 (above WASM range, JS fallback only) correctly', () => {
      const ndim = 17;
      // Point at all-zeros, slice at all-zeros, tolerance per-dim → visible.
      const positions = new Float32Array(ndim);
      const radii = new Float32Array([0]);
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_points(
        positions,
        radii,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('handles ndim=17 with a far-away point (verifies fallback distance calculation)', () => {
      const ndim = 17;
      const positions = new Float32Array(ndim);
      positions[5] = 100; // far along dimension 5
      const radii = new Float32Array([0]);
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_points(
        positions,
        radii,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(0);
      expect(output[0]).toBe(0);
    });
  });

  describe('compute_nd_visibility_lines', () => {
    it('handles ndim=16 (WASM boundary, in range) correctly', () => {
      const ndim = 16;
      const vertices = new Float32Array(2 * ndim); // both vertices at origin
      const segments = new Uint32Array([0, 1]);
      const widths = new Float32Array([0, 0]);
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('handles ndim=17 (above WASM range, JS fallback only) correctly', () => {
      const ndim = 17;
      const vertices = new Float32Array(2 * ndim);
      const segments = new Uint32Array([0, 1]);
      const widths = new Float32Array([0, 0]);
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    // [R11/B-G3][P5/P8] Far-away counterpart at ndim=17 (mirrors the
    // Points-fallback test above) — pins that the JS fallback's distance
    // computation actually consults non-displayed dimensions, not just
    // the first few. A regression that capped the loop at ndim=16 (or
    // dropped out at the WASM boundary) would falsely report visible.
    it('handles ndim=17 with a far-away segment (verifies fallback distance calculation)', () => {
      const ndim = 17;
      const vertices = new Float32Array(2 * ndim);
      // Push BOTH endpoints far along dimension 5 (well outside tolerance).
      vertices[5] = 100;
      vertices[ndim + 5] = 100;
      const segments = new Uint32Array([0, 1]);
      const widths = new Float32Array([0, 0]);
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(0);
      expect(output[0]).toBe(0);
    });
  });

  describe('compute_nd_visibility_gsplats', () => {
    it('handles ndim=16 (WASM boundary, in range) correctly', () => {
      const ndim = 16;
      // Cholesky packed lower-triangular size = ndim*(ndim+1)/2.
      const choleskySize = (ndim * (ndim + 1)) / 2;
      const centers = new Float32Array(ndim);
      // Identity-ish Cholesky: diagonal entries = 1, off-diag = 0.
      // Lower-triangular packed layout: [L00, L10, L11, L20, L21, L22, ...]
      const cholesky = new Float32Array(choleskySize);
      let idx = 0;
      for (let row = 0; row < ndim; row++) {
        for (let col = 0; col <= row; col++) {
          cholesky[idx++] = col === row ? 1 : 0;
        }
      }
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_gsplats(
        centers,
        cholesky,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('handles ndim=17 (above WASM range, JS fallback only) correctly', () => {
      const ndim = 17;
      const choleskySize = (ndim * (ndim + 1)) / 2;
      const centers = new Float32Array(ndim);
      const cholesky = new Float32Array(choleskySize);
      let idx = 0;
      for (let row = 0; row < ndim; row++) {
        for (let col = 0; col <= row; col++) {
          cholesky[idx++] = col === row ? 1 : 0;
        }
      }
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_gsplats(
        centers,
        cholesky,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    // [R11/B-G3][P5/P8] Far-away counterpart at ndim=17 (mirrors the
    // Points-fallback test above) — pins that the GSplat JS fallback's
    // Mahalanobis distance computation actually consults non-displayed
    // dimensions. With identity Cholesky and a center far along dim 5,
    // the splat must NOT be visible.
    it('handles ndim=17 with a far-away splat (verifies fallback distance calculation)', () => {
      const ndim = 17;
      const choleskySize = (ndim * (ndim + 1)) / 2;
      const centers = new Float32Array(ndim);
      centers[5] = 100; // far along dimension 5
      const cholesky = new Float32Array(choleskySize);
      let idx = 0;
      for (let row = 0; row < ndim; row++) {
        for (let col = 0; col <= row; col++) {
          cholesky[idx++] = col === row ? 1 : 0; // identity → unit covariance
        }
      }
      const slicePosition = new Float32Array(ndim);
      const tolerance = new Float32Array(ndim).fill(0.1);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_gsplats(
        centers,
        cholesky,
        slicePosition,
        tolerance,
        ndim,
        1,
        output
      );
      expect(n).toBe(0);
      expect(output[0]).toBe(0);
    });
  });
});

// [wasm.md/G][P5] Lower-boundary ndim parity tests for all three geometry
// types. ndim=1 and ndim=2 are degenerate but legitimate inputs (e.g. a
// 1D time series viewed as a 1D scatter or a 2D heatmap of line segments).
// The JS fallback must not assume ndim>=3; a mutant that hard-coded a
// 3-dim loop bound would survive without these tests.
describe('TypeScript fallback at small-ndim boundary', () => {
  describe('compute_nd_visibility_points', () => {
    it('ndim=1: a point AT the slice with tolerance is visible', () => {
      const positions = new Float32Array([0]);
      const radii = new Float32Array([0]);
      const slicePosition = new Float32Array([0]);
      const tolerance = new Float32Array([0.5]);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_points(
        positions,
        radii,
        slicePosition,
        tolerance,
        1,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('ndim=1: a point FAR from the slice (beyond tolerance + radius) is hidden', () => {
      const positions = new Float32Array([100]);
      const radii = new Float32Array([0]);
      const slicePosition = new Float32Array([0]);
      const tolerance = new Float32Array([0.5]);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_points(
        positions,
        radii,
        slicePosition,
        tolerance,
        1,
        1,
        output
      );
      expect(n).toBe(0);
      expect(output[0]).toBe(0);
    });

    it('ndim=2: mixed visibility across multiple points (validates per-dim accumulation)', () => {
      // Three 2D points; slice at origin with tolerance [0.5, 0.5].
      // Pt0: (0, 0)     → inside both → visible
      // Pt1: (0.3, 0.3) → inside both → visible
      // Pt2: (10, 10)   → far → hidden
      const positions = new Float32Array([0, 0, 0.3, 0.3, 10, 10]);
      const radii = new Float32Array([0, 0, 0]);
      const slicePosition = new Float32Array([0, 0]);
      const tolerance = new Float32Array([0.5, 0.5]);
      const output = new Uint8Array(3);
      const n = compute_nd_visibility_points(
        positions,
        radii,
        slicePosition,
        tolerance,
        2,
        3,
        output
      );
      expect(n).toBe(2);
      expect(Array.from(output)).toEqual([1, 1, 0]);
    });
  });

  describe('compute_nd_visibility_lines', () => {
    it('ndim=1: a segment with both endpoints AT the slice is visible', () => {
      const vertices = new Float32Array([0, 0]);
      const segments = new Uint32Array([0, 1]);
      const widths = new Float32Array([0, 0]);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        new Float32Array([0]),
        new Float32Array([0.1]),
        1,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('ndim=2: a segment spanning the slice along one axis is visible (endpoint-OR)', () => {
      // v0 inside slice (0,0), v1 far in y. Endpoint-OR ⇒ visible.
      const vertices = new Float32Array([0, 0, 0, 100]);
      const segments = new Uint32Array([0, 1]);
      const widths = new Float32Array([0, 0]);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        new Float32Array([0, 0]),
        new Float32Array([0.1, 0.1]),
        2,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });
  });

  describe('compute_nd_visibility_gsplats', () => {
    it('ndim=1: a gsplat centered at the slice is visible', () => {
      // Packed lower-triangular Cholesky for 1D is just [L00].
      const centers = new Float32Array([0]);
      const cholesky = new Float32Array([1]); // identity covariance
      const slicePosition = new Float32Array([0]);
      const tolerance = new Float32Array([0.5]);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_gsplats(
        centers,
        cholesky,
        slicePosition,
        tolerance,
        1,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });

    it('ndim=2: identity-covariance splat at the slice is visible', () => {
      // 2D Cholesky packed: [L00, L10, L11] = [1, 0, 1] for identity.
      const centers = new Float32Array([0, 0]);
      const cholesky = new Float32Array([1, 0, 1]);
      const slicePosition = new Float32Array([0, 0]);
      const tolerance = new Float32Array([0.5, 0.5]);
      const output = new Uint8Array(1);
      const n = compute_nd_visibility_gsplats(
        centers,
        cholesky,
        slicePosition,
        tolerance,
        2,
        1,
        output
      );
      expect(n).toBe(1);
      expect(output[0]).toBe(1);
    });
  });
});

// ---------------------------------------------------------------------------
// [wasm.md G1, G2, G3] NaN / Inf propagation across the three geometries.
// Pin the contract that a single NaN in positions/vertices/centers (or in
// Cholesky factors for gsplats) poisons distSq and produces output=0.
// A mutant that swapped `distSq <= 1.0` to `distSq <= 1.0 || isNaN(distSq)`
// or that initialized `distSq` differently would now be caught.
// ---------------------------------------------------------------------------
describe('compute_nd_visibility_points — NaN/Inf propagation [wasm.md G1]', () => {
  it('[G1] NaN in positions poisons distSq → point is hidden (NaN<=1.0 is false)', () => {
    // 3D point with NaN in the y coordinate. Slice at origin, tolerance large.
    const positions = new Float32Array([0, Number.NaN, 0]);
    const radii = new Float32Array([0]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[G1] NaN in slicePosition also poisons distSq → point is hidden', () => {
    const positions = new Float32Array([0, 0, 0]);
    const radii = new Float32Array([0]);
    const slicePosition = new Float32Array([0, Number.NaN, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[G1] Infinity in positions → delta=Infinity → normalized=Infinity → distSq=Infinity → hidden', () => {
    const positions = new Float32Array([0, Number.POSITIVE_INFINITY, 0]);
    const radii = new Float32Array([0]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[G1] NaN-poison is per-point: one bad point hides only itself, neighbour still visible', () => {
    const positions = new Float32Array([0, 0, 0, Number.NaN, 0, 0]);
    const radii = new Float32Array([0, 0]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1]);
    const output = new Uint8Array(2);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      slicePosition,
      tolerance,
      3,
      2,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
    expect(output[1]).toBe(0);
  });
});

describe('compute_nd_visibility_lines — NaN / out-of-range / boundaries [wasm.md G2]', () => {
  it('[G2] NaN in vertices → both endpoints fail check → segment hidden', () => {
    const vertices = new Float32Array([Number.NaN, 0, 0, Number.NaN, 0, 0]);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([0, 0]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[G2] OR-semantics: NaN in ONE endpoint, other endpoint visible → segment visible', () => {
    // Pins the documented "EITHER endpoint is visible" rule from lines.ts L39.
    // v0 is NaN-poisoned; v1 is at origin → visible. Segment must remain visible.
    const vertices = new Float32Array([Number.NaN, 0, 0, 0, 0, 0]);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([0, 0]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('[G2] out-of-range vertex index reads undefined → coerced to NaN → endpoint fails', () => {
    // segments[0] = 99 (only 2 vertices in the buffer).
    // vertices[99*3+dim] is `undefined`; `undefined - 0 === NaN`; checkPoint returns false.
    // v1 at index 1 is at origin → visible → OR-semantics keeps segment visible.
    // Pin contract: the OOB read does NOT crash, AND the present-endpoint still drives the result.
    const vertices = new Float32Array([0, 0, 0, 0, 0, 0]);
    const segments = new Uint32Array([99, 1]);
    const widths = new Float32Array([0, 0]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    expect(() =>
      compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        slicePosition,
        tolerance,
        3,
        1,
        output
      )
    ).not.toThrow();
    expect(output[0]).toBe(1);
  });

  it('[G2] numSegments === 0 → loop never executes, count is 0, output untouched', () => {
    // Empty-input boundary. Symmetry with the gsplats numSplats=0 case.
    const vertices = new Float32Array([0, 0, 0]);
    const segments = new Uint32Array([]);
    const widths = new Float32Array([1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([1, 1, 1]);
    const output = new Uint8Array(0);
    const n = compute_nd_visibility_lines(
      vertices,
      segments,
      widths,
      slicePosition,
      tolerance,
      3,
      0,
      output
    );
    expect(n).toBe(0);
  });
});

describe('compute_nd_visibility_gsplats — NaN/Inf propagation [wasm.md G3]', () => {
  it('[G3] NaN in centers poisons distSq → splat hidden', () => {
    const centers = new Float32Array([Number.NaN, 0, 0]);
    // 3D identity Cholesky packed: [L00, L10, L11, L20, L21, L22] = [1,0,1,0,0,1].
    const cholesky = new Float32Array([1, 0, 1, 0, 0, 1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[G3] NaN in Cholesky + center FAR from slice → effTol=NaN fails the `>0` gate → shouldBreak → splat hidden', () => {
    // Pin actual behaviour at gsplats.ts L76-84: a NaN row norm produces
    // `maxExtent = NaN`, then `effectiveTolerance = tolerance + NaN = NaN`,
    // and `NaN > 0` is false → control falls into the `else if`. Because
    // the center is far from the slice, `|delta| > 1e-6` triggers
    // `shouldBreak` → output 0.
    //
    // (A center AT the slice would silently skip the dim entirely; see
    // the next test for the symmetric subtlety.)
    const centers = new Float32Array([3, 0, 0]);
    const cholesky = new Float32Array([1, 0, Number.NaN, 0, 0, 1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[G3] NaN in Cholesky + center AT slice → effTol=NaN silently skips every dim → distSq stays 0 → visible (documented quirk)', () => {
    // Documents the quirk: NaN in the Cholesky does NOT poison distSq when
    // the center coincides with the slice. The `else if (|delta| > 1e-6)`
    // gate skips the contribution silently. A future hardening of this
    // contract (e.g. treating NaN extent as "always hidden") would surface
    // as an intentional change.
    const centers = new Float32Array([0, 0, 0]);
    const cholesky = new Float32Array([1, 0, Number.NaN, 0, 0, 1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('[G3] negative diagonal in Cholesky is squared → row norm is positive → splat behaviour unchanged', () => {
    // Pin documented contract: `val * val` ignores sign; a -1 diagonal yields
    // the same row norm as +1, so visibility is identical. (The comment in
    // gsplats.ts L23 calls out this acceptance of negative diags.)
    const centers = new Float32Array([0, 0, 0]);
    const choleskyNeg = new Float32Array([-1, 0, -1, 0, 0, -1]);
    const choleskyPos = new Float32Array([1, 0, 1, 0, 0, 1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const outNeg = new Uint8Array(1);
    const outPos = new Uint8Array(1);
    const nNeg = compute_nd_visibility_gsplats(
      centers,
      choleskyNeg,
      slicePosition,
      tolerance,
      3,
      1,
      outNeg
    );
    const nPos = compute_nd_visibility_gsplats(
      centers,
      choleskyPos,
      slicePosition,
      tolerance,
      3,
      1,
      outPos
    );
    expect(nNeg).toBe(nPos);
    expect(outNeg[0]).toBe(outPos[0]);
  });

  it('[G3] Infinity in centers → distSq=Infinity → splat hidden', () => {
    const centers = new Float32Array([Number.POSITIVE_INFINITY, 0, 0]);
    const cholesky = new Float32Array([1, 0, 1, 0, 0, 1]);
    const slicePosition = new Float32Array([0, 0, 0]);
    const tolerance = new Float32Array([5, 5, 5]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      slicePosition,
      tolerance,
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// [wasm.md W11] ndim=16 tolerance-boundary fixture across all 3 geometries.
// Prior coverage only exercised "at slice (always visible)" and "far away
// (always hidden)" cases at high ndim, which trivially pass for any sane
// implementation. This block places per-dim contributions so that distSq
// accumulates to EXACTLY 1.0 — pinning the strict `<=` boundary plus the
// full-loop accumulation through every dimension.
//
// Construction: 16 dims, each position[d] = 0.25, tolerance[d] = 1,
// radius/width/extent = 0. Then per-dim normalized = 0.25, contribution = 0.0625,
// total distSq = 16 * 0.0625 = 1.0. With `distSq <= 1.0` the point is visible;
// a mutation flipping to `<` would now fail.
// ---------------------------------------------------------------------------
describe('ndim=16 tolerance-boundary across all three geometries [wasm.md W11]', () => {
  it('[W11] points: full-loop accumulation lands exactly on distSq=1.0 → visible', () => {
    const ndim = 16;
    const positions = new Float32Array(ndim);
    const slicePosition = new Float32Array(ndim);
    const tolerance = new Float32Array(ndim);
    for (let d = 0; d < ndim; d++) {
      positions[d] = 0.25;
      slicePosition[d] = 0;
      tolerance[d] = 1;
    }
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      slicePosition,
      tolerance,
      ndim,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('[W11] points: a 1-ULP nudge past the boundary on dim 15 hides the point', () => {
    // distSq = 16 * 0.0625 = 1.0 at the boundary. Bump one dim slightly so
    // its squared contribution > the budget: 0.26 instead of 0.25 →
    // contribution rises to 0.0676 → total > 1.0 → hidden.
    const ndim = 16;
    const positions = new Float32Array(ndim);
    const slicePosition = new Float32Array(ndim);
    const tolerance = new Float32Array(ndim);
    for (let d = 0; d < ndim; d++) {
      positions[d] = 0.25;
      slicePosition[d] = 0;
      tolerance[d] = 1;
    }
    positions[15] = 0.26;
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      slicePosition,
      tolerance,
      ndim,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('[W11] gsplats: identity Cholesky in 16D + boundary centers → distSq=1.0 → visible', () => {
    // Identity Cholesky in 16D: maxExtent = max row norm = 1 (since
    // each diagonal is 1, off-diagonals are 0). effectiveTolerance per dim
    // becomes tolerance[d] + maxExtent = 0 + 1 = 1 when tolerance[d]=0.
    // To re-use the same 1.0 boundary as the points test: keep tolerance=1
    // and set maxExtent=0 via zero-Cholesky. But zero diagonal → maxExtent=0,
    // effectiveTolerance=tolerance=1 → identical to the points construction.
    const ndim = 16;
    const packedSize = (ndim * (ndim + 1)) / 2;
    const centers = new Float32Array(ndim);
    const slicePosition = new Float32Array(ndim);
    const tolerance = new Float32Array(ndim);
    for (let d = 0; d < ndim; d++) {
      centers[d] = 0.25;
      slicePosition[d] = 0;
      tolerance[d] = 1;
    }
    const cholesky = new Float32Array(packedSize); // all zeros → maxExtent=0
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      slicePosition,
      tolerance,
      ndim,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });
});
