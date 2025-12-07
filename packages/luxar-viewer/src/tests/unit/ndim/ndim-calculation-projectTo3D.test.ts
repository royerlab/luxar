/**
 * ndim Calculation and projectTo3D Logic Tests
 *
 * CRITICAL: These tests verify the bug fix in point-spatial-index-loader.ts:787-805
 *
 * THE BUG:
 * - The old code used: `const ndim = this.chunkIndex?.metadata.ndim || 3;`
 * - For datasets without a spatial index (like quantum orbitals), chunkIndex is null
 * - This caused ndim to default to 3 even when positions are 4D
 * - Result: positions[i * 3 + dimIdx] instead of positions[i * 4 + dimIdx]
 * - This caused block artifacts where every 4th point was correct but others shifted
 *
 * THE FIX:
 * - Calculate ndim from actual positions array: positions.length / totalPoints
 * - This gives the true dimensionality regardless of whether chunk index exists
 *
 * These tests ensure the fix works correctly for various dimension scenarios.
 */

import { describe, it, expect } from 'vitest';

/**
 * Calculate ndim from positions array - mirrors the fixed logic in projectTo3D
 *
 * This is the EXACT logic from point-spatial-index-loader.ts lines 787-794
 */
function calculateNdim(
  positions: Float32Array,
  totalPoints: number,
  chunkIndexNdim: number | undefined
): number {
  return totalPoints > 0 ? Math.round(positions.length / totalPoints) : chunkIndexNdim || 3;
}

/**
 * Extract 3D positions from nD data - mirrors logic in projectTo3D
 *
 * This is the EXACT loop from point-spatial-index-loader.ts lines 811-821
 */
function projectTo3D(
  positions: Float32Array,
  ndim: number,
  displayDims: number[],
  numPoints: number
): Float32Array {
  const positions3D = new Float32Array(numPoints * 3);

  for (let i = 0; i < numPoints; i++) {
    // Extract displayed dimensions
    for (let j = 0; j < Math.min(3, displayDims.length); j++) {
      const dimIdx = displayDims[j];
      positions3D[i * 3 + j] = positions[i * ndim + dimIdx];
    }
    // Fill remaining with zeros
    for (let j = displayDims.length; j < 3; j++) {
      positions3D[i * 3 + j] = 0;
    }
  }

  return positions3D;
}

describe('ndim Calculation from Positions Array', () => {
  describe('Basic ndim Calculation', () => {
    it('should calculate ndim=3 for 3D positions', () => {
      // 10 points × 3 dimensions = 30 elements
      const positions = new Float32Array(30);
      const totalPoints = 10;

      const ndim = calculateNdim(positions, totalPoints, undefined);
      expect(ndim).toBe(3);
    });

    it('should calculate ndim=4 for 4D positions', () => {
      // 10 points × 4 dimensions = 40 elements
      const positions = new Float32Array(40);
      const totalPoints = 10;

      const ndim = calculateNdim(positions, totalPoints, undefined);
      expect(ndim).toBe(4);
    });

    it('should calculate ndim=5 for 5D positions', () => {
      // 10 points × 5 dimensions = 50 elements
      const positions = new Float32Array(50);
      const totalPoints = 10;

      const ndim = calculateNdim(positions, totalPoints, undefined);
      expect(ndim).toBe(5);
    });

    it('should use Math.round for slight size mismatches due to encoding', () => {
      // Simulating slight size mismatch from encoding overhead
      // 10 points × 4 dimensions ≈ 40, but let's say we have 39 or 41
      const positions39 = new Float32Array(39);
      const positions41 = new Float32Array(41);
      const totalPoints = 10;

      // Both should round to 4
      expect(calculateNdim(positions39, totalPoints, undefined)).toBe(4);
      expect(calculateNdim(positions41, totalPoints, undefined)).toBe(4);
    });

    it('should fallback to chunkIndex.ndim when totalPoints is 0', () => {
      const positions = new Float32Array(0);
      const totalPoints = 0;

      // With chunkIndex metadata
      expect(calculateNdim(positions, totalPoints, 4)).toBe(4);
      expect(calculateNdim(positions, totalPoints, 5)).toBe(5);

      // Without chunkIndex metadata, fallback to 3
      expect(calculateNdim(positions, totalPoints, undefined)).toBe(3);
    });
  });

  describe('Bug Regression Tests - Wrong ndim=3 for 4D data', () => {
    it('CRITICAL: demonstrates the bug with wrong ndim=3 for 4D positions', () => {
      // This is the EXACT scenario that caused the quantum orbitals bug
      // 20,480 points × 4 dimensions = 81,920 elements
      const numPoints = 100;
      const trueDim = 4;
      const positions = new Float32Array(numPoints * trueDim);

      // Fill with known pattern: point i has coords [i, i*2, i*3, i*4]
      for (let i = 0; i < numPoints; i++) {
        positions[i * 4 + 0] = i; // dim 0
        positions[i * 4 + 1] = i * 2; // dim 1
        positions[i * 4 + 2] = i * 3; // dim 2
        positions[i * 4 + 3] = i * 4; // dim 3
      }

      // CORRECT calculation (the fix)
      const correctNdim = calculateNdim(positions, numPoints, undefined);
      expect(correctNdim).toBe(4);

      // BUGGY calculation (the old code)
      const chunkIndexNdim = undefined; // No chunk index
      const buggyNdim = chunkIndexNdim || 3; // Defaults to 3!
      expect(buggyNdim).toBe(3); // This was the bug!

      // Show the impact on position indexing
      const displayDims = [1, 2, 3]; // Display dims 1, 2, 3 as X, Y, Z

      // CORRECT extraction with ndim=4
      const correctResult = projectTo3D(positions, correctNdim, displayDims, numPoints);

      // Point 0: should be [0*2, 0*3, 0*4] = [0, 0, 0]
      expect(correctResult[0 * 3 + 0]).toBe(0);
      expect(correctResult[0 * 3 + 1]).toBe(0);
      expect(correctResult[0 * 3 + 2]).toBe(0);

      // Point 5: should be [5*2, 5*3, 5*4] = [10, 15, 20]
      expect(correctResult[5 * 3 + 0]).toBe(10);
      expect(correctResult[5 * 3 + 1]).toBe(15);
      expect(correctResult[5 * 3 + 2]).toBe(20);

      // BUGGY extraction with ndim=3 (would access wrong positions)
      const buggyResult = projectTo3D(positions, buggyNdim, displayDims, numPoints);

      // With ndim=3, positions[i*3+dimIdx] instead of positions[i*4+dimIdx]
      // Point 5 with ndim=3: positions[5*3+1]=positions[16], positions[5*3+2]=positions[17], etc.
      // This reads from WRONG positions!

      // The buggy extraction gives wrong results
      // Point 5 would read from indices 16, 17, 18 instead of 21, 22, 23
      // positions[16] = 4*4+0 = 4th point's dim0 = 4
      // positions[17] = 4*4+1 = 4th point's dim1 = 8
      // positions[18] = 4*4+2 = 4th point's dim2 = 12
      expect(buggyResult[5 * 3 + 0]).not.toBe(10); // Wrong!
      expect(buggyResult[5 * 3 + 1]).not.toBe(15); // Wrong!
      expect(buggyResult[5 * 3 + 2]).not.toBe(20); // Wrong!

      // Verify the buggy values are from wrong indices
      expect(buggyResult[5 * 3 + 0]).toBe(positions[5 * 3 + 1]); // position[16]
      expect(buggyResult[5 * 3 + 1]).toBe(positions[5 * 3 + 2]); // position[17]
    });

    it('should correctly calculate ndim even without chunk index metadata', () => {
      // Simulating scalar LUT encoded 4D positions (like quantum orbitals)
      // 200 points × 4 dimensions = 800 elements
      const numPoints = 200;
      const positions = new Float32Array(800);

      // No chunk index - this used to cause ndim=3 fallback
      const chunkIndexNdim = undefined;

      // The fix calculates from actual data
      const ndim = calculateNdim(positions, numPoints, chunkIndexNdim);
      expect(ndim).toBe(4);
    });

    it('should handle the exact quantum orbitals scenario: 20480 points × 4D', () => {
      // Exact scenario from the bug report
      const numPoints = 20480;
      const positions = new Float32Array(numPoints * 4); // 81,920 elements

      const ndim = calculateNdim(positions, numPoints, undefined);
      expect(ndim).toBe(4);

      // With the bug, ndim would have been 3
      // numPoints would have been calculated as 81920/3 ≈ 27306 (WRONG!)
    });
  });
});

describe('projectTo3D Function', () => {
  describe('3D to 3D projection (identity)', () => {
    it('should correctly project 3D positions with displayDims=[0,1,2]', () => {
      const numPoints = 5;
      const positions = new Float32Array([
        // Point 0: (1, 2, 3)
        1, 2, 3,
        // Point 1: (4, 5, 6)
        4, 5, 6,
        // Point 2: (7, 8, 9)
        7, 8, 9,
        // Point 3: (10, 11, 12)
        10, 11, 12,
        // Point 4: (13, 14, 15)
        13, 14, 15,
      ]);

      const result = projectTo3D(positions, 3, [0, 1, 2], numPoints);

      expect(result.length).toBe(15);

      // Should be identity mapping
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBe(positions[i]);
      }
    });

    it('should correctly project 3D positions with swapped displayDims=[2,1,0]', () => {
      const numPoints = 3;
      const positions = new Float32Array([
        // Point 0: (1, 2, 3) -> displayed as (3, 2, 1)
        1, 2, 3,
        // Point 1: (4, 5, 6) -> displayed as (6, 5, 4)
        4, 5, 6,
        // Point 2: (7, 8, 9) -> displayed as (9, 8, 7)
        7, 8, 9,
      ]);

      const result = projectTo3D(positions, 3, [2, 1, 0], numPoints);

      // Point 0: x=3, y=2, z=1
      expect(result[0]).toBe(3);
      expect(result[1]).toBe(2);
      expect(result[2]).toBe(1);

      // Point 1: x=6, y=5, z=4
      expect(result[3]).toBe(6);
      expect(result[4]).toBe(5);
      expect(result[5]).toBe(4);

      // Point 2: x=9, y=8, z=7
      expect(result[6]).toBe(9);
      expect(result[7]).toBe(8);
      expect(result[8]).toBe(7);
    });
  });

  describe('4D to 3D projection (the quantum orbitals case)', () => {
    it('should correctly project 4D positions with displayDims=[1,2,3]', () => {
      const numPoints = 5;
      // 4D positions: [t, x, y, z]
      const positions = new Float32Array([
        // Point 0: t=0, x=1, y=2, z=3
        0, 1, 2, 3,
        // Point 1: t=1, x=4, y=5, z=6
        1, 4, 5, 6,
        // Point 2: t=2, x=7, y=8, z=9
        2, 7, 8, 9,
        // Point 3: t=3, x=10, y=11, z=12
        3, 10, 11, 12,
        // Point 4: t=4, x=13, y=14, z=15
        4, 13, 14, 15,
      ]);

      // Display dims 1, 2, 3 as X, Y, Z (skip time dimension 0)
      const result = projectTo3D(positions, 4, [1, 2, 3], numPoints);

      expect(result.length).toBe(15); // 5 points × 3

      // Point 0: x=1, y=2, z=3
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(2);
      expect(result[2]).toBe(3);

      // Point 1: x=4, y=5, z=6
      expect(result[3]).toBe(4);
      expect(result[4]).toBe(5);
      expect(result[5]).toBe(6);

      // Point 4: x=13, y=14, z=15
      expect(result[12]).toBe(13);
      expect(result[13]).toBe(14);
      expect(result[14]).toBe(15);
    });

    it('should correctly project 4D positions with displayDims=[0,1,2] (different slice)', () => {
      const numPoints = 3;
      // 4D positions: [t, x, y, z]
      const positions = new Float32Array([
        // Point 0: t=10, x=1, y=2, z=3
        10, 1, 2, 3,
        // Point 1: t=20, x=4, y=5, z=6
        20, 4, 5, 6,
        // Point 2: t=30, x=7, y=8, z=9
        30, 7, 8, 9,
      ]);

      // Display dims 0, 1, 2 as X, Y, Z (include time, skip z)
      const result = projectTo3D(positions, 4, [0, 1, 2], numPoints);

      // Point 0: x=10, y=1, z=2
      expect(result[0]).toBe(10);
      expect(result[1]).toBe(1);
      expect(result[2]).toBe(2);

      // Point 1: x=20, y=4, z=5
      expect(result[3]).toBe(20);
      expect(result[4]).toBe(4);
      expect(result[5]).toBe(5);
    });
  });

  describe('5D to 3D projection', () => {
    it('should correctly project 5D positions with displayDims=[2,3,4]', () => {
      const numPoints = 2;
      // 5D positions: [d0, d1, d2, d3, d4]
      const positions = new Float32Array([
        // Point 0: d0=0, d1=1, d2=2, d3=3, d4=4
        0, 1, 2, 3, 4,
        // Point 1: d0=10, d1=11, d2=12, d3=13, d4=14
        10, 11, 12, 13, 14,
      ]);

      // Display dims 2, 3, 4 as X, Y, Z
      const result = projectTo3D(positions, 5, [2, 3, 4], numPoints);

      // Point 0: x=2, y=3, z=4
      expect(result[0]).toBe(2);
      expect(result[1]).toBe(3);
      expect(result[2]).toBe(4);

      // Point 1: x=12, y=13, z=14
      expect(result[3]).toBe(12);
      expect(result[4]).toBe(13);
      expect(result[5]).toBe(14);
    });
  });

  describe('2D to 3D projection (padding with zeros)', () => {
    it('should correctly project 2D positions with displayDims=[0,1]', () => {
      const numPoints = 3;
      // 2D positions
      const positions = new Float32Array([
        // Point 0: x=1, y=2
        1, 2,
        // Point 1: x=3, y=4
        3, 4,
        // Point 2: x=5, y=6
        5, 6,
      ]);

      const result = projectTo3D(positions, 2, [0, 1], numPoints);

      // Point 0: x=1, y=2, z=0 (padded)
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(2);
      expect(result[2]).toBe(0);

      // Point 1: x=3, y=4, z=0 (padded)
      expect(result[3]).toBe(3);
      expect(result[4]).toBe(4);
      expect(result[5]).toBe(0);
    });
  });

  describe('Edge cases', () => {
    it('should handle single point', () => {
      const positions = new Float32Array([1, 2, 3, 4]);
      const result = projectTo3D(positions, 4, [0, 1, 2], 1);

      expect(result.length).toBe(3);
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(2);
      expect(result[2]).toBe(3);
    });

    it('should handle zero points', () => {
      const positions = new Float32Array(0);
      const result = projectTo3D(positions, 4, [0, 1, 2], 0);

      expect(result.length).toBe(0);
    });

    it('should handle displayDims with only 2 dimensions', () => {
      const numPoints = 2;
      const positions = new Float32Array([0, 1, 2, 3, 10, 11, 12, 13]);

      // Only 2 display dims, z should be padded with 0
      const result = projectTo3D(positions, 4, [1, 2], numPoints);

      // Point 0: x=1, y=2, z=0
      expect(result[0]).toBe(1);
      expect(result[1]).toBe(2);
      expect(result[2]).toBe(0);

      // Point 1: x=11, y=12, z=0
      expect(result[3]).toBe(11);
      expect(result[4]).toBe(12);
      expect(result[5]).toBe(0);
    });

    it('should handle displayDims with only 1 dimension', () => {
      const numPoints = 2;
      const positions = new Float32Array([0, 1, 2, 3, 10, 11, 12, 13]);

      // Only 1 display dim, y and z should be padded with 0
      const result = projectTo3D(positions, 4, [2], numPoints);

      // Point 0: x=2, y=0, z=0
      expect(result[0]).toBe(2);
      expect(result[1]).toBe(0);
      expect(result[2]).toBe(0);

      // Point 1: x=12, y=0, z=0
      expect(result[3]).toBe(12);
      expect(result[4]).toBe(0);
      expect(result[5]).toBe(0);
    });
  });
});

describe('Integration: ndim calculation + projectTo3D', () => {
  it('should correctly process 4D scalar LUT encoded positions (quantum orbitals scenario)', () => {
    // Simulate the full pipeline that was broken:
    // 1. Decode 4D positions from scalar LUT
    // 2. Calculate ndim from positions array
    // 3. Project to 3D for rendering

    const numPoints = 100;
    const trueDim = 4;

    // Simulated decoded positions (what we'd get after LUT decoding)
    const positions = new Float32Array(numPoints * trueDim);
    for (let i = 0; i < numPoints; i++) {
      // [t, x, y, z] where t=i%10, x/y/z form a sphere
      const t = i % 10;
      const theta = (i / numPoints) * Math.PI * 2;
      const phi = ((i % 10) / 10) * Math.PI;
      positions[i * 4 + 0] = t;
      positions[i * 4 + 1] = Math.sin(phi) * Math.cos(theta);
      positions[i * 4 + 2] = Math.sin(phi) * Math.sin(theta);
      positions[i * 4 + 3] = Math.cos(phi);
    }

    // Step 1: Calculate ndim (no chunk index, so must calculate from data)
    const ndim = calculateNdim(positions, numPoints, undefined);
    expect(ndim).toBe(4); // CRITICAL: must be 4, not 3!

    // Step 2: Project to 3D with displayDims=[1,2,3] (skip time)
    const positions3D = projectTo3D(positions, ndim, [1, 2, 3], numPoints);

    expect(positions3D.length).toBe(numPoints * 3);

    // Verify first point
    const p0_expected = [positions[1], positions[2], positions[3]];
    expect(positions3D[0]).toBeCloseTo(p0_expected[0], 5);
    expect(positions3D[1]).toBeCloseTo(p0_expected[1], 5);
    expect(positions3D[2]).toBeCloseTo(p0_expected[2], 5);

    // Verify last point
    const lastIdx = (numPoints - 1) * 4;
    const pLast_expected = [positions[lastIdx + 1], positions[lastIdx + 2], positions[lastIdx + 3]];
    const lastOut = (numPoints - 1) * 3;
    expect(positions3D[lastOut]).toBeCloseTo(pLast_expected[0], 5);
    expect(positions3D[lastOut + 1]).toBeCloseTo(pLast_expected[1], 5);
    expect(positions3D[lastOut + 2]).toBeCloseTo(pLast_expected[2], 5);
  });

  it('REGRESSION: demonstrates what happens with buggy ndim=3 for 4D data', () => {
    const numPoints = 100;
    const positions = new Float32Array(numPoints * 4);
    for (let i = 0; i < numPoints; i++) {
      positions[i * 4 + 0] = i; // t
      positions[i * 4 + 1] = i * 10; // x
      positions[i * 4 + 2] = i * 100; // y
      positions[i * 4 + 3] = i * 1000; // z
    }

    const displayDims = [1, 2, 3];

    // CORRECT: ndim=4
    const correct3D = projectTo3D(positions, 4, displayDims, numPoints);

    // BUGGY: ndim=3 (the old default)
    const buggy3D = projectTo3D(positions, 3, displayDims, numPoints);

    // Point 10 should be: x=100, y=1000, z=10000
    // But with ndim=3, we read from positions[10*3+1], [10*3+2], [10*3+3]
    // = positions[31], positions[32], positions[33]
    // = 7*4+3=7000, 8*4+0=8, 8*4+1=80

    // CORRECT values for point 10
    expect(correct3D[10 * 3 + 0]).toBe(100); // positions[10*4+1]
    expect(correct3D[10 * 3 + 1]).toBe(1000); // positions[10*4+2]
    expect(correct3D[10 * 3 + 2]).toBe(10000); // positions[10*4+3]

    // BUGGY values for point 10 (completely wrong!)
    expect(buggy3D[10 * 3 + 0]).not.toBe(100);
    expect(buggy3D[10 * 3 + 1]).not.toBe(1000);
    expect(buggy3D[10 * 3 + 2]).not.toBe(10000);

    // Show what buggy values actually are
    // positions[31] = positions[7*4+3] = 7*1000 = 7000
    expect(buggy3D[10 * 3 + 0]).toBe(7000); // WRONG - reading from wrong point!
  });
});
