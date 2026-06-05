import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  projectGSplatsTo3D,
  projectGSplats3DOnly,
  projectGSplats,
  createEmptyGSplatsData,
} from '../../../../data/gsplats/projection';
import type {
  LoadedGSplatsData,
  GSplatsViewState,
  GSplatsMetadata,
} from '../../../../types/gsplats';

describe('projectGSplats3DOnly', () => {
  it('should copy 3D data directly', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      amplitudes: new Float32Array([1.0, 0.5, 0.3]),
      choleskyFactors: new Float32Array([
        1,
        0,
        1,
        0,
        0,
        1, // Identity covariance
        2,
        0,
        2,
        0,
        0,
        2, // Scaled
        1,
        0.5,
        1,
        0,
        0.5,
        1, // Anisotropic
      ]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),

      splatCount: 3,
      ndim: 3,
    };

    const result = projectGSplats3DOnly(loaded);

    expect(result.splatCount).toBe(3);
    expect(result.centers3D.length).toBe(9);
    expect(result.amplitudes.length).toBe(3);
    expect(result.choleskyFactors3D.length).toBe(18);
    expect(result.colors.length).toBe(9);

    // Check values are copied correctly
    expect(result.centers3D[0]).toBe(0);
    expect(result.centers3D[3]).toBe(1);
    expect(result.amplitudes[0]).toBe(1.0);
    expect(result.amplitudes[1]).toBe(0.5);
  });

  it('should handle data without colors', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      splatCount: 1,
      ndim: 3,
    };

    const result = projectGSplats3DOnly(loaded);
    expect(result.splatCount).toBe(1);
  });

  it('should default colors to white if not present', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 3,
    };

    const result = projectGSplats3DOnly(loaded);

    expect(result.colors[0]).toBe(1.0);
    expect(result.colors[1]).toBe(1.0);
    expect(result.colors[2]).toBe(1.0);
  });

  it('should throw for non-3D data', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    expect(() => projectGSplats3DOnly(loaded)).toThrow('requires ndim=3');
  });
});

describe('projectGSplatsTo3D', () => {
  it('should process 3D data with no hidden dims', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),

      splatCount: 2,
      ndim: 3,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(2);
    expect(result.amplitudes[0]).toBe(1.0); // No attenuation
    expect(result.amplitudes[1]).toBe(0.5);
  });

  it('should attenuate amplitude based on hidden dimension distance', () => {
    // 4D data with display dims [0, 1, 2] and hidden dim [3]
    const loaded: LoadedGSplatsData = {
      // Splat at (0,0,0,0) - on the slice
      // Splat at (0,0,0,10) - 10 units from slice in dim 3
      positions: new Float32Array([0, 0, 0, 0, 0, 0, 0, 10]),
      amplitudes: new Float32Array([1.0, 1.0]),
      // 4D Cholesky has 10 elements: [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
      // Use identity-like covariance (σ=1 in all dims)
      choleskyFactors: new Float32Array([
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1, // First splat: identity
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1, // Second splat: identity
      ]),
      colors: new Float32Array([1, 1, 1, 1, 1, 1]),

      splatCount: 2,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0], // Slice at dim3 = 0
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // First splat: on slice, no attenuation
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);

    // Second splat: 10 units away in hidden dim with σ=1
    // Mahalanobis distance = 10
    // Attenuation = exp(-0.5 * 10^2) = exp(-50) ≈ 1.9e-22
    // This should be filtered out as below threshold (1e-6)
    expect(result.splatCount).toBe(1); // Only first splat visible
  });

  it('should extract correct 3D submatrix from higher-dim Cholesky', () => {
    // 4D data, display dims [0, 1, 2]
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([1, 2, 3, 0]), // 4D center
      amplitudes: new Float32Array([1.0]),
      // 4D Cholesky (10 elements): [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
      // Set specific values to verify extraction
      choleskyFactors: new Float32Array([
        1, // L00
        2, // L10
        3, // L11
        4, // L20
        5, // L21
        6, // L22
        0, // L30
        0, // L31
        0, // L32
        1, // L33
      ]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // 3D Cholesky should be [L00, L10, L11, L20, L21, L22] = [1, 2, 3, 4, 5, 6]
    expect(result.choleskyFactors3D[0]).toBe(1); // L00
    expect(result.choleskyFactors3D[1]).toBe(2); // L10
    expect(result.choleskyFactors3D[2]).toBe(3); // L11
    expect(result.choleskyFactors3D[3]).toBe(4); // L20
    expect(result.choleskyFactors3D[4]).toBe(5); // L21
    expect(result.choleskyFactors3D[5]).toBe(6); // L22
  });

  it('should extract 3D center using display dimensions', () => {
    // 5D data, display dims [1, 3, 4]
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([10, 20, 30, 40, 50]), // 5D center
      amplitudes: new Float32Array([1.0]),
      // Minimal valid Cholesky for 5D (15 elements)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      colors: new Float32Array([1, 0, 0]),

      splatCount: 1,
      ndim: 5,
    };

    const viewState: GSplatsViewState = {
      displayDims: [1, 3, 4],
      slicePosition: [10, 0, 30, 0, 0], // At hidden dims 0=10, 2=30
      tolerance: [1, 1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // 3D center should be [dim1, dim3, dim4] = [20, 40, 50]
    expect(result.centers3D[0]).toBe(20);
    expect(result.centers3D[1]).toBe(40);
    expect(result.centers3D[2]).toBe(50);
  });
});

describe('projectGSplats', () => {
  it('should use optimized path for standard 3D data', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 3,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplats(loaded, viewState);

    expect(result.splatCount).toBe(1);
  });

  it('should use general path for non-standard display dims', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([10, 20, 30]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1, 0, 0]),

      splatCount: 1,
      ndim: 3,
    };

    // Non-standard order: [2, 0, 1] — output XYZ = source [dim2, dim0, dim1]
    const viewState: GSplatsViewState = {
      displayDims: [2, 0, 1],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplats(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // requested displayDims order is preserved (matches Points/Lines).
    // displayDims=[2,0,1] → output[0,1,2] = source[2,0,1] = [30, 10, 20].
    expect(result.centers3D[0]).toBe(30);
    expect(result.centers3D[1]).toBe(10);
    expect(result.centers3D[2]).toBe(20);
  });

  it('should use general path for higher-dimensional data', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplats(loaded, viewState);

    expect(result.splatCount).toBe(1);
  });
});

describe('edge cases', () => {
  it('should handle empty input', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array(0),
      amplitudes: new Float32Array(0),
      choleskyFactors: new Float32Array(0),
      colors: null,

      splatCount: 0,
      ndim: 3,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplats(loaded, viewState);

    expect(result.splatCount).toBe(0);
    expect(result.centers3D.length).toBe(0);
  });

  it('should filter out splats with zero amplitude', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      amplitudes: new Float32Array([1.0, 0.0]), // Second has zero amplitude
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      colors: null,

      splatCount: 2,
      ndim: 3,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1); // Only first splat
    expect(result.centers3D[0]).toBe(0);
    expect(result.centers3D[1]).toBe(0);
    expect(result.centers3D[2]).toBe(0);
  });

  it('should use standard Gaussian falloff', () => {
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      splatCount: 1,
      ndim: 3,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);
    expect(result.splatCount).toBe(1);
  });
});

describe('workspace reuse safety', () => {
  it('should produce correct results when called consecutively with different ndim', () => {
    // First call: 4D data (1 hidden dim)
    const loaded4D: LoadedGSplatsData = {
      positions: new Float32Array([1, 2, 3, 0.5]),
      amplitudes: new Float32Array([1.0]),
      // 4D identity Cholesky (10 elements)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };
    const viewState4D: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };
    const result4D = projectGSplatsTo3D(loaded4D, viewState4D);

    // Second call: 5D data (2 hidden dims) — workspace must be clean
    const loaded5D: LoadedGSplatsData = {
      positions: new Float32Array([10, 20, 30, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      // 5D identity Cholesky (15 elements)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 5,
    };
    const viewState5D: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0, 0],
      tolerance: [1, 1, 1, 1, 1],
    };
    const result5D = projectGSplatsTo3D(loaded5D, viewState5D);

    // Third call: back to 3D (0 hidden dims) — workspace must not interfere
    const loaded3D: LoadedGSplatsData = {
      positions: new Float32Array([5, 6, 7]),
      amplitudes: new Float32Array([0.8]),
      choleskyFactors: new Float32Array([2, 0, 2, 0, 0, 2]),
      colors: null,

      splatCount: 1,
      ndim: 3,
    };
    const viewState3D: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };
    const result3D = projectGSplatsTo3D(loaded3D, viewState3D);

    // Verify each call produced correct independent results
    expect(result4D.splatCount).toBe(1);
    expect(result4D.centers3D[0]).toBe(1);
    expect(result4D.centers3D[1]).toBe(2);
    expect(result4D.centers3D[2]).toBe(3);

    expect(result5D.splatCount).toBe(1);
    expect(result5D.centers3D[0]).toBe(10);
    expect(result5D.centers3D[1]).toBe(20);
    expect(result5D.centers3D[2]).toBe(30);

    expect(result3D.splatCount).toBe(1);
    expect(result3D.centers3D[0]).toBe(5);
    expect(result3D.centers3D[1]).toBe(6);
    expect(result3D.centers3D[2]).toBe(7);
    expect(result3D.amplitudes[0]).toBeCloseTo(0.8, 5); // No attenuation for pure 3D
  });

  it('should correctly attenuate with cross-correlated hidden dimensions', () => {
    // 4D data where hidden dim (3) is correlated with display dims via L[3,0..2] != 0
    // This verifies marginal Cholesky (not just submatrix extraction) is correct.
    //
    // Full 4D Cholesky L:
    //   L00=1, L10=0, L11=1, L20=0, L21=0, L22=1, L30=0.5, L31=0.5, L32=0.5, L33=1
    //
    // Covariance Σ = L @ L^T:
    //   Σ[3,3] = L30^2 + L31^2 + L32^2 + L33^2 = 0.25 + 0.25 + 0.25 + 1 = 1.75
    // Hidden marginal covariance (just dim 3): Σ_h = [[1.75]]
    // Marginal Cholesky: L_h = [[sqrt(1.75)]] ≈ [[1.3229]]
    //
    // Splat center at dim3 = 1.0, slice at dim3 = 0
    // Mahalanobis distance = |1.0| / sqrt(1.75) ≈ 0.7559
    // Attenuation = exp(-0.5 * 0.7559^2) = exp(-0.2857) ≈ 0.7514
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1.0]), // dim3 = 1.0
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([
        1, // L00
        0,
        1, // L10, L11
        0,
        0,
        1, // L20, L21, L22
        0.5,
        0.5,
        0.5,
        1, // L30, L31, L32, L33 - cross-correlations!
      ]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // Shifted Gaussian attenuation: scale * max(0, exp(-0.5 * D²) - C) where C = exp(-4.5)
    const shiftC = Math.exp(-0.5 * 3.0 * 3.0);
    const invOneMinusC = 1.0 / (1.0 - shiftC);
    const mahalSq = (1.0 / Math.sqrt(1.75)) ** 2;
    const expectedAttenuation = invOneMinusC * (Math.exp(-0.5 * mahalSq) - shiftC);
    expect(result.amplitudes[0]).toBeCloseTo(expectedAttenuation, 4);
  });

  it('should produce consistent attenuation across mix of visible and filtered splats', () => {
    // 4D data with 4 splats at varying distances in hidden dim
    // Verifies the attenuation cache correctly pairs cached values with visible indices
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([
        0,
        0,
        0,
        0, // Splat 0: on slice → visible
        0,
        0,
        0,
        100, // Splat 1: far away → filtered out
        0,
        0,
        0,
        0.5, // Splat 2: close → visible with some attenuation
        0,
        0,
        0,
        50, // Splat 3: very far → filtered out
      ]),
      amplitudes: new Float32Array([1.0, 1.0, 1.0, 1.0]),
      // 4D identity Cholesky for each splat (10 elements each)
      choleskyFactors: new Float32Array([
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1, // splat 0
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1, // splat 1
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1, // splat 2
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1, // splat 3
      ]),
      colors: new Float32Array([
        1,
        0,
        0, // red
        0,
        1,
        0, // green (should be filtered)
        0,
        0,
        1, // blue
        1,
        1,
        0, // yellow (should be filtered)
      ]),

      splatCount: 4,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // Only splats 0 and 2 should survive
    expect(result.splatCount).toBe(2);

    // Splat 0: on slice, full amplitude
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);
    // Color should be red (from splat 0, not green from splat 1)
    expect(result.colors[0]).toBeCloseTo(1.0, 5); // R
    expect(result.colors[1]).toBeCloseTo(0.0, 5); // G
    expect(result.colors[2]).toBeCloseTo(0.0, 5); // B

    // Splat 2: 0.5 units away, shifted Gaussian attenuation
    const shiftC2 = Math.exp(-0.5 * 3.0 * 3.0);
    const invScale2 = 1.0 / (1.0 - shiftC2);
    const expectedAtt = invScale2 * (Math.exp(-0.5 * 0.25) - shiftC2);
    expect(result.amplitudes[1]).toBeCloseTo(expectedAtt, 4);
    // Color should be blue (from splat 2)
    expect(result.colors[3]).toBeCloseTo(0.0, 5); // R
    expect(result.colors[4]).toBeCloseTo(0.0, 5); // G
    expect(result.colors[5]).toBeCloseTo(1.0, 5); // B
  });
});

describe('discrete dimension handling', () => {
  it('should use binary visibility for discrete hidden dims (on-slice = full amplitude)', () => {
    // 4D data with dim 3 as discrete (time=23, slicePosition=23)
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 23.0]),
      amplitudes: new Float32Array([1.0]),
      // 4D Cholesky: identity in spatial, tiny sigma=0.3 in time (L[3,3]=0.3)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0.3]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 23.0],
      tolerance: [1e10, 1e10, 1e10, 0.5],
      dimensions: [
        { name: 'X', unit: 'um', scale: 1 },
        { name: 'Y', unit: 'um', scale: 1 },
        { name: 'Z', unit: 'um', scale: 1 },
        { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1.0 },
      ],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // Discrete dim: no Gaussian attenuation, full amplitude
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);
  });

  it('should filter out splats from wrong discrete step', () => {
    // Splat at time=24, slicePosition=23, step=1.0 → |diff|=1.0 > 0.5 → invisible
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 24.0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0.3]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 23.0],
      tolerance: [1e10, 1e10, 1e10, 0.5],
      dimensions: [
        { name: 'X', unit: 'um', scale: 1 },
        { name: 'Y', unit: 'um', scale: 1 },
        { name: 'Z', unit: 'um', scale: 1 },
        { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1.0 },
      ],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(0); // Filtered out by discrete check
  });

  it('should handle mixed discrete + continuous hidden dims', () => {
    // 5D: dims 0-2 display, dim 3 discrete (time), dim 4 continuous (wavelength)
    // Splat at correct time but offset in wavelength
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 5.0, 2.0]),
      amplitudes: new Float32Array([1.0]),
      // 5D identity Cholesky (15 elements)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 5,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 5.0, 0.0], // At time=5, wavelength=0
      tolerance: [1e10, 1e10, 1e10, 0.5, 3.0],
      dimensions: [
        { name: 'X', unit: 'um', scale: 1 },
        { name: 'Y', unit: 'um', scale: 1 },
        { name: 'Z', unit: 'um', scale: 1 },
        { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1.0 },
        { name: 'Wavelength', unit: 'nm', scale: 1 }, // continuous
      ],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // Time passes (discrete, exact match), wavelength offset = 2.0
    // Continuous Mahalanobis for dim 4 only: diff=2.0, sigma=1 → mahal=2.0
    // Shifted Gaussian attenuation
    expect(result.splatCount).toBe(1);
    const shiftC3 = Math.exp(-0.5 * 9.0);
    const invScale3 = 1.0 / (1.0 - shiftC3);
    const expected = invScale3 * (Math.exp(-0.5 * 4.0) - shiftC3);
    expect(result.amplitudes[0]).toBeCloseTo(expected, 4);
  });

  it('should skip discrete check for extend_to_all dims (tolerance >= 1e9)', () => {
    // 5D: dims 0-2 display, dim 3 "membranes" (discrete, fill=1), dim 4 "nuclei" (discrete, extend_to_all)
    // Splat at membranes=1.0, nuclei=0.0 (unmapped default).
    // Slice at membranes=1, nuclei=1. Without extend_to_all fix, |1-0|=1 > 0.5 kills the splat.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1.0, 0.0]),
      amplitudes: new Float32Array([1.0]),
      // 5D identity Cholesky (15 elements)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 5,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 1.0, 1.0], // Both toggles ON
      tolerance: [1e10, 1e10, 1e10, 0.5, 1e10], // nuclei (dim 4) has extend_to_all tolerance
      dimensions: [
        { name: 'X', unit: 'px', scale: 1 },
        { name: 'Y', unit: 'px', scale: 1 },
        { name: 'Z', unit: 'px', scale: 1 },
        { name: 'Membranes', unit: '', scale: 1, discrete: true, step: 1.0 },
        { name: 'Nuclei', unit: '', scale: 1, discrete: true, step: 1.0 },
      ],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // Splat should be visible: membranes=1 matches, nuclei is extend_to_all (skipped)
    expect(result.splatCount).toBe(1);
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);
  });

  it('should default to continuous (backward compat) when no dimensions metadata', () => {
    // Without dimensions metadata, all hidden dims use Gaussian attenuation
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0.5]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
      colors: null,

      splatCount: 1,
      ndim: 4,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
      // No dimensions metadata → backward-compatible Gaussian behavior
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // Shifted Gaussian attenuation: mahal=0.5
    const shiftC4 = Math.exp(-0.5 * 9.0);
    const invScale4 = 1.0 / (1.0 - shiftC4);
    const expected = invScale4 * (Math.exp(-0.5 * 0.25) - shiftC4);
    expect(result.amplitudes[0]).toBeCloseTo(expected, 4);
  });
});

// ============================================================================
// [P5] BOUNDARY / edge-case coverage: degenerate covariance, sign guards,
// discrete fence-post, and Uint16 color normalization. Each assertion was
// verified against the REAL projectGSplatsTo3D / projectGSplats3DOnly behavior
// (the epsilon-clamp path in computeMarginalCholesky / mahalanobisDistanceReuse,
// the `attenuatedAmplitude >= minAmplitude` filter at 1e-6, the strict `>`
// discrete-step comparison, and the 1/65535 Uint16 normFactor).
// ============================================================================

describe('boundary / degenerate inputs', () => {
  it('clamps NaN cholesky diagonal via the epsilon path — output amplitudes stay finite', () => {
    // 4D splat, hidden dim 3. The full Cholesky has a NaN in the hidden-dim
    // diagonal (L33). computeMarginalCholesky reconstructs Σ_h from L; the
    // NaN makes `sum > CHOLESKY_EPSILON` false (NaN comparisons are false),
    // so the diagonal is clamped to sqrt(CHOLESKY_EPSILON). The Mahalanobis
    // distance then uses `diag > 1e-10 ? ... : 0`, again NaN-safe.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0.1]),
      amplitudes: new Float32Array([1.0]),
      // identity spatial, NaN in L33 (hidden-dim diagonal)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, NaN]),
      colors: null,
      splatCount: 1,
      ndim: 4,
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // Whatever survives the filter must carry FINITE amplitudes (no NaN leak).
    for (let i = 0; i < result.amplitudes.length; i++) {
      expect(Number.isFinite(result.amplitudes[i])).toBe(true);
    }
    // 3D Cholesky factors (display dims) must also be finite.
    for (let i = 0; i < result.choleskyFactors3D.length; i++) {
      expect(Number.isFinite(result.choleskyFactors3D[i])).toBe(true);
    }
  });

  it('NaN in the HIDDEN-dim diagonal keeps the surviving amplitude finite (epsilon path)', () => {
    // Sibling of the prior NaN test, but the NaN lives only in the hidden-dim
    // diagonal L33 used for attenuation. The marginal Σ_h reconstruction +
    // mahalanobisDistanceReuse both gate on `diag > EPSILON / 1e-10` (false
    // for NaN), so the attenuation — and any surviving amplitude — is finite.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0]), // on slice in hidden dim
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, NaN]),
      colors: null,
      splatCount: 1,
      ndim: 4,
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    for (let i = 0; i < result.amplitudes.length; i++) {
      expect(Number.isFinite(result.amplitudes[i])).toBe(true);
    }
  });

  it('filters out a splat with negative input amplitude (attenuatedAmplitude < minAmplitude)', () => {
    // The only amplitude guard is `attenuatedAmplitude >= minAmplitude`
    // (minAmplitude = 1e-6). A negative amplitude × positive attenuation is
    // negative, hence < 1e-6, hence dropped. Pair it with a positive splat
    // to confirm only the positive one survives.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      amplitudes: new Float32Array([-2.0, 0.5]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      colors: null,
      splatCount: 2,
      ndim: 3,
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1); // negative-amplitude splat dropped
    expect(result.centers3D[0]).toBe(1); // surviving splat is the second
    expect(result.amplitudes[0]).toBeCloseTo(0.5, 5);
  });

  it('keeps Mahalanobis attenuation finite for a near-singular (near-zero diagonal) covariance', () => {
    // Hidden-dim diagonal L33 is below CHOLESKY_EPSILON. The reconstructed
    // marginal Σ_h ≈ L33² ≈ 0; the Crout step clamps to sqrt(EPSILON), and
    // mahalanobisDistanceReuse uses `diag > 1e-10 ? val/diag : 0`, so the
    // distance — and therefore the attenuation/amplitude — stays finite.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 0.0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1e-12]),
      colors: null,
      splatCount: 1,
      ndim: 4,
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0], // diff in hidden dim = 0 → mahal = 0
      tolerance: [1, 1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    // On-slice (diff 0): attenuation 1.0, fully finite.
    expect(result.splatCount).toBe(1);
    expect(Number.isFinite(result.amplitudes[0])).toBe(true);
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);
  });

  it('discrete fence-post: |diff| == step*0.5 EXACTLY is KEPT (comparison is strict >)', () => {
    // The discrete check drops a splat only when `absDiff > step * 0.5`.
    // At exactly step*0.5 the strict `>` is false, so the splat is KEPT.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 23.5]), // 0.5 past slice center
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0.3]),
      colors: null,
      splatCount: 1,
      ndim: 4,
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 23.0], // |23.5 - 23.0| = 0.5 == step*0.5
      tolerance: [1e10, 1e10, 1e10, 0.5],
      dimensions: [
        { name: 'X', unit: 'um', scale: 1 },
        { name: 'Y', unit: 'um', scale: 1 },
        { name: 'Z', unit: 'um', scale: 1 },
        { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1.0 },
      ],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1); // on the threshold → kept (strict >)
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);
  });

  it('discrete fence-post: just past step*0.5 (0.5 + epsilon) is DROPPED', () => {
    // Sibling of the previous test: nudging just past the threshold flips
    // the strict `>` to true and drops the splat. Pins the boundary on both
    // sides so a `>` → `>=` mutation is caught.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0, 23.5001]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0.3]),
      colors: null,
      splatCount: 1,
      ndim: 4,
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 23.0], // |23.5001 - 23| = 0.5001 > 0.5
      tolerance: [1e10, 1e10, 1e10, 0.5],
      dimensions: [
        { name: 'X', unit: 'um', scale: 1 },
        { name: 'Y', unit: 'um', scale: 1 },
        { name: 'Z', unit: 'um', scale: 1 },
        { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1.0 },
      ],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(0); // past threshold → dropped
  });

  it('normalizes Uint16 colors so 65535 maps to ~1.0 (1/65535 normFactor)', () => {
    // Exercises the nD path (projectGSplatsTo3D) Uint16 normFactor branch.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      // R=65535 (full), G=0, B=32768 (~half)
      colors: new Uint16Array([65535, 0, 32768]),
      splatCount: 1,
      ndim: 3,
    };
    // Force the general nD path (non-standard display order) so the inlined
    // Uint16 normFactor in projectGSplatsTo3D is exercised.
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = projectGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1);
    expect(result.colors[0]).toBeCloseTo(1.0, 5); // 65535 / 65535
    expect(result.colors[1]).toBeCloseTo(0.0, 5);
    expect(result.colors[2]).toBeCloseTo(0.5, 4); // 32768 / 65535 ≈ 0.50001
  });

  it('normalizes Uint16 colors in the optimized 3D-only path (65535 → ~1.0)', () => {
    // projectGSplats3DOnly has its own Uint16 branch (1/65535). Pin it too.
    const loaded: LoadedGSplatsData = {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: new Uint16Array([65535, 0, 32768]),
      splatCount: 1,
      ndim: 3,
    };

    const result = projectGSplats3DOnly(loaded);

    expect(result.colors[0]).toBeCloseTo(1.0, 5);
    expect(result.colors[1]).toBeCloseTo(0.0, 5);
    expect(result.colors[2]).toBeCloseTo(0.5, 4);
  });
});

describe('createEmptyGSplatsData', () => {
  function makeAttrs(overrides: Partial<GSplatsMetadata> = {}): GSplatsMetadata {
    return {
      type: 'gsplats',
      ndim: 3,
      n_splats: 0,
      ...overrides,
    } as GSplatsMetadata;
  }

  it('returns zero-length typed arrays with the canonical "no visible splats" shape', () => {
    const data = createEmptyGSplatsData(makeAttrs());
    expect(data.positions).toBeInstanceOf(Float32Array);
    expect(data.positions.length).toBe(0);
    expect(data.amplitudes).toBeInstanceOf(Float32Array);
    expect(data.amplitudes.length).toBe(0);
    expect(data.choleskyFactors).toBeInstanceOf(Float32Array);
    expect(data.choleskyFactors.length).toBe(0);
    expect(data.splatCount).toBe(0);
  });

  it('omits the optional colors array', () => {
    const data = createEmptyGSplatsData(makeAttrs());
    expect(data.colors).toBeNull();
  });

  it('forwards ndim from the metadata', () => {
    const data3 = createEmptyGSplatsData(makeAttrs({ ndim: 3 }));
    expect(data3.ndim).toBe(3);
    const data5 = createEmptyGSplatsData(makeAttrs({ ndim: 5 }));
    expect(data5.ndim).toBe(5);
    const data10 = createEmptyGSplatsData(makeAttrs({ ndim: 10 }));
    expect(data10.ndim).toBe(10);
  });

  it('returns fresh arrays on each call (no shared buffer state)', () => {
    const a = createEmptyGSplatsData(makeAttrs());
    const b = createEmptyGSplatsData(makeAttrs());
    expect(a.positions).not.toBe(b.positions);
    expect(a.amplitudes).not.toBe(b.amplitudes);
    expect(a.choleskyFactors).not.toBe(b.choleskyFactors);
  });
});

// ============================================================================
// [data.md/H4][P12] Property test: translation invariance under hidden-dim shifts
//
// The gsplat attenuation along hidden dims is a function of the difference
// `slicePosition[h] - center[h]`. If we shift BOTH slicePosition and the
// splat's hidden-dim coordinate by the same scalar k, the relative
// position in hidden space is unchanged and the attenuation/amplitude
// of every splat MUST be unchanged. The 3D-projected centers are taken
// from displayDims only, so those are also unchanged by hidden-dim shifts.
//
// This is the algebraic property that downstream slice navigation
// (keyboard nav, etc.) relies on: moving "through" the volume with the
// slider is geometrically equivalent to moving the splats relative to
// the slider — pin it as an invariant.
// ============================================================================

describe('projectGSplatsTo3D (property tests)', () => {
  const f = Math.fround;

  it('[H4] translation invariance: shifting slicePos[hidden] and centers[hidden] by k leaves amplitudes/centers3D unchanged', () => {
    // 4D scene with 1 hidden dim (dim 3). Splats with random centers
    // and amplitudes; identity 4D Cholesky for tractable attenuation;
    // tolerance 1.0 on the hidden dim so the Gaussian is not flat.
    fc.assert(
      fc.property(
        // splat 0 center: (cx0, cy0, cz0, cw0)
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: f(-1), max: f(1), noNaN: true, noDefaultInfinity: true }),
        // splat 1 center: (cx1, cy1, cz1, cw1)
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: f(-10), max: f(10), noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: f(-1), max: f(1), noNaN: true, noDefaultInfinity: true }),
        // slicePos on hidden dim
        fc.float({ min: f(-1), max: f(1), noNaN: true, noDefaultInfinity: true }),
        // translation k
        fc.float({ min: f(-5), max: f(5), noNaN: true, noDefaultInfinity: true }),
        (cx0, cy0, cz0, cw0, cx1, cy1, cz1, cw1, sliceW, k) => {
          const ndim = 4;
          // 4D identity Cholesky (10 elements: lower-triangular packed)
          const identity4DCholesky = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1];
          const baseChol = new Float32Array([...identity4DCholesky, ...identity4DCholesky]);

          const baseLoaded: LoadedGSplatsData = {
            positions: new Float32Array([cx0, cy0, cz0, cw0, cx1, cy1, cz1, cw1]),
            amplitudes: new Float32Array([1.0, 1.0]),
            choleskyFactors: baseChol,
            colors: null,
            splatCount: 2,
            ndim,
          };
          const baseViewState: GSplatsViewState = {
            displayDims: [0, 1, 2],
            slicePosition: [0, 0, 0, sliceW],
            tolerance: [1, 1, 1, 1],
          };

          // Shifted: both slicePos[3] AND center[3] for each splat shifted by k.
          // Display-dim coordinates (0,1,2) are unchanged.
          const shiftedLoaded: LoadedGSplatsData = {
            ...baseLoaded,
            positions: new Float32Array([cx0, cy0, cz0, cw0 + k, cx1, cy1, cz1, cw1 + k]),
            choleskyFactors: new Float32Array(baseChol),
          };
          const shiftedViewState: GSplatsViewState = {
            ...baseViewState,
            slicePosition: [0, 0, 0, sliceW + k],
          };

          const r0 = projectGSplatsTo3D(baseLoaded, baseViewState);
          const r1 = projectGSplatsTo3D(shiftedLoaded, shiftedViewState);

          // Same set of splats survive the visibility test.
          expect(r1.splatCount).toBe(r0.splatCount);
          // Same per-splat attenuation (hence same amplitudes).
          for (let i = 0; i < r0.amplitudes.length; i++) {
            expect(r1.amplitudes[i]).toBeCloseTo(r0.amplitudes[i], 4);
          }
          // Same projected 3D centers (display dims untouched by hidden shift).
          for (let i = 0; i < r0.centers3D.length; i++) {
            expect(r1.centers3D[i]).toBeCloseTo(r0.centers3D[i], 3);
          }
        }
      ),
      // Fixed seed for reproducible CI / bisect (P6: stochastic tests must be deterministic).
      { numRuns: 60, seed: 0x5eed }
    );
  });
});
