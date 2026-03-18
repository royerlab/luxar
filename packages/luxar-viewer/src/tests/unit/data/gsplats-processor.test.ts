import { describe, it, expect } from 'vitest';
import {
  processGSplatsTo3D,
  processGSplats3DOnly,
  processGSplats,
} from '../../../data/gsplats-processor';
import type { LoadedGSplatsData, GSplatsViewState } from '../../../types/gsplats';

describe('processGSplats3DOnly', () => {
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

    const result = processGSplats3DOnly(loaded);

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

    const result = processGSplats3DOnly(loaded);
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

    const result = processGSplats3DOnly(loaded);

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

    expect(() => processGSplats3DOnly(loaded)).toThrow('requires ndim=3');
  });
});

describe('processGSplatsTo3D', () => {
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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

    // 3D center should be [dim1, dim3, dim4] = [20, 40, 50]
    expect(result.centers3D[0]).toBe(20);
    expect(result.centers3D[1]).toBe(40);
    expect(result.centers3D[2]).toBe(50);
  });
});

describe('processGSplats', () => {
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

    const result = processGSplats(loaded, viewState);

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

    // Non-standard order: [2, 0, 1]
    const viewState: GSplatsViewState = {
      displayDims: [2, 0, 1],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = processGSplats(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // Centers are extracted in SORTED dimension order (for consistency with Cholesky)
    // sortedDisplayDims = [0, 1, 2], so output = [dim0, dim1, dim2] = [10, 20, 30]
    expect(result.centers3D[0]).toBe(10);
    expect(result.centers3D[1]).toBe(20);
    expect(result.centers3D[2]).toBe(30);
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

    const result = processGSplats(loaded, viewState);

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

    const result = processGSplats(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);
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
    const result4D = processGSplatsTo3D(loaded4D, viewState4D);

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
    const result5D = processGSplatsTo3D(loaded5D, viewState5D);

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
    const result3D = processGSplatsTo3D(loaded3D, viewState3D);

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

    const result = processGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // Attenuation = exp(-0.5 * (1.0 / sqrt(1.75))^2) = exp(-0.5 / 1.75) ≈ exp(-0.2857) ≈ 0.7514
    const expectedAttenuation = Math.exp(-0.5 * (1.0 / Math.sqrt(1.75)) ** 2);
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

    const result = processGSplatsTo3D(loaded, viewState);

    // Only splats 0 and 2 should survive
    expect(result.splatCount).toBe(2);

    // Splat 0: on slice, full amplitude
    expect(result.amplitudes[0]).toBeCloseTo(1.0, 5);
    // Color should be red (from splat 0, not green from splat 1)
    expect(result.colors[0]).toBeCloseTo(1.0, 5); // R
    expect(result.colors[1]).toBeCloseTo(0.0, 5); // G
    expect(result.colors[2]).toBeCloseTo(0.0, 5); // B

    // Splat 2: 0.5 units away, attenuation = exp(-0.5 * 0.5^2) = exp(-0.125) ≈ 0.8825
    const expectedAtt = Math.exp(-0.5 * 0.25);
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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

    // Time passes (discrete, exact match), wavelength offset = 2.0
    // Continuous Mahalanobis for dim 4 only: diff=2.0, sigma=1 → mahal=2.0
    // Attenuation = exp(-0.5 * 2.0^2) = exp(-2.0) ≈ 0.1353
    expect(result.splatCount).toBe(1);
    const expected = Math.exp(-0.5 * 4.0);
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

    const result = processGSplatsTo3D(loaded, viewState);

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

    const result = processGSplatsTo3D(loaded, viewState);

    expect(result.splatCount).toBe(1);
    // Gaussian attenuation: mahal=0.5 → exp(-0.5 * 0.25) ≈ 0.8825
    const expected = Math.exp(-0.5 * 0.25);
    expect(result.amplitudes[0]).toBeCloseTo(expected, 4);
  });
});
