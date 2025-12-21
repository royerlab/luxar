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
      centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
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
      sharpness: new Float32Array([2.0, 1.5, 2.5]),
      splatCount: 3,
      ndim: 3,
    };

    const result = processGSplats3DOnly(loaded);

    expect(result.splatCount).toBe(3);
    expect(result.centers3D.length).toBe(9);
    expect(result.amplitudes.length).toBe(3);
    expect(result.choleskyFactors3D.length).toBe(18);
    expect(result.colors.length).toBe(9);
    expect(result.sharpness.length).toBe(3);

    // Check values are copied correctly
    expect(result.centers3D[0]).toBe(0);
    expect(result.centers3D[3]).toBe(1);
    expect(result.amplitudes[0]).toBe(1.0);
    expect(result.amplitudes[1]).toBe(0.5);
    expect(result.sharpness[0]).toBe(2.0);
    expect(result.sharpness[1]).toBe(1.5);
  });

  it('should default sharpness to 2.0 if not present', () => {
    const loaded: LoadedGSplatsData = {
      centers: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      sharpness: null,
      splatCount: 1,
      ndim: 3,
    };

    const result = processGSplats3DOnly(loaded);

    expect(result.sharpness[0]).toBe(2.0);
  });

  it('should default colors to white if not present', () => {
    const loaded: LoadedGSplatsData = {
      centers: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      sharpness: null,
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
      centers: new Float32Array([0, 0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
      colors: null,
      sharpness: null,
      splatCount: 1,
      ndim: 4,
    };

    expect(() => processGSplats3DOnly(loaded)).toThrow('requires ndim=3');
  });
});

describe('processGSplatsTo3D', () => {
  it('should process 3D data with no hidden dims', () => {
    const loaded: LoadedGSplatsData = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      sharpness: new Float32Array([2.0, 2.0]),
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
      centers: new Float32Array([0, 0, 0, 0, 0, 0, 0, 10]),
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
      sharpness: new Float32Array([2.0, 2.0]),
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
      centers: new Float32Array([1, 2, 3, 0]), // 4D center
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
      sharpness: null,
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
      centers: new Float32Array([10, 20, 30, 40, 50]), // 5D center
      amplitudes: new Float32Array([1.0]),
      // Minimal valid Cholesky for 5D (15 elements)
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      colors: new Float32Array([1, 0, 0]),
      sharpness: new Float32Array([2.0]),
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
      centers: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      sharpness: null,
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
    expect(result.sharpness[0]).toBe(2.0); // Default sharpness
  });

  it('should use general path for non-standard display dims', () => {
    const loaded: LoadedGSplatsData = {
      centers: new Float32Array([10, 20, 30]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1, 0, 0]),
      sharpness: new Float32Array([2.0]),
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
      centers: new Float32Array([0, 0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
      colors: null,
      sharpness: null,
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
      centers: new Float32Array(0),
      amplitudes: new Float32Array(0),
      choleskyFactors: new Float32Array(0),
      colors: null,
      sharpness: null,
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
      centers: new Float32Array([0, 0, 0, 1, 1, 1]),
      amplitudes: new Float32Array([1.0, 0.0]), // Second has zero amplitude
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      colors: null,
      sharpness: null,
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

  it('should use default sharpness 2.0 when not provided', () => {
    const loaded: LoadedGSplatsData = {
      centers: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      sharpness: null, // No sharpness
      splatCount: 1,
      ndim: 3,
    };

    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = processGSplatsTo3D(loaded, viewState);

    expect(result.sharpness[0]).toBe(2.0);
  });
});
