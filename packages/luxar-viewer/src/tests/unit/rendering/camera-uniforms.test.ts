import { describe, it, expect } from 'vitest';
import {
  computePointSizeFactor,
  computeMaxPointSize,
  computeFocalLength,
} from '../../../rendering/materials/_shared/camera-uniforms';

const PI = Math.PI;

describe('camera-uniforms — computePointSizeFactor', () => {
  it('matches the perspective formula 2*resY / tan(fov/2)', () => {
    const fov = PI / 3; // 60°
    const resY = 1080;
    const expected = (2 * resY) / Math.tan(fov / 2);

    expect(computePointSizeFactor(fov, resY, false)).toBeCloseTo(expected, 10);
  });

  it('matches the ortho formula (4*resY) / frustumHeight', () => {
    const frustumHeight = 5; // world units
    const resY = 720;
    const expected = (4 * resY) / frustumHeight;

    // computePointSizeFactor returns (2*resY)/(frustumHeight/2), which
    // simplifies to (4*resY)/frustumHeight — guard against the
    // simplification drifting.
    expect(computePointSizeFactor(frustumHeight, resY, true)).toBeCloseTo(expected, 10);
  });

  it('grows linearly with viewport height in perspective', () => {
    const fov = PI / 4;
    const a = computePointSizeFactor(fov, 600, false);
    const b = computePointSizeFactor(fov, 1200, false);
    expect(b / a).toBeCloseTo(2, 10);
  });

  it('shrinks as fov widens in perspective', () => {
    const resY = 1000;
    const narrow = computePointSizeFactor(PI / 6, resY, false); // 30°
    const wide = computePointSizeFactor(PI / 2, resY, false); // 90°
    expect(wide).toBeLessThan(narrow);
  });
});

describe('camera-uniforms — computeMaxPointSize', () => {
  it('returns half the viewport height', () => {
    expect(computeMaxPointSize(1080)).toBe(540);
    expect(computeMaxPointSize(0)).toBe(0);
    expect(computeMaxPointSize(720)).toBe(360);
  });
});

describe('camera-uniforms — computeFocalLength', () => {
  it('matches the perspective formula resY / (2*tan(fov/2))', () => {
    const fov = PI / 3;
    const resY = 1080;
    const expected = resY / (2 * Math.tan(fov / 2));

    expect(computeFocalLength(fov, resY, false)).toBeCloseTo(expected, 10);
  });

  it('matches the ortho formula resY / frustumHeight', () => {
    const frustumHeight = 4;
    const resY = 800;

    expect(computeFocalLength(frustumHeight, resY, true)).toBeCloseTo(resY / frustumHeight, 10);
  });

  it('agrees with the inline math in gsplat-material', () => {
    // Spot-check several configurations against the previous inline
    // formula so the helper is interchangeable.
    for (const fov of [PI / 6, PI / 4, PI / 3]) {
      for (const resY of [480, 1080, 2160]) {
        const tanHalf = Math.tan(fov / 2);
        const expected = resY / (2 * tanHalf);
        expect(computeFocalLength(fov, resY, false)).toBeCloseTo(expected, 10);
      }
    }
  });
});
