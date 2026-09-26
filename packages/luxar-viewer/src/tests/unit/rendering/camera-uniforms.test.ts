import { describe, it, expect } from 'vitest';
import { computeMaxPointSize } from '../../../rendering/materials/_shared/camera-uniforms';

describe('camera-uniforms — computeMaxPointSize', () => {
  it('returns half the viewport height', () => {
    expect(computeMaxPointSize(1080)).toBe(540);
    expect(computeMaxPointSize(0)).toBe(0);
    expect(computeMaxPointSize(720)).toBe(360);
  });
});
