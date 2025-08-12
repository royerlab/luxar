import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  createGaussianPointMaterial,
  ShaderValidator,
  SHADER_CONFIG,
} from '../rendering/shader-manager';
import * as THREE from 'three';

// Mock Three.js
vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  return {
    ...actual,
    ShaderMaterial: vi.fn().mockImplementation((params) => ({
      ...params,
      isShaderMaterial: true,
      dispose: vi.fn(),
    })),
  };
});

describe('shader-manager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('SHADER_CONFIG', () => {
    it('should have valid point configuration', () => {
      expect(SHADER_CONFIG.POINTS).toBeDefined();
      expect(SHADER_CONFIG.POINTS.size).toBe(8.0);
      expect(SHADER_CONFIG.POINTS.hdrMultiplier).toBe(16.0);
      expect(SHADER_CONFIG.POINTS.baseAlpha).toBe(0.01);
      expect(SHADER_CONFIG.POINTS.falloffSteepness).toBe(20.0);
    });
  });

  describe('createGaussianPointMaterial', () => {
    it('should create a shader material with correct properties', () => {
      createGaussianPointMaterial();

      expect(THREE.ShaderMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          uniforms: expect.objectContaining({
            hdrMultiplier: expect.objectContaining({ value: 16.0 }),
          }),
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          vertexColors: true,
          toneMapped: false,
        })
      );
    });

    it('should include vertex shader with radius and sharpness attributes', () => {
      createGaussianPointMaterial();

      const vertexShader = (THREE.ShaderMaterial as any).mock.calls[0][0].vertexShader;
      expect(vertexShader).toContain('attribute float radius');
      expect(vertexShader).toContain('attribute float sharpness');
      expect(vertexShader).toContain('varying float vSharpness');
      expect(vertexShader).toContain('gl_PointSize');
    });

    it('should include fragment shader with power-based falloff', () => {
      createGaussianPointMaterial();

      const fragmentShader = (THREE.ShaderMaterial as any).mock.calls[0][0].fragmentShader;
      expect(fragmentShader).toContain('varying float vSharpness');
      expect(fragmentShader).toContain('float falloff = pow(1.0 - normalizedR, vSharpness)');
      expect(fragmentShader).toContain('gl_FragColor');
    });

    it('should include size compensation in vertex shader', () => {
      createGaussianPointMaterial();

      const vertexShader = (THREE.ShaderMaterial as any).mock.calls[0][0].vertexShader;
      expect(vertexShader).toContain('float sizeCompensation = sqrt(vSharpness / 2.0)');
      // Updated to match world-space sizing implementation
      expect(vertexShader).toContain(
        'gl_PointSize = clamp(pointSize * sizeCompensation, 1.0, 500.0)'
      );
    });

    it('should use HDR multiplier uniform in fragment shader', () => {
      createGaussianPointMaterial();

      const fragmentShader = (THREE.ShaderMaterial as any).mock.calls[0][0].fragmentShader;
      expect(fragmentShader).toContain('uniform float hdrMultiplier');
      expect(fragmentShader).toContain('vec3 hdr = vColor * hdrMultiplier');
    });
  });

  describe('ShaderValidator', () => {
    let validator: ShaderValidator;

    beforeEach(() => {
      validator = new ShaderValidator();
    });

    it('should be instantiable', () => {
      expect(validator).toBeDefined();
      expect(validator).toBeInstanceOf(ShaderValidator);
    });
  });
});
