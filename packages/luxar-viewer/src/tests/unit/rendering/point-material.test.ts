import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../rendering/point-material';
import { config } from '../../../config';

// Mock THREE.ShaderMaterial
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  const ShaderMaterial = vi.fn(function (this: any, params: any) {
    Object.assign(this, {
      uniforms: params.uniforms,
      vertexShader: params.vertexShader,
      fragmentShader: params.fragmentShader,
      vertexColors: params.vertexColors,
      transparent: params.transparent,
      depthWrite: params.depthWrite,
      toneMapped: params.toneMapped,
      blending: params.blending,
      userData: {},
      dispose: vi.fn(),
    });
  });

  return {
    ...actual,
    ShaderMaterial: ShaderMaterial as any,
    Vector2: actual.Vector2,
    AdditiveBlending: 'AdditiveBlending',
    NormalBlending: 'NormalBlending',
  };
});

describe('PointMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new PointMaterial();

      expect(material.uniforms.hdrMultiplier.value).toBe(config.shader.points.hdrMultiplier);
      expect(material.uniforms.baseAlpha.value).toBe(config.shader.points.baseAlpha);
      expect(material.uniforms.opacity.value).toBe(1.0);
      expect(material.uniforms.gamma.value).toBe(1.0);
      expect(material.uniforms.invGamma.value).toBe(1.0);
      // Check pre-computed tanHalfFov (default 60 degrees)
      expect(material.uniforms.tanHalfFov.value).toBeCloseTo(
        Math.tan((60 * Math.PI) / 180 / 2),
        10
      );
      expect(material.uniforms.resolution.value).toBeInstanceOf(THREE.Vector2);

      expect(material.vertexColors).toBe(true);
      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.toneMapped).toBe(false);
      expect(material.blending).toBe('AdditiveBlending');
    });

    it('should accept custom configuration', () => {
      const material = new PointMaterial({
        opacity: 0.5,
        gamma: 2.2,
        blending: 'NormalBlending' as any,
        depthWrite: true,
      });

      expect(material.uniforms.opacity.value).toBe(0.5);
      expect(material.uniforms.gamma.value).toBe(2.2);
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.2);
      expect(material.blending).toBe('NormalBlending');
      expect(material.depthWrite).toBe(true);
    });
  });

  describe('shaders', () => {
    it('should have correct vertex shader with world-space sizing formula', () => {
      const material = new PointMaterial();

      // Check for correct world-space sizing formula with radius scaling
      expect(material.vertexShader).toContain('float normalizedRadius = radius * radiusScale');

      // Check that gl_PointSize avoids undefined behavior (always >= 1.0)
      expect(material.vertexShader).toContain(
        'gl_PointSize = max(1.0, min(pointSize, resolution.y * 0.5))'
      );

      // Check that optimized point sizing uses pre-computed tanHalfFov
      expect(material.vertexShader).toContain(
        'float basePointSize = 2.0 * normalizedRadius * resolution.y / (distance * tanHalfFov)'
      );

      // Check that sharpness compensation IS applied
      expect(material.vertexShader).toContain(
        'float sharpnessCompensation = 1.0 + (vSharpness - 1.0) * 0.15'
      );
      expect(material.vertexShader).toContain(
        'float pointSize = basePointSize * sharpnessCompensation'
      );
      // Check for proper clamping to avoid undefined behavior
      expect(material.vertexShader).toContain(
        'gl_PointSize = max(1.0, min(pointSize, resolution.y * 0.5))'
      );

      // Check for attributes
      expect(material.vertexShader).toContain('attribute float radius');
      expect(material.vertexShader).toContain('attribute float sharpness');

      // Check for optimized uniforms (pre-computed tan(fov/2))
      expect(material.vertexShader).toContain('uniform float tanHalfFov');
      expect(material.vertexShader).toContain('uniform vec2 resolution');
      expect(material.vertexShader).toContain('uniform float radiusScale');
      expect(material.vertexShader).toContain('uniform float sharpnessScale');

      // Check for sharpness normalization and default handling
      expect(material.vertexShader).toContain(
        'float normalizedSharpness = sharpness * sharpnessScale'
      );
      expect(material.vertexShader).toContain(
        'vSharpness = normalizedSharpness > 0.0 ? normalizedSharpness : 2.0'
      );
    });

    it('should have correct fragment shader with HDR handling', () => {
      const material = new PointMaterial();

      // Check HDR is applied BEFORE gamma correction
      expect(material.fragmentShader).toContain('vec3 hdrColor = vColor * hdrMultiplier');

      // Check for optimizations in the shader
      expect(material.fragmentShader).toContain('vec2 centered = gl_PointCoord - 0.5');
      expect(material.fragmentShader).toContain('float r2 = dot(centered, centered)');

      // Check for uniforms
      expect(material.fragmentShader).toContain('uniform float hdrMultiplier');
      expect(material.fragmentShader).toContain('uniform float opacity');
      expect(material.fragmentShader).toContain('uniform float gamma');
      expect(material.fragmentShader).toContain('uniform float baseAlpha');
      expect(material.fragmentShader).toContain('uniform float invGamma');

      // Check for simple falloff calculation
      expect(material.fragmentShader).toContain(
        'float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness)'
      );
      expect(material.fragmentShader).toContain('vec3 finalColor = pow(hdrColor, vec3(invGamma))');
    });
  });

  describe('methods', () => {
    it('should update camera parameters', () => {
      const material = new PointMaterial();
      const fov = (45 * Math.PI) / 180;
      const resolution = new THREE.Vector2(1920, 1080);

      material.updateCameraParams(fov, resolution);

      // Verify tanHalfFov was computed correctly (fov is pre-computed as tan(fov/2))
      const expectedTanHalfFov = Math.tan(fov / 2);
      expect(material.uniforms.tanHalfFov.value).toBeCloseTo(expectedTanHalfFov, 10);
      expect(material.uniforms.resolution.value.x).toBe(1920);
      expect(material.uniforms.resolution.value.y).toBe(1080);
    });

    it('should update HDR multiplier', () => {
      const material = new PointMaterial();

      material.updateHDRMultiplier(32.0);

      expect(material.uniforms.hdrMultiplier.value).toBe(32.0);
    });

    it('should update opacity', () => {
      const material = new PointMaterial();

      material.updateOpacity(0.75);

      expect(material.uniforms.opacity.value).toBe(0.75);
    });

    it('should update gamma and invGamma', () => {
      const material = new PointMaterial();

      material.updateGamma(1.8);

      expect(material.uniforms.gamma.value).toBe(1.8);
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 1.8);
    });

    it('should clone material with current values', () => {
      const original = new PointMaterial({
        opacity: 0.5,
        gamma: 2.0,
      });

      original.updateHDRMultiplier(24.0);

      const cloned = original.clone();

      expect(cloned.uniforms.opacity.value).toBe(0.5);
      expect(cloned.uniforms.gamma.value).toBe(2.0);
      expect(cloned.uniforms.hdrMultiplier.value).toBe(24.0);

      // Ensure it's a new instance
      expect(cloned).not.toBe(original);
    });
  });

  describe('shader correctness', () => {
    it('should have correct sharpness compensation in vertex shader', () => {
      const material = new PointMaterial();

      // The old incorrect sharpness compensation should be removed
      expect(material.vertexShader).not.toContain('sizeCompensation');
      expect(material.vertexShader).not.toContain('sqrt(vSharpness / 2.0)');
      expect(material.vertexShader).not.toContain('pow(sharpness, 0.15)');

      // The new correct compensation should be present
      expect(material.vertexShader).toContain('sharpnessCompensation');
      expect(material.vertexShader).toContain('1.0 + (vSharpness - 1.0) * 0.15');
    });

    it('should clamp point size to avoid undefined behavior', () => {
      const material = new PointMaterial();

      // Check gl_PointSize has minimum of 1.0 to avoid undefined behavior
      expect(material.vertexShader).toContain(
        'gl_PointSize = max(1.0, min(pointSize, resolution.y * 0.5))'
      );

      // Check comment about zero-radius filtering
      expect(material.vertexShader).toContain('Zero-radius filtering happens in fragment shader');
    });

    it('should discard zero-radius points in fragment shader', () => {
      const material = new PointMaterial();

      // Check for zero-radius discard
      expect(material.fragmentShader).toContain('if (vRadius < 0.0001)');
      expect(material.fragmentShader).toContain('discard');

      // Check that vRadius is passed from vertex shader
      expect(material.vertexShader).toContain('varying float vRadius');
      expect(material.vertexShader).toContain('vRadius = normalizedRadius');
    });

    it('should discard pixels outside circular area', () => {
      const material = new PointMaterial();

      // Check for optimized circle discard logic using squared distance
      expect(material.fragmentShader).toContain('if (r2 > 0.25)');
      expect(material.fragmentShader).toContain('discard');
    });
  });
});
