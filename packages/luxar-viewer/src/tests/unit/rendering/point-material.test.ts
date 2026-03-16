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
    CustomBlending: 'CustomBlending',
    AddEquation: 'AddEquation',
    OneFactor: 'OneFactor',
  };
});

describe('PointMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new PointMaterial();

      expect(material.uniforms.baseAlpha.value).toBe(config.shader.points.baseAlpha);
      expect(material.uniforms.opacity.value).toBe(1.0);
      expect(material.uniforms.invGamma.value).toBe(1.0);
      expect(material.userData.gamma).toBe(1.0); // gamma stored in userData, not uniforms

      // Check pre-computed pointSizeFactor (default 60 degrees, 1080p)
      const defaultTanHalfFov = Math.tan((60 * Math.PI) / 180 / 2);
      const expectedPointSizeFactor = (2.0 * 1080) / defaultTanHalfFov;
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material.uniforms.maxPointSize.value).toBe(1080 * 0.5);

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
      expect(material.userData.gamma).toBe(2.2); // gamma stored in userData
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.2);
      expect(material.blending).toBe('NormalBlending');
      expect(material.depthWrite).toBe(true);
    });
  });

  describe('shaders', () => {
    it('should have correct vertex shader with optimized world-space sizing', () => {
      const material = new PointMaterial();

      // Check for correct world-space sizing formula with radius scaling
      expect(material.vertexShader).toContain('float normalizedRadius = radius * radiusScale');

      // OPTIMIZATION: Check for inversesqrt with ortho branching
      expect(material.vertexShader).toContain('inversesqrt(dot(mvPosition.xyz, mvPosition.xyz))');

      // OPTIMIZATION: Check for pre-computed pointSizeFactor uniform
      expect(material.vertexShader).toContain('uniform float pointSizeFactor');
      expect(material.vertexShader).toContain(
        'float basePointSize = normalizedRadius * pointSizeFactor * invDistance'
      );

      // Check that gl_PointSize uses pre-computed maxPointSize
      expect(material.vertexShader).toContain('uniform float maxPointSize');
      expect(material.vertexShader).toContain(
        'gl_PointSize = max(1.0, min(pointSize, maxPointSize))'
      );

      // Check that sharpness compensation IS applied
      expect(material.vertexShader).toContain(
        'float sharpnessCompensation = 1.0 + (vSharpness - 1.0) * 0.15'
      );
      expect(material.vertexShader).toContain(
        'float pointSize = basePointSize * sharpnessCompensation'
      );

      // Check for attributes (GLSL ES 3.0 uses "in" instead of "attribute")
      expect(material.vertexShader).toContain('in float radius');
      expect(material.vertexShader).toContain('in float sharpness');

      // Check for optimized uniforms
      expect(material.vertexShader).toContain('uniform float radiusScale');
      expect(material.vertexShader).toContain('uniform float sharpnessScale');

      // Check for sharpness normalization and default handling
      expect(material.vertexShader).toContain(
        'float normalizedSharpness = sharpness * sharpnessScale'
      );
      expect(material.vertexShader).toContain(
        'vSharpness = normalizedSharpness > 0.0 ? normalizedSharpness : 2.0'
      );

      // Check for mediump precision on varyings (reduces register pressure)
      expect(material.vertexShader).toContain('out mediump vec3 vColor');
      expect(material.vertexShader).toContain('out mediump float vSharpness');
    });

    it('should have correct fragment shader with HDR handling and optimizations', () => {
      const material = new PointMaterial();

      // Check for optimizations in the shader
      expect(material.fragmentShader).toContain('vec2 centered = gl_PointCoord - 0.5');
      expect(material.fragmentShader).toContain('float r2 = dot(centered, centered)');

      // OPTIMIZATION: sqrt(4.0 * r2) combines sqrt and multiply
      expect(material.fragmentShader).toContain('mediump float normalizedR = sqrt(4.0 * r2)');

      // Check for uniforms (gamma removed from fragment shader, only invGamma used)
      expect(material.fragmentShader).toContain('uniform mediump float opacity');
      expect(material.fragmentShader).toContain('uniform mediump float baseAlpha');
      expect(material.fragmentShader).toContain('uniform mediump float invGamma');
      expect(material.fragmentShader).not.toContain('uniform float gamma'); // gamma removed

      // Check for simple falloff calculation with mediump
      expect(material.fragmentShader).toContain(
        'mediump float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness)'
      );
      // GOG model: intensity * color + offset, clip, gamma
      expect(material.fragmentShader).toContain('vColor * uIntensity + uOffset');
      expect(material.fragmentShader).toContain(
        'mediump vec3 finalColor = pow(adjusted, vec3(invGamma))'
      );
    });
  });

  describe('methods', () => {
    it('should update camera parameters with pre-computed values', () => {
      const material = new PointMaterial();
      const fov = (45 * Math.PI) / 180;
      const resolution = new THREE.Vector2(1920, 1080);

      material.updateCameraParams(fov, resolution);

      // Verify pointSizeFactor was computed correctly: 2.0 * resolution.y / tan(fov/2)
      const tanHalfFov = Math.tan(fov / 2);
      const expectedPointSizeFactor = (2.0 * 1080) / tanHalfFov;
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);

      // Verify maxPointSize is resolution.y * 0.5
      expect(material.uniforms.maxPointSize.value).toBe(1080 * 0.5);
    });

    it('should update opacity', () => {
      const material = new PointMaterial();

      material.updateOpacity(0.75);

      expect(material.uniforms.opacity.value).toBe(0.75);
    });

    it('should update gamma (stored in userData) and invGamma', () => {
      const material = new PointMaterial();

      material.updateGamma(1.8);

      expect(material.userData.gamma).toBe(1.8); // gamma stored in userData
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 1.8);
    });

    it('should clone material with current values', () => {
      const original = new PointMaterial({
        opacity: 0.5,
        gamma: 2.0,
      });

      const cloned = original.clone();

      expect(cloned.uniforms.opacity.value).toBe(0.5);
      expect(cloned.userData.gamma).toBe(2.0); // gamma stored in userData
      expect(cloned.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.0);

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

      // Check gl_PointSize has minimum of 1.0, uses pre-computed maxPointSize
      expect(material.vertexShader).toContain(
        'gl_PointSize = max(1.0, min(pointSize, maxPointSize))'
      );

      // Check comment about zero-radius filtering
      expect(material.vertexShader).toContain('Zero-radius filtering happens in fragment shader');
    });

    it('should discard zero-radius points in fragment shader', () => {
      const material = new PointMaterial();

      // Check for zero-radius discard
      expect(material.fragmentShader).toContain('if (vRadius < 0.0001)');
      expect(material.fragmentShader).toContain('discard');

      // Check that vRadius is passed from vertex shader (highp for precision)
      expect(material.vertexShader).toContain('out highp float vRadius');
      expect(material.vertexShader).toContain('vRadius = normalizedRadius');
    });

    it('should discard pixels outside circular area', () => {
      const material = new PointMaterial();

      // Check for optimized circle discard logic using squared distance
      expect(material.fragmentShader).toContain('if (r2 > 0.25)');
      expect(material.fragmentShader).toContain('discard');
    });

    it('should use inversesqrt optimization for distance calculation', () => {
      const material = new PointMaterial();

      // inversesqrt is a native GPU instruction (faster than length + divide)
      expect(material.vertexShader).toContain('inversesqrt');
      expect(material.vertexShader).toContain('dot(mvPosition.xyz, mvPosition.xyz)');

      // Should NOT use length() for distance calculation
      expect(material.vertexShader).not.toContain('length(mvPosition');
    });
  });

  describe('depth test configuration', () => {
    it('should have depthTest true by default', () => {
      const material = new PointMaterial();

      // Default depthTest is true
      expect(material.userData.depthTest).toBe(true);
    });

    it('should allow disabling depthTest via config', () => {
      const material = new PointMaterial({ depthTest: false });

      expect(material.userData.depthTest).toBe(false);
    });

    it('should use simple alpha output in fragment shader', () => {
      const material = new PointMaterial();

      // No uLuminous uniform - shader always uses same output pattern
      expect(material.fragmentShader).not.toContain('uniform bool uLuminous');
      expect(material.fragmentShader).not.toContain('if (uLuminous)');

      // Check for alpha output for AdditiveBlending (SrcAlpha, One)
      expect(material.fragmentShader).toContain('fragColor = vec4(finalColor, alpha)');
    });

    it('should preserve depthTest setting when cloning', () => {
      const original = new PointMaterial({ depthTest: false });
      const cloned = original.clone();

      expect(cloned.userData.depthTest).toBe(false);
    });

    it('should respect transparent config for opaque mode', () => {
      const opaqueMaterial = new PointMaterial({
        transparent: false,
        depthWrite: true,
      });

      expect(opaqueMaterial.transparent).toBe(false);
      expect(opaqueMaterial.depthWrite).toBe(true);
    });
  });
});
