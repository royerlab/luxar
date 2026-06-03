/**
 * Audit acknowledgment (rendering.md [W1], [W4], [EXCLUDED-CATEGORY:
 * shader-tsl-parity]):
 *
 *   - The "shaders" / "shader correctness" describe blocks below
 *     contain `vertexShader.toContain('substring')` assertions
 *     ([W1]). These are deliberate "regression locks" — same
 *     pattern documented in `shader-hot-path.test.ts`. The structural
 *     shader contract is owned by the parity harness +
 *     Playwright visual regression suite; this file pins the
 *     wrapper-level uniform / config behaviour around the shader.
 *
 *   - The `vi.mock('three', ...)` block stubs `THREE.AdditiveBlending`
 *     etc. to literal string sentinels ([W4]). Mutation note: the
 *     assertions therefore test the mock's symbolic dispatch, not
 *     the real THREE enum. Keep the mock — the real
 *     `THREE.ShaderMaterial` constructor needs a WebGL context for
 *     uniforms, and these wrapper tests intentionally only verify
 *     the constructor argument shape. The TSL counterpart
 *     (`material-tsl.test.ts`) covers the actual blending-state
 *     contract against unmocked THREE constants.
 *
 * The same acknowledgment applies to the per-geometry GLSL test
 * files for lines + gsplats (rendering.md W1 / W5).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../../../rendering/materials/point/material-glsl';

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

      expect(material.uniforms.opacity.value).toBe(1.0);
      expect(material.uniforms.invGamma.value).toBe(1.0);
      expect(material.userData.gamma).toBe(1.0); // gamma stored in userData, not uniforms

      // Check pre-computed pointSizeFactor (default 60 degrees, 1080p)
      const defaultTanHalfFov = Math.tan((60 * Math.PI) / 180 / 2);
      const expectedPointSizeFactor = (2.0 * 1080) / defaultTanHalfFov;
      expect(material.uniforms.pointSizeFactor.value).toBeCloseTo(expectedPointSizeFactor, 5);
      expect(material.uniforms.maxPointSize.value).toBe(1080 * 0.5);

      // vertexColors is unconditionally false — the shader reads aColor
      // as an explicit InstancedBufferAttribute.
      expect(material.vertexColors).toBe(false);
      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.toneMapped).toBe(false);
      expect(material.blending).toBe('AdditiveBlending');
    });

    it('should accept custom configuration', () => {
      const material = new PointMaterial({
        opacity: 0.5,
        gamma: 2.2,
        blendingMode: 'normal',
      });

      expect(material.uniforms.opacity.value).toBe(0.5);
      expect(material.userData.gamma).toBe(2.2); // gamma stored in userData
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(material.blending).toBe('NormalBlending');
      // depthWrite is mode-derived (normal + opacity<0.99 → false),
      // matching the Line/GSplat canonical pattern.
      expect(material.depthWrite).toBe(false);
    });
  });

  describe('shaders', () => {
    it('should have correct vertex shader with optimized world-space sizing', () => {
      const material = new PointMaterial();

      // Radius normalization flows through sanitizeNonNegative from glsl-lib.
      // The input attribute is the per-instance `aRadius`.
      expect(material.vertexShader).toContain(
        'float normalizedRadius = sanitizeNonNegative(aRadius * radiusScale'
      );

      // OPTIMIZATION: Check for inversesqrt with ortho branching
      expect(material.vertexShader).toContain('inversesqrt(dot(mvPosition.xyz, mvPosition.xyz))');

      // OPTIMIZATION: Check for pre-computed pointSizeFactor uniform
      expect(material.vertexShader).toContain('uniform float pointSizeFactor');
      expect(material.vertexShader).toContain(
        'float basePointSize = normalizedRadius * pointSizeFactor * invDistance'
      );

      // Check pointSize clamp + sprite expansion (replaces gl_PointSize).
      expect(material.vertexShader).toContain('uniform float maxPointSize');
      expect(material.vertexShader).toContain('pointSize = max(1.0, min(pointSize, maxPointSize))');
      expect(material.vertexShader).toContain(
        'vec2 offsetClip = aQuadCorner * (pointSize / uResolution) * projCenter.w'
      );

      // Check that sharpness compensation IS applied
      expect(material.vertexShader).toContain(
        '1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)))'
      );
      expect(material.vertexShader).toContain(
        'float pointSize = basePointSize * sharpnessCompensation'
      );

      // Per-instance attributes. aQuadCorner is per-vertex.
      expect(material.vertexShader).toContain('in vec2 aQuadCorner');
      expect(material.vertexShader).toContain('in vec3 aCenter');
      expect(material.vertexShader).toContain('in float aRadius');
      expect(material.vertexShader).toContain('in float aSharpness');
      expect(material.vertexShader).toContain('in vec3 aColor');

      // Check for optimized uniforms
      expect(material.vertexShader).toContain('uniform float radiusScale');
      expect(material.vertexShader).toContain('uniform float sharpnessScale');
      expect(material.vertexShader).toContain('uniform vec2 uResolution');

      // Sharpness normalization flows through sanitizePositive from glsl-lib.
      expect(material.vertexShader).toContain(
        'float normalizedSharpness = sanitizePositive(aSharpness * sharpnessScale'
      );
      expect(material.vertexShader).toContain('vSharpness = normalizedSharpness');

      // GLSL sanitize lib is injected; verify the helper functions are present.
      expect(material.vertexShader).toContain('bool isInvalidFloat(float v)');
      expect(material.vertexShader).toContain('float sanitizePositive(float v, float fallback)');
      expect(material.vertexShader).toContain('float sanitizeNonNegative(float v, float fallback)');
      expect(material.vertexShader).toContain('isInvalidFloat(sharpnessCompensationRaw)');

      // Check for mediump precision on varyings (reduces register pressure)
      expect(material.vertexShader).toContain('out mediump vec3 vColor');
      expect(material.vertexShader).toContain('out mediump float vSharpness');
      // Sprite UV varying (replaces gl_PointCoord).
      expect(material.vertexShader).toContain('out mediump vec2 vSpriteCoord');
    });

    it('should have correct fragment shader with HDR handling and optimizations', () => {
      const material = new PointMaterial();

      // The fragment reads the sprite UV from a varying.
      expect(material.fragmentShader).toContain('vec2 centered = vSpriteCoord - 0.5');
      expect(material.fragmentShader).toContain('float r2 = dot(centered, centered)');

      // OPTIMIZATION: sqrt(4.0 * r2) combines sqrt and multiply
      expect(material.fragmentShader).toContain('mediump float normalizedR = sqrt(4.0 * r2)');

      // Check for uniforms (gamma removed from fragment shader, only invGamma used)
      expect(material.fragmentShader).toContain('uniform mediump float opacity');
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
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 1.8, 5);
    });

    it('toggles the LUXAR_GAMMA_ONE fast-path define across the gamma==1 threshold', () => {
      const material = new PointMaterial();

      // Non-unit gamma → fast-path define absent (the pow() runs).
      material.updateGamma(2.2);
      expect('LUXAR_GAMMA_ONE' in (material.defines ?? {})).toBe(false);

      // gamma == 1.0 → define present so the shader skips the pow().
      material.updateGamma(1.0);
      expect('LUXAR_GAMMA_ONE' in (material.defines ?? {})).toBe(true);

      // Back to non-unit → define removed again.
      material.updateGamma(1.8);
      expect('LUXAR_GAMMA_ONE' in (material.defines ?? {})).toBe(false);
    });

    it('should clone material with current values', () => {
      const original = new PointMaterial({
        opacity: 0.5,
        gamma: 2.0,
      });

      const cloned = original.clone();

      expect(cloned.uniforms.opacity.value).toBe(0.5);
      expect(cloned.userData.gamma).toBe(2.0); // gamma stored in userData
      expect(cloned.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.0, 5);

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
      expect(material.vertexShader).toContain(
        '1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)))'
      );
    });

    it('should clamp point size to avoid undefined behavior', () => {
      const material = new PointMaterial();

      // Point size clamps to [1, maxPointSize]; the sprite is then
      // expanded in NDC via aQuadCorner.
      expect(material.vertexShader).toContain('pointSize = max(1.0, min(pointSize, maxPointSize))');

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
    it('should have depthTest false for default additive mode', () => {
      // Default blendingMode is 'additive'; applyBlendingMode sets
      // depthTest=false for additive (renders on top, ignores depth).
      // Same canonical mode-derived state as Line/GSplat.
      const material = new PointMaterial();

      expect(material.userData.depthTest).toBe(false);
    });

    it('should have depthTest true for non-additive modes', () => {
      const luminous = new PointMaterial({ blendingMode: 'luminous' });
      expect(luminous.userData.depthTest).toBe(true);

      const normal = new PointMaterial({ blendingMode: 'normal' });
      expect(normal.userData.depthTest).toBe(true);
    });

    it('should allow explicit depthTest override via config', () => {
      // Explicit `depthTest` in config wins over mode-derived value
      // (mirrors the same override path in LineMaterial).
      const material = new PointMaterial({ blendingMode: 'normal', depthTest: false });

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

    it('should configure opaque mode correctly', () => {
      // Opaque mode: applyBlendingMode sets transparent=false +
      // depthWrite=true. Mirrors GSplatMaterial / LineMaterial.
      const opaqueMaterial = new PointMaterial({ blendingMode: 'opaque' });

      expect(opaqueMaterial.transparent).toBe(false);
      expect(opaqueMaterial.depthWrite).toBe(true);
    });
  });
});
