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
import { POINT_PICK_FRAGMENT_SHADER } from '../../../../../rendering/picking/point/shaders';

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

      expect(material.uniforms.uOpacity.value).toBe(1.0);
      expect(material.uniforms.uInvGamma.value).toBe(1.0);
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

      expect(material.uniforms.uOpacity.value).toBe(0.5);
      expect(material.userData.gamma).toBe(2.2); // gamma stored in userData
      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(material.blending).toBe('NormalBlending');
      // Points ALWAYS have depthWrite:false in `normal` (not opacity-gated) —
      // a point sprite stamps a flat depth plane across the whole disc, so
      // sorted transparency never depth-writes, mirroring the gsplat rule (#1002).
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

      // View-space depth sizing (matches lines/gsplats), ortho branch = 1.0.
      // 1e-20 = pure INF guard (scale-free, tiny-unit scenes keep
      // correct perspective sizing).
      expect(material.vertexShader).toContain('1.0 / max(-mvPosition.z, 1e-20)');

      // OPTIMIZATION: Check for pre-computed pointSizeFactor uniform
      expect(material.vertexShader).toContain('uniform float pointSizeFactor');
      expect(material.vertexShader).toContain(
        'float basePointSize = normalizedRadius * pointSizeFactor * invDistance'
      );

      // Check pointSize clamp + sprite expansion (replaces gl_PointSize).
      expect(material.vertexShader).toContain('uniform float maxPointSize');
      expect(material.vertexShader).toContain('float minPointSize = 1.5 * uPixelRatio');
      expect(material.vertexShader).toContain('clamp(basePointSize, minPointSize, maxPointSize)');
      expect(material.vertexShader).toContain(
        'vec2 offsetClip = aQuadCorner * (pointSize / uResolution) * projCenter.w'
      );

      // Sharpness compensation is GONE — the shifted-truncated super-Gaussian
      // truncates at the sprite edge, so the sprite size IS the visible extent.
      expect(material.vertexShader).not.toContain('sharpnessCompensation');
      expect(material.vertexShader).toContain('vPointSize = basePointSize');

      // The single per-instance attribute (aSortedIndex) + the point
      // data texture. aQuadCorner is per-vertex. Per-point values are
      // texelFetch'd into locals with the historical names (aCenter,
      // aRadius, aColor, aSharpness) so the downstream math is unchanged.
      expect(material.vertexShader).toContain('in vec2 aQuadCorner');
      expect(material.vertexShader).toContain('in uint aSortedIndex');
      expect(material.vertexShader).toContain('uniform highp sampler2D uPointTex');
      expect(material.vertexShader).toContain('vec3 aCenter = pointT0.xyz');
      expect(material.vertexShader).toContain('float aRadius = pointT0.w');
      expect(material.vertexShader).toContain('vec3 aColor = pointT1.rgb');
      expect(material.vertexShader).toContain('float aSharpness = pointT1.w');

      // Check for optimized uniforms (sharpnessScale is removed — sharpness
      // is now authored natively in [0, 1], no dtype scale needed).
      expect(material.vertexShader).toContain('uniform float radiusScale');
      expect(material.vertexShader).not.toContain('uniform float sharpnessScale');
      expect(material.vertexShader).toContain('uniform vec2 uResolution');
      expect(material.vertexShader).toContain('uniform float uPixelRatio');

      // sharpness -> beta mapping: beta = 2^(6s - 2), s clamped to [0, 1] and
      // NaN/Inf-guarded to the 0.5 default via sanitizeNonNegative.
      expect(material.vertexShader).toContain(
        'float s = clamp(sanitizeNonNegative(aSharpness, 0.5), 0.0, 1.0)'
      );
      expect(material.vertexShader).toContain('vBeta = exp2(6.0 * s - 2.0)');

      // GLSL sanitize lib is injected; verify the helper functions are present.
      expect(material.vertexShader).toContain('bool isInvalidFloat(float v)');
      expect(material.vertexShader).toContain('float sanitizePositive(float v, float fallback)');
      expect(material.vertexShader).toContain('float sanitizeNonNegative(float v, float fallback)');

      // Check for mediump precision on varyings (reduces register pressure)
      expect(material.vertexShader).toContain('out mediump vec3 vColor');
      expect(material.vertexShader).toContain('out mediump float vBeta');
      // Sprite UV varying (replaces gl_PointCoord).
      expect(material.vertexShader).toContain('out mediump vec2 vSpriteCoord');
    });

    it('should have correct fragment shader with HDR handling and optimizations', () => {
      const material = new PointMaterial();

      // Fragment-stage uniforms must be declared independently in GLSL.
      expect(material.fragmentShader).toContain('uniform float uPixelRatio');
      expect(POINT_PICK_FRAGMENT_SHADER).toContain('uniform float uPixelRatio');

      // The fragment reads the sprite UV from a varying.
      expect(material.fragmentShader).toContain('vec2 centered = vSpriteCoord - 0.5');
      expect(material.fragmentShader).toContain('float r2 = dot(centered, centered)');

      // OPTIMIZATION: sqrt(4.0 * r2) combines sqrt and multiply
      expect(material.fragmentShader).toContain('mediump float normalizedR = sqrt(4.0 * r2)');

      // Check for uniforms (gamma removed from fragment shader, only invGamma used)
      expect(material.fragmentShader).toContain('uniform mediump float uOpacity');
      expect(material.fragmentShader).toContain('uniform mediump float uInvGamma');
      expect(material.fragmentShader).not.toContain('uniform float gamma'); // gamma removed

      // Shifted-truncated super-Gaussian falloff (beta=2 reproduces the gsplat Gaussian).
      expect(material.fragmentShader).toContain(
        'mediump float falloff = max(exp(-K * pow(normalizedR, vBeta)) - C, 0.0) * INV_ONE_MINUS_C'
      );
      // GOG model: intensity * color + offset, clip, gamma
      expect(material.fragmentShader).toContain('vColor * uIntensity + uOffset');
      expect(material.fragmentShader).toContain(
        'mediump vec3 finalColor = pow(adjusted, vec3(uInvGamma))'
      );
    });
  });

  describe('methods', () => {
    it('clone() carries the camera-STATE uniforms (uIsOrtho, uNearCull, uResolution)', () => {
      // A clone taken in ortho mode used to keep the constructor
      // defaults (perspective branch, stale resolution/nearCull) until
      // the next global updateCameraParams broadcast. Lines clones are
      // the reference implementation.
      const original = new PointMaterial();
      original.updateCameraParams(2.0, new THREE.Vector2(640, 480), /*isOrtho=*/ true, 0.42);

      const cloned = original.clone();

      expect(cloned.uniforms.uIsOrtho.value).toBe(1);
      expect(cloned.uniforms.uNearCull.value).toBeCloseTo(0.42, 5);
      expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(640);
      expect((cloned.uniforms.uResolution.value as THREE.Vector2).y).toBe(480);
    });

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

      expect(material.uniforms.uOpacity.value).toBe(0.75);
    });

    it('should update gamma (stored in userData) and invGamma', () => {
      const material = new PointMaterial();

      material.updateGamma(1.8);

      expect(material.userData.gamma).toBe(1.8); // gamma stored in userData
      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 1.8, 5);
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

      expect(cloned.uniforms.uOpacity.value).toBe(0.5);
      expect(cloned.userData.gamma).toBe(2.0); // gamma stored in userData
      expect(cloned.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.0, 5);

      // Ensure it's a new instance
      expect(cloned).not.toBe(original);
    });
  });

  describe('shader correctness', () => {
    it('should map sharpness to the super-Gaussian exponent with no size compensation', () => {
      const material = new PointMaterial();

      // No size compensation of any kind — the truncated kernel sizes itself.
      expect(material.vertexShader).not.toContain('sizeCompensation');
      expect(material.vertexShader).not.toContain('sharpnessCompensation');
      expect(material.fragmentShader).not.toContain('vSharpness');

      // sharpness in [0, 1] -> beta = 2^(6s - 2); s=0.5 -> beta=2 (Gaussian).
      // exp2(6*0.5 - 2) = exp2(1) = 2.
      expect(material.vertexShader).toContain('vBeta = exp2(6.0 * s - 2.0)');
      expect(2 ** (6 * 0.5 - 2)).toBeCloseTo(2.0, 10);
      expect(2 ** (6 * 0.0 - 2)).toBeCloseTo(0.25, 10);
      expect(2 ** (6 * 1.0 - 2)).toBeCloseTo(16.0, 10);
    });

    it('has a C0-truncated super-Gaussian: falloff(0)=1, falloff(1)=0', () => {
      // Reproduce the fragment kernel constants in JS and check the endpoints.
      const K = Math.log(100); // ln(1/floor), floor = 0.01
      const C = Math.exp(-K); // = 0.01
      const invOneMinusC = 1 / (1 - C);
      const falloff = (rho: number, beta: number) =>
        Math.max(Math.exp(-K * rho ** beta) - C, 0) * invOneMinusC;

      for (const beta of [0.25, 0.71, 2.0, 5.66, 16.0]) {
        expect(falloff(0, beta)).toBeCloseTo(1.0, 6);
        expect(falloff(1, beta)).toBeCloseTo(0.0, 6);
      }
      // Monotonic shape ordering at the half-extent: higher beta = flatter,
      // harder-edged plateau (brighter mid-disk); lower beta = peakier cusp.
      expect(falloff(0.5, 0.25)).toBeLessThan(falloff(0.5, 2.0));
      expect(falloff(0.5, 2.0)).toBeLessThan(falloff(0.5, 16.0));
    });

    it('should clamp point size to avoid undefined behavior', () => {
      const material = new PointMaterial();

      // Point size clamps to [1.5 CSS px, maxPointSize] (the floor matches
      // the line shader; sub-pixel energy preserved via sizeScale^2);
      // the sprite is then expanded in NDC via aQuadCorner.
      expect(material.vertexShader).toContain('float minPointSize = 1.5 * uPixelRatio');

      // Check comment about zero-radius filtering
      expect(material.vertexShader).toContain('Zero-radius filtering happens in the fragment');
    });

    it('should discard zero-radius points in fragment shader', () => {
      const material = new PointMaterial();

      // Exact-zero discard only: an absolute epsilon discarded valid
      // sub-1e-4-unit radii (tiny-unit scenes rendered black).
      expect(material.fragmentShader).toContain('if (vRadius <= 0.0)');
      expect(material.fragmentShader).not.toContain('vRadius < 0.0001');
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

    it('should size from view-space depth, not Euclidean distance', () => {
      const material = new PointMaterial();

      // View-z (matches lines/gsplats): edge-of-screen points render
      // the same size as centered ones.
      expect(material.vertexShader).toContain('max(-mvPosition.z, 1e-20)');

      // Should NOT use Euclidean distance for sizing
      expect(material.vertexShader).not.toContain('length(mvPosition');
      expect(material.vertexShader).not.toContain('inversesqrt(dot(mvPosition');
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
      expect(opaqueMaterial.blending).toBe('CustomBlending');
      expect(opaqueMaterial.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(opaqueMaterial.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    });
  });
});
