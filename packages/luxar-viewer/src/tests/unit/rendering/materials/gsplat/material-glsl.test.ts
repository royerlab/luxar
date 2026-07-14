import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { GSplatMaterial } from '../../../../../rendering/materials/gsplat/material-glsl';
import {
  createGSplatQuadGeometry,
  createInstancedGSplatsMesh,
  packCholeskyForShader,
  updateInstancedGSplatsMesh,
} from '../../../../../rendering/gsplat-geometry';

// Mock THREE.ShaderMaterial
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  const ShaderMaterial = vi.fn(function (this: any, params: any) {
    Object.assign(this, {
      uniforms: params?.uniforms ?? {},
      vertexShader: params?.vertexShader ?? '',
      fragmentShader: params?.fragmentShader ?? '',
      transparent: params?.transparent,
      depthWrite: params?.depthWrite,
      toneMapped: params?.toneMapped,
      blending: params?.blending,
      side: params?.side,
      defines: params?.defines ?? {},
      userData: {},
      dispose: vi.fn(),
    });
  });

  // Add clone to prototype so subclasses (GSplatMaterial) can call super.clone() if needed
  ShaderMaterial.prototype.clone = function (this: any) {
    const cloned = new (ShaderMaterial as any)({
      vertexShader: this.vertexShader,
      fragmentShader: this.fragmentShader,
      defines: { ...this.defines },
      uniforms: Object.fromEntries(
        Object.entries(this.uniforms).map(([k, u]: [string, any]) => [
          k,
          {
            value:
              u &&
              typeof u.value === 'object' &&
              u.value !== null &&
              typeof u.value.clone === 'function'
                ? u.value.clone()
                : u?.value,
          },
        ])
      ),
      transparent: this.transparent,
      depthWrite: this.depthWrite,
      toneMapped: this.toneMapped,
      blending: this.blending,
      side: this.side,
    });
    cloned.blendEquation = this.blendEquation;
    cloned.blendSrc = this.blendSrc;
    cloned.blendDst = this.blendDst;
    cloned.blendEquationAlpha = this.blendEquationAlpha;
    cloned.blendSrcAlpha = this.blendSrcAlpha;
    cloned.blendDstAlpha = this.blendDstAlpha;
    cloned.userData = JSON.parse(JSON.stringify(this.userData || {}));
    return cloned;
  };

  return {
    ...actual,
    ShaderMaterial: ShaderMaterial as any,
    Vector2: actual.Vector2,
    AdditiveBlending: 'AdditiveBlending',
    NormalBlending: 'NormalBlending',
    CustomBlending: 'CustomBlending',
    AddEquation: 'AddEquation',
    MaxEquation: 'MaxEquation',
    OneFactor: 'OneFactor',
    DoubleSide: 'DoubleSide',
  };
});

describe('GSplatMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new GSplatMaterial();

      expect(material.uniforms.uOpacity.value).toBe(1.0);
      expect(material.uniforms.uTruncate.value).toBe(3.0);
      expect(material.uniforms.uResolution.value).toBeInstanceOf(THREE.Vector2);
      expect(material.uniforms.uFx.value).toBe(500);
      expect(material.uniforms.uFy.value).toBe(500);

      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.toneMapped).toBe(false);
      // Default 'additive' takes the SHARED blending state (AdditiveBlending,
      // SrcAlpha + One). With the shader's alpha=1.0 contract, SrcAlpha is
      // the identity factor — pixel-identical to the historical
      // CustomBlending One/One dance, now unified with the TSL wrapper.
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(material.blendDst).toBe('OneFactor');
      // Symmetric alpha channel — the alpha-MaxEquation overflow guard is
      // gone (raw-scene-hdr capture sanitizes alpha at readback instead).
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.blendSrcAlpha).toBe(null);
      expect(material.blendDstAlpha).toBe(null);
      expect(material.side).toBe('DoubleSide');
      expect(material.uniforms.uProjectionMode.value).toBe(0); // Default additive uses sum projection
      // 'additive' ignores depth (depthTest=false)
      expect(material.userData.depthTest).toBe(false);
    });

    it('should default invGamma to 1.0', () => {
      const material = new GSplatMaterial();

      expect(material.uniforms.uInvGamma.value).toBe(1.0);
      expect(material.userData.gamma).toBe(1.0);
    });

    it('should accept custom gamma', () => {
      const material = new GSplatMaterial({ gamma: 2.2 });

      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(material.userData.gamma).toBe(2.2);
    });

    it('sets the LUXAR_GAMMA_ONE fast-path define at the default gamma==1', () => {
      // Default gamma is 1.0 → the constructor compiles in the fast path.
      expect('LUXAR_GAMMA_ONE' in new GSplatMaterial().defines).toBe(true);
      // Non-unit gamma → absent (the pow() runs).
      expect('LUXAR_GAMMA_ONE' in new GSplatMaterial({ gamma: 2.2 }).defines).toBe(false);
    });

    it('toggles the LUXAR_GAMMA_ONE define across the gamma==1 threshold', () => {
      const material = new GSplatMaterial({ gamma: 2.2 });
      expect('LUXAR_GAMMA_ONE' in material.defines).toBe(false);

      material.updateGamma(1.0);
      expect('LUXAR_GAMMA_ONE' in material.defines).toBe(true);

      material.updateGamma(1.8);
      expect('LUXAR_GAMMA_ONE' in material.defines).toBe(false);
    });

    it('should accept custom configuration', () => {
      const material = new GSplatMaterial({
        opacity: 0.5,
        truncationRadius: 4.0,
        blendingMode: 'normal',
      });

      expect(material.uniforms.uOpacity.value).toBe(0.5);
      expect(material.uniforms.uTruncate.value).toBe(4.0);
      // normal mode = gsplat premultiplied alpha-over: CustomBlending
      // One/OneMinusSrcAlpha (see getGSplatNormalBlendingState).
      expect(material.blending).toBe('CustomBlending');
      // GSplat normal mode never writes depth (no opacity gate).
      expect(material.depthWrite).toBe(false);
      expect(material.uniforms.uProjectionMode.value).toBe(0); // Sum projection for normal
    });

    it('should configure max blending with max projection', () => {
      const material = new GSplatMaterial({
        blendingMode: 'max',
      });

      expect(material.blending).toBe('CustomBlending');
      expect(material.blendEquation).toBe('MaxEquation');
      expect(material.blendSrc).toBe('OneFactor');
      expect(material.blendDst).toBe('OneFactor');
      expect(material.depthWrite).toBe(false); // Max blending disables depth write
      expect(material.uniforms.uProjectionMode.value).toBe(1); // Max projection
    });

    it('should configure luminous mode via the shared additive state + depth test', () => {
      const material = new GSplatMaterial({ blendingMode: 'luminous' });

      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(material.blendDst).toBe('OneFactor');
      // Symmetric alpha channel (shared state; no per-alpha overrides).
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.blendSrcAlpha).toBe(null);
      expect(material.blendDstAlpha).toBe(null);
      // luminous = additive visuals + depth occlusion.
      expect(material.userData.depthTest).toBe(true);
    });
  });

  describe('shaders', () => {
    it('should have correct vertex shader structure', () => {
      const material = new GSplatMaterial();

      // Check for GLSL ES 3.0 syntax: "in" for attributes, "flat out" for varyings
      // (glslVersion: THREE.GLSL3 is set in constructor, THREE.js adds #version 300 es)

      // Check for quad corner attribute (GLSL ES 3.0 uses "in" instead of "attribute")
      expect(material.vertexShader).toContain('in vec2 aQuadCorner');

      // Check for per-instance attributes
      expect(material.vertexShader).toContain('in vec3 aCenter');
      expect(material.vertexShader).toContain('in vec2 aCholesky01');
      expect(material.vertexShader).toContain('in vec2 aCholesky23');
      expect(material.vertexShader).toContain('in vec2 aCholesky45');
      expect(material.vertexShader).toContain('in float aAmplitude');
      expect(material.vertexShader).toContain('in vec3 aColor');

      // Check for uniforms
      expect(material.vertexShader).toContain('uniform vec2 uResolution');
      expect(material.vertexShader).toContain('uniform float uFx, uFy');
      expect(material.vertexShader).toContain('uniform float uTruncate');
      expect(material.vertexShader).toContain('uniform int uProjectionMode');

      // Check for flat varyings (GLSL ES 3.0 uses "flat out" for non-interpolated values)
      expect(material.vertexShader).toContain('flat out mediump vec3 vColor');
      expect(material.vertexShader).toContain('flat out mediump float vAmplitude2D');
      expect(material.vertexShader).toContain('flat out highp vec3 vL2D');
      expect(material.vertexShader).toContain('flat out highp vec2 vCenterScreen');
    });

    it('should have Cholesky unpacking function', () => {
      const material = new GSplatMaterial();

      // Check for unpackCholesky3D function
      expect(material.vertexShader).toContain('mat3 unpackCholesky3D()');
      expect(material.vertexShader).toContain('return mat3(');
    });

    it('should have 2D Cholesky computation with reciprocal optimization', () => {
      const material = new GSplatMaterial();

      expect(material.vertexShader).toContain('vec3 cholesky2x2(mat2 S)');
      expect(material.vertexShader).toContain('bool invalidFloat(float v)');
      expect(material.vertexShader).toContain('bool invalidCov2D(mat2 S)');
      expect(material.vertexShader).toContain('float L00 = sqrt');
      // OPTIMIZATION: Uses reciprocal multiplication instead of division
      expect(material.vertexShader).toContain('float invL00 = 1.0 / L00');
      expect(material.vertexShader).toContain('float L10 = s10 * invL00');
      expect(material.vertexShader).toContain('float L11 = sqrt');
      expect(material.vertexShader).toContain('float invL11 = 1.0 / L11');
      // Returns reciprocals for faster fragment shader
      expect(material.vertexShader).toContain('return vec3(invL00, L10, invL11)');
    });

    it('should guard invalid projected covariance and amplitudes', () => {
      const material = new GSplatMaterial();

      expect(material.vertexShader).toContain('invalidCov2D(Sigma2D)');
      expect(material.vertexShader).toContain('invalidFloat(aAmplitude)');
      expect(material.vertexShader).toContain('return isnan(v) || isinf(v)');
    });

    it('should use uniform ray integral factor for shifted Gaussian', () => {
      const material = new GSplatMaterial();

      // Shifted Gaussian ray integral factor passed as uniform (precomputed in TypeScript)
      expect(material.vertexShader).toContain('sigmaRay * uRayIntegralFactor');
    });

    it('should have near-plane guard with smooth fade and screen-coverage cull', () => {
      const material = new GSplatMaterial();

      // Near-cull uses the shared perspectiveNearFade helper (smooth
      // fade, not hard discard) with the uNearCull uniform
      expect(material.vertexShader).toContain('perspectiveNearFade(uIsOrtho, centerCam.z, uNearCull)');
      // Screen-coverage fade uses projected extent and uMaxExtentFactor
      expect(material.vertexShader).toContain('uMaxExtentFactor');
      expect(material.vertexShader).toContain('projectedExtent');
      // Hard cull only when fully faded
      expect(material.vertexShader).toContain('gl_Position = vec4(0.0, 0.0, -2.0, 1.0)');
      // Ortho passes through the helper (fade = 1; NDC clipping is the
      // cull authority) — the helper body carries the isOrtho branch.
      expect(material.vertexShader).toContain('if (isOrtho == 1) return 1.0;');
      // nearFade applied to amplitude
      expect(material.vertexShader).toContain('nearFade');
    });

    it('should NOT redeclare built-in THREE.js uniforms', () => {
      const material = new GSplatMaterial();

      // Bug #10: Shader was declaring modelViewMatrix and projectionMatrix
      // which are built-in THREE.js uniforms, causing compilation errors.
      // This test prevents regression.

      // The shader should use these built-ins but NOT declare them
      expect(material.vertexShader).toContain('modelViewMatrix'); // Uses it
      expect(material.vertexShader).toContain('projectionMatrix'); // Uses it

      // Should NOT have uniform declarations for these (would cause redefinition error)
      expect(material.vertexShader).not.toContain('uniform mat4 modelViewMatrix');
      expect(material.vertexShader).not.toContain('uniform mat4 projectionMatrix');
    });

    it('should compute eigenvalues for oriented quad', () => {
      const material = new GSplatMaterial();

      expect(material.vertexShader).toContain('float trace = Sigma2D[0][0] + Sigma2D[1][1]');
      expect(material.vertexShader).toContain('float det = Sigma2D[0][0] * Sigma2D[1][1]');
      expect(material.vertexShader).toContain('float lambda1 =');
      expect(material.vertexShader).toContain('float lambda2 =');
    });

    it('should have correct fragment shader with Mahalanobis distance', () => {
      const material = new GSplatMaterial();

      // Check for uniforms (mediump for GPU optimization)
      expect(material.fragmentShader).toContain('uniform mediump float uOpacity');
      expect(material.fragmentShader).toContain('uniform mediump float uInvGamma');

      // Check for GOG model (gain-offset-gamma)
      expect(material.fragmentShader).toContain('vColor * uIntensity + uOffset');
      expect(material.fragmentShader).toContain('pow(adjusted, vec3(uInvGamma))');

      // Check for forward substitution to solve L·y = d
      // OPTIMIZATION: Uses multiplication with precomputed reciprocals instead of division
      expect(material.fragmentShader).toContain('float y0 = d.x * vL2D.x'); // MUL with invL00
      expect(material.fragmentShader).toContain('float y1 = (d.y - vL2D.y * y0) * vL2D.z'); // MUL with invL11

      // Check for Mahalanobis distance
      expect(material.fragmentShader).toContain('float mahalSq = y0 * y0 + y1 * y1');

      // Check for early discard at truncation radius (uniform, not hardcoded)
      expect(material.fragmentShader).toContain('if (mahalSq > uTruncateSq) discard');

      // Check for shifted Gaussian falloff with C⁰ continuity
      expect(material.fragmentShader).toContain(
        'float intensity = vAmplitude2D * uInvOneMinusC * max(exp(-0.5 * mahalSq) - uShiftC, 0.0)'
      );
    });

    it('should discard negligible contributions', () => {
      const material = new GSplatMaterial();

      // Higher threshold (1e-4) for better performance while still invisible
      expect(material.fragmentShader).toContain('if (intensity < 1e-4) discard');
    });
  });

  describe('methods', () => {
    it('should update camera parameters', () => {
      const material = new GSplatMaterial();
      const fov = (45 * Math.PI) / 180;
      const resolution = new THREE.Vector2(1920, 1080);

      material.updateCameraParams(fov, resolution);

      expect(material.uniforms.uResolution.value.x).toBe(1920);
      expect(material.uniforms.uResolution.value.y).toBe(1080);

      // Check focal length computation (fy = height / (2 * tan(fov/2)))
      const tanHalfFov = Math.tan(fov / 2);
      const expectedFy = resolution.y / (2 * tanHalfFov);
      expect(material.uniforms.uFy.value).toBeCloseTo(expectedFy, 5);
      expect(material.uniforms.uFx.value).toBeCloseTo(expectedFy, 5); // Same for square pixels
    });

    it('should update opacity', () => {
      const material = new GSplatMaterial();

      material.updateOpacity(0.75);

      expect(material.uniforms.uOpacity.value).toBe(0.75);
    });

    it('should update truncation radius', () => {
      const material = new GSplatMaterial();

      material.updateTruncationRadius(5.0);

      expect(material.uniforms.uTruncate.value).toBe(5.0);
    });

    it('should update gamma', () => {
      const material = new GSplatMaterial();

      material.updateGamma(2.2);

      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(material.userData.gamma).toBe(2.2);
    });

    it('should clamp gamma to prevent division by zero', () => {
      const material = new GSplatMaterial();

      material.updateGamma(0);

      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 0.001, 5);
      expect(material.userData.gamma).toBe(0.001);
    });

    it('should clone material with current values including gamma', () => {
      const original = new GSplatMaterial({
        opacity: 0.5,
        gamma: 2.2,
        truncationRadius: 4.0,
      });

      const cloned = original.clone();

      expect(cloned.uniforms.uOpacity.value).toBe(0.5);
      expect(cloned.uniforms.uTruncate.value).toBe(4.0);
      expect(cloned.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(cloned.userData.gamma).toBe(2.2);

      // Clone preserves the (symmetric) alpha blend channel state.
      expect(cloned.blendEquationAlpha).toBe(null);
      expect(cloned.blendSrcAlpha).toBe(null);
      expect(cloned.blendDstAlpha).toBe(null);

      // Ensure it's a new instance
      expect(cloned).not.toBe(original);
    });

    it('clone preserves a tuned uMaxExtentFactor (was silently reset to 0.33)', () => {
      const material = new GSplatMaterial({ maxExtentFactor: 0.7 });
      expect(material.uniforms.uMaxExtentFactor.value).toBe(0.7);

      const cloned = material.clone();
      expect(cloned.uniforms.uMaxExtentFactor.value).toBe(0.7);
    });
  });

  describe('blending mode depth test configuration', () => {
    it('should have depthTest false for additive mode (ignores depth)', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });

      // 'additive' ignores depth entirely (renders on top of everything)
      expect(material.userData.depthTest).toBe(false);
      // Shared blending state (unified with TSL): AdditiveBlending —
      // SrcAlpha is identity under the shader's alpha=1.0 contract.
      expect(material.blending).toBe('AdditiveBlending');
    });

    it('should have depthTest true for luminous mode (respects depth occlusion)', () => {
      const material = new GSplatMaterial({ blendingMode: 'luminous' });

      // 'luminous' respects depth occlusion but uses same visual output as additive
      expect(material.userData.depthTest).toBe(true);
      // Shared blending state (unified with TSL): AdditiveBlending.
      expect(material.blending).toBe('AdditiveBlending');
    });

    it('should have depthTest true for normal mode (premultiplied alpha-over)', () => {
      const material = new GSplatMaterial({ blendingMode: 'normal' });

      expect(material.userData.depthTest).toBe(true);
      // Premultiplied coverage-alpha state — CustomBlending with
      // One / OneMinusSrcAlpha, NOT the generic NormalBlending
      // (the shader premultiplies; SrcAlpha would double-multiply).
      expect(material.blending).toBe('CustomBlending');
      expect(material.blendSrc).toBe('OneFactor');
      expect(material.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    });

    it('should apply opacity directly to RGB for linear additive blending', () => {
      const material = new GSplatMaterial();

      // No uLuminous uniform - shader always uses same output pattern
      expect(material.fragmentShader).not.toContain('uniform bool uLuminous');
      expect(material.fragmentShader).not.toContain('if (uLuminous)');

      // With OneFactor blending, opacity is applied directly to RGB (alpha is ignored)
      // This ensures correct linear sum projection without squaring intensity
      // Gamma correction is applied before intensity multiplication
      expect(material.fragmentShader).toContain(
        'vec3 finalColor = gammaColor * intensity * uOpacity'
      );
      expect(material.fragmentShader).toContain('fragColor = vec4(finalColor, 1.0)');
    });

    it('should configure opaque mode correctly', () => {
      const material = new GSplatMaterial({ blendingMode: 'opaque' });

      expect(material.transparent).toBe(false);
      expect(material.depthWrite).toBe(true);
      expect(material.blending).toBe('NormalBlending');
      expect(material.userData.depthTest).toBe(true);
    });
  });

  describe('applyBlendingMode', () => {
    // The constructor delegates to applyBlendingMode, so initial-state
    // tests exist above. These cover live transitions — the bug the
    // method fixes is the layers panel's previous generic
    // `mat.blending = ...` write that ignored uProjectionMode and
    // OneFactor / alpha-equation requirements.

    it('switches additive → max: uProjectionMode flips to 1, blendEquation to MaxEquation', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });
      expect(material.uniforms.uProjectionMode.value).toBe(0);

      material.applyBlendingMode('max');

      expect(material.uniforms.uProjectionMode.value).toBe(1);
      expect(material.blending).toBe('CustomBlending');
      expect(material.blendEquation).toBe('MaxEquation');
      expect(material.blendSrc).toBe('OneFactor');
      expect(material.blendDst).toBe('OneFactor');
      expect(material.userData.blendingMode).toBe('max');
      expect(material.needsUpdate).toBe(true);
    });

    it('switches max → additive: uProjectionMode resets to 0, shared additive state', () => {
      // Before this fix, switching from max → additive via the layers
      // panel left blendEquation stuck at MaxEquation (wrong intensity)
      // and uProjectionMode stuck at 1 (shader took max-projection
      // branch but framebuffer blended additively).
      const material = new GSplatMaterial({ blendingMode: 'max' });
      expect(material.uniforms.uProjectionMode.value).toBe(1);
      expect(material.blendEquation).toBe('MaxEquation');

      material.applyBlendingMode('additive');

      expect(material.uniforms.uProjectionMode.value).toBe(0);
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(material.blendDst).toBe('OneFactor');
      // Symmetric alpha channel — no stranded MaxEquation state.
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.blendSrcAlpha).toBe(null);
      expect(material.blendDstAlpha).toBe(null);
      expect(material.userData.blendingMode).toBe('additive');
    });

    it('switches max → luminous: same factors as additive, but depthTest=true', () => {
      const material = new GSplatMaterial({ blendingMode: 'max' });

      material.applyBlendingMode('luminous');

      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(material.blendDst).toBe('OneFactor');
      expect(material.uniforms.uProjectionMode.value).toBe(0);
      // luminous respects depth (vs additive which ignores it).
      expect(material.depthTest).toBe(true);
      expect(material.userData.blendingMode).toBe('luminous');
    });

    it('switches additive → opaque: transparent flips, alpha state resets to defaults', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });
      // additive keeps a symmetric alpha channel (shared state).
      expect(material.blendEquationAlpha).toBe(null);

      material.applyBlendingMode('opaque');

      expect(material.blending).toBe('NormalBlending');
      expect(material.transparent).toBe(false);
      expect(material.depthWrite).toBe(true);
      // Alpha state cleared so it doesn't haunt a future custom-blending switch.
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.blendSrcAlpha).toBe(null);
      expect(material.blendDstAlpha).toBe(null);
      expect(material.userData.blendingMode).toBe('opaque');
    });

    it('round-trips additive → max → additive without stranding state', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });
      material.applyBlendingMode('max');
      material.applyBlendingMode('additive');

      // Should be identical to a freshly-constructed additive material.
      expect(material.uniforms.uProjectionMode.value).toBe(0);
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(material.blendDst).toBe('OneFactor');
      expect(material.blendEquationAlpha).toBe(null);
    });

    it('clone after applyBlendingMode preserves the live mode', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });
      material.applyBlendingMode('max');

      const cloned = material.clone();

      expect(cloned.userData.blendingMode).toBe('max');
      expect(cloned.uniforms.uProjectionMode.value).toBe(1);
      expect(cloned.blendEquation).toBe('MaxEquation');
    });
  });

  describe('normal mode — premultiplied coverage alpha (LUXAR_NORMAL_PREMULT)', () => {
    it('constructor normal: define set, One/OneMinusSrcAlpha, symmetric alpha, depthWrite off', () => {
      const material = new GSplatMaterial({ blendingMode: 'normal' });

      expect('LUXAR_NORMAL_PREMULT' in material.defines).toBe(true);
      expect(material.blending).toBe('CustomBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe('OneFactor');
      expect(material.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
      // Symmetric alpha channel (null = track RGB) — separate alpha
      // equation state trips gl.getError() under the WebGPU bridge.
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.blendSrcAlpha).toBe(null);
      expect(material.blendDstAlpha).toBe(null);
      expect(material.transparent).toBe(true);
      expect(material.depthTest).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.uniforms.uProjectionMode.value).toBe(0); // sum projection
    });

    it('depthWrite stays OFF at opacity 1.0 (no generic opacity>=0.99 gate)', () => {
      // The generic normal state flips depthWrite on at high opacity; a
      // coverage-alpha splat fragment with alpha ~1e-4 writing depth
      // would punch occlusion halos, so the gsplat state never writes.
      const material = new GSplatMaterial({ blendingMode: 'normal', opacity: 1.0 });
      expect(material.depthWrite).toBe(false);
    });

    it('fragment shader has both output branches (premult under the define)', () => {
      const material = new GSplatMaterial({ blendingMode: 'normal' });
      expect(material.fragmentShader).toContain('#ifdef LUXAR_NORMAL_PREMULT');
      expect(material.fragmentShader).toContain('clamp(intensity * uOpacity, 0.0, 1.0)');
      expect(material.fragmentShader).toContain('fragColor = vec4(finalColor, coverage)');
      // The alpha=1.0 contract stays intact for every other mode.
      expect(material.fragmentShader).toContain('fragColor = vec4(finalColor, 1.0)');
    });

    it('round-trips normal → additive → normal without stranding state', () => {
      const material = new GSplatMaterial({ blendingMode: 'normal' });

      material.applyBlendingMode('additive');
      // Define removed; shared additive state applied.
      expect('LUXAR_NORMAL_PREMULT' in material.defines).toBe(false);
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(material.blendDst).toBe('OneFactor');
      expect(material.depthTest).toBe(false);

      material.applyBlendingMode('normal');
      // Identical to a freshly-constructed normal material.
      expect('LUXAR_NORMAL_PREMULT' in material.defines).toBe(true);
      expect(material.blending).toBe('CustomBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.blendSrc).toBe('OneFactor');
      expect(material.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
      expect(material.blendEquationAlpha).toBe(null);
      expect(material.depthWrite).toBe(false);
      expect(material.transparent).toBe(true);
      expect(material.needsUpdate).toBe(true);
    });

    it('normal → max: define removed, MaxEquation + projection-mode flip', () => {
      const material = new GSplatMaterial({ blendingMode: 'normal' });

      material.applyBlendingMode('max');

      expect('LUXAR_NORMAL_PREMULT' in material.defines).toBe(false);
      expect(material.blendEquation).toBe('MaxEquation');
      expect(material.uniforms.uProjectionMode.value).toBe(1);
    });

    it('clone preserves normal mode and the define', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });
      material.applyBlendingMode('normal');

      const cloned = material.clone();

      expect(cloned.userData.blendingMode).toBe('normal');
      expect('LUXAR_NORMAL_PREMULT' in cloned.defines).toBe(true);
      expect(cloned.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
      expect(cloned.depthWrite).toBe(false);
    });
  });
});

describe('createGSplatQuadGeometry', () => {
  it('should create geometry with quad corners', () => {
    const geometry = createGSplatQuadGeometry();

    const quadCorner = geometry.getAttribute('aQuadCorner');
    expect(quadCorner).toBeDefined();
    expect(quadCorner.count).toBe(4);
    expect(quadCorner.itemSize).toBe(2);

    // Check corner values: (-1,-1), (1,-1), (-1,1), (1,1)
    const array = quadCorner.array as Float32Array;
    expect(array[0]).toBe(-1); // x0
    expect(array[1]).toBe(-1); // y0
    expect(array[2]).toBe(1); // x1
    expect(array[3]).toBe(-1); // y1
    expect(array[4]).toBe(-1); // x2
    expect(array[5]).toBe(1); // y2
    expect(array[6]).toBe(1); // x3
    expect(array[7]).toBe(1); // y3
  });

  it('should have correct indices for two triangles', () => {
    const geometry = createGSplatQuadGeometry();

    const index = geometry.getIndex();
    expect(index).not.toBeNull();
    expect(index!.count).toBe(6);

    const indices = index!.array as Uint16Array;
    // First triangle: 0, 1, 2
    expect(indices[0]).toBe(0);
    expect(indices[1]).toBe(1);
    expect(indices[2]).toBe(2);
    // Second triangle: 2, 1, 3
    expect(indices[3]).toBe(2);
    expect(indices[4]).toBe(1);
    expect(indices[5]).toBe(3);
  });
});

describe('packCholeskyForShader', () => {
  it('should pack Cholesky factors into shader attribute format', () => {
    // 2 splats, 6 elements each: [L00, L10, L11, L20, L21, L22]
    const choleskyFactors = new Float32Array([
      1.0,
      0.5,
      2.0,
      0.3,
      0.4,
      3.0, // Splat 0
      4.0,
      0.1,
      5.0,
      0.2,
      0.6,
      6.0, // Splat 1
    ]);

    const { cholesky01, cholesky23, cholesky45 } = packCholeskyForShader(choleskyFactors, 2);

    // Check cholesky01: [L00, L10] per splat
    expect(cholesky01[0]).toBeCloseTo(1.0, 5); // L00 splat 0
    expect(cholesky01[1]).toBeCloseTo(0.5, 5); // L10 splat 0
    expect(cholesky01[2]).toBeCloseTo(4.0, 5); // L00 splat 1
    expect(cholesky01[3]).toBeCloseTo(0.1, 5); // L10 splat 1

    // Check cholesky23: [L11, L20] per splat
    expect(cholesky23[0]).toBeCloseTo(2.0, 5); // L11 splat 0
    expect(cholesky23[1]).toBeCloseTo(0.3, 5); // L20 splat 0
    expect(cholesky23[2]).toBeCloseTo(5.0, 5); // L11 splat 1
    expect(cholesky23[3]).toBeCloseTo(0.2, 5); // L20 splat 1

    // Check cholesky45: [L21, L22] per splat
    expect(cholesky45[0]).toBeCloseTo(0.4, 5); // L21 splat 0
    expect(cholesky45[1]).toBeCloseTo(3.0, 5); // L22 splat 0
    expect(cholesky45[2]).toBeCloseTo(0.6, 5); // L21 splat 1
    expect(cholesky45[3]).toBeCloseTo(6.0, 5); // L22 splat 1
  });
});

describe('createInstancedGSplatsMesh', () => {
  it('should create mesh with instanced geometry', () => {
    const material = new GSplatMaterial();
    const config = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1]), // 2 splats
      cholesky01: new Float32Array([1, 0, 1, 0]),
      cholesky23: new Float32Array([1, 0, 1, 0]),
      cholesky45: new Float32Array([0, 1, 0, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]), // Red and green
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(config, material);

    expect(mesh).toBeInstanceOf(THREE.Mesh);
    expect(mesh.material).toBe(material);
    expect(mesh.frustumCulled).toBe(true);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(geometry.instanceCount).toBe(2);

    // Check instanced attributes exist
    expect(geometry.getAttribute('aCenter')).toBeDefined();
    expect(geometry.getAttribute('aCholesky01')).toBeDefined();
    expect(geometry.getAttribute('aCholesky23')).toBeDefined();
    expect(geometry.getAttribute('aCholesky45')).toBeDefined();
    expect(geometry.getAttribute('aAmplitude')).toBeDefined();
    expect(geometry.getAttribute('aColor')).toBeDefined();
  });

  it('should compute bounding box from centers', () => {
    const material = new GSplatMaterial();
    const config = {
      centers: new Float32Array([0, 0, 0, 10, 20, 30]), // 2 splats
      cholesky01: new Float32Array([1, 0, 1, 0]),
      cholesky23: new Float32Array([1, 0, 1, 0]),
      cholesky45: new Float32Array([0, 1, 0, 1]),
      amplitudes: new Float32Array([1.0, 1.0]),
      colors: new Float32Array([1, 1, 1, 1, 1, 1]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(config, material);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

    expect(geometry.boundingBox).not.toBeNull();
    // Bounding box is expanded by maxRowNorm * truncationRadius (default 3.0)
    // Cholesky factors give maxRowNorm=1, so expansion=3.0
    const expansion = 3.0;
    expect(geometry.boundingBox!.min.x).toBe(0 - expansion);
    expect(geometry.boundingBox!.min.y).toBe(0 - expansion);
    expect(geometry.boundingBox!.min.z).toBe(0 - expansion);
    expect(geometry.boundingBox!.max.x).toBe(10 + expansion);
    expect(geometry.boundingBox!.max.y).toBe(20 + expansion);
    expect(geometry.boundingBox!.max.z).toBe(30 + expansion);
  });
});

describe('updateInstancedGSplatsMesh', () => {
  it('should update mesh attributes in place when count unchanged', () => {
    const material = new GSplatMaterial();
    const initialConfig = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1]),
      cholesky01: new Float32Array([1, 0, 1, 0]),
      cholesky23: new Float32Array([1, 0, 1, 0]),
      cholesky45: new Float32Array([0, 1, 0, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(initialConfig, material);

    // Update with new data, same count
    const updateConfig = {
      centers: new Float32Array([2, 2, 2, 3, 3, 3]),
      cholesky01: new Float32Array([2, 0, 2, 0]),
      cholesky23: new Float32Array([2, 0, 2, 0]),
      cholesky45: new Float32Array([0, 2, 0, 2]),
      amplitudes: new Float32Array([0.8, 0.3]),
      colors: new Float32Array([0, 0, 1, 1, 1, 0]),
      splatCount: 2,
    };

    const rebuilt = updateInstancedGSplatsMesh(mesh, updateConfig);
    // In-place write → no buffer rebuild → the commit layer must NOT
    // invalidate the cached RenderObject.
    expect(rebuilt).toBe(false);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const centerAttr = geometry.getAttribute('aCenter');
    const centerArray = centerAttr.array as Float32Array;

    expect(centerArray[0]).toBe(2);
    expect(centerArray[1]).toBe(2);
    expect(centerArray[2]).toBe(2);
  });

  it('should recreate attributes when count changes', () => {
    const material = new GSplatMaterial();
    const initialConfig = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1]),
      cholesky01: new Float32Array([1, 0, 1, 0]),
      cholesky23: new Float32Array([1, 0, 1, 0]),
      cholesky45: new Float32Array([0, 1, 0, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(initialConfig, material);

    // Update with different count
    const updateConfig = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      cholesky01: new Float32Array([1, 0, 1, 0, 1, 0]),
      cholesky23: new Float32Array([1, 0, 1, 0, 1, 0]),
      cholesky45: new Float32Array([0, 1, 0, 1, 0, 1]),
      amplitudes: new Float32Array([1.0, 0.5, 0.3]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      splatCount: 3,
    };

    const rebuilt = updateInstancedGSplatsMesh(mesh, updateConfig);
    // Count change rebinds a fresh interleaved buffer → the commit layer
    // must invalidate the cached RenderObject (WebGPU stale vertexBuffers).
    expect(rebuilt).toBe(true);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(geometry.instanceCount).toBe(3);
  });
});
