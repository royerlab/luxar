import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { GSplatMaterial } from '../../../../../rendering/materials/gsplat/material-glsl';
import {
  clampTruncationRadius,
  MIN_TRUNCATION_RADIUS,
} from '../../../../../rendering/materials/gsplat/math';
import {
  createGSplatQuadGeometry,
  createInstancedGSplatsMesh,
  getSplatTexture,
  updateInstancedGSplatsMesh,
} from '../../../../../rendering/gsplat-geometry';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../../../config/constants';

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

describe('clampTruncationRadius guard', () => {
  it('falls back to the module default for radii that are not finite in float32', () => {
    // NaN slips past a plain comparison clamp (NaN < min is false) and
    // would make uShiftC/uInvOneMinusC NaN — the exact degenerate-uniform
    // failure the clamp exists to prevent (truncation_radius arrives
    // unvalidated from dataset attrs).
    expect(clampTruncationRadius(Number.NaN)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    expect(clampTruncationRadius(Number.POSITIVE_INFINITY)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    // Finite in float64, Infinity once narrowed to the float32 GPU uniform —
    // the hostile-attr mirror image of the tiny-radius degeneracy. The
    // write-side validator rejects these; the read-side clamp is the layer
    // that sees unvalidated stores, so it must catch them too.
    expect(clampTruncationRadius(3.5e38)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    expect(clampTruncationRadius(1e308)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    // Finite even as a float32, but its SQUARE — uploaded as the uTruncateSq
    // uniform (the fragment discard threshold) — narrows to Infinity. The
    // usable ceiling is sqrt(float32.max) ≈ 1.84e19, matching the write-side
    // MAX_TRUNCATION_RADIUS_FLOAT32.
    expect(clampTruncationRadius(1e30)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    expect(clampTruncationRadius(2e19)).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    expect(clampTruncationRadius(1e19)).toBe(1e19);
    expect(clampTruncationRadius(0)).toBe(MIN_TRUNCATION_RADIUS);
    expect(clampTruncationRadius(2.5)).toBe(2.5);
  });

  it('treats a NON-NUMBER exactly like a NaN, so a JSON string cannot reach the uniform', () => {
    // `truncation_radius` is untrusted zarr JSON. A store stamping `"6"` satisfies
    // `GSplatsMetadata` only nominally, and every numeric test in this function
    // COERCES it — `Math.fround("6" * "6")` is 36 (finite) and `"6" < MIN` is false —
    // so before the type guard the string was returned unchanged and uploaded as
    // `uTruncate`, giving a 6σ material band while the fetch tolerance
    // (`gsplats-spatial-index-loader.ts::resolveTruncationRadius`, which treats a
    // non-number as absent) covered only 2.75σ. One rule, both call sites.
    for (const hostile of ['6', '2.75', 'nonsense', null, undefined, {}, [], true]) {
      expect(clampTruncationRadius(hostile as unknown as number)).toBe(
        GSPLAT_DEFAULT_TRUNCATION_RADIUS
      );
    }
  });

  it('clamps at the float32 degeneracy bound, agreeing with the Python writer', () => {
    // The write-side validator (MIN_TRUNCATION_RADIUS_FLOAT32 in
    // luxar/validation/types.py) accepts anything that normalizes in
    // float32, and on-disk chunk bounds are computed from the stored
    // radius — so the read-side clamp must not silently rewrite those
    // values (a 0.05 store must render at 0.05, matching its bounds).
    expect(clampTruncationRadius(0.05)).toBe(0.05);
    expect(clampTruncationRadius(0.01)).toBe(0.01);
    // The bound sits where exp(-T²/2) rounds to 1.0 in float32 (~2.44e-4):
    // well below float64's ~1.5e-8 threshold, well above zero.
    expect(MIN_TRUNCATION_RADIUS).toBeGreaterThan(1e-4);
    expect(MIN_TRUNCATION_RADIUS).toBeLessThan(1e-3);
    expect(Math.fround(Math.exp(-0.5 * MIN_TRUNCATION_RADIUS ** 2))).toBeLessThan(1.0);
    expect(clampTruncationRadius(1e-5)).toBe(MIN_TRUNCATION_RADIUS);
  });
});

describe('GSplatMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new GSplatMaterial();

      expect(material.uniforms.uOpacity.value).toBe(1.0);
      expect(material.uniforms.uTruncate.value).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
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

    it('updates and clone-preserves categorical label styling', () => {
      const material = new GSplatMaterial();
      material.updateLabelStyle(true, 3.8);

      expect(material.uniforms.uLabelColorMode.value).toBe(1);
      expect(material.uniforms.uLabelFilterIndex.value).toBe(3);
      expect(material.vertexShader).toContain('categoricalColor(aLabelIndex)');

      const cloned = material.clone();
      expect(cloned.uniforms.uLabelColorMode.value).toBe(1);
      expect(cloned.uniforms.uLabelFilterIndex.value).toBe(3);
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
      // Peak (2D-projected) projection: alpha-over is the surface model, so a
      // splat's contribution is its projected-Gaussian peak, not the emissive
      // ray-integral (which would saturate coverage-alpha to opaque + streak).
      expect(material.uniforms.uProjectionMode.value).toBe(1); // Peak projection for normal
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

      // Splat data lives in the RGBA32F splat texture; the only
      // per-instance attribute is the draw-slot -> storage-slot map.
      expect(material.vertexShader).toContain('in uint aSortedIndex');
      expect(material.vertexShader).toContain('uniform highp sampler2D uSplatTex');
      // texelFetch prologue reconstructs the per-splat locals.
      expect(material.vertexShader).toContain('texelFetch(uSplatTex');
      expect(material.vertexShader).toContain('vec3 aCenter = splatT0.xyz');
      expect(material.vertexShader).toContain('float aAmplitude = splatT0.w');
      expect(material.vertexShader).toContain('vec3 aColor = vec3(splatT2.zw, splatT3.x)');

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

      // Check for unpackCholesky3D function (takes the texel-fetched
      // values as parameters since the texture-storage migration)
      expect(material.vertexShader).toContain(
        'mat3 unpackCholesky3D(vec2 c01, vec2 c23, vec2 c45)'
      );
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
      // String-pin the trace-normalized inversion's RESTORE factor: the
      // ray sigma of the normalized covariance must be rescaled by
      // sqrt(sTrace) (sigma_ray = sqrt(s/quadN)) — a wrong or dropped
      // exponent here is invisible to the TS-mirror unit test and to the
      // peak-mode codegen snapshot.
      expect(material.vertexShader).toContain('inversesqrt(quad) * sqrt(sTrace)');
    });

    it('should have near-plane guard with smooth fade and screen-coverage cull', () => {
      const material = new GSplatMaterial();

      // Near-cull uses the shared perspectiveNearFade helper (smooth
      // fade, not hard discard) with the uNearCull uniform
      // 1e-20 floor = degenerate-smoothstep guard only (scene-relative
      // uNearCull is never overridden on tiny-unit scenes).
      expect(material.vertexShader).toContain(
        'perspectiveNearFade(uIsOrtho, centerCam.z, max(uNearCull, 1e-20))'
      );
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

    it('should discard negligible contributions (gain-aware gate)', () => {
      const material = new GSplatMaterial();

      // Higher threshold (1e-4) for performance, scaled by the layer gain
      // so high-gain dim splats are not gated out before uIntensity
      // applies; max(uIntensity, 1.0) keeps gain <= 1 at the historical
      // threshold exactly.
      expect(material.fragmentShader).toContain(
        'if (intensity * max(uIntensity, 1.0) < 1e-4) discard'
      );
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

    it('defaults and clone-preserves uCov2DDilation (2D low-pass, default 0.3)', () => {
      const material = new GSplatMaterial({});
      expect(material.uniforms.uCov2DDilation.value).toBe(0.3);

      const tuned = new GSplatMaterial({ cov2DDilation: 0.5 });
      expect(tuned.uniforms.uCov2DDilation.value).toBe(0.5);
      expect(tuned.clone().uniforms.uCov2DDilation.value).toBe(0.5);
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
      expect(material.blending).toBe('CustomBlending');
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

      expect(material.blending).toBe('CustomBlending');
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
      expect(material.uniforms.uProjectionMode.value).toBe(1); // peak projection (surface)
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

describe('6-stride cholesky texel layout (packCholeskyForShader retirement)', () => {
  it('writes texel floats identical to the retired split→re-interleave path', () => {
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

    const material = new GSplatMaterial();
    const mesh = createInstancedGSplatsMesh(
      {
        centers: new Float32Array([0, 0, 0, 1, 1, 1]),
        choleskyFactors,
        amplitudes: new Float32Array([1.0, 0.5]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0]),
        splatCount: 2,
      },
      material
    );
    const texels = getSplatTexture(mesh.geometry)!.image.data as Float32Array;

    // Texel 1 = [cholesky01.xy, cholesky23.xy], texel 2 starts with
    // cholesky45.xy — exactly what the retired packCholeskyForShader
    // split (pairs [L00,L10], [L11,L20], [L21,L22]) re-interleaved to.
    for (let i = 0; i < 2; i++) {
      const o = i * 16;
      const c6 = i * 6;
      expect(texels[o + 4]).toBe(choleskyFactors[c6]); // L00
      expect(texels[o + 5]).toBe(choleskyFactors[c6 + 1]); // L10
      expect(texels[o + 6]).toBe(choleskyFactors[c6 + 2]); // L11
      expect(texels[o + 7]).toBe(choleskyFactors[c6 + 3]); // L20
      expect(texels[o + 8]).toBe(choleskyFactors[c6 + 4]); // L21
      expect(texels[o + 9]).toBe(choleskyFactors[c6 + 5]); // L22
    }
  });
});

describe('createInstancedGSplatsMesh', () => {
  it('should create mesh with instanced geometry', () => {
    const material = new GSplatMaterial();
    const config = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1]), // 2 splats
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
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

    // Texture-backed storage: the geometry carries the splat texture +
    // the identity ordering attribute (no per-splat data attributes).
    const texture = getSplatTexture(geometry);
    expect(texture).not.toBeNull();
    const sortedIndex = geometry.getAttribute('aSortedIndex');
    expect(sortedIndex).toBeDefined();
    expect((sortedIndex.array as Uint32Array)[0]).toBe(0);
    expect((sortedIndex.array as Uint32Array)[1]).toBe(1);
    // Texel layout round-trip for splat 1: center/amplitude in texel 0,
    // color split across texels 2-3.
    const texels = texture!.image.data as Float32Array;
    expect(texels[16 + 0]).toBe(1); // center.x of splat 1
    expect(texels[16 + 3]).toBe(0.5); // amplitude of splat 1
    expect(texels[16 + 11]).toBe(1); // color.g of splat 1
    // The material was bound to the mesh-owned texture at creation.
    expect(material.uniforms.uSplatTex.value).toBe(texture);
  });

  it('should compute bounding box from centers', () => {
    const material = new GSplatMaterial();
    const config = {
      centers: new Float32Array([0, 0, 0, 10, 20, 30]), // 2 splats
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      amplitudes: new Float32Array([1.0, 1.0]),
      colors: new Float32Array([1, 1, 1, 1, 1, 1]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(config, material);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

    expect(geometry.boundingBox).not.toBeNull();
    // Bounding box is expanded by maxRowNorm * truncationRadius (the default).
    // Cholesky factors give maxRowNorm=1, so expansion is the default itself.
    const expansion = GSPLAT_DEFAULT_TRUNCATION_RADIUS;
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
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(initialConfig, material);

    // Update with new data, same count
    const updateConfig = {
      centers: new Float32Array([2, 2, 2, 3, 3, 3]),
      choleskyFactors: new Float32Array([2, 0, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2]),
      amplitudes: new Float32Array([0.8, 0.3]),
      colors: new Float32Array([0, 0, 1, 1, 1, 0]),
      splatCount: 2,
    };

    const rebuilt = updateInstancedGSplatsMesh(mesh, updateConfig);
    // In-place write → no buffer rebuild → the commit layer must NOT
    // invalidate the cached RenderObject.
    expect(rebuilt).toBe(false);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const texels = getSplatTexture(geometry)!.image.data as Float32Array;

    expect(texels[0]).toBe(2); // center.x of splat 0
    expect(texels[1]).toBe(2);
    expect(texels[2]).toBe(2);
  });

  it('should recreate attributes when count changes', () => {
    const material = new GSplatMaterial();
    const initialConfig = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
      amplitudes: new Float32Array([1.0, 0.5]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(initialConfig, material);

    // Update with different count
    const updateConfig = {
      centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
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
