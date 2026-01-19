import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  GSplatMaterial,
  createGSplatQuadGeometry,
  createInstancedGSplatsMesh,
  packCholeskyForShader,
  updateInstancedGSplatsMesh,
} from '../../../rendering/gsplat-material';
import { config } from '../../../config';

// Mock THREE.ShaderMaterial
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  const ShaderMaterial = vi.fn(function (this: any, params: any) {
    Object.assign(this, {
      uniforms: params.uniforms,
      vertexShader: params.vertexShader,
      fragmentShader: params.fragmentShader,
      transparent: params.transparent,
      depthWrite: params.depthWrite,
      toneMapped: params.toneMapped,
      blending: params.blending,
      side: params.side,
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
    MaxEquation: 'MaxEquation',
    OneFactor: 'OneFactor',
    DoubleSide: 'DoubleSide',
  };
});

describe('GSplatMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new GSplatMaterial();

      expect(material.uniforms.uHDRMultiplier.value).toBe(config.shader.points.hdrMultiplier);
      expect(material.uniforms.uOpacity.value).toBe(1.0);
      expect(material.uniforms.uTruncate.value).toBe(3.0);
      expect(material.uniforms.uResolution.value).toBeInstanceOf(THREE.Vector2);
      expect(material.uniforms.uFx.value).toBe(500);
      expect(material.uniforms.uFy.value).toBe(500);

      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false);
      expect(material.toneMapped).toBe(false);
      // Default 'additive' mode uses CustomBlending with OneFactor for correct linear sum
      // (AdditiveBlending uses SrcAlpha which would incorrectly square the intensity)
      expect(material.blending).toBe('CustomBlending');
      expect(material.side).toBe('DoubleSide');
      expect(material.uniforms.uProjectionMode.value).toBe(0); // Default additive uses sum projection
      // 'additive' ignores depth (depthTest=false)
      expect(material.userData.depthTest).toBe(false);
    });

    it('should accept custom configuration', () => {
      const material = new GSplatMaterial({
        opacity: 0.5,
        hdrMultiplier: 32.0,
        truncationRadius: 4.0,
        blendingMode: 'normal',
      });

      expect(material.uniforms.uOpacity.value).toBe(0.5);
      expect(material.uniforms.uHDRMultiplier.value).toBe(32.0);
      expect(material.uniforms.uTruncate.value).toBe(4.0);
      expect(material.blending).toBe('NormalBlending');
      // depthWrite is only true for normal blending when opacity >= 0.99
      expect(material.depthWrite).toBe(false); // opacity 0.5 < 0.99, so no depth write
      expect(material.uniforms.uProjectionMode.value).toBe(0); // Sum projection for normal
    });

    it('should configure max blending with max projection', () => {
      const material = new GSplatMaterial({
        blendingMode: 'max',
      });

      expect(material.blending).toBe('CustomBlending');
      expect(material.depthWrite).toBe(false); // Max blending disables depth write
      expect(material.uniforms.uProjectionMode.value).toBe(1); // Max projection
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
      expect(material.vertexShader).toContain('in float aSharpness');
      expect(material.vertexShader).toContain('in vec3 aColor');

      // Check for uniforms
      expect(material.vertexShader).toContain('uniform vec2 uResolution');
      expect(material.vertexShader).toContain('uniform float uFx, uFy');
      expect(material.vertexShader).toContain('uniform float uTruncate');
      expect(material.vertexShader).toContain('uniform int uProjectionMode');

      // Check for flat varyings (GLSL ES 3.0 uses "flat out" for non-interpolated values)
      expect(material.vertexShader).toContain('flat out mediump vec3 vColor');
      expect(material.vertexShader).toContain('flat out mediump float vAmplitude2D');
      expect(material.vertexShader).toContain('flat out mediump float vSharpness');
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
      expect(material.vertexShader).toContain('float L00 = sqrt');
      // OPTIMIZATION: Uses reciprocal multiplication instead of division
      expect(material.vertexShader).toContain('float invL00 = 1.0 / L00');
      expect(material.vertexShader).toContain('float L10 = S[1][0] * invL00');
      expect(material.vertexShader).toContain('float L11 = sqrt');
      expect(material.vertexShader).toContain('float invL11 = 1.0 / L11');
      // Returns reciprocals for faster fragment shader
      expect(material.vertexShader).toContain('return vec3(invL00, L10, invL11)');
    });

    it('should have sharpness integral factor function', () => {
      const material = new GSplatMaterial();

      expect(material.vertexShader).toContain('float sharpnessIntegralFactor(float s)');
      // Check for the simple approximation formula
      expect(material.vertexShader).toContain('1.97 + 1.95 * exp(-0.64 * s)');
    });

    it('should have near-plane guard', () => {
      const material = new GSplatMaterial();

      expect(material.vertexShader).toContain('if (-centerCam.z < 0.1)');
      expect(material.vertexShader).toContain('gl_Position = vec4(0.0, 0.0, -2.0, 1.0)');
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
      expect(material.fragmentShader).toContain('uniform mediump float uHDRMultiplier');

      // Check for forward substitution to solve L·y = d
      // OPTIMIZATION: Uses multiplication with precomputed reciprocals instead of division
      expect(material.fragmentShader).toContain('float y0 = d.x * vL2D.x'); // MUL with invL00
      expect(material.fragmentShader).toContain('float y1 = (d.y - vL2D.y * y0) * vL2D.z'); // MUL with invL11

      // Check for Mahalanobis distance
      expect(material.fragmentShader).toContain('float mahalSq = y0 * y0 + y1 * y1');

      // Check for early discard at 3σ
      expect(material.fragmentShader).toContain('if (mahalSq > 9.0) discard');

      // Check for generalized Gaussian falloff with projection correction
      // The correction factor handles non-separability of 3D→2D projection for s≠2
      expect(material.fragmentShader).toContain('float correctionFactor(float r, float s, float alpha)');
      expect(material.fragmentShader).toContain('float gauss_2d = exp(-0.5 * mahalSq)');
      expect(material.fragmentShader).toContain(
        'float correction = correctionFactor(r_2D, vSharpness, vAspectRatio)'
      );
      expect(material.fragmentShader).toContain('intensity = vAmplitude2D * gauss_2d * correction');

      // Check for sharpness=2.0 optimization (standard Gaussian fast path)
      expect(material.fragmentShader).toContain('if (abs(vSharpness - 2.0) < 0.001)');
      expect(material.fragmentShader).toContain('intensity = vAmplitude2D * exp(-0.5 * mahalSq)');
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

    it('should update HDR multiplier', () => {
      const material = new GSplatMaterial();

      material.updateHDRMultiplier(32.0);

      expect(material.uniforms.uHDRMultiplier.value).toBe(32.0);
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

    it('should clone material with current values', () => {
      const original = new GSplatMaterial({
        opacity: 0.5,
        truncationRadius: 4.0,
      });

      original.updateHDRMultiplier(24.0);

      const cloned = original.clone();

      expect(cloned.uniforms.uOpacity.value).toBe(0.5);
      expect(cloned.uniforms.uTruncate.value).toBe(4.0);
      expect(cloned.uniforms.uHDRMultiplier.value).toBe(24.0);

      // Ensure it's a new instance
      expect(cloned).not.toBe(original);
    });
  });

  describe('blending mode depth test configuration', () => {
    it('should have depthTest false for additive mode (ignores depth)', () => {
      const material = new GSplatMaterial({ blendingMode: 'additive' });

      // 'additive' ignores depth entirely (renders on top of everything)
      expect(material.userData.depthTest).toBe(false);
      // Uses CustomBlending with OneFactor for correct linear sum (not AdditiveBlending)
      expect(material.blending).toBe('CustomBlending');
    });

    it('should have depthTest true for luminous mode (respects depth occlusion)', () => {
      const material = new GSplatMaterial({ blendingMode: 'luminous' });

      // 'luminous' respects depth occlusion but uses same visual output as additive
      expect(material.userData.depthTest).toBe(true);
      // Uses CustomBlending with OneFactor for correct linear sum (same as additive)
      expect(material.blending).toBe('CustomBlending');
    });

    it('should have depthTest true for normal mode', () => {
      const material = new GSplatMaterial({ blendingMode: 'normal' });

      expect(material.userData.depthTest).toBe(true);
      expect(material.blending).toBe('NormalBlending');
    });

    it('should apply opacity directly to RGB for linear additive blending', () => {
      const material = new GSplatMaterial();

      // No uLuminous uniform - shader always uses same output pattern
      expect(material.fragmentShader).not.toContain('uniform bool uLuminous');
      expect(material.fragmentShader).not.toContain('if (uLuminous)');

      // With OneFactor blending, opacity is applied directly to RGB (alpha is ignored)
      // This ensures correct linear sum projection without squaring intensity
      expect(material.fragmentShader).toContain(
        'vec3 finalColor = vColor * intensity * uHDRMultiplier * uOpacity'
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
      sharpness: new Float32Array([2.0, 2.0]),
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
    expect(geometry.getAttribute('aSharpness')).toBeDefined();
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
      sharpness: new Float32Array([2.0, 2.0]),
      colors: new Float32Array([1, 1, 1, 1, 1, 1]),
      splatCount: 2,
    };

    const mesh = createInstancedGSplatsMesh(config, material);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

    expect(geometry.boundingBox).not.toBeNull();
    expect(geometry.boundingBox!.min.x).toBe(0);
    expect(geometry.boundingBox!.min.y).toBe(0);
    expect(geometry.boundingBox!.min.z).toBe(0);
    expect(geometry.boundingBox!.max.x).toBe(10);
    expect(geometry.boundingBox!.max.y).toBe(20);
    expect(geometry.boundingBox!.max.z).toBe(30);
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
      sharpness: new Float32Array([2.0, 2.0]),
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
      sharpness: new Float32Array([1.5, 2.5]),
      colors: new Float32Array([0, 0, 1, 1, 1, 0]),
      splatCount: 2,
    };

    updateInstancedGSplatsMesh(mesh, updateConfig);

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
      sharpness: new Float32Array([2.0, 2.0]),
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
      sharpness: new Float32Array([2.0, 2.0, 2.0]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      splatCount: 3,
    };

    updateInstancedGSplatsMesh(mesh, updateConfig);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(geometry.instanceCount).toBe(3);
  });
});
