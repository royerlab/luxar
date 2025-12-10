import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  LineMaterial,
  createLineQuadGeometry,
  createInstancedLinesMesh,
} from '../../../rendering/line-material';

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
    DoubleSide: 2,
  };
});

describe('LineMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new LineMaterial();

      // Default FOV is 60 degrees in radians
      const expectedFOV = (60 * Math.PI) / 180;
      expect(material.uniforms.uFOV.value).toBeCloseTo(expectedFOV);
      expect(material.uniforms.uResolution.value).toBeInstanceOf(THREE.Vector2);
      expect(material.uniforms.uOpacity.value).toBe(1.0);

      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false); // Additive blending default
      expect(material.toneMapped).toBe(false);
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.side).toBe(2); // DoubleSide
    });

    it('should accept custom configuration', () => {
      const material = new LineMaterial({
        opacity: 0.5,
        blendingMode: 'normal',
        hdrMultiplier: 20.0,
      });

      expect(material.uniforms.uOpacity.value).toBe(0.5);
      expect(material.uniforms.uHDRMultiplier.value).toBe(20.0);
      expect(material.blending).toBe('NormalBlending');
      expect(material.depthWrite).toBe(true); // Normal blending has depth write
    });

    it('should create additive material without depth write', () => {
      const material = new LineMaterial({
        blendingMode: 'additive',
      });

      expect(material.blending).toBe('AdditiveBlending');
      expect(material.depthWrite).toBe(false);
    });
  });

  describe('shaders', () => {
    it('should have correct vertex shader with screen-space expansion', () => {
      const material = new LineMaterial();

      // Check for instanced attributes
      expect(material.vertexShader).toContain('attribute vec3 aStartPos');
      expect(material.vertexShader).toContain('attribute vec3 aEndPos');
      expect(material.vertexShader).toContain('attribute vec3 aStartColor');
      expect(material.vertexShader).toContain('attribute vec3 aEndColor');
      expect(material.vertexShader).toContain('attribute float aStartWidth');
      expect(material.vertexShader).toContain('attribute float aEndWidth');
      expect(material.vertexShader).toContain('attribute float aStartSharpness');
      expect(material.vertexShader).toContain('attribute float aEndSharpness');
      expect(material.vertexShader).toContain('attribute float aSegmentLength');
      expect(material.vertexShader).toContain('attribute float aStartClipped');
      expect(material.vertexShader).toContain('attribute float aEndClipped');

      // Check for uniforms
      expect(material.vertexShader).toContain('uniform float uFOV');
      expect(material.vertexShader).toContain('uniform vec2 uResolution');

      // Check for varyings
      expect(material.vertexShader).toContain('varying vec3 vColor');
      expect(material.vertexShader).toContain('varying float vSharpness');
      expect(material.vertexShader).toContain('varying float vPerpNorm');
      expect(material.vertexShader).toContain('varying float vCapFactor');

      // Check for screen-space expansion
      expect(material.vertexShader).toContain('perpendicular');
      expect(material.vertexShader).toContain('pixelWidth');

      // Check for cap factor calculation
      expect(material.vertexShader).toContain('baseCap');
      expect(material.vertexShader).toContain('aSegmentLength');
    });

    it('should have correct fragment shader with parabolic falloff', () => {
      const material = new LineMaterial();

      // Check for uniforms
      expect(material.fragmentShader).toContain('uniform float uHDRMultiplier');
      expect(material.fragmentShader).toContain('uniform float uOpacity');

      // Check for parabolic falloff formula
      expect(material.fragmentShader).toContain('1.0 - p * p');
      expect(material.fragmentShader).toContain('vSharpness');

      // Check for cap factor application
      expect(material.fragmentShader).toContain('vCapFactor');

      // Check for discard outside line width
      expect(material.fragmentShader).toContain('discard');
      expect(material.fragmentShader).toContain('p >= 1.0');
    });
  });

  describe('methods', () => {
    it('should update camera parameters', () => {
      const material = new LineMaterial();
      const fov = (45 * Math.PI) / 180;
      const resolution = new THREE.Vector2(1920, 1080);

      material.updateCameraParams(fov, resolution);

      expect(material.uniforms.uFOV.value).toBe(fov);
      expect(material.uniforms.uResolution.value.x).toBe(1920);
      expect(material.uniforms.uResolution.value.y).toBe(1080);
    });

    it('should update HDR multiplier', () => {
      const material = new LineMaterial();

      material.updateHDRMultiplier(32.0);

      expect(material.uniforms.uHDRMultiplier.value).toBe(32.0);
    });

    it('should update opacity', () => {
      const material = new LineMaterial();

      material.updateOpacity(0.75);

      expect(material.uniforms.uOpacity.value).toBe(0.75);
    });

    it('should clone material with current values', () => {
      const original = new LineMaterial({
        opacity: 0.5,
        hdrMultiplier: 24.0,
      });

      const cloned = original.clone();

      expect(cloned.uniforms.uOpacity.value).toBe(0.5);
      expect(cloned.uniforms.uHDRMultiplier.value).toBe(24.0);

      // Ensure it's a new instance
      expect(cloned).not.toBe(original);
    });
  });

  describe('shader correctness', () => {
    it('should use semicircle kernel model for joints', () => {
      const material = new LineMaterial();

      // Cap factor at endpoints should be 0.5 for seamless joints
      expect(material.vertexShader).toContain('0.5 + 0.5');
      expect(material.vertexShader).toContain('distToNearest');
    });

    it('should handle clipped endpoints correctly', () => {
      const material = new LineMaterial();

      // Clipped endpoints should use full intensity (1.0)
      expect(material.vertexShader).toContain('nearestClipped');
      expect(material.vertexShader).toContain('mix(baseCap, 1.0, nearestClipped)');
    });

    it('should use world-space to pixel conversion', () => {
      const material = new LineMaterial();

      // Check for perspective-correct pixel width calculation
      expect(material.vertexShader).toContain('tanHalfFov');
      expect(material.vertexShader).toContain('uResolution.y');
    });
  });
});

describe('createLineQuadGeometry', () => {
  it('should create geometry with quad corners', () => {
    const geometry = createLineQuadGeometry();

    const quadCorner = geometry.getAttribute('aQuadCorner');
    expect(quadCorner).toBeDefined();
    expect(quadCorner.count).toBe(4); // 4 vertices per quad
    expect(quadCorner.itemSize).toBe(2);

    // Check corner values
    const array = quadCorner.array as Float32Array;
    expect(array[0]).toBe(-1); // First vertex x
    expect(array[1]).toBe(-1); // First vertex y
    expect(array[6]).toBe(1); // Last vertex x
    expect(array[7]).toBe(1); // Last vertex y
  });

  it('should have correct index buffer', () => {
    const geometry = createLineQuadGeometry();

    const index = geometry.index;
    expect(index).toBeDefined();
    expect(index!.count).toBe(6); // 2 triangles * 3 indices
  });
});

describe('createInstancedLinesMesh', () => {
  it('should create instanced mesh with correct attributes', () => {
    const config = {
      startPositions: new Float32Array([0, 0, 0, 1, 1, 1]),
      endPositions: new Float32Array([1, 0, 0, 2, 1, 1]),
      startColors: new Float32Array([1, 0, 0, 0, 1, 0]),
      endColors: new Float32Array([1, 0, 0, 0, 1, 0]),
      startWidths: new Float32Array([0.1, 0.1]),
      endWidths: new Float32Array([0.1, 0.1]),
      startSharpness: new Float32Array([1.0, 1.0]),
      endSharpness: new Float32Array([1.0, 1.0]),
      segmentLengths: new Float32Array([1.0, 1.414]),
      startClipped: new Uint8Array([0, 0]),
      endClipped: new Uint8Array([0, 0]),
      segmentCount: 2,
    };

    const material = new LineMaterial();
    const mesh = createInstancedLinesMesh(config, material);

    // Lines use THREE.Mesh with InstancedBufferGeometry (not InstancedMesh)
    // to avoid exceeding WebGL's 16 attribute location limit
    expect(mesh).toBeInstanceOf(THREE.Mesh);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(geometry.instanceCount).toBe(2);
    expect(geometry.getAttribute('aStartPos')).toBeDefined();
    expect(geometry.getAttribute('aEndPos')).toBeDefined();
    expect(geometry.getAttribute('aStartColor')).toBeDefined();
    expect(geometry.getAttribute('aEndColor')).toBeDefined();
    expect(geometry.getAttribute('aStartWidth')).toBeDefined();
    expect(geometry.getAttribute('aEndWidth')).toBeDefined();
    expect(geometry.getAttribute('aStartSharpness')).toBeDefined();
    expect(geometry.getAttribute('aEndSharpness')).toBeDefined();
    expect(geometry.getAttribute('aSegmentLength')).toBeDefined();
    expect(geometry.getAttribute('aStartClipped')).toBeDefined();
    expect(geometry.getAttribute('aEndClipped')).toBeDefined();
  });

  it('should compute bounding box and sphere', () => {
    const config = {
      startPositions: new Float32Array([0, 0, 0]),
      endPositions: new Float32Array([10, 10, 10]),
      startColors: new Float32Array([1, 0, 0]),
      endColors: new Float32Array([0, 1, 0]),
      startWidths: new Float32Array([0.1]),
      endWidths: new Float32Array([0.1]),
      startSharpness: new Float32Array([1.0]),
      endSharpness: new Float32Array([1.0]),
      segmentLengths: new Float32Array([17.32]),
      startClipped: new Uint8Array([0]),
      endClipped: new Uint8Array([0]),
      segmentCount: 1,
    };

    const material = new LineMaterial();
    const mesh = createInstancedLinesMesh(config, material);

    const geometry = mesh.geometry;
    expect(geometry.boundingBox).toBeDefined();
    expect(geometry.boundingSphere).toBeDefined();

    // Bounding box should encompass both endpoints
    expect(geometry.boundingBox!.min.x).toBe(0);
    expect(geometry.boundingBox!.max.x).toBe(10);
  });
});
