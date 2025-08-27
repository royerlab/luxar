import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  MaterialManager,
  BlendingMode,
  PointMaterialProperties,
} from '../rendering/material-manager';
import { PointMaterial } from '../rendering/point-material';

// Mock Three.js
vi.mock('three', () => ({
  ShaderMaterial: vi.fn().mockImplementation((params) => ({
    ...params,
    userData: {},
    dispose: vi.fn(),
  })),
  Vector2: vi.fn().mockImplementation((x, y) => {
    const vec = {
      x: x || 0,
      y: y || 0,
      clone: vi.fn(),
      copy: vi.fn(),
    };
    vec.clone.mockImplementation(() => ({
      x: vec.x,
      y: vec.y,
      clone: vi.fn(),
      copy: vi.fn(),
    }));
    vec.copy.mockImplementation((v: any) => {
      vec.x = v.x;
      vec.y = v.y;
      return vec;
    });
    return vec;
  }),
  NormalBlending: 'NormalBlending',
  AdditiveBlending: 'AdditiveBlending',
  SubtractiveBlending: 'SubtractiveBlending',
  HalfFloatType: 'HalfFloatType',
  LinearSRGBColorSpace: 'LinearSRGBColorSpace',
  NoToneMapping: 'NoToneMapping',
  SRGBColorSpace: 'SRGBColorSpace',
  ACESFilmicToneMapping: 'ACESFilmicToneMapping',
}));

// Mock PointMaterial
vi.mock('../rendering/point-material', () => {
  class MockPointMaterial {
    uniforms: any;
    vertexShader: string;
    fragmentShader: string;
    vertexColors: boolean;
    transparent: boolean;
    depthWrite: boolean;
    toneMapped: boolean;
    blending: string;
    userData: any;
    updateCameraParams: any;
    updateHDRMultiplier: any;
    updateOpacity: any;
    updateGamma: any;
    dispose: any;

    constructor(config: any) {
      this.uniforms = {
        hdrMultiplier: { value: 16.0 },
        baseAlpha: { value: 0.01 },
        opacity: { value: config?.opacity || 1.0 },
        gamma: { value: config?.gamma || 1.0 },
        fov: { value: (60 * Math.PI) / 180 },
        resolution: { value: { x: 1, y: 1, copy: vi.fn() } },
      };
      this.vertexShader = `
        attribute float radius;
        attribute float sharpness;
        uniform float fov;
        uniform vec2 resolution;
        
        void main() {
          float pointSize = 2.0 * radius * resolution.y / (distance * tan(fov * 0.5));
        }
      `;
      this.fragmentShader = `
        uniform float hdrMultiplier;
        uniform float baseAlpha;
        void main() {
          vec3 hdrColor = vColor * hdrMultiplier;
        }
      `;
      this.vertexColors = true;
      this.transparent = true;
      this.depthWrite = config?.depthWrite || false;
      this.toneMapped = false;
      this.blending = config?.blending || 'AdditiveBlending';
      this.userData = {};
      this.updateCameraParams = vi.fn();
      this.updateHDRMultiplier = vi.fn((value: number) => {
        this.uniforms.hdrMultiplier.value = value;
      });
      this.updateOpacity = vi.fn();
      this.updateGamma = vi.fn();
      this.dispose = vi.fn();
    }
  }

  // Make it constructable with `new` but also spy-able
  const PointMaterialSpy = vi.fn(function (this: any, config: any) {
    return new MockPointMaterial(config);
  }) as any;

  // Copy prototype so instanceof checks work
  PointMaterialSpy.prototype = MockPointMaterial.prototype;

  return { PointMaterial: PointMaterialSpy };
});

describe('MaterialManager', () => {
  let materialManager: MaterialManager;

  beforeEach(() => {
    vi.clearAllMocks();
    materialManager = new MaterialManager();
  });

  describe('getPointMaterial', () => {
    it('should create a new material with correct properties', () => {
      const props: PointMaterialProperties = {
        blendingMode: 'normal',
        opacity: 0.8,
        gamma: 1.5,
      };

      materialManager.getPointMaterial(props);

      expect(PointMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          opacity: props.opacity,
          gamma: props.gamma,
          blending: 'NormalBlending',
          depthWrite: false, // opacity < 0.99
        })
      );
    });

    it('should enable depth write for opaque normal blending', () => {
      const props = {
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      };

      materialManager.getPointMaterial(props);

      expect(PointMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          depthWrite: true, // opacity >= 0.99
          blending: 'NormalBlending',
        })
      );
    });

    it('should cache materials with the same properties', () => {
      const props = {
        blendingMode: 'additive' as BlendingMode,
        opacity: 0.5,
        gamma: 1.2,
      };

      const material1 = materialManager.getPointMaterial(props);
      const material2 = materialManager.getPointMaterial(props);

      // Should only create one material
      expect(PointMaterial).toHaveBeenCalledTimes(1);
      expect(material1).toBe(material2);
    });

    it('should create different materials for different properties', () => {
      const props1 = {
        blendingMode: 'additive' as BlendingMode,
        opacity: 0.5,
        gamma: 1.0,
      };

      const props2 = {
        blendingMode: 'additive' as BlendingMode,
        opacity: 0.6, // Different opacity
        gamma: 1.0,
      };

      const material1 = materialManager.getPointMaterial(props1);
      const material2 = materialManager.getPointMaterial(props2);

      expect(PointMaterial).toHaveBeenCalledTimes(2);
      expect(material1).not.toBe(material2);
    });

    it('should set correct blending modes', () => {
      const testCases: Array<{ mode: BlendingMode; expected: string }> = [
        { mode: 'normal', expected: 'NormalBlending' },
        { mode: 'additive', expected: 'AdditiveBlending' },
      ];

      testCases.forEach(({ mode, expected }) => {
        vi.clearAllMocks();
        materialManager.getPointMaterial({
          blendingMode: mode,
          opacity: 1.0,
          gamma: 1.0,
        });

        expect(PointMaterial).toHaveBeenCalledWith(
          expect.objectContaining({
            blending: expected,
          })
        );
      });
    });
  });

  describe('updateHDRMultiplier', () => {
    it('should update HDR multiplier for all registered materials', () => {
      // Create a few materials
      const materials = [
        materialManager.getPointMaterial({
          blendingMode: 'normal',
          opacity: 1.0,
          gamma: 1.0,
        }),
        materialManager.getPointMaterial({
          blendingMode: 'additive',
          opacity: 0.5,
          gamma: 1.2,
        }),
      ];

      const newMultiplier = 5.0;
      materialManager.updateHDRMultiplier(newMultiplier);

      // Check that all materials were updated
      materials.forEach((material) => {
        expect(material.uniforms.hdrMultiplier.value).toBe(newMultiplier);
      });
    });
  });

  describe('dispose', () => {
    it('should dispose all cached materials', () => {
      // Create a few materials
      const materials = [
        materialManager.getPointMaterial({
          blendingMode: 'normal',
          opacity: 1.0,
          gamma: 1.0,
        }),
        materialManager.getPointMaterial({
          blendingMode: 'additive',
          opacity: 0.5,
          gamma: 1.2,
        }),
      ];

      materialManager.dispose();

      // Check that all materials were disposed
      materials.forEach((material) => {
        expect(material.dispose).toHaveBeenCalled();
      });

      // Check that cache is cleared by trying to get the same material again
      vi.clearAllMocks();
      materialManager.getPointMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      // Should create a new material since cache was cleared
      expect(PointMaterial).toHaveBeenCalledTimes(1);
    });
  });

  describe('getCacheStats', () => {
    it('should return correct cache statistics', () => {
      // Initially empty
      let stats = materialManager.getCacheStats();
      expect(stats.pointMaterials).toBe(0);
      expect(stats.keys).toEqual([]);

      // Add some materials
      materialManager.getPointMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      materialManager.getPointMaterial({
        blendingMode: 'additive' as BlendingMode,
        opacity: 0.5,
        gamma: 1.2,
      });

      stats = materialManager.getCacheStats();
      expect(stats.pointMaterials).toBe(2);
      expect(stats.keys).toHaveLength(2);
      expect(stats.keys).toContain('point_normal_1.00_1.00');
      expect(stats.keys).toContain('point_additive_0.50_1.20');
    });
  });

  describe('shader generation', () => {
    it('should create PointMaterial with correct shaders', () => {
      const material = materialManager.getPointMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      // Check that material has vertex shader with proper world-space sizing
      expect(material.vertexShader).toBeDefined();
      expect(material.vertexShader).toContain('2.0 * radius * resolution.y');

      // Check that material has fragment shader with HDR and alpha handling
      expect(material.fragmentShader).toBeDefined();
      expect(material.fragmentShader).toContain('hdrMultiplier');
      expect(material.fragmentShader).toContain('baseAlpha');
    });
  });
});
