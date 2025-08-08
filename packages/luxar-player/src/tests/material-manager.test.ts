import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MaterialManager, BlendingMode } from '../rendering/material-manager';
import * as THREE from 'three';
import { SHADER_CONFIG } from '../rendering/shader-manager';

// Mock Three.js
vi.mock('three', () => ({
  ShaderMaterial: vi.fn().mockImplementation((params) => ({
    ...params,
    userData: {},
    dispose: vi.fn(),
  })),
  NormalBlending: 'NormalBlending',
  AdditiveBlending: 'AdditiveBlending',
  SubtractiveBlending: 'SubtractiveBlending',
  HalfFloatType: 'HalfFloatType',
  LinearSRGBColorSpace: 'LinearSRGBColorSpace',
  NoToneMapping: 'NoToneMapping',
  SRGBColorSpace: 'SRGBColorSpace',
  ACESFilmicToneMapping: 'ACESFilmicToneMapping',
}));

// Mock shader manager
vi.mock('../rendering/shader-manager', () => ({
  SHADER_CONFIG: {
    POINTS: {
      size: 1.0,
      baseAlpha: 0.9,
      hdrMultiplier: 2.0,
    },
  },
}));

describe('MaterialManager', () => {
  let materialManager: MaterialManager;

  beforeEach(() => {
    vi.clearAllMocks();
    materialManager = new MaterialManager();
  });

  describe('getMaterial', () => {
    it('should create a new material with correct properties', () => {
      const props = {
        blendingMode: 'normal' as BlendingMode,
        opacity: 0.8,
        gamma: 1.5,
      };

      const material = materialManager.getMaterial(props);

      expect(THREE.ShaderMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          uniforms: expect.objectContaining({
            hdrMultiplier: { value: SHADER_CONFIG.POINTS.hdrMultiplier },
            opacity: { value: props.opacity },
            gamma: { value: props.gamma },
          }),
          vertexColors: true,
          transparent: true,
          depthWrite: false, // opacity < 0.99
          toneMapped: false,
          blending: 'NormalBlending',
        })
      );

      expect(material.userData.renderOrder).toBe(100); // transparent normal blending
    });

    it('should enable depth write for opaque normal blending', () => {
      const props = {
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      };

      materialManager.getMaterial(props);

      expect(THREE.ShaderMaterial).toHaveBeenCalledWith(
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

      const material1 = materialManager.getMaterial(props);
      const material2 = materialManager.getMaterial(props);

      // Should only create one material
      expect(THREE.ShaderMaterial).toHaveBeenCalledTimes(1);
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

      const material1 = materialManager.getMaterial(props1);
      const material2 = materialManager.getMaterial(props2);

      expect(THREE.ShaderMaterial).toHaveBeenCalledTimes(2);
      expect(material1).not.toBe(material2);
    });

    it('should set correct blending modes', () => {
      const testCases: Array<{ mode: BlendingMode; expected: string }> = [
        { mode: 'normal', expected: 'NormalBlending' },
        { mode: 'additive', expected: 'AdditiveBlending' },
        { mode: 'subtractive', expected: 'SubtractiveBlending' },
        { mode: 'minimum', expected: 'SubtractiveBlending' },
        { mode: 'maximum', expected: 'AdditiveBlending' },
      ];

      testCases.forEach(({ mode, expected }) => {
        vi.clearAllMocks();
        materialManager.getMaterial({
          blendingMode: mode,
          opacity: 1.0,
          gamma: 1.0,
        });

        expect(THREE.ShaderMaterial).toHaveBeenCalledWith(
          expect.objectContaining({
            blending: expected,
          })
        );
      });
    });

    it('should set correct render order based on blending mode', () => {
      const testCases: Array<{
        mode: BlendingMode;
        opacity: number;
        expectedOrder: number;
      }> = [
        { mode: 'normal', opacity: 1.0, expectedOrder: 0 }, // opaque
        { mode: 'normal', opacity: 0.5, expectedOrder: 100 }, // transparent
        { mode: 'subtractive', opacity: 0.8, expectedOrder: 200 },
        { mode: 'additive', opacity: 0.7, expectedOrder: 300 },
        { mode: 'minimum', opacity: 0.9, expectedOrder: 400 },
        { mode: 'maximum', opacity: 0.6, expectedOrder: 400 },
      ];

      testCases.forEach(({ mode, opacity, expectedOrder }) => {
        const material = materialManager.getMaterial({
          blendingMode: mode,
          opacity,
          gamma: 1.0,
        });

        expect(material.userData.renderOrder).toBe(expectedOrder);
      });
    });
  });

  describe('updateHDRMultiplier', () => {
    it('should update HDR multiplier for all cached materials', () => {
      // Create a few materials
      const materials = [
        materialManager.getMaterial({
          blendingMode: 'normal' as BlendingMode,
          opacity: 1.0,
          gamma: 1.0,
        }),
        materialManager.getMaterial({
          blendingMode: 'additive' as BlendingMode,
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
        materialManager.getMaterial({
          blendingMode: 'normal' as BlendingMode,
          opacity: 1.0,
          gamma: 1.0,
        }),
        materialManager.getMaterial({
          blendingMode: 'additive' as BlendingMode,
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
      materialManager.getMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      // Should create a new material since cache was cleared
      expect(THREE.ShaderMaterial).toHaveBeenCalledTimes(1);
    });
  });

  describe('getCacheStats', () => {
    it('should return correct cache statistics', () => {
      // Initially empty
      let stats = materialManager.getCacheStats();
      expect(stats.size).toBe(0);
      expect(stats.keys).toEqual([]);

      // Add some materials
      materialManager.getMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      materialManager.getMaterial({
        blendingMode: 'additive' as BlendingMode,
        opacity: 0.5,
        gamma: 1.2,
      });

      stats = materialManager.getCacheStats();
      expect(stats.size).toBe(2);
      expect(stats.keys).toHaveLength(2);
      expect(stats.keys).toContain('normal_1.00_1.00');
      expect(stats.keys).toContain('additive_0.50_1.20');
    });
  });

  describe('shader generation', () => {
    it('should include correct shader configuration in vertex shader', () => {
      materialManager.getMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      const shaderCall = (THREE.ShaderMaterial as any).mock.calls[0][0];
      const vertexShader = shaderCall.vertexShader;

      // Check that shader config values are included
      expect(vertexShader).toContain(`${SHADER_CONFIG.POINTS.size.toFixed(1)}`);
    });

    it('should include correct shader configuration in fragment shader', () => {
      materialManager.getMaterial({
        blendingMode: 'normal' as BlendingMode,
        opacity: 1.0,
        gamma: 1.0,
      });

      const shaderCall = (THREE.ShaderMaterial as any).mock.calls[0][0];
      const fragmentShader = shaderCall.fragmentShader;

      // Check that shader config values are included
      expect(fragmentShader).toContain(`${SHADER_CONFIG.POINTS.baseAlpha.toFixed(3)}`);
    });
  });
});
