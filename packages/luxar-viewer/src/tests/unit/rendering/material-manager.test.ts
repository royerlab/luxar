/**
 * Tests for MaterialManager
 *
 * This test suite verifies:
 * - Material creation and caching
 * - Camera parameter updates across all materials
 * - HDR multiplier updates
 * - Material disposal and cleanup
 *
 * IMPORTANT: We test the REAL PointMaterial class (not mocked) to ensure
 * shader code generation, uniform initialization, and material behavior are correct.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MaterialManager, type PointMaterialProperties } from '../../../rendering/material-manager';
import { PointMaterial } from '../../../rendering/point-material';
import * as THREE from 'three';

// Mock only THREE.js (dependency), NOT PointMaterial (system under test)
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  // Mock ShaderMaterial to avoid WebGL dependencies
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
      clone: vi.fn(function (this: any) {
        // Simple clone for testing
        return {
          ...this,
          uniforms: { ...this.uniforms },
        };
      }),
    });
  });

  return {
    ...actual,
    ShaderMaterial: ShaderMaterial as any,
    Vector2: actual.Vector2,
    NormalBlending: 0,
    AdditiveBlending: 1,
    SubtractiveBlending: 2,
  };
});

describe('MaterialManager', () => {
  let manager: MaterialManager;

  beforeEach(() => {
    manager = new MaterialManager();
  });

  afterEach(() => {
    manager.dispose();
    vi.clearAllMocks();
  });

  // =========================================================================
  // MATERIAL CREATION
  // =========================================================================

  describe('getPointMaterial', () => {
    it('should create real PointMaterial with actual shaders', () => {
      const props: PointMaterialProperties = {
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      };

      const material = manager.getPointMaterial(props);

      // Verify it's a real PointMaterial instance
      expect(material).toBeInstanceOf(PointMaterial);

      // Test REAL vertex shader content (THREE.js provides position, we provide custom attributes)
      expect(material.vertexShader).toContain('attribute float radius');
      expect(material.vertexShader).toContain('attribute float sharpness');
      expect(material.vertexShader).toContain('uniform float tanHalfFov');
      expect(material.vertexShader).toContain('uniform vec2 resolution');
      expect(material.vertexShader).toContain('varying vec3 vColor');

      // Test REAL fragment shader content
      expect(material.fragmentShader).toContain('uniform float hdrMultiplier');
      expect(material.fragmentShader).toContain('uniform float opacity');
      expect(material.fragmentShader).toContain('uniform float gamma');
      expect(material.fragmentShader).toContain('gl_FragColor');

      // Test REAL uniforms initialized (with optimized tanHalfFov)
      expect(material.uniforms.tanHalfFov).toBeDefined();
      expect(material.uniforms.resolution).toBeDefined();
      expect(material.uniforms.hdrMultiplier).toBeDefined();
      expect(material.uniforms.opacity).toBeDefined();
      expect(material.uniforms.gamma).toBeDefined();
    });

    it('should respect custom opacity', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 1.0,
      });

      expect(material.uniforms.opacity.value).toBe(0.5);
    });

    it('should respect custom gamma', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 2.2,
      });

      expect(material.uniforms.gamma.value).toBe(2.2);
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.2);
    });

    it('should set correct blending mode', () => {
      const additive = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const normal = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 1.0,
      });

      expect(additive.blending).toBe(THREE.AdditiveBlending);
      expect(normal.blending).toBe(THREE.NormalBlending);
    });

    it('should handle radius scale parameter', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        radiusScale: 1.0 / 255.0, // For uint8 radii
      });

      expect(material.uniforms.radiusScale.value).toBeCloseTo(1.0 / 255.0);
    });

    it('should handle sharpness scale parameter', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        sharpnessScale: 1.0 / 255.0, // For uint8 sharpness
      });

      expect(material.uniforms.sharpnessScale.value).toBeCloseTo(1.0 / 255.0);
    });
  });

  // =========================================================================
  // CACHING
  // =========================================================================

  describe('Material Caching', () => {
    it('should cache materials with same properties', () => {
      const props: PointMaterialProperties = {
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      };

      const material1 = manager.getPointMaterial(props);
      const material2 = manager.getPointMaterial(props);

      // Should return same instance (cached)
      expect(material1).toBe(material2);

      // Verify cache statistics
      const stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(1); // Only 1 unique material
    });

    it('should create different materials for different opacity', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 0.5, // Different!
        gamma: 1.0,
      });

      expect(material1).not.toBe(material2);

      const stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(2); // 2 different materials
    });

    it('should create different materials for different gamma', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 2.2, // Different!
      });

      expect(material1).not.toBe(material2);
    });

    it('should create different materials for different blending modes', () => {
      const additive = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const normal = manager.getPointMaterial({
        blendingMode: 'normal', // Different!
        opacity: 1.0,
        gamma: 1.0,
      });

      expect(additive).not.toBe(normal);
    });

    it('should create different materials for different radius scales', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        radiusScale: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        radiusScale: 1.0 / 255.0, // Different!
      });

      expect(material1).not.toBe(material2);
    });
  });

  // =========================================================================
  // GLOBAL UPDATES
  // =========================================================================

  describe('Global Updates', () => {
    it('should update camera params for all materials', () => {
      // Create multiple materials
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
      });

      // Update camera params globally
      const newFov = Math.PI / 3; // 60 degrees
      const newResolution = new THREE.Vector2(1920, 1080);

      manager.updateCameraParams(newFov, newResolution);

      // Both materials should be updated with pre-computed tanHalfFov
      expect(material1.uniforms.tanHalfFov.value).toBeCloseTo(Math.tan(newFov / 2), 10);
      expect(material1.uniforms.resolution.value.x).toBe(1920);
      expect(material1.uniforms.resolution.value.y).toBe(1080);

      expect(material2.uniforms.tanHalfFov.value).toBeCloseTo(Math.tan(newFov / 2), 10);
      expect(material2.uniforms.resolution.value.x).toBe(1920);
      expect(material2.uniforms.resolution.value.y).toBe(1080);
    });

    it('should update HDR multiplier for all materials', () => {
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 1.0,
      });

      // Update HDR multiplier globally
      manager.updateHDRMultiplier(32.0);

      // Both materials should be updated
      expect(material1.uniforms.hdrMultiplier.value).toBe(32.0);
      expect(material2.uniforms.hdrMultiplier.value).toBe(32.0);
    });

    it('should store current camera params for new materials', () => {
      // Update params before creating material
      const fov = Math.PI / 4; // 45 degrees
      const resolution = new THREE.Vector2(2560, 1440);

      manager.updateCameraParams(fov, resolution);

      // Create new material
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      // Should have current params (with pre-computed tanHalfFov)
      expect(material.uniforms.tanHalfFov.value).toBeCloseTo(Math.tan(fov / 2), 10);
      expect(material.uniforms.resolution.value.x).toBe(2560);
      expect(material.uniforms.resolution.value.y).toBe(1440);
    });

    it('should store current HDR multiplier for new materials (Bug Fix #11)', () => {
      // This tests the critical bug fix: materials created AFTER updateHDRMultiplier()
      // must receive the updated HDR value, not the config default.
      // This is essential for settings loaded from localStorage before scene loading.

      // Update HDR multiplier BEFORE creating material (simulates settings loaded from localStorage)
      manager.updateHDRMultiplier(25.0);

      // Create new material AFTER HDR update
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      // Material should have the updated HDR value, not the config default (16.0)
      expect(material.uniforms.hdrMultiplier.value).toBe(25.0);
    });

    it('should apply stored HDR multiplier to multiple new materials', () => {
      // Update HDR multiplier
      manager.updateHDRMultiplier(50.0);

      // Create multiple materials after HDR update
      const material1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const material2 = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
      });

      // Both materials should have the updated HDR value
      expect(material1.uniforms.hdrMultiplier.value).toBe(50.0);
      expect(material2.uniforms.hdrMultiplier.value).toBe(50.0);
    });
  });

  // =========================================================================
  // DISPOSAL
  // =========================================================================

  describe('dispose', () => {
    it('should dispose all cached materials', () => {
      // Create several materials
      const mat1 = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const mat2 = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
      });

      // Dispose manager
      manager.dispose();

      // Materials should be disposed
      expect(mat1.dispose).toHaveBeenCalled();
      expect(mat2.dispose).toHaveBeenCalled();

      // Cache should be cleared
      const stats = manager.getCacheStats();
      expect(stats.pointMaterials).toBe(0);
      expect(stats.totalRegistered).toBe(0);
    });

    it('should clear cache on dispose', () => {
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      expect(manager.getCacheStats().pointMaterials).toBe(1);

      manager.dispose();

      expect(manager.getCacheStats().pointMaterials).toBe(0);
    });
  });

  // =========================================================================
  // CACHE STATISTICS
  // =========================================================================

  describe('getCacheStats', () => {
    it('should return accurate cache statistics', () => {
      const stats1 = manager.getCacheStats();
      expect(stats1.pointMaterials).toBe(0);
      expect(stats1.totalRegistered).toBe(0);
      expect(stats1.keys).toEqual([]);

      // Create materials
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5,
        gamma: 2.2,
      });

      const stats2 = manager.getCacheStats();
      expect(stats2.pointMaterials).toBe(2);
      expect(stats2.totalRegistered).toBe(2);
      expect(stats2.keys.length).toBe(2);
    });

    it('should include cache keys in statistics', () => {
      manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const stats = manager.getCacheStats();
      expect(stats.keys.length).toBeGreaterThan(0);
      expect(stats.keys[0]).toContain('point_');
      expect(stats.keys[0]).toContain('additive');
    });
  });

  // =========================================================================
  // BLENDING MODE CONVERSION
  // =========================================================================

  describe('Blending Mode Conversion', () => {
    it('should convert normal blending mode', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 1.0,
      });

      expect(material.blending).toBe(THREE.NormalBlending);
    });

    it('should convert additive blending mode', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      expect(material.blending).toBe(THREE.AdditiveBlending);
    });

    it('should set depth write for opaque normal blending', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.99, // Opaque threshold
        gamma: 1.0,
      });

      expect(material.depthWrite).toBe(true);
    });

    it('should disable depth write for transparent materials', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.5, // Transparent
        gamma: 1.0,
      });

      expect(material.depthWrite).toBe(false);
    });

    it('should disable depth write for additive blending', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0, // Even if opaque
        gamma: 1.0,
      });

      expect(material.depthWrite).toBe(false);
    });
  });

  // =========================================================================
  // SHADER VERIFICATION
  // =========================================================================

  describe('Shader Content Verification', () => {
    it('should generate shaders with world-space sizing', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      // Verify world-space sizing formula
      expect(material.vertexShader).toContain('2.0 * normalizedRadius * resolution.y');
      expect(material.vertexShader).toContain('distance * tanHalfFov');
    });

    it('should generate shaders with HDR support', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      // Verify HDR is applied before gamma
      expect(material.fragmentShader).toContain('vColor * hdrMultiplier');
    });

    it('should generate shaders with sharpness compensation', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      // Verify sharpness compensation exists
      expect(material.vertexShader).toContain('sharpnessCompensation');
      expect(material.vertexShader).toContain('vSharpness');
    });

    it('should generate shaders with gamma correction', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 2.2,
      });

      // Verify gamma correction in fragment shader
      expect(material.fragmentShader).toContain('pow(hdrColor, vec3(invGamma))');
      expect(material.uniforms.invGamma.value).toBeCloseTo(1.0 / 2.2);
    });
  });

  // =========================================================================
  // MATERIAL UPDATES
  // =========================================================================

  describe('Material Updates', () => {
    it('should update newly created materials with current params', () => {
      // Set camera params first
      const fov = Math.PI / 2;
      const resolution = new THREE.Vector2(3840, 2160);
      manager.updateCameraParams(fov, resolution);

      // Create material after update
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      // Should have current params (with pre-computed tanHalfFov)
      expect(material.uniforms.tanHalfFov.value).toBeCloseTo(Math.tan(fov / 2), 10);
      expect(material.uniforms.resolution.value.x).toBe(3840);
      expect(material.uniforms.resolution.value.y).toBe(2160);
    });

    it('should update existing materials when params change', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
      });

      const initialTanHalfFov = material.uniforms.tanHalfFov.value;

      // Update params
      const newFov = Math.PI / 4;
      manager.updateCameraParams(newFov, new THREE.Vector2(1280, 720));

      // Material should be updated with pre-computed tanHalfFov
      expect(material.uniforms.tanHalfFov.value).toBeCloseTo(Math.tan(newFov / 2), 10);
      expect(material.uniforms.tanHalfFov.value).not.toBe(initialTanHalfFov);
    });
  });

  // =========================================================================
  // EDGE CASES
  // =========================================================================

  describe('Edge Cases', () => {
    it('should handle zero opacity', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 0.0,
        gamma: 1.0,
      });

      expect(material.uniforms.opacity.value).toBe(0.0);
      expect(material.depthWrite).toBe(false); // Transparent
    });

    it('should handle maximum gamma', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'normal',
        opacity: 1.0,
        gamma: 10.0, // Extreme value
      });

      expect(material.uniforms.gamma.value).toBe(10.0);
      expect(material.uniforms.invGamma.value).toBeCloseTo(0.1);
    });

    it('should handle very small radius scale', () => {
      const material = manager.getPointMaterial({
        blendingMode: 'additive',
        opacity: 1.0,
        gamma: 1.0,
        radiusScale: 0.00001,
      });

      expect(material.uniforms.radiusScale.value).toBeCloseTo(0.00001);
    });
  });
});
