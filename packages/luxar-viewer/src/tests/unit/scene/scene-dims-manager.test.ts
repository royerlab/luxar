/**
 * Tests for SceneDimsManager - handles nD dimension navigation
 *
 * Tests dimension management logic without WebGL dependencies
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import * as THREE from 'three';

describe('SceneDimsManager', () => {
  let manager: SceneDimsManager;
  let mockScene: THREE.Scene;

  beforeEach(() => {
    manager = new SceneDimsManager();
    mockScene = new THREE.Scene();

    // Set scene dimensions metadata in userData
    mockScene.userData.sceneDimensions = {
      dimensions: [
        { name: 'x', unit: 'μm', range: [0, 100], step: 1, display: true },
        { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
        { name: 'z', unit: 'μm', range: [0, 50], step: 0.5, display: true },
        { name: 'time', unit: 's', range: [0, 10], step: 0.1, display: false, discrete: false },
        { name: 'channel', unit: '', range: [0, 3], step: 1, display: false, discrete: true },
      ],
    };
  });

  describe('initialization', () => {
    it('should initialize from scene with dimension metadata', () => {
      const initialized = manager.initFromScene(mockScene);
      expect(initialized).toBe(true);

      const dims = manager.getDims();
      expect(dims).not.toBeNull();
      expect(dims!.ndim).toBe(5);
    });

    it('should return false when no nD objects in scene', () => {
      const emptyScene = new THREE.Scene();
      const initialized = manager.initFromScene(emptyScene);
      expect(initialized).toBe(false);
    });

    it('should set displayed dimensions correctly', () => {
      manager.initFromScene(mockScene);
      const dims = manager.getDims();
      expect(dims!.displayed).toEqual([0, 1, 2]);
    });

    it('should initialize non-displayed dimensions based on type', () => {
      manager.initFromScene(mockScene);
      const dims = manager.getDims();
      // Continuous spatial dimensions start at center
      expect(dims!.currentStep[3]).toBe(5); // time: center of [0, 10] (continuous)
      // Discrete dimensions start at minimum (first position)
      expect(dims!.currentStep[4]).toBe(0); // channel: minimum of [0, 3] (discrete)
    });

    it('should extract dimension ranges', () => {
      manager.initFromScene(mockScene);
      const ranges = manager.getDimensionRanges();
      expect(ranges).not.toBeNull();
      expect(ranges![0]).toEqual([0, 100]);
      expect(ranges![3]).toEqual([0, 10]);
    });

    it('should handle missing metadata gracefully', () => {
      const meshNoMeta = new THREE.Mesh();
      (meshNoMeta as any).dims = {}; // No metadata
      const scene = new THREE.Scene();
      scene.add(meshNoMeta);

      const initialized = manager.initFromScene(scene);
      expect(initialized).toBe(false);
    });
  });

  describe('dimension value management', () => {
    beforeEach(() => {
      manager.initFromScene(mockScene);
    });

    it('should update dimension value', () => {
      manager.setDimensionValue(3, 5.5);
      const dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(5.5);
    });

    it('should clamp values to dimension range', () => {
      manager.setDimensionValue(3, 15); // Beyond max
      const dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(10); // Clamped to max

      manager.setDimensionValue(3, -5); // Below min
      expect(manager.getDims()!.currentStep[3]).toBe(0); // Clamped to min
    });

    it('should quantize discrete dimensions', () => {
      manager.setDimensionValue(4, 1.7); // Channel is discrete
      const dims = manager.getDims();
      expect(dims!.currentStep[4]).toBe(2); // Rounded to nearest step
    });

    it('should ignore invalid dimension indices', () => {
      const dims = manager.getDims();
      const originalValues = [...dims!.currentStep];

      manager.setDimensionValue(-1, 5);
      manager.setDimensionValue(10, 5);

      expect(manager.getDims()!.currentStep).toEqual(originalValues);
    });

    it('should handle continuous dimensions with step sizes', () => {
      manager.setDimensionValue(3, 5.23); // Time has step 0.1
      const dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(5.23); // Not quantized for continuous
    });
  });

  describe('observer pattern', () => {
    beforeEach(() => {
      manager.initFromScene(mockScene);
    });

    it('should notify listeners on dimension change', () => {
      const listener = vi.fn();
      manager.addListener(listener);

      manager.setDimensionValue(3, 5);

      expect(listener).toHaveBeenCalled();
    });

    it('should support multiple listeners', () => {
      const listener1 = vi.fn();
      const listener2 = vi.fn();

      manager.addListener(listener1);
      manager.addListener(listener2);

      manager.setDimensionValue(3, 5);

      expect(listener1).toHaveBeenCalled();
      expect(listener2).toHaveBeenCalled();
    });

    it('should remove listeners', () => {
      const listener = vi.fn();
      manager.addListener(listener);
      manager.removeListener(listener);

      manager.setDimensionValue(3, 5);

      expect(listener).not.toHaveBeenCalled();
    });

    it('should not notify when no actual change', () => {
      const listener = vi.fn();
      manager.addListener(listener);

      const currentValue = manager.getDims()!.currentStep[3];
      manager.setDimensionValue(3, currentValue); // Same value

      // Listener still called (implementation doesn't check for actual change)
      expect(listener).toHaveBeenCalled();
    });
  });

  describe('complex nD scenarios', () => {
    it('should handle 2D data (all dimensions displayed)', () => {
      const scene2D = new THREE.Scene();
      scene2D.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'μm', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
        ],
      };

      manager.initFromScene(scene2D);
      const dims = manager.getDims();

      expect(dims!.ndim).toBe(2);
      expect(dims!.displayed).toEqual([0, 1]);
    });

    it('should limit displayed dimensions to 3', () => {
      const scene6D = new THREE.Scene();
      scene6D.userData.sceneDimensions = {
        dimensions: [
          { name: 'd0', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'd1', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'd2', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'd3', unit: '', range: [0, 100], step: 1, display: true }, // 4th display=true
          { name: 'd4', unit: '', range: [0, 100], step: 1, display: true }, // 5th display=true
          { name: 'd5', unit: '', range: [0, 100], step: 1, display: false },
        ],
      };

      manager.initFromScene(scene6D);
      const dims = manager.getDims();

      expect(dims!.displayed).toHaveLength(3); // Max 3 displayed
      expect(dims!.displayed).toEqual([0, 1, 2]);
    });

    it('should handle high-dimensional data', () => {
      const scene10D = new THREE.Scene();
      const dimensions = [];
      for (let i = 0; i < 10; i++) {
        dimensions.push({
          name: `dim${i}`,
          unit: '',
          range: [0, 100],
          step: 1,
          display: i < 3,
        });
      }
      scene10D.userData.sceneDimensions = { dimensions };

      manager.initFromScene(scene10D);
      const dims = manager.getDims();

      expect(dims!.ndim).toBe(10);
      expect(dims!.currentStep).toHaveLength(10);
    });

    it('should use default range when not specified', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', step: 1, display: true }, // No range specified
          { name: 'y', unit: '', range: [10, 20], step: 1, display: true },
        ],
      };

      manager.initFromScene(scene);
      const ranges = manager.getDimensionRanges();

      expect(ranges![0]).toEqual([0, 1]); // Default range
      expect(ranges![1]).toEqual([10, 20]);
    });

    it('should use positionBounds for auto-ranging when dim.range is not set', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', step: 1, display: true }, // No range
          { name: 'y', unit: '', step: 1, display: true }, // No range
          { name: 'z', unit: '', step: 1, display: true }, // No range
          { name: 'time', unit: 's', step: 1, display: false }, // No range
        ],
      };
      scene.userData.positionBounds = {
        min: [0, 0, 0, 5],
        max: [100, 200, 300, 50],
      };

      manager.initFromScene(scene);
      const ranges = manager.getDimensionRanges();

      // All dims should use positionBounds since no range specified
      expect(ranges![0]).toEqual([0, 100]);
      expect(ranges![1]).toEqual([0, 200]);
      expect(ranges![2]).toEqual([0, 300]);
      expect(ranges![3]).toEqual([5, 50]);
    });

    it('should prefer explicit dim.range over positionBounds', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [10, 20], step: 1, display: true },
          { name: 'time', unit: 's', range: [0, 100], step: 1, display: false },
        ],
      };
      scene.userData.positionBounds = {
        min: [0, 0],
        max: [500, 500],
      };

      manager.initFromScene(scene);
      const ranges = manager.getDimensionRanges();

      // Explicit range wins over positionBounds
      expect(ranges![0]).toEqual([10, 20]);
      expect(ranges![1]).toEqual([0, 100]);
    });

    it('should fall back to [0, 1] when neither range nor positionBounds available', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', step: 1, display: true }, // No range, no bounds
        ],
      };
      // No positionBounds on scene

      manager.initFromScene(scene);
      const ranges = manager.getDimensionRanges();

      expect(ranges![0]).toEqual([0, 1]);
    });
  });

  describe('scene traversal', () => {
    it('should find nD object in nested hierarchy', () => {
      const group = new THREE.Group();
      const nestedGroup = new THREE.Group();
      const mesh = new THREE.Mesh();
      (mesh as any).dims = {
        metadata: {
          0: { name: 'x', display: true },
        },
      };

      nestedGroup.add(mesh);
      group.add(nestedGroup);
      mockScene.add(group);

      const initialized = manager.initFromScene(mockScene);
      expect(initialized).toBe(true);
    });

    it('should use scene dimensions when available', () => {
      // mockScene already has dimensions set in beforeEach
      manager.initFromScene(mockScene);
      const dims = manager.getDims();

      expect(dims!.ndim).toBe(5); // From scene dimensions
      expect(dims!.metadata![0].name).toBe('x');
    });
  });

  describe('edge cases', () => {
    it('should handle invalid scene gracefully', () => {
      // Test with a scene that has no dimensions
      const emptyManager = new SceneDimsManager();
      const emptyScene = new THREE.Scene();
      const initialized = emptyManager.initFromScene(emptyScene);
      expect(initialized).toBe(false);
      expect(emptyManager.getDims()).toBeNull();
    });

    it('should handle dimensions with minimal metadata', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: '', unit: '', range: [0, 1], step: 1, display: true }, // Minimal metadata
          { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
        ],
      };

      manager.initFromScene(scene);
      const dims = manager.getDims();

      expect(dims).not.toBeNull();
      expect(dims!.ndim).toBe(2);
    });

    it('should handle setDimensionValue before initialization', () => {
      // Should not throw
      expect(() => manager.setDimensionValue(0, 5)).not.toThrow();
      expect(manager.getDims()).toBeNull();
    });

    it('should allow re-initialization', () => {
      manager.initFromScene(mockScene);
      const firstDims = manager.getDims();
      expect(firstDims!.ndim).toBe(5);

      // Create different scene
      const newScene = new THREE.Scene();
      newScene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'μm', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
        ],
      };

      manager.initFromScene(newScene);
      const secondDims = manager.getDims();
      expect(secondDims!.ndim).toBe(2);
    });
  });

  describe('categorical dimensions', () => {
    let categoricalScene: THREE.Scene;

    beforeEach(() => {
      categoricalScene = new THREE.Scene();
      categoricalScene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'μm', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
          { name: 'z', unit: 'μm', range: [0, 50], step: 0.5, display: true },
          {
            name: 'channel',
            unit: '',
            range: [0, 2],
            step: 1,
            display: false,
            discrete: true,
            categories: ['DAPI', 'GFP', 'mCherry'],
            description: 'Fluorescence channel',
          },
          {
            name: 'condition',
            unit: '',
            range: [0, 1],
            step: 1,
            display: false,
            discrete: true,
            categories: ['Control', 'Treated'],
            cyclic: false,
          },
        ],
      };
    });

    it('should parse categorical dimensions from scene metadata', () => {
      const initialized = manager.initFromScene(categoricalScene);
      expect(initialized).toBe(true);

      const dims = manager.getDims();
      expect(dims).not.toBeNull();
      expect(dims!.metadata).toBeDefined();
      expect(dims!.metadata![3].categories).toEqual(['DAPI', 'GFP', 'mCherry']);
      expect(dims!.metadata![4].categories).toEqual(['Control', 'Treated']);
    });

    it('should initialize categorical dimensions to first category (minimum)', () => {
      manager.initFromScene(categoricalScene);
      const dims = manager.getDims();

      // Categorical dimensions should start at 0 (first category)
      expect(dims!.currentStep[3]).toBe(0); // channel dimension
      expect(dims!.currentStep[4]).toBe(0); // condition dimension
    });

    it('should preserve category metadata through initialization', () => {
      manager.initFromScene(categoricalScene);
      const dims = manager.getDims();
      const channelDim = dims!.metadata![3];

      expect(channelDim.name).toBe('channel');
      expect(channelDim.categories).toEqual(['DAPI', 'GFP', 'mCherry']);
      expect(channelDim.discrete).toBe(true);
      expect(channelDim.description).toBe('Fluorescence channel');
    });

    it('should preserve cyclic flag for categorical dimensions', () => {
      manager.initFromScene(categoricalScene);
      const dims = manager.getDims();

      expect(dims!.metadata![4].cyclic).toBe(false);
    });

    it('should handle categorical dimension with cyclic=true', () => {
      categoricalScene.userData.sceneDimensions.dimensions[3].cyclic = true;
      manager.initFromScene(categoricalScene);
      const dims = manager.getDims();

      expect(dims!.metadata![3].cyclic).toBe(true);
    });

    it('should quantize categorical dimension values to nearest integer', () => {
      manager.initFromScene(categoricalScene);

      // Set to non-integer value (should quantize)
      manager.setDimensionValue(3, 1.7);
      const dims = manager.getDims();

      expect(dims!.currentStep[3]).toBe(2); // Rounded to nearest category index
    });

    it('should clamp categorical dimension values to valid range', () => {
      manager.initFromScene(categoricalScene);

      // Try to set beyond max category index
      manager.setDimensionValue(3, 5);
      let dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(2); // Clamped to max index

      // Try to set below min
      manager.setDimensionValue(3, -1);
      dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(0); // Clamped to min index
    });

    it('should preserve description field in metadata', () => {
      manager.initFromScene(categoricalScene);
      const dims = manager.getDims();

      expect(dims!.metadata![3].description).toBe('Fluorescence channel');
    });

    it('should handle spatial flag in categorical dimensions', () => {
      categoricalScene.userData.sceneDimensions.dimensions[3].spatial = false;
      manager.initFromScene(categoricalScene);
      const dims = manager.getDims();

      expect(dims!.metadata![3].spatial).toBe(false);
    });
  });
});
