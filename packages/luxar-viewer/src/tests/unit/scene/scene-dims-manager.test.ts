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
      // W1: a truthy `dims` is not enough — pin the full shape so a mutant that
      // returns true but builds a malformed state object is killed.
      expect(dims!.currentStep).toHaveLength(5);
      expect(dims!.displayed).toEqual([0, 1, 2]);
      expect(dims!.metadata).toHaveLength(5);
      expect(dims!.metadata!.map((m) => m.name)).toEqual(['x', 'y', 'z', 'time', 'channel']);
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

    // Regression (deep-double-check): the INITIAL discrete position must be
    // snapped onto the k·step grid like every subsequent navigation
    // (setDimensionValue snaps; the discrete chunk query reaches only a
    // quarter-step). A raw off-grid range.min (e.g. 1.3, step 1) left the
    // initial view silently empty until the first manual navigation.
    it('snaps the initial discrete position onto the step grid (first on-grid ≥ min)', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 50], step: 1, display: true },
          // Off-grid declared min: first on-grid position at/above 1.3 is 2.
          { name: 'time', unit: '', range: [1.3, 5.3], step: 1, display: false, discrete: true },
        ],
      };
      manager.initFromScene(scene);
      expect(manager.getDims()!.currentStep[3]).toBe(2);
    });

    it('keeps an already on-grid discrete minimum unchanged', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 50], step: 1, display: true },
          { name: 'time', unit: '', range: [2, 8], step: 2, display: false, discrete: true },
        ],
      };
      manager.initFromScene(scene);
      expect(manager.getDims()!.currentStep[3]).toBe(2);
    });

    it('falls back to the raw min when no on-grid point exists inside the range', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 100], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 50], step: 1, display: true },
          // Range narrower than a step with no multiple of 10 inside.
          { name: 'time', unit: '', range: [1.2, 1.8], step: 10, display: false, discrete: true },
        ],
      };
      manager.initFromScene(scene);
      expect(manager.getDims()!.currentStep[3]).toBe(1.2);
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
      const before = [...manager.getDims()!.currentStep];
      manager.setDimensionValue(3, 5.5);
      const after = manager.getDims()!.currentStep;
      expect(after[3]).toBe(5.5);
      // M3: only the targeted dimension may change — a mutant that writes the
      // wrong index or mutates a sibling element is caught.
      const changedIndices = after.map((v, i) => (v !== before[i] ? i : -1)).filter((i) => i >= 0);
      expect(changedIndices).toEqual([3]);
    });

    it('should clamp values to dimension range', () => {
      manager.setDimensionValue(3, 15); // Beyond max
      const dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(10); // Clamped to max

      manager.setDimensionValue(3, -5); // Below min
      expect(manager.getDims()!.currentStep[3]).toBe(0); // Clamped to min

      // M4: a value strictly inside [min, max] must pass through UNCHANGED.
      // Without this, a mutant clamp that always returns `min` (or `max`)
      // would survive the two out-of-range cases above.
      manager.setDimensionValue(3, 4); // 4 ∈ [0, 10]
      expect(manager.getDims()!.currentStep[3]).toBe(4);
    });

    it('should quantize discrete dimensions', () => {
      manager.setDimensionValue(4, 1.7); // Channel is discrete
      const dims = manager.getDims();
      expect(dims!.currentStep[4]).toBe(2); // Rounded to nearest step
    });

    // M5: the discrete quantizer is Math.round(value/step)*step. With an
    // integer step (1) and value 1.7, both round and ceil give 2, so the
    // earlier test cannot distinguish them. Use a fractional step where
    // round ≠ ceil ≠ floor to pin the rounding rule precisely.
    it('quantizes discrete dimensions with Math.round (not floor/ceil) for fractional steps', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 10], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 10], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 10], step: 1, display: true },
          { name: 'g', unit: '', range: [0, 10], step: 0.3, display: false, discrete: true },
        ],
      };
      const m = new SceneDimsManager();
      m.initFromScene(scene);
      // value 1.0, step 0.3 → 1/0.3 = 3.333 → round=3 → 3*0.3 = 0.9.
      // (floor would give 0.9 too, but ceil would give 4*0.3=1.2 → distinct.)
      m.setDimensionValue(3, 1.0);
      expect(m.getDims()!.currentStep[3]).toBeCloseTo(0.9, 10);
      // value 1.4, step 0.3 → 1.4/0.3 = 4.667 → round=5 → 1.5 (floor=1.2, ceil=1.5).
      m.setDimensionValue(3, 1.4);
      expect(m.getDims()!.currentStep[3]).toBeCloseTo(1.5, 10);
    });

    it('should ignore invalid dimension indices', () => {
      const dims = manager.getDims();
      const originalValues = [...dims!.currentStep];

      manager.setDimensionValue(-1, 5);
      manager.setDimensionValue(10, 5);

      expect(manager.getDims()!.currentStep).toEqual(originalValues);
    });

    // G1: boundary indices — the first (0) and last (ndim-1) valid dimensions
    // must be writable, and the just-out-of-range index (ndim) must be ignored.
    it('accepts the first and last valid dimension indices and rejects ndim', () => {
      // dim 0 is displayed (range [0,100]); dim 4 is the last (channel, discrete).
      manager.setDimensionValue(0, 42);
      expect(manager.getDims()!.currentStep[0]).toBe(42);

      const ndim = manager.getDims()!.ndim; // 5
      manager.setDimensionValue(ndim - 1, 2); // last dim
      expect(manager.getDims()!.currentStep[ndim - 1]).toBe(2);

      const before = [...manager.getDims()!.currentStep];
      manager.setDimensionValue(ndim, 7); // exactly out of range → ignored
      expect(manager.getDims()!.currentStep).toEqual(before);
    });

    it('should handle continuous dimensions with step sizes', () => {
      manager.setDimensionValue(3, 5.23); // Time has step 0.1
      const dims = manager.getDims();
      expect(dims!.currentStep[3]).toBe(5.23); // Not quantized for continuous
    });

    // Bug fix: NaN/Infinity inputs silently poisoned currentStep, breaking
    // every downstream slicing computation. The guard at the top of
    // setDimensionValue must reject non-finite values without mutating
    // state or notifying observers.
    it('ignores non-finite values (NaN, Infinity) without mutating state or notifying listeners', () => {
      const listener = vi.fn();
      manager.addListener(listener);

      const before = [...manager.getDims()!.currentStep];

      manager.setDimensionValue(3, NaN);
      expect(manager.getDims()!.currentStep).toEqual(before);
      expect(listener).not.toHaveBeenCalled();

      manager.setDimensionValue(3, Infinity);
      expect(manager.getDims()!.currentStep).toEqual(before);
      expect(listener).not.toHaveBeenCalled();

      manager.setDimensionValue(3, -Infinity);
      expect(manager.getDims()!.currentStep).toEqual(before);
      expect(listener).not.toHaveBeenCalled();

      // Sanity: a valid value after still works and notifies.
      manager.setDimensionValue(3, 5);
      expect(manager.getDims()!.currentStep[3]).toBe(5);
      expect(listener).toHaveBeenCalledTimes(1);
    });
  });

  describe('resetPositions', () => {
    // The rail Home popover's "Reset dimensions" action. Must apply the SAME
    // default-position policy as initFromScene: displayed → 0, discrete → min,
    // continuous non-displayed → center.

    it('restores the initFromScene defaults after navigation moved the dims', () => {
      manager.initFromScene(mockScene);
      manager.setDimensionValue(3, 8.7); // time (continuous): away from center 5
      manager.setDimensionValue(4, 3); // channel (discrete): away from min 0

      manager.resetPositions();

      const step = manager.getDims()!.currentStep;
      expect(step[0]).toBe(0); // displayed x
      expect(step[1]).toBe(0); // displayed y
      expect(step[2]).toBe(0); // displayed z
      expect(step[3]).toBe(5); // time: center of [0, 10]
      expect(step[4]).toBe(0); // channel: min of [0, 3]
    });

    it('notifies listeners exactly once', () => {
      manager.initFromScene(mockScene);
      manager.setDimensionValue(3, 8.7);
      const listener = vi.fn();
      manager.addListener(listener);

      manager.resetPositions();

      expect(listener).toHaveBeenCalledTimes(1);
    });

    it('is a safe no-op before initialization (no throw, no notify)', () => {
      const listener = vi.fn();
      manager.addListener(listener);

      expect(() => manager.resetPositions()).not.toThrow();
      expect(listener).not.toHaveBeenCalled();
      expect(manager.getDims()).toBeNull();
    });

    it('resets a categorical dimension to its first category', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: 'μm', range: [0, 100], step: 1, display: true },
          { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
          { name: 'z', unit: 'μm', range: [0, 50], step: 1, display: true },
          {
            name: 'stain',
            unit: '',
            range: [0, 2],
            step: 1,
            display: false,
            categories: ['DAPI', 'GFP', 'RFP'],
          },
        ],
      };
      manager.initFromScene(scene);
      manager.setDimensionValue(3, 2); // navigate to 'RFP'

      manager.resetPositions();

      expect(manager.getDims()!.currentStep[3]).toBe(0); // back to 'DAPI'
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

    it("an OLDER update settling does not clobber a NEWER update's waitForUpdate tracking", async () => {
      // Regression: notifyListeners used to null pendingUpdatePromise
      // unconditionally when ANY update settled — so an older (slower)
      // update's completion made waitForUpdate() resolve immediately while
      // a newer update was still loading (a real hazard now that queued
      // scene-loader updates keep listener promises pending until their
      // pass commits). The guard only nulls the field when it still points
      // at the settling promise.
      const deferreds: Array<() => void> = [];
      manager.addListener(
        () =>
          new Promise<void>((resolve) => {
            deferreds.push(resolve);
          })
      );

      manager.setDimensionValue(3, 1); // update #1 → deferred[0]
      manager.setDimensionValue(3, 2); // update #2 → deferred[1] (tracked)

      // Settle the OLDER update first. Pre-guard, its completion nulled
      // pendingUpdatePromise unconditionally — so the waitForUpdate() call
      // BELOW got an instantly-resolved promise while update #2 was still
      // loading.
      deferreds[0]();
      await new Promise((resolve) => setTimeout(resolve, 0)); // full microtask drain

      let settled = false;
      void manager.waitForUpdate().then(() => {
        settled = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(settled).toBe(false); // newer update still pending

      deferreds[1]();
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(settled).toBe(true);
    });

    it('should support multiple listeners', () => {
      const listener1 = vi.fn();
      const listener2 = vi.fn();

      manager.addListener(listener1);
      manager.addListener(listener2);

      manager.setDimensionValue(3, 5);

      // M7: pin the exact call count (once each) — `toHaveBeenCalled()` alone
      // would survive a mutant that fired a listener twice or registered it
      // under a deduping bug.
      expect(listener1).toHaveBeenCalledTimes(1);
      expect(listener2).toHaveBeenCalledTimes(1);
    });

    it('should remove listeners', () => {
      const listener = vi.fn();
      manager.addListener(listener);
      manager.removeListener(listener);

      // M8: call twice after removal. A no-op removeListener would let the
      // listener fire; asserting zero across two updates makes that survive
      // only if removal genuinely unregistered the callback.
      manager.setDimensionValue(3, 5);
      manager.setDimensionValue(3, 6);

      expect(listener).not.toHaveBeenCalled();

      // Re-adding the same callback after removal must work again.
      manager.addListener(listener);
      manager.setDimensionValue(3, 7);
      expect(listener).toHaveBeenCalledTimes(1);
    });

    // G10: notifyListeners wraps each callback in try/catch — a throwing
    // listener must not prevent the others from running.
    it('continues notifying remaining listeners when one listener throws', () => {
      const order: string[] = [];
      const throwing = vi.fn(() => {
        order.push('throwing');
        throw new Error('boom');
      });
      const survivor = vi.fn(() => {
        order.push('survivor');
      });
      manager.addListener(throwing);
      manager.addListener(survivor);

      expect(() => manager.setDimensionValue(3, 5)).not.toThrow();
      expect(throwing).toHaveBeenCalledTimes(1);
      expect(survivor).toHaveBeenCalledTimes(1);
      expect(order).toContain('survivor');
    });

    it('still notifies listeners even when setDimensionValue is given the current value', () => {
      // The implementation does not dedupe identical-value writes; listeners
      // fire on every setDimensionValue call regardless of whether the value
      // changed. This test pins that contract so we notice if dedup is added.
      const listener = vi.fn();
      manager.addListener(listener);

      const currentValue = manager.getDims()!.currentStep[3];
      manager.setDimensionValue(3, currentValue);

      expect(listener).toHaveBeenCalledTimes(1);
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

    // M9: the previous tests all had display=true on the first dimensions, so
    // a mutant that pushes the first three indices while IGNORING the display
    // flag would survive. Use a non-contiguous flag pattern [F,T,F,T,T] so the
    // displayed list must be exactly the flagged indices.
    it('selects displayed dimensions by the display flag, not by position', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'd0', unit: '', range: [0, 1], step: 1, display: false },
          { name: 'd1', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'd2', unit: '', range: [0, 1], step: 1, display: false },
          { name: 'd3', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'd4', unit: '', range: [0, 1], step: 1, display: true },
        ],
      };
      const m = new SceneDimsManager();
      m.initFromScene(scene);
      expect(m.getDims()!.displayed).toEqual([1, 3, 4]);
    });

    // G4: negative ranges must clamp/center correctly (the helpers must not
    // assume min >= 0). A continuous non-displayed dim over [-100, -20]
    // initializes at the center (-60); clamping respects the negative bounds.
    it('handles negative dimension ranges for clamping and centering', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'w', unit: '', range: [-100, -20], step: 1, display: false, discrete: false },
        ],
      };
      const m = new SceneDimsManager();
      m.initFromScene(scene);
      // Continuous non-displayed dim starts at center of [-100, -20] = -60.
      expect(m.getDims()!.currentStep[3]).toBe(-60);
      // Clamp below min and above max within the negative range.
      m.setDimensionValue(3, -200);
      expect(m.getDims()!.currentStep[3]).toBe(-100);
      m.setDimensionValue(3, 0);
      expect(m.getDims()!.currentStep[3]).toBe(-20);
      // A value inside the range passes through.
      m.setDimensionValue(3, -50);
      expect(m.getDims()!.currentStep[3]).toBe(-50);
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

  describe('getDimensionNames / getDimensionUnits', () => {
    // G8: the name/unit getters return fallbacks for missing metadata. These
    // were previously untested.
    it('returns empty arrays before initialization', () => {
      const m = new SceneDimsManager();
      expect(m.getDimensionNames()).toEqual([]);
      expect(m.getDimensionUnits()).toEqual([]);
    });

    it('returns the metadata names and units when present', () => {
      manager.initFromScene(mockScene);
      expect(manager.getDimensionNames()).toEqual(['x', 'y', 'z', 'time', 'channel']);
      expect(manager.getDimensionUnits()).toEqual(['μm', 'μm', 'μm', 's', '']);
    });

    it('falls back to "Dim i" names and "" units when metadata fields are empty', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: '', unit: '', range: [0, 1], step: 1, display: true },
          { name: '', unit: '', range: [0, 1], step: 1, display: true },
        ],
      };
      const m = new SceneDimsManager();
      m.initFromScene(scene);
      // Empty name → "Dim i" fallback; empty unit stays "".
      expect(m.getDimensionNames()).toEqual(['Dim 0', 'Dim 1']);
      expect(m.getDimensionUnits()).toEqual(['', '']);
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
