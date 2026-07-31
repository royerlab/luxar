import { SimpleDims, DimensionMetadata } from '../types/dims';
import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import { clamp } from '../utils/clamp';

/**
 * Centralized dimension state manager ensuring consistency across all nD objects in the scene.
 *
 * This singleton is a critical architectural component that solves the fundamental problem
 * of keeping multiple nD points synchronized when navigating through dimensional space.
 * Without it, each points could have its own slice position, leading to confusing
 * and inconsistent visualizations.
 *
 * Core responsibilities:
 * - Initialize dimension metadata from scene-level configuration
 * - Maintain a single source of truth for current dimension positions
 * - Coordinate slider updates with points re-slicing
 * - Handle discrete vs continuous dimension semantics
 * - Provide change notification system for reactive updates
 *
 * Design decisions:
 * - Singleton pattern ensures only one dimension state exists per scene
 * - Non-displayed dimensions start at minimum values for predictable behavior
 * - Discrete dimensions are quantized to specified step sizes
 * - Observable pattern allows UI components to react to dimension changes
 *
 * Initialization flow:
 * 1. Scene loads with nD objects containing dimension metadata
 * 2. SceneDimsManager extracts metadata from first nD object found
 * 3. Establishes ranges, display preferences, and initial positions
 * 4. UI components (sliders, keyboard handlers) register for updates
 * 5. Navigation events trigger coordinated updates across all objects
 *
 * @class SceneDimsManager
 */
export class SceneDimsManager {
  /** The shared dimension state for the entire scene */
  private dims: SimpleDims | null = null;

  /** Min/max bounds for each dimension derived from metadata */
  private dimensionRanges: Array<[number, number]> | null = null;

  /** Observer callbacks that react to dimension changes (can be async) */
  private listeners: Set<() => void | Promise<void>> = new Set();

  /** Promise tracking pending listener completion (for animation synchronization) */
  private pendingUpdatePromise: Promise<void> | null = null;

  /**
   * Initializes the scene-level dimension state from metadata embedded in the scene.
   *
   * This method searches through the THREE.js scene hierarchy to find nD objects
   * with embedded dimension metadata, then establishes the shared dimensional
   * coordinate system for the entire scene.
   *
   * Search strategy:
   * 1. Check scene.userData.sceneDimensions first (direct attachment)
   * 2. Recursively search scene children for embedded metadata
   * 3. Use the first valid dimension metadata found
   *
   * Initialization logic:
   * - Parse metadata into standardized DimensionMetadata format
   * - Establish ranges from metadata or sensible defaults
   * - Set non-displayed dimensions to their minimum values
   * - Identify which dimensions should be displayed (max 3)
   *
   * @param scene - THREE.js scene containing nD objects with metadata
   * @returns True if dimensions were successfully initialized, false if no metadata found
   */
  initFromScene(scene: THREE.Scene): boolean {
    // Step 1: Search for scene dimensions metadata
    let sceneDimensions = scene.userData.sceneDimensions;

    // Step 2: If not found on scene root, search immediate children
    if (!sceneDimensions) {
      for (const child of scene.children) {
        if (child.userData.sceneDimensions) {
          sceneDimensions = child.userData.sceneDimensions;
          break;
        }
      }
    }

    // Step 3: If still not found, look for the LuxarScene group specifically
    if (!sceneDimensions) {
      const luxarScene = scene.getObjectByName('LuxarScene');
      if (luxarScene?.userData.sceneDimensions) {
        sceneDimensions = luxarScene.userData.sceneDimensions;
      }
    }

    // Validation: Ensure we found valid dimension metadata
    if (!sceneDimensions?.dimensions) {
      // Only log error if it's expected (not all scenes have dimensions)
      // For 3D-only scenes, this is normal behavior
      // This is normal for 3D-only scenes, no need to log as error
      // The calling code will handle the false return appropriately
      //
      // Drop any PREVIOUS scene's dims: this is a singleton, so leaving them
      // in place would let a 3D-only scene loaded after an nD one be read
      // through the old scene's `displayed` axes (bounds projection,
      // auto-framing). Listeners are deliberately kept — they are owned by
      // the input handler across scene switches, unlike `reset()`.
      this.dims = null;
      this.dimensionRanges = null;
      return false;
    }

    // Step 4: Parse and normalize dimension metadata
    type RawDim = Partial<DimensionMetadata> & { range?: number[] };
    const metadata: DimensionMetadata[] = sceneDimensions.dimensions.map((dim: RawDim) => ({
      name: dim.name as string,
      unit: dim.unit as string,
      scale: dim.scale || 1.0,
      range: dim.range ? [dim.range[0], dim.range[1]] : undefined,
      display: dim.display,
      discrete: dim.discrete || false,
      cyclic: dim.cyclic || false,
      step: dim.step || 1.0,
      spatial: dim.spatial,
      categories: dim.categories,
      description: dim.description || '',
    }));

    const ndim = metadata.length;

    // Step 4b: Find positionBounds for auto-ranging fallback
    // Uses same 3-step search as sceneDimensions above
    let positionBounds: { min: number[]; max: number[] } | null = null;
    if (scene.userData.positionBounds) {
      positionBounds = scene.userData.positionBounds;
    } else {
      for (const child of scene.children) {
        if (child.userData.positionBounds) {
          positionBounds = child.userData.positionBounds;
          break;
        }
      }
      if (!positionBounds) {
        const luxarScene = scene.getObjectByName('LuxarScene');
        if (luxarScene?.userData.positionBounds) {
          positionBounds = luxarScene.userData.positionBounds;
        }
      }
    }

    // Step 5: Establish dimension ranges for navigation bounds
    this.dimensionRanges = metadata.map((meta, i) => {
      if (meta.range) {
        return meta.range as [number, number];
      }
      // Fallback: use positionBounds from zarr metadata (world-space)
      if (positionBounds && i < positionBounds.min.length && i < positionBounds.max.length) {
        return [positionBounds.min[i], positionBounds.max[i]] as [number, number];
      }
      // Last resort: unit range
      return [0, 1];
    });

    // Step 6: Initialize dimension positions (see defaultPosition for the policy)
    const currentStep = new Array(ndim).fill(0);
    for (let i = 0; i < ndim; i++) {
      currentStep[i] = SceneDimsManager.defaultPosition(metadata[i], this.dimensionRanges[i]);
    }

    // Step 7: Identify which dimensions should be displayed in 3D scene
    const displayed: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (metadata[i].display === true && displayed.length < 3) {
        displayed.push(i);
      }
    }

    // Step 8: Create the shared dimension state object
    this.dims = {
      ndim,
      currentStep,
      displayed,
      metadata,
    };

    // Note: We don't call notifyListeners() here because listeners haven't been
    // registered yet. The initial update is triggered manually in input-handler.ts
    // after the listener is registered.

    return true;
  }

  /**
   * Provides read-only access to the current dimension state.
   *
   * This is the primary interface for components that need to access
   * the current slice positions, displayed dimensions, and metadata.
   *
   * @returns Current dimension state or null if not initialized
   */
  getDims(): SimpleDims | null {
    return this.dims;
  }

  /**
   * Gets the navigable bounds for each dimension.
   *
   * These ranges define the valid navigation space and are used by
   * UI components (sliders, keyboard handlers) to constrain user input
   * and calculate appropriate step sizes.
   *
   * @returns Array of [min, max] bounds for each dimension, or null if not initialized
   */
  getDimensionRanges(): Array<[number, number]> | null {
    return this.dimensionRanges;
  }

  /**
   * Updates the position in a specific dimension and triggers re-slicing.
   *
   * This is the central method for dimension navigation, handling both
   * user input validation and observer notification. It ensures all
   * dimension changes are properly constrained and communicated.
   *
   * Value processing:
   * 1. Validate dimension index bounds
   * 2. Clamp value to valid range for this dimension
   * 3. Quantize discrete dimensions to their step size
   * 4. Update internal state
   * 5. Notify all observers (triggers UI updates and re-slicing)
   *
   * @param dimIndex - Index of dimension to update
   * @param value - New position value in dimension units
   */
  setDimensionValue(dimIndex: number, value: number): void {
    if (!this.dims || dimIndex < 0 || dimIndex >= this.dims.ndim) {
      return;
    }

    // Reject non-finite (NaN, ±Infinity) inputs — silently writing NaN into
    // currentStep would poison every downstream slicing computation.
    if (!Number.isFinite(value)) {
      log.warning(
        Modules.SCENE_DIMS,
        `setDimensionValue: ignoring non-finite value ${value} for dim ${dimIndex}`
      );
      return;
    }

    // Apply range constraints to prevent navigation beyond data bounds
    if (this.dimensionRanges) {
      const [min, max] = this.dimensionRanges[dimIndex];
      value = clamp(value, min, max);
    }

    // Handle discrete dimensions (e.g., time frames, categorical data)
    const dimMeta = this.dims.metadata?.[dimIndex];
    if (dimMeta?.discrete) {
      const step = dimMeta.step || 1.0;
      value = Math.round(value / step) * step;
    }

    this.dims.currentStep[dimIndex] = value;
    this.notifyListeners(); // Trigger reactive updates throughout the system
  }

  /**
   * The default position policy — shared by {@link initFromScene} and
   * {@link resetPositions} so the two can never diverge:
   * - Displayed dimensions (X, Y, Z): 0 (camera-controlled)
   * - Discrete/categorical dimensions (time, channels, frames): FIRST ON-GRID
   *   position at or above the range minimum. Snapping matters: every later
   *   navigation snaps to the k·step grid ({@link setDimensionValue}), and the
   *   discrete chunk query only reaches a quarter-step around the position —
   *   a raw off-grid `min` (e.g. 1.3 with step 1) would make the INITIAL view
   *   silently empty until the first user navigation snapped it.
   * - Continuous non-displayed dimensions (4th+ spatial dims): CENTER (no natural "first")
   */
  private static defaultPosition(meta: DimensionMetadata, range: [number, number]): number {
    if (meta.display === true) return 0;
    const [min, max] = range;
    if (meta.discrete || meta.categories) {
      const step = meta.step || 1.0;
      // Step-relative epsilon: for FP-hostile fractional steps an EXACTLY
      // on-grid min can round an ulp low (e.g. Math.round(2.1/0.7)*0.7 =
      // 2.0999999999999996 < 2.1) — a strict `< min` bump would then skip
      // the whole first category. Tolerate sub-epsilon undershoot.
      const eps = step * 1e-9;
      let snapped = Math.round(min / step) * step;
      if (snapped < min - eps) {
        // Bump to the next grid point, then RE-SNAP: `k*step + step` can
        // differ from `(k+1)*step` by an ulp, and the initial position must
        // be byte-identical to what setDimensionValue's own snap produces
        // for the same target (viewStatesEqual / S-cache keys compare
        // exact floats).
        snapped = Math.round((snapped + step) / step) * step;
      }
      // Pathological range narrower than one step with no on-grid point:
      // fall back to the raw min rather than leaving the range entirely.
      return snapped <= max + eps ? snapped : min;
    }
    return (min + max) / 2;
  }

  /**
   * Reset every dimension back to its initial default position (same policy
   * as {@link initFromScene}) and notify listeners — sliders, slicing, and
   * status displays all refresh reactively. Used by the rail Home popover's
   * "Reset dimensions" action. No-op before initialization.
   */
  resetPositions(): void {
    if (!this.dims || !this.dimensionRanges) return;
    for (let i = 0; i < this.dims.ndim; i++) {
      const meta = this.dims.metadata?.[i];
      if (!meta) continue;
      this.dims.currentStep[i] = SceneDimsManager.defaultPosition(meta, this.dimensionRanges[i]);
    }
    this.notifyListeners();
    log.info(Modules.SCENE_DIMS, 'Dimension positions reset to defaults');
  }

  /**
   * Registers a callback to be invoked whenever dimension state changes.
   *
   * This implements the observer pattern, allowing UI components and
   * visualization objects to react automatically to navigation events.
   * Typical subscribers include sliders, points, and status displays.
   *
   * Callbacks can be async — their completion can be awaited via waitForUpdate().
   *
   * @param callback - Function to call when dimensions change (can return Promise)
   */
  addListener(callback: () => void | Promise<void>): void {
    this.listeners.add(callback);
  }

  /**
   * Unregisters a dimension change callback.
   *
   * Important for preventing memory leaks when components are destroyed.
   *
   * @param callback - Previously registered callback function
   */
  removeListener(callback: () => void | Promise<void>): void {
    this.listeners.delete(callback);
  }

  /**
   * Triggers all registered observer callbacks.
   *
   * This is called internally whenever dimension state changes,
   * propagating updates throughout the reactive system. Async callbacks
   * are collected and their completion is tracked via currentUpdatePromise.
   *
   * @private
   */
  private notifyListeners(): void {
    const promises: Promise<void>[] = [];

    this.listeners.forEach((callback) => {
      try {
        const result = callback();
        if (result instanceof Promise) {
          promises.push(
            result.catch((error) => {
              log.error(Modules.SCENE_MANAGER, 'Dimension listener rejected:', error);
            })
          );
        }
      } catch (error) {
        log.error(Modules.SCENE_MANAGER, 'Dimension listener threw:', error);
      }
    });

    // Track combined promise for async synchronization. Only null the field
    // if it still points at THIS update's promise — an older update settling
    // late must not clobber a newer update's tracking (listener promises are
    // long-lived now that queued scene-loader updates resolve on real pass
    // completion, which widened this pre-existing race).
    if (promises.length > 0) {
      const tracked: Promise<void> = Promise.all(promises).then(() => {
        if (this.pendingUpdatePromise === tracked) {
          this.pendingUpdatePromise = null;
        }
      });
      this.pendingUpdatePromise = tracked;
    }
  }

  /**
   * Returns a promise that resolves when all pending listener updates complete.
   *
   * Used by animation systems to wait for data loading before advancing frames.
   * Returns immediately resolved promise if no update is in progress.
   *
   * @returns Promise that resolves when pending updates complete
   */
  waitForUpdate(): Promise<void> {
    return this.pendingUpdatePromise || Promise.resolve();
  }

  /**
   * Provides access to the complete dimension metadata array.
   *
   * Used by UI components that need detailed information about
   * dimension properties like names, units, discreteness, etc.
   *
   * @returns Array of dimension metadata, empty if not initialized
   */
  getDimensionMetadata(): DimensionMetadata[] {
    return this.dims?.metadata || [];
  }

  /**
   * Reset the dimension manager to initial state
   */
  reset(): void {
    this.dims = null;
    this.dimensionRanges = null;
    this.pendingUpdatePromise = null;
    this.listeners.clear();
    // Scene dimension manager has been reset
  }

  /**
   * Extracts human-readable names for all dimensions.
   *
   * Provides fallback names when metadata doesn't specify custom names.
   * Used by UI components for labeling sliders and status displays.
   *
   * @returns Array of dimension names (e.g., ["Time", "X", "Y", "Z"])
   */
  getDimensionNames(): string[] {
    if (!this.dims) return [];

    const names: string[] = [];
    for (let i = 0; i < this.dims.ndim; i++) {
      const meta = this.dims.metadata?.[i];
      const name = meta?.name || `Dim ${i}`;
      names.push(name);
    }
    return names;
  }

  /**
   * Extracts physical units for all dimensions.
   *
   * Used by UI components to display appropriate unit labels
   * next to numeric values (e.g., "μm", "s", "nm").
   *
   * @returns Array of dimension units, empty strings for dimensionless quantities
   */
  getDimensionUnits(): string[] {
    if (!this.dims) return [];

    const units: string[] = [];
    for (let i = 0; i < this.dims.ndim; i++) {
      const meta = this.dims.metadata?.[i];
      units.push(meta?.unit || '');
    }
    return units;
  }

  /**
   * Determines if the dataset has navigable dimensions beyond the displayed 3D view.
   *
   * This is used by UI components to decide whether to show dimension navigation
   * controls (sliders, keyboard hints). If all dimensions are displayed in 3D,
   * no additional navigation UI is needed.
   *
   * @returns True if there are dimensions not currently displayed in 3D space
   */
  hasNonDisplayedDimensions(): boolean {
    return this.dims ? this.dims.ndim > this.dims.displayed.length : false;
  }
}

/**
 * Page-level singleton instance of the scene dimension manager.
 *
 * Construction is **deferred until first access** via a Proxy: importing
 * this symbol no longer triggers the constructor side-effect. Tests can
 * call {@link __resetSceneDimsManagerForTests} to start fresh between
 * cases. All call-site syntax (`sceneDimsManager.getDims()`, etc.)
 * remains identical to a directly-exported instance.
 *
 * @example
 * ```typescript
 * import { sceneDimsManager } from './scene-dims-manager';
 *
 * sceneDimsManager.initFromScene(scene);
 * sceneDimsManager.addListener(() => updatePointsSlice());
 * sceneDimsManager.setDimensionValue(0, 5.2);
 * ```
 */
let _sceneDimsManagerInstance: SceneDimsManager | undefined;

export const sceneDimsManager: SceneDimsManager = new Proxy({} as SceneDimsManager, {
  get(_target, prop, _receiver) {
    _sceneDimsManagerInstance ??= new SceneDimsManager();
    const value = Reflect.get(_sceneDimsManagerInstance, prop, _sceneDimsManagerInstance);
    return typeof value === 'function' ? value.bind(_sceneDimsManagerInstance) : value;
  },
  set(_target, prop, value, _receiver) {
    _sceneDimsManagerInstance ??= new SceneDimsManager();
    return Reflect.set(_sceneDimsManagerInstance, prop, value, _sceneDimsManagerInstance);
  },
  has(_target, prop) {
    _sceneDimsManagerInstance ??= new SceneDimsManager();
    return prop in _sceneDimsManagerInstance;
  },
});

/**
 * Discard the current singleton so the next access constructs a fresh
 * instance. Intended for tests; safe to leave un-called in production.
 */
export const __resetSceneDimsManagerForTests = (): void => {
  _sceneDimsManagerInstance = undefined;
};
