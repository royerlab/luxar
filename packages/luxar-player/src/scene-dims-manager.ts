import { SimpleDims, DimensionMetadata } from './types/dims';
import * as THREE from 'three';

/**
 * Centralized dimension state manager ensuring consistency across all nD objects in the scene.
 * 
 * This singleton is a critical architectural component that solves the fundamental problem
 * of keeping multiple nD point clouds synchronized when navigating through dimensional space.
 * Without it, each point cloud could have its own slice position, leading to confusing
 * and inconsistent visualizations.
 * 
 * Core responsibilities:
 * - Initialize dimension metadata from scene-level configuration
 * - Maintain a single source of truth for current dimension positions
 * - Coordinate slider updates with point cloud re-slicing
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
  
  /** Observer callbacks that react to dimension changes */
  private listeners: Set<() => void> = new Set();

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
    
    // Validation: Ensure we found valid dimension metadata
    if (!sceneDimensions?.dimensions) {
      console.error('No scene dimensions found in scene or its children');
      return false;
    }
    
    // Step 3: Parse and normalize dimension metadata
    const metadata: DimensionMetadata[] = sceneDimensions.dimensions.map((dim: any) => ({
      name: dim.name,
      unit: dim.unit,
      scale: dim.scale || 1.0,
      range: dim.range ? [dim.range[0], dim.range[1]] : undefined,
      display: dim.display,
      discrete: dim.discrete || false,
      step: dim.step || 1.0,
    }));
    
    const ndim = metadata.length;
    
    // Step 4: Establish dimension ranges for navigation bounds
    this.dimensionRanges = metadata.map((meta) => {
      if (meta.range) {
        return meta.range as [number, number];
      }
      // Fallback: provide unit range if no bounds specified
      return [0, 1];
    });
    
    // Step 5: Initialize dimension positions
    // Critical decision: non-displayed dimensions start at minimum for predictability
    const currentStep = new Array(ndim).fill(0);
    
    for (let i = 0; i < ndim; i++) {
      if (metadata[i].display !== true) {
        // Non-displayed dimensions start at minimum bound
        currentStep[i] = this.dimensionRanges[i][0];
      }
      // Displayed dimensions start at 0 (camera will determine actual position)
    }
    
    // Step 6: Identify which dimensions should be displayed in 3D scene
    const displayed: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (metadata[i].display === true && displayed.length < 3) {
        displayed.push(i);
      }
    }
    
    // Step 7: Create the shared dimension state object
    this.dims = {
      ndim,
      currentStep,
      displayed,
      metadata
    };
    
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

    // Apply range constraints to prevent navigation beyond data bounds
    if (this.dimensionRanges) {
      const [min, max] = this.dimensionRanges[dimIndex];
      value = Math.max(min, Math.min(max, value));
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
   * Registers a callback to be invoked whenever dimension state changes.
   * 
   * This implements the observer pattern, allowing UI components and
   * visualization objects to react automatically to navigation events.
   * Typical subscribers include sliders, point clouds, and status displays.
   * 
   * @param callback - Function to call when dimensions change
   */
  addListener(callback: () => void): void {
    this.listeners.add(callback);
  }

  /**
   * Unregisters a dimension change callback.
   * 
   * Important for preventing memory leaks when components are destroyed.
   * 
   * @param callback - Previously registered callback function
   */
  removeListener(callback: () => void): void {
    this.listeners.delete(callback);
  }

  /**
   * Triggers all registered observer callbacks.
   * 
   * This is called internally whenever dimension state changes,
   * propagating updates throughout the reactive system.
   * 
   * @private
   */
  private notifyListeners(): void {
    this.listeners.forEach((callback) => callback());
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
 * Global singleton instance of the scene dimension manager.
 * 
 * This singleton ensures that all components in the application share
 * the same dimensional coordinate system. Import and use this instance
 * rather than creating new SceneDimsManager instances.
 * 
 * @example
 * ```typescript
 * import { sceneDimsManager } from './scene-dims-manager';
 * 
 * // Initialize from loaded scene
 * sceneDimsManager.initFromScene(scene);
 * 
 * // Register for dimension changes
 * sceneDimsManager.addListener(() => {
 *   console.log('Dimensions changed!');
 *   updatePointCloudSlice();
 * });
 * 
 * // Navigate through time dimension
 * sceneDimsManager.setDimensionValue(0, 5.2);
 * ```
 */
export const sceneDimsManager = new SceneDimsManager();
