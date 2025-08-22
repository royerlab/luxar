/**
 * Zarr-based nD point cloud data loader with advanced slicing capabilities.
 *
 * This module provides the core data loading infrastructure for Luxar's nD visualization
 * system. It handles loading point cloud data from Zarr stores, processes dimension
 * metadata, and performs initial slicing operations for nD datasets.
 *
 * Key capabilities:
 * - Hierarchical scene loading from Zarr group structures
 * - nD point cloud data loading with automatic slicing
 * - Scene-level dimension metadata extraction and propagation
 * - Radius-based hypersphere slicing for smooth navigation
 * - Optional attribute loading (colors, radii, sharpness)
 * - GPU-optimized data format conversion
 * - Rendering parameter inheritance in nested scenes
 *
 * Data pipeline:
 * 1. Load Zarr store with consolidated metadata
 * 2. Extract scene-level dimension configuration
 * 3. Enumerate and load point cloud groups hierarchically
 * 4. Perform initial nD slicing based on dimension state
 * 5. Convert data formats for GPU compatibility
 * 6. Create THREE.js geometry with appropriate materials
 *
 * Performance considerations:
 * - Uses streaming Zarr loading for large datasets
 * - Minimizes memory usage through on-demand slicing
 * - Batch-processes attribute arrays for efficiency
 * - Provides fallbacks for missing optional attributes
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import * as THREE from 'three';
import { materialManager, BlendingMode } from '../rendering/material-manager';
import { SimpleDims, DimensionMetadata } from '../types/dims';
import {
  slicePoints,
  extractDisplayDimensions,
  sliceColorsFloat32,
  sliceScalarAttribute,
  computeEffectiveRadii,
} from '../utils/slicing';
import { LazyDataManager } from './lazy-data-manager';
import { normalizeZarrPath, inheritRenderingAttributes as inheritAttrs } from './zarr-loader-utils';

/* ------------------------------------------------------------------ utils */

/**
 * Normalizes a path string to a valid URL for Zarr store access.
 * Delegates to the extracted utility function.
 *
 * @param path - Local path or full URL to Zarr store
 * @returns Normalized URL with trailing slash
 */
function toURL(path: string) {
  return normalizeZarrPath(path);
}

/**
 * Implements rendering attribute inheritance in hierarchical scenes.
 * Delegates to the extracted utility function.
 *
 * @param attrs - Current group's attributes
 * @param parentAttrs - Parent group's attributes for inheritance
 * @returns Merged attributes with inheritance applied
 */
function inheritRenderingAttributes(
  attrs: ZarrGroupAttrs,
  parentAttrs?: ZarrGroupAttrs
): ZarrGroupAttrs {
  return inheritAttrs(attrs as any, parentAttrs as any) as ZarrGroupAttrs;
}

/** Internal record for tracking THREE.js objects and their Zarr metadata */
type ObjRecord = { obj: THREE.Object3D; path: string; attrs?: ZarrGroupAttrs };

/**
 * Zarr group attributes defining rendering and dimensional properties.
 *
 * These attributes are stored in Zarr group metadata and control how
 * point clouds are rendered and how their dimensions are interpreted.
 *
 * @interface ZarrGroupAttrs
 */
interface ZarrGroupAttrs {
  /** Object type identifier (e.g., 'points') */
  type?: string;

  /** 4x4 transformation matrix as 16-element array */
  transform?: number[];

  /** Metadata describing each dimension's properties */
  dimension_metadata?: DimensionMetadata[];

  /** Total number of points in the dataset */
  num_points?: number;

  /** Rendering opacity (0.0 - 1.0) */
  opacity?: number;

  /** Gamma correction factor for color */
  gamma?: number;

  /** Blending mode for compositing */
  blending_mode?: BlendingMode;

  /** Scene-level dimension configuration */
  scene_dimensions?: any;

  /** List of dimension names that should be auto-broadcasted */
  broadcast_dims?: string[];
}

/* ------------------------------------------------------------------ main */

// Global lazy data manager instance (shared across all point clouds)
let globalLazyManager: LazyDataManager | null = null;

/**
 * Get or create the global lazy data manager
 */
function getLazyManager(): LazyDataManager {
  if (!globalLazyManager) {
    globalLazyManager = new LazyDataManager({
      // maxMemoryMB is auto-detected now, no need to hardcode
      preloadRadius: 1,
      debug: false, // Can be enabled via config later
    });
    // Expose globally for monitoring
    (window as any).__luxarLazyManager = globalLazyManager;

    // Set up event callback for monitoring panel if it exists
    const monitor = (window as any).__luxarLazyMonitor;
    if (monitor) {
      globalLazyManager.setEventCallback((type, message, details) => {
        monitor.logEvent(type, message, details);
      });
    }
  }
  return globalLazyManager;
}

/**
 * Loads an nD scene from a Zarr store, creating a complete THREE.js scene graph.
 *
 * This is the main entry point for loading Luxar scene data. It handles the complete
 * pipeline from Zarr store access to THREE.js scene construction, including:
 *
 * - Scene-level dimension metadata extraction and initialization
 * - Hierarchical loading of nested groups and point clouds
 * - Automatic nD slicing for high-dimensional datasets
 * - Rendering attribute inheritance through the scene hierarchy
 * - GPU-optimized geometry creation with appropriate materials
 *
 * The loading process is designed to handle large scientific datasets efficiently
 * while providing immediate visual feedback and smooth navigation capabilities.
 *
 * Scene structure:
 * - Root group contains scene-level dimension metadata
 * - Child groups can contain point clouds or nested groups
 * - Each point cloud can have its own transformation and rendering properties
 * - Dimension metadata is propagated down the hierarchy for consistency
 *
 * @param src - URL or path to the Zarr store containing the scene data
 * @returns Promise resolving to a THREE.Group containing the complete scene
 */
export async function loadScene(src: string): Promise<THREE.Group> {
  try {
    // Phase 1: Initialize Zarr store with optimized metadata access
    const rawStore = new zarr.FetchStore(toURL(src));
    const store = await zarr.tryWithConsolidated(rawStore); // Use consolidated metadata when available

    console.log('[🔍] [Luxar] Store type:', store.constructor.name);
    console.log('[🔍] [Luxar] Has contents method:', typeof store.contents === 'function');

    // Phase 2: Discover scene structure
    // Check if store has contents method (consolidated) or use fallback
    let listing: any[];
    if (typeof store.contents === 'function') {
      listing = await store.contents(); // Enumerate all groups and arrays
      console.log(`[✓] [Luxar] Using consolidated metadata, found ${listing.length} items`);

      // Debug: Log first few items
      const groups = listing.filter((e: any) => e.kind === 'group');
      const arrays = listing.filter((e: any) => e.kind === 'array');
      console.log(`[🔍] [Luxar] Found ${groups.length} groups and ${arrays.length} arrays`);
      console.log(
        '[🔍] [Luxar] Groups:',
        groups.slice(0, 20).map((g: any) => g.path)
      );
    } else {
      // Fallback: manually enumerate by walking the store recursively
      console.warn(
        '[⚠️] [Luxar] Store does not have contents() method, using fallback enumeration'
      );
      listing = [];

      // Use a simpler approach: manually check known paths
      // Since we can't reliably enumerate children, we'll probe for common patterns
      const checkPath = async (path: string, kind: 'group' | 'array') => {
        try {
          const loc = path === '' ? zarr.root(store) : zarr.root(store).resolve(path);
          await zarr.open(loc, { kind });
          const fullPath = path === '' ? '/' : `/${path}`;
          listing.push({ path: fullPath, kind });
          return true;
        } catch {
          return false;
        }
      };

      // Add root
      await checkPath('', 'group');

      // Try to enumerate root-level groups by checking for .zgroup files
      // This is a workaround since we can't reliably list directory contents
      const rootGroup = await zarr.open(zarr.root(store), { kind: 'group' });

      // Check for common group patterns at root level
      // We'll try to detect groups by attempting to open them
      const potentialGroups: string[] = [];

      // Try to get a listing from the store itself if possible
      if (typeof (store as any).listDir === 'function') {
        try {
          const items = await (store as any).listDir('/');
          for (const item of items) {
            if (!item.startsWith('.') && !item.endsWith('/')) {
              potentialGroups.push(item);
            }
          }
        } catch {
          // listDir not available or failed
        }
      }

      // If we couldn't get a listing, try to read the root group's keys
      // Note: This gives us the Python zarr group's dictionary keys, not child groups
      // But we can try each key to see if it's a subgroup
      if (potentialGroups.length === 0) {
        // Try to access the store's keys directly
        // Different stores may expose this differently
        const keys = Object.keys(rootGroup);
        for (const key of keys) {
          if (!key.startsWith('_') && !key.startsWith('.') && key !== 'attrs' && key !== 'zarr') {
            potentialGroups.push(key);
          }
        }
      }

      // Check each potential group
      for (const name of potentialGroups) {
        const isGroup = await checkPath(name, 'group');
        if (isGroup) {
          // For each group, also check for standard arrays
          const arrays = ['positions', 'colors', 'radii', 'sharpness'];
          for (const arrName of arrays) {
            await checkPath(`${name}/${arrName}`, 'array');
          }
        }
      }

      if (listing.length === 0) {
        console.error('[❌] [Luxar] No groups or arrays found in store');
      } else {
        console.log(`[✓] [Luxar] Found ${listing.length} groups/arrays in store`);
      }
    }

    const rootLoc = zarr.root(store);
    const rootThree = new THREE.Group();

    // Phase 3: Load scene-level configuration and dimension metadata
    const rootGroup = await zarr.open(rootLoc, { kind: 'group' });
    const sceneAttrs = rootGroup.attrs as ZarrGroupAttrs;

    const lookup = new Map<string, ObjRecord>([
      ['/', { obj: rootThree, path: '/', attrs: sceneAttrs }],
    ]);

    // Phase 4: Initialize scene-level dimension system
    let sceneDims: SimpleDims | undefined;
    if (sceneAttrs?.scene_dimensions) {
      // Store raw metadata in THREE.js userData for scene manager access
      rootThree.userData.sceneDimensions = sceneAttrs.scene_dimensions;

      // Parse and normalize dimension metadata
      const metadata = sceneAttrs.scene_dimensions.dimensions.map((dim: any) => ({
        name: dim.name,
        unit: dim.unit,
        scale: dim.scale || 1.0,
        range: dim.range ? [dim.range[0], dim.range[1]] : undefined,
        display: dim.display,
        discrete: dim.discrete || false,
        step: dim.step || 1.0,
      }));

      const ndim = metadata.length;
      const displayed: number[] = [];

      // Identify which dimensions should be displayed in 3D
      for (let i = 0; i < ndim; i++) {
        if (metadata[i].display === true && displayed.length < 3) {
          displayed.push(i);
        }
      }

      // Initialize positions: non-displayed dims start at minimum for predictable slicing
      const currentStep = new Array(ndim).fill(0);
      for (let i = 0; i < ndim; i++) {
        if (!displayed.includes(i) && metadata[i].range) {
          currentStep[i] = metadata[i].range[0];
        }
      }

      // Create dimension state object for initial slicing
      sceneDims = {
        ndim,
        currentStep,
        displayed,
        metadata,
      };
    }

    /* 3. create Three.js objects in path-depth order */
    const groups = listing
      .filter((e: { kind: string; path: string }) => e.kind === 'group' && e.path !== '/')
      .sort(
        (a: { path: string }, b: { path: string }) =>
          a.path.split('/').length - b.path.split('/').length
      );

    for (const entry of groups) {
      try {
        const loc = rootLoc.resolve(entry.path.slice(1)); // drop leading "/"
        const grp = await zarr.open(loc, { kind: 'group' });
        let attrs = grp.attrs as ZarrGroupAttrs;

        // Get parent attributes for inheritance
        const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
        const parentRecord = lookup.get(parentPath);
        const parentAttrs = parentRecord?.attrs;

        // Debug logging
        console.log(
          `Loading ${entry.path}, parent: ${parentPath}, has transform: ${!!attrs?.transform}`
        );

        // Inherit rendering attributes from parent
        attrs = inheritRenderingAttributes(attrs, parentAttrs);

        // Pass scene dimensions down through attrs
        if (rootThree.userData.sceneDimensions && !attrs.scene_dimensions) {
          attrs.scene_dimensions = rootThree.userData.sceneDimensions;
        }

        /* build renderable */
        let obj: THREE.Object3D;
        if (attrs?.type === 'points') {
          obj = await buildPoints(loc, attrs, sceneDims);
        } else {
          obj = new THREE.Group();
          // Store rendering attrs on groups for child inheritance
          obj.userData.opacity = attrs.opacity;
          obj.userData.gamma = attrs.gamma;
          obj.userData.blendingMode = attrs.blending_mode;
        }

        /* attach to parent in Three.js graph FIRST */
        const parent = lookup.get(parentPath);
        if (!parent) {
          console.error(
            `[❌] [Luxar] Parent not found for ${entry.path}, parent path: ${parentPath}`
          );
          continue;
        }
        parent.obj.add(obj);
        console.log(
          `[✓] [Luxar] Added ${entry.path} to parent ${parentPath}, obj type: ${obj.type}, children count: ${obj.children?.length ?? 0}`
        );

        // Apply transform AFTER adding to parent
        if (Array.isArray(attrs?.transform) && attrs.transform.length === 16) {
          const matrix = new THREE.Matrix4().fromArray(attrs.transform);
          // Use matrix.decompose to set position, rotation, and scale
          // This preserves the hierarchical transform chain
          const position = new THREE.Vector3();
          const quaternion = new THREE.Quaternion();
          const scale = new THREE.Vector3();
          matrix.decompose(position, quaternion, scale);

          console.log(
            `Setting transform for ${entry.path}: position=(${position.x}, ${position.y}, ${position.z})`
          );

          obj.position.copy(position);
          obj.quaternion.copy(quaternion);
          obj.scale.copy(scale);
          // Don't call updateMatrix() - let THREE.js handle it automatically
        }
        lookup.set(entry.path, { obj, path: entry.path, attrs });
      } catch (error) {
        console.error(`Failed to load group ${entry.path}:`, error);
        // Continue loading other groups instead of failing completely
      }
    }

    // Debug: Check final scene structure
    console.log('[🎯] [Luxar] Final scene structure:');
    console.log(`  Root has ${rootThree.children?.length ?? 0} children`);
    rootThree.children?.forEach((child, i) => {
      if (child instanceof THREE.Points) {
        const points = child as THREE.Points;
        const posCount = points.geometry.attributes.position?.count || 0;
        console.log(`  Child ${i}: Points with ${posCount} points`);
      } else if (child instanceof THREE.Group) {
        console.log(`  Child ${i}: Group with ${child.children?.length ?? 0} children`);
      } else {
        console.log(`  Child ${i}: ${child.type}`);
      }
    });

    return rootThree;
  } catch (error) {
    console.error('Failed to load scene from Zarr store:', error);
    throw new Error(
      `Unable to load scene from ${src}: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/* ---------------------------------------------------------------- geometry */

/**
 * Constructs a THREE.js Points object from nD point cloud data in a Zarr group.
 *
 * This function is the heart of the nD visualization system, responsible for:
 * - Loading nD position data and determining dimensionality
 * - Performing initial radius-based hypersphere slicing for nD datasets
 * - Loading and processing optional attributes (colors, radii, sharpness)
 * - Converting data formats for GPU compatibility
 * - Creating optimized THREE.js geometry with appropriate materials
 * - Handling edge cases like empty slices and missing attributes
 *
 * The function automatically detects whether slicing is needed based on the
 * dimensionality of the data relative to the display dimensions, ensuring
 * optimal performance for both 3D and nD datasets.
 *
 * Data processing pipeline:
 * 1. Load position array and infer dimensionality
 * 2. Load optional radii array for hypersphere slicing
 * 3. Perform nD slicing if data exceeds 3D
 * 4. Load and slice optional attributes (colors, sharpness)
 * 5. Compute effective radii for sliced hyperspheres
 * 6. Handle HDR colors (float32 format, values can exceed 1.0)
 * 7. Create THREE.js geometry with all attributes
 * 8. Apply materials and store metadata for runtime updates
 *
 * @param loc - Zarr location containing the point cloud arrays
 * @param attrs - Group attributes with rendering and metadata properties
 * @param sceneDims - Scene-level dimension state for consistent slicing
 * @returns Promise resolving to a THREE.Points object ready for rendering
 */
async function buildPoints(
  loc: zarr.Location<zarr.Readable>,
  attrs: ZarrGroupAttrs,
  sceneDims?: SimpleDims
): Promise<THREE.Points> {
  // Phase 1: Open position array to check size
  const posArr = await zarr.open(loc.resolve('positions'), { kind: 'array' });

  // Determine dataset size and check if we should use lazy loading
  const totalPoints = attrs.num_points || (posArr.shape[0] as number);
  const dimensions = posArr.shape.length === 2 ? posArr.shape[1] : 3;

  // Check for broadcast dimensions early
  const broadcastDims = attrs.broadcast_dims || [];
  const hasBroadcastDims = broadcastDims.length > 0;

  // Always use lazy loading machinery for consistency
  // This provides a unified code path and enables monitoring for all datasets
  const useLazyLoading = true;

  let posData: Float32Array;

  if (useLazyLoading) {
    console.log(
      `[🔄] [Luxar] Using lazy loading for dataset (${totalPoints.toLocaleString()} points in ${dimensions}D)`
    );

    if (hasBroadcastDims) {
      console.log(`[📡] [Luxar] Group has broadcast dimensions: ${broadcastDims.join(', ')}`);
    }

    // Use lazy loading for all datasets for consistency and monitoring
    const lazyManager = getLazyManager();

    // For nD data where points are organized by non-displayed dimension values,
    // we need to calculate which slice of points to load based on current dimension positions.
    // The data is assumed to be organized such that all points with the same
    // non-displayed dimension values are contiguous.

    // Identify all non-displayed dimensions (if we have dimension metadata)
    const nonDisplayedDims: number[] = [];
    if (sceneDims && dimensions > 3) {
      for (let d = 0; d < dimensions; d++) {
        if (!sceneDims.displayed.includes(d)) {
          nonDisplayedDims.push(d);
        }
      }
    }

    if (nonDisplayedDims.length > 0 && sceneDims) {
      // Calculate the total number of unique combinations for non-displayed dimensions
      let totalSlices = 1;
      const sliceSizes: number[] = [];

      for (const dimIdx of nonDisplayedDims) {
        const dimMeta = sceneDims.metadata?.[dimIdx];
        if (dimMeta && dimMeta.range) {
          // Calculate dimension size - be careful with the range calculation
          const minVal = dimMeta.range[0];
          const maxVal = dimMeta.range[1];
          const step = dimMeta.step || 1.0;

          // If discrete dimension, count the actual steps
          if (dimMeta.discrete) {
            const dimSize = Math.floor((maxVal - minVal) / step) + 1;
            sliceSizes.push(dimSize);
            totalSlices *= dimSize;
          } else {
            // For continuous dimensions, still use range but consider step
            const dimSize = Math.round((maxVal - minVal) / step) + 1;
            sliceSizes.push(dimSize);
            totalSlices *= dimSize;
          }

          console.log(
            `[📊] [Luxar] Dimension ${dimMeta.name || dimIdx}: range [${minVal}, ${maxVal}], step ${step}, size ${sliceSizes[sliceSizes.length - 1]}`
          );
        } else {
          // If no range info, assume dimension size of 1
          sliceSizes.push(1);
        }
      }

      // Calculate the linear index for the current combination of non-displayed dimensions
      let linearIndex = 0;
      let multiplier = 1;

      // Process dimensions in reverse order (like row-major indexing)
      for (let i = nonDisplayedDims.length - 1; i >= 0; i--) {
        const dimIdx = nonDisplayedDims[i];
        const dimMeta = sceneDims.metadata?.[dimIdx];
        const minVal = dimMeta?.range?.[0] || 0;
        const currentVal = Math.round(sceneDims.currentStep[dimIdx] - minVal);

        linearIndex += currentVal * multiplier;
        multiplier *= sliceSizes[i];
      }

      // Calculate point indices for this slice
      let startIdx: number;
      let endIdx: number;

      // Check if this group should use broadcasting
      const shouldBroadcast =
        hasBroadcastDims &&
        nonDisplayedDims.some((dimIdx) => {
          const dimMeta = sceneDims.metadata?.[dimIdx];
          return dimMeta && broadcastDims.includes(dimMeta.name || '');
        });

      if (shouldBroadcast) {
        // For explicitly broadcast groups, always load all points
        console.log(`[📡] [Luxar] Using broadcast mode - loading all ${totalPoints} points`);
        startIdx = 0;
        endIdx = totalPoints;
      } else {
        // For groups with full coverage, load the appropriate slice
        const pointsPerSlice = Math.round(totalPoints / totalSlices);

        // Bounds check: ensure linearIndex is valid
        if (linearIndex < 0 || linearIndex >= totalSlices) {
          console.warn(
            `[⚠️] [Luxar] Invalid linear index ${linearIndex} (total slices: ${totalSlices}). Clamping to valid range.`
          );
          linearIndex = Math.max(0, Math.min(totalSlices - 1, linearIndex));
        }

        startIdx = linearIndex * pointsPerSlice;
        endIdx = Math.min((linearIndex + 1) * pointsPerSlice, totalPoints);
      }

      // Store these for use when loading auxiliary arrays
      (loc as any)._sliceStartIdx = startIdx;
      (loc as any)._sliceEndIdx = endIdx;
      (loc as any)._shouldBroadcast = shouldBroadcast;

      // Create slice spec for these points
      const sliceSpec: (zarr.Slice | null)[] = [
        zarr.slice(startIdx, endIdx), // Points for this slice
        null, // All coordinate dimensions
      ];

      console.log(
        `[📥] [Luxar] Loading slice ${linearIndex}/${totalSlices} (points ${startIdx}-${endIdx})`
      );
      console.log(
        `[📊] [Luxar] Non-displayed dims: ${nonDisplayedDims
          .map(
            (d) =>
              `${sceneDims.metadata?.[d]?.name || `dim${d}`}=${sceneDims.currentStep[d].toFixed(1)}`
          )
          .join(', ')}`
      );

      // Use LazyDataManager to load the slice
      // Include object path in array name to prevent cache collisions
      const positionsPath = loc.path ? `${loc.path}/positions` : 'positions';
      posData = (await lazyManager.loadSlice(posArr, positionsPath, sliceSpec)) as Float32Array;

      // Store references for future updates
      if (!globalLazyManager) globalLazyManager = lazyManager;

      // Store total slices metadata in the lazy manager for the monitor
      lazyManager.setDatasetMetadata('totalSlices', totalSlices);
      lazyManager.setDatasetMetadata('nonDisplayedDims', nonDisplayedDims);

      console.log(
        `[📊] [Luxar] Dataset has ${totalSlices} total slices (${nonDisplayedDims.length} non-displayed dims)`
      );

      // Reconnect monitoring callback if monitor exists
      const monitor = (window as any).__luxarLazyMonitor;
      if (monitor && lazyManager) {
        lazyManager.setEventCallback((type, message, details) => {
          monitor.logEvent(type, message, details);
        });
      }

      // Store the array reference in userData for dynamic updates
      (loc as any)._posArr = posArr;
      (loc as any)._lazyManager = lazyManager;
      (loc as any)._totalSlices = totalSlices;
      (loc as any)._nonDisplayedDims = nonDisplayedDims;
    } else {
      // No non-displayed dimensions or regular 3D data - load everything through lazy manager
      console.log('[ℹ️] [Luxar] Loading full dataset through lazy manager for consistency');

      // Use lazy manager even for full dataset load - this enables monitoring
      const sliceSpec: (zarr.Slice | null)[] = [null, null]; // Load all data
      // Include object path in array name to prevent cache collisions
      const positionsPath = loc.path ? `${loc.path}/positions` : 'positions';
      posData = (await lazyManager.loadSlice(posArr, positionsPath, sliceSpec)) as Float32Array;

      // Store references for monitoring
      if (!globalLazyManager) globalLazyManager = lazyManager;

      // For 3D data, we have just one "slice" containing all data
      lazyManager.setDatasetMetadata('totalSlices', 1);
      lazyManager.setDatasetMetadata('nonDisplayedDims', []);

      // Connect monitoring callback if monitor exists
      const monitor = (window as any).__luxarLazyMonitor;
      if (monitor && lazyManager) {
        lazyManager.setEventCallback((type, message, details) => {
          monitor.logEvent(type, message, details);
        });
      }

      // Store references for consistency
      (loc as any)._posArr = posArr;
      (loc as any)._lazyManager = lazyManager;
    }
  } else {
    // Use traditional full loading for small datasets
    posData = (await get(posArr)).data as Float32Array;
  }

  // Determine dataset size and actual loaded points
  let numPoints: number;
  let actualLoadedPoints: number;

  if (useLazyLoading) {
    // When lazy loading, we may load a subset of points
    numPoints = attrs.num_points || totalPoints; // Total points in dataset
    actualLoadedPoints = posData.length / dimensions; // Points actually loaded
  } else {
    // Traditional loading - all points are loaded
    numPoints = attrs.num_points || posData.length / 3;
    actualLoadedPoints = numPoints;
  }

  // Infer dimensionality from data structure
  const ndim = posData.length / actualLoadedPoints;
  if (!Number.isInteger(ndim)) {
    throw new Error(
      `Invalid positions array: ${posData.length} elements for ${actualLoadedPoints} points`
    );
  }

  // Use scene-level dimensions for consistency, or create default for standalone data
  const dims = sceneDims || {
    ndim,
    currentStep: new Array(ndim).fill(0),
    displayed:
      ndim <= 3 ? Array.from({ length: ndim }, (_, i) => i) : [ndim - 3, ndim - 2, ndim - 1],
    metadata: undefined,
  };

  // Phase 2: Load radii for hypersphere slicing (critical for nD navigation)
  let radiiData: Float32Array | undefined;
  try {
    const radiiArr = await zarr.open(loc.resolve('radii'), { kind: 'array' });
    console.log(`[📐] [Luxar] Loading radii for ${attrs.type === 'points' ? loc.path : 'unknown'}`);
    if (useLazyLoading && (loc as any)._lazyManager) {
      // Load same slice of radii as positions
      // Use the same slice indices that were calculated for positions
      const nonDisplayedDims = (loc as any)._nonDisplayedDims;
      const totalSlices = (loc as any)._totalSlices;

      if (nonDisplayedDims && nonDisplayedDims.length > 0 && totalSlices && sceneDims) {
        // Use the same slice indices that were calculated for positions
        const startIdx = (loc as any)._sliceStartIdx || 0;
        const endIdx = (loc as any)._sliceEndIdx || totalPoints;
        const shouldBroadcast = (loc as any)._shouldBroadcast || false;

        if (shouldBroadcast) {
          console.log(
            `[📏] [Luxar] Loading all radii for broadcast group: ${loc.path} (${startIdx}-${endIdx})`
          );
        }

        const sliceSpec = [zarr.slice(startIdx, endIdx)];
        // Include object path in array name to prevent cache collisions
        const radiiPath = loc.path ? `${loc.path}/radii` : 'radii';
        radiiData = await (loc as any)._lazyManager.loadSlice(radiiArr, radiiPath, sliceSpec);
      } else {
        // For 3D data or when no slicing is needed, load all radii through lazy manager
        const sliceSpec: (zarr.Slice | null)[] = [null];
        // Include object path in array name to prevent cache collisions
        const radiiPath = loc.path ? `${loc.path}/radii` : 'radii';
        radiiData = await (loc as any)._lazyManager.loadSlice(radiiArr, radiiPath, sliceSpec);
      }
    } else {
      radiiData = (await get(radiiArr)).data as Float32Array;
    }
  } catch (error) {
    console.debug('No radii array found:', error);
    // Radii are optional but highly recommended for nD datasets
  }

  // Phase 3: Perform nD slicing if needed
  let pos: Float32Array;
  let visibleIndices: Uint32Array | null = null;

  if (dims.ndim > 3 && !useLazyLoading) {
    // Apply radius-based hypersphere slicing for nD visualization
    // (Only for non-lazy loading; lazy loading already provides the slice)
    visibleIndices = slicePoints(posData, dims, actualLoadedPoints, radiiData);
    pos = extractDisplayDimensions(posData, visibleIndices, dims);

    // Handle edge case: empty slice (user navigated to region with no data)
    if (visibleIndices.length === 0) {
      console.warn('No points visible at current slice position!');
      // Provide dummy geometry to prevent GPU errors
      pos = new Float32Array([0, 0, 0]);
      visibleIndices = new Uint32Array([0]);
    }
  } else if (dims.ndim > 3 && useLazyLoading && sceneDims) {
    // With lazy loading and nD data, we have the exact slice we need
    // Just extract the displayed dimensions (first 3)
    const numLoadedPoints = posData.length / dims.ndim;
    pos = new Float32Array(numLoadedPoints * 3);

    for (let i = 0; i < numLoadedPoints; i++) {
      for (let d = 0; d < 3; d++) {
        if (d < dims.displayed.length) {
          const dimIdx = dims.displayed[d];
          pos[i * 3 + d] = posData[i * dims.ndim + dimIdx];
        } else {
          pos[i * 3 + d] = 0; // Pad with zeros if less than 3 displayed dims
        }
      }
    }
  } else {
    // 3D or lower dimensional data requires no slicing
    pos = posData;
  }

  // Load optional attributes - HDR colors are now float32
  let colData: Float32Array | undefined;
  let col: Float32Array | undefined;
  try {
    const colArr = await zarr.open(loc.resolve('colors'), { kind: 'array' });
    let rawData: any;

    if (useLazyLoading && (loc as any)._lazyManager) {
      // Load same slice of colors as positions
      const nonDisplayedDims = (loc as any)._nonDisplayedDims;
      const totalSlices = (loc as any)._totalSlices;

      if (nonDisplayedDims && nonDisplayedDims.length > 0 && totalSlices && sceneDims) {
        // Use the same slice indices that were calculated for positions
        const startIdx = (loc as any)._sliceStartIdx || 0;
        const endIdx = (loc as any)._sliceEndIdx || totalPoints;
        const shouldBroadcast = (loc as any)._shouldBroadcast || false;

        if (shouldBroadcast) {
          console.log(
            `[🎨] [Luxar] Loading all colors for broadcast group: ${loc.path} (${startIdx}-${endIdx})`
          );
        }

        const sliceSpec = [zarr.slice(startIdx, endIdx), null]; // All color channels
        // Include object path in array name to prevent cache collisions
        const colorsPath = loc.path ? `${loc.path}/colors` : 'colors';
        rawData = await (loc as any)._lazyManager.loadSlice(colArr, colorsPath, sliceSpec);
      } else {
        // For 3D data or when no slicing is needed, load all colors through lazy manager
        const sliceSpec: (zarr.Slice | null)[] = [null, null]; // All points, all channels
        // Include object path in array name to prevent cache collisions
        const colorsPath = loc.path ? `${loc.path}/colors` : 'colors';
        rawData = await (loc as any)._lazyManager.loadSlice(colArr, colorsPath, sliceSpec);
      }
    } else {
      rawData = (await get(colArr)).data;
    }

    // Handle both legacy uint8 and new HDR float32 formats
    if (rawData instanceof Uint8Array) {
      // Legacy format: convert uint8 to float32
      console.log('Converting legacy uint8 colors to HDR float32');
      const floatData = new Float32Array(rawData.length);
      for (let i = 0; i < rawData.length; i++) {
        floatData[i] = rawData[i] / 255.0;
      }
      colData = floatData;
    } else {
      // New HDR format: already float32
      colData = rawData as Float32Array;
    }

    col = visibleIndices ? sliceColorsFloat32(colData, visibleIndices) || undefined : colData;
  } catch (error) {
    console.debug('Optional array not found:', error);
    /* optional */
  }

  // Process radii for visible points
  let radii: Float32Array | undefined;
  if (radiiData) {
    if (visibleIndices && dims.ndim > 3) {
      // Compute effective radii based on slice intersection
      radii = computeEffectiveRadii(posData, visibleIndices, dims, radiiData);
    } else {
      radii = radiiData;
    }
  }

  let sharpnessData: Float32Array | undefined;
  let sharpness: Float32Array | undefined;
  try {
    const sharpnessArr = await zarr.open(loc.resolve('sharpness'), { kind: 'array' });

    if (useLazyLoading && (loc as any)._lazyManager) {
      // Load same slice of sharpness as positions
      const nonDisplayedDims = (loc as any)._nonDisplayedDims;
      const totalSlices = (loc as any)._totalSlices;

      if (nonDisplayedDims && nonDisplayedDims.length > 0 && totalSlices && sceneDims) {
        // Use the same slice indices that were calculated for positions
        const startIdx = (loc as any)._sliceStartIdx || 0;
        const endIdx = (loc as any)._sliceEndIdx || totalPoints;
        const shouldBroadcast = (loc as any)._shouldBroadcast || false;

        if (shouldBroadcast) {
          console.log(
            `[✨] [Luxar] Loading all sharpness for broadcast group: ${loc.path} (${startIdx}-${endIdx})`
          );
        }

        const sliceSpec = [zarr.slice(startIdx, endIdx)];
        // Include object path in array name to prevent cache collisions
        const sharpnessPath = loc.path ? `${loc.path}/sharpness` : 'sharpness';
        sharpnessData = await (loc as any)._lazyManager.loadSlice(
          sharpnessArr,
          sharpnessPath,
          sliceSpec
        );
      } else {
        // For 3D data or when no slicing is needed, load all sharpness through lazy manager
        const sliceSpec: (zarr.Slice | null)[] = [null];
        // Include object path in array name to prevent cache collisions
        const sharpnessPath = loc.path ? `${loc.path}/sharpness` : 'sharpness';
        sharpnessData = await (loc as any)._lazyManager.loadSlice(
          sharpnessArr,
          sharpnessPath,
          sliceSpec
        );
      }
    } else {
      sharpnessData = (await get(sharpnessArr)).data as Float32Array;
    }

    const slicedSharpness = visibleIndices
      ? sliceScalarAttribute(sharpnessData || null, visibleIndices)
      : sharpnessData;
    sharpness = slicedSharpness || undefined;
  } catch (error) {
    console.debug('Optional array not found:', error);
    /* optional */
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  // Debug: Check position data
  const numPointsInGeom = pos.length / 3;
  if (numPointsInGeom > 0) {
    const xValues = Array.from({ length: Math.min(100, numPointsInGeom) }, (_, i) => pos[i * 3]);
    const yValues = Array.from(
      { length: Math.min(100, numPointsInGeom) },
      (_, i) => pos[i * 3 + 1]
    );
    const zValues = Array.from(
      { length: Math.min(100, numPointsInGeom) },
      (_, i) => pos[i * 3 + 2]
    );
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const zMin = Math.min(...zValues);
    const zMax = Math.max(...zValues);
    console.log(`[📍] [Luxar] Position data for ${loc.path}:`);
    console.log(`  - Points: ${numPointsInGeom}`);
    console.log(`  - X range: [${xMin.toFixed(2)}, ${xMax.toFixed(2)}]`);
    console.log(`  - Y range: [${yMin.toFixed(2)}, ${yMax.toFixed(2)}]`);
    console.log(`  - Z range: [${zMin.toFixed(2)}, ${zMax.toFixed(2)}]`);
  }

  if (col) {
    // Colors are already in HDR float32 format - can exceed 1.0 for bright emission
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  } else {
    // Provide default white colors if no color data is available
    // This ensures the vertex shader always has color data to work with
    const numVertices = pos.length / 3;
    const defaultColors = new Float32Array(numVertices * 3);
    defaultColors.fill(1.0); // All white (RGB = 1.0, 1.0, 1.0)
    geom.setAttribute('color', new THREE.BufferAttribute(defaultColors, 3));
  }

  if (radii) {
    // Add radius attribute for custom shaders
    geom.setAttribute('radius', new THREE.BufferAttribute(radii, 1));
    const avgRadius =
      radii.slice(0, Math.min(100, radii.length)).reduce((a, b) => a + b, 0) /
      Math.min(100, radii.length);
    console.log(`[📏] [Luxar] Radii loaded for ${loc.path}:`);
    console.log(`  - Count: ${radii.length}`);
    console.log(`  - Average: ${avgRadius.toFixed(4)}`);
    console.log(
      `  - First 5: [${Array.from(radii.slice(0, 5))
        .map((r) => r.toFixed(4))
        .join(', ')}]`
    );
  } else {
    // Provide default radii if not specified
    const numVertices = pos.length / 3;
    const defaultRadii = new Float32Array(numVertices);
    defaultRadii.fill(1.0); // Default radius - increased for better visibility
    geom.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
    console.log(`[⚠️] [Luxar] No radii data for ${loc.path}, using default: 1.0`);
  }

  if (sharpness) {
    // Add sharpness attribute for custom shaders
    geom.setAttribute('sharpness', new THREE.BufferAttribute(sharpness, 1));
  } else {
    // Provide default sharpness if not specified
    const numVertices = pos.length / 3;
    const defaultSharpness = new Float32Array(numVertices);
    defaultSharpness.fill(2.0); // Default sharpness (quadratic falloff)
    geom.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
  }

  // Get rendering properties from attributes, with defaults
  const opacity = attrs.opacity ?? 1.0;
  const gamma = attrs.gamma ?? 1.0;
  const blendingMode = attrs.blending_mode ?? 'additive';

  // Get or create material from the manager
  const material = materialManager.getMaterial({
    blendingMode: blendingMode as BlendingMode,
    opacity,
    gamma,
  });

  const points = new THREE.Points(geom, material);

  // Debug: Check the created points object
  console.log(`[🎯] [Luxar] Created Points for ${loc.path}:`);
  console.log(`  - Point count: ${geom.attributes.position.count}`);
  console.log(`  - Has colors: ${!!geom.attributes.color}`);
  console.log(`  - Has radii: ${!!geom.attributes.radius}`);
  console.log(`  - Material blending: ${blendingMode}`);
  console.log(`  - Material opacity: ${opacity}`);
  console.log(`  - Visible: ${points.visible}`);

  // Apply render order from material
  if (material.userData.renderOrder !== undefined) {
    points.renderOrder = material.userData.renderOrder;
    console.log(`  - Render order: ${points.renderOrder}`);
  }

  // Store rendering properties in userData for runtime updates
  points.userData.opacity = opacity;
  points.userData.gamma = gamma;
  points.userData.blendingMode = blendingMode;

  // Store original data in userData for later access (but NOT dims!)
  if (useLazyLoading && (loc as any)._lazyManager) {
    // For lazy loading, store references for dynamic updates
    points.userData.isLazyLoaded = true;
    points.userData.lazyManager = (loc as any)._lazyManager;
    points.userData.positionsArray = (loc as any)._posArr;
    points.userData.totalPoints = totalPoints;
    points.userData.originalNumPoints = numPoints;

    // Store array references for reloading
    points.userData.zarrLocation = loc;
    points.userData.sceneDims = sceneDims;

    // Store broadcast_dims attribute if present
    if (broadcastDims.length > 0) {
      points.userData.broadcastDims = broadcastDims;
    }
  } else if (dims.ndim > 3) {
    // Traditional full data storage for nD data without lazy loading
    points.userData.originalNumPoints = numPoints;
    points.userData.originalPositions = posData;
    points.userData.originalColors = colData;
    points.userData.originalRadii = radiiData;
    points.userData.originalSharpness = sharpnessData;
    points.userData.visibleIndices = visibleIndices;
  }

  return points;
}
