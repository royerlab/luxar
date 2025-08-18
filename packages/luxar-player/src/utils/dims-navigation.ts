/**
 * Navigation utilities for smooth movement through nD space.
 *
 * This module provides the core functionality for user interaction with
 * high-dimensional datasets, including keyboard navigation, boundary handling,
 * and real-time point cloud updates. It bridges user input with the underlying
 * mathematical slicing operations.
 *
 * Key features:
 * - Adaptive step sizing based on data ranges
 * - Boundary wrapping for periodic dimensions
 * - Efficient GPU buffer updates during navigation
 * - Automatic camera centering and bounds management
 */

import { SimpleDims } from '../types/dims';
import {
  slicePoints,
  extractDisplayDimensions,
  sliceColorsFloat32,
  sliceScalarAttribute,
  computeEffectiveRadii,
} from './slicing';
import * as THREE from 'three';

/**
 * Configuration options for dimension navigation behavior.
 *
 * @interface NavigationOptions
 */
interface NavigationOptions {
  /** Fraction of dimension range to step (0.1 = 10% of range per step) */
  stepSize?: number;

  /** Whether to wrap around at dimension boundaries (useful for periodic data) */
  wrap?: boolean;

  /** Absolute step size in data units (overrides relative stepSize) */
  absoluteStep?: number;
}

/**
 * Steps through a non-displayed dimension by a calculated or specified amount.
 *
 * This function implements intelligent navigation that adapts to the data's natural
 * scale while respecting dimension boundaries and navigation constraints.
 *
 * Design decisions:
 * - Only non-displayed dimensions can be stepped (displayed dims are controlled by camera)
 * - Default step size is 10% of dimension range for intuitive navigation
 * - Boundary handling prevents navigation beyond data bounds
 * - Wrapping support enables navigation through periodic dimensions (e.g., angle, time)
 *
 * @param dims - Current dimension state to modify
 * @param dimIndex - Index of dimension to step through
 * @param direction - Direction to step: 1 for forward, -1 for backward
 * @param ranges - Min/max bounds for each dimension
 * @param options - Navigation behavior configuration
 * @returns True if dimension position changed, false otherwise
 */
export function stepDimension(
  dims: SimpleDims,
  dimIndex: number,
  direction: 1 | -1,
  ranges: Array<[number, number]>,
  options: NavigationOptions = {}
): boolean {
  const { stepSize = 0.1, wrap = false, absoluteStep } = options;

  // Validate dimension index
  if (dimIndex < 0 || dimIndex >= dims.ndim) {
    return false;
  }

  // Safety check: displayed dimensions are controlled by camera movement
  if (dims.displayed.includes(dimIndex)) {
    return false;
  }

  const [min, max] = ranges[dimIndex];
  const range = max - min;

  // Calculate step size: absolute takes precedence over relative
  const step = absoluteStep !== undefined ? absoluteStep : range * stepSize;

  const current = dims.currentStep[dimIndex];
  let newValue = current + direction * step;

  // Apply boundary conditions
  if (newValue > max) {
    newValue = wrap ? min : max; // Wrap to start or clamp to max
  } else if (newValue < min) {
    newValue = wrap ? max : min; // Wrap to end or clamp to min
  }

  // Update dimension position if it actually changed
  if (newValue !== current) {
    dims.currentStep[dimIndex] = newValue;
    return true; // Signal that re-slicing is needed
  }

  return false; // No change, no re-slicing needed
}

/**
 * Directly jumps to a specific fractional position within a dimension's range.
 *
 * This function enables precise positioning based on UI elements like sliders,
 * where the user specifies an exact location as a percentage of the total range.
 *
 * @param dims - Dimension state to modify
 * @param dimIndex - Index of dimension to position
 * @param fraction - Position as fraction of range (0.0 = min, 1.0 = max)
 * @param ranges - Min/max bounds for each dimension
 * @returns True if position changed, false otherwise
 */
export function jumpToDimension(
  dims: SimpleDims,
  dimIndex: number,
  fraction: number,
  ranges: Array<[number, number]>
): boolean {
  // Validate inputs and ensure only non-displayed dimensions can be jumped
  if (dimIndex < 0 || dimIndex >= dims.ndim || dims.displayed.includes(dimIndex)) {
    return false;
  }

  const [min, max] = ranges[dimIndex];
  // Clamp fraction to [0,1] and map to dimension range
  const newValue = min + (max - min) * Math.max(0, Math.min(1, fraction));

  if (newValue !== dims.currentStep[dimIndex]) {
    dims.currentStep[dimIndex] = newValue;
    return true;
  }

  return false;
}

/**
 * Identifies the first two non-displayed dimensions for keyboard navigation.
 *
 * This function establishes a consistent mapping between keyboard inputs and
 * dimensions, typically assigning arrow keys or WASD to the first two
 * non-displayed dimensions for intuitive navigation.
 *
 * Design rationale: Users need predictable keyboard controls that don't change
 * as they modify which dimensions are displayed.
 *
 * @param dims - Current dimension state
 * @returns Tuple of [primary, secondary] dimension indices (-1 if not available)
 */
export function getNavigableDimensions(dims: SimpleDims): [number, number] {
  const nonDisplayed = [];
  for (let i = 0; i < dims.ndim; i++) {
    if (!dims.displayed.includes(i)) {
      nonDisplayed.push(i);
    }
  }

  return [nonDisplayed[0] ?? -1, nonDisplayed[1] ?? -1];
}

/**
 * Updates the GPU point cloud geometry after a dimension navigation event.
 *
 * This function is the critical performance bottleneck that must run smoothly
 * during real-time navigation. It performs the complete pipeline from nD slicing
 * to GPU buffer updates, ensuring visual consistency and responsiveness.
 *
 * Pipeline stages:
 * 1. Slice nD data using radius-based hypersphere intersection
 * 2. Project visible points to 3D display coordinates
 * 3. Compute effective radii for sliced nD spheres
 * 4. Update GPU vertex attributes (position, color, radius, sharpness)
 * 5. Trigger GPU buffer updates and bounding volume recalculation
 *
 * Performance considerations:
 * - Minimizes GPU memory allocations by reusing buffers when possible
 * - Handles empty slices gracefully with dummy geometry
 * - Batches all GPU updates to minimize state changes
 *
 * Edge case handling:
 * - Empty slices display a single dummy point to prevent GPU errors
 * - Missing attributes get sensible defaults
 * - Color format conversion from uint8 to float32 for GPU
 *
 * @param points - THREE.js Points object to update
 * @param originalPositions - Full nD position data
 * @param originalColors - Per-point RGB colors (optional)
 * @param originalRadii - Per-point radii for hypersphere slicing (optional)
 * @param originalSharpness - Per-point sharpness values (optional)
 * @param dims - Current dimension state after navigation
 * @param numPoints - Total number of points in original dataset
 */
export function updatePointCloudSlice(
  points: THREE.Points,
  originalPositions: Float32Array,
  originalColors: Float32Array | undefined, // Changed to Float32Array for HDR
  originalRadii: Float32Array | undefined,
  originalSharpness: Float32Array | undefined,
  dims: SimpleDims,
  numPoints: number
): void {
  // Phase 1: Perform nD slicing with radius-based visibility
  const visibleIndices = slicePoints(originalPositions, dims, numPoints, originalRadii);
  const positions3D = extractDisplayDimensions(originalPositions, visibleIndices, dims);

  // Edge case: Handle empty slices (user navigated to region with no data)
  if (visibleIndices.length === 0) {
    console.warn('No points visible at current slice position!');
    // Render a single dummy point to prevent GPU geometry errors
    const geom = points.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3));
    geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array([1, 1, 1]), 3));
    geom.setAttribute('radius', new THREE.BufferAttribute(new Float32Array([0.1]), 1));
    geom.setAttribute('sharpness', new THREE.BufferAttribute(new Float32Array([2.0]), 1));
    return;
  }

  // Phase 2: Update GPU geometry with sliced data
  const geom = points.geometry;
  geom.setAttribute('position', new THREE.BufferAttribute(positions3D, 3));

  // Phase 3: Handle color attributes (HDR float32 format)
  if (originalColors) {
    const colors = sliceColorsFloat32(originalColors, visibleIndices);
    if (colors) {
      // Colors are already in float32 HDR format
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
  } else {
    // Fallback: default white coloring for all points
    const numVis = visibleIndices.length;
    const defaultColors = new Float32Array(numVis * 3);
    defaultColors.fill(1.0);
    geom.setAttribute('color', new THREE.BufferAttribute(defaultColors, 3));
  }

  // Phase 4: Compute effective radii using nD hypersphere intersection math
  if (originalRadii) {
    const effectiveRadii = computeEffectiveRadii(
      originalPositions,
      visibleIndices,
      dims,
      originalRadii
    );
    geom.setAttribute('radius', new THREE.BufferAttribute(effectiveRadii, 1));
  } else {
    // Fallback: uniform default radius
    const defaultRadii = new Float32Array(visibleIndices.length);
    defaultRadii.fill(0.1);
    geom.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
  }

  // Phase 5: Handle sharpness attributes
  if (originalSharpness) {
    const sharpness = sliceScalarAttribute(originalSharpness, visibleIndices);
    if (sharpness) {
      geom.setAttribute('sharpness', new THREE.BufferAttribute(sharpness, 1));
    }
  } else {
    // Fallback: moderate sharpness for all points
    const defaultSharpness = new Float32Array(visibleIndices.length);
    defaultSharpness.fill(2.0);
    geom.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
  }

  // Phase 6: Signal GPU to update buffers and recalculate bounds
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate = true;
  geom.attributes.radius.needsUpdate = true;
  geom.attributes.sharpness.needsUpdate = true;
  geom.computeBoundingSphere(); // Critical for frustum culling and camera bounds
}

/**
 * Updates a lazy-loaded point cloud by fetching new data from zarr.
 * This is more efficient than the traditional approach as it only loads
 * the data needed for the current slice.
 */
export async function updateLazyLoadedPointCloud(
  points: THREE.Points,
  dims: SimpleDims
): Promise<void> {
  const { lazyManager, positionsArray, zarrLocation, totalPoints } = points.userData;

  if (!lazyManager || !positionsArray || !zarrLocation) {
    console.warn('[⚠️] [Luxar] Missing lazy loading data in userData');
    console.log('userData:', points.userData);
    return;
  }

  // Dynamic import of zarr at runtime
  const zarr = await import('zarrita');
  type Slice = typeof zarr.slice extends (...args: any[]) => infer R ? R : never;

  // Identify all non-displayed dimensions
  const nonDisplayedDims: number[] = [];
  for (let d = 0; d < dims.ndim; d++) {
    if (!dims.displayed.includes(d)) {
      nonDisplayedDims.push(d);
    }
  }

  if (nonDisplayedDims.length === 0) {
    console.warn('[⚠️] [Luxar] No non-displayed dimensions found');
    return;
  }

  // Calculate the total number of unique combinations for non-displayed dimensions
  let totalSlices = 1;
  const sliceSizes: number[] = [];

  for (const dimIdx of nonDisplayedDims) {
    const dimMeta = dims.metadata?.[dimIdx];
    if (dimMeta && dimMeta.range) {
      const dimSize = Math.round(dimMeta.range[1] - dimMeta.range[0] + 1);
      sliceSizes.push(dimSize);
      totalSlices *= dimSize;
    } else {
      sliceSizes.push(1);
    }
  }

  // Calculate the linear index for the current combination of non-displayed dimensions
  let linearIndex = 0;
  let multiplier = 1;

  // Process dimensions in reverse order (like row-major indexing)
  for (let i = nonDisplayedDims.length - 1; i >= 0; i--) {
    const dimIdx = nonDisplayedDims[i];
    const dimMeta = dims.metadata?.[dimIdx];
    const minVal = dimMeta?.range?.[0] || 0;
    const currentVal = Math.round(dims.currentStep[dimIdx] - minVal);

    linearIndex += currentVal * multiplier;
    multiplier *= sliceSizes[i];
  }

  // Calculate point indices for this slice
  const pointsPerSlice = Math.round(totalPoints / totalSlices);
  const startIdx = linearIndex * pointsPerSlice;
  const endIdx = Math.min((linearIndex + 1) * pointsPerSlice, totalPoints);

  console.log(
    `[🔄] [Luxar] Loading slice ${linearIndex}/${totalSlices} (points ${startIdx}-${endIdx})`
  );
  console.log(
    `[📊] [Luxar] Non-displayed dims: ${nonDisplayedDims
      .map((d) => `${dims.metadata?.[d]?.name || `dim${d}`}=${dims.currentStep[d].toFixed(1)}`)
      .join(', ')}`
  );

  try {
    // Load positions for the new time frame
    const sliceSpec: (Slice | null)[] = [
      zarr.slice(startIdx, endIdx),
      null, // All dimensions
    ];
    const posData = (await lazyManager.loadSlice(
      positionsArray,
      'positions',
      sliceSpec
    )) as Float32Array;

    // Extract displayed dimensions
    const numLoadedPoints = posData.length / dims.ndim;
    const pos = new Float32Array(numLoadedPoints * 3);

    for (let i = 0; i < numLoadedPoints; i++) {
      for (let d = 0; d < 3; d++) {
        if (d < dims.displayed.length) {
          const dimIdx = dims.displayed[d];
          pos[i * 3 + d] = posData[i * dims.ndim + dimIdx];
        } else {
          pos[i * 3 + d] = 0;
        }
      }
    }

    // Load colors if available
    let col: Float32Array | undefined;
    try {
      const colArr = await zarr.open(zarrLocation.resolve('colors'), { kind: 'array' });
      const colSliceSpec: (Slice | null)[] = [zarr.slice(startIdx, endIdx), null];
      col = (await lazyManager.loadSlice(colArr, 'colors', colSliceSpec)) as Float32Array;
    } catch (error) {
      console.debug('[🎨] [Luxar] No colors array found:', error);
      // Colors are optional
    }

    // Load radii if available
    let radii: Float32Array | undefined;
    try {
      const radiiArr = await zarr.open(zarrLocation.resolve('radii'), { kind: 'array' });
      const radiiSliceSpec: (Slice | null)[] = [zarr.slice(startIdx, endIdx)];
      radii = (await lazyManager.loadSlice(radiiArr, 'radii', radiiSliceSpec)) as Float32Array;
    } catch (error) {
      console.debug('[📏] [Luxar] No radii array found:', error);
      // Radii are optional
    }

    // Load sharpness if available
    let sharpness: Float32Array | undefined;
    try {
      const sharpnessArr = await zarr.open(zarrLocation.resolve('sharpness'), { kind: 'array' });
      const sharpnessSliceSpec: (Slice | null)[] = [zarr.slice(startIdx, endIdx)];
      sharpness = (await lazyManager.loadSlice(
        sharpnessArr,
        'sharpness',
        sharpnessSliceSpec
      )) as Float32Array;
    } catch (error) {
      console.debug('[✨] [Luxar] No sharpness array found:', error);
      // Sharpness is optional
    }

    // Update geometry attributes
    const geom = points.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));

    if (col) {
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    } else {
      const defaultColors = new Float32Array(numLoadedPoints * 3);
      defaultColors.fill(1.0);
      geom.setAttribute('color', new THREE.BufferAttribute(defaultColors, 3));
    }

    if (radii) {
      geom.setAttribute('radius', new THREE.BufferAttribute(radii, 1));
    } else {
      const defaultRadii = new Float32Array(numLoadedPoints);
      defaultRadii.fill(1.0);
      geom.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
    }

    if (sharpness) {
      geom.setAttribute('sharpness', new THREE.BufferAttribute(sharpness, 1));
    } else {
      const defaultSharpness = new Float32Array(numLoadedPoints);
      defaultSharpness.fill(2.0);
      geom.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
    }

    // Signal that geometry has changed
    geom.computeBoundingBox();
    geom.computeBoundingSphere();

    // Preload adjacent frames for smooth navigation
    // The positions array typically has shape (n_points, n_coords) where:
    // - First dimension: points (sliced based on navigation)
    // - Second dimension: coordinate values (always fully loaded)

    // Calculate current chunk position in the array
    const pointsPerChunk = positionsArray.chunks[0];
    const currentChunkIdx = Math.floor((linearIndex * numLoadedPoints) / pointsPerChunk);

    // Build array position based on array dimensions
    const arrayNdims = positionsArray.shape.length;
    const arrayPosition = new Array(arrayNdims).fill(0);
    arrayPosition[0] = currentChunkIdx; // First dimension is the points dimension

    // Determine which dimensions to fully load
    // For positions arrays, all dimensions except the first (points) should be fully loaded
    const arrayDimsToFullyLoad = [];
    for (let d = 1; d < arrayNdims; d++) {
      arrayDimsToFullyLoad.push(d);
    }

    lazyManager
      .preloadChunks(positionsArray, 'positions', arrayPosition, arrayDimsToFullyLoad)
      .catch((error: any) => {
        console.warn('[⚠️] [Luxar] Failed to preload chunks:', error);
      });
  } catch (error) {
    console.error('[❌] [Luxar] Failed to update lazy-loaded point cloud:', error);
  }
}
