/**
 * Keyboard navigation for dimensions
 */

import { SimpleDims } from '../types/dims';
import {
  slicePoints,
  extractDisplayDimensions,
  sliceColors,
  sliceScalarAttribute,
  getDimsLabel,
  computeEffectiveRadii,
} from './slicing';
import * as THREE from 'three';

interface NavigationOptions {
  stepSize?: number; // Fraction of range to step (default 0.1 = 10%)
  wrap?: boolean; // Wrap around at boundaries
  absoluteStep?: number; // Absolute step size (overrides stepSize)
}

/**
 * Step through a dimension
 */
export function stepDimension(
  dims: SimpleDims,
  dimIndex: number,
  direction: 1 | -1,
  ranges: Array<[number, number]>,
  options: NavigationOptions = {}
): boolean {
  const { stepSize = 0.1, wrap = false, absoluteStep } = options;

  if (dimIndex < 0 || dimIndex >= dims.ndim) {
    return false;
  }

  // Don't step displayed dimensions
  if (dims.displayed.includes(dimIndex)) {
    return false;
  }

  const [min, max] = ranges[dimIndex];
  const range = max - min;
  // Use absolute step if provided, otherwise calculate from range
  const step = absoluteStep !== undefined ? absoluteStep : range * stepSize;

  const current = dims.currentStep[dimIndex];
  let newValue = current + direction * step;

  // Handle boundaries
  if (newValue > max) {
    newValue = wrap ? min : max;
  } else if (newValue < min) {
    newValue = wrap ? max : min;
  }

  // Update if changed
  if (newValue !== current) {
    dims.currentStep[dimIndex] = newValue;
    return true;
  }

  return false;
}

/**
 * Jump to specific position in dimension
 */
export function jumpToDimension(
  dims: SimpleDims,
  dimIndex: number,
  fraction: number, // 0-1 where to jump in the range
  ranges: Array<[number, number]>
): boolean {
  if (dimIndex < 0 || dimIndex >= dims.ndim || dims.displayed.includes(dimIndex)) {
    return false;
  }

  const [min, max] = ranges[dimIndex];
  const newValue = min + (max - min) * Math.max(0, Math.min(1, fraction));

  if (newValue !== dims.currentStep[dimIndex]) {
    dims.currentStep[dimIndex] = newValue;
    return true;
  }

  return false;
}

/**
 * Get the first two non-displayed dimensions for primary/secondary navigation
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
 * Update point cloud geometry after dimension change
 */
export function updatePointCloudSlice(
  points: THREE.Points,
  originalPositions: Float32Array,
  originalColors: Uint8Array | undefined,
  originalRadii: Float32Array | undefined,
  originalSharpness: Float32Array | undefined,
  dims: SimpleDims,
  numPoints: number
): void {
  // Slice the data with new dims state using radius-based visibility
  const visibleIndices = slicePoints(originalPositions, dims, numPoints, originalRadii);
  const positions3D = extractDisplayDimensions(originalPositions, visibleIndices, dims);

  console.log(`Updated slice: ${getDimsLabel(dims)}`);
  console.log(`Showing ${visibleIndices.length} of ${numPoints} points`);

  // Handle empty slice
  if (visibleIndices.length === 0) {
    console.warn('No points visible at current slice position!');
    // Update with dummy point
    const geom = points.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3));
    geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array([1, 1, 1]), 3));
    geom.setAttribute('radius', new THREE.BufferAttribute(new Float32Array([0.1]), 1));
    geom.setAttribute('sharpness', new THREE.BufferAttribute(new Float32Array([2.0]), 1));
    return;
  }

  // Update geometry attributes
  const geom = points.geometry;
  geom.setAttribute('position', new THREE.BufferAttribute(positions3D, 3));

  // Update colors
  if (originalColors) {
    const colors = sliceColors(originalColors, visibleIndices);
    if (colors) {
      // Convert to float
      const floatColors = new Float32Array(colors.length);
      for (let i = 0; i < colors.length; i++) {
        floatColors[i] = colors[i] / 255.0;
      }
      geom.setAttribute('color', new THREE.BufferAttribute(floatColors, 3));
    }
  } else {
    // Default white
    const numVis = visibleIndices.length;
    const defaultColors = new Float32Array(numVis * 3);
    defaultColors.fill(1.0);
    geom.setAttribute('color', new THREE.BufferAttribute(defaultColors, 3));
  }

  // Update radii with effective radii based on slice intersection
  if (originalRadii) {
    const effectiveRadii = computeEffectiveRadii(
      originalPositions,
      visibleIndices,
      dims,
      originalRadii
    );
    geom.setAttribute('radius', new THREE.BufferAttribute(effectiveRadii, 1));
  } else {
    // Default radius
    const defaultRadii = new Float32Array(visibleIndices.length);
    defaultRadii.fill(0.1);
    geom.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
  }

  // Update sharpness
  if (originalSharpness) {
    const sharpness = sliceScalarAttribute(originalSharpness, visibleIndices);
    if (sharpness) {
      geom.setAttribute('sharpness', new THREE.BufferAttribute(sharpness, 1));
    }
  } else {
    // Default sharpness
    const defaultSharpness = new Float32Array(visibleIndices.length);
    defaultSharpness.fill(2.0);
    geom.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
  }

  // Force update
  geom.attributes.position.needsUpdate = true;
  geom.attributes.color.needsUpdate = true;
  geom.attributes.radius.needsUpdate = true;
  geom.attributes.sharpness.needsUpdate = true;
  geom.computeBoundingSphere();
}
