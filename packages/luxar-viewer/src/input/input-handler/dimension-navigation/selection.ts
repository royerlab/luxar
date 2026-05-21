/**
 * Pure dimension-selection helpers (which dims are navigable, which key
 * maps to which dim, how to cycle through them). Pure functions so the
 * selection rules are unit-testable in isolation.
 *
 * @module input/input-handler/dimension-navigation/selection
 */

import type { SimpleDims } from '../../../types/dims';

/**
 * Get list of non-displayed dimension indices available for keyboard navigation.
 *
 * Returns dimensions that are not part of the 3D spatial view (X, Y, Z).
 * These are the dimensions that can be controlled with [ ] keys and appear
 * in dimension sliders. For a 5D dataset with X, Y, Z displayed, returns
 * the indices of Time and Channel dimensions.
 *
 * @param dims - Complete dimension configuration with display settings
 * @returns Array of dimension indices not in the displayed list, sorted ascending.
 *          Empty array if all dimensions are displayed (pure 3D dataset).
 */
export function getNonDisplayedDimensions(dims: SimpleDims): number[] {
  const nonDisplayed: number[] = [];

  for (let i = 0; i < dims.ndim; i++) {
    if (!dims.displayed.includes(i)) {
      nonDisplayed.push(i);
    }
  }

  return nonDisplayed;
}

/**
 * Calculate the next dimension index for cyclic selection.
 *
 * Cycles through non-displayed dimensions in the given direction with
 * wrap-around at boundaries. Used for Tab/Shift+Tab dimension selection.
 * If the current dimension is not in the non-displayed list, jumps to
 * the first (forward) or last (backward) non-displayed dimension.
 *
 * @param currentDim - Currently selected dimension index (0-based)
 * @param direction - Navigation direction: 1 for next, -1 for previous
 * @param dims - Complete dimension configuration from scene
 * @returns Next dimension index to select, or -1 if no non-displayed dimensions exist
 */
export function getNextDimensionIndex(
  currentDim: number,
  direction: 1 | -1,
  dims: SimpleDims
): number {
  const nonDisplayed = getNonDisplayedDimensions(dims);

  if (nonDisplayed.length === 0) {
    return -1; // No non-displayed dimensions
  }

  const currentIndex = nonDisplayed.indexOf(currentDim);
  let nextIndex: number;

  if (currentIndex === -1) {
    // Not currently on a non-displayed dimension
    nextIndex = direction > 0 ? 0 : nonDisplayed.length - 1;
  } else {
    // Move to next/previous
    nextIndex = currentIndex + direction;

    // Handle wrap-around
    if (nextIndex < 0) {
      nextIndex = nonDisplayed.length - 1;
    } else if (nextIndex >= nonDisplayed.length) {
      nextIndex = 0;
    }
  }

  return nonDisplayed[nextIndex];
}

/**
 * Map number key (1-9) to dimension index using navigable-position mapping.
 *
 * Converts keyboard number input to actual dimension indices by mapping
 * key N to the N-th non-displayed (navigable) dimension. This provides
 * an intuitive napari-style UX where keys always start at 1 regardless
 * of how many displayed dimensions exist.
 *
 * @param key - String representation of number key pressed ('1' through '9')
 * @param dims - Complete dimension configuration with display settings
 * @returns Actual dimension index (0-based), or -1 if key is invalid
 *          or there aren't enough non-displayed dimensions
 */
export function mapKeyToDimension(key: string, dims: SimpleDims): number {
  const num = parseInt(key);

  if (isNaN(num) || num < 1 || num > 9) {
    return -1;
  }

  // Map key N to the N-th non-displayed (navigable) dimension
  const nonDisplayed = getNonDisplayedDimensions(dims);
  const navigableIndex = num - 1;

  if (navigableIndex >= nonDisplayed.length) {
    return -1;
  }

  return nonDisplayed[navigableIndex];
}
