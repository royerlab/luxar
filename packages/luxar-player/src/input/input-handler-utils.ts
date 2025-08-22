/**
 * Pure utility functions for input handling and navigation
 *
 * This module contains extracted functions for keyboard navigation,
 * dimension selection, and input validation. All functions are pure
 * and side-effect free for better testability.
 */

import { SimpleDims } from '../types/dims';

/**
 * Keyboard navigation configuration
 */
export interface NavigationConfig {
  stepSizeMultiplier: number;
  fineStepDivisor: number;
  coarseStepMultiplier: number;
  wrapAround: boolean;
}

/**
 * Default navigation configuration
 */
export const DEFAULT_NAV_CONFIG: NavigationConfig = {
  stepSizeMultiplier: 1.0,
  fineStepDivisor: 10,
  coarseStepMultiplier: 10,
  wrapAround: false,
};

/**
 * Calculates the next dimension index for selection
 *
 * @param currentDim - Currently selected dimension
 * @param direction - Direction to move (1 or -1)
 * @param dims - Dimension configuration
 * @returns Next dimension index to select
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
 * Gets list of non-displayed dimension indices
 *
 * @param dims - Dimension configuration
 * @returns Array of non-displayed dimension indices
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
 * Calculates navigation step size for a dimension
 *
 * @param dimIndex - Dimension index
 * @param dims - Dimension configuration
 * @param modifiers - Keyboard modifiers
 * @param config - Navigation configuration
 * @returns Step size for navigation
 */
export function calculateStepSize(
  dimIndex: number,
  dims: SimpleDims,
  modifiers: { shift?: boolean; ctrl?: boolean; alt?: boolean } = {},
  config: NavigationConfig = DEFAULT_NAV_CONFIG
): number {
  const meta = dims.metadata?.[dimIndex];

  // Get base step size
  let stepSize: number;
  if (meta?.step) {
    stepSize = meta.step;
  } else if (meta?.range) {
    // Calculate step as percentage of range
    const range = meta.range[1] - meta.range[0];
    stepSize = range * 0.01; // 1% of range
  } else {
    stepSize = 1.0; // Default
  }

  // Apply modifiers
  if (modifiers.shift) {
    stepSize /= config.fineStepDivisor; // Fine control
  } else if (modifiers.ctrl) {
    stepSize *= config.coarseStepMultiplier; // Coarse control
  }

  // Apply global multiplier
  stepSize *= config.stepSizeMultiplier;

  // For discrete dimensions, ensure step is at least 1
  if (meta?.discrete) {
    stepSize = Math.max(1, Math.round(stepSize));
  }

  return stepSize;
}

/**
 * Calculates the next position in a dimension after navigation
 *
 * @param currentPos - Current position in dimension
 * @param direction - Navigation direction (1 or -1)
 * @param stepSize - Size of navigation step
 * @param range - Valid range for dimension
 * @param discrete - Whether dimension is discrete
 * @param wrapAround - Whether to wrap at boundaries
 * @returns New position after navigation
 */
export function calculateNextPosition(
  currentPos: number,
  direction: 1 | -1,
  stepSize: number,
  range: [number, number],
  discrete: boolean = false,
  wrapAround: boolean = false
): number {
  let newPos = currentPos + direction * stepSize;

  // Handle discrete dimensions
  if (discrete) {
    newPos = Math.round(newPos);
  }

  // Handle boundaries
  if (wrapAround) {
    const rangeSize = range[1] - range[0];
    if (newPos < range[0]) {
      newPos = range[1] - ((range[0] - newPos) % rangeSize);
    } else if (newPos > range[1]) {
      newPos = range[0] + ((newPos - range[1]) % rangeSize);
    }
  } else {
    // Clamp to range
    newPos = Math.max(range[0], Math.min(range[1], newPos));
  }

  return newPos;
}

/**
 * Maps number key to dimension index
 *
 * @param key - Key pressed ('1' through '9')
 * @param dims - Dimension configuration
 * @returns Dimension index or -1 if invalid
 */
export function mapKeyToDimension(key: string, dims: SimpleDims): number {
  const num = parseInt(key);

  if (isNaN(num) || num < 1 || num > 9) {
    return -1;
  }

  // Map 1-9 to dimension indices 0-8
  const dimIndex = num - 1;

  // Check if dimension exists and is not displayed
  if (dimIndex >= dims.ndim || dims.displayed.includes(dimIndex)) {
    return -1;
  }

  return dimIndex;
}

/**
 * Formats dimension value for display
 *
 * @param value - Dimension value
 * @param dimIndex - Dimension index
 * @param dims - Dimension configuration
 * @returns Formatted string for display
 */
export function formatDimensionValue(value: number, dimIndex: number, dims: SimpleDims): string {
  const meta = dims.metadata?.[dimIndex];

  // Format based on dimension type
  let formatted: string;
  if (meta?.discrete) {
    formatted = Math.round(value).toString();
  } else {
    // Use appropriate decimal places
    const decimals = meta?.step ? Math.max(0, -Math.floor(Math.log10(meta.step))) : 2;
    formatted = value.toFixed(decimals);
  }

  // Add unit if available
  if (meta?.unit) {
    formatted += meta.unit;
  }

  return formatted;
}

/**
 * Generates help text for current dimension state
 *
 * @param selectedDim - Currently selected dimension
 * @param dims - Dimension configuration
 * @returns Help text array
 */
export function generateNavigationHelp(selectedDim: number, dims: SimpleDims): string[] {
  const help: string[] = [];
  const nonDisplayed = getNonDisplayedDimensions(dims);

  if (nonDisplayed.length === 0) {
    help.push('All dimensions are displayed (3D view)');
    return help;
  }

  // Current selection
  if (selectedDim >= 0 && selectedDim < dims.ndim) {
    const meta = dims.metadata?.[selectedDim];
    const name = meta?.name || `Dimension ${selectedDim}`;
    const value = formatDimensionValue(dims.currentStep[selectedDim], selectedDim, dims);
    help.push(`Selected: ${name} = ${value}`);
  } else {
    help.push('No dimension selected');
  }

  // Available dimensions
  help.push('');
  help.push('Non-displayed dimensions:');
  for (const dimIdx of nonDisplayed) {
    const meta = dims.metadata?.[dimIdx];
    const name = meta?.name || `Dim ${dimIdx}`;
    const key = dimIdx < 9 ? `[${dimIdx + 1}]` : '   ';
    const value = formatDimensionValue(dims.currentStep[dimIdx], dimIdx, dims);
    const selected = dimIdx === selectedDim ? ' ←' : '';
    help.push(`  ${key} ${name}: ${value}${selected}`);
  }

  // Navigation instructions
  help.push('');
  help.push('Navigation:');
  help.push('  [1-9] Select dimension');
  help.push('  [ ]   Navigate selected dimension');
  help.push('  Shift Hold for fine control');
  help.push('  Ctrl  Hold for coarse control');

  return help;
}

/**
 * Validates keyboard event for navigation
 *
 * @param event - Keyboard event
 * @returns True if event should be handled for navigation
 */
export function isNavigationKey(event: KeyboardEvent): boolean {
  // Ignore if typing in input field
  const target = event.target as HTMLElement;
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
    return false;
  }

  // Check for navigation keys
  const navKeys = ['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  return navKeys.includes(event.key);
}

/**
 * Calculates field of view change
 *
 * @param currentFov - Current FOV in degrees
 * @param delta - Mouse wheel delta
 * @param sensitivity - Sensitivity multiplier
 * @returns New FOV value clamped to valid range
 */
export function calculateFovChange(
  currentFov: number,
  delta: number,
  sensitivity: number = 0.1
): number {
  const change = delta * sensitivity;
  const newFov = currentFov + change;

  // Clamp to reasonable range
  return Math.max(10, Math.min(120, newFov));
}

/**
 * Determines if a keyboard shortcut should be blocked
 *
 * @param event - Keyboard event
 * @param activeModals - List of active modal IDs
 * @returns True if shortcut should be blocked
 */
export function shouldBlockShortcut(event: KeyboardEvent, activeModals: string[] = []): boolean {
  // Block if modal is active
  if (activeModals.length > 0) {
    return true;
  }

  // Block if typing in input
  const target = event.target as HTMLElement;
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
    return true;
  }

  // Block browser shortcuts
  if (event.metaKey || (event.ctrlKey && ['s', 'o', 'p'].includes(event.key))) {
    return false; // Let browser handle
  }

  return false;
}
