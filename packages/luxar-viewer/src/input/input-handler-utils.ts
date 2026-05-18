/**
 * Pure utility functions for input handling and navigation
 *
 * This module contains extracted functions for keyboard navigation,
 * dimension selection, and input validation. All functions are pure
 * and side-effect free for better testability.
 */

import { SimpleDims } from '../types/dims';
import { config } from '../config';
import { clamp } from '../utils/clamp';

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
 *
 * @example
 * ```typescript
 * // 5D dataset with X, Y, Z displayed (dims 0, 1, 2)
 * // Non-displayed: Time (dim 3), Channel (dim 4)
 * const dims = { ndim: 5, displayed: [0, 1, 2], ... };
 *
 * // Navigate from Time to Channel
 * const next = getNextDimensionIndex(3, 1, dims);
 * console.log(next); // 4 (Channel)
 *
 * // Navigate from Channel (last), wraps to Time
 * const wrapped = getNextDimensionIndex(4, 1, dims);
 * console.log(wrapped); // 3 (wraps around)
 * ```
 *
 * @example
 * ```typescript
 * // Jump to first non-displayed dimension from displayed dimension
 * const dims = { ndim: 4, displayed: [0, 1], ... };
 * const next = getNextDimensionIndex(1, 1, dims); // 1 is displayed (Y)
 * console.log(next); // 2 (first non-displayed)
 * ```
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
 *
 * @example
 * ```typescript
 * // 4D dataset: X, Y, Z, Time
 * const dims = {
 *   ndim: 4,
 *   displayed: [0, 1, 2], // X, Y, Z displayed
 *   currentStep: [0, 0, 0, 5.2],
 * };
 * const navigable = getNonDisplayedDimensions(dims);
 * console.log(navigable); // [3] - Time dimension
 * ```
 *
 * @example
 * ```typescript
 * // 3D dataset: X, Y, Z only
 * const dims3d = { ndim: 3, displayed: [0, 1, 2] };
 * const navigable = getNonDisplayedDimensions(dims3d);
 * console.log(navigable); // [] - No non-displayed dimensions
 * ```
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
 * Calculate adaptive step size for dimension navigation.
 *
 * Computes the appropriate step size based on dimension metadata, keyboard
 * modifiers, and navigation configuration. Step sizes adapt to:
 * - Discrete dimensions (frames): Step by 1 or more whole units
 * - Continuous dimensions (time): Step by 1% of range by default
 * - Shift modifier: Fine control (10x smaller steps)
 * - Ctrl modifier: Coarse control (10x larger steps)
 *
 * The adaptive strategy ensures smooth navigation regardless of data scale.
 * For example, navigating through 1000 time points uses reasonable step sizes
 * (10 by default, 1 with Shift, 100 with Ctrl).
 *
 * @param dimIndex - Zero-based index of dimension to navigate
 * @param dims - Complete dimension configuration including metadata
 * @param modifiers - Keyboard modifier state for fine/coarse control
 * @param modifiers.shift - If true, divides step size by 10 (fine control)
 * @param modifiers.ctrl - If true, multiplies step size by 10 (coarse control)
 * @param config - Navigation configuration (step multipliers, etc.)
 * @returns Step size for navigation, guaranteed positive and >= 1 for discrete dims
 *
 * @example
 * ```typescript
 * // Continuous time dimension: 0-100 seconds
 * const dims = {
 *   metadata: [{ name: 'time', range: [0, 100], step: undefined }],
 * };
 *
 * // Normal navigation: 1% of range = 1 second
 * const normal = calculateStepSize(0, dims);
 * console.log(normal); // 1.0
 *
 * // Fine control with Shift: 0.1 second
 * const fine = calculateStepSize(0, dims, { shift: true });
 * console.log(fine); // 0.1
 *
 * // Coarse control with Ctrl: 10 seconds
 * const coarse = calculateStepSize(0, dims, { ctrl: true });
 * console.log(coarse); // 10.0
 * ```
 *
 * @example
 * ```typescript
 * // Discrete frame dimension: 1000 frames
 * const dims = {
 *   metadata: [{ name: 'frame', range: [0, 999], discrete: true }],
 * };
 *
 * // Default: step by 10 frames (1% of 1000, rounded up)
 * const normal = calculateStepSize(0, dims);
 * console.log(normal); // 10
 *
 * // With Shift: step by 1 frame (minimum for discrete)
 * const fine = calculateStepSize(0, dims, { shift: true });
 * console.log(fine); // 1
 * ```
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
 * Calculate the next position in a dimension after applying navigation step.
 *
 * Handles dimension navigation with support for:
 * - Discrete (frame-based) and continuous (time-based) dimensions
 * - Boundary clamping or wrap-around behavior
 * - Rounding for discrete dimensions
 * - Min/max range enforcement
 *
 * This function is the core of keyboard navigation ([/] keys). It ensures
 * positions stay within valid bounds and provides predictable stepping
 * behavior for both discrete and continuous data.
 *
 * @param currentPos - Current position value in dimension coordinates
 * @param direction - Navigation direction: 1 for forward (]), -1 for backward ([)
 * @param stepSize - Step size to apply (from calculateStepSize)
 * @param range - Valid [min, max] bounds for this dimension
 * @param discrete - If true, rounds to nearest integer (for frame indices)
 * @param wrapAround - If true, wraps at boundaries; if false, clamps to range
 * @returns New position after navigation, guaranteed to be within range
 *
 * @example
 * ```typescript
 * // Continuous time dimension: navigate forward
 * const newPos = calculateNextPosition(
 *   5.2,        // current: 5.2 seconds
 *   1,          // forward
 *   1.0,        // step: 1 second
 *   [0, 100],   // range: 0-100 seconds
 *   false       // continuous
 * );
 * console.log(newPos); // 6.2 seconds
 * ```
 *
 * @example
 * ```typescript
 * // Discrete frame dimension: navigate backward
 * const newFrame = calculateNextPosition(
 *   50,         // current: frame 50
 *   -1,         // backward
 *   10,         // step: 10 frames
 *   [0, 999],   // range: 0-999 frames
 *   true,       // discrete (will round)
 *   false       // clamp at boundaries
 * );
 * console.log(newFrame); // 40 (integer)
 * ```
 *
 * @example
 * ```typescript
 * // Wrap-around at boundary (circular time loop)
 * const wrapped = calculateNextPosition(
 *   98,         // current: near end
 *   1,          // forward
 *   5,          // step: 5 units
 *   [0, 100],   // range
 *   false,      // continuous
 *   true        // wrap-around enabled
 * );
 * console.log(wrapped); // 3 (wraps from 103 to 3)
 * ```
 *
 * @example
 * ```typescript
 * // Clamp at boundary (default behavior)
 * const clamped = calculateNextPosition(
 *   98,         // current: near end
 *   1,          // forward
 *   5,          // step: 5 units
 *   [0, 100],   // range
 *   false,      // continuous
 *   false       // clamp at boundaries
 * );
 * console.log(clamped); // 100 (clamped to max)
 * ```
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
    if (rangeSize <= 0) {
      return range[0]; // Degenerate or invalid range
    }
    if (newPos < range[0]) {
      newPos = range[1] - ((range[0] - newPos) % rangeSize);
    } else if (newPos > range[1]) {
      newPos = range[0] + ((newPos - range[1]) % rangeSize);
    }
  } else {
    // Clamp to range
    newPos = clamp(newPos, range[0], range[1]);
  }

  return newPos;
}

/**
 * Map number key (1-9) to dimension index using navigable-position mapping.
 *
 * Converts keyboard number input to actual dimension indices by mapping
 * key N to the N-th non-displayed (navigable) dimension. This provides
 * an intuitive napari-style UX where keys always start at 1 regardless
 * of how many displayed dimensions exist.
 *
 * For a 5D dataset [X(0), Y(1), Z(2), Time(3), Channel(4)] with
 * X,Y,Z displayed:
 * - Key '1' → Time (dim 3, first navigable)
 * - Key '2' → Channel (dim 4, second navigable)
 * - Key '3' → -1 (no third navigable dim)
 *
 * @param key - String representation of number key pressed ('1' through '9')
 * @param dims - Complete dimension configuration with display settings
 * @returns Actual dimension index (0-based), or -1 if key is invalid
 *          or there aren't enough non-displayed dimensions
 *
 * @example
 * ```typescript
 * // 5D dataset: X, Y, Z displayed (dims 0, 1, 2)
 * const dims = { ndim: 5, displayed: [0, 1, 2], ... };
 * mapKeyToDimension('1', dims); // 3 (Time, first navigable)
 * mapKeyToDimension('2', dims); // 4 (Channel, second navigable)
 * mapKeyToDimension('3', dims); // -1 (only 2 navigable dims)
 * ```
 *
 * @example
 * ```typescript
 * // 5D dataset with non-contiguous display: dims 1, 2, 3 displayed
 * const dims = { ndim: 5, displayed: [1, 2, 3], ... };
 * mapKeyToDimension('1', dims); // 0 (first non-displayed)
 * mapKeyToDimension('2', dims); // 4 (second non-displayed)
 * ```
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

/**
 * Format dimension value for user-friendly display in UI.
 *
 * Converts raw dimension values to human-readable strings with appropriate
 * precision and units. Handles:
 * - Discrete dimensions: Rounded to nearest integer (e.g., "Frame 42")
 * - Continuous dimensions: Adaptive decimal places based on step size
 * - Unit suffixes: Appends unit if defined in metadata (e.g., "5.2s", "10μm")
 *
 * Used for dimension sliders, help overlays, and debug output. The adaptive
 * precision ensures values are displayed with appropriate detail - large
 * steps show fewer decimals, fine steps show more.
 *
 * @param value - Raw dimension value to format (e.g., 5.234)
 * @param dimIndex - Zero-based dimension index for metadata lookup
 * @param dims - Complete dimension configuration including metadata
 * @returns Formatted string ready for display (e.g., "5.2s" or "Frame 42")
 *
 * @example
 * ```typescript
 * // Continuous time dimension with units
 * const dims = {
 *   metadata: [{
 *     name: 'time',
 *     unit: 's',
 *     step: 0.1,  // Fine step = more decimals
 *     discrete: false
 *   }]
 * };
 * const formatted = formatDimensionValue(5.234, 0, dims);
 * console.log(formatted); // "5.2s" (1 decimal for step=0.1)
 * ```
 *
 * @example
 * ```typescript
 * // Discrete frame dimension
 * const dims = {
 *   metadata: [{
 *     name: 'frame',
 *     discrete: true,
 *     step: 1
 *   }]
 * };
 * const formatted = formatDimensionValue(42.7, 0, dims);
 * console.log(formatted); // "43" (rounded to integer)
 * ```
 *
 * @example
 * ```typescript
 * // Continuous without unit, default precision
 * const dims = {
 *   metadata: [{ name: 'channel' }]  // No step or unit
 * };
 * const formatted = formatDimensionValue(3.14159, 0, dims);
 * console.log(formatted); // "3.14" (2 decimals default)
 * ```
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
 * Generate formatted help text displaying current dimension navigation state.
 *
 * Creates a multi-line help display showing:
 * - Currently selected dimension with current value
 * - List of all non-displayed dimensions with values and key bindings
 * - Navigation instructions for keyboard controls
 *
 * Used in help overlays and debug panels to show users which dimensions
 * are available for navigation and how to control them. The output is
 * formatted for monospace display with clear alignment and indicators.
 *
 * @param selectedDim - Index of currently selected dimension for [/] navigation
 * @param dims - Complete dimension configuration with current positions
 * @returns Array of formatted strings, one per line, ready for display.
 *          Returns simplified message if no non-displayed dimensions exist.
 *
 * @example
 * ```typescript
 * // 5D dataset with Time selected
 * const dims = {
 *   ndim: 5,
 *   displayed: [0, 1, 2],  // X, Y, Z
 *   currentStep: [0, 0, 0, 5.2, 1],
 *   metadata: [
 *     { name: 'X' },
 *     { name: 'Y' },
 *     { name: 'Z' },
 *     { name: 'Time', unit: 's', range: [0, 10] },
 *     { name: 'Channel', discrete: true, range: [0, 3] }
 *   ]
 * };
 *
 * const help = generateNavigationHelp(3, dims);
 * console.log(help.join('\n'));
 * // Output:
 * // Selected: Time = 5.2s
 * //
 * // Non-displayed dimensions:
 * //   [1] Time: 5.2s ←
 * //   [2] Channel: 1
 * //
 * // Navigation:
 * //   [1-9] Select dimension
 * //   [ ]   Navigate selected dimension
 * //   Shift Hold for fine control
 * //   Ctrl  Hold for coarse control
 * ```
 *
 * @example
 * ```typescript
 * // 3D dataset: no non-displayed dimensions
 * const dims3d = { ndim: 3, displayed: [0, 1, 2], ... };
 * const help = generateNavigationHelp(0, dims3d);
 * console.log(help);
 * // ['All dimensions are displayed (3D view)']
 * ```
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

  // Available dimensions (keys map to navigable position, not raw index)
  help.push('');
  help.push('Non-displayed dimensions:');
  for (let navIdx = 0; navIdx < nonDisplayed.length; navIdx++) {
    const dimIdx = nonDisplayed[navIdx];
    const meta = dims.metadata?.[dimIdx];
    const name = meta?.name || `Dim ${dimIdx}`;
    const key = navIdx < 9 ? `[${navIdx + 1}]` : '   ';
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
 * Validate if keyboard event should trigger dimension navigation.
 *
 * Checks if the pressed key is a navigation key ([, ], or number keys 1-9)
 * and if the event context allows navigation (not typing in input field).
 * This prevents navigation from interfering with text input.
 *
 * Used as a guard before processing navigation events to ensure they're
 * appropriate for the current UI context. Returns false if user is typing
 * in a form field, search box, or any other text input element.
 *
 * @param event - Keyboard event to validate
 * @returns true if event should trigger navigation, false if it should be
 *          ignored (e.g., user is typing in an input field)
 *
 * @example
 * ```typescript
 * document.addEventListener('keydown', (event) => {
 *   if (isNavigationKey(event)) {
 *     event.preventDefault();
 *     handleDimensionNavigation(event);
 *   }
 *   // Otherwise, let event propagate normally (typing, etc.)
 * });
 * ```
 *
 * @example
 * ```typescript
 * // Returns true for navigation keys when not typing
 * const event1 = new KeyboardEvent('keydown', { key: '[' });
 * console.log(isNavigationKey(event1)); // true
 *
 * // Returns false when focus is in text input
 * const input = document.createElement('input');
 * input.focus();
 * const event2 = new KeyboardEvent('keydown', {
 *   key: '[',
 *   target: input
 * });
 * console.log(isNavigationKey(event2)); // false (typing in input)
 * ```
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
 * Calculate field of view (FOV) change from mouse wheel input.
 *
 * Converts mouse wheel delta to FOV adjustment with configurable sensitivity.
 * FOV is clamped to reasonable range (10°-120°) to prevent extreme distortion.
 * Used for Ctrl+wheel FOV control, allowing users to adjust perspective
 * from telephoto (narrow FOV) to wide-angle (wide FOV) views.
 *
 * Positive delta increases FOV (zoom out), negative delta decreases FOV
 * (zoom in). The sensitivity parameter scales the change rate.
 *
 * @param currentFov - Current field of view in degrees (typically 50-75°)
 * @param delta - Mouse wheel delta from WheelEvent.deltaY (typically -100 to 100)
 * @param sensitivity - Sensitivity multiplier (default from config, typically 0.1)
 * @returns New FOV value in degrees, clamped to [10°, 120°] range
 *
 * @example
 * ```typescript
 * // Zoom in (decrease FOV) with negative wheel delta
 * const currentFov = 60; // degrees
 * const newFov = calculateFovChange(
 *   currentFov,
 *   -100,  // scroll up
 *   0.1    // default sensitivity
 * );
 * console.log(newFov); // 50° (decreased by 10°)
 * ```
 *
 * @example
 * ```typescript
 * // Zoom out (increase FOV) with positive wheel delta
 * const wideAngle = calculateFovChange(60, 200, 0.1);
 * console.log(wideAngle); // 80° (increased by 20°)
 * ```
 *
 * @example
 * ```typescript
 * // Clamping at minimum (fovMin from config)
 * const minFov = calculateFovChange(15, -100, 0.1);
 * console.log(minFov); // 10° (clamped to minimum)
 *
 * // Clamping at maximum (fovMax from config = 170°)
 * const maxFov = calculateFovChange(160, 200, 0.1);
 * console.log(maxFov); // 170° (clamped to maximum)
 * ```
 */
export function calculateFovChange(
  currentFov: number,
  delta: number,
  sensitivity: number = config.input.defaultSensitivity
): number {
  const change = delta * sensitivity;
  const newFov = currentFov + change;

  // Clamp to config-defined FOV range
  return clamp(newFov, config.camera.fovMin, config.camera.fovMax);
}

/**
 * Determine if keyboard shortcut should be blocked in current UI context.
 *
 * Checks various conditions that should prevent shortcut execution:
 * - Active modals/dialogs (shortcuts should not leak through)
 * - Text input focus (prevent navigation while typing)
 * - Browser shortcuts (Ctrl/Cmd+S, etc. should pass through)
 *
 * Used as a guard before processing keyboard shortcuts to prevent conflicts
 * and ensure predictable behavior. Returns true when shortcuts should be
 * suppressed, false when they should execute normally.
 *
 * @param event - Keyboard event to check for blocking conditions
 * @param activeModals - Array of modal IDs currently open (e.g., ['luxar-help-overlay', 'settings'])
 * @returns true if shortcut should be blocked (don't execute), false if it
 *          should proceed normally
 *
 * @example
 * ```typescript
 * // Block all shortcuts when modal is open
 * const activeModals = ['settings-dialog'];
 * const event = new KeyboardEvent('keydown', { key: 'p' });
 * console.log(shouldBlockShortcut(event, activeModals)); // true
 * ```
 *
 * @example
 * ```typescript
 * // Block shortcuts when typing in input field
 * const input = document.createElement('input');
 * input.focus();
 * const event = new KeyboardEvent('keydown', {
 *   key: 'r',
 *   target: input
 * });
 * console.log(shouldBlockShortcut(event, [])); // true (typing)
 * ```
 *
 * @example
 * ```typescript
 * // Allow browser shortcuts (Ctrl+S for save)
 * const saveEvent = new KeyboardEvent('keydown', {
 *   key: 's',
 *   ctrlKey: true
 * });
 * console.log(shouldBlockShortcut(saveEvent, [])); // false (browser handles)
 * ```
 *
 * @example
 * ```typescript
 * // Allow shortcuts in normal view (no modals, not typing)
 * const normalEvent = new KeyboardEvent('keydown', { key: 'p' });
 * console.log(shouldBlockShortcut(normalEvent, [])); // false (OK to execute)
 * ```
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
