/**
 * Pure display-formatting helpers for nD dimension values: converts raw
 * dimension positions to user-readable strings with units, and builds the
 * multi-line help text shown in the navigation overlay.
 *
 * @module input/input-handler/dimension-navigation/format
 */

import type { SimpleDims } from '../../../types/dims';
import { getNonDisplayedDimensions } from './selection';

/**
 * Format dimension value for user-friendly display in UI.
 *
 * Converts raw dimension values to human-readable strings with appropriate
 * precision and units. Handles:
 * - Discrete dimensions: Rounded to nearest integer (e.g., "Frame 42")
 * - Continuous dimensions: Adaptive decimal places based on step size
 * - Unit suffixes: Appends unit if defined in metadata (e.g., "5.2s", "10μm")
 *
 * @param value - Raw dimension value to format (e.g., 5.234)
 * @param dimIndex - Zero-based dimension index for metadata lookup
 * @param dims - Complete dimension configuration including metadata
 * @returns Formatted string ready for display (e.g., "5.2s" or "Frame 42")
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
