/**
 * High Contrast Theme - Maximum accessibility
 *
 * WCAG AAA compliant theme with pure colors and maximum contrast ratios
 * for users with low vision or in high-ambient-light environments.
 */

import type { Theme } from '../types';
import { darkTheme } from './dark.theme';

/**
 * High contrast theme definition
 *
 * Pure colors with 7:1 minimum contrast ratios (WCAG AAA).
 * No transparency, minimal shadows, no rounded corners.
 */
export const highContrastTheme: Theme = {
  id: 'high-contrast',
  name: 'High Contrast',
  description: 'Maximum accessibility with WCAG AAA contrast ratios',

  colors: {
    background: {
      primary: '#000000', // Pure black (21:1 contrast with white text)
      secondary: '#000000', // Also pure black for panels
      tertiary: '#1a1a1a', // Very dark gray for nested elements
      overlay: 'rgba(0, 0, 0, 0.95)', // Almost opaque overlay
    },
    text: {
      primary: '#ffffff', // Pure white (21:1 contrast with black bg)
      secondary: '#ffffff', // Also pure white
      muted: '#cccccc', // Light gray still has 12.6:1 contrast
      disabled: '#999999', // Gray with 7.5:1 contrast (still AAA)
      inverse: '#000000', // Pure black for rare light backgrounds
    },
    semantic: {
      success: '#00ff00', // Pure green (maximum visibility)
      warning: '#ffff00', // Pure yellow (maximum visibility)
      error: '#ff0000', // Pure red (maximum visibility)
      info: '#00ffff', // Pure cyan (maximum visibility)
      highlight: '#00ff00', // Use pure green for highlights
    },
    interactive: {
      default: 'rgba(255, 255, 255, 0.2)', // Visible but subtle
      hover: 'rgba(255, 255, 255, 0.3)', // Clear hover state
      active: 'rgba(255, 255, 255, 0.4)', // Clear active state
      focus: 'rgba(0, 255, 0, 0.5)', // Bright green focus
      disabled: 'rgba(255, 255, 255, 0.1)', // Very subtle disabled
    },
    border: {
      default: 'rgba(255, 255, 255, 0.5)', // Visible borders
      subtle: 'rgba(255, 255, 255, 0.3)', // Still visible dividers
      strong: 'rgba(255, 255, 255, 0.8)', // Very visible borders
      focus: '#00ff00', // Pure green focus border
    },
    visualization: {
      hot: '#ff0000', // Pure red
      warm: '#ffff00', // Pure yellow
      cold: '#00ff00', // Pure green
      neutral: '#ffffff', // Pure white
    },
  },

  // Typography - same as dark theme
  typography: darkTheme.typography,

  // Spacing - same as dark theme
  spacing: darkTheme.spacing,

  // Effects - modified for high contrast
  effects: {
    // No rounded corners for clarity
    borderRadius: {
      none: '0px',
      sm: '0px',
      md: '0px',
      lg: '0px',
      full: '0px',
    },
    // Stronger, more visible shadows (borders instead of shadows)
    shadow: {
      sm: '0 0 0 2px rgba(255, 255, 255, 0.5)',
      md: '0 0 0 3px rgba(255, 255, 255, 0.5)',
      lg: '0 0 0 4px rgba(255, 255, 255, 0.5)',
      xl: '0 0 0 5px rgba(255, 255, 255, 0.5)',
    },
    // No blur effects for clarity
    blur: {
      none: 'none',
      sm: 'none',
      md: 'none',
      lg: 'none',
    },
    // Same opacity scale
    opacity: darkTheme.effects.opacity,
    // Same transitions
    transition: darkTheme.effects.transition,
  },

  // Z-index - same as dark theme
  zIndex: darkTheme.zIndex,
};
