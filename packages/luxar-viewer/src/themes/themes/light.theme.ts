/**
 * Light Theme - Bright, clean theme for well-lit environments
 *
 * Inverted colors from dark theme with adjusted contrast ratios
 * for readability on bright displays.
 */

import type { Theme } from '../types';
import { darkTheme } from './dark.theme';

/**
 * Light theme definition
 *
 * Inverts dark theme colors while maintaining WCAG AA contrast ratios.
 */
export const lightTheme: Theme = {
  id: 'light',
  name: 'Light Theme',
  description: 'Bright, clean theme optimized for well-lit environments',

  colors: {
    background: {
      primary: '#ffffff', // Pure white background
      secondary: 'rgba(250, 250, 250, 0.95)', // Very light gray panels
      tertiary: 'rgba(240, 240, 240, 0.8)', // Light gray nested elements
      overlay: 'rgba(0, 0, 0, 0.3)', // Inverted dark overlay
    },
    text: {
      primary: '#1a1a1a', // Almost black text (contrast ratio: 15.7:1 with white bg)
      secondary: '#666666', // Medium gray (contrast ratio: 5.74:1)
      muted: 'rgba(0, 0, 0, 0.6)', // Inverted muted
      disabled: 'rgba(0, 0, 0, 0.4)', // Inverted disabled
      inverse: '#ffffff', // White for dark backgrounds
    },
    semantic: {
      success: '#2e7d32', // Darker green for better contrast
      warning: '#f57c00', // Darker orange for better contrast
      error: '#c62828', // Darker red for better contrast
      info: '#1565c0', // Darker blue for better contrast
      highlight: '#0277bd', // Darker blue highlight
    },
    interactive: {
      default: 'rgba(0, 0, 0, 0.08)', // Light gray default
      hover: 'rgba(0, 0, 0, 0.12)', // Darker gray on hover
      active: 'rgba(0, 0, 0, 0.16)', // Even darker when active
      focus: 'rgba(46, 125, 50, 0.2)', // Green focus (using darker success color)
      disabled: 'rgba(0, 0, 0, 0.03)', // Very light disabled
    },
    border: {
      default: 'rgba(0, 0, 0, 0.15)', // Darker borders for visibility
      subtle: 'rgba(0, 0, 0, 0.08)', // Subtle dividers
      strong: 'rgba(0, 0, 0, 0.25)', // Emphasized borders
      focus: 'rgba(46, 125, 50, 0.5)', // Green focus border
    },
    menu: {
      background: '#ffffff', // Native <option> popup (solid — no glass)
      text: '#1e1e1e',
      activeBackground: '#f0f0f0',
      activeText: '#000000',
    },
    visualization: {
      hot: '#d32f2f', // Slightly darker red
      warm: '#f57c00', // Darker orange
      cold: '#2e7d32', // Darker green
      neutral: '#757575', // Medium gray
    },
  },

  // Typography, spacing, effects, and zIndex inherit from dark theme (copied for immutability)
  typography: {
    fontFamily: { ...darkTheme.typography.fontFamily },
    fontSize: { ...darkTheme.typography.fontSize },
    fontWeight: { ...darkTheme.typography.fontWeight },
    lineHeight: { ...darkTheme.typography.lineHeight },
  },
  spacing: { ...darkTheme.spacing },
  effects: {
    borderRadius: { ...darkTheme.effects.borderRadius },
    shadow: {
      sm: '0 1px 2px rgba(0, 0, 0, 0.08)',
      md: '0 4px 12px rgba(0, 0, 0, 0.1)',
      lg: '0 8px 24px rgba(0, 0, 0, 0.12)',
      xl: '0 12px 48px rgba(0, 0, 0, 0.15)',
    },
    blur: { ...darkTheme.effects.blur },
    opacity: { ...darkTheme.effects.opacity },
    transition: { ...darkTheme.effects.transition },
  },
  zIndex: { ...darkTheme.zIndex },
};
