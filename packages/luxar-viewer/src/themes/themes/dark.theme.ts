/**
 * Dark Theme - Default Luxar Theme
 *
 * The classic dark theme optimized for scientific visualization.
 * High contrast, easy on the eyes, and excellent for extended use.
 */

import type { Theme } from '../types';

/** Dark theme definition. */
export const darkTheme: Theme = {
  id: 'dark',
  name: 'Dark Theme',
  description: 'Classic dark theme optimized for scientific visualization',

  colors: {
    background: {
      primary: '#111111', // Main background
      secondary: 'rgba(30, 30, 30, 0.95)', // Panels, cards
      tertiary: 'rgba(0, 0, 0, 0.3)', // Nested elements
      overlay: 'rgba(0, 0, 0, 0.5)', // Modal overlays
    },
    text: {
      primary: '#e0e0e0', // Main text
      secondary: '#888888', // Labels, captions
      muted: 'rgba(255, 255, 255, 0.6)', // Disabled, placeholder
      disabled: 'rgba(255, 255, 255, 0.4)', // Disabled text
      inverse: '#111111', // For light backgrounds
    },
    semantic: {
      success: '#4CAF50', // Green
      warning: '#FFC107', // Yellow/amber
      error: '#f44336', // Red
      info: '#2196F3', // Blue
      highlight: '#00a0ff', // Accent highlight
    },
    interactive: {
      default: 'rgba(255, 255, 255, 0.1)', // Default state
      hover: 'rgba(255, 255, 255, 0.15)', // Hover state
      active: 'rgba(255, 255, 255, 0.2)', // Active/pressed
      focus: 'rgba(76, 175, 80, 0.3)', // Focus (success color)
      disabled: 'rgba(255, 255, 255, 0.05)', // Disabled
    },
    border: {
      default: 'rgba(255, 255, 255, 0.1)', // Default borders
      subtle: 'rgba(255, 255, 255, 0.05)', // Dividers
      strong: 'rgba(255, 255, 255, 0.2)', // Emphasized borders
      focus: 'rgba(76, 175, 80, 0.5)', // Focus border
    },
    menu: {
      background: '#1e1e1e', // Native <option> popup (solid — no glass)
      text: '#e0e0e0',
      activeBackground: '#2a2a2a',
      activeText: '#ffffff',
    },
    visualization: {
      hot: '#ff6b6b', // Hot/high intensity (red)
      warm: '#FFC107', // Warm/medium-high (amber)
      cold: '#4CAF50', // Cold/low intensity (green)
      neutral: '#9E9E9E', // Neutral/zero (gray)
    },
  },

  typography: {
    fontFamily: {
      base: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif',
      mono: '"Monaco", "Menlo", "Ubuntu Mono", monospace',
      display: '"Inter", -apple-system, sans-serif',
    },
    // Unified type scale shared by all four themes (see AskUserQuestion
    // decision 2026-07): a slightly larger, more readable base.
    fontSize: {
      xs: '10px',
      sm: '11px',
      base: '13px',
      md: '14px',
      lg: '16px',
      xl: '18px',
      '2xl': '20px',
      '3xl': '28px',
      '4xl': '32px',
    },
    fontWeight: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.4,
      relaxed: 1.6,
    },
  },

  spacing: {
    0: '0px',
    1: '2px',
    2: '4px',
    3: '6px',
    4: '8px',
    5: '10px',
    6: '12px',
    8: '16px',
    10: '20px',
    12: '24px',
    16: '32px',
    20: '40px',
  },

  effects: {
    borderRadius: {
      none: '0px',
      sm: '4px',
      md: '8px',
      lg: '12px',
      full: '9999px',
    },
    shadow: {
      sm: '0 1px 2px rgba(0, 0, 0, 0.2)',
      md: '0 4px 12px rgba(0, 0, 0, 0.3)',
      lg: '0 8px 24px rgba(0, 0, 0, 0.4)',
      xl: '0 12px 48px rgba(0, 0, 0, 0.5)',
    },
    blur: {
      none: 'none',
      sm: 'blur(4px)',
      md: 'blur(10px)',
      lg: 'blur(20px)',
    },
    opacity: {
      disabled: 0.4,
      secondary: 0.6,
      hover: 0.8,
      full: 1.0,
    },
    transition: {
      fast: 'all 0.1s ease',
      normal: 'all 0.2s ease',
      slow: 'all 0.3s ease',
    },
  },

  zIndex: {
    base: 100,
    dropdown: 1000,
    modal: 2000,
    popover: 3000,
    tooltip: 4000,
  },
};
