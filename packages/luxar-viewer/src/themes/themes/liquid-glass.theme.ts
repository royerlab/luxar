/**
 * Liquid Glass Theme - True Glassmorphism with Inner Glow
 *
 * Based on advanced glassmorphism techniques with inner shadow glow,
 * higher opacity glass tint, and subtle background blur.
 *
 * Inspired by: https://github.com/archisvaze/liquid-glass
 *
 * Key characteristics:
 * - Inner shadow glow for depth (inset white shadow)
 * - Higher opacity glass tint (0.4) for visible frosted effect
 * - Light background blur (2-8px, not heavy)
 * - Soft outer shadows
 * - Large rounded corners (18-28px)
 * - Optional: SVG noise for texture (future enhancement)
 *
 * Technical parameters (adjustable):
 * - Glass tint: rgba(255, 255, 255, 0.4)
 * - Inner shadow: inset 0 0 20px -5px rgba(255, 255, 255, 0.7)
 * - Background blur: blur(2-8px)
 * - Border radius: 18-28px
 * - Outer shadow: 0 6px 24px rgba(0, 0, 0, 0.2)
 */

import type { Theme } from '../types';

export const liquidGlassTheme: Theme = {
  id: 'liquid-glass',
  name: 'Liquid Glass',
  description: 'True glassmorphism with inner glow and glass tint',

  colors: {
    background: {
      // More transparent for liquid, flowing appearance
      primary: 'rgba(255, 255, 255, 0.18)', // Subtle glass tint
      secondary: 'rgba(255, 255, 255, 0.15)', // Main glass panels
      tertiary: 'rgba(255, 255, 255, 0.1)', // Very subtle inner elements
      overlay: 'rgba(0, 0, 0, 0.6)', // Modal overlay
    },
    text: {
      // Light text for translucent glass on dark backgrounds. The panel's
      // actual contrast floor is the `--luxar-glass-tint` dark tint painted
      // by the `.luxar-glass-surface::after` layer in liquid-glass.css (this
      // theme's own background.secondary token is a translucent WHITE and is
      // not what text sits on). Issue #1513: pre-fix, that layer's tint was
      // painted at an effective 0.385 alpha (a 0.55 tint further multiplied
      // by a stray `opacity: 0.7` on the same rule, since deleted) — over
      // the worst-case (pure-white scene) that composited to rgb(157) and
      // gave primary/secondary/muted 2.61:1 / 2.19:1 / 1.73:1: ALL THREE
      // failed AA (4.5:1), not only secondary/muted. Raised alongside
      // frosted-glass's text alphas (same alpha numbers — both themes share
      // this text-on-dark-tint shape, though the tint values themselves
      // differ) and alongside deleting the opacity multiplier; now
      // 7.3:1 / 6.2:1 / 5.0:1. Pinned by
      // tests/unit/themes/glass-contrast.test.ts, which reads the tint's
      // declared alpha AND the ::after rule's opacity out of liquid-glass.css
      // so the two can never drift apart.
      primary: 'rgba(255, 255, 255, 0.95)', // Bright white — 7.3:1 worst-case
      secondary: 'rgba(255, 255, 255, 0.85)', // Translucent white — 6.2:1 worst-case
      muted: 'rgba(255, 255, 255, 0.72)', // Subtle white — 5.0:1 worst-case
      disabled: 'rgba(255, 255, 255, 0.3)', // Very subtle
      inverse: 'rgba(0, 0, 0, 0.85)', // For light backgrounds
    },
    semantic: {
      // Vibrant, modern colors
      success: 'rgba(52, 199, 89, 1)', // Green
      warning: 'rgba(255, 149, 0, 1)', // Orange
      error: 'rgba(255, 59, 48, 1)', // Red
      info: 'rgba(0, 122, 255, 1)', // Blue
      // The brand interactive accent — same bright blue as the dark theme.
      // (Was indigo rgba(88,86,214,1), ~1.6:1 contrast on dark glass panels
      // — active chips, ZARR badges, and the wordmark were barely legible.)
      highlight: 'rgba(0, 160, 255, 1)',
    },
    interactive: {
      // Light interactive states for glass
      default: 'rgba(120, 120, 128, 0.16)', // Subtle fill
      hover: 'rgba(120, 120, 128, 0.24)', // More visible
      active: 'rgba(120, 120, 128, 0.32)', // Active state
      focus: 'rgba(0, 122, 255, 0.15)', // Blue focus
      disabled: 'rgba(120, 120, 128, 0.08)', // Very subtle
    },
    border: {
      // Subtle borders for glass
      default: 'rgba(255, 255, 255, 0.2)', // Light border
      subtle: 'rgba(255, 255, 255, 0.1)', // Very subtle
      strong: 'rgba(255, 255, 255, 0.3)', // More visible
      focus: 'rgba(0, 122, 255, 0.5)', // Blue focus ring
    },
    menu: {
      // Native <option> popups render off the glass layer → solid dark for readability
      background: '#1a1a1a',
      text: '#e0e0e0',
      activeBackground: '#2a2a2a',
      activeText: '#ffffff',
    },
    visualization: {
      // Vibrant visualization colors
      hot: 'rgba(255, 59, 48, 1)', // Red
      warm: 'rgba(255, 149, 0, 1)', // Orange
      cold: 'rgba(52, 199, 89, 1)', // Green
      neutral: 'rgba(142, 142, 147, 1)', // Gray
    },
  },

  typography: {
    fontFamily: {
      base: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
      mono: '"Monaco", "Menlo", "Consolas", "Courier New", monospace',
      display: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
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
      normal: 1.5,
      relaxed: 1.7,
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
      // Large, soft corners for glass effect
      none: '0px',
      sm: '12px',
      md: '18px', // Generous roundness
      lg: '28px', // Like example (28px)
      full: '9999px',
    },
    shadow: {
      // Multi-layer soft shadows + inner glow
      // Note: Inner shadow added via CSS (inset box-shadow)
      sm: '0 2px 8px rgba(0, 0, 0, 0.08), inset 0 0 10px -3px rgba(255, 255, 255, 0.5)',
      md: '0 4px 12px rgba(0, 0, 0, 0.1), inset 0 0 15px -4px rgba(255, 255, 255, 0.6)',
      lg: '0 6px 24px rgba(0, 0, 0, 0.2), inset 0 0 20px -5px rgba(255, 255, 255, 0.7)',
      xl: '0 12px 40px rgba(0, 0, 0, 0.25), inset 0 0 30px -6px rgba(255, 255, 255, 0.8)',
    },
    blur: {
      // Light blur for background (not heavy blur on panel)
      // Example uses blur(2px) on backdrop
      none: 'none',
      sm: 'blur(1px)', // Like example
      md: 'blur(2px)', // Light blur
      lg: 'blur(3px)', // Medium blur
    },
    opacity: {
      disabled: 0.3,
      secondary: 0.6,
      hover: 0.8,
      full: 1.0,
    },
    transition: {
      // Smooth transitions
      fast: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
      normal: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      slow: 'all 0.45s cubic-bezier(0.4, 0, 0.2, 1)',
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
