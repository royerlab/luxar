/**
 * Liquid Glass Theme - Apple-inspired Modern Design
 *
 * Inspired by Apple's frosted glass design language with translucent
 * panels, soft shadows, and modern aesthetics.
 *
 * Key characteristics:
 * - Heavy blur for frosted glass effect
 * - Translucent backgrounds (high opacity backdrop-filter)
 * - Soft, diffused shadows
 * - Modern color palette with subtle tints
 * - Rounded corners throughout
 * - Smooth, fluid animations
 */

import type { Theme } from '../types';

export const liquidGlassTheme: Theme = {
  id: 'liquid-glass',
  name: 'Liquid Glass',
  description: 'Apple-inspired frosted glass design with modern aesthetics',

  colors: {
    background: {
      // Bright frosted glass with high opacity and heavy blur
      primary: 'rgba(255, 255, 255, 0.15)', // Subtle white tint
      secondary: 'rgba(255, 255, 255, 0.12)', // Even more subtle
      tertiary: 'rgba(255, 255, 255, 0.08)', // Very subtle glass layer
      overlay: 'rgba(0, 0, 0, 0.6)', // Darker overlay for modals
    },
    text: {
      // Light text for dark backgrounds with glass effect
      primary: 'rgba(255, 255, 255, 0.95)', // Bright white
      secondary: 'rgba(255, 255, 255, 0.75)', // Translucent white
      muted: 'rgba(255, 255, 255, 0.5)', // More subtle
      disabled: 'rgba(255, 255, 255, 0.3)', // Very subtle
      inverse: 'rgba(0, 0, 0, 0.9)', // For light backgrounds
    },
    semantic: {
      // Modern, vibrant but subtle semantic colors
      success: 'rgba(52, 199, 89, 1)', // Apple green
      warning: 'rgba(255, 149, 0, 1)', // Apple orange
      error: 'rgba(255, 59, 48, 1)', // Apple red
      info: 'rgba(0, 122, 255, 1)', // Apple blue
      highlight: 'rgba(88, 86, 214, 1)', // Apple purple
    },
    interactive: {
      // Subtle interactive states with glassmorphism
      default: 'rgba(255, 255, 255, 0.15)', // Subtle white
      hover: 'rgba(255, 255, 255, 0.22)', // Brighter on hover
      active: 'rgba(255, 255, 255, 0.3)', // Even brighter when pressed
      focus: 'rgba(0, 122, 255, 0.25)', // Blue tint on focus
      disabled: 'rgba(255, 255, 255, 0.08)', // Very subtle
    },
    border: {
      // Subtle glass borders with slight luminosity
      default: 'rgba(255, 255, 255, 0.18)', // Subtle white border
      subtle: 'rgba(255, 255, 255, 0.1)', // Very subtle
      strong: 'rgba(255, 255, 255, 0.25)', // More visible
      focus: 'rgba(0, 122, 255, 0.6)', // Bright blue focus ring
    },
    visualization: {
      // Vibrant but sophisticated visualization colors
      hot: 'rgba(255, 59, 48, 1)', // Red
      warm: 'rgba(255, 149, 0, 1)', // Orange
      cold: 'rgba(52, 199, 89, 1)', // Green
      neutral: 'rgba(142, 142, 147, 1)', // Gray
    },
  },

  typography: {
    fontFamily: {
      // San Francisco-inspired fonts (Apple's system font)
      base: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif',
      mono: '"SF Mono", "Monaco", "Menlo", "Courier New", monospace',
      display: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
    },
    fontSize: {
      xs: '10px',
      sm: '11px',
      base: '13px', // Slightly larger for better readability
      md: '14px',
      lg: '16px',
      xl: '18px',
      '2xl': '20px',
      '3xl': '28px',
    },
    fontWeight: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5, // Slightly more spacious
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
      // Apple's characteristic rounded corners
      none: '0px',
      sm: '8px', // More rounded than default
      md: '12px', // Apple-like roundness
      lg: '18px', // Large, soft corners
      full: '9999px', // Perfect circles
    },
    shadow: {
      // Soft, diffused shadows (Apple style)
      sm: '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.06)',
      md: '0 4px 6px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.05)',
      lg: '0 10px 20px rgba(0, 0, 0, 0.08), 0 3px 6px rgba(0, 0, 0, 0.05)',
      xl: '0 20px 40px rgba(0, 0, 0, 0.1), 0 5px 10px rgba(0, 0, 0, 0.06)',
    },
    blur: {
      // Very heavy blur for authentic frosted glass
      none: 'none',
      sm: 'blur(12px)',
      md: 'blur(32px)', // Strong blur for frosted glass
      lg: 'blur(64px)', // Extreme blur for heavy frosted effect
    },
    opacity: {
      disabled: 0.3,
      secondary: 0.6,
      hover: 0.8,
      full: 1.0,
    },
    transition: {
      // Smooth, fluid transitions (Apple style)
      fast: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)', // Ease-in-out
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
