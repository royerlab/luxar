/**
 * Luxar Theme System - Type Definitions
 *
 * Defines the structure for themes in the Luxar viewer.
 * Themes control all visual aspects: colors, typography, spacing, and effects.
 */

/**
 * Color palette for backgrounds
 */
export interface ThemeBackgroundColors {
  /** Primary background (main canvas/window) */
  primary: string;
  /** Secondary background (panels, cards) */
  secondary: string;
  /** Tertiary background (nested elements) */
  tertiary: string;
  /** Overlay background (modals, overlays) */
  overlay: string;
}

/**
 * Color palette for text content
 */
export interface ThemeTextColors {
  /** Primary text color (main content) */
  primary: string;
  /** Secondary text color (labels, captions) */
  secondary: string;
  /** Muted text color (disabled, placeholder) */
  muted: string;
  /** Disabled text color */
  disabled?: string;
  /** Inverse text color (for dark backgrounds in light themes) */
  inverse?: string;
}

/**
 * Semantic colors for UI states and feedback
 */
export interface ThemeSemanticColors {
  /** Success state (green) */
  success: string;
  /** Warning state (yellow/orange) */
  warning: string;
  /** Error state (red) */
  error: string;
  /** Info state (blue) */
  info: string;
  /** Highlight color (accent) */
  highlight?: string;
}

/**
 * Interactive element colors (buttons, inputs, etc.)
 */
export interface ThemeInteractiveColors {
  /** Default state */
  default: string;
  /** Hover state */
  hover: string;
  /** Active/pressed state */
  active: string;
  /** Focus state (keyboard navigation) */
  focus: string;
  /** Disabled state */
  disabled: string;
}

/**
 * Border colors
 */
export interface ThemeBorderColors {
  /** Default border color */
  default: string;
  /** Subtle border (dividers) */
  subtle: string;
  /** Strong border (emphasis) */
  strong: string;
  /** Focus border (keyboard navigation) */
  focus?: string;
}

/**
 * Native menu / dropdown-option colors.
 *
 * Native `<option>` popups render on an OS layer without the panel's
 * backdrop-filter, so they need solid, opaque colors (translucent glass
 * backgrounds would be unreadable). Centralising them here removes the
 * hardcoded `<select> option` blocks that were duplicated across components.
 */
export interface ThemeMenuColors {
  /** Option background */
  background: string;
  /** Option text */
  text: string;
  /** Hovered/selected option background */
  activeBackground: string;
  /** Hovered/selected option text */
  activeText: string;
}

/**
 * Visualization-specific colors
 */
export interface ThemeVisualizationColors {
  /** Hot/high intensity color */
  hot: string;
  /** Warm/medium-high intensity color */
  warm: string;
  /** Cold/low intensity color */
  cold: string;
  /** Neutral/zero color */
  neutral: string;
}

/**
 * Complete color palette for a theme
 */
export interface ThemeColors {
  background: ThemeBackgroundColors;
  text: ThemeTextColors;
  semantic: ThemeSemanticColors;
  interactive: ThemeInteractiveColors;
  border: ThemeBorderColors;
  menu: ThemeMenuColors;
  visualization: ThemeVisualizationColors;
}

/**
 * Font family definitions
 */
export interface ThemeFontFamily {
  /** Base UI font (sans-serif) */
  base: string;
  /** Monospace font (code, data) */
  mono: string;
  /** Display font (headings) */
  display?: string;
}

/**
 * Font size scale
 */
export interface ThemeFontSizes {
  /** Extra small: 9px */
  xs: string;
  /** Small: 10px */
  sm: string;
  /** Base: 11px */
  base: string;
  /** Medium: 12px */
  md: string;
  /** Large: 14px */
  lg: string;
  /** Extra large: 16px */
  xl: string;
  /** 2XL: 20px */
  '2xl': string;
  /** 3XL: 28px */
  '3xl': string;
  /** 4XL: 32px (large metric readouts) */
  '4xl': string;
}

/**
 * Font weight scale
 */
export interface ThemeFontWeights {
  /** Normal: 400 */
  normal: number;
  /** Medium: 500 */
  medium: number;
  /** Semibold: 600 */
  semibold: number;
  /** Bold: 700 */
  bold: number;
}

/**
 * Line height scale
 */
export interface ThemeLineHeights {
  /** Tight: 1.2 */
  tight: number;
  /** Normal: 1.4 */
  normal: number;
  /** Relaxed: 1.6 */
  relaxed: number;
}

/**
 * Typography settings
 */
export interface ThemeTypography {
  fontFamily: ThemeFontFamily;
  fontSize: ThemeFontSizes;
  fontWeight: ThemeFontWeights;
  lineHeight: ThemeLineHeights;
}

/**
 * Spacing scale (8px grid system)
 */
export interface ThemeSpacing {
  /** 0px */
  0: string;
  /** 2px */
  1: string;
  /** 4px */
  2: string;
  /** 6px */
  3: string;
  /** 8px */
  4: string;
  /** 10px */
  5: string;
  /** 12px */
  6: string;
  /** 16px */
  8: string;
  /** 20px */
  10: string;
  /** 24px */
  12: string;
  /** 32px */
  16: string;
  /** 40px */
  20: string;
}

/**
 * Border radius scale
 */
export interface ThemeBorderRadius {
  /** None: 0px */
  none: string;
  /** Small: 4px */
  sm: string;
  /** Medium: 8px */
  md: string;
  /** Large: 12px */
  lg: string;
  /** Full: 9999px (pill shape) */
  full: string;
}

/**
 * Box shadow scale
 */
export interface ThemeShadows {
  /** Small shadow */
  sm: string;
  /** Medium shadow */
  md: string;
  /** Large shadow */
  lg: string;
  /** Extra large shadow */
  xl: string;
}

/**
 * Backdrop blur scale
 */
export interface ThemeBlur {
  /** None */
  none: string;
  /** Small: 4px */
  sm: string;
  /** Medium: 10px */
  md: string;
  /** Large: 20px */
  lg: string;
}

/**
 * Opacity scale
 */
export interface ThemeOpacity {
  /** Disabled: 0.4 */
  disabled: number;
  /** Secondary: 0.6 */
  secondary: number;
  /** Hover: 0.8 */
  hover: number;
  /** Full: 1.0 */
  full: number;
}

/**
 * Transition presets
 */
export interface ThemeTransitions {
  /** Fast: 0.1s */
  fast: string;
  /** Normal: 0.2s */
  normal: string;
  /** Slow: 0.3s */
  slow: string;
}

/**
 * Visual effects (shadows, blur, etc.)
 */
export interface ThemeEffects {
  borderRadius: ThemeBorderRadius;
  shadow: ThemeShadows;
  blur: ThemeBlur;
  opacity: ThemeOpacity;
  transition: ThemeTransitions;
}

/**
 * Z-index layers
 */
export interface ThemeZIndex {
  /** Base layer: 100 */
  base: number;
  /** Dropdown layer: 1000 */
  dropdown: number;
  /** Modal layer: 2000 */
  modal: number;
  /** Popover layer: 3000 */
  popover: number;
  /** Tooltip layer: 4000 */
  tooltip: number;
}

/**
 * Complete theme definition
 *
 * A theme controls all visual aspects of the Luxar viewer.
 * Themes can be switched at runtime without reloading.
 *
 * @example
 * ```typescript
 * const myTheme: Theme = {
 *   id: 'my-theme',
 *   name: 'My Custom Theme',
 *   description: 'A beautiful custom theme',
 *   colors: { ... },
 *   typography: { ... },
 *   spacing: { ... },
 *   effects: { ... },
 *   zIndex: { ... },
 * };
 * ```
 */
export interface Theme {
  /** Unique theme identifier (kebab-case) */
  id: string;
  /** Human-readable theme name */
  name: string;
  /** Optional theme description */
  description?: string;

  /** Color palette */
  colors: ThemeColors;
  /** Typography settings */
  typography: ThemeTypography;
  /** Spacing scale */
  spacing: ThemeSpacing;
  /** Visual effects */
  effects: ThemeEffects;
  /** Z-index layers */
  zIndex: ThemeZIndex;
}

/**
 * Theme change event handler
 */
export type ThemeChangeHandler = (theme: Theme) => void;
