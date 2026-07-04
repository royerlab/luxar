/**
 * Theme Manager - Singleton for managing theme registration and switching
 *
 * Handles:
 * - Theme registration and retrieval
 * - Runtime theme switching with CSS variable injection
 * - Theme persistence via localStorage
 * - Observer pattern for theme change notifications
 */

import type { Theme, ThemeChangeHandler } from './types';
import { darkTheme } from './themes/dark.theme';
import { lightTheme } from './themes/light.theme';
import { frostedGlassTheme } from './themes/frosted-glass.theme';
import { liquidGlassTheme } from './themes/liquid-glass.theme';
import {
  injectGlassFilters,
  removeGlassFilters,
  injectGlassRefractionLayers,
  removeGlassRefractionLayers,
  setupGlassRefractionObserver,
} from './glass-filters';
import { log, Modules } from '../utils/log';
import { StorageKeys } from '../utils/storage-keys';

/**
 * ThemeManager singleton class
 *
 * Manages all themes and handles runtime theme switching by injecting
 * CSS custom properties into the document root.
 */
export class ThemeManager {
  private static instance: ThemeManager | null = null;

  /** Map of registered themes by ID */
  private themes: Map<string, Theme> = new Map();

  /** Currently active theme */
  private currentTheme: Theme | null = null;

  /** Theme change observers */
  private observers: Set<ThemeChangeHandler> = new Set();

  /** LocalStorage key for theme persistence (see {@link StorageKeys.theme}). */
  private readonly STORAGE_KEY = StorageKeys.theme;

  /** Cleanup function for the glass refraction observer */
  private glassRefractionObserverCleanup: (() => void) | null = null;

  /**
   * Handle for the requestAnimationFrame call that defers
   * `injectGlassRefractionLayers()` until the next frame. Tracked so that a
   * `dispose()` racing the deferred injection can cancel it — otherwise the
   * rAF fires after teardown and re-injects DOM that nothing will clean up.
   */
  private pendingRefractionRAF: number | null = null;

  /**
   * Private constructor (singleton pattern)
   */
  private constructor() {
    // Register all built-in themes
    this.registerTheme(darkTheme);
    this.registerTheme(lightTheme);
    this.registerTheme(frostedGlassTheme);
    this.registerTheme(liquidGlassTheme);

    // Load saved theme or use default
    this.initializeTheme();
  }

  /**
   * Get ThemeManager singleton instance
   */
  public static getInstance(): ThemeManager {
    if (!ThemeManager.instance) {
      ThemeManager.instance = new ThemeManager();
    }
    return ThemeManager.instance;
  }

  /**
   * Dispose the current instance and clear the singleton slot.
   *
   * Call this on app shutdown (so the next `getInstance()` builds a fresh
   * manager) or between tests (to isolate state). The instance reference is
   * cleared; the next `getInstance()` will lazily construct a new one.
   */
  public static disposeInstance(): void {
    if (ThemeManager.instance) {
      ThemeManager.instance.dispose();
      ThemeManager.instance = null;
    }
  }

  /**
   * Dispose the theme manager and clean up all resources
   *
   * This method:
   * - Clears all theme change observers
   * - Clears all registered themes
   * - Removes all CSS variables
   * - Removes data-theme attribute
   *
   * After calling dispose(), the ThemeManager instance cannot be reused.
   */
  public dispose(): void {
    // Clear all observers
    this.observers.clear();

    // Clear themes
    this.themes.clear();

    // Clear CSS variables
    this.clearThemeVariables();

    // Remove data-theme attribute
    document.documentElement.removeAttribute('data-theme');

    // Cancel any pending rAF that would inject refraction layers AFTER cleanup.
    if (this.pendingRefractionRAF !== null) {
      cancelAnimationFrame(this.pendingRefractionRAF);
      this.pendingRefractionRAF = null;
    }

    // Clean up glass refraction observer and layers
    if (this.glassRefractionObserverCleanup) {
      this.glassRefractionObserverCleanup();
      this.glassRefractionObserverCleanup = null;
    }
    removeGlassRefractionLayers();
    removeGlassFilters();

    // Clear current theme
    this.currentTheme = null;
  }

  /**
   * Register a theme for use
   *
   * @param theme - Theme to register
   * @throws Error if theme with same ID already exists
   */
  public registerTheme(theme: Theme): void {
    if (this.themes.has(theme.id)) {
      throw new Error(`[ThemeManager] Theme "${theme.id}" is already registered`);
    }
    this.themes.set(theme.id, theme);
  }

  /**
   * Get a theme by ID
   *
   * @param id - Theme ID
   * @returns Theme or undefined if not found
   */
  public getTheme(id: string): Theme | undefined {
    return this.themes.get(id);
  }

  /**
   * Get all registered themes
   *
   * @returns Array of all themes
   */
  public getAllThemes(): Theme[] {
    return Array.from(this.themes.values());
  }

  /**
   * Get the currently active theme
   *
   * @returns Current theme
   */
  public getCurrentTheme(): Theme {
    if (!this.currentTheme) {
      throw new Error('[ThemeManager] No theme is currently active');
    }
    return this.currentTheme;
  }

  /**
   * Set the active theme
   *
   * @param themeId - ID of theme to activate
   * @throws Error if theme not found
   */
  public setTheme(themeId: string): void {
    const theme = this.themes.get(themeId);
    if (!theme) {
      throw new Error(`[ThemeManager] Theme "${themeId}" not found`);
    }

    // Apply theme to DOM FIRST. If applyTheme throws (e.g. document/DOM
    // unavailable, root element missing, CSS injection failure), the throw
    // propagates and the steps below are skipped:
    //   - currentTheme is NOT updated (stays on the previous theme)
    //   - saveTheme is NOT called (localStorage stays on the previous theme)
    //   - observers are NOT notified
    // This preserves the invariant: persisted state == currentTheme == DOM
    // state. A partial-apply failure on this call leaves the previous theme
    // as the source of truth, rather than silently persisting a theme that
    // never fully applied (which would mask the failure on next reload).
    this.applyTheme(theme);

    // Update current theme (only reached if applyTheme succeeded).
    this.currentTheme = theme;

    // Persist to localStorage (only reached if applyTheme succeeded).
    this.saveTheme(themeId);

    // Notify observers
    this.notifyObservers(theme);

    log.info(Modules.UI, `Theme switched to "${theme.name}" (${theme.id})`);
  }

  /**
   * Subscribe to theme changes
   *
   * @param handler - Callback function called when theme changes
   * @returns Unsubscribe function
   */
  public onChange(handler: ThemeChangeHandler): () => void {
    this.observers.add(handler);

    // Return unsubscribe function
    return () => {
      this.observers.delete(handler);
    };
  }

  /**
   * Initialize theme system (load saved or default theme)
   */
  private initializeTheme(): void {
    // Try to load saved theme
    const savedThemeId = this.loadTheme();
    if (savedThemeId && this.themes.has(savedThemeId)) {
      this.setTheme(savedThemeId);
    } else {
      // Default to frosted-glass theme
      this.setTheme('frosted-glass');
    }
  }

  /**
   * Apply a theme to the DOM by injecting CSS variables
   *
   * @param theme - Theme to apply
   */
  private applyTheme(theme: Theme): void {
    const root = document.documentElement;

    // Clear old theme CSS variables (remove all --luxar-* variables)
    this.clearThemeVariables();

    // Remove any existing glass filters and refraction layers
    removeGlassFilters();
    removeGlassRefractionLayers();

    // Clean up existing glass refraction observer
    if (this.glassRefractionObserverCleanup) {
      this.glassRefractionObserverCleanup();
      this.glassRefractionObserverCleanup = null;
    }
    // Cancel any in-flight refraction-layer injection from a previous
    // liquid-glass apply — otherwise it would fire after we switched away.
    if (this.pendingRefractionRAF !== null) {
      cancelAnimationFrame(this.pendingRefractionRAF);
      this.pendingRefractionRAF = null;
    }

    // Set data-theme attribute for theme-specific CSS overrides
    root.setAttribute('data-theme', theme.id);

    // Inject glass distortion filters for Liquid Glass theme
    if (theme.id === 'liquid-glass') {
      injectGlassFilters(); // Uses default params, adjustable in glass-filters.ts

      // Inject real DOM elements for glass refraction (SVG filters don't work on pseudo-elements)
      // Use requestAnimationFrame to ensure DOM is ready. Track the handle so
      // dispose() can cancel a still-pending injection.
      this.pendingRefractionRAF = requestAnimationFrame(() => {
        this.pendingRefractionRAF = null;
        injectGlassRefractionLayers();
      });

      // Set up observer to inject refraction layers for dynamically created panels
      this.glassRefractionObserverCleanup = setupGlassRefractionObserver();
    }

    // Inject CSS custom properties
    this.setCSSVariables(root, theme);
  }

  /**
   * Clear all existing theme CSS variables
   */
  private clearThemeVariables(): void {
    const root = document.documentElement;
    const styles = root.style;

    // Get all CSS properties
    const propertiesToRemove: string[] = [];
    for (let i = 0; i < styles.length; i++) {
      const prop = styles[i];
      if (prop.startsWith('--luxar-')) {
        propertiesToRemove.push(prop);
      }
    }

    // Remove them
    propertiesToRemove.forEach((prop) => {
      root.style.removeProperty(prop);
    });
  }

  /**
   * Build the flat `--luxar-*` -> CSS-value map for a theme.
   *
   * Numbers (font weights, line heights, opacity, z-index) are stringified
   * here. Optional theme fields that resolve to undefined are kept in the
   * map and skipped at write time, so variables only land in the DOM when
   * the theme actually defines them.
   */
  private themeToCSSVariables(theme: Theme): Record<string, string | undefined> {
    return {
      // Background colors
      '--luxar-bg-primary': theme.colors.background.primary,
      '--luxar-bg-secondary': theme.colors.background.secondary,
      '--luxar-bg-tertiary': theme.colors.background.tertiary,
      '--luxar-bg-overlay': theme.colors.background.overlay,

      // Text colors
      '--luxar-text-primary': theme.colors.text.primary,
      '--luxar-text-secondary': theme.colors.text.secondary,
      '--luxar-text-muted': theme.colors.text.muted,
      '--luxar-text-disabled': theme.colors.text.disabled,
      '--luxar-text-inverse': theme.colors.text.inverse,

      // Semantic colors
      '--luxar-success': theme.colors.semantic.success,
      '--luxar-warning': theme.colors.semantic.warning,
      '--luxar-error': theme.colors.semantic.error,
      '--luxar-info': theme.colors.semantic.info,
      '--luxar-highlight': theme.colors.semantic.highlight,

      // Interactive colors
      '--luxar-interactive-default': theme.colors.interactive.default,
      '--luxar-interactive-hover': theme.colors.interactive.hover,
      '--luxar-interactive-active': theme.colors.interactive.active,
      '--luxar-interactive-focus': theme.colors.interactive.focus,
      '--luxar-interactive-disabled': theme.colors.interactive.disabled,

      // Border colors
      '--luxar-border-default': theme.colors.border.default,
      '--luxar-border-subtle': theme.colors.border.subtle,
      '--luxar-border-strong': theme.colors.border.strong,
      '--luxar-border-focus': theme.colors.border.focus,

      // Menu / native dropdown-option colors
      '--luxar-menu-bg': theme.colors.menu.background,
      '--luxar-menu-fg': theme.colors.menu.text,
      '--luxar-menu-active-bg': theme.colors.menu.activeBackground,
      '--luxar-menu-active-fg': theme.colors.menu.activeText,

      // Visualization colors
      '--luxar-viz-hot': theme.colors.visualization.hot,
      '--luxar-viz-warm': theme.colors.visualization.warm,
      '--luxar-viz-cold': theme.colors.visualization.cold,
      '--luxar-viz-neutral': theme.colors.visualization.neutral,

      // Typography — font families
      '--luxar-font-base': theme.typography.fontFamily.base,
      '--luxar-font-mono': theme.typography.fontFamily.mono,
      '--luxar-font-display': theme.typography.fontFamily.display,

      // Typography — font sizes
      '--luxar-text-xs': theme.typography.fontSize.xs,
      '--luxar-text-sm': theme.typography.fontSize.sm,
      '--luxar-text-base': theme.typography.fontSize.base,
      '--luxar-text-md': theme.typography.fontSize.md,
      '--luxar-text-lg': theme.typography.fontSize.lg,
      '--luxar-text-xl': theme.typography.fontSize.xl,
      '--luxar-text-2xl': theme.typography.fontSize['2xl'],
      '--luxar-text-3xl': theme.typography.fontSize['3xl'],
      '--luxar-text-4xl': theme.typography.fontSize['4xl'],

      // Typography — font weights
      '--luxar-font-normal': String(theme.typography.fontWeight.normal),
      '--luxar-font-medium': String(theme.typography.fontWeight.medium),
      '--luxar-font-semibold': String(theme.typography.fontWeight.semibold),
      '--luxar-font-bold': String(theme.typography.fontWeight.bold),

      // Typography — line heights
      '--luxar-line-tight': String(theme.typography.lineHeight.tight),
      '--luxar-line-normal': String(theme.typography.lineHeight.normal),
      '--luxar-line-relaxed': String(theme.typography.lineHeight.relaxed),

      // Spacing
      '--luxar-spacing-0': theme.spacing[0],
      '--luxar-spacing-1': theme.spacing[1],
      '--luxar-spacing-2': theme.spacing[2],
      '--luxar-spacing-3': theme.spacing[3],
      '--luxar-spacing-4': theme.spacing[4],
      '--luxar-spacing-5': theme.spacing[5],
      '--luxar-spacing-6': theme.spacing[6],
      '--luxar-spacing-8': theme.spacing[8],
      '--luxar-spacing-10': theme.spacing[10],
      '--luxar-spacing-12': theme.spacing[12],
      '--luxar-spacing-16': theme.spacing[16],
      '--luxar-spacing-20': theme.spacing[20],

      // Border radius
      '--luxar-radius-none': theme.effects.borderRadius.none,
      '--luxar-radius-sm': theme.effects.borderRadius.sm,
      '--luxar-radius-md': theme.effects.borderRadius.md,
      '--luxar-radius-lg': theme.effects.borderRadius.lg,
      '--luxar-radius-full': theme.effects.borderRadius.full,

      // Shadows
      '--luxar-shadow-sm': theme.effects.shadow.sm,
      '--luxar-shadow-md': theme.effects.shadow.md,
      '--luxar-shadow-lg': theme.effects.shadow.lg,
      '--luxar-shadow-xl': theme.effects.shadow.xl,

      // Blur
      '--luxar-blur-none': theme.effects.blur.none,
      '--luxar-blur-sm': theme.effects.blur.sm,
      '--luxar-blur-md': theme.effects.blur.md,
      '--luxar-blur-lg': theme.effects.blur.lg,

      // Opacity
      '--luxar-opacity-disabled': String(theme.effects.opacity.disabled),
      '--luxar-opacity-secondary': String(theme.effects.opacity.secondary),
      '--luxar-opacity-hover': String(theme.effects.opacity.hover),
      '--luxar-opacity-full': String(theme.effects.opacity.full),

      // Transitions
      '--luxar-transition-fast': theme.effects.transition.fast,
      '--luxar-transition-normal': theme.effects.transition.normal,
      '--luxar-transition-slow': theme.effects.transition.slow,

      // Z-index
      '--luxar-z-base': String(theme.zIndex.base),
      '--luxar-z-dropdown': String(theme.zIndex.dropdown),
      '--luxar-z-modal': String(theme.zIndex.modal),
      '--luxar-z-popover': String(theme.zIndex.popover),
      '--luxar-z-tooltip': String(theme.zIndex.tooltip),
    };
  }

  /**
   * Set CSS custom properties from theme.
   *
   * Iterates the flat variable map from {@link themeToCSSVariables}; entries
   * with `undefined` values are skipped, preserving the existing behavior of
   * not setting properties for unset optional theme fields.
   *
   * @param root - Document root element
   * @param theme - Theme to inject
   */
  private setCSSVariables(root: HTMLElement, theme: Theme): void {
    const variables = this.themeToCSSVariables(theme);
    for (const [property, value] of Object.entries(variables)) {
      if (value !== undefined) {
        root.style.setProperty(property, value);
      }
    }
  }

  /**
   * Notify all observers of theme change
   *
   * @param theme - New theme
   */
  private notifyObservers(theme: Theme): void {
    this.observers.forEach((handler) => {
      try {
        handler(theme);
      } catch (error) {
        log.error(Modules.UI, 'Error in theme change handler', error);
      }
    });
  }

  /**
   * Save current theme to localStorage
   *
   * @param themeId - Theme ID to save
   */
  private saveTheme(themeId: string): void {
    try {
      localStorage.setItem(this.STORAGE_KEY, themeId);
    } catch (error) {
      log.warning(Modules.UI, 'Failed to save theme to localStorage', error);
    }
  }

  /**
   * Load saved theme from localStorage
   *
   * @returns Saved theme ID or null
   */
  private loadTheme(): string | null {
    try {
      return localStorage.getItem(this.STORAGE_KEY);
    } catch (error) {
      log.warning(Modules.UI, 'Failed to load theme from localStorage', error);
      return null;
    }
  }
}
