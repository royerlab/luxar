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
import { log, Modules } from '../utils/log';

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

  /** LocalStorage key for theme persistence */
  private readonly STORAGE_KEY = 'luxar-theme';

  /**
   * Private constructor (singleton pattern)
   */
  private constructor() {
    // Register all built-in themes
    this.registerTheme(darkTheme);
    this.registerTheme(lightTheme);
    this.registerTheme(frostedGlassTheme);

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
   * Reset the singleton instance (for testing only)
   *
   * WARNING: This should only be used in test environments to reset state
   * between tests. Never call this in production code.
   *
   * @internal
   */
  public static resetInstance(): void {
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
      console.warn(`[ThemeManager] Theme "${theme.id}" is already registered, overwriting`);
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

    // Apply theme to DOM
    this.applyTheme(theme);

    // Update current theme
    this.currentTheme = theme;

    // Persist to localStorage
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
      // Default to dark theme
      this.setTheme('dark');
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

    // Set data-theme attribute for theme-specific CSS overrides
    root.setAttribute('data-theme', theme.id);

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
   * Set CSS custom properties from theme
   *
   * @param root - Document root element
   * @param theme - Theme to inject
   */
  private setCSSVariables(root: HTMLElement, theme: Theme): void {
    // Background colors
    root.style.setProperty('--luxar-bg-primary', theme.colors.background.primary);
    root.style.setProperty('--luxar-bg-secondary', theme.colors.background.secondary);
    root.style.setProperty('--luxar-bg-tertiary', theme.colors.background.tertiary);
    root.style.setProperty('--luxar-bg-overlay', theme.colors.background.overlay);

    // Text colors
    root.style.setProperty('--luxar-text-primary', theme.colors.text.primary);
    root.style.setProperty('--luxar-text-secondary', theme.colors.text.secondary);
    root.style.setProperty('--luxar-text-muted', theme.colors.text.muted);
    if (theme.colors.text.disabled) {
      root.style.setProperty('--luxar-text-disabled', theme.colors.text.disabled);
    }
    if (theme.colors.text.inverse) {
      root.style.setProperty('--luxar-text-inverse', theme.colors.text.inverse);
    }

    // Semantic colors
    root.style.setProperty('--luxar-success', theme.colors.semantic.success);
    root.style.setProperty('--luxar-warning', theme.colors.semantic.warning);
    root.style.setProperty('--luxar-error', theme.colors.semantic.error);
    root.style.setProperty('--luxar-info', theme.colors.semantic.info);
    if (theme.colors.semantic.highlight) {
      root.style.setProperty('--luxar-highlight', theme.colors.semantic.highlight);
    }

    // Interactive colors
    root.style.setProperty('--luxar-interactive-default', theme.colors.interactive.default);
    root.style.setProperty('--luxar-interactive-hover', theme.colors.interactive.hover);
    root.style.setProperty('--luxar-interactive-active', theme.colors.interactive.active);
    root.style.setProperty('--luxar-interactive-focus', theme.colors.interactive.focus);
    root.style.setProperty('--luxar-interactive-disabled', theme.colors.interactive.disabled);

    // Border colors
    root.style.setProperty('--luxar-border-default', theme.colors.border.default);
    root.style.setProperty('--luxar-border-subtle', theme.colors.border.subtle);
    root.style.setProperty('--luxar-border-strong', theme.colors.border.strong);
    if (theme.colors.border.focus) {
      root.style.setProperty('--luxar-border-focus', theme.colors.border.focus);
    }

    // Visualization colors
    root.style.setProperty('--luxar-viz-hot', theme.colors.visualization.hot);
    root.style.setProperty('--luxar-viz-warm', theme.colors.visualization.warm);
    root.style.setProperty('--luxar-viz-cold', theme.colors.visualization.cold);
    root.style.setProperty('--luxar-viz-neutral', theme.colors.visualization.neutral);

    // Typography - Font families
    root.style.setProperty('--luxar-font-base', theme.typography.fontFamily.base);
    root.style.setProperty('--luxar-font-mono', theme.typography.fontFamily.mono);
    if (theme.typography.fontFamily.display) {
      root.style.setProperty('--luxar-font-display', theme.typography.fontFamily.display);
    }

    // Typography - Font sizes
    root.style.setProperty('--luxar-text-xs', theme.typography.fontSize.xs);
    root.style.setProperty('--luxar-text-sm', theme.typography.fontSize.sm);
    root.style.setProperty('--luxar-text-base', theme.typography.fontSize.base);
    root.style.setProperty('--luxar-text-md', theme.typography.fontSize.md);
    root.style.setProperty('--luxar-text-lg', theme.typography.fontSize.lg);
    root.style.setProperty('--luxar-text-xl', theme.typography.fontSize.xl);
    root.style.setProperty('--luxar-text-2xl', theme.typography.fontSize['2xl']);
    root.style.setProperty('--luxar-text-3xl', theme.typography.fontSize['3xl']);

    // Typography - Font weights
    root.style.setProperty('--luxar-font-normal', theme.typography.fontWeight.normal.toString());
    root.style.setProperty('--luxar-font-medium', theme.typography.fontWeight.medium.toString());
    root.style.setProperty(
      '--luxar-font-semibold',
      theme.typography.fontWeight.semibold.toString()
    );
    root.style.setProperty('--luxar-font-bold', theme.typography.fontWeight.bold.toString());

    // Typography - Line heights
    root.style.setProperty('--luxar-line-tight', theme.typography.lineHeight.tight.toString());
    root.style.setProperty('--luxar-line-normal', theme.typography.lineHeight.normal.toString());
    root.style.setProperty('--luxar-line-relaxed', theme.typography.lineHeight.relaxed.toString());

    // Spacing
    root.style.setProperty('--luxar-spacing-0', theme.spacing[0]);
    root.style.setProperty('--luxar-spacing-1', theme.spacing[1]);
    root.style.setProperty('--luxar-spacing-2', theme.spacing[2]);
    root.style.setProperty('--luxar-spacing-3', theme.spacing[3]);
    root.style.setProperty('--luxar-spacing-4', theme.spacing[4]);
    root.style.setProperty('--luxar-spacing-5', theme.spacing[5]);
    root.style.setProperty('--luxar-spacing-6', theme.spacing[6]);
    root.style.setProperty('--luxar-spacing-8', theme.spacing[8]);
    root.style.setProperty('--luxar-spacing-10', theme.spacing[10]);
    root.style.setProperty('--luxar-spacing-12', theme.spacing[12]);
    root.style.setProperty('--luxar-spacing-16', theme.spacing[16]);
    root.style.setProperty('--luxar-spacing-20', theme.spacing[20]);

    // Border radius
    root.style.setProperty('--luxar-radius-none', theme.effects.borderRadius.none);
    root.style.setProperty('--luxar-radius-sm', theme.effects.borderRadius.sm);
    root.style.setProperty('--luxar-radius-md', theme.effects.borderRadius.md);
    root.style.setProperty('--luxar-radius-lg', theme.effects.borderRadius.lg);
    root.style.setProperty('--luxar-radius-full', theme.effects.borderRadius.full);

    // Shadows
    root.style.setProperty('--luxar-shadow-sm', theme.effects.shadow.sm);
    root.style.setProperty('--luxar-shadow-md', theme.effects.shadow.md);
    root.style.setProperty('--luxar-shadow-lg', theme.effects.shadow.lg);
    root.style.setProperty('--luxar-shadow-xl', theme.effects.shadow.xl);

    // Blur
    root.style.setProperty('--luxar-blur-none', theme.effects.blur.none);
    root.style.setProperty('--luxar-blur-sm', theme.effects.blur.sm);
    root.style.setProperty('--luxar-blur-md', theme.effects.blur.md);
    root.style.setProperty('--luxar-blur-lg', theme.effects.blur.lg);

    // Opacity
    root.style.setProperty('--luxar-opacity-disabled', theme.effects.opacity.disabled.toString());
    root.style.setProperty('--luxar-opacity-secondary', theme.effects.opacity.secondary.toString());
    root.style.setProperty('--luxar-opacity-hover', theme.effects.opacity.hover.toString());
    root.style.setProperty('--luxar-opacity-full', theme.effects.opacity.full.toString());

    // Transitions
    root.style.setProperty('--luxar-transition-fast', theme.effects.transition.fast);
    root.style.setProperty('--luxar-transition-normal', theme.effects.transition.normal);
    root.style.setProperty('--luxar-transition-slow', theme.effects.transition.slow);

    // Z-index
    root.style.setProperty('--luxar-z-base', theme.zIndex.base.toString());
    root.style.setProperty('--luxar-z-dropdown', theme.zIndex.dropdown.toString());
    root.style.setProperty('--luxar-z-modal', theme.zIndex.modal.toString());
    root.style.setProperty('--luxar-z-popover', theme.zIndex.popover.toString());
    root.style.setProperty('--luxar-z-tooltip', theme.zIndex.tooltip.toString());
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
        console.error('[ThemeManager] Error in theme change handler:', error);
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
      console.warn('[ThemeManager] Failed to save theme to localStorage:', error);
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
      console.warn('[ThemeManager] Failed to load theme from localStorage:', error);
      return null;
    }
  }
}
