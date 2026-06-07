/**
 * UIComponent Base Class
 *
 * Abstract base class for all UI components with managed lifecycle.
 * Provides automatic event listener cleanup and theme subscription.
 *
 * Key features:
 * - Managed event listeners (prevents memory leaks!)
 * - Theme subscription (automatic theme change handling)
 * - Lifecycle methods (render, attachEventListeners, onThemeChange, onDispose)
 * - Visibility management (show, hide, toggle)
 * - Disposal pattern (cleanup all resources)
 */

import { ThemeManager } from '../../themes/theme-manager';
import { getViewerContainer } from '../../utils/viewer-container';
import type { Theme } from '../../themes/types';

/**
 * Abstract base class for UI components
 *
 * All UI components should extend this class to get automatic
 * event listener cleanup and theme subscription.
 *
 * @example
 * ```typescript
 * class MyComponent extends UIComponent<MyConfig> {
 *   protected getClassName(): string {
 *     return 'luxar-my-component';
 *   }
 *
 *   protected render(): HTMLElement {
 *     const el = document.createElement('div');
 *     el.className = this.getClassName();
 *     return el;
 *   }
 *
 *   protected attachEventListeners(): void {
 *     // Use this.addEventListener for automatic cleanup!
 *     this.addEventListener(this.element, 'click', this.handleClick);
 *   }
 *
 *   private handleClick = (e: MouseEvent) => {
 *     // Event handler logic
 *   };
 * }
 * ```
 */
export abstract class UIComponent<TConfig = unknown> {
  protected element: HTMLElement;
  protected config: TConfig;

  // Managed event listeners - automatic cleanup!
  private eventListeners: Map<EventTarget, Map<string, EventListenerOrEventListenerObject>> =
    new Map();

  // Theme subscription
  private themeUnsubscribe: (() => void) | null = null;

  /**
   * Construct a new UI component
   *
   * @param config - Component configuration
   */
  constructor(config: TConfig) {
    this.config = config;
    this.element = this.render();
    this.attachEventListeners();
    this.subscribeToTheme();
  }

  // ==================== Abstract Methods ====================
  // Subclasses MUST implement these

  /**
   * Render the component DOM structure.
   * Should return a root element with appropriate class names.
   * NO INLINE STYLES - use CSS classes only!
   *
   * @returns Root element for this component
   */
  protected abstract render(): HTMLElement;

  /**
   * Get the base CSS class name for this component.
   * Used for BEM-style naming.
   *
   * @returns Base CSS class name (e.g., 'luxar-error-dialog')
   */
  protected abstract getClassName(): string;

  // ==================== Optional Lifecycle Hooks ====================

  /**
   * Attach event listeners to the component.
   * Use this.addEventListener() for automatic cleanup!
   *
   * Called automatically after render().
   */
  protected attachEventListeners(): void {
    // Override in subclass if needed
  }

  /**
   * Called when theme changes.
   * Override for theme-specific logic (e.g., canvas colors).
   *
   * @param _theme - New theme
   */
  protected onThemeChange(_theme: Theme): void {
    // Override in subclass if needed
  }

  /**
   * Called before disposal.
   * Override for custom cleanup logic.
   *
   * Called automatically by dispose().
   */
  protected onDispose(): void {
    // Override in subclass if needed
  }

  // ==================== Managed Event Listeners ====================

  /**
   * Add an event listener with automatic cleanup tracking.
   * Prevents memory leaks - all listeners removed on dispose()!
   *
   * IMPORTANT: Use arrow functions for event handlers to maintain component context:
   * ```typescript
   * private handleClick = (ev: MouseEvent) => {
   *   // 'this' refers to component instance
   *   this.someMethod();
   * };
   *
   * protected attachEventListeners(): void {
   *   this.addEventListener(this.element, 'click', this.handleClick);
   * }
   * ```
   *
   * @param target - Event target (element, window, document, etc.)
   * @param type - Event type ('click', 'keydown', etc.)
   * @param listener - Event listener function (should be arrow function)
   * @param options - Optional event listener options
   */
  protected addEventListener<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    type: K,
    listener: (ev: HTMLElementEventMap[K]) => void,
    options?: AddEventListenerOptions
  ): void {
    // Store for cleanup
    if (!this.eventListeners.has(target)) {
      this.eventListeners.set(target, new Map());
    }

    // Remove existing listener if present (prevent duplicates)
    const existingListener = this.eventListeners.get(target)?.get(type);
    if (existingListener) {
      target.removeEventListener(type, existingListener);
    }

    // Store listener directly (no binding needed for arrow functions)
    const listenerFn = listener as EventListener;
    this.eventListeners.get(target)!.set(type, listenerFn);
    target.addEventListener(type, listenerFn, options);
  }

  /**
   * Add a generic event listener (for non-standard events)
   *
   * @param target - Event target
   * @param type - Event type (string for custom events)
   * @param listener - Event listener function
   * @param options - Optional event listener options
   */
  protected addEventListenerGeneric(
    target: EventTarget,
    type: string,
    listener: EventListener,
    options?: AddEventListenerOptions
  ): void {
    // Store for cleanup
    if (!this.eventListeners.has(target)) {
      this.eventListeners.set(target, new Map());
    }

    // Store and add listener
    this.eventListeners.get(target)!.set(type, listener);
    target.addEventListener(type, listener, options);
  }

  /**
   * Remove a specific event listener.
   *
   * @param target - Event target
   * @param type - Event type
   */
  protected removeEventListener<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    type: K
  ): void {
    const listeners = this.eventListeners.get(target);
    if (!listeners) return;

    const listener = listeners.get(type);
    if (listener) {
      target.removeEventListener(type, listener);
      listeners.delete(type);
    }
  }

  // ==================== Theme Management ====================

  /**
   * Subscribe to theme changes
   */
  private subscribeToTheme(): void {
    this.themeUnsubscribe = ThemeManager.getInstance().onChange((theme) => {
      this.onThemeChange(theme);
    });
  }

  // ==================== Public API ====================

  /**
   * Show the component (add to DOM and apply visible class).
   */
  public show(): void {
    if (!this.element.parentNode) {
      getViewerContainer().appendChild(this.element);
    }
    this.element.classList.add(`${this.getClassName()}--visible`);
  }

  /**
   * Hide the component (remove visible class, keep in DOM).
   */
  public hide(): void {
    this.element.classList.remove(`${this.getClassName()}--visible`);
  }

  /**
   * Toggle visibility.
   */
  public toggle(): void {
    const isVisible = this.element.classList.contains(`${this.getClassName()}--visible`);
    if (isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Check if component is visible.
   *
   * @returns True if visible
   */
  public isVisible(): boolean {
    return this.element.classList.contains(`${this.getClassName()}--visible`);
  }

  /**
   * Get the root element of this component.
   *
   * @returns Root element
   */
  public getElement(): HTMLElement {
    return this.element;
  }

  /**
   * Dispose the component - cleanup all resources.
   * NO MORE MEMORY LEAKS!
   *
   * Automatically:
   * - Removes all tracked event listeners
   * - Unsubscribes from theme changes
   * - Calls onDispose() for custom cleanup
   * - Removes element from DOM
   */
  public dispose(): void {
    // 1. Remove all tracked event listeners
    for (const [target, listeners] of this.eventListeners) {
      for (const [type, listener] of listeners) {
        target.removeEventListener(type, listener);
      }
    }
    this.eventListeners.clear();

    // 2. Unsubscribe from theme changes
    if (this.themeUnsubscribe) {
      this.themeUnsubscribe();
      this.themeUnsubscribe = null;
    }

    // 3. Custom cleanup
    this.onDispose();

    // 4. Remove from DOM
    this.element.remove();
  }
}
