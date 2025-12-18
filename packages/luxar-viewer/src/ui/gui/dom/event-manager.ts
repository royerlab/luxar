/**
 * Centralized event listener management for memory leak prevention
 *
 * CRITICAL: All controllers MUST use EventManager to track event listeners.
 * This ensures proper cleanup on dispose, preventing memory leaks.
 *
 * @example
 * ```typescript
 * const manager = new EventManager();
 * manager.add(element, 'click', handler);
 * // Later...
 * manager.removeAll(); // Removes ALL tracked listeners
 * ```
 */

import type { EventListenerRecord } from '../core/types';

export class EventManager {
  private listeners: EventListenerRecord[] = [];

  /**
   * Add an event listener and track it for cleanup
   *
   * @param element - Target element
   * @param event - Event name
   * @param handler - Event handler
   * @param options - Event listener options
   */
  public add<K extends keyof HTMLElementEventMap>(
    element: HTMLElement | Window | Document,
    event: K | string,
    handler: EventListener,
    options?: AddEventListenerOptions
  ): void {
    element.addEventListener(event as string, handler, options);

    this.listeners.push({
      element,
      event: event as string,
      handler,
      options,
    });
  }

  /**
   * Remove a specific event listener
   *
   * @param element - Target element
   * @param event - Event name
   * @param handler - Event handler
   */
  public remove(
    element: HTMLElement | Window | Document,
    event: string,
    handler: EventListener
  ): void {
    element.removeEventListener(event, handler);

    this.listeners = this.listeners.filter(
      (record) => record.element !== element || record.event !== event || record.handler !== handler
    );
  }

  /**
   * Remove all tracked event listeners
   *
   * MUST be called on dispose to prevent memory leaks
   */
  public removeAll(): void {
    for (const record of this.listeners) {
      record.element.removeEventListener(record.event, record.handler, record.options);
    }
    this.listeners = [];
  }

  /**
   * Get count of tracked listeners (for debugging/testing)
   */
  public count(): number {
    return this.listeners.length;
  }
}
