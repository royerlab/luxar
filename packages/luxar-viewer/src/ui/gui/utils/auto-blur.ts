/**
 * Auto-blur utilities for input elements
 *
 * Prevents focus from getting stuck on inputs, which breaks keyboard shortcuts.
 */

import { EventManager } from '../dom/event-manager';

/**
 * Apply auto-blur behavior to an input element
 *
 * Different input types have different blur triggers:
 * - Checkboxes: Blur immediately after change
 * - Sliders: Blur on mouseup/touchend
 * - Number/Text inputs: Blur on Enter key
 * - Selects: Blur after change
 *
 * @param element - Input element to apply auto-blur to
 * @param eventManager - Event manager for tracking listeners
 */
export function applyAutoBlur(
  element: HTMLInputElement | HTMLSelectElement,
  eventManager: EventManager
): void {
  if (element instanceof HTMLInputElement) {
    if (element.type === 'checkbox') {
      // Blur immediately after change
      eventManager.add(element, 'change', () => {
        setTimeout(() => element.blur(), 10);
      });
      // Also blur on click for checkboxes
      eventManager.add(element, 'click', () => {
        setTimeout(() => element.blur(), 10);
      });
    } else if (element.type === 'range') {
      // Blur on mouseup
      eventManager.add(element, 'mouseup', () => {
        setTimeout(() => element.blur(), 10);
      });
      // And on touchend for mobile
      eventManager.add(element, 'touchend', () => {
        setTimeout(() => element.blur(), 10);
      });
    } else if (element.type === 'number' || element.type === 'text') {
      // Blur on Enter key
      eventManager.add(element, 'keydown', (e: Event) => {
        const keyEvent = e as KeyboardEvent;
        if (keyEvent.key === 'Enter') {
          element.blur();
        }
      });
      // Also blur on mouseup for number input arrows
      if (element.type === 'number') {
        eventManager.add(element, 'mouseup', () => {
          setTimeout(() => element.blur(), 10);
        });
      }
    }
  } else if (element instanceof HTMLSelectElement) {
    // Blur after change
    eventManager.add(element, 'change', () => {
      setTimeout(() => element.blur(), 10);
    });
  }
}
