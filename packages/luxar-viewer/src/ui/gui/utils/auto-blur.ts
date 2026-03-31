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
      // Blur after wheel scrolling stops (debounced)
      let wheelBlurTimer: ReturnType<typeof setTimeout> | undefined;
      eventManager.add(element, 'wheel', () => {
        clearTimeout(wheelBlurTimer);
        wheelBlurTimer = setTimeout(() => element.blur(), 200);
      });
    } else if (element.type === 'number' || element.type === 'text') {
      // Blur on Enter (commit) or Escape (cancel)
      eventManager.add(element, 'keydown', (e: Event) => {
        const keyEvent = e as KeyboardEvent;
        if (keyEvent.key === 'Enter' || keyEvent.key === 'Escape') {
          element.blur();
        }
      });
      // NOTE: Don't blur on mouseup for number inputs - it prevents text editing!
      // The spinner buttons work fine without this, and users need to click to edit.
    }
  } else if (element instanceof HTMLSelectElement) {
    // Blur after change
    eventManager.add(element, 'change', () => {
      setTimeout(() => element.blur(), 10);
    });
  }
}
