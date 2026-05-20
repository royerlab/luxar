/**
 * Pure keyboard-event validation helpers. Currently used only by tests;
 * kept as a documented public surface (referenced in input/README.md).
 *
 * @module input/input-handler/keyboard-validation
 */

/**
 * Validate if keyboard event should trigger dimension navigation.
 *
 * Checks if the pressed key is a navigation key ([, ], or number keys 1-9)
 * and if the event context allows navigation (not typing in input field).
 * This prevents navigation from interfering with text input.
 *
 * @param event - Keyboard event to validate
 * @returns true if event should trigger navigation, false if it should be
 *          ignored (e.g., user is typing in an input field)
 */
export function isNavigationKey(event: KeyboardEvent): boolean {
  // Ignore if typing in input field
  const target = event.target as HTMLElement;
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
    return false;
  }

  // Check for navigation keys
  const navKeys = ['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  return navKeys.includes(event.key);
}

/**
 * Determine if keyboard shortcut should be blocked in current UI context.
 *
 * Checks various conditions that should prevent shortcut execution:
 * - Active modals/dialogs (shortcuts should not leak through)
 * - Text input focus (prevent navigation while typing)
 * - Browser shortcuts (Ctrl/Cmd+S, etc. should pass through)
 *
 * @param event - Keyboard event to check for blocking conditions
 * @param activeModals - Array of modal IDs currently open
 * @returns true if shortcut should be blocked, false if it should proceed
 */
export function shouldBlockShortcut(event: KeyboardEvent, activeModals: string[] = []): boolean {
  // Block if modal is active
  if (activeModals.length > 0) {
    return true;
  }

  // Block if typing in input
  const target = event.target as HTMLElement;
  if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
    return true;
  }

  // Block browser shortcuts
  if (event.metaKey || (event.ctrlKey && ['s', 'o', 'p'].includes(event.key))) {
    return false; // Let browser handle
  }

  return false;
}
