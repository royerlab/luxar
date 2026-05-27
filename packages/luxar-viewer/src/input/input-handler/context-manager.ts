/**
 * Input Context Manager for Luxar
 *
 * This module manages different input contexts to prevent key binding conflicts
 * and ensure proper input handling based on the current application state.
 *
 * Key features:
 * - Context-aware key binding registration
 * - Context stack for nested contexts
 * - Automatic conflict detection
 * - Priority-based key handling
 */

import { config } from '../../config';
import { log, Modules, LogEmoji } from '../../utils/log';
import {
  isKeyAllowedInContext as isKeyAllowedInContextPure,
  sortContextsByPriority,
} from './context-manager/routing-rules';
import { isTypingInInput } from './commands/focus-utils';

/**
 * Maximum recursion depth for {@link InputContextManager.handleKeyEvent}.
 * A binding handler that (re-)dispatches a keyboard event through this
 * manager would otherwise recurse forever; this cap limits the
 * blast radius to a finite stack and surfaces the misconfiguration
 * via a single `log.error`.
 *
 * 10 is comfortably above any realistic UI depth (the deepest
 * documented passthrough chain is `TYPING → UI_INTERACTION →
 * NAVIGATION`, depth 3).
 */
export const MAX_KEY_EVENT_DEPTH = 10;

/**
 * Available input contexts
 */
export enum InputContext {
  NAVIGATION = 'navigation', // Default 3D navigation mode
  FLY_CONTROLS = 'fly_controls', // Fly mode with WASD movement
  TYPING = 'typing', // Text input (forms, search, etc.)
  UI_INTERACTION = 'ui_interaction', // UI panels and controls
  DIMENSION_NAV = 'dimension_nav', // nD dimension navigation
}

/**
 * Key binding configuration
 */
export interface KeyBinding {
  key: string;
  modifiers?: {
    ctrl?: boolean;
    shift?: boolean;
    alt?: boolean;
    meta?: boolean;
  };
  handler: (event: KeyboardEvent) => void;
  keyupHandler?: (event: KeyboardEvent) => void; // Optional separate handler for keyup events
  preventDefault?: boolean;
  description?: string;
}

/**
 * Context configuration
 */
export interface ContextConfig {
  name: string;
  priority: number; // Higher priority contexts override lower ones
  allowedKeys?: string[]; // If specified, only these keys are handled
  blockedKeys?: string[]; // These keys are never handled in this context
  passthrough?: boolean; // If true, unhandled keys pass to lower contexts
}

/**
 * Input Context Manager
 * Manages keyboard input routing based on application context
 */
/**
 * Manages context-aware keyboard input routing to prevent conflicts.
 *
 * Provides a hierarchical context system where different parts of the UI
 * can register key bindings without conflicting. For example, WASD keys
 * are blocked in navigation mode but enabled in fly control mode.
 *
 * Key features:
 * - Priority-based context system (higher priority contexts take precedence)
 * - Context stack for nested contexts (modal over main view)
 * - Automatic typing detection (blocks shortcuts when typing in inputs)
 * - Passthrough support (unhandled keys pass to lower priority contexts)
 *
 * @example
 * ```typescript
 * const manager = new InputContextManager();
 *
 * // Register a key binding for navigation context
 * manager.registerBinding(InputContext.NAVIGATION, {
 *   key: '[',
 *   handler: () => navigateBackward(),
 *   preventDefault: true,
 *   description: 'Navigate backward in dimension'
 * });
 *
 * // Switch to fly controls context
 * manager.setContext(InputContext.FLY_CONTROLS);
 * // Now WASD keys are enabled, [ ] keys still work via passthrough
 * ```
 */
export class InputContextManager {
  private currentContext: InputContext = InputContext.NAVIGATION;
  private contextStack: InputContext[] = [];
  private bindings = new Map<string, Map<string, KeyBinding>>();
  private contextConfigs = new Map<InputContext, ContextConfig>();
  private enabled = true;

  /**
   * Re-entrance depth for `handleKeyEvent`. A binding handler that
   * (mis)configured itself to dispatch keyboard events back through
   * the context manager could otherwise recurse infinitely; cap at
   * {@link MAX_KEY_EVENT_DEPTH} and bail with a single error log.
   */
  private keyEventDepth = 0;

  /**
   * Create a new input context manager with default context configurations.
   *
   * Initializes all predefined contexts (NAVIGATION, FLY_CONTROLS, TYPING,
   * UI_INTERACTION, DIMENSION_NAV) with appropriate priorities and key filters.
   * Starts in NAVIGATION context.
   */
  constructor() {
    this.initializeContexts();
  }

  /**
   * Initialize default context configurations with priorities and key filters.
   *
   * Sets up five predefined contexts:
   * - NAVIGATION (priority 0): Default mode, blocks WASD keys
   * - FLY_CONTROLS (priority 1): Enables WASD + arrow keys for fly mode
   * - TYPING (priority 10): Highest priority, blocks all shortcuts
   * - UI_INTERACTION (priority 5): For UI panels
   * - DIMENSION_NAV (priority 2): For dimension navigation keys
   *
   * @private
   */
  private initializeContexts(): void {
    // Navigation context - default mode
    // Block WASD keys but NOT Shift (Shift needed for FOV control in orbit mode)
    const flyModeKeysWithoutShift = config.input.keyboard.flyModeKeys.filter((k) => k !== 'Shift');

    this.contextConfigs.set(InputContext.NAVIGATION, {
      name: 'Navigation',
      priority: 0,
      passthrough: true,
      blockedKeys: [...flyModeKeysWithoutShift], // Block WASD but allow Shift
    });

    // Fly controls context - WASD movement active
    this.contextConfigs.set(InputContext.FLY_CONTROLS, {
      name: 'Fly Controls',
      priority: 1,
      passthrough: true,
      allowedKeys: [
        ...config.input.keyboard.flyModeKeys,
        'ArrowUp',
        'ArrowDown',
        'ArrowLeft',
        'ArrowRight',
      ],
    });

    // Typing context - highest priority, blocks most shortcuts
    this.contextConfigs.set(InputContext.TYPING, {
      name: 'Typing',
      priority: 10,
      passthrough: false,
      allowedKeys: [], // No shortcuts while typing
    });

    // UI interaction context
    this.contextConfigs.set(InputContext.UI_INTERACTION, {
      name: 'UI Interaction',
      priority: 5,
      passthrough: true,
    });

    // Dimension navigation context
    this.contextConfigs.set(InputContext.DIMENSION_NAV, {
      name: 'Dimension Navigation',
      priority: 2,
      passthrough: true,
      allowedKeys: [...config.input.keyboard.dimensionKeys],
    });
  }

  /**
   * Push a new context onto the stack, saving current context.
   *
   * Used for nested contexts like modal dialogs over main view. The previous
   * context is saved and will be restored when popContext() is called. If
   * the new context is the same as current, does nothing.
   *
   * @param context - Context to activate and push onto stack
   *
   * @example
   * ```typescript
   * // Open modal dialog
   * contextManager.pushContext(InputContext.UI_INTERACTION);
   * // Now UI has higher priority
   *
   * // Close modal dialog
   * contextManager.popContext();
   * // Back to previous context
   * ```
   */
  public pushContext(context: InputContext): void {
    if (this.currentContext !== context) {
      this.contextStack.push(this.currentContext);
      this.setContext(context);
    }
  }

  /**
   * Pop the current context and restore the previous one from stack.
   *
   * Used to return to previous context after closing modal/dialog. If stack
   * is empty, does nothing (stays in current context).
   *
   * @example
   * ```typescript
   * // Open settings dialog
   * contextManager.pushContext(InputContext.UI_INTERACTION);
   * // ... user interacts with settings ...
   * contextManager.popContext();  // Back to NAVIGATION
   * ```
   */
  public popContext(): void {
    const previous = this.contextStack.pop();
    if (previous) {
      this.setContext(previous);
    }
  }

  /**
   * Set the current context, replacing current without saving to stack.
   *
   * Use this for mode switches (navigation → fly controls) rather than
   * nested contexts. The previous context is NOT saved - use pushContext()
   * if you need to restore the previous context later.
   *
   * Logs context change for debugging.
   *
   * @param context - Context to switch to
   *
   * @example
   * ```typescript
   * // Switch to fly control mode
   * contextManager.setContext(InputContext.FLY_CONTROLS);
   * // WASD keys now enabled, can't go back with popContext()
   *
   * // Switch back to navigation
   * contextManager.setContext(InputContext.NAVIGATION);
   * ```
   */
  public setContext(context: InputContext): void {
    const oldContext = this.currentContext;
    this.currentContext = context;

    // Log context change for debugging
    log.custom(
      LogEmoji.CONTROLS,
      Modules.INPUT,
      `Input context changed: ${oldContext} → ${context}`
    );
  }

  /**
   * Get the currently active input context.
   *
   * @returns Current context enum value (NAVIGATION, FLY_CONTROLS, etc.)
   */
  public getContext(): InputContext {
    return this.currentContext;
  }

  /**
   * Register a key binding for a specific context.
   *
   * Associates a key (with optional modifiers) to a handler function within
   * a specific context. The binding will only be active when that context is
   * current. Warns if the binding conflicts with an existing binding.
   *
   * @param context - Context where this binding should be active
   * @param binding - Key binding configuration with key, modifiers, and handler
   * @param binding.key - Key to bind (e.g., '[', 'w', 'Escape')
   * @param binding.modifiers - Optional modifiers (ctrl, shift, alt, meta)
   * @param binding.handler - Function to call when key is pressed
   * @param binding.preventDefault - If true, calls event.preventDefault()
   * @param binding.description - Optional description for debugging/help
   *
   * @example
   * ```typescript
   * // Register [ key for backward navigation
   * manager.registerBinding(InputContext.NAVIGATION, {
   *   key: '[',
   *   handler: () => navigateBackward(),
   *   preventDefault: true,
   *   description: 'Navigate backward'
   * });
   *
   * // Register Ctrl+S for save (with modifier)
   * manager.registerBinding(InputContext.UI_INTERACTION, {
   *   key: 's',
   *   modifiers: { ctrl: true },
   *   handler: () => save(),
   *   preventDefault: true
   * });
   * ```
   */
  public registerBinding(context: InputContext, binding: KeyBinding): void {
    const contextKey = context;
    if (!this.bindings.has(contextKey)) {
      this.bindings.set(contextKey, new Map());
    }

    const bindingKey = this.getBindingKey(binding);
    const contextBindings = this.bindings.get(contextKey)!;

    // Check for conflicts
    if (contextBindings.has(bindingKey)) {
      log.warning(
        Modules.INPUT_CONTEXT,
        `Key binding conflict in ${context}: ${bindingKey} is already registered`
      );
    }

    contextBindings.set(bindingKey, binding);
  }

  /**
   * Unregister a previously registered key binding.
   *
   * Removes the binding for the specified key and modifiers in the given
   * context. Has no effect if the binding doesn't exist.
   *
   * @param context - Context containing the binding to remove
   * @param key - Key that was bound
   * @param modifiers - Optional modifiers that were bound
   *
   * @example
   * ```typescript
   * // Remove [ key binding
   * manager.unregisterBinding(InputContext.NAVIGATION, '[');
   *
   * // Remove Ctrl+S binding
   * manager.unregisterBinding(
   *   InputContext.UI_INTERACTION,
   *   's',
   *   { ctrl: true }
   * );
   * ```
   */
  public unregisterBinding(
    context: InputContext,
    key: string,
    modifiers?: KeyBinding['modifiers']
  ): void {
    const contextBindings = this.bindings.get(context);
    if (contextBindings) {
      const bindingKey = this.getBindingKey({ key, modifiers } as KeyBinding);
      contextBindings.delete(bindingKey);
    }
  }

  /**
   * Handle a keyboard event with context-aware routing.
   *
   * Routes the event through the context system to find and execute the
   * appropriate handler. Processing order:
   * 1. Check if manager is enabled
   * 2. Check if in typing context (blocks most keys)
   * 3. Check if key is allowed in current context
   * 4. Look for registered binding in current context
   * 5. If passthrough enabled, try lower priority contexts
   *
   * @param event - Keyboard event to handle
   * @param type - Event type ('down' for keydown, 'up' for keyup)
   * @returns true if event was handled by a binding, false if not handled
   *          (return value indicates whether to prevent default behavior)
   *
   * @example
   * ```typescript
   * // In event listener
   * document.addEventListener('keydown', (event) => {
   *   const handled = contextManager.handleKeyEvent(event, 'down');
   *   if (handled) {
   *     // Event was handled by context system
   *     console.log('Key handled by context manager');
   *   } else {
   *     // No handler found, let it propagate
   *     console.log('Key not handled, continuing...');
   *   }
   * });
   * ```
   */
  public handleKeyEvent(event: KeyboardEvent, type: 'down' | 'up'): boolean {
    if (!this.enabled) return false;

    // Re-entrance guard: a misbehaving binding handler that triggers
    // another keyboard event through this manager could otherwise
    // recurse indefinitely. Cap at MAX_KEY_EVENT_DEPTH and bail.
    if (this.keyEventDepth >= MAX_KEY_EVENT_DEPTH) {
      log.error(
        Modules.INPUT,
        `handleKeyEvent recursion limit (${MAX_KEY_EVENT_DEPTH}) reached for key '${event.key}'; ` +
          'a binding handler is dispatching keyboard events back through the context manager.'
      );
      return false;
    }
    this.keyEventDepth++;
    try {
      return this.handleKeyEventInternal(event, type);
    } finally {
      this.keyEventDepth--;
    }
  }

  private handleKeyEventInternal(event: KeyboardEvent, type: 'down' | 'up'): boolean {
    // Check if we're in a typing context
    if (this.isTypingContext()) {
      // Escape from a typing context (e.g. focus inside the
      // dataset-browser manual-path field, debug-console filter input)
      // must still close the panel. Look up the Escape binding in the
      // current context AND every other context, firing the first
      // match. Going through the normal dispatch path would re-enter
      // this branch, and `tryLowerContexts` alone would skip the
      // current context where Escape is usually registered.
      if (event.key === 'Escape') {
        return this.dispatchEscapeFromTypingContext(event, type);
      }
      // Block all other keys while typing
      return true;
    }

    // Get the current context configuration
    const config = this.contextConfigs.get(this.currentContext);
    if (!config) return false;

    // Check if this key is allowed in the current context
    if (!this.isKeyAllowedInContext(event.key, config)) {
      // Key not allowed in this context - try passthrough if enabled
      if (config.passthrough) {
        return this.tryLowerContexts(event, type);
      }
      return false;
    }

    // Find and execute the binding
    const contextBindings = this.bindings.get(this.currentContext);
    if (contextBindings) {
      const bindingKey = this.getBindingKeyFromEvent(event);
      const binding = contextBindings.get(bindingKey);

      if (binding) {
        // Handle keydown and keyup separately
        if (type === 'up') {
          // On keyup: ONLY call keyupHandler if it exists
          if (binding.keyupHandler) {
            if (binding.preventDefault) {
              event.preventDefault();
            }
            binding.keyupHandler(event);
            return true;
          }
          // No keyupHandler = this binding doesn't handle keyup
          return false;
        } else {
          // On keydown: call main handler
          if (binding.preventDefault) {
            event.preventDefault();
          }
          binding.handler(event);
          return true;
        }
      }
    }

    // Key is allowed but no binding found - try passthrough if enabled
    if (config.passthrough) {
      return this.tryLowerContexts(event, type);
    }

    return false;
  }

  /**
   * Check if a key is allowed in the given context based on filters.
   *
   * Checks both blockedKeys and allowedKeys filters:
   * - If key is in blockedKeys: returns false
   * - If allowedKeys is defined and key is not in it: returns false
   * - Otherwise: returns true
   *
   * @param key - Key to check (lowercase string)
   * @param config - Context configuration with key filters
   * @returns true if key is allowed in this context, false if blocked
   * @private
   */
  private isKeyAllowedInContext(key: string, config: ContextConfig): boolean {
    return isKeyAllowedInContextPure(key, config);
  }

  /**
   * Try to handle event in lower priority contexts (passthrough mechanism).
   *
   * When current context doesn't handle a key and has passthrough enabled,
   * this method tries other contexts in descending priority order. Enables
   * fallback behavior - e.g., [ ] keys work in FLY_CONTROLS context even
   * though they're not registered there, because they fall through to
   * DIMENSION_NAV context.
   *
   * @param event - Keyboard event to handle
   * @param type - Event type ('down' or 'up')
   * @returns true if any lower context handled the event, false otherwise
   * @private
   */
  /**
   * Dispatch Escape from a typing context.
   *
   * Walks all contexts in priority order (including the current one)
   * and fires the first matching Escape binding. Mirrors the dispatch
   * shape of {@link tryLowerContexts} but does not exclude the current
   * context — Escape is most often registered in NAVIGATION (the
   * default current context), so excluding the current context like
   * `tryLowerContexts` does would skip it.
   */
  private dispatchEscapeFromTypingContext(event: KeyboardEvent, type: 'down' | 'up'): boolean {
    const sortedContexts = Array.from(this.contextConfigs.entries()).sort(
      (a, b) => (b[1].priority ?? 0) - (a[1].priority ?? 0)
    );

    for (const [context] of sortedContexts) {
      const contextBindings = this.bindings.get(context);
      if (!contextBindings) continue;
      const bindingKey = this.getBindingKeyFromEvent(event);
      const binding = contextBindings.get(bindingKey);
      if (!binding) continue;

      // Route both `handler` and `keyupHandler` through this dispatch
      // path. If the binding only declares a keyupHandler (no main
      // handler is meaningful for keyup), keep searching; otherwise
      // any future Escape-on-keyup feature would silently fail to
      // route from a typing context.
      if (type === 'up') {
        if (binding.keyupHandler) {
          if (binding.preventDefault) event.preventDefault();
          binding.keyupHandler(event);
          return true;
        }
        continue;
      }
      if (binding.preventDefault) event.preventDefault();
      binding.handler(event);
      return true;
    }

    return false;
  }

  private tryLowerContexts(event: KeyboardEvent, type: 'down' | 'up'): boolean {
    const sortedContexts = sortContextsByPriority(this.contextConfigs, this.currentContext);

    for (const [context, config] of sortedContexts) {
      if (this.isKeyAllowedInContext(event.key, config)) {
        const contextBindings = this.bindings.get(context);
        if (contextBindings) {
          const bindingKey = this.getBindingKeyFromEvent(event);
          const binding = contextBindings.get(bindingKey);

          if (binding) {
            // Same keyupHandler logic as main handleKeyEvent
            if (type === 'up') {
              // On keyup: only call keyupHandler if it exists
              if (binding.keyupHandler) {
                if (binding.preventDefault) {
                  event.preventDefault();
                }
                binding.keyupHandler(event);
                return true;
              }
              // No keyupHandler = doesn't handle keyup
              return false;
            } else {
              // On keydown: call main handler
              if (binding.preventDefault) {
                event.preventDefault();
              }
              binding.handler(event);
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  /**
   * Check if currently in a typing context (should block shortcuts).
   *
   * Returns true if:
   * - Current context is explicitly set to TYPING
   * - Focus is in a text input, textarea, select, or contenteditable element
   *
   * Used to prevent keyboard shortcuts from interfering with text entry.
   * For example, prevents 'p' key from toggling performance stats while
   * user is typing "apple" in a search box.
   *
   * @returns true if in typing context, false otherwise
   * @private
   */
  private isTypingContext(): boolean {
    if (this.currentContext === InputContext.TYPING) {
      return true;
    }

    // Delegate the DOM-focus check to the canonical typing-detection
    // helper so this path cannot diverge from InputHandler's
    // isTypingInInput(). Both call sites read the same classification
    // of activeElement (text inputs, textarea, select, contenteditable;
    // excludes range/checkbox/radio).
    return isTypingInInput(document.activeElement);
  }

  /**
   * Generate unique string key for a key binding (for Map storage).
   *
   * Combines key and modifiers into a sorted string representation.
   * Format: "key+mod1+mod2" (alphabetically sorted modifiers).
   * Used as key in Map to store and lookup bindings.
   *
   * @param binding - Key binding configuration
   * @returns Unique string key (e.g., "w", "[", "s+ctrl", "z+ctrl+shift")
   * @private
   *
   * @example
   * ```typescript
   * // Simple key
   * const key1 = getBindingKey({ key: '[', handler: () => {} });
   * console.log(key1); // "["
   *
   * // Key with modifiers (always sorted)
   * const key2 = getBindingKey({
   *   key: 's',
   *   modifiers: { ctrl: true, shift: true },
   *   handler: () => {}
   * });
   * console.log(key2); // "ctrl+s+shift" (sorted alphabetically)
   * ```
   */
  private getBindingKey(binding: KeyBinding): string {
    const parts = [binding.key.toLowerCase()];

    if (binding.modifiers) {
      if (binding.modifiers.ctrl) parts.push('ctrl');
      if (binding.modifiers.shift) parts.push('shift');
      if (binding.modifiers.alt) parts.push('alt');
      if (binding.modifiers.meta) parts.push('meta');
    }

    return parts.sort().join('+');
  }

  /**
   * Generate binding key from keyboard event for lookup.
   *
   * Converts KeyboardEvent to the same string format as getBindingKey()
   * for Map lookup. Checks modifier properties (ctrlKey, shiftKey, etc.)
   * and combines with key in sorted format.
   *
   * @param event - Keyboard event to convert
   * @returns Unique string key matching getBindingKey() format
   * @private
   *
   * @example
   * ```typescript
   * // Event with Ctrl+S
   * const event = new KeyboardEvent('keydown', {
   *   key: 's',
   *   ctrlKey: true
   * });
   * const key = getBindingKeyFromEvent(event);
   * console.log(key); // "ctrl+s"
   * ```
   */
  private getBindingKeyFromEvent(event: KeyboardEvent): string {
    const key = event.key.toLowerCase();
    const parts = [key];

    // When the pressed key IS a modifier (Shift/Control/Alt/Meta), the
    // corresponding modifier flag is also `true` on the keydown event.
    // Adding it again would produce "shift+shift", which never matches
    // the registered "shift" binding and breaks modifier-only handlers.
    if (event.ctrlKey && key !== 'control') parts.push('ctrl');
    if (event.shiftKey && key !== 'shift') parts.push('shift');
    if (event.altKey && key !== 'alt') parts.push('alt');
    if (event.metaKey && key !== 'meta') parts.push('meta');

    return parts.sort().join('+');
  }

  /**
   * Enable or disable the entire context manager.
   *
   * When disabled, handleKeyEvent() immediately returns false without
   * processing. Useful for temporarily suspending all context-based
   * input handling (e.g., during initialization or modal dialogs that
   * need to bypass the context system).
   *
   * @param enabled - true to enable context management, false to disable
   */
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Get debug information about current context manager state.
   *
   * Returns snapshot of current state for debugging and diagnostics.
   * Useful for understanding why a key isn't working or what context
   * is active.
   *
   * @returns Object containing:
   *          - currentContext: Active context
   *          - contextStack: Stack of pushed contexts
   *          - registeredBindings: Map of context → binding keys
   *
   * @example
   * ```typescript
   * const debug = contextManager.getDebugInfo();
   * console.log('Current context:', debug.currentContext);
   * console.log('Context stack:', debug.contextStack);
   * console.log('Bindings in NAVIGATION:', debug.registeredBindings.get(InputContext.NAVIGATION));
   * // Output:
   * // Current context: navigation
   * // Context stack: []
   * // Bindings in NAVIGATION: ['[', ']', '1+ctrl', '2+ctrl']
   * ```
   */
  public getDebugInfo(): {
    currentContext: InputContext;
    contextStack: InputContext[];
    registeredBindings: Map<InputContext, string[]>;
  } {
    const registeredBindings = new Map<InputContext, string[]>();

    this.bindings.forEach((bindings, context) => {
      registeredBindings.set(context as InputContext, Array.from(bindings.keys()));
    });

    return {
      currentContext: this.currentContext,
      contextStack: [...this.contextStack],
      registeredBindings,
    };
  }

  /**
   * Clear all registered key bindings for a specific context.
   *
   * Removes all bindings associated with the specified context. Useful
   * when dynamically changing context configuration or cleaning up
   * temporary bindings.
   *
   * @param context - Context whose bindings should be cleared
   */
  public clearContextBindings(context: InputContext): void {
    this.bindings.delete(context);
  }

  /**
   * Reset context manager to initial state.
   *
   * Clears all registered bindings, empties context stack, and returns
   * to NAVIGATION context. Useful when reinitializing the application
   * or cleaning up for testing.
   *
   * Does NOT reset context configurations (those remain from initialization).
   */
  public reset(): void {
    this.currentContext = InputContext.NAVIGATION;
    this.contextStack = [];
    this.bindings.clear();
  }
}
