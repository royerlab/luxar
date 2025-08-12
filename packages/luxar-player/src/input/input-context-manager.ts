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

import { INPUT_CONFIG } from '../controls/control-config';

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
export class InputContextManager {
  private currentContext: InputContext = InputContext.NAVIGATION;
  private contextStack: InputContext[] = [];
  private bindings = new Map<string, Map<string, KeyBinding>>();
  private contextConfigs = new Map<InputContext, ContextConfig>();
  private enabled = true;

  constructor() {
    this.initializeContexts();
  }

  /**
   * Initialize default context configurations
   */
  private initializeContexts(): void {
    // Navigation context - default mode
    this.contextConfigs.set(InputContext.NAVIGATION, {
      name: 'Navigation',
      priority: 0,
      passthrough: true,
      blockedKeys: [...INPUT_CONFIG.keyboard.flyModeKeys], // Block WASD in orbit mode
    });

    // Fly controls context - WASD movement active
    this.contextConfigs.set(InputContext.FLY_CONTROLS, {
      name: 'Fly Controls',
      priority: 1,
      passthrough: true,
      allowedKeys: [
        ...INPUT_CONFIG.keyboard.flyModeKeys,
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
      allowedKeys: [...INPUT_CONFIG.keyboard.dimensionKeys],
    });
  }

  /**
   * Push a new context onto the stack
   */
  public pushContext(context: InputContext): void {
    if (this.currentContext !== context) {
      this.contextStack.push(this.currentContext);
      this.setContext(context);
    }
  }

  /**
   * Pop the current context and restore the previous one
   */
  public popContext(): void {
    const previous = this.contextStack.pop();
    if (previous) {
      this.setContext(previous);
    }
  }

  /**
   * Set the current context (replaces current, doesn't push to stack)
   */
  public setContext(context: InputContext): void {
    const oldContext = this.currentContext;
    this.currentContext = context;

    // Log context change for debugging
    console.log(`🎮 [Luxar] Input context changed: ${oldContext} → ${context}`);
  }

  /**
   * Get the current context
   */
  public getContext(): InputContext {
    return this.currentContext;
  }

  /**
   * Register a key binding for a specific context
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
      console.warn(
        `⚠️ [Luxar] Key binding conflict in ${context}: ${bindingKey} is already registered`
      );
    }

    contextBindings.set(bindingKey, binding);
  }

  /**
   * Unregister a key binding
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
   * Handle a keyboard event
   * Returns true if the event was handled
   */
  public handleKeyEvent(event: KeyboardEvent, type: 'down' | 'up'): boolean {
    if (!this.enabled) return false;

    // Check if we're in a typing context
    if (this.isTypingContext()) {
      // Allow Escape to exit typing contexts
      if (event.key === 'Escape') {
        return false; // Let it pass through to close dialogs
      }
      // Block all other keys while typing
      return true;
    }

    // Get the current context configuration
    const config = this.contextConfigs.get(this.currentContext);
    if (!config) return false;

    // Check if this key is allowed in the current context
    if (!this.isKeyAllowedInContext(event.key, config)) {
      return false;
    }

    // Find and execute the binding
    const contextBindings = this.bindings.get(this.currentContext);
    if (contextBindings) {
      const bindingKey = this.getBindingKeyFromEvent(event);
      const binding = contextBindings.get(bindingKey);

      if (binding) {
        if (binding.preventDefault) {
          event.preventDefault();
        }
        binding.handler(event);
        return true;
      }
    }

    // Check if we should pass through to lower contexts
    if (config.passthrough) {
      // Try lower priority contexts
      return this.tryLowerContexts(event, type);
    }

    return false;
  }

  /**
   * Check if a key is allowed in the given context
   */
  private isKeyAllowedInContext(key: string, config: ContextConfig): boolean {
    // Check blocked keys
    if (config.blockedKeys && config.blockedKeys.includes(key)) {
      return false;
    }

    // Check allowed keys
    if (config.allowedKeys && !config.allowedKeys.includes(key)) {
      return false;
    }

    return true;
  }

  /**
   * Try to handle the event in lower priority contexts
   */
  private tryLowerContexts(event: KeyboardEvent, _type: 'down' | 'up'): boolean {
    // Sort contexts by priority
    const sortedContexts = Array.from(this.contextConfigs.entries())
      .filter(([ctx]) => ctx !== this.currentContext)
      .sort((a, b) => (b[1].priority || 0) - (a[1].priority || 0));

    for (const [context, config] of sortedContexts) {
      if (this.isKeyAllowedInContext(event.key, config)) {
        const contextBindings = this.bindings.get(context);
        if (contextBindings) {
          const bindingKey = this.getBindingKeyFromEvent(event);
          const binding = contextBindings.get(bindingKey);

          if (binding) {
            if (binding.preventDefault) {
              event.preventDefault();
            }
            binding.handler(event);
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * Check if we're currently in a typing context
   */
  private isTypingContext(): boolean {
    if (this.currentContext === InputContext.TYPING) {
      return true;
    }

    // Also check if focus is in an input element
    const activeElement = document.activeElement;
    if (activeElement) {
      const tagName = activeElement.tagName.toLowerCase();
      if (
        tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select' ||
        activeElement.getAttribute('contenteditable') === 'true'
      ) {
        return true;
      }
    }

    return false;
  }

  /**
   * Generate a unique key for a binding
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
   * Generate a binding key from a keyboard event
   */
  private getBindingKeyFromEvent(event: KeyboardEvent): string {
    const parts = [event.key.toLowerCase()];

    if (event.ctrlKey) parts.push('ctrl');
    if (event.shiftKey) parts.push('shift');
    if (event.altKey) parts.push('alt');
    if (event.metaKey) parts.push('meta');

    return parts.sort().join('+');
  }

  /**
   * Enable/disable the context manager
   */
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Get debug information about current state
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
   * Clear all bindings for a specific context
   */
  public clearContextBindings(context: InputContext): void {
    this.bindings.delete(context);
  }

  /**
   * Reset to default state
   */
  public reset(): void {
    this.currentContext = InputContext.NAVIGATION;
    this.contextStack = [];
    this.bindings.clear();
  }
}
