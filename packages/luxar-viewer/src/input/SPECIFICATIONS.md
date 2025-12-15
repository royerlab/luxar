# luxar-viewer.input - Technical Specification

**Version**: 2.0.0
**Last Updated**: 2025-12-15

## Purpose

The `luxar-viewer.input` package provides context-aware keyboard and mouse input handling with intelligent conflict resolution. It prevents input conflicts between navigation, UI controls, and text input through a priority-based context system.

**Core Responsibility**: Route input events to appropriate handlers based on application state, preventing keyboard shortcuts from interfering with text input and ensuring mode-specific keys (WASD) only work in appropriate contexts.

**Related Specifications**:

- `luxar-viewer.controls` - Camera control integration (see `../controls/SPECIFICATIONS.md`)

---

## Table of Contents

1. [Input Context System](#input-context-system)
2. [Key Binding System](#key-binding-system)
3. [Typing Detection](#typing-detection)
4. [Integration with InputHandler](#integration-with-inputhandler)

---

## 1. Input Context System

### 1.1 Context Types

**Implementation**: String literal enum with separate priority field

```typescript
enum InputContext {
  NAVIGATION = 'navigation', // Default 3D navigation (orbit/arcball)
  FLY_CONTROLS = 'fly_controls', // Fly mode active (WASD enabled)
  TYPING = 'typing', // Text input focused (all shortcuts disabled)
  UI_INTERACTION = 'ui_interaction', // UI panels and controls
  DIMENSION_NAV = 'dimension_nav', // nD dimension navigation (1-9, [, ])
}
```

**Context Configuration**:

Each context has a separate configuration with explicit priority:

```typescript
interface ContextConfig {
  name: string;
  priority: number; // Higher priority overrides lower
  allowedKeys?: string[]; // Whitelist of allowed keys
  blockedKeys?: string[]; // Blacklist of blocked keys
  passthrough?: boolean; // Pass unhandled keys to lower contexts
}
```

**Priority Order** (higher number = higher priority):

```
TYPING (10) > UI_INTERACTION (5) > DIMENSION_NAV (2) > FLY_CONTROLS (1) > NAVIGATION (0)
```

**Note**: No MODAL context currently implemented (5 contexts total)

### 1.2 Context Stack

**Purpose**: Support nested contexts (e.g., opening modal while in fly mode).

**Data Structure**:

```typescript
class InputContextManager {
  private contextStack: InputContext[] = [InputContext.NAVIGATION];

  getCurrentContext(): InputContext {
    return this.contextStack[this.contextStack.length - 1];
  }

  pushContext(context: InputContext): void {
    this.contextStack.push(context);
  }

  popContext(): InputContext | undefined {
    if (this.contextStack.length <= 1) {
      return undefined; // Can't pop last context
    }
    return this.contextStack.pop();
  }
}
```

**Example Flow**:

```
1. Start: ['navigation']
2. Enter fly mode: ['navigation', 'fly_controls']
3. Open UI panel: ['navigation', 'fly_controls', 'ui_interaction']
4. Close UI panel: ['navigation', 'fly_controls']
5. Exit fly mode: ['navigation']
```

**Current Implementation**: Context stack maintains history but current context is tracked separately via `currentContext` field rather than stack top.

---

## 2. Key Binding System

### 2.1 Architecture Overview

The InputContextManager provides a **dynamic key binding registration system** for loose coupling between components. Components register key handlers for specific contexts, and the manager routes events to registered handlers based on the current context.

**Core Principle**: Registration-based, not hard-coded routing.

**Key Components**:

- `KeyBinding` interface: Defines a key handler with key(s), modifiers, and callback function
- `ContextConfig`: Per-context configuration with allowedKeys, blockedKeys, and passthrough rules
- Registration API: `registerBinding()`, `unregisterBinding()`, `clearContextBindings()`
- Event routing: `handleKeyEvent()` checks registered bindings and returns boolean

**Implementation**: `input-context-manager.ts:31-394`

---

### 2.2 KeyBinding Interface

**Definition** (`input-context-manager.ts:31-43`):

```typescript
export interface KeyBinding {
  key: string; // Key to bind (single key: 'w', 'h', 'Shift')
  modifiers?: {
    // Optional modifier requirements
    ctrl?: boolean; // Require Ctrl/Cmd modifier
    shift?: boolean; // Require Shift modifier
    alt?: boolean; // Require Alt/Option modifier
    meta?: boolean; // Require Meta/Cmd modifier (macOS)
  };
  handler: (event: KeyboardEvent) => void; // Keydown handler
  keyupHandler?: (event: KeyboardEvent) => void; // Optional keyup handler (NEW in v2.0)
  preventDefault?: boolean; // If true, calls event.preventDefault() before handler
  description?: string; // Human-readable description for debug/help
}
```

**keyupHandler Feature** (NEW in v2.0):

- If `keyupHandler` provided: Called on keyup events, `handler` called on keydown
- If `keyupHandler` NOT provided: `handler` called for both keydown and keyup
- Use for keys that need different up/down behavior (Shift, fly controls)

**Example Bindings**:

```typescript
// Simple toggle action (keydown only)
const helpBinding: KeyBinding = {
  key: 'h',
  handler: () => this.toggleHelp(),
  preventDefault: true,
  description: 'Toggle help overlay',
};

// With modifier
const debugBinding: KeyBinding = {
  key: 'l',
  modifiers: { ctrl: true },
  handler: () => this.toggleDebugConsole(),
  preventDefault: true,
  description: 'Toggle debug console',
};

// With separate keydown/keyup handlers (NEW)
const shiftBinding: KeyBinding = {
  key: 'Shift',
  handler: () => this.disableZoom(), // Called on keydown
  keyupHandler: () => this.enableZoom(), // Called on keyup
  description: 'Zoom control',
};

// Fly controls with keyup for stopping movement (NEW)
const flyBinding: KeyBinding = {
  key: 'w',
  modifiers: { shift: true },
  handler: (e) => flyControls.handleKeyDown(e), // Start movement
  keyupHandler: (e) => flyControls.handleKeyUp(e), // Stop movement
  description: 'Fly forward (fast)',
};
```

---

### 2.3 Context Configuration

**Definition** (`input-context-manager.ts:44-59`):

```typescript
interface ContextConfig {
  priority: number; // Higher priority contexts checked first in passthrough
  allowedKeys?: Set<string>; // Whitelist (undefined = all keys allowed)
  blockedKeys?: Set<string>; // Blacklist (checked before allowedKeys)
  passthrough: boolean; // Try lower-priority contexts if key not handled
}
```

**Initialization** (`input-context-manager.ts:75-117`):

Contexts are initialized from `config.input.keyboard.*` with appropriate allowed/blocked keys:

```typescript
this.contextConfigs.set(InputContext.NAVIGATION, {
  priority: 0,
  allowedKeys: undefined, // All keys allowed
  blockedKeys: new Set(),
  passthrough: false,
});

this.contextConfigs.set(InputContext.FLY_CONTROLS, {
  priority: 1,
  allowedKeys: new Set(config.input.keyboard.flyModeKeys), // WASD, QE, etc.
  blockedKeys: new Set(),
  passthrough: true, // Fall through to NAVIGATION for unhandled keys
});

this.contextConfigs.set(InputContext.DIMENSION_NAV, {
  priority: 2,
  allowedKeys: new Set([...config.input.keyboard.dimensionKeys, '[', ']']), // 1-9, brackets
  blockedKeys: new Set(),
  passthrough: true,
});

this.contextConfigs.set(InputContext.UI_INTERACTION, {
  priority: 5,
  allowedKeys: undefined, // UI controls handle their own keys
  blockedKeys: new Set(),
  passthrough: false,
});

this.contextConfigs.set(InputContext.TYPING, {
  priority: 10, // Highest priority - blocks almost everything
  allowedKeys: new Set(['Escape']), // Only Escape allowed
  blockedKeys: new Set(),
  passthrough: false,
});
```

---

### 2.4 Registration API

**registerBinding(context: InputContext, binding: KeyBinding)** (`input-context-manager.ts:290-308`):

**Purpose**: Register a key handler for a specific context.

**Algorithm**:

```typescript
public registerBinding(context: InputContext, binding: KeyBinding): void {
  // 1. Ensure context entry exists
  if (!this.bindings.has(context)) {
    this.bindings.set(context, new Map());
  }

  // 2. Generate unique binding key from key + modifiers
  const bindingKey = this.getBindingKey(binding);

  // 3. Check for conflicts
  const contextBindings = this.bindings.get(context)!;
  if (contextBindings.has(bindingKey)) {
    log.warning(Modules.INPUT_CONTEXT,
      `Key binding conflict in ${context}: ${bindingKey} already registered`);
  }

  // 4. Store binding
  contextBindings.set(bindingKey, binding);
}
```

**Example Usage**:

```typescript
// Simple binding
inputContextManager.registerBinding(InputContext.NAVIGATION, {
  key: 'r',
  handler: () => this.sceneManager.resetCamera(),
  preventDefault: true,
  description: 'Reset camera view',
});

// With modifiers
inputContextManager.registerBinding(InputContext.NAVIGATION, {
  key: 'l',
  modifiers: { ctrl: true },
  handler: () => this.debugConsole.toggle(),
  preventDefault: true,
  description: 'Toggle debug console',
});

// With keydown and keyup handlers (NEW in v2.0)
inputContextManager.registerBinding(InputContext.FLY_CONTROLS, {
  key: 'w',
  modifiers: { shift: true },
  handler: (event) => flyControls.handleKeyDown(event),
  keyupHandler: (event) => flyControls.handleKeyUp(event),
  description: 'Fly forward (fast)',
});
```

**unregisterBinding(context: InputContext, key: string, modifiers?)** (`input-context-manager.ts:333-343`):

**Purpose**: Remove a previously registered binding.

**clearContextBindings(context: InputContext)** (`input-context-manager.ts:663-665`):

**Purpose**: Remove all bindings for a specific context (useful for cleanup).

---

### 2.5 Event Routing Algorithm

**Method**: `handleKeyEvent(event: KeyboardEvent, type: 'down' | 'up'): boolean` (`input-context-manager.ts:376-423`)

**Purpose**: Central event router that dispatches keyboard events to registered handlers based on current context. Supports separate keydown and keyup handlers.

**Algorithm**:

```typescript
public handleKeyEvent(event: KeyboardEvent, type: 'down' | 'up'): boolean {
  // 1. Check if manager is enabled
  if (!this.enabled) {
    return false;
  }

  // 2. Block all keys if in typing context (except Escape)
  if (this.isTypingContext()) {
    if (event.key === 'Escape') {
      return false;  // Let Escape pass through to close dialogs
    }
    return true;  // Block all other keys while typing
  }

  // 3. Get context configuration
  const config = this.contextConfigs.get(this.currentContext);
  if (!config) return false;

  // 4. Check if key is allowed in current context
  if (!this.isKeyAllowedInContext(event.key, config)) {
    // Try passthrough if enabled (FIXED in v2.0)
    if (config.passthrough) {
      return this.tryLowerContexts(event, type);
    }
    return false;
  }

  // 5. Find matching binding in current context
  const contextBindings = this.bindings.get(this.currentContext);
  if (contextBindings) {
    const bindingKey = this.getBindingKeyFromEvent(event);
    const binding = contextBindings.get(bindingKey);

    if (binding) {
      // Call preventDefault if specified
      if (binding.preventDefault) {
        event.preventDefault();
      }

      // Use keyupHandler for 'up' events if provided (NEW in v2.0)
      if (type === 'up' && binding.keyupHandler) {
        binding.keyupHandler(event);
      } else {
        binding.handler(event);
      }
      return true;  // Handled
    }
  }

  // 6. Try passthrough to lower-priority contexts if enabled
  if (config.passthrough) {
    return this.tryLowerContexts(event, type);
  }

  return false;  // No handler found
}
```

**Key Features** (v2.0):

- Returns `boolean` indicating if event was handled
- Supports separate `keyupHandler` for keys that need up/down behavior
- Respects typing context (blocks all keys except Escape)
- Passthrough checks BEFORE rejecting disallowed keys (FIXED)
- Uses modifier-aware key matching

---

### 2.6 Passthrough Mechanism

**Method**: `tryLowerContexts(event: KeyboardEvent): boolean` (`input-context-manager.ts:270-295`)

**Purpose**: When current context doesn't handle a key, try contexts with lower priority (if passthrough enabled).

**Algorithm**:

```typescript
private tryLowerContexts(event: KeyboardEvent): boolean {
  const currentPriority = this.contextConfigs.get(this.currentContext)?.priority ?? 0;

  // 1. Get all contexts sorted by priority (descending)
  const sortedContexts = Array.from(this.contextConfigs.entries())
    .filter(([ctx, _]) => ctx !== this.currentContext)
    .sort((a, b) => b[1].priority - a[1].priority);

  // 2. Try each lower-priority context
  for (const [context, config] of sortedContexts) {
    if (config.priority >= currentPriority) continue;  // Skip higher/equal priority

    // 3. Check if key is allowed in this context
    if (!this.isKeyAllowedInContext(event.key, context)) {
      continue;
    }

    // 4. Look for binding in this context
    const eventKey = this.getBindingKeyFromEvent(event);
    const binding = this.bindings.get(context)?.get(eventKey);

    if (binding) {
      const result = binding.handler(event);
      return result !== false;
    }
  }

  return false;  // No lower context handled it
}
```

**Example Use Case**:

- User in DIMENSION_NAV context
- Presses 'h' (help key)
- DIMENSION_NAV has no 'h' binding
- Passthrough enabled → tries NAVIGATION context
- NAVIGATION has 'h' → help overlay shown

---

### 2.7 Binding Key Generation

**Method**: `getBindingKeyFromEvent(event: KeyboardEvent): string` (`input-context-manager.ts:250-268`)

**Purpose**: Create unique key string from keyboard event including modifiers.

**Algorithm**:

```typescript
const modifiers: string[] = [];
if (event.ctrlKey) modifiers.push('Ctrl');
if (event.altKey) modifiers.push('Alt');
if (event.shiftKey) modifiers.push('Shift');
if (event.metaKey) modifiers.push('Meta');

const key = event.key.toLowerCase();
const bindingKey = modifiers.length > 0 ? `${modifiers.join('+')}+${key}` : key;
```

**Examples**:

- `Ctrl+L` → `"Ctrl+l"`
- `Shift+W` → `"Shift+w"`
- `Alt+Meta+A` → `"Alt+Meta+a"`
- `w` → `"w"`

---

## 3. Typing Detection

### 3.1 Typing Context Detection

**Purpose**: Block keyboard shortcuts when user is typing in text fields to prevent conflicts.

**Method**: `isTypingContext(): boolean` (`input-context-manager.ts:300-320`)

**Algorithm** (Dual-check approach):

```typescript
public isTypingContext(): boolean {
  // Check 1: Are we in TYPING context?
  if (this.currentContext === InputContext.TYPING) {
    return true;
  }

  // Check 2: Is active DOM element a text input?
  const active = document.activeElement;
  if (!active) return false;

  // Check input elements
  if (active.tagName === 'INPUT') {
    const type = (active as HTMLInputElement).type;
    const textTypes = ['text', 'search', 'url', 'email', 'password', 'tel', 'number'];
    return textTypes.includes(type);
  }

  // Check textarea
  if (active.tagName === 'TEXTAREA') {
    return true;
  }

  // Check contentEditable
  if (active.getAttribute('contenteditable') === 'true') {
    return true;
  }

  return false;
}
```

**Design Note**: Checks both context state AND DOM active element for robustness. Even if context not explicitly set to TYPING, will detect text input focus.

---

### 3.2 Usage in InputHandler

**InputHandler also has typing detection** (`input-handler.ts:713-725`):

```typescript
private isTypingInInput(): boolean {
  const activeElement = document.activeElement;
  if (!activeElement) return false;

  if (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA') {
    return true;
  }

  if (activeElement.getAttribute('contenteditable') === 'true') {
    return true;
  }

  return false;
}
```

**Note**: InputHandler uses simpler DOM-only check. This is intentional - InputHandler performs direct key handling and doesn't rely on context manager for typing detection.

---

### 3.3 No Automatic Context Switching

**Important**: The InputContextManager does NOT automatically push/pop TYPING context on focus/blur events.

**Rationale**: Automatic event listeners would create coupling with DOM lifecycle. Instead:

- Context manager provides `isTypingContext()` for checking
- Handlers call it before processing keys
- UI components manage their own context state if needed

**Example** (how it actually works):

```typescript
// In handleKeyEvent:
if (this.isTypingContext()) {
  return event.key === 'Escape'; // Only Escape allowed
}
```

---

## 4. Integration with InputHandler

### 4.1 Unified Architecture

**IMPORTANT**: InputHandler now uses InputContextManager's binding registration system exclusively for all key handling.

**InputContextManager Role**:

- Provides context tracking (current context state)
- Provides binding registration system
- Routes all keyboard events via `handleKeyEvent()`
- Filters keys based on context (allowedKeys, blockedKeys)
- Supports passthrough to lower-priority contexts

**InputHandler Role**:

- Registers all key bindings during initialization via `registerAllKeyBindings()`
- Routes keyboard events through `contextManager.handleKeyEvent()`
- Manages context switches (e.g., orbit ↔ fly mode)
- Handles special cases (Shift key for zoom, fly control keyup events)

**Design**: Single source of truth for key handling - all keys go through the binding registration system.

---

### 4.2 InputHandler Key Registration

**Method**: `registerAllKeyBindings()` (`input-handler.ts:549-799`)

**Total Bindings Registered**: 58

- NAVIGATION context: 25 bindings
- FLY_CONTROLS context: 33 bindings

**Registration Pattern**:

```typescript
private registerAllKeyBindings(): void {
  // ===== NAVIGATION CONTEXT (25 bindings) =====

  // Shift key with keydown/keyup handlers
  this.contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'Shift',
    handler: () => this.sceneManager.controls.setEnableZoom(false),
    keyupHandler: () => this.sceneManager.controls.setEnableZoom(true),
    description: 'Zoom control',
  });

  // Dimension navigation
  this.contextManager.registerBinding(InputContext.NAVIGATION, {
    key: '[',
    handler: () => this.handleDimensionNavigation(-1),
    preventDefault: true,
  });

  // Dimension selection (9 bindings)
  for (let i = 1; i <= 9; i++) {
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: String(i),
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
          event.preventDefault();
          this.selectDimension(i - 1);
        }
      },
      preventDefault: false,  // Conditional to allow browser shortcuts
    });
  }

  // ===== FLY_CONTROLS CONTEXT (33 bindings) =====

  // Movement keys with ALL modifier combinations (24 bindings)
  const flyMovementKeys = ['w', 'a', 's', 'd', 'q', 'e'];
  const modifierCombos = [
    {},                          // Base
    { shift: true },             // Speed boost
    { alt: true },               // Vertical (W/S)
    { shift: true, alt: true }   // Fast vertical
  ];

  for (const key of flyMovementKeys) {
    for (const mods of modifierCombos) {
      this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
        key,
        modifiers: Object.keys(mods).length > 0 ? mods : undefined,
        handler: (e) => getFlyControls()?.handleKeyDown(e),
        keyupHandler: (e) => getFlyControls()?.handleKeyUp(e),  // Stop movement
      });
    }
  }

  // Arrow keys (8 bindings: 4 base + 4 with Shift)
  for (const key of ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) {
    this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key,
      handler: (e) => getFlyControls()?.handleKeyDown(e),
      keyupHandler: (e) => getFlyControls()?.handleKeyUp(e),
    });

    // With Shift modifier
    this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key,
      modifiers: { shift: true },
      handler: (e) => getFlyControls()?.handleKeyDown(e),
      keyupHandler: (e) => getFlyControls()?.handleKeyUp(e),
    });
  }

  // Shift in fly mode (1 binding)
  this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
    key: 'Shift',
    handler: () => this.sceneManager.controls.setEnableZoom(false),
    keyupHandler: () => this.sceneManager.controls.setEnableZoom(true),
  });
}
```

**Registration happens once** during `setupWindowEvents()` initialization (line 368).

---

### 4.3 InputHandler Key Routing

**Methods**: `onKeyDown()` and `onKeyUp()` (`input-handler.ts:807-828`)

**Ultra-Simplified Implementation**:

```typescript
private onKeyDown(event: KeyboardEvent): void {
  if (this.isTypingInInput()) return;  // Only necessary guard
  this.contextManager.handleKeyEvent(event, 'down');  // Route ALL keys
}

private onKeyUp(event: KeyboardEvent): void {
  this.contextManager.handleKeyEvent(event, 'up');  // Route ALL keys
}
```

**Design**: Minimal, clean routing (8 + 5 = 13 lines total)

- ✅ ZERO special cases
- ✅ ZERO switch statements
- ✅ ZERO pre-checks
- ✅ ONE guard (typing detection)
- ✅ ONE routing call per method

**Key Insight**: The binding system with `keyupHandler` support handles ALL complexity:

- Shift key: Registered with keydown/keyup handlers
- Fly controls: Registered with keydown/keyup handlers + all modifier combos
- Toggle actions: Only have keydown handler (keyup is no-op)
- Browser shortcuts: Conditional preventDefault in handlers

---

### 4.4 Context State Management

**How InputHandler manages contexts**:

```typescript
// Switching context when toggling control mode
private toggleControlMode(): void {
  const currentType = this.sceneManager.controls.getControlType();
  let newType: 'orbit' | 'arcball' | 'fly';

  // Cycle through modes
  switch (currentType) {
    case 'orbit':
      newType = 'arcball';
      break;
    case 'arcball':
      newType = 'fly';
      break;
    case 'fly':
      newType = 'orbit';
      break;
  }

  this.sceneManager.controls.setControlType(newType);

  // Update input context based on control mode
  if (newType === 'fly') {
    this.contextManager.setContext(InputContext.FLY_CONTROLS);
  } else {
    this.contextManager.setContext(InputContext.NAVIGATION);
  }
}
```

**Usage Pattern**: Context switches activate different key binding sets automatically.

---

## Data Structures

### InputContextManager (Actual Implementation)

**Location**: `input-context-manager.ts:18-394`

```typescript
export class InputContextManager {
  // State
  private currentContext: InputContext = InputContext.NAVIGATION;
  private contextStack: InputContext[] = []; // Initially empty
  private enabled: boolean = true;

  // Binding system
  private bindings: Map<InputContext, Map<string, KeyBinding>> = new Map();
  private contextConfigs: Map<InputContext, ContextConfig> = new Map();

  // Public API - Context Management
  setContext(context: InputContext): void; // Set current context directly
  pushContext(context: InputContext): void; // Push old context, set new
  popContext(): void; // Restore previous context
  getContext(): InputContext; // Get current context (NOT getCurrentContext!)
  reset(): void; // Reset to NAVIGATION, clear bindings

  // Public API - Binding Registration
  registerBinding(context: InputContext, binding: KeyBinding): void;
  unregisterBinding(context: InputContext, binding: KeyBinding): void;
  clearContextBindings(context: InputContext): void;

  // Public API - Event Handling
  handleKeyEvent(event: KeyboardEvent): boolean; // Route to registered handlers
  isTypingContext(): boolean; // Check if in typing mode

  // Public API - Utility
  setEnabled(enabled: boolean): void; // Enable/disable manager
  getDebugInfo(): object; // Get state for debugging
}
```

**Key Differences from Previous Spec**:

- Method is `getContext()` not `getCurrentContext()`
- Has `registerBinding()` / `unregisterBinding()` / `clearContextBindings()`
- Has `handleKeyEvent()` that returns boolean
- Has `setEnabled()` and `getDebugInfo()`
- `currentContext` and `contextStack` are separate (not stack-top reading)

---

### InputHandler (Actual Implementation)

**Location**: `input-handler.ts:27-829`

```typescript
export class InputHandler {
  // Dependencies
  private container: HTMLElement;
  private controlsManager: ControlsManager;
  private sceneDimsManager: SceneDimsManager;
  private contextManager: InputContextManager;
  private sceneManager: SceneManager;
  private renderingControls?: RenderingControls;

  // State
  private flyControls?: LuxarFlyControls;
  private dimensionNavigationConfig: NavigationConfig;

  // Public API
  constructor(
    container: HTMLElement,
    controlsManager: ControlsManager,
    sceneDimsManager: SceneDimsManager,
    contextManager: InputContextManager,
    sceneManager: SceneManager
  );

  // Methods
  setRenderingControls(controls: RenderingControls): void;
  initializeListeners(): void;
  dispose(): void;

  // Private - Event Handlers
  private onKeyDown(event: KeyboardEvent): void; // Routes to context manager (8 lines)
  private onKeyUp(event: KeyboardEvent): void; // Routes to context manager (5 lines)
  private isTypingInInput(): boolean; // Typing detection
  private registerAllKeyBindings(): void; // Registers all 58 bindings (228 lines)
  private selectDimension(index: number): void;
  private handleDimensionNavigation(direction: 1 | -1): void;
  // ... many other private methods for specific actions
}
```

---

## Changelog

- **v2.0.0** (2025-12-15): **BREAKING CHANGE** - Unified binding system migration

  **New Features**:
  - **ADDED**: `keyupHandler` optional field in KeyBinding interface
    - Enables separate handlers for keydown and keyup events
    - Used for Shift key (zoom control) and fly controls (movement stop)
    - Prevents toggle actions from double-triggering
  - **ADDED**: `registerAllKeyBindings()` method in InputHandler (228 lines)
    - Registers 58 total bindings (25 NAVIGATION + 33 FLY_CONTROLS)
    - Fly controls with all modifier combinations (w, shift+w, alt+w, shift+alt+w)
    - Centralized registration in one location
  - **ADDED**: `handleDataMonitorCycle()` extracted method for async import

  **Architecture Changes**:
  - **REMOVED**: Dual-track architecture (switch statement + unused binding system)
  - **REMOVED**: Giant switch statement in `onKeyDown()` (~175 lines)
  - **REMOVED**: Fly control pre-check before switch
  - **REMOVED**: Special Shift key handling in onKeyDown/onKeyUp
  - **SIMPLIFIED**: `onKeyDown()` from 175 lines → 8 lines (96% reduction!)
  - **SIMPLIFIED**: `onKeyUp()` from 52 lines → 5 lines (90% reduction!)
  - **UNIFIED**: All keys route through InputContextManager.handleKeyEvent()

  **Bug Fixes**:
  - **FIXED**: Passthrough not working when key not in allowedKeys (line 394-399)
    - Now checks passthrough BEFORE returning false
    - Global shortcuts (h, p, r) now work in FLY_CONTROLS mode
  - **FIXED**: Shift blocked in NAVIGATION context (line 123-125)
    - Filtered Shift from blockedKeys
    - Shift now works for FOV wheel control in orbit mode
  - **FIXED**: Browser shortcuts blocked incorrectly
    - Conditional preventDefault for r, m, f, v, i, c, Space, 1-9
    - Cmd+R, Cmd+F, Ctrl+1-9 no longer blocked
  - **FIXED**: keyup toggling actions twice
    - Toggle actions (c, r, etc.) don't have keyupHandler
    - Only keydown triggers toggles, keyup is no-op
  - **FIXED**: Shift+W not working in fly mode
    - Registered all 24 modifier combinations for fly keys
    - w, shift+w, alt+w, shift+alt+w all work

  **Documentation**:
  - Section 2.2: Updated KeyBinding interface with keyupHandler
  - Section 2.4: Updated registration examples
  - Section 2.5: Updated event routing algorithm
  - Section 4: Completely rewritten (unified architecture)
  - Changelog: Comprehensive v2.0.0 documentation

  **Tests**:
  - **ADDED**: 7 new tests for keyupHandler functionality
  - **UPDATED**: Passthrough test to expect correct behavior
  - 160/160 tests pass (153 original + 7 new)
  - TypeScript: 0 errors

- **v1.2.0** (2025-12-12): Comprehensive rewrite to match actual implementation
  - **BREAKING DOCUMENTATION**: Completely rewrote sections 2-4 to reflect actual architecture
  - **ADDED**: Section 2: Key binding registration system documentation (165+ lines)
    - `registerBinding()` / `unregisterBinding()` / `clearContextBindings()` API
    - `KeyBinding` interface with handler functions
    - `ContextConfig` with allowedKeys, blockedKeys, passthrough
    - Dynamic binding registration pattern
  - **ADDED**: Section 2.5-2.7: Event routing algorithm documentation
    - `handleKeyEvent()` returns boolean
    - `tryLowerContexts()` passthrough mechanism
    - `getBindingKeyFromEvent()` modifier handling
  - **ADDED**: Section 4: Integration with InputHandler architecture
    - Documented loose coupling (not tight integration)
    - Explained InputHandler uses direct key handling (not binding system)
    - Context manager used for state tracking only
  - **FIXED**: Method names (getContext vs getCurrentContext)
  - **FIXED**: Context stack architecture (separate currentContext field)
  - **REMOVED**: CONTEXT_KEYS constant documentation (doesn't exist - uses dynamic bindings)
  - **REMOVED**: shouldHandleKey() / routeKeyEvent() / getEffectiveKey() (don't exist as described)
  - **CLARIFIED**: No automatic focus/blur event listeners
  - **CLARIFIED**: InputHandler and ContextManager relationship
  - **NOTE**: Previous spec described desired architecture, not actual implementation

- **v1.1.0** (2025-12-08): Partial updates
  - Changed InputContext from numeric enum to string literal enum
  - Added ContextConfig interface
  - Updated priority order
  - **NOTE**: This version still contained significant inaccuracies

- **v1.0.0** (2025-01-30): Initial specification
  - Context-based input routing
  - Context stack concept
  - Typing detection
  - Mode-specific key filtering
