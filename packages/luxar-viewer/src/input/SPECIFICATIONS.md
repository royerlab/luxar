# luxar-viewer.input - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2025-12-12

## Purpose

The `luxar-viewer.input` package provides context-aware keyboard and mouse input handling with intelligent conflict resolution. It prevents input conflicts between navigation, UI controls, and text input through a priority-based context system.

**Core Responsibility**: Route input events to appropriate handlers based on application state, preventing keyboard shortcuts from interfering with text input and ensuring mode-specific keys (WASD) only work in appropriate contexts.

**Related Specifications**:

- `luxar-viewer.controls` - Camera control integration (see `../controls/SPECIFICATIONS.md`)

---

## Table of Contents

1. [Input Context System](#input-context-system)
2. [Event Routing](#event-routing)
3. [Typing Detection](#typing-detection)
4. [Key Filtering](#key-filtering)

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

**Definition** (`input-context-manager.ts:31-42`):

```typescript
export interface KeyBinding {
  keys: string | string[];           // Key or array of keys to bind ('w', ['w', 'W'])
  withShift?: boolean;                // Require Shift modifier
  withCtrl?: boolean;                 // Require Ctrl/Cmd modifier
  withAlt?: boolean;                  // Require Alt/Option modifier
  withMeta?: boolean;                 // Require Meta/Cmd modifier (macOS)
  description?: string;               // Human-readable description for debug
  handler: (event: KeyboardEvent) => boolean | void;  // Handler function
}
```

**Handler Return Value**:
- `true` or `undefined`: Event was handled, prevent default browser behavior
- `false`: Event not handled, continue to propagation/passthrough

**Example Binding**:
```typescript
const resetBinding: KeyBinding = {
  keys: 'r',
  description: 'Reset camera view',
  handler: (event) => {
    sceneManager.resetView();
    return true; // Handled, prevent default
  }
};
```

---

### 2.3 Context Configuration

**Definition** (`input-context-manager.ts:44-59`):

```typescript
interface ContextConfig {
  priority: number;                   // Higher priority contexts checked first in passthrough
  allowedKeys?: Set<string>;          // Whitelist (undefined = all keys allowed)
  blockedKeys?: Set<string>;          // Blacklist (checked before allowedKeys)
  passthrough: boolean;               // Try lower-priority contexts if key not handled
}
```

**Initialization** (`input-context-manager.ts:75-117`):

Contexts are initialized from `config.input.keyboard.*` with appropriate allowed/blocked keys:

```typescript
this.contextConfigs.set(InputContext.NAVIGATION, {
  priority: 0,
  allowedKeys: undefined,  // All keys allowed
  blockedKeys: new Set(),
  passthrough: false
});

this.contextConfigs.set(InputContext.FLY_CONTROLS, {
  priority: 1,
  allowedKeys: new Set(config.input.keyboard.flyModeKeys),  // WASD, QE, etc.
  blockedKeys: new Set(),
  passthrough: true  // Fall through to NAVIGATION for unhandled keys
});

this.contextConfigs.set(InputContext.DIMENSION_NAV, {
  priority: 2,
  allowedKeys: new Set([...config.input.keyboard.dimensionKeys, '[', ']']),  // 1-9, brackets
  blockedKeys: new Set(),
  passthrough: true
});

this.contextConfigs.set(InputContext.UI_INTERACTION, {
  priority: 5,
  allowedKeys: undefined,  // UI controls handle their own keys
  blockedKeys: new Set(),
  passthrough: false
});

this.contextConfigs.set(InputContext.TYPING, {
  priority: 10,  // Highest priority - blocks almost everything
  allowedKeys: new Set(['Escape']),  // Only Escape allowed
  blockedKeys: new Set(),
  passthrough: false
});
```

---

### 2.4 Registration API

**registerBinding(context: InputContext, binding: KeyBinding)** (`input-context-manager.ts:165-186`):

**Purpose**: Register a key handler for a specific context.

**Algorithm**:
```typescript
public registerBinding(context: InputContext, binding: KeyBinding): void {
  // 1. Ensure context entry exists
  if (!this.bindings.has(context)) {
    this.bindings.set(context, new Map());
  }

  // 2. Generate unique binding key from keys + modifiers
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
inputContextManager.registerBinding(InputContext.NAVIGATION, {
  keys: 'r',
  description: 'Reset camera view',
  handler: (event) => {
    this.sceneManager.resetCamera();
    return true;
  }
});

// Register with modifiers
inputContextManager.registerBinding(InputContext.NAVIGATION, {
  keys: 'l',
  withCtrl: true,
  description: 'Toggle debug console',
  handler: (event) => {
    this.debugConsole.toggle();
    return true;
  }
});
```

**unregisterBinding(context: InputContext, binding: KeyBinding)** (`input-context-manager.ts:188-200`):

**Purpose**: Remove a previously registered binding.

**clearContextBindings(context: InputContext)** (`input-context-manager.ts:383-388`):

**Purpose**: Remove all bindings for a specific context (useful for cleanup).

---

### 2.5 Event Routing Algorithm

**Method**: `handleKeyEvent(event: KeyboardEvent): boolean` (`input-context-manager.ts:204-248`)

**Purpose**: Central event router that dispatches keyboard events to registered handlers based on current context.

**Algorithm**:

```typescript
public handleKeyEvent(event: KeyboardEvent): boolean {
  // 1. Check if manager is enabled
  if (!this.enabled) {
    return false;
  }

  // 2. Block all keys if in typing context (except Escape)
  if (this.isTypingContext()) {
    return event.key === 'Escape';  // Only allow Escape to exit typing
  }

  // 3. Get bindings for current context
  const contextBindings = this.bindings.get(this.currentContext);
  if (!contextBindings) {
    return false;
  }

  // 4. Generate key from event (with modifiers)
  const eventKey = this.getBindingKeyFromEvent(event);

  // 5. Look for matching binding in current context
  const binding = contextBindings.get(eventKey);
  if (binding) {
    const result = binding.handler(event);
    return result !== false;  // Handler returns true/undefined = handled
  }

  // 6. Try passthrough to lower-priority contexts if enabled
  const contextConfig = this.contextConfigs.get(this.currentContext);
  if (contextConfig?.passthrough) {
    return this.tryLowerContexts(event);
  }

  return false;  // No handler found
}
```

**Key Features**:
- Returns `boolean` indicating if event was handled
- Respects typing context (blocks all keys except Escape)
- Supports passthrough to lower-priority contexts
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
const bindingKey = modifiers.length > 0
  ? `${modifiers.join('+')}+${key}`
  : key;
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
  return event.key === 'Escape';  // Only Escape allowed
}
```

---

## 4. Integration with InputHandler

### 4.1 Relationship Architecture

**IMPORTANT**: InputContextManager and InputHandler have **loose coupling**, not tight integration.

**InputContextManager Role**:
- Provides context tracking (current context state)
- Provides binding registration system
- Offers `handleKeyEvent()` for components that want to use it

**InputHandler Role**:
- Performs direct keyboard event handling with giant switch statement
- Uses context manager for **state tracking only** (calls `pushContext()`, `popContext()`)
- Does NOT route keys through `contextManager.handleKeyEvent()`
- Implements own typing detection

**Why Loose Coupling**: Historical evolution - InputHandler predates binding system. Binding system added later for extensibility without breaking existing InputHandler logic.

**Code Evidence**:
- InputHandler.onKeyDown (input-handler.ts:358-533): 175 lines of direct key handling
- InputHandler uses contextManager.getContext() for state checks (lines 654-658)
- InputHandler does NOT call contextManager.handleKeyEvent()

---

### 4.2 Actual InputHandler Key Routing

**Method**: `onKeyDown(event: KeyboardEvent)` (`input-handler.ts:358-533`)

**Algorithm** (simplified):

```typescript
private onKeyDown(event: KeyboardEvent): void {
  // 1. Check typing (blocks all keys)
  if (this.isTypingInInput()) {
    return;
  }

  const key = event.key.toLowerCase();

  // 2. Check for fly controls keys (WASD, QE, arrows)
  if (this.controlsManager.getCurrentType() === 'fly') {
    const flyKeys = ['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'];
    if (flyKeys.includes(key) || event.shiftKey || event.altKey) {
      this.flyControls.handleKeyDown(event);
      event.preventDefault();
      return;
    }
  }

  // 3. Giant switch statement for all other keys
  switch (key) {
    case 'h': this.toggleHelp(); break;
    case 'f': this.toggleFlyMode(); break;
    case 'p': this.togglePerformanceMonitor(); break;
    case 'r': this.resetCamera(); break;
    case 'c': this.centerCamera(); break;
    case 'm': this.toggleRenderingControls(); break;
    case 'n': this.toggleDimensionSliders(); break;
    case '1': case '2': case '3': case '4': case '5':
    case '6': case '7': case '8': case '9':
      this.selectDimension(parseInt(key) - 1);
      break;
    case '[': this.stepDimension(-1); break;
    case ']': this.stepDimension(1); break;
    // ... many more cases
  }

  event.preventDefault();
}
```

**Design Note**: This is **different** from the binding registration system. InputHandler does NOT use `contextManager.handleKeyEvent()`. The binding system is available for future refactoring or for new components that want cleaner integration.

---

### 4.3 Context State Usage

**How InputHandler uses Context Manager**:

```typescript
// Setting context (input-handler.ts:654-658)
private toggleFlyMode(): void {
  if (this.controlsManager.getCurrentType() === 'fly') {
    this.controlsManager.setControlType('orbit');
    this.contextManager.popContext();  // Exit FLY_CONTROLS context
  } else {
    this.controlsManager.setControlType('fly');
    this.contextManager.pushContext(InputContext.FLY_CONTROLS);  // Enter FLY_CONTROLS context
  }
}

// Checking context (for state-dependent behavior)
const currentContext = this.contextManager.getContext();
if (currentContext === InputContext.FLY_CONTROLS) {
  // Enable fly-specific behavior
}
```

**Usage Pattern**: Context manager is a **state tracker**, not a key router (for InputHandler).

---

### 4.4 Future Refactoring Opportunity

**Current State**: Two parallel key handling approaches:
1. InputHandler with direct switch statement (legacy, works well)
2. Binding registration system (modern, extensible)

**Future**: Consider migrating InputHandler to use binding registration for:
- Consistency with modern approach
- Easier extensibility
- Better testability
- Cleaner separation of concerns

**Not Breaking**: Current approach works correctly, refactoring is optional optimization.

---

## Data Structures

### InputContextManager (Actual Implementation)

**Location**: `input-context-manager.ts:18-394`

```typescript
export class InputContextManager {
  // State
  private currentContext: InputContext = InputContext.NAVIGATION;
  private contextStack: InputContext[] = [];  // Initially empty
  private enabled: boolean = true;

  // Binding system
  private bindings: Map<InputContext, Map<string, KeyBinding>> = new Map();
  private contextConfigs: Map<InputContext, ContextConfig> = new Map();

  // Public API - Context Management
  setContext(context: InputContext): void;           // Set current context directly
  pushContext(context: InputContext): void;          // Push old context, set new
  popContext(): void;                                // Restore previous context
  getContext(): InputContext;                        // Get current context (NOT getCurrentContext!)
  reset(): void;                                     // Reset to NAVIGATION, clear bindings

  // Public API - Binding Registration
  registerBinding(context: InputContext, binding: KeyBinding): void;
  unregisterBinding(context: InputContext, binding: KeyBinding): void;
  clearContextBindings(context: InputContext): void;

  // Public API - Event Handling
  handleKeyEvent(event: KeyboardEvent): boolean;     // Route to registered handlers
  isTypingContext(): boolean;                        // Check if in typing mode

  // Public API - Utility
  setEnabled(enabled: boolean): void;                // Enable/disable manager
  getDebugInfo(): object;                            // Get state for debugging
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
  private onKeyDown(event: KeyboardEvent): void;     // Main key handling (175 lines)
  private onKeyUp(event: KeyboardEvent): void;
  private isTypingInInput(): boolean;                // Typing detection
  private selectDimension(index: number): void;
  private stepDimension(direction: 1 | -1): void;
  // ... many other private methods for specific actions
}
```

**Key Point**: InputHandler does direct key handling with switch statements, not via binding registration.

---

## Changelog

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
