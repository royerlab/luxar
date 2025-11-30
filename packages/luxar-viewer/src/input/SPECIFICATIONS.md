# luxar-viewer.input - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-30

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

```typescript
enum InputContext {
  NAVIGATION = 0, // Default 3D navigation (orbit/arcball)
  FLY_CONTROLS = 1, // Fly mode active (WASD enabled)
  TYPING = 2, // Text input focused (all shortcuts disabled)
  DIMENSION_NAV = 3, // nD dimension navigation (1-9, [, ])
  UI_OVERLAY = 4, // UI panel open (Tab, Enter, Esc)
  MODAL = 5, // Modal dialog (highest priority)
}
```

**Priority Order**: Higher enum value = higher priority

**Example**:

```
MODAL (5) > UI_OVERLAY (4) > TYPING (2) > FLY_CONTROLS (1) > NAVIGATION (0)
```

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
1. Start: [NAVIGATION]
2. Enter fly mode: [NAVIGATION, FLY_CONTROLS]
3. Open modal: [NAVIGATION, FLY_CONTROLS, MODAL]
4. Close modal: [NAVIGATION, FLY_CONTROLS]
5. Exit fly mode: [NAVIGATION]
```

---

## 2. Event Routing

### 2.1 Keyboard Event Flow

**Algorithm**:

```typescript
function handleKeyDown(event: KeyboardEvent): void {
  // 1. Check if typing in text field
  if (isTypingInInput()) {
    return; // Let browser handle it
  }

  // 2. Get current context
  const context = contextManager.getCurrentContext();

  // 3. Check if current context should handle this key
  if (!shouldHandleKey(event.key, context)) {
    return; // Key not relevant to current context
  }

  // 4. Prevent default browser behavior
  event.preventDefault();

  // 5. Route to appropriate handler
  routeKeyEvent(event, context);
}
```

### 2.2 Key Routing Decision Tree

```
Key Press
    ↓
Typing in input?
    ├─ Yes → Let browser handle
    └─ No → Continue
         ↓
Get current context
    ↓
Check context-specific key list
    ├─ Not in list → Ignore
    └─ In list → Continue
         ↓
Route to handler:
    ├─ MODAL → Modal handler (Esc, Enter)
    ├─ UI_OVERLAY → UI handler (Tab, Esc)
    ├─ DIMENSION_NAV → Dimension handler (1-9, [, ])
    ├─ FLY_CONTROLS → Fly handler (WASD, QE)
    └─ NAVIGATION → Navigation handler (Arrow keys, global shortcuts)
```

### 2.3 Context-Specific Key Sets

**Definition**:

```typescript
const CONTEXT_KEYS = {
  [InputContext.NAVIGATION]: [
    'ArrowUp',
    'ArrowDown',
    'ArrowLeft',
    'ArrowRight',
    'h',
    'p',
    'r',
    'v',
    'f',
    'c',
    'm',
    'o',
    'n',
  ],
  [InputContext.FLY_CONTROLS]: [
    'w',
    'a',
    's',
    'd',
    'q',
    'e',
    'i',
    'ArrowUp',
    'ArrowDown',
    'ArrowLeft',
    'ArrowRight',
  ],
  [InputContext.DIMENSION_NAV]: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']'],
  [InputContext.UI_OVERLAY]: ['Tab', 'Enter', 'Escape'],
  [InputContext.MODAL]: ['Escape', 'Enter'],
};
```

**Lookup Algorithm**:

```typescript
function shouldHandleKey(key: string, context: InputContext): boolean {
  const contextKeys = CONTEXT_KEYS[context] || [];

  // Global keys always handled
  const globalKeys = ['h', 'Space', 'Escape', 'm', 'Control+l'];

  return contextKeys.includes(key) || globalKeys.includes(key);
}
```

---

## 3. Typing Detection

### 3.1 Active Element Detection

**Purpose**: Automatically detect when user is typing in a text field.

**Algorithm**:

```typescript
function isTypingInInput(): boolean {
  const active = document.activeElement;

  if (!active) return false;

  // Check element types that accept text
  if (active.tagName === 'INPUT') {
    const type = (active as HTMLInputElement).type;
    const textTypes = ['text', 'search', 'url', 'email', 'password', 'tel', 'number'];
    return textTypes.includes(type);
  }

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

### 3.2 Automatic Context Switching

**Focus Event Handling**:

```typescript
// When text input gains focus
inputElement.addEventListener('focus', () => {
  contextManager.pushContext(InputContext.TYPING);
});

// When text input loses focus
inputElement.addEventListener('blur', () => {
  if (contextManager.getCurrentContext() === InputContext.TYPING) {
    contextManager.popContext();
  }
});
```

---

## 4. Key Filtering

### 4.1 Modifier Key Handling

**Purpose**: Distinguish between plain keys and modified keys (Ctrl+X, Shift+W).

**Algorithm**:

```typescript
function getEffectiveKey(event: KeyboardEvent): string {
  let key = event.key;

  // Build key string with modifiers
  const modifiers = [];
  if (event.ctrlKey || event.metaKey) modifiers.push('Control');
  if (event.altKey) modifiers.push('Alt');
  if (event.shiftKey) modifiers.push('Shift');

  if (modifiers.length > 0) {
    key = modifiers.join('+') + '+' + key;
  }

  return key;
}
```

**Examples**:

- `Ctrl+L` → `"Control+l"`
- `Shift+W` → `"Shift+w"` (speed boost in fly mode)
- `Alt+W` → `"Alt+w"` (move up in fly mode)

### 4.2 Conflict Resolution

**Problem**: WASD keys should type in text fields but move camera in fly mode.

**Solution**:

```typescript
function handleKeyDown(event: KeyboardEvent): void {
  // Priority 1: Typing context overrides everything
  if (isTypingInInput()) {
    return; // Browser handles typing
  }

  const key = event.key.toLowerCase();
  const context = contextManager.getCurrentContext();

  // Priority 2: Check if key valid for current context
  if (context === InputContext.FLY_CONTROLS) {
    if (['w', 'a', 's', 'd'].includes(key)) {
      event.preventDefault();
      flyControls.handleKeyDown(event);
      return;
    }
  }

  // Priority 3: Global shortcuts
  if (key === 'h') {
    event.preventDefault();
    toggleHelp();
    return;
  }

  // Priority 4: Context-specific handlers
  routeToContextHandler(event, context);
}
```

---

## Data Structures

### InputContextManager

```typescript
interface InputContextManager {
  contextStack: InputContext[];
  debugMode: boolean;

  setContext(context: InputContext): void;
  pushContext(context: InputContext): void;
  popContext(): InputContext | undefined;
  getCurrentContext(): InputContext;
  shouldHandleKey(key: string): boolean;
  clearContextStack(): void;
}
```

### InputHandler

```typescript
interface InputHandler {
  container: HTMLElement;
  controlsManager: ControlsManager;
  contextManager: InputContextManager;

  handleKeyDown(event: KeyboardEvent): void;
  handleKeyUp(event: KeyboardEvent): void;
  handleMouseDown(event: MouseEvent): void;
  isTypingInInput(): boolean;
  dispose(): void;
}
```

---

## Changelog

- **v1.0.0** (2025-01-30): Initial specification
  - Context-based input routing with priority system
  - Context stack for nested states
  - Automatic typing detection
  - Mode-specific key filtering (WASD only in fly mode)
  - Modifier key support (Ctrl, Alt, Shift)
  - Conflict resolution algorithms
