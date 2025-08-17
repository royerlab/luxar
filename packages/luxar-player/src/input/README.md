# Luxar Input Package

> Advanced input handling system with context-aware keyboard and mouse management

## Overview

The Luxar Input package provides a sophisticated input handling system that manages keyboard and mouse events across different application contexts. It prevents input conflicts, enables context-specific shortcuts, and ensures smooth interaction between 3D navigation, UI controls, and text input fields.

### Key Features

- **Context-Aware Input**: Different input handling for different modes
- **Conflict Prevention**: Automatic detection and prevention of input conflicts
- **Context Stack**: Nested context management for complex UI states
- **Typing Detection**: Automatic disabling of shortcuts during text input
- **Mode-Specific Keys**: Enable/disable key groups based on control mode
- **Event Delegation**: Centralized input handling with proper routing
- **Debug Support**: Comprehensive input event logging

### Package Architecture

```
input/
├── input-handler.ts         # Main input event processing
├── input-context-manager.ts # Context-based input routing
└── README.md               # This documentation
```

---

## Components

### 1. Input Handler

The `InputHandler` class centralizes all input event processing and delegates to appropriate systems.

**Responsibilities:**

- Capture keyboard and mouse events
- Detect typing in input fields
- Route events based on context
- Manage mode-specific key bindings
- Handle global shortcuts
- Coordinate with controls system

**Event Flow:**

```
Browser Event
    ↓
InputHandler.handleKeyDown/Up
    ↓
Check if typing in input field
    ↓
Check InputContext
    ↓
Filter based on active mode
    ↓
Route to appropriate handler
    ↓
Execute action
```

**Key Methods:**

```typescript
class InputHandler {
  constructor(
    container: HTMLElement,
    controlsManager: ControlsManager,
    contextManager: InputContextManager
  );

  // Event handlers
  private handleKeyDown(event: KeyboardEvent): void;
  private handleKeyUp(event: KeyboardEvent): void;
  private handleMouseDown(event: MouseEvent): void;

  // Context checks
  private isTypingInInput(): boolean;
  private shouldHandleKey(key: string): boolean;

  // Cleanup
  dispose(): void;
}
```

### 2. Input Context Manager

The `InputContextManager` manages input contexts to prevent conflicts between different UI systems.

**Context Types:**

```typescript
enum InputContext {
  NAVIGATION = 0, // Default 3D navigation
  FLY_CONTROLS = 1, // Fly mode active
  TYPING = 2, // Text input active
  DIMENSION_NAV = 3, // nD dimension navigation
  UI_OVERLAY = 4, // UI panels open
  MODAL = 5, // Modal dialog active
}
```

**Features:**

- Context priority system
- Context stack for nested states
- Automatic focus management
- Key filtering based on context
- Debug mode for troubleshooting

**Usage:**

```typescript
class InputContextManager {
  // Context management
  setContext(context: InputContext): void;
  pushContext(context: InputContext): void;
  popContext(): InputContext | undefined;
  clearContextStack(): void;

  // State queries
  getCurrentContext(): InputContext;
  hasContext(context: InputContext): boolean;
  shouldHandleKey(key: string): boolean;

  // Debug support
  setDebugMode(enabled: boolean): void;
}
```

---

## Context System

### Context Priority

Contexts have implicit priority based on their enum value:

```
MODAL (5)        ← Highest priority
UI_OVERLAY (4)
DIMENSION_NAV (3)
TYPING (2)
FLY_CONTROLS (1)
NAVIGATION (0)   ← Lowest priority
```

Higher priority contexts override lower ones.

### Context Stack

Support for nested contexts:

```typescript
// Example: Opening a modal while in fly mode
contextManager.setContext(InputContext.FLY_CONTROLS);
// User opens settings modal
contextManager.pushContext(InputContext.MODAL);
// Modal captures all input

// User closes modal
contextManager.popContext();
// Returns to FLY_CONTROLS context
```

### Context-Specific Key Filtering

Each context defines which keys it handles:

```typescript
// Navigation context: Arrow keys for camera
// Fly controls: WASD for movement
// Typing: All keys for text input
// Dimension nav: Number keys and brackets
// UI overlay: Tab, Enter, Escape
// Modal: Escape to close

function getContextKeys(context: InputContext): string[] {
  switch (context) {
    case InputContext.FLY_CONTROLS:
      return ['w', 'a', 's', 'd', 'q', 'e', 'shift'];
    case InputContext.DIMENSION_NAV:
      return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']'];
    case InputContext.MODAL:
      return ['escape', 'enter'];
    default:
      return [];
  }
}
```

---

## Input Routing

### Keyboard Event Routing

Complete keyboard event flow:

```typescript
handleKeyDown(event: KeyboardEvent) {
  // 1. Check if typing in input field
  if (this.isTypingInInput()) {
    return; // Let browser handle it
  }

  // 2. Get current context
  const context = contextManager.getCurrentContext();

  // 3. Check if context handles this key
  if (!contextManager.shouldHandleKey(event.key)) {
    return;
  }

  // 4. Route to appropriate handler
  switch (context) {
    case InputContext.FLY_CONTROLS:
      this.handleFlyControlsKey(event);
      break;
    case InputContext.DIMENSION_NAV:
      this.handleDimensionKey(event);
      break;
    default:
      this.handleGlobalKey(event);
  }
}
```

### Mouse Event Routing

Mouse events with context awareness:

```typescript
handleMouseDown(event: MouseEvent) {
  // Check if clicking on UI element
  if (this.isUIElement(event.target)) {
    contextManager.pushContext(InputContext.UI_OVERLAY);
    return;
  }

  // Otherwise handle as navigation
  this.controlsManager.handleMouseDown(event);
}
```

---

## Key Bindings

### Global Shortcuts

Always available regardless of context:

| Key      | Action        | Description                |
| -------- | ------------- | -------------------------- |
| `H`      | Show help     | Display keyboard shortcuts |
| `F11`    | Fullscreen    | Toggle fullscreen mode     |
| `Ctrl+S` | Screenshot    | Capture current view       |
| `Ctrl+L` | Debug console | Toggle debug console       |

### Mode-Specific Keys

#### Orbit Mode

| Key     | Action           |
| ------- | ---------------- |
| `←→↑↓`  | Rotate camera    |
| `+/-`   | Zoom in/out      |
| `C`     | Toggle centering |
| `Space` | Reset view       |

#### Fly Mode

| Key       | Action                       |
| --------- | ---------------------------- |
| `W/A/S/D` | Move forward/left/back/right |
| `Q/E`     | Roll left/right              |
| `Alt+W/S` | Move up/down                 |
| `Shift`   | Speed boost                  |
| `I`       | Toggle inertia               |

#### Dimension Navigation

| Key         | Action            |
| ----------- | ----------------- |
| `1-9`       | Select dimension  |
| `[`         | Step backward     |
| `]`         | Step forward      |
| `Shift+[/]` | Jump to start/end |

---

## Typing Detection

### Automatic Detection

The system automatically detects when users are typing:

```typescript
private isTypingInInput(): boolean {
  const activeElement = document.activeElement;

  if (!activeElement) return false;

  // Check if focused element accepts text input
  const isTextInput =
    activeElement.tagName === 'INPUT' ||
    activeElement.tagName === 'TEXTAREA' ||
    activeElement.contentEditable === 'true';

  // Check for specific input types
  if (activeElement.tagName === 'INPUT') {
    const type = (activeElement as HTMLInputElement).type;
    const textTypes = ['text', 'search', 'url', 'email', 'password'];
    return textTypes.includes(type);
  }

  return isTextInput;
}
```

### Manual Override

Force typing context for custom components:

```typescript
// When custom component gains focus
inputElement.addEventListener('focus', () => {
  contextManager.pushContext(InputContext.TYPING);
});

// When it loses focus
inputElement.addEventListener('blur', () => {
  contextManager.popContext();
});
```

---

## Usage Examples

### Basic Setup

```typescript
import { InputHandler } from './input/input-handler';
import { InputContextManager, InputContext } from './input/input-context-manager';

// Initialize context manager
const contextManager = new InputContextManager();

// Create input handler
const inputHandler = new InputHandler(document.body, controlsManager, contextManager);

// Set initial context
contextManager.setContext(InputContext.NAVIGATION);
```

### Mode Switching

```typescript
// Switch to fly controls
function enableFlyMode() {
  controlsManager.setControlType('fly');
  contextManager.setContext(InputContext.FLY_CONTROLS);
}

// Return to orbit mode
function enableOrbitMode() {
  controlsManager.setControlType('orbit');
  contextManager.setContext(InputContext.NAVIGATION);
}
```

### UI Integration

```typescript
// When opening a modal
function openModal() {
  contextManager.pushContext(InputContext.MODAL);
  // Modal now captures all input
}

// When closing
function closeModal() {
  contextManager.popContext();
  // Returns to previous context
}
```

### Custom Input Handling

```typescript
// Add custom key handler
class CustomHandler {
  constructor(private contextManager: InputContextManager) {
    document.addEventListener('keydown', this.handleKey.bind(this));
  }

  handleKey(event: KeyboardEvent) {
    // Only handle in specific context
    if (this.contextManager.getCurrentContext() !== InputContext.CUSTOM) {
      return;
    }

    // Custom key handling
    switch (event.key) {
      case 'x':
        this.customAction();
        break;
    }
  }
}
```

---

## Configuration

### Input Settings

```typescript
const INPUT_CONFIG = {
  // Key repeat
  keyRepeatDelay: 500, // ms before repeat starts
  keyRepeatRate: 30, // ms between repeats

  // Mouse
  mouseSensitivity: 1.0,
  invertMouse: false,

  // Context switching
  contextSwitchDelay: 0, // ms delay when switching

  // Debug
  logInputEvents: false,
  showContextIndicator: false,
};
```

### Key Binding Configuration

```typescript
// Define custom key bindings
const KEY_BINDINGS = {
  navigation: {
    rotateLeft: 'ArrowLeft',
    rotateRight: 'ArrowRight',
    zoomIn: '+',
    zoomOut: '-',
  },

  fly: {
    forward: 'w',
    backward: 's',
    strafeLeft: 'a',
    strafeRight: 'd',
    rollLeft: 'q',
    rollRight: 'e',
  },

  global: {
    help: 'h',
    fullscreen: 'F11',
    screenshot: 'F12',
  },
};
```

---

## Debug Mode

### Enabling Debug Mode

```typescript
// Enable comprehensive input logging
contextManager.setDebugMode(true);

// Also available via console
window.__luxarDebug = { input: true };
```

### Debug Output

With debug mode enabled:

```
[Input] Context changed: NAVIGATION → FLY_CONTROLS
[Input] Key down: 'w' (context: FLY_CONTROLS)
[Input] Routing to fly controls handler
[Input] Key up: 'w'
[Input] Context pushed: MODAL
[Input] Key filtered: 'w' (blocked by MODAL context)
```

### Context Visualization

Show current context on screen:

```typescript
function showContextIndicator() {
  const indicator = document.createElement('div');
  indicator.className = 'context-indicator';
  indicator.textContent = `Context: ${InputContext[currentContext]}`;
  document.body.appendChild(indicator);
}
```

---

## Performance Considerations

### Event Handling Optimization

1. **Event Delegation**: Single listener at root level
2. **Early Returns**: Quick checks to avoid processing
3. **Passive Listeners**: For scroll and touch events
4. **Throttling**: Limit high-frequency events
5. **Key Caching**: Cache frequently checked keys

### Memory Management

```typescript
class InputHandler {
  dispose() {
    // Remove all event listeners
    this.container.removeEventListener('keydown', this.handleKeyDown);
    this.container.removeEventListener('keyup', this.handleKeyUp);
    this.container.removeEventListener('mousedown', this.handleMouseDown);

    // Clear references
    this.controlsManager = null;
    this.contextManager = null;

    // Clear key states
    this.keysPressed.clear();
  }
}
```

---

## Best Practices

### Context Management

1. **Use appropriate contexts**: Choose the right context for each UI state
2. **Clean up contexts**: Always pop pushed contexts
3. **Avoid context conflicts**: Don't set conflicting contexts simultaneously
4. **Document context usage**: Clear comments about context requirements
5. **Test context transitions**: Verify smooth switching

### Input Handling

1. **Prevent default carefully**: Only when necessary
2. **Stop propagation sparingly**: Can break other handlers
3. **Check active element**: Respect focus state
4. **Handle edge cases**: Cmd vs Ctrl, different keyboards
5. **Provide feedback**: Visual/audio confirmation

---

## Troubleshooting

### Common Issues

**Problem: Keys not working**

```typescript
// Check context
console.log('Current context:', contextManager.getCurrentContext());
// Check if typing
console.log('Typing:', inputHandler.isTypingInInput());
// Enable debug mode
contextManager.setDebugMode(true);
```

**Problem: Input conflicts**

```typescript
// Review context stack
console.log('Context stack:', contextManager.getContextStack());
// Clear stuck contexts
contextManager.clearContextStack();
```

**Problem: Keys working in wrong mode**

```typescript
// Verify key filtering
const shouldHandle = contextManager.shouldHandleKey('w');
console.log('Should handle W:', shouldHandle);
```

---

## API Reference

### InputHandler

| Method                                      | Description                   |
| ------------------------------------------- | ----------------------------- |
| `constructor(container, controls, context)` | Initialize handler            |
| `setEnabled(enabled)`                       | Enable/disable input handling |
| `dispose()`                                 | Clean up event listeners      |

### InputContextManager

| Method                  | Description                    |
| ----------------------- | ------------------------------ |
| `setContext(context)`   | Set active context             |
| `pushContext(context)`  | Add context to stack           |
| `popContext()`          | Remove top context             |
| `getCurrentContext()`   | Get active context             |
| `hasContext(context)`   | Check if context active        |
| `shouldHandleKey(key)`  | Check if key should be handled |
| `clearContextStack()`   | Reset context stack            |
| `setDebugMode(enabled)` | Toggle debug logging           |

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
