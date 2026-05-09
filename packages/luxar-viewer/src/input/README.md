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
├── input-handler-utils.ts   # Pure utility functions (dimension nav, FOV, camera)
└── README.md               # This documentation
```

---

## Three-layer architecture (read this first)

`input-handler.ts` is **not** the place where every input event in Luxar is
handled — its scope is viewer-level. Input handling is deliberately split
across three layers, each owning a different scope. Pick the right layer
when adding a new listener.

### Layer 1 — `input/input-handler.ts` (viewer-wide)

Owns concerns that span the whole viewer surface:

- Window-level events: `resize`, `wheel`, `keydown`/`keyup`, `fullscreenchange`.
- Global keyboard shortcuts: `H` (help), `P` (perf), `R` (rendering controls),
  `N` (sliders), `T` (recording), dimension navigation `[`/`]`/`1–9`,
  control-mode switches `V`/`I`/`F`.
- Wheel-based zoom and FOV adjust (Ctrl+wheel).
- Fullscreen enter/exit canvas styling.

It hosts the `InputContextManager` (Layer 1's keyboard dispatcher) which is
the **only** place that should attach a `window` `keydown`/`keyup` listener
in the entire viewer.

### Layer 2 — `controls/luxar-fly-controls.ts` (camera motion)

Camera-control input — WASD, mouse-look, arrow keys, roll on Shift+wheel.

Fly-controls keyboard listeners are **mediated by Layer 1's
`InputContextManager`**, not registered independently. When constructed
with `externalInputManagement: true` (which `controls-manager.ts` always
does), the controls do not attach `window` keyboard listeners; instead
Layer 1 calls `controls.handleKeyDown(event)` / `handleKeyUp(event)` from
inside its key dispatcher. Mouse-look stays local because it's tied to the
canvas DOM element.

This seam is what prevents "two listeners both fire keydown" bugs while
fly mode is active.

### Layer 3 — UI-local handlers (in `ui/*.ts`)

UI components attach their own listeners for genuinely _local_ concerns
that have no business going through a viewer-wide handler:

- `ui/helpers.ts` — close help overlay on outside-click.
- `ui/dimension-sliders.ts` — context-menu close on outside-click /
  Escape inside a popup.
- `ui/rendering-controls.ts` — mousedown-capture for click-outside-panel.
- `ui/debug-console.ts` — mousemove/mouseup for drag-resize.

These listeners are scoped to the component's lifecycle (added in `init`,
removed in `dispose`), and they don't duplicate any Layer 1 concern. A
viewer-wide handler has no idea which of three popups should close on an
outside click — that's strictly local knowledge.

### Rule of thumb

Adding a new listener? Pick the layer by _who knows what_:

- **Affects the whole viewer** (resize, global shortcut, fullscreen) → Layer 1.
- **Camera motion** → Layer 2 (and register through `InputContextManager`,
  not `window`).
- **Local to one panel/overlay** (click-outside, drag handle, focus trap) →
  Layer 3, in the component's own dispose lifecycle.

Watch out in particular for:

- Two `keydown` listeners on `window` from different layers — Layer 1's
  `InputContextManager` is the _only_ place that should attach a global
  `keydown`.
- Document-level click handlers competing for "outside click" semantics —
  if two popups can be open simultaneously and both want the next click,
  use `OverlayManager` to mediate rather than racing handlers.

The rest of this README zooms into Layer 1 specifically.

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
InputHandler.onKeyDown/Up
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
  constructor(sceneManager: SceneManager, animationController: AnimationController);

  // Event handlers
  private onKeyDown(event: KeyboardEvent): void;
  private onKeyUp(event: KeyboardEvent): void;
  private onMouseDown(event: MouseEvent): void;

  // Context checks
  private isTypingInInput(): boolean;
  private shouldHandleKey(key: string): boolean;

  // Cleanup
  dispose(): void;
}
```

**Dimension Initialization Lifecycle:**

The `InputHandler` coordinates dimension system initialization through `initDimensionSliders()`:

```typescript
initDimensionSliders(): void {
  // 1. Initialize scene dimension manager from loaded scene
  if (!sceneDimsManager.initFromScene(this.sceneManager.scene)) {
    return; // No nD objects found
  }

  // 2. Create dimension slider UI
  this.dimensionSliders = new DimensionSliders({...});

  // 3. Register listener for dimension changes
  sceneDimsManager.addListener(() => {
    this.updateAllNDNodes();
    this.dimensionSliders?.update();
    this.animationController.startAnimation();
  });

  // 4. CRITICAL: Trigger initial update (lines 191-194)
  // This ensures data loads at correct initial slice position
  this.updateAllNDNodes();
  this.animationController.startAnimation();
}
```

**Why the Manual Initial Trigger?**

The `sceneDimsManager.initFromScene()` sets up dimension state but does NOT call `notifyListeners()` because listeners haven't been registered yet. The initial update is triggered manually after listener registration to ensure:

- Data loads at correct initial position (t=0 for time, channel=0 for channels)
- Slider position matches displayed data from first render
- No race condition between initialization and listener registration

See `input-handler.ts:191-194` for implementation and `scene-dims-manager.ts:157-160` for the rationale.

### 2. Input Context Manager

The `InputContextManager` manages input contexts to prevent conflicts between different UI systems.

**Context Types:**

```typescript
enum InputContext {
  NAVIGATION = 'navigation', // Default 3D navigation
  FLY_CONTROLS = 'fly_controls', // Fly mode active
  TYPING = 'typing', // Text input active
  DIMENSION_NAV = 'dimension_nav', // nD dimension navigation
  UI_INTERACTION = 'ui_interaction', // UI panels open
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

Contexts have explicit priority values configured in the manager:

```
TYPING (10)            ← Highest priority
UI_INTERACTION (5)
DIMENSION_NAV (2)
FLY_CONTROLS (1)
NAVIGATION (0)         ← Lowest priority
```

Higher priority contexts override lower ones.

### Context Stack

Support for nested contexts:

```typescript
// Example: Opening a modal while in fly mode
contextManager.setContext(InputContext.FLY_CONTROLS);
// User opens settings panel
contextManager.pushContext(InputContext.UI_INTERACTION);
// UI interaction captures relevant input

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
    case InputContext.UI_INTERACTION:
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
onKeyDown(event: KeyboardEvent) {
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
    contextManager.pushContext(InputContext.UI_INTERACTION);
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

| Key      | Action        | Description                                     |
| -------- | ------------- | ----------------------------------------------- |
| `H`      | Show help     | Display keyboard shortcuts                      |
| `Space`  | Fullscreen    | Toggle fullscreen mode                          |
| `Escape` | Exit/Close    | Exit fullscreen OR close panels (context-aware) |
| `M`      | Monitor       | Cycle data loading monitor                      |
| `Ctrl+L` | Debug console | Toggle debug console                            |

**Note on ESC key behavior:**

- In fullscreen mode: Exits fullscreen (handled by browser)
- Not in fullscreen: Closes all open UI panels (help, controls, etc.)
- ESC performs only one action at a time for predictable behavior

### Mode-Specific Keys

#### Navigation Keys

| Key | Action                                              |
| --- | --------------------------------------------------- |
| `F` | Recenter/focus camera on scene                      |
| `V` | Cycle control modes: Orbit -> Fly -> Ortho -> Orbit |
| `N` | Toggle dimension sliders                            |
| `O` | Open dataset browser                                |
| `P` | Toggle performance stats                            |
| `R` | Toggle rendering controls                           |
| `C` | Toggle cinematic mode                               |

#### Fly Mode

| Key       | Action                       |
| --------- | ---------------------------- |
| `W/A/S/D` | Move forward/left/back/right |
| `Q/E`     | Roll left/right              |
| `Alt+W/S` | Move up/down                 |
| `Shift`   | Speed boost                  |
| `I`       | Toggle inertia               |

#### Dimension Navigation

| Key       | Action                   |
| --------- | ------------------------ |
| `1-9`     | Select dimension         |
| `[`       | Step backward            |
| `]`       | Step forward             |
| `K`       | Play/pause animation     |
| `Home`    | Jump to dimension start  |
| `End`     | Jump to dimension end    |
| `Shift+↑` | Increase animation speed |
| `Shift+↓` | Decrease animation speed |

---

## Typing Detection

### Automatic Detection

The system automatically detects when users are typing:

```typescript
private isTypingInInput(): boolean {
  const activeElement = document.activeElement;
  if (!activeElement) return false;

  const tagName = activeElement.tagName.toLowerCase();
  // Exclude non-text input types (range sliders, checkboxes, radios)
  if (tagName === 'input') {
    const inputType = (activeElement as HTMLInputElement).type?.toLowerCase();
    if (inputType === 'range' || inputType === 'checkbox' || inputType === 'radio') {
      return false;
    }
    return true;
  }
  return (
    tagName === 'textarea' ||
    tagName === 'select' ||
    activeElement.getAttribute('contenteditable') === 'true'
  );
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
import { SceneManager } from './scene/scene-manager';
import { AnimationController } from './scene/animation-controller';
import { PerformanceMonitor } from './ui/monitors/performance-monitor';
import { DebugConsole } from './ui/panels/debug-console';

// Create input handler. PerformanceMonitor and DebugConsole are owned
// by LuxarApp and passed in here. The optional dimensionSlidersFactory
// keeps the `input/` layer free of `ui/` imports — pass
// `(c) => new DimensionSliders(c)` if dimension navigation is needed.
const inputHandler = new InputHandler(
  sceneManager,
  animationController,
  performanceMonitor,
  debugConsole,
  // optional: dimension sliders factory
);
inputHandler.init();
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
// When opening a UI panel
function openUIPanel() {
  contextManager.pushContext(InputContext.UI_INTERACTION);
  // UI panel now captures relevant input
}

// When closing
function closeModal() {
  contextManager.popContext();
  // Returns to previous context
}
```

### Adding New Keyboard Shortcuts

**IMPORTANT**: All key handling uses the unified binding registration system. To add a new shortcut, register it in `InputHandler.registerAllKeyBindings()`.

#### 1. Simple Binding (Keydown Only)

Most shortcuts only need keydown handling:

```typescript
this.contextManager.registerBinding(InputContext.NAVIGATION, {
  key: 'g',
  handler: () => this.myNewAction(),
  preventDefault: true,
  description: 'My new action',
});
```

#### 2. With Modifiers

```typescript
this.contextManager.registerBinding(InputContext.NAVIGATION, {
  key: 's',
  modifiers: { ctrl: true }, // Requires Ctrl+S
  handler: () => this.saveScene(),
  preventDefault: true,
  description: 'Save scene',
});
```

#### 3. With Keydown AND Keyup Handlers (NEW in v2.0)

For keys that need different up/down behavior:

```typescript
this.contextManager.registerBinding(InputContext.NAVIGATION, {
  key: 'Shift',
  handler: () => this.disableZoom(), // Called on keydown
  keyupHandler: () => this.enableZoom(), // Called on keyup
  description: 'Zoom control',
});

// Fly controls example
this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
  key: 'w',
  handler: (e) => flyControls.handleKeyDown(e), // Start movement
  keyupHandler: (e) => flyControls.handleKeyUp(e), // Stop movement
  description: 'Fly forward',
});
```

**When to use keyupHandler**:

- ✅ Keys that toggle state on press/release (Shift, modifier keys)
- ✅ Keys that start/stop continuous actions (fly movement)
- ❌ Toggle actions (ON/OFF) - use keydown only, no keyupHandler

#### 4. Context-Specific Bindings

Only active in specific modes:

```typescript
this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
  key: 'b',
  handler: () => this.boost(),
  description: 'Speed boost',
});
```

#### 5. Conditional preventDefault

For keys that shouldn't block browser shortcuts:

```typescript
this.contextManager.registerBinding(InputContext.NAVIGATION, {
  key: 'r',
  preventDefault: false, // Don't always prevent
  handler: (event) => {
    // Only execute without modifiers (allows Cmd+R browser refresh)
    if (!event.metaKey && !event.ctrlKey && !event.shiftKey) {
      event.preventDefault(); // Prevent here conditionally
      this.resetView();
    }
  },
});
```

#### 6. Multiple Modifier Combinations

For keys that work with different modifier combinations:

```typescript
// Register w, Shift+w, Alt+w, Shift+Alt+w separately
const modifierCombos = [{}, { shift: true }, { alt: true }, { shift: true, alt: true }];

for (const mods of modifierCombos) {
  this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
    key: 'w',
    modifiers: Object.keys(mods).length > 0 ? mods : undefined,
    handler: (e) => this.handleMovement(e),
    keyupHandler: (e) => this.stopMovement(e),
  });
}
```

**Automatic Features**:

- ✅ Context filtering (key only works in registered context)
- ✅ Case-insensitive matching ('h' and 'H' both work)
- ✅ Typing detection (shortcuts blocked when typing in inputs)
- ✅ Passthrough (unhandled keys pass to lower-priority contexts)
- ✅ keyupHandler called automatically on keyup events

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
[Input] Context pushed: UI_INTERACTION
[Input] Key filtered: 'w' (blocked by UI_INTERACTION context)
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
    this.container.removeEventListener('keydown', this.onKeyDown);
    this.container.removeEventListener('keyup', this.onKeyUp);
    this.container.removeEventListener('mousedown', this.onMouseDown);

    // Clear references
    this.controlsManager = null;
    this.contextManager = null;

    // Clear key states
    this.keysPressed.clear();
  }
}
```

---

## Utility Functions (input-handler-utils.ts)

The `input-handler-utils.ts` module provides pure utility functions for input processing. All functions are stateless and testable.

### Dimension Navigation Utilities

**`getNonDisplayedDimensions(dimensions, displayedDims)`**

- Returns array of non-displayed dimension indices for nD navigation
- Used to determine which dimensions can be navigated with keyboard shortcuts

**`mapKeyToDimension(key, nonDisplayedDims)`**

- Maps number keys (1-9) to dimension indices
- Returns dimension index or `null` if key doesn't map to a dimension
- Example: Key '1' → first non-displayed dimension

**`isNavigationKey(key, nonDisplayedDims)`**

- Checks if a key is valid for dimension navigation
- Returns `true` for: number keys (1-9), bracket keys ([, ]), or arrow keys if non-displayed dimensions exist

**`calculateStepSize(dimension, direction)`**

- Calculates step size for dimension navigation based on dimension configuration
- Uses dimension's `step` property if available
- Falls back to range-based calculation: `(max - min) / 100`
- `direction`: 1 for forward, -1 for backward

**`calculateNextPosition(currentPos, dimension, direction, stepSize)`**

- Calculates next slider position with proper clamping to dimension range
- Handles categorical dimensions (snaps to category indices)
- Clamps continuous dimensions to [min, max] range

**`formatDimensionValue(value, dimension)`**

- Formats dimension value for display
- Categorical: returns category label at index
- Continuous: returns number with appropriate precision
- Handles edge cases (out of range, missing categories)

**`generateNavigationHelp(dimensions, displayedDims)`**

- Generates help text showing dimension navigation keyboard shortcuts
- Returns formatted string like: "1: time [0-100], 2: channel [0-3]"
- Only includes non-displayed dimensions

### FOV (Field of View) Utilities

**`calculateFovChange(currentFov, direction, speed = 1.0)`**

- Calculates new FOV value for zoom operations
- `direction`: 1 for zoom out, -1 for zoom in
- `speed`: multiplier for zoom speed (default 1.0)
- Returns new FOV clamped to [10°, 120°] range
- Uses exponential scaling: 5° per step

**Example**:

```typescript
const newFov = calculateFovChange(60, -1); // Zoom in: 60° → 55°
```

### Camera Control Utilities

**`shouldBlockShortcut(key, modifiers)`**

- Determines if a keyboard shortcut should be blocked based on context
- Blocks shortcuts when typing in input fields, textareas, or contenteditable elements
- `modifiers`: object with `ctrl`, `shift`, `alt`, `meta` booleans
- Returns `true` if shortcut should be blocked

**Example**:

```typescript
if (shouldBlockShortcut('v', { ctrl: false, shift: false })) {
  return; // User is typing, don't trigger view mode switch
}
```

### Design Principles

1. **Pure Functions**: All utilities are pure functions with no side effects
2. **Stateless**: No internal state, all data passed as parameters
3. **Testable**: Easy to unit test with predictable inputs/outputs
4. **Type-Safe**: Full TypeScript type annotations
5. **Focused**: Each function has a single, clear responsibility

### Usage Example

```typescript
import {
  getNonDisplayedDimensions,
  mapKeyToDimension,
  calculateStepSize,
  calculateNextPosition,
  formatDimensionValue,
} from './input-handler-utils';

// Get dimensions available for keyboard navigation
const nonDisplayed = getNonDisplayedDimensions(allDimensions, displayedDims);

// Handle number key press
const dimIndex = mapKeyToDimension('1', nonDisplayed); // Maps to first non-displayed dim

if (dimIndex !== null) {
  const dimension = allDimensions[dimIndex];
  const stepSize = calculateStepSize(dimension, 1); // Forward direction
  const currentPos = sliderPositions[dimIndex];
  const nextPos = calculateNextPosition(currentPos, dimension, 1, stepSize);

  // Update slider
  updateSlider(dimIndex, nextPos);

  // Show formatted value
  console.log(`${dimension.name}: ${formatDimensionValue(nextPos, dimension)}`);
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

| Method                                           | Description                         |
| ------------------------------------------------ | ----------------------------------- |
| `constructor(sceneManager, animationController)` | Initialize with scene and animation |
| `initDimensionSliders()`                         | Initialize nD dimension UI          |
| `dispose()`                                      | Clean up event listeners            |

### InputContextManager

| Method                                        | Description                  |
| --------------------------------------------- | ---------------------------- |
| `registerBinding(context, binding)`           | Register a key binding       |
| `setContext(context)`                         | Set active context           |
| `pushContext(context)`                        | Add context to stack         |
| `popContext()`                                | Remove top context           |
| `getCurrentContext()`                         | Get active context           |
| `handleKeyDown(event)` / `handleKeyUp(event)` | Route key events to bindings |
| `setDebugMode(enabled)`                       | Toggle debug logging         |

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
