# Custom GUI Library

Drop-in replacement for lil-gui with native Luxar theme integration.

## Overview

This custom GUI library provides a clean, themeable control panel for the Luxar viewer. It implements the same API as lil-gui but integrates directly with the Luxar CSS variable system.

## Features

- **Theme Integration**: Uses Luxar CSS variables for all styling
- **Memory Leak Prevention**: Centralized EventManager tracks all listeners
- **Auto-blur Behavior**: Inputs automatically blur after interaction to preserve keyboard shortcuts
- **Full API Compatibility**: Drop-in replacement for lil-gui

## Usage

```typescript
import { GUI } from '../ui/gui';

// Create GUI panel
const gui = new GUI({
  title: 'Controls',
  width: 300,
});

// Add controls
gui.add(settings, 'fov', 10, 170, 1).name('Field of View');
gui.add(settings, 'autoRotate').name('Auto Rotate');
gui.add(settings, 'preset', ['Low', 'Medium', 'High']).name('Quality');

// Create folders
const cameraFolder = gui.addFolder('Camera');
cameraFolder.add(settings, 'near', 0.1, 10).name('Near Plane');

// Show the panel
gui.show();
```

## Controller Types

| Type                 | Input          | Use Case                  |
| -------------------- | -------------- | ------------------------- |
| `NumberController`   | Slider + Input | Numeric values with range |
| `BooleanController`  | Checkbox       | Boolean toggles           |
| `StringController`   | Text Input     | String values             |
| `OptionController`   | Dropdown       | Enumerated choices        |
| `FunctionController` | Button         | Function calls            |

## API Reference

### GUI

```typescript
new GUI(options?: GUIOptions)
```

Options:

- `title?: string` - Panel title (default: 'Controls')
- `width?: number` - Panel width in pixels (default: 300)
- `closeFolders?: boolean` - Start folders closed (default: false)
- `container?: HTMLElement` - Parent element (default: document.body)
- `onClose?: () => void` - Callback fired when the close button is clicked. Supplying this option also causes a close button (`×`) to render in the header.
- `closeButtonTitle?: string` - Tooltip text for the close button (e.g. `"Close (R)"`).

Methods (inherited from `Folder`, plus root-only `show`/`hide`/`destroy`):

- `add(object, property, ...args)` - Add a controller (type auto-detected from the property value and extra args)
- `addFolder(name)` - Add a nested folder
- `controllersRecursive()` - Return all controllers including those in nested folders
- `open()` / `close()` - Toggle folder open/closed state (no-op on the root GUI)
- `show()` / `hide()` - Toggle visibility
- `destroy()` - Dispose all children, remove DOM, clean up listeners

### Controller

All controllers share these methods:

- `name(label)` - Set display label
- `setValue(value)` - Set value programmatically
- `getValue()` - Get current value
- `updateDisplay()` - Refresh UI from object
- `onChange(callback)` - Register change handler
- `onFinishChange(callback)` - Register finish handler
- `show()` / `hide()` - Toggle visibility
- `dispose()` - Clean up resources

NumberController additional methods:

- `min(value)` - Set minimum
- `max(value)` - Set maximum
- `step(value)` - Set step size

## Memory Management

**Critical**: All event listeners are tracked by `EventManager` and must be cleaned up.

```typescript
// Automatic cleanup on destroy
gui.destroy();

// Or dispose individual controllers
controller.dispose();
```

## Theming

The GUI uses Luxar CSS variables directly:

```css
.luxar-gui {
  background: var(--luxar-bg-secondary);
  color: var(--luxar-text-primary);
  border-radius: var(--luxar-radius-md);
}
```

Theme changes are automatic - no JavaScript intervention needed.

## File Structure

```
gui/
├── gui.ts                # Root GUI class (extends Folder; root DOM, show/hide/destroy)
├── folder.ts             # Folder container: add() factory, addFolder(), open/close state
├── controller.ts         # Base Controller: target/property binding, callbacks, dispose
├── types.ts              # GUIOptions, ControllerOptions, ControllerType, callback types
├── controllers/          # Concrete controllers (Number, Boolean, String, Option, Function)
├── format/               # Input helpers (auto-blur behaviour, number parsing/formatting)
├── dom/
│   └── event-manager.ts  # Tracks listeners for guaranteed cleanup
└── styles/               # Scoped CSS (gui.css, controller.css, folder.css)
```

The public entry point lives one level up at [`../gui.ts`](../gui.ts),
which re-exports `GUI` (default and named) along with `Folder`,
`Controller`, and the concrete controller classes.

## Subpackages

- [controllers/](./controllers/README.md) — Concrete subclasses: `NumberController`, `BooleanController`, `StringController`, `OptionController`, `FunctionController`.
- [format/](./format/README.md) — Input formatting helpers: `applyAutoBlur`, `clamp`, `formatNumber`, `parseNumber`.
- [dom/](./dom/README.md) — DOM plumbing for the GUI; currently the centralized `EventManager` used by every controller for guaranteed listener cleanup.
- [styles/](./styles/README.md) — Scoped CSS for the GUI, split into `gui.css`, `controller.css`, and `folder.css`; namespaced under `.luxar-gui`.

## See Also

- [`../rendering-controls/`](../rendering-controls/README.md) — Primary consumer of this library.
- [`../gui.ts`](../gui.ts) — Public re-export module that this folder backs.
