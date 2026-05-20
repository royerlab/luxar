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
import GUI from './gui';

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

Methods:

- `add(object, property, ...args)` - Add a controller
- `addFolder(name)` - Add a nested folder
- `show()` / `hide()` - Toggle visibility
- `destroy()` - Clean up resources

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
├── index.ts              # Public exports
├── core/
│   ├── gui.ts           # Root GUI class
│   ├── folder.ts        # Folder implementation
│   ├── controller.ts    # Base controller
│   └── types.ts         # Type definitions
├── controllers/
│   ├── number-controller.ts
│   ├── boolean-controller.ts
│   ├── string-controller.ts
│   ├── option-controller.ts
│   └── function-controller.ts
├── dom/
│   └── event-manager.ts # Memory leak prevention
├── utils/
│   ├── auto-blur.ts     # Auto-blur behavior
│   └── value-formatting.ts
└── styles/
    ├── gui.css          # Root panel styles
    ├── controller.css   # Controller styles
    └── folder.css       # Folder styles
```

## See Also

- `../../rendering-controls/` - Primary consumer of this library
