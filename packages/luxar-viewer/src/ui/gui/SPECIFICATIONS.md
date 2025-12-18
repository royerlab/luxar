# Custom GUI Library - Technical Specifications

Technical specifications for the custom GUI library that replaces lil-gui.

## Architecture

### Class Hierarchy

```
GUI (extends Folder)
  └─ Folder
       ├─ Controller (abstract)
       │   ├─ NumberController
       │   ├─ BooleanController
       │   ├─ StringController
       │   ├─ OptionController
       │   └─ FunctionController
       └─ Folder (nested)
```

### Design Decisions

1. **GUI extends Folder**: Root panel inherits folder functionality
2. **Abstract Controller**: Common interface for all controller types
3. **EventManager**: Centralized listener tracking prevents memory leaks
4. **Direct CSS Variables**: No JavaScript mapping, themes work automatically

## Controller Type Detection

The `add()` method auto-detects controller type:

```typescript
add(object, property, arg1?, arg2?, arg3?) {
  const value = object[property];

  if (typeof value === 'function') → FunctionController
  if (typeof value === 'boolean') → BooleanController
  if (typeof value === 'string') {
    if (arg1 is array/object) → OptionController
    else → StringController
  }
  if (typeof value === 'number') {
    if (arg1 is array/object) → OptionController
    if (arg1 is number) → NumberController(min, max, step)
    else → NumberController(no range)
  }
}
```

**Order matters**: Most specific types checked first.

## Memory Management

### EventManager Pattern

**Critical**: All event listeners MUST be tracked.

```typescript
class EventManager {
  private listeners: EventListenerRecord[] = [];

  add(element, event, handler, options?) {
    element.addEventListener(event, handler, options);
    this.listeners.push({ element, event, handler, options });
  }

  removeAll() {
    for (const record of this.listeners) {
      record.element.removeEventListener(record.event, record.handler, record.options);
    }
    this.listeners = [];
  }
}
```

### Disposal Chain

```
GUI.destroy()
  → Folder.dispose() (recursive)
    → Controller.dispose() (each)
      → EventManager.removeAll()
    → Folder.dispose() (nested)
```

## lil-gui Behavior Compatibility

### setValue() Does NOT Trigger Callbacks

```typescript
// lil-gui behavior: setValue() updates value silently
controller.setValue(50); // Does NOT call onChange

// Only user interaction triggers callbacks
slider.dispatchEvent(new Event('input')); // DOES call onChange
```

**Rationale**: Prevents infinite loops in bidirectional sync patterns.

### Auto-blur After Interaction

Inputs blur automatically to restore keyboard shortcuts:

| Input Type   | Blur Trigger        |
| ------------ | ------------------- |
| Checkbox     | After change        |
| Slider       | On mouseup/touchend |
| Number Input | On Enter key        |
| Select       | After change        |

## DOM Structure (BEM Naming)

### Root GUI

```html
<div class="luxar-gui">
  <div class="luxar-gui__title">Controls</div>
  <div class="luxar-gui__children">
    <!-- controllers/folders -->
  </div>
</div>
```

### Controller

```html
<div class="luxar-gui__controller luxar-gui__controller--number">
  <label class="luxar-gui__controller-name">FOV</label>
  <div class="luxar-gui__controller-widget">
    <input type="range" class="luxar-gui__slider" />
    <input type="number" class="luxar-gui__input luxar-gui__input--number" />
  </div>
</div>
```

### Folder

```html
<div class="luxar-gui__folder luxar-gui__folder--open">
  <div class="luxar-gui__folder-title">
    <span class="luxar-gui__folder-caret">▼</span>
    Camera
  </div>
  <div class="luxar-gui__children">
    <!-- nested controllers -->
  </div>
</div>
```

## CSS Variable Integration

### Direct Variable Usage

```css
.luxar-gui {
  background: var(--luxar-bg-secondary);
  color: var(--luxar-text-primary);
  font-family: var(--luxar-font-base);
  border-radius: var(--luxar-radius-md);
  box-shadow: var(--luxar-shadow-lg);
}
```

### Theme-Specific Overrides

```css
[data-theme='light'] .luxar-gui {
  background: rgba(255, 255, 255, 0.95);
}

[data-theme='frosted-glass'] .luxar-gui {
  backdrop-filter: var(--luxar-blur-md) saturate(1.5);
}

[data-theme='liquid-glass'] .luxar-gui {
  backdrop-filter: var(--luxar-blur-md);
}
```

## Special Patterns

### Logarithmic Slider (HDR Intensity)

```typescript
// Shadow object pattern
const hdrLogValue = { log: Math.log10(settings.hdrMultiplier) };

const control = gui.add(hdrLogValue, 'log', -2, 2, 0.01);

// Override display to show actual value
control.setCustomUpdateDisplay(() => {
  const actual = Math.pow(10, hdrLogValue.log);
  control.$input.value = formatNumber(actual);
});

// onChange converts log to actual
control.onChange((logValue) => {
  settings.hdrMultiplier = Math.pow(10, logValue);
});
```

### Controller Synchronization

```typescript
// FOV preset → FOV slider sync
const fovControl = gui.add(settings, 'fov', 10, 170);
const presetControl = gui.add(settings, 'fovPreset', presets).onChange((preset) => {
  settings.fov = presetValues[preset];
  fovControl.setValue(settings.fov); // Silent update
  fovControl.updateDisplay(); // Refresh UI
});
```

### Conditional Visibility

```typescript
const dampingControl = gui.add(settings, 'damping', 0.9, 0.999);
gui.add(settings, 'inertialMode').onChange((enabled) => {
  enabled ? dampingControl.show() : dampingControl.hide();
});

// Initial state
if (!settings.inertialMode) dampingControl.hide();
```

## Value Formatting

### Number Display

```typescript
function formatNumber(value: number, step?: number): string {
  if (step === undefined) return String(value);

  // Count decimal places in step
  const stepStr = String(step);
  const decimalIndex = stepStr.indexOf('.');
  const decimals = decimalIndex === -1 ? 0 : stepStr.length - decimalIndex - 1;

  return value.toFixed(decimals);
}
```

### Value Clamping

```typescript
function clamp(value: number, min?: number, max?: number): number {
  if (min !== undefined && value < min) return min;
  if (max !== undefined && value > max) return max;
  return value;
}
```

## Testing Strategy

### Unit Tests

Each class has dedicated tests:

- `gui.test.ts` - Root GUI functionality
- `folder.test.ts` - Folder open/close, nesting
- `*-controller.test.ts` - Individual controller types

### Key Test Cases

1. **Memory Leak Prevention**

   ```typescript
   it('should clean up all event listeners on dispose', () => {
     const spy = vi.spyOn(controller['eventManager'], 'removeAll');
     controller.dispose();
     expect(spy).toHaveBeenCalled();
   });
   ```

2. **setValue() Behavior**

   ```typescript
   it('should NOT trigger onChange when setValue is called', () => {
     const onChange = vi.fn();
     controller.onChange(onChange);
     controller.setValue(50);
     expect(onChange).not.toHaveBeenCalled();
   });
   ```

3. **Controller Type Detection**
   ```typescript
   it('should create NumberController for numeric values', () => {
     const ctrl = folder.add({ x: 50 }, 'x', 0, 100);
     expect(ctrl).toBeInstanceOf(NumberController);
   });
   ```

## Migration from lil-gui

### Import Change

```diff
- import GUI from 'lil-gui';
+ import GUI from './gui';
```

### API Compatibility

All existing code works without changes:

- Same constructor signature
- Same `add()` method signatures
- Same controller methods
- Same folder methods

### Removed Dependencies

```diff
// package.json
- "lil-gui": "^0.20.0"
```

## Performance Considerations

- Controllers created synchronously (no async)
- Event listeners added once per control
- CSS transitions handled by browser
- No JavaScript animation loops for UI

## File Reference

| File                   | Purpose                 | Lines         |
| ---------------------- | ----------------------- | ------------- |
| `core/gui.ts`          | Root panel class        | ~125          |
| `core/folder.ts`       | Folder implementation   | ~315          |
| `core/controller.ts`   | Base controller         | ~250          |
| `core/types.ts`        | Type definitions        | ~65           |
| `controllers/*.ts`     | Specialized controllers | ~50-200 each  |
| `dom/event-manager.ts` | Memory management       | ~85           |
| `utils/*.ts`           | Utility functions       | ~50-70 each   |
| `styles/*.css`         | Styling                 | ~100-300 each |
