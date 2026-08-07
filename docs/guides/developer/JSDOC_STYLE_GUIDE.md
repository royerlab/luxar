# JSDoc Style Guide for Luxar Viewer

This guide ensures consistency across all TypeScript documentation in the luxar-viewer project.

## General Principles

1. **Every exported function/class/interface must have JSDoc**
2. **Use complete sentences** with proper capitalization and punctuation
3. **Focus on WHY and HOW**, not just WHAT (code shows what)
4. **Include examples** for complex or frequently-used APIs
5. **Document error conditions** with `@throws`
6. **Cross-reference** related code with `@see`

## File-Level Documentation

Every TypeScript file should start with a file-level comment:

```typescript
/**
 * Module description (1-2 sentences).
 *
 * Additional context about the module's role, key concepts,
 * or important design decisions (2-3 sentences).
 *
 * Key components:
 * - ComponentName: Purpose and responsibility
 * - ComponentName: Purpose and responsibility
 *
 * @module package-name/module-name
 */
```

**Example**:
```typescript
/**
 * Input handling system with context-aware key binding management.
 *
 * Provides a hierarchical input context system that allows different
 * parts of the UI to handle keyboard and mouse input independently
 * without conflicts. Uses event delegation for efficiency.
 *
 * Key components:
 * - InputHandler: Main input coordinator and event router
 * - InputContextManager: Context stack and priority management
 * - Navigation utilities: Dimension navigation helpers
 *
 * @module input/input-handler
 */
```

---

## Function Documentation

### Complete Template

```typescript
/**
 * One-line summary of what the function does (imperative mood).
 *
 * Detailed description providing context, design rationale, or
 * important behavior notes (2-4 sentences). Explain edge cases,
 * performance characteristics, or when to use this vs alternatives.
 *
 * @param paramName - Description including type constraints, valid ranges,
 *                    units, and any important behavior. Use hyphens for multi-line.
 * @param optionalParam - Optional parameters should indicate default behavior
 *                        if not provided
 * @returns Description of return value, including type information,
 *          possible values, and what they signify
 * @throws {ErrorType} When and why this error is thrown (be specific)
 * @throws {AnotherError} Document all possible error conditions
 *
 * @example
 * ```typescript
 * // Basic usage with explanation
 * const result = functionName(arg1, arg2);
 * console.log(result); // Expected output
 * ```
 *
 * @example
 * ```typescript
 * // Advanced usage or error handling
 * try {
 *   const result = functionName(complexArg);
 * } catch (error) {
 *   console.error('Failed:', error);
 * }
 * ```
 *
 * @see {@link RelatedFunction} for related functionality
 * @see {@link ../package/README.md#section} for usage guide
 */
```

### Minimal Acceptable (Simple Functions)

For simple functions with obvious behavior:

```typescript
/**
 * One-line summary.
 *
 * @param param - Brief description
 * @returns Brief description
 */
```

---

## Real Examples from Luxar

### Example 1: Public Method with Complex Logic

```typescript
/**
 * Navigate forward or backward in the specified dimension.
 *
 * Advances or retreats the slice position by the dimension's step size,
 * respecting min/max bounds. Emits 'dimensionChanged' event on success.
 * Frame-rate independent: step size determines travel distance.
 *
 * @param dimIndex - Zero-based index of dimension to navigate (0=X, 1=Y, etc.)
 *                   Must be < total dimensionality
 * @param direction - Navigation direction: 1 for forward, -1 for backward
 * @returns true if navigation succeeded, false if already at boundary
 *
 * @throws {Error} If dims not initialized (call initFromScene first)
 *
 * @example
 * ```typescript
 * // Navigate forward in time dimension (typically index 3 for 4D data)
 * const success = inputHandler.navigateDimension(3, 1);
 * if (!success) {
 *   console.log('Already at last time point');
 * }
 * ```
 *
 * @example
 * ```typescript
 * // Bind to arrow keys
 * document.addEventListener('keydown', (e) => {
 *   if (e.key === 'ArrowLeft') {
 *     inputHandler.navigateDimension(currentDim, -1);
 *   }
 * });
 * ```
 *
 * @see {@link NavigationManager.clampPosition} for clamping logic
 * @see {@link SceneDimsManager.getDims} for dimension metadata
 */
navigateDimension(dimIndex: number, direction: 1 | -1): boolean {
  // Implementation...
}
```

### Example 2: Interface Documentation

```typescript
/**
 * Configuration for dimension sliders UI component.
 *
 * Controls the appearance and behavior of the nD navigation sliders.
 * All visual properties use CSS units (px, rem, etc.).
 */
export interface SliderConfig {
  /**
   * Container element for slider UI.
   * Sliders will be appended as children to this element.
   */
  container: HTMLElement;

  /**
   * Initial dimensions to display.
   * If null, dimensions will be initialized from first loaded scene.
   * @default null
   */
  initialDims?: DimensionMetadata[] | null;

  /**
   * Slider track height in CSS units.
   * Affects click target size and visual prominence.
   * @default '4px'
   */
  trackHeight?: string;

  /**
   * Show numeric value labels next to sliders.
   * Labels display current slice position with dimension units.
   * @default true
   */
  showLabels?: boolean;
}
```

### Example 3: Class Documentation

```typescript
/**
 * Manages keyboard and mouse input with context-aware routing.
 *
 * Coordinates input across multiple UI contexts (main view, modal dialogs,
 * text input fields) using a priority-based context stack. Higher priority
 * contexts receive events first and can prevent propagation to lower contexts.
 *
 * Uses event delegation for efficiency - one global listener handles all
 * input and routes to appropriate contexts. This avoids performance issues
 * from adding/removing many individual listeners.
 *
 * Event Flow:
 * 1. Global listener captures event
 * 2. InputContextManager queries active contexts
 * 3. Contexts receive event in priority order
 * 4. First context to handle event stops propagation
 *
 * @example
 * ```typescript
 * const handler = new InputHandler(canvas, dimsManager);
 *
 * // Register context for modal dialog
 * const modalContext = handler.createContext('modal', priority: 100);
 * modalContext.on('keydown', (e) => {
 *   if (e.key === 'Escape') {
 *     closeModal();
 *     return true; // Handled, stop propagation
 *   }
 *   return false; // Not handled, continue to lower contexts
 * });
 *
 * // Main view receives events only if modal doesn't handle them
 * const mainContext = handler.createContext('main', priority: 0);
 * mainContext.on('keydown', handleMainViewInput);
 * ```
 */
export class InputHandler {
  // Implementation...
}
```

### Example 4: Utility Function

```typescript
/**
 * Clamp a position value to dimension bounds.
 *
 * Ensures position stays within valid range defined by dimension metadata.
 * Handles edge case where min === max (returns min).
 *
 * @param value - Position value to clamp
 * @param dimension - Dimension metadata containing min/max bounds
 * @returns Clamped value guaranteed to be in range [min, max]
 *
 * @example
 * ```typescript
 * const dim = { name: 'time', min: 0, max: 100, step: 1 };
 * const clamped = clampPosition(150, dim);
 * console.log(clamped); // 100 (clamped to max)
 * ```
 */
export function clampPosition(
  value: number,
  dimension: DimensionMetadata
): number {
  return Math.max(dimension.min, Math.min(dimension.max, value));
}
```

---

## Tag Usage Guidelines

### @param

- **Always include** for every parameter
- **Describe constraints**: valid ranges, units, null behavior
- **Explain purpose**: not just type, but WHY the parameter exists
- **Multi-line format**: Use hyphens and indent continuation

```typescript
/**
 * @param positions - Array of nD coordinates with shape (N, ndim).
 *                    Must be float32 dtype. Coordinates are in world space
 *                    units (not pixels). Can contain NaN for hidden points.
 */
```

### @returns

- **Always include** unless void
- **Describe meaning**: not just type, but what the value signifies
- **Document special values**: null, undefined, empty array, etc.

```typescript
/**
 * @returns Matching chunk indices as array, or empty array if no chunks
 *          intersect the view region. Array is sorted by chunk index.
 */
```

### @throws

- **Document all errors**: don't just document one, list all possibilities
- **Be specific**: explain exact conditions that trigger the error
- **Include error type**: use `{ErrorType}` syntax

```typescript
/**
 * @throws {ValidationError} If array is not 2D
 * @throws {ValidationError} If dtype is not float32
 * @throws {RangeError} If dimension index >= ndim
 */
```

### @example

- **Include for complex APIs**: especially public methods used frequently
- **Show complete code**: include imports, setup, and expected output
- **Multiple examples**: basic usage first, then advanced/error handling
- **Add comments**: explain what the example demonstrates

```typescript
/**
 * @example
 * ```typescript
 * // Basic: Load and display a scene
 * import { SceneLoader } from './data/scene-loader';
 *
 * const loader = new SceneLoader(store);
 * const scene = await loader.loadScene('http://example.com/data.luxar.zarr');
 * threeScene.add(scene);
 * console.log(`Loaded ${scene.children.length} nodes`);
 * ```
 *
 * @example
 * ```typescript
 * // Advanced: Handle errors and show progress
 * try {
 *   loader.on('progress', (loaded, total) => {
 *     console.log(`Loading: ${(loaded/total*100).toFixed(0)}%`);
 *   });
 *   const scene = await loader.loadScene(url);
 * } catch (error) {
 *   if (error.message.includes('404')) {
 *     console.error('Dataset not found');
 *   } else {
 *     console.error('Load failed:', error);
 *   }
 * }
 * ```
 */
```

### @see

- **Cross-reference related code**: link to functions that work together
- **Link to documentation**: reference README sections for deeper explanation
- **Use relative paths**: keep links working if files move

```typescript
/**
 * @see {@link clampPosition} for bounds clamping logic
 * @see {@link ../input/README.md#navigation} for navigation guide
 * @see {@link https://github.com/luxar/docs/navigation} for external docs
 */
```

---

## Special Cases

### Constructors

Document parameters but not return value (implied):

```typescript
/**
 * Create a new input handler.
 *
 * Sets up global event listeners for keyboard and mouse input.
 * Automatically detects typing contexts (input/textarea elements)
 * and prevents navigation in those contexts.
 *
 * @param canvas - Canvas element to attach mouse listeners to
 * @param dimsManager - Dimension manager for navigation target
 * @param config - Optional configuration overrides
 *
 * @example
 * ```typescript
 * const handler = new InputHandler(canvas, dimsManager, {
 *   enableMouse: true,
 *   enableKeyboard: true,
 * });
 * ```
 */
constructor(
  canvas: HTMLCanvasElement,
  dimsManager: SceneDimsManager,
  config?: Partial<InputConfig>
) {
  // Implementation...
}
```

### Getters/Setters

Brief documentation focusing on behavior:

```typescript
/**
 * Current slice position in nD space.
 *
 * Setting position triggers view update and data reload.
 * Position is clamped to dimension bounds automatically.
 */
get position(): number[] {
  return this._position;
}

set position(value: number[]) {
  this._position = this.clampPosition(value);
  this.updateView();
}
```

### Event Handlers

Document event type and callback signature:

```typescript
/**
 * Register callback for dimension change events.
 *
 * Emitted whenever user navigates in any dimension via keyboard,
 * slider, or programmatic call to setPosition().
 *
 * @param event - Event name (only 'dimensionChanged' supported)
 * @param callback - Handler receiving dimension index and new position
 *
 * @example
 * ```typescript
 * handler.on('dimensionChanged', (dimIndex, newPos) => {
 *   console.log(`Dimension ${dimIndex} moved to ${newPos}`);
 *   updateDataView();
 * });
 * ```
 */
on(
  event: 'dimensionChanged',
  callback: (dimIndex: number, position: number) => void
): void {
  // Implementation...
}
```

### Deprecated Functions

Mark clearly and provide alternatives:

```typescript
/**
 * Load scene from URL.
 *
 * @deprecated Since v1.4.0 - Use {@link SceneLoader.loadScene} instead.
 *             This method will be removed in v2.0.0.
 *
 * @param url - Scene URL
 * @returns Promise resolving to scene
 *
 * @example
 * ```typescript
 * // Old way (deprecated)
 * const scene = await loadSceneFromUrl(url);
 *
 * // New way (recommended)
 * const loader = new SceneLoader(store);
 * const scene = await loader.loadScene(url);
 * ```
 */
```

---

## Common Mistakes to Avoid

### ❌ Don't: Repeat Type Information

TypeScript already provides type information via signatures.

```typescript
// ❌ BAD - Just repeating types
/**
 * @param value - A number
 * @returns A number
 */
function double(value: number): number {
  return value * 2;
}

// ✅ GOOD - Explain meaning and constraints
/**
 * Double a numeric value.
 *
 * @param value - Value to double. Can be any finite number.
 * @returns Doubled value, guaranteed to have same sign as input
 */
function double(value: number): number {
  return value * 2;
}
```

### ❌ Don't: State the Obvious

```typescript
// ❌ BAD - Obvious from function name
/**
 * Gets the position.
 * @returns The position
 */
getPosition(): number[] { }

// ✅ GOOD - Explains behavior and units
/**
 * Current slice position in nD space.
 *
 * @returns Position array with length equal to dimensionality.
 *          Values are in world space units (not pixels).
 */
getPosition(): number[] { }
```

### ❌ Don't: Use Vague Descriptions

```typescript
// ❌ BAD - Too vague
/**
 * Handles input.
 * @param event - The event
 */
handleInput(event: KeyboardEvent): void { }

// ✅ GOOD - Specific about what and why
/**
 * Process keyboard input for dimension navigation.
 *
 * Detects arrow keys, number keys (dimension selection), and
 * bracket keys (forward/backward navigation). Respects input
 * context priority and typing detection.
 *
 * @param event - Keyboard event from global listener. Event may
 *                be from any element, not just canvas.
 */
handleInput(event: KeyboardEvent): void { }
```

### ❌ Don't: Write Novel-Length Documentation

Keep it concise but complete:

```typescript
// ❌ BAD - Too much detail obscures key points
/**
 * This function is used to calculate the size of chunks. Chunks are
 * pieces of data that are loaded from the zarr store. The size of
 * chunks affects performance. Larger chunks mean fewer HTTP requests
 * but more data transfer and memory usage. Smaller chunks mean more
 * HTTP requests but less memory. This function tries to balance these
 * tradeoffs by calculating an optimal chunk size based on the array
 * size and a target number of bytes per chunk. The target is typically
 * 64KB because that's a good balance between...
 * (continues for 20 more lines)
 */

// ✅ GOOD - Concise with key information
/**
 * Calculate optimal chunk size for array.
 *
 * Balances memory usage vs HTTP request count. Targets 64KB chunks
 * which provides good performance for most datasets. Prefers power-of-2
 * sizes for alignment efficiency.
 *
 * @param arraySize - Total array size in bytes
 * @param targetBytes - Target chunk size in bytes (default: 65536 = 64KB)
 * @returns Chunk size in bytes, will be <= arraySize
 */
```

---

## Documentation Checklist

Before committing code with JSDoc, verify:

- [ ] File has module-level documentation
- [ ] All exported functions/classes/interfaces have JSDoc
- [ ] All `@param` tags present with meaningful descriptions
- [ ] `@returns` documented (unless void)
- [ ] `@throws` documented for all error conditions
- [ ] At least one `@example` for complex/frequently-used APIs
- [ ] Cross-references added with `@see` where applicable
- [ ] No typos or grammatical errors
- [ ] Descriptions explain WHY and HOW, not just WHAT
- [ ] Code examples are complete and runnable

---

## Tools and Automation

### VS Code Extension

Install **Document This** extension for quick JSDoc scaffolding:
- Place cursor on function
- Press `Ctrl+Alt+D` twice
- Edit generated template

### Linting

Add to `eslint.config.js`:
```javascript
{
  "plugins": ["jsdoc"],
  "rules": {
    "jsdoc/require-jsdoc": ["error", {
      "require": {
        "FunctionDeclaration": true,
        "MethodDefinition": true,
        "ClassDeclaration": true,
        "ArrowFunctionExpression": false,
        "FunctionExpression": false
      }
    }],
    "jsdoc/require-param": "error",
    "jsdoc/require-param-description": "error",
    "jsdoc/require-returns": "error",
    "jsdoc/require-returns-description": "error"
  }
}
```

### Coverage Checking

JSDoc coverage is part of the baseline-driven `make check-docs` gate. New
under-documented files fail CI; existing file-level debt is recorded in
`scripts/docs_baseline.json`. TypeDoc's separate warning set is ratcheted by
`pnpm run typedoc:check-warnings`. See
[Documentation Quality Gate](DOCUMENTATION_QUALITY.md) for scope, limitations,
and baseline-update rules.

---

## Questions?

See existing examples in:
- `src/cache/lru-cache.ts` - Excellent JSDoc examples
- `src/types/dims.ts` - Good interface documentation
- `src/scene/scene-dims-manager.ts` - Complete method documentation

For Python docstrings, follow the conventions in `CLAUDE.md` and the per-package `README.md` files.
