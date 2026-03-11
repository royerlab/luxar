# Rendering Controls Setup Modules - Technical Specifications

Technical specifications for the modular rendering controls setup architecture.

## Architecture Overview

### Design Pattern

The rendering controls use a **modular setup pattern** where each category of controls is initialized by a dedicated setup function:

```
RenderingControls (main class)
    │
    ├─> setupNavigationControls()     → Navigation folder
    ├─> setupCameraControls()          → Camera folder
    ├─> setupHDRControls()             → HDR folder
    ├─> setupAntiAliasingControls()    → Anti-Aliasing folder
    └─> setupPostProcessingControls()  → Post-Processing Effects folder
```

Each setup function:

1. Creates a GUI folder with the custom GUI library
2. Adds controls to the folder
3. Configures onChange callbacks
4. Returns controller references

### Data Flow

```
User Interaction
    ↓
GUI Control (onChange)
    ↓
Update settings object
    ↓
Call manager method (sceneManager/postProcessing)
    ↓
Trigger render (animationController)
    ↓
Save to localStorage
```

## Type Definitions

### SetupContext

Input parameter for all setup functions. Contains all dependencies needed to create controls.

```typescript
interface SetupContext {
  // Core dependencies
  gui: GUI; // GUI instance or folder
  settings: RenderingSettings; // Mutable settings object
  postProcessing: PostProcessingManager; // Effects management
  sceneManager: SceneManager; // Scene/camera management
  animationController?: AnimationController; // Optional animation trigger

  // Callbacks
  saveSettings: () => void; // Persist to localStorage
  triggerAnimation: () => void; // Request single frame render

  // State management callbacks
  updateClippingControlsState: (enabled: boolean) => void;
  updateNavigationControls: (type: 'orbit' | 'arcball' | 'fly') => void;
}
```

**Design decisions:**

- **Callbacks as arrow functions**: Preserve `this` context from main class
- **Optional animationController**: Some effects need continuous rendering (detector noise, auto-rotate)
- **Mutable settings**: Direct mutation for real-time updates, saved on callback

### SetupResult

Output from all setup functions. Contains references needed by the main class.

```typescript
interface SetupResult {
  // Required
  controllers: Partial<RenderingControllers>; // UI controller references

  // Optional
  folders?: {
    // Folder references for visibility control
    orbitFolder?: GUI;
    flyFolder?: GUI;
    [key: string]: GUI | undefined;
  };

  shadowObjects?: {
    // Special UI objects
    exposureLogValue?: { log: number };
    [key: string]: any;
  };
}
```

**Design decisions:**

- **Partial controllers**: Each module returns only its own controllers
- **Optional folders**: Only navigation needs folder visibility control
- **Shadow objects**: Enable advanced UI patterns (logarithmic sliders)

## Module Specifications

### types.ts

**Purpose**: Shared type definitions for all setup modules

**Exports**:

- `SetupContext` - Input parameter interface
- `SetupResult` - Output return interface

**Dependencies**: None (pure type definitions)

**Lines**: 88

---

### navigation-setup.ts

**Purpose**: Create navigation controls for camera movement and rotation

**Signature**:

```typescript
function setupNavigationControls(context: SetupContext): SetupResult;
```

**Creates**:

- Navigation folder (open by default)
  - Control type dropdown (orbit, arcball, fly)
  - Orbit Controls subfolder
    - Auto Rotate checkbox
    - Rotation Speed slider (0.1-5)
  - Fly Controls subfolder
    - Movement Speed slider
    - Rotation Speed slider
    - Inertial Mode checkbox
    - Translation Damping slider (conditional)
    - Rotation Damping slider (conditional)

**Returns**:

```typescript
{
  controllers: {
    controlType,
    autoRotate,
    autoRotateSpeed,
    flyMovementSpeed,
    flyRotationSpeed,
    flyInertialMode,
    flyDamping,
    flyRotationDamping
  },
  folders: {
    orbitFolder,
    flyFolder
  }
}
```

**Special Behavior**:

- Damping controls hidden when inertial mode disabled
- Calls `updateNavigationControls()` to set initial folder visibility
- Auto-rotate triggers continuous animation via `animationController.startAnimation()`

**Lines**: 242

---

### camera-setup.ts

**Purpose**: Create camera-specific settings controls

**Signature**:

```typescript
function setupCameraControls(
  context: SetupContext,
  controllersRef: SetupResult['controllers']
): SetupResult;
```

**Parameters**:

- `context`: Standard setup context
- `controllersRef`: Reference to controllers object for lens distortion sync

**Creates**:

- Camera folder (open by default)
  - FOV Preset dropdown
  - Field of View slider (config.camera.fovMin - fovMax)
  - Clipping Planes subfolder (closed by default)
    - Near Plane slider (0.001-10.0)
    - Far Plane slider (10-10000)
    - Dynamic Clipping checkbox
    - Adapt Speed slider (conditional)

**Returns**:

```typescript
{
  controllers: {
    (fovPreset, fov, nearPlane, farPlane, dynamicClippingEnabled, clippingAdaptSpeed);
  }
}
```

**Special Behavior**:

- **FOV Preset Sync**: When preset changes, updates lens distortion via `controllersRef`
- **Bidirectional Binding**: FOV slider changes switch preset to "Custom"
- **Conditional Controls**: Adapt speed only visible when dynamic clipping enabled
- **Validation**: Near < far plane enforced

**Cross-Module Dependencies**:

- Reads `controllersRef.lensDistortion*` to sync FOV presets with lens parameters
- Updates `controllersRef.fov` when preset changes

**Lines**: 287

---

### hdr-setup.ts

**Purpose**: Create global EOG (Exposure-Offset-Gamma) and tone mapping controls

**Signature**:

```typescript
function setupHDRControls(context: SetupContext, exposureLogValue: { log: number }): SetupResult;
```

**Parameters**:

- `context`: Standard setup context
- `exposureLogValue`: Shadow object for logarithmic exposure slider

**Creates**:

- HDR folder (open by default)
  - Exposure slider (logarithmic, range: 0.01-100)
  - Global Offset slider (linear, range: -1.0 to 1.0)
  - Global Gamma slider (linear, range: 0.1 to 10.0)
  - Tone Mapping dropdown (None, Linear, Reinhard, Cineon, ACES, AgX, Neutral)

**Returns**:

```typescript
{
  controllers: {
    exposure
  },
  shadowObjects: {
    exposureLogValue
  }
}
```

**Special Behavior**:

- **Logarithmic Exposure Slider**: Uses shadow object pattern for perceptual linearity
  - Slider controls `exposureLogValue.log` (linear log10 values)
  - onChange converts: `actualValue = 10^logValue`
  - Custom `updateDisplay()` shows actual value, not log value
- **Global EOG Model**: `adjusted = color * exposure + globalOffset; clip; pow(adjusted, 1/globalGamma)` applied in LuxarToneMappingEffect before tone mapping
- **Format Helper**: Shows `0.01` (3 decimals) to `10` (0 decimals) to `100` (0 decimals)

**Algorithm**:

```typescript
// Initialize
exposureLogValue.log = Math.log10(settings.exposure);

// On change
const actualValue = Math.pow(10, logValue);
settings.exposure = actualValue;

// Display override
const formatExposure = (logValue: number): string => {
  const actual = Math.pow(10, logValue);
  if (actual >= 10) return actual.toFixed(0);
  if (actual >= 1) return actual.toFixed(1);
  if (actual >= 0.1) return actual.toFixed(2);
  return actual.toFixed(3);
};
```

**Lines**: 140

---

### anti-aliasing-setup.ts

**Purpose**: Create anti-aliasing technique controls

**Signature**:

```typescript
function setupAntiAliasingControls(context: SetupContext): SetupResult;
```

**Creates**:

- Anti-Aliasing folder (closed by default)
  - SSAA Enabled checkbox
  - SSAA Settings subfolder (conditional)
    - Resolution Multiplier dropdown (1.5, 2.0, 3.0, 4.0)
  - FXAA Enabled checkbox
  - MSAA Enabled checkbox ⚠️
  - MSAA Settings subfolder (conditional)
    - Sample Count dropdown (2, 4, 8)
  - SMAA Enabled checkbox

**Returns**:

```typescript
{
  controllers: {
  } // No controller references exported (toggles only)
}
```

**Special Behavior**:

- **Conditional Subfolders**: SSAA and MSAA settings only visible when enabled
- **MSAA Warning**: Tooltip warns about brightness issues with additive blending
- **SMAA Preset-Only**: Fine-grained controls (threshold, search steps) not exposed because pmndrs/postprocessing only supports preset modes

**Initial State**:

- Subfolders hidden if corresponding AA method disabled
- Called after setup to show/hide based on current settings

**Lines**: 156

---

### post-processing-setup.ts

**Purpose**: Create all post-processing effect controls

**Signature**:

```typescript
function setupPostProcessingControls(
  context: SetupContext,
  controllersRef: SetupResult['controllers']
): SetupResult;
```

**Parameters**:

- `context`: Standard setup context
- `controllersRef`: Reference to controllers object for FOV preset sync

**Creates**:

- Post-Processing Effects folder (closed by default)
  - **Bloom** subfolder (closed)
    - Threshold (0-1), Strength (0-2), Radius (0-1), Mipmap Levels (1-12)
  - **Detector Noise** subfolder (closed)
    - Enabled, Readout Sigma, Photon Gain, FPN Sigma
  - **Depth of Field** subfolder (closed)
    - Enabled, Focus distance, Strength
  - **Chromatic Aberration** subfolder (closed)
    - Enabled, Strength
  - **Ambient Occlusion** subfolder (closed)
    - Enabled, Quality (low, medium, high, ultra)
  - **Vignette** subfolder (closed)
    - Enabled, Darkness, Offset
  - **Lens Distortion** subfolder (closed)
    - Enabled, Distortion X/Y, Principal Point X/Y, Focal Length X/Y, Skew

**Returns**:

```typescript
{
  controllers: {
    // Lens distortion controllers for FOV preset sync
    (lensDistortionX,
      lensDistortionY,
      lensPrincipalPointX,
      lensPrincipalPointY,
      lensFocalLengthX,
      lensFocalLengthY,
      lensSkew);
  }
}
```

**Special Behavior**:

- **Detector Noise Animation**: When enabled, triggers continuous rendering via `animationController.startAnimation()`
- **Lens Distortion Export**: Controller references exported for FOV preset synchronization in camera module
- **All Subfolders Closed**: User must explicitly open effect categories

**Effect Count**: 7 distinct effects, each with 2-8 parameters

**Lines**: 598

## Control Type Reference

### GUI Controller Types

All controls use the custom GUI's controller system:

| Method                              | Type     | Use Case               |
| ----------------------------------- | -------- | ---------------------- |
| `.add(obj, 'prop')`                 | Checkbox | Boolean toggles        |
| `.add(obj, 'prop', min, max, step)` | Slider   | Numeric ranges         |
| `.add(obj, 'prop', [options])`      | Dropdown | Predefined choices     |
| `.addFolder(name)`                  | Folder   | Group related controls |

### Common Patterns

**Conditional Visibility**:

```typescript
if (value) {
  subfolder.show();
  subfolder.open();
} else {
  subfolder.close();
  subfolder.hide();
}
```

**Tooltips**:

```typescript
control.domElement.setAttribute('title', 'Help text\n• Bullet 1\n• Bullet 2');
```

**Controller Storage**:

```typescript
controllers.propertyName = control;
return { controllers };
```

## State Management

### Settings Persistence

All controls update `settings` object and call `saveSettings()`:

```typescript
.onChange((value: number) => {
  settings.property = value;
  manager.updateProperty(value);
  saveSettings();         // Persist to localStorage
  triggerAnimation();     // Request render
})
```

**Persistence Key**: `renderingSettings_${sceneId}`

**Storage Format**: JSON serialization of `RenderingSettings` interface

### Programmatic Updates

Controllers stored in `RenderingControllers` map enable programmatic updates:

```typescript
if (this.controllers.bloomStrength) {
  this.controllers.bloomStrength.setValue(newValue);
  this.controllers.bloomStrength.updateDisplay();
}
```

**Update Pattern**:

1. `setValue()` - Update internal value
2. `updateDisplay()` - Refresh UI display
3. Does NOT trigger onChange callback

## Performance Considerations

### Render Triggering

- **Most controls**: `triggerAnimation()` - Single frame render
- **Continuous effects**: `animationController.startAnimation()` - Continuous rendering
  - Auto-rotate (navigation)
  - Detector noise (post-processing)

### Deferred Initialization

Controls created synchronously during `setupControls()` call:

- No async/await needed
- Instant UI response
- Settings loaded from localStorage before setup

### Memory Management

- Controllers stored as weak references (garbage collected when GUI destroyed)
- No manual cleanup needed
- Main class `dispose()` destroys GUI and all controllers

## Testing Strategy

### Unit Testing Approach

1. **Mock SetupContext**:

```typescript
const mockContext: SetupContext = {
  gui: new GUI(),
  settings: createMockSettings(),
  postProcessing: createMockPostProcessing(),
  sceneManager: createMockSceneManager(),
  saveSettings: vi.fn(),
  triggerAnimation: vi.fn(),
  updateClippingControlsState: vi.fn(),
  updateNavigationControls: vi.fn(),
};
```

2. **Call Setup Function**:

```typescript
const result = setupNavigationControls(mockContext);
```

3. **Verify Results**:

```typescript
expect(result.controllers.controlType).toBeDefined();
expect(result.folders?.orbitFolder).toBeDefined();
```

4. **Test Interactions**:

```typescript
// Simulate user interaction
const control = result.controllers.autoRotate;
control?.setValue(true);

// Verify callbacks
expect(mockContext.saveSettings).toHaveBeenCalled();
```

### Integration Testing

Full integration tests in `rendering-controls.test.ts` verify:

- All setup functions called in correct order
- Controller references properly stored
- Cross-module dependencies work correctly
- Settings persistence functions

## Migration Notes

### Refactoring from v1 (Monolithic)

**Before** (rendering-controls.ts - 2,514 lines):

```typescript
private setupControls(): void {
  // 1,290 lines of inline GUI creation
  const navigationFolder = this.gui.addFolder('Navigation');
  // ... 220 lines of navigation controls ...
  const cameraFolder = this.gui.addFolder('Camera');
  // ... 252 lines of camera controls ...
  // ... etc for all categories ...
}
```

**After** (v2 - Modular - 1,350 lines main + 6 modules):

```typescript
private setupControls(): void {
  this.setupAutoBlur();

  const navResult = setupNavigationControls(context);
  Object.assign(this.controllers, navResult.controllers);

  const camResult = setupCameraControls(context, this.controllers);
  Object.assign(this.controllers, camResult.controllers);

  // ... etc for all categories ...
}
```

**Benefits Achieved**:

- 46% reduction in main file size
- Under Read tool token limit
- Better code organization
- Improved testability
- No behavior changes

### API Compatibility

Public API unchanged:

- `new RenderingControls(postProcessing, sceneManager)` - Same constructor
- `show()`, `hide()`, `toggle()` - Same methods
- `syncCurrentState()` - Same synchronization
- Settings format unchanged - localStorage compatible

Internal refactoring only - **no breaking changes**.

## Future Enhancements

### Potential Improvements

1. **Lazy Loading**: Load effect modules only when folders opened
2. **Preset System**: Save/load control presets
3. **Keyboard Shortcuts**: Hotkeys for common adjustments
4. **Real-time Validation**: Prevent invalid value combinations
5. **Undo/Redo**: Track setting history

### Extension Points

To add a new control category:

1. Create `{category}-setup.ts` in this folder
2. Export `setup{Category}Controls(context: SetupContext): SetupResult`
3. Call from `RenderingControls.setupControls()`
4. Update this documentation

**Example skeleton**:

```typescript
export function setupNewCategoryControls(context: SetupContext): SetupResult {
  const { gui, settings, saveSettings, triggerAnimation } = context;
  const controllers: SetupResult['controllers'] = {};

  const folder = gui.addFolder('New Category');
  folder.close();

  // Add controls...

  return { controllers };
}
```

## Related Specifications

- **Parent**: `../SPECIFICATIONS.md` - Full UI specifications
- **Main Class**: `../rendering-controls.ts` - RenderingControls implementation
- **Config**: `../../config/index.ts` - Configuration constants
- **Types**: `../../controls/types.ts` - RenderingControllers interface
