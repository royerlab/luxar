# Rendering Controls Setup Modules

Modular setup functions for the rendering controls UI. Each module is responsible for creating a specific category of controls in the GUI interface.

## Overview

The rendering controls UI was refactored from a monolithic 2,514-line file into focused, maintainable modules. This improves code organization, testability, and makes the codebase easier to navigate.

### Architecture

```
rendering-controls/
├── types.ts                    # Shared types (SetupContext, SetupResult)
├── navigation-setup.ts         # Navigation controls (orbit, arcball, fly)
├── camera-setup.ts             # Camera settings (FOV, clipping)
├── hdr-setup.ts                # HDR intensity & tone mapping
├── anti-aliasing-setup.ts      # AA techniques (FXAA, SMAA, MSAA, SSAA)
└── post-processing-setup.ts    # Effects (bloom, noise, DoF, etc.)
```

Each module exports a setup function that:

1. Takes a `SetupContext` with GUI, settings, and callbacks
2. Creates its category's controls using the GUI library
3. Returns a `SetupResult` with controller references

## Module Responsibilities

### types.ts

Defines shared interfaces used across all setup modules:

```typescript
interface SetupContext {
  gui: GUI; // GUI instance
  settings: RenderingSettings; // Current settings
  postProcessing: PostProcessingManager; // Effect manager
  sceneManager: SceneManager; // Scene/camera manager
  animationController?: AnimationController; // Animation trigger
  saveSettings: () => void; // Persist settings
  triggerAnimation: () => void; // Request render
  updateClippingControlsState: (enabled: boolean) => void;
  updateNavigationControls: (type: 'orbit' | 'arcball' | 'fly') => void;
}

interface SetupResult {
  controllers: Partial<RenderingControllers>; // UI controller refs
  folders?: { [key: string]: GUI }; // Folder refs
  shadowObjects?: { [key: string]: any }; // Special UI objects
}
```

### navigation-setup.ts

**Navigation controls for camera movement and rotation**

Creates controls for:

- Control type selector (orbit, arcball, fly)
- Orbit controls (auto-rotate, rotation speed)
- Fly controls (movement speed, rotation speed, inertial mode, damping)

Returns folder references for `orbitFolder` and `flyFolder` which are used by `updateNavigationControls()` to show/hide controls based on the active control type.

### camera-setup.ts

**Camera-specific settings**

Creates controls for:

- FOV presets (28mm, 35mm, 50mm, 85mm, 135mm, Custom)
- FOV slider with real-time adjustment
- Clipping planes (near, far, dynamic clipping with adapt speed)

**Special behavior:** Takes `controllersRef` parameter to enable FOV preset synchronization with lens distortion controls (defined in post-processing module).

### hdr-setup.ts

**HDR intensity and tone mapping**

Creates controls for:

- HDR intensity (logarithmic slider for perceptual linearity)
- Tone mapping selector (None, Linear, Reinhard, Cineon, ACES, AgX, Neutral)

**Special behavior:** Takes `hdrLogValue` shadow object for the logarithmic slider implementation.

### anti-aliasing-setup.ts

**Anti-aliasing techniques**

Creates controls for:

- SSAA (Supersampling) with resolution multiplier
- FXAA (Fast Approximate AA)
- MSAA (Multisample AA) with sample count ⚠️
- SMAA (Subpixel Morphological AA)

Includes dynamic subfolder showing/hiding based on AA enablement.

### post-processing-setup.ts

**Post-processing visual effects**

Creates controls for 7 effects:

- **Bloom**: Glow/light bleeding (threshold, strength, radius, mipmap levels)
- **Detector Noise**: Physics-based noise (shot, readout, FPN)
- **Depth of Field**: Bokeh blur (focus, strength)
- **Chromatic Aberration**: Lens color fringing
- **Ambient Occlusion**: Contact shadows (quality presets)
- **Vignette**: Edge darkening (darkness, offset)
- **Lens Distortion**: Full camera model (distortion, principal point, focal length, skew)

**Special behavior:** Takes `controllersRef` parameter and exports lens distortion controller references for FOV preset synchronization.

## Usage

The main `RenderingControls` class calls these setup functions during initialization:

```typescript
private setupControls(): void {
  this.setupAutoBlur();

  // Navigation
  const navResult = setupNavigationControls({
    gui: this.gui,
    settings: this.settings,
    postProcessing: this.postProcessing,
    sceneManager: this.sceneManager,
    animationController: this.animationController,
    saveSettings: () => this.saveSettings(),
    triggerAnimation: () => this.triggerAnimation(),
    updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
    updateNavigationControls: (controlType) => this.updateNavigationControls(controlType),
  });

  Object.assign(this.controllers, navResult.controllers);
  this.orbitFolder = navResult.folders?.orbitFolder;
  this.flyFolder = navResult.folders?.flyFolder;

  // Camera (needs controller reference for lens distortion sync)
  const camResult = setupCameraControls(context, this.controllers);
  Object.assign(this.controllers, camResult.controllers);

  // HDR (needs shadow object for logarithmic slider)
  const hdrResult = setupHDRControls(context, this.hdrLogValue);
  Object.assign(this.controllers, hdrResult.controllers);

  // Anti-aliasing
  const aaResult = setupAntiAliasingControls(context);
  Object.assign(this.controllers, aaResult.controllers);

  // Post-processing (needs controller reference for lens distortion)
  const ppResult = setupPostProcessingControls(context, this.controllers);
  Object.assign(this.controllers, ppResult.controllers);
}
```

## Cross-Module Dependencies

### FOV Preset ↔ Lens Distortion Sync

When a FOV preset is selected, the camera module updates the corresponding lens distortion parameters:

- Camera module receives `controllersRef` to access lens distortion controllers
- Post-processing module exports lens distortion controller references
- FOV preset onChange updates both FOV and lens distortion settings

### HDR Logarithmic Slider

The HDR intensity slider uses a shadow object pattern:

- Slider controls `hdrLogValue.log` (logarithmic value)
- onChange converts to actual value: `10^log`
- Custom `updateDisplay()` shows actual value, not log value

### Navigation Folder Visibility

The navigation module returns folder references:

- `orbitFolder` and `flyFolder` stored by main class
- `updateNavigationControls()` shows/hides folders based on control type
- Shared between orbit and arcball modes

## Benefits of Modular Design

1. **Improved Maintainability**: Each module has a single, clear responsibility
2. **Better Testability**: Smaller functions easier to test in isolation
3. **Enhanced Readability**: Clear separation of concerns, easier navigation
4. **Reduced Complexity**: Main file reduced from 2,514 to 1,350 lines (46% reduction)
5. **Tool Compatibility**: Files now within Read tool token limits
6. **Scalability**: Easy to add new control categories without bloating main file

## Best Practices

### When Adding New Controls

1. **Determine the category**: Does it fit in an existing module or need a new one?
2. **Follow the pattern**: Export a setup function that takes `SetupContext`
3. **Return controller refs**: Enable programmatic updates via `SetupResult`
4. **Add tooltips**: Use `domElement.setAttribute('title', ...)` for help text
5. **Handle dependencies**: Pass additional parameters if cross-module sync needed
6. **Document behavior**: Update this README with special behaviors

### Cross-Module Communication

- **Prefer explicit parameters** over global state
- **Pass controller references** when one module needs to update another's controls
- **Use shadow objects** for complex UI patterns (like logarithmic sliders)
- **Return folder references** when visibility toggling is needed

## Testing

Each module can be tested independently by:

1. Creating a mock `SetupContext` with required dependencies
2. Calling the setup function
3. Verifying the returned `SetupResult` contains expected controllers
4. Testing UI interactions trigger the correct callbacks

See the main rendering-controls tests for examples of testing the integrated system.

## Related Documentation

- Parent: `../README.md` - UI package overview
- Specifications: `./SPECIFICATIONS.md` - Technical specifications for setup modules
- Main class: `../rendering-controls.ts` - RenderingControls class that uses these modules
