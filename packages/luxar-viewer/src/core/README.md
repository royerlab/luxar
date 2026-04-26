# Core Package

Application initialization and lifecycle management for the Luxar player. This package contains the main application class and entry point that orchestrates all other components into a cohesive visualization system.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Application Lifecycle](#application-lifecycle)
- [Component Integration](#component-integration)
- [Dataset Loading](#dataset-loading)
- [Error Handling](#error-handling)
- [Debug Features](#debug-features)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Overview

The core package serves as the central orchestrator for the Luxar application, managing component initialization, lifecycle events, and the integration of all subsystems. It provides a clean, high-level API while handling the complex coordination of scene management, input handling, rendering controls, and dataset loading.

**Core Philosophy**: Provide a simple, reliable application entry point that handles complex initialization sequences, error recovery, and resource management while exposing a clean API for advanced usage.

## Key Features

- **Component Orchestration**: Manages initialization order and dependencies between subsystems
- **Lifecycle Management**: Handles application startup, dataset loading, and cleanup
- **Error Recovery**: Comprehensive error handling with user-friendly messaging
- **Dataset Detection**: Intelligent detection of Zarr datasets vs directory navigation
- **Debug Interface**: Development tools and debugging capabilities
- **Resource Management**: Proper cleanup of WebGL resources and event listeners
- **URL Parameter Handling**: Automatic dataset loading from URL parameters

## Architecture

```typescript
core/
├── main.ts    # Application entry point and global setup
└── app.ts     # Main application class and lifecycle management
```

**Component Dependency Flow**:

```
main.ts → LuxarApp → SceneManager → AnimationController
                  → InputHandler ← RenderingControls
                  → DatasetBrowser (conditional)
```

The architecture follows a hierarchical initialization pattern where each component is responsible for its own setup while the core coordinates the overall sequence.

## Application Lifecycle

### Initialization Sequence

```typescript
class LuxarApp {
  async init(options: LuxarAppOptions): Promise<void> {
    // 1. Scene Management Foundation
    this.sceneManager = new SceneManager();
    await this.sceneManager.init({ canvas: options.canvas, debug: options.debug });

    // 2. Animation System Setup
    this.animationController = new AnimationController(/*...*/);

    // 3. Input System Integration
    this.inputHandler = new InputHandler(this.sceneManager, this.animationController);
    this.inputHandler.init();

    // 4. UI Controls Configuration
    this.renderingControls = new RenderingControls(/*...*/);

    // 5. Component Cross-Linking
    this.renderingControls.setAnimationController(this.animationController);
    this.inputHandler.setRenderingControls(this.renderingControls);

    // 6. Animation Loop Start
    this.animationController.startAnimation();

    // 7. Dataset Loading or Browser Display
    const src = options.src ?? config.defaultZarrPath;
    if (await this.shouldShowBrowser(src)) {
      this.showDatasetBrowser();
    } else {
      await this.loadDataset(src);
    }

    // 8. System Event Handling
    this.setupDisposeOnUnload();
    this.setupDatasetBrowserShortcut();
    this.setupFocusHandling();
  }
}
```

**Critical Design Decisions**:

- **Animation First**: Start rendering loop before loading data for immediate visual feedback
- **Error Isolation**: Component failures don't prevent other systems from initializing
- **Progressive Enhancement**: Core 3D functionality works even if data loading fails

### Disposal and Resource Management

```typescript
dispose(): void {
  // 1. Stop animation loop (prevents further rendering)
  this.animationController?.dispose();

  // 2. Remove event listeners (prevents memory leaks)
  this.inputHandler?.dispose();

  // 3. Clean up UI components (remove DOM elements)
  this.renderingControls?.dispose();

  // 4. Dispose WebGL resources (textures, buffers, shaders)
  this.sceneManager?.dispose();

  // 5. Remove global event listeners
  if (this.boundDispose) {
    window.removeEventListener('beforeunload', this.boundDispose);
    this.boundDispose = null;
  }
}
```

## Component Integration

### Scene Manager Integration

The app coordinates scene management with other systems:

```typescript
// Initialize scene foundation (canvas comes in via LuxarAppOptions)
this.sceneManager = new SceneManager();
await this.sceneManager.init({ canvas: options.canvas });

// Pass scene components to animation controller
this.animationController = new AnimationController(
  this.sceneManager.controls, // Camera controls manager
  this.sceneManager.postProcessing // HDR post-processing
);
```

### Input System Coordination

The app establishes bidirectional communication between input and rendering systems:

```typescript
// Input handler needs scene access for dimension navigation
this.inputHandler = new InputHandler(this.sceneManager, this.animationController);

// Rendering controls need input system for keyboard shortcuts
this.inputHandler.setRenderingControls(this.renderingControls);

// Rendering controls need animation system for triggering updates
this.renderingControls.setAnimationController(this.animationController);
```

### Dataset Loading Pipeline

```typescript
private async loadDataset(src: string): Promise<void> {
  // 1. Clear existing UI state
  this.inputHandler.clearDimensionUI();

  // 2. Set scene ID for rendering controls persistence BEFORE loading
  this.renderingControls.setSceneId(src);

  // 3. Load and parse Zarr data (async)
  await this.sceneManager.loadSceneData(src);

  // 4. Apply zarr viewer_config defaults if no stored settings exist
  const viewerConfig = this.sceneManager.getSceneViewerConfig();
  this.renderingControls.setZarrViewerConfig(viewerConfig);

  // 5. Initialize nD navigation UI and scale bar
  this.inputHandler.initDimensionSliders();
  this.initScaleBar();

  // 6. Initialize layers panel and colormap legend
  // ...

  // 7. Apply viewer config state (UI visibility, theme, dimensions)
  this.applyViewerConfigState(viewerConfig);

  // 8. Trigger immediate render
  this.animationController.startAnimation();
}
```

## Dataset Loading

### Intelligent Dataset Detection

The app automatically determines whether to show a dataset browser or load data directly:

```typescript
private async shouldShowBrowser(src: string): Promise<boolean> {
  // No source or empty string: show browser
  if (!src || src.trim() === '') {
    return true;
  }

  // Directory URLs (ending with /) show browser
  if (src.endsWith('/')) {
    return true;
  }

  // Check for Zarr dataset markers (v2 .zgroup/.zattrs or v3 zarr.json)
  // Short-circuits as soon as any probe confirms zarr metadata exists
  const zarrChecks = [
    fetch(src + '/.zgroup', { method: 'HEAD' }),
    fetch(src + '/.zattrs', { method: 'HEAD' }),
    fetch(src + '/zarr.json', { method: 'HEAD' }),
  ];

  try {
    await Promise.any(
      zarrChecks.map((p) =>
        p.then((r) => {
          if (!r.ok) throw new Error('not ok');
          return r;
        })
      )
    );
    return false; // At least one zarr metadata file exists
  } catch {
    // All probes failed — likely a directory, show browser
  }

  return true;
}
```

### Dataset Browser Integration

When directory navigation is needed, the app provides an integrated browser:

```typescript
private showDatasetBrowser(): void {
  this.datasetBrowser = new DatasetBrowser({
    container: document.body,
    onDatasetSelect: async (path: string) => {
      // Update URL parameters
      const params = new URLSearchParams(window.location.search);
      params.set('src', constructFullURL(path));
      window.history.replaceState({}, '', `${window.location.pathname}?${params}`);

      // Load selected dataset
      await this.loadDataset(constructFullURL(path));
    },
    onClose: () => {
      this.datasetBrowser = undefined;
    }
  });
}
```

## Error Handling

### Error Recovery

The `init()` method uses a single top-level try/catch. All initialization steps (scene manager, animation controller, input handler, rendering controls, and dataset loading) run inside this block. If any step fails, the error propagates to the caller:

```typescript
async init(options: LuxarAppOptions): Promise<void> {
  try {
    // All initialization in sequence inside one try block:
    // 1. Scene manager + animation controller
    // 2. Input handler + rendering controls
    // 3. Start animation loop
    // 4. Dataset loading or browser display
    // 5. Dispose/focus handlers
  } catch (error) {
    // Single catch handles all initialization failures
    throw error;
  }
}
```

The caller (in `main.ts`) catches and displays errors:

```typescript
app.init({ canvas, src }).catch((error) => {
  console.error('Failed to start Luxar application:', error);
  showError('Failed to start the application. Please check the console for details.');
});
```

### User-Friendly Error Display

```typescript
// Global error handler for unhandled initialization failures
app.init({ canvas, src }).catch((error) => {
  console.error('Failed to start Luxar application:', error);
  showError('Failed to start the application. Please check the console for details.');
});
```

## Debug Features

### Development Interface

The app exposes a debug interface when in development mode:

```typescript
// main.ts debug setup
const isDebugMode = params.has('debug') || localStorage.getItem('luxar_debug') === 'true';

if (isDebugMode) {
  window.__luxarDebug = {
    app, // Access to main app instance
    consoleInterceptor, // Console message buffer
    version: '1.0.0', // Application version
  };
  console.log('🔧 [Luxar] Debug interface available at window.__luxarDebug');
}
```

### Console Interception Setup

```typescript
// Critical: Import console interceptor FIRST
import { consoleInterceptor } from '../utils/console-interceptor';
console.log('🚀 [Luxar] Application starting...');

// This ensures ALL console output is captured from app start
```

### Component Access for Testing

```typescript
get components() {
  return {
    sceneManager: this.sceneManager,
    animationController: this.animationController,
    inputHandler: this.inputHandler,
    renderingControls: this.renderingControls,
    adaptiveDPRManager: this.adaptiveDPRManager,
  };
}
```

## Usage Examples

### Basic Application Initialization

```typescript
import { LuxarApp } from './core/app';

// Create and initialize app
const app = new LuxarApp();

// Start with specific dataset
await app.init('/data/my-dataset.zarr');

// Start with directory browser
await app.init('/data/');

// Start with default dataset
await app.init();
```

### URL Parameter Integration

```typescript
// Automatic URL parameter parsing
const params = new URLSearchParams(window.location.search);
const datasetURL = params.get('src') ?? '/data/default.zarr';

// App automatically handles the URL
const app = new LuxarApp();
await app.init(datasetURL);
```

### Error Handling and Recovery

```typescript
const app = new LuxarApp();

try {
  await app.init(datasetURL);
  console.log('Application started successfully');
} catch (error) {
  console.error('Initialization failed:', error);

  // App may still be partially functional
  if (app.initialized) {
    console.log('App partially initialized - some features may work');
  }

  // Manual cleanup if needed
  app.dispose();
}
```

### Component Access and Testing

```typescript
const app = new LuxarApp();
await app.init();

// Access components for testing or advanced usage
const { sceneManager, animationController } = app.components;

// Direct scene manipulation
sceneManager.scene.add(customObject);

// Animation control
animationController.startAnimation();
animationController.stopAnimation();
```

### Focus and Visibility Handling

The app automatically handles window focus and visibility changes for power efficiency:

```typescript
// Setup in app initialization
private setupFocusHandling(): void {
  // Refresh rendering when window gains focus
  window.addEventListener('focus', () => {
    this.animationController.startAnimation();
    log.info(Modules.LUXAR, 'Window focused - triggering render refresh');
  });

  // Handle tab switching - STOP animation when hidden to save resources
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      // Tab hidden - stop completely to guarantee zero CPU/GPU usage
      this.animationController.stopAnimation();
      log.info(Modules.LUXAR, 'Document hidden - stopping animation to save resources');
    } else {
      // Tab visible - resume rendering
      this.animationController.startAnimation();
      log.info(Modules.LUXAR, 'Document became visible - resuming animation');
    }
  });
}
```

**Power Saving Behavior**:

| Event          | Action             | Console Message                                        |
| -------------- | ------------------ | ------------------------------------------------------ |
| Tab hidden     | `stopAnimation()`  | `[ℹ️] [Luxar] Document hidden - stopping animation...` |
| Tab visible    | `startAnimation()` | `[ℹ️] [Luxar] Document became visible - resuming...`   |
| Window focused | `startAnimation()` | `[ℹ️] [Luxar] Window focused - triggering render...`   |

This ensures zero CPU/GPU usage when the tab is not visible, even if continuous effects (detector noise, auto-rotate) are enabled.

## Best Practices

### Initialization Order

```typescript
// ✅ Good: Respect dependency order
// 1. Foundation (SceneManager)
// 2. Animation System
// 3. Input Handling
// 4. UI Controls
// 5. Cross-linking
// 6. Data Loading

// ❌ Avoid: Initializing components before their dependencies
// Don't create InputHandler before SceneManager
```

### Error Handling Strategy

```typescript
// ✅ Good: Single try/catch in init(), let errors propagate to caller
async init(options: LuxarAppOptions): Promise<void> {
  try {
    // All steps in sequence; any failure propagates
    await this.sceneManager.init({ canvas: options.canvas });
    // ... other initialization ...
    await this.loadDataset(options.src ?? config.defaultZarrPath);
  } catch (error) {
    throw error;
  }
}

// ✅ Good: Caller displays errors to user
app.init({ canvas, src }).catch((error) => {
  showError('Failed to start the application.');
});

// ✅ Good: Provide fallbacks for optional values
const src = params.get('src') ?? config.defaultZarrPath;
```

### Resource Management

```typescript
// ✅ Good: Comprehensive disposal
dispose(): void {
  // Stop animation first (prevents new work)
  this.animationController?.dispose();

  // Clean up in reverse initialization order
  this.inputHandler?.dispose();
  this.renderingControls?.dispose();
  this.sceneManager?.dispose();

  // Remove global listeners
  if (this.boundDispose) {
    window.removeEventListener('beforeunload', this.boundDispose);
    this.boundDispose = null;
  }
}

// ✅ Good: Store bound reference for proper cleanup
private boundDispose: (() => void) | null = null;

private setupDisposeOnUnload(): void {
  this.boundDispose = this.dispose.bind(this);
  window.addEventListener('beforeunload', this.boundDispose);
}
```

### Component Communication

```typescript
// ✅ Good: Explicit dependency injection
this.inputHandler = new InputHandler(this.sceneManager, this.animationController);

// ✅ Good: Post-initialization linking
this.renderingControls.setAnimationController(this.animationController);
this.inputHandler.setRenderingControls(this.renderingControls);

// ❌ Avoid: Hidden global dependencies or tight coupling
// Don't access components through global variables
```

### URL and State Management

```typescript
// ✅ Good: Automatic URL synchronization
onDatasetSelect: async (path: string) => {
  // Update browser URL to reflect current dataset
  const params = new URLSearchParams(window.location.search);
  params.set('src', fullURL);
  window.history.replaceState({}, '', `${window.location.pathname}?${params}`);

  // Load dataset
  await this.loadDataset(fullURL);
};

// ✅ Good: URL parameter parsing
const params = new URLSearchParams(window.location.search);
const src = params.get('src') ?? config.defaultZarrPath;
```

The core package provides the essential coordination and lifecycle management that transforms individual Luxar components into a cohesive, reliable visualization application.
