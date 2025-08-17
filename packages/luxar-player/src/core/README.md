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
  async init(src?: string): Promise<void> {
    // 1. Scene Management Foundation
    this.sceneManager = new SceneManager();
    await this.sceneManager.init();

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
    if (await this.shouldShowBrowser(src)) {
      this.showDatasetBrowser();
    } else {
      await this.loadDataset(src);
    }

    // 8. System Event Handling
    this.setupCleanup();
    this.setupDatasetBrowserShortcut();
    this.setupFocusHandling();
  }
}
```

**Critical Design Decisions**:
- **Animation First**: Start rendering loop before loading data for immediate visual feedback
- **Error Isolation**: Component failures don't prevent other systems from initializing
- **Progressive Enhancement**: Core 3D functionality works even if data loading fails

### Cleanup and Resource Management

```typescript
cleanup(): void {
  // 1. Stop animation loop (prevents further rendering)
  this.animationController?.dispose();

  // 2. Remove event listeners (prevents memory leaks)
  this.inputHandler?.dispose();

  // 3. Clean up UI components (remove DOM elements)
  this.renderingControls?.dispose();

  // 4. Dispose WebGL resources (textures, buffers, shaders)
  this.sceneManager?.dispose();

  // 5. Remove global event listeners
  window.removeEventListener('beforeunload', this.cleanup);
}
```

## Component Integration

### Scene Manager Integration

The app coordinates scene management with other systems:

```typescript
// Initialize scene foundation
this.sceneManager = new SceneManager();
await this.sceneManager.init();

// Pass scene components to animation controller
this.animationController = new AnimationController(
  this.sceneManager.renderer,    // WebGL renderer
  this.sceneManager.scene,       // THREE.js scene graph
  this.sceneManager.camera,      // Perspective camera
  this.sceneManager.controls,    // Orbit/fly controls
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

  // 2. Load and parse Zarr data (async)
  await this.sceneManager.loadSceneData(src);

  // 3. Initialize nD navigation UI
  this.inputHandler.initDimensionSliders();

  // 4. Configure persistent settings
  this.renderingControls.setSceneId(src);

  // 5. Trigger immediate render
  this.animationController.startAnimation();
}
```

## Dataset Loading

### Intelligent Dataset Detection

The app automatically determines whether to show a dataset browser or load data directly:

```typescript
private async shouldShowBrowser(src: string): Promise<boolean> {
  // Directory URLs (ending with /) show browser
  if (!src || src.endsWith('/')) {
    return true;
  }

  // Check for Zarr dataset markers
  try {
    const response = await fetch(src + '/.zgroup', { method: 'HEAD' });
    if (response.ok) {
      return false; // Valid Zarr dataset, load directly
    }
  } catch {
    // Network errors don't prevent browser display
  }

  // Files without extensions are likely directories
  const hasExtension = src.split('/').pop()?.includes('.');
  return !hasExtension;
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

### Comprehensive Error Recovery

The app implements multiple layers of error handling:

```typescript
// 1. Initialization Error Handling
try {
  await this.init(src);
} catch (error) {
  console.error('Failed to initialize Luxar app:', error);
  // Don't cleanup - preserve error messages for user
  throw error;
}

// 2. Component-Level Error Isolation
async init(src?: string): Promise<void> {
  try {
    // Scene manager initialization
    await this.sceneManager.init();
  } catch (sceneError) {
    // Scene errors are critical but don't prevent UI setup
    console.error('Scene initialization failed:', sceneError);
    throw sceneError;
  }

  // Animation continues even if data loading fails
  this.animationController.startAnimation();

  try {
    // Dataset loading is isolated - failure doesn't break app
    if (await this.shouldShowBrowser(src)) {
      this.showDatasetBrowser();
    } else {
      await this.loadDataset(src);
    }
  } catch (dataError) {
    // Data loading errors are displayed but don't crash app
    console.error('Data loading failed:', dataError);
    showError('Failed to load dataset. Check console for details.');
  }
}
```

### User-Friendly Error Display

```typescript
// Global error handler for unhandled initialization failures
app.init(src).catch((error) => {
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
    app,                    // Access to main app instance
    consoleInterceptor,     // Console message buffer
    version: '1.0.0'        // Application version
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
  app.cleanup();
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

The app automatically handles window focus changes:

```typescript
// Setup in app initialization
private setupFocusHandling(): void {
  // Refresh rendering when window gains focus
  window.addEventListener('focus', () => {
    this.animationController.startAnimation();
    console.log('Window focused - triggering render refresh');
  });

  // Handle tab switching
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      this.animationController.startAnimation();
      console.log('Document became visible - triggering render refresh');
    }
  });
}
```

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
// ✅ Good: Isolate error domains
try {
  await this.sceneManager.init();
} catch (sceneError) {
  // Handle scene-specific errors
  throw sceneError; // Critical errors should propagate
}

try {
  await this.loadDataset(src);
} catch (dataError) {
  // Handle data-specific errors
  showError('Dataset loading failed');
  // Don't propagate - app can continue without data
}

// ✅ Good: Provide fallbacks
const src = params.get('src') ?? config.defaultZarrPath;
```

### Resource Management

```typescript
// ✅ Good: Comprehensive cleanup
cleanup(): void {
  // Stop animation first (prevents new work)
  this.animationController?.dispose();
  
  // Clean up in reverse initialization order
  this.inputHandler?.dispose();
  this.renderingControls?.dispose();
  this.sceneManager?.dispose();
  
  // Remove global listeners
  window.removeEventListener('beforeunload', this.cleanup);
}

// ✅ Good: Automatic cleanup registration
private setupCleanup(): void {
  window.addEventListener('beforeunload', this.cleanup.bind(this));
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
}

// ✅ Good: URL parameter parsing
const params = new URLSearchParams(window.location.search);
const src = params.get('src') ?? config.defaultZarrPath;
```

The core package provides the essential coordination and lifecycle management that transforms individual Luxar components into a cohesive, reliable visualization application.