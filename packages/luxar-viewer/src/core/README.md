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
├── main.ts                          # Standalone-app HTML entry point (~30 lines)
├── bootstrap.ts                     # Pre-init sequence (theme, console patch,
│                                    #   codec warm, debug surface) + the standalone
│                                    #   factory `bootstrapStandalone()`
├── document-title.ts                # Browser tab title: authored `viewer_config.title`
│                                    #   > `?title=` > the page's own `<title>`
├── app.ts                           # `LuxarApp` — the core embed surface
│                                    #   (`init` / `switchDataset` / `dispose`)
│                                    # ~600 LOC of orchestration glue: every method is
│                                    # a 3–15 line delegate to a helper in `app/`.
└── app/                             # LuxarApp's support tree (private except embedder/ types)
    ├── factories.ts                 # Factory overrides resolver for heavy subsystems
    ├── options.ts                   # `LuxarAppOptions` interface (re-exported from app.ts)
    ├── init/                        # init() pipeline
    │   ├── pipeline.ts              # Build the subsystem graph, returns components
    │   ├── build-rail-items.ts      # Build control-rail descriptors from shared UI actions
    │   ├── environment-guards.ts    # Browser + THREE.REVISION peer-dep checks
    │   └── module-overrides.ts      # Wire `wasmPath` / `workerPath` into module singletons
    ├── lifecycle/                   # dispose / focus / unload / connectivity recovery
    │   ├── dispose-pipeline.ts      # Per-component safeDispose teardown + singleton clear
    │   ├── focus-handling.ts        # Window-focus + visibility-change → animation pause/resume
    │   ├── online-retry.ts          # `online` event → bounded retry of failed scene loads
    │   └── unload-handling.ts       # `beforeunload` → app.dispose()
    ├── dataset/                     # Dataset routing
    │   ├── load-dataset.ts          # Scene load + scene-dependent UI init sequence
    │   ├── show-browser.ts          # Open the DatasetBrowser modal
    │   ├── should-show-browser.ts   # URL-classification + zarr-metadata HEAD probe
    │   ├── browser-decision.ts      # Pure URL classifier (must-browse / probe / load)
    │   └── browser-shortcut.ts      # `luxar-open-dataset-browser` custom-event listener
    ├── viewer-config/               # Zarr `viewer_config` + panel visibility
    │   ├── apply-state.ts           # Dispatch `viewer_config` fields to UI subsystems
    │   └── panel-visibility.ts      # Capture / restore RenderingControls + RecordingPanel state
    ├── snapshot/                    # Camera / dimension state JSON
    │   └── viewer-snapshot.ts       # captureSnapshot / restoreSnapshot for embed share-links
    ├── embedder/                    # Public programmatic embedder API
    │   ├── events.ts                # `LuxarEmbedderEventMap` + value types (`EmbedderDimensions`,
    │   │                            #   `ScreenshotOptions`, …) — re-exported from `src/index.ts`
    │   └── screenshot.ts            # `captureScreenshot` — headless frame → encoded Blob
    ├── debug/                       # `window.__luxarDebug` surface
    │   ├── debug-interface.ts       # Populate `__luxarDebug` with scene/camera/runtime refs
    │   ├── debug-state.ts           # Scene-walking helper for `__luxarDebug.getState()`
    │   ├── debug-cache-helpers.ts   # `__luxarDebug.cache.*` thin wrappers over SceneLoader
    │   └── cache-stats-view.ts      # Open the data-monitor on the Cache tab
    ├── picking/                     # GPU picking init + result handler
    │   ├── init-picking.ts          # Stand up PickingSystem + LabelLoaders + listeners
    │   └── pick-result-handler.ts   # Pure pick-result → OverlayManager hover content
    └── overlays/                    # Screen-space overlay subsystems
        ├── init-overlays.ts         # OverlayManager + zarr overlay_groups load
        ├── init-scale-bar.ts        # ScaleBar + per-frame update + keyboard toggle
        ├── init-colormap-legend.ts  # ColormapLegend + layer-state wiring
        └── dispose-overlays.ts      # Tear down OverlayManager (called by loadDataset)
```

**Layout rule** (Principle 1: depth ≠ specificity, applied recursively):

- `app.ts`, `bootstrap.ts`, `main.ts` sit at the package root because they have external importers (the package barrel + the standalone bundler entry).
- Nearly every file under `app/` has zero non-test external importers — it's LuxarApp's private support code, grouped thematically into `init/`, `lifecycle/`, `dataset/`, `viewer-config/`, `snapshot/`, `embedder/`, `debug/`, `picking/`, `overlays/`. The public exceptions are `app/embedder/events.ts` (event/screenshot types) and `app/snapshot/viewer-snapshot.ts` (`type ViewerSnapshot`), both re-exported from `src/index.ts`; nothing else under `app/` is exported.
- Tests mirror the source layout under `tests/unit/core/app/<theme>/`.

**Public exports**: `class LuxarApp` (with `init(options)`, `dispose()`, `captureSnapshot()`, `restoreSnapshot(snap)`, plus the programmatic embedder API: `on(event, listener)`, `switchDataset(src)`, `getDimensions()`, `setDimensionValue(index, value)`, `getCameraPose()` / `setCameraPose(pose)` / `flyTo(pose, opts)`, `getRenderingSettings()` / `setRenderingSettings(patch)`, `getLayers()` / `setLayer(path, patch)`, `getViewerState()`, `resize()`, `screenshot(opts)`, and the flat keyboard methods `registerContext`, `unregisterContext`, `registerBinding`, `unregisterBinding`, `pushContext`, `popContext`, `setInputEnabled`, and `shortcutForAction`) and `LuxarAppOptions` from `app.ts`; `bootstrapStandalone()` from `bootstrap.ts`. Inside `app/`, only the embedder types from `app/embedder/events.ts` (`LuxarEmbedderEventMap`, `EmbedderDimensions`, `ScreenshotOptions`, …) and `type ViewerSnapshot` from `app/snapshot/viewer-snapshot.ts` are exported externally — they are re-exported from `src/index.ts`.

**Dependencies**: internally couples to `scene`, `controls`, `input`, `data`,
`rendering`, `ui/*`, `config`, `themes`, and `utils`; externally only `three`.

**Component Dependency Flow**:

```
main.ts → bootstrapStandalone() → LuxarApp.init() → SceneManager → AnimationController
                                                  → InputHandler ← RenderingControls
                                                  → DatasetBrowser (conditional)
```

Embedded callers skip `main.ts` and `bootstrapStandalone()` entirely — they
construct `LuxarApp` themselves and call `init({ canvas, src, ... })`. Embedded
`LuxarApp` instances do not rewrite the host page URL unless
`updateBrowserUrl: true` is passed explicitly.

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
    this.inputHandler = new InputHandler(
      this.sceneManager,
      this.animationController,
      this.performanceMonitor,
      this.debugConsole,
      /* optional */ (parent) => new DimensionSliders(parent)
    );
    this.inputHandler.init();

    // 4. UI Controls Configuration
    this.renderingControls = new RenderingControls(/*...*/);

    // 5. Component Cross-Linking
    this.renderingControls.setAnimationController(this.animationController);
    this.inputHandler.setRenderingControls(this.renderingControls);

    // 6. Animation Loop Start
    this.animationController.startAnimation();

    // 7. Dataset Browser Shortcut
    this.setupDatasetBrowserShortcut();

    // 8. Dataset Loading or Browser Display
    const src = options.src ?? config.defaultZarrPath;
    if (await this.shouldShowBrowser(src)) {
      this.showDatasetBrowser();
    } else {
      await this.loadDataset(src);
    }

    // 9. System Event Handling
    this.setupDisposeOnUnload();
    this.setupFocusHandling();
  }
}
```

**Critical Design Decisions**:

- **Animation First**: Start rendering loop before loading data for immediate visual feedback
- **Serialized Dataset Switches**: Programmatic switches and built-in browser selections share one in-flight guard; the browser cannot reopen until the active teardown+reload finishes
- **Error Isolation**: Component failures don't prevent other systems from initializing
- **Progressive Enhancement**: Core 3D functionality works even if data loading fails

### Disposal and Resource Management

`dispose()` is a thin orchestrator: it flips the idempotency / re-entrance
guards, marks the app uninitialized, then hands every component to
`runDisposePipeline()` (in `app/lifecycle/dispose-pipeline.ts`), which
`safeDispose`s each one and clears the singleton state. App-level listeners
(`beforeunload`, `focus`, `visibilitychange`, the dataset-browser shortcut,
plus the picking system's per-scene listeners) are owned by two `EventGroup`s
(`this.events`, `this.pickingEvents`) and torn down in one call from the
pipeline — there is no hand-rolled `removeEventListener` here.

```typescript
dispose(): void {
  // Idempotency: a second dispose() after a successful one is a no-op.
  if (this.isDisposed) return;
  // Re-entrance guard: a beforeunload firing mid-dispose does nothing.
  if (this.isDisposing) return;
  this.isDisposing = true;

  // Flip initialized at entry so concurrent observers of `app.initialized`
  // see teardown immediately, even if teardown throws partway through.
  this.isInitialized = false;

  runDisposePipeline({
    events: this.events,             // beforeunload / focus / shortcut listeners
    pickingEvents: this.pickingEvents,
    sceneManager: this.sceneManager, // WebGL/WebGPU resources
    animationController: this.animationController, // stops the render loop
    inputHandler: this.inputHandler,
    renderingControls: this.renderingControls,
    // …all panels, overlays, picking system, label loaders, dataset browser…
    clearScaleBar: () => { this.scaleBar = undefined; },
    // …per-field clear callbacks so the orchestrator never reaches in…
  });

  this.isDisposing = false;
  this.isDisposed = true;
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
this.inputHandler = new InputHandler(
  this.sceneManager,
  this.animationController,
  this.performanceMonitor,
  this.debugConsole,
  /* optional */ (parent) => new DimensionSliders(parent)
);

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
      const fullURL = constructFullURL(path);

      // Optional: standalone bootstrap opts in, embeds default out.
      if (this.options.updateBrowserUrl === true) {
        replaceBrowserDataSourceUrl(fullURL);
      }

      // Load selected dataset
      await this.loadDataset(fullURL);
    },
    onClose: () => {
      this.datasetBrowser = undefined;
    }
  });
}
```

## Error Handling

### Error Recovery

The `init()` method uses a single top-level try/catch. All initialization steps (the subsystem-building pipeline, dataset routing, dispose/focus/browser/debug handler installation) run inside this block. If any step fails, `init()` calls `dispose()` to tear down whatever partial state was constructed, then re-throws to the caller:

```typescript
async init(options: LuxarAppOptions): Promise<void> {
  try {
    // 1. runInitPipeline — builds scene/animation/input/UI subsystems
    //    (each assigned to `partial` so dispose() can find them after a throw)
    // 2. setupDatasetBrowserShortcut()
    // 3. Dataset routing — showDatasetBrowser() or loadDataset()
    // 4. setupDisposeOnUnload / setupFocusHandling / setupDebugInterface
  } catch (error) {
    log.error(Modules.APP, 'Failed to initialize Luxar app:', error);
    // Surface whatever subsystems were built before the throw, then dispose.
    assignFromPartial();
    this.dispose();
    throw error;
  }
}
```

`bootstrapStandalone()` catches and displays startup errors. Authored archive
faults carry safe, actionable remedies; unrecognized failures keep the generic
fallback. Fatal startup dialogs remain until the user dismisses them:

```typescript
app.init({ canvas, src }).catch((error) => {
  console.error('Failed to start Luxar application:', error);
  const message =
    error instanceof ArchiveFaultError
      ? error.message
      : 'Failed to start the application. Please check the console for details.';
  showError(message, shortcutForAction, shortcutActions, { autoDismiss: false });
});
```

### User-Friendly Error Display

```typescript
// Global error handler for unhandled initialization failures
app.init({ canvas, src }).catch((error) => {
  console.error('Failed to start Luxar application:', error);
  const message =
    error instanceof ArchiveFaultError
      ? error.message
      : 'Failed to start the application. Please check the console for details.';
  showError(message, shortcutForAction, shortcutActions, { autoDismiss: false });
});
```

## Debug Features

### Development Interface

The app exposes a debug interface when in development mode:

```typescript
// bootstrap.ts debug setup
const isDebugMode = urlParams.debug || localStorage.getItem('luxar.debug') === 'true';

// The build stamp is published FIRST and unconditionally — a bug report needs
// a revision whether or not the reporter knew to pass `?debug`.
window.__luxarBuild = buildInfo();

if (isDebugMode) {
  window.__luxarDebug = {
    app, // Access to main app instance
    consoleInterceptor, // Console message buffer
    version: buildInfo().version, // Build stamp — see config/build-info.ts
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
    renderingControls: this.renderingControls,
    adaptiveDPRManager: this.adaptiveDPRManager,
  };
}
```

## Usage Examples

### Basic Application Initialization

```typescript
import { LuxarApp } from './core/app';

const canvas = document.getElementById('app') as HTMLCanvasElement;

// Create and initialize app
const app = new LuxarApp();

// Start with specific dataset
await app.init({ canvas, src: '/data/my-dataset.luxar.zarr' });

// Start with directory browser (trailing slash → browser)
await app.init({ canvas, src: '/data/' });

// Start with no source — app shows the dataset browser
await app.init({ canvas });
```

### URL Parameter Integration

```typescript
// Standalone callers use `bootstrapStandalone` (which calls `readUrlParams`).
// Embedders typically parse URL params themselves and pass the result:
const params = new URLSearchParams(window.location.search);
const datasetURL = params.get('src') ?? '/data/default.luxar.zarr';

const app = new LuxarApp();
await app.init({ canvas, src: datasetURL });
```

### Error Handling and Recovery

```typescript
const app = new LuxarApp();

try {
  await app.init({ canvas, src: datasetURL });
  console.log('Application started successfully');
} catch (error) {
  console.error('Initialization failed:', error);

  // init() always calls dispose() on its own failure path before
  // re-throwing, so the instance is already torn down. Calling
  // dispose() again here is safe (it's idempotent) but redundant.
  app.dispose();
}
```

### Component Access and Testing

```typescript
const app = new LuxarApp();
await app.init({ canvas });

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

The orchestrator delegates to `installFocusHandling` (in
`app/lifecycle/focus-handling.ts`), which registers both listeners on
`this.events` so dispose() reclaims them. Both handlers early-return while a
recording is in progress, so offline capture keeps a stable, deterministic
loop regardless of tab focus:

```typescript
// app/lifecycle/focus-handling.ts (essence)
ports.events.on(window, 'focus', () => {
  if (ports.getRecordingPanel()?.isCurrentlyRecording()) return;
  ports.animationController.startAnimation();
  log.info(Modules.LUXAR, 'Window focused - triggering render refresh');
});

ports.events.on(document, 'visibilitychange', () => {
  if (ports.getRecordingPanel()?.isCurrentlyRecording()) return;
  if (document.hidden) {
    // Tab hidden - stop completely to guarantee zero CPU/GPU usage
    ports.animationController.stopAnimation();
    log.info(Modules.LUXAR, 'Document hidden - stopping animation to save resources');
  } else {
    // Tab visible - resume rendering
    ports.animationController.startAnimation();
    log.info(Modules.LUXAR, 'Document became visible - resuming animation');
  }
});
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
// ✅ Good: Single try/catch in init(); dispose partial state, then re-throw
async init(options: LuxarAppOptions): Promise<void> {
  try {
    // All steps in sequence; any failure propagates
    const result = await runInitPipeline(/* … */, partial);
    // ... dataset routing + handler installation ...
  } catch (error) {
    assignFromPartial(); // surface partial subsystems on this.*
    this.dispose();      // tear them down
    throw error;
  }
}

// ✅ Good: Caller displays errors to user
app.init({ canvas, src }).catch((error) => {
  const message =
    error instanceof ArchiveFaultError
      ? error.message
      : 'Failed to start the application. Please check the console for details.';
  showError(message, shortcutForAction, shortcutActions, { autoDismiss: false });
});

// ✅ Good: Provide fallbacks for optional values
const src = params.get('src') ?? config.defaultZarrPath;
```

### Resource Management

```typescript
// ✅ Good: Idempotent disposal delegated to the dispose pipeline
dispose(): void {
  if (this.isDisposed || this.isDisposing) return; // guard double / nested calls
  this.isDisposing = true;
  this.isInitialized = false;
  runDisposePipeline({ events: this.events, sceneManager: this.sceneManager, /* … */ });
  this.isDisposing = false;
  this.isDisposed = true;
}

// ✅ Good: own listeners through an EventGroup, not hand-rolled bound refs.
// `installUnloadHandler` registers the beforeunload listener AND its
// removal on `this.events`, so dispose() reclaims it with one teardown call.
private events = new EventGroup();

private setupDisposeOnUnload(): void {
  installUnloadHandler({ events: this.events, dispose: () => this.dispose() });
}
```

### Component Communication

```typescript
// ✅ Good: Explicit dependency injection
this.inputHandler = new InputHandler(
  this.sceneManager,
  this.animationController,
  this.performanceMonitor,
  this.debugConsole,
  /* optional */ (parent) => new DimensionSliders(parent)
);

// ✅ Good: Post-initialization linking
this.renderingControls.setAnimationController(this.animationController);
this.inputHandler.setRenderingControls(this.renderingControls);

// ❌ Avoid: Hidden global dependencies or tight coupling
// Don't access components through global variables
```

### URL and State Management

```typescript
// ✅ Good: Opt-in URL synchronization through the centralized helper
onDatasetSelect: async (path: string) => {
  const fullURL = constructFullURL(path);
  if (this.options.updateBrowserUrl === true) {
    replaceBrowserDataSourceUrl(fullURL);
  }

  // Load dataset
  await this.loadDataset(fullURL);
};

// ✅ Good: URL parameter parsing is centralized in bootstrap/readUrlParams
const urlParams = readUrlParams();
const src = urlParams.src ?? config.defaultZarrPath;
```

The core package provides the essential coordination and lifecycle management that transforms individual Luxar components into a cohesive, reliable visualization application.
