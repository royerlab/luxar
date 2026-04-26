# luxar-viewer.core - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2025-12-09

## Purpose

The `luxar-viewer.core` package provides application initialization, component orchestration, and lifecycle management for the Luxar visualization client. It coordinates all subsystems and manages the application startup sequence.

**Core Responsibility**: Initialize all components in correct dependency order, coordinate inter-component communication, handle URL parameters, manage dataset loading, and ensure proper resource cleanup.

**Related Specifications**:

- All other packages - Core orchestrates all systems

---

## Table of Contents

1. [Application Initialization](#application-initialization)
2. [Component Dependency Graph](#component-dependency-graph)
3. [Dataset Loading](#dataset-loading)
4. [Error Handling](#error-handling)
5. [Resource Cleanup](#resource-cleanup)
6. [Debug Interface](#debug-interface)

---

## 1. Application Initialization

### 1.1 Initialization Sequence

**Critical Requirement**: Components must be initialized in dependency order.

**Sequence**:

```typescript
async function init(src?: string): Promise<void> {
  // 1. Foundation: Scene management
  this.sceneManager = new SceneManager();
  await this.sceneManager.init();

  // 2. Animation system (depends on scene)
  this.animationController = new AnimationController(
    this.sceneManager.controls,
    this.sceneManager.postProcessing
  );

  // 3. Input handling (depends on scene and animation)
  this.inputHandler = new InputHandler(this.sceneManager, this.animationController);
  this.inputHandler.init();

  // 4. UI controls (depends on scene and animation)
  this.renderingControls = new RenderingControls(
    this.sceneManager.postProcessing,
    this.sceneManager
  );

  // 5. Cross-linking (bidirectional dependencies)
  this.renderingControls.setAnimationController(this.animationController);
  this.inputHandler.setRenderingControls(this.renderingControls);

  // 6. Start rendering
  this.animationController.startAnimation();

  // 7. Load data or show browser
  if (await this.shouldShowBrowser(src)) {
    this.showDatasetBrowser();
  } else {
    await this.loadDataset(src);
  }

  // 8. Setup lifecycle handlers
  this.setupCleanup();
  this.setupFocusHandling();
}
```

**Critical**: Animation **must** start before loading data to show loading indicators.

---

## 2. Component Dependency Graph

### 2.1 Dependency Relationships

```
SceneManager (foundation)
    ├─→ AnimationController
    │   └─→ RenderingControls ←──┐
    ├─→ InputHandler              │
    │   └─→ RenderingControls ────┘
    └─→ DataLoader
```

**Initialization Rules**:

1. Initialize foundation first (SceneManager)
2. Initialize dependent components in topological order
3. Perform cross-linking after all components exist
4. Start animation before data loading

### 2.2 Cross-Component Communication

**Bidirectional Dependencies**:

```typescript
// RenderingControls needs AnimationController
renderingControls.setAnimationController(animationController);

// InputHandler needs RenderingControls
inputHandler.setRenderingControls(renderingControls);
```

**Rationale**: Avoids circular dependencies while enabling necessary communication.

---

## 3. Dataset Loading

### 3.1 Dataset Detection Algorithm

**Purpose**: Determine if URL is a Zarr dataset or directory.

```typescript
async function shouldShowBrowser(src: string): Promise<boolean> {
  // 1. Empty or trailing slash → directory
  if (!src || src.endsWith('/')) {
    return true;
  }

  // 2. Check for Zarr markers (.zgroup, .zattrs, zarr.json) via Promise.any()
  try {
    await Promise.any([
      fetch(src + '/.zgroup', { method: 'HEAD' }).then(r => { if (!r.ok) throw r; return r; }),
      fetch(src + '/.zattrs', { method: 'HEAD' }).then(r => { if (!r.ok) throw r; return r; }),
      fetch(src + '/zarr.json', { method: 'HEAD' }).then(r => { if (!r.ok) throw r; return r; }),
    ]);
    return false; // Valid Zarr, load directly
  } catch {
    // None of the markers found or network error
  }

  // 3. Files without extensions likely directories
  const lastSegment = src.split('/').pop() || '';
  const hasExtension = lastSegment.includes('.');

  return !hasExtension;
}
```

### 3.2 Dataset Loading Sequence

```typescript
async function loadDataset(src: string): Promise<void> {
  // 1. Clear previous dataset UI
  this.inputHandler.clearDimensionUI();

  // 2. Load scene data (async)
  try {
    await this.sceneManager.loadSceneData(src);
  } catch (error) {
    showError(`Failed to load dataset: ${error.message}`);
    throw error;
  }

  // 3. Initialize dimension UI (if nD data)
  this.inputHandler.initDimensionSliders();

  // 4. Configure persistent settings
  this.renderingControls.setSceneId(src);

  // 5. Trigger render
  this.animationController.startAnimation();

  // 6. Update URL
  this.updateURLParameter('src', src);
}
```

---

## 4. Error Handling

### 4.1 Error Isolation

**Principle**: Component failures should not cascade.

**Implementation**:

```typescript
async function init(src?: string): Promise<void> {
  try {
    // Critical: Scene initialization (failure is fatal)
    await this.sceneManager.init();

    // Start animation before data loading (enables progress rendering)
    this.animationController.startAnimation();

    // Load data or show browser
    if (await this.shouldShowBrowser(src)) {
      this.showDatasetBrowser();
    } else {
      await this.loadDataset(src);
    }
  } catch (error) {
    console.error('Initialization failed:', error);
    showError(`Failed to initialize: ${error.message}`);
    throw error;
  }
}
```

### 4.2 User-Facing Error Messages

```typescript
function showError(message: string): void {
  // Create error overlay
  const errorDiv = document.createElement('div');
  errorDiv.className = 'luxar-error-overlay';
  errorDiv.innerHTML = `
        <div class="error-content">
            <h3>❌ Error</h3>
            <p>${message}</p>
            <button data-action="dismiss">Dismiss</button>
        </div>
    `;

  document.body.appendChild(errorDiv);

  // Auto-dismiss after 10 seconds
  setTimeout(() => errorDiv.remove(), 10000);
}
```

---

## 5. Resource Cleanup

### 5.1 Cleanup Sequence

**Purpose**: Properly dispose all resources to prevent memory leaks.

**Order** (reverse of initialization):

```typescript
function cleanup(): void {
  // 1. Stop animation (prevents new work)
  this.animationController?.dispose();

  // 2. Remove input listeners
  this.inputHandler?.dispose();

  // 3. Dispose UI components
  this.renderingControls?.dispose();
  this.datasetBrowser?.dispose();

  // 4. Dispose data loaders
  dispose(); // From data/zarr-loader

  // 5. Dispose scene (WebGL resources)
  this.sceneManager?.dispose();

  // 6. Remove global listeners
  window.removeEventListener('beforeunload', this.cleanup);
  window.removeEventListener('resize', this.handleResize);
}
```

### 5.2 WebGL Resource Disposal

**Critical**: Properly dispose geometries and materials to free GPU memory.

```typescript
function disposeScene(): void {
  scene.traverse((object) => {
    // Dispose geometries
    if (object.geometry) {
      object.geometry.dispose();
    }

    // Dispose materials
    if (object.material) {
      if (Array.isArray(object.material)) {
        object.material.forEach((m) => m.dispose());
      } else {
        object.material.dispose();
      }
    }

    // Dispose textures
    if (object.material?.map) {
      object.material.map.dispose();
    }
  });

  // Dispose renderer
  renderer.dispose();
  renderer.forceContextLoss();
}
```

---

## Data Structures

### LuxarApp

```typescript
interface LuxarApp {
  // Components
  sceneManager: SceneManager;
  animationController: AnimationController;
  inputHandler: InputHandler;
  renderingControls: RenderingControls;
  datasetBrowser?: DatasetBrowser;

  // State
  initialized: boolean;
  currentDataset: string | null;

  // Methods
  init(src?: string): Promise<void>;
  loadDataset(src: string): Promise<void>;
  cleanup(): void;
}
```

### LuxarAppOptions

```typescript
interface LuxarAppOptions {
  canvas: HTMLCanvasElement;     // Render target (resolved by main.ts)
  src?: string;                  // Dataset URL (defaults to config.defaultZarrPath)
  debug?: boolean;               // Enable window.__luxarDebug + verbose logging
  loaderConfig?: LoaderConfig;   // Cache and prefetch flags
  updateBrowserUrl?: boolean;    // Mirror selected dataset into the URL bar (default true)
}
```

---

## 6. Debug Interface

### 6.1 Debug Interface Architecture

**Purpose**: Expose internal application state and helpers for testing, development, and AI-assisted debugging.

**Critical Design**: Two-stage initialization pattern:

1. **Stage 1 - Base Interface** (`main.ts`): Expose foundation properties before runtime initialization
2. **Stage 2 - Runtime Components** (`app.ts`): Add runtime properties after components are initialized

**Activation**: Debug interface only available when `?debug` URL parameter is present OR `localStorage.luxar_debug === 'true'`

```typescript
// URL activation
//localhost:5173/?debug

// Programmatic activation
http: localStorage.setItem('luxar_debug', 'true');
```

**TypeScript Declaration**:

```typescript
declare global {
  interface Window {
    __luxarDebug?: {
      // Base properties (available from main.ts)
      app: LuxarApp;
      consoleInterceptor: typeof consoleInterceptor;
      version: string;

      // Runtime properties (added by app.ts after initialization)
      scene?: THREE.Scene;
      camera?: THREE.PerspectiveCamera;
      renderer?: THREE.WebGLRenderer;
      controls?: any;
      postProcessing?: any;
      animationController?: any;
      inputHandler?: any;
      renderingControls?: any;
      getState?: () => any;
      renderOnce?: () => void;
      getSceneLoader?: () => Promise<any>;
      runtimeReady?: boolean;
      cache?: CacheDebugAPI;
    };
  }
}
```

---

### 6.2 Base Debug Properties (Stage 1)

**Location**: `main.ts` (lines 64-71)

**Properties**:

```typescript
window.__luxarDebug = {
  app: LuxarApp, // Application instance
  consoleInterceptor: object, // Ring buffer console interceptor
  version: string, // Application version (e.g., '1.0.0')
};
```

**Purpose**: Available immediately during initialization, before any components are created.

**Usage**:

```javascript
// Check version
console.log(window.__luxarDebug.version);

// Access app instance
window.__luxarDebug.app.components;

// Read console history
window.__luxarDebug.consoleInterceptor.getMessages();
```

---

### 6.3 Runtime Debug Properties (Stage 2)

**Location**: `app.ts` `setupDebugInterface()` method (lines 273-440)

**Properties**:

```typescript
{
  // Three.js core components
  scene: THREE.Scene,                      // Main scene object
  camera: THREE.PerspectiveCamera,         // Active camera
  renderer: THREE.WebGLRenderer,           // WebGL renderer instance
  controls: ControlsManager,               // Camera controls manager
  postProcessing: PostProcessingManager,   // HDR post-processing pipeline

  // Application components
  animationController: AnimationController, // Animation loop controller
  inputHandler: InputHandler,              // Input event handler
  renderingControls: RenderingControls,    // Rendering UI controls

  // Helper functions (see sections below)
  getState: () => StateSnapshot,           // Get current state
  renderOnce: () => void,                  // Trigger single frame
  getSceneLoader: () => Promise<SceneLoaderManager>, // Get loader instance

  // Cache API (see section 6.5)
  cache: CacheDebugAPI,

  // Ready flag
  runtimeReady: boolean                    // True when runtime props added
}
```

**Important**: Runtime properties only available after `app.init()` completes. Check `runtimeReady` flag.

**Usage**:

```javascript
// Access Three.js scene
window.__luxarDebug.scene.children;

// Get camera position
const pos = window.__luxarDebug.camera.position;
console.log(`Camera at (${pos.x}, ${pos.y}, ${pos.z})`);

// Access controls
window.__luxarDebug.controls.setOrbitMode();
```

---

### 6.4 Helper Functions

#### getState()

**Purpose**: Get comprehensive snapshot of current application state.

**Signature**:

```typescript
getState(): {
  totalPoints: number;
  pointClouds: Array<{
    name: string;
    pointCount: number;
    visible: boolean;
    hasColors: boolean;
    hasRadii: boolean;
    hasSharpness: boolean;
  }>;
  dimensions: {
    ndim: number;
    displayed: number[];
    currentStep: number[];
  } | null;
  cameraPosition: { x: number; y: number; z: number };
  cameraFov: number;
  isAnimating: boolean;
  initialized: boolean;
}
```

**Usage**:

```javascript
const state = window.__luxarDebug.getState();
console.log(`Loaded ${state.totalPoints} points`);
console.log(`Camera FOV: ${state.cameraFov}`);
console.log(`Dimensions: ${state.dimensions?.ndim}D`);
```

**Implementation Details**:

- Traverses scene to count points across all `THREE.Points` objects
- Inspects geometry attributes for metadata
- Queries `sceneDimsManager` for dimensional state
- Returns camera position and FOV

#### renderOnce()

**Purpose**: Trigger a single frame render (useful for stable screenshots).

**Signature**:

```typescript
renderOnce(): void
```

**Usage**:

```javascript
// Take screenshot after ensuring fresh render
window.__luxarDebug.renderOnce();
setTimeout(() => {
  // Screenshot code here
}, 100);
```

**Implementation**: Calls `animationController.startAnimation()` to trigger render loop.

#### getSceneLoader()

**Purpose**: Get `SceneLoaderManager` singleton for cache inspection.

**Signature**:

```typescript
async getSceneLoader(): Promise<SceneLoaderManager>
```

**Usage**:

```javascript
const manager = await window.__luxarDebug.getSceneLoader();
const loader = manager.getDefaultLoader();
console.log(loader);
```

**Note**: Uses dynamic import to avoid circular dependencies.

---

### 6.5 Cache Debug API

**Purpose**: Inspect and manipulate two-tier cache (L1 memory + L2 OPFS).

**Location**: `app.ts` lines 362-423

**API Methods**:

```typescript
interface CacheDebugAPI {
  getStats(): Promise<CacheStats>;
  listDatasets(): Promise<DatasetList>;
  clearL1(): Promise<void>;
  clearL2(): Promise<void>;
  clearAll(): Promise<void>;
}
```

#### cache.getStats()

**Purpose**: Get cache statistics (sizes, hit rates, datasets).

**Returns**:

```typescript
{
  l1: { size: number; maxSize: number; hitRate: number };
  l2: { size: number; datasets: string[] };
}
```

**Usage**:

```javascript
const stats = await window.__luxarDebug.cache.getStats();
console.log(`L1 cache: ${stats.l1.size} / ${stats.l1.maxSize} bytes`);
console.log(`L2 datasets: ${stats.l2.datasets.join(', ')}`);
```

#### cache.listDatasets()

**Purpose**: List all datasets in L2 cache with metadata.

**Returns**:

```typescript
{
  datasets: Array<{
    name: string;
    chunkCount: number;
    totalSize: number;
  }>;
}
```

**Usage**:

```javascript
const datasets = await window.__luxarDebug.cache.listDatasets();
datasets.forEach((d) => {
  console.log(`${d.name}: ${d.chunkCount} chunks, ${d.totalSize} bytes`);
});
```

#### cache.clearL1()

**Purpose**: Clear L1 (memory) cache only, preserving L2 (OPFS).

**Usage**:

```javascript
await window.__luxarDebug.cache.clearL1();
console.log('L1 cache cleared - L2 preserved');
```

#### cache.clearL2()

**Purpose**: Clear L2 (OPFS) cache only, preserving L1 (memory).

**Usage**:

```javascript
await window.__luxarDebug.cache.clearL2();
console.log('L2 cache cleared - L1 preserved');
```

#### cache.clearAll()

**Purpose**: Clear both L1 and L2 caches completely.

**Usage**:

```javascript
await window.__luxarDebug.cache.clearAll();
console.log('All caches cleared');
```

---

### 6.6 Console Interceptor Requirements

**Critical**: Console interceptor MUST be imported FIRST in `main.ts` (line 6), before any other code.

**Why**: Ensures ALL console output from application start is captured, including early errors and initialization logs.

**Implementation**:

```typescript
// CRITICAL: Import console interceptor FIRST before any other code
// This ensures we capture ALL console output from the very beginning
import { consoleInterceptor } from '../utils/console-interceptor';

// Log that we're starting (this will be captured)
import { log, Modules, LogEmoji } from '../utils/log';
log.custom(LogEmoji.START, Modules.LUXAR, 'Application starting...');

// ... rest of imports ...
```

**Ring Buffer Mechanism**:

- Fixed-size circular buffer (10,000 messages)
- Oldest messages discarded when buffer full
- Preserves message type (log, warn, error, info, debug)
- Timestamp for each message

**Usage**:

```javascript
// Get all captured console messages
const messages = window.__luxarDebug.consoleInterceptor.getMessages();

// Filter by type
const errors = messages.filter((m) => m.type === 'error');

// Recent messages (last 100)
const recent = messages.slice(-100);
```

**Importance**: Without early import, initialization errors and startup logs are lost.

---

### 6.7 Debug Console (Ctrl+L)

**Purpose**: In-app console overlay for viewing captured console output.

**Activation**: Press `Ctrl+L` to toggle debug console visibility.

**Features**:

- Displays all captured console messages with timestamps
- Color-coded by log level (error, warn, info, log)
- Auto-scroll to latest messages
- Filterable by log level
- Resizable and draggable

**Configuration**: See `config/index.ts` under `ui.debugConsole` for dimensions and styling.

**Usage**:

```javascript
// Programmatically show debug console
document.dispatchEvent(new CustomEvent('toggle-debug-console'));
```

---

### 6.8 AI-Assisted Debugging Workflow

**Purpose**: Enable autonomous debugging by AI agents (Claude Code) without user intervention.

**Playwright Integration**: See `PLAYWRIGHT_GUIDE.md` for complete details.

**Quick Commands**:

```bash
# Headless mode - shows browser console in terminal
cd packages/luxar-viewer
pnpm agent:debug

# Visible browser - watch AI interact with app
pnpm agent:debug:visible
```

**Output Format**:

```
[BROWSER-CONSOLE-LOG] [🚀] [Luxar] Application starting...
[BROWSER-CONSOLE-ERROR] Failed to load spatial index: 404
[BROWSER-CONSOLE-LOG] Loaded 50,000 points in 245ms

Scene State JSON: {
  "totalPoints": 50000,
  "pointClouds": [...],
  "dimensions": {...},
  "cameraPosition": {...}
}

Screenshot saved: debug-view.png
```

**AI Debug Pattern**:

1. User reports issue
2. AI runs `pnpm agent:debug`
3. AI reads console output and state JSON
4. AI identifies root cause
5. AI makes fixes
6. AI verifies with `pnpm agent:debug` again

---

## Data Structures

### LuxarApp

```typescript
interface LuxarApp {
  // Components
  sceneManager: SceneManager;
  animationController: AnimationController;
  inputHandler: InputHandler;
  renderingControls: RenderingControls;
  datasetBrowser?: DatasetBrowser;

  // State
  initialized: boolean;
  currentDataset: string | null;

  // Methods
  init(src?: string): Promise<void>;
  loadDataset(src: string): Promise<void>;
  cleanup(): void;
}
```

### LuxarAppOptions

```typescript
interface LuxarAppOptions {
  canvas: HTMLCanvasElement;     // Render target (resolved by main.ts)
  src?: string;                  // Dataset URL (defaults to config.defaultZarrPath)
  debug?: boolean;               // Enable window.__luxarDebug + verbose logging
  loaderConfig?: LoaderConfig;   // Cache and prefetch flags
  updateBrowserUrl?: boolean;    // Mirror selected dataset into the URL bar (default true)
}
```

---

## Changelog

- **v1.1.0** (2025-12-09): Added comprehensive debug interface documentation
  - Two-stage debug interface architecture (base + runtime)
  - Base properties: app, consoleInterceptor, version
  - Runtime properties: scene, camera, renderer, controls, postProcessing, etc.
  - Helper functions: getState(), renderOnce(), getSceneLoader()
  - Cache debug API: getStats(), listDatasets(), clearL1(), clearL2(), clearAll()
  - Console interceptor requirements and early import importance
  - Debug console (Ctrl+L) documentation
  - AI-assisted debugging workflow with Playwright

- **v1.0.0** (2025-01-30): Initial specification
  - Component initialization in dependency order
  - Intelligent dataset detection (Zarr vs directory)
  - Error isolation (component failures don't cascade)
  - Proper resource cleanup in reverse order
  - URL parameter handling
  - Focus and visibility event handling
  - Debug interface exposure
  - Cross-component communication patterns
