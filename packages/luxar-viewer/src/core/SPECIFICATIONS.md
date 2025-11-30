# luxar-viewer.core - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-30

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

---

## 1. Application Initialization

### 1.1 Initialization Sequence

**Critical Requirement**: Components must be initialized in dependency order.

**Sequence**:

```typescript
async function init(src?: string): Promise<void> {
  // 1. Foundation: Scene management
  this.sceneManager = new SceneManager(canvasId);
  await this.sceneManager.init();

  // 2. Animation system (depends on scene)
  this.animationController = new AnimationController(
    this.sceneManager.renderer,
    this.sceneManager.scene,
    this.sceneManager.camera,
    this.sceneManager.controls,
    this.sceneManager.postProcessing
  );

  // 3. Input handling (depends on scene and animation)
  this.inputHandler = new InputHandler(this.sceneManager, this.animationController);
  this.inputHandler.init();

  // 4. UI controls (depends on scene and animation)
  this.renderingControls = new RenderingControls(
    this.sceneManager.postProcessing,
    this.sceneManager.controls
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

  // 2. Check for .zgroup marker (Zarr dataset)
  try {
    const response = await fetch(src + '/.zgroup', { method: 'HEAD' });
    if (response.ok) {
      return false; // Valid Zarr, load directly
    }
  } catch {
    // Network error, can't determine
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
  // Critical: Scene initialization
  try {
    await this.sceneManager.init();
  } catch (sceneError) {
    // Scene failure is fatal
    console.error('Scene initialization failed:', sceneError);
    throw sceneError;
  }

  // Non-critical: Animation (continues on failure)
  try {
    this.animationController.startAnimation();
  } catch (animError) {
    console.warn('Animation start failed:', animError);
    // Continue without animation
  }

  // Non-critical: Data loading (show browser on failure)
  try {
    if (await this.shouldShowBrowser(src)) {
      this.showDatasetBrowser();
    } else {
      await this.loadDataset(src);
    }
  } catch (dataError) {
    console.error('Data loading failed:', dataError);
    showError('Failed to load dataset');
    // App continues without data
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

### InitializationConfig

```typescript
interface InitializationConfig {
  canvasId: string;
  defaultDataset?: string;
  enableDebug?: boolean;
  autoStart?: boolean;
}
```

---

## Changelog

- **v1.0.0** (2025-01-30): Initial specification
  - Component initialization in dependency order
  - Intelligent dataset detection (Zarr vs directory)
  - Error isolation (component failures don't cascade)
  - Proper resource cleanup in reverse order
  - URL parameter handling
  - Focus and visibility event handling
  - Debug interface exposure
  - Cross-component communication patterns
