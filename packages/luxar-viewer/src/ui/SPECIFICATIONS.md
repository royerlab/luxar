# luxar-viewer.ui - Technical Specification

**Version**: 1.4.0
**Last Updated**: 2025-12-17

## Purpose

The `luxar-viewer.ui` package provides user interface components for controlling visualization parameters, navigating nD datasets, monitoring performance, and debugging. It implements napari-inspired dimension sliders, comprehensive rendering controls, data loading monitors, and developer tools.

**Core Responsibility**: Provide intuitive, responsive UI for real-time visualization control with efficient DOM updates and minimal performance overhead.

**Related Specifications**:

- `luxar-viewer.data` - Data loading monitoring (see `../data/SPECIFICATIONS.md`)
- `luxar-viewer.scene` - Scene state integration (see `../scene/SPECIFICATIONS.md`)
- `luxar-viewer.rendering` - Post-processing integration (see `../rendering/SPECIFICATIONS.md`)

---

## Table of Contents

1. [Dimension Sliders](#1-dimension-sliders)
2. [Data Loading Monitor](#2-data-loading-monitor)
3. [Performance Monitor](#3-performance-monitor)
4. [Event Delegation Pattern](#4-event-delegation-pattern)
5. [Rendering Controls](#5-rendering-controls)
6. [Dataset Browser](#6-dataset-browser)
7. [Debug Console](#7-debug-console)
8. [Helper Utilities](#8-helper-utilities)
9. [Data Monitor Templates](#9-data-monitor-templates)
10. [Loading Advisor](#10-loading-advisor)
11. [Performance Timeline](#11-performance-timeline)
12. [Scene Graph Tree](#12-scene-graph-tree)
13. [Component Integration](#13-component-integration)

---

## 1. Dimension Sliders

### 1.1 Purpose

Napari-inspired UI for navigating non-displayed dimensions in nD datasets.

### 1.2 Slider Generation

**Input**: `SimpleDims` with metadata

**Output**: DOM elements for each non-displayed dimension

**Algorithm**:

```typescript
function createDimensionSliders(dims: SimpleDims): HTMLElement {
  const container = document.createElement('div');
  container.className = 'dimension-sliders';

  // Get non-displayed dimensions
  const displayedSet = new Set(dims.displayed);
  const nonDisplayed = [];
  for (let i = 0; i < dims.ndim; i++) {
    if (!displayedSet.has(i)) {
      nonDisplayed.push(i);
    }
  }

  // Create slider for each non-displayed dimension
  for (const dimIdx of nonDisplayed) {
    const meta = dims.metadata?.[dimIdx];
    const slider = createSlider(dimIdx, meta, dims.currentStep[dimIdx]);
    container.appendChild(slider);
  }

  return container;
}
```

### 1.3 Slider Update Algorithm

**Purpose**: Sync slider position with current dimension value.

```typescript
function updateSlider(dimIdx: number, value: number): void {
  const slider = sliderElements.get(dimIdx);
  if (!slider) return;

  const meta = dims.metadata[dimIdx];
  const [min, max] = meta.range;

  // Convert value to slider position (0-100)
  const percentage = ((value - min) / (max - min)) * 100;

  // Update slider element
  slider.value = percentage.toString();

  // Update label
  const label = slider.querySelector('.value-label');
  label.textContent = `${value.toFixed(2)} ${meta.unit}`;
}
```

---

## 2. Data Loading Monitor

### 2.1 Three-State UI

**States**:

1. **Hidden**: No UI visible (default)
2. **Mini**: Compact metrics bar
3. **Expanded**: Full panel with tabs

**Transition**:

```
Hidden → Mini → Expanded → Hidden (cycles with M key)
```

### 2.2 Event Monitoring

**Event Types**:

```typescript
type MonitorEvent =
  | { type: 'query'; cellCount: number; pointCount: number; duration: number }
  | { type: 'load'; rangeCount: number; bytesLoaded: number; duration: number }
  | { type: 'cache-hit'; rangeCount: number }
  | { type: 'cache-miss'; rangeCount: number }
  | { type: 'evict'; bytesEvicted: number }
  | { type: 'error'; message: string };
```

### 2.3 Statistics Aggregation

**Purpose**: Aggregate events into actionable metrics.

**Metrics**:

```typescript
interface LoadingMetrics {
  // Queries
  totalQueries: number;
  avgQueryTime: number;
  avgCellsPerQuery: number;
  avgPointsPerQuery: number;

  // Cache
  cacheHitRate: number; // hits / (hits + misses)
  totalCachedBytes: number;
  evictionCount: number;

  // Loading
  totalBytesLoaded: number;
  avgLoadTime: number;
  loadBandwidth: number; // bytes/sec

  // Errors
  errorCount: number;
  lastError?: string;
}
```

**Calculation**:

```typescript
function calculateMetrics(events: MonitorEvent[]): LoadingMetrics {
  // Filter by event type
  const queries = events.filter((e) => e.type === 'query');
  const loads = events.filter((e) => e.type === 'load');
  const hits = events.filter((e) => e.type === 'cache-hit');
  const misses = events.filter((e) => e.type === 'cache-miss');
  const errors = events.filter((e) => e.type === 'error');

  // Aggregate
  const totalQueries = queries.length;
  const avgQueryTime = average(queries.map((q) => q.duration));
  const avgCellsPerQuery = average(queries.map((q) => q.cellCount));
  const avgPointsPerQuery = average(queries.map((q) => q.pointCount));

  // Cache metrics
  const cacheHitRate = hits.length / (hits.length + misses.length);

  // Load metrics
  const totalBytesLoaded = sum(loads.map((l) => l.bytesLoaded));
  const avgLoadTime = average(loads.map((l) => l.duration));

  // Calculate bandwidth
  const timeWindow = 60000; // Last 60 seconds
  const recentLoads = loads.filter((l) => Date.now() - l.timestamp < timeWindow);
  const recentBytes = sum(recentLoads.map((l) => l.bytesLoaded));
  const loadBandwidth = (recentBytes / timeWindow) * 1000; // bytes/sec

  return {
    totalQueries,
    avgQueryTime,
    avgCellsPerQuery,
    avgPointsPerQuery,
    cacheHitRate,
    totalBytesLoaded,
    avgLoadTime,
    loadBandwidth,
    errorCount: errors.length,
    lastError: errors[errors.length - 1]?.message,
  };
}
```

### 2.4 Performance Timeline

**Purpose**: Real-time graph of loading performance over time.

**Data Structure**:

```typescript
interface TimelineDataPoint {
  timestamp: number; // Unix timestamp
  queryTime: number; // Query duration (ms)
  loadTime: number; // Load duration (ms)
  pointsLoaded: number; // Number of points
  bytesLoaded: number; // Bytes transferred
}
```

**Rendering**:

```typescript
// Canvas-based timeline with 60-second window
function renderTimeline(dataPoints: TimelineDataPoint[]): void {
  const canvas = getTimelineCanvas();
  const ctx = canvas.getContext('2d');

  const now = Date.now();
  const timeWindow = 60000; // 60 seconds

  // Filter to recent data
  const recent = dataPoints.filter((p) => now - p.timestamp < timeWindow);

  // Draw background
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw data points
  for (let i = 1; i < recent.length; i++) {
    const prev = recent[i - 1];
    const curr = recent[i];

    // Map time to x-axis
    const x1 = mapTimeToX(prev.timestamp, now, timeWindow, canvas.width);
    const x2 = mapTimeToX(curr.timestamp, now, timeWindow, canvas.width);

    // Map value to y-axis (query time)
    const y1 = mapValueToY(prev.queryTime, maxQueryTime, canvas.height);
    const y2 = mapValueToY(curr.queryTime, maxQueryTime, canvas.height);

    // Draw line
    ctx.strokeStyle = '#00a0ff';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}
```

---

## 3. Performance Monitor

### 3.1 Purpose

Provides real-time FPS and performance tracking using stats.js library.

### 3.2 Architecture

**Design Decisions**:

- Uses industry-standard stats.js library for accuracy
- Minimal performance overhead (only measures when visible)
- Positioned bottom-left to avoid blocking other UI
- Supports panel cycling (FPS → Frame Time → Memory)

### 3.3 Public API

```typescript
class PerformanceMonitor {
  // Frame timing
  begin(): void; // Call at start of render loop
  end(): void; // Call at end of render loop

  // Visibility control
  toggle(): void;
  show(): void;
  hide(): void;
  get visible(): boolean;

  // Panel cycling
  cyclePanels(): void; // FPS → MS → MB → FPS

  // Cleanup
  dispose(): void;
}
```

### 3.4 Integration Points

- **Scene Manager**: Called at render loop boundaries
- **Input Handler**: Toggle via 'P' key
- **Config System**: Z-index and positioning from config

### 3.5 Usage Example

```typescript
const monitor = new PerformanceMonitor();

function renderLoop() {
  monitor.begin(); // Start timing

  // ... render scene ...

  monitor.end(); // End timing and update display
  requestAnimationFrame(renderLoop);
}
```

---

## 4. Event Delegation Pattern

### 4.1 Purpose

Avoid inline event handlers and global namespace pollution by using data attributes and event delegation.

### 4.2 Implementation

**HTML Structure**:

```html
<!-- Instead of: onclick="globalFunction()" -->
<!-- Use data attributes: -->
<button data-action="expand">Expand</button>
<button data-action="hide">Hide</button>
```

**Event Handler**:

```typescript
class UIComponent {
  private container: HTMLElement;

  constructor() {
    // Single event listener for entire component
    this.container.addEventListener('click', this.handleClick.bind(this));
  }

  private handleClick(event: Event): void {
    const target = event.target as HTMLElement;
    const action = target.dataset.action;

    switch (action) {
      case 'expand':
        this.expand();
        break;
      case 'hide':
        this.hide();
        break;
      case 'toggle-tab':
        const tab = target.dataset.tab;
        this.setActiveTab(tab);
        break;
    }
  }
}
```

**Benefits**:

- No global functions
- Type-safe event handling
- Single listener for multiple elements
- Better security (no inline JS)
- Easier testing

---

## 5. Rendering Controls

### 5.1 Purpose

Comprehensive UI for controlling all rendering parameters in real-time using a custom GUI library (drop-in replacement for lil-gui).

**Features**:

- All effect controls (bloom, DOF, AO, tone mapping, AA, vignette, chromatic aberration, lens distortion, detector noise)
- Cinematic mode presets with intelligent toggle
- Settings persistence per-scene using localStorage
- FOV presets with realistic lens distortion
- Navigation controls (orbit, arcball, fly)
- **Adaptive resolution controls** with manual DPR adjustment

**Folder Icons**:

Top-level folders use emoji prefixes for visual identification:

- 🕹️ Navigation
- 🎥 Camera
- 💡 HDR
- 🔲 Anti-Aliasing
- ✨ Post-Processing Effects

### 5.2 Architecture

**Design Decisions**:

- Uses custom GUI library (see `./gui/SPECIFICATIONS.md`) for theme integration
- Settings bound directly to live objects (no manual syncing)
- Per-scene settings storage with unique keys
- Deferred rebuild pattern to batch multiple effect changes
- Auto-blur inputs to return focus to canvas
- Tooltips on all controls for discoverability
- **Modular setup functions**: Each control category in separate file (maintainability)

**Modular Structure**:

The `RenderingControls` class delegates GUI creation to specialized setup modules in `./rendering-controls/`:

1. **navigation-setup.ts**: Control type selector + orbit/fly mode settings
2. **camera-setup.ts**: FOV presets + manual FOV slider + clipping planes
3. **hdr-setup.ts**: Intensity (logarithmic slider) + tone mapping type
4. **anti-aliasing-setup.ts**: SSAA, FXAA, MSAA, SMAA with settings
5. **post-processing-setup.ts**: Bloom, noise, DoF, chromatic aberration, vignette, lens distortion, AO

Each setup function takes a `SetupContext` (dependencies + callbacks) and returns a `SetupResult` (controller references). This reduces the main file from 2,514 to 1,350 lines while maintaining identical functionality. See [`./rendering-controls/README.md`](./rendering-controls/README.md) and [`./rendering-controls/SPECIFICATIONS.md`](./rendering-controls/SPECIFICATIONS.md) for details.

**Key Control Categories**:

1. **Navigation Folder**: Control type selector + mode-specific settings
2. **Camera Folder**: FOV presets + manual FOV slider + clipping planes
3. **HDR Folder**: Intensity (logarithmic slider) + tone mapping type
4. **Anti-Aliasing Folder**: SSAA, FXAA, MSAA, SMAA with settings
5. **Post-Processing Effects Folder**: Bloom, noise, DOF, chromatic aberration, vignette, lens distortion, AO

### 5.2a Logarithmic Slider Pattern

**Purpose**: For parameters with wide range where perceptual differences are proportional to ratios rather than absolute differences (e.g., HDR intensity from 0.01 to 100).

**Problem**: Linear sliders make it difficult to select low values precisely when the range spans multiple orders of magnitude. The difference between 0.01 and 0.1 is perceptually significant but occupies only 0.09% of a linear slider's range.

**Solution**: Use a shadow property representing the log10 of the actual value, giving equal slider distance to each order of magnitude.

**Algorithm**:

```typescript
// Define range in log space
const logMin = Math.log10(0.01); // -2 (maps to slider left)
const logMax = Math.log10(100); // +2 (maps to slider right)

// Shadow object holds log value for lil-gui binding
const hdrLogValue = { log: Math.log10(this.settings.hdrMultiplier) };

// Create slider bound to log value
const hdrControl = hdrFolder
  .add(hdrLogValue, 'log', logMin, logMax, 0.01)
  .name('Intensity')
  .onChange((logValue: number) => {
    // Convert log to actual value
    const actualValue = Math.pow(10, logValue);
    this.settings.hdrMultiplier = actualValue;
    this.sceneManager.updateHDRMultiplier(actualValue);
    this.saveSettings();
    this.triggerAnimation();
  });

// Update controller reference for external sync
this.controllers.hdrMultiplier = hdrControl;
this.hdrLogValue = hdrLogValue; // Store for sync updates
```

**Sync from External State**:

When `syncCurrentState()` is called, the log value must be updated to match the actual hdrMultiplier:

```typescript
syncCurrentState(): void {
  // ... other sync code ...

  // Sync logarithmic HDR slider
  if (this.hdrLogValue) {
    this.hdrLogValue.log = Math.log10(this.settings.hdrMultiplier);
    this.controllers.hdrMultiplier?.updateDisplay();
  }
}
```

**Display Value Formatting**:

Since lil-gui doesn't support custom value formatters, we override the controller's `updateDisplay` method to show the actual intensity value instead of the log value:

```typescript
// Helper to format the actual intensity value for display
const formatIntensity = (logValue: number): string => {
  const actual = Math.pow(10, logValue);
  if (actual >= 10) return actual.toFixed(0); // "15"
  if (actual >= 1) return actual.toFixed(1); // "1.5"
  if (actual >= 0.1) return actual.toFixed(2); // "0.15"
  return actual.toFixed(3); // "0.015"
};

// Override updateDisplay to show actual intensity value
const originalUpdateDisplay = hdrControl.updateDisplay.bind(hdrControl);
(hdrControl as any).updateDisplay = () => {
  originalUpdateDisplay();
  // After lil-gui updates, override the input value with formatted intensity
  const input = (hdrControl as any).$input as HTMLInputElement;
  if (input) {
    input.value = formatIntensity(this.hdrLogValue.log);
  }
  return hdrControl;
};
```

This ensures users see meaningful values like "0.08" or "1.5" instead of "-1.12" or "0.18". The override approach is necessary because lil-gui's `updateDisplay()` always sets `$input.value` from the bound property.

**Slider Position to Value Mapping**:

- **Left edge** (log=-2): displays "0.01" (very dim)
- **Center** (log=0): displays "1.0" (neutral)
- **Right edge** (log=+2): displays "100" (very bright)

**Perceptual Benefits**:

- Equal slider distance for equal perceptual change
- Easy to select 0.1, 1.0, 10.0 (each one slider-width apart)
- Fine control at both low and high ends of range
- Display shows actual multiplier value (not logarithm)

### 5.3 Public API

```typescript
class RenderingControls {
  // Setup
  constructor(postProcessing: PostProcessingManager, sceneManager: SceneManager);
  setAnimationController(controller: AnimationController): void;
  setSceneId(zarrUrl: string, sceneName?: string): void;

  // Visibility
  show(): void;
  hide(): void;
  toggle(): void;
  isVisible(): boolean;

  // State management
  syncCurrentState(): void; // Sync from scene manager
  toggleCinematicMode(): void; // Smart toggle with majority vote

  // Cleanup
  dispose(): void;
}
```

### 5.4 Settings Persistence

**Storage Key Generation**:

```typescript
function generateSettingsKey(sceneId: string): string {
  return `luxar-rendering-settings-${sceneId}`;
}
```

**Serialization**:

```typescript
function serializeSettings(settings: RenderingSettings): string {
  return JSON.stringify(settings);
}

function deserializeSettings(json: string): RenderingSettings | null {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}
```

**Scene ID Format**: `{zarr_url_sanitized}_{scene_name}`

- Example: `http___localhost_8000_data_zarr_scene1`

### 5.5 Cinematic Mode Algorithm

**Purpose**: Toggle multiple effects intelligently using majority vote, with snapshot-restore
to preserve user customizations.

**Effects controlled**: ACES tone mapping, detector noise, vignette, chromatic lens distortion, FOV/lens preset.

**Algorithm**:

```typescript
function toggleCinematicMode(): void {
  // 1. Check current state of all cinematic effects (4 signals)
  const effects = [
    detectorNoiseEnabled,
    vignetteEnabled,
    chromaticLensDistortionEnabled,
    toneMapping === 'ACES',
  ];

  // 2. Majority vote (>= 50% enabled = turn all off, < 50% = turn all on)
  const shouldEnableAll = enabledCount < effects.length / 2;

  if (shouldEnableAll) {
    // 3a. ENABLE: snapshot current settings, then apply cinematic values
    cinematicSnapshot = snapshotCurrentSettings();
    Object.assign(settings, buildCinematicValues()); // ACES, noise, vignette, 35mm lens
  } else {
    // 3b. DISABLE: restore each setting from snapshot (dirty-check)
    // Only restore if user hasn't manually changed it while cinematic was on
    for (key of snapshotKeys) {
      if (settings[key] === cinematicValues[key]) {
        settings[key] = cinematicSnapshot[key]; // Untouched → restore
      }
      // else: user changed it → keep their value
    }
    cinematicSnapshot = null;
  }

  // 4. Batch apply using deferred rebuild
  postProcessing.startDeferRebuild();
  postProcessing.setToneMapping(settings.toneMapping);
  // ... apply noise, vignette, lens distortion ...
  postProcessing.endDeferRebuild();
}
```

### 5.6 Integration Points

- **PostProcessingManager**: All effect settings
- **SceneManager**: Camera FOV, clipping planes, control type, auto-rotate
- **AnimationController**: Trigger re-renders on changes
- **Input Manager**: Keyboard shortcut 'R' to toggle, 'C' for cinematic mode
- **Config System**: Default values, FOV presets, lens distortion presets

### 5.7 Usage Example

```typescript
const controls = new RenderingControls(postProcessing, sceneManager);
controls.setAnimationController(animationController);
controls.setSceneId(zarrUrl, 'my_scene');

// Toggle visibility
controls.toggle(); // Press 'R' in app

// Cinematic mode
controls.toggleCinematicMode(); // Press 'C' in app
```

### 5.8 Adaptive Resolution Controls

**Purpose**: Controls for managing rendering resolution (DPR) for performance/quality tradeoff.

**UI Behavior**:

The adaptive resolution section displays different controls based on whether adaptive mode is enabled or disabled:

| Adaptive Mode | Controls Shown                                  |
| ------------- | ----------------------------------------------- |
| **ON**        | Enable toggle, Read-only DPR (%), Read-only FPS |
| **OFF**       | Enable toggle, Manual DPR slider (25%-100%)     |

**Implementation**:

```typescript
function setAdaptiveDPRManager(manager: AdaptiveDPRManager): void {
  // Store reference for state queries
  this.adaptiveDPRManager = manager;

  // Create controls
  const adaptiveToggle = folder
    .add(settings, 'adaptiveResolutionEnabled')
    .name('Enable Adaptive')
    .onChange((enabled) => {
      manager.setEnabled(enabled);
      updateVisibility(enabled);
    });

  // Manual DPR control (visible when adaptive OFF)
  const manualDPRControl = folder
    .add(manualDPRSettings, 'dpr', 0.25, 1.0, 0.05)
    .name('Resolution')
    .onChange((value) => {
      manager.setManualDPR(value * manager.getNativeDPR());
    });

  // Read-only displays (visible when adaptive ON)
  const dprDisplay = folder.add(readOnlyState, 'dpr').name('DPR (%)');
  dprDisplay.$input.disabled = true;

  const fpsDisplay = folder.add(readOnlyState, 'fps').name('FPS');
  fpsDisplay.$input.disabled = true;
}
```

**Rationale**:

- When adaptive mode is ON, the system automatically adjusts DPR based on performance, so manual control would conflict
- When adaptive mode is OFF, users may want to manually reduce resolution for better performance
- FPS display is only meaningful when adaptive mode is tracking performance
- DPR display shows current value as percentage of native resolution

---

## 6. Dataset Browser

### 6.1 Purpose

File browser UI for navigating directories and selecting Zarr datasets from various server types (WebDAV, S3, nginx, etc.).

### 6.2 Architecture

**Design Decisions**:

- Uses DirectoryNavigator from data package for server-agnostic browsing
- Breadcrumb navigation for easy path traversal
- Visual indicators for Zarr datasets vs regular files
- Highlights currently loaded dataset
- Manual path entry fallback for servers without listing
- Modal overlay with backdrop blur

**Navigation Strategies**:

1. **WebDAV**: PROPFIND requests for directory listings
2. **HTML Parsing**: Parse `<a>` tags from directory index pages
3. **Index File**: Read `.directory-index.json` if present
4. **Manual Entry**: Text input fallback

### 6.3 Public API

```typescript
interface DatasetBrowserConfig {
  container: HTMLElement;
  onDatasetSelect: (path: string) => void;
  onClose?: () => void;
}

class DatasetBrowser {
  constructor(config: DatasetBrowserConfig);

  // Navigation
  private navigate(path: string): Promise<void>;

  // Visibility
  show(): void;
  hide(): void;
  close(): void; // Hide + cleanup
}
```

### 6.4 URL Path Extraction

**Algorithm**:

```typescript
function extractBaseUrl(url: string): string {
  // Parse URL
  const parsed = new URL(url);

  // If ends with .zarr, go up one level
  let pathname = parsed.pathname;
  if (pathname.endsWith('.zarr') || pathname.endsWith('.zarr/')) {
    const parts = pathname.split('/').filter(Boolean);
    parts.pop();
    pathname = '/' + parts.join('/') + '/';
  }

  return parsed.origin + pathname;
}

function extractPath(url: string): string {
  // Parse URL
  const parsed = new URL(url);
  const pathname = parsed.pathname;

  // Extract dataset name if it's a .zarr
  if (pathname.includes('.zarr')) {
    const parts = pathname.split('/').filter(Boolean);
    const zarrIndex = parts.findIndex((p) => p.endsWith('.zarr'));
    if (zarrIndex >= 0) {
      return parts.slice(0, zarrIndex + 1).join('/');
    }
  }

  return '';
}
```

### 6.5 Integration Points

- **DirectoryNavigator**: Server communication and listing
- **SceneLoader**: Dataset loading after selection
- **Input Manager**: Keyboard shortcut 'O' to open browser
- **URL Parameters**: Initial path from `?src=` parameter

### 6.6 Usage Example

```typescript
const browser = new DatasetBrowser({
  container: document.body,
  onDatasetSelect: (path) => {
    // Load the selected dataset
    sceneLoader.loadScene(path);
  },
  onClose: () => {
    // Browser closed without selection
  },
});

browser.show();
```

---

## 7. Debug Console

### 7.1 Purpose

In-app developer console that captures all browser console output in a persistent ring buffer and displays it in an accessible UI overlay.

**Features**:

- Captures console.log, warn, error, info, debug from app start
- Ring buffer prevents memory overflow (10,000 messages max)
- Real-time filtering by text
- Message type indicators with color coding
- Copy all messages to clipboard
- Draggable and resizable panel
- Auto-scroll option

### 7.2 Architecture

**Design Decisions**:

- Uses global console interceptor that starts early (before app init)
- Ring buffer implementation to prevent memory growth
- Listener pattern for real-time updates when visible
- Lazy rendering: only renders buffered messages when shown
- DOM manipulation optimized with fragment building

**Key Components**:

1. **Console Interceptor**: Global singleton that intercepts all console calls
2. **Ring Buffer**: Fixed-size circular buffer for message storage
3. **Message Formatting**: Type-aware formatting (strings, objects, errors)
4. **Filter System**: Client-side text filtering
5. **Status Bar**: Message count and filter status

### 7.3 Public API

```typescript
class DebugConsole {
  constructor(); // Auto-registers with global interceptor

  // Visibility
  show(): void; // Renders all buffered messages
  hide(): void;
  toggle(): void;
  getIsVisible(): boolean;

  // Actions
  clear(): void; // Clears buffer and UI
  private copyToClipboard(): void; // Internal

  // Cleanup
  dispose(): void;
}
```

### 7.4 Console Interceptor Integration

**Architecture**:

```typescript
// Early initialization (src/utils/console-interceptor.ts)
class ConsoleInterceptor {
  private buffer: RingBuffer<BufferedMessage>;
  private listeners: Set<(message: BufferedMessage) => void>;

  intercept(): void {
    const original = console.log;
    console.log = (...args) => {
      // Store in buffer
      this.buffer.push({ type: 'log', args, timestamp: new Date() });

      // Notify listeners (DebugConsole)
      this.listeners.forEach((fn) => fn(this.buffer.last()));

      // Call original
      original.apply(console, args);
    };
  }

  addListener(fn: (message: BufferedMessage) => void): void;
  removeListener(fn: (message: BufferedMessage) => void): void;
  getBufferedMessages(): BufferedMessage[];
  clearBuffer(): void;
}

export const consoleInterceptor = new ConsoleInterceptor();
consoleInterceptor.intercept(); // Starts immediately on import
```

### 7.5 Message Rendering

**Algorithm**:

```typescript
function renderMessage(message: BufferedMessage): void {
  const messageEl = document.createElement('div');
  messageEl.className = `console-message console-message-${message.type}`;

  // Format timestamp
  const timestamp = message.timestamp.toLocaleTimeString('en-US', {
    hour12: false,
    fractionalSecondDigits: 3,
  });

  // Build HTML with type-aware formatting
  let html = `<span class="timestamp">${timestamp}</span>`;

  message.args.forEach((arg) => {
    html += formatArgWithStyle(arg); // Colors by type
  });

  messageEl.innerHTML = html;

  // Add stack trace for errors
  if (message.stack && message.type === 'error') {
    const stackEl = document.createElement('div');
    stackEl.className = 'console-message-stack';
    stackEl.textContent = message.stack;
    messageEl.appendChild(stackEl);
  }

  // Apply filter
  if (filter && !matchesFilter(message)) {
    messageEl.style.display = 'none';
  }

  contentArea.appendChild(messageEl);
}

function formatArgWithStyle(arg: any): string {
  if (typeof arg === 'string') {
    return `<span class="console-message-string">"${escapeHtml(arg)}"</span>`;
  }
  if (typeof arg === 'number') {
    return `<span class="console-message-number">${arg}</span>`;
  }
  if (typeof arg === 'boolean') {
    return `<span class="console-message-boolean">${arg}</span>`;
  }
  if (typeof arg === 'object') {
    return `<span class="console-message-object">${escapeHtml(JSON.stringify(arg, null, 2))}</span>`;
  }
  if (arg === undefined) {
    return `<span class="console-message-undefined">undefined</span>`;
  }
  return escapeHtml(String(arg));
}
```

### 7.6 Integration Points

- **Console Interceptor**: Global singleton for message capture
- **Input Manager**: Keyboard shortcut 'Ctrl+L' to toggle
- **Config System**: Panel size, position, styling from config
- **Ring Buffer**: Fixed-size buffer to prevent memory growth

### 7.7 Usage Example

```typescript
// Debug console auto-initializes on import
import { DebugConsole } from './ui/debug-console';

const debugConsole = new DebugConsole();

// All console output is automatically captured:
console.log('Hello, world!'); // Captured
console.warn('Warning message'); // Captured
console.error('Error occurred'); // Captured with stack trace

// Toggle visibility
debugConsole.toggle(); // Press Ctrl+L in app
```

---

## 8. Helper Utilities

### 8.1 Purpose

Utility functions for common UI operations: loading indicators, error display, help overlay, and cleanup.

### 8.2 Architecture

**Design Decisions**:

- Pure functions for simple operations
- Singleton pattern for spinners (only one CSS keyframe injection)
- Auto-dismiss for error messages with manual dismiss option
- Collapsible help categories for progressive disclosure
- Consistent styling across all helpers

### 8.3 Public API

```typescript
// Loading indicators
function showLoadingIndicator(): HTMLElement;
function hideLoadingIndicator(): void;

// Error display
function showError(message: string): void; // Auto-dismiss after timeout
function clearError(): void;

// Help overlay
function showHelpOverlay(): void;
function hideHelpOverlay(): void;

// Cleanup
function cleanupUI(): void; // Removes all UI elements
```

### 8.4 Error Display Algorithm

**Features**:

- Clear error message with icon
- Helpful guidance for common issues
- Keyboard shortcuts displayed
- Click-to-dismiss
- Auto-dismiss after timeout
- Prevents multiple error dialogs (removes existing)

**Structure**:

```typescript
function showError(message: string): void {
  // 1. Remove existing error (prevent duplicates)
  const existing = document.getElementById('error-message');
  if (existing) existing.remove();

  // 2. Create error dialog
  const errorDiv = document.createElement('div');
  errorDiv.id = 'error-message';

  // 3. Build content structure
  errorDiv.innerHTML = `
    <div class="header">
      <div class="icon">⚠️</div>
      <div class="title">Unable to Load Dataset</div>
    </div>
    <div class="message">${message}</div>
    <div class="guidance">
      <div class="guidance-title">💡 How to Load a Dataset:</div>
      <div class="guidance-list">
        <div>1. Add dataset to URL: ?src=/path/to/dataset.zarr</div>
        <div>2. Or browse: Press O key</div>
        <div>3. Dataset format: Zarr-format with points, lines, and other primitives</div>
        <div>4. Need help? Press H for shortcuts</div>
      </div>
    </div>
    <div class="dismiss">Click anywhere or press Escape to dismiss</div>
  `;

  // 4. Add event listeners
  errorDiv.addEventListener('click', () => errorDiv.remove());
  errorDiv.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') {
      errorDiv.remove();
    }
  });

  // 5. Auto-dismiss after timeout
  setTimeout(() => {
    if (errorDiv.parentNode) errorDiv.remove();
  }, config.ui.timings.errorAutoDismissMs);

  // 6. Add to DOM
  document.body.appendChild(errorDiv);
}
```

### 8.5 Help Overlay Structure

**Collapsible Categories**:

1. **Basic Controls**: Mouse, keyboard basics (expanded by default)
2. **Fly Mode Controls**: WASD, arrows, roll (collapsed)
3. **nD Navigation**: Dimension selection and navigation (collapsed)
4. **Advanced Settings**: Rendering, performance, debug (collapsed)
5. **Tips**: Best practices and hints (collapsed)

**Expansion Algorithm**:

```typescript
function createCollapsibleCategory(category: HelpCategory): HTMLElement {
  const header = document.createElement('div');
  header.className = 'category-header';

  const arrow = document.createElement('span');
  arrow.textContent = '▶';
  arrow.style.transform = category.expanded ? 'rotate(90deg)' : 'rotate(0deg)';

  const title = document.createElement('span');
  title.textContent = category.title;

  header.appendChild(arrow);
  header.appendChild(title);

  const content = document.createElement('div');
  content.className = 'category-content';
  content.style.display = category.expanded ? 'block' : 'none';

  // Populate content
  category.items.forEach((item) => {
    const itemDiv = document.createElement('div');
    itemDiv.textContent = item;
    content.appendChild(itemDiv);
  });

  // Toggle on click
  header.addEventListener('click', () => {
    const isExpanded = content.style.display !== 'none';
    content.style.display = isExpanded ? 'none' : 'block';
    arrow.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(90deg)';
  });

  const container = document.createElement('div');
  container.appendChild(header);
  container.appendChild(content);
  return container;
}
```

### 8.6 Integration Points

- **Scene Loader**: Loading indicators during scene load
- **Input Manager**: Help overlay via 'H' key
- **Error Handling**: Global error display for dataset loading failures
- **Config System**: Timeouts, z-index, styling from config

### 8.7 Usage Example

```typescript
// Show loading
const loader = showLoadingIndicator();

try {
  await loadDataset(url);
  hideLoadingIndicator();
} catch (error) {
  hideLoadingIndicator();
  showError(`Failed to load dataset: ${error.message}`);
}

// Show help
showHelpOverlay(); // Press 'H' in app

// Cleanup on app shutdown
cleanupUI();
```

---

## 9. Data Monitor Templates

### 9.1 Purpose

HTML template functions for generating UI content in the Data Loading Monitor. Extracts presentation logic from business logic.

### 9.2 Architecture

**Design Decisions**:

- Pure functions for each template component
- Centralized formatting (bytes, numbers, percentages)
- Consistent color system from config
- Reusable metric cards, progress bars, stat grids
- Separation of concerns: templates don't contain logic

### 9.3 Public API

```typescript
// Components
function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  color?: string,
  size?: 'small' | 'medium' | 'large'
): string;

function renderProgressBar(
  percent: number,
  color?: string,
  label?: string,
  height?: number
): string;

function renderStatGrid(
  stats: Array<{ label: string; value: string | number; color?: string }>
): string;

function renderLoaderItem(path: string, metrics: LoaderMetrics): string;

// Tab content
function renderOverviewContent(stats: GlobalStats, cacheMetrics: CacheMetrics): string;
function renderCacheContent(stats: GlobalStats, cacheMetrics: CacheMetrics): string;
function renderInsightsContent(recommendations: Recommendation[]): string;

// Sub-components
function renderSecondaryMetrics(
  memory: { used: number; limit: number },
  querySpeed: { avgTime: number; perSec: number },
  loadSpeed: { count: number; bandwidth: number }
): string;

function renderRecommendation(rec: Recommendation): string;
```

### 9.4 Metric Card Template

**Algorithm**:

```typescript
// CSS color classes for semantic colors
type SemanticColor = 'success' | 'warning' | 'error' | 'info' | 'muted' | 'dimmed' | 'primary';

function getColorClass(color: SemanticColor): string {
  return `luxar-color--${color}`;
}

function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  colorClass: string = '', // CSS class for color (e.g., 'luxar-color--success')
  size: 'small' | 'medium' | 'large' = 'medium'
): string {
  return `
    <div class="luxar-metric-card luxar-metric-card--${size}">
      ${title ? `<div class="luxar-metric-card__title">${title}</div>` : ''}
      <div class="luxar-metric-card__value luxar-metric-card__value--${size} ${colorClass}">
        ${value}
      </div>
      ${subtitle ? `<div class="luxar-metric-card__subtitle">${subtitle}</div>` : ''}
    </div>
  `;
}
```

### 9.5 Progress Bar Algorithm

**Color Determination**:

```typescript
function getProgressColorClass(percent: number): string {
  if (percent <= 60) return getColorClass('success');
  if (percent <= 80) return getColorClass('warning');
  return getColorClass('error');
}

function renderProgressBar(
  percent: number,
  colorClass?: string, // CSS color class
  label?: string,
  height: number = 4
): string {
  const barColorClass = colorClass || getProgressColorClass(percent);
  // Using CSS custom properties for dynamic height
  const trackStyle = `style="--bar-height: ${height}px; height: var(--bar-height); border-radius: calc(var(--bar-height) / 2);"`;
  const fillStyle = `style="width: ${Math.min(100, percent)}%; border-radius: calc(var(--bar-height, 4px) / 2);"`;

  return `
    <div class="luxar-progress-bar__container">
      <div class="luxar-progress-bar__track" ${trackStyle}>
        <div class="luxar-progress-bar__fill ${barColorClass}" ${fillStyle}></div>
      </div>
      ${label ? `<div class="luxar-progress-bar__label">${label}</div>` : ''}
    </div>
  `;
}
```

### 9.6 Formatting Utilities

**Number Formatting**:

```typescript
function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + 'KB';
  return bytes.toFixed(0) + 'B';
}
```

### 9.7 Integration Points

- **Data Loading Monitor**: Primary consumer of all templates
- **CSS Theme System**: Colors from CSS variables (--luxar-success, --luxar-warning, etc.)
- **Type Definitions**: Uses types from data-monitor-types.ts
- **Styling**: All styles in src/styles/components/data-loading-monitor.css

### 9.8 Usage Example

```typescript
import { renderMetricCard, renderProgressBar, renderStatGrid } from './data-monitor-templates';

// Large metric card with CSS color class
const html = renderMetricCard(
  'VISIBLE POINTS',
  formatNumber(stats.visiblePoints),
  `${percent}% of ${formatNumber(stats.datasetSize)} total`,
  getColorClass('success'), // CSS class for color
  'large'
);

// Progress bar with auto color class
const progressHtml = renderProgressBar(
  cacheMemoryPercent,
  undefined, // Auto color class based on percent
  'Cache Usage'
);

// Stat grid with CSS color classes
const gridHtml = renderStatGrid([
  { label: 'Queries', value: '1.2K', colorClass: getColorClass('info') },
  { label: 'Hit Rate', value: '95%', colorClass: getColorClass('success') },
  { label: 'Evictions', value: '12', colorClass: getColorClass('warning') },
]);
```

---

## 10. Loading Advisor

### 10.1 Purpose

AI-powered recommendation engine that analyzes loading performance and provides actionable optimization suggestions.

### 10.2 Architecture

**Design Decisions**:

- Rule-based system with severity levels (error, warning, info)
- Analyzes global statistics to detect patterns
- Provides specific, actionable recommendations
- Prioritizes recommendations by severity
- Stateless analysis (no historical tracking)

**Recommendation Categories**:

1. **Performance Issues**: Slow queries, high latency
2. **Memory Issues**: Cache pressure, high eviction rate
3. **Network Issues**: Low bandwidth, packet loss indicators
4. **Configuration Issues**: Suboptimal settings
5. **Informational**: Best practices and tips

### 10.3 Public API

```typescript
interface Recommendation {
  severity: 'error' | 'warning' | 'info';
  title: string;
  message: string;
  suggestion?: string;
}

class LoadingAdvisor {
  analyze(stats: GlobalStats): Recommendation[];
}
```

### 10.4 Analysis Algorithm

**Rule Evaluation**:

```typescript
function analyze(stats: GlobalStats): Recommendation[] {
  const recommendations: Recommendation[] = [];

  // Rule 1: Slow queries
  if (stats.avgQueryTime > 100) {
    recommendations.push({
      severity: stats.avgQueryTime > 500 ? 'error' : 'warning',
      title: 'Slow Query Performance',
      message: `Average query time is ${stats.avgQueryTime.toFixed(0)}ms`,
      suggestion: 'Consider reducing spatial index resolution or dataset complexity',
    });
  }

  // Rule 2: Low cache hit rate
  if (stats.cacheHitRate < 0.5) {
    recommendations.push({
      severity: 'warning',
      title: 'Low Cache Hit Rate',
      message: `Cache hit rate is ${(stats.cacheHitRate * 100).toFixed(0)}%`,
      suggestion: 'Increase cache size or reduce camera movement speed',
    });
  }

  // Rule 3: High eviction rate
  const evictionsPerMin = stats.totalEvictions / (stats.uptime / 60000);
  if (evictionsPerMin > 10) {
    recommendations.push({
      severity: 'warning',
      title: 'Frequent Cache Evictions',
      message: `${evictionsPerMin.toFixed(0)} evictions per minute`,
      suggestion: 'Increase cache memory limit in config',
    });
  }

  // Rule 4: No data loaded
  if (stats.totalLoads === 0 && stats.uptime > 5000) {
    recommendations.push({
      severity: 'error',
      title: 'No Data Loaded',
      message: 'No data chunks have been loaded',
      suggestion: 'Check spatial index configuration and dataset format',
    });
  }

  // Rule 5: Good performance (informational)
  if (stats.avgQueryTime < 50 && stats.cacheHitRate > 0.8 && stats.totalQueries > 10) {
    recommendations.push({
      severity: 'info',
      title: 'Excellent Performance',
      message: 'Loading system is operating efficiently',
    });
  }

  return recommendations;
}
```

### 10.5 Integration Points

- **Data Loading Monitor**: Displays recommendations in Insights tab
- **Global Stats**: Source of analysis data
- **Template System**: Uses renderRecommendation() for display

### 10.6 Usage Example

```typescript
import { LoadingAdvisor } from './components/loading-advisor';

const advisor = new LoadingAdvisor();
const recommendations = advisor.analyze(globalStats);

// Display in monitor
recommendations.forEach((rec) => {
  const html = renderRecommendation(rec);
  insightsTab.innerHTML += html;
});
```

---

## 11. Performance Timeline

### 11.1 Purpose

Real-time canvas-based graph showing loading performance metrics over time.

### 11.2 Architecture

**Design Decisions**:

- HTML5 Canvas for efficient rendering
- 60-second rolling window
- Multiple metric tracks (query time, load time, points loaded)
- Auto-scaling based on data range
- Throttled updates (10 FPS) to reduce CPU usage

**Data Structure**:

```typescript
interface TimelineDataPoint {
  timestamp: number; // Unix timestamp in ms
  queryTime: number; // Query duration in ms
  loadTime: number; // Load duration in ms
  pointsLoaded: number; // Number of points
  bytesLoaded: number; // Bytes transferred
}
```

### 11.3 Public API

```typescript
class PerformanceTimeline {
  constructor(canvas: HTMLCanvasElement);

  // Data management
  addDataPoint(point: TimelineDataPoint): void;
  clear(): void;

  // Rendering
  render(): void; // Throttled to 10 FPS

  // Cleanup
  dispose(): void;
}
```

### 11.4 Rendering Algorithm

**Canvas Drawing**:

```typescript
function render(): void {
  const ctx = canvas.getContext('2d');
  const now = Date.now();
  const timeWindow = 60000; // 60 seconds

  // 1. Filter to visible window
  const visible = dataPoints.filter((p) => now - p.timestamp < timeWindow);

  // 2. Clear canvas
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 3. Calculate scales
  const maxQueryTime = Math.max(...visible.map((p) => p.queryTime));
  const scaleX = canvas.width / timeWindow;
  const scaleY = canvas.height / maxQueryTime;

  // 4. Draw grid lines (10 horizontal lines)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  for (let i = 0; i <= 10; i++) {
    const y = (i / 10) * canvas.height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  // 5. Draw query time line
  ctx.strokeStyle = '#00a0ff';
  ctx.lineWidth = 2;
  ctx.beginPath();

  visible.forEach((point, index) => {
    const x = canvas.width - (now - point.timestamp) * scaleX;
    const y = canvas.height - point.queryTime * scaleY;

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();

  // 6. Draw axis labels
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.font = '10px monospace';
  ctx.fillText(`${maxQueryTime.toFixed(0)}ms`, 5, 10);
  ctx.fillText('0ms', 5, canvas.height - 5);
}
```

### 11.5 Integration Points

- **Data Loading Monitor**: Displayed in Performance tab
- **Monitor Events**: Receives data from monitor event stream
- **RequestAnimationFrame**: Throttled rendering loop

### 11.6 Usage Example

```typescript
const canvas = document.getElementById('timeline-canvas') as HTMLCanvasElement;
const timeline = new PerformanceTimeline(canvas);

// Add data from monitor events
monitor.addEventListener('query', (event) => {
  timeline.addDataPoint({
    timestamp: Date.now(),
    queryTime: event.duration,
    loadTime: 0,
    pointsLoaded: event.pointCount,
    bytesLoaded: 0,
  });
});

// Cleanup
timeline.dispose();
```

---

## 12. Scene Graph Tree

### 12.1 Purpose

Collapsible tree view showing the hierarchical scene structure with per-node statistics. Replaces the simple "Active Loaders" list in the Overview tab.

### 12.2 Architecture

**Design Decisions**:

- Recursive tree rendering for arbitrary depth
- Click-to-expand/collapse for nodes with children
- Per-node type icons and color coding
- Statistics display (points count, segment count) per node
- Integration with SceneLoader's scene graph structure

**Node Types**:

| Type   | Icon | Color   | Stats Displayed |
| ------ | ---- | ------- | --------------- |
| scene  | 🌐   | Info    | -               |
| group  | 📁   | Muted   | -               |
| points | ⚬    | Success | Point count     |
| lines  | ╱    | Warning | Segment count   |
| mesh   | ⬡    | Purple  | -               |

### 12.3 Data Structures

**SceneGraphNode Interface**:

```typescript
interface SceneGraphNode {
  path: string; // Path in zarr store
  name: string; // Display name (last path component)
  type: SceneGraphNodeType; // 'scene' | 'group' | 'points' | 'lines' | 'mesh'
  pointCount?: number; // For points nodes
  segmentCount?: number; // For lines nodes
  vertexCount?: number; // For lines nodes
  isLoading?: boolean; // Loading indicator
  hasSpatialIndex?: boolean; // Whether node has spatial index
  children: SceneGraphNode[];
}

interface SceneGraphState {
  root: SceneGraphNode | null;
  totalNodes: number;
  pointsNodes: number;
  linesNodes: number;
  totalPoints: number;
  totalSegments: number;
}
```

### 12.4 Tree Rendering Algorithm

**Recursive Rendering**:

```typescript
function renderSceneGraphNode(
  node: SceneGraphNode,
  expandedNodes: Set<string>,
  depth: number = 0
): string {
  const hasChildren = node.children.length > 0;
  const isExpanded = expandedNodes.has(node.path);
  const indent = depth * 16; // px per level

  // Build node stats text
  let statsText = '';
  if (node.type === 'points' && node.pointCount !== undefined) {
    statsText = formatNumber(node.pointCount) + ' pts';
  } else if (node.type === 'lines' && node.segmentCount !== undefined) {
    statsText = formatNumber(node.segmentCount) + ' seg';
  }

  // Toggle icon
  const toggleIcon = hasChildren ? (isExpanded ? '▼' : '▶') : '•';

  // Render node HTML
  let html = `
    <div style="padding: 2px 0;">
      <div style="display: flex; align-items: center; padding-left: ${indent}px;">
        <span data-action="toggleNode" data-node-path="${node.path}"
              style="width: 16px; cursor: pointer;">${toggleIcon}</span>
        <span style="font-size: 11px;">${getNodeTypeIcon(node.type)}</span>
        <span style="color: ${getNodeTypeColor(node.type)};">${node.name}</span>
        ${statsText ? `<span style="color: dimmed;">${statsText}</span>` : ''}
        ${node.isLoading ? '<span>⏳</span>' : ''}
      </div>
      ${
        isExpanded && hasChildren
          ? node.children
              .map((child) => renderSceneGraphNode(child, expandedNodes, depth + 1))
              .join('')
          : ''
      }
    </div>
  `;

  return html;
}
```

### 12.5 SceneLoader Integration

**Conversion from SceneNode**:

```typescript
// In SceneLoader
private convertToSceneGraphNode(node: SceneNode): SceneGraphNode {
  const name = node.path === '/'
    ? 'Scene'
    : node.path.split('/').filter(Boolean).pop() || node.path;

  const graphNode: SceneGraphNode = {
    path: node.path,
    name,
    type: node.type as SceneGraphNodeType,
    children: [],
    hasSpatialIndex: node.hasSpatialIndex,
  };

  // Add type-specific stats
  if (node.type === 'points') {
    graphNode.pointCount = node.attrs.n_points;
  } else if (node.type === 'lines') {
    graphNode.segmentCount = node.attrs.n_segments;
    graphNode.vertexCount = node.attrs.n_vertices;
  }

  // Recursively convert children
  if (node.children) {
    graphNode.children = node.children.map(child =>
      this.convertToSceneGraphNode(child)
    );
  }

  return graphNode;
}
```

**Integration Point**:

```typescript
// After scene load completes
const sceneGraphRoot = this.convertToSceneGraphNode(sceneGraph);
monitor.setSceneGraph(sceneGraphRoot);
```

### 12.6 State Management

**Expansion State**:

- Tracked in `DataLoadingMonitor` via `expandedNodes: Set<string>`
- Node paths used as keys for O(1) lookup
- Root node (`/`) expanded by default
- State persists during monitor visibility toggle

**Toggle Handler**:

```typescript
// In DataLoadingMonitor.handleUIEvent()
case 'toggleNode': {
  const nodePath = target.dataset.nodePath;
  if (nodePath) {
    this.toggleNodeExpansion(nodePath);
  }
  break;
}

public toggleNodeExpansion(path: string): void {
  if (this.expandedNodes.has(path)) {
    this.expandedNodes.delete(path);
  } else {
    this.expandedNodes.add(path);
  }
  this.updateUI();
}
```

### 12.7 Integration Points

- **SceneLoader**: Provides scene graph data after loading
- **DataLoadingMonitor**: Manages tree state and rendering
- **data-monitor-templates.ts**: Contains renderSceneGraphTree() function
- **data-monitor-types.ts**: Defines SceneGraphNode and SceneGraphState types

### 12.8 Usage Example

```typescript
// SceneLoader sets scene graph after loading
const sceneGraphRoot = this.convertToSceneGraphNode(sceneGraph);
monitor.setSceneGraph(sceneGraphRoot);

// Monitor renders tree in Overview tab
const treeHtml = renderSceneGraphTree(this.sceneGraphState, this.expandedNodes);

// User clicks toggle
// → handleUIEvent() → toggleNodeExpansion() → updateUI()
```

---

## 13. Component Integration

### 13.1 Lifecycle Management

**Initialization Order**:

```typescript
// 1. Core UI components (independent)
const performanceMonitor = new PerformanceMonitor();
const debugConsole = new DebugConsole();

// 2. Scene-dependent components (require scene manager)
const renderingControls = new RenderingControls(postProcessing, sceneManager);
renderingControls.setAnimationController(animationController);

// 3. Data-dependent components (require loaders)
const dataMonitor = new DataLoadingMonitor(container);
dataMonitor.connectLoader(path, loader);

// 4. On-demand components (created when needed)
const datasetBrowser = new DatasetBrowser({
  container: document.body,
  onDatasetSelect: (path) => loadScene(path),
});
```

**Disposal Order** (reverse of initialization):

```typescript
// 1. On-demand components
datasetBrowser?.close();

// 2. Data-dependent components
dataMonitor.dispose();

// 3. Scene-dependent components
renderingControls.dispose();

// 4. Core UI components
debugConsole.dispose();
performanceMonitor.dispose();

// 5. Global cleanup
cleanupUI();
```

### 12.2 Cross-Component Communication

**Event Flow**:

```
User Input → Input Manager
  ↓
Keyboard Shortcuts
  ↓
UI Component Actions (show/hide/toggle)
  ↓
State Changes
  ↓
Scene Manager / Post-Processing Manager
  ↓
Render Update
  ↓
Animation Controller
```

**State Synchronization**:

```typescript
// Rendering controls sync from scene on show
renderingControls.show(); // Calls syncCurrentState() internally

// Data monitor receives events from loaders
loader.dispatchEvent(new MonitorEvent('query', { ... }));
monitor.handleEvent(event); // Updates UI

// Performance monitor measures render loop
function renderLoop() {
  performanceMonitor.begin();
  // ... render ...
  performanceMonitor.end();
}
```

### 12.3 Global Keyboard Shortcuts

**Priority Handling** (highest to lowest):

1. **Escape**: Close active panel (highest priority)
2. **Modal Dialogs**: Error messages, help overlay
3. **Panel Toggles**: R (rendering), P (performance), M (data monitor), O (dataset browser)
4. **Debug Tools**: Ctrl+L (debug console)
5. **Scene Actions**: H (help), V (view mode), F (focus), C (cinematic)
6. **Camera Controls**: WASD, arrows, Space (lowest priority)

**Conflict Resolution**:

- Input elements inside panels prevent propagation
- Escape always closes topmost panel
- Rendering controls panel absorbs focus during interaction
- Auto-blur inputs to return focus to canvas

---

## 14. Resolution Indicator

### 14.1 Purpose

A subtle visual indicator that appears when the adaptive DPR system has reduced rendering resolution to maintain smooth frame rates. Provides feedback to users that the system has scaled resolution to maintain target FPS.

### 14.2 Architecture

**Design Decisions**:

- Lazy creation (DOM element created only when first shown)
- Auto-hide after 4 seconds per activation
- Single show per mode activation (prevents visual spam)
- Reset mechanism when exiting reduced resolution mode
- Displays target FPS from config (not hardcoded)

**Component Lifecycle**:

```
Reduced Resolution Mode Activated
  ↓
show(dpr) called
  ↓
[hasShownForCurrentMode?] → Yes → Return (don't show again)
  ↓ No
Create/show element with animation
  ↓
Set hasShownForCurrentMode = true
  ↓
Start 4-second auto-hide timer
  ↓
Timer expires → hide()
  ↓
Reduced Resolution Mode Deactivated
  ↓
reset() called → hasShownForCurrentMode = false
```

### 14.3 Public API

```typescript
class ResolutionIndicator {
  /**
   * Set the target FPS for display purposes.
   * @param fps - Target FPS from config
   */
  setTargetFPS(fps: number): void;

  /**
   * Show the indicator with optional DPR value.
   * Only shows once per reduced resolution mode activation.
   * Auto-hides after 4 seconds.
   * @param dpr - Current device pixel ratio to display
   */
  show(dpr?: number): void;

  /**
   * Hide the indicator with animation.
   */
  hide(): void;

  /**
   * Reset state when exiting reduced resolution mode.
   * Allows indicator to show again on next activation.
   */
  reset(): void;

  /**
   * Check if the indicator is currently visible.
   */
  getIsVisible(): boolean;

  /**
   * Clean up resources.
   */
  dispose(): void;
}
```

### 14.4 DOM Structure

```html
<div class="luxar-resolution-indicator luxar-resolution-indicator--hidden">
  <span class="luxar-resolution-indicator__icon">&#x21C5;</span>
  <span class="luxar-resolution-indicator__text">Resolution Scaled to 75% to maintain 60 fps</span>
</div>
```

### 14.5 Integration with AdaptiveDPRManager

```typescript
// In app.ts initialization
this.resolutionIndicator = new ResolutionIndicator();
// Display target FPS rounded up from maxFPS (58 → 60) since targetFPS (55) is a hysteresis threshold
const displayTargetFPS = Math.ceil(config.adaptiveDPR.maxFPS / 5) * 5;
this.resolutionIndicator.setTargetFPS(displayTargetFPS);
this.adaptiveDPRManager.setOnDPRChangeCallback((dpr, isReducedResolution) => {
  if (isReducedResolution) {
    this.resolutionIndicator.show(dpr);
  } else {
    // Exiting reduced resolution mode - reset indicator state
    this.resolutionIndicator.reset();
  }
});
```

**Display FPS Calculation**:

The config uses hysteresis thresholds (targetFPS=55, maxFPS=58) to prevent rapid toggling, but users expect to see the actual target (60 fps). We compute the display value by rounding `maxFPS` up to the nearest 5:

- `maxFPS = 58` → displays "60 fps"
- `maxFPS = 48` → displays "50 fps"
- `maxFPS = 118` → displays "120 fps" (high refresh rate)

**Rationale**:

- The indicator shows briefly (4 seconds) when entering reduced resolution mode to inform the user
- It doesn't stay visible permanently because that would be distracting
- The reset() call when exiting ensures the indicator can show again if reduced resolution mode reactivates
- The "once per activation" behavior prevents rapid show/hide cycling during FPS fluctuations
- The display FPS is derived from config, not hardcoded, adapting to different refresh rate targets

---

## Data Structures

### DimensionSliders

```typescript
interface DimensionSliders {
  container: HTMLElement;
  sliders: Map<number, SliderElement>;
  dims: SimpleDims | null;

  create(dims: SimpleDims): void;
  update(dimIdx: number, value: number): void;
  show(): void;
  hide(): void;
  dispose(): void;
}

interface SliderElement {
  container: HTMLElement;
  input: HTMLInputElement;
  label: HTMLElement;
  dimIndex: number;
}
```

### DataLoadingMonitor

```typescript
interface DataLoadingMonitor {
  state: 'hidden' | 'mini' | 'expanded';
  events: MonitorEvent[];
  connectedLoaders: Map<string, Loader>;

  cycleState(): void; // hidden → mini → expanded → hidden
  connectLoader(path: string, loader: Loader): void;
  disconnectLoader(path: string): void;
  getMetrics(): LoadingMetrics;
  dispose(): void;
}
```

### RenderingControls

```typescript
interface RenderingSettings {
  // Navigation
  controlType: 'orbit' | 'arcball' | 'fly';
  autoRotate: boolean;
  autoRotateSpeed: number;
  flyMovementSpeed: number;
  flyRotationSpeed: number;
  flyInertialMode: boolean;
  flyDamping: number;
  flyRotationDamping: number;

  // Camera
  fov: number;
  fovPreset: string;
  near: number;
  far: number;

  // HDR
  hdrMultiplier: number;
  toneMapping: string;

  // Anti-Aliasing
  ssaaEnabled: boolean;
  ssaaMultiplier: number;
  fxaaEnabled: boolean;
  msaaEnabled: boolean;
  msaaSamples: number;
  smaaEnabled: boolean;
  // Note: smaaThreshold and smaaSearchSteps exist in config but are not exposed
  // in the UI because pmndrs/postprocessing SMAAEffect only supports preset-based
  // configuration (LOW/MEDIUM/HIGH/ULTRA). The UI shows only an on/off toggle.

  // Effects
  bloomStrength: number;
  bloomRadius: number;
  bloomThreshold: number;
  bloomLevels: number;

  detectorNoiseEnabled: boolean;
  detectorNoiseReadoutSigma: number;
  detectorNoisePhotonGain: number;
  detectorNoiseFpnSigma: number;

  dofEnabled: boolean;
  dofFocus: number;
  dofStrength: number;

  vignetteEnabled: boolean;
  vignetteDarkness: number;
  vignetteOffset: number;

  chromaticLensDistortionEnabled: boolean;
  chromaticLensDistortionX: number;
  chromaticLensDistortionY: number;
  chromaticLensDispersion: number;
  chromaticLensPrincipalPointX: number;
  chromaticLensPrincipalPointY: number;
  chromaticLensFocalLengthX: number;
  chromaticLensFocalLengthY: number;
  chromaticLensSkew: number;

  aoEnabled: boolean;
  aoQuality: 'low' | 'medium' | 'high' | 'ultra';
}
```

### DebugConsole

```typescript
interface ConsoleMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  formatted: string;
  stack?: string;
}

interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  stack?: string;
}
```

---

## Changelog

- **v1.5.0** (2025-12-28): Resolution indicator terminology fix
  - **RENAMED**: "Low Power Indicator" → "Resolution Indicator" (terminology was misleading)
  - **RENAMED**: `LowPowerIndicator` class → `ResolutionIndicator`
  - **RENAMED**: `isLowPowerMode` → `isReducedResolution` throughout codebase
  - **CHANGED**: Message now shows "Resolution Scaled to X% to maintain Y fps" (Y from config)
  - **ADDED**: `setTargetFPS(fps)` method to set target FPS from config
  - **CHANGED**: Icon from ⚡ to ⇅ (scaling arrows, more accurate)

- **v1.4.0** (2025-12-17): Adaptive resolution and reduced resolution mode
  - **ADDED**: Section 5.8 "Adaptive Resolution Controls" - Manual DPR control when adaptive mode is OFF
  - **ADDED**: Section 14 "Resolution Indicator" - New component with auto-hide behavior
  - **ADDED**: Folder emoji prefixes for visual identification (🕹️ Navigation, 🎥 Camera, etc.)
  - **CHANGED**: Updated to use custom GUI library (lil-gui replacement) for theme integration
  - **IMPROVED**: Manual DPR slider when adaptive resolution is disabled
  - **IMPROVED**: FPS display hidden when adaptive mode is OFF (no longer shows "0")
  - **IMPROVED**: Resolution indicator auto-hides after 4 seconds per activation

- **v1.3.0** (2025-12-10): Logarithmic HDR intensity slider
  - **ADDED**: Section 5.2a "Logarithmic Slider Pattern" documenting the shadow property approach
  - **IMPROVED**: HDR intensity slider now uses logarithmic scale (0.01 to 100)
  - **IMPROVED**: Equal slider distance for equal perceptual change (orders of magnitude)
  - **IMPROVED**: Display shows actual intensity value (e.g., "0.08") instead of log value (e.g., "-1.12")
  - **IMPROVED**: Tooltip explains the logarithmic scale behavior
  - Fine control at both low and high ends of intensity range

- **v1.2.0** (2025-12-09): Scene Graph Tree and monitoring enhancements
  - **Added**: Scene Graph Tree component (Section 12)
    - Collapsible tree view showing hierarchical scene structure
    - Per-node type icons and color coding (scene, group, points, lines, mesh)
    - Per-node statistics display (point count, segment count)
    - Click-to-expand/collapse interaction
    - Integration with SceneLoader's scene graph structure
    - SceneGraphNode and SceneGraphState type definitions
  - **Added**: Cache monitoring integration
    - L1/L2 cache statistics display with eviction counters
    - Network I/O tracking (bytes transferred, request count, bandwidth)
    - Clear L1/L2 buttons in Cache tab
    - CacheStatsProvider interface for loose coupling
  - **Enhanced**: Overview tab now shows scene graph tree instead of simple loader list
  - **Enhanced**: Secondary metrics bar shows Network I/O instead of Load Speed

- **v1.1.0** (2025-12-09): Comprehensive specification update
  - **Added**: Rendering Controls (2,153 lines) - Complete documentation of all effect controls, cinematic mode, settings persistence, lil-gui integration, FOV presets, chromatic lens distortion, navigation controls
  - **Added**: Dataset Browser (632 lines) - Directory navigation, path extraction algorithm, server detection strategies, manual entry fallback
  - **Added**: Debug Console (762 lines) - Console interceptor integration, ring buffer architecture, message rendering, filtering, copy-to-clipboard
  - **Added**: Helper Utilities (502 lines) - Loading indicators, error display with guidance, collapsible help overlay, cleanup functions
  - **Added**: Performance Monitor (~200 lines) - stats.js integration, FPS/frame time tracking, panel cycling, visibility control
  - **Added**: Data Monitor Templates (~315 lines) - HTML template functions, metric cards, progress bars, stat grids, recommendation rendering
  - **Added**: Loading Advisor - Rule-based recommendation engine, performance analysis, actionable suggestions
  - **Added**: Performance Timeline - Canvas-based real-time graphing, 60-second rolling window, auto-scaling
  - **Enhanced**: Component Integration section - Lifecycle management, cross-component communication, keyboard shortcut priority handling
  - **Enhanced**: Data Structures section - Added RenderingSettings, DebugConsole types, BufferedMessage
  - Version incremented to 1.1.0 for major content additions

- **v1.0.0** (2025-01-30): Initial specification
  - Napari-inspired dimension sliders for nD navigation
  - Three-state data loading monitor (hidden/mini/expanded)
  - Event-driven monitoring with aggregated metrics
  - Performance timeline with canvas rendering
  - FPS calculation with rolling average
  - GPU memory estimation from geometry attributes
  - Event delegation pattern (no global functions)
