# luxar-viewer.ui - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2025-12-09

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
12. [Component Integration](#12-component-integration)

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

Comprehensive UI for controlling all rendering parameters in real-time using lil-gui library.

**Features**:

- All effect controls (bloom, DOF, AO, tone mapping, AA, vignette, chromatic aberration, lens distortion, detector noise)
- Cinematic mode presets with intelligent toggle
- Settings persistence per-scene using localStorage
- FOV presets with realistic lens distortion
- Navigation controls (orbit, arcball, fly)

### 5.2 Architecture

**Design Decisions**:

- Uses lil-gui library for immediate-mode GUI
- Settings bound directly to live objects (no manual syncing)
- Per-scene settings storage with unique keys
- Deferred rebuild pattern to batch multiple effect changes
- Auto-blur inputs to return focus to canvas
- Tooltips on all controls for discoverability

**Key Components**:

1. **Navigation Folder**: Control type selector + mode-specific settings
2. **Camera Folder**: FOV presets + manual FOV slider + clipping planes
3. **HDR Folder**: Intensity + tone mapping type
4. **Anti-Aliasing Folder**: SSAA, FXAA, MSAA, SMAA with settings
5. **Post-Processing Effects Folder**: Bloom, noise, DOF, chromatic aberration, vignette, lens distortion, AO

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

**Purpose**: Toggle multiple effects intelligently using majority vote.

**Algorithm**:

```typescript
function toggleCinematicMode(): void {
  // 1. Check current state of all cinematic effects
  const effects = [
    detectorNoiseEnabled,
    vignetteEnabled,
    chromaticAberrationEnabled,
    lensDistortionEnabled,
  ];

  // 2. Count enabled effects
  const enabledCount = effects.filter(Boolean).length;

  // 3. Majority vote (>= 50% enabled = turn all off, < 50% = turn all on)
  const shouldEnableAll = enabledCount < effects.length / 2;

  // 4. Apply to all effects
  detectorNoiseEnabled = shouldEnableAll;
  vignetteEnabled = shouldEnableAll;
  chromaticAberrationEnabled = shouldEnableAll;
  lensDistortionEnabled = shouldEnableAll;

  // 5. Set cinematic parameters if enabling
  if (shouldEnableAll) {
    detectorNoiseReadoutSigma = 0.015;
    detectorNoisePhotonGain = 0.008;
    detectorNoiseFpnSigma = 0.003;
  }

  // 6. Switch FOV: 35mm for cinematic, 50mm Normal for regular
  fov = shouldEnableAll ? 63 : 47; // degrees
  fovPreset = shouldEnableAll ? '35mm' : '50mm Normal';

  // 7. Apply corresponding lens distortion preset
  if (shouldEnableAll) {
    applyLensPreset('35mm'); // Barrel distortion
  } else {
    applyLensPreset('50mm Normal'); // No distortion
  }

  // 8. Batch apply using deferred rebuild
  postProcessing.startDeferRebuild();
  // ... apply all effects ...
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
        <div>3. Dataset format: Zarr-format with point cloud data</div>
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
function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  color: string = MonitorColors.primaryText,
  size: 'small' | 'medium' | 'large' = 'medium'
): string {
  const fontSize = {
    large: '32px',
    medium: '24px',
    small: '16px',
  }[size];

  const padding = {
    large: '16px',
    medium: '12px',
    small: '8px',
  }[size];

  return `
    <div style="
      background: ${MonitorColors.sectionBg};
      padding: ${padding};
      border-radius: ${size === 'large' ? '8px' : '6px'};
    ">
      ${title ? `<div style="font-size: 10px; color: ${MonitorColors.muted}; margin-bottom: 4px;">${title}</div>` : ''}
      <div style="font-size: ${fontSize}; font-weight: bold; color: ${color};">
        ${value}
      </div>
      ${subtitle ? `<div style="font-size: 10px; color: ${MonitorColors.dimmed}; margin-top: 4px;">${subtitle}</div>` : ''}
    </div>
  `;
}
```

### 9.5 Progress Bar Algorithm

**Color Determination**:

```typescript
function getProgressColor(percent: number): string {
  if (percent <= 60) return MonitorColors.success;
  if (percent <= 80) return MonitorColors.warning;
  return MonitorColors.error;
}

function renderProgressBar(
  percent: number,
  color?: string,
  label?: string,
  height: number = 4
): string {
  const barColor = color || getProgressColor(percent);

  return `
    <div style="margin-top: ${label ? 6 : 0}px;">
      <div style="
        height: ${height}px;
        background: rgba(255,255,255,0.1);
        border-radius: ${height / 2}px;
      ">
        <div style="
          height: 100%;
          background: ${barColor};
          width: ${Math.min(100, percent)}%;
          border-radius: ${height / 2}px;
        "></div>
      </div>
      ${label ? `<div style="font-size: 9px; color: ${MonitorColors.dimmed}; margin-top: 2px;">${label}</div>` : ''}
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
- **Config System**: Colors, spacing, border radius from config
- **Type Definitions**: Uses types from data-monitor-types.ts

### 9.8 Usage Example

```typescript
import { renderMetricCard, renderProgressBar, renderStatGrid } from './data-monitor-templates';

// Large metric card
const html = renderMetricCard(
  'VISIBLE POINTS',
  formatNumber(stats.visiblePoints),
  `${percent}% of ${formatNumber(stats.datasetSize)} total`,
  MonitorColors.success,
  'large'
);

// Progress bar with auto color
const progressHtml = renderProgressBar(
  cacheMemoryPercent,
  undefined, // Auto color based on percent
  'Cache Usage'
);

// Stat grid
const gridHtml = renderStatGrid([
  { label: 'Queries', value: '1.2K', color: MonitorColors.info },
  { label: 'Hit Rate', value: '95%', color: MonitorColors.success },
  { label: 'Evictions', value: '12', color: MonitorColors.warning },
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

## 12. Component Integration

### 12.1 Lifecycle Management

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
  smaaThreshold: number;
  smaaSearchSteps: number;

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

  chromaticAberrationEnabled: boolean;
  chromaticAberrationStrength: number;

  vignetteEnabled: boolean;
  vignetteDarkness: number;
  vignetteOffset: number;

  lensDistortionEnabled: boolean;
  lensDistortionX: number;
  lensDistortionY: number;
  lensPrincipalPointX: number;
  lensPrincipalPointY: number;
  lensFocalLengthX: number;
  lensFocalLengthY: number;
  lensSkew: number;

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

- **v1.1.0** (2025-12-09): Comprehensive specification update
  - **Added**: Rendering Controls (2,153 lines) - Complete documentation of all effect controls, cinematic mode, settings persistence, lil-gui integration, FOV presets, lens distortion, navigation controls
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
