# Luxar UI Package

> Beautiful, responsive user interface components for nD visualization and control

## Overview

The Luxar UI package provides a comprehensive set of user interface components for controlling visualization parameters, navigating nD datasets, monitoring performance, and debugging. Built with modern web technologies, it offers an intuitive interface inspired by scientific visualization tools like napari.

### Key Features

- **Dimension Sliders**: Napari-inspired sliders for nD navigation
- **Rendering Controls**: Real-time adjustment of visual parameters
- **Dataset Browser**: Navigate and load Zarr datasets
- **Performance Monitor**: FPS and GPU memory tracking
- **Data Loading Monitor**: Real-time monitoring of spatial index-based data loading
- **Debug Console**: In-app console for development
- **Helper Overlays**: Keyboard shortcuts and tips
- **Recording Panel**: Screenshot and video capture with turntable mode
- **Scale Bar**: Physical scale bar overlay using dimension units
- **Responsive Design**: Mobile and desktop friendly

### Package Architecture

```
ui/
├── dimension-sliders.ts         # nD navigation controls
├── rendering-controls.ts        # Visual parameter adjustments (main class)
├── rendering-controls/          # Modular setup functions
│   ├── types.ts                 # Shared types (SetupContext, SetupResult)
│   ├── navigation-setup.ts      # Navigation controls (orbit, arcball, fly)
│   ├── camera-setup.ts          # Camera settings (FOV, clipping)
│   ├── hdr-setup.ts             # HDR intensity & tone mapping
│   ├── anti-aliasing-setup.ts   # AA techniques (FXAA, SMAA, MSAA, SSAA)
│   └── post-processing-setup.ts # Effects (bloom, noise, DoF, etc.)
├── dataset-browser.ts           # Zarr dataset navigation
├── performance-monitor.ts       # FPS and performance stats
├── data-loading-monitor.ts      # Data loading performance monitoring
├── data-monitor-types.ts        # Type definitions for monitoring
├── debug-console.ts             # Developer console overlay
├── helpers.ts                   # Help overlays and tooltips
├── recording-panel.ts           # Screenshot and video capture panel
├── components/                  # Reusable UI components
│   ├── loading-advisor.ts       # Smart recommendations engine
│   ├── hierarchical-timing-panel.ts  # Hierarchical timing breakdown UI
│   └── scale-bar.ts            # Physical scale bar overlay
└── README.md                    # This documentation
```

---

## Components

### 1. Dimension Sliders

Beautiful napari-inspired sliders for navigating through nD datasets.

**Features:**

- Smooth slider controls for each dimension
- Real-time value display with units
- Color-coded dimension indicators
- Keyboard navigation support
- Auto-hide for 3D-only datasets
- Step-based navigation for discrete dimensions
- **Animation controls** for automated playback through dimension ranges

**UI Structure:**

```typescript
class DimensionSliders {
  // Creates slider panel with:
  - Header with collapse toggle
  - Slider for each non-displayed dimension
  - Value display with unit labels
  - Keyboard hints
}
```

**Usage:**

```typescript
const sliders = new DimensionSliders();
sliders.setDimensions(sceneDims);

// Listen for changes
sliders.on('dimensionChanged', (dim, value) => {
  updateVisualization(dim, value);
});
```

**Animation Controls:**

Each dimension slider includes animation controls for automated playback through dimension ranges (e.g., time-lapse, z-stack traversal):

```
┌─────────────────────────────────┐
│ Time                    50ms    │
│ [▶] ◀━━━━━━━●━━━━━━━━━━▶       │
└─────────────────────────────────┘
```

**Control Elements:**

- **Play Button** (`▶`/`⏸`): Compact button to the left of slider
  - Left-click: Toggle animation play/pause
  - Right-click: Open settings context menu (Napari-style)

**Context Menu Settings** (right-click play button):

- **Speed Section**: Set target animation speed
  - Presets: 1, 2, 5, 10, 15, 30, 60 FPS
  - Radio button selection with current speed marked

- **Loop Mode Section**: Choose loop behavior
  - `Once`: Play once and stop at end
  - `Loop`: Loop continuously from start to end
  - `Bounce`: Ping-pong back and forth
  - Radio button selection with current mode marked

**Animation Features:**

- FPS-based throttling with actual FPS measurement
- Independent animation state per dimension
- Forward/backward direction control
- Discrete and continuous dimension support
- Performance monitoring with warnings if target FPS not achieved
- Event system for UI synchronization

**Keyboard Shortcuts:**

- `K`: Toggle play/pause for selected dimension
- `Home`: Jump to dimension start
- `End`: Jump to dimension end
- `Shift+↑`: Increase animation speed
- `Shift+↓`: Decrease animation speed

**Integration:**

```typescript
import { DimensionAnimationManager } from '../scene/dimension-animation-manager';

// Animation manager is automatically created by InputHandler
// and connected to dimension sliders

// Programmatic control
const animManager = inputHandler.getAnimationManager();
animManager.play(dimIndex, { targetFPS: 30, loopMode: 'loop' });
animManager.pause(dimIndex);
animManager.setTargetFPS(dimIndex, 60);

// Listen for animation events
animManager.addEventListener('play', (e) => {
  console.log(`Dimension ${e.dimIndex} started animating`);
});

animManager.addEventListener('complete', (e) => {
  console.log(`Dimension ${e.dimIndex} animation complete`);
});
```

**Implementation Details:**

- Uses `DimensionAnimationManager` for FPS-based animation logic
- Integrates with `AnimationController` for frame updates
- Updates dimension values via `SceneDimsManager`
- CSS styling in `styles/components/dimension-sliders.css`
- See `scene/dimension-animation-manager.ts` for core animation logic

### 2. Rendering Controls

Comprehensive controls for adjusting rendering parameters in real-time.

**Control Categories:**

- **Visual Effects**: Bloom, tone mapping, noise, DOF, vignette, chromatic aberration, lens distortion
- **HDR**: Intensity control with **logarithmic slider** (0.01-100, equal slider distance per order of magnitude)
- **Anti-Aliasing**: FXAA, SMAA (HIGH preset), MSAA, SSAA toggles
- **Performance**: Quality presets, FPS targets
- **Camera**: FOV presets (28mm-135mm equivalents), manual FOV control, clipping plane adjustments
- **Materials**: Opacity, gamma, blending modes

**Logarithmic HDR Intensity Slider:**

The HDR intensity slider uses a logarithmic scale to provide equal perceptual control across its entire range:

- **Left edge**: 0.01 (very dim)
- **Center**: 1.0 (neutral)
- **Right edge**: 100 (very bright)

This ensures that moving the slider the same distance always produces the same perceived change in brightness, regardless of the current value.

**Panel Layout:**

```
Rendering Controls
├── Navigation
│   ├── Control Type (Orbit|Arcball|Fly)
│   ├── Orbit Controls
│   │   ├── Auto Rotate □
│   │   └── Rotation Speed (slider)
│   └── Fly Controls
│       ├── Movement Speed (slider)
│       ├── Rotation Speed (slider)
│       └── Inertial Mode □
├── Camera
│   ├── FOV Preset (dropdown: 28mm/35mm/50mm/85mm/135mm/Custom)
│   ├── Field of View (slider, 10°-200°)
│   └── Clipping Planes ▼
│       ├── Near Plane (slider, 0.001-10.0)
│       ├── Far Plane (slider, 10-10000)
│       └── Auto Adjust (button)
├── HDR
│   ├── Intensity (logarithmic slider, 0.01-100)
│   └── Tone Mapping (type selector)
├── Anti-Aliasing
│   ├── SSAA □ (with resolution multiplier)
│   ├── FXAA □
│   ├── SMAA □ (toggle only - preset-based)
│   └── MSAA □ (with samples)
└── Post-Processing Effects
    ├── Bloom
    │   ├── Threshold (slider)
    │   ├── Strength (slider)
    │   ├── Radius (slider)
    │   └── Mipmap Levels (slider)
    ├── Noise
    │   ├── Enabled □
    │   ├── Intensity (slider)
    │   ├── Film Grain Mode □
    │   └── Blend Mode (selector)
    ├── Depth of Field
    │   ├── Enabled □
    │   ├── Focus Distance (slider)
    │   └── Strength (slider)
    ├── Chromatic Aberration
    │   ├── Enabled □
    │   └── Strength (slider)
    ├── Ambient Occlusion
    │   ├── Enabled □
    │   └── Quality (selector)
    └── Vignette
        ├── Enabled □
        ├── Darkness (slider)
        └── Offset (slider)
```

### 3. Dataset Browser

File browser for navigating and loading Zarr datasets from servers.

**Features:**

- Server-agnostic navigation (WebDAV, S3, nginx)
- Zarr dataset detection and highlighting
- Breadcrumb navigation
- File size and date display
- Search and filter capabilities
- Recent datasets history

**Interface:**

```typescript
class DatasetBrowser {
  // Navigation
  navigate(path: string): Promise<void>;

  // Selection
  onSelect(callback: (dataset: string) => void);

  // History
  addRecent(path: string): void;
  clearRecent(): void;
}
```

### 4. Performance Monitor

Real-time performance statistics overlay.

**Metrics Displayed:**

- FPS (current, average, min/max)
- Frame time (ms)
- GPU memory usage
- Point count
- Draw calls
- Render resolution

**Visualization:**

```
┌─────────────────┐
│ FPS: 60 (58-60) │
│ Frame: 16.7ms   │
│ Points: 1.2M    │
│ Memory: 128MB   │
└─────────────────┘
```

### 5. Data Loading Monitor

Advanced real-time monitoring system for spatial index-based data loading with performance analytics and smart recommendations.

**Features:**

- **Three-State UI**: Cycles through hidden → mini → expanded views
- **Event-Driven Monitoring**: Tracks queries, loads, cache hits/misses, evictions
- **Cache Analytics**: Detailed cache memory usage and hit rate statistics
- **Performance Timeline**: Real-time graphing of loading performance
- **Smart Recommendations**: AI-powered suggestions for optimization
- **Multi-Loader Support**: Monitors multiple data loaders simultaneously

**UI States:**

1. **Hidden**: No UI visible (default on startup)
2. **Mini View**: Compact metrics bar showing key statistics
3. **Expanded View**: Full panel with tabs for detailed analytics

**Keyboard Shortcut:** `M` to cycle through states

**Architecture:**

```typescript
// Managed by DataMonitorManager singleton
DataMonitorManager.getInstance().createMonitor(id, container).connectLoader(path, loader);

// Monitor receives events from loaders
loader.addEventListener((event: MonitorEvent) => {
  // Event types: query, load, cache-hit, cache-miss, evict, error
});
```

**Tabs in Expanded View:**

1. **Overview Tab**
   - Global statistics (total points, memory, loaders)
   - Active loader list with real-time status
   - Key performance indicators
   - Recent events stream

2. **Cache Tab**
   - Cache memory usage with visual gauge
   - Hit rate statistics (global and recent)
   - Cached ranges count and average size
   - Cache performance metrics (hits/sec, misses/sec)
   - Memory breakdown by data type

3. **Performance Tab**
   - Real-time performance timeline graph
   - Query latency tracking
   - Load time analysis
   - Bandwidth utilization
   - Historical trends

4. **Insights Tab**
   - Smart recommendations from LoadingAdvisor
   - Severity-based alerts (error, warning, info)
   - Actionable optimization suggestions
   - Performance bottleneck detection

**Performance Optimizations:**

- **Event Cleanup**: Automatically removes events older than 5 minutes
- **Rate Caching**: Calculations cached for 1 second to reduce CPU usage
- **Timeline Batching**: Uses requestAnimationFrame with 10 FPS throttling
- **Efficient Updates**: Only re-renders changed UI sections

**Integration with Scene Loading:**

```typescript
// Automatic integration - no setup required!
// SceneLoader creates monitor on construction
const loader = new SceneLoader(config);

// Monitor automatically connects to spatial index loaders
// Tracks all data loading operations
// Disconnects old loaders when loading new scenes
```

**Event Types Monitored:**

- `query`: Spatial index query with cell/point counts
- `load`: Data chunk loaded with memory usage
- `cache-hit`: Data served from cache
- `cache-miss`: Cache miss requiring network load
- `evict`: Cache eviction due to memory pressure
- `error`: Loading errors with details

**Usage Example:**

```typescript
import { cycleDataMonitor } from '../data';

// Toggle monitor with M key
document.addEventListener('keydown', (e) => {
  if (!e.ctrlKey && !e.metaKey && e.key === 'm') {
    cycleDataMonitor(); // Cycles: hidden → mini → expanded → hidden
  }
});
```

**Configuration Options:**

```typescript
interface MonitorConfig {
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  theme: 'dark' | 'light';
  defaultView: 'compact' | 'detailed';
  updateInterval: number; // UI update frequency (ms)
  maxEvents: number; // Maximum events to store
  showSpatialGrid: boolean; // Show spatial index visualization
  showTimeline: boolean; // Show performance timeline
  showRecommendations: boolean; // Show LoadingAdvisor tips
  autoExpand: boolean; // Auto-expand on warnings
  enableProfiling: boolean; // Enable detailed profiling
  sampleRate: number; // Event sampling rate (0-1)
}
```

**Key Benefits:**

- **Zero Configuration**: Automatic integration with scene loading
- **Real-time Insights**: Immediate visibility into loading performance
- **Smart Recommendations**: AI-powered optimization suggestions
- **Performance Optimized**: Minimal overhead with intelligent batching
- **Developer Friendly**: Clean API and comprehensive documentation

### 6. Debug Console

In-app developer console for debugging and diagnostics.

**Features:**

- Console output capture and display
- Command execution
- Network request logging
- Error stack traces
- Performance profiling
- Local storage inspection

**Keyboard Shortcut:** `Ctrl+L` to toggle

**Interface:**

```typescript
class DebugConsole {
  toggle(): void;
  clear(): void;
  log(message: string, level?: 'info' | 'warn' | 'error'): void;
  executeCommand(command: string): void;
}
```

### 7. Helper Overlays

Context-sensitive help and keyboard shortcuts.

**Components:**

- Keyboard shortcut reference
- Control mode indicators
- Tooltip system
- First-time user hints
- Loading indicators

### 8. Recording Panel

Screenshot and video capture panel with multiple export options.

**Features:**

- Screenshot export (PNG, WebP, JPEG) with configurable resolution
- Video recording (WebM) with turntable rotation mode
- Transparent background support for compositing
- Dimension slider synchronization during recording
- Resolution multiplier for high-DPI exports

**Keyboard Shortcuts:**

- `T`: Toggle recording panel
- `G`: Quick screenshot

**CSS:** `styles/components/recording-panel.css`

### 9. Scale Bar

Physical scale bar overlay that automatically computes width from camera distance, FOV, and dimension units.

**Features:**

- Automatic width computation from camera/viewport parameters
- Physical unit display using scene dimension metadata
- Supports all Luxar physical units (nm, um, mm, cm, m, etc.)
- Auto-hide when no unit information is available

**Keyboard Shortcut:** `B` to toggle visibility

**CSS:** `styles/components/scale-bar.css`

---

## Theming System

### Overview

Luxar viewer now features a **modular theming system** with runtime theme switching. Styles are separated into external CSS files with CSS custom properties (variables) for easy customization.

### Available Themes

1. **Dark Theme** (default) - Optimized for scientific visualization with subdued colors
2. **Light Theme** - Bright theme for well-lit environments, WCAG AA compliant
3. **Frosted Glass** - Apple-inspired frosted glass design with translucent panels and heavy blur

### Using Themes

**Programmatically**:

```typescript
import { ThemeManager } from '../themes';

// Switch themes
ThemeManager.getInstance().setTheme('dark');
ThemeManager.getInstance().setTheme('light');
ThemeManager.getInstance().setTheme('frosted-glass');
ThemeManager.getInstance().setTheme('liquid-glass');

// Get current theme
const current = ThemeManager.getInstance().getCurrentTheme();
console.log(current.name); // "Dark"

// Subscribe to theme changes
ThemeManager.getInstance().onChange((theme) => {
  console.log('Theme changed to:', theme.name);
});
```

**Via UI**:

- Press `R` to open Rendering Controls
- Expand "🎨 Theme" folder
- Select theme from dropdown

**Via URL**:

```
http://localhost:5173/?theme=dark
http://localhost:5173/?theme=light
http://localhost:5173/?theme=frosted-glass
http://localhost:5173/?theme=liquid-glass
```

### CSS Architecture

**Three-Layer System**:

```
Theme System (CSS variables)
    ↓
Component Styles (CSS files, BEM classes)
    ↓
Component Logic (TypeScript, no styling)
```

**Animation Consistency**: All UI panels use a consistent 0.15s fade-in animation for smooth, professional transitions.

**File Structure**:

```
src/styles/
├── index.css           # Main entry point
├── reset.css           # CSS reset
├── base/               # Base styles
│   ├── typography.css
│   ├── layout.css
│   └── utilities.css   # 80+ utility classes
└── components/         # Component-specific styles
    ├── error-dialog.css
    ├── help-overlay.css
    ├── dimension-sliders.css
    ├── debug-console.css
    ├── dataset-browser.css
    ├── data-loading-monitor.css
    ├── recording-panel.css
    └── scale-bar.css
```

**BEM Naming Convention**:

```css
/* Component */
.luxar-error-dialog {
}

/* Element */
.luxar-error-dialog__header {
}
.luxar-error-dialog__title {
}

/* Modifier */
.luxar-error-dialog--visible {
}
```

**CSS Custom Properties**:

```css
/* Colors */
--luxar-bg-primary, --luxar-bg-secondary, --luxar-bg-tertiary
--luxar-text-primary, --luxar-text-secondary, --luxar-text-muted
--luxar-success, --luxar-warning, --luxar-error, --luxar-info

/* Spacing (8px grid) */
--luxar-spacing-0 through --luxar-spacing-20

/* Typography */
--luxar-font-base, --luxar-font-mono
--luxar-text-xs through --luxar-text-3xl
--luxar-font-normal, --luxar-font-medium, --luxar-font-semibold, --luxar-font-bold

/* Effects */
--luxar-radius-sm, --luxar-radius-md, --luxar-radius-lg
--luxar-shadow-sm, --luxar-shadow-md, --luxar-shadow-lg
--luxar-blur-sm, --luxar-blur-md, --luxar-blur-lg
```

### Component Styling

Components now use **CSS classes** instead of inline styles:

**Before**:

```typescript
element.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
element.style.padding = '15px';
element.style.borderRadius = '8px';
// ... 50 more inline styles
```

**After**:

```typescript
element.className = 'luxar-dimension-sliders';
// All styling in CSS file
```

### Migrated Components

These components fully support theming:

- ✅ Error Dialog (helpers.ts)
- ✅ Help Overlay (helpers.ts)
- ✅ Loading Indicator (helpers.ts)
- ✅ Dimension Sliders (dimension-sliders.ts)
- ✅ Debug Console (debug-console.ts) - proper BEM naming (.luxar-debug-console)
- ✅ Dataset Browser (dataset-browser.ts)
- ✅ Data Loading Monitor (data-loading-monitor.ts) - CSS complete, core templates refactored
- ✅ Rendering Controls (rendering-controls.ts) - lil-gui theme integration via CSS variables

---

## Implementation Details

### Event Delegation Pattern

The UI components use event delegation for efficient event handling:

```typescript
// Instead of inline handlers:
// ❌ onclick="__luxarMonitor.expand()"

// We use data attributes:
// ✅ data-action="expand"

// Single event handler manages all interactions:
private handleUIEvent(event: Event): void {
  const target = event.target as HTMLElement;
  const action = target.dataset.action;

  switch (action) {
    case 'expand': this.expand(); break;
    case 'hide': this.hide(); break;
    // ... other actions
  }
}
```

**Benefits:**

- No global namespace pollution
- Better security (no inline JavaScript)
- Improved testability
- Type-safe event handling
- Single listener for multiple elements

### Component Lifecycle

All UI components follow a consistent lifecycle:

1. **Construction**: Initialize state and configuration
2. **Creation**: Build DOM elements with event delegation
3. **Updates**: Efficient DOM updates via requestAnimationFrame
4. **Disposal**: Clean up listeners and resources

## User Interactions

### Keyboard Controls

Global keyboard shortcuts managed by the UI system:

| Key       | Action                     | Context                  |
| --------- | -------------------------- | ------------------------ |
| `H`       | Toggle help                | Global                   |
| `P`       | Toggle performance monitor | Global                   |
| `R`       | Toggle rendering controls  | Global                   |
| `N`       | Toggle dimension sliders   | When nD data loaded      |
| `M`       | Cycle data loading monitor | Global                   |
| `T`       | Toggle recording panel     | Global                   |
| `G`       | Quick screenshot           | Global                   |
| `B`       | Toggle scale bar           | Global                   |
| `Ctrl+L`  | Toggle debug console       | Development mode         |
| `Esc`     | Close active panel         | Any panel open           |
| `K`       | Toggle dimension animation | When dimension selected  |
| `Home`    | Jump to dimension start    | When dimension selected  |
| `End`     | Jump to dimension end      | When dimension selected  |
| `Shift+↑` | Increase animation speed   | When dimension animating |
| `Shift+↓` | Decrease animation speed   | When dimension animating |

### Mouse Interactions

- **Drag**: Move panels around screen
- **Scroll**: Adjust slider values with precision
- **Right-click**: Context menus
- **Double-click**: Reset to default values

### Touch Support

Mobile-friendly interactions:

- Touch drag for panel movement
- Pinch to zoom in browser view
- Tap outside to close panels
- Swipe for slider adjustments

---

## Layout Management

### Panel System

Flexible panel layout with:

```typescript
interface PanelConfig {
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
  size: 'small' | 'medium' | 'large' | 'auto';
  collapsible: boolean;
  draggable: boolean;
  resizable: boolean;
  persistent: boolean; // Remember state
}
```

### Responsive Design

Automatic layout adjustments:

```typescript
// Desktop: Multi-column layout
// Tablet: Stacked panels
// Mobile: Full-screen panels

function adaptLayout() {
  const width = window.innerWidth;

  if (width < 768) {
    // Mobile: full-screen panels
    setLayout('mobile');
  } else if (width < 1024) {
    // Tablet: stacked layout
    setLayout('tablet');
  } else {
    // Desktop: floating panels
    setLayout('desktop');
  }
}
```

---

## State Management

### UI State

Centralized state management for UI components:

```typescript
interface UIState {
  panels: {
    rendering: { visible: boolean; collapsed: boolean };
    dimensions: { visible: boolean; position: Point };
    performance: { visible: boolean };
    debug: { visible: boolean };
  };

  settings: {
    theme: 'dark' | 'light';
    compactMode: boolean;
    animations: boolean;
  };
}
```

### Persistence

Save UI preferences:

```typescript
// Save to localStorage
function saveUIState(state: UIState) {
  localStorage.setItem('luxar-ui-state', JSON.stringify(state));
}

// Restore on load
function restoreUIState(): UIState {
  const saved = localStorage.getItem('luxar-ui-state');
  return saved ? JSON.parse(saved) : defaultState;
}
```

---

## Usage Examples

### Complete UI Setup

```typescript
import { DimensionSliders, RenderingControls, PerformanceMonitor, DatasetBrowser } from './ui';
import { DataMonitorManager } from '../data';

// Initialize UI components
const ui = {
  dimensions: new DimensionSliders(),
  rendering: new RenderingControls(postProcessing),
  performance: new PerformanceMonitor(renderer),
  browser: new DatasetBrowser(),
};

// Data loading monitor is automatically created by SceneLoader
// But you can access it via the manager:
const monitor = DataMonitorManager.getInstance().getDefaultMonitor();
if (monitor) {
  // Monitor is already connected to loaders automatically
  // Use M key to show/hide/expand
}

// Connect to application
ui.dimensions.on('change', updateSlice);
ui.rendering.on('change', updateRendering);
ui.browser.on('select', loadDataset);
```

### Custom Panel Creation

```typescript
class CustomPanel extends UIPanel {
  constructor() {
    super({
      title: 'Custom Controls',
      position: 'top-right',
      collapsible: true,
    });
  }

  render() {
    return `
      <div class="custom-panel">
        <button onclick="this.handleAction()">
          Custom Action
        </button>
      </div>
    `;
  }
}
```

### Responsive UI Updates

```typescript
// Update UI based on data
function updateUIForDataset(dataset) {
  // Show/hide dimension sliders
  if (dataset.ndim > 3) {
    ui.dimensions.show();
    ui.dimensions.setDimensions(dataset.dims);
  } else {
    ui.dimensions.hide();
  }

  // Update performance monitor
  ui.performance.setPointCount(dataset.pointCount);

  // Configure rendering controls
  ui.rendering.setDefaults(dataset.renderingConfig);
}
```

---

## Accessibility

### ARIA Support

All UI components include proper ARIA attributes:

```html
<div
  role="slider"
  aria-label="Time dimension"
  aria-valuenow="50"
  aria-valuemin="0"
  aria-valuemax="100"
></div>
```

### Keyboard Navigation

Full keyboard support for all controls:

- Tab navigation between controls
- Arrow keys for sliders
- Enter/Space for buttons
- Escape to close panels

### Screen Reader Support

Descriptive labels and live regions:

```typescript
<div aria-live="polite" aria-atomic="true">
  Dimension changed: Time = 50ms
</div>
```

---

## Performance Considerations

### Rendering Optimization

1. **Virtual scrolling** for long lists
2. **Debounced updates** for sliders
3. **RAF-based animations**
4. **CSS transforms** for movement
5. **Will-change** for animated properties

### Memory Management

```typescript
class UIComponent {
  dispose() {
    // Remove event listeners
    this.removeEventListeners();

    // Clear references
    this.elements = null;

    // Remove from DOM
    this.container.remove();
  }
}
```

---

## Configuration

### UI Settings

```typescript
const UI_CONFIG = {
  // Appearance
  theme: 'dark',
  fontSize: 14,
  animations: true,

  // Layout
  panelOpacity: 0.95,
  panelBlur: 10,
  cornerRadius: 8,

  // Behavior
  autoHideDelay: 3000,
  doubleClickReset: true,
  persistState: true,

  // Performance
  updateThrottle: 16, // 60 FPS
  debounceDelay: 100,
};
```

### Customization

Override default styles:

```css
/* Custom theme */
.luxar-ui-panel {
  --panel-bg: #2a2a2a;
  --panel-border: #4a4a4a;
  --accent-color: #00ff88;
}
```

---

## API Reference

### DimensionSliders

| Method                            | Description                     |
| --------------------------------- | ------------------------------- |
| `setDimensions(dims)`             | Configure dimension sliders     |
| `setValue(dim, value)`            | Set dimension value             |
| `show()/hide()`                   | Toggle visibility               |
| `on(event, handler)`              | Subscribe to events             |
| `setAnimationManager(manager)`    | Connect animation manager to UI |
| `updatePlayButtonState(dimIndex)` | Update play button visual state |
| `updateFPSDisplay(dimIndex, fps)` | Update FPS display value        |

### RenderingControls

| Method                                  | Description          |
| --------------------------------------- | -------------------- |
| `setBloom(strength, radius, threshold)` | Configure bloom      |
| `setToneMapping(type)`                  | Set tone mapping     |
| `setAntiAliasing(type, enabled)`        | Toggle AA methods    |
| `getState()`                            | Get current settings |

**Architecture**: RenderingControls uses a modular setup architecture where each category of controls (navigation, camera, HDR, anti-aliasing, post-processing) is initialized by a dedicated setup module in `./rendering-controls/`. This improves maintainability and keeps files under token limits. See [`./rendering-controls/README.md`](./rendering-controls/README.md) for details.

### PerformanceMonitor

| Method                 | Description          |
| ---------------------- | -------------------- |
| `begin()/end()`        | Frame timing markers |
| `setPointCount(count)` | Update point counter |
| `show()/hide()`        | Toggle visibility    |
| `reset()`              | Clear statistics     |

### DatasetBrowser

| Method           | Description              |
| ---------------- | ------------------------ |
| `navigate(path)` | Browse to path           |
| `refresh()`      | Reload current directory |
| `setServer(url)` | Change data server       |
| `getSelection()` | Get selected dataset     |

### DataLoadingMonitor

| Method                           | Description                           |
| -------------------------------- | ------------------------------------- |
| `connectLoader(path, loader)`    | Connect a loader for monitoring       |
| `disconnectLoader(path)`         | Disconnect a specific loader          |
| `disconnectAllLoaders()`         | Disconnect all loaders (scene change) |
| `show()/hide()/toggle()`         | Control visibility                    |
| `cycleState()`                   | Cycle through hidden→mini→expanded    |
| `expand()/minimize()/collapse()` | Control expanded state                |
| `getGlobalStats()`               | Get aggregated statistics             |
| `getLoaderMetrics(path)`         | Get metrics for specific loader       |
| `getRecommendations()`           | Get optimization recommendations      |
| `setActiveTab(tab)`              | Switch between overview/cache/spatial |
| `dispose()`                      | Clean up resources                    |

---

## Best Practices

### Component Design

1. **Keep components focused**: Single responsibility
2. **Use composition**: Build complex UIs from simple parts
3. **Maintain consistency**: Follow design system
4. **Optimize updates**: Batch DOM changes
5. **Handle edge cases**: Empty states, errors

### User Experience

1. **Provide feedback**: Loading states, confirmations
2. **Be responsive**: Immediate visual feedback
3. **Guide users**: Tooltips, hints, documentation
4. **Remember preferences**: Persist user settings
5. **Support undo**: Allow reverting changes

---

## Troubleshooting

### Common Issues

**Problem: Panels not visible**

- Check z-index conflicts
- Verify panel state in localStorage
- Ensure container element exists

**Problem: Sliders not responding**

- Check input event listeners
- Verify dimension data is valid
- Check for JavaScript errors

**Problem: Performance monitor inaccurate**

- Ensure begin/end pairs match
- Check for blocking operations
- Verify RAF timing

---

## License

Part of the Luxar project. See root LICENSE file for details.

---

_For implementation details, see the source files in this directory._
