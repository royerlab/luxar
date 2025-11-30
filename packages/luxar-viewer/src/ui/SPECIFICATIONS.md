# luxar-viewer.ui - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-30

## Purpose

The `luxar-viewer.ui` package provides user interface components for controlling visualization parameters, navigating nD datasets, monitoring performance, and debugging. It implements napari-inspired dimension sliders, rendering controls, data loading monitors, and developer tools.

**Core Responsibility**: Provide intuitive, responsive UI for real-time visualization control with efficient DOM updates and minimal performance overhead.

**Related Specifications**:
- `luxar-viewer.data` - Data loading monitoring (see `../data/SPECIFICATIONS.md`)
- `luxar-viewer.scene` - Scene state integration (see `../scene/SPECIFICATIONS.md`)

---

## Table of Contents

1. [Dimension Sliders](#dimension-sliders)
2. [Data Loading Monitor](#data-loading-monitor)
3. [Performance Monitor](#performance-monitor)
4. [Event Delegation Pattern](#event-delegation-pattern)

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
    const container = document.createElement('div')
    container.className = 'dimension-sliders'

    // Get non-displayed dimensions
    const displayedSet = new Set(dims.displayed)
    const nonDisplayed = []
    for (let i = 0; i < dims.ndim; i++) {
        if (!displayedSet.has(i)) {
            nonDisplayed.push(i)
        }
    }

    // Create slider for each non-displayed dimension
    for (const dimIdx of nonDisplayed) {
        const meta = dims.metadata?.[dimIdx]
        const slider = createSlider(dimIdx, meta, dims.currentStep[dimIdx])
        container.appendChild(slider)
    }

    return container
}
```

### 1.3 Slider Update Algorithm

**Purpose**: Sync slider position with current dimension value.

```typescript
function updateSlider(dimIdx: number, value: number): void {
    const slider = sliderElements.get(dimIdx)
    if (!slider) return

    const meta = dims.metadata[dimIdx]
    const [min, max] = meta.range

    // Convert value to slider position (0-100)
    const percentage = ((value - min) / (max - min)) * 100

    // Update slider element
    slider.value = percentage.toString()

    // Update label
    const label = slider.querySelector('.value-label')
    label.textContent = `${value.toFixed(2)} ${meta.unit}`
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
    | { type: 'query', cellCount: number, pointCount: number, duration: number }
    | { type: 'load', rangeCount: number, bytesLoaded: number, duration: number }
    | { type: 'cache-hit', rangeCount: number }
    | { type: 'cache-miss', rangeCount: number }
    | { type: 'evict', bytesEvicted: number }
    | { type: 'error', message: string }
```

### 2.3 Statistics Aggregation

**Purpose**: Aggregate events into actionable metrics.

**Metrics**:

```typescript
interface LoadingMetrics {
    // Queries
    totalQueries: number
    avgQueryTime: number
    avgCellsPerQuery: number
    avgPointsPerQuery: number

    // Cache
    cacheHitRate: number       // hits / (hits + misses)
    totalCachedBytes: number
    evictionCount: number

    // Loading
    totalBytesLoaded: number
    avgLoadTime: number
    loadBandwidth: number      // bytes/sec

    // Errors
    errorCount: number
    lastError?: string
}
```

**Calculation**:

```typescript
function calculateMetrics(events: MonitorEvent[]): LoadingMetrics {
    // Filter by event type
    const queries = events.filter(e => e.type === 'query')
    const loads = events.filter(e => e.type === 'load')
    const hits = events.filter(e => e.type === 'cache-hit')
    const misses = events.filter(e => e.type === 'cache-miss')
    const errors = events.filter(e => e.type === 'error')

    // Aggregate
    const totalQueries = queries.length
    const avgQueryTime = average(queries.map(q => q.duration))
    const avgCellsPerQuery = average(queries.map(q => q.cellCount))
    const avgPointsPerQuery = average(queries.map(q => q.pointCount))

    // Cache metrics
    const cacheHitRate = hits.length / (hits.length + misses.length)

    // Load metrics
    const totalBytesLoaded = sum(loads.map(l => l.bytesLoaded))
    const avgLoadTime = average(loads.map(l => l.duration))

    // Calculate bandwidth
    const timeWindow = 60000  // Last 60 seconds
    const recentLoads = loads.filter(l => Date.now() - l.timestamp < timeWindow)
    const recentBytes = sum(recentLoads.map(l => l.bytesLoaded))
    const loadBandwidth = (recentBytes / timeWindow) * 1000  // bytes/sec

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
        lastError: errors[errors.length - 1]?.message
    }
}
```

### 2.4 Performance Timeline

**Purpose**: Real-time graph of loading performance over time.

**Data Structure**:

```typescript
interface TimelineDataPoint {
    timestamp: number       // Unix timestamp
    queryTime: number       // Query duration (ms)
    loadTime: number        // Load duration (ms)
    pointsLoaded: number    // Number of points
    bytesLoaded: number     // Bytes transferred
}
```

**Rendering**:

```typescript
// Canvas-based timeline with 60-second window
function renderTimeline(dataPoints: TimelineDataPoint[]): void {
    const canvas = getTimelineCanvas()
    const ctx = canvas.getContext('2d')

    const now = Date.now()
    const timeWindow = 60000  // 60 seconds

    // Filter to recent data
    const recent = dataPoints.filter(p => now - p.timestamp < timeWindow)

    // Draw background
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw data points
    for (let i = 1; i < recent.length; i++) {
        const prev = recent[i - 1]
        const curr = recent[i]

        // Map time to x-axis
        const x1 = mapTimeToX(prev.timestamp, now, timeWindow, canvas.width)
        const x2 = mapTimeToX(curr.timestamp, now, timeWindow, canvas.width)

        // Map value to y-axis (query time)
        const y1 = mapValueToY(prev.queryTime, maxQueryTime, canvas.height)
        const y2 = mapValueToY(curr.queryTime, maxQueryTime, canvas.height)

        // Draw line
        ctx.strokeStyle = '#00a0ff'
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
    }
}
```

---

## 3. Performance Monitor

### 3.1 FPS Calculation

**Algorithm**: Rolling average over last N frames

```typescript
class PerformanceMonitor {
    private frameTimes: number[] = []
    private maxFrameCount = 60

    begin(): void {
        this.frameStartTime = performance.now()
    }

    end(): void {
        const frameTime = performance.now() - this.frameStartTime
        this.frameTimes.push(frameTime)

        if (this.frameTimes.length > this.maxFrameCount) {
            this.frameTimes.shift()  // Remove oldest
        }

        // Update display
        this.updateDisplay()
    }

    private calculateFPS(): number {
        if (this.frameTimes.length === 0) return 0

        const avgFrameTime = average(this.frameTimes)
        return 1000 / avgFrameTime  // Convert ms to FPS
    }
}
```

### 3.2 GPU Memory Estimation

**Purpose**: Estimate GPU memory usage from loaded geometries.

```typescript
function estimateGPUMemory(): number {
    let totalBytes = 0

    scene.traverse(object => {
        if (object instanceof THREE.Points) {
            const geometry = object.geometry

            // Position attribute: Float32[N, 3]
            const positions = geometry.getAttribute('position')
            totalBytes += positions.count * 3 * 4  // 4 bytes per float

            // Color attribute: Float32[N, 3]
            const colors = geometry.getAttribute('color')
            if (colors) {
                totalBytes += colors.count * 3 * 4
            }

            // Radius attribute: Float32[N]
            const radii = geometry.getAttribute('radius')
            if (radii) {
                totalBytes += radii.count * 4
            }

            // Sharpness attribute: Float32[N]
            const sharpness = geometry.getAttribute('sharpness')
            if (sharpness) {
                totalBytes += sharpness.count * 4
            }
        }
    })

    return totalBytes
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
    private container: HTMLElement

    constructor() {
        // Single event listener for entire component
        this.container.addEventListener('click', this.handleClick.bind(this))
    }

    private handleClick(event: Event): void {
        const target = event.target as HTMLElement
        const action = target.dataset.action

        switch (action) {
            case 'expand':
                this.expand()
                break
            case 'hide':
                this.hide()
                break
            case 'toggle-tab':
                const tab = target.dataset.tab
                this.setActiveTab(tab)
                break
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

## Data Structures

### DimensionSliders

```typescript
interface DimensionSliders {
    container: HTMLElement
    sliders: Map<number, SliderElement>
    dims: SimpleDims | null

    create(dims: SimpleDims): void
    update(dimIdx: number, value: number): void
    show(): void
    hide(): void
    dispose(): void
}

interface SliderElement {
    container: HTMLElement
    input: HTMLInputElement
    label: HTMLElement
    dimIndex: number
}
```

### DataLoadingMonitor

```typescript
interface DataLoadingMonitor {
    state: 'hidden' | 'mini' | 'expanded'
    events: MonitorEvent[]
    connectedLoaders: Map<string, Loader>

    cycleState(): void  // hidden → mini → expanded → hidden
    connectLoader(path: string, loader: Loader): void
    disconnectLoader(path: string): void
    getMetrics(): LoadingMetrics
    dispose(): void
}
```

---

## Changelog

- **v1.0.0** (2025-01-30): Initial specification
  - Napari-inspired dimension sliders for nD navigation
  - Three-state data loading monitor (hidden/mini/expanded)
  - Event-driven monitoring with aggregated metrics
  - Performance timeline with canvas rendering
  - FPS calculation with rolling average
  - GPU memory estimation from geometry attributes
  - Event delegation pattern (no global functions)
