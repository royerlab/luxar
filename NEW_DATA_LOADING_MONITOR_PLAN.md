# Data Loading Monitor Overhaul Plan (Detailed Version)

## Executive Summary
This plan outlines a comprehensive overhaul of the Luxar Data Loading Monitor to transform it from a basic stats display into a professional, highly functional monitoring system that provides deep insights into the complex nD data loading, spatial indexing, and caching architecture.

## Current State Analysis

### Identified Issues
1. **UI Design Non-Compliance**: Inline styles, inconsistent spacing, missing backdrop blur
2. **Missing Critical Information**: No dimension context, no scene hierarchy, no network breakdown
3. **Poor Visualizations**: Abstract grid, unlabeled timeline, minimal compact view
4. **Generic Recommendations**: Non-actionable advice, no specific thresholds
5. **No Keyboard Integration**: Missing Ctrl+M cycling functionality
6. **Missing Zarr Metrics**: No compression stats, chunk efficiency unknown

## Phase 1: Foundation & UI Compliance (Priority: Critical)

### 1.1 Keyboard Integration
- **Current**: No keyboard shortcut exists
- **Target**: Ctrl+M cycles through three states
  - First press: Show mini monitor (if hidden)
  - Second press: Expand to full monitor
  - Third press: Hide monitor
- **Implementation**:
  ```typescript
  // In input-handler.ts
  if (event.ctrlKey && event.key === 'm') {
    cycleMonitorState(); // mini → expanded → hidden → mini
  }
  ```
- **Documentation**: Update help-overlay.ts with "Ctrl+M: Cycle monitor (mini/expanded/off)"

### 1.2 UI Design System Compliance

#### Style Architecture
Create `data-loading-monitor-styles.ts`:
```typescript
export const MonitorStyles = {
  panel: {
    background: 'rgba(30, 30, 30, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
    padding: '15px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
    color: '#e0e0e0',
    zIndex: 1000,
  },
  colors: {
    success: '#4CAF50',
    warning: '#FFC107', 
    error: '#f44336',
    info: '#2196F3',
    secondary: '#9C27B0',
    muted: 'rgba(255, 255, 255, 0.6)',
  },
  spacing: {
    panel: 15,
    section: 10,
    gap: 8,
    compact: 5,
  },
  typography: {
    title: { fontSize: '14px', fontWeight: 'bold' },
    sectionHeader: { fontSize: '12px', fontWeight: 600 },
    body: { fontSize: '11px' },
    small: { fontSize: '10px' },
    mono: { fontFamily: 'monospace' },
  }
};
```

#### Components to Update
- Replace all inline style strings with style objects
- Ensure consistent spacing throughout
- Add proper backdrop blur to all panels
- Fix z-index hierarchy (1000 for panels, 1100 for overlays)
- Apply semantic color system

### 1.3 Visual Clarity & Tooltips

#### Tooltip System
```typescript
interface MetricTooltip {
  metric: string;
  description: string;
  unit: string;
  goodRange?: string;
  badRange?: string;
}

const tooltips: Record<string, MetricTooltip> = {
  cacheHitRate: {
    metric: 'Cache Hit Rate',
    description: 'Percentage of requests served from memory cache',
    unit: '%',
    goodRange: '>70%',
    badRange: '<30%',
  },
  avgQueryTime: {
    metric: 'Average Query Time',
    description: 'Time to query spatial index and identify visible points',
    unit: 'ms',
    goodRange: '<50ms',
    badRange: '>100ms',
  },
  // ... more tooltips
};
```

#### Visual Improvements
- Add `title` attribute to all metrics with detailed explanations
- Show units inline with values (e.g., "45ms" not just "45")
- Color code values based on severity:
  - Green: Good performance
  - Amber: Warning, needs attention
  - Red: Critical, immediate action needed
- Add hover effects for interactive elements

## Phase 2: Critical Missing Information

### 2.1 Dimension & Slicing Context

#### Information to Display
```typescript
interface DimensionContext {
  // Scene dimensions
  totalDimensions: string[];        // e.g., ['X', 'Y', 'Z', 'C', 'T']
  displayedDimensions: string[];    // e.g., ['X', 'Y', 'Z']
  slicedDimensions: string[];       // e.g., ['C', 'T']
  
  // Current slice state
  slicePositions: Map<string, number>;  // e.g., {C: 0, T: 42}
  sliceTolerances: Map<string, number>; // e.g., {C: 0, T: 10}
  
  // Navigation state
  selectedDimension: string | null;     // Currently selected for navigation
  stepSizes: Map<string, number>;       // Step sizes for keyboard nav
  
  // Effective radius
  effectiveRadiusEnabled: boolean;
  effectiveRadiusValues: Map<string, number>;
}
```

#### UI Representation
```html
<div class="dimension-info">
  <div class="viewing-dims">
    <label>Viewing:</label>
    <value>XYZ of XYZCT</value>
  </div>
  <div class="slice-positions">
    <label>Slice:</label>
    <value>C=0, T=42±10</value>
    <tooltip>T has tolerance of ±10 units</tooltip>
  </div>
  <div class="effective-radius" if={enabled}>
    <label>Effective Radius:</label>
    <value>Active (dynamic sizing)</value>
  </div>
</div>
```

### 2.2 Scene Loading Context

#### Node Loading Information
```typescript
interface NodeLoadingInfo {
  path: string;
  type: 'points' | 'group';
  status: 'idle' | 'loading' | 'loaded' | 'error';
  pointsLoaded: number;
  totalPoints: number;
  memoryUsed: number;
  cacheStatus: 'cold' | 'warm' | 'hot';
  transform?: {
    position: [number, number, number];
    scale: [number, number, number];
    rotation: [number, number, number];
  };
  blendingMode?: 'normal' | 'additive' | 'max';
  material?: {
    transparent: boolean;
    opacity: number;
  };
}
```

#### Hierarchical Display
```html
<div class="scene-hierarchy">
  <div class="node-item">
    <icon>📦</icon>
    <name>/scene/points_0</name>
    <status class="loading">⟳ 45%</status>
    <stats>12.5K/28K pts</stats>
  </div>
  <div class="node-item nested">
    <icon>📦</icon>
    <name>/scene/group_1/points_1</name>
    <status class="loaded">✓</status>
    <stats>8K pts cached</stats>
  </div>
</div>
```

### 2.3 Network & Memory Breakdown

#### Detailed Metrics
```typescript
interface DetailedMetrics {
  // Network breakdown
  networkLatency: number;      // Time in network transfer
  parseLatency: number;         // Time parsing zarr data
  gpuUploadLatency: number;     // Time uploading to GPU
  totalLatency: number;         // Total end-to-end
  
  // Memory breakdown
  cacheMemory: number;          // Memory in range cache
  gpuMemory: number;            // Memory in WebGL buffers
  indexMemory: number;          // Memory for spatial index
  totalMemory: number;          // Total memory used
  
  // Zarr specific
  compressedBytes: number;      // Bytes transferred (compressed)
  uncompressedBytes: number;    // Bytes after decompression
  compressionRatio: number;     // Compression efficiency
  
  // Chunk statistics
  chunksLoaded: number;         // Number of chunks loaded
  chunksUsed: number;           // Chunks with visible points
  chunkEfficiency: number;      // % of loaded chunks used
}
```

#### Visual Representation
```html
<div class="network-breakdown">
  <div class="latency-bar">
    <div class="segment network" style="width: 60%">Network: 60ms</div>
    <div class="segment parse" style="width: 30%">Parse: 30ms</div>
    <div class="segment gpu" style="width: 10%">GPU: 10ms</div>
  </div>
</div>

<div class="memory-breakdown">
  <div class="memory-bar">
    <div class="segment cache" style="width: 70%">Cache: 350MB</div>
    <div class="segment gpu" style="width: 25%">GPU: 125MB</div>
    <div class="segment index" style="width: 5%">Index: 25MB</div>
  </div>
</div>

<div class="zarr-stats">
  <stat>
    <label>Compression:</label>
    <value>3.2x</value>
    <detail>125MB → 39MB</detail>
  </stat>
  <stat>
    <label>Chunk Efficiency:</label>
    <value>78%</value>
    <detail>39/50 chunks used</detail>
  </stat>
</div>
```

## Phase 3: Improved Visualizations (Focused)

### 3.1 Spatial Grid Enhancement

#### Enhanced Grid Cell Information
```typescript
interface EnhancedGridCell {
  // Existing
  x: number;
  y: number;
  isOccupied: boolean;
  isCached: boolean;
  
  // New additions
  pointDensity: number;        // Points per cell
  lastAccessTime: number;      // For recency visualization
  accessCount: number;         // For heat mapping
  compressionRatio: number;    // Zarr compression for this cell
  loadTime: number;            // Time taken to load
}
```

#### Visualization Features
- Color intensity based on point density
- Fade effect for access recency (recent = bright)
- Query bounds overlay with dashed lines
- Legend with clear state indicators:
  ```
  □ Empty  ▨ Occupied  ■ Cached  ▤ Loading  ◈ Queried
  ```

### 3.2 Timeline Improvements

#### Clear Axis Labels
```typescript
// Y-axis labels with units
const yAxisLabels = {
  queryTime: { label: 'Query (ms)', max: 200 },
  loadTime: { label: 'Load (ms)', max: 500 },
  cacheRate: { label: 'Cache Hit %', max: 100 },
  memory: { label: 'Memory (MB)', max: 1000 },
};

// X-axis time markers
const timeMarkers = ['1m ago', '30s ago', 'now'];
```

#### Event Markers
- Red vertical lines for errors
- Orange markers for cache evictions
- Green markers for successful prefetches
- Correlation markers when FPS < 30

### 3.3 Compact View Enhancement

#### Mini Monitor Layout
```html
<div class="monitor-compact">
  <div class="status-icon">🔍</div> <!-- Spatial index active -->
  <div class="key-metrics">
    <span class="nodes">3/5 ▲</span> <!-- 3 of 5 nodes loading -->
    <span class="points">125K</span>
    <span class="cache">78% ●</span> <!-- Cache hit rate with trend -->
    <span class="memory">450/800MB</span>
    <span class="compression">3.2x</span>
  </div>
  <div class="sparkline">
    <!-- Mini cache hit rate trend graph -->
  </div>
  <div class="alerts">
    <span class="warning" if={hasWarning}>⚠</span>
    <span class="error" if={hasError}>⚡</span>
  </div>
</div>
```

## Phase 4: Better Recommendations

### 4.1 Specific Actionable Advice

#### Example Recommendations
```typescript
const recommendations = [
  {
    id: 'low-cache-hit',
    severity: 'warning',
    title: 'Low Cache Hit Rate',
    current: 'Cache hit rate: 23%',
    impact: 'Queries taking 3x longer than optimal',
    suggestion: 'Increase cache size from 500MB to 1.2GB',
    expected: 'Would improve hit rate to ~80%, reducing query time by 65%',
    action: { 
      type: 'config',
      setting: 'maxMemoryMB',
      value: 1200,
    },
  },
  {
    id: 'inefficient-grid',
    severity: 'info',
    title: 'Suboptimal Grid Resolution',
    current: 'Grid cells contain avg 50K points',
    impact: 'Loading 5x more data than visible',
    suggestion: 'Rebuild index with 2x finer grid resolution',
    expected: 'Reduce average cell size to 10K points, improving query efficiency by 80%',
    action: {
      type: 'rebuild',
      command: 'luxar rebuild-index --grid-scale=2',
    },
  },
];
```

### 4.2 Common Issues Detection

#### Detection Patterns
```typescript
class IssueDetector {
  detectThrashing(events: MonitorEvent[]): boolean {
    // Frequent evictions followed by reloads
    const evictions = events.filter(e => e.type === 'evict');
    const reloads = events.filter(e => e.type === 'load');
    return evictions.length > 10 && reloads.length > evictions.length * 0.8;
  }
  
  detectNetworkBottleneck(metrics: DetailedMetrics): boolean {
    // Network time dominates total latency
    return metrics.networkLatency > metrics.totalLatency * 0.7;
  }
  
  detectIneffiecientChunking(metrics: DetailedMetrics): boolean {
    // Low chunk utilization
    return metrics.chunkEfficiency < 0.5;
  }
  
  detectMemoryPressure(metrics: DetailedMetrics): boolean {
    // Near memory limit with frequent evictions
    return metrics.totalMemory > metrics.memoryLimit * 0.9;
  }
}
```

## Phase 5: Testing & Documentation

### 5.1 Testing Strategy

#### Unit Tests
```typescript
// data-loading-monitor.test.ts
describe('DataLoadingMonitor', () => {
  describe('Metrics Calculation', () => {
    it('should calculate cache hit rate correctly');
    it('should aggregate metrics across multiple loaders');
    it('should handle missing data gracefully');
  });
  
  describe('UI State Management', () => {
    it('should cycle through states with Ctrl+M');
    it('should persist preferences to localStorage');
    it('should restore state on reload');
  });
  
  describe('Visualization', () => {
    it('should render spatial grid with correct density');
    it('should update timeline without memory leaks');
    it('should show tooltips on hover');
  });
  
  describe('Recommendations', () => {
    it('should detect thrashing condition');
    it('should provide specific cache size recommendations');
    it('should calculate expected improvements');
  });
});
```

#### Performance Tests
- Monitor UI updates should complete in <50ms
- No memory leaks over extended sessions
- Canvas rendering at 60fps
- Event batching efficiency

### 5.2 Documentation

#### User Documentation
```markdown
# Data Loading Monitor Guide

## Overview
The Data Loading Monitor provides real-time insights into Luxar's data loading performance...

## Keyboard Shortcuts
- `Ctrl+M`: Cycle monitor (mini → expanded → hidden)

## Understanding Metrics

### Cache Hit Rate
- **What it means**: Percentage of data requests served from memory
- **Good range**: >70%
- **Warning range**: 30-70%
- **Critical range**: <30%
- **How to improve**: Increase cache size or preload radius

### Query Time
- **What it means**: Time to identify visible points using spatial index
- **Good range**: <50ms
- **Warning range**: 50-100ms
- **Critical range**: >100ms
- **How to improve**: Optimize grid resolution or increase cache

[... continue for all metrics ...]
```

#### Developer Documentation
```markdown
# Data Loading Monitor Architecture

## Components
- `DataLoadingMonitor`: Main orchestrator
- `SpatialQueryVisualizer`: Grid visualization
- `PerformanceTimeline`: Time-series graphs
- `LoadingAdvisor`: Intelligent recommendations

## Event Flow
1. Loaders emit events via `MonitorEvent`
2. Monitor aggregates into `LoaderMetrics`
3. UI updates at 10Hz via `updateUI()`
4. Components render visualizations

## Adding New Metrics
1. Add to `MonitorEvent.data` interface
2. Update `LoaderMetrics` interface
3. Add aggregation in `updateMetricsFromEvent()`
4. Add UI representation in relevant tab
5. Add tooltip in `MetricTooltip` registry
```

## Implementation Timeline

### Week 1: Foundation (Phase 1 + Phase 2.1)
- Day 1-2: Keyboard integration and state cycling
- Day 3-4: Style system implementation
- Day 5: Dimension context display

### Week 2: Information (Phase 2.2-2.3 + Zarr stats)
- Day 1-2: Scene hierarchy and loading status
- Day 3-4: Network/memory breakdown
- Day 5: Zarr compression statistics

### Week 3: Visualization & UX (Phase 3 + Phase 4)
- Day 1-2: Enhanced spatial grid
- Day 3: Timeline improvements  
- Day 4: Compact view enhancement
- Day 5: Better recommendations

### Week 4: Quality (Phase 5)
- Day 1-2: Unit test implementation
- Day 3: Performance testing
- Day 4: User documentation
- Day 5: Developer documentation

## Success Criteria

### Functional Requirements
- ✅ Ctrl+M cycles through three states
- ✅ All metrics have tooltips with explanations
- ✅ Dimension context clearly visible
- ✅ Scene loading progress shown
- ✅ Network/memory breakdown available
- ✅ Zarr compression stats displayed
- ✅ Recommendations are specific and actionable

### Non-Functional Requirements
- ✅ 100% UI_DESIGN.md compliance
- ✅ <50ms UI update latency
- ✅ No memory leaks
- ✅ 80% test coverage
- ✅ Complete documentation

### Quality Metrics
- Code follows TypeScript best practices
- All components properly typed
- Error boundaries prevent crashes
- Graceful degradation without data
- Accessible via keyboard navigation

## Risk Mitigation

### Performance Impact
- **Risk**: Monitor impacts main app performance
- **Mitigation**: Use requestAnimationFrame, batch updates, lazy rendering

### Browser Compatibility
- **Risk**: Canvas features not supported
- **Mitigation**: Feature detection, fallback to HTML rendering

### Data Volume
- **Risk**: Too many events overwhelm monitor
- **Mitigation**: Ring buffer, event sampling, aggregation


This comprehensive plan ensures the Data Loading Monitor becomes a powerful, professional tool that provides genuine value for understanding and optimizing Luxar's complex data loading architecture.