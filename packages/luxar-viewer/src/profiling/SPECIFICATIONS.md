# Update Profiler Specifications

## 1. Overview

The Update Profiler provides hierarchical timing instrumentation for the scene update pipeline. It measures the duration of each step when a view change occurs (slider move, dimension navigation), aggregates results using exponential moving average, and provides data for UI display.

### 1.1 Goals

1. **Low overhead**: Just `performance.now()` calls and object updates
2. **Hierarchical**: Parent-child timing relationships (tree structure)
3. **Persistent**: Timing history survives across updates (needed for EMA)
4. **Observable**: Listener pattern for UI updates
5. **Informative**: Full breakdown from top-level to leaf operations

### 1.2 Non-Goals

- Profiling rendering (that's THREE.js territory)
- Profiling outside update cycles (initialization, etc.)
- GPU timing (requires WebGL queries, different approach)

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              App                                         │
│  ┌─────────────────┐                                                    │
│  │ UpdateProfiler  │◄──────────────────────────────────────────────┐    │
│  │   (singleton)   │                                               │    │
│  └────────┬────────┘                                               │    │
│           │ injected                                               │    │
│           ▼                                                        │    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐ │    │
│  │  SceneLoader    │───►│PointSpatialIndex│───►│ ChunkLoader    │ │    │
│  │                 │    │     Loader      │    │                │ │    │
│  └─────────────────┘    └─────────────────┘    └────────────────┘ │    │
│           │                     │                     │            │    │
│           │ session.begin()     │ session.begin()     │            │    │
│           │ session.end()       │ session.end()       │            │    │
│           └─────────────────────┴─────────────────────┘            │    │
│                                                                    │    │
│  ┌─────────────────┐                                               │    │
│  │DataLoadingMonitor│◄─── profiler.addListener() ──────────────────┘    │
│  │ (Performance Tab)│                                                   │
│  └─────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User moves slider
       │
       ▼
SceneLoader.updateView()
       │
       ├─► profiler.beginUpdate()
       │         │
       │         ▼
       │   RootSession created
       │   startTime = performance.now()
       │
       ├─► For each Points loader:
       │     session.begin('Points (/scene/nuclei)')
       │         │
       │         ├─► session.begin('Spatial Query')
       │         │   ... chunk index query ...
       │         │   session.end() ─► duration recorded
       │         │
       │         ├─► session.begin('Load Arrays')
       │         │   ... load positions, colors, radii, sharpness ...
       │         │   session.end() ─► duration recorded
       │         │
       │         └─► session.begin('Project to 3D')
       │             ... nD visibility + 3D projection ...
       │             session.end() ─► duration recorded
       │
       │     session.setMetadata({ points: 50000 })
       │     session.end()
       │
       ├─► For each Lines loader:
       │     session.begin('Lines (/scene/tracks)')
       │         │
       │         ├─► session.begin('Spatial Query')
       │         ├─► session.begin('Load Segments')
       │         ├─► session.begin('Load Vertices')
       │         └─► session.begin('Project to 3D')
       │
       │     session.setMetadata({ segments: 10000 })
       │     session.end()
       │
       ├─► For each GSplats loader:
       │     session.begin('GSplats (/scene/gaussians)')
       │         │
       │         ├─► session.begin('Spatial Query')
       │         └─► session.begin('Load Arrays')
       │
       │     session.setMetadata({ splats: 5000 })
       │     session.end()
       │
       └─► session.end()
                 │
                 ▼
           _mergeEntry() called
           EMA updated for all entries
           notifyListeners()
                 │
                 ▼
           DataLoadingMonitor.onProfilerUpdate()
                 │
                 ▼
           UI re-rendered with new timing data
```

---

## 3. Timing Hierarchy

### 3.1 Current Instrumentation Tree

```
Total Update                              [root]
├── Points (/scene/nuclei)                [per-loader]
│   ├── Spatial Query                     [chunk index query]
│   ├── Load Arrays                       [fetch + decode positions, colors, etc.]
│   └── Project to 3D                     [nD visibility + projection]
│
├── Lines (/scene/tracks)                 [per-loader]
│   ├── Spatial Query                     [chunk index query]
│   ├── Load Segments                     [fetch segment indices]
│   ├── Load Vertices                     [fetch vertex data]
│   └── Project to 3D                     [nD clipping + projection]
│
├── GSplats (/scene/gaussians)            [per-loader]
│   ├── Spatial Query                     [chunk index query]
│   └── Load Arrays                       [fetch + decode splat data]
│
└── [Skipped: /scene/detector]            [extend_to_all optimization]
```

**Notes**:

- Operations run concurrently for different loaders using `timeTopLevel()`
- Each loader type (Points, Lines, GSplats) has its own instrumentation
- Skipped loaders show `skipped` metadata with reason (e.g., "extend_to_all")
- Metadata (point/segment/splat counts) is set after successful load

### 3.2 Metadata Per Entry

| Entry Type      | Metadata Fields               |
| --------------- | ----------------------------- |
| Total Update    | (none)                        |
| Points          | `points` (visible count)      |
| Lines           | `segments` (visible count)    |
| GSplats         | `splats` (visible count)      |
| Skipped entries | `skipped: true`, `skipReason` |
| Sub-operations  | (none, time is the metric)    |

---

## 4. API Specification

### 4.1 UpdateProfiler Class

The profiler supports two usage patterns:

1. **Context/ambient pattern (recommended)**: Use `time()` helper, profiler manages session stack
2. **Manual session management**: Use `beginUpdate()`/`endUpdate()` with explicit sessions

```typescript
class UpdateProfiler {
  // === Context/Ambient Pattern (Recommended) ===

  /**
   * Begin a new update cycle (root session).
   * Call this at the start of updateView().
   * Initializes the session stack.
   */
  beginUpdate(): UpdateSession;

  /**
   * End the current update cycle.
   * Call this at the end of updateView().
   * Clears the session stack and notifies listeners.
   */
  endUpdate(): void;

  /**
   * Begin a child timing entry of the current innermost session.
   * Uses the session stack - no need to pass sessions around.
   * Returns a ScopedSession that auto-pops from stack on end().
   */
  begin(name: string): UpdateSession;

  /**
   * Convenience: run a function with timing.
   * Automatically handles begin/end and preserves return value.
   * Works with both sync and async functions.
   *
   * @example
   * const data = await profiler.time('Spatial Query', () =>
   *   this.queryChunks(viewState)
   * );
   */
  time<T>(name: string, fn: () => T): T;

  /**
   * Run a function with timing and access to session for metadata.
   *
   * @example
   * const data = await profiler.timeWithMeta('Chunk Load', (session) => {
   *   session.setMetadata({ chunks: ranges.length });
   *   return this.loadChunks(ranges);
   * });
   */
  timeWithMeta<T>(name: string, fn: (session: UpdateSession) => T): T;

  /**
   * Mark an operation as skipped.
   * Creates a timing entry with 0 duration and skip reason.
   *
   * @example
   * profiler.skip('Points (/scene/nuclei)', 'extend_to_all');
   */
  skip(name: string, reason: string): void;

  // === State & Listeners ===

  /**
   * Check if profiling is currently active.
   */
  isActive(): boolean;

  /**
   * Get the current innermost session (for setting metadata).
   * Returns NoOpSession if no update is active.
   *
   * @example
   * profiler.current().setMetadata({ points: 50000 });
   */
  current(): UpdateSession;

  /**
   * Get the persistent timing hierarchy for UI display.
   * Returns the root TimingEntry with all children.
   */
  getTimings(): TimingEntry;

  /**
   * Reset all timing data (clears history and averages).
   */
  reset(): void;

  /**
   * Add a listener for timing updates.
   * Called when endUpdate() completes.
   */
  addListener(listener: () => void): void;

  /**
   * Remove a listener.
   */
  removeListener(listener: () => void): void;
}
```

### 4.2 UpdateSession Interface

```typescript
interface UpdateSession {
  /**
   * Begin a child timing entry.
   * Returns a new session for the child.
   */
  begin(name: string): UpdateSession;

  /**
   * End this timing entry.
   * Records duration, updates EMA, triggers merge into persistent state.
   */
  end(): void;

  /**
   * Set metadata on this entry.
   * Can be called multiple times (merges).
   */
  setMetadata(meta: Partial<TimingMetadata>): void;

  /**
   * Mark this entry as skipped (e.g., extend_to_all optimization).
   * Sets duration to 0, skipped=true in metadata.
   */
  markSkipped(reason: string): void;
}
```

### 4.3 TimingEntry Interface

```typescript
interface TimingEntry {
  /** Display name (e.g., "Points (/scene/nuclei)") */
  name: string;

  /** Last measured duration in milliseconds */
  lastMs: number;

  /** Exponential moving average in milliseconds */
  avgMs: number;

  /** Number of measurements (for debugging) */
  count: number;

  /** Child timing entries */
  children: TimingEntry[];

  /** Optional metadata */
  metadata?: TimingMetadata;

  /** Whether lastMs exceeds 60fps budget (16.67ms) */
  overBudget?: boolean;
}

interface TimingMetadata {
  chunks?: number;
  cacheHits?: number;
  cacheMisses?: number;
  points?: number;
  segments?: number;
  splats?: number;
  bytes?: number;
  count?: number;
  skipped?: boolean;
  skipReason?: string;
  info?: string;
}
```

---

## 5. Implementation Details

### 5.1 Exponential Moving Average

```typescript
const EMA_ALPHA = 0.1; // Slow adaptation, stable averages

function updateEMA(current: number, newValue: number, count: number): number {
  if (count === 1) {
    // First measurement: just use the value
    return newValue;
  }
  // EMA formula: new = α * value + (1 - α) * old
  return EMA_ALPHA * newValue + (1 - EMA_ALPHA) * current;
}
```

**Rationale**: With α=0.1, it takes ~10 measurements for a change to be 65% reflected in the average, ~23 measurements for 90%. This provides stability while still responding to sustained changes.

### 5.2 Session Lifecycle

```
beginUpdate()
     │
     ▼
┌─────────────┐
│ RootSession │ ◄── startTime = performance.now()
│  (active)   │
└──────┬──────┘
       │
       │ session.begin('Points')
       ▼
┌─────────────┐
│ChildSession │ ◄── startTime = performance.now()
│  (active)   │     parent = RootSession
└──────┬──────┘
       │
       │ session.end()
       ▼
┌─────────────┐
│ChildSession │ ◄── duration = now - startTime
│  (ended)    │     entry merged into parent.children
└─────────────┘     EMA updated
       │
       │ (back in RootSession)
       │ session.end()
       ▼
┌─────────────┐
│ RootSession │ ◄── duration = now - startTime
│  (ended)    │     entry merged into profiler.rootEntry
└─────────────┘     notifyListeners()
```

### 5.3 Entry Merging Strategy

When a session ends, its entry is merged into the persistent state:

1. **Find existing entry** by name path (e.g., "Total Update/Points (/scene/nuclei)")
2. **If found**: Update lastMs, avgMs, count, metadata, children
3. **If not found**: Clone entry and add to parent's children

This allows the timing tree to grow dynamically as new code paths are instrumented.

### 5.4 Concurrency Handling

Updates run in parallel (Promise.all for Points/Lines/GSplats). Sessions are safe for concurrent use because:

1. Each loader gets its own child session
2. Sessions only write to their own entry
3. Merge happens in end() which is called sequentially within each loader
4. Final root merge happens after Promise.all completes

```typescript
// Safe concurrent usage:
const pointsUpdates = loaders.map(async ([path, loader]) => {
  const session = rootSession.begin(`Points (${path})`);
  // ... async work ...
  session.end(); // Safe: only touches this session's entry
});

await Promise.all(pointsUpdates);
rootSession.end(); // Safe: all children already merged
```

---

## 6. Integration Points

### 6.1 App Initialization

```typescript
// In App constructor or initialization
this.profiler = new UpdateProfiler();

// Pass to SceneLoader
this.sceneLoader = new SceneLoader(store, {
  profiler: this.profiler,
  // ... other options
});

// Connect to DataLoadingMonitor
this.monitor.setProfiler(this.profiler);
```

### 6.2 SceneLoader Integration (Option C - Context/Ambient Pattern)

The profiler uses a session stack, so nested `time()` calls automatically create parent-child relationships without passing sessions around.

```typescript
// In SceneLoader.updateView()
async updateView(viewState: Partial<ViewState>): Promise<void> {
  // Start the update cycle
  this.profiler.beginUpdate();

  try {
    // Points updates - profiler.time() nests under current session
    const pointsUpdates = Array.from(this.loaders.entries()).map(
      async ([path, loader]) => {
        await this.profiler.time(`Points (${path})`, async () => {
          // Check if we should skip (extend_to_all)
          const isSkipped = this.checkExtendToAll(path);
          if (isSkipped) {
            this.profiler.skip('Skip Check', 'extend_to_all');
            return;
          }

          // Query - nested under "Points (/path)"
          const ranges = await this.profiler.time('Spatial Query', () =>
            loader.queryRanges(viewState)
          );

          // Load with metadata
          const data = await this.profiler.timeWithMeta('Chunk Load', (session) => {
            session.setMetadata({ chunks: ranges.length });
            return loader.loadChunks(ranges);
          });

          // GPU Upload
          await this.profiler.time('GPU Upload', () =>
            this.updatePointsGeometry(path, data)
          );

          // Add final metadata to the current Points session
          this.profiler.current().setMetadata({ points: data.metadata.loadedPoints });
        });
      }
    );

    // Similar for Lines and GSplats...

    await Promise.all([...pointsUpdates, ...linesUpdates, ...gsplatsUpdates]);

  } finally {
    // End the update cycle - triggers listeners
    this.profiler.endUpdate();
  }
}
```

### 6.3 Loader Integration (Chunk Load Breakdown)

Loaders receive the profiler reference and use `time()` for clean nesting:

```typescript
// In PointSpatialIndexLoader
class PointSpatialIndexLoader {
  constructor(private profiler: UpdateProfiler) {}

  async loadChunks(ranges: PointRange[]): Promise<PointsData> {
    let l1Hits = 0,
      l2Hits = 0,
      networkFetches = 0;

    for (const range of ranges) {
      const key = this.getCacheKey(range);

      // Try L1 - each time() call nests under parent "Chunk Load"
      const l1Result = this.profiler.time('L1 Cache', () => this.l1Cache.get(key));
      if (l1Result) {
        l1Hits++;
        continue;
      }

      // Try L2
      const l2Result = await this.profiler.time('L2 Cache', () => this.l2Cache.get(key));
      if (l2Result) {
        l2Hits++;
        continue;
      }

      // Network fetch
      await this.profiler.time('Network Fetch', () => this.fetchFromNetwork(range));
      networkFetches++;
    }

    // Set metadata on the current session (the "Chunk Load" session)
    this.profiler.current().setMetadata({
      cacheHits: l1Hits + l2Hits,
      cacheMisses: networkFetches,
    });

    // ... rest of loading
  }
}
```

**Note**: The `time()` helper handles both sync and async functions automatically. Promises are handled with `.finally()` to ensure timing ends even if the function throws.

### 6.4 GPU Upload Breakdown

```typescript
// In SceneLoader.updatePointsGeometry()
private updatePointsGeometry(
  path: string,
  data: PointsData,
  session?: UpdateSession
): void {
  if (this._gpuBufferPool) {
    // Pool acquire
    const acquireSession = session?.begin('Pool Acquire');
    const geometry = this._gpuBufferPool.acquirePointsGeometry(path, data, count);
    acquireSession?.end();

    // TypedArray copy
    const copySession = session?.begin('TypedArray Copy');
    this._gpuBufferPool.updatePointsGeometry(geometry, data, count);
    copySession?.end();

    // Bounding box
    const bboxSession = session?.begin('Bounding Box');
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    bboxSession?.end();

    points.geometry = geometry;
  }
}
```

### 6.5 DataLoadingMonitor Integration

```typescript
// In DataLoadingMonitor
private profiler: UpdateProfiler | null = null;
private profilerListener: (() => void) | null = null;

setProfiler(profiler: UpdateProfiler | null): void {
  // Remove old listener
  if (this.profiler && this.profilerListener) {
    this.profiler.removeListener(this.profilerListener);
  }

  this.profiler = profiler;

  // Add new listener
  if (profiler) {
    this.profilerListener = () => this.onProfilerUpdate();
    profiler.addListener(this.profilerListener);
  }
}

private onProfilerUpdate(): void {
  // Only update if visible and on performance tab
  if (this.uiState.isVisible && this.uiState.activeTab === 'performance') {
    this.updatePerformanceTab();
  }
}

private renderPerformanceTab(): string {
  if (!this.profiler) {
    return '<div class="timing-empty">Profiler not connected</div>';
  }

  const timings = this.profiler.getTimings();
  return renderHierarchicalTimingPanel(timings);
}
```

---

## 7. UI Specification

### 7.1 Panel Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Operation                                      Last      Avg    │
├─────────────────────────────────────────────────────────────────┤
│ ▼ Total Update                               47.3ms   52.1ms   │
│   ▼ Points (/scene/nuclei)     50K pts       32.1ms   35.4ms   │
│     ├─ Skip Check                             0.1ms    0.1ms   │
│     ├─ Spatial Query           5 chunks       2.3ms    2.8ms   │
│     ▼ Chunk Load               80% cache     18.2ms   20.1ms   │
│       ├─ L1 Cache              3 hits         0.1ms    0.1ms   │
│       ├─ L2 Cache              1 hit          0.8ms    0.9ms   │
│       └─ Network Fetch         1 req        17.3ms   19.1ms   │ ← RED
│     ├─ Decompress                             4.2ms    5.1ms   │
│     ├─ Accumulate                             1.1ms    1.2ms   │
│     ▼ GPU Upload                              6.3ms    6.2ms   │
│       ├─ Pool Acquire                         0.1ms    0.1ms   │
│       ├─ TypedArray Copy                      2.8ms    2.9ms   │
│       └─ Bounding Box                         3.4ms    3.2ms   │
│   ► Lines (/scene/tracks)      12K segs      12.4ms   13.8ms   │
│   ─ Skip: extend_to_all (/scene/detector)       —        —     │
├─────────────────────────────────────────────────────────────────┤
│ ● Normal  ● >16ms (60fps)  ○ Skipped          42 updates       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Visual Indicators

| Indicator         | Meaning                               |
| ----------------- | ------------------------------------- |
| ▼ / ►             | Expanded / Collapsed (has children)   |
| RED background    | Entry exceeds 16ms (60fps budget)     |
| YELLOW background | Child exceeds budget (warning)        |
| GRAY / italic     | Skipped entry                         |
| Tags after name   | Metadata (points, chunks, cache rate) |

### 7.3 Interactions

- **Click ▼/►**: Toggle expand/collapse
- **Hover row**: Highlight row
- **Click "Reset"** (optional): Clear timing history

---

## 8. Performance Considerations

### 8.1 Overhead Analysis

| Operation             | Cost  | Frequency               |
| --------------------- | ----- | ----------------------- |
| `performance.now()`   | ~1μs  | 2 per entry (begin/end) |
| Object allocation     | ~10μs | 1 per entry per update  |
| EMA calculation       | ~1μs  | 1 per entry per update  |
| Entry merge           | ~5μs  | 1 per entry per update  |
| Listener notification | ~1μs  | 1 per update            |

**Total overhead per update** (assuming 20 timing entries):

- 20 × 2 × 1μs (performance.now) = 40μs
- 20 × 10μs (allocations) = 200μs
- 20 × 6μs (EMA + merge) = 120μs
- 1μs (notification)

**Total: ~360μs per update** (0.36ms)

This is <1% of a typical 50ms update cycle, well within acceptable overhead.

### 8.2 Memory Usage

- Each TimingEntry: ~200 bytes
- Typical tree with 30 entries: ~6KB
- No unbounded growth (tree structure is fixed by code paths)

---

## 9. Testing Strategy

### 9.1 Unit Tests

```typescript
describe('UpdateProfiler', () => {
  it('should record timing hierarchy', () => {
    const profiler = new UpdateProfiler();
    const session = profiler.beginUpdate();

    const child = session.begin('Child');
    // Simulate work
    await sleep(10);
    child.end();

    session.end();

    const timings = profiler.getTimings();
    expect(timings.lastMs).toBeGreaterThan(10);
    expect(timings.children[0].name).toBe('Child');
  });

  it('should calculate EMA correctly', () => {
    const profiler = new UpdateProfiler();

    // First update: 100ms
    let session = profiler.beginUpdate();
    await sleep(100);
    session.end();
    expect(profiler.getTimings().avgMs).toBeCloseTo(100, -1);

    // Second update: 0ms (simulated)
    // EMA should be: 0.1 * 0 + 0.9 * 100 = 90
    // (actual test would need mocking)
  });

  it('should handle skipped entries', () => {
    const profiler = new UpdateProfiler();
    const session = profiler.beginUpdate();

    const child = session.begin('Skipped');
    child.markSkipped('extend_to_all');
    child.end();

    session.end();

    const timings = profiler.getTimings();
    expect(timings.children[0].metadata?.skipped).toBe(true);
    expect(timings.children[0].lastMs).toBe(0);
  });
});
```

### 9.2 Integration Tests

- Test with real SceneLoader update cycle
- Verify timing tree structure matches expected hierarchy
- Verify UI renders without errors

---

## 10. Migration Plan

### 10.1 Phase 1: Core Infrastructure

1. ✅ Create `UpdateProfiler` class
2. ✅ Create `HierarchicalTimingPanel` component
3. Create CSS styles for timing panel
4. Write unit tests for profiler

### 10.2 Phase 2: Basic Integration

1. Add profiler to App, inject into SceneLoader
2. Add top-level instrumentation (per-loader only)
3. Wire up DataLoadingMonitor to profiler
4. Replace Performance tab with timing panel

### 10.3 Phase 3: Full Instrumentation

1. Add loader-level breakdown (Query, Load, GPU Upload)
2. Add cache breakdown (L1, L2, Network)
3. Add GPU upload breakdown (Pool, Copy, BBox)
4. Add metadata to all entries

### 10.4 Phase 4: Cleanup

1. Remove PerformanceTimeline class
2. Remove timeline-related code from DataLoadingMonitor
3. Remove unused TimelinePoint handling
4. Update documentation

---

## 11. Future Enhancements

### 11.1 Potential Additions

- **Export to JSON**: Download timing data for analysis
- **Comparison mode**: Show delta from previous session
- **Flame graph view**: Alternative visualization
- **GPU timing**: WebGL query-based GPU profiling
- **Automated recommendations**: "Network is bottleneck, consider prefetching"

### 11.2 Configuration Options

```typescript
interface ProfilerConfig {
  enabled: boolean; // Master switch
  emaAlpha: number; // EMA smoothing factor (default 0.1)
  budgetMs: number; // Frame budget threshold (default 16.67)
  maxDepth: number; // Max instrumentation depth (default unlimited)
  excludePatterns: string[]; // Patterns to exclude from timing
}
```

---

## Changelog

- **2025-12-26**: Initial specification
  - Defined API and data structures
  - Documented full instrumentation hierarchy
  - Specified integration points
  - Outlined implementation phases
