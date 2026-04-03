# Luxar Profiling Package

Hierarchical timing instrumentation for the scene update pipeline with EMA smoothing and session-based nesting.

## Overview

The Profiling package provides performance profiling for data loading and rendering operations. It measures timing of each update cycle, tracks metrics like cache hits/misses, and displays results in the UI.

### Key Features

- **Hierarchical Timing**: Parent-child relationships with automatic nesting
- **Exponential Moving Average**: Stable timing averages with EMA (alpha=0.1)
- **Session Stack**: Ambient pattern for clean nesting without passing sessions around
- **Concurrent-Safe**: Handles parallel loader updates automatically
- **Metadata Tracking**: Points, segments, cache hits, bytes, and custom info
- **UI Integration**: DataLoadingMonitor displays timing panel

## Quick Start

```typescript
import { UpdateProfiler } from './update-profiler';

// Initialize
const profiler = new UpdateProfiler();

// In your update function
profiler.beginUpdate();

try {
  // Automatic nesting via session stack
  const data = await profiler.time('Load Data', async () => {
    return this.loadData();
  });

  // With metadata
  await profiler.timeWithMeta('Process', (session) => {
    session.setMetadata({ count: data.length });
    return this.processData(data);
  });
} finally {
  profiler.endUpdate(); // Triggers listeners, updates UI
}
```

## API Overview

| Method                   | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `beginUpdate()`          | Start timing cycle (root session)                            |
| `endUpdate()`            | End cycle, notify listeners                                  |
| `time(name, fn)`         | Time a function with automatic nesting                       |
| `timeWithMeta(name, fn)` | Time with metadata callback                                  |
| `timeTopLevel(name, fn)` | Time a top-level parallel operation (safe for `Promise.all`) |
| `begin(name)`            | Start manual timing entry (child of current)                 |
| `beginTopLevel(name)`    | Start timing entry directly under root                       |
| `skip(name, reason)`     | Mark operation as skipped                                    |
| `isActive()`             | Check if profiling is active                                 |
| `current()`              | Get current innermost session                                |
| `getTimings()`           | Get timing hierarchy for UI                                  |
| `addListener(fn)`        | Add update listener                                          |
| `removeListener(fn)`     | Remove update listener                                       |
| `reset()`                | Clear all timing data                                        |

### Utility Functions

| Function               | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `formatMs(ms)`         | Format milliseconds for display               |
| `hasOverBudget(entry)` | Check if entry or children exceed 16ms budget |

## Session Stack Pattern

Sessions automatically nest under the current session via stack:

```
beginUpdate()           <- Root session created, pushed to stack
  time('Query', ...)    <- Child of root
  time('Load', ...)     <- Child of root (same level)
    time('Decode', ...) <- Child of Load
  time('GPU', ...)      <- Child of root
endUpdate()             <- Stack cleared, listeners notified
```

No need to pass sessions around - the stack handles parent-child relationships.

## Timing Hierarchy

The profiler captures this structure during scene updates:

```
Total Update                              [root]
+-- Points (/scene/nuclei)                [per-loader]
|   +-- Spatial Query                     [chunk index query]
|   +-- Load Arrays                       [fetch + decode]
|   +-- Project to 3D                     [nD visibility + projection]
|
+-- Lines (/scene/tracks)                 [per-loader]
|   +-- Spatial Query
|   +-- Load Segments
|   +-- Load Vertices
|   +-- Project to 3D
|
+-- GSplats (/scene/gaussians)            [per-loader]
|   +-- Spatial Query
|   +-- Load Arrays
|
+-- [Skipped: /scene/detector]            [extend_to_all optimization]
```

## Metadata Per Entry

| Entry Type      | Metadata Fields               |
| --------------- | ----------------------------- |
| Total Update    | (none)                        |
| Points          | `points` (visible count)      |
| Lines           | `segments` (visible count)    |
| GSplats         | `splats` (visible count)      |
| Skipped entries | `skipped: true`, `skipReason` |
| Sub-operations  | (none, time is the metric)    |

## EMA Smoothing

The profiler uses Exponential Moving Average with alpha=0.1:

```typescript
avgMs = 0.1 * newValue + 0.9 * previousAvg;
```

This provides stability (takes ~10 measurements for 65% reflection of changes) while still responding to sustained changes.

## Performance Overhead

With ~20 timing entries per update:

- ~40us for `performance.now()` calls
- ~200us for object allocations
- ~120us for EMA + merge operations

**Total: ~360us per update** (<1% of typical 50ms update cycle)

## Integration

### With SceneLoader

```typescript
// App initialization
this.profiler = new UpdateProfiler();
this.sceneLoader = new SceneLoader(store, {
  profiler: this.profiler,
});
```

### With DataLoadingMonitor

```typescript
// Connect profiler to UI
this.monitor.setProfiler(this.profiler);
```

## Complete Documentation

See `SPECIFICATIONS.md` for:

- Full API specification
- Implementation details
- UI panel layout
- Integration examples
- Testing strategy
