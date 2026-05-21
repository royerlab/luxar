# Luxar Profiling Package

Hierarchical timing instrumentation for the scene update pipeline with EMA smoothing and session-based nesting.

## Overview

The Profiling package provides performance profiling for data loading and rendering operations. It measures timing of each update cycle, tracks per-loader metadata (visible counts, skip reasons), and displays results in the UI.

### Key Features

- **Hierarchical Timing**: Parent-child relationships built via explicit session passing (`session.begin(name)` / `timeWithMeta`)
- **Exponential Moving Average**: Stable timing averages with EMA (alpha=0.1)
- **Ambient root context**: `time()`/`begin()` attach to the root session — handy for flat top-level entries
- **Concurrent-safe top-level entries**: `timeTopLevel()` is safe under `Promise.all` for parallel loader updates
- **Metadata Tracking**: Points, segments, splats, skip flags, plus optional chunk / cache / info fields on `TimingMetadata`
- **UI Integration**: `DataLoadingMonitor` displays the timing panel

## Quick Start

```typescript
import { UpdateProfiler } from './update-profiler';

// Initialize
const profiler = new UpdateProfiler();

// In your update function
profiler.beginUpdate();

try {
  // Top-level entry under the root — safe for Promise.all
  const data = await profiler.timeTopLevel('Points (/scene/foo)', async (session) => {
    // Use the passed session to nest child entries
    const result = await session.begin('Load Arrays');
    // ... do work ...
    result.end();
    session.setMetadata({ points: 12345 });
    return this.loadData();
  });

  // Flat top-level entry with metadata
  await profiler.timeWithMeta('Process', (session) => {
    session.setMetadata({ info: `count=${data.length}` });
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

| Function               | Purpose                                                      |
| ---------------------- | ------------------------------------------------------------ |
| `formatMs(ms)`         | Format milliseconds for display                              |
| `hasOverBudget(entry)` | Check if entry or children exceed the 60fps budget (16.67ms) |

## Session Model

The profiler tracks a single ambient context — the root session created by
`beginUpdate()`. `time()`, `begin()`, `timeWithMeta()`, and `skip()` all
attach their entries as direct children of that root. **`time()` does NOT
push/pop the context**, so a nested `profiler.time('Inner', …)` inside
`profiler.time('Outer', …)` registers `Inner` as a _sibling_ of `Outer`,
not a child (pinned by `update-profiler.test.ts`).

To build a true hierarchy, pass a session explicitly and call
`session.begin(childName)`:

```typescript
profiler.beginUpdate();
const points = profiler.beginTopLevel('Points (/scene/nuclei)');
const query = points.begin('Spatial Query');
// ... do work ...
query.end();
points.setMetadata({ points: 50000 });
points.end();
profiler.endUpdate();
```

`timeTopLevel(name, fn)` is the recommended sugar — it creates a direct
child of root, passes the session to `fn`, and is the only entry point
that is safe under `Promise.all` (each call tracks its own ID, so
concurrent top-level operations do not corrupt one another).

## Timing Hierarchy

A typical scene update produces this structure (each per-loader subtree
is built by handlers receiving the `timeTopLevel` session):

```
Total Update                              [root]
+-- Points (/scene/nuclei)                [per-loader, timeTopLevel]
|   +-- Spatial Query                     [child of Points session]
|   +-- Load Arrays
|   +-- Project to 3D
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
+-- [Skipped: /scene/detector]            [skip(): 0-duration, skipReason]
```

## Metadata Per Entry

`TimingMetadata` fields (all optional): `chunks`, `cacheHits`,
`cacheMisses`, `points`, `segments`, `splats`, `skipped`, `skipReason`,
`info`.

| Entry Type      | Metadata Fields typically set |
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

### With SceneLoaderManager

`SceneLoaderManager` owns the singleton `UpdateProfiler` and passes it
positionally into every `SceneLoader` it constructs:

```typescript
// SceneLoaderManager constructor
this.profiler = new UpdateProfiler();

// Per loader
const loader = new SceneLoader(config, id, this.profiler, this.monitorFactory);
```

Consumers reach the profiler via `SceneLoaderManager.getInstance().getProfiler()`.

### With DataLoadingMonitor

```typescript
// Connect profiler to UI (called by the app bootstrap)
this.monitor.setProfiler(this.profiler);
```

`DataLoadingMonitor` calls `profiler.getTimings()` to render the Performance tab.

---

## Contents

- `update-profiler.ts` — `UpdateProfiler`, `RootSession`, the
  `TimingEntry` / `TimingMetadata` / `UpdateSession` interfaces, plus the
  `formatMs` and `hasOverBudget` helpers.

## Public Exports

- `class UpdateProfiler`, `class RootSession`
- Interfaces: `UpdateSession`, `TimingEntry`, `TimingMetadata`
- Functions: `formatMs(ms)`, `hasOverBudget(entry)`

## Dependencies

- Internal: `../utils/log` (for `log.warning` / `log.error` on session
  misuse and listener errors).
