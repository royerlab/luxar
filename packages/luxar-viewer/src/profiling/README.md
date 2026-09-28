# Luxar Profiling Package

Hierarchical timing instrumentation for the scene update pipeline with EMA smoothing and session-based nesting.

## Overview

The Profiling package provides performance profiling for data loading and rendering operations. It measures timing of each update cycle, tracks per-loader metadata (visible counts, skip reasons), and displays results in the UI.

### Key Features

- **Hierarchical Timing**: Parent-child relationships built via explicit session passing (`session.begin(name)` / `timeWithMeta`)
- **Exponential Moving Average**: Stable timing averages with EMA (alpha=0.1)
- **Ambient root context**: `time()`/`begin()` attach to the root session — handy for flat top-level entries
- **Concurrent-safe top-level entries**: `timeTopLevel()` is safe under `Promise.all` for parallel loader updates
- **Metadata Tracking**: Points, segments, splats, geometry-neutral elements, skip flags, plus optional chunk / cache / info fields on `TimingMetadata`
- **UI Integration**: `DataLoadingMonitor` displays the timing panel

## Load timeline (`load-timeline.ts`)

Wall-clock milestones of one scene load, recorded once per load as
`performance.mark('luxar:<name>')` entries (visible in the DevTools Performance
panel) and mirrored into an allocation-free in-memory snapshot read by
`__luxarDebug.getPerf()`:

| Mark                 | Recorded at                                                |
| -------------------- | ---------------------------------------------------------- |
| `loadStart`          | `loadScene` entry (resets the timeline)                    |
| `metadataReady`      | root group opened (consolidated metadata parsed)           |
| `poolReady`          | data-worker pool initialized                               |
| `wasmReady`          | shared WASM module compiled                                |
| `firstCommit:<kind>` | first geometry commit per kind (points/lines/gsplats/mesh) |
| `sceneLoaded`        | every eager node committed (`Scene loaded successfully`)   |
| `initUpdateDone`     | first `updateAllNDNodes` after load resolved               |
| `refinementComplete` | final refinement phase ran every ladder to completion      |

`getLoadTimeline().measures` derives `ttfpMs` (earliest first commit),
`sceneLoadedMs`, `initUpdateDoneMs`, `refinementCompleteMs`, … relative to
`loadStart`; `refinement` counts passes and rungs. Everything is best-effort:
a runtime without `performance.mark` still gets the snapshot, and nothing
throws into the load path.

## Perf counters (`perf-counters.ts`)

Always-on tallies for performance probes and the render gate
(`scripts/render-gate/`): bytes uploaded, renders, decodes, requests, console
calls, and so on. A hot path resolves its slot once at module scope
(`const S = perfCounters.slot('gpu.uploadBytes')`) and then pays one typed-array
store per event (`perfCounters.add(S, n)`); `max` keeps a high-water mark and
`gauge` overwrites. `record(kind, rec)` appends to a bounded per-kind ring
(4096) for per-tick traces.

Read them through `__luxarDebug.getPerf().counters` (a flat `name -> number`
map, available from bootstrap on), `__luxarDebug.getPerfRecords(kind)`, and
reset with `__luxarDebug.resetPerfCounters()`. Counters are the base for every
gated perf comparison, so a counter must exist on both the base and the
candidate build before a commit can be judged on it.

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

The profiler tracks a single ambient context, `currentSessionContext`,
seeded with the root session by `beginUpdate()`. `begin()`,
`timeWithMeta()`, and `skip()` attach their entries as children of
whatever the current context is. **`time()` saves and restores the
context around its callback** — it sets the current context to its own
session before running `fn`, then restores the previous context after.
So a nested `profiler.time('Inner', …)` inside
`profiler.time('Outer', …)` registers `Inner` as a _child_ of `Outer`
(pinned by `update-profiler.test.ts`'s "nested time() builds a
parent/child hierarchy via currentSessionContext push/pop"). For async
callbacks `time()` restores the context immediately after `fn` returns
the promise, so sibling synchronous code at the caller's level does not
see the in-flight session as parent.

To build a top-level subtree explicitly, pass a session around and call
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
is built by `run-loader-updates.ts`, which opens one `beginTopLevel`
session per node and keeps it alive across the atomic commit so the
per-node child entries nest under it):

```
Total Update                              [root]
+-- Points (/scene/nuclei)                [per-loader, beginTopLevel]
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

Each finished session merges into the persistent tree by its full PATH
(root name + every ancestor's name + its own), resolved through a
path-keyed `Map` index. Loaders share child names (`Spatial Query`,
`Load Arrays`, …), so a merge keyed on the parent's name alone would land
a second loader's grandchildren under the first loader's row; the path key
rules that out, and makes each merge O(1) instead of a tree search, so an
update with N loaders costs O(N) to merge rather than O(N²).

## Metadata Per Entry

`TimingMetadata` fields (all optional): `chunks`, `cacheHits`,
`cacheMisses`, `points`, `segments`, `splats`, `elements`, `skipped`,
`skipReason`, `info`.

| Entry Type      | Metadata Fields typically set |
| --------------- | ----------------------------- |
| Total Update    | (none)                        |
| Points          | `points` (visible count)      |
| Lines           | `segments` (visible count)    |
| GSplats         | `splats` (visible count)      |
| Depth Sort      | `elements` (sorted count)     |
| Skipped entries | `skipped: true`, `skipReason` |
| Sub-operations  | (none, time is the metric)    |

`elements` is the geometry-NEUTRAL count, used by passes that serve every
geometry type through one machinery — the depth-sort rows sort points,
line segments, Gaussian splats or mesh triangles, so a per-type tag there
would be wrong (or not even well defined when the root aggregates sorts
across nodes of different types).

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

Merging is linear in the number of entries. A whole steady-state update
(sessions plus merges, 4 entries per loader) takes about 0.2 ms with 200
loaders and 1.3 ms with 1000, measured in Node; with the per-name tree
search this replaced it took 1.0 ms and 28 ms. The
`profiler.mergeMs` perf counter tracks it live.

## Integration

### With SceneLoaderManager

`SceneLoaderManager` owns the singleton `UpdateProfiler` and passes it
positionally into every `SceneLoader` it constructs:

```typescript
// SceneLoaderManager constructor
this.profiler = new UpdateProfiler();

// Per loader
const loader = new SceneLoader(
  config,
  id,
  this.profiler,
  this.monitorFactory,
  this.lodGroupRegistryFactory
);
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
