# rendering/depth-sort-coordinator

> Main-thread side of the persistent depth-sort worker — camera-driven re-sorts (Phase 3) and cross-node back-to-front ordering

## Purpose

Depth sorting orders transparent elements back-to-front for correct order-dependent blending (normal and volumetric modes). **Within-node sorting** is driven by the `SortWorker` (Phase 2); **cross-node ordering** (THREE.js `renderOrder`) is driven here in Phase 3.

This module is the **main-thread authority** for the depth-sort subsystem. It serves **all four geometry types** via a geometry-agnostic mechanism: projected 3D centers in, back-to-front permutation out. The commit call sites (`commit-gsplats-geometry.ts`, `commit-points-geometry.ts`, `commit-lines-geometry.ts`, `commit-mesh-geometry.ts`) and the app lifecycle (init/dispose) are far apart, so all talk to this module instead of threading a coordinator object through constructors.

**There are two APPLY paths, and only the apply differs.** A "center" is a splat/point center, a line segment midpoint, or a triangle centroid — 3 floats either way, so registration, worker and kernel are shared:

|                  | Instanced (gsplats, points, lines)                             | Indexed (mesh)                                                                                              |
| ---------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| What is permuted | `aSortedIndex`, a per-instance draw-slot indirection           | `geometry.index` itself                                                                                     |
| Buffering        | Double-buffered pair + `uSortedIndexSlot` uniform              | Single buffer                                                                                               |
| Cadence          | Chunked: one 4 MB slice per rendered frame, flip on completion | Atomic: the whole visible prefix in one write                                                               |
| Why              | A half-written INACTIVE buffer is never drawn                  | A half-written index buffer is not a permutation — it would draw some triangles twice and others not at all |
| Owner            | `rendering/element-storage.ts`                                 | `triangle-ordering.ts`                                                                                      |

Mesh cannot double-buffer because `geometry.index` is _bound_ state: no uniform can select between two index buffers, and reassigning `geometry.index` is the drawn-geometry rebind `applyMeshIndices` exists to avoid. The atomic write costs one upload of a prefix the mesh path already re-uploads on every slice move. Its permutation is applied to the commit's CANONICAL triples (retained on `NodeSortState.triangleSource` only while the node is sorting), never to the live buffer, which would compose successive permutations.

## File Map

```
depth-sort-coordinator/
├── render-order.ts      — Cross-node renderOrder assignment: BSP partition traversal +
│                          centroid fallback, ONE global scale
└── triangle-ordering.ts — The INDEXED (mesh) apply: face centroids + atomic index
                           permutation + the draw-acknowledgement lifecycle
```

The main facade `rendering/depth-sort-coordinator.ts` owns:

- SortWorker spawn + initialization + disposal
- Per-node **generation** tracking (bumped on every non-noop commit)
- Single-in-flight-per-node sort rule + queue-exactly-one re-sort
- Ordering application: `writeSortedIndexOrdering` (element-storage) for the instanced types, `writeSortedTriangleOrdering` (`triangle-ordering.ts`) for mesh
- Per-frame camera-motion re-sort scheduler (angle/translation thresholds)
- Offline-capture entry point (`resortForCapture` / `isCaptureQuiescent`) — pose-fresh sort + drain to quiescence when the rAF loop is stopped
- Blending-mode switch hook (TO sorted: reprocess; AWAY: release)
- Node release (disposal / LOD demotion)

The submodule `render-order.ts` owns:

- Per-frame `renderOrder` assignment for every visible sorted-mode mesh
- BSP tree back-to-front traversal (exact, Fuchs–Kedem–Naylor)
- Centroid fallback (view-space z)
- One global scale (wrapper groups by mean view-z, parts by BSP rank or centroid)

The submodule `triangle-ordering.ts` owns:

- `computeFaceCentroids` — the mesh "center" payload for the shared kernel
- `writeSortedTriangleOrdering` — the atomic index permutation, with the rejections that keep a stale ordering from writing a corrupt one
- The written-but-undrawn acknowledgement map, so the profiler pass reports _uploaded_ only after THREE has drawn the permuted index (issue #713), mirroring `acknowledgeSortedIndexOrderingDraw`

## Architecture

### Module-Scoped Live Authority

Both files are **module-scoped singletons** (the `element-texture-layout.ts` pattern):

- Commit paths (all four geometry types) and app lifecycle are far apart
- All callers talk to this module via exported functions
- No coordinator object is threaded through constructors

**Module state** (depth-sort-coordinator.ts):

- `worker` — the single persistent SortWorker (spawned at app init by `warmUpDepthSortWorker()`, terminated on app dispose)
- `api` — Comlink-wrapped `SortWorkerAPI` (registerNode, sort, releaseNode, releaseAllNodes)
- `initPromise` — cached initialization promise (deduplicated). A cached REJECTION is dropped only by `maybeRetryStarvedWorkerInit()`, so a commit can never spend a retry attempt
- `initEpoch` — attempt epoch; every module write past an `await` inside the init closure is skipped when it no longer matches (bumped by `disposeDepthSort` and by a retry that supersedes an attempt, because the deadline timer lives in a closure and nothing outside it can cancel it)
- `workerInitState: 'idle' | 'ready' | 'starved' | 'failed'` — the observable init verdict (`getDepthSortWorkerStatus()`), and what `isDepthSortAvailable()` reads (false only at `'failed'` AND with a node that wants sorting)
- `initTimeoutRetryPending` / `initTimeoutCount` / `initRetryNotBeforeMs` / `initRetryWakeTimer` — starved-retry bookkeeping: whether a retry is armed, deadline misses so far (bounds the attempts), the `performance.now()` instant the next attempt may start, and the one self-wake `setTimeout` armed alongside it
- `nodeStates: Map<string, NodeSortState>` — per-node tracking (generation, in-flight flag, last sort pose, queued re-sort, registered flag)
- `nextGeneration` — monotonic counter (unique across a node's LIFETIMES, not just within one)
- Injected callbacks: `getCamera` (getter, not captured reference), `requestRender`, `requestReprocess`, `isLoadInProgress`, `getProfiler`
- Session master switch: `depthSortEnabled` (from `config.depthSort.enabled` + `?depthSort=0`)

**Module state** (render-order.ts):

- `partitionRankCache: Map<THREE.Object3D, Map<number, number> | null>` — per-frame BSP rank cache (cleared every frame)
- `warnedAxisMappingWrappers: WeakSet<THREE.Object3D>` — session-lifetime deduplication for wrappers whose BSP axes cannot be mapped through the live display dimensions
- `orderSlots: OrderSlot[]` — per-frame collect buffer (fresh array every frame)
- `scratch: RenderOrderScratch` — per-frame allocation-free scratch (lazily allocated on first use)

### SortWorker Lifecycle

1. **Warm up** — at app init, `warmUpDepthSortWorker()` (called from `core/app/init/pipeline.ts`, gated on `depthSortEnabled`) runs `ensureWorker()`:
   - Constructs `new SortWorker()` (Vite `?worker` import) or `new Worker(sortWorkerUrlOverride)` (embedder override)
   - Wraps via Comlink: `api = wrap<SortWorkerAPI>(worker)`
   - Calls `api.initialize(sortWorkerWasmPathOverride)` through the shared `worker-pool/lifecycle/init-with-guard.ts` (deadline + `onerror` + `onmessageerror`)
   - Logs ready message with `wasmFallback` flag (WASM compiled or TypeScript fallback)
   - **Why at init, not on first commit**: the first order-dependent commit lands exactly when this thread and the data-worker pool are saturated decoding the scene, and the worker's reply must be dispatched on this thread. At ~3 M points it missed a 30 s deadline there, and the session rendered unsorted from then on. Starting while the app is idle removes the overlap.
   - **What it costs**: the sort-worker chunk plus its WASM are fetched on EVERY page load, including scenes with no order-dependent geometry at all. `?depthSort=0` is the opt-out (nothing is spawned). The warm-up also no-ops where there is no `Worker` global (node/jsdom unit runs, SSR) rather than latching a `'failed'` verdict for the environment's sake; the commit path stays unguarded.
2. **Persist** — the worker lives across commits and frames (NOT part of the round-robin data-worker pool). `noteDepthSortCommit` still calls `ensureWorker()`; it dedupes on `initPromise` and covers embedders that configure late.
3. **Terminate** — on `disposeDepthSort()` (app teardown / test reset), or on ANY failed init attempt (a permanent one latches; a starved retry spawns a fresh worker).

**Init-settle guard**: A worker whose script dies during async module evaluation (before `expose()`) emits an `error` event but never settles the Comlink `initialize` RPC. Left pending forever, every order-dependent commit would attach a continuation (closing over its centers provider, which for points pins the full `LoadedPointsData`), accumulating unbounded. The shared `worker-pool/lifecycle/init-with-guard.ts` races the RPC against the `config.depthSort.workerInitTimeoutMs` deadline and the worker's own `onerror` / `onmessageerror` events, so the promise settles on every attempt, draining all queued continuations into the documented warn-once degrade path. (Sharing the guard with the data pool is what keeps the two startup paths from drifting; the coordinator's hand-rolled copy never had the `onmessageerror` arm.) Settling is not the same as giving up, though — how long that degrade lasts depends on WHICH failure it was:

**Failure taxonomy** (`ensureWorker()`), and why a deadline is not a death:

| trigger                                                      | verdict       | behaviour                                                              |
| ------------------------------------------------------------ | ------------- | ---------------------------------------------------------------------- |
| constructor throws (CSP-blocked script)                      | permanent     | terminate, cache the rejection, warn once                              |
| `onerror` / `onmessageerror` (404, module-evaluation death)  | permanent     | terminate, cache the rejection, warn once                              |
| `initialize()` rejects (the worker answered: it cannot work) | permanent     | terminate, cache the rejection, warn once                              |
| deadline missed                                              | **transient** | terminate, keep the rejection cached, respawn from the per-frame retry |

**The settle is unconditional; the degrade is CLASSIFIED** (issue #1694). Only the deadline arm rejects with `WorkerInitTimeoutError` (the guard's own type — the same reasoning `isWorkerInfrastructureError` already applies to RPC timeouts); a worker `error`/`messageerror` event, a rejected `initialize` and a constructor throw (CSP-blocked script) stay permanently fatal. A missed deadline says nothing about the worker's health — at ~3M points the main thread stays saturated long enough during a load for worker startup to lose the race — so it is retried up to `SORT_WORKER_INIT_MAX_ATTEMPTS` (3: the initial attempt plus 2) with a `SORT_WORKER_INIT_RETRY_BASE_MS × miss` backoff on the monotonic `performance.now()` clock. The bound is what preserves the anti-leak property: each attempt opens one more deadline window in which commits can queue continuations, and between attempts the cached rejection is still what every commit drains into — which is also why a commit re-entering `ensureWorker()` cannot spend an attempt. `maybeRetryStarvedWorkerInit()` runs from `evaluateDepthSortPerFrame`, where it is explicitly suppressed while `isLoadInProgress()` is true (the in-flight sweep IS the starvation), and declines to spend an attempt during an offline capture or when no visible, still-committed, order-dependent node wants sorting; since the on-demand loop idle-pauses after `config.animation.idleTimeoutMs`, arming a retry also arms ONE self-wake `setTimeout` (`scheduleInitRetryWake`) that requests a frame just after the backoff — re-arming itself if `requestRender` is unavailable at that instant (before `configureDepthSort`, or nulled by an offline capture). A late success runs `reregisterAfterLateWorkerInit()`: the fresh worker holds no registrations and no centers are retained, so it invalidates BOTH freshness stamps on every unregistered, still-committed, order-dependent node and requests ONE reprocess (the same recovery as the switch-to-sorted mode hook, shared via `invalidateSortedNodeCommitStamps`). `getDepthSortWorkerStatus()` reports `idle` / `ready` / `starved` (armed **or in flight**) / `failed` plus the deadline-miss count, and is exposed on `__luxarDebug`. Once the verdict is `'failed'` — a dead script, or the retry budget spent — AND some visible, still-committed, order-dependent node actually wants sorting (`anyNodeWantsSorting()`, the same predicate the retry gate uses), `isDepthSortAvailable()` goes false and the data-loading monitor's footer says `depth sort UNAVAILABLE`. What that note proves is narrow and worth stating exactly: visible order-dependent geometry IS being drawn in storage order. Its absence proves nothing — a `'starved'` init reports available for the whole retry window while identity order is drawn, `?depthSort=0` reports available by definition, and a single commit that could not reach a healthy worker never shows up here either. The demand half exists because the warm-up spawns unconditionally: without it a CSP-blocked chunk would announce a degraded subsystem on an all-additive scene that never sorts.

`getDepthSortWorkerStatus()`'s verdict is likewise about INIT only — a worker that dies after initializing keeps reporting `'ready'`, and each sort then burns `SORT_RPC_TIMEOUT_MS`.

### Generation Contract (Spec §5)

**Invariant**: An ordering may only be applied to the exact commit it was computed for.

- `generation` is a per-node **monotonic counter**, bumped on every NON-noop commit (stamp-only noops leave it unchanged)
- Every sort request carries the generation it was issued for
- Stale-drop at TWO checkpoints:
  1. **Worker side** (`workers/sort-worker/sorting.ts::sortNode`): returns `null` when `node.generation !== params.generation`
  2. **Main thread** (`scheduleSort` resolve handler): applies the ordering only when `result.generation === current.generation` AND `hasCommittedData(mesh)` (LOD demotion signal)
- **Unique across lifetimes**, not just within one: `releaseDepthSortNode` deletes the node state, and a re-promotion recommit would otherwise restart the counter — letting a stale in-flight sort from the previous life pass the guard and apply a CORRUPT permutation over the new (differently-sized) commit (found by randomized interleaving fuzz; see the `nextGeneration` invariant in `depth-sort-coordinator.ts`)

### NodeSortState (Per-Node Tracking)

Each registered node gets a `NodeSortState` entry:

```typescript
{
  mesh: THREE.Mesh; // The render mesh (pick node shares its geometry)
  generation: number; // Lifetime-unique non-noop commit stamp
  inFlight: boolean; // True while a sort RPC is outstanding
  resortQueued: boolean; // A newer commit landed mid-sort — re-sort once resolved
  lastSortAxis: Vector3 | null; // Model-space view axis at last DISPATCHED sort
  lastSortOffset: number; // Normalized view-axis offset (m14 / |axis|)
  registered: boolean; // Centers for CURRENT generation dispatched to worker
}
```

**Key fields**:

- `lastSortAxis` / `lastSortOffset` — the pose the last sort was dispatched from; `null` before the first dispatch. The per-frame scheduler compares live poses against these to decide when a re-sort is due.
- `registered` — true iff centers for the CURRENT generation were dispatched to the worker. Set where the register RPC is issued; cleared on every release branch (empty/commutative commit, mode-switch-away). Used by `evaluateDepthSortPerFrame()` to recover a node whose first dispatch raced a null camera: `registered && lastSortAxis === null` means "worker has centers, no sort ever left" — dispatch one now.

### Commit Flow (noteDepthSortCommit)

Called on every non-noop commit of a sortable node (gsplats, points, lines). Always bumps the node's generation (dropping any in-flight sort's result). When the node's LIVE effective blending mode is order-dependent, transfers the projected centers to the SortWorker and requests one sort from the current camera pose.

**Parameters**:

- `mesh: THREE.Mesh` — the render mesh (pick node shares its geometry)
- `centers3: Float32Array | (() => Float32Array)` — projected 3D centers, `count * 3` floats:
  - As a `Float32Array` it is **TRANSFERRED** (detached) on the order-dependent path, so the caller must hand over a buffer with no other readers (gsplats pass `processed.centers3D`: the commit's texture-write loops are the last main-thread readers)
  - As a **THUNK** it is invoked lazily, only when the node actually registers (order-dependent mode, non-empty, latest generation) — points use this to pay the O(N) fresh-copy of `data.positions` only on the sorted path. The thunk MUST return a freshly allocated array: the returned buffer is transferred, and a `subarray` view of a live array would detach that array with it.
- `count: number` — element count

**Flow**:

1. Get or create `NodeSortState` for `mesh.uuid`
2. Bump `state.generation = ++nextGeneration`
3. **Early-exit** (identity ordering) when:
   - Depth sorting disabled (`!depthSortEnabled` — session master switch)
   - Commutative blending (`!isLiveOrderDependent(mode)`)
   - Empty (`count === 0`)
   - On early-exit: clear the node's recorded pose (`clearSortPose`), release worker-side registration (`releaseWorkerNode`), return
4. Otherwise: `ensureWorker()` (spawns if needed), then when ready:
   - Re-check `current.generation === generation` (a newer commit may have landed while the worker was spawning)
   - Resolve lazy centers provider NOW (after the generation re-check, so a superseded commit never pays the copy)
   - Set `current.registered = true` BEFORE the register RPC (a throwing provider leaves the node unregistered instead of stranding a phantom registration)
   - Transfer centers: `api.registerNode(transfer({ nodeId, generation, centers3, count }, [buffer.buffer]))`
   - `scheduleSort(mesh, nodeId)`
5. Catch: once per EPISODE (a successful — possibly retried — init re-arms the flag), log that this commit could not reach the SortWorker and the identity order is drawn, quoting `getDepthSortWorkerStatus()` rather than asserting a cause: every later commit lands here while the cached `initPromise` is rejected, but the same catch also fires for a throwing lazy centers provider or a `transfer()` of an already-detached buffer. Rendering degrades gracefully to unsorted normal mode

### Single-In-Flight Rule (scheduleSort)

Enforces **at most one in-flight sort per node**; a commit landing mid-sort queues exactly one re-sort (stored in `state.resortQueued`, drained on resolve).

**No apply-gate**: sorting and applying run CONCURRENTLY. A chunked apply streams into the INACTIVE buffer of the double-buffered `aSortedIndex` pair, so the displayed ordering stays a complete permutation throughout, and a fresher ordering resolving mid-stream is held and swapped in at the next flip — never wasted. (A gate existed while streams wrote into the LIVE attribute, where out-sorting the apply cadence only prolonged the mixed state; the double-buffer swap removed it.)

**Flow**:

1. Get `state`, `camera` (from `getCamera()`) — and return unless the worker is actually usable: `api` is wrapped BEFORE `initializeWithGuard` is awaited, so a live handle is not a ready worker, and `sort` is the one RPC that `requireWasm`s worker-side. `workerInitState === 'ready'` is therefore part of the gate, here rather than at each caller (the offline-capture force loop dispatches without consulting `registered`). Skipping also keeps `recordSortPose` below from stamping a pose for a sort that never ran, which would silence the per-frame `!lastSortAxis` recovery dispatch
2. Return if already `inFlight` — set `resortQueued = true` and return
3. Set `inFlight = true`
4. Refresh `mesh.matrixWorld` and `camera.matrixWorld` (can be stale when a commit fires before the next frame)
5. Derive `viewMatrix = inverse(camera.matrixWorld)`, `modelView = viewMatrix × mesh.matrixWorld`
6. Record the pose (`recordSortPose(state, modelView)`) — stores the model-view z-row (axis direction + normalized offset)
7. Begin a detached profiler pass (`getProfiler()?.beginDepthSortPass()`) — its duration is the dispatch→applied round-trip
8. `withTimeout('depth-sort', api.sort({ nodeId, generation, modelView }), 30s)`:
   - **Timeout**: a worker that CRASHES mid-session leaves the Comlink RPC pending forever, and a stuck `inFlight` is unrecoverable — the per-frame scheduler skips in-flight nodes and later commits only set `resortQueued`, which never drains. Routing the timeout through the existing .catch clears `inFlight` and drains the queue (bounded staleness degrade instead of a permanently unsorted node).
9. On **resolve**:
   - Clear `inFlight`
   - **Stale-drop**: stage only when `result.generation === current.generation` AND `hasCommittedData(mesh)` (LOD demotion signal)
   - Set profiler metadata: `splats`, `kernelMs`, `boundaryMs` (worker overhead), `queueMs` (round-trip minus worker time), and `info` with ordering bytes labeled `sched`
   - Stage the ordering with lifecycle callbacks: `writeSortedIndexOrdering(geometry, result.ordering, count, callbacks)`; the per-frame pump streams slices into the inactive buffer
   - Transfer profiler-session ownership to those callbacks and request the bootstrap frame
   - Drain `resortQueued` if set: `scheduleSort(mesh, nodeId)`
10. On **rendered application**:
    - The final pump selects the complete buffer but does not yet report it applied
    - The mesh's chained `onAfterRender` hook acknowledges that THREE consumed the pending attribute ranges and completed a draw with the selected buffer
    - Upgrade `info` from `sched` to `up`, record `applyMs` (resolve→draw) and the monotonic completion event, then end the profiler pass
    - If the ordering is superseded, cancelled, demoted, released, or disposed before that draw, end the pass still labeled `sched` and do not record a completion
11. On **reject** (timeout or worker error):
    - End the profiler pass (unless ownership was already transferred to an accepted ordering)
    - Log error
    - Clear `inFlight`
    - Drain `resortQueued` even on failure (bounded: only a real commit sets `resortQueued`, so a persistently failing worker cannot loop)

### Chunked Ordering Apply (Perf Lever L8)

Every ordering uses the same staged-apply path; orderings larger than one 1M-index (4 MB) slice span multiple frames on the classic WebGL backend:

- The resolve path only records the pending state (`writeSortedIndexOrdering`)
- The per-frame pump (`pumpChunkedOrderingApplies`, called from `evaluateDepthSortPerFrame`) writes at most one slice per rendered frame
- Each written slice arms an upload-pending flag cleared by the attribute's `onUploadCallback`; the next pump stalls until that callback proves the prior slice reached the GPU
- Hidden nodes are not pumped, and a stalled pump requests no extra frame; the consumed-slice upload callback requests the precise continuation frame when a culled node becomes drawable again
- Slices land in the INACTIVE buffer of the double-buffered `aSortedIndex` / `aSortedIndexB` pair; the `uSortedIndexSlot` uniform flips only once that buffer holds the whole new permutation, so mid-application frames keep rendering the previous COMPLETE ordering (no old/new mix)
- The flip only **selects** the new buffer. The ordering becomes applied only after the mesh's `onAfterRender` hook acknowledges a completed draw with that selection
- A newer ordering arriving mid-stream is HELD and started after the flip (restart-on-arrival never converges under a continuous orbit)
- On **completion**, drain a queued re-sort (a commit that landed mid-sort parked it)
- **Abort** when `committedData` is cleared, the node/scene is released, or the geometry is disposed; accepted-but-unrendered profiler sessions close as abandoned rather than uploaded
- The WebGPU backends ignore attribute update ranges (full re-upload per flush), so chunk slicing/back-pressure is gated off there (`configureSortedIndexChunkedApply`, wired in renderer-setup); double buffering and post-render acknowledgement remain unconditional

**Bounded per node, not globally**: several large nodes resolving simultaneously each add one slice's cost to a frame (simultaneous 10M-scale resolves are already serialized by the per-node single-in-flight sort rule).

### Per-Frame Camera Re-Sort Scheduler (Phase 3)

`evaluateDepthSortPerFrame()` runs as the `'depth-sort-scheduler'` per-frame callback (registered beside `'lod-group-selector'`). For each order-dependent node with a completed dispatch on record, compares the live model-view z-row against the pose the last sort was dispatched from and dispatches a re-sort when either:

- The view axis has rotated past `config.depthSort.angleThresholdDeg` (default 3°, relative to the node — a spinning node triggers it too), OR
- The camera has translated ALONG the view axis past `config.depthSort.translationFraction` (default 0.05) × the node's bounding-sphere radius (which changes the behind-camera set the kernel clamps to the far bucket)

**Translation orthogonal to the view axis is deliberately ignored**: the kernel sorts by `view-space z = axis·p + offset`, so the permutation cannot change unless the axis direction or the offset does.

**Hysteresis**: dispatch-updates-reference. `scheduleSort` records the fresh pose, so a triggered node goes quiet until the camera moves past the threshold AGAIN. Frames between dispatch and resolve render the previous order — bounded staleness, standard 3DGS behavior.

**Skips**:

- Pending view updates (`isLoadInProgress()` — the commit will sort anyway)
- In-flight sorts (the resolve is at most a frame away). In-flight chunked ordering applies do NOT skip — a fresher sort streams into the inactive buffer concurrently
- Invisible meshes (`isEffectivelyVisible` checks all ancestors — an LOD level can be a hidden GROUP)
- Demoted meshes (`!hasCommittedData(mesh)`)
- Nodes whose live mode is no longer order-dependent (switched to additive/luminous/max)

**Flow**:

1. **Clear render-order state FIRST** (before any early-return) — `clearRenderOrderFrameState()` drops the previous frame's partition-wrapper subtree references
2. **Pump chunked applies** (before any early-return) — delaying them during a load would postpone convergence to the newest complete ordering
3. Early-exit if `!depthSortEnabled` or `nodeStates.size === 0` or no camera
4. For each node in `nodeStates`:
   - Skip if invisible or no longer order-dependent
   - Refresh `camera.matrixWorld` (cheap when nothing changed)
   - Derive `modelView = inverse(camera.matrixWorld) × mesh.matrixWorld`
   - **Collect cross-mesh order slot** (see render-order.ts below)
   - **Within-mesh re-sort trigger**:
     - Skip while a load is in progress or when the mesh is demoted (`!hasCommittedData(mesh)`)
     - Skip if a sort is in flight; an apply-pending ordering does not block a fresher dispatch because it streams into the inactive buffer
     - If `!lastSortAxis && registered`: first commit raced a null camera; recover with one dispatch now
     - Compare live axis/offset against `lastSortAxis` / `lastSortOffset`:
       - Angle: `axis.dot(lastSortAxis) < cosThreshold`
       - Translation: `abs(offset - lastSortOffset) > translationFraction × radius`
     - If moved: `scheduleSort(mesh, nodeId)`
5. **Assign global renderOrder** — `assignGlobalRenderOrder()` (see render-order.ts below)

### Offline Capture (resortForCapture)

The per-frame scheduler runs ONLY inside the rAF loop. Offline capture (the gallery orbit-video pass) STOPS that loop, moves the camera per frame, and renders synchronously — so the scheduler never fires and every frame would be filmed with the back-to-front permutation frozen at the pre-orbit pose (order-dependent modes: normal / volumetric).

`resortForCapture(maxWaitMs = 3000)` (exposed as `__luxarDebug.resortDepthOrderingForCapture`) drives the ordering by hand for the current pose:

1. FORCE a fresh sort on every eligible node (`isEffectivelyVisible` + `hasCommittedData` + `isLiveOrderDependent`) — offline can afford a full sort per frame, so the ordering is exact for THIS pose, not only when a threshold trips
2. Run the cross-node renderOrder pass + pump chunked applies via `evaluateDepthSortPerFrame`
3. Drain worker sorts + chunked applies to quiescence (`isCaptureQuiescent`: no node `inFlight` / `resortQueued` / `hasPendingSortedIndexOrderingApply`), yielding a macrotask (`setTimeout(0)`) per iteration, bounded by `maxWaitMs` so a wedged worker can never hang the capture. The drain BYPASSES the #715 slice back-pressure (`setSortedIndexApplyBackPressureBypassed`): it never draws, so the upload ack that releases a stall can never fire, and a multi-slice apply (>1M elements) would otherwise wedge every frame until the timeout — offline, the slices fold into one upload on the capture's own render, which is exactly acceptable

It **suppresses the `requestRender` wake** for the duration (depth-counted / reentrancy-safe via `captureSuppressDepth` / `requestRenderBeforeCapture`; only the OUTERMOST call snapshots + restores) so draining can't re-arm the loop the capture deliberately stopped. **No-op** when depth sorting is disabled, no order-dependent node exists, or the worker is unavailable — including the merely-not-ready-yet case of a capture launched during startup warm-up, since `scheduleSort` gates on `workerInitState === 'ready'` (the pure-main-thread renderOrder pass still runs).

### Blending-Mode Switch Hook (noteDepthSortBlendingModeSwitch)

Reacts to a sortable layer's blending mode changing at runtime (the LayersPanel compose chain). Wired for all four geometry types (gsplats, points, lines, mesh).

**Flow**:

1. Exit if depth sorting disabled (identity ordering is pinned for every mode)
2. Exit if `!newMode` or `newMode === prevMode`
3. Switching TO a sorted mode (`!wasSorted && isSorted`):
   - Clear the node's noop stamp: `clearCommittedData(mesh)`
   - Clear the per-slice freshness stamp: `delete mesh.userData.loadedViewVersion` (so lazy LOD levels reload on re-show)
   - Request a view reprocess: `requestReprocess()` — the memoized-concat noop path would otherwise skip re-projection entirely; the SliceCache still holds the source nD data, so this is one O(N) re-projection per mode switch (a rare user action)
4. Switching AWAY from a sorted mode (`wasSorted && !isSorted`):
   - Bump `state.generation = ++nextGeneration` (invalidate any in-flight sort's result)
   - Clear the node's recorded pose: `clearSortPose(state)` (hygiene)
   - Release worker-side registration: `releaseWorkerNode(nodeId)`

**Sorted modes = normal ∪ volumetric (`needsDepthSort`)**, for the three emissive types; mesh has no `volumetric` (its material maps the request onto `opaque`), so its sorted set is just `normal`. A switch BETWEEN two sorted modes (e.g. normal→volumetric) is deliberately a no-op here: the ordering stays valid; the projection/output change is the material's problem (TSL rebuild / GLSL define recompile).

### Node Release (releaseDepthSortNode / releaseAllDepthSortNodes)

**`releaseDepthSortNode(mesh)`** — wired to node disposal and lazy-LOD release for every sortable type:

1. Delete `nodeStates.get(mesh.uuid)`
2. Abort any in-flight chunked apply: `cancelSortedIndexOrderingApply(geometry)` (with the node state gone the per-frame pump would never visit this geometry again)
3. Fire-and-forget worker-side release: `releaseWorkerNode(nodeId)` (swallows rejections — releases run during teardown flows where the worker may already be terminating)

**`releaseAllDepthSortNodes()`** — wired to dataset-switch teardown (`clearLoadedSceneContent`) ahead of the per-mesh walk:

1. Clear `nodeStates` map
2. Abort all in-flight chunked applies: `cancelAllSortedIndexOrderingApplies()`
3. Fire-and-forget worker-side release: `api?.releaseAllNodes().catch(() => {})`

The sweep covers registrations whose mesh was never attached to the scene; the walk's per-mesh releases then no-op.

### Disposal (disposeDepthSort)

App teardown / test reset:

1. Clear `nodeStates` map
2. Abort all in-flight chunked applies: `cancelAllSortedIndexOrderingApplies()`
3. Clear render-order frame state: `clearRenderOrderFrameState()` (module-state reset completeness — an embedder that disposes and re-inits in one page must not have the old scene pinned)
4. Terminate worker: `worker?.terminate()`
5. Reset module state: `worker = null`, `api = null`, `initPromise = null` (+ `initEpoch++`, which ORPHANS an attempt still in flight — the one reset nulling a variable cannot do, since its init deadline timer lives in a closure nothing here can cancel), all injected callbacks `= null`, `depthSortEnabled = true`, `warnedWorkerUnavailable = false`, the offline-capture suppression state (`captureSuppressDepth = 0`, `requestRenderBeforeCapture = null`, back-pressure bypass off), and the init-failure bookkeeping (`workerInitState = 'idle'`, `initTimeoutRetryPending = false`, `initTimeoutCount = 0`, `initRetryNotBeforeMs = 0`, pending self-wake cancelled) — so an embedder that disposes and re-inits in one page starts with a full retry budget and a truthful status

## Cross-Node (Inter-Mesh) Ordering (render-order.ts)

### Purpose

Depth sorting orders elements WITHIN a mesh; THREE orders transparent MESHES by their `matrixWorld` origin — but every Luxar data mesh bakes element centers into the geometry and shares the world origin, so THREE's per-object sort key is identical for all parts and they draw in fixed creation order, NOT back-to-front. This module assigns every visible sorted-mode mesh a `renderOrder` each frame — THREE sorts transparent objects by `renderOrder` before `z`, ascending, so the lowest (farthest) draws first.

### Per-Frame Protocol

Driven by `evaluateDepthSortPerFrame` (depth-sort-coordinator.ts):

1. `clearRenderOrderFrameState()` at the top of the frame (before any early-return)
2. `collectRenderOrderSlot(mesh, mv, camPos)` once per surviving sorted-mode mesh
3. `assignGlobalRenderOrder()` after the loop

### Global Scale Assignment

Because `renderOrder` is compared **globally** across all transparent meshes, the assignment must put every sorted-mode mesh on ONE sequential integer scale (1..M, farthest first), reserving THREE's default 0 for empty parts that have not committed yet. Steps:

1. **Group by wrapper/leaf identity** — meshes sharing a partition wrapper form one order group; a single-leaf mesh is its own group of one
1. **Band by authored `depth_level`** (the PRIMARY key — `docs/guides/specs/LAYER_DEPTH_LEVEL_SPEC.md`) — a layer may author an integer draw order, higher = nearer the camera = drawn later, composed nearest-setter-wins and read off `mesh.userData.attrs.depth_level`. Groups sort by `(level, meanZ)`, and **containment edges below are honoured only WITHIN a band** — bands are hard partitions of the draw order, so an authored level is the author's stated intent overriding an inferred relation, and a cross-band containment relation is not outranked but unrepresentable (it warns once, naming both paths, because dropping it is the one way a level can wash out an embedded layer). Unset ⇒ band 0, so with nothing authored every group lands in ONE band, the comparator falls through to rule 2 alone, the edge set is unchanged, and the output is bit-identical to before the feature existed — structurally, not via a guarded fast path. A **commutative** layer (`additive`/`luminous`/`max`) is normally pinned at `renderOrder` 0 and always drawn first; authoring a level is the only way to state where it sits relative to an order-dependent one, and is the sole reason such a layer enters this pass at all (it never requests a within-mesh sort). Two things a band deliberately does NOT do: it cannot cross THREE's opaque→transparent bucket split (`renderOrder` is only compared within a bucket, so a band holding both is half-honoured and warns), and it buys _stability_, not correctness — for concave interpenetrating layers no valid whole-object order exists from any viewpoint, so a level pins such a pair to one stated wrong answer instead of a camera-dependent one
1. **Order groups by mean view-z** — the mean of each group's members' content centroids (bounding-sphere centers through model-view); more negative = farther. A documented approximation: exact inter-group ordering does not exist for arbitrarily interleaved groups, but wrappers/leaves are normally spatially disjoint datasets, and co-located overlapping layers have no meaningful cross order anyway.
   **Containment overrides depth**: when one group's bounding sphere strictly contains another's (a tiny reference-marker node embedded inside a huge cloud), no single order integer is correct — the container's centroid sorts nearer for ~half of all camera orientations, and an order-dependent mode drawn container-last multiplies the embedded node's pixels by the container's whole transmittance (≈ erases it). A priority topological pass (`orderGroupsWithContainment`) forces every strict container to draw before its contents (embedded content composites on top — under-attenuation is the lesser error vs. blinking out on orbit); containment edges always point from a strictly larger to a strictly smaller sphere, so the relation is acyclic, and remaining freedom stays farthest-first.
1. **Within a group, BSP ranks or view-z**:
   - **BSP tree** — when both sides have a BSP rank (a ranked wrapper ranks ALL its members): order by `partRank` ascending (0 = farthest). This is exact for point/gsplat BSP cells at any camera pose, including inside the volume (Fuchs–Kedem–Naylor painter's algorithm), and approximate for centroid-split lines/mesh or overlapping uniform tiles.
   - **Centroid (fallback)** — when either side lacks a rank (rank-less legacy wrapper members, or a single-leaf mesh): order by `viewZ` ascending (more negative = farther). Per-object ordering: approximate, and it degenerates when the camera is inside the volume — which is why the BSP path exists.
1. **Write sequential integers** — 1..M to `mesh.renderOrder`

**Transparent objects OUTSIDE the coordinator's sorted set** (commutative modes), plus empty parts that have never committed, keep `renderOrder` 0 and draw before the globally-farthest sorted mesh — depth interleaving with unsorted content stays out of scope.

### BSP Tree Traversal

Native `partition=` adders and gsplat spatial-partition producers (`tiles`/`adaptive`, content/uniform tiling, and batch merge) store their recursive split planes as a `bsp_tree` attr (see `docs/guides/user/LUXAR_ZARR_FORMAT.md` and `docs/specs/GSPLATS_ZARR_FORMAT.md`). `load-partition-group-node.ts` validates the tree structure and, when the parts have `position_bounds`, checks its split geometry; malformed or unsound trees drop to the centroid fallback, while well-formed trees without verifiable bounds retain the stored ordering. Accepted trees are stashed on the wrapper `THREE.Group`'s `userData.bspTree`, and each part object is stamped with its `userData.partIndex` (`bsp_tree` leaves name this index, not the child group's `part_<i>` suffix).

**Per-frame flow**:

1. Transform the camera into the wrapper's local space: `eyeLocal = camPos.applyMatrix4(inverse(wrapper.matrixWorld))`
2. Traverse the tree back-to-front: `traverseBspBackToFront(tree, eyeLocal, order)`
   - At each split: the eye is on one side of the plane; everything on the far side draws before everything on the near side
   - `left` holds `coord < split` (the near side when `eyeLocal[axis] < split`)
3. Number the resulting order: 0 = farthest
4. Memoize in `partitionRankCache` (cleared every frame)

**Result**: exact back-to-front part order for point/gsplat BSP cells at any camera pose, including inside the volume; approximate traversal for centroid-split lines/mesh and overlapping uniform tiles, where geometry can cross a cut.

### Centroid Fallback

When a wrapper has no `bspTree`, when any stored split axis is absent from the live `displayDims`, or for a single-leaf mesh, the slot's `partRank = -1` and ordering falls back to `viewZ` (bounding-sphere center pushed through the model-view). An axis-mapping rejection logs once per wrapper. The fallback is approximate and degenerates when the camera is inside the volume.

### OrderSlot Structure

```typescript
{
  mesh: THREE.Mesh; // The render mesh
  groupKey: THREE.Object3D; // Partition wrapper, or the mesh itself for a single leaf
  partRank: number; // BSP painter rank (0 = farthest), or -1 when none
  viewZ: number; // View-space z of the bounding-sphere center (more negative = farther)
}
```

Fresh array every frame (a grow-only pool would pin disposed meshes across frames; counts are tens, matching the per-frame allocations `wrapperPartRanks` already makes).

### Frame-State Containers

- **`partitionRankCache: Map<THREE.Object3D, Map<number, number> | null>`** — per-frame cache of a wrapper's back-to-front part order; `null` marks a wrapper with no usable `bspTree`. Cleared at the top of every frame.
- **`orderSlots: OrderSlot[]`** — per-frame collect buffer, built during the node loop, consumed by `assignGlobalRenderOrder`, then reset to `[]`.
- **`scratch: RenderOrderScratch`** — per-frame allocation-free scratch (wrapper-inverse matrix, eye-local vector, center vector). Lazily allocated on first use (several unit-test files partially mock 'three', and an import-time `new THREE.Matrix4()` would break every test that transitively imports this module).

`warnedAxisMappingWrappers` is intentionally not frame-scoped: its weak entries suppress repeated diagnostics without retaining disposed wrappers.

## Configuration

### App Init (core/app/init/pipeline.ts)

```typescript
configureDepthSort({
  getCamera: () => sceneManager.camera, // GETTER, not captured reference
  requestRender: () => animationController.requestRender(),
  requestReprocess: () => sceneLoader.updateView({}),
  isLoadInProgress: () => sceneLoader.isUpdateInProgress(),
  getProfiler: () => updateProfiler ?? null,
});

// Session master switch
const depthSortEnabled = config.depthSort.enabled && urlParams.get('depthSort') !== '0';
setDepthSortEnabled(depthSortEnabled);
```

### Per-Frame Registration

```typescript
animationController.addPerFrameCallback('depth-sort-scheduler', () => {
  evaluateDepthSortPerFrame();
});
```

### Commit-Time First Sort

Eligible instanced nodes publish their first ordering synchronously inside the
commit so the next frame does not wait for the SortWorker round trip.
`depthSort.syncSortMaxElements` is both the per-node ceiling and the shared
element budget between `evaluateDepthSortPerFrame()` calls; once spent, later
commits keep the repaired prior ordering and use the normal async pipeline.
Indexed Mesh nodes skip this path because they apply orderings through
`geometry.index`, not `aSortedIndex`.

### Disposal

```typescript
// In app dispose pipeline
disposeDepthSort();
```

## Invariants & Contracts

### Generation Contract

- An ordering may only be applied to the exact commit it was computed for
- Generations are unique across a node's LIFETIMES, not just within one
- Stale orderings are dropped at two checkpoints (worker + main thread)

### Single-In-Flight Rule

- At most one sort RPC is in flight per node
- A commit landing mid-sort queues exactly one re-sort (drained on resolve)
- A chunked ordering apply does NOT park requests — it streams into the inactive buffer while a fresher sort runs

### Pose-Clearing Invariant

Every release path (empty/commutative commit, mode-switch-away, node disposal) must clear `lastSortAxis` and `registered` — otherwise the per-frame scheduler would keep firing guaranteed-null sort RPCs against a released worker registration on every threshold crossing.

### Frame-State Lifetime

The per-frame render-order containers (`partitionRankCache`, `orderSlots`) are cleared/reset every frame and never outlive a frame. `clearRenderOrderFrameState()` resets exactly these two — it runs FIRST in `evaluateDepthSortPerFrame`, before any early-return, so a disposed/dataset-switched frame can't leave them holding stale THREE object references. `scratch` is NOT reset each frame: it is module-scoped and persistent, holding only copied matrix/vector values (no scene references), which is why it is safe to keep across frames.

### Allocation-Free Hot Path

The per-frame scheduler and render-order assignment do no per-element allocation; a small O(#meshes + #wrappers) per-frame allocation (a fresh `orderSlots` array + one slot object per mesh, and a fresh `order` array + rank Map per wrapper) is intentional so disposed meshes are never pinned across frames:

- `scratch` — lazily allocated on first use, reused every frame
- `orderSlots` — fresh array per frame, but the array itself is cheap; the slots are plain objects
- Per-node pose comparison — reuses `scratch.axis`, `scratch.mv`
- BSP traversal — reuses `scratch.wrapperInv`, `scratch.eyeLocal`; the `order` array is per-wrapper-per-frame

## See Also

- **`workers/sort-worker.ts`** — Worker-side entry point (Phase 2)
- **`workers/sort-worker/README.md`** — Worker internals (registration, sorting, state)
- **`rendering/element-storage.ts`** — `writeSortedIndexOrdering` / `pumpSortedIndexOrderingApply` / chunked-apply machinery
- **`rendering/blending-state.ts`** — `needsDepthSort(mode)` predicate
- **`docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`** — Full depth-sorting specification (Phases 1-3)
- **`config/sections/depth-sort/`** — `depthSort.enabled`, `depthSort.angleThresholdDeg`, `depthSort.translationFraction`, `depthSort.workerInitTimeoutMs`, `depthSort.syncSortMaxElements`
- **`types/committed-data.ts`** — `hasCommittedData` / `clearCommittedData` (LOD demotion signal)
