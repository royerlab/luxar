# workers/sort-worker

> Persistent depth-sort worker — Phase 2 of the depth-sorting subsystem

## Purpose

The SortWorker is a single **persistent** Comlink worker (NOT part of the round-robin data worker pool) that computes back-to-front orderings for order-dependent blending modes (normal and volumetric). It serves every texture-backed geometry with an `aSortedIndex` indirection — **gsplats, points, and lines** — via a geometry-agnostic mechanism: projected 3D centers in, back-to-front permutation out.

**Why persistent?** Node registrations and their transferred center buffers must live in exactly one worker. The main thread transfers each order-dependent node's projected 3D centers to this worker on commit, and camera-driven re-sorts (Phase 3, main-thread `depth-sort-coordinator.ts`) keep requesting orderings from the SAME centers — no re-copy from the main thread.

**Why dedicated?** Ongoing and camera-driven sorting lives exclusively here, never inside projection (the plain-3D projection fast path doesn't run in a worker at all). The first ordering after an eligible instanced commit may also be computed on the main thread within the shared `syncSortMaxElements` frame budget; indexed Mesh is excluded, and async registration plus all subsequent sorting remain in the worker.

## File Map

```
sort-worker/
├── initialize.ts    — Bootstrap: honor wasmPath override, load compiled WASM with TS fallback
├── state.ts         — Shared SortWorkerCtx: wasm module, per-node center registry
└── sorting.ts       — Task bodies: registerNode, sortNode, releaseNode, releaseAllNodes
```

The facade `workers/sort-worker.ts` (entry point bundled by Vite's `?worker` import) wires the task functions into `workerAPI` and `expose()`s it — the same layout as `data-worker.ts`.

## Architecture

### Module Layout

**`sort-worker.ts`** (facade) — Vite `?worker` entry point:

- Imports task helpers from `./sort-worker/<file>.ts`
- Constructs `workerAPI` object (binds `state` to each task)
- `expose(workerAPI)` for Comlink RPC

**`sort-worker/state.ts`** — Shared mutable context:

- `wasm: WasmModule | null` — WASM module instance (compiled or TypeScript fallback)
- `nodes: Map<string, RegisteredNode>` — per-node center registry (nodeId → { generation, centers3, count })
- `requireWasm(ctx)` — throws when `ctx.wasm` is null (used before `initialize()`)
- `NOT_INITIALIZED_MSG` — single source of truth for the error message

**`sort-worker/initialize.ts`** — One-time bootstrap:

- Honors embedder-relocated WASM shim URL via `setWasmJsUrl(wasmPath)` (the main-thread override does NOT cross the worker boundary)
- Loads compiled WASM with automatic TypeScript fallback: `ctx.wasm = await initWasm()`
- Returns `{ wasmFallback: boolean }` to the main thread

**`sort-worker/sorting.ts`** — Task bodies (called via Comlink RPC):

- `registerNode(ctx, params)` — store/refresh a node's centers
- `sortNode(ctx, params)` — compute back-to-front ordering, return transferred result or null (stale-drop)
- `releaseNode(ctx, nodeId)` — drop a node's registration
- `releaseAllNodes(ctx)` — drop every registration (dataset switch / app teardown)

### SortWorkerCtx (Shared State)

```typescript
{
  wasm: WasmModule | null; // WASM module (compiled or TS fallback)
  nodes: Map<string, RegisteredNode>; // nodeId → { generation, centers3, count }
}
```

**RegisteredNode**:

```typescript
{
  generation: number; // Per-node monotonic non-noop commit counter
  centers3: Float32Array; // Projected 3D centers [count * 3], TRANSFERRED from main thread
  count: number; // Number of elements
}
```

**Key invariant**: A node's `generation` must match the sort request's `generation` for the ordering to be valid (see generation contract below).

## Lifecycle

### Spawn (Main Thread → Worker)

1. Main thread calls `ensureWorker()` (depth-sort-coordinator.ts) — at APP INIT via `warmUpDepthSortWorker()`, not on the first order-dependent commit. That commit arrives while the loader saturates the main thread, which is where the init deadline below used to be missed
2. Constructs `new SortWorker()` (Vite `?worker` import) or `new Worker(sortWorkerUrlOverride)` (embedder override)
3. Wraps via Comlink: `api = wrap<SortWorkerAPI>(worker)`
4. Calls `api.initialize(sortWorkerWasmPathOverride)` through the shared `worker-pool/lifecycle/init-with-guard.ts` — `config.depthSort.workerInitTimeoutMs` deadline + `onerror`/`onmessageerror` guard (see the failure taxonomy in the depth-sort-coordinator README)

### Initialize (Worker)

`initialize(ctx: SortWorkerCtx, wasmPath?: string): Promise<SortWorkerInitResult>`

1. If `wasmPath` is provided, call `setWasmJsUrl(wasmPath)` (must run before `initWasm()`)
2. `ctx.wasm = await initWasm()` — loads compiled WASM or falls back to TypeScript
3. Returns `{ wasmFallback: isWasmFallback(ctx.wasm) }`

**Error handling**: If WASM initialization fails, throws `'WASM unavailable. Luxar requires WebAssembly support...'`. The main thread catches this, terminates the wedged worker, and degrades to unsorted normal mode (all later commits land in a warn-once catch — see `noteDepthSortCommit` in depth-sort-coordinator.ts).

### Register Node (Worker)

`registerNode(ctx: SortWorkerCtx, params: RegisterNodeParams): void`

**Parameters**:

```typescript
{
  nodeId: string; // Node identity (mesh UUID on the main thread)
  generation: number; // Per-node monotonic non-noop commit counter
  centers3: Float32Array; // Projected 3D centers [count * 3] — TRANSFERRED
  count: number; // Number of elements
}
```

**Flow**:

1. Clamp `count` to `Math.min(count, Math.floor(centers3.length / 3))` (safety)
2. Log a warning if clamped
3. Store/replace `ctx.nodes.set(nodeId, { generation, centers3, count })`

**Called** on every non-noop commit of an order-dependent node (main thread `noteDepthSortCommit`). Replaces any previous registration wholesale.

**Transfer semantics**: `centers3.buffer` is TRANSFERRED from the main thread (detached), so the main thread must hand over a buffer with no other readers. Gsplats pass `processed.centers3D` (the commit's texture-write loops are the last main-thread readers); points pass a lazy provider that fresh-copies `data.positions` only on the sorted path; lines pass a lazy provider computing fresh segment midpoints.

### Sort Node (Worker)

`sortNode(ctx: SortWorkerCtx, params: SortParams): SortResult | null`

**Parameters**:

```typescript
{
  nodeId: string; // Node identity
  generation: number; // Generation this request was issued for (stale-drop guard)
  modelView: Float32Array; // Column-major 4x4 model-view matrix [16]
  shardCount?: number; // Depth shards to report bounds for; 0/absent = none
}
```

**Flow**:

1. `requireWasm(ctx)` — throws if not initialized
2. Lookup `node = ctx.nodes.get(nodeId)`
3. **Stale-drop**: return `null` when `!node || node.generation !== params.generation`
4. Allocate output: `ordering = new Uint32Array(node.count)`
5. Clamp `shardCount` to `node.count` and allocate the two `[shardCount * 3]` bound arrays
6. Time the backend call: `wasm.sort_splats_by_depth(node.centers3, params.modelView, ordering, node.count, shardCount, shardBoundsMin, shardBoundsMax)`
7. Return TRANSFERRED result: `transfer({ generation, ordering, kernelMs, workerMs, shardCount, shardBoundsMin?, shardBoundsMax? }, [ordering.buffer, ...bounds])`

**Per-shard depth bounds** (cross-node depth ordering — `docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md` §3.3):

The permutation is already back-to-front, so any contiguous range of it is a depth interval. On request the kernel additionally reports the LOCAL-space AABB of the elements each of `shardCount` equal-population ranges draws, and the main thread re-projects those boxes every frame to order shards of DIFFERENT nodes against each other.

- **`shardCount: 0` (or absent) is the default and costs nothing** — the kernel skips the pass and leaves the output arrays untouched. No node is sharded until the shard-count policy lands, so this is the state everywhere today.
- **A returned `shardCount` of `0` means "do not treat this node's ranges as depth intervals"**, and it collapses two cases on purpose: none were requested, OR the kernel took its identity-ordering fallback (`sort_splats_by_depth` returned 0 — degenerate depth range). In the fallback the ordering is fully written and the boxes are valid boxes *of it*, but they are storage ranges rather than depth intervals, so the node must merge as ONE whole-node interval — exactly today's behaviour.
- Bounds are adopted on the main thread under the SAME generation guard as the ordering, and only when that ordering is actually applied. A box retained past its permutation would place a shard by where its elements *used to be* — a silent wrong ordering rather than a visible failure — so every branch that invalidates the ordering clears them.
- An empty axis (no finite element, e.g. an all-NaN shard, or a surplus shard when `count < shardCount`) carries `min = +Infinity` / `max = -Infinity`, so `min > max` is the "no usable bounds" test. NaN centers are EXCLUDED from a box (one bad element must not cost a whole shard its place in the merge); ±Infinity propagates into it.

**Timing**:

- `kernelMs` — time spent inside the backend `sort_splats_by_depth` call (includes wasm-bindgen boundary copies for compiled WASM; pure kernel for TS fallback)
- `workerMs` — whole `sortNode` body duration (registry lookup + output allocation + kernel)
- `boundaryMs = workerMs - kernelMs` — worker-side overhead around the backend call (computed on the main thread, in `scheduleSort`'s resolve handler)
- `queueMs = roundTripMs - workerMs` — Comlink RPC + structured clone + event-loop queueing (computed on the main thread, in the same handler)

**Stale-drop**: The ordering is computed ONLY when `node.generation === params.generation`. A stale request (a newer commit landed between dispatch and resolve) returns `null`, and the main thread discards it. This is the FIRST checkpoint of the generation contract (the second is on the main thread).

### Release Node (Worker)

`releaseNode(ctx: SortWorkerCtx, nodeId: string): void`

Deletes `ctx.nodes.delete(nodeId)`. Called on node disposal / pool release / mode-switch-away (main thread).

### Release All Nodes (Worker)

`releaseAllNodes(ctx: SortWorkerCtx): void`

Clears `ctx.nodes.clear()`. Called on dataset switch / app teardown (main thread).

### Terminate (Main Thread → Worker)

`disposeDepthSort()` (depth-sort-coordinator.ts) terminates the worker and resets all main-thread module state. The worker's async module evaluation is interrupted, and any in-flight RPC promises reject.

## Generation Contract (Spec §5)

**Invariant**: An ordering may only be applied to the exact commit it was computed for.

- `generation` is a per-node **monotonic counter**, bumped on every NON-noop commit (stamp-only noops leave it unchanged)
- **Unique across lifetimes**, not just within one: `releaseDepthSortNode` deletes the node state, and a re-promotion recommit would otherwise restart the counter — letting a stale in-flight sort from the previous life pass the guard and apply a CORRUPT permutation over the new (differently-sized) commit
- Enforced at **two checkpoints**:
  1. **Worker side** (here, `sortNode` line 87): returns `null` when `node.generation !== params.generation`
  2. **Main thread** (depth-sort-coordinator.ts, `scheduleSort`'s resolve handler): applies the ordering only when `result.generation === current.generation` AND `hasCommittedData(mesh)` (LOD demotion signal)

**Why the double-check?** The worker check catches a re-registration that raced the RPC (new centers transferred mid-flight). The main-thread check catches a commit or LOD demotion that raced the RPC's return.

## WASM Backend

The worker uses the same `wasm/` module as the data workers:

- `initWasm()` — loads compiled WASM with automatic TypeScript fallback
- `isWasmFallback(wasm)` — true when running the TS fallback
- `wasm.sort_splats_by_depth(centers3, modelView, ordering, count, shardCount, shardBoundsMin, shardBoundsMax)` — the depth-sort kernel (the last three are the optional per-shard depth bounds; pass `0` and two empty arrays to skip)

**ndim-agnostic**: Input is always projected 3D centers (geometry-agnostic), so there is no `pickBackend` / 16-dimension routing here — the kernel is unconditional. The data workers route >16D operations to the TS backend; the SortWorker never sees nD data at all (projection happens on the main thread or in the data workers).

**Timing**: The WASM kernel timing (`kernelMs`) includes wasm-bindgen boundary copies (centers copy-in, ordering copy-in/out, mallocs) — the shim performs them inside the exported function. The TS fallback timing is the pure kernel. Both are measured with `performance.now()` on the worker's clock.

## RPC Surface (Comlink)

**Exposed via `workers/sort-worker.ts::workerAPI`**:

```typescript
{
  initialize: (wasmPath?: string) => Promise<SortWorkerInitResult>;
  registerNode: (params: RegisterNodeParams) => void;
  sort: (params: SortParams) => SortResult | null;
  releaseNode: (nodeId: string) => void;
  releaseAllNodes: () => void;
}
```

**Used by** `depth-sort-coordinator.ts::api` (Comlink-wrapped Remote<SortWorkerAPI>).

## Integration Points

### Main Thread (depth-sort-coordinator.ts)

(Line numbers deliberately omitted — the previous ones had drifted by ~70 lines and quietly misled.)

- **Warm up**: `warmUpDepthSortWorker()` — fire-and-forget `ensureWorker()` at app init
- **Spawn**: `ensureWorker()` — constructs worker, wraps via Comlink, calls `initialize()` under the shared init guard
- **Register**: `noteDepthSortCommit()` — transfers centers, calls `api.registerNode()`
- **Sort**: `scheduleSort()` — calls `withTimeout('depth-sort', api.sort(...), SORT_RPC_TIMEOUT_MS)`
- **Release node**: `releaseDepthSortNode()` — calls `releaseWorkerNode()` → `api.releaseNode()` (fire-and-forget)
- **Release all**: `releaseAllDepthSortNodes()` — calls `api.releaseAllNodes()` (fire-and-forget)
- **Terminate**: `disposeDepthSort()` — calls `worker.terminate()`

### Commit Paths (geometry commits)

- **GSplats**: `data/scene-loader/commit/commit-gsplats-geometry.ts` → `noteDepthSortCommit(mesh, processed.centers3D, count)`
- **Points**: `data/scene-loader/commit/commit-points-geometry.ts` → `noteDepthSortCommit(mesh, () => freshCopy(data.positions), count)`
- **Lines**: `data/scene-loader/commit/commit-lines-geometry.ts` → `noteDepthSortCommit(mesh, () => computeSegmentMidpoints(data), count)`

### Per-Frame Scheduler (depth-sort-coordinator.ts)

`evaluateDepthSortPerFrame()` — registered as the `'depth-sort-scheduler'` per-frame callback:

- Compares live camera pose against `lastSortAxis` / `lastSortOffset` for each registered node
- Dispatches a re-sort via `scheduleSort()` when the angle or translation threshold is crossed

## Error Handling

### Initialization Failures

- Worker script dies during async module evaluation → `onerror` / `onmessageerror` event → `initPromise` rejects — PERMANENT (nothing about waiting longer would help)
- WASM initialization fails → throws `'WASM unavailable...'` → main thread catches, terminates worker, degrades to unsorted normal mode — also PERMANENT
- Init deadline missed (`config.depthSort.workerInitTimeoutMs`, default 30 s) → the guard rejects `initPromise` with `WorkerInitTimeoutError`, the worker is terminated and the scene renders unsorted MEANWHILE — but this one is RETRYABLE (issue #1694): a missed deadline means only that init lost a race to a busy main thread (on a multi-million-element scene the thread that has to dispatch the reply is busy decoding), so the coordinator respawns up to 3 attempts in total with a growing backoff, and a late success re-registers every sorted node. Only after the budget is spent is the degrade permanent.

### Sort Failures

- Worker unavailable (init failed) → main thread's cached `initPromise` is rejected, every commit lands in the warn-once catch — warn-once per EPISODE: a successful (possibly retried) init re-arms it
- Worker crashes mid-session → Comlink RPC pending forever → timeout (`SORT_RPC_TIMEOUT_MS`, see `scheduleSort`) → main thread clears `inFlight` and drains the queue (bounded staleness degrade)
- Stale request (newer commit landed) → worker returns `null` → main thread discards the result

### Release Failures

Worker may already be terminating → RPC rejects → swallowed (fire-and-forget, see `releaseWorkerNode`).

## Invariants & Contracts

### Single Worker Instance

Only one SortWorker exists per session. Spawned at app init (`warmUpDepthSortWorker()`), terminated on app dispose. NOT part of the data worker pool (no round-robin) — but it now shares the pool's `init-with-guard.ts`, so the two cannot drift apart on startup semantics.

### Transfer Semantics

- `centers3.buffer` is TRANSFERRED from the main thread (detached) — the main thread must hand over a buffer with no other readers
- `ordering.buffer` is TRANSFERRED to the main thread (detached) — the worker must allocate a fresh array

### Generation Uniqueness

Generations are unique across a node's LIFETIMES, not just within one. `nextGeneration` is a module-scoped monotonic counter on the main thread (never resets).

### Not-Initialized Guard

Every task except `initialize()` calls `requireWasm(ctx)`, which throws when `ctx.wasm` is null. The main thread always calls `initialize()` before any other RPC.

## Performance Characteristics

### Kernel Complexity

`O(N + B)` where N = element count, B = bucket count (`DEPTH_SORT_BUCKETS = 1 << 16 = 65536` depth buckets, the full uint16 key range — see `wasm/typescript/depth-sort.ts`). Radix-like: one count pass, one prefix-sum, one write pass. Milliseconds for millions of elements on any live worker.

### Transfer Overhead

- Centers: one transfer per commit (detaches the main thread's buffer)
- Ordering: one transfer per resolve (detaches the worker's buffer)
- Model-view matrix: structured-cloned (16 floats, negligible)

### Timing Split (Perf Campaign)

The profiler metadata captures a four-way timing split:

- `kernelMs` — inside the backend call (worker clock)
- `boundaryMs` — worker overhead around the kernel (worker clock)
- `queueMs` — Comlink RPC + structured clone + event-loop queueing (main clock - worker clock)
- Round-trip latency — dispatch to applied (main clock)

All durations, so worker-clock vs main-clock is safe. Clamped at 0 against timer granularity.

## See Also

- **`rendering/depth-sort-coordinator.ts`** — Main-thread side (Phase 3): spawn, register, schedule, apply, per-frame scheduler
- **`rendering/depth-sort-coordinator/README.md`** — Coordinator architecture and contracts
- **`rendering/element-storage.ts`** — `writeSortedIndexOrdering` / chunked-apply machinery
- **`docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`** — Full depth-sorting specification (Phases 1-3)
- **`wasm/`** — WASM module (`sort_splats_by_depth` kernel)
- **`workers/data-worker.ts`** — Data worker entry point (sibling pattern)
- **`workers/README.md`** — Worker pool architecture (the SortWorker is NOT part of the pool)
