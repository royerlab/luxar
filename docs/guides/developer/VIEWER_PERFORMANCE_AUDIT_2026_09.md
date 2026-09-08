# Luxar Viewer Performance Audit (2026-09-04)

Fetch → cache → decode → process → commit → render, audited by reading the code
(four parallel read-only passes over `packages/luxar-viewer/src`) and by
measuring the **production build** (`vite build` of `67ec42091`, served by
`vite preview`) in headless Chromium 1234 on the real GPU
(`ANGLE Metal Renderer: Apple M4 Max`, 40 cores, 128 GB, 1600×1000 canvas,
DPR pinned to 1 unless stated). Every number below was measured; every code
claim was read. Items marked *[inferred]* were not directly verified.

> **Status.** This is the dated record that drove the September 2026 viewer
> performance work (PR #2528, follow-ups in #2527). Every finding below is now
> either landed, or dismissed with a measurement in that PR's description —
> including finding 8 (OPFS write-queue drops): its fix landed, then #2528's
> byte cap on the queue brought the cmu1 drops back (0 → 3 966), because the cap
> was `max(resolved L1 size, 64 MB)` and a pending write's buffer is the same
> one L1 already holds and bounds. #2561 re-sizes that cap from the non-cache
> heap remainder instead — a quarter of it, so 328 MB on a 4 GiB Chrome heap
> like this machine's, which clears both the 214 MB cmu1 streams before the cap
> binds (peak pending ≈ cumulative streamed, since its 10 961 writes drain far
> slower than they arrive — *[inferred]*, not a measured retention) and its
> 277 MB whole-store ceiling, though only by 1.18× on the latter; the allowance
> is heap-relative, so for this scene it binds again below roughly a 2.6 GiB
> heap, which is intended. #2561 also flips the overflow policy to
> drop-the-arrival, so the oldest end drains in order. The cmu1 before/after row
> below has NOT been re-measured on the audit machine since; what is in place is
> the fix plus an unconditional assertion in the audit bench (§8) that
> `opfsDropped` is 0 — load-phase and end-of-run on every cold repetition, and
> on the warm revisit — with a second assertion that the resolved allowance
> still covers what the scene streamed, so a cap that binds again shows up as a
> failed bench run rather than only as a slower revisit.
> The ad-hoc probe kit it describes was replaced by a repeatable harness:
> `pnpm test:perf:e2e -g "viewer audit"` (see §8). Numbers in §3 are the
> BEFORE state; the PR carries the same-browser before/after table.

Probe kit (Playwright scripts + analysers) lived in `delme/viewer-perf-probes/`
(not committed); raw results (JSON, request logs, console logs, traces,
screenshots) were in the job tmp dir. See §8 for how to re-run today.

---

## 1. Headline findings, ranked by impact

| # | Finding | Evidence | Impact |
|---|---|---|---|
| 1 | **Lower DPR makes overdraw-heavy scenes SLOWER, and the adaptive-DPR controller drives DPR down when fps drops.** 1.5 M points framed into ~1 600 px: DPR 1.0 → 41.7 ms/frame, 0.9 → 50, 0.75 → 75, 0.5 → 83 ms. 29.6 M 2-D splats: 1.0 → 108 ms, 0.5 → 192 ms. Same 1.5 M points zoomed so the cube covers ~24 k px: **8.3 ms (120 fps)**. | §3.2, §3.3 | Frame rate collapses exactly where the controller "helps"; the U-shape probe catches it only after a 1.5 s excursion, and it then parks at 0.9 (20 % slower than 1.0). |
| 2 | **No screen-space density cap for points/splats.** Additive ladders always stream to 100 %; partition parts are only frustum-culled; `kind=lod` groups are the only geometry that shrinks with screen area. The pathology slide loads all 29.6 M splats into a few hundred px and renders at 9 fps. | §3.2, `progressive/refinement.ts` (no coverage/density rule) | GPU wall is overdraw, not element count. A projected-density guard (keep ≈ 1–4 elements/px) would give 5–10× on these views and stop the ladder early on hosted loads. |
| 3 | **Every scene is loaded twice at startup.** `loadScene` fetches + decodes + projects + commits every node, then `initDimensionSliders` fires an unconditional `updateAllNDNodes()` (`input/input-handler/dimension-navigation/setup.ts:187`) → `updateView` v1 re-queries every loader and re-runs decode → project → commit. L0 hits == misses on every 3-D scene (dense 1501/1501, ct 615/615, bench 300/300); on cmu1 L0 had been evicted so 10 720 chunks were re-read from L1 and re-decompressed. | §3.4, console timelines | Doubles CPU/GPU work on the critical path right after first paint. On cmu1 the second pass + refinement blocked the main thread **5.9 s of a 10 s load** (25 long tasks, max 937 ms). |
| 4 | **GSplat projection for ndim ≤ 3 runs on the main thread in WASM** (`data-processor-gsplats.ts:294`: worker only when `ndim > 3`). Main-thread CPU profile of the cmu1 load, 3.6–9.7 s: luxar WASM kernels **39 %**, blosc decompression ≈ 11 %, `texSubImage2D` 8 %, wasm copy-in 3 %, idle 7 %. | §3.4 | Most real scenes are 2-D/3-D splats, so the 15-worker pool sits idle while the UI freezes. |
| 5 | **Worker pool is created lazily and each of the 16 workers fetches + compiles its own WASM.** 17 `luxar_wasm_bg.wasm` requests per session. Under a 25 Mbps/30 ms throttle the pool became ready at **4.3 s (HTTP/2) / 4.1 s (HTTP/1.1)**, i.e. at the very end of the load, because its 32 fetches queue behind hundreds of chunk fetches; first geometry appears only then. On a warm OPFS cache the whole ct load (1.1 s) is gated by worker startup (ready at 1.10 s, scene at 1.11 s). | §3.1 | Under real network conditions worker bootstrap, not data, sets first-paint. Invisible on localhost (ready in 70 ms). |
| 6 | **Small chunks are RTT-bound on HTTP/1.1.** dense (1 504 chunks, p50 4.2 KB): 10.6 s on HTTP/1.1 (7.4 Mbps of 25) vs **4.0 s on HTTP/2** (19.6 Mbps). ct (621 chunks, p50 20 KB): 6.4 s vs 4.7 s. | §3.1 | `luxar serve`, `python -m http.server`, exports' `serve.py` are HTTP/1.1 → 6 connections. The 64-wide fetch gate cannot help there. |
| 7 | **Refinement is frame-bound, not data-bound.** Fully cached neuromast timepoint (16 nodes, ~110 k splats, 0 network requests): first commit 14 ms, **full ladder 40 ms** because `refinement.ts:177,203` yields one rAF per rung and walks loaders serially. | §3.5 | Playback of cached 4-D data caps near 25 fps regardless of CPU. |
| 8 | **OPFS write queue drops under burst** (`maxDepth 1024`, drop-oldest): dense 252 / 1 252 writes dropped, cmu1 **9 775 / 10 961**. A second session of cmu1 would re-fetch ~90 % of the store. (ct: 0 dropped; warm reload served 621/621 from L2 and cut 7.1 s → 1.1 s.) | §3.1 | The L2 tier silently fails for exactly the scenes that need it. |
| 9 | **Blend warm-up scales with node count, not program count.** 4 keeper materials per node compile one per idle callback: 100-node scene → 400 idle callbacks, 3.3 s until readiness (5 s budget then "releasing readiness"). Only 9 distinct GL programs exist. Keepers are retained per node. | §3.4 | Harmless at 2 nodes; a 1 000-part partition would spend ~30 s and hold 4 000 material clones. |
| 10 | **4 wasted 404 probes per node** for optional arrays (zarr.json + 3 v2 docs): 404 of 820 requests on the 100-node scene. | §3.1 | One extra RTT-wave per node on hosted stores; cacheable but never cached. |

Things that are **fine** and were suspected: bloom (15 passes) costs nothing measurable at 1600×1000 (ct/celegans/neuromast: 120 fps with and without); the depth-sort worker is not a bottleneck; main-thread decompression is negligible for small chunks (dense load: main thread 91 % idle); 654 k 3-D splats render at 120 fps; the 100-node scene draws 101 calls at 120 fps; adaptive-DPR does restore full DPR at idle; per-update abort/coalescing works (no stale-commit artefacts seen).

---

## 2. Current state of the code (what exists)

Condensed from the four code passes; file references are the anchors to read.

### 2.1 Fetch + cache
- Three chunk tiers + a per-slice geometry cache: **L0** decoded typed arrays (byte-LRU, 200 MB, `cache/decompressed-chunk-cache/*`, Proxy on zarrita `getChunk`, **clones** every decoded chunk at `cached-zarr-array.ts:200`); **L1** compressed bytes (segmented LRU, 100 MB, metadata segment); **L2** OPFS (2 GB, background `OpfsWriteQueue`, delete barriers, circuit breaker); **S-cache** post-projection slice payloads (`cache/slice-cache.ts`, ~1.4 GB budget here).
- Heap-aware budgets (`cache/heap-budget.ts`), `?cacheBudgetMB`, content-hash validation with format-2/3 probing (`validation-queue.ts`), same-key request coalescing (bypassed when the caller passes a signal, `multi-level-caching-store.ts:492`), global 64-wide fetch gate (`utils/fetch-concurrency.ts`), retry with jittered backoff, body cancellation, zip range-reader with EOCD stitching.
- Decompression is **numcodecs Emscripten WASM on the main thread** (blosc/zstd/lz4); decode (dequantise/LUT/broadcast) goes to the worker pool above 1 000 elements and returns transferables.
- Prefetch: ±1 chunk adjacency (`chunk-prefetcher.ts`, silently no-op without registered bounds), range warming, t+1 slice prefetch on shadow loaders.

### 2.2 Processing
- Comlink pool of `hardwareConcurrency − 1` workers (15 here), least-busy dispatch, per-call timeout, pool-wide abort; created **lazily on first decode**; each worker runs `initWasm()` itself (`workers/data-worker/initialize.ts:35`).
- Points: decode in workers, **projection + radii + compaction + 3 bounds passes on the main thread** (`data/points/projection.ts`). GSplats: fused `project_gsplats_nd_to_3d` kernel; **worker only when `ndim > 3`**; inputs structured-cloned (not transferred) to keep the S-cache intact. Lines symmetric. Mesh in-process.
- WASM: SIMD, `opt-level 3`, LTO, 50 KB; **no threads**; every call copies inputs into the wasm heap and outputs back (13 arrays in for the gsplat kernel).
- Depth sort: dedicated worker, 65 536-bucket radix, triggers on commit and on >3° rotation / 5 % view-axis translation, double-buffered `aSortedIndex` with chunked upload, sync first sort ≤ 250 k.
- Accumulators allocate-once with 1.5× growth (monotonic capacity).

### 2.3 Orchestration
- `loadScene` (`data/scene-loader/lifecycle/load-scene.ts`) is a strict serial chain: dispose → caches → `openStore` (consolidated metadata) → root → `buildSceneGraph` (serial N+1 `zarr.open`, cheap with `.zmetadata`) → `loadSceneNodes` (8-wide per parent, FIFO in authored order, byte-gated) → overlays → refinement kick.
- Each leaf attaches a placeholder, then fetch → process → **synchronous atomic commit** (`commit/README.md`: no async inside commit), so pixels appear per node.
- `updateView`: single pending slot, latest-wins, in-flight update aborted, pass-waiters resolve on the next commit; no debounce.
- Refinement (`progressive/refinement.ts`): one rung per `scheduleFrame`, loaders serial, holds the update lock; `RefinementResidencyBudget` is a refusal gate (never evicts) capped at 512 MiB (`heap-budget.ts:41`).
- LOD: screen-area selector with asymmetric 10 % hysteresis for `kind=lod`; partitions frustum-culled per frame via `Box3.setFromObject` per child (`lod-group-registry.ts:1282`); additive ladders have **no** coverage-based stop.

### 2.4 Rendering
- On-demand rAF loop with 2 s idle stop and idle full-DPR frame; frame pacing after two >250 ms frames.
- Instanced unit quads + one **RGBA32F DataTexture** per node (3/6/4 texels per element), partial row uploads with a 75 % full-upload knee, `GPUBufferPool` capacity buckets + byte budget shared with LOD eviction (hidden levels only).
- Per-node materials (texture binding), 9 distinct programs in practice; compile-time `#ifdef` variants; blend warm-up keepers.
- Post: HDR half-float target → optional bloom (8 mips) → mega-shader → optional FXAA; defaults off except tone mapping. LDR half-float target allocated even when FXAA is off. Bloom/FXAA/MSAA/SSAA all opt-in; demos enable bloom.
- Adaptive DPR: 500 ms evaluation over a 1 s window, ×0.9 down / ×1.05 up, U-shape probe with learned floors and backoff, idle restore, default cap 1.0.
- Points: `minPointSize = 1.5 * max(uPixelRatio, 1)` framebuffer pixels (`materials/point/shader-glsl.ts:190`). GSplats: extent clamp + coverage fade bound overdraw per instance, not per pixel.

---

## 3. Measurements

Scenes: `performance_benchmark_example` (bench: 100 nodes × 1 000 points, format 2),
`dense_cubic_gradient_example` (dense: 1 M-point lattice + 500 k stars, 1 503 chunks, 9.8 MB),
`gsplats_3d_ct_totalsegmentator` (ct: 654 k splats, 5 nodes, 621 chunks, 13 MB, bloom on),
`gsplats_4d_celegans_tracking` (4-D, ~1.5 k splats per timepoint),
`gsplats_4d_neuromast_2ch` (16 nodes, 11.4 M splats, ~110 k per timepoint, 8-rung ladders),
`gsplats_2d_cmu1_pathology` (cmu1: 12 parts, 29.6 M 2-D splats, 10 961 chunks, 277 MB).

### 3.1 Loading

Localhost (no throttle). "scene loaded" = `Scene loaded successfully` console line, relative to navigation start.

| scene | load start | workers ready | scene loaded | first `[GEOM]` commit | notes |
|---|---|---|---|---|---|
| bench | 125 ms | not needed (≤1 000 pts/node) | 404 ms | 440 ms | 404 requests were 404s (4 per node) |
| dense | 118 | 335 (created at 266) | 504 | 600–609 (**after the 2nd pass**) | 1 504 chunks, p50 16 ms each |
| ct | – | – | ~600 | – | 621 chunks |
| cmu1 | 121 | – | 3 593 | – | 2nd pass 3 888→4 903, refinement →8 103, GPU evictions →9 001, readiness 9 677 |
| neuromast (t=0) | 117 | 176 | 198 | – | 298 requests, 5.4 MB |

Throttled to 25 Mbps / 30 ms RTT (CDP emulation), same scenes, full-load wall time and link utilisation:

| scene | server | wall | avg Mbps (of 25) | workers ready | chunk p50 duration |
|---|---|---|---|---|---|
| dense | HTTP/1.1 (6 conns) | **10.6 s** | 7.4 | 4.06 s | 436 ms |
| dense | HTTP/2 | **4.0 s** | 19.6 | 4.31 s | 340 ms |
| ct | HTTP/1.1 | 6.4 s | 16.4 | 1.27 s | 639 ms |
| ct | HTTP/2 | 4.7 s | 22.6 | 4.51 s | 2 437 ms |
| ct, warm OPFS | HTTP/1.1 | **1.1 s** (0 network chunks, 621 L2 reads) | – | 1.10 s | – |
| dense, cold, persistent profile | HTTP/1.1 | 11.7 s | – | 5.20 s | – |
| dense, warm OPFS | HTTP/1.1 | **1.2 s** (0 network chunks, 1 504 L2 reads) | – | 1.05 s | – |

Request mix per session (any scene): 22 JS bundle files, **16 worker scripts, 17 `luxar_wasm_bg.wasm`**, 5–10 metadata probes (1–6 of them 404).

Cache counters after a cold load: L0 `hits == misses` on every 3-D scene (the second pass); L2 write-queue `dropped` = 252 (dense, ephemeral context; 0 in the persistent-profile run, so it is timing-dependent), 9 775 (cmu1), 0 (ct, bench).

### 3.2 Steady-state frame rate (continuous render, `probe-frames.mjs`)

rAF cadence while forcing a frame every rAF; main thread idle >88 % in all rows, so this is GPU time. Bloom on/off made no difference in any row.

| scene | view | DPR 1.0 | 0.9 | 0.75 | 0.5 | 0.35 |
|---|---|---|---|---|---|---|
| dense 1.5 M pts | default fit (cube ≈ 40×40 px) | **41.7 ms** | 50 | 75 | 83 | 84 |
| dense 1.5 M pts | zoomed (cube ≈ 155×155 px) | **8.3 ms** | – | – | 16.7 | – |
| cmu1 29.6 M splats | default fit | **108 ms** | 126 | 142 | 192 | – |
| ct 654 k splats | default | 8.3 (120 fps) | – | – | 8.3 | – |
| bench 100 k pts / 101 draws | default | 8.3 | – | – | 8.3 | – |
| neuromast 110 k splats | default | 8.3 | – | – | 8.3 | – |
| celegans | default | 8.3 | – | – | 8.3 | – |

The dense and cmu1 rows are the pathology: cost tracks **elements per pixel**, not pixels. Lowering DPR concentrates the same ≥1.5 px sprites into fewer tiles and gets slower. Zooming the same 1.5 M points to 15× the area made them 5× faster.

Main-thread trace during the dense orbit: 8 % busy, `animate` self-time 114 ms over 15 s → the render loop CPU cost is negligible; the wall is the GPU.

### 3.3 Adaptive DPR behaviour observed (dense, console)

```
 643  Scaled down: DPR 1.00 → 0.90 (FPS: 8.7, probing for U-shape)     <- fired DURING load
2175  Probe accepted at DPR 0.90: 8.7 → 19.2 FPS (×2.21)                <- load finished, misattributed
11445 Probe rejected at DPR 0.81: FPS 23.4 → 13.4 (×0.57) … floor set, retry in 30s
```
The controller reads loading hitches as GPU load, steps down, and its probe then credits the end of loading to the lower DPR (accepted ×2.21). The next step is correctly refused, and it parks at 0.9, which we measured at 50 ms vs 41.7 ms at 1.0. The idle restore to 1.0 works. It also repeatedly fired on bench2 (100 k points) during load.

### 3.4 Main-thread attribution on the large scene (cmu1, CDP trace + sampling profile)

| window | main thread busy | >50 ms tasks | top self-time (main thread only) |
|---|---|---|---|
| 0.1–3.6 s (first pass) | 103 % (overlapping) | 11 (max 259 ms) | luxar WASM 11 %, blob WASM (blosc) 4 %, `uv` 2 %, `loadCholeskyRanges` 2 % *(whole-run denominator)* |
| 3.6–9.7 s (v1 re-load + refinement + GPU evictions) | 91 % | 14 (max 634 ms; `animate` 357 ms) | **luxar WASM 39 %** (`wasm-function[3]` 24 %, `[8]` 7 %, `[14]` 7 %), blosc WASM 9 % + JS 2 %, `texSubImage2D` 8 %, `uv` 5 %, `passArrayF32ToWasm0` 3 %, fetch 2 %, idle 7 % |

Long tasks over the 10 s load: 25, summing **5 885 ms** (0.6–0.95 s blocked per second in seconds 5–9). The 100-node bench scene's 3.3 s "gap" between first paint and readiness is 400 idle-callback steps of the blend warm-up (trace: 400 `FireIdleCallback`, main thread 7 % busy).

### 3.5 4-D playback (neuromast, `probe-playback.mjs`, step time dimension, `--full` waits for `isLoading=false`)

| loop | first commit (median) | full ladder (median) | requests/step | KB/step | elements at first commit → full |
|---|---|---|---|---|---|
| 0 cold | 32 ms | 57 ms | 100 | 651 | 37 k → 80 k |
| 1 | 12 ms | 39 ms | 13 | 52 | 105 k → 111 k |
| 2 fully cached | **14 ms** | **40 ms** | 0 | 0 | 111 k → 111 k |

Zero I/O still costs 40 ms per timepoint: the ladder is paced at one rung per animation frame. celegans (tiny per-timepoint) stepped in 1–4 ms.

---

## 4. Optimisation opportunities, ranked

Each item names the mechanism, the evidence, and the expected effect. "Verified" means the code path was read and the effect measured or directly implied by a measurement.

### High impact

1. **Projected-density guard for points and splats** (render + load). Compute per node each frame: `elements / projectedBboxArea(px)`. Above a threshold (say 4 elements/px) either (a) cap the additive ladder rung the refinement loop is allowed to commit (load side: `refinement.ts` currently has no such rule), and (b) stochastically skip instances in the vertex shader by a per-node `uKeepFraction` (render side; the instance ID gives a free hash). Expected: dense default view 42 → <10 ms; cmu1 default view 108 → ~10–15 ms and a fraction of the bytes on hosted loads. The screen-area selector machinery (`lod-selector-math.ts`) already projects bboxes every frame, so the metric is nearly free.
2. **Remove the startup double pass.** `setup.ts:187` fires `updateAllNDNodes()` unconditionally after `loadScene` already committed the initial view. Skip it when the resolved view state equals the one `loadScene` used (or have `loadScene` publish its view version and let the init pass no-op). Verified by L0 hit/miss parity on every 3-D scene and by the cmu1 timeline (1 s re-load + a second accumulator growth per node). Expected: halve post-first-paint CPU/GPU work; remove ~1 s of the cmu1 freeze and the duplicate 1.1 M-splat texture uploads.
3. **Move ndim ≤ 3 gsplat projection (and points projection) off the main thread.** Flip the `ndim > 3` gate in `data-processor-gsplats.ts:294` to a size gate; for points, move `projectPointsTo3D` + compaction into the pool. Transfer inputs when the S-cache does not need them (or copy in the worker). Expected: the 39 % WASM share of cmu1's main-thread time moves to 15 idle workers; long tasks shrink to the commit (upload) itself.
4. **Eager worker pool + shared compiled WASM.** Create the pool at boot (before `openStore`), fetch and `WebAssembly.compile` once on the main thread and post the `WebAssembly.Module` to workers (structured-clone-able). Verified: 17 wasm fetches + 16 worker-script fetches per session; pool ready at 4.3–4.5 s under a 25 Mbps throttle, gating first paint; warm-cache ct load bounded by pool startup. Expected on hosted: first paint moves from "when the pool finally starts" to "when the first chunks land"; also prioritise worker assets over chunk fetches (`fetch(..., {priority: 'high'})`).
5. **Coverage-aware additive streaming for hosted loads.** Same metric as (1) on the load side: stop committing rungs beyond ~4 elements/px for the current view, resume when the camera zooms. cmu1 would stop after ~2 of 8 rungs at the opening framing (the memory notes it takes 68 s hosted to stream everything). This is the largest bandwidth win available without changing the format.

### Medium impact

6. **Batch the refinement pass.** `refinement.ts:203` awaits loaders serially and yields one rAF per rung. Allow up to N loaders (or a byte budget) per frame and commit all rungs already resident in the S-cache in one frame. Verified: fully cached timepoint costs 40 ms; a one-frame commit would be ≈ 8 ms. Playback of cached 4-D data goes from ~25 fps to display rate.
7. **Adaptive DPR: gate on GPU-boundness and direction.** Do not evaluate while `isLoading` or while a commit landed in the window; compare the probe against a same-load-state baseline (the first probe credited the end of loading to the DPR step, §3.3); and treat "lower DPR got slower" as evidence of overdraw, which should feed (1), not DPR. Verified: monotonic slow-down at lower DPR on both overdraw scenes.
8. **OPFS write queue.** Raise `maxDepth` to cover a burst (chunks/session is 10 k+ here), or coalesce writes into fewer, larger files, or drop to a "write after load settles" policy instead of drop-oldest. Verified 89 % dropped on cmu1. Also `?no-opfs` skipping `?clear-cache` (`multi-level-caching-store.ts:271`). *What landed:* the depth cap went to 16 384, the retained-byte cap is a quarter of the non-cache heap remainder (#2528, #2561), and overflow now drops the arriving write rather than the oldest — so the "write after load settles" alternative was never needed. The `?clear-cache` note is untouched.
9. **Chunk sizing guidance for HTTP/1.1 hosts.** `luxar optimise` already exists; the `dense` example is authored at 4 KB chunks (1 504 requests for 9.8 MB) and is 2.6× slower on HTTP/1.1 than HTTP/2. Make `luxar serve` speak HTTP/2 (or at least advise `optimise --profile hosting`), and have `luxar info --stats` flag stores whose chunk count exceeds ~1 request per 30 KB.
10. **Cut the 404 probes.** Gate `colors`/`radii`/`sharpnesses` opens on the node attrs or the consolidated metadata (the loader already has the group listing), as `scalars` is (`points-spatial-index-loader.ts:424`). 4 requests per node on hosted stores, uncached.
11. **Blend warm-up: one keeper per distinct program, not per node.** Key keepers by `getProgramCacheKey` (9 programs) instead of per source material; drop the retained per-node clones. Verified: 400 idle steps and 3.3 s readiness on 100 nodes.

### Lower impact / hygiene (verified in code, not measured)

12. Drop the L0 clone (`cached-zarr-array.ts:200`); zarrita copies into the selection anyway.
13. Transfer gsplat projection inputs when no S-cache retention is needed; pool the four worst-case output buffers.
14. RGBA32F → RGBA16F element textures (planned in `gpu-byte-budget.ts:27-41`), halves resident GPU bytes and the `texSubImage2D` share.
15. Free the LDR half-float target when FXAA is off (`resource-lifecycle.ts:71,118`); don't `dispose()` the HDR target on DPR steps when sample count is unchanged; coalesce adaptive-DPR resizes.
16. Partition frustum bounds: the loaded-geometry footprint is now cached per part and dirtied by geometry commits, while transformed authored boxes reuse caller-owned storage (#2605). An h2afva-shaped 44-part / 4,448-node selector microbenchmark (300 static frames, five runs) dropped the median registry pass from 2.93 ms/frame and 44 `Box3.setFromObject` subtree walks/frame to 0.061 ms/frame and zero steady-state walks. The separate O(groups²) containment in `render-order.ts:605-660,811-840` remains.
17. `gpuBufferPool.beginFrame()` is per commit cycle, not per frame (`gpu-buffer-pool.ts:172` vs `atomic-commit.ts:115`), so `evictionFrames` is mis-unit'd.
18. WebGPU backend ignores `updateRanges` (full re-upload) — parity gap for streaming scenes.
19. `get()` returns `undefined` after exhausted retries → zarrita fills silently (`multi-level-caching-store.ts:403,413`).
20. Console volume: 4 717 console lines for the 100-node load (47 per node), all through the interceptor. Cheap here, but it scales with node count.

---

## 5. Hypotheses tested and rejected

- **"Bloom is expensive."** 15 passes per frame at 1600×1000: no measurable change on any scene (ct, celegans, neuromast, dense, cmu1 all identical with bloom on/off).
- **"The sort worker was spinning for 3.3 s on the bench scene."** A sampling artefact (`_t` = the worker's async `initialize`); per-thread busy time shows the sort worker idle; the gap was the blend warm-up.
- **"Main-thread blosc decompression dominates loads."** Only ~11 % of main-thread samples even on the 277 MB scene; negligible on small-chunk scenes. It matters, but behind projection and upload.
- **"WebGL timer queries measure GPU frame time."** `EXT_disjoint_timer_query_webgl2` on ANGLE/Metal reported 40–56 ms for scenes that render at a solid 120 fps and 253 ms for cmu1 (actual 108 ms), and `gl.finish()` returned in 0.1 ms. Do not use it here; use the rAF cadence under forced continuous rendering (`probe-frames.mjs`).
- **"100 draw calls are slow."** bench renders 101 calls at 120 fps under forced continuous rendering; one earlier drag-orbit reading of 33 ms did not reproduce (that run also logged an adaptive-DPR step during load), so the drag-based number is treated as instrumentation noise.
- **"The 64-wide fetch gate is the hosted bottleneck."** On HTTP/2 the link ran at 78–90 % utilisation; the gate is fine. On HTTP/1.1 the browser's 6 connections are the wall, not the gate.

---

## 6. Observability gaps found while measuring

- No time-to-first-paint / time-to-full-load stamps anywhere; the monitor has cumulative counters only. The console lines (`Loading scene from`, `Loaded N … for`, `[GEOM] vN`, `Scene loaded successfully`, `Progressive: k/n LODs loaded`) are the only timeline. A `performance.mark` at each would make this audit a one-liner.
- `renderer.info` (draw calls, triangles, programs) is not surfaced; `info.autoReset` makes it a per-pass counter.
- `__luxarDebug.getState()` is installed only after the blend warm-up settles (up to 5 s after first paint), so readiness-gated probes under-observe the load (this bit me twice).
- `getLodLoadStats()` is off by default and the only view into lazy/additive level loads.
- Adaptive-DPR decisions are logged but not exposed as state (current operating DPR, floor, last verdict).

---

## 7. What was measured with what

- **Harness**: `@playwright/test` 1.62 driving Chromium 1234 headless with `--use-angle=metal --ignore-gpu-blocklist` (real GPU; verified by `UNMASKED_RENDERER_WEBGL`). Datasets served from the repo root by a CORS `http.server` (HTTP/1.1) on :8099 and a Node `http2` TLS server on :8443; viewer from `vite preview` on :4377.
- **Load probe** (`probe-load.mjs`): every request start/end/size/status, full console with timestamps, `PerformanceObserver('longtask')`, poll of `getState().totalElements/isLoading`, `cache.getStats()`, `workers.getStats()`, CDP `Performance.getMetrics`, optional CDP network throttle (`--net=hosted` = 25 Mbps / 30 ms), optional CPU throttle, optional Chrome trace with sampling profiler (`--trace`), optional persistent profile for warm-cache runs (`--profile=dir`).
- **Frame probe** (`probe-frames.mjs`): forces a render every rAF for 3 s per variant (base, bloom off/on, DPR sweep via `adaptiveDPRManager.setManualDPR`, auto-rotate) and reports cadence p50/p95 and passes per rAF from `renderer.info.render.frame`.
- **Playback probe** (`probe-playback.mjs`): `app.setDimensionValue` + `awaitDimensionUpdate`, optional wait for `isLoading=false`, per-step requests/bytes/long tasks, S-cache and L0 counters.
- **Trace analyser** (`analyze-trace.mjs`): renderer main-thread busy %, long tasks, self-time by trace event, main-thread-only sampling profile restricted to a time window.
- **Request analyser** (`analyze-requests.mjs`): in-flight and Mbps per 250 ms bin over the load window, request kinds, chunk-size percentiles.

## 8. Re-running

The measurements above are reproduced by the committed perf bench
(`packages/luxar-viewer/src/tests/e2e/viewer-audit-perf-bench.spec.ts`), which
replaced the ad-hoc probe kit. It writes `perf-results/<git sha>/results.json`
(medians of 3 per metric) and `scripts/perf-diff.mjs` diffs two runs.

```bash
# data server: the repo root (datasets/ underneath), CORS, HTTP/1.1 like the audit
python3 -m http.server 8099 --bind 127.0.0.1 &            # or any CORS-enabled static server

cd packages/luxar-viewer && pnpm build                      # production bundle (audit numbers are production)
LUXAR_PERF_BROWSER=chromium LUXAR_PERF_PREVIEW=1 LUXAR_PERF_HEADLESS=1 \
LUXAR_PERF_CHROME_ARGS=--use-angle=metal LUXAR_PERF_PORT=4380 \
LUXAR_PERF_DATA_BASE=http://127.0.0.1:8099 \
  pnpm test:perf:e2e -g "viewer audit"                    # LUXAR_PERF_AUDIT_SCENES=dense-points,cmu1-2d to subset

node scripts/perf-diff.mjs perf-results/<before>/results.json perf-results/<after>/results.json
```

Two measurement rules learned here: frame cadence is taken under forced
continuous render (`renderOnce()` every rAF), never with
`EXT_disjoint_timer_query` (unreliable on ANGLE/Metal); and frame rows come
from the bundled Chromium (`LUXAR_PERF_BROWSER=chromium`) because headless
system Chrome on macOS intermittently caps rAF at 30 Hz. `__luxarDebug.getPerf()`
exposes the load timeline, renderer counters, adaptive-DPR diagnostics and the
per-node projected density the bench reads.
