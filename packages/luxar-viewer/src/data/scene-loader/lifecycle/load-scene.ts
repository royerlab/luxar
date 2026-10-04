/**
 * Initial scene-load orchestrator.
 *
 * Sequence:
 *   1. Reset the monitor's loader bindings.
 *   2. (The loader is one-shot: `SceneLoader.loadScene` refuses a second
 *      load, so there is never an earlier dataset of THIS loader to dispose.)
 *   3. Create a fresh dataset AbortController for this loader's worker calls.
 *   4. Set up L0 + L1/L2 caches; open the zarr root.
 *   5. Build the empty root THREE.Group + initialize scene dimensions
 *      from `scene_dimensions` metadata.
 *   6. Surface a toast when ndim > 16 (WASM ceiling — TS fallback works
 *      but is slower).
 *   7. Persist `viewer_config` + `position_bounds` onto the root group's
 *      userData for the UI to read.
 *   8. Build the scene graph (zarr group enumeration → SceneNode tree).
 *   9. Hand the root to `config.onSceneMetadata` (the scene manager frames
 *      the opening camera from the metadata), then recursively load every
 *      leaf via `loadSceneNodes`.
 *  10. Load overlay configs (screen-space annotations).
 *  11. Wire post-load monitor providers (cache stats, loader maps, etc.).
 *  12. Schedule progressive LOD refinement (all four geometry types) when
 *      any multi-LOD loader still has higher LODs to fetch (the initial-load
 *      path only fetches LOD 0; without this kick, higher LODs would not
 *      load until the user's first updateView).
 *
 * The orchestrator (`SceneLoader.loadScene`) is a thin wrapper that
 * builds the ctx, calls this helper, and returns the root group. The
 * helper writes new resource references back through the ctx setters.
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules, LogEmoji } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';
import { attachSceneGraphIndex } from '../../../utils/scene-graph-index';
import { warmUpDataWorkerPool } from '../../../workers/worker-pool';
import { retainCustomColormapTextures } from '../../../rendering/colormap-textures';
import { markLoad, noteRefinementComplete } from '../../../profiling/load-timeline';
import type { RefinementHoldReason } from '../../../types/data-monitor-types';
import { ZarrSceneAttrs, SceneDimensionAttrs } from '../../../types/zarr';
import { enforceFormatVersion } from '../../format-version';
import type { GeometryKind, LoaderConfig, SceneNode, ViewState } from '../../data-loader-types';
import type { DataLoader } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { MeshDataLoader } from '../../../types/mesh';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import { sceneEffectiveLineLoad, setSceneLineLoad } from '../../../types/line-primitive';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { UpdateProfiler } from '../../../profiling/update-profiler';
import type { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import type { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import type { SliceCache } from '../../../cache/slice-cache';
import type { CacheBudgets } from '../../../cache/heap-budget';
import type {
  SceneLoaderMonitorPort,
  FailedLoadsProviderPort,
} from '../../scene-loader-monitor-port';
import type { LODGroupRegistry } from '../../../scene/lod-group-registry';
import { setupCaches } from '../cache/cache-setup';
import { adoptChunkPacks } from '../cache/chunk-packs';
import { wireMonitorAfterLoad } from '../monitor/monitor-wiring';
import { createCommittedLODCountReader } from '../monitor/committed-lod-reader';
import { createDrawOrderProvider } from '../monitor/draw-order-provider';
import { loadOverlayConfigs } from '../../loaders';
import { loadBakedEnvironment } from '../../loaders/environment/environment-loader';
import { buildSceneGraph } from '../nodes/build-scene-graph';
import { loadSceneNodes } from '../nodes/load-scene-nodes';
import { reportLoadOutcome } from '../loaders/failure-report';
import { SceneIdentityWatchdog, canonicalJson } from '../../scene-identity-watchdog';
import type { RootDocumentFetch } from '../../../cache/root-document-prefetch';
import type { NodeBuildCtx } from '../nodes/build-ctx';

/**
 * Narrow context the load-scene path needs. Captures the orchestrator's
 * mutable resource references via getter callbacks so the helper can
 * read the latest snapshot of `gpuBufferPool` / `monitor` / etc., and
 * uses explicit setters to write the new dataset-scoped resources back.
 */
export interface LoadSceneCtx {
  /** Loader configuration (cache flags, prefetch flags). */
  config: LoaderConfig;
  /** Current view state — read after `initializeSceneDimensions` for logging. */
  viewState: () => ViewState;
  /** Active per-type loader maps for the post-load monitor wiring. */
  loaders: Map<string, DataLoader>;
  linesLoaders: Map<string, LinesDataLoader>;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  /**
   * Mesh loaders, for the post-load refinement kick and the monitor's
   * LOD-progress provider.
   *
   * Deliberately NOT passed to `reportLoadOutcome` below, which has never
   * carried mesh — widening the load-outcome grading is a separate change with
   * its own observable output. The refinement kick is needed because a mesh
   * reveal ladder commits only its first level during `loadScene`, so without it
   * the surface would sit at that first patch until the user moved a slider; the
   * LOD-progress provider is needed because the scene-graph converter stamps
   * `additiveSublods` on a laddered mesh parent, and a node with that stamp but
   * no live loader state renders as `LOD -/N` ("not streaming") throughout.
   */
  meshLoaders: Map<string, MeshDataLoader>;
  /** Current GPU buffer pool reference (may be null when disabled). */
  gpuBufferPool: () => GPUBufferPool | null;
  /** Current monitor reference. */
  monitor: () => SceneLoaderMonitorPort | null;
  /** Profiler reference. */
  profiler: UpdateProfiler | null;
  /** LOD-group registry for the live LOD-progress provider (substitutive levels). */
  lodGroupRegistry: LODGroupRegistry | null;

  // Lifecycle callbacks the orchestrator owns:
  normalizeURL(url: string): string;
  /**
   * Validate `scene_dimensions` blob and update the loader's viewState.
   * Implementation in `initialize-scene-dimensions.ts`; the orchestrator
   * writes the result back to its own viewState field.
   */
  initializeSceneDimensions(sceneDims: unknown): void;
  /**
   * Build the NodeBuildCtx used by the recursive scene-graph walk.
   * Built once before `loadSceneNodes` runs so the leaves all see the
   * same viewState snapshot.
   */
  makeNodeBuildCtx(): NodeBuildCtx;
  /** Stats helper used by the post-load monitor wiring. */
  updateVisibleCountsInMonitor(): void;
  /**
   * Registered failed-load paths — graded against the registered path set by
   * `reportLoadOutcome` to log an honest load outcome.
   */
  getFailedLoaderPaths(): string[];
  /** Recorded failure messages in the same encounter order as the paths. */
  getFailedLoaderReasons(): string[];
  /**
   * The shared failed-loads provider (paths + retry-all + per-path reason).
   * Single construction point so the monitor banner and the layers-panel
   * error badge read the same live failure set.
   */
  getFailedLoadsProvider(): FailedLoadsProviderPort;
  /**
   * Why this path's next rung is held back, if it is (`SceneLoader.
   * refinementHoldReason`: density gate or residency ceiling); the monitor
   * marks such rungs as held rather than streaming. Optional so headless test
   * contexts need not supply it.
   */
  refinementHoldReason?(path: string): RefinementHoldReason | null;
  /** Geometry kinds with a sweep-registered loader that has rungs left to stream. */
  kindsWithMoreLODs(): GeometryKind[];
  /**
   * Take the serialization lock and run the progressive refinement drain
   * (`PassScheduler.startRefinement`), which releases it — and, should the
   * orchestrator glue die, releases it and drains what queued meanwhile.
   */
  startRefinement(): void;

  // Resource-write setters — orchestrator nulls/sets its own fields.
  /**
   * Hand the started scene-identity watchdog to the orchestrator, which
   * owns its disposal (dataset switch / teardown).
   */
  setIdentityWatchdog(watchdog: SceneIdentityWatchdog | null): void;
  setDatasetAbortController(controller: AbortController | null): void;
  setCachingStore(store: MultiLevelCachingStore | null): void;
  setL0Cache(cache: DecompressedChunkCache | null): void;
  setSliceCache(cache: SliceCache | null): void;
  /** Resolved per-tier cache budgets (for the Settings popover readout). */
  setCacheBudgets(budgets: CacheBudgets | null): void;
  setZarrStore(store: zarr.Readable): void;
  setRootGroup(group: THREE.Group): void;
  setSceneGraph(graph: SceneNode): void;
}

/**
 * Synthesize a default scene-dimensions set for a standalone *bare node*
 * (a detached `.gsplats.zarr` carries no `scene_dimensions`). Derives the
 * dimension count from the node's `ndim` (or its bounds) and per-dim ranges
 * from `position_bounds`/`center_bounds` so framing and non-displayed-dim
 * centering work. The first three dims display as spatial; any axis >= 3 is
 * synthesized as a discrete index (step 1) so the initial slice lands on a real
 * frame rather than a continuous midpoint. A producer can still embed real
 * dimension names/ranges for richer nD navigation.
 *
 * Returns ``undefined`` for a true scene root (no node ndim/bounds to derive
 * from) so the caller falls through to the "no scene_dimensions" warning.
 */
function synthesizeSceneDimensionsFromNode(
  attrs: ZarrSceneAttrs | undefined
): SceneDimensionAttrs | undefined {
  const a = attrs as Record<string, unknown> | undefined;
  if (!a) return undefined;
  const bounds = (a.position_bounds ?? a.center_bounds) as
    { min?: number[]; max?: number[] } | undefined;
  const ndim =
    typeof a.ndim === 'number'
      ? a.ndim
      : Array.isArray(bounds?.min)
        ? bounds!.min!.length
        : undefined;
  if (!ndim || ndim < 1) return undefined;

  const SPATIAL_NAMES = ['X', 'Y', 'Z'];
  const dimensions = Array.from({ length: ndim }, (_, i) => {
    const lo = bounds?.min?.[i];
    const hi = bounds?.max?.[i];
    // The first up-to-3 axes are the displayed spatial dims. Any axis >= 3 on
    // a bare-node file is almost always a discrete index (time / channel), so
    // mark it discrete with step 1 — that routes it through the "start at the
    // range minimum with zero tolerance" slice path instead of treating it as
    // a continuous axis centered at the range midpoint with a 0.1 window
    // (which would silently miss integer frames and render a thin / empty slice).
    const isSpatial = i < 3;
    return {
      name: i < SPATIAL_NAMES.length ? SPATIAL_NAMES[i] : `dim${i}`,
      unit: isSpatial ? 'px' : 'index',
      display: isSpatial,
      spatial: isSpatial,
      ...(isSpatial ? {} : { discrete: true, step: 1 }),
      ...(typeof lo === 'number' && typeof hi === 'number'
        ? { range: [lo, hi] as [number, number] }
        : {}),
    };
  });
  return { dimensions };
}

/**
 * Hand the load-time root fetch's `ETag` to the identity watchdog, so its first
 * poll is conditional rather than a third full download of the root document.
 *
 * Asynchronous and non-blocking: by the time the root is open the shared fetch
 * has settled (the open read it, or validation did), and a fetch that has not
 * simply leaves the first poll unconditional, as before. The evidence passed is
 * the cheapest available — the token validation already derived when there is
 * one, else the bytes for the watchdog to check at its first probe.
 */
function seedWatchdogFromLoad(
  watchdog: SceneIdentityWatchdog,
  rootDocument: Promise<RootDocumentFetch> | null
): void {
  if (!rootDocument) return;
  void rootDocument.then((result) => {
    const served = result.served;
    if (!served?.etag) return;
    const token = result.peekToken();
    watchdog.seedFromLoad(
      token?.mode === 'content-hash'
        ? { doc: served.doc, etag: served.etag, contentHash: token.hash }
        : { doc: served.doc, etag: served.etag, body: served.bytes }
    );
  });
}

/**
 * Execute the full initial-load sequence and return the populated root
 * THREE.Group. Mutates the orchestrator's resource references via the
 * ctx setters.
 */
export async function loadScene(url: string, ctx: LoadSceneCtx): Promise<THREE.Group> {
  log.custom(LogEmoji.SCENE, Modules.SCENE_LOADER, `Loading scene from ${url}`);
  markLoad('loadStart', { url });
  // The profiler is manager-wide: drop the previous dataset's rows, or they
  // accumulate across switches and every merge's stale sweep walks them all.
  // reset() bumps the profiler's generation, so a session still in flight from
  // the outgoing dataset ends as a no-op instead of merging into the new tree.
  ctx.profiler?.reset();
  setSceneLineLoad(0);

  // Clear any existing loaders from monitor before loading new scene
  ctx.monitor()?.disconnectAllLoaders();

  // Spawn the data workers NOW, in parallel with the metadata fetch, rather
  // than letting the first chunk decode pay for it.
  // Measured on a hosted demo: lazy creation started the pool 1.93 s after
  // this point, by which time LOD 0's bytes had already arrived and were
  // simply waiting. Fire-and-forget and idempotent across dataset switches.
  warmUpDataWorkerPool();

  // Fresh abort source for THIS dataset. The loader threads it into its own
  // worker calls; the pool is shared by every host on the page, so it holds
  // no dataset signal of its own.
  const datasetAbortController = new AbortController();
  ctx.setDatasetAbortController(datasetAbortController);
  // ...which also names this dataset's hold on the page-wide custom-LUT cache
  // (released by `dispose.ts`).
  retainCustomColormapTextures(datasetAbortController);

  const cacheResult = await setupCaches(ctx.normalizeURL(url), {
    noCache: ctx.config.noCache,
    noSliceCache: ctx.config.noSliceCache,
    noOpfs: ctx.config.noOpfs,
    cacheDebug: ctx.config.cacheDebug,
    clearCache: ctx.config.clearCache,
    noPrefetch: ctx.config.noPrefetch,
    prefetchDebug: ctx.config.prefetchDebug,
    cacheBudgetMB: ctx.config.cacheBudgetMB,
  });
  ctx.setL0Cache(cacheResult.l0Cache);
  ctx.setSliceCache(cacheResult.sliceCache);
  ctx.setCachingStore(cacheResult.cachingStore);
  ctx.setCacheBudgets(cacheResult.budgets);
  const zarrStore = (await zarr.openStore(cacheResult.rawStore)) as zarr.Readable;
  ctx.setZarrStore(zarrStore);

  // Create root THREE.js group
  const rootGroup = new THREE.Group();
  rootGroup.name = 'LuxarScene';
  // Path lookups (commit / process / sweep / release hooks) resolve through this
  // index instead of an O(scene) `getObjectByName` walk each (B9a). It maintains
  // itself from the graph's own add/remove/rename events, so no builder below has
  // to know about it; a dataset switch builds a fresh root and a fresh index.
  attachSceneGraphIndex(rootGroup);
  ctx.setRootGroup(rootGroup);

  // Load scene metadata
  const rootLoc = zarr.root(zarrStore);
  // v3-first: the generic `open` guesses format 2 on a store object it has not
  // seen, costing two 404s (`.zattrs`, `.zgroup`) before the first data byte.
  const rootZarrGroup = await zarr.openGroupPreferV3(rootLoc);
  markLoad('metadataReady');
  const sceneAttrs = rootZarrGroup.attrs as ZarrSceneAttrs;
  await adoptChunkPacks(
    cacheResult.chunkPacks,
    rootLoc,
    (sceneAttrs as Record<string, unknown>).content_hash,
    cacheResult.rootIndexFromNetwork(),
    zarr.root(zarr.createStoreForUrl(url))
  );

  // Watch the dataset's identity from here on: a demo/dev server dying and a
  // different one later binding the same port would otherwise leave this tab
  // silently fronting the wrong scene. Started as soon as the root attrs are
  // read — not after the full load — so a swap during a LONG load (or a load
  // that subsequently fails because the server vanished) is caught too.
  // Identity is baselined on these attrs, so there is no window to race.
  const normalizedUrl = ctx.normalizeURL(url);
  if (SceneIdentityWatchdog.isWatchable(normalizedUrl)) {
    const loadedHash = (sceneAttrs as Record<string, unknown>)?.content_hash;
    // Hash-less fallback baseline: the attrs we actually loaded, canonically
    // serialized (these come from the store's consolidated metadata, the
    // probe reads the raw `.zattrs` — key order must not decide identity).
    // Root scene attrs are small (KBs); a stringify failure only downgrades
    // the watchdog to reachability-only for this (already hash-less) scene.
    let attrsJson: string | null;
    try {
      attrsJson = canonicalJson(sceneAttrs) ?? null;
    } catch {
      attrsJson = null;
    }
    const watchdog = new SceneIdentityWatchdog({
      datasetUrl: normalizedUrl,
      expectedContentHash: typeof loadedHash === 'string' ? loadedHash : null,
      expectedAttrsJson: attrsJson,
    });
    seedWatchdogFromLoad(watchdog, cacheResult.rootDocument);
    watchdog.start();
    ctx.setIdentityWatchdog(watchdog);
  }
  // The root is open: validation and the store open are both past the shared
  // load-time fetch, so let its bytes go (a later re-read reaches the server).
  cacheResult.releaseRootDocument();

  // Format-version policy, shared with the Python reader (data/format-version.ts
  // mirrors typing_utils/format_version.py): a supported version loads
  // silently, a same-major NEWER minor loads with a warning toast, and
  // anything else THROWS here — the error propagates to the `dataset-error`
  // path and the overlay names the version and the remedy. A scene root
  // carrying only the 0.1 legacy `luxar_version` key is a supported arm; a
  // detached .gsplats.zarr is dispatched on `format_type` to the gsplats set
  // (v3.0–v3.4 are all
  // readable: v3.1 splits the Cholesky factors into diag + offdiag; v3.2
  // renames the lod selector attrs to coverage_fraction — v3.0/3.1 stores with
  // the legacy attrs are auto-adapted by load-lod-group-node; v3.3 adds the
  // optional luxar_delta_v1 filter, undone transparently by the codec that
  // data/zarr.ts registers; v3.4 adds the screen-area lod selector while legacy
  // coverage stores read unchanged). Both supported sets are single-sourced
  // from format-contract/contract.yaml.
  enforceFormatVersion(sceneAttrs as Record<string, unknown>);

  // Initialize scene dimensions - CRITICAL for extend_to_all feature.
  // A standalone bare node carries no scene_dimensions; synthesize a default
  // (3D-safe) set from the node's ndim + bounds so framing / slicing work.
  const effectiveSceneDimensions =
    sceneAttrs?.scene_dimensions ?? synthesizeSceneDimensionsFromNode(sceneAttrs);
  if (effectiveSceneDimensions) {
    if (!sceneAttrs?.scene_dimensions) {
      log.info(
        Modules.SCENE_LOADER,
        'Synthesized scene_dimensions for a bare node ' +
          `(${effectiveSceneDimensions.dimensions.length}D); ` +
          'embed real dimensions for >3D navigation.'
      );
    }
    ctx.initializeSceneDimensions(effectiveSceneDimensions);
    rootGroup.userData.sceneDimensions = effectiveSceneDimensions;

    // Log dimension initialization status for debugging
    const vs = ctx.viewState();
    const ndim = vs.dimensions?.length ?? 0;
    if (ndim > 0) {
      log.success(
        Modules.SCENE_LOADER,
        `Scene dimensions initialized: ${ndim} dimensions, ` +
          `displayed=[${vs.displayDims.join(', ')}]`
      );
    }

    // Surface a user-facing toast when the scene exceeds the WASM
    // 16-dim ceiling — the worker auto-falls-back to TS, which is
    // correct but slower, and silent fallback can confuse users
    // wondering why interaction feels sluggish.
    if (ndim > 16) {
      notifier.toast(
        `Scene has ${ndim} dimensions — WASM acceleration limited to 16D, using TypeScript fallback. ` +
          'Consider reducing dimensions for better performance.',
        5000
      );
    }
  } else {
    log.warning(
      Modules.SCENE_LOADER,
      'No scene_dimensions found in scene metadata. extend_to_all features will not work.'
    );
  }

  // Extract viewer_config if present (Python API scene defaults)
  if (sceneAttrs?.viewer_config) {
    rootGroup.userData.viewerConfig = sceneAttrs.viewer_config;
    log.info(
      Modules.SCENE_LOADER,
      `Viewer config found in zarr: ${Object.keys(sceneAttrs.viewer_config).join(', ')}`
    );
  }

  // Store scene-level position bounds (from Python compiler). A bare gsplats
  // leaf root writes only `center_bounds`; fall back to it so auto-framing /
  // clipping work immediately for a standalone file (no geometry load needed).
  const rootBounds =
    sceneAttrs?.position_bounds ??
    ((sceneAttrs as Record<string, unknown>)?.center_bounds as
      typeof sceneAttrs.position_bounds | undefined);
  if (rootBounds) {
    rootGroup.userData.positionBounds = rootBounds;
    log.info(
      Modules.SCENE_LOADER,
      `Scene bounds loaded: min=[${rootBounds.min.join(', ')}], ` +
        `max=[${rootBounds.max.join(', ')}]`
    );
  }

  // Build scene graph
  const sceneGraph = await buildSceneGraph(rootLoc, sceneAttrs, zarrStore);
  ctx.setSceneGraph(sceneGraph);
  setSceneLineLoad(sceneEffectiveLineLoad(sceneGraph));

  // The opening camera is framed from this metadata before any node reads the view.
  ctx.config.onSceneMetadata?.(rootGroup);

  // Load points / lines / gsplats / nested groups recursively
  await loadSceneNodes(sceneGraph, rootGroup, rootLoc, ctx.makeNodeBuildCtx());

  // The store's base URL: image overlays and a store-relative `hdri` environment
  // url both resolve against it. The root digest rides along for the environment
  // bake, which records what it was baked against.
  rootGroup.userData.zarrBaseUrl = ctx.normalizeURL(url);
  rootGroup.userData.sceneContentHash = (sceneAttrs as Record<string, unknown> | undefined)
    ?.content_hash as string | undefined;

  // Load overlay configs (screen-space annotations)
  const overlayConfigs = await loadOverlayConfigs(zarrStore, rootLoc);
  if (overlayConfigs.length > 0) {
    rootGroup.userData.overlayConfigs = overlayConfigs;
    // Opaque-file access for image overlays (zipped stores have no child URLs).
    rootGroup.userData.readOverlayFile = async (path: string) =>
      zarrStore.get(path as zarr.AbsolutePath);
  }

  // A baked environment map (`luxar env attach`), if the store carries one that
  // matches this scene's digest. Parked on the root; `load-dataset.ts` hands it to
  // the scene environment together with the authored `viewer_config.environment`.
  // The index's silence is trusted only when it came from the network: `env
  // attach` keeps content_hash, so a warm L2 may still hold the pre-attach index.
  const bakedEnvironment = await loadBakedEnvironment(
    rootLoc,
    (sceneAttrs as Record<string, unknown> | undefined)?.content_hash as string | undefined,
    normalizedUrl,
    cacheResult.rootIndexFromNetwork()
  );
  if (bakedEnvironment) rootGroup.userData.bakedEnvironment = bakedEnvironment;

  // Post-load monitor-tab provider wiring (extracted to
  // scene-loader/monitor-wiring.ts).
  wireMonitorAfterLoad({
    monitor: ctx.monitor(),
    cachingStore: cacheResult.cachingStore,
    l0Cache: cacheResult.l0Cache,
    sliceCache: cacheResult.sliceCache,
    cacheTelemetryState: cacheResult.telemetryState,
    gpuBufferPool: ctx.gpuBufferPool(),
    profiler: ctx.profiler,
    loaders: ctx.loaders,
    linesLoaders: ctx.linesLoaders,
    gsplatLoaders: ctx.gsplatLoaders,
    meshLoaders: ctx.meshLoaders,
    lodGroupRegistry: ctx.lodGroupRegistry,
    sceneGraph,
    updateVisibleCounts: () => ctx.updateVisibleCountsInMonitor(),
    failedLoads: ctx.getFailedLoadsProvider(),
    refinementHold: ctx.refinementHoldReason ? (path) => ctx.refinementHoldReason!(path) : null,
    drawOrderProvider: createDrawOrderProvider(rootGroup),
    committedLODCounts: createCommittedLODCountReader(rootGroup),
  });

  // Report what ACTUALLY happened. Ordinary leaf-local LoaderErrors are swallowed
  // so the rest of the scene still builds; without this, a scene whose every node
  // failed logged success over an empty viewport. Container-wide archive faults
  // rethrow before this point. Totality is graded against the registered path set
  // because a failed lazy LOD level records a failure without registering.
  markLoad('sceneLoaded');
  reportLoadOutcome(
    ctx.getFailedLoaderPaths(),
    [...ctx.loaders.keys(), ...ctx.linesLoaders.keys(), ...ctx.gsplatLoaders.keys()],
    ctx.getFailedLoaderReasons()
  );

  // Schedule progressive LOD refinement after initial load: progressive
  // loaders of every type emit only their first rung on the initial load, and
  // the refinement drain streams the rest frame by frame. It holds the
  // serialization lock, so the init pipeline's first `updateView`, landing
  // while it runs, is queued — which cancels the drain into that pass.
  const kinds = ctx.kindsWithMoreLODs();
  if (kinds.length === 0) {
    // Nothing to stream: the load timeline's "refinement complete" milestone
    // is reached trivially (perf probes gate `isSettled` on it).
    noteRefinementComplete();
  } else {
    log.info(
      Modules.SCENE_LOADER,
      `Scheduling post-load progressive LOD refinement (${kinds.join(', ')})`
    );
    ctx.startRefinement();
  }

  return rootGroup;
}
