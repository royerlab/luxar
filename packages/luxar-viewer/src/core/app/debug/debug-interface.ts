import * as THREE from 'three';
import { log, Modules } from '../../../utils/log';
import { consoleInterceptor } from '../../../utils/console-interceptor';
import { buildInfo } from '../../../config/build-info';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { SceneLoaderManager } from '../../../data/scene-loader-manager';
import { getWorkerPool } from '../../../workers/worker-pool';
import { showError } from '../../../ui/error-overlay';
import { KeyAction } from '../../../input';
import { createInstancedLinesMesh } from '../../../rendering/line-geometry';
import { createInstancedGSplatsMesh } from '../../../rendering/gsplat-geometry';
import { createPointsGeometry } from '../../../rendering/node-factory/create-points-node';
import {
  clampLineCapacity,
  clampPointCapacity,
  clampSplatCapacity,
} from '../../../rendering/element-texture-layout';
import {
  syncLineMaterialWithGeometry,
  syncPointMaterialWithGeometry,
} from '../../../rendering/material-sync-helpers';
import { getBlendModeProgramWarmupStats } from '../../../rendering/webgl-blend-warmup';
import { materialManager, type BlendingMode } from '../../../rendering/material-manager';
import { normalizeBlendingMode } from '../../../rendering/blending-state';
import {
  getDepthSortWorkerStatus,
  noteDepthSortCommit,
  resortForCapture,
} from '../../../rendering/depth-sort-coordinator';
import { setCommittedData } from '../../../types/committed-data';
import { resolveLinePrimitiveForNode } from '../../../types/line-primitive';
import type { LoadedPointsData } from '../../../types/points';
import type { SyntheticInjectionResult, SyntheticSceneSpec } from '../../../scene/synthetic-scene';
import { computeDebugState, computeDrawOrder } from './debug-state';
import { buildDebugCacheHelpers } from './debug-cache-helpers';
import {
  setLodLoadStatsEnabled,
  snapshotLodLoadStats,
  resetLodLoadStats,
} from '../../../data/scene-loader/lod-load-stats';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { InputHandler } from '../../../input';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { RecordingPanel } from '../../../ui/recording-panel';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import type { PickingSystem } from '../../../rendering/picking/picking-system';
import type { OverlayManager } from '../../../ui/overlay-manager';
import type { LuxarApp } from '../../app';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../config/constants';
import { computePerfSnapshot } from './perf-snapshot';
import { getRendererInfoSnapshot, installRendererInfoSampler } from './renderer-info-sampler';
import { snapshotProjectedDensity } from '../../../scene/projected-density';
import type { LODGroupRegistry } from '../../../scene/lod-group-registry';

/**
 * Populate `window.__luxarDebug` with runtime components, helper
 * functions, and the synthetic-scene injector. No-op when
 * `ports.debug` is false. Preserves whatever properties the bootstrap
 * already seeded on `window.__luxarDebug` so the merged object exposes
 * both static (app/consoleInterceptor/version) and runtime fields.
 *
 * `app` is treated as an opaque handle — the helper only attaches it
 * to the global so embedders + tests can call back into LuxarApp.
 */
export interface InstallDebugInterfacePorts {
  debug: boolean;
  app: LuxarApp;
  sceneManager: SceneManager;
  animationController: AnimationController;
  inputHandler: InputHandler;
  renderingControls: RenderingControls;
  adaptiveDPRManager: AdaptiveDPRManager;
  recordingPanel: RecordingPanel | undefined;
  getPickingSystem: () => PickingSystem | undefined;
  getOverlayManager: () => OverlayManager | undefined;
  isInitialized: () => boolean;
}

export function installDebugInterface(ports: InstallDebugInterfacePorts): void {
  if (!ports.debug) {
    return;
  }

  log.info(Modules.LUXAR, 'Extending debug interface with runtime components');

  // Enable lazy-LOD-load per-stage timing plus additive per-level ladder
  // timing under ?debug only. The lazy ensureLoaded loads run outside any
  // updateView cycle, so the UpdateProfiler never captures them — this fills
  // that gap for navigation-cost diagnosis. Snapshot via
  // __luxarDebug.getLodLoadStats().
  setLodLoadStatsEnabled(true);

  // Sample renderer.info at every frame-end for getPerf().rendererInfo —
  // between frames Three's autoReset leaves only the last post-processing
  // pass in it. Debug sessions only; re-install replaces a prior subscription.
  installRendererInfoSampler(() => ports.sceneManager.renderer);

  // Extend whatever bootstrap seeded (app/consoleInterceptor/version). When
  // LuxarApp is instantiated outside the standalone-app entry point
  // (tests, embeds), bootstrap hasn't run; fall back to a fresh base.
  const existing = window.__luxarDebug ?? {
    app: ports.app,
    consoleInterceptor: consoleInterceptor,
    version: buildInfo().version,
  };

  window.__luxarDebug = {
    // Preserve existing properties from main.ts
    ...existing,

    // Add runtime components (only available after initialization)
    scene: ports.sceneManager.scene,
    // Live getter, NOT a snapshot: the camera is the one component the
    // scene manager REPLACES at runtime (perspective ↔ ortho swap in
    // camera-mode.ts). A snapshot taken here goes stale after the first
    // V-key mode switch — tests reading `__luxarDebug.camera.zoom` in
    // ortho mode would silently watch the abandoned perspective camera.
    get camera() {
      return ports.sceneManager.camera;
    },
    renderer: ports.sceneManager.renderer,
    controls: ports.sceneManager.controls,
    postProcessing: ports.sceneManager.postProcessing,
    animationController: ports.animationController,
    inputHandler: ports.inputHandler,
    renderingControls: ports.renderingControls,
    get fps() {
      const fps = ports.adaptiveDPRManager.getCurrentFPS();
      return fps > 0 ? fps : undefined;
    },
    get fpsSamplingEnabled() {
      return ports.adaptiveDPRManager.getState().enabled;
    },
    recordingPanel: ports.recordingPanel,
    sceneDimsManager: sceneDimsManager,
    app: ports.app,

    // Worker pool diagnostics. `queueDepth` is the aggregate count of
    // in-flight worker tasks; useful for spotting prefetch
    // backpressure or task accumulation after rapid dataset switches.
    // Returns 0 when the pool is idle / uninitialized.
    workers: {
      getQueueDepth: () => getWorkerPool().getQueueDepth(),
      getStats: () => getWorkerPool().getStats(),
    },

    // Helper function to get current state snapshot.
    // Implementation lives in `./debug-state.ts` so the
    // scene-walking logic can be unit-tested directly.
    //
    // `isLoading`, the pool stats and the refinement stop are all read from the
    // loader manager INSIDE the getter, per snapshot — capturing any of them
    // once here would freeze it at install time (when nothing is loading, the
    // pool is empty and refinement has not run) and hand every polling E2E
    // helper a permanent "idle".
    //
    // `gpuPoolStats` has been declared and documented on the context since the
    // field was added but production never passed it, so `state.gpuPool` was
    // always undefined; wiring it here is what makes the pool side legible at
    // all. Projected down to the debug subset rather than forwarded whole:
    // `PoolStats` also carries a per-type breakdown that no snapshot consumer
    // reads.
    getState: () =>
      computeDebugState({
        scene: ports.sceneManager.scene,
        camera: ports.sceneManager.camera,
        currentFov: ports.sceneManager.currentFov,
        isAnimating: ports.animationController.isActive,
        initialized: ports.isInitialized(),
        isLoading: SceneLoaderManager.getInstance().isAnyLoadPassInProgress(),
        dims: sceneDimsManager.getDims(),
        gpuPoolStats: () => {
          const stats = SceneLoaderManager.getInstance().gpuPoolStats();
          if (!stats) return undefined;
          return {
            activeBuffers: stats.activeBuffers,
            pooledBuffers: stats.pooledBuffers,
            activeBytes: stats.activeBytes,
            pooledBytes: stats.pooledBytes,
            totalBytes: stats.totalBytes,
            largestPooledBytes: stats.largestPooledBytes,
            evictions: stats.evictions,
            byteBudgetEvictions: stats.byteBudgetEvictions,
          };
        },
        refinementResidency: () => SceneLoaderManager.getInstance().refinementResidencyStop(),
      }),

    // Helper to trigger a single frame render (for stable screenshots)
    renderOnce: () => {
      ports.animationController.startAnimation();
    },

    // Effective cross-node draw order of every data mesh: opaque bucket
    // first (THREE renders its whole opaque list before the transparent
    // list), then renderOrder ascending within each bucket. Each
    // entry carries the blending bucket (opaque/transparent), depthWrite,
    // the resolved renderOrder, and the element count — enough to diagnose a
    // compositing-order bug (e.g. a backdrop drawn after the content in front
    // of it) from the console. Implementation in `./debug-state.ts` so it can
    // be unit-tested against real THREE fixtures.
    getDrawOrder: () => computeDrawOrder(ports.sceneManager.scene),

    // Force a fresh, quiescent depth ordering for the current camera pose,
    // awaiting the worker sort + chunked apply. Used by offline capture
    // (gallery orbit) which stops the rAF loop, leaving the per-frame
    // depth-sort scheduler dead — without this each frame renders the
    // permutation frozen at the pre-orbit pose. Resolves when ordering is
    // settled (or after an internal safety timeout).
    resortDepthOrderingForCapture: (maxWaitMs?: number) => resortForCapture(maxWaitMs),

    // Why the scene may be drawn UNSORTED (issue #1694). `idle` = the sort
    // worker was never spawned or its init is still in flight; `ready` =
    // sorts are flowing; `starved` = init missed its 30 s deadline (a busy
    // main thread during a multi-million-element load) and a bounded retry
    // is armed or in flight, so sorting is off MEANWHILE, not for the
    // session; `failed` =
    // permanently unavailable (dead/blocked worker script, or the retries
    // ran out). `initTimeouts` counts the deadline misses. Exposed because
    // the degrade used to be visible only as one console line, which is
    // impossible to check after the fact from an E2E run or a bug report.
    getDepthSortWorkerStatus: () => getDepthSortWorkerStatus(),

    // Helper to get scene loader manager (for cache inspection)
    getSceneLoader: () => {
      return SceneLoaderManager.getInstance();
    },

    // Live accessors for the picking + overlay subsystems. Both are
    // disposed and reconstructed across dataset reloads, so a direct
    // snapshot would go stale; the accessor pattern always returns
    // the current instance (or undefined before init / between
    // disposals).
    getPickingSystem: () => ports.getPickingSystem(),
    getOverlayManager: () => ports.getOverlayManager(),

    // Cache-specific helpers — thin wrappers over the SceneLoader cache
    // API. Implementation lives in `./debug-cache-helpers.ts` so the
    // not-found / no-cache / success branches can be unit-tested
    // directly with a stub loader.
    cache: buildDebugCacheHelpers(() => SceneLoaderManager.getInstance().getDefaultLoader()),

    // Test-friendly hook for the error-dialog component. Lets
    // visual-regression specs render the dialog directly without going
    // through URL-routing failure paths (whose semantics evolve
    // independently of the dialog's appearance).
    showError: (message) =>
      showError(message, (actionId) => ports.app.shortcutForAction(actionId), {
        datasetBrowser: KeyAction.toggleDatasetBrowser,
        help: KeyAction.toggleHelp,
      }),

    // Debug-only synthetic-scene injector for the perf bench. Builds
    // a large lines / points / gsplats payload purely in JS, wires it
    // through the existing material-manager + node-factory pipeline,
    // and adds the resulting mesh to the scene. Returns `{type,
    // elementCount, <per-type count>, mesh}` so the bench can capture
    // the actual instance count it ran against. `synthetic-scene.ts`
    // is imported dynamically so the synthetic builders stay out of
    // the main chunk; the geometry/material modules are already in
    // the main bundle (they're production modules), so importing them
    // statically here costs nothing extra.
    //
    // Blending defaults preserve the historical lines contract
    // ('additive') while points/gsplats default to 'normal' so the
    // depth-sort subsystem engages; `spec.blending` overrides either.
    // Points/gsplats also stamp `committedData` and call
    // `noteDepthSortCommit` — the same signals the production commit
    // path emits — so the SortWorker registers the node and orderings
    // actually apply (the coordinator drops orderings for meshes
    // without the stamp).
    injectSyntheticScene: async (spec: SyntheticSceneSpec): Promise<SyntheticInjectionResult> => {
      // [core OOS] Wrap the dynamic synthetic-scene import +
      // injection body in try/catch. Pre-fix, a rejection in the
      // dynamic import (bundle issue, transient network failure,
      // code-split chunk missing) or in the generators /
      // material creation became an unhandled promise rejection.
      // Debug consumers (`__luxarDebug.injectSyntheticScene({...})`)
      // typically don't `await` with their own try/catch, so a URL
      // like `?debug=1&inject=lines` could leave the page in a broken
      // state with no visible signal. Now we surface the failure to
      // the user-facing error overlay + log.error AND re-throw so
      // callers that DO `await` still see the rejection.
      try {
        const scene = ports.sceneManager.scene;
        // 'additive' default for lines = the historical bench contract
        // (and normalizeBlendingMode's undefined→'additive' identity);
        // points/gsplats default to 'normal' (order-dependent).
        const blendingMode: BlendingMode =
          spec.blending !== undefined
            ? normalizeBlendingMode(spec.blending)
            : spec.type === 'lines'
              ? 'additive'
              : 'normal';

        if (spec.type === 'lines') {
          const { generateSyntheticLines, syntheticLinesBoundsDiagonal } =
            await import('../../../scene/synthetic-scene');
          const cfg = generateSyntheticLines(spec);
          const maxWidth = spec.width ?? 1.0;
          // Build the visual material directly through the
          // material-manager so the same blending / dispatch logic
          // production uses applies. Picking material is intentionally
          // skipped — the synthetic scenarios don't exercise picking.
          // Mirror createLinesNode's authored per-node inputs; the resolver
          // also applies the installed scene load. The perf bench's bootstrap
          // scene is far below the threshold, so the count AND rendered-width
          // factor (normalized by the generation volume the walk fills) still
          // select the same primitive production would for a node of this
          // size. An explicit ?linePrimitive= arm still wins inside the
          // resolver, so bench A/B arms are unaffected.
          const material = materialManager.getLineMaterial({
            blendingMode,
            opacity: 1.0,
            gamma: 1.0,
            intensity: 1.0,
            offset: 0.0,
            primitive: resolveLinePrimitiveForNode({
              nSegments: cfg.segmentCount,
              maxWidth,
              bboxDiagonal: syntheticLinesBoundsDiagonal(spec),
            }),
          });
          const mesh = createInstancedLinesMesh(cfg, material);
          const clamped = clampLineCapacity(cfg.segmentCount);
          mesh.userData = {
            nodeType: 'lines',
            // `n_segments` mirrors the authored total a production
            // `.zattrs` carries (`LinesMetadata.n_segments`) — the
            // stable per-node count a size-aware primitive policy keys
            // on. Without it a synthetic 10 M-segment bench scene would
            // read as "no authored count" and resolve as a tiny scene.
            attrs: { n_segments: cfg.segmentCount },
            // Mirrors createLinesNode's `attrs.max_width`: the widest
            // authored width, which `spec.width` now controls (the
            // thick perf scenarios inject 3.0, not the 1.0 default).
            maxWidth,
            visibleSegmentCount: clamped,
            synthetic: true,
          };
          // Bind the geometry-owned line texture on the per-node material —
          // without this the shader samples the shared zero placeholder and
          // the bench renders N invisible instances (segment data lives in
          // `uLineTex` since the texture-storage migration; the production
          // paths bind via createLinesNode / the commit sync, neither of
          // which runs for this debug injection).
          syncLineMaterialWithGeometry(mesh);
          scene.add(mesh);
          // Kick the renderer so the new mesh is uploaded before the
          // bench's first measurement frame.
          ports.animationController.startAnimation();
          return { type: 'lines', segmentCount: cfg.segmentCount, elementCount: clamped, mesh };
        }

        if (spec.type === 'gsplats') {
          const { generateSyntheticGSplats } = await import('../../../scene/synthetic-scene');
          const cfg = generateSyntheticGSplats(spec);
          const material = materialManager.getGSplatMaterial({
            blendingMode,
            opacity: 1.0,
            absorption: 1.0,
            gamma: 1.0,
            intensity: 1.0,
            offset: 0.0,
            truncationRadius: GSPLAT_DEFAULT_TRUNCATION_RADIUS,
          });
          // The real gsplat mesh path: splat texture + `aSortedIndex`
          // storage, identity ordering, footprint-expanded bounds, and
          // the creation-time `updateSplatTexture` bind (no sync helper
          // needed — there is no pick node here).
          const mesh = createInstancedGSplatsMesh(cfg, material);
          mesh.name = 'synthetic-gsplats';
          const clamped = clampSplatCapacity(cfg.splatCount);
          mesh.userData = {
            nodeType: 'gsplats',
            attrs: {},
            visibleSplatCount: clamped,
            synthetic: true,
            _layerMaterialCloned: true,
          };
          // Commit stamp — the depth-sort coordinator drops resolved
          // orderings (and skips per-frame re-sort triggers) for meshes
          // without it: absence is its LOD-demotion signal.
          setCommittedData(mesh, cfg);
          scene.add(mesh);
          // The production commit chokepoint: registers the centers
          // with the SortWorker and dispatches the first sort when the
          // live mode is order-dependent. The thunk hands the worker a
          // FRESH buffer (it is transferred) and only pays the copy on
          // the sorted path.
          noteDepthSortCommit(mesh, () => cfg.centers.slice(0, clamped * 3), clamped);
          ports.animationController.startAnimation();
          return { type: 'gsplats', splatCount: cfg.splatCount, elementCount: clamped, mesh };
        }

        if (spec.type === 'points') {
          const { generateSyntheticPoints } = await import('../../../scene/synthetic-scene');
          const cfg = generateSyntheticPoints(spec);
          const data: LoadedPointsData = {
            positions: cfg.positions,
            colors: cfg.colors,
            radii: cfg.radii,
            sharpness: cfg.sharpness,
            pointCount: cfg.pointCount,
            ndim: 3,
            metadata: {
              totalPoints: cfg.pointCount,
              loadedPoints: cfg.pointCount,
              bounds: new THREE.Box3(
                new THREE.Vector3(...cfg.boundsMin),
                new THREE.Vector3(...cfg.boundsMax)
              ),
              usedSpatialIndex: false,
              dtypes: {},
            },
          };
          // The real points geometry path: point texture +
          // `aSortedIndex` storage, identity ordering,
          // footprint-expanded bounds, presence stamps.
          const geometry = createPointsGeometry(data, cfg.maxRadius);
          const material = materialManager.getPointMaterial({
            blendingMode,
            opacity: 1.0,
            absorption: 1.0,
            gamma: 1.0,
            intensity: 1.0,
            offset: 0.0,
            radiusScale: (geometry.userData.radiusScale as number | undefined) ?? 1.0,
          });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.name = 'synthetic-points';
          mesh.frustumCulled = true;
          const clamped = clampPointCapacity(cfg.pointCount);
          mesh.userData = {
            nodeType: 'points',
            attrs: {},
            maxRadius: cfg.maxRadius,
            visiblePointCount: clamped,
            synthetic: true,
            _layerMaterialCloned: true,
          };
          // Bind the geometry-owned point texture on the render
          // material (mirrors createPointsNode's creation-time bind).
          syncPointMaterialWithGeometry(mesh);
          setCommittedData(mesh, data);
          scene.add(mesh);
          noteDepthSortCommit(mesh, () => cfg.positions.slice(0, clamped * 3), clamped);
          ports.animationController.startAnimation();
          return { type: 'points', pointCount: cfg.pointCount, elementCount: clamped, mesh };
        }

        throw new Error(`Unknown synthetic scene type: ${String(spec.type)}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log.error(Modules.LUXAR, `__luxarDebug.injectSyntheticScene failed: ${message}`, error);
        showError(
          `Synthetic-scene injection failed: ${message}`,
          (actionId) => ports.app.shortcutForAction(actionId),
          {
            datasetBrowser: KeyAction.toggleDatasetBrowser,
            help: KeyAction.toggleHelp,
          }
        );
        throw error;
      }
    },

    // Per-stage timing for lazy LOD level loads (fetch/decode, process,
    // commit, release), plus additive per-level ladder keys. Reset before a
    // measurement drive, snapshot after. See
    // data/scene-loader/lod-load-stats.ts. Only meaningful under ?debug.
    getLodLoadStats: () => snapshotLodLoadStats(),
    resetLodLoadStats: () => resetLodLoadStats(),

    // Perf snapshot for probes (see perf-snapshot.ts). Bootstrap seeds a
    // timeline-only version before init(); this one adds the runtime hooks.
    // Every hook is read INSIDE the getter, per snapshot, for the same
    // reason `isLoading` is: capturing once here would freeze it.
    perfReady: true,
    getPerf: () =>
      computePerfSnapshot({
        rendererInfo: getRendererInfoSnapshot,
        adaptiveDpr: () => ports.adaptiveDPRManager.getDiagnostics(),
        workers: () => getWorkerPool().getStats(),
        density: snapshotProjectedDensity,
        cache: () => SceneLoaderManager.getInstance().getDefaultLoader()?.getCacheStats() ?? null,
        blendWarmup: getBlendModeProgramWarmupStats,
        isUpdateInProgress: () => {
          const loader = SceneLoaderManager.getInstance().getDefaultLoader() as {
            isUpdateInProgress?: () => boolean;
          } | null;
          return loader?.isUpdateInProgress?.() ?? false;
        },
        isAnyLoadPassInProgress: () => SceneLoaderManager.getInstance().isAnyLoadPassInProgress(),
        isAnyLodLevelLoading: () => {
          const loader = SceneLoaderManager.getInstance().getDefaultLoader() as {
            lodGroupRegistry?: LODGroupRegistry | null;
          } | null;
          return loader?.lodGroupRegistry?.isAnyLevelLoading() ?? false;
        },
      }),

    // Mark that runtime components are now available
    runtimeReady: true,
  };

  // Log available debug commands
  log.info(Modules.LUXAR, 'Debug interface ready:');
  log.info(Modules.LUXAR, '  __luxarDebug.getState() - Get current state snapshot');
  log.info(
    Modules.LUXAR,
    '  __luxarDebug.getDrawOrder() - Per-mesh draw order (bucket, depthWrite, renderOrder)'
  );
  log.info(Modules.LUXAR, '  __luxarDebug.renderOnce() - Trigger single frame render');
  log.info(
    Modules.LUXAR,
    '  __luxarDebug.resortDepthOrderingForCapture() - Re-sort depth ordering for the current pose (offline capture)'
  );
  log.info(
    Modules.LUXAR,
    '  __luxarDebug.getDepthSortWorkerStatus() - Why a scene may be drawn unsorted (idle/ready/starved/failed + deadline misses)'
  );
  log.info(Modules.LUXAR, '  __luxarDebug.scene - Access THREE.js scene');
  log.info(Modules.LUXAR, '  __luxarDebug.camera - Access camera');
  log.info(Modules.LUXAR, '  __luxarDebug.app - Access LuxarApp instance');
  log.info(Modules.LUXAR, '  __luxarDebug.cache.getStats() - Get cache statistics (L0, L1, L2)');
  log.info(
    Modules.LUXAR,
    '  __luxarDebug.workers.getQueueDepth() - In-flight worker task count (backpressure diagnostic)'
  );
  log.info(
    Modules.LUXAR,
    '  __luxarDebug.cache.listDatasets() - List all cached datasets (URL, hash, size)'
  );
  log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL0() - Clear L0 decompressed chunk cache');
  log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL1() - Clear L1 memory cache');
  log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL2() - Clear L2 OPFS cache');
  log.info(Modules.LUXAR, '  __luxarDebug.cache.clearAll() - Clear all caches (L0, L1, L2)');
}
