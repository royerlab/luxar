import { log, Modules } from '../../../utils/log';
import { consoleInterceptor } from '../../../utils/console-interceptor';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { SceneLoaderManager } from '../../../data/scene-loader-manager';
import { getWorkerPool } from '../../../workers/worker-pool';
import { showError } from '../../../ui/error-overlay';
import { createInstancedLinesMesh } from '../../../rendering/line-geometry';
import { materialManager } from '../../../rendering/material-manager';
import { computeDebugState } from './debug-state';
import { buildDebugCacheHelpers } from './debug-cache-helpers';
import {
  setLodLoadStatsEnabled,
  snapshotLodLoadStats,
  resetLodLoadStats,
} from '../../../data/scene-loader/lod-load-stats';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { InputHandler } from '../../../input/input-handler';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { RecordingPanel } from '../../../ui/recording-panel';
import type { PickingSystem } from '../../../rendering/picking/picking-system';
import type { OverlayManager } from '../../../ui/overlay-manager';
import type { LuxarApp } from '../../app';

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

  // Enable lazy-LOD-load per-stage timing under ?debug only. The lazy
  // ensureLoaded loads run outside any updateView cycle, so the
  // UpdateProfiler never captures them — this fills that gap for
  // navigation-cost diagnosis. Snapshot via __luxarDebug.getLodLoadStats().
  setLodLoadStatsEnabled(true);

  // Extend whatever bootstrap seeded (app/consoleInterceptor/version). When
  // LuxarApp is instantiated outside the standalone-app entry point
  // (tests, embeds), bootstrap hasn't run; fall back to a fresh base.
  const existing = window.__luxarDebug ?? {
    app: ports.app,
    consoleInterceptor: consoleInterceptor,
    version: '1.0.0',
  };

  window.__luxarDebug = {
    // Preserve existing properties from main.ts
    ...existing,

    // Add runtime components (only available after initialization)
    scene: ports.sceneManager.scene,
    camera: ports.sceneManager.camera,
    renderer: ports.sceneManager.renderer,
    controls: ports.sceneManager.controls,
    postProcessing: ports.sceneManager.postProcessing,
    animationController: ports.animationController,
    inputHandler: ports.inputHandler,
    renderingControls: ports.renderingControls,
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
    getState: () =>
      computeDebugState({
        scene: ports.sceneManager.scene,
        camera: ports.sceneManager.camera,
        currentFov: ports.sceneManager.currentFov,
        isAnimating: ports.animationController.isActive,
        initialized: ports.isInitialized(),
        dims: sceneDimsManager.getDims(),
      }),

    // Helper to trigger a single frame render (for stable screenshots)
    renderOnce: () => {
      ports.animationController.startAnimation();
    },

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
    showError,

    // Debug-only synthetic-scene injector for the perf bench. Builds
    // a large `InstancedLinesMeshConfig` purely in JS, wires it
    // through the existing material-manager + node-factory pipeline,
    // and adds the resulting mesh to the scene. Returns `{type,
    // segmentCount, mesh}` so the bench can capture the actual
    // instance count it ran against. `synthetic-scene.ts` is imported
    // dynamically so the synthetic-line builder stays out of the main
    // chunk; `line-geometry` and `material-manager` are already in
    // the main bundle (they're production modules), so importing them
    // statically here costs nothing extra.
    injectSyntheticScene: async (spec: {
      type: 'lines';
      count: number;
      bounds?: number;
      seed?: number;
    }) => {
      // [core OOS] Wrap the dynamic synthetic-scene import +
      // injection body in try/catch. Pre-fix, a rejection in the
      // dynamic import (bundle issue, transient network failure,
      // code-split chunk missing) or in `generateSyntheticLines` /
      // material creation became an unhandled promise rejection.
      // Debug consumers (`__luxarDebug.injectSyntheticScene({...})`)
      // typically don't `await` with their own try/catch, so a URL
      // like `?debug=1&inject=lines` could leave the page in a broken
      // state with no visible signal. Now we surface the failure to
      // the user-facing error overlay + log.error AND re-throw so
      // callers that DO `await` still see the rejection.
      try {
        const { generateSyntheticLines } = await import('../../../scene/synthetic-scene');
        const cfg = generateSyntheticLines(spec);
        // Build the visual material directly through the
        // material-manager so the same blending / dispatch logic
        // production uses applies. Picking material is intentionally
        // skipped — the synthetic scenarios don't exercise picking.
        const material = materialManager.getLineMaterial({
          blendingMode: 'additive',
          opacity: 1.0,
          gamma: 1.0,
          intensity: 1.0,
          offset: 0.0,
        });
        const mesh = createInstancedLinesMesh(cfg, material);
        mesh.userData = {
          nodeType: 'lines',
          attrs: {},
          maxWidth: 1.0,
          visibleSegmentCount: cfg.segmentCount,
          synthetic: true,
        };
        ports.sceneManager.scene.add(mesh);
        // Kick the renderer so the new mesh is uploaded before the
        // bench's first measurement frame.
        ports.animationController.startAnimation();
        return { type: spec.type, segmentCount: cfg.segmentCount, mesh };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log.error(Modules.LUXAR, `__luxarDebug.injectSyntheticScene failed: ${message}`, error);
        showError(`Synthetic-scene injection failed: ${message}`);
        throw error;
      }
    },

    // Per-stage timing for lazy LOD level loads (fetch/decode, process,
    // commit, release). Reset before a measurement drive, snapshot after.
    // See data/scene-loader/lod-load-stats.ts. Only meaningful under ?debug.
    getLodLoadStats: () => snapshotLodLoadStats(),
    resetLodLoadStats: () => resetLodLoadStats(),

    // Mark that runtime components are now available
    runtimeReady: true,
  };

  // Log available debug commands
  log.info(Modules.LUXAR, 'Debug interface ready:');
  log.info(Modules.LUXAR, '  __luxarDebug.getState() - Get current state snapshot');
  log.info(Modules.LUXAR, '  __luxarDebug.renderOnce() - Trigger single frame render');
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
