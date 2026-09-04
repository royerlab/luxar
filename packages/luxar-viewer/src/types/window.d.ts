/**
 * Global `Window` augmentation for Luxar's optional debug surface.
 *
 * The viewer attaches `window.__luxarDebug` only when debug mode is active
 * (via the `?debug` URL parameter or persisted `luxar.debug` localStorage
 * flag). Production builds without those flags do not attach the property.
 *
 * The type is intentionally permissive — `getState`, `getSceneLoader`, and
 * the cache helpers return ad-hoc dynamic shapes that are exercised
 * interactively from a browser console or by Playwright tests. We use
 * `unknown` rather than `any` so consumers must narrow before reading.
 */

import type { LuxarApp } from '../core/app';
import type { ConsoleInterceptor } from '../utils/console-interceptor';
import type { LuxarCamera } from '../utils/camera-utils';
import type { AnimationController } from '../scene/animation/animation-controller';
import type { InputHandler } from '../input';
import type { RenderingControls } from '../ui/rendering-controls';
import type { RecordingPanel } from '../ui/recording-panel';
import type { ControlsManager } from '../controls/controls-manager';
import type { PostProcessingManager } from '../rendering';
import type { SceneDimsManager } from '../scene/scene-dims-manager';
import type * as THREE from 'three';

declare global {
  interface Window {
    /**
     * Debug surface attached when `?debug` (or persisted `luxar.debug`
     * localStorage flag) is set. Undefined in production / non-debug
     * sessions.
     *
     * Populated in two stages:
     * - bootstrapStandalone() (or any caller that opts in) attaches
     *   `app`, `consoleInterceptor`, `version` before `init()` runs.
     * - LuxarApp.setupDebugInterface() extends with runtime references
     *   after `init()` completes, and sets `runtimeReady = true`.
     */
    __luxarDebug?: {
      app: LuxarApp;
      consoleInterceptor: ConsoleInterceptor;
      version: string;

      // Populated by LuxarApp.setupDebugInterface (post-init).
      scene?: THREE.Scene;
      camera?: LuxarCamera;
      // Either backend may be active — `THREE.WebGLRenderer` under
      // the production default, or `WebGPURenderer` when opted in via
      // `?renderer=webgpu` / `VITE_LUXAR_USE_WEBGPU=1`. Typed loosely
      // as `unknown` so the window declaration doesn't import from
      // `three/webgpu`; callers narrow via the `isWebGLRenderer`
      // helper or `caps.apiSurface`.
      renderer?: unknown;
      controls?: ControlsManager;
      postProcessing?: PostProcessingManager;
      animationController?: AnimationController;
      inputHandler?: InputHandler;
      renderingControls?: RenderingControls;
      /** Healthy-cadence estimate; undefined until the sampling window has enough frames. */
      fps?: number;
      /** Whether adaptive-DPR frame sampling is currently active. */
      fpsSamplingEnabled?: boolean;
      recordingPanel?: RecordingPanel;
      sceneDimsManager?: SceneDimsManager;
      runtimeReady?: boolean;

      // Helpers for interactive debugging / Playwright agents. Returned
      // shapes are intentionally dynamic — use `unknown` to force callers
      // to narrow.
      getState?: () => unknown;
      /**
       * Per-mesh effective draw order (path, blending bucket, depthWrite,
       * renderOrder, element count): opaque bucket first, then renderOrder
       * ascending within each bucket. Returned as `unknown` to force
       * narrowing; the concrete shape is `DrawOrderEntry[]` from
       * `core/app/debug/debug-state.ts`.
       */
      getDrawOrder?: () => unknown;
      renderOnce?: () => void;
      /**
       * Force a fresh, quiescent depth ordering for the current camera pose,
       * awaiting the SortWorker round-trip + chunked apply. For offline capture
       * (gallery orbit) which stops the rAF loop, leaving the per-frame
       * depth-sort scheduler dead. Resolves when settled or after a safety timeout.
       */
      resortDepthOrderingForCapture?: (maxWaitMs?: number) => Promise<void>;
      /**
       * Why depth sorting may be off (issue #1694): `idle` = never spawned /
       * init in flight, `ready` = sorts are flowing, `starved` = init missed
       * its 30 s deadline and a bounded retry is armed or in flight, `failed` =
       * permanently unavailable. `initTimeouts` counts the deadline misses
       * this session.
       */
      getDepthSortWorkerStatus?: () => {
        state: 'idle' | 'ready' | 'starved' | 'failed';
        initTimeouts: number;
      };
      getSceneLoader?: () => unknown;

      /**
       * Live accessors for the picking and overlay subsystems. Both are
       * disposed and reconstructed across dataset reloads, so callers must
       * re-read after a load. Returns `undefined` before init / between
       * disposals. Returned as `unknown` to avoid pulling the runtime
       * classes into the window type and to force narrowing on consumers.
       */
      getPickingSystem?: () => unknown;
      getOverlayManager?: () => unknown;

      /**
       * Render the error dialog directly with the supplied message, without
       * going through URL-routing or load failures. Used by visual-
       * regression tests so the dialog's appearance can be verified in
       * isolation from the routing logic in `app.ts:shouldShowBrowser`.
       */
      showError?: (message: string) => void;

      cache?: {
        getStats: () => unknown;
        listDatasets: () => unknown;
        clearL0: () => void;
        clearL1: () => void;
        clearL2: () => Promise<void>;
        clearAll: () => Promise<void>;
      };

      /**
       * Worker pool diagnostics. `getQueueDepth()` returns the aggregate
       * in-flight task count across the pool — useful for spotting
       * prefetch backpressure or task accumulation after rapid dataset
       * switches.
       */
      workers?: {
        getQueueDepth: () => number;
        getStats: () => unknown;
      };

      /**
       * Last viewer state exported via the keyboard shortcut handler in
       * input-handler.ts (in addition to the clipboard copy). Surfaced for
       * Playwright agent flows that want a stable reference across runs.
       */
      lastExportedState?: unknown;

      /**
       * Inject a synthetic scene (debug / perf-bench only). Builds a
       * lines / points / gsplats payload in JS via
       * `scene/synthetic-scene.ts`, wires it through the existing
       * material-manager + node-factory pipeline, and adds the resulting
       * mesh to the scene. Resolves once the node is committed and
       * renderable, with `{type, elementCount, <per-type count>, mesh}`
       * so the caller can capture the actual instance count it ran
       * against (`segmentCount` for lines — the original shape —
       * `pointCount` / `splatCount` for the others; `elementCount` is
       * the capacity-clamped drawn count, uniform across types).
       *
       * `blending` defaults to 'additive' for lines (historical bench
       * contract) and 'normal' for points/gsplats, whose injection also
       * emits the production depth-sort commit signals so the sort
       * subsystem engages; the first ordering lands asynchronously a
       * frame or two later. `clusters` (points/gsplats) is the gaussian
       * blob count, default 256. `width` / `stepScale` / `turnAngle`
       * are lines-only walk knobs (default 1.0 / 0.01 / unset = the
       * historical fully-random walk).
       *
       * Not present in production bundles when `?debug` is unset.
       */
      injectSyntheticScene?: (spec: {
        type: 'lines' | 'points' | 'gsplats';
        count: number;
        bounds?: number;
        seed?: number;
        clusters?: number;
        blending?: string;
        width?: number;
        stepScale?: number;
        turnAngle?: number;
      }) => Promise<
        | { type: 'lines'; segmentCount: number; elementCount: number; mesh: THREE.Mesh }
        | { type: 'points'; pointCount: number; elementCount: number; mesh: THREE.Mesh }
        | { type: 'gsplats'; splatCount: number; elementCount: number; mesh: THREE.Mesh }
      >;

      /**
       * Per-stage timing snapshot for lazy and additive LOD level loads
       * (`lazy:loadGSplats` / `lazy:process` / `lazy:commit` /
       * `lazy:release`), keyed by stage → {count, totalMs, avgMs, maxMs}.
       * Additive keys are bounded by geometry type × level × residency, plus
       * an `:aborted` key when the level load is cancelled; they deliberately
       * exclude slice/timepoint identifiers. Debug-only.
       */
      getLodLoadStats?: () => Record<
        string,
        { count: number; totalMs: number; avgMs: number; maxMs: number }
      >;
      /** Clear the lazy and additive LOD-load timing accumulator (debug-only). */
      resetLodLoadStats?: () => void;

      /**
       * Performance snapshot for probes: load-timeline marks and derived
       * durations, last-frame `renderer.info`, adaptive-DPR diagnostics,
       * worker stats and the wide `isSettled` predicate. Seeded by
       * bootstrap BEFORE `init()` (timeline only, `runtimeReady: false`)
       * and enriched by `installDebugInterface`. Concrete shape:
       * `PerfSnapshot` in `core/app/debug/perf-snapshot.ts`.
       */
      getPerf?: () => unknown;
      /** True as soon as `getPerf` exists (bootstrap), before `runtimeReady`. */
      perfReady?: boolean;
    };
  }
}

export {};
