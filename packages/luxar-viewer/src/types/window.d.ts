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
import type { LuxarCamera } from '../scene/camera-utils';
import type { AnimationController } from '../scene/animation-controller';
import type { InputHandler } from '../input/input-handler';
import type { RenderingControls } from '../ui/rendering-controls';
import type { RecordingPanel } from '../ui/recording-panel';
import type { ControlsManager } from '../controls/controls-manager';
import type { PostProcessingManager } from '../rendering/post-processing/post-processing-manager';
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
      renderer?: THREE.WebGLRenderer;
      controls?: ControlsManager;
      postProcessing?: PostProcessingManager;
      animationController?: AnimationController;
      inputHandler?: InputHandler;
      renderingControls?: RenderingControls;
      recordingPanel?: RecordingPanel;
      sceneDimsManager?: SceneDimsManager;
      runtimeReady?: boolean;

      // Helpers for interactive debugging / Playwright agents. Returned
      // shapes are intentionally dynamic — use `unknown` to force callers
      // to narrow.
      getState?: () => unknown;
      renderOnce?: () => void;
      getSceneLoader?: () => unknown;

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
       * Last viewer state exported via the keyboard shortcut handler in
       * input-handler.ts (in addition to the clipboard copy). Surfaced for
       * Playwright agent flows that want a stable reference across runs.
       */
      lastExportedState?: unknown;
    };
  }
}

export {};
