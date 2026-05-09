/**
 * Phase 21C: optional factory overrides for the heavy components
 * constructed by `LuxarApp.init()`.
 *
 * Each factory is keyed by the field name on `LuxarApp` and returns
 * a fresh instance with the same arguments the inline `new X(...)`
 * site used to pass. The default factories simply call the
 * constructor; the production path is byte-for-byte equivalent to
 * the pre-21C inline code.
 *
 * The seam exists for embedders + tests:
 *   - Embedders can substitute alternate scene managers, custom
 *     recording panels, etc., without subclassing LuxarApp.
 *   - Tests can inject pre-built stubs in place of `vi.mock(...)` for
 *     these specific components, making mock setup explicit data
 *     instead of module-level magic.
 *
 * Only the heavy, frequently-mocked components are exposed here.
 * Smaller helpers (PerformanceMonitor, DebugConsole, ScaleBar, …)
 * stay inline — adding factories for them would cost more
 * indirection than it saves.
 */

import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { RenderingControls } from '../ui/rendering-controls';
import { RecordingPanel } from '../ui/recording-panel';
import { LayersPanel } from '../ui/layers';

/**
 * Optional construction overrides. Each entry is a function that
 * takes the same arguments the inline `new X(...)` site used to pass
 * and returns the constructed instance.
 *
 * `Partial<AppFactories>` is what callers see on `LuxarAppOptions`;
 * the resolved-at-init record is `Required<AppFactories>` after
 * merging with `defaultFactories`.
 */
export interface AppFactories {
  sceneManager?: () => SceneManager;
  animationController?: (
    controls: SceneManager['controls'],
    postProcessing: SceneManager['postProcessing']
  ) => AnimationController;
  renderingControls?: (
    postProcessing: SceneManager['postProcessing'],
    sceneManager: SceneManager
  ) => RenderingControls;
  recordingPanel?: (
    sceneManager: SceneManager,
    animationController: AnimationController
  ) => RecordingPanel;
  layersPanel?: (
    parent: HTMLElement,
    animationController: AnimationController
  ) => LayersPanel;
}

/**
 * Default factories. Each entry is byte-for-byte equivalent to the
 * pre-21C inline `new X(...)` call.
 */
export const defaultFactories: Required<AppFactories> = {
  sceneManager: () => new SceneManager(),
  animationController: (controls, postProcessing) =>
    new AnimationController(controls, postProcessing),
  renderingControls: (postProcessing, sceneManager) =>
    new RenderingControls(postProcessing, sceneManager),
  recordingPanel: (sceneManager, animationController) =>
    new RecordingPanel(sceneManager, animationController),
  layersPanel: (parent, animationController) =>
    new LayersPanel(parent, animationController),
};

/**
 * Merge user-supplied factory overrides with the defaults. Returns
 * a fully-populated record where every entry is callable.
 */
export function resolveFactories(overrides?: AppFactories): Required<AppFactories> {
  return { ...defaultFactories, ...(overrides ?? {}) };
}
