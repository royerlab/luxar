/**
 * Optional factory overrides for the heavy components constructed
 * by `LuxarApp.init()`. Each factory is keyed by the field name on
 * `LuxarApp` and returns a fresh instance with the same arguments
 * the inline `new X(...)` site uses. Default factories just call
 * the constructor.
 *
 * The seam exists for:
 *   - Embedders that need to substitute alternate scene managers,
 *     custom recording panels, etc. without subclassing LuxarApp.
 *   - Tests that prefer injecting pre-built stubs over `vi.mock(...)`
 *     for the heaviest components.
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
  layersPanel?: (parent: HTMLElement, animationController: AnimationController) => LayersPanel;
}

/**
 * Default factories — each entry simply calls `new X(...)` with the
 * same arguments the inline construction would use.
 */
export const defaultFactories: Required<AppFactories> = {
  sceneManager: () => new SceneManager(),
  animationController: (controls, postProcessing) =>
    new AnimationController(controls, postProcessing),
  renderingControls: (postProcessing, sceneManager) =>
    new RenderingControls(postProcessing, sceneManager),
  recordingPanel: (sceneManager, animationController) =>
    new RecordingPanel(sceneManager, animationController),
  layersPanel: (parent, animationController) => new LayersPanel(parent, animationController),
};

/**
 * Merge user-supplied factory overrides with the defaults. Returns
 * a fully-populated record where every entry is callable.
 */
export function resolveFactories(overrides?: AppFactories): Required<AppFactories> {
  return { ...defaultFactories, ...(overrides ?? {}) };
}
