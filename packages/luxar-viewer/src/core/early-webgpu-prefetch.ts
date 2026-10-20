/**
 * Tiny standalone entry that starts the lazy WebGPU chunk while the main
 * viewer bundle is still downloading. Keep its static imports small: this
 * script runs before `main.ts` and must not pull the viewer into WebGL loads.
 */
import {
  selectBackend,
  shouldPrefetchTsl,
} from '../scene/scene-manager/render-pipeline/backend-selection';
import { earlyRendererOverride } from '../config/renderer-preference';

const selection = selectBackend(earlyRendererOverride(location.search));
if (selection.backend === 'webgpu') {
  const hasGpu = (navigator as Navigator & { gpu?: unknown }).gpu !== undefined;
  if (shouldPrefetchTsl(hasGpu, selection.requireNativeAdapter, selection.platformRules)) {
    // The normal load in SceneManager still installs the registry and handles
    // failures. The browser shares this in-flight module fetch with that load.
    void import('../rendering/tsl/registry').catch(() => undefined);
  }
}
