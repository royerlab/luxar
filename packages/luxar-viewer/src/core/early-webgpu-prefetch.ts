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
import { hasParam } from '../config/url-param-access';
import {
  hasWebGPU,
  preRequestWebGPUDevice,
} from '../scene/scene-manager/render-pipeline/early-device-request';

const rendererOverride = earlyRendererOverride(location.search);
const selection = selectBackend(rendererOverride);
if (selection.backend === 'webgpu') {
  const forceWebGL = hasParam(new URLSearchParams(location.search), 'webgpuForceWebgl');
  const hasGpu = !forceWebGL && hasWebGPU();
  // The normal load in SceneManager still installs the registry and handles
  // failures. The browser shares this in-flight module fetch with that load.
  const prefetchRegistry = (): void => {
    void import('../rendering/tsl/registry').catch(() => undefined);
  };
  // Start device negotiation now. Under the platform policy only the adapter
  // request starts here: the device request, the `three/webgpu` chunk and (via
  // the callback) the registry wait until the adapter vendor matches a rule.
  if (hasGpu) preRequestWebGPUDevice(rendererOverride, selection.platformRules, prefetchRegistry);
  // Explicit or env-selected WebGPU prefetches the registry at once. Their
  // `webgpuForceWebgl` path also needs it for three's WebGL2 bridge.
  if (
    (forceWebGL && !selection.platformRules) ||
    shouldPrefetchTsl(hasGpu, selection.requireNativeAdapter, selection.platformRules)
  ) {
    prefetchRegistry();
  }
}
