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
  if (forceWebGL || shouldPrefetchTsl(hasGpu, selection.requireNativeAdapter)) {
    if (hasGpu) preRequestWebGPUDevice(rendererOverride);
    // The normal load in SceneManager still installs the registry and handles
    // failures. The browser shares this in-flight module fetch with that load.
    void import('../rendering/tsl/registry').catch(() => undefined);
  }
}
