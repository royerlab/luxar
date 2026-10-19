/**
 * Tiny standalone entry that starts the lazy WebGPU chunk while the main
 * viewer bundle is still downloading. Keep its static imports small: this
 * script runs before `main.ts` and must not pull the viewer into WebGL loads.
 */
import { selectBackend } from '../scene/scene-manager/render-pipeline/backend-selection';
import { StorageKeys } from '../utils/storage-keys';

function storedRenderer(): 'webgl' | 'webgpu' | undefined {
  try {
    const settings: unknown = JSON.parse(localStorage.getItem(StorageKeys.settings) ?? 'null');
    if (typeof settings !== 'object' || settings === null) return undefined;
    const value = settings as { version?: unknown; advanced?: { renderer?: unknown } };
    if (value.version !== 1) return undefined;
    const renderer = value.advanced?.renderer;
    return renderer === 'webgl' || renderer === 'webgpu' ? renderer : undefined;
  } catch {
    return undefined;
  }
}

const urlRenderer = new URLSearchParams(location.search).get('renderer')?.trim().toLowerCase();
const override =
  urlRenderer === 'webgl' || urlRenderer === 'webgl2'
    ? 'webgl'
    : urlRenderer === 'webgpu'
      ? 'webgpu'
      : storedRenderer();
const selection = selectBackend(override);
if (selection.backend === 'webgpu') {
  const hasGpu = (navigator as Navigator & { gpu?: unknown }).gpu !== undefined;
  if (hasGpu || !selection.requireNativeAdapter) {
    // The normal load in SceneManager still installs the registry and handles
    // failures. The browser shares this in-flight module fetch with that load.
    void import('../rendering/tsl/registry').catch(() => undefined);
  }
}
