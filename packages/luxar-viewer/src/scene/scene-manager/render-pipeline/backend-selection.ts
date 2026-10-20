import {
  candidateRules,
  detectBrowserPlatform,
  defaultPlatformRules,
  type BrowserPlatform,
  type WebGPUPlatformRule,
} from './backend-policy';

/** The backend decision shared by standalone preload and renderer setup. */
export interface BackendSelection {
  backend: 'webgl' | 'webgpu';
  source: 'url-param' | 'env-var' | 'platform-policy';
  /** Only an explicit WebGPU override may use three's WebGL2 bridge. */
  requireNativeAdapter: boolean;
  /** Candidate vendor rules to check after the adapter answers. */
  platformRules?: readonly WebGPUPlatformRule[];
}

/**
 * Select the backend in order: URL/Settings override, WebGPU env vars,
 * legacy WebGL env var, then the measured platform policy. Only an explicit
 * WebGPU override may use three's WebGL2 bridge.
 */
export function selectBackend(
  rendererOverride: 'webgl' | 'webgpu' | undefined,
  platform: BrowserPlatform = detectBrowserPlatform(),
  rules: readonly WebGPUPlatformRule[] = defaultPlatformRules()
): BackendSelection {
  if (rendererOverride === 'webgl') {
    return { backend: 'webgl', source: 'url-param', requireNativeAdapter: true };
  }
  if (rendererOverride === 'webgpu') {
    return { backend: 'webgpu', source: 'url-param', requireNativeAdapter: false };
  }
  if (
    import.meta.env.VITE_LUXAR_USE_WEBGPU === '1' ||
    import.meta.env.VITE_LUXAR_USE_WEBGPU_RENDERER === '1'
  ) {
    return { backend: 'webgpu', source: 'env-var', requireNativeAdapter: true };
  }
  if (import.meta.env.VITE_LUXAR_USE_LEGACY_WEBGL === '1') {
    return { backend: 'webgl', source: 'env-var', requireNativeAdapter: true };
  }
  const platformRules = candidateRules(platform, rules);
  if (platformRules.length === 0) {
    return { backend: 'webgl', source: 'platform-policy', requireNativeAdapter: true };
  }
  return {
    backend: 'webgpu',
    source: 'platform-policy',
    requireNativeAdapter: true,
    platformRules,
  };
}

/** Under the platform policy, wait for the adapter vendor before fetching TSL. */
export function shouldPrefetchTsl(
  hasGpu: boolean,
  requireNativeAdapter?: boolean,
  platformRules?: readonly WebGPUPlatformRule[]
): boolean {
  return !platformRules && (hasGpu || !requireNativeAdapter);
}
