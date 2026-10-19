/** The backend decision shared by standalone preload and renderer setup. */
export interface BackendSelection {
  backend: 'webgl' | 'webgpu';
  source: 'url-param' | 'env-var' | 'default';
  /** Only an explicit WebGPU override may use three's WebGL2 bridge. */
  requireNativeAdapter: boolean;
}

/** WebGL remains the default until the go/no-go checks in #3146 pass. */
const DEFAULT_BACKEND: BackendSelection['backend'] = 'webgl';

/** URL/Settings override, then build environment, then the default. */
export function selectBackend(
  rendererOverride: 'webgl' | 'webgpu' | undefined,
  defaultBackend: BackendSelection['backend'] = DEFAULT_BACKEND
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
  return { backend: defaultBackend, source: 'default', requireNativeAdapter: true };
}
