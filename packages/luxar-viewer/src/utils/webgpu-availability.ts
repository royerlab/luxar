/**
 * Page-load-time graphics-API availability probe.
 *
 * Returns which rendering backend the browser supports *before* a
 * renderer is constructed. Used for diagnostics ("you'll get the
 * fast path" badges), feature flags, and runtime backend messaging.
 *
 * For the API the renderer *actually used*, read
 * `RendererCapabilities.apiSurface` instead — this helper answers a
 * different question ("what's available?") than that one ("what
 * did we pick?").
 *
 * See `BROWSER_SUPPORT_POLICY.md` for the policy decision around
 * WebGPU vs WebGL2 vs unsupported.
 *
 * @module utils/webgpu-availability
 */

/** What the browser can run. */
export type RendererAPI = 'webgpu' | 'webgl2' | 'unsupported';

/**
 * Page-load-time probe. Returns:
 *   - `'webgpu'` if `navigator.gpu` is present and an adapter can be
 *     requested (cached after first call — no extra adapter requests
 *     on repeated probes).
 *   - `'webgl2'` if a WebGL2 context can be created on a throwaway
 *     canvas.
 *   - `'unsupported'` if neither path is available.
 *
 * Async because `navigator.gpu.requestAdapter()` is async. Callers
 * that need a synchronous answer can use `hasWebGPUSync()` (which
 * may return `'webgl2'` even if WebGPU is actually available — it
 * cannot probe the adapter without going async).
 */
export async function getRendererAPI(): Promise<RendererAPI> {
  if (cachedAPI !== null) return cachedAPI;
  cachedAPI = await detectRendererAPI();
  return cachedAPI;
}

/**
 * Synchronous fast-path. Returns `'webgpu'` if `navigator.gpu` is
 * present at all (no adapter probe — that requires async), otherwise
 * falls back to `'webgl2'` / `'unsupported'`. Cheaper than the async
 * variant; use when an answer-in-this-tick is required and a false
 * positive on WebGPU is acceptable (a follow-up adapter request will
 * confirm).
 */
export function getRendererAPISync(): RendererAPI {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) return 'webgpu';
  return hasWebGL2() ? 'webgl2' : 'unsupported';
}

/**
 * Reset the cached probe result. Test-only escape hatch — production
 * code should never need this.
 */
export function _resetCachedAPI(): void {
  cachedAPI = null;
}

// ---------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------

let cachedAPI: RendererAPI | null = null;

async function detectRendererAPI(): Promise<RendererAPI> {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      // `requestAdapter` resolves to `null` when no adapter is
      // available (e.g. driver flag off, hardware unsupported).
      //
      // MED-33: request the discrete / high-performance adapter so
      // this probe predicts the same adapter the 3D renderer will
      // ultimately select. Browsers may otherwise return the
      // low-power (integrated) adapter, which could falsely report
      // WebGPU as unavailable on systems where only the discrete
      // adapter is exposed.
      const adapter = await (
        navigator as Navigator & {
          gpu: {
            requestAdapter: (options?: {
              powerPreference?: 'low-power' | 'high-performance';
            }) => Promise<unknown>;
          };
        }
      ).gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) return 'webgpu';
    } catch {
      // Older Safari / Firefox-on-some-platforms throws rather than
      // resolving to null. Fall through to WebGL2.
    }
  }
  return hasWebGL2() ? 'webgl2' : 'unsupported';
}

function hasWebGL2(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    return canvas.getContext('webgl2') !== null;
  } catch {
    return false;
  }
}
