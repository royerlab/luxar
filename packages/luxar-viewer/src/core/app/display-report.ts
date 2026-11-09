/**
 * The display report: what this page actually renders with, sent once to the
 * server that opened it.
 *
 * An exported kiosk's `serve.py --chrome` opens the display with
 * `?displayReport=<nonce>`, and needs to tell its operator whether the wall got what
 * it was set up for -- WebGPU on the real GPU, full screen -- or something
 * worse that would otherwise go unnoticed: WebGL, the WebGPU renderer's
 * WebGL2 backend, a software adapter, a window smaller than the screen. Only
 * this page knows, so it says so: one same-origin POST to `/display-report`,
 * which `serve.py` turns into OK / WARNING lines in its terminal. It goes once
 * the first dataset load has finished or failed, so a scene that did not load
 * is reported as such, and it echoes the nonce so the server takes only its
 * own page's report. Best effort
 * and fire-and-forget: a server that does not listen answers 404 and nothing
 * else happens.
 *
 * @module core/app/display-report
 */

import type { RendererCapabilities } from '../../rendering/renderer-capabilities';

/** The path `serve.py` listens on (see its `LuxarHandler.do_POST`). */
export const DISPLAY_REPORT_PATH = '/display-report';

/** What the page renders with, as `serve.py` reads it. */
export interface DisplayReport {
  /** The API the renderer exposes: `'webgpu'` or `'webgl2'`; null if none came up. */
  api: RendererCapabilities['apiSurface'] | null;
  /**
   * What actually runs: `'webgpu-native'`, `'webgpu-gl'` (WebGL2 backend) or
   * `'webgl'`; null if no renderer came up.
   */
  backend: RendererCapabilities['backendKind'] | null;
  /** Why the viewer failed to start, when it did; absent otherwise. */
  error?: string;
  /** Why the first dataset failed to load, when it did; absent otherwise. */
  datasetError?: string;
  /** The `?displayReport=` value, echoed so the server knows the report is its page's. */
  nonce: string;
  /** The renderer the URL asked for (`?renderer=`), or null for the default policy. */
  requested: 'webgl' | 'webgpu' | null;
  /** Whether the browser exposes WebGPU at all (`navigator.gpu`). */
  webgpuAvailable: boolean;
  /** The high-performance WebGPU adapter, when one answers. */
  adapter: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    isFallbackAdapter: boolean;
  } | null;
  screen: { width: number; height: number };
  window: { width: number; height: number };
  devicePixelRatio: number;
  userAgent: string;
}

/** How long the adapter probe may take before the report goes without it. */
const ADAPTER_PROBE_MS = 3000;

/** What a report says beyond the renderer. */
export interface DisplayReportContext {
  /** The renderer the URL asked for (`?renderer=`), or null. */
  requested: 'webgl' | 'webgpu' | null;
  /** The `?displayReport=` value to echo. */
  nonce: string;
  /** Why the viewer failed to start, if it did. */
  error?: string;
  /** Why the first dataset failed to load, if it did. */
  datasetError?: string;
}

/** The GPU adapter fields we read; typed locally (DOM lib versions differ). */
interface AdapterLike {
  info?: {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
    isFallbackAdapter?: boolean;
  };
  isFallbackAdapter?: boolean;
}

interface GpuLike {
  requestAdapter(options?: { powerPreference?: string }): Promise<AdapterLike | null>;
}

/**
 * A separate high-performance adapter probe -- not necessarily the adapter
 * the renderer runs on (a WebGL renderer has none) -- bounded in time, since
 * a broken GPU process can leave `requestAdapter` pending forever.
 */
async function adapterSummary(gpu: GpuLike | undefined): Promise<DisplayReport['adapter']> {
  if (!gpu) return null;
  try {
    const adapter = await Promise.race([
      gpu.requestAdapter({ powerPreference: 'high-performance' }),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), ADAPTER_PROBE_MS)),
    ]);
    if (!adapter) return null;
    const info = adapter.info ?? {};
    return {
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
      isFallbackAdapter: Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter),
    };
  } catch {
    return null;
  }
}

/**
 * Assemble the report for this page: from its renderer, or -- when the viewer
 * failed to start, possibly before it had one -- from the ``error``.
 */
export async function buildDisplayReport(
  capabilities: RendererCapabilities | undefined,
  context: DisplayReportContext
): Promise<DisplayReport> {
  const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu;
  const { requested, nonce, error, datasetError } = context;
  return {
    api: capabilities?.apiSurface ?? null,
    backend: capabilities?.backendKind ?? null,
    ...(error !== undefined ? { error } : {}),
    ...(datasetError !== undefined ? { datasetError } : {}),
    nonce,
    requested,
    webgpuAvailable: gpu !== undefined,
    adapter: await adapterSummary(gpu),
    screen: { width: window.screen.width, height: window.screen.height },
    window: { width: window.innerWidth, height: window.innerHeight },
    devicePixelRatio: window.devicePixelRatio,
    userAgent: navigator.userAgent,
  };
}

/**
 * Build and send the report. Never throws and never blocks the viewer:
 * a failure is the server's to notice (it warns when no report arrives).
 */
export async function sendDisplayReport(
  capabilities: RendererCapabilities | undefined,
  context: DisplayReportContext,
  post: (url: string, body: string) => Promise<unknown> = (url, body) =>
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: true,
    })
): Promise<void> {
  try {
    const report = await buildDisplayReport(capabilities, context);
    await post(new URL(DISPLAY_REPORT_PATH, window.location.origin).href, JSON.stringify(report));
  } catch {
    // Best effort by design: see the module doc.
  }
}
