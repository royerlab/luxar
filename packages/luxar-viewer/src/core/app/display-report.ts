/**
 * The display report: what this page actually renders with, sent once to the
 * server that opened it.
 *
 * An exported kiosk's `serve.py --chrome` opens the display with
 * `?displayReport`, and needs to tell its operator whether the wall got what
 * it was set up for -- WebGPU on the real GPU, full screen -- or something
 * worse that would otherwise go unnoticed: WebGL, the WebGPU renderer's
 * WebGL2 backend, a software adapter, a window smaller than the screen. Only
 * this page knows, so it says so: one same-origin POST to `/display-report`,
 * which `serve.py` turns into OK / WARNING lines in its terminal. Best effort
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
  /** The API the renderer exposes: `'webgpu'` or `'webgl2'`. */
  api: RendererCapabilities['apiSurface'];
  /** What actually runs: `'webgpu-native'`, `'webgpu-gl'` (WebGL2 backend) or `'webgl'`. */
  backend: RendererCapabilities['backendKind'];
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

async function adapterSummary(gpu: GpuLike | undefined): Promise<DisplayReport['adapter']> {
  if (!gpu) return null;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
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

/** Assemble the report for this page. */
export async function buildDisplayReport(
  capabilities: RendererCapabilities,
  requested: 'webgl' | 'webgpu' | null
): Promise<DisplayReport> {
  const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu;
  return {
    api: capabilities.apiSurface,
    backend: capabilities.backendKind,
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
  capabilities: RendererCapabilities,
  requested: 'webgl' | 'webgpu' | null,
  post: (url: string, body: string) => Promise<unknown> = (url, body) =>
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: true,
    })
): Promise<void> {
  try {
    const report = await buildDisplayReport(capabilities, requested);
    await post(new URL(DISPLAY_REPORT_PATH, window.location.origin).href, JSON.stringify(report));
  } catch {
    // Best effort by design: see the module doc.
  }
}
