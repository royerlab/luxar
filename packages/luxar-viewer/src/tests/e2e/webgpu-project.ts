/**
 * The `chromium-webgpu` Playwright project: what it changes, in one place.
 *
 * The default `chromium` project drives the viewer through `THREE.WebGLRenderer`,
 * and the handful of specs that ask for `?renderer=webgpu` there land on
 * `WebGPURenderer`'s internal WebGL2 backend, because Playwright's bundled
 * headless Chromium has no hardware WebGPU adapter. So WGSL codegen, real
 * adapter limits and the 256-byte readback row padding were exercised by no
 * automated run at all.
 *
 * The `chromium-webgpu` project (opt-in, `pnpm test:e2e:webgpu`; see
 * `playwright.config.ts`) runs the same specs on system Chrome with a real
 * adapter. Three things make that a NATIVE run rather than a relabelled one:
 *
 * 1. {@link withWebGPURenderer}: the shared fixture appends `renderer=webgpu`
 *    to every viewer URL the spec did not already pin with `renderer=`.
 * 2. `waitForLuxarReady` (helpers.ts) fails fast, through
 *    {@link requiresNativeWebGPU}, when such a page did not come up on a
 *    native, non-fallback WebGPU backend. A silent WebGL2 fallback would turn
 *    the whole project into a second copy of the default one.
 * 3. {@link skipUnlessWebGL} keeps out the specs that drive raw WebGL
 *    (`WEBGL_lose_context`, a `webgl2` context's own error queue), which have
 *    nothing to observe on a canvas already in `webgpu` mode.
 *
 * This module deliberately imports nothing from `helpers.ts`, which imports it.
 *
 * @module tests/e2e/webgpu-project
 */

import { test, type TestInfo } from '@playwright/test';

/** Name of the opt-in native-WebGPU Playwright project. */
export const WEBGPU_PROJECT = 'chromium-webgpu';

/**
 * Whether the test runs in the native-WebGPU project.
 *
 * @param testInfo - The running test's info (defaults to `test.info()`)
 * @returns True only under the `chromium-webgpu` project
 */
export function isWebGPUProject(testInfo: TestInfo = test.info()): boolean {
  return testInfo.project.name === WEBGPU_PROJECT;
}

/**
 * Skip the current test in the native-WebGPU project because it drives WebGL
 * itself (a `webgl2` context, `WEBGL_lose_context`, the GL error queue).
 *
 * Call it at the top of a test body, or from a no-argument `test.beforeEach`
 * to skip a whole file or describe block. A no-op in every other project.
 *
 * @param reason - What the test needs from WebGL, for the skip annotation
 * @param testInfo - The running test's info (defaults to `test.info()`)
 */
export function skipUnlessWebGL(reason: string, testInfo: TestInfo = test.info()): void {
  testInfo.skip(isWebGPUProject(testInfo), `WebGL-only (${WEBGPU_PROJECT} project): ${reason}`);
}

const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * Whether a resolved URL is the viewer app page (as opposed to a harness page,
 * an embed example, or a data-server URL): a loopback origin serving `/` or
 * `/index.html`.
 */
function isViewerPage(url: URL): boolean {
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;
  if (!LOOPBACK_HOSTS.has(url.hostname)) return false;
  return url.pathname === '/' || url.pathname === '/index.html';
}

/**
 * Append `renderer=webgpu` to a viewer URL that does not already choose a
 * renderer. Anything else — a spec that pins `renderer=webgl` (or any
 * `renderer=` value), a harness page, an absolute non-loopback URL — is
 * returned untouched, byte for byte.
 *
 * Relative URLs are resolved against `baseURL` only to classify them; the
 * returned string stays relative so Playwright resolves it exactly as before.
 *
 * @param url - The URL handed to `page.goto`
 * @param baseURL - The project's `use.baseURL`
 * @returns The URL to navigate to in the native-WebGPU project
 */
export function withWebGPURenderer(url: string, baseURL: string | undefined): string {
  let resolved: URL;
  try {
    resolved = new URL(url, baseURL);
  } catch {
    return url;
  }
  if (!isViewerPage(resolved) || resolved.searchParams.has('renderer')) return url;

  // Insert before any fragment, and keep the spec's own query text verbatim
  // (re-serializing through URLSearchParams would re-encode flag-only params
  // such as `&debug` as `&debug=`).
  const hashAt = url.indexOf('#');
  const head = hashAt === -1 ? url : url.slice(0, hashAt);
  const hash = hashAt === -1 ? '' : url.slice(hashAt);
  const separator = !head.includes('?') ? '?' : head.endsWith('?') || head.endsWith('&') ? '' : '&';
  return `${head}${separator}renderer=webgpu${hash}`;
}

/**
 * Whether a page at `pageURL` must be running on a native WebGPU backend in
 * the native-WebGPU project: it is a viewer page asking for
 * `renderer=webgpu` without the `webgpuForceWebgl` diagnostic (which pins
 * three's WebGL2 backend on purpose).
 *
 * @param pageURL - `page.url()` after the viewer reported ready
 * @returns True when a non-native backend there is a failure, not a fallback
 */
export function requiresNativeWebGPU(pageURL: string): boolean {
  let url: URL;
  try {
    url = new URL(pageURL);
  } catch {
    return false;
  }
  return (
    isViewerPage(url) &&
    url.searchParams.get('renderer') === 'webgpu' &&
    !url.searchParams.has('webgpuForceWebgl')
  );
}
