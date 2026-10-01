/**
 * Browser plumbing shared by every render-gate suite: the Chrome flags, a
 * browser session that survives a crashed browser, a case's URL, and opening
 * a case in a fresh context with the renderer checks every measurement needs.
 *
 * @module scripts/render-gate/browser
 */

import { chromium } from '@playwright/test';

import * as ops from './page-ops.mjs';

const SOFTWARE = /swiftshader|llvmpipe|software|basic render/i;

/** Keep a requested backend list within the current suite or case. */
export function selectBackends(requested, allowed) {
  return (requested ?? allowed).filter((backend) => allowed.includes(backend));
}

/**
 * A case's URL: the suite defaults, then the case's own `urlParams` appended.
 * `liveLod: true` drops the defaults' `lodFinest`, which every other exact case
 * wants (deterministic finest content) and which would otherwise pin a LOD
 * case to its finest level, so the gate never saw LOD selection change.
 */
export function caseUrl(origin, c, backend, dsf, urlParams) {
  const params = [];
  if (c.store) params.push(`src=${origin}/datasets/${c.store}`);
  params.push('debug', `dpr=${dsf}`, `renderer=${backend === 'webgpu-gl' ? 'webgpu' : backend}`);
  if (backend === 'webgpu-gl') params.push('webgpuForceWebgl');
  const defaults = c.liveLod
    ? urlParams
        ?.split('&')
        .filter((p) => p !== 'lodFinest')
        .join('&')
    : urlParams;
  if (defaults) params.push(defaults);
  if (c.urlParams) params.push(c.urlParams);
  return `${origin}/?${params.join('&')}`;
}

/**
 * Browser helpers bound to the run's options.
 *
 * @param {{ channel: string, headless: boolean, chromeArgs: string[],
 *   log: (m: string) => void }} cfg
 */
export function browserKit({ channel, headless, chromeArgs, log }) {
  /** Every adapter string a page reported, for the report header. */
  const seenGpus = new Set();

  function browserArgs(perf, extraArgs = []) {
    const a = [
      '--ignore-gpu-blocklist',
      '--enable-unsafe-webgpu',
      '--enable-webgpu-developer-features',
    ];
    if (process.platform === 'linux') {
      // ANGLE on Vulkan puts WebGL on the discrete GPU (the default is
      // SwiftShader headless). Do NOT add the `Vulkan` feature: it moves
      // Chrome's own compositor onto Skia-Vulkan, which has no window surface
      // in headless mode ("Failed to initialize vulkan surface"), and the first
      // composited WebGL frame then loses the context (restored ~1 s later).
      // Measured on obsidian (system Chrome 146, RTX 3070): with
      // `--enable-features=Vulkan,WebGPU` every WebGL page lost its context;
      // with `WebGPU` alone none did, and both backends stayed on the NVIDIA
      // card. Not a viewer bug: a bare canvas clearing each frame reproduces it.
      a.push(
        '--use-angle=vulkan',
        '--enable-features=WebGPU',
        '--no-sandbox',
        '--disable-dev-shm-usage'
      );
    }
    if (perf) a.push('--disable-gpu-vsync', '--disable-frame-rate-limit');
    return [...a, ...extraArgs, ...chromeArgs];
  }

  /**
   * The suite's browser, relaunched when it has died.
   *
   * A browser that crashes mid-run (seen on obsidian under systemd-oomd memory
   * pressure) used to take every later case down with it: each reported
   * "browser has been closed" and the run's verdict became FAIL with nothing
   * wrong in the build. `run(fn)` hands `fn` a live browser and, when `fn` throws
   * BECAUSE the browser disconnected, relaunches and retries it once. A case that
   * kills the browser twice is reported as the error it is.
   */
  function browserSession(perf, extraArgs = []) {
    let browser = null;
    let relaunches = 0;
    const launch = () => chromium.launch({ channel, headless, args: browserArgs(perf, extraArgs) });
    const live = async () => {
      if (!browser?.isConnected()) {
        if (browser) {
          relaunches++;
          log('browser disconnected; relaunching');
        }
        browser = await launch();
      }
      return browser;
    };
    return {
      async run(fn) {
        try {
          return await fn(await live());
        } catch (e) {
          if (browser?.isConnected()) throw e;
          return fn(await live());
        }
      },
      relaunches: () => relaunches,
      async close() {
        if (browser?.isConnected()) await browser.close();
      },
    };
  }

  /**
   * Open a case in a fresh context: navigate, wait for the debug surface,
   * refuse a software or wrong-API renderer, inject a synthetic scene, and
   * switch off time-varying post effects.
   *
   * `contextOptions` extend the context (e.g. `ignoreHTTPSErrors`),
   * `initScripts` run before any page script, and `beforeGoto` is called
   * right before navigation (a hosted suite resets its server log there).
   */
  async function openCase(
    browser,
    origin,
    c,
    { backend, dsf, viewport, urlParams, contextOptions = {}, initScripts = [], beforeGoto }
  ) {
    const context = await browser.newContext({
      viewport,
      deviceScaleFactor: dsf,
      ...contextOptions,
    });
    for (const script of initScripts) await context.addInitScript(script);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    try {
      beforeGoto?.();
      await page.goto(caseUrl(origin, c, backend, dsf, urlParams), { timeout: 300000 });
      await page.waitForFunction(ops.debugReady, undefined, { timeout: 300000 });
      const info = await page.evaluate(ops.rendererInfo);
      seenGpus.add(`${info.api}: ${info.gpu}`);
      if (!info.usable)
        throw new Error(`${info.why} (${info.gpu || 'no adapter string'}); refusing to measure`);
      if (SOFTWARE.test(info.gpu))
        throw new Error(`software renderer (${info.gpu}); refusing to measure`);
      if (info.api !== backend)
        throw new Error(`asked for ${backend}, got ${info.api} (${info.gpu})`);
      if (c.synthetic) {
        await page.waitForFunction(ops.closeDatasetBrowser, undefined, { timeout: 10000 });
        await page.evaluate(ops.injectSynthetic, c.synthetic);
      }
      await page.evaluate(ops.disableEffects);
      return { context, page, info, errors };
    } catch (e) {
      await context.close();
      throw e;
    }
  }

  return { seenGpus, browserSession, openCase };
}
