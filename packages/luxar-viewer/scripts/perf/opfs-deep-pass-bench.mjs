/**
 * Reproduce deep additive-ladder playback against a warm OPFS cache.
 *
 * The browser profile is persistent so every concurrency arm reads the same
 * origin-scoped L2 cache. Run a warm-up pass first, then rerun the sweep and
 * require zero L2 misses before comparing completed update timings.
 */
import { writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const arg = (name, fallback) => {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
};

const viewerDir = path.resolve(arg('viewer-dir', '.'));
const baseUrl = arg('url', 'http://127.0.0.1:5198/?src=http://127.0.0.1:9011');
const profileDir = path.resolve(arg('profile-dir', '/tmp/luxar-opfs-deep-pass'));
const out = path.resolve(arg('out', 'opfs-deep-pass.json'));
const ladderDepth = Number(arg('ladder-depth', '6'));
const targetFps = Number(arg('target-fps', '2'));
const startFrame = Number(arg('start-frame', '49'));
const headless = arg('headless', 'false') === 'true';
const clearFirst = arg('clear-first', 'false') === 'true';
const concurrencyValues = arg('concurrency', '8,64,512,4096').split(',').map(Number);

const require = createRequire(path.join(viewerDir, 'package.json'));
const { chromium } = require('@playwright/test');
const gpuArgs =
  process.platform === 'darwin'
    ? ['--use-angle=metal']
    : process.platform === 'linux'
      ? ['--ozone-platform=x11', '--use-angle=vulkan', '--enable-features=Vulkan']
      : [];

function collectStageDurations(node, name, durations = []) {
  if (!node) return durations;
  if (node.name === name) durations.push(node.lastMs);
  for (const child of node.children ?? []) {
    collectStageDurations(child, name, durations);
  }
  return durations;
}

const context = await chromium.launchPersistentContext(profileDir, {
  channel: 'chrome',
  headless,
  viewport: { width: 1280, height: 720 },
  args: [
    '--disable-web-security',
    '--ignore-gpu-blocklist',
    '--no-sandbox',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    ...gpuArgs,
  ],
});

async function openViewer(concurrency) {
  const page = await context.newPage();
  page.on('pageerror', (error) => console.error('[pageerror]', error.message));
  const separator = baseUrl.includes('?') ? '&' : '?';
  await page.goto(
    `${baseUrl}${separator}debug&dpr=1&no-prefetch&opfsReadConcurrency=${concurrency}`,
    { waitUntil: 'domcontentloaded' }
  );
  await page.waitForFunction(() => window.__luxarDebug?.getState?.().initialized, undefined, {
    timeout: 120_000,
  });
  await page.waitForFunction(
    () => {
      const debug = window.__luxarDebug;
      if (!debug?.scene) return false;
      let loaded = false;
      debug.scene.traverse((object) => {
        if (object.userData?.nodeType === 'gsplats' && (object.geometry?.instanceCount ?? 0) > 0) {
          loaded = true;
        }
      });
      return loaded;
    },
    undefined,
    { timeout: 180_000 }
  );
  return page;
}

async function waitForIdle(page, quietMs = 500, timeoutMs = 120_000) {
  await page.evaluate(
    async ({ quietMs, timeoutMs }) => {
      const debug = window.__luxarDebug;
      const loader = debug.getSceneLoader().getDefaultLoader();
      const snapshot = () => {
        const stats = loader.getCacheStats();
        return JSON.stringify([
          debug.getSceneLoader().getProfiler().getTimings().count,
          stats.l2?.activeReads ?? 0,
          stats.l2?.queuedReads ?? 0,
          stats.l2WriteQueue?.queued ?? 0,
          stats.l2WriteQueue?.inFlight ?? 0,
        ]);
      };
      const started = performance.now();
      let last = snapshot();
      let lastChange = performance.now();
      while (performance.now() - started < timeoutMs) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        const current = snapshot();
        if (current !== last) {
          last = current;
          lastChange = performance.now();
        } else if (performance.now() - lastChange >= quietMs) {
          return;
        }
      }
      throw new Error(`viewer did not become idle within ${timeoutMs}ms`);
    },
    { quietMs, timeoutMs }
  );
}

async function runPass(page, concurrency) {
  await waitForIdle(page);
  return page.evaluate(
    async ({ ladderDepth, targetFps, startFrame, concurrency }) => {
      const debug = window.__luxarDebug;
      const sceneLoader = debug.getSceneLoader();
      const loader = sceneLoader.getDefaultLoader();
      const profiler = sceneLoader.getProfiler();
      const manager = debug.app.inputHandler.getAnimationManager();
      const dimensions = debug.app.getDimensions();
      const metadata = dimensions.metadata ?? [];
      let timeIndex = metadata.findIndex((dimension) => /^t(ime)?$/i.test(dimension?.name ?? ''));
      if (timeIndex < 0) {
        timeIndex = metadata.findIndex((dimension) => dimension?.display === false);
      }
      if (timeIndex < 0) throw new Error('no non-displayed time dimension found');

      debug.sceneDimsManager.setDimensionValue(timeIndex, startFrame);
      await debug.sceneDimsManager.waitForUpdate();

      const before = structuredClone(loader.getCacheStats());
      const initialCount = profiler.getTimings().count;
      const updates = [];
      let maxActiveReads = 0;
      let maxQueuedReads = 0;
      let maxWriteQueue = 0;
      let lastCount = initialCount;
      const started = performance.now();

      manager.play(timeIndex, { targetFPS: targetFps, loopMode: 'once', ladderDepth });
      while (manager.isAnimating(timeIndex)) {
        const timings = profiler.getTimings();
        const stats = loader.getCacheStats();
        maxActiveReads = Math.max(maxActiveReads, stats.l2?.activeReads ?? 0);
        maxQueuedReads = Math.max(maxQueuedReads, stats.l2?.queuedReads ?? 0);
        maxWriteQueue = Math.max(maxWriteQueue, stats.l2WriteQueue?.queued ?? 0);
        if (timings.count !== lastCount) {
          updates.push(structuredClone(timings));
          lastCount = timings.count;
        }
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      let quietSince = performance.now();
      while (performance.now() - quietSince < 1500) {
        const timings = profiler.getTimings();
        const stats = loader.getCacheStats();
        maxActiveReads = Math.max(maxActiveReads, stats.l2?.activeReads ?? 0);
        maxQueuedReads = Math.max(maxQueuedReads, stats.l2?.queuedReads ?? 0);
        maxWriteQueue = Math.max(maxWriteQueue, stats.l2WriteQueue?.queued ?? 0);
        if (timings.count !== lastCount) {
          updates.push(structuredClone(timings));
          lastCount = timings.count;
          quietSince = performance.now();
        }
        if (
          (stats.l2?.activeReads ?? 0) > 0 ||
          (stats.l2?.queuedReads ?? 0) > 0 ||
          (stats.l2WriteQueue?.queued ?? 0) > 0 ||
          (stats.l2WriteQueue?.inFlight ?? 0) > 0
        ) {
          quietSince = performance.now();
        }
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      const after = structuredClone(loader.getCacheStats());
      return {
        concurrency,
        ladderDepth,
        timeIndex,
        wallMs: performance.now() - started,
        maxActiveReads,
        maxQueuedReads,
        maxWriteQueue,
        before,
        after,
        updates,
        renderer: (() => {
          const canvas = document.createElement('canvas');
          const gl = canvas.getContext('webgl2');
          const extension = gl?.getExtension('WEBGL_debug_renderer_info');
          return extension ? gl?.getParameter(extension.UNMASKED_RENDERER_WEBGL) : null;
        })(),
        userAgent: navigator.userAgent,
      };
    },
    { ladderDepth, targetFps, startFrame, concurrency }
  );
}

const results = [];
for (let index = 0; index < concurrencyValues.length; index += 1) {
  const concurrency = concurrencyValues[index];
  console.error(`[opfs=${concurrency}] opening viewer`);
  const page = await openViewer(concurrency);
  if (index === 0 && clearFirst) {
    await page.evaluate(async () => {
      const loader = window.__luxarDebug.getSceneLoader().getDefaultLoader();
      loader.clearL1Cache();
      await loader.clearL2Cache();
    });
  }
  const result = await runPass(page, concurrency);
  result.loadArraysMs = result.updates.map((timings) =>
    collectStageDurations(timings, 'Load Arrays')
  );
  results.push(result);
  console.error(
    `[opfs=${concurrency}] wall=${result.wallMs.toFixed(0)}ms ` +
      `updates=${result.updates.length} active=${result.maxActiveReads} queued=${result.maxQueuedReads} ` +
      `misses=${result.after.l2.misses - result.before.l2.misses}`
  );
  await page.close();
}

writeFileSync(out, JSON.stringify({ baseUrl, results }, null, 2));
await context.close();
console.error(`wrote ${out}`);
