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
import { pathToFileURL } from 'node:url';

export const arg = (name, fallback, argv = process.argv) => {
  const index = argv.indexOf(`--${name}`);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value === undefined || value.startsWith('--') ? fallback : value;
};

export const flag = (name, argv = process.argv) => argv.includes(`--${name}`);

function parsePositiveNumber(name, value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`--${name} must be a positive number, got ${value}`);
  }
  return parsed;
}

function parseOptionalNumber(name, value) {
  if (value === undefined) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`--${name} must be a finite number, got ${value}`);
  }
  return parsed;
}

export function parseConcurrencyValues(value) {
  const values = value.split(',').map((part) => Number(part));
  if (values.length === 0 || values.some((item) => !Number.isInteger(item) || item <= 0)) {
    throw new Error(`--concurrency must contain positive integers, got ${value}`);
  }
  return values;
}

export function collectStageDurations(node, name, durations = []) {
  if (!node || node.stale) return durations;
  if (node.name === name) durations.push(node.lastMs);
  for (const child of node.children ?? []) {
    collectStageDurations(child, name, durations);
  }
  return durations;
}

export function resolveStartCoordinate(range, step, requestedStart) {
  const start = requestedStart ?? range[1] - step;
  if (start >= range[1]) {
    throw new Error(`start ${start} leaves no forward transition within [${range.join(', ')}]`);
  }
  return start;
}

function buildViewerUrl(baseUrl, concurrency, prefetch) {
  const separator = baseUrl.includes('?') ? '&' : '?';
  const noPrefetch = prefetch ? '' : '&no-prefetch';
  return `${baseUrl}${separator}debug&dpr=1${noPrefetch}&opfsReadConcurrency=${concurrency}`;
}

async function openViewer(context, options, concurrency) {
  const page = await context.newPage();
  page.on('pageerror', (error) => console.error('[pageerror]', error.message));
  await page.goto(buildViewerUrl(options.baseUrl, concurrency, options.prefetch), {
    waitUntil: 'domcontentloaded',
  });
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
          stats.l2WriteQueue?.depth ?? 0,
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

async function preparePass(page, requestedStart) {
  const dimension = await page.evaluate(() => {
    const debug = window.__luxarDebug;
    const manager = debug.inputHandler.getAnimationManager();
    if (!manager) throw new Error('dimension animation manager is unavailable');
    const dimensions = debug.app.getDimensions();
    const metadata = dimensions.metadata ?? [];
    let timeIndex = metadata.findIndex((dimension) => /^t(ime)?$/i.test(dimension?.name ?? ''));
    if (timeIndex < 0) timeIndex = metadata.findIndex((dimension) => dimension?.display === false);
    if (timeIndex < 0) throw new Error('no non-displayed time dimension found');

    const dimension = metadata[timeIndex];
    const range = dimension?.range;
    if (!range) throw new Error('time dimension has no range metadata');
    const step = dimension.step && dimension.step > 0 ? dimension.step : 1;
    return { timeIndex, step, range };
  });
  const start = resolveStartCoordinate(dimension.range, dimension.step, requestedStart);
  const actualStart = await page.evaluate(
    async ({ timeIndex, start }) => {
      const debug = window.__luxarDebug;
      debug.sceneDimsManager.setDimensionValue(timeIndex, start);
      await debug.sceneDimsManager.waitForUpdate();
      return debug.app.getDimensions().currentStep[timeIndex];
    },
    { timeIndex: dimension.timeIndex, start }
  );
  resolveStartCoordinate(dimension.range, dimension.step, actualStart);
  await waitForIdle(page);
  return { ...dimension, start: actualStart };
}

async function measurePass(page, options, prepared, concurrency) {
  return page.evaluate(
    async ({ ladderDepth, targetFps, prepared, concurrency }) => {
      const debug = window.__luxarDebug;
      const sceneLoader = debug.getSceneLoader();
      const loader = sceneLoader.getDefaultLoader();
      const profiler = sceneLoader.getProfiler();
      const manager = debug.inputHandler.getAnimationManager();
      if (!manager) throw new Error('dimension animation manager is unavailable');
      const before = structuredClone(loader.getCacheStats());
      const updates = [];
      const refinements = [];
      let lastCount = profiler.getTimings().count;
      let lastRefinementCount = profiler.getRefinementTimings().count;
      let maxActiveReads = 0;
      let maxQueuedReads = 0;
      let maxWriteQueue = 0;

      const sample = () => {
        const timings = profiler.getTimings();
        const refinement = profiler.getRefinementTimings();
        const stats = loader.getCacheStats();
        maxActiveReads = Math.max(maxActiveReads, stats.l2?.activeReads ?? 0);
        maxQueuedReads = Math.max(maxQueuedReads, stats.l2?.queuedReads ?? 0);
        maxWriteQueue = Math.max(maxWriteQueue, stats.l2WriteQueue?.depth ?? 0);
        if (timings.count !== lastCount) {
          updates.push(structuredClone(timings));
          lastCount = timings.count;
        }
        if (refinement.count !== lastRefinementCount) {
          refinements.push(structuredClone(refinement));
          lastRefinementCount = refinement.count;
        }
        return stats;
      };

      const started = performance.now();
      const played = manager.play(prepared.timeIndex, {
        targetFPS: targetFps,
        loopMode: 'once',
        ladderDepth,
      });
      if (!played) throw new Error('dimension animation did not start');
      while (manager.isAnimating(prepared.timeIndex)) {
        sample();
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      sample();
      const animationMs = performance.now() - started;
      const transitionUpdateCount = updates.length;
      const transitionRefinementCount = refinements.length;

      let quietSince = performance.now();
      while (performance.now() - quietSince < 1500) {
        const previousUpdateCount = updates.length;
        const previousRefinementCount = refinements.length;
        const stats = sample();
        if (
          updates.length !== previousUpdateCount ||
          refinements.length !== previousRefinementCount ||
          (stats.l2?.activeReads ?? 0) > 0 ||
          (stats.l2?.queuedReads ?? 0) > 0 ||
          (stats.l2WriteQueue?.depth ?? 0) > 0 ||
          (stats.l2WriteQueue?.inFlight ?? 0) > 0
        ) {
          quietSince = performance.now();
        }
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      const wallMs = performance.now() - started;
      const finalPosition = debug.app.getDimensions().currentStep[prepared.timeIndex];
      if (finalPosition === prepared.start || transitionUpdateCount === 0) {
        throw new Error(
          `measured pass did not advance: start=${prepared.start}, end=${finalPosition}, updates=${transitionUpdateCount}`
        );
      }
      return {
        concurrency,
        ladderDepth,
        timeIndex: prepared.timeIndex,
        start: prepared.start,
        end: finalPosition,
        step: prepared.step,
        range: prepared.range,
        animationMs,
        settleMs: wallMs - animationMs,
        wallMs,
        transitionUpdateCount,
        transitionRefinementCount,
        maxActiveReads,
        maxQueuedReads,
        maxWriteQueue,
        before,
        after: structuredClone(loader.getCacheStats()),
        updates,
        refinements,
        renderer: (() => {
          const renderer = debug.renderer;
          const backend = renderer.backend;
          const context = renderer.getContext?.();
          const extension = context?.getExtension?.('WEBGL_debug_renderer_info');
          const adapter = backend?.device?.adapterInfo;
          return {
            type: renderer.constructor?.name ?? null,
            backend: backend?.isWebGPUBackend
              ? 'webgpu'
              : backend?.isWebGLBackend || renderer.isWebGLRenderer
                ? 'webgl'
                : null,
            gpu: extension
              ? context.getParameter(extension.UNMASKED_RENDERER_WEBGL)
              : adapter
                ? {
                    vendor: adapter.vendor,
                    architecture: adapter.architecture,
                    device: adapter.device,
                    description: adapter.description,
                  }
                : null,
          };
        })(),
        userAgent: navigator.userAgent,
      };
    },
    { ladderDepth: options.ladderDepth, targetFps: options.targetFps, prepared, concurrency }
  );
}

async function runPass(page, options, concurrency) {
  await waitForIdle(page);
  const prepared = await preparePass(page, options.startFrame);
  return measurePass(page, options, prepared, concurrency);
}

export function enrichResult(result) {
  const transitionUpdates = result.updates.slice(0, result.transitionUpdateCount);
  const settleUpdates = result.updates.slice(result.transitionUpdateCount);
  result.loadArraysMs = {
    transition: transitionUpdates.map((timings) => collectStageDurations(timings, 'Load Arrays')),
    settle: settleUpdates.map((timings) => collectStageDurations(timings, 'Load Arrays')),
  };
  return result;
}

function writeResults(options, results, error = null) {
  writeFileSync(
    options.out,
    JSON.stringify(
      {
        baseUrl: options.baseUrl,
        prefetch: options.prefetch,
        ...(error ? { error } : {}),
        results,
      },
      null,
      2
    )
  );
}

function parseOptions(argv) {
  const viewerDir = path.resolve(arg('viewer-dir', '.', argv));
  const startValue = arg('start-frame', undefined, argv);
  return {
    viewerDir,
    baseUrl: arg('url', 'http://127.0.0.1:5198/?src=http://127.0.0.1:9011', argv),
    profileDir: path.resolve(arg('profile-dir', '/tmp/luxar-opfs-deep-pass', argv)),
    out: path.resolve(arg('out', 'opfs-deep-pass.json', argv)),
    ladderDepth: parsePositiveNumber('ladder-depth', arg('ladder-depth', '6', argv)),
    targetFps: parsePositiveNumber('target-fps', arg('target-fps', '2', argv)),
    startFrame: parseOptionalNumber('start-frame', startValue),
    headless: arg('headless', 'false', argv) === 'true',
    clearFirst: arg('clear-first', 'false', argv) === 'true',
    prefetch: flag('prefetch', argv),
    concurrencyValues: parseConcurrencyValues(arg('concurrency', '8,64,512,4096', argv)),
  };
}

export async function main(argv = process.argv) {
  const options = parseOptions(argv);
  const require = createRequire(path.join(options.viewerDir, 'package.json'));
  const { chromium } = require('@playwright/test');
  const gpuArgs =
    process.platform === 'darwin'
      ? ['--use-angle=metal']
      : process.platform === 'linux'
        ? ['--ozone-platform=x11', '--use-angle=vulkan', '--enable-features=Vulkan']
        : [];
  const context = await chromium.launchPersistentContext(options.profileDir, {
    channel: 'chrome',
    headless: options.headless,
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
  const results = [];
  let failure = null;
  try {
    for (let index = 0; index < options.concurrencyValues.length; index += 1) {
      const concurrency = options.concurrencyValues[index];
      console.error(`[opfs=${concurrency}] opening viewer`);
      const page = await openViewer(context, options, concurrency);
      try {
        if (index === 0 && options.clearFirst) {
          await page.evaluate(async () => {
            await window.__luxarDebug.getSceneLoader().getDefaultLoader().clearAllCaches();
          });
        }
        const result = enrichResult(await runPass(page, options, concurrency));
        results.push(result);
        writeResults(options, results);
        const misses = (result.after.l2?.misses ?? 0) - (result.before.l2?.misses ?? 0);
        console.error(
          `[opfs=${concurrency}] animation=${result.animationMs.toFixed(0)}ms ` +
            `settle=${result.settleMs.toFixed(0)}ms updates=${result.transitionUpdateCount}+` +
            `${result.updates.length - result.transitionUpdateCount} active=${result.maxActiveReads} ` +
            `queued=${result.maxQueuedReads} misses=${misses}`
        );
        if (result.maxActiveReads > concurrency) {
          throw new Error(
            `observed ${result.maxActiveReads} active OPFS reads with concurrency ${concurrency}`
          );
        }
      } finally {
        await page.close();
      }
    }
  } catch (error) {
    failure = error instanceof Error ? (error.stack ?? error.message) : String(error);
    throw error;
  } finally {
    try {
      writeResults(options, results, failure);
    } finally {
      await context.close();
    }
  }
  console.error(`wrote ${options.out}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
