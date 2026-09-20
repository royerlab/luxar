#!/usr/bin/env node
/**
 * Real-WebGPU vs WebGL A/B of one scene, judged by STRUCTURAL similarity — the
 * acceptance test `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.6 asks for, as a script rather
 * than a manual recipe.
 *
 * Why this exists: three's `MeshPhysicalMaterial` (WebGL) and `MeshPhysicalNodeMaterial`
 * (WebGPU) are three's own and are close but not pixel-identical, and the house codegen
 * snapshot harness pins only TSL Luxar writes. So the two backends are compared as
 * IMAGES: the same scene, the same baked camera, `dpr=1`, screenshot each backend,
 * then SSIM + normalised cross-correlation on a 256² greyscale downscale, plus the
 * lit-pixel counts that catch a black frame outright. Pixel equality is deliberately
 * not the criterion.
 *
 * Both arms run in the SYSTEM Chrome (`channel: 'chrome'`): Playwright's bundled
 * Chromium has no WebGPU adapter headless, and the WebGL arm runs in the same browser
 * so it is a genuine control rather than a different renderer stack. The script
 * verifies the WebGPU arm actually got a WebGPU backend (not the WebGL2 fallback) and
 * refuses to report a "parity" that compared WebGL with itself.
 *
 * Usage (from packages/luxar-viewer/, with the dataset already built):
 *
 *   node scripts/ab-webgpu-vs-webgl.mjs \
 *     --dataset ../../datasets/examples/mesh_physical_materials_example.luxar.zarr \
 *     --out ../../delme/ab-physical [--bloom] [--min-ssim 0.9] [--channel chrome]
 *
 * Spawns a Vite dev server and a plain HTTP data server (CORS sidestepped with
 * `--disable-web-security`, exactly as `scripts/perf/timelapse-nav-bench.mjs` does),
 * writes `webgl.png`, `webgpu.png` and `summary.json` to `--out`, prints the metrics,
 * and exits 1 when SSIM is below `--min-ssim` or either arm rendered nothing.
 *
 * `--bloom` additionally captures each arm with bloom enabled at the default
 * threshold, so a bright specular highlight's behaviour under the bloom pass is
 * MEASURED rather than assumed (spec §3.5): the summary reports the lit fraction and
 * the fraction of near-white pixels with and without bloom.
 */

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { captureCanvasImage, ncc, ssim } from './visual-ab-core.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const viewerRoot = resolve(here, '..');
const repoRoot = resolve(viewerRoot, '../..');

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return fallback;
  const next = process.argv[i + 1];
  return next === undefined || next.startsWith('--') ? true : next;
}

const datasetArg = arg(
  'dataset',
  '../../datasets/examples/mesh_physical_materials_example.luxar.zarr'
);
const outDir = resolve(viewerRoot, arg('out', '../../delme/ab-webgpu-vs-webgl'));
const withBloom = arg('bloom', false) === true;
const minSsim = Number(arg('min-ssim', 0.9));
const channel = arg('channel', 'chrome');
const viewerPort = Number(arg('viewer-port', 5199));
const dataPort = Number(arg('data-port', 9007));
const viewport = { width: 1280, height: 720 };

const datasetAbs = resolve(viewerRoot, datasetArg);
const datasetRel = relative(repoRoot, datasetAbs).split('\\').join('/');
if (datasetRel.startsWith('..')) {
  console.error(`--dataset must live under the repository root (${repoRoot}); got ${datasetAbs}`);
  process.exit(2);
}

mkdirSync(outDir, { recursive: true });

const children = [];
function spawnServer(cmd, args, cwd) {
  const child = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.on('data', () => {});
  child.stderr.on('data', () => {});
  children.push(child);
  return child;
}
function signalAll(signal) {
  for (const c of children) {
    try {
      c.kill(signal);
    } catch {
      /* already gone */
    }
  }
}

async function stopAll() {
  signalAll('SIGTERM');
  await Promise.all(
    children.map(async (child) => {
      if (await waitForExit(child, 3000)) return;
      try {
        child.kill('SIGKILL');
      } catch {
        return;
      }
      await waitForExit(child, 3000);
    })
  );
}

function waitForExit(child, timeoutMs) {
  if (child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolveExit) => {
    const timeout = setTimeout(() => resolveExit(false), timeoutMs);
    child.once('exit', () => {
      clearTimeout(timeout);
      resolveExit(true);
    });
  });
}

async function waitPortFree(port, timeoutMs = 5000) {
  const start = Date.now();
  while (Date.now() - start <= timeoutMs) {
    try {
      await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(250) });
    } catch {
      return;
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }
  throw new Error(`port ${port} remained in use after server teardown`);
}

process.on('exit', () => signalAll('SIGTERM'));
process.on('SIGINT', async () => {
  await stopAll();
  process.exit(130);
});

async function waitHttp(url, timeoutMs = 60000) {
  const t0 = Date.now();
  for (;;) {
    try {
      const res = await fetch(url);
      if (res.ok || res.status === 404) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() - t0 > timeoutMs) throw new Error(`timed out waiting for ${url}`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

// ---------------------------------------------------------------------------
// One arm
// ---------------------------------------------------------------------------

async function captureArm(browser, renderer, bloom) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const url =
    `http://localhost:${viewerPort}/?src=http://127.0.0.1:${dataPort}/${datasetRel}&debug&dpr=1` +
    (renderer === 'webgpu' ? '&renderer=webgpu' : '');
  await page.goto(url);
  await page.waitForFunction(() => !!window.__luxarDebug?.app?.sceneManager?.scene, undefined, {
    timeout: 60000,
  });
  // Every mesh committed: the physical spheres are whole-node loads, so "every mesh
  // node has committed vertices" is the settled state. Read off the scene graph rather
  // than `getState()`, which is installed later than `app` and is absent on some
  // debug-surface versions.
  await page.waitForFunction(
    () => {
      const scene = window.__luxarDebug.app.sceneManager.scene;
      let meshes = 0;
      let committed = 0;
      scene.traverse((o) => {
        if (o.userData?.nodeType === 'mesh') {
          meshes++;
          if ((o.userData.committedVertexCount ?? 0) > 0) committed++;
        }
      });
      return meshes > 0 && committed === meshes;
    },
    undefined,
    { timeout: 120000 }
  );
  // Let the environment build, the programs compile and a few frames land.
  await page.waitForTimeout(1500);

  if (bloom) {
    await page.evaluate(() =>
      window.__luxarDebug.app.sceneManager.postProcessing.setBloomEnabled(true)
    );
    await page.waitForTimeout(500);
  }
  await page.evaluate(() => window.__luxarDebug.renderOnce());
  await page.waitForTimeout(300);

  const backend = await page.evaluate(() => {
    const sm = window.__luxarDebug.app.sceneManager;
    const r = sm.renderer;
    let totalTriangles = 0;
    let physicalMeshes = 0;
    sm.scene.traverse((o) => {
      if (o.userData?.nodeType === 'mesh') {
        totalTriangles += o.userData.visibleTriangleCount ?? 0;
        if (o.material?.userData?.material === 'physical') physicalMeshes++;
      }
    });
    return {
      apiSurface: sm.capabilities?.apiSurface ?? null,
      isWebGPURenderer: r?.isWebGPURenderer === true,
      backendIsWebGPU: r?.backend?.isWebGPUBackend === true,
      environmentReady: sm.environment?.isReady?.() ?? null,
      sceneHasEnvironment: !!sm.scene?.environment,
      totalTriangles,
      physicalMeshes,
    };
  });

  const stats = await captureCanvasImage(page, page.locator('canvas#app'));
  await page.close();
  return { renderer, bloom, backend, errors, ...stats };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const summary = { dataset: datasetRel, channel, arms: {}, pairs: {} };
let failed = false;

/**
 * Refuse a port something else already answers on. `--strictPort` makes Vite exit
 * rather than drift, but its exit is silent here (stderr is swallowed), and
 * `waitHttp` would then happily accept the FOREIGN server — measuring some other
 * checkout's viewer and reporting it as this one. That exact mistake produced a
 * confident wrong answer once; hence the check.
 */
async function assertPortFree(port, label) {
  try {
    await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(1000) });
  } catch {
    return; // nothing answered: free
  }
  console.error(
    `${label} port ${port} is already in use — another server would be measured. ` +
      'Pass a free --viewer-port / --data-port.'
  );
  process.exit(2);
}
await assertPortFree(viewerPort, 'viewer');
await assertPortFree(dataPort, 'data');

spawnServer(
  resolve(viewerRoot, 'node_modules/.bin/vite'),
  ['--port', String(viewerPort), '--strictPort', '--clearScreen', 'false'],
  viewerRoot
);
spawnServer('python3', ['-m', 'http.server', String(dataPort), '--bind', '127.0.0.1'], repoRoot);

let browser = null;
try {
  await waitHttp(`http://localhost:${viewerPort}/`);
  await waitHttp(`http://127.0.0.1:${dataPort}/`);

  browser = await chromium.launch({
    headless: true,
    channel,
    args: [
      '--disable-web-security',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
    ],
  });

  const variants = withBloom ? [false, true] : [false];
  for (const bloom of variants) {
    const suffix = bloom ? '-bloom' : '';
    const webgl = await captureArm(browser, 'webgl', bloom);
    const webgpu = await captureArm(browser, 'webgpu', bloom);
    writeFileSync(resolve(outDir, `webgl${suffix}.png`), webgl.png);
    writeFileSync(resolve(outDir, `webgpu${suffix}.png`), webgpu.png);

    for (const arm of [webgl, webgpu]) {
      const rest = { ...arm };
      delete rest.png;
      delete rest.grey;
      delete rest.rgb;
      summary.arms[`${arm.renderer}${suffix}`] = rest;
    }
    const pair = {
      ssim: ssim(webgl.grey, webgpu.grey, 256),
      ncc: ncc(webgl.grey, webgpu.grey),
      litFractionDelta: webgpu.litFraction - webgl.litFraction,
      webgpuIsReal: webgpu.backend.isWebGPURenderer && webgpu.backend.backendIsWebGPU,
    };
    summary.pairs[`webgl-vs-webgpu${suffix}`] = pair;

    console.log(`\n=== ${bloom ? 'bloom ON' : 'bloom OFF'} ===`);
    for (const arm of [webgl, webgpu]) {
      console.log(
        `${arm.renderer.padEnd(6)} lit=${(arm.litFraction * 100).toFixed(2)}%  ` +
          `nearWhite=${(arm.nearWhiteFraction * 100).toFixed(3)}%  meanLuma=${arm.meanLuma.toFixed(2)}  ` +
          `env=${arm.backend.sceneHasEnvironment}  tris=${arm.backend.totalTriangles}  ` +
          `apiSurface=${arm.backend.apiSurface} realWebGPU=${arm.backend.isWebGPURenderer && arm.backend.backendIsWebGPU}` +
          (arm.errors.length ? `  pageErrors=${arm.errors.length}` : '')
      );
    }
    console.log(`SSIM=${pair.ssim.toFixed(4)}  NCC=${pair.ncc.toFixed(4)}`);

    if (!pair.webgpuIsReal) {
      console.error(
        'WebGPU arm did not get a real WebGPU backend — this A/B compared WebGL with itself.'
      );
      failed = true;
    }
    if (webgl.litFraction < 0.01 || webgpu.litFraction < 0.01) {
      console.error('An arm rendered (almost) nothing.');
      failed = true;
    }
    if (pair.ssim < minSsim) {
      console.error(`SSIM ${pair.ssim.toFixed(4)} below --min-ssim ${minSsim}.`);
      failed = true;
    }
  }
} finally {
  await browser?.close();
  await stopAll();
  if (children.length > 0) {
    await Promise.all([waitPortFree(viewerPort), waitPortFree(dataPort)]);
  }
}

writeFileSync(resolve(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(`\nWrote ${relative(process.cwd(), outDir)}/{webgl,webgpu}*.png and summary.json`);
process.exit(failed ? 1 : 0);
