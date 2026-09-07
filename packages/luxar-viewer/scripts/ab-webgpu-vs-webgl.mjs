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
function killAll() {
  for (const c of children) {
    try {
      c.kill('SIGTERM');
    } catch {
      /* already gone */
    }
  }
}
process.on('exit', killAll);
process.on('SIGINT', () => {
  killAll();
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
// Image metrics (Node side, on the 256² greyscale the page hands back)
// ---------------------------------------------------------------------------

/** Global SSIM over 8×8 box windows on two equal-length greyscale arrays (0..255). */
function ssim(a, b, size) {
  const K1 = 0.01;
  const K2 = 0.03;
  const L = 255;
  const C1 = (K1 * L) ** 2;
  const C2 = (K2 * L) ** 2;
  const win = 8;
  let sum = 0;
  let n = 0;
  for (let y = 0; y + win <= size; y += win) {
    for (let x = 0; x + win <= size; x += win) {
      let ma = 0;
      let mb = 0;
      for (let j = 0; j < win; j++) {
        for (let i = 0; i < win; i++) {
          const k = (y + j) * size + x + i;
          ma += a[k];
          mb += b[k];
        }
      }
      const count = win * win;
      ma /= count;
      mb /= count;
      let va = 0;
      let vb = 0;
      let cov = 0;
      for (let j = 0; j < win; j++) {
        for (let i = 0; i < win; i++) {
          const k = (y + j) * size + x + i;
          const da = a[k] - ma;
          const db = b[k] - mb;
          va += da * da;
          vb += db * db;
          cov += da * db;
        }
      }
      va /= count - 1;
      vb /= count - 1;
      cov /= count - 1;
      sum += ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
      n++;
    }
  }
  return sum / n;
}

/** Normalised cross-correlation (Pearson) of two greyscale arrays. */
function ncc(a, b) {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da2 = 0;
  let db2 = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    num += da * db;
    da2 += da * da;
    db2 += db * db;
  }
  return da2 === 0 || db2 === 0 ? 0 : num / Math.sqrt(da2 * db2);
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
  await page.evaluate(() => window.__luxarDebug.app.sceneManager.requestRender?.());
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

  const png = await page.locator('canvas#app').screenshot();
  // Decode IN-PAGE (an Image + 2D canvas): drawing the live WebGL/WebGPU canvas
  // directly reads back black, per the repo's own recipe.
  const stats = await page.evaluate(
    async ({ b64, size }) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      const full = document.createElement('canvas');
      full.width = img.width;
      full.height = img.height;
      const fctx = full.getContext('2d');
      fctx.drawImage(img, 0, 0);
      const { data } = fctx.getImageData(0, 0, img.width, img.height);
      let lit = 0;
      let nearWhite = 0;
      let sum = 0;
      const n = img.width * img.height;
      for (let i = 0; i < data.length; i += 4) {
        const m = Math.max(data[i], data[i + 1], data[i + 2]);
        if (m > 16) lit++;
        if (m > 235) nearWhite++;
        sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
      }
      const small = document.createElement('canvas');
      small.width = size;
      small.height = size;
      const sctx = small.getContext('2d');
      sctx.drawImage(img, 0, 0, size, size);
      const sd = sctx.getImageData(0, 0, size, size).data;
      const grey = new Array(size * size);
      for (let p = 0; p < size * size; p++) {
        grey[p] = 0.299 * sd[p * 4] + 0.587 * sd[p * 4 + 1] + 0.114 * sd[p * 4 + 2];
      }
      return {
        width: img.width,
        height: img.height,
        litFraction: lit / n,
        nearWhiteFraction: nearWhite / n,
        meanLuma: sum / n,
        grey,
      };
    },
    { b64: png.toString('base64'), size: 256 }
  );
  await page.close();
  return { renderer, bloom, backend, errors, png, ...stats };
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
  'pnpm',
  ['exec', 'vite', '--port', String(viewerPort), '--strictPort', '--clearScreen', 'false'],
  viewerRoot
);
spawnServer('python3', ['-m', 'http.server', String(dataPort), '--bind', '127.0.0.1'], repoRoot);

try {
  await waitHttp(`http://localhost:${viewerPort}/`);
  await waitHttp(`http://127.0.0.1:${dataPort}/`);

  const browser = await chromium.launch({
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
  await browser.close();
} finally {
  killAll();
}

writeFileSync(resolve(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(`\nWrote ${relative(process.cwd(), outDir)}/{webgl,webgpu}*.png and summary.json`);
process.exit(failed ? 1 : 0);
