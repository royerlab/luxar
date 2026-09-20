#!/usr/bin/env node
/** Opt-in rendered coarse-vs-finest LOD acceptance bench. */

import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { captureCanvasImage, evaluateThresholds, scoreImagePair } from './visual-ab-core.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const viewerRoot = resolve(here, '..');
const repoRoot = resolve(viewerRoot, '../..');
const scoreSize = 256;

if (process.platform !== 'linux') {
  console.error('LOD visual A/B thresholds are recorded for Linux Chromium only.');
  process.exit(2);
}

function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return fallback;
  const next = process.argv[index + 1];
  return next === undefined || next.startsWith('--') ? true : next;
}

const outDir = resolve(viewerRoot, arg('out', 'test-results/lod-visual-ab'));
const viewerPort = Number(arg('viewer-port', 5198));
const dataPort = Number(arg('data-port', 9006));
const channel = arg('channel', null);
const viewport = { width: 960, height: 720 };
const fixture = resolve(outDir, 'fixture.luxar.zarr');
const fixtureRelative = relative(repoRoot, fixture).split('\\').join('/');
if (fixtureRelative.startsWith('..')) {
  console.error(`--out must live under the repository root (${repoRoot}); got ${outDir}`);
  process.exit(2);
}

const thresholdsDocument = JSON.parse(
  readFileSync(resolve(here, 'lod-visual-ab-thresholds.json'), 'utf8')
);
if (thresholdsDocument.schemaVersion !== 1 || !Array.isArray(thresholdsDocument.benches)) {
  throw new Error('unsupported lod-visual-ab-thresholds.json schema');
}

mkdirSync(outDir, { recursive: true });
const generated = spawnSync(
  'hatch',
  ['run', 'python', resolve(here, 'generate-lod-visual-ab-fixture.py'), fixture],
  { cwd: repoRoot, stdio: 'inherit' }
);
if (generated.status !== 0) process.exit(generated.status ?? 1);

const children = [];
function spawnServer(command, args, cwd) {
  const child = spawn(command, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.on('data', () => {});
  child.stderr.on('data', () => {});
  children.push(child);
}

function killAll() {
  for (const child of children) {
    try {
      child.kill('SIGTERM');
    } catch {
      // Already exited.
    }
  }
}
process.on('exit', killAll);
process.on('SIGINT', () => {
  killAll();
  process.exit(130);
});

async function assertPortFree(port, label) {
  try {
    await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(1000) });
  } catch {
    return;
  }
  throw new Error(`${label} port ${port} is already in use; pass a free port override`);
}

async function waitHttp(url, timeoutMs = 60_000) {
  const start = Date.now();
  for (;;) {
    try {
      const response = await fetch(url);
      if (response.ok || response.status === 404) return;
    } catch {
      // Server is still starting.
    }
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${url}`);
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
}

async function captureLevel(browser, bench, label, expectedLevel, query) {
  const page = await browser.newPage({ viewport });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const src = `http://127.0.0.1:${dataPort}/${fixtureRelative}`;
  await page.goto(
    `http://127.0.0.1:${viewerPort}/?src=${src}&debug&dpr=1&noLodFade&noLodEnergy&${query}`
  );
  await page.waitForFunction(
    ({ lodGroup, activeLevel, expectedPointCount }) => {
      const debug = window.__luxarDebug;
      const group = debug?.getState?.()?.lodGroups?.find((entry) => entry.name === lodGroup);
      if (group?.activeLevel !== activeLevel) return false;
      let lodObject = null;
      debug?.scene?.traverse((object) => {
        if (object.name === lodGroup) lodObject = object;
      });
      const selectedChild = lodObject?.children?.[activeLevel];
      if (!selectedChild?.visible) return false;
      let visiblePoints = 0;
      selectedChild.traverse((object) => {
        if (object.userData?.nodeType === 'points') {
          visiblePoints += object.userData.visiblePointCount ?? 0;
        }
      });
      return visiblePoints === expectedPointCount;
    },
    {
      lodGroup: bench.lodGroup,
      activeLevel: expectedLevel,
      expectedPointCount: bench.levelPointCounts[expectedLevel],
    },
    { timeout: 120_000 }
  );
  await page.evaluate(() => window.__luxarDebug.app.sceneManager.requestRender?.());
  await page.waitForTimeout(500);
  const selection = await page.evaluate(
    ({ lodGroup, activeLevel }) => {
      const state = window.__luxarDebug.getState();
      let lodObject = null;
      window.__luxarDebug.scene.traverse((object) => {
        if (object.name === lodGroup) lodObject = object;
      });
      let visiblePointCount = 0;
      lodObject?.children?.[activeLevel]?.traverse((object) => {
        if (object.userData?.nodeType === 'points') {
          visiblePointCount += object.userData.visiblePointCount ?? 0;
        }
      });
      return {
        selectedLevel: state.lodGroups.find((entry) => entry.name === lodGroup)?.activeLevel ?? -1,
        visiblePointCount,
      };
    },
    { lodGroup: bench.lodGroup, activeLevel: expectedLevel }
  );
  const image = await captureCanvasImage(page, page.locator('canvas#app'), scoreSize);
  await page.close();
  return { label, ...selection, pageErrors, ...image };
}

function serialisableArm(arm) {
  const result = { ...arm };
  delete result.png;
  delete result.rgb;
  delete result.grey;
  return result;
}

await assertPortFree(viewerPort, 'viewer');
await assertPortFree(dataPort, 'data');
spawnServer(
  'pnpm',
  ['exec', 'vite', '--port', String(viewerPort), '--strictPort', '--clearScreen', 'false'],
  viewerRoot
);
spawnServer('python3', ['-m', 'http.server', String(dataPort), '--bind', '127.0.0.1'], repoRoot);

const summary = { fixture: fixtureRelative, scoreSize, benches: [] };
let failed = false;
let browser = null;
try {
  await waitHttp(`http://127.0.0.1:${viewerPort}/`);
  await waitHttp(`http://127.0.0.1:${dataPort}/`);
  const launchOptions = {
    headless: true,
    args: [
      '--disable-web-security',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
    ],
  };
  if (channel) launchOptions.channel = channel;
  browser = await chromium.launch(launchOptions);
  for (const bench of thresholdsDocument.benches) {
    const benchDir = resolve(outDir, bench.id);
    mkdirSync(benchDir, { recursive: true });
    const finest = await captureLevel(browser, bench, 'finest', bench.finestLevel, 'lodFinest');
    const coarse = await captureLevel(
      browser,
      bench,
      'coarse',
      bench.coarseLevel,
      'lodBias=0.000001'
    );
    writeFileSync(resolve(benchDir, 'finest.png'), finest.png);
    writeFileSync(resolve(benchDir, 'coarse.png'), coarse.png);
    const score = scoreImagePair(finest.rgb, coarse.rgb, scoreSize, {
      reference: finest.blownPixelFraction,
      candidate: coarse.blownPixelFraction,
    });
    const verdict = evaluateThresholds(score, bench.thresholds);
    const captureChecks = {
      noPageErrors: finest.pageErrors.length === 0 && coarse.pageErrors.length === 0,
      rendered: finest.litFraction >= 0.01 && coarse.litFraction >= 0.01,
    };
    const pass = verdict.pass && Object.values(captureChecks).every(Boolean);
    const result = {
      id: bench.id,
      geometry: bench.geometry,
      blendingMode: bench.blendingMode,
      thresholds: bench.thresholds,
      arms: { finest: serialisableArm(finest), coarse: serialisableArm(coarse) },
      score,
      verdict,
      captureChecks,
      pass,
    };
    summary.benches.push(result);
    console.log(`\n=== ${bench.id} (${bench.geometry}/${bench.blendingMode}) ===`);
    console.log(
      `SSIM=${score.ssim.toFixed(4)}  NCC=${score.ncc.toFixed(4)}  ` +
        `meanDeltaE=${score.meanDeltaE.toFixed(3)}  ` +
        `blownDelta=${(score.blownPixelFraction.delta * 100).toFixed(3)}%  ` +
        `${pass ? 'PASS' : 'FAIL'}`
    );
    if (!captureChecks.noPageErrors) {
      console.error(`Page errors: ${[...finest.pageErrors, ...coarse.pageErrors].join('; ')}`);
    }
    if (!captureChecks.rendered) console.error('An arm rendered (almost) nothing.');
    if (!pass) failed = true;
  }
} finally {
  await browser?.close();
  killAll();
}

writeFileSync(resolve(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(`\nWrote ${relative(process.cwd(), outDir)}/summary.json and per-bench PNGs`);
process.exit(failed ? 1 : 0);
