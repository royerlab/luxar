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
const viewport = { width: 768, height: 768 };
const fixture = resolve(outDir, 'fixture.luxar.zarr');
const fixtureMetadataPath = resolve(outDir, 'fixture-metadata.json');
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

const servers = [];
function spawnServer(label, command, args, cwd) {
  const child = spawn(command, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
  const recentLines = [];
  const capture = (chunk) => {
    for (const line of chunk.toString().split(/\r?\n/)) {
      if (!line) continue;
      recentLines.push(line);
      if (recentLines.length > 20) recentLines.shift();
    }
  };
  child.stdout.on('data', capture);
  child.stderr.on('data', capture);
  const server = { label, child, recentLines };
  servers.push(server);
  return server;
}

function killAll() {
  for (const { child } of servers) {
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

async function waitHttp(url, server, timeoutMs = 60_000) {
  const start = Date.now();
  for (;;) {
    try {
      const response = await fetch(url);
      if (response.ok || response.status === 404) return;
    } catch {
      // Server is still starting.
    }
    if (Date.now() - start > timeoutMs) {
      const output = server.recentLines.length
        ? `\nRecent ${server.label} output:\n${server.recentLines.join('\n')}`
        : `\n${server.label} produced no output.`;
      throw new Error(`timed out waiting for ${url}${output}`);
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
}

async function readSelection(page, lodGroup, activeLevel) {
  return page.evaluate(
    ({ groupName, level }) => {
      const state = window.__luxarDebug.getState();
      let lodObject = null;
      window.__luxarDebug.scene.traverse((object) => {
        if (object.name === groupName) lodObject = object;
      });
      let visibleElementCount = 0;
      lodObject?.children?.[level]?.traverse((object) => {
        const count =
          object.userData?.visiblePointCount ??
          object.userData?.visibleSplatCount ??
          object.userData?.visibleSegmentCount ??
          object.userData?.visibleTriangleCount;
        if (typeof count === 'number') visibleElementCount += count;
      });
      return {
        selectedLevel: state.lodGroups.find((entry) => entry.name === groupName)?.activeLevel ?? -1,
        visibleElementCount,
      };
    },
    { groupName: lodGroup, level: activeLevel }
  );
}

async function captureLevel(browser, bench, fixtureBench, label, expectedLevel, query) {
  const page = await browser.newPage({ viewport });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const src = `http://127.0.0.1:${dataPort}/${fixtureRelative}`;
  await page.goto(
    `http://127.0.0.1:${viewerPort}/?src=${src}&debug&dpr=1&noLodFade&noLodEnergy&${query}`
  );
  const expectedElementCount = fixtureBench.levelElementCounts[expectedLevel];
  try {
    await page.waitForFunction(
      ({ lodGroup, activeLevel, elementCount }) => {
        const debug = window.__luxarDebug;
        const group = debug?.getState?.()?.lodGroups?.find((entry) => entry.name === lodGroup);
        if (group?.activeLevel !== activeLevel) return false;
        let lodObject = null;
        debug?.scene?.traverse((object) => {
          if (object.name === lodGroup) lodObject = object;
        });
        const selectedChild = lodObject?.children?.[activeLevel];
        if (!selectedChild?.visible) return false;
        let visibleElementCount = 0;
        selectedChild.traverse((object) => {
          const count =
            object.userData?.visiblePointCount ??
            object.userData?.visibleSplatCount ??
            object.userData?.visibleSegmentCount ??
            object.userData?.visibleTriangleCount;
          if (typeof count === 'number') visibleElementCount += count;
        });
        return visibleElementCount === elementCount;
      },
      { lodGroup: bench.lodGroup, activeLevel: expectedLevel, elementCount: expectedElementCount },
      { timeout: 120_000 }
    );
  } catch (error) {
    const observed = await readSelection(page, bench.lodGroup, expectedLevel);
    throw new Error(
      `${label} level did not settle: expected level ${expectedLevel} with ` +
        `${expectedElementCount} ${bench.geometry} elements; observed level ` +
        `${observed.selectedLevel} with ${observed.visibleElementCount}`,
      { cause: error }
    );
  }
  await page.evaluate(() => window.__luxarDebug.renderOnce());
  await page.waitForTimeout(500);
  const selection = await readSelection(page, bench.lodGroup, expectedLevel);
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
const viewerServer = spawnServer(
  'viewer server',
  'pnpm',
  ['exec', 'vite', '--port', String(viewerPort), '--strictPort', '--clearScreen', 'false'],
  viewerRoot
);
const dataServer = spawnServer(
  'data server',
  'python3',
  ['-m', 'http.server', String(dataPort), '--bind', '127.0.0.1'],
  repoRoot
);

const fixtureMetadata = JSON.parse(readFileSync(fixtureMetadataPath, 'utf8'));
if (fixtureMetadata.schemaVersion !== 1 || !Array.isArray(fixtureMetadata.benches)) {
  throw new Error('unsupported fixture-metadata.json schema');
}

const summary = { fixture: fixtureRelative, scoreSize, benches: [] };
let failed = false;
let browser = null;
try {
  await waitHttp(`http://127.0.0.1:${viewerPort}/`, viewerServer);
  await waitHttp(`http://127.0.0.1:${dataPort}/`, dataServer);
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
    const fixtureBench = fixtureMetadata.benches.find((entry) => entry.id === bench.id);
    if (!fixtureBench) throw new Error(`fixture metadata is missing bench ${bench.id}`);
    if (fixtureBench.geometry !== bench.geometry || fixtureBench.lodGroup !== bench.lodGroup) {
      throw new Error(`fixture metadata disagrees with threshold record for ${bench.id}`);
    }
    if (
      fixtureBench.levelElementCounts.length !== 2 ||
      bench.coarseLevel !== 0 ||
      bench.finestLevel !== 1
    ) {
      throw new Error(`${bench.id} must declare a two-level coarsest-vs-finest ladder`);
    }
    const benchDir = resolve(outDir, bench.id);
    mkdirSync(benchDir, { recursive: true });
    const finest = await captureLevel(
      browser,
      bench,
      fixtureBench,
      'finest',
      bench.finestLevel,
      'lodFinest'
    );
    const coarse = await captureLevel(
      browser,
      bench,
      fixtureBench,
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
      blownSentinelActive:
        finest.blownPixelFraction >= bench.thresholds.minReferenceBlownPixelFraction,
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
    if (!captureChecks.blownSentinelActive) {
      console.error('The finest arm has too few blown pixels for the clipping sentinel.');
    }
    if (!pass) failed = true;
  }
} finally {
  await browser?.close();
  killAll();
}

writeFileSync(resolve(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(`\nWrote ${relative(process.cwd(), outDir)}/summary.json and per-bench PNGs`);
process.exit(failed ? 1 : 0);
