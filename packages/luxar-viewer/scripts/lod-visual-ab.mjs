#!/usr/bin/env node
/** Opt-in rendered coarse-vs-finest LOD acceptance bench. */

import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import {
  captureCanvasImage,
  captureStableImage,
  evaluateCaptureChecks,
  evaluateThresholds,
  scoreImagePair,
} from './visual-ab-core.mjs';

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
  child.on('error', (error) => capture(error));
  const server = { label, child, recentLines };
  servers.push(server);
  return server;
}

function signalAll(signal) {
  for (const { child } of servers) {
    try {
      child.kill(signal);
    } catch {
      // Already exited.
    }
  }
}

async function stopAll() {
  signalAll('SIGTERM');
  await Promise.all(
    servers.map(async ({ child }) => {
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
  try {
    return await page.evaluate(
      ({ groupName, level }) => {
        const debug = window.__luxarDebug;
        const state = debug?.getState?.();
        let lodObject = null;
        debug?.scene?.traverse((object) => {
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
          selectedLevel:
            state?.lodGroups?.find((entry) => entry.name === groupName)?.activeLevel ?? -1,
          visibleElementCount,
        };
      },
      { groupName: lodGroup, level: activeLevel }
    );
  } catch {
    return { selectedLevel: -1, visibleElementCount: 0 };
  }
}

async function captureLevel(
  browser,
  bench,
  fixtureBench,
  allLodGroups,
  label,
  expectedLevel,
  query
) {
  const page = await browser.newPage({ viewport });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const src = `http://127.0.0.1:${dataPort}/${fixtureRelative}`;
  try {
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
        {
          lodGroup: bench.lodGroup,
          activeLevel: expectedLevel,
          elementCount: expectedElementCount,
        },
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
    const selection = await readSelection(page, bench.lodGroup, expectedLevel);
    await page.evaluate(
      ({ activeGroup, lodGroups }) => {
        window.__luxarDebug?.scene?.traverse((object) => {
          if (lodGroups.includes(object.name)) object.visible = object.name === activeGroup;
        });
      },
      { activeGroup: bench.lodGroup, lodGroups: allLodGroups }
    );
    const canvas = page.locator('canvas#app');
    const image = await captureStableImage(
      async () => {
        await page.evaluate(() => window.__luxarDebug.renderOnce());
        await page.waitForTimeout(300);
        return captureCanvasImage(page, canvas, scoreSize);
      },
      { minCaptures: 8 }
    );
    const visibleGroups = await page.evaluate((lodGroups) => {
      const visible = [];
      window.__luxarDebug?.scene?.traverse((object) => {
        if (lodGroups.includes(object.name) && object.visible) visible.push(object.name);
      });
      return visible;
    }, allLodGroups);
    if (visibleGroups.length !== 1 || visibleGroups[0] !== bench.lodGroup) {
      throw new Error(`${label} isolation changed during capture: ${visibleGroups.join(', ')}`);
    }
    return { label, ...selection, pageErrors, ...image };
  } finally {
    await page.close();
  }
}

function serialisableArm(arm) {
  const result = { ...arm };
  delete result.png;
  delete result.rgb;
  delete result.grey;
  return result;
}

mkdirSync(outDir, { recursive: true });
const summary = { fixture: fixtureRelative, scoreSize, benches: [], errors: [] };
let failed = false;
let browser = null;
try {
  const thresholdsDocument = JSON.parse(
    readFileSync(resolve(here, 'lod-visual-ab-thresholds.json'), 'utf8')
  );
  if (thresholdsDocument.schemaVersion !== 1 || !Array.isArray(thresholdsDocument.benches)) {
    throw new Error('unsupported lod-visual-ab-thresholds.json schema');
  }
  const generated = spawnSync(
    'hatch',
    ['run', 'python', resolve(here, 'generate-lod-visual-ab-fixture.py'), fixture],
    { cwd: repoRoot, stdio: 'inherit' }
  );
  if (generated.status !== 0) {
    throw new Error(`fixture generation failed with exit code ${generated.status ?? 1}`);
  }
  const fixtureMetadata = JSON.parse(readFileSync(fixtureMetadataPath, 'utf8'));
  if (fixtureMetadata.schemaVersion !== 1 || !Array.isArray(fixtureMetadata.benches)) {
    throw new Error('unsupported fixture-metadata.json schema');
  }
  const allLodGroups = fixtureMetadata.benches.map((bench) => bench.lodGroup);
  await assertPortFree(viewerPort, 'viewer');
  await assertPortFree(dataPort, 'data');
  const viewerServer = spawnServer(
    'viewer server',
    resolve(viewerRoot, 'node_modules/.bin/vite'),
    ['--port', String(viewerPort), '--strictPort', '--clearScreen', 'false'],
    viewerRoot
  );
  const dataServer = spawnServer(
    'data server',
    'python3',
    ['-m', 'http.server', String(dataPort), '--bind', '127.0.0.1'],
    repoRoot
  );
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
    try {
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
        allLodGroups,
        'finest',
        bench.finestLevel,
        'lodFinest'
      );
      const coarse = await captureLevel(
        browser,
        bench,
        fixtureBench,
        allLodGroups,
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
      const capture = evaluateCaptureChecks(finest, coarse, bench.thresholds);
      const pass = verdict.pass && capture.pass;
      const result = {
        id: bench.id,
        geometry: bench.geometry,
        blendingMode: bench.blendingMode,
        thresholds: bench.thresholds,
        arms: { finest: serialisableArm(finest), coarse: serialisableArm(coarse) },
        score,
        captureMetrics: capture.metrics,
        verdict,
        captureChecks: capture.checks,
        pass,
      };
      summary.benches.push(result);
      console.log(`\n=== ${bench.id} (${bench.geometry}/${bench.blendingMode}) ===`);
      console.log(
        `SSIM=${score.ssim.toFixed(4)}  NCC=${score.ncc.toFixed(4)}  ` +
          `meanDeltaE=${score.meanDeltaE.toFixed(3)}  ` +
          `meanLumaRatio=${capture.metrics.meanLumaRatio.toFixed(4)}  ` +
          `blownDelta=${(score.blownPixelFraction.delta * 100).toFixed(3)}%  ` +
          `${pass ? 'PASS' : 'FAIL'}`
      );
      if (!capture.checks.noPageErrors) {
        console.error(`Page errors: ${[...finest.pageErrors, ...coarse.pageErrors].join('; ')}`);
      }
      if (!capture.checks.rendered) console.error('An arm rendered (almost) nothing.');
      if (!capture.checks.blownSentinelActive) {
        console.error('The finest arm has too few blown pixels for the clipping sentinel.');
      }
      if (!capture.checks.meanLumaRatio) {
        console.error(
          `Mean-luma ratio ${capture.metrics.meanLumaRatio.toFixed(4)} is below the recorded floor.`
        );
      }
      if (!pass) failed = true;
    } catch (error) {
      failed = true;
      summary.benches.push({
        id: bench.id,
        geometry: bench.geometry,
        blendingMode: bench.blendingMode,
        pass: false,
        error: error instanceof Error ? error.message : String(error),
      });
      console.error(`\n${bench.id} failed:`, error);
    }
  }
} catch (error) {
  failed = true;
  summary.errors.push(error instanceof Error ? error.message : String(error));
  console.error(error);
} finally {
  try {
    await browser?.close();
    await stopAll();
    if (servers.length > 0) {
      await Promise.all([waitPortFree(viewerPort), waitPortFree(dataPort)]);
    }
  } catch (error) {
    failed = true;
    summary.errors.push(error instanceof Error ? error.message : String(error));
    console.error(error);
  }
  writeFileSync(resolve(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
}

const writtenOutputs = summary.benches.length ? 'summary.json and per-bench PNGs' : 'summary.json';
console.log(`\nWrote ${relative(process.cwd(), outDir)}/${writtenOutputs}`);
process.exitCode = failed ? 1 : 0;
