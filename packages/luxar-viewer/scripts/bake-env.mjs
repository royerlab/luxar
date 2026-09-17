#!/usr/bin/env node
/**
 * Headless driver for `luxar env bake`: open the viewer on a scene with
 * `?bakeEnv&probe=…&envResolution=…`, wait for the capture the viewer runs once
 * the load settles, and write the container it hands back
 * (`__luxarDebug.environment.lastBake`, base64) to `--out`.
 *
 *   node scripts/bake-env.mjs \
 *     --url 'http://127.0.0.1:8123/viewer/?src=http://127.0.0.1:8123&debug&bakeEnv&probe=auto&envResolution=128' \
 *     --out /tmp/scene.env.bin [--timeout 300000] [--channel chrome|chromium]
 *
 * The Python side (`luxar.environment.bake`) serves the store and the viewer and invokes this;
 * it only needs the viewer's Playwright devDependency, hence a Node script. The
 * system Chrome channel is the default — it is the one headless browser with a real
 * GPU (and WebGPU) adapter, and a bake should be lit and sized like a real session.
 * `--channel chromium` falls back to the bundled build (SwiftShader) where no Chrome
 * is installed. Exits 1 on a bake error, a timeout, or an empty container.
 *
 * Bytes travel back through `page.evaluate` as base64 rather than a download event:
 * the download path depends on the browser's download handling, the evaluate path
 * does not, and the container is small (a 128 px cube is ~0.8 MB).
 */

import { writeFileSync } from 'node:fs';
import { chromium } from '@playwright/test';

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const next = process.argv[i + 1];
  return next === undefined || next.startsWith('--') ? true : next;
}

const url = arg('url');
const out = arg('out');
const timeoutMs = Number(arg('timeout', '300000'));
const channel = arg('channel', 'chrome');
if (typeof url !== 'string' || typeof out !== 'string') {
  console.error(
    'usage: node scripts/bake-env.mjs --url <viewer url> --out <file> [--timeout ms] [--channel chrome|chromium]'
  );
  process.exit(2);
}
if (!/[?&]bakeEnv(&|$)/.test(url)) {
  console.error('--url must carry ?bakeEnv (the viewer bakes only when asked to)');
  process.exit(2);
}

const launchOptions = {
  headless: true,
  args: [
    '--ignore-gpu-blocklist',
    '--no-sandbox',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
  ],
};
if (channel !== 'chromium') launchOptions.channel = channel;

let browser;
let failed = false;
try {
  browser = await chromium.launch(launchOptions);
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.type() === 'warning') console.log(`[browser] ${msg.text()}`);
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  const handle = await page.waitForFunction(
    () => {
      const env = window.__luxarDebug?.environment;
      if (!env) return null;
      if (env.bakeError) return { error: env.bakeError };
      if (env.lastBake) return { ok: true };
      return null;
    },
    undefined,
    { timeout: timeoutMs, polling: 250 }
  );
  const verdict = await handle.jsonValue();
  if (verdict?.error) throw new Error(`viewer reported: ${verdict.error}`);

  const bake = await page.evaluate(() => {
    const last = window.__luxarDebug.environment.lastBake;
    return { header: last.header, base64: last.base64, byteLength: last.byteLength };
  });
  const bytes = Buffer.from(bake.base64, 'base64');
  if (bytes.length !== bake.byteLength || bytes.length < 16) {
    throw new Error(`container came back with ${bytes.length} bytes (expected ${bake.byteLength})`);
  }
  writeFileSync(out, bytes);
  console.log(
    `baked ${bake.header.resolution}px cube at probe ${bake.header.probe.spec} ` +
      `(${bytes.length} bytes, ${bake.header.coordinate_system}) → ${out}`
  );
  if (errors.length) console.log(`page errors during the bake: ${errors.length}`);
} catch (error) {
  console.error(`bake failed: ${error instanceof Error ? error.message : String(error)}`);
  failed = true;
} finally {
  await browser?.close();
}
process.exit(failed ? 1 : 0);
