/**
 * Capture high-resolution screenshots of the Luxar viewer for paper figures.
 *
 * Usage:
 *   APP_URL="http://.../?src=...&debug" OUT=/path/to/out.png WAIT=15000 WIDTH=2400 HEIGHT=1600 \
 *     pnpm exec tsx tools/capture-hires.ts
 */

import { chromium } from '@playwright/test';

const URL = process.env.APP_URL || '';
const OUT = process.env.OUT || 'test-results/debug/hires.png';
const WAIT = parseInt(process.env.WAIT || '15000', 10);
const WIDTH = parseInt(process.env.WIDTH || '2400', 10);
const HEIGHT = parseInt(process.env.HEIGHT || '1600', 10);
const DSF = parseFloat(process.env.DSF || '1');

async function main() {
  if (!URL) {
    console.error('ERROR: APP_URL env var required');
    process.exit(1);
  }
  console.log(`[CAPTURE] ${WIDTH}x${HEIGHT} DSF=${DSF} wait=${WAIT}ms`);
  console.log(`[CAPTURE] URL: ${URL}`);
  console.log(`[CAPTURE] OUT: ${OUT}`);

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--use-gl=egl',
      '--ignore-gpu-blocklist',
      '--enable-webgl-developer-extensions',
      '--enable-webgl-draft-extensions',
      '--disable-web-security',
    ],
  });

  const context = await browser.newContext({
    viewport: { width: WIDTH, height: HEIGHT },
    deviceScaleFactor: DSF,
  });

  const page = await context.newPage();

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      console.error(`[BROWSER-ERROR] ${msg.text()}`);
    }
  });
  page.on('pageerror', (err) => console.error(`[PAGE-ERROR] ${err.message}`));

  await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
  console.log('[CAPTURE] Page loaded, waiting for scene init...');

  // Initial wait for scene auto-fit and first chunks
  await page.waitForTimeout(Math.min(WAIT, 15000));

  // Optional camera overrides (env vars):
  //   CAMERA_ZOOM=1.5    scale dist from target by factor (>1 out, <1 in)
  //   CAMERA_POS=x,y,z   absolute camera position (world units)
  //   CAMERA_TARGET=x,y,z  absolute orbit target (default: origin)
  //   CAMERA_FOV=degrees  vertical FOV override
  const camZoom = parseFloat(process.env.CAMERA_ZOOM || '1');
  const camPos = process.env.CAMERA_POS;
  const camTarget = process.env.CAMERA_TARGET;
  const camFov = process.env.CAMERA_FOV;
  if (camZoom !== 1 || camPos || camTarget || camFov) {
    console.log(
      `[CAPTURE] Applying camera overrides: zoom=${camZoom} pos=${camPos} target=${camTarget} fov=${camFov}`
    );
    await page.evaluate(
      ({ z, pos, target, fov }) => {
        const d = (window as any).__luxarDebug;
        if (!d || !d.camera) return;
        const cam = d.camera;
        if (target) {
          const [tx, ty, tz] = target.split(',').map(Number);
          if (d.controls && d.controls.target) d.controls.target.set(tx, ty, tz);
        }
        if (pos) {
          const [px, py, pz] = pos.split(',').map(Number);
          cam.position.set(px, py, pz);
        } else if (z !== 1 && d.controls) {
          const t = d.controls.target || { x: 0, y: 0, z: 0 };
          cam.position.set(
            t.x + (cam.position.x - t.x) * z,
            t.y + (cam.position.y - t.y) * z,
            t.z + (cam.position.z - t.z) * z
          );
        }
        if (fov) {
          cam.fov = parseFloat(fov);
        }
        if (d.controls && d.controls.target) {
          cam.lookAt(d.controls.target);
        }
        cam.updateProjectionMatrix();
        d.controls?.update?.();
        d.renderOnce?.();
      },
      { z: camZoom, pos: camPos, target: camTarget, fov: camFov }
    );
  }

  // Optional nD slice-position override (for navigating non-displayed dims
  // like time or channel). Format: SLICE="dim:value,dim:value" e.g.
  // SLICE="time:240,channel:0"
  const sliceSpec = process.env.SLICE;
  if (sliceSpec) {
    console.log(`[CAPTURE] Applying SLICE override: ${sliceSpec}`);
    await page.evaluate((spec) => {
      const pairs = spec.split(',').map((p) => {
        const [name, valueStr] = p.split(':');
        return { name: name.trim(), value: Number(valueStr.trim()) };
      });
      const d = (window as any).__luxarDebug;
      const dm = d?.sceneDimsManager;
      if (!dm) {
        console.warn('[SLICE] no sceneDimsManager on __luxarDebug');
        return;
      }
      const dims = dm.dims || dm.getDims?.();
      if (!dims || !dims.metadata) {
        console.warn('[SLICE] no dims metadata');
        return;
      }
      for (const { name, value } of pairs) {
        const idx = dims.metadata.findIndex((m: any) => m.name === name);
        if (idx < 0) {
          console.warn(
            `[SLICE] dimension '${name}' not found; known: ${dims.metadata
              .map((m: any) => m.name)
              .join(',')}`
          );
          continue;
        }
        dm.setDimensionValue(idx, value);
        console.log(`[SLICE] ${name} (idx=${idx}) -> ${value}`);
      }
      d.renderOnce?.();
    }, sliceSpec);
  }

  // Remaining wait budget — lets new frustum chunks stream in after zoom
  await page.waitForTimeout(Math.max(0, WAIT - 15000));

  // Inspect state to confirm data loaded
  const state = await page.evaluate(() => {
    const d = (window as any).__luxarDebug;
    if (!d) return { ok: false, reason: 'no debug object' };
    const perf = d.getState?.().performance || {};
    return {
      ok: perf.totalElements > 0,
      totalPoints: perf.totalPoints,
      totalGSplats: perf.totalGSplats,
      totalElements: perf.totalElements,
      pointCloudCount: perf.pointClouds?.length || 0,
      gsplatCount: perf.gsplatMeshes?.length || 0,
    };
  });
  console.log('[CAPTURE] State:', JSON.stringify(state));

  if (!state.ok) {
    console.error('[CAPTURE] Warning: no elements loaded');
  }

  // Take high-res screenshot
  await page.screenshot({ path: OUT, fullPage: false, timeout: 15000 });
  console.log(`[CAPTURE] ✓ Saved ${OUT}`);

  await browser.close();
}

main().catch((e) => {
  console.error('[CAPTURE] FATAL:', e);
  process.exit(1);
});
