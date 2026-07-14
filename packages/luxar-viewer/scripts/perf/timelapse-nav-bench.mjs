/**
 * Timelapse-nav perf bench: scrubs the time dimension of a 4D gsplat
 * scene and records per-timepoint UpdateProfiler stage timings, pool
 * stats, and renderer.info.memory. Run once per checkout (before/after)
 * with a distinct vite port; compare with compare.mjs.
 *
 * node bench.mjs --viewer-dir <path> --port 5198 --label before \
 *   --out before.json [--renderer webgpu] [--scrubs 3]
 */
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const viewerDir = path.resolve(arg('viewer-dir', '.'));
const port = Number(arg('port', '5198'));
const label = arg('label', 'run');
const out = arg('out', `${label}.json`);
const renderer = arg('renderer', 'webgl');
const scrubs = Number(arg('scrubs', '3'));
const dataPort = Number(arg('data-port', '9009'));
const tpsCap = Number(arg('tps', '0')); // 0 = all timepoints
const dataset = arg(
  'dataset',
  `http://127.0.0.1:${dataPort}/datasets/demos/gsplats_4d_neuromast_2ch.luxar.zarr`
);

const require = createRequire(path.join(viewerDir, 'package.json'));
const { chromium } = require('@playwright/test');

async function waitHttp(url, timeoutMs = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok || r.status === 404) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`server at ${url} not up after ${timeoutMs}ms`);
}

const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], {
  cwd: viewerDir,
  stdio: 'ignore',
});
process.on('exit', () => vite.kill('SIGKILL'));

try {
  await waitHttp(`http://localhost:${port}/`);
  await waitHttp(`http://127.0.0.1:${dataPort}/`);

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--disable-web-security', // same as playwright.config.ts — CORS for the local data server
      '--use-gl=egl',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));

  await page.addInitScript(() => {
    const rec = { deltas: [], last: 0, on: true };
    window.__frameRec = rec;
    const tick = (ts) => {
      if (rec.on) {
        if (rec.last > 0) rec.deltas.push(ts - rec.last);
        rec.last = ts;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  const url =
    `http://localhost:${port}/?src=${dataset}&debug&dpr=1` +
    (renderer === 'webgpu' ? '&renderer=webgpu' : '');
  await page.goto(url);
  await page.waitForFunction(() => !!window.__luxarDebug?.app, undefined, { timeout: 60000 });
  // gsplats committed
  await page.waitForFunction(
    () => {
      const dbg = window.__luxarDebug;
      if (!dbg?.scene) return false;
      let ok = false;
      dbg.scene.traverse((o) => {
        if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) ok = true;
      });
      return ok;
    },
    undefined,
    { timeout: 120000 }
  );

  // settle: profiler seq + lod-load stats stable for `quietMs`
  const settle = async (quietMs = 400, timeoutMs = 6000) => {
    await page.evaluate(
      async ({ quietMs, timeoutMs }) => {
        const dbg = window.__luxarDebug;
        const snap = () =>
          JSON.stringify([
            dbg.getSceneLoader().getProfiler().getTimings().count,
            dbg.getSceneLoader().getProfiler().getRefinementTimings().count,
            dbg.getLodLoadStats?.() ?? null,
          ]);
        const t0 = performance.now();
        let last = snap();
        let lastChange = performance.now();
        while (performance.now() - t0 < timeoutMs) {
          await new Promise((r) => setTimeout(r, 100));
          const cur = snap();
          if (cur !== last) {
            last = cur;
            lastChange = performance.now();
          } else if (performance.now() - lastChange >= quietMs) {
            return;
          }
        }
      },
      { quietMs, timeoutMs }
    );
  };

  const loadT0 = Date.now();
  await settle(800, 30000); // initial load + ladder quiesce
  const initialLoad = await page.evaluate(() => {
    const dbg = window.__luxarDebug;
    const rec = window.__frameRec;
    const pool = dbg.getSceneLoader().getDefaultLoader()?.gpuBufferPool;
    const deltas = rec.deltas.slice();
    rec.deltas = [];
    rec.last = 0;
    return {
      poolStats: pool ? pool.getStats() : null,
      memory: dbg.renderer?.info?.memory
        ? JSON.parse(JSON.stringify(dbg.renderer.info.memory))
        : null,
      frameDeltas: deltas,
    };
  });

  const meta = await page.evaluate(() => {
    const dbg = window.__luxarDebug;
    const dims = dbg.app.getDimensions();
    const pool = dbg.getSceneLoader().getDefaultLoader()?.gpuBufferPool;
    return {
      dims,
      poolStats: pool ? pool.getStats() : null,
      memory: dbg.renderer?.info?.memory ?? null,
      userAgentData: navigator.userAgent,
    };
  });

  // EmbedderDimensions shape: { ndim, displayed[], currentStep[], metadata[], ranges[] }
  console.error('[dims]', JSON.stringify(meta.dims).slice(0, 400));
  const md = meta.dims?.metadata ?? [];
  let timeIdx = md.findIndex((d) => /^t(ime)?$/i.test(d?.name ?? ''));
  if (timeIdx < 0) timeIdx = (meta.dims?.displayed ?? []).findIndex((v, i) => v === false && i < md.length);
  if (timeIdx < 0) throw new Error('no time dimension found: ' + JSON.stringify(meta.dims));
  const r = meta.dims.ranges[timeIdx];
  const rMin = Array.isArray(r) ? r[0] : (r.min ?? r.start ?? 0);
  const rMax = Array.isArray(r) ? r[1] : (r.max ?? r.end ?? 0);
  const step = (Array.isArray(r) ? r[2] : r.step) || md[timeIdx]?.step || 1;
  const dim = { range: [rMin, rMax], step };
  let tCount = Math.round((rMax - rMin) / step) + 1;
  if (tpsCap > 0 && tpsCap < tCount) tCount = tpsCap;
  console.error(`[dims] timeIdx=${timeIdx} range=[${rMin},${rMax}] step=${step} tCount=${tCount}`);

  const serialize = (e, depth = 0) =>
    !e || depth > 2
      ? null
      : {
          name: e.name,
          lastMs: e.lastMs,
          count: e.count,
          stale: e.stale ?? false,
          children: (e.children ?? []).map((c) => serialize(c, depth + 1)).filter(Boolean),
        };

  function summarizeDeltas(deltas) {
  if (!deltas?.length) return null;
  const sorted = [...deltas].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  const over = (ms) => deltas.filter((d) => d > ms).length;
  return {
    frames: deltas.length,
    medianMs: +q(0.5).toFixed(2),
    p95Ms: +q(0.95).toFixed(2),
    p99Ms: +q(0.99).toFixed(2),
    maxMs: +sorted[sorted.length - 1].toFixed(2),
    over33ms: over(33.4),
    over100ms: over(100),
  };
}

const results = {
  label, renderer, dataset, timeIdx, tCount, scrubs, meta,
  initialLoad: {
    wallMs: Date.now() - loadT0,
    poolStats: initialLoad.poolStats,
    memory: initialLoad.memory,
    frameStats: summarizeDeltas(initialLoad.frameDeltas),
  },
  runs: [],
};

  for (let s = 0; s < scrubs; s++) {
    // Per-scrub reset: lazy-load stage stats + frame recorder.
    await page.evaluate(() => {
      window.__luxarDebug.resetLodLoadStats?.();
      const rec = window.__frameRec;
      rec.deltas = [];
      rec.last = 0;
      rec.on = true;
    });
    const poolBefore = await page.evaluate(
      () => window.__luxarDebug.getSceneLoader().getDefaultLoader()?.gpuBufferPool?.getStats() ?? null
    );

    const perTp = [];
    for (let t = 0; t < tCount; t++) {
      const tv = dim.range[0] + t * (dim.step || 1);
      const t0 = Date.now();
      await page.evaluate(
        ({ timeIdx, tv }) => window.__luxarDebug.app.setDimensionValue(timeIdx, tv),
        { timeIdx, tv }
      );
      await settle(250, 6000);
      perTp.push({ t, wallMs: Date.now() - t0 });
    }

    const scrubEnd = await page.evaluate(() => {
      const dbg = window.__luxarDebug;
      const rec = window.__frameRec;
      rec.on = false;
      const pool = dbg.getSceneLoader().getDefaultLoader()?.gpuBufferPool;
      return {
        lodLoadStats: dbg.getLodLoadStats?.() ?? null,
        poolStats: pool ? pool.getStats() : null,
        memory: dbg.renderer?.info?.memory ? JSON.parse(JSON.stringify(dbg.renderer.info.memory)) : null,
        frameDeltas: rec.deltas,
        profilerTotals: dbg.getSceneLoader().getProfiler().getTimings(),
      };
    });
    results.runs.push({
      perTp,
      poolBefore,
      poolAfter: scrubEnd.poolStats,
      lodLoadStats: scrubEnd.lodLoadStats,
      memory: scrubEnd.memory,
      profilerTotals: serialize(scrubEnd.profilerTotals),
      frameStats: summarizeDeltas(scrubEnd.frameDeltas),
    });
    console.error(`[${label}] scrub ${s + 1}/${scrubs} done (${tCount} tps)`);
  }

  // ---- Dataset-switch churn: release-all + reacquire-all cycles ----
  // Exercises the pool's release/acquire paths (eviction grace,
  // best-fit vs first-fit, fresh-alloc sweeps) and, on webgpu, any
  // buffer stranding (info.memory monotonic growth = leak).
  const altDataset = dataset.replace(
    'gsplats_4d_neuromast_2ch',
    'gsplats_4d_celegans_tracking'
  );
  const churnPool0 = await page.evaluate(
    () => window.__luxarDebug.getSceneLoader().getDefaultLoader()?.gpuBufferPool?.getStats() ?? null
  );
  await page.evaluate(() => {
    const rec = window.__frameRec;
    rec.deltas = [];
    rec.last = 0;
    rec.on = true;
  });
  const churnT0 = Date.now();
  const SWITCHES = 6;
  for (let i = 0; i < SWITCHES; i++) {
    const target = i % 2 === 0 ? altDataset : dataset;
    await page.evaluate((src) => window.__luxarDebug.app.switchDataset(src), target);
    await page.waitForFunction(
      () => {
        const dbg = window.__luxarDebug;
        if (!dbg?.scene) return false;
        let ok = false;
        dbg.scene.traverse((o) => {
          if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) ok = true;
        });
        return ok;
      },
      undefined,
      { timeout: 120000 }
    );
    await settle(400, 20000);
  }
  const churnEnd = await page.evaluate(() => {
    const dbg = window.__luxarDebug;
    const rec = window.__frameRec;
    rec.on = false;
    const pool = dbg.getSceneLoader().getDefaultLoader()?.gpuBufferPool;
    return {
      poolStats: pool ? pool.getStats() : null,
      memory: dbg.renderer?.info?.memory
        ? JSON.parse(JSON.stringify(dbg.renderer.info.memory))
        : null,
      frameDeltas: rec.deltas,
    };
  });
  results.churn = {
    switches: SWITCHES,
    wallMs: Date.now() - churnT0,
    poolBefore: churnPool0,
    poolAfter: churnEnd.poolStats,
    memory: churnEnd.memory,
    frameStats: summarizeDeltas(churnEnd.frameDeltas),
  };
  console.error(`[${label}] churn done (${SWITCHES} switches)`);

  writeFileSync(out, JSON.stringify(results, null, 1));
  console.error(`[${label}] wrote ${out}`);
  await browser.close();
} finally {
  vite.kill('SIGKILL');
}
