/**
 * Grow-leak probe (native WebGPU): forces the GPU pool's GROW path
 * (same nodeId, doubling splat counts, rendered each step), then
 * releases + LRU-evicts, and reports renderer.info.memory.attributes
 * at every step. BEFORE (main, in-place rebuild): each grow replaces
 * the geometry's views; the replaced views stay pinned in the strong
 * Info.memoryMap forever. AFTER (grow = release + reacquire): old
 * buffers are pooled/evicted through geometry.dispose() and the count
 * returns to baseline.
 *
 * node grow-leak-probe.mjs --viewer-dir <dir> --port 5196 [--chrome]
 */
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';

const arg = (n, d) => {
  const i = process.argv.indexOf(`--${n}`);
  return i >= 0 ? process.argv[i + 1] : d;
};
const viewerDir = path.resolve(arg('viewer-dir', '.'));
const port = Number(arg('port', '5196'));
const channel = process.argv.includes('--chrome') ? 'chrome' : undefined;
const require = createRequire(path.join(viewerDir, 'package.json'));
const { chromium } = require('@playwright/test');

async function waitHttp(url, t = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < t) {
    try {
      const r = await fetch(url);
      if (r.ok || r.status === 404) return;
    } catch {
      // Connection refused while the server is still coming up: retry below.
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`no server at ${url}`);
}

const vite = spawn('npx', ['vite', '--port', String(port), '--strictPort'], {
  cwd: viewerDir,
  stdio: 'ignore',
});
process.on('exit', () => vite.kill('SIGKILL'));

try {
  await waitHttp(`http://localhost:${port}/`);
  const browser = await chromium.launch({
    headless: true,
    ...(channel ? { channel } : {}),
    args: [
      '--disable-web-security',
      ...(channel ? [] : ['--use-gl=egl']),
      '--ignore-gpu-blocklist',
      '--no-sandbox',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message.slice(0, 160)));
  await page.goto(
    `http://localhost:${port}/?src=http://127.0.0.1:9009/datasets/demos/gsplats_4d_neuromast_2ch.luxar.zarr&debug&dpr=1&renderer=webgpu`
  );
  await page.waitForFunction(() => !!window.__luxarDebug?.app, undefined, { timeout: 60000 });
  await page.waitForFunction(
    () => {
      const dbg = window.__luxarDebug;
      let ok = false;
      dbg?.scene?.traverse((o) => {
        if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) ok = true;
      });
      return ok;
    },
    undefined,
    { timeout: 120000 }
  );
  await new Promise((r) => setTimeout(r, 3000));

  const result = await page.evaluate(async () => {
    const dbg = window.__luxarDebug;
    const renderer = dbg.renderer;
    const backend = renderer?.backend?.constructor?.name;
    const pool = dbg.getSceneLoader().getDefaultLoader()?.gpuBufferPool;
    if (!pool) return { error: 'no pool' };

    // Find a live gsplat mesh to steal Mesh ctor + material from.
    let host = null;
    dbg.scene.traverse((o) => {
      if (!host && o.userData?.nodeType === 'gsplats' && o.geometry?.instanceCount > 0) host = o;
    });
    if (!host) return { error: 'no gsplat mesh' };
    const MeshCtor = host.constructor;

    // Render until the measurement ground truth moves: the wrapper's
    // attribute views REGISTER in info.memory (or timeout). Frame
    // counters and fixed rAF waits both proved unreliable across
    // display timings (a headed X11 run registered nothing).
    const renderUntilRegistered = async (prevAttrs) => {
      const t0 = performance.now();
      while (performance.now() - t0 < 8000) {
        dbg.renderOnce();
        await new Promise((r) => requestAnimationFrame(r));
        await new Promise((r) => setTimeout(r, 50));
        if (renderer.info.memory.attributes !== prevAttrs) return true;
      }
      return false;
    };
    const renderTick = async () => {
      dbg.renderOnce();
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    };
    const mem = () => ({
      attributes: renderer.info.memory.attributes,
      attributesSize: renderer.info.memory.attributesSize,
      // Phase 1 texture storage: splat data lives in RGBA32F textures
      // that must return to the clean floor after grows + evict, same
      // as attributes (the Info.memoryMap strand class applies to
      // textures too).
      textures: renderer.info.memory.textures,
    });

    const steps = [];

    // DRAIN the pool of initial-load leftovers first: best-fit reuse
    // would otherwise serve small 'grows' from an already-registered
    // pooled buffer (observed: a ladder leftover with capacity >= 32K
    // absorbed grows 0-3, so only ONE real grow ever happened and the
    // in-place-rebuild strand went unexercised).
    for (let i = 0; i < 305; i++) pool.beginFrame();
    pool.evictUnused();
    steps.push({ step: 'drained', ...mem(), pool: pool.getStats().pooledBuffers });

    // PRE-WARM: the first draw of a wrapper mesh needs the (async)
    // WebGPU pipeline compile for this material+geometry-layout combo —
    // observed at 10+ seconds cold on Vulkan, swallowing several grow
    // windows and making generations 'invisible' to registration. Warm
    // it once with a throwaway acquire, wait for registration, release.
    {
      const warmGeo = pool.acquireGSplatsGeometry('synthetic-warm', 1000);
      warmGeo.instanceCount = 1000;
      const warm = new MeshCtor(warmGeo, host.material);
      warm.frustumCulled = false;
      let warmDraws = 0;
      warm.onAfterRender = () => warmDraws++;
      dbg.scene.add(warm);
      const a0 = renderer.info.memory.attributes;
      const t0 = performance.now();
      let warmed = false;
      while (performance.now() - t0 < 45000) {
        dbg.renderOnce();
        await new Promise((r) => requestAnimationFrame(r));
        await new Promise((r) => setTimeout(r, 50));
        if (renderer.info.memory.attributes !== a0) {
          warmed = true;
          break;
        }
      }
      dbg.scene.remove(warm);
      pool.releaseGSplatsGeometry('synthetic-warm');
      // Drain the warm buffer too — it must not be best-fit-served to
      // the measured grows below.
      for (let i = 0; i < 305; i++) pool.beginFrame();
      pool.evictUnused();
      steps.push({ step: 'prewarm', ...mem(), warmed, warmDraws });
    }
    steps.push({ step: 'baseline', ...mem() });

    let wrapper = null;
    let count = 4000;
    for (let g = 0; g < 5; g++) {
      const geo = pool.acquireGSplatsGeometry('synthetic-grow', count);
      geo.instanceCount = count;
      // Fresh wrapper mesh each step: avoids stale RenderObject caches
      // on geometry-identity swaps (the production commit path handles
      // that via invalidateRenderObjectFor, unreachable from here).
      if (wrapper) dbg.scene.remove(wrapper);
      wrapper = new MeshCtor(geo, host.material);
      wrapper.frustumCulled = false;
      dbg.scene.add(wrapper);
      const framesBefore = renderer.info.render?.frame ?? -1;
      const callsBefore = renderer.info.render?.calls ?? -1;
      const registered = await renderUntilRegistered(renderer.info.memory.attributes);
      steps.push({
        step: `grow${g}(n=${count})`,
        ...mem(),
        registered,
        sameGeom: wrapper.geometry === geo,
        frames: `${framesBefore}->${renderer.info.render?.frame ?? -1}`,
        calls: `${callsBefore}->${renderer.info.render?.calls ?? -1}`,
        wrapperInScene: wrapper.parent === dbg.scene,
        geoInstanceCount: geo.instanceCount,
        geoAttrCount: Object.keys(geo.attributes).length,
      });
      count *= 2;
    }

    // Tear down: remove wrapper, release, age the pool past the LRU
    // window, evict everything evictable.
    dbg.scene.remove(wrapper);
    pool.releaseGSplatsGeometry('synthetic-grow');
    for (let i = 0; i < 305; i++) pool.beginFrame();
    pool.evictUnused();
    await renderTick();
    steps.push({ step: 'after-evict', ...mem(), pool: pool.getStats() });

    return { backend, steps };
  });
  console.log(JSON.stringify(result, null, 1));
  await browser.close();
} finally {
  vite.kill('SIGKILL');
}
