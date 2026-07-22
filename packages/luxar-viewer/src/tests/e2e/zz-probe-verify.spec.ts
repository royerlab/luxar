/**
 * TEMPORARY verification probes (double-check campaign 3, iters 7/10/11).
 * DELETE THIS FILE — never commit.
 *
 * Iter 7  — real-GPU validation of the pending-full-upload fix
 *           (element-storage.ts pendingFullUpload WeakSet).
 * Iter 10 — unit-extremes rendering (scene scaled 1e6 / 1e-6).
 * Iter 11 — WEBGL_lose_context storm with dimension commits between cycles.
 *
 * Hardened against concurrent-agent interference: unexpected page
 * navigations/reloads (e.g. vite full-reload from concurrent source edits)
 * are detected and reported as ENVIRONMENT-INTERFERENCE, distinct from a
 * product failure. Evidence is written to delme/zz-probe-evidence/ (NOT
 * test-results/, which a concurrent playwright run wipes).
 */

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import type { Page } from '@playwright/test';
import * as fs from 'fs';
import { waitForLuxarReady, getLuxarState } from './helpers';

const PROGRESSIVE =
  'http://localhost:9000/datasets/examples/progressive_timelapse_example.luxar.zarr';
const DENSE_CUBIC =
  'http://localhost:9000/datasets/examples/dense_cubic_gradient_example.luxar.zarr';
const FRAME_DIM = 3;

const EVIDENCE_DIR = 'delme/zz-probe-evidence';
const MAIN_CANVAS = 'canvas[data-zzprobe-main="1"]';

/* ------------------------------------------------------------------ */
/* interference detection                                              */
/* ------------------------------------------------------------------ */

interface NavWatch {
  /** navigations OTHER than ones we initiated via expectNav(). */
  unexpected: string[];
  expectNav<T>(fn: () => Promise<T>): Promise<T>;
}

function watchNavigation(page: Page): NavWatch {
  let expecting = 0;
  const unexpected: string[] = [];
  page.on('framenavigated', (frame) => {
    if (frame !== page.mainFrame()) return;
    if (expecting > 0) return;
    unexpected.push(`${new Date().toISOString()} -> ${frame.url()}`);
    console.log(`[probe][INTERFERENCE?] unexpected main-frame navigation to ${frame.url()}`);
  });
  page.on('console', (msg) => {
    const t = msg.text();
    if (t.includes('[vite]') && /reload|restart/i.test(t)) {
      unexpected.push(`${new Date().toISOString()} vite: ${t}`);
      console.log(`[probe][INTERFERENCE?] ${t}`);
    }
  });
  return {
    unexpected,
    async expectNav<T>(fn: () => Promise<T>): Promise<T> {
      expecting++;
      try {
        return await fn();
      } finally {
        // Allow the navigation events triggered by fn() to drain.
        setTimeout(() => expecting--, 1000);
      }
    },
  };
}

function assertNoInterference(watch: NavWatch, where: string): void {
  if (watch.unexpected.length > 0) {
    throw new Error(
      `ENVIRONMENT-INTERFERENCE at "${where}": unexpected page navigation(s)/reload(s) ` +
        `occurred mid-test (concurrent agent editing sources / vite full-reload?):\n` +
        watch.unexpected.join('\n')
    );
  }
}

/* ------------------------------------------------------------------ */
/* shared helpers                                                      */
/* ------------------------------------------------------------------ */

/** Tag the app's real WebGL canvas (renderer.domElement) for locators. */
async function tagMainCanvas(page: Page): Promise<{ w: number; h: number; canvases: number }> {
  return await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    const el = debug?.renderer?.domElement as HTMLCanvasElement | undefined;
    if (!el) throw new Error('no renderer.domElement');
    el.setAttribute('data-zzprobe-main', '1');
    return {
      w: el.width,
      h: el.height,
      canvases: document.querySelectorAll('canvas').length,
    };
  });
}

async function renderFrames(page: Page, n = 3): Promise<void> {
  for (let i = 0; i < n; i++) {
    await page.evaluate(() => (window as any).__luxarDebug.renderOnce());
    await page.waitForTimeout(120);
  }
}

async function setFrame(page: Page, frame: number): Promise<void> {
  await page.evaluate(
    ({ dim, f }: { dim: number; f: number }) => {
      (window as any).__luxarDebug.sceneDimsManager.setDimensionValue(dim, f);
    },
    { dim: FRAME_DIM, f: frame }
  );
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      const loader = debug?.getSceneLoader?.()?.getDefaultLoader?.();
      return loader && (loader as any)._updateInProgress === false;
    },
    null,
    { timeout: 20000 }
  );
}

/** Wait until getState().totalPoints >= minPoints AND unchanged for stableMs. */
async function waitPointsStable(
  page: Page,
  minPoints: number,
  stableMs = 1500,
  timeout = 30000
): Promise<number> {
  const t0 = Date.now();
  let last = -1;
  let lastChange = Date.now();
  while (Date.now() - t0 < timeout) {
    const n = await page.evaluate(
      () => (window as any).__luxarDebug.getState().totalPoints as number
    );
    if (n !== last) {
      last = n;
      lastChange = Date.now();
    } else if (n >= minPoints && Date.now() - lastChange >= stableMs) {
      return n;
    }
    await page.waitForTimeout(250);
  }
  throw new Error(`waitPointsStable: timeout (last totalPoints=${last}, min=${minPoints})`);
}

async function canvasDataUrl(page: Page): Promise<string> {
  const loc = page.locator(MAIN_CANVAS);
  await loc.waitFor({ state: 'visible', timeout: 15000 });
  const buf = await loc.screenshot({ animations: 'disabled', timeout: 15000 });
  return `data:image/png;base64,${buf.toString('base64')}`;
}

/** Coverage stats computed from a PNG data URL, in-page. */
async function coverageStats(
  page: Page,
  dataUrl: string,
  cutoff = 10
): Promise<{ width: number; height: number; nonBlackPixels: number; brightest: number[] }> {
  return await page.evaluate(
    async ({ url, cut }) => {
      const img = new Image();
      const loaded = new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('decode failed'));
      });
      img.src = url;
      await loaded;
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      let nonBlack = 0;
      let brightest = [0, 0, 0];
      let bSum = -1;
      for (let i = 0; i < d.length; i += 4) {
        const s = d[i] + d[i + 1] + d[i + 2];
        if (s > cut) nonBlack++;
        if (s > bSum) {
          bSum = s;
          brightest = [d[i], d[i + 1], d[i + 2]];
        }
      }
      return { width: c.width, height: c.height, nonBlackPixels: nonBlack, brightest };
    },
    { url: dataUrl, cut: cutoff }
  );
}

/** Decode two PNG data URLs in-page and compute diff + coverage metrics. */
async function diffDataUrls(
  page: Page,
  a: string,
  b: string,
  channelThreshold = 40,
  coverageCutoff = 10
): Promise<{
  width: number;
  height: number;
  totalPixels: number;
  differingPixels: number;
  diffFraction: number;
  meanAbsDiff: number;
  coverageA: number;
  coverageB: number;
}> {
  return await page.evaluate(
    async ({ urlA, urlB, thr, cut }) => {
      async function decode(url: string): Promise<ImageData> {
        const img = new Image();
        const loaded = new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error('decode failed'));
        });
        img.src = url;
        await loaded;
        const c = document.createElement('canvas');
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        const ctx = c.getContext('2d', { willReadFrequently: true })!;
        ctx.drawImage(img, 0, 0);
        return ctx.getImageData(0, 0, c.width, c.height);
      }
      const A = await decode(urlA);
      const B = await decode(urlB);
      if (A.width !== B.width || A.height !== B.height) {
        throw new Error(`size mismatch ${A.width}x${A.height} vs ${B.width}x${B.height}`);
      }
      const n = A.width * A.height;
      let differing = 0;
      let sumAbs = 0;
      let covA = 0;
      let covB = 0;
      for (let i = 0; i < n * 4; i += 4) {
        const dr = Math.abs(A.data[i] - B.data[i]);
        const dg = Math.abs(A.data[i + 1] - B.data[i + 1]);
        const db = Math.abs(A.data[i + 2] - B.data[i + 2]);
        sumAbs += (dr + dg + db) / 3;
        if (dr > thr || dg > thr || db > thr) differing++;
        if (A.data[i] + A.data[i + 1] + A.data[i + 2] > cut) covA++;
        if (B.data[i] + B.data[i + 1] + B.data[i + 2] > cut) covB++;
      }
      return {
        width: A.width,
        height: A.height,
        totalPixels: n,
        differingPixels: differing,
        diffFraction: differing / n,
        meanAbsDiff: sumAbs / n,
        coverageA: covA,
        coverageB: covB,
      };
    },
    { urlA: a, urlB: b, thr: channelThreshold, cut: coverageCutoff }
  );
}

function saveDataUrl(dataUrl: string, path: string): void {
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
  fs.writeFileSync(path, Buffer.from(dataUrl.split(',')[1], 'base64'));
}

function saveJson(obj: unknown, path: string): void {
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
  fs.writeFileSync(path, JSON.stringify(obj, null, 2));
}

/* ------------------------------------------------------------------ */
/* Iter 7 — pending-full-upload fix on a real GPU                      */
/* ------------------------------------------------------------------ */

test.describe('zz-probe iter 7: pending-full-upload (hidden nav then reveal)', () => {
  test('hidden-navigation reveal renders identically to visible navigation', async ({ page }) => {
    test.setTimeout(240_000);
    const watch = watchNavigation(page);
    const url = `/?src=${PROGRESSIVE}&debug&renderer=webgl&dpr=1`;

    // ---------- run A: navigate frames 1,2 while the mesh is HIDDEN ----------
    await watch.expectNav(() => page.goto(url));
    await waitForLuxarReady(page);
    await waitPointsStable(page, 1);
    const tag = await tagMainCanvas(page);
    console.log(`[probe7] main canvas ${tag.w}x${tag.h}, canvases in DOM: ${tag.canvases}`);

    // Instrument the element texture BEFORE hiding.
    const info = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const meshes: any[] = [];
      debug.scene.traverse((o: any) => {
        if (o?.userData?.nodeType === 'points') meshes.push(o);
      });
      const mesh = meshes[0];
      const tex = mesh?.geometry?.userData?.elementTexture;
      if (!mesh || !tex) return { ok: false as const };
      const log: any[] = [];
      (window as any).__probe = { meshes, mesh, tex, log };
      const origAdd = tex.addUpdateRange.bind(tex);
      tex.addUpdateRange = (start: number, count: number) => {
        log.push({ op: 'add', start, count, v: tex.version, t: performance.now() });
        return origAdd(start, count);
      };
      const origClear = tex.clearUpdateRanges.bind(tex);
      tex.clearUpdateRanges = () => {
        log.push({ op: 'clear', v: tex.version, t: performance.now() });
        return origClear();
      };
      const origOnUpdate = tex.onUpdate;
      tex.onUpdate = function (...args: unknown[]) {
        log.push({ op: 'flush', v: tex.version, t: performance.now() });
        return origOnUpdate ? origOnUpdate.apply(this, args) : undefined;
      };
      return {
        ok: true as const,
        meshCount: meshes.length,
        texW: tex.image.width,
        texH: tex.image.height,
        apiSurface: debug?.app?.sceneManager?.capabilities?.apiSurface ?? 'unknown',
      };
    });
    expect(info.ok).toBe(true);
    expect((info as any).apiSurface).toBe('webgl2');
    console.log(
      `[probe7] meshes=${(info as any).meshCount} tex=${(info as any).texW}x${(info as any).texH}`
    );

    // Hide ALL points meshes (hidden meshes are not drawn -> element
    // textures never bound -> no renderer flush).
    await page.evaluate(() => {
      const p = (window as any).__probe;
      for (const m of p.meshes) m.visible = false;
      p.log.push({ op: 'hide', t: performance.now() });
    });

    // Navigate two slices while hidden.
    await setFrame(page, 1);
    await waitPointsStable(page, 1);
    await setFrame(page, 2);
    const hiddenPoints = await waitPointsStable(page, 1, 2000);

    // Snapshot texture state at reveal time + retrieve the op log.
    const texState = await page.evaluate(() => {
      const p = (window as any).__probe;
      p.log.push({ op: 'reveal', t: performance.now() });
      const liveTex = p.mesh.geometry?.userData?.elementTexture;
      return {
        sameTexture: liveTex === p.tex,
        updateRangesLen: p.tex.updateRanges.length,
        version: p.tex.version,
        log: p.log,
      };
    });

    // Analyze the hidden window: full-mode entries, downgrades, flushes.
    const log: any[] = texState.log;
    const hideT = log.find((e) => e.op === 'hide')?.t ?? 0;
    const revealT = log.find((e) => e.op === 'reveal')?.t ?? Infinity;
    const hidden = log.filter((e) => e.t >= hideT && e.t <= revealT);
    const flushesWhileHidden = hidden.filter((e) => e.op === 'flush').length;
    // full-mode entry = 'clear' whose next texture-op is NOT 'add'
    let lastFullModeIdx = -1;
    for (let i = 0; i < hidden.length; i++) {
      if (hidden[i].op !== 'clear') continue;
      const next = hidden
        .slice(i + 1)
        .find((e) => e.op === 'add' || e.op === 'clear' || e.op === 'flush');
      if (!next || next.op !== 'add') lastFullModeIdx = i;
    }
    let downgradeDetected = false;
    if (lastFullModeIdx >= 0) {
      for (let j = lastFullModeIdx + 1; j < hidden.length; j++) {
        if (hidden[j].op === 'flush') break;
        if (hidden[j].op === 'add') {
          downgradeDetected = true;
          break;
        }
      }
    }
    const counts = hidden.reduce((acc: Record<string, number>, e) => {
      acc[e.op] = (acc[e.op] ?? 0) + 1;
      return acc;
    }, {});
    console.log(
      `[probe7] hidden-window ops=${JSON.stringify(counts)} fullModeEntrySeen=${lastFullModeIdx >= 0} ` +
        `downgradeDetected=${downgradeDetected} flushesWhileHidden=${flushesWhileHidden} ` +
        `sameTexture=${texState.sameTexture} updateRangesAtReveal=${texState.updateRangesLen} version=${texState.version}`
    );
    saveJson(texState, `${EVIDENCE_DIR}/iter7-texture-log.json`);

    // Premise check: hidden meshes must not have flushed.
    expect(flushesWhileHidden).toBe(0);
    // The fix's contract: no ranged downgrade after a pending-full entry.
    expect(downgradeDetected).toBe(false);

    // Reveal + render.
    await page.evaluate(() => {
      const p = (window as any).__probe;
      for (const m of p.meshes) m.visible = true;
    });
    await renderFrames(page, 4);
    const shotA = await canvasDataUrl(page);
    saveDataUrl(shotA, `${EVIDENCE_DIR}/iter7-hidden-nav-reveal.png`);
    const stateA = await getLuxarState(page);

    // Sanity: rendered mesh instance count == getState() count, > 0.
    const meshCount = await page.evaluate(() => {
      const p = (window as any).__probe;
      return p.mesh.geometry.instanceCount as number;
    });
    expect(stateA.totalPoints).toBe(meshCount);
    expect(stateA.totalPoints).toBeGreaterThan(0);
    assertNoInterference(watch, 'end of hidden run A');

    // ---------- run B: same navigation fully VISIBLE (reference) ----------
    await watch.expectNav(() => page.goto(url));
    await waitForLuxarReady(page);
    await waitPointsStable(page, 1);
    await tagMainCanvas(page);
    await setFrame(page, 1);
    await waitPointsStable(page, 1);
    await setFrame(page, 2);
    const visiblePoints = await waitPointsStable(page, 1, 2000);
    await renderFrames(page, 4);
    const shotB = await canvasDataUrl(page);
    saveDataUrl(shotB, `${EVIDENCE_DIR}/iter7-visible-nav-reference.png`);
    const stateB = await getLuxarState(page);

    console.log(
      `[probe7] points hidden-run=${hiddenPoints} visible-run=${visiblePoints} ` +
        `stateA=${stateA.totalPoints} stateB=${stateB.totalPoints}`
    );
    expect(stateA.totalPoints).toBe(stateB.totalPoints);

    // ---------- pixel comparison ----------
    const diff = await diffDataUrls(page, shotA, shotB);
    console.log(`[probe7] diff=${JSON.stringify(diff)}`);
    saveJson(diff, `${EVIDENCE_DIR}/iter7-diff.json`);
    assertNoInterference(watch, 'end of iter 7');
    expect(diff.coverageA).toBeGreaterThan(500);
    expect(diff.coverageB).toBeGreaterThan(500);
    // Stale-prefix rendering would move thousands of points -> large diff.
    expect(diff.diffFraction).toBeLessThan(0.02);
  });
});

/* ------------------------------------------------------------------ */
/* Iter 10 — unit extremes (scene scaled 1e6 / 1e-6)                   */
/* ------------------------------------------------------------------ */

test.describe('zz-probe iter 10: unit-extremes rendering', () => {
  for (const factor of [1e6, 1e-6]) {
    test(`scene scaled by ${factor} still renders points (highp path)`, async ({ page }) => {
      test.setTimeout(180_000);
      const watch = watchNavigation(page);
      const pageErrors: string[] = [];
      page.on('pageerror', (err) => pageErrors.push(err.message));

      await watch.expectNav(() =>
        page.goto(`/?src=${DENSE_CUBIC}&debug&renderer=webgl&dpr=1`)
      );
      await waitForLuxarReady(page);
      console.log(`[probe10 x${factor}] ready`);
      const pts = await waitPointsStable(page, 1, 1500, 45000);
      console.log(`[probe10 x${factor}] points stable: ${pts}`);
      await tagMainCanvas(page);
      await renderFrames(page, 3);

      const baseShot = await canvasDataUrl(page);
      const baseline = await coverageStats(page, baseShot);
      saveDataUrl(baseShot, `${EVIDENCE_DIR}/iter10-baseline-${factor}.png`);
      console.log(`[probe10 x${factor}] baseline coverage=${baseline.nonBlackPixels}`);
      expect(baseline.nonBlackPixels).toBeGreaterThan(500);

      const applied = await page.evaluate((S: number) => {
        const debug = (window as any).__luxarDebug;
        const app = debug.app;
        const scene = debug.scene;
        // 1. Scale the world about the origin.
        scene.scale.multiplyScalar(S);
        scene.updateMatrixWorld(true);
        // 2. Keep position-bounds metadata consistent (drives nearCull +
        //    per-frame dynamic near/far via SceneBoundsCache).
        let boundsCount = 0;
        scene.traverse((o: any) => {
          const pb = o?.userData?.positionBounds;
          if (pb && Array.isArray(pb.min) && Array.isArray(pb.max)) {
            pb.min = pb.min.map((v: number) => v * S);
            pb.max = pb.max.map((v: number) => v * S);
            boundsCount++;
          }
        });
        const sm = app.sceneManager as any;
        sm.boundsCache?.invalidate?.();
        // 3. Radii are element-texture data (not scaled by the model
        //    matrix): compensate via the radiusScale uniform so screen-space
        //    sizes are scale-invariant (radius*S * k / (dist*S)).
        const mats = new Set<any>();
        scene.traverse((o: any) => {
          if (o?.userData?.nodeType === 'points' && o.material) mats.add(o.material);
        });
        let matCount = 0;
        for (const m of mats) {
          if (m?.uniforms?.radiusScale) {
            m.uniforms.radiusScale.value *= S;
            matCount++;
          }
        }
        // 4. Move the camera out/in by S (controls re-initialised so the
        //    orbit state doesn't snap back).
        const pose = app.getCameraPose();
        app.setCameraPose({
          ...pose,
          position: pose.position.map((v: number) => v * S),
          target: pose.target.map((v: number) => v * S),
          near: pose.near * S,
          far: pose.far * S,
        });
        return { boundsCount, matCount, camPos: app.getCameraPose().position };
      }, factor);
      console.log(`[probe10 x${factor}] applied=${JSON.stringify(applied)}`);
      expect(applied.matCount).toBeGreaterThan(0);

      await page.waitForTimeout(500);
      await renderFrames(page, 4);
      console.log(`[probe10 x${factor}] rendered post-scale`);

      const shot = await canvasDataUrl(page);
      const scaled = await coverageStats(page, shot);
      saveDataUrl(shot, `${EVIDENCE_DIR}/iter10-scale-${factor}.png`);

      const glErrors = await page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const el = debug?.renderer?.domElement as HTMLCanvasElement | undefined;
        const gl = el?.getContext('webgl2') as WebGL2RenderingContext | null;
        if (!gl) return ['no webgl2 context on renderer.domElement'];
        const errors: string[] = [];
        let e = 0;
        let guard = 0;
        while ((e = gl.getError()) !== gl.NO_ERROR && guard++ < 100) {
          errors.push(`GL_0x${e.toString(16)}`);
        }
        return errors;
      });
      const camAfter = await page.evaluate(
        () => (window as any).__luxarDebug.getState().camera.position
      );
      console.log(
        `[probe10 x${factor}] baseline=${baseline.nonBlackPixels} scaled=${scaled.nonBlackPixels} ` +
          `brightest=${JSON.stringify(scaled.brightest)} glErrors=${JSON.stringify(glErrors)} ` +
          `pageErrors=${JSON.stringify(pageErrors)} camAfter=${JSON.stringify(camAfter)}`
      );
      assertNoInterference(watch, `iter 10 x${factor}`);

      // Points still visible, with coverage in the same ballpark.
      expect(scaled.nonBlackPixels).toBeGreaterThan(500);
      expect(scaled.nonBlackPixels).toBeGreaterThan(baseline.nonBlackPixels * 0.3);
      expect(scaled.nonBlackPixels).toBeLessThan(baseline.nonBlackPixels * 3);
      // No WebGL errors, no page exceptions.
      expect(glErrors).toEqual([]);
      expect(pageErrors).toEqual([]);
    });
  }
});

/* ------------------------------------------------------------------ */
/* Iter 11 — context-loss storm                                        */
/* ------------------------------------------------------------------ */

test.describe('zz-probe iter 11: context-loss storm', () => {
  test('3x lose/restore with commits between cycles; renders after storm', async ({ page }) => {
    test.setTimeout(240_000);
    test.info().annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description: 'Intentional WEBGL_lose_context storm; loss-time errors are expected.',
    });
    const watch = watchNavigation(page);

    const events: Array<{ t: number; kind: string; text: string }> = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') events.push({ t: Date.now(), kind: 'console', text: msg.text() });
    });
    page.on('pageerror', (err) =>
      events.push({ t: Date.now(), kind: 'pageerror', text: err.message })
    );

    await watch.expectNav(() => page.goto(`/?src=${PROGRESSIVE}&debug&renderer=webgl&dpr=1`));
    await waitForLuxarReady(page);
    await waitPointsStable(page, 1);
    await tagMainCanvas(page);

    const apiSurface = await page.evaluate(
      () =>
        ((window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface as string) ??
        'unknown'
    );
    expect(apiSurface).toBe('webgl2');

    for (let cycle = 1; cycle <= 3; cycle++) {
      // Lose/restore on the RENDERER's canvas (not querySelector — other
      // canvases may exist in the DOM).
      const lost = await page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const el = debug?.renderer?.domElement as HTMLCanvasElement | undefined;
        const gl = el?.getContext('webgl2') as WebGL2RenderingContext | null;
        const ext = gl?.getExtension('WEBGL_lose_context');
        if (!ext) {
          return {
            ok: false,
            hasEl: !!el,
            hasGl: !!gl,
            canvases: document.querySelectorAll('canvas').length,
          };
        }
        ext.loseContext();
        return { ok: true };
      });
      console.log(`[probe11] cycle ${cycle} loseContext -> ${JSON.stringify(lost)}`);
      expect(lost.ok).toBe(true);
      await page.waitForTimeout(300);
      await page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const el = debug?.renderer?.domElement as HTMLCanvasElement | undefined;
        const gl = el?.getContext('webgl2') as WebGL2RenderingContext | null;
        gl?.getExtension('WEBGL_lose_context')?.restoreContext();
      });
      await page.waitForTimeout(800);
      // Dimension navigation (a real commit) between cycles.
      await setFrame(page, cycle);
      const cyclePts = await waitPointsStable(page, 1);
      console.log(`[probe11] cycle ${cycle}: totalPoints after nav = ${cyclePts}`);
    }

    const tAfterStorm = Date.now();
    await renderFrames(page, 4);
    await page.waitForTimeout(500);

    const state = await getLuxarState(page);
    // Re-tag: the recovery path may have rebuilt the renderer/canvas.
    await tagMainCanvas(page);
    const shot = await canvasDataUrl(page);
    const stats = await coverageStats(page, shot);
    saveDataUrl(shot, `${EVIDENCE_DIR}/iter11-after-storm.png`);
    const glErrors = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const el = debug?.renderer?.domElement as HTMLCanvasElement | undefined;
      const gl = el?.getContext('webgl2') as WebGL2RenderingContext | null;
      if (!gl) return ['no webgl2 context on renderer.domElement'];
      if (gl.isContextLost()) return ['context still lost'];
      const errors: string[] = [];
      let e = 0;
      let guard = 0;
      while ((e = gl.getError()) !== gl.NO_ERROR && guard++ < 100) {
        errors.push(`GL_0x${e.toString(16)}`);
      }
      return errors;
    });
    const lateErrors = events.filter((e) => e.t >= tAfterStorm);

    console.log(
      `[probe11] after storm: totalPoints=${state.totalPoints} initialized=${state.initialized} ` +
        `coverage=${stats.nonBlackPixels} brightest=${JSON.stringify(stats.brightest)} ` +
        `glErrors=${JSON.stringify(glErrors)} lateErrors=${JSON.stringify(lateErrors)}`
    );
    saveJson(events, `${EVIDENCE_DIR}/iter11-error-events.json`);
    assertNoInterference(watch, 'end of iter 11');

    // Points render after the storm.
    expect(stats.nonBlackPixels).toBeGreaterThan(500);
    // Counts sane.
    expect(state.initialized).toBe(true);
    expect(Number.isFinite(state.totalPoints)).toBe(true);
    expect(state.totalPoints).toBeGreaterThan(0);
    // No unrecovered errors after the final restore + render.
    expect(lateErrors).toEqual([]);
    expect(glErrors).toEqual([]);
  });
});
