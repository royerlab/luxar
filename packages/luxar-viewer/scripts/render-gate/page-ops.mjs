/**
 * In-page operations for the render gate.
 *
 * Every export is a SELF-CONTAINED function handed to `page.evaluate`, so it may
 * not reference anything outside its own body. They use only debug surfaces
 * that exist on main (`__luxarDebug.app`, `injectSyntheticScene`,
 * `getPerf`, `resortDepthOrderingForCapture`, `getPickingSystem`,
 * `environment`), so the SAME harness drives a baseline build that predates
 * the gate and a candidate build.
 *
 * @module scripts/render-gate/page-ops
 */

/**
 * True once the debug surface is up AND the app finished `init()` (which, with
 * a `?src=` dataset, is after the dataset load starts). `isInitialized` is
 * TypeScript-private but readable at runtime; before it flips, the camera
 * pose API throws.
 */
export function debugReady() {
  const dbg = window.__luxarDebug;
  return !!(
    dbg?.app?.sceneManager?.scene &&
    dbg.app.isInitialized === true &&
    typeof dbg.getPerf === 'function'
  );
}

/** Close the dataset browser a no-`src` boot opens (it would composite over the canvas). */
export function closeDatasetBrowser() {
  document.querySelector('.luxar-dataset-browser__close-btn')?.click();
  return !document.getElementById('luxar-dataset-browser');
}

/**
 * Inject a seeded synthetic scene and hide every non-synthetic geometry node.
 *
 * @param {object} spec `SyntheticSceneSpec`.
 * @returns {Promise<number>} Drawn element count.
 */
export async function injectSynthetic(spec) {
  const dbg = window.__luxarDebug;
  dbg.app.sceneManager.scene.traverse((o) => {
    const t = o.userData?.nodeType;
    if ((t === 'lines' || t === 'points' || t === 'gsplats') && o.userData?.synthetic !== true) {
      o.visible = false;
    }
  });
  const result = await dbg.injectSyntheticScene(spec);
  return result.elementCount;
}

/**
 * Turn off every time-varying or build-independent post effect, so captures
 * and GPU costs measure the scene and nothing else.
 */
export function disableEffects() {
  const pp = window.__luxarDebug.app.sceneManager.postProcessing;
  pp.setDetectorNoiseEnabled(false);
  pp.setVignetteEnabled(false);
  pp.setChromaticLensDistortionEnabled(false);
  pp.setBloomEnabled(false);
}

/**
 * Apply projection, fov and pose.
 *
 * FOV goes through `SceneManager.setFov`, the one path that re-pushes material
 * camera uniforms on main. `setCameraPose` would change `camera.fov` WITHOUT
 * that push on a pre-fix build, making the baseline render the stale-fov bug
 * this work fixes rather than a reference.
 *
 * @param {{ projection: 'perspective'|'ortho', fov?: number,
 *   pose: { position: number[], target: number[], up: number[], zoom?: number } }} view
 */
export function applyView(view) {
  const dbg = window.__luxarDebug;
  const sm = dbg.app.sceneManager;
  const isOrtho = sm.getControlType() === 'ortho';
  if (view.projection === 'ortho' && !isOrtho) sm.setControlType('ortho');
  if (view.projection === 'perspective' && isOrtho) sm.setControlType('orbit');
  if (view.projection === 'perspective' && typeof view.fov === 'number') sm.setFov(view.fov);
  const current = dbg.app.getCameraPose();
  const pose = {
    ...current,
    position: view.pose.position,
    target: view.pose.target,
    up: view.pose.up,
  };
  if (view.projection === 'ortho' && typeof view.pose.zoom === 'number') pose.zoom = view.pose.zoom;
  dbg.app.setCameraPose(pose);
  dbg.renderOnce();
}

/**
 * Drive frames until the viewer is settled for `minFrames` consecutive frames
 * (and, when asked, the scene-captured environment exists), then force a
 * quiescent depth sort for the current pose.
 *
 * A synthetic scene has no dataset loader, so the loader's refinement never
 * completes and `getPerf().isSettled` never turns true; pass
 * `requireLoaderSettled: false` there and it settles on frames alone (plus the
 * forced depth sort, which is what an injected node actually waits on).
 *
 * @param {{ minFrames?: number, waitEnvironment?: boolean, timeoutMs?: number,
 *   requireLoaderSettled?: boolean }} opts
 * @returns {Promise<{ settled: boolean, frames: number }>}
 */
export async function settle({
  minFrames = 30,
  waitEnvironment = false,
  timeoutMs = 120000,
  requireLoaderSettled = true,
}) {
  const dbg = window.__luxarDebug;
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const envReady = () => {
    if (!waitEnvironment) return true;
    const env = dbg.environment;
    return !!env && env.kind() === 'scene' && env.captureCount() >= 1;
  };
  const t0 = performance.now();
  let streak = 0;
  let frames = 0;
  while (performance.now() - t0 < timeoutMs) {
    dbg.renderOnce();
    await frame();
    frames++;
    const loaderOk = !requireLoaderSettled || !!dbg.getPerf()?.isSettled;
    streak = loaderOk && envReady() ? streak + 1 : 0;
    if (streak >= minFrames) {
      await dbg.resortDepthOrderingForCapture?.(10000);
      for (let i = 0; i < 5; i++) {
        dbg.renderOnce();
        await frame();
      }
      return { settled: true, frames };
    }
  }
  return { settled: false, frames };
}

/**
 * Per-node drawn element counts and blending modes. Two builds rendering the
 * same frame must agree on these; a difference means LOD selection, residency
 * or streaming diverged, which would otherwise surface as a baffling pixel diff.
 *
 * @returns {Record<string, string>} node path → "type:count:blending:visible".
 */
export function elementCounts() {
  const out = {};
  window.__luxarDebug.app.sceneManager.scene.traverse((o) => {
    const t = o.userData?.nodeType;
    if (!t) return;
    const names = [];
    for (let n = o; n; n = n.parent) names.unshift(n.name || n.type);
    const u = o.userData;
    const count =
      u.visiblePointCount ??
      u.visibleSplatCount ??
      u.visibleSegmentCount ??
      u.committedVertexCount ??
      o.geometry?.instanceCount ??
      '?';
    let visible = true;
    for (let n = o; n; n = n.parent) visible = visible && n.visible;
    out[names.join('/')] =
      `${t}:${count}:${u.blendingMode ?? o.material?.userData?.blendingMode ?? '?'}:${visible}`;
  });
  return out;
}

/**
 * Capture the frame: raw scene HDR (float), visible LDR (float, pre-sRGB),
 * each twice for a self-consistency check, and optionally the pick-ID buffer.
 *
 * The render loop is stopped for the capture so no rAF frame interleaves with
 * a readback. Half-float captures are shipped as half-float bits (lossless: the
 * targets are HalfFloat) to halve the transfer; a capture that does not round-
 * trip exactly is shipped as float32 instead and flagged.
 *
 * @param {{ pick: boolean }} opts
 * @returns {Promise<object>} Base64 buffers, sizes, and consistency flags.
 */
export async function captureFrame({ pick }) {
  const dbg = window.__luxarDebug;
  const sm = dbg.app.sceneManager;
  const pp = sm.postProcessing;

  // --- float32 -> float16 bits, and back, for the lossless check ---
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const toHalf = (v) => {
    f32[0] = v;
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    const mant = x & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
    const e = exp - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00;
    if (e <= 0) {
      if (e < -10) return sign;
      const m = (mant | 0x800000) >>> (1 - e + 13);
      return sign | m;
    }
    return sign | (e << 10) | (mant >>> 13);
  };
  const fromHalf = (h) => {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >>> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) return s * m * 2 ** -24;
    if (e === 0x1f) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * 2 ** (e - 15);
  };
  const b64 = (bytes) => {
    let s = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(s);
  };
  const pack = (pixels) => {
    const half = new Uint16Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) {
      const h = toHalf(pixels[i]);
      const back = fromHalf(h);
      if (!(back === pixels[i] || (Number.isNaN(back) && Number.isNaN(pixels[i])))) {
        return { format: 'f32', data: b64(new Uint8Array(pixels.buffer.slice(0))) };
      }
      half[i] = h;
    }
    return { format: 'f16', data: b64(new Uint8Array(half.buffer)) };
  };
  const same = (a, b) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!(Object.is(a[i], b[i]) || a[i] === b[i])) return false;
    }
    return true;
  };

  dbg.animationController.stopAnimation();
  try {
    const hdr1 = await pp.captureHDRPixels('raw-scene-hdr', { flipY: true });
    const hdr2 = await pp.captureHDRPixels('raw-scene-hdr', { flipY: true });
    const ldr1 = await pp.captureHDRPixels('visible-ldr', { flipY: true });
    const ldr2 = await pp.captureHDRPixels('visible-ldr', { flipY: true });
    const cam = sm.camera;
    cam.updateMatrixWorld();
    const result = {
      width: hdr1.width,
      height: hdr1.height,
      // The exact camera the capture used: two arms whose captures differ but
      // whose camera does not point at the renderer; differing cameras point
      // at the harness (pose not converged) instead.
      camera: {
        world: Array.from(cam.matrixWorld.elements),
        projection: Array.from(cam.projectionMatrix.elements),
      },
      hdr: pack(hdr1.pixels),
      ldr: pack(ldr1.pixels),
      hdrStable: same(hdr1.pixels, hdr2.pixels),
      ldrStable: same(ldr1.pixels, ldr2.pixels),
      pick: null,
    };
    if (pick) {
      const ps = dbg.getPickingSystem?.();
      if (ps) {
        const canvas = sm.renderer.domElement;
        ps.markDirty();
        await ps.performPick(canvas.clientWidth / 2, canvas.clientHeight / 2, true);
        const target = ps.pickTarget;
        const w = target.width;
        const h = target.height;
        // WebGLRenderer fills a caller-supplied buffer; WebGPURenderer returns one.
        let pixels = new Float32Array(w * h * 4);
        if (sm.renderer.isWebGPURenderer) {
          const raw = await sm.renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h);
          pixels = raw instanceof Float32Array ? raw : new Float32Array(raw.buffer);
        } else {
          await sm.renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h, pixels);
        }
        result.pick = { width: w, height: h, data: b64(new Uint8Array(pixels.buffer.slice(0))) };
      }
    }
    return result;
  } finally {
    dbg.animationController.startAnimation();
  }
}

/**
 * GPU cost of one frame: `k` back-to-back full post-processing renders, then a
 * GPU sync, timed on the wall clock. Independent of vsync and of timer queries
 * (which ANGLE/Metal reports unreliably). Repeated `reps` times; `minMs`, the
 * fastest repetition, is the least-interfered estimate of the true cost and
 * is what the gate compares (a slower repetition measured something else
 * sharing the GPU, not the build).
 *
 * @param {{ k?: number, warm?: number, reps?: number }} opts
 * @returns {Promise<{ msPerFrame: number[], minMs: number }>}
 */
export async function gpuCost({ k = 20, warm = 10, reps = 5 }) {
  const dbg = window.__luxarDebug;
  const sm = dbg.app.sceneManager;
  const pp = sm.postProcessing;
  const renderer = sm.renderer;
  const backend = renderer.backend;
  const px = new Uint8Array(4);
  const sync = async () => {
    if (backend?.isWebGPUBackend && backend.device) {
      await backend.device.queue.onSubmittedWorkDone();
      return;
    }
    const gl = backend?.gl ?? renderer.getContext();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  };
  dbg.animationController.stopAnimation();
  try {
    for (let i = 0; i < warm; i++) pp.render();
    await sync();
    const msPerFrame = [];
    for (let r = 0; r < reps; r++) {
      const t0 = performance.now();
      for (let i = 0; i < k; i++) pp.render();
      await sync();
      msPerFrame.push((performance.now() - t0) / k);
    }
    return { msPerFrame, minMs: Math.min(...msPerFrame) };
  } finally {
    dbg.animationController.startAnimation();
  }
}

/**
 * Full-frame cost under camera MOTION: a deterministic orbit, one step per
 * animation frame, so the per-frame CPU work that only runs when the view
 * changes (LOD selection, depth-sort scheduling, density guard, dynamic
 * clipping) is exercised, identically in both builds. A static camera
 * measures almost none of it.
 *
 * Reports ms per presented frame (wall time / rAF ticks), the p95 rAF
 * interval, and `rendersPerFrame`, counted by wrapping
 * `postProcessing.render` (present in every build): a frame that renders
 * twice costs twice and should show up. Launch the browser with
 * `--disable-gpu-vsync --disable-frame-rate-limit` so frames are not pinned to
 * the display refresh. The caller reads CPU script time around this call.
 *
 * @param {{ frames?: number, warm?: number, degPerFrame?: number,
 *   pose: { position: number[], target: number[], up: number[] } }} opts
 * @returns {Promise<{ frames: number, renders: number, frameMs: number,
 *   p95Ms: number, rendersPerFrame: number }>}
 */
export async function motion({ frames = 180, warm = 30, degPerFrame = 1, pose }) {
  const dbg = window.__luxarDebug;
  const pp = dbg.app.sceneManager.postProcessing;
  const tick = () => new Promise((r) => requestAnimationFrame((t) => r(t)));
  const [tx, , tz] = pose.target;
  const dx = pose.position[0] - tx;
  const dz = pose.position[2] - tz;
  const radius = Math.hypot(dx, dz);
  const phase = Math.atan2(dz, dx);
  const place = (i) => {
    const a = phase + (i * degPerFrame * Math.PI) / 180;
    dbg.app.setCameraPose({
      ...dbg.app.getCameraPose(),
      position: [tx + radius * Math.cos(a), pose.position[1], tz + radius * Math.sin(a)],
      target: pose.target,
      up: pose.up,
    });
    dbg.renderOnce();
  };
  for (let i = 0; i < warm; i++) {
    place(i - warm);
    await tick();
  }
  let renders = 0;
  const original = pp.render;
  pp.render = function countedRender(...args) {
    renders++;
    return original.apply(this, args);
  };
  const deltas = [];
  let t0;
  let last;
  try {
    t0 = await tick();
    last = t0;
    for (let i = 0; i < frames; i++) {
      place(i);
      const t = await tick();
      deltas.push(t - last);
      last = t;
    }
  } finally {
    pp.render = original;
  }
  deltas.sort((a, b) => a - b);
  return {
    frames,
    renders,
    frameMs: (last - t0) / frames,
    p95Ms: deltas[Math.min(deltas.length - 1, Math.floor(0.95 * deltas.length))],
    rendersPerFrame: renders / frames,
  };
}

/** The GL/GPU adapter string, to refuse a software renderer. */
export function rendererInfo() {
  const r = window.__luxarDebug.app.sceneManager.renderer;
  const backend = r.backend;
  if (backend?.isWebGPUBackend) {
    const info = backend.adapter?.info ?? {};
    return {
      api: 'webgpu',
      gpu: `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.description ?? ''}`.trim(),
    };
  }
  const gl = backend?.gl ?? r.getContext();
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const gpu = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return { api: 'webgl', gpu: String(gpu) };
}
