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
  // Orient the camera to the pose first. Before "Camera: a restored pose keeps
  // its up vector", setCameraPose set position and up but not the orientation,
  // and orbit's reinitialize() read the stale quaternion: a freshly swapped
  // ortho camera (identity) or the previous pose's roll survived until a
  // controls update happened to run, which an uncapped frame rate could lose.
  // Such a baseline measured a different view from the candidate.
  const cam = dbg.camera;
  cam.position.fromArray(pose.position);
  cam.up.fromArray(pose.up);
  cam.lookAt(pose.target[0], pose.target[1], pose.target[2]);
  cam.updateMatrixWorld();
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
 * Every measured frame ENDS WITH A GPU SYNC (a 1-pixel `readPixels` on
 * WebGL, `queue.onSubmittedWorkDone()` on WebGPU, as in `gpuCost`), and the
 * frame's time is taken after it. Without it the numbers are CPU submission
 * only: in headless Chrome with vsync off an animation frame never waits for
 * the GPU, so a 100 ms GPU frame reads 0.1 ms, a GPU-bound change is
 * invisible, and a CPU stall on a resource the GPU still holds (a blocking
 * `bufferSubData`) swings the number by orders of magnitude at unchanged GPU
 * cost. With the sync a frame costs its CPU work plus its GPU work, serialized
 * (no CPU/GPU overlap), which is what the gate compares.
 *
 * Reports ms per frame (wall time / frames), the p95 frame time, and
 * `rendersPerFrame`, counted by wrapping `postProcessing.render` (present in
 * every build): a frame that renders twice costs twice and should show up.
 * Launch the browser with `--disable-gpu-vsync --disable-frame-rate-limit` so
 * frames are not pinned to the display refresh. The caller reads CPU script
 * time around this call (CDP `ScriptDuration`, which does not count the time
 * spent blocked in the sync).
 *
 * @param {{ frames?: number, warm?: number, degPerFrame?: number,
 *   pose: { position: number[], target: number[], up: number[] } }} opts
 * @returns {Promise<{ frames: number, renders: number, frameMs: number,
 *   p95Ms: number, rendersPerFrame: number }>}
 */
export async function motion({ frames = 180, warm = 30, degPerFrame = 1, pose }) {
  const dbg = window.__luxarDebug;
  const pp = dbg.app.sceneManager.postProcessing;
  const renderer = dbg.app.sceneManager.renderer;
  const backend = renderer.backend;
  const tick = () => new Promise((r) => requestAnimationFrame(() => r()));
  const px = new Uint8Array(4);
  // Wait for the GPU to finish everything submitted so far (as gpuCost does),
  // and return the time it did. The WebGL framebuffer binding is restored so
  // three's state cache stays truthful.
  const sync = async () => {
    if (backend?.isWebGPUBackend && backend.device) {
      await backend.device.queue.onSubmittedWorkDone();
    } else {
      const gl = backend?.gl ?? renderer.getContext();
      const bound = gl.getParameter(gl.FRAMEBUFFER_BINDING);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.bindFramebuffer(gl.FRAMEBUFFER, bound);
    }
    return performance.now();
  };
  const [tx, , tz] = pose.target;
  const dx = pose.position[0] - tx;
  const dz = pose.position[2] - tz;
  const radius = Math.hypot(dx, dz);
  const phase = Math.atan2(dz, dx);
  const place = (i) => {
    const a = phase + (i * degPerFrame * Math.PI) / 180;
    const position = [tx + radius * Math.cos(a), pose.position[1], tz + radius * Math.sin(a)];
    // Orient first, as applyView does (serialized into the page: no shared helper).
    const cam = dbg.camera;
    cam.position.fromArray(position);
    cam.up.fromArray(pose.up);
    cam.lookAt(pose.target[0], pose.target[1], pose.target[2]);
    cam.updateMatrixWorld();
    dbg.app.setCameraPose({
      ...dbg.app.getCameraPose(),
      position,
      target: pose.target,
      up: pose.up,
    });
    dbg.renderOnce();
  };
  for (let i = 0; i < warm; i++) {
    place(i - warm);
    await tick();
    await sync();
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
    // The loop renders in the animation frame before this tick resolves, so
    // the sync after it covers that frame's GPU work.
    await tick();
    t0 = await sync();
    last = t0;
    for (let i = 0; i < frames; i++) {
      place(i);
      await tick();
      const t = await sync();
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

/**
 * The cost of WAKING a stopped loop: stop it, let two animation frames pass,
 * then call `startAnimation()` as an input handler would, and measure
 *
 * - `renders`: how many times the pipeline rendered from the wake up to and
 *   including the first animation frame after it (the frame that paints).
 *   One is the minimum a wake needs; a wake that also renders inline inside
 *   the handler costs two.
 * - `blockMs`: how long the `startAnimation()` call itself blocked its
 *   caller, i.e. the input latency a wake adds to the handler that caused it.
 *
 * Our own rAF is registered AFTER the wake, so the loop's armed frame (if
 * any) runs before it in the same animation frame. Medians over `reps`.
 */
export async function wake({ reps = 9 }) {
  const dbg = window.__luxarDebug;
  const ac = dbg.animationController;
  const pp = dbg.app.sceneManager.postProcessing;
  const tick = () => new Promise((r) => requestAnimationFrame(() => r()));
  const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
  let renders = 0;
  const original = pp.render;
  pp.render = function countedRender(...args) {
    renders++;
    return original.apply(this, args);
  };
  const perWake = [];
  const blocked = [];
  try {
    for (let i = 0; i < reps; i++) {
      ac.stopAnimation();
      await tick();
      await tick();
      renders = 0;
      const t0 = performance.now();
      ac.startAnimation();
      blocked.push(performance.now() - t0);
      await tick();
      perWake.push(renders);
    }
  } finally {
    pp.render = original;
    ac.startAnimation();
  }
  return { renders: med(perWake), blockMs: med(blocked) };
}

/**
 * The adapter the page renders on, so the harness can refuse to measure a
 * software renderer, a WebGPU fallback adapter, or a lost WebGL context.
 *
 * WebGPU: three does not expose its adapter, so the page's own
 * `navigator.gpu.requestAdapter()` answers (same browser, same flags, same
 * adapter selection). WebGL: the unmasked renderer string, or null when the
 * context is lost (`getParameter` returns null on a lost context).
 *
 * @returns {Promise<{ api: 'webgl'|'webgpu'|'webgpu-gl', gpu: string, usable: boolean, why?: string }>}
 */
export async function rendererInfo() {
  const r = window.__luxarDebug.app.sceneManager.renderer;
  const backend = r.backend;
  if (backend?.isWebGPUBackend) {
    const adapter = await navigator.gpu?.requestAdapter();
    const info = adapter?.info ?? {};
    const gpu = [info.vendor, info.architecture, info.device, info.description]
      .filter(Boolean)
      .join(' ');
    const fallback = info.isFallbackAdapter ?? adapter?.isFallbackAdapter ?? false;
    if (!adapter) return { api: 'webgpu', gpu: '', usable: false, why: 'no WebGPU adapter' };
    if (fallback) return { api: 'webgpu', gpu, usable: false, why: 'fallback (software) adapter' };
    if (!gpu) return { api: 'webgpu', gpu, usable: false, why: 'adapter reports no identity' };
    return { api: 'webgpu', gpu, usable: true };
  }
  const gl = backend?.gl ?? r.getContext();
  const api = r.isWebGPURenderer && backend?.isWebGLBackend ? 'webgpu-gl' : 'webgl';
  if (gl.isContextLost()) return { api, gpu: '', usable: false, why: 'WebGL context lost' };
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const gpu = String(
    ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)
  );
  return { api, gpu, usable: true };
}

// ---------------------------------------------------------------------------
// Suite workloads
//
// Each is SELF-CONTAINED (no shared helpers: `page.evaluate` serializes one
// function). They use only surfaces present on older builds too —
// `__luxarDebug.{app, camera, renderOnce, getPerf, getSceneLoader,
// inputHandler, sceneDimsManager}`, `app.{getCameraPose, setCameraPose,
// getDimensions, setDimensionValue, awaitDimensionUpdate}` and
// `sceneManager.postProcessing.render` (wrapped to count renders) — plus the
// perf-counter APIs (`resetPerfCounters`, `getPerfRecords`), always behind
// `?.`. The suite runner reads the counters after the workload; a workload
// that has to move the scene first (to the start of a timeline) zeroes them
// again itself once it is in place.
// ---------------------------------------------------------------------------

/** Zero the viewer's perf counters (if the build has them) and the gate's ext counters. */
export function resetCounters() {
  window.__luxarDebug.resetPerfCounters?.();
  window.__gateExtReset?.();
  return true;
}

/** Flat snapshot of the viewer's perf counters (`{}` on a build without them). */
export function readCounters() {
  return { ...(window.__luxarDebug.getPerf?.()?.counters ?? {}) };
}

/** Snapshot of the build-independent ext counters (`ext-counters.mjs`). */
export function readExtCounters() {
  return window.__gateExtSnapshot?.() ?? {};
}

/**
 * Start counting `postProcessing.render` calls into `window.__gateRenders`
 * (the render count every build exposes, counters or not).
 */
export function beginRenderCount() {
  const pp = window.__luxarDebug.app.sceneManager.postProcessing;
  if (pp.__gateOriginalRender) return true;
  window.__gateRenders = 0;
  pp.__gateOriginalRender = pp.render;
  pp.render = function countedRender(...args) {
    window.__gateRenders++;
    return pp.__gateOriginalRender.apply(this, args);
  };
  return true;
}

/** Stop counting renders; returns the count since `beginRenderCount`. */
export function endRenderCount() {
  const pp = window.__luxarDebug.app.sceneManager.postProcessing;
  if (pp.__gateOriginalRender) {
    pp.render = pp.__gateOriginalRender;
    delete pp.__gateOriginalRender;
  }
  return window.__gateRenders ?? 0;
}

/**
 * Do nothing for `ms` and count what the viewer does anyway: renders and
 * animation frames. An idle viewer should render nothing.
 *
 * @param {{ ms?: number }} opts
 * @returns {Promise<{ renders: number, frames: number, idleMs: number }>}
 */
export async function idle({ ms = 1000 }) {
  const pp = window.__luxarDebug.app.sceneManager.postProcessing;
  let renders = 0;
  const original = pp.render;
  pp.render = function countedRender(...args) {
    renders++;
    return original.apply(this, args);
  };
  let frames = 0;
  let live = true;
  const count = () => {
    if (!live) return;
    frames++;
    requestAnimationFrame(count);
  };
  requestAnimationFrame(count);
  try {
    await new Promise((r) => setTimeout(r, ms));
  } finally {
    live = false;
    pp.render = original;
  }
  return { renders, frames, idleMs: ms };
}

/**
 * Play a hidden dimension through the viewer's own playback
 * (`inputHandler.getAnimationManager().play(dim, { targetFPS, loopMode })`)
 * and measure what it delivered.
 *
 * The dimension is first moved to its range minimum and allowed to load; the
 * counters are zeroed after that, so they cover playback only. Runs for
 * `durationMs`, or with `loops` until the value has wrapped that many times.
 *
 * Returns:
 * - `lodLevelMean` / `lodCoarsestFrac`: over every frame and every lod group
 *   with two or more levels, the mean DISPLAYED level as a fraction of the
 *   finest (0 = coarsest, 1 = finest) and the share of samples at the
 *   coarsest level (null without lod groups) — whether playback keeps detail.
 * - `ticks`: value advances (from `playback.tick` records when the build has
 *   them, else the observed changes of the dimension's value per frame);
 * - `achievedFps`: ticks per second over the first..last tick;
 * - `renders`, `rendersPerTick`: `postProcessing.render` calls;
 * - `lastTimepointShown`: 1 if the range maximum was ever applied, else 0;
 * - `tickLatencyP50Ms` / `tickLatencyP90Ms`: lateness `t - due` of the ticks
 *   (records only; absent otherwise);
 * - `wraps`, `durationMs`.
 *
 * @param {{ dim?: number, fps?: number, durationMs?: number, loops?: number,
 *   timeoutMs?: number }} opts `dim` defaults to the first non-displayed dimension.
 */
export async function playback({ dim, fps = 10, durationMs = 6000, loops, timeoutMs = 180000 }) {
  const dbg = window.__luxarDebug;
  const app = dbg.app;
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const dims = app.getDimensions();
  const d = dim ?? dims.metadata.findIndex((_, i) => !dims.displayed.includes(i));
  if (!(d >= 0 && d < dims.ndim)) throw new Error(`playback: no hidden dimension (dim=${dim})`);
  const [lo, hi] = dims.ranges[d];
  const manager = (dbg.inputHandler ?? app.inputHandler)?.getAnimationManager?.();
  if (!manager) throw new Error('playback: no dimension animation manager');
  const valueOf = () =>
    dbg.sceneDimsManager?.getDims?.()?.currentStep?.[d] ?? app.getDimensions().currentStep[d];

  app.setDimensionValue(d, lo);
  await app.awaitDimensionUpdate();
  for (let i = 0; i < 10; i++) await frame();
  dbg.resetPerfCounters?.();
  window.__gateExtReset?.();

  const pp = app.sceneManager.postProcessing;
  let renders = 0;
  const original = pp.render;
  pp.render = function countedRender(...args) {
    renders++;
    return original.apply(this, args);
  };
  const eps = Math.max(1e-9, Math.abs(hi - lo) * 1e-6);
  const lod = { sum: 0, coarsest: 0, n: 0 };
  const sampleLod = () => {
    const reg = dbg.getSceneLoader?.()?.getDefaultLoader?.()?.lodGroupRegistry;
    for (const e of reg?.list?.() ?? []) {
      const levels = e.children?.length ?? 0;
      const shown = e.displayedChildIndex;
      if (levels < 2 || typeof shown !== 'number' || shown < 0) continue;
      lod.sum += shown / (levels - 1);
      if (shown === 0) lod.coarsest++;
      lod.n++;
    }
  };
  const observed = [];
  let last = valueOf();
  let wraps = 0;
  let sawMax = Math.abs(last - hi) <= eps;
  const t0 = performance.now();
  manager.play(d, { targetFPS: fps, loopMode: 'loop' });
  try {
    for (;;) {
      await frame();
      sampleLod();
      const v = valueOf();
      if (v !== last) {
        observed.push({ t: performance.now(), value: v });
        if (v < last) wraps++;
        if (Math.abs(v - hi) <= eps) sawMax = true;
        last = v;
      }
      const elapsed = performance.now() - t0;
      if (loops ? wraps >= loops || elapsed > timeoutMs : elapsed >= durationMs) break;
    }
  } finally {
    manager.pause(d);
    pp.render = original;
  }
  const records = (dbg.getPerfRecords?.('playback.tick') ?? []).filter((r) => r.dim === d);
  const ticksSrc = records.length > 0 ? records : observed;
  const ticks = ticksSrc.length;
  const span = ticks > 1 ? ticksSrc[ticks - 1].t - ticksSrc[0].t : 0;
  const out = {
    ticks,
    achievedFps: span > 0 ? ((ticks - 1) * 1000) / span : 0,
    renders,
    rendersPerTick: ticks > 0 ? renders / ticks : null,
    lastTimepointShown: sawMax || records.some((r) => Math.abs(r.value - hi) <= eps) ? 1 : 0,
    wraps,
    lodLevelMean: lod.n > 0 ? lod.sum / lod.n : null,
    lodCoarsestFrac: lod.n > 0 ? lod.coarsest / lod.n : null,
    durationMs: performance.now() - t0,
    tickSource: records.length > 0 ? 'records' : 'observed',
  };
  const lateness = records
    .filter((r) => typeof r.due === 'number')
    .map((r) => r.t - r.due)
    .sort((a, b) => a - b);
  if (lateness.length > 0) {
    const q = (p) => lateness[Math.min(lateness.length - 1, Math.floor(p * lateness.length))];
    out.tickLatencyP50Ms = q(0.5);
    out.tickLatencyP90Ms = q(0.9);
  }
  return out;
}

/**
 * Scrub a hidden dimension the way a user does with its slider.
 *
 * - `mode: 'step'`: `steps` single steps of the dimension's step size, each
 *   via `app.setDimensionValue` then waiting for `awaitDimensionUpdate` and
 *   for `getPerf().isSettled` to hold for 3 frames (when the build reports
 *   it). `stepMs` is the median step latency, `stepP90Ms` the p90.
 * - `mode: 'drag'`: `steps` values `intervalMs` apart WITHOUT waiting (a
 *   slider drag), then a settle. `commitsDuringDrag` counts the animation
 *   frames during the drag in which committed geometry changed. "Changed" is
 *   a build-independent signature over every geometry node: its effective
 *   visibility, geometry id,
 *   every attribute's (and the index's) `version`, the drawn count
 *   (`visible*Count` / `instanceCount` / draw range) and the `version` of
 *   every texture uniform on its material (points keep positions in a data
 *   texture). A commit uploads, and an upload bumps a version.
 *
 * The dimension starts at its range minimum; counters are zeroed after that.
 *
 * @param {{ dim?: number, mode?: 'step'|'drag', steps?: number,
 *   intervalMs?: number, settleTimeoutMs?: number }} opts
 */
export async function scrub({
  dim,
  mode = 'step',
  steps = 5,
  intervalMs = 50,
  settleTimeoutMs = 60000,
}) {
  const dbg = window.__luxarDebug;
  const app = dbg.app;
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const dims = app.getDimensions();
  const d = dim ?? dims.metadata.findIndex((_, i) => !dims.displayed.includes(i));
  if (!(d >= 0 && d < dims.ndim)) throw new Error(`scrub: no hidden dimension (dim=${dim})`);
  const [lo, hi] = dims.ranges[d];
  const step = dims.metadata[d]?.step || (hi - lo) / Math.max(1, steps);
  const valueAt = (i) => {
    const n = Math.max(1, Math.round((hi - lo) / step) + 1);
    return lo + (i % n) * step;
  };
  const settled = async () => {
    const t0 = performance.now();
    let streak = 0;
    while (performance.now() - t0 < settleTimeoutMs) {
      await frame();
      const s = dbg.getPerf?.()?.isSettled;
      streak = s === false ? 0 : streak + 1;
      if (streak >= 3) return true;
    }
    return false;
  };
  const signature = () => {
    const parts = [];
    app.sceneManager.scene.traverse((o) => {
      const u = o.userData;
      if (!u?.nodeType || !o.geometry) return;
      const g = o.geometry;
      let shown = true;
      for (let n = o; n; n = n.parent) shown = shown && n.visible;
      let s = `${shown ? 1 : 0}:${g.id}:${g.index?.version ?? '-'}`;
      for (const k in g.attributes) s += `:${g.attributes[k].version}`;
      s += `|${u.visiblePointCount ?? u.visibleSplatCount ?? u.visibleSegmentCount ?? u.committedVertexCount ?? g.instanceCount ?? ''}`;
      s += `|${g.drawRange?.start}-${g.drawRange?.count}`;
      const uniforms = o.material?.uniforms;
      if (uniforms) {
        for (const k in uniforms) {
          const v = uniforms[k]?.value;
          if (v?.isTexture) s += `|${k}:${v.id}:${v.version}`;
        }
      }
      parts.push(s);
    });
    return parts.join(';');
  };

  app.setDimensionValue(d, lo);
  await app.awaitDimensionUpdate();
  await settled();
  dbg.resetPerfCounters?.();
  window.__gateExtReset?.();

  if (mode === 'drag') {
    let commits = 0;
    let sig = signature();
    let live = true;
    let frames = 0;
    const watch = () => {
      if (!live) return;
      frames++;
      const s = signature();
      if (s !== sig) {
        commits++;
        sig = s;
      }
      requestAnimationFrame(watch);
    };
    requestAnimationFrame(watch);
    const t0 = performance.now();
    for (let i = 1; i <= steps; i++) {
      app.setDimensionValue(d, valueAt(i));
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    const dragMs = performance.now() - t0;
    live = false;
    const ok = await settled();
    return { commitsDuringDrag: commits, dragFrames: frames, dragMs, settled: ok ? 1 : 0 };
  }

  const lat = [];
  let allSettled = true;
  for (let i = 1; i <= steps; i++) {
    const t0 = performance.now();
    app.setDimensionValue(d, valueAt(i));
    await app.awaitDimensionUpdate();
    allSettled = (await settled()) && allSettled;
    lat.push(performance.now() - t0);
  }
  const sorted = [...lat].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  return { stepMs: q(0.5), stepP90Ms: q(0.9), steps: lat.length, settled: allSettled ? 1 : 0 };
}

/**
 * Orbit the camera around the current pose's target: `revolutions` turns of
 * `steps` frames each, at `radius` (default: the current horizontal distance)
 * and the current height. One step per animation frame.
 *
 * `levelFlips` counts, build-independently, how often a LOD group's DISPLAYED
 * level changed between frames, polled from
 * `getSceneLoader().getDefaultLoader().lodGroupRegistry.list()` entries'
 * `displayedChildIndex` (null when the registry is unreachable). The viewer's
 * own `lod.levelSwaps` counter, when present, is in the counters.
 *
 * @param {{ radius?: number, steps?: number, revolutions?: number }} opts
 */
export async function orbit({ radius, steps = 120, revolutions = 1 }) {
  const dbg = window.__luxarDebug;
  const app = dbg.app;
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const registry = () => dbg.getSceneLoader?.()?.getDefaultLoader?.()?.lodGroupRegistry ?? null;
  const lodState = () => {
    const reg = registry();
    if (!reg?.list) return null;
    return reg.list().map((e) => `${e.path ?? ''}=${e.displayedChildIndex ?? '-'}`);
  };
  const pose = app.getCameraPose();
  const [tx, ty, tz] = pose.target;
  const dx = pose.position[0] - tx;
  const dz = pose.position[2] - tz;
  const r = radius ?? Math.hypot(dx, dz);
  const height = pose.position[1] - ty;
  const phase = Math.atan2(dz, dx);
  const pp = app.sceneManager.postProcessing;
  let renders = 0;
  const original = pp.render;
  pp.render = function countedRender(...args) {
    renders++;
    return original.apply(this, args);
  };
  let flips = 0;
  let prev = lodState();
  const total = Math.max(1, Math.round(steps * revolutions));
  try {
    for (let i = 1; i <= total; i++) {
      const a = phase + (2 * Math.PI * i) / steps;
      const position = [tx + r * Math.cos(a), ty + height, tz + r * Math.sin(a)];
      const cam = dbg.camera;
      cam.position.fromArray(position);
      cam.up.fromArray(pose.up);
      cam.lookAt(tx, ty, tz);
      cam.updateMatrixWorld();
      app.setCameraPose({ ...app.getCameraPose(), position, target: pose.target, up: pose.up });
      dbg.renderOnce();
      await frame();
      const now = lodState();
      if (prev && now) {
        for (let k = 0; k < Math.min(prev.length, now.length); k++) if (prev[k] !== now[k]) flips++;
      }
      prev = now;
    }
  } finally {
    pp.render = original;
  }
  return { levelFlips: prev === null ? null : flips, renders, frames: total };
}

/**
 * Dolly the camera along its view axis from distance `from` to `to` (from the
 * current target) in `steps` frames. Reports renders and `levelFlips` (see
 * `orbit`).
 *
 * @param {{ from: number, to: number, steps?: number }} opts
 */
export async function zoom({ from, to, steps = 60 }) {
  const dbg = window.__luxarDebug;
  const app = dbg.app;
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const lodState = () => {
    const reg = dbg.getSceneLoader?.()?.getDefaultLoader?.()?.lodGroupRegistry;
    if (!reg?.list) return null;
    return reg.list().map((e) => `${e.path ?? ''}=${e.displayedChildIndex ?? '-'}`);
  };
  const pose = app.getCameraPose();
  const t = pose.target;
  const dir = [0, 1, 2].map((k) => pose.position[k] - t[k]);
  const len = Math.hypot(...dir) || 1;
  const unit = dir.map((v) => v / len);
  const pp = app.sceneManager.postProcessing;
  let renders = 0;
  const original = pp.render;
  pp.render = function countedRender(...args) {
    renders++;
    return original.apply(this, args);
  };
  let flips = 0;
  let prev = lodState();
  try {
    for (let i = 0; i <= steps; i++) {
      const dist = from + ((to - from) * i) / Math.max(1, steps);
      const position = unit.map((u, k) => t[k] + u * dist);
      const cam = dbg.camera;
      cam.position.fromArray(position);
      cam.up.fromArray(pose.up);
      cam.lookAt(t[0], t[1], t[2]);
      cam.updateMatrixWorld();
      app.setCameraPose({ ...app.getCameraPose(), position, target: t, up: pose.up });
      dbg.renderOnce();
      await frame();
      const now = lodState();
      if (prev && now) {
        for (let k = 0; k < Math.min(prev.length, now.length); k++) if (prev[k] !== now[k]) flips++;
      }
      prev = now;
    }
  } finally {
    pp.render = original;
  }
  return { levelFlips: prev === null ? null : flips, renders, frames: steps + 1 };
}

/**
 * A cold load, called on a fresh context right after navigation (the page
 * is at `debugReady`). Nothing is forced: it only watches animation frames.
 *
 * - `firstFrameMs`: the earliest `getPerf().timeline.firstCommit` of any
 *   geometry kind, in ms since navigation start (`performance.now()` origin,
 *   so it includes fetching the viewer itself);
 * - `ttfpMs`: `timeline.measures.ttfpMs` (from the viewer's own `loadStart`);
 * - `settledMs`: first frame, in ms since navigation, at which
 *   `getPerf().isSettled` has held for 3 frames after a first commit.
 *
 * A `pose` is applied at once and again when `sceneLoaded` is first marked,
 * because loading the scene frames the camera on it (no URL parameter sets
 * a pose before the load starts).
 *
 * @param {{ pose?: object, timeoutMs?: number }} opts
 */
export async function coldLoad({ pose, timeoutMs = 300000 }) {
  const dbg = window.__luxarDebug;
  const app = dbg.app;
  const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
  const place = () => {
    if (!pose) return;
    const cam = dbg.camera;
    cam.position.fromArray(pose.position);
    cam.up.fromArray(pose.up);
    cam.lookAt(pose.target[0], pose.target[1], pose.target[2]);
    cam.updateMatrixWorld();
    app.setCameraPose({
      ...app.getCameraPose(),
      position: pose.position,
      target: pose.target,
      up: pose.up,
    });
  };
  place();
  let repositioned = false;
  let streak = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < timeoutMs) {
    await frame();
    const perf = dbg.getPerf?.();
    const tl = perf?.timeline;
    if (!repositioned && tl?.milestones?.sceneLoaded !== undefined) {
      repositioned = true;
      place();
    }
    const commits = Object.values(tl?.firstCommit ?? {});
    streak = commits.length > 0 && perf?.isSettled === true ? streak + 1 : 0;
    if (streak >= 3) {
      return {
        firstFrameMs: Math.min(...commits),
        ttfpMs: tl.measures?.ttfpMs ?? null,
        settledMs: performance.now(),
        settled: 1,
      };
    }
  }
  const tl = dbg.getPerf?.()?.timeline;
  const commits = Object.values(tl?.firstCommit ?? {});
  return {
    firstFrameMs: commits.length ? Math.min(...commits) : null,
    ttfpMs: tl?.measures?.ttfpMs ?? null,
    settledMs: null,
    settled: 0,
  };
}
