import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { motion } from './page-ops.mjs';

/**
 * A fake page whose GPU runs behind its CPU, the way headless Chrome with
 * vsync off does: a render SUBMITS in `SUBMIT_MS` of CPU and queues
 * `GPU_MS` of GPU work, and an animation frame never waits for that queue.
 * Only a GPU sync (a WebGL `readPixels`, or WebGPU's
 * `queue.onSubmittedWorkDone()`) drains it, advancing the clock.
 */
const SUBMIT_MS = 0.1;
const GPU_MS = 20;
const RAF_MS = 0.05;

function fakePage(kind) {
  const page = { clock: 0, gpuQueued: 0, dirty: false };
  const drain = () => {
    page.clock += page.gpuQueued;
    page.gpuQueued = 0;
  };
  const gl = {
    FRAMEBUFFER: 1,
    FRAMEBUFFER_BINDING: 2,
    RGBA: 3,
    UNSIGNED_BYTE: 4,
    getParameter: () => null,
    bindFramebuffer: () => {},
    readPixels: drain,
  };
  const backend =
    kind === 'webgpu'
      ? {
          isWebGPUBackend: true,
          // Resolves later, like the real promise: the wait is not script time.
          device: { queue: { onSubmittedWorkDone: () => Promise.resolve().then(drain) } },
        }
      : { gl };
  const postProcessing = {
    render() {
      page.clock += SUBMIT_MS;
      page.gpuQueued += GPU_MS;
    },
  };
  const vec = () => ({ fromArray() {} });
  let pose = { position: [0, 0, 5], target: [0, 0, 0], up: [0, 1, 0] };
  page.dbg = {
    camera: { position: vec(), up: vec(), lookAt() {}, updateMatrixWorld() {} },
    app: {
      sceneManager: { postProcessing, renderer: { backend } },
      getCameraPose: () => pose,
      setCameraPose: (p) => {
        pose = p;
      },
    },
    renderOnce: () => {
      page.dirty = true;
    },
  };
  // The viewer's loop renders in the animation frame, before the harness's
  // own frame callback (registered later) runs.
  page.raf = (cb) =>
    setTimeout(() => {
      page.clock += RAF_MS;
      if (page.dirty) {
        page.dirty = false;
        page.dbg.app.sceneManager.postProcessing.render();
      }
      cb(page.clock);
    }, 0);
  return page;
}

describe('motion frame timing', () => {
  let page;
  const install = (kind) => {
    page = fakePage(kind);
    vi.stubGlobal('window', { __luxarDebug: page.dbg });
    vi.stubGlobal('requestAnimationFrame', page.raf);
    vi.spyOn(performance, 'now').mockImplementation(() => page.clock);
  };
  beforeEach(() => {
    page = null;
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  const pose = { position: [0, 0, 5], target: [0, 0, 0], up: [0, 1, 0] };

  for (const kind of ['webgl', 'webgpu']) {
    it.fails(`frameMs and p95 include GPU completion on ${kind}`, async () => {
      install(kind);
      const mo = await motion({ frames: 12, warm: 3, pose });
      expect(mo.rendersPerFrame).toBe(1);
      // Honest per-frame time: submission + GPU, not the ~0.15 ms submission alone.
      expect(mo.frameMs).toBeGreaterThan(GPU_MS);
      expect(mo.frameMs).toBeLessThan(GPU_MS + 1);
      expect(mo.p95Ms).toBeGreaterThan(GPU_MS);
    });
  }

  it.fails('restores the WebGL framebuffer binding it reads through', async () => {
    install('webgl');
    const gl = page.dbg.app.sceneManager.renderer.backend.gl;
    const bound = [];
    gl.getParameter = (p) => (p === gl.FRAMEBUFFER_BINDING ? 'three-target' : null);
    gl.bindFramebuffer = (_target, fb) => bound.push(fb);
    await motion({ frames: 4, warm: 1, pose });
    expect(bound.length).toBeGreaterThan(0);
    expect(bound.at(-1)).toBe('three-target');
  });
});
