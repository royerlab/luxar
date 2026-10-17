import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { motion, rendererInfo } from './page-ops.mjs';

describe('rendererInfo', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('distinguishes WebGPURenderer on WebGL2 from WebGLRenderer', async () => {
    const gl = {
      isContextLost: () => false,
      getExtension: () => null,
      getParameter: () => 'Hardware GPU',
    };
    const renderer = { isWebGPURenderer: true, backend: { isWebGLBackend: true, gl } };
    vi.stubGlobal('window', { __luxarDebug: { app: { sceneManager: { renderer } } } });
    expect((await rendererInfo()).api).toBe('webgpu-gl');
    renderer.isWebGPURenderer = false;
    expect((await rendererInfo()).api).toBe('webgl');
  });

  it('reports native WebGPU and an unrecognized backend without claiming the forced arm', async () => {
    const gl = {
      isContextLost: () => false,
      getExtension: () => null,
      getParameter: () => 'Hardware GPU',
    };
    const renderer = { isWebGPURenderer: true, backend: { isWebGPUBackend: true } };
    vi.stubGlobal('window', { __luxarDebug: { app: { sceneManager: { renderer } } } });
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: async () => ({ info: { vendor: 'Hardware GPU' } }) },
    });
    expect(await rendererInfo()).toEqual({ api: 'webgpu', gpu: 'Hardware GPU', usable: true });
    renderer.backend = { gl };
    expect(await rendererInfo()).toEqual({ api: 'webgl', gpu: 'Hardware GPU', usable: true });
  });
});

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
  const page = {
    clock: 0,
    gpuQueued: 0,
    dirty: false,
    preparedPoses: [],
    rafCalls: 0,
    syncs: 0,
    starts: 0,
  };
  const drain = () => {
    page.syncs++;
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
    animationController: {
      stopAnimation: () => {
        page.dirty = false;
      },
      startAnimation: () => {
        page.starts++;
      },
      prepareFrame: () => {
        page.preparedPoses.push([...pose.position]);
      },
    },
  };
  // The viewer's loop renders in the animation frame, before the harness's
  // own frame callback (registered later) runs.
  page.raf = (cb) => {
    page.rafCalls++;
    return setTimeout(() => {
      page.clock += RAF_MS;
      if (page.dirty) {
        page.dirty = false;
        page.dbg.app.sceneManager.postProcessing.render();
      }
      cb(page.clock);
    }, 0);
  };
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
    it(`frameMs and p95 include GPU completion on ${kind}`, async () => {
      install(kind);
      const mo = await motion({ frames: 12, warm: 3, pose });
      expect(mo.rendersPerFrame).toBe(1);
      // Honest per-frame time: submission + GPU, not the ~0.15 ms submission alone.
      expect(mo.frameMs).toBeGreaterThan(GPU_MS);
      expect(mo.frameMs).toBeLessThan(GPU_MS + 1);
      expect(mo.p95Ms).toBeGreaterThan(GPU_MS);
    });

    it(`measures unpaced motion with one sync and per-step view updates on ${kind}`, async () => {
      install(kind);
      const mo = await motion({ frames: 12, warm: 3, pose, unpaced: true });
      expect(mo.throughputMs).toBeGreaterThan(GPU_MS);
      expect(mo.throughputMs).toBeLessThan(GPU_MS + 1);
      expect(mo.rendersPerFrame).toBe(1);
      expect(page.preparedPoses).toHaveLength(12);
      expect(new Set(page.preparedPoses.map((p) => p.join(','))).size).toBe(12);
      expect(page.rafCalls).toBe(3);
      expect(page.syncs).toBe(5); // warm 3, drain before batch, drain after batch
    });

    it(`pipelined rAF includes the final GPU work on ${kind}`, async () => {
      install(kind);
      const mo = await motion({ frames: 12, warm: 3, pose, pipelined: true });
      expect(mo.frameMs).toBeGreaterThan(GPU_MS);
      expect(mo.rendersPerFrame).toBe(1);
      expect(page.syncs).toBe(3); // warm, initial, final
    });
  }

  it('leaves older baselines usable when prepareFrame is unavailable', async () => {
    install('webgl');
    delete page.dbg.animationController.prepareFrame;
    const mo = await motion({ frames: 4, warm: 1, pose, unpaced: true });
    expect(mo.throughputMs).toBeUndefined();
    expect(page.rafCalls).toBe(0);
  });

  it('rejects combining pipelined and unpaced modes', async () => {
    install('webgl');
    await expect(
      motion({ frames: 4, warm: 1, pose, pipelined: true, unpaced: true })
    ).rejects.toThrow(/pipelined.*unpaced/);
    expect(page.rafCalls).toBe(0);
  });

  it('restores the WebGL framebuffer binding it reads through', async () => {
    install('webgl');
    const gl = page.dbg.app.sceneManager.renderer.backend.gl;
    const bound = [];
    gl.getParameter = (p) => (p === gl.FRAMEBUFFER_BINDING ? 'three-target' : null);
    gl.bindFramebuffer = (_target, fb) => bound.push(fb);
    await motion({ frames: 4, warm: 1, pose });
    expect(bound.length).toBeGreaterThan(0);
    expect(bound.at(-1)).toBe('three-target');
  });

  it('restores the render method and animation after a view callback fails', async () => {
    install('webgl');
    const pp = page.dbg.app.sceneManager.postProcessing;
    const original = pp.render;
    page.dbg.animationController.prepareFrame = () => {
      throw new Error('view callback failed');
    };
    await expect(motion({ frames: 4, warm: 1, pose, unpaced: true })).rejects.toThrow(
      'view callback failed'
    );
    expect(pp.render).toBe(original);
    expect(page.starts).toBe(1);
  });
});
