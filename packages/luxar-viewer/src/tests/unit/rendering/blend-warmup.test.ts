import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { BLENDING_MODES, type BlendingMode } from '../../../types/blending';
import {
  BlendWarmupManager,
  buildBlendWarmupFingerprint,
  type WarmupRenderer,
} from '../../../rendering/blend-warmup';

class FakeWarmupMaterial extends THREE.ShaderMaterial {
  declare defines: Record<string, unknown>;

  constructor(mode: BlendingMode = 'additive', extraDefines: Record<string, unknown> = {}) {
    super({
      vertexShader: 'void main() { gl_Position = vec4( position, 1.0 ); }',
      fragmentShader: 'void main() { gl_FragColor = vec4( 1.0 ); }',
    });
    this.defines = { ...extraDefines };
    this.userData = { blendingMode: mode };
    this.applyBlendingMode(mode);
  }

  applyBlendingMode(mode: BlendingMode): void {
    delete this.defines.LUXAR_MAX_RGB_CONTRIBUTION;
    delete this.defines.LUXAR_VOLUMETRIC;
    delete this.defines.LUXAR_NORMAL_PREMULT;
    delete this.defines.LUXAR_MESH_ALPHA_CUTOUT;

    if (mode === 'max') this.defines.LUXAR_MAX_RGB_CONTRIBUTION = '';
    if (mode === 'volumetric') this.defines.LUXAR_VOLUMETRIC = '';

    this.userData.blendingMode = mode;
  }

  clone(): this {
    const cloned = new FakeWarmupMaterial('additive');
    cloned.defines = { ...this.defines };
    cloned.userData = { ...this.userData };
    return cloned as this;
  }
}

function makeRenderableMesh(
  nodeType: 'points' | 'lines' | 'gsplats' | 'mesh',
  material: THREE.Material = new FakeWarmupMaterial(),
  visibleCount: number = 8
): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3)
  );
  geometry.setIndex([0, 1, 2]);

  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.nodeType = nodeType;
  if (nodeType === 'points') mesh.userData.visiblePointCount = visibleCount;
  if (nodeType === 'lines') mesh.userData.visibleSegmentCount = visibleCount;
  if (nodeType === 'gsplats') mesh.userData.visibleSplatCount = visibleCount;
  if (nodeType === 'mesh') mesh.userData.visibleTriangleCount = visibleCount;
  return mesh;
}

function makeWarmupTurnController() {
  const waiters: Array<() => void> = [];

  return {
    get pending(): number {
      return waiters.length;
    },
    wait: () =>
      new Promise<void>((resolve) => {
        waiters.push(resolve);
      }),
    async releaseNext(): Promise<void> {
      const resolve = waiters.shift();
      resolve?.();
      for (let i = 0; i < 8; i++) await Promise.resolve();
    },
  };
}

function makeActivationController() {
  const activations: Array<() => void> = [];

  return {
    get pending(): number {
      return activations.length;
    },
    defer: (activate: () => void) => {
      activations.push(activate);
    },
    releaseNext(): void {
      activations.shift()?.();
    },
  };
}

const activateImmediately = (activate: () => void): void => activate();

describe('BlendWarmupManager', () => {
  it('compiles WebGPU keepers against the live HDR target, restores the prior target, and awaits completion', async () => {
    const previous = new THREE.WebGLRenderTarget(1, 1);
    const hdr = new THREE.WebGLRenderTarget(1, 1);
    const resizedHdr = new THREE.WebGLRenderTarget(2, 2);
    let liveHdr = hdr;
    const compiledTargets: Array<THREE.RenderTarget | null> = [];
    let current: THREE.RenderTarget | null = previous;
    let finish!: () => void;
    const compileAsync = vi.fn((scene: THREE.Object3D) => {
      expect(current).toBe(liveHdr);
      compiledTargets.push(current);
      expect(scene.children).toHaveLength(1);
      if (compileAsync.mock.calls.length > 1) return Promise.resolve();
      return new Promise<void>((resolve) => {
        finish = resolve;
      });
    });
    const renderer = {
      getRenderTarget: () => current,
      setRenderTarget: (target: THREE.RenderTarget | null) => {
        current = target;
      },
      compileAsync,
    } as WarmupRenderer;
    const manager = new BlendWarmupManager(() => Promise.resolve(), undefined, activateImmediately);
    const root = new THREE.Group();
    root.add(makeRenderableMesh('points'));
    manager.configure({
      enabled: true,
      renderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
      pipelineVariants: true,
      getRenderTarget: () => liveHdr,
    });

    const done = manager.warmScene(root);
    await Promise.resolve();
    await Promise.resolve();
    expect(compileAsync).toHaveBeenCalledTimes(1);
    expect(current).toBe(previous);
    const compileScene = compileAsync.mock.calls[0][0];
    expect(compileScene.children).toHaveLength(1);
    liveHdr = resizedHdr;
    finish();
    await done;
    expect(compileScene.children).toHaveLength(0);
    expect(compileAsync).toHaveBeenCalledTimes(BLENDING_MODES.length);
    expect(compiledTargets).toEqual([hdr, ...Array(BLENDING_MODES.length - 1).fill(resizedHdr)]);
    expect(current).toBe(previous);
  });

  it('uses WebGL compile and detaches a rejected WebGPU compile object', async () => {
    const compile = vi.fn((_scene: THREE.Scene) => new Set<THREE.Material>());
    const webgl = {
      isWebGLRenderer: true,
      compile,
      getContext: () => ({ isContextLost: () => false }),
    } as unknown as WarmupRenderer;
    const manager = new BlendWarmupManager(() => Promise.resolve(), undefined, activateImmediately);
    const root = new THREE.Group();
    root.add(makeRenderableMesh('points'));
    manager.configure({
      enabled: true,
      renderer: webgl,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    await manager.warmScene(root);
    expect(compile).toHaveBeenCalledTimes(3);
    expect((compile.mock.calls[0][0] as THREE.Scene).children).toHaveLength(0);

    let compileScene!: THREE.Scene;
    const rejecting = {
      compileAsync: vi.fn((scene: THREE.Scene) => {
        compileScene = scene;
        return Promise.reject(new Error('compile failed'));
      }),
      getRenderTarget: () => null,
      setRenderTarget: vi.fn(),
    } as unknown as WarmupRenderer;
    manager.configure({
      enabled: true,
      renderer: rejecting,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    await manager.warmScene(root);
    expect(rejecting.compileAsync).toHaveBeenCalled();
    expect(compileScene.children).toHaveLength(0);
  });

  it('compiles WebGL keepers against the live target and restores the host target', async () => {
    const previous = new THREE.WebGLRenderTarget(1, 1);
    const first = new THREE.WebGLRenderTarget(2, 2);
    const resized = new THREE.WebGLRenderTarget(3, 3);
    let live = first;
    let current: THREE.RenderTarget | null = previous;
    const compiledTargets: Array<THREE.RenderTarget | null> = [];
    const renderer = {
      isWebGLRenderer: true,
      getRenderTarget: () => current,
      setRenderTarget: (target: THREE.RenderTarget | null) => {
        current = target;
      },
      getContext: () => ({ isContextLost: () => false }),
      compile: () => {
        compiledTargets.push(current);
        live = resized;
        return new Set<THREE.Material>();
      },
    } as unknown as WarmupRenderer;
    const manager = new BlendWarmupManager(() => Promise.resolve(), undefined, activateImmediately);
    manager.configure({
      enabled: true,
      renderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
      getRenderTarget: () => live,
    });

    await manager.warmScene(makeRenderableMesh('points'));
    expect(compiledTargets).toHaveLength(3);
    expect(compiledTargets[0]).toBe(first);
    expect(compiledTargets[1]).toBe(resized);
    expect(compiledTargets[2]).toBe(resized);
    expect(current).toBe(previous);
  });

  it('restores the host target when a WebGL compile throws', async () => {
    const previous = new THREE.WebGLRenderTarget(1, 1);
    const live = new THREE.WebGLRenderTarget(2, 2);
    let current: THREE.RenderTarget | null = previous;
    const compiledTargets: Array<THREE.RenderTarget | null> = [];
    const renderer = {
      isWebGLRenderer: true,
      getRenderTarget: () => current,
      setRenderTarget: (target: THREE.RenderTarget | null) => {
        current = target;
      },
      getContext: () => ({ isContextLost: () => false }),
      compile: () => {
        compiledTargets.push(current);
        throw new Error('compile failed');
      },
    } as unknown as WarmupRenderer;
    const manager = new BlendWarmupManager(() => Promise.resolve(), undefined, activateImmediately);
    manager.configure({
      enabled: true,
      renderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
      getRenderTarget: () => live,
    });

    await manager.warmScene(makeRenderableMesh('points'));
    expect(compiledTargets).toHaveLength(3);
    for (const target of compiledTargets) expect(target).toBe(live);
    expect(current).toBe(previous);
    expect(manager.getStats().compiled).toBe(0);
  });

  it.each(['canvas', 'XR target'] as const)(
    'keeps the %s program without drawing into a differently keyed target',
    async (output) => {
      const liveOutput =
        output === 'canvas'
          ? null
          : Object.assign(new THREE.WebGLRenderTarget(2, 2), { isXRRenderTarget: true });
      if (liveOutput) liveOutput.texture.colorSpace = THREE.SRGBColorSpace;
      const previous = new THREE.WebGLRenderTarget(1, 1);
      let current: THREE.RenderTarget | null = previous;
      const programs = new Map<THREE.Material, { currentProgram: { isReady(): boolean } }>();
      const compiled: Array<{ material: THREE.Material; program: { isReady(): boolean } }> = [];
      const render = vi.fn((scene: THREE.Scene) => {
        const material = (scene.children[0] as THREE.Mesh).material as THREE.Material;
        programs.get(material)!.currentProgram = { isReady: () => true };
      });
      const renderer = {
        isWebGLRenderer: true,
        outputColorSpace: THREE.SRGBColorSpace,
        getRenderTarget: () => current,
        setRenderTarget: (target: THREE.RenderTarget | null) => {
          current = target;
        },
        getContext: () => ({ isContextLost: () => false }),
        compile: (scene: THREE.Scene) => {
          expect(current).toBe(liveOutput);
          const material = (scene.children[0] as THREE.Mesh).material as THREE.Material;
          const program = { isReady: () => true };
          programs.set(material, { currentProgram: program });
          compiled.push({ material, program });
          return new Set([material]);
        },
        properties: { get: (material: THREE.Material) => programs.get(material) },
        getViewport: () => new THREE.Vector4(),
        getScissor: () => new THREE.Vector4(),
        getScissorTest: () => false,
        setViewport: () => undefined,
        setScissor: () => undefined,
        setScissorTest: () => undefined,
        info: { render: { calls: 0, triangles: 0, points: 0, lines: 0 } },
        render,
      } as unknown as WarmupRenderer;
      const manager = new BlendWarmupManager(
        () => Promise.resolve(),
        undefined,
        activateImmediately
      );
      manager.configure({
        enabled: true,
        renderer,
        camera: new THREE.PerspectiveCamera(),
        targetScene: new THREE.Scene(),
        getRenderTarget: () => liveOutput,
      });

      await manager.warmScene(makeRenderableMesh('points'));
      expect(compiled).toHaveLength(3);
      for (const { material, program } of compiled) {
        expect(programs.get(material)?.currentProgram).toBe(program);
      }
      expect(render).not.toHaveBeenCalled();
      expect(current).toBe(previous);
    }
  );

  it('pipelineVariants (WebGPU) queues every blend mode, not only define-distinct ones', async () => {
    const run = async (pipelineVariants: boolean) => {
      const compileOne = vi.fn();
      const manager = new BlendWarmupManager(
        () => Promise.resolve(),
        compileOne,
        activateImmediately
      );
      const root = new THREE.Group();
      root.add(makeRenderableMesh('points'));
      manager.configure({
        enabled: true,
        renderer: {} as THREE.WebGLRenderer,
        camera: new THREE.PerspectiveCamera(),
        targetScene: new THREE.Scene(),
        pipelineVariants,
      });
      await manager.warmScene(root);
      return compileOne.mock.calls.length;
    };
    // FakeWarmupMaterial: only max and volumetric change defines, so WebGL
    // collapses the other modes onto one program.
    expect(await run(false)).toBe(3);
    expect(await run(true)).toBe(BLENDING_MODES.length);
  });

  it('awaits an asynchronous compile (WebGPU compileAsync) before the next one', async () => {
    let finish!: () => void;
    let inFlight = 0;
    let maxInFlight = 0;
    const compileOne = vi.fn(async () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      if (!finish) await new Promise<void>((r) => (finish = r));
      inFlight--;
    });
    const manager = new BlendWarmupManager(
      () => Promise.resolve(),
      compileOne,
      activateImmediately
    );
    const root = new THREE.Group();
    root.add(makeRenderableMesh('points'));
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
      pipelineVariants: true,
    });
    const done = manager.warmScene(root);
    await Promise.resolve();
    await Promise.resolve();
    expect(compileOne).toHaveBeenCalledTimes(1);
    finish();
    await done;
    expect(maxInFlight).toBe(1);
    expect(compileOne).toHaveBeenCalledTimes(BLENDING_MODES.length);
  });

  it('defers explicit scene activation and ignores scheduling until it runs', async () => {
    const warmupTurns = makeWarmupTurnController();
    const activations = makeActivationController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activations.defer);
    const root = new THREE.Group();
    const mesh = makeRenderableMesh('points');
    root.add(mesh);

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    manager.scheduleObject(mesh);
    expect(compileOne).not.toHaveBeenCalled();

    void manager.warmScene(root);
    manager.scheduleObject(makeRenderableMesh('points'));
    expect(activations.pending).toBe(1);
    expect(warmupTurns.pending).toBe(0);

    activations.releaseNext();
    expect(warmupTurns.pending).toBe(1);
    await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(1);
  });

  it('clear cancels pending activation, resolves it, and keeps scheduling disarmed', async () => {
    const warmupTurns = makeWarmupTurnController();
    const activations = makeActivationController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activations.defer);
    const root = new THREE.Group();
    root.add(makeRenderableMesh('points'));

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    const completion = manager.warmScene(root);
    manager.clear();
    await completion;

    activations.releaseNext();
    manager.scheduleObject(makeRenderableMesh('points'));

    expect(warmupTurns.pending).toBe(0);
    expect(compileOne).not.toHaveBeenCalled();
  });

  it('waits for an idle callback after the rendered frame', async () => {
    vi.useFakeTimers();
    const frameCallbacks: FrameRequestCallback[] = [];
    const idleCallbacks: Array<{
      callback: () => void;
      options?: { timeout: number };
    }> = [];
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frameCallbacks.push(callback);
      return frameCallbacks.length;
    });
    vi.stubGlobal('requestIdleCallback', (callback: () => void, options?: { timeout: number }) => {
      idleCallbacks.push({ callback, options });
      return idleCallbacks.length;
    });

    try {
      const compileOne = vi.fn();
      const manager = new BlendWarmupManager(undefined, compileOne);
      manager.configure({
        enabled: true,
        renderer: {} as THREE.WebGLRenderer,
        camera: new THREE.PerspectiveCamera(),
        targetScene: new THREE.Scene(),
      });

      manager.warmScene(makeRenderableMesh('points'));
      expect(compileOne).not.toHaveBeenCalled();
      expect(frameCallbacks).toHaveLength(0);

      await vi.advanceTimersByTimeAsync(0);
      expect(frameCallbacks).toHaveLength(1);

      frameCallbacks.shift()?.(0);
      await Promise.resolve();
      expect(compileOne).not.toHaveBeenCalled();
      expect(idleCallbacks).toHaveLength(1);
      expect(idleCallbacks[0]?.options).toEqual({ timeout: 250 });

      await vi.advanceTimersByTimeAsync(0);
      expect(compileOne).not.toHaveBeenCalled();

      idleCallbacks.shift()?.callback();
      await Promise.resolve();
      await Promise.resolve();
      expect(compileOne).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });

  it('uses a nonzero delay when idle callbacks are unavailable', async () => {
    vi.useFakeTimers();
    const frameCallbacks: FrameRequestCallback[] = [];
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frameCallbacks.push(callback);
      return frameCallbacks.length;
    });
    vi.stubGlobal('requestIdleCallback', undefined);

    try {
      const compileOne = vi.fn();
      const manager = new BlendWarmupManager(undefined, compileOne);
      manager.configure({
        enabled: true,
        renderer: {} as THREE.WebGLRenderer,
        camera: new THREE.PerspectiveCamera(),
        targetScene: new THREE.Scene(),
      });

      manager.warmScene(makeRenderableMesh('points'));
      await vi.advanceTimersByTimeAsync(0);
      frameCallbacks.shift()?.(0);

      await vi.advanceTimersByTimeAsync(0);
      expect(compileOne).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(49);
      expect(compileOne).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1);
      expect(compileOne).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });

  it('waits for a render turn before each distinct variant compile', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compiledModes: string[] = [];
    const manager = new BlendWarmupManager(
      warmupTurns.wait,
      (_renderer, _camera, _scene, obj) => {
        compiledModes.push(
          String(
            (
              (obj as THREE.Mesh).material as THREE.Material & {
                userData: { blendingMode?: string };
              }
            ).userData.blendingMode
          )
        );
      },
      activateImmediately
    );

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    manager.warmScene(makeRenderableMesh('points'));
    expect(compiledModes).toEqual([]);

    await warmupTurns.releaseNext();
    expect(compiledModes).toEqual(['additive']);

    await warmupTurns.releaseNext();
    expect(compiledModes).toEqual(['additive', 'volumetric']);

    await warmupTurns.releaseNext();
    expect(compiledModes).toEqual(['additive', 'volumetric', 'max']);

    await warmupTurns.releaseNext();
    expect(compiledModes).toHaveLength(3);
  });

  it('resolves scene warming only after every queued variant compiles', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    let resolved = false;
    const completion = manager.warmScene(makeRenderableMesh('points')).then(() => {
      resolved = true;
    });

    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();
    expect(resolved).toBe(false);

    await warmupTurns.releaseNext();
    await completion;
    expect(compileOne).toHaveBeenCalledTimes(3);
    expect(resolved).toBe(true);
  });

  it('counts a variant only after the driver finishes linking it', async () => {
    const warmupTurns = makeWarmupTurnController();
    const finishLinks: Array<() => void> = [];
    const compileOne = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishLinks.push(resolve);
        })
    );
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    let ready = false;
    const completion = manager.warmScene(makeRenderableMesh('points')).then(() => {
      ready = true;
    });
    await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(1);
    expect(manager.getStats().compiled).toBe(0);
    expect(warmupTurns.pending).toBe(0);
    expect(ready).toBe(false);

    finishLinks[0]?.();
    await vi.waitFor(() => expect(manager.getStats().compiled).toBe(1));
    expect(manager.getStats().compiled).toBe(1);
    expect(warmupTurns.pending).toBe(1);
    await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(2);
    finishLinks[1]?.();
    await vi.waitFor(() => expect(warmupTurns.pending).toBe(1));
    await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(3);
    finishLinks[2]?.();
    await completion;
    expect(manager.getStats().compiled).toBe(3);
    expect(ready).toBe(true);
  });

  it('waits for Three.js program readiness without polling after context loss', async () => {
    let ready = false;
    let lost = false;
    const isReady = vi.fn(() => ready);
    const compile = vi.fn(
      (scene: THREE.Object3D) =>
        new Set([(scene.children[0] as THREE.Mesh).material as THREE.Material])
    );
    const renderer = {
      compile,
      getContext: () => ({ isContextLost: () => lost }),
      properties: { get: () => ({ currentProgram: { isReady } }) },
    } as unknown as THREE.WebGLRenderer;
    const manager = new BlendWarmupManager(async () => undefined, undefined, activateImmediately);
    manager.configure({
      enabled: true,
      renderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const completion = manager.warmScene(makeRenderableMesh('points'));
    await vi.waitFor(() => expect(compile).toHaveBeenCalledTimes(1));
    expect(manager.getStats().compiled).toBe(0);
    ready = true;
    await vi.waitFor(() => expect(compile).toHaveBeenCalledTimes(3));
    await completion;
    expect(manager.getStats()).toMatchObject({ queued: 3, compiled: 3 });
    ready = false;
    const next = manager.warmScene(
      makeRenderableMesh('points', new FakeWarmupMaterial('normal', { OTHER: 1 }))
    );
    await vi.waitFor(() => expect(compile).toHaveBeenCalledTimes(4));
    lost = true;
    await next;
    const polls = isReady.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(isReady).toHaveBeenCalledTimes(polls);
    expect(manager.getStats().compiled).toBe(3);
    await manager.warmScene(
      makeRenderableMesh('points', new FakeWarmupMaterial('normal', { OTHER: 2 }))
    );
    expect(compile).toHaveBeenCalledTimes(4);
  });

  it('starts a new warm-up after clear while an old link is pending', async () => {
    const finishLinks: Array<() => void> = [];
    const compileOne = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishLinks.push(resolve);
        })
    );
    const manager = new BlendWarmupManager(async () => undefined, compileOne, activateImmediately);
    const config = {
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    };
    manager.configure(config);
    const first = manager.warmScene(makeRenderableMesh('points'));
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(1));

    manager.configure(config);
    await first;
    const second = manager.warmScene(makeRenderableMesh('points'));
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(2));
    expect(manager.getStats().compiled).toBe(0);
    finishLinks[0]?.();
    expect(manager.getStats().compiled).toBe(0);
    finishLinks[1]?.();
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(3));
    finishLinks[2]?.();
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(4));
    finishLinks[3]?.();
    await second;
    expect(manager.getStats().compiled).toBe(3);
  });

  it('keeps a removed node keeper alive until its link settles and warms the next node', async () => {
    const finishLinks: Array<() => void> = [];
    const compileOne = vi.fn(
      (
        _renderer: WarmupRenderer,
        _camera: THREE.Camera,
        _scene: THREE.Scene,
        _object: THREE.Object3D
      ) => new Promise<void>((resolve) => finishLinks.push(resolve))
    );
    const manager = new BlendWarmupManager(async () => undefined, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    const root = new THREE.Group();
    const first = makeRenderableMesh('points');
    root.add(first);
    const completion = manager.warmScene(root);
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(1));
    const keeper = (compileOne.mock.calls[0]?.[3] as THREE.Mesh).material as THREE.Material;
    const disposed = vi.fn();
    keeper.addEventListener('dispose', disposed);

    root.remove(first);
    expect(disposed).not.toHaveBeenCalled();
    manager.scheduleObject(
      makeRenderableMesh('points', new FakeWarmupMaterial('normal', { OTHER: 1 }))
    );
    finishLinks[0]?.();
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(2));
    expect(disposed).toHaveBeenCalledTimes(1);
    for (let index = 1; index < 4; index++) {
      finishLinks[index]?.();
      if (index < 3) await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(index + 2));
    }
    await completion;
    expect(manager.getStats().compiled).toBe(3);
  });

  it('defers disposal of a pending keeper across clear', async () => {
    let finishLink: (() => void) | undefined;
    const compileOne = vi.fn(
      (
        _renderer: WarmupRenderer,
        _camera: THREE.Camera,
        _scene: THREE.Scene,
        _object: THREE.Object3D
      ) =>
        new Promise<void>((resolve) => {
          finishLink = resolve;
        })
    );
    const manager = new BlendWarmupManager(async () => undefined, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    const completion = manager.warmScene(makeRenderableMesh('points'));
    await vi.waitFor(() => expect(compileOne).toHaveBeenCalledTimes(1));
    const keeper = (compileOne.mock.calls[0]?.[3] as THREE.Mesh).material as THREE.Material;
    const disposed = vi.fn();
    keeper.addEventListener('dispose', disposed);
    manager.clear();
    await completion;
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(disposed).not.toHaveBeenCalled();
    finishLink?.();
    await vi.waitFor(() => expect(disposed).toHaveBeenCalledTimes(1));
    expect(manager.getStats().compiled).toBe(0);
  });

  it('stops the default readiness poll when clearing a scene', async () => {
    const isReady = vi.fn(() => false);
    const compile = vi.fn(
      (scene: THREE.Object3D) =>
        new Set([(scene.children[0] as THREE.Mesh).material as THREE.Material])
    );
    const renderer = {
      compile,
      getContext: () => ({ isContextLost: () => false }),
      properties: { get: () => ({ currentProgram: { isReady } }) },
    } as unknown as THREE.WebGLRenderer;
    const manager = new BlendWarmupManager(async () => undefined, undefined, activateImmediately);
    manager.configure({
      enabled: true,
      renderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    const completion = manager.warmScene(makeRenderableMesh('points'));
    await vi.waitFor(() => expect(isReady).toHaveBeenCalled());
    const keeper = ((compile.mock.calls[0]?.[0] as THREE.Scene).children[0] as THREE.Mesh)
      .material as THREE.Material;
    const disposed = vi.fn();
    keeper.addEventListener('dispose', disposed);
    manager.clear();
    await completion;
    const polls = isReady.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(isReady).toHaveBeenCalledTimes(polls);
    expect(disposed).toHaveBeenCalledTimes(1);
    expect(manager.getStats().compiled).toBe(0);
  });

  it('continues with a later node when a removed node link never settles', async () => {
    vi.useFakeTimers();
    try {
      let finishOld: (() => void) | undefined;
      const compileOne = vi
        .fn(
          (
            _renderer: WarmupRenderer,
            _camera: THREE.Camera,
            _scene: THREE.Scene,
            _object: THREE.Object3D
          ): Promise<void> | void => undefined
        )
        .mockImplementationOnce(
          () =>
            new Promise<void>((resolve) => {
              finishOld = resolve;
            })
        )
        .mockImplementation(() => undefined);
      const manager = new BlendWarmupManager(
        async () => undefined,
        compileOne,
        activateImmediately
      );
      manager.configure({
        enabled: true,
        renderer: {} as THREE.WebGLRenderer,
        camera: new THREE.PerspectiveCamera(),
        targetScene: new THREE.Scene(),
      });
      const root = new THREE.Group();
      const first = makeRenderableMesh('points');
      root.add(first);
      void manager.warmScene(root);
      for (let i = 0; i < 8; i++) await Promise.resolve();
      expect(compileOne).toHaveBeenCalledTimes(1);
      const keeper = (compileOne.mock.calls[0]?.[3] as THREE.Mesh).material as THREE.Material;
      const disposed = vi.fn();
      keeper.addEventListener('dispose', disposed);
      root.remove(first);
      manager.scheduleObject(
        makeRenderableMesh('points', new FakeWarmupMaterial('normal', { OTHER: 1 }))
      );
      await vi.advanceTimersByTimeAsync(5000);
      expect(disposed).not.toHaveBeenCalled();
      expect(compileOne).toHaveBeenCalledTimes(4);
      expect(manager.getStats().compiled).toBe(3);
      finishOld?.();
      for (let i = 0; i < 8; i++) await Promise.resolve();
      expect(disposed).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('releases scene warming on its readiness budget and keeps draining after', async () => {
    vi.useFakeTimers();
    try {
      const warmupTurns = makeWarmupTurnController();
      const compileOne = vi.fn();
      const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
      manager.configure({
        enabled: true,
        renderer: {} as THREE.WebGLRenderer,
        camera: new THREE.PerspectiveCamera(),
        targetScene: new THREE.Scene(),
      });

      let resolved = false;
      // A turn that never comes — the background-tab case, where
      // requestAnimationFrame is never serviced.
      const completion = manager.warmScene(makeRenderableMesh('points')).then(() => {
        resolved = true;
      });

      await vi.advanceTimersByTimeAsync(4999);
      expect(resolved).toBe(false);

      await vi.advanceTimersByTimeAsync(1);
      await completion;
      expect(resolved).toBe(true);
      expect(compileOne).not.toHaveBeenCalled();

      // Readiness was released, not cancelled: the queue is still live.
      await warmupTurns.releaseNext();
      expect(compileOne).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('dedupes blend modes that land on the same program variant', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    manager.warmScene(makeRenderableMesh('points'));
    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();

    expect(compileOne).toHaveBeenCalledTimes(3);
  });

  it('shares one keeper per program variant across nodes with identical materials', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    // Two nodes, two DISTINCT material instances with identical compile-time
    // state (the node factory hands every points node the same config): they
    // compile to the same programs, so the second node must queue nothing.
    const root = new THREE.Group();
    root.add(makeRenderableMesh('points', new FakeWarmupMaterial()));
    root.add(makeRenderableMesh('points', new FakeWarmupMaterial()));
    void manager.warmScene(root);
    expect(warmupTurns.pending).toBe(1);
    for (let i = 0; i < 3; i++) await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(3);
    expect(warmupTurns.pending).toBe(0);
    expect(manager.getStats()).toMatchObject({
      queued: 3,
      compiled: 3,
      dedupedAcrossSources: 3,
      ownershipTransfers: 0,
    });
  });

  it('does not transfer shared variants while clearing every source', async () => {
    const warmupTurns = makeWarmupTurnController();
    const manager = new BlendWarmupManager(warmupTurns.wait, vi.fn(), activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const root = new THREE.Group();
    root.add(makeRenderableMesh('points', new FakeWarmupMaterial()));
    root.add(makeRenderableMesh('points', new FakeWarmupMaterial()));
    void manager.warmScene(root);
    for (let i = 0; i < 3; i++) await warmupTurns.releaseNext();

    manager.clear();

    expect(warmupTurns.pending).toBe(0);
    expect(manager.getStats().ownershipTransfers).toBe(0);
  });

  it('hands a shared variant to a surviving node when its owner is released', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const owner = new FakeWarmupMaterial();
    const root = new THREE.Group();
    root.add(makeRenderableMesh('points', owner));
    root.add(makeRenderableMesh('points', new FakeWarmupMaterial()));
    void manager.warmScene(root);
    for (let i = 0; i < 3; i++) await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(3);

    // The owner's node is torn down: its keepers are disposed (the programs
    // would be released with them), so the survivor must re-pin every shared
    // variant with its own keeper — otherwise the warm-up silently evaporates
    // for the node that is still on screen.
    owner.dispose();
    expect(warmupTurns.pending).toBe(1);
    for (let i = 0; i < 3; i++) await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(6);
    expect(manager.getStats()).toMatchObject({ queued: 6, compiled: 6, ownershipTransfers: 3 });
  });

  it('a variant whose last user is released is forgotten, not transferred', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    const only = new FakeWarmupMaterial();
    void manager.warmScene(makeRenderableMesh('points', only));
    for (let i = 0; i < 3; i++) await warmupTurns.releaseNext();
    only.dispose();
    expect(warmupTurns.pending).toBe(0);
    expect(manager.getStats().ownershipTransfers).toBe(0);
    // A NEW node with the same material state owns the variants afresh.
    manager.scheduleObject(makeRenderableMesh('points', new FakeWarmupMaterial()));
    expect(warmupTurns.pending).toBe(1);
  });

  it('drops queued keepers when the source material is disposed', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const material = new FakeWarmupMaterial();
    const mesh = makeRenderableMesh('points', material);
    manager.warmScene(mesh);
    expect(compileOne).not.toHaveBeenCalled();

    material.dispose();
    await warmupTurns.releaseNext();

    expect(compileOne).not.toHaveBeenCalled();
  });

  it('drops old keepers when an object switches to a replacement material', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const mesh = makeRenderableMesh('points', new FakeWarmupMaterial());
    manager.warmScene(mesh);
    expect(compileOne).not.toHaveBeenCalled();

    mesh.material = new FakeWarmupMaterial('normal', { POINT_WIDTH: 32 });
    manager.scheduleObject(mesh);
    expect(compileOne).not.toHaveBeenCalled();

    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();

    expect(compileOne).toHaveBeenCalledTimes(3);
  });

  it('drops queued keepers when a tracked object is removed from the scene', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const root = new THREE.Group();
    const mesh = makeRenderableMesh('points');
    root.add(mesh);

    void manager.warmScene(root);
    expect(compileOne).not.toHaveBeenCalled();

    root.remove(mesh);
    await warmupTurns.releaseNext();

    expect(compileOne).not.toHaveBeenCalled();
  });

  it('resolves immediately for disabled sessions and empty placeholder nodes', async () => {
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(async () => undefined, compileOne, activateImmediately);

    manager.configure({
      enabled: false,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    await manager.warmScene(makeRenderableMesh('points'));
    expect(compileOne).not.toHaveBeenCalled();

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });
    await manager.warmScene(makeRenderableMesh('points', new FakeWarmupMaterial(), 0));
    expect(compileOne).not.toHaveBeenCalled();
  });

  it('preserves the source material state and refreshes on non-blend define changes', async () => {
    const warmupTurns = makeWarmupTurnController();
    const compileOne = vi.fn();
    const manager = new BlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

    manager.configure({
      enabled: true,
      renderer: {} as THREE.WebGLRenderer,
      camera: new THREE.PerspectiveCamera(),
      targetScene: new THREE.Scene(),
    });

    const material = new FakeWarmupMaterial('max', { KEEP_ME: 'yes', POINT_WIDTH: 64 });
    const mesh = makeRenderableMesh('points', material);
    const before = buildBlendWarmupFingerprint(material, mesh, false);

    manager.warmScene(mesh);

    expect(material.userData.blendingMode).toBe('max');
    expect(material.defines).toMatchObject({
      KEEP_ME: 'yes',
      POINT_WIDTH: 64,
      LUXAR_MAX_RGB_CONTRIBUTION: '',
    });

    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();
    await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(3);

    material.defines.POINT_WIDTH = 128;
    const after = buildBlendWarmupFingerprint(material, mesh, false);
    expect(after).not.toBe(before);

    manager.scheduleObject(mesh);
    expect(compileOne).toHaveBeenCalledTimes(3);

    await warmupTurns.releaseNext();
    expect(compileOne).toHaveBeenCalledTimes(4);
  });
});
