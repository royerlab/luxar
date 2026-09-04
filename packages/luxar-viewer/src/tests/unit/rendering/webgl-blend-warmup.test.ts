import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import type { BlendingMode } from '../../../types/blending';
import {
  WebGLBlendWarmupManager,
  buildBlendWarmupFingerprint,
} from '../../../rendering/webgl-blend-warmup';

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
      await Promise.resolve();
      await Promise.resolve();
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

describe('WebGLBlendWarmupManager', () => {
  it('defers explicit scene activation and ignores scheduling until it runs', async () => {
    const warmupTurns = makeWarmupTurnController();
    const activations = makeActivationController();
    const compileOne = vi.fn();
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activations.defer);
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activations.defer);
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
      const manager = new WebGLBlendWarmupManager(undefined, compileOne);
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
      const manager = new WebGLBlendWarmupManager(undefined, compileOne);
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
    const manager = new WebGLBlendWarmupManager(
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
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

  it('releases scene warming on its readiness budget and keeps draining after', async () => {
    vi.useFakeTimers();
    try {
      const warmupTurns = makeWarmupTurnController();
      const compileOne = vi.fn();
      const manager = new WebGLBlendWarmupManager(
        warmupTurns.wait,
        compileOne,
        activateImmediately
      );
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, vi.fn(), activateImmediately);
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);
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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

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
    const manager = new WebGLBlendWarmupManager(
      async () => undefined,
      compileOne,
      activateImmediately
    );

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
    const manager = new WebGLBlendWarmupManager(warmupTurns.wait, compileOne, activateImmediately);

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
