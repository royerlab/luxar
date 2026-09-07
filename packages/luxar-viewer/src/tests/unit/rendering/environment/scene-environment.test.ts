/**
 * The scene environment: lazy, invisible to house materials, released on dispose —
 * and, since Phase 4, three sources behind one precedence rule (baked > authored
 * source > room) with a live capture that re-runs on stale marks once settled.
 *
 * The property that matters most is the negative one — a scene with no physical mesh
 * must keep `scene.environment === null` and render exactly as before — so the tests
 * pin what does NOT happen as carefully as what does. The capture itself runs against
 * a stub renderer that records the six `setRenderTarget` / `render` pairs a
 * `CubeCamera.update` issues, so the orchestration (hide physical meshes, push and
 * restore camera params, assign the texture) is exercised without a GPU.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  CAPTURE_DEBOUNCE_MS,
  ROOM_ENVIRONMENT_SIGMA,
  SceneEnvironment,
  type CaptureRuntime,
  type PmremGeneratorLike,
  type SceneEnvironmentDeps,
} from '../../../../rendering/environment/scene-environment';
import type { CubeTargetLike } from '../../../../rendering/environment/cube-capture';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { PhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/material-glsl';
import { DEFAULT_ENVIRONMENT_CONFIG, type BakedEnvironment } from '../../../../types/environment';

function makeGenerator(): {
  factory: () => PmremGeneratorLike;
  fromScene: ReturnType<typeof vi.fn>;
  disposeGenerator: ReturnType<typeof vi.fn>;
  disposeTarget: ReturnType<typeof vi.fn>;
  texture: THREE.Texture;
  constructed: { count: number };
} {
  const texture = new THREE.Texture();
  const disposeTarget = vi.fn();
  const disposeGenerator = vi.fn();
  const fromScene = vi.fn(() => ({ texture, dispose: disposeTarget }));
  const constructed = { count: 0 };
  const factory = (): PmremGeneratorLike => {
    constructed.count++;
    return { fromScene, dispose: disposeGenerator };
  };
  return { factory, fromScene, disposeGenerator, disposeTarget, texture, constructed };
}

/** A renderer stub that satisfies `CubeCamera.update` and records its draws. */
function makeRenderer(): {
  renderer: unknown;
  renders: Array<{ face: number; camera: THREE.Camera }>;
} {
  const renders: Array<{ face: number; camera: THREE.Camera }> = [];
  let activeFace = 0;
  const renderer = {
    isWebGLRenderer: true,
    coordinateSystem: THREE.WebGLCoordinateSystem,
    xr: { enabled: false },
    state: { buffers: { depth: { getReversed: () => false } } },
    getRenderTarget: () => null,
    getActiveCubeFace: () => 0,
    getActiveMipmapLevel: () => 0,
    setRenderTarget: (_t: unknown, face = 0) => {
      activeFace = face;
    },
    render: (_scene: THREE.Scene, camera: THREE.Camera) => {
      renders.push({ face: activeFace, camera });
    },
  };
  return { renderer, renders };
}

function makeCubeTarget(resolution: number): CubeTargetLike & { disposed: boolean } {
  const texture = new THREE.CubeTexture();
  const target = {
    texture,
    width: resolution,
    height: resolution,
    disposed: false,
    dispose(): void {
      this.disposed = true;
    },
  };
  return target;
}

function makeEnv(overrides: Partial<SceneEnvironmentDeps> = {}) {
  const scene = new THREE.Scene();
  const gen = makeGenerator();
  const { renderer, renders } = makeRenderer();
  const targets: Array<CubeTargetLike & { disposed: boolean }> = [];
  const deps: SceneEnvironmentDeps = {
    scene,
    renderer,
    createGenerator: gen.factory,
    createCubeTarget: (resolution) => {
      const t = makeCubeTarget(resolution);
      targets.push(t);
      return t;
    },
    ...overrides,
  };
  const env = new SceneEnvironment(deps);
  return { env, scene, gen, renders, targets };
}

function makeRuntime(root: THREE.Object3D | null, settled = true) {
  const pushes: number[] = [];
  const restores = { count: 0 };
  const runtime: CaptureRuntime & { settled: boolean } = {
    settled,
    sceneRoot: () => root,
    pushCaptureCameraParams: (resolution) => {
      pushes.push(resolution);
    },
    restoreCameraParams: () => {
      restores.count++;
    },
    isSettled: () => runtime.settled,
    baseUrl: () => undefined,
  };
  return { runtime, pushes, restores };
}

function makeBaked(resolution = 4): BakedEnvironment {
  const face = new Uint16Array(resolution * resolution * 4).fill(0x3c00); // half 1.0
  return {
    header: {
      format: 'cube-faces-half',
      face_order: ['px', 'nx', 'py', 'ny', 'pz', 'nz'],
      coordinate_system: 'webgl',
      probe: { spec: 'auto', position: [0, 0, 0] },
      resolution,
      scene_content_hash: 'abc',
    },
    resolution,
    faces: [face, face, face, face, face, face],
  };
}

describe('SceneEnvironment — lazy room (Phase 1 contract)', () => {
  it('builds nothing until ensure() — a scene without physical meshes stays environment-free', () => {
    const { env, scene, gen } = makeEnv();
    expect(env.isReady()).toBe(false);
    expect(env.activeKind()).toBe('none');
    expect(scene.environment).toBeNull();
    expect(gen.constructed.count).toBe(0);
    expect(gen.fromScene).not.toHaveBeenCalled();
    // Configuring and attaching a runtime change nothing either: light is only
    // built when a physical material asks for it.
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(makeRuntime(null).runtime);
    expect(scene.environment).toBeNull();
    expect(env.captureCount).toBe(0);
  });

  it('ensure() prefilters a RoomEnvironment once, assigns it, and releases the scaffolding', () => {
    const { env, scene, gen } = makeEnv();

    expect(env.ensure()).toBe(true);
    expect(env.isReady()).toBe(true);
    expect(env.activeKind()).toBe('room');
    expect(scene.environment).toBe(gen.texture);
    expect(gen.fromScene).toHaveBeenCalledTimes(1);
    const [room, sigma] = gen.fromScene.mock.calls[0] as unknown as [THREE.Scene, number];
    expect(room).toBeInstanceOf(THREE.Scene);
    expect(room.children.length).toBeGreaterThan(0);
    expect(sigma).toBe(ROOM_ENVIRONMENT_SIGMA);
    expect(gen.disposeGenerator).toHaveBeenCalledTimes(1);

    // Idempotent.
    expect(env.ensure()).toBe(false);
    expect(gen.constructed.count).toBe(1);
    expect(gen.fromScene).toHaveBeenCalledTimes(1);
  });

  it('house materials never read scene.environment, so setting it changes none of their state', () => {
    const { env, scene } = makeEnv();
    const mesh = new MeshMaterial({});
    const points = new PointMaterial({});
    const snapshot = (m: THREE.ShaderMaterial): string =>
      JSON.stringify({
        uniforms: Object.fromEntries(
          Object.entries(m.uniforms).map(([k, u]) => [
            k,
            u.value instanceof THREE.Texture ? 'tex' : u.value,
          ])
        ),
        defines: m.defines,
        envMap: (m as unknown as { envMap?: unknown }).envMap ?? null,
      });
    const before = [snapshot(mesh), snapshot(points)];

    env.ensure();
    expect(scene.environment).not.toBeNull();

    expect([snapshot(mesh), snapshot(points)]).toEqual(before);
    expect('envMap' in mesh).toBe(false);
    expect(mesh.fragmentShader).not.toContain('envMap');
  });

  it('dispose() releases the target and clears scene.environment only if it is still ours', () => {
    const { env, scene, gen } = makeEnv();
    env.ensure();
    env.dispose();
    expect(gen.disposeTarget).toHaveBeenCalledTimes(1);
    expect(scene.environment).toBeNull();
    expect(env.isReady()).toBe(false);
    env.dispose();
    expect(gen.disposeTarget).toHaveBeenCalledTimes(1);

    const other = new THREE.Texture();
    const second = makeEnv();
    second.env.ensure();
    second.scene.environment = other;
    second.env.dispose();
    expect(second.scene.environment).toBe(other);
  });

  it('a throwing generator leaves the environment unbuilt and still disposes the scaffolding', () => {
    const disposeGenerator = vi.fn();
    const { env, scene } = makeEnv({
      createGenerator: () => ({
        fromScene: () => {
          throw new Error('no GPU');
        },
        dispose: disposeGenerator,
      }),
    });
    expect(() => env.ensure()).toThrow('no GPU');
    expect(env.isReady()).toBe(false);
    expect(scene.environment).toBeNull();
    expect(disposeGenerator).toHaveBeenCalledTimes(1);
  });
});

describe('SceneEnvironment — sources and precedence (Phase 4)', () => {
  it('configure() applies the intensity immediately and, once lit, the source', () => {
    const { env, scene, gen } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, intensity: 0.4 });
    expect(scene.environmentIntensity).toBe(0.4);
    expect(scene.environment).toBeNull();
    env.ensure();
    expect(scene.environment).toBe(gen.texture);
    env.configure(null);
    expect(scene.environmentIntensity).toBe(1);
    expect(env.getConfig()).toBe(DEFAULT_ENVIRONMENT_CONFIG);
  });

  it("source 'scene' without a runtime falls back to the room; attaching one captures", () => {
    const { env, scene, gen, renders, targets } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene', resolution: 32 });
    env.ensure();
    expect(env.activeKind()).toBe('room');
    expect(scene.environment).toBe(gen.texture);

    const root = new THREE.Group();
    root.name = 'LuxarScene';
    const { runtime, pushes, restores } = makeRuntime(root);
    env.attachRuntime(runtime);

    expect(env.activeKind()).toBe('scene');
    expect(env.captureCount).toBe(1);
    expect(targets).toHaveLength(1);
    expect(targets[0].width).toBe(32);
    expect(scene.environment).toBe(targets[0].texture);
    // Six faces drawn through a 90° camera; params pushed before, restored after.
    expect(renders).toHaveLength(6);
    expect(new Set(renders.map((r) => r.face))).toEqual(new Set([0, 1, 2, 3, 4, 5]));
    expect((renders[0].camera as THREE.PerspectiveCamera).fov).toBe(-90);
    expect(pushes).toEqual([32]);
    expect(restores.count).toBe(1);
    // The capture asked three to re-prefilter (CubeCamera.update's contract).
    expect(targets[0].texture.pmremVersion).toBeGreaterThan(0);
  });

  it('ensure is idempotent for an unchanged live scene capture', () => {
    const { env, targets, renders } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene', resolution: 32 });
    env.attachRuntime(makeRuntime(new THREE.Group()).runtime);

    expect(env.ensure()).toBe(true);
    expect(env.ensure()).toBe(false);
    expect(env.ensure()).toBe(false);
    expect(env.captureCount).toBe(1);
    expect(targets).toHaveLength(1);
    expect(renders).toHaveLength(6);

    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene', resolution: 64 });
    expect(env.captureCount).toBe(2);
    expect(targets).toHaveLength(2);
    expect(targets[0].disposed).toBe(true);
  });

  it('rebuilds a live capture without dropping its runtime', () => {
    const { env, targets } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(makeRuntime(new THREE.Group()).runtime);
    env.ensure();

    expect(env.rebuild()).toBe(true);
    expect(targets[0].disposed).toBe(true);
    expect(targets).toHaveLength(2);
    expect(env.activeKind()).toBe('scene');
    expect(env.captureCount).toBe(2);
  });

  it('resetForDataset clears demand and releases dataset-owned textures', () => {
    const { env, scene, targets } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(makeRuntime(new THREE.Group()).runtime);
    env.ensure();
    env.resetForDataset();

    expect(targets[0].disposed).toBe(true);
    expect(scene.environment).toBeNull();
    expect(env.activeKind()).toBe('none');
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    expect(env.captureCount).toBe(1);

    const room = makeEnv();
    room.env.ensure();
    room.env.resetForDataset();
    expect(room.scene.environment).toBeNull();
    expect(room.env.activeKind()).toBe('none');
  });

  it('switching away from a live capture disposes its cube target', () => {
    const { env, targets } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(makeRuntime(new THREE.Group()).runtime);
    env.ensure();
    env.configure(DEFAULT_ENVIRONMENT_CONFIG);
    expect(targets[0].disposed).toBe(true);
    expect(env.activeKind()).toBe('room');
  });

  it('a capture hides physical meshes for the six draws and restores their previous visibility', () => {
    const { env, scene, renders } = makeEnv();
    const root = new THREE.Group();
    scene.add(root);
    const physical = new THREE.Mesh(new THREE.BufferGeometry(), new PhysicalMeshMaterial({}));
    const hiddenPhysical = new THREE.Mesh(new THREE.BufferGeometry(), new PhysicalMeshMaterial({}));
    hiddenPhysical.visible = false;
    const house = new THREE.Mesh(new THREE.BufferGeometry(), new MeshMaterial({}));
    root.add(physical, hiddenPhysical, house);
    const seen: boolean[][] = [];
    const { renderer } = makeRenderer();
    (renderer as { render: (s: THREE.Scene) => void }).render = () => {
      seen.push([physical.visible, hiddenPhysical.visible, house.visible]);
    };
    const withSpy = new SceneEnvironment({
      scene,
      renderer,
      createGenerator: makeGenerator().factory,
      createCubeTarget: makeCubeTarget,
    });
    withSpy.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    withSpy.attachRuntime(makeRuntime(root).runtime);
    const result = withSpy.captureScene();
    expect(result?.hiddenPhysicalMeshes).toBe(1);
    expect(seen).toHaveLength(6);
    for (const frame of seen) expect(frame).toEqual([false, false, true]);
    // Restored EXACTLY: the user-hidden one stays hidden.
    expect(physical.visible).toBe(true);
    expect(hiddenPhysical.visible).toBe(false);
    expect(house.visible).toBe(true);
    expect(renders).toHaveLength(0); // the spy renderer, not the shared one, drew
    expect(env.captureCount).toBe(0);
  });

  it('a stale mark re-captures after the debounce, only once settled, and only while live', () => {
    const { env } = makeEnv();
    const root = new THREE.Group();
    const { runtime } = makeRuntime(root, false);
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(runtime);
    env.ensure();
    expect(env.captureCount).toBe(1);

    const t0 = 10_000;
    env.markStale();
    // Too soon, and not settled: nothing.
    expect(env.tick(t0)).toBe(false);
    expect(env.tick(t0 + CAPTURE_DEBOUNCE_MS + 1)).toBe(false);
    runtime.settled = true;
    // The mark was taken at performance.now(); a far-future tick is past the debounce.
    expect(env.tick(performance.now() + CAPTURE_DEBOUNCE_MS + 1)).toBe(true);
    expect(env.captureCount).toBe(2);
    // Consumed: the next tick is idle.
    expect(env.tick(performance.now() + CAPTURE_DEBOUNCE_MS + 2)).toBe(false);

    // A mark while lit by the room is a no-op (nothing to re-capture).
    const roomOnly = makeEnv();
    roomOnly.env.ensure();
    roomOnly.env.markStale();
    roomOnly.env.attachRuntime(makeRuntime(root).runtime);
    expect(roomOnly.env.tick(performance.now() + 10_000)).toBe(false);
  });

  it('a baked map wins over the authored source, and clearing it falls back', () => {
    const { env, scene, gen, targets } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(makeRuntime(new THREE.Group()).runtime);
    env.setBaked(makeBaked());
    env.ensure();
    expect(env.activeKind()).toBe('baked');
    expect(env.hasBaked()).toBe(true);
    expect(scene.environment).toBeInstanceOf(THREE.CubeTexture);
    expect((scene.environment as THREE.CubeTexture).type).toBe(THREE.HalfFloatType);
    expect(env.captureCount).toBe(0);
    expect(targets).toHaveLength(0);
    // A stale mark under a baked map does nothing: the map is frozen by design.
    env.markStale();
    expect(env.tick(performance.now() + 10_000)).toBe(false);

    const baked = scene.environment;
    env.setBaked(null);
    expect(env.activeKind()).toBe('scene');
    expect(env.captureCount).toBe(1);
    expect(scene.environment).not.toBe(baked);
    expect(gen.fromScene).not.toHaveBeenCalled();
  });

  it("source 'hdri' lights with the room until the image arrives", () => {
    const { env, scene, gen } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'hdri', url: 'nowhere/studio.hdr' });
    env.ensure();
    expect(env.activeKind()).toBe('room');
    expect(scene.environment).toBe(gen.texture);
  });

  it('dispose() releases every texture it owns, whichever was active', () => {
    const { env, scene, targets, gen } = makeEnv();
    env.configure({ ...DEFAULT_ENVIRONMENT_CONFIG, source: 'scene' });
    env.attachRuntime(makeRuntime(new THREE.Group()).runtime);
    env.ensure();
    env.setBaked(makeBaked());
    env.dispose();
    expect(scene.environment).toBeNull();
    expect(targets[0].disposed).toBe(true);
    expect(env.isReady()).toBe(false);
    expect(env.captureCount).toBe(1);
    expect(gen.disposeTarget).not.toHaveBeenCalled(); // the room was never built here
  });
});
