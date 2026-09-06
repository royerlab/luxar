/**
 * The lazy scene environment: built once, on demand, invisible to house materials,
 * released on dispose.
 *
 * The property that matters most is the negative one — a scene with no physical mesh
 * must keep `scene.environment === null` and render exactly as before — so the tests
 * pin what does NOT happen as carefully as what does.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  SceneEnvironment,
  ROOM_ENVIRONMENT_SIGMA,
  type PmremGeneratorLike,
} from '../../../../rendering/environment/scene-environment';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';

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

describe('SceneEnvironment', () => {
  it('builds nothing until ensure() — a scene without physical meshes stays environment-free', () => {
    const scene = new THREE.Scene();
    const gen = makeGenerator();
    const env = new SceneEnvironment(scene, gen.factory);
    expect(env.isReady()).toBe(false);
    expect(scene.environment).toBeNull();
    expect(gen.constructed.count).toBe(0);
    expect(gen.fromScene).not.toHaveBeenCalled();
  });

  it('ensure() prefilters a RoomEnvironment once, assigns it, and releases the scaffolding', () => {
    const scene = new THREE.Scene();
    const gen = makeGenerator();
    const env = new SceneEnvironment(scene, gen.factory);

    expect(env.ensure()).toBe(true);
    expect(env.isReady()).toBe(true);
    expect(scene.environment).toBe(gen.texture);
    expect(gen.fromScene).toHaveBeenCalledTimes(1);
    const [room, sigma] = gen.fromScene.mock.calls[0] as unknown as [THREE.Scene, number];
    // A RoomEnvironment is a Scene of area-light boxes, not an empty scene.
    expect(room).toBeInstanceOf(THREE.Scene);
    expect(room.children.length).toBeGreaterThan(0);
    expect(sigma).toBe(ROOM_ENVIRONMENT_SIGMA);
    // The generator is scaffolding: disposed right after the build.
    expect(gen.disposeGenerator).toHaveBeenCalledTimes(1);

    // Idempotent.
    expect(env.ensure()).toBe(false);
    expect(gen.constructed.count).toBe(1);
    expect(gen.fromScene).toHaveBeenCalledTimes(1);
  });

  it('house materials never read scene.environment, so setting it changes none of their state', () => {
    const scene = new THREE.Scene();
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
        // A house material declares no envMap slot at all — the property three's
        // lighting-model materials read `scene.environment` through.
        envMap: (m as unknown as { envMap?: unknown }).envMap ?? null,
      });
    const before = [snapshot(mesh), snapshot(points)];

    new SceneEnvironment(scene, makeGenerator().factory).ensure();
    expect(scene.environment).not.toBeNull();

    expect([snapshot(mesh), snapshot(points)]).toEqual(before);
    // ShaderMaterial has no environment hook: the vertex/fragment source is Luxar's own.
    expect('envMap' in mesh).toBe(false);
    expect(mesh.fragmentShader).not.toContain('envMap');
  });

  it('dispose() releases the target and clears scene.environment only if it is still ours', () => {
    const scene = new THREE.Scene();
    const gen = makeGenerator();
    const env = new SceneEnvironment(scene, gen.factory);
    env.ensure();
    env.dispose();
    expect(gen.disposeTarget).toHaveBeenCalledTimes(1);
    expect(scene.environment).toBeNull();
    expect(env.isReady()).toBe(false);
    // Disposing twice is safe.
    env.dispose();
    expect(gen.disposeTarget).toHaveBeenCalledTimes(1);

    // Someone else's environment is left alone.
    const other = new THREE.Texture();
    const env2 = new SceneEnvironment(scene, makeGenerator().factory);
    env2.ensure();
    scene.environment = other;
    env2.dispose();
    expect(scene.environment).toBe(other);
  });

  it('a throwing generator leaves the environment unbuilt and still disposes the scaffolding', () => {
    const scene = new THREE.Scene();
    const disposeGenerator = vi.fn();
    const env = new SceneEnvironment(scene, () => ({
      fromScene: () => {
        throw new Error('no GPU');
      },
      dispose: disposeGenerator,
    }));
    expect(() => env.ensure()).toThrow('no GPU');
    expect(env.isReady()).toBe(false);
    expect(scene.environment).toBeNull();
    expect(disposeGenerator).toHaveBeenCalledTimes(1);
  });
});
