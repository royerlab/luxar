/**
 * The scene environment — the ONE lighting input Luxar's otherwise light-free scene
 * has, built lazily for physical mesh materials.
 *
 * Luxar's four geometry types are emissive by design and the house mesh shader lights
 * itself from a fixed view-space key, so the scene has never held a light or an
 * environment map. A physically based material renders BLACK without one. Rather than
 * add light objects to the graph (spec `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3 / §5),
 * the viewer sets `scene.environment` to a prefiltered (PMREM) copy of three's
 * procedural `RoomEnvironment`: no asset, a neutral key, believable reflections, and
 * what three's own examples light with.
 *
 * Two properties are load-bearing:
 *
 * - **Lazy.** Nothing is built until {@link SceneEnvironment.ensure} is called, and
 *   the only caller is the material manager's physical-material hook. A scene with no
 *   physical mesh keeps `scene.environment === null`, so it renders byte-identically
 *   to before this module existed.
 * - **Invisible to house materials.** `scene.environment` is read only by three's
 *   lighting-model materials. Every Luxar material is a `ShaderMaterial` /
 *   `NodeMaterial` with its own fragment code that never samples an environment, so
 *   setting it changes nothing about points, lines, splats or house meshes — a
 *   property the unit tests assert rather than assume.
 *
 * The PMREM generator is backend-specific — `three`'s for `WebGLRenderer`,
 * `three/webgpu`'s for `WebGPURenderer` — so the caller injects a factory; the
 * WebGPU one is reached through `rendering/tsl/registry.ts` to keep the lazy chunk
 * lazy (see {@link createSceneEnvironment}).
 *
 * @module rendering/environment/scene-environment
 */

import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { MaterialBackend } from '../material-manager/factories';
import { requireTslMaterials } from '../tsl/slot';
import { log, Modules } from '../../utils/log';

/**
 * The slice of a PMREM generator this module uses — the same on both backends, and
 * narrow enough that a unit test can hand in a stub without a GPU.
 */
export interface PmremGeneratorLike {
  fromScene(scene: THREE.Scene, sigma?: number): { texture: THREE.Texture; dispose(): void };
  dispose(): void;
}

/**
 * Blur applied when prefiltering the room. Three's `RoomEnvironment` examples use
 * `0.04`; it softens the room's hard box edges into the gentle gradients a reflection
 * on a curved surface should show, while leaving the area lights sharp enough to read
 * as highlights.
 */
export const ROOM_ENVIRONMENT_SIGMA = 0.04;

export class SceneEnvironment {
  private target: { texture: THREE.Texture; dispose(): void } | null = null;

  constructor(
    private readonly scene: THREE.Scene,
    private readonly createGenerator: () => PmremGeneratorLike
  ) {}

  /** Whether the environment has been built and assigned. */
  isReady(): boolean {
    return this.target !== null;
  }

  /**
   * Build the room environment once and assign it to `scene.environment`.
   *
   * Idempotent: every call after the first returns immediately. Synchronous on
   * purpose — a physical mesh created this frame should be lit on its first draw, and
   * `PMREMGenerator.fromScene` is synchronous on both backends.
   *
   * @returns `true` when this call did the build, `false` when it was already built.
   */
  ensure(): boolean {
    if (this.target) return false;
    const start = performance.now();
    const generator = this.createGenerator();
    const room = new RoomEnvironment();
    try {
      this.target = generator.fromScene(room, ROOM_ENVIRONMENT_SIGMA);
    } finally {
      // Both are scaffolding: the prefiltered target is the only thing kept.
      room.dispose();
      generator.dispose();
    }
    this.scene.environment = this.target.texture;
    log.info(
      Modules.RENDERER,
      `Scene environment built (RoomEnvironment PMREM) in ${(performance.now() - start).toFixed(1)} ms`
    );
    return true;
  }

  /** Release the prefiltered target and clear `scene.environment` if it is ours. */
  dispose(): void {
    if (!this.target) return;
    if (this.scene.environment === this.target.texture) this.scene.environment = null;
    this.target.dispose();
    this.target = null;
  }
}

/**
 * Build the environment for the active renderer.
 *
 * Dispatches on the material backend rather than on the renderer class: the WebGPU
 * renderer's PMREM generator is a different class from `three`'s and lives in the
 * lazy chunk, so it is fetched through the TSL registry — which is loaded before any
 * WebGPU material can exist, hence before this can ever be called on that backend.
 */
export function createSceneEnvironment(
  renderer: THREE.WebGLRenderer | object,
  backend: MaterialBackend,
  scene: THREE.Scene
): SceneEnvironment {
  const createGenerator = (): PmremGeneratorLike =>
    backend === 'tsl'
      ? (new (requireTslMaterials().environment.PMREMGenerator)(
          renderer as never
        ) as unknown as PmremGeneratorLike)
      : (new THREE.PMREMGenerator(renderer as THREE.WebGLRenderer) as PmremGeneratorLike);
  return new SceneEnvironment(scene, createGenerator);
}
