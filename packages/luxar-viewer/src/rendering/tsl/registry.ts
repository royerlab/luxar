/**
 * The TSL / WebGPU cone, gathered behind ONE module.
 *
 * This is the **only** module under `src/` that may statically import
 * `three/webgpu` or `three/tsl` (transitively, via the `*-tsl` / `*.tsl`
 * modules it pulls in). Everything else reaches these classes and factories
 * through `rendering/tsl/load`, which imports this file
 * dynamically — so rolldown places this whole subgraph, plus the ~182 kB
 * gzipped `three-webgpu` chunk, in a lazy chunk that is fetched only when the
 * WebGPU backend is actually selected.
 *
 * That is the entire point of the file's existence, and the reason it looks
 * like a barrel with nothing of its own to say. Two gates hold it: the ESLint
 * `no-restricted-imports` rule in `eslint.config.js`, which bans value imports
 * of `three/webgpu` / `three/tsl` outside this file and the `*-tsl` modules it
 * owns, and `scripts/check-eager-chunks.mjs`, which asserts against the built
 * `dist/` that nothing eagerly reachable from the entry imports the chunk.
 * Both are needed: the previous arrangement — 23 production modules importing
 * `three/webgpu` directly — cost every WebGL user a download they never
 * executed, and the last edge was not in the source at all but in how the
 * shared `three.core.js` was chunked. Nothing in the repo noticed for as long
 * as either was true. See issue #1679.
 *
 * Two things do NOT belong here:
 *
 * - Type-only imports of these symbols. `import type { PointTSLMaterial }` is
 *   erased at build time and costs nothing, so consumers should keep doing
 *   that directly rather than routing types through the registry.
 * - Anything that the WebGL path needs. If a symbol turns out to be required
 *   before the backend is known, it does not belong in the lazy cone at all —
 *   move it to a module with no `three/webgpu` edge (the precedent is
 *   `material-manager/soft-dispose-flag.ts`, a zero-import leaf that exists
 *   for exactly this reason).
 *
 * @module rendering/tsl/registry
 */

// Visual material classes (`extends NodeMaterial`).
import { PointTSLMaterial } from '../materials/point/material-tsl';
import { LineTSLMaterial } from '../materials/line/material-tsl';
import { GSplatTSLMaterial } from '../materials/gsplat/material-tsl';
import { MeshTSLMaterial } from '../materials/mesh/material-tsl';
import { PhysicalMeshTSLMaterial } from '../materials/mesh-physical/material-tsl';

// The WebGPU renderer's own PMREM generator and cube render target, for the scene
// environment that lights physical meshes (`rendering/environment/`). Different
// classes from `three`'s WebGL ones (`PMREMGenerator`, `WebGLCubeRenderTarget`), and
// the one place outside the material classes where the environment code needs a
// `three/webgpu` symbol.
import { CubeRenderTarget, PMREMGenerator } from 'three/webgpu';

// Picking material classes (`extends NodeMaterial`).
import { PointPickingTSLMaterial } from '../picking/point/material-tsl';
import { LinePickingTSLMaterial } from '../picking/line/material-tsl';
import { GSplatPickingTSLMaterial } from '../picking/gsplat/material-tsl';
import { MeshPickingTSLMaterial } from '../picking/mesh/material-tsl';

// Post-processing mega-shader material class.
import { MegaShaderTSLMaterial } from '../post-processing/mega/material-tsl';

// Visual TSL graph factories, as consumed by the `ShaderSource.webgpu`
// closures that stay co-located with their GLSL twin.
import { pointWebGPUFactory, buildPointTSLNodesFromUniforms } from '../materials/point/shader-tsl';
import { lineWebGPUFactory, buildLineTSLNodesFromUniforms } from '../materials/line/shader-tsl';
import { capsuleLineWebGPUFactory } from '../materials/line/shader-tsl-capsule';
import {
  gsplatWebGPUFactory,
  buildGSplatTSLNodesFromUniforms,
} from '../materials/gsplat/shader-tsl';
import { meshWebGPUFactory, buildMeshTSLNodesFromUniforms } from '../materials/mesh/shader-tsl';

// Picking TSL graph factories.
import {
  pointPickWebGPUFactory,
  buildPointPickTSLNodesFromUniforms,
} from '../picking/point/pick.tsl';
import { linePickWebGPUFactory, buildLinePickTSLNodesFromUniforms } from '../picking/line/pick.tsl';
import { capsuleLinePickWebGPUFactory } from '../picking/line/pick-capsule.tsl';
import {
  gsplatPickWebGPUFactory,
  buildGSplatPickTSLNodesFromUniforms,
} from '../picking/gsplat/pick.tsl';
import { meshPickWebGPUFactory, buildMeshPickTSLNodesFromUniforms } from '../picking/mesh/pick.tsl';

// Post-processing TSL graph factories.
import { megaWebGPUFactory } from '../post-processing/mega/shader.tsl';
import {
  bloomThresholdWebGPUFactory,
  bloomDownsampleWebGPUFactory,
  bloomUpsampleWebGPUFactory,
} from '../post-processing/bloom/bloom.tsl';
import { fxaaWebGPUFactory } from '../post-processing/fxaa/fxaa.tsl';

/**
 * Everything the WebGPU path needs, in one value.
 *
 * Grouped by role rather than by directory so call sites read as
 * `registry.materials.point` / `registry.factories.pickLine` — the shape the
 * `MaterialManager` dispatch tables and the `ShaderSource.webgpu` closures
 * actually index by.
 */
export const TSL_REGISTRY = {
  /** Visual material classes, keyed by geometry kind. */
  materials: {
    point: PointTSLMaterial,
    line: LineTSLMaterial,
    gsplat: GSplatTSLMaterial,
    mesh: MeshTSLMaterial,
    /**
     * The physical mesh family — three's `MeshPhysicalNodeMaterial` behind the Luxar
     * leaf-material surface. Its own key rather than a variant of `mesh`, because none
     * of the house contracts (codegen snapshots, per-epoch shading variant, blend-mode
     * defines) apply to it; picking still uses `picking.mesh`.
     */
    meshPhysical: PhysicalMeshTSLMaterial,
  },
  /** Backend-specific scene-environment tooling (`rendering/environment/`). */
  environment: { PMREMGenerator, CubeRenderTarget },
  /** Picking material classes, keyed by geometry kind. */
  picking: {
    point: PointPickingTSLMaterial,
    line: LinePickingTSLMaterial,
    gsplat: GSplatPickingTSLMaterial,
    mesh: MeshPickingTSLMaterial,
  },
  /** The single post-processing mega-shader material class. */
  mega: MegaShaderTSLMaterial,
  /**
   * Graph factories, for the `ShaderSource.webgpu` closures. Those closures
   * stay next to their GLSL twin (they carry shader-specific notes about
   * reading build-time uniforms) and pull the factory from here instead of
   * importing it, which is what actually cuts the eager edge.
   */
  factories: {
    point: { pointWebGPUFactory, buildPointTSLNodesFromUniforms },
    line: { lineWebGPUFactory, buildLineTSLNodesFromUniforms },
    capsuleLine: { capsuleLineWebGPUFactory, buildLineTSLNodesFromUniforms },
    gsplat: { gsplatWebGPUFactory, buildGSplatTSLNodesFromUniforms },
    mesh: { meshWebGPUFactory, buildMeshTSLNodesFromUniforms },
    mega: { megaWebGPUFactory },
    pickPoint: { pointPickWebGPUFactory, buildPointPickTSLNodesFromUniforms },
    pickLine: { linePickWebGPUFactory, buildLinePickTSLNodesFromUniforms },
    pickCapsuleLine: { capsuleLinePickWebGPUFactory, buildLinePickTSLNodesFromUniforms },
    pickGsplat: { gsplatPickWebGPUFactory, buildGSplatPickTSLNodesFromUniforms },
    pickMesh: { meshPickWebGPUFactory, buildMeshPickTSLNodesFromUniforms },
    bloom: {
      bloomThresholdWebGPUFactory,
      bloomDownsampleWebGPUFactory,
      bloomUpsampleWebGPUFactory,
    },
    fxaa: { fxaaWebGPUFactory },
  },
} as const;

/** Shape of {@link TSL_REGISTRY}, for the type-only consumers in `load.ts`. */
export type TslRegistry = typeof TSL_REGISTRY;
