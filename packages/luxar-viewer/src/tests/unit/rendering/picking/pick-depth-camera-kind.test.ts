/**
 * TSL pick depth does not depend on the camera kind the graph was BUILT under.
 *
 * Under the surface depth convention (opaque / normal) every pick graph writes the
 * fragment's real depth, and front-most wins. three's `depth` node picks
 * `viewZToPerspectiveDepth` or `viewZToOrthographicDepth` from
 * `camera.isPerspectiveCamera` when the graph is built. three's node-build cache key
 * does not include the camera kind, so a build is reused across cameras: a graph
 * first built under a perspective camera kept the perspective formula under an
 * orthographic one. Graphs built under different kinds then compared different
 * formulas in one pick depth buffer, and front-most picking broke after a
 * perspective/ortho switch, or for a node first drawn after one.
 *
 * The observable without a GPU is the generated WGSL. A real `WGSLNodeBuilder`
 * builds each pick material once under a perspective camera and once under an
 * orthographic one. The two fragment shaders must be identical, so a build reused
 * across kinds is correct for both, and must carry the orthographic formula (the
 * branch is taken at run time from the projection matrix).
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';

import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { MeshPickingTSLMaterial } from '../../../../rendering/picking/mesh/material-tsl';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';

/** The renderer surface `WGSLNodeBuilder.build()` reads (no GPU device needed). */
function stubRenderer(): unknown {
  return {
    getRenderTarget: () => null,
    getMRT: () => null,
    depth: true,
    reversedDepthBuffer: false,
    logarithmicDepthBuffer: false,
    contextNode: context(),
    library: { fromMaterial: (m: unknown) => m },
    hasFeature: () => false,
    hasCompatibility: () => false,
    backend: {
      compatibilityMode: false,
      capabilities: { getUniformBufferLimit: () => 65536 },
      utils: {
        getTextureSampleData: () => ({ samples: 1, primarySamples: 1, isMSAA: false }),
      },
    },
  };
}

/** The `NodeBuilder` members this test drives (three's typings omit them). */
interface BuilderSurface {
  camera: THREE.Camera;
  scene: THREE.Scene;
  material: NodeMaterial;
  fragmentShader: string;
  build(): void;
}

/** Build `material` for `camera` and return its WGSL fragment shader. */
function fragmentFor(material: THREE.Material, camera: THREE.Camera): string {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
  const builder = new WGSLNodeBuilder(
    mesh,
    stubRenderer() as ConstructorParameters<typeof WGSLNodeBuilder>[1]
  ) as unknown as BuilderSurface;
  builder.camera = camera;
  builder.scene = new THREE.Scene();
  builder.material = material as NodeMaterial;
  builder.build();
  return String(builder.fragmentShader);
}

/** viewZToOrthographicDepth as three emits it. */
const ORTHO_DEPTH =
  /\(\s*positionView\.z \+ render\.cameraNear\s*\)\s*\/\s*\(\s*render\.cameraNear - render\.cameraFar\s*\)/;

describe('TSL pick depth is chosen per draw, not per build', () => {
  it.each([
    ['point', () => new PointPickingTSLMaterial({ nodeId: 1 })],
    ['line quad', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' })],
    ['line capsule', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'capsule' })],
    ['gsplat', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
    ['mesh', () => new MeshPickingTSLMaterial({ nodeId: 1 })],
  ])('%s: a perspective build writes the ortho depth under an ortho camera', (_n, make) => {
    const persp = fragmentFor(make(), new THREE.PerspectiveCamera(50, 1, 0.1, 100));
    const ortho = fragmentFor(make(), new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100));
    expect(persp).toContain('frag_depth');
    expect(persp).toMatch(ORTHO_DEPTH);
    expect(persp).toBe(ortho);
  });
});
