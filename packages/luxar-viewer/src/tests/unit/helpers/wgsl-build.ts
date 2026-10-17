import * as THREE from 'three';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';

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

/** The `NodeBuilder` members these tests drive (three's typings omit them). */
interface BuilderSurface {
  camera: THREE.Camera;
  scene: THREE.Scene;
  material: NodeMaterial;
  fragmentShader: string;
  build(): void;
}

/** Build a material without a GPU and return its WGSL fragment shader. */
export function wgslFragmentFor(
  material: THREE.Material,
  camera: THREE.Camera = new THREE.PerspectiveCamera(),
  geometry: THREE.BufferGeometry = new THREE.BufferGeometry()
): string {
  const mesh = new THREE.Mesh(geometry, material);
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
