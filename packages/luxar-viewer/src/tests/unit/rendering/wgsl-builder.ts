import * as THREE from 'three';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';

/** The renderer surface `WGSLNodeBuilder.build()` reads (no GPU device needed). */
function stubRenderer(): ConstructorParameters<typeof WGSLNodeBuilder>[1] {
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
      utils: { getTextureSampleData: () => ({ samples: 1, primarySamples: 1, isMSAA: false }) },
    },
  } as unknown as ConstructorParameters<typeof WGSLNodeBuilder>[1];
}

interface BuilderSurface {
  camera: THREE.Camera;
  scene: THREE.Scene;
  material: NodeMaterial;
  vertexShader: string;
  fragmentShader: string;
  build(): void;
}

/** Generated shader source for a material built by the WGSL test helper. */
export interface BuiltWGSL {
  vertexShader: string;
  fragmentShader: string;
}

/** Build a real material graph and return its generated WGSL. */
export function buildWGSL(material: THREE.Material, camera: THREE.Camera): BuiltWGSL {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('aSortedIndex', new THREE.Uint32BufferAttribute([0], 1));
  geometry.setAttribute('aSortedIndexB', new THREE.Uint32BufferAttribute([0], 1));
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute([0, 0], 2));
  const builder = new WGSLNodeBuilder(
    new THREE.Mesh(geometry, material),
    stubRenderer()
  ) as unknown as BuilderSurface;
  builder.camera = camera;
  builder.scene = new THREE.Scene();
  builder.material = material as NodeMaterial;
  builder.build();
  return { vertexShader: builder.vertexShader, fragmentShader: builder.fragmentShader };
}
