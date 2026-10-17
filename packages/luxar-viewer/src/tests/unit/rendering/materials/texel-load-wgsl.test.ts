import * as THREE from 'three';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';

interface BuilderSurface {
  camera: THREE.Camera;
  scene: THREE.Scene;
  material: NodeMaterial;
  vertexShader: string;
  build(): void;
}

function vertexWGSL(material: NodeMaterial): string {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('aSortedIndex', new THREE.Uint32BufferAttribute([0], 1));
  geometry.setAttribute('aSortedIndexB', new THREE.Uint32BufferAttribute([0], 1));
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute([0, 0], 2));
  const renderer = {
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
  };
  const builder = new WGSLNodeBuilder(
    new THREE.Mesh(geometry, material),
    renderer as unknown as ConstructorParameters<typeof WGSLNodeBuilder>[1]
  ) as unknown as BuilderSurface;
  builder.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  builder.scene = new THREE.Scene();
  builder.material = material;
  builder.build();
  return builder.vertexShader;
}

describe('element texture loads in generated WGSL', () => {
  it.each([
    ['gsplat', () => new GSplatTSLMaterial(), 4],
    ['point', () => new PointTSLMaterial(), 3],
    ['line quad', () => new LineTSLMaterial({ primitive: 'screen-space' }), 10],
    ['line capsule', () => new LineTSLMaterial({ primitive: 'capsule' }), 10],
  ] as const)('%s uses untransformed integer texel coordinates', (_name, make, loadCount) => {
    const vertex = vertexWGSL(make());
    const loads = vertex.split('\n').filter((line) => line.includes('textureLoad('));
    expect(loads).toHaveLength(loadCount);
    for (const load of loads) {
      expect(load).not.toMatch(/nodeUniform\d+ \* vec3<f32>/);
    }
  });
});
