import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { MeshTSLMaterial } from '../../../../rendering/materials/mesh/material-tsl';

const materials = [
  ['points', () => new PointTSLMaterial()],
  ['lines', () => new LineTSLMaterial()],
  ['gsplats', () => new GSplatTSLMaterial()],
  ['mesh', () => new MeshTSLMaterial({ blendingMode: 'additive' })],
] as const;

describe.each(materials)('%s TSL blend state', (_name, make) => {
  it('switches additive ↔ luminous without invalidating the node graph', () => {
    const material = make();
    const version = material.version;
    const vertexNode = material.vertexNode;
    const fragmentNode = material.fragmentNode;

    material.applyBlendingMode('luminous');
    expect(material.depthTest).toBe(true);
    expect(material.blending).toBe(THREE.AdditiveBlending);
    expect(material.vertexNode).toBe(vertexNode);
    expect(material.fragmentNode).toBe(fragmentNode);
    expect(material.version).toBe(version);

    material.applyBlendingMode('additive');
    expect(material.depthTest).toBe(false);
    expect(material.vertexNode).toBe(vertexNode);
    expect(material.fragmentNode).toBe(fragmentNode);
    expect(material.version).toBe(version);
  });
});
