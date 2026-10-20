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
    const colorNode = material.colorNode;

    material.applyBlendingMode('luminous');
    expect(material.depthTest).toBe(true);
    expect(material.blending).toBe(THREE.AdditiveBlending);
    expect(material.vertexNode).toBe(vertexNode);
    expect(material.colorNode).toBe(colorNode);
    expect(material.version).toBe(version);

    material.applyBlendingMode('additive');
    expect(material.depthTest).toBe(false);
    expect(material.vertexNode).toBe(vertexNode);
    expect(material.colorNode).toBe(colorNode);
    expect(material.version).toBe(version);
  });
});

// geometry-subset: mesh has no normal-only density alpha branch.
describe.each(materials.filter(([name]) => name === 'mesh'))(
  '%s TSL blend factors',
  (_name, make) => {
    it('switches normal ↔ luminous without invalidating the node graph', () => {
      const material = make();
      material.applyBlendingMode('normal');
      const version = material.version;
      const colorNode = material.colorNode;

      material.applyBlendingMode('luminous');
      expect(material.blending).toBe(THREE.AdditiveBlending);
      expect(material.colorNode).toBe(colorNode);
      expect(material.version).toBe(version);

      material.applyBlendingMode('normal');
      expect(material.blending).toBe(THREE.NormalBlending);
      expect(material.colorNode).toBe(colorNode);
      expect(material.version).toBe(version);
    });
  }
);

// geometry-subset: point and line normal use the density alpha graph branch.
describe.each([
  ['points', () => new PointTSLMaterial({ blendingMode: 'normal' }), () => new PointTSLMaterial()],
  ['lines', () => new LineTSLMaterial({ blendingMode: 'normal' }), () => new LineTSLMaterial()],
] as const)('%s TSL normal graph', (_name, makeNormal, makeAdditive) => {
  it('switches to the same normal graph as a fresh material and back to additive', () => {
    const freshNormal = makeNormal();
    const material = makeAdditive();
    const additiveGraph = material.colorNode;

    material.applyBlendingMode('normal');
    expect(material.colorNode).toBe(freshNormal.colorNode);
    expect(material.colorNode).not.toBe(additiveGraph);

    material.applyBlendingMode('luminous');
    expect(material.colorNode).not.toBe(freshNormal.colorNode);

    material.applyBlendingMode('normal');
    expect(material.colorNode).toBe(freshNormal.colorNode);

    material.applyBlendingMode('additive');
    expect(material.colorNode).toBe(additiveGraph);
  });
});

it('gsplats switch max ↔ opaque blend equations without rebuilding their peak graph', () => {
  const material = new GSplatTSLMaterial({ blendingMode: 'max' });
  const version = material.version;
  const vertexNode = material.vertexNode;
  const colorNode = material.colorNode;

  material.applyBlendingMode('opaque');
  expect(material.blendEquation).toBe(THREE.AddEquation);
  expect(material.depthWrite).toBe(true);
  expect(material.vertexNode).toBe(vertexNode);
  expect(material.colorNode).toBe(colorNode);
  expect(material.version).toBe(version);

  material.applyBlendingMode('max');
  expect(material.blendEquation).toBe(THREE.MaxEquation);
  expect(material.depthWrite).toBe(false);
  expect(material.vertexNode).toBe(vertexNode);
  expect(material.colorNode).toBe(colorNode);
  expect(material.version).toBe(version);
});
