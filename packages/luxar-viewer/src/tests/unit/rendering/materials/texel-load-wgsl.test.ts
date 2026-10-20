import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import { wgslVertexFor } from '../../helpers/wgsl-build';

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('aSortedIndex', new THREE.Uint32BufferAttribute([0], 1));
geometry.setAttribute('aSortedIndexB', new THREE.Uint32BufferAttribute([0], 1));
geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute([0, 0], 2));

describe('element texture loads in generated WGSL', () => {
  it('bounds the six line texel coordinates once per segment', () => {
    const vertex = wgslVertexFor(
      new LineTSLMaterial({ primitive: 'screen-space' }),
      new THREE.PerspectiveCamera(50, 1, 0.1, 100),
      geometry
    );
    expect(vertex).toContain('textureDimensions(');
    expect(vertex).toMatch(/clamp\( \( nodeVar\d+ % nodeVar\d+ \), 0, \( nodeVar\d+ - 6 \) \)/);
    expect(vertex).toMatch(/clamp\( \( nodeVar\d+ \/ nodeVar\d+ \), 0, \( nodeVar\d+ - 1 \) \)/);
  });

  // geometry-subset: Mesh reads vertex buffers rather than element data textures.
  it.each([
    ['gsplat', () => new GSplatTSLMaterial(), 4], // splatT0..T3
    ['point', () => new PointTSLMaterial(), 3], // pointT0..T2, including color data
    ['line quad', () => new LineTSLMaterial({ primitive: 'screen-space' }), 10], // 6 segment + 2x2 partner
    ['line capsule', () => new LineTSLMaterial({ primitive: 'capsule' }), 10], // 6 segment + 2x2 partner
    ['gsplat pick', () => new GSplatPickingTSLMaterial({ nodeId: 1 }), 4], // splatT0..T3
    ['point pick', () => new PointPickingTSLMaterial({ nodeId: 1 }), 3], // pointT0..T2
    [
      'line quad pick',
      () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' }),
      10,
    ], // 6 segment + 2x2 partner
    [
      'line capsule pick',
      () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'capsule' }),
      10,
    ], // 6 segment + 2x2 partner
  ] as const)('%s uses untransformed integer texel coordinates', (_name, make, loadCount) => {
    const vertex = wgslVertexFor(make(), new THREE.PerspectiveCamera(50, 1, 0.1, 100), geometry);
    const loads = vertex.split('\n').filter((line) => line.includes('textureLoad('));
    expect(loads).toHaveLength(loadCount);
    for (const load of loads) {
      expect(load).not.toContain('vec2<f32>(');
    }
  });
});
