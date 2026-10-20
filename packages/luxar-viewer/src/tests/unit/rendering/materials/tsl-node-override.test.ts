import { describe, expect, it } from 'vitest';
import { mul, vec4 } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { setTSLNodeOverride } from '../../../../index';
import { getSharedTSLGraph } from '../../../../rendering/materials/_shared/shared-graph-tsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { MeshTSLMaterial } from '../../../../rendering/materials/mesh/material-tsl';
import type { GeometryTypeName } from '../../../../types/format-contract';
import { defineBehaviourConformance } from '../../../_conformance/define-behaviour-conformance';

const visualMaterials = {
  points: () => new PointTSLMaterial(),
  lines: () => new LineTSLMaterial(),
  gsplats: () => new GSplatTSLMaterial(),
  mesh: () => new MeshTSLMaterial(),
} satisfies Record<GeometryTypeName, () => NodeMaterial>;

describe('public TSL node override for built-in materials', () => {
  defineBehaviourConformance('tslNodeOverride', {
    holds(type) {
      const make = visualMaterials[type];
      const material = make();
      const peer = make();
      const baseColor = material.colorNode;
      expect(peer.colorNode).toBe(baseColor);

      const replacement = vec4(0.25, 0.5, 0.75, 1);
      const seen: (typeof baseColor)[] = [];
      const override = ({ colorNode }: { colorNode: typeof baseColor }) => {
        seen.push(colorNode);
        return { colorNode: replacement };
      };
      setTSLNodeOverride(material, override);
      expect(seen).toEqual([baseColor]);
      expect(material.colorNode).toBe(replacement);
      expect(peer.colorNode).toBe(baseColor);
      expect(material.customProgramCacheKey()).not.toBe(peer.customProgramCacheKey());

      material.updateGamma(2);
      expect(seen).toHaveLength(2);
      expect(seen[1]).not.toBeNull();
      expect(material.colorNode).toBe(replacement);
      const clone = material.clone();
      expect(clone.colorNode).toBe(replacement);
      expect(clone.customProgramCacheKey()).toBe(material.customProgramCacheKey());

      setTSLNodeOverride(material, null);
      expect(material.colorNode).not.toBe(replacement);
      expect(material.colorNode).toBe(getSharedTSLGraph(material)?.colorNode);
      expect(clone.colorNode).toBe(replacement);
    },
  });

  it('refuses a material outside Luxar’s built-in shared graphs', () => {
    expect(() => setTSLNodeOverride(new NodeMaterial(), () => ({}))).toThrow(
      /requires a Luxar shared-graph TSL material/
    );
  });

  it('composes a TSL output with the original graph', () => {
    const material = new PointTSLMaterial();
    const baseColor = material.colorNode;
    setTSLNodeOverride(material, ({ colorNode }) => {
      expect(colorNode).toBe(baseColor);
      return { colorNode: mul(colorNode, vec4(1, 0.5, 1, 1)) };
    });
    expect(material.colorNode).not.toBe(baseColor);
  });

  it('keeps the prior override if a replacement throws, then restores the new base on rebuild failure', () => {
    const material = new PointTSLMaterial();
    const replacement = vec4(1, 0, 0, 1);
    setTSLNodeOverride(material, () => ({ colorNode: replacement }));
    expect(() =>
      setTSLNodeOverride(material, () => {
        throw new Error('bad extension');
      })
    ).toThrow('bad extension');
    expect(material.colorNode).toBe(replacement);

    let calls = 0;
    setTSLNodeOverride(material, () => {
      if (++calls === 2) throw new Error('failed rebuild');
      return { colorNode: replacement };
    });
    expect(() => material.updateGamma(2)).toThrow('failed rebuild');
    expect(material.colorNode).toBe(getSharedTSLGraph(material)?.colorNode);
    setTSLNodeOverride(material, null);
    expect(material.colorNode).toBe(getSharedTSLGraph(material)?.colorNode);
  });
});
