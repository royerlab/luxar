import { describe, expect, it } from 'vitest';
import { mul, vec4 } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { setTSLNodeOverride } from '../../../../index';
import { getSharedTSLGraph } from '../../../../rendering/materials/_shared/shared-graph-tsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { MeshTSLMaterial } from '../../../../rendering/materials/mesh/material-tsl';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import { MaterialManager } from '../../../../rendering/material-manager';
import { loadTslMaterials } from '../../../../rendering/tsl/load';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';
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

  it('refuses a shared-graph picking material', () => {
    const pick = new PointPickingTSLMaterial({ nodeId: 1 });
    const baseColor = pick.colorNode;
    expect(() => setTSLNodeOverride(pick, () => ({ colorNode: vec4(1, 0, 0, 1) }))).toThrow(
      /requires a Luxar shared-graph TSL material/
    );
    expect(pick.colorNode).toBe(baseColor);
    expect(pick.clone().colorNode).toBe(baseColor);
  });

  it('overrides a manager-created WebGPU material without changing its peer', async () => {
    await loadTslMaterials();
    const manager = new MaterialManager();
    manager.setCaps({ apiSurface: 'webgpu' } as RendererCapabilities);
    const props = { blendingMode: 'additive' as const, opacity: 1, gamma: 1, intensity: 1, offset: 0 };
    try {
      const material = manager.getPointMaterial(props);
      const peer = manager.getPointMaterial(props);
      const baseColor = (material as PointTSLMaterial).colorNode;
      expect((peer as PointTSLMaterial).colorNode).toBe(baseColor);
      const replacement = vec4(0, 1, 0, 1);
      setTSLNodeOverride(material, () => ({ colorNode: replacement }));
      expect((material as PointTSLMaterial).colorNode).toBe(replacement);
      expect((peer as PointTSLMaterial).colorNode).toBe(baseColor);
    } finally {
      manager.dispose();
    }
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
    material.updateGamma(1);
    expect(calls).toBe(3);
    expect(material.colorNode).toBe(replacement);
    setTSLNodeOverride(material, null);
    expect(material.colorNode).toBe(getSharedTSLGraph(material)?.colorNode);
  });
});
