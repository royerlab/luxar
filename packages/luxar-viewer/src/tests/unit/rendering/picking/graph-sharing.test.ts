/**
 * Pick TSL materials of one configuration share one node graph, as their
 * visual twins do (#2992; see materials/graph-sharing-cases.ts for why the
 * cache key is the observable). The node id is a per-material VALUE; the
 * baked element-texture width, the line primitive / join variant and the
 * mesh texture presence select code (the projection is read per draw — an
 * ortho flip keeping the graph is pinned in ortho-from-projection.test.ts).
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { MeshPickingTSLMaterial } from '../../../../rendering/picking/mesh/material-tsl';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { setLineJoinOverride } from '../../../../types/line-join';
import { describeGraphSharing } from '../materials/graph-sharing-cases';

const nodeId = (config: Record<string, unknown>): number => (config.nodeId as number) ?? 1;

describeGraphSharing('GSplatPickingTSLMaterial', {
  make: (config = {}) => new GSplatPickingTSLMaterial({ nodeId: nodeId(config) }),
  valueConfigs: [{ nodeId: 3 }, { nodeId: 9 }],
  variants: [],
  bindElementTexture: (m, tex) => m.updateSplatTexture(tex),
});

describeGraphSharing('PointPickingTSLMaterial', {
  make: (config = {}) => new PointPickingTSLMaterial({ nodeId: nodeId(config) }),
  valueConfigs: [{ nodeId: 3 }, { nodeId: 9, radiusScale: 2 }],
  variants: [],
  bindElementTexture: (m, tex) => m.updatePointTexture(tex),
});

describeGraphSharing('LinePickingTSLMaterial', {
  make: (config = {}) => new LinePickingTSLMaterial({ ...config, nodeId: nodeId(config) }),
  valueConfigs: [{ nodeId: 3 }, { nodeId: 9 }],
  variants: [
    ['screen-space quad', { primitive: 'screen-space' }],
    ['no join', { join: 'none' }],
  ],
  bindElementTexture: (m, tex) => m.updateLineTexture(tex),
});

describeGraphSharing('MeshPickingTSLMaterial', {
  make: (config = {}) => new MeshPickingTSLMaterial({ ...config, nodeId: nodeId(config) }),
  valueConfigs: [
    { nodeId: 3, opacity: 0.4 },
    { nodeId: 9, alphaCutoff: 0.2 },
  ],
  variants: [['base-colour texture', { baseColorTexture: new THREE.Texture() }]],
});

describe('pick materials keep the pick-pass state under a shared graph', () => {
  it.each([
    ['gsplat', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
    ['point', () => new PointPickingTSLMaterial({ nodeId: 1 })],
    ['line', () => new LinePickingTSLMaterial({ nodeId: 1 })],
    ['mesh', () => new MeshPickingTSLMaterial({ nodeId: 1 })],
  ])('%s: opaque ID buffer, opacity pinned to 1, depth written', (_n, make) => {
    const m = make();
    expect(m.blending).toBe(THREE.NoBlending);
    expect(m.opacity).toBe(1);
    expect(m.transparent).toBe(false);
    expect(m.depthWrite).toBe(true);
    expect(m.toneMapped).toBe(false);
    // The depth convention is part of the shared graph.
    expect(m.depthNode).toBeTruthy();
  });
});

/**
 * The line factories resolve the session `?lineJoin=` override at BUILD time
 * (`resolveLineJoin`), so the resolved join selects code: an unauthored line
 * built under the override must not reuse the graph an unauthored line built
 * without it — the visual twin's key already carries `resolvedJoin`.
 */
describe('line TSL graphs key on the RESOLVED join (?lineJoin= override)', () => {
  it.each([
    ['visual quad', () => new LineTSLMaterial({ primitive: 'screen-space' })],
    ['pick quad', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' })],
  ])('%s: an override changes the shared graph of an unauthored join', (_n, make) => {
    const miter = make();
    try {
      setLineJoinOverride('none');
      const overridden = make();
      expect(overridden.customProgramCacheKey()).not.toBe(miter.customProgramCacheKey());
      expect(overridden.vertexNode).not.toBe(miter.vertexNode);
    } finally {
      setLineJoinOverride(null);
    }
  });
});
