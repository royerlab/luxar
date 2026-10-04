/**
 * Geometry-behaviour rows `lodLevel` and `partitionPart`: which types may be
 * the `display_type` of a `kind=lod` / `kind=partition` group.
 *
 * The capability itself lives in `GEOMETRY_CAPABILITIES` (and its Python
 * mirror), and the matrix meta-check pins these two rows to it cell for cell.
 * The probe drives a real consumer of the flag: the monitor's scene-graph
 * converter, which admits a group's `display_type` only when the type can back
 * that container — for `kind=lod` it is the only gate in the pipeline, since
 * the writer derives the attr from the children without validating it. A
 * `hidden` cell would assert the converter leaves the type off the tree.
 */
import { expect } from 'vitest';

import type { SceneNode } from '../../../data/data-loader-types';
import { convertToSceneGraphNode } from '../../../data/scene-loader/monitor/scene-graph-converter';
import type { GeometryTypeName } from '../../../types/format-contract';
import { defineBehaviourConformance } from '../../_conformance/define-behaviour-conformance';

/** The monitor-tree node for a `kind` group of two `type` children. */
function wrapperFor(kind: 'lod' | 'partition', type: GeometryTypeName) {
  const child = (i: number): SceneNode => ({
    path: `/g/${i}`,
    type,
    attrs: {},
    hasSpatialIndex: false,
  });
  return convertToSceneGraphNode({
    path: '/g',
    type: 'group',
    attrs: { kind, display_type: type },
    hasSpatialIndex: false,
    children: [child(0), child(1)],
  });
}

for (const [id, kind] of [
  ['lodLevel', 'lod'],
  ['partitionPart', 'partition'],
] as const) {
  defineBehaviourConformance(id, {
    holds(type) {
      const node = wrapperFor(kind, type);
      expect(node.kind).toBe(kind);
      expect(node.displayType).toBe(type);
    },
    enforced: {
      hidden: (type) => {
        expect(wrapperFor(kind, type).displayType).toBeUndefined();
      },
    },
  });
}
