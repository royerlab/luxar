/**
 * Every visual and pick material of every geometry type, per primitive
 * variant and per backend — the constructor table the draw/pick conformance
 * probes iterate.
 *
 * A "variant" is a shader-source pair chosen at construction: lines build a
 * screen-space quad or a capsule program; the other types have one shape each.
 * `Record<GeometryTypeName, …>` makes a new geometry type a compile error here,
 * and the draw/pick rule table (`tests/_conformance/pick-visibility-rules.ts`)
 * is keyed by the same variants, so a new variant shows up in both.
 *
 * @module tests/helpers/geometry-materials
 */

import type * as THREE from 'three';

import { GSplatMaterial } from '../../rendering/materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from '../../rendering/materials/gsplat/material-tsl';
import { LineMaterial } from '../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../rendering/materials/line/material-tsl';
import { MeshMaterial } from '../../rendering/materials/mesh/material-glsl';
import { MeshTSLMaterial } from '../../rendering/materials/mesh/material-tsl';
import { PointMaterial } from '../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../rendering/materials/point/material-tsl';
import { GSplatPickingMaterial } from '../../rendering/picking/gsplat/material';
import { GSplatPickingTSLMaterial } from '../../rendering/picking/gsplat/material-tsl';
import { LinePickingMaterial } from '../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../rendering/picking/line/material-tsl';
import { MeshPickingMaterial } from '../../rendering/picking/mesh/material';
import { MeshPickingTSLMaterial } from '../../rendering/picking/mesh/material-tsl';
import { PointPickingMaterial } from '../../rendering/picking/point/material';
import { PointPickingTSLMaterial } from '../../rendering/picking/point/material-tsl';
import type { MaterialBackend } from '../../rendering/material-manager/factories';
import type { GeometryTypeName } from '../../types/format-contract';
import type { BlendingMode } from '../../types/blending';

/** The primitive a variant draws and picks. */
export type PrimitiveVariant = 'quad' | 'capsule' | 'triangle';

/** One variant's four materials: visual and pick, on each backend. */
export interface MaterialVariant {
  readonly variant: PrimitiveVariant;
  readonly visual: Readonly<Record<MaterialBackend, (mode?: BlendingMode) => THREE.Material>>;
  readonly pick: Readonly<Record<MaterialBackend, () => THREE.Material>>;
}

const PICK = { nodeId: 1 };

/** Every variant of every geometry type. */
export const GEOMETRY_MATERIAL_VARIANTS: Readonly<
  Record<GeometryTypeName, readonly MaterialVariant[]>
> = {
  points: [
    {
      variant: 'quad',
      visual: {
        glsl: (blendingMode) => new PointMaterial({ blendingMode }),
        tsl: (blendingMode) => new PointTSLMaterial({ blendingMode }),
      },
      pick: {
        glsl: () => new PointPickingMaterial(PICK),
        tsl: () => new PointPickingTSLMaterial(PICK),
      },
    },
  ],
  lines: (['screen-space', 'capsule'] as const).map((primitive) => ({
    variant: primitive === 'capsule' ? 'capsule' : 'quad',
    visual: {
      glsl: (blendingMode?: BlendingMode) => new LineMaterial({ blendingMode, primitive }),
      tsl: (blendingMode?: BlendingMode) => new LineTSLMaterial({ blendingMode, primitive }),
    },
    pick: {
      glsl: () => new LinePickingMaterial({ ...PICK, primitive }),
      tsl: () => new LinePickingTSLMaterial({ ...PICK, primitive }),
    },
  })),
  gsplats: [
    {
      variant: 'quad',
      visual: {
        glsl: (blendingMode) => new GSplatMaterial({ blendingMode }),
        tsl: (blendingMode) => new GSplatTSLMaterial({ blendingMode }),
      },
      pick: {
        glsl: () => new GSplatPickingMaterial(PICK),
        tsl: () => new GSplatPickingTSLMaterial(PICK),
      },
    },
  ],
  mesh: [
    {
      variant: 'triangle',
      visual: {
        glsl: (blendingMode) => new MeshMaterial({ blendingMode }),
        tsl: (blendingMode) => new MeshTSLMaterial({ blendingMode }),
      },
      pick: {
        glsl: () => new MeshPickingMaterial(PICK),
        tsl: () => new MeshPickingTSLMaterial(PICK),
      },
    },
  ],
};

/** Every material of `type`, labelled `<variant> <visual|pick> <backend>`. */
export function allMaterialsOf(type: GeometryTypeName): [string, THREE.Material][] {
  const out: [string, THREE.Material][] = [];
  for (const v of GEOMETRY_MATERIAL_VARIANTS[type]) {
    for (const backend of ['glsl', 'tsl'] as const) {
      out.push([`${v.variant} visual ${backend}`, v.visual[backend]()]);
      out.push([`${v.variant} pick ${backend}`, v.pick[backend]()]);
    }
  }
  return out;
}
