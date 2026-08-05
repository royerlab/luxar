/**
 * Registry assembly + re-export surface for the TSL ↔ GLSL parity
 * harness. The page entry (`../tsl-harness.ts`, loaded by
 * `tsl-harness.html`) imports everything it needs from here.
 *
 * The 73 registry entries live in five shader-family modules —
 * `post-processing.ts` (8), `points.ts` (18), `lines.ts` (21),
 * `gsplats.ts` (18), `mesh.ts` (8) — and are merged here into the single
 * `SHADER_REGISTRY` the Playwright parity/codegen specs drive by name.
 *
 * The registry is deliberately WIDER than the codegen snapshot list: an entry earns
 * a snapshot only when it generates distinct shader code, while an entry that
 * exercises a distinct BINDING or a distinct RUNTIME-UNIFORM ARM belongs in the
 * parity spec alone. `mesh-rgb`'s size-3 colour attribute and
 * `mesh-pick-commutative`'s cutout-off state each compile to a shader byte-identical
 * to one already snapshotted, so a second snapshot would pin nothing new — while the
 * rendered pixels differ, which is what those entries are for.
 *
 * @module tests/e2e/harnesses/tsl-harness/index
 */

import type { RegistryEntry } from './types';
import { POST_PROCESSING_SHADERS } from './post-processing';
import { POINT_SHADERS } from './points';
import { LINE_SHADERS } from './lines';
import { GSPLAT_SHADERS } from './gsplats';
import { MESH_SHADERS } from './mesh';

export const SHADER_REGISTRY: Record<string, RegistryEntry> = {
  ...POST_PROCESSING_SHADERS,
  ...POINT_SHADERS,
  ...LINE_SHADERS,
  ...GSPLAT_SHADERS,
  ...MESH_SHADERS,
};

export { HARNESS_SIZE, renderGLSL, renderTSL } from './render';
export type { RegistryEntry } from './types';
