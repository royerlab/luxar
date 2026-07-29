/**
 * Registry assembly + re-export surface for the TSL ↔ GLSL parity
 * harness. The page entry (`../tsl-harness.ts`, loaded by
 * `tsl-harness.html`) imports everything it needs from here.
 *
 * The 61 registry entries live in four shader-family modules —
 * `post-processing.ts` (8), `points.ts` (18), `lines.ts` (17),
 * `gsplats.ts` (18) — and are merged here into the single
 * `SHADER_REGISTRY` the Playwright parity/codegen specs drive by name.
 *
 * @module tests/e2e/harnesses/tsl-harness/index
 */

import type { RegistryEntry } from './types';
import { POST_PROCESSING_SHADERS } from './post-processing';
import { POINT_SHADERS } from './points';
import { LINE_SHADERS } from './lines';
import { GSPLAT_SHADERS } from './gsplats';

export const SHADER_REGISTRY: Record<string, RegistryEntry> = {
  ...POST_PROCESSING_SHADERS,
  ...POINT_SHADERS,
  ...LINE_SHADERS,
  ...GSPLAT_SHADERS,
};

export { HARNESS_SIZE, renderGLSL, renderTSL } from './render';
export type { RegistryEntry } from './types';
