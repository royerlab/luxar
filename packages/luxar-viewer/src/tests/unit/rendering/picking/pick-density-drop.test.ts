/**
 * A thinned element can neither be seen nor picked
 * (`materials/_shared/density-drop.ts`): every pick source the density guard
 * thins must apply the drop its visual twin applies, on both backends.
 *
 * The pick pass mirrors `uDensityDrop` onto each pick material
 * (`picking-system/visibility-sync.ts`), but the uniform only matters if the
 * pick shader acts on it. Including `GLSL_SORTED_INDEX` DECLARES
 * `luxarDensityDropped()`; only a call culls anything. Mesh is not in the
 * table: the density guard never thins a mesh, so neither of its shaders
 * declares the uniform.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { GSPLAT_PICK_VERTEX_SHADER } from '../../../../rendering/picking/gsplat/shaders';
import { LINE_PICK_VERTEX_SHADER } from '../../../../rendering/picking/line/shaders';
import { CAPSULE_LINE_PICK_VERTEX_SHADER } from '../../../../rendering/picking/line/shaders-capsule';
import { POINT_PICK_VERTEX_SHADER } from '../../../../rendering/picking/point/shaders';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const src = (rel: string): string =>
  readFileSync(path.resolve(HERE, '../../../../rendering/picking', rel), 'utf8');

/** Calls of `luxarDensityDropped()` — not its `bool luxarDensityDropped() {` definition. */
const glslDropCalls = (shader: string): number =>
  [...shader.matchAll(/luxarDensityDropped\(\)(?!\s*\{)/g)].length;

describe('every GLSL pick vertex shader culls a density-dropped element', () => {
  it.each([
    ['point', POINT_PICK_VERTEX_SHADER],
    ['gsplat', GSPLAT_PICK_VERTEX_SHADER],
    ['line', LINE_PICK_VERTEX_SHADER],
    ['line capsule', CAPSULE_LINE_PICK_VERTEX_SHADER],
  ])('%s', (_name, shader) => {
    expect(glslDropCalls(shader)).toBeGreaterThan(0);
  });
});

describe('every TSL pick factory culls a density-dropped element', () => {
  it.each([
    ['point/pick.tsl.ts'],
    ['gsplat/pick.tsl.ts'],
    ['line/pick.tsl.ts'],
    ['line/pick-capsule.tsl.ts'],
  ])('%s', (file) => {
    const tsl = src(file);
    expect(tsl).toContain('densityDroppedNode(nodes.uDensityDrop, aSortedIndex)');
    // The node is consumed, not merely built (an unused TSL node emits nothing).
    expect([...tsl.matchAll(/\bdensityDropped\b/g)].length).toBeGreaterThan(1);
  });
});
