/**
 * TSL pick depth does not depend on the camera kind the graph was BUILT under.
 *
 * Under the surface depth convention (opaque / normal) every pick graph writes the
 * fragment's real depth, and front-most wins. three's `depth` node picks
 * `viewZToPerspectiveDepth` or `viewZToOrthographicDepth` from
 * `camera.isPerspectiveCamera` when the graph is built. three's node-build cache key
 * does not include the camera kind, so a build is reused across cameras: a graph
 * first built under a perspective camera kept the perspective formula under an
 * orthographic one. Graphs built under different kinds then compared different
 * formulas in one pick depth buffer, and front-most picking broke after a
 * perspective/ortho switch, or for a node first drawn after one.
 *
 * The observable without a GPU is the generated WGSL. A real `WGSLNodeBuilder`
 * builds each pick material once under a perspective camera and once under an
 * orthographic one. The two fragment shaders must be identical, so a build reused
 * across kinds is correct for both, and must carry the orthographic formula (the
 * branch is taken at run time from the projection matrix).
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { MeshPickingTSLMaterial } from '../../../../rendering/picking/mesh/material-tsl';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import { buildWGSL } from '../wgsl-builder';

/** viewZToOrthographicDepth as three emits it. */
const ORTHO_DEPTH =
  /\(\s*(?:positionView\.z|nodeVar\d+) \+ render\.cameraNear\s*\)\s*\/\s*\(\s*render\.cameraNear - render\.cameraFar\s*\)/;

describe('TSL pick depth is chosen per draw, not per build', () => {
  it.each([
    ['point', () => new PointPickingTSLMaterial({ nodeId: 1 })],
    ['line quad', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' })],
    ['line capsule', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'capsule' })],
    ['gsplat', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
    ['mesh', () => new MeshPickingTSLMaterial({ nodeId: 1 })],
  ])('%s: a perspective build writes the ortho depth under an ortho camera', (_n, make) => {
    const persp = buildWGSL(make(), new THREE.PerspectiveCamera(50, 1, 0.1, 100)).fragmentShader;
    const ortho = buildWGSL(
      make(),
      new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100)
    ).fragmentShader;
    expect(persp).toContain('frag_depth');
    expect(persp).toMatch(ORTHO_DEPTH);
    // The inverse projection feeding both select arms must run before the
    // projection-kind branch; otherwise the perspective arm reads an unset var.
    const inverseProjection = persp.indexOf('render.cameraProjectionMatrixInverse * v_clipSpace');
    const projectionKindBranch = persp.indexOf(
      'render.cameraProjectionMatrix[ 3 ].w > 0.5',
      inverseProjection
    );
    expect(inverseProjection).toBeGreaterThan(-1);
    expect(projectionKindBranch).toBeGreaterThan(inverseProjection);
    const flowBeforeInverse = persp.slice(persp.indexOf('@fragment\nfn main'), inverseProjection);
    expect(
      (flowBeforeInverse.match(/\{/g) ?? []).length - (flowBeforeInverse.match(/\}/g) ?? []).length
    ).toBe(1);
    expect(persp).toBe(ortho);
  });
});
