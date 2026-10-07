/**
 * Every GLSL shader takes its ortho branch from the projection matrix of the
 * draw (`luxarIsOrthoProjection()`, glsl-lib), not from a CPU-pushed flag.
 *
 * Points and gsplats always did. Lines read a broadcast `uIsOrtho` uniform —
 * the camera TYPE of the main view, pushed by `updateCameraParams` — so a draw
 * through a different projection (the environment capture's cube faces, an
 * embedder camera) ran the wrong branch unless every such caller remembered to
 * re-push and restore it (the scene capture flips it twice per capture under
 * an ortho camera). Mesh read three's `isOrthographic` camera flag in its
 * fragment stage, which is the camera's class, not its matrix. Both now derive
 * the flag from P (the line visual fragment re-reads P itself; the others get
 * the vertex stage's flag as a flat varying), so the twins agree with each other and with their TSL
 * counterparts' `isOrthoProjectionTSL()`.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GLSL_LINE_JOIN } from '../../../../rendering/materials/_shared/glsl-lib';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import {
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-capsule';
import {
  LINE_FRAGMENT_SHADER,
  LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl';
import { MESH_FRAGMENT_SHADER } from '../../../../rendering/materials/mesh/shader-glsl';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import {
  CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  CAPSULE_LINE_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/line/shaders-capsule';
import {
  LINE_PICK_FRAGMENT_SHADER,
  LINE_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/line/shaders';
import { MESH_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/mesh/shaders';

const LINE_SHADERS: Record<string, string> = {
  'visual quad vertex': LINE_VERTEX_SHADER,
  'visual quad fragment': LINE_FRAGMENT_SHADER,
  'visual capsule vertex': CAPSULE_LINE_VERTEX_SHADER,
  'visual capsule fragment': CAPSULE_LINE_FRAGMENT_SHADER,
  'pick quad vertex': LINE_PICK_VERTEX_SHADER,
  'pick quad fragment': LINE_PICK_FRAGMENT_SHADER,
  'pick capsule vertex': CAPSULE_LINE_PICK_VERTEX_SHADER,
  'pick capsule fragment': CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  'shared join helpers': GLSL_LINE_JOIN,
};

describe('GLSL line shaders derive ortho from the projection matrix', () => {
  it.each(Object.entries(LINE_SHADERS))('%s reads no uIsOrtho uniform', (_n, src) => {
    expect(src).not.toMatch(/\buIsOrtho\b/);
  });

  it.each([
    ['visual quad', LINE_VERTEX_SHADER],
    ['visual capsule', CAPSULE_LINE_VERTEX_SHADER],
    ['pick quad', LINE_PICK_VERTEX_SHADER],
    ['pick capsule', CAPSULE_LINE_PICK_VERTEX_SHADER],
  ])('%s vertex stage takes it from luxarIsOrthoProjection()', (_n, vs) => {
    expect(vs).toContain('luxarLineIsOrtho = luxarIsOrthoProjection();');
  });

  it('visual quad fragment re-derives it from P instead of a flat varying', () => {
    // Carrying the vertex stage's flag in one more flat varying cost the
    // 10M-segment draw ~1.4% GPU time; the fragment stage can read the same
    // projection uniform instead, which gives the same branch by construction.
    expect(LINE_VERTEX_SHADER).not.toMatch(/\bvLineIsOrtho\b/);
    expect(LINE_FRAGMENT_SHADER).not.toMatch(/\bvLineIsOrtho\b/);
    expect(LINE_FRAGMENT_SHADER).toContain('uniform mat4 projectionMatrix;');
    expect(LINE_FRAGMENT_SHADER).toContain('perspectiveNearFade(luxarIsOrthoProjection(), vViewZ');
  });

  it.each([
    ['LineMaterial', () => new LineMaterial({})],
    ['LinePickingMaterial', () => new LinePickingMaterial({ nodeId: 1 })],
  ])('%s binds no uIsOrtho uniform', (_n, make) => {
    expect(make().uniforms.uIsOrtho).toBeUndefined();
  });
});

describe('GLSL mesh shaders derive ortho from the projection matrix', () => {
  it.each([
    ['visual', MESH_FRAGMENT_SHADER],
    ['pick', MESH_PICK_FRAGMENT_SHADER],
  ])('%s fragment reads no isOrthographic camera flag', (_n, fs) => {
    expect(fs).not.toMatch(/\bisOrthographic\b/);
    expect(fs).toContain('flat in int vIsOrtho;');
  });
});

/**
 * The TSL line twins read the same per-draw ortho test (`isOrthoProjectionTSL()`),
 * so the camera kind is not an input at all: `updateCameraParams` takes no
 * projection flag (a boolean second argument is a type error), and a material
 * keeps its graph (and the program cache key it shares with every material of
 * its configuration) across camera pushes. Before, `isOrtho` was a build-time
 * graph variant: each flip rebuilt the graph and moved the material to another
 * shared-graph key — twice per scene-environment capture under an ortho camera
 * (the capture pushed a perspective cube camera, then restored).
 */
describe('TSL line materials derive ortho from the projection matrix', () => {
  const res = new THREE.Vector2(800, 600);
  it.each([
    ['visual quad', () => new LineTSLMaterial({ primitive: 'screen-space' })],
    ['visual capsule', () => new LineTSLMaterial({ primitive: 'capsule' })],
    ['pick quad', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' })],
    ['pick capsule', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'capsule' })],
  ])('%s keeps its graph and cache key across camera pushes', (_n, make) => {
    const m = make();
    m.updateCameraParams(res);
    const key = m.customProgramCacheKey();
    const vertex = m.vertexNode;
    // These wrappers write `colorNode` (their `fragmentNode` is always null).
    const color = m.colorNode;
    expect(color).toBeTruthy();
    m.updateCameraParams(new THREE.Vector2(512, 512), 0.25, 2);
    expect(m.customProgramCacheKey()).toBe(key);
    expect(m.vertexNode).toBe(vertex);
    expect(m.colorNode).toBe(color);
    // @ts-expect-error — the camera kind is not an input (read per draw from P).
    m.updateCameraParams(res, true);
    expect(m.customProgramCacheKey()).toBe(key);
    expect(m.vertexNode).toBe(vertex);
    expect(m.colorNode).toBe(color);
  });

  it.each([
    ['LineTSLMaterial', () => new LineTSLMaterial({})],
    ['LinePickingTSLMaterial', () => new LinePickingTSLMaterial({ nodeId: 1 })],
  ])('%s binds no uIsOrtho uniform', (_n, make) => {
    expect(make().uniforms.uIsOrtho).toBeUndefined();
  });
});
