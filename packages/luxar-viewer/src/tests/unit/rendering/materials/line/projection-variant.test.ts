/**
 * The TSL screen-space quad (visual + pick) carries a COMPILE-TIME projection
 * variant again — a single runtime-ortho graph measured +4.6% of the GPU pass
 * on the 10M-segment ortho quad under WebGPU, recovered in full only by a
 * constant ortho test — but the variant is chosen PER DRAW from the drawn
 * camera's projection matrix (the line mesh's `onBeforeRender`), never pushed
 * from the CPU. So a draw through another projection kind (the scene
 * environment capture's cube faces under an orthographic main camera) gets the
 * right variant by itself, and alternating kinds only re-points the material at
 * one of two cached shared graphs: no new graph is built.
 *
 * The capsule keeps its runtime switch (neutral in the same measurement).
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';
import { installProjectionVariantHook } from '../../../../../rendering/materials/_shared/projection-variant';
import { sharedTSLGraphCount } from '../../../../../rendering/materials/_shared/shared-graph-tsl';
import { LinePickingTSLMaterial } from '../../../../../rendering/picking/line/material-tsl';

const persp = (): THREE.Camera => new THREE.PerspectiveCamera(50, 1, 0.1, 100);
const ortho = (): THREE.Camera => new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);

type Drawable = THREE.Material & {
  vertexNode: unknown;
  colorNode: unknown;
  customProgramCacheKey(): string;
  updateCameraParams(res: THREE.Vector2, nearCull?: number, pixelRatio?: number): void;
};

/** Run the mesh's per-draw hook exactly as three's renderer does before a draw. */
function draw(mesh: THREE.Mesh, camera: THREE.Camera): void {
  mesh.onBeforeRender(
    {} as THREE.WebGLRenderer,
    new THREE.Scene(),
    camera,
    mesh.geometry,
    mesh.material as THREE.Material,
    null as unknown as THREE.Group
  );
}

function meshWith(material: Drawable): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
  installProjectionVariantHook(mesh);
  return mesh;
}

describe.each([
  ['visual quad', () => new LineTSLMaterial({ primitive: 'screen-space' }) as unknown as Drawable],
  [
    'pick quad',
    () =>
      new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' }) as unknown as Drawable,
  ],
])('%s: projection variant chosen per draw from the camera', (_n, make) => {
  it('draws through an ortho camera with a different graph than a perspective one', () => {
    const m = make();
    const mesh = meshWith(m);
    draw(mesh, persp());
    const perspVertex = m.vertexNode;
    const perspKey = m.customProgramCacheKey();
    draw(mesh, ortho());
    expect(m.vertexNode).not.toBe(perspVertex);
    expect(m.customProgramCacheKey()).not.toBe(perspKey);
  });

  it('alternating kinds re-points at the two cached graphs and builds none', () => {
    const m = make();
    const mesh = meshWith(m);
    draw(mesh, persp());
    const p = m.vertexNode;
    draw(mesh, ortho());
    const o = m.vertexNode;
    const built = sharedTSLGraphCount();
    for (let i = 0; i < 6; i++) {
      draw(mesh, persp());
      expect(m.vertexNode).toBe(p);
      draw(mesh, ortho());
      expect(m.vertexNode).toBe(o);
    }
    expect(sharedTSLGraphCount()).toBe(built);
  });

  it('a repeated draw of the same kind changes nothing (no version bump)', () => {
    const m = make();
    const mesh = meshWith(m);
    draw(mesh, ortho());
    const version = m.version;
    draw(mesh, ortho());
    draw(mesh, ortho());
    expect(m.version).toBe(version);
  });

  it('a CPU camera push selects no variant: only the drawn camera does', () => {
    const m = make();
    const mesh = meshWith(m);
    draw(mesh, ortho());
    const o = m.vertexNode;
    m.updateCameraParams(new THREE.Vector2(800, 600), 0.25, 2);
    expect(m.vertexNode).toBe(o);
    draw(mesh, persp());
    expect(m.vertexNode).not.toBe(o);
  });
});

describe('capsule keeps its runtime ortho switch', () => {
  it.each([
    ['visual', () => new LineTSLMaterial({ primitive: 'capsule' }) as unknown as Drawable],
    [
      'pick',
      () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'capsule' }) as unknown as Drawable,
    ],
  ])('%s capsule draws both kinds with one graph', (_n, make) => {
    const m = make();
    const mesh = meshWith(m);
    draw(mesh, persp());
    const v = m.vertexNode;
    draw(mesh, ortho());
    expect(m.vertexNode).toBe(v);
  });
});
