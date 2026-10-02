// @vitest-environment jsdom
/**
 * The gsplat pick pass culls near/giant splats with the DRAW pass's
 * truncation radius (review A5).
 *
 * The pick footprint is deliberately tighter than the drawn one (1.5σ — the
 * bright core), but the screen-coverage FADE is not a footprint: it is the
 * rule that culls a splat whose projected extent exceeds the viewport limit,
 * and the visual shader evaluates it with the node's own T (2.75 by default,
 * per node). Evaluated at the pick's 1.5σ instead, a splat the draw pass has
 * already faded out still projects under the limit in the pick pass — a
 * giant near splat invisible on screen kept winning hovers.
 *
 * So the pick shaders read a separate `uCoverageTruncate`, synced from the
 * visual material each pick render, while `uTruncate` keeps sizing the 1.5σ
 * pick footprint. The coverage limit's other inputs (`uMaxExtentFactor`, the
 * 2D dilation) ride along in the same sync.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../../config/constants';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { GSPLAT_VERTEX_SHADER } from '../../../../rendering/materials/gsplat/shader-glsl';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { GSPLAT_PICK_VERTEX_SHADER } from '../../../../rendering/picking/gsplat/shaders';
import { PickingSystem } from '../../../../rendering/picking/picking-system';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** The one `projectedExtent` assignment of a vertex shader. */
function projectedExtentLine(vs: string): string {
  const lines = vs.split('\n').filter((l) => l.includes('float projectedExtent ='));
  expect(lines).toHaveLength(1);
  return lines[0].trim();
}

describe('gsplat pick coverage fade uses the draw truncation', () => {
  it('GLSL: the pick coverage extent is the visual one over uCoverageTruncate', () => {
    const visual = projectedExtentLine(GSPLAT_VERTEX_SHADER);
    const pick = projectedExtentLine(GSPLAT_PICK_VERTEX_SHADER);
    expect(pick).toBe(visual.replace('uTruncate', 'uCoverageTruncate'));
    // The pick FOOTPRINT keeps the tight pick radius.
    expect(GSPLAT_PICK_VERTEX_SHADER).toContain('float extent1 = uTruncate * sqrt(lambda1);');
  });

  it('TSL: the pick coverage extent reads uCoverageTruncate', () => {
    const src = readFileSync(
      path.resolve(HERE, '../../../../rendering/picking/gsplat/pick.tsl.ts'),
      'utf8'
    );
    const block = src.slice(src.indexOf('const projectedExtent'), src.indexOf('const maxExtent'));
    expect(block).toContain('.mul(uCoverageTruncate)');
    expect(block).not.toContain('.mul(uTruncate)');
  });

  it.each([
    ['GLSL', () => new GSplatPickingMaterial({ nodeId: 1 })],
    ['TSL', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
  ])('%s pick material declares uCoverageTruncate at the draw default', (_n, make) => {
    const mat = make() as unknown as { uniforms: Record<string, { value: unknown }> };
    expect(mat.uniforms.uCoverageTruncate?.value).toBe(GSPLAT_DEFAULT_TRUNCATION_RADIUS);
    // The pick footprint radius is unchanged.
    expect(mat.uniforms.uTruncate.value).toBe(1.5);
  });
});

function makeRenderer(): THREE.WebGLRenderer {
  const canvas = document.createElement('canvas');
  Object.defineProperty(canvas, 'clientHeight', { value: 600, configurable: true });
  return {
    domElement: canvas,
    getDrawingBufferSize: vi.fn(),
    getPixelRatio: vi.fn(() => 1),
    readRenderTargetPixels: vi.fn(),
    getRenderTarget: vi.fn(() => null),
    getScissorTest: vi.fn(() => false),
    getClearColor: vi.fn(),
    getClearAlpha: vi.fn(() => 0),
    setRenderTarget: vi.fn(),
    setScissorTest: vi.fn(),
    setClearColor: vi.fn(),
    clear: vi.fn(),
    render: vi.fn(),
  } as unknown as THREE.WebGLRenderer;
}

function renderPick(main: THREE.Material, pick: THREE.Material): void {
  const system = new PickingSystem(
    makeRenderer(),
    { apiSurface: 'webgl2', framebufferYDown: false } as never,
    new THREE.PerspectiveCamera(60, 1, 0.1, 1000),
    vi.fn()
  );
  const geom = new THREE.BufferGeometry();
  system.registerNode(
    new THREE.Mesh(geom, main),
    new THREE.Mesh(geom, pick),
    system.allocatePickId()
  );
  (system as unknown as { renderPickBuffer: () => void }).renderPickBuffer();
}

describe('renderPickBuffer syncs the coverage-limit inputs from the visual material', () => {
  it.each([
    ['GLSL', () => new GSplatPickingMaterial({ nodeId: 1 })],
    ['TSL', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
  ])('%s pick material', (_n, makePick) => {
    const main = new GSplatMaterial({ truncationRadius: 3.5, maxExtentFactor: 0.5 });
    main.uniforms.uCov2DDilation.value = 0.7;
    const pick = makePick() as unknown as THREE.Material & {
      uniforms: Record<string, { value: unknown }>;
    };

    renderPick(main, pick);

    expect(pick.uniforms.uCoverageTruncate.value).toBe(3.5);
    expect(pick.uniforms.uMaxExtentFactor.value).toBe(0.5);
    expect(pick.uniforms.uCov2DDilation.value).toBe(0.7);
    // The pick footprint radius is NOT overwritten by the draw's T.
    expect(pick.uniforms.uTruncate.value).toBe(1.5);
  });
});
