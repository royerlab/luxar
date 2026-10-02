/**
 * Drive `PickingSystem.renderPickBuffer()` over real (main, pick) material
 * pairs without a GL context: the renderer is a stub of exactly the surface
 * the pick render touches, so the per-node visual → pick syncs run for real.
 *
 * @module tests/unit/rendering/picking/render-pick-helper
 */
import * as THREE from 'three';
import { vi } from 'vitest';

import { PickingSystem } from '../../../../rendering/picking/picking-system';

/** A stub renderer covering what `renderPickBuffer` saves, sets and restores. */
export function makePickRenderer(): THREE.WebGLRenderer {
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

/** A picking system whose private pick render is callable, plus a pair registrar. */
export function makePickHarness(
  camera: THREE.Camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000)
) {
  const system = new PickingSystem(
    makePickRenderer(),
    { apiSurface: 'webgl2', framebufferYDown: false } as never,
    camera as never,
    vi.fn()
  );
  const register = (main: THREE.Material, pick: THREE.Material): THREE.Mesh => {
    const geom = new THREE.BufferGeometry();
    const mainNode = new THREE.Mesh(geom, main);
    system.registerNode(mainNode, new THREE.Mesh(geom, pick), system.allocatePickId());
    return mainNode;
  };
  const renderPickBuffer = (): void =>
    (system as unknown as { renderPickBuffer: () => void }).renderPickBuffer();
  return { system, register, renderPickBuffer };
}

/** `material.uniforms[name].value`, for GLSL records and TSL proxies alike. */
export function uniformValue(material: THREE.Material, name: string): unknown {
  return (material as unknown as { uniforms: Record<string, { value: unknown }> }).uniforms[name]
    ?.value;
}

/** Write `material.uniforms[name].value`. */
export function setUniform(material: THREE.Material, name: string, value: unknown): void {
  (material as unknown as { uniforms: Record<string, { value: unknown }> }).uniforms[name].value =
    value;
}
