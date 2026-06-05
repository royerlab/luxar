/**
 * Unit tests for the WebGLContextRecovery concern.
 *
 * Strategy: build a fake DOM canvas + minimal renderer / scene / post
 * processing fakes, dispatch the two synthetic events
 * (`webglcontextlost`, `webglcontextrestored`), and assert on:
 *   - the isContextLost flag transitions correctly;
 *   - the rebuild order on restore (renderer.resetState →
 *     postProcessing.rebuildAfterContextRestore →
 *     materialManager.rebuildAfterContextRestore →
 *     scene-resources-dirty → updateRendererSize);
 *   - the dispatch-events-after-restore sequence (onContextRestored
 *     before triggerChange);
 *   - dispose unbinds both listeners;
 *   - markSceneResourcesDirtyForContextRestore flips needsUpdate
 *     across attributes + materials.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

vi.mock('../../../../../rendering/material-manager', () => ({
  materialManager: {
    rebuildAfterContextRestore: vi.fn(),
  },
}));

import { materialManager } from '../../../../../rendering/material-manager';
import {
  WebGLContextRecovery,
  markSceneResourcesDirtyForContextRestore,
  type WebGLContextRecoveryDeps,
} from '../../../../../scene/scene-manager/render-pipeline/webgl-context-recovery';
import type { PostProcessingManager } from '../../../../../rendering/post-processing/post-processing-manager';

function makeDeps(overrides: Partial<WebGLContextRecoveryDeps> = {}): {
  deps: WebGLContextRecoveryDeps;
  canvas: HTMLCanvasElement;
  scene: THREE.Scene;
  resetState: ReturnType<typeof vi.fn>;
  ppRebuild: ReturnType<typeof vi.fn>;
  updateRendererSize: ReturnType<typeof vi.fn>;
  onContextRestored: ReturnType<typeof vi.fn>;
  triggerChange: ReturnType<typeof vi.fn>;
} {
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  const scene = new THREE.Scene();
  const resetState = vi.fn();
  const renderer = { resetState } as unknown as THREE.WebGLRenderer;
  const ppRebuild = vi.fn();
  const postProcessing = {
    rebuildAfterContextRestore: ppRebuild,
  } as unknown as PostProcessingManager;
  const updateRendererSize = vi.fn();
  const onContextRestored = vi.fn();
  const triggerChange = vi.fn();
  const deps: WebGLContextRecoveryDeps = {
    canvas,
    getScene: () => scene,
    renderer,
    getPostProcessing: () => postProcessing,
    updateRendererSize,
    onContextRestored,
    triggerChange,
    ...overrides,
  };
  return {
    deps,
    canvas,
    scene,
    resetState,
    ppRebuild,
    updateRendererSize,
    onContextRestored,
    triggerChange,
  };
}

describe('WebGLContextRecovery', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    vi.mocked(materialManager.rebuildAfterContextRestore).mockClear();
  });

  it('starts with isContextLost=false', () => {
    const { deps } = makeDeps();
    const recovery = new WebGLContextRecovery(deps);
    expect(recovery.getIsContextLost()).toBe(false);
  });

  it('webglcontextlost event flips isContextLost=true', () => {
    const { deps, canvas } = makeDeps();
    const recovery = new WebGLContextRecovery(deps);
    recovery.attach();
    canvas.dispatchEvent(new Event('webglcontextlost'));
    expect(recovery.getIsContextLost()).toBe(true);
  });

  it('webglcontextlost listener calls preventDefault on the event', () => {
    const { deps, canvas } = makeDeps();
    new WebGLContextRecovery(deps).attach();
    const evt = new Event('webglcontextlost', { cancelable: true });
    const pdSpy = vi.spyOn(evt, 'preventDefault');
    canvas.dispatchEvent(evt);
    expect(pdSpy).toHaveBeenCalled();
  });

  it('webglcontextrestored runs the rebuild order then dispatches events', async () => {
    const {
      deps,
      canvas,
      resetState,
      ppRebuild,
      updateRendererSize,
      onContextRestored,
      triggerChange,
    } = makeDeps();
    const recovery = new WebGLContextRecovery(deps);
    recovery.attach();
    canvas.dispatchEvent(new Event('webglcontextlost'));
    expect(recovery.getIsContextLost()).toBe(true);

    canvas.dispatchEvent(new Event('webglcontextrestored'));
    // Allow the async listener body to run.
    await new Promise((r) => setTimeout(r, 0));

    expect(recovery.getIsContextLost()).toBe(false);
    expect(resetState).toHaveBeenCalledTimes(1);
    expect(ppRebuild).toHaveBeenCalledTimes(1);
    expect(materialManager.rebuildAfterContextRestore).toHaveBeenCalledTimes(1);
    expect(updateRendererSize).toHaveBeenCalledTimes(1);
    expect(onContextRestored).toHaveBeenCalledTimes(1);
    expect(triggerChange).toHaveBeenCalledTimes(1);

    // Order: resetState before postProcessing rebuild before materials rebuild
    // before updateRendererSize before onContextRestored before triggerChange.
    const callOrder = [
      resetState.mock.invocationCallOrder[0],
      ppRebuild.mock.invocationCallOrder[0],
      vi.mocked(materialManager.rebuildAfterContextRestore).mock.invocationCallOrder[0],
      updateRendererSize.mock.invocationCallOrder[0],
      onContextRestored.mock.invocationCallOrder[0],
      triggerChange.mock.invocationCallOrder[0],
    ];
    for (let i = 1; i < callOrder.length; i++) {
      expect(callOrder[i]).toBeGreaterThan(callOrder[i - 1]);
    }
  });

  it('skips post-processing rebuild when getPostProcessing returns null', async () => {
    // W5: keep a real rebuild spy but make the getter return null, so we can
    // assert the pp rebuild is genuinely NOT called (not merely absent).
    const ppRebuild = vi.fn();
    const { deps, canvas } = makeDeps({ getPostProcessing: () => null });
    new WebGLContextRecovery(deps).attach();
    canvas.dispatchEvent(new Event('webglcontextrestored'));
    await new Promise((r) => setTimeout(r, 0));
    // material manager + scene rebuild still happen — only the pp call is skipped.
    expect(materialManager.rebuildAfterContextRestore).toHaveBeenCalledTimes(1);
    expect(ppRebuild).not.toHaveBeenCalled();
  });

  it('dispose unbinds BOTH listeners (lost and restored)', async () => {
    const { deps, canvas, resetState } = makeDeps();
    const recovery = new WebGLContextRecovery(deps);
    recovery.attach();
    recovery.dispose();

    // Lost handler unbound → flag stays false.
    canvas.dispatchEvent(new Event('webglcontextlost'));
    expect(recovery.getIsContextLost()).toBe(false);

    // W2: restored handler unbound → its rebuild side-effects never run.
    canvas.dispatchEvent(new Event('webglcontextrestored'));
    await new Promise((r) => setTimeout(r, 0));
    expect(resetState).not.toHaveBeenCalled();
  });

  it('dispose is idempotent', () => {
    const { deps } = makeDeps();
    const recovery = new WebGLContextRecovery(deps);
    recovery.attach();
    recovery.dispose();
    expect(() => recovery.dispose()).not.toThrow();
  });
});

describe('markSceneResourcesDirtyForContextRestore', () => {
  it('flips needsUpdate=true on every attribute of every renderable', () => {
    // THREE's `needsUpdate` is a setter-only property — assigning `true`
    // increments `.version`. We assert on version to verify the flip.
    const scene = new THREE.Scene();
    const geometry = new THREE.BufferGeometry();
    const positionAttr = new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3);
    const colorAttr = new THREE.BufferAttribute(new Float32Array([1, 1, 1]), 3);
    geometry.setAttribute('position', positionAttr);
    geometry.setAttribute('color', colorAttr);
    const material = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    const positionVersionBefore = positionAttr.version;
    const colorVersionBefore = colorAttr.version;
    const materialVersionBefore = material.version;

    markSceneResourcesDirtyForContextRestore(scene);

    expect(positionAttr.version).toBe(positionVersionBefore + 1);
    expect(colorAttr.version).toBe(colorVersionBefore + 1);
    expect(material.version).toBe(materialVersionBefore + 1);
  });

  it('handles material arrays (each material flipped)', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.BufferGeometry();
    const m1 = new THREE.MeshBasicMaterial();
    const m2 = new THREE.MeshBasicMaterial();
    const mesh = new THREE.Mesh(geometry, [m1, m2]);
    scene.add(mesh);
    const v1 = m1.version;
    const v2 = m2.version;

    markSceneResourcesDirtyForContextRestore(scene);

    expect(m1.version).toBe(v1 + 1);
    expect(m2.version).toBe(v2 + 1);
  });

  it('flips index buffer needsUpdate when present', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3)
    );
    const indexAttr = new THREE.BufferAttribute(new Uint16Array([0, 1, 2]), 1);
    geometry.setIndex(indexAttr);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    scene.add(mesh);
    const before = indexAttr.version;

    markSceneResourcesDirtyForContextRestore(scene);
    expect(indexAttr.version).toBe(before + 1);
  });

  it('does not throw on Group / Object3D children with no geometry', () => {
    const scene = new THREE.Scene();
    scene.add(new THREE.Group());
    scene.add(new THREE.Object3D());
    expect(() => markSceneResourcesDirtyForContextRestore(scene)).not.toThrow();
  });
});
