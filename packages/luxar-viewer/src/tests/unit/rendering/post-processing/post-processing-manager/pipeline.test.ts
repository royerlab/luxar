/**
 * `runPipeline` / `renderSceneToHdr` with the refraction split (spec
 * MESH_PHYSICAL_MATERIALS §3.4 Phase 3), against a `vi.fn` renderer.
 *
 * What is worth pinning is every piece of renderer state the split borrows and must
 * hand back — the glass layer masks, the camera mask, `autoClear`, the transmission
 * scale, the quad's membership of the scene — including on the throw path, because a
 * mask left behind makes glass vanish from every other pass at once. And the four quad
 * flags that make pass B correct under MSAA.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  renderSceneToHdr,
  runPipeline,
  type PipelineCtx,
} from '../../../../../rendering/post-processing/post-processing-manager/pipeline';
import { DataRefractionSplit } from '../../../../../rendering/post-processing/post-processing-manager/refraction-split';
import {
  RENDER_LAYER_DEFAULT,
  RENDER_LAYER_REFRACTING_GLASS,
} from '../../../../../rendering/render-layers';
import type { Renderer } from '../../../../../rendering/renderer-capabilities';

interface RenderCall {
  /** What was drawn: the app scene or the split's private blit scene. */
  what: 'scene' | 'other';
  target: THREE.WebGLRenderTarget | null;
  cameraMask: number;
  autoClear: boolean;
  scale: number;
  quadInScene: boolean;
  glassMasks: number[];
}

/** A renderer stub that records the state at every `render` call. */
function makeRenderer(
  scene: THREE.Scene,
  glass: THREE.Mesh[],
  split: DataRefractionSplit | null,
  opts: { throwOnCall?: number } = {}
) {
  const calls: RenderCall[] = [];
  let target: THREE.WebGLRenderTarget | null = null;
  const renderer = {
    autoClear: true,
    transmissionResolutionScale: 1,
    getRenderTarget: () => target,
    setRenderTarget: vi.fn((t: THREE.WebGLRenderTarget | null) => {
      target = t;
    }),
    clear: vi.fn(),
    render: vi.fn((drawn: THREE.Object3D, camera: THREE.Camera) => {
      calls.push({
        what: drawn === scene ? 'scene' : 'other',
        target,
        cameraMask: camera.layers.mask,
        autoClear: renderer.autoClear,
        scale: renderer.transmissionResolutionScale,
        quadInScene: split ? scene.children.includes(split.screenQuad) : false,
        glassMasks: glass.map((g) => g.layers.mask),
      });
      if (opts.throwOnCall === calls.length) throw new Error('lost context mid-pass');
    }),
  };
  return { renderer: renderer as unknown as Renderer, raw: renderer, calls };
}

function makeCtx(
  renderer: Renderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  refractionSplit: DataRefractionSplit | null
) {
  const hdrTarget = new THREE.WebGLRenderTarget(4, 4);
  const ldrTarget = new THREE.WebGLRenderTarget(4, 4);
  const megaShader = { setHdrSceneTexture: vi.fn() };
  const megaPass = { render: vi.fn() };
  const ctx: PipelineCtx = {
    renderer,
    scene,
    camera,
    hdrTarget,
    ldrTarget,
    megaShader: megaShader as unknown as PipelineCtx['megaShader'],
    megaPass: megaPass as unknown as PipelineCtx['megaPass'],
    bloomChain: null,
    fxaaPass: null,
    refractionSplit,
  };
  return { ctx, hdrTarget, megaShader, megaPass };
}

function makeGlass(): THREE.Mesh {
  return new THREE.Mesh(new THREE.BufferGeometry(), new THREE.Material());
}

describe('renderSceneToHdr — the refraction split (spec §3.4 Phase 3)', () => {
  it('renders once, exactly as before, when there is no split (WebGPU)', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const { renderer, raw, calls } = makeRenderer(scene, [], null);
    const { ctx, hdrTarget } = makeCtx(renderer, scene, camera, null);

    renderSceneToHdr(ctx);

    expect(raw.clear).toHaveBeenCalledTimes(1);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ what: 'scene', target: hdrTarget, autoClear: true, scale: 1 });
  });

  it('renders once when the split exists but no visible glass asks to refract', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const split = new DataRefractionSplit({
      width: 4,
      height: 4,
      transmissionResolutionScale: 0.5,
      collectRefractingGlass: (out) => {
        out.length = 0;
        return out;
      },
    });
    const { renderer, calls } = makeRenderer(scene, [], split);
    const { ctx, hdrTarget } = makeCtx(renderer, scene, camera, split);

    renderSceneToHdr(ctx);

    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ what: 'scene', target: hdrTarget, cameraMask: 1 });
    expect(split.framesSplit).toBe(0);
    expect(scene.children).not.toContain(split.screenQuad);
  });

  it('with refracting glass: pass A without the glass, a blit, pass B with only the glass and the quad', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeGlass();
    scene.add(glass);
    const split = new DataRefractionSplit({
      width: 4,
      height: 4,
      transmissionResolutionScale: 0.5,
      collectRefractingGlass: (out) => {
        out.length = 0;
        out.push(glass);
        return out;
      },
    });
    const { renderer, raw, calls } = makeRenderer(scene, [glass], split);
    const { ctx, hdrTarget } = makeCtx(renderer, scene, camera, split);

    renderSceneToHdr(ctx);

    expect(raw.clear).toHaveBeenCalledTimes(1); // only the caller's clear; pass B keeps pass A
    expect(calls.map((c) => c.what)).toEqual(['scene', 'other', 'scene']);
    const [passA, blit, passB] = calls;
    // Pass A: the glass is off the camera's layer, everything else as usual.
    expect(passA.target).toBe(hdrTarget);
    expect(passA.cameraMask).toBe(1 << RENDER_LAYER_DEFAULT);
    expect(passA.glassMasks).toEqual([1 << RENDER_LAYER_REFRACTING_GLASS]);
    expect(passA.autoClear).toBe(true);
    expect(passA.scale).toBe(1);
    expect(passA.quadInScene).toBe(false);
    // The blit goes to the copy target, not the HDR target.
    expect(blit.target).not.toBe(hdrTarget);
    expect(blit.target).not.toBeNull();
    // Pass B: back into the HDR target, no clear, glass layer only, the quad present,
    // the transmission scale applied for THIS pass alone.
    expect(passB.target).toBe(hdrTarget);
    expect(passB.autoClear).toBe(false);
    expect(passB.cameraMask).toBe(1 << RENDER_LAYER_REFRACTING_GLASS);
    expect(passB.quadInScene).toBe(true);
    expect(passB.scale).toBe(0.5);
    // Everything handed back.
    expect(camera.layers.mask).toBe(1 << RENDER_LAYER_DEFAULT);
    expect(glass.layers.mask).toBe(1 << RENDER_LAYER_DEFAULT);
    expect(raw.autoClear).toBe(true);
    expect(raw.transmissionResolutionScale).toBe(1);
    expect(scene.children).not.toContain(split.screenQuad);
    expect(split.framesSplit).toBe(1);
  });

  it('restores every borrowed state even when pass B throws', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeGlass();
    glass.layers.set(RENDER_LAYER_DEFAULT);
    scene.add(glass);
    const split = new DataRefractionSplit({
      width: 4,
      height: 4,
      transmissionResolutionScale: 0.5,
      collectRefractingGlass: (out) => {
        out.length = 0;
        out.push(glass);
        return out;
      },
    });
    const { renderer, raw } = makeRenderer(scene, [glass], split, { throwOnCall: 3 });
    const { ctx } = makeCtx(renderer, scene, camera, split);

    expect(() => renderSceneToHdr(ctx)).toThrow('lost context mid-pass');

    expect(camera.layers.mask).toBe(1 << RENDER_LAYER_DEFAULT);
    expect(glass.layers.mask).toBe(1 << RENDER_LAYER_DEFAULT);
    expect(raw.autoClear).toBe(true);
    expect(raw.transmissionResolutionScale).toBe(1);
    expect(scene.children).not.toContain(split.screenQuad);
  });

  it('the screen quad carries the four flags pass B depends on under MSAA', () => {
    const split = new DataRefractionSplit({
      width: 4,
      height: 4,
      transmissionResolutionScale: 0.5,
      collectRefractingGlass: (out) => out,
    });
    const quad = split.screenQuad;
    // Opaque, so three's transmission pass draws it into the texture the glass samples.
    expect(quad.material.transparent).toBe(false);
    // Repaints pass A's colour onto the (MSAA-invalidated) target without touching depth.
    expect(quad.material.depthTest).toBe(false);
    expect(quad.material.depthWrite).toBe(false);
    // Its vertex shader ignores the camera; culling by a projected bound would drop it.
    expect(quad.frustumCulled).toBe(false);
    expect(quad.material.toneMapped).toBe(false);
    // It lives on the glass layer, so pass A never draws it.
    expect(quad.layers.mask).toBe(1 << RENDER_LAYER_REFRACTING_GLASS);
    split.dispose();
  });

  it('runPipeline goes through the same stage 0, then the mega pass', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeGlass();
    scene.add(glass);
    const split = new DataRefractionSplit({
      width: 4,
      height: 4,
      transmissionResolutionScale: 0.5,
      collectRefractingGlass: (out) => {
        out.length = 0;
        out.push(glass);
        return out;
      },
    });
    const { renderer, raw, calls } = makeRenderer(scene, [glass], split);
    const { ctx, hdrTarget, megaShader, megaPass } = makeCtx(renderer, scene, camera, split);

    runPipeline(ctx, { applyFxaa: false, finalTarget: null });

    expect(calls.map((c) => c.what)).toEqual(['scene', 'other', 'scene']);
    expect(megaShader.setHdrSceneTexture).toHaveBeenCalledWith(hdrTarget.texture);
    expect(megaPass.render).toHaveBeenCalledTimes(1);
    // The pipeline's own save/restore leaves the renderer on the previous target.
    expect(raw.setRenderTarget).toHaveBeenLastCalledWith(null);
    expect(raw.autoClear).toBe(true);
  });
});
