/**
 * `runPipeline` / `renderSceneToHdr` with the refraction split (spec
 * MESH_PHYSICAL_MATERIALS §3.4 Phase 3), against a `vi.fn` renderer.
 *
 * What is worth pinning is the pass SEQUENCE on each backend — glass depth, data behind,
 * (blit,) glass, (blit,) data in front — with the partition mode, the camera mask, the
 * target, `autoClear` and the transmission scale at every `render` call, and every piece
 * of state the split borrows and must hand back, including on the throw path: a mask
 * left behind makes glass vanish from every other pass at once, and a partition mode
 * left at 1 or 2 makes the data materials discard half the scene in every later frame.
 * And the four quad flags that make the copies correct under MSAA.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  renderSceneToHdr,
  runPipeline,
  type PipelineCtx,
} from '../../../../../rendering/post-processing/post-processing-manager/pipeline';
import {
  DataRefractionSplit,
  type DataRefractionSplitOptions,
} from '../../../../../rendering/post-processing/post-processing-manager/refraction-split';
import {
  RENDER_LAYER_DEFAULT,
  RENDER_LAYER_REFRACTING_GLASS,
  RENDER_LAYER_UNPARTITIONED,
} from '../../../../../rendering/render-layers';
import type { GlassPartition } from '../../../../../rendering/materials/_shared/glass-partition';
import type { Renderer } from '../../../../../rendering/renderer-capabilities';

type Drawn = 'glass-depth' | 'scene' | 'blit' | 'other';

interface RenderCall {
  /** What was drawn: the app scene, the split's depth-proxy scene, or its blit scene. */
  what: Drawn;
  target: THREE.WebGLRenderTarget | null;
  cameraMask: number;
  autoClear: boolean;
  scale: number | undefined;
  /** The partition mode the data materials would see during this draw. */
  partition: GlassPartition;
  screenQuadInScene: boolean;
  restoreQuadInScene: boolean;
  glassMasks: number[];
  unpartitionedMasks: number[];
}

interface Rig {
  renderer: Renderer;
  raw: {
    autoClear: boolean;
    transmissionResolutionScale?: number;
    backend?: { isWebGLBackend?: boolean };
    setRenderTarget: ReturnType<typeof vi.fn>;
    clear: ReturnType<typeof vi.fn>;
    render: ReturnType<typeof vi.fn>;
  };
  calls: RenderCall[];
  /** Every partition mode the split broadcast, in order. */
  partitionWrites: GlassPartition[];
  partition: () => GlassPartition;
}

const DEPTH_TARGET = (split: DataRefractionSplit | null) => split?.glassDepthTarget ?? null;

/** A renderer stub that records the state at every `render` call. */
function makeRig(
  scene: THREE.Scene,
  meshes: { glass: THREE.Mesh[]; unpartitioned?: THREE.Mesh[] },
  opts: { throwOnCall?: number; webgpu?: boolean; webglBackend?: boolean } = {}
): Rig & { split: (o?: Partial<DataRefractionSplitOptions>) => DataRefractionSplit } {
  const calls: RenderCall[] = [];
  const partitionWrites: GlassPartition[] = [];
  let partition: GlassPartition = 0;
  let target: THREE.WebGLRenderTarget | null = null;
  let split: DataRefractionSplit | null = null;
  const unpartitioned = meshes.unpartitioned ?? [];
  const renderer: Rig['raw'] = {
    autoClear: true,
    ...(opts.webgpu ? {} : { transmissionResolutionScale: 1 }),
    ...(opts.webglBackend ? { backend: { isWebGLBackend: true } } : {}),
    setRenderTarget: vi.fn((t: THREE.WebGLRenderTarget | null) => {
      target = t;
    }),
    clear: vi.fn(),
    render: vi.fn((drawn: THREE.Object3D, camera: THREE.Camera) => {
      const what: Drawn =
        drawn === scene
          ? 'scene'
          : drawn === split?.glassDepthScene
            ? 'glass-depth'
            : drawn === split?.blitScene
              ? 'blit'
              : 'other';
      calls.push({
        what,
        target,
        cameraMask: camera.layers.mask,
        autoClear: renderer.autoClear,
        scale: renderer.transmissionResolutionScale,
        partition,
        screenQuadInScene: split ? scene.children.includes(split.screenQuad) : false,
        restoreQuadInScene: split ? scene.children.includes(split.restoreQuad) : false,
        glassMasks: meshes.glass.map((g) => g.layers.mask),
        unpartitionedMasks: unpartitioned.map((g) => g.layers.mask),
      });
      if (opts.throwOnCall === calls.length) throw new Error('lost context mid-pass');
    }),
  };
  Object.defineProperty(renderer, 'getRenderTarget', { value: () => target });
  return {
    renderer: renderer as unknown as Renderer,
    raw: renderer,
    calls,
    partitionWrites,
    partition: () => partition,
    split: (o = {}) => {
      split = new DataRefractionSplit({
        width: 4,
        height: 4,
        transmissionResolutionScale: 0.5,
        apiSurface: opts.webgpu ? 'webgpu' : 'webgl2',
        collectRefractingGlass: (out) => {
          out.length = 0;
          out.push(...meshes.glass);
          return out;
        },
        collectUnpartitionedMeshes: (out) => {
          out.length = 0;
          out.push(...unpartitioned);
          return out;
        },
        setGlassPartition: (mode) => {
          partition = mode;
          partitionWrites.push(mode);
        },
        ...o,
      });
      return split;
    },
  };
}

function makeCtx(
  renderer: Renderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  refractionSplit: DataRefractionSplit | null,
  msaaSamples = 0
) {
  const hdrTarget = new THREE.WebGLRenderTarget(4, 4, { samples: msaaSamples });
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

function makeMesh(): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.Material());
  mesh.position.set(1, 2, 3);
  mesh.updateMatrixWorld(true);
  return mesh;
}

const DEFAULT = 1 << RENDER_LAYER_DEFAULT;
const GLASS = 1 << RENDER_LAYER_REFRACTING_GLASS;
const UNPARTITIONED = 1 << RENDER_LAYER_UNPARTITIONED;

describe('renderSceneToHdr — the refraction split (spec §3.4 Phase 3)', () => {
  it('renders once, exactly as before, when there is no split (disposed manager)', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const rig = makeRig(scene, { glass: [] });
    const { ctx, hdrTarget } = makeCtx(rig.renderer, scene, camera, null);

    renderSceneToHdr(ctx);

    expect(rig.raw.clear).toHaveBeenCalledTimes(1);
    expect(rig.calls).toHaveLength(1);
    expect(rig.calls[0]).toMatchObject({ what: 'scene', target: hdrTarget, autoClear: true });
  });

  it('renders once when no visible glass asks to refract, having primed the depth target', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const rig = makeRig(scene, { glass: [] });
    const split = rig.split();
    const { ctx, hdrTarget } = makeCtx(rig.renderer, scene, camera, split);

    renderSceneToHdr(ctx);

    // The prime: bind the glass depth target, clear it, hand the HDR target back.
    expect(rig.raw.setRenderTarget.mock.calls.map((c) => c[0])).toEqual([
      hdrTarget,
      split.glassDepthTarget,
      hdrTarget,
    ]);
    expect(rig.raw.clear).toHaveBeenCalledTimes(2);
    expect(rig.calls).toHaveLength(1);
    expect(rig.calls[0]).toMatchObject({ what: 'scene', target: hdrTarget, cameraMask: DEFAULT });
    expect(split.framesSplit).toBe(0);
    expect(rig.partitionWrites).toEqual([]);
    expect(scene.children).not.toContain(split.screenQuad);

    // Idempotent until a resize.
    renderSceneToHdr(ctx);
    expect(rig.raw.clear).toHaveBeenCalledTimes(3);
    split.setSize(8, 8);
    renderSceneToHdr(ctx);
    expect(rig.raw.clear).toHaveBeenCalledTimes(5);
  });

  it('WebGL: glass depth, data behind, blit, glass + quad, data in front — every borrowed state handed back', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeMesh();
    const phaseTwo = makeMesh();
    scene.add(glass, phaseTwo);
    const rig = makeRig(scene, { glass: [glass], unpartitioned: [phaseTwo] });
    const split = rig.split();
    const { ctx, hdrTarget } = makeCtx(rig.renderer, scene, camera, split);

    renderSceneToHdr(ctx);

    expect(rig.calls.map((c) => c.what)).toEqual([
      'glass-depth',
      'scene',
      'blit',
      'scene',
      'scene',
    ]);
    const [passG, passA, blit, passB, passC] = rig.calls;
    // Pass G: the proxies alone, into the depth target, cleared; nothing partitioned yet.
    expect(passG.target).toBe(split.glassDepthTarget);
    expect(passG.autoClear).toBe(true);
    expect(passG.partition).toBe(0);
    // Pass A: the glass off the camera's layers, the unpartitioned meshes ON them, the
    // data in "behind" mode, into the HDR target (cleared again — the first draw there).
    expect(passA.target).toBe(hdrTarget);
    expect(passA.cameraMask).toBe(DEFAULT | UNPARTITIONED);
    expect(passA.glassMasks).toEqual([GLASS]);
    expect(passA.unpartitionedMasks).toEqual([UNPARTITIONED]);
    expect(passA.partition).toBe(1);
    expect(passA.autoClear).toBe(true);
    expect(passA.scale).toBe(1);
    expect(passA.screenQuadInScene).toBe(false);
    // The blit goes to the copy target, not the HDR target.
    expect(blit.target).not.toBe(hdrTarget);
    expect(blit.target).not.toBe(split.glassDepthTarget);
    expect(blit.target).not.toBeNull();
    // Pass B: back into the HDR target, no clear, glass layer only, the quad present,
    // the transmission scale applied for THIS pass alone.
    expect(passB.target).toBe(hdrTarget);
    expect(passB.autoClear).toBe(false);
    expect(passB.cameraMask).toBe(GLASS);
    expect(passB.screenQuadInScene).toBe(true);
    expect(passB.scale).toBe(0.5);
    // Pass C: the data in "front" mode on the default layer only (no glass, no
    // unpartitioned meshes), no clear, the quad gone, the scale back.
    expect(passC.target).toBe(hdrTarget);
    expect(passC.autoClear).toBe(false);
    expect(passC.cameraMask).toBe(DEFAULT);
    expect(passC.partition).toBe(2);
    expect(passC.screenQuadInScene).toBe(false);
    expect(passC.restoreQuadInScene).toBe(false); // no MSAA: nothing to repaint
    expect(passC.scale).toBe(1);
    // Everything handed back, the partition first of all.
    expect(rig.partition()).toBe(0);
    expect(rig.partitionWrites).toEqual([1, 2, 0]);
    expect(camera.layers.mask).toBe(DEFAULT);
    expect(glass.layers.mask).toBe(DEFAULT);
    expect(phaseTwo.layers.mask).toBe(DEFAULT);
    expect(rig.raw.autoClear).toBe(true);
    expect(rig.raw.transmissionResolutionScale).toBe(1);
    expect(scene.children).not.toContain(split.screenQuad);
    expect(scene.children).not.toContain(split.restoreQuad);
    expect(split.framesSplit).toBe(1);
    expect(split.lastPassCount).toBe(5);
  });

  it('WebGL under MSAA: a second blit and the restore quad repaint the target under pass C', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeMesh();
    scene.add(glass);
    const rig = makeRig(scene, { glass: [glass] });
    const split = rig.split();
    const { ctx } = makeCtx(rig.renderer, scene, camera, split, 4);

    renderSceneToHdr(ctx);

    expect(rig.calls.map((c) => c.what)).toEqual([
      'glass-depth',
      'scene',
      'blit',
      'scene',
      'blit',
      'scene',
    ]);
    const passC = rig.calls[5];
    expect(passC.restoreQuadInScene).toBe(true);
    expect(passC.screenQuadInScene).toBe(false);
    expect(passC.partition).toBe(2);
    expect(scene.children).not.toContain(split.restoreQuad);
    expect(split.lastPassCount).toBe(6);
  });

  it('WebGPU: the same partition without the copy or the quads, and no transmission scale', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeMesh();
    scene.add(glass);
    const rig = makeRig(scene, { glass: [glass] }, { webgpu: true });
    const split = rig.split();
    const { ctx, hdrTarget } = makeCtx(rig.renderer, scene, camera, split, 4);

    renderSceneToHdr(ctx);

    expect(rig.calls.map((c) => c.what)).toEqual(['glass-depth', 'scene', 'scene', 'scene']);
    const [, passA, passB, passC] = rig.calls;
    expect(passA).toMatchObject({ target: hdrTarget, partition: 1, autoClear: true });
    expect(passB).toMatchObject({ target: hdrTarget, cameraMask: GLASS, autoClear: false });
    expect(passC).toMatchObject({ target: hdrTarget, partition: 2, cameraMask: DEFAULT });
    expect(rig.calls.every((c) => !c.screenQuadInScene && !c.restoreQuadInScene)).toBe(true);
    expect(rig.calls.every((c) => c.scale === undefined)).toBe(true);
    expect('transmissionResolutionScale' in rig.raw).toBe(false);
    expect(rig.partitionWrites).toEqual([1, 2, 0]);
    expect(split.lastPassCount).toBe(4);
  });

  it('WebGPU on its WebGL2 fallback with MSAA falls back to one pass and never partitions', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeMesh();
    scene.add(glass);
    const rig = makeRig(scene, { glass: [glass] }, { webgpu: true, webglBackend: true });
    const split = rig.split();
    const { ctx } = makeCtx(rig.renderer, scene, camera, split, 4);

    renderSceneToHdr(ctx);
    renderSceneToHdr(ctx);

    expect(rig.calls.map((c) => c.what)).toEqual(['scene', 'scene']);
    expect(rig.partitionWrites).toEqual([]);
    expect(glass.layers.mask).toBe(DEFAULT);
    expect(split.framesSplit).toBe(0);

    // Without MSAA the same renderer partitions normally.
    const { ctx: plain } = makeCtx(rig.renderer, scene, camera, split, 0);
    renderSceneToHdr(plain);
    expect(rig.calls.slice(2).map((c) => c.what)).toEqual([
      'glass-depth',
      'scene',
      'scene',
      'scene',
    ]);
  });

  it.each([1, 2, 3, 4, 5])('restores every borrowed state when render call %i throws', (n) => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeMesh();
    const phaseTwo = makeMesh();
    scene.add(glass, phaseTwo);
    const rig = makeRig(scene, { glass: [glass], unpartitioned: [phaseTwo] }, { throwOnCall: n });
    const split = rig.split();
    const { ctx } = makeCtx(rig.renderer, scene, camera, split);

    expect(() => renderSceneToHdr(ctx)).toThrow('lost context mid-pass');

    expect(rig.partition()).toBe(0);
    expect(camera.layers.mask).toBe(DEFAULT);
    expect(glass.layers.mask).toBe(DEFAULT);
    expect(phaseTwo.layers.mask).toBe(DEFAULT);
    expect(rig.raw.autoClear).toBe(true);
    expect(rig.raw.transmissionResolutionScale).toBe(1);
    expect(scene.children).not.toContain(split.screenQuad);
    expect(scene.children).not.toContain(split.restoreQuad);
  });

  it('the two screen quads carry the flags the copies depend on under MSAA', () => {
    const rig = makeRig(new THREE.Scene(), { glass: [] });
    const split = rig.split();
    for (const quad of [split.screenQuad, split.restoreQuad]) {
      // Opaque, so three's transmission pass draws it into the texture the glass samples.
      expect(quad.material.transparent).toBe(false);
      // Repaints the (MSAA-invalidated) target without touching depth.
      expect(quad.material.depthTest).toBe(false);
      expect(quad.material.depthWrite).toBe(false);
      // Its vertex shader ignores the camera; culling by a projected bound would drop it.
      expect(quad.frustumCulled).toBe(false);
      expect(quad.material.toneMapped).toBe(false);
      // Leads the opaque list, so no opaque-mode data layer of its pass is painted over.
      expect(quad.renderOrder).toBe(-1e9);
    }
    // The pass-B quad lives on the glass layer (pass A never draws it); the pass-C quad on
    // the default layer, where the data in front is drawn over it.
    expect(split.screenQuad.layers.mask).toBe(GLASS);
    expect(split.restoreQuad.layers.mask).toBe(DEFAULT);
    split.dispose();
  });

  it('pass G draws one front-face, depth-only proxy per glass carrying its world matrix; surplus proxies hide', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const a = makeMesh();
    const b = makeMesh();
    b.position.set(-5, 0, 0);
    b.updateMatrixWorld(true);
    scene.add(a, b);
    const glass = [a, b];
    const rig = makeRig(scene, { glass });
    const split = rig.split();
    const { ctx } = makeCtx(rig.renderer, scene, camera, split);

    renderSceneToHdr(ctx);
    const proxies = split.glassDepthScene.children as THREE.Mesh[];
    expect(proxies).toHaveLength(2);
    for (const [i, proxy] of proxies.entries()) {
      expect(proxy.geometry).toBe(glass[i].geometry);
      expect(proxy.matrixWorld.equals(glass[i].matrixWorld)).toBe(true);
      expect(proxy.visible).toBe(true);
      const m = proxy.material as THREE.MeshBasicMaterial;
      expect(m.colorWrite).toBe(false);
      expect(m.side).toBe(THREE.FrontSide);
    }
    // The glass mesh's own material is untouched — the proxy is what carries depth-only.
    expect((a.material as THREE.Material).colorWrite).toBe(true);

    glass.pop();
    renderSceneToHdr(ctx);
    expect(split.glassDepthScene.children).toHaveLength(2);
    expect((split.glassDepthScene.children[1] as THREE.Mesh).visible).toBe(false);
  });

  it('runPipeline goes through the same stage 0, then the mega pass', () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const glass = makeMesh();
    scene.add(glass);
    const rig = makeRig(scene, { glass: [glass] });
    const split = rig.split();
    const { ctx, hdrTarget, megaShader, megaPass } = makeCtx(rig.renderer, scene, camera, split);

    runPipeline(ctx, { applyFxaa: false, finalTarget: null });

    expect(rig.calls.map((c) => c.what)).toEqual([
      'glass-depth',
      'scene',
      'blit',
      'scene',
      'scene',
    ]);
    expect(megaShader.setHdrSceneTexture).toHaveBeenCalledWith(hdrTarget.texture);
    expect(megaPass.render).toHaveBeenCalledTimes(1);
    // The pipeline's own save/restore leaves the renderer on the previous target.
    expect(rig.raw.setRenderTarget).toHaveBeenLastCalledWith(null);
    expect(rig.raw.autoClear).toBe(true);
    expect(DEPTH_TARGET(split)).toBe(split.glassDepthTarget);
  });
});
