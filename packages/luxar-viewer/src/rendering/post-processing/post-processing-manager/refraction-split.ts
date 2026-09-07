/**
 * The scene-pass split that lets `refract_data` glass refract the emissive data behind
 * it while the data in FRONT of it stays crisp (spec MESH_PHYSICAL_MATERIALS §3.4,
 * Phase 3), on both backends.
 *
 * Two facts force a multi-pass frame. Every Luxar point, line and splat material is
 * transparent and writes no depth (additive mode has the depth test off altogether), so
 * a glass drawn after the data cannot tell a lattice row in front of it from one behind
 * it — and three's WebGL `renderTransmissionPass` draws only the OPAQUE render list
 * into the texture a transmissive material samples, so on that backend the glass could
 * not see the data at all in one pass. The split therefore renders, per frame with a
 * visible refracting glass:
 *
 * 1. **Pass G** — the refracting glass's FRONT faces, depth only, into
 *    {@link DataRefractionSplit.glassDepthTarget} (cleared depth 1.0 = "no glass here").
 *    Drawn through depth-only PROXY meshes in a private scene (geometry shared, world
 *    matrix copied — the pick system's pattern), so the glass mesh's own material and
 *    `side` never matter and the main scene is not traversed a third time.
 * 2. **Pass A** — everything except the refracting glass, with every data material in
 *    partition mode 1 (`glass-partition.ts`): each fragment compares its own window
 *    depth with the glass depth at its pixel and keeps itself only when it is BEHIND the
 *    glass or under no glass at all. Meshes three's own materials draw (physical glass
 *    without the flag, physical opaque surfaces) cannot classify themselves; they are
 *    drawn here in full and excluded from pass C via {@link RENDER_LAYER_UNPARTITIONED}.
 * 3. **Blit** (WebGL only) — pass A's colour into a copy target: pass B must read it
 *    while writing the HDR target, and a texture cannot be both.
 * 4. **Pass B** — only the glass (plus, on WebGL, one screen-space quad textured with
 *    the copy: an OPAQUE object, so three's transmission pass draws it into the
 *    transmission texture and the glass refracts the data), into the HDR target WITHOUT
 *    clearing. On WebGPU three samples the live framebuffer, so the glass alone is
 *    enough. Pass A's depth survives, so an opaque mesh in front still occludes the
 *    glass.
 * 5. **Pass C** — the data again in partition mode 2: only the fragments IN FRONT of the
 *    glass survive, and they land crisp on top of it. Modes 1 and 2 are complements of
 *    the one predicate, so every data fragment is drawn exactly once across A and C.
 *
 * **Under MSAA the copies are load-bearing, not an optimisation.** Three's WebGL
 * renderer resolves a multisampled target's colour to its texture at the end of every
 * `render()` and then INVALIDATES the multisampled colour attachment (the depth
 * attachment is resolved and kept), so passes B and C would each draw onto undefined
 * colour. The pass-B quad repaints it from the copy; before pass C a second blit and a
 * second full-screen quad (leading the opaque list) do the same. The four quad flags —
 * `depthTest: false`, `depthWrite: false`, `transparent: false`, `frustumCulled: false`
 * — are pinned by test. `WebGPURenderer`'s native backend keeps its multisampled colour
 * between passes; its WebGL2 fallback does not and cannot take the GLSL quad, so that
 * one combination (`?webgpu-force-webgl` + MSAA, both diagnostic switches) falls back
 * to the single pass and says so once.
 *
 * Everything the split touches is restored in a `finally` before `render()` returns —
 * the partition mode FIRST (an exception must never leave the data materials
 * discarding), then the glass and unpartitioned meshes' layer masks, the camera mask,
 * `autoClear`, the renderer's `transmissionResolutionScale`, and the quads' membership
 * of the scene — so the pick pass, the environment cube capture, the blend warm-up and
 * scene disposal never see any of it. The scene holds no lights today; a future light
 * would need `layers.enableAll()` to reach pass B.
 *
 * `transmissionResolutionScale` (config `renderingControls.refraction`) applies to
 * pass B only: three reads it per transmission pass, so Phase 2 glass in pass A keeps
 * three's default and its pixels are untouched.
 *
 * The glass depth target is created WITH the one shared `DepthTexture` every data
 * material already samples (`getGlassDepthTexture`), so no material is ever rebound.
 * The target is PRIMED (bound and cleared) at the start of the first split-capable
 * frame and after every resize, before any material's first bind in that frame, so the
 * GPU texture a bind group captures is always the full-size one.
 *
 * @module rendering/post-processing/post-processing-manager/refraction-split
 */

import * as THREE from 'three';
import type { Renderer } from '../../renderer-capabilities';
import {
  RENDER_LAYER_DEFAULT,
  RENDER_LAYER_REFRACTING_GLASS,
  RENDER_LAYER_UNPARTITIONED,
} from '../../render-layers';
import {
  getGlassDepthTexture,
  GLASS_PARTITION_BEHIND,
  GLASS_PARTITION_FRONT,
  GLASS_PARTITION_OFF,
  type GlassPartition,
} from '../../materials/_shared/glass-partition';
import { log, Modules } from '../../../utils/log';

/** How the split is built (all from the orchestrator). */
export interface DataRefractionSplitOptions {
  /** Physical width of the HDR target the copy and depth targets must match. */
  readonly width: number;
  /** Physical height of the HDR target the copy and depth targets must match. */
  readonly height: number;
  /** Pass B's `renderer.transmissionResolutionScale`, in `(0, 1]` (WebGL only). */
  readonly transmissionResolutionScale: number;
  /**
   * Which renderer owns the split: three's `WebGLRenderer` (`'webgl2'`, whose
   * transmission pass needs the copy + quad) or `WebGPURenderer` (`'webgpu'`, which
   * samples the live framebuffer).
   */
  readonly apiSurface: 'webgl2' | 'webgpu';
  /**
   * The visible refracting glass this frame, into a caller-owned array (the depth-sort
   * coordinator's `collectRefractingGlass`). Injected so the split owns no scene
   * knowledge and a unit test can hand it any list.
   */
  readonly collectRefractingGlass: (out: THREE.Mesh[]) => THREE.Mesh[];
  /**
   * The visible meshes drawn by three's own materials (physical, not refracting), which
   * cannot partition themselves: drawn in pass A only.
   */
  readonly collectUnpartitionedMeshes: (out: THREE.Mesh[]) => THREE.Mesh[];
  /** Broadcast a partition mode to every data material (`applyGlassPartition`). */
  readonly setGlassPartition: (mode: GlassPartition) => void;
}

/** The slice of either renderer the split writes; `WebGPURenderer` lacks the scale. */
interface SplitRenderer {
  render(scene: THREE.Object3D, camera: THREE.Camera): void;
  setRenderTarget(target: THREE.WebGLRenderTarget | null): void;
  getRenderTarget(): THREE.WebGLRenderTarget | null;
  clear(): void;
  autoClear: boolean;
  transmissionResolutionScale?: number;
  backend?: { isWebGLBackend?: boolean };
}

const COPY_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const COPY_FRAGMENT = /* glsl */ `
precision highp float;
uniform sampler2D map;
varying vec2 vUv;
void main() {
  gl_FragColor = texture2D(map, vUv);
}`;

/** Allocate the colour-only copy of a pass (same format as the HDR target, no depth). */
export function createRefractionCopyTarget(width: number, height: number): THREE.WebGLRenderTarget {
  const t = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
  t.texture.name = 'PostProcessing.refractionCopy';
  return t;
}

/**
 * Allocate the glass front-face depth target around the ONE shared depth texture. A
 * colour attachment is mandatory, so it carries the cheapest one (RGBA8, never read).
 */
export function createGlassDepthTarget(width: number, height: number): THREE.WebGLRenderTarget {
  const t = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    generateMipmaps: false,
    depthBuffer: true,
    stencilBuffer: false,
    depthTexture: getGlassDepthTexture(),
  });
  t.texture.name = 'PostProcessing.glassDepthColor';
  return t;
}

/**
 * A screen-space quad whose vertex shader ignores the camera, so it covers the whole
 * viewport of whatever target it is drawn into — including three's transmission
 * target, which is what makes the data visible to the glass. See the module doc for
 * why every one of the four flags is load-bearing.
 */
function createScreenQuad(
  name: string,
  layer: number
): THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial> {
  const material = new THREE.ShaderMaterial({
    uniforms: { map: { value: null } },
    vertexShader: COPY_VERTEX,
    fragmentShader: COPY_FRAGMENT,
    depthTest: false,
    depthWrite: false,
    transparent: false,
    toneMapped: false,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  quad.matrixAutoUpdate = false;
  quad.name = name;
  quad.layers.set(layer);
  // Lead the opaque list: the quad must repaint the target BEFORE any opaque-mode data
  // layer of the same pass draws (three sorts opaques by renderOrder first).
  quad.renderOrder = -1e9;
  return quad;
}

/** The depth-only material the pass-G proxies share: front faces, no colour. */
function createDepthOnlyMaterial(): THREE.MeshBasicMaterial {
  const m = new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.FrontSide });
  m.name = 'PostProcessing.glassDepthOnly';
  return m;
}

/** Renderer and camera state the split borrows for one `render()` call. */
interface BorrowedState {
  readonly cameraMask: number;
  readonly autoClear: boolean;
  /** `undefined` on `WebGPURenderer`, which has no transmission scale. */
  readonly scale: number | undefined;
}

/** Owns the depth target, the copy target, the proxies and the quads; one per manager. */
export class DataRefractionSplit {
  /** The pass-B quad (a scene member only between pass B's start and its `finally`). */
  readonly screenQuad = createScreenQuad(
    'PostProcessing.refractionScreenQuad',
    RENDER_LAYER_REFRACTING_GLASS
  );
  /** The pass-C colour-restore quad, MSAA on WebGL only (same transient membership). */
  readonly restoreQuad = createScreenQuad(
    'PostProcessing.refractionRestoreQuad',
    RENDER_LAYER_DEFAULT
  );
  /** The private scene pass G renders: depth-only proxies of the refracting glass. */
  readonly glassDepthScene = new THREE.Scene();
  /** The blit scene (one quad, orthographic camera), WebGL only. */
  readonly blitScene = new THREE.Scene();
  /** The glass front-face depth target, wrapping the shared depth texture. */
  readonly glassDepthTarget: THREE.WebGLRenderTarget;
  private readonly blitQuad = createScreenQuad(
    'PostProcessing.refractionBlitQuad',
    RENDER_LAYER_DEFAULT
  );
  private readonly blitCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly copyTarget: THREE.WebGLRenderTarget;
  private readonly depthOnlyMaterial = createDepthOnlyMaterial();
  private readonly proxies: THREE.Mesh[] = [];
  private readonly apiSurface: 'webgl2' | 'webgpu';
  private readonly collectRefractingGlass: (out: THREE.Mesh[]) => THREE.Mesh[];
  private readonly collectUnpartitionedMeshes: (out: THREE.Mesh[]) => THREE.Mesh[];
  private readonly setGlassPartition: (mode: GlassPartition) => void;
  /** Pass B's transmission target scale (see the module doc). */
  readonly transmissionResolutionScale: number;
  /** Frames in which the split ran (a diagnostic; the debug surface reads it). */
  framesSplit = 0;
  /** `render()` calls the last split frame issued (4, 5 or 6 by backend and MSAA). */
  lastPassCount = 0;
  private needsPrime = true;
  private warnedFallback = false;
  private readonly glassScratch: THREE.Mesh[] = [];
  private readonly unpartitionedScratch: THREE.Mesh[] = [];
  private readonly maskScratch: number[] = [];
  private readonly unpartitionedMaskScratch: number[] = [];

  constructor(opts: DataRefractionSplitOptions) {
    this.copyTarget = createRefractionCopyTarget(opts.width, opts.height);
    this.glassDepthTarget = createGlassDepthTarget(opts.width, opts.height);
    this.transmissionResolutionScale = opts.transmissionResolutionScale;
    this.apiSurface = opts.apiSurface;
    this.collectRefractingGlass = opts.collectRefractingGlass;
    this.collectUnpartitionedMeshes = opts.collectUnpartitionedMeshes;
    this.setGlassPartition = opts.setGlassPartition;
    this.blitScene.add(this.blitQuad);
  }

  /** Whether this renderer needs the copy + quad dance (three's WebGLRenderer). */
  get isWebGL(): boolean {
    return this.apiSurface === 'webgl2';
  }

  /** Follow the HDR target's physical size; the depth texture follows on next bind. */
  setSize(width: number, height: number): void {
    this.copyTarget.setSize(width, height);
    this.glassDepthTarget.setSize(width, height);
    this.needsPrime = true;
  }

  /**
   * Bind and clear the glass depth target once, so its GPU texture exists at full size
   * before any data material binds the shared depth texture this frame (see the module
   * doc). Idempotent until the next resize.
   */
  prime(renderer: Renderer): void {
    if (!this.needsPrime) return;
    const r = renderer as unknown as SplitRenderer;
    const prev = r.getRenderTarget();
    r.setRenderTarget(this.glassDepthTarget);
    r.clear();
    r.setRenderTarget(prev);
    this.needsPrime = false;
  }

  /**
   * Render `scene` into `hdrTarget` (already bound and cleared by the caller) with the
   * refracting glass drawn after — and sampling — the data behind it, and the data in
   * front of it drawn last. Returns false, having drawn nothing, when no visible glass
   * asks for it (or the one unsupported renderer + MSAA combination is in force), so
   * the caller renders normally.
   */
  render(
    renderer: Renderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    hdrTarget: THREE.WebGLRenderTarget
  ): boolean {
    this.prime(renderer);
    const glass = this.collectRefractingGlass(this.glassScratch);
    if (glass.length === 0) return false;
    const r = renderer as unknown as SplitRenderer;
    const msaa = hdrTarget.samples > 0;
    if (msaa && !this.isWebGL && r.backend?.isWebGLBackend === true) {
      this.warnFallbackOnce();
      return false;
    }
    const unpartitioned = this.collectUnpartitionedMeshes(this.unpartitionedScratch);
    this.syncProxies(glass);
    const borrowed = this.borrow(r, camera, glass, unpartitioned);
    try {
      this.passGlassDepth(r, camera);
      this.passBehind(r, scene, camera, hdrTarget);
      this.passGlass(r, scene, camera, hdrTarget, borrowed.scale);
      this.passFront(r, scene, camera, hdrTarget, msaa);
    } finally {
      this.restore(r, scene, camera, borrowed);
    }
    this.framesSplit++;
    return true;
  }

  /** Save every piece of renderer / scene state the passes write, and park the meshes. */
  private borrow(
    r: SplitRenderer,
    camera: THREE.Camera,
    glass: readonly THREE.Mesh[],
    unpartitioned: readonly THREE.Mesh[]
  ): BorrowedState {
    const masks = this.maskScratch;
    masks.length = 0;
    for (const mesh of glass) {
      masks.push(mesh.layers.mask);
      mesh.layers.set(RENDER_LAYER_REFRACTING_GLASS);
    }
    const unpartitionedMasks = this.unpartitionedMaskScratch;
    unpartitionedMasks.length = 0;
    for (const mesh of unpartitioned) {
      unpartitionedMasks.push(mesh.layers.mask);
      mesh.layers.set(RENDER_LAYER_UNPARTITIONED);
    }
    return {
      cameraMask: camera.layers.mask,
      autoClear: r.autoClear,
      scale: r.transmissionResolutionScale,
    };
  }

  /**
   * Hand everything back — the partition mode FIRST, so an exception can never leave
   * the data materials discarding. `scene.remove` of a quad that is not a child is a
   * no-op, so the quads need no bookkeeping. The meshes are the frame's scratch lists
   * (`glassScratch` / `unpartitionedScratch`), parallel to the saved-mask lists.
   */
  private restore(
    r: SplitRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    borrowed: BorrowedState
  ): void {
    this.setGlassPartition(GLASS_PARTITION_OFF);
    scene.remove(this.screenQuad);
    scene.remove(this.restoreQuad);
    camera.layers.mask = borrowed.cameraMask;
    r.autoClear = borrowed.autoClear;
    if (borrowed.scale !== undefined) r.transmissionResolutionScale = borrowed.scale;
    const glass = this.glassScratch;
    for (let i = 0; i < glass.length; i++) glass[i].layers.mask = this.maskScratch[i];
    const unpartitioned = this.unpartitionedScratch;
    for (let i = 0; i < unpartitioned.length; i++) {
      unpartitioned[i].layers.mask = this.unpartitionedMaskScratch[i];
    }
  }

  /** Pass G: the refracting glass's front-face depth, via the proxies. */
  private passGlassDepth(r: SplitRenderer, camera: THREE.Camera): void {
    camera.layers.set(RENDER_LAYER_DEFAULT);
    r.setRenderTarget(this.glassDepthTarget);
    r.autoClear = true;
    r.render(this.glassDepthScene, camera);
    this.lastPassCount = 1;
  }

  /** Pass A: everything except the refracting glass; data behind-or-none only. */
  private passBehind(
    r: SplitRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    hdrTarget: THREE.WebGLRenderTarget
  ): void {
    this.setGlassPartition(GLASS_PARTITION_BEHIND);
    camera.layers.mask = (1 << RENDER_LAYER_DEFAULT) | (1 << RENDER_LAYER_UNPARTITIONED);
    r.setRenderTarget(hdrTarget);
    r.autoClear = true;
    r.render(scene, camera);
    this.lastPassCount++;
  }

  /**
   * Pass B: the glass alone (plus, on WebGL, the copy of pass A behind it as a screen
   * quad) into the HDR target without clearing, at the pass-B transmission scale.
   */
  private passGlass(
    r: SplitRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    hdrTarget: THREE.WebGLRenderTarget,
    prevScale: number | undefined
  ): void {
    if (this.isWebGL) {
      this.blit(r, hdrTarget);
      this.screenQuad.material.uniforms.map.value = this.copyTarget.texture;
      scene.add(this.screenQuad);
    }
    r.setRenderTarget(hdrTarget);
    r.autoClear = false;
    camera.layers.set(RENDER_LAYER_REFRACTING_GLASS);
    if (prevScale !== undefined) r.transmissionResolutionScale = this.transmissionResolutionScale;
    r.render(scene, camera);
    this.lastPassCount++;
    if (prevScale !== undefined) r.transmissionResolutionScale = prevScale;
    scene.remove(this.screenQuad);
  }

  /**
   * Pass C: the data in front of the glass, crisp on top. Under MSAA on WebGL the
   * target's colour was invalidated after pass B, so a fresh copy is repainted first.
   */
  private passFront(
    r: SplitRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    hdrTarget: THREE.WebGLRenderTarget,
    msaa: boolean
  ): void {
    if (this.isWebGL && msaa) {
      this.blit(r, hdrTarget);
      this.restoreQuad.material.uniforms.map.value = this.copyTarget.texture;
      scene.add(this.restoreQuad);
    }
    this.setGlassPartition(GLASS_PARTITION_FRONT);
    camera.layers.set(RENDER_LAYER_DEFAULT);
    r.setRenderTarget(hdrTarget);
    r.autoClear = false;
    r.render(scene, camera);
    this.lastPassCount++;
  }

  /** Release both targets, the proxies' material and the quads (not the shared depth texture's owner). */
  dispose(): void {
    this.copyTarget.dispose();
    this.glassDepthTarget.dispose();
    this.depthOnlyMaterial.dispose();
    this.proxies.length = 0;
    this.glassDepthScene.clear();
    this.screenQuad.geometry.dispose();
    this.screenQuad.material.dispose();
    this.restoreQuad.geometry.dispose();
    this.restoreQuad.material.dispose();
    this.blitQuad.geometry.dispose();
    this.blitQuad.material.dispose();
  }

  /** `hdrTarget` colour → the copy target (WebGL only). */
  private blit(r: SplitRenderer, hdrTarget: THREE.WebGLRenderTarget): void {
    this.blitQuad.material.uniforms.map.value = hdrTarget.texture;
    r.setRenderTarget(this.copyTarget);
    r.autoClear = true;
    r.render(this.blitScene, this.blitCamera);
    this.lastPassCount++;
  }

  /**
   * One depth-only proxy per refracting glass this frame, sharing its geometry and
   * carrying its world matrix (the proxies are direct children of an identity scene,
   * so `matrix` IS the world matrix). Surplus proxies from a busier frame are hidden.
   */
  private syncProxies(glass: readonly THREE.Mesh[]): void {
    for (let i = 0; i < glass.length; i++) {
      let proxy = this.proxies[i];
      if (!proxy) {
        proxy = new THREE.Mesh(glass[i].geometry, this.depthOnlyMaterial);
        proxy.name = 'PostProcessing.glassDepthProxy';
        proxy.matrixAutoUpdate = false;
        proxy.frustumCulled = false;
        this.proxies.push(proxy);
        this.glassDepthScene.add(proxy);
      }
      proxy.geometry = glass[i].geometry;
      proxy.matrix.copy(glass[i].matrixWorld);
      proxy.matrixWorld.copy(glass[i].matrixWorld);
      proxy.visible = true;
    }
    for (let i = glass.length; i < this.proxies.length; i++) this.proxies[i].visible = false;
  }

  private warnFallbackOnce(): void {
    if (this.warnedFallback) return;
    this.warnedFallback = true;
    log.warning(
      Modules.POST_PROCESSING,
      'refract_data: WebGPURenderer on its WebGL2 fallback invalidates multisampled ' +
        'colour between passes, so with MSAA on the data partition is skipped (data in ' +
        'front of a refracting glass is painted over). Turn MSAA off or use native WebGPU.'
    );
  }
}
