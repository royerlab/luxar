/**
 * The WebGL scene-pass split that lets `refract_data` glass refract the emissive data
 * (spec MESH_PHYSICAL_MATERIALS §3.4, Phase 3).
 *
 * Three's WebGL `renderTransmissionPass` draws only the OPAQUE render list into the
 * texture a transmissive material samples, and every Luxar point, line and splat
 * material is transparent — so a glass mesh can never see the data behind it in one
 * pass. (WebGPU needs none of this: there the glass samples the framebuffer as it stands
 * when it draws, and ranking it last in its band — `render-order.ts` — is the whole
 * mechanism; a `WebGPURenderer` therefore never owns one of these.)
 *
 * The split renders the scene twice into the same HDR target:
 *
 * 1. **Pass A** — everything except the refracting glass, exactly as the single pass
 *    used to (the glass sits on {@link RENDER_LAYER_REFRACTING_GLASS} for the duration,
 *    the camera on the default layer).
 * 2. **Blit** — pass A's colour is copied into a second target: pass B must read it while
 *    writing the HDR target, and a texture cannot be both.
 * 3. **Pass B** — only the glass plus one screen-space quad textured with that copy, into
 *    the HDR target WITHOUT clearing. The quad is an OPAQUE object, so three's
 *    transmission pass draws it into the transmission texture and the glass refracts
 *    the data; in the main pass the quad repaints pixels that are already there. The
 *    depth buffer from pass A survives, so an opaque mesh in front still occludes the
 *    glass.
 *
 * **The copy is load-bearing under MSAA, not an optimisation.** Three resolves a
 * multisampled target's colour to its texture at the end of every `render()` and then
 * INVALIDATES the multisampled colour attachment (the depth attachment is resolved and
 * kept, `resolveDepthBuffer` defaults to true). Pass B therefore draws onto undefined
 * colour until the full-screen quad has repainted every pixel from the copy — which is
 * why the quad has `depthTest: false`, `depthWrite: false`, `transparent: false` and
 * `frustumCulled: false`, and why those four are pinned by test.
 *
 * Everything the split touches is restored in a `finally` before `render()` returns —
 * the glass meshes' layer masks, the camera mask, `autoClear`, the renderer's
 * `transmissionResolutionScale`, and the quad's membership of the scene — so the pick
 * pass, the environment cube capture, the blend warm-up and scene disposal never see
 * any of it. The scene holds no lights today; a future light would need
 * `layers.enableAll()` to reach pass B.
 *
 * `transmissionResolutionScale` (config `renderingControls.refraction`) applies to
 * pass B only: three reads it per transmission pass, so Phase 2 glass in pass A keeps
 * three's default and its pixels are untouched.
 *
 * @module rendering/post-processing/post-processing-manager/refraction-split
 */

import * as THREE from 'three';
import type { Renderer } from '../../renderer-capabilities';
import { RENDER_LAYER_DEFAULT, RENDER_LAYER_REFRACTING_GLASS } from '../../render-layers';

/** How the split is built (all from the orchestrator). */
export interface DataRefractionSplitOptions {
  /** Physical width of the HDR target the copy must match. */
  readonly width: number;
  /** Physical height of the HDR target the copy must match. */
  readonly height: number;
  /** Pass B's `renderer.transmissionResolutionScale`, in `(0, 1]`. */
  readonly transmissionResolutionScale: number;
  /**
   * The visible refracting glass this frame, into a caller-owned array (the depth-sort
   * coordinator's `collectRefractingGlass`). Injected so the split owns no scene
   * knowledge and a unit test can hand it any list.
   */
  readonly collectRefractingGlass: (out: THREE.Mesh[]) => THREE.Mesh[];
}

/** The slice of `THREE.WebGLRenderer` the split writes; `WebGPURenderer` lacks the scale. */
interface SplitRenderer {
  render(scene: THREE.Object3D, camera: THREE.Camera): void;
  setRenderTarget(target: THREE.WebGLRenderTarget | null): void;
  autoClear: boolean;
  transmissionResolutionScale?: number;
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

/** Allocate the colour-only copy of pass A (same format as the HDR target, no depth). */
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
 * A screen-space quad whose vertex shader ignores the camera, so it covers the whole
 * viewport of whatever target it is drawn into — including three's transmission
 * target, which is what makes the data visible to the glass. See the module doc for
 * why every one of the four flags is load-bearing.
 */
function createScreenQuad(name: string): THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial> {
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
  return quad;
}

/** Owns the copy target and the two quads; one per `PostProcessingManager`, WebGL only. */
export class DataRefractionSplit {
  /** The pass-B quad (a scene member only between pass B's start and its `finally`). */
  readonly screenQuad = createScreenQuad('PostProcessing.refractionScreenQuad');
  private readonly blitQuad = createScreenQuad('PostProcessing.refractionBlitQuad');
  private readonly blitScene = new THREE.Scene();
  private readonly blitCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly copyTarget: THREE.WebGLRenderTarget;
  private readonly collectRefractingGlass: (out: THREE.Mesh[]) => THREE.Mesh[];
  /** Pass B's transmission target scale (see the module doc). */
  readonly transmissionResolutionScale: number;
  /** Frames in which the split ran (a diagnostic; the debug surface reads it). */
  framesSplit = 0;
  private readonly glassScratch: THREE.Mesh[] = [];
  private readonly maskScratch: number[] = [];

  constructor(opts: DataRefractionSplitOptions) {
    this.copyTarget = createRefractionCopyTarget(opts.width, opts.height);
    this.transmissionResolutionScale = opts.transmissionResolutionScale;
    this.collectRefractingGlass = opts.collectRefractingGlass;
    this.blitScene.add(this.blitQuad);
    this.screenQuad.layers.set(RENDER_LAYER_REFRACTING_GLASS);
  }

  /** Follow the HDR target's physical size (no MSAA to re-create for). */
  setSize(width: number, height: number): void {
    this.copyTarget.setSize(width, height);
  }

  /**
   * Render `scene` into `hdrTarget` (already bound and cleared by the caller) with the
   * refracting glass drawn after — and sampling — the data. Returns false, having drawn
   * nothing, when no visible glass asks for it, so the caller renders normally.
   */
  render(
    renderer: Renderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    hdrTarget: THREE.WebGLRenderTarget
  ): boolean {
    const glass = this.collectRefractingGlass(this.glassScratch);
    if (glass.length === 0) return false;
    const r = renderer as unknown as SplitRenderer;

    const prevCameraMask = camera.layers.mask;
    const prevAutoClear = r.autoClear;
    const prevScale = r.transmissionResolutionScale;
    const masks = this.maskScratch;
    masks.length = 0;
    for (const mesh of glass) {
      masks.push(mesh.layers.mask);
      mesh.layers.set(RENDER_LAYER_REFRACTING_GLASS);
    }
    let quadAdded = false;
    try {
      // Pass A: everything except the glass → hdrTarget (bound + cleared by the caller).
      camera.layers.set(RENDER_LAYER_DEFAULT);
      r.render(scene, camera);

      // Copy pass A's colour: pass B reads it while writing hdrTarget.
      this.blitQuad.material.uniforms.map.value = hdrTarget.texture;
      r.setRenderTarget(this.copyTarget);
      r.render(this.blitScene, this.blitCamera);

      // Pass B: glass + screen quad → hdrTarget, keeping pass A's depth (and, under
      // MSAA, repainting its colour from the copy — see the module doc).
      this.screenQuad.material.uniforms.map.value = this.copyTarget.texture;
      scene.add(this.screenQuad);
      quadAdded = true;
      r.setRenderTarget(hdrTarget);
      r.autoClear = false;
      camera.layers.set(RENDER_LAYER_REFRACTING_GLASS);
      if (prevScale !== undefined) r.transmissionResolutionScale = this.transmissionResolutionScale;
      r.render(scene, camera);
    } finally {
      if (quadAdded) scene.remove(this.screenQuad);
      camera.layers.mask = prevCameraMask;
      r.autoClear = prevAutoClear;
      if (prevScale !== undefined) r.transmissionResolutionScale = prevScale;
      for (let i = 0; i < glass.length; i++) glass[i].layers.mask = masks[i];
    }
    this.framesSplit++;
    return true;
  }

  /** Release the copy target and both quads. */
  dispose(): void {
    this.copyTarget.dispose();
    this.screenQuad.geometry.dispose();
    this.screenQuad.material.dispose();
    this.blitQuad.geometry.dispose();
    this.blitQuad.material.dispose();
  }
}
