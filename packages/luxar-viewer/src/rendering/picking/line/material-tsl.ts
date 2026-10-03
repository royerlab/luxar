/**
 * TSL / NodeMaterial counterpart to `LinePickingMaterial`.
 *
 * Mirrors the GLSL `LinePickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` / `dispose`
 * surface, same `CameraAwareMaterial` contract.
 *
 * **Uniform plumbing.** This class owns one `UniformNode` per shader
 * input. The public `uniforms` record exposes each node as an
 * `IUniform`-shaped getter/setter proxy (see `proxyIUniform` in
 * `tsl-helpers.ts`). Mutations to `material.uniforms.uX.value`
 * therefore land directly on `node.value` — no per-render
 * `.onUpdate` callback bridge.
 *
 * @module rendering/picking/line/material-tsl
 */

import * as THREE from 'three';
import { uniform, texture } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { linePickWebGPUFactory, type LinePickTSLConfig } from './pick.tsl';
import { capsuleLinePickWebGPUFactory } from './pick-capsule.tsl';
import type { LineJoinStyle } from '../../../types/line-join';
import { resolveLinePrimitive, type LinePrimitive } from '../../../types/line-primitive';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import {
  createPickVisibilityTSLNodes,
  proxyPickVisibilityUniforms,
} from '../_shared/visibility-tsl';
import { copyPickVisibilityUniforms } from '../_shared/visibility-uniforms';
import type { SurfacePickAwareMaterial } from '../_shared/surface-pick';
import {
  getPlaceholderElementTexture,
  LINE_TEXTURE_LAYOUT,
  resolveElementTextureWidth,
} from '../../element-texture-layout';
import { applySharedPickGraph } from '../_shared/shared-pick-graph-tsl';
import {
  isOrthographicProjection,
  type ProjectionVariantMaterial,
} from '../../materials/_shared/projection-variant';
import type { LinePickingMaterialConfig } from './material';

/** A camera whose projection reads orthographic (for `clone()`'s variant carry-over). */
const ORTHO_PROBE = new THREE.OrthographicCamera();

export class LinePickingTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, SurfacePickAwareMaterial, ProjectionVariantMaterial
{
  uniforms: Record<string, THREE.IUniform>;

  /** The projection kind of the camera last drawn with (quad variant only). */
  private _orthoVariant = false;

  private tslNodes: {
    uLineTex: TSLNode;
    uResolution: TSLNode;
    uPixelRatio: TSLNode;
    uNearCull: TSLNode;
    uMaxLinePixelWidth: TSLNode;
    uNodeId: TSLNode;
    uSortedIndexSlot: TSLNode;
    uDensityDrop: TSLNode;
    uIntensity: TSLNode;
    uOpacity: TSLNode;
    uHasElementAlpha: TSLNode;
    uVolumetric: TSLNode;
    uSurfaceDepth: TSLNode;
  };

  constructor(config: LinePickingMaterialConfig) {
    super();

    this.tslNodes = {
      // Line data texture node. Starts on the shared placeholder; the
      // commit's material sync rebinds the acquired pool entry's
      // texture via `updateLineTexture` (fresh node + graph rebuild —
      // TSL texture() captures the Texture at build time).
      uLineTex: texture(getPlaceholderElementTexture()),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uPixelRatio: uniform(1),
      // 0.1 matches the visual line material ctor default (pre-first-broadcast only).
      uNearCull: uniform(0.1),
      uMaxLinePixelWidth: uniform(540),
      uNodeId: uniform(config.nodeId),
      // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
      // The pick pass MUST track the visual one — it emits `vElementId`
      // from the same index, so reading the other buffer resolves hovers
      // against a stale permutation. Pushed by the depth-sort
      // coordinator's `syncSortedIndexSlot`, which finds it through
      // `uniforms` below.
      uSortedIndexSlot: uniform(0),
      uDensityDrop: uniform(0),
      // Visual-pass weight inputs, neutral until the first pick render
      // syncs the node's own (picking-system/visibility-sync.ts).
      ...createPickVisibilityTSLNodes(),
      // 0 = brightness-as-depth, 1 = real fragment depth (front-most wins;
      // opaque/normal) — mirrors the GLSL wrapper.
      uSurfaceDepth: uniform(0),
    };

    // Join style — a BUILD-time graph variant (see pick.tsl.ts), so it is
    // stashed on userData before the first factory call and re-read by every
    // rebuild. Stored unresolved so the ?lineJoin= session override still wins
    // at build time. MUST match the visual material's value: the two build the
    // same screen-space quad.
    this.userData.lineJoin = config.join;
    // Line primitive (#1352) — likewise a build-time variant, dispatching
    // between the pick factories on every rebuild. UNLIKE lineJoin above
    // this is stamped RESOLVED (exactly like the visual TSL material):
    // rebuilds and clone() must never re-run a per-node policy decision
    // made at first build.
    this.userData.linePrimitive = resolveLinePrimitive(config.primitive);

    this.uniforms = {
      // WARNING: a direct `uniforms.uLineTex.value = tex` write does NOT
      // rebind — `updateLineTexture()` is the only rebind chokepoint.
      uLineTex: proxyIUniform(this.tslNodes.uLineTex),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uPixelRatio: proxyIUniform(this.tslNodes.uPixelRatio),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxLinePixelWidth: proxyIUniform(this.tslNodes.uMaxLinePixelWidth),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uSortedIndexSlot: proxyIUniform(this.tslNodes.uSortedIndexSlot),
      uDensityDrop: proxyIUniform(this.tslNodes.uDensityDrop),
      ...proxyPickVisibilityUniforms(this.tslNodes),
      uSurfaceDepth: proxyIUniform(this.tslNodes.uSurfaceDepth),
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;
    // Picking is opaque so the transparent-and-DoubleSide two-pass
    // guard never trips, but setting `forceSinglePass` explicitly
    // matches the visual material and documents intent.
    this.forceSinglePass = true;

    this._rebuild();
  }

  /**
   * Build the per-rebuild factory config from current uniforms.
   */
  private _currentConfig(): LinePickTSLConfig {
    const quad =
      resolveLinePrimitive(this.userData.linePrimitive as LinePrimitive | undefined) !== 'capsule';
    return {
      join: this.userData.lineJoin as LineJoinStyle | undefined,
      // The quad's compile-time projection variant (see selectProjectionVariant).
      ...(quad ? { projection: this._orthoVariant ? 'ortho' : 'perspective' } : {}),
    };
  }

  /**
   * Select the screen-space quad's projection variant for the camera this pick
   * draw uses — the visual twin's rule (`LineTSLMaterial.selectProjectionVariant`):
   * a re-point at the other kind's cached shared graph, only on a kind change,
   * never driven by a CPU push. The capsule ignores it.
   */
  selectProjectionVariant(camera: THREE.Camera): void {
    const ortho = isOrthographicProjection(camera);
    if (ortho === this._orthoVariant) return;
    this._orthoVariant = ortho;
    if (
      resolveLinePrimitive(this.userData.linePrimitive as LinePrimitive | undefined) === 'capsule'
    ) {
      return;
    }
    this._rebuild();
    this.needsUpdate = true;
  }

  /**
   * (Re)build the pick graph, dispatching on the line primitive (#1352).
   * Every build site — constructor, texture rebind — funnels through here so the two factories can never drift.
   */
  private _rebuild(): void {
    const primitive = resolveLinePrimitive(
      this.userData.linePrimitive as LinePrimitive | undefined
    );
    // ONE graph per configuration (`../_shared/shared-pick-graph-tsl.ts`):
    // the primitive, the build-time join and projection variants and the baked
    // line-texture width are what select code. The BOUND texture's width is
    // handed to the factory explicitly: the shared graph's own leaf
    // is a forwarding twin over a stand-in texture.
    const config = {
      ...this._currentConfig(),
      elementTextureWidth: resolveElementTextureWidth(
        LINE_TEXTURE_LAYOUT,
        this.tslNodes.uLineTex.value as { image?: { width?: number } } | null
      ),
    };
    const key = { primitive, ...config };
    applySharedPickGraph(this, 'line-pick', key, this.tslNodes, (inputs, scratch) => {
      if (primitive === 'capsule') {
        capsuleLinePickWebGPUFactory(inputs, config, scratch);
      } else {
        linePickWebGPUFactory(inputs, config, scratch);
      }
    });
  }

  /**
   * Clone this picking material. Mirrors the GLSL wrapper's explicit
   * clone (the inherited `Material.clone()` calls the constructor with
   * no config and would throw; `NodeMaterial.copy` would alias the
   * source's node graph instead of this instance's own uniform nodes).
   */
  clone(): this {
    const cloned = new LinePickingTSLMaterial({
      nodeId: this.uniforms.uNodeId.value as number,
      // Graph variants — must ride the CONSTRUCTOR, not a post-hoc copy.
      join: this.userData.lineJoin as LineJoinStyle | undefined,
      primitive: this.userData.linePrimitive as LinePrimitive | undefined,
    });
    // Rebind the line data texture (no-op when still on the placeholder).
    const lineTex = this.uniforms.uLineTex?.value as THREE.DataTexture | null | undefined;
    if (lineTex) cloned.updateLineTexture(lineTex);
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    cloned.uniforms.uDensityDrop.value = this.uniforms.uDensityDrop.value;
    copyPickVisibilityUniforms(this.uniforms, cloned.uniforms);
    cloned.uniforms.uSurfaceDepth.value = this.uniforms.uSurfaceDepth.value;
    // Carry the last-drawn projection variant (no switch on the clone's first draw).
    if (this._orthoVariant) cloned.selectProjectionVariant(ORTHO_PROBE);
    return cloned as this;
  }

  /**
   * Select the pick depth convention (`SurfacePickAwareMaterial`): `true`
   * under the depth-ordered surface modes (`opaque` / `normal`) writes the
   * real projected depth so the FRONT-MOST element wins, as the user sees
   * it; `false` (default) keeps brightness-as-depth so the BRIGHTEST wins,
   * right for the commutative modes. Synced per pick render by
   * `PickingSystem.renderPickBuffer()`.
   */
  setSurfacePickDepth(on: boolean): void {
    this.uniforms.uSurfaceDepth.value = on ? 1 : 0;
  }

  /**
   * Camera-dependent uniforms. The ortho branch is read in the graph from the
   * projection matrix of the draw (`isOrthoProjectionTSL()`, the GLSL twin's
   * `luxarLineIsOrtho`), so none is pushed.
   */
  updateCameraParams(resolution: THREE.Vector2, nearCull?: number, pixelRatio: number = 1): void {
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    // Accept ANY defined value, including 0 — matching the point/gsplat
    // wrappers (the shader floors at 1e-20). The old `> 0` gate silently
    // KEPT a stale value on zero-diagonal scenes (or, with LRU-cached
    // materials, the previous dataset's nearCull), re-creating the
    // cross-geometry near-fade divergence B9c fixed.
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
    this.uniforms.uPixelRatio.value = pixelRatio;
  }

  /**
   * Rebind the line data texture. TSL `texture()` captures the
   * THREE.Texture at factory time, so an identity change needs a fresh
   * node + graph rebuild. Mirrors `PointPickingTSLMaterial.
   * updatePointTexture`. No-op when unchanged (the common per-commit
   * case).
   */
  updateLineTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uLineTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uLineTex = texture(next);
    this.uniforms.uLineTex = proxyIUniform(this.tslNodes.uLineTex);
    this._rebuild();
    this.needsUpdate = true;
  }
}
