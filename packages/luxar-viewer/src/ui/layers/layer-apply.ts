/**
 * LayerApplyEngine — pushes layers-panel state into scene materials.
 *
 * The scene-application half of the layers panel, extracted from
 * `layers-panel.ts`: given a layer whose panel state changed, resolve every
 * affected data-leaf (the layer itself, or every data descendant for a group
 * layer), recompose its effective attrs along the scene-graph ancestry per the
 * Luxar composition spec (opacity/gamma/intensity multiply, offset adds,
 * blending_mode takes the nearest ancestor's choice — except INSIDE the edited
 * layer's own subtree, where the layer's single Blend control wins; see
 * `composeEffective`), and push the result into the leaf material
 * (clone-on-first-use so shared cached materials are never mutated in place).
 * Authoring-time zarr values are used for non-layer nodes in the chain; live
 * panel state overrides them for `layer=True` nodes.
 *
 * Constructed with ACCESSORS for the root group and scene graph (both are
 * reassigned by `LayersPanel.initFromScene` on every scene load) — never with
 * captured values, which would go stale on the second scene.
 */

import * as THREE from 'three';
import type { SceneNode } from '../../data/data-loader-types';
import type { BlendingMode } from '../../rendering';
import { materialManager } from '../../rendering';
import { log, Modules } from '../../utils/log';
import { getColormapTexture } from '../../rendering/colormap-textures';
import { supportsScalarColormap } from '../../rendering/material-colormap-helpers';
import { noteDepthSortBlendingModeSwitch } from '../../rendering/depth-sort-coordinator';
import {
  composeAttrs,
  collectAncestorNodes,
  collectDataDescendants,
  type ComposableAttrs,
  type EffectiveAttrs,
} from '../../data/attrs-composer';
import { getBlendingState, liveLayerAttrs as deriveLiveLayerAttrs } from './attrs-utils';
import { computeDisplayRange, type LayerInfo, type LayerStateManager } from './layer-state';
import {
  applyColorAdjustments,
  isColormapActive,
  isLuxarMaterial,
  type LuxarMaterial,
} from './luxar-material';

/**
 * Dependencies injected by the owning {@link LayersPanel}. `getRootGroup` /
 * `getSceneGraph` are accessors because both fields are replaced on every
 * `initFromScene`; `state` is the panel's (stable) layer-state manager and
 * `requestRender` wakes the on-demand render loop after a material write.
 */
export interface LayerApplyEngineDeps {
  getRootGroup: () => THREE.Group | null;
  getSceneGraph: () => SceneNode | null;
  state: LayerStateManager;
  requestRender: () => void;
}

export class LayerApplyEngine {
  constructor(private deps: LayerApplyEngineDeps) {}

  private getMesh(path: string): THREE.Object3D | null {
    const rootGroup = this.deps.getRootGroup();
    if (!rootGroup) return null;
    return rootGroup.getObjectByName(path) ?? null;
  }

  /**
   * Clone-on-first-use for the material at a data-leaf, registering the
   * clone with MaterialManager so camera-dependent uniforms stay current.
   * Non-luxar materials return null.
   */
  private getLeafMaterial(obj: THREE.Object3D): LuxarMaterial | null {
    const mesh = obj as THREE.Points | THREE.Mesh;
    if (!mesh.material) return null;
    const mat = mesh.material as THREE.Material;
    if (!isLuxarMaterial(mat)) return null;

    if (!mesh.userData._layerMaterialCloned) {
      const cloned = mat.clone() as LuxarMaterial;
      mesh.material = cloned;
      mesh.userData._layerMaterialCloned = true;
      materialManager.register(cloned);
      return cloned;
    }
    return mat as LuxarMaterial;
  }

  /**
   * Resolve every data-leaf affected by changes to a layer at `path`.
   * Data-node layers map to themselves; group layers fan out to all
   * descendant points/lines/gsplats.
   */
  private getAffectedDataLeaves(path: string): SceneNode[] {
    const sceneGraph = this.deps.getSceneGraph();
    if (!sceneGraph) return [];
    const chain = collectAncestorNodes(sceneGraph, path);
    const target = chain[chain.length - 1];
    if (!target) return [];
    if (target.type === 'group') return collectDataDescendants(target);
    return [target];
  }

  /**
   * Compute the layer's current live composable attributes. Thin wrapper
   * around {@link deriveLiveLayerAttrs} so `composeEffective` keeps its
   * compact `this.liveLayerAttrs(...)` shape.
   */
  private liveLayerAttrs(layer: LayerInfo): ComposableAttrs {
    return deriveLiveLayerAttrs(layer);
  }

  /**
   * Recompose the effective attrs for a single data-leaf by walking the
   * scene-graph ancestry, substituting panel state for every `layer=true`
   * node in the chain.
   *
   * `layerPath` is the layer whose control was just used. Inside that layer's
   * own subtree the LAYER owns `blending_mode`: a mode authored on a
   * descendant that is not itself a layer is dropped. `blending_mode` is
   * nearest-setter-wins and a layer exposes exactly one Blend control, so
   * without this a `kind=partition` / `kind=lod` layer whose parts carry
   * their own stamped mode has an inert control — every part shadows the
   * wrapper (the `graft_gsplat_node` stamping bug, and every scene already
   * written by it). A nested node that IS a layer keeps its live value: it
   * has its own control. Only `blending_mode` is affected — the
   * multiplicative attrs still compose and `offset` still sums, so a part's
   * authored opacity/gamma/κ is preserved.
   *
   * `identityLayerWindow` substitutes the IDENTITY for the edited layer's own
   * display window (intensity/offset) — used by `applyComposed` for a leaf
   * that renders direct colour while the layer's window is a SCALAR window
   * (a mixed group layer), so the scalar window is never applied as a colour
   * gain. Ancestor/leaf-authored windows still compose.
   */
  private composeEffective(
    leafPath: string,
    layerPath: string,
    identityLayerWindow = false
  ): EffectiveAttrs | null {
    const sceneGraph = this.deps.getSceneGraph();
    if (!sceneGraph) return null;
    const ancestors = collectAncestorNodes(sceneGraph, leafPath);
    // Required, not optional: an omitted `layerPath` would silently disable the
    // subtree rule below and re-open the inert-Blend-control bug.
    const layerDepth = ancestors.findIndex((n) => n.path === layerPath);
    const chain: ComposableAttrs[] = ancestors.map((node, i) => {
      const layerInfo = this.deps.state.getLayer(node.path);
      if (layerInfo) {
        const live = this.liveLayerAttrs(layerInfo);
        if (identityLayerWindow && node.path === layerPath) {
          return { ...live, intensity: 1, offset: 0 };
        }
        return live;
      }
      const insideLayerSubtree = layerDepth >= 0 && i > layerDepth;
      return {
        opacity: node.attrs.opacity as number | undefined,
        absorption: node.attrs.absorption as number | undefined,
        gamma: node.attrs.gamma as number | undefined,
        intensity: node.attrs.intensity as number | undefined,
        offset: node.attrs.offset as number | undefined,
        blending_mode: insideLayerSubtree
          ? undefined
          : (node.attrs.blending_mode as string | undefined),
      };
    });
    return composeAttrs(chain);
  }

  private applyBlendingStateToMaterial(mat: LuxarMaterial, mode: string): void {
    // All Luxar materials (Points, Lines, GSplats) now implement
    // `applyBlendingMode`. That single source of truth handles type-
    // specific concerns (GSplat `uProjectionMode`, Point
    // `LUXAR_MAX_RGB_CONTRIBUTION` define, max-mode `OneFactor` blend
    // factors) and is used by both creation (in MaterialManager) and
    // runtime UI transitions. The generic fallback below remains for
    // defensiveness against external/future materials that lack the
    // method, and now applies the *complete* state (including
    // blend factors) so it matches the canonical mapping.
    if (typeof mat.applyBlendingMode === 'function') {
      mat.applyBlendingMode(mode as BlendingMode);
      return;
    }

    const opacityUniform = (
      mat as unknown as {
        uniforms?: { uOpacity?: { value?: number } };
      }
    ).uniforms;
    const liveOpacity = opacityUniform?.uOpacity?.value ?? 1.0;
    const state = getBlendingState(mode, liveOpacity);
    mat.blending = state.blending;
    mat.depthTest = state.depthTest;
    mat.depthWrite = state.depthWrite;
    mat.transparent = state.transparent;
    mat.blendEquation = state.blendEquation;
    if (state.blendSrc !== undefined) mat.blendSrc = state.blendSrc;
    if (state.blendDst !== undefined) mat.blendDst = state.blendDst;
    mat.needsUpdate = true;
  }

  /**
   * Push each composed effective attribute (except colormap, which is
   * per-leaf and doesn't chain through ancestors) to every affected leaf
   * material. Colormap is handled separately because textures don't
   * compose — the nearest ancestor's colormap wins.
   */
  private applyComposed(layer: LayerInfo): void {
    const leaves = this.getAffectedDataLeaves(layer.path);
    if (leaves.length === 0) return;

    for (const leaf of leaves) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat) continue;
      // Per-leaf window routing. A layer whose window is a SCALAR window
      // (colormap in play) can still contain leaves rendering direct colour:
      // the C1 guard suppresses the LUT on geometry with no scalars bound,
      // and a MIXED group keeps the scalar window because some other leaf
      // accepted it. Pushing that window into a direct-colour leaf's colour
      // GOG is exactly the contrast stretch this panel no longer does — such
      // a leaf gets the identity window instead.
      const identityLayerWindow = layer.scalarWindow && !isColormapActive(mat);
      const eff = this.composeEffective(leaf.path, layer.path, identityLayerWindow);
      if (!eff) continue;
      // An in-flight LOD fade owns the live opacity uniform: it re-renders
      // `_lodFadeBase × fadeProduct` every frame (scene/lod-fade.ts), so a
      // direct uniform write here would be clobbered on the next fade frame
      // and the panel edit lost until the fade ends. Rebase the fade's
      // snapshot instead — the registry composes `newBase × product` on the
      // very next frame and restores `newBase` when the fade completes.
      if (obj.userData._lodFadeBase != null) {
        obj.userData._lodFadeBase = eff.opacity;
      } else {
        mat.updateOpacity(eff.opacity);
      }
      // All three geometry-material families implement it (gsplats
      // phase 1, points phase 3, lines phase 4); optional-chained for
      // non-Luxar materials.
      mat.updateAbsorption?.(eff.absorption);
      applyColorAdjustments(mat, eff.gamma, eff.intensity, eff.offset);
      const prevBlendingMode = mat.userData?.blendingMode as BlendingMode | undefined;
      this.applyBlendingStateToMaterial(mat, eff.blending_mode);
      // Depth sorting: a sortable layer switching blending mode may need
      // to start (TO an effective sorted mode: clear the noop stamp +
      // reprocess so the next commit registers with the SortWorker) or
      // stop (AWAY: release) depth sorting. All three geometry types
      // register centers with the coordinator (gsplats/points centers,
      // lines segment midpoints).
      const sortableType = obj.userData?.nodeType;
      if (sortableType === 'gsplats' || sortableType === 'points' || sortableType === 'lines') {
        noteDepthSortBlendingModeSwitch(obj as THREE.Mesh, eff.blending_mode, prevBlendingMode);
      }
    }
    this.deps.requestRender();
  }

  applyVisibility(path: string, visible: boolean): void {
    const obj = this.getMesh(path);
    if (obj) {
      obj.visible = visible;
      this.deps.requestRender();
    }
  }

  applyDisplayRange(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  applyGamma(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  applyOpacity(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  applyAbsorption(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  applyBlendingMode(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  /**
   * Colormap applies per-leaf (not composed). For a group-layer we push
   * the selected colormap to every data descendant that accepts one.
   *
   * Returns whether the layer now actually renders through a colormap — i.e.
   * at least one leaf accepted the LUT. Clearing a colormap always "takes", so
   * that returns `false` (no colormap in effect). The caller needs this because
   * the C1 fail-closed guard below can suppress the colormap on every leaf
   * (a group layer over scalar-less points still offers the dropdown): the
   * layer then keeps rendering DIRECT COLOUR, so its display window must stay
   * the direct-colour identity rather than move to a scalar range.
   */
  applyColormap(layer: LayerInfo): boolean {
    const leaves = this.getAffectedDataLeaves(layer.path);
    if (leaves.length === 0) return false;

    let colormapInEffect = false;
    const tex = layer.colormap ? getColormapTexture(layer.colormap) : null;
    for (const leaf of leaves) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat || !mat.updateColormapTexture) continue;
      if (layer.colormap && tex) {
        // C1 fail-closed guard: enabling USE_COLORMAP requires real
        // scalar data behind the geometry (the `userData.hasScalars`
        // stamp for points/lines texel storage; always true for gsplats,
        // whose amplitude is the scalar).
        const nodeType = leaf.type as 'points' | 'lines' | 'gsplats';
        const geometry = (obj as THREE.Points | THREE.Mesh).geometry as THREE.BufferGeometry;
        if (!supportsScalarColormap(nodeType, geometry)) {
          log.warning(
            Modules.UI,
            `[LayersPanel][${leaf.path}] Scalar colormap suppressed: required attribute(s) not bound on geometry (pending C4 implementation).`
          );
          continue;
        }
        mat.updateColormapTexture(tex);
        colormapInEffect = true;
        // The scalar window (value→LUT mapping) is driven by the display
        // range, not a static attr — recover it from the composed
        // gain/offset so it matches what `applyComposed` will push. Falls
        // back to the authored scalar range when no composition exists.
        if (mat.updateScalarRange) {
          const eff = this.composeEffective(leaf.path, layer.path);
          if (eff) {
            const { min, max } = computeDisplayRange(eff.intensity, eff.offset);
            mat.updateScalarRange(min, max);
          } else if (layer.scalarDataRange) {
            mat.updateScalarRange(layer.scalarDataRange[0], layer.scalarDataRange[1]);
          }
        }
      } else {
        mat.updateColormapTexture(null);
      }
      // do NOT mark `mat.needsUpdate = true` here. Material methods
      // (`updateColormapTexture`, `applyColormapTextureToMaterial`)
      // already toggle `needsUpdate` when defines change. Setting it
      // unconditionally for every per-leaf colormap apply caused
      // shader recompilation on every UI tick during group-layer
      // scalar-range drags, even when the colormap define hadn't
      // toggled.
    }
    // Enabling/disabling a colormap flips how display-range + gamma must
    // be routed (value window vs color GOG). Recompose so each affected
    // leaf's intensity/offset/scalar-range match its new mode — in
    // particular, restoring the color GOG when a colormap is turned off.
    this.applyComposed(layer);
    this.deps.requestRender();
    return colormapInEffect;
  }
}
