/**
 * LayerApplyEngine — pushes layers-panel state into scene materials.
 *
 * The scene-application half of the layers panel, extracted from
 * `layers-panel.ts`: given a layer whose panel state changed, resolve every
 * affected data-leaf (the layer itself, or every data descendant for a group
 * layer), recompose its effective attrs along the scene-graph ancestry per the
 * Luxar composition spec (opacity/gamma/intensity multiply, offset adds,
 * blending_mode takes the nearest ancestor's choice), and push the result into
 * the leaf material (clone-on-first-use so shared cached materials are never
 * mutated in place). Authoring-time zarr values are used for non-layer nodes
 * in the chain; live panel state overrides them for `layer=True` nodes.
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
import { applyColorAdjustments, isLuxarMaterial, type LuxarMaterial } from './luxar-material';

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
   */
  private composeEffective(leafPath: string): EffectiveAttrs | null {
    const sceneGraph = this.deps.getSceneGraph();
    if (!sceneGraph) return null;
    const ancestors = collectAncestorNodes(sceneGraph, leafPath);
    const chain: ComposableAttrs[] = ancestors.map((node) => {
      const layerInfo = this.deps.state.getLayer(node.path);
      if (layerInfo) return this.liveLayerAttrs(layerInfo);
      return {
        opacity: node.attrs.opacity as number | undefined,
        absorption: node.attrs.absorption as number | undefined,
        gamma: node.attrs.gamma as number | undefined,
        intensity: node.attrs.intensity as number | undefined,
        offset: node.attrs.offset as number | undefined,
        blending_mode: node.attrs.blending_mode as string | undefined,
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
        uniforms?: { opacity?: { value?: number }; uOpacity?: { value?: number } };
      }
    ).uniforms;
    const liveOpacity = opacityUniform?.opacity?.value ?? opacityUniform?.uOpacity?.value ?? 1.0;
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
      const eff = this.composeEffective(leaf.path);
      if (!eff) continue;
      mat.updateOpacity(eff.opacity);
      // Optional-chained: only gsplat materials implement it in phase 1
      // (points/lines render volumetric's additive fallback, where κ is
      // inert anyway).
      mat.updateAbsorption?.(eff.absorption);
      applyColorAdjustments(mat, eff.gamma, eff.intensity, eff.offset);
      const prevBlendingMode = mat.userData?.blendingMode as BlendingMode | undefined;
      this.applyBlendingStateToMaterial(mat, eff.blending_mode);
      // Depth sorting: a sortable layer switching blending mode may need
      // to start (TO an effective sorted mode: clear the noop stamp +
      // reprocess so the next commit registers with the SortWorker) or
      // stop (AWAY: release) depth sorting. Gsplats + points today;
      // lines join when their texture storage lands (the coordinator
      // has no centers pipeline for them yet, so a reprocess would be
      // pure waste).
      const sortableType = obj.userData?.nodeType;
      if (sortableType === 'gsplats' || sortableType === 'points') {
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
   */
  applyColormap(layer: LayerInfo): void {
    const leaves = this.getAffectedDataLeaves(layer.path);
    if (leaves.length === 0) return;

    const tex = layer.colormap ? getColormapTexture(layer.colormap) : null;
    for (const leaf of leaves) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat || !mat.updateColormapTexture) continue;
      if (layer.colormap && tex) {
        // C1 fail-closed guard: enabling USE_COLORMAP requires the right
        // scalar attribute on geometry (`scalar` for points,
        // `aStartScalar`/`aEndScalar` for lines, `aAmplitude` for gsplats).
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
        // The scalar window (value→LUT mapping) is driven by the display
        // range, not a static attr — recover it from the composed
        // gain/offset so it matches what `applyComposed` will push. Falls
        // back to the authored scalar range when no composition exists.
        if (mat.updateScalarRange) {
          const eff = this.composeEffective(leaf.path);
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
  }
}
