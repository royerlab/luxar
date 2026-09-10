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
import { scheduleBlendModeProgramWarmupForObject } from '../../rendering/webgl-blend-warmup';
import { log, Modules } from '../../utils/log';
import { getColormapTexture } from '../../rendering/colormap-textures';
import { supportsScalarColormap } from '../../rendering/material-colormap-helpers';
import { noteDepthSortBlendingModeSwitch } from '../../rendering/depth-sort-coordinator';
import { syncMeshPickAppearance } from '../../rendering/node-factory/create-mesh-node';
import {
  PHYSICAL_MESH_KNOB_KEYS,
  isPhysicalMeshMaterial,
} from '../../rendering/materials/mesh-physical/config';
import type { GeometryTypeName } from '../../types/format-contract';
import {
  defaultBlendingMode,
  isDepthSortable,
  isGeometryType,
} from '../../types/geometry-capabilities';
import {
  composeAttrs,
  collectAncestorNodes,
  collectDataDescendants,
  getEffectiveAttrs,
  type ComposableAttrs,
  type EffectiveAttrs,
} from '../../data/attrs-composer';
import { getBlendingState, liveLayerAttrs as deriveLiveLayerAttrs } from './attrs-utils';
import {
  computeDisplayRange,
  computeUniforms,
  type LayerInfo,
  type LayerStateManager,
} from './layer-state';
import { remapWindowToLeafRange } from '../../rendering/display-range';
import {
  applyColorAdjustments,
  isColormapActive,
  isLuxarMaterial,
  type LuxarMaterial,
} from './luxar-material';

/**
 * A gain contributes nothing: `intensity` multiplies (identity 1) and `offset`
 * adds (identity 0), and both are absent far more often than they are set.
 */
function isIdentityGain(intensity: number | undefined, offset: number | undefined): boolean {
  return (intensity ?? 1) === 1 && (offset ?? 0) === 0;
}

/**
 * Dependencies injected by the owning {@link LayersPanel}. `getRootGroup` /
 * `getSceneGraph` are accessors because both fields are replaced on every
 * `initFromScene`; `state` is the panel's (stable) layer-state manager and
 * `requestRender` wakes the on-demand render loop after a material write, and
 * `requestReprocess` refreshes a layer that became load-eligible again.
 */
export interface LayerApplyEngineDeps {
  getRootGroup: () => THREE.Group | null;
  getSceneGraph: () => SceneNode | null;
  state: LayerStateManager;
  requestRender: () => void;
  requestReprocess: (paths: readonly string[]) => void;
  /**
   * Marks the cached GPU pick buffer dirty so it re-renders after a panel edit
   * changed a mesh's pick coverage (opacity/cutoff/blending/physical knobs).
   * No-op when picking is inactive.
   */
  invalidatePickBuffer?: () => void;
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
   * own subtree — and ONLY when the edited layer actually OWNS a mode
   * (`blendingModeExplicit`) — the LAYER owns `blending_mode`: a mode authored on a
   * descendant that is not itself a layer is dropped. `blending_mode` is
   * nearest-setter-wins and a layer exposes exactly one Blend control, so
   * without this a `kind=partition` / `kind=lod` layer whose parts carry
   * their own stamped mode has an inert control — every part shadows the
   * wrapper (the `graft_gsplat_node` stamping bug, and every scene already
   * written by it). A wrapper that owns NO mode has no control value to impose,
   * so it must NOT suppress its descendants' authored modes — otherwise a plain
   * `layer=true` group over a mesh authored `additive` would snap the mesh to
   * its `opaque` type-default on any non-blend edit (#1275). A nested node that
   * IS a layer keeps its live value: it has its own control. Only
   * `blending_mode` is affected — the multiplicative attrs still compose and
   * `offset` still sums, so a part's authored opacity/gamma/κ is preserved.
   *
   * `identityLayerWindow` substitutes the IDENTITY for the edited layer's own
   * display window (intensity/offset) — used by `applyComposed` for a leaf
   * that renders direct colour while the layer's window is a SCALAR window
   * (a mixed group layer), so the scalar window is never applied as a colour
   * gain. Ancestor/leaf-authored windows still compose.
   *
   * `precomputedAncestors` lets a caller that already walked the chain (both
   * fan-out loops do) hand it over instead of paying for a second walk.
   * `collectAncestorNodes` resolves each step with a linear `children.find`, so
   * one fan-out over a P-part wrapper costs ~P²/2 path comparisons — and
   * `applyComposed` runs on every slider tick.
   */
  private composeEffective(
    leafPath: string,
    layerPath: string,
    identityLayerWindow = false,
    precomputedAncestors?: readonly SceneNode[]
  ): EffectiveAttrs | null {
    let ancestors: readonly SceneNode[];
    if (precomputedAncestors) {
      ancestors = precomputedAncestors;
    } else {
      const sceneGraph = this.deps.getSceneGraph();
      if (!sceneGraph) return null;
      ancestors = collectAncestorNodes(sceneGraph, leafPath);
    }
    // Required, not optional: an omitted `layerPath` would silently disable the
    // subtree rule below and re-open the inert-Blend-control bug.
    const layerDepth = ancestors.findIndex((n) => n.path === layerPath);
    // The subtree-drop suppresses descendant authored modes so the edited
    // layer's Blend control wins — but only when that layer actually OWNS a
    // mode. A wrapper owning none has nothing to impose (see doc comment; #1275).
    const editedLayerOwnsMode = this.deps.state.getLayer(layerPath)?.blendingModeExplicit ?? false;
    // Same rule as the mode, tracked separately: a layer exposes exactly ONE
    // Layer order control, so an order authored on a non-layer DESCENDANT would
    // make that control inert. A nested LAYER descendant is different — it has
    // its own control and its own row, so it goes through `liveLayerAttrs`
    // below and rightly wins by nearest-setter-wins.
    const editedLayerOwnsOrder = this.deps.state.getLayer(layerPath)?.layerOrderExplicit ?? false;
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
        blending_mode:
          insideLayerSubtree && editedLayerOwnsMode
            ? undefined
            : (node.attrs.blending_mode as string | undefined),
        // Omitting this dropped an order authored on a non-layer intermediate
        // group from composition entirely.
        layer_order:
          insideLayerSubtree && editedLayerOwnsOrder
            ? undefined
            : (node.attrs.layer_order as number | undefined),
      };
    });
    return composeAttrs(chain);
  }

  /**
   * May the composed window for this leaf be re-stated in the leaf's OWN range?
   *
   * Two independent questions, both of which must answer yes:
   *
   * * **Is the relation between the layer and the leaf one that a range change
   *   MEANS something across?** Only `kind=lod`.
   * * **Is the composed window actually STATED in the layer's reference range**
   *   (`LayerInfo.scalarDataRange`), so that reading it as a position inside
   *   that range is legitimate?
   *
   * ## Only across `kind=lod`
   *
   * A LOD *level* and a partition *part* are not the same kind of sibling.
   *
   * Levels are alternative representations of the WHOLE object, and gsplat LOD
   * merging SUMS amplitudes — a coarse level's amplitude is the same physical
   * signal at a different numeric SCALE. Re-expressing the window per level is
   * exactly the correction that makes every level render one physical value
   * identically, which is #1753's own repro.
   *
   * Parts are disjoint spatial subsets of ONE field at the SAME scale, and
   * `packages/luxar/src/luxar/io/_compiler/gsplat_assembly.py` derives
   * `amplitude_data_range = [min, p99.9]` per splat set — so two tiles differ
   * purely by CONTENT. Windowing each part on its own range is per-tile
   * auto-contrast: the same physical value renders as a different colour in
   * different tiles and the colormap goes non-monotone, with a visible
   * discontinuity at every BSP seam. On the repo's own
   * `tests/fixtures/test_partition_layer.luxar.zarr` (`layer=True,
   * kind=partition` — and `layer=True` is the default of `luxar gsplat
   * convert`), `part_0` is `[0.5000, 0.7455]` and `part_1` `[0.7542, 0.9998]`:
   * remapping would ramp black→white across BOTH, so the field would step
   * 0.7455 (white) → 0.7542 (black) at the seam. Saturated-but-monotone is the
   * correct failure. The producers say the same thing twice —
   * `core/group/adders/mesh.py::_shared_scalar_window` ("a level or a **part**
   * that stamps its own subset min/max renders the same value as a different
   * colour … which is exactly the discontinuity this helper exists to
   * prevent") and `gsplats/lift.py`, where beads share the finest node's
   * `scalar_data_range` rather than a per-segment one.
   *
   * So every GROUP node from the edited layer down to (excluding) the leaf must
   * be `kind === 'lod'`. Consequences, all deliberate: `adaptive` (a partition
   * of per-tile lod groups) declines outright, because the right reference for a
   * tile's ladder is that TILE's finest level rather than the layer's, and
   * building that is more than #1753 asks for; and a plain group layer over
   * several colormapped leaves (two channels, say) declines too — different
   * physical fields, not one field at two scales.
   *
   * An `overview` tree (an lod group whose coarse cap is a leaf and whose fine
   * branch is a nested partition) is a NO-OP in both branches, and it is worth
   * being precise about why rather than claiming half a win. The tiles decline
   * on the partition, as above. The cap is structurally eligible — but on a
   * measured `luxar gsplat lod --recipe overview` store the cap and all four
   * parts carry 432 splats each, and `deriveScalarRangeFromDescendants` breaks
   * that tie with a strict `count > bestCount` while visiting the cap FIRST, so
   * the cap IS the reference and `remapWindowToLeafRange`'s equality
   * short-circuit returns its window untouched. The fine parts therefore keep
   * rendering on the cap's window (measured: `part_2`'s own
   * `[0.00059, 0.19962]` on the cap's `[0.000116, 0.44551]`, so its brightest
   * splat lands at LUT 0.45 instead of 1.0). Fixing that needs a per-branch
   * reference, which is the same change `adaptive` would need.
   *
   * Once #1691 / PR #1752 lands (it harmonizes `amplitude_data_range` across a
   * gsplat structure so siblings SHARE a window) partition parts will carry
   * equal ranges and `remapWindowToLeafRange`'s equality short-circuit would
   * make this a no-op anyway. The `kind` gate is the safety net until then, and
   * for every store already written.
   *
   * ## …and only from a window in the reference basis
   *
   * `composeEffective` multiplies `intensity` and sums `offset` over the WHOLE
   * ancestry, substituting live panel state for every `layer=true` node, so
   * several reachable shapes hand back a window in a completely different
   * basis. Remapping one of those does not refine a correct window, it corrupts
   * it — hence a predicate rather than a best-effort. Every arm below is pinned
   * by a test in `tests/unit/ui/layers/layer-apply-per-leaf-window.test.ts`:
   *
   * 1. **The edited layer is not on this leaf's ancestry** (`layerDepth < 0`).
   *    Nothing can be said about the basis, so nothing is done.
   * 2. **The edited layer AUTHORED a gain.** `intensity`/`offset` are
   *    compositing attrs, so `add_gsplats_from_file(…, layer=True,
   *    intensity=0.5)` / `luxar gsplat convert --intensity 0.5` stamps them on
   *    the `kind=lod` wrapper itself. `walkSceneGraph` then seeds
   *    `displayMin/Max` from `computeDisplayRange(0.5, 0) = [0, 2]` — a window
   *    in the normalized-GAIN basis, with no relation to a `scalarDataRange` of,
   *    say, `[0, 0.02]`. That window was already wrong before this change (100x
   *    too wide); remapping it would multiply the error by `leafSpan / refSpan`
   *    on top. Declining leaves the pre-existing behaviour exactly as it was.
   * 3. **A node strictly below the layer is itself TRACKED AS A LAYER.** Not
   *    "contributes a gain" — a nested layer owns its own window and its own
   *    panel row, full stop, and its gain is not evidence either way. A
   *    colormapped child whose own range happens to be `[0, 1]` composes
   *    `computeUniforms(0, 1) = {1, -0}`, indistinguishable from "no window
   *    authored", so a gain test passed it and remapped a window that was
   *    already correct. `deriveScalarRangeFromDescendants` is the one derivation
   *    in `layer-state.ts` that does NOT stop at a nested layer, so the outer
   *    reference can be a sibling's range: over `ch0` (`[0, 1]`, 1e3 splats) and
   *    `ch1` (`[0, 5]`, 1e4 splats) it is `[0, 5]`, and dragging the wrapper's
   *    opacity composed ch0's own correct `[0, 1]` — which remapping turned into
   *    `[0, 0.2]`, 5x too narrow, with nothing re-applying ch0 afterwards.
   *    `[0, 1]` is not exotic: normalized scalars, probabilities, masks,
   *    fractions.
   * 4. **A non-layer node strictly below the layer contributes an AUTHORED
   *    gain.** This arm mirrors `resolveColormapWindow`'s first branch, which
   *    says the same thing at load time: a non-identity RAW LEAF gain means the
   *    composed gain IS the window and the data range is not consulted.
   *
   * Ancestors ABOVE the edited layer are deliberately NOT gated — but they are
   * not folded into the remap either. `leafScalarWindow` re-expresses the
   * LAYER'S OWN window and re-applies the ancestor gain afterwards, which is
   * what makes the panel match `resolveColormapWindow`'s ancestor-only branch
   * exactly (ancestor `intensity = 2`, reference `[0, 2]`, leaf `[0, 8]` → both
   * give `[0, 4]`) — see the note there for why remapping the COMPOSED window
   * instead only agrees when the two ranges happen to share a relative origin.
   */
  private composedWindowIsInReferenceBasis(
    leafPath: string,
    layerPath: string,
    precomputedAncestors?: readonly SceneNode[]
  ): boolean {
    let ancestors: readonly SceneNode[];
    if (precomputedAncestors) {
      ancestors = precomputedAncestors;
    } else {
      const sceneGraph = this.deps.getSceneGraph();
      if (!sceneGraph) return false;
      ancestors = collectAncestorNodes(sceneGraph, leafPath);
    }
    const layerDepth = ancestors.findIndex((n) => n.path === layerPath);
    if (layerDepth < 0) return false;
    const layerNode = ancestors[layerDepth];
    if (
      !isIdentityGain(
        layerNode.attrs.intensity as number | undefined,
        layerNode.attrs.offset as number | undefined
      )
    ) {
      return false;
    }
    // Every group between the edited layer and the leaf (the layer itself
    // included, the leaf excluded) must be a LOD group. When the edited layer IS
    // the leaf this loop is empty — and the remap is a no-op there anyway, since
    // the leaf range and the reference range are then the same range.
    for (let i = layerDepth; i < ancestors.length - 1; i++) {
      if (ancestors[i].attrs.kind !== 'lod') return false;
    }
    for (let i = layerDepth + 1; i < ancestors.length; i++) {
      const node = ancestors[i];
      // A nested layer owns its window outright — arm 3. Checked before the gain
      // so a nested layer whose window happens to compose to the identity gain
      // (any `[0, 1]` range) is not mistaken for "nothing authored here".
      if (this.deps.state.getLayer(node.path)) return false;
      if (
        !isIdentityGain(
          node.attrs.intensity as number | undefined,
          node.attrs.offset as number | undefined
        )
      ) {
        return false;
      }
    }
    return true;
  }

  /**
   * The scalar LUT window for ONE colormap-active leaf: the layer's composed
   * window, re-expressed in that leaf's own data range.
   *
   * A layer composes a single window, but a `kind=lod` layer fans out over
   * LEVELS whose scalars need not share a range — LOD merging sums amplitudes,
   * so a coarsened level carries its own `amplitude_data_range` for the same
   * physical signal, and the layer's reference range is merely whichever
   * descendant `deriveScalarRangeFromDescendants` picked. Pushing the composed
   * window verbatim rendered every level on the reference level's window,
   * discarding exactly the per-level differentiation the producer stamped, and
   * made a lazily-created level's window depend on load order (#1753).
   *
   * The remap runs only when the layer↔leaf relation is one a range change means
   * something across, AND the composed window really is stated in the reference
   * basis ({@link composedWindowIsInReferenceBasis}) — otherwise it would corrupt
   * a window that was already correct, or auto-contrast a spatial partition tile
   * by tile. The remap itself, and the range-shaped cases in which it must not
   * happen, live in the pure `remapWindowToLeafRange`; the common single-range
   * layer short-circuits there and gets the composed window back bit-exact.
   *
   * What gets remapped is the LAYER'S OWN window, not the composed one, and the
   * ancestor gain is re-applied to the result. Only the layer's own window is a
   * position inside `layer.scalarDataRange`; the composed window is that
   * position already transformed by whatever gain the ancestry above the layer
   * contributes, and reading it as a reference-basis position is a basis error
   * of exactly the kind {@link composedWindowIsInReferenceBasis} exists to
   * refuse. It cancels out when the two ranges share a relative origin
   * (`ref₀/refSpan === leaf₀/leafSpan` — notably when both start at 0) and not
   * otherwise: with `ref = [1, 3]`, `leaf = [0, 8]` and an ancestor
   * `intensity = 2`, remapping the composed window gives `[-2, 2]` where node
   * creation gives `[0, 4]`. Swapping the layer's contribution reproduces
   * `resolveColormapWindow`'s ancestor-only branch identically for every range
   * pair.
   *
   * Only meaningful for a colormap-active material: a direct-colour leaf has no
   * scalar window (its gain/offset are a colour GOG), which is the same condition
   * `applyColorAdjustments` routes on. That is also why `eff` here is never the
   * `identityLayerWindow` composition — that substitution is made only for
   * direct-colour leaves, which never reach this helper.
   */
  private leafScalarWindow(
    leaf: SceneNode,
    layer: LayerInfo,
    eff: EffectiveAttrs,
    ancestors?: readonly SceneNode[]
  ): { min: number; max: number } {
    const composed = computeDisplayRange(eff.intensity, eff.offset);
    if (!this.composedWindowIsInReferenceBasis(leaf.path, layer.path, ancestors)) return composed;
    const leafRange = (leaf.attrs.scalar_data_range || leaf.attrs.amplitude_data_range) as
      [number, number] | undefined;
    const own = { min: layer.displayMin, max: layer.displayMax };
    const remapped = remapWindowToLeafRange(own, layer.scalarDataRange, leafRange);
    // `remapWindowToLeafRange` hands back the very object it was given when it
    // declines, so identity is the cheapest "nothing to re-express" test.
    if (remapped === own) return composed;
    const from = computeUniforms(own.min, own.max);
    // Nothing above the layer contributes a gain (the overwhelmingly common
    // case): `eff` IS the layer's own window, so the remapped window is the
    // answer verbatim — and bit-exactly, without a multiply/divide round trip.
    if (eff.intensity === from.intensity && eff.offset === from.offset) return remapped;
    // Otherwise swap the layer's contribution from its own window to the leaf's
    // and leave the ancestor gain in `eff` exactly where it was.
    const to = computeUniforms(remapped.min, remapped.max);
    return computeDisplayRange(
      (eff.intensity * to.intensity) / from.intensity,
      eff.offset - from.offset + to.offset
    );
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
    // Non-null whenever `leaves` is non-empty (`getAffectedDataLeaves` returns
    // nothing without a graph); read once so the ancestry is walked ONCE per
    // leaf below rather than once per consumer of it.
    const sceneGraph = this.deps.getSceneGraph();
    if (!sceneGraph) return;

    for (const leaf of leaves) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat) continue;
      // ONE ancestry walk per leaf, shared by the composition and the
      // reference-basis gate. `collectAncestorNodes` resolves each step with a
      // linear `children.find` that allocates a string per comparison, so a
      // fan-out over a P-part wrapper is ~P²/2 comparisons — measured at 7.0 ms
      // for 512 parts and 129 ms for 2000, per slider tick. Walking it twice
      // doubled that.
      const ancestors = collectAncestorNodes(sceneGraph, leaf.path);
      // Per-leaf window routing. A layer whose window is a SCALAR window
      // (colormap in play) can still contain leaves rendering direct colour:
      // the C1 guard suppresses the LUT on geometry with no scalars bound,
      // and a MIXED group keeps the scalar window because some other leaf
      // accepted it. Pushing that window into a direct-colour leaf's colour
      // GOG is exactly the contrast stretch this panel no longer does — such
      // a leaf gets the identity window instead.
      const identityLayerWindow = layer.scalarWindow && !isColormapActive(mat);
      const eff = this.composeEffective(leaf.path, layer.path, identityLayerWindow, ancestors);
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
      // A mesh's PICK material reads the same coverage the visual one does — node
      // opacity times per-vertex alpha (§6.5) — so it has to move with the slider.
      // Without this, dragging opacity below the `opaque` cutoff would dissolve the
      // surface on screen while leaving every triangle pickable, and hover tooltips
      // would keep naming vertices of an invisible mesh. A no-op for the other three
      // types, whose pick materials derive coverage from their own element data.
      //
      // Written OUTSIDE the LOD-fade branch above on purpose: a mesh CAN be a
      // substitutive LOD level now (§9), so the sync has to run whichever branch the
      // node took — an unconditional sync is what keeps a faded mesh level pickable
      // at the coverage it actually renders with.
      //
      // When the sync actually touched a mesh pick material, invalidate the cached
      // pick buffer: a stationary-camera layers-panel edit invalidates nothing else,
      // so hover would otherwise keep naming vertices of the pre-edit coverage.
      if (syncMeshPickAppearance(obj as THREE.Mesh, { opacity: eff.opacity })) {
        this.deps.invalidatePickBuffer?.();
      }
      // All three geometry-material families implement it (gsplats
      // phase 1, points phase 3, lines phase 4); optional-chained for
      // non-Luxar materials.
      mat.updateAbsorption?.(eff.absorption);
      // A colormap-active leaf windows a SCALAR, and (when the composed window
      // really is stated on the layer's reference range — see
      // `composedWindowIsInReferenceBasis`) it is re-expressed in this leaf's
      // own range, so a multi-level / multi-part layer stops rendering every
      // leaf on the reference leaf's window (#1753). A direct-colour leaf has
      // no scalar window at all, so it is not computed there.
      const scalarWindow = isColormapActive(mat)
        ? this.leafScalarWindow(leaf, layer, eff, ancestors)
        : undefined;
      applyColorAdjustments(mat, eff.gamma, eff.intensity, eff.offset, scalarWindow);
      const prevBlendingMode = mat.userData?.blendingMode as BlendingMode | undefined;
      // An unset ancestry composes to `undefined`; apply this leaf's per-type
      // default (mesh → opaque, emissive → additive) — the same mode the
      // material factory would have baked in.
      const blendingMode = eff.blending_mode ?? defaultBlendingMode(leaf.type);
      this.applyBlendingStateToMaterial(mat, blendingMode);
      // Depth sorting: a sortable layer switching blending mode may need
      // to start (TO an effective sorted mode: clear the committed-data
      // stamp + reprocess so the next commit registers with the SortWorker)
      // or stop (AWAY: release) depth sorting. Only types that register
      // per-element centers with the coordinator qualify (gsplats/points
      // centers, lines segment midpoints, mesh face centroids) — see
      // `depthSortable` in `types/geometry-capabilities`.
      //
      // BOTH arguments must be the RESOLVED mode, which is why the new one is
      // re-read off the material AFTER `applyBlendingStateToMaterial` rather
      // than passing the requested `blendingMode`. `prevBlendingMode` above was
      // already resolved (it came off `userData`), so passing the request here
      // made the pair asymmetric — invisible for the three emissive types,
      // whose request IS their resolved mode, and wrong for mesh, which maps
      // the unsupported `volumetric` onto `opaque`:
      //   normal → volumetric  looked like sorted → sorted, so the node kept
      //     its worker registration and retained `triangleSource` after the
      //     material had gone opaque and would never sort again;
      //   opaque → volumetric  looked like a switch INTO a sorted mode and
      //     triggered a full clear + O(N) reprocess for nothing.
      // The coordinator's own `liveBlendingMode` reads the resolved mode, so
      // this is also what makes the hook and the per-frame scheduler agree.
      // A physical mesh deliberately leaves the stamp unset while translucent,
      // but is never triangle-sorted; resolve it to opaque for this hook rather
      // than falling back to an ignored inherited mode. The `?? blendingMode`
      // covers the generic fallback arm of
      // `applyBlendingStateToMaterial` (a material without `applyBlendingMode`,
      // kept for external/future materials): that arm never stamps
      // `userData.blendingMode`, so reading the material alone would leave the
      // value unchanged and silently make this hook a no-op for such a node.
      if (isDepthSortable(obj.userData?.nodeType)) {
        const resolvedMode = isPhysicalMeshMaterial(mat)
          ? 'opaque'
          : ((mat.userData?.blendingMode as BlendingMode | undefined) ?? blendingMode);
        noteDepthSortBlendingModeSwitch(obj as THREE.Mesh, resolvedMode, prevBlendingMode);
      }
      scheduleBlendModeProgramWarmupForObject(obj);
    }
    this.deps.requestRender();
  }

  applyVisibility(path: string, visible: boolean): void {
    const obj = this.getMesh(path);
    if (obj) {
      const wasLayerVisible = obj.userData.layerVisible !== false;
      obj.userData.layerVisible = visible;
      obj.visible = visible;
      this.deps.requestRender();
      if (visible && !wasLayerVisible) this.deps.requestReprocess([path]);
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

  /**
   * Push the composed draw order onto the meshes the depth-sort coordinator
   * reads it from.
   *
   * Does not go through `applyComposed` — an order is a cross-node SORT KEY,
   * not a material uniform, so there is no `mat.updateX` to call and nothing in
   * the shader to refresh. But it DOES compose: the value written to each leaf
   * is `composeEffective`'s, not this layer's raw one.
   *
   * That distinction is the bug this replaced. Assigning `layer.layerOrder`
   * directly to every affected leaf clobbered the order of a nested leaf that
   * is ITSELF a layer with its own authored order — `getAffectedDataLeaves`
   * returns every data descendant, including nested layers, and
   * nearest-setter-wins says the nested one should win. Composing per leaf
   * restores that, and lets an order on a non-layer intermediate group
   * participate too.
   *
   * This is render-only session state. It must not be written into
   * `userData.attrs`, which is the loaded SceneNode attrs object for lines and
   * gsplats and would make a panel edit look authored on the next composition.
   */
  applyLayerOrder(layer: LayerInfo): void {
    const sceneGraph = this.deps.getSceneGraph();
    for (const leaf of this.getAffectedDataLeaves(layer.path)) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const ancestors = sceneGraph ? collectAncestorNodes(sceneGraph, leaf.path) : undefined;
      const eff = this.composeEffective(leaf.path, layer.path, false, ancestors);
      obj.userData.layerOrder = eff?.layer_order;
    }
    this.deps.requestRender();
  }

  applyLabelStyle(layer: LayerInfo): void {
    let applied = false;
    let pickDirty = false;
    for (const leaf of this.getAffectedDataLeaves(layer.path)) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat?.updateLabelStyle) continue;
      const vocabulary = leaf.attrs.label_vocabulary as Record<string, string> | undefined;
      const labelFilterIndex = layer.labelFilterId
        ? Object.keys(vocabulary ?? {}).indexOf(layer.labelFilterId) + 1
        : 0;
      mat.updateLabelStyle(layer.colorByLabel, labelFilterIndex);
      applied = true;
      const pickMaterial = (obj.userData.pickNode as THREE.Mesh | undefined)?.material;
      if (pickMaterial && !Array.isArray(pickMaterial) && 'updateLabelFilter' in pickMaterial) {
        (
          pickMaterial as THREE.Material & { updateLabelFilter(index: number): void }
        ).updateLabelFilter(labelFilterIndex);
        pickDirty = true;
      }
    }
    if (pickDirty) this.deps.invalidatePickBuffer?.();
    if (applied) this.deps.requestRender();
  }

  /**
   * Push the mesh shading values (§6.2) to the layer's mesh leaves.
   *
   * Deliberately NOT routed through {@link applyComposed}, which is what every other
   * control here uses, because these values **do not compose along the ancestry**:
   * `opacity`/`gamma`/`intensity` multiply and `offset` sums, so an ancestor's value
   * has to fold into a descendant's, whereas a shade floor is a per-surface appearance
   * choice with no composition rule — multiplying two ambients would mean nothing.
   *
   * It still has to FAN OUT like `applyComposed` does, though. A mesh layer is no
   * longer always a leaf: `add_mesh(partition=…)` writes a kind=partition wrapper and
   * the panel presents that wrapper as one `mesh` layer, so `layer.path` resolves to a
   * `THREE.Group` with no material of its own. Writing only there left all mesh appearance
   * sliders visible and completely inert on a partitioned surface. A non-mesh leaf
   * needs no extra gate — it simply has no `updateAmbient`, so the optional chaining
   * below is the type check.
   *
   * `alphaCutoff` also rides to the PICK material, because the pick pass applies the
   * identical cutout (§6.5): a threshold that moved on screen but not in the pick
   * buffer would make a freshly-dissolved region still hoverable.
   */
  applyMeshAppearance(layer: LayerInfo): void {
    let applied = false;
    let pickDirty = false;
    for (const leaf of this.getAffectedDataLeaves(layer.path)) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat) continue;
      applied = true;
      mat.updateAmbient?.(layer.ambient);
      mat.updateShadeExponent?.(layer.shadeExponent);
      mat.updateSpecular?.(layer.specular);
      mat.updateShininess?.(layer.shininess);
      mat.updateAlphaCutoff?.(layer.alphaCutoff);
      if (syncMeshPickAppearance(obj as THREE.Mesh, { alphaCutoff: layer.alphaCutoff })) {
        pickDirty = true;
      }
      scheduleBlendModeProgramWarmupForObject(obj);
    }
    if (pickDirty) this.deps.invalidatePickBuffer?.();
    if (applied) this.deps.requestRender();
  }

  /**
   * Push a physical layer's live knobs (`LayerInfo.physicalKnobs`, in material space)
   * onto every `material="physical"` leaf beneath it. The optional-chained
   * `updatePhysicalKnob` is the type gate, as for the house knobs above: a house or
   * emissive leaf simply lacks it. A no-op for a layer with no knob record.
   */
  applyPhysicalKnobs(layer: LayerInfo): void {
    const knobs = layer.physicalKnobs;
    if (!knobs) return;
    let applied = false;
    for (const leaf of this.getAffectedDataLeaves(layer.path)) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat?.updatePhysicalKnob) continue;
      for (const key of PHYSICAL_MESH_KNOB_KEYS) {
        mat.updatePhysicalKnob(key, knobs[key]);
      }
      // The `refract_data` switch rides the same record (spec §3.4 Phase 3).
      mat.updateRefractData?.(knobs.refract_data === true);
      applied = true;
    }
    if (applied) {
      this.deps.invalidatePickBuffer?.();
      this.deps.requestRender();
    }
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
   *
   * A leaf whose mesh/material is not in the scene yet (partition parts and
   * LOD levels stream in) is NOT a guard suppression — when no leaf material
   * was reachable at all, the request is taken at face value so the caller
   * keeps the user's pick instead of reverting it mid-load.
   */
  applyColormap(layer: LayerInfo): boolean {
    const leaves = this.getAffectedDataLeaves(layer.path);
    if (leaves.length === 0) return false;
    const sceneGraph = this.deps.getSceneGraph();
    if (!sceneGraph) return false;

    let colormapInEffect = false;
    let anyMaterialReached = false;
    for (const leaf of leaves) {
      const obj = this.getMesh(leaf.path);
      if (!obj) continue;
      const mat = this.getLeafMaterial(obj);
      if (!mat || !mat.updateColormapTexture) continue;
      anyMaterialReached = true;
      const tex = layer.colormap
        ? getColormapTexture(
            layer.colormap,
            getEffectiveAttrs(sceneGraph, leaf.path).customLutBytes
          )
        : null;
      if (layer.colormap && tex) {
        // C1 fail-closed guard: enabling USE_COLORMAP requires real
        // scalar data behind the geometry (the `userData.hasScalars`
        // stamp for points/lines texel storage; always true for gsplats,
        // whose amplitude is the scalar).
        //
        // `leaf.type` is a raw string off the node, so narrow it rather than
        // asserting: `supportsScalarColormap` is exhaustive over the geometry
        // vocabulary and must not be handed a value outside it.
        const geometry = (obj as THREE.Points | THREE.Mesh).geometry as THREE.BufferGeometry;
        const nodeType: GeometryTypeName | undefined = isGeometryType(leaf.type)
          ? leaf.type
          : undefined;
        if (!nodeType || !supportsScalarColormap(nodeType, geometry)) {
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
        //
        // Routed through the SAME `leafScalarWindow` helper `applyComposed`
        // uses, per-leaf remap included. The `applyComposed(layer)` at the end
        // of this method immediately supersedes what is written here, so the
        // two agreeing is about not leaving a trap for the next reader (#1753).
        if (mat.updateScalarRange) {
          // One walk, shared by the composition and the gate — see the same
          // note in `applyComposed`.
          const ancestors = collectAncestorNodes(sceneGraph, leaf.path);
          const eff = this.composeEffective(leaf.path, layer.path, false, ancestors);
          if (eff) {
            const { min, max } = this.leafScalarWindow(leaf, layer, eff, ancestors);
            mat.updateScalarRange(min, max);
          } else if (layer.scalarDataRange) {
            mat.updateScalarRange(layer.scalarDataRange[0], layer.scalarDataRange[1]);
          }
          // The composed window drives the LUT lookup; reset the post-LUT
          // color GOG to identity so a previously-stamped gain never
          // double-applies once the colormap takes over (#936). Only on the
          // colormap-active path — the direct-color branch below leaves the
          // GOG to `applyComposed`.
          mat.updateIntensity(1);
          mat.updateOffset(0);
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
    // "No colormap in effect" is only meaningful when at least one leaf
    // material was actually evaluated; with none reachable (still streaming),
    // report the requested state so the caller doesn't fight the user.
    return colormapInEffect || (!anyMaterialReached && !!layer.colormap);
  }
}
