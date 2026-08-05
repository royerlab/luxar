/**
 * Layer state management for the Layers panel.
 *
 * Pure data model (no DOM) that tracks per-layer state and provides
 * the min/max-to-intensity/offset mapping for the shader uniforms.
 */

import type { SceneNode } from '../../data/data-loader-types';
import { getEffectiveAttrs } from '../../data/attrs-composer';
import type { BlendingMode } from '../../types/blending';
import type { GeometryTypeName, NodeKind } from '../../types/format-contract';
import { defaultBlendingMode, isGeometryType } from '../../types/geometry-capabilities';
import { log, Modules } from '../../utils/log';
// Pure display-window ↔ shader-uniform math now lives in the rendering layer
// (`rendering/display-range`) so `rendering/` modules can import it without
// violating the module layer order. Imported for this module's own internal
// use and re-exported below so every existing `./layer-state` consumer keeps
// working unchanged.
import { computeUniforms, computeDisplayRange } from '../../rendering/display-range';
import { MESH_DEFAULTS, resolveMeshBlendingMode } from '../../rendering/materials/mesh/appearance';

/**
 * Geometry type of a layer.
 *
 * - ``group`` — composite container; controls fan out to descendants.
 * - every {@link GeometryTypeName} — leaf data layers. Deliberately the whole
 *   geometry vocabulary: any leaf type the format can carry is a layer the panel
 *   must be able to list.
 *
 * Specialized groups (``kind === 'lod'`` / ``'partition'``) appear in the layers
 * panel as their underlying ``display_type`` — never as ``'group'``. That
 * ``display_type`` is drawn from the LOD/partition-capable *subset* of the
 * vocabulary, not from `LayerType` itself (see `types/geometry-capabilities`).
 * The specialized-group nature is surfaced via the ``kind`` field on
 * ``LayerInfo``, which drives the per-layer badge / LOD dropdown.
 */
export type LayerType = GeometryTypeName | 'group';

/**
 * Coerce a node's raw `layer` attr into "exposed in the Layers panel".
 *
 * The Python writer (`validate_layer`) always normalises to a JSON boolean,
 * but hand-edited or third-party zarr may carry a number (`1`/`0`). Accept
 * strict `true` and truthy finite numbers so such values don't silently drop
 * the node from the panel; everything else (incl. `undefined`, strings) is
 * not-a-layer. Pure — callers log a warning for malformed (non-boolean) values.
 */
export function isLayerEnabled(value: unknown): boolean {
  if (value === true) return true;
  if (typeof value === 'number') return Number.isFinite(value) && value !== 0;
  return false;
}

/**
 * Derive a SCALAR display data-range for a composite group (kind=lod /
 * kind=partition) that carries no range of its own, by walking to the FINEST
 * descendant leaf (largest `n_splats`) that declares one. The finest level is
 * the full-detail data, so its range is the representative signal window; using
 * it avoids the near-black render a [0, 1] fallback produces for a colormapped
 * gsplat layer. Returns undefined when no descendant declares a range.
 *
 * Deliberately does NOT consider `color_data_range`: that attr describes the
 * spread of authored RGB, not a scalar signal, and windowing on it
 * contrast-stretches the author's colours (see `initialDisplayRange`).
 */
function deriveScalarRangeFromDescendants(node: SceneNode): [number, number] | undefined {
  let best: [number, number] | undefined;
  let bestCount = -1;
  const visit = (n: SceneNode): void => {
    const r = (n.attrs.scalar_data_range || n.attrs.amplitude_data_range) as
      [number, number] | undefined;
    const count = (n.attrs.n_splats as number | undefined) ?? 0;
    if (r && count > bestCount) {
      best = r;
      bestCount = count;
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return best;
}

/**
 * `color_data_range` of the node, else the UNION of every descendant's.
 * A partition's parts each declare their own spread, and the slider bounds
 * must cover the whole layer — taking the first part's range alone would
 * leave a later HDR part's colours unreachable.
 */
function deriveColorRangeFromDescendants(node: SceneNode): [number, number] | undefined {
  const own = node.attrs.color_data_range as [number, number] | undefined;
  if (own) return own;
  let min = Infinity;
  let max = -Infinity;
  const visit = (n: SceneNode): void => {
    const r = n.attrs.color_data_range as [number, number] | undefined;
    if (r) {
      min = Math.min(min, r[0]);
      max = Math.max(max, r[1]);
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return min <= max ? [min, max] : undefined;
}

/**
 * True when `node` — or a descendant it OWNS — renders through a colormap LUT.
 *
 * The walk stops at a nested `layer=true` descendant: that node is its own row
 * in the panel with its own colormap control, so its palette can change at any
 * time. Deriving THIS layer's mode from it would freeze a snapshot that goes
 * stale the moment the inner control is used, leaving the outer layer claiming
 * a scalar window over a leaf that has since gone back to direct colour (its
 * range control then routes to the identity and does nothing). Every writer
 * routes `layer` through `COMPOSITING_ATTRS` onto the wrapper only, so a
 * kind=partition / kind=lod layer never has layer descendants — the boundary
 * only bites on explicitly nested authored layers.
 */
function usesColormap(node: SceneNode): boolean {
  if (node.attrs.colormap) return true;
  return (node.children ?? []).some((c) => !isLayerEnabled(c.attrs.layer) && usesColormap(c));
}

/**
 * The palette a wrapper layer EFFECTIVELY renders through when it carries no
 * `colormap` attr of its own: the single distinct palette authored on its
 * descendants, or undefined when there is none or they disagree (a mixed
 * subtree has no single palette the dropdown could show). `colormap` is
 * deliberately NOT a compositing attr on the Python side — the writer copies
 * it onto every leaf — so a kind=partition / kind=lod layer from
 * `add_gsplats_from_file(..., colormap=...)` always has this shape. Surfacing
 * the descendant palette keeps the dropdown and the legend truthful and makes
 * "(direct colors)" an actual off-switch there (a select only emits `change`
 * when its value moves).
 *
 * Stops at nested `layer=true` descendants for the same reason
 * {@link usesColormap} does — they own their palette.
 */
function deriveColormapFromDescendants(node: SceneNode): string | undefined {
  let found: string | undefined;
  let mixed = false;
  const visit = (n: SceneNode): void => {
    if (mixed || isLayerEnabled(n.attrs.layer)) return;
    const cm = n.attrs.colormap as string | undefined;
    if (cm) {
      if (found === undefined) found = cm;
      else if (found !== cm) mixed = true;
    }
    n.children?.forEach(visit);
  };
  node.children?.forEach(visit);
  return mixed ? undefined : found;
}

/**
 * The display window a layer starts at — i.e. what `[displayMin, displayMax]`
 * the panel pushes into the material before the user touches anything.
 *
 * The window maps the rendered VALUE to [0, 1], so which range is right depends
 * on what that value is:
 *
 * * **Colormapped layers** window a scalar (gsplat amplitude / points-lines
 *   scalar), whose useful range is the data range — a linear [0, 1] window on a
 *   heavily right-skewed amplitude renders near-black (#522).
 * * **Direct-colour layers** window authored RGB, whose range IS [0, 1]. Any
 *   other window is a contrast stretch of colours the author already chose:
 *   a uniform grey `(0.72, 0.74, 0.78)` has `color_data_range` [0.72, 0.78],
 *   which windows to gain 16.7 / offset −12 and renders as saturated BLUE.
 *   So direct colour starts at the identity window.
 */
function initialDisplayRange(node: SceneNode): [number, number] {
  if (!usesColormap(node)) return [0, 1];
  const scalarRange = (node.attrs.scalar_data_range || node.attrs.amplitude_data_range) as
    [number, number] | undefined;
  return scalarRange ?? deriveScalarRangeFromDescendants(node) ?? [0, 1];
}

/**
 * Specialized-group discriminant on a layer. Single-sourced from the
 * cross-language format contract (format-contract/contract.yaml).
 */
export type LayerKind = NodeKind;

/** Information about a single layer in the Layers panel */
export interface LayerInfo {
  /** Zarr path (e.g. "group/channel_gfp") — used as unique key */
  path: string;
  /** Display name (last segment of path) */
  name: string;
  /**
   * Geometry type the user sees this layer as. For specialized groups
   * (kind=lod / kind=partition), this is the resolved ``display_type`` attr
   * from disk — not ``'group'``.
   */
  type: LayerType;
  /**
   * Specialized-group kind, if the underlying scene-graph node is a
   * kind=lod or kind=partition ``Group``. Drives the layer-header badge and
   * the inline LOD-level dropdown.
   */
  kind?: LayerKind;
  /** Whether the layer is visible in the scene */
  visible: boolean;
  /** Opacity (0–1) */
  opacity: number;
  /** Absorption coefficient κ (≥ 0; only meaningful in volumetric mode) */
  absorption: number;
  /**
   * Mesh shade floor (0–1) — the §6.2 headlight's `ambient`. Only meaningful on a
   * mesh layer, where it is what keeps a silhouette readable rather than black; `1.0`
   * collapses the shade term and reproduces the other three types' emissive look.
   */
  ambient: number;
  /**
   * Mesh headlight falloff exponent (> 0) — the §6.2 `shade_exponent`. `1.0` is the
   * plain linear wrap. Mesh-only, like the two around it.
   */
  shadeExponent: number;
  /**
   * Mesh `opaque`-mode cutout threshold (0–1) — the §6.2 `alpha_cutoff`.
   *
   * Only meaningful in `opaque`, which is a NARROWER condition than the other two
   * (they apply in every mesh mode), so the panel gates its slider on the mode as well
   * as the type — the same shape as absorption's volumetric gate.
   */
  alphaCutoff: number;
  /** Current display-range minimum (maps to intensity+offset in shader) */
  displayMin: number;
  /** Current display-range maximum */
  displayMax: number;
  /**
   * Slider lower bound. Wide enough to reach every window the layer can
   * usefully take: the mode's default (scalar range / identity), the authored
   * `intensity`/`offset` window, and — for a direct-colour layer — its
   * `color_data_range`, so "stretch these colours" stays one drag away.
   */
  dataMin: number;
  /** Slider upper bound (see {@link LayerInfo.dataMin}). */
  dataMax: number;
  /** Gamma correction value */
  gamma: number;
  /** Blending mode */
  blendingMode: BlendingMode;
  /**
   * Whether a blend mode is set EXPLICITLY for this layer — authored somewhere
   * in its composed ancestry, or chosen by the user via the panel. When false,
   * `blendingMode` is merely the per-type default shown in the dropdown, and the
   * live-attrs path must NOT push it onto descendants (that would override each
   * leaf's own per-type default — e.g. force a mesh under a plain group layer to
   * the group's `additive` default instead of its own `opaque`; see #1272).
   */
  blendingModeExplicit: boolean;
  /** Whether this layer is selected in the list */
  selected: boolean;
  /** Active colormap name (undefined = direct RGB colors) */
  colormap?: string;
  /** Whether this node supports colormap (has scalars or amplitudes) */
  supportsColormap: boolean;
  /** Scalar data range for colormap normalization */
  scalarDataRange?: [number, number];
  /**
   * Authored RGB spread (`color_data_range`). Never the starting window — see
   * {@link LayerInfo.dataMin} — but kept so a colormap toggle can restore the
   * direct-colour slider bounds, symmetric with {@link LayerInfo.scalarDataRange}.
   */
  colorDataRange?: [number, number];
  /**
   * Whether the current display window is a SCALAR window (a colormap is in
   * play for the layer — on the node or a descendant) rather than an
   * authored-RGB one. `LayerApplyEngine.applyComposed` routes on this per
   * leaf: a leaf the C1 guard keeps on direct colour must NOT receive the
   * scalar window as a colour gain (a mixed group layer contains both kinds).
   * Follows the colormap toggle via {@link LayerStateManager.setColormapWindow}.
   */
  scalarWindow: boolean;
  /**
   * For ``kind === 'lod'`` layers: number of child levels. Drives the
   * "Active level" dropdown's option count and the "N LODs" badge.
   * Absent for non-LOD layers.
   */
  lodGroupChildCount?: number;
  /**
   * For ``kind === 'partition'`` layers: number of BSP child parts. Drives
   * the "N parts" badge on the layer header. Absent for non-Partition
   * layers.
   */
  partCount?: number;
  /**
   * For ``kind === 'partition'`` layers that wrap kind=lod descendants:
   * absolute paths of every nested lod_group in the subtree. The
   * "Active level" dropdown on the wrapper broadcasts to every path
   * (clamped per-group by ``LODGroupRegistry.setSelectorMode`` on
   * ragged ladders). Absent for non-Partition layers and for Partition layers
   * with no nested LODs.
   *
   * Also used to compute the ``[N parts × M LODs]`` combined badge:
   * when this list is non-empty, M is the maximum child count across
   * nested groups (matches the dropdown's "highest level offered"
   * affordance).
   */
  nestedLodGroupPaths?: string[];
  /**
   * Maximum LOD child count across the layer's nested lod_groups.
   * Used by the dropdown to size its option list and by the combined
   * badge. Absent unless ``nestedLodGroupPaths`` is non-empty.
   */
  nestedLodMaxChildCount?: number;
}

// Re-export the relocated display-range helpers (imported at the top of this
// file) so every existing `./layer-state` consumer keeps working unchanged.
export { computeUniforms, computeDisplayRange };
export type { DisplayUniforms } from '../../rendering/display-range';

/** Selection mode for layer clicks */
export type SelectionMode = 'single' | 'add' | 'range';

/** Listener callback type */
export type LayerChangeListener = () => void;

/**
 * Manages the state of all layers in the Layers panel.
 *
 * Walks the SceneNode tree to collect nodes with `layer: true`,
 * tracks selection, and provides mutators that notify listeners.
 */
/**
 * The blending mode a layer of `type` will ACTUALLY render with.
 *
 * For mesh this is not the composed value: `volumetric` has no meaning for a
 * zero-thickness surface, so the mesh material maps it to `opaque` and stamps the
 * RESOLVED mode into `userData.blendingMode` (spec §6.3). Storing the unresolved value
 * in `LayerInfo` made the panel disagree with the render in two visible ways at once —
 * it showed the Absorption slider (which no mesh shader reads) and HID the Alpha-cutoff
 * slider precisely when the cutout was active. The pick pass reads the material's
 * resolved mode, so it was correct and the UI was not.
 *
 * Resolved at the point of STORAGE rather than at each display gate, so every consumer
 * of `LayerInfo.blendingMode` — the Blend dropdown's own displayed value included — sees
 * the mode that renders. A user who explicitly picks `volumetric` on a mesh sees the
 * dropdown snap back to `opaque`, which is honest: it is what the surface is doing, and
 * it matches the one-time warning the loader already emits.
 */
export function resolveLayerBlendingMode(
  type: LayerType | undefined,
  mode: BlendingMode
): BlendingMode {
  return type === 'mesh' ? resolveMeshBlendingMode(mode) : mode;
}

export class LayerStateManager {
  /** Ordered list of layer paths (insertion order from scene graph walk) */
  private layerOrder: string[] = [];
  /** Map from path to LayerInfo */
  private layers = new Map<string, LayerInfo>();
  /** Path of the last clicked layer (anchor for Shift+click range) */
  private lastClickedPath: string | null = null;
  /** Change listeners */
  private listeners = new Set<LayerChangeListener>();

  /**
   * Initialize layers from the scene graph.
   * Walks children recursively and collects nodes with `layer: true`.
   */
  initFromSceneGraph(root: SceneNode): void {
    this.layers.clear();
    this.layerOrder = [];
    this.lastClickedPath = null;

    this.walkSceneGraph(root, root);
  }

  private walkSceneGraph(node: SceneNode, root: SceneNode): void {
    // Skip the root scene node; collect anything else with layer=true.
    // Groups exposed as layers act as composites — their controls fan out
    // to every data descendant when applied in the scene.
    const layerAttr = node.attrs.layer;
    if (layerAttr !== undefined && typeof layerAttr !== 'boolean') {
      // Producers should write a JSON boolean; coerce truthy values but warn
      // so a malformed (e.g. string) attr doesn't silently drop the node.
      log.warning(
        Modules.UI,
        `Node "${node.path}" has a non-boolean 'layer' attr (${typeof layerAttr}); coercing.`
      );
    }
    if (node.type !== 'scene' && isLayerEnabled(layerAttr)) {
      // Vocabulary question, not a capability one: any geometry leaf the
      // format can carry must be listable in the panel (see `LayerType`).
      const isLayerType = isGeometryType(node.type) || node.type === 'group';
      if (isLayerType) {
        const name = node.path.split('/').pop() || node.path;

        // The window the layer starts at. Colormapped layers window a scalar
        // (from this node or, for a composite kind=lod / kind=partition group,
        // its finest descendant leaf); direct-colour layers window authored RGB
        // and so start at the identity [0, 1] — see `initialDisplayRange`.
        const ampRange = node.attrs.amplitude_data_range as [number, number] | undefined;
        const scalarRange = node.attrs.scalar_data_range as [number, number] | undefined;
        const dataRange = initialDisplayRange(node);

        // Colormap support — groups inherit no colormap, but they do apply
        // a chosen colormap to every data descendant that can accept one.
        // Gsplats only support a colormap when they actually have scalar
        // data (`has_scalars`) or an authored `colormap`; a bare gsplats
        // node with no scalars must NOT advertise colormap support, or
        // the UI offers a no-op colormap dropdown.
        // kind=lod / kind=partition groups are composite containers; the
        // colormap applies to descendants via composition just like a
        // plain group.
        // The wrapper's own attr, else the (uniform) palette its descendants
        // carry — `scalarWindow` below is descendant-aware, and a `colormap`
        // that isn't would misreport an actively colormapped partition/lod
        // layer as "(direct colors)" in the dropdown and hide it from the
        // legend.
        const colormap =
          (node.attrs.colormap as string | undefined) || deriveColormapFromDescendants(node);
        const supportsColormap = node.type === 'group' || !!node.attrs.has_scalars || !!colormap;
        const colormapScalarRange =
          scalarRange || ampRange || deriveScalarRangeFromDescendants(node);

        // Initialize display range from existing intensity/offset if present,
        // otherwise default to full data range
        const intensity = (node.attrs.intensity as number) ?? 1.0;
        const offset = (node.attrs.offset as number) ?? 0.0;
        let displayMin: number;
        let displayMax: number;

        if (intensity === 1.0 && offset === 0.0) {
          displayMin = dataRange[0];
          displayMax = dataRange[1];
        } else {
          const recovered = computeDisplayRange(intensity, offset);
          displayMin = recovered.min;
          displayMax = recovered.max;
        }

        // Slider bounds must encompass the current display range. The recovered
        // [displayMin, displayMax] can extend beyond color_data_range when the
        // node has authored intensity/offset (e.g., intensity=0.2 implies a
        // display range of [0, 5] regardless of where the actual color values
        // lie). If the slider's <input type="range"> min/max stayed at the
        // narrower data range, the browser would silently clamp the thumb
        // values on first render, and the first slider interaction would write
        // the clamped values back — snapping intensity from the authored value
        // to the slider-implied one and causing a sudden brightness jump.
        // A direct-colour layer no longer STARTS at its colour range, but the
        // slider must still reach it so "stretch these colours" stays a
        // one-drag operation. Kept on the layer either way (even when a
        // colormap currently wins) so `setColormapWindow` can restore these
        // bounds when the colormap is switched off.
        const colorDataRange = deriveColorRangeFromDescendants(node);
        const colorRange = usesColormap(node) ? undefined : colorDataRange;
        const dataMin = Math.min(dataRange[0], displayMin, colorRange?.[0] ?? Infinity);
        const dataMax = Math.max(dataRange[1], displayMax, colorRange?.[1] ?? -Infinity);

        // Honor the authoring-time `visible` attr (default true). Allows
        // Python authors to start a layer hidden via add_points(..., visible=False).
        const initialVisible = node.attrs.visible !== false;

        // Specialized-group discriminant: read ``kind`` from the on-disk
        // attrs of a ``type === 'group'`` node. ``undefined`` for plain
        // groups and leaf nodes.
        const rawKind = node.attrs.kind as LayerKind | undefined;
        const kind: LayerKind | undefined =
          node.type === 'group' && (rawKind === 'lod' || rawKind === 'partition')
            ? rawKind
            : undefined;

        // The layer's user-facing geometry type. For a specialized
        // group, use its ``display_type`` (resolved at write time on the
        // Python side) so the layer reads as e.g. "gsplats" not "group".
        // Fall back to the node's raw type for plain groups / leaves.
        const displayType = node.attrs.display_type as LayerType | undefined;
        const layerType: LayerType =
          kind !== undefined && displayType ? displayType : (node.type as LayerType);

        // Kind-specific badge counts. lod → "N LODs" + dropdown; partition →
        // "N parts" status chip. Other layers leave both undefined.
        const lodGroupChildCount = kind === 'lod' ? (node.children?.length ?? 0) : undefined;
        const partCount = kind === 'partition' ? (node.children?.length ?? 0) : undefined;

        // Partition-of-LOD discovery. A kind=partition layer that wraps
        // kind=lod descendants gets a broadcast dropdown over every
        // nested lod_group. The walk stops at the first lod_group it
        // hits per branch — an lod_group's own children are leaves of
        // the LOD ladder, not further LOD wrappers.
        let nestedLodGroupPaths: string[] | undefined;
        let nestedLodMaxChildCount: number | undefined;
        if (kind === 'partition') {
          const collected: string[] = [];
          let maxCount = 0;
          const visit = (n: SceneNode): void => {
            const childKind = n.attrs.kind as LayerKind | undefined;
            if (
              n.type === 'group' &&
              childKind === 'lod' &&
              n !== node // skip self
            ) {
              collected.push(n.path);
              maxCount = Math.max(maxCount, n.children?.length ?? 0);
              return; // do not descend into this lod_group's children
            }
            for (const c of n.children ?? []) visit(c);
          };
          for (const c of node.children ?? []) visit(c);
          if (collected.length > 0) {
            nestedLodGroupPaths = collected;
            nestedLodMaxChildCount = maxCount;
          }
        }

        // Composed blend mode: `undefined` when NO ancestry level set one, else
        // the canonical inherited/authored mode (#1272). The `undefined` case is
        // exactly what tells us to fall back to the per-type default AND to leave
        // the layer non-explicit so it does not impose that default on descendants.
        const composedBlendingMode = getEffectiveAttrs(root, node.path).blending_mode;

        this.layerOrder.push(node.path);
        this.layers.set(node.path, {
          path: node.path,
          name,
          type: layerType,
          kind,
          visible: initialVisible,
          opacity: (node.attrs.opacity as number) ?? 1.0,
          // RAW like opacity, NOT composed: composeEffective substitutes
          // each layer's live values per ancestry node, so a composed
          // init would multiply ancestor κ in twice.
          absorption: (node.attrs.absorption as number) ?? 1.0,
          // RAW, and defaulted from the material's own constants rather than
          // re-spelled here: these are the values `MeshMaterial` starts at when the
          // attr is absent, so the slider must open on the same number the surface is
          // already rendering with. They do NOT compose along the ancestry (unlike
          // opacity/gamma) — a shade floor is a per-surface appearance choice, not a
          // multiplicative attr, and the writer never stamps them on a group.
          ambient: (node.attrs.ambient as number) ?? MESH_DEFAULTS.ambient,
          shadeExponent: (node.attrs.shade_exponent as number) ?? MESH_DEFAULTS.shadeExponent,
          alphaCutoff: (node.attrs.alpha_cutoff as number) ?? MESH_DEFAULTS.alphaCutoff,
          displayMin,
          displayMax,
          dataMin,
          dataMax,
          gamma: (node.attrs.gamma as number) ?? 1.0,
          // Blending mode is COMPOSED along the ancestry (nearest set
          // ancestor wins, normalized by composeAttrs) — a geometry leaf's
          // dropdown must show the mode the material actually renders with,
          // not the node's own (possibly absent / malformed) raw attr. An
          // unset chain composes to `undefined`; show the per-type default
          // (mesh → opaque, emissive → additive) the material would use — but
          // leave it NON-explicit so it is not pushed onto descendants (a
          // plain group layer merely displays `additive` as a neutral default;
          // each descendant keeps its own default until the control is used).
          //
          // Then RESOLVED for the layer's type, which is a separate concern from the
          // default and applies to an EXPLICIT mode too: a mesh cannot render
          // `volumetric`, so its material maps that to `opaque` and stamps the resolved
          // value. Without this wrap the panel showed Absorption (which no mesh shader
          // reads) and hid Alpha cutoff exactly when the cutout was active. A no-op for
          // the default path, since `defaultBlendingMode('mesh')` is already `opaque`.
          blendingMode: resolveLayerBlendingMode(
            layerType,
            composedBlendingMode ?? defaultBlendingMode(node.type)
          ),
          blendingModeExplicit: composedBlendingMode !== undefined,
          selected: false,
          colormap,
          supportsColormap,
          scalarDataRange: colormapScalarRange,
          colorDataRange,
          scalarWindow: usesColormap(node),
          lodGroupChildCount,
          partCount,
          nestedLodGroupPaths,
          nestedLodMaxChildCount,
        });
      }
    }

    // Recurse into children
    if (node.children) {
      for (const child of node.children) {
        this.walkSceneGraph(child, root);
      }
    }
  }

  /** Get all layers in display order */
  getLayers(): LayerInfo[] {
    return this.layerOrder.map((path) => this.layers.get(path)!);
  }

  /** Get a single layer by path */
  getLayer(path: string): LayerInfo | undefined {
    return this.layers.get(path);
  }

  /** Get the number of layers */
  get count(): number {
    return this.layerOrder.length;
  }

  // ─── Selection ───────────────────────────────────────────

  /**
   * Select a layer with napari-style modifiers.
   * - 'single': clear all, select this one
   * - 'add': toggle this one (Ctrl+click)
   * - 'range': select from lastClicked to this (Shift+click)
   */
  select(path: string, mode: SelectionMode): void {
    const layer = this.layers.get(path);
    if (!layer) return;

    switch (mode) {
      case 'single':
        for (const l of this.layers.values()) l.selected = false;
        layer.selected = true;
        this.lastClickedPath = path;
        break;

      case 'add':
        layer.selected = !layer.selected;
        this.lastClickedPath = path;
        break;

      case 'range': {
        const anchor = this.lastClickedPath;
        if (!anchor) {
          // No anchor — behave like single
          for (const l of this.layers.values()) l.selected = false;
          layer.selected = true;
          this.lastClickedPath = path;
          break;
        }

        const anchorIdx = this.layerOrder.indexOf(anchor);
        const targetIdx = this.layerOrder.indexOf(path);
        const lo = Math.min(anchorIdx, targetIdx);
        const hi = Math.max(anchorIdx, targetIdx);

        for (const l of this.layers.values()) l.selected = false;
        for (let i = lo; i <= hi; i++) {
          this.layers.get(this.layerOrder[i])!.selected = true;
        }
        // Don't update lastClickedPath for range — anchor stays
        break;
      }
    }

    this.notify();
  }

  /** Get all currently selected layers */
  getSelected(): LayerInfo[] {
    return this.getLayers().filter((l) => l.selected);
  }

  /** Get the "primary" selected layer (last clicked) for showing in controls */
  getPrimarySelected(): LayerInfo | undefined {
    if (this.lastClickedPath) {
      const layer = this.layers.get(this.lastClickedPath);
      if (layer?.selected) return layer;
    }
    // Fallback: first selected
    return this.getSelected()[0];
  }

  // ─── Mutations ───────────────────────────────────────────

  /** Toggle visibility (independent of selection) */
  setVisible(path: string, visible: boolean): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.visible = visible;
    this.notify();
  }

  /** Set display range for a layer */
  setDisplayRange(path: string, min: number, max: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.displayMin = min;
    layer.displayMax = max;
    this.notify();
  }

  /** Set gamma for a layer */
  setGamma(path: string, gamma: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.gamma = gamma;
    this.notify();
  }

  /** Set opacity for a layer */
  setOpacity(path: string, opacity: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.opacity = opacity;
    this.notify();
  }

  /** Set colormap for a layer */
  setColormap(path: string, colormap: string): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.colormap = colormap;
    this.notify();
  }

  /**
   * Re-default the display window after the layer's colormap was toggled.
   *
   * The window maps the RENDERED VALUE to [0, 1], and toggling a colormap
   * changes what that value is — so the sensible default changes with it:
   *
   * * colormap ON  → the value is the scalar (gsplat amplitude / per-element
   *   scalar), whose useful window is its data range. Keeping the direct-colour
   *   identity here would push `updateScalarRange(0, 1)` onto amplitudes that
   *   live in ~[1e-4, 0.02] and render near-black (the #522 failure).
   * * colormap OFF → the value is authored RGB, whose range IS [0, 1].
   *
   * A window carried over from the other mode is meaningless, so this
   * deliberately overwrites a user-set one.
   *
   * The slider BOUNDS follow the mode too, landing on exactly what a natively
   * authored layer of that mode gets at init. Merely widening them instead
   * would leave the useful window as an unusable sliver of the track — a
   * gsplat amplitude window of [1e-4, 0.02] inside [0, 1] bounds is 2% of it.
   */
  setColormapWindow(path: string, colormapOn: boolean): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.scalarWindow = colormapOn;
    const [min, max] = colormapOn ? (layer.scalarDataRange ?? [0, 1]) : [0, 1];
    layer.displayMin = min;
    layer.displayMax = max;
    // colormap ON  -> the scalar range (what a colormapped layer inits to)
    // colormap OFF -> [0, 1] widened to the authored RGB spread (what a
    //                 direct-colour layer inits to; keeps HDR colours > 1 in reach)
    const [boundMin, boundMax] = colormapOn
      ? [min, max]
      : [Math.min(0, layer.colorDataRange?.[0] ?? 0), Math.max(1, layer.colorDataRange?.[1] ?? 1)];
    layer.dataMin = Math.min(boundMin, min);
    layer.dataMax = Math.max(boundMax, max);
    this.notify();
  }

  /** Set blending mode for a layer. A user pick marks the mode EXPLICIT. */
  setBlendingMode(path: string, mode: BlendingMode): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.blendingMode = mode;
    layer.blendingModeExplicit = true;
    this.notify();
  }

  /** Set absorption (volumetric κ) for a layer */
  setAbsorption(path: string, absorption: number): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.absorption = absorption;
    this.notify();
  }

  /**
   * Apply a mutation to all selected layers.
   * The mutator receives each selected LayerInfo to modify in place.
   */
  applyToSelected(mutator: (layer: LayerInfo) => void): void {
    for (const layer of this.getSelected()) {
      mutator(layer);
    }
    this.notify();
  }

  // ─── Events ──────────────────────────────────────────────

  /** Subscribe to state changes. Returns an unsubscribe function. */
  onChange(listener: LayerChangeListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }

  /** Clear all state */
  dispose(): void {
    this.layers.clear();
    this.layerOrder = [];
    this.lastClickedPath = null;
    this.listeners.clear();
  }
}
