/**
 * Layer state management for the Layers panel.
 *
 * Pure data model (no DOM) that tracks per-layer state and provides
 * the min/max-to-intensity/offset mapping for the shader uniforms.
 */

import type { SceneNode } from '../../data/data-loader-types';
import { getEffectiveAttrs } from '../../data/attrs-composer';
import type { BlendingMode } from '../../types/blending';
import type { NodeKind } from '../../types/format-contract';
import { log, Modules } from '../../utils/log';
import { absorptionBoundsForNode } from './absorption-range';

/**
 * Geometry type of a layer.
 *
 * - ``group`` — composite container; controls fan out to descendants.
 * - ``points`` / ``lines`` / ``gsplats`` — leaf data layers.
 *
 * Specialized groups (``kind === 'lod'``, future ``'partition'``) appear in
 * the layers panel as their underlying ``display_type`` (one of the
 * three leaf types) — never as ``'group'``. The specialized-group
 * nature is surfaced via the ``kind`` field on ``LayerInfo``, which
 * drives the per-layer badge / LOD dropdown.
 */
export type LayerType = 'points' | 'lines' | 'gsplats' | 'group';

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
 * Derive a display data-range for a composite group (kind=lod / kind=partition)
 * that carries no range of its own, by walking to the FINEST descendant leaf
 * (largest `n_splats`) that declares one. The finest level is the full-detail
 * data, so its range is the representative signal window; using it avoids the
 * near-black render a [0, 1] fallback produces for a colormapped gsplat layer.
 * Returns undefined when no descendant declares a range.
 */
function deriveRangeFromDescendants(node: SceneNode): [number, number] | undefined {
  let best: [number, number] | undefined;
  let bestCount = -1;
  const visit = (n: SceneNode): void => {
    const r = (n.attrs.scalar_data_range ||
      n.attrs.color_data_range ||
      n.attrs.amplitude_data_range) as [number, number] | undefined;
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
   * Upper κ bound for this layer's Absorption slider, derived from the
   * geometry thickness recorded in the subtree's zarr metadata — κ's useful
   * magnitude goes as 1/thickness, so the track has to be per-layer. See
   * `absorption-range.ts`.
   */
  absorptionMax: number;
  /**
   * Smallest per-leaf κ bound in the subtree (the THICKEST descendant's
   * opaque point) — anchors the slider FLOOR, so a mixed-thickness group
   * can still reach near-transparency for its fattest geometry. Equals
   * `absorptionMax` for single-thickness or stat-less layers.
   */
  absorptionMinBound: number;
  /** Current display-range minimum (maps to intensity+offset in shader) */
  displayMin: number;
  /** Current display-range maximum */
  displayMax: number;
  /** Data-range minimum from color_data_range (slider lower bound) */
  dataMin: number;
  /** Data-range maximum from color_data_range (slider upper bound) */
  dataMax: number;
  /** Gamma correction value */
  gamma: number;
  /** Blending mode */
  blendingMode: BlendingMode;
  /** Whether this layer is selected in the list */
  selected: boolean;
  /** Active colormap name (undefined = direct RGB colors) */
  colormap?: string;
  /** Whether this node supports colormap (has scalars or amplitudes) */
  supportsColormap: boolean;
  /** Scalar data range for colormap normalization */
  scalarDataRange?: [number, number];
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

/** Computed shader uniforms from display range */
export interface DisplayUniforms {
  intensity: number;
  offset: number;
}

/**
 * Convert display [min, max] range to shader intensity (gain) and offset.
 *
 * The shader computes: `final_color = clamp(color * intensity + offset, 0, 1)`
 * We want `displayMin → 0` and `displayMax → 1`:
 *   intensity = 1 / (max - min)
 *   offset    = -min / (max - min)
 */
export function computeUniforms(displayMin: number, displayMax: number): DisplayUniforms {
  const range = displayMax - displayMin;
  if (Math.abs(range) < 1e-10) {
    // Degenerate range (min == max): there is nothing to window, so pass the
    // color through unchanged (identity gain/offset). A constant data range is
    // legitimate — e.g. classical-splat imports where per-splat opacity rides
    // in the color alpha and `amplitude_data_range` is [1, 1] (constant). The
    // old "very high contrast" mapping (gain 1000, offset -1000·min) turned
    // `color·1000 − 1000` into 0 for every non-white color, rendering the
    // whole scene black.
    return { intensity: 1.0, offset: 0.0 };
  }
  return {
    intensity: 1.0 / range,
    offset: -displayMin / range,
  };
}

/**
 * Inverse: recover display [min, max] from shader intensity and offset.
 *   displayMin = -offset / intensity
 *   displayMax = (1 - offset) / intensity
 */
export function computeDisplayRange(
  intensity: number,
  offset: number
): { min: number; max: number } {
  if (Math.abs(intensity) < 1e-10) {
    return { min: 0, max: 1 };
  }
  return {
    min: -offset / intensity,
    max: (1 - offset) / intensity,
  };
}

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
      const isLayerType =
        node.type === 'points' ||
        node.type === 'lines' ||
        node.type === 'gsplats' ||
        node.type === 'group';
      if (isLayerType) {
        const name = node.path.split('/').pop() || node.path;

        // Determine data range from zarr attrs. A composite group (kind=lod /
        // kind=partition) carries no range of its own, so derive it from the
        // finest (largest-n_splats) descendant leaf — otherwise the [0, 1]
        // fallback makes a colormapped gsplat layer render near-black (the
        // signal occupies only the bottom few % of [0, 1]).
        const colorRange = node.attrs.color_data_range as [number, number] | undefined;
        const ampRange = node.attrs.amplitude_data_range as [number, number] | undefined;
        const scalarRange = node.attrs.scalar_data_range as [number, number] | undefined;
        const dataRange = scalarRange ||
          colorRange ||
          ampRange ||
          deriveRangeFromDescendants(node) || [0, 1];

        // Colormap support — groups inherit no colormap, but they do apply
        // a chosen colormap to every data descendant that can accept one.
        // Gsplats only support a colormap when they actually have scalar
        // data (`has_scalars`) or an authored `colormap`; a bare gsplats
        // node with no scalars must NOT advertise colormap support, or
        // the UI offers a no-op colormap dropdown.
        // kind=lod / kind=partition groups are composite containers; the
        // colormap applies to descendants via composition just like a
        // plain group.
        const colormap = node.attrs.colormap as string | undefined;
        const supportsColormap = node.type === 'group' || !!node.attrs.has_scalars || !!colormap;
        const colormapScalarRange = scalarRange || ampRange || deriveRangeFromDescendants(node);

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
        const dataMin = Math.min(dataRange[0], displayMin);
        const dataMax = Math.max(dataRange[1], displayMax);

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

        // Per-layer κ track bounds from the subtree's recorded thickness —
        // one walk yields both ends (thinnest → max, thickest → floor anchor).
        const absorptionBounds = absorptionBoundsForNode(node);

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
          absorptionMax: absorptionBounds.max,
          absorptionMinBound: absorptionBounds.minBound,
          displayMin,
          displayMax,
          dataMin,
          dataMax,
          gamma: (node.attrs.gamma as number) ?? 1.0,
          // Blending mode is COMPOSED along the ancestry (nearest set
          // ancestor wins, normalized by composeAttrs) — the panel must
          // show the mode the material actually renders with, not the
          // node's own (possibly absent / malformed) raw attr.
          blendingMode: getEffectiveAttrs(root, node.path).blending_mode,
          selected: false,
          colormap,
          supportsColormap,
          scalarDataRange: colormapScalarRange,
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

  /** Set blending mode for a layer */
  setBlendingMode(path: string, mode: BlendingMode): void {
    const layer = this.layers.get(path);
    if (!layer) return;
    layer.blendingMode = mode;
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
