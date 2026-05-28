/**
 * Layer state management for the Layers panel.
 *
 * Pure data model (no DOM) that tracks per-layer state and provides
 * the min/max-to-intensity/offset mapping for the shader uniforms.
 */

import type { SceneNode } from '../../data/data-loader-types';
import type { BlendingMode } from '../../rendering/material-manager';

/**
 * Geometry type of a layer.
 *
 * - ``group`` — composite container; controls fan out to descendants.
 * - ``lod_group`` — view-driven LOD container with an active-level
 *   selector dropdown (auto / lock to level N).
 * - ``points`` / ``lines`` / ``gsplats`` — leaf data layers.
 */
export type LayerType = 'points' | 'lines' | 'gsplats' | 'group' | 'lod_group';

/** Information about a single layer in the Layers panel */
export interface LayerInfo {
  /** Zarr path (e.g. "group/channel_gfp") — used as unique key */
  path: string;
  /** Display name (last segment of path) */
  name: string;
  /** Geometry type */
  type: LayerType;
  /** Whether the layer is visible in the scene */
  visible: boolean;
  /** Opacity (0–1) */
  opacity: number;
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
   * For ``type === 'lod_group'`` layers: number of child levels. Drives
   * the "Active level" dropdown's option count. Absent for other types.
   */
  lodGroupChildCount?: number;
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
    // Degenerate range: clamp to a very high contrast
    return { intensity: 1000, offset: -displayMin * 1000 };
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

    this.walkSceneGraph(root);
  }

  private walkSceneGraph(node: SceneNode): void {
    // Skip the root scene node; collect anything else with layer=true.
    // Groups exposed as layers act as composites — their controls fan out
    // to every data descendant when applied in the scene.
    if (node.type !== 'scene' && node.attrs.layer === true) {
      const isLayerType =
        node.type === 'points' ||
        node.type === 'lines' ||
        node.type === 'gsplats' ||
        node.type === 'group' ||
        node.type === 'lod_group';
      if (isLayerType) {
        const name = node.path.split('/').pop() || node.path;

        // Determine data range from zarr attrs. Groups don't have their
        // own ranges — fall back to [0, 1] so the slider remains usable.
        const colorRange = node.attrs.color_data_range as [number, number] | undefined;
        const ampRange = node.attrs.amplitude_data_range as [number, number] | undefined;
        const scalarRange = node.attrs.scalar_data_range as [number, number] | undefined;
        const dataRange = scalarRange || colorRange || ampRange || [0, 1];

        // Colormap support — groups inherit no colormap, but they do apply
        // a chosen colormap to every data descendant that can accept one.
        // Gsplats only support a colormap when they actually have scalar
        // data (`has_scalars`) or an authored `colormap`; a bare gsplats
        // node with no scalars must NOT advertise colormap support, or
        // the UI offers a no-op colormap dropdown.
        // lod_group is a composite container (like group); colormap
        // applies to descendants via composition.
        const colormap = node.attrs.colormap as string | undefined;
        const supportsColormap =
          node.type === 'group' ||
          node.type === 'lod_group' ||
          !!node.attrs.has_scalars ||
          !!colormap;
        const colormapScalarRange = scalarRange || ampRange;

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

        // lod_group child count — drives the "Active level" dropdown's
        // option list. Other node types leave this undefined.
        const lodGroupChildCount =
          node.type === 'lod_group' ? (node.children?.length ?? 0) : undefined;

        this.layerOrder.push(node.path);
        this.layers.set(node.path, {
          path: node.path,
          name,
          type: node.type as LayerType,
          visible: initialVisible,
          opacity: (node.attrs.opacity as number) ?? 1.0,
          displayMin,
          displayMax,
          dataMin,
          dataMax,
          gamma: (node.attrs.gamma as number) ?? 1.0,
          blendingMode: ((node.attrs.blending_mode as string) ?? 'additive') as BlendingMode,
          selected: false,
          colormap,
          supportsColormap,
          scalarDataRange: colormapScalarRange,
          lodGroupChildCount,
        });
      }
    }

    // Recurse into children
    if (node.children) {
      for (const child of node.children) {
        this.walkSceneGraph(child);
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
