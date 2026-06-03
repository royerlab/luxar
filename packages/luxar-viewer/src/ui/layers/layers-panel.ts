/**
 * Layers Panel — napari-inspired per-layer control panel.
 *
 * Provides visibility toggle, [min, max] display range, gamma, and blending
 * mode controls for each scene node marked as `layer: true` in the zarr attrs.
 *
 * Multi-select: click = single, Ctrl+click = toggle, Shift+click = range.
 * Controls apply to all selected layers simultaneously.
 */

import * as THREE from 'three';
import type { SceneNode } from '../../data/data-loader-types';
import type { BlendingMode, CameraAwareMaterial } from '../../rendering';
import {
  LayerStateManager,
  computeDisplayRange,
  type LayerInfo,
  type SelectionMode,
} from './layer-state';
import { RangeSlider } from './range-slider';
import { LabeledSlider } from './labeled-slider';
import { config } from '../../config';
import { materialManager } from '../../rendering';
import { log, Modules } from '../../utils/log';
import { EventGroup } from '../../utils/cross-layer/event-group';
import { showToast } from '../toast';
import type { AnimationController } from '../../scene/animation/animation-controller';
import { getColormapTexture } from '../../rendering/colormap-textures';
import { supportsScalarColormap } from '../../rendering/material-colormap-helpers';
import { COLORMAP_CATEGORIES } from '../../rendering/colormap-data';
import { SceneLoaderManager } from '../../data/scene-loader-manager';
import type { LODGroupRegistry } from '../../scene/lod-group-registry';
import {
  composeAttrs,
  collectAncestorNodes,
  collectDataDescendants,
  type ComposableAttrs,
  type EffectiveAttrs,
} from '../../data/attrs-composer';
import {
  clampGamma,
  getBlendingState,
  liveLayerAttrs as deriveLiveLayerAttrs,
} from './attrs-utils';
import { clamp } from '../gui/format/value-formatting';

// Type guard: does this material have our update* methods?
export interface LuxarMaterial extends THREE.Material, CameraAwareMaterial {
  updateIntensity(v: number): void;
  updateOffset(v: number): void;
  updateGamma(v: number): void;
  updateOpacity(v: number): void;
  updateColormapTexture?(texture: THREE.DataTexture | null): void;
  updateScalarRange?(min: number, max: number): void;
  /**
   * Apply a blending mode to this material in-place.
   *
   * Optional because PointMaterial doesn't need it — its blending is
   * mode-agnostic at the material level (no `uProjectionMode`, no
   * intensity-squaring concern). For materials that DO need it
   * (GSplatMaterial, LineMaterial), call this instead of writing
   * `mat.blending`/`mat.blendEquation` directly so type-specific
   * factors and uniforms stay in sync.
   */
  applyBlendingMode?(mode: BlendingMode): void;
}

/**
 * Whether a material is currently rendering in colormap (LUT) mode —
 * the `USE_COLORMAP` shader define is the source of truth (set/cleared
 * by `updateColormapTexture`). In this mode the display range drives the
 * LUT value window and gamma warps the value pre-lookup, so neither
 * should be applied to the output color (see the material shaders).
 *
 * @internal Exported for unit testing the colormap-vs-direct routing.
 */
export function isColormapActive(mat: LuxarMaterial): boolean {
  const defines = (mat as unknown as { defines?: Record<string, unknown> | null }).defines;
  return !!defines && 'USE_COLORMAP' in defines;
}

/**
 * Push gamma + the display-range adjustment to a leaf material, routed by
 * whether it renders through a colormap LUT:
 *
 * - **Colormap (LUT) mode**: the display range defines the value window
 *   mapped into the LUT (`uScalarMin`/`uScalarScale`) and gamma warps that
 *   value before the lookup — both operate on the scalar, not the color.
 *   The composed display window is recovered from the gain/offset pair and
 *   pushed via `updateScalarRange`; the color GOG is bypassed in-shader, so
 *   `intensity`/`offset` are intentionally NOT pushed.
 * - **Direct-color mode**: GOG operates on the color (`intensity`/`offset`).
 *
 * Gamma is pushed in both modes (the shader applies it pre-LUT in colormap
 * mode, on the color otherwise). Opacity and blending are handled by the
 * caller. See the material shaders' `USE_COLORMAP` path.
 *
 * @internal Exported for unit testing.
 */
export function applyColorAdjustments(
  mat: LuxarMaterial,
  gamma: number,
  intensity: number,
  offset: number
): void {
  mat.updateGamma(gamma);
  if (isColormapActive(mat) && mat.updateScalarRange) {
    const { min, max } = computeDisplayRange(intensity, offset);
    mat.updateScalarRange(min, max);
  } else {
    mat.updateIntensity(intensity);
    mat.updateOffset(offset);
  }
}

function isLuxarMaterial(m: THREE.Material): m is LuxarMaterial {
  return (
    typeof (m as LuxarMaterial).updateIntensity === 'function' &&
    typeof (m as LuxarMaterial).updateGamma === 'function'
  );
}

const BLENDING_MODES: BlendingMode[] = ['additive', 'normal', 'max', 'opaque', 'luminous'];

export class LayersPanel {
  private container: HTMLElement;
  private rootGroup: THREE.Group | null = null;
  private sceneGraph: SceneNode | null = null;
  private animationController: AnimationController;

  private state = new LayerStateManager();

  /** Expose layer state for external consumers (e.g., colormap legend). */
  get layerState(): LayerStateManager {
    return this.state;
  }
  private panelEl: HTMLElement | null = null;
  private listEl: HTMLElement | null = null;
  private controlsEl: HTMLElement | null = null;
  private rangeSlider: RangeSlider | null = null;
  private gammaSlider: LabeledSlider | null = null;
  private opacitySlider: LabeledSlider | null = null;
  private blendSelect: HTMLSelectElement | null = null;
  private colormapSelect: HTMLSelectElement | null = null;
  /**
   * "Active level" dropdown for ``lod_group`` layers. Shown only when
   * the primary selected layer is an lod_group; hidden otherwise.
   * Options: ``auto`` plus one ``lock to level <i>`` entry per child.
   */
  private lodLevelSelect: HTMLSelectElement | null = null;
  /**
   * Status span next to the dropdown showing the currently-rendering
   * level (e.g. "rendering: 2"). Refreshed on each renderControls()
   * call; not per-frame for v1.
   */
  private lodLevelStatus: HTMLSpanElement | null = null;
  private visible = false;
  /**
   * Tracks every event listener attached during buildPanel/renderList
   * so clear()/dispose() can tear them all down with a single call.
   * Without this, listeners attached to detached DOM nodes hold
   * closures referencing the panel until the GC reclaims the
   * subtree — fragile, hard to test, and inconsistent with the rest
   * of the viewer's listener-tracking pattern.
   */
  private events = new EventGroup();

  // Row elements keyed by layer path for targeted DOM updates
  private rowElements = new Map<string, HTMLElement>();

  // State change unsubscribe handle
  private unsubscribeState: (() => void) | null = null;

  // Suppresses renderControls() during user-driven control interactions
  // to prevent programmatic .value= from fighting with the user's drag
  private controlsInteracting = false;

  // Tracks layers panel height to reposition the rendering controls (GUI) below
  private resizeObserver: ResizeObserver | null = null;

  constructor(container: HTMLElement, animationController: AnimationController) {
    this.container = container;
    this.animationController = animationController;
  }

  // ─── Lifecycle ─────────────────────────────────────────

  /**
   * Initialize the panel from a loaded scene.
   * Must be called after the scene is loaded and rootGroup is available.
   */
  initFromScene(rootGroup: THREE.Group, sceneGraph: SceneNode): void {
    // Clean up previous state
    this.clear();

    this.rootGroup = rootGroup;
    this.sceneGraph = sceneGraph;
    this.state.initFromSceneGraph(sceneGraph);

    // Subscribe to state changes — only update selection highlights and controls,
    // NOT full list re-renders (those are expensive and cause flicker)
    this.unsubscribeState = this.state.onChange(() => {
      this.updateRowHighlights();
      // Only re-sync controls from state when the change came from selection,
      // NOT when it came from the controls themselves (which would fight with
      // the user's ongoing slider drag).
      if (!this.controlsInteracting) {
        this.renderControls();
      }
    });

    if (this.state.count === 0) {
      log.info(Modules.UI, 'No layers found in scene (no nodes with layer=true)');
      return;
    }

    log.info(Modules.UI, `Layers panel: ${this.state.count} layer(s) found`);

    // Build DOM
    this.buildPanel();

    // Sync initial display range to materials — the layer state may compute a
    // displayMin/displayMax from the data range that differs from the material's
    // default intensity=1/offset=0 (which corresponds to display range [0,1]).
    // Without this, the first slider interaction causes a sudden brightness jump.
    // Also honor the authoring-time `visible` attr by applying initial
    // visibility to the scene object.
    const layers = this.state.getLayers();
    for (const layer of layers) {
      this.applyDisplayRange(layer);
      if (!layer.visible) {
        this.applyVisibility(layer.path, false);
      }
    }

    // Auto-select first layer
    if (layers.length > 0) {
      this.state.select(layers[0].path, 'single');
    }
  }

  show(): void {
    if (!this.panelEl || this.state.count === 0) return;
    this.panelEl.style.display = 'flex';
    this.visible = true;
    this.repositionGUI();
  }

  hide(): void {
    if (!this.panelEl) return;
    this.panelEl.style.display = 'none';
    this.visible = false;
    this.repositionGUI();
  }

  toggle(): void {
    if (this.visible) {
      this.hide();
    } else {
      if (this.state.count === 0) {
        showToast('No layers in this scene (use layer=True in Python API)');
        log.info(
          Modules.UI,
          'Layers panel toggle: no layers found. Use layer=True on add_points/add_lines/add_gsplats.'
        );
        return;
      }
      this.show();
    }
  }

  isVisible(): boolean {
    return this.visible;
  }

  dispose(): void {
    this.clear();
    this.state.dispose();
  }

  private clear(): void {
    // Unsubscribe from state changes to prevent ghost callbacks
    if (this.unsubscribeState) {
      this.unsubscribeState();
      this.unsubscribeState = null;
    }
    // Tear down every listener attached during buildPanel/renderList.
    // Re-instantiate so a subsequent show() / initFromScene() starts
    // with a fresh group rather than a disposed one.
    this.events.dispose();
    this.events = new EventGroup();
    this.sceneGraph = null;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    // Reset visibility and GUI position before removing the panel DOM
    const wasVisible = this.visible;
    this.visible = false;
    if (wasVisible) this.repositionGUI();
    this.rangeSlider?.dispose();
    this.rangeSlider = null;
    this.gammaSlider?.dispose();
    this.gammaSlider = null;
    this.opacitySlider?.dispose();
    this.opacitySlider = null;
    this.blendSelect = null;
    this.lodLevelSelect = null;
    this.lodLevelStatus = null;
    this.rowElements.clear();
    this.panelEl?.remove();
    this.panelEl = null;
    this.listEl = null;
    this.controlsEl = null;
  }

  // ─── DOM Construction ──────────────────────────────────

  private buildPanel(): void {
    // Panel container
    const panel = document.createElement('div');
    panel.className = 'luxar-layers-panel';
    panel.style.zIndex = String(config.ui.zIndex.layersPanel);
    panel.style.display = 'none'; // Hidden by default
    this.panelEl = panel;

    // Header
    const header = document.createElement('div');
    header.className = 'luxar-layers-panel__header';
    const title = document.createElement('span');
    title.className = 'luxar-layers-panel__title';
    title.textContent = 'Layers';
    const closeBtn = document.createElement('button');
    closeBtn.className = 'luxar-layers-panel__close';
    closeBtn.textContent = '\u00d7';
    // Advertise Escape rather than L: the L key-binding early-returns
    // when focus is inside the panel, so it doesn't actually close
    // from keyboard while the panel has focus. Escape is handled by
    // the global key dispatcher and works regardless of focus.
    closeBtn.title = 'Close (Esc)';
    closeBtn.setAttribute('aria-label', 'Close layers panel');
    closeBtn.setAttribute('aria-keyshortcuts', 'escape');
    this.events.on(closeBtn, 'click', () => this.hide());
    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    // Layer list (scrollable). ARIA listbox semantics so keyboard users can
    // navigate rows with ArrowUp/ArrowDown and select with Enter/Space.
    // Multi-select via Ctrl/Cmd/Shift is reflected with aria-multiselectable.
    const list = document.createElement('div');
    list.className = 'luxar-layers-panel__list';
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', 'Scene layers');
    list.setAttribute('aria-multiselectable', 'true');
    this.listEl = list;
    panel.appendChild(list);

    // Controls section
    const controls = document.createElement('div');
    controls.className = 'luxar-layers-panel__controls';
    this.controlsEl = controls;
    panel.appendChild(controls);

    this.container.appendChild(panel);

    // Track panel size changes to keep the GUI positioned below
    this.resizeObserver?.disconnect();
    this.resizeObserver = new ResizeObserver(() => {
      if (this.visible) this.repositionGUI();
    });
    this.resizeObserver.observe(panel);

    // Build the list rows and controls once
    this.renderList();
    this.buildControls();
    this.renderControls();
  }

  // ─── Layer List ────────────────────────────────────────

  /** Build the full list of layer rows (called once on init, not on every state change) */
  private renderList(): void {
    if (!this.listEl) return;

    const layers = this.state.getLayers();

    this.listEl.innerHTML = '';
    this.rowElements.clear();

    for (const layer of layers) {
      const row = this.createLayerRow(layer);
      this.rowElements.set(layer.path, row);
      this.listEl.appendChild(row);
    }
  }

  /** Update selection highlights and visibility classes without rebuilding DOM */
  private updateRowHighlights(): void {
    let hasFocusable = false;
    for (const layer of this.state.getLayers()) {
      const row = this.rowElements.get(layer.path);
      if (!row) continue;

      row.classList.toggle('luxar-layer-row--selected', layer.selected);
      row.classList.toggle('luxar-layer-row--hidden', !layer.visible);
      row.setAttribute('aria-selected', layer.selected ? 'true' : 'false');

      // First selected row is the keyboard tab stop; others get tabIndex -1
      // (still focusable programmatically for ArrowUp/Down).
      if (layer.selected && !hasFocusable) {
        row.tabIndex = 0;
        hasFocusable = true;
      } else {
        row.tabIndex = -1;
      }

      // Update eye button text + ARIA state
      const eyeBtn = row.querySelector('.luxar-layer-row__eye') as HTMLButtonElement | null;
      if (eyeBtn) {
        eyeBtn.textContent = layer.visible ? '\u{1F441}' : '\u{1F441}\u200D\u{1F5E8}';
        const tooltip = layer.visible ? 'Hide layer' : 'Show layer';
        eyeBtn.title = tooltip;
        eyeBtn.setAttribute('aria-label', `${tooltip}: ${layer.name}`);
        eyeBtn.setAttribute('aria-pressed', layer.visible ? 'true' : 'false');
      }
    }

    // If nothing is selected, make the first row the tab stop so users can
    // enter the listbox with the keyboard.
    if (!hasFocusable) {
      const first = this.rowElements.values().next().value as HTMLElement | undefined;
      if (first) first.tabIndex = 0;
    }
  }

  private createLayerRow(layer: LayerInfo): HTMLElement {
    const row = document.createElement('div');
    row.className = 'luxar-layer-row';
    if (layer.selected) row.classList.add('luxar-layer-row--selected');
    if (!layer.visible) row.classList.add('luxar-layer-row--hidden');

    // ARIA option semantics — see listbox setup in buildPanel().
    //
    // Note: a listbox option ideally shouldn't contain nested
    // interactive elements, but the eye toggle is a real <button>.
    // The arrow-key navigation + Enter/Space selection on rows is
    // the listbox idiom screen readers expect. The eye button is
    // reachable via Tab as a separate focusable element. A move to
    // role=tree+treeitem (which permits nested controls) would be
    // cleaner but breaks the row-selection pattern. The escape
    // valve is the explicit aria-label on the eye button so AT
    // users hear "Hide layer: <name>" distinctly from "Layer
    // <name> (<type>)".
    row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', layer.selected ? 'true' : 'false');
    row.setAttribute('aria-label', `${layer.name} (${layer.type})`);
    row.tabIndex = -1; // updateRowHighlights() promotes the active one to 0

    // Eye toggle — visibility is independent of selection. <button> already
    // has role=button, is focusable, and triggers click on Space/Enter, so we
    // only need aria-pressed + a descriptive aria-label for screen readers.
    const eyeBtn = document.createElement('button');
    eyeBtn.type = 'button';
    eyeBtn.className = 'luxar-layer-row__eye';
    eyeBtn.textContent = layer.visible ? '\u{1F441}' : '\u{1F441}\u200D\u{1F5E8}';
    const tooltip = layer.visible ? 'Hide layer' : 'Show layer';
    eyeBtn.title = tooltip;
    eyeBtn.setAttribute('aria-label', `${tooltip}: ${layer.name}`);
    eyeBtn.setAttribute('aria-pressed', layer.visible ? 'true' : 'false');
    this.events.on(eyeBtn, 'click', (e) => {
      e.stopPropagation(); // Don't trigger row selection

      // Capture the new visibility BEFORE mutating state
      const currentLayer = this.state.getLayer(layer.path);
      if (!currentLayer) return;
      const newVisible = !currentLayer.visible;

      // Update state and apply to scene
      this.state.setVisible(layer.path, newVisible);
      this.applyVisibility(layer.path, newVisible);
    });

    // Layer name
    const nameEl = document.createElement('span');
    nameEl.className = 'luxar-layer-row__name';
    nameEl.textContent = layer.name;
    nameEl.title = layer.path;

    // Type badge
    const typeMap: Record<string, string> = {
      points: 'pts',
      lines: 'lines',
      gsplats: 'splat',
      group: 'group',
    };
    const badge = document.createElement('span');
    badge.className = 'luxar-layer-row__badge';
    badge.textContent = typeMap[layer.type] || layer.type;

    // Optional kind-specific badge: ``N LODs`` for kind=lod, ``N parts``
    // for kind=partition. When a kind=partition layer wraps kind=lod
    // descendants, the badge combines both counts as
    // ``N parts × M LODs`` (M = max child count across nested groups).
    // Appears alongside the type badge; visible-only when the count is
    // > 0.
    let kindBadge: HTMLSpanElement | null = null;
    if (layer.kind === 'lod' && (layer.lodGroupChildCount ?? 0) > 0) {
      kindBadge = document.createElement('span');
      kindBadge.className = 'luxar-layer-row__badge luxar-layer-row__badge--kind';
      kindBadge.textContent = `${layer.lodGroupChildCount} LODs`;
    } else if (layer.kind === 'partition' && (layer.partCount ?? 0) > 0) {
      kindBadge = document.createElement('span');
      kindBadge.className = 'luxar-layer-row__badge luxar-layer-row__badge--kind';
      if (
        layer.nestedLodGroupPaths &&
        layer.nestedLodGroupPaths.length > 0 &&
        (layer.nestedLodMaxChildCount ?? 0) > 0
      ) {
        kindBadge.textContent = `${layer.partCount} parts × ${layer.nestedLodMaxChildCount} LODs`;
      } else {
        kindBadge.textContent = `${layer.partCount} parts`;
      }
    }

    // Row click — selection
    this.events.on(row, 'click', (e) => {
      let mode: SelectionMode = 'single';
      if (e.ctrlKey || e.metaKey) mode = 'add';
      else if (e.shiftKey) mode = 'range';
      this.state.select(layer.path, mode);
    });

    // Row keyboard navigation — listbox idiom: ArrowUp/Down moves focus
    // (and selects on simple navigation), Enter/Space select with the
    // current modifier.
    this.events.on(row, 'keydown', (e) => {
      const layers = this.state.getLayers();
      const idx = layers.findIndex((l) => l.path === layer.path);
      if (idx < 0) return;

      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const nextIdx =
          e.key === 'ArrowDown' ? Math.min(layers.length - 1, idx + 1) : Math.max(0, idx - 1);
        const next = layers[nextIdx];
        const nextRow = this.rowElements.get(next.path);
        if (nextRow) {
          this.state.select(next.path, 'single');
          nextRow.focus();
        }
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        let mode: SelectionMode = 'single';
        if (e.ctrlKey || e.metaKey) mode = 'add';
        else if (e.shiftKey) mode = 'range';
        this.state.select(layer.path, mode);
      }
    });

    row.appendChild(eyeBtn);
    row.appendChild(nameEl);
    row.appendChild(badge);
    if (kindBadge !== null) {
      row.appendChild(kindBadge);
    }
    return row;
  }

  // ─── Controls ──────────────────────────────────────────

  private buildControls(): void {
    if (!this.controlsEl) return;
    this.controlsEl.innerHTML = '';

    // Display range
    const rangeContainer = document.createElement('div');
    rangeContainer.className = 'luxar-layers-panel__control-group';
    this.controlsEl.appendChild(rangeContainer);

    this.rangeSlider = new RangeSlider({
      container: rangeContainer,
      min: 0,
      max: 1,
      valueLow: 0,
      valueHigh: 1,
      label: 'Display range',
      onChange: (low, high) => {
        this.controlsInteracting = true;
        this.state.applyToSelected((l) => {
          l.displayMin = low;
          l.displayMax = high;
        });
        for (const sel of this.state.getSelected()) {
          this.applyDisplayRange(sel);
        }
        this.controlsInteracting = false;
      },
      onBoundsChange: (min, max) => {
        this.controlsInteracting = true;
        this.state.applyToSelected((l) => {
          l.dataMin = min;
          l.dataMax = max;
        });
        this.controlsInteracting = false;
      },
    });

    this.gammaSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Gamma',
      min: 0.2,
      max: 5.0,
      step: 0.01,
      initialValue: 1.0,
      constrain: clampGamma,
      onChange: (val) => {
        this.controlsInteracting = true;
        this.state.applyToSelected((l) => {
          l.gamma = val;
        });
        for (const sel of this.state.getSelected()) {
          this.applyGamma(sel);
        }
        this.controlsInteracting = false;
      },
    });

    this.opacitySlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Opacity',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: 1.0,
      constrain: (v) => clamp(v, 0, 1),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.state.applyToSelected((l) => {
          l.opacity = val;
        });
        for (const sel of this.state.getSelected()) {
          this.applyOpacity(sel);
        }
        this.controlsInteracting = false;
      },
    });

    // Blending mode
    const blendGroup = document.createElement('div');
    blendGroup.className = 'luxar-layers-panel__control-group';
    const blendLabel = document.createElement('div');
    blendLabel.className = 'luxar-layers-panel__control-label';
    blendLabel.textContent = 'Blend';

    this.blendSelect = document.createElement('select');
    this.blendSelect.className = 'luxar-layers-panel__select';
    for (const mode of BLENDING_MODES) {
      const opt = document.createElement('option');
      opt.value = mode;
      opt.textContent = mode;
      this.blendSelect.appendChild(opt);
    }
    this.events.on(this.blendSelect, 'change', () => {
      this.controlsInteracting = true;
      const mode = this.blendSelect!.value as BlendingMode;
      this.state.applyToSelected((l) => {
        l.blendingMode = mode;
      });
      for (const sel of this.state.getSelected()) {
        this.applyBlendingMode(sel);
      }
      this.controlsInteracting = false;
    });
    blendGroup.appendChild(blendLabel);
    blendGroup.appendChild(this.blendSelect);
    this.controlsEl.appendChild(blendGroup);

    // Colormap selector (only shown for layers that support colormap)
    const cmGroup = document.createElement('div');
    cmGroup.className = 'luxar-layers-panel__control-group';
    const cmLabel = document.createElement('div');
    cmLabel.className = 'luxar-layers-panel__control-label';
    cmLabel.textContent = 'Colormap';

    this.colormapSelect = document.createElement('select');
    this.colormapSelect.className = 'luxar-layers-panel__select';
    // "None" option for layers using direct RGB colors
    const noneOpt = document.createElement('option');
    noneOpt.value = '';
    noneOpt.textContent = '(direct colors)';
    this.colormapSelect.appendChild(noneOpt);

    // Add categorized options
    for (const [category, names] of Object.entries(COLORMAP_CATEGORIES)) {
      const optgroup = document.createElement('optgroup');
      optgroup.label = category;
      for (const name of names) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        optgroup.appendChild(opt);
      }
      this.colormapSelect.appendChild(optgroup);
    }

    this.events.on(this.colormapSelect, 'change', () => {
      this.controlsInteracting = true;
      const cmName = this.colormapSelect!.value || undefined;
      this.state.applyToSelected((l) => {
        l.colormap = cmName;
      });
      for (const sel of this.state.getSelected()) {
        this.applyColormap(sel);
      }
      this.controlsInteracting = false;
    });
    cmGroup.appendChild(cmLabel);
    cmGroup.appendChild(this.colormapSelect);
    this.controlsEl.appendChild(cmGroup);

    // Active-level selector — only meaningful for lod_group layers,
    // hidden otherwise (see renderControls). The dropdown's option
    // list is rebuilt per layer in renderControls() because child
    // counts vary; here we just allocate the container + handler.
    const lodGroup = document.createElement('div');
    lodGroup.className = 'luxar-layers-panel__control-group';
    const lodLabel = document.createElement('div');
    lodLabel.className = 'luxar-layers-panel__control-label';
    lodLabel.textContent = 'Active level';

    this.lodLevelSelect = document.createElement('select');
    this.lodLevelSelect.className = 'luxar-layers-panel__select';

    this.lodLevelStatus = document.createElement('span');
    this.lodLevelStatus.className = 'luxar-layers-panel__control-value';

    this.events.on(this.lodLevelSelect, 'change', () => {
      this.controlsInteracting = true;
      const value = this.lodLevelSelect!.value;
      const primary = this.state.getPrimarySelected();
      if (primary) {
        const registry = this.getLodGroupRegistry();
        if (registry) {
          const mode = value === 'auto' ? 'auto' : { lockLevel: Number(value) };
          // Resolve the set of paths to update. A kind=lod layer updates
          // itself; a kind=partition layer that wraps lod_groups broadcasts
          // to every nested path (clamped per-group by setSelectorMode
          // on ragged ladders — see lod-group-registry).
          const paths: string[] =
            primary.kind === 'lod'
              ? [primary.path]
              : primary.kind === 'partition' &&
                  primary.nestedLodGroupPaths &&
                  primary.nestedLodGroupPaths.length > 0
                ? primary.nestedLodGroupPaths
                : [];
          let anyApplied = false;
          for (const p of paths) {
            try {
              registry.setSelectorMode(p, mode);
              anyApplied = true;
            } catch (err) {
              log.warning(Modules.UI, `Failed to set lod_group selector: ${err}`);
            }
          }
          // The actual visibility swap happens in a per-frame callback;
          // if the animation loop is idle (no camera/slice change),
          // setSelectorMode alone is not enough. Wake the loop so the
          // new active level is painted.
          if (anyApplied) this.requestRender();
        }
      }
      this.controlsInteracting = false;
    });
    lodGroup.appendChild(lodLabel);
    lodGroup.appendChild(this.lodLevelSelect);
    lodGroup.appendChild(this.lodLevelStatus);
    this.controlsEl.appendChild(lodGroup);
  }

  /**
   * Look up the LOD-group registry for the currently-loaded scene.
   *
   * Lazy lookup via the SceneLoaderManager (the layers panel can't
   * import scene/ directly without violating the data → ui layer
   * direction; the SceneLoaderManager hands us the loader's registry
   * field). Returns ``null`` when no scene is loaded or the loader
   * was created without a registry factory wired up.
   */
  private getLodGroupRegistry(): LODGroupRegistry | null {
    const loader = SceneLoaderManager.getInstance().getDefaultLoader();
    return loader?.lodGroupRegistry ?? null;
  }

  /** Populate the lod-level dropdown's options for the given child count. */
  private renderLodLevelOptions(childCount: number): void {
    if (!this.lodLevelSelect) return;
    // Clear and rebuild — option counts vary per lod_group.
    this.lodLevelSelect.innerHTML = '';
    const autoOpt = document.createElement('option');
    autoOpt.value = 'auto';
    autoOpt.textContent = 'auto';
    this.lodLevelSelect.appendChild(autoOpt);
    for (let i = 0; i < childCount; i++) {
      const opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = `lock to level ${i}`;
      this.lodLevelSelect.appendChild(opt);
    }
  }

  /** Update controls to reflect the primary selected layer's values */
  private renderControls(): void {
    const primary = this.state.getPrimarySelected();
    if (!primary) return;

    if (this.rangeSlider) {
      this.rangeSlider.setBounds(primary.dataMin, primary.dataMax);
      this.rangeSlider.setValues(primary.displayMin, primary.displayMax);
    }

    this.gammaSlider?.setValue(primary.gamma);
    this.opacitySlider?.setValue(primary.opacity);

    if (this.blendSelect) {
      this.blendSelect.value = primary.blendingMode;
    }

    if (this.colormapSelect) {
      if (primary.supportsColormap) {
        this.colormapSelect.parentElement!.style.display = '';
        this.colormapSelect.value = primary.colormap ?? '';
      } else {
        // Hide colormap control for layers that don't support it
        this.colormapSelect.parentElement!.style.display = 'none';
      }
    }

    // Active-level dropdown — shown for kind=lod layers AND for kind=partition
    // layers that wrap nested lod_groups (broadcast). The dropdown
    // option list reflects either the layer's own child count
    // (kind=lod) or the largest nested ladder (kind=partition).
    if (this.lodLevelSelect && this.lodLevelStatus) {
      const lodContainer = this.lodLevelSelect.parentElement!;
      const broadcastPartition =
        primary.kind === 'partition' &&
        primary.nestedLodGroupPaths &&
        primary.nestedLodGroupPaths.length > 0 &&
        (primary.nestedLodMaxChildCount ?? 0) > 0;
      if (primary.kind === 'lod' && (primary.lodGroupChildCount ?? 0) > 0) {
        lodContainer.style.display = '';
        this.renderLodLevelOptions(primary.lodGroupChildCount!);

        const registry = this.getLodGroupRegistry();
        const entry = registry?.get(primary.path);
        if (entry) {
          this.lodLevelSelect.value =
            entry.selectorMode === 'auto' ? 'auto' : String(entry.selectorMode.lockLevel);
          this.lodLevelStatus.textContent = `rendering: ${entry.activeChildIndex}`;
        } else {
          // Registry not yet populated (e.g., scene still loading) —
          // default to "auto" and clear the status.
          this.lodLevelSelect.value = 'auto';
          this.lodLevelStatus.textContent = '';
        }
      } else if (broadcastPartition) {
        lodContainer.style.display = '';
        this.renderLodLevelOptions(primary.nestedLodMaxChildCount!);

        // Sync widget state from the FIRST nested entry — they should
        // be lock-stepped after a broadcast change, and pre-broadcast
        // divergence (rare: legacy authored values) is acceptable
        // ambiguity here.
        const registry = this.getLodGroupRegistry();
        const firstPath = primary.nestedLodGroupPaths![0];
        const entry = registry?.get(firstPath);
        if (entry) {
          this.lodLevelSelect.value =
            entry.selectorMode === 'auto' ? 'auto' : String(entry.selectorMode.lockLevel);
          this.lodLevelStatus.textContent = `${primary.nestedLodGroupPaths!.length} nested LOD groups`;
        } else {
          this.lodLevelSelect.value = 'auto';
          this.lodLevelStatus.textContent = '';
        }
      } else {
        lodContainer.style.display = 'none';
      }
    }
  }

  // ─── Scene Application ─────────────────────────────────
  //
  // Rendering attributes compose along the scene graph per the Luxar
  // composition spec (opacity/gamma/intensity multiply, offset adds,
  // blending_mode takes the nearest ancestor's choice). Every time a
  // layer's slider moves, we recompose effective values for each affected
  // data-leaf (the layer itself for a data-node layer, or every data
  // descendant for a group layer) and push the result into the material.
  // Authoring-time zarr values are used for non-layer nodes in the chain;
  // live panel state overrides them for `layer=True` nodes.

  private getMesh(path: string): THREE.Object3D | null {
    if (!this.rootGroup) return null;
    return this.rootGroup.getObjectByName(path) ?? null;
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
    if (!this.sceneGraph) return [];
    const chain = collectAncestorNodes(this.sceneGraph, path);
    const target = chain[chain.length - 1];
    if (!target) return [];
    if (target.type === 'group') return collectDataDescendants(target);
    return [target];
  }

  /**
   * Compute the layer's current live composable attributes. Thin wrapper
   * around {@link liveLayerAttrs} so the four call sites in this file
   * keep their compact `this.liveLayerAttrs(...)` shape.
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
    if (!this.sceneGraph) return null;
    const ancestors = collectAncestorNodes(this.sceneGraph, leafPath);
    const chain: ComposableAttrs[] = ancestors.map((node) => {
      const layerInfo = this.state.getLayer(node.path);
      if (layerInfo) return this.liveLayerAttrs(layerInfo);
      return {
        opacity: node.attrs.opacity as number | undefined,
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
      applyColorAdjustments(mat, eff.gamma, eff.intensity, eff.offset);
      this.applyBlendingStateToMaterial(mat, eff.blending_mode);
    }
    this.requestRender();
  }

  private applyVisibility(path: string, visible: boolean): void {
    const obj = this.getMesh(path);
    if (obj) {
      obj.visible = visible;
      this.requestRender();
    }
  }

  private applyDisplayRange(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  private applyGamma(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  private applyOpacity(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  private applyBlendingMode(layer: LayerInfo): void {
    this.applyComposed(layer);
  }

  /**
   * Colormap applies per-leaf (not composed). For a group-layer we push
   * the selected colormap to every data descendant that accepts one.
   */
  private applyColormap(layer: LayerInfo): void {
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
    this.requestRender();
  }

  /**
   * Reposition the rendering controls (`.luxar-gui`) so it sits below the
   * layers panel when visible, or resets to its default position when hidden.
   */
  private repositionGUI(): void {
    // Target the rendering controls GUI specifically (not the recording panel GUI)
    const gui = document.querySelector(
      '.luxar-gui:not(.luxar-recording-panel)'
    ) as HTMLElement | null;
    if (!gui) return;

    if (this.visible && this.panelEl) {
      const rect = this.panelEl.getBoundingClientRect();
      gui.style.top = `${rect.bottom + 8}px`;
    } else {
      gui.style.top = '20px';
    }
  }

  private requestRender(): void {
    this.animationController.startAnimation();
  }
}
