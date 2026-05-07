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
import type { BlendingMode } from '../../rendering/material-manager';
import type { CameraAwareMaterial } from '../../rendering/camera-aware-material';
import {
  LayerStateManager,
  computeUniforms,
  type LayerInfo,
  type SelectionMode,
} from './layer-state';
import { RangeSlider } from './range-slider';
import { LabeledSlider } from './labeled-slider';
import { config } from '../../config';
import { materialManager } from '../../rendering/material-manager';
import { log, Modules } from '../../utils/log';
import { showToast } from '../helpers';
import type { AnimationController } from '../../scene/animation-controller';
import { getColormapTexture } from '../../rendering/colormap-textures';
import { COLORMAP_CATEGORIES } from '../../rendering/colormap-data';
import {
  composeAttrs,
  collectAncestorNodes,
  collectDataDescendants,
  type ComposableAttrs,
  type EffectiveAttrs,
} from '../../data/utils/attrs-composer';
import { clamp } from '../gui/utils/value-formatting';

/** Clamp a gamma value to a safe range for the shader (prevents division by zero and extreme exponents) */
function clampGamma(gamma: number): number {
  return clamp(gamma, 0.2, 5.0);
}

// Type guard: does this material have our update* methods?
export interface LuxarMaterial extends THREE.Material, CameraAwareMaterial {
  updateIntensity(v: number): void;
  updateOffset(v: number): void;
  updateGamma(v: number): void;
  updateOpacity(v: number): void;
  updateColormapTexture?(texture: THREE.DataTexture | null): void;
  updateScalarRange?(min: number, max: number): void;
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
  private visible = false;

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
    closeBtn.title = 'Close (L)';
    closeBtn.setAttribute('aria-label', 'Close layers panel');
    closeBtn.setAttribute('aria-keyshortcuts', 'l');
    closeBtn.addEventListener('click', () => this.hide());
    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    // Layer list (scrollable)
    const list = document.createElement('div');
    list.className = 'luxar-layers-panel__list';
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
    for (const layer of this.state.getLayers()) {
      const row = this.rowElements.get(layer.path);
      if (!row) continue;

      row.classList.toggle('luxar-layer-row--selected', layer.selected);
      row.classList.toggle('luxar-layer-row--hidden', !layer.visible);

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
  }

  private createLayerRow(layer: LayerInfo): HTMLElement {
    const row = document.createElement('div');
    row.className = 'luxar-layer-row';
    if (layer.selected) row.classList.add('luxar-layer-row--selected');
    if (!layer.visible) row.classList.add('luxar-layer-row--hidden');

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
    eyeBtn.addEventListener('click', (e) => {
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

    // Row click — selection
    row.addEventListener('click', (e) => {
      let mode: SelectionMode = 'single';
      if (e.ctrlKey || e.metaKey) mode = 'add';
      else if (e.shiftKey) mode = 'range';
      this.state.select(layer.path, mode);
    });

    row.appendChild(eyeBtn);
    row.appendChild(nameEl);
    row.appendChild(badge);
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
    this.blendSelect.addEventListener('change', () => {
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

    this.colormapSelect.addEventListener('change', () => {
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
   * Compute the layer's current live composable attributes. For layers
   * whose user hasn't touched a control, these match the authored zarr
   * values — so composition stays a no-op for untouched scenes.
   */
  private liveLayerAttrs(layer: LayerInfo): ComposableAttrs {
    const { intensity, offset } = computeUniforms(layer.displayMin, layer.displayMax);
    return {
      opacity: layer.opacity,
      gamma: clampGamma(layer.gamma),
      intensity,
      offset,
      blending_mode: layer.blendingMode as string,
    };
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
    switch (mode) {
      case 'additive':
        mat.blending = THREE.AdditiveBlending;
        mat.depthTest = false;
        mat.depthWrite = false;
        mat.transparent = true;
        break;
      case 'normal':
        mat.blending = THREE.NormalBlending;
        mat.depthTest = true;
        mat.depthWrite = false;
        mat.transparent = true;
        break;
      case 'max':
        mat.blending = THREE.CustomBlending;
        mat.blendEquation = THREE.MaxEquation;
        mat.depthTest = true;
        mat.depthWrite = false;
        mat.transparent = true;
        break;
      case 'opaque':
        mat.blending = THREE.NormalBlending;
        mat.depthTest = true;
        mat.depthWrite = true;
        mat.transparent = false;
        break;
      case 'luminous':
        mat.blending = THREE.AdditiveBlending;
        mat.depthTest = true;
        mat.depthWrite = false;
        mat.transparent = true;
        break;
    }
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
      mat.updateGamma(eff.gamma);
      mat.updateIntensity(eff.intensity);
      mat.updateOffset(eff.offset);
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
        mat.updateColormapTexture(tex);
        if (layer.scalarDataRange && mat.updateScalarRange) {
          mat.updateScalarRange(layer.scalarDataRange[0], layer.scalarDataRange[1]);
        }
      } else {
        mat.updateColormapTexture(null);
      }
      mat.needsUpdate = true;
    }
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
