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
import {
  LayerStateManager,
  computeUniforms,
  type LayerInfo,
  type SelectionMode,
} from './layer-state';
import { RangeSlider } from './range-slider';
import { config } from '../../config';
import { materialManager } from '../../rendering/material-manager';
import { log, Modules } from '../../utils/log';
import { showToast } from '../helpers';
import type { AnimationController } from '../../scene/animation-controller';

/** Clamp a gamma value to a safe range for the shader (prevents division by zero and extreme exponents) */
function clampGamma(gamma: number): number {
  return Math.max(0.2, Math.min(5.0, gamma));
}

// Type guard: does this material have our update* methods?
interface LuxarMaterial extends THREE.Material {
  updateIntensity(v: number): void;
  updateOffset(v: number): void;
  updateGamma(v: number): void;
  updateOpacity(v: number): void;
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
  private animationController: AnimationController;

  private state = new LayerStateManager();
  private panelEl: HTMLElement | null = null;
  private listEl: HTMLElement | null = null;
  private controlsEl: HTMLElement | null = null;
  private rangeSlider: RangeSlider | null = null;
  private gammaSlider: HTMLInputElement | null = null;
  private gammaValueEl: HTMLElement | null = null;
  private blendSelect: HTMLSelectElement | null = null;
  private visible = false;

  // Row elements keyed by layer path for targeted DOM updates
  private rowElements = new Map<string, HTMLElement>();

  // State change unsubscribe handle
  private unsubscribeState: (() => void) | null = null;

  // Suppresses renderControls() during user-driven control interactions
  // to prevent programmatic .value= from fighting with the user's drag
  private controlsInteracting = false;

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

    // Auto-select first layer
    const layers = this.state.getLayers();
    if (layers.length > 0) {
      this.state.select(layers[0].path, 'single');
    }
  }

  show(): void {
    if (!this.panelEl || this.state.count === 0) return;
    this.panelEl.style.display = 'flex';
    this.visible = true;
  }

  hide(): void {
    if (!this.panelEl) return;
    this.panelEl.style.display = 'none';
    this.visible = false;
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
    this.rangeSlider?.dispose();
    this.rangeSlider = null;
    this.gammaSlider = null;
    this.gammaValueEl = null;
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

      // Update eye button text
      const eyeBtn = row.querySelector('.luxar-layer-row__eye') as HTMLButtonElement | null;
      if (eyeBtn) {
        eyeBtn.textContent = layer.visible ? '\u{1F441}' : '\u{1F441}\u200D\u{1F5E8}';
        eyeBtn.title = layer.visible ? 'Hide layer' : 'Show layer';
      }
    }
  }

  private createLayerRow(layer: LayerInfo): HTMLElement {
    const row = document.createElement('div');
    row.className = 'luxar-layer-row';
    if (layer.selected) row.classList.add('luxar-layer-row--selected');
    if (!layer.visible) row.classList.add('luxar-layer-row--hidden');

    // Eye toggle — visibility is independent of selection
    const eyeBtn = document.createElement('button');
    eyeBtn.className = 'luxar-layer-row__eye';
    eyeBtn.textContent = layer.visible ? '\u{1F441}' : '\u{1F441}\u200D\u{1F5E8}';
    eyeBtn.title = layer.visible ? 'Hide layer' : 'Show layer';
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
    });

    // Gamma
    const gammaGroup = document.createElement('div');
    gammaGroup.className = 'luxar-layers-panel__control-group';
    const gammaLabel = document.createElement('div');
    gammaLabel.className = 'luxar-layers-panel__control-label';

    const gammaText = document.createElement('span');
    gammaText.textContent = 'Gamma';
    this.gammaValueEl = document.createElement('span');
    this.gammaValueEl.className = 'luxar-layers-panel__control-value';
    gammaLabel.appendChild(gammaText);
    gammaLabel.appendChild(this.gammaValueEl);

    this.gammaSlider = document.createElement('input');
    this.gammaSlider.type = 'range';
    this.gammaSlider.min = '0.2';
    this.gammaSlider.max = '5.0';
    this.gammaSlider.step = '0.01';
    this.gammaSlider.className = 'luxar-layers-panel__slider';
    this.gammaSlider.addEventListener('input', () => {
      this.controlsInteracting = true;
      const val = clampGamma(parseFloat(this.gammaSlider!.value));
      this.gammaValueEl!.textContent = val.toFixed(2);
      this.state.applyToSelected((l) => {
        l.gamma = val;
      });
      for (const sel of this.state.getSelected()) {
        this.applyGamma(sel);
      }
      this.controlsInteracting = false;
    });
    gammaGroup.appendChild(gammaLabel);
    gammaGroup.appendChild(this.gammaSlider);
    this.controlsEl.appendChild(gammaGroup);

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
  }

  /** Update controls to reflect the primary selected layer's values */
  private renderControls(): void {
    const primary = this.state.getPrimarySelected();
    if (!primary) return;

    if (this.rangeSlider) {
      this.rangeSlider.setBounds(primary.dataMin, primary.dataMax);
      this.rangeSlider.setValues(primary.displayMin, primary.displayMax);
    }

    if (this.gammaSlider) {
      this.gammaSlider.value = String(primary.gamma);
    }
    if (this.gammaValueEl) {
      this.gammaValueEl.textContent = primary.gamma.toFixed(2);
    }

    if (this.blendSelect) {
      this.blendSelect.value = primary.blendingMode;
    }
  }

  // ─── Scene Application ─────────────────────────────────

  private getMesh(path: string): THREE.Object3D | null {
    if (!this.rootGroup) return null;
    return this.rootGroup.getObjectByName(path) ?? null;
  }

  private getMaterial(obj: THREE.Object3D): LuxarMaterial | null {
    const mesh = obj as THREE.Points | THREE.Mesh;
    if (!mesh.material) return null;

    const mat = mesh.material as THREE.Material;

    // Clone on first use to avoid mutating shared cached materials.
    // Register the clone with MaterialManager so camera-dependent uniforms
    // (pointSizeFactor, maxPointSize, uIsOrtho) continue to be updated.
    if (!mesh.userData._layerMaterialCloned && isLuxarMaterial(mat)) {
      const cloned = mat.clone() as LuxarMaterial;
      mesh.material = cloned;
      mesh.userData._layerMaterialCloned = true;

      // Register so MaterialManager.updateCameraParams() keeps this material current
      materialManager.register(cloned);

      return cloned;
    }

    return isLuxarMaterial(mat) ? (mat as LuxarMaterial) : null;
  }

  private applyVisibility(path: string, visible: boolean): void {
    const obj = this.getMesh(path);
    if (obj) {
      obj.visible = visible;
      this.requestRender();
    }
  }

  private applyDisplayRange(layer: LayerInfo): void {
    const obj = this.getMesh(layer.path);
    if (!obj) return;
    const mat = this.getMaterial(obj);
    if (!mat) return;

    const { intensity, offset } = computeUniforms(layer.displayMin, layer.displayMax);
    mat.updateIntensity(intensity);
    mat.updateOffset(offset);
    this.requestRender();
  }

  private applyGamma(layer: LayerInfo): void {
    const obj = this.getMesh(layer.path);
    if (!obj) return;
    const mat = this.getMaterial(obj);
    if (!mat) return;

    const safeGamma = clampGamma(layer.gamma);
    mat.updateGamma(safeGamma);
    this.requestRender();
  }

  private applyBlendingMode(layer: LayerInfo): void {
    const obj = this.getMesh(layer.path);
    if (!obj) return;
    const mat = this.getMaterial(obj);
    if (!mat) return;

    // Directly mutate THREE.js blending state
    switch (layer.blendingMode) {
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
    this.requestRender();
  }

  private requestRender(): void {
    this.animationController.startAnimation();
  }
}
