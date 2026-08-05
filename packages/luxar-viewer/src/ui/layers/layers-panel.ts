/**
 * Layers Panel — napari-inspired per-layer control panel.
 *
 * Provides visibility toggle, [min, max] display range, gamma, and blending
 * mode controls for each scene node marked as `layer: true` in the zarr attrs.
 *
 * Multi-select: click = single, Ctrl+click = toggle, Shift+click = range.
 * Controls apply to all selected layers simultaneously.
 *
 * The panel owns the container/list DOM and lifecycle; two collaborators own
 * the rest (facade extraction, behavior-preserving):
 *
 * - {@link LayerControls} (`layer-controls.ts`) — the controls section
 *   (sliders, blend/colormap/LOD-level selects, live LOD readout).
 * - {@link LayerApplyEngine} (`layer-apply.ts`) — recomposes effective attrs
 *   per data-leaf and pushes them into the scene materials.
 *
 * The material contract (`LuxarMaterial` + the colormap-routing helpers)
 * lives in `luxar-material.ts` and is re-exported here for existing importers.
 */

import * as THREE from 'three';
import type { SceneNode } from '../../data/data-loader-types';
import type { FailedLoadsProviderPort } from '../../data/scene-loader-monitor-port';
import { LayerStateManager, type LayerInfo, type SelectionMode } from './layer-state';
import { config } from '../../config';
import { log, Modules } from '../../utils/log';
import { EventGroup } from '../../utils/cross-layer/event-group';
import { showToast } from '../toast';
import type { AnimationController } from '../../scene/animation/animation-controller';
import { LayerApplyEngine } from './layer-apply';
import { LayerControls } from './layer-controls';

export { applyColorAdjustments, isColormapActive, type LuxarMaterial } from './luxar-material';

/**
 * Visibility-toggle glyphs — stroke SVG in the rail-icon style (currentColor,
 * round caps), replacing the old eye emoji so the toggle themes with the
 * panel and reads crisply at small sizes.
 */
const EYE_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l18 18"/><path d="M10.6 5.2A11.3 11.3 0 0 1 12 5c6.5 0 10 7 10 7a17.6 17.6 0 0 1-3 3.9M6.5 6.5C3.6 8.4 2 12 2 12s3.5 7 10 7c1.4 0 2.7-.3 3.9-.7"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>';

/**
 * Per-row load-failure glyph — a warning triangle in the same stroke SVG
 * style as the eye toggle (currentColor, round caps), so a node whose loader
 * threw is visible in the always-open layers panel instead of only in the
 * collapsed data monitor / console.
 */
const ERROR_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';

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

  /**
   * Scene-application engine: recomposes + pushes attrs to materials.
   * Constructed with ACCESSORS for rootGroup/sceneGraph (both reassigned in
   * initFromScene), never captured values — see the stale-capture pitfall.
   */
  private applyEngine = new LayerApplyEngine({
    getRootGroup: () => this.rootGroup,
    getSceneGraph: () => this.sceneGraph,
    state: this.state,
    requestRender: () => this.requestRender(),
    invalidatePickBuffer: () => this.pickBufferInvalidator?.(),
  });

  /** The controls section (sliders/selects/LOD readout) below the list. */
  private controls = new LayerControls({
    state: this.state,
    apply: this.applyEngine,
    requestRender: () => this.requestRender(),
    isPanelVisible: () => this.visible,
  });

  private panelEl: HTMLElement | null = null;
  private listEl: HTMLElement | null = null;
  private visible = false;
  /**
   * Tracks every event listener attached during buildPanel/renderList
   * so clear()/dispose() can tear them all down with a single call.
   * Without this, listeners attached to detached DOM nodes hold
   * closures referencing the panel until the GC reclaims the
   * subtree — fragile, hard to test, and inconsistent with the rest
   * of the viewer's listener-tracking pattern. (The controls section's
   * listeners are tracked by LayerControls' own group.)
   */
  private events = new EventGroup();

  // Row elements keyed by layer path for targeted DOM updates
  private rowElements = new Map<string, HTMLElement>();

  /**
   * Failed-load provider (paths + per-path reason), injected by the app after
   * `initFromScene` (see `core/app/dataset/load-dataset.ts`). The SAME provider
   * the data monitor uses — reads the loader's live failure set. Null before a
   * scene loads and after dispose.
   */
  private failedLoadsProvider: FailedLoadsProviderPort | null = null;

  /**
   * Late-bound accessor to the current PickingSystem's `markDirty`, injected by
   * the app. Kept as a callback (not a captured PickingSystem) so it survives the
   * per-dataset picking re-creation. Null before wiring / in tests.
   */
  private pickBufferInvalidator: (() => void) | null = null;

  /**
   * Cheap change-detector for the failed set (JSON of sorted `[path, reason]`
   * pairs), mirroring DataMonitor's `lastFailedLoadsSignature`: the per-frame refresh
   * only touches the DOM when the signature changes. `null` is the reset
   * sentinel — no real signature (not even the empty-set `''`) can equal it, so
   * the first comparison after `setFailedLoadsProvider` / `renderList` always
   * falls through and re-applies (an empty set then correctly clears badges).
   */
  private lastFailedLoadsSignature: string | null = null;

  // State change unsubscribe handle
  private unsubscribeState: (() => void) | null = null;

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
      if (!this.controls.interacting) {
        this.controls.render();
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
      this.applyEngine.applyDisplayRange(layer);
      if (!layer.visible) {
        this.applyEngine.applyVisibility(layer.path, false);
      }
    }

    // Auto-select first layer
    if (layers.length > 0) {
      this.state.select(layers[0].path, 'single');
    }
  }

  /**
   * Reset every layer's parameters — visibility, display range, gamma,
   * opacity, blending mode, and colormap — back to their authored defaults.
   *
   * Re-derives the default state from the scene graph (the same walk
   * `initFromScene` uses) and pushes every parameter through the regular
   * apply paths, so the materials, the row list, and the controls all agree.
   * No-op before a scene loads or when the scene exposes no layers.
   */
  resetAllLayers(): void {
    if (!this.sceneGraph || this.state.count === 0) return;

    this.state.initFromSceneGraph(this.sceneGraph);
    const layers = this.state.getLayers();
    for (const layer of layers) {
      // Visibility applies unconditionally: a currently-hidden layer whose
      // authored default is visible must come back.
      this.applyEngine.applyVisibility(layer.path, layer.visible);
      // applyColormap restores the authored colormap (or none) and then
      // recomposes opacity/gamma/intensity/offset/blending via applyComposed.
      this.applyEngine.applyColormap(layer);
    }

    // Rebuild the row list + controls so the panel reflects the fresh state
    // (initFromSceneGraph replaced every LayerInfo the rows were bound to).
    if (this.panelEl) {
      this.renderList();
      this.controls.render();
    }
    if (layers.length > 0) this.state.select(layers[0].path, 'single');

    log.info(Modules.UI, `Layers reset to defaults (${layers.length} layer(s))`);
  }

  show(): void {
    if (!this.panelEl || this.state.count === 0) return;
    this.panelEl.style.display = 'flex';
    this.visible = true;
    this.repositionGUI();
    // The per-frame readout refresh is gated on `visible`, so while hidden
    // it does not track auto-LOD swaps. Refresh once on show so a level that
    // changed while hidden (or with the loop now idle) is reflected
    // immediately rather than only after the next swap.
    this.controls.refreshLodStatus();
    this.updateRowErrorStates();
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

  /**
   * Inject the shared failed-loads provider so per-row error badges can surface
   * a node whose loader threw (corrupt data / network failure). The app wires
   * the SAME provider the data monitor uses, after `initFromScene`. Resets the
   * change signature and applies the current failure set immediately so a scene
   * that already had failures at load lights up its rows without waiting for a
   * frame. Passing null (dispose) clears the provider and removes any badges
   * (the `null` reset sentinel forces the following pass through the gate even
   * when the resulting failed set is empty).
   */
  setFailedLoadsProvider(provider: FailedLoadsProviderPort | null): void {
    this.failedLoadsProvider = provider;
    this.lastFailedLoadsSignature = null;
    this.updateRowErrorStates();
  }

  /** Inject a callback that marks the GPU pick buffer dirty (PickingSystem.markDirty). Late-bound so it survives per-dataset picking re-creation. */
  setPickBufferInvalidator(fn: (() => void) | null): void {
    this.pickBufferInvalidator = fn;
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
    // with a fresh group rather than a disposed one. The controls
    // section tears down its own listeners + widgets symmetrically.
    this.events.dispose();
    this.events = new EventGroup();
    // Remove the live LOD-readout callback; buildPanel() re-registers it
    // on the next scene load. LayerControls.dispose() resets the cached
    // readout text so the fresh panel writes its first readout
    // unconditionally.
    this.animationController.removePerFrameCallback('layers-lod-status');
    this.sceneGraph = null;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    // Reset visibility and GUI position before removing the panel DOM
    const wasVisible = this.visible;
    this.visible = false;
    if (wasVisible) this.repositionGUI();
    this.controls.dispose();
    this.rowElements.clear();
    // Drop the failed-loads provider so a disposed panel holds no reference to
    // the old scene's loader; a subsequent load re-injects a fresh one.
    this.failedLoadsProvider = null;
    this.lastFailedLoadsSignature = null;
    this.panelEl?.remove();
    this.panelEl = null;
    this.listEl = null;
  }

  // ─── DOM Construction ──────────────────────────────────

  private buildPanel(): void {
    // Panel container
    const panel = document.createElement('div');
    panel.className = 'luxar-layers-panel luxar-glass-surface';
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
    closeBtn.textContent = '×';
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
    this.controls.build(controls);
    this.controls.render();

    // Keep the "Active level" readout live. The auto-LOD selector swaps
    // the active child per-frame as the camera moves (lod-group-registry
    // evaluatePerFrame), but the controls only re-render on layer-state
    // changes — so without this the readout went stale and disagreed
    // with the data-monitor chip. Non-continuous: it must not keep the
    // loop awake (no swaps happen while idle anyway), and the pipeline's
    // own 'lod-group-selector' callback is registered first, so by the
    // time this runs activeChildIndex is already updated for the frame.
    // Removed (and re-registered fresh) symmetrically in clear().
    this.animationController.addPerFrameCallback('layers-lod-status', () => {
      this.controls.refreshLodStatus();
      // Refresh per-row load-failure badges as the failed set changes. Gated on
      // visibility (like refreshLodStatus) so a hidden panel doesn't run the
      // signature build every frame — a real cost when a batch-fit leaves
      // thousands of failed tile paths. show() refreshes when the panel opens.
      if (this.visible) this.updateRowErrorStates();
    });
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

    // Rows were rebuilt badge-less. Invalidate the failed-set signature so the
    // next `updateRowErrorStates()` re-applies badges to the fresh rows instead
    // of early-returning on an unchanged signature (e.g. after resetAllLayers()
    // while a failure persists).
    this.lastFailedLoadsSignature = null;
    this.updateRowErrorStates();
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

      // Update eye button icon + ARIA state
      const eyeBtn = row.querySelector('.luxar-layer-row__eye') as HTMLButtonElement | null;
      if (eyeBtn) {
        eyeBtn.innerHTML = layer.visible ? EYE_ICON : EYE_OFF_ICON;
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

  /**
   * Patch each row's load-failure badge from the injected provider's failed set.
   * A layer is in error if its own path failed OR any descendant leaf failed
   * (`failedPath === layer.path || failedPath.startsWith(layer.path + '/')`), so
   * a failure inside a kind=lod/kind=partition group lights up the group's row.
   * Signature-gated so unchanged frames touch no DOM; the signature folds in
   * each path's reason (JSON of sorted `[path, reason]` pairs — unambiguous
   * even when a reason contains `:` or `|`) so a changed reason for a
   * still-failing path re-triggers the refresh instead of stranding a stale
   * tooltip.
   */
  private updateRowErrorStates(): void {
    const provider = this.failedLoadsProvider;
    const failedPaths = provider?.getFailedPaths() ?? [];
    const signature = JSON.stringify(
      [...failedPaths].sort().map((p) => [p, provider?.getFailedReason?.(p) ?? ''])
    );
    if (signature === this.lastFailedLoadsSignature) return;
    this.lastFailedLoadsSignature = signature;

    for (const layer of this.state.getLayers()) {
      const row = this.rowElements.get(layer.path);
      if (!row) continue;
      // Sorted so the reported reason (matches[0]) is deterministic rather than
      // dependent on the provider's Map-insertion order.
      const matches = failedPaths
        .filter((fp) => fp === layer.path || fp.startsWith(layer.path + '/'))
        .sort();
      this.applyRowError(row, matches);
    }
  }

  /**
   * Add / update / remove a single row's error badge + `--error` class. The
   * badge is a warning glyph with an accessible label naming the reason; the
   * row's own aria-label is left untouched so the base "name (type)" reading
   * is preserved.
   */
  private applyRowError(row: HTMLElement, matches: string[]): void {
    const inError = matches.length > 0;
    row.classList.toggle('luxar-layer-row--error', inError);

    let badge = row.querySelector('.luxar-layer-row__error') as HTMLElement | null;
    if (!inError) {
      badge?.remove();
      return;
    }

    const reason = this.describeFailure(matches);
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'luxar-layer-row__error';
      badge.setAttribute('role', 'img');
      badge.innerHTML = ERROR_ICON;
      row.appendChild(badge);
    }
    badge.title = reason;
    badge.setAttribute('aria-label', reason);
  }

  /**
   * Tooltip text for a failing row. Prefers the provider's per-path reason
   * (loader `error.message` / classified kind); falls back to a clear generic
   * message. Appends the descendant-failure count when more than one leaf under
   * the row failed.
   */
  private describeFailure(matches: string[]): string {
    const detail = this.failedLoadsProvider?.getFailedReason?.(matches[0]);
    const base = detail
      ? `Failed to load: ${detail}`
      : 'Failed to load — see the data monitor for details';
    return matches.length > 1 ? `${base} (${matches.length} parts failed)` : base;
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
    eyeBtn.innerHTML = layer.visible ? EYE_ICON : EYE_OFF_ICON;
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
      this.applyEngine.applyVisibility(layer.path, newVisible);
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

  // ─── GUI Repositioning ─────────────────────────────────

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
