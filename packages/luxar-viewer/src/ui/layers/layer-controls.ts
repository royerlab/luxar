/**
 * LayerControls — the controls section of the layers panel.
 *
 * Owns the per-layer control widgets (display-range + gamma + opacity
 * sliders, blend / colormap / LOD-level selects, and the live LOD readout)
 * that sit below the layer list, extracted from `layers-panel.ts`. The panel
 * hands it a container element via {@link build}, re-syncs it from state via
 * {@link render}, and drives the per-frame LOD readout via
 * {@link refreshLodStatus}; every user interaction is applied to the selected
 * layers through the injected {@link LayerApplyEngine}.
 *
 * While the user drags a control, {@link interacting} is true — the panel's
 * state-change subscription checks it to skip re-rendering the controls
 * (programmatic `.value=` writes would fight the drag).
 */

import type { BlendingMode } from '../../rendering';
import { type LayerInfo, type LayerStateManager } from './layer-state';
import { RangeSlider } from './range-slider';
import { LabeledSlider } from './labeled-slider';
import { log, Modules } from '../../utils/log';
import { EventGroup } from '../../utils/cross-layer/event-group';
import { BLENDING_MODES } from '../../rendering/blending-state';
import { COLORMAP_CATEGORIES } from '../../rendering/colormap-data';
import { SceneLoaderManager } from '../../data/scene-loader-manager';
import type { LODGroupRegistry } from '../../scene/lod-group-registry';
import { displayedQualityFraction } from '../../scene/lod-display-gate';
import { clampGamma } from './attrs-utils';
import { clamp } from '../gui/format/value-formatting';
import type { LayerApplyEngine } from './layer-apply';

/**
 * Dependencies injected by the owning {@link LayersPanel}. `state` and
 * `apply` are the panel's (stable) layer-state manager and material-apply
 * engine; `requestRender` wakes the on-demand render loop after a selector
 * change; `isPanelVisible` gates the per-frame LOD readout (an accessor —
 * panel visibility toggles at runtime).
 */
export interface LayerControlsDeps {
  state: LayerStateManager;
  apply: LayerApplyEngine;
  requestRender: () => void;
  isPanelVisible: () => boolean;
}

export class LayerControls {
  private controlsEl: HTMLElement | null = null;
  private rangeSlider: RangeSlider | null = null;
  private gammaSlider: LabeledSlider | null = null;
  private opacitySlider: LabeledSlider | null = null;
  private absorptionSlider: LabeledSlider | null = null;
  private blendSelect: HTMLSelectElement | null = null;
  private colormapSelect: HTMLSelectElement | null = null;
  /**
   * "Active level" dropdown for ``lod_group`` layers. Shown only when
   * the primary selected layer is an lod_group; hidden otherwise.
   * Options: ``auto`` plus one ``lock to level <n>`` entry per child
   * (1-based label; the option value stays 0-based for the registry's
   * ``lockLevel`` API).
   */
  private lodLevelSelect: HTMLSelectElement | null = null;
  /**
   * Status span next to the dropdown showing the currently-rendering
   * level (e.g. "L3/5", 1-based to match the data-monitor chip). Kept
   * live by a per-frame callback (``layers-lod-status``) registered by
   * the panel in buildPanel(), so it tracks auto-selection swaps driven
   * by camera motion — not only ``render()`` state changes.
   */
  private lodLevelStatus: HTMLSpanElement | null = null;
  /**
   * Last text written to {@link lodLevelStatus}. The per-frame callback
   * compares against this and only touches the DOM when the readout
   * actually changes, so a static scene costs a string compare per
   * frame rather than a DOM write. Reset in dispose().
   */
  private lastShownLodStatus: string | null = null;

  // Suppresses render() during user-driven control interactions
  // to prevent programmatic .value= from fighting with the user's drag
  private controlsInteracting = false;

  /**
   * Tracks every event listener attached during build() so dispose()
   * can tear them all down with a single call (the panel-wide
   * listener-tracking pattern).
   */
  private events = new EventGroup();

  constructor(private deps: LayerControlsDeps) {}

  /**
   * True while a user-driven control interaction is applying state — the
   * panel's state-change subscription skips control re-renders meanwhile.
   */
  get interacting(): boolean {
    return this.controlsInteracting;
  }

  /**
   * Tear down the widgets + listeners so a subsequent build() starts
   * fresh. Mirrors the panel's clear(): the EventGroup is re-instantiated
   * (not left disposed) and the cached LOD readout text is reset so the
   * fresh panel writes its first readout unconditionally.
   */
  dispose(): void {
    this.events.dispose();
    this.events = new EventGroup();
    this.lastShownLodStatus = null;
    this.rangeSlider?.dispose();
    this.rangeSlider = null;
    this.gammaSlider?.dispose();
    this.gammaSlider = null;
    this.opacitySlider?.dispose();
    this.opacitySlider = null;
    this.absorptionSlider?.dispose();
    this.absorptionSlider = null;
    this.blendSelect = null;
    this.colormapSelect = null;
    this.lodLevelSelect = null;
    this.lodLevelStatus = null;
    this.controlsEl = null;
  }

  /** Build the control widgets into `container` (the panel's controls div). */
  build(container: HTMLElement): void {
    this.controlsEl = container;
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
        this.deps.state.applyToSelected((l) => {
          l.displayMin = low;
          l.displayMax = high;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyDisplayRange(sel);
        }
        this.controlsInteracting = false;
      },
      onBoundsChange: (min, max) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
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
        this.deps.state.applyToSelected((l) => {
          l.gamma = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyGamma(sel);
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
        this.deps.state.applyToSelected((l) => {
          l.opacity = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyOpacity(sel);
        }
        this.controlsInteracting = false;
      },
    });

    // Absorption κ — only meaningful in volumetric mode; hidden for every
    // other mode (see syncAbsorptionVisibility). Range 0–10 covers the
    // useful span (the attr itself is unbounded); κ=0 looks additive.
    this.absorptionSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Absorption',
      min: 0,
      max: 10,
      step: 0.05,
      initialValue: 1.0,
      constrain: (v) => Math.max(0, v),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
          l.absorption = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyAbsorption(sel);
        }
        this.controlsInteracting = false;
      },
    });
    this.absorptionSlider.setVisible(false);

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
      this.deps.state.applyToSelected((l) => {
        l.blendingMode = mode;
      });
      for (const sel of this.deps.state.getSelected()) {
        this.deps.apply.applyBlendingMode(sel);
      }
      // Switching to/from volumetric must reveal/hide the κ slider
      // immediately, not on the next selection refresh.
      this.syncAbsorptionVisibility();
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
      this.deps.state.applyToSelected((l) => {
        l.colormap = cmName;
      });
      for (const sel of this.deps.state.getSelected()) {
        this.deps.apply.applyColormap(sel);
      }
      this.controlsInteracting = false;
    });
    cmGroup.appendChild(cmLabel);
    cmGroup.appendChild(this.colormapSelect);
    this.controlsEl.appendChild(cmGroup);

    // Active-level selector — only meaningful for lod_group layers,
    // hidden otherwise (see render). The dropdown's option
    // list is rebuilt per layer in render() because child
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
      const primary = this.deps.state.getPrimarySelected();
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
          if (anyApplied) this.deps.requestRender();
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
      // Value stays 0-based (the registry's lockLevel API), but the
      // label is 1-based to match the readout ("L3/5") and the
      // data-monitor chip — so all three LOD surfaces agree numerically.
      opt.value = String(i);
      opt.textContent = `lock to level ${i + 1}`;
      this.lodLevelSelect.appendChild(opt);
    }
  }

  /** Update controls to reflect the primary selected layer's values */
  render(): void {
    const primary = this.deps.state.getPrimarySelected();
    if (!primary) return;

    if (this.rangeSlider) {
      this.rangeSlider.setBounds(primary.dataMin, primary.dataMax);
      this.rangeSlider.setValues(primary.displayMin, primary.displayMax);
    }

    this.gammaSlider?.setValue(primary.gamma);
    this.opacitySlider?.setValue(primary.opacity);
    this.absorptionSlider?.setValue(primary.absorption);

    if (this.blendSelect) {
      this.blendSelect.value = primary.blendingMode;
    }
    this.syncAbsorptionVisibility();

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
      const registry = this.getLodGroupRegistry();
      if (primary.kind === 'lod' && (primary.lodGroupChildCount ?? 0) > 0) {
        lodContainer.style.display = '';
        this.renderLodLevelOptions(primary.lodGroupChildCount!);
        // Sync the dropdown to the registry's selector mode. No entry
        // yet (scene still loading) → default to "auto".
        const entry = registry?.get(primary.path);
        this.lodLevelSelect.value =
          entry && entry.selectorMode !== 'auto' ? String(entry.selectorMode.lockLevel) : 'auto';
      } else if (this.isBroadcastPartition(primary)) {
        lodContainer.style.display = '';
        this.renderLodLevelOptions(primary.nestedLodMaxChildCount!);
        // Sync widget state from the FIRST nested entry — they should
        // be lock-stepped after a broadcast change, and pre-broadcast
        // divergence (rare: legacy authored values) is acceptable
        // ambiguity here.
        const entry = registry?.get(primary.nestedLodGroupPaths![0]);
        this.lodLevelSelect.value =
          entry && entry.selectorMode !== 'auto' ? String(entry.selectorMode.lockLevel) : 'auto';
      } else {
        lodContainer.style.display = 'none';
      }
      // Single source of truth for the readout text, shared with the
      // per-frame refreshLodStatus() so both always agree. null → no
      // readout applies (non-LOD layer, or registry not yet populated).
      this.setLodStatusText(this.computeLodStatusText(primary) ?? '');
    }
  }

  /**
   * Show the Absorption (κ) slider only when it can do something: the
   * primary selection's mode is `volumetric` AND the layer is (or can
   * contain) gsplats — points/lines render volumetric's additive
   * fallback in phase 1, where κ is inert, so showing a dead slider
   * would mislead. Called from render() and the blend-dropdown change
   * handler (mode switches must reveal/hide it immediately).
   */
  private syncAbsorptionVisibility(): void {
    if (!this.absorptionSlider) return;
    const primary = this.deps.state.getPrimarySelected();
    const show =
      !!primary &&
      primary.blendingMode === 'volumetric' &&
      (primary.type === 'gsplats' || primary.type === 'group');
    this.absorptionSlider.setVisible(show);
  }

  /**
   * True when ``primary`` is a kind=partition layer wrapping one or more
   * nested lod_groups, so the "Active level" dropdown broadcasts to them.
   */
  private isBroadcastPartition(primary: LayerInfo): boolean {
    return (
      primary.kind === 'partition' &&
      primary.nestedLodGroupPaths != null &&
      primary.nestedLodGroupPaths.length > 0 &&
      (primary.nestedLodMaxChildCount ?? 0) > 0
    );
  }

  /**
   * Compute the "Active level" readout text for the primary-selected
   * layer, or ``null`` when no LOD readout applies (non-LOD layer, or the
   * lod_group registry isn't populated yet). 1-based ("L3/5") to match
   * the data-monitor chip (data-loading-monitor/templates.ts) and the
   * dropdown labels. Reads the live active level straight from the
   * registry, so it is correct on any frame — including auto-selection
   * swaps driven by camera motion.
   */
  private computeLodStatusText(primary: LayerInfo): string | null {
    const registry = this.getLodGroupRegistry();
    if (!registry) return null;
    if (primary.kind === 'lod' && (primary.lodGroupChildCount ?? 0) > 0) {
      const entry = registry.get(primary.path);
      if (!entry || entry.children.length === 0) return null;
      // The off-screen gate holds the group at its coarsest level while it
      // is outside the frustum; flag it so a coarse level isn't read as a
      // selection bug.
      const suffix = entry.offScreen ? ' (off-screen)' : '';
      // Show the level on SCREEN (``displayedChildIndex``), not the selector's
      // aspiration — during a slice scrub the displayed level is a coarser fresh
      // one while ``activeChildIndex`` is the stale fine level reloading, and
      // during a never-downgrade hold it is the better previously-shown level
      // while the aspiration's additive ladder catches up.
      const shown = entry.displayedChildIndex ?? entry.activeChildIndex;
      // Displayed-quality estimate q = Q·e from the commit-time quality
      // stamps (didactic: how close what is ON SCREEN is to the group's
      // finest content — Q the level's measured complete quality, e the
      // committed energy fraction of its streaming ladder). Absent on
      // unstamped (legacy) datasets.
      const q = displayedQualityFraction(entry.children[shown]?.object ?? {});
      const qualityStr = q == null ? '' : ` · ~${Math.round(q * 100)}%`;
      return `L${shown + 1}/${entry.children.length}${qualityStr}${suffix}`;
    }
    if (this.isBroadcastPartition(primary)) {
      // Aggregate across EVERY nested lod_group, not just the first: under
      // auto-selection each part picks its own level by its own on-screen
      // size, so they legitimately diverge (the mosaic recipe is unbalanced
      // by design). Show a range when they do, and use the dropdown's
      // max-ladder depth (nestedLodMaxChildCount) as the denominator so the
      // readout and the option list agree on {n}.
      const paths = primary.nestedLodGroupPaths!;
      let min = Infinity;
      let max = -Infinity;
      for (const p of paths) {
        const e = registry.get(p);
        if (!e || e.children.length === 0) continue;
        const lvl = (e.displayedChildIndex ?? e.activeChildIndex) + 1;
        if (lvl < min) min = lvl;
        if (lvl > max) max = lvl;
      }
      if (max < 0) return null; // no populated nested group yet
      const n = primary.nestedLodMaxChildCount!;
      const levelStr = min === max ? `L${min}/${n}` : `L${min}–${max}/${n}`;
      return `${levelStr} · ${paths.length} groups`;
    }
    return null;
  }

  /**
   * Write the LOD readout, skipping the DOM touch when the text is
   * unchanged. Lets the per-frame callback run every frame at the cost of
   * a string compare on a static scene rather than a DOM write.
   */
  private setLodStatusText(text: string): void {
    if (!this.lodLevelStatus || text === this.lastShownLodStatus) return;
    this.lodLevelStatus.textContent = text;
    this.lastShownLodStatus = text;
  }

  /**
   * Per-frame: keep the LOD readout in sync with the live active level
   * chosen by the auto-selector. Cheap — early-returns when the panel is
   * hidden or the primary layer has no LOD readout, and setLodStatusText
   * skips the DOM write unless the text actually changed. A ``null`` text
   * (non-LOD layer / registry not ready) leaves whatever render()
   * last set in place rather than clobbering it.
   */
  refreshLodStatus(): void {
    if (!this.deps.isPanelVisible() || !this.lodLevelStatus) return;
    const primary = this.deps.state.getPrimarySelected();
    if (!primary) return;
    const text = this.computeLodStatusText(primary);
    if (text !== null) this.setLodStatusText(text);
  }
}
