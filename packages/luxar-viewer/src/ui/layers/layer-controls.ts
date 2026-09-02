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
import { resolveLayerBlendingMode, type LayerInfo, type LayerStateManager } from './layer-state';
import { RangeSlider } from './range-slider';
import { LabeledSlider } from './labeled-slider';
import {
  ABSORPTION_DEFAULT_MAX,
  ABSORPTION_LOG_DECADES,
  absorptionSliderRange,
  formatAbsorption,
} from './absorption-range';
import { log, Modules } from '../../utils/log';
import { EventGroup } from '../../utils/cross-layer/event-group';
import { BLENDING_MODES } from '../../rendering/blending-state';
import { COLORMAP_CATEGORIES } from '../../rendering/colormap-data';
import { SceneLoaderManager } from '../../data/scene-loader-manager';
import type { LODGroupRegistry } from '../../scene/lod-group-registry';
import { displayedQualityFraction } from '../../scene/lod-display-gate';
import { MESH_DEFAULTS } from '../../rendering/materials/mesh/appearance';
import { clampGamma } from './attrs-utils';
import { clamp } from '../gui/format/value-formatting';
import type { LayerApplyEngine } from './layer-apply';

const SCALAR_RANGE_LABEL = 'Display range';
const COLOUR_RANGE_LABEL = 'Colour range';
const SCALAR_RANGE_TOOLTIP = 'Scalar data values in this range are mapped across the colormap.';
const COLOUR_RANGE_TOOLTIP =
  "Input RGB values in this range are mapped to the full output range. This controls colour gain and offset, not the layer's data extents.";

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
  /**
   * Mesh-only appearance sliders (spec §6.2). Hidden for every other geometry type —
   * mesh is the only SHADED type, so a shade floor and a falloff exponent have nothing
   * to act on elsewhere. `alphaCutoff` is gated more narrowly still: only in `opaque`,
   * the one mode that applies the cutout.
   */
  private ambientSlider: LabeledSlider | null = null;
  private shadeExponentSlider: LabeledSlider | null = null;
  private specularSlider: LabeledSlider | null = null;
  private shininessSlider: LabeledSlider | null = null;
  private alphaCutoffSlider: LabeledSlider | null = null;
  private blendSelect: HTMLSelectElement | null = null;
  private layerOrderInput: HTMLInputElement | null = null;
  private colormapSelect: HTMLSelectElement | null = null;
  private labelColorSelect: HTMLSelectElement | null = null;
  private labelFilterSelect: HTMLSelectElement | null = null;
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
    this.ambientSlider?.dispose();
    this.ambientSlider = null;
    this.shadeExponentSlider?.dispose();
    this.shadeExponentSlider = null;
    this.specularSlider?.dispose();
    this.specularSlider = null;
    this.shininessSlider?.dispose();
    this.shininessSlider = null;
    this.alphaCutoffSlider?.dispose();
    this.alphaCutoffSlider = null;
    this.blendSelect = null;
    this.layerOrderInput = null;
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
      label: SCALAR_RANGE_LABEL,
      tooltip: SCALAR_RANGE_TOOLTIP,
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
    // other mode (see syncAbsorptionVisibility). τ = κ · rayMass with the
    // same normalised ray mass in all three geometry families, so κ ≈ 1 is
    // the useful anchor everywhere and one fixed LOG track serves every
    // scene (absorption-range.ts widens it only for an out-of-range authored
    // κ). Position 0 on a log track is an exact κ=0 — the additive limit.
    this.absorptionSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Absorption',
      min: ABSORPTION_DEFAULT_MAX / Math.pow(10, ABSORPTION_LOG_DECADES),
      max: ABSORPTION_DEFAULT_MAX,
      step: 0.05,
      scale: 'log',
      format: formatAbsorption,
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

    // --- Mesh shading (§6.2) ------------------------------------------------
    //
    // Five mesh-only sliders (see syncMeshAppearanceVisibility). The four lighting
    // controls also require resolved shading other than `none`; Alpha cutoff has its
    // own narrower opaque-mode gate.
    //
    // Linear tracks, unlike absorption's log one: the bounded fractions and small
    // exponents have meaningful midpoints, rather than being scale-free coefficients
    // spanning decades.
    this.ambientSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Ambient',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: MESH_DEFAULTS.ambient,
      // The clamp the material would apply anyway, applied here so the READOUT cannot
      // show a value the surface is not using.
      constrain: (v) => Math.min(1, Math.max(0, v)),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
          l.ambient = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyMeshAppearance(sel);
        }
        this.controlsInteracting = false;
      },
    });
    this.ambientSlider.setVisible(false);

    this.shadeExponentSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Shade falloff',
      // The floor is the material's own clamp, not 0: `pow(wrap, 0)` is undefined in
      // GLSL at a face-away fragment, where `wrap` is exactly 0. Starting the track at
      // the clamp means the slider cannot ask for a value the material must refuse.
      min: 0.001,
      max: 4,
      step: 0.05,
      initialValue: MESH_DEFAULTS.shadeExponent,
      constrain: (v) => Math.max(0.001, v),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
          l.shadeExponent = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyMeshAppearance(sel);
        }
        this.controlsInteracting = false;
      },
    });
    this.shadeExponentSlider.setVisible(false);

    this.specularSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Specular',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: MESH_DEFAULTS.specular,
      constrain: (v) => Math.min(1, Math.max(0, v)),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
          l.specular = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyMeshAppearance(sel);
        }
        this.controlsInteracting = false;
      },
    });
    this.specularSlider.setVisible(false);

    this.shininessSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Shininess',
      min: 0.001,
      max: 128,
      step: 0.5,
      initialValue: MESH_DEFAULTS.shininess,
      constrain: (v) => Math.max(0.001, v),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
          l.shininess = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyMeshAppearance(sel);
        }
        this.controlsInteracting = false;
      },
    });
    this.shininessSlider.setVisible(false);

    this.alphaCutoffSlider = new LabeledSlider({
      container: this.controlsEl,
      label: 'Alpha cutoff',
      min: 0,
      max: 1,
      step: 0.01,
      initialValue: MESH_DEFAULTS.alphaCutoff,
      constrain: (v) => Math.min(1, Math.max(0, v)),
      onChange: (val) => {
        this.controlsInteracting = true;
        this.deps.state.applyToSelected((l) => {
          l.alphaCutoff = val;
        });
        for (const sel of this.deps.state.getSelected()) {
          this.deps.apply.applyMeshAppearance(sel);
        }
        this.controlsInteracting = false;
      },
    });
    this.alphaCutoffSlider.setVisible(false);

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
        // Resolved PER LAYER, not once for the whole selection: a multi-select can mix
        // types, and only mesh maps `volumetric` away. Storing the raw pick would make
        // the panel claim a mode the mesh shader does not implement — see
        // `resolveLayerBlendingMode`.
        l.blendingMode = resolveLayerBlendingMode(l.type, mode);
        // A user pick is EXPLICIT — the layer now OWNS a mode, so `liveLayerAttrs`
        // emits it as a composition setter and `composeEffective` lets it win over the
        // subtree (matters for a GROUP layer, which otherwise owns no mode and would
        // drop the pick; see #1272/#1275). Still explicit even when the resolution
        // changed the value: the user DID choose, and the choice was honoured as far
        // as the surface can express it.
        l.blendingModeExplicit = true;
      });
      for (const sel of this.deps.state.getSelected()) {
        this.deps.apply.applyBlendingMode(sel);
      }
      // Switching to/from volumetric must reveal/hide the κ slider
      // immediately, not on the next selection refresh.
      this.syncAbsorptionVisibility();
      // Same for the cutoff slider, whose gate includes the mode: leaving `opaque`
      // must hide it on the click, not on the next selection change.
      this.syncMeshAppearanceVisibility();
      this.controlsInteracting = false;
    });
    blendGroup.appendChild(blendLabel);
    blendGroup.appendChild(this.blendSelect);
    this.controlsEl.appendChild(blendGroup);

    // Layer order — the authored cross-layer draw order
    // (`LAYER_ORDER_SPEC.md`). A number input rather than a slider,
    // because the value is an unbounded signed integer AND must be able to be
    // BLANK: empty means "unset", which hands the layer back to the renderer's
    // inferred containment ordering and is a genuinely different state from 0.
    const orderGroup = document.createElement('div');
    orderGroup.className = 'luxar-layers-panel__control-group';
    const orderLabel = document.createElement('div');
    orderLabel.className = 'luxar-layers-panel__control-label';
    orderLabel.textContent = 'Layer order';

    this.layerOrderInput = document.createElement('input');
    this.layerOrderInput.type = 'number';
    this.layerOrderInput.step = '1';
    this.layerOrderInput.className = 'luxar-layers-panel__number';
    this.layerOrderInput.placeholder = 'auto';
    // The visible label is a sibling `div`, matching every other control here,
    // so nothing associates it with the field. A `select` at least announces
    // its selected option; a bare number input announces nothing, so give it a
    // name of its own.
    this.layerOrderInput.setAttribute('aria-label', 'Layer order');
    this.layerOrderInput.title =
      'Draw order against the layers this one overlaps. Higher draws nearer the ' +
      'camera (on top), like a CSS z-index. Leave blank to let the viewer infer ' +
      'the order from the geometry. Sparse values (10/20/30) leave room to insert.';
    this.events.on(this.layerOrderInput, 'change', () => {
      this.controlsInteracting = true;
      const raw = this.layerOrderInput!.value.trim();
      // Blank CLEARS. A non-numeric entry is treated as blank rather than as 0,
      // since 0 is a real band and guessing it from junk would state an order
      // the user did not choose.
      const parsed = raw === '' ? undefined : Number.parseInt(raw, 10);
      const level = parsed === undefined || !Number.isFinite(parsed) ? undefined : parsed;
      this.deps.state.applyToSelected((l) => {
        l.layerOrder = level;
        l.layerOrderExplicit = level !== undefined;
      });
      for (const sel of this.deps.state.getSelected()) {
        this.deps.apply.applyLayerOrder(sel);
      }
      // Echo back what was actually stored, so junk input does not sit in the
      // field looking authoritative.
      this.layerOrderInput!.value = level === undefined ? '' : String(level);
      this.controlsInteracting = false;
    });
    orderGroup.appendChild(orderLabel);
    orderGroup.appendChild(this.layerOrderInput);
    this.controlsEl.appendChild(orderGroup);

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
      // Capture each layer's mode BEFORE mutating: the window re-default
      // below applies only to an off→on / on→off flip. Switching between two
      // active palettes keeps the user's scalar window — the rendered value
      // is the same scalar either side. Key on `scalarWindow` (the effective
      // mode), NOT on the layer's own `colormap` attr: a group layer whose
      // colormap lives on a DESCENDANT has no palette of its own but already
      // windows a scalar, so the attr would misread a palette change as
      // off→on (wiping the user's window) and miss the on→off flip entirely
      // (stranding the layer on an inert scalar window).
      const wasColormapped = new Map(
        this.deps.state.getSelected().map((l) => [l.path, l.scalarWindow])
      );
      this.deps.state.applyToSelected((l) => {
        l.colormap = cmName;
      });
      for (const sel of this.deps.state.getSelected()) {
        // The display window means a different thing on each side of the
        // off↔on toggle (scalar data range vs authored-RGB identity), so
        // re-default it BEFORE applying — `applyColormap` derives the
        // material's scalar range from the composed window.
        if (wasColormapped.get(sel.path) !== !!cmName) {
          this.deps.state.setColormapWindow(sel.path, !!cmName);
        }
        if (!this.deps.apply.applyColormap(sel) && cmName) {
          // The C1 fail-closed guard suppressed the colormap on every leaf
          // (e.g. an explicitly colormapped group over scalar-less points).
          // The layer keeps rendering DIRECT COLOUR, so the
          // scalar window would be applied as a colour gain — put the identity
          // window back and re-push the corrected GOG. Drop the rejected
          // palette too: keeping it would leave contradictory state (`colormap`
          // set, `scalarWindow` false) that lies to the dropdown and the
          // legend, and mis-keys the next toggle's off→on detection.
          sel.colormap = undefined;
          this.deps.state.setColormapWindow(sel.path, false);
          this.deps.apply.applyColormap(sel);
        }
      }
      this.controlsInteracting = false;
      // Re-sync the widgets this handler just invalidated. `setColormapWindow`
      // moved the display window (and widened the bounds), but
      // `controlsInteracting` suppressed the state-change re-render — and
      // RangeSlider emits values parsed from its own <input> elements, which
      // only `render()` updates. Without this the thumbs keep the OLD window
      // and the first drag writes it back, silently reverting the re-default:
      // toggling a colormap on would re-apply a [0, 1] window to amplitudes
      // (the #522 near-black), and off would re-apply the scalar range as a
      // colour gain (the contrast stretch this campaign removed). The blend
      // handler does the same for the κ slider via syncAbsorptionVisibility().
      this.render();
    });
    cmGroup.appendChild(cmLabel);
    cmGroup.appendChild(this.colormapSelect);
    this.controlsEl.appendChild(cmGroup);

    const labelGroup = document.createElement('div');
    labelGroup.className = 'luxar-layers-panel__control-group';
    const labelTitle = document.createElement('div');
    labelTitle.className = 'luxar-layers-panel__control-label';
    labelTitle.textContent = 'Classes';
    this.labelColorSelect = document.createElement('select');
    this.labelColorSelect.className = 'luxar-layers-panel__select';
    this.labelColorSelect.append(new Option('authored colors', 'authored'));
    this.labelColorSelect.append(new Option('color by class', 'categorical'));
    this.labelFilterSelect = document.createElement('select');
    this.labelFilterSelect.className = 'luxar-layers-panel__select';
    this.events.on(this.labelColorSelect, 'change', () => {
      const categorical = this.labelColorSelect!.value === 'categorical';
      this.deps.state.applyToSelected((layer) => {
        layer.colorByLabel = categorical;
      });
      for (const layer of this.deps.state.getSelected()) this.deps.apply.applyLabelStyle(layer);
    });
    this.events.on(this.labelFilterSelect, 'change', () => {
      const id = this.labelFilterSelect!.value || undefined;
      this.deps.state.applyToSelected((layer) => {
        layer.labelFilterId = id;
      });
      for (const layer of this.deps.state.getSelected()) this.deps.apply.applyLabelStyle(layer);
    });
    labelGroup.append(labelTitle, this.labelColorSelect, this.labelFilterSelect);
    this.controlsEl.appendChild(labelGroup);

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
      this.rangeSlider.setLabel(
        primary.scalarWindow ? SCALAR_RANGE_LABEL : COLOUR_RANGE_LABEL,
        primary.scalarWindow ? SCALAR_RANGE_TOOLTIP : COLOUR_RANGE_TOOLTIP
      );
      this.rangeSlider.setBounds(primary.dataMin, primary.dataMax);
      this.rangeSlider.setValues(primary.displayMin, primary.displayMax);
    }

    this.gammaSlider?.setValue(primary.gamma);
    if (this.layerOrderInput) {
      // Blank when the layer states no level, so the placeholder ('auto')
      // shows the inferred-ordering state rather than a fabricated 0.
      this.layerOrderInput.value =
        primary.layerOrderExplicit && primary.layerOrder !== undefined
          ? String(primary.layerOrder)
          : '';
    }
    this.opacitySlider?.setValue(primary.opacity);
    // Seat the thumb on a track that can represent THIS layer's live κ. The
    // track itself is now layer-independent — every geometry family builds
    // τ = κ · rayMass from the same normalised ray mass, so κ is comparable
    // across points/lines/gsplats and across scene scales — and only widens
    // when an authored κ falls outside the nominal span. setRange keeps the
    // value put.
    // Multi-selection: onChange fans the PRIMARY layer's κ out to every
    // selected layer (same as opacity / gamma).
    if (this.absorptionSlider) {
      const { min, max } = absorptionSliderRange(primary.absorption);
      this.absorptionSlider.setRange(min, max);
      this.absorptionSlider.setValue(primary.absorption);
    }

    // Mesh appearance: seat all five thumbs on the layer's live values. Pushed
    // unconditionally, before the visibility gate below — a hidden slider still has to
    // hold the right value, or selecting a mesh layer would briefly show the previous
    // layer's numbers.
    this.ambientSlider?.setValue(primary.ambient);
    this.shadeExponentSlider?.setValue(primary.shadeExponent);
    this.specularSlider?.setValue(primary.specular);
    this.shininessSlider?.setValue(primary.shininess);
    this.alphaCutoffSlider?.setValue(primary.alphaCutoff);

    if (this.blendSelect) {
      this.blendSelect.value = primary.blendingMode;
    }
    this.syncAbsorptionVisibility();
    this.syncMeshAppearanceVisibility();

    if (this.colormapSelect) {
      if (primary.supportsColormap) {
        this.colormapSelect.parentElement!.style.display = '';
        this.colormapSelect.value = primary.colormap ?? '';
      } else {
        // Hide colormap control for layers that don't support it
        this.colormapSelect.parentElement!.style.display = 'none';
      }
    }

    if (this.labelColorSelect && this.labelFilterSelect) {
      const container = this.labelColorSelect.parentElement!;
      const vocabulary = primary.labelVocabulary;
      container.style.display = vocabulary?.length ? '' : 'none';
      if (vocabulary?.length) {
        this.labelColorSelect.value = primary.colorByLabel ? 'categorical' : 'authored';
        this.labelFilterSelect.innerHTML = '';
        this.labelFilterSelect.append(new Option('all classes', ''));
        vocabulary.forEach((entry) => {
          this.labelFilterSelect!.append(new Option(`${entry.name} (${entry.id})`, entry.id));
        });
        this.labelFilterSelect.value = primary.labelFilterId ?? '';
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
   * primary selection's mode is `volumetric`. All three geometry types
   * implement the volumetric math (gsplats phase 1, points phase 3,
   * lines phase 4), so the mode alone decides. Called from render() and
   * the blend-dropdown change handler (mode switches must reveal/hide
   * it immediately).
   */
  private syncAbsorptionVisibility(): void {
    if (!this.absorptionSlider) return;
    const primary = this.deps.state.getPrimarySelected();
    // Gate on the MESH-RESOLVED mode: a mesh resolves `volumetric` → `opaque`
    // (it has no `updateAbsorption`), so absorption stays correctly hidden for a
    // mesh; a real volumetric gsplat/points/lines still shows it.
    const show =
      !!primary && resolveLayerBlendingMode(primary.type, primary.blendingMode) === 'volumetric';
    this.absorptionSlider.setVisible(show);
  }

  /**
   * Show the five mesh appearance sliders only when they can do something.
   *
   * All five are type-gated. The four lighting controls are additionally hidden when
   * the material resolves to `shading="none"`, where their uniforms are compiled out.
   *
   * `alphaCutoff` instead carries a mode gate ON TOP: the cutout only exists in
   * `opaque`, so in any other mesh mode the threshold is read by no branch of the
   * fragment shader.
   * The gate is on the MESH-RESOLVED mode, so a mesh in a volumetric-inherited/selected
   * mode (which a mesh resolves back to `opaque`) still shows its active cutout slider.
   *
   * A GROUP layer over meshes deliberately does NOT get these. Unlike opacity and
   * gamma, they do not compose along the ancestry (a shade floor is not a
   * multiplicative attr), so a group control would have to mean "set all descendants",
   * which is a different verb from every other control in this panel.
   */
  private syncMeshAppearanceVisibility(): void {
    const primary = this.deps.state.getPrimarySelected();
    const isMesh = primary?.type === 'mesh';
    const hasLighting = isMesh && primary.shading !== 'none';
    this.ambientSlider?.setVisible(hasLighting);
    this.shadeExponentSlider?.setVisible(hasLighting);
    this.specularSlider?.setVisible(hasLighting);
    this.shininessSlider?.setVisible(hasLighting);
    this.alphaCutoffSlider?.setVisible(
      isMesh && resolveLayerBlendingMode(primary.type, primary.blendingMode) === 'opaque'
    );
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
   * the data-monitor chip (data-loading-monitor/templates/scene-graph.ts) and the
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
