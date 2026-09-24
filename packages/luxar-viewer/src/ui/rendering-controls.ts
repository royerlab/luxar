// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI from './gui';
import { PostProcessingManager } from '../rendering';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation/animation-controller';
import { config, type RenderingSettings } from '../config';
import {
  minNearForRadius,
  SPHERE_SAFETY_EXPANSION,
} from '../scene/scene-manager/clipping/bounds-math';

import type { DensityGuardControl, RenderingControllers } from './rendering-controls/types';
import { log, Modules } from '../utils/log';
import { setupCameraControls } from './rendering-controls/setup/camera-setup';
import { setupHDRControls } from './rendering-controls/setup/hdr-setup';
import { setupAntiAliasingControls } from './rendering-controls/setup/anti-aliasing-setup';
import { setupPostProcessingControls } from './rendering-controls/setup/post-processing-setup';
import { CinematicModeController } from './rendering-controls/cinematic-mode';
import { applyRenderingSettings } from './rendering-controls/apply-settings';
import {
  syncCameraFovState as syncCameraFovStateImpl,
  syncCurrentState as syncCurrentStateImpl,
} from './rendering-controls/sync-current-state';
import { FocusManager } from './rendering-controls/focus-manager';
import {
  buildBaseDefaults,
  buildResetDefaults,
  clearStoredSettings,
  saveSettingsToStorage,
  loadSettingsFromStorage,
} from './rendering-controls/settings-persistence';
import { ClippingDisplay } from './rendering-controls/clipping-display';
import { validateRenderingSettings } from './rendering-controls/controls-utils';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import type { ZarrViewerConfig } from '../types/zarr';
import { extractRenderingOverrides } from '../config/zarr-bridge/viewer-config-utils';

/**
 * Re-export of the cinematic-mode snapshot types so consumers can import them
 * from the rendering-controls module: `CinematicSnapshot` is the captured set of
 * cinematic effect values and `CinematicSnapshotKeys` its key union.
 */
export type { CinematicSnapshot, CinematicSnapshotKeys } from './rendering-controls/cinematic-mode';

/**
 * Largest power of ten at or below `v`, floored at 1e-6 — a slider step whose
 * `String()` form is an exact short decimal.
 *
 * The GUI derives a controller's displayed decimal count from
 * `String(step).split('.')[1].length` (`slider-kit/format.ts`), so the
 * step's *textual* form is load-bearing, not just its magnitude:
 *
 *  - A scene-derived value carries float noise. `String(1.05e-4)` is
 *    `"0.00010499999999999999"` → 20 decimals → a near of 117.5 renders as
 *    `"117.50000000000000000000"`.
 *  - Below 1e-6, `String` switches to exponential (`String(1e-7) === "1e-7"`),
 *    where that split reads the decimal count off the MANTISSA — or finds no
 *    `.` at all and reports 0, rendering every small value as `"0"`.
 *
 * What is load-bearing is only that the result is a POWER OF TEN (so `String()`
 * is short and exact) and that the exponent is floored at -6 (so `String()`
 * stays decimal). The literal spelling is not: `Number('1e'+e)` and
 * `Math.pow(10, e)` were measured identical at every exponent from -13 to +12,
 * so either works — the literal just reads as the intent.
 *
 * The 1e-6 floor costs slider granularity on sub-micron scenes and buys a
 * correct readout, which is the right trade for a control that is a read-only
 * live display whenever dynamic clipping is on.
 *
 * Exported for test. The underlying `formatNumber` limitation is the GUI's, not
 * this module's — this is the caller-side accommodation.
 */
export function decadeStep(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1e-6;
  return Number(`1e${Math.max(-6, Math.floor(Math.log10(v)))}`);
}

/**
 * Advanced rendering parameters GUI for real-time visual control.
 *
 * Provides comprehensive UI for controlling:
 * - Post-processing effects (bloom, noise, vignette, chromatic aberration, lens distortion)
 * - HDR intensity and tone mapping
 * - Anti-aliasing options (FXAA, MSAA, SSAA)
 * - Camera field of view (FOV and FOV presets)
 * - Dynamic clipping planes for nD visualization
 *
 * Features:
 * - Settings persistence per scene (localStorage)
 * - Cinematic mode presets (C key for quick film-like look)
 * - Real-time updates with deferred rebuild to prevent lag
 * - Clean collapsible UI using custom GUI library
 * - Auto-blur behavior to keep keyboard shortcuts working
 *
 * @example
 * ```typescript
 * const renderingControls = new RenderingControls(
 *   postProcessingManager,
 *   sceneManager
 * );
 *
 * // Associate with input handler for R key toggle
 * inputHandler.setRenderingControls(renderingControls);
 *
 * // Set animation controller for effects requiring continuous render
 * renderingControls.setAnimationController(animController);
 *
 * // User can now press R to toggle controls panel
 * ```
 */
export class RenderingControls {
  /** The custom GUI instance */
  private gui: GUI;

  /** Current rendering settings (public for state capture) */
  public settings: RenderingSettings;

  /** Scene identifier for settings persistence */
  private sceneId: string = '';
  private zarrViewerConfig: ZarrViewerConfig | undefined = undefined;
  private hasStoredLocalSettings: boolean = false;

  /** Reference to post-processing manager */
  private postProcessing: PostProcessingManager;

  /** Reference to scene manager */
  private sceneManager: SceneManager;

  /** Reference to animation controller for triggering re-renders */
  private animationController?: AnimationController;

  /** Visibility state */
  private visible: boolean = false;

  /** Outside-click + focus management for the panel. */
  private readonly focusManager: FocusManager;

  /** References to GUI controllers for updates */
  private controllers: RenderingControllers = {};

  /** Reference to adaptive DPR manager (persisted enabled state applied on load). */
  private adaptiveDPRManager?: AdaptiveDPRManager;
  private densityGuardControl?: DensityGuardControl;

  /** RAF-driven mirror of the camera near/far values into the slider displays. */
  private readonly clippingDisplay: ClippingDisplay;

  /** Cleanup callbacks collected during setup, called on dispose */
  private cleanupCallbacks: (() => void)[] = [];

  /** Cinematic mode preset controller (created in `setupControls` during construction). */
  private cinematic?: CinematicModeController;

  /**
   * Create rendering controls UI with complete parameter access.
   *
   * Initializes GUI with all post-processing and rendering controls
   * organized in folders. Sets up auto-blur behavior and keyboard handling.
   * Starts hidden - call show() or toggle() to display.
   *
   * @param postProcessing - Post-processing manager for bloom, noise, etc.
   * @param sceneManager - Scene manager for camera and control access
   *
   * @example
   * ```typescript
   * const controls = new RenderingControls(
   *   postProcessingManager,
   *   sceneManager
   * );
   * controls.show();  // Display controls panel
   * ```
   */
  constructor(postProcessing: PostProcessingManager, sceneManager: SceneManager) {
    this.postProcessing = postProcessing;
    this.sceneManager = sceneManager;
    this.settings = buildBaseDefaults();

    // Initialize GUI with close button callback
    this.gui = new GUI({
      title: 'Rendering Controls',
      width: 300,
      closeFolders: false,
      onClose: () => this.hide(),
      closeButtonTitle: 'Close (R)',
    });

    // Position on the left side with standard margins
    this.gui.domElement.style.position = 'fixed';
    this.gui.domElement.style.top = 'calc(20px + env(safe-area-inset-top, 0px))';
    this.gui.domElement.style.left = '20px'; // Standard 20px margin
    this.gui.domElement.style.zIndex = String(config.ui.zIndex.renderingControls);

    // Start hidden
    this.gui.hide();

    // Custom styling now in src/styles/components/rendering-controls.css

    this.clippingDisplay = new ClippingDisplay({
      sceneManager: this.sceneManager,
      settings: this.settings,
      getNearPlane: () => this.controllers.nearPlane,
      getFarPlane: () => this.controllers.farPlane,
    });

    this.focusManager = new FocusManager({
      panel: this.gui.domElement,
      canvas: this.sceneManager.renderer.domElement,
    });

    this.setupControls();
  }

  /**
   * Setup all GUI controls
   */
  private setupControls(): void {
    // Setup auto-blur for all controls

    // Navigation controls live in the Navigation rail popover (right-click the
    // Navigation gauge; left-click cycles orbit/fly/ortho) — see
    // ui/rail-panels/navigation-popover. Navigation is not a rendering concern.

    // Camera controls (pass controllers reference for FOV preset lens distortion updates)
    const cameraResult = setupCameraControls(
      {
        gui: this.gui,
        settings: this.settings,
        postProcessing: this.postProcessing,
        sceneManager: this.sceneManager,
        animationController: this.animationController,
        saveSettings: () => this.saveSettings(),
        triggerAnimation: () => this.triggerAnimation(),
        updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
        updateNavigationControls: (controlType) => this.updateNavigationControls(controlType),
      },
      this.controllers
    );

    // Store controller references
    Object.assign(this.controllers, cameraResult.controllers);

    // HDR controls
    const hdrResult = setupHDRControls({
      gui: this.gui,
      settings: this.settings,
      postProcessing: this.postProcessing,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      saveSettings: () => this.saveSettings(),
      triggerAnimation: () => this.triggerAnimation(),
      updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
      updateNavigationControls: (controlType) => this.updateNavigationControls(controlType),
    });

    // Store controller references
    Object.assign(this.controllers, hdrResult.controllers);

    // Anti-aliasing controls
    const aaResult = setupAntiAliasingControls({
      gui: this.gui,
      settings: this.settings,
      postProcessing: this.postProcessing,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      saveSettings: () => this.saveSettings(),
      triggerAnimation: () => this.triggerAnimation(),
      updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
      updateNavigationControls: (controlType) => this.updateNavigationControls(controlType),
    });

    // Store controller references
    Object.assign(this.controllers, aaResult.controllers);

    // Post-processing effects controls (pass controllers reference for FOV preset sync)
    const ppResult = setupPostProcessingControls(
      {
        gui: this.gui,
        settings: this.settings,
        postProcessing: this.postProcessing,
        sceneManager: this.sceneManager,
        animationController: this.animationController,
        saveSettings: () => this.saveSettings(),
        triggerAnimation: () => this.triggerAnimation(),
        updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
        updateNavigationControls: (controlType) => this.updateNavigationControls(controlType),
      },
      this.controllers
    );

    // Store controller references
    Object.assign(this.controllers, ppResult.controllers);

    // Theme lives in the Settings rail popover (see ui/rail-panels/settings-popover)
    // and the Performance folder in the Performance rail popover — neither is a
    // rendering concern, so both were moved out of this panel.

    this.cinematic = this.createCinematicController();

    // Reset to Defaults button — added last so it sits at the panel bottom.
    this.addResetButton();
  }

  /** Add the root-level "Reset to Defaults" button at the bottom of the panel. */
  private addResetButton(): void {
    const resetButton = {
      'Reset to Defaults': () => {
        this.resetToDefaults();
      },
    };

    const resetControl = this.gui.add(resetButton, 'Reset to Defaults');
    resetControl.domElement.setAttribute(
      'title',
      'Reset to Defaults: Restore all rendering settings\n' +
        '• Resets camera, HDR, bloom, anti-aliasing\n' +
        '• Resets all post-processing effects\n' +
        '• Resets navigation controls\n' +
        '• Clears saved settings for this scene'
    );
  }

  /** Build the cinematic-mode controller. `animationController` is read lazily so
   * the cinematic toggle works correctly after `setAnimationController()` runs. */
  private createCinematicController(): CinematicModeController {
    const self = this;
    return new CinematicModeController({
      settings: this.settings,
      postProcessing: this.postProcessing,
      sceneManager: this.sceneManager,
      controllers: this.controllers,
      get animationController() {
        return self.animationController;
      },
      saveSettings: () => this.saveSettings(),
      triggerAnimation: () => this.triggerAnimation(),
      refreshAllControllers: () => {
        this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
      },
    });
  }

  /**
   * Reset all rendering settings to their default values.
   *
   * Public because the rail Home popover's "Reset rendering" action calls it
   * too (same behavior as this panel's own "Reset to Defaults" button).
   */
  public resetToDefaults(): void {
    // Clear cinematic snapshot since we're resetting all settings
    this.cinematic?.clearSnapshot();

    // Get fresh defaults (zarr overrides are layered on top when available).
    const defaults = buildResetDefaults(this.zarrViewerConfig);

    // Update settings object in place to maintain GUI bindings
    Object.assign(this.settings, defaults);

    // Clear saved settings for this scene (before applying, so user sees clean state).
    clearStoredSettings(this.sceneId);

    // Apply the reset DPR settings. Neither reaches the manager through
    // `applySettings()` below — that drives the post-processing pipeline
    // — so without this the reset repaints both toggles while the viewer
    // keeps rendering at the old ceiling with the old adaptation state,
    // until a reload. High-DPR FIRST, as in `loadSettings`: it sets the
    // ceiling that `setEnabled` then settles the operating DPR against.
    this.adaptiveDPRManager?.setHighDPRAllowed(this.settings.allowHighDPR);
    this.adaptiveDPRManager?.setEnabled(this.settings.adaptiveDPREnabled);
    this.applyDensityGuardSetting();

    // Apply camera settings to scene manager (before post-processing)
    this.sceneManager.setFov(this.settings.fov);

    // Apply clipping planes (reset to defaults)
    this.sceneManager.updateClippingPlanes(this.settings.near, this.settings.far);

    // Apply navigation control settings
    // Use defaults object directly since we just assigned these concrete values
    this.sceneManager.setControlType(this.settings.controlType);
    this.sceneManager.setAutoRotate(this.settings.autoRotate);
    this.sceneManager.setAutoRotateSpeed(this.settings.autoRotateSpeed);
    this.sceneManager.setAutoRotateAxis(this.settings.autoRotateAxis);
    this.sceneManager.setAutoDolly(this.settings.autoDolly);
    this.sceneManager.setAutoDollyAmplitudePercent(this.settings.autoDollyAmplitudePercent);
    this.sceneManager.setAutoDollyPeriod(this.settings.autoDollyPeriod);
    this.sceneManager.setNaturalDrag(this.settings.naturalDrag);
    this.sceneManager.setOrbitZoomSpeed(defaults.orbitZoomSpeed);
    this.sceneManager.setOrbitDampingFactor(defaults.orbitDampingFactor);
    this.sceneManager.setFlyMovementSpeed(defaults.flyMovementSpeed);
    this.sceneManager.setFlyRotationSpeed(defaults.flyRotationSpeed);
    this.sceneManager.setFlyLookSpeed(defaults.flyLookSpeed);
    this.sceneManager.setFlyInertialMode(defaults.flyInertialMode);
    this.sceneManager.setFlyDamping(defaults.flyDamping);
    this.sceneManager.setFlyRotationDamping(defaults.flyRotationDamping);

    // Apply all post-processing settings to the rendering pipeline
    // Note: applySettings() also applies dynamic clipping and triggers animation
    this.applySettings();

    // Update all GUI controllers to reflect new values (after all settings applied)
    this.gui.controllersRecursive().forEach((controller) => {
      controller.updateDisplay();
    });

    // Update cinematic mode checkbox based on reset effects state
    this.updateCinematicModeCheckbox();

    // Update folder visibility based on control type
    this.updateNavigationControls(this.settings.controlType);

    // Update clipping controls state
    this.updateClippingControlsState(this.settings.dynamicClippingEnabled);

    // Show/hide fly damping controls based on inertial mode
    if (this.controllers.flyDamping) {
      if (this.settings.flyInertialMode) {
        this.controllers.flyDamping.show();
      } else {
        this.controllers.flyDamping.hide();
      }
    }
    if (this.controllers.flyRotationDamping) {
      if (this.settings.flyInertialMode) {
        this.controllers.flyRotationDamping.show();
      } else {
        this.controllers.flyRotationDamping.hide();
      }
    }

    // Re-apply scale-aware fly speed (resetToDefaults uses hardcoded config defaults
    // which don't account for scene scale)
    this.updateSceneScale();

    log.info(Modules.RENDERER, 'Rendering settings reset to defaults');
  }

  /**
   * Apply custom styling to match dimension sliders and help panel
   */
  // applyCustomStyling() method removed - all styling now in src/styles/components/rendering-controls.css

  /**
   * Set the animation controller reference
   * @param animationController - The animation controller instance
   */
  setAnimationController(animationController: AnimationController): void {
    this.animationController = animationController;
  }

  /**
   * Store the adaptive DPR manager reference.
   *
   * The Performance controls (Adaptive Resolution toggle, Manual DPR slider,
   * live DPR/FPS readout) live in the Performance rail popover (right-click the
   * gauge) — see `ui/rail-panels/performance-popover`. The panel only needs the
   * manager so {@link loadSettings} can apply the persisted enabled state.
   *
   * @param manager - The AdaptiveDPRManager instance
   */
  setAdaptiveDPRManager(manager: AdaptiveDPRManager): void {
    this.adaptiveDPRManager = manager;
  }

  /**
   * Store the density-guard handle so {@link loadSettings} (and a reset) can
   * apply the persisted `densityGuardEnabled` flag. The toggle itself lives in
   * the Performance rail popover, which self-syncs from the handle when opened.
   */
  setDensityGuardControl(control: DensityGuardControl): void {
    this.densityGuardControl = control;
  }

  /**
   * Apply the stored Density Guard choice — unless `?noDensityGuard` turned
   * the guard off for this session, in which case the stored flag is left
   * alone (neither applied nor overwritten), like a URL DPR pin.
   */
  private applyDensityGuardSetting(): void {
    const control = this.densityGuardControl;
    if (control && !control.sessionDisabled) {
      control.setEnabled(this.settings.densityGuardEnabled);
    }
  }

  /**
   * Set the scene identifier for settings persistence
   * @param zarrUrl - URL of the zarr store
   * @param sceneName - Name of the scene
   */
  setSceneId(zarrUrl: string, sceneName?: string): void {
    // Generate a unique ID from URL and scene name
    const baseId = zarrUrl.replace(/[^a-zA-Z0-9]/g, '_');
    this.sceneId = sceneName ? `${baseId}_${sceneName}` : baseId;

    // Load settings for this scene (will apply if found)
    this.loadSettings();

    // Apply camera settings after loading (fov, near, far)
    // This ensures loaded settings are actually applied to the camera
    // Note: Dynamic clipping is applied later via applySettings()
    this.sceneManager.setFov(this.settings.fov);

    // Apply clipping planes
    this.sceneManager.updateClippingPlanes(this.settings.near, this.settings.far);

    // Apply navigation control settings
    this.sceneManager.setControlType(this.settings.controlType);
    this.sceneManager.setAutoRotate(this.settings.autoRotate);
    this.sceneManager.setAutoRotateSpeed(this.settings.autoRotateSpeed);
    this.sceneManager.setAutoRotateAxis(this.settings.autoRotateAxis);
    this.sceneManager.setAutoDolly(this.settings.autoDolly);
    this.sceneManager.setAutoDollyAmplitudePercent(this.settings.autoDollyAmplitudePercent);
    this.sceneManager.setAutoDollyPeriod(this.settings.autoDollyPeriod);
    this.sceneManager.setNaturalDrag(this.settings.naturalDrag);

    // Apply orbit feel settings (if they exist in loaded settings)
    if (this.settings.orbitZoomSpeed !== undefined) {
      this.sceneManager.setOrbitZoomSpeed(this.settings.orbitZoomSpeed);
    }
    if (this.settings.orbitDampingFactor !== undefined) {
      this.sceneManager.setOrbitDampingFactor(this.settings.orbitDampingFactor);
    }

    // Apply fly control settings (if they exist in loaded settings)
    if (this.settings.flyMovementSpeed !== undefined) {
      this.sceneManager.setFlyMovementSpeed(this.settings.flyMovementSpeed);
    }
    if (this.settings.flyRotationSpeed !== undefined) {
      this.sceneManager.setFlyRotationSpeed(this.settings.flyRotationSpeed);
    }
    if (this.settings.flyLookSpeed !== undefined) {
      this.sceneManager.setFlyLookSpeed(this.settings.flyLookSpeed);
    }
    if (this.settings.flyInertialMode !== undefined) {
      this.sceneManager.setFlyInertialMode(this.settings.flyInertialMode);
    }
    if (this.settings.flyDamping !== undefined) {
      this.sceneManager.setFlyDamping(this.settings.flyDamping);
    }
    if (this.settings.flyRotationDamping !== undefined) {
      this.sceneManager.setFlyRotationDamping(this.settings.flyRotationDamping);
    }

    // Always apply current settings to ensure proper initialization
    // This is needed when no stored settings exist (first time loading)
    this.applySettings();
  }

  /**
   * Set the zarr viewer config from the loaded scene.
   * Called after loadSceneData() completes so the zarr metadata is available.
   */
  setZarrViewerConfig(viewerConfig: ZarrViewerConfig | undefined): void {
    this.zarrViewerConfig = viewerConfig;
  }

  /**
   * Whether this scene has stored settings in localStorage.
   */
  hasStoredSettings(): boolean {
    return this.hasStoredLocalSettings;
  }

  /**
   * Apply zarr viewer_config as defaults for a first-time scene visit.
   * Called when no localStorage exists and zarr provides scene-specific defaults.
   * Re-applies the full 3-tier priority chain and updates the scene.
   */
  applyZarrDefaults(): void {
    if (!this.zarrViewerConfig) return;
    this.applyOverrides(
      extractRenderingOverrides(this.zarrViewerConfig),
      'Applied viewer config defaults from zarr'
    );
  }

  /**
   * JSON-safe copy of the live rendering settings (the embedder API's
   * `getRenderingSettings()`). A copy, so a consumer cannot mutate the
   * panel's state behind its back.
   */
  getSettingsSnapshot(): RenderingSettings {
    return { ...this.settings };
  }

  /**
   * Apply a partial settings override programmatically — the one path a
   * scene's authored `viewer_config` and the embedder API's
   * `setRenderingSettings()` share, so a remote controller can set exactly
   * what an author can bake, with the same validation and the same
   * side-effects (camera FOV/planes, navigation, DPR ceiling, post-processing).
   *
   * Values go through `validateRenderingSettings` so NaN / Infinity /
   * out-of-range input clamps to defaults instead of reaching the renderer.
   * Unknown keys are ignored by validation. Nothing is persisted: like the
   * authored defaults, an override describes THIS session's scene, not a
   * user preference.
   */
  applyOverrides(
    overrides: Partial<RenderingSettings>,
    logMessage = 'Applied programmatic rendering settings'
  ): void {
    // Route overrides through validateRenderingSettings so a corrupted
    // viewer_config (or a remote controller) can't inject NaN/Infinity/
    // out-of-range values into runtime rendering state. Validation clamps
    // to defaults.
    const zarrOverrides = overrides;
    const validated = validateRenderingSettings({ ...this.settings, ...zarrOverrides });
    Object.assign(this.settings, validated);

    // Apply FOV if overridden
    if (zarrOverrides.fov !== undefined) {
      this.sceneManager.setFov(this.settings.fov);
    }

    // Apply clipping planes if overridden
    if (zarrOverrides.near !== undefined || zarrOverrides.far !== undefined) {
      this.sceneManager.updateClippingPlanes(this.settings.near, this.settings.far);

      // Authored planes and dynamic clipping are mutually exclusive in effect:
      // the per-frame update recomputes near/far from scene bounds on the very
      // next frame, so authored values survive for one frame and then vanish
      // with no diagnostic. Precedence is intentionally NOT changed here —
      // dynamic clipping is an explicit auto mode and silently disabling it
      // would be the more surprising behaviour — but the author deserves to
      // know why their setting appears to be ignored. Pair
      // `camera.near`/`camera.far` with `dynamic_clipping_enabled=False` in
      // viewer_config to make them stick.
      if (this.settings.dynamicClippingEnabled) {
        log.warning(
          Modules.RENDERER,
          'viewer_config sets camera.near/far while dynamic clipping is enabled; ' +
            'the per-frame update will override them. Set dynamic_clipping_enabled=False to keep them.'
        );
      }
    }

    // Apply navigation settings if overridden
    if (zarrOverrides.controlType !== undefined) {
      this.sceneManager.setControlType(this.settings.controlType);
    }
    if (zarrOverrides.autoRotate !== undefined) {
      this.sceneManager.setAutoRotate(this.settings.autoRotate);
    }
    if (zarrOverrides.autoRotateSpeed !== undefined) {
      this.sceneManager.setAutoRotateSpeed(this.settings.autoRotateSpeed);
    }
    if (zarrOverrides.autoRotateAxis !== undefined) {
      this.sceneManager.setAutoRotateAxis(this.settings.autoRotateAxis);
    }
    if (zarrOverrides.autoDolly !== undefined) {
      this.sceneManager.setAutoDolly(this.settings.autoDolly);
    }
    if (zarrOverrides.autoDollyAmplitudePercent !== undefined) {
      this.sceneManager.setAutoDollyAmplitudePercent(this.settings.autoDollyAmplitudePercent);
    }
    if (zarrOverrides.autoDollyPeriod !== undefined) {
      this.sceneManager.setAutoDollyPeriod(this.settings.autoDollyPeriod);
    }
    if (zarrOverrides.naturalDrag !== undefined) {
      this.sceneManager.setNaturalDrag(this.settings.naturalDrag);
    }

    // An authored `allow_high_dpr` has to reach the manager here:
    // `applySettings` below drives post-processing, not the DPR ceiling,
    // and the ceiling must move before the first frame is sized.
    if (zarrOverrides.allowHighDPR !== undefined) {
      this.adaptiveDPRManager?.setHighDPRAllowed(this.settings.allowHighDPR);
    }

    // Update GUI controllers to reflect new values
    this.gui.controllersRecursive().forEach((controller) => {
      controller.updateDisplay();
    });

    // Sync HDR log slider
    // Apply post-processing and other rendering settings
    this.applySettings();
    log.info(Modules.RENDERER, logMessage);
  }

  /**
   * Trigger animation when parameters change
   */
  private triggerAnimation(): void {
    // Start animation to see changes immediately
    this.animationController?.startAnimation();
  }

  /**
   * Update clipping controls state based on dynamic clipping setting.
   * When dynamic clipping is enabled:
   * - Grey out manual near/far controls (but keep them visible)
   * - Start periodic updates to show actual camera clipping values
   * - Disable pointer events so sliders can't be manually adjusted
   */
  private updateClippingControlsState(dynamicEnabled: boolean): void {
    this.clippingDisplay.setDynamicEnabled(dynamicEnabled);
  }

  /**
   * Update fly speed UI slider range and value based on scene scale.
   * Called after scene data loads and scale is known.
   */
  public updateSceneScale(): void {
    const scale = this.sceneManager.getSceneScale();
    if (scale <= 0) return;

    // Re-apply scale to ControlsManager. This is necessary because applyZarrDefaults()
    // may have called setFlyMovementSpeed() with config defaults after
    // autoAdjustClippingPlanes() set the scale-derived speed, overwriting it.
    this.sceneManager.controls.setSceneScale(scale);

    const m = config.controls.scaleMultipliers;
    const scaledSpeed = scale * m.flySpeedFactor;

    // Update slider range: allow 0.1x to 10x of the scale-derived speed
    const newMin = Math.max(0.01, scaledSpeed * 0.1);
    const newMax = scaledSpeed * 10;
    const newStep = Math.max(0.01, scaledSpeed * 0.01);

    if (this.controllers.flyMovementSpeed) {
      // NumberController supports dynamic .min()/.max()/.step() but the
      // base Controller type doesn't expose them; structural cast targets
      // just those three fluent methods.
      type ChainableNumber = {
        min(v: number): ChainableNumber;
        max(v: number): ChainableNumber;
        step(v: number): ChainableNumber;
      };
      const ctrl = this.controllers.flyMovementSpeed as unknown as ChainableNumber;
      if (typeof ctrl.min === 'function') {
        ctrl.min(newMin).max(newMax).step(newStep);
      }
    }

    // Sync the settings and UI with the scale-derived speed
    this.settings.flyMovementSpeed = scaledSpeed;
    this.sceneManager.setFlyMovementSpeed(scaledSpeed);

    // Update slider display
    if (this.controllers.flyMovementSpeed) {
      this.controllers.flyMovementSpeed.setValue(scaledSpeed);
      this.controllers.flyMovementSpeed.updateDisplay();
    }

    log.info(
      Modules.UI,
      `Fly speed range updated for scale ${scale.toFixed(1)}: ` +
        `[${newMin.toFixed(2)}, ${newMax.toFixed(1)}], speed=${scaledSpeed.toFixed(1)}`
    );

    this.updateClippingSliderRanges(scale);
  }

  /**
   * Re-range the near/far sliders from the scene scale.
   *
   * Their authored range is ABSOLUTE (`near` 0.0001–10, `far` 1–100000) while
   * every value they ever show is scene-relative, so on any scene that is not
   * roughly 100 world units across the two disagree. The visible symptom is
   * under dynamic clipping, where the sliders are read-only live readouts: at a
   * framed camera on a diagonal-100 scene `near` is ~44, so the number input
   * reads 44 truthfully while `<input type=range>` clamps its own value and
   * pins the thumb at the 10 end. On a micron-scale scene the whole useful
   * range collapses below the 0.0001 minimum instead.
   *
   * Ranged off the same scene diagonal the rest of the scale-aware machinery
   * uses, and bracketing what the clipping policy can actually produce:
   * `near` bottoms out at `MIN_NEAR_RADIUS_FACTOR · R` (the ortho floor) and
   * tops out near the framed distance; `far` reaches `dist + R` at the
   * zoom-out limit. Mirrors the fly-speed re-ranging directly above — same
   * trigger, same structural cast, same reason.
   *
   * The STEP is not simply the range minimum, because the GUI derives the
   * displayed decimal count from `String(step)` (`slider-kit/format.ts`).
   * A scene-derived step carries float noise into that string — `String(1.05e-4)`
   * is `"0.00010499999999999999"`, which renders every value with TWENTY
   * decimals. `decimalsForStep` handles exponential notation, but it cannot
   * distinguish meaningful precision from that binary float noise, so
   * `decadeStep` hands it a clean value; see that helper for the exact bounds.
   */
  private updateClippingSliderRanges(scale: number): void {
    type ChainableNumber = {
      min(v: number): ChainableNumber;
      max(v: number): ChainableNumber;
      step(v: number): ChainableNumber;
    };
    const reRange = (
      controller: (typeof this.controllers)[keyof typeof this.controllers],
      min: number,
      max: number,
      step: number
    ): void => {
      if (!controller) return;
      const ctrl = controller as unknown as ChainableNumber;
      if (typeof ctrl.min === 'function') ctrl.min(min).max(max).step(step);
    };

    // Both maxima come from the SAME sphere equations the clipping policy uses,
    // evaluated at the furthest camera distance the controls allow. Anything
    // less and the range input clamps again — which is the entire symptom this
    // method exists to remove.
    //
    // `nearMax = scale` (an earlier spelling) was not merely short at the
    // zoom-out limit: `near = dist - R` overtakes it once `dist > scale + R`,
    // i.e. at 0.9x the framed distance, so the thumb pinned at the OPENING
    // pose of any ordinary scene. It also made things worse below diagonal ~10,
    // where the old absolute max of 10 was the larger of the two and a manual
    // `near` above the scene diagonal stopped being settable.
    //
    // R is the safety-expanded radius the clipping policy works in.
    const R = 0.5 * scale * SPHERE_SAFETY_EXPANSION;
    const distMax = scale * config.controls.scaleMultipliers.maxDistanceFactor;
    const nearMin = minNearForRadius(R);
    const nearMax = distMax; // near = dist - R, so distMax bounds it
    const farMin = nearMin * 10;
    const farMax = distMax + R; // far = dist + R at the limit

    reRange(this.controllers.nearPlane, nearMin, nearMax, decadeStep(nearMin));
    reRange(this.controllers.farPlane, farMin, farMax, decadeStep(scale / 1000));

    // Re-assert the live values: `<input type=range>` clamped them to the OLD
    // bounds, so the thumbs stay stale until the display is refreshed.
    this.controllers.nearPlane?.updateDisplay();
    this.controllers.farPlane?.updateDisplay();

    log.info(
      Modules.UI,
      `Clipping slider ranges updated for scale ${scale.toFixed(1)}: ` +
        `near [${nearMin.toExponential(1)}, ${nearMax.toFixed(1)}], ` +
        `far [${farMin.toExponential(1)}, ${farMax.toFixed(0)}]`
    );
  }

  /**
   * Apply control-type-driven visibility to the panel's camera controls.
   *
   * The navigation parameter controls themselves now live in the Navigation
   * rail popover; the only control-type-dependent UI left in this panel is the
   * FOV row, which is irrelevant under the ortho (orthographic) projection.
   * Called on show/sync and after a control-mode switch (`syncCurrentState`).
   */
  private updateNavigationControls(controlType: 'orbit' | 'fly' | 'ortho'): void {
    const isOrtho = controlType === 'ortho';
    if (this.controllers.fov) {
      isOrtho ? this.controllers.fov.hide() : this.controllers.fov.show();
    }
    if (this.controllers.fovPreset) {
      isOrtho ? this.controllers.fovPreset.hide() : this.controllers.fovPreset.show();
    }
  }

  /**
   * Save current settings to localStorage. Public so the rail popovers
   * (navigation / performance) persist through this panel's per-scene key —
   * the shared `settings` object stays the single source of truth.
   */
  public saveSettings(): void {
    saveSettingsToStorage(this.sceneId, this.settings);
  }

  /** Load settings from localStorage and apply them to GUI / managers. */
  private loadSettings(): void {
    if (!this.sceneId) return;

    // Snapshot is session-only; clear it when loading persisted settings
    this.cinematic?.clearSnapshot();

    const { stored, loaded } = loadSettingsFromStorage(this.sceneId);
    this.hasStoredLocalSettings = stored;

    if (!stored) return;

    if (!loaded) {
      log.warning(Modules.RENDERER, 'Failed to parse rendering settings');
      return;
    }

    // Update settings properties IN PLACE to maintain GUI controller
    // bindings (replacing the entire settings object would break the
    // GUI bindings). Validate the merged base+loaded settings so
    // corrupted localStorage can't inject NaN/Infinity into runtime
    // rendering state.
    const validated = validateRenderingSettings({ ...buildBaseDefaults(), ...loaded });
    Object.assign(this.settings, validated);

    // Update GUI to reflect loaded values
    // Note: HDR controller's updateDisplay is overridden to show actual intensity
    this.gui.controllersRecursive().forEach((controller) => {
      controller.updateDisplay();
    });

    // Apply the persisted DPR settings to the manager. The Performance
    // rail popover self-syncs from the manager when opened.
    //
    // High-DPR first: it sets the ceiling that setEnabled() then settles
    // the operating DPR against, so the reverse order would apply an
    // uncapped DPR for one step and reallocate render targets twice.
    this.adaptiveDPRManager?.setHighDPRAllowed(this.settings.allowHighDPR);
    this.adaptiveDPRManager?.setEnabled(this.settings.adaptiveDPREnabled);
    this.applyDensityGuardSetting();

    // Update cinematic mode checkbox based on loaded effects state
    this.updateCinematicModeCheckbox();

    log.info(Modules.RENDERER, `Loaded rendering settings for scene: ${this.sceneId}`);
  }

  /**
   * Sync current state from scene manager
   * This ensures the GUI reflects the actual state when opened
   */
  public syncCurrentState(): void {
    syncCurrentStateImpl({
      gui: this.gui,
      settings: this.settings,
      sceneManager: this.sceneManager,
      controllers: this.controllers,
      updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
      updateCinematicModeCheckbox: () => this.updateCinematicModeCheckbox(),
      updateNavigationControls: (controlType) => this.updateNavigationControls(controlType),
    });
  }

  /** Sync only the live camera FOV and its derived preset into settings. */
  public syncCameraFovState(): void {
    syncCameraFovStateImpl(this.settings, this.sceneManager);
    this.controllers.fov?.updateDisplay();
    this.controllers.fovPreset?.updateDisplay();
  }

  /** Apply current settings to the rendering pipeline. Delegates to a pure helper. */
  private applySettings(): void {
    applyRenderingSettings({
      settings: this.settings,
      postProcessing: this.postProcessing,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      updateClippingControlsState: (enabled) => this.updateClippingControlsState(enabled),
      triggerAnimation: () => this.triggerAnimation(),
    });
  }

  /**
   * Show the rendering controls panel.
   *
   * Syncs current state from scene/post-processing managers before showing
   * to ensure GUI displays accurate values. Adds click-outside handler for
   * better focus management.
   *
   * Triggered by R key when controls are hidden.
   *
   * @example
   * ```typescript
   * // Show controls programmatically
   * renderingControls.show();
   *
   * // Or user presses R key (handled by input handler)
   * ```
   */
  show(): void {
    this.syncCurrentState();
    this.gui.show();
    this.visible = true;
    this.focusManager.onPanelShown();
  }

  /**
   * Hide the rendering controls panel. Returns focus to the canvas so
   * keyboard shortcuts keep working. Triggered by R or Escape.
   */
  hide(): void {
    this.gui.hide();
    this.visible = false;
    this.focusManager.onPanelHidden();
  }

  /**
   * Toggle rendering controls panel visibility (show ↔ hide).
   *
   * Primary method for R key binding. Syncs state before showing.
   *
   * @example
   * ```typescript
   * // User presses R key
   * renderingControls.toggle();
   * ```
   */
  toggle(): void {
    if (this.visible) {
      this.hide();
    } else {
      // Sync state before showing
      this.show();
    }
  }

  /**
   * Check if rendering controls panel is currently visible.
   *
   * @returns true if panel is shown, false if hidden
   */
  isVisible(): boolean {
    return this.visible;
  }

  /**
   * Update the cinematic mode checkbox to reflect the current state.
   * Called after toggleCinematicMode or when 'C' key is pressed.
   */
  private updateCinematicModeCheckbox(): void {
    this.cinematic?.updateCheckbox();
  }

  /**
   * Toggle cinematic mode (film-like visual preset). Delegates to
   * {@link CinematicModeController}. Triggered by the C key.
   */
  toggleCinematicMode(): void {
    this.cinematic?.toggle();
  }

  /**
   * Clean up GUI resources and remove from DOM.
   *
   * Destroys the GUI instance. Should be called when rendering controls
   * are no longer needed (e.g., application teardown).
   *
   * After calling dispose(), the RenderingControls instance cannot be reused.
   */
  dispose(): void {
    // Clean up clipping display RAF loop
    this.clippingDisplay.dispose();

    // Run all registered cleanup callbacks (e.g., adaptive DPR update interval)
    for (const cb of this.cleanupCallbacks) {
      cb();
    }
    this.cleanupCallbacks = [];

    this.focusManager.dispose();

    // Auto-blur cleanup is now handled by the custom GUI library
    this.gui.destroy();
  }
}
