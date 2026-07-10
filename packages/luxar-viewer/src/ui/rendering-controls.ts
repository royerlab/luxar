// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI from './gui';
import { PostProcessingManager } from '../rendering';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation/animation-controller';
import { config, type RenderingSettings } from '../config';

import type { RenderingControllers } from './rendering-controls/types';
import { log, Modules } from '../utils/log';
import { setupCameraControls } from './rendering-controls/setup/camera-setup';
import { setupHDRControls } from './rendering-controls/setup/hdr-setup';
import { setupAntiAliasingControls } from './rendering-controls/setup/anti-aliasing-setup';
import { setupPostProcessingControls } from './rendering-controls/setup/post-processing-setup';
import { CinematicModeController } from './rendering-controls/cinematic-mode';
import { applyRenderingSettings } from './rendering-controls/apply-settings';
import { syncCurrentState as syncCurrentStateImpl } from './rendering-controls/sync-current-state';
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

export type { CinematicSnapshot, CinematicSnapshotKeys } from './rendering-controls/cinematic-mode';

/**
 * Advanced rendering parameters GUI for real-time visual control.
 *
 * Provides comprehensive UI for controlling:
 * - Post-processing effects (bloom, noise, vignette, chromatic aberration, lens distortion)
 * - HDR intensity and tone mapping
 * - Anti-aliasing options (FXAA, MSAA, SSAA)
 * - Camera controls (orbit, fly, ortho modes with physics parameters)
 * - Point rendering (base size, near/far size, sharpness, saturation)
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

  /** RAF-driven mirror of the camera near/far values into the slider displays. */
  private readonly clippingDisplay: ClippingDisplay;

  /** Cleanup callbacks collected during setup, called on dispose */
  private cleanupCallbacks: (() => void)[] = [];

  /** Cinematic mode preset controller (lazily wired in `setAdaptiveDPRManager`). */
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
    this.gui.domElement.style.top = '20px'; // Standard 20px margin
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
    this.setupAutoBlur();

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

    // Apply camera settings to scene manager (before post-processing)
    const currentFOV = this.sceneManager.currentFov;
    if (Math.abs(currentFOV - this.settings.fov) > 0.5) {
      const delta = (this.settings.fov - currentFOV) / config.camera.fovSensitivity;
      this.sceneManager.updateFOV(delta);
    }

    // Apply clipping planes (reset to defaults)
    this.sceneManager.updateClippingPlanes(this.settings.near, this.settings.far);

    // Apply navigation control settings
    // Use defaults object directly since we just assigned these concrete values
    this.sceneManager.setControlType(this.settings.controlType);
    this.sceneManager.setAutoRotate(this.settings.autoRotate);
    this.sceneManager.setAutoRotateSpeed(this.settings.autoRotateSpeed);
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
    const currentFOV = this.sceneManager.currentFov;
    if (Math.abs(currentFOV - this.settings.fov) > 0.5) {
      const delta = (this.settings.fov - currentFOV) / config.camera.fovSensitivity;
      this.sceneManager.updateFOV(delta);
    }

    // Apply clipping planes
    this.sceneManager.updateClippingPlanes(this.settings.near, this.settings.far);

    // Apply navigation control settings
    this.sceneManager.setControlType(this.settings.controlType);
    this.sceneManager.setAutoRotate(this.settings.autoRotate);
    this.sceneManager.setAutoRotateSpeed(this.settings.autoRotateSpeed);
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

    // Route zarr overrides through validateRenderingSettings so a
    // corrupted viewer_config can't inject NaN/Infinity/out-of-range
    // values into runtime rendering state. Validation clamps to defaults.
    const zarrOverrides = extractRenderingOverrides(this.zarrViewerConfig);
    const validated = validateRenderingSettings({ ...this.settings, ...zarrOverrides });
    Object.assign(this.settings, validated);

    // Apply FOV if overridden
    if (zarrOverrides.fov !== undefined) {
      const currentFOV = this.sceneManager.currentFov;
      if (Math.abs(currentFOV - this.settings.fov) > 0.5) {
        const delta = (this.settings.fov - currentFOV) / config.camera.fovSensitivity;
        this.sceneManager.updateFOV(delta);
      }
    }

    // Apply clipping planes if overridden
    if (zarrOverrides.near !== undefined || zarrOverrides.far !== undefined) {
      this.sceneManager.updateClippingPlanes(this.settings.near, this.settings.far);
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
    if (zarrOverrides.naturalDrag !== undefined) {
      this.sceneManager.setNaturalDrag(this.settings.naturalDrag);
    }

    // Update GUI controllers to reflect new values
    this.gui.controllersRecursive().forEach((controller) => {
      controller.updateDisplay();
    });

    // Sync HDR log slider
    // Apply post-processing and other rendering settings
    this.applySettings();
    log.info(Modules.RENDERER, 'Applied viewer config defaults from zarr');
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

    // Apply the persisted adaptive-DPR enabled state to the manager. The
    // Performance rail popover self-syncs from the manager when opened.
    this.adaptiveDPRManager?.setEnabled(this.settings.adaptiveDPREnabled);

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
   * Setup auto-blur for all GUI controls
   *
   * NOTE: The custom GUI library now handles auto-blur internally via
   * src/ui/gui/format/auto-blur.ts. This method is kept for backwards
   * compatibility but does nothing - all auto-blur logic is in the GUI library.
   */
  private setupAutoBlur(): void {
    // Auto-blur is now handled by the custom GUI library's applyAutoBlur() utility
    // No additional setup needed here
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
