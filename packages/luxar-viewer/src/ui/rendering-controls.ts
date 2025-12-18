// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI, { Folder } from './gui';
import * as THREE from 'three';
import { PostProcessingManager } from '../rendering/post-processing-manager';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { config, type RenderingSettings } from '../config';
import { ThemeManager } from '../themes/theme-manager';

import type { RenderingControllers } from '../controls/types';
import { isOrbitControls } from '../controls/types';
import {
  generateSettingsKey,
  serializeSettings,
  deserializeSettings,
} from './rendering-controls-utils';
import { log, Modules } from '../utils/log';
import { setupNavigationControls } from './rendering-controls/navigation-setup';
import { setupCameraControls } from './rendering-controls/camera-setup';
import { setupHDRControls } from './rendering-controls/hdr-setup';
import { setupAntiAliasingControls } from './rendering-controls/anti-aliasing-setup';
import { setupPostProcessingControls } from './rendering-controls/post-processing-setup';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';

/**
 * Advanced rendering parameters GUI for real-time visual control.
 *
 * Provides comprehensive UI for controlling:
 * - Post-processing effects (bloom, noise, vignette, chromatic aberration, lens distortion)
 * - HDR intensity and tone mapping
 * - Anti-aliasing options (FXAA, SMAA, MSAA, SSAA)
 * - Camera controls (orbit, arcball, fly modes with physics parameters)
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

  /** Current rendering settings */
  private settings: RenderingSettings;

  /** Scene identifier for settings persistence */
  private sceneId: string = '';

  /** Reference to post-processing manager */
  private postProcessing: PostProcessingManager;

  /** Reference to scene manager */
  private sceneManager: SceneManager;

  /** Reference to animation controller for triggering re-renders */
  private animationController?: AnimationController;

  /** Visibility state */
  private visible: boolean = false;

  /** References to GUI controllers for updates */
  private controllers: RenderingControllers = {};

  /** Folder references for visibility control */
  private orbitFolder?: Folder;
  private flyFolder?: Folder;

  /** Shadow object for logarithmic HDR intensity slider */
  private hdrLogValue: { log: number } = { log: 0 };

  /** Reference to adaptive DPR manager for performance controls */
  private adaptiveDPRManager?: AdaptiveDPRManager;

  /** Callback to update adaptive DPR control visibility (set by setAdaptiveDPRManager) */
  private updateAdaptiveDPRVisibility?: (enabled: boolean) => void;

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
    this.settings = {
      ...config.renderingControls.defaults,
      // Add fly control defaults from config.controls.fly
      flyMovementSpeed: config.controls.fly.movement.speed.default,
      flyRotationSpeed: config.controls.fly.rotation.speed.default,
      flyInertialMode: config.controls.fly.inertialMode.default,
      flyDamping: config.controls.fly.movement.damping.default,
      flyRotationDamping: config.controls.fly.rotation.damping.default,
    };

    // Initialize GUI with close button callback
    this.gui = new GUI({
      title: 'Rendering Controls',
      width: 300,
      closeFolders: false,
      onClose: () => this.hide(),
    });

    // Position on the left side with standard margins
    this.gui.domElement.style.position = 'fixed';
    this.gui.domElement.style.top = '20px'; // Standard 20px margin
    this.gui.domElement.style.left = '20px'; // Standard 20px margin
    this.gui.domElement.style.zIndex = String(config.ui.zIndex.renderingControls);

    // Start hidden
    this.gui.hide();

    // Custom styling now in src/styles/components/rendering-controls.css

    this.setupControls();

    // Add keyboard event listener to the GUI container to handle 'R' key
    // This ensures the panel can be closed even when a control has focus
    this.setupKeyboardHandling();
  }

  /**
   * Setup all GUI controls
   */
  private setupControls(): void {
    // Setup auto-blur for all controls
    this.setupAutoBlur();

    // Navigation controls
    const navigationResult = setupNavigationControls({
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
    Object.assign(this.controllers, navigationResult.controllers);

    // Store folder references
    if (navigationResult.folders) {
      this.orbitFolder = navigationResult.folders.orbitFolder;
      this.flyFolder = navigationResult.folders.flyFolder;
    }

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
    const hdrResult = setupHDRControls(
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
      this.hdrLogValue
    );

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

    // Theme selector
    this.setupThemeControls();

    // Note: Reset to Defaults button is added at the end of setAdaptiveDPRManager
    // to ensure it appears after the Performance folder
  }

  /**
   * Setup theme controls
   */
  private setupThemeControls(): void {
    const themeFolder = this.gui.addFolder('🎨 Theme');

    const themeManager = ThemeManager.getInstance();
    const themes = themeManager.getAllThemes();

    // Create theme options object { 'Dark Theme': 'dark', 'Light Theme': 'light', ... }
    const themeOptions = themes.reduce(
      (acc, theme) => {
        acc[theme.name] = theme.id;
        return acc;
      },
      {} as Record<string, string>
    );

    const themeSettings = {
      theme: themeManager.getCurrentTheme().id,
    };

    const themeControl = themeFolder
      .add(themeSettings, 'theme', themeOptions)
      .name('Active Theme')
      .onChange((themeId: string) => {
        themeManager.setTheme(themeId);
        // Theme is persisted automatically by ThemeManager
        log.info(Modules.RENDERER, `Theme changed to: ${themeId}`);

        // Trigger re-render to apply theme changes immediately
        this.triggerAnimation();
      });

    themeControl.domElement.setAttribute(
      'title',
      'Switch between visual themes\n' +
        '• Dark: Default scientific visualization theme\n' +
        '• Light: Bright theme for well-lit environments\n' +
        '• Frosted Glass: Subtle translucent glassmorphism\n' +
        '• Liquid Glass: True glass effect with inner glow and tint'
    );

    // Close folder by default
    themeFolder.close();
  }

  /**
   * Reset all rendering settings to their default values
   */
  private resetToDefaults(): void {
    // Get fresh defaults from config, including fly control defaults
    const defaults = {
      ...config.renderingControls.defaults,
      flyMovementSpeed: config.controls.fly.movement.speed.default,
      flyRotationSpeed: config.controls.fly.rotation.speed.default,
      flyInertialMode: config.controls.fly.inertialMode.default,
      flyDamping: config.controls.fly.movement.damping.default,
      flyRotationDamping: config.controls.fly.rotation.damping.default,
    };

    // Update settings object in place to maintain GUI bindings
    Object.assign(this.settings, defaults);

    // Sync logarithmic HDR slider shadow value
    this.hdrLogValue.log = Math.log10(this.settings.hdrMultiplier);

    // Clear saved settings for this scene (before applying, so user sees clean state)
    if (this.sceneId) {
      const key = generateSettingsKey(this.sceneId);
      localStorage.removeItem(key);
    }

    // Apply camera settings to scene manager (before post-processing)
    const currentFOV = this.sceneManager.camera.fov;
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
    this.sceneManager.setFlyMovementSpeed(defaults.flyMovementSpeed);
    this.sceneManager.setFlyRotationSpeed(defaults.flyRotationSpeed);
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

    // Update folder visibility based on control type
    this.updateNavigationControls(this.settings.controlType);

    // Update clipping controls state
    this.updateClippingControlsState(this.settings.dynamicClippingEnabled);

    // Show/hide adapt speed based on dynamic clipping
    if (this.controllers.clippingAdaptSpeed) {
      if (this.settings.dynamicClippingEnabled) {
        this.controllers.clippingAdaptSpeed.show();
      } else {
        this.controllers.clippingAdaptSpeed.hide();
      }
    }

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
   * Set the adaptive DPR manager reference and create performance controls.
   *
   * Creates a "Performance" folder with:
   * - Adaptive Resolution toggle (enables/disables dynamic DPR scaling)
   * - Current DPR display (read-only, shows current pixel ratio)
   *
   * @param manager - The AdaptiveDPRManager instance
   */
  setAdaptiveDPRManager(manager: AdaptiveDPRManager): void {
    this.adaptiveDPRManager = manager;

    // Create Performance folder with adaptive DPR controls
    const performanceFolder = this.gui.addFolder('⚡ Performance');

    // Adaptive Resolution toggle (onChange registered below after manual DPR control is created)
    const adaptiveToggle = performanceFolder
      .add(this.settings, 'adaptiveDPREnabled')
      .name('Adaptive Resolution');

    adaptiveToggle.domElement.setAttribute(
      'title',
      'Adaptive Resolution: Automatically adjusts rendering quality for smooth FPS\n' +
        '• When FPS drops below 50, reduces pixel ratio\n' +
        '• Gradually restores quality when FPS stabilizes above 58\n' +
        '• Minimum DPR: 0.5 (50% of native resolution)'
    );

    // Manual DPR control (shown when adaptive is OFF)
    const nativeDPR = manager.getNativeDPR();
    const manualDPRSettings = { dpr: nativeDPR };

    const manualDPRControl = performanceFolder
      .add(manualDPRSettings, 'dpr', 0.25, nativeDPR, 0.05)
      .name('Manual DPR')
      .onChange((value: number) => {
        if (this.adaptiveDPRManager && !this.settings.adaptiveDPREnabled) {
          this.adaptiveDPRManager.setManualDPR(value);
          this.triggerAnimation();
        }
      });

    manualDPRControl.domElement.setAttribute(
      'title',
      'Manual Device Pixel Ratio (when adaptive is off)\n' +
        `• Native: ${nativeDPR.toFixed(2)}\n` +
        '• Lower values = better performance, less sharpness\n' +
        '• 1.0 = 100% resolution, 0.5 = 50% resolution'
    );

    // Create simple text displays for DPR and FPS (read-only info, shown when adaptive is ON)
    const createDisplayRow = (label: string, tooltip: string): HTMLElement => {
      const row = document.createElement('div');
      row.className = 'luxar-gui__controller';
      row.style.opacity = '0.7';
      row.setAttribute('title', tooltip);

      const nameEl = document.createElement('div');
      nameEl.className = 'luxar-gui__controller-name';
      nameEl.textContent = label;

      const valueEl = document.createElement('div');
      valueEl.className = 'luxar-gui__controller-widget';
      valueEl.style.textAlign = 'right';
      valueEl.style.paddingRight = '8px';
      valueEl.style.fontFamily = 'monospace';

      row.appendChild(nameEl);
      row.appendChild(valueEl);
      return row;
    };

    const dprRow = createDisplayRow(
      'Current DPR',
      `Current Device Pixel Ratio\n• Native: ${nativeDPR.toFixed(2)}\n• Lower values = better performance, less sharpness`
    );
    const dprValue = dprRow.querySelector('.luxar-gui__controller-widget') as HTMLElement;

    const fpsRow = createDisplayRow(
      'Current FPS',
      'Current Frames Per Second\n• Target: 55-60 FPS\n• Scales down if below 50 FPS'
    );
    const fpsValue = fpsRow.querySelector('.luxar-gui__controller-widget') as HTMLElement;

    // Access folder's children container to append display rows
    const folderEl = performanceFolder.domElement;
    if (folderEl) {
      const childrenContainer = folderEl.querySelector('.luxar-gui__children');
      if (childrenContainer) {
        childrenContainer.appendChild(dprRow);
        childrenContainer.appendChild(fpsRow);
      }
    }

    // Helper function to update visibility of controls based on adaptive state
    const updateControlVisibility = (adaptiveEnabled: boolean) => {
      if (adaptiveEnabled) {
        // Adaptive ON: show display rows, hide manual control
        manualDPRControl.hide();
        dprRow.style.display = '';
        fpsRow.style.display = '';
        // Update display values immediately
        if (this.adaptiveDPRManager) {
          const state = this.adaptiveDPRManager.getState();
          dprValue.textContent = state.currentDPR.toFixed(2);
          fpsValue.textContent = Math.round(state.currentFPS).toString();
        }
      } else {
        // Adaptive OFF: show manual control, hide display rows
        manualDPRControl.show();
        // Sync manual DPR slider with current value
        manualDPRSettings.dpr = this.adaptiveDPRManager?.getCurrentDPR() ?? nativeDPR;
        manualDPRControl.updateDisplay();
        dprRow.style.display = 'none';
        fpsRow.style.display = 'none';
      }
    };

    // Store visibility update callback for use by loadSettings
    this.updateAdaptiveDPRVisibility = updateControlVisibility;

    // Sync initial state from manager BEFORE setting visibility
    this.settings.adaptiveDPREnabled = manager.isActive();
    adaptiveToggle.updateDisplay();

    // Set initial visibility based on synced state
    updateControlVisibility(this.settings.adaptiveDPREnabled);

    // Register onChange handler for adaptive toggle
    adaptiveToggle.onChange((enabled: boolean) => {
      if (this.adaptiveDPRManager) {
        this.adaptiveDPRManager.setEnabled(enabled);
      }
      this.saveSettings();
      log.info(Modules.RENDERER, `Adaptive resolution ${enabled ? 'enabled' : 'disabled'}`);
      updateControlVisibility(enabled);
    });

    // Update displays periodically (only when adaptive is enabled)
    const updateInterval = setInterval(() => {
      if (this.adaptiveDPRManager && this.settings.adaptiveDPREnabled) {
        const state = this.adaptiveDPRManager.getState();
        dprValue.textContent = state.currentDPR.toFixed(2);
        fpsValue.textContent = Math.round(state.currentFPS).toString();
      }
    }, 500);

    // Store cleanup reference
    const originalDispose = this.dispose.bind(this);
    this.dispose = () => {
      clearInterval(updateInterval);
      originalDispose();
    };

    // Close folder by default
    performanceFolder.close();

    // Store controller references
    this.controllers.adaptiveDPREnabled = adaptiveToggle;

    // Reset to Defaults button at root level (added last to appear at bottom)
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
    const currentFOV = this.sceneManager.camera.fov;
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

    // Apply fly control settings (if they exist in loaded settings)
    if (this.settings.flyMovementSpeed !== undefined) {
      this.sceneManager.setFlyMovementSpeed(this.settings.flyMovementSpeed);
    }
    if (this.settings.flyRotationSpeed !== undefined) {
      this.sceneManager.setFlyRotationSpeed(this.settings.flyRotationSpeed);
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
   * Trigger animation when parameters change
   */
  private triggerAnimation(): void {
    // Start animation to see changes immediately
    this.animationController?.startAnimation();
  }

  /**
   * Update clipping controls state based on dynamic clipping setting
   * When dynamic clipping is enabled, grey out manual near/far controls
   */
  private updateClippingControlsState(dynamicEnabled: boolean): void {
    const opacity = dynamicEnabled ? '0.5' : '1.0';
    const pointerEvents = dynamicEnabled ? 'none' : 'auto';

    if (this.controllers.nearPlane) {
      const container = this.controllers.nearPlane.domElement.closest('.controller');
      if (container instanceof HTMLElement) {
        container.style.opacity = opacity;
        container.style.pointerEvents = pointerEvents;
      }
    }

    if (this.controllers.farPlane) {
      const container = this.controllers.farPlane.domElement.closest('.controller');
      if (container instanceof HTMLElement) {
        container.style.opacity = opacity;
        container.style.pointerEvents = pointerEvents;
      }
    }
  }

  /**
   * Update navigation controls visibility based on control type
   */
  private updateNavigationControls(controlType: 'orbit' | 'arcball' | 'fly'): void {
    const orbitFolder = this.orbitFolder;
    const flyFolder = this.flyFolder;

    if (controlType === 'orbit' || controlType === 'arcball') {
      // Show orbit folder for both orbit and arcball (they share similar settings)
      if (orbitFolder) {
        orbitFolder.show();
        orbitFolder.open();
      }
      if (flyFolder) {
        flyFolder.close();
        flyFolder.hide();
      }

      // Hide auto-rotate controls for arcball mode (not supported)
      if (controlType === 'arcball') {
        if (this.controllers.autoRotate) {
          this.controllers.autoRotate.domElement.parentElement?.parentElement?.style.setProperty(
            'display',
            'none'
          );
        }
        if (this.controllers.autoRotateSpeed) {
          this.controllers.autoRotateSpeed.domElement.parentElement?.parentElement?.style.setProperty(
            'display',
            'none'
          );
        }
      } else {
        // Show auto-rotate controls for orbit mode
        if (this.controllers.autoRotate) {
          this.controllers.autoRotate.domElement.parentElement?.parentElement?.style.removeProperty(
            'display'
          );
        }
        if (this.controllers.autoRotateSpeed) {
          this.controllers.autoRotateSpeed.domElement.parentElement?.parentElement?.style.removeProperty(
            'display'
          );
        }
      }
    } else {
      // Show fly folder for fly controls
      if (orbitFolder) {
        orbitFolder.close();
        orbitFolder.hide();
      }
      if (flyFolder) {
        flyFolder.show();
        flyFolder.open();
      }
    }
  }

  /**
   * Save current settings to localStorage
   */
  private saveSettings(): void {
    if (!this.sceneId) return;

    const key = generateSettingsKey(this.sceneId);
    localStorage.setItem(key, serializeSettings(this.settings));
  }

  /**
   * Load settings from localStorage
   */
  private loadSettings(): void {
    if (!this.sceneId) return;

    const key = generateSettingsKey(this.sceneId);
    const stored = localStorage.getItem(key);

    if (stored) {
      const loadedSettings = deserializeSettings(stored);
      if (loadedSettings) {
        // Update settings properties IN PLACE to maintain GUI controller bindings
        // This is critical - replacing the entire settings object breaks the GUI bindings
        Object.assign(this.settings, {
          ...config.renderingControls.defaults,
          // Add fly control defaults from config.controls.fly
          flyMovementSpeed: config.controls.fly.movement.speed.default,
          flyRotationSpeed: config.controls.fly.rotation.speed.default,
          flyInertialMode: config.controls.fly.inertialMode.default,
          flyDamping: config.controls.fly.movement.damping.default,
          flyRotationDamping: config.controls.fly.rotation.damping.default,
          ...loadedSettings,
        });

        // Sync logarithmic HDR slider shadow value
        this.hdrLogValue.log = Math.log10(this.settings.hdrMultiplier);

        // Update GUI to reflect loaded values
        // Note: HDR controller's updateDisplay is overridden to show actual intensity
        this.gui.controllersRecursive().forEach((controller) => {
          controller.updateDisplay();
        });

        // Sync adaptive DPR manager state and update visibility
        if (this.adaptiveDPRManager && this.updateAdaptiveDPRVisibility) {
          this.adaptiveDPRManager.setEnabled(this.settings.adaptiveDPREnabled);
          this.updateAdaptiveDPRVisibility(this.settings.adaptiveDPREnabled);
        }

        log.info(Modules.RENDERER, `Loaded rendering settings for scene: ${this.sceneId}`);
      } else {
        log.warning(Modules.RENDERER, 'Failed to parse rendering settings');
      }
    }
  }

  /**
   * Sync current state from scene manager
   * This ensures the GUI reflects the actual state when opened
   */
  public syncCurrentState(): void {
    // Sync camera settings
    this.settings.fov = this.sceneManager.camera.fov;
    this.settings.near = this.sceneManager.camera.near;
    this.settings.far = this.sceneManager.camera.far;

    // Check if current FOV matches any preset
    const currentPreset = Object.entries(config.camera.fovPresets).find(
      ([_, fovValue]) => fovValue > 0 && Math.abs(fovValue - this.settings.fov) < 0.5
    );
    this.settings.fovPreset = (
      currentPreset ? currentPreset[0] : 'Custom'
    ) as typeof this.settings.fovPreset;

    // Get current control type
    const currentControlType = this.sceneManager.controls.getControlType();
    this.settings.controlType = currentControlType;

    // Get current controls instance
    const controls = this.sceneManager.controls.getControls();

    // Always get fly controls config from ControlsManager
    // This ensures settings persist even when in orbit mode
    const flyConfig = this.sceneManager.controls.getFlyConfig();
    this.settings.flyInertialMode = flyConfig.inertialMode;
    this.settings.flyMovementSpeed = flyConfig.movementSpeed;
    this.settings.flyRotationSpeed = flyConfig.rotationSpeed;
    this.settings.flyDamping = flyConfig.damping;
    this.settings.flyRotationDamping = flyConfig.rotationDamping;

    // Update orbit controls state using type guard
    if (isOrbitControls(controls)) {
      this.settings.autoRotate = controls.autoRotate;
      this.settings.autoRotateSpeed = controls.autoRotateSpeed;
    }

    // Update specific controllers that we have references to
    if (this.controllers.controlType) {
      this.controllers.controlType.setValue(currentControlType);
      this.controllers.controlType.updateDisplay();
    }

    if (this.controllers.flyInertialMode) {
      this.controllers.flyInertialMode.setValue(this.settings.flyInertialMode);
      this.controllers.flyInertialMode.updateDisplay();
    }

    if (this.controllers.flyMovementSpeed) {
      this.controllers.flyMovementSpeed.setValue(this.settings.flyMovementSpeed);
      this.controllers.flyMovementSpeed.updateDisplay();
    }

    if (this.controllers.flyRotationSpeed) {
      this.controllers.flyRotationSpeed.setValue(this.settings.flyRotationSpeed);
      this.controllers.flyRotationSpeed.updateDisplay();
    }

    if (this.controllers.flyDamping) {
      this.controllers.flyDamping.setValue(this.settings.flyDamping);
      this.controllers.flyDamping.updateDisplay();
      // Show/hide damping based on inertial mode
      if (this.settings.flyInertialMode) {
        this.controllers.flyDamping.show();
      } else {
        this.controllers.flyDamping.hide();
      }
    }

    if (this.controllers.flyRotationDamping) {
      this.controllers.flyRotationDamping.setValue(this.settings.flyRotationDamping);
      this.controllers.flyRotationDamping.updateDisplay();
      // Show/hide rotation damping based on inertial mode
      if (this.settings.flyInertialMode) {
        this.controllers.flyRotationDamping.show();
      } else {
        this.controllers.flyRotationDamping.hide();
      }
    }

    if (this.controllers.autoRotate) {
      this.controllers.autoRotate.setValue(this.settings.autoRotate);
      this.controllers.autoRotate.updateDisplay();
    }

    if (this.controllers.autoRotateSpeed) {
      this.controllers.autoRotateSpeed.setValue(this.settings.autoRotateSpeed);
      this.controllers.autoRotateSpeed.updateDisplay();
    }

    if (this.controllers.fov) {
      this.controllers.fov.setValue(this.settings.fov);
      this.controllers.fov.updateDisplay();
    }

    if (this.controllers.fovPreset) {
      this.controllers.fovPreset.setValue(this.settings.fovPreset);
      this.controllers.fovPreset.updateDisplay();
    }

    if (this.controllers.nearPlane) {
      this.controllers.nearPlane.setValue(this.settings.near);
      this.controllers.nearPlane.updateDisplay();
    }

    if (this.controllers.farPlane) {
      this.controllers.farPlane.setValue(this.settings.far);
      this.controllers.farPlane.updateDisplay();
    }

    // Sync dynamic clipping state from scene manager
    const dynamicClippingState = this.sceneManager.getDynamicClippingState();
    this.settings.dynamicClippingEnabled = dynamicClippingState.enabled;
    this.settings.clippingAdaptSpeed = dynamicClippingState.adaptSpeed;

    if (this.controllers.dynamicClippingEnabled) {
      this.controllers.dynamicClippingEnabled.setValue(this.settings.dynamicClippingEnabled);
      this.controllers.dynamicClippingEnabled.updateDisplay();
    }

    if (this.controllers.clippingAdaptSpeed) {
      this.controllers.clippingAdaptSpeed.setValue(this.settings.clippingAdaptSpeed);
      this.controllers.clippingAdaptSpeed.updateDisplay();
      // Show/hide adapt speed based on dynamic clipping state
      if (this.settings.dynamicClippingEnabled) {
        this.controllers.clippingAdaptSpeed.show();
      } else {
        this.controllers.clippingAdaptSpeed.hide();
      }
    }

    // Update near/far control state based on dynamic clipping
    this.updateClippingControlsState(this.settings.dynamicClippingEnabled);

    // Sync logarithmic HDR slider
    // The shadow log value must be updated to match the actual hdrMultiplier
    this.hdrLogValue.log = Math.log10(this.settings.hdrMultiplier);
    if (this.controllers.hdrMultiplier) {
      // updateDisplay is overridden to show actual intensity value
      this.controllers.hdrMultiplier.updateDisplay();
    }

    // Update all other controllers
    this.gui.controllersRecursive().forEach((controller) => {
      controller.updateDisplay();
    });

    // Update folder visibility based on current control type
    this.updateNavigationControls(currentControlType);
  }

  /**
   * Apply current settings to rendering pipeline
   */
  private applySettings(): void {
    // Apply bloom settings
    this.postProcessing.updateBloomSettings(
      this.settings.bloomStrength,
      this.settings.bloomRadius,
      this.settings.bloomThreshold
    );
    this.postProcessing.setBloomLevels(this.settings.bloomLevels);

    // Apply HDR multiplier - must update both config AND materials
    // HDR multiplier is now handled through material manager
    this.sceneManager.updateHDRMultiplier(this.settings.hdrMultiplier);

    // Apply SSAA settings
    this.postProcessing.setSSAAEnabled(this.settings.ssaaEnabled);
    this.postProcessing.setSSAAMultiplier(this.settings.ssaaMultiplier);

    // Apply FXAA setting
    this.postProcessing.setFXAAEnabled(this.settings.fxaaEnabled);

    // Apply MSAA settings
    this.postProcessing.setMSAAEnabled(this.settings.msaaEnabled);
    this.postProcessing.setMSAASamples(this.settings.msaaSamples);

    // Apply SMAA settings
    this.postProcessing.setSMAAEnabled(this.settings.smaaEnabled);
    if (this.settings.smaaEnabled) {
      this.postProcessing.updateSMAASettings();
    }

    // Apply tone mapping
    const toneMappingMap: { [key: string]: THREE.ToneMapping } = {
      None: THREE.NoToneMapping,
      Linear: THREE.LinearToneMapping,
      Reinhard: THREE.ReinhardToneMapping,
      Cineon: THREE.CineonToneMapping,
      ACES: THREE.ACESFilmicToneMapping,
      AgX: THREE.AgXToneMapping,
      Neutral: THREE.NeutralToneMapping,
    };
    this.postProcessing.setToneMapping(toneMappingMap[this.settings.toneMapping]);

    // Apply DOF settings
    this.postProcessing.setDOF(
      this.settings.dofEnabled,
      this.settings.dofFocus,
      this.settings.dofStrength
    );

    // Apply chromatic aberration
    this.postProcessing.setChromaticAberration(
      this.settings.chromaticAberrationEnabled,
      this.settings.chromaticAberrationStrength
    );

    // Apply detector noise effect
    this.postProcessing.setDetectorNoiseEnabled(
      this.settings.detectorNoiseEnabled,
      this.settings.detectorNoiseReadoutSigma,
      this.settings.detectorNoisePhotonGain,
      this.settings.detectorNoiseFpnSigma
    );

    // Start animation if detector noise is enabled (from loaded settings)
    if (this.settings.detectorNoiseEnabled) {
      this.animationController?.startAnimation();
    }

    // Apply vignette effect
    this.postProcessing.setVignetteEnabled(
      this.settings.vignetteEnabled,
      this.settings.vignetteDarkness,
      this.settings.vignetteOffset
    );

    // Apply lens distortion effect
    this.postProcessing.setLensDistortionEnabled(
      this.settings.lensDistortionEnabled,
      this.settings.lensDistortionX,
      this.settings.lensDistortionY,
      this.settings.lensPrincipalPointX,
      this.settings.lensPrincipalPointY,
      this.settings.lensFocalLengthX,
      this.settings.lensFocalLengthY,
      this.settings.lensSkew
    );

    // Apply ambient occlusion (always call to ensure proper enable/disable)
    this.postProcessing.setAOEnabled(this.settings.aoEnabled, this.settings.aoQuality);

    // Apply dynamic clipping settings
    this.sceneManager.setDynamicClipping(
      this.settings.dynamicClippingEnabled,
      this.settings.clippingAdaptSpeed
    );

    // Update near/far control state based on dynamic clipping
    this.updateClippingControlsState(this.settings.dynamicClippingEnabled);

    // Trigger render to ensure changes are visible
    this.triggerAnimation();
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
    // Sync current state from scene manager before showing
    this.syncCurrentState();

    this.gui.show();
    this.visible = true;

    // Add click handler to auto-blur inputs when clicking outside them
    // This helps prevent focus getting stuck
    setTimeout(() => {
      this.addClickOutsideHandler();
    }, 100);
  }

  /**
   * Hide the rendering controls panel.
   *
   * Blurs any focused input element to return focus to canvas, ensuring
   * keyboard shortcuts work after closing. Removes click-outside handler.
   *
   * Triggered by R key when controls are visible, or by Escape key.
   */
  hide(): void {
    // Blur any focused element to return focus to the main document
    // This ensures keyboard shortcuts work after closing the panel
    const activeElement = document.activeElement as HTMLElement;
    if (activeElement && activeElement.blur) {
      activeElement.blur();
    }

    // Remove click outside handler
    this.removeClickOutsideHandler();

    this.gui.hide();
    this.visible = false;

    // Focus the canvas to ensure keyboard events work
    this.sceneManager.renderer.domElement.focus();
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
   * src/ui/gui/utils/auto-blur.ts. This method is kept for backwards
   * compatibility but does nothing - all auto-blur logic is in the GUI library.
   */
  private setupAutoBlur(): void {
    // Auto-blur is now handled by the custom GUI library's applyAutoBlur() utility
    // No additional setup needed here
  }

  /**
   * Setup keyboard handling for the GUI panel
   */
  private setupKeyboardHandling(): void {
    // Add keydown listener to the GUI's DOM element
    // Note: We don't stopPropagation() for toggle keys so they can be handled globally
    this.gui.domElement.addEventListener('keydown', (event: KeyboardEvent) => {
      // For toggle keys (R), don't stopPropagation so main handler can process it
      if (
        (event.key === 'r' || event.key === 'R') &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.shiftKey
      ) {
        // Don't prevent the event from bubbling up
        // The main input handler will toggle the panel properly
      }
      // For Escape, also let it bubble up for proper priority handling
      else if (event.key === 'Escape') {
        // Don't prevent the event from bubbling up
      }
    });
  }

  /**
   * Add click outside handler to blur inputs
   */
  private clickOutsideHandler?: (e: MouseEvent) => void;

  private addClickOutsideHandler(): void {
    if (this.clickOutsideHandler) return;

    this.clickOutsideHandler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      // If clicking outside the GUI panel, blur any focused element within it
      if (!this.gui.domElement.contains(target)) {
        const activeElement = document.activeElement as HTMLElement;
        if (activeElement && activeElement.blur && this.gui.domElement.contains(activeElement)) {
          activeElement.blur();
          // Also focus the canvas for good measure
          this.sceneManager.renderer.domElement.focus();
        }
      }
    };

    // Use capture phase to ensure we get the event first
    document.addEventListener('mousedown', this.clickOutsideHandler, true);
  }

  private removeClickOutsideHandler(): void {
    if (this.clickOutsideHandler) {
      document.removeEventListener('mousedown', this.clickOutsideHandler, true);
      this.clickOutsideHandler = undefined;
    }
  }

  /**
   * Toggle cinematic mode (film-like visual preset).
   *
   * Triggered by C key. Uses intelligent majority-vote algorithm to determine
   * whether to enable or disable effects:
   * - If < 50% effects enabled: Turn ALL on
   * - If >= 50% effects enabled: Turn ALL off
   *
   * Cinematic mode affects:
   * - Detector noise (subtle film grain)
   * - Vignette (darkened corners)
   * - Chromatic aberration (color fringing)
   * - Lens distortion (barrel/pincushion)
   * - FOV (35mm wide-angle for cinematic, 50mm normal for regular)
   *
   * Uses deferred rebuild to apply all changes in single pass (performance).
   *
   * @example
   * ```typescript
   * // User presses C key
   * renderingControls.toggleCinematicMode();
   * // All cinematic effects either turn on or off together
   *
   * // Check resulting state
   * console.log('Cinematic:', renderingControls.settings.detectorNoiseEnabled);
   * ```
   */
  toggleCinematicMode(): void {
    // Get current state of cinematic effects
    const cinematicEffects = [
      this.settings.detectorNoiseEnabled,
      this.settings.vignetteEnabled,
      this.settings.chromaticAberrationEnabled,
      this.settings.lensDistortionEnabled,
    ];

    // Count how many effects are currently enabled
    const enabledCount = cinematicEffects.filter(Boolean).length;
    const totalEffects = cinematicEffects.length;

    // Use majority vote to decide direction (>= 50% enabled = turn all off, < 50% = turn all on)
    const shouldEnableAll = enabledCount < totalEffects / 2;

    // Apply cinematic mode settings
    this.settings.detectorNoiseEnabled = shouldEnableAll;
    this.settings.vignetteEnabled = shouldEnableAll;
    this.settings.chromaticAberrationEnabled = shouldEnableAll;
    this.settings.lensDistortionEnabled = shouldEnableAll;

    // Set cinematic detector noise parameters when turning ON cinematic mode
    // Uses subtle physics-based noise for film-like look
    if (shouldEnableAll) {
      this.settings.detectorNoiseReadoutSigma = 0.002; // Subtle temporal noise
      this.settings.detectorNoisePhotonGain = 0.002; // Low shot noise
      this.settings.detectorNoiseFpnSigma = 0.001; // Subtle fixed pattern
    }

    // FOV switching: 35mm for cinematic, 50mm Normal for regular
    const targetFOV = shouldEnableAll
      ? config.camera.fovPresets['35mm'] // 63° - Wide angle for cinematic
      : config.camera.fovPresets['50mm Normal']; // 47° - Normal for regular use

    this.settings.fov = targetFOV;
    this.settings.fovPreset = shouldEnableAll ? '35mm' : '50mm Normal';

    // Apply appropriate lens distortion preset when enabling lens distortion in cinematic mode
    if (shouldEnableAll) {
      const lensPreset = config.camera.lensDistortionPresets['35mm'];
      this.settings.lensDistortionX = lensPreset.distortionX;
      this.settings.lensDistortionY = lensPreset.distortionY;
      this.settings.lensPrincipalPointX = lensPreset.principalPointX;
      this.settings.lensPrincipalPointY = lensPreset.principalPointY;
      this.settings.lensFocalLengthX = lensPreset.focalLengthX;
      this.settings.lensFocalLengthY = lensPreset.focalLengthY;
      this.settings.lensSkew = lensPreset.skew;
    } else {
      // Return to 50mm Normal lens distortion when disabling cinematic mode
      const lensPreset = config.camera.lensDistortionPresets['50mm Normal'];
      this.settings.lensDistortionX = lensPreset.distortionX;
      this.settings.lensDistortionY = lensPreset.distortionY;
      this.settings.lensPrincipalPointX = lensPreset.principalPointX;
      this.settings.lensPrincipalPointY = lensPreset.principalPointY;
      this.settings.lensFocalLengthX = lensPreset.focalLengthX;
      this.settings.lensFocalLengthY = lensPreset.focalLengthY;
      this.settings.lensSkew = lensPreset.skew;
    }

    // Apply the changes to post-processing using deferred rebuild to prevent multiple rebuilds
    this.postProcessing.startDeferRebuild();

    this.postProcessing.setDetectorNoiseEnabled(
      this.settings.detectorNoiseEnabled,
      this.settings.detectorNoiseReadoutSigma,
      this.settings.detectorNoisePhotonGain,
      this.settings.detectorNoiseFpnSigma
    );

    this.postProcessing.setVignetteEnabled(
      this.settings.vignetteEnabled,
      this.settings.vignetteDarkness,
      this.settings.vignetteOffset
    );

    this.postProcessing.setChromaticAberration(
      this.settings.chromaticAberrationEnabled,
      this.settings.chromaticAberrationStrength
    );

    this.postProcessing.setLensDistortionEnabled(
      this.settings.lensDistortionEnabled,
      this.settings.lensDistortionX,
      this.settings.lensDistortionY,
      this.settings.lensPrincipalPointX,
      this.settings.lensPrincipalPointY,
      this.settings.lensFocalLengthX,
      this.settings.lensFocalLengthY,
      this.settings.lensSkew
    );

    // End deferred mode and trigger single rebuild with all effects
    this.postProcessing.endDeferRebuild();

    // Apply FOV change to camera
    const currentFOV = this.sceneManager.camera.fov;
    if (Math.abs(currentFOV - targetFOV) > 0.5) {
      const delta = (targetFOV - currentFOV) / config.camera.fovSensitivity;
      this.sceneManager.updateFOV(delta);
    }

    // Update GUI to reflect new state
    this.gui.controllersRecursive().forEach((controller) => {
      controller.updateDisplay();
    });

    // Save settings and trigger animation
    this.saveSettings();
    this.triggerAnimation();

    // Start animation if detector noise is now enabled (requires continuous rendering)
    if (this.settings.detectorNoiseEnabled) {
      this.animationController?.startAnimation();
    }

    // Log the action
    const modeText = shouldEnableAll ? 'enabled' : 'disabled';
    const fovText = shouldEnableAll ? '35mm (63°)' : '50mm Normal (47°)';
    log.info(
      Modules.RENDERER,
      `Cinematic mode ${modeText}: detector noise=${shouldEnableAll}, vignette=${shouldEnableAll}, ` +
        `chromatic aberration=${shouldEnableAll}, lens distortion=${shouldEnableAll}, FOV=${fovText}`
    );
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
    // Auto-blur cleanup is now handled by the custom GUI library
    this.gui.destroy();
  }
}
