// Advanced rendering controls UI for the Luxar scene player
// Provides real-time control over post-processing and rendering parameters

import GUI from 'lil-gui';
import * as THREE from 'three';
import { PostProcessingManager } from '../rendering/post-processing-manager';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { config, type RenderingSettings } from '../config';

// Extract component configuration
const controlsConfig = config.ui.components.renderingControls;
const spacingConfig = config.ui.styles.spacing;
import type { RenderingControllers } from '../controls/types';
import { isOrbitControls } from '../controls/types';
import {
  generateSettingsKey,
  serializeSettings,
  deserializeSettings,
} from './rendering-controls-utils';
import { log, Modules } from '../utils/log';
import { setupNavigationControls } from './rendering-controls/navigation-setup';

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
 * - Clean collapsible UI using lil-gui library
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
  /** The lil-gui instance */
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
  private orbitFolder?: GUI;
  private flyFolder?: GUI;

  /** Shadow object for logarithmic HDR intensity slider */
  private hdrLogValue: { log: number } = { log: 0 };

  /**
   * Create rendering controls UI with complete parameter access.
   *
   * Initializes lil-gui with all post-processing and rendering controls
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

    // Initialize GUI
    this.gui = new GUI({
      title: 'Rendering Controls',
      width: 300,
      closeFolders: false,
    });

    // Position on the left side with standard margins
    this.gui.domElement.style.position = 'fixed';
    this.gui.domElement.style.top = '20px'; // Standard 20px margin
    this.gui.domElement.style.left = '20px'; // Standard 20px margin
    this.gui.domElement.style.zIndex = String(config.ui.zIndex.renderingControls);

    // Start hidden
    this.gui.hide();

    // Apply custom styling to match other panels
    this.applyCustomStyling();

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

    // Camera folder - for camera-specific settings
    const cameraFolder = this.gui.addFolder('Camera');
    cameraFolder.open();

    // FOV Preset dropdown
    const presetOptions = Object.keys(config.camera.fovPresets);
    const fovPresetControl = cameraFolder
      .add(this.settings, 'fovPreset', presetOptions)
      .name('FOV Preset')
      .onChange((presetName: string) => {
        const fovValue = config.camera.fovPresets[presetName];
        if (fovValue > 0) {
          // Apply preset FOV
          this.settings.fov = fovValue;

          // Calculate delta and apply to camera
          const currentFOV = this.sceneManager.camera.fov;
          const delta = (fovValue - currentFOV) / config.camera.fovSensitivity;
          this.sceneManager.updateFOV(delta);

          // Update FOV slider display
          if (this.controllers.fov) {
            this.controllers.fov.setValue(fovValue);
            this.controllers.fov.updateDisplay();
          }

          // Apply corresponding lens distortion preset (if lens distortion is enabled)
          const lensPreset = config.camera.lensDistortionPresets[presetName];
          if (lensPreset && this.settings.lensDistortionEnabled) {
            this.settings.lensDistortionX = lensPreset.distortionX;
            this.settings.lensDistortionY = lensPreset.distortionY;
            this.settings.lensPrincipalPointX = lensPreset.principalPointX;
            this.settings.lensPrincipalPointY = lensPreset.principalPointY;
            this.settings.lensFocalLengthX = lensPreset.focalLengthX;
            this.settings.lensFocalLengthY = lensPreset.focalLengthY;
            this.settings.lensSkew = lensPreset.skew;

            // Apply lens distortion changes
            this.postProcessing.updateLensDistortion({
              distortionX: lensPreset.distortionX,
              distortionY: lensPreset.distortionY,
              principalPointX: lensPreset.principalPointX,
              principalPointY: lensPreset.principalPointY,
              focalLengthX: lensPreset.focalLengthX,
              focalLengthY: lensPreset.focalLengthY,
              skew: lensPreset.skew,
            });

            // Update lens distortion UI controllers to reflect new values
            if (this.controllers.lensDistortionX) {
              this.controllers.lensDistortionX.setValue(lensPreset.distortionX);
              this.controllers.lensDistortionX.updateDisplay();
            }
            if (this.controllers.lensDistortionY) {
              this.controllers.lensDistortionY.setValue(lensPreset.distortionY);
              this.controllers.lensDistortionY.updateDisplay();
            }
            if (this.controllers.lensPrincipalPointX) {
              this.controllers.lensPrincipalPointX.setValue(lensPreset.principalPointX);
              this.controllers.lensPrincipalPointX.updateDisplay();
            }
            if (this.controllers.lensPrincipalPointY) {
              this.controllers.lensPrincipalPointY.setValue(lensPreset.principalPointY);
              this.controllers.lensPrincipalPointY.updateDisplay();
            }
            if (this.controllers.lensFocalLengthX) {
              this.controllers.lensFocalLengthX.setValue(lensPreset.focalLengthX);
              this.controllers.lensFocalLengthX.updateDisplay();
            }
            if (this.controllers.lensFocalLengthY) {
              this.controllers.lensFocalLengthY.setValue(lensPreset.focalLengthY);
              this.controllers.lensFocalLengthY.updateDisplay();
            }
            if (this.controllers.lensSkew) {
              this.controllers.lensSkew.setValue(lensPreset.skew);
              this.controllers.lensSkew.updateDisplay();
            }
          }
        }

        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.fovPreset = fovPresetControl;

    // Set tooltip for FOV presets
    fovPresetControl.domElement.setAttribute(
      'title',
      'FOV Preset: Professional camera lens equivalents (horizontal FOV)\n' +
        '• 28mm Wide (75°): Ultra-wide angle + barrel distortion\n' +
        '• 35mm (63°): Wide angle + moderate barrel distortion\n' +
        '• 50mm Normal (47°): Natural human vision + no distortion\n' +
        '• 85mm Portrait (29°): Telephoto + slight pincushion\n' +
        '• 135mm Tele (18°): Strong telephoto + pincushion distortion\n' +
        '• Custom: Manual FOV control via slider or Shift+Wheel\n' +
        '• Note: Also applies realistic lens distortion when enabled'
    );

    const fovControl = cameraFolder
      .add(this.settings, 'fov', config.camera.fovMin, config.camera.fovMax, 1)
      .name('Field of View')
      .onChange((value: number) => {
        // When FOV slider changes, switch to Custom preset
        this.settings.fovPreset = 'Custom';
        if (this.controllers.fovPreset) {
          this.controllers.fovPreset.setValue('Custom');
          this.controllers.fovPreset.updateDisplay();
        }

        // Calculate the delta needed to reach the target FOV
        const currentFOV = this.sceneManager.camera.fov;
        const targetFOV = value;
        const delta = (targetFOV - currentFOV) / config.camera.fovSensitivity;

        // Use the existing updateFOV method which handles bounds checking and material updates
        this.sceneManager.updateFOV(delta);

        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.fov = fovControl;

    // Set tooltip for FOV control
    fovControl.domElement.setAttribute(
      'title',
      'Field of View: Camera viewing angle in degrees\n' +
        '• Lower values: Telephoto lens effect (narrow view)\n' +
        '• Higher values: Wide-angle lens effect (broader view)\n' +
        '• 47° (50mm Normal) provides natural human-like viewing angle\n' +
        '• Also controllable with Shift+Wheel for fine adjustment\n' +
        '• Maintains world-space point sizing (points stay same physical size)'
    );

    // Clipping Planes sub-folder
    const clippingFolder = cameraFolder.addFolder('Clipping Planes');
    clippingFolder.close(); // Collapsed by default (advanced setting)

    const nearPlaneControl = clippingFolder
      .add(this.settings, 'near', 0.001, 10.0, 0.001)
      .name('Near Plane')
      .onChange((value: number) => {
        // Validate near plane is less than far plane
        if (value >= this.settings.far) {
          log.warning(Modules.RENDERER, 'Near plane must be less than far plane');
          return;
        }
        this.sceneManager.updateClippingPlanes(value, this.settings.far);
        this.saveSettings();
        this.triggerAnimation();
      });

    const farPlaneControl = clippingFolder
      .add(this.settings, 'far', 10, 10000, 1)
      .name('Far Plane')
      .onChange((value: number) => {
        // Validate far plane is greater than near plane
        if (value <= this.settings.near) {
          log.warning(Modules.RENDERER, 'Far plane must be greater than near plane');
          return;
        }
        this.sceneManager.updateClippingPlanes(this.settings.near, value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store references for updates
    this.controllers.nearPlane = nearPlaneControl;
    this.controllers.farPlane = farPlaneControl;

    // Set tooltips for clipping planes
    nearPlaneControl.domElement.setAttribute(
      'title',
      'Near Clipping Plane: Closest visible distance\n' +
        '• Objects closer than this are not rendered\n' +
        '• Lower values: See objects very close to camera\n' +
        '• Higher values: Better Z-buffer precision\n' +
        '• Too low can cause Z-fighting artifacts'
    );

    farPlaneControl.domElement.setAttribute(
      'title',
      'Far Clipping Plane: Furthest visible distance\n' +
        '• Objects further than this are not rendered\n' +
        '• Higher values: See distant objects\n' +
        '• Lower values: Better Z-buffer precision\n' +
        '• Keep near/far ratio under 10,000:1 for best precision'
    );

    // Dynamic clipping checkbox - auto-adjusts clipping planes each frame
    const dynamicClippingControl = clippingFolder
      .add(this.settings, 'dynamicClippingEnabled')
      .name('Dynamic Clipping')
      .onChange((value: boolean) => {
        this.sceneManager.setDynamicClipping(value, this.settings.clippingAdaptSpeed);
        this.saveSettings();
        this.triggerAnimation();

        // Enable/disable manual near/far controls
        this.updateClippingControlsState(value);

        // Show/hide adapt speed slider
        if (value) {
          adaptSpeedControl.show();
        } else {
          adaptSpeedControl.hide();
        }
      });

    // Store reference
    this.controllers.dynamicClippingEnabled = dynamicClippingControl;

    dynamicClippingControl.domElement.setAttribute(
      'title',
      'Dynamic Clipping: Auto-adjust clipping planes each frame\n' +
        '• Smoothly adapts to camera position and scene bounds\n' +
        '• Uses exponential smoothing for stable transitions\n' +
        '• Maximizes Z-buffer precision at all times\n' +
        '• When enabled, manual near/far controls are disabled'
    );

    // Adapt speed slider (only visible when dynamic clipping is enabled)
    const adaptSpeedControl = clippingFolder
      .add(this.settings, 'clippingAdaptSpeed', 0.01, 0.5, 0.01)
      .name('Adapt Speed')
      .onChange((value: number) => {
        this.sceneManager.setDynamicClipping(this.settings.dynamicClippingEnabled, value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference
    this.controllers.clippingAdaptSpeed = adaptSpeedControl;

    adaptSpeedControl.domElement.setAttribute(
      'title',
      'Adapt Speed: How quickly clipping planes adjust\n' +
        '• 0.01 = Very slow, smooth transitions\n' +
        '• 0.1 = Balanced responsiveness (default)\n' +
        '• 0.5 = Fast adaptation, may cause jitter\n' +
        '• Lower values = smoother but slower response'
    );

    // Initially show/hide adapt speed based on dynamic clipping state
    if (!this.settings.dynamicClippingEnabled) {
      adaptSpeedControl.hide();
    }

    // Update manual controls state based on initial dynamic clipping setting
    this.updateClippingControlsState(this.settings.dynamicClippingEnabled);

    // HDR folder
    const hdrFolder = this.gui.addFolder('HDR');
    hdrFolder.open();

    // Logarithmic HDR intensity slider
    // Using shadow property pattern: slider controls log10(value), giving equal
    // distance for equal perceptual change (orders of magnitude)
    const logMin = Math.log10(0.01); // -2
    const logMax = Math.log10(100); // +2

    // Initialize shadow log value from current settings
    this.hdrLogValue.log = Math.log10(this.settings.hdrMultiplier);

    // Helper to format the actual intensity value for display
    const formatIntensity = (logValue: number): string => {
      const actual = Math.pow(10, logValue);
      if (actual >= 10) return actual.toFixed(0);
      if (actual >= 1) return actual.toFixed(1);
      if (actual >= 0.1) return actual.toFixed(2);
      return actual.toFixed(3);
    };

    const hdrControl = hdrFolder
      .add(this.hdrLogValue, 'log', logMin, logMax, 0.01)
      .name('Intensity')
      .onChange((logValue: number) => {
        // Convert log to actual value
        const actualValue = Math.pow(10, logValue);
        this.settings.hdrMultiplier = actualValue;
        // Update shader config and trigger material updates
        // HDR multiplier is now handled through material manager
        this.sceneManager.updateHDRMultiplier(actualValue);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Override updateDisplay to show actual intensity value instead of log value
    // This is necessary because lil-gui doesn't support custom value formatters
    const originalUpdateDisplay = hdrControl.updateDisplay.bind(hdrControl);

    (hdrControl as any).updateDisplay = () => {
      originalUpdateDisplay();
      // After lil-gui updates the display, override the input value with formatted intensity

      const input = (hdrControl as any).$input as HTMLInputElement | undefined;
      if (input) {
        input.value = formatIntensity(this.hdrLogValue.log);
      }
      return hdrControl;
    };

    // Store controller reference for sync updates
    this.controllers.hdrMultiplier = hdrControl;

    // Initial display update to show actual value
    hdrControl.updateDisplay();

    // Set tooltip on the DOM element
    hdrControl.domElement.setAttribute(
      'title',
      'Intensity: Multiplies point brightness (logarithmic slider)\n' +
        '• Range: 0.01× (dim) to 100× (bright)\n' +
        '• 1.0 = neutral (no change)\n' +
        '• Higher values increase glow/bloom effects\n' +
        '• Logarithmic scale: equal slider distance = equal perceived change'
    );

    // Tone Mapping selector - moved to HDR folder
    const toneMappingControl = hdrFolder
      .add(this.settings, 'toneMapping', [
        'None',
        'Linear',
        'Reinhard',
        'Cineon',
        'ACES',
        'AgX',
        'Neutral',
      ])
      .name('Tone Mapping')
      .onChange((value: string) => {
        const toneMappingMap: { [key: string]: THREE.ToneMapping } = {
          None: THREE.NoToneMapping,
          Linear: THREE.LinearToneMapping,
          Reinhard: THREE.ReinhardToneMapping,
          Cineon: THREE.CineonToneMapping,
          ACES: THREE.ACESFilmicToneMapping,
          AgX: THREE.AgXToneMapping,
          Neutral: THREE.NeutralToneMapping,
        };
        this.postProcessing.setToneMapping(toneMappingMap[value]);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for tone mapping
    toneMappingControl.domElement.setAttribute(
      'title',
      'Tone Mapping: Converts HDR colors to display range\n' +
        '• None: No tone mapping (may clip bright values)\n' +
        '• Linear: Simple linear mapping\n' +
        '• Reinhard: Classic tone mapping operator\n' +
        '• Cineon: Film-like response curve\n' +
        '• ACES: Academy Color Encoding (film industry standard)\n' +
        '• AgX: Modern filmic mapping with good color preservation\n' +
        '• Neutral: Minimal color shift tone mapping'
    );

    // Anti-aliasing folder
    const aaFolder = this.gui.addFolder('Anti-Aliasing');
    aaFolder.close(); // Collapsed by default

    // SSAA settings (collapsible) - First because it's the highest quality
    const ssaaFolder = aaFolder.addFolder('SSAA Settings (Supersampling)');

    aaFolder
      .add(this.settings, 'ssaaEnabled')
      .name('SSAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setSSAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
        // Show/hide SSAA settings folder
        if (value) {
          ssaaFolder.show();
          ssaaFolder.open();
        } else {
          ssaaFolder.close();
          ssaaFolder.hide();
        }
      });

    ssaaFolder
      .add(this.settings, 'ssaaMultiplier', [1.5, 2.0, 3.0, 4.0])
      .name('Resolution Multiplier')
      .onChange((value: number) => {
        this.postProcessing.setSSAAMultiplier(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // FXAA toggle
    const fxaaControl = aaFolder
      .add(this.settings, 'fxaaEnabled')
      .name('FXAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setFXAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for FXAA
    fxaaControl.domElement.setAttribute(
      'title',
      'FXAA (Fast Approximate Anti-Aliasing)\n' +
        '• Fast post-process anti-aliasing\n' +
        '• Good performance, decent quality\n' +
        '• May slightly blur the image\n' +
        '• Works well with additive blending'
    );

    // MSAA settings (collapsible)
    const msaaFolder = aaFolder.addFolder('MSAA Settings ⚠️');

    const msaaControl = aaFolder
      .add(this.settings, 'msaaEnabled')
      .name('MSAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setMSAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
        // Show/hide MSAA settings folder
        if (value) {
          msaaFolder.show();
          msaaFolder.open();
        } else {
          msaaFolder.close();
          msaaFolder.hide();
        }
      });

    // Set tooltip for MSAA with warning
    msaaControl.domElement.setAttribute(
      'title',
      'MSAA (Multisample Anti-Aliasing) ⚠️\n' +
        '• Hardware-accelerated anti-aliasing\n' +
        '• WARNING: Causes brightness issues with additive blending\n' +
        '• Points will appear brighter with more samples\n' +
        '• Consider using FXAA or SMAA instead'
    );

    msaaFolder
      .add(this.settings, 'msaaSamples', [2, 4, 8])
      .name('Sample Count')
      .onChange((value: number) => {
        this.postProcessing.setMSAASamples(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // SMAA anti-aliasing (no subfolder needed - only on/off toggle)
    const smaaControl = aaFolder
      .add(this.settings, 'smaaEnabled')
      .name('SMAA Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setSMAAEnabled(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for SMAA
    smaaControl.domElement.setAttribute(
      'title',
      'SMAA (Subpixel Morphological Anti-Aliasing)\n' +
        '• Advanced edge detection anti-aliasing\n' +
        '• Better quality than FXAA, faster than SSAA\n' +
        '• Preserves sharpness while smoothing edges\n' +
        '• Good balance of quality and performance\n' +
        '• Uses HIGH preset (optimal quality/performance balance)'
    );

    // Note: SMAA threshold and search steps controls removed because pmndrs/postprocessing
    // SMAAEffect only supports preset-based configuration (LOW/MEDIUM/HIGH/ULTRA).
    // Fine-grained control is not available in the underlying library.

    // Initially show/hide folders based on settings
    if (this.settings.ssaaEnabled) {
      ssaaFolder.show();
      ssaaFolder.open();
    } else {
      ssaaFolder.hide();
    }

    if (!this.settings.msaaEnabled) {
      msaaFolder.hide();
    }

    // Post-Processing Effects folder
    const effectsFolder = this.gui.addFolder('Post-Processing Effects');
    effectsFolder.close(); // Closed by default

    // Bloom subfolder - moved here from top level
    const bloomFolder = effectsFolder.addFolder('Bloom');
    bloomFolder.close(); // Closed by default like all other effects

    const bloomThresholdControl = bloomFolder
      .add(this.settings, 'bloomThreshold', 0, 1, 0.01)
      .name('Threshold')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, undefined, value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    bloomThresholdControl.domElement.setAttribute(
      'title',
      'Bloom Threshold: Minimum brightness for bloom\n' +
        '• Only pixels brighter than this value will bloom\n' +
        '• 0 = everything blooms, 1 = only brightest areas bloom\n' +
        '• Use with HDR intensity for best results'
    );

    const bloomStrengthControl = bloomFolder
      .add(this.settings, 'bloomStrength', 0, 2, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(value, undefined, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    bloomStrengthControl.domElement.setAttribute(
      'title',
      'Bloom Strength: Intensity of the glow effect\n' +
        '• 0 = no bloom, 1 = normal, 2 = intense glow\n' +
        '• Creates realistic light bleeding from bright areas'
    );

    const bloomRadiusControl = bloomFolder
      .add(this.settings, 'bloomRadius', 0, 1, 0.01)
      .name('Radius')
      .onChange((value: number) => {
        this.postProcessing.updateBloomSettings(undefined, value, undefined);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip on the DOM element
    bloomRadiusControl.domElement.setAttribute(
      'title',
      'Bloom Radius: Size of the glow spread\n' +
        '• 0 = tight glow, 1 = wide spread\n' +
        '• Larger radius = softer, more diffuse glow\n' +
        '• Affects computational cost'
    );

    // Bloom levels control (mipmap blur levels)
    const bloomLevelsControl = bloomFolder
      .add(this.settings, 'bloomLevels', 1, 12, 1)
      .name('Mipmap Levels')
      .onChange((value: number) => {
        this.postProcessing.setBloomLevels(Math.round(value));
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for bloom levels
    bloomLevelsControl.domElement.setAttribute(
      'title',
      'Bloom Mipmap Levels: Quality vs Performance\n' +
        '• 1-3 = Coarse bloom (fastest)\n' +
        '• 4-6 = Balanced quality\n' +
        '• 7-9 = Smooth bloom (default 8)\n' +
        '• 10-12 = Very smooth (slowest)'
    );

    // Detector Noise subfolder (physics-based: Shot + Readout + FPN)
    const detectorNoiseFolder = effectsFolder.addFolder('Detector Noise');
    detectorNoiseFolder.close();

    const detectorNoiseEnabledControl = detectorNoiseFolder
      .add(this.settings, 'detectorNoiseEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setDetectorNoiseEnabled(
          value,
          this.settings.detectorNoiseReadoutSigma,
          this.settings.detectorNoisePhotonGain,
          this.settings.detectorNoiseFpnSigma
        );
        this.saveSettings();
        // Keep animation running when detector noise is enabled
        if (value) {
          this.animationController?.startAnimation();
        }
        this.triggerAnimation();
      });

    // Set tooltip for detector noise enabled
    detectorNoiseEnabledControl.domElement.setAttribute(
      'title',
      'Detector Noise (Physics-Based):\n' +
        '• Shot noise: Poisson noise from photon statistics\n' +
        '• Readout noise: Temporal Gaussian noise from electronics\n' +
        '• Fixed Pattern Noise: Static per-pixel offset\n' +
        '• Ideal for scientific imaging aesthetics'
    );

    const detectorNoiseReadoutSigmaControl = detectorNoiseFolder
      .add(this.settings, 'detectorNoiseReadoutSigma', 0, 0.1, 0.001)
      .name('Readout Noise')
      .onChange((value: number) => {
        this.postProcessing.updateDetectorNoiseSettings({ readoutSigma: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for readout sigma
    detectorNoiseReadoutSigmaControl.domElement.setAttribute(
      'title',
      'Readout Noise (temporal, varies each frame):\n' +
        '• Signal-independent electronic noise\n' +
        '• 0 = No readout noise\n' +
        '• 0.01 = Subtle (default)\n' +
        '• 0.05 = Moderate (old detector)\n' +
        '• 0.1 = High (uncooled sensor)'
    );

    const detectorNoisePhotonGainControl = detectorNoiseFolder
      .add(this.settings, 'detectorNoisePhotonGain', 0.0001, 0.1, 0.0001)
      .name('Shot Noise')
      .onChange((value: number) => {
        this.postProcessing.updateDetectorNoiseSettings({ photonGain: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for photon gain
    detectorNoisePhotonGainControl.domElement.setAttribute(
      'title',
      'Shot Noise (Poisson, signal-dependent):\n' +
        '• Higher = more visible shot noise (fewer photons)\n' +
        '• 0.001 = Bright illumination (minimal shot noise)\n' +
        '• 0.01 = Normal conditions (default)\n' +
        '• 0.05 = Low light (visible shot noise)\n' +
        '• 0.1 = Very low light (strong shot noise)'
    );

    const detectorNoiseFpnControl = detectorNoiseFolder
      .add(this.settings, 'detectorNoiseFpnSigma', 0, 0.05, 0.001)
      .name('Fixed Pattern')
      .onChange((value: number) => {
        this.postProcessing.updateDetectorNoiseSettings({ fpnSigma: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for FPN
    detectorNoiseFpnControl.domElement.setAttribute(
      'title',
      'Fixed Pattern Noise (static, constant per pixel):\n' +
        '• Per-pixel offset from detector non-uniformities\n' +
        '• 0 = No FPN (perfect detector)\n' +
        '• 0.005 = Subtle (default, good detector)\n' +
        '• 0.02 = Moderate (older detector)\n' +
        '• 0.05 = High (uncalibrated sensor)'
    );

    // Depth of Field subfolder
    const dofFolder = effectsFolder.addFolder('Depth of Field');
    dofFolder.close();

    const dofEnabledControl = dofFolder
      .add(this.settings, 'dofEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setDOF(value, this.settings.dofFocus, this.settings.dofStrength);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for DOF enabled
    dofEnabledControl.domElement.setAttribute(
      'title',
      'Depth of Field: Simulates camera focus\n' +
        '• Blurs objects outside the focal distance\n' +
        '• Creates cinematic depth effect\n' +
        '• Performance impact when enabled'
    );

    const dofFocusControl = dofFolder
      .add(this.settings, 'dofFocus', 0.1, 100, 0.1)
      .name('Focus Distance')
      .onChange((value: number) => {
        // Always update and trigger animation so user can see changes immediately
        this.postProcessing.updateDOF({ focus: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for DOF focus
    dofFocusControl.domElement.setAttribute(
      'title',
      'Focus Distance: Distance to the sharp focal plane\n' +
        '• Objects at this distance will be sharp\n' +
        '• Objects closer or farther will be blurred\n' +
        '• Value in world units (adjust based on scene scale)'
    );

    const dofStrengthControl = dofFolder
      .add(this.settings, 'dofStrength', 0, 1, 0.01)
      .name('Blur Strength')
      .onChange((value: number) => {
        // Always update and trigger animation so user can see changes immediately
        this.postProcessing.updateDOF({ strength: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for DOF strength
    dofStrengthControl.domElement.setAttribute(
      'title',
      'Blur Strength: Amount of out-of-focus blur\n' +
        '• 0 = No blur (everything in focus)\n' +
        '• 0.5 = Moderate blur\n' +
        '• 1.0 = Maximum blur\n' +
        '• Higher values create stronger bokeh effect'
    );

    // Chromatic Aberration subfolder
    const chromaticFolder = effectsFolder.addFolder('Chromatic Aberration');
    chromaticFolder.close();

    const chromaticEnabledControl = chromaticFolder
      .add(this.settings, 'chromaticAberrationEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setChromaticAberration(
          value,
          this.settings.chromaticAberrationStrength
        );
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for chromatic aberration enabled
    chromaticEnabledControl.domElement.setAttribute(
      'title',
      'Chromatic Aberration: Simulates lens color fringing\n' +
        '• Separates RGB channels slightly\n' +
        '• Creates rainbow edges on high contrast areas\n' +
        '• Adds cinematic/stylistic effect'
    );

    const chromaticStrengthControl = chromaticFolder
      .add(this.settings, 'chromaticAberrationStrength', 0, 1, 0.01)
      .name('Strength')
      .onChange((value: number) => {
        // Always update the uniform, even if disabled (so it's ready when enabled)
        this.postProcessing.updateChromaticAberration(value);
        this.saveSettings();
        this.triggerAnimation();
      });

    // Set tooltip for chromatic aberration strength
    chromaticStrengthControl.domElement.setAttribute(
      'title',
      'Chromatic Aberration Strength\n' +
        '• 0 = No color separation\n' +
        '• 0.15 = Subtle effect (default)\n' +
        '• 0.5 = Moderate color fringing\n' +
        '• 1.0 = Strong rainbow edges'
    );

    // Ambient Occlusion subfolder
    const aoFolder = effectsFolder.addFolder('Ambient Occlusion');
    aoFolder.close();

    const aoEnabledControl = aoFolder
      .add(this.settings, 'aoEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setAOEnabled(value, this.settings.aoQuality);
        this.saveSettings();
        this.triggerAnimation();
      });

    aoEnabledControl.domElement.setAttribute(
      'title',
      'Ambient Occlusion: Darkens corners and crevices\n' +
        '• Adds depth and realism to scene\n' +
        '• Simulates indirect shadows\n' +
        '• Performance impact scales with quality'
    );

    const aoQualityControl = aoFolder
      .add(this.settings, 'aoQuality', ['low', 'medium', 'high', 'ultra'])
      .name('Quality')
      .onChange((value: 'low' | 'medium' | 'high' | 'ultra') => {
        if (this.settings.aoEnabled) {
          this.postProcessing.setAOEnabled(true, value);
          this.saveSettings();
          this.triggerAnimation();
        }
      });

    aoQualityControl.domElement.setAttribute(
      'title',
      'AO Quality Level:\n' +
        '• Low: Fast, lower quality (4 samples)\n' +
        '• Medium: Balanced (8 samples)\n' +
        '• High: Better quality (16 samples)\n' +
        '• Ultra: Best quality, slower (32 samples)'
    );

    // Vignette subfolder
    const vignetteFolder = effectsFolder.addFolder('Vignette');
    vignetteFolder.close();

    const vignetteEnabledControl = vignetteFolder
      .add(this.settings, 'vignetteEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setVignetteEnabled(
          value,
          this.settings.vignetteDarkness,
          this.settings.vignetteOffset
        );
        this.saveSettings();
        this.triggerAnimation();
      });

    vignetteEnabledControl.domElement.setAttribute(
      'title',
      'Vignette: Darkens edges of the screen\n' +
        '• Draws focus to center of view\n' +
        '• Creates cinematic look\n' +
        '• Minimal performance impact'
    );

    const vignetteDarknessControl = vignetteFolder
      .add(this.settings, 'vignetteDarkness', 0, 1, 0.01)
      .name('Darkness')
      .onChange((value: number) => {
        if (this.settings.vignetteEnabled) {
          this.postProcessing.setVignetteEnabled(true, value, this.settings.vignetteOffset);
          this.saveSettings();
          this.triggerAnimation();
        }
      });

    vignetteDarknessControl.domElement.setAttribute(
      'title',
      'Vignette Darkness:\n' +
        '• 0 = No darkening\n' +
        '• 0.5 = Moderate darkness (default)\n' +
        '• 1.0 = Maximum darkness'
    );

    const vignetteOffsetControl = vignetteFolder
      .add(this.settings, 'vignetteOffset', 0, 1, 0.01)
      .name('Offset')
      .onChange((value: number) => {
        if (this.settings.vignetteEnabled) {
          this.postProcessing.setVignetteEnabled(true, this.settings.vignetteDarkness, value);
          this.saveSettings();
          this.triggerAnimation();
        }
      });

    vignetteOffsetControl.domElement.setAttribute(
      'title',
      'Vignette Offset:\n' +
        '• 0 = Effect starts at center\n' +
        '• 0.5 = Effect starts mid-way (default)\n' +
        '• 1.0 = Effect only at very edges'
    );

    // Lens Distortion subfolder
    const lensDistortionFolder = effectsFolder.addFolder('Lens Distortion');
    lensDistortionFolder.close();

    const lensDistortionEnabledControl = lensDistortionFolder
      .add(this.settings, 'lensDistortionEnabled')
      .name('Enabled')
      .onChange((value: boolean) => {
        this.postProcessing.setLensDistortionEnabled(
          value,
          this.settings.lensDistortionX,
          this.settings.lensDistortionY,
          this.settings.lensPrincipalPointX,
          this.settings.lensPrincipalPointY,
          this.settings.lensFocalLengthX,
          this.settings.lensFocalLengthY,
          this.settings.lensSkew
        );
        this.saveSettings();
        this.triggerAnimation();
      });

    lensDistortionEnabledControl.domElement.setAttribute(
      'title',
      'Lens Distortion: Simulates camera lens imperfections\n' +
        '• Barrel/pincushion distortion effects\n' +
        '• Principal point and focal length adjustment\n' +
        '• Skew correction for non-square pixels'
    );

    const lensDistortionXControl = lensDistortionFolder
      .add(this.settings, 'lensDistortionX', -1, 1, 0.001)
      .name('Distortion X')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ distortionX: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensDistortionX = lensDistortionXControl;

    lensDistortionXControl.domElement.setAttribute(
      'title',
      'Radial Distortion X:\n' +
        '• 0 = No distortion\n' +
        '• Negative = Barrel distortion (fish-eye)\n' +
        '• Positive = Pincushion distortion'
    );

    const lensDistortionYControl = lensDistortionFolder
      .add(this.settings, 'lensDistortionY', -1, 1, 0.001)
      .name('Distortion Y')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ distortionY: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensDistortionY = lensDistortionYControl;

    lensDistortionYControl.domElement.setAttribute(
      'title',
      'Radial Distortion Y:\n' +
        '• 0 = No distortion\n' +
        '• Negative = Barrel distortion (fish-eye)\n' +
        '• Positive = Pincushion distortion'
    );

    const lensPrincipalPointXControl = lensDistortionFolder
      .add(this.settings, 'lensPrincipalPointX', -1, 1, 0.001)
      .name('Principal Point X')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ principalPointX: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensPrincipalPointX = lensPrincipalPointXControl;

    lensPrincipalPointXControl.domElement.setAttribute(
      'title',
      'Principal Point X offset:\n' +
        '• 0 = Centered (default)\n' +
        '• Negative = Shift distortion center left\n' +
        '• Positive = Shift distortion center right'
    );

    const lensPrincipalPointYControl = lensDistortionFolder
      .add(this.settings, 'lensPrincipalPointY', -1, 1, 0.001)
      .name('Principal Point Y')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ principalPointY: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensPrincipalPointY = lensPrincipalPointYControl;

    lensPrincipalPointYControl.domElement.setAttribute(
      'title',
      'Principal Point Y offset:\n' +
        '• 0 = Centered (default)\n' +
        '• Negative = Shift distortion center up\n' +
        '• Positive = Shift distortion center down'
    );

    const lensFocalLengthXControl = lensDistortionFolder
      .add(this.settings, 'lensFocalLengthX', 0.1, 3, 0.001)
      .name('Focal Length X')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ focalLengthX: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensFocalLengthX = lensFocalLengthXControl;

    lensFocalLengthXControl.domElement.setAttribute(
      'title',
      'Focal Length X:\n' +
        '• 1 = Normal (default)\n' +
        '• < 1 = Wide angle effect\n' +
        '• > 1 = Telephoto effect'
    );

    const lensFocalLengthYControl = lensDistortionFolder
      .add(this.settings, 'lensFocalLengthY', 0.1, 3, 0.001)
      .name('Focal Length Y')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ focalLengthY: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensFocalLengthY = lensFocalLengthYControl;

    lensFocalLengthYControl.domElement.setAttribute(
      'title',
      'Focal Length Y:\n' +
        '• 1 = Normal (default)\n' +
        '• < 1 = Wide angle effect\n' +
        '• > 1 = Telephoto effect'
    );

    const lensSkewControl = lensDistortionFolder
      .add(this.settings, 'lensSkew', -0.1, 0.1, 0.001)
      .name('Skew')
      .onChange((value: number) => {
        this.postProcessing.updateLensDistortion({ skew: value });
        this.saveSettings();
        this.triggerAnimation();
      });

    // Store reference for updates
    this.controllers.lensSkew = lensSkewControl;

    lensSkewControl.domElement.setAttribute(
      'title',
      'Lens Skew (radians):\n' +
        '• 0 = No skew (default)\n' +
        '• Corrects for non-square pixels\n' +
        '• Usually very small values'
    );

    // Reset to Defaults button at root level
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
  private applyCustomStyling(): void {
    const root = this.gui.domElement;

    // Style the main container
    root.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
    root.style.borderRadius = '8px';
    root.style.backdropFilter = 'blur(10px)';
    root.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
    root.style.fontFamily =
      '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif';
    root.style.fontSize = '12px';

    // Style the title
    const title = root.querySelector('.title') as HTMLElement;
    if (title) {
      title.style.backgroundColor = 'transparent';
      title.style.color = '#e0e0e0';
      title.style.fontSize = '14px';
      title.style.fontWeight = 'bold';
      title.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';
      title.style.paddingBottom = '6px';
      title.style.marginBottom = '10px';
    }

    // Override lil-gui's default styles with CSS
    const style = document.createElement('style');
    style.textContent = `
      .lil-gui {
        --background-color: rgba(30, 30, 30, 0.9) !important;
        --title-background-color: transparent !important;
        --title-text-color: #e0e0e0 !important;
        --widget-color: rgba(255, 255, 255, 0.1) !important;
        --hover-color: rgba(255, 255, 255, 0.15) !important;
        --focus-color: #4CAF50 !important;
        --number-color: #4CAF50 !important;
        --string-color: #4CAF50 !important;
        --font-size: 12px !important;
        --input-font-size: 12px !important;
        --folder-border-color: rgba(255, 255, 255, 0.2) !important;
        --checkbox-border-radius: ${controlsConfig.borderRadius.checkbox}px !important;
        color: #e0e0e0 !important;
      }
      
      .lil-gui .controller {
        border-radius: ${controlsConfig.borderRadius.section}px !important;
        margin: 2px 0 !important;
      }
      
      .lil-gui .controller:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
      }
      
      .lil-gui .title {
        padding: ${spacingConfig.compactGap}px !important;
        border-radius: ${controlsConfig.borderRadius.header}px ${controlsConfig.borderRadius.header}px 0 0 !important;
      }
      
      .lil-gui button {
        border-radius: ${controlsConfig.borderRadius.section}px !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
      }
      
      .lil-gui button:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
      }
      
      .lil-gui input[type="number"],
      .lil-gui input[type="text"] {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: none !important;
        border-radius: ${controlsConfig.borderRadius.section}px !important;
        color: #4CAF50 !important;
        padding: ${spacingConfig.tinyGap}px ${spacingConfig.borderPadding}px !important;
      }
      
      .lil-gui select {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: none !important;
        border-radius: ${controlsConfig.borderRadius.section}px !important;
        color: #e0e0e0 !important;
        padding: ${spacingConfig.tinyGap}px ${spacingConfig.borderPadding}px !important;
      }
      
      /* Remove borders from sliders too */
      .lil-gui .widget {
        border: none !important;
      }
      
      .lil-gui .controller.number .slider {
        background-color: rgba(255, 255, 255, 0.1) !important;
      }
      
      /* Remove all controller borders and outlines */
      .lil-gui .controller {
        border: none !important;
        outline: none !important;
      }
      
      .lil-gui .controller.number {
        border: none !important;
      }
      
      /* Clean folder styling */
      .lil-gui .children {
        border: none !important;
        margin-left: 20px !important;
      }
      
      .lil-gui .folder {
        border: none !important;
        margin-bottom: 2px !important;
      }
      
      /* Remove all borders from folder titles */
      .lil-gui .title {
        border: none !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
      }
      
      .lil-gui > .title {
        background-color: transparent !important;
        border: none !important;
      }
      
      .lil-gui .folder > .title {
        border: none !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
      }
    `;

    // Only add style once
    if (!document.getElementById('lil-gui-custom-styles')) {
      style.id = 'lil-gui-custom-styles';
      document.head.appendChild(style);
    }
  }

  /**
   * Set the animation controller reference
   * @param animationController - The animation controller instance
   */
  setAnimationController(animationController: AnimationController): void {
    this.animationController = animationController;
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
   * This prevents focus from getting stuck on checkboxes and other controls
   */
  private setupAutoBlur(): void {
    // Use MutationObserver to watch for new controls being added
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node instanceof HTMLElement) {
            // Find all input elements (checkboxes, sliders, etc.)
            const inputs = node.querySelectorAll('input, select');
            inputs.forEach((input) => {
              this.addAutoBlurToElement(input as HTMLElement);
            });
            // Also check if the node itself is an input
            if (node.tagName === 'INPUT' || node.tagName === 'SELECT') {
              this.addAutoBlurToElement(node);
            }
          }
        });
      });
    });

    // Start observing the GUI element for changes
    observer.observe(this.gui.domElement, {
      childList: true,
      subtree: true,
    });

    // Also handle existing inputs
    setTimeout(() => {
      const inputs = this.gui.domElement.querySelectorAll('input, select');
      inputs.forEach((input) => {
        this.addAutoBlurToElement(input as HTMLElement);
      });
    }, 100);
  }

  /**
   * Add auto-blur behavior to an element
   */
  private addAutoBlurToElement(element: HTMLElement): void {
    // For checkboxes and select dropdowns, blur immediately after change
    if (element instanceof HTMLInputElement && element.type === 'checkbox') {
      element.addEventListener('change', () => {
        // Small delay to ensure the change is processed
        setTimeout(() => element.blur(), 10);
      });
      // Also blur on click for checkboxes
      element.addEventListener('click', () => {
        setTimeout(() => element.blur(), 10);
      });
    }
    // For select dropdowns
    else if (element instanceof HTMLSelectElement) {
      element.addEventListener('change', () => {
        setTimeout(() => element.blur(), 10);
      });
    }
    // For text inputs and sliders, blur on Enter key or when value is committed
    else if (element instanceof HTMLInputElement) {
      element.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          element.blur();
        }
      });
      // For sliders, also blur when mouse is released
      if (element.type === 'range' || element.type === 'number') {
        element.addEventListener('mouseup', () => {
          setTimeout(() => element.blur(), 10);
        });
        element.addEventListener('touchend', () => {
          setTimeout(() => element.blur(), 10);
        });
      }
    }
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
      this.settings.detectorNoiseReadoutSigma = 0.015; // Subtle temporal noise
      this.settings.detectorNoisePhotonGain = 0.008; // Low shot noise
      this.settings.detectorNoiseFpnSigma = 0.003; // Subtle fixed pattern
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
   * Clean up lil-gui resources and remove from DOM.
   *
   * Destroys the GUI instance. Should be called when rendering controls
   * are no longer needed (e.g., application teardown).
   *
   * After calling dispose(), the RenderingControls instance cannot be reused.
   */
  dispose(): void {
    this.gui.destroy();
  }
}
