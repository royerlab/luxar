/**
 * Camera controls setup for rendering controls UI.
 *
 * Creates controls for camera-specific settings:
 * - FOV presets (28mm, 35mm, 50mm, 85mm, 135mm, Custom)
 * - FOV slider
 * - Clipping planes (near, far, dynamic clipping)
 */

import { config } from '../../../config';
import { log, Modules } from '../../../utils/log';
import type { SetupContext, SetupResult } from '../types';
import { FOLDER_ICONS } from '../folder-icons';
import { fovToFocalLength } from '../fov-utils';

/**
 * Set up camera controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @param controllersRef - Reference to controllers object (needed for FOV preset lens distortion updates)
 * @returns Setup result with controller references
 */
export function setupCameraControls(
  context: SetupContext,
  controllersRef: SetupResult['controllers']
): SetupResult {
  const {
    gui,
    settings,
    sceneManager,
    postProcessing,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState,
  } = context;

  const controllers: SetupResult['controllers'] = {};

  // Camera folder - for camera-specific settings
  const cameraFolder = gui.addFolder('Camera', FOLDER_ICONS.camera);
  cameraFolder.open();

  cameraFolder.domElement?.setAttribute(
    'title',
    'Camera: Controls the virtual camera lens and projection\n\n' +
      '• FOV Preset: Choose a standard lens (28mm wide to 135mm telephoto)\n' +
      '• Field of View: Fine-tune viewing angle (also Ctrl+Wheel)\n' +
      '• Clipping Planes: Control what range of distances is visible'
  );

  // FOV Preset dropdown
  const presetOptions = Object.keys(config.camera.fovPresets);
  const fovPresetControl = cameraFolder
    .add(settings, 'fovPreset', presetOptions)
    .name('FOV Preset')
    .onChange((presetName: string) => {
      // Reset "Custom" option text back to "Custom" when a preset is selected
      const select = fovPresetControl.$input as HTMLSelectElement;
      if (select) {
        const customOption = Array.from(select.options).find((opt) => opt.value === 'Custom');
        if (customOption) {
          customOption.textContent = 'Custom';
        }
      }

      const fovValue = config.camera.fovPresets[presetName];
      if (fovValue > 0) {
        // Apply preset FOV
        settings.fov = fovValue;

        // Calculate delta and apply to camera
        const currentFOV = sceneManager.currentFov;
        const delta = (fovValue - currentFOV) / config.camera.fovSensitivity;
        sceneManager.updateFOV(delta);

        // Update FOV slider display
        if (controllersRef.fov) {
          controllersRef.fov.setValue(fovValue);
          controllersRef.fov.updateDisplay();
        }

        // Apply corresponding chromatic lens distortion preset (if enabled)
        const lensPreset = config.camera.lensDistortionPresets[presetName];
        if (lensPreset && settings.chromaticLensDistortionEnabled) {
          settings.chromaticLensDistortionX = lensPreset.distortionX;
          settings.chromaticLensDistortionY = lensPreset.distortionY;
          settings.chromaticLensDispersion = lensPreset.dispersion;
          settings.chromaticLensPrincipalPointX = lensPreset.principalPointX;
          settings.chromaticLensPrincipalPointY = lensPreset.principalPointY;
          settings.chromaticLensFocalLengthX = lensPreset.focalLengthX;
          settings.chromaticLensFocalLengthY = lensPreset.focalLengthY;
          settings.chromaticLensSkew = lensPreset.skew;

          // Apply chromatic lens distortion changes
          postProcessing.updateChromaticLensDistortion({
            distortionX: lensPreset.distortionX,
            distortionY: lensPreset.distortionY,
            dispersion: lensPreset.dispersion,
            principalPointX: lensPreset.principalPointX,
            principalPointY: lensPreset.principalPointY,
            focalLengthX: lensPreset.focalLengthX,
            focalLengthY: lensPreset.focalLengthY,
            skew: lensPreset.skew,
          });

          // Update chromatic lens distortion UI controllers to reflect new values
          if (controllersRef.chromaticLensDistortionX) {
            controllersRef.chromaticLensDistortionX.setValue(lensPreset.distortionX);
            controllersRef.chromaticLensDistortionX.updateDisplay();
          }
          if (controllersRef.chromaticLensDistortionY) {
            controllersRef.chromaticLensDistortionY.setValue(lensPreset.distortionY);
            controllersRef.chromaticLensDistortionY.updateDisplay();
          }
          if (controllersRef.chromaticLensDispersion) {
            controllersRef.chromaticLensDispersion.setValue(lensPreset.dispersion);
            controllersRef.chromaticLensDispersion.updateDisplay();
          }
          if (controllersRef.chromaticLensPrincipalPointX) {
            controllersRef.chromaticLensPrincipalPointX.setValue(lensPreset.principalPointX);
            controllersRef.chromaticLensPrincipalPointX.updateDisplay();
          }
          if (controllersRef.chromaticLensPrincipalPointY) {
            controllersRef.chromaticLensPrincipalPointY.setValue(lensPreset.principalPointY);
            controllersRef.chromaticLensPrincipalPointY.updateDisplay();
          }
          if (controllersRef.chromaticLensFocalLengthX) {
            controllersRef.chromaticLensFocalLengthX.setValue(lensPreset.focalLengthX);
            controllersRef.chromaticLensFocalLengthX.updateDisplay();
          }
          if (controllersRef.chromaticLensFocalLengthY) {
            controllersRef.chromaticLensFocalLengthY.setValue(lensPreset.focalLengthY);
            controllersRef.chromaticLensFocalLengthY.updateDisplay();
          }
          if (controllersRef.chromaticLensSkew) {
            controllersRef.chromaticLensSkew.setValue(lensPreset.skew);
            controllersRef.chromaticLensSkew.updateDisplay();
          }
        }
      }

      saveSettings();
      triggerAnimation();
    });

  // Store reference for updates
  controllers.fovPreset = fovPresetControl;

  // Set tooltip for FOV presets
  fovPresetControl.domElement.setAttribute(
    'title',
    'FOV Preset: Professional camera lens equivalents (horizontal FOV)\n' +
      '• 28mm Wide (75°): Ultra-wide angle + barrel distortion\n' +
      '• 35mm (63°): Wide angle + moderate barrel distortion\n' +
      '• 50mm Normal (47°): Natural human vision + no distortion\n' +
      '• 85mm Portrait (29°): Telephoto + slight pincushion\n' +
      '• 135mm Tele (18°): Strong telephoto + pincushion distortion\n' +
      '• Custom: Manual FOV control via slider or Ctrl+Wheel\n' +
      '• Note: Also applies realistic lens distortion when enabled'
  );

  const fovControl = cameraFolder
    .add(settings, 'fov', config.camera.fovMin, config.camera.fovMax, 1)
    .name('Field of View')
    .onChange((value: number) => {
      // When FOV slider changes, show approximate focal length
      const focalLength = fovToFocalLength(value);
      const customLabel = `~${focalLength}mm`;

      // Update the "Custom" option text to show the approximate focal length
      settings.fovPreset = 'Custom';
      if (controllersRef.fovPreset) {
        const select = controllersRef.fovPreset.$input as HTMLSelectElement;
        if (select) {
          // Find and update the "Custom" option
          const customOption = Array.from(select.options).find((opt) => opt.value === 'Custom');
          if (customOption) {
            customOption.textContent = customLabel;
          }
        }
        controllersRef.fovPreset.setValue('Custom');
        controllersRef.fovPreset.updateDisplay();
      }

      // Calculate the delta needed to reach the target FOV
      const currentFOV = sceneManager.currentFov;
      const targetFOV = value;
      const delta = (targetFOV - currentFOV) / config.camera.fovSensitivity;

      // Use the existing updateFOV method which handles bounds checking and material updates
      sceneManager.updateFOV(delta);

      saveSettings();
      triggerAnimation();
    });

  // Store reference for updates
  controllers.fov = fovControl;

  // Set tooltip for FOV control
  fovControl.domElement.setAttribute(
    'title',
    'Field of View: Camera viewing angle in degrees\n' +
      '• Lower values: Telephoto lens effect (narrow view)\n' +
      '• Higher values: Wide-angle lens effect (broader view)\n' +
      '• 47° (50mm Normal) provides natural human-like viewing angle\n' +
      '• Also controllable with Ctrl+Wheel for fine adjustment\n' +
      '• Maintains world-space point sizing (points stay same physical size)'
  );

  // Clipping Planes sub-folder
  const clippingFolder = cameraFolder.addFolder('Clipping Planes');
  clippingFolder.close(); // Collapsed by default (advanced setting)

  clippingFolder.domElement?.setAttribute(
    'title',
    'Clipping Planes: Define the visible depth range of the camera\n\n' +
      'Only objects between the near and far planes are rendered.\n' +
      '• Near plane: Closest visible distance (too low = Z-fighting)\n' +
      '• Far plane: Furthest visible distance\n' +
      '• Dynamic mode auto-adjusts both for optimal precision'
  );

  const nearPlaneControl = clippingFolder
    .add(settings, 'near', 0.0001, 10.0, 0.0001)
    .name('Near Plane')
    .onChange((value: number) => {
      // Validate near plane is less than far plane
      if (value >= settings.far) {
        log.warning(Modules.RENDERER, 'Near plane must be less than far plane');
        return;
      }
      sceneManager.updateClippingPlanes(value, settings.far);
      saveSettings();
      triggerAnimation();
    });

  const farPlaneControl = clippingFolder
    .add(settings, 'far', 1, 100000, 1)
    .name('Far Plane')
    .onChange((value: number) => {
      // Validate far plane is greater than near plane
      if (value <= settings.near) {
        log.warning(Modules.RENDERER, 'Far plane must be greater than near plane');
        return;
      }
      sceneManager.updateClippingPlanes(settings.near, value);
      saveSettings();
      triggerAnimation();
    });

  // Store references for updates
  controllers.nearPlane = nearPlaneControl;
  controllers.farPlane = farPlaneControl;

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
    .add(settings, 'dynamicClippingEnabled')
    .name('Dynamic Clipping')
    .onChange((value: boolean) => {
      sceneManager.setDynamicClipping(value);
      saveSettings();
      triggerAnimation();

      // Enable/disable manual near/far controls
      updateClippingControlsState(value);
    });

  // Store reference
  controllers.dynamicClippingEnabled = dynamicClippingControl;

  dynamicClippingControl.domElement.setAttribute(
    'title',
    'Dynamic Clipping: Auto-adjust clipping planes each frame\n' +
      '• Sphere-based: smooth adaptation to camera position\n' +
      '• When enabled, manual near/far controls are disabled'
  );

  // Update manual controls state based on initial dynamic clipping setting
  updateClippingControlsState(settings.dynamicClippingEnabled);

  return {
    controllers,
  };
}
