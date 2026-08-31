/**
 * Pull live state from the scene manager + camera into the
 * `RenderingSettings` object and refresh every relevant GUI controller.
 *
 * Called whenever the rendering-controls panel becomes visible (so the
 * user sees what the camera/controls actually report rather than what
 * they were last told to be).
 *
 * The function is large because there are many fields, but each block
 * follows the same shape: pull from sceneManager → mutate `settings` →
 * push the new value into the matching `controllers.<id>` if present.
 */

import type GUI from '../gui';
import type { RenderingSettings } from '../../config';
import { config } from '../../config';
import type { SceneManager } from '../../scene/scene-manager';
import type { RenderingControllers } from './types';
import { dollyAmplitudeToPercent, isOrbitControls } from '../../controls/types';

export interface SyncCurrentStateContext {
  gui: GUI;
  settings: RenderingSettings;
  sceneManager: SceneManager;
  controllers: RenderingControllers;
  /** Apply the dynamic-clipping enabled state to the panel sliders. */
  updateClippingControlsState: (dynamicEnabled: boolean) => void;
  /** Recompute the cinematic-mode checkbox from the effects state. */
  updateCinematicModeCheckbox: () => void;
  /** Apply the orbit/fly/ortho folder visibility for the given control type. */
  updateNavigationControls: (controlType: 'orbit' | 'fly' | 'ortho') => void;
}

export function syncCameraFovState(settings: RenderingSettings, sceneManager: SceneManager): void {
  settings.fov = sceneManager.currentFov;
  const currentPreset = Object.entries(config.camera.fovPresets).find(
    ([, fovValue]) => fovValue > 0 && Math.abs(fovValue - settings.fov) < 0.5
  );
  settings.fovPreset = (currentPreset ? currentPreset[0] : 'Custom') as typeof settings.fovPreset;
}

export function syncCurrentState(context: SyncCurrentStateContext): void {
  const {
    gui,
    settings,
    sceneManager,
    controllers,
    updateClippingControlsState,
    updateCinematicModeCheckbox,
    updateNavigationControls,
  } = context;

  // Camera settings.
  syncCameraFovState(settings, sceneManager);
  settings.near = sceneManager.camera.near;
  settings.far = sceneManager.camera.far;

  // Control type + active controls instance.
  const currentControlType = sceneManager.controls.getControlType();
  settings.controlType = currentControlType;
  const controls = sceneManager.controls.getControls();

  // Always read fly config from ControlsManager (persists across orbit/fly switches).
  const flyConfig = sceneManager.controls.getFlyConfig();
  settings.flyInertialMode = flyConfig.inertialMode;
  settings.flyMovementSpeed = flyConfig.movementSpeed;
  settings.flyRotationSpeed = flyConfig.rotationSpeed;
  settings.flyDamping = flyConfig.damping;
  settings.flyRotationDamping = flyConfig.rotationDamping;

  if (isOrbitControls(controls)) {
    settings.autoRotate = controls.autoRotate;
    settings.autoRotateSpeed = controls.autoRotateSpeed;
    settings.autoRotateAxis = controls.autoRotateAxis;
    settings.autoDolly = controls.autoDolly;
    // The control holds a fraction, the setting a percent — see
    // `dollyAmplitudeToPercent`. Ortho is the same class and carries these
    // fields too, so a visit to 2D reads back real values rather than
    // overwriting the user's choice with defaults.
    settings.autoDollyAmplitudePercent = dollyAmplitudeToPercent(controls.autoDollyAmplitude);
    settings.autoDollyPeriod = controls.autoDollyPeriod;
  }
  // naturalDrag is persisted at the ControlsManager level (not the active
  // controls instance, so it survives mode switches).
  settings.naturalDrag = sceneManager.controls.getNaturalDrag();

  // Update individual controller bindings we hold a reference to.
  if (controllers.controlType) {
    controllers.controlType.setValue(currentControlType);
    controllers.controlType.updateDisplay();
  }

  if (controllers.flyInertialMode) {
    controllers.flyInertialMode.setValue(settings.flyInertialMode);
    controllers.flyInertialMode.updateDisplay();
  }

  if (controllers.flyMovementSpeed) {
    controllers.flyMovementSpeed.setValue(settings.flyMovementSpeed);
    controllers.flyMovementSpeed.updateDisplay();
  }

  if (controllers.flyRotationSpeed) {
    controllers.flyRotationSpeed.setValue(settings.flyRotationSpeed);
    controllers.flyRotationSpeed.updateDisplay();
  }

  if (controllers.flyDamping) {
    controllers.flyDamping.setValue(settings.flyDamping);
    controllers.flyDamping.updateDisplay();
    if (settings.flyInertialMode) controllers.flyDamping.show();
    else controllers.flyDamping.hide();
  }

  if (controllers.flyRotationDamping) {
    controllers.flyRotationDamping.setValue(settings.flyRotationDamping);
    controllers.flyRotationDamping.updateDisplay();
    if (settings.flyInertialMode) controllers.flyRotationDamping.show();
    else controllers.flyRotationDamping.hide();
  }

  if (controllers.autoRotate) {
    controllers.autoRotate.setValue(settings.autoRotate);
    controllers.autoRotate.updateDisplay();
  }
  if (controllers.autoRotateSpeed) {
    controllers.autoRotateSpeed.setValue(settings.autoRotateSpeed);
    controllers.autoRotateSpeed.updateDisplay();
  }
  if (controllers.naturalDrag) {
    controllers.naturalDrag.setValue(settings.naturalDrag);
    controllers.naturalDrag.updateDisplay();
  }

  if (controllers.fov) {
    controllers.fov.setValue(settings.fov);
    controllers.fov.updateDisplay();
  }
  if (controllers.fovPreset) {
    controllers.fovPreset.setValue(settings.fovPreset);
    controllers.fovPreset.updateDisplay();
  }

  if (controllers.nearPlane) {
    controllers.nearPlane.setValue(settings.near);
    controllers.nearPlane.updateDisplay();
  }
  if (controllers.farPlane) {
    controllers.farPlane.setValue(settings.far);
    controllers.farPlane.updateDisplay();
  }

  // Dynamic clipping enabled state.
  const dynamicClippingState = sceneManager.getDynamicClippingState();
  settings.dynamicClippingEnabled = dynamicClippingState.enabled;
  if (controllers.dynamicClippingEnabled) {
    controllers.dynamicClippingEnabled.setValue(settings.dynamicClippingEnabled);
    controllers.dynamicClippingEnabled.updateDisplay();
  }
  updateClippingControlsState(settings.dynamicClippingEnabled);

  // Final pass: refresh every controller (HDR's updateDisplay overrides included).
  gui.controllersRecursive().forEach((controller) => controller.updateDisplay());

  // Cinematic checkbox + folder visibility for the freshly-pulled control type.
  updateCinematicModeCheckbox();
  updateNavigationControls(currentControlType);
}
