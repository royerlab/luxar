/**
 * Navigation controls setup for rendering controls UI.
 *
 * Creates controls for camera movement and rotation:
 * - Control type selector (orbit, fly, ortho)
 * - Orbit controls (auto-rotate, rotation speed)
 * - Fly controls (movement speed, rotation speed, inertial mode, damping)
 */

import { config } from '../../../config';
import type { SetupContext, SetupResult } from '../types';
import { FOLDER_ICONS } from '../folder-icons';

/**
 * Set up navigation controls in the rendering controls GUI.
 *
 * @param context - Setup context with GUI, settings, and callbacks
 * @returns Setup result with controller and folder references
 */
export function setupNavigationControls(context: SetupContext): SetupResult {
  const {
    gui,
    settings,
    sceneManager,
    animationController,
    saveSettings,
    triggerAnimation,
    updateNavigationControls,
  } = context;

  const controllers: SetupResult['controllers'] = {};

  // Navigation folder - for camera movement and rotation controls
  const navigationFolder = gui.addFolder('Navigation', FOLDER_ICONS.navigation);
  navigationFolder.open();

  navigationFolder.domElement?.setAttribute(
    'title',
    'Navigation: Controls how you move and look around the 3D scene\n\n' +
      '• Orbit: Drag to pan, right-drag to rotate freely (no gimbal lock), scroll to zoom.\n' +
      '  Best for inspecting objects from any angle.\n' +
      '• Fly: Move freely through the scene like a drone (WASD + arrows).\n' +
      '  Best for exploring large or spatially complex datasets.\n' +
      '• Ortho: Orthographic projection with pan and zoom (no perspective).\n' +
      '  Best for 2D data and precise measurements.'
  );

  // Control type selector
  const controlTypeControl = navigationFolder
    .add(settings, 'controlType', ['orbit', 'fly', 'ortho'])
    .name('Control Type')
    .onChange((value: 'orbit' | 'fly' | 'ortho') => {
      sceneManager.setControlType(value);
      saveSettings();
      triggerAnimation();

      // Show/hide relevant controls
      updateNavigationControls(value);
    });

  // Store reference for updates
  controllers.controlType = controlTypeControl;

  // Set tooltip for control type
  controlTypeControl.domElement.setAttribute(
    'title',
    'Camera Control Type\n' +
      '• Orbit: Drag to pan, right-drag to rotate (no gimbal lock), Shift+scroll to roll\n' +
      '• Fly: First-person flying controls (WASD to move, arrows to look)\n' +
      '• Ortho: Orthographic projection with pan and zoom (no perspective distortion)'
  );

  // Create sub-folders for each control type
  const orbitFolder = navigationFolder.addFolder('Orbit Controls');
  orbitFolder.domElement?.setAttribute(
    'title',
    'Orbit Controls: Settings for orbit camera mode\n' +
      '• Auto-rotate for hands-free viewing\n' +
      '• Adjust rotation speed for presentations'
  );

  const flyFolder = navigationFolder.addFolder('Fly Controls');
  flyFolder.domElement?.setAttribute(
    'title',
    'Fly Controls: Settings for first-person flying camera mode\n' +
      '• WASD keys to move, arrow keys to look around\n' +
      '• Alt+W/S for vertical movement\n' +
      '• Switch between direct and inertial (momentum-based) movement'
  );

  // Auto-rotation controls (for orbit mode)
  const autoRotateControl = orbitFolder
    .add(settings, 'autoRotate')
    .name('Auto Rotate')
    .onChange((value: boolean) => {
      sceneManager.setAutoRotate(value);
      saveSettings();
      // Need to keep animation running when auto-rotating
      if (value) {
        animationController?.startAnimation();
      }
    });

  // Store reference
  controllers.autoRotate = autoRotateControl;

  // Set tooltip for auto-rotation
  autoRotateControl.domElement.setAttribute(
    'title',
    'Auto Rotate: Continuously orbit camera around the scene\n' +
      '• Creates cinematic rotating view\n' +
      '• Useful for presentations and showcases\n' +
      '• Click and drag to manually control camera'
  );

  const rotationSpeedControl = orbitFolder
    .add(settings, 'autoRotateSpeed', 0.1, 5, 0.1)
    .name('Rotation Speed')
    .onChange((value: number) => {
      sceneManager.setAutoRotateSpeed(value);
      saveSettings();
      triggerAnimation();
    });

  // Store reference
  controllers.autoRotateSpeed = rotationSpeedControl;

  // Set tooltip for rotation speed
  rotationSpeedControl.domElement.setAttribute(
    'title',
    'Rotation Speed: How fast the camera orbits\n' +
      '• 0.1 = Very slow (10 minutes per rotation)\n' +
      '• 0.25 = Slow (4 minutes per rotation, default)\n' +
      '• 1.0 = Medium (60 seconds per rotation)\n' +
      '• 5.0 = Fast (12 seconds per rotation)'
  );

  // Natural drag — touchpad-friendly orbit-mode mapping (LEFT=rotate,
  // RIGHT=pan). Defaults to ON on macOS via RenderingSettings defaults.
  const naturalDragControl = orbitFolder
    .add(settings, 'naturalDrag')
    .name('Natural drag')
    .onChange((value: boolean) => {
      sceneManager.setNaturalDrag(value);
      saveSettings();
      triggerAnimation();
    });
  controllers.naturalDrag = naturalDragControl;
  naturalDragControl.domElement.setAttribute(
    'title',
    'Natural drag: invert left- and right-drag for touchpad ergonomics\n' +
      '• OFF: left-drag pans, right-drag rotates (mouse-default)\n' +
      '• ON: left-drag rotates, right-drag pans (touchpad-friendly)\n' +
      '• Shift+left-drag always picks the inverse action\n' +
      '• Defaults to ON on macOS'
  );

  // Fly controls settings - use ranges from config.controls.fly
  const flyMovementConfig = config.controls.fly.movement.speed;
  const flySpeedControl = flyFolder
    .add(
      settings,
      'flyMovementSpeed',
      flyMovementConfig.min,
      flyMovementConfig.max,
      flyMovementConfig.step || 0.1
    )
    .name('Movement Speed')
    .onChange((value: number) => {
      sceneManager.setFlyMovementSpeed(value);
      saveSettings();
    });

  // Store reference
  controllers.flyMovementSpeed = flySpeedControl;

  flySpeedControl.domElement.setAttribute(
    'title',
    'Movement Speed: How fast you move in fly mode\n' +
      '• Units per second (or acceleration in inertial mode)\n' +
      '• Use WASD keys to move\n' +
      '• Alt+W/S for vertical movement'
  );

  const flyRotationConfig = config.controls.fly.rotation.speed;
  const flyRotationSpeedControl = flyFolder
    .add(
      settings,
      'flyRotationSpeed',
      flyRotationConfig.min,
      flyRotationConfig.max,
      flyRotationConfig.step || 0.1
    )
    .name('Rotation Speed')
    .onChange((value: number) => {
      sceneManager.setFlyRotationSpeed(value);
      saveSettings();
    });

  // Store reference
  controllers.flyRotationSpeed = flyRotationSpeedControl;

  flyRotationSpeedControl.domElement.setAttribute(
    'title',
    'Rotation Speed: How fast the camera rotates\n' +
      '• Radians per second (or acceleration in inertial mode)\n' +
      '• Use arrow keys to rotate: ↑↓←→\n' +
      '• Mouse drag also rotates camera'
  );

  const flyInertialControl = flyFolder
    .add(settings, 'flyInertialMode')
    .name('Inertial Mode')
    .onChange((value: boolean) => {
      sceneManager.setFlyInertialMode(value);
      saveSettings();
      // Show/hide damping controls
      if (value) {
        flyDampingControl.show();
        flyRotationDampingControl.show();
      } else {
        flyDampingControl.hide();
        flyRotationDampingControl.hide();
      }
    });

  // Store reference
  controllers.flyInertialMode = flyInertialControl;

  flyInertialControl.domElement.setAttribute(
    'title',
    'Movement Mode\n' +
      '• Direct: Immediate velocity control (stop when key released)\n' +
      '• Inertial: Acceleration-based with momentum (drift to stop)'
  );

  const flyDampingConfig = config.controls.fly.movement.damping;
  const flyDampingControl = flyFolder
    .add(
      settings,
      'flyDamping',
      flyDampingConfig.min,
      flyDampingConfig.max,
      flyDampingConfig.step || 0.0001
    )
    .name('Translation Damping')
    .onChange((value: number) => {
      sceneManager.setFlyDamping(value);
      saveSettings();
    });

  // Store reference
  controllers.flyDamping = flyDampingControl;

  flyDampingControl.domElement.setAttribute(
    'title',
    'Translation Damping (Inertial Mode Only)\n' +
      '• Controls how quickly movement slows down\n' +
      '• 0.90 = Quick stop\n' +
      '• 0.97 = Moderate drift\n' +
      '• 0.999 = Long drift (default)\n' +
      '• 0.9999 = Very long drift'
  );

  const flyRotationDampingControl = flyFolder
    .add(settings, 'flyRotationDamping', 0.9, 0.9999, 0.0001)
    .name('Rotation Damping')
    .onChange((value: number) => {
      sceneManager.setFlyRotationDamping(value);
      saveSettings();
    });

  // Store reference
  controllers.flyRotationDamping = flyRotationDampingControl;

  flyRotationDampingControl.domElement.setAttribute(
    'title',
    'Rotation Damping (Inertial Mode Only)\n' +
      '• Controls how quickly rotation slows down\n' +
      '• 0.90 = Quick stop\n' +
      '• 0.97 = Moderate drift\n' +
      '• 0.999 = Long drift (default)\n' +
      '• 0.9999 = Very long drift'
  );

  // Initially show/hide based on current control type
  updateNavigationControls(settings.controlType);

  // Hide damping controls if not in inertial mode
  if (!settings.flyInertialMode) {
    flyDampingControl.hide();
    flyRotationDampingControl.hide();
  }

  return {
    controllers,
    folders: {
      orbitFolder,
      flyFolder,
    },
  };
}
