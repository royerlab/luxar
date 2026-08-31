/**
 * Shared types for rendering controls setup modules.
 *
 * Each setup module exports a function that creates GUI controls for a specific
 * category (navigation, camera, HDR, etc.) and returns controller references.
 */

import type GUI from '../gui';
import type { Controller } from '../gui';
import type { RenderingSettings } from '../../config';
import type { PostProcessingManager } from '../../rendering';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';

/** Rendering controls GUI controllers for the entire rendering panel. */
export interface RenderingControllers {
  controlType?: Controller;
  // Camera controls
  fov?: Controller;
  fovPreset?: Controller;
  nearPlane?: Controller;
  farPlane?: Controller;
  // Dynamic clipping controls
  dynamicClippingEnabled?: Controller;
  // HDR controls (global EOG)
  exposure?: Controller;
  globalOffset?: Controller;
  globalGamma?: Controller;
  // Chromatic lens distortion controls
  chromaticLensDistortionX?: Controller;
  chromaticLensDistortionY?: Controller;
  chromaticLensDispersion?: Controller;
  chromaticLensPrincipalPointX?: Controller;
  chromaticLensPrincipalPointY?: Controller;
  chromaticLensFocalLengthX?: Controller;
  chromaticLensFocalLengthY?: Controller;
  chromaticLensSkew?: Controller;
  // Orbit controls
  autoRotate?: Controller;
  autoRotateSpeed?: Controller;
  naturalDrag?: Controller;
  // Fly controls
  flyMovementSpeed?: Controller;
  flyRotationSpeed?: Controller;
  flyInertialMode?: Controller;
  flyDamping?: Controller;
  flyRotationDamping?: Controller;
  // Performance controls
  adaptiveDPREnabled?: Controller;
  allowHighDPR?: Controller;
  // Cinematic mode
  cinematicMode?: Controller;
}

/**
 * Context object passed to all setup functions.
 * Contains all dependencies needed to create controls.
 */
export interface SetupContext {
  /** The GUI instance or folder to add controls to */
  gui: GUI;

  /** Current rendering settings (mutable) */
  settings: RenderingSettings;

  /** Reference to post-processing manager */
  postProcessing: PostProcessingManager;

  /** Reference to scene manager */
  sceneManager: SceneManager;

  /** Reference to animation controller (optional) */
  animationController?: AnimationController;

  /** Callback to save settings to localStorage */
  saveSettings: () => void;

  /** Callback to trigger a single animation frame render */
  triggerAnimation: () => void;

  /** Callback to update clipping controls enabled/disabled state */
  updateClippingControlsState: (dynamicEnabled: boolean) => void;

  /** Callback to update navigation controls visibility based on control type */
  updateNavigationControls: (controlType: 'orbit' | 'fly' | 'ortho') => void;
}

/**
 * Result returned by setup functions.
 * Contains controller references and any folder references needed for visibility toggling.
 */
export interface SetupResult {
  /** Map of controller references for programmatic updates */
  controllers: Partial<RenderingControllers>;

  /** Shadow objects for special UI patterns (optional) */
  shadowObjects?: {
    hdrLogValue?: { log: number };
    [key: string]: unknown;
  };
}
