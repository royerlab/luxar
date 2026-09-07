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
  densityGuardEnabled?: Controller;
  // Cinematic mode
  cinematicMode?: Controller;
}

/**
 * Runtime handle on the projected-density guard, for the Performance popover
 * toggle and the settings persistence layer. Implemented by
 * `core/app/init/density-guard-wiring` (which owns the tracker + ladder).
 */
export interface DensityGuardControl {
  isEnabled(): boolean;
  /**
   * True when `?no-density-guard` turned the guard off for this session. The
   * stored per-scene setting is then neither applied nor overwritten, the
   * same way a URL DPR pin leaves the stored DPR flags alone.
   */
  sessionDisabled: boolean;
  /** Turn the guard on or off live: off releases every thinned node and resumes deferred rungs. */
  setEnabled(enabled: boolean): void;
  /** Nodes currently thinned and the smallest keep fraction among them (1 when none). */
  thinning(): { nodes: number; minKeep: number };
  /** The effective blendable cap (elements per pixel), `?density-cap=N` included. */
  capElementsPerPixel(): number;
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
