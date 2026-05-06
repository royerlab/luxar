/**
 * Shared types for rendering controls setup modules.
 *
 * Each setup module exports a function that creates GUI controls for a specific
 * category (navigation, camera, HDR, etc.) and returns controller references.
 */

import type GUI from '../gui';
import type { Folder } from '../gui';
import type { RenderingSettings } from '../../config';
import type { PostProcessingManager } from '../../rendering/post-processing/post-processing-manager';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation-controller';
import type { RenderingControllers } from '../../controls/types';

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

  /** Folder references for visibility control (optional) */
  folders?: {
    orbitFolder?: Folder;
    flyFolder?: Folder;
    [key: string]: Folder | undefined;
  };

  /** Shadow objects for special UI patterns (optional) */
  shadowObjects?: {
    hdrLogValue?: { log: number };
    [key: string]: any;
  };
}
