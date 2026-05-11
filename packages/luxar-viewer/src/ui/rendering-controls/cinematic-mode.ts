/**
 * Cinematic Mode preset for the rendering-controls panel.
 *
 * Encapsulates the C-key toggle that flips a film-look preset:
 * ACES tone mapping, detector noise, vignette, chromatic lens distortion,
 * and a 35 mm wide-angle FOV. On enable, all affected settings are
 * snapshot. On disable, each setting is restored from the snapshot
 * unless the user manually changed it (dirty-check).
 *
 * The class owns the snapshot field so the facade does not need to
 * carry that state — only a stable `getSnapshot()` / `setSnapshot()`
 * pair if external code wants to clear it (e.g. resetToDefaults).
 */

import * as THREE from 'three';
import { config, type RenderingSettings } from '../../config';
import { log, Modules } from '../../utils/log';
import type { PostProcessingManager } from '../../rendering/post-processing/post-processing-manager';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation-controller';
import type { RenderingControllers } from '../../controls/types';

/** String → THREE.ToneMapping map shared between applySettings and the cinematic toggle. */
export const TONE_MAPPING_MAP: Record<string, THREE.ToneMapping> = {
  None: THREE.NoToneMapping,
  Linear: THREE.LinearToneMapping,
  Reinhard: THREE.ReinhardToneMapping,
  Cineon: THREE.CineonToneMapping,
  ACES: THREE.ACESFilmicToneMapping,
  AgX: THREE.AgXToneMapping,
  Neutral: THREE.NeutralToneMapping,
};

/** Keys of RenderingSettings that cinematic mode touches. */
export type CinematicSnapshotKeys =
  | 'toneMapping'
  | 'detectorNoiseEnabled'
  | 'detectorNoiseReadoutSigma'
  | 'detectorNoisePhotonGain'
  | 'detectorNoiseFpnSigma'
  | 'vignetteEnabled'
  | 'chromaticLensDistortionEnabled'
  | 'chromaticLensDistortionX'
  | 'chromaticLensDistortionY'
  | 'chromaticLensDispersion'
  | 'chromaticLensPrincipalPointX'
  | 'chromaticLensPrincipalPointY'
  | 'chromaticLensFocalLengthX'
  | 'chromaticLensFocalLengthY'
  | 'chromaticLensSkew'
  | 'fov'
  | 'fovPreset';

export type CinematicSnapshot = Pick<RenderingSettings, CinematicSnapshotKeys>;

const CINEMATIC_SNAPSHOT_KEYS: CinematicSnapshotKeys[] = [
  'toneMapping',
  'detectorNoiseEnabled',
  'detectorNoiseReadoutSigma',
  'detectorNoisePhotonGain',
  'detectorNoiseFpnSigma',
  'vignetteEnabled',
  'chromaticLensDistortionEnabled',
  'chromaticLensDistortionX',
  'chromaticLensDistortionY',
  'chromaticLensDispersion',
  'chromaticLensPrincipalPointX',
  'chromaticLensPrincipalPointY',
  'chromaticLensFocalLengthX',
  'chromaticLensFocalLengthY',
  'chromaticLensSkew',
  'fov',
  'fovPreset',
];

/** Cinematic-ON values used both for apply and for the dirty-check on restore. */
export function buildCinematicValues(): CinematicSnapshot {
  const lens35 = config.camera.lensDistortionPresets['35mm'];
  return {
    toneMapping: 'ACES',
    detectorNoiseEnabled: true,
    detectorNoiseReadoutSigma: 0.002,
    detectorNoisePhotonGain: 0.002,
    detectorNoiseFpnSigma: 0.001,
    vignetteEnabled: true,
    chromaticLensDistortionEnabled: true,
    chromaticLensDistortionX: lens35.distortionX,
    chromaticLensDistortionY: lens35.distortionY,
    chromaticLensDispersion: lens35.dispersion,
    chromaticLensPrincipalPointX: lens35.principalPointX,
    chromaticLensPrincipalPointY: lens35.principalPointY,
    chromaticLensFocalLengthX: lens35.focalLengthX,
    chromaticLensFocalLengthY: lens35.focalLengthY,
    chromaticLensSkew: lens35.skew,
    fov: config.camera.fovPresets['35mm'],
    fovPreset: '35mm',
  };
}

export interface CinematicContext {
  settings: RenderingSettings;
  postProcessing: PostProcessingManager;
  sceneManager: SceneManager;
  controllers: RenderingControllers;
  animationController?: AnimationController;
  saveSettings: () => void;
  triggerAnimation: () => void;
  /** Refresh every controller in the GUI tree (typically `gui.controllersRecursive().forEach(...updateDisplay)`). */
  refreshAllControllers: () => void;
}

/**
 * Owns the cinematic-mode snapshot and exposes the toggle/checkbox-update
 * methods the facade used to host directly.
 */
export class CinematicModeController {
  private snapshot: CinematicSnapshot | null = null;

  constructor(private readonly context: CinematicContext) {}

  /** Clear the snapshot — called from resetToDefaults / loadSettings. */
  clearSnapshot(): void {
    this.snapshot = null;
  }

  /**
   * Recompute the cinematic-mode checkbox from the current effects state
   * (majority vote over the four signal effects, including ACES tone mapping).
   */
  updateCheckbox(): void {
    const { settings, controllers } = this.context;
    const cinematicEffects = [
      settings.detectorNoiseEnabled,
      settings.vignetteEnabled,
      settings.chromaticLensDistortionEnabled,
      settings.toneMapping === 'ACES',
    ];

    const enabledCount = cinematicEffects.filter(Boolean).length;
    const isEnabled = enabledCount >= cinematicEffects.length / 2;

    settings.cinematicMode = isEnabled;
    if (controllers.cinematicMode) {
      controllers.cinematicMode.updateDisplay();
    }
  }

  /**
   * Toggle cinematic mode. Uses the same majority-vote algorithm as the
   * checkbox: enables if < 50 % of signal effects are on, otherwise disables.
   * Snapshots / restores the affected settings (with dirty-check on restore).
   */
  toggle(): void {
    const ctx = this.context;
    const { settings, postProcessing, sceneManager, animationController } = ctx;

    const cinematicEffects = [
      settings.detectorNoiseEnabled,
      settings.vignetteEnabled,
      settings.chromaticLensDistortionEnabled,
      settings.toneMapping === 'ACES',
    ];

    const enabledCount = cinematicEffects.filter(Boolean).length;
    const shouldEnableAll = enabledCount < cinematicEffects.length / 2;

    const cinematicValues = buildCinematicValues();

    if (shouldEnableAll) {
      // ENABLE: snapshot current settings, then apply cinematic values.
      const snapshot = {} as CinematicSnapshot;
      for (const key of CINEMATIC_SNAPSHOT_KEYS) {
        (snapshot as Record<string, unknown>)[key] = settings[key];
      }
      this.snapshot = snapshot;

      Object.assign(settings, cinematicValues);
    } else if (this.snapshot) {
      // DISABLE: restore from snapshot (dirty-check per setting).
      for (const key of CINEMATIC_SNAPSHOT_KEYS) {
        if (settings[key] === cinematicValues[key]) {
          (settings as unknown as Record<string, unknown>)[key] = this.snapshot[key];
        }
      }
      this.snapshot = null;
    } else {
      // No snapshot (e.g. loaded from localStorage with cinematic on).
      // Fall back to non-cinematic defaults.
      settings.toneMapping = config.renderingControls.defaults.toneMapping;
      settings.detectorNoiseEnabled = false;
      settings.vignetteEnabled = false;
      settings.chromaticLensDistortionEnabled = false;
      const lens50 = config.camera.lensDistortionPresets['50mm Normal'];
      settings.chromaticLensDistortionX = lens50.distortionX;
      settings.chromaticLensDistortionY = lens50.distortionY;
      settings.chromaticLensDispersion = lens50.dispersion;
      settings.chromaticLensPrincipalPointX = lens50.principalPointX;
      settings.chromaticLensPrincipalPointY = lens50.principalPointY;
      settings.chromaticLensFocalLengthX = lens50.focalLengthX;
      settings.chromaticLensFocalLengthY = lens50.focalLengthY;
      settings.chromaticLensSkew = lens50.skew;
      settings.fov = config.camera.fovPresets['50mm Normal'];
      settings.fovPreset = '50mm Normal';
    }

    // Apply post-processing changes via deferred rebuild. The
    // try/finally guarantees `endDeferRebuild()` runs even if any
    // sub-setter throws, so the depth counter cannot strand above zero
    // and silently disable future rebuilds.
    postProcessing.startDeferRebuild();
    try {
      postProcessing.setToneMapping(TONE_MAPPING_MAP[settings.toneMapping]);

      postProcessing.setDetectorNoiseEnabled(
        settings.detectorNoiseEnabled,
        settings.detectorNoiseReadoutSigma,
        settings.detectorNoisePhotonGain,
        settings.detectorNoiseFpnSigma
      );

      postProcessing.setVignetteEnabled(
        settings.vignetteEnabled,
        settings.vignetteDarkness,
        settings.vignetteOffset
      );

      postProcessing.setChromaticLensDistortionEnabled(
        settings.chromaticLensDistortionEnabled,
        settings.chromaticLensDistortionX,
        settings.chromaticLensDistortionY,
        settings.chromaticLensDispersion,
        settings.chromaticLensPrincipalPointX,
        settings.chromaticLensPrincipalPointY,
        settings.chromaticLensFocalLengthX,
        settings.chromaticLensFocalLengthY,
        settings.chromaticLensSkew
      );
    } finally {
      postProcessing.endDeferRebuild();
    }

    // Apply FOV change to camera.
    const targetFOV = settings.fov;
    const currentFOV = sceneManager.currentFov;
    if (Math.abs(currentFOV - targetFOV) > 0.5) {
      const delta = (targetFOV - currentFOV) / config.camera.fovSensitivity;
      sceneManager.updateFOV(delta);
    }

    ctx.refreshAllControllers();

    ctx.saveSettings();
    ctx.triggerAnimation();

    if (settings.detectorNoiseEnabled) {
      animationController?.startAnimation();
    }

    this.updateCheckbox();

    const modeText = shouldEnableAll ? 'enabled' : 'disabled';
    log.info(
      Modules.RENDERER,
      `Cinematic mode ${modeText}: tone=${settings.toneMapping}, ` +
        `noise=${settings.detectorNoiseEnabled}, vignette=${settings.vignetteEnabled}, ` +
        `lens=${settings.chromaticLensDistortionEnabled}, FOV=${settings.fovPreset}`
    );
  }
}
