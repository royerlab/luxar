/**
 * Cinematic Mode preset for the rendering-controls panel.
 *
 * Encapsulates the C-key toggle that flips a film-look preset:
 * ACES tone mapping, a subtle wide bloom, detector noise, vignette,
 * chromatic lens distortion, and a 35 mm wide-angle FOV.
 * On enable, all affected settings are
 * snapshot. On disable, each setting is restored from the snapshot
 * unless the user manually changed it (dirty-check).
 *
 * The class owns the snapshot field so the facade does not need to
 * carry that state — only a stable `getSnapshot()` / `setSnapshot()`
 * pair if external code wants to clear it (e.g. resetToDefaults).
 */

import { config, type RenderingSettings } from '../../config';
import { toneMappingFromName } from '../../rendering/post-processing/tone-mapping';
import { log, Modules } from '../../utils/log';
import type { PostProcessingManager } from '../../rendering';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import type { RenderingControllers } from './types';

/** Keys of RenderingSettings that cinematic mode touches. */
export type CinematicSnapshotKeys =
  | 'toneMapping'
  | 'bloomEnabled'
  | 'bloomThreshold'
  | 'bloomStrength'
  | 'bloomRadius'
  | 'bloomLevels'
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
  'bloomEnabled',
  'bloomThreshold',
  'bloomStrength',
  'bloomRadius',
  'bloomLevels',
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
    // Subtle, wide glow: a near-zero threshold so mid-tones contribute, a
    // gentle strength so the halo reads as lens veiling rather than a bloom
    // effect, and the full mipmap ladder for a smooth spread.
    bloomEnabled: true,
    bloomThreshold: 0.01,
    bloomStrength: 0.05,
    bloomRadius: 1.0,
    bloomLevels: 8,
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
   *
   * Bloom is applied by the preset but deliberately kept OUT of the vote: it
   * is commonly enabled on its own for HDR data, so counting it would flip the
   * checkbox on scenes that are not cinematic at all.
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
      const defaults = config.renderingControls.defaults;
      settings.toneMapping = defaults.toneMapping;
      settings.bloomEnabled = defaults.bloomEnabled;
      settings.bloomThreshold = defaults.bloomThreshold;
      settings.bloomStrength = defaults.bloomStrength;
      settings.bloomRadius = defaults.bloomRadius;
      settings.bloomLevels = defaults.bloomLevels;
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

    // Batch post-processing changes through `withDeferredRebuild` so the
    // depth counter unwinds even when a sub-setter throws.
    postProcessing.withDeferredRebuild(() => {
      postProcessing.setToneMapping(toneMappingFromName(settings.toneMapping));

      postProcessing.setBloomEnabled(
        settings.bloomEnabled,
        settings.bloomStrength,
        settings.bloomRadius,
        settings.bloomThreshold
      );
      postProcessing.setBloomLevels(settings.bloomLevels);

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
    });

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
        `bloom=${settings.bloomEnabled}, ` +
        `noise=${settings.detectorNoiseEnabled}, vignette=${settings.vignetteEnabled}, ` +
        `lens=${settings.chromaticLensDistortionEnabled}, FOV=${settings.fovPreset}`
    );
  }
}
