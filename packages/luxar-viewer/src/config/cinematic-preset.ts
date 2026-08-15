/**
 * The cinematic-mode preset: the film-look values the C key / rail item
 * applies, and that a scene's `viewer_config.cinematic_mode = True` expands
 * into at load time.
 *
 * This lives in `config/` rather than next to the UI controller because BOTH
 * consumers need it: `ui/rendering-controls/cinematic-mode.ts` (the toggle)
 * and `config/zarr-bridge/viewer-config-utils.ts` (the zarr bridge). The
 * bridge must not import from `ui/`, so the values moved down here and the UI
 * module re-exports them for its existing importers.
 *
 * Import discipline: this module deliberately imports the camera SECTION
 * (`./sections/camera/data`) rather than the aggregate `config` object from
 * `./index`. The section is a leaf — it imports only its own types — so there
 * is no chance of a cycle with the zarr-bridge consumer.
 */

import { cameraConfig } from './sections/camera/data';
import type { RenderingSettings } from './types';

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

/** The subset of RenderingSettings that cinematic mode snapshots and restores. */
export type CinematicSnapshot = Pick<RenderingSettings, CinematicSnapshotKeys>;

/**
 * Runtime companion to {@link CinematicSnapshotKeys} — the iteration order used
 * to snapshot, restore, and expand the preset key by key.
 */
export const CINEMATIC_SNAPSHOT_KEYS: CinematicSnapshotKeys[] = [
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
  const lens35 = cameraConfig.lensDistortionPresets['35mm'];
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
    fov: cameraConfig.fovPresets['35mm'],
    fovPreset: '35mm',
  };
}
