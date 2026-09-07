/**
 * User-adjustable rendering settings that can be persisted
 */

import type { AutoRotateAxis } from '../../../controls/types';

export interface RenderingSettings {
  // Camera settings
  fov: number;
  fovPreset: '28mm Wide' | '35mm' | '50mm Normal' | '85mm Portrait' | '135mm Tele' | 'Custom';
  near: number;
  far: number;
  // Dynamic clipping planes
  dynamicClippingEnabled: boolean;
  // Rendering effects (bloom is now the single source of truth)
  bloomEnabled: boolean;
  bloomThreshold: number;
  bloomStrength: number;
  bloomRadius: number;
  bloomLevels: number;
  exposure: number; // Log2 stops, default 0.0
  globalOffset: number; // Additive shift, default 0.0
  globalGamma: number; // Midtone curve, default 1.0
  fxaaEnabled: boolean;
  msaaEnabled: boolean;
  msaaSamples: number;
  ssaaEnabled: boolean;
  ssaaMultiplier: number;
  toneMapping: 'None' | 'Linear' | 'Reinhard' | 'Cineon' | 'ACES' | 'AgX' | 'Neutral';
  vignetteEnabled: boolean;
  vignetteDarkness: number;
  vignetteOffset: number;
  // Detector noise effect (physics-based: Poisson + Gaussian + FPN)
  detectorNoiseEnabled: boolean;
  detectorNoiseReadoutSigma: number;
  detectorNoisePhotonGain: number;
  detectorNoiseFpnSigma: number;
  // Chromatic lens distortion effect
  chromaticLensDistortionEnabled: boolean;
  chromaticLensDistortionX: number;
  chromaticLensDistortionY: number;
  chromaticLensDispersion: number;
  chromaticLensPrincipalPointX: number;
  chromaticLensPrincipalPointY: number;
  chromaticLensFocalLengthX: number;
  chromaticLensFocalLengthY: number;
  chromaticLensSkew: number;
  // Navigation controls
  controlType: 'orbit' | 'fly' | 'ortho';
  autoRotate: boolean;
  autoRotateSpeed: number;
  /**
   * Camera-frame or fixed scene axis the orbit turntable revolves around; see
   * {@link AutoRotateAxis}. Orbit only; ortho disables rotation and fly has no
   * turntable.
   */
  autoRotateAxis: AutoRotateAxis;
  /**
   * Auto-dolly: oscillate the viewing distance on a sine — the turntable's
   * radial sibling. Unlike the turntable it is alive in ortho too, where it
   * breathes `camera.zoom`.
   */
  autoDolly: boolean;
  /**
   * Peak dolly swing as a PERCENT of the viewing distance (15 → ±15%), which
   * is what the slider shows. The control itself holds the fraction; see
   * `dollyAmplitudeFromPercent` in `controls/types.ts`.
   */
  autoDollyAmplitudePercent: number;
  /** Seconds per full dolly oscillation. */
  autoDollyPeriod: number;
  /**
   * "Natural drag" — swap LEFT ↔ RIGHT mouse buttons in orbit mode so a
   * one-finger touchpad drag rotates (and two-finger / right-drag pans).
   * Defaults to true on macOS. Orbit (3D) only; ortho and fly modes ignore.
   */
  naturalDrag: boolean;
  // Orbit feel — added at runtime from config.controls.orbit (also applied
  // to ortho: same control class, shared feel).
  orbitZoomSpeed?: number;
  orbitDampingFactor?: number;
  // Fly controls - these are added at runtime from config.controls.fly
  flyMovementSpeed?: number;
  flyRotationSpeed?: number;
  flyLookSpeed?: number;
  flyInertialMode?: boolean;
  flyDamping?: number;
  flyRotationDamping?: number;
  // Adaptive resolution
  adaptiveDPREnabled: boolean;
  /** Whether the viewer may render above CSS resolution on a HiDPI display. */
  allowHighDPR: boolean;
  /**
   * Runtime toggle for the projected-density guard (shader thinning of
   * over-dense blendable nodes + the refinement rung cap). Persisted per
   * scene; `?no-density-guard` overrides it for one session without
   * touching the stored value.
   */
  densityGuardEnabled: boolean;
  // Cinematic mode toggle (for UI only, actual state determined by effects)
  cinematicMode: boolean;
}

/**
 * Rendering controls configuration
 */
export interface RenderingControlsConfig {
  defaults: RenderingSettings;
}
