import { isMacPlatform } from '../../../utils/platform';
import type { RenderingControlsConfig } from './types';

/**
 * Rendering controls configuration with user-adjustable defaults
 */
export const renderingControlsConfig: RenderingControlsConfig = {
  defaults: {
    // Camera settings
    fov: 47, // Field of view in degrees (50mm Normal, default matches camera.fov)
    fovPreset: '50mm Normal', // Default to normal lens equivalent
    near: 0.1, // Near clipping plane (default matches camera.near)
    far: 1000, // Far clipping plane (default matches camera.far)
    // Dynamic clipping planes
    dynamicClippingEnabled: true, // Auto-adjust clipping planes based on camera position
    // Bloom settings - single source of truth
    bloomEnabled: false, // Enable/disable bloom effect (opt-in via zarr viewer_config)
    bloomThreshold: 0.01, // Luminance threshold (0-1), lower = more bloom, higher = less bloom
    bloomStrength: 0.25, // Bloom intensity multiplier
    bloomRadius: 1.0, // Blur radius for bloom spread
    bloomLevels: 8, // Number of mipmap levels (1-12, lower = coarser/faster, higher = smoother)
    exposure: 0.0, // Global exposure in log2 stops (0 = neutral, +1 = 2x brighter)
    globalOffset: 0.0, // Global additive brightness shift
    globalGamma: 1.0, // Global gamma correction (1.0 = linear)
    fxaaEnabled: false, // FXAA disabled by default
    msaaEnabled: false, // MSAA disabled by default (enable for fast hardware-accelerated AA)
    msaaSamples: 4, // MSAA sample count (2, 4, 8)
    ssaaEnabled: false, // SSAA disabled by default (highest quality, heavy performance cost)
    ssaaMultiplier: 2.0, // SSAA resolution multiplier (1.5x, 2x, 4x)
    // Tone mapping (ACES gives the most consistent, pleasing HDR look).
    // Note: ACES intentionally shifts hues, which can distort color LUTs;
    // the Python compiler warns when a colormap LUT is used so authors can
    // switch to None — an exact passthrough — if exact hue fidelity
    // matters and the scene stays inside [0, 1]. (Neutral is not a passthrough:
    // even below its knee it subtracts an offset taken from the channel
    // minimum, so anything but a fully saturated colour moves, and well above
    // 1.0 it desaturates hard — hue is kept, chroma is not.)
    toneMapping: 'ACES' as const,
    vignetteEnabled: false, // Vignette disabled by default
    vignetteDarkness: 0.5, // Vignette darkness (0-1)
    vignetteOffset: 0.5, // Vignette offset from center (0-1)
    // Detector noise effect settings (physics-based: Poisson + Gaussian + FPN)
    detectorNoiseEnabled: false, // Detector noise disabled by default
    detectorNoiseReadoutSigma: 0.002, // Temporal readout noise sigma (0-0.1)
    detectorNoisePhotonGain: 0.002, // Photon gain for shot noise visibility (0.0001-0.1)
    detectorNoiseFpnSigma: 0.001, // Fixed pattern noise sigma (0-0.05)
    // Chromatic lens distortion effect settings
    chromaticLensDistortionEnabled: false, // Chromatic lens distortion disabled by default
    chromaticLensDistortionX: 0, // Radial distortion coefficient X (50mm Normal: no distortion)
    chromaticLensDistortionY: 0, // Radial distortion coefficient Y (50mm Normal: no distortion)
    chromaticLensDispersion: 0.02, // Chromatic dispersion strength (50mm Normal: minimal)
    chromaticLensPrincipalPointX: 0, // Principal point offset X
    chromaticLensPrincipalPointY: 0, // Principal point offset Y
    chromaticLensFocalLengthX: 1.0, // Focal length X (50mm Normal: neutral)
    chromaticLensFocalLengthY: 1.0, // Focal length Y (50mm Normal: neutral)
    chromaticLensSkew: 0, // Skew in radians
    // Navigation controls
    controlType: 'orbit' as const, // Default to orbit controls
    autoRotate: false, // Auto-rotation disabled by default
    autoRotateSpeed: 0.25, // Slow rotation speed for presentations
    // Screen-vertical turntable — the axis auto-rotation always used, so
    // every existing scene and persisted setting keeps its exact behavior.
    autoRotateAxis: 'vertical' as const,
    // Touchpad-friendly orbit drag mapping (LEFT=rotate, RIGHT=pan).
    // Default-on for Mac users; off elsewhere. The rendering-controls
    // persistence layer overrides this with the user's stored choice.
    naturalDrag: isMacPlatform(),
    // Note: Fly control settings are referenced directly from controls.fly to avoid duplication
    // Adaptive resolution (runtime/UI toggle; overrides adaptiveDPR.enabled after init)
    adaptiveDPREnabled: true, // Persisted per-scene via localStorage
    // Cinematic mode (disabled by default)
    cinematicMode: false,
  },
};
