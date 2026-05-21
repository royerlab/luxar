import type { CameraConfig } from './types';

/**
 * Camera configuration for 3D perspective and navigation
 *
 * Note: fov, near, far live in renderingControls.defaults as the single source of truth.
 */
export const cameraConfig: CameraConfig = {
  initialPosition: { x: 0, y: 0, z: 8 }, // Initial camera position in 3D space (world coordinates)
  fovMin: 10, // Minimum field of view for zoom limits - prevents excessive zoom-in
  fovMax: 170, // Maximum field of view for zoom limits - must be <180° (fish-eye territory)
  fovSensitivity: 0.05, // FOV change sensitivity for Ctrl+wheel input - lower = finer control
  // FOV presets based on 35mm equivalent focal lengths (horizontal FOV - photography standard)
  fovPresets: {
    '28mm Wide': 75, // Wide angle - 75° horizontal FOV, good for large scenes and landscapes
    '35mm': 63, // Wide normal - 63° horizontal FOV, comfortable wide viewing
    '50mm Normal': 47, // Normal lens - 47° horizontal FOV, closest to human vision
    '85mm Portrait': 29, // Portrait lens - 29° horizontal FOV, good for isolating subjects
    '135mm Tele': 18, // Telephoto - 18° horizontal FOV, extreme subject isolation
    Custom: -1, // Custom value - preserves current FOV slider setting
  },
  // Lens distortion presets matching realistic lens characteristics for each focal length
  // Negative distortion = barrel (wide angle), positive = pincushion (telephoto)
  // Dispersion values simulate chromatic aberration (wavelength-dependent refraction)
  lensDistortionPresets: {
    '28mm Wide': {
      distortionX: -0.07,
      distortionY: -0.07, // Barrel distortion from your screenshot
      principalPointX: 0,
      principalPointY: 0, // Centered (standard)
      focalLengthX: 1.075,
      focalLengthY: 1.08, // Focal length values from your screenshot
      skew: 0, // No skew (perfect optics)
      dispersion: 0.05, // Wide angle = more chromatic aberration (higher light bending angles)
    },
    '35mm': {
      distortionX: -0.05,
      distortionY: -0.05, // Moderate barrel distortion from your screenshot
      principalPointX: 0,
      principalPointY: 0,
      focalLengthX: 1.054,
      focalLengthY: 1.055, // Focal length values from your screenshot
      skew: 0,
      dispersion: 0.035, // Moderate chromatic aberration
    },
    '50mm Normal': {
      distortionX: 0,
      distortionY: 0, // No distortion (ideal normal lens)
      principalPointX: 0,
      principalPointY: 0,
      focalLengthX: 1,
      focalLengthY: 1,
      skew: 0,
      dispersion: 0.02, // Minimal chromatic aberration (normal focal length)
    },
    '85mm Portrait': {
      distortionX: 0.05,
      distortionY: 0.05, // Pincushion (telephoto) - similar magnitude to 35mm barrel
      principalPointX: 0,
      principalPointY: 0,
      focalLengthX: 0.91,
      focalLengthY: 0.91, // Slight compression (opposite of wide angle expansion)
      skew: 0,
      dispersion: 0.025, // Low chromatic aberration (longer focal length = less bending)
    },
    '135mm Tele': {
      distortionX: 0.07,
      distortionY: 0.07, // Barrel distortion from your screenshot
      principalPointX: 0,
      principalPointY: 0,
      focalLengthX: 0.882,
      focalLengthY: 0.883, // Focal length values from your screenshot
      skew: 0,
      dispersion: 0.03, // Telephoto with some chromatic aberration at edges
    },
  },
};
