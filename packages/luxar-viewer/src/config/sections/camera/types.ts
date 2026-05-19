/**
 * Camera configuration for 3D perspective and navigation
 */
export interface CameraConfig {
  // Note: fov, near, far live in RenderingSettings (renderingControls.defaults) as the single source of truth
  initialPosition: { x: number; y: number; z: number };
  fovMin: number;
  fovMax: number;
  fovSensitivity: number;
  fovPresets: Record<string, number>;
  lensDistortionPresets: Record<
    string,
    {
      distortionX: number;
      distortionY: number;
      principalPointX: number;
      principalPointY: number;
      focalLengthX: number;
      focalLengthY: number;
      skew: number;
      dispersion: number;
    }
  >;
}
