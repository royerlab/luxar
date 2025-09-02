/**
 * Test Configuration Helper
 *
 * Provides centralized configuration values for tests, ensuring tests use
 * the same configuration as production code instead of hardcoded values.
 */

import { config } from '../config';
import * as THREE from 'three';

/**
 * Camera configuration for tests
 * Uses actual config values to ensure tests match production behavior
 */
export const testCameraConfig = {
  fov: config.camera.fov,
  near: config.camera.near,
  far: config.camera.far,
  aspectRatio: 1, // Default aspect ratio for tests
  position: config.camera.initialPosition,
};

/**
 * Helper to create a configured test camera
 */
export function createTestCamera(aspectRatio: number = 1): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(
    testCameraConfig.fov,
    aspectRatio,
    testCameraConfig.near,
    testCameraConfig.far
  );
  camera.position.set(
    testCameraConfig.position.x,
    testCameraConfig.position.y,
    testCameraConfig.position.z
  );
  return camera;
}

/**
 * Test bloom configuration - now from renderingControls.defaults
 */
export const testBloomConfig = {
  strength: config.renderingControls.defaults.bloomStrength,
  radius: config.renderingControls.defaults.bloomRadius,
  threshold: config.renderingControls.defaults.bloomThreshold,
  levels: config.renderingControls.defaults.bloomLevels,
};

/**
 * Test control configuration
 */
export const testControlsConfig = {
  fly: {
    movementSpeed: config.controls.fly.movement.speed.default,
    rotationSpeed: config.controls.fly.rotation.speed.default,
    inertialMode: config.controls.fly.inertialMode.default,
    damping: config.controls.fly.movement.damping.default,
    rotationDamping: config.controls.fly.rotation.damping.default,
  },
  orbit: {
    autoRotateSpeed: config.controls.orbit.autoRotate.speed.default,
    zoomSpeed: config.controls.orbit.zoom.speed.default,
    dampingFactor: config.controls.orbit.damping.factor.default,
  },
};

/**
 * Test shader configuration
 */
export const testShaderConfig = {
  points: config.shader.points,
};
