/**
 * Unit tests for camera-utils: LuxarCamera type guards and helpers.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  isPerspectiveCamera,
  isOrthographicCamera,
  getCameraFovRadians,
  updateCameraAspect,
  getOrthoFrustumHeight,
} from '../../../utils/camera-utils';

describe('camera-utils', () => {
  describe('isPerspectiveCamera', () => {
    it('should return true for PerspectiveCamera', () => {
      const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
      expect(isPerspectiveCamera(cam)).toBe(true);
    });

    it('should return false for OrthographicCamera', () => {
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
      expect(isPerspectiveCamera(cam)).toBe(false);
    });
  });

  describe('isOrthographicCamera', () => {
    it('should return true for OrthographicCamera', () => {
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
      expect(isOrthographicCamera(cam)).toBe(true);
    });

    it('should return false for PerspectiveCamera', () => {
      const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
      expect(isOrthographicCamera(cam)).toBe(false);
    });
  });

  describe('getCameraFovRadians', () => {
    // utils.md O2 / Phase E47: P9 rename — pins the deg→rad conversion
    // (THREE stores fov in degrees; getCameraFovRadians returns radians).
    it('PerspectiveCamera with 90° fov returns π/2 radians', () => {
      const cam = new THREE.PerspectiveCamera(90, 1, 0.1, 100);
      expect(getCameraFovRadians(cam)).toBeCloseTo(Math.PI / 2, 5);
    });

    // utils.md O2 / Phase E47: P9 rename — orthographic cameras have no
    // fov in the perspective sense; the helper returns 0 to surface that.
    it('OrthographicCamera returns 0 radians (no perspective fov)', () => {
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
      expect(getCameraFovRadians(cam)).toBe(0);
    });

    // utils.md O2 / Phase E47: P9 rename — sanity for the project's
    // default 47° fov used in many test cameras.
    it('PerspectiveCamera with default 47° fov returns 47·π/180 radians', () => {
      const cam = new THREE.PerspectiveCamera(47, 1, 0.1, 100);
      expect(getCameraFovRadians(cam)).toBeCloseTo((47 * Math.PI) / 180, 5);
    });
  });

  describe('updateCameraAspect', () => {
    // utils.md O2 / Phase E47: P9 rename — pins the basic perspective
    // aspect-ratio update (width/height ratio is stored on cam.aspect).
    it('PerspectiveCamera: aspect updates to width/height', () => {
      const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
      updateCameraAspect(cam, 1920, 1080);
      expect(cam.aspect).toBeCloseTo(1920 / 1080, 5);
    });

    it('should scale orthographic frustum width while preserving height', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      // Vertical extent = 10 (top - bottom = 5 - (-5))
      updateCameraAspect(cam, 1920, 1080);

      // Height should be preserved
      expect(cam.top - cam.bottom).toBeCloseTo(10, 5);
      // Width should be adjusted for 16:9 aspect
      const expectedHalfWidth = 5 * (1920 / 1080);
      expect(cam.left).toBeCloseTo(-expectedHalfWidth, 5);
      expect(cam.right).toBeCloseTo(expectedHalfWidth, 5);
    });

    it('should handle square viewport for orthographic camera', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      updateCameraAspect(cam, 800, 800);

      expect(cam.left).toBeCloseTo(-5, 5);
      expect(cam.right).toBeCloseTo(5, 5);
      expect(cam.top).toBeCloseTo(5, 5);
      expect(cam.bottom).toBeCloseTo(-5, 5);
    });

    it('should handle portrait viewport for orthographic camera', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      updateCameraAspect(cam, 600, 1200);

      expect(cam.top - cam.bottom).toBeCloseTo(10, 5);
      const expectedHalfWidth = 5 * (600 / 1200);
      expect(cam.left).toBeCloseTo(-expectedHalfWidth, 5);
      expect(cam.right).toBeCloseTo(expectedHalfWidth, 5);
    });

    it('should call updateProjectionMatrix for both camera types', () => {
      const persp = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
      const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);

      // Get initial projection matrices
      const perspInitial = persp.projectionMatrix.clone();
      const orthoInitial = ortho.projectionMatrix.clone();

      updateCameraAspect(persp, 1920, 1080);
      updateCameraAspect(ortho, 1920, 1080);

      // Projection matrices should have been updated
      expect(persp.projectionMatrix.equals(perspInitial)).toBe(false);
      expect(ortho.projectionMatrix.equals(orthoInitial)).toBe(false);
    });
  });

  describe('getOrthoFrustumHeight', () => {
    it('should return frustum height at default zoom (1.0)', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      // top - bottom = 5 - (-5) = 10, zoom = 1
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(10, 5);
    });

    it('should account for zoom level', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      cam.zoom = 2;
      // Effective height = (top - bottom) / zoom = 10 / 2 = 5
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(5, 5);
    });

    it('should handle asymmetric frustum', () => {
      const cam = new THREE.OrthographicCamera(-3, 7, 8, -2, 0.1, 100);
      // top - bottom = 8 - (-2) = 10, zoom = 1
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(10, 5);
    });

    it('should handle high zoom', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      cam.zoom = 100;
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(0.1, 5);
    });
  });
});
