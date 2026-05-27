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
    it('should convert perspective FOV from degrees to radians', () => {
      const cam = new THREE.PerspectiveCamera(90, 1, 0.1, 100);
      expect(getCameraFovRadians(cam)).toBeCloseTo(Math.PI / 2, 5);
    });

    it('should return 0 for orthographic camera', () => {
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
      expect(getCameraFovRadians(cam)).toBe(0);
    });

    it('should handle 47-degree FOV (default)', () => {
      const cam = new THREE.PerspectiveCamera(47, 1, 0.1, 100);
      expect(getCameraFovRadians(cam)).toBeCloseTo((47 * Math.PI) / 180, 5);
    });
  });

  describe('updateCameraAspect', () => {
    it('should update perspective camera aspect ratio', () => {
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
    // utils.md O2 / Phase E48: P9 renames — pin the height formula
    // (top - bottom) / zoom across {default zoom, 2x zoom, asymmetric
    // frustum, very high zoom}.
    it('zoom=1, symmetric frustum: height = top - bottom = 10', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(10, 5);
    });

    it('zoom=2 halves the effective frustum height (10 → 5)', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      cam.zoom = 2;
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(5, 5);
    });

    it('asymmetric frustum (top=8, bottom=-2): height = top - bottom = 10', () => {
      const cam = new THREE.OrthographicCamera(-3, 7, 8, -2, 0.1, 100);
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(10, 5);
    });

    it('zoom=100 shrinks height by 100x (10 → 0.1)', () => {
      const cam = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
      cam.zoom = 100;
      expect(getOrthoFrustumHeight(cam)).toBeCloseTo(0.1, 5);
    });
  });
});
