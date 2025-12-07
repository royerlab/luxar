/**
 * Mock for OrbitControls from three/examples/jsm/controls/OrbitControls
 */

import { vi } from 'vitest';

export class OrbitControls {
  object: any;
  domElement: any;
  enabled = true;
  target: any;
  minDistance = 0;
  maxDistance = Infinity;
  minZoom = 0;
  maxZoom = Infinity;
  minPolarAngle = 0;
  maxPolarAngle = Math.PI;
  minAzimuthAngle = -Infinity;
  maxAzimuthAngle = Infinity;
  enableDamping = false;
  dampingFactor = 0.05;
  enableZoom = true;
  zoomSpeed = 1.0;
  enableRotate = true;
  rotateSpeed = 1.0;
  enablePan = true;
  panSpeed = 1.0;
  screenSpacePanning = true;
  keyPanSpeed = 7.0;
  autoRotate = false;
  autoRotateSpeed = 2.0;
  enableKeys = true;
  keys = { LEFT: 37, UP: 38, RIGHT: 39, BOTTOM: 40 };
  mouseButtons = { LEFT: 0, MIDDLE: 1, RIGHT: 2 };
  touches = { ONE: 0, TWO: 2 };

  constructor(object: any, domElement: any) {
    this.object = object;
    this.domElement = domElement;
    this.target = { x: 0, y: 0, z: 0, set: vi.fn(), copy: vi.fn() };
  }

  update = vi.fn(() => false);
  dispose = vi.fn();
  getDistance = vi.fn(() => 1);
  getPolarAngle = vi.fn(() => Math.PI / 2);
  getAzimuthalAngle = vi.fn(() => 0);
  saveState = vi.fn();
  reset = vi.fn();
  listenToKeyEvents = vi.fn();
  stopListenToKeyEvents = vi.fn();

  // Event handling
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
  dispatchEvent = vi.fn();
}
