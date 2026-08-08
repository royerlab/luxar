/**
 * Comprehensive tests for RenderingControls class
 * Tests cover all synchronization bugs found and fixed
 *
 * This file focuses on RenderingControls state management, persistence, and sync.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { RenderingControls, decadeStep } from '../../../ui/rendering-controls';
import { config } from '../../../config';
import { log } from '../../../utils/log';
import { MAX_NEAR_FAR_RATIO } from '../../../scene/scene-manager/clipping/bounds-math';
import type { AnimationController } from '../../../scene/animation/animation-controller';

// Note: PostProcessingManager and SceneManager are not imported because we use
// duck-typed mocks (any type) to avoid loading the actual modules which have
// complex dependencies that would require extensive mocking.

// Mock custom GUI library
vi.mock('../../../ui/gui', () => {
  class MockController {
    domElement = {
      setAttribute: vi.fn(),
      style: {},
      closest: vi.fn(() => ({ style: {} })),
    };
    updateDisplay = vi.fn().mockReturnThis();
    setValue = vi.fn().mockReturnThis();
    onChange = vi.fn().mockReturnThis();
    onFinishChange = vi.fn().mockReturnThis();
    name = vi.fn().mockReturnThis();
    show = vi.fn().mockReturnThis();
    hide = vi.fn().mockReturnThis();
    // NumberController's fluent range setters. Present so the scale-aware
    // re-ranging in updateSceneScale is observable: the production code guards
    // on `typeof ctrl.min === 'function'`, so a mock lacking these silently
    // skips the whole path and any test of it would be vacuous.
    min = vi.fn().mockReturnThis();
    max = vi.fn().mockReturnThis();
    step = vi.fn().mockReturnThis();
    $input = { value: '' };
  }

  class MockFolder {
    controllers: MockController[] = [];
    domElement = document.createElement('div');

    add(_obj: any, _prop: string, ..._args: any[]): MockController {
      const controller = new MockController();
      this.controllers.push(controller);
      return controller;
    }

    addFolder(_name: string): MockFolder {
      return new MockFolder();
    }

    controllersRecursive(): MockController[] {
      return this.controllers;
    }

    open = vi.fn().mockReturnThis();
    close = vi.fn().mockReturnThis();
    show = vi.fn().mockReturnThis();
    hide = vi.fn().mockReturnThis();
  }

  class MockGUI extends MockFolder {
    constructor(_options?: any) {
      super();
    }

    destroy = vi.fn();
  }

  return { default: MockGUI, GUI: MockGUI, Folder: MockFolder };
});

describe('RenderingControls', () => {
  let renderingControls: RenderingControls;
  let mockPostProcessing: any;
  let mockSceneManager: any;
  let mockCamera: any;
  let mockControls: any;
  let mockAnimationController: AnimationController;

  beforeEach(() => {
    // Clear localStorage
    localStorage.clear();

    // Setup mock camera
    mockCamera = {
      fov: 47,
      near: 0.1,
      far: 1000,
      updateProjectionMatrix: vi.fn(),
    };

    // Setup mock controls
    mockControls = {
      autoRotate: false,
      autoRotateSpeed: 1.0,
      getControlType: vi.fn(() => 'orbit'),
      getControls: vi.fn(() => mockControls),
      // Reached via sceneManager.controls by updateSceneScale, whose scale-aware
      // slider re-ranging is that method's first test coverage.
      setSceneScale: vi.fn(),
      getFlyConfig: vi.fn(() => ({
        inertialMode: false,
        movementSpeed: 1.0,
        rotationSpeed: 1.0,
        damping: 0.999,
        rotationDamping: 0.999,
      })),
      setControlType: vi.fn(),
      setAutoRotate: vi.fn(),
      setAutoRotateSpeed: vi.fn(),
      setNaturalDrag: vi.fn(),
      getNaturalDrag: vi.fn(() => false),
      setOrbitZoomSpeed: vi.fn(),
      setOrbitDampingFactor: vi.fn(),
      setFlyLookSpeed: vi.fn(),
      setFlyMovementSpeed: vi.fn(),
      setFlyRotationSpeed: vi.fn(),
      setFlyInertialMode: vi.fn(),
      setFlyDamping: vi.fn(),
      setFlyRotationDamping: vi.fn(),
    };

    // Setup mock scene manager
    mockSceneManager = {
      camera: mockCamera,
      get currentFov() {
        return mockCamera.fov;
      },
      controls: mockControls,
      renderer: { domElement: document.createElement('canvas') },
      updateFOV: vi.fn((delta) => {
        mockCamera.fov += delta * 0.05;
      }),
      updateClippingPlanes: vi.fn((near, far) => {
        mockCamera.near = near;
        mockCamera.far = far;
      }),
      setDynamicClipping: vi.fn(),
      getDynamicClippingState: vi.fn(() => ({
        enabled: false,
        near: 0.1,
        far: 1000,
      })),
      updateExposure: vi.fn(),
      updateGlobalOffset: vi.fn(),
      updateGlobalGamma: vi.fn(),
      setControlType: vi.fn(),
      setAutoRotate: vi.fn(),
      setAutoRotateSpeed: vi.fn(),
      setNaturalDrag: vi.fn(),
      setOrbitZoomSpeed: vi.fn(),
      setOrbitDampingFactor: vi.fn(),
      setFlyLookSpeed: vi.fn(),
      setFlyMovementSpeed: vi.fn(),
      setFlyRotationSpeed: vi.fn(),
      setFlyInertialMode: vi.fn(),
      setFlyDamping: vi.fn(),
      setFlyRotationDamping: vi.fn(),
      getSceneScale: vi.fn(() => 0),
    };

    // Setup mock post-processing
    mockPostProcessing = {
      setBloomEnabled: vi.fn(),
      updateBloomSettings: vi.fn(),
      setBloomLevels: vi.fn(),
      setSSAAEnabled: vi.fn(),
      setSSAAMultiplier: vi.fn(),
      setFXAAEnabled: vi.fn(),
      setMSAAEnabled: vi.fn(),
      setMSAASamples: vi.fn(),
      setToneMapping: vi.fn(),
      setDetectorNoiseEnabled: vi.fn(),
      setVignetteEnabled: vi.fn(),
      setChromaticLensDistortionEnabled: vi.fn(),
      updateChromaticLensDistortion: vi.fn(),
      startDeferRebuild: vi.fn(),
      endDeferRebuild: vi.fn(),
    };

    // Setup mock animation controller
    mockAnimationController = {
      startAnimation: vi.fn(),
      stopAnimation: vi.fn(),
      isAnimating: vi.fn(() => false),
    } as any;

    // Create rendering controls instance
    renderingControls = new RenderingControls(mockPostProcessing, mockSceneManager);
    renderingControls.setAnimationController(mockAnimationController);
  });

  afterEach(() => {
    // Clean up rendering controls to prevent timer leaks
    if (renderingControls) {
      renderingControls.dispose();
    }
  });

  describe('lifecycle cleanup', () => {
    it('should cancel deferred click-outside setup on dispose', () => {
      vi.useFakeTimers();
      const addSpy = vi.spyOn(document, 'addEventListener');

      renderingControls.show();
      renderingControls.dispose();
      vi.advanceTimersByTime(150);

      expect(addSpy).not.toHaveBeenCalledWith('mousedown', expect.any(Function), true);

      addSpy.mockRestore();
      vi.useRealTimers();
    });

    it('should remove active click-outside handler on dispose', () => {
      vi.useFakeTimers();
      const addSpy = vi.spyOn(document, 'addEventListener');
      const removeSpy = vi.spyOn(document, 'removeEventListener');

      // show() defers the handler install by 100ms via setTimeout, so let
      // it fire so the handler is actually attached.
      renderingControls.show();
      vi.advanceTimersByTime(150);

      // Recover the actual handler reference the FocusManager installed.
      const installCall = addSpy.mock.calls.find((c) => c[0] === 'mousedown');
      expect(installCall).toBeDefined();
      const handler = installCall![1];

      renderingControls.dispose();

      expect(removeSpy).toHaveBeenCalledWith('mousedown', handler, true);

      addSpy.mockRestore();
      removeSpy.mockRestore();
      vi.useRealTimers();
    });
  });

  describe('Bug Fix #1: Exposure Value Sync', () => {
    it('should sync exposure value when resetting to defaults', () => {
      // Setup: Change exposure to non-default value
      const controls = renderingControls as any;
      controls.settings.exposure = 3.0;

      // Action: Reset to defaults
      controls.resetToDefaults();

      // Verify: Exposure should be synced to default (0.0)
      expect(controls.settings.exposure).toBe(0.0);
    });

    it('should sync exposure value when loading settings', () => {
      // Setup: Save settings with custom exposure value
      const controls = renderingControls as any;
      controls.settings.exposure = 2.5;
      controls.sceneId = 'test-scene';
      controls.saveSettings();

      // Reset to different value
      controls.settings.exposure = 0;

      // Action: Load settings
      controls.loadSettings();

      // Verify: Exposure should match loaded value
      expect(controls.settings.exposure).toBe(2.5);
    });
  });

  describe('Initialization Sync', () => {
    it('should apply camera settings after loading from localStorage', () => {
      const controls = renderingControls as any;

      // Setup: Save custom camera settings
      controls.sceneId = 'test-scene';
      controls.settings.fov = 75;
      controls.settings.near = 0.5;
      controls.settings.far = 2000;
      controls.saveSettings();

      // Reset camera to defaults
      mockCamera.fov = 47;
      mockCamera.near = 0.1;
      mockCamera.far = 1000;
      mockSceneManager.updateFOV.mockClear();
      mockSceneManager.updateClippingPlanes.mockClear();

      // Action: Load scene (simulates initialization)
      controls.setSceneId('test-scene');

      // Verify: Camera settings were applied
      expect(mockSceneManager.updateFOV).toHaveBeenCalled();
      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.5, 2000);
    });

    it('should apply navigation settings after loading from localStorage', () => {
      const controls = renderingControls as any;

      // Setup: Save custom navigation settings
      controls.sceneId = 'test-scene';
      controls.settings.controlType = 'fly';
      controls.settings.autoRotate = true;
      controls.settings.flyMovementSpeed = 5.0;
      controls.settings.flyRotationSpeed = 2.0;
      controls.saveSettings();

      // Clear mocks
      mockSceneManager.setControlType.mockClear();
      mockSceneManager.setAutoRotate.mockClear();
      mockSceneManager.setFlyMovementSpeed.mockClear();
      mockSceneManager.setFlyRotationSpeed.mockClear();

      // Action: Load scene
      controls.setSceneId('test-scene');

      // Verify: Navigation settings were applied
      expect(mockSceneManager.setControlType).toHaveBeenCalledWith('fly');
      expect(mockSceneManager.setAutoRotate).toHaveBeenCalledWith(true);
      expect(mockSceneManager.setFlyMovementSpeed).toHaveBeenCalledWith(5.0);
      expect(mockSceneManager.setFlyRotationSpeed).toHaveBeenCalledWith(2.0);
    });
  });

  describe('Bug Fix #5-6: Reset to Defaults Completeness', () => {
    it('should apply camera settings when resetting to defaults', () => {
      const controls = renderingControls as any;

      // Setup: Change camera settings
      controls.settings.fov = 120;
      controls.settings.near = 5.0;
      controls.settings.far = 5000;
      mockCamera.fov = 120;
      mockSceneManager.updateFOV.mockClear();
      mockSceneManager.updateClippingPlanes.mockClear();

      // Action: Reset to defaults
      controls.resetToDefaults();

      // Verify: Camera settings applied
      expect(mockSceneManager.updateFOV).toHaveBeenCalled();
      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.1, 1000);
    });

    it('should apply navigation settings when resetting to defaults', () => {
      const controls = renderingControls as any;

      // Setup: Change navigation settings
      controls.settings.controlType = 'fly';
      controls.settings.autoRotate = true;
      mockSceneManager.setControlType.mockClear();
      mockSceneManager.setAutoRotate.mockClear();

      // Action: Reset to defaults
      controls.resetToDefaults();

      // Verify: Navigation settings applied
      expect(mockSceneManager.setControlType).toHaveBeenCalledWith('orbit');
      expect(mockSceneManager.setAutoRotate).toHaveBeenCalledWith(false);
      expect(mockSceneManager.setFlyMovementSpeed).toHaveBeenCalled();
      expect(mockSceneManager.setFlyRotationSpeed).toHaveBeenCalled();
      expect(mockSceneManager.setFlyInertialMode).toHaveBeenCalled();
      expect(mockSceneManager.setFlyDamping).toHaveBeenCalled();
      expect(mockSceneManager.setFlyRotationDamping).toHaveBeenCalled();
    });
  });

  describe('Bug Fix #7-8: flyRotationSpeed Sync', () => {
    it('should sync flyRotationSpeed in syncCurrentState', () => {
      const controls = renderingControls as any;

      // Setup: Controls manager has different flyRotationSpeed
      mockControls.getFlyConfig.mockReturnValue({
        inertialMode: true,
        movementSpeed: 2.0,
        rotationSpeed: 3.5,
        damping: 0.99,
        rotationDamping: 0.98,
      });

      // Action: Sync current state
      controls.syncCurrentState();

      // Verify: flyRotationSpeed is synced from controls
      expect(controls.settings.flyRotationSpeed).toBe(3.5);
    });

    it('should update flyRotationSpeed controller in syncCurrentState', () => {
      const controls = renderingControls as any;

      // Setup: Mock controller exists
      const mockController = {
        setValue: vi.fn().mockReturnThis(),
        updateDisplay: vi.fn().mockReturnThis(),
      };
      controls.controllers.flyRotationSpeed = mockController;

      mockControls.getFlyConfig.mockReturnValue({
        inertialMode: false,
        movementSpeed: 1.0,
        rotationSpeed: 2.5,
        damping: 0.999,
        rotationDamping: 0.999,
      });

      // Action: Sync current state
      controls.syncCurrentState();

      // Verify: Controller was updated
      expect(mockController.setValue).toHaveBeenCalledWith(2.5);
      expect(mockController.updateDisplay).toHaveBeenCalled();
    });
  });

  describe('Settings Persistence', () => {
    it('should save and load all settings correctly', () => {
      const controls = renderingControls as any;
      controls.sceneId = 'test-persistence';

      // Setup: Modify various settings
      controls.settings.fov = 85;
      controls.settings.bloomStrength = 1.5;
      controls.settings.vignetteEnabled = true;
      controls.settings.controlType = 'fly';
      controls.settings.flyMovementSpeed = 3.0;
      controls.settings.flyRotationSpeed = 2.0;

      // Action: Save and reload
      controls.saveSettings();

      // Reset to different values
      controls.settings.fov = 47;
      controls.settings.bloomStrength = 0.5;
      controls.settings.vignetteEnabled = false;

      // Load back
      controls.loadSettings();

      // Verify: All settings restored
      expect(controls.settings.fov).toBe(85);
      expect(controls.settings.bloomStrength).toBe(1.5);
      expect(controls.settings.vignetteEnabled).toBe(true);
      expect(controls.settings.controlType).toBe('fly');
      expect(controls.settings.flyMovementSpeed).toBe(3.0);
      expect(controls.settings.flyRotationSpeed).toBe(2.0);
    });

    it('should clear localStorage when resetting to defaults', () => {
      const controls = renderingControls as any;
      controls.sceneId = 'test-clear';

      // Setup: Save custom settings
      controls.settings.fov = 100;
      controls.saveSettings();
      expect(localStorage.getItem('luxar.rendering.test-clear')).not.toBeNull();

      // Action: Reset to defaults
      controls.resetToDefaults();

      // Verify: localStorage cleared
      expect(localStorage.getItem('luxar.rendering.test-clear')).toBeNull();
    });
  });

  describe('Integration: Complete Reset Flow', () => {
    it('should fully reset all 49 settings and apply them', () => {
      const controls = renderingControls as any;

      // Setup: Change many settings to non-default values
      controls.settings.fov = 120;
      controls.settings.bloomStrength = 2.0;
      controls.settings.vignetteEnabled = true;
      controls.settings.controlType = 'fly';
      controls.settings.flyInertialMode = true;
      controls.settings.exposure = 3.0;

      // Clear all mock calls
      Object.values(mockSceneManager).forEach((fn: any) => {
        if (typeof fn === 'function' && fn.mockClear) fn.mockClear();
      });
      Object.values(mockPostProcessing).forEach((fn: any) => {
        if (typeof fn === 'function' && fn.mockClear) fn.mockClear();
      });

      // Action: Reset to defaults
      controls.resetToDefaults();

      // Verify: Settings reset to defaults
      expect(controls.settings.fov).toBe(47);
      expect(controls.settings.bloomStrength).toBe(0.25);
      expect(controls.settings.vignetteEnabled).toBe(false);
      expect(controls.settings.controlType).toBe('orbit');

      // Verify: Exposure synced to default (0.0)
      expect(controls.settings.exposure).toBe(0.0);

      // Verify: Camera settings applied
      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.1, 1000);

      // Verify: Navigation settings applied
      expect(mockSceneManager.setControlType).toHaveBeenCalledWith('orbit');
      expect(mockSceneManager.setAutoRotate).toHaveBeenCalled();
      expect(mockSceneManager.setFlyMovementSpeed).toHaveBeenCalled();
      expect(mockSceneManager.setFlyRotationSpeed).toHaveBeenCalled();

      // Verify: Post-processing settings applied
      expect(mockPostProcessing.setBloomEnabled).toHaveBeenCalled();
    });
  });

  describe('Integration: Complete Initialization Flow', () => {
    it('should properly sync everything on first scene load', () => {
      const controls = renderingControls as any;

      // Setup: Use consistent sceneId (no hyphens to avoid ID transformation issues)
      const testUrl = 'initTest';
      controls.sceneId = 'initTest'; // Must match what setSceneId() generates
      controls.settings.fov = 75;
      // A MANUAL near only persists with dynamic clipping off — with it on,
      // settings.near is a live camera readout that saveSettingsToStorage
      // deliberately omits (see stripDynamicClippingPlanes). This test is
      // about round-tripping user intent, so it takes the manual branch.
      controls.settings.dynamicClippingEnabled = false;
      controls.settings.near = 0.5;
      controls.settings.flyMovementSpeed = 4.0;
      controls.settings.flyRotationSpeed = 3.0;
      controls.settings.bloomStrength = 1.8;
      controls.saveSettings();

      // Reset to defaults in memory (simulating fresh instance)
      controls.settings.fov = 47;
      controls.settings.near = 0.1;
      controls.settings.flyMovementSpeed = 1.0;
      controls.settings.flyRotationSpeed = 1.0;
      controls.settings.bloomStrength = 0.25; // Actual default
      controls.settings.exposure = 0.0; // Actual default
      mockCamera.fov = 47;
      mockCamera.near = 0.1;

      // Clear mocks
      mockSceneManager.updateFOV.mockClear();
      mockSceneManager.updateClippingPlanes.mockClear();
      mockSceneManager.setFlyMovementSpeed.mockClear();
      mockSceneManager.setFlyRotationSpeed.mockClear();

      // Action: Load scene (simulates initialization)
      controls.setSceneId(testUrl);

      // Verify: Settings loaded from localStorage
      expect(controls.settings.fov).toBe(75);
      expect(controls.settings.near).toBe(0.5);
      expect(controls.settings.flyMovementSpeed).toBe(4.0);
      expect(controls.settings.flyRotationSpeed).toBe(3.0);
      expect(controls.settings.bloomStrength).toBe(1.8);

      // Verify: Camera settings applied to managers
      expect(mockSceneManager.updateFOV).toHaveBeenCalled();
      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.5, 1000);

      // Verify: Navigation settings applied
      expect(mockSceneManager.setFlyMovementSpeed).toHaveBeenCalledWith(4.0);
      expect(mockSceneManager.setFlyRotationSpeed).toHaveBeenCalledWith(3.0);

      // Verify: Post-processing settings applied (setBloomEnabled is called with enabled + params)
      expect(mockPostProcessing.setBloomEnabled).toHaveBeenCalledWith(
        expect.any(Boolean),
        1.8,
        expect.any(Number),
        expect.any(Number)
      );
    });

    // The user-visible half of the dynamic-clipping persistence bug, pinned at
    // the facade rather than at the storage helper: a scene visited while
    // zoomed in used to come BACK with the camera-derived near/far reapplied as
    // fixed manual planes (e.g. near 1.05e-4 / far 61 with the Dynamic
    // Clipping box unchecked), untraceable to anything the user did. The
    // storage-level test proves the keys are omitted; this proves the whole
    // save → load → apply chain lands on the defaults instead.
    it('does not resurrect a zoomed-in camera near/far after a dynamic-clipping session', () => {
      const controls = renderingControls as any;
      controls.sceneId = 'dynZoom';

      // A dynamic-clipping session: the display RAF loop has stamped the live
      // camera pose into settings, then some unrelated control saved.
      controls.settings.dynamicClippingEnabled = true;
      controls.settings.near = 1.05e-4;
      controls.settings.far = 61;
      controls.settings.bloomStrength = 1.23; // the "unrelated control"
      controls.saveSettings();

      // Fresh instance state.
      controls.settings.near = 0.1;
      controls.settings.far = 1000;
      mockSceneManager.updateClippingPlanes.mockClear();

      controls.setSceneId('dynZoom');

      // The unrelated setting round-trips; the transient planes do not.
      expect(controls.settings.bloomStrength).toBe(1.23);
      expect(controls.settings.near).toBe(config.renderingControls.defaults.near);
      expect(controls.settings.far).toBe(config.renderingControls.defaults.far);
      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(
        config.renderingControls.defaults.near,
        config.renderingControls.defaults.far
      );
    });
  });

  describe('Zarr-authored clipping planes', () => {
    // Authoring works (applyZarrDefaults applies the planes AFTER
    // autoAdjustClippingPlanes and pushes dynamicClippingEnabled through
    // applySettings), but authored planes and dynamic clipping conflict: the
    // per-frame update recomputes near/far next frame. Precedence is
    // unchanged — dynamic clipping is an explicit auto mode — so the author
    // gets told instead of silently ignored.
    it('warns when authored planes coexist with dynamic clipping enabled', () => {
      const controls = renderingControls as any;
      const warnSpy = vi.spyOn(log, 'warning');
      controls.setZarrViewerConfig({
        camera: { near: 0.5, far: 400 },
        // Rendering keys live at the TOP level of viewer_config (see
        // RENDERING_SETTINGS_MAP / ViewerConfig.to_dict), not under `rendering`.
        dynamic_clipping_enabled: true,
      });

      controls.applyZarrDefaults();

      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.5, 400);
      expect(
        warnSpy.mock.calls.some((c) => String(c[1]).includes('dynamic clipping is enabled'))
      ).toBe(true);
      warnSpy.mockRestore();
    });

    it('does not warn when the author also disables dynamic clipping', () => {
      const controls = renderingControls as any;
      const warnSpy = vi.spyOn(log, 'warning');
      controls.setZarrViewerConfig({
        camera: { near: 0.5, far: 400 },
        dynamic_clipping_enabled: false,
      });

      controls.applyZarrDefaults();

      expect(mockSceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.5, 400);
      expect(mockSceneManager.setDynamicClipping).toHaveBeenCalledWith(false);
      expect(
        warnSpy.mock.calls.some((c) => String(c[1]).includes('dynamic clipping is enabled'))
      ).toBe(false);
      warnSpy.mockRestore();
    });
  });

  describe('Scale-aware clipping slider ranges', () => {
    // The sliders' authored range is ABSOLUTE (near 0.0001-10) while every
    // value they display is scene-relative. Under dynamic clipping they are
    // live read-only readouts, so on a scene whose framed `near` exceeds 10 the
    // number input reads the truth while `<input type=range>` clamps and pins
    // the thumb at the wrong end. Re-ranged off the scene diagonal, alongside
    // the fly-speed slider that already works this way.
    it('re-ranges near/far from the scene diagonal on updateSceneScale', () => {
      const controls = renderingControls as any;
      mockSceneManager.getSceneScale = vi.fn(() => 95.3); // the mesh demo's diagonal

      controls.updateSceneScale();

      const nearCtrl = controls.controllers.nearPlane;
      const farCtrl = controls.controllers.farPlane;
      expect(nearCtrl.min).toHaveBeenCalled();
      expect(farCtrl.min).toHaveBeenCalled();

      const nearMin = nearCtrl.min.mock.calls.at(-1)[0];
      const nearMax = nearCtrl.max.mock.calls.at(-1)[0];
      const farMax = farCtrl.max.mock.calls.at(-1)[0];

      // The range must CONTAIN every near/far the policy can produce over the
      // whole legal orbit range — asserted by evaluating the policy's own
      // equations rather than against a hand-picked threshold. An earlier
      // version used `nearMax = scale` and a sample value of 111.98, which is
      // itself outside that range: the assertion encoded the bug.
      const scale = 95.3;
      const R = 0.5 * scale * 1.05;
      const distMax = scale * config.controls.scaleMultipliers.maxDistanceFactor;
      expect(nearMin).toBeCloseTo(R * 2e-6, 12);

      for (const dist of [
        R * 0.5, // inside the sphere (floor binds)
        1.7 * scale, // framed — where `nearMax = scale` used to pin
        scale + R, // the crossover that exposed it
        distMax, // zoom-out limit
      ]) {
        const far = dist + R;
        const near = Math.max(Math.max(1e-9, R * 2e-6), far / MAX_NEAR_FAR_RATIO, dist - R);
        expect(near).toBeGreaterThanOrEqual(nearMin);
        expect(near).toBeLessThanOrEqual(nearMax);
        expect(far).toBeLessThanOrEqual(farMax);
      }
      // far must include the expanded radius, not stop at dist alone.
      expect(farMax).toBeCloseTo(distMax + R, 6);
    });

    it('scales the range down for a micron-scale scene', () => {
      const controls = renderingControls as any;
      mockSceneManager.getSceneScale = vi.fn(() => 0.01);

      controls.updateSceneScale();

      const nearMin = controls.controllers.nearPlane.min.mock.calls.at(-1)[0];
      const nearMax = controls.controllers.nearPlane.max.mock.calls.at(-1)[0];
      // The point of this case is the BOTTOM end: on a micron scene the whole
      // useful range sat below the old absolute 0.0001 minimum.
      expect(nearMin).toBeLessThan(0.0001);
      expect(nearMin).toBeCloseTo(0.5 * 0.01 * 1.05 * 2e-6, 15);
      // The top end still spans the legal orbit range, so it must NOT collapse
      // to the diagonal — that spelling made small scenes worse than the old
      // absolute max of 10.
      expect(nearMax).toBeGreaterThan(10);
    });

    // The step is what the GUI reads the DISPLAYED DECIMAL COUNT off of, via
    // `String(step)`. My first version passed the range minimum straight
    // through, which is scene-derived and therefore carries float noise:
    // `String(1.05e-4)` is "0.00010499999999999999", so a near of 117.5
    // rendered as "117.50000000000000000000". Asserting min/max alone never saw
    // it — these assert the rendered string, which is what a user sees.
    const formatLikeGui = (value: number, step: number): string => {
      const s = String(step);
      const dot = s.indexOf('.');
      return value.toFixed(dot === -1 ? 0 : s.length - dot - 1);
    };

    it.each([
      [0.01, 0.01175, 0.02225],
      [1, 1.175, 2.225],
      [95.3, 111.9775, 212.0425],
      [100, 117.5, 222.5],
      [100000, 117500, 222500],
    ])('keeps the rendered readout clean at scale %p', (scale, sampleNear, sampleFar) => {
      const controls = renderingControls as any;
      mockSceneManager.getSceneScale = vi.fn(() => scale);
      controls.updateSceneScale();

      const nearStep = controls.controllers.nearPlane.step.mock.calls.at(-1)[0];
      const farStep = controls.controllers.farPlane.step.mock.calls.at(-1)[0];

      // Exact short decimal: no float noise, no exponential notation.
      for (const step of [nearStep, farStep]) {
        expect(String(step)).toMatch(/^(?:0\.0*1|1(?:0*)?)$/);
        expect(String(step)).not.toContain('e');
      }
      // ...so the readout carries a sane number of decimals, not 20 and not 0.
      for (const [value, step] of [
        [sampleNear, nearStep],
        [sampleFar, farStep],
      ] as const) {
        const rendered = formatLikeGui(value, step);
        const decimals = (rendered.split('.')[1] ?? '').length;
        expect(decimals).toBeLessThanOrEqual(6);
        expect(Number(rendered)).toBeCloseTo(value, 2);
      }
    });

    it('decadeStep floors at 1e-6, where String() would go exponential', () => {
      expect(String(decadeStep(1e-8))).toBe('0.000001');
      expect(String(decadeStep(1.05e-4))).toBe('0.0001');
      expect(String(decadeStep(0.0953))).toBe('0.01');
      expect(String(decadeStep(222500))).toBe('100000');
      // Degenerate input must not produce NaN/Infinity as a slider step.
      for (const bad of [0, -1, NaN, Infinity]) {
        expect(decadeStep(bad)).toBe(1e-6);
      }
    });

    it('is a no-op when the scene scale is unknown', () => {
      const controls = renderingControls as any;
      mockSceneManager.getSceneScale = vi.fn(() => 0);
      controls.controllers.nearPlane.min.mockClear();

      controls.updateSceneScale();

      expect(controls.controllers.nearPlane.min).not.toHaveBeenCalled();
    });
  });

  describe('SyncCurrentState Completeness', () => {
    it('should sync all external-changeable settings from managers', () => {
      const controls = renderingControls as any;

      // Setup: Managers have different values
      mockCamera.fov = 85;
      mockCamera.near = 0.2;
      mockCamera.far = 2000;
      mockControls.getControlType.mockReturnValue('fly');
      mockControls.autoRotate = true;
      mockControls.autoRotateSpeed = 2.5;
      mockControls.getFlyConfig.mockReturnValue({
        inertialMode: true,
        movementSpeed: 3.0,
        rotationSpeed: 4.0,
        damping: 0.95,
        rotationDamping: 0.96,
      });
      mockSceneManager.getDynamicClippingState.mockReturnValue({
        enabled: true,
        near: 0.2,
        far: 2000,
      });

      // Action: Sync from managers
      controls.syncCurrentState();

      // Verify: All settings synced
      expect(controls.settings.fov).toBe(85);
      expect(controls.settings.near).toBe(0.2);
      expect(controls.settings.far).toBe(2000);
      expect(controls.settings.controlType).toBe('fly');
      expect(controls.settings.flyMovementSpeed).toBe(3.0);
      expect(controls.settings.flyRotationSpeed).toBe(4.0); // Bug #7 fix
      expect(controls.settings.flyInertialMode).toBe(true);
      expect(controls.settings.flyDamping).toBe(0.95);
      expect(controls.settings.flyRotationDamping).toBe(0.96);
      expect(controls.settings.dynamicClippingEnabled).toBe(true);
    });
  });
});
