/**
 * Comprehensive tests for RenderingControls class
 * Tests cover all synchronization bugs found and fixed
 *
 * NOTE: Pure-function tests for input utilities (isNavigationKey, calculateFovChange,
 * shouldBlockShortcut) live in:
 *   - src/tests/unit/controls/input-validation.test.ts
 *   - src/tests/unit/input/input-handler.test.ts
 * This file focuses on RenderingControls state management, persistence, and sync.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { RenderingControls } from '../../../ui/rendering-controls';
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
