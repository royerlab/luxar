import { describe, expect, it } from 'vitest';
import { config } from '../../../../config';
import {
  syncCameraFovState,
  syncCurrentState,
} from '../../../../ui/rendering-controls/sync-current-state';

describe('syncCameraFovState', () => {
  it('updates only FOV state and derives the matching preset', () => {
    const settings = { ...config.renderingControls.defaults, autoRotate: true };

    syncCameraFovState(settings, { currentFov: 63 } as never);

    expect(settings.fov).toBe(63);
    expect(settings.fovPreset).toBe('35mm');
    expect(settings.autoRotate).toBe(true);
  });

  it('marks an unmatched live FOV as custom', () => {
    const settings = { ...config.renderingControls.defaults };

    syncCameraFovState(settings, { currentFov: 61 } as never);

    expect(settings.fov).toBe(61);
    expect(settings.fovPreset).toBe('Custom');
  });
});

describe('syncCurrentState — orbit turntable pull', () => {
  /**
   * Minimal context: the function's whole dependency surface is the FOV/camera
   * readouts, the ControlsManager getters, the dynamic-clipping state and a
   * `controllersRecursive()` sweep.
   */
  function makeContext(controlsOverride: Record<string, unknown>) {
    const settings = { ...config.renderingControls.defaults };
    const controls = {
      // `isOrbitControls` narrows structurally on these two fields.
      target: { x: 0, y: 0, z: 0 },
      autoRotate: true,
      autoRotateSpeed: 1.5,
      ...controlsOverride,
    };
    return {
      settings,
      context: {
        gui: { controllersRecursive: () => [] },
        settings,
        sceneManager: {
          currentFov: 47,
          camera: { near: 0.1, far: 1000 },
          controls: {
            getControlType: () => 'orbit',
            getControls: () => controls,
            getFlyConfig: () => ({
              inertialMode: false,
              movementSpeed: 1,
              rotationSpeed: 1,
              damping: 0.999,
              rotationDamping: 0.999,
            }),
            getNaturalDrag: () => false,
          },
          getDynamicClippingState: () => ({ enabled: false, near: 0.1, far: 1000 }),
        },
        controllers: {},
        updateClippingControlsState: () => {},
        updateCinematicModeCheckbox: () => {},
        updateNavigationControls: () => {},
      } as never,
    };
  }

  it('pulls the live turntable axis into settings', () => {
    // The panel writes settings → controls; this is the return leg. Without it
    // the popover would reopen showing 'vertical' after a scene authored
    // something else, and the stale value would then be persisted.
    const { settings, context } = makeContext({ autoRotateAxis: 'horizontal' });
    syncCurrentState(context);
    expect(settings.autoRotateAxis).toBe('horizontal');
    expect(settings.autoRotate).toBe(true);
    expect(settings.autoRotateSpeed).toBe(1.5);
  });
});
