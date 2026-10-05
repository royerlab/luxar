/**
 * `applySceneKiosk` — resolve the scene/URL kiosk mode and apply it. The
 * ports are built only for a mode that is on: the viewer-config pass can run
 * on a partially constructed app, and "kiosk off" must not touch it.
 */

import { describe, expect, it, vi } from 'vitest';

import { applySceneKiosk } from '../../../../../core/app/kiosk/apply-kiosk';

describe('applySceneKiosk', () => {
  it('builds no ports and returns a no-op teardown when kiosk mode is off', () => {
    const ports = vi.fn();
    const teardown = applySceneKiosk(undefined, false, ports);
    expect(ports).not.toHaveBeenCalled();
    expect(() => teardown.disposeWatchdog()).not.toThrow();
    expect(() => teardown.restoreInput()).not.toThrow();
  });

  it('applies the URL flag and returns the watchdog teardown', () => {
    const setKeyboardEnabled = vi.fn();
    const canvas = {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    };
    const teardown = applySceneKiosk({ watchdog_reload: true }, true, () => ({
      setKeyboardEnabled,
      getKeyboardEnabled: () => true,
      setPointerEnabled: vi.fn(),
      getPointerEnabled: () => true,
      canvas,
      reload: vi.fn(),
    }));

    expect(setKeyboardEnabled).toHaveBeenCalledWith(false);
    expect(canvas.addEventListener).toHaveBeenCalledWith('webglcontextlost', expect.any(Function));
    teardown.disposeWatchdog();
    expect(canvas.removeEventListener).toHaveBeenCalledWith(
      'webglcontextlost',
      expect.any(Function)
    );
  });
});
