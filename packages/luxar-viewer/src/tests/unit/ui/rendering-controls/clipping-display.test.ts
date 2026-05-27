/**
 * Unit tests for ClippingDisplay — RAF-driven mirror of camera near/far
 * into the rendering-controls near/far sliders when dynamic clipping
 * is enabled.
 *
 * The class owns its own RAF id and a throttle timestamp, so the tests
 * drive it through `setDynamicEnabled` / `dispose` and inspect the
 * resulting style mutations on stub controllers.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ClippingDisplay } from '../../../../ui/rendering-controls/clipping-display';

interface StubController {
  domElement: HTMLElement;
  setValue: ReturnType<typeof vi.fn>;
  updateDisplay: ReturnType<typeof vi.fn>;
}

function makeController(): StubController {
  const wrapper = document.createElement('div');
  wrapper.className = 'luxar-gui__controller';
  const domElement = document.createElement('input');
  wrapper.appendChild(domElement);
  return {
    domElement,
    setValue: vi.fn(),
    updateDisplay: vi.fn(),
  };
}

let nearPlane: StubController;
let farPlane: StubController;
let camera: { near: number; far: number };
let settings: { near: number; far: number };

beforeEach(() => {
  nearPlane = makeController();
  farPlane = makeController();
  camera = { near: 0.1, far: 1000 };
  settings = { near: 0.1, far: 1000 };
});

function makeClipping() {
  return new ClippingDisplay({
    sceneManager: { camera } as never,
    settings: settings as never,
    getNearPlane: () => nearPlane as never,
    getFarPlane: () => farPlane as never,
  });
}

describe('ClippingDisplay — setDynamicEnabled(true)', () => {
  it('greys out the slider rows and disables pointer events', () => {
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    const nearWrap = nearPlane.domElement.parentElement as HTMLElement;
    const farWrap = farPlane.domElement.parentElement as HTMLElement;
    expect(nearWrap.style.opacity).toBe('0.5');
    expect(farWrap.style.opacity).toBe('0.5');
    expect(nearWrap.style.pointerEvents).toBe('none');
    expect(farWrap.style.pointerEvents).toBe('none');
    cd.dispose();
  });

  it('refreshes displays once immediately to reflect current camera state', () => {
    camera.near = 5;
    camera.far = 500;
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    expect(settings.near).toBe(5);
    expect(settings.far).toBe(500);
    expect(nearPlane.updateDisplay).toHaveBeenCalled();
    expect(farPlane.updateDisplay).toHaveBeenCalled();
    cd.dispose();
  });

  it('schedules the RAF loop', () => {
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame');
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    expect(rafSpy).toHaveBeenCalled();
    cd.dispose();
    rafSpy.mockRestore();
  });
});

describe('ClippingDisplay — setDynamicEnabled(false)', () => {
  it('restores opacity and pointer events', () => {
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    cd.setDynamicEnabled(false);
    const nearWrap = nearPlane.domElement.parentElement as HTMLElement;
    // jsdom canonicalises '1.0' → '1' when setting CSS opacity.
    expect(parseFloat(nearWrap.style.opacity)).toBe(1);
    expect(nearWrap.style.pointerEvents).toBe('auto');
    cd.dispose();
  });

  it('cancels the RAF loop', () => {
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    cd.setDynamicEnabled(false);
    expect(cancelSpy).toHaveBeenCalled();
    cd.dispose();
    cancelSpy.mockRestore();
  });
});

describe('ClippingDisplay — refreshDisplays', () => {
  it('only updates near when the value drift exceeds the 0.0001 threshold', () => {
    const cd = makeClipping();
    settings.near = 0.10001; // sub-threshold drift
    camera.near = 0.10005;
    cd.refreshDisplays();
    // settings.near unchanged because |0.10005 - 0.10001| = 4e-5 < 1e-4
    expect(settings.near).toBe(0.10001);
    expect(nearPlane.updateDisplay).not.toHaveBeenCalled();
    cd.dispose();
  });

  it('updates near + calls updateDisplay when drift is significant', () => {
    const cd = makeClipping();
    settings.near = 0.1;
    camera.near = 0.5;
    cd.refreshDisplays();
    expect(settings.near).toBe(0.5);
    expect(nearPlane.updateDisplay).toHaveBeenCalled();
    cd.dispose();
  });

  it('uses a 0.1 threshold for far plane (looser, since values are larger)', () => {
    const cd = makeClipping();
    settings.far = 1000;
    camera.far = 1000.05; // sub-threshold
    cd.refreshDisplays();
    expect(settings.far).toBe(1000);
    expect(farPlane.updateDisplay).not.toHaveBeenCalled();

    camera.far = 1000.5; // super-threshold
    cd.refreshDisplays();
    expect(settings.far).toBe(1000.5);
    expect(farPlane.updateDisplay).toHaveBeenCalled();
    cd.dispose();
  });

  it('returns early if the camera has no value', () => {
    const cd = new ClippingDisplay({
      sceneManager: { camera: null } as never,
      settings: settings as never,
      getNearPlane: () => nearPlane as never,
      getFarPlane: () => farPlane as never,
    });
    expect(() => cd.refreshDisplays()).not.toThrow();
    expect(nearPlane.updateDisplay).not.toHaveBeenCalled();
  });

  it('handles missing controllers gracefully (loader hasnt populated them yet)', () => {
    const cd = new ClippingDisplay({
      sceneManager: { camera } as never,
      settings: settings as never,
      getNearPlane: () => undefined,
      getFarPlane: () => undefined,
    });
    // Audit W22 fix: assert the observable contract — no updateDisplay
    // calls because the controllers were undefined, no settings mutation.
    const settingsBefore = { near: settings.near, far: settings.far };
    expect(() => cd.refreshDisplays()).not.toThrow();
    expect(settings.near).toBe(settingsBefore.near);
    expect(settings.far).toBe(settingsBefore.far);
    cd.dispose();
  });
});

describe('ClippingDisplay — dispose', () => {
  it('cancels any active RAF loop', () => {
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    cd.dispose();
    expect(cancelSpy).toHaveBeenCalled();
    cancelSpy.mockRestore();
  });

  it('idempotent: dispose twice is a no-op', () => {
    // Audit W22 fix: pin the observable contract — second dispose() must
    // NOT call cancelAnimationFrame a SECOND time (RAF id should be
    // cleared by the first dispose). A mutant that re-cancels a stale id
    // would surface as the spy.callCount being > 1.
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');
    const cd = makeClipping();
    cd.setDynamicEnabled(true);
    cd.dispose();
    const cancelsAfterFirst = cancelSpy.mock.calls.length;

    expect(() => cd.dispose()).not.toThrow();

    // Second dispose must not have triggered another cancelAnimationFrame.
    expect(cancelSpy.mock.calls.length).toBe(cancelsAfterFirst);
    cancelSpy.mockRestore();
  });
});
