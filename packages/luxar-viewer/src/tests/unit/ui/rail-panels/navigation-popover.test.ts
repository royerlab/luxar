// @vitest-environment jsdom
/**
 * Unit tests for ui/rail-panels/navigation-popover.ts.
 *
 * The navigation parameter controls moved out of the Rendering Controls panel
 * into this rail popover (left-click the Navigation rail button cycles modes;
 * right-click opens this popover with the current mode's params). These tests
 * stub the popover GUI + sceneManager + animationController and verify each
 * onChange forwards to the right SceneManager setter — the same behavioural
 * coverage the old navigation-setup tests provided, per mode.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AUTO_ROTATE_AXES } from '../../../../controls/types';
import { buildNavigationPopover } from '../../../../ui/rail-panels/navigation-popover';

interface ControllerStub {
  prop: string;
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  _onChangeFn: ((value: unknown) => void) | null;
  /**
   * Third `gui.add` argument when it is an options map (a dropdown row) —
   * `undefined` for a checkbox and a numeric `min` for a slider, so only the
   * dropdown case is recorded.
   */
  _options?: Record<string, unknown>;
}

interface GuiStub {
  add: ReturnType<typeof vi.fn>;
  destroy: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  controllers: ControllerStub[];
}

// A stub GUI captured by the mocked makePopoverGui. Controls are keyed by the
// bound property name so tests don't depend on insertion order.
let currentGui: GuiStub;

function makeController(prop: string, options?: Record<string, unknown>): ControllerStub {
  const ctrl: ControllerStub = {
    prop,
    _options: options,
    name: vi.fn(),
    onChange: vi.fn(),
    hide: vi.fn(),
    show: vi.fn(),
    _onChangeFn: null,
  };
  ctrl.name.mockReturnValue(ctrl);
  ctrl.onChange.mockImplementation((fn: (v: unknown) => void) => {
    ctrl._onChangeFn = fn;
    return ctrl;
  });
  return ctrl;
}

function makeGui(): GuiStub {
  const controllers: ControllerStub[] = [];
  return {
    add: vi.fn().mockImplementation((_obj: object, prop: string, arg3?: unknown) => {
      const isOptionsMap = typeof arg3 === 'object' && arg3 !== null;
      const c = makeController(prop, isOptionsMap ? (arg3 as Record<string, unknown>) : undefined);
      controllers.push(c);
      return c;
    }),
    destroy: vi.fn(),
    domElement: document.createElement('div'),
    controllers,
  };
}

// Mock the shared popover-GUI helper so build() gets our stub instead of a real
// GUI (which would need full DOM + theme wiring).
vi.mock('../../../../ui/rail-panels/popover-gui', () => ({
  makePopoverGui: vi.fn(() => currentGui),
}));

function byProp(prop: string): ControllerStub {
  const c = currentGui.controllers.find((x) => x.prop === prop);
  if (!c)
    throw new Error(
      `no controller for "${prop}" (have: ${currentGui.controllers.map((x) => x.prop).join(', ')})`
    );
  return c;
}

interface Stubs {
  host: HTMLElement;
  sceneManager: Record<string, ReturnType<typeof vi.fn>>;
  animationController: { startAnimation: ReturnType<typeof vi.fn> };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  setMode: ReturnType<typeof vi.fn>;
  settings: Record<string, unknown>;
}

function makeStubs(
  mode: 'orbit' | 'fly' | 'ortho',
  settingsOverride: Record<string, unknown> = {}
): Stubs {
  currentGui = makeGui();
  const sceneManager = {
    getControlType: vi.fn().mockReturnValue(mode),
    getSceneScale: vi.fn().mockReturnValue(0), // 0 → static config ranges
    setAutoRotate: vi.fn(),
    setAutoRotateSpeed: vi.fn(),
    setAutoRotateAxis: vi.fn(),
    setNaturalDrag: vi.fn(),
    setOrbitZoomSpeed: vi.fn(),
    setOrbitDampingFactor: vi.fn(),
    setFlyMovementSpeed: vi.fn(),
    setFlyRotationSpeed: vi.fn(),
    setFlyLookSpeed: vi.fn(),
    setFlyInertialMode: vi.fn(),
    setFlyDamping: vi.fn(),
    setFlyRotationDamping: vi.fn(),
  };
  const settings = {
    autoRotate: false,
    autoRotateSpeed: 0.25,
    autoRotateAxis: 'vertical',
    naturalDrag: false,
    orbitZoomSpeed: 1.0,
    orbitDampingFactor: 0.25,
    flyMovementSpeed: 1.0,
    flyRotationSpeed: 1.0,
    flyLookSpeed: 0.002,
    flyInertialMode: true,
    flyDamping: 0.999,
    flyRotationDamping: 0.999,
    ...settingsOverride,
  };
  return {
    host: document.createElement('div'),
    sceneManager,
    animationController: { startAnimation: vi.fn() },
    saveSettings: vi.fn(),
    triggerAnimation: vi.fn(),
    setMode: vi.fn(),
    settings,
  };
}

function build(stubs: Stubs): () => void {
  return buildNavigationPopover(stubs.host, {
    settings: stubs.settings as never,
    sceneManager: stubs.sceneManager as never,
    animationController: stubs.animationController as never,
    saveSettings: stubs.saveSettings as unknown as () => void,
    triggerAnimation: stubs.triggerAnimation as unknown as () => void,
    setMode: stubs.setMode as unknown as (t: 'orbit' | 'fly' | 'ortho') => void,
  });
}

describe('buildNavigationPopover', () => {
  describe('orbit mode', () => {
    let stubs: Stubs;
    beforeEach(() => {
      stubs = makeStubs('orbit');
      build(stubs);
    });

    it('wires autoRotate=true → setAutoRotate + startAnimation', () => {
      byProp('autoRotate')._onChangeFn?.(true);
      expect(stubs.sceneManager.setAutoRotate).toHaveBeenCalledWith(true);
      expect(stubs.animationController.startAnimation).toHaveBeenCalled();
      expect(stubs.saveSettings).toHaveBeenCalled();
    });

    it('wires autoRotate=false → setAutoRotate, no startAnimation', () => {
      byProp('autoRotate')._onChangeFn?.(false);
      expect(stubs.sceneManager.setAutoRotate).toHaveBeenCalledWith(false);
      expect(stubs.animationController.startAnimation).not.toHaveBeenCalled();
    });

    it('wires autoRotateSpeed → setAutoRotateSpeed', () => {
      byProp('autoRotateSpeed')._onChangeFn?.(2.5);
      expect(stubs.sceneManager.setAutoRotateSpeed).toHaveBeenCalledWith(2.5);
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });

    it('wires autoRotateAxis → setAutoRotateAxis', () => {
      byProp('autoRotateAxis')._onChangeFn?.('view');
      expect(stubs.sceneManager.setAutoRotateAxis).toHaveBeenCalledWith('view');
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });

    it('offers exactly the AUTO_ROTATE_AXES vocabulary as a dropdown', () => {
      // Asserted against the vocabulary rather than a literal, so BOTH drift
      // directions fail: a label whose token validation would reject, and a
      // token added to AUTO_ROTATE_AXES that the dropdown cannot reach.
      const options = byProp('autoRotateAxis')._options ?? {};
      expect(Object.values(options)).toEqual([...AUTO_ROTATE_AXES]);
      expect(Object.keys(options)).toEqual([
        'Vertical',
        'Horizontal',
        'View axis',
        'World X',
        'World Y',
        'World Z',
      ]);
    });

    it('wires naturalDrag → setNaturalDrag', () => {
      byProp('naturalDrag')._onChangeFn?.(true);
      expect(stubs.sceneManager.setNaturalDrag).toHaveBeenCalledWith(true);
    });

    it('wires orbitZoomSpeed → setOrbitZoomSpeed', () => {
      byProp('orbitZoomSpeed')._onChangeFn?.(2.0);
      expect(stubs.sceneManager.setOrbitZoomSpeed).toHaveBeenCalledWith(2.0);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });

    it('wires orbitDampingFactor → setOrbitDampingFactor', () => {
      byProp('orbitDampingFactor')._onChangeFn?.(0.1);
      expect(stubs.sceneManager.setOrbitDampingFactor).toHaveBeenCalledWith(0.1);
      expect(stubs.saveSettings).toHaveBeenCalled();
    });

    it('does NOT build fly controls in orbit mode', () => {
      expect(currentGui.controllers.some((c) => c.prop === 'flyMovementSpeed')).toBe(false);
    });
  });

  describe('fly mode', () => {
    let stubs: Stubs;
    beforeEach(() => {
      stubs = makeStubs('fly');
      build(stubs);
    });

    it('wires flyMovementSpeed → setFlyMovementSpeed', () => {
      byProp('flyMovementSpeed')._onChangeFn?.(2.0);
      expect(stubs.sceneManager.setFlyMovementSpeed).toHaveBeenCalledWith(2.0);
      expect(stubs.saveSettings).toHaveBeenCalled();
    });

    it('wires flyRotationSpeed → setFlyRotationSpeed', () => {
      byProp('flyRotationSpeed')._onChangeFn?.(1.5);
      expect(stubs.sceneManager.setFlyRotationSpeed).toHaveBeenCalledWith(1.5);
    });

    it('wires flyLookSpeed → setFlyLookSpeed', () => {
      byProp('flyLookSpeed')._onChangeFn?.(0.005);
      expect(stubs.sceneManager.setFlyLookSpeed).toHaveBeenCalledWith(0.005);
      expect(stubs.saveSettings).toHaveBeenCalled();
    });

    it('wires flyDamping → setFlyDamping', () => {
      byProp('flyDamping')._onChangeFn?.(0.97);
      expect(stubs.sceneManager.setFlyDamping).toHaveBeenCalledWith(0.97);
    });

    it('wires flyRotationDamping → setFlyRotationDamping', () => {
      byProp('flyRotationDamping')._onChangeFn?.(0.95);
      expect(stubs.sceneManager.setFlyRotationDamping).toHaveBeenCalledWith(0.95);
    });

    it('inertial=true shows the damping controls', () => {
      const damping = byProp('flyDamping');
      const rot = byProp('flyRotationDamping');
      damping.show.mockClear();
      rot.show.mockClear();
      byProp('flyInertialMode')._onChangeFn?.(true);
      expect(stubs.sceneManager.setFlyInertialMode).toHaveBeenCalledWith(true);
      expect(damping.show).toHaveBeenCalled();
      expect(rot.show).toHaveBeenCalled();
    });

    it('inertial=false hides the damping controls', () => {
      const damping = byProp('flyDamping');
      const rot = byProp('flyRotationDamping');
      damping.hide.mockClear();
      rot.hide.mockClear();
      byProp('flyInertialMode')._onChangeFn?.(false);
      expect(stubs.sceneManager.setFlyInertialMode).toHaveBeenCalledWith(false);
      expect(damping.hide).toHaveBeenCalled();
      expect(rot.hide).toHaveBeenCalled();
    });
  });

  describe('initial damping visibility', () => {
    it('hides damping when flyInertialMode=false at build', () => {
      const stubs = makeStubs('fly', { flyInertialMode: false });
      build(stubs);
      expect(byProp('flyDamping').hide).toHaveBeenCalled();
      expect(byProp('flyRotationDamping').hide).toHaveBeenCalled();
    });

    it('does NOT hide damping when flyInertialMode=true at build', () => {
      const stubs = makeStubs('fly', { flyInertialMode: true });
      build(stubs);
      expect(byProp('flyDamping').hide).not.toHaveBeenCalled();
      expect(byProp('flyRotationDamping').hide).not.toHaveBeenCalled();
    });
  });

  describe('ortho mode', () => {
    it('adds no controls and appends an explanatory note', () => {
      const stubs = makeStubs('ortho');
      build(stubs);
      expect(currentGui.controllers).toHaveLength(0);
      expect(stubs.host.querySelector('.luxar-control-rail__popover-note')).not.toBeNull();
    });
  });

  describe('mode selector', () => {
    it('renders three segments with the current mode highlighted', () => {
      const stubs = makeStubs('fly');
      build(stubs);
      const segs = Array.from(stubs.host.querySelectorAll('.luxar-control-rail__mode-seg'));
      expect(segs.map((s) => s.textContent)).toEqual(['Orbit', 'Fly', 'Ortho']);
      const active = segs.filter((s) => s.classList.contains('is-active'));
      expect(active).toHaveLength(1);
      expect(active[0].textContent).toBe('Fly');
    });

    it('clicking another mode calls setMode with that mode', () => {
      const stubs = makeStubs('orbit');
      build(stubs);
      const ortho = Array.from(stubs.host.querySelectorAll('.luxar-control-rail__mode-seg')).find(
        (s) => s.textContent === 'Ortho'
      ) as HTMLButtonElement;
      ortho.click();
      expect(stubs.setMode).toHaveBeenCalledWith('ortho');
    });

    it('clicking the already-active mode does NOT call setMode', () => {
      const stubs = makeStubs('orbit');
      build(stubs);
      const orbit = Array.from(stubs.host.querySelectorAll('.luxar-control-rail__mode-seg')).find(
        (s) => s.textContent === 'Orbit'
      ) as HTMLButtonElement;
      orbit.click();
      expect(stubs.setMode).not.toHaveBeenCalled();
    });
  });

  it('teardown disposes the GUI', () => {
    const stubs = makeStubs('orbit');
    const teardown = build(stubs);
    teardown();
    expect(currentGui.destroy).toHaveBeenCalled();
  });
});
