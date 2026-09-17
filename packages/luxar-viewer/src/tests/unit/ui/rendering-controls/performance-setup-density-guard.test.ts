// @vitest-environment jsdom
/**
 * The Density Guard toggle + Thinning readout in the Performance folder
 * (`ui/rendering-controls/setup/performance-setup.ts`), driven through a
 * stub `DensityGuardControl`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  formatKeepFraction,
  formatThinning,
  setupPerformanceControls,
  type PerformanceSetupContext,
} from '../../../../ui/rendering-controls/setup/performance-setup';
import type { DensityGuardControl } from '../../../../ui/rendering-controls/types';

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  onFinishChange: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  updateDisplay: ReturnType<typeof vi.fn>;
  min: ReturnType<typeof vi.fn>;
  max: ReturnType<typeof vi.fn>;
  step: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  _name: string | null;
  _target: Record<string, unknown> | null;
  _onChangeFn: ((value: unknown) => void) | null;
}

function makeController(target: Record<string, unknown>): ControllerStub {
  const ctrl: ControllerStub = {
    name: vi.fn(),
    onChange: vi.fn(),
    onFinishChange: vi.fn(),
    hide: vi.fn(),
    show: vi.fn(),
    updateDisplay: vi.fn(),
    min: vi.fn(),
    max: vi.fn(),
    step: vi.fn(),
    domElement: document.createElement('div'),
    _name: null,
    _target: target,
    _onChangeFn: null,
  };
  ctrl.name.mockImplementation((label: string) => {
    ctrl._name = label;
    return ctrl;
  });
  ctrl.min.mockReturnValue(ctrl);
  ctrl.max.mockReturnValue(ctrl);
  ctrl.step.mockReturnValue(ctrl);
  ctrl.onChange.mockImplementation((fn: (v: unknown) => void) => {
    ctrl._onChangeFn = fn;
    return ctrl;
  });
  ctrl.onFinishChange.mockReturnValue(ctrl);
  return ctrl;
}

interface FolderStub {
  add: ReturnType<typeof vi.fn>;
  open: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  controllers: ControllerStub[];
}

function makeFolder(): FolderStub {
  const controllers: ControllerStub[] = [];
  const root = document.createElement('div');
  const children = document.createElement('div');
  children.className = 'luxar-gui__children';
  root.appendChild(children);
  return {
    add: vi.fn().mockImplementation((target: Record<string, unknown>) => {
      const c = makeController(target);
      controllers.push(c);
      return c;
    }),
    open: vi.fn(),
    domElement: root,
    controllers,
  };
}

function byName(folder: FolderStub, label: string): ControllerStub {
  const found = folder.controllers.find((c) => c._name === label);
  if (!found) throw new Error(`No control named "${label}"`);
  return found;
}

function rowValue(folder: FolderStub, label: string): HTMLElement {
  const rows = Array.from(
    folder.domElement.querySelectorAll('.luxar-gui__children .luxar-gui__controller')
  );
  const row = rows.find(
    (r) => r.querySelector('.luxar-gui__controller-name')?.textContent === label
  );
  if (!row) throw new Error(`No row named "${label}"`);
  return row.querySelector('.luxar-gui__controller-widget') as HTMLElement;
}

function makeControl(overrides: Partial<DensityGuardControl> = {}): DensityGuardControl & {
  setEnabled: ReturnType<typeof vi.fn>;
} {
  let enabled = true;
  const control = {
    isEnabled: () => enabled,
    sessionDisabled: false,
    setEnabled: vi.fn((on: boolean) => {
      enabled = on;
    }),
    thinning: () => ({ nodes: 0, minKeep: 1 }),
    capElementsPerPixel: () => 4,
    ...overrides,
  };
  return control as DensityGuardControl & { setEnabled: ReturnType<typeof vi.fn> };
}

describe('formatKeepFraction / formatThinning', () => {
  it('formats keep fractions as 1/K', () => {
    expect(formatKeepFraction(1)).toBe('1');
    expect(formatKeepFraction(0.5)).toBe('1/2');
    expect(formatKeepFraction(1 / 64)).toBe('1/64');
  });

  it('reads off / none / N nodes · keep 1/K, with the cap in force', () => {
    let enabled = false;
    const control = makeControl({ isEnabled: () => enabled });
    expect(formatThinning(control)).toBe('off');
    enabled = true;
    expect(formatThinning(control)).toBe('none · cap 4');
    expect(formatThinning(makeControl({ thinning: () => ({ nodes: 1, minKeep: 0.25 }) }))).toBe(
      '1 node · keep 1/4 · cap 4'
    );
    expect(formatThinning(makeControl({ thinning: () => ({ nodes: 3, minKeep: 1 / 8 }) }))).toBe(
      '3 nodes · keep 1/8 · cap 4'
    );
    // A `?densityCap=` sweep shows the value as typed — halving down from 4
    // reaches 0.25, which must not round to 0.3.
    expect(formatThinning(makeControl({ capElementsPerPixel: () => 2.5 }))).toBe('none · cap 2.5');
    expect(formatThinning(makeControl({ capElementsPerPixel: () => 0.25 }))).toBe(
      'none · cap 0.25'
    );
    expect(formatThinning(makeControl({ capElementsPerPixel: () => 4.0 }))).toBe('none · cap 4');
  });
});

describe('setupPerformanceControls — Density Guard', () => {
  let folder: FolderStub;
  let saveSettings: ReturnType<typeof vi.fn>;
  let triggerAnimation: ReturnType<typeof vi.fn>;
  let settings: {
    adaptiveDPREnabled: boolean;
    allowHighDPR: boolean;
    densityGuardEnabled: boolean;
  };

  function makeContext(control?: DensityGuardControl): PerformanceSetupContext {
    return {
      gui: { addFolder: vi.fn().mockReturnValue(folder) },
      settings,
      manager: {
        getNativeDPR: vi.fn().mockReturnValue(2),
        setManualDPR: vi.fn(),
        setEnabled: vi.fn(),
        isActive: vi.fn().mockReturnValue(true),
        getCurrentDPR: vi.fn().mockReturnValue(1),
        getState: vi.fn().mockReturnValue({ currentDPR: 1, currentFPS: 60 }),
        isPinned: vi.fn().mockReturnValue(false),
        isHighDPRAllowed: vi.fn().mockReturnValue(false),
        setHighDPRAllowed: vi.fn(),
      },
      densityGuard: control,
      saveSettings,
      triggerAnimation,
    } as unknown as PerformanceSetupContext;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    folder = makeFolder();
    saveSettings = vi.fn();
    triggerAnimation = vi.fn();
    settings = { adaptiveDPREnabled: true, allowHighDPR: false, densityGuardEnabled: true };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('without a handle: no toggle, no Thinning row, no controller returned', () => {
    const result = setupPerformanceControls(makeContext(undefined));
    expect(folder.controllers.map((c) => c._name)).toEqual([
      'Allow High DPR',
      'Adaptive Resolution',
      'Manual DPR',
    ]);
    expect(result.densityGuardEnabled).toBeUndefined();
    expect(() => rowValue(folder, 'Thinning')).toThrow();
  });

  it('adds the toggle between Adaptive Resolution and Manual DPR, plus the Thinning row', () => {
    const control = makeControl({ thinning: () => ({ nodes: 2, minKeep: 1 / 8 }) });
    const result = setupPerformanceControls(makeContext(control));
    expect(folder.controllers.map((c) => c._name)).toEqual([
      'Allow High DPR',
      'Adaptive Resolution',
      'Density Guard',
      'Manual DPR',
    ]);
    expect(result.densityGuardEnabled).toBe(byName(folder, 'Density Guard'));
    expect(rowValue(folder, 'Thinning').textContent).toBe('2 nodes · keep 1/8 · cap 4');
  });

  it('binds the toggle to the LIVE guard state, not the stored flag', () => {
    // Stored says on, session says off (`?noDensityGuard`).
    const control = makeControl({ isEnabled: () => false, sessionDisabled: true });
    setupPerformanceControls(makeContext(control));
    expect(byName(folder, 'Density Guard')._target).toEqual({ enabled: false });
    expect(settings.densityGuardEnabled).toBe(true); // untouched
    expect(rowValue(folder, 'Thinning').textContent).toBe('off');
  });

  it('toggle off → guard off, flag stored, saved, redrawn, row updated', () => {
    const control = makeControl();
    setupPerformanceControls(makeContext(control));
    byName(folder, 'Density Guard')._onChangeFn?.(false);
    expect(control.setEnabled).toHaveBeenCalledWith(false);
    expect(settings.densityGuardEnabled).toBe(false);
    expect(saveSettings).toHaveBeenCalledTimes(1);
    expect(triggerAnimation).toHaveBeenCalledTimes(1);
    expect(rowValue(folder, 'Thinning').textContent).toBe('off');
  });

  it('under a session disable the stored flag is neither written nor saved', () => {
    const control = makeControl({ isEnabled: () => false, sessionDisabled: true });
    settings.densityGuardEnabled = false;
    setupPerformanceControls(makeContext(control));
    byName(folder, 'Density Guard')._onChangeFn?.(true);
    expect(control.setEnabled).toHaveBeenCalledWith(true);
    expect(settings.densityGuardEnabled).toBe(false);
    expect(saveSettings).not.toHaveBeenCalled();
    expect(triggerAnimation).toHaveBeenCalledTimes(1);
  });

  it('refreshes the Thinning row on the 500 ms tick, adaptive on or off', () => {
    let nodes = 0;
    const control = makeControl({ thinning: () => ({ nodes, minKeep: nodes ? 0.5 : 1 }) });
    settings.adaptiveDPREnabled = false;
    const context = makeContext(control);
    (context.manager as unknown as { isActive: ReturnType<typeof vi.fn> }).isActive.mockReturnValue(
      false
    );
    setupPerformanceControls(context);
    expect(rowValue(folder, 'Thinning').textContent).toBe('none · cap 4');
    nodes = 1;
    vi.advanceTimersByTime(500);
    expect(rowValue(folder, 'Thinning').textContent).toBe('1 node · keep 1/2 · cap 4');
  });
});
