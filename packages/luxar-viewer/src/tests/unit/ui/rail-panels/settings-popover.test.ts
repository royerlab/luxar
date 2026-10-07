// @vitest-environment jsdom
/**
 * Unit tests for ui/rail-panels/settings-popover.ts.
 *
 * The Settings popover hosts the theme picker plus the persisted global
 * preferences (config/user-settings.ts) as folders: Input, Performance,
 * Caching, Advanced. These tests stub the popover GUI (folders + controls
 * keyed by bound property) and verify: folder structure, live apply +
 * persistence on change, the Budget (MB) visibility toggle, the Clear Caches
 * action (incl. the no-loader case), the reload-hint row lifecycle, and
 * Reset All.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { buildSettingsPopover } from '../../../../ui/rail-panels/settings-popover';
import {
  defaultUserSettings,
  initUserSettings,
  loadUserSettings,
  saveUserSettings,
  resetUserSettingsForTests,
  applyUrlPerformanceOverrides,
} from '../../../../config/user-settings';
import { config } from '../../../../config';
import type { SceneLoader } from '../../../../data/scene-loader';

interface ControllerStub {
  prop: string;
  boundObj: Record<string, unknown>;
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  _onChangeFn: ((value: unknown) => void) | null;
  domElement: HTMLElement;
  /** Simulate a user edit: write the value onto the bound object, then fire onChange. */
  set(value: unknown): void;
}

interface FolderStub {
  title: string;
  add: ReturnType<typeof vi.fn>;
  addFolder: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
}

interface GuiStub extends FolderStub {
  destroy: ReturnType<typeof vi.fn>;
  controllers: ControllerStub[];
  folders: FolderStub[];
}

let currentGui: GuiStub;

function makeController(boundObj: Record<string, unknown>, prop: string): ControllerStub {
  const ctrl: ControllerStub = {
    prop,
    boundObj,
    name: vi.fn(),
    onChange: vi.fn(),
    hide: vi.fn(),
    show: vi.fn(),
    _onChangeFn: null,
    domElement: document.createElement('div'),
    set(value: unknown) {
      boundObj[prop] = value;
      this._onChangeFn?.(value);
    },
  };
  if (prop === 'workerCount' || prop === 'networkMaxConcurrent') {
    const slider = document.createElement('input');
    slider.type = 'range';
    const number = document.createElement('input');
    number.type = 'number';
    ctrl.domElement.append(slider, number);
  } else if (prop === 'renderer') {
    ctrl.domElement.appendChild(document.createElement('select'));
  }
  ctrl.name.mockReturnValue(ctrl);
  ctrl.onChange.mockImplementation((fn: (v: unknown) => void) => {
    ctrl._onChangeFn = fn;
    return ctrl;
  });
  return ctrl;
}

function makeFolder(title: string, sink: ControllerStub[], folders: FolderStub[]): FolderStub {
  const folder: FolderStub = {
    title,
    add: vi.fn().mockImplementation((obj: Record<string, unknown>, prop: string) => {
      const c = makeController(obj, prop);
      sink.push(c);
      return c;
    }),
    addFolder: vi.fn().mockImplementation((childTitle: string) => {
      const child = makeFolder(childTitle, sink, folders);
      folders.push(child);
      return child;
    }),
    domElement: document.createElement('div'),
  };
  return folder;
}

function makeGui(): GuiStub {
  const controllers: ControllerStub[] = [];
  const folders: FolderStub[] = [];
  const root = makeFolder('Settings', controllers, folders) as GuiStub;
  root.destroy = vi.fn();
  root.controllers = controllers;
  root.folders = folders;
  return root;
}

vi.mock('../../../../ui/rail-panels/popover-gui', () => ({
  makePopoverGui: vi.fn(() => currentGui),
}));

// Theme controls pull in the ThemeManager singleton — stub them out; the
// theme picker has its own tests.
vi.mock('../../../../ui/rendering-controls/setup/theme-setup', () => ({
  setupThemeControls: vi.fn(),
}));

function byProp(prop: string): ControllerStub {
  const c = currentGui.controllers.find((x) => x.prop === prop);
  if (!c)
    throw new Error(
      `no controller for "${prop}" (have: ${currentGui.controllers.map((x) => x.prop).join(', ')})`
    );
  return c;
}

function makeLoader(): { loader: SceneLoader; clearAllCaches: ReturnType<typeof vi.fn> } {
  const clearAllCaches = vi.fn().mockResolvedValue(undefined);
  const loader = {
    clearAllCaches,
    getCacheBudgets: vi.fn().mockReturnValue({
      l0Bytes: 200 * 1024 * 1024,
      l1Bytes: 100 * 1024 * 1024,
      sliceBytes: 128 * 1024 * 1024,
      heapAware: true,
      source: 'heap',
    }),
  } as unknown as SceneLoader;
  return { loader, clearAllCaches };
}

function build(loader: SceneLoader | null = null): {
  host: HTMLElement;
  teardown: () => void;
  triggerAnimation: ReturnType<typeof vi.fn>;
} {
  currentGui = makeGui();
  const host = document.createElement('div');
  const triggerAnimation = vi.fn();
  const teardown = buildSettingsPopover(host, {
    triggerAnimation,
    getSceneLoader: () => loader,
  });
  return { host, teardown, triggerAnimation };
}

const configSnapshot = {
  fovSensitivity: config.camera.fovSensitivity,
  wheelZoomSensitivity: config.controls.wheelZoomSensitivity,
  idleTimeoutMs: config.animation.idleTimeoutMs,
  useWebWorkers: config.dataLoading.performance.useWebWorkers,
  workerCount: config.dataLoading.performance.workerCount,
  maxConcurrent: config.dataLoading.network.maxConcurrent,
};

// Captured at MODULE LOAD, before any test mutates config. Reset-behavior
// assertions must compare against this snapshot — a fresh
// defaultUserSettings() call would drift in lockstep if defaults were
// (re-)derived from the mutated config, masking the very bug under test.
const BUILTIN_DEFAULTS = defaultUserSettings();

beforeEach(() => {
  localStorage.clear();
  resetUserSettingsForTests();
  initUserSettings(); // establish a boot snapshot, as bootstrap does
});

afterEach(() => {
  config.camera.fovSensitivity = configSnapshot.fovSensitivity;
  config.controls.wheelZoomSensitivity = configSnapshot.wheelZoomSensitivity;
  config.animation.idleTimeoutMs = configSnapshot.idleTimeoutMs;
  config.dataLoading.performance.useWebWorkers = configSnapshot.useWebWorkers;
  config.dataLoading.performance.workerCount = configSnapshot.workerCount;
  config.dataLoading.network.maxConcurrent = configSnapshot.maxConcurrent;
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('buildSettingsPopover', () => {
  it('builds the four preference folders (Input / Performance / Caching / Advanced)', () => {
    build();
    expect(currentGui.folders.map((f) => f.title)).toEqual([
      'Input',
      'Performance',
      'Caching',
      'Advanced',
    ]);
  });

  it('changing FOV sensitivity applies live to config AND persists', () => {
    build();
    byProp('fovSensitivity').set(0.15);
    expect(config.camera.fovSensitivity).toBe(0.15);
    expect(loadUserSettings().input.fovSensitivity).toBe(0.15);
  });

  it('changing Zoom sensitivity applies live to config AND persists (no reload hint)', () => {
    const { host } = build();
    byProp('wheelZoomSensitivity').set(0.25);
    expect(config.controls.wheelZoomSensitivity).toBe(0.25);
    expect(loadUserSettings().input.wheelZoomSensitivity).toBe(0.25);
    const hint = host.querySelector<HTMLElement>('.luxar-control-rail__popover-hint')!;
    expect(hint.style.display).toBe('none');
  });

  it('idle timeout is edited in seconds but stored in ms', () => {
    build();
    byProp('idleTimeoutS').set(4);
    expect(config.animation.idleTimeoutMs).toBe(4000);
    expect(loadUserSettings().performance.idleTimeoutMs).toBe(4000);
  });

  it('Budget (MB) slider is hidden in Auto mode and shown in Custom', () => {
    build();
    const mb = byProp('budgetMB');
    expect(mb.hide).toHaveBeenCalled(); // starts hidden (defaults are auto)
    byProp('budgetMode').set('custom');
    expect(mb.show).toHaveBeenCalled();
    byProp('budgetMode').set('auto');
    expect(mb.hide).toHaveBeenCalledTimes(2);
  });

  it('reload hint is hidden at boot state and appears after a reload-key change', () => {
    const { host } = build();
    const hint = host.querySelector<HTMLElement>('.luxar-control-rail__popover-hint')!;
    expect(hint.style.display).toBe('none');
    byProp('renderer').set('webgpu');
    expect(hint.style.display).toBe('');
    // Reverting to the boot value clears the hint again.
    byProp('renderer').set('auto');
    expect(hint.style.display).toBe('none');
  });

  it('live-only changes never show the reload hint', () => {
    const { host } = build();
    byProp('fovSensitivity').set(0.19);
    byProp('useWebWorkers').set(false);
    const hint = host.querySelector<HTMLElement>('.luxar-control-rail__popover-hint')!;
    expect(hint.style.display).toBe('none');
  });

  it('Clear Caches invokes the loader clear-all (all tiers)', async () => {
    const { loader, clearAllCaches } = makeLoader();
    build(loader);
    byProp('clearCaches').boundObj.clearCaches; // function controller binds the action
    await (byProp('clearCaches').boundObj.clearCaches as () => Promise<void>)();
    expect(clearAllCaches).toHaveBeenCalledTimes(1);
  });

  it('Clear Caches tolerates a missing loader (before first scene load)', async () => {
    build(null);
    await expect(
      (byProp('clearCaches').boundObj.clearCaches as () => Promise<void>)()
    ).resolves.toBeUndefined();
  });

  it('shows the resolved budget line when a loader is present, fallback otherwise', () => {
    const { loader } = makeLoader();
    build(loader);
    const notes = currentGui.folders.map((f) => f.domElement.textContent).join(' ');
    expect(notes).toContain('Resolved (heap)');
    expect(notes).toContain('L0 200');

    build(null);
    const fallback = currentGui.folders.map((f) => f.domElement.textContent).join(' ');
    expect(fallback).toContain('Budgets resolve on scene load');
  });

  it('Reset All restores the BUILT-IN defaults (pre-mutation snapshot), persists, rebuilds', () => {
    build();
    byProp('fovSensitivity').set(0.19);
    expect(loadUserSettings().input.fovSensitivity).toBe(0.19);

    (byProp('resetAll').boundObj.resetAll as () => void)();
    // Compare against the module-load snapshot, NOT a fresh
    // defaultUserSettings() call — the fresh call would drift together with
    // the bug (defaults re-derived from mutated config) and mask it.
    expect(loadUserSettings()).toEqual(BUILTIN_DEFAULTS);
    expect(config.camera.fovSensitivity).toBe(BUILTIN_DEFAULTS.input.fovSensitivity);
    expect(config.camera.fovSensitivity).not.toBe(0.19);
  });

  it('teardown destroys the GUI', () => {
    const { teardown } = build();
    const gui = currentGui;
    teardown();
    expect(gui.destroy).toHaveBeenCalledTimes(1);
  });

  it('a stored custom budget at boot does NOT show the hint until it changes again', () => {
    // Simulate: user set custom budget, reloaded — boot snapshot has it.
    const s = defaultUserSettings();
    s.caching.budgetMode = 'custom';
    s.caching.budgetMB = 512;
    saveUserSettings(s);
    resetUserSettingsForTests();
    initUserSettings();

    const { host } = build();
    const hint = host.querySelector<HTMLElement>('.luxar-control-rail__popover-hint')!;
    expect(hint.style.display).toBe('none');
    byProp('budgetMB').set(1024);
    expect(hint.style.display).toBe('');
  });
});

describe('machine settings the launch URL set', () => {
  it('show the URL value, locked and labelled, and never touch the saved preference', () => {
    saveUserSettings({
      ...defaultUserSettings(),
      performance: {
        ...defaultUserSettings().performance,
        workerCount: 0,
        networkMaxConcurrent: 4,
      },
    });
    applyUrlPerformanceOverrides(16, 12, 'webgpu');
    const { teardown } = build();

    for (const [prop, value, label] of [
      ['workerCount', 16, 'Workers (launch URL)'],
      ['networkMaxConcurrent', 12, 'Prefetch Limit (launch URL)'],
      ['renderer', 'webgpu', 'Renderer (launch URL)'],
    ] as const) {
      const c = byProp(prop);
      expect(c.boundObj[prop]).toBe(value);
      expect(c.name).toHaveBeenCalledWith(label);
      expect(c.onChange).not.toHaveBeenCalled();
      expect(c.domElement.style.pointerEvents).toBe('none');
      const inputs = c.domElement.querySelectorAll<HTMLInputElement | HTMLSelectElement>(
        'input, select'
      );
      expect(inputs.length).toBe(prop === 'renderer' ? 1 : 2);
      inputs.forEach((input) => expect(input.disabled).toBe(true));
    }
    expect(loadUserSettings().performance.workerCount).toBe(0);
    expect(loadUserSettings().performance.networkMaxConcurrent).toBe(4);
    teardown();
  });

  it('edit the saved preference as usual when the URL set nothing', () => {
    const { teardown } = build();
    const c = byProp('workerCount');
    expect(c.name).toHaveBeenCalledWith('Workers (0 = auto)');
    c.domElement
      .querySelectorAll<HTMLInputElement>('input')
      .forEach((input) => expect(input.disabled).toBe(false));
    c.set(8);
    expect(loadUserSettings().performance.workerCount).toBe(8);
    teardown();
  });
});
