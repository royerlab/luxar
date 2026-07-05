/**
 * Unit tests for ui/rendering-controls/setup/theme-setup.ts.
 *
 * Mocks ThemeManager.getInstance() and verifies that
 * setupThemeControls builds a single dropdown bound to setTheme()
 * + triggers a re-render on change.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { setupThemeControls } from '../../../../ui/rendering-controls/setup/theme-setup';

const setThemeMock = vi.fn();
const getCurrentThemeMock = vi.fn();
const getAllThemesMock = vi.fn();

vi.mock('../../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: vi.fn(() => ({
      setTheme: setThemeMock,
      getCurrentTheme: getCurrentThemeMock,
      getAllThemes: getAllThemesMock,
    })),
  },
}));

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  _onChangeFn: ((value: unknown) => void) | null;
}

function makeController(): ControllerStub {
  const ctrl: ControllerStub = {
    name: vi.fn(),
    onChange: vi.fn(),
    domElement: document.createElement('div'),
    _onChangeFn: null,
  };
  ctrl.name.mockReturnValue(ctrl);
  ctrl.onChange.mockImplementation((fn: (v: unknown) => void) => {
    ctrl._onChangeFn = fn;
    return ctrl;
  });
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
  return {
    add: vi.fn().mockImplementation(() => {
      const c = makeController();
      controllers.push(c);
      return c;
    }),
    open: vi.fn(),
    domElement: document.createElement('div'),
    controllers,
  };
}

describe('setupThemeControls', () => {
  let triggerAnimation: ReturnType<typeof vi.fn> & (() => void);
  let folder: FolderStub;
  let gui: { addFolder: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    vi.clearAllMocks();
    triggerAnimation = vi.fn() as ReturnType<typeof vi.fn> & (() => void);
    folder = makeFolder();
    gui = { addFolder: vi.fn().mockReturnValue(folder) };

    getAllThemesMock.mockReturnValue([
      { id: 'dark', name: 'Dark' },
      { id: 'light', name: 'Light' },
      { id: 'frosted-glass', name: 'Frosted Glass' },
    ]);
    getCurrentThemeMock.mockReturnValue({ id: 'frosted-glass', name: 'Frosted Glass' });
  });

  it('creates the Theme folder and adds one dropdown', () => {
    setupThemeControls({
      gui: gui as unknown as Parameters<typeof setupThemeControls>[0]['gui'],
      triggerAnimation,
    });

    expect(gui.addFolder).toHaveBeenCalledWith('Theme', expect.any(String));
    expect(folder.add).toHaveBeenCalledTimes(1);
    expect(folder.controllers).toHaveLength(1);
  });

  it('builds the options dictionary from getAllThemes() (name → id mapping)', () => {
    setupThemeControls({
      gui: gui as unknown as Parameters<typeof setupThemeControls>[0]['gui'],
      triggerAnimation,
    });

    const addCallArgs = folder.add.mock.calls[0];
    const optionsArg = addCallArgs[2];
    expect(optionsArg).toEqual({
      Dark: 'dark',
      Light: 'light',
      'Frosted Glass': 'frosted-glass',
    });
  });

  it('seeds the dropdown with the current theme id', () => {
    setupThemeControls({
      gui: gui as unknown as Parameters<typeof setupThemeControls>[0]['gui'],
      triggerAnimation,
    });

    const addCallArgs = folder.add.mock.calls[0];
    const settingsArg = addCallArgs[0];
    const propArg = addCallArgs[1];
    expect(settingsArg[propArg]).toBe('frosted-glass');
  });

  it('forwards a theme change to ThemeManager.setTheme + triggerAnimation', () => {
    setupThemeControls({
      gui: gui as unknown as Parameters<typeof setupThemeControls>[0]['gui'],
      triggerAnimation,
    });

    folder.controllers[0]._onChangeFn?.('dark');

    expect(setThemeMock).toHaveBeenCalledWith('dark');
    expect(triggerAnimation).toHaveBeenCalled();
  });

  it('opens the folder by default (shown directly in the Settings popover)', () => {
    setupThemeControls({
      gui: gui as unknown as Parameters<typeof setupThemeControls>[0]['gui'],
      triggerAnimation,
    });
    expect(folder.open).toHaveBeenCalled();
  });

  it('sets a tooltip on the folder DOM', () => {
    setupThemeControls({
      gui: gui as unknown as Parameters<typeof setupThemeControls>[0]['gui'],
      triggerAnimation,
    });
    expect(folder.domElement.getAttribute('title')).toContain('Theme');
  });
});
