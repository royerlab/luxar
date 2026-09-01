// @vitest-environment jsdom
/**
 * Unit tests for ui/rendering-controls/setup/anti-aliasing-setup.ts.
 *
 * Stubs the GUI/Folder + postProcessing dependencies and verifies
 * that setupAntiAliasingControls wires its onChange callbacks
 * (SSAA, SSAA multiplier, FXAA, MSAA, MSAA samples, SMAA) to the
 * right downstream methods + re-saves settings + triggers a re-render.
 * Also covers the show/hide of SSAA + MSAA subfolders.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { SetupContext } from '../../../../ui/rendering-controls/types';
import { setupAntiAliasingControls } from '../../../../ui/rendering-controls/setup/anti-aliasing-setup';

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
  open: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  add: ReturnType<typeof vi.fn>;
  addFolder: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  controllers: ControllerStub[];
  childOrder: string[];
  /** Sub-folders created via addFolder, keyed by display name. */
  subFolders: Map<string, FolderStub>;
}

function makeFolder(): FolderStub {
  const controllers: ControllerStub[] = [];
  const childOrder: string[] = [];
  const subFolders = new Map<string, FolderStub>();
  const folder: FolderStub = {
    open: vi.fn(),
    close: vi.fn(),
    show: vi.fn(),
    hide: vi.fn(),
    add: vi.fn().mockImplementation((_object: unknown, property: string) => {
      const c = makeController();
      controllers.push(c);
      childOrder.push(`controller:${property}`);
      return c;
    }),
    addFolder: vi.fn().mockImplementation((displayName: string) => {
      const sub = makeFolder();
      subFolders.set(displayName, sub);
      childOrder.push(`folder:${displayName}`);
      return sub;
    }),
    domElement: document.createElement('div'),
    controllers,
    childOrder,
    subFolders,
  };
  return folder;
}

interface ContextStubs {
  context: SetupContext;
  aaFolder: FolderStub;
  postProcessing: {
    setSSAAEnabled: ReturnType<typeof vi.fn>;
    setSSAAMultiplier: ReturnType<typeof vi.fn>;
    setFXAAEnabled: ReturnType<typeof vi.fn>;
    setMSAAEnabled: ReturnType<typeof vi.fn>;
    setMSAASamples: ReturnType<typeof vi.fn>;
  };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  settings: {
    ssaaEnabled: boolean;
    ssaaMultiplier: number;
    fxaaEnabled: boolean;
    msaaEnabled: boolean;
    msaaSamples: number;
  };
}

function makeContext(initialSettings: Partial<ContextStubs['settings']> = {}): ContextStubs {
  const aaFolder = makeFolder();
  const gui = {
    addFolder: vi.fn().mockReturnValue(aaFolder),
  };
  const postProcessing = {
    setSSAAEnabled: vi.fn(),
    setSSAAMultiplier: vi.fn(),
    setFXAAEnabled: vi.fn(),
    setMSAAEnabled: vi.fn(),
    setMSAASamples: vi.fn(),
  };
  const saveSettings = vi.fn();
  const triggerAnimation = vi.fn();
  const settings = {
    ssaaEnabled: false,
    ssaaMultiplier: 2.0,
    fxaaEnabled: false,
    msaaEnabled: false,
    msaaSamples: 4,
    ...initialSettings,
  };
  const context = {
    gui,
    settings,
    postProcessing,
    sceneManager: {} as unknown,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState: vi.fn(),
    updateNavigationControls: vi.fn(),
  } as unknown as SetupContext;
  return { context, aaFolder, postProcessing, saveSettings, triggerAnimation, settings };
}

describe('setupAntiAliasingControls', () => {
  let stubs: ContextStubs;

  beforeEach(() => {
    stubs = makeContext();
  });

  it('creates the AA folder, closes it by default, and adds 3 toggles', () => {
    setupAntiAliasingControls(stubs.context);
    // Top folder closed (collapsed).
    expect(stubs.aaFolder.close).toHaveBeenCalled();
    // Three toggles added directly under aa folder: SSAA, FXAA, MSAA.
    expect(stubs.aaFolder.controllers).toHaveLength(3);
  });

  it('creates SSAA + MSAA sub-folders', () => {
    setupAntiAliasingControls(stubs.context);
    expect(stubs.aaFolder.subFolders.has('SSAA Settings (Supersampling)')).toBe(true);
    expect(stubs.aaFolder.subFolders.has('MSAA Settings')).toBe(true);
  });

  it('places each settings sub-folder after its enable toggle', () => {
    setupAntiAliasingControls(stubs.context);

    expect(stubs.aaFolder.childOrder).toEqual([
      'controller:ssaaEnabled',
      'folder:SSAA Settings (Supersampling)',
      'controller:fxaaEnabled',
      'controller:msaaEnabled',
      'folder:MSAA Settings',
    ]);
  });

  it('SSAA sub-folder receives the multiplier control', () => {
    setupAntiAliasingControls(stubs.context);
    const ssaaSub = stubs.aaFolder.subFolders.get('SSAA Settings (Supersampling)');
    expect(ssaaSub?.controllers).toHaveLength(1);
  });

  describe('SSAA toggle', () => {
    it('forwards on, opens + shows the SSAA sub-folder', () => {
      setupAntiAliasingControls(stubs.context);
      const ssaaToggle = stubs.aaFolder.controllers[0];
      const ssaaSub = stubs.aaFolder.subFolders.get('SSAA Settings (Supersampling)')!;

      ssaaToggle._onChangeFn?.(true);

      expect(stubs.postProcessing.setSSAAEnabled).toHaveBeenCalledWith(true);
      expect(ssaaSub.show).toHaveBeenCalled();
      expect(ssaaSub.open).toHaveBeenCalled();
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });

    it('forwards off, closes + hides the SSAA sub-folder', () => {
      setupAntiAliasingControls(stubs.context);
      const ssaaToggle = stubs.aaFolder.controllers[0];
      const ssaaSub = stubs.aaFolder.subFolders.get('SSAA Settings (Supersampling)')!;

      ssaaToggle._onChangeFn?.(false);

      expect(stubs.postProcessing.setSSAAEnabled).toHaveBeenCalledWith(false);
      expect(ssaaSub.close).toHaveBeenCalled();
      expect(ssaaSub.hide).toHaveBeenCalled();
    });
  });

  describe('SSAA multiplier', () => {
    it('forwards the multiplier value to postProcessing', () => {
      setupAntiAliasingControls(stubs.context);
      const ssaaSub = stubs.aaFolder.subFolders.get('SSAA Settings (Supersampling)')!;
      const multiplierCtrl = ssaaSub.controllers[0];

      multiplierCtrl._onChangeFn?.(3.0);

      expect(stubs.postProcessing.setSSAAMultiplier).toHaveBeenCalledWith(3.0);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });

  describe('FXAA toggle', () => {
    it('forwards to postProcessing.setFXAAEnabled', () => {
      setupAntiAliasingControls(stubs.context);
      const fxaaToggle = stubs.aaFolder.controllers[1];
      fxaaToggle._onChangeFn?.(true);
      expect(stubs.postProcessing.setFXAAEnabled).toHaveBeenCalledWith(true);
    });
  });

  describe('MSAA toggle', () => {
    it('forwards on + opens/shows MSAA sub-folder', () => {
      setupAntiAliasingControls(stubs.context);
      const msaaToggle = stubs.aaFolder.controllers[2];
      const msaaSub = stubs.aaFolder.subFolders.get('MSAA Settings')!;

      msaaToggle._onChangeFn?.(true);

      expect(stubs.postProcessing.setMSAAEnabled).toHaveBeenCalledWith(true);
      expect(msaaSub.show).toHaveBeenCalled();
      expect(msaaSub.open).toHaveBeenCalled();
    });

    it('forwards off + closes/hides MSAA sub-folder', () => {
      setupAntiAliasingControls(stubs.context);
      const msaaToggle = stubs.aaFolder.controllers[2];
      const msaaSub = stubs.aaFolder.subFolders.get('MSAA Settings')!;

      msaaToggle._onChangeFn?.(false);

      expect(stubs.postProcessing.setMSAAEnabled).toHaveBeenCalledWith(false);
      expect(msaaSub.close).toHaveBeenCalled();
      expect(msaaSub.hide).toHaveBeenCalled();
    });
  });

  describe('MSAA samples', () => {
    it('forwards the sample count value to postProcessing', () => {
      setupAntiAliasingControls(stubs.context);
      const msaaSub = stubs.aaFolder.subFolders.get('MSAA Settings')!;
      const samplesCtrl = msaaSub.controllers[0];

      samplesCtrl._onChangeFn?.(8);

      expect(stubs.postProcessing.setMSAASamples).toHaveBeenCalledWith(8);
    });
  });

  describe('initial folder visibility', () => {
    it('hides SSAA sub-folder when ssaaEnabled=false at init', () => {
      stubs = makeContext({ ssaaEnabled: false });
      setupAntiAliasingControls(stubs.context);
      const ssaaSub = stubs.aaFolder.subFolders.get('SSAA Settings (Supersampling)')!;
      expect(ssaaSub.hide).toHaveBeenCalled();
      expect(ssaaSub.show).not.toHaveBeenCalled();
    });

    it('shows + opens SSAA sub-folder when ssaaEnabled=true at init', () => {
      stubs = makeContext({ ssaaEnabled: true });
      setupAntiAliasingControls(stubs.context);
      const ssaaSub = stubs.aaFolder.subFolders.get('SSAA Settings (Supersampling)')!;
      expect(ssaaSub.show).toHaveBeenCalled();
      expect(ssaaSub.open).toHaveBeenCalled();
    });

    it('hides MSAA sub-folder when msaaEnabled=false at init', () => {
      stubs = makeContext({ msaaEnabled: false });
      setupAntiAliasingControls(stubs.context);
      const msaaSub = stubs.aaFolder.subFolders.get('MSAA Settings')!;
      expect(msaaSub.hide).toHaveBeenCalled();
    });

    it('does not hide MSAA sub-folder when msaaEnabled=true at init', () => {
      stubs = makeContext({ msaaEnabled: true });
      setupAntiAliasingControls(stubs.context);
      const msaaSub = stubs.aaFolder.subFolders.get('MSAA Settings')!;
      // The init block only calls hide() when disabled — the toggle's
      // onChange would handle show() when enabled.
      expect(msaaSub.hide).not.toHaveBeenCalled();
    });
  });
});
