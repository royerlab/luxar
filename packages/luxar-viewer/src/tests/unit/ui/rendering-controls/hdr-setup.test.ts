// @vitest-environment jsdom
/**
 * Unit tests for ui/rendering-controls/setup/hdr-setup.ts.
 *
 * Stubs the GUI/Folder + sceneManager/postProcessing dependencies and
 * verifies that setupHDRControls wires its four onChange callbacks
 * (Exposure, Offset, Gamma, Tone Mapping) to the right downstream
 * methods + re-saves settings + triggers a re-render.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import type { SetupContext } from '../../../../ui/rendering-controls/types';
import { setupHDRControls } from '../../../../ui/rendering-controls/setup/hdr-setup';

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  /** Captured by the stub so the test can fire it. */
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
  ctrl.onChange.mockImplementation((fn: (value: unknown) => void) => {
    ctrl._onChangeFn = fn;
    return ctrl;
  });
  return ctrl;
}

interface FolderStub {
  open: ReturnType<typeof vi.fn>;
  add: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  /** All controllers created in order: exposure, globalOffset, globalGamma, toneMapping. */
  controllers: ControllerStub[];
}

function makeFolder(): FolderStub {
  const controllers: ControllerStub[] = [];
  const folder: FolderStub = {
    open: vi.fn(),
    add: vi.fn().mockImplementation(() => {
      const c = makeController();
      controllers.push(c);
      return c;
    }),
    domElement: document.createElement('div'),
    controllers,
  };
  return folder;
}

interface ContextStubs {
  context: SetupContext;
  folder: FolderStub;
  sceneManager: {
    updateExposure: ReturnType<typeof vi.fn>;
    updateGlobalOffset: ReturnType<typeof vi.fn>;
    updateGlobalGamma: ReturnType<typeof vi.fn>;
  };
  postProcessing: { setToneMapping: ReturnType<typeof vi.fn> };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  settings: { exposure: number; globalOffset: number; globalGamma: number; toneMapping: string };
}

function makeContext(): ContextStubs {
  const folder = makeFolder();
  const gui = {
    addFolder: vi.fn().mockReturnValue(folder),
  };
  const sceneManager = {
    updateExposure: vi.fn(),
    updateGlobalOffset: vi.fn(),
    updateGlobalGamma: vi.fn(),
  };
  const postProcessing = { setToneMapping: vi.fn() };
  const saveSettings = vi.fn();
  const triggerAnimation = vi.fn();
  const settings = {
    exposure: 0,
    globalOffset: 0,
    globalGamma: 1,
    toneMapping: 'ACES',
  };
  const context = {
    gui,
    settings,
    postProcessing,
    sceneManager,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState: vi.fn(),
    updateNavigationControls: vi.fn(),
  } as unknown as SetupContext;
  return {
    context,
    folder,
    sceneManager,
    postProcessing,
    saveSettings,
    triggerAnimation,
    settings,
  };
}

describe('setupHDRControls', () => {
  let stubs: ContextStubs;

  beforeEach(() => {
    stubs = makeContext();
  });

  it('creates the HDR folder, opens it, and adds 4 controls', () => {
    setupHDRControls(stubs.context);

    expect(stubs.folder.open).toHaveBeenCalled();
    expect(stubs.folder.add).toHaveBeenCalledTimes(4);
    expect(stubs.folder.controllers).toHaveLength(4);
  });

  it('sets a friendly title attribute on the folder DOM element', () => {
    setupHDRControls(stubs.context);
    expect(stubs.folder.domElement.getAttribute('title')).toContain('HDR');
  });

  it('returns controller refs for exposure / globalOffset / globalGamma', () => {
    const result = setupHDRControls(stubs.context);
    expect(result.controllers.exposure).toBe(stubs.folder.controllers[0]);
    expect(result.controllers.globalOffset).toBe(stubs.folder.controllers[1]);
    expect(result.controllers.globalGamma).toBe(stubs.folder.controllers[2]);
  });

  describe('exposure callback', () => {
    it('forwards to sceneManager.updateExposure + saveSettings + triggerAnimation', () => {
      setupHDRControls(stubs.context);
      const onChange = stubs.folder.controllers[0]._onChangeFn;
      onChange?.(2.5);

      expect(stubs.sceneManager.updateExposure).toHaveBeenCalledWith(2.5);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });

  describe('offset callback', () => {
    it('forwards to sceneManager.updateGlobalOffset', () => {
      setupHDRControls(stubs.context);
      stubs.folder.controllers[1]._onChangeFn?.(-0.25);

      expect(stubs.sceneManager.updateGlobalOffset).toHaveBeenCalledWith(-0.25);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });

  describe('gamma callback', () => {
    it('forwards to sceneManager.updateGlobalGamma', () => {
      setupHDRControls(stubs.context);
      stubs.folder.controllers[2]._onChangeFn?.(2.2);

      expect(stubs.sceneManager.updateGlobalGamma).toHaveBeenCalledWith(2.2);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });

  describe('tone mapping callback', () => {
    it.each([
      ['None', THREE.NoToneMapping],
      ['Linear', THREE.LinearToneMapping],
      ['Reinhard', THREE.ReinhardToneMapping],
      ['Cineon', THREE.CineonToneMapping],
      ['ACES', THREE.ACESFilmicToneMapping],
      ['AgX', THREE.AgXToneMapping],
      ['Neutral', THREE.NeutralToneMapping],
    ])('%s string maps to the correct THREE tone mapping constant', (label, constant) => {
      setupHDRControls(stubs.context);
      stubs.folder.controllers[3]._onChangeFn?.(label);

      expect(stubs.postProcessing.setToneMapping).toHaveBeenCalledWith(constant);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });
});
