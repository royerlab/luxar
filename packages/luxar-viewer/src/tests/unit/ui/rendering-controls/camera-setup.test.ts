// @vitest-environment jsdom
/**
 * Unit tests for ui/rendering-controls/setup/camera-setup.ts.
 *
 * Stubs the GUI/Folder + sceneManager + postProcessing dependencies,
 * then verifies setupCameraControls wires the FOV preset dropdown,
 * FOV slider, near/far clipping plane sliders, and dynamic-clipping
 * toggle to the right downstream methods.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { SetupContext, SetupResult } from '../../../../ui/rendering-controls/types';
import { setupCameraControls } from '../../../../ui/rendering-controls/setup/camera-setup';

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  setValue: ReturnType<typeof vi.fn>;
  updateDisplay: ReturnType<typeof vi.fn>;
  $input: HTMLSelectElement | null;
  domElement: HTMLElement;
  _onChangeFn: ((value: unknown) => void) | null;
}

function makeController(): ControllerStub {
  // Provide a real <select> element so the production code's option
  // toggling between 'Custom' and the focal-length label can run for real.
  const select = document.createElement('select');
  ['28mm Wide', '35mm', '50mm Normal', '85mm Portrait', '135mm Tele', 'Custom'].forEach((label) => {
    const opt = document.createElement('option');
    opt.value = label;
    opt.textContent = label;
    select.appendChild(opt);
  });

  const ctrl: ControllerStub = {
    name: vi.fn(),
    onChange: vi.fn(),
    setValue: vi.fn(),
    updateDisplay: vi.fn(),
    $input: select,
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
  addFolder: ReturnType<typeof vi.fn>;
  open: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  controllers: ControllerStub[];
  subFolders: Map<string, FolderStub>;
}

function makeFolder(): FolderStub {
  const controllers: ControllerStub[] = [];
  const subFolders = new Map<string, FolderStub>();
  const folder: FolderStub = {
    add: vi.fn().mockImplementation(() => {
      const c = makeController();
      controllers.push(c);
      return c;
    }),
    addFolder: vi.fn().mockImplementation((name: string) => {
      const sub = makeFolder();
      subFolders.set(name, sub);
      return sub;
    }),
    open: vi.fn(),
    close: vi.fn(),
    domElement: document.createElement('div'),
    controllers,
    subFolders,
  };
  return folder;
}

interface ContextStubs {
  context: SetupContext;
  cameraFolder: FolderStub;
  sceneManager: {
    currentFov: number;
    updateFOV: ReturnType<typeof vi.fn>;
    updateClippingPlanes: ReturnType<typeof vi.fn>;
    setDynamicClipping: ReturnType<typeof vi.fn>;
  };
  postProcessing: { updateChromaticLensDistortion: ReturnType<typeof vi.fn> };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  updateClippingControlsState: ReturnType<typeof vi.fn>;
  controllersRef: SetupResult['controllers'];
  settings: {
    fov: number;
    fovPreset: string;
    near: number;
    far: number;
    dynamicClippingEnabled: boolean;
    chromaticLensDistortionEnabled: boolean;
    chromaticLensDistortionX: number;
    chromaticLensDistortionY: number;
    chromaticLensDispersion: number;
    chromaticLensPrincipalPointX: number;
    chromaticLensPrincipalPointY: number;
    chromaticLensFocalLengthX: number;
    chromaticLensFocalLengthY: number;
    chromaticLensSkew: number;
  };
}

function makeContext(initial: Partial<ContextStubs['settings']> = {}): ContextStubs {
  const cameraFolder = makeFolder();
  const gui = { addFolder: vi.fn().mockReturnValue(cameraFolder) };
  const sceneManager = {
    currentFov: 50,
    updateFOV: vi.fn(),
    updateClippingPlanes: vi.fn(),
    setDynamicClipping: vi.fn(),
  };
  const postProcessing = { updateChromaticLensDistortion: vi.fn() };
  const saveSettings = vi.fn();
  const triggerAnimation = vi.fn();
  const updateClippingControlsState = vi.fn();
  // Pre-populate controllersRef with stub controllers for the chromatic
  // lens distortion sync paths (mimicking the post-processing setup output).
  const controllersRef: SetupResult['controllers'] = {
    fov: makeController() as unknown as SetupResult['controllers']['fov'],
    fovPreset: makeController() as unknown as SetupResult['controllers']['fovPreset'],
    chromaticLensDistortionX:
      makeController() as unknown as SetupResult['controllers']['chromaticLensDistortionX'],
    chromaticLensDistortionY:
      makeController() as unknown as SetupResult['controllers']['chromaticLensDistortionY'],
    chromaticLensDispersion:
      makeController() as unknown as SetupResult['controllers']['chromaticLensDispersion'],
    chromaticLensPrincipalPointX:
      makeController() as unknown as SetupResult['controllers']['chromaticLensPrincipalPointX'],
    chromaticLensPrincipalPointY:
      makeController() as unknown as SetupResult['controllers']['chromaticLensPrincipalPointY'],
    chromaticLensFocalLengthX:
      makeController() as unknown as SetupResult['controllers']['chromaticLensFocalLengthX'],
    chromaticLensFocalLengthY:
      makeController() as unknown as SetupResult['controllers']['chromaticLensFocalLengthY'],
    chromaticLensSkew:
      makeController() as unknown as SetupResult['controllers']['chromaticLensSkew'],
  };
  const settings = {
    fov: 50,
    fovPreset: '50mm Normal',
    near: 0.1,
    far: 1000,
    dynamicClippingEnabled: false,
    chromaticLensDistortionEnabled: false,
    chromaticLensDistortionX: 0,
    chromaticLensDistortionY: 0,
    chromaticLensDispersion: 0.03,
    chromaticLensPrincipalPointX: 0,
    chromaticLensPrincipalPointY: 0,
    chromaticLensFocalLengthX: 1,
    chromaticLensFocalLengthY: 1,
    chromaticLensSkew: 0,
    ...initial,
  };
  const context = {
    gui,
    settings,
    postProcessing,
    sceneManager,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState,
    updateNavigationControls: vi.fn(),
  } as unknown as SetupContext;
  return {
    context,
    cameraFolder,
    sceneManager,
    postProcessing,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState,
    controllersRef,
    settings,
  };
}

describe('setupCameraControls', () => {
  let stubs: ContextStubs;

  beforeEach(() => {
    stubs = makeContext();
  });

  it('creates the Camera folder and opens it', () => {
    setupCameraControls(stubs.context, stubs.controllersRef);
    expect(stubs.cameraFolder.open).toHaveBeenCalled();
  });

  it('adds FOV preset dropdown + FOV slider, plus a Clipping Planes sub-folder', () => {
    setupCameraControls(stubs.context, stubs.controllersRef);
    // 2 controls at the top level: FOV preset, FOV slider.
    expect(stubs.cameraFolder.controllers).toHaveLength(2);
    expect(stubs.cameraFolder.subFolders.has('Clipping Planes')).toBe(true);
  });

  it('adds 3 controls inside the Clipping Planes sub-folder (near, far, dynamic)', () => {
    setupCameraControls(stubs.context, stubs.controllersRef);
    const clipping = stubs.cameraFolder.subFolders.get('Clipping Planes')!;
    expect(clipping.controllers).toHaveLength(3);
  });

  it('returns controller refs for fovPreset / fov / nearPlane / farPlane / dynamicClippingEnabled', () => {
    const result = setupCameraControls(stubs.context, stubs.controllersRef);
    expect(result.controllers.fovPreset).toBeDefined();
    expect(result.controllers.fov).toBeDefined();
    expect(result.controllers.nearPlane).toBeDefined();
    expect(result.controllers.farPlane).toBeDefined();
    expect(result.controllers.dynamicClippingEnabled).toBeDefined();
  });

  it('applies updateClippingControlsState with the initial setting on init', () => {
    stubs = makeContext({ dynamicClippingEnabled: true });
    setupCameraControls(stubs.context, stubs.controllersRef);
    expect(stubs.updateClippingControlsState).toHaveBeenCalledWith(true);
  });

  describe('FOV preset dropdown', () => {
    it('applies a known preset (28mm Wide) → fov=75 + sceneManager.updateFOV', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      const presetCtrl = stubs.cameraFolder.controllers[0];

      presetCtrl._onChangeFn?.('28mm Wide');

      expect(stubs.settings.fov).toBe(75);
      expect(stubs.sceneManager.updateFOV).toHaveBeenCalled();
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });

    it('updates the fov controller display via setValue + updateDisplay', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      const presetCtrl = stubs.cameraFolder.controllers[0];

      presetCtrl._onChangeFn?.('50mm Normal');

      const fovRef = stubs.controllersRef.fov as unknown as ControllerStub;
      expect(fovRef.setValue).toHaveBeenCalledWith(47);
      expect(fovRef.updateDisplay).toHaveBeenCalled();
    });

    it('Custom preset (fovValue=-1) does NOT mutate fov settings', () => {
      stubs.settings.fov = 50;
      setupCameraControls(stubs.context, stubs.controllersRef);
      const presetCtrl = stubs.cameraFolder.controllers[0];

      presetCtrl._onChangeFn?.('Custom');

      expect(stubs.settings.fov).toBe(50);
      expect(stubs.sceneManager.updateFOV).not.toHaveBeenCalled();
    });

    it('applies chromatic-lens preset only when chromaticLensDistortionEnabled', () => {
      stubs = makeContext({ chromaticLensDistortionEnabled: true });
      setupCameraControls(stubs.context, stubs.controllersRef);
      const presetCtrl = stubs.cameraFolder.controllers[0];

      presetCtrl._onChangeFn?.('28mm Wide');
      expect(stubs.postProcessing.updateChromaticLensDistortion).toHaveBeenCalled();
    });

    it('does NOT apply chromatic-lens preset when distortion is disabled', () => {
      stubs = makeContext({ chromaticLensDistortionEnabled: false });
      setupCameraControls(stubs.context, stubs.controllersRef);
      const presetCtrl = stubs.cameraFolder.controllers[0];

      presetCtrl._onChangeFn?.('28mm Wide');
      expect(stubs.postProcessing.updateChromaticLensDistortion).not.toHaveBeenCalled();
    });
  });

  describe('FOV slider', () => {
    it('flips fovPreset to "Custom" + updates fov controller', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      const fovCtrl = stubs.cameraFolder.controllers[1];

      fovCtrl._onChangeFn?.(60);

      expect(stubs.settings.fovPreset).toBe('Custom');
      expect(stubs.sceneManager.updateFOV).toHaveBeenCalled();
    });

    it('renames the Custom option text to ~XXmm to show focal length equivalent', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      const fovCtrl = stubs.cameraFolder.controllers[1];

      fovCtrl._onChangeFn?.(50);

      const presetRef = stubs.controllersRef.fovPreset as unknown as ControllerStub;
      const customOpt = Array.from(presetRef.$input!.options).find((o) => o.value === 'Custom');
      expect(customOpt?.textContent?.startsWith('~')).toBe(true);
      expect(customOpt?.textContent?.endsWith('mm')).toBe(true);
    });
  });

  describe('clipping planes', () => {
    function clipCtrls() {
      return stubs.cameraFolder.subFolders.get('Clipping Planes')!.controllers;
    }

    it('near plane updates pass through when value < far', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      clipCtrls()[0]._onChangeFn?.(0.5);

      expect(stubs.sceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.5, 1000);
    });

    it('near plane reject when value ≥ far (no scene update, no save)', () => {
      stubs.settings.far = 100;
      setupCameraControls(stubs.context, stubs.controllersRef);
      clipCtrls()[0]._onChangeFn?.(150);

      expect(stubs.sceneManager.updateClippingPlanes).not.toHaveBeenCalled();
      expect(stubs.saveSettings).not.toHaveBeenCalled();
    });

    it('far plane updates pass through when value > near', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      clipCtrls()[1]._onChangeFn?.(2000);

      expect(stubs.sceneManager.updateClippingPlanes).toHaveBeenCalledWith(0.1, 2000);
    });

    it('far plane reject when value ≤ near', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      clipCtrls()[1]._onChangeFn?.(0.05);

      expect(stubs.sceneManager.updateClippingPlanes).not.toHaveBeenCalled();
    });

    it('dynamic clipping toggle forwards to sceneManager + updateClippingControlsState', () => {
      setupCameraControls(stubs.context, stubs.controllersRef);
      // updateClippingControlsState is called once at init (false). Clear so we
      // can isolate the toggle's call.
      stubs.updateClippingControlsState.mockClear();

      clipCtrls()[2]._onChangeFn?.(true);

      expect(stubs.sceneManager.setDynamicClipping).toHaveBeenCalledWith(true);
      expect(stubs.updateClippingControlsState).toHaveBeenCalledWith(true);
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });
});
