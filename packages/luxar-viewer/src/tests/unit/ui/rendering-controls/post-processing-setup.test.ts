/**
 * Unit tests for ui/rendering-controls/post-processing-setup.ts.
 *
 * Stubs the GUI/Folder + postProcessing dependencies and verifies
 * that setupPostProcessingControls wires the 26 onChange callbacks
 * across 4 effect sub-folders (Bloom, Detector Noise,
 * Vignette, Chromatic Lens Distortion) to the right downstream
 * methods + re-saves settings + triggers a re-render.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { SetupContext, SetupResult } from '../../../../ui/rendering-controls/types';
import { setupPostProcessingControls } from '../../../../ui/rendering-controls/post-processing-setup';

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
  addFolder: ReturnType<typeof vi.fn>;
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
    close: vi.fn(),
    domElement: document.createElement('div'),
    controllers,
    subFolders,
  };
  return folder;
}

interface ContextStubs {
  context: SetupContext;
  effectsFolder: FolderStub;
  postProcessing: {
    setBloomEnabled: ReturnType<typeof vi.fn>;
    updateBloomSettings: ReturnType<typeof vi.fn>;
    setBloomLevels: ReturnType<typeof vi.fn>;
    setDetectorNoiseEnabled: ReturnType<typeof vi.fn>;
    updateDetectorNoiseSettings: ReturnType<typeof vi.fn>;
    setDOF: ReturnType<typeof vi.fn>;
    updateDOF: ReturnType<typeof vi.fn>;
    setAOEnabled: ReturnType<typeof vi.fn>;
    setVignetteEnabled: ReturnType<typeof vi.fn>;
    setChromaticLensDistortionEnabled: ReturnType<typeof vi.fn>;
    updateChromaticLensDistortion: ReturnType<typeof vi.fn>;
  };
  animationController: { startAnimation: ReturnType<typeof vi.fn> };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  settings: Record<string, unknown>;
}

function makeContext(initial: Partial<Record<string, unknown>> = {}): ContextStubs {
  const effectsFolder = makeFolder();
  const gui = { addFolder: vi.fn().mockReturnValue(effectsFolder) };
  const postProcessing = {
    setBloomEnabled: vi.fn(),
    updateBloomSettings: vi.fn(),
    setBloomLevels: vi.fn(),
    setDetectorNoiseEnabled: vi.fn(),
    updateDetectorNoiseSettings: vi.fn(),
    setDOF: vi.fn(),
    updateDOF: vi.fn(),
    setAOEnabled: vi.fn(),
    setVignetteEnabled: vi.fn(),
    setChromaticLensDistortionEnabled: vi.fn(),
    updateChromaticLensDistortion: vi.fn(),
  };
  const animationController = { startAnimation: vi.fn() };
  const saveSettings = vi.fn();
  const triggerAnimation = vi.fn();
  const settings: Record<string, unknown> = {
    bloomEnabled: false,
    bloomThreshold: 0.5,
    bloomStrength: 1.0,
    bloomRadius: 0.5,
    bloomLevels: 8,
    detectorNoiseEnabled: false,
    detectorNoiseReadoutSigma: 0.01,
    detectorNoisePhotonGain: 0.01,
    detectorNoiseFpnSigma: 0.005,
    dofEnabled: false,
    dofFocus: 10,
    dofStrength: 0.5,
    aoEnabled: false,
    aoQuality: 'medium',
    vignetteEnabled: false,
    vignetteDarkness: 0.5,
    vignetteOffset: 0.5,
    chromaticLensDistortionEnabled: false,
    chromaticLensDispersion: 0.03,
    chromaticLensDistortionX: 0,
    chromaticLensDistortionY: 0,
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
    sceneManager: {} as unknown,
    animationController,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState: vi.fn(),
    updateNavigationControls: vi.fn(),
  } as unknown as SetupContext;
  return {
    context,
    effectsFolder,
    postProcessing,
    animationController,
    saveSettings,
    triggerAnimation,
    settings,
  };
}

const noControllers: SetupResult['controllers'] = {};

describe('setupPostProcessingControls', () => {
  let stubs: ContextStubs;

  beforeEach(() => {
    stubs = makeContext();
  });

  it('creates the Post-Processing folder, closed by default, with 4 sub-folders', () => {
    setupPostProcessingControls(stubs.context, noControllers);
    expect(stubs.effectsFolder.close).toHaveBeenCalled();
    expect(stubs.effectsFolder.subFolders.has('Bloom')).toBe(true);
    expect(stubs.effectsFolder.subFolders.has('Detector Noise')).toBe(true);
    expect(stubs.effectsFolder.subFolders.has('Vignette')).toBe(true);
    expect(stubs.effectsFolder.subFolders.has('Chromatic Lens Distortion')).toBe(true);
  });

  it('returns the chromatic-lens controllers for FOV preset sync', () => {
    const result = setupPostProcessingControls(stubs.context, noControllers);
    expect(result.controllers.chromaticLensDistortionX).toBeDefined();
    expect(result.controllers.chromaticLensDistortionY).toBeDefined();
    expect(result.controllers.chromaticLensDispersion).toBeDefined();
    expect(result.controllers.chromaticLensPrincipalPointX).toBeDefined();
    expect(result.controllers.chromaticLensPrincipalPointY).toBeDefined();
    expect(result.controllers.chromaticLensFocalLengthX).toBeDefined();
    expect(result.controllers.chromaticLensFocalLengthY).toBeDefined();
    expect(result.controllers.chromaticLensSkew).toBeDefined();
  });

  describe('Bloom', () => {
    function bloomCtrls() {
      return stubs.effectsFolder.subFolders.get('Bloom')!.controllers;
    }

    it('enabled toggle forwards full settings tuple to setBloomEnabled', () => {
      stubs.settings.bloomStrength = 0.8;
      stubs.settings.bloomRadius = 0.3;
      stubs.settings.bloomThreshold = 0.7;
      setupPostProcessingControls(stubs.context, noControllers);

      bloomCtrls()[0]._onChangeFn?.(true);
      expect(stubs.postProcessing.setBloomEnabled).toHaveBeenCalledWith(true, 0.8, 0.3, 0.7);
    });

    it('threshold slider forwards via updateBloomSettings(undefined, undefined, value)', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      bloomCtrls()[1]._onChangeFn?.(0.9);
      expect(stubs.postProcessing.updateBloomSettings).toHaveBeenCalledWith(
        undefined,
        undefined,
        0.9
      );
    });

    it('strength slider forwards via updateBloomSettings(value, undefined, undefined)', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      bloomCtrls()[2]._onChangeFn?.(1.5);
      expect(stubs.postProcessing.updateBloomSettings).toHaveBeenCalledWith(
        1.5,
        undefined,
        undefined
      );
    });

    it('radius slider forwards via updateBloomSettings(undefined, value, undefined)', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      bloomCtrls()[3]._onChangeFn?.(0.6);
      expect(stubs.postProcessing.updateBloomSettings).toHaveBeenCalledWith(
        undefined,
        0.6,
        undefined
      );
    });

    it('levels slider forwards via setBloomLevels (rounded)', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      // The non-integer 7.6 hits Math.round → 8.
      bloomCtrls()[4]._onChangeFn?.(7.6);
      expect(stubs.postProcessing.setBloomLevels).toHaveBeenCalledWith(8);
    });
  });

  describe('Detector Noise', () => {
    function ctrls() {
      return stubs.effectsFolder.subFolders.get('Detector Noise')!.controllers;
    }

    it('enabled toggle forwards full settings tuple to setDetectorNoiseEnabled', () => {
      stubs.settings.detectorNoiseReadoutSigma = 0.02;
      stubs.settings.detectorNoisePhotonGain = 0.05;
      stubs.settings.detectorNoiseFpnSigma = 0.01;
      setupPostProcessingControls(stubs.context, noControllers);

      ctrls()[0]._onChangeFn?.(true);
      expect(stubs.postProcessing.setDetectorNoiseEnabled).toHaveBeenCalledWith(
        true,
        0.02,
        0.05,
        0.01
      );
    });

    it('enabled=true also starts the animation loop', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      ctrls()[0]._onChangeFn?.(true);
      expect(stubs.animationController.startAnimation).toHaveBeenCalled();
    });

    it('enabled=false does NOT start animation', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      ctrls()[0]._onChangeFn?.(false);
      expect(stubs.animationController.startAnimation).not.toHaveBeenCalled();
    });

    it('readout-sigma forwards as { readoutSigma } to updateDetectorNoiseSettings', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      ctrls()[1]._onChangeFn?.(0.04);
      expect(stubs.postProcessing.updateDetectorNoiseSettings).toHaveBeenCalledWith({
        readoutSigma: 0.04,
      });
    });

    it('photon-gain forwards as { photonGain } to updateDetectorNoiseSettings', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      ctrls()[2]._onChangeFn?.(0.02);
      expect(stubs.postProcessing.updateDetectorNoiseSettings).toHaveBeenCalledWith({
        photonGain: 0.02,
      });
    });

    it('FPN-sigma forwards as { fpnSigma } to updateDetectorNoiseSettings', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      ctrls()[3]._onChangeFn?.(0.01);
      expect(stubs.postProcessing.updateDetectorNoiseSettings).toHaveBeenCalledWith({
        fpnSigma: 0.01,
      });
    });
  });

  // Depth-of-field and Ambient Occlusion UI sections were removed in the
  // mega-shader refactor (DoF niche for scientific viz; SSAO has no
  // surface normals for point/gsplat geometry). Their UI tests dropped.

  describe('Vignette', () => {
    function ctrls() {
      return stubs.effectsFolder.subFolders.get('Vignette')!.controllers;
    }

    it('enabled toggle forwards (value, darkness, offset)', () => {
      stubs.settings.vignetteDarkness = 0.7;
      stubs.settings.vignetteOffset = 0.4;
      setupPostProcessingControls(stubs.context, noControllers);

      ctrls()[0]._onChangeFn?.(true);
      expect(stubs.postProcessing.setVignetteEnabled).toHaveBeenCalledWith(true, 0.7, 0.4);
    });

    it('darkness slider only fires when vignette is enabled', () => {
      stubs.settings.vignetteEnabled = true;
      stubs.settings.vignetteOffset = 0.5;
      setupPostProcessingControls(stubs.context, noControllers);

      ctrls()[1]._onChangeFn?.(0.8);
      expect(stubs.postProcessing.setVignetteEnabled).toHaveBeenCalledWith(true, 0.8, 0.5);
    });

    it('darkness slider is a no-op when vignette is disabled', () => {
      stubs.settings.vignetteEnabled = false;
      setupPostProcessingControls(stubs.context, noControllers);

      ctrls()[1]._onChangeFn?.(0.8);
      expect(stubs.postProcessing.setVignetteEnabled).not.toHaveBeenCalled();
    });

    it('offset slider only fires when vignette is enabled', () => {
      stubs.settings.vignetteEnabled = true;
      stubs.settings.vignetteDarkness = 0.6;
      setupPostProcessingControls(stubs.context, noControllers);

      ctrls()[2]._onChangeFn?.(0.3);
      expect(stubs.postProcessing.setVignetteEnabled).toHaveBeenCalledWith(true, 0.6, 0.3);
    });
  });

  describe('Chromatic Lens Distortion', () => {
    function ctrls() {
      return stubs.effectsFolder.subFolders.get('Chromatic Lens Distortion')!.controllers;
    }

    it('enabled toggle forwards full 9-arg tuple to setChromaticLensDistortionEnabled', () => {
      stubs.settings.chromaticLensDistortionX = 0.1;
      stubs.settings.chromaticLensDistortionY = -0.1;
      stubs.settings.chromaticLensDispersion = 0.05;
      stubs.settings.chromaticLensPrincipalPointX = 0.02;
      stubs.settings.chromaticLensPrincipalPointY = -0.02;
      stubs.settings.chromaticLensFocalLengthX = 1.2;
      stubs.settings.chromaticLensFocalLengthY = 1.1;
      stubs.settings.chromaticLensSkew = 0.01;
      setupPostProcessingControls(stubs.context, noControllers);

      ctrls()[0]._onChangeFn?.(true);
      expect(stubs.postProcessing.setChromaticLensDistortionEnabled).toHaveBeenCalledWith(
        true,
        0.1,
        -0.1,
        0.05,
        0.02,
        -0.02,
        1.2,
        1.1,
        0.01
      );
    });

    it.each([
      ['dispersion', 1, 'dispersion'],
      ['distortionX', 2, 'distortionX'],
      ['distortionY', 3, 'distortionY'],
      ['principalPointX', 4, 'principalPointX'],
      ['principalPointY', 5, 'principalPointY'],
      ['focalLengthX', 6, 'focalLengthX'],
      ['focalLengthY', 7, 'focalLengthY'],
      ['skew', 8, 'skew'],
    ])('%s slider forwards via updateChromaticLensDistortion({%s})', (_label, idx, key) => {
      setupPostProcessingControls(stubs.context, noControllers);
      ctrls()[idx as number]._onChangeFn?.(0.42);
      expect(stubs.postProcessing.updateChromaticLensDistortion).toHaveBeenCalledWith({
        [key as string]: 0.42,
      });
    });
  });

  describe('callback side effects', () => {
    it('every onChange triggers saveSettings and triggerAnimation', () => {
      setupPostProcessingControls(stubs.context, noControllers);
      const bloomCtrls = stubs.effectsFolder.subFolders.get('Bloom')!.controllers;

      bloomCtrls[1]._onChangeFn?.(0.9); // bloom threshold

      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });
});
