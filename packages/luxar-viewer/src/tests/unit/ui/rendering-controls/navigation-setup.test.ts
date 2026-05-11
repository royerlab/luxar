/**
 * Unit tests for ui/rendering-controls/navigation-setup.ts.
 *
 * Stubs the GUI/Folder + sceneManager + animationController + the two
 * setup-context callbacks, then verifies that setupNavigationControls
 * wires its 7 onChange callbacks (controlType, autoRotate,
 * autoRotateSpeed, flyMovementSpeed, flyRotationSpeed, flyInertialMode,
 * flyDamping, flyRotationDamping) to the right downstream calls.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { SetupContext } from '../../../../ui/rendering-controls/types';
import { setupNavigationControls } from '../../../../ui/rendering-controls/navigation-setup';

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  _onChangeFn: ((value: unknown) => void) | null;
}

function makeController(): ControllerStub {
  const ctrl: ControllerStub = {
    name: vi.fn(),
    onChange: vi.fn(),
    hide: vi.fn(),
    show: vi.fn(),
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
    domElement: document.createElement('div'),
    controllers,
    subFolders,
  };
  return folder;
}

interface ContextStubs {
  context: SetupContext;
  navFolder: FolderStub;
  sceneManager: {
    setControlType: ReturnType<typeof vi.fn>;
    setAutoRotate: ReturnType<typeof vi.fn>;
    setAutoRotateSpeed: ReturnType<typeof vi.fn>;
    setFlyMovementSpeed: ReturnType<typeof vi.fn>;
    setFlyRotationSpeed: ReturnType<typeof vi.fn>;
    setFlyInertialMode: ReturnType<typeof vi.fn>;
    setFlyDamping: ReturnType<typeof vi.fn>;
    setFlyRotationDamping: ReturnType<typeof vi.fn>;
  };
  animationController: { startAnimation: ReturnType<typeof vi.fn> };
  saveSettings: ReturnType<typeof vi.fn>;
  triggerAnimation: ReturnType<typeof vi.fn>;
  updateNavigationControls: ReturnType<typeof vi.fn>;
  settings: {
    controlType: 'orbit' | 'fly' | 'ortho';
    autoRotate: boolean;
    autoRotateSpeed: number;
    flyMovementSpeed: number;
    flyRotationSpeed: number;
    flyInertialMode: boolean;
    flyDamping: number;
    flyRotationDamping: number;
  };
}

function makeContext(initial: Partial<ContextStubs['settings']> = {}): ContextStubs {
  const navFolder = makeFolder();
  const gui = { addFolder: vi.fn().mockReturnValue(navFolder) };
  const sceneManager = {
    setControlType: vi.fn(),
    setAutoRotate: vi.fn(),
    setAutoRotateSpeed: vi.fn(),
    setFlyMovementSpeed: vi.fn(),
    setFlyRotationSpeed: vi.fn(),
    setFlyInertialMode: vi.fn(),
    setFlyDamping: vi.fn(),
    setFlyRotationDamping: vi.fn(),
  };
  const animationController = { startAnimation: vi.fn() };
  const saveSettings = vi.fn();
  const triggerAnimation = vi.fn();
  const updateNavigationControls = vi.fn();
  const settings = {
    controlType: 'orbit' as const,
    autoRotate: false,
    autoRotateSpeed: 0.25,
    flyMovementSpeed: 1.0,
    flyRotationSpeed: 1.0,
    flyInertialMode: true,
    flyDamping: 0.999,
    flyRotationDamping: 0.999,
    ...initial,
  };
  const context = {
    gui,
    settings,
    postProcessing: {} as unknown,
    sceneManager,
    animationController,
    saveSettings,
    triggerAnimation,
    updateClippingControlsState: vi.fn(),
    updateNavigationControls,
  } as unknown as SetupContext;
  return {
    context,
    navFolder,
    sceneManager,
    animationController,
    saveSettings,
    triggerAnimation,
    updateNavigationControls,
    settings,
  };
}

describe('setupNavigationControls', () => {
  let stubs: ContextStubs;

  beforeEach(() => {
    stubs = makeContext();
  });

  it('creates the Navigation folder + opens it', () => {
    setupNavigationControls(stubs.context);
    expect(stubs.navFolder.open).toHaveBeenCalled();
  });

  it('creates Orbit + Fly sub-folders', () => {
    setupNavigationControls(stubs.context);
    expect(stubs.navFolder.subFolders.has('Orbit Controls')).toBe(true);
    expect(stubs.navFolder.subFolders.has('Fly Controls')).toBe(true);
  });

  it('returns folder refs in result.folders', () => {
    const result = setupNavigationControls(stubs.context);
    expect(result.folders?.orbitFolder).toBeDefined();
    expect(result.folders?.flyFolder).toBeDefined();
  });

  it('returns controllers in result.controllers', () => {
    const result = setupNavigationControls(stubs.context);
    expect(result.controllers.controlType).toBeDefined();
    expect(result.controllers.autoRotate).toBeDefined();
    expect(result.controllers.flyMovementSpeed).toBeDefined();
    expect(result.controllers.flyDamping).toBeDefined();
  });

  describe('control type onChange', () => {
    it('forwards the value to sceneManager.setControlType + updateNavigationControls', () => {
      setupNavigationControls(stubs.context);
      // controlType is the first controller added to navFolder.
      stubs.navFolder.controllers[0]._onChangeFn?.('fly');

      expect(stubs.sceneManager.setControlType).toHaveBeenCalledWith('fly');
      expect(stubs.updateNavigationControls).toHaveBeenCalledWith('fly');
      expect(stubs.saveSettings).toHaveBeenCalled();
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });

  describe('orbit sub-folder controls', () => {
    it('autoRotate=true forwards to setAutoRotate + starts animation', () => {
      setupNavigationControls(stubs.context);
      const orbitFolder = stubs.navFolder.subFolders.get('Orbit Controls')!;
      orbitFolder.controllers[0]._onChangeFn?.(true);

      expect(stubs.sceneManager.setAutoRotate).toHaveBeenCalledWith(true);
      expect(stubs.animationController.startAnimation).toHaveBeenCalled();
      expect(stubs.saveSettings).toHaveBeenCalled();
    });

    it('autoRotate=false does NOT start animation', () => {
      setupNavigationControls(stubs.context);
      const orbitFolder = stubs.navFolder.subFolders.get('Orbit Controls')!;
      orbitFolder.controllers[0]._onChangeFn?.(false);

      expect(stubs.sceneManager.setAutoRotate).toHaveBeenCalledWith(false);
      expect(stubs.animationController.startAnimation).not.toHaveBeenCalled();
    });

    it('autoRotateSpeed forwards to setAutoRotateSpeed', () => {
      setupNavigationControls(stubs.context);
      const orbitFolder = stubs.navFolder.subFolders.get('Orbit Controls')!;
      orbitFolder.controllers[1]._onChangeFn?.(2.5);

      expect(stubs.sceneManager.setAutoRotateSpeed).toHaveBeenCalledWith(2.5);
      expect(stubs.triggerAnimation).toHaveBeenCalled();
    });
  });

  describe('fly sub-folder controls', () => {
    it('flyMovementSpeed forwards to setFlyMovementSpeed', () => {
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      flyFolder.controllers[0]._onChangeFn?.(2.0);

      expect(stubs.sceneManager.setFlyMovementSpeed).toHaveBeenCalledWith(2.0);
      expect(stubs.saveSettings).toHaveBeenCalled();
    });

    it('flyRotationSpeed forwards to setFlyRotationSpeed', () => {
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      flyFolder.controllers[1]._onChangeFn?.(1.5);

      expect(stubs.sceneManager.setFlyRotationSpeed).toHaveBeenCalledWith(1.5);
    });

    it('flyInertialMode=true shows the damping controls', () => {
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      const dampingCtrl = flyFolder.controllers[3]; // flyDamping
      const rotDampingCtrl = flyFolder.controllers[4]; // flyRotationDamping
      dampingCtrl.show.mockClear();
      rotDampingCtrl.show.mockClear();

      flyFolder.controllers[2]._onChangeFn?.(true);

      expect(stubs.sceneManager.setFlyInertialMode).toHaveBeenCalledWith(true);
      expect(dampingCtrl.show).toHaveBeenCalled();
      expect(rotDampingCtrl.show).toHaveBeenCalled();
    });

    it('flyInertialMode=false hides the damping controls', () => {
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      const dampingCtrl = flyFolder.controllers[3];
      const rotDampingCtrl = flyFolder.controllers[4];
      dampingCtrl.hide.mockClear();
      rotDampingCtrl.hide.mockClear();

      flyFolder.controllers[2]._onChangeFn?.(false);

      expect(stubs.sceneManager.setFlyInertialMode).toHaveBeenCalledWith(false);
      expect(dampingCtrl.hide).toHaveBeenCalled();
      expect(rotDampingCtrl.hide).toHaveBeenCalled();
    });

    it('flyDamping forwards to setFlyDamping', () => {
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      flyFolder.controllers[3]._onChangeFn?.(0.97);
      expect(stubs.sceneManager.setFlyDamping).toHaveBeenCalledWith(0.97);
    });

    it('flyRotationDamping forwards to setFlyRotationDamping', () => {
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      flyFolder.controllers[4]._onChangeFn?.(0.95);
      expect(stubs.sceneManager.setFlyRotationDamping).toHaveBeenCalledWith(0.95);
    });
  });

  describe('initial visibility', () => {
    it('calls updateNavigationControls with the initial controlType', () => {
      stubs = makeContext({ controlType: 'fly' });
      setupNavigationControls(stubs.context);
      expect(stubs.updateNavigationControls).toHaveBeenCalledWith('fly');
    });

    it('hides damping controls when flyInertialMode=false at init', () => {
      stubs = makeContext({ flyInertialMode: false });
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      // Damping controllers are #3 (flyDamping) and #4 (flyRotationDamping).
      expect(flyFolder.controllers[3].hide).toHaveBeenCalled();
      expect(flyFolder.controllers[4].hide).toHaveBeenCalled();
    });

    it('does NOT hide damping when flyInertialMode=true at init', () => {
      stubs = makeContext({ flyInertialMode: true });
      setupNavigationControls(stubs.context);
      const flyFolder = stubs.navFolder.subFolders.get('Fly Controls')!;
      expect(flyFolder.controllers[3].hide).not.toHaveBeenCalled();
      expect(flyFolder.controllers[4].hide).not.toHaveBeenCalled();
    });
  });
});
