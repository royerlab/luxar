/**
 * Unit tests for ui/rendering-controls/setup/performance-setup.ts.
 *
 * Stubs the GUI/Folder + AdaptiveDPRManager dependencies and verifies
 * that setupPerformanceControls wires the adaptive toggle, manual DPR
 * slider, and the periodic Current DPR / Current FPS display rows.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  setupPerformanceControls,
  type PerformanceSetupContext,
} from '../../../../ui/rendering-controls/setup/performance-setup';

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  onFinishChange: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  updateDisplay: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  _onChangeFn: ((value: unknown) => void) | null;
  _onFinishChangeFn: ((value: unknown) => void) | null;
}

function makeController(): ControllerStub {
  const ctrl: ControllerStub = {
    name: vi.fn(),
    onChange: vi.fn(),
    onFinishChange: vi.fn(),
    hide: vi.fn(),
    show: vi.fn(),
    updateDisplay: vi.fn(),
    domElement: document.createElement('div'),
    _onChangeFn: null,
    _onFinishChangeFn: null,
  };
  ctrl.name.mockReturnValue(ctrl);
  ctrl.onChange.mockImplementation((fn: (v: unknown) => void) => {
    ctrl._onChangeFn = fn;
    return ctrl;
  });
  ctrl.onFinishChange.mockImplementation((fn: (v: unknown) => void) => {
    ctrl._onFinishChangeFn = fn;
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
  const root = document.createElement('div');
  // setupPerformanceControls appends DPR/FPS rows into a child element with
  // class luxar-gui__children — provide one so that branch is exercised.
  const childrenContainer = document.createElement('div');
  childrenContainer.className = 'luxar-gui__children';
  root.appendChild(childrenContainer);

  return {
    add: vi.fn().mockImplementation(() => {
      const c = makeController();
      controllers.push(c);
      return c;
    }),
    open: vi.fn(),
    domElement: root,
    controllers,
  };
}

describe('setupPerformanceControls', () => {
  let folder: FolderStub;
  let gui: { addFolder: ReturnType<typeof vi.fn> };
  let manager: {
    getNativeDPR: ReturnType<typeof vi.fn>;
    setManualDPR: ReturnType<typeof vi.fn>;
    setEnabled: ReturnType<typeof vi.fn>;
    isActive: ReturnType<typeof vi.fn>;
    getCurrentDPR: ReturnType<typeof vi.fn>;
    getState: ReturnType<typeof vi.fn>;
  };
  let saveSettings: ReturnType<typeof vi.fn>;
  let triggerAnimation: ReturnType<typeof vi.fn>;
  let settings: { adaptiveDPREnabled: boolean };

  function makeContext(): PerformanceSetupContext {
    return {
      gui,
      settings,
      manager,
      saveSettings,
      triggerAnimation,
    } as unknown as PerformanceSetupContext;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    folder = makeFolder();
    gui = { addFolder: vi.fn().mockReturnValue(folder) };
    manager = {
      getNativeDPR: vi.fn().mockReturnValue(2),
      setManualDPR: vi.fn(),
      setEnabled: vi.fn(),
      isActive: vi.fn().mockReturnValue(false),
      getCurrentDPR: vi.fn().mockReturnValue(1.5),
      getState: vi.fn().mockReturnValue({ currentDPR: 1.0, currentFPS: 55 }),
    };
    saveSettings = vi.fn();
    triggerAnimation = vi.fn();
    settings = { adaptiveDPREnabled: false };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('creates the Performance folder + 2 controls (toggle + manual DPR)', () => {
    setupPerformanceControls(makeContext());
    expect(gui.addFolder).toHaveBeenCalledWith('Performance', expect.any(String));
    expect(folder.controllers).toHaveLength(2);
  });

  it('opens the folder by default (shown directly in the Performance rail popover)', () => {
    setupPerformanceControls(makeContext());
    expect(folder.open).toHaveBeenCalled();
  });

  it('appends DPR + FPS read-only rows into the children container', () => {
    setupPerformanceControls(makeContext());

    const childrenContainer = folder.domElement.querySelector(
      '.luxar-gui__children'
    ) as HTMLElement;
    const rows = childrenContainer.querySelectorAll('.luxar-gui__controller');
    expect(rows.length).toBe(2);

    const labels = Array.from(rows).map(
      (r) => (r.querySelector('.luxar-gui__controller-name') as HTMLElement).textContent
    );
    expect(labels).toEqual(['Current DPR', 'Current FPS']);
  });

  it('returns adaptiveDPREnabled controller, updateVisibility, cleanup', () => {
    const result = setupPerformanceControls(makeContext());
    expect(result.adaptiveDPREnabled).toBeDefined();
    expect(typeof result.updateVisibility).toBe('function');
    expect(typeof result.cleanup).toBe('function');
  });

  it('seeds settings.adaptiveDPREnabled from manager.isActive()', () => {
    manager.isActive.mockReturnValue(true);
    setupPerformanceControls(makeContext());
    expect(settings.adaptiveDPREnabled).toBe(true);
  });

  describe('adaptive toggle onChange', () => {
    it('on → enable adaptive, save, log', () => {
      setupPerformanceControls(makeContext());
      // The toggle is the first controller added to the folder. The
      // production code attaches its onChange via .onChange(...) after
      // initial-state sync; the stub captures the latest callback in
      // `_onChangeFn`.
      folder.controllers[0]._onChangeFn?.(true);

      expect(manager.setEnabled).toHaveBeenCalledWith(true);
      expect(saveSettings).toHaveBeenCalled();
    });

    it('off → disable adaptive', () => {
      setupPerformanceControls(makeContext());
      folder.controllers[0]._onChangeFn?.(false);
      expect(manager.setEnabled).toHaveBeenCalledWith(false);
    });
  });

  describe('manual DPR slider', () => {
    it('defers applying DPR until slider/input interaction is committed', () => {
      settings.adaptiveDPREnabled = false;
      setupPerformanceControls(makeContext());
      const manualCtrl = folder.controllers[1];

      manualCtrl._onChangeFn?.(1.0);
      expect(manager.setManualDPR).not.toHaveBeenCalled();

      manualCtrl._onFinishChangeFn?.(1.0);
      expect(manager.setManualDPR).toHaveBeenCalledWith(1.0);
      expect(triggerAnimation).toHaveBeenCalled();
    });

    it('does NOT forward when adaptive is ON (manual is hidden)', () => {
      // settings.adaptiveDPREnabled is overwritten from manager.isActive()
      // during setup, so flip the manager flag rather than the settings.
      manager.isActive.mockReturnValue(true);
      setupPerformanceControls(makeContext());
      const manualCtrl = folder.controllers[1];
      manualCtrl._onFinishChangeFn?.(1.0);
      expect(manager.setManualDPR).not.toHaveBeenCalled();
    });
  });

  describe('updateVisibility', () => {
    it('adaptive=true hides manual DPR + shows the read-only rows', () => {
      const result = setupPerformanceControls(makeContext());
      const manualCtrl = folder.controllers[1];
      manualCtrl.hide.mockClear();

      result.updateVisibility(true);

      expect(manualCtrl.hide).toHaveBeenCalled();
      const rows = folder.domElement.querySelectorAll(
        '.luxar-gui__children .luxar-gui__controller'
      );
      expect((rows[0] as HTMLElement).style.display).toBe('');
      expect((rows[1] as HTMLElement).style.display).toBe('');
    });

    it('adaptive=false shows manual DPR + hides the read-only rows', () => {
      const result = setupPerformanceControls(makeContext());
      const manualCtrl = folder.controllers[1];
      manualCtrl.show.mockClear();

      result.updateVisibility(false);

      expect(manualCtrl.show).toHaveBeenCalled();
      expect(manualCtrl.updateDisplay).toHaveBeenCalled();
      const rows = folder.domElement.querySelectorAll(
        '.luxar-gui__children .luxar-gui__controller'
      );
      expect((rows[0] as HTMLElement).style.display).toBe('none');
      expect((rows[1] as HTMLElement).style.display).toBe('none');
    });
  });

  describe('periodic interval', () => {
    it('updates the DPR + FPS row text every 500ms when adaptive is ON', () => {
      settings.adaptiveDPREnabled = true;
      manager.isActive.mockReturnValue(true);
      manager.getState.mockReturnValue({ currentDPR: 0.75, currentFPS: 47 });

      setupPerformanceControls(makeContext());

      vi.advanceTimersByTime(500);

      const rows = folder.domElement.querySelectorAll(
        '.luxar-gui__children .luxar-gui__controller'
      );
      const dprValue = rows[0].querySelector('.luxar-gui__controller-widget') as HTMLElement;
      const fpsValue = rows[1].querySelector('.luxar-gui__controller-widget') as HTMLElement;
      expect(dprValue.textContent).toBe('0.75');
      expect(fpsValue.textContent).toBe('47');
    });

    it('cleanup() stops the interval', () => {
      settings.adaptiveDPREnabled = true;
      manager.isActive.mockReturnValue(true);
      const result = setupPerformanceControls(makeContext());

      result.cleanup();

      // After cleanup the values should not change on subsequent ticks.
      manager.getState.mockReturnValue({ currentDPR: 0.5, currentFPS: 30 });
      vi.advanceTimersByTime(500);

      const rows = folder.domElement.querySelectorAll(
        '.luxar-gui__children .luxar-gui__controller'
      );
      const dprValue = rows[0].querySelector('.luxar-gui__controller-widget') as HTMLElement;
      // The initial sync via updateVisibility uses {currentDPR: 1.0, currentFPS: 55}.
      // After cleanup no further updates fire, so the value stays at 1.00.
      expect(dprValue.textContent).toBe('1.00');
    });
  });
});
