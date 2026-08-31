// @vitest-environment jsdom
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
import {
  DEFAULT_MAX_PIXEL_RATIO,
  setMaxPixelRatioCap,
} from '../../../../rendering/pixel-ratio-cap';
import { setNativeDPR } from '../../../helpers/device-pixel-ratio';

interface ControllerStub {
  name: ReturnType<typeof vi.fn>;
  onChange: ReturnType<typeof vi.fn>;
  onFinishChange: ReturnType<typeof vi.fn>;
  hide: ReturnType<typeof vi.fn>;
  show: ReturnType<typeof vi.fn>;
  updateDisplay: ReturnType<typeof vi.fn>;
  // NumberController's fluent range setters. Only a number control gets
  // these in lil-gui, but the stub is shared; the Manual DPR slider is
  // retargeted through them when the Allow High DPR ceiling moves.
  min: ReturnType<typeof vi.fn>;
  max: ReturnType<typeof vi.fn>;
  step: ReturnType<typeof vi.fn>;
  domElement: HTMLElement;
  _name: string | null;
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
    min: vi.fn(),
    max: vi.fn(),
    step: vi.fn(),
    domElement: document.createElement('div'),
    _name: null,
    _onChangeFn: null,
    _onFinishChangeFn: null,
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

/**
 * Look a control up by its label rather than by position. The Performance
 * folder gained a control at the FRONT (Allow High DPR), which silently
 * re-pointed every positional index in this file at the wrong control.
 */
function byName(folder: FolderStub, label: string): ControllerStub {
  const found = folder.controllers.find((c) => c._name === label);
  if (!found) {
    throw new Error(
      `No control named "${label}" — have: ${folder.controllers.map((c) => c._name).join(', ')}`
    );
  }
  return found;
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
    isHighDPRAllowed: ReturnType<typeof vi.fn>;
    setHighDPRAllowed: ReturnType<typeof vi.fn>;
  };
  let saveSettings: ReturnType<typeof vi.fn>;
  let triggerAnimation: ReturnType<typeof vi.fn>;
  let settings: { adaptiveDPREnabled: boolean; allowHighDPR: boolean };

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
      isHighDPRAllowed: vi.fn().mockReturnValue(false),
      setHighDPRAllowed: vi.fn(),
    };
    saveSettings = vi.fn();
    triggerAnimation = vi.fn();
    settings = { adaptiveDPREnabled: false, allowHighDPR: false };
    setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
  });

  afterEach(() => {
    vi.useRealTimers();
    setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
  });

  it('creates the Performance folder + 3 controls (both toggles + manual DPR)', () => {
    setupPerformanceControls(makeContext());
    expect(gui.addFolder).toHaveBeenCalledWith('Performance', expect.any(String));
    expect(folder.controllers).toHaveLength(3);
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
      byName(folder, 'Adaptive Resolution')._onChangeFn?.(true);

      expect(manager.setEnabled).toHaveBeenCalledWith(true);
      expect(saveSettings).toHaveBeenCalled();
    });

    it('off → disable adaptive', () => {
      setupPerformanceControls(makeContext());
      byName(folder, 'Adaptive Resolution')._onChangeFn?.(false);
      expect(manager.setEnabled).toHaveBeenCalledWith(false);
    });
  });

  describe('manual DPR slider', () => {
    it('defers applying DPR until slider/input interaction is committed', () => {
      settings.adaptiveDPREnabled = false;
      setupPerformanceControls(makeContext());
      const manualCtrl = byName(folder, 'Manual DPR');

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
      const manualCtrl = byName(folder, 'Manual DPR');
      manualCtrl._onFinishChangeFn?.(1.0);
      expect(manager.setManualDPR).not.toHaveBeenCalled();
    });
  });

  describe('updateVisibility', () => {
    it('adaptive=true hides manual DPR + shows the read-only rows', () => {
      const result = setupPerformanceControls(makeContext());
      const manualCtrl = byName(folder, 'Manual DPR');
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
      const manualCtrl = byName(folder, 'Manual DPR');
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

    it("shows 'idle' when the manager reports 0 FPS (loop paused)", () => {
      settings.adaptiveDPREnabled = true;
      manager.isActive.mockReturnValue(true);
      // notifyPaused clears the FPS window, so a paused loop reads 0 —
      // the row must say so instead of freezing a stale number.
      manager.getState.mockReturnValue({ currentDPR: 1.0, currentFPS: 0 });

      setupPerformanceControls(makeContext());

      vi.advanceTimersByTime(500);

      const rows = folder.domElement.querySelectorAll(
        '.luxar-gui__children .luxar-gui__controller'
      );
      const fpsValue = rows[1].querySelector('.luxar-gui__controller-widget') as HTMLElement;
      expect(fpsValue.textContent).toBe('idle');
    });

    it.each([
      [0.4, '0.4'],
      [0.04, '0.04'],
      [0.96, '1.0'],
    ])('renders a sub-1fps rate as %f, not the "not rendering" 0', (fps, expected) => {
      // The FPS estimate is DEFINED below 1fps (the window keeps a
      // two-sample minimum), so a software-rasterized scene reports e.g.
      // 0.4 — which Math.round() turned into "0", the row's unambiguous
      // sentinel for "not rendering at all".
      settings.adaptiveDPREnabled = true;
      manager.isActive.mockReturnValue(true);
      manager.getState.mockReturnValue({ currentDPR: 0.5, currentFPS: fps });

      setupPerformanceControls(makeContext());
      vi.advanceTimersByTime(500);

      const rows = folder.domElement.querySelectorAll(
        '.luxar-gui__children .luxar-gui__controller'
      );
      const fpsValue = rows[1].querySelector('.luxar-gui__controller-widget') as HTMLElement;
      expect(fpsValue.textContent).toBe(expected);
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

  /**
   * The Manual DPR slider's RANGE was completely unasserted before this
   * block, so a regression in it was invisible. It is the top of that
   * range that makes Allow High DPR a hard ceiling rather than a hint.
   */
  describe('Allow High DPR', () => {
    it('bounds the Manual DPR slider by the ceiling, not the display DPR', () => {
      setupPerformanceControls(makeContext());

      const addCall = folder.add.mock.calls.find((c) => c[1] === 'dpr');
      expect(addCall).toBeDefined();
      // [target, prop, min, max, step] — max is the 1.0 ceiling even
      // though the mocked display reports 2.
      expect(addCall![2]).toBe(0.25);
      expect(addCall![3]).toBe(1);
      expect(addCall![4]).toBe(0.05);
    });

    it('opens the slider up to the display DPR once high DPR is allowed', () => {
      setMaxPixelRatioCap(Infinity);
      const restore = setNativeDPR(2);
      try {
        setupPerformanceControls(makeContext());
        const addCall = folder.add.mock.calls.find((c) => c[1] === 'dpr');
        expect(addCall![3]).toBe(2);
      } finally {
        restore();
      }
    });

    it('seeds the checkbox from the manager, not from the settings object', () => {
      // A `?dpr=` pin ignores both toggles, so the panel has to read back
      // what is actually in force rather than trusting stored settings.
      manager.isHighDPRAllowed.mockReturnValue(true);
      settings.allowHighDPR = false;

      setupPerformanceControls(makeContext());

      expect(settings.allowHighDPR).toBe(true);
    });

    it('forwards the toggle to the manager, persists, and repaints', () => {
      setupPerformanceControls(makeContext());

      byName(folder, 'Allow High DPR')._onChangeFn?.(true);

      expect(manager.setHighDPRAllowed).toHaveBeenCalledWith(true);
      expect(saveSettings).toHaveBeenCalled();
      expect(triggerAnimation).toHaveBeenCalled();
    });

    it('retargets the slider range when the toggle flips with the panel open', () => {
      settings.adaptiveDPREnabled = false;
      const result = setupPerformanceControls(makeContext());
      const manualCtrl = byName(folder, 'Manual DPR');
      manualCtrl.max.mockClear();

      // The manager owns the cap in production; mirror that here.
      manager.setHighDPRAllowed.mockImplementation((allowed: boolean) => {
        setMaxPixelRatioCap(allowed ? Infinity : DEFAULT_MAX_PIXEL_RATIO);
      });
      const restore = setNativeDPR(2);
      try {
        byName(folder, 'Allow High DPR')._onChangeFn?.(true);
        expect(manualCtrl.max).toHaveBeenCalledWith(2);
      } finally {
        restore();
        result.cleanup();
      }
    });
  });
});
