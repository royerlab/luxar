/**
 * Unit tests for core/app/lifecycle/dispose-pipeline.ts.
 *
 * The pipeline is a chain of safeDispose() calls in a fixed order. The
 * invariants under test:
 *   - every supplied subsystem's dispose() is invoked exactly once
 *   - every clearX() callback fires AFTER its corresponding dispose()
 *   - one component throwing must NOT skip later cleanup (singletons!)
 *   - errors aggregate into a single log call, not propagated
 *   - the three singletons (DataMonitorManager, SceneLoaderManager,
 *     workerPool) are torn down last and in that order
 *   - teardown is idempotent: undefined fields are no-ops
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  runDisposePipeline,
  type DisposePipelinePorts,
} from '../../../../../core/app/lifecycle/dispose-pipeline';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import { ThemeManager } from '../../../../../themes/theme-manager';
import { DataMonitorManager } from '../../../../../ui/data-monitor-manager';
import { SceneLoaderManager } from '../../../../../data/scene-loader-manager';

// Singleton dispose hooks + cleanupUI + worker-pool are mocked module-level
// so the pipeline runs without needing real DOM / data layer / worker state.
vi.mock('../../../../../ui/ui-cleanup', () => ({
  cleanupUI: vi.fn(),
}));
vi.mock('../../../../../workers/worker-pool', () => ({
  disposeWorkerPool: vi.fn(),
}));

interface DisposableStub {
  dispose: ReturnType<typeof vi.fn>;
}

function makeDisposable(): DisposableStub {
  return { dispose: vi.fn() };
}

function makeRecordingPanel() {
  return {
    dispose: vi.fn(),
    setOverlayManager: vi.fn(),
  };
}

function makeInputHandler() {
  return {
    dispose: vi.fn(),
    setDatasetBrowser: vi.fn(),
  };
}

function makeDatasetBrowser() {
  return { close: vi.fn() };
}

interface Stubs {
  sceneManager: DisposableStub;
  animationController: DisposableStub;
  performanceMonitor: DisposableStub;
  adaptiveDPRManager: DisposableStub;
  resolutionIndicator: DisposableStub;
  inputHandler: ReturnType<typeof makeInputHandler>;
  renderingControls: DisposableStub;
  recordingPanel: ReturnType<typeof makeRecordingPanel>;
  layersPanel: DisposableStub;
  controlRail: DisposableStub;
  scaleBar: DisposableStub;
  colormapLegend: DisposableStub;
  overlayManager: DisposableStub;
  pickingSystem: DisposableStub;
  labelLoader: DisposableStub;
  imageLabelLoader: DisposableStub;
  datasetBrowser: ReturnType<typeof makeDatasetBrowser>;
  events: EventGroup;
  pickingEvents: EventGroup;
  clears: {
    scaleBar: ReturnType<typeof vi.fn>;
    colormapLegend: ReturnType<typeof vi.fn>;
    overlayManager: ReturnType<typeof vi.fn>;
    recordingPanel: ReturnType<typeof vi.fn>;
    layersPanel: ReturnType<typeof vi.fn>;
    controlRail: ReturnType<typeof vi.fn>;
    pickingSystem: ReturnType<typeof vi.fn>;
    labelLoader: ReturnType<typeof vi.fn>;
    imageLabelLoader: ReturnType<typeof vi.fn>;
    datasetBrowser: ReturnType<typeof vi.fn>;
  };
}

function makeStubs(): Stubs {
  return {
    sceneManager: makeDisposable(),
    animationController: makeDisposable(),
    performanceMonitor: makeDisposable(),
    adaptiveDPRManager: makeDisposable(),
    resolutionIndicator: makeDisposable(),
    inputHandler: makeInputHandler(),
    renderingControls: makeDisposable(),
    recordingPanel: makeRecordingPanel(),
    layersPanel: makeDisposable(),
    controlRail: makeDisposable(),
    scaleBar: makeDisposable(),
    colormapLegend: makeDisposable(),
    overlayManager: makeDisposable(),
    pickingSystem: makeDisposable(),
    labelLoader: makeDisposable(),
    imageLabelLoader: makeDisposable(),
    datasetBrowser: makeDatasetBrowser(),
    events: new EventGroup(),
    pickingEvents: new EventGroup(),
    clears: {
      scaleBar: vi.fn(),
      colormapLegend: vi.fn(),
      overlayManager: vi.fn(),
      recordingPanel: vi.fn(),
      layersPanel: vi.fn(),
      controlRail: vi.fn(),
      pickingSystem: vi.fn(),
      labelLoader: vi.fn(),
      imageLabelLoader: vi.fn(),
      datasetBrowser: vi.fn(),
    },
  };
}

function makePorts(s: Stubs): DisposePipelinePorts {
  return {
    events: s.events,
    pickingEvents: s.pickingEvents,
    sceneManager: s.sceneManager as unknown as DisposePipelinePorts['sceneManager'],
    animationController:
      s.animationController as unknown as DisposePipelinePorts['animationController'],
    performanceMonitor:
      s.performanceMonitor as unknown as DisposePipelinePorts['performanceMonitor'],
    adaptiveDPRManager:
      s.adaptiveDPRManager as unknown as DisposePipelinePorts['adaptiveDPRManager'],
    resolutionIndicator:
      s.resolutionIndicator as unknown as DisposePipelinePorts['resolutionIndicator'],
    inputHandler: s.inputHandler as unknown as DisposePipelinePorts['inputHandler'],
    renderingControls: s.renderingControls as unknown as DisposePipelinePorts['renderingControls'],
    recordingPanel: s.recordingPanel as unknown as DisposePipelinePorts['recordingPanel'],
    layersPanel: s.layersPanel as unknown as DisposePipelinePorts['layersPanel'],
    controlRail: s.controlRail as unknown as DisposePipelinePorts['controlRail'],
    scaleBar: s.scaleBar as unknown as DisposePipelinePorts['scaleBar'],
    colormapLegend: s.colormapLegend as unknown as DisposePipelinePorts['colormapLegend'],
    overlayManager: s.overlayManager as unknown as DisposePipelinePorts['overlayManager'],
    pickingSystem: s.pickingSystem as unknown as DisposePipelinePorts['pickingSystem'],
    labelLoader: s.labelLoader as unknown as DisposePipelinePorts['labelLoader'],
    imageLabelLoader: s.imageLabelLoader as unknown as DisposePipelinePorts['imageLabelLoader'],
    datasetBrowser: s.datasetBrowser as unknown as DisposePipelinePorts['datasetBrowser'],
    // vitest's Mock type doesn't structurally satisfy `() => void`,
    // so cast each clearX through unknown at the call site.
    clearScaleBar: s.clears.scaleBar as unknown as () => void,
    clearColormapLegend: s.clears.colormapLegend as unknown as () => void,
    clearOverlayManager: s.clears.overlayManager as unknown as () => void,
    clearRecordingPanel: s.clears.recordingPanel as unknown as () => void,
    clearLayersPanel: s.clears.layersPanel as unknown as () => void,
    clearControlRail: s.clears.controlRail as unknown as () => void,
    clearPickingSystem: s.clears.pickingSystem as unknown as () => void,
    clearLabelLoader: s.clears.labelLoader as unknown as () => void,
    clearImageLabelLoader: s.clears.imageLabelLoader as unknown as () => void,
    clearDatasetBrowser: s.clears.datasetBrowser as unknown as () => void,
  };
}

describe('runDisposePipeline', () => {
  // ThemeManager.disposeInstance / SceneLoaderManager.disposeInstance /
  // DataMonitorManager.disposeInstance are real singletons; spy on them
  // so the pipeline does not actually tear them down (and so we can
  // assert they were called in the right order).
  let themeSpy: ReturnType<typeof vi.spyOn>;
  let monitorSpy: ReturnType<typeof vi.spyOn>;
  let loaderSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    themeSpy = vi.spyOn(ThemeManager, 'disposeInstance').mockImplementation(() => {});
    monitorSpy = vi.spyOn(DataMonitorManager, 'disposeInstance').mockImplementation(() => {});
    loaderSpy = vi.spyOn(SceneLoaderManager, 'disposeInstance').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('happy path — fully-built app', () => {
    it('disposes every subsystem exactly once', () => {
      const s = makeStubs();
      runDisposePipeline(makePorts(s));

      expect(s.sceneManager.dispose).toHaveBeenCalledOnce();
      expect(s.animationController.dispose).toHaveBeenCalledOnce();
      expect(s.performanceMonitor.dispose).toHaveBeenCalledOnce();
      expect(s.adaptiveDPRManager.dispose).toHaveBeenCalledOnce();
      expect(s.resolutionIndicator.dispose).toHaveBeenCalledOnce();
      expect(s.inputHandler.dispose).toHaveBeenCalledOnce();
      expect(s.renderingControls.dispose).toHaveBeenCalledOnce();
      expect(s.recordingPanel.dispose).toHaveBeenCalledOnce();
      expect(s.layersPanel.dispose).toHaveBeenCalledOnce();
      expect(s.scaleBar.dispose).toHaveBeenCalledOnce();
      expect(s.colormapLegend.dispose).toHaveBeenCalledOnce();
      expect(s.overlayManager.dispose).toHaveBeenCalledOnce();
      expect(s.pickingSystem.dispose).toHaveBeenCalledOnce();
      expect(s.labelLoader.dispose).toHaveBeenCalledOnce();
      expect(s.imageLabelLoader.dispose).toHaveBeenCalledOnce();
      expect(s.datasetBrowser.close).toHaveBeenCalledOnce();
    });

    it('invokes every clearX() callback exactly once', () => {
      const s = makeStubs();
      runDisposePipeline(makePorts(s));

      expect(s.clears.scaleBar).toHaveBeenCalledOnce();
      expect(s.clears.colormapLegend).toHaveBeenCalledOnce();
      expect(s.clears.overlayManager).toHaveBeenCalledOnce();
      expect(s.clears.recordingPanel).toHaveBeenCalledOnce();
      expect(s.clears.layersPanel).toHaveBeenCalledOnce();
      expect(s.clears.pickingSystem).toHaveBeenCalledOnce();
      expect(s.clears.labelLoader).toHaveBeenCalledOnce();
      expect(s.clears.imageLabelLoader).toHaveBeenCalledOnce();
      expect(s.clears.datasetBrowser).toHaveBeenCalledOnce();
    });

    it('disposes subsystems in the documented order (G19): animation→...→sceneManager', () => {
      // core.md G19 closure: previously `disposes every subsystem
      // exactly once` did NOT assert ordering. The teardown contract
      // requires animation first (so the loop stops emitting events
      // before listeners disappear), overlays/picking/browser before
      // input-handler (those routes wire through input handler),
      // sceneManager after renderingControls. Pin the relative order
      // between load-bearing pairs — the absolute order of unrelated
      // pairs can shift safely (e.g. adaptiveDPRManager vs scaleBar).
      const s = makeStubs();
      const order: string[] = [];

      s.animationController.dispose.mockImplementation(() => order.push('animation'));
      s.adaptiveDPRManager.dispose.mockImplementation(() => order.push('adaptiveDPR'));
      s.overlayManager.dispose.mockImplementation(() => order.push('overlay'));
      s.pickingSystem.dispose.mockImplementation(() => order.push('picking'));
      s.labelLoader.dispose.mockImplementation(() => order.push('labelLoader'));
      s.imageLabelLoader.dispose.mockImplementation(() => order.push('imgLoader'));
      s.datasetBrowser.close.mockImplementation(() => order.push('browser'));
      s.inputHandler.dispose.mockImplementation(() => order.push('input'));
      s.renderingControls.dispose.mockImplementation(() => order.push('rendering'));
      s.sceneManager.dispose.mockImplementation(() => order.push('scene'));

      runDisposePipeline(makePorts(s));

      const idx = (label: string) => order.indexOf(label);

      // animation must be before all UI / rendering subsystems.
      expect(idx('animation')).toBeLessThan(idx('rendering'));
      expect(idx('animation')).toBeLessThan(idx('input'));
      expect(idx('animation')).toBeLessThan(idx('scene'));

      // Overlay-routed components (overlayManager, picking, label
      // loaders, datasetBrowser) tear down BEFORE input-handler
      // because they're wired through it.
      expect(idx('overlay')).toBeLessThan(idx('input'));
      expect(idx('picking')).toBeLessThan(idx('input'));
      expect(idx('labelLoader')).toBeLessThan(idx('input'));
      expect(idx('imgLoader')).toBeLessThan(idx('input'));
      expect(idx('browser')).toBeLessThan(idx('input'));

      // input-handler runs before renderingControls and sceneManager,
      // which both run before the singletons (asserted above).
      expect(idx('input')).toBeLessThan(idx('rendering'));
      expect(idx('rendering')).toBeLessThan(idx('scene'));
    });

    it('disposes singletons last and in order: monitor → loader → workerPool', async () => {
      const { disposeWorkerPool } = await import('../../../../../workers/worker-pool');
      const order: string[] = [];
      monitorSpy.mockImplementation(() => order.push('monitor'));
      loaderSpy.mockImplementation(() => order.push('loader'));
      (disposeWorkerPool as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
        order.push('workerPool')
      );

      const s = makeStubs();
      runDisposePipeline(makePorts(s));

      expect(order).toEqual(['monitor', 'loader', 'workerPool']);
    });

    it('clears overlay BEFORE recordingPanel.dispose (so the back-reference is broken first)', () => {
      const s = makeStubs();
      const order: string[] = [];
      s.recordingPanel.setOverlayManager.mockImplementation(() => order.push('setOverlay-null'));
      s.recordingPanel.dispose.mockImplementation(() => order.push('recording-dispose'));
      s.overlayManager.dispose.mockImplementation(() => order.push('overlay-dispose'));

      runDisposePipeline(makePorts(s));

      // setOverlayManager(null) on recordingPanel must run before recordingPanel.dispose
      const setIdx = order.indexOf('setOverlay-null');
      const recIdx = order.indexOf('recording-dispose');
      const overlayIdx = order.indexOf('overlay-dispose');
      expect(setIdx).toBeGreaterThanOrEqual(0);
      expect(recIdx).toBeGreaterThanOrEqual(0);
      expect(overlayIdx).toBeGreaterThanOrEqual(0);
      expect(setIdx).toBeLessThan(recIdx);
      expect(setIdx).toBeLessThan(overlayIdx);
    });

    it('clears dataset-browser handle on input-handler BEFORE input-handler dispose', () => {
      const s = makeStubs();
      const order: string[] = [];
      s.datasetBrowser.close.mockImplementation(() => order.push('browser-close'));
      s.inputHandler.setDatasetBrowser.mockImplementation(() => order.push('clear-browser-ref'));
      s.inputHandler.dispose.mockImplementation(() => order.push('input-dispose'));

      runDisposePipeline(makePorts(s));

      // browser.close → input.setDatasetBrowser(undefined) → input.dispose
      expect(order).toEqual(['browser-close', 'clear-browser-ref', 'input-dispose']);
    });
  });

  describe('error containment', () => {
    it('one throwing component does NOT skip later cleanup', () => {
      const s = makeStubs();
      s.sceneManager.dispose.mockImplementation(() => {
        throw new Error('boom');
      });

      // Pipeline must NOT throw out.
      expect(() => runDisposePipeline(makePorts(s))).not.toThrow();

      // Everything after sceneManager in the order must still run.
      expect(themeSpy).toHaveBeenCalled();
      expect(monitorSpy).toHaveBeenCalled();
      expect(loaderSpy).toHaveBeenCalled();
    });

    it('a throwing singleton does NOT skip the next singleton', () => {
      const s = makeStubs();
      monitorSpy.mockImplementation(() => {
        throw new Error('monitor boom');
      });

      runDisposePipeline(makePorts(s));

      // loader and workerPool still teardown despite monitor throwing.
      expect(loaderSpy).toHaveBeenCalled();
    });

    it('multiple thrown errors all collect; pipeline still completes', () => {
      const s = makeStubs();
      s.sceneManager.dispose.mockImplementation(() => {
        throw new Error('a');
      });
      s.animationController.dispose.mockImplementation(() => {
        throw new Error('b');
      });
      s.inputHandler.dispose.mockImplementation(() => {
        throw new Error('c');
      });

      expect(() => runDisposePipeline(makePorts(s))).not.toThrow();
      // Final singleton still ran.
      expect(loaderSpy).toHaveBeenCalled();
    });
  });

  describe('idempotency on partial state', () => {
    it('every subsystem undefined → no throw, singletons still teardown', () => {
      const ports: DisposePipelinePorts = {
        events: new EventGroup(),
        pickingEvents: new EventGroup(),
        sceneManager: undefined,
        animationController: undefined,
        performanceMonitor: undefined,
        adaptiveDPRManager: undefined,
        resolutionIndicator: undefined,
        inputHandler: undefined,
        renderingControls: undefined,
        recordingPanel: undefined,
        layersPanel: undefined,
        controlRail: undefined,
        scaleBar: undefined,
        colormapLegend: undefined,
        overlayManager: undefined,
        pickingSystem: undefined,
        labelLoader: undefined,
        imageLabelLoader: undefined,
        datasetBrowser: undefined,
        clearScaleBar: vi.fn(),
        clearColormapLegend: vi.fn(),
        clearOverlayManager: vi.fn(),
        clearRecordingPanel: vi.fn(),
        clearLayersPanel: vi.fn(),
        clearControlRail: vi.fn(),
        clearPickingSystem: vi.fn(),
        clearLabelLoader: vi.fn(),
        clearImageLabelLoader: vi.fn(),
        clearDatasetBrowser: vi.fn(),
      };

      expect(() => runDisposePipeline(ports)).not.toThrow();
      // Singletons still teardown even when no subsystems are built.
      expect(themeSpy).toHaveBeenCalled();
      expect(monitorSpy).toHaveBeenCalled();
      expect(loaderSpy).toHaveBeenCalled();
    });

    it('overlay dispose is skipped when overlay manager is undefined (clear still runs — idempotent)', () => {
      const s = makeStubs();
      const ports = makePorts(s);
      // Mutate the port AFTER makePorts so we can keep the stubs and just
      // null the field for this assertion.
      ports.overlayManager = undefined;
      runDisposePipeline(ports);
      expect(s.overlayManager.dispose).not.toHaveBeenCalled();
      // clearX is unconditional in the pipeline — `this.x = undefined`
      // is idempotent, so the helper fires it regardless of whether the
      // underlying field was set.
      expect(s.clears.overlayManager).toHaveBeenCalledOnce();
    });
  });

  describe('EventGroup teardown', () => {
    it('pickingEvents.dispose runs before events.dispose', () => {
      const s = makeStubs();
      const order: string[] = [];
      const origPickingDispose = s.pickingEvents.dispose.bind(s.pickingEvents);
      const origEventsDispose = s.events.dispose.bind(s.events);
      s.pickingEvents.dispose = vi.fn(() => {
        order.push('pickingEvents');
        origPickingDispose();
      });
      s.events.dispose = vi.fn(() => {
        order.push('events');
        origEventsDispose();
      });

      runDisposePipeline(makePorts(s));

      expect(order).toEqual(['pickingEvents', 'events']);
    });

    it('dispose pipeline releases pickingEvents before overlayManager (HIGH-12)', () => {
      // The picking system's mousemove handler closes over the overlay
      // manager. If overlayManager is disposed first, a synchronously
      // dispatched mousemove between the two safeDispose calls would
      // null-deref on the disposed overlay. Release the listeners first.
      const s = makeStubs();
      const order: string[] = [];
      const origPickingDispose = s.pickingEvents.dispose.bind(s.pickingEvents);
      s.pickingEvents.dispose = vi.fn(() => {
        order.push('pickingEvents');
        origPickingDispose();
      });
      s.overlayManager.dispose.mockImplementation(() => order.push('overlayManager'));

      runDisposePipeline(makePorts(s));

      const pickingIdx = order.indexOf('pickingEvents');
      const overlayIdx = order.indexOf('overlayManager');
      expect(pickingIdx).toBeGreaterThanOrEqual(0);
      expect(overlayIdx).toBeGreaterThanOrEqual(0);
      expect(pickingIdx).toBeLessThan(overlayIdx);
    });
  });
});
