// Recording panel for screenshot and video capture of the WebGL canvas.
//
// Thin coordinator over a RecordingSession (shared scaffolding) and
// three CaptureStrategy implementations (Screenshot / Video / Offline).
// Owns the GUI and the mode-dispatch logic; capture pipelines live in
// the strategies.

import GUI, { type Controller } from './gui';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import type { SceneManager } from '../scene/scene-manager';
import type { AnimationController } from '../scene/animation/animation-controller';
import type { DimensionAnimationManager } from '../scene/animation/dimension-animation-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import type { OverlayManager } from './overlay-manager';
import { generateFilename as generateFilenamePure } from './recording-panel/media-utilities';
import {
  renderFrameToCanvas as renderFrameToCanvasHelper,
  downloadBlob as downloadBlobHelper,
} from './recording-panel/screenshot-exporter';
import {
  getTurntableInfo as getTurntableInfoHelper,
  getNavigableDimensionOptions as getNavigableDimensionOptionsHelper,
} from './recording-panel/animation-sync';
import {
  computeControlVisibility,
  FORMAT_LABEL_TO_VALUE,
  CODEC_LABEL_TO_VALUE,
} from './recording-panel/gui-builder';
import { buildRecordingGUI, captureLabelForMode } from './recording-panel/ui/gui-construction';
import { RecordingSession } from './recording-panel/session';
import { ScreenshotStrategy } from './recording-panel/screenshot-strategy';
import { VideoRecordingStrategy } from './recording-panel/video-recording-strategy';
import { OfflineCaptureStrategy } from './recording-panel/offline-capture-strategy';
import { showToast } from './toast';

/**
 * Re-export of the shared recording types (defined in `recording-panel/types.ts`)
 * so external consumers can import them directly from the panel module rather
 * than reaching into its subdirectory.
 */
export type {
  RecordingMode,
  VideoResolution,
  OutputFormat,
  VideoCodecOption,
  VideoQuality,
  PanelStates,
  RecordingOptions,
} from './recording-panel/types';
import type {
  RecordingMode,
  OutputFormat,
  PanelStates,
  RecordingOptions,
} from './recording-panel/types';

/**
 * Recording panel for capturing screenshots and recording video.
 *
 * The Panel is a thin coordinator: GUI + mode dispatch only. All capture
 * pipelines (screenshot, real-time video, frame-by-frame offline) live
 * in CaptureStrategy implementations that share state via RecordingSession.
 */
export class RecordingPanel {
  private gui: GUI;
  private visible: boolean = false;
  private mode: RecordingMode = 'video';
  private options: RecordingOptions = {
    outputFormat: 'webp',
    imageQuality: 0.92,
    // Null follows the live on-screen ceiling until the user moves the
    // Capture DPR slider.
    captureDPR: null,
    transparentBackground: false,
    videoDurationLimit: 60,
    videoFPS: 30,
    videoCodec: 'h265',
    videoQuality: 'high',
    videoResolution: 0,
    syncToSlider: false,
    syncDimensionIndex: -1,
    turntableSpeed: 36,
    frameByFrame: true,
    showPanels: false,
    includeOverlays: true,
  };

  // Collaborators — non-private so tests can probe internal state via
  // `(panel as any).session.X`, `(panel as any).videoRecordingStrategy.Y`, etc.
  readonly session: RecordingSession;
  readonly screenshotStrategy: ScreenshotStrategy;
  readonly videoRecordingStrategy: VideoRecordingStrategy;
  readonly offlineCaptureStrategy: OfflineCaptureStrategy;

  // GUI controller references for dynamic show/hide
  private imageControllers: Controller[] = [];
  private videoControllers: Controller[] = [];
  private turntableControllers: Controller[] = [];
  private qualityController: Controller | null = null;
  private transparentController: Controller | null = null;
  private syncToggleController: Controller | null = null;
  private syncDimensionController: Controller | null = null;
  private videoCodecController: Controller | null = null;
  private videoQualityController: Controller | null = null;
  private videoDurationController: Controller | null = null;
  private formatController: Controller | null = null;
  private captureController: Controller | null = null;
  private refreshCaptureDPR: (() => void) | null = null;

  /** LOD-quiescence predicate for the offline loop — see {@link setLODSettledProvider}. */
  private lodSettledProvider: (() => boolean | null) | null = null;

  constructor(
    private readonly sceneManager: SceneManager,
    animationController: AnimationController
  ) {
    this.session = new RecordingSession(sceneManager, animationController, {
      isExrSequenceActive: () => this.session.isEXRSequenceRecording,
    });
    // Wire the indicator's click-to-stop and Escape-to-stop into our
    // existing stopVideoRecording path so the user can stop from anywhere.
    this.session.setStopVideoCallback(() => this.stopVideoRecording());

    // Hooks shared by all three strategies. Pure helpers are referenced
    // directly (no Panel-internal indirection).
    const hideAllPanels = (): void => this.hideAllPanels();
    const renderFrameToCanvas = (): Promise<HTMLCanvasElement> =>
      renderFrameToCanvasHelper(
        this.sceneManager.postProcessing,
        this.options.includeOverlays,
        this.session.overlayManager,
        this.sceneManager.renderer.domElement
      );

    // downloadBlob and generateFilename are routed through Panel methods
    // (not directly to the helpers) so test spies on Panel still intercept.
    const downloadBlob = (blob: Blob, filename: string): void => this.downloadBlob(blob, filename);
    const generateFilename = (ext: string): string => this.generateFilename(ext);

    this.screenshotStrategy = new ScreenshotStrategy(sceneManager, {
      hideAllPanels,
      downloadBlob,
      generateFilename,
    });

    this.videoRecordingStrategy = new VideoRecordingStrategy(sceneManager, animationController, {
      hideAllPanels,
      downloadBlob,
      generateFilename,
    });

    this.offlineCaptureStrategy = new OfflineCaptureStrategy(sceneManager, animationController, {
      hideAllPanels,
      renderFrameToCanvas,
      downloadBlob,
      generateFilename,
      // Let the offline loop wait (bounded) for the auto-LOD selector to
      // settle before exporting each frame, so a tile reloading its fine
      // level after re-entering the frustum is not filmed coarse (#1695).
      // Read through the injected provider (see setLODSettledProvider). No
      // provider ⇒ `null`, NOT `true`: the loop distinguishes "settled" (which
      // still costs it a mandatory selector-catch-up rAF per frame) from
      // "there is no lod_group here to wait for", and only the latter is free.
      // Answering `true` would charge every provider-less capture an extra
      // frame for nothing.
      isLODSettled: () => this.lodSettledProvider?.() ?? null,
    });

    this.gui = new GUI({
      title: 'Recording',
      width: 280,
      closeFolders: false,
      onClose: () => this.hide(),
      closeButtonTitle: 'Close (T)',
    });

    this.gui.domElement.classList.add('luxar-recording-panel');

    Object.assign(this.gui.domElement.style, {
      position: 'fixed',
      top: 'auto',
      bottom: 'calc(20px + env(safe-area-inset-bottom, 0px))',
      left: '20px',
      zIndex: String(config.ui.zIndex.recordingPanel),
    });

    this.gui.hide();
    this.buildGUI();
  }

  // ========== Public API ==========

  show(): void {
    this.refreshCaptureDPR?.();
    this.gui.show();
    this.visible = true;
  }

  hide(): void {
    const activeElement = document.activeElement as HTMLElement;
    if (activeElement?.blur) activeElement.blur();
    this.gui.hide();
    this.visible = false;
    // Return focus to the canvas so keyboard shortcuts keep working.
    // Guarded: if the renderer DOM element is missing or detached (focus
    // not callable), this is a no-op rather than a throw that would abort
    // the rest of hide().
    this.sceneManager.renderer.domElement?.focus?.();
  }

  toggle(): void {
    this.visible ? this.hide() : this.show();
  }

  isVisible(): boolean {
    return this.visible;
  }

  isCurrentlyRecording(): boolean {
    return this.session.isAnyCaptureActive();
  }

  /**
   * True only while the frame-by-frame offline capture loop owns the
   * pipeline, i.e. while the animation loop's own render is redundant.
   * Narrower than `isCurrentlyRecording()`: the real-time MediaRecorder
   * path records the canvas the animation loop paints, so callers that
   * suppress rendering must key off THIS flag, not the general one.
   * Narrower than `session.isOfflineCaptureActive` too — it is dropped
   * as soon as the capture stops driving the pipeline, so a wedged
   * teardown can never leave the viewport frozen.
   */
  isLoopRenderSuppressed(): boolean {
    return this.session.isLoopRenderSuppressed;
  }

  setPanelStateCallbacks(
    getStates: () => PanelStates,
    restoreStates: (states: PanelStates) => void
  ): void {
    this.session.getPanelStates = getStates;
    this.session.restorePanelStatesCallback = restoreStates;
  }

  setAnimationManager(manager: DimensionAnimationManager): void {
    this.session.animationManager = manager;
  }

  setAdaptiveDPRManager(manager: AdaptiveDPRManager): void {
    this.session.adaptiveDPRManager = manager;
  }

  setOverlayManager(manager: OverlayManager | null): void {
    this.session.overlayManager = manager;
  }

  /**
   * Supply the predicate the offline capture loop drains on before exporting
   * each frame: "is every in-frame LOD group showing its selected level at
   * final quality?" (`LODGroupRegistry.isCaptureQuiescent()`). Without it an
   * orbiting turntable bakes coarse-level pops into the sequence whenever a
   * tile re-enters the frustum mid-reload (#1695).
   *
   * TRI-STATE: the provider returns `null` for "this scene has no lod_group to
   * wait for" (no scene loader, no LOD registry, or a registry with none in
   * it), which makes the loop skip the drain entirely — including the one
   * mandatory catch-up rAF a `true` still costs. The provider is wired
   * unconditionally at init, so a plain points/lines scene depends on that
   * `null` to stay exactly as fast as it was pre-#1695. It is not a claim that
   * nothing in the scene can be mid-load: a `--recipe stream` leaf answers
   * `null` while its additive ladder is still streaming (see the hook's JSDoc
   * in `recording-panel/offline-capture-strategy`).
   *
   * INJECTED rather than read directly: reaching the registry means importing
   * `data/scene-loader-manager`, which pulls the whole data/cache stack into
   * this module's graph and breaks every consumer that stubs a minimal
   * `config`. `core/app/init/pipeline` already imports `getSceneLoader` and
   * already wires this panel's other cross-cutting predicates (render-skip,
   * pacing-suspend), so it is the natural place. Unset ⇒ the hook answers
   * `null` and the loop never waits, which is the pre-#1695 behaviour.
   */
  setLODSettledProvider(provider: (() => boolean | null) | null): void {
    this.lodSettledProvider = provider;
  }

  async captureScreenshot(): Promise<void> {
    return this.screenshotStrategy.run(this.options, this.mode, this.session);
  }

  async startVideoRecording(): Promise<void> {
    if (this.session.isRecording) return;

    // Mode dispatch. The real-time MediaRecorder path can ONLY produce a
    // continuous WebM stream — it cannot emit MP4/MKV, image sequences,
    // or EXR. So the only case it can correctly serve is:
    //   • Video mode (always WebM), or
    //   • Turntable mode when frame-by-frame is off AND the format is WebM.
    // Everything else (any non-WebM turntable format, EXR, or smooth
    // capture) must go through the deterministic offline loop.
    const fmt = this.options.outputFormat;

    // EXR carries full float precision and can only be produced by the
    // offline HDR path — never by the real-time MediaRecorder, in any mode.
    if (fmt === 'exr') {
      return this.offlineCaptureStrategy.run(this.options, this.mode, this.session);
    }

    if (this.mode === 'turntable') {
      const realtimeEligible = !this.options.frameByFrame && fmt === 'webm';
      return realtimeEligible
        ? this.videoRecordingStrategy.run(this.options, this.mode, this.session)
        : this.offlineCaptureStrategy.run(this.options, this.mode, this.session);
    }

    return this.videoRecordingStrategy.run(this.options, this.mode, this.session);
  }

  stopVideoRecording(): void {
    if (!this.session.isRecording) return;

    if (this.session.isOfflineCaptureActive || this.session.isEXRSequenceRecording) {
      this.offlineCaptureStrategy.abort();
      return;
    }

    // recordingStartTime is stamped by the video strategy when recording
    // actually starts. Guard against the (defensive) case where isRecording
    // was set without a start time — otherwise `Date.now() - 0` would log a
    // nonsense ~1.7-billion-second elapsed instead of a real duration.
    const elapsed =
      this.session.recordingStartTime > 0
        ? `${((Date.now() - this.session.recordingStartTime) / 1000).toFixed(1)}s`
        : 'unknown duration';
    log.info(Modules.RECORDING, `Stopping video recording after ${elapsed}...`);
    this.videoRecordingStrategy.abort();
  }

  dispose(): void {
    if (this.session.disposed) return;
    // Mark disposed FIRST so any async callback (e.g. mediaRecorder.onstop)
    // fired during the cleanup below short-circuits via session.isDisposed().
    this.session.disposed = true;

    if (this.session.isRecording) {
      this.stopVideoRecording();
    }

    // Each strategy owns its own teardown (per-frame callbacks, media
    // tracks, AbortControllers, overlays).
    this.offlineCaptureStrategy.dispose();
    this.videoRecordingStrategy.dispose();
    this.screenshotStrategy.dispose();

    // Session unwinds shared scaffolding (confirmation dialog,
    // indicator, slider sync, auto-rotate, renderer/DPR/resize-lock,
    // panel-state restore).
    this.session.dispose();
    this.gui.destroy();
  }

  // ========== GUI Construction ==========

  private buildGUI(): void {
    const result = buildRecordingGUI({
      gui: this.gui,
      options: this.options,
      initialMode: this.mode,
      setMode: (mode) => {
        this.mode = mode;
      },
      updateControlVisibility: () => this.updateControlVisibility(),
      getTurntableInfo: () =>
        getTurntableInfoHelper(this.options.turntableSpeed, this.options.videoFPS),
      getNavigableDimensionOptions: () => getNavigableDimensionOptionsHelper(),
      captureScreenshot: () => {
        this.captureScreenshot().catch((error: unknown) => {
          log.error(Modules.RECORDING, 'Screenshot capture failed', error);
          showToast('Screenshot failed');
        });
      },
      startVideoRecording: () => {
        this.startVideoRecording().catch((error: unknown) => {
          log.error(Modules.RECORDING, 'Video recording failed to start', error);
          showToast('Video recording failed to start');
        });
      },
    });
    this.formatController = result.formatController;
    this.qualityController = result.qualityController;
    this.transparentController = result.transparentController;
    this.videoCodecController = result.videoCodecController;
    this.videoQualityController = result.videoQualityController;
    this.videoDurationController = result.videoDurationController;
    this.syncToggleController = result.syncToggleController;
    this.syncDimensionController = result.syncDimensionController;
    this.captureController = result.captureController;
    this.refreshCaptureDPR = result.refreshCaptureDPR;
    this.imageControllers = result.imageControllers;
    this.videoControllers = result.videoControllers;
    this.turntableControllers = result.turntableControllers;
    this.updateControlVisibility();
  }

  /**
   * Show/hide controls based on current mode + format. The per-control
   * decision logic lives in
   * `recording-panel/gui-builder.ts:computeControlVisibility` (pure function);
   * this method only applies the decisions to its named lil-gui
   * controllers and dropdown <option> elements.
   */
  private updateControlVisibility(): void {
    const decision = computeControlVisibility(this.mode, this.options);

    // Action-button verb tracks the mode: "Capture" a still in Image
    // mode, "Record" a clip in Video / Turntable mode.
    this.captureController?.name(captureLabelForMode(this.mode));

    // Filter format-dropdown options by mode. Our GUI uses display
    // labels as option.value (e.g., "PNG" not "png"); map labels →
    // internal values to compare.
    const selectEl = this.formatController?.domElement.querySelector(
      'select'
    ) as HTMLSelectElement | null;
    if (selectEl?.options) {
      for (const opt of Array.from(selectEl.options)) {
        const val = FORMAT_LABEL_TO_VALUE[opt.value] || opt.value;
        opt.hidden = !decision.validFormats.includes(val as OutputFormat);
      }
    }

    if (decision.correctedFormat) {
      this.options.outputFormat = decision.correctedFormat;
      this.formatController?.updateDisplay();
    }

    // Format dropdown is hidden when the mode offers a single choice
    // (Video mode = WebM only), so the one-item dropdown and its silent
    // WebP→WebM rewrite don't confuse the user.
    decision.showFormat ? this.formatController?.show() : this.formatController?.hide();

    for (const ctrl of this.imageControllers) {
      decision.showImageGroup ? ctrl.show() : ctrl.hide();
    }
    // Image quality is decided independently of the image group: WebP /
    // JPEG turntable sequences also encode with `imageQuality`, so the
    // slider must be available in turntable mode too.
    decision.showImageQuality ? this.qualityController?.show() : this.qualityController?.hide();
    if (decision.showImageGroup && !decision.showImageTransparent) {
      this.transparentController?.hide();
    }

    for (const ctrl of this.videoControllers) {
      decision.showVideoGroup ? ctrl.show() : ctrl.hide();
    }
    if (!decision.showVideoCodec) this.videoCodecController?.hide();
    if (!decision.showVideoQuality) this.videoQualityController?.hide();
    if (!decision.showVideoDuration) this.videoDurationController?.hide();
    if (!decision.showSyncToggle) this.syncToggleController?.hide();
    if (!decision.showSyncDimension) this.syncDimensionController?.hide();

    if (decision.visibleVideoCodecs) {
      const visible = decision.visibleVideoCodecs;
      const codecSelect = this.videoCodecController?.domElement.querySelector(
        'select'
      ) as HTMLSelectElement | null;
      if (codecSelect?.options) {
        for (const opt of Array.from(codecSelect.options)) {
          const val = CODEC_LABEL_TO_VALUE[opt.value] || opt.value;
          opt.hidden = !visible.includes(val);
        }
      }
    }

    for (const ctrl of this.turntableControllers) {
      decision.showTurntableGroup ? ctrl.show() : ctrl.hide();
    }
  }

  /**
   * Snapshot panel visibility, hide own GUI, optionally hide all others.
   * Provided to strategies via the `hideAllPanels` hook.
   */
  private hideAllPanels(): void {
    this.session.hideAllPanels(
      {
        hideOwnPanel: () => {
          if (this.visible) {
            this.gui.hide();
            this.visible = false;
          }
        },
      },
      this.options.showPanels
    );
  }

  /** Routed via the `downloadBlob` strategy hook. Spied by tests. */
  private downloadBlob(blob: Blob, filename: string): void {
    downloadBlobHelper(blob, filename);
  }

  /** Routed via the `generateFilename` strategy hook. */
  private generateFilename(ext: string): string {
    return generateFilenamePure(ext);
  }
}
