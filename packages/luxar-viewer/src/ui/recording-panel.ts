// Recording panel for screenshot and video capture of the WebGL canvas
// Uses browser-native APIs: canvas.toBlob() for screenshots,
// canvas.captureStream() + MediaRecorder for video recording (WebM)
// HDR export: EXR screenshots, EXR frame sequences (ZIP), 10-bit HDR video (WebCodecs)

import GUI, { type Controller } from './gui';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import type { SceneManager } from '../scene/scene-manager';
import type { AnimationController } from '../scene/animation/animation-controller';
import type { DimensionAnimationManager } from '../scene/animation/dimension-animation-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import type { OverlayManager } from './overlay-manager';
import {
  computeVideoBitrate as computeVideoBitratePure,
  generateFfmpegScript as generateFfmpegScriptPure,
  generateFilename as generateFilenamePure,
  getSupportedMimeType as getSupportedMimeTypePure,
} from './recording-panel/media-utilities';
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
import { buildRecordingGUI } from './recording-panel/ui/gui-construction';
import { RecordingSession, type SaveRecordingStateOptions } from './recording-panel/session';
import { ScreenshotStrategy } from './recording-panel/screenshot-strategy';
import { VideoRecordingStrategy } from './recording-panel/video-recording-strategy';
import { OfflineCaptureStrategy } from './recording-panel/offline-capture-strategy';

// Shared recording types live in `recording-panel/types.ts`. Re-exported
// here for external consumers that import from the panel directly.
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
 * Features:
 * - Screenshot with transparent background and max DPR
 * - Video recording with slider sync
 * - Turntable 360° rotation recording
 * - HDR EXR screenshots and frame sequences (fflate for ZIP)
 * - 10-bit HDR video encoding (mediabunny for WebCodecs muxing)
 */
export class RecordingPanel {
  private gui: GUI;
  private visible: boolean = false;
  private mode: RecordingMode = 'image';
  private options: RecordingOptions = {
    outputFormat: 'webp',
    imageQuality: 0.92,
    maxDPR: true,
    transparentBackground: false,
    videoDurationLimit: 0,
    videoFPS: 60,
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

  private sceneManager: SceneManager;
  private animationController: AnimationController;

  // Shared scaffolding: state save/restore, dialog, indicator, slider-sync,
  // mutual-exclusion flags, optional deps, disposed guard.
  // Strategies (introduced in subsequent phases) will use this directly.
  private session: RecordingSession;

  // ── Proxy accessors for state that now lives on Session.
  //    These are temporary scaffolding so existing internal code and
  //    test probes (`(panel as any).isRecording`) keep working without
  //    bulk edits. Phase 6 deletes them when tests are split per-collaborator.
  //    Only proxies whose underlying state is read INSIDE this file are
  //    declared here — purely test-facing fields (savedRecordingState,
  //    savedPanelStates, recordingIndicator, recordingTimeInterval, …)
  //    must be accessed via `panel.session.X` directly.
  private get isRecording(): boolean { return this.session.isRecording; }
  private set isRecording(v: boolean) { this.session.isRecording = v; }
  private get isOfflineCaptureActive(): boolean { return this.session.isOfflineCaptureActive; }
  private set isOfflineCaptureActive(v: boolean) { this.session.isOfflineCaptureActive = v; }
  private get isEXRSequenceRecording(): boolean { return this.session.isEXRSequenceRecording; }
  private set isEXRSequenceRecording(v: boolean) { this.session.isEXRSequenceRecording = v; }
  private get disposed(): boolean { return this.session.disposed; }
  private get recordingStartTime(): number { return this.session.recordingStartTime; }
  private set recordingStartTime(v: number) { this.session.recordingStartTime = v; }
  // Used only by test probes; non-private to silence TS6133.
  get animationManager(): DimensionAnimationManager | null { return this.session.animationManager; }
  private get overlayManager(): OverlayManager | null { return this.session.overlayManager; }

  // ── Proxies into VideoRecordingStrategy (test-only; tests still probe
  //    `(panel as any).mediaRecorder` etc.). Removed in Phase 6.
  //    Non-`private` so TS doesn't flag them as unused (they're accessed
  //    only via `(panel as any)` in tests, which TS can't see).
  get mediaRecorder(): MediaRecorder | null { return this.videoRecordingStrategy.mediaRecorder; }
  set mediaRecorder(v: MediaRecorder | null) { this.videoRecordingStrategy.mediaRecorder = v; }
  get captureStream(): MediaStream | null { return this.videoRecordingStrategy.captureStream; }
  set captureStream(v: MediaStream | null) { this.videoRecordingStrategy.captureStream = v; }
  get recordedChunks(): Blob[] { return this.videoRecordingStrategy.recordedChunks; }
  set recordedChunks(v: Blob[]) { this.videoRecordingStrategy.recordedChunks = v; }
  get durationTimer(): ReturnType<typeof setTimeout> | null { return this.videoRecordingStrategy.durationTimer; }
  set durationTimer(v: ReturnType<typeof setTimeout> | null) { this.videoRecordingStrategy.durationTimer = v; }
  private get keepAliveCallbackId(): string { return this.videoRecordingStrategy.keepAliveCallbackId; }
  private get turntableCallbackId(): string { return this.videoRecordingStrategy.turntableCallbackId; }
  // Offline-capture state moved to OfflineCaptureStrategy. The Panel
  // keeps proxies for tests that probe these fields via `(panel as any)`.
  private get offlineSessionAbort(): AbortController | null {
    return this.offlineCaptureStrategy.sessionAbort;
  }
  private set offlineSessionAbort(v: AbortController | null) {
    this.offlineCaptureStrategy.sessionAbort = v;
  }
  private get offlineOverlayCleanup(): (() => void) | null {
    return this.offlineCaptureStrategy.overlayCleanup;
  }
  private set offlineOverlayCleanup(v: (() => void) | null) {
    this.offlineCaptureStrategy.overlayCleanup = v;
  }

  // Strategies (own their own internal state; see capture-strategy.ts).
  private screenshotStrategy: ScreenshotStrategy;
  private videoRecordingStrategy: VideoRecordingStrategy;
  private offlineCaptureStrategy: OfflineCaptureStrategy;

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

  constructor(sceneManager: SceneManager, animationController: AnimationController) {
    this.sceneManager = sceneManager;
    this.animationController = animationController;

    this.session = new RecordingSession(sceneManager, animationController, {
      isExrSequenceActive: () => this.isEXRSequenceRecording,
    });
    // Wire the indicator's click-to-stop and Escape-to-stop into our
    // existing stopVideoRecording path so the user can stop from anywhere.
    this.session.setStopVideoCallback(() => this.stopVideoRecording());

    const downloadHooks = {
      downloadBlob: (blob: Blob, filename: string) => this.downloadBlob(blob, filename),
      generateFilename: (ext: string) => this.generateFilename(ext),
    };

    this.screenshotStrategy = new ScreenshotStrategy(sceneManager, {
      hideAllPanels: () => this.hideAllPanels(),
      ...downloadHooks,
    });

    this.videoRecordingStrategy = new VideoRecordingStrategy(sceneManager, animationController, {
      hideAllPanels: () => this.hideAllPanels(),
      ...downloadHooks,
    });

    this.offlineCaptureStrategy = new OfflineCaptureStrategy(sceneManager, animationController, {
      hideAllPanels: () => this.hideAllPanels(),
      renderFrameToCanvas: () => this.renderFrameToCanvas(),
      ...downloadHooks,
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
      bottom: '20px',
      left: '20px',
      zIndex: String(config.ui.zIndex.recordingPanel),
    });

    this.gui.hide();
    this.buildGUI();
  }

  // ========== Public API ==========

  show(): void {
    this.gui.show();
    this.visible = true;
  }

  hide(): void {
    const activeElement = document.activeElement as HTMLElement;
    if (activeElement?.blur) activeElement.blur();
    this.gui.hide();
    this.visible = false;
    this.sceneManager.renderer.domElement.focus();
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

  dispose(): void {
    if (this.disposed) return;
    // Mark disposed FIRST so any async callback (e.g. mediaRecorder.onstop)
    // fired during the cleanup below short-circuits via `this.disposed` /
    // `session.isDisposed()` checks.
    this.session.disposed = true;

    if (this.isRecording) {
      this.stopVideoRecording();
    }

    // Abort any in-flight offline-capture session so the loop's
    // next await checkpoint sees signal.aborted and short-circuits
    // before downloading/toasting on a disposed panel.
    this.offlineSessionAbort?.abort('disposed');
    this.offlineOverlayCleanup?.();

    this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
    this.animationController.removePerFrameCallback(this.turntableCallbackId);
    // Remove offline-capture callbacks. The normal loop path also
    // removes them in finally; this covers the dispose-while-awaiting
    // case where the loop hasn't reached its finally yet.
    this.animationController.removePerFrameCallback(OfflineCaptureStrategy.CAPTURE_CALLBACK_ID);
    this.animationController.removePerFrameCallback(OfflineCaptureStrategy.KEEPALIVE_CALLBACK_ID);

    // VideoRecordingStrategy unwinds: defensive cleanupCaptureStream
    // (mediaRecorder.onstop may have suppressed itself due to disposed)
    // and clears duration timer + nulls mediaRecorder/recordedChunks.
    this.videoRecordingStrategy.dispose();

    // Session unwinds: confirmation dialog, indicator, slider sync,
    // auto-rotate, renderer/DPR/resize-lock state, panel-state restore.
    this.session.dispose();
    this.gui.destroy();
  }

  // ========== Screenshot Capture ==========

  async captureScreenshot(): Promise<void> {
    return this.screenshotStrategy.run(this.options, this.mode, this.session);
  }

  // ========== Video Recording ==========

  async startVideoRecording(): Promise<void> {
    if (this.isRecording) return;

    // Mode dispatch: decide which capture pipeline to use based on
    // format + turntable-smooth setting. Real-time MediaRecorder path
    // (VideoRecordingStrategy) handles MP4/WebM/MKV in non-smooth mode;
    // everything else falls through to OfflineCaptureStrategy
    // (still in-Panel until Phase 4).
    const fmt = this.options.outputFormat;
    const isImageFormat = fmt === 'png' || fmt === 'webp' || fmt === 'jpeg';
    const isTurntableSmooth = this.mode === 'turntable' && this.options.frameByFrame;

    if (fmt === 'exr') {
      return this.startEXRSequenceRecording();
    }
    if (isTurntableSmooth) {
      if (isImageFormat) {
        return this.runOfflineCaptureLoop(fmt);
      }
      return this.runOfflineCaptureLoop(fmt as 'mp4' | 'webm' | 'mkv');
    }

    return this.videoRecordingStrategy.run(this.options, this.mode, this.session);
  }

  stopVideoRecording(): void {
    if (!this.isRecording) return;

    // Branch: HDR sequence stop
    if (this.isEXRSequenceRecording) {
      this.stopEXRSequenceRecording();
      return;
    }

    // Branch: offline capture loop (non-EXR) — signal via flag to break the loop
    if (this.isOfflineCaptureActive) {
      this.isRecording = false;
      return;
    }
    const elapsed = ((Date.now() - this.recordingStartTime) / 1000).toFixed(1);
    log.info(Modules.RECORDING, `Stopping video recording after ${elapsed}s...`);

    this.videoRecordingStrategy.abort();
  }

  // ========== Offline Capture (image / video / EXR sequences) ==========

  /**
   * Delegate to OfflineCaptureStrategy. The strategy temporarily overrides
   * `opts.outputFormat` so legacy `runOfflineCaptureLoop(mode)` callers
   * (currently EXR-sequence shortcut + the image-format turntable path)
   * still drive the right per-mode driver.
   */
  private async runOfflineCaptureLoop(
    mode: 'exr' | 'webm' | 'mp4' | 'mkv' | 'png' | 'webp' | 'jpeg'
  ): Promise<void> {
    const savedFormat = this.options.outputFormat;
    this.options.outputFormat = mode as typeof savedFormat;
    try {
      await this.offlineCaptureStrategy.run(this.options, this.mode, this.session);
    } finally {
      this.options.outputFormat = savedFormat;
    }
  }


  private async startEXRSequenceRecording(): Promise<void> {
    return this.runOfflineCaptureLoop('exr');
  }

  private stopEXRSequenceRecording(): void {
    // Offline loop checks this.isRecording — setting it to false stops the loop
    this.isRecording = false;
    this.isEXRSequenceRecording = false;
  }

  // ========== Slider Sync ==========

  // Wrappers below are non-private to silence TS6133 — they're only
  // called via `(panel as any).X()` in tests, which the compiler can't see.

  startSliderSync(): void {
    this.session.startSliderSync(
      this.options.syncDimensionIndex,
      () => this.stopVideoRecording()
    );
  }

  /**
   * Stop captureStream tracks — wrapper around VideoRecordingStrategy
   * for tests that still call `(panel as any).cleanupCaptureStream()`.
   */
  cleanupCaptureStream(): void {
    this.videoRecordingStrategy.cleanupCaptureStream();
  }

  /** Restore auto-rotation to its pre-turntable state. Non-private for test probes. */
  restoreAutoRotate(): void {
    this.session.restoreAutoRotate();
  }

  /**
   * Wrapper for tests still probing `(panel as any).startTurntableRotation()`.
   * Internally delegates to the VideoRecordingStrategy.
   */
  startTurntableRotation(): void {
    this.videoRecordingStrategy.startTurntableRotationForTests(this.options, this.session);
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
      getTurntableInfo: () => this.getTurntableInfo(),
      getNavigableDimensionOptions: () => this.getNavigableDimensionOptions(),
      captureScreenshot: () => this.captureScreenshot(),
      startVideoRecording: () => this.startVideoRecording(),
    });
    this.formatController = result.formatController;
    this.qualityController = result.qualityController;
    this.transparentController = result.transparentController;
    this.videoCodecController = result.videoCodecController;
    this.videoQualityController = result.videoQualityController;
    this.videoDurationController = result.videoDurationController;
    this.syncToggleController = result.syncToggleController;
    this.syncDimensionController = result.syncDimensionController;
    this.imageControllers = result.imageControllers;
    this.videoControllers = result.videoControllers;
    this.turntableControllers = result.turntableControllers;
    this.updateControlVisibility();
  }

  /** Compute turntable info string from current speed and FPS */
  private getTurntableInfo(): string {
    return getTurntableInfoHelper(this.options.turntableSpeed, this.options.videoFPS);
  }

  /** Get navigable dimension names as dropdown options */
  private getNavigableDimensionOptions(): Record<string, number> {
    return getNavigableDimensionOptionsHelper();
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

    // Auto-correct format if invalid for this mode.
    if (decision.correctedFormat) {
      this.options.outputFormat = decision.correctedFormat;
      this.formatController?.updateDisplay();
    }

    this.formatController?.show();

    // Image group (quality, max DPR, transparent BG).
    for (const ctrl of this.imageControllers) {
      decision.showImageGroup ? ctrl.show() : ctrl.hide();
    }
    if (decision.showImageGroup) {
      if (!decision.showImageQuality) this.qualityController?.hide();
      if (!decision.showImageTransparent) this.transparentController?.hide();
    }

    // Video / turntable shared group.
    for (const ctrl of this.videoControllers) {
      decision.showVideoGroup ? ctrl.show() : ctrl.hide();
    }
    if (!decision.showVideoCodec) this.videoCodecController?.hide();
    if (!decision.showVideoQuality) this.videoQualityController?.hide();
    if (!decision.showVideoDuration) this.videoDurationController?.hide();
    if (!decision.showSyncToggle) this.syncToggleController?.hide();
    if (!decision.showSyncDimension) this.syncDimensionController?.hide();

    // Filter codec dropdown options when visible.
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

    // Turntable group.
    for (const ctrl of this.turntableControllers) {
      decision.showTurntableGroup ? ctrl.show() : ctrl.hide();
    }
  }

  // ========== Panel Hide/Restore ==========

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

  // ========== Confirmation Dialog ==========
  // Wrappers below are non-private (tests still probe via `(panel as any).X()`).

  showConfirmationDialog(): Promise<boolean> {
    return this.session.showConfirmationDialog({ mode: this.mode, options: this.options });
  }

  // ========== Recording Indicator ==========

  showRecordingIndicator(): void {
    this.session.showRecordingIndicator();
  }

  hideRecordingIndicator(): void {
    this.session.hideRecordingIndicator();
  }

  // ========== Recording State Guard ==========

  saveRecordingState(options: SaveRecordingStateOptions): void {
    this.session.saveRecordingState(options);
  }

  restoreRecordingState(): void {
    this.session.restoreRecordingState();
  }

  // ========== Utilities ==========

  /**
   * Render a frame and return it on an offscreen canvas.
   * Centralizes the renderToImageData → putImageData pattern used by
   * captureScreenshot (SDR path) and runOfflineCaptureLoop (image + video modes).
   * The returned canvas can be passed to toBlob() or to VideoSample.
   */
  private renderFrameToCanvas(): Promise<HTMLCanvasElement> {
    return renderFrameToCanvasHelper(
      this.sceneManager.postProcessing,
      this.options.includeOverlays,
      this.overlayManager,
      this.sceneManager.renderer.domElement
    );
  }

  /** Compute video bitrate based on canvas size, FPS, and quality preset. Non-private for tests. */
  computeVideoBitrate(width: number, height: number): number {
    return computeVideoBitratePure(width, height, this.options.videoFPS, this.options.videoQuality);
  }

  /**
   * Get a supported MIME type for real-time MediaRecorder capture.
   * Non-private — test-only access via `(panel as any).getSupportedMimeType()`.
   */
  getSupportedMimeType(): string | null {
    return getSupportedMimeTypePure();
  }

  /**
   * Generate the ffmpeg shell script bundled with image-sequence and
   * EXR-sequence ZIPs. Non-private for tests.
   */
  generateFfmpegScript(fps: number, frameCount: number, ext = 'exr'): string {
    return generateFfmpegScriptPure(fps, frameCount, ext);
  }

  private generateFilename(ext: string): string {
    return generateFilenamePure(ext);
  }

  private downloadBlob(blob: Blob, filename: string): void {
    downloadBlobHelper(blob, filename);
  }
}
