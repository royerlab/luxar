// Recording panel for screenshot and video capture of the WebGL canvas
// Uses browser-native APIs: canvas.toBlob() for screenshots,
// canvas.captureStream() + MediaRecorder for video recording (WebM)
// HDR export: EXR screenshots, EXR frame sequences (ZIP), 10-bit HDR video (WebCodecs)

import * as THREE from 'three';
import GUI, { type Controller } from './gui';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import { showToast } from './toast';
// HDR video encoder kept for future use when browser 10-bit support matures
// import { isHDRVideoSupported, HDRVideoEncoder } from '../utils/hdr-video-encoder';
import type { SceneManager } from '../scene/scene-manager';
import type { AnimationController } from '../scene/animation/animation-controller';
import type { DimensionAnimationManager } from '../scene/animation/dimension-animation-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import { LuxarOrbitControls } from '../controls/luxar-orbit-controls';
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
  SliderSyncCoordinator,
} from './recording-panel/animation-sync';
import {
  computeControlVisibility,
  FORMAT_LABEL_TO_VALUE,
  CODEC_LABEL_TO_VALUE,
} from './recording-panel/gui-builder';
import type { CaptureContext, OfflineCaptureDriver } from './recording-panel/drivers/offline-capture-driver';
import { ImageSequenceDriver } from './recording-panel/drivers/image-sequence-driver';
import { ExrSequenceDriver } from './recording-panel/drivers/exr-sequence-driver';
import { VideoModeDriver } from './recording-panel/drivers/video-mode-driver';
import { captureScreenshot, type ScreenshotCtx } from './recording-panel/modes/screenshot-mode';

// Shared recording types live in `recording/types.ts`. Re-exported
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
  VideoResolution,
  OutputFormat,
  VideoCodecOption,
  VideoQuality,
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

  // Optional dependencies (set via setters)
  private animationManager: DimensionAnimationManager | null = null;
  private adaptiveDPRManager: AdaptiveDPRManager | null = null;
  private overlayManager: OverlayManager | null = null;

  // Video recording state
  private isRecording: boolean = false;
  private mediaRecorder: MediaRecorder | null = null;
  // canvas.captureStream() returns a MediaStream whose tracks live
  // until explicitly stopped. mediaRecorder.stop() does NOT stop the
  // underlying tracks, so we track the stream here and stop its
  // tracks in every onstop branch to release browser media resources.
  private captureStream: MediaStream | null = null;
  private recordedChunks: Blob[] = [];
  private recordingIndicator: HTMLElement | null = null;
  private recordingTimeInterval: ReturnType<typeof setInterval> | null = null;
  private recordingStartTime: number = 0;
  private durationTimer: ReturnType<typeof setTimeout> | null = null;
  private keepAliveCallbackId = 'recording-keepalive';
  private turntableCallbackId = 'recording-turntable';
  // Offline-capture callback IDs are class-level constants so
  // dispose() can remove them unconditionally even if the loop is
  // parked on an `await` and hasn't reached its finally yet.
  private static readonly OFFLINE_CAPTURE_CALLBACK_ID = 'recording-offline-capture';
  private static readonly OFFLINE_KEEPALIVE_CALLBACK_ID = 'recording-offline-keepalive';
  // AbortController for the offline-capture session. Set immediately
  // after the confirmation check in runOfflineCaptureLoop, BEFORE any
  // state mutation, so dispose() during the early state-save / rAF
  // window can abort the in-flight session. Aborted by dispose() or
  // the cancel button. Drivers + the loop body check signal.aborted
  // between awaits so dispose-during-capture skips finalize cleanly.
  private offlineSessionAbort: AbortController | null = null;
  private sliderSync = new SliderSyncCoordinator();
  private savedAutoRotate: boolean = false;

  // Event/listener cleanup handles for transient recording DOM
  private recordingIndicatorClickCleanup: (() => void) | null = null;
  private offlineOverlayCleanup: (() => void) | null = null;
  private confirmationDialogCancel: (() => void) | null = null;

  // EXR sequence recording state
  private isEXRSequenceRecording: boolean = false;
  private isOfflineCaptureActive: boolean = false;

  // Screenshot debounce
  private isCaptureInProgress: boolean = false;

  // Dispose guard for async onstop handler
  private disposed: boolean = false;

  // Panel state callbacks (set by app.ts)
  private getPanelStates: (() => PanelStates) | null = null;
  private restorePanelStatesCallback: ((states: PanelStates) => void) | null = null;
  private savedPanelStates: PanelStates | null = null;

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

  // Saved recording state for restore after capture/recording
  private savedRecordingState: {
    dprEnabled: boolean;
    dpr: number;
    rendererSize: { width: number; height: number } | null;
    resizeLocked: boolean;
  } | null = null;

  constructor(sceneManager: SceneManager, animationController: AnimationController) {
    this.sceneManager = sceneManager;
    this.animationController = animationController;

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
    return this.isRecording;
  }

  setPanelStateCallbacks(
    getStates: () => PanelStates,
    restoreStates: (states: PanelStates) => void
  ): void {
    this.getPanelStates = getStates;
    this.restorePanelStatesCallback = restoreStates;
  }

  setAnimationManager(manager: DimensionAnimationManager): void {
    this.animationManager = manager;
  }

  setAdaptiveDPRManager(manager: AdaptiveDPRManager): void {
    this.adaptiveDPRManager = manager;
  }

  setOverlayManager(manager: OverlayManager | null): void {
    this.overlayManager = manager;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    this.confirmationDialogCancel?.();

    if (this.isRecording) {
      this.stopVideoRecording();
    }
    if (this.durationTimer) {
      clearTimeout(this.durationTimer);
      this.durationTimer = null;
    }

    // Abort any in-flight offline-capture session so the loop's
    // next await checkpoint sees signal.aborted and short-circuits
    // before downloading/toasting on a disposed panel.
    this.offlineSessionAbort?.abort('disposed');

    this.offlineOverlayCleanup?.();
    this.hideRecordingIndicator();
    this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
    this.animationController.removePerFrameCallback(this.turntableCallbackId);
    // Remove offline-capture callbacks. The normal loop path also
    // removes them in finally; this covers the dispose-while-awaiting
    // case where the loop hasn't reached its finally yet.
    this.animationController.removePerFrameCallback(RecordingPanel.OFFLINE_CAPTURE_CALLBACK_ID);
    this.animationController.removePerFrameCallback(RecordingPanel.OFFLINE_KEEPALIVE_CALLBACK_ID);
    this.cleanupSyncListener();
    this.restoreAutoRotate();
    this.restoreRecordingState();
    // Defensive: stop any captureStream tracks even if mediaRecorder.onstop
    // didn't fire (browser quirks, mid-init dispose).
    this.cleanupCaptureStream();
    this.gui.destroy();
  }

  // ========== Screenshot Capture ==========

  async captureScreenshot(): Promise<void> {
    await captureScreenshot(this.makeScreenshotCtx());
  }

  private makeScreenshotCtx(): ScreenshotCtx {
    const self = this;
    return {
      get isRecording() {
        return self.isRecording;
      },
      get isOfflineCaptureActive() {
        return self.isOfflineCaptureActive;
      },
      get isCaptureInProgress() {
        return self.isCaptureInProgress;
      },
      set isCaptureInProgress(v: boolean) {
        self.isCaptureInProgress = v;
      },
      get options() {
        return self.options;
      },
      get sceneManager() {
        return self.sceneManager;
      },
      get adaptiveDPRManager() {
        return self.adaptiveDPRManager;
      },
      hideAllPanels: () => self.hideAllPanels(),
      saveRecordingState: (opts) => self.saveRecordingState(opts),
      restoreRecordingState: () => self.restoreRecordingState(),
      downloadBlob: (blob, filename) => self.downloadBlob(blob, filename),
      generateFilename: (ext) => self.generateFilename(ext),
      renderFrameToCanvas: () => self.renderFrameToCanvas(),
    };
  }

  // ========== Video Recording ==========

  async startVideoRecording(): Promise<void> {
    if (this.isRecording) return;

    // Branch: frame-by-frame capture modes
    const fmt = this.options.outputFormat;
    const isImageFormat = fmt === 'png' || fmt === 'webp' || fmt === 'jpeg';
    const isTurntableSmooth = this.mode === 'turntable' && this.options.frameByFrame;

    // EXR → ZIP of EXR frames (always offline)
    if (fmt === 'exr') {
      return this.startEXRSequenceRecording();
    }
    // Turntable smooth mode — all formats use offline capture
    if (isTurntableSmooth) {
      if (isImageFormat) {
        return this.runOfflineCaptureLoop(fmt); // → ZIP of images
      }
      return this.runOfflineCaptureLoop(fmt as 'mp4' | 'webm' | 'mkv'); // → video file
    }
    // Video mode with image formats falls through to real-time MediaRecorder below

    const mimeType = this.getSupportedMimeType();
    if (!mimeType) {
      showToast('Video recording not supported in this browser');
      return;
    }

    const confirmed = await this.showConfirmationDialog();
    if (!confirmed || this.disposed) return;

    // Wrap the entire setup phase. Without this, a throw from
    // canvas.captureStream(), `new MediaRecorder(...)`, or
    // mediaRecorder.start() would leave panels hidden, DPR disabled,
    // resize locked, the keepalive callback registered, and no onstop
    // to unwind any of it.
    try {
      this.hideAllPanels();

      // Disable adaptive DPR during recording — resolution changes mid-capture cause
      // frozen frames, aspect ratio glitches, and partial rotations
      this.saveRecordingState({
        disableDPR: true,
        lockResize: true,
        scaleResolution:
          this.options.videoResolution > 0 ? { targetH: this.options.videoResolution } : undefined,
      });
      if (this.options.videoResolution > 0) {
        await new Promise((r) => requestAnimationFrame(r));
        if (this.disposed) {
          this.restoreRecordingState();
          return;
        }
      }

      const canvas = this.sceneManager.renderer.domElement;
      const videoBitsPerSecond = this.computeVideoBitrate(canvas.width, canvas.height);

      log.info(
        Modules.RECORDING,
        `Starting video recording (${mimeType}, ${this.options.videoFPS} FPS, ` +
          `${Math.round(videoBitsPerSecond / 1_000_000)}Mbps, ${canvas.width}x${canvas.height}, ` +
          `mode: ${this.mode})`
      );

      this.animationController.startAnimation();
      this.animationController.addPerFrameCallback(this.keepAliveCallbackId, () => {}, {
        continuous: true,
      });

      this.captureStream = canvas.captureStream(this.options.videoFPS);
      this.mediaRecorder = new MediaRecorder(this.captureStream, { mimeType, videoBitsPerSecond });
      this.recordedChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.recordedChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        if (this.disposed) {
          this.recordedChunks = [];
          this.isRecording = false;
          this.hideRecordingIndicator();
          this.cleanupCaptureStream();
          return;
        }

        const blob = new Blob(this.recordedChunks, { type: mimeType });
        const totalElapsed = ((Date.now() - this.recordingStartTime) / 1000).toFixed(1);
        log.info(
          Modules.RECORDING,
          `Recording finalized: ${this.recordedChunks.length} chunks, ` +
            `${(blob.size / (1024 * 1024)).toFixed(1)} MB, ${totalElapsed}s elapsed`
        );
        this.downloadBlob(blob, this.generateFilename('webm'));
        this.recordedChunks = [];
        this.isRecording = false;
        this.hideRecordingIndicator();

        this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
        this.animationController.removePerFrameCallback(this.turntableCallbackId);
        this.cleanupSyncListener();
        this.restoreAutoRotate();
        this.restoreRecordingState();
        this.cleanupCaptureStream();
        showToast('Video saved');
      };

      this.mediaRecorder.start(100);
      this.isRecording = true;
      this.recordingStartTime = Date.now();
      this.showRecordingIndicator();

      // Duration limit
      if (this.options.videoDurationLimit > 0) {
        this.durationTimer = setTimeout(() => {
          this.stopVideoRecording();
        }, this.options.videoDurationLimit * 1000);
      }

      // Start slider sync if enabled
      if (this.mode === 'video' && this.options.syncToSlider && this.animationManager) {
        this.startSliderSync();
      }

      // Start turntable rotation if in turntable mode
      if (this.mode === 'turntable') {
        this.startTurntableRotation();
      }
    } catch (err) {
      log.error(Modules.RECORDING, `Real-time recording setup failed: ${err}`);
      // Symmetric undo of every state mutation up to this point.
      // mediaRecorder may or may not have been constructed; clearing
      // the field is defensive. captureStream cleanup also runs even
      // if it was never assigned (helper handles null).
      this.cleanupCaptureStream();
      this.mediaRecorder = null;
      this.recordedChunks = [];
      this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
      this.animationController.removePerFrameCallback(this.turntableCallbackId);
      this.cleanupSyncListener();
      this.restoreAutoRotate();
      this.restoreRecordingState();
      this.isRecording = false;
      this.hideRecordingIndicator();
      showToast('Video recording failed to start');
      throw err;
    }
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

    if (this.durationTimer) {
      clearTimeout(this.durationTimer);
      this.durationTimer = null;
    }

    this.mediaRecorder?.stop();
  }

  // ========== Offline Capture (image / video / EXR sequences) ==========

  /**
   * Run a deterministic offline capture loop for image, video, or EXR
   * output. The mode parameter selects the per-mode driver
   * (ImageSequenceDriver, VideoModeDriver, or ExrSequenceDriver).
   *
   * Unlike real-time MediaRecorder capture, this loop is fully decoupled from the
   * browser's animation frame rate. Each frame is:
   * 1. Camera orbited by one step (quaternion rotation, same as auto-rotate)
   * 2. Scene rendered (full pipeline)
   * 3. Pixels read back (synchronous GPU stall — intentional)
   * 4. Frame stored / encoded
   * 5. Brief yield to keep the browser responsive (UI updates, recording indicator)
   *
   * This guarantees every frame is perfectly rendered regardless of GPU speed.
   * The output will be smooth 60fps even if capture takes seconds per frame.
   */
  /**
   * Run a deterministic offline capture loop for image / video / EXR.
   *
   * The loop owns shared scaffolding (state save/restore, modal
   * overlay, animation pump, progress display, error tolerance);
   * per-mode capture (PNG/WebP/JPEG sequence, video container, EXR
   * sequence) is delegated to a driver implementing
   * {@link OfflineCaptureDriver}. Each driver runs its own setup,
   * captures one frame at a time, and finalizes (download/save).
   *
   * The entire post-saveRecordingState body is wrapped in
   * try/finally so an exception from driver.setup, driver.captureFrame,
   * driver.finalize, or any DOM/state mutation cannot leave the panel
   * with a stuck overlay, hidden panels, scaled renderer, or stale
   * recording flags. The finally block is idempotent — every
   * removal/restore handles the "wasn't set" case gracefully.
   */
  private async runOfflineCaptureLoop(
    mode: 'exr' | 'webm' | 'mp4' | 'mkv' | 'png' | 'webp' | 'jpeg'
  ): Promise<void> {
    const confirmed = await this.showConfirmationDialog();
    if (!confirmed || this.disposed) return;

    // Establish session ownership BEFORE any state mutation. dispose()
    // reads `offlineSessionAbort` to abort an in-flight session; if we
    // assign it later (after hideAllPanels / saveRecordingState / the
    // first rAF), a dispose during that early window leaves the
    // abort controller null and the function continues to bring up
    // overlay/recording flags on a disposed panel.
    const sessionAbort = new AbortController();
    this.offlineSessionAbort = sessionAbort;

    // Helper: bail out, releasing session ownership and restoring any
    // state that may have been saved. Called from the early-abort
    // checkpoints below.
    const bailEarly = (): void => {
      this.restoreRecordingState();
      if (this.offlineSessionAbort === sessionAbort) {
        this.offlineSessionAbort = null;
      }
    };

    this.hideAllPanels();

    // Save state, disable DPR, lock resize, and scale resolution.
    // Dimensions are rounded to a multiple of 16 (macroblock alignment for H.264/H.265).
    const targetH = this.options.videoResolution > 0 ? this.options.videoResolution : 1080;
    this.saveRecordingState({
      disableDPR: true,
      lockResize: true,
      scaleResolution: { targetH, align16: true },
    });
    await new Promise((r) => requestAnimationFrame(r));

    // Re-check after the rAF wait. dispose() during this await fires
    // sessionAbort, which we observe here so the function does not
    // proceed to overlay creation / driver setup on a disposed panel.
    if (this.disposed || sessionAbort.signal.aborted) {
      bailEarly();
      return;
    }

    // Compute turntable parameters
    const fps = this.options.videoFPS;
    const durationSeconds = 360 / this.options.turntableSpeed;
    const totalFrames = Math.ceil(durationSeconds * fps);

    const controls = this.sceneManager.controls.getControls();
    if (!(controls instanceof LuxarOrbitControls)) {
      log.warning(Modules.RECORDING, 'Turntable requires orbit controls');
      bailEarly();
      return;
    }

    // Pause auto-rotation so it doesn't compound with the turntable.
    // savedAutoRotate is the only field we mutate before the
    // try/finally; restoreAutoRotate inside the finally restores it
    // unconditionally.
    this.savedAutoRotate = this.sceneManager.controls.getAutoRotate();
    this.sceneManager.controls.setAutoRotate(false);

    // Per-frame rotation step: frame 0 captures the starting view without rotation,
    // then frames 1..N-1 each advance by one step to complete exactly 2π total.
    const anglePerFrame = totalFrames > 1 ? (2 * Math.PI) / (totalFrames - 1) : 0;

    log.info(
      Modules.RECORDING,
      `Starting offline ${mode} capture: ${totalFrames} frames, ${fps} FPS, ${durationSeconds.toFixed(1)}s`
    );

    // Build the per-mode driver. EXR mode flips the panel's
    // isEXRSequenceRecording flag in finalize via a callback so the
    // driver doesn't need to know about that field.
    const driver: OfflineCaptureDriver =
      mode === 'png' || mode === 'webp' || mode === 'jpeg'
        ? new ImageSequenceDriver(mode)
        : mode === 'exr'
          ? new ExrSequenceDriver(() => {
              this.isEXRSequenceRecording = false;
            })
          : new VideoModeDriver(mode);

    // Set recording state
    if (mode === 'exr') {
      this.isEXRSequenceRecording = true;
    }
    this.isRecording = true;
    this.isOfflineCaptureActive = true;
    this.recordingStartTime = Date.now();
    this.showRecordingIndicator();

    // Offline overlay built as a real modal dialog: dialog/aria-modal
    // semantics + cancel-button focus + explicit Escape handling that
    // aborts the session. The keydown listener stops propagation for
    // non-Escape keys so navigation/dimension shortcuts don't fire
    // mid-capture.
    const overlay = document.createElement('div');
    overlay.className = 'luxar-recording-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-labelledby', 'luxar-recording-overlay-label');
    overlay.setAttribute('aria-describedby', 'luxar-recording-overlay-counter');
    overlay.innerHTML = `
      <div class="luxar-recording-overlay__content">
        <canvas class="luxar-recording-overlay__preview"></canvas>
        <div class="luxar-recording-overlay__progress">
          <span id="luxar-recording-overlay-label" class="luxar-recording-overlay__label">Capturing frames...</span>
          <span id="luxar-recording-overlay-counter" class="luxar-recording-overlay__counter">0/${totalFrames}</span>
        </div>
        <button class="luxar-recording-overlay__cancel">Cancel</button>
      </div>
    `;
    const previewCanvas = overlay.querySelector(
      '.luxar-recording-overlay__preview'
    ) as HTMLCanvasElement;
    const previewCtx = previewCanvas.getContext('2d');
    const cancelButton = overlay.querySelector(
      '.luxar-recording-overlay__cancel'
    ) as HTMLButtonElement | null;
    // Remember the previously-focused element so we can restore focus
    // when the overlay closes (modal dialog convention).
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const handleCancel = (): void => {
      this.isRecording = false;
      sessionAbort.abort('user-cancel');
    };
    const handleOverlayKeydown = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        handleCancel();
        return;
      }
      // Focus trap: keep Tab inside the overlay so focus can't
      // escape to the canvas mid-capture and route Escape through
      // the global input handler instead of this overlay's cancel
      // path.
      if (e.key === 'Tab') {
        const focusable = Array.from(
          overlay.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => !el.hasAttribute('disabled'));
        if (focusable.length > 0) {
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          const active = document.activeElement as HTMLElement | null;
          if (e.shiftKey && active === first) {
            e.preventDefault();
            last.focus();
          } else if (!e.shiftKey && active === last) {
            e.preventDefault();
            first.focus();
          }
        }
        e.stopPropagation();
        return;
      }
      // Other keys still get blocked from reaching the canvas/global
      // shortcuts so typing doesn't inadvertently fire dimensions
      // navigation etc. mid-capture.
      e.stopPropagation();
    };
    cancelButton?.addEventListener('click', handleCancel);
    overlay.addEventListener('keydown', handleOverlayKeydown, true);

    let overlayCleaned = false;
    const cleanupOfflineOverlay = (): void => {
      if (overlayCleaned) return;
      overlayCleaned = true;
      cancelButton?.removeEventListener('click', handleCancel);
      overlay.removeEventListener('keydown', handleOverlayKeydown, true);
      overlay.remove();
      // Restore focus to whatever was focused before we hijacked the
      // page (modal-dialog convention).
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
        previouslyFocused.focus();
      }
      if (this.offlineOverlayCleanup === cleanupOfflineOverlay) {
        this.offlineOverlayCleanup = null;
      }
    };
    this.offlineOverlayCleanup = cleanupOfflineOverlay;

    document.body.appendChild(overlay);
    // Focus the cancel button so Escape / Enter route through the
    // overlay's keydown handler rather than wherever focus was before.
    cancelButton?.focus();

    const counterEl = overlay.querySelector('.luxar-recording-overlay__counter');
    const labelEl = overlay.querySelector('.luxar-recording-overlay__label');

    // Build the dependency context the driver needs. Methods are
    // pre-bound so the driver doesn't need a panel reference.
    const ctx: CaptureContext = {
      sceneManager: this.sceneManager,
      fps,
      renderFrameToCanvas: () => this.renderFrameToCanvas(),
      generateFilename: (ext) => this.generateFilename(ext),
      generateFfmpegScript: (rate, frames, ext) => this.generateFfmpegScript(rate, frames, ext),
      downloadBlob: (blob, filename) => this.downloadBlob(blob, filename),
      computeVideoBitrate: (w, h) => this.computeVideoBitrate(w, h),
      showToast,
      logWarning: (msg) => log.warning(Modules.RECORDING, msg),
      logError: (msg) => log.error(Modules.RECORDING, msg),
      imageQuality: this.options.imageQuality,
      videoCodec: this.options.videoCodec,
      env: window as unknown as CaptureContext['env'],
      signal: sessionAbort.signal,
    };

    const progress = {
      setLabel: (text: string): void => {
        if (labelEl) labelEl.textContent = text;
      },
      setPreview: (canvas: HTMLCanvasElement): void => {
        if (!previewCtx) return;
        if (previewCanvas.width !== canvas.width || previewCanvas.height !== canvas.height) {
          previewCanvas.width = canvas.width;
          previewCanvas.height = canvas.height;
        }
        previewCtx.drawImage(canvas, 0, 0);
      },
    };

    // Use the class-static IDs so dispose() can remove these
    // callbacks even if this loop is parked on an await.
    const captureCallbackId = RecordingPanel.OFFLINE_CAPTURE_CALLBACK_ID;
    const keepAliveId = RecordingPanel.OFFLINE_KEEPALIVE_CALLBACK_ID;
    let capturedFrames = 0;
    let setupCompleted = false;
    let finalizeSucceeded = false;

    try {
      // Per-mode driver setup (encoder construction, file picker, …).
      // A false return means the driver couldn't proceed — currently
      // only VideoModeDriver returns false (no supported codec at the
      // requested resolution). The ZIP-based image/EXR drivers always
      // return true; a user-cancelled file picker falls through to
      // browser-download mode rather than aborting. Drivers that
      // return false are expected to have already toasted the user.
      const setupOk = await driver.setup(ctx);
      if (!setupOk) {
        return; // finally restores all state
      }
      setupCompleted = true;
      if (sessionAbort.signal.aborted) return; // disposed during setup

      // Frame-by-frame capture using the live animation loop.
      //
      // The animation loop runs: controls.update() → per-frame callbacks → render().
      // We register a per-frame callback that applies an incremental quaternion rotation
      // (same math as auto-rotate), then the animation loop renders and we read pixels.
      //
      // This guarantees we read pixels from a properly rendered frame — the same
      // pipeline that produces visible on-screen output.
      let consecutiveErrors = 0;
      const MAX_CONSECUTIVE_ERRORS = 3;

      // Keep the animation loop alive during the entire capture session.
      // This is a separate no-op callback so we can safely add/remove the rotation
      // callback each frame without risking the animation loop pausing mid-capture.
      this.animationController.addPerFrameCallback(keepAliveId, () => {}, { continuous: true });

      for (let i = 0; i < totalFrames; i++) {
        if (!this.isRecording) break;
        if (sessionAbort.signal.aborted) break;
        if (driver.shouldAbort?.()) break;

        // Orbit camera by one step and capture in a single animation frame.
        // The callback applies a quaternion rotation (same as auto-rotate) AFTER
        // controls.update, BEFORE render, so the frame is rendered at the new angle.
        //
        // IMPORTANT: The rotation callback is registered fresh each iteration and
        // removed immediately after the frame renders. If it stayed registered
        // (continuous: true), the animation loop would apply extra rotations during
        // the async capture work between iterations.
        this.animationController.addPerFrameCallback(captureCallbackId, () => {
          if (i > 0) controls.applyOrbitRotation(anglePerFrame);
        });

        // Wait for one full animation frame (callback + render)
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

        // Remove the rotation callback immediately so the animation loop cannot
        // apply extra rotations while we do async capture work below.
        this.animationController.removePerFrameCallback(captureCallbackId);

        // Re-check the abort signal after the rAF wait. A dispose
        // during the wait must NOT proceed to captureFrame, which
        // could download/toast or observe disposed renderer state.
        if (sessionAbort.signal.aborted) break;

        // The animation loop has rendered with the rotated camera. Hand off
        // to the per-mode driver to capture the frame. Pass the OUTPUT
        // frame index (capturedFrames so far), not the source loop
        // index `i`. With tolerated frame failures, source `i` skips
        // ahead while the output sequence stays contiguous —
        // VideoModeDriver uses this for VideoSample.timestamp so the
        // encoded video has gap-free timing; image/EXR drivers ignore
        // it because ZipSequenceCapture maintains its own success
        // counter for filenames.
        try {
          await driver.captureFrame(ctx, capturedFrames, progress);
          capturedFrames++;
          consecutiveErrors = 0;
        } catch (err) {
          consecutiveErrors++;
          log.error(Modules.RECORDING, `Frame ${i + 1} capture failed: ${err}`);
          if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            log.error(
              Modules.RECORDING,
              `${MAX_CONSECUTIVE_ERRORS} consecutive failures — aborting capture. ` +
                'The browser may not support 10-bit encoding at this resolution.'
            );
            showToast('HDR video encoding failed — try EXR sequence instead');
            break;
          }
        }

        if (counterEl) counterEl.textContent = `${i + 1}/${totalFrames}`;
      }

      // If the session was aborted, skip finalize so we don't
      // download a partial artifact. The finally block will route
      // to driver.abort instead.
      if (sessionAbort.signal.aborted) {
        return;
      }

      // Driver-specific finalize. Wrapped in its own try/catch so a
      // throw here surfaces a toast but doesn't bypass the outer
      // finally — the panel state still gets restored, and the
      // finally-block driver.abort() runs because finalizeSucceeded
      // remains false.
      try {
        await driver.finalize(ctx, capturedFrames, progress);
        finalizeSucceeded = true;
      } catch (err) {
        log.error(Modules.RECORDING, `Offline ${mode} finalize failed: ${err}`);
        showToast('Recording finalize failed');
      }
    } catch (err) {
      log.error(Modules.RECORDING, `Offline ${mode} capture failed: ${err}`);
      showToast('Recording failed');
    } finally {
      // If setup completed and finalize did not succeed (aborted
      // session, captureFrame threw past the tolerance limit, or
      // finalize itself threw), give the driver a chance to release
      // partial encoder/zip resources without delivering an artifact.
      if (setupCompleted && !finalizeSucceeded) {
        try {
          const reason = sessionAbort.signal.aborted
            ? sessionAbort.signal.reason === 'user-cancel'
              ? 'user-cancel'
              : 'disposed'
            : 'error';
          await driver.abort?.(ctx, reason as 'disposed' | 'user-cancel' | 'error');
        } catch (abortErr) {
          log.warning(Modules.RECORDING, `Driver abort during cleanup failed: ${abortErr}`);
        }
      }
      // Idempotent cleanup. removePerFrameCallback tolerates unknown
      // IDs; cleanupOfflineOverlay short-circuits if already cleaned;
      // restoreAutoRotate / restoreRecordingState are no-ops if the
      // saved state is missing.
      this.animationController.removePerFrameCallback(captureCallbackId);
      this.animationController.removePerFrameCallback(keepAliveId);
      this.hideRecordingIndicator();
      this.isRecording = false;
      this.isOfflineCaptureActive = false;
      this.isEXRSequenceRecording = false;
      cleanupOfflineOverlay();
      this.restoreAutoRotate();
      this.restoreRecordingState();
      // Clear the session-abort field iff it's still pointing to ours
      // (a re-entrant call would have already set up a new one).
      if (this.offlineSessionAbort === sessionAbort) {
        this.offlineSessionAbort = null;
      }
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

  private startSliderSync(): void {
    if (!this.animationManager) return;
    this.sliderSync.start(
      this.options.syncDimensionIndex,
      this.animationManager,
      () => this.stopVideoRecording(),
      () => !this.disposed
    );
  }

  private cleanupSyncListener(): void {
    this.sliderSync.cleanup(this.animationManager);
  }

  /**
   * Stop every track on the captureStream and drop the reference.
   * `mediaRecorder.stop()` does NOT stop the underlying tracks, so
   * without this call canvas-capture media tracks accumulate across
   * repeated recordings.
   */
  private cleanupCaptureStream(): void {
    this.captureStream?.getTracks().forEach((track) => track.stop());
    this.captureStream = null;
  }

  /** Restore auto-rotation to its pre-turntable state. */
  private restoreAutoRotate(): void {
    if (this.savedAutoRotate) {
      this.sceneManager.controls.setAutoRotate(true);
      this.savedAutoRotate = false;
    }
  }

  // ========== Turntable Rotation ==========

  /**
   * Start time-based turntable rotation for the standard MediaRecorder path.
   *
   * Uses the same quaternion-based orbit rotation as auto-rotate (screen-up axis),
   * driven by wall clock time because MediaRecorder operates in real time.
   * The rotation completes after the correct wall clock duration regardless of GPU FPS.
   *
   * Note: The offline capture loop (for EXR/HDR video) uses its own
   * frame-index-based stepping and does NOT use this method.
   */
  private startTurntableRotation(): void {
    const controls = this.sceneManager.controls.getControls();
    if (!(controls instanceof LuxarOrbitControls)) {
      log.warning(Modules.RECORDING, 'Turntable requires orbit controls');
      return;
    }

    // Pause auto-rotation so it doesn't compound with the turntable
    this.savedAutoRotate = this.sceneManager.controls.getAutoRotate();
    this.sceneManager.controls.setAutoRotate(false);

    const totalDuration = (360 / this.options.turntableSpeed) * 1000;
    const startTime = Date.now();

    log.info(
      Modules.RECORDING,
      `Turntable started: speed=${this.options.turntableSpeed}°/s, ` +
        `duration=${(totalDuration / 1000).toFixed(1)}s`
    );

    let turntableDone = false;
    let frameCount = 0;
    let lastProgress = 0;
    this.animationController.addPerFrameCallback(
      this.turntableCallbackId,
      () => {
        if (turntableDone) return;
        frameCount++;

        // Time-based progress — rotation completes after the correct wall clock duration
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / totalDuration, 1);
        const deltaAngle = (progress - lastProgress) * Math.PI * 2;
        lastProgress = progress;

        // Quaternion orbit — same math as auto-rotation (screen-up axis)
        controls.applyOrbitRotation(deltaAngle);

        if (progress >= 1) {
          turntableDone = true;
          log.info(
            Modules.RECORDING,
            `Turntable completed: ${frameCount} rendered frames in ` +
              `${(elapsed / 1000).toFixed(1)}s (${(frameCount / (elapsed / 1000)).toFixed(1)} FPS)`
          );
          this.stopVideoRecording();
        }
      },
      { continuous: true }
    );
  }

  // ========== GUI Construction ==========

  private buildGUI(): void {
    // Mode toggle — at the top
    const modeObj = { mode: this.mode };
    this.gui
      .add(modeObj, 'mode', { Image: 'image', Video: 'video', Turntable: 'turntable' })
      .name('Mode')
      .onChange((val: string) => {
        this.mode = val as RecordingMode;
        this.updateControlVisibility();
      });

    // Advanced Options folder (starts closed)
    const advanced = this.gui.addFolder('Advanced Options');
    advanced.close();

    // ── General options (inside Advanced) ──
    const generalSettings = { showPanels: this.options.showPanels };
    const showPanelsCtrl = advanced
      .add(generalSettings, 'showPanels')
      .name('Show Panels')
      .onChange((val: boolean) => {
        this.options.showPanels = val;
      });
    showPanelsCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Keep other panels visible during capture');

    const overlaySettings = { includeOverlays: this.options.includeOverlays };
    const overlayCtrl = advanced
      .add(overlaySettings, 'includeOverlays')
      .name('Include Overlays')
      .onChange((val: boolean) => {
        this.options.includeOverlays = val;
      });
    overlayCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Composite text/image/HTML overlays into the capture');

    // ── Image options ──
    const imgSettings = {
      format: this.options.outputFormat,
      quality: this.options.imageQuality,
      maxDPR: this.options.maxDPR,
      transparentBg: this.options.transparentBackground,
    };

    const formatCtrl = advanced
      .add(imgSettings, 'format', {
        PNG: 'png',
        WebP: 'webp',
        JPEG: 'jpeg',
        EXR: 'exr',
        MP4: 'mp4',
        WebM: 'webm',
        MKV: 'mkv',
      })
      .name('Format')
      .onChange((val: string) => {
        this.options.outputFormat = val as OutputFormat;
        this.updateControlVisibility();
      });
    this.formatController = formatCtrl;
    // Format is always visible (applies to both image and video/turntable modes)

    const qualityCtrl = advanced
      .add(imgSettings, 'quality', 0.1, 1.0, 0.05)
      .name('Image Quality')
      .onChange((val: number) => {
        this.options.imageQuality = val;
      });
    this.imageControllers.push(qualityCtrl);
    this.qualityController = qualityCtrl;

    const maxDPRCtrl = advanced
      .add(imgSettings, 'maxDPR')
      .name('Max Resolution')
      .onChange((val: boolean) => {
        this.options.maxDPR = val;
      });
    maxDPRCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Maximize pixel ratio for highest resolution screenshot');
    this.imageControllers.push(maxDPRCtrl);

    const transparentCtrl = advanced
      .add(imgSettings, 'transparentBg')
      .name('Transparent BG')
      .onChange((val: boolean) => {
        this.options.transparentBackground = val;
      });
    transparentCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Transparent background (PNG/WebP only, auto-switches from JPEG)');
    this.imageControllers.push(transparentCtrl);
    this.transparentController = transparentCtrl;

    // ── Video options ──
    const vidSettings = {
      duration: this.options.videoDurationLimit,
      fps: this.options.videoFPS,
      codec: this.options.videoCodec,
      quality: this.options.videoQuality,
      resolution: this.options.videoResolution,
      syncToSlider: this.options.syncToSlider,
      syncDim: this.options.syncDimensionIndex,
    };

    const videoQualityCtrl = advanced
      .add(vidSettings, 'quality', { Low: 'low', Medium: 'medium', High: 'high', Max: 'max' })
      .name('Video Quality')
      .onChange((val: string) => {
        this.options.videoQuality = val as VideoQuality;
      });
    videoQualityCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute(
        'title',
        'Video bitrate quality (Low ~2.5Mbps, Medium ~5Mbps, High ~9Mbps, Max ~19Mbps at 1080p)'
      );
    this.videoControllers.push(videoQualityCtrl);
    this.videoQualityController = videoQualityCtrl;

    const resolutionCtrl = advanced
      .add(vidSettings, 'resolution', { Native: 0, '1080p': 1080, '1440p': 1440, '4K': 2160 })
      .name('Resolution')
      .onChange((val: number) => {
        this.options.videoResolution = val as VideoResolution;
      });
    resolutionCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Output video resolution (Native = current canvas size)');
    this.videoControllers.push(resolutionCtrl);

    const durationCtrl = advanced
      .add(vidSettings, 'duration', 0, 300, 1)
      .name('Max Duration (s)')
      .onChange((val: number) => {
        this.options.videoDurationLimit = val;
      });
    durationCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Recording duration limit in seconds (0 = unlimited)');
    this.videoControllers.push(durationCtrl);
    this.videoDurationController = durationCtrl;

    const fpsCtrl = advanced
      .add(vidSettings, 'fps', { '30 FPS': 30, '60 FPS': 60 })
      .name('Frame Rate')
      .onChange((val: number) => {
        this.options.videoFPS = val;
      });
    this.videoControllers.push(fpsCtrl);

    // Video codec selector (shown only for MP4/WebM formats)
    const codecCtrl = advanced
      .add(vidSettings, 'codec', {
        'H.265': 'h265',
        VP9: 'vp9',
        'H.264': 'h264',
        VP8: 'vp8',
      })
      .name('Codec')
      .onChange((val: string) => {
        this.options.videoCodec = val as VideoCodecOption;
      });
    this.videoControllers.push(codecCtrl);
    this.videoCodecController = codecCtrl;

    // Sync to slider
    const syncCtrl = advanced
      .add(vidSettings, 'syncToSlider')
      .name('Sync to Slider')
      .onChange((val: boolean) => {
        this.options.syncToSlider = val;
        this.syncDimensionController?.[val ? 'show' : 'hide']();
      });
    syncCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Sync recording to a dimension animation (auto-stop at end)');
    this.videoControllers.push(syncCtrl);
    this.syncToggleController = syncCtrl;

    // Sync dimension dropdown
    const dimNames = this.getNavigableDimensionOptions();
    const syncDimCtrl = advanced
      .add(vidSettings, 'syncDim', dimNames)
      .name('Dimension')
      .onChange((val: number) => {
        this.options.syncDimensionIndex = val;
      });
    this.videoControllers.push(syncDimCtrl);
    this.syncDimensionController = syncDimCtrl;

    // ── Turntable options ──
    const ttSettings = { speed: this.options.turntableSpeed };

    // Turntable info display (computed from speed + FPS, read-only)
    const turntableInfo = { info: this.getTurntableInfo() };
    const turntableInfoCtrl = advanced.add(turntableInfo, 'info').name('Output');
    // Make the input read-only (this is a computed display, not user-editable)
    const infoInput = turntableInfoCtrl.domElement.querySelector('input');
    if (infoInput) {
      infoInput.readOnly = true;
      infoInput.style.opacity = '0.7';
      infoInput.style.cursor = 'default';
    }
    this.turntableControllers.push(turntableInfoCtrl);

    const updateTurntableInfo = () => {
      turntableInfo.info = this.getTurntableInfo();
      turntableInfoCtrl.updateDisplay();
    };

    const speedCtrl = advanced
      .add(ttSettings, 'speed', 6, 180, 1)
      .name('Speed (°/s)')
      .onChange((val: number) => {
        this.options.turntableSpeed = val;
        updateTurntableInfo();
      });
    speedCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Rotation speed in degrees per second (36 = 10s for 360°)');
    this.turntableControllers.push(speedCtrl);

    // Also update turntable info when FPS changes
    fpsCtrl.onChange((val: number) => {
      this.options.videoFPS = val;
      updateTurntableInfo();
    });

    // Frame-by-frame (smooth) checkbox
    const fbfSettings = { frameByFrame: this.options.frameByFrame };
    const fbfCtrl = advanced
      .add(fbfSettings, 'frameByFrame')
      .name('Smooth (offline)')
      .onChange((val: boolean) => {
        this.options.frameByFrame = val;
      });
    fbfCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute(
        'title',
        'Render each frame individually for perfectly smooth video. ' +
          'Slower to capture, but guarantees every frame is fully rendered. ' +
          'Recommended for heavy scenes.'
      );
    this.turntableControllers.push(fbfCtrl);

    // Capture/Record button — at the bottom, prominent
    const actions = {
      capture: () => {
        if (this.mode === 'image') {
          this.captureScreenshot();
        } else {
          this.startVideoRecording();
        }
      },
    };
    const captureBtn = this.gui.add(actions, 'capture').name('Capture');
    captureBtn.domElement.closest('.luxar-gui__controller')?.classList.add('luxar-recording-btn');

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
   * `recording/gui-builder.ts:computeControlVisibility` (pure function);
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
    if (!this.savedPanelStates && this.getPanelStates) {
      this.savedPanelStates = this.getPanelStates();
    }
    if (this.visible) {
      this.gui.hide();
      this.visible = false;
    }
    if (this.options.showPanels) return;

    if (this.savedPanelStates && this.restorePanelStatesCallback) {
      const allHidden = new Map<string, boolean>();
      for (const key of this.savedPanelStates.keys()) {
        allHidden.set(key, false);
      }
      this.restorePanelStatesCallback(allHidden);
    }
  }

  private restoreAllPanels(): void {
    if (this.isRecording) return;
    if (this.savedPanelStates && this.restorePanelStatesCallback) {
      this.restorePanelStatesCallback(this.savedPanelStates);
      this.savedPanelStates = null;
    }
  }

  // ========== Confirmation Dialog ==========

  private showConfirmationDialog(): Promise<boolean> {
    return new Promise((resolve) => {
      const overlay = document.createElement('div');
      overlay.className = 'luxar-recording-confirm';
      // Modal dialog ARIA semantics — match the offline overlay so
      // screen readers announce the dialog, and Tab is trapped inside
      // it. Without these, the dialog is just a styled <div>.
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');
      overlay.setAttribute('aria-labelledby', 'luxar-recording-confirm-title');
      overlay.setAttribute('aria-describedby', 'luxar-recording-confirm-message');

      const fmt = this.options.outputFormat;
      let details = `Recording will capture at ${this.options.videoFPS} FPS.`;
      if (this.mode === 'turntable') {
        const duration = Math.round(360 / this.options.turntableSpeed);
        const expectedFrames = Math.ceil(duration * this.options.videoFPS);
        details = `Camera will rotate 360° — ${expectedFrames} frames at ${this.options.videoFPS} FPS (${duration}s video).`;
        if (this.options.frameByFrame) {
          details += '<br><strong>Offline capture</strong> — each frame is rendered individually.';
        }
      } else if (this.options.syncToSlider) {
        details += '<br>Recording will stop when the slider animation completes.';
      } else if (this.options.videoDurationLimit > 0) {
        details += `<br>Duration limit: ${this.options.videoDurationLimit} seconds.`;
      }
      if (fmt === 'exr') {
        details += '<br>Output: <strong>ZIP of EXR frames</strong> (full float precision).';
      } else if (fmt === 'png' || fmt === 'webp' || fmt === 'jpeg') {
        details += `<br>Output: <strong>ZIP of ${fmt.toUpperCase()} frames</strong> + ffmpeg script.`;
      } else if (fmt === 'mp4' || fmt === 'webm' || fmt === 'mkv') {
        details += `<br>Output: <strong>${fmt.toUpperCase()} video</strong> (${this.options.videoCodec.toUpperCase()}).`;
      }

      overlay.innerHTML = `
        <div class="luxar-recording-confirm__dialog">
          <div id="luxar-recording-confirm-title" class="luxar-recording-confirm__title">Start ${this.mode === 'turntable' ? 'Turntable' : 'Video'} Recording</div>
          <p id="luxar-recording-confirm-message" class="luxar-recording-confirm__message">
            ${details}<br>
            Press <span class="luxar-recording-confirm__keybinding">Escape</span> to stop recording.
          </p>
          <div class="luxar-recording-confirm__buttons">
            <button class="luxar-recording-confirm__btn" data-action="cancel">Cancel</button>
            <button class="luxar-recording-confirm__btn luxar-recording-confirm__btn--primary" data-action="start">Start Recording</button>
          </div>
        </div>
      `;

      // Capture the previously-focused element so we can restore
      // focus when the dialog closes (modal dialog convention).
      const previouslyFocused = document.activeElement as HTMLElement | null;

      let settled = false;
      const finish = (result: boolean): void => {
        if (settled) return;
        settled = true;
        cleanup();
        // Restore focus to where it was before the dialog opened so
        // keyboard users don't lose their place.
        previouslyFocused?.focus?.();
        resolve(result);
      };

      const focusableSelector =
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
      const getFocusable = (): HTMLElement[] =>
        Array.from(overlay.querySelectorAll<HTMLElement>(focusableSelector)).filter(
          (el) => !el.hasAttribute('disabled')
        );

      const trapKeyboard = (e: KeyboardEvent) => {
        e.stopPropagation();
        if (e.key === 'Escape') {
          finish(false);
          return;
        }
        if (e.key === 'Enter') {
          finish(true);
          return;
        }
        // Focus trap: cycle Tab inside the dialog so focus can't
        // escape to the canvas / external UI.
        if (e.key === 'Tab') {
          const focusable = getFocusable();
          if (focusable.length === 0) return;
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          const active = document.activeElement as HTMLElement | null;
          if (e.shiftKey && active === first) {
            e.preventDefault();
            last.focus();
          } else if (!e.shiftKey && active === last) {
            e.preventDefault();
            first.focus();
          }
        }
      };

      const handleClick = (e: MouseEvent) => {
        const action = (e.target as HTMLElement).dataset.action;
        if (action === 'cancel') {
          finish(false);
        } else if (action === 'start') {
          finish(true);
        }
      };

      const cleanup = () => {
        overlay.removeEventListener('keydown', trapKeyboard, true);
        overlay.removeEventListener('click', handleClick);
        overlay.remove();
        if (this.confirmationDialogCancel === cancelConfirmation) {
          this.confirmationDialogCancel = null;
        }
      };

      const cancelConfirmation = (): void => finish(false);
      this.confirmationDialogCancel = cancelConfirmation;

      overlay.addEventListener('keydown', trapKeyboard, true);
      overlay.addEventListener('click', handleClick);
      document.body.appendChild(overlay);
      // Focus the primary action so Enter confirms by default.
      // Falls back to the overlay itself if the button isn't found
      // (e.g. innerHTML was overridden by tests).
      const primaryBtn = overlay.querySelector<HTMLButtonElement>(
        '.luxar-recording-confirm__btn--primary'
      );
      if (primaryBtn) {
        primaryBtn.focus();
      } else {
        overlay.tabIndex = -1;
        overlay.focus();
      }
    });
  }

  // ========== Recording Indicator ==========

  private showRecordingIndicator(): void {
    this.hideRecordingIndicator();

    const indicator = document.createElement('div');
    indicator.className = 'luxar-recording-indicator';
    indicator.innerHTML = `
      <div class="luxar-recording-indicator__dot"></div>
      <span class="luxar-recording-indicator__text">REC</span>
      <span class="luxar-recording-indicator__time">00:00</span>
    `;
    indicator.title = 'Click to stop recording';
    const handleIndicatorClick = (): void => this.stopVideoRecording();
    indicator.addEventListener('click', handleIndicatorClick);
    this.recordingIndicatorClickCleanup = () => {
      indicator.removeEventListener('click', handleIndicatorClick);
      this.recordingIndicatorClickCleanup = null;
    };
    document.body.appendChild(indicator);
    this.recordingIndicator = indicator;

    const timeEl = indicator.querySelector('.luxar-recording-indicator__time');
    this.recordingTimeInterval = setInterval(() => {
      if (timeEl) {
        if (this.isEXRSequenceRecording) {
          // Offline capture uses the overlay for progress, not this indicator
          timeEl.textContent = 'capturing...';
        } else {
          const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
          const mins = Math.floor(elapsed / 60)
            .toString()
            .padStart(2, '0');
          const secs = (elapsed % 60).toString().padStart(2, '0');
          timeEl.textContent = `${mins}:${secs}`;
        }
      }
    }, 1000);
  }

  private hideRecordingIndicator(): void {
    if (this.recordingTimeInterval) {
      clearInterval(this.recordingTimeInterval);
      this.recordingTimeInterval = null;
    }
    this.recordingIndicatorClickCleanup?.();
    if (this.recordingIndicator) {
      this.recordingIndicator.remove();
      this.recordingIndicator = null;
    }
  }

  // ========== Recording State Guard ==========

  /**
   * Save current DPR, renderer size, and resize-lock state, then apply
   * recording-safe defaults (disable adaptive DPR, lock resize, optionally
   * scale resolution). Call `restoreRecordingState()` to undo all changes.
   *
   * Used by captureScreenshot, startVideoRecording, and runOfflineCaptureLoop
   * to avoid duplicating the save/restore logic.
   */
  private saveRecordingState(options: {
    lockResize?: boolean;
    disableDPR?: boolean;
    scaleResolution?: { targetH: number; align16?: boolean };
  }): void {
    // Save current state
    const dprEnabled = this.adaptiveDPRManager?.isActive() ?? false;
    const dpr = this.adaptiveDPRManager?.getCurrentDPR() ?? window.devicePixelRatio;
    const renderer = this.sceneManager.renderer;
    const currentSize = renderer.getSize(new THREE.Vector2());
    this.savedRecordingState = {
      dprEnabled,
      dpr,
      rendererSize: null, // set below only if we actually scale
      resizeLocked: this.sceneManager.resizeLocked,
    };

    // Disable adaptive DPR — resolution changes during capture cause corruption
    if (options.disableDPR && this.adaptiveDPRManager) {
      this.adaptiveDPRManager.setEnabled(false);
      this.sceneManager.setAdaptivePixelRatio(this.adaptiveDPRManager.getNativeDPR());
    }

    // Lock resize to prevent mid-capture resolution changes
    if (options.lockResize) {
      this.sceneManager.resizeLocked = true;
    }

    // Scale renderer to target resolution.
    // Only call postProcessing.resize() — it internally calls renderer.setSize()
    // and composer.setSize() with correct SSAA handling. Calling renderer.setSize()
    // separately would be overridden by postProcessing.resize() anyway.
    if (options.scaleResolution) {
      this.savedRecordingState.rendererSize = { width: currentSize.x, height: currentSize.y };
      const { targetH, align16 } = options.scaleResolution;
      const aspect = currentSize.x / currentSize.y;
      let w = Math.round(targetH * aspect);
      let h = targetH;
      if (align16) {
        w = w & ~15;
        h = h & ~15;
      }
      renderer.setPixelRatio(1);
      this.sceneManager.postProcessing.resize(w, h);
      // postProcessing.resize() sets canvas CSS to the capture dimensions,
      // which shrinks the visible canvas and exposes the page background.
      // Override CSS back to full viewport — the WebGL framebuffer is already
      // at the correct capture resolution regardless of CSS size.
      const canvas = renderer.domElement;
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      // Update camera projection to match new aspect ratio
      const camera = this.sceneManager.camera;
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
      // Update material uniforms (viewport size, FOV) for the new resolution.
      // Without this, point/splat shaders use stale viewport dimensions and
      // render geometry at the wrong screen-space size (nearly invisible).
      this.sceneManager.updateMaterialsForCurrentCamera();
    }
  }

  /**
   * Restore DPR, renderer size, resize lock, and panels to pre-recording state.
   * Safe to call multiple times (no-ops if no saved state).
   */
  private restoreRecordingState(): void {
    const saved = this.savedRecordingState;
    if (!saved) return;

    // Restore renderer size and camera projection
    if (saved.rendererSize) {
      const renderer = this.sceneManager.renderer;
      renderer.setPixelRatio(window.devicePixelRatio);
      this.sceneManager.postProcessing.resize(saved.rendererSize.width, saved.rendererSize.height);
      const camera = this.sceneManager.camera;
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.aspect = saved.rendererSize.width / saved.rendererSize.height;
        camera.updateProjectionMatrix();
      }
      this.sceneManager.updateMaterialsForCurrentCamera();
    }

    // Restore resize lock
    this.sceneManager.resizeLocked = saved.resizeLocked;

    // Restore adaptive DPR
    if (this.adaptiveDPRManager) {
      if (saved.dprEnabled) {
        this.adaptiveDPRManager.setEnabled(true);
      } else {
        this.sceneManager.setAdaptivePixelRatio(saved.dpr);
      }
    }

    this.savedRecordingState = null;
    this.restoreAllPanels();
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

  /** Compute video bitrate based on canvas size, FPS, and quality preset */
  private computeVideoBitrate(width: number, height: number): number {
    return computeVideoBitratePure(width, height, this.options.videoFPS, this.options.videoQuality);
  }

  /**
   * Get a supported MIME type for real-time MediaRecorder capture.
   * MediaRecorder only supports WebM with VP9 or VP8 — H.264/H.265 are
   * NOT valid WebM codecs. The videoCodec option only applies to the
   * offline mediabunny path (turntable mode).
   */
  private getSupportedMimeType(): string | null {
    return getSupportedMimeTypePure();
  }

  /**
   * Generate the ffmpeg shell script bundled with image-sequence and
   * EXR-sequence ZIPs. The `ext` parameter is the per-frame extension
   * (`png` / `jpg` / `webp` / `exr`); ZipSequenceCapture passes the
   * matching value when packaging.
   */
  private generateFfmpegScript(fps: number, frameCount: number, ext = 'exr'): string {
    return generateFfmpegScriptPure(fps, frameCount, ext);
  }

  private generateFilename(ext: string): string {
    return generateFilenamePure(ext);
  }

  private downloadBlob(blob: Blob, filename: string): void {
    downloadBlobHelper(blob, filename);
  }
}
