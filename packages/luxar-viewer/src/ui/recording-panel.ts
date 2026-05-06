// Recording panel for screenshot and video capture of the WebGL canvas
// Uses browser-native APIs: canvas.toBlob() for screenshots,
// canvas.captureStream() + MediaRecorder for video recording (WebM)
// HDR export: EXR screenshots, EXR frame sequences (ZIP), 10-bit HDR video (WebCodecs)

import * as THREE from 'three';
import { Zip, ZipPassThrough } from 'fflate';
import GUI, { type Controller } from './gui';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import { showToast } from './helpers';
import { sceneDimsManager } from '../scene/scene-dims-manager';
// HDR video encoder kept for future use when browser 10-bit support matures
// import { isHDRVideoSupported, HDRVideoEncoder } from '../utils/hdr-video-encoder';
import {
  Output,
  WebMOutputFormat,
  Mp4OutputFormat,
  MkvOutputFormat,
  BufferTarget,
  VideoSampleSource,
  VideoSample,
  canEncodeVideo,
} from 'mediabunny';
import type { SceneManager } from '../scene/scene-manager';
import type { AnimationController } from '../scene/animation-controller';
import type { DimensionAnimationManager } from '../scene/dimension-animation-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import { LuxarOrbitControls } from '../controls/luxar-orbit-controls';
import type { OverlayManager } from './overlay-manager';
import {
  computeVideoBitrate as computeVideoBitratePure,
  generateFfmpegScript as generateFfmpegScriptPure,
  generateFilename as generateFilenamePure,
  getSupportedMimeType as getSupportedMimeTypePure,
} from './recording/media-utilities';
import {
  renderFrameToCanvas as renderFrameToCanvasHelper,
  encodeScreenshotBlob,
  normalizeScreenshotFormat,
  downloadBlob as downloadBlobHelper,
} from './recording/screenshot-exporter';

export type RecordingMode = 'image' | 'video' | 'turntable';

// Video quality presets — defined in media-utilities; re-exported here for
// existing consumers that import VideoQuality from this module.
import type { VideoQuality } from './recording/media-utilities';
export type { VideoQuality };

/** Video resolution presets — 0 means native canvas size */
export type VideoResolution = 0 | 1080 | 1440 | 2160;

/** Output format — what file type you get */
export type OutputFormat = 'png' | 'webp' | 'jpeg' | 'exr' | 'mp4' | 'webm' | 'mkv';

/** Video codec for MP4/WebM encoding (ordered by modernity) */
export type VideoCodecOption = 'h265' | 'vp9' | 'h264' | 'vp8';

/** Recording options for image and video capture */
export interface RecordingOptions {
  // Image options
  outputFormat: OutputFormat;
  imageQuality: number;
  maxDPR: boolean;
  transparentBackground: boolean;
  // Video options
  videoDurationLimit: number; // 0 = unlimited, else seconds
  videoFPS: number;
  videoCodec: VideoCodecOption;
  videoQuality: VideoQuality;
  videoResolution: VideoResolution; // target height in pixels, 0 = native
  syncToSlider: boolean;
  syncDimensionIndex: number; // -1 = none
  // Turntable options
  turntableSpeed: number; // degrees per second
  frameByFrame: boolean; // offline frame-by-frame capture (smooth but slow)
  // General
  showPanels: boolean;
  includeOverlays: boolean;
}

/** Panel visibility state snapshot for hide/restore */
export type PanelStates = Map<string, boolean>;

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
  private recordedChunks: Blob[] = [];
  private recordingIndicator: HTMLElement | null = null;
  private recordingTimeInterval: ReturnType<typeof setInterval> | null = null;
  private recordingStartTime: number = 0;
  private durationTimer: ReturnType<typeof setTimeout> | null = null;
  private keepAliveCallbackId = 'recording-keepalive';
  private turntableCallbackId = 'recording-turntable';
  private syncCompleteHandler: (() => void) | null = null;
  private syncPlayTimeout: ReturnType<typeof setTimeout> | null = null;
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

    this.offlineOverlayCleanup?.();
    this.hideRecordingIndicator();
    this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
    this.animationController.removePerFrameCallback(this.turntableCallbackId);
    this.cleanupSyncListener();
    this.restoreAutoRotate();
    this.restoreRecordingState();
    this.gui.destroy();
  }

  // ========== Screenshot Capture ==========

  async captureScreenshot(): Promise<void> {
    if (this.isCaptureInProgress) return;
    this.isCaptureInProgress = true;

    let savedBackground: THREE.Color | THREE.Texture | null = null;

    try {
      log.info(Modules.RECORDING, 'Capturing screenshot...');

      this.hideAllPanels();
      await new Promise((r) => requestAnimationFrame(r));

      // Save state (always — ensures restoreRecordingState restores panels)
      const wantMaxDPR = this.options.maxDPR && !!this.adaptiveDPRManager;
      this.saveRecordingState({ disableDPR: wantMaxDPR });
      if (wantMaxDPR) {
        await new Promise((r) => requestAnimationFrame(r));
      }

      // Set transparent background
      if (this.options.transparentBackground) {
        savedBackground = this.sceneManager.scene.background as THREE.Color | THREE.Texture | null;
        this.sceneManager.scene.background = null;
      }

      const format = this.options.outputFormat;

      if (format === 'exr') {
        const exrData = await this.sceneManager.postProcessing.captureHDRAsEXR();
        const blob = new Blob([exrData as BlobPart], { type: 'application/octet-stream' });
        this.downloadBlob(blob, this.generateFilename('exr'));
        showToast('HDR screenshot saved (EXR)');
      } else {
        const captureCanvas = this.renderFrameToCanvas();

        const { format: effectiveFormat, warning } = normalizeScreenshotFormat(
          format,
          this.options.transparentBackground
        );
        if (warning === 'video-fallback') {
          log.warning(
            Modules.RECORDING,
            `Screenshot format '${format}' is a video format, falling back to PNG`
          );
        } else if (warning === 'jpeg-no-alpha') {
          showToast('Switched to PNG (JPEG has no alpha)');
        }

        const blob = await encodeScreenshotBlob(
          captureCanvas,
          effectiveFormat,
          this.options.imageQuality
        );

        if (blob) {
          this.downloadBlob(blob, this.generateFilename(effectiveFormat));
          showToast('Screenshot saved');
        } else {
          showToast('Screenshot failed');
        }
      }
    } finally {
      if (savedBackground !== null) {
        this.sceneManager.scene.background = savedBackground;
      }
      this.restoreRecordingState();
      this.isCaptureInProgress = false;
    }
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
      if (this.disposed) return;
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

    const stream = canvas.captureStream(this.options.videoFPS);
    this.mediaRecorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond });
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

  // ========== EXR Sequence Recording ==========

  /**
   * Run a deterministic offline capture loop for HDR recording (EXR or HDR video).
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
  private async runOfflineCaptureLoop(
    mode: 'exr' | 'webm' | 'mp4' | 'mkv' | 'png' | 'webp' | 'jpeg'
  ): Promise<void> {
    const confirmed = await this.showConfirmationDialog();
    if (!confirmed || this.disposed) return;

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

    // Compute turntable parameters
    const fps = this.options.videoFPS;
    const durationSeconds = 360 / this.options.turntableSpeed;
    const totalFrames = Math.ceil(durationSeconds * fps);

    const controls = this.sceneManager.controls.getControls();
    if (!(controls instanceof LuxarOrbitControls)) {
      log.warning(Modules.RECORDING, 'Turntable requires orbit controls');
      this.restoreRecordingState();
      return;
    }

    // Pause auto-rotation so it doesn't compound with the turntable
    this.savedAutoRotate = this.sceneManager.controls.getAutoRotate();
    this.sceneManager.controls.setAutoRotate(false);

    // Per-frame rotation step: frame 0 captures the starting view without rotation,
    // then frames 1..N-1 each advance by one step to complete exactly 2π total.
    const anglePerFrame = totalFrames > 1 ? (2 * Math.PI) / (totalFrames - 1) : 0;

    log.info(
      Modules.RECORDING,
      `Starting offline ${mode} capture: ${totalFrames} frames, ${fps} FPS, ${durationSeconds.toFixed(1)}s`
    );

    // Set recording state
    if (mode === 'exr') {
      this.isEXRSequenceRecording = true;
    }
    this.isRecording = true;
    this.isOfflineCaptureActive = true;
    this.recordingStartTime = Date.now();
    this.showRecordingIndicator();

    // Block user interaction during offline capture with a modal overlay.
    // Camera manipulation or resize during frame-by-frame capture would corrupt the sequence.
    // Includes a live preview canvas so the user can see each frame as it's captured.
    const overlay = document.createElement('div');
    overlay.className = 'luxar-recording-overlay';
    overlay.innerHTML = `
      <div class="luxar-recording-overlay__content">
        <canvas class="luxar-recording-overlay__preview"></canvas>
        <div class="luxar-recording-overlay__progress">
          <span class="luxar-recording-overlay__label">Capturing frames...</span>
          <span class="luxar-recording-overlay__counter">0/${totalFrames}</span>
        </div>
        <button class="luxar-recording-overlay__cancel">Cancel</button>
      </div>
    `;
    const previewCanvas = overlay.querySelector(
      '.luxar-recording-overlay__preview'
    ) as HTMLCanvasElement;
    const previewCtx = previewCanvas.getContext('2d');
    const cancelButton = overlay.querySelector('.luxar-recording-overlay__cancel');
    const handleCancel = (): void => {
      this.isRecording = false;
    };
    const stopOverlayKeydown = (e: KeyboardEvent): void => e.stopPropagation();
    cancelButton?.addEventListener('click', handleCancel);
    // Block all pointer/keyboard events from reaching the canvas
    overlay.addEventListener('keydown', stopOverlayKeydown, true);

    let overlayCleaned = false;
    const cleanupOfflineOverlay = (): void => {
      if (overlayCleaned) return;
      overlayCleaned = true;
      cancelButton?.removeEventListener('click', handleCancel);
      overlay.removeEventListener('keydown', stopOverlayKeydown, true);
      overlay.remove();
      if (this.offlineOverlayCleanup === cleanupOfflineOverlay) {
        this.offlineOverlayCleanup = null;
      }
    };
    this.offlineOverlayCleanup = cleanupOfflineOverlay;

    document.body.appendChild(overlay);

    const counterEl = overlay.querySelector('.luxar-recording-overlay__counter');
    const labelEl = overlay.querySelector('.luxar-recording-overlay__label');

    // For WebM/MP4: set up mediabunny encoder (NOT captureStream — WebGL canvases
    // with preserveDrawingBuffer:false don't work reliably with captureStream)
    let videoOutput: Output | null = null;
    let videoTarget: BufferTarget | null = null;
    let videoSource: VideoSampleSource | null = null;
    const isVideoMode = mode === 'webm' || mode === 'mp4' || mode === 'mkv';
    let videoExt = 'webm';
    let videoMime = 'video/webm';
    if (isVideoMode) {
      const canvas = this.sceneManager.renderer.domElement;
      const videoBitsPerSecond = this.computeVideoBitrate(canvas.width, canvas.height);

      // Map user codec selection to mediabunny codec names
      type MBCodec = 'av1' | 'vp9' | 'avc' | 'hevc' | 'vp8';
      const codecMap: Record<VideoCodecOption, MBCodec> = {
        h265: 'hevc',
        vp9: 'vp9',
        h264: 'avc',
        vp8: 'vp8',
      };
      const encOpts = { width: canvas.width, height: canvas.height, bitrate: videoBitsPerSecond };
      let codec: MBCodec = codecMap[this.options.videoCodec];

      // Check support and fall back
      // WebM only supports vp9, av1, vp8 — avc/hevc require MP4 container
      const webmOnly: MBCodec[] = ['vp9', 'av1', 'vp8'];
      if (mode === 'webm' && !webmOnly.includes(codec)) {
        // Force codec to a WebM-compatible one before even checking support
        codec = 'vp9';
      }
      if (!(await canEncodeVideo(codec, encOpts))) {
        const allFallbacks: MBCodec[] =
          codec === 'hevc' ? ['avc', 'vp9', 'av1', 'vp8'] : ['vp9', 'av1', 'vp8', 'avc'];
        const fallbacks =
          mode === 'webm' ? allFallbacks.filter((c) => webmOnly.includes(c)) : allFallbacks;
        let found = false;
        for (const fb of fallbacks) {
          if (await canEncodeVideo(fb, encOpts)) {
            log.warning(Modules.RECORDING, `${codec} not supported, falling back to ${fb}`);
            codec = fb;
            found = true;
            break;
          }
        }
        if (!found) {
          showToast('No supported video codec at this resolution');
          overlay.remove();
          this.restoreRecordingState();
          return;
        }
      }

      const format =
        mode === 'mp4'
          ? new Mp4OutputFormat()
          : mode === 'mkv'
            ? new MkvOutputFormat()
            : new WebMOutputFormat();
      videoExt = mode;
      videoMime = mode === 'mp4' ? 'video/mp4' : mode === 'mkv' ? 'video/x-matroska' : 'video/webm';
      videoTarget = new BufferTarget();
      videoSource = new VideoSampleSource({
        codec,
        bitrate: videoBitsPerSecond,
        latencyMode: 'quality',
      });
      videoOutput = new Output({ format, target: videoTarget });
      videoOutput.addVideoTrack(videoSource, { frameRate: fps });
      await videoOutput.start();
      log.info(
        Modules.RECORDING,
        `Offline ${videoExt.toUpperCase()} encoder started (${codec.toUpperCase()}, ` +
          `${canvas.width}x${canvas.height}, ${Math.round(videoBitsPerSecond / 1_000_000)}Mbps)`
      );
    }

    // Frame-by-frame capture using the live animation loop.
    //
    // The animation loop runs: controls.update() → per-frame callbacks → render().
    // We register a per-frame callback that applies an incremental quaternion rotation
    // (same math as auto-rotate), then the animation loop renders and we read pixels.
    //
    // This guarantees we read pixels from a properly rendered frame — the same
    // pipeline that produces visible on-screen output.
    let captureCanvas: HTMLCanvasElement | null = null;
    let capturedFrames = 0;
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 3;
    const captureCallbackId = 'recording-offline-capture';

    // Streaming ZIP for image/EXR sequences: frames are written incrementally.
    // When the File System Access API is available (Chrome/Edge), ZIP chunks are
    // flushed directly to disk via showSaveFilePicker, keeping memory usage O(1).
    // Otherwise chunks are collected and passed to `new Blob(chunks)` which avoids
    // a single contiguous allocation (the old approach's OOM trigger).
    const isImageMode = mode === 'png' || mode === 'webp' || mode === 'jpeg';
    const isZipMode = isImageMode || mode === 'exr';
    const zipChunks: Uint8Array[] = [];
    let zipTotalBytes = 0;
    let zipDiskFailed = false;
    let zipWritable: FileSystemWritableFileStream | null = null;
    let streamingZip: InstanceType<typeof Zip> | null = null;
    if (isZipMode) {
      // Try to get a file handle for true disk-streaming (user gesture context
      // from the confirmation dialog click propagates here).
      try {
        const w = window as unknown as {
          showSaveFilePicker?: (opts: unknown) => Promise<FileSystemFileHandle>;
        };
        if (typeof w.showSaveFilePicker === 'function') {
          const handle = await w.showSaveFilePicker({
            suggestedName: this.generateFilename('zip'),
            types: [{ description: 'ZIP archive', accept: { 'application/zip': ['.zip'] } }],
          });
          zipWritable = await handle.createWritable();
        }
      } catch {
        // User cancelled or API unavailable — fall back to in-memory Blob
        zipWritable = null;
      }
      streamingZip = new Zip((err, chunk, _final) => {
        if (err) {
          if (!zipDiskFailed) {
            zipDiskFailed = true;
            log.error(Modules.RECORDING, `ZIP stream error: ${err}`);
          }
          return;
        }
        if (chunk) {
          if (zipWritable && !zipDiskFailed) {
            zipWritable.write(chunk as BlobPart).catch((writeErr) => {
              if (!zipDiskFailed) {
                zipDiskFailed = true;
                log.error(
                  Modules.RECORDING,
                  `Disk write failed: ${writeErr}. ` +
                    'Free browser storage or use video format instead of image sequence.'
                );
              }
            });
          } else if (!zipDiskFailed) {
            zipChunks.push(chunk);
          }
          zipTotalBytes += chunk.length;
        }
      });
    }

    // Keep the animation loop alive during the entire capture session.
    // This is a separate no-op callback so we can safely add/remove the rotation
    // callback each frame without risking the animation loop pausing mid-capture.
    const keepAliveId = 'recording-offline-keepalive';
    this.animationController.addPerFrameCallback(keepAliveId, () => {}, { continuous: true });

    /** Add a single frame to the streaming ZIP archive. */
    const addZipFrame = (data: Uint8Array, ext: string): void => {
      const padded = String(capturedFrames).padStart(6, '0');
      const entry = new ZipPassThrough(`frame_${padded}.${ext}`);
      streamingZip!.add(entry);
      entry.push(data, true);
    };

    for (let i = 0; i < totalFrames; i++) {
      if (!this.isRecording || zipDiskFailed) break;

      // Orbit camera by one step and capture in a single animation frame.
      // The callback applies a quaternion rotation (same as auto-rotate) AFTER
      // controls.update, BEFORE render, so the frame is rendered at the new angle.
      //
      // IMPORTANT: The rotation callback is registered fresh each iteration and
      // removed immediately after the frame renders. If it stayed registered
      // (continuous: true), the animation loop would apply extra rotations during
      // the async capture work (toBlob, arrayBuffer, etc.) between iterations.
      this.animationController.addPerFrameCallback(captureCallbackId, () => {
        if (i > 0) controls.applyOrbitRotation(anglePerFrame);
      });

      // Wait for one full animation frame (callback + render)
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

      // Remove the rotation callback immediately so the animation loop cannot
      // apply extra rotations while we do async capture work below.
      this.animationController.removePerFrameCallback(captureCallbackId);

      // The animation loop has rendered with the rotated camera.
      // Now capture the pixels from the default framebuffer.
      try {
        if (isImageMode && streamingZip) {
          captureCanvas = this.renderFrameToCanvas();
          const mimeType = mode === 'jpeg' ? 'image/jpeg' : `image/${mode}`;
          const quality = mode === 'png' ? undefined : this.options.imageQuality;
          const blob = await new Promise<Blob | null>((resolve) =>
            captureCanvas!.toBlob(resolve, mimeType, quality)
          );
          if (blob) {
            const buf = new Uint8Array(await blob.arrayBuffer());
            addZipFrame(buf, mode === 'jpeg' ? 'jpg' : mode);
          }
        } else if (isVideoMode && videoSource) {
          captureCanvas = this.renderFrameToCanvas();
          const frameDuration = 1 / fps;
          const sample = new VideoSample(captureCanvas, {
            timestamp: i * frameDuration,
            duration: frameDuration,
          });
          await videoSource.add(sample);
          sample.close();
        } else if (mode === 'exr' && streamingZip) {
          const exrData = await this.sceneManager.postProcessing.captureHDRAsEXR();
          addZipFrame(exrData, 'exr');
        }
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

      // Update progress overlay + live preview
      if (counterEl) counterEl.textContent = `${i + 1}/${totalFrames}`;
      if (captureCanvas && previewCtx) {
        if (i === 0) {
          previewCanvas.width = captureCanvas.width;
          previewCanvas.height = captureCanvas.height;
        }
        previewCtx.drawImage(captureCanvas, 0, 0);
      }
    }

    // Remove capture callbacks
    this.animationController.removePerFrameCallback(captureCallbackId);
    this.animationController.removePerFrameCallback(keepAliveId);

    // Finalize
    this.hideRecordingIndicator();
    this.isRecording = false;
    this.isOfflineCaptureActive = false;

    /** Finalize a streamed ZIP: add ffmpeg script, close stream, save or download. */
    const finalizeZipSequence = async (ext: string, label: string): Promise<void> => {
      if (zipDiskFailed) {
        // Disk write failed — abort the partial file and inform the user
        streamingZip!.end();
        if (zipWritable) {
          try {
            await zipWritable.abort();
          } catch {
            /* already closed/aborted */
          }
        }
        showToast(
          'Recording failed: disk storage quota exceeded. ' +
            'Free browser storage, reduce duration/resolution, or use video format.'
        );
        return;
      }
      if (capturedFrames > 0) {
        if (labelEl) labelEl.textContent = 'Packaging ZIP...';
        await new Promise((r) => requestAnimationFrame(r));
        const encoder = new TextEncoder();
        const scriptEntry = new ZipPassThrough('encode_video.sh');
        streamingZip!.add(scriptEntry);
        scriptEntry.push(encoder.encode(this.generateFfmpegScript(fps, capturedFrames, ext)), true);
        streamingZip!.end();
        if (zipWritable) {
          await zipWritable.close();
          showToast(
            `${label} sequence saved to disk (${capturedFrames} frames, ` +
              `${(zipTotalBytes / (1024 * 1024)).toFixed(1)} MB)`
          );
        } else {
          const blob = new Blob(zipChunks as BlobPart[], { type: 'application/zip' });
          this.downloadBlob(blob, this.generateFilename('zip'));
          showToast(`${label} sequence saved (${capturedFrames} frames)`);
        }
      } else {
        streamingZip!.end();
        if (zipWritable) {
          try {
            await zipWritable.abort();
          } catch {
            /* already closed/aborted */
          }
        }
        showToast('No frames captured');
      }
    };

    if (isImageMode && streamingZip) {
      await finalizeZipSequence(mode === 'jpeg' ? 'jpg' : mode, mode.toUpperCase());
    } else if (isVideoMode) {
      if (videoOutput && capturedFrames > 0) {
        if (labelEl) labelEl.textContent = 'Finalizing video...';
        if (counterEl) counterEl.textContent = '';
        try {
          await videoOutput.finalize();
          const buffer = videoTarget?.buffer;
          if (buffer) {
            const blob = new Blob([buffer], { type: videoMime });
            log.info(
              Modules.RECORDING,
              `Offline ${videoExt.toUpperCase()} capture: ${capturedFrames} frames, ${(blob.size / (1024 * 1024)).toFixed(1)} MB`
            );
            this.downloadBlob(blob, this.generateFilename(videoExt));
            showToast(`Video saved (${capturedFrames} frames)`);
          } else {
            showToast('Video encoding produced no output');
          }
        } catch (err) {
          log.error(Modules.RECORDING, `Video finalization failed: ${err}`);
          showToast('Video encoding failed');
        }
      } else {
        showToast('No frames captured');
      }
    } else if (mode === 'exr' && streamingZip) {
      this.isEXRSequenceRecording = false;
      await finalizeZipSequence('exr', 'EXR');
    }

    // Remove overlay, restore all saved state
    cleanupOfflineOverlay();
    this.restoreAutoRotate();
    this.restoreRecordingState();
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
    const dimIndex = this.options.syncDimensionIndex;
    if (dimIndex < 0 || !this.animationManager) return;

    this.cleanupSyncListener();

    const ranges = sceneDimsManager.getDimensionRanges();
    if (ranges && ranges[dimIndex]) {
      const [min] = ranges[dimIndex];
      sceneDimsManager.setDimensionValue(dimIndex, min);
    }

    this.syncCompleteHandler = () => {
      log.info(Modules.RECORDING, 'Slider sync complete — stopping recording');
      this.stopVideoRecording();
    };
    this.animationManager.addEventListener('complete', this.syncCompleteHandler as any);

    // Small delay to let the initial position update propagate.
    this.syncPlayTimeout = setTimeout(() => {
      this.syncPlayTimeout = null;
      if (!this.disposed) {
        this.animationManager?.play(dimIndex, { loopMode: 'once', direction: 'forward' });
      }
    }, 100);
  }

  private cleanupSyncListener(): void {
    if (this.syncPlayTimeout !== null) {
      clearTimeout(this.syncPlayTimeout);
      this.syncPlayTimeout = null;
    }
    if (this.syncCompleteHandler && this.animationManager) {
      this.animationManager.removeEventListener('complete', this.syncCompleteHandler as any);
      this.syncCompleteHandler = null;
    }
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
    const duration = 360 / this.options.turntableSpeed;
    const frames = Math.ceil(duration * this.options.videoFPS);
    return `${duration.toFixed(1)}s, ${frames} frames`;
  }

  /** Get navigable dimension names as dropdown options */
  private getNavigableDimensionOptions(): Record<string, number> {
    const options: Record<string, number> = {};
    const dims = sceneDimsManager.getDims();
    if (dims) {
      const names = sceneDimsManager.getDimensionNames();
      for (let i = 0; i < dims.ndim; i++) {
        if (!dims.displayed.includes(i)) {
          options[names[i] || `dim ${i}`] = i;
        }
      }
    }
    if (Object.keys(options).length === 0) {
      options['(no dimensions)'] = -1;
    }
    return options;
  }

  /** Show/hide controls based on current mode */
  private updateControlVisibility(): void {
    const isImage = this.mode === 'image';
    const isVideo = this.mode === 'video';
    const isTurntable = this.mode === 'turntable';

    // Filter format dropdown options based on mode:
    // - Image: PNG, WebP, JPEG, EXR (screenshots)
    // - Video: WebM only (real-time MediaRecorder outputs WebM)
    // - Turntable: image formats (→ ZIP) + video formats (→ offline encode)
    const imageFormats = ['png', 'webp', 'jpeg', 'exr'];
    const videoFormats = ['webm'];
    const turntableFormats = ['png', 'webp', 'jpeg', 'exr', 'mp4', 'webm', 'mkv'];
    const validFormats = isImage ? imageFormats : isTurntable ? turntableFormats : videoFormats;

    // Show/hide <option> elements in the format dropdown.
    // Note: our GUI uses the display label as option.value (e.g., "PNG" not "png"),
    // and maps labels→values internally. We match by label→value mapping.
    const labelToValue: Record<string, string> = {
      PNG: 'png',
      WebP: 'webp',
      JPEG: 'jpeg',
      EXR: 'exr',
      MP4: 'mp4',
      WebM: 'webm',
      MKV: 'mkv',
    };
    const selectEl = this.formatController?.domElement.querySelector(
      'select'
    ) as HTMLSelectElement | null;
    if (selectEl?.options) {
      for (const opt of Array.from(selectEl.options)) {
        const val = labelToValue[opt.value] || opt.value;
        opt.hidden = !validFormats.includes(val);
      }
    }

    // Auto-correct if current format is invalid for this mode
    if (!validFormats.includes(this.options.outputFormat)) {
      this.options.outputFormat = isVideo ? 'webm' : isImage ? 'webp' : 'mp4';
      this.formatController?.updateDisplay();
    }
    const fmt = this.options.outputFormat;

    // Format selector is always visible
    this.formatController?.show();

    // Image-only controls (quality, max DPR, transparent BG)
    for (const ctrl of this.imageControllers) {
      isImage ? ctrl.show() : ctrl.hide();
    }
    if (isImage) {
      // Hide quality slider for lossless/HDR formats
      // (mp4/webm cannot occur in Image mode — auto-correction above prevents it)
      if (fmt === 'png' || fmt === 'exr') {
        this.qualityController?.hide();
      }
      // Hide transparent BG for HDR format (EXR always has alpha)
      if (fmt === 'exr') {
        this.transparentController?.hide();
      }
    }

    for (const ctrl of this.videoControllers) {
      isVideo || isTurntable ? ctrl.show() : ctrl.hide();
    }
    // Codec dropdown: only show for video formats (MP4/WebM/MKV)
    const isVideoFormat = fmt === 'mp4' || fmt === 'webm' || fmt === 'mkv';
    if (!isVideoFormat) {
      this.videoCodecController?.hide();
    }
    // Filter codec options by mode: MediaRecorder (Video) only supports VP9/VP8.
    // Turntable (mediabunny) supports all codecs.
    if (isVideo && isVideoFormat) {
      const codecLabelToValue: Record<string, string> = {
        'H.265': 'h265',
        VP9: 'vp9',
        'H.264': 'h264',
        VP8: 'vp8',
      };
      const mediaRecorderCodecs = ['vp9', 'vp8'];
      const codecSelect = this.videoCodecController?.domElement.querySelector(
        'select'
      ) as HTMLSelectElement | null;
      if (codecSelect?.options) {
        for (const opt of Array.from(codecSelect.options)) {
          const val = codecLabelToValue[opt.value] || opt.value;
          opt.hidden = !mediaRecorderCodecs.includes(val);
        }
      }
    } else if (isTurntable && isVideoFormat) {
      // Turntable: show all codec options (mediabunny supports all)
      const codecSelect = this.videoCodecController?.domElement.querySelector(
        'select'
      ) as HTMLSelectElement | null;
      if (codecSelect?.options) {
        for (const opt of Array.from(codecSelect.options)) {
          opt.hidden = false;
        }
      }
    }
    // Video quality: hide only for image sequence formats where bitrate is irrelevant.
    // Both MediaRecorder (Video mode) and mediabunny (Turntable video) use computeVideoBitrate().
    const isImageSequenceFormat =
      fmt === 'exr' || fmt === 'png' || fmt === 'webp' || fmt === 'jpeg';
    if ((isVideo || isTurntable) && isImageSequenceFormat) {
      this.videoQualityController?.hide();
    }
    // Turntable: hide duration limit and sync (turntable has its own computed duration from speed)
    if (isTurntable) {
      this.videoDurationController?.hide();
      this.syncToggleController?.hide();
      this.syncDimensionController?.hide();
    }
    // Hide sync dimension dropdown unless sync is enabled
    if (isVideo && !this.options.syncToSlider) {
      this.syncDimensionController?.hide();
    }

    for (const ctrl of this.turntableControllers) {
      isTurntable ? ctrl.show() : ctrl.hide();
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
          <div class="luxar-recording-confirm__title">Start ${this.mode === 'turntable' ? 'Turntable' : 'Video'} Recording</div>
          <p class="luxar-recording-confirm__message">
            ${details}<br>
            Press <span class="luxar-recording-confirm__keybinding">Escape</span> to stop recording.
          </p>
          <div class="luxar-recording-confirm__buttons">
            <button class="luxar-recording-confirm__btn" data-action="cancel">Cancel</button>
            <button class="luxar-recording-confirm__btn luxar-recording-confirm__btn--primary" data-action="start">Start Recording</button>
          </div>
        </div>
      `;

      let settled = false;
      const finish = (result: boolean): void => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(result);
      };

      const trapKeyboard = (e: KeyboardEvent) => {
        e.stopPropagation();
        if (e.key === 'Escape') {
          finish(false);
        } else if (e.key === 'Enter') {
          finish(true);
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
      overlay.tabIndex = -1;
      overlay.focus();
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
  private renderFrameToCanvas(): HTMLCanvasElement {
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

  /** Generate a shell script with ffmpeg commands to encode the EXR sequence */
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
