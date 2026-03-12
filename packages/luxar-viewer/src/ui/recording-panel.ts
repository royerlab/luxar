// Recording panel for screenshot and video capture of the WebGL canvas
// Uses browser-native APIs: canvas.toBlob() for screenshots,
// canvas.captureStream() + MediaRecorder for video recording (WebM)

import * as THREE from 'three';
import GUI, { type Controller } from './gui';
import { config } from '../config';
import { log, Modules } from '../utils/log';
import { showToast } from './helpers';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import type { SceneManager } from '../scene/scene-manager';
import type { AnimationController } from '../scene/animation-controller';
import type { DimensionAnimationManager } from '../scene/dimension-animation-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';

type RecordingMode = 'image' | 'video' | 'turntable';

/** Recording options for image and video capture */
interface RecordingOptions {
  // Image options
  imageFormat: 'png' | 'webp' | 'jpeg';
  imageQuality: number;
  maxDPR: boolean;
  transparentBackground: boolean;
  // Video options
  videoDurationLimit: number; // 0 = unlimited, else seconds
  videoFPS: number;
  videoCodec: 'vp9' | 'vp8';
  syncToSlider: boolean;
  syncDimensionIndex: number; // -1 = none
  // Turntable options
  turntableSpeed: number; // degrees per second
  // General
  showPanels: boolean;
}

/** Panel visibility state snapshot for hide/restore */
type PanelStates = Map<string, boolean>;

/**
 * Recording panel for capturing screenshots and recording video.
 *
 * Features:
 * - Screenshot with transparent background and max DPR
 * - Video recording with slider sync
 * - Turntable 360° rotation recording
 *
 * Uses zero external dependencies — all browser-native APIs.
 */
export class RecordingPanel {
  private gui: GUI;
  private visible: boolean = false;
  private mode: RecordingMode = 'image';
  private options: RecordingOptions = {
    imageFormat: 'webp',
    imageQuality: 0.92,
    maxDPR: true,
    transparentBackground: false,
    videoDurationLimit: 0,
    videoFPS: 30,
    videoCodec: 'vp9',
    syncToSlider: false,
    syncDimensionIndex: -1,
    turntableSpeed: 36,
    showPanels: false,
  };

  private sceneManager: SceneManager;
  private animationController: AnimationController;

  // Optional dependencies (set via setters)
  private animationManager: DimensionAnimationManager | null = null;
  private adaptiveDPRManager: AdaptiveDPRManager | null = null;

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
  private syncToggleController: Controller | null = null;
  private syncDimensionController: Controller | null = null;

  constructor(sceneManager: SceneManager, animationController: AnimationController) {
    this.sceneManager = sceneManager;
    this.animationController = animationController;

    this.gui = new GUI({
      title: 'Recording',
      width: 280,
      closeFolders: false,
      onClose: () => this.hide(),
    });

    this.gui.domElement.classList.add('luxar-recording-panel');

    Object.assign(this.gui.domElement.style, {
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

  dispose(): void {
    this.disposed = true;
    if (this.isRecording) {
      this.stopVideoRecording();
    }
    this.hideRecordingIndicator();
    this.gui.destroy();
  }

  // ========== Screenshot Capture ==========

  async captureScreenshot(): Promise<void> {
    if (this.isCaptureInProgress) return;
    this.isCaptureInProgress = true;

    // Save state for restore
    let savedBackground: THREE.Color | THREE.Texture | null = null;
    let savedDPREnabled: boolean | null = null;
    let savedDPR: number | null = null;

    try {
      log.info(Modules.RECORDING, 'Capturing screenshot...');

      this.hideAllPanels();
      await new Promise((r) => requestAnimationFrame(r));

      // Maximize DPR for highest resolution
      if (this.options.maxDPR && this.adaptiveDPRManager) {
        savedDPREnabled = this.adaptiveDPRManager.isActive();
        savedDPR = this.adaptiveDPRManager.getCurrentDPR();
        const nativeDPR = this.adaptiveDPRManager.getNativeDPR();
        this.adaptiveDPRManager.setEnabled(false);
        this.sceneManager.setAdaptivePixelRatio(nativeDPR);
        // Wait one more frame for resize to take effect
        await new Promise((r) => requestAnimationFrame(r));
      }

      // Set transparent background
      if (this.options.transparentBackground) {
        savedBackground = this.sceneManager.scene.background as THREE.Color | THREE.Texture | null;
        this.sceneManager.scene.background = null;
      }

      // Force render through the full post-processing pipeline
      this.sceneManager.postProcessing.render();

      // Capture canvas. toBlob() captures pixel data synchronously at call time
      // (only encoding is async), so this works despite preserveDrawingBuffer: false.
      const canvas = this.sceneManager.renderer.domElement;

      // Auto-switch from JPEG to PNG if transparent background (JPEG has no alpha)
      let format = this.options.imageFormat;
      if (this.options.transparentBackground && format === 'jpeg') {
        format = 'png';
        showToast('Switched to PNG (JPEG has no alpha)');
      }
      const mimeType = `image/${format === 'jpeg' ? 'jpeg' : format}`;
      const quality = format === 'png' ? undefined : this.options.imageQuality;

      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, mimeType, quality)
      );

      if (!blob) {
        log.warning(Modules.RECORDING, 'toBlob returned null, falling back to PNG');
        this.sceneManager.postProcessing.render();
        const fallbackBlob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob(resolve, 'image/png')
        );
        if (fallbackBlob) {
          this.downloadBlob(fallbackBlob, this.generateFilename('png'));
          showToast('Screenshot saved (PNG fallback)');
        } else {
          showToast('Screenshot failed');
        }
      } else {
        this.downloadBlob(blob, this.generateFilename(format));
        showToast('Screenshot saved');
      }
    } finally {
      // Restore transparent background (savedBackground is non-null only if we changed it)
      if (savedBackground !== null) {
        this.sceneManager.scene.background = savedBackground;
      }

      // Restore DPR
      if (savedDPREnabled !== null && this.adaptiveDPRManager) {
        if (savedDPREnabled) {
          this.adaptiveDPRManager.setEnabled(true);
        } else {
          this.sceneManager.setAdaptivePixelRatio(savedDPR!);
        }
      }

      this.restoreAllPanels();
      this.isCaptureInProgress = false;
    }
  }

  // ========== Video Recording ==========

  async startVideoRecording(): Promise<void> {
    if (this.isRecording) return;

    const mimeType = this.getSupportedMimeType();
    if (!mimeType) {
      showToast('Video recording not supported in this browser');
      return;
    }

    const confirmed = await this.showConfirmationDialog();
    if (!confirmed) return;

    log.info(
      Modules.RECORDING,
      `Starting video recording (${mimeType}, ${this.options.videoFPS} FPS)`
    );

    this.hideAllPanels();
    this.animationController.startAnimation();
    this.animationController.addPerFrameCallback(this.keepAliveCallbackId, () => {});

    const canvas = this.sceneManager.renderer.domElement;
    const stream = canvas.captureStream(this.options.videoFPS);

    this.mediaRecorder = new MediaRecorder(stream, { mimeType });
    this.recordedChunks = [];

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        this.recordedChunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      log.info(
        Modules.RECORDING,
        `Recording stopped, ${this.recordedChunks.length} chunks collected`
      );
      const blob = new Blob(this.recordedChunks, { type: mimeType });
      this.downloadBlob(blob, this.generateFilename('webm'));
      this.recordedChunks = [];
      this.isRecording = false;
      this.hideRecordingIndicator();

      if (!this.disposed) {
        this.animationController.removePerFrameCallback(this.keepAliveCallbackId);
        this.animationController.removePerFrameCallback(this.turntableCallbackId);
        this.cleanupSyncListener();
        this.restoreAllPanels();
        showToast('Video saved');
      }
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
    if (!this.isRecording || !this.mediaRecorder) return;
    log.info(Modules.RECORDING, 'Stopping video recording...');

    if (this.durationTimer) {
      clearTimeout(this.durationTimer);
      this.durationTimer = null;
    }

    this.mediaRecorder.stop();
  }

  // ========== Slider Sync ==========

  private startSliderSync(): void {
    const dimIndex = this.options.syncDimensionIndex;
    if (dimIndex < 0 || !this.animationManager) return;

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

    // Small delay to let the initial position update propagate
    setTimeout(() => {
      this.animationManager?.play(dimIndex, { loopMode: 'once', direction: 'forward' });
    }, 100);
  }

  private cleanupSyncListener(): void {
    if (this.syncCompleteHandler && this.animationManager) {
      this.animationManager.removeEventListener('complete', this.syncCompleteHandler as any);
      this.syncCompleteHandler = null;
    }
  }

  // ========== Turntable Rotation ==========

  private startTurntableRotation(): void {
    const camera = this.sceneManager.camera;
    const controls = this.sceneManager.controls.getControls();
    const target = (controls as any)?.target?.clone() ?? new THREE.Vector3();
    const offset = new THREE.Vector3().subVectors(camera.position, target);
    const radius = Math.sqrt(offset.x * offset.x + offset.z * offset.z);
    const startAngle = Math.atan2(offset.z, offset.x);
    const startY = camera.position.y;
    const totalDuration = (360 / this.options.turntableSpeed) * 1000;
    const startTime = Date.now();

    let turntableDone = false;
    this.animationController.addPerFrameCallback(this.turntableCallbackId, () => {
      if (turntableDone) return;
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / totalDuration, 1);
      const angle = startAngle + progress * Math.PI * 2;

      camera.position.x = target.x + radius * Math.cos(angle);
      camera.position.z = target.z + radius * Math.sin(angle);
      camera.position.y = startY;
      camera.lookAt(target);

      if (progress >= 1) {
        turntableDone = true;
        this.stopVideoRecording();
      }
    });
  }

  // ========== GUI Construction ==========

  private buildGUI(): void {
    // Capture/Record button at the top
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

    // Mode toggle
    const modeObj = { mode: this.mode };
    this.gui
      .add(modeObj, 'mode', { Image: 'image', Video: 'video', Turntable: 'turntable' })
      .name('Mode')
      .onChange((val: string) => {
        this.mode = val as RecordingMode;
        this.updateControlVisibility();
      });

    // Show panels toggle
    const panelSettings = { showPanels: this.options.showPanels };
    const showPanelsCtrl = this.gui
      .add(panelSettings, 'showPanels')
      .name('Show Panels')
      .onChange((val: boolean) => {
        this.options.showPanels = val;
      });
    showPanelsCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Keep other panels visible during capture');

    // Advanced Options folder (starts closed)
    const advanced = this.gui.addFolder('Advanced Options');
    advanced.close();

    // ── Image options ──
    const imgSettings = {
      format: this.options.imageFormat,
      quality: this.options.imageQuality,
      maxDPR: this.options.maxDPR,
      transparentBg: this.options.transparentBackground,
    };

    const formatCtrl = advanced
      .add(imgSettings, 'format', { PNG: 'png', WebP: 'webp', JPEG: 'jpeg' })
      .name('Image Format')
      .onChange((val: string) => {
        this.options.imageFormat = val as 'png' | 'webp' | 'jpeg';
        if (val === 'png') {
          qualityCtrl.hide();
        } else {
          qualityCtrl.show();
        }
      });
    this.imageControllers.push(formatCtrl);

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

    // ── Video options ──
    const vidSettings = {
      duration: this.options.videoDurationLimit,
      fps: this.options.videoFPS,
      codec: this.options.videoCodec,
      syncToSlider: this.options.syncToSlider,
      syncDim: this.options.syncDimensionIndex,
    };

    const durationCtrl = advanced
      .add(vidSettings, 'duration', 0, 300, 1)
      .name('Duration (s)')
      .onChange((val: number) => {
        this.options.videoDurationLimit = val;
      });
    durationCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Recording duration limit in seconds (0 = unlimited)');
    this.videoControllers.push(durationCtrl);

    const fpsCtrl = advanced
      .add(vidSettings, 'fps', { '30 FPS': 30, '60 FPS': 60 })
      .name('Frame Rate')
      .onChange((val: number) => {
        this.options.videoFPS = val;
      });
    this.videoControllers.push(fpsCtrl);

    // Codecs (only supported ones)
    const codecOptions: Record<string, string> = {};
    if (typeof MediaRecorder !== 'undefined') {
      if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) codecOptions['VP9'] = 'vp9';
      if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) codecOptions['VP8'] = 'vp8';
    }
    if (Object.keys(codecOptions).length > 0) {
      const codecCtrl = advanced
        .add(vidSettings, 'codec', codecOptions)
        .name('Codec')
        .onChange((val: string) => {
          this.options.videoCodec = val as 'vp9' | 'vp8';
        });
      this.videoControllers.push(codecCtrl);
    }

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

    // Sync dimension dropdown — populated dynamically from scene dims
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

    const speedCtrl = advanced
      .add(ttSettings, 'speed', 6, 180, 1)
      .name('Speed (°/s)')
      .onChange((val: number) => {
        this.options.turntableSpeed = val;
      });
    speedCtrl.domElement
      .closest('.luxar-gui__controller')
      ?.setAttribute('title', 'Rotation speed in degrees per second (36 = 10s for 360°)');
    this.turntableControllers.push(speedCtrl);

    this.updateControlVisibility();
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

    for (const ctrl of this.imageControllers) {
      isImage ? ctrl.show() : ctrl.hide();
    }
    if (isImage && this.options.imageFormat === 'png') {
      this.qualityController?.hide();
    }

    for (const ctrl of this.videoControllers) {
      isVideo || isTurntable ? ctrl.show() : ctrl.hide();
    }
    // Sync to slider only in video mode (not turntable)
    if (isTurntable) {
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

      let details = `Recording will capture the canvas at ${this.options.videoFPS} FPS.`;
      if (this.mode === 'turntable') {
        const duration = Math.round(360 / this.options.turntableSpeed);
        details = `Camera will rotate 360° in ~${duration}s at ${this.options.videoFPS} FPS.`;
      } else if (this.options.syncToSlider) {
        details += '<br>Recording will stop when the slider animation completes.';
      } else if (this.options.videoDurationLimit > 0) {
        details += `<br>Duration limit: ${this.options.videoDurationLimit} seconds.`;
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

      const trapKeyboard = (e: KeyboardEvent) => {
        e.stopPropagation();
        if (e.key === 'Escape') {
          cleanup();
          resolve(false);
        } else if (e.key === 'Enter') {
          cleanup();
          resolve(true);
        }
      };

      const handleClick = (e: MouseEvent) => {
        const action = (e.target as HTMLElement).dataset.action;
        if (action === 'cancel') {
          cleanup();
          resolve(false);
        } else if (action === 'start') {
          cleanup();
          resolve(true);
        }
      };

      const cleanup = () => {
        overlay.removeEventListener('keydown', trapKeyboard);
        overlay.removeEventListener('click', handleClick);
        overlay.remove();
      };

      overlay.addEventListener('keydown', trapKeyboard, true);
      overlay.addEventListener('click', handleClick);
      document.body.appendChild(overlay);
      overlay.tabIndex = -1;
      overlay.focus();
    });
  }

  // ========== Recording Indicator ==========

  private showRecordingIndicator(): void {
    const indicator = document.createElement('div');
    indicator.className = 'luxar-recording-indicator';
    indicator.innerHTML = `
      <div class="luxar-recording-indicator__dot"></div>
      <span class="luxar-recording-indicator__text">REC</span>
      <span class="luxar-recording-indicator__time">00:00</span>
    `;
    indicator.title = 'Click to stop recording';
    indicator.addEventListener('click', () => this.stopVideoRecording());
    document.body.appendChild(indicator);
    this.recordingIndicator = indicator;

    const timeEl = indicator.querySelector('.luxar-recording-indicator__time');
    this.recordingTimeInterval = setInterval(() => {
      if (timeEl) {
        const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
        const mins = Math.floor(elapsed / 60)
          .toString()
          .padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        timeEl.textContent = `${mins}:${secs}`;
      }
    }, 1000);
  }

  private hideRecordingIndicator(): void {
    if (this.recordingTimeInterval) {
      clearInterval(this.recordingTimeInterval);
      this.recordingTimeInterval = null;
    }
    if (this.recordingIndicator) {
      this.recordingIndicator.remove();
      this.recordingIndicator = null;
    }
  }

  // ========== Utilities ==========

  private getSupportedMimeType(): string | null {
    const codec = this.options.videoCodec;
    const preferred = `video/webm;codecs=${codec}`;
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(preferred)) {
      return preferred;
    }
    const fallbacks = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    for (const type of fallbacks) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return null;
  }

  private generateFilename(ext: string): string {
    const now = new Date();
    const pad = (n: number) => n.toString().padStart(2, '0');
    const ts = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    return `luxar-capture-${ts}.${ext}`;
  }

  private downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      a.remove();
      URL.revokeObjectURL(url);
    }, 100);
  }
}
