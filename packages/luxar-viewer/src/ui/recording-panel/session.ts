/**
 * Shared scaffolding for recording strategies.
 *
 * Owns the cross-cutting state that every capture path needs:
 * mutual-exclusion lock, save/restore of renderer + panel + auto-rotate
 * state, the confirmation dialog, the REC indicator, and the slider-sync
 * coordinator. Strategies (Screenshot / VideoRecording / OfflineCapture)
 * hold a reference and call into Session for these concerns instead of
 * duplicating the save→capture→unwind sequence five times.
 *
 * The class also owns `disposed` (single lifecycle guard) and the
 * optional collaborator references (animation/DPR/overlay managers).
 */

import * as THREE from 'three';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import type { DimensionAnimationManager } from '../../scene/animation/dimension-animation-manager';
import type { AdaptiveDPRManager } from '../../rendering/adaptive-dpr-manager';
import type { OverlayManager } from '../overlay-manager';
import { SliderSyncCoordinator } from './animation-sync';
import { getViewerContainer } from '../../utils/viewer-container';
import type { PanelStates, RecordingMode, RecordingOptions } from './types';

export type CaptureKind = 'screenshot' | 'video' | 'offline';

export interface SavedRecordingState {
  dprEnabled: boolean;
  dpr: number;
  rendererSize: { width: number; height: number } | null;
  resizeLocked: boolean;
}

export interface SaveRecordingStateOptions {
  lockResize?: boolean;
  disableDPR?: boolean;
  scaleResolution?: { targetH: number; align16?: boolean };
}

export interface ConfirmationDialogInfo {
  mode: RecordingMode;
  options: RecordingOptions;
}

/**
 * Hook surface a strategy may need to ask the panel about (e.g. the
 * recording indicator's elapsed-time formatter wants to know whether
 * the current capture is an EXR sequence to switch the label text).
 */
export interface SessionHooks {
  isExrSequenceActive(): boolean;
}

export class RecordingSession {
  // ── State save/restore ────────────────────────────────────────
  private savedRecordingState: SavedRecordingState | null = null;
  private savedPanelStates: PanelStates | null = null;
  private savedAutoRotate: boolean = false;

  // ── Mutual-exclusion (single source of truth) ─────────────────
  // Strategies cannot mutate these directly — they go through
  // canStart / reserve / release. Public reads (e.g. for the existing
  // public `isCurrentlyRecording()` API) go through isAnyCaptureActive.
  isRecording: boolean = false;
  isOfflineCaptureActive: boolean = false;
  isEXRSequenceRecording: boolean = false;

  // ── Lifecycle ─────────────────────────────────────────────────
  disposed: boolean = false;

  // ── Indicator DOM state ───────────────────────────────────────
  recordingIndicator: HTMLElement | null = null;
  recordingTimeInterval: ReturnType<typeof setInterval> | null = null;
  recordingStartTime: number = 0;
  private recordingIndicatorClickCleanup: (() => void) | null = null;

  // ── Confirmation dialog ───────────────────────────────────────
  confirmationDialogCancel: (() => void) | null = null;

  // ── Optional collaborator references ──────────────────────────
  animationManager: DimensionAnimationManager | null = null;
  adaptiveDPRManager: AdaptiveDPRManager | null = null;
  overlayManager: OverlayManager | null = null;

  // ── Panel-state callbacks (set by app via Panel.setPanelStateCallbacks) ──
  getPanelStates: (() => PanelStates) | null = null;
  restorePanelStatesCallback: ((states: PanelStates) => void) | null = null;

  // ── Slider sync coordinator ───────────────────────────────────
  readonly sliderSync = new SliderSyncCoordinator();

  // ── Stop-recording callback wired by Panel (so the click-to-stop on
  //    the indicator and Escape can trigger the same teardown).
  private stopVideoCallback: (() => void) | null = null;

  constructor(
    public readonly sceneManager: SceneManager,
    public readonly animationController: AnimationController,
    private readonly hooks: SessionHooks
  ) {}

  // ── Lifecycle helpers ─────────────────────────────────────────

  isDisposed(): boolean {
    return this.disposed;
  }

  isAnyCaptureActive(): boolean {
    return this.isRecording;
  }

  /** Wire the panel's stop callback used by indicator-click and Escape. */
  setStopVideoCallback(cb: () => void): void {
    this.stopVideoCallback = cb;
  }

  // ── Mutual-exclusion API ──────────────────────────────────────

  canStart(_kind: CaptureKind): boolean {
    return !this.isRecording && !this.isOfflineCaptureActive;
  }

  // ── Renderer / DPR / resize-lock state guard ──────────────────

  /**
   * Save current DPR, renderer size, and resize-lock state, then apply
   * recording-safe defaults (disable adaptive DPR, lock resize, optionally
   * scale resolution). Call `restoreRecordingState()` to undo all changes.
   */
  saveRecordingState(options: SaveRecordingStateOptions): void {
    const dprEnabled = this.adaptiveDPRManager?.isActive() ?? false;
    const dpr = this.adaptiveDPRManager?.getCurrentDPR() ?? window.devicePixelRatio;
    const renderer = this.sceneManager.renderer;
    const currentSize = renderer.getSize(new THREE.Vector2());
    this.savedRecordingState = {
      dprEnabled,
      dpr,
      rendererSize: null,
      resizeLocked: this.sceneManager.resizeLocked,
    };

    if (options.disableDPR && this.adaptiveDPRManager) {
      // When adaptive was enabled, setEnabled(false) already resets to
      // native and applies it through the full resize path — calling
      // setAdaptivePixelRatio again would repeat the HDR-target
      // dispose/recreate for nothing.
      this.adaptiveDPRManager.setEnabled(false);
      if (!dprEnabled) {
        // Manual-DPR mode: setEnabled(false) early-returned (state
        // unchanged), so this call is the only thing forcing native
        // resolution for the capture. NOT dead code.
        this.sceneManager.setAdaptivePixelRatio(this.adaptiveDPRManager.getNativeDPR());
      }
    }

    if (options.lockResize) {
      this.sceneManager.resizeLocked = true;
    }

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
      const canvas = renderer.domElement;
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      const camera = this.sceneManager.camera;
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
      this.sceneManager.updateMaterialsForCurrentCamera();
    }
  }

  /**
   * Restore DPR, renderer size, resize lock, and panels to pre-recording state.
   * Safe to call multiple times (no-ops if no saved state).
   */
  restoreRecordingState(): void {
    const saved = this.savedRecordingState;
    if (!saved) return;

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

    this.sceneManager.resizeLocked = saved.resizeLocked;

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

  /** Restore auto-rotation to its pre-turntable state. */
  restoreAutoRotate(): void {
    if (this.savedAutoRotate) {
      this.sceneManager.controls.setAutoRotate(true);
      this.savedAutoRotate = false;
    }
  }

  /** Capture and pause auto-rotation; restore via restoreAutoRotate(). */
  pauseAutoRotate(): void {
    this.savedAutoRotate = this.sceneManager.controls.getAutoRotate();
    this.sceneManager.controls.setAutoRotate(false);
  }

  // ── Panel-state hide/restore ──────────────────────────────────

  /**
   * Snapshot current panel visibility, hide the recording panel itself,
   * and optionally hide all other panels (per `showPanels` flag).
   */
  hideAllPanels(panelVisibleHook: { hideOwnPanel: () => void }, showPanels: boolean): void {
    if (!this.savedPanelStates && this.getPanelStates) {
      this.savedPanelStates = this.getPanelStates();
    }
    panelVisibleHook.hideOwnPanel();
    if (showPanels) return;

    if (this.savedPanelStates && this.restorePanelStatesCallback) {
      const allHidden = new Map<string, boolean>();
      for (const key of this.savedPanelStates.keys()) {
        allHidden.set(key, false);
      }
      this.restorePanelStatesCallback(allHidden);
    }
  }

  /** Restore the snapshot taken by hideAllPanels. No-op while capture active. */
  restoreAllPanels(): void {
    if (this.isRecording) return;
    if (this.savedPanelStates && this.restorePanelStatesCallback) {
      this.restorePanelStatesCallback(this.savedPanelStates);
      this.savedPanelStates = null;
    }
  }

  // ── Confirmation dialog ───────────────────────────────────────

  showConfirmationDialog(info: ConfirmationDialogInfo): Promise<boolean> {
    return new Promise((resolve) => {
      const { mode, options } = info;
      const overlay = document.createElement('div');
      overlay.className = 'luxar-recording-confirm';
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');
      overlay.setAttribute('aria-labelledby', 'luxar-recording-confirm-title');
      overlay.setAttribute('aria-describedby', 'luxar-recording-confirm-message');

      const fmt = options.outputFormat;
      let details = `Recording will capture at ${options.videoFPS} FPS.`;
      if (mode === 'turntable') {
        const duration = Math.round(360 / options.turntableSpeed);
        const expectedFrames = Math.ceil(duration * options.videoFPS);
        details = `Camera will rotate 360° — ${expectedFrames} frames at ${options.videoFPS} FPS (${duration}s video).`;
        if (options.frameByFrame) {
          details += '<br><strong>Offline capture</strong> — each frame is rendered individually.';
        }
      } else if (options.syncToSlider) {
        details += '<br>Recording will stop when the slider animation completes.';
      } else if (options.videoDurationLimit > 0) {
        details += `<br>Duration limit: ${options.videoDurationLimit} seconds.`;
      }
      if (fmt === 'exr') {
        details += '<br>Output: <strong>ZIP of EXR frames</strong> (full float precision).';
      } else if (fmt === 'png' || fmt === 'webp' || fmt === 'jpeg') {
        details += `<br>Output: <strong>ZIP of ${fmt.toUpperCase()} frames</strong> + ffmpeg script.`;
      } else if (fmt === 'mp4' || fmt === 'webm' || fmt === 'mkv') {
        // The real-time MediaRecorder path always emits WebM and picks the
        // codec itself, so don't advertise `options.videoCodec` there. That
        // path runs for Video mode and for a non-smooth WebM turntable
        // (mirrors the dispatch in RecordingPanel.startVideoRecording). The
        // offline mediabunny path (smooth turntable, or any mp4/mkv) does
        // honor the codec.
        const realtimeWebm = fmt === 'webm' && (mode === 'video' || !options.frameByFrame);
        details += realtimeWebm
          ? '<br>Output: <strong>WebM video</strong>.'
          : `<br>Output: <strong>${fmt.toUpperCase()} video</strong> (${options.videoCodec.toUpperCase()}).`;
      }

      overlay.innerHTML = `
        <div class="luxar-recording-confirm__dialog">
          <div id="luxar-recording-confirm-title" class="luxar-recording-confirm__title">Start ${mode === 'turntable' ? 'Turntable' : 'Video'} Recording</div>
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

      const previouslyFocused = document.activeElement as HTMLElement | null;

      let settled = false;
      const finish = (result: boolean): void => {
        if (settled) return;
        settled = true;
        cleanup();
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
      getViewerContainer().appendChild(overlay);
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

  // ── REC indicator widget ──────────────────────────────────────

  showRecordingIndicator(): void {
    this.hideRecordingIndicator();

    const indicator = document.createElement('div');
    indicator.className = 'luxar-recording-indicator';
    indicator.innerHTML = `
      <div class="luxar-recording-indicator__dot"></div>
      <span class="luxar-recording-indicator__text">REC</span>
      <span class="luxar-recording-indicator__time">00:00</span>
    `;
    indicator.title = 'Click to stop recording';
    const handleIndicatorClick = (): void => {
      this.stopVideoCallback?.();
    };
    indicator.addEventListener('click', handleIndicatorClick);
    this.recordingIndicatorClickCleanup = () => {
      indicator.removeEventListener('click', handleIndicatorClick);
      this.recordingIndicatorClickCleanup = null;
    };
    getViewerContainer().appendChild(indicator);
    this.recordingIndicator = indicator;

    const timeEl = indicator.querySelector('.luxar-recording-indicator__time');
    this.recordingTimeInterval = setInterval(() => {
      if (timeEl) {
        if (this.hooks.isExrSequenceActive()) {
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

  hideRecordingIndicator(): void {
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

  // ── Slider sync ───────────────────────────────────────────────

  startSliderSync(syncDimIndex: number, onComplete: () => void): void {
    if (!this.animationManager) return;
    this.sliderSync.start(syncDimIndex, this.animationManager, onComplete, () => !this.disposed);
  }

  cleanupSyncListener(): void {
    this.sliderSync.cleanup(this.animationManager);
  }

  // ── Dispose ───────────────────────────────────────────────────

  /**
   * Unwind Session-owned state. Idempotent — every step tolerates
   * missing-state, so dispose() is safe to call multiple times.
   *
   * Important: this does NOT early-return on `disposed=true`. The Panel
   * may have set `disposed=true` BEFORE calling `dispose()` so that
   * any async callback (e.g. `mediaRecorder.onstop`) fired during
   * Panel's own cleanup short-circuits via `isDisposed()`. We still
   * need the actual unwind steps to run.
   *
   * NOTE: Strategy-owned state (mediaRecorder tracks, offline AbortController,
   * per-frame callbacks registered by Video/Offline strategies) is NOT cleaned
   * up here. The Panel's `dispose()` is responsible for orchestrating those
   * teardowns BEFORE calling `session.dispose()`.
   */
  dispose(): void {
    this.disposed = true;

    this.confirmationDialogCancel?.();
    this.hideRecordingIndicator();
    this.cleanupSyncListener();
    this.restoreAutoRotate();
    this.restoreRecordingState();
  }
}
