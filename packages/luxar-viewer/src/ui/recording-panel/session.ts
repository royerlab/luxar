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
import { getMaxPixelRatioCap, setMaxPixelRatioCap } from '../../rendering/pixel-ratio-cap';
import type { OverlayManager } from '../overlay-manager';
import { SliderSyncCoordinator } from './animation-sync';
import { getViewerContainer } from '../../utils/viewer-container';
import type { PanelStates, RecordingMode, RecordingOptions } from './types';

export type CaptureKind = 'screenshot' | 'video' | 'offline';

/**
 * How many candidate sizes the encoder alignment tries — the requested
 * one plus at most ten pixels of walk down — before giving up.
 */
const MAX_ALIGN_STEPS = 11;

/**
 * Align a capture dimension DOWN until the PHYSICAL frame it produces —
 * `round(value × scale)` — is even.
 *
 * H.264/H.265 with yuv420p need even dimensions — x265 refuses an odd
 * one outright ("height must be an integer multiple of the specified
 * chroma subsampling") and WebCodecs H.264 falls back to another codec
 * with nothing but a console warning (`drivers/video-mode-driver.ts`)
 * — but the encoder never sees the size requested here.
 * SSAA renders at `scale` times it, and the frames written to disk are
 * that physical size, so aligning the requested size alone still lets an
 * odd frame through: a 3024×1698 native target at SSAA 1.5× captures at
 * 4536×2547. Only the product has to be even, so the walk steps by ONE
 * and accepts an odd request whose product is even (at scale 1 that
 * collapses to the plain even floor, and at scale 2 nothing ever moves).
 * Requiring the request to be even as well throws away half the
 * candidates and costs far more than a factor of two: at a legal 1.05×
 * the even-only walk needs up to ten steps (six at height 1700) where
 * stepping by one never needs more than two, so it throws away twenty
 * pixels of frame where the unit walk throws away two — and when it runs
 * out it ships the odd physical frame it was there to prevent.
 * The multiplier is a free-form float (a scene's `viewer_config` clamps
 * it to [1, 8] and `setSSAAMultiplier` re-clamps to [1, 4]), so no fixed
 * alignment covers it.
 *
 * Best-effort, and deliberately so. Swept over every millesimal scale in
 * [1, 4] and every height in [16, 8000], two steps cover 87% of the
 * pairs — but 888 of those 3001 scales need three or more somewhere even
 * when they sit further than 0.2 from an even multiplier, worst case
 * five (1.751, where 753 walks to 748). Hence eleven candidates. What
 * the cap still gives up on is the scales within 0.1 of 2 or of 4 but
 * not ON them, 297 of the 3001: there the product's parity is locked
 * across a thousand consecutive heights — at 2.001 it flips at 500,
 * 1500, 2500, … — so no bounded walk can help, and at 1.999/2.001 about
 * half of all heights end odd (2.5% of the swept pairs overall). Those
 * take the even floor, which beats shrinking the frame by hundreds of
 * pixels to chase an even product. Exactly 2 and 4 never need a step;
 * 1 and 3 need at most one (1701 → 1700).
 */
function alignForEncoder(value: number, scale: number): number {
  if (!Number.isFinite(value)) return 2;
  const evenFloor = Math.max(2, value - (value % 2));
  if (!Number.isFinite(scale) || scale <= 0) return evenFloor;
  const start = Math.floor(value);
  for (let i = 0; i < MAX_ALIGN_STEPS; i++) {
    const candidate = start - i;
    if (candidate < 2) break;
    if (Math.round(candidate * scale) % 2 === 0) return candidate;
  }
  return evenFloor;
}

export interface SavedRecordingState {
  dprEnabled: boolean;
  dpr: number;
  /** Whether this capture explicitly pinned DPR through the manager. */
  dprPinned: boolean;
  /** Pixel-ratio cap in force before the capture raised it. */
  pixelRatioCap: number;
  rendererSize: { width: number; height: number } | null;
  resizeLocked: boolean;
}

export interface SaveRecordingStateOptions {
  lockResize?: boolean;
  /**
   * Pixel ratio to render the capture at. Adaptive DPR is switched off
   * and this exact ratio is pinned for the duration.
   *
   * Defaults (in the panel) to the on-screen ceiling, so an export
   * matches what you see. A HIGHER value is honoured by temporarily
   * raising the pixel-ratio cap — otherwise the renderer boundary would
   * clamp the request straight back down and the option would look
   * broken. Bounded by the display's own DPR.
   */
  captureDPR?: number;
  scaleResolution?: { targetH: number; alignEven?: boolean };
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
  private savedAutoDolly: boolean = false;

  // ── Mutual-exclusion (single source of truth) ─────────────────
  // Strategies cannot mutate these directly — they go through
  // canStart / reserve / release. Public reads (e.g. for the existing
  // public `isCurrentlyRecording()` API) go through isAnyCaptureActive.
  isRecording: boolean = false;
  isOfflineCaptureActive: boolean = false;
  isEXRSequenceRecording: boolean = false;

  /**
   * True while the animation loop's OWN render is redundant because
   * someone else owns the pipeline for the frame (today: the offline
   * capture, which renders its own pass per frame).
   *
   * Deliberately separate from `isOfflineCaptureActive`. This is the
   * narrower, shorter-lived claim — it is dropped the moment the capture
   * stops driving the pipeline, so a wedged teardown can never freeze the
   * viewport. `isOfflineCaptureActive` is a mutual-exclusion flag and
   * must stay true until the session has fully unwound.
   */
  isLoopRenderSuppressed: boolean = false;

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
    const dpr = this.adaptiveDPRManager?.getCurrentDPR() ?? this.sceneManager.activePixelRatio;
    const renderer = this.sceneManager.renderer;
    // The DISPLAY size, not `renderer.getSize()`: post-processing hands
    // the renderer the SSAA-multiplied size, so the renderer reports
    // `display × multiplier` while `resize()` — which is what both the
    // capture below and `restoreRecordingState` call — takes the display
    // size and applies the multiplier itself. Restoring the renderer's
    // own number grew the viewport by the multiplier on every capture.
    const currentSize = this.sceneManager.postProcessing.getDisplaySize();
    this.savedRecordingState = {
      dprEnabled,
      dpr,
      dprPinned: options.captureDPR !== undefined && this.adaptiveDPRManager !== null,
      pixelRatioCap: getMaxPixelRatioCap(),
      rendererSize: null,
      resizeLocked: this.sceneManager.resizeLocked,
    };

    if (options.captureDPR !== undefined && this.adaptiveDPRManager) {
      // Freeze the resolution for the capture: adaptation off, then the
      // requested ratio pinned.
      this.adaptiveDPRManager.setEnabled(false);
      // Raising the cap FIRST — a captureDPR above the on-screen ceiling
      // is the whole point of the panel's override, and the renderer
      // boundary would otherwise clamp it straight back down. Bounded by
      // the display, which `setAdaptivePixelRatio` applies for us.
      if (options.captureDPR > getMaxPixelRatioCap()) {
        setMaxPixelRatioCap(options.captureDPR);
      }
      // Unconditional, unlike the pre-cap version which leaned on
      // setEnabled(false) resetting to native: that reset now lands on
      // the CEILING, which is not necessarily the requested ratio.
      this.sceneManager.setAdaptivePixelRatio(options.captureDPR);
    }

    if (options.lockResize) {
      this.sceneManager.resizeLocked = true;
    }

    if (options.scaleResolution) {
      this.savedRecordingState.rendererSize = {
        width: currentSize.width,
        height: currentSize.height,
      };
      const { targetH, alignEven } = options.scaleResolution;
      // A canvas with no height yet (hidden container, pre-layout) would
      // make the aspect Infinity or NaN and carry it into the render
      // target and the camera. Square is a harmless stand-in.
      const rawAspect = currentSize.width / currentSize.height;
      const aspect = currentSize.height > 0 && Number.isFinite(rawAspect) ? rawAspect : 1;
      // Align the HEIGHT first, then derive the width from the aligned
      // height, so the output aspect still tracks the source. Deriving
      // the width from the requested height and then truncating both
      // independently widened the frame relative to what the user
      // framed. Even, not a multiple of 16: H.264/H.265 with yuv420p
      // need even dimensions and the encoder pads to its own macroblock
      // size. `alignForEncoder` also folds in the SSAA scale, since the
      // frames on disk are the physical size, not this one.
      const scale = this.sceneManager.postProcessing.getEffectiveRenderScale();
      const h = alignEven ? alignForEncoder(targetH, scale) : targetH;
      const wRaw = Math.round(h * aspect);
      const w = alignEven ? alignForEncoder(wRaw, scale) : wRaw;
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

    // Drop the capture's cap FIRST, so everything below re-sizes against
    // the ratio the session is going back to rather than the one the
    // export ran at.
    setMaxPixelRatioCap(saved.pixelRatioCap);

    if (saved.rendererSize) {
      const renderer = this.sceneManager.renderer;
      // Through the policy, NOT `window.devicePixelRatio` directly: the
      // raw native value ignores both the cap and the scene manager's
      // override, so on a capped session it left the renderer drawing at
      // 2x while `pixelRatioOverride` still said 1.0 — a doubled
      // backbuffer that only snapped back at the next resize, as a
      // visible resolution flip.
      renderer.setPixelRatio(this.sceneManager.activePixelRatio);
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
        if (saved.dprPinned) {
          this.sceneManager.setAdaptivePixelRatio(this.adaptiveDPRManager.getCurrentDPR());
        }
      } else if (saved.dprPinned) {
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

  /** Restore the auto-dolly to its pre-recording state. */
  restoreAutoDolly(): void {
    if (this.savedAutoDolly) {
      this.sceneManager.controls.setAutoDolly(true);
      this.savedAutoDolly = false;
    }
  }

  /**
   * Capture and pause the auto-dolly; restore via restoreAutoDolly().
   *
   * Kept separate from {@link pauseAutoRotate} rather than folded into it so
   * neither name lies about what it touches. Both are paused for the same
   * reason: a turntable recording drives the camera itself (from a frame index
   * offline, from wall-clock progress live), and the interactive animation
   * would compound with it.
   */
  pauseAutoDolly(): void {
    this.savedAutoDolly = this.sceneManager.controls.getAutoDolly();
    this.sceneManager.controls.setAutoDolly(false);
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
        // Round for DISPLAY only, never before the frame-count multiply:
        // the capture loop and the panel's Output field both derive their
        // counts from the unrounded duration, so rounding first promises a
        // different number of frames than the capture actually produces
        // (7°/s at 30 FPS: 1530 promised vs 1543 captured).
        const duration = 360 / options.turntableSpeed;
        const expectedFrames = Math.ceil(duration * options.videoFPS);
        details = `Camera will rotate 360° — ${expectedFrames} frames at ${options.videoFPS} FPS (${duration.toFixed(1)}s video).`;
        if (options.frameByFrame) {
          details += '<br><strong>Offline capture</strong> — each frame is rendered individually.';
        }
      } else if (options.syncToSlider) {
        details += '<br>Recording will stop when the slider animation completes.';
      } else if (options.videoDurationLimit > 0) {
        details += `<br>Duration limit: ${options.videoDurationLimit} seconds.`;
      }
      // The real-time MediaRecorder path always emits WebM and picks the
      // codec itself, so don't advertise `options.videoCodec` there. That
      // path runs for Video mode and for a non-smooth WebM turntable
      // (mirrors the dispatch in RecordingPanel.startVideoRecording). The
      // offline mediabunny path (smooth turntable, or any mp4/mkv) does
      // honor the codec.
      const realtimeWebm = fmt === 'webm' && (mode === 'video' || !options.frameByFrame);

      if (fmt === 'exr') {
        details += '<br>Output: <strong>ZIP of EXR frames</strong> (full float precision).';
      } else if (fmt === 'png' || fmt === 'webp' || fmt === 'jpeg') {
        details += `<br>Output: <strong>ZIP of ${fmt.toUpperCase()} frames</strong> + ffmpeg script.`;
      } else if (fmt === 'mp4' || fmt === 'webm' || fmt === 'mkv') {
        details += realtimeWebm
          ? '<br>Output: <strong>WebM video</strong>.'
          : `<br>Output: <strong>${fmt.toUpperCase()} video</strong> (${options.videoCodec.toUpperCase()}).`;
      }

      // One capture path still cannot composite DOM overlays: the EXR
      // driver writes the raw pre-grade HDR buffer, and an overlay is a
      // display-space object with no meaning there. Say so rather than
      // letting "Include Overlays" quietly produce a file without them.
      // (The real-time recorder USED to be the other one — it now captures
      // a mirror canvas that `LiveOverlayCompositor` re-composites per
      // rendered frame, so WebM carries overlays like every other format.)
      const hasOverlays = (this.overlayManager?.getVisibleOverlays().length ?? 0) > 0;
      if (options.includeOverlays && hasOverlays && fmt === 'exr') {
        details +=
          '<br><strong>Overlays will NOT be included</strong> — EXR frames are the raw HDR buffer.';
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
