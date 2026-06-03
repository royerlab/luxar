/**
 * GUI construction — extracted from `recording-panel.ts::buildGUI`.
 *
 * Pure-ish DOM-building helper. Takes the target GUI folder, a
 * mutable options object, and a small set of callbacks that the
 * controllers' onChange handlers fire. Returns the allocated
 * Controller references so the panel can keep them for runtime
 * visibility toggling.
 *
 * No coupling to the orchestrator beyond the deps interface — all
 * `this.x` mutations are expressed as setter callbacks.
 */

import GUI, { type Controller } from '../../gui';
import type {
  OutputFormat,
  RecordingMode,
  RecordingOptions,
  VideoCodecOption,
  VideoQuality,
  VideoResolution,
} from '../types';

export interface BuildGUIResult {
  formatController: Controller;
  qualityController: Controller;
  transparentController: Controller;
  videoCodecController: Controller;
  videoQualityController: Controller;
  videoDurationController: Controller;
  syncToggleController: Controller;
  syncDimensionController: Controller;
  captureController: Controller;
  imageControllers: Controller[];
  videoControllers: Controller[];
  turntableControllers: Controller[];
}

/** Action-button label for a mode: still capture vs. continuous record. */
export function captureLabelForMode(mode: RecordingMode): string {
  return mode === 'image' ? 'Capture' : 'Record';
}

export interface BuildGUIDeps {
  gui: GUI;
  options: RecordingOptions;
  initialMode: RecordingMode;
  setMode(mode: RecordingMode): void;
  updateControlVisibility(): void;
  getTurntableInfo(): string;
  getNavigableDimensionOptions(): Record<string, number>;
  captureScreenshot(): void;
  startVideoRecording(): void;
}

export function buildRecordingGUI(deps: BuildGUIDeps): BuildGUIResult {
  const imageControllers: Controller[] = [];
  const videoControllers: Controller[] = [];
  const turntableControllers: Controller[] = [];
  let syncDimensionController: Controller;

  // The panel is laid out in two tiers: primary capture settings sit
  // directly on the root GUI (always one glance away), while niche
  // toggles live in a collapsed "Advanced Options" folder created
  // *after* the primary controls so they render below them.
  //   Primary : Mode, Format, Image Quality, Transparent BG, Video
  //             Quality, Resolution, Max Duration, Frame Rate, turntable
  //             Output/Speed/Smooth.
  //   Advanced: Show Panels, Include Overlays, Max Resolution (DPR),
  //             Codec, Sync to Slider, Dimension.
  const root = deps.gui;

  // Mode toggle — at the top
  const modeObj = { mode: deps.initialMode };
  root
    .add(modeObj, 'mode', { Image: 'image', Video: 'video', Turntable: 'turntable' })
    .name('Mode')
    .onChange((val: string) => {
      deps.setMode(val as RecordingMode);
      deps.updateControlVisibility();
    });

  // ── Image options (primary) ──
  const imgSettings = {
    format: deps.options.outputFormat,
    quality: deps.options.imageQuality,
    maxDPR: deps.options.maxDPR,
    transparentBg: deps.options.transparentBackground,
  };

  const formatCtrl = root
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
      deps.options.outputFormat = val as OutputFormat;
      deps.updateControlVisibility();
    });
  // Format visibility is mode-dependent (hidden in WebM-only Video mode).

  const qualityCtrl = root
    .add(imgSettings, 'quality', 0.1, 1.0, 0.05)
    .name('Image Quality')
    .onChange((val: number) => {
      deps.options.imageQuality = val;
    });
  imageControllers.push(qualityCtrl);

  const transparentCtrl = root
    .add(imgSettings, 'transparentBg')
    .name('Transparent BG')
    .onChange((val: boolean) => {
      deps.options.transparentBackground = val;
    });
  transparentCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Transparent background (PNG/WebP only, auto-switches from JPEG)');
  imageControllers.push(transparentCtrl);

  // ── Video options (primary) ──
  const vidSettings = {
    duration: deps.options.videoDurationLimit,
    fps: deps.options.videoFPS,
    codec: deps.options.videoCodec,
    quality: deps.options.videoQuality,
    resolution: deps.options.videoResolution,
    syncToSlider: deps.options.syncToSlider,
    syncDim: deps.options.syncDimensionIndex,
  };

  const videoQualityCtrl = root
    .add(vidSettings, 'quality', { Low: 'low', Medium: 'medium', High: 'high', Max: 'max' })
    .name('Video Quality')
    .onChange((val: string) => {
      deps.options.videoQuality = val as VideoQuality;
    });
  videoQualityCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute(
      'title',
      'Video bitrate quality (Low ~2.5Mbps, Medium ~5Mbps, High ~9Mbps, Max ~19Mbps at 1080p)'
    );
  videoControllers.push(videoQualityCtrl);

  const resolutionCtrl = root
    .add(vidSettings, 'resolution', { Native: 0, '1080p': 1080, '1440p': 1440, '4K': 2160 })
    .name('Resolution')
    .onChange((val: number) => {
      deps.options.videoResolution = val as VideoResolution;
    });
  resolutionCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Output video resolution (Native = current canvas size)');
  videoControllers.push(resolutionCtrl);

  const durationCtrl = root
    .add(vidSettings, 'duration', 0, 300, 1)
    .name('Max Duration (s)')
    .onChange((val: number) => {
      deps.options.videoDurationLimit = val;
    });
  durationCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Recording duration limit in seconds (0 = unlimited)');
  videoControllers.push(durationCtrl);

  const fpsCtrl = root
    .add(vidSettings, 'fps', { '30 FPS': 30, '60 FPS': 60 })
    .name('Frame Rate')
    .onChange((val: number) => {
      deps.options.videoFPS = val;
    });
  videoControllers.push(fpsCtrl);

  // ── Turntable options (primary) ──
  const ttSettings = { speed: deps.options.turntableSpeed };

  // Turntable info display (computed from speed + FPS, read-only)
  const turntableInfo = { info: deps.getTurntableInfo() };
  const turntableInfoCtrl = root.add(turntableInfo, 'info').name('Output');
  // Make the input read-only (this is a computed display, not user-editable)
  const infoInput = turntableInfoCtrl.domElement.querySelector('input');
  if (infoInput) {
    infoInput.readOnly = true;
    infoInput.style.opacity = '0.7';
    infoInput.style.cursor = 'default';
  }
  turntableControllers.push(turntableInfoCtrl);

  const updateTurntableInfo = () => {
    turntableInfo.info = deps.getTurntableInfo();
    turntableInfoCtrl.updateDisplay();
  };

  const speedCtrl = root
    .add(ttSettings, 'speed', 6, 180, 1)
    .name('Speed (°/s)')
    .onChange((val: number) => {
      deps.options.turntableSpeed = val;
      updateTurntableInfo();
    });
  speedCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Rotation speed in degrees per second (36 = 10s for 360°)');
  turntableControllers.push(speedCtrl);

  // Also update turntable info when FPS changes
  fpsCtrl.onChange((val: number) => {
    deps.options.videoFPS = val;
    updateTurntableInfo();
  });

  // Frame-by-frame (smooth) checkbox
  const fbfSettings = { frameByFrame: deps.options.frameByFrame };
  const fbfCtrl = root
    .add(fbfSettings, 'frameByFrame')
    .name('Smooth (offline)')
    .onChange((val: boolean) => {
      deps.options.frameByFrame = val;
    });
  fbfCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute(
      'title',
      'Render each frame individually for perfectly smooth video. ' +
        'Slower to capture, but guarantees every frame is fully rendered. ' +
        'Recommended for heavy scenes.'
    );
  turntableControllers.push(fbfCtrl);

  // ── Advanced Options folder (created after the primary controls so it
  //    renders below them; starts closed) ──
  const advanced = root.addFolder('Advanced Options');
  advanced.close();

  const generalSettings = { showPanels: deps.options.showPanels };
  const showPanelsCtrl = advanced
    .add(generalSettings, 'showPanels')
    .name('Show Panels')
    .onChange((val: boolean) => {
      deps.options.showPanels = val;
    });
  showPanelsCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Keep other panels visible during capture');

  const overlaySettings = { includeOverlays: deps.options.includeOverlays };
  const overlayCtrl = advanced
    .add(overlaySettings, 'includeOverlays')
    .name('Include Overlays')
    .onChange((val: boolean) => {
      deps.options.includeOverlays = val;
    });
  overlayCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Composite text/image/HTML overlays into the capture');

  const maxDPRCtrl = advanced
    .add(imgSettings, 'maxDPR')
    .name('Max Resolution')
    .onChange((val: boolean) => {
      deps.options.maxDPR = val;
    });
  maxDPRCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Maximize pixel ratio for highest resolution screenshot');
  imageControllers.push(maxDPRCtrl);

  // Video codec selector (turntable/offline only — the real-time Video
  // mode always emits WebM via MediaRecorder and ignores this).
  const codecCtrl = advanced
    .add(vidSettings, 'codec', {
      'H.265': 'h265',
      VP9: 'vp9',
      'H.264': 'h264',
      VP8: 'vp8',
    })
    .name('Codec')
    .onChange((val: string) => {
      deps.options.videoCodec = val as VideoCodecOption;
    });
  videoControllers.push(codecCtrl);

  // Sync to slider
  const syncCtrl = advanced
    .add(vidSettings, 'syncToSlider')
    .name('Sync to Slider')
    .onChange((val: boolean) => {
      deps.options.syncToSlider = val;
      syncDimensionController?.[val ? 'show' : 'hide']();
    });
  syncCtrl.domElement
    .closest('.luxar-gui__controller')
    ?.setAttribute('title', 'Sync recording to a dimension animation (auto-stop at end)');
  videoControllers.push(syncCtrl);

  // Sync dimension dropdown
  const dimNames = deps.getNavigableDimensionOptions();
  const syncDimCtrl = advanced
    .add(vidSettings, 'syncDim', dimNames)
    .name('Dimension')
    .onChange((val: number) => {
      deps.options.syncDimensionIndex = val;
    });
  videoControllers.push(syncDimCtrl);
  syncDimensionController = syncDimCtrl;

  // Capture/Record button — at the bottom, prominent. Label tracks the
  // mode: a still capture in Image mode, a continuous record otherwise.
  const actions = {
    capture: () => {
      if (modeObj.mode === 'image') {
        deps.captureScreenshot();
      } else {
        deps.startVideoRecording();
      }
    },
  };
  // The button's capture callback reads the current mode from the modeObj
  // closure rather than from deps, since the user may have changed mode
  // since the panel was built.
  const captureBtn = root.add(actions, 'capture').name(captureLabelForMode(deps.initialMode));
  captureBtn.domElement.closest('.luxar-gui__controller')?.classList.add('luxar-recording-btn');

  return {
    formatController: formatCtrl,
    qualityController: qualityCtrl,
    transparentController: transparentCtrl,
    videoCodecController: codecCtrl,
    videoQualityController: videoQualityCtrl,
    videoDurationController: durationCtrl,
    syncToggleController: syncCtrl,
    syncDimensionController: syncDimCtrl,
    captureController: captureBtn,
    imageControllers,
    videoControllers,
    turntableControllers,
  };
}
