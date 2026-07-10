/**
 * Navigation rail popover — a current-mode selector plus the current mode's
 * parameters.
 *
 * The rail's Navigation button cycles orbit → fly → ortho on left-click (via
 * the shared `toggleControlMode` command); right-click opens this popover. The
 * popover shows a segmented mode selector (current mode highlighted, click to
 * switch) followed by ONLY the current mode's parameters. Rebuilt in place on a
 * mode switch and on each open, so it always reflects the live mode + scene
 * scale. This is the sole home of the navigation parameter controls (moved out
 * of the Rendering Controls panel — navigation is not a rendering concern).
 *
 * Handlers mirror the scene-manager wiring exactly: mutate the shared
 * `RenderingSettings`, push to the SceneManager, persist, and trigger a frame.
 *
 * @module ui/rail-panels/navigation-popover
 */

import { config, type RenderingSettings } from '../../config';
import type { SceneManager } from '../../scene/scene-manager';
import type { AnimationController } from '../../scene/animation/animation-controller';
import type { ControlType } from '../../controls/types';
import { makePopoverGui } from './popover-gui';

export interface NavigationPopoverContext {
  /** Shared rendering settings (persisted by the rendering-controls panel). */
  settings: RenderingSettings;
  sceneManager: SceneManager;
  animationController: AnimationController;
  /** Persist the shared settings to localStorage (rendering-controls owns the key). */
  saveSettings: () => void;
  /** Request a render so changes are visible immediately. */
  triggerAnimation: () => void;
  /** Switch to a specific control mode (reuses the V-cycle context wiring). */
  setMode: (type: ControlType) => void;
}

/** The three modes, in cycle order, with labels for the selector. */
const MODES: { id: ControlType; label: string }[] = [
  { id: 'orbit', label: 'Orbit' },
  { id: 'fly', label: 'Fly' },
  { id: 'ortho', label: 'Ortho' },
];

/**
 * Compute the fly movement-speed slider range for the current scene scale,
 * mirroring `RenderingControls.updateSceneScale`. Falls back to the static
 * config range when the scale is unknown (scale <= 0).
 */
function flySpeedRange(sceneManager: SceneManager): { min: number; max: number; step: number } {
  const cfg = config.controls.fly.movement.speed;
  const scale = sceneManager.getSceneScale();
  if (scale > 0) {
    const scaled = scale * config.controls.scaleMultipliers.flySpeedFactor;
    return {
      min: Math.max(0.01, scaled * 0.1),
      max: scaled * 10,
      step: Math.max(0.01, scaled * 0.01),
    };
  }
  return { min: cfg.min, max: cfg.max, step: cfg.step || 0.1 };
}

/**
 * Build the segmented mode selector: three pills, the current one highlighted,
 * each switching directly to that mode (and rebuilding the popover).
 */
function buildModeSelector(
  currentMode: ControlType,
  onSelect: (mode: ControlType) => void
): HTMLElement {
  const strip = document.createElement('div');
  strip.className = 'luxar-control-rail__mode-strip';
  strip.setAttribute('role', 'group');
  strip.setAttribute('aria-label', 'Camera control mode');
  for (const { id, label } of MODES) {
    const seg = document.createElement('button');
    seg.type = 'button';
    seg.className = 'luxar-control-rail__mode-seg';
    seg.textContent = label;
    const active = id === currentMode;
    seg.classList.toggle('is-active', active);
    seg.setAttribute('aria-pressed', String(active));
    seg.addEventListener('click', () => {
      if (id !== currentMode) onSelect(id);
    });
    strip.appendChild(seg);
  }
  return strip;
}

/** Build the parameter controls for the current mode into `gui`. */
function buildModeParams(
  gui: ReturnType<typeof makePopoverGui>,
  mode: ControlType,
  ctx: NavigationPopoverContext
): void {
  const { settings, sceneManager, animationController, saveSettings, triggerAnimation } = ctx;

  if (mode === 'orbit') {
    gui
      .add(settings, 'autoRotate')
      .name('Auto Rotate')
      .onChange((value: boolean) => {
        sceneManager.setAutoRotate(value);
        saveSettings();
        if (value) animationController.startAnimation();
      });

    gui
      .add(settings, 'autoRotateSpeed', 0.1, 5, 0.1)
      .name('Rotation Speed')
      .onChange((value: number) => {
        sceneManager.setAutoRotateSpeed(value);
        saveSettings();
        triggerAnimation();
      });

    gui
      .add(settings, 'naturalDrag')
      .name('Natural drag')
      .onChange((value: boolean) => {
        sceneManager.setNaturalDrag(value);
        saveSettings();
        triggerAnimation();
      });

    const zs = config.controls.orbit.zoom.speed;
    gui
      .add(settings, 'orbitZoomSpeed', zs.min, zs.max, zs.step || 0.1)
      .name('Zoom Speed')
      .onChange((value: number) => {
        sceneManager.setOrbitZoomSpeed(value);
        saveSettings();
        triggerAnimation();
      });

    const df = config.controls.orbit.damping.factor;
    gui
      .add(settings, 'orbitDampingFactor', df.min, df.max, df.step || 0.01)
      .name('Damping')
      .onChange((value: number) => {
        sceneManager.setOrbitDampingFactor(value);
        saveSettings();
        triggerAnimation();
      });
  } else if (mode === 'fly') {
    const range = flySpeedRange(sceneManager);
    gui
      .add(settings, 'flyMovementSpeed', range.min, range.max, range.step)
      .name('Movement Speed')
      .onChange((value: number) => {
        sceneManager.setFlyMovementSpeed(value);
        saveSettings();
      });

    const rot = config.controls.fly.rotation.speed;
    gui
      .add(settings, 'flyRotationSpeed', rot.min, rot.max, rot.step || 0.1)
      .name('Rotation Speed')
      .onChange((value: number) => {
        sceneManager.setFlyRotationSpeed(value);
        saveSettings();
      });

    const look = config.controls.fly.look.mouseSpeed;
    gui
      .add(settings, 'flyLookSpeed', look.min, look.max, look.step || 0.0005)
      .name('Look Sensitivity')
      .onChange((value: number) => {
        sceneManager.setFlyLookSpeed(value);
        saveSettings();
      });

    const damp = config.controls.fly.movement.damping;
    const dampingControl = gui
      .add(settings, 'flyDamping', damp.min, damp.max, damp.step || 0.0001)
      .name('Translation Damping')
      .onChange((value: number) => {
        sceneManager.setFlyDamping(value);
        saveSettings();
      });

    const rotationDampingControl = gui
      .add(settings, 'flyRotationDamping', 0.9, 0.9999, 0.0001)
      .name('Rotation Damping')
      .onChange((value: number) => {
        sceneManager.setFlyRotationDamping(value);
        saveSettings();
      });

    gui
      .add(settings, 'flyInertialMode')
      .name('Inertial Mode')
      .onChange((value: boolean) => {
        sceneManager.setFlyInertialMode(value);
        saveSettings();
        // Damping only applies with momentum-based movement.
        if (value) {
          dampingControl.show();
          rotationDampingControl.show();
        } else {
          dampingControl.hide();
          rotationDampingControl.hide();
        }
      });

    // Initial damping visibility matches the current inertial state.
    if (!settings.flyInertialMode) {
      dampingControl.hide();
      rotationDampingControl.hide();
    }
  }
  // ortho: no per-mode parameters (handled by the caller's note).
}

/**
 * Build the Navigation popover into `host`. Returns a teardown that disposes
 * the GUI when the popover closes.
 */
export function buildNavigationPopover(
  host: HTMLElement,
  ctx: NavigationPopoverContext
): () => void {
  let gui: ReturnType<typeof makePopoverGui> | null = null;

  // (Re)render the popover for the CURRENT mode — called on open and after a
  // mode switch, so the selector highlight + params always match live state.
  const render = (): void => {
    // A mode switch from the strip removes the focused segment on rebuild; if
    // focus was inside the popover (keyboard user), move it to the new active
    // segment afterward so the tab order is preserved (WCAG 2.4.3).
    const restoreFocus = host.contains(document.activeElement);

    if (gui) {
      gui.destroy();
      gui = null;
    }
    host.replaceChildren();

    const mode = ctx.sceneManager.getControlType();

    // Mode selector at the top — shows + switches the current mode. The switch
    // fires 'luxar-control-mode-changed', which rebuilds this popover (see the
    // listener below) AND refreshes the rail button — so both stay in sync no
    // matter where the switch originates.
    const modeStrip = buildModeSelector(mode, (target) => ctx.setMode(target));
    host.appendChild(modeStrip);

    // The GUI holds the current mode's parameters. It's headerless — the mode
    // selector strip above serves as the popover's header.
    gui = makePopoverGui(host, 'Navigation');
    gui.domElement.classList.add('luxar-gui--headerless');
    buildModeParams(gui, mode, ctx);

    if (mode === 'ortho') {
      const note = document.createElement('div');
      note.className = 'luxar-control-rail__popover-note';
      note.textContent = 'Orthographic projection — pan & zoom only. No parameters.';
      host.appendChild(note);
    }

    if (restoreFocus) {
      modeStrip
        .querySelector<HTMLButtonElement>('.luxar-control-rail__mode-seg.is-active')
        ?.focus();
    }
  };

  render();

  // Rebuild when the mode changes via any path (rail cycle button / V key /
  // this popover's selector) so the selector highlight + params always match.
  const onModeChanged = (): void => render();
  window.addEventListener('luxar-control-mode-changed', onModeChanged);

  return () => {
    window.removeEventListener('luxar-control-mode-changed', onModeChanged);
    if (gui) gui.destroy();
  };
}
