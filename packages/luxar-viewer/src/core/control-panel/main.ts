/**
 * Entry point for `control.html` — the touch panel.
 *
 * Wiring only, deliberately: read the URL, open the socket, ask the viewer what
 * chapters it has, draw them, and keep the active tile in step. Every decision
 * worth testing lives in a module this file calls —
 * `config/control-panel/derive-chapters` for what the chapters *are*,
 * `ui/control-panel/render-panel` for the DOM, `core/control-panel/controller-socket`
 * for the wire. That split is what keeps this file short enough to read in one
 * go, and the coverage floors honest.
 *
 * The panel drives the viewer through exactly one call — `setDimensionValue`.
 * The authored waypoints do the rest: the camera flies, the dimension-bound
 * overlays swap, the narration cues. See REMOTE_CONTROL_SPEC.md §4.1.
 */

import '../../styles/control-panel.css';

import {
  activeChapterIndex,
  deriveChapters,
  type ChapterSource,
} from '../../config/control-panel/derive-chapters';
import { readUrlParams } from '../../config/url-params';
import type { DimensionMetadata } from '../../types/dims';
import { createControlPanel } from '../../ui/control-panel/render-panel';
import { log, Modules } from '../../utils/log';
import { ControllerSocket, type ControllerStatus } from './controller-socket';

/**
 * Idle timeout before the panel sends the display back to its first chapter.
 *
 * The attract behaviour a kiosk needs: a visitor wanders off mid-tour and the
 * exhibit is found at chapter seven with no way back. Two minutes is long
 * enough not to interrupt someone reading.
 */
const IDLE_RESET_MS = 120_000;

/** Shape of `getDimensions()` over the wire, as far as this page cares. */
interface WireDimensions {
  displayed?: number[];
  metadata?: unknown[];
  ranges?: unknown[];
  currentStep?: number[];
}

function isWireDimensions(value: unknown): value is Required<WireDimensions> {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as WireDimensions;
  return (
    Array.isArray(candidate.displayed) &&
    Array.isArray(candidate.metadata) &&
    Array.isArray(candidate.ranges) &&
    Array.isArray(candidate.currentStep)
  );
}

/**
 * What `?panel=<module>` receives.
 *
 * The supported way to replace this page without forking it: a module default-
 * exporting `mount` gets a live, already-connected socket and the body to draw
 * into, and does whatever it likes. The wire protocol
 * (`docs/guides/specs/REMOTE_CONTROL_SPEC.md` §3.2) is the stable contract
 * underneath; this object just saves a custom panel from re-implementing the
 * connection and the reconnect.
 */
export interface CustomPanelContext {
  socket: ControllerSocket;
  root: HTMLElement;
  /** The page's URL parameters, already validated. */
  params: ReturnType<typeof readUrlParams>;
}

/** The shape `?panel=` modules must default-export. */
export type CustomPanelMount = (context: CustomPanelContext) => void | Promise<void>;

/**
 * Hand control to an authored panel module.
 *
 * Failure is reported on the page rather than swallowed: a kiosk whose custom
 * panel 404s should say so, not sit blank.
 */
async function mountCustomPanel(url: string, context: CustomPanelContext): Promise<boolean> {
  try {
    // @vite-ignore — the specifier is a runtime value by design, validated
    // same-origin by `normalizePanelModuleUrl`.
    const module = (await import(/* @vite-ignore */ url)) as { default?: CustomPanelMount };
    if (typeof module.default !== 'function') {
      throw new Error('the module has no default-exported mount function');
    }
    await module.default(context);
    return true;
  } catch (error) {
    log.warning(Modules.APP, `control panel: ?panel=${url} failed to mount:`, error);
    return false;
  }
}

function bootstrap(): void {
  const params = readUrlParams();
  let socket: ControllerSocket | undefined;
  let source: ChapterSource | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;

  const panel = createControlPanel({
    root: document.body,
    call: (method, callParams) => socket?.notify(method, callParams),
    onInteraction: () => restartIdleTimer(),
  });

  // No hub configured. The honest answer, not an empty grid: this is exactly
  // what an exported folder gets until its launcher grows a relay.
  if (params.control === null) {
    panel.showMessage(
      'No control hub',
      'This page drives a Luxar viewer over a control hub, and none was given. ' +
        'Serve the scene with `luxar serve <scene> --viewer --control`, then open ' +
        'the control URL it prints.'
    );
    return;
  }

  function restartIdleTimer(): void {
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      const first = source?.chapters[0];
      if (first !== undefined && source !== null) {
        socket?.notify('setDimensionValue', [source.dimensionIndex, first.value]);
      }
    }, IDLE_RESET_MS);
  }

  function markActive(position: number | undefined): void {
    if (source === null || position === undefined) return;
    panel.setActive(activeChapterIndex(source, position));
  }

  async function loadChapters(): Promise<void> {
    const dims = await socket?.call('getDimensions');
    if (!isWireDimensions(dims)) {
      panel.showMessage('Unexpected reply', 'The viewer did not describe its dimensions.');
      return;
    }
    source = deriveChapters({
      displayed: dims.displayed,
      metadata: dims.metadata as DimensionMetadata[],
      ranges: dims.ranges as Array<[number, number]>,
    });
    panel.render(source, { title: document.title });
    markActive(dims.currentStep[source?.dimensionIndex ?? -1]);
    await socket?.call('subscribe', ['dimensions-changed']);
    restartIdleTimer();
  }

  const onStatus = (status: ControllerStatus): void => {
    if (status === 'open' && params.panel !== null && socket !== undefined) {
      // An authored panel takes over from here, or we fall back to the built-in
      // one rather than leaving a blank screen on a plinth.
      const custom = params.panel;
      void mountCustomPanel(custom, { socket, root: document.body, params }).then((mounted) => {
        if (!mounted) {
          panel.showMessage(
            'Custom panel failed',
            `Could not mount ${custom}. Falling back to the built-in chapter menu.`
          );
          void loadChapters().catch(() => undefined);
        }
      });
      return;
    }
    if (status === 'open') {
      // A viewer may not be attached yet (the display boots slower than the
      // panel), so a failure here is a state to retry, not an error to show.
      void loadChapters().catch((error: unknown) => {
        log.warning(Modules.APP, 'control panel: could not load chapters:', error);
        panel.showMessage(
          'Waiting for the display',
          'Connected to the hub, but no viewer has attached yet.'
        );
      });
    }
    if (status === 'closed') {
      panel.showMessage('Reconnecting', 'Lost the control hub. Trying again...');
    }
  };

  socket = new ControllerSocket({
    url: params.control,
    token: params.controlToken,
    onStatus,
    onEvent: (name, payload) => {
      if (name !== 'dimensions-changed' || source === null) return;
      const dims = payload as WireDimensions;
      markActive(dims.currentStep?.[source.dimensionIndex]);
    },
  });
  panel.showMessage('Connecting', 'Looking for the control hub...');
  socket.connect();

  window.addEventListener('beforeunload', () => {
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    socket?.dispose();
  });
}

bootstrap();
