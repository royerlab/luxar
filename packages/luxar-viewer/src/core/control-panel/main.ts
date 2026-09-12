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
import { readUrlParams, type UrlParams } from '../../config/url-params';
import type { DimensionMetadata } from '../../types/dims';
import {
  createControlPanel,
  type ControlPanelPorts,
  type ControlPanelView,
} from '../../ui/control-panel/render-panel';
import { log, Modules } from '../../utils/log';
import {
  ControllerCallError,
  ControllerSocket,
  type ControllerSocketPorts,
  type ControllerStatus,
} from './controller-socket';

/**
 * Idle timeout before the panel sends the display back to its first chapter.
 *
 * The attract behaviour a kiosk needs: a visitor wanders off mid-tour and the
 * exhibit is found at chapter seven with no way back. Two minutes is long
 * enough not to interrupt someone reading.
 */
const IDLE_RESET_MS = 120_000;
/** First retry when the hub is up before the viewer attaches. */
export const CHAPTER_RETRY_BASE_MS = 500;
/** Keep retry traffic bounded while a display remains offline. */
export const CHAPTER_RETRY_MAX_MS = 5_000;

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

export interface ControlPanelBootstrapPorts {
  params?: UrlParams;
  root?: HTMLElement;
  createPanel?: (ports: ControlPanelPorts) => ControlPanelView;
  createSocket?: (ports: ControllerSocketPorts) => ControllerSocket;
  mountPanel?: (url: string, context: CustomPanelContext) => Promise<boolean>;
  onBeforeUnload?: (listener: () => void) => void;
}

type CustomPanelState = 'unmounted' | 'mounting' | 'mounted' | 'failed';
type ConnectedUrlParams = UrlParams & { control: string };

interface StatusHandlerContext {
  params: UrlParams;
  root: HTMLElement;
  panel: ControlPanelView;
  ports: ControlPanelBootstrapPorts;
  socket: () => ControllerSocket | undefined;
  customPanelState: { current: CustomPanelState };
  cancelChapterRetry: (resetDelay?: boolean) => void;
  loadChaptersWithRetry: () => Promise<void>;
}

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

function customPanelOwnsPage(state: CustomPanelState): boolean {
  return state === 'mounting' || state === 'mounted';
}

function startCustomPanel(context: StatusHandlerContext): boolean {
  const custom = context.params.panel;
  const socket = context.socket();
  if (custom === null || context.customPanelState.current === 'failed') return false;
  if (context.customPanelState.current !== 'unmounted') return true;
  if (socket === undefined) return false;

  context.customPanelState.current = 'mounting';
  void (context.ports.mountPanel ?? mountCustomPanel)(custom, {
    socket,
    root: context.root,
    params: context.params,
  }).then((mounted) => {
    context.customPanelState.current = mounted ? 'mounted' : 'failed';
    if (mounted) return;
    context.panel.showMessage(
      'Custom panel failed',
      `Could not mount ${custom}. Falling back to the built-in chapter menu.`
    );
    void context.loadChaptersWithRetry();
  });
  return true;
}

function createStatusHandler(context: StatusHandlerContext): (status: ControllerStatus) => void {
  return (status) => {
    if (status === 'connecting') return;
    if (status === 'open') {
      context.cancelChapterRetry();
      if (startCustomPanel(context)) return;
      void context.loadChaptersWithRetry();
      return;
    }
    context.cancelChapterRetry(false);
    if (!customPanelOwnsPage(context.customPanelState.current)) {
      context.panel.showMessage('Reconnecting', 'Lost the control hub. Trying again...');
    }
  };
}

function startConnectedPanel(
  params: ConnectedUrlParams,
  root: HTMLElement,
  ports: ControlPanelBootstrapPorts
): void {
  let socket: ControllerSocket | undefined;
  let source: ChapterSource | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  let chapterRetryTimer: ReturnType<typeof setTimeout> | undefined;
  let chapterRetryDelayMs = CHAPTER_RETRY_BASE_MS;
  const customPanelState: { current: CustomPanelState } = { current: 'unmounted' };

  const panel = (ports.createPanel ?? createControlPanel)({
    root,
    call: (method, callParams) => socket?.notify(method, callParams),
    onInteraction: () => restartIdleTimer(),
  });

  function restartIdleTimer(): void {
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      const first = source?.chapters[0];
      if (first !== undefined && source !== null) {
        socket?.notify('setDimensionValue', [source.dimensionIndex, first.value]);
      }
    }, IDLE_RESET_MS);
  }

  function cancelChapterRetry(resetDelay = true): void {
    if (chapterRetryTimer !== undefined) clearTimeout(chapterRetryTimer);
    chapterRetryTimer = undefined;
    if (resetDelay) chapterRetryDelayMs = CHAPTER_RETRY_BASE_MS;
  }

  function scheduleChapterRetry(): void {
    if (chapterRetryTimer !== undefined) return;
    const delay = chapterRetryDelayMs;
    chapterRetryTimer = setTimeout(() => {
      chapterRetryTimer = undefined;
      void loadChaptersWithRetry();
    }, delay);
    chapterRetryDelayMs = Math.min(delay * 2, CHAPTER_RETRY_MAX_MS);
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
  }

  async function loadChaptersWithRetry(): Promise<void> {
    try {
      await loadChapters();
      cancelChapterRetry();
    } catch (error) {
      log.warning(Modules.APP, 'control panel: could not load chapters:', error);
      panel.showMessage(
        'Waiting for the display',
        'Connected to the hub, but no viewer has attached yet.'
      );
      if (error instanceof ControllerCallError && error.noViewerAttached) {
        scheduleChapterRetry();
      }
    }
  }

  const onStatus = createStatusHandler({
    params,
    root,
    panel,
    ports,
    socket: () => socket,
    customPanelState,
    cancelChapterRetry,
    loadChaptersWithRetry,
  });

  socket = (ports.createSocket ?? ((socketPorts) => new ControllerSocket(socketPorts)))({
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

  const teardown = (): void => {
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    cancelChapterRetry();
    socket?.dispose();
  };
  (ports.onBeforeUnload ?? ((listener) => window.addEventListener('beforeunload', listener)))(
    teardown
  );
}

export function bootstrap(ports: ControlPanelBootstrapPorts = {}): void {
  const params = ports.params ?? readUrlParams();
  const root = ports.root ?? document.body;
  if (params.control !== null) {
    startConnectedPanel({ ...params, control: params.control }, root, ports);
    return;
  }

  // No hub configured. The honest answer, not an empty grid: this is exactly
  // what an exported folder gets until its launcher grows a relay.
  const panel = (ports.createPanel ?? createControlPanel)({ root, call: () => undefined });
  panel.showMessage(
    'No control hub',
    'This page drives a Luxar viewer over a control hub, and none was given. ' +
      'Serve the scene with luxar serve <scene> --viewer --control, then open ' +
      'the control URL it prints.'
  );
}

bootstrap();
