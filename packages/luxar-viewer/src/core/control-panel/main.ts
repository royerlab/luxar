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
import {
  validateControlPanelSettings,
  type ControlPanelSettings,
} from '../../config/zarr-bridge/control-panel';
import type { DimensionMetadata } from '../../types/dims';
import {
  createControlPanel,
  type ControlPanelOptions,
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
const DEFAULT_IDLE_RESET_S = 120;
/** First retry when the hub is up before the viewer attaches. */
export const CHAPTER_RETRY_BASE_MS = 500;
/** Keep retry traffic bounded while a display remains offline. */
export const CHAPTER_RETRY_MAX_MS = 5_000;

/**
 * The one line of instruction the panel gives.
 *
 * A kiosk visitor needs to be told the tiles are touchable — an unlabelled
 * grid is often read as a legend rather than a control.
 */
const CONTROL_PANEL_HINT = 'Touch a tile to travel there';

/**
 * Id of the `<style>` element carrying an authored stylesheet.
 *
 * Fixed so a re-render replaces it rather than stacking another copy — a
 * kiosk reconnects for weeks.
 */
const AUTHOR_STYLE_ELEMENT_ID = 'luxar-control-author-style';

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

function registerBeforeUnload(ports: ControlPanelBootstrapPorts, teardown: () => void): void {
  (ports.onBeforeUnload ?? ((listener) => window.addEventListener('beforeunload', listener)))(
    teardown
  );
}

interface StatusHandlerContext {
  params: UrlParams;
  root: HTMLElement;
  panel: ControlPanelView;
  ports: ControlPanelBootstrapPorts;
  socket: () => ControllerSocket | undefined;
  customPanelState: { current: CustomPanelState };
  /**
   * Set once the hub refuses this panel, and never cleared.
   *
   * Shared with the chapter loader because a refusal arrives as a close, which
   * rejects the `getDimensions` already in flight — and that rejection is
   * handled a microtask LATER than the status change. Without this the loader's
   * catch painted over the refusal, so a mistyped token still read as "waiting
   * for the display". Found by loading the page with a wrong token, not by the
   * unit test that drove the status alone.
   */
  connectionState: { refused: boolean };
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
    if (status === 'refused') context.connectionState.refused = true;
    if (customPanelOwnsPage(context.customPanelState.current)) return;
    if (status === 'refused') {
      // Terminal, and worth naming: the hub refused this panel rather than
      // dropping it, so "reconnecting" would be a lie and nobody would think
      // to look at the token they typed on the tablet.
      context.panel.showMessage(
        'Refused by the hub',
        'The control hub would not accept this panel. Check the token in this ' +
          'page URL against the one the server printed.'
      );
      return;
    }
    context.panel.showMessage('Reconnecting', 'Lost the control hub. Trying again...');
  };
}

/**
 * The scene title out of a `getViewerState` reply, or `null`.
 *
 * Separate and pure: the panel is a different page from the display, so the
 * title has to arrive over the wire, and everything about that reply is
 * untrusted shape until checked. `null` for anything missing, non-string or
 * blank, so the caller has one thing to test.
 */
function titleFromViewerState(state: unknown): string | null {
  if (state === null || typeof state !== 'object') return null;
  const candidate = (state as { title?: unknown }).title;
  if (typeof candidate !== 'string') return null;
  const trimmed = candidate.trim();
  return trimmed === '' ? null : trimmed;
}

/**
 * The scene's authored control-panel block out of the same reply, or `null`.
 *
 * The panel may be connected to a third-party hub, so it re-validates the wire
 * copy rather than trusting that it passed through our display's zarr bridge.
 */
function panelConfigFromViewerState(state: unknown): ControlPanelSettings | null {
  if (state === null || typeof state !== 'object') return null;
  const candidate = (state as { controlPanel?: unknown }).controlPanel;
  return validateControlPanelSettings(candidate);
}

/**
 * Apply an authored stylesheet to this page, replacing any earlier one.
 *
 * `textContent`, never `innerHTML`: the CSS came out of a store, so it is
 * untrusted input on the same footing as `overlay_html`. Setting text on a
 * `<style>` element cannot introduce markup however the string is shaped, and
 * the settings validator has already stripped remote fetch constructs.
 */
function applyAuthorStylesheet(css: string | undefined): void {
  const existing = document.getElementById(AUTHOR_STYLE_ELEMENT_ID);
  if (css === undefined || css.trim() === '') {
    existing?.remove();
    return;
  }
  const style = existing ?? document.createElement('style');
  style.id = AUTHOR_STYLE_ELEMENT_ID;
  style.textContent = css;
  if (existing === null) document.head.append(style);
}

/** What the display told us about itself, fetched once. */
interface PresentationCache {
  /** Fetch and remember. A failure leaves it unloaded so a later call retries. */
  load(fetchState: () => Promise<unknown>): Promise<void>;
  readonly title: string | null;
  readonly config: ControlPanelSettings | null;
}

/**
 * Remember the display's title and authored panel block, fetched once.
 *
 * Its own object rather than three variables in the page closure: the caching
 * rule ("succeeded once, never ask again; failed, ask next time") is a real
 * little state machine, and keeping it here means the bootstrap reads as
 * wiring instead of carrying a third concern.
 */
function createPresentationCache(): PresentationCache {
  let title: string | null = null;
  let config: ControlPanelSettings | null = null;
  // A flag rather than a null check on the two values: a scene may legitimately
  // have neither, and testing the values would re-ask on every chapter load
  // for exactly those scenes.
  let loaded = false;
  return {
    async load(fetchState) {
      if (loaded) return;
      try {
        const state = await fetchState();
        title = titleFromViewerState(state);
        config = panelConfigFromViewerState(state);
        applyAuthorStylesheet(config?.stylesheet);
        loaded = true;
      } catch {
        // Presentation is decoration. A panel that refuses to draw because one
        // RPC failed is far worse than one with a derived heading, so this
        // falls through and a later load retries.
      }
    },
    get title() {
      return title;
    },
    get config() {
      return config;
    },
  };
}

/**
 * Assemble the renderer's presentation options.
 *
 * Authored wins, derived fills in — each field INDEPENDENTLY, so a scene that
 * names only its title keeps the built-in touch hint rather than losing it to
 * a single all-or-nothing branch.
 */
function presentationOptions(
  config: ControlPanelSettings | null,
  wireTitle: string | null,
  source: ChapterSource | null
): ControlPanelOptions {
  return {
    title: config?.title ?? wireTitle ?? document.title,
    // No hint when there are no tiles to touch.
    subtitle: source === null ? undefined : (config?.subtitle ?? CONTROL_PANEL_HINT),
    columns: config?.columns ?? null,
    sublabels: chapterSublabels(config),
  };
}

/**
 * Authored per-chapter sublabels, keyed the way the renderer wants them.
 *
 * Returns `undefined` rather than an empty object so the renderer's
 * `sublabels?.[i]` lookup is skipped entirely for an unauthored scene.
 */
function chapterSublabels(config: ControlPanelSettings | null): Record<number, string> | undefined {
  if (config?.chapters === undefined) return undefined;
  const out: Record<number, string> = {};
  for (const [index, override] of Object.entries(config.chapters)) {
    if (override.sublabel !== undefined) out[Number(index)] = override.sublabel;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/**
 * Overwrite derived labels with authored ones, in place.
 *
 * Keyed overrides, not a parallel list: the labels come from the dimension's
 * `categories`, which already hold them once in the store, so an author only
 * names the positions they want to read differently. `authored` is set too, so
 * the stylesheet stops styling the label as machine-generated.
 */
function applyChapterLabels(
  source: ChapterSource | null,
  config: ControlPanelSettings | null
): void {
  if (source === null || config?.chapters === undefined) return;
  for (const chapter of source.chapters) {
    const label = config.chapters[chapter.index]?.label;
    if (label !== undefined) {
      chapter.label = label;
      chapter.authored = true;
    }
  }
}

function startConnectedPanel(
  params: ConnectedUrlParams,
  root: HTMLElement,
  ports: ControlPanelBootstrapPorts
): void {
  let socket: ControllerSocket | undefined;
  let source: ChapterSource | null = null;
  /** The display's title and authored block, fetched once. */
  const presentation = createPresentationCache();
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  let chapterRetryTimer: ReturnType<typeof setTimeout> | undefined;
  let chapterRetryDelayMs = CHAPTER_RETRY_BASE_MS;
  const customPanelState: { current: CustomPanelState } = { current: 'unmounted' };
  const connectionState = { refused: false };

  const panel = (ports.createPanel ?? createControlPanel)({
    root,
    call: (method, callParams) => socket?.notify(method, callParams),
    onInteraction: () => restartIdleTimer(),
  });

  function restartIdleTimer(): void {
    if (idleTimer !== undefined) clearTimeout(idleTimer);
    idleTimer = undefined;
    const idleResetS = presentation.config?.idleResetS ?? DEFAULT_IDLE_RESET_S;
    if (idleResetS === 0) return;
    idleTimer = setTimeout(() => {
      const first = source?.chapters[0];
      if (first !== undefined && source !== null) {
        socket?.notify('setDimensionValue', [source.dimensionIndex, first.value]);
      }
    }, idleResetS * 1000);
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
    // AFTER the dimensions land, and once. Asking earlier would spend a call
    // on every failed retry — and while no viewer is attached the hub answers
    // -32001, so those calls could only fail anyway.
    await presentation.load(() => socket?.call('getViewerState') ?? Promise.resolve(null));
    source = deriveChapters(
      {
        displayed: dims.displayed,
        metadata: dims.metadata as DimensionMetadata[],
        ranges: dims.ranges as Array<[number, number]>,
      },
      // By NAME, so the authoring survives a `Dimensions([...])` reorder.
      { dimensionName: presentation.config?.chapterDimension ?? null }
    );
    applyChapterLabels(source, presentation.config);
    panel.render(source, presentationOptions(presentation.config, presentation.title, source));
    markActive(dims.currentStep[source?.dimensionIndex ?? -1]);
    await socket?.call('subscribe', ['dimensions-changed']);
  }

  async function loadChaptersWithRetry(): Promise<void> {
    try {
      await loadChapters();
      cancelChapterRetry();
    } catch (error) {
      log.warning(Modules.APP, 'control panel: could not load chapters:', error);
      if (error instanceof ControllerCallError && error.noViewerAttached) {
        // The ONLY error that licenses this message: the hub said so.
        panel.showMessage(
          'Waiting for the display',
          'Connected to the hub, but no viewer has attached yet.'
        );
        scheduleChapterRetry();
        return;
      }
      // A refusal or a dropped socket already has a message from the status
      // handler, and it is more specific than anything this catch could say.
      if (connectionState.refused) return;
      panel.showMessage(
        'Could not read the chapters',
        'The hub answered, but the display did not describe its chapters.'
      );
    }
  }

  const onStatus = createStatusHandler({
    params,
    root,
    panel,
    ports,
    socket: () => socket,
    customPanelState,
    connectionState,
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
  registerBeforeUnload(ports, teardown);
}

export function bootstrap(ports: ControlPanelBootstrapPorts = {}): void {
  const params = ports.params ?? readUrlParams();
  const root = ports.root ?? document.body;
  if (params.control !== null) {
    startConnectedPanel({ ...params, control: params.control }, root, ports);
    return;
  }

  // No hub configured. The honest answer, not an empty grid — and it names
  // every host that can provide one, because this page is reached from three
  // of them: a dev checkout, an exported folder, and a native app.
  const panel = (ports.createPanel ?? createControlPanel)({ root, call: () => undefined });
  panel.showMessage(
    'No control hub',
    'This page drives a Luxar viewer over a control hub, and none was given. ' +
      'Start one with luxar serve <scene> --viewer --control, or, in an ' +
      'exported folder, python serve.py --control — then open the control ' +
      'URL it prints.'
  );
}

bootstrap();
