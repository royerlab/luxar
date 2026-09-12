// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { readUrlParams, type UrlParams } from '../../../../config/url-params';
import type { ControllerSocketPorts } from '../../../../core/control-panel/controller-socket';
import {
  CHAPTER_RETRY_BASE_MS,
  CHAPTER_RETRY_MAX_MS,
  bootstrap,
  type ControlPanelBootstrapPorts,
} from '../../../../core/control-panel/main';
import {
  ControllerCallError,
  type ControllerSocket,
} from '../../../../core/control-panel/controller-socket';
import { MAX_CONTROL_STYLESHEET_CHARS } from '../../../../config/zarr-bridge/control-panel';
import type {
  ControlPanelPorts,
  ControlPanelView,
} from '../../../../ui/control-panel/render-panel';

const ORIGIN = { protocol: 'http:', host: 'kiosk.local:5173' };
const DIMS = {
  displayed: [0, 1, 2],
  metadata: [
    { name: 'x', unit: '', scale: 1, spatial: true },
    { name: 'y', unit: '', scale: 1, spatial: true },
    { name: 'z', unit: '', scale: 1, spatial: true },
    {
      name: 'story',
      unit: '',
      scale: 1,
      discrete: true,
      step: 1,
      categories: ['Overview', 'Haemoglobin'],
    },
  ],
  ranges: [
    [0, 100],
    [0, 100],
    [0, 100],
    [0, 1],
  ],
  currentStep: [0, 0, 0, 1],
};

function params(search = '?control'): UrlParams {
  return readUrlParams(search, ORIGIN);
}

async function settle(): Promise<void> {
  // Drain the microtask queue rather than counting ticks. How many `await`s
  // the bootstrap chains before it paints is an implementation detail, and a
  // fixed count silently stops being enough the moment one is added — which
  // showed up as `render` never being called rather than as anything that
  // pointed at the cause. Plain microtask ticks (not a `setTimeout`) because
  // several of these tests install fake timers.
  for (let tick = 0; tick < 20; tick += 1) await Promise.resolve();
}

function harness(overrides: Partial<ControlPanelBootstrapPorts> = {}) {
  let socketPorts: ControllerSocketPorts | undefined;
  let panelPorts: ControlPanelPorts | undefined;
  let teardown: (() => void) | undefined;
  const panel: ControlPanelView = {
    render: vi.fn(),
    setActive: vi.fn(),
    showMessage: vi.fn(),
    dispose: vi.fn(),
  };
  const socket = {
    call: vi.fn(async (method: string) => (method === 'getDimensions' ? DIMS : undefined)),
    notify: vi.fn(),
    connect: vi.fn(),
    dispose: vi.fn(),
  } as unknown as ControllerSocket;

  bootstrap({
    params: params(),
    root: document.body,
    createPanel: (ports) => {
      panelPorts = ports;
      return panel;
    },
    createSocket: (ports) => {
      socketPorts = ports;
      return socket;
    },
    onBeforeUnload: (listener) => {
      teardown = listener;
    },
    ...overrides,
  });

  return {
    panel,
    socket,
    get socketPorts() {
      if (socketPorts === undefined) throw new Error('socket was not created');
      return socketPorts;
    },
    get panelPorts() {
      if (panelPorts === undefined) throw new Error('panel was not created');
      return panelPorts;
    },
    get teardown() {
      if (teardown === undefined) throw new Error('teardown was not registered');
      return teardown;
    },
  };
}

beforeEach(() => {
  document.body.replaceChildren();
  document.getElementById('luxar-control-author-style')?.remove();
  document.title = 'Protein stories';
});

afterEach(() => {
  vi.useRealTimers();
});

describe('control-panel bootstrap', () => {
  it('shows an actionable no-hub message without literal markdown', () => {
    const createSocket = vi.fn();
    const context = harness({ params: params(''), createSocket });

    expect(context.panel.showMessage).toHaveBeenCalledWith(
      'No control hub',
      expect.stringContaining('luxar serve <scene> --viewer --control')
    );
    // This page is reached from an exported folder too, where that command
    // does not exist and `python serve.py --control` is the one that does.
    expect(context.panel.showMessage).toHaveBeenCalledWith(
      'No control hub',
      expect.stringContaining('python serve.py --control')
    );
    expect(context.panel.showMessage).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('`')
    );
    expect(createSocket).not.toHaveBeenCalled();
  });

  it('loads chapters on open and arms idle reset only after interaction', async () => {
    vi.useFakeTimers();
    const context = harness();

    context.socketPorts.onStatus?.('open');
    await settle();

    expect(context.panel.render).toHaveBeenCalledWith(
      expect.objectContaining({ dimensionIndex: 3 }),
      // `objectContaining`, not an exact match: presentation options grow
      // (columns, sublabels, authored overrides) and this test is about the
      // title and the hint, so an exact match would fail for unrelated
      // additions. The scene's own title comes over the wire; the hint is what
      // tells a visitor the tiles are touchable.
      expect.objectContaining({
        title: 'Protein stories',
        subtitle: 'Touch a tile to travel there',
      })
    );
    expect(context.panel.setActive).toHaveBeenCalledWith(1);
    // Third: `getDimensions`, then `getViewerState` for the title, then this.
    expect(context.socket.call).toHaveBeenNthCalledWith(3, 'subscribe', ['dimensions-changed']);

    await vi.advanceTimersByTimeAsync(120_000);
    expect(context.socket.notify).not.toHaveBeenCalled();

    context.panelPorts.onInteraction?.();
    await vi.advanceTimersByTimeAsync(120_000);
    expect(context.socket.notify).toHaveBeenCalledWith('setDimensionValue', [3, 0]);
  });

  it('applies authored presentation and replaces the author stylesheet', async () => {
    const existing = document.createElement('style');
    existing.id = 'luxar-control-author-style';
    existing.textContent = '.old { color: red; }';
    document.head.append(existing);
    const context = harness();
    vi.mocked(context.socket.call).mockImplementation(async (method: string) => {
      if (method === 'getDimensions') return structuredClone(DIMS);
      if (method === 'getViewerState') {
        return {
          title: 'Wire title',
          controlPanel: {
            title: 'Authored title',
            subtitle: 'Authored subtitle',
            columns: 2,
            stylesheet: '.tile { color: blue; }',
            chapters: { 1: { label: 'Blood', sublabel: 'Protein family' } },
          },
        };
      }
      return undefined;
    });

    context.socketPorts.onStatus?.('open');
    await settle();

    const [source, options] = vi.mocked(context.panel.render).mock.calls[0] ?? [];
    expect(source?.chapters[1]).toMatchObject({ label: 'Blood', authored: true });
    expect(options).toMatchObject({
      title: 'Authored title',
      subtitle: 'Authored subtitle',
      columns: 2,
      sublabels: { 1: 'Protein family' },
    });
    const styles = document.querySelectorAll('#luxar-control-author-style');
    expect(styles).toHaveLength(1);
    expect(styles[0]?.textContent).toBe('.tile { color: blue; }');
  });

  it('validates the wire copy of author CSS before applying it', async () => {
    const context = harness();
    vi.mocked(context.socket.call).mockImplementation(async (method: string) => {
      if (method === 'getDimensions') return DIMS;
      if (method === 'getViewerState') {
        return {
          controlPanel: {
            stylesheet: '.tile { background: image-set("https://evil.example/b.png" 1x); }',
          },
        };
      }
      return undefined;
    });

    context.socketPorts.onStatus?.('open');
    await settle();

    expect(document.getElementById('luxar-control-author-style')?.textContent).not.toContain(
      'https://evil.example'
    );

    context.teardown();
    document.getElementById('luxar-control-author-style')?.remove();
    const oversized = harness();
    vi.mocked(oversized.socket.call).mockImplementation(async (method: string) => {
      if (method === 'getDimensions') return DIMS;
      if (method === 'getViewerState') {
        return { controlPanel: { stylesheet: 'x'.repeat(MAX_CONTROL_STYLESHEET_CHARS + 1) } };
      }
      return undefined;
    });
    oversized.socketPorts.onStatus?.('open');
    await settle();
    expect(document.getElementById('luxar-control-author-style')).toBeNull();
  });

  it('does not arm an idle reset when the scene disables it', async () => {
    vi.useFakeTimers();
    const context = harness();
    vi.mocked(context.socket.call).mockImplementation(async (method: string) => {
      if (method === 'getDimensions') return DIMS;
      if (method === 'getViewerState') return { controlPanel: { idleResetS: 0 } };
      return undefined;
    });
    context.socketPorts.onStatus?.('open');
    await settle();

    context.panelPorts.onInteraction?.();
    await vi.runAllTimersAsync();

    expect(context.socket.notify).not.toHaveBeenCalled();
  });

  it('keeps the connecting message while a socket dial is in progress', () => {
    const context = harness();
    expect(context.panel.showMessage).toHaveBeenLastCalledWith(
      'Connecting',
      'Looking for the control hub...'
    );

    context.socketPorts.onStatus?.('connecting');
    expect(context.panel.showMessage).toHaveBeenCalledTimes(1);
  });

  it('retries a missing viewer with bounded backoff until chapters load', async () => {
    vi.useFakeTimers();
    const context = harness();
    vi.mocked(context.socket.call)
      .mockRejectedValueOnce(new ControllerCallError({ code: -32001, message: 'no viewer' }))
      .mockRejectedValueOnce(new ControllerCallError({ code: -32001, message: 'no viewer' }))
      .mockImplementation(async (method: string) =>
        method === 'getDimensions' ? DIMS : undefined
      );

    context.socketPorts.onStatus?.('open');
    await settle();
    expect(context.panel.showMessage).toHaveBeenCalledWith(
      'Waiting for the display',
      expect.any(String)
    );

    await vi.advanceTimersByTimeAsync(CHAPTER_RETRY_BASE_MS);
    expect(context.socket.call).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(CHAPTER_RETRY_BASE_MS * 2);
    await settle();

    expect(context.socket.call).toHaveBeenCalledWith('subscribe', ['dimensions-changed']);
    expect(context.panel.render).toHaveBeenCalledTimes(1);
  });

  it('caps the missing-viewer retry interval', async () => {
    vi.useFakeTimers();
    const context = harness();
    vi.mocked(context.socket.call).mockRejectedValue(
      new ControllerCallError({ code: -32001, message: 'no viewer' })
    );

    context.socketPorts.onStatus?.('open');
    await settle();
    const delays = [
      CHAPTER_RETRY_BASE_MS,
      CHAPTER_RETRY_BASE_MS * 2,
      CHAPTER_RETRY_BASE_MS * 4,
      CHAPTER_RETRY_BASE_MS * 8,
      CHAPTER_RETRY_MAX_MS,
      CHAPTER_RETRY_MAX_MS,
    ];
    for (const [index, delay] of delays.entries()) {
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(context.socket.call).toHaveBeenCalledTimes(index + 1);
      await vi.advanceTimersByTimeAsync(1);
      expect(context.socket.call).toHaveBeenCalledTimes(index + 2);
    }
  });

  it('cancels a pending chapter retry during teardown', async () => {
    vi.useFakeTimers();
    const context = harness();
    vi.mocked(context.socket.call).mockRejectedValue(
      new ControllerCallError({ code: -32001, message: 'no viewer' })
    );

    context.socketPorts.onStatus?.('open');
    await settle();
    context.teardown();
    await vi.runAllTimersAsync();

    expect(context.socket.call).toHaveBeenCalledTimes(1);
    expect(context.socket.dispose).toHaveBeenCalledTimes(1);
  });

  it('cancels an armed idle reset during teardown', async () => {
    vi.useFakeTimers();
    const context = harness();
    context.socketPorts.onStatus?.('open');
    await settle();

    context.panelPorts.onInteraction?.();
    context.teardown();
    await vi.runAllTimersAsync();

    expect(context.socket.notify).not.toHaveBeenCalled();
  });

  it('mounts a custom panel once and never overwrites its page on reconnect', async () => {
    let finishMount: ((mounted: boolean) => void) | undefined;
    const mountPanel = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finishMount = resolve;
        })
    );
    const context = harness({ params: params('?control&panel=/panel.js'), mountPanel });

    context.socketPorts.onStatus?.('open');
    context.socketPorts.onStatus?.('closed');
    expect(context.panel.showMessage).not.toHaveBeenCalledWith('Reconnecting', expect.any(String));

    finishMount?.(true);
    await settle();
    context.socketPorts.onStatus?.('open');

    expect(mountPanel).toHaveBeenCalledTimes(1);
    expect(context.socket.call).not.toHaveBeenCalled();
  });

  it('names a refusal instead of promising to reconnect', async () => {
    const context = harness({ params: params('?control') });

    context.socketPorts.onStatus?.('refused');

    const [title, body] = (
      context.panel.showMessage as unknown as {
        mock: { calls: string[][] };
      }
    ).mock.calls.at(-1)!;
    expect(title).toBe('Refused by the hub');
    // The one thing an operator can act on: the token they typed on the tablet.
    expect(body).toContain('token');
    expect(context.panel.showMessage).not.toHaveBeenCalledWith('Reconnecting', expect.any(String));
  });

  it('keeps the refusal message when the in-flight call rejects after it', async () => {
    // The real sequence with a wrong token, which the status-only test above
    // does NOT reproduce: the socket OPENS (the hub accepts a handshake it
    // means to refuse), the panel asks for dimensions, then the close arrives
    // and rejects that call a microtask later. The loader's catch used to
    // paint over the refusal, so a mistyped token read as "waiting for the
    // display" and nobody thought to look at the token.
    const context = harness({ params: params('?control') });
    vi.mocked(context.socket.call).mockRejectedValue(new Error('the hub refused this panel'));

    context.socketPorts.onStatus?.('open');
    context.socketPorts.onStatus?.('refused');
    await settle();

    expect(context.panel.showMessage).toHaveBeenLastCalledWith(
      'Refused by the hub',
      expect.stringContaining('token')
    );
    expect(context.panel.showMessage).not.toHaveBeenCalledWith(
      'Waiting for the display',
      expect.any(String)
    );
  });

  it('only claims the display is missing when the hub said so', async () => {
    // `-32001` is the hub answering "nothing to drive". Any other failure is
    // something else, and asserting "connected, no viewer yet" for all of them
    // was a claim the code had not established.
    const context = harness({ params: params('?control') });
    vi.mocked(context.socket.call).mockResolvedValue({ not: 'dimensions' });

    context.socketPorts.onStatus?.('open');
    await settle();

    expect(context.panel.showMessage).not.toHaveBeenCalledWith(
      'Waiting for the display',
      expect.any(String)
    );
  });

  it('falls back once after a custom panel fails, without remounting it', async () => {
    const mountPanel = vi.fn(async () => false);
    const context = harness({ params: params('?control&panel=/panel.js'), mountPanel });

    context.socketPorts.onStatus?.('open');
    await settle();
    expect(context.panel.showMessage).toHaveBeenCalledWith(
      'Custom panel failed',
      expect.stringContaining('Falling back')
    );
    expect(context.panel.render).toHaveBeenCalledTimes(1);

    context.socketPorts.onStatus?.('closed');
    context.socketPorts.onStatus?.('open');
    await settle();
    expect(mountPanel).toHaveBeenCalledTimes(1);
    expect(context.panel.render).toHaveBeenCalledTimes(2);
  });

  it('tracks dimension events and ignores unrelated or incomplete payloads', async () => {
    const context = harness();
    context.socketPorts.onStatus?.('open');
    await settle();
    vi.mocked(context.panel.setActive).mockClear();

    context.socketPorts.onEvent?.('camera-changed', { currentStep: [0, 0, 0, 0] });
    context.socketPorts.onEvent?.('dimensions-changed', {});
    context.socketPorts.onEvent?.('dimensions-changed', { currentStep: [0, 0, 0, 0] });

    expect(context.panel.setActive).toHaveBeenCalledTimes(1);
    expect(context.panel.setActive).toHaveBeenCalledWith(0);
  });

  it('reports a malformed dimensions reply without rendering chapters', async () => {
    const context = harness();
    vi.mocked(context.socket.call).mockResolvedValue({ displayed: [] });

    context.socketPorts.onStatus?.('open');
    await settle();

    expect(context.panel.showMessage).toHaveBeenCalledWith(
      'Unexpected reply',
      'The viewer did not describe its dimensions.'
    );
    expect(context.panel.render).not.toHaveBeenCalled();
  });
});
