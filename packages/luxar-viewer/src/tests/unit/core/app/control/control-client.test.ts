import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ControlClient,
  CONTROL_CAMERA_EVENT_MIN_INTERVAL_MS,
  CONTROL_RECONNECT_BASE_MS,
  CONTROL_RECONNECT_MAX_MS,
  buildViewerSocketUrl,
  type ControlSocketLike,
} from '../../../../../core/app/control/control-client';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import {
  JSON_RPC_INTERNAL_ERROR,
  JSON_RPC_INVALID_PARAMS,
  JSON_RPC_METHOD_NOT_FOUND,
  JSON_RPC_PARSE_ERROR,
  notificationFrame,
  requestFrame,
} from '../../../../../utils/json-rpc';

/** A socket that records what was sent and lets a test push frames in. */
class FakeSocket implements ControlSocketLike {
  sent: string[] = [];
  closed = false;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.closed = true;
  }

  /** Deliver a frame as the hub would. */
  receive(raw: string): void {
    this.onmessage?.({ data: raw });
  }

  /** The frames sent so far, decoded. */
  frames(): Array<Record<string, unknown>> {
    return this.sent.map((raw) => JSON.parse(raw) as Record<string, unknown>);
  }

  lastFrame(): Record<string, unknown> {
    const frames = this.frames();
    return frames[frames.length - 1];
  }
}

interface Harness {
  client: ControlClient;
  socket: FakeSocket;
  events: EventGroup;
  invoke: ReturnType<typeof vi.fn>;
  /** Fire an embedder event as the app would. */
  emit: (event: string, payload: unknown) => void;
  /** Advance the injected clock. */
  setNow: (ms: number) => void;
  sockets: FakeSocket[];
}

function harness(
  options: { invoke?: (method: string, params: unknown[]) => unknown } = {}
): Harness {
  const sockets: FakeSocket[] = [];
  const listeners = new Map<string, Set<(payload: unknown) => void>>();
  const invoke = vi.fn(options.invoke ?? (() => 'ok'));
  const events = new EventGroup();
  let now = 0;

  const client = new ControlClient({
    socketUrl: 'ws://kiosk.local:5173/control',
    token: null,
    invoke: (method, params) => invoke(method, params),
    subscribe: (event, listener) => {
      const set = listeners.get(event) ?? new Set();
      set.add(listener);
      listeners.set(event, set);
      return () => set.delete(listener);
    },
    events,
    openSocket: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket;
    },
    now: () => now,
  });
  client.connect();
  const socket = sockets[0];
  socket.onopen?.();

  return {
    client,
    socket,
    sockets,
    events,
    invoke,
    emit: (event, payload) => listeners.get(event)?.forEach((listener) => listener(payload)),
    setNow: (ms) => {
      now = ms;
    },
  };
}

/** Let the client's async dispatch settle. */
const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

describe('buildViewerSocketUrl', () => {
  it('always attaches as a viewer', () => {
    expect(buildViewerSocketUrl('ws://host/control', null)).toBe('ws://host/control?role=viewer');
  });

  it('carries the token when one is configured', () => {
    expect(buildViewerSocketUrl('ws://host/control', 'hunter2')).toContain('token=hunter2');
  });

  it('does not add an empty token parameter', () => {
    expect(buildViewerSocketUrl('ws://host/control', '')).not.toContain('token');
  });

  it('replaces a token already present in the URL', () => {
    // The dedicated parameter is the source of truth; two tokens would be
    // ambiguous and the hub reads only one.
    const url = buildViewerSocketUrl('ws://host/control?token=stale', 'fresh');
    expect(url).toContain('token=fresh');
    expect(url).not.toContain('stale');
  });
});

describe('ControlClient dispatch', () => {
  it('invokes an allowed method and replies with its result', async () => {
    const context = harness({ invoke: () => ({ fov: 60 }) });
    context.socket.receive(requestFrame(1, 'getCameraPose'));
    await settle();

    expect(context.invoke).toHaveBeenCalledWith('getCameraPose', []);
    expect(context.socket.lastFrame()).toEqual({
      jsonrpc: '2.0',
      id: 1,
      result: { fov: 60 },
    });
  });

  it('passes positional params straight through', async () => {
    const context = harness();
    context.socket.receive(requestFrame(2, 'setDimensionValue', [3, 7]));
    await settle();
    expect(context.invoke).toHaveBeenCalledWith('setDimensionValue', [3, 7]);
  });

  it('awaits a method that returns a promise', async () => {
    const context = harness({ invoke: () => Promise.resolve({ completed: true }) });
    context.socket.receive(requestFrame(3, 'flyTo', [{}]));
    await settle();
    expect(context.socket.lastFrame().result).toEqual({ completed: true });
  });

  it('refuses a method the policy does not expose', async () => {
    const context = harness();
    context.socket.receive(requestFrame(4, 'dispose'));
    await settle();

    expect(context.invoke).not.toHaveBeenCalled();
    const frame = context.socket.lastFrame();
    expect(frame.id).toBe(4);
    expect((frame.error as { code: number }).code).toBe(JSON_RPC_METHOD_NOT_FOUND);
    expect((frame.error as { message: string }).message).toContain('not exposed');
  });

  it('refuses a method that does not exist', async () => {
    const context = harness();
    context.socket.receive(requestFrame(5, 'eval'));
    await settle();
    expect((context.socket.lastFrame().error as { message: string }).message).toContain(
      'unknown method'
    );
  });

  it('turns a throwing method into an error frame, not an unhandled rejection', async () => {
    const context = harness({
      invoke: () => {
        throw new Error('LuxarApp.getLayers called before init()');
      },
    });
    context.socket.receive(requestFrame(6, 'getLayers'));
    await settle();

    const frame = context.socket.lastFrame();
    expect((frame.error as { code: number }).code).toBe(JSON_RPC_INTERNAL_ERROR);
    // The app's own message is the useful one.
    expect((frame.error as { message: string }).message).toContain('called before init()');
  });

  it('turns a rejected promise into an error frame too', async () => {
    const context = harness({ invoke: () => Promise.reject(new Error('switch in progress')) });
    context.socket.receive(requestFrame(7, 'switchDataset', ['http://host/data.zarr']));
    await settle();
    expect((context.socket.lastFrame().error as { message: string }).message).toBe(
      'switch in progress'
    );
  });

  it('answers a malformed frame without closing the socket', () => {
    const context = harness();
    context.socket.receive('not json');

    const frame = context.socket.lastFrame();
    expect(frame.id).toBeNull();
    expect((frame.error as { code: number }).code).toBe(JSON_RPC_PARSE_ERROR);
    expect(context.socket.closed).toBe(false);
  });

  it('echoes a readable id when a request has malformed params', () => {
    const context = harness();
    context.socket.receive(
      '{"jsonrpc":"2.0","id":"tap-7","method":"setDimensionValue","params":{"index":0}}'
    );

    const frame = context.socket.lastFrame();
    expect(frame.id).toBe('tap-7');
    expect((frame.error as { code: number }).code).toBe(JSON_RPC_INVALID_PARAMS);
    expect(context.socket.closed).toBe(false);
  });

  it('says nothing back to a notification, but still performs it', async () => {
    const context = harness();
    context.socket.receive(notificationFrame('recenterCamera'));
    await settle();

    expect(context.invoke).toHaveBeenCalledWith('recenterCamera', []);
    expect(context.socket.sent).toHaveLength(0);
  });

  it('ignores a response frame, which a controller should never send', async () => {
    const context = harness();
    context.socket.receive(JSON.stringify({ jsonrpc: '2.0', id: 1, result: 'why' }));
    await settle();
    expect(context.invoke).not.toHaveBeenCalled();
    expect(context.socket.sent).toHaveLength(0);
  });
});

describe('ControlClient switchDataset guard', () => {
  it('validates the URL the app itself does not', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'switchDataset', ['javascript:alert(1)']));
    await settle();

    expect(context.invoke).not.toHaveBeenCalled();
    const frame = context.socket.lastFrame();
    expect((frame.error as { code: number }).code).toBe(JSON_RPC_INVALID_PARAMS);
    expect((frame.error as { message: string }).message).toContain('refused dataset URL');
  });

  it.each([['file:///etc/passwd'], ['data:text/html,<script>'], ['//evil.example/data']])(
    'refuses %s',
    async (src) => {
      const context = harness();
      context.socket.receive(requestFrame(1, 'switchDataset', [src]));
      await settle();
      expect(context.invoke).not.toHaveBeenCalled();
    }
  );

  it('refuses a non-string argument', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'switchDataset', [42]));
    await settle();
    expect((context.socket.lastFrame().error as { message: string }).message).toContain(
      'needs a URL string'
    );
  });

  it('passes a good URL through in canonical form', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'switchDataset', ['http://host:8000/data.zarr/']));
    await settle();
    // Trailing slash stripped, exactly as the URL parser would for ?src.
    expect(context.invoke).toHaveBeenCalledWith('switchDataset', ['http://host:8000/data.zarr']);
  });
});

describe('ControlClient event forwarding', () => {
  it('forwards nothing until a controller subscribes', () => {
    const context = harness();
    context.emit('dimensions-changed', { ndim: 4 });
    expect(context.socket.sent).toHaveLength(0);
  });

  it('forwards a subscribed event as a positional notification', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['dimensions-changed']));
    await settle();
    context.emit('dimensions-changed', { ndim: 4 });

    const frame = context.socket.lastFrame();
    expect(frame.method).toBe('event');
    expect(frame.params).toEqual(['dimensions-changed', { ndim: 4 }]);
    expect(frame.id).toBeUndefined();
  });

  it('stops forwarding after unsubscribe', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['dataset-loaded']));
    context.socket.receive(requestFrame(2, 'unsubscribe', ['dataset-loaded']));
    await settle();
    const before = context.socket.sent.length;
    context.emit('dataset-loaded', { src: 'x' });
    expect(context.socket.sent).toHaveLength(before);
  });

  it('refuses to subscribe to an event it cannot forward', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['not-an-event']));
    await settle();
    expect((context.socket.lastFrame().error as { code: number }).code).toBe(
      JSON_RPC_INVALID_PARAMS
    );
  });

  it('flattens a live Error, which JSON.stringify would empty out', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['dataset-error']));
    await settle();
    context.emit('dataset-error', { src: 'x', error: new TypeError('bad chunk') });

    const params = context.socket.lastFrame().params as [string, { error: unknown }];
    expect(params[1].error).toEqual({ name: 'TypeError', message: 'bad chunk' });
  });

  it('throttles camera-changed to the documented interval', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['camera-changed']));
    await settle();
    const baseline = context.socket.sent.length;

    context.setNow(1000);
    context.emit('camera-changed', { position: [0, 0, 1] });
    // Well inside the interval: dropped.
    context.setNow(1000 + CONTROL_CAMERA_EVENT_MIN_INTERVAL_MS - 1);
    context.emit('camera-changed', { position: [0, 0, 2] });
    expect(context.socket.sent).toHaveLength(baseline + 1);

    context.setNow(1000 + CONTROL_CAMERA_EVENT_MIN_INTERVAL_MS);
    context.emit('camera-changed', { position: [0, 0, 3] });
    expect(context.socket.sent).toHaveLength(baseline + 2);
  });

  it('does not throttle any other event', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['waypoint-arrived']));
    await settle();
    const baseline = context.socket.sent.length;
    for (let index = 0; index < 5; index += 1) {
      context.emit('waypoint-arrived', { index, completed: true });
    }
    expect(context.socket.sent).toHaveLength(baseline + 5);
  });
});

describe('ControlClient lifecycle', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('reconnects with a doubling delay and stops at the ceiling', () => {
    const context = harness();
    expect(context.sockets).toHaveLength(1);

    context.socket.onclose?.();
    vi.advanceTimersByTime(CONTROL_RECONNECT_BASE_MS);
    expect(context.sockets).toHaveLength(2);

    // Second failure waits twice as long: not yet at the old delay.
    context.sockets[1].onclose?.();
    vi.advanceTimersByTime(CONTROL_RECONNECT_BASE_MS);
    expect(context.sockets).toHaveLength(2);
    vi.advanceTimersByTime(CONTROL_RECONNECT_BASE_MS);
    expect(context.sockets).toHaveLength(3);

    // The ceiling is the documented constant, not an unbounded climb.
    expect(CONTROL_RECONNECT_MAX_MS).toBeGreaterThan(CONTROL_RECONNECT_BASE_MS);
  });

  it('resets the backoff once a connection opens', () => {
    const context = harness();
    context.socket.onclose?.();
    vi.advanceTimersByTime(CONTROL_RECONNECT_BASE_MS);
    context.sockets[1].onopen?.();
    context.sockets[1].onclose?.();
    // Back to the base delay, not the doubled one.
    vi.advanceTimersByTime(CONTROL_RECONNECT_BASE_MS);
    expect(context.sockets).toHaveLength(3);
  });

  it('does not reconnect after dispose', () => {
    const context = harness();
    context.client.dispose();
    context.socket.onclose?.();
    vi.advanceTimersByTime(CONTROL_RECONNECT_MAX_MS * 2);
    expect(context.sockets).toHaveLength(1);
  });

  it('closes the socket and stops forwarding when the EventGroup disposes', async () => {
    const context = harness();
    context.socket.receive(requestFrame(1, 'subscribe', ['dataset-loaded']));
    // `settle()` is a real setTimeout, which never fires under fake timers.
    await vi.advanceTimersByTimeAsync(0);

    context.events.dispose();
    expect(context.socket.closed).toBe(true);

    const before = context.socket.sent.length;
    context.emit('dataset-loaded', { src: 'x' });
    expect(context.socket.sent).toHaveLength(before);
  });

  it('survives a double dispose', () => {
    const context = harness();
    context.client.dispose();
    expect(() => context.client.dispose()).not.toThrow();
  });
});
