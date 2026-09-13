import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ControlSocketLike } from '../../../../core/app/control/control-client';
import { CLOSE_POLICY_VIOLATION } from '../../../../config/control-contract';
import {
  buildControllerSocketUrl,
  CONTROLLER_CALL_TIMEOUT_MS,
  CONTROLLER_RECONNECT_BASE_MS,
  CONTROLLER_RECONNECT_MAX_MS,
  ControllerCallError,
  ControllerSocket,
} from '../../../../core/control-panel/controller-socket';
import { errorFrame, notificationFrame, successFrame } from '../../../../utils/json-rpc';

class FakeSocket implements ControlSocketLike {
  readonly sent: string[] = [];
  closed = false;
  onopen: (() => void) | null = null;
  onclose: ((event?: { code?: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.closed = true;
  }

  open(): void {
    this.onopen?.();
  }

  disconnect(code?: number): void {
    this.onclose?.(code === undefined ? undefined : { code });
  }

  receive(data: unknown): void {
    this.onmessage?.({ data });
  }
}

function harness() {
  const sockets: FakeSocket[] = [];
  const onStatus = vi.fn();
  const onEvent = vi.fn();
  const openSocket = vi.fn(() => {
    const socket = new FakeSocket();
    sockets.push(socket);
    return socket;
  });
  const controller = new ControllerSocket({
    url: 'ws://kiosk.local/control',
    token: 'space & slash/',
    onStatus,
    onEvent,
    openSocket,
  });
  controller.connect();
  return { controller, sockets, onStatus, onEvent, openSocket };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('buildControllerSocketUrl', () => {
  it('sets the controller role and quotes the configured token', () => {
    expect(buildControllerSocketUrl('ws://host/control?token=stale', 'space & slash/')).toBe(
      'ws://host/control?token=space+%26+slash%2F&role=controller'
    );
  });

  it('does not add an empty token', () => {
    expect(buildControllerSocketUrl('ws://host/control', '')).toBe(
      'ws://host/control?role=controller'
    );
  });
});

describe('ControllerSocket calls', () => {
  it('resolves a call from its matching response', async () => {
    const { controller, sockets } = harness();
    sockets[0].open();

    const result = controller.call('getDimensions');
    expect(JSON.parse(sockets[0].sent[0])).toMatchObject({ id: 1, method: 'getDimensions' });
    sockets[0].receive(successFrame(1, { displayed: [0, 1, 2] }));

    await expect(result).resolves.toEqual({ displayed: [0, 1, 2] });
  });

  it('rejects a call with the JSON-RPC code intact', async () => {
    const { controller, sockets } = harness();
    const result = controller.call('getDimensions');
    sockets[0].receive(errorFrame(1, -32001, 'no viewer attached'));

    await expect(result).rejects.toMatchObject({
      name: 'ControllerCallError',
      code: -32001,
      noViewerAttached: true,
    });
    expect(new ControllerCallError({ code: -32603, message: 'failed' }).noViewerAttached).toBe(
      false
    );
  });

  it('times out a call and ignores its late reply', async () => {
    vi.useFakeTimers();
    const { controller, sockets } = harness();
    const result = controller.call('flyTo');
    const timedOut = expect(result).rejects.toThrow(
      `flyTo did not answer within ${CONTROLLER_CALL_TIMEOUT_MS}ms`
    );

    await vi.advanceTimersByTimeAsync(CONTROLLER_CALL_TIMEOUT_MS);
    await timedOut;
    sockets[0].receive(successFrame(1, 'too late'));

    const next = controller.call('getCameraPose');
    sockets[0].receive(successFrame(2, { fov: 60 }));
    await expect(next).resolves.toEqual({ fov: 60 });
  });

  it('rejects every outstanding call when the socket closes', async () => {
    const { controller, sockets } = harness();
    const first = controller.call('getDimensions');
    const second = controller.call('getCameraPose');
    sockets[0].disconnect();

    await expect(first).rejects.toThrow('the control socket closed');
    await expect(second).rejects.toThrow('the control socket closed');
  });

  it('refuses calls before a socket exists', async () => {
    const controller = new ControllerSocket({ url: 'ws://host/control', token: null });
    await expect(controller.call('getDimensions')).rejects.toThrow(
      'not connected; cannot call getDimensions'
    );
  });
});

describe('ControllerSocket events and lifecycle', () => {
  it('sends notifications and dispatches viewer events', () => {
    const { controller, sockets, onEvent, onStatus } = harness();
    sockets[0].open();
    controller.notify('setDimensionValue', [3, 2]);
    sockets[0].receive(notificationFrame('event', ['dimensions-changed', { currentStep: [2] }]));

    expect(JSON.parse(sockets[0].sent[0])).toEqual({
      jsonrpc: '2.0',
      method: 'setDimensionValue',
      params: [3, 2],
    });
    expect(onEvent).toHaveBeenCalledWith('dimensions-changed', { currentStep: [2] });
    expect(onStatus.mock.calls.map(([status]) => status)).toEqual(['connecting', 'open']);
  });

  it('ignores non-string messages and malformed or unrelated frames', () => {
    const { sockets, onEvent } = harness();
    sockets[0].receive(new Uint8Array([1, 2, 3]));
    sockets[0].receive('not json');
    sockets[0].receive(notificationFrame('other', ['dimensions-changed', {}]));
    sockets[0].receive(notificationFrame('event', [42, {}]));
    expect(onEvent).not.toHaveBeenCalled();
  });

  it('doubles reconnect delays and resets them after an open', async () => {
    vi.useFakeTimers();
    const { sockets, openSocket } = harness();

    sockets[0].disconnect();
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_BASE_MS - 1);
    expect(openSocket).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(openSocket).toHaveBeenCalledTimes(2);

    sockets[1].disconnect();
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_BASE_MS);
    expect(openSocket).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_BASE_MS);
    expect(openSocket).toHaveBeenCalledTimes(3);

    sockets[2].open();
    sockets[2].disconnect();
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_BASE_MS);
    expect(openSocket).toHaveBeenCalledTimes(4);
  });

  it('reports a policy-violation close as refused and stops retrying', async () => {
    vi.useFakeTimers();
    const { sockets, openSocket, onStatus } = harness();

    // 1008 is what the hub closes a wrong token, an unknown role or a foreign
    // origin with. Retrying cannot change any of those, and the panel used to
    // report all three as "Reconnecting" forever.
    sockets[0].disconnect(CLOSE_POLICY_VIOLATION);

    expect(onStatus).toHaveBeenLastCalledWith('refused');
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_MAX_MS * 4);
    expect(openSocket).toHaveBeenCalledTimes(1);
  });

  it('still retries an ordinary close, and settles pending work as refused', async () => {
    vi.useFakeTimers();
    const { controller, sockets, openSocket, onStatus } = harness();
    const pending = controller.call('getDimensions');
    const refused = expect(pending).rejects.toThrow('the hub refused this panel');

    sockets[0].disconnect(CLOSE_POLICY_VIOLATION);
    await refused;

    // A close with no code at all (a dropped network) must still reconnect —
    // the refusal path must not have swallowed the ordinary one.
    const second = new ControllerSocket({ url: 'ws://host/control', token: null, openSocket });
    second.connect();
    sockets[1].disconnect();
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_BASE_MS);
    expect(openSocket).toHaveBeenCalledTimes(3);
    expect(onStatus).toHaveBeenCalledWith('refused');
  });

  it('retries when opening the platform socket throws', async () => {
    vi.useFakeTimers();
    const openSocket = vi
      .fn<() => ControlSocketLike>()
      .mockImplementationOnce(() => {
        throw new Error('dial failed');
      })
      .mockImplementation(() => new FakeSocket());
    const controller = new ControllerSocket({
      url: 'ws://host/control',
      token: null,
      openSocket,
    });

    controller.connect();
    await vi.advanceTimersByTimeAsync(CONTROLLER_RECONNECT_BASE_MS);
    expect(openSocket).toHaveBeenCalledTimes(2);
  });

  it('dispose rejects pending work, closes once, and cancels reconnect', async () => {
    vi.useFakeTimers();
    const { controller, sockets, openSocket } = harness();
    const pending = controller.call('getDimensions');
    const disposed = expect(pending).rejects.toThrow('the controller was disposed');

    controller.dispose();
    await disposed;
    controller.dispose();
    controller.connect();
    await vi.runAllTimersAsync();

    expect(sockets[0].closed).toBe(true);
    expect(openSocket).toHaveBeenCalledTimes(1);
  });

  it('dispose cancels a reconnect already scheduled by a remote close', async () => {
    vi.useFakeTimers();
    const { controller, sockets, openSocket } = harness();
    sockets[0].disconnect();
    controller.dispose();

    await vi.runAllTimersAsync();
    expect(openSocket).toHaveBeenCalledTimes(1);
  });
});
