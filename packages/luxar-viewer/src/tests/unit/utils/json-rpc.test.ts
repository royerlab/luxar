import { describe, it, expect } from 'vitest';

import {
  decodeFrame,
  errorFrame,
  notificationFrame,
  requestFrame,
  successFrame,
  JSON_RPC_INVALID_PARAMS,
  JSON_RPC_INVALID_REQUEST,
  JSON_RPC_PARSE_ERROR,
  JSON_RPC_VERSION,
} from '../../../utils/json-rpc';

describe('decodeFrame', () => {
  it('decodes a request with positional params', () => {
    const frame = decodeFrame(requestFrame(7, 'setDimensionValue', [3, 12]));
    expect(frame).toEqual({
      kind: 'request',
      id: 7,
      method: 'setDimensionValue',
      params: [3, 12],
    });
  });

  it('decodes a string id, since a controller may number requests its own way', () => {
    const frame = decodeFrame(requestFrame('tap-4', 'recenterCamera'));
    expect(frame).toMatchObject({ kind: 'request', id: 'tap-4', params: [] });
  });

  it('treats a missing id as a notification', () => {
    expect(decodeFrame(notificationFrame('resize'))).toEqual({
      kind: 'notification',
      method: 'resize',
      params: [],
    });
  });

  it('treats an explicit null id as a notification too', () => {
    // The hub emits events this way, and the spec allows both spellings.
    const raw = JSON.stringify({ jsonrpc: '2.0', id: null, method: 'event', params: [1] });
    expect(decodeFrame(raw)).toEqual({ kind: 'notification', method: 'event', params: [1] });
  });

  it('decodes a success response, preserving a falsy result', () => {
    // `result: false` and `result: 0` must not be mistaken for "no result".
    const raw = JSON.stringify({ jsonrpc: '2.0', id: 1, result: false });
    expect(decodeFrame(raw)).toEqual({ kind: 'response', id: 1, result: false });
  });

  it('decodes a success response carrying an explicit null result', () => {
    expect(decodeFrame(successFrame(2, undefined))).toEqual({
      kind: 'response',
      id: 2,
      result: null,
    });
  });

  it('decodes an error response with its data member', () => {
    const frame = decodeFrame(errorFrame(5, -32601, 'unknown method', { method: 'nope' }));
    expect(frame).toEqual({
      kind: 'response',
      id: 5,
      error: { code: -32601, message: 'unknown method', data: { method: 'nope' } },
    });
  });

  describe('rejections', () => {
    it.each([
      ['not json', 'null-ish garbage', 'definitely { not json', JSON_RPC_PARSE_ERROR],
      [
        'a batch array',
        'no batching',
        '[{"jsonrpc":"2.0","method":"a"}]',
        JSON_RPC_INVALID_REQUEST,
      ],
      ['a bare scalar', 'frames are objects', '42', JSON_RPC_INVALID_REQUEST],
      [
        'a missing version',
        'the version is mandatory',
        '{"method":"a","id":1}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'a wrong version',
        'JSON-RPC 1.0 is not this protocol',
        '{"jsonrpc":"1.0","method":"a","id":1}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'an empty method',
        'an empty name can match nothing',
        '{"jsonrpc":"2.0","method":"","id":1}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'a non-string method',
        'a method is a name',
        '{"jsonrpc":"2.0","method":7,"id":1}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'object params',
        'positional only, so a named form cannot silently become no arguments',
        '{"jsonrpc":"2.0","method":"a","id":1,"params":{"index":0}}',
        JSON_RPC_INVALID_PARAMS,
      ],
      [
        'a float id',
        'a float id round-trips badly across languages',
        '{"jsonrpc":"2.0","method":"a","id":1.5}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'a response with no id',
        'an unroutable reply',
        '{"jsonrpc":"2.0","result":1}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'an error member that is not an error',
        'a code and a message are the contract',
        '{"jsonrpc":"2.0","id":1,"error":"boom"}',
        JSON_RPC_INVALID_REQUEST,
      ],
      [
        'an error member missing its code',
        'a code and a message are the contract',
        '{"jsonrpc":"2.0","id":1,"error":{"message":"boom"}}',
        JSON_RPC_INVALID_REQUEST,
      ],
    ])('refuses %s (%s)', (_label, _why, raw, code) => {
      const frame = decodeFrame(raw);
      expect(frame.kind).toBe('malformed');
      if (frame.kind !== 'malformed') throw new Error('unreachable');
      expect(frame.code).toBe(code);
      expect(frame.message).toBeTruthy();
    });

    it('never throws, whatever it is handed', () => {
      // The socket must survive a bad frame; throwing here would tear it down.
      for (const raw of ['', String.fromCharCode(0), '{', '[]', 'undefined', '"x"']) {
        expect(() => decodeFrame(raw)).not.toThrow();
        expect(decodeFrame(raw).kind).toBe('malformed');
      }
    });

    it('preserves a readable request id on malformed params', () => {
      const frame = decodeFrame(
        '{"jsonrpc":"2.0","method":"setDimensionValue","id":"tap-7","params":{"index":0}}'
      );
      expect(frame).toEqual({
        kind: 'malformed',
        id: 'tap-7',
        code: JSON_RPC_INVALID_PARAMS,
        message: 'params must be an array (positional)',
      });
    });
  });
});

describe('encoders', () => {
  it('stamps the version on every frame', () => {
    for (const raw of [
      requestFrame(1, 'a'),
      notificationFrame('a'),
      successFrame(1, 'x'),
      errorFrame(1, -1, 'x'),
    ]) {
      expect(JSON.parse(raw).jsonrpc).toBe(JSON_RPC_VERSION);
    }
  });

  it('omits the data member when there is none', () => {
    expect(Object.keys(JSON.parse(errorFrame(1, -1, 'x')).error)).toEqual(['code', 'message']);
  });

  it('allows a null id on an error, for a frame whose id could not be read', () => {
    expect(JSON.parse(errorFrame(null, JSON_RPC_PARSE_ERROR, 'parse error')).id).toBeNull();
  });

  it('round-trips every encoder through the decoder', () => {
    expect(decodeFrame(requestFrame(1, 'flyTo', [{ fov: 60 }]))).toMatchObject({
      kind: 'request',
      method: 'flyTo',
      params: [{ fov: 60 }],
    });
    expect(decodeFrame(successFrame('a', [1, 2]))).toMatchObject({
      kind: 'response',
      result: [1, 2],
    });
  });
});
