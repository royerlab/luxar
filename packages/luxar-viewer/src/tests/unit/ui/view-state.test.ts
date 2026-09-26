/**
 * Unit tests for the view-state document: JSON parse/validate of the layers
 * and camera blocks, the Neuroglancer-style `#!<json>` fragment codec, the
 * fragment clear on dataset switch, and the debounced URL writer.
 */

import { describe, it, expect, vi } from 'vitest';
import type { CameraSnapshot } from '../../../core/app/snapshot/viewer-snapshot';
import {
  VIEW_STATE_VERSION,
  clearViewStateHash,
  isEmptyViewState,
  parseViewState,
  readViewStateHash,
  startViewStateUrlSync,
  writeViewStateHash,
  type ViewStateDoc,
} from '../../../ui/view-state';

const CAMERA: CameraSnapshot = {
  position: [1, 2, 3],
  target: [0, 0, 0],
  up: [0, -1, 0],
  isOrtho: false,
  fov: 40,
  near: 0.1,
  far: 1000,
};

const DOC: ViewStateDoc = {
  version: VIEW_STATE_VERSION,
  layers: { c0: { visible: false, blendingMode: 'max', colormap: 'viridis', layerOrder: null } },
  camera: CAMERA,
};

describe('parseViewState', () => {
  it('round-trips a document through JSON', () => {
    expect(parseViewState(JSON.stringify(DOC))).toEqual(DOC);
  });

  it('accepts a layers-only file and a camera-only document', () => {
    expect(parseViewState('{"version":1,"layers":{"c0":{"gamma":2}}}')).toEqual({
      version: 1,
      layers: { c0: { gamma: 2 } },
    });
    expect(parseViewState(JSON.stringify({ version: 1, camera: CAMERA }))).toEqual({
      version: 1,
      camera: CAMERA,
    });
  });

  it.each([
    ['not json', 'not valid JSON'],
    ['[]', 'not an object'],
    ['{"version":2}', 'version 2'],
    ['{"version":1,"layers":3}', '"layers"'],
    ['{"version":1,"layers":{"c0":3}}', "'c0'"],
    ['{"version":1,"layers":{"c0":{"gamma":"1"}}}', "'c0'.gamma"],
    ['{"version":1,"layers":{"c0":{"displayRange":[1]}}}', 'displayRange'],
    ['{"version":1,"layers":{"c0":{"blendingMode":"foo"}}}', 'blendingMode'],
    ['{"version":1,"layers":{"c0":{"colormap":3}}}', 'colormap'],
    ['{"version":1,"layers":{"c0":{"bogus":1}}}', "'c0'.bogus"],
    ['{"version":1,"camera":[]}', '"camera"'],
    ['{"version":1,"camera":{"position":[1,2],"target":[0,0,0],"up":[0,1,0],"isOrtho":false,"near":1,"far":2}}', 'camera.position'],
    ['{"version":1,"camera":{"position":[1,2,3],"target":[0,0,0],"up":[0,1,0],"isOrtho":"no","near":1,"far":2}}', 'camera.isOrtho'],
    ['{"version":1,"camera":{"position":[1,2,3],"target":[0,0,0],"up":[0,1,0],"isOrtho":false,"near":1}}', 'camera.far'],
    ['{"version":1,"camera":{"position":[1,2,3],"target":[0,0,0],"up":[0,1,0],"isOrtho":false,"near":1,"far":2,"fov":"x"}}', 'camera.fov'],
  ])('rejects %s', (text, fragment) => {
    expect(() => parseViewState(text)).toThrow(fragment);
  });
});

describe('hash codec', () => {
  it('writes the whole document after #! and reads it back', () => {
    const hash = writeViewStateHash(DOC);
    expect(hash.startsWith('#!')).toBe(true);
    expect(hash).not.toContain('"'); // URL-encoded
    expect(readViewStateHash(hash)).toEqual(DOC);
  });

  it('writes nothing for an empty document', () => {
    expect(isEmptyViewState({ version: 1 })).toBe(true);
    expect(isEmptyViewState({ version: 1, layers: {} })).toBe(true);
    expect(isEmptyViewState({ version: 1, camera: CAMERA })).toBe(false);
    expect(writeViewStateHash({ version: 1, layers: {} })).toBe('');
  });

  it('returns null for an absent, foreign or malformed fragment', () => {
    expect(readViewStateHash('')).toBeNull();
    expect(readViewStateHash('#foo=1')).toBeNull();
    expect(readViewStateHash('#layers=%7B%22version%22%3A1%7D')).toBeNull(); // the pre-#! spelling
    expect(readViewStateHash('#!%7Bnope')).toBeNull();
    expect(readViewStateHash('#!%7B%22version%22%3A9%7D')).toBeNull();
  });

  it('reads an unencoded fragment too (a hand-typed link)', () => {
    expect(readViewStateHash('#!{"version":1,"layers":{"c0":{"gamma":2}}}')).toEqual({
      version: 1,
      layers: { c0: { gamma: 2 } },
    });
  });
});

describe('clearViewStateHash', () => {
  it('drops the fragment and does not touch an already-clean URL', () => {
    const win = {
      location: { pathname: '/v/', search: '?src=a', hash: '#!%7B%22version%22%3A1%7D' },
      history: { replaceState: vi.fn() },
    };
    clearViewStateHash(win);
    expect(win.history.replaceState).toHaveBeenCalledWith(null, '', '/v/?src=a');
    win.location.hash = '';
    clearViewStateHash(win);
    expect(win.history.replaceState).toHaveBeenCalledOnce();
  });
});

describe('startViewStateUrlSync', () => {
  it('writes the document into the fragment once, after the changes settle', () => {
    vi.useFakeTimers();
    try {
      const listeners = new Set<() => void>();
      let current: ViewStateDoc = { version: 1 };
      const win = {
        location: { pathname: '/v/', search: '?src=a', hash: '' },
        history: {
          replaceState: vi.fn((_s: unknown, _t: string, url: string) => {
            const i = url.indexOf('#');
            win.location.hash = i < 0 ? '' : url.slice(i);
          }),
        },
      };
      const stop = startViewStateUrlSync(
        {
          onChange: (l) => {
            listeners.add(l);
            return () => listeners.delete(l);
          },
          getViewState: () => current,
        },
        win,
        300
      );
      current = { version: 1, camera: { ...CAMERA, position: [1, 1, 1] } };
      listeners.forEach((l) => l());
      current = { version: 1, camera: CAMERA, layers: { c0: { gamma: 1.4 } } };
      listeners.forEach((l) => l());
      vi.advanceTimersByTime(299);
      expect(win.history.replaceState).not.toHaveBeenCalled();
      vi.advanceTimersByTime(1);
      expect(win.history.replaceState).toHaveBeenCalledOnce();
      expect(win.history.replaceState.mock.calls[0][2]).toMatch(/^\/v\/\?src=a#!/);
      expect(readViewStateHash(win.location.hash)).toEqual(current);

      // A change that leaves the document identical does not rewrite the URL.
      listeners.forEach((l) => l());
      vi.advanceTimersByTime(300);
      expect(win.history.replaceState).toHaveBeenCalledOnce();

      stop();
      current = { version: 1 };
      listeners.forEach((l) => l());
      vi.advanceTimersByTime(300);
      expect(win.history.replaceState).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });
});
