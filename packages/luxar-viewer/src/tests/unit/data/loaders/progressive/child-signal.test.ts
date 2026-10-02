import { describe, expect, it } from 'vitest';

import {
  signalOrigin,
  tagSignalOrigin,
} from '../../../../../cache/decompressed-chunk-cache/decode-origin';
import { createChildController } from '../../../../../data/loaders/progressive/child-signal';
import { signalPriority, tagSignalPriority } from '../../../../../utils/fetch-concurrency';

describe('createChildController', () => {
  it('aborts with its parent, but can also be aborted alone', () => {
    const parent = new AbortController();
    const a = createChildController(parent.signal);
    const b = createChildController(parent.signal);
    a.controller.abort();
    expect(a.controller.signal.aborted).toBe(true);
    expect(parent.signal.aborted).toBe(false);
    expect(b.controller.signal.aborted).toBe(false);
    parent.abort();
    expect(b.controller.signal.aborted).toBe(true);
  });

  it('is born aborted under an aborted parent', () => {
    const parent = new AbortController();
    parent.abort();
    expect(createChildController(parent.signal).controller.signal.aborted).toBe(true);
  });

  it('stops following the parent once detached', () => {
    const parent = new AbortController();
    const child = createChildController(parent.signal);
    child.detach();
    parent.abort();
    expect(child.controller.signal.aborted).toBe(false);
  });

  it('inherits the parent priority cell and decode origin', () => {
    const parent = new AbortController();
    tagSignalPriority(parent.signal, 'refinement');
    tagSignalOrigin(parent.signal, 'lookahead');
    const child = createChildController(parent.signal).controller.signal;
    // The SAME cell, so a later raise on the parent reaches the child.
    expect(signalPriority(child)).toBe(signalPriority(parent.signal));
    expect(signalOrigin(child)).toBe('lookahead');
  });

  it('is a plain controller without a parent', () => {
    const child = createChildController(undefined).controller.signal;
    expect(child.aborted).toBe(false);
    expect(signalPriority(child)).toBeUndefined();
  });
});
