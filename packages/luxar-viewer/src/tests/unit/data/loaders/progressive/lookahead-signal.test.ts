import { describe, expect, it } from 'vitest';

import { signalOrigin } from '../../../../../cache/decompressed-chunk-cache/decode-origin';
import { createLookaheadController } from '../../../../../data/loaders/progressive/lookahead-signal';
import { signalPriority } from '../../../../../utils/fetch-concurrency';

describe('createLookaheadController', () => {
  it('tags the lookahead SPECULATIVE, so it queues behind frame-blocking reads', () => {
    // Untagged, the MLC resolves the decode to `demand` and the warm-up bypasses
    // the speculative cap while competing with the reads a frame waits on.
    const { signal } = createLookaheadController();
    expect(signalPriority(signal)?.value).toBe('speculative');
    expect(signalOrigin(signal)).toBe('lookahead');
  });

  it('aborts with the update that scheduled it', () => {
    const update = new AbortController();
    const lookahead = createLookaheadController(update.signal);
    update.abort();
    expect(lookahead.signal.aborted).toBe(true);
  });
});
