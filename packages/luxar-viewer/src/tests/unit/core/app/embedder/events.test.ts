/**
 * Unit tests for the per-app embedder event bus — the mechanics behind
 * `LuxarApp.on(...)`. Drives `createEventBus<LuxarEmbedderEventMap>()`
 * directly (the same factory LuxarApp uses), so no app/DOM is needed.
 *
 * @see src/core/app/embedder/events.ts
 * @see src/utils/cross-layer/event-bus.ts
 */
import { describe, it, expect, vi } from 'vitest';
import { createEventBus } from '../../../../../utils/cross-layer/event-bus';
import type { LuxarEmbedderEventMap } from '../../../../../core/app/embedder/events';

describe('embedder event bus', () => {
  it('delivers emitted payloads to subscribers', () => {
    const bus = createEventBus<LuxarEmbedderEventMap>();
    const seen: string[] = [];
    bus.on('dataset-loaded', ({ src }) => seen.push(src));

    bus.emit('dataset-loaded', { src: 'a.zarr' });
    bus.emit('dataset-loaded', { src: 'b.zarr' });

    expect(seen).toEqual(['a.zarr', 'b.zarr']);
  });

  it('on() returns an unsubscribe that stops further delivery', () => {
    const bus = createEventBus<LuxarEmbedderEventMap>();
    const fn = vi.fn();
    const off = bus.on('dataset-error', fn);

    bus.emit('dataset-error', { src: 'x', error: new Error('boom') });
    off();
    bus.emit('dataset-error', { src: 'x', error: new Error('again') });

    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('only notifies listeners for the emitted event type', () => {
    const bus = createEventBus<LuxarEmbedderEventMap>();
    const onDims = vi.fn();
    const onSel = vi.fn();
    bus.on('dimensions-changed', onDims);
    bus.on('selection', onSel);

    bus.emit('selection', { nodeName: 'pts', elementIndex: 3 });

    expect(onSel).toHaveBeenCalledWith({ nodeName: 'pts', elementIndex: 3 });
    expect(onDims).not.toHaveBeenCalled();
  });

  it('carries a null selection payload', () => {
    const bus = createEventBus<LuxarEmbedderEventMap>();
    const fn = vi.fn();
    bus.on('selection', fn);

    bus.emit('selection', null);

    expect(fn).toHaveBeenCalledWith(null);
  });
});
