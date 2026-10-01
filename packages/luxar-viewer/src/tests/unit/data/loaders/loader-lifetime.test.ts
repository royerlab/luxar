import { describe, expect, it, vi } from 'vitest';

import { LoaderLifetime } from '../../../../data/loaders/loader-lifetime';

describe('LoaderLifetime', () => {
  it('latches disposal and aborts its signal, idempotently', () => {
    const lifetime = new LoaderLifetime();
    expect(lifetime.disposed).toBe(false);
    lifetime.dispose();
    lifetime.dispose();
    expect(lifetime.disposed).toBe(true);
    expect(lifetime.signal.aborted).toBe(true);
    expect(() => lifetime.throwIfDisposed('x')).toThrow(
      expect.objectContaining({ name: 'AbortError' })
    );
  });

  it('guardInit refuses to start on a disposed loader', async () => {
    const lifetime = new LoaderLifetime();
    lifetime.dispose();
    const init = vi.fn().mockResolvedValue(undefined);
    await expect(lifetime.guardInit('x', init, vi.fn())).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(init).not.toHaveBeenCalled();
  });

  it('guardInit discards an initialization that disposal overtook', async () => {
    const lifetime = new LoaderLifetime();
    const discard = vi.fn();
    let finish!: () => void;
    const pending = lifetime.guardInit(
      'x',
      () => new Promise<void>((resolve) => (finish = resolve)),
      discard
    );
    lifetime.dispose();
    finish();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(discard).toHaveBeenCalledTimes(1);
  });

  it('guardInit reports a disposal-caused failure as the cancellation', async () => {
    const lifetime = new LoaderLifetime();
    const pending = lifetime.guardInit(
      'x',
      () =>
        new Promise<void>((_, reject) =>
          lifetime.signal.addEventListener('abort', () => reject(new Error('read aborted')))
        ),
      vi.fn()
    );
    lifetime.dispose();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('guardInit passes a genuine failure through untouched', async () => {
    const lifetime = new LoaderLifetime();
    const boom = new Error('boom');
    await expect(lifetime.guardInit('x', () => Promise.reject(boom), vi.fn())).rejects.toBe(boom);
  });
});
