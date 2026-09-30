// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { buildBookmarksPopover } from '../../../../ui/rail-panels/bookmarks-popover';
import { buildBookmarkUrl, type ViewBookmark } from '../../../../core/app/bookmark-state';

const state = {
  version: 1,
  src: '/sample.zarr',
  snapshot: {
    version: 1,
    camera: {
      position: [1, 2, 3],
      target: [0, 0, 0],
      up: [0, 1, 0],
      isOrtho: false,
      fov: 45,
      near: 0.1,
      far: 100,
    },
  },
  rendering: {},
  layers: [],
} as unknown as ViewBookmark;

const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

describe('Bookmarks panel', () => {
  it('adds a named link, copies it, reopens the view, and clears the list', async () => {
    const host = document.createElement('div');
    const list: Array<{ name: string; url: string; state: ViewBookmark }> = [];
    const copy = vi.fn().mockResolvedValue(undefined);
    const restore = vi.fn().mockResolvedValue(undefined);
    const ctx = {
      capture: () => state,
      restore,
      baseUrl: () => 'https://example.org/viewer',
      buildUrl: buildBookmarkUrl,
      copy,
    };
    buildBookmarksPopover(host, ctx, list);
    const input = host.querySelector('input')!;
    input.value = 'Cells at T7';
    host
      .querySelector('form')!
      .dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await flush();
    expect(list).toHaveLength(1);
    expect(list[0].name).toBe('Cells at T7');
    expect(copy).toHaveBeenCalledWith(list[0].url);
    expect(host.querySelector('[role="status"]')?.textContent).toBe('Link copied');
    host.querySelector<HTMLButtonElement>('li button')!.click();
    await flush();
    expect(restore).toHaveBeenCalledWith(state);
    expect(host.querySelector('[role="status"]')?.textContent).toBe('Opened Cells at T7');
    buildBookmarksPopover(document.createElement('div'), ctx, list);
    expect(list).toHaveLength(1);
    Array.from(host.querySelectorAll('button'))
      .find((b) => b.textContent === 'Clear')!
      .click();
    expect(list).toHaveLength(0);
    expect(host.querySelectorAll('li')).toHaveLength(0);
  });
});
