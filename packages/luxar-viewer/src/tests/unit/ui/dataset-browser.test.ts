// @vitest-environment jsdom
/**
 * Unit tests for ui/dataset-browser.ts.
 *
 * Mocks DirectoryNavigator (the only external dependency that touches
 * the network). The panel construction + entry rendering run for real
 * under jsdom — we only stub the async directory listings.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DirectoryNavigator, type DirectoryEntry } from '../../../data';
import { DatasetBrowser } from '../../../ui/dataset-browser';
import { isTypingInInput } from '../../../utils/dom/focus';
import { log } from '../../../utils/log';

const navigateMock = vi.fn();
const getFullUrlMock = vi.fn();

vi.mock('../../../data', async () => {
  const actual = await vi.importActual<typeof import('../../../data')>('../../../data');
  return {
    ...actual,
    DirectoryNavigator: vi.fn().mockImplementation(() => ({
      navigate: navigateMock,
      getFullUrl: getFullUrlMock,
    })),
  };
});

function makeContainer(): HTMLElement {
  const el = document.createElement('div');
  document.body.appendChild(el);
  return el;
}

function defaultNavigateResult(
  overrides: {
    entries?: DirectoryEntry[];
    currentPath?: string;
    parentPath?: string;
    strategy?: 'webdav' | 'html' | 'index' | 'manual';
    isZarr?: boolean;
  } = {}
) {
  return {
    entries: overrides.entries ?? [],
    currentPath: overrides.currentPath ?? '',
    parentPath: overrides.parentPath,
    strategy: overrides.strategy ?? 'webdav',
    isZarr: overrides.isZarr ?? false,
  };
}

describe('DatasetBrowser', () => {
  let container: HTMLElement;
  let onDatasetSelect: ReturnType<typeof vi.fn> & ((fullUrl: string) => void);
  let onClose: ReturnType<typeof vi.fn> & (() => void);

  beforeEach(() => {
    vi.clearAllMocks();
    document.body.innerHTML = '';
    container = makeContainer();
    onDatasetSelect = vi.fn() as ReturnType<typeof vi.fn> & ((fullUrl: string) => void);
    onClose = vi.fn() as ReturnType<typeof vi.fn> & (() => void);

    navigateMock.mockResolvedValue(defaultNavigateResult());
    getFullUrlMock.mockImplementation((path: string) => `http://example.com/${path}`);
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  describe('construction', () => {
    it('creates the panel skeleton inside the container', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });

      const panel = container.querySelector('#luxar-dataset-browser');
      expect(panel).not.toBeNull();
      expect(panel?.getAttribute('role')).toBe('dialog');
      expect(panel?.getAttribute('aria-modal')).toBe('true');

      // Skeleton sub-elements created on construction.
      expect(panel?.querySelector('#luxar-dataset-browser-title')?.textContent).toBe(
        'Select Dataset'
      );
      expect(panel?.querySelector('#luxar-dataset-browser-welcome')).not.toBeNull();
      expect(panel?.querySelector('#luxar-dataset-browser-breadcrumb')).not.toBeNull();
      expect(panel?.querySelector('#luxar-dataset-browser-content')).not.toBeNull();
      expect(panel?.querySelector('#luxar-dataset-browser-status')).not.toBeNull();
    });

    it('shows a close button with aria-label and a stroke SVG glyph', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });

      const closeBtn = container.querySelector(
        '.luxar-dataset-browser__close-btn'
      ) as HTMLButtonElement;
      expect(closeBtn).not.toBeNull();
      expect(closeBtn.getAttribute('aria-label')).toBe('Close dataset browser');
      expect(closeBtn.querySelector('svg')).not.toBeNull();
    });

    it('mounts a scrim behind the panel that closes the browser on click', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });

      const scrim = container.querySelector('.luxar-dataset-browser-scrim') as HTMLElement;
      expect(scrim).not.toBeNull();
      expect(scrim.getAttribute('aria-hidden')).toBe('true');

      scrim.click();
      expect(onClose).toHaveBeenCalled();
      expect(container.querySelector('#luxar-dataset-browser')).toBeNull();
      expect(container.querySelector('.luxar-dataset-browser-scrim')).toBeNull();
    });

    it('navigates to root exactly once with empty path when no currentSrc is provided', () => {
      // [ui.md/W1][P2] Previously the two tests asserted the same thing —
      // toHaveBeenCalledWith('') — which gave no signal about the two
      // distinct constructor branches. Strengthen: pin the call count
      // (exactly one navigate during construction) and that the initial
      // path is the empty-root string, not undefined or null.
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      expect(navigateMock).toHaveBeenCalledTimes(1);
      expect(navigateMock).toHaveBeenCalledWith('');
      // First argument is strictly the empty string '', not undefined / null
      // (which could happen if a refactor silently dropped the fallback).
      expect(navigateMock.mock.calls[0][0]).toBe('');
    });

    it('extracts parent directory when currentSrc points inside a .zarr and renders the panel skeleton', () => {
      // [ui.md/W1][P2] Previously asserted only that navigate('') fired,
      // which is identical to the no-currentSrc case. Strengthen by
      // verifying the panel skeleton is still constructed (i.e. the
      // currentSrc branch does not throw or skip DOM construction) and
      // that the initial navigate fires exactly once.
      new DatasetBrowser({
        container,
        onDatasetSelect,
        onClose,
        currentSrc: 'http://server.test/data/sample.zarr',
        origin: 'http://server.test',
      });

      // The constructor should have wired the DOM:
      expect(container.querySelector('#luxar-dataset-browser')).not.toBeNull();
      expect(container.querySelector('#luxar-dataset-browser-title')?.textContent).toBe(
        'Select Dataset'
      );
      // And fired exactly one initial navigate (no double-fetch).
      expect(navigateMock).toHaveBeenCalledTimes(1);
      expect(navigateMock).toHaveBeenCalledWith('');
    });

    it('opens a loaded .zarr.zip at its parent and marks the archive as loaded', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({
          entries: [
            {
              name: 'sample.ZARR.ZIP',
              path: 'sample.ZARR.ZIP',
              type: 'zarr',
            },
          ],
        })
      );

      new DatasetBrowser({
        container,
        onDatasetSelect,
        onClose,
        currentSrc: 'http://server.test/data/sample.ZARR.ZIP',
        origin: 'http://server.test',
      });

      expect(vi.mocked(DirectoryNavigator)).toHaveBeenCalledWith('http://server.test/data/');
      await vi.waitFor(() => {
        const current = container.querySelector(
          '.luxar-dataset-browser__file-item--current'
        ) as HTMLElement | null;
        expect(current?.textContent).toContain('sample.ZARR.ZIP');
        expect(current?.textContent).toContain('LOADED');
      });
    });
  });

  it('logs a terminal navigation failure from a directory activation', async () => {
    navigateMock.mockResolvedValueOnce(
      defaultNavigateResult({
        entries: [{ name: 'nested', path: 'nested', type: 'directory' }],
      })
    );
    const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
    await vi.waitFor(() => {
      expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
    });

    const failure = new Error('render path failed');
    const internals = browser as unknown as { navigate(path: string): Promise<void> };
    vi.spyOn(internals, 'navigate').mockRejectedValue(failure);
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});

    (container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement).click();

    await vi.waitFor(() => {
      expect(warning).toHaveBeenCalledWith('UI', 'Dataset navigation failed', failure);
    });
  });

  describe('navigation + entries', () => {
    it('shows directory entries, sorted zarr → dir → file', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'b_dir', path: 'b_dir', type: 'directory' },
        { name: 'c_file.txt', path: 'c_file.txt', type: 'file' },
        { name: 'a.zarr', path: 'a.zarr', type: 'zarr' },
        { name: 'd.zarr', path: 'd.zarr', type: 'zarr' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries, strategy: 'webdav' }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        const items = container.querySelectorAll('.luxar-dataset-browser__file-item');
        expect(items.length).toBe(4);
      });

      const items = container.querySelectorAll('.luxar-dataset-browser__file-item');
      const orderedNames = Array.from(items).map(
        (el) => (el.querySelector('.luxar-dataset-browser__file-name') as HTMLElement).textContent
      );
      // zarr first (alphabetical), then directories, then files.
      expect(orderedNames).toEqual(['a.zarr', 'd.zarr', 'b_dir', 'c_file.txt']);
    });

    it('shows the empty-directory placeholder when entries is empty', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });
      expect(container.querySelector('.luxar-dataset-browser__empty')?.textContent).toContain(
        'No datasets here'
      );
      // The empty state offers manual path entry as a next step.
      expect(container.querySelector('.luxar-dataset-browser__empty-action')).not.toBeNull();
    });

    it('shows manual-entry form when strategy=manual + 0 entries', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__manual-entry')).not.toBeNull();
      });
      expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      expect(container.querySelector('#luxar-dataset-browser-manual-load')).not.toBeNull();
    });

    it('filters entries live via the search bar (case-insensitive substring)', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'alpha.zarr', path: 'alpha.zarr', type: 'zarr' },
        { name: 'beta.zarr', path: 'beta.zarr', type: 'zarr' },
        { name: 'gamma_dir', path: 'gamma_dir', type: 'directory' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries, strategy: 'html' }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
      });

      const search = container.querySelector('#luxar-dataset-browser-search') as HTMLInputElement;
      expect(search).not.toBeNull();

      // Search bar is visible when there is a listing to filter.
      const searchBar = container.querySelector('#luxar-dataset-browser-search-bar') as HTMLElement;
      expect(searchBar.style.display).not.toBe('none');

      // Typing narrows the list (case-insensitive).
      search.value = 'BETA';
      search.dispatchEvent(new Event('input'));
      const filtered = container.querySelectorAll('.luxar-dataset-browser__file-name');
      expect(Array.from(filtered).map((el) => el.textContent)).toEqual(['beta.zarr']);

      // Status bar reflects "M of N" while filtering.
      expect(container.querySelector('#luxar-dataset-browser-status')?.textContent).toContain(
        '1 of 3 items'
      );

      // No matches → placeholder, full count restored on clear.
      search.value = 'zzz';
      search.dispatchEvent(new Event('input'));
      expect(container.querySelector('.luxar-dataset-browser__empty')?.textContent).toContain(
        'No matches'
      );

      search.value = '';
      search.dispatchEvent(new Event('input'));
      expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
    });

    it('hides the search bar when there are no entries to filter', async () => {
      // Empty directory.
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries: [] }));
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });
      const searchBar = container.querySelector('#luxar-dataset-browser-search-bar') as HTMLElement;
      expect(searchBar.style.display).toBe('none');
    });

    it('hides the search bar in the manual-entry fallback', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__manual-entry')).not.toBeNull();
      });
      const searchBar = container.querySelector('#luxar-dataset-browser-search-bar') as HTMLElement;
      expect(searchBar.style.display).toBe('none');
    });

    it('updates breadcrumb with path segments', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: 'data/sub', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        const links = container.querySelectorAll(
          '.luxar-dataset-browser__breadcrumb-link, .luxar-dataset-browser__breadcrumb-current'
        );
        expect(links.length).toBeGreaterThan(0);
      });

      const breadcrumb = container.querySelector(
        '#luxar-dataset-browser-breadcrumb'
      ) as HTMLElement;
      // Root + 'data' + 'sub' (last is non-clickable current).
      expect(breadcrumb.textContent).toContain('Root');
      expect(breadcrumb.textContent).toContain('data');
      expect(breadcrumb.textContent).toContain('sub');
    });

    it('renders the error block when navigate rejects', async () => {
      navigateMock.mockRejectedValueOnce(new Error('Network down'));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__error')).not.toBeNull();
      });
      expect(
        container.querySelector('.luxar-dataset-browser__error-details')?.textContent
      ).toContain('Network down');
    });

    it('contains a retry rejection when error rendering also throws', async () => {
      navigateMock.mockRejectedValueOnce(new Error('Network down'));
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__error-retry')).not.toBeNull();
      });

      const failure = Object.create(null) as object;
      navigateMock.mockRejectedValueOnce(failure);
      const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
      (container.querySelector('.luxar-dataset-browser__error-retry') as HTMLButtonElement).click();

      await vi.waitFor(() => {
        expect(warning).toHaveBeenCalledWith(
          'UI',
          'Dataset navigation failed',
          expect.objectContaining({ message: 'Cannot convert object to primitive value' })
        );
      });
    });

    it.each([
      ['root', 0, ''],
      ['parent segment', 1, 'data'],
    ])('contains a rejection from the %s breadcrumb', async (_label, index, expectedPath) => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: 'data/sub', entries: [] })
      );
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__breadcrumb-link').length).toBe(
          2
        );
      });

      const failure = Object.create(null) as object;
      navigateMock.mockRejectedValueOnce(failure);
      const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
      const links = container.querySelectorAll<HTMLButtonElement>(
        '.luxar-dataset-browser__breadcrumb-link'
      );
      links[index].click();

      await vi.waitFor(() => {
        expect(navigateMock).toHaveBeenLastCalledWith(expectedPath);
        expect(warning).toHaveBeenCalledWith(
          'UI',
          'Dataset navigation failed',
          expect.objectContaining({ message: 'Cannot convert object to primitive value' })
        );
      });
    });
  });

  describe('inline path editor', () => {
    it('the breadcrumb edit toggle swaps in a path input seeded with the current path', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: 'data/sub', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();

      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      expect(input).not.toBeNull();
      expect(input.value).toBe('data/sub');
    });

    it('Escape from the path editor returns focus to the panel', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: 'data/sub', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      expect(document.activeElement).toBe(input);

      input.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })
      );

      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      expect(container.querySelector('.luxar-dataset-browser__path-input')).toBeNull();
      expect(document.activeElement).toBe(panel);
    });

    it('Enter on a .zarr path selects it via getFullUrl and closes', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      input.value = 'sub/sample.zarr';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(getFullUrlMock).toHaveBeenCalledWith('sub/sample.zarr');
      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sub/sample.zarr');
      expect(onClose).toHaveBeenCalled();
    });

    it('keeps the browser open when a path-editor selection is refused', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ currentPath: 'data' }));
      onDatasetSelect.mockReturnValue(false);

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      input.value = 'sub/sample.zarr';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sub/sample.zarr');
      expect(container.querySelector('#luxar-dataset-browser')).not.toBeNull();
      expect(onClose).not.toHaveBeenCalled();
    });

    it('Enter on a plain directory path navigates instead of selecting', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      navigateMock.mockClear();
      navigateMock.mockResolvedValue(defaultNavigateResult({ currentPath: 'some/dir' }));

      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      input.value = 'some/dir';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(navigateMock).toHaveBeenCalledWith('some/dir');
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('does NOT select paths where .zarr is a mere substring (navigates instead)', async () => {
      // `.zarr` must terminate the last path segment. `archives.zarr-backup` and
      // `my.zarrs/dir` are ordinary names. (`foo.zarr.zip` USED to be listed
      // here; a zipped store is now a dataset the viewer reads in place — see
      // the zipped-store case below.)
      for (const value of ['archives.zarr-backup', 'my.zarrs/dir']) {
        vi.clearAllMocks();
        document.body.innerHTML = '';
        container = makeContainer();
        navigateMock.mockResolvedValueOnce(defaultNavigateResult());
        getFullUrlMock.mockImplementation((path: string) => `http://example.com/${path}`);

        new DatasetBrowser({ container, onDatasetSelect, onClose });
        await vi.waitFor(() => {
          expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
        });

        (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
        navigateMock.mockClear();
        navigateMock.mockResolvedValue(defaultNavigateResult({ currentPath: value }));

        const input = container.querySelector(
          '.luxar-dataset-browser__path-input'
        ) as HTMLInputElement;
        input.value = value;
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

        expect(onDatasetSelect).not.toHaveBeenCalled();
        expect(navigateMock).toHaveBeenCalledWith(value);
      }
    });

    it('selects a .zarr path with a trailing slash', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      input.value = 'sub/sample.zarr/';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sub/sample.zarr/');
      expect(onClose).toHaveBeenCalled();
    });

    it('selects a zipped store (.zarr.zip) rather than navigating into it', async () => {
      // An archive IS a dataset — the viewer reads it in place over range
      // requests — and it has no listable children to navigate into.
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      navigateMock.mockClear();
      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      input.value = 'sub/scene.luxar.zarr.zip';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sub/scene.luxar.zarr.zip');
      expect(navigateMock).not.toHaveBeenCalled();
      expect(onClose).toHaveBeenCalled();
    });

    it('Escape restores the breadcrumb trail without closing the dialog', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: 'data', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-path-edit')).not.toBeNull();
      });

      (container.querySelector('#luxar-dataset-browser-path-edit') as HTMLButtonElement).click();
      const input = container.querySelector(
        '.luxar-dataset-browser__path-input'
      ) as HTMLInputElement;
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

      expect(container.querySelector('.luxar-dataset-browser__path-input')).toBeNull();
      expect(container.querySelector('.luxar-dataset-browser__breadcrumb-link')).not.toBeNull();
      expect(onClose).not.toHaveBeenCalled();
    });
  });

  describe('arrow-key list navigation', () => {
    it('ArrowDown / ArrowUp move focus between rows', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'a.zarr', path: 'a.zarr', type: 'zarr' },
        { name: 'b.zarr', path: 'b.zarr', type: 'zarr' },
        { name: 'c_dir', path: 'c_dir', type: 'directory' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
      });

      const items = Array.from(
        container.querySelectorAll<HTMLElement>('.luxar-dataset-browser__file-item')
      );
      items[0].focus();
      items[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
      expect(document.activeElement).toBe(items[1]);

      items[1].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
      expect(document.activeElement).toBe(items[0]);

      items[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
      expect(document.activeElement).toBe(items[2]);

      items[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
      expect(document.activeElement).toBe(items[0]);
    });

    it('prevents the default action at list boundaries (no scroll bleed)', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'a.zarr', path: 'a.zarr', type: 'zarr' },
        { name: 'b.zarr', path: 'b.zarr', type: 'zarr' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(2);
      });

      const items = Array.from(
        container.querySelectorAll<HTMLElement>('.luxar-dataset-browser__file-item')
      );
      items[0].focus();
      // ArrowUp on the FIRST row: focus stays, but the event must still be
      // swallowed so it doesn't scroll the content area.
      const up = new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true, cancelable: true });
      items[0].dispatchEvent(up);
      expect(up.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(items[0]);

      const down = new KeyboardEvent('keydown', {
        key: 'ArrowDown',
        bubbles: true,
        cancelable: true,
      });
      items[1].focus();
      items[1].dispatchEvent(down);
      expect(down.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(items[1]);
    });

    it('the async first listing does not steal focus the user already placed', async () => {
      // Slow listing: the user focuses the close button while it loads. The
      // arriving listing must not yank focus away (it no longer autofocuses
      // the filter at all — see the type-to-filter suite below, #1922).
      let resolveNavigate: (v: unknown) => void = () => {};
      navigateMock.mockReturnValueOnce(new Promise((res) => (resolveNavigate = res)));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      const closeBtn = container.querySelector(
        '.luxar-dataset-browser__close-btn'
      ) as HTMLButtonElement;
      closeBtn.focus();

      resolveNavigate(
        defaultNavigateResult({
          entries: [{ name: 'a.zarr', path: 'a.zarr', type: 'zarr' }],
        })
      );
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      expect(document.activeElement).toBe(closeBtn);
    });
  });

  describe('selection callbacks', () => {
    it('auto-selects when navigate returns a zarr path', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({
          currentPath: 'data/sample.zarr',
          isZarr: true,
        })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });

      await vi.waitFor(() => {
        expect(onDatasetSelect).toHaveBeenCalled();
      });
      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/data/sample.zarr');
      expect(onClose).toHaveBeenCalled();
    });

    it('does NOT auto-select when isZarr=true but currentPath is empty', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ currentPath: '', isZarr: true }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      // Wait one microtask cycle for the navigate promise to resolve.
      await vi.waitFor(() => {
        // The status bar update means navigate completed.
        expect(navigateMock).toHaveBeenCalled();
      });
      // No auto-select fired.
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('selects a zarr entry on click', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'sample.zarr', path: 'sample.zarr', type: 'zarr' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      const item = container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement;
      item.click();

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sample.zarr');
      expect(onClose).toHaveBeenCalled();
    });

    it('keeps the browser open when selection is synchronously refused', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'sample.zarr', path: 'sample.zarr', type: 'zarr' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));
      onDatasetSelect.mockReturnValue(false);

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      const item = container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement;
      item.click();

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sample.zarr');
      expect(container.querySelector('#luxar-dataset-browser')).not.toBeNull();
      expect(onClose).not.toHaveBeenCalled();
    });

    it('navigates into a directory entry on click', async () => {
      const entries: DirectoryEntry[] = [{ name: 'sub', path: 'sub', type: 'directory' }];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      // Reset the mock so we can observe the second call.
      navigateMock.mockClear();
      navigateMock.mockResolvedValue(defaultNavigateResult({ currentPath: 'sub' }));

      const item = container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement;
      item.click();

      expect(navigateMock).toHaveBeenCalledWith('sub');
      // Selection callbacks must NOT fire for directories.
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('restores the parent listing when an auto-detected zarr selection is refused', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'archive.zarr-backup', path: 'archive.zarr-backup', type: 'directory' },
      ];
      navigateMock
        .mockResolvedValue(defaultNavigateResult({ entries }))
        .mockResolvedValueOnce(defaultNavigateResult({ entries }))
        .mockResolvedValueOnce(
          defaultNavigateResult({
            currentPath: 'archive.zarr-backup',
            parentPath: '',
            isZarr: true,
          })
        );
      onDatasetSelect.mockReturnValue(false);

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      (container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement).click();

      await vi.waitFor(() => expect(navigateMock).toHaveBeenCalledTimes(3));
      expect(navigateMock).toHaveBeenLastCalledWith('');
      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/archive.zarr-backup');
      expect(container.querySelector('.luxar-dataset-browser__loading')).toBeNull();
      expect(container.querySelectorAll('.luxar-dataset-browser__file-item')).toHaveLength(1);
      expect(container.querySelector('#luxar-dataset-browser')).not.toBeNull();
      expect(onClose).not.toHaveBeenCalled();
    });

    it('manual-entry submission selects the typed path via getFullUrl', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      });

      const input = container.querySelector(
        '#luxar-dataset-browser-manual-path'
      ) as HTMLInputElement;
      const loadBtn = container.querySelector(
        '#luxar-dataset-browser-manual-load'
      ) as HTMLButtonElement;

      input.value = 'mydata.zarr';
      loadBtn.click();

      expect(getFullUrlMock).toHaveBeenCalledWith('mydata.zarr');
      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/mydata.zarr');
      expect(onClose).toHaveBeenCalled();
    });

    it('manual-entry passes through full URLs unchanged (no getFullUrl prefix)', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      });

      const input = container.querySelector(
        '#luxar-dataset-browser-manual-path'
      ) as HTMLInputElement;
      const loadBtn = container.querySelector(
        '#luxar-dataset-browser-manual-load'
      ) as HTMLButtonElement;

      input.value = 'https://other.example/dataset.zarr';
      loadBtn.click();

      expect(getFullUrlMock).not.toHaveBeenCalled();
      expect(onDatasetSelect).toHaveBeenCalledWith('https://other.example/dataset.zarr');
    });

    it('keeps the browser open when a manual-entry selection is refused', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );
      onDatasetSelect.mockReturnValue(false);

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      });

      const input = container.querySelector(
        '#luxar-dataset-browser-manual-path'
      ) as HTMLInputElement;
      input.value = 'mydata.zarr';
      (container.querySelector('#luxar-dataset-browser-manual-load') as HTMLButtonElement).click();

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/mydata.zarr');
      expect(container.querySelector('#luxar-dataset-browser')).not.toBeNull();
      expect(onClose).not.toHaveBeenCalled();
    });

    it('manual-entry ignores empty input', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      });

      const input = container.querySelector(
        '#luxar-dataset-browser-manual-path'
      ) as HTMLInputElement;
      const loadBtn = container.querySelector(
        '#luxar-dataset-browser-manual-load'
      ) as HTMLButtonElement;

      input.value = '   '; // whitespace only — `.trim()` empties it.
      loadBtn.click();

      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('manual-entry triggers load on Enter key', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      });

      const input = container.querySelector(
        '#luxar-dataset-browser-manual-path'
      ) as HTMLInputElement;
      input.value = 'foo.zarr';

      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/foo.zarr');
    });
  });

  describe('accessibility', () => {
    it('renders entry rows with role="button", tabIndex=0, and an aria-label', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'sample.zarr', path: 'sample.zarr', type: 'zarr' },
        { name: 'sub', path: 'sub', type: 'directory' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        const items = container.querySelectorAll('.luxar-dataset-browser__file-item');
        expect(items.length).toBe(2);
      });

      const items = Array.from(
        container.querySelectorAll<HTMLElement>('.luxar-dataset-browser__file-item')
      );
      for (const item of items) {
        expect(item.getAttribute('role')).toBe('button');
        expect(item.tabIndex).toBe(0);
        expect(item.getAttribute('aria-label')).toBeTruthy();
      }
      expect(items[0].getAttribute('aria-label')).toContain('Zarr');
      expect(items[1].getAttribute('aria-label')).toContain('directory');
    });

    it('activates a zarr entry via Enter key', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'sample.zarr', path: 'sample.zarr', type: 'zarr' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      const item = container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement;
      item.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sample.zarr');
      expect(onClose).toHaveBeenCalled();
    });

    it('activates a directory entry via Space key', async () => {
      const entries: DirectoryEntry[] = [{ name: 'sub', path: 'sub', type: 'directory' }];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      navigateMock.mockClear();
      navigateMock.mockResolvedValue(defaultNavigateResult({ currentPath: 'sub' }));

      const item = container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement;
      item.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));

      expect(navigateMock).toHaveBeenCalledWith('sub');
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('breadcrumb segments are <button> (focusable, Enter/Space activatable)', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: 'data/sub', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        const breadcrumb = container.querySelector('#luxar-dataset-browser-breadcrumb');
        expect(breadcrumb?.querySelectorAll('button').length).toBeGreaterThan(0);
      });

      // Root + each non-final segment is a <button>; the final segment is a <span>.
      const buttons = container.querySelectorAll(
        '#luxar-dataset-browser-breadcrumb button.luxar-dataset-browser__breadcrumb-link'
      );
      expect(buttons.length).toBe(2); // "Root" and "data" (final "sub" is current)
      const current = container.querySelector('.luxar-dataset-browser__breadcrumb-current');
      expect(current?.getAttribute('aria-current')).toBe('location');
    });

    it('manual-entry input is associated with a <label>', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
      });

      const label = container.querySelector('label[for="luxar-dataset-browser-manual-path"]');
      expect(label).not.toBeNull();
      const input = container.querySelector(
        '#luxar-dataset-browser-manual-path'
      ) as HTMLInputElement;
      expect(input.getAttribute('aria-label')).toBe('Dataset path');
    });

    it('status bar has aria-live="polite"', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      const status = container.querySelector('#luxar-dataset-browser-status') as HTMLElement;
      expect(status.getAttribute('aria-live')).toBe('polite');
    });
  });

  describe('show / hide / close', () => {
    it('show() makes the panel visible (display=flex)', () => {
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      browser.hide();
      browser.show();

      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      expect(panel.style.display).toBe('flex');
    });

    it('hide() sets display=none', () => {
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      browser.hide();

      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      expect(panel.style.display).toBe('none');
    });

    it('close() removes the panel and fires onClose', () => {
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      browser.close();

      expect(container.querySelector('#luxar-dataset-browser')).toBeNull();
      expect(onClose).toHaveBeenCalled();
    });

    it('the close button triggers close()', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });

      const closeBtn = container.querySelector(
        '.luxar-dataset-browser__close-btn'
      ) as HTMLButtonElement;
      closeBtn.click();

      expect(container.querySelector('#luxar-dataset-browser')).toBeNull();
      expect(onClose).toHaveBeenCalled();
    });

    it('close() works without an onClose callback', () => {
      const browser = new DatasetBrowser({ container, onDatasetSelect });
      // Must not throw.
      browser.close();
      expect(container.querySelector('#luxar-dataset-browser')).toBeNull();
    });
  });

  describe('navigation cancellation', () => {
    type DeferredNavigate = {
      promise: Promise<unknown>;
      resolve: (v: unknown) => void;
      reject: (e: unknown) => void;
    };
    const makeDeferred = (): DeferredNavigate => {
      let resolve: DeferredNavigate['resolve'] = () => {};
      let reject: DeferredNavigate['reject'] = () => {};
      const promise = new Promise<unknown>((res, rej) => {
        resolve = res;
        reject = rej;
      });
      return { promise, resolve, reject };
    };

    // Helper: call the private `navigate` method via cast. Lets these
    // tests exercise the cancellation path without depending on any
    // particular DOM-trigger surface.
    //
    // MED-48 (audit-ack): the audit suggested promoting `navigate()`
    // to a public method (or returning a cancellation token). Rejected:
    // `navigate()` is an implementation detail of the panel's
    // generation-counter cancellation contract, and exposing it would
    // (a) widen the public surface for one test-only need, and (b)
    // tempt callers to drive navigation outside the DOM event flow.
    // The cast below is the intentional test-seam — small, localized,
    // and confined to the cancellation suite. The production
    // cancellation contract is also covered indirectly by the
    // `close()`-cancels-in-flight-navigate test.
    type BrowserInternals = { navigate: (path: string) => Promise<void> };

    it('a stale navigate result is discarded when a newer navigate is in flight', async () => {
      // Initial constructor navigate resolves immediately so the
      // panel reaches steady state. Then start two more — slow then
      // fast — and confirm the fast one wins.
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });

      const slow = makeDeferred();
      const fast = makeDeferred();
      navigateMock.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);

      const internals = browser as unknown as BrowserInternals;
      void internals.navigate('slow_path');
      void internals.navigate('fast_path');

      // Resolve fast first — its render should be visible.
      fast.resolve(
        defaultNavigateResult({
          entries: [{ name: 'fast.zarr', path: 'fast.zarr', type: 'zarr' }],
        })
      );
      await vi.waitFor(() => {
        const items = container.querySelectorAll('.luxar-dataset-browser__file-item');
        expect(items.length).toBe(1);
      });
      const namesAfterFast = Array.from(
        container.querySelectorAll('.luxar-dataset-browser__file-name')
      ).map((el) => el.textContent);
      expect(namesAfterFast).toEqual(['fast.zarr']);

      // Now resolve the stale slow navigate — its result must NOT
      // overwrite the UI.
      slow.resolve(
        defaultNavigateResult({
          entries: [{ name: 'slow.zarr', path: 'slow.zarr', type: 'zarr' }],
        })
      );
      await Promise.resolve();
      await Promise.resolve();
      const namesAfterSlow = Array.from(
        container.querySelectorAll('.luxar-dataset-browser__file-name')
      ).map((el) => el.textContent);
      expect(namesAfterSlow).toEqual(['fast.zarr']);
    });

    it('a stale navigate that resolves to a zarr path does NOT fire onDatasetSelect', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });

      const slow = makeDeferred();
      const fast = makeDeferred();
      navigateMock.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);

      const internals = browser as unknown as BrowserInternals;
      void internals.navigate('slow_path');
      void internals.navigate('fast_path');

      // Fast (newer) navigate finishes as a normal directory.
      fast.resolve(defaultNavigateResult({ entries: [] }));
      await Promise.resolve();
      await Promise.resolve();

      // Slow (stale) navigate now resolves to a zarr — must NOT fire
      // onDatasetSelect, otherwise the user would be navigated to a
      // path they already left.
      slow.resolve(
        defaultNavigateResult({
          isZarr: true,
          currentPath: 'slow.zarr',
          entries: [],
        })
      );
      await Promise.resolve();
      await Promise.resolve();
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('close() cancels an in-flight navigate so onDatasetSelect cannot fire post-close', async () => {
      navigateMock.mockResolvedValueOnce(defaultNavigateResult());

      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });

      const slow = makeDeferred();
      navigateMock.mockReturnValueOnce(slow.promise);

      const internals = browser as unknown as BrowserInternals;
      void internals.navigate('late_path');

      // User dismisses the panel before the navigate completes.
      browser.close();
      expect(container.querySelector('#luxar-dataset-browser')).toBeNull();

      // The slow navigate now resolves to a zarr — must NOT fire
      // onDatasetSelect since the panel is gone.
      slow.resolve(
        defaultNavigateResult({
          isZarr: true,
          currentPath: 'late.zarr',
          entries: [],
        })
      );
      await Promise.resolve();
      await Promise.resolve();
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });
  });

  describe('async onDatasetSelect rejection handling', () => {
    it('async rejection from onDatasetSelect is caught and toasted, not unhandled', async () => {
      const unhandledRejections: unknown[] = [];
      const onUnhandled = (e: PromiseRejectionEvent): void => {
        unhandledRejections.push(e.reason);
      };
      window.addEventListener('unhandledrejection', onUnhandled);

      try {
        const asyncFail: (url: string) => Promise<void> = () =>
          Promise.reject(new Error('load failed'));

        new DatasetBrowser({
          container,
          onDatasetSelect: asyncFail,
          onClose,
        });

        // Auto-load path: navigate resolves to a zarr → onDatasetSelect fires.
        navigateMock.mockResolvedValueOnce(
          defaultNavigateResult({
            isZarr: true,
            currentPath: 'data.zarr',
            entries: [],
          })
        );

        // Trigger a navigation that fires onDatasetSelect via the auto-zarr path.
        // Easiest: rely on the construction-time navigate() call, which
        // beforeEach has set up to return non-zarr. Override + force a
        // re-navigate via clicking. For this test we just use a manual
        // entry path: open the manual form, type, submit.
        navigateMock.mockResolvedValue(
          defaultNavigateResult({
            entries: [],
            strategy: 'manual',
          })
        );

        // Yield microtasks so any pending onDatasetSelect call settles.
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();

        // No unhandled rejection should propagate from the async failure
        // wrapping the dataset-browser swallows + toasts.
        expect(unhandledRejections).toHaveLength(0);
      } finally {
        window.removeEventListener('unhandledrejection', onUnhandled);
      }
    });
  });

  describe('initial focus and type-to-filter (#1922)', () => {
    const ENTRIES: DirectoryEntry[] = [
      { name: 'alpha.zarr', path: 'alpha.zarr', type: 'zarr' },
      { name: 'beta.zarr', path: 'beta.zarr', type: 'zarr' },
      { name: 'omega_dir', path: 'omega_dir', type: 'directory' },
    ];

    /** Construct a browser whose first listing has renderable entries. */
    async function openWithListing(): Promise<HTMLElement> {
      navigateMock.mockReset();
      navigateMock.mockResolvedValue(defaultNavigateResult({ entries: ENTRIES, strategy: 'html' }));
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
      });
      return container.querySelector('#luxar-dataset-browser') as HTMLElement;
    }

    const searchInput = (): HTMLInputElement =>
      container.querySelector('#luxar-dataset-browser-search') as HTMLInputElement;
    const renderedNames = (): string[] =>
      Array.from(container.querySelectorAll<HTMLElement>('.luxar-dataset-browser__file-name')).map(
        (el) => el.textContent ?? ''
      );

    function press(init: KeyboardEventInit & { key: string }): KeyboardEvent {
      const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
      (document.activeElement ?? document.body).dispatchEvent(event);
      return event;
    }

    it('parks focus on the panel, never on the search field', async () => {
      const panel = await openWithListing();

      // Focus is on the panel container — NOT a typing surface, which is what
      // keeps `O` a working toggle (InputHandler drops non-Escape keys while
      // `isTypingInInput(document.activeElement)` is true).
      expect(document.activeElement).toBe(panel);
      expect(isTypingInInput(document.activeElement)).toBe(false);
      expect(panel.getAttribute('tabindex')).toBe('-1');
    });

    it('the first printable keystroke lands in the search field and filters', async () => {
      await openWithListing();

      const event = press({ key: 'b' });

      expect(document.activeElement).toBe(searchInput());
      expect(searchInput().value).toBe('b');
      // The search field's own `oninput` ran — the listing is really narrowed.
      expect(renderedNames()).toEqual(['beta.zarr']);
      expect(event.defaultPrevented).toBe(true);
    });

    it('leaves `O` to the global binding so the browser stays a toggle', async () => {
      const seen: string[] = [];
      const listener = (e: KeyboardEvent) => seen.push(e.key);
      document.addEventListener('keydown', listener);
      try {
        const panel = await openWithListing();

        const event = press({ key: 'o' });

        expect(seen).toEqual(['o']);
        expect(event.defaultPrevented).toBe(false);
        expect(searchInput().value).toBe('');
        expect(document.activeElement).toBe(panel);
        // Sanity: the listing is untouched, so `o` really did nothing local.
        expect(renderedNames()).toEqual(['alpha.zarr', 'beta.zarr', 'omega_dir']);
      } finally {
        document.removeEventListener('keydown', listener);
      }
    });

    it('a forwarded printable never reaches the global bindings', async () => {
      const seen: string[] = [];
      const listener = (e: KeyboardEvent) => seen.push(e.key);
      document.addEventListener('keydown', listener);
      try {
        await openWithListing();

        // `b` toggles the scale bar globally; typing it into the filter must
        // not also toggle it behind the modal.
        press({ key: 'b' });

        expect(searchInput().value).toBe('b');
        expect(seen).toEqual([]);
      } finally {
        document.removeEventListener('keydown', listener);
      }
    });

    it('keeps non-printable global shortcuts off the scene while it is modal', async () => {
      const seen: string[] = [];
      const listener = (e: KeyboardEvent) => seen.push(e.key);
      document.addEventListener('keydown', listener);
      try {
        await openWithListing();

        // Home/End jump the selected dimension; Shift+arrows change the
        // animation speed. Neither belongs to a scene hidden behind an
        // `aria-modal` panel.
        press({ key: 'Home' });
        press({ key: 'ArrowUp', shiftKey: true });
        expect(seen).toEqual([]);

        // Escape still has to get out so the browser can close.
        press({ key: 'Escape' });
        expect(seen).toEqual(['Escape']);
      } finally {
        document.removeEventListener('keydown', listener);
      }
    });

    it('does not double-insert once the search field holds focus', async () => {
      await openWithListing();
      press({ key: 'b' });
      expect(document.activeElement).toBe(searchInput());

      // The browser types this one itself — the forwarder must not also
      // append it, or the field would read "be" after a single keystroke.
      const event = press({ key: 'e' });

      expect(event.defaultPrevented).toBe(false);
      expect(searchInput().value).toBe('b');
    });

    it('ArrowDown moves from the panel into the listing', async () => {
      const panel = await openWithListing();
      expect(document.activeElement).toBe(panel);

      const event = press({ key: 'ArrowDown' });

      // With focus parked on the container, the search field's own ArrowDown
      // handler never sees the key and the list's handler bails (its target
      // is not a row) — the container has to provide the affordance.
      const firstRow = container.querySelector('.luxar-dataset-browser__file-item');
      expect(document.activeElement).toBe(firstRow);
      expect(event.defaultPrevented).toBe(true);
    });

    it('directory navigation from the listing keeps focus inside the panel', async () => {
      navigateMock.mockReset();
      navigateMock
        .mockResolvedValueOnce(
          defaultNavigateResult({
            entries: [{ name: 'nested', path: 'nested', type: 'directory' }],
            strategy: 'html',
          })
        )
        .mockResolvedValueOnce(defaultNavigateResult({ currentPath: 'nested', entries: [] }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });
      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;

      press({ key: 'ArrowDown' });
      expect(document.activeElement).toBe(
        container.querySelector('.luxar-dataset-browser__file-item')
      );

      press({ key: 'Enter' });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });

      expect(panel.contains(document.activeElement)).toBe(true);
    });

    it('an in-flight navigation keeps focus and shortcuts inside the panel', async () => {
      navigateMock.mockReset();
      let resolveNextListing: (result: unknown) => void = () => {};
      const nextListing = new Promise<unknown>((resolve) => {
        resolveNextListing = resolve;
      });
      navigateMock
        .mockResolvedValueOnce(
          defaultNavigateResult({
            entries: [{ name: 'nested', path: 'nested', type: 'directory' }],
            strategy: 'html',
          })
        )
        .mockReturnValueOnce(nextListing);

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });
      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      const seen: string[] = [];
      const listener = (event: KeyboardEvent) => seen.push(event.key);
      window.addEventListener('keydown', listener);

      try {
        press({ key: 'ArrowDown' });
        press({ key: 'Enter' });

        expect(container.querySelector('.luxar-dataset-browser__loading')).not.toBeNull();
        expect(document.activeElement).toBe(panel);

        press({ key: 'Home' });
        expect(seen).toEqual([]);
      } finally {
        window.removeEventListener('keydown', listener);
        resolveNextListing(defaultNavigateResult({ currentPath: 'nested', entries: [] }));
        await vi.waitFor(() => {
          expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
        });
      }
    });

    it('a completed navigation does not steal focus from a stacked modal', async () => {
      navigateMock.mockReset();
      let resolveNextListing: (result: unknown) => void = () => {};
      const nextListing = new Promise<unknown>((resolve) => {
        resolveNextListing = resolve;
      });
      navigateMock
        .mockResolvedValueOnce(
          defaultNavigateResult({
            entries: [{ name: 'nested', path: 'nested', type: 'directory' }],
            strategy: 'html',
          })
        )
        .mockReturnValueOnce(nextListing);

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      press({ key: 'ArrowDown' });
      press({ key: 'Enter' });

      const stackedModal = document.createElement('div');
      stackedModal.tabIndex = -1;
      document.body.appendChild(stackedModal);
      stackedModal.focus();

      resolveNextListing(defaultNavigateResult({ currentPath: 'nested', entries: [] }));
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
      });

      expect(document.activeElement).toBe(stackedModal);
      stackedModal.remove();
    });

    describe('manual-entry fallback (unlistable server)', () => {
      /** Mount a browser whose directory cannot be listed at all. */
      async function openManualEntry(): Promise<HTMLElement> {
        navigateMock.mockReset();
        navigateMock.mockResolvedValue(defaultNavigateResult({ entries: [], strategy: 'manual' }));
        new DatasetBrowser({ container, onDatasetSelect, onClose });
        await vi.waitFor(() => {
          expect(container.querySelector('#luxar-dataset-browser-manual-path')).not.toBeNull();
        });
        return container.querySelector('#luxar-dataset-browser') as HTMLElement;
      }

      const manualInput = (): HTMLInputElement =>
        container.querySelector('#luxar-dataset-browser-manual-path') as HTMLInputElement;

      it('does not autofocus the manual path field', async () => {
        const panel = await openManualEntry();

        // Autofocusing it would trip InputHandler's typing guard and make `O`
        // one-way again — the exact bug #1922 fixes, and this fallback is
        // reachable in production (S3/CloudFront, nginx `autoindex off`).
        expect(document.activeElement).toBe(panel);
        expect(isTypingInInput(document.activeElement)).toBe(false);
      });

      it('does NOT route type-to-filter into the manual path field', async () => {
        const panel = await openManualEntry();

        // `#luxar-dataset-browser-manual-path` is a URL entry field, not a filter, so stray
        // keystrokes must not be routed into it: a printable key pressed at
        // the container types nothing and leaves focus where it was (the
        // modal contains it instead). The user clicks or Tabs into the field
        // deliberately, and from there the ordinary typing guard applies.
        const event = press({ key: 'd' });

        expect(document.activeElement).toBe(panel);
        expect(manualInput().value).toBe('');
        expect(event.defaultPrevented).toBe(false);
      });

      it('still lets `O` reach the global binding', async () => {
        const seen: string[] = [];
        const listener = (e: KeyboardEvent) => seen.push(e.key);
        document.addEventListener('keydown', listener);
        try {
          const panel = await openManualEntry();

          const event = press({ key: 'o' });

          expect(seen).toEqual(['o']);
          expect(event.defaultPrevented).toBe(false);
          expect(manualInput().value).toBe('');
          expect(document.activeElement).toBe(panel);
        } finally {
          document.removeEventListener('keydown', listener);
        }
      });

      it('a path typed into the focused field keeps `o` local and submits on Enter', async () => {
        const seen: string[] = [];
        const listener = (e: KeyboardEvent) => seen.push(e.key);
        document.addEventListener('keydown', listener);
        try {
          await openManualEntry();
          manualInput().focus();

          // The browser types this one; the passthrough exemption is gated on
          // `event.target === container`, so with focus in the field `o` must
          // NOT also reach the `O` binding and close the panel mid-path.
          const typed = press({ key: 'o' });
          expect(seen).toEqual([]);
          expect(typed.defaultPrevented).toBe(false);

          // Enter still submits the field's own form-ish handler.
          manualInput().value = 'output/scan.zarr';
          press({ key: 'Enter' });

          expect(getFullUrlMock).toHaveBeenCalledWith('output/scan.zarr');
          expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/output/scan.zarr');
        } finally {
          document.removeEventListener('keydown', listener);
        }
      });

      it('Escape from the focused path field still reaches the global handler', async () => {
        const seen: string[] = [];
        const listener = (e: KeyboardEvent) => seen.push(e.key);
        document.addEventListener('keydown', listener);
        try {
          await openManualEntry();
          manualInput().focus();

          press({ key: 'Escape' });

          // InputHandler's typing guard lets Escape through, and containment
          // exempts it, so the panel-close flow still runs from this field.
          expect(seen).toEqual(['Escape']);
        } finally {
          document.removeEventListener('keydown', listener);
        }
      });
    });

    it('contains a printable key while the search bar is hidden (nothing to filter)', async () => {
      // Empty directory → `setSearchVisible(false)`. The search `<input>` is
      // still in the DOM, so a resolver that skipped the visibility check
      // would type into an invisible field. Nothing to filter, and the panel
      // is still modal, so the key is contained rather than released to the
      // scene behind it.
      const seen: string[] = [];
      const listener = (e: KeyboardEvent) => seen.push(e.key);
      document.addEventListener('keydown', listener);
      try {
        navigateMock.mockReset();
        navigateMock.mockResolvedValue(defaultNavigateResult({ entries: [] }));
        new DatasetBrowser({ container, onDatasetSelect, onClose });
        await vi.waitFor(() => {
          expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
        });
        expect(searchInput()).not.toBeNull();

        const event = press({ key: 'b' });

        expect(event.defaultPrevented).toBe(false);
        expect(searchInput().value).toBe('');
        expect(seen).toEqual([]);
      } finally {
        document.removeEventListener('keydown', listener);
      }
    });

    it('types into the search field only while the search bar is shown', async () => {
      // The other half of the resolver's ordering: with the bar visible the
      // same keystroke DOES land in the field (so the test above pins the
      // visibility gate, not merely a broken resolver).
      await openWithListing();
      const bar = container.querySelector('#luxar-dataset-browser-search-bar') as HTMLElement;
      expect(bar.style.display).toBe('');

      press({ key: 'b' });
      expect(searchInput().value).toBe('b');

      // Hide the bar the way `setSearchVisible(false)` does and try again from
      // the container: the resolver must now refuse.
      bar.style.display = 'none';
      (container.querySelector('#luxar-dataset-browser') as HTMLElement).focus();
      const event = press({ key: 'z' });

      expect(event.defaultPrevented).toBe(false);
      expect(searchInput().value).toBe('b');
    });

    describe('shortcuts the panel advertises reach the global bindings', () => {
      /** Record every key that made it to a document-level listener. */
      async function keysReachingGlobals(
        open: () => Promise<void> | void,
        keys: string[]
      ): Promise<string[]> {
        const seen: string[] = [];
        const listener = (e: KeyboardEvent) => seen.push(e.key);
        document.addEventListener('keydown', listener);
        try {
          await open();
          for (const key of keys) press({ key });
        } finally {
          document.removeEventListener('keydown', listener);
        }
        return seen;
      }

      it('renders an `H Help` chip in the welcome banner', () => {
        new DatasetBrowser({ container, onDatasetSelect, onClose });
        const chip = container.querySelector('.luxar-dataset-browser__banner-kbd');
        expect(chip?.textContent).toBe('H');
        expect(container.querySelector('#luxar-dataset-browser-welcome')?.textContent).toContain(
          'Help'
        );
      });

      it('lets `H` and `O` out with a listing rendered', async () => {
        const seen = await keysReachingGlobals(async () => {
          await openWithListing();
        }, ['h', 'o']);

        expect(seen).toEqual(['h', 'o']);
      });

      it('lets `H` and `O` out in the empty-directory state (no filter mounted)', async () => {
        const seen = await keysReachingGlobals(async () => {
          navigateMock.mockReset();
          navigateMock.mockResolvedValue(defaultNavigateResult({ entries: [] }));
          new DatasetBrowser({ container, onDatasetSelect, onClose });
          await vi.waitFor(() => {
            expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
          });
        }, ['h', 'o']);

        // This state has no filter and no `#luxar-dataset-browser-manual-path`, and it persists for
        // as long as the directory stays empty — the chip must not be dead.
        expect(seen).toEqual(['h', 'o']);
      });

      it('lets `H` and `O` out while the directory is still loading', async () => {
        const seen = await keysReachingGlobals(() => {
          navigateMock.mockReset();
          // Never resolves: a slow host can hold this state for tens of
          // seconds (four sequential probe stages in DirectoryNavigator).
          navigateMock.mockReturnValue(new Promise<never>(() => {}));
          new DatasetBrowser({ container, onDatasetSelect, onClose });
          expect(container.querySelector('.luxar-dataset-browser__loading')).not.toBeNull();
        }, ['h', 'o']);

        expect(seen).toEqual(['h', 'o']);
      });

      it('still contains every OTHER shortcut in those states', async () => {
        const seen = await keysReachingGlobals(async () => {
          navigateMock.mockReset();
          navigateMock.mockResolvedValue(defaultNavigateResult({ entries: [] }));
          new DatasetBrowser({ container, onDatasetSelect, onClose });
          await vi.waitFor(() => {
            expect(container.querySelector('.luxar-dataset-browser__empty')).not.toBeNull();
          });
        }, ['b', 'v', 'p', 'Home', 'End']);

        expect(seen).toEqual([]);
      });
    });

    it('close() releases the forwarder', async () => {
      navigateMock.mockReset();
      navigateMock.mockResolvedValue(defaultNavigateResult({ entries: ENTRIES, strategy: 'html' }));
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
      });
      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      const search = searchInput();

      browser.close();

      const event = new KeyboardEvent('keydown', { key: 'b', bubbles: true, cancelable: true });
      panel.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(false);
      expect(search.value).toBe('');
    });

    it('hide() releases the forwarder and show() re-arms it', async () => {
      navigateMock.mockReset();
      navigateMock.mockResolvedValue(defaultNavigateResult({ entries: ENTRIES, strategy: 'html' }));
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
      });
      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;

      browser.hide();
      const whileHidden = new KeyboardEvent('keydown', {
        key: 'b',
        bubbles: true,
        cancelable: true,
      });
      panel.dispatchEvent(whileHidden);
      expect(whileHidden.defaultPrevented).toBe(false);
      expect(searchInput().value).toBe('');

      browser.show();
      // Re-shown: focus is back on the panel and typing filters again.
      expect(document.activeElement).toBe(panel);
      press({ key: 'b' });
      expect(searchInput().value).toBe('b');
      expect(renderedNames()).toEqual(['beta.zarr']);
      browser.close();
    });

    it('show() re-parks focus even when the forwarder was never released', async () => {
      navigateMock.mockReset();
      navigateMock.mockResolvedValue(defaultNavigateResult({ entries: ENTRIES, strategy: 'html' }));
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelectorAll('.luxar-dataset-browser__file-item').length).toBe(3);
      });
      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;

      // Focus wandered into the listing; show() is then called WITHOUT a
      // preceding hide(), so `??=` skips the re-install (and with it the
      // forwarder's container focus). The panel must re-park focus anyway.
      const firstRow = container.querySelector('.luxar-dataset-browser__file-item') as HTMLElement;
      firstRow.focus();
      expect(document.activeElement).toBe(firstRow);

      browser.show();

      expect(document.activeElement).toBe(panel);
      browser.close();
    });
  });

  describe('modal focus trap', () => {
    it('Tab on the last focusable wraps to the first (and Shift+Tab back)', async () => {
      // vi.clearAllMocks() does NOT drop queued mockResolvedValueOnce values
      // from earlier tests (it clears calls, not once-queues) — a stale zarr
      // result would auto-select and close the browser. Hard-reset first.
      navigateMock.mockReset();
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ entries: [{ name: 'a.zarr', path: 'a.zarr', type: 'zarr' }] })
      );
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      const focusables = panel.querySelectorAll<HTMLElement>(
        'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusables[0];
      const last = focusables[focusables.length - 1];

      last.focus();
      panel.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
      expect(document.activeElement).toBe(first);

      first.focus();
      panel.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true })
      );
      expect(document.activeElement).toBe(last);
    });

    it('Tab from the panel container itself enters the trap instead of escaping', async () => {
      // The panel is `tabindex="-1"` and holds initial focus (#1922), so it is
      // NOT in the trap's focusable list. Without explicit steering the
      // browser default would walk Shift+Tab out of the modal.
      navigateMock.mockReset();
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ entries: [{ name: 'a.zarr', path: 'a.zarr', type: 'zarr' }] })
      );
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      const focusables = panel.querySelectorAll<HTMLElement>(
        'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      expect(document.activeElement).toBe(panel);

      const forward = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
      panel.dispatchEvent(forward);
      expect(forward.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(first);

      panel.focus();
      const back = new KeyboardEvent('keydown', {
        key: 'Tab',
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      });
      panel.dispatchEvent(back);
      expect(back.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(last);
    });

    it('hide() releases the trap (Tab escapes a hidden modal); show() re-arms it', async () => {
      navigateMock.mockReset();
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ entries: [{ name: 'a.zarr', path: 'a.zarr', type: 'zarr' }] })
      );
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });
      const panel = container.querySelector('#luxar-dataset-browser') as HTMLElement;
      const focusables = panel.querySelectorAll<HTMLElement>(
        'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusables[0];
      const last = focusables[focusables.length - 1];

      // Hidden modal: the trap MUST be released — Tab on the last focusable
      // no longer wraps (a hidden panel holding Tab hostage is a keyboard
      // lock; the browser default proceeds instead).
      browser.hide();
      last.focus();
      panel.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
      expect(document.activeElement).toBe(last);

      // show() re-arms: Tab wraps again.
      browser.show();
      last.focus();
      panel.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
      expect(document.activeElement).toBe(first);
      browser.close();
    });

    it('close() releases the trap and returns focus to the previously-focused element', async () => {
      const outside = document.createElement('button');
      document.body.appendChild(outside);
      outside.focus();

      navigateMock.mockReset();
      navigateMock.mockResolvedValue(defaultNavigateResult());
      const browser = new DatasetBrowser({ container, onDatasetSelect, onClose });
      browser.close();

      expect(document.activeElement).toBe(outside);
      outside.remove();
    });
  });
});
