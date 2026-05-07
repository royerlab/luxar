/**
 * Unit tests for ui/panels/dataset-browser.ts.
 *
 * Mocks DirectoryNavigator (the only external dependency that touches
 * the network). The panel construction + entry rendering run for real
 * under jsdom — we only stub the async directory listings.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { DirectoryEntry } from '../../../../data';
import { DatasetBrowser } from '../../../../ui/panels/dataset-browser';

const navigateMock = vi.fn();
const getFullUrlMock = vi.fn();

vi.mock('../../../../data', async () => {
  const actual = await vi.importActual<typeof import('../../../../data')>(
    '../../../../data'
  );
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

function defaultNavigateResult(overrides: {
  entries?: DirectoryEntry[];
  currentPath?: string;
  strategy?: 'webdav' | 'html' | 'index' | 'manual';
  isZarr?: boolean;
} = {}) {
  return {
    entries: overrides.entries ?? [],
    currentPath: overrides.currentPath ?? '',
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

    it('shows a close button with aria-label', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });

      const closeBtn = container.querySelector(
        '.luxar-dataset-browser__close-btn'
      ) as HTMLButtonElement;
      expect(closeBtn).not.toBeNull();
      expect(closeBtn.getAttribute('aria-label')).toBe('Close dataset browser');
      expect(closeBtn.textContent).toBe('×');
    });

    it('navigates to root with no currentSrc', () => {
      new DatasetBrowser({ container, onDatasetSelect, onClose });
      // Initial navigate('') call lives inside the constructor.
      expect(navigateMock).toHaveBeenCalledWith('');
    });

    it('extracts parent directory when currentSrc points inside a .zarr', () => {
      new DatasetBrowser({
        container,
        onDatasetSelect,
        onClose,
        currentSrc: 'http://server.test/data/sample.zarr',
        origin: 'http://server.test',
      });

      // The parent-directory branch sets initialPath='' and stores the
      // dataset name for highlighting later — assertion is just that the
      // initial navigate call fires.
      expect(navigateMock).toHaveBeenCalledWith('');
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
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ entries, strategy: 'webdav' })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        const items = container.querySelectorAll('.luxar-dataset-browser__file-item');
        expect(items.length).toBe(4);
      });

      const items = container.querySelectorAll('.luxar-dataset-browser__file-item');
      const orderedNames = Array.from(items).map((el) =>
        (el.querySelector('.luxar-dataset-browser__file-name') as HTMLElement).textContent
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
      expect(container.querySelector('.luxar-dataset-browser__empty')?.textContent).toBe(
        'Empty directory'
      );
    });

    it('shows manual-entry form when strategy=manual + 0 entries', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__manual-entry')).not.toBeNull();
      });
      expect(container.querySelector('#manual-path')).not.toBeNull();
      expect(container.querySelector('#manual-load')).not.toBeNull();
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
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ currentPath: '', isZarr: true })
      );

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

      const item = container.querySelector(
        '.luxar-dataset-browser__file-item'
      ) as HTMLElement;
      item.click();

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/sample.zarr');
      expect(onClose).toHaveBeenCalled();
    });

    it('navigates into a directory entry on click', async () => {
      const entries: DirectoryEntry[] = [
        { name: 'sub', path: 'sub', type: 'directory' },
      ];
      navigateMock.mockResolvedValueOnce(defaultNavigateResult({ entries }));

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('.luxar-dataset-browser__file-item')).not.toBeNull();
      });

      // Reset the mock so we can observe the second call.
      navigateMock.mockClear();
      navigateMock.mockResolvedValue(defaultNavigateResult({ currentPath: 'sub' }));

      const item = container.querySelector(
        '.luxar-dataset-browser__file-item'
      ) as HTMLElement;
      item.click();

      expect(navigateMock).toHaveBeenCalledWith('sub');
      // Selection callbacks must NOT fire for directories.
      expect(onDatasetSelect).not.toHaveBeenCalled();
    });

    it('manual-entry submission selects the typed path via getFullUrl', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#manual-path')).not.toBeNull();
      });

      const input = container.querySelector('#manual-path') as HTMLInputElement;
      const loadBtn = container.querySelector('#manual-load') as HTMLButtonElement;

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
        expect(container.querySelector('#manual-path')).not.toBeNull();
      });

      const input = container.querySelector('#manual-path') as HTMLInputElement;
      const loadBtn = container.querySelector('#manual-load') as HTMLButtonElement;

      input.value = 'https://other.example/dataset.zarr';
      loadBtn.click();

      expect(getFullUrlMock).not.toHaveBeenCalled();
      expect(onDatasetSelect).toHaveBeenCalledWith('https://other.example/dataset.zarr');
    });

    it('manual-entry ignores empty input', async () => {
      navigateMock.mockResolvedValueOnce(
        defaultNavigateResult({ strategy: 'manual', entries: [] })
      );

      new DatasetBrowser({ container, onDatasetSelect, onClose });
      await vi.waitFor(() => {
        expect(container.querySelector('#manual-path')).not.toBeNull();
      });

      const input = container.querySelector('#manual-path') as HTMLInputElement;
      const loadBtn = container.querySelector('#manual-load') as HTMLButtonElement;

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
        expect(container.querySelector('#manual-path')).not.toBeNull();
      });

      const input = container.querySelector('#manual-path') as HTMLInputElement;
      input.value = 'foo.zarr';

      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

      expect(onDatasetSelect).toHaveBeenCalledWith('http://example.com/foo.zarr');
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
});
