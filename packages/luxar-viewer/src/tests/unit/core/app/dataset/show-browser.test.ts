// @vitest-environment jsdom
/**
 * Unit tests for core/app/dataset/show-browser.ts (G12).
 *
 * `showDatasetBrowser` is critical-path: it's the modal that opens
 * when bootstrap can't auto-load a dataset (empty src, trailing slash,
 * non-zarr URL). Contract under test:
 *
 *   - Pre-open: `clearError()` runs (any prior load failure should
 *     disappear when the browser opens).
 *   - Construct: new DatasetBrowser with the supplied src + body
 *     container.
 *   - Wire: inputHandler.setDatasetBrowser(browser) so Escape can
 *     route through close().
 *   - On select: trailing slashes are stripped, options.src is
 *     updated via onSrcChange, host URL is replaced when
 *     updateBrowserUrl is true, and loadDataset receives the clean URL.
 *   - On close: onClose fires, then inputHandler.setDatasetBrowser(undefined)
 *     clears the close handle.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const mocks = vi.hoisted(() => ({
  clearError: vi.fn(),
  showError: vi.fn(),
  showToast: vi.fn(),
  replaceBrowserDataSourceUrl: vi.fn(),
  DatasetBrowserCtor: vi.fn(),
  logError: vi.fn(),
}));

vi.mock('../../../../../ui/dataset-browser', () => ({
  DatasetBrowser: mocks.DatasetBrowserCtor,
}));
vi.mock('../../../../../ui/error-overlay', () => ({
  clearError: mocks.clearError,
  showError: mocks.showError,
}));
vi.mock('../../../../../ui/toast', () => ({
  showToast: mocks.showToast,
}));
vi.mock('../../../../../config/url-params', () => ({
  replaceBrowserDataSourceUrl: mocks.replaceBrowserDataSourceUrl,
}));
vi.mock('../../../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: mocks.logError,
  },
  Modules: { LUXAR: 'LUXAR' },
}));

import { showDatasetBrowser } from '../../../../../core/app/dataset/show-browser';

interface CapturedOpts {
  container: HTMLElement;
  currentSrc: string | undefined;
  onDatasetSelect: (url: string) => false | Promise<void>;
  onClose: () => void;
}

beforeEach(() => {
  mocks.clearError.mockReset();
  mocks.showError.mockReset();
  mocks.showToast.mockReset();
  mocks.replaceBrowserDataSourceUrl.mockReset();
  mocks.DatasetBrowserCtor.mockReset();
  mocks.logError.mockReset();
  mocks.DatasetBrowserCtor.mockImplementation((opts: CapturedOpts) => ({
    close: vi.fn(),
    __opts: opts,
  }));
});

import type { InputHandler } from '../../../../../input';

interface MockInputHandler {
  setDatasetBrowser: ReturnType<typeof vi.fn>;
}

function makePorts() {
  const inputHandler: MockInputHandler = {
    setDatasetBrowser: vi.fn(),
  };
  return {
    currentSrc: 'http://example.com/initial.zarr' as string | undefined,
    updateBrowserUrl: true,
    inputHandler: inputHandler as unknown as InputHandler & MockInputHandler,
    onSrcChange: vi.fn(),
    isInitializing: vi.fn().mockReturnValue(false),
    isSwitchInFlight: vi.fn().mockReturnValue(false),
    loadDataset: vi.fn().mockResolvedValue(undefined),
    shortcutForAction: vi.fn().mockReturnValue(undefined),
    onClose: vi.fn(),
  };
}

describe('showDatasetBrowser', () => {
  it('clears any prior error overlay before opening the modal', () => {
    const ports = makePorts();
    showDatasetBrowser(ports);
    expect(mocks.clearError).toHaveBeenCalledOnce();
  });

  it('constructs DatasetBrowser with document.body + currentSrc', () => {
    const ports = makePorts();
    showDatasetBrowser(ports);

    expect(mocks.DatasetBrowserCtor).toHaveBeenCalledOnce();
    const args = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;
    expect(args.container).toBe(document.body);
    expect(args.currentSrc).toBe('http://example.com/initial.zarr');
    expect(typeof args.onDatasetSelect).toBe('function');
    expect(typeof args.onClose).toBe('function');
  });

  it('returns the constructed DatasetBrowser instance', () => {
    const ports = makePorts();
    const browser = showDatasetBrowser(ports);
    expect(browser).toBeDefined();
    // The returned object IS the one DatasetBrowserCtor produced.
    expect(browser).toBe(mocks.DatasetBrowserCtor.mock.results[0].value);
  });

  it('wires the browser into inputHandler.setDatasetBrowser for Escape routing', () => {
    const ports = makePorts();
    const browser = showDatasetBrowser(ports);
    expect(ports.inputHandler.setDatasetBrowser).toHaveBeenCalledExactlyOnceWith(browser);
  });

  describe('onDatasetSelect callback', () => {
    it('strips trailing slashes from the URL before loading', async () => {
      const ports = makePorts();
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await opts.onDatasetSelect('http://example.com/dataset.zarr///');

      expect(ports.onSrcChange).toHaveBeenCalledWith('http://example.com/dataset.zarr');
      expect(ports.loadDataset).toHaveBeenCalledWith('http://example.com/dataset.zarr');
      // Hostname-only is left alone.
    });

    it('passes URL without trailing slash unchanged', async () => {
      const ports = makePorts();
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await opts.onDatasetSelect('http://example.com/data.zarr');

      expect(ports.loadDataset).toHaveBeenCalledWith('http://example.com/data.zarr');
    });

    it('updates the host URL when updateBrowserUrl is true', async () => {
      const ports = makePorts();
      ports.updateBrowserUrl = true;
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await opts.onDatasetSelect('http://example.com/data.zarr/');

      expect(mocks.replaceBrowserDataSourceUrl).toHaveBeenCalledExactlyOnceWith(
        'http://example.com/data.zarr'
      );
    });

    it('drops a #layers= fragment so settings do not carry onto the new dataset', async () => {
      const ports = makePorts();
      ports.updateBrowserUrl = true;
      window.history.replaceState(
        null,
        '',
        '#foo=1&layers=%7B%22version%22%3A1%2C%22layers%22%3A%7B%7D%7D'
      );
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await opts.onDatasetSelect('http://example.com/data.zarr');

      expect(window.location.hash).toBe('#foo=1');
      window.history.replaceState(null, '', window.location.pathname);
    });

    it('does NOT update the host URL when updateBrowserUrl is false', async () => {
      const ports = makePorts();
      ports.updateBrowserUrl = false;
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await opts.onDatasetSelect('http://example.com/data.zarr');

      expect(mocks.replaceBrowserDataSourceUrl).not.toHaveBeenCalled();
      // …but onSrcChange + loadDataset still run.
      expect(ports.onSrcChange).toHaveBeenCalled();
      expect(ports.loadDataset).toHaveBeenCalled();
    });

    it('skips the URL/src side effects (but still dispatches) when a switch is in flight', async () => {
      const ports = makePorts();
      ports.updateBrowserUrl = true;
      ports.isSwitchInFlight.mockReturnValue(true);
      ports.loadDataset.mockRejectedValue(new Error('a dataset switch is already in progress'));
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await expect(opts.onDatasetSelect('http://example.com/stale.zarr')).rejects.toThrow(
        /in progress/
      );

      // The rejected selection must not leave the host URL or the src
      // snapshot pointing at a dataset that never loaded.
      expect(mocks.replaceBrowserDataSourceUrl).not.toHaveBeenCalled();
      expect(ports.onSrcChange).not.toHaveBeenCalled();
      // The guarded dispatch still runs so the caller observes the rejection.
      expect(ports.loadDataset).toHaveBeenCalledWith('http://example.com/stale.zarr');
    });

    it('keeps the browser open and shows one neutral hint while the app is initializing', () => {
      const ports = makePorts();
      ports.isInitializing.mockReturnValue(true);
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      expect(opts.onDatasetSelect('http://example.com/stale.zarr')).toBe(false);

      expect(mocks.showToast).toHaveBeenCalledExactlyOnceWith(
        'Luxar is still starting up; try again in a moment.'
      );
      expect(mocks.replaceBrowserDataSourceUrl).not.toHaveBeenCalled();
      expect(ports.onSrcChange).not.toHaveBeenCalled();
      expect(ports.loadDataset).not.toHaveBeenCalled();
      expect(mocks.showError).not.toHaveBeenCalled();
      expect(mocks.logError).not.toHaveBeenCalled();
    });

    it('calls onSrcChange BEFORE loadDataset (so a follow-on browser open lands in the right dir)', async () => {
      const ports = makePorts();
      const order: string[] = [];
      ports.onSrcChange.mockImplementation(() => order.push('onSrcChange'));
      ports.loadDataset.mockImplementation(async () => {
        order.push('loadDataset');
      });
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      await opts.onDatasetSelect('http://example.com/data.zarr');

      expect(order).toEqual(['onSrcChange', 'loadDataset']);
    });

    // [core OOS] Pre-fix, loadDataset rejections became unhandled promise
    // rejections (the normal DatasetBrowser selection path returns
    // `Promise<void>` and the modal doesn't surface its own errors).
    // Now we wrap the call: log + show the failure in the user-facing
    // overlay, and re-throw so awaiting callers still see it.
    it('surfaces a loadDataset failure via showError + log.error AND re-throws', async () => {
      const ports = makePorts();
      ports.loadDataset.mockRejectedValue(new Error('simulated zarr 404'));
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      // Awaiting caller MUST see the rejection (regression guard).
      await expect(opts.onDatasetSelect('http://example.com/bad.zarr')).rejects.toThrow(
        /simulated zarr 404/
      );

      // AND the error overlay must be surfaced (silent-caller path).
      expect(mocks.showError).toHaveBeenCalledWith(
        expect.stringMatching(/Failed to load dataset.*simulated zarr 404/),
        ports.shortcutForAction,
        { datasetBrowser: 'dataset-browser.toggle', help: 'help.toggle' }
      );
      // AND log.error was called with the structured payload.
      expect(mocks.logError).toHaveBeenCalledWith(
        'LUXAR',
        expect.stringMatching(
          /loadDataset failed for http:\/\/example\.com\/bad\.zarr.*simulated zarr 404/
        ),
        expect.any(Error)
      );
    });
  });

  describe('onClose callback', () => {
    it('fires onClose then clears the input-handler dataset-browser handle', () => {
      const ports = makePorts();
      const order: string[] = [];
      ports.onClose.mockImplementation(() => order.push('onClose'));
      ports.inputHandler.setDatasetBrowser.mockImplementation((arg: unknown) => {
        order.push(arg === undefined ? 'clearHandle' : 'wireHandle');
      });

      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;
      opts.onClose();

      // First the setDatasetBrowser(browser) wire (during show), then
      // onClose, then setDatasetBrowser(undefined) clear.
      expect(order).toEqual(['wireHandle', 'onClose', 'clearHandle']);
    });

    it('clearHandle path passes undefined explicitly', () => {
      const ports = makePorts();
      showDatasetBrowser(ports);
      const opts = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;

      ports.inputHandler.setDatasetBrowser.mockClear();
      opts.onClose();

      // The post-close call is `setDatasetBrowser(undefined)` — a
      // mutation that passed `null` would still satisfy a loose
      // truthiness check but the implementation pins `undefined`.
      expect(ports.inputHandler.setDatasetBrowser).toHaveBeenCalledExactlyOnceWith(undefined);
    });
  });

  it('forwards an undefined currentSrc unchanged', () => {
    const ports = makePorts();
    ports.currentSrc = undefined;
    showDatasetBrowser(ports);
    const args = mocks.DatasetBrowserCtor.mock.calls[0][0] as CapturedOpts;
    expect(args.currentSrc).toBeUndefined();
  });
});
