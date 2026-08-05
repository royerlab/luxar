/**
 * Dataset browser UI panel for navigating and selecting Zarr datasets.
 *
 * Provides a user-friendly interface for browsing directories and loading
 * Zarr datasets from various server types.
 */

import { DirectoryNavigator, type DirectoryEntry } from '../data';
import { escapeHtml } from '../utils/escape-html';
import { extractBaseUrl, extractPath } from './dataset-browser/url-utils';
import { log, Modules } from '../utils/log';
import { showToast } from './toast';

/**
 * Wrap an `onDatasetSelect` invocation so a Promise-returning
 * callback (the production path is async — `LuxarApp.loadDataset`)
 * doesn't surface as an unhandled rejection when the browser closes
 * synchronously after firing it.
 */
function safeFireSelect(cb: (url: string) => void | Promise<void>, url: string): void {
  try {
    const result = cb(url);
    if (result && typeof (result as Promise<void>).then === 'function') {
      (result as Promise<void>).catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        log.warning(Modules.UI, `Dataset load failed: ${msg}`);
        showToast(`Failed to load dataset: ${msg}`);
      });
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log.warning(Modules.UI, `Dataset selection threw: ${msg}`);
    showToast(`Failed to load dataset: ${msg}`);
  }
}

/**
 * Construction options for {@link DatasetBrowser} — where to mount the panel and
 * how to react to selection/close, plus the context needed to pick the initial
 * directory and resolve relative paths.
 */
export interface DatasetBrowserConfig {
  /** Host element the browser panel is appended into. */
  container: HTMLElement;
  /**
   * Callback when a dataset is selected. Receives the full URL (not
   * just the path). May be sync or async; the browser does NOT wait for
   * the returned Promise — it fires the callback, attaches a `.catch`
   * (so an async load failure is logged + toasted rather than becoming an
   * unhandled rejection), and closes immediately.
   */
  onDatasetSelect: (fullUrl: string) => void | Promise<void>;
  onClose?: () => void;
  /** Currently loaded dataset URL, used to determine the initial directory. */
  currentSrc?: string;
  /** Fallback origin for relative path resolution (defaults to `window.location.origin`). */
  origin?: string;
}

/**
 * Interactive panel for browsing a server's directory tree and selecting a Zarr
 * dataset to load. Wraps a {@link DirectoryNavigator} for the async listing,
 * renders directory entries, and fires `onDatasetSelect` with the full dataset
 * URL when a `.zarr` is chosen. Navigation uses a generation token so stale
 * async responses from abandoned directories are discarded rather than rendered.
 */
export class DatasetBrowser {
  private container: HTMLElement;
  private panel: HTMLElement;
  private navigator: DirectoryNavigator;
  private onDatasetSelect: (fullUrl: string) => void | Promise<void>;
  private onClose?: () => void;
  private currentDataset?: string;

  /** Origin used for relative path resolution; captured at construction. */
  private readonly origin: string;

  /**
   * Navigation generation token. Incremented on every `navigate()`
   * call; the navigation discards its result if the
   * generation has moved by the time the async navigator fetch
   * resolves. Without this, fast user picks (or simply a slow first
   * response while the user clicks something else) could let the
   * stale result render entries — or worse, fire `onDatasetSelect`
   * for a directory the user already left when the stale response
   * arrives at a `.zarr` path.
   */
  private navigationGeneration = 0;

  /** Entries for the current directory (unfiltered), cached so the search bar can re-filter without re-fetching. */
  private currentEntries: DirectoryEntry[] = [];

  /** Detection strategy label for the current directory, shown in the status bar. */
  private currentStrategy = '';

  /** Live filter text from the search bar; matched case-insensitively against entry names. */
  private filterText = '';

  constructor(config: DatasetBrowserConfig) {
    this.container = config.container;
    this.onDatasetSelect = config.onDatasetSelect;
    this.onClose = config.onClose;
    this.origin = config.origin ?? (typeof window !== 'undefined' ? window.location.origin : '');

    const src = (config.currentSrc ?? '').trim();

    // Determine base URL and initial path for browsing
    let baseUrl: string;
    let initialPath: string;

    if (!src) {
      // No dataset specified - show browser at root, will display manual entry
      baseUrl = this.origin + '/';
      initialPath = ''; // Don't navigate - just show the browser UI
    } else if (src.includes('.zarr/') || src.endsWith('.zarr')) {
      // We're inside or at a zarr dataset - navigate to parent directory
      try {
        const parsed = new URL(src);
        const pathname = parsed.pathname;

        // Find the .zarr part and go to parent directory
        const zarrIndex = pathname.lastIndexOf('.zarr');
        if (zarrIndex > 0) {
          const parentPath = pathname.substring(0, pathname.lastIndexOf('/', zarrIndex - 1));
          baseUrl = parsed.origin + parentPath + '/';

          // Extract just the dataset name for highlighting
          const datasetPath = pathname.substring(parentPath.length + 1);
          const datasetName = datasetPath.split('/')[0];
          initialPath = '';

          // Store the current dataset for highlighting
          this.currentDataset = datasetName;
        } else {
          baseUrl = this.extractBaseUrl(src);
          initialPath = this.extractPath(src);
        }
      } catch {
        // If URL parsing fails, fall back to sensible defaults
        baseUrl = this.origin + '/';
        initialPath = '';
      }
    } else {
      // Has src but not a zarr dataset - use as base for navigation
      baseUrl = this.extractBaseUrl(src);
      initialPath = this.extractPath(src);
    }

    this.navigator = new DirectoryNavigator(baseUrl);
    this.panel = this.createPanel();

    // Start navigation at the determined path
    this.navigate(initialPath);
  }

  /**
   * Extract base URL from a full URL.
   */
  private extractBaseUrl(url: string): string {
    return extractBaseUrl(url, this.origin);
  }

  /**
   * Extract relative path from a full URL. Thin wrapper around the
   * shared {@link extractPath} helper.
   */
  private extractPath(url: string): string {
    return extractPath(url);
  }

  /**
   * Create the browser panel UI.
   */
  private createPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.id = 'luxar-dataset-browser';
    panel.className = 'luxar-dataset-browser luxar-glass-surface dataset-browser'; // luxar-dataset-browser for styling, dataset-browser for E2E tests

    // ARIA attributes for accessibility
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-labelledby', 'luxar-dataset-browser-title');

    // Header
    const header = document.createElement('div');
    header.className = 'luxar-dataset-browser__header';

    const title = document.createElement('h2');
    title.id = 'luxar-dataset-browser-title';
    title.className = 'luxar-dataset-browser__title';
    title.textContent = 'Select Dataset';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'luxar-dataset-browser__close-btn';
    closeBtn.textContent = '×';
    closeBtn.title = 'Close (Escape)';
    closeBtn.setAttribute('aria-label', 'Close dataset browser');
    closeBtn.onclick = () => this.close();

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Compact help banner with essential guidance
    const welcomeBanner = document.createElement('div');
    welcomeBanner.id = 'luxar-dataset-browser-welcome';
    welcomeBanner.className = 'luxar-dataset-browser__banner';

    welcomeBanner.innerHTML = `
      <div class="luxar-dataset-browser__banner-content">
        <div class="luxar-dataset-browser__banner-text">
          <div><strong class="luxar-dataset-browser__banner-title">Luxar</strong> - Interactive Scientific Data Visualization</div>
          <div class="luxar-dataset-browser__banner-description">Browse for <code class="luxar-dataset-browser__banner-code">.zarr</code> or enter path manually</div>
        </div>
        <div class="luxar-dataset-browser__banner-help">
          <kbd class="luxar-dataset-browser__banner-kbd">H</kbd> Help
        </div>
      </div>
    `;

    // Breadcrumb navigation
    const breadcrumb = document.createElement('div');
    breadcrumb.id = 'luxar-dataset-browser-breadcrumb';
    breadcrumb.className = 'luxar-dataset-browser__breadcrumb';

    // Search / filter bar (filters the current directory listing live).
    // Hidden whenever there are no entries to filter (loading, error,
    // empty directory, or the manual-entry fallback) — see setSearchVisible.
    const search = document.createElement('div');
    search.id = 'luxar-dataset-browser-search-bar';
    search.className = 'luxar-dataset-browser__search';
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.id = 'luxar-dataset-browser-search';
    searchInput.className = 'luxar-dataset-browser__search-input';
    searchInput.placeholder = 'Filter datasets…';
    searchInput.setAttribute('aria-label', 'Filter datasets in this directory');
    searchInput.autocomplete = 'off';
    searchInput.oninput = () => {
      this.filterText = searchInput.value;
      this.renderEntries();
    };
    search.appendChild(searchInput);

    // Content area
    const content = document.createElement('div');
    content.id = 'luxar-dataset-browser-content';
    content.className = 'luxar-dataset-browser__content';

    // Status bar
    const statusBar = document.createElement('div');
    statusBar.id = 'luxar-dataset-browser-status';
    statusBar.className = 'luxar-dataset-browser__status';
    statusBar.setAttribute('aria-live', 'polite');

    panel.appendChild(header);
    panel.appendChild(welcomeBanner);
    panel.appendChild(breadcrumb);
    panel.appendChild(search);
    panel.appendChild(content);
    panel.appendChild(statusBar);

    this.container.appendChild(panel);
    return panel;
  }

  /**
   * Navigate to a path and update the UI.
   *
   * Cancellable: each call bumps `navigationGeneration`. If the user
   * kicks off a newer navigate while an older one's
   * `navigator.navigate()` is still in flight, the older call discards
   * its result on resume instead of overwriting the UI or (worst case)
   * firing `onDatasetSelect` for a path the user already left.
   */
  private async navigate(path: string): Promise<void> {
    const content = this.panel.querySelector('#luxar-dataset-browser-content') as HTMLElement;
    const statusBar = this.panel.querySelector('#luxar-dataset-browser-status') as HTMLElement;

    const myGeneration = ++this.navigationGeneration;

    // Show loading state. Hide the search bar while loading; renderEntries
    // re-shows it once we have a non-empty listing to filter.
    content.innerHTML = '<div class="luxar-dataset-browser__loading">Loading...</div>';
    statusBar.textContent = 'Fetching directory contents...';
    this.setSearchVisible(false);

    try {
      const result = await this.navigator.navigate(path);
      // Bail if a newer navigate has started while we were awaiting.
      // The newer call already wrote its loading indicator and is
      // responsible for the next render.
      if (this.navigationGeneration !== myGeneration) return;

      // Update breadcrumb
      this.updateBreadcrumb(result.currentPath);

      // If it's a Zarr dataset, load it directly
      // But only if the path is actually valid (not empty or just the root)
      if (
        result.isZarr &&
        result.currentPath &&
        result.currentPath !== '/' &&
        result.currentPath.includes('.zarr')
      ) {
        // Pass full URL to preserve directory context
        safeFireSelect(this.onDatasetSelect, this.navigator.getFullUrl(result.currentPath));
        this.close();
        return;
      }

      // Cache entries + strategy for the search bar, reset any stale
      // filter from the previous directory, then render.
      this.currentEntries = result.entries;
      this.currentStrategy = result.strategy;
      this.resetFilter();
      this.renderEntries();

      // Handle manual fallback
      if (result.strategy === 'manual' && result.entries.length === 0) {
        this.showManualEntry();
      }
    } catch (error) {
      // If a newer navigate started, don't paint the older error over
      // the newer loading indicator.
      if (this.navigationGeneration !== myGeneration) return;
      content.innerHTML = `
        <div class="luxar-dataset-browser__error">
          <p>Failed to load directory</p>
          <p class="luxar-dataset-browser__error-details">${escapeHtml(String(error))}</p>
        </div>
      `;
      statusBar.textContent = 'Error loading directory';
    }
  }

  /**
   * Update breadcrumb navigation.
   */
  private updateBreadcrumb(currentPath: string): void {
    const breadcrumb = this.panel.querySelector('#luxar-dataset-browser-breadcrumb') as HTMLElement;
    breadcrumb.innerHTML = '';

    // Root link — `<button type="button">` so the breadcrumb is keyboard-
    // focusable (anchors without href aren't) and Enter/Space activate
    // natively. CSS class names are unchanged so existing styling/snapshot
    // tests stay intact.
    const rootLink = document.createElement('button');
    rootLink.type = 'button';
    rootLink.className = 'luxar-dataset-browser__breadcrumb-link';
    rootLink.textContent = 'Root';
    rootLink.onclick = () => this.navigate('');
    breadcrumb.appendChild(rootLink);

    // Path segments
    if (currentPath) {
      const parts = currentPath.split('/').filter(Boolean);
      let accumulated = '';

      parts.forEach((part, index) => {
        // Separator
        const sep = document.createElement('span');
        sep.className = 'luxar-dataset-browser__breadcrumb-separator';
        sep.textContent = '›';
        sep.setAttribute('aria-hidden', 'true');
        breadcrumb.appendChild(sep);

        accumulated += (accumulated ? '/' : '') + part;
        const pathToNavigate = accumulated;

        if (index === parts.length - 1) {
          // Current location (not clickable)
          const current = document.createElement('span');
          current.className = 'luxar-dataset-browser__breadcrumb-current';
          current.textContent = part;
          current.setAttribute('aria-current', 'location');
          breadcrumb.appendChild(current);
        } else {
          // Clickable parent — same a11y rationale as the Root button above.
          const link = document.createElement('button');
          link.type = 'button';
          link.className = 'luxar-dataset-browser__breadcrumb-link';
          link.textContent = part;
          link.onclick = () => this.navigate(pathToNavigate);
          breadcrumb.appendChild(link);
        }
      });
    }
  }

  /**
   * Clear the search bar and filter state (called on every navigation so
   * a filter from the previous directory doesn't carry over).
   */
  private resetFilter(): void {
    this.filterText = '';
    const searchInput = this.panel.querySelector(
      '#luxar-dataset-browser-search'
    ) as HTMLInputElement | null;
    if (searchInput) searchInput.value = '';
  }

  /**
   * Show or hide the search/filter bar. Hidden in states where filtering
   * is meaningless (loading, error, empty directory, manual-entry fallback)
   * so it only appears when there's an actual listing to narrow.
   */
  private setSearchVisible(visible: boolean): void {
    const search = this.panel.querySelector(
      '#luxar-dataset-browser-search-bar'
    ) as HTMLElement | null;
    if (search) search.style.display = visible ? '' : 'none';
  }

  /**
   * Render the current directory's entries, applying the live search
   * filter, and refresh the status bar (visible / total counts +
   * detection strategy). Re-run on navigation and on every keystroke in
   * the search bar.
   */
  private renderEntries(): void {
    const content = this.panel.querySelector('#luxar-dataset-browser-content') as HTMLElement;
    const statusBar = this.panel.querySelector('#luxar-dataset-browser-status') as HTMLElement;
    content.innerHTML = '';

    const total = this.currentEntries.length;
    const strategyText =
      {
        webdav: 'WebDAV',
        html: 'HTML parsing',
        index: 'Index file',
        manual: 'Manual entry',
      }[this.currentStrategy] ?? this.currentStrategy;

    // Apply the case-insensitive substring filter on entry names.
    const query = this.filterText.trim().toLowerCase();
    const entries = query
      ? this.currentEntries.filter((e) => e.name.toLowerCase().includes(query))
      : this.currentEntries;

    // Status bar: show "M of N" while filtering, otherwise just the total.
    const countLabel =
      query && entries.length !== total
        ? `${entries.length} of ${total} items`
        : `${total} item${total === 1 ? '' : 's'}`;
    statusBar.innerHTML = `
      <span>${countLabel}</span>
      <span>Detection: ${strategyText}</span>
    `;

    // Only offer the filter when there's an actual listing to narrow.
    // Stays visible in the no-matches case (total > 0) so the user can
    // edit their query.
    this.setSearchVisible(total > 0);

    if (total === 0) {
      content.innerHTML = '<div class="luxar-dataset-browser__empty">Empty directory</div>';
      return;
    }

    if (entries.length === 0) {
      content.innerHTML = `<div class="luxar-dataset-browser__empty">No matches for “${escapeHtml(this.filterText.trim())}”</div>`;
      return;
    }

    // Sort entries: zarr datasets first, then directories, then files
    const sorted = [...entries].sort((a, b) => {
      if (a.type === b.type) return a.name.localeCompare(b.name);
      if (a.type === 'zarr') return -1;
      if (b.type === 'zarr') return 1;
      if (a.type === 'directory') return -1;
      if (b.type === 'directory') return 1;
      return 0;
    });

    // Create entry list
    const list = document.createElement('div');
    list.className = 'luxar-dataset-browser__file-list';

    sorted.forEach((entry) => {
      const item = document.createElement('div');

      // Check if this is the currently selected dataset
      const isCurrentDataset = this.currentDataset && entry.name === this.currentDataset;

      item.className = `luxar-dataset-browser__file-item ${isCurrentDataset ? 'luxar-dataset-browser__file-item--current' : ''}`;

      // Keyboard accessibility — make rows focusable and ARIA-labeled.
      // Kept as `<div role="button">` (not `<button>`) to preserve the
      // existing flex-row layout containing icon + name + badges; using a
      // real button would force a CSS rewrite. Enter/Space below mirrors
      // the click handler.
      item.setAttribute('role', 'button');
      item.tabIndex = 0;
      const ariaLabel =
        entry.type === 'zarr'
          ? `Open Zarr dataset ${entry.name}`
          : entry.type === 'directory'
            ? `Open directory ${entry.name}`
            : entry.name;
      item.setAttribute('aria-label', ariaLabel);

      // Icon
      const icon = document.createElement('span');
      icon.className = 'luxar-dataset-browser__file-icon';
      if (entry.type === 'zarr') {
        // Use the Luxar emoji icon
        icon.textContent = '🌌';
        icon.title = 'Zarr Dataset';
      } else if (entry.type === 'directory') {
        icon.textContent = '📁';
        icon.title = 'Directory';
      } else {
        icon.textContent = '📄';
        icon.title = 'File';
      }

      // Name
      const name = document.createElement('span');
      name.className = 'luxar-dataset-browser__file-name';
      name.textContent = entry.name;

      // Type badge and current indicator
      if (entry.type === 'zarr') {
        const badgeContainer = document.createElement('div');
        badgeContainer.className = 'luxar-dataset-browser__badges';

        // ZARR badge
        const badge = document.createElement('span');
        badge.className = 'luxar-dataset-browser__badge';
        badge.textContent = 'ZARR';
        badgeContainer.appendChild(badge);

        // Current dataset indicator
        if (isCurrentDataset) {
          const currentBadge = document.createElement('span');
          currentBadge.className =
            'luxar-dataset-browser__badge luxar-dataset-browser__badge--loaded';
          currentBadge.textContent = 'LOADED';
          badgeContainer.appendChild(currentBadge);
        }

        item.appendChild(icon);
        item.appendChild(name);
        item.appendChild(badgeContainer);
      } else {
        item.appendChild(icon);
        item.appendChild(name);
      }

      // Click handler
      const activate = (): void => {
        if (entry.type === 'zarr') {
          // Pass full URL to preserve directory context
          safeFireSelect(this.onDatasetSelect, this.navigator.getFullUrl(entry.path));
          this.close();
        } else if (entry.type === 'directory') {
          this.navigate(entry.path);
        }
      };
      item.onclick = activate;
      item.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          activate();
        }
      };

      list.appendChild(item);
    });

    content.appendChild(list);
  }

  /**
   * Show manual entry form for servers that don't support listing.
   */
  private showManualEntry(): void {
    const content = this.panel.querySelector('#luxar-dataset-browser-content') as HTMLElement;

    content.innerHTML = `
      <div class="luxar-dataset-browser__manual-entry">
        <label for="manual-path" class="luxar-dataset-browser__manual-entry-title">Directory listing not available. Enter dataset path manually:</label>
        <input
          type="text"
          id="manual-path"
          class="luxar-dataset-browser__manual-entry-input"
          placeholder="e.g., datasets/example.luxar.zarr"
          aria-label="Dataset path"
        />
        <div>
          <button id="manual-load" type="button" class="luxar-dataset-browser__manual-entry-btn">Load Dataset</button>
        </div>
        <p class="luxar-dataset-browser__manual-entry-tip">
          Tip: Ask your server administrator to enable directory listing or WebDAV
        </p>
      </div>
    `;

    const input = content.querySelector('#manual-path') as HTMLInputElement;
    const loadBtn = content.querySelector('#manual-load') as HTMLButtonElement;

    loadBtn.onclick = () => {
      const path = input.value.trim();
      if (path) {
        // If it's already a full URL, use it directly; otherwise use navigator's base URL
        const isFullUrl = path.startsWith('http://') || path.startsWith('https://');
        const fullUrl = isFullUrl ? path : this.navigator.getFullUrl(path);
        safeFireSelect(this.onDatasetSelect, fullUrl);
        this.close();
      }
    };

    // Enter key support
    input.onkeydown = (e) => {
      if (e.key === 'Enter') {
        loadBtn.click();
      }
    };

    input.focus();
  }

  /**
   * Show the browser panel.
   */
  show(): void {
    this.panel.style.display = 'flex';
  }

  /**
   * Hide the browser panel.
   */
  hide(): void {
    this.panel.style.display = 'none';
  }

  /**
   * Close and dispose the browser. Idempotent: a second call is a no-op,
   * so app teardown can call close() defensively without checking
   * whether the user already dismissed the browser.
   */
  close(): void {
    // Use isConnected as the "is this browser still alive" check —
    // matches what `LuxarApp.dispose()` sees if the user already
    // dismissed via Escape/×.
    if (!this.panel.isConnected) return;
    // Bump generation so any in-flight `navigate()` resolving after
    // close discards its result rather than firing `onDatasetSelect`
    // for a path the user backed out of.
    this.navigationGeneration++;
    if (this.onClose) {
      this.onClose();
    }
    this.panel.remove();
  }
}
