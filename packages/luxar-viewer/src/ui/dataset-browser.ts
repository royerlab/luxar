/**
 * Dataset browser UI panel for navigating and selecting Zarr datasets.
 *
 * Provides a user-friendly interface for browsing directories and loading
 * Zarr datasets from various server types. Styled after the viewer's
 * "quiet instrument" design language (control rail / help overlay): glass
 * surface, tick-motif header, stroke icons, and the blue highlight accent
 * for interactive states.
 */

import { isZippedZarrStoreUrl } from '../data/zip/entries';
import { DirectoryNavigator, type DirectoryEntry } from '../data';
import { escapeHtml } from '../utils/escape-html';
import { extractBaseUrl, extractPath } from './dataset-browser/url-utils';
import { BROWSER_ICONS } from './dataset-browser/icons';
import { trapFocus } from './help-overlay/focus-trap';
import { installTypeToFilter } from './help-overlay/type-to-filter';
// Aliased: the constructor parameter is also called `config`
// ({@link DatasetBrowserConfig}), and a bare import would be shadowed by it.
import { config as viewerConfig } from '../config';
import { log, Modules } from '../utils/log';
import { showToast } from './toast';

/**
 * Wrap an `onDatasetSelect` invocation so a Promise-returning
 * callback (the production path is async — `LuxarApp.loadDataset`)
 * doesn't surface as an unhandled rejection when the browser closes
 * synchronously after firing it.
 *
 * @returns `false` when the callback synchronously refuses the selection.
 */
function safeFireSelect(cb: (url: string) => void | false | Promise<void>, url: string): boolean {
  try {
    const result = cb(url);
    if (result === false) return false;
    if (result && typeof (result as Promise<void>).then === 'function') {
      (result as Promise<void>).catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        log.warning(Modules.UI, `Dataset load failed: ${msg}`);
        showToast(`Failed to load dataset: ${msg}`);
      });
    }
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log.warning(Modules.UI, `Dataset selection threw: ${msg}`);
    showToast(`Failed to load dataset: ${msg}`);
    return true;
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
   * unhandled rejection), and closes immediately. Return `false`
   * synchronously to keep the browser open without treating the selection
   * as a failure.
   */
  onDatasetSelect: (fullUrl: string) => void | false | Promise<void>;
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
 *
 * A dimming scrim is mounted behind the panel (click-to-close), the listing is
 * arrow-key navigable, and the breadcrumb row hosts an inline "enter path
 * manually" editor so a path/URL can always be typed — not only when the
 * server falls back to the `manual` detection strategy.
 *
 * Initial focus goes to the panel container rather than the search field, so
 * the `O` shortcut still toggles the browser shut (a focused text field trips
 * `InputHandler`'s typing guard — issue #1922); the first printable keystroke
 * is forwarded into the search field so typing still filters immediately.
 */
export class DatasetBrowser {
  private container: HTMLElement;
  private panel: HTMLElement;
  private scrim: HTMLElement;
  private navigator: DirectoryNavigator;
  private onDatasetSelect: (fullUrl: string) => void | false | Promise<void>;
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

  /** Path of the current (last successfully rendered) directory — seeds the inline path editor. */
  private currentPath = '';

  /** Path of the most recent `navigate()` attempt — target for the error-state Retry button. */
  private lastAttemptedPath = '';

  /** Teardown for the modal focus trap (Tab must not escape behind the scrim). */
  private untrapFocus?: () => void;

  /** Teardown for the container-focus + type-to-filter forwarder. */
  private untypeToFilter?: () => void;

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
    } else if (src.includes('.zarr/') || src.endsWith('.zarr') || isZippedZarrStoreUrl(src)) {
      // We're inside or at a zarr dataset - navigate to parent directory
      try {
        const parsed = new URL(src);
        const pathname = parsed.pathname;

        // Find the .zarr part and go to parent directory
        const zarrIndex = pathname.toLowerCase().lastIndexOf('.zarr');
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
    this.scrim = this.createScrim();
    this.panel = this.createPanel();
    // Modal focus containment: the panel is aria-modal with a scrim, so Tab
    // must cycle inside it rather than escaping to the rail behind.
    // `autoFocusFirst: false` — initial focus belongs to the panel container
    // (see installTypeToFilterOnPanel), not to the close button.
    this.untrapFocus = trapFocus(this.panel, { autoFocusFirst: false });
    this.untypeToFilter = this.installTypeToFilterOnPanel();

    // Start navigation at the determined path
    this.navigateSafely(initialPath);
  }

  /**
   * Park focus on the panel container and forward the first printable
   * keystroke into the search field.
   *
   * The browser used to autofocus the search field on the first successful
   * listing, which trips `InputHandler`'s typing guard and made `O` one-way —
   * it opened the browser but the second `O` was swallowed as typing
   * (issue #1922). Focus now stays on the (non-typing) panel container, so
   * `O` toggles, while typing still filters from the very first key. `O`
   * itself is passed through to the global binding, so it cannot be the
   * FIRST character of a filter query (it types normally once the field has
   * focus).
   *
   * The resolver only ever names the SEARCH field, and only while the search
   * bar is actually shown: it returns `null` while that bar is hidden
   * (loading, error, empty directory, manual-entry fallback), and the
   * keystroke is then contained by the modal. The manual-entry `#manual-path`
   * field is deliberately NOT a resolver target — it is a URL entry field,
   * not a filter, so stray keystrokes should not be routed into it. The user
   * clicks or Tabs into that field, which is a deliberate act, and from there
   * the ordinary typing guard applies exactly as it does for every other text
   * field in the app.
   *
   * `passthroughKeys` lists every shortcut this panel advertises while it is
   * open: `O` (its own toggle) and `H` (the `H Help` chip in the welcome
   * banner). Containment would otherwise make that chip a lie.
   *
   * `resolveFirstItem` restores the "`ArrowDown` enters the listing"
   * affordance the search field's own handler provides: with focus parked on
   * the container, that handler never sees the key.
   */
  private installTypeToFilterOnPanel(): () => void {
    return installTypeToFilter(
      this.panel,
      () => {
        const bar = this.panel.querySelector<HTMLElement>('#luxar-dataset-browser-search-bar');
        if (!bar || bar.style.display === 'none') return null;
        return this.panel.querySelector<HTMLInputElement>('#luxar-dataset-browser-search');
      },
      {
        passthroughKeys: [
          viewerConfig.input.keyboard.shortcuts.toggleDatasetBrowser,
          viewerConfig.input.keyboard.shortcuts.toggleHelp,
        ],
        resolveFirstItem: () =>
          this.panel.querySelector<HTMLElement>('.luxar-dataset-browser__file-item'),
      }
    );
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
   * Create the dimming scrim mounted behind the panel. Clicking it closes
   * the browser (standard modal affordance); it fades in with the panel
   * and is removed together with it in `close()`.
   */
  private createScrim(): HTMLElement {
    const scrim = document.createElement('div');
    scrim.className = 'luxar-dataset-browser-scrim';
    scrim.setAttribute('aria-hidden', 'true');
    scrim.onclick = () => this.close();
    this.container.appendChild(scrim);
    return scrim;
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

    // Header — tick-motif micro-header (title row) + quiet tagline row.
    const header = document.createElement('div');
    header.className = 'luxar-dataset-browser__header';

    const title = document.createElement('h2');
    title.id = 'luxar-dataset-browser-title';
    title.className = 'luxar-dataset-browser__title';
    title.textContent = 'Select Dataset';

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'luxar-dataset-browser__close-btn';
    closeBtn.innerHTML = BROWSER_ICONS.close;
    closeBtn.title = 'Close (Escape)';
    closeBtn.setAttribute('aria-label', 'Close dataset browser');
    closeBtn.onclick = () => this.close();

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Compact tagline row with essential guidance (kept quiet — no accent
    // wash). The element id + text content are pinned by first-time-UX E2E.
    const welcomeBanner = document.createElement('div');
    welcomeBanner.id = 'luxar-dataset-browser-welcome';
    welcomeBanner.className = 'luxar-dataset-browser__banner';

    welcomeBanner.innerHTML = `
      <div class="luxar-dataset-browser__banner-content">
        <div class="luxar-dataset-browser__banner-text">
          <div class="luxar-dataset-browser__banner-tagline"><strong class="luxar-dataset-browser__banner-title">Luxar</strong> — Interactive Scientific Data Visualization</div>
          <div class="luxar-dataset-browser__banner-description">Browse for <code class="luxar-dataset-browser__banner-code">.zarr</code> or enter path manually</div>
        </div>
        <div class="luxar-dataset-browser__banner-help">
          <kbd class="luxar-dataset-browser__banner-kbd">H</kbd> Help
        </div>
      </div>
    `;

    // Breadcrumb navigation (+ inline path editor toggle at its right edge)
    const breadcrumb = document.createElement('div');
    breadcrumb.id = 'luxar-dataset-browser-breadcrumb';
    breadcrumb.className = 'luxar-dataset-browser__breadcrumb';

    // Search / filter bar (filters the current directory listing live).
    // Hidden whenever there are no entries to filter (loading, error,
    // empty directory, or the manual-entry fallback) — see setSearchVisible.
    const search = document.createElement('div');
    search.id = 'luxar-dataset-browser-search-bar';
    search.className = 'luxar-dataset-browser__search';
    const searchIcon = document.createElement('span');
    searchIcon.className = 'luxar-dataset-browser__search-icon';
    searchIcon.innerHTML = BROWSER_ICONS.search;
    searchIcon.setAttribute('aria-hidden', 'true');
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
    // ArrowDown hands focus from the filter to the first row of the listing.
    searchInput.onkeydown = (e) => {
      if (e.key === 'ArrowDown') {
        const first = this.panel.querySelector<HTMLElement>('.luxar-dataset-browser__file-item');
        if (first) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    search.appendChild(searchIcon);
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
    this.lastAttemptedPath = path;

    // Show loading state (skeleton rows). Hide the search bar while loading;
    // renderEntries re-shows it once we have a non-empty listing to filter.
    content.innerHTML = `
      <div class="luxar-dataset-browser__loading" role="status" aria-label="Loading directory contents">
        <div class="luxar-dataset-browser__skeleton-row"></div>
        <div class="luxar-dataset-browser__skeleton-row"></div>
        <div class="luxar-dataset-browser__skeleton-row"></div>
      </div>
    `;
    this.parkFocusIfUnclaimed();
    statusBar.textContent = 'Fetching directory contents...';
    this.setSearchVisible(false);

    try {
      const result = await this.navigator.navigate(path);
      // Bail if a newer navigate has started while we were awaiting.
      // The newer call already wrote its loading indicator and is
      // responsible for the next render.
      if (this.navigationGeneration !== myGeneration) return;

      // Update breadcrumb
      this.currentPath = result.currentPath;
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
        if (safeFireSelect(this.onDatasetSelect, this.navigator.getFullUrl(result.currentPath))) {
          this.close();
        } else {
          this.navigateSafely(result.parentPath ?? '');
        }
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
          <span class="luxar-dataset-browser__error-icon" aria-hidden="true">${BROWSER_ICONS.alert}</span>
          <p class="luxar-dataset-browser__error-title">Failed to load directory</p>
          <p class="luxar-dataset-browser__error-details">${escapeHtml(String(error))}</p>
          <button type="button" class="luxar-dataset-browser__error-retry">Retry</button>
        </div>
      `;
      const retry = content.querySelector(
        '.luxar-dataset-browser__error-retry'
      ) as HTMLButtonElement | null;
      if (retry) retry.onclick = () => this.navigate(this.lastAttemptedPath);
      statusBar.textContent = 'Error loading directory';
    }
  }

  private navigateSafely(path: string): void {
    this.navigate(path).catch((error: unknown) => {
      log.warning(Modules.UI, 'Dataset navigation failed', error);
    });
  }

  /**
   * Update breadcrumb navigation. Rebuilds the crumb trail and the
   * "enter path manually" toggle (which swaps the row for an inline editor).
   */
  private updateBreadcrumb(currentPath: string): void {
    const breadcrumb = this.panel.querySelector('#luxar-dataset-browser-breadcrumb') as HTMLElement;
    breadcrumb.innerHTML = '';

    const crumbs = document.createElement('div');
    crumbs.className = 'luxar-dataset-browser__breadcrumb-trail';

    // Root link — `<button type="button">` so the breadcrumb is keyboard-
    // focusable (anchors without href aren't) and Enter/Space activate
    // natively. CSS class names are unchanged so existing styling/snapshot
    // tests stay intact.
    const rootLink = document.createElement('button');
    rootLink.type = 'button';
    rootLink.className = 'luxar-dataset-browser__breadcrumb-link';
    rootLink.innerHTML = `<span class="luxar-dataset-browser__breadcrumb-home" aria-hidden="true">${BROWSER_ICONS.home}</span>Root`;
    rootLink.onclick = () => this.navigate('');
    crumbs.appendChild(rootLink);

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
        crumbs.appendChild(sep);

        accumulated += (accumulated ? '/' : '') + part;
        const pathToNavigate = accumulated;

        if (index === parts.length - 1) {
          // Current location (not clickable)
          const current = document.createElement('span');
          current.className = 'luxar-dataset-browser__breadcrumb-current';
          current.textContent = part;
          current.setAttribute('aria-current', 'location');
          crumbs.appendChild(current);
        } else {
          // Clickable parent — same a11y rationale as the Root button above.
          const link = document.createElement('button');
          link.type = 'button';
          link.className = 'luxar-dataset-browser__breadcrumb-link';
          link.textContent = part;
          link.onclick = () => this.navigate(pathToNavigate);
          crumbs.appendChild(link);
        }
      });
    }

    breadcrumb.appendChild(crumbs);

    // Path-editor toggle: manual entry is always one click away, not only
    // when the server's listing detection falls back to `manual`.
    const editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.id = 'luxar-dataset-browser-path-edit';
    editBtn.className = 'luxar-dataset-browser__path-edit';
    editBtn.innerHTML = BROWSER_ICONS.edit;
    editBtn.title = 'Enter path manually';
    editBtn.setAttribute('aria-label', 'Enter dataset path manually');
    editBtn.onclick = () => this.openPathEditor();
    breadcrumb.appendChild(editBtn);

    this.parkFocusIfUnclaimed();
  }

  /** Keep modal keyboard handling active after a focused child is replaced. */
  private parkFocusIfUnclaimed(): void {
    if (!document.activeElement || document.activeElement === document.body) {
      this.panel.focus();
    }
  }

  /**
   * Swap the breadcrumb row for an inline path editor (mono input seeded
   * with the current path). Enter commits — a value containing `.zarr`
   * (or a full URL) is selected as a dataset, anything else is navigated
   * to; Escape or blur restores the crumb trail.
   */
  private openPathEditor(): void {
    const breadcrumb = this.panel.querySelector('#luxar-dataset-browser-breadcrumb') as HTMLElement;
    breadcrumb.innerHTML = '';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'luxar-dataset-browser__path-input';
    input.value = this.currentPath;
    input.placeholder = 'path/to/dataset.zarr or full URL';
    input.setAttribute('aria-label', 'Dataset path or URL');
    input.autocomplete = 'off';
    input.spellcheck = false;

    const cancel = (): void => this.updateBreadcrumb(this.currentPath);

    input.onkeydown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.commitPathEditor(input.value);
      } else if (e.key === 'Escape') {
        // Keep Escape local: restore the crumbs without closing the dialog.
        e.preventDefault();
        e.stopPropagation();
        cancel();
      }
    };
    input.onblur = () => {
      // Restore the trail when focus leaves without committing. Deferred a
      // tick so an Enter-commit close doesn't race the blur restore.
      setTimeout(() => {
        if (input.isConnected) cancel();
      }, 0);
    };

    breadcrumb.appendChild(input);
    input.focus();
    input.select();
  }

  /**
   * Commit the inline path editor's value (see {@link openPathEditor}).
   *
   * Full URLs are selected as-is (parity with the manual-entry form). A
   * relative path is selected only when it deliberately names a `.zarr`
   * directory — i.e. its last path segment ends with `.zarr` (trailing
   * slashes ignored), OR deliberately names a zipped store (`foo.zarr.zip`,
   * read in place over range requests). A mere `.zarr` SUBSTRING
   * (`archives.zarr-backup`) is not a dataset; those navigate instead, and
   * `navigate()` still auto-selects if the server reports a real zarr.
   */
  private commitPathEditor(raw: string): void {
    const path = raw.trim();
    if (!path) {
      this.updateBreadcrumb(this.currentPath);
      return;
    }
    const isFullUrl = path.startsWith('http://') || path.startsWith('https://');
    const trimmed = path.replace(/\/+$/, '');
    // A zipped store is a dataset too: the viewer reads it in place over range
    // requests. Selecting it is right; navigating INTO it would be meaningless
    // (an archive has no listable children).
    const endsWithZarr = trimmed.endsWith('.zarr') || isZippedZarrStoreUrl(trimmed);
    if (isFullUrl || endsWithZarr) {
      const fullUrl = isFullUrl ? path : this.navigator.getFullUrl(path);
      if (safeFireSelect(this.onDatasetSelect, fullUrl)) this.close();
      return;
    }
    this.navigateSafely(path);
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
      <span class="luxar-dataset-browser__status-detection" title="Directory listing via ${escapeHtml(strategyText)}">${escapeHtml(strategyText)}</span>
    `;

    // Only offer the filter when there's an actual listing to narrow.
    // Stays visible in the no-matches case (total > 0) so the user can
    // edit their query.
    this.setSearchVisible(total > 0);

    if (total === 0) {
      content.innerHTML = `
        <div class="luxar-dataset-browser__empty">
          <span class="luxar-dataset-browser__empty-icon" aria-hidden="true">${BROWSER_ICONS.folder}</span>
          <p class="luxar-dataset-browser__empty-title">No datasets here</p>
          <p class="luxar-dataset-browser__empty-hint">This directory is empty — browse elsewhere or enter a path.</p>
          <button type="button" class="luxar-dataset-browser__empty-action">Enter path…</button>
        </div>
      `;
      const action = content.querySelector(
        '.luxar-dataset-browser__empty-action'
      ) as HTMLButtonElement | null;
      if (action) action.onclick = () => this.openPathEditor();
      return;
    }

    if (entries.length === 0) {
      content.innerHTML = `<div class="luxar-dataset-browser__empty"><p class="luxar-dataset-browser__empty-title">No matches for “${escapeHtml(this.filterText.trim())}”</p></div>`;
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

    // Arrow-key navigation between rows (the rows are plain focusables, so
    // Tab order alone would make long listings tedious).
    list.onkeydown = (e) => {
      const target = e.target as HTMLElement;
      if (!target.classList.contains('luxar-dataset-browser__file-item')) return;
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp' && e.key !== 'Home' && e.key !== 'End') {
        return;
      }
      // Swallow the default for every handled key — even at the list
      // boundaries where focus doesn't move — so ArrowUp on the first row
      // (or ArrowDown on the last) doesn't scroll the content area.
      e.preventDefault();
      const items = Array.from(
        list.querySelectorAll<HTMLElement>('.luxar-dataset-browser__file-item')
      );
      const index = items.indexOf(target);
      let next = index;
      if (e.key === 'ArrowDown') next = Math.min(index + 1, items.length - 1);
      else if (e.key === 'ArrowUp') next = Math.max(index - 1, 0);
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = items.length - 1;
      if (next !== index) items[next].focus();
    };

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

      // Icon — stroke SVG in the rail's register (was emoji).
      const icon = document.createElement('span');
      icon.className = 'luxar-dataset-browser__file-icon';
      if (entry.type === 'zarr') {
        icon.classList.add('luxar-dataset-browser__file-icon--zarr');
        icon.innerHTML = BROWSER_ICONS.zarr;
        icon.title = 'Zarr Dataset';
      } else if (entry.type === 'directory') {
        icon.innerHTML = BROWSER_ICONS.folder;
        icon.title = 'Directory';
      } else {
        icon.innerHTML = BROWSER_ICONS.file;
        icon.title = 'File';
      }

      // Name
      const name = document.createElement('span');
      name.className = 'luxar-dataset-browser__file-name';
      name.textContent = entry.name;

      item.appendChild(icon);
      item.appendChild(name);

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

        item.appendChild(badgeContainer);
      } else if (entry.type === 'directory') {
        // Quiet "navigates deeper" affordance.
        const chevron = document.createElement('span');
        chevron.className = 'luxar-dataset-browser__chevron';
        chevron.innerHTML = BROWSER_ICONS.chevron;
        chevron.setAttribute('aria-hidden', 'true');
        item.appendChild(chevron);
      }

      // Click handler
      const activate = (): void => {
        if (entry.type === 'zarr') {
          // Pass full URL to preserve directory context
          if (safeFireSelect(this.onDatasetSelect, this.navigator.getFullUrl(entry.path))) {
            this.close();
          }
        } else if (entry.type === 'directory') {
          this.navigateSafely(entry.path);
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

    // No autofocus here, deliberately. The listing arrives asynchronously and
    // grabbing focus for the search field would (a) yank focus from whatever
    // the user reached in the meantime and (b) trip InputHandler's typing
    // guard, making `O` one-way (issue #1922). Typing narrows the list from
    // the first key anyway — see installTypeToFilterOnPanel.
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
        if (safeFireSelect(this.onDatasetSelect, fullUrl)) this.close();
      }
    };

    // Enter key support
    input.onkeydown = (e) => {
      if (e.key === 'Enter') {
        loadBtn.click();
      }
    };

    // Deliberately NOT focused: a focused text field trips `InputHandler`'s
    // typing guard, which would make `O` one-way again — the exact bug
    // issue #1922 fixes, and this fallback is reachable in production (any
    // host that serves an `index.html` instead of a listing). Focus stays on
    // the panel container; the user clicks or Tabs into this field to type a
    // path, and from then on it behaves like every other text field in the
    // app. It is NOT wired into type-to-filter either: this is a URL, not a
    // filter query, so stray keystrokes should not be routed into it.
  }

  /**
   * Show the browser panel.
   */
  show(): void {
    this.scrim.style.display = '';
    this.panel.style.display = 'flex';
    // Re-arm what hide() released (no-op when already armed).
    this.untrapFocus ??= trapFocus(this.panel, { autoFocusFirst: false });
    this.untypeToFilter ??= this.installTypeToFilterOnPanel();
    // Re-park focus unconditionally: `??=` skips the re-install (and with it
    // the forwarder's own container focus) whenever the panel was never
    // hidden, so a re-shown panel would otherwise start wherever focus
    // happened to be. It must start fresh on the container — that is what
    // keeps `O` a toggle and type-to-filter armed.
    this.panel.focus({ preventScroll: true });
  }

  /**
   * Hide the browser panel.
   */
  hide(): void {
    // Release BEFORE hiding: a hidden modal must not keep Tab hostage or keep
    // forwarding keystrokes, and the trap's cleanup hands focus back to the
    // pre-open element. Forwarder first, so it is gone before the trap moves
    // focus out.
    this.untypeToFilter?.();
    this.untypeToFilter = undefined;
    this.untrapFocus?.();
    this.untrapFocus = undefined;
    this.scrim.style.display = 'none';
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
    this.untypeToFilter?.();
    this.untypeToFilter = undefined;
    this.untrapFocus?.();
    this.untrapFocus = undefined;
    if (this.onClose) {
      this.onClose();
    }
    this.scrim.remove();
    this.panel.remove();
  }
}
