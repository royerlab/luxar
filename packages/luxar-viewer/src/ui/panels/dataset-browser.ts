/**
 * Dataset browser UI panel for navigating and selecting Zarr datasets.
 *
 * Provides a user-friendly interface for browsing directories and loading
 * Zarr datasets from various server types.
 */

import { DirectoryNavigator, type DirectoryEntry } from '../../data';
import { escapeHtml } from '../../utils/escape-html';
import { extractBaseUrl, extractPath } from './dataset-url-utils';

export interface DatasetBrowserConfig {
  container: HTMLElement;
  /** Callback when a dataset is selected. Receives the full URL (not just the path). */
  onDatasetSelect: (fullUrl: string) => void;
  onClose?: () => void;
  /** Currently loaded dataset URL, used to determine the initial directory. */
  currentSrc?: string;
  /** Fallback origin for relative path resolution (defaults to `window.location.origin`). */
  origin?: string;
}

/**
 * Interactive dataset browser panel.
 */
export class DatasetBrowser {
  private container: HTMLElement;
  private panel: HTMLElement;
  private navigator: DirectoryNavigator;
  private onDatasetSelect: (fullUrl: string) => void;
  private onClose?: () => void;
  private currentDataset?: string;

  /** Origin used for relative path resolution; captured at construction. */
  private readonly origin: string;

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
    panel.className = 'luxar-dataset-browser dataset-browser'; // luxar-dataset-browser for styling, dataset-browser for E2E tests

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
        <div>
          <strong class="luxar-dataset-browser__banner-title">Luxar</strong> - Interactive Scientific Data Visualization
          <span class="luxar-dataset-browser__banner-description">•</span>
          <span class="luxar-dataset-browser__banner-description">Browse for <code class="luxar-dataset-browser__banner-code">.zarr</code> or enter path manually</span>
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
    panel.appendChild(content);
    panel.appendChild(statusBar);

    this.container.appendChild(panel);
    return panel;
  }

  /**
   * Navigate to a path and update the UI.
   */
  private async navigate(path: string): Promise<void> {
    const content = this.panel.querySelector('#luxar-dataset-browser-content') as HTMLElement;
    const statusBar = this.panel.querySelector('#luxar-dataset-browser-status') as HTMLElement;

    // Show loading state
    content.innerHTML = '<div class="luxar-dataset-browser__loading">Loading...</div>';
    statusBar.textContent = 'Fetching directory contents...';

    try {
      const result = await this.navigator.navigate(path);
      // Store result for future use if needed

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
        this.onDatasetSelect(this.navigator.getFullUrl(result.currentPath));
        this.close();
        return;
      }

      // Display directory contents
      this.displayEntries(result.entries);

      // Update status
      const strategyText = {
        webdav: 'WebDAV',
        html: 'HTML parsing',
        index: 'Index file',
        manual: 'Manual entry',
      }[result.strategy];

      statusBar.innerHTML = `
        <span>${result.entries.length} items</span>
        <span>Detection: ${strategyText}</span>
      `;

      // Handle manual fallback
      if (result.strategy === 'manual' && result.entries.length === 0) {
        this.showManualEntry();
      }
    } catch (error) {
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
   * Display directory entries.
   */
  private displayEntries(entries: DirectoryEntry[]): void {
    const content = this.panel.querySelector('#luxar-dataset-browser-content') as HTMLElement;
    content.innerHTML = '';

    if (entries.length === 0) {
      content.innerHTML = '<div class="luxar-dataset-browser__empty">Empty directory</div>';
      return;
    }

    // Sort entries: directories first, then files
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
          this.onDatasetSelect(this.navigator.getFullUrl(entry.path));
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
          placeholder="e.g., datasets/example.zarr"
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
        this.onDatasetSelect(fullUrl);
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
    if (this.onClose) {
      this.onClose();
    }
    this.panel.remove();
  }
}
