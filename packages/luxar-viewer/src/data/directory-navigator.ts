/**
 * Server-agnostic directory navigation service for Luxar datasets.
 *
 * This module provides a flexible system for navigating directories containing
 * Zarr datasets across different server types (WebDAV, S3, nginx, etc.).
 * It uses multiple detection strategies to work with any static file server.
 */

export interface DirectoryEntry {
  name: string;
  path: string;
  type: 'file' | 'directory' | 'zarr';
  size?: number;
  modified?: Date;
}

export interface NavigationResult {
  entries: DirectoryEntry[];
  currentPath: string;
  parentPath?: string;
  strategy: 'webdav' | 'html' | 'index' | 'manual';
  isZarr: boolean;
}

/**
 * Multi-strategy directory navigator that works with various server types.
 */
export class DirectoryNavigator {
  private baseUrl: string;
  private currentPath: string;

  constructor(baseUrl: string) {
    // Ensure URL ends with trailing slash
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
    this.currentPath = '';
  }

  /**
   * Navigate to a specific path and detect its contents.
   */
  async navigate(path: string = ''): Promise<NavigationResult> {
    this.currentPath = path;
    const fullUrl = this.baseUrl + path;

    // Strategy 1: Check if it's a Zarr dataset
    const isZarr = await this.checkIfZarr(fullUrl);
    if (isZarr) {
      return {
        entries: [],
        currentPath: path,
        parentPath: this.getParentPath(path),
        strategy: 'webdav',
        isZarr: true,
      };
    }

    // Strategy 2: Try WebDAV PROPFIND
    const webdavResult = await this.tryWebDAV(fullUrl);
    if (webdavResult) {
      return {
        ...webdavResult,
        currentPath: path,
        parentPath: this.getParentPath(path),
        strategy: 'webdav',
        isZarr: false,
      };
    }

    // Strategy 3: Try parsing HTML directory listing
    const htmlResult = await this.tryHTMLParsing(fullUrl);
    if (htmlResult) {
      return {
        ...htmlResult,
        currentPath: path,
        parentPath: this.getParentPath(path),
        strategy: 'html',
        isZarr: false,
      };
    }

    // Strategy 4: Look for .luxar-index.json
    const indexResult = await this.tryIndexFile(fullUrl);
    if (indexResult) {
      return {
        ...indexResult,
        currentPath: path,
        parentPath: this.getParentPath(path),
        strategy: 'index',
        isZarr: false,
      };
    }

    // Strategy 5: Manual fallback
    return {
      entries: [],
      currentPath: path,
      parentPath: this.getParentPath(path),
      strategy: 'manual',
      isZarr: false,
    };
  }

  /** Default timeout for fetch requests (ms) */
  private static readonly FETCH_TIMEOUT = 10000;

  /**
   * Check if a path is a Zarr dataset by looking for .zgroup file.
   */
  private async checkIfZarr(url: string): Promise<boolean> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DirectoryNavigator.FETCH_TIMEOUT);
    try {
      const response = await fetch(url + '.zgroup', { method: 'HEAD', signal: controller.signal });
      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Try WebDAV PROPFIND method for directory listing.
   */
  private async tryWebDAV(url: string): Promise<{ entries: DirectoryEntry[] } | null> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DirectoryNavigator.FETCH_TIMEOUT);
    try {
      const response = await fetch(url, {
        method: 'PROPFIND',
        headers: {
          Depth: '1',
          'Content-Type': 'application/xml',
        },
        signal: controller.signal,
      });

      if (!response.ok) return null;

      const text = await response.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(text, 'application/xml');

      const entries: DirectoryEntry[] = [];
      const responses = doc.getElementsByTagNameNS('DAV:', 'response');

      for (let i = 0; i < responses.length; i++) {
        const resp = responses[i];
        const href = resp.getElementsByTagNameNS('DAV:', 'href')[0]?.textContent;
        const displayName = resp.getElementsByTagNameNS('DAV:', 'displayname')[0]?.textContent;
        const collection = resp.getElementsByTagNameNS('DAV:', 'collection')[0];

        if (href && href !== url) {
          const name = displayName || href.split('/').filter(Boolean).pop() || '';
          const isDir = !!collection;

          // Check if it's a Zarr directory
          let type: 'file' | 'directory' | 'zarr' = isDir ? 'directory' : 'file';
          if (isDir && name.endsWith('.zarr')) {
            type = 'zarr';
          }

          entries.push({
            name,
            path: this.currentPath ? `${this.currentPath}/${name}` : name,
            type,
          });
        }
      }

      return { entries };
    } catch {
      return null;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Try parsing HTML directory listing (works with nginx, Apache, etc.).
   */
  private async tryHTMLParsing(url: string): Promise<{ entries: DirectoryEntry[] } | null> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DirectoryNavigator.FETCH_TIMEOUT);
    try {
      // First try to get JSON response
      const jsonResponse = await fetch(url, {
        headers: {
          Accept: 'application/json',
        },
        signal: controller.signal,
      });

      if (jsonResponse.ok) {
        const contentType = jsonResponse.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const data = await jsonResponse.json();
          if (data.entries && Array.isArray(data.entries)) {
            // Parse JSON directory listing from luxar serve
            const entries: DirectoryEntry[] = data.entries.map((entry: any) => ({
              name: entry.name,
              path: this.currentPath ? `${this.currentPath}/${entry.name}` : entry.name,
              type: entry.type as 'file' | 'directory' | 'zarr',
              size: entry.size,
            }));
            return { entries };
          }
        }
      }

      // Fall back to HTML parsing
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) return null;

      const html = await response.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');

      const entries: DirectoryEntry[] = [];

      // Try common patterns for directory listings
      // Pattern 1: Links in pre tags (nginx)
      const preLinks = doc.querySelectorAll('pre a');
      if (preLinks.length > 0) {
        preLinks.forEach((link) => {
          const href = link.getAttribute('href');
          const text = link.textContent;
          if (href && text && !text.startsWith('..')) {
            const isDir = href.endsWith('/');
            const name = text.replace(/\/$/, '');

            let type: 'file' | 'directory' | 'zarr' = isDir ? 'directory' : 'file';
            if (isDir && name.endsWith('.zarr')) {
              type = 'zarr';
            }

            entries.push({
              name,
              path: this.currentPath ? `${this.currentPath}/${name}` : name,
              type,
            });
          }
        });
      }

      // Pattern 2: Table rows (Apache, IIS)
      const tableRows = doc.querySelectorAll('tr');
      tableRows.forEach((row) => {
        const link = row.querySelector('a');
        if (link) {
          const href = link.getAttribute('href');
          const text = link.textContent;
          if (href && text && !text.startsWith('..') && !text.startsWith('Parent')) {
            const isDir = href.endsWith('/');
            const name = text.replace(/\/$/, '');

            let type: 'file' | 'directory' | 'zarr' = isDir ? 'directory' : 'file';
            if (isDir && name.endsWith('.zarr')) {
              type = 'zarr';
            }

            entries.push({
              name,
              path: this.currentPath ? `${this.currentPath}/${name}` : name,
              type,
            });
          }
        }
      });

      // Pattern 3: List items (some custom servers)
      const listItems = doc.querySelectorAll('li a');
      listItems.forEach((link) => {
        const href = link.getAttribute('href');
        const text = link.textContent;
        if (href && text && !text.startsWith('..')) {
          const isDir = href.endsWith('/');
          const name = text.replace(/\/$/, '');

          let type: 'file' | 'directory' | 'zarr' = isDir ? 'directory' : 'file';
          if (isDir && name.endsWith('.zarr')) {
            type = 'zarr';
          }

          entries.push({
            name,
            path: this.currentPath ? `${this.currentPath}/${name}` : name,
            type,
          });
        }
      });

      // Deduplicate entries from multiple HTML parsing strategies
      const seen = new Set<string>();
      const uniqueEntries = entries.filter((entry) => {
        if (seen.has(entry.name)) return false;
        seen.add(entry.name);
        return true;
      });

      return uniqueEntries.length > 0 ? { entries: uniqueEntries } : null;
    } catch {
      return null;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Try loading a .luxar-index.json file with directory contents.
   */
  private async tryIndexFile(url: string): Promise<{ entries: DirectoryEntry[] } | null> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), DirectoryNavigator.FETCH_TIMEOUT);
    try {
      const response = await fetch(url + '.luxar-index.json', { signal: controller.signal });
      if (!response.ok) return null;

      const index = await response.json();
      if (!index.entries || !Array.isArray(index.entries)) return null;

      const entries: DirectoryEntry[] = index.entries.map((entry: any) => ({
        name: entry.name,
        path: this.currentPath ? `${this.currentPath}/${entry.name}` : entry.name,
        type:
          entry.type ||
          (entry.name.endsWith('.zarr') ? 'zarr' : entry.isDirectory ? 'directory' : 'file'),
        size: entry.size,
        modified: entry.modified ? new Date(entry.modified) : undefined,
      }));

      return { entries };
    } catch {
      return null;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Get parent path from current path.
   */
  private getParentPath(path: string): string | undefined {
    if (!path) return undefined;
    const parts = path.split('/').filter(Boolean);
    if (parts.length === 0) return undefined;
    parts.pop();
    return parts.join('/');
  }

  /**
   * Get the full URL for a given path.
   */
  getFullUrl(path: string): string {
    return this.baseUrl + path;
  }

  /**
   * Check if we can list directories at all (for feature detection).
   */
  async canListDirectories(): Promise<boolean> {
    const result = await this.navigate('');
    return result.strategy !== 'manual' || result.entries.length > 0;
  }
}
