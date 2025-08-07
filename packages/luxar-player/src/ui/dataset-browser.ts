/**
 * Dataset browser UI panel for navigating and selecting Zarr datasets.
 * 
 * Provides a user-friendly interface for browsing directories and loading
 * Zarr datasets from various server types.
 */

import { DirectoryNavigator, DirectoryEntry } from '../data/directory-navigator';

export interface DatasetBrowserConfig {
  container: HTMLElement;
  onDatasetSelect: (path: string) => void;
  onClose?: () => void;
}

/**
 * Interactive dataset browser panel.
 */
export class DatasetBrowser {
  private container: HTMLElement;
  private panel: HTMLElement;
  private navigator: DirectoryNavigator;
  private onDatasetSelect: (path: string) => void;
  private onClose?: () => void;
  private currentDataset?: string;
  
  constructor(config: DatasetBrowserConfig) {
    this.container = config.container;
    this.onDatasetSelect = config.onDatasetSelect;
    this.onClose = config.onClose;
    
    // Parse base URL from current location
    const params = new URLSearchParams(window.location.search);
    const src = params.get('src') || '';
    
    // Check if we're currently inside a zarr dataset
    let baseUrl: string;
    let initialPath: string;
    
    if (src && (src.includes('.zarr/') || src.endsWith('.zarr'))) {
      // We're inside a zarr dataset - navigate to parent directory
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
        initialPath = '';  // Start at parent directory
        
        // Store the current dataset for highlighting
        this.currentDataset = datasetName;
      } else {
        baseUrl = this.extractBaseUrl(src);
        initialPath = this.extractPath(src);
      }
    } else {
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
    if (!url) return window.location.origin + '/';
    
    try {
      const parsed = new URL(url);
      // If it ends with .zarr, go up one level
      let pathname = parsed.pathname;
      if (pathname.endsWith('.zarr') || pathname.endsWith('.zarr/')) {
        const parts = pathname.split('/').filter(Boolean);
        parts.pop();
        pathname = '/' + parts.join('/') + '/';
      }
      return parsed.origin + pathname;
    } catch {
      return url;
    }
  }
  
  /**
   * Extract relative path from a full URL.
   */
  private extractPath(url: string): string {
    if (!url) return '';
    
    try {
      const parsed = new URL(url);
      const pathname = parsed.pathname;
      
      // Extract just the dataset name if it's a .zarr
      if (pathname.includes('.zarr')) {
        const parts = pathname.split('/').filter(Boolean);
        const zarrIndex = parts.findIndex(p => p.endsWith('.zarr'));
        if (zarrIndex >= 0) {
          return parts.slice(0, zarrIndex + 1).join('/');
        }
      }
      
      return '';
    } catch {
      return '';
    }
  }
  
  /**
   * Create the browser panel UI.
   */
  private createPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.id = 'dataset-browser';
    panel.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 600px;
      max-width: 90vw;
      height: 500px;
      max-height: 80vh;
      background: rgba(30, 30, 30, 0.95);
      backdrop-filter: blur(10px);
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
      color: #e0e0e0;
      z-index: 1000;
    `;
    
    // Header
    const header = document.createElement('div');
    header.style.cssText = `
      padding: 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;
    
    const title = document.createElement('h2');
    title.textContent = 'Select Dataset';
    title.style.cssText = `
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    `;
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.cssText = `
      background: none;
      border: none;
      color: #999;
      font-size: 24px;
      cursor: pointer;
      padding: 0;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
    `;
    closeBtn.onmouseover = () => closeBtn.style.color = '#fff';
    closeBtn.onmouseout = () => closeBtn.style.color = '#999';
    closeBtn.onclick = () => this.close();
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    
    // Breadcrumb navigation
    const breadcrumb = document.createElement('div');
    breadcrumb.id = 'breadcrumb';
    breadcrumb.style.cssText = `
      padding: 10px 20px;
      background: rgba(255, 255, 255, 0.05);
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
      overflow-x: auto;
    `;
    
    // Content area
    const content = document.createElement('div');
    content.id = 'browser-content';
    content.style.cssText = `
      flex: 1;
      overflow-y: auto;
      padding: 20px;
    `;
    
    // Status bar
    const statusBar = document.createElement('div');
    statusBar.id = 'browser-status';
    statusBar.style.cssText = `
      padding: 10px 20px;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      font-size: 12px;
      color: #888;
      display: flex;
      justify-content: space-between;
    `;
    
    panel.appendChild(header);
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
    const content = this.panel.querySelector('#browser-content') as HTMLElement;
    const statusBar = this.panel.querySelector('#browser-status') as HTMLElement;
    
    // Show loading state
    content.innerHTML = '<div style="text-align: center; padding: 40px;">Loading...</div>';
    statusBar.textContent = 'Fetching directory contents...';
    
    try {
      const result = await this.navigator.navigate(path);
      // Store result for future use if needed
      
      // Update breadcrumb
      this.updateBreadcrumb(result.currentPath);
      
      // If it's a Zarr dataset, load it directly
      if (result.isZarr) {
        this.onDatasetSelect(result.currentPath);
        this.close();
        return;
      }
      
      // Display directory contents
      this.displayEntries(result.entries);
      
      // Update status
      const strategyText = {
        'webdav': 'WebDAV',
        'html': 'HTML parsing',
        'index': 'Index file',
        'manual': 'Manual entry'
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
        <div style="text-align: center; padding: 40px; color: #f44336;">
          <p>Failed to load directory</p>
          <p style="font-size: 12px; margin-top: 10px;">${error}</p>
        </div>
      `;
      statusBar.textContent = 'Error loading directory';
    }
  }
  
  /**
   * Update breadcrumb navigation.
   */
  private updateBreadcrumb(currentPath: string): void {
    const breadcrumb = this.panel.querySelector('#breadcrumb') as HTMLElement;
    breadcrumb.innerHTML = '';
    
    // Root link
    const rootLink = document.createElement('a');
    rootLink.textContent = 'Root';
    rootLink.style.cssText = `
      color: #4CAF50;
      text-decoration: none;
      cursor: pointer;
    `;
    rootLink.onclick = () => this.navigate('');
    breadcrumb.appendChild(rootLink);
    
    // Path segments
    if (currentPath) {
      const parts = currentPath.split('/').filter(Boolean);
      let accumulated = '';
      
      parts.forEach((part, index) => {
        // Separator
        const sep = document.createElement('span');
        sep.textContent = '›';
        sep.style.color = '#666';
        breadcrumb.appendChild(sep);
        
        accumulated += (accumulated ? '/' : '') + part;
        const pathToNavigate = accumulated;
        
        if (index === parts.length - 1) {
          // Current location (not clickable)
          const current = document.createElement('span');
          current.textContent = part;
          breadcrumb.appendChild(current);
        } else {
          // Clickable parent
          const link = document.createElement('a');
          link.textContent = part;
          link.style.cssText = `
            color: #4CAF50;
            text-decoration: none;
            cursor: pointer;
          `;
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
    const content = this.panel.querySelector('#browser-content') as HTMLElement;
    content.innerHTML = '';
    
    if (entries.length === 0) {
      content.innerHTML = '<div style="text-align: center; padding: 40px; color: #888;">Empty directory</div>';
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
    list.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 2px;
    `;
    
    sorted.forEach(entry => {
      const item = document.createElement('div');
      
      // Check if this is the currently selected dataset
      const isCurrentDataset = this.currentDataset && entry.name === this.currentDataset;
      
      item.style.cssText = `
        padding: 12px 16px;
        background: ${isCurrentDataset ? 'rgba(76, 175, 80, 0.15)' : 'rgba(255, 255, 255, 0.05)'};
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: background 0.2s;
        ${isCurrentDataset ? 'border: 1px solid rgba(76, 175, 80, 0.4);' : ''}
      `;
      
      item.onmouseover = () => {
        item.style.background = 'rgba(76, 175, 80, 0.2)';
      };
      item.onmouseout = () => {
        item.style.background = isCurrentDataset ? 'rgba(76, 175, 80, 0.15)' : 'rgba(255, 255, 255, 0.05)';
      };
      
      // Icon
      const icon = document.createElement('span');
      icon.style.fontSize = '18px';
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
      name.textContent = entry.name;
      name.style.flex = '1';
      
      // Type badge and current indicator
      if (entry.type === 'zarr') {
        const badgeContainer = document.createElement('div');
        badgeContainer.style.cssText = `
          display: flex;
          gap: 6px;
          align-items: center;
        `;
        
        // ZARR badge
        const badge = document.createElement('span');
        badge.textContent = 'ZARR';
        badge.style.cssText = `
          background: #4CAF50;
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 10px;
          font-weight: 600;
        `;
        badgeContainer.appendChild(badge);
        
        // Current dataset indicator
        if (isCurrentDataset) {
          const currentBadge = document.createElement('span');
          currentBadge.textContent = 'LOADED';
          currentBadge.style.cssText = `
            background: #2196F3;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
          `;
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
      item.onclick = () => {
        if (entry.type === 'zarr') {
          this.onDatasetSelect(entry.path);
          this.close();
        } else if (entry.type === 'directory') {
          this.navigate(entry.path);
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
    const content = this.panel.querySelector('#browser-content') as HTMLElement;
    
    content.innerHTML = `
      <div style="text-align: center; padding: 40px;">
        <p style="margin-bottom: 20px;">Directory listing not available. Enter dataset path manually:</p>
        <input 
          type="text" 
          id="manual-path" 
          placeholder="e.g., datasets/example.zarr"
          style="
            width: 100%;
            max-width: 400px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            color: white;
            font-size: 14px;
          "
        />
        <div style="margin-top: 20px;">
          <button 
            id="manual-load"
            style="
              background: #4CAF50;
              color: white;
              border: none;
              padding: 10px 20px;
              border-radius: 6px;
              cursor: pointer;
              font-size: 14px;
              margin-right: 10px;
            "
          >Load Dataset</button>
          <button 
            id="manual-cancel"
            style="
              background: rgba(255, 255, 255, 0.1);
              color: white;
              border: 1px solid rgba(255, 255, 255, 0.2);
              padding: 10px 20px;
              border-radius: 6px;
              cursor: pointer;
              font-size: 14px;
            "
          >Cancel</button>
        </div>
        <p style="margin-top: 20px; font-size: 12px; color: #888;">
          Tip: Ask your server administrator to enable directory listing or WebDAV
        </p>
      </div>
    `;
    
    const input = content.querySelector('#manual-path') as HTMLInputElement;
    const loadBtn = content.querySelector('#manual-load') as HTMLButtonElement;
    const cancelBtn = content.querySelector('#manual-cancel') as HTMLButtonElement;
    
    loadBtn.onclick = () => {
      const path = input.value.trim();
      if (path) {
        this.onDatasetSelect(path);
        this.close();
      }
    };
    
    cancelBtn.onclick = () => this.close();
    
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
   * Close and dispose the browser.
   */
  close(): void {
    if (this.onClose) {
      this.onClose();
    }
    this.panel.remove();
  }
}