/**
 * Hierarchical Timing Panel Component
 *
 * Displays a collapsible tree view of timing data from the UpdateProfiler.
 * Shows last value and exponential moving average for each timing entry.
 * Highlights operations that exceed the 60fps budget (>16ms).
 */

import type { TimingEntry, TimingMetadata } from '../../profiling/update-profiler';
import { formatMs, hasOverBudget } from '../../profiling/update-profiler';

/**
 * State tracking for collapsed/expanded nodes
 */
const expandedState = new Map<string, boolean>();

/**
 * Interaction lock - prevents updates while user is interacting
 */
let interactionLock = false;
let interactionLockTimeout: ReturnType<typeof setTimeout> | null = null;

/**
 * Check if interaction is locked (updates should be skipped)
 */
export function isInteractionLocked(): boolean {
  return interactionLock;
}

/**
 * Set interaction lock (called when user hovers over expand buttons)
 */
export function setInteractionLock(locked: boolean): void {
  interactionLock = locked;

  // Clear any pending timeout
  if (interactionLockTimeout) {
    clearTimeout(interactionLockTimeout);
    interactionLockTimeout = null;
  }

  // If locking, also set a timeout to auto-unlock after interaction
  if (locked) {
    interactionLockTimeout = setTimeout(() => {
      interactionLock = false;
      interactionLockTimeout = null;
    }, 2000); // Keep locked for 2 seconds after last interaction
  }
}

/**
 * Get a unique path for an entry (for tracking expanded state)
 */
function getEntryPath(entry: TimingEntry, parentPath = ''): string {
  return parentPath ? `${parentPath}/${entry.name}` : entry.name;
}

/**
 * Check if an entry is expanded (default: top 2 levels expanded)
 */
function isExpanded(path: string, depth: number): boolean {
  if (expandedState.has(path)) {
    return expandedState.get(path)!;
  }
  // Default: expand top 2 levels
  return depth < 2;
}

/**
 * Toggle expanded state
 */
export function toggleExpanded(path: string): void {
  const current = expandedState.get(path);
  if (current === undefined) {
    // Was using default (expanded for depth < 2)
    expandedState.set(path, false);
  } else {
    expandedState.set(path, !current);
  }
}

/**
 * Render metadata tags
 */
function renderMetadata(metadata: TimingMetadata | undefined): string {
  if (!metadata) return '';

  const tags: string[] = [];

  if (metadata.skipped) {
    return `<span class="timing-tag timing-tag-skip">${metadata.skipReason || 'skipped'}</span>`;
  }

  if (metadata.chunks !== undefined) {
    tags.push(`<span class="timing-tag">${metadata.chunks} chunks</span>`);
  }

  if (metadata.cacheHits !== undefined && metadata.cacheMisses !== undefined) {
    const total = metadata.cacheHits + metadata.cacheMisses;
    if (total > 0) {
      const hitRate = Math.round((metadata.cacheHits / total) * 100);
      const tagClass = hitRate > 80 ? 'timing-tag-good' : hitRate > 50 ? '' : 'timing-tag-warn';
      tags.push(`<span class="timing-tag ${tagClass}">${hitRate}% cache</span>`);
    }
  }

  if (metadata.points !== undefined) {
    tags.push(`<span class="timing-tag">${formatCount(metadata.points)} pts</span>`);
  }

  if (metadata.segments !== undefined) {
    tags.push(`<span class="timing-tag">${formatCount(metadata.segments)} segs</span>`);
  }

  if (metadata.splats !== undefined) {
    tags.push(`<span class="timing-tag">${formatCount(metadata.splats)} splats</span>`);
  }

  if (metadata.info) {
    tags.push(`<span class="timing-tag">${metadata.info}</span>`);
  }

  return tags.join(' ');
}

/**
 * Format large numbers with K/M suffix
 */
function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

/**
 * Render a single timing entry row
 */
function renderEntry(entry: TimingEntry, depth: number, parentPath: string): string {
  const path = getEntryPath(entry, parentPath);
  const hasChildren = entry.children.length > 0;
  const expanded = isExpanded(path, depth);
  const indent = depth * 16;

  // Determine row classes
  const rowClasses = ['timing-row'];
  if (entry.overBudget) rowClasses.push('timing-over-budget');
  if (entry.metadata?.skipped) rowClasses.push('timing-skipped');
  if (hasOverBudget(entry) && !entry.overBudget) rowClasses.push('timing-child-over-budget');

  // Expand/collapse indicator
  const expandIcon = hasChildren
    ? `<span class="timing-expand" data-path="${path}">${expanded ? '▼' : '►'}</span>`
    : '<span class="timing-expand-placeholder"></span>';

  // Time values (skip for skipped entries)
  const lastValue = entry.metadata?.skipped ? '—' : formatMs(entry.lastMs);
  const avgValue = entry.metadata?.skipped ? '—' : formatMs(entry.avgMs);

  // Build row HTML (data-path on row for incremental updates)
  let html = `
    <div class="${rowClasses.join(' ')}" style="padding-left: ${indent}px" data-path="${path}">
      <div class="timing-name">
        ${expandIcon}
        <span class="timing-label">${escapeHtml(entry.name)}</span>
        ${renderMetadata(entry.metadata)}
      </div>
      <div class="timing-values">
        <span class="timing-last">${lastValue}</span>
        <span class="timing-avg">${avgValue}</span>
      </div>
    </div>
  `;

  // Render children if expanded
  if (hasChildren && expanded) {
    for (const child of entry.children) {
      html += renderEntry(child, depth + 1, path);
    }
  }

  return html;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Render the complete hierarchical timing panel
 */
export function renderHierarchicalTimingPanel(root: TimingEntry): string {
  if (root.count === 0) {
    return `
      <div class="timing-panel timing-empty">
        <div class="timing-empty-message">
          No timing data yet. Move a slider or navigate dimensions to see performance breakdown.
        </div>
      </div>
    `;
  }

  return `
    <div class="timing-panel">
      <div class="timing-header">
        <div class="timing-header-label">Operation</div>
        <div class="timing-header-values">
          <span class="timing-header-last">Last</span>
          <span class="timing-header-avg">Avg</span>
        </div>
      </div>
      <div class="timing-body">
        ${renderEntry(root, 0, '')}
      </div>
      <div class="timing-footer">
        <span class="timing-legend">
          <span class="timing-legend-item timing-legend-normal">Normal</span>
          <span class="timing-legend-item timing-legend-over">&gt;16ms (60fps)</span>
          <span class="timing-legend-item timing-legend-skip">Skipped</span>
        </span>
        <span class="timing-update-count">${root.count} updates</span>
      </div>
    </div>
  `;
}

/**
 * Attach click handlers for expand/collapse
 * Call this after rendering the panel
 */
export function attachTimingPanelHandlers(container: HTMLElement, onUpdate: () => void): void {
  container.querySelectorAll('.timing-expand').forEach((el) => {
    // Avoid adding duplicate listeners by checking for marker
    if ((el as HTMLElement).dataset.hasListener) return;
    (el as HTMLElement).dataset.hasListener = 'true';

    // Lock interaction when hovering over expand buttons
    el.addEventListener('mouseenter', () => {
      setInteractionLock(true);
    });

    el.addEventListener('mouseleave', () => {
      // Keep lock for a bit after leaving to allow click
      setTimeout(() => {
        setInteractionLock(false);
      }, 500);
    });

    el.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      const path = (el as HTMLElement).dataset.path;
      if (path) {
        // Extend lock during toggle
        setInteractionLock(true);
        toggleExpanded(path);
        onUpdate();
      }
    });
  });
}

/**
 * Update timing values in-place without re-rendering the DOM structure.
 * This prevents losing expand/collapse state and click handlers during rapid updates.
 *
 * @param container - The container element with the timing panel
 * @param root - The updated timing data
 * @returns true if update was successful, false if full re-render is needed
 */
export function updateTimingPanelValues(container: HTMLElement, root: TimingEntry): boolean {
  const timingBody = container.querySelector('.timing-body');
  if (!timingBody) return false;

  // Update the update count in footer
  const updateCount = container.querySelector('.timing-update-count');
  if (updateCount) {
    updateCount.textContent = `${root.count} updates`;
  }

  // Recursively update values for each entry
  return updateEntryValues(timingBody as HTMLElement, root, 0, '');
}

/**
 * Update a single entry's values in-place
 */
function updateEntryValues(
  container: HTMLElement,
  entry: TimingEntry,
  depth: number,
  parentPath: string
): boolean {
  const path = getEntryPath(entry, parentPath);
  const hasChildren = entry.children.length > 0;
  const expanded = isExpanded(path, depth);

  // Find the row for this entry
  const rows = Array.from(container.querySelectorAll(':scope > .timing-row'));
  let rowIndex = 0;

  // Find the row matching this path (by checking the row's data-path)
  for (const row of rows) {
    const rowEl = row as HTMLElement;
    const rowPath = rowEl.dataset.path;

    // Match by path
    if (rowPath === path) {
      const expandBtn = row.querySelector('.timing-expand') as HTMLElement;
      // Update row classes
      const rowClasses = ['timing-row'];
      if (entry.overBudget) rowClasses.push('timing-over-budget');
      if (entry.metadata?.skipped) rowClasses.push('timing-skipped');
      if (hasOverBudget(entry) && !entry.overBudget) rowClasses.push('timing-child-over-budget');
      (row as HTMLElement).className = rowClasses.join(' ');

      // Update expand icon
      if (expandBtn && hasChildren) {
        expandBtn.textContent = expanded ? '▼' : '►';
      }

      // Update timing values
      const lastSpan = row.querySelector('.timing-last');
      const avgSpan = row.querySelector('.timing-avg');
      if (lastSpan) {
        lastSpan.textContent = entry.metadata?.skipped ? '—' : formatMs(entry.lastMs);
      }
      if (avgSpan) {
        avgSpan.textContent = entry.metadata?.skipped ? '—' : formatMs(entry.avgMs);
      }

      // Update metadata tags
      const timingName = row.querySelector('.timing-name');
      if (timingName) {
        // Remove existing metadata tags
        timingName.querySelectorAll('.timing-tag').forEach((tag: Element) => tag.remove());
        // Add new metadata
        const metadataHtml = renderMetadata(entry.metadata);
        if (metadataHtml) {
          const temp = document.createElement('div');
          temp.innerHTML = metadataHtml;
          while (temp.firstChild) {
            timingName.appendChild(temp.firstChild);
          }
        }
      }

      // Handle children - this is where we may need to signal a full re-render
      // If the number of children changed, we need a full re-render
      if (hasChildren && expanded) {
        // Count existing child rows by checking row's data-path attribute
        const siblingRows = Array.from(rows).slice(rowIndex + 1);
        const existingChildPaths = new Set<string>();
        for (const sibling of siblingRows) {
          const siblingEl = sibling as HTMLElement;
          const siblingPath = siblingEl.dataset.path;
          if (siblingPath?.startsWith(path + '/')) {
            // Only count direct children, not grandchildren
            const relativePath = siblingPath.slice(path.length + 1);
            if (!relativePath.includes('/')) {
              existingChildPaths.add(siblingPath);
            }
          } else if (siblingPath && !siblingPath.startsWith(path)) {
            break; // This is a sibling, not a descendant
          }
        }

        // Check if children match
        const currentChildPaths = entry.children.map((c) => getEntryPath(c, path));
        const childrenMatch =
          existingChildPaths.size === currentChildPaths.length &&
          currentChildPaths.every((p) => existingChildPaths.has(p));

        if (!childrenMatch) {
          // Children changed, need full re-render
          return false;
        }

        // Recursively update children
        for (const child of entry.children) {
          if (!updateEntryValues(container, child, depth + 1, path)) {
            return false;
          }
        }
      }

      return true;
    }

    rowIndex++;
  }

  // Entry not found - need full re-render
  return false;
}

// CSS styles for the timing panel are in:
// src/styles/components/data-loading-monitor.css
// (See the "Hierarchical Timing Panel" section)
