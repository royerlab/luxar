/**
 * Hierarchical Timing Panel Component
 *
 * Displays a collapsible tree view of timing data from the UpdateProfiler.
 * Shows last value and exponential moving average for each timing entry.
 * Highlights operations that exceed the 60fps budget (>16ms).
 */

import type { TimingEntry, TimingMetadata } from '../../profiling/update-profiler';
import { formatMs, hasOverBudget } from '../../profiling/update-profiler';
import { escapeHtml } from '../../utils/escape-html';

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
    return `<span class="luxar-timing-panel__tag luxar-timing-panel__tag--skip">${escapeHtml(metadata.skipReason || 'skipped')}</span>`;
  }

  if (metadata.chunks !== undefined) {
    tags.push(`<span class="luxar-timing-panel__tag">${metadata.chunks} chunks</span>`);
  }

  if (metadata.cacheHits !== undefined && metadata.cacheMisses !== undefined) {
    const total = metadata.cacheHits + metadata.cacheMisses;
    if (total > 0) {
      const hitRate = Math.round((metadata.cacheHits / total) * 100);
      const tagClass =
        hitRate > 80
          ? 'luxar-timing-panel__tag--good'
          : hitRate > 50
            ? ''
            : 'luxar-timing-panel__tag--warn';
      tags.push(`<span class="luxar-timing-panel__tag ${tagClass}">${hitRate}% cache</span>`);
    }
  }

  if (metadata.points !== undefined) {
    tags.push(`<span class="luxar-timing-panel__tag">${formatCount(metadata.points)} pts</span>`);
  }

  if (metadata.segments !== undefined) {
    tags.push(
      `<span class="luxar-timing-panel__tag">${formatCount(metadata.segments)} segs</span>`
    );
  }

  if (metadata.splats !== undefined) {
    tags.push(
      `<span class="luxar-timing-panel__tag">${formatCount(metadata.splats)} splats</span>`
    );
  }

  if (metadata.info) {
    tags.push(`<span class="luxar-timing-panel__tag">${escapeHtml(metadata.info)}</span>`);
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
 * Node type for aggregation
 */
type NodeType = 'Points' | 'Lines' | 'GSplats' | 'Unknown';

/**
 * Extract node type from entry name (optimized - checks first char)
 * Entry names look like: "Points (/path)", "Lines (/path)", "GSplats (/path)"
 */
function getNodeType(name: string): NodeType {
  // Fast path: check first character
  const first = name.charCodeAt(0);
  if (first === 80 && name.startsWith('Points')) return 'Points'; // 'P'
  if (first === 76 && name.startsWith('Lines')) return 'Lines'; // 'L'
  if (first === 71 && name.startsWith('GSplats')) return 'GSplats'; // 'G'
  return 'Unknown';
}

/**
 * Aggregate children entries by node type (optimized single-pass)
 * Returns a new array with aggregated entries for Points, Lines, GSplats
 */
function aggregateByNodeType(children: TimingEntry[]): TimingEntry[] {
  if (children.length === 0) return [];

  // Reusable accumulators for each type (pre-allocated)
  const accumulators: Record<
    NodeType,
    {
      entries: TimingEntry[];
      lastMs: number;
      avgMs: number;
      count: number;
      overBudget: boolean;
      allSkipped: boolean;
      points: number;
      segments: number;
      splats: number;
      childrenByName: Map<string, TimingEntry[]>;
    }
  > = {
    Points: {
      entries: [],
      lastMs: 0,
      avgMs: 0,
      count: 0,
      overBudget: false,
      allSkipped: true,
      points: 0,
      segments: 0,
      splats: 0,
      childrenByName: new Map(),
    },
    Lines: {
      entries: [],
      lastMs: 0,
      avgMs: 0,
      count: 0,
      overBudget: false,
      allSkipped: true,
      points: 0,
      segments: 0,
      splats: 0,
      childrenByName: new Map(),
    },
    GSplats: {
      entries: [],
      lastMs: 0,
      avgMs: 0,
      count: 0,
      overBudget: false,
      allSkipped: true,
      points: 0,
      segments: 0,
      splats: 0,
      childrenByName: new Map(),
    },
    Unknown: {
      entries: [],
      lastMs: 0,
      avgMs: 0,
      count: 0,
      overBudget: false,
      allSkipped: true,
      points: 0,
      segments: 0,
      splats: 0,
      childrenByName: new Map(),
    },
  };

  // Single pass: accumulate all data
  for (let i = 0; i < children.length; i++) {
    const child = children[i];
    const nodeType = getNodeType(child.name);
    const acc = accumulators[nodeType];

    acc.entries.push(child);

    // Accumulate timings
    if (!child.metadata?.skipped) {
      acc.lastMs += child.lastMs;
      acc.avgMs += child.avgMs;
      acc.allSkipped = false;
    }
    if (child.count > acc.count) acc.count = child.count;
    if (child.overBudget) acc.overBudget = true;

    // Accumulate metadata
    const meta = child.metadata;
    if (meta) {
      if (meta.points) acc.points += meta.points;
      if (meta.segments) acc.segments += meta.segments;
      if (meta.splats) acc.splats += meta.splats;
    }

    // Group children by name
    for (let j = 0; j < child.children.length; j++) {
      const subChild = child.children[j];
      let arr = acc.childrenByName.get(subChild.name);
      if (!arr) {
        arr = [];
        acc.childrenByName.set(subChild.name, arr);
      }
      arr.push(subChild);
    }
  }

  // Build result in order: Points, Lines, GSplats, Unknown
  const result: TimingEntry[] = [];
  const typeOrder: NodeType[] = ['Points', 'Lines', 'GSplats', 'Unknown'];

  for (let t = 0; t < typeOrder.length; t++) {
    const nodeType = typeOrder[t];
    const acc = accumulators[nodeType];
    if (acc.entries.length === 0) continue;

    if (nodeType === 'Unknown') {
      // Pass through unknown types as-is
      for (let i = 0; i < acc.entries.length; i++) {
        result.push(acc.entries[i]);
      }
      continue;
    }

    // Build aggregated children
    const aggregatedChildren: TimingEntry[] = [];
    for (const [name, childEntries] of acc.childrenByName) {
      aggregatedChildren.push(aggregateChildEntriesFast(name, childEntries));
    }

    // Sort by avg time descending (in-place for efficiency)
    aggregatedChildren.sort((a, b) => b.avgMs - a.avgMs);

    // Build metadata only if needed
    let metadata: TimingMetadata | undefined;
    const nodeCount = acc.entries.length;
    if (
      acc.allSkipped ||
      (nodeType === 'Points' && acc.points > 0) ||
      (nodeType === 'Lines' && acc.segments > 0) ||
      (nodeType === 'GSplats' && acc.splats > 0) ||
      nodeCount > 1
    ) {
      metadata = {};
      if (acc.allSkipped) {
        metadata.skipped = true;
        metadata.skipReason = acc.entries[0].metadata?.skipReason;
      }
      if (nodeType === 'Points' && acc.points > 0) metadata.points = acc.points;
      if (nodeType === 'Lines' && acc.segments > 0) metadata.segments = acc.segments;
      if (nodeType === 'GSplats' && acc.splats > 0) metadata.splats = acc.splats;
      if (nodeCount > 1) metadata.info = `${nodeCount} nodes`;
    }

    result.push({
      name: nodeType,
      lastMs: acc.lastMs,
      avgMs: acc.avgMs,
      count: acc.count,
      children: aggregatedChildren,
      metadata,
      overBudget: acc.overBudget || acc.lastMs > 16.67,
    });
  }

  return result;
}

/**
 * Aggregate child entries with same name (optimized, non-recursive for common case)
 */
function aggregateChildEntriesFast(name: string, entries: TimingEntry[]): TimingEntry {
  let totalLastMs = 0;
  let totalAvgMs = 0;
  let maxCount = 0;
  let anyOverBudget = false;

  // Check if any entry has sub-children
  let hasSubChildren = false;
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    totalLastMs += entry.lastMs;
    totalAvgMs += entry.avgMs;
    if (entry.count > maxCount) maxCount = entry.count;
    if (entry.overBudget) anyOverBudget = true;
    if (entry.children.length > 0) hasSubChildren = true;
  }

  // Fast path: no sub-children
  if (!hasSubChildren) {
    return {
      name,
      lastMs: totalLastMs,
      avgMs: totalAvgMs,
      count: maxCount,
      children: [],
      overBudget: anyOverBudget || totalLastMs > 16.67,
    };
  }

  // Slow path: aggregate sub-children recursively
  const subChildrenByName = new Map<string, TimingEntry[]>();
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    for (let j = 0; j < entry.children.length; j++) {
      const child = entry.children[j];
      let arr = subChildrenByName.get(child.name);
      if (!arr) {
        arr = [];
        subChildrenByName.set(child.name, arr);
      }
      arr.push(child);
    }
  }

  const aggregatedSubChildren: TimingEntry[] = [];
  for (const [subName, subEntries] of subChildrenByName) {
    aggregatedSubChildren.push(aggregateChildEntriesFast(subName, subEntries));
  }

  return {
    name,
    lastMs: totalLastMs,
    avgMs: totalAvgMs,
    count: maxCount,
    children: aggregatedSubChildren,
    overBudget: anyOverBudget || totalLastMs > 16.67,
  };
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
  const rowClasses = ['luxar-timing-panel__row'];
  if (entry.overBudget) rowClasses.push('luxar-timing-panel__row--over');
  if (entry.metadata?.skipped) rowClasses.push('luxar-timing-panel__row--skipped');
  if (hasOverBudget(entry) && !entry.overBudget)
    rowClasses.push('luxar-timing-panel__row--child-over');

  // Expand/collapse indicator
  const expandIcon = hasChildren
    ? `<span class="luxar-timing-panel__expand" data-path="${escapeHtml(path)}">${expanded ? '▼' : '►'}</span>`
    : '<span class="luxar-timing-panel__expand-placeholder"></span>';

  // Time values (skip for skipped entries)
  const lastValue = entry.metadata?.skipped ? '—' : formatMs(entry.lastMs);
  const avgValue = entry.metadata?.skipped ? '—' : formatMs(entry.avgMs);

  // Build row HTML (data-path on row for incremental updates)
  let html = `
    <div class="${rowClasses.join(' ')}" style="padding-left: ${indent}px" data-path="${escapeHtml(path)}">
      <div class="luxar-timing-panel__name">
        ${expandIcon}
        <span class="luxar-timing-panel__label">${escapeHtml(entry.name)}</span>
        ${renderMetadata(entry.metadata)}
      </div>
      <div class="luxar-timing-panel__values">
        <span class="luxar-timing-panel__last">${lastValue}</span>
        <span class="luxar-timing-panel__avg">${avgValue}</span>
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
 * Create an aggregated version of the root entry with children grouped by node type
 */
function createAggregatedRoot(root: TimingEntry): TimingEntry {
  return {
    ...root,
    children: aggregateByNodeType(root.children),
  };
}

/**
 * Render the complete hierarchical timing panel
 * Aggregates performance data by node type (Points, Lines, GSplats) instead of individual nodes
 */
export function renderHierarchicalTimingPanel(root: TimingEntry): string {
  if (root.count === 0) {
    return `
      <div class="luxar-timing-panel luxar-timing-panel--empty">
        <div class="luxar-timing-panel__empty-msg">
          No timing data yet. Move a slider or navigate dimensions to see performance breakdown.
        </div>
      </div>
    `;
  }

  // Aggregate children by node type for cleaner display
  const aggregatedRoot = createAggregatedRoot(root);

  return `
    <div class="luxar-timing-panel">
      <div class="luxar-timing-panel__header">
        <div class="luxar-timing-panel__header-label">Operation</div>
        <div class="luxar-timing-panel__header-values">
          <span class="luxar-timing-panel__header-last">Last</span>
          <span class="luxar-timing-panel__header-avg">Avg</span>
        </div>
      </div>
      <div class="luxar-timing-panel__body">
        ${renderEntry(aggregatedRoot, 0, '')}
      </div>
      <div class="luxar-timing-panel__footer">
        <span class="luxar-timing-panel__legend">
          <span class="luxar-timing-panel__legend-item luxar-timing-panel__legend-item--normal">Normal</span>
          <span class="luxar-timing-panel__legend-item luxar-timing-panel__legend-item--over">&gt;16ms (60fps)</span>
          <span class="luxar-timing-panel__legend-item luxar-timing-panel__legend-item--skip">Skipped</span>
        </span>
        <span class="luxar-timing-panel__update-count">${root.count} updates</span>
      </div>
    </div>
  `;
}

/**
 * Attach click handlers for expand/collapse
 * Call this after rendering the panel
 */
export function attachTimingPanelHandlers(container: HTMLElement, onUpdate: () => void): void {
  container.querySelectorAll('.luxar-timing-panel__expand').forEach((el) => {
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
 * @param root - The updated timing data (will be aggregated to match rendered structure)
 * @returns true if update was successful, false if full re-render is needed
 */
export function updateTimingPanelValues(container: HTMLElement, root: TimingEntry): boolean {
  const timingBody = container.querySelector('.luxar-timing-panel__body');
  if (!timingBody) return false;

  // Update the update count in footer
  const updateCount = container.querySelector('.luxar-timing-panel__update-count');
  if (updateCount) {
    updateCount.textContent = `${root.count} updates`;
  }

  // Aggregate to match the rendered structure
  const aggregatedRoot = createAggregatedRoot(root);

  // Recursively update values for each entry
  return updateEntryValues(timingBody as HTMLElement, aggregatedRoot, 0, '');
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
  const rows = Array.from(container.querySelectorAll(':scope > .luxar-timing-panel__row'));
  let rowIndex = 0;

  // Find the row matching this path (by checking the row's data-path)
  for (const row of rows) {
    const rowEl = row as HTMLElement;
    const rowPath = rowEl.dataset.path;

    // Match by path
    if (rowPath === path) {
      const expandBtn = row.querySelector('.luxar-timing-panel__expand') as HTMLElement;
      // Update row classes
      const rowClasses = ['luxar-timing-panel__row'];
      if (entry.overBudget) rowClasses.push('luxar-timing-panel__row--over');
      if (entry.metadata?.skipped) rowClasses.push('luxar-timing-panel__row--skipped');
      if (hasOverBudget(entry) && !entry.overBudget)
        rowClasses.push('luxar-timing-panel__row--child-over');
      (row as HTMLElement).className = rowClasses.join(' ');

      // Update expand icon
      if (expandBtn && hasChildren) {
        expandBtn.textContent = expanded ? '▼' : '►';
      }

      // Update timing values
      const lastSpan = row.querySelector('.luxar-timing-panel__last');
      const avgSpan = row.querySelector('.luxar-timing-panel__avg');
      if (lastSpan) {
        lastSpan.textContent = entry.metadata?.skipped ? '—' : formatMs(entry.lastMs);
      }
      if (avgSpan) {
        avgSpan.textContent = entry.metadata?.skipped ? '—' : formatMs(entry.avgMs);
      }

      // Update metadata tags
      const timingName = row.querySelector('.luxar-timing-panel__name');
      if (timingName) {
        // Remove existing metadata tags
        timingName
          .querySelectorAll('.luxar-timing-panel__tag')
          .forEach((tag: Element) => tag.remove());
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
