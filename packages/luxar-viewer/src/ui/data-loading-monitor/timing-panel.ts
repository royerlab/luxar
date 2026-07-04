/**
 * Hierarchical Timing Panel Component
 *
 * Displays a collapsible tree view of timing data from the UpdateProfiler.
 * Shows last value and exponential moving average for each timing entry.
 * Highlights operations that exceed the 60fps budget (>16ms).
 */

import type { TimingEntry, TimingMetadata } from '../../profiling/update-profiler';
import { formatMs, hasOverBudget, REFINEMENT_ROOT } from '../../profiling/update-profiler';
import { escapeHtml } from '../../utils/escape-html';

/**
 * Tooltip descriptions for timing operations.
 * Keys are matched against entry names (exact match).
 */
const TOOLTIPS: Record<string, string> = {
  'Total Update': 'End-to-end time for all node updates in this frame',
  'LOD Refinement': 'Background passes loading remaining LOD levels after first paint',
  Points: 'Point cloud data: query, load, project, and upload',
  Lines: 'Line/track data: query segments, load vertices, project, and upload',
  GSplats: 'Gaussian splat data: query, load, project, and upload',
  'Spatial Query': 'Find which data chunks intersect the current view slice',
  'Load Arrays': 'Fetch and decompress zarr chunks from the data source',
  'Load Segments': 'Fetch segment index arrays (pairs of vertex references)',
  'Load Vertices': 'Fetch vertex positions and attributes for referenced vertices',
  'Index Remap':
    'Build global → local vertex map and remap segment indices into local buffer space',
  'Project to 3D': 'Slice nD data to 3D display space (visibility filtering, Cholesky marginals)',
  'Update Buffers': 'Upload processed data to GPU buffer attributes',
  'Concatenate LODs': 'Merge loaded LOD levels into one contiguous buffer set',
};

/** Suffix appended to the tooltip of rows that did not run in the latest update. */
const STALE_TOOLTIP = 'Did not run in the latest update — value is from an earlier one';

/**
 * Get tooltip text for a timing entry name
 */
function getTooltip(name: string): string | undefined {
  return TOOLTIPS[name];
}

/**
 * State tracking for collapsed/expanded nodes
 */
const expandedState = new Map<string, boolean>();

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
      allStale: boolean;
      staleLastMs: number;
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
      allStale: true,
      staleLastMs: 0,
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
      allStale: true,
      staleLastMs: 0,
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
      allStale: true,
      staleLastMs: 0,
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
      allStale: true,
      staleLastMs: 0,
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

    // Aggregate timings as the MAX across nodes, not the sum: the per-node
    // sessions of one update run CONCURRENTLY (they all open at update start
    // and close at the atomic commit), so summing their wall-clock spans
    // multiplies by node count — 64 nodes × ~186ms would display 11.9s under
    // a 187ms Total Update. Max = the slowest node = the type's critical
    // path within the update. (Numeric metadata below stays summed — those
    // are element counts, not durations.)
    // Stale entries (not touched by the latest update) are excluded from
    // the fresh max — a stale 94ms child must not mask a fresh 29ms one.
    // When EVERY entry is stale the row itself renders stale, showing the
    // stale max greyed instead of a misleading 0.
    if (!child.metadata?.skipped) {
      if (!child.stale) {
        if (child.lastMs > acc.lastMs) acc.lastMs = child.lastMs;
        acc.allStale = false;
      } else if (child.lastMs > acc.staleLastMs) {
        acc.staleLastMs = child.lastMs;
      }
      if (child.avgMs > acc.avgMs) acc.avgMs = child.avgMs;
      acc.allSkipped = false;
    }
    if (child.count > acc.count) acc.count = child.count;
    if (child.overBudget && !child.stale) acc.overBudget = true;

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

    const stale = !acc.allSkipped && acc.allStale;
    result.push({
      name: nodeType,
      // An all-stale row shows its last-known (stale) max, greyed by the
      // `stale` flag; a fresh or mixed row shows only the fresh max.
      lastMs: stale ? acc.staleLastMs : acc.lastMs,
      avgMs: acc.avgMs,
      count: acc.count,
      children: aggregatedChildren,
      metadata,
      overBudget: acc.overBudget || (!stale && acc.lastMs > 16.67),
      stale,
    });
  }

  return result;
}

/**
 * Aggregate child entries with same name (optimized, non-recursive for common case)
 */
function aggregateChildEntriesFast(name: string, entries: TimingEntry[]): TimingEntry {
  let maxLastMs = 0;
  let staleLastMs = 0;
  let maxAvgMs = 0;
  let maxCount = 0;
  let anyOverBudget = false;
  let allStale = true;

  // Check if any entry has sub-children
  let hasSubChildren = false;
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    // MAX across nodes, not sum — sibling entries run concurrently within
    // one update (see aggregateByNodeType). Stale entries don't contribute
    // to the fresh max; an all-stale row falls back to the stale max so it
    // shows its last-known value (greyed) instead of 0.
    if (!entry.stale) {
      if (entry.lastMs > maxLastMs) maxLastMs = entry.lastMs;
      allStale = false;
      if (entry.overBudget) anyOverBudget = true;
    } else if (entry.lastMs > staleLastMs) {
      staleLastMs = entry.lastMs;
    }
    if (entry.avgMs > maxAvgMs) maxAvgMs = entry.avgMs;
    if (entry.count > maxCount) maxCount = entry.count;
    if (entry.children.length > 0) hasSubChildren = true;
  }

  const displayLastMs = allStale ? staleLastMs : maxLastMs;

  // Fast path: no sub-children
  if (!hasSubChildren) {
    return {
      name,
      lastMs: displayLastMs,
      avgMs: maxAvgMs,
      count: maxCount,
      children: [],
      overBudget: anyOverBudget || (!allStale && maxLastMs > 16.67),
      stale: allStale,
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
    lastMs: displayLastMs,
    avgMs: maxAvgMs,
    count: maxCount,
    children: aggregatedSubChildren,
    overBudget: anyOverBudget || (!allStale && maxLastMs > 16.67),
    stale: allStale,
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
  if (entry.stale) rowClasses.push('luxar-timing-panel__row--stale');
  if (hasOverBudget(entry) && !entry.overBudget)
    rowClasses.push('luxar-timing-panel__row--child-over');

  // Expand/collapse indicator
  const expandIcon = hasChildren
    ? `<span class="luxar-timing-panel__expand" data-path="${escapeHtml(path)}">${expanded ? '▼' : '►'}</span>`
    : '<span class="luxar-timing-panel__expand-placeholder"></span>';

  // Time values (skip for skipped entries)
  const lastValue = entry.metadata?.skipped ? '—' : formatMs(entry.lastMs);
  const avgValue = entry.metadata?.skipped ? '—' : formatMs(entry.avgMs);

  // Tooltip for the operation label (stale rows explain their grey state)
  const tooltip = getTooltip(entry.name);
  const tooltipText = entry.stale
    ? tooltip
      ? `${tooltip} — ${STALE_TOOLTIP}`
      : STALE_TOOLTIP
    : tooltip;
  const titleAttr = tooltipText ? ` title="${escapeHtml(tooltipText)}"` : '';

  // Build row HTML (data-path on row for incremental updates)
  let html = `
    <div class="${rowClasses.join(' ')}" style="padding-left: ${indent}px" data-path="${escapeHtml(path)}">
      <div class="luxar-timing-panel__name">
        ${expandIcon}
        <span class="luxar-timing-panel__label"${titleAttr}>${escapeHtml(entry.name)}</span>
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
 * Aggregates performance data by node type (Points, Lines, GSplats) instead of individual nodes.
 *
 * @param root - The 'Total Update' timing tree (per-frame demand updates)
 * @param refinementRoot - Optional 'LOD Refinement' tree (background passes),
 *   rendered as a second section below the main tree when it has data
 */
export function renderHierarchicalTimingPanel(
  root: TimingEntry,
  refinementRoot?: TimingEntry
): string {
  const hasRefinement = refinementRoot !== undefined && refinementRoot.count > 0;
  if (root.count === 0 && !hasRefinement) {
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
  const refinementHtml = hasRefinement
    ? renderEntry(createAggregatedRoot(refinementRoot), 0, '')
    : '';
  const refinementCount = hasRefinement
    ? ` · ${refinementRoot.count} refinement ${refinementRoot.count === 1 ? 'pass' : 'passes'}`
    : '';

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
        ${renderEntry(aggregatedRoot, 0, '')}${refinementHtml}
      </div>
      <div class="luxar-timing-panel__footer">
        <span class="luxar-timing-panel__legend">
          <span class="luxar-timing-panel__legend-item luxar-timing-panel__legend-item--normal">Normal</span>
          <span class="luxar-timing-panel__legend-item luxar-timing-panel__legend-item--over">&gt;16ms (60fps)</span>
          <span class="luxar-timing-panel__legend-item luxar-timing-panel__legend-item--skip">Skipped</span>
        </span>
        <span class="luxar-timing-panel__update-count">${root.count} updates${refinementCount}</span>
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

    el.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      const path = (el as HTMLElement).dataset.path;
      if (path) {
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
export function updateTimingPanelValues(
  container: HTMLElement,
  root: TimingEntry,
  refinementRoot?: TimingEntry
): boolean {
  const timingBody = container.querySelector('.luxar-timing-panel__body');
  if (!timingBody) return false;

  const hasRefinement = refinementRoot !== undefined && refinementRoot.count > 0;

  // The refinement tree appears once its first pass records — that structural
  // change needs a full re-render.
  const refinementRendered =
    timingBody.querySelector(
      `:scope > .luxar-timing-panel__row[data-path="${REFINEMENT_ROOT}"]`
    ) !== null;
  if (hasRefinement !== refinementRendered) return false;

  // Update the update count in footer
  const updateCount = container.querySelector('.luxar-timing-panel__update-count');
  if (updateCount) {
    const refinementCount = hasRefinement
      ? ` · ${refinementRoot.count} refinement ${refinementRoot.count === 1 ? 'pass' : 'passes'}`
      : '';
    updateCount.textContent = `${root.count} updates${refinementCount}`;
  }

  // Aggregate to match the rendered structure
  const aggregatedRoot = createAggregatedRoot(root);

  // Recursively update values for each entry
  if (!updateEntryValues(timingBody as HTMLElement, aggregatedRoot, 0, '')) {
    return false;
  }
  if (hasRefinement) {
    return updateEntryValues(
      timingBody as HTMLElement,
      createAggregatedRoot(refinementRoot),
      0,
      ''
    );
  }
  return true;
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
      if (entry.stale) rowClasses.push('luxar-timing-panel__row--stale');
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
