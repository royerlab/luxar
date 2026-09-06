/**
 * Shared HTML primitives and semantic styling for monitor templates.
 */

import { escapeHtml } from '../../../utils/escape-html';

/**
 * Semantic color names mapped to CSS class modifiers.
 * These are used with the luxar-color--{name} classes.
 */
export type SemanticColor =
  'success' | 'warning' | 'error' | 'info' | 'muted' | 'dimmed' | 'primary';

/**
 * Get CSS class for a semantic color.
 */
export function getColorClass(color: SemanticColor): string {
  return `luxar-color--${color}`;
}

/**
 * Inline SVG line-icon set for the monitor (tab bar, section headers,
 * status glyphs). Replaces the previous emoji glyphs, which rendered
 * differently on every platform and clashed with the viewer rail's
 * stroke-icon language. All icons are 14×14 stroke = currentColor, so
 * they inherit the color of their context (including the semantic
 * `luxar-color--*` classes).
 */
const MICON =
  '<svg class="luxar-micon" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">';
export const MONITOR_ICONS = {
  /** Ascending bars — the Overview tab. */
  overview: `${MICON}<path d="M2.5 12V8.5M7 12V4M11.5 12V6.5"/></svg>`,
  /** Database cylinder — the Cache tab. */
  cache: `${MICON}<ellipse cx="7" cy="3.4" rx="4.6" ry="1.9"/><path d="M2.4 3.4v7.2c0 1.05 2.06 1.9 4.6 1.9s4.6-.85 4.6-1.9V3.4"/><path d="M2.4 7c0 1.05 2.06 1.9 4.6 1.9S11.6 8.05 11.6 7"/></svg>`,
  /** Memory chip — the Memory tab. */
  memory: `${MICON}<rect x="3.6" y="3.6" width="6.8" height="6.8" rx="1.2"/><path d="M5.6 3.6V1.9M8.4 3.6V1.9M5.6 12.1v-1.7M8.4 12.1v-1.7M3.6 5.6H1.9M3.6 8.4H1.9M12.1 5.6h-1.7M12.1 8.4h-1.7"/></svg>`,
  /** Lightning bolt — the Performance tab. */
  performance: `${MICON}<path d="M7.9 1.4 3.4 7.9h2.9L5.5 12.6 10.6 6H7.4l.5-4.6Z"/></svg>`,
  /** Lightbulb — the Insights tab. */
  insights: `${MICON}<path d="M4.7 9.3a3.8 3.8 0 1 1 4.6 0c-.5.4-.8 1-.8 1.6H5.5c0-.6-.3-1.2-.8-1.6Z"/><path d="M5.7 12.9h2.6"/></svg>`,
  /** Crosshair scan — spatial-index streaming mode (compact view). */
  stream: `${MICON}<circle cx="7" cy="7" r="3.1"/><path d="M7 1.6v1.9M7 10.5v1.9M1.6 7h1.9M10.5 7h1.9"/></svg>`,
  /** Closed box — direct (whole-dataset) loading mode (compact view). */
  box: `${MICON}<path d="M2.5 4.5 7 2l4.5 2.5v5L7 12 2.5 9.5v-5Z"/><path d="M2.5 4.5 7 7l4.5-2.5M7 7v5"/></svg>`,
  /** Check in a circle — all-clear states. */
  check: `${MICON}<circle cx="7" cy="7" r="5.4"/><path d="m4.6 7.2 1.7 1.7 3.1-3.5"/></svg>`,
  /** Filled dot — status/severity marker (colored by context). */
  dot: '<svg class="luxar-micon luxar-micon--dot" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="currentColor"/></svg>',
  /** Info circle — informational recommendations. */
  info: `${MICON}<circle cx="7" cy="7" r="5.4"/><path d="M7 6.6v3M7 4.3v.1"/></svg>`,
  /** Expand corners — grow the compact pill into the full panel. */
  expand: `${MICON}<path d="M5.4 1.8H1.8v3.6M8.6 1.8h3.6v3.6M5.4 12.2H1.8V8.6M8.6 12.2h3.6V8.6"/></svg>`,
  /** Warning triangle — failure states (was the ⚠ emoji). */
  alert: `${MICON}<path d="M7 2 12.8 12H1.2L7 2Z"/><path d="M7 6v2.6M7 10.4v.1"/></svg>`,
  /** Circle-slash — a disabled subsystem (was the 🚫 emoji). */
  blocked: `${MICON}<circle cx="7" cy="7" r="5.4"/><path d="M3.2 3.2l7.6 7.6"/></svg>`,
  /** Scene root — globe (scene-graph tree). */
  nodeScene: `${MICON}<circle cx="7" cy="7" r="5.4"/><path d="M1.6 7h10.8M7 1.6c1.9 1.5 1.9 9.3 0 10.8M7 1.6c-1.9 1.5-1.9 9.3 0 10.8"/></svg>`,
  /** Group — folder (scene-graph tree). */
  nodeGroup: `${MICON}<path d="M1.8 4h3l1.2 1.2h6.2V11a1.2 1.2 0 0 1-1.2 1.2H1.8V4Z"/></svg>`,
  /** Points — dot triplet (scene-graph tree). */
  nodePoints: `${MICON}<circle cx="4" cy="9.5" r="1.5"/><circle cx="9.8" cy="8.2" r="1.5"/><circle cx="6.4" cy="3.8" r="1.5"/></svg>`,
  /** Lines — open polyline (scene-graph tree). */
  nodeLines: `${MICON}<path d="M1.8 11.2 5.6 5.4l3 3.2 3.6-6"/></svg>`,
  /** GSplats — soft Gaussian: outer iso-contour + filled core (scene-graph tree). */
  nodeGsplats: `${MICON}<ellipse cx="7" cy="7" rx="5.2" ry="3.9" transform="rotate(-18 7 7)"/><circle cx="7" cy="7" r="1.4" fill="currentColor" stroke="none"/></svg>`,
  /** Mesh — hexagon (scene-graph tree; matches the old ⬡). */
  nodeMesh: `${MICON}<path d="M7 1.6 11.7 4.3v5.4L7 12.4 2.3 9.7V4.3L7 1.6Z"/></svg>`,
  /** kind=lod group — level slider pair (was the 🎚️ emoji). */
  kindLod: `${MICON}<path d="M1.8 4.4h10.4M1.8 9.6h10.4"/><circle cx="5" cy="4.4" r="1.6"/><circle cx="9" cy="9.6" r="1.6"/></svg>`,
  /** kind=partition group — BSP-split square (was the 🧩 emoji). */
  kindPartition: `${MICON}<rect x="2" y="2" width="10" height="10" rx="1.2"/><path d="M7.6 2v6.2M2 8.2h5.6M7.6 8.2H12" /></svg>`,
  /** Draw-order chip: `transparent` render bucket — two overlapping outlines (blended). */
  bucketTransparent: `${MICON}<rect x="1.8" y="4.2" width="7.2" height="7.2" rx="1"/><rect x="5" y="2.6" width="7.2" height="7.2" rx="1"/></svg>`,
  /** Draw-order chip: `opaque` render bucket — one filled square (depth-first, covers). */
  bucketOpaque: `${MICON}<rect x="3" y="3" width="8" height="8" rx="1" fill="currentColor"/></svg>`,
  /** LOD chip: next additive rung HELD (density guard or residency ceiling), not ⏳ streaming. */
  lodHeld: `${MICON}<path d="M5 3.4v7.2M9 3.4v7.2"/></svg>`,
  /** Density chip: a dot lattice with every other dot missing — a hashed subset is drawn. */
  densityThinned: `${MICON}<circle cx="3" cy="3" r="1.1" fill="currentColor" stroke="none"/><circle cx="11" cy="3" r="1.1" fill="currentColor" stroke="none"/><circle cx="7" cy="7" r="1.1" fill="currentColor" stroke="none"/><circle cx="3" cy="11" r="1.1" fill="currentColor" stroke="none"/><circle cx="11" cy="11" r="1.1" fill="currentColor" stroke="none"/><circle cx="7" cy="3" r="1.1" stroke-width="0.8"/><circle cx="3" cy="7" r="1.1" stroke-width="0.8"/><circle cx="11" cy="7" r="1.1" stroke-width="0.8"/><circle cx="7" cy="11" r="1.1" stroke-width="0.8"/></svg>`,
} as const;

/**
 * Color for a visible-count metric: neutral while data is on screen,
 * dimmed at zero. Counts are state, not identity — the previous
 * per-geometry palette (points green / lines orange / splats blue)
 * made a healthy lines card read as a permanent warning.
 */
export function countColorClass(count: number): string {
  return count > 0 ? '' : getColorClass('dimmed');
}

/**
 * Template for metric card component
 * @param colorClass - CSS class for color (e.g., 'luxar-color--success')
 * @param dataField - Optional data-field attribute for targeted DOM patching
 * @param tooltip - Optional hover tooltip describing the metric
 */
export function renderMetricCard(
  title: string,
  value: string | number,
  subtitle?: string,
  colorClass: string = '',
  size: 'small' | 'medium' | 'large' = 'medium',
  dataField?: string,
  tooltip?: string
): string {
  const fieldAttr = dataField ? ` data-field="${dataField}"` : '';
  const subFieldAttr = dataField ? ` data-field="${dataField}-sub"` : '';
  const tooltipAttr = tooltip ? ` title="${escapeHtml(tooltip)}"` : '';
  return `
    <div class="luxar-metric-card luxar-metric-card--${size}"${tooltipAttr}>
      ${title ? `<div class="luxar-metric-card__title">${title}</div>` : ''}
      <div class="luxar-metric-card__value luxar-metric-card__value--${size} ${colorClass}"${fieldAttr}>
        ${value}
      </div>
      ${subtitle ? `<div class="luxar-metric-card__subtitle"${subFieldAttr}>${subtitle}</div>` : ''}
    </div>
  `;
}

/**
 * Template for progress bar component
 * Note: Uses CSS custom properties for dynamic sizing
 * @param colorClass - CSS color class (e.g., 'luxar-color--success')
 */
export function renderProgressBar(
  percent: number,
  colorClass?: string,
  label?: string,
  height: number = 4
): string {
  const barColorClass = colorClass || getProgressColorClass(percent);
  // Using CSS custom properties for dynamic values that can't be pure CSS
  const trackStyle = `style="--bar-height: ${height}px; height: var(--bar-height); border-radius: calc(var(--bar-height) / 2);"`;
  const fillStyle = `style="width: ${Math.min(100, percent)}%; border-radius: calc(var(--bar-height, 4px) / 2);"`;

  return `
    <div class="luxar-progress-bar__container">
      <div class="luxar-progress-bar__track" ${trackStyle}>
        <div class="luxar-progress-bar__fill ${barColorClass}" ${fillStyle}></div>
      </div>
      ${label ? `<div class="luxar-progress-bar__label">${label}</div>` : ''}
    </div>
  `;
}

/**
 * Template for stat grid component
 * @param stats - Array of stats with colorClass for CSS class-based coloring
 */
export function renderStatGrid(
  stats: Array<{ label: string; value: string | number; colorClass?: string }>
): string {
  const cols = Math.min(3, stats.length);

  return `
    <div class="luxar-stat-grid luxar-stat-grid--cols-${cols}">
      ${stats
        .map((stat) => {
          return `
        <div class="luxar-stat-grid__item">
          <div class="luxar-stat-grid__value ${stat.colorClass || ''}">
            ${stat.value}
          </div>
          <div class="luxar-stat-grid__label">${stat.label}</div>
        </div>
      `;
        })
        .join('')}
    </div>
  `;
}

/**
 * Get CSS color class for progress percentage
 */
function getProgressColorClass(percent: number): string {
  if (percent <= 60) return getColorClass('success');
  if (percent <= 80) return getColorClass('warning');
  return getColorClass('error');
}
