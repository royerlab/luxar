/**
 * Insights-tab recommendation templates.
 */

import type { Recommendation } from '../../../types/data-monitor-types';
import { escapeHtml } from '../../../utils/escape-html';
import { MONITOR_ICONS, getColorClass } from './primitives';

/**
 * Template for recommendation item
 */
export function renderRecommendation(rec: Recommendation): string {
  const severityIcons = {
    error: `<span class="${getColorClass('error')}">${MONITOR_ICONS.dot}</span>`,
    warning: `<span class="${getColorClass('warning')}">${MONITOR_ICONS.dot}</span>`,
    info: `<span class="${getColorClass('info')}">${MONITOR_ICONS.info}</span>`,
  };

  return `
    <div class="luxar-recommendation luxar-recommendation--${rec.severity}">
      <div class="luxar-recommendation__header">
        <span class="luxar-recommendation__icon">${severityIcons[rec.severity]}</span>
        <strong class="luxar-recommendation__title">${escapeHtml(rec.title)}</strong>
      </div>
      <div class="luxar-recommendation__message">
        ${escapeHtml(rec.message)}
      </div>
      ${
        rec.suggestion
          ? `
        <div class="luxar-recommendation__message luxar-suggestion">
          ${MONITOR_ICONS.insights} ${escapeHtml(rec.suggestion)}
        </div>
      `
          : ''
      }
    </div>
  `;
}

/**
 * Template for insights tab content
 */
export function renderInsightsContent(recommendations: Recommendation[]): string {
  if (recommendations.length === 0) {
    return `
      <div class="luxar-data-monitor__empty luxar-data-monitor__empty--faded luxar-monitor-allclear">
        <span class="luxar-monitor-allclear__icon">${MONITOR_ICONS.check}</span>
        <div class="luxar-monitor-allclear__title">All clear</div>
        <div class="luxar-monitor-allclear__sub">No loading, caching, or performance issues detected.</div>
      </div>
    `;
  }

  return `
    <div class="luxar-tab-content--insights">
      ${recommendations.map((rec) => renderRecommendation(rec)).join('')}
    </div>
  `;
}
