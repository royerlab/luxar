/**
 * Data Loading Monitor Style System
 *
 * Centralized styles following UI_DESIGN.md guidelines.
 * This ensures consistent appearance across all monitor components.
 */

/**
 * Core color palette following UI_DESIGN.md semantic colors
 */
export const MonitorColors = {
  // Semantic colors
  success: '#4CAF50',
  warning: '#FFC107',
  error: '#f44336',
  info: '#2196F3',
  secondary: '#9C27B0',

  // Text colors
  primaryText: '#e0e0e0',
  secondaryText: '#888',
  muted: 'rgba(255, 255, 255, 0.6)',
  dimmed: 'rgba(255, 255, 255, 0.4)',

  // Background colors
  panelBg: 'rgba(30, 30, 30, 0.95)',
  sectionBg: 'rgba(0, 0, 0, 0.3)',
  hoverBg: 'rgba(40, 40, 40, 0.9)',

  // Special colors for cache visualization
  cacheHot: '#ff6b6b',
  cacheWarm: '#FFC107',
  cacheCold: '#4CAF50',

  // Grid/separator colors
  separator: 'rgba(255, 255, 255, 0.1)',
  separatorStrong: 'rgba(255, 255, 255, 0.2)',
} as const;

/**
 * Typography styles following UI_DESIGN.md specifications
 */
export const MonitorTypography = {
  fontFamily:
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif',
  fontFamilyMono: 'monospace',

  // Font sizes
  title: { fontSize: '14px', fontWeight: 'bold' },
  sectionHeader: { fontSize: '12px', fontWeight: 600 },
  body: { fontSize: '11px' },
  small: { fontSize: '10px' },
  tiny: { fontSize: '9px' },

  // Line heights
  compact: { lineHeight: 1.2 },
  normal: { lineHeight: 1.4 },
  relaxed: { lineHeight: 1.6 },
} as const;

/**
 * Spacing values following UI_DESIGN.md
 */
export const MonitorSpacing = {
  // Panel spacing
  panelPadding: 15,
  panelMargin: 20,

  // Section spacing
  sectionPadding: 10,
  sectionGap: 15,

  // Element spacing
  elementGap: 8,
  compactGap: 5,
  tinyGap: 2,

  // Border and separator spacing
  borderPadding: 6,
} as const;

/**
 * Visual effects following UI_DESIGN.md
 */
export const MonitorEffects = {
  // Backdrop and shadows
  backdropBlur: 'blur(10px)',
  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
  boxShadowStrong: '0 8px 32px rgba(0, 0, 0, 0.4)',

  // Border radius
  borderRadius: 8,
  borderRadiusSmall: 4,
  borderRadiusLarge: 12,

  // Transitions
  transition: 'all 0.2s ease',
  transitionFast: 'all 0.1s ease',
  transitionSlow: 'all 0.3s ease',
} as const;

/**
 * Z-index hierarchy
 */
export const MonitorZIndex = {
  base: 1000, // Monitor panels
  overlay: 1100, // Overlays and tooltips
  modal: 1200, // Modal dialogs
  critical: 2000, // Critical overlays
} as const;

/**
 * Pre-composed style objects for common components
 */
export const MonitorStyles = {
  // Main panel styles
  panel: {
    base: `
      background: ${MonitorColors.panelBg};
      backdrop-filter: ${MonitorEffects.backdropBlur};
      border-radius: ${MonitorEffects.borderRadius}px;
      box-shadow: ${MonitorEffects.boxShadow};
      font-family: ${MonitorTypography.fontFamily};
      color: ${MonitorColors.primaryText};
      z-index: ${MonitorZIndex.base};
    `,
    compact: `
      padding: ${MonitorSpacing.compactGap}px ${MonitorSpacing.sectionPadding}px;
    `,
    normal: `
      padding: ${MonitorSpacing.panelPadding}px;
    `,
    expanded: `
      padding: 0;
      overflow: hidden;
    `,
  },

  // Header styles
  header: `
    padding: ${MonitorSpacing.sectionPadding}px ${MonitorSpacing.panelPadding}px;
    border-bottom: 1px solid ${MonitorColors.separator};
    display: flex;
    justify-content: space-between;
    align-items: center;
  `,

  // Section styles
  section: {
    base: `
      padding: ${MonitorSpacing.sectionPadding}px;
      margin-bottom: ${MonitorSpacing.sectionGap}px;
    `,
    nested: `
      background: ${MonitorColors.sectionBg};
      border-radius: ${MonitorEffects.borderRadiusSmall}px;
      padding: ${MonitorSpacing.sectionPadding}px;
    `,
    separator: `
      border-bottom: 1px solid ${MonitorColors.separator};
      padding-bottom: ${MonitorSpacing.borderPadding}px;
      margin-bottom: ${MonitorSpacing.sectionPadding}px;
    `,
  },

  // Typography styles
  text: {
    title: `
      font-size: ${MonitorTypography.title.fontSize};
      font-weight: ${MonitorTypography.title.fontWeight};
      color: ${MonitorColors.primaryText};
      margin: 0;
    `,
    sectionHeader: `
      font-size: ${MonitorTypography.sectionHeader.fontSize};
      font-weight: ${MonitorTypography.sectionHeader.fontWeight};
      color: ${MonitorColors.success};
      margin: 0 0 ${MonitorSpacing.elementGap}px 0;
    `,
    body: `
      font-size: ${MonitorTypography.body.fontSize};
      color: ${MonitorColors.primaryText};
      line-height: ${MonitorTypography.normal.lineHeight};
    `,
    small: `
      font-size: ${MonitorTypography.small.fontSize};
      color: ${MonitorColors.secondaryText};
    `,
    muted: `
      color: ${MonitorColors.muted};
      opacity: 0.8;
    `,
    mono: `
      font-family: ${MonitorTypography.fontFamilyMono};
    `,
  },

  // Button styles
  button: {
    base: `
      background: none;
      border: none;
      color: ${MonitorColors.secondaryText};
      cursor: pointer;
      padding: ${MonitorSpacing.compactGap}px;
      transition: ${MonitorEffects.transition};
      font-size: ${MonitorTypography.body.fontSize};
    `,
    close: `
      background: none;
      border: none;
      color: ${MonitorColors.secondaryText};
      font-size: 24px;
      cursor: pointer;
      padding: 0;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: ${MonitorEffects.transition};
    `,
    tab: {
      base: `
        flex: 1;
        padding: ${MonitorSpacing.elementGap}px;
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: ${MonitorTypography.body.fontSize};
        transition: ${MonitorEffects.transition};
      `,
      active: `
        background: rgba(255, 255, 255, 0.1);
        color: #fff;
      `,
      inactive: `
        background: transparent;
        color: ${MonitorColors.secondaryText};
      `,
    },
  },

  // Stat card styles
  statCard: `
    background: ${MonitorColors.sectionBg};
    padding: ${MonitorSpacing.sectionPadding}px;
    border-radius: ${MonitorEffects.borderRadiusSmall}px;
    
    label {
      display: block;
      font-size: ${MonitorTypography.small.fontSize};
      color: ${MonitorColors.muted};
      margin-bottom: ${MonitorSpacing.compactGap}px;
    }
    
    value {
      display: block;
      font-size: 18px;
      font-weight: bold;
      color: ${MonitorColors.success};
    }
    
    detail {
      display: block;
      font-size: ${MonitorTypography.small.fontSize};
      color: ${MonitorColors.muted};
      margin-top: ${MonitorSpacing.tinyGap}px;
    }
  `,

  // Grid layouts
  grid: {
    stats: `
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: ${MonitorSpacing.elementGap}px;
    `,
    cache: `
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: ${MonitorSpacing.elementGap}px;
    `,
  },

  // Progress bars
  progressBar: {
    container: `
      width: 100%;
      height: 4px;
      background: ${MonitorColors.sectionBg};
      border-radius: 2px;
      overflow: hidden;
    `,
    fill: (percent: number, color: string = MonitorColors.success) => `
      width: ${percent}%;
      height: 100%;
      background: ${color};
      transition: width 0.3s ease;
    `,
  },

  // Tooltips
  tooltip: `
    position: absolute;
    background: rgba(0, 0, 0, 0.9);
    color: ${MonitorColors.primaryText};
    padding: ${MonitorSpacing.compactGap}px ${MonitorSpacing.elementGap}px;
    border-radius: ${MonitorEffects.borderRadiusSmall}px;
    font-size: ${MonitorTypography.small.fontSize};
    z-index: ${MonitorZIndex.overlay};
    pointer-events: none;
    white-space: nowrap;
  `,

  // Alert styles
  alert: {
    success: `
      background: rgba(76, 175, 80, 0.2);
      color: ${MonitorColors.success};
      padding: ${MonitorSpacing.elementGap}px;
      border-radius: ${MonitorEffects.borderRadiusSmall}px;
      border-left: 3px solid ${MonitorColors.success};
    `,
    warning: `
      background: rgba(255, 193, 7, 0.2);
      color: ${MonitorColors.warning};
      padding: ${MonitorSpacing.elementGap}px;
      border-radius: ${MonitorEffects.borderRadiusSmall}px;
      border-left: 3px solid ${MonitorColors.warning};
    `,
    error: `
      background: rgba(244, 67, 54, 0.2);
      color: ${MonitorColors.error};
      padding: ${MonitorSpacing.elementGap}px;
      border-radius: ${MonitorEffects.borderRadiusSmall}px;
      border-left: 3px solid ${MonitorColors.error};
    `,
  },
} as const;

/**
 * Helper functions for dynamic styling
 */
export const MonitorStyleHelpers = {
  /**
   * Get color based on performance metric
   */
  getPerformanceColor(value: number, goodThreshold: number, badThreshold: number): string {
    if (value >= goodThreshold) return MonitorColors.success;
    if (value >= badThreshold) return MonitorColors.warning;
    return MonitorColors.error;
  },

  /**
   * Get cache temperature color
   */
  getCacheTemperatureColor(accessCount: number, maxCount: number): string {
    const ratio = accessCount / maxCount;
    if (ratio > 0.7) return MonitorColors.cacheHot;
    if (ratio > 0.3) return MonitorColors.cacheWarm;
    return MonitorColors.cacheCold;
  },

  /**
   * Format inline style object to CSS string
   */
  styleObjectToString(style: Record<string, any>): string {
    return Object.entries(style)
      .map(([key, value]) => {
        const cssKey = key.replace(/([A-Z])/g, '-$1').toLowerCase();
        return `${cssKey}: ${value}`;
      })
      .join('; ');
  },

  /**
   * Apply hover effect styles
   */
  applyHoverEffect(element: HTMLElement): void {
    element.style.transition = MonitorEffects.transition;
    element.addEventListener('mouseenter', () => {
      element.style.opacity = '0.8';
      element.style.transform = 'scale(1.02)';
    });
    element.addEventListener('mouseleave', () => {
      element.style.opacity = '1';
      element.style.transform = 'scale(1)';
    });
  },
} as const;

/**
 * CSS class definitions for use with stylesheets
 */
export const MonitorCSSClasses = `
  .luxar-data-monitor {
    ${MonitorStyles.panel.base}
  }
  
  .monitor-compact {
    ${MonitorStyles.panel.compact}
  }
  
  .monitor-expanded {
    ${MonitorStyles.panel.expanded}
  }
  
  .monitor-header {
    ${MonitorStyles.header}
  }
  
  .monitor-section {
    ${MonitorStyles.section.base}
  }
  
  .monitor-stat-card {
    ${MonitorStyles.statCard}
  }
  
  .monitor-tooltip {
    ${MonitorStyles.tooltip}
  }
`;
