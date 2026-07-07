/**
 * Default control-rail icon set (inline SVG, currentColor stroke) keyed by rail
 * item id. Kept in its own module so the rail class + item builders stay
 * readable and the icon data is easy to scan/extend.
 *
 * @module ui/control-rail/icons
 */

export const RAIL_ICONS: Record<string, string> = {
  help: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7"/><line x1="12" y1="17" x2="12" y2="17.01"/></svg>',
  home: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11.5L12 4l9 7.5"/><path d="M5.5 9.8V20h13V9.8"/><path d="M10 20v-5.5h4V20"/></svg>',
  dims: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 3v18M4 7.5l8 4.5 8-4.5"/></svg>',
  render:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2.2"/><line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2.2"/></svg>',
  layers:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
  perf: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18a8 8 0 0 1 16 0"/><line x1="12" y1="18" x2="16" y2="11"/></svg>',
  data: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h5l2 2h9v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7z"/></svg>',
  monitor: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l2 6 4-13 2 7h6"/></svg>',
  recording:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
  screenshot:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg>',
  logs: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3"/><line x1="13" y1="15" x2="17" y2="15"/></svg>',
  view: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>',
  scalebar:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="9" width="18" height="6" rx="1"/><path d="M7 9v3M11 9v4M15 9v3M19 9v4"/></svg>',
  legend:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="4" width="5" height="16" rx="1"/><line x1="13" y1="6" x2="18" y2="6"/><line x1="13" y1="12" x2="18" y2="12"/><line x1="13" y1="18" x2="18" y2="18"/></svg>',
  overlays:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="12" height="12" rx="1"/><rect x="9" y="9" width="11" height="11" rx="1"/></svg>',
  cinematic:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l7-7"/><path d="M15 3l1.2 3.3L19.5 7.5l-3.3 1.2L15 12l-1.2-3.3L10.5 7.5l3.3-1.2z"/></svg>',
  // Navigation modes — the rail button swaps between these to mirror the live
  // camera control type (orbit / fly / ortho). Same 24×24 / currentColor style.
  navOrbit:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><ellipse cx="12" cy="12" rx="10" ry="4.2"/></svg>',
  navFly:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11l18-7-7 18-2.5-8L3 11z"/></svg>',
  // Ortho: a 2×2 quadrant grid — the classic orthographic multi-view glyph.
  // Deliberately NOT a cube (which would collide with the dimensions icon).
  navOrtho:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="6.5" height="6.5" rx="1"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1"/></svg>',
  // Settings (gear) — houses theme + other viewer-wide preferences.
  settings:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
};
