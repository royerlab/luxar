/**
 * Line-icons for the Rendering Controls top-level folders.
 *
 * Same convention as the control rail's `RAIL_ICONS` (see ui/control-rail.ts):
 * inline SVG strings, `viewBox="0 0 24 24"`, `aria-hidden="true"`, NO inline
 * width/height/stroke/fill — all appearance comes from CSS
 * (`.luxar-gui__folder-icon svg` / `.luxar-gui__controller-icon svg`), which
 * strokes them with `currentColor` exactly like the rail buttons. Passed to
 * `Folder.addFolder(name, icon)` so folder headers get a monochrome glyph that
 * matches the rail instead of an emoji.
 */

import { RAIL_ICONS } from '../control-rail';

export const FOLDER_ICONS = {
  // Compass — Navigation.
  navigation:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M15.6 8.4l-2.2 5-5 2.2 2.2-5z"/></svg>',
  // Still camera (reused from the rail's Screenshot glyph) — Camera.
  camera: RAIL_ICONS.screenshot,
  // Sun with rays — HDR / exposure.
  hdr: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  // Twin sparkles — Anti-Aliasing.
  antiAliasing:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 3l1.7 4.3L17 9l-4.3 1.7L11 15l-1.7-4.3L5 9l4.3-1.7z"/><path d="M18 14l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></svg>',
  // Film strip — Post-Processing.
  postProcessing:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7.5 4v16M16.5 4v16M3 8h4.5M3 12h4.5M3 16h4.5M16.5 8H21M16.5 12H21M16.5 16H21"/></svg>',
  // Speed gauge (reused from the rail's Performance glyph) — Performance.
  performance: RAIL_ICONS.perf,
  // Paint palette — Theme.
  theme:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 0 0 0 18c1.6 0 2-1.2 1.2-2.1-.8-1 .1-2.4 1.3-2.4H17a4 4 0 0 0 4-4C21 6 17 3 12 3z"/><circle cx="7.5" cy="12" r="1.1"/><circle cx="9.5" cy="8" r="1.1"/><circle cx="14.5" cy="8" r="1.1"/></svg>',
  // Sparkle-trail (reused from the rail's Cinematic glyph) — Cinematic Mode.
  cinematic: RAIL_ICONS.cinematic,
} as const;
