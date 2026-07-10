/**
 * Line-icons for the Rendering Controls top-level folders.
 *
 * Same convention as the control rail's `RAIL_ICONS` (see ui/control-rail/):
 * inline SVG strings, `viewBox="0 0 24 24"`, `aria-hidden="true"`, NO inline
 * width/height/stroke/fill — all appearance comes from CSS
 * (`.luxar-gui__folder-icon svg` / `.luxar-gui__controller-icon svg`), which
 * strokes them with `currentColor` exactly like the rail buttons. Passed to
 * `Folder.addFolder(name, icon)` so folder headers get a monochrome glyph that
 * matches the rail instead of an emoji.
 */

import { RAIL_ICONS } from '../control-rail';

export const FOLDER_ICONS = {
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
  // Mouse pointer — Input (Settings popover).
  input:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 7-6.5 1.5L14 18l-3 1.3-1.5-6.5L5 3z"/></svg>',
  // Database stack — Caching (Settings popover).
  caching:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="8" ry="2.8"/><path d="M4 5.5v6c0 1.6 3.6 2.8 8 2.8s8-1.2 8-2.8v-6"/><path d="M4 11.5v6c0 1.6 3.6 2.8 8 2.8s8-1.2 8-2.8v-6"/></svg>',
  // Wrench — Advanced (Settings popover).
  advanced:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.5 6.5a4.5 4.5 0 0 0-6 6L3 18l3 3 5.5-5.5a4.5 4.5 0 0 0 6-6L14 13l-3-3 3.5-3.5z"/></svg>',
} as const;
