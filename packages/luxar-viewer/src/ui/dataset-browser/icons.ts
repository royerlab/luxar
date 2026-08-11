/**
 * Stroke-icon set for the dataset browser (inline SVG, currentColor stroke)
 * in the same 24×24 / stroke-1.7 register as the control rail's RAIL_ICONS —
 * replacing the legacy emoji glyphs (🌌 📁 📄) which rendered inconsistently
 * across platforms and clashed with the viewer's quiet-instrument language.
 *
 * @module ui/dataset-browser/icons
 */

export const BROWSER_ICONS: Record<string, string> = {
  /** Zarr dataset — database/stack glyph (distinct from every rail icon). */
  zarr: '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/><path d="M4.5 5.5v13c0 1.55 3.36 2.8 7.5 2.8s7.5-1.25 7.5-2.8v-13"/><path d="M4.5 12c0 1.55 3.36 2.8 7.5 2.8s7.5-1.25 7.5-2.8"/></svg>',
  /** Directory — same folder glyph as the rail's Data button (consistency). */
  folder:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h5l2 2h9v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7z"/></svg>',
  /** Plain file — document with folded corner. */
  file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6V3z"/><path d="M14 3v4h4"/></svg>',
  /** Root crumb — house (matches the rail's Home). */
  home: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11.5L12 4l9 7.5"/><path d="M5.5 9.8V20h13V9.8"/></svg>',
  /** Search field affordance. */
  search:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="M15.8 15.8L21 21"/></svg>',
  /** Close (✕) — replaces the text glyph. */
  close:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12"/><path d="M18 6L6 18"/></svg>',
  /** Enter-path-manually toggle in the breadcrumb row. */
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l1-4L16.5 4.5a2.12 2.12 0 0 1 3 3L8 19l-4 1z"/><path d="M14 7l3 3"/></svg>',
  /** Error state. */
  alert:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5L22 20H2L12 3.5z"/><line x1="12" y1="10" x2="12" y2="14.5"/><line x1="12" y1="17.2" x2="12" y2="17.21"/></svg>',
  /** Directory row affordance — "this navigates deeper". */
  chevron: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg>',
};
