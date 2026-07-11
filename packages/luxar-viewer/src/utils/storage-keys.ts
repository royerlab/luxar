/**
 * Single source of truth for every `localStorage` key the viewer reads or
 * writes.
 *
 * Keys are dot-namespaced under `luxar.*` so they can never collide with a
 * host page's storage. Adding a new persistent setting? Add the key here so
 * future readers can find every storage touch in one file.
 */

/** Sanitize an arbitrary identifier (e.g. scene URL) for use inside a key. */
function sanitizeKeySegment(segment: string): string {
  return segment.replace(/[^a-zA-Z0-9-_]/g, '_');
}

// [api.md OOS] Frozen at runtime so embedders cannot mutate the
// namespacing contract. `as const` gives TypeScript readonly tags but
// the runtime object was still a plain mutable record — a host page
// doing `StorageKeys.theme = 'pwned'` would have corrupted every
// subsequent read/write. Object.freeze makes such mutations throw in
// strict mode and no-op in sloppy mode, locking the contract at
// module load.
export const StorageKeys = Object.freeze({
  /** Active theme id (`'dark' | 'light' | 'frosted-glass' | 'liquid-glass'`). */
  theme: 'luxar.theme',
  /** Persisted debug-mode toggle (mirrors `?debug` URL parameter). */
  debug: 'luxar.debug',
  /** Global viewer preferences (Settings popover) — see config/user-settings.ts. */
  settings: 'luxar.settings',
  /** Per-scene rendering settings (bloom, HDR, lens, etc). */
  rendering(sceneId: string): string {
    return `luxar.rendering.${sanitizeKeySegment(sceneId)}`;
  },
} as const);
