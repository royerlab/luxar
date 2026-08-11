/**
 * Persistent scene-identity banner — "this tab no longer shows what its
 * address serves".
 *
 * Local dev/demo servers share ports and come and go, so a long-lived tab can
 * end up fronting a DIFFERENT scene than the one it loaded (another server
 * took the port), or a dead one. The scene-identity watchdog
 * (`data/scene-identity-watchdog.ts`) detects both; this module renders the
 * verdict as a top-center banner:
 *
 * - `changed`: the address now serves a different scene — permanent, with a
 *   Reload button (the only honest remedy). Never auto-dismisses: unlike a
 *   toast, the tab stays misleading for as long as it stays open.
 * - `unreachable`: the data server stopped responding — informational, and
 *   removed automatically by the watchdog when the server comes back.
 *
 * One banner at a time: `changed` outranks (and replaces) `unreachable`.
 * Deliberately NOT the error dialog: that auto-dismisses, traps focus, and
 * blocks the canvas — wrong shape for a standing condition the user may
 * consciously ignore while inspecting the already-loaded scene.
 */

import { getViewerContainer } from '../utils/viewer-container';

const BANNER_ID = 'luxar-scene-identity-banner';

/**
 * Which standing condition the banner reports: the address now serves a
 * different scene (`changed`, permanent + Reload) or stopped answering
 * usefully (`unreachable`, cleared on recovery).
 */
export type SceneIdentityBannerKind = 'changed' | 'unreachable';

const MESSAGES: Record<SceneIdentityBannerKind, string> = {
  changed: 'This address now serves a different scene — the view below is stale.',
  unreachable: 'Data server unreachable — it may have been stopped.',
};

/** The kind currently displayed, or null when no banner is up. */
let shownKind: SceneIdentityBannerKind | null = null;

/**
 * Show (or replace) the scene-identity banner.
 *
 * Re-showing the same kind is a no-op so the watchdog can call this on
 * every failed probe without DOM churn; `changed` replaces `unreachable`
 * but never the reverse (a flapping server must not demote a definitive
 * scene-swap verdict).
 */
export function showSceneIdentityBanner(kind: SceneIdentityBannerKind): void {
  if (shownKind === kind) return;
  if (shownKind === 'changed' && kind === 'unreachable') return;
  hideSceneIdentityBanner();

  const banner = document.createElement('div');
  banner.id = BANNER_ID;
  banner.className = 'luxar-glass-surface';
  banner.setAttribute('role', 'alert');
  banner.style.cssText = [
    'position: absolute',
    'top: 12px',
    'left: 50%',
    'transform: translateX(-50%)',
    'z-index: 10000',
    'display: flex',
    'align-items: center',
    'gap: 12px',
    'padding: 10px 16px',
    'border-radius: 8px',
    'font: 13px system-ui, sans-serif',
    'color: #fff',
    'pointer-events: auto',
    'max-width: min(90%, 640px)',
    kind === 'changed'
      ? 'background: rgba(160, 44, 44, 0.92); border: 1px solid rgba(255,120,120,0.5)'
      : 'background: rgba(150, 110, 20, 0.92); border: 1px solid rgba(255,200,90,0.5)',
  ].join(';');

  const text = document.createElement('span');
  text.textContent = `${kind === 'changed' ? '⚠️' : '\u{1F50C}'} ${MESSAGES[kind]}`;
  banner.appendChild(text);

  if (kind === 'changed') {
    const reload = document.createElement('button');
    reload.textContent = 'Reload';
    reload.style.cssText = [
      'padding: 4px 14px',
      'border-radius: 6px',
      'border: 1px solid rgba(255,255,255,0.6)',
      'background: rgba(255,255,255,0.15)',
      'color: #fff',
      'font: inherit',
      'cursor: pointer',
      'white-space: nowrap',
    ].join(';');
    reload.addEventListener('click', () => window.location.reload());
    banner.appendChild(reload);
  }

  getViewerContainer().appendChild(banner);
  shownKind = kind;
}

/**
 * Remove the banner. `onlyKind` restricts removal to that kind — the
 * watchdog's recovery path clears `unreachable` without being able to
 * accidentally clear a standing `changed` verdict.
 */
export function hideSceneIdentityBanner(onlyKind?: SceneIdentityBannerKind): void {
  if (shownKind === null) return;
  if (onlyKind !== undefined && shownKind !== onlyKind) return;
  document.getElementById(BANNER_ID)?.remove();
  shownKind = null;
}
