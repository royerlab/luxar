/**
 * Unit tests for `renderFailedLoadsBanner` — the Overview tab's failed-load
 * warning banner with its Retry action (the monitor-side trigger for
 * `SceneLoader.retryAllFailedLoaders`).
 */

import { describe, it, expect } from 'vitest';
import { renderFailedLoadsBanner } from '../../../../ui/data-loading-monitor/templates/overview';

describe('renderFailedLoadsBanner', () => {
  it('renders nothing when no loads have failed (the common case)', () => {
    expect(renderFailedLoadsBanner([], false)).toBe('');
  });

  it('renders the count, the failed paths in the tooltip, and an enabled Retry button', () => {
    const html = renderFailedLoadsBanner(['/points/a', '/gsplats/b'], false);
    expect(html).toContain('2 failed loads');
    expect(html).toContain('data-action="retryFailedLoads"');
    expect(html).toContain('/points/a');
    expect(html).toContain('/gsplats/b');
    expect(html).not.toContain('disabled');
    expect(html).toContain('>Retry<');
    expect(html).toContain('Retryable loader failures and latched LOD branches');
  });

  it('leads with the inline alert glyph rather than a platform-dependent emoji', () => {
    const html = renderFailedLoadsBanner(['/p'], false);
    expect(html).toContain('<span class="luxar-failed-loads__label"><svg class="luxar-micon"');
    expect(html).not.toContain('⚠');
  });

  it('singularizes for one failure', () => {
    expect(renderFailedLoadsBanner(['/p'], false)).toContain('1 failed load<');
  });

  it('disables the button and shows progress while a retry batch is in flight', () => {
    const html = renderFailedLoadsBanner(['/p'], true);
    expect(html).toContain('disabled');
    expect(html).toContain('Retrying…');
  });

  it('escapes markup in failed paths (tooltip is attribute-injected)', () => {
    const html = renderFailedLoadsBanner(['/x"><script>alert(1)</script>'], false);
    expect(html).not.toContain('<script>');
  });
});
