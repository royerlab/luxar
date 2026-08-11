/**
 * Unit tests for the scene-identity banner (real DOM via jsdom).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  showSceneIdentityBanner,
  hideSceneIdentityBanner,
} from '../../../ui/scene-identity-banner';

const BANNER_ID = 'luxar-scene-identity-banner';

function banner(): HTMLElement | null {
  return document.getElementById(BANNER_ID);
}

afterEach(() => {
  hideSceneIdentityBanner();
  document.body.innerHTML = '';
});

describe('scene-identity banner', () => {
  it('shows a persistent changed banner with a Reload button', () => {
    showSceneIdentityBanner('changed');
    const el = banner();
    expect(el).not.toBeNull();
    expect(el!.getAttribute('role')).toBe('alert');
    expect(el!.textContent).toContain('different scene');
    const button = el!.querySelector('button');
    expect(button?.textContent).toBe('Reload');
  });

  it('reload button triggers a page reload', () => {
    const reload = vi.fn();
    // jsdom's location.reload is non-configurable on some versions; replace
    // the whole location value for the duration of the click.
    const original = window.location;
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...original, reload },
    });
    try {
      showSceneIdentityBanner('changed');
      banner()!.querySelector('button')!.click();
      expect(reload).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(window, 'location', {
        configurable: true,
        value: original,
      });
    }
  });

  it('unreachable banner has no reload button and can be cleared by kind', () => {
    showSceneIdentityBanner('unreachable');
    expect(banner()!.querySelector('button')).toBeNull();
    expect(banner()!.textContent).toContain('unreachable');
    // Kind-scoped hide of the OTHER kind is a no-op…
    hideSceneIdentityBanner('changed');
    expect(banner()).not.toBeNull();
    // …and of the matching kind removes it.
    hideSceneIdentityBanner('unreachable');
    expect(banner()).toBeNull();
  });

  it('changed outranks unreachable and is never demoted', () => {
    showSceneIdentityBanner('unreachable');
    showSceneIdentityBanner('changed');
    expect(banner()!.textContent).toContain('different scene');
    // A later unreachable verdict must not replace the changed banner.
    showSceneIdentityBanner('unreachable');
    expect(banner()!.textContent).toContain('different scene');
    // Kind-scoped unreachable clear (the recovery path) must not remove it.
    hideSceneIdentityBanner('unreachable');
    expect(banner()).not.toBeNull();
  });

  it('re-showing the same kind does not churn the DOM node', () => {
    showSceneIdentityBanner('unreachable');
    const first = banner();
    showSceneIdentityBanner('unreachable');
    expect(banner()).toBe(first);
  });

  it('only one banner exists at a time', () => {
    showSceneIdentityBanner('unreachable');
    showSceneIdentityBanner('changed');
    expect(document.querySelectorAll(`#${BANNER_ID}`).length).toBe(1);
  });
});
