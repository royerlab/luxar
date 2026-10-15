// @vitest-environment jsdom
/**
 * The notifier backend reports data-server reachability after every
 * scene-identity banner change — `false` exactly while the `unreachable`
 * banner stands. LuxarApp uses it to stop the render loop sustaining itself
 * while the server is gone.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { installNotifierBackend } from '../../../../../core/app/lifecycle/notifier-backend';
import { clearNotifierBackend, notifier } from '../../../../../utils/cross-layer/notifier';
import { hideSceneIdentityBanner } from '../../../../../ui/scene-identity-banner';

afterEach(() => {
  hideSceneIdentityBanner();
  clearNotifierBackend();
  document.body.innerHTML = '';
});

describe('installNotifierBackend data-server reachability', () => {
  it('reports unreachable while the unreachable banner stands, reachable on recovery', () => {
    const onReachability = vi.fn();
    installNotifierBackend(() => undefined, onReachability);

    notifier.showSceneIdentityBanner('unreachable', { connectionLost: true });
    expect(onReachability).toHaveBeenLastCalledWith(false);

    notifier.hideSceneIdentityBanner('unreachable');
    expect(onReachability).toHaveBeenLastCalledWith(true);
  });

  it('treats a changed verdict as reachable (something answered)', () => {
    const onReachability = vi.fn();
    installNotifierBackend(() => undefined, onReachability);

    notifier.showSceneIdentityBanner('unreachable', { connectionLost: true });
    notifier.showSceneIdentityBanner('changed');
    expect(onReachability).toHaveBeenLastCalledWith(true);

    // A flapping server cannot demote `changed` back to unreachable.
    notifier.showSceneIdentityBanner('unreachable');
    expect(onReachability).toHaveBeenLastCalledWith(true);
  });

  it('reports reachable when a dataset switch clears the banner', () => {
    const onReachability = vi.fn();
    installNotifierBackend(() => undefined, onReachability);

    notifier.showSceneIdentityBanner('unreachable', { connectionLost: true });
    notifier.hideSceneIdentityBanner();
    expect(onReachability).toHaveBeenLastCalledWith(true);
  });

  it('keeps reporting reachable for an HTTP-status failure (banner still shows)', () => {
    const onReachability = vi.fn();
    installNotifierBackend(() => undefined, onReachability);

    // An expired presigned credential / rate limit / 5xx: not "the server is gone".
    notifier.showSceneIdentityBanner('unreachable', { connectionLost: false });
    expect(document.getElementById('luxar-scene-identity-banner')).not.toBeNull();
    expect(onReachability).toHaveBeenLastCalledWith(true);

    // The next failure is a lost connection: now it is unreachable.
    notifier.showSceneIdentityBanner('unreachable', { connectionLost: true });
    expect(onReachability).toHaveBeenLastCalledWith(false);

    // And back to a status answer: something is answering again.
    notifier.showSceneIdentityBanner('unreachable', { connectionLost: false });
    expect(onReachability).toHaveBeenLastCalledWith(true);
  });

  it('works without a reachability receiver', () => {
    installNotifierBackend(() => undefined);
    expect(() => notifier.showSceneIdentityBanner('unreachable')).not.toThrow();
    expect(document.getElementById('luxar-scene-identity-banner')).not.toBeNull();
  });
});
