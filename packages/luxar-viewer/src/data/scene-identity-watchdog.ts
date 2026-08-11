/**
 * Scene-identity watchdog — detects when a tab's `?src=` address stops
 * serving the scene the tab loaded.
 *
 * Local demo/dev servers share ports and come and go: server A (scene A)
 * dies, server B (scene B) later binds the same port, and every tab that
 * loaded scene A now fronts scene B's data with full confidence — the
 * "I started demo B and got demo A" trap. The load-time cache validation
 * protects fresh loads; nothing protected a tab that simply stayed open.
 *
 * This watchdog re-fetches the dataset's root `.zattrs` (cache-bypassing)
 * on an interval and whenever the tab regains focus/visibility — the exact
 * moment a user returns to a stale tab — and compares scene identity:
 *
 * - `content_hash` when the loaded scene had one (every compiler-written
 *   scene does): the same identity stamp the L2 cache validates against.
 * - Otherwise the raw `.zattrs` text, baselined on the first successful
 *   probe (hash-less bare nodes).
 *
 * Verdicts surface through the cross-layer notifier as a persistent banner
 * (`ui/scene-identity-banner.ts`):
 *
 * - **changed** (different hash, unparseable body, or HTTP error status):
 *   terminal — polling stops, the banner offers Reload. An HTTP error is
 *   "changed" rather than "unreachable" because something IS answering the
 *   address; whatever it is, it no longer serves this scene.
 * - **unreachable** (fetch throws — server gone): shown only after two
 *   consecutive failures so a single blip stays silent, and cleared
 *   automatically when the server answers again (a recovered server that
 *   serves a different scene escalates straight to `changed`).
 *
 * Only `http(s)` sources are watched — there is nothing to race against on
 * an in-memory or file-backed store. All timers/listeners are removed by
 * `dispose()`, which the SceneLoader calls on dataset switch and teardown.
 */

import { log, Modules } from '../utils/log';
import { notifier } from '../utils/cross-layer/notifier';

/** Periodic probe cadence. Focus/visibility probes fire immediately. */
const CHECK_INTERVAL_MS = 15_000;
/** Minimum spacing between probes (guards focus+visibility double-fire). */
const MIN_PROBE_SPACING_MS = 2_000;
/** Consecutive fetch failures before the `unreachable` banner shows. */
const UNREACHABLE_THRESHOLD = 2;

export interface SceneIdentityWatchdogOptions {
  /** Dataset base URL (the `?src=` value, trailing slash tolerated). */
  datasetUrl: string;
  /** `content_hash` of the scene actually loaded (null when absent). */
  expectedContentHash: string | null;
  /** Probe cadence override (tests). */
  intervalMs?: number;
  /** Fetch override (tests). */
  fetchImpl?: typeof fetch;
}

type Verdict = 'ok' | 'changed' | 'unreachable';

export class SceneIdentityWatchdog {
  private readonly url: string;
  private readonly expectedHash: string | null;
  private readonly intervalMs: number;
  private readonly fetchImpl: typeof fetch;

  /** Raw-text baseline for hash-less scenes (set by the first OK probe). */
  private textBaseline: string | null = null;
  private consecutiveFailures = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastProbeAt = 0;
  private probeInFlight = false;
  private disposed = false;
  /** Terminal once the address demonstrably serves something else. */
  private changed = false;

  private readonly onWake = (): void => {
    if (document.visibilityState === 'hidden') return;
    void this.probe();
  };

  constructor(opts: SceneIdentityWatchdogOptions) {
    this.url = opts.datasetUrl.replace(/\/+$/, '');
    this.expectedHash = opts.expectedContentHash;
    this.intervalMs = opts.intervalMs ?? CHECK_INTERVAL_MS;
    // Bind: an unbound window.fetch reference throws "Illegal invocation".
    this.fetchImpl = opts.fetchImpl ?? fetch.bind(globalThis);
  }

  /** Whether this dataset URL is one the watchdog can meaningfully watch. */
  static isWatchable(datasetUrl: string): boolean {
    return /^https?:\/\//i.test(datasetUrl);
  }

  /**
   * Begin watching. No immediate probe: the caller baselines identity from
   * the attrs it actually loaded, and an instant re-fetch would only race
   * the load it just finished.
   */
  start(): void {
    if (this.disposed || this.timer !== null) return;
    this.timer = setInterval(() => void this.probe(), this.intervalMs);
    // Returning to a backgrounded tab is the highest-risk moment for
    // staleness (interval timers are throttled while hidden) — check now.
    window.addEventListener('focus', this.onWake);
    document.addEventListener('visibilitychange', this.onWake);
  }

  dispose(): void {
    this.disposed = true;
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    window.removeEventListener('focus', this.onWake);
    document.removeEventListener('visibilitychange', this.onWake);
    // A dataset switch must not leave the previous dataset's verdict up.
    notifier.hideSceneIdentityBanner();
  }

  /** One identity probe; exposed for tests (never rejects). */
  async probe(): Promise<void> {
    const now = Date.now();
    if (
      this.disposed ||
      this.changed ||
      this.probeInFlight ||
      now - this.lastProbeAt < MIN_PROBE_SPACING_MS
    ) {
      return;
    }
    this.probeInFlight = true;
    this.lastProbeAt = now;
    try {
      this.apply(await this.fetchVerdict());
    } finally {
      this.probeInFlight = false;
    }
  }

  private async fetchVerdict(): Promise<Verdict> {
    let body: string;
    try {
      const res = await this.fetchImpl(`${this.url}/.zattrs`, {
        cache: 'no-store',
      });
      if (!res.ok) return 'changed';
      body = await res.text();
    } catch {
      return 'unreachable';
    }

    if (this.expectedHash !== null) {
      try {
        const attrs = JSON.parse(body) as { content_hash?: unknown };
        return attrs.content_hash === this.expectedHash ? 'ok' : 'changed';
      } catch {
        return 'changed';
      }
    }
    // Hash-less scene: first successful probe defines the identity.
    if (this.textBaseline === null) {
      this.textBaseline = body;
      return 'ok';
    }
    return body === this.textBaseline ? 'ok' : 'changed';
  }

  private apply(verdict: Verdict): void {
    if (this.disposed) return;
    switch (verdict) {
      case 'ok':
        this.consecutiveFailures = 0;
        notifier.hideSceneIdentityBanner('unreachable');
        return;
      case 'unreachable':
        this.consecutiveFailures += 1;
        if (this.consecutiveFailures >= UNREACHABLE_THRESHOLD) {
          notifier.showSceneIdentityBanner('unreachable');
        }
        return;
      case 'changed':
        this.changed = true;
        log.warning(
          Modules.SCENE_LOADER,
          `Scene identity changed at ${this.url} — the loaded scene is stale.`
        );
        notifier.showSceneIdentityBanner('changed');
        // Terminal: the verdict cannot un-happen; stop burning requests.
        if (this.timer !== null) {
          clearInterval(this.timer);
          this.timer = null;
        }
        window.removeEventListener('focus', this.onWake);
        document.removeEventListener('visibilitychange', this.onWake);
        return;
    }
  }
}
