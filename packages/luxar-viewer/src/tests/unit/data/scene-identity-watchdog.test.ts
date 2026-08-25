// @vitest-environment jsdom
/**
 * Unit tests for the scene-identity watchdog.
 *
 * The watchdog is driven with fake timers and an injected fetch stub;
 * verdicts are observed through a recording notifier backend (the same
 * cross-layer surface production uses), so these tests cover the full
 * data-side path without any DOM banner.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SceneIdentityWatchdog, canonicalJson } from '../../../data/scene-identity-watchdog';
import { clearNotifierBackend, setNotifierBackend } from '../../../utils/cross-layer/notifier';

const HASH = 'abc123';
const ATTRS = JSON.stringify({ content_hash: HASH, luxar_version: '0.1' });
const OTHER_ATTRS = JSON.stringify({ content_hash: 'zzz999' });

type FetchStub = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function okResponse(body: string): Response {
  return new Response(body, { status: 200 });
}

interface BannerLog {
  shown: string[];
  hidden: Array<string | undefined>;
}

function installRecordingBanner(): BannerLog {
  const bannerLog: BannerLog = { shown: [], hidden: [] };
  setNotifierBackend({
    showError: () => {},
    showToast: () => {},
    showHelpOverlay: () => {},
    hideHelpOverlay: () => {},
    showLoadingIndicator: () => {},
    hideLoadingIndicator: () => {},
    clearError: () => {},
    showSceneIdentityBanner: (kind) => bannerLog.shown.push(kind),
    hideSceneIdentityBanner: (onlyKind) => bannerLog.hidden.push(onlyKind),
  });
  return bannerLog;
}

/** Advance past the probe-spacing guard and run one interval tick. */
async function tick(intervalMs: number): Promise<void> {
  await vi.advanceTimersByTimeAsync(intervalMs);
}

describe('SceneIdentityWatchdog', () => {
  let banner: BannerLog;

  beforeEach(() => {
    vi.useFakeTimers();
    banner = installRecordingBanner();
  });

  it('does not watch zipped stores whose attrs are archive members', () => {
    expect(SceneIdentityWatchdog.isWatchable('https://example.com/scene.luxar.zarr.zip')).toBe(
      false
    );
    expect(
      SceneIdentityWatchdog.isWatchable(
        'https://example.com/SCENE.LUXAR.ZARR.ZIP?token=abc#view'
      )
    ).toBe(false);
    expect(SceneIdentityWatchdog.isWatchable('https://example.com/scene.luxar.zarr')).toBe(true);
  });

  afterEach(() => {
    vi.useRealTimers();
    clearNotifierBackend();
  });

  function makeWatchdog(fetchImpl: FetchStub, intervalMs = 5000) {
    return new SceneIdentityWatchdog({
      datasetUrl: 'http://127.0.0.1:8000/',
      expectedContentHash: HASH,
      intervalMs,
      fetchImpl: fetchImpl as typeof fetch,
    });
  }

  it('is watchable only for http(s) sources', () => {
    expect(SceneIdentityWatchdog.isWatchable('http://x:8000')).toBe(true);
    expect(SceneIdentityWatchdog.isWatchable('https://x')).toBe(true);
    expect(SceneIdentityWatchdog.isWatchable('file:///data.zarr')).toBe(false);
    expect(SceneIdentityWatchdog.isWatchable('/relative/data.zarr')).toBe(false);
  });

  it('probes the trailing-slash-trimmed root document with cache bypass', async () => {
    const urls: string[] = [];
    const wd = makeWatchdog(async (input, init) => {
      urls.push(String(input));
      expect(init?.cache).toBe('no-store');
      return okResponse(ATTRS);
    });
    wd.start();
    await tick(5000);
    // `zarr.json` is probed FIRST because new datasets are format 3; the stub
    // answers it, so the format-2 `.zattrs` is never requested.
    expect(urls).toEqual(['http://127.0.0.1:8000/zarr.json']);
    wd.dispose();
  });

  it('appends the root document to the PATH, keeping a query and dropping a fragment', async () => {
    // A presigned/tokenized source carries its credentials in the query, and
    // the zarr fetch store copies them onto every key it reads — string
    // concatenation would bury the appended path inside the query (or behind the
    // fragment) and 404 the probe into a bogus "different scene" verdict.
    const urls: string[] = [];
    const probeUrl = async (datasetUrl: string) => {
      const wd = new SceneIdentityWatchdog({
        datasetUrl,
        expectedContentHash: HASH,
        intervalMs: 5000,
        fetchImpl: (async (input: RequestInfo | URL) => {
          urls.push(String(input));
          return okResponse(ATTRS);
        }) as typeof fetch,
      });
      wd.start();
      await tick(5000);
      wd.dispose();
    };

    await probeUrl('https://host/scene.zarr?token=abc');
    await probeUrl('https://host/scene.zarr/?token=abc');
    await probeUrl('https://host/scene.zarr#frag');
    // A credential whose last character is a literal `/`. The zarr store
    // sends the query verbatim, so the probe has to as well: trimming the
    // raw string would authenticate as a different, truncated credential,
    // get itself refused, and raise a spurious "server unreachable" over a
    // scene that is loading perfectly well.
    await probeUrl('https://host/scene.zarr?token=abc/');
    expect(urls).toEqual([
      'https://host/scene.zarr/zarr.json?token=abc',
      'https://host/scene.zarr/zarr.json?token=abc',
      'https://host/scene.zarr/zarr.json',
      'https://host/scene.zarr/zarr.json?token=abc/',
    ]);
  });

  it('falls back to .zattrs when the dataset is format 2', async () => {
    // A format-2 store has no `zarr.json`, and 404 is deliberately NOT an
    // inconclusive status — so without this fallback the probe would take the
    // 404 as a definite answer and report `changed` on EVERY poll, driving a
    // reload loop over a scene that never moved.
    const urls: string[] = [];
    const wd = makeWatchdog(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith('/zarr.json')) {
        return new Response('', { status: 404 });
      }
      return okResponse(ATTRS);
    });
    wd.start();
    await tick(5000);
    expect(urls).toEqual(['http://127.0.0.1:8000/zarr.json', 'http://127.0.0.1:8000/.zattrs']);

    // ...and the resolved document is remembered, so the next poll costs one
    // request rather than re-paying the 404 forever.
    urls.length = 0;
    await tick(5000);
    expect(urls).toEqual(['http://127.0.0.1:8000/.zattrs']);
    wd.dispose();
  });

  it('reads content_hash out of a format-3 envelope', async () => {
    // Format 3 nests attributes under `attributes`. Reading the top level finds
    // `undefined`, which does not equal the expected hash -- so an UNCHANGED
    // scene would be reported as changed on every poll. The absence of a banner
    // is the assertion here.
    const wd = makeWatchdog(async () =>
      okResponse(
        JSON.stringify({
          zarr_format: 3,
          node_type: 'group',
          attributes: { content_hash: HASH },
        })
      )
    );
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual([]);
    wd.dispose();

    // ...and a format-3 envelope carrying a DIFFERENT hash still trips it, so
    // the unwrapping did not simply stop comparing.
    banner.shown.length = 0;
    const wd2 = makeWatchdog(async () =>
      okResponse(
        JSON.stringify({
          zarr_format: 3,
          node_type: 'group',
          attributes: { content_hash: 'other' },
        })
      )
    );
    wd2.start();
    await tick(5000);
    expect(banner.shown).toEqual(['changed']);
    wd2.dispose();
  });

  it('matching hash keeps quiet and clears a prior unreachable banner', async () => {
    const wd = makeWatchdog(async () => okResponse(ATTRS));
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual([]);
    // The OK path clears only the unreachable kind — never a changed verdict.
    expect(banner.hidden).toEqual(['unreachable']);
    wd.dispose();
  });

  it('different hash raises the changed banner and stops polling', async () => {
    let calls = 0;
    const wd = makeWatchdog(async () => {
      calls++;
      return okResponse(OTHER_ATTRS);
    });
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual(['changed']);
    // Terminal: no further probes on later ticks.
    await tick(5000);
    await tick(5000);
    expect(calls).toBe(1);
    wd.dispose();
  });

  it('HTTP 404 and unparseable bodies read as changed', async () => {
    const wd = makeWatchdog(async () => new Response('nope', { status: 404 }));
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual(['changed']);
    wd.dispose();

    banner.shown.length = 0;
    const wd2 = makeWatchdog(async () => okResponse('<html>not zarr</html>'));
    wd2.start();
    await tick(5000);
    expect(banner.shown).toEqual(['changed']);
    wd2.dispose();
  });

  it('an auth wall (401/403) is inconclusive, not a scene change', async () => {
    // A presigned/tokenized source whose credential expired answers 403. The
    // scene may well be unchanged, and the changed banner's Reload cannot
    // repair a stale credential — so this must stay non-terminal and clear
    // itself once the source is reachable again.
    for (const status of [401, 403]) {
      banner.shown.length = 0;
      banner.hidden.length = 0;
      let mode: 'refused' | 'up' = 'refused';
      const wd = makeWatchdog(async () =>
        mode === 'refused' ? new Response('denied', { status }) : okResponse(ATTRS)
      );
      wd.start();
      await tick(5000);
      expect(banner.shown).toEqual([]); // single blip stays silent
      await tick(5000);
      expect(banner.shown).toEqual(['unreachable']);
      // Non-terminal: polling continues and recovery clears the banner.
      mode = 'up';
      await tick(5000);
      expect(banner.shown).toEqual(['unreachable']);
      expect(banner.hidden).toContain('unreachable');
      wd.dispose();
    }
  });

  it('a transient 503 is a reachability failure, not a scene change', async () => {
    let mode: 'overloaded' | 'up' = 'overloaded';
    let calls = 0;
    const wd = makeWatchdog(async () => {
      calls++;
      if (mode === 'overloaded') return new Response('busy', { status: 503 });
      return okResponse(ATTRS);
    });
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual([]); // single blip stays silent
    await tick(5000);
    expect(banner.shown).toEqual(['unreachable']);
    // Non-terminal: polling continues and the banner clears on recovery.
    mode = 'up';
    await tick(5000);
    expect(calls).toBe(3);
    expect(banner.shown).toEqual(['unreachable']);
    expect(banner.hidden).toContain('unreachable');
    wd.dispose();
  });

  it('a probe that never answers times out instead of wedging the watchdog', async () => {
    const aborted: boolean[] = [];
    let calls = 0;
    const wd = makeWatchdog(async (_input, init) => {
      calls++;
      // A server that accepts the connection and then goes silent.
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          aborted.push(true);
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
    });
    wd.start();
    await tick(5000);
    expect(calls).toBe(1);
    // Ten seconds in, the probe is aborted and counted as a failure...
    await vi.advanceTimersByTimeAsync(10_000);
    expect(aborted).toEqual([true]);
    // ...and the next interval tick probes again rather than being blocked
    // forever by the wedged in-flight guard.
    await tick(5000);
    expect(calls).toBe(2);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(banner.shown).toEqual(['unreachable']);
    wd.dispose();
  });

  it('dispose aborts a probe still in flight', async () => {
    let aborted = false;
    const wd = makeWatchdog(
      async (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            aborted = true;
            reject(new DOMException('aborted', 'AbortError'));
          });
        })
    );
    wd.start();
    await tick(5000);
    wd.dispose();
    await vi.advanceTimersByTimeAsync(0);
    expect(aborted).toBe(true);
    expect(banner.shown).toEqual([]); // a disposed watchdog raises nothing
  });

  it('needs two consecutive failures before unreachable, then recovers', async () => {
    let mode: 'down' | 'up' = 'down';
    const wd = makeWatchdog(async () => {
      if (mode === 'down') throw new TypeError('fetch failed');
      return okResponse(ATTRS);
    });
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual([]); // single blip stays silent
    await tick(5000);
    expect(banner.shown).toEqual(['unreachable']);
    mode = 'up';
    await tick(5000);
    expect(banner.hidden).toContain('unreachable');
    wd.dispose();
  });

  it('a server that recovers with a different scene escalates to changed', async () => {
    let mode: 'down' | 'other' = 'down';
    const wd = makeWatchdog(async () => {
      if (mode === 'down') throw new TypeError('fetch failed');
      return okResponse(OTHER_ATTRS);
    });
    wd.start();
    await tick(5000);
    await tick(5000);
    expect(banner.shown).toEqual(['unreachable']);
    mode = 'other';
    await tick(5000);
    expect(banner.shown).toEqual(['unreachable', 'changed']);
    wd.dispose();
  });

  it('hash-less scenes compare probes against the LOADED attrs json', async () => {
    // Whitespace/formatting differences are canonicalized away; only a
    // structural change reads as a different scene.
    const bodies = ['{ "a": 1 }', '{"a":1}', '{"a": 2}'];
    let i = 0;
    const wd = new SceneIdentityWatchdog({
      datasetUrl: 'http://127.0.0.1:8000',
      expectedContentHash: null,
      expectedAttrsJson: JSON.stringify({ a: 1 }),
      intervalMs: 5000,
      fetchImpl: (async () => okResponse(bodies[i++])) as typeof fetch,
    });
    wd.start();
    await tick(5000); // spaced formatting — same structure
    await tick(5000); // compact formatting — same structure
    expect(banner.shown).toEqual([]);
    await tick(5000); // different structure
    expect(banner.shown).toEqual(['changed']);
    wd.dispose();
  });

  it('hash-less identity ignores key order, at every nesting level', async () => {
    // The baseline comes from the store's consolidated metadata while the
    // probe reads the raw `.zattrs` — a key-order difference between the two
    // encodings of the SAME attrs must not read as a different scene.
    const wd = new SceneIdentityWatchdog({
      datasetUrl: 'http://127.0.0.1:8000',
      expectedContentHash: null,
      expectedAttrsJson: canonicalJson({ b: 1, a: { d: [1, 2], c: 3 } }),
      intervalMs: 5000,
      // Served in a different (unsorted) key order than the canonical baseline.
      fetchImpl: (async () => okResponse('{"b":1,"a":{"d":[1,2],"c":3}}')) as typeof fetch,
    });
    wd.start();
    await tick(5000);
    expect(banner.shown).toEqual([]);
    wd.dispose();
  });

  it('hash-less scenes catch a swap BEFORE the first probe', async () => {
    // The old first-probe baseline would have adopted the impostor as the
    // identity; baselining on the loaded attrs catches it immediately.
    const wd = new SceneIdentityWatchdog({
      datasetUrl: 'http://127.0.0.1:8000',
      expectedContentHash: null,
      expectedAttrsJson: JSON.stringify({ scene: 'loaded-one' }),
      intervalMs: 5000,
      fetchImpl: (async () => okResponse('{"scene": "impostor"}')) as typeof fetch,
    });
    wd.start();
    await tick(5000); // FIRST probe already sees the impostor
    expect(banner.shown).toEqual(['changed']);
    wd.dispose();
  });

  it('with neither hash nor attrs json, only reachability is watched', async () => {
    const wd = new SceneIdentityWatchdog({
      datasetUrl: 'http://127.0.0.1:8000',
      expectedContentHash: null,
      expectedAttrsJson: null,
      intervalMs: 5000,
      fetchImpl: (async () => okResponse('{"anything": 1}')) as typeof fetch,
    });
    wd.start();
    await tick(5000);
    await tick(5000);
    expect(banner.shown).toEqual([]);
    wd.dispose();
  });

  it('dispose stops probing and clears any banner', async () => {
    let calls = 0;
    const wd = makeWatchdog(async () => {
      calls++;
      throw new TypeError('down');
    });
    wd.start();
    await tick(5000);
    wd.dispose();
    expect(banner.hidden).toContain(undefined); // unconditional clear
    await tick(5000);
    await tick(5000);
    expect(calls).toBe(1);
  });

  it('focus wakeups are rate-limited by the probe spacing guard', async () => {
    let calls = 0;
    const wd = makeWatchdog(async () => {
      calls++;
      return okResponse(ATTRS);
    });
    wd.start();
    await tick(5000);
    expect(calls).toBe(1);
    // Immediately after a probe, a focus event must not double-probe.
    window.dispatchEvent(new Event('focus'));
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toBe(1);
    // Past the spacing guard, focus probes immediately (no interval wait).
    await vi.advanceTimersByTimeAsync(2500);
    window.dispatchEvent(new Event('focus'));
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toBe(2);
    wd.dispose();
  });
});
