/**
 * Which machines get WebGPU when nothing selects a backend.
 *
 * WebGPU is the intended renderer, but whether it beats WebGL depends on the
 * platform: the #3146 go/no-go measured it 2-9% faster in GPU time on Linux
 * with an NVIDIA card (Vulkan), and up to ~20% slower on Apple GPUs (Metal),
 * where Dawn's bounds-check clamps run inside the shader (#3224). So the
 * default is a table of measured platforms rather than one global switch: a
 * row admits a platform to WebGPU, and everything without a row stays on
 * WebGL. Add a row only after the render gate's perf and hosted suites have
 * been run on that platform (`--backends webgl,webgpu`), and only once cold
 * load there is within 0.15 s of WebGL (#3172). Until then the table is empty
 * and measured rows wait in {@link STAGED_WEBGPU_PLATFORM_RULES}.
 *
 * Matching runs in two steps so a session that cannot qualify never pays for
 * WebGPU. The OS and browser come from the user agent and are checked before
 * any adapter request; the GPU vendor comes from `GPUAdapter.info` and is
 * checked once the adapter answers, before the device request and the
 * `three/webgpu` download. The vendor is the reliable key: user-agent strings
 * misreport the hardware (Safari on Apple silicon says "Intel Mac OS X").
 *
 * @module scene/scene-manager/render-pipeline/backend-policy
 */

/** Operating system, as far as the user agent tells. */
export type PlatformOS = 'linux' | 'macos' | 'windows' | 'android' | 'chromeos' | 'ios' | 'unknown';

/** Browser engine family. */
export type BrowserFamily = 'chromium' | 'firefox' | 'safari' | 'unknown';

/** What the user agent says about the machine. */
export interface BrowserPlatform {
  os: PlatformOS;
  browser: BrowserFamily;
}

/** One platform admitted to WebGPU by default. */
export interface WebGPUPlatformRule {
  os: PlatformOS;
  browser: BrowserFamily;
  /** `GPUAdapter.info.vendor`, lower case (e.g. `'nvidia'`, `'apple'`, `'amd'`). */
  vendor: string;
  /** The measurement that admitted the row. */
  evidence: string;
}

/**
 * Measured platforms that passed the GPU-speed half of the go/no-go but are
 * held off the default: WebGPU is not the default anywhere until its cold load
 * is within 0.15 s of WebGL (#3172). Move a row into
 * {@link WEBGPU_PLATFORM_RULES} once that holds on its platform.
 */
export const STAGED_WEBGPU_PLATFORM_RULES: readonly WebGPUPlatformRule[] = [
  {
    os: 'linux',
    browser: 'chromium',
    vendor: 'nvidia',
    evidence:
      '#3146 go/no-go (RTX 3070, Vulkan): WebGPU 2-9% faster GPU time on every heavy case; ' +
      'cold load ~0.5 s slower on a first visit (held on #3172)',
  },
];

/** The platforms where WebGPU is the default: none yet (see {@link STAGED_WEBGPU_PLATFORM_RULES}). */
export const WEBGPU_PLATFORM_RULES: readonly WebGPUPlatformRule[] = [];

/**
 * The rules the default selection uses: {@link WEBGPU_PLATFORM_RULES}, plus the
 * staged rows when a dev or test build sets `VITE_LUXAR_WEBGPU_STAGED_RULES=1`
 * (how the E2E suite exercises the policy path). Read per call, not at import.
 */
export function defaultPlatformRules(): readonly WebGPUPlatformRule[] {
  return import.meta.env.VITE_LUXAR_WEBGPU_STAGED_RULES === '1'
    ? [...WEBGPU_PLATFORM_RULES, ...STAGED_WEBGPU_PLATFORM_RULES]
    : WEBGPU_PLATFORM_RULES;
}

export interface NavigatorLike {
  userAgent?: string;
  userAgentData?: { platform?: string; brands?: ReadonlyArray<{ brand: string }> };
}

function osFromPlatformHint(platform: string): PlatformOS | undefined {
  const p = platform.toLowerCase();
  if (p === 'linux') return 'linux';
  if (p === 'macos') return 'macos';
  if (p === 'windows') return 'windows';
  if (p === 'android') return 'android';
  if (p === 'chrome os' || p === 'chromeos') return 'chromeos';
  return undefined;
}

function osFromUserAgent(ua: string): PlatformOS {
  // Order matters: Android and ChromeOS user agents also say "Linux", and
  // iPadOS Safari says "Macintosh".
  if (/Android/i.test(ua)) return 'android';
  if (/CrOS/.test(ua)) return 'chromeos';
  if (/iPhone|iPad|iPod/.test(ua)) return 'ios';
  if (/Windows/.test(ua)) return 'windows';
  if (/Mac OS X|Macintosh/.test(ua)) return 'macos';
  if (/Linux|X11/.test(ua)) return 'linux';
  return 'unknown';
}

function browserFromAgent(nav: NavigatorLike, ua: string): BrowserFamily {
  if (nav.userAgentData?.brands?.some((b) => b.brand === 'Chromium')) return 'chromium';
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Chrome\/|Chromium\/|Edg\//.test(ua)) return 'chromium';
  if (/Safari\//.test(ua)) return 'safari';
  return 'unknown';
}

/** Read the OS and browser family from the user agent. */
export function detectBrowserPlatform(
  nav: NavigatorLike | undefined = globalThis.navigator
): BrowserPlatform {
  const ua = nav?.userAgent ?? '';
  const hinted = nav?.userAgentData?.platform;
  const os = (hinted && osFromPlatformHint(hinted)) || osFromUserAgent(ua);
  return { os, browser: nav ? browserFromAgent(nav, ua) : 'unknown' };
}

/** The rules this OS and browser could still satisfy, before any adapter request. */
export function candidateRules(
  platform: BrowserPlatform,
  rules: readonly WebGPUPlatformRule[] = defaultPlatformRules()
): WebGPUPlatformRule[] {
  return rules.filter((r) => r.os === platform.os && r.browser === platform.browser);
}

/** The candidate rule this adapter satisfies, if any. A software fallback adapter never does. */
export function matchAdapterRule(
  candidates: readonly WebGPUPlatformRule[],
  info: { vendor?: string; isFallbackAdapter?: boolean } | undefined
): WebGPUPlatformRule | undefined {
  if (!info || info.isFallbackAdapter) return undefined;
  const vendor = (info.vendor ?? '').toLowerCase();
  return candidates.find((r) => r.vendor === vendor);
}
