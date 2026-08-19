import * as THREE from 'three';

/**
 * Throw early if `window` / `document` is missing. SceneManager and
 * InputHandler reach for them unconditionally, so a friendly upfront
 * error beats a cryptic ReferenceError half-way through init for SSR
 * or non-browser callers.
 */
export function assertBrowserEnvironment(): void {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    throw new Error(
      'LuxarApp requires a browser environment (window and document must be defined).'
    );
  }
}

/**
 * Throw if the host's THREE.js revision is below what the viewer needs.
 *
 * We use APIs (Timer, current postprocessing ToneMappingEffect shape) that
 * are not present in older revisions. Fail fast with a clear message instead
 * of a cryptic "X is not a constructor" deep in initialization.
 *
 * The floor here (185) is only the LOWER bound of the peer range in
 * package.json (`three@~0.185.1`, which is closed at 0.186). That upper bound
 * lives in the peer range alone — a hard error under npm, a warning under
 * pnpm/yarn — and this guard deliberately does not re-check it. The floor
 * tracks the peer range rather than the oldest revision the shaders happen to
 * run on: the TSL graphs are measured clean at r184 *and* r185, but r185 is
 * the supported configuration and the only one the codegen snapshots pin.
 * `@types/three` now sits on the same minor as the runtime — the deliberate
 * one-minor skew that used to exist here is gone. THREE_VERSION_NOTES.md is
 * the single record of both pins — read it before moving either.
 */
export function assertThreeRevision(minRevision = 185): void {
  const threeRevision = parseInt(THREE.REVISION ?? '0', 10);
  if (!Number.isFinite(threeRevision) || threeRevision < minRevision) {
    throw new Error(
      `Luxar requires three@>=0.${minRevision}.0 (found r${THREE.REVISION ?? '?'}). ` +
        'Update the three peer dependency in your embedder.'
    );
  }
}
