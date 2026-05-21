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
 * The package.json declares `three@^0.184.0` as a peer; we use APIs
 * (Timer, current postprocessing ToneMappingEffect shape) that are not
 * present in older revisions. Fail fast with a clear message instead
 * of a cryptic "X is not a constructor" deep in initialization.
 */
export function assertThreeRevision(minRevision = 184): void {
  const threeRevision = parseInt(THREE.REVISION ?? '0', 10);
  if (!Number.isFinite(threeRevision) || threeRevision < minRevision) {
    throw new Error(
      `Luxar requires three@>=0.${minRevision}.0 (found r${THREE.REVISION ?? '?'}). ` +
        'Update the three peer dependency in your embedder.'
    );
  }
}
