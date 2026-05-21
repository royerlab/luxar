import type { OverlayManager } from '../../../ui/overlay-manager';

/**
 * Tear down the OverlayManager when present. The caller clears its
 * own reference via {@link DisposeOverlaysPorts.onDisposed} so the
 * helper never mutates orchestrator state directly.
 */
export interface DisposeOverlaysPorts {
  manager: OverlayManager | undefined;
  onDisposed: () => void;
}

export function disposeOverlays(ports: DisposeOverlaysPorts): void {
  if (ports.manager) {
    ports.manager.dispose();
    ports.onDisposed();
  }
}
