/**
 * Everything `LuxarApp` holds for ONE dataset, in one owner.
 *
 * A session is created when a dataset load starts and disposed when the next
 * one starts or the app is disposed, so nothing a scene installed — its story
 * waypoints' dims listener, its kiosk watchdog, its archive-fault
 * subscription, its control-panel authoring — can outlive it or leak into the
 * next scene. Before this, each of those was an `app.ts` field with its own
 * hand-written reset, and the kiosk watchdog's was missing from `dispose()`.
 *
 * Process-wide per-dataset state that layer mode needs too stays at the
 * loader / scene seam both owners share: the update profiler is reset at
 * `loadStart` (`data/scene-loader/lifecycle/load-scene.ts`), and the outgoing
 * root's path index is detached by whichever owner drops the root
 * (`SceneManager.clearSceneContent`, `LuxarLayer`'s root detach).
 *
 * @module core/app/dataset/dataset-session
 */

import type { ControlPanelSettings } from '../../../config/zarr-bridge/control-panel';
import type { WaypointDriver } from '../camera/waypoint-driver';
import type { InstalledWaypoints } from '../camera/install-waypoints';
import type { DatasetFaultPayload } from '../embedder/events';

/** The part of a scene loader a session reads its archive faults from. */
export interface ArchiveFaultSource {
  readonly archiveFault: Error | null;
  onArchiveFault(listener: (error: Error) => void, options: { replayCurrent: boolean }): () => void;
}

export class DatasetSession {
  private loaded = false;
  private disposed = false;
  private faultSource: ArchiveFaultSource | null = null;
  private faultUnsubscribe: (() => void) | null = null;
  private waypoints: InstalledWaypoints | null = null;
  private kioskTeardown: (() => void) | null = null;
  /**
   * The scene's authored control-panel block. Held for `getViewerState()`
   * alone: the touch panel is a separate page and cannot read the store's
   * attributes itself.
   */
  controlPanelConfig: ControlPanelSettings | null = null;

  /** @param src The dataset this session loads; undefined for "no dataset yet". */
  constructor(readonly src: string | undefined) {}

  /** The dataset's src once its load has succeeded, else undefined. */
  get loadedSrc(): string | undefined {
    return this.loaded ? this.src : undefined;
  }

  /**
   * The load committed: the dataset is available and its fault latch is the
   * given loader's. A no-op on a session already disposed (the app was
   * disposed, or a newer load started, while this one was in flight).
   */
  markLoaded(source: ArchiveFaultSource | null): void {
    if (this.disposed) return;
    this.loaded = true;
    this.faultSource = source;
  }

  /**
   * Report the loaded dataset's archive faults from now on. A fault latched
   * during the load is replayed immediately — so call this AFTER announcing the
   * dataset (the public ordering: available, then faulted).
   */
  reportFaults(onFault: (error: Error) => void): void {
    if (this.disposed || !this.faultSource) return;
    this.faultUnsubscribe?.();
    this.faultUnsubscribe = this.faultSource.onArchiveFault(onFault, { replayCurrent: true });
  }

  /** The loaded dataset's latched archive fault, if any. */
  get fault(): DatasetFaultPayload | null {
    const error = this.faultSource?.archiveFault;
    const src = this.loadedSrc;
    return error && src ? { src, error } : null;
  }

  /** The scene's story-waypoint driver, while one is installed. */
  get waypointDriver(): WaypointDriver | undefined {
    return this.waypoints?.driver;
  }

  /**
   * Replace the installed waypoints (the previous binding is released). A
   * disposed session releases the new binding at once: a load superseded
   * mid-flight must not leave its dims listener behind.
   */
  setWaypoints(waypoints: InstalledWaypoints | null): void {
    this.waypoints?.dispose();
    this.waypoints = null;
    if (this.disposed) waypoints?.dispose();
    else this.waypoints = waypoints;
  }

  /** Replace the kiosk watchdog teardown (stopped at once on a disposed session). */
  setKioskTeardown(teardown: (() => void) | null): void {
    this.kioskTeardown?.();
    this.kioskTeardown = null;
    if (this.disposed) teardown?.();
    else this.kioskTeardown = teardown;
  }

  /** Release everything the dataset installed. Idempotent. */
  dispose(): void {
    this.disposed = true;
    this.faultUnsubscribe?.();
    this.faultUnsubscribe = null;
    this.faultSource = null;
    this.loaded = false;
    this.setWaypoints(null);
    this.setKioskTeardown(null);
    this.controlPanelConfig = null;
  }
}
