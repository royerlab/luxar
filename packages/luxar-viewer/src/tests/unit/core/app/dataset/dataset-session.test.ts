/**
 * `DatasetSession` — the one owner of what a loaded dataset installed in the
 * app. The contract: dispose releases all of it, a disposed session (a load
 * superseded or the app disposed mid-flight) releases anything installed late,
 * and the fault latch only reports a dataset that actually loaded.
 */

import { describe, expect, it, vi, type Mock } from 'vitest';

import {
  DatasetSession,
  type ArchiveFaultSource,
} from '../../../../../core/app/dataset/dataset-session';
import type { InstalledWaypoints } from '../../../../../core/app/camera/install-waypoints';

function faultSource(error: Error | null = null): ArchiveFaultSource & {
  unsubscribe: ReturnType<typeof vi.fn>;
} {
  const unsubscribe = vi.fn();
  return {
    archiveFault: error,
    onArchiveFault: vi.fn((listener: (e: Error) => void, options: { replayCurrent: boolean }) => {
      if (options.replayCurrent && error) listener(error);
      return unsubscribe;
    }),
    unsubscribe,
  };
}

function waypoints(): InstalledWaypoints & { dispose: Mock<() => void> } {
  return { driver: { inTransit: false } as never, dispose: vi.fn<() => void>() };
}

describe('DatasetSession', () => {
  it('reports the src and fault only once the load has succeeded', () => {
    const error = new Error('archive gone');
    const session = new DatasetSession('http://x/a.zarr');
    expect(session.loadedSrc).toBeUndefined();
    expect(session.fault).toBeNull();

    session.markLoaded(faultSource(error));
    expect(session.loadedSrc).toBe('http://x/a.zarr');
    expect(session.fault).toEqual({ src: 'http://x/a.zarr', error });
  });

  it('replays a latched fault when reporting starts, and unsubscribes on dispose', () => {
    const error = new Error('archive gone');
    const source = faultSource(error);
    const session = new DatasetSession('http://x/a.zarr');
    session.markLoaded(source);
    const onFault = vi.fn();

    session.reportFaults(onFault);
    expect(onFault).toHaveBeenCalledWith(error);

    session.dispose();
    expect(source.unsubscribe).toHaveBeenCalledOnce();
    expect(session.fault).toBeNull();
    expect(session.loadedSrc).toBeUndefined();
  });

  it('releases the waypoints, the kiosk watchdog and the panel authoring on dispose', () => {
    const session = new DatasetSession('http://x/a.zarr');
    const installed = waypoints();
    const kiosk = vi.fn();
    session.setWaypoints(installed);
    session.setKioskTeardown(kiosk);
    session.controlPanelConfig = { title: 'Tour' } as never;
    expect(session.waypointDriver).toBe(installed.driver);

    session.dispose();
    session.dispose(); // idempotent

    expect(installed.dispose).toHaveBeenCalledOnce();
    expect(kiosk).toHaveBeenCalledOnce();
    expect(session.waypointDriver).toBeUndefined();
    expect(session.controlPanelConfig).toBeNull();
  });

  it('replaces a binding by releasing the previous one', () => {
    const session = new DatasetSession('http://x/a.zarr');
    const first = waypoints();
    const firstKiosk = vi.fn();
    session.setWaypoints(first);
    session.setKioskTeardown(firstKiosk);

    session.setWaypoints(waypoints());
    session.setKioskTeardown(vi.fn());

    expect(first.dispose).toHaveBeenCalledOnce();
    expect(firstKiosk).toHaveBeenCalledOnce();
  });

  it('releases at once what a superseded load installs after its session was disposed', () => {
    const session = new DatasetSession('http://x/a.zarr');
    session.dispose();
    const late = waypoints();
    const lateKiosk = vi.fn();
    const source = faultSource();

    session.setWaypoints(late);
    session.setKioskTeardown(lateKiosk);
    session.markLoaded(source);
    session.reportFaults(vi.fn());

    expect(late.dispose).toHaveBeenCalledOnce();
    expect(lateKiosk).toHaveBeenCalledOnce();
    expect(session.waypointDriver).toBeUndefined();
    expect(session.loadedSrc).toBeUndefined();
    expect(source.onArchiveFault).not.toHaveBeenCalled();
  });
});
