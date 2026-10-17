import { describe, it, expect, vi } from 'vitest';
import {
  WebGPUDeviceRecovery,
  type DeviceLostInfo,
  type WebGPUDeviceRecoveryDeps,
} from '../../../../../scene/scene-manager/render-pipeline/webgpu-device-recovery';

/** A device whose `lost` promise the test resolves. */
function fakeDevice() {
  let resolve!: (info: DeviceLostInfo) => void;
  const lost = new Promise<DeviceLostInfo>((r) => (resolve = r));
  return { device: { lost }, lose: (info: DeviceLostInfo) => resolve(info) };
}

/** Let the `lost.then` chain and the recovery's awaits run. */
const settle = () => new Promise((r) => setTimeout(r, 0));

function harness(recreate: () => Promise<boolean>) {
  let current = fakeDevice();
  const devices = [current];
  const deps = {
    getDevice: vi.fn(() => current.device),
    recreateRenderer: vi.fn(async () => {
      const ok = await recreate();
      if (ok) {
        current = fakeDevice();
        devices.push(current);
      }
      return ok;
    }),
    onLost: vi.fn(),
    onRestored: vi.fn(),
    onUnrecoverable: vi.fn(),
    delay: vi.fn(() => Promise.resolve()),
  } satisfies WebGPUDeviceRecoveryDeps;
  const recovery = new WebGPUDeviceRecovery(deps);
  recovery.attach();
  return { recovery, deps, devices, current: () => current };
}

describe('WebGPUDeviceRecovery', () => {
  it('rebuilds the renderer after a loss and re-arms on the new device', async () => {
    const h = harness(async () => true);
    h.devices[0].lose({ reason: 'unknown', message: 'driver reset' });
    await settle();

    expect(h.deps.onLost).toHaveBeenCalledWith({ reason: 'unknown', message: 'driver reset' });
    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(1);
    expect(h.deps.onRestored).toHaveBeenCalledTimes(1);
    expect(h.recovery.isLost()).toBe(false);

    // The NEW device is observed: losing it starts a second recovery.
    h.devices[1].lose({ reason: 'unknown' });
    await settle();
    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(2);
    expect(h.deps.onRestored).toHaveBeenCalledTimes(2);
  });

  it('ignores an explicitly destroyed device', async () => {
    const h = harness(async () => true);
    h.devices[0].lose({ reason: 'destroyed' });
    await settle();
    expect(h.deps.onLost).not.toHaveBeenCalled();
    expect(h.deps.recreateRenderer).not.toHaveBeenCalled();
    expect(h.recovery.isLost()).toBe(false);
  });

  it('retries once, then reports the loss as unrecoverable', async () => {
    const h = harness(async () => false);
    h.devices[0].lose({ reason: 'unknown' });
    await settle();
    await settle();

    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(2);
    expect(h.deps.delay).toHaveBeenCalledTimes(1);
    expect(h.deps.onRestored).not.toHaveBeenCalled();
    expect(h.deps.onUnrecoverable).toHaveBeenCalledWith({ reason: 'unknown' });
    expect(h.recovery.isLost()).toBe(true);
  });

  it('treats a throwing rebuild as a failed attempt and recovers on the retry', async () => {
    let calls = 0;
    const h = harness(async () => {
      calls++;
      if (calls === 1) throw new Error('requestDevice rejected');
      return true;
    });
    h.devices[0].lose({ reason: 'unknown' });
    await settle();
    await settle();
    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(2);
    expect(h.deps.onRestored).toHaveBeenCalledTimes(1);
    expect(h.deps.onUnrecoverable).not.toHaveBeenCalled();
  });

  it('reports lost while the rebuild is in flight', async () => {
    let finish!: (ok: boolean) => void;
    const h = harness(() => new Promise<boolean>((r) => (finish = r)));
    h.devices[0].lose({ reason: 'unknown' });
    await settle();
    expect(h.recovery.isLost()).toBe(true);
    finish(true);
    await settle();
    expect(h.recovery.isLost()).toBe(false);
  });

  it('does not start a second rebuild for an overlapping simulated loss', async () => {
    let finish!: (ok: boolean) => void;
    const h = harness(() => new Promise<boolean>((resolve) => (finish = resolve)));
    const first = h.recovery.simulateLoss();
    const second = h.recovery.simulateLoss();
    await settle();
    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(1);
    expect(h.deps.onLost).toHaveBeenCalledTimes(1);
    finish(true);
    await Promise.all([first, second]);
    expect(h.deps.onRestored).toHaveBeenCalledTimes(1);
  });

  it('does not simulate a second rebuild while recovering from a real loss', async () => {
    let finish!: (ok: boolean) => void;
    const h = harness(() => new Promise<boolean>((resolve) => (finish = resolve)));
    h.devices[0].lose({ reason: 'unknown' });
    await settle();
    await h.recovery.simulateLoss();
    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(1);
    expect(h.deps.onLost).toHaveBeenCalledTimes(1);
    finish(true);
    await settle();
    expect(h.deps.onRestored).toHaveBeenCalledTimes(1);
  });

  it('does nothing after dispose', async () => {
    const h = harness(async () => true);
    h.recovery.dispose();
    h.devices[0].lose({ reason: 'unknown' });
    await settle();
    expect(h.deps.onLost).not.toHaveBeenCalled();
  });

  it('simulateLoss runs the same sequence without a real loss', async () => {
    const h = harness(async () => true);
    await h.recovery.simulateLoss();
    expect(h.deps.onLost).toHaveBeenCalledTimes(1);
    expect(h.deps.onRestored).toHaveBeenCalledTimes(1);
    // A late loss from the old device is ignored after the new one is attached.
    h.devices[0].lose({ reason: 'unknown' });
    await settle();
    expect(h.deps.recreateRenderer).toHaveBeenCalledTimes(1);
  });

  it('attach() reports false when the renderer has no device (WebGL2 bridge)', () => {
    const recovery = new WebGPUDeviceRecovery({
      getDevice: () => undefined,
      recreateRenderer: async () => true,
      onLost: vi.fn(),
      onRestored: vi.fn(),
      onUnrecoverable: vi.fn(),
    });
    expect(recovery.attach()).toBe(false);
  });
});
