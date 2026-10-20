/** Core WebGPU negotiation shared by the early entry and renderer setup. */
export interface GPUAdapterLike {
  readonly features: ReadonlySet<string>;
  readonly limits: Record<string, number | undefined>;
  requestDevice: (descriptor: {
    requiredFeatures?: string[];
    requiredLimits?: Record<string, number>;
  }) => Promise<unknown>;
}

type NavigatorWithGPU = Navigator & {
  gpu?: {
    requestAdapter?: (options?: {
      featureLevel?: 'core' | 'compatibility';
      powerPreference?: 'low-power' | 'high-performance';
    }) => Promise<GPUAdapterLike | null>;
  };
};

/**
 * Adapter limits forwarded verbatim into the device's `requiredLimits`
 * when the adapter advertises them as numbers (absent / non-numeric
 * entries are omitted so the device keeps the spec default).
 *
 * `maxTextureDimension2D` is the per-node element-data ceiling:
 * element textures (points/lines/gsplats) grow DOWNWARD in rows, and
 * `renderer-capabilities` reads the LIVE DEVICE limit to derive every
 * capacity clamp. The WebGPU default is 8192 rows — for lines
 * (6 texels/segment, 682 segments/row at width 4096) that caps a node
 * at 5,586,944 segments while the same GPU's WebGL context exposes
 * 16384 rows = 11.2 M. Same forward-the-adapter policy as the buffer
 * limits.
 *
 * `maxVertexBuffers` is deliberately NOT here — it is clamped to 16
 * rather than forwarded verbatim (see `createWebGPURenderer`).
 */
const FORWARDED_ADAPTER_LIMIT_NAMES = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxTextureDimension2D',
] as const;
/** Names of the limits copied from the adapter into the device request. */
type ForwardedAdapterLimitName = (typeof FORWARDED_ADAPTER_LIMIT_NAMES)[number];

/** Forward numeric adapter limits; absent entries keep their spec defaults. */
export function forwardedAdapterLimits(
  limits: Partial<Record<ForwardedAdapterLimitName, number>> | undefined
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of FORWARDED_ADAPTER_LIMIT_NAMES) {
    const value = limits?.[key];
    if (typeof value === 'number') out[key] = value;
  }
  return out;
}

/** Whether the browser exposes the WebGPU entry point. */
export function hasWebGPU(): boolean {
  return (navigator as NavigatorWithGPU).gpu !== undefined;
}

async function requestCoreAdapter(): Promise<GPUAdapterLike | null> {
  const gpu = (navigator as NavigatorWithGPU).gpu;
  if (!gpu?.requestAdapter) return null;
  let adapter: GPUAdapterLike | null = null;
  try {
    adapter = await gpu.requestAdapter({
      featureLevel: 'core',
      powerPreference: 'high-performance',
    });
  } catch {
    // Older browsers reject the featureLevel option — fall through.
  }
  return adapter ?? gpu.requestAdapter().catch(() => null);
}

/** The result of native adapter and device negotiation before renderer setup. */
export type CoreDeviceRequest =
  | { kind: 'unavailable'; adapter: null }
  | { kind: 'insufficient'; adapter: GPUAdapterLike }
  | {
      kind: 'requested';
      adapter: GPUAdapterLike;
      requiredLimits: Record<string, number>;
      requestedMax?: number;
      webgpuModule?: Promise<typeof import('three/webgpu')>;
      device?: unknown;
      requestFailed?: boolean;
      requestError?: unknown;
    };

/** Negotiate the adapter, then start the module fetch and device request together. */
export async function requestCoreDevice(
  rendererOverride?: 'webgl' | 'webgpu'
): Promise<CoreDeviceRequest> {
  const adapter = await requestCoreAdapter();
  if (!adapter) return { kind: 'unavailable', adapter: null };

  const adapterMax = adapter.limits?.maxVertexBuffers;
  if (typeof adapterMax === 'number' && adapterMax < 8 && rendererOverride !== 'webgpu') {
    return { kind: 'insufficient', adapter };
  }

  const requestedMax = typeof adapterMax === 'number' ? Math.min(adapterMax, 16) : undefined;
  const requiredLimits = forwardedAdapterLimits(adapter.limits);
  if (requestedMax !== undefined) requiredLimits.maxVertexBuffers = requestedMax;

  const webgpuModule = import('three/webgpu');
  void webgpuModule.catch(() => undefined);
  try {
    const device = await adapter.requestDevice({
      requiredFeatures: [...adapter.features],
      requiredLimits,
    });
    return { kind: 'requested', adapter, requestedMax, requiredLimits, webgpuModule, device };
  } catch (requestError) {
    return {
      kind: 'requested',
      adapter,
      requestedMax,
      requiredLimits,
      webgpuModule,
      requestFailed: true,
      requestError,
    };
  }
}

let earlyRequest: Promise<CoreDeviceRequest> | undefined;

/** Start only after the standalone entry has selected native WebGPU. */
export function preRequestWebGPUDevice(rendererOverride?: 'webgl' | 'webgpu'): void {
  earlyRequest = requestCoreDevice(rendererOverride);
  void earlyRequest.catch(() => undefined);
}

/** A standalone bootstrap consumes this once; recovery always starts fresh. */
export function takePreRequestedWebGPUDevice(): Promise<CoreDeviceRequest> | undefined {
  const request = earlyRequest;
  earlyRequest = undefined;
  return request;
}
