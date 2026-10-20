/** Adapter negotiation shared by the early standalone entry and renderer setup. */
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

export function hasWebGPU(): boolean {
  return (navigator as NavigatorWithGPU).gpu !== undefined;
}

export async function requestCoreAdapter(): Promise<GPUAdapterLike | null> {
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

let earlyRequest: Promise<GPUAdapterLike | null> | undefined;

/** Start only after the standalone entry has selected WebGPU. */
export function preRequestWebGPUAdapter(): void {
  earlyRequest = requestCoreAdapter();
  void earlyRequest.catch(() => undefined);
}

/** A standalone bootstrap may consume the request once; recovery always starts fresh. */
export function takePreRequestedWebGPUAdapter(): Promise<GPUAdapterLike | null> | undefined {
  const request = earlyRequest;
  earlyRequest = undefined;
  return request;
}
