import { ROOT_ATTR_DOCS, rootAttributes } from '../../types/zarr-documents';

interface GalleryDebugState {
  isLoading?: boolean;
  totalElements?: number;
  dimensions?: { currentStep?: number[]; displayed?: number[] } | null;
}

function extractBakedDimensionStep(attributes: Record<string, unknown>): number[] | null {
  const viewerConfig = attributes.viewer_config;
  if (viewerConfig === null || typeof viewerConfig !== 'object') return null;
  const dimensions = (viewerConfig as Record<string, unknown>).dimensions;
  if (dimensions === null || typeof dimensions !== 'object') return null;
  const currentStep = (dimensions as Record<string, unknown>).current_step;
  if (currentStep === undefined) return null;
  if (!Array.isArray(currentStep) || !currentStep.every(Number.isFinite)) {
    throw new Error('viewer_config.dimensions.current_step must be an array of finite numbers');
  }
  return [...currentStep] as number[];
}

export async function readBakedDimensionStep(
  dataUrl: string,
  fetchRootDocument: (url: string) => Promise<{ ok: boolean; json: () => Promise<unknown> }> = fetch
): Promise<number[] | null> {
  const rootUrl = dataUrl.replace(/\/$/, '');
  for (const documentName of ROOT_ATTR_DOCS) {
    const response = await fetchRootDocument(`${rootUrl}/${documentName}`);
    if (!response.ok) continue;
    const attributes = rootAttributes(await response.json(), documentName);
    return extractBakedDimensionStep(attributes);
  }
  throw new Error(`Could not read root attributes from ${dataUrl}`);
}

export function isGalleryDataReady(options: {
  requireElements: boolean;
  expectedDimensionStep: number[] | null;
}): boolean {
  const galleryGlobal = globalThis as typeof globalThis & {
    __luxarDebug?: { getState?: () => GalleryDebugState | null };
    __luxarGalleryDimensionStepReached?: boolean;
  };
  const debug = galleryGlobal.__luxarDebug;
  const state = debug?.getState?.();
  if (!state) return false;

  const expectedStep = options.expectedDimensionStep;
  if (expectedStep && !galleryGlobal.__luxarGalleryDimensionStepReached) {
    const currentStep = state.dimensions?.currentStep;
    if (currentStep) {
      if (expectedStep.length > currentStep.length) {
        throw new Error(
          `Authored dimension step has ${expectedStep.length} values, but the scene exposes ${currentStep.length} dimensions`
        );
      }
      const displayed = new Set(state.dimensions?.displayed ?? []);
      galleryGlobal.__luxarGalleryDimensionStepReached = expectedStep.every(
        (value, index) => displayed.has(index) || currentStep[index] === value
      );
    }
  }
  if (state.isLoading || (expectedStep && !galleryGlobal.__luxarGalleryDimensionStepReached)) {
    return false;
  }

  return options.requireElements ? (state.totalElements ?? 0) > 0 : true;
}

export function describeGalleryDataState(options: {
  requireElements: boolean;
  expectedDimensionStep: number[] | null;
}): string {
  const galleryGlobal = globalThis as typeof globalThis & {
    __luxarDebug?: { getState?: () => GalleryDebugState | null };
    __luxarGalleryDimensionStepReached?: boolean;
  };
  const state = galleryGlobal.__luxarDebug?.getState?.();
  return [
    `expectedDimensionStep=${JSON.stringify(options.expectedDimensionStep)}`,
    `currentStep=${JSON.stringify(state?.dimensions?.currentStep ?? null)}`,
    `displayed=${JSON.stringify(state?.dimensions?.displayed ?? null)}`,
    `isLoading=${String(state?.isLoading)}`,
    `totalElements=${String(state?.totalElements)}`,
    `expectedStepReached=${String(galleryGlobal.__luxarGalleryDimensionStepReached ?? false)}`,
  ].join(', ');
}
