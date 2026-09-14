import { ROOT_ATTR_DOCS, rootAttributes } from '../../types/zarr-documents';

interface RootDocumentResponse {
  ok: boolean;
  json: () => Promise<unknown>;
}

type FetchRootDocument = (url: string) => Promise<RootDocumentResponse>;

interface GalleryDataReadyOptions {
  requireElements: boolean;
  expectedDimensionStep: number[] | null;
}

interface GalleryDebugState {
  isLoading?: boolean;
  totalElements?: number;
  dimensions?: { currentStep?: number[] } | null;
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
  fetchRootDocument: FetchRootDocument = fetch
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

export function hasNonZeroDimensionStep(step: number[] | null): boolean {
  return step?.some((value) => value !== 0) ?? false;
}

export function isGalleryDataReady(options: GalleryDataReadyOptions): boolean {
  const debug = (
    globalThis as typeof globalThis & {
      __luxarDebug?: { getState?: () => GalleryDebugState | null };
    }
  ).__luxarDebug;
  const state = debug?.getState?.();
  if (!state || state.isLoading) return false;

  const expectedStep = options.expectedDimensionStep;
  if (expectedStep) {
    const currentStep = state.dimensions?.currentStep;
    if (!currentStep || expectedStep.some((value, index) => currentStep[index] !== value)) {
      return false;
    }
  }

  return options.requireElements ? (state.totalElements ?? 0) > 0 : true;
}
