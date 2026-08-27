import type { CoverageMeasurement } from './crop-policy';

const MEBIBYTE = 1024 * 1024;

export const GALLERY_MEDIA_WARNING_BYTES = 20 * MEBIBYTE;
export const GALLERY_MEDIA_LIMIT_BYTES = 25 * MEBIBYTE;

export interface GalleryMediaFile {
  demoId: string;
  extension: 'png' | 'webp' | 'webm';
  fileName: string;
  sizeBytes: number;
}

export function formatMediaSize(sizeBytes: number): string {
  return `${(sizeBytes / MEBIBYTE).toFixed(2)} MiB`;
}

export function checkGalleryMediaSize(media: GalleryMediaFile): { warning: string | undefined } {
  const size = formatMediaSize(media.sizeBytes);
  if (media.sizeBytes >= GALLERY_MEDIA_LIMIT_BYTES) {
    throw new Error(
      `[${media.demoId}] ${media.fileName} is ${size}; Pages requires each file below 25.00 MiB`
    );
  }
  return {
    warning:
      media.sizeBytes >= GALLERY_MEDIA_WARNING_BYTES
        ? `[${media.demoId}] ${media.fileName} is ${size}; the Pages limit is 25.00 MiB`
        : undefined,
  };
}

export function summarizeGalleryMedia(
  media: readonly GalleryMediaFile[],
  largestCount = 5
): { totalLine: string; largestLines: string[] } {
  const totalBytes = media.reduce((sum, file) => sum + file.sizeBytes, 0);
  const largestLines = [...media]
    .sort((left, right) => right.sizeBytes - left.sizeBytes)
    .slice(0, largestCount)
    .map((file) => `${formatMediaSize(file.sizeBytes).padStart(11)}  ${file.fileName}`);
  const fileLabel = media.length === 1 ? 'file' : 'files';
  return {
    totalLine: `Media: ${formatMediaSize(totalBytes)} total across ${media.length} ${fileLabel}`,
    largestLines,
  };
}

export function formatGalleryCaptureMetrics(
  measurement: CoverageMeasurement | null,
  totalDroppedElements: number
): string {
  if (!measurement) {
    return `final coverage=unmeasured final lit=unmeasured dropped=${totalDroppedElements}`;
  }
  return (
    `final coverage=${(measurement.coverage * 100).toFixed(1)}% ` +
    `final lit=${(measurement.litFraction * 100).toFixed(1)}% ` +
    `dropped=${totalDroppedElements}`
  );
}
