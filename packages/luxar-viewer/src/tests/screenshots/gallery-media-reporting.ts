/**
 * Gallery capture reporting for Cloudflare Pages static assets.
 *
 * Pages rejects individual files at 25 MiB, so captures warn from 20 MiB and
 * fail immediately after each encode. Checking completed files preserves the
 * encoder's constant-quality policy while making size drift visible before a
 * deployment fails.
 */
import fs from 'node:fs';

import type { CoverageMeasurement } from './crop-policy';

const MEBIBYTE = 1024 * 1024;

export const GALLERY_MEDIA_WARNING_BYTES = 20 * MEBIBYTE;
export const GALLERY_MEDIA_LIMIT_BYTES = 25 * MEBIBYTE;

/** Encoded gallery file metadata used by the guard and run summary. */
export interface GalleryMediaFile {
  demoId: string;
  fileName: string;
  sizeBytes: number;
}

/** Delete stale media when this capture intentionally skips that variant. */
export function skipGalleryMediaWhenRequested(skip: boolean, filePath: string): boolean {
  if (!skip) return false;
  fs.rmSync(filePath, { force: true });
  return true;
}

/** Format bytes as a two-decimal mebibyte value for compact logs. */
export function formatMediaSize(sizeBytes: number): string {
  return `${(sizeBytes / MEBIBYTE).toFixed(2)} MiB`;
}

/** Warn near the Pages limit and reject files at or above it. */
export function checkGalleryMediaSize(media: GalleryMediaFile): { warning: string | undefined } {
  const size = `${formatMediaSize(media.sizeBytes)} (${media.sizeBytes.toLocaleString('en-US')} bytes)`;
  const limit = formatMediaSize(GALLERY_MEDIA_LIMIT_BYTES);
  if (media.sizeBytes >= GALLERY_MEDIA_LIMIT_BYTES) {
    throw new Error(
      `[${media.demoId}] ${media.fileName} is ${size}; Pages requires each file below ${limit}`
    );
  }
  return {
    warning:
      media.sizeBytes >= GALLERY_MEDIA_WARNING_BYTES
        ? `[${media.demoId}] ${media.fileName} is ${size}; the Pages limit is ${limit}`
        : undefined,
  };
}

/** Collect warnings and failures without stopping at the first oversized file. */
export function collectGalleryMediaSizeIssues(media: readonly GalleryMediaFile[]): {
  warnings: string[];
  errors: string[];
} {
  const warnings: string[] = [];
  const errors: string[] = [];
  for (const file of media) {
    try {
      const { warning } = checkGalleryMediaSize(file);
      if (warning) warnings.push(warning);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  return { warnings, errors };
}

/** Summarize total encoded bytes and the largest files. */
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

/** Format final framing and renderer-capacity diagnostics on one line. */
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

/** Return a warning when renderer capacity dropped scene elements. */
export function galleryDroppedElementsWarning(
  demoId: string,
  totalDroppedElements: number
): string | undefined {
  return totalDroppedElements > 0
    ? `[${demoId}] renderer dropped ${totalDroppedElements} elements at capacity limits`
    : undefined;
}
