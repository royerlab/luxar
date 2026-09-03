import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';
import {
  GALLERY_MEDIA_LIMIT_BYTES,
  GALLERY_MEDIA_WARNING_BYTES,
  checkGalleryMediaSize,
  collectGalleryMediaSizeIssues,
  formatGalleryCaptureMetrics,
  galleryDroppedElementsWarning,
  skipGalleryMediaWhenRequested,
  summarizeGalleryMedia,
  type GalleryMediaFile,
} from '../screenshots/gallery-media-reporting';

function mediaFile(
  demoId: string,
  extension: 'png' | 'webp' | 'webm',
  sizeBytes: number
): GalleryMediaFile {
  return { demoId, fileName: `${demoId}.${extension}`, sizeBytes };
}

describe('gallery media reporting', () => {
  it('warns at 20 MiB and hard-fails at 25 MiB', () => {
    expect(
      checkGalleryMediaSize(mediaFile('safe', 'webm', GALLERY_MEDIA_WARNING_BYTES - 1))
    ).toEqual({ warning: undefined });
    expect(
      checkGalleryMediaSize(mediaFile('drifting', 'webm', GALLERY_MEDIA_WARNING_BYTES))
    ).toEqual({
      warning:
        '[drifting] drifting.webm is 20.00 MiB (20,971,520 bytes); the Pages limit is 25.00 MiB',
    });
    expect(
      checkGalleryMediaSize(mediaFile('near-limit', 'webm', GALLERY_MEDIA_LIMIT_BYTES - 1)).warning
    ).toBe(
      '[near-limit] near-limit.webm is 25.00 MiB (26,214,399 bytes); the Pages limit is 25.00 MiB'
    );
    expect(() =>
      checkGalleryMediaSize(mediaFile('oversized', 'webm', GALLERY_MEDIA_LIMIT_BYTES))
    ).toThrow(
      '[oversized] oversized.webm is 25.00 MiB (26,214,400 bytes); Pages requires each file below 25.00 MiB'
    );
  });

  it('summarizes total bytes and the largest files in descending order', () => {
    const summary = summarizeGalleryMedia(
      [
        mediaFile('small', 'png', 1 * 1024 * 1024),
        mediaFile('largest', 'webm', 24 * 1024 * 1024),
        mediaFile('middle', 'webp', 5 * 1024 * 1024),
      ],
      2
    );

    expect(summary.totalLine).toBe('Media: 30.00 MiB total across 3 files');
    expect(summary.largestLines).toEqual(['  24.00 MiB  largest.webm', '   5.00 MiB  middle.webp']);
  });

  it('collects every committed-media size failure before reporting', () => {
    const issues = collectGalleryMediaSizeIssues([
      mediaFile('warning', 'webm', GALLERY_MEDIA_WARNING_BYTES),
      mediaFile('first-oversized', 'webm', GALLERY_MEDIA_LIMIT_BYTES),
      mediaFile('safe', 'png', 1 * 1024 * 1024),
      mediaFile('second-oversized', 'webp', GALLERY_MEDIA_LIMIT_BYTES + 1),
    ]);

    expect(issues.warnings).toEqual([
      '[warning] warning.webm is 20.00 MiB (20,971,520 bytes); the Pages limit is 25.00 MiB',
    ]);
    expect(issues.errors).toEqual([
      '[first-oversized] first-oversized.webm is 25.00 MiB (26,214,400 bytes); Pages requires each file below 25.00 MiB',
      '[second-oversized] second-oversized.webp is 25.00 MiB (26,214,401 bytes); Pages requires each file below 25.00 MiB',
    ]);
  });

  it('deletes stale media only when that variant is intentionally skipped', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'luxar-gallery-media-'));
    const filePath = path.join(directory, 'demo.webm');
    try {
      fs.writeFileSync(filePath, 'stale video');

      expect(skipGalleryMediaWhenRequested(false, filePath)).toBe(false);
      expect(fs.existsSync(filePath)).toBe(true);

      expect(skipGalleryMediaWhenRequested(true, filePath)).toBe(true);
      expect(fs.existsSync(filePath)).toBe(false);
      expect(skipGalleryMediaWhenRequested(true, filePath)).toBe(true);
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it('prints renderer truncation beside final coverage metrics', () => {
    expect(formatGalleryCaptureMetrics({ coverage: 0.812, litFraction: 0.154 }, 37)).toBe(
      'final coverage=81.2% final lit=15.4% dropped=37'
    );
    expect(formatGalleryCaptureMetrics(null, 0)).toBe(
      'final coverage=unmeasured final lit=unmeasured dropped=0'
    );
    expect(galleryDroppedElementsWarning('truncated', 37)).toBe(
      '[truncated] renderer dropped 37 elements at capacity limits'
    );
    expect(galleryDroppedElementsWarning('complete', 0)).toBeUndefined();
  });
});
