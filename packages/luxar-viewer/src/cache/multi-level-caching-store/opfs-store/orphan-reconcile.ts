/**
 * Orphan crawl for the OPFS L2 directory.
 *
 * An ORPHAN is a chunk file on disk that the persisted index
 * (`_cache_meta.json`) does not list: the index is saved on a debounce, so a
 * session killed (or navigated away) between a chunk write and the next index
 * save leaves files the index cannot name. Without this crawl they would never
 * be reclaimed (disk use grows across interrupted sessions) nor accounted in the
 * index's size and LRU. Serving them does not wait for it: until the crawl is
 * complete, `OPFSStore.get()` reads an unindexed key's file directly.
 *
 * This module only WALKS the bucket directories and hands each unlisted file to
 * the caller, under a hard per-session budget; the decision (re-index vs
 * delete) and every mutation of the index belong to `OPFSStore`, which owns the
 * race rules against its own in-flight writes.
 *
 * @module cache/multi-level-caching-store/opfs-store/orphan-reconcile
 */

import { fileNameToKey } from './buckets';

type IterableFileSystemDirectoryHandle = FileSystemDirectoryHandle & {
  keys(): AsyncIterableIterator<string>;
};

/** One file present on disk but absent from the index snapshot. */
export interface OrphanFile {
  bucket: FileSystemDirectoryHandle;
  bucketName: string;
  fileName: string;
  /** The key the name encodes, or null for a name this store cannot produce. */
  key: string | null;
}

export interface OrphanCrawlOptions {
  /** File names the index listed when the crawl began. */
  expectedFileNames: ReadonlySet<string>;
  /** Most orphans handed to `onOrphan` this crawl (the per-session budget). */
  maxOrphans: number;
  /** Most file names examined this crawl, orphan or not. */
  maxExamined: number;
  /** Re-polled between awaits; true halts the crawl (dispose / clear / breaker). */
  shouldStop: () => boolean;
  /** Decide and act on one orphan. Must not throw (errors are swallowed). */
  onOrphan: (orphan: OrphanFile) => Promise<void>;
}

export interface OrphanCrawlResult {
  examined: number;
  orphans: number;
  /** False when a budget or `shouldStop` ended the crawl early. */
  complete: boolean;
}

const HEX_BUCKET = /^[0-9a-f]{2}$/;

/**
 * Walk every hex bucket under `root`, calling `onOrphan` for each file whose
 * name is not in `expectedFileNames`, until a budget is spent or `shouldStop`.
 * Names are listed per bucket BEFORE acting, so a delete never mutates a
 * directory that is being iterated.
 */
export async function crawlOrphans(
  root: FileSystemDirectoryHandle,
  options: OrphanCrawlOptions
): Promise<OrphanCrawlResult> {
  const result: OrphanCrawlResult = { examined: 0, orphans: 0, complete: false };
  const bucketNames = await listNames(root, options, Number.POSITIVE_INFINITY);
  for (const bucketName of bucketNames) {
    if (!HEX_BUCKET.test(bucketName)) continue;
    if (!(await crawlBucket(root, bucketName, options, result))) return result;
  }
  result.complete = !options.shouldStop();
  return result;
}

/** Crawl one bucket. Returns false when the whole crawl must stop. */
async function crawlBucket(
  root: FileSystemDirectoryHandle,
  bucketName: string,
  options: OrphanCrawlOptions,
  result: OrphanCrawlResult
): Promise<boolean> {
  if (options.shouldStop()) return false;
  let bucket: FileSystemDirectoryHandle;
  try {
    bucket = await root.getDirectoryHandle(bucketName);
  } catch {
    return true; // bucket vanished mid-crawl; keep going
  }
  const names = await listNames(bucket, options, options.maxExamined - result.examined);
  for (const fileName of names) {
    if (options.shouldStop()) return false;
    result.examined++;
    if (options.expectedFileNames.has(fileName)) continue;
    if (result.orphans >= options.maxOrphans) return false;
    result.orphans++;
    try {
      await options.onOrphan({ bucket, bucketName, fileName, key: fileNameToKey(fileName) });
    } catch {
      // The handler owns its errors; one bad file must not end the crawl.
    }
  }
  return result.examined < options.maxExamined;
}

/** List up to `limit` entry names of a directory; an iteration error ends the list. */
async function listNames(
  dir: FileSystemDirectoryHandle,
  options: OrphanCrawlOptions,
  limit: number
): Promise<string[]> {
  const names: string[] = [];
  try {
    for await (const name of (dir as IterableFileSystemDirectoryHandle).keys()) {
      if (names.length >= limit || options.shouldStop()) break;
      names.push(name);
    }
  } catch {
    // Directory disappeared or is unreadable; work with what was listed.
  }
  return names;
}
