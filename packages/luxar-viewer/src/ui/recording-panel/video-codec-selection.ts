/**
 * Video codec selection + fallback chain. Maps the user's preferred
 * codec, downgrades to a container-compatible alternative, then
 * walks a fallback list of mediabunny codecs until one supports the
 * encoder options at the requested resolution.
 *
 * Used by `VideoModeDriver`. Pure async function with no panel
 * state — depends only on the user-selected codec, the container
 * mode, and the canvas size, so it tests in isolation.
 */

import type { VideoCodecOption } from './types';

/** mediabunny codec names. Keep in sync with mediabunny's `Codec` literal. */
export type MediabunnyCodec = 'av1' | 'vp9' | 'avc' | 'hevc' | 'vp8';

export interface CodecSelectionInput {
  /** User-selected codec from the recording panel UI. */
  preferredCodec: VideoCodecOption;
  /** Output container mode. */
  containerMode: 'webm' | 'mp4' | 'mkv';
  /** Encoder options shape required by mediabunny's canEncodeVideo. */
  encOpts: { width: number; height: number; bitrate: number };
  /**
   * Async predicate equivalent to mediabunny's `canEncodeVideo`. Caller
   * supplies the import so this module stays free of mediabunny
   * coupling and is unit-testable with a stub.
   */
  canEncodeVideo: (
    codec: MediabunnyCodec,
    opts: CodecSelectionInput['encOpts']
  ) => Promise<boolean>;
}

export interface CodecSelectionResult {
  /** The codec that should be used. Null if no fallback supports the request. */
  codec: MediabunnyCodec | null;
  /** True when the chosen codec is the user's preference; false on a fallback. */
  isPreferred: boolean;
  /** When isPreferred is false, the original preferred codec for logging. */
  fallbackFrom?: MediabunnyCodec;
}

/** WebM only supports vp9, av1, and vp8; avc/hevc need MP4 / MKV. */
const WEBM_ONLY: ReadonlySet<MediabunnyCodec> = new Set(['vp9', 'av1', 'vp8']);

const USER_TO_MEDIABUNNY: Readonly<Record<VideoCodecOption, MediabunnyCodec>> = {
  h265: 'hevc',
  vp9: 'vp9',
  h264: 'avc',
  vp8: 'vp8',
};

/**
 * Select a codec that the platform can actually encode at the
 * requested resolution and bitrate.
 *
 * Algorithm:
 * 1. Map the user's preference to a mediabunny codec.
 * 2. If the container is WebM and the mapped codec isn't WebM-compatible,
 *    pre-downgrade to vp9 (the most universally available WebM codec).
 * 3. Test the mapped codec with `canEncodeVideo`. Use it if supported.
 * 4. Otherwise walk a fallback list: HEVC's chain prefers AVC first;
 *    everything else prefers VP9. WebM-mode filters the list to its
 *    compatible subset.
 * 5. Return the first supported codec, or `null` if nothing works.
 */
export async function selectVideoCodec(input: CodecSelectionInput): Promise<CodecSelectionResult> {
  const { preferredCodec, containerMode, encOpts, canEncodeVideo } = input;

  const initial = USER_TO_MEDIABUNNY[preferredCodec];
  let codec = initial;

  // Container compatibility downgrade (no encode test needed).
  if (containerMode === 'webm' && !WEBM_ONLY.has(codec)) {
    codec = 'vp9';
  }

  if (await canEncodeVideo(codec, encOpts)) {
    return {
      codec,
      isPreferred: codec === initial,
      fallbackFrom: codec === initial ? undefined : initial,
    };
  }

  // Fallback chain: HEVC's own fallback prefers AVC; everything else
  // prefers VP9.
  const fullFallbacks: MediabunnyCodec[] =
    codec === 'hevc' ? ['avc', 'vp9', 'av1', 'vp8'] : ['vp9', 'av1', 'vp8', 'avc'];
  const fallbacks =
    containerMode === 'webm' ? fullFallbacks.filter((c) => WEBM_ONLY.has(c)) : fullFallbacks;

  for (const fb of fallbacks) {
    if (await canEncodeVideo(fb, encOpts)) {
      return { codec: fb, isPreferred: false, fallbackFrom: initial };
    }
  }

  return { codec: null, isPreferred: false, fallbackFrom: initial };
}
