/**
 * Persisted listener preferences: the rail mute and the master gain.
 *
 * One global key (`StorageKeys.audio`), not per-scene like the rendering blob:
 * a kiosk that was muted stays muted across dataset switches, and audio is a
 * listener's preference rather than a scene's look.
 *
 * @module audio/audio-prefs
 */

import { StorageKeys } from '../utils/storage-keys';

/** What the listener chose: the rail mute and the master gain. */
export interface AudioPrefs {
  muted: boolean;
  masterGain: number;
}

function clamp01to2(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  return Math.min(2, Math.max(0, value));
}

/** Read the stored preferences; missing or malformed fields are simply absent. */
export function loadAudioPrefs(): Partial<AudioPrefs> {
  try {
    if (typeof localStorage === 'undefined') return {};
    const raw = localStorage.getItem(StorageKeys.audio);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const prefs: Partial<AudioPrefs> = {};
    if (typeof parsed.muted === 'boolean') prefs.muted = parsed.muted;
    const gain = clamp01to2(parsed.masterGain);
    if (gain !== undefined) prefs.masterGain = gain;
    return prefs;
  } catch {
    return {};
  }
}

/** Merge and persist chosen fields (quota/private-mode refusals are swallowed). */
export function saveAudioPrefs(patch: Partial<AudioPrefs>): void {
  try {
    if (typeof localStorage === 'undefined') return;
    const prefs = { ...loadAudioPrefs(), ...patch };
    localStorage.setItem(
      StorageKeys.audio,
      JSON.stringify({
        ...(prefs.muted !== undefined ? { muted: prefs.muted } : {}),
        ...(prefs.masterGain !== undefined
          ? { masterGain: clamp01to2(prefs.masterGain) ?? 0.8 }
          : {}),
      })
    );
  } catch {
    /* ignore */
  }
}
