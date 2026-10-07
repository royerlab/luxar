/**
 * The key a keyboard binding names, lowercased.
 *
 * macOS Option composes characters: Option+W reports `event.key === '∑'`
 * (Option+E a dead key), so an Alt binding on a letter would never match.
 * When Alt is held and the reported key is not a plain letter or digit, the
 * key is read from the physical `event.code` instead. Everywhere else —
 * including Alt chords on Windows/Linux, which keep the layout's letter — the
 * layout key wins, so non-QWERTY layouts keep their own letters.
 *
 * @module utils/keyboard-key
 */

const PLAIN_KEY = /^[a-z0-9]$/i;
const PHYSICAL_LETTER_OR_DIGIT = /^(?:Key([A-Z])|Digit([0-9]))$/;

/** The lowercased key `event` presses, undoing macOS Option composition. */
export function pressedKey(event: KeyboardEvent): string {
  if (event.altKey && !PLAIN_KEY.test(event.key)) {
    const physical = PHYSICAL_LETTER_OR_DIGIT.exec(event.code ?? '');
    if (physical) return (physical[1] ?? physical[2]).toLowerCase();
  }
  return event.key.toLowerCase();
}
