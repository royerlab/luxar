/**
 * Escape HTML special characters to prevent XSS injection.
 *
 * Escapes `& < > " '` — including the apostrophe (`&#39;`) for
 * defense-in-depth against single-quoted attribute contexts. Callers
 * embedding into either attribute style (double- or single-quoted)
 * are safe.
 */
export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
