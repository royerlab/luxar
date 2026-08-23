/**
 * The hover placeholder vocabulary, and the one place it is substituted.
 *
 * A picked element resolves to a small set of values — its label, the layer
 * path it belongs to, its element index — and THREE different consumers turn
 * those into a string:
 *
 * | consumer            | escaping                        | where            |
 * | ------------------- | ------------------------------- | ---------------- |
 * | hover tooltip       | `escapeHtml` for `overlay_html` | `ui/overlay-manager.ts` |
 * | `link` (left-click) | `encodeURIComponent`, always    | `core/app/interaction/` |
 * | `copy` (right-click)| none — plain text to clipboard  | `core/app/interaction/` |
 *
 * They must agree on the vocabulary and disagree on the escaping, which is
 * exactly the shape that rots when it is written out three times. Hence one
 * substitution function taking a {@link TemplateMode}.
 *
 * Pure: no DOM, no config, no layer dependencies — it lives in `utils/`
 * beside `escape-html.ts` so `ui/` and `core/` can both reach it.
 *
 * @module utils/hover-template
 */

import { escapeHtml } from './escape-html';

/** Values a picked element contributes to the vocabulary. */
export interface HoverTemplateValues {
  /** The element's label, or null/empty when it has none. */
  label?: string | null;
  /** Reported layer path — the outermost `kind=partition` wrapper if any. */
  nodeName: string;
  /** Element index within the hit leaf. */
  elementIndex: number;
}

/**
 * How substituted values are escaped.
 *
 * - `text` — none. The result reaches `textContent`, which is inert.
 * - `html` — {@link escapeHtml}. The result reaches `innerHTML`.
 * - `url` — `encodeURIComponent`. The result is a URL that will be NAVIGATED
 *   to, so a value containing `/`, `?`, `#` or `&` must not be able to
 *   restructure it: `../../evil` in a label would otherwise traverse the
 *   target site's path.
 */
export type TemplateMode = 'text' | 'html' | 'url';

/** Result of a substitution pass. */
export interface HoverTemplateResult {
  /** The template with every recognised placeholder replaced. */
  text: string;
  /**
   * True when a placeholder the template actually REFERENCES resolved to an
   * empty string.
   *
   * The tooltip ignores this (it renders the gap and is gated separately on
   * having any content at all), but `link` and `copy` must not: a template
   * like `https://uniprot.org/{hover_label}` with no label yields
   * `https://uniprot.org/`, which is a plausible-looking URL pointing at the
   * wrong place. Suppressing beats navigating somewhere unintended.
   *
   * This is not hypothetical. Under `substitutive_lod=` the Python adders copy
   * every non-compositing attr onto the COARSE children as well as the finest
   * (`core/group/adders/points.py`, `child_attrs` / `gsplat_child_attrs`), and
   * those coarse levels are synthesised gsplats that carry no labels at all —
   * so a labelled layer with a link is guaranteed to hover label-less elements
   * at coarse LOD.
   */
  hadEmptySubstitution: boolean;
}

/**
 * Placeholders in the shared vocabulary.
 *
 * `{hover_image_label}` is deliberately absent: it expands to an `<img>`
 * element and needs overlay config (`hover_image_size`) to size it, so it
 * stays in `OverlayManager` where both are in scope. It is meaningless in a
 * URL or on the clipboard.
 */
const PLACEHOLDER_PATTERN = /\{(hover_label|hover_node|hover_index)\}/g;

/** Escape a substituted value for the target consumer. */
function escapeFor(mode: TemplateMode, value: string): string {
  switch (mode) {
    case 'html':
      return escapeHtml(value);
    case 'url':
      return encodeURIComponent(value);
    case 'text':
      return value;
  }
}

/**
 * Substitute the hover vocabulary into `template`.
 *
 * Unrecognised `{...}` runs are left verbatim rather than blanked — an author
 * writing `{hover_labl}` gets a visibly wrong string they can find, instead of
 * a silent hole. For the same reason an unknown placeholder does NOT count as
 * an empty substitution: it was never a value we failed to supply.
 */
export function substituteHoverTemplate(
  template: string,
  values: HoverTemplateValues,
  mode: TemplateMode
): HoverTemplateResult {
  let hadEmptySubstitution = false;

  const text = template.replace(PLACEHOLDER_PATTERN, (_match, name: string) => {
    let raw: string;
    switch (name) {
      case 'hover_label':
        raw = values.label ?? '';
        break;
      case 'hover_node':
        raw = values.nodeName;
        break;
      case 'hover_index':
        // Always non-empty for a real pick. Not treated as suppressible: a
        // valid index of 0 stringifies to "0", which is falsy in JS and would
        // otherwise kill every link on the first element of a layer.
        raw = String(values.elementIndex);
        break;
      default:
        return _match as string;
    }
    if (raw === '') hadEmptySubstitution = true;
    return escapeFor(mode, raw);
  });

  return { text, hadEmptySubstitution };
}
