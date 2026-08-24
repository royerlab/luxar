/**
 * Resolve a picked element's authored interaction templates into a URL and a
 * copy string (issue #1917).
 *
 * Pure: takes a scene node and the pick's values, returns strings. All DOM
 * work — opening the link, the context menu, the clipboard — lives in
 * `canvas-actions.ts`, so everything security-relevant here is unit-testable
 * without a browser.
 *
 * ## Threat model
 *
 * A `.luxar.zarr` can be served from anywhere, and `.zattrs` is JSON the
 * viewer did not author. BOTH halves of a link are therefore hostile input:
 * the template (author-controlled) and the values substituted into it
 * (data-controlled, one per element). The Python writer runs an equivalent
 * check at authoring time, but that only covers stores written by Luxar —
 * nothing stops a hand-edited `.zattrs`, so the viewer re-validates rather
 * than trusting the producer.
 *
 * The guarantees, in the order they are established:
 *
 * 1. Substituted values are `encodeURIComponent`-escaped, so a label can
 *    contribute *content* to a URL but never *structure* — no injected query,
 *    fragment, path segment or authority. (`utils/hover-template.ts`.)
 * 2. The result is parsed with `new URL(built)` and **no base**, so a relative
 *    template throws instead of resolving against the viewer's own origin. A
 *    third-party store must not be able to aim a click at an embedder's site.
 * 3. The scheme must be `http:` or `https:` — an ALLOWLIST. This is
 *    deliberately stricter than the denylist `OverlayManager.sanitizeHtml`
 *    uses for rendered markup: that decides what to *display*, this decides
 *    where to *navigate*, and the safe set for navigation is small and known.
 * 4. Length is capped, so a hostile store cannot push an unbounded string at
 *    the browser after per-element substitution.
 *
 * One bounded gap is accepted knowingly, because closing it costs more than it
 * buys: `.` and `..` are *unreserved* characters, so `encodeURIComponent`
 * leaves them intact and the URL parser then normalizes them away — a label of
 * exactly `".."` turns `https://site/entry/{hover_label}` into `https://site/`.
 * That is a wrong destination, not a boundary crossing: the origin is fixed by
 * the template, a label is a single path segment (its slashes ARE encoded), so
 * the worst reachable outcome is the origin root. Escaping is therefore about
 * structure, not about pinning the exact path.
 *
 * @module core/app/interaction/element-actions
 */

import type * as THREE from 'three';
import { substituteHoverTemplate, type HoverTemplateValues } from '../../../utils/hover-template';

/**
 * Schemes a resolved link may use. Mirrors `LINK_SCHEMES` in
 * `luxar/typing_utils/constants.py`; keep the two in step.
 */
const ALLOWED_SCHEMES: ReadonlySet<string> = new Set(['http:', 'https:']);

/** Browsing contexts that imply `noopener`. Mirrors `LINK_TARGETS` in Python. */
const ALLOWED_TARGETS: ReadonlySet<string> = new Set(['_blank', '_self']);

/** Default when a node names no `link_target`. */
export const DEFAULT_LINK_TARGET = '_blank';

/** Ceiling on a resolved URL. Mirrors `MAX_LINK_CHARS` in Python. */
export const MAX_LINK_CHARS = 2048;

/** Ceiling on a resolved copy string. Mirrors `MAX_COPY_CHARS` in Python. */
export const MAX_COPY_CHARS = 8 * 1024;

/** The interaction templates a node may carry, as read from its `.zattrs`. */
export interface InteractionTemplates {
  link?: string;
  copy?: string;
  linkTarget?: string;
}

/** What a picked element offers once its templates are resolved. */
export interface ResolvedElementActions {
  /** A safe, absolute http(s) URL, or null when there is none to offer. */
  url: string | null;
  /** Where to open it. Always `_blank` or `_self`. */
  target: string;
  /** The string a `Copy` action should write, or null. */
  copyText: string | null;
}

/**
 * Read interaction templates from `node`, falling back to the nearest
 * ancestor that carries any.
 *
 * Both placements occur in practice, decided by `COMPOSITING_ATTRS` in
 * `luxar/core/group/compositing.py`: a non-compositing attr like `link` is
 * copied onto every `part_<i>` LEAF of a partition (and onto every child of a
 * substitutive-LOD ladder), while a compositing attr rides the WRAPPER alone.
 * `link` is deliberately in the first group — the leaf is what a pick hits —
 * but the walk upward costs nothing and makes the viewer indifferent to that
 * choice, including for stores written by hand or by a future adder.
 *
 * The first ancestor carrying ANY of the three keys wins outright; templates
 * are not merged across levels, so a leaf declaring only `copy` does not
 * silently inherit a grandparent's `link`.
 */
export function readInteractionTemplates(node: THREE.Object3D | null): InteractionTemplates {
  let cur: THREE.Object3D | null = node;
  while (cur) {
    const attrs = cur.userData?.attrs as Record<string, unknown> | undefined;
    if (attrs) {
      const link = typeof attrs.link === 'string' ? attrs.link : undefined;
      const copy = typeof attrs.copy === 'string' ? attrs.copy : undefined;
      const linkTarget = typeof attrs.link_target === 'string' ? attrs.link_target : undefined;
      if (link !== undefined || copy !== undefined || linkTarget !== undefined) {
        return { link, copy, linkTarget };
      }
    }
    cur = cur.parent;
  }
  return {};
}

/**
 * Build a safe URL from `template`, or null if it cannot be made safe.
 *
 * Returns null rather than throwing: an unusable link is a no-op click, not an
 * error condition, and the caller has nothing useful to do with an exception.
 * The reason is returned separately by {@link explainLinkRejection} for the
 * one caller that wants to log it once at scene load.
 */
export function buildElementUrl(template: string, values: HoverTemplateValues): string | null {
  const { text, hadEmptySubstitution } = substituteHoverTemplate(template, values, 'url');

  // A referenced placeholder that resolved empty leaves a hole:
  // `https://uniprot.org/{hover_label}` with no label becomes
  // `https://uniprot.org/`, which is a perfectly valid URL pointing at
  // entirely the wrong thing. Under substitutive LOD this is the NORMAL case
  // at coarse levels, which carry the attrs but no labels/keys — so it must be
  // suppression, not a best guess.
  if (hadEmptySubstitution) return null;

  if (text.length > MAX_LINK_CHARS) return null;

  let parsed: URL;
  try {
    // No base argument: a relative template throws here rather than resolving
    // against the viewer's own origin.
    parsed = new URL(text);
  } catch {
    return null;
  }

  if (!ALLOWED_SCHEMES.has(parsed.protocol)) return null;
  // A URL like `https:///x` parses with an empty host and would navigate
  // somewhere unintended.
  if (!parsed.host) return null;
  // Refuse embedded credentials. `https://good.example@evil.example/` navigates
  // to evil.example while reading as good.example — including in the
  // `Copy link address` menu item, which is the one place a user might vet the
  // destination before following it. Nothing legitimate needs userinfo here
  // (credentials in a shareable scene file would be a mistake of its own), so
  // the deception vector is worth more than the capability.
  if (parsed.username !== '' || parsed.password !== '') return null;

  // Return the parsed form: it is normalized, and re-serializing what the
  // parser accepted removes any discrepancy between what was validated and
  // what is opened.
  return parsed.href;
}

/**
 * Why a template can never produce a usable URL, independent of any element.
 *
 * Called once per distinct template at scene load so an authoring mistake
 * surfaces as a log line naming the first node that carries it, instead of a
 * click that silently does nothing. Only reports element-INDEPENDENT faults: a
 * per-element failure (an empty label) is normal and must not be logged per
 * hover.
 *
 * Returns null when the template looks usable.
 */
export function explainLinkRejection(template: string): string | null {
  // Substitute placeholders with a harmless non-empty token so the shape can
  // be judged without an element. `x` cannot itself introduce structure.
  const probe = template.replace(/\{(hover_key|hover_label|hover_node|hover_index)\}/g, 'x');

  if (probe.length > MAX_LINK_CHARS) {
    return `link is ${probe.length} characters, over the ${MAX_LINK_CHARS} limit`;
  }
  let parsed: URL;
  try {
    parsed = new URL(probe);
  } catch {
    return "link is not an absolute URL (a relative link would resolve against the viewer's own origin)";
  }
  if (!ALLOWED_SCHEMES.has(parsed.protocol)) {
    return `link scheme "${parsed.protocol}" is not allowed (only http and https)`;
  }
  if (!parsed.host) return 'link has no host';
  if (parsed.username !== '' || parsed.password !== '') {
    return 'link embeds credentials (user:pass@host), which disguise the real destination';
  }
  return null;
}

/** Normalize a node's `link_target`, falling back to `_blank`. */
export function resolveLinkTarget(raw: string | undefined): string {
  return raw !== undefined && ALLOWED_TARGETS.has(raw) ? raw : DEFAULT_LINK_TARGET;
}

/**
 * Build the copy string, or null if there is none.
 *
 * Falls back to the bare label when the node authored no `copy` template, so
 * every labelled layer gets a working `Copy` with no authoring at all. Not
 * escaped — plain text destined for the clipboard is exactly what was asked
 * for — but length-capped, because it outlives the page.
 */
export function buildElementCopyText(
  template: string | undefined,
  values: HoverTemplateValues
): string | null {
  if (template === undefined) {
    const label = values.label ?? '';
    return label === '' ? null : label.slice(0, MAX_COPY_CHARS);
  }
  const { text, hadEmptySubstitution } = substituteHoverTemplate(template, values, 'text');
  if (hadEmptySubstitution) return null;
  if (text === '') return null;
  return text.length > MAX_COPY_CHARS ? text.slice(0, MAX_COPY_CHARS) : text;
}

/** Resolve everything a picked element offers, in one call. */
export function resolveElementActions(
  node: THREE.Object3D | null,
  values: HoverTemplateValues
): ResolvedElementActions {
  const templates = readInteractionTemplates(node);
  return {
    url: templates.link === undefined ? null : buildElementUrl(templates.link, values),
    target: resolveLinkTarget(templates.linkTarget),
    copyText: buildElementCopyText(templates.copy, values),
  };
}
