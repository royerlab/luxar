/**
 * Where draw and pick decide "is this element here?" — and whether they decide
 * it the same way.
 *
 * Draw and pick are two shaders over one geometry. Every per-element
 * visibility rule (thinning, near fade, the label filter, …) is written into
 * both, and every time the two drifted a pick landed on something not drawn,
 * or a drawn element could not be picked: the capsule line pick ignored the
 * density guard's drop for a release; the gsplat pick quad once reached less
 * far than the drawn one. Neither changes a pixel of the drawn image, so no
 * image gate could see it.
 *
 * So each rule is declared here, for every `(type, primitive variant)`:
 *
 * - `'same'`: draw and pick both run the ONE shared helper, on both backends.
 *   `tests/unit/rendering/picking/draw-pick-parity.test.ts` asserts the
 *   helper's call in both GLSL sources and its emitted block in both TSL
 *   codegen snapshots (`src/tests/__codegen__`), so a pick that forgets the
 *   rule — or calls it without consuming the result, which TSL does not even
 *   emit — fails.
 * - `{ deliberate }`: the pick footprint is a deliberate SUBSET of the drawn
 *   one (pick ⊂ draw), with the evidence that it still is.
 * - `{ drawOnly }`: a draw-pass split that picking must not see.
 * - `{ neither }`: the rule does not exist for this type — asserted absent
 *   from both, so a rule that starts applying to it shows up here.
 *
 * The CPU-side rules (nD slice, slab, position clipping) need no row per
 * shader: the pick node REUSES the visual node's geometry, which the parity
 * test asserts for every type.
 *
 * The pixel-level half lives in the TSL harness (`tsl-shader-parity.spec.ts`,
 * pick ⊆ draw masks) and the render gate's `pickWithinDraw` judgement.
 *
 * @module tests/_conformance/pick-visibility-rules
 */

import type { GeometryTypeName } from '../../types/format-contract';
import type { PrimitiveVariant } from '../helpers/geometry-materials';

/** How one rule relates draw and pick for one `(type, variant)`. */
export type RuleVerdict =
  | 'same'
  | {
      readonly deliberate: string;
      readonly pickEvidence: RegExp;
      readonly tslEvidence: readonly RegExp[];
    }
  | { readonly drawOnly: string }
  | { readonly neither: string };

/** How a rule shows up in source: a GLSL call and its emitted TSL codegen. */
export interface RuleSignature {
  readonly what: string;
  /** A use (not the definition) in a GLSL shader source (vertex + fragment). */
  readonly glsl: RegExp;
  /**
   * Its trace in TSL. `codegen` is matched against the generated-GLSL snapshot
   * (`src/tests/__codegen__`) — the strong check, since an unconsumed TSL node
   * emits nothing. Where the snapshot fixture does not exercise the rule (a
   * label channel the fixture lacks), `source` is matched against the TSL
   * factory source instead.
   */
  readonly tsl:
    | {
        readonly codegen: RegExp;
        /**
         * The shared TSL helper's call in the factory source — checked too, so
         * a source edit fails here before the snapshot is regenerated.
         */
        readonly helper: RegExp;
        /** Snapshot stems where BOTH sides legitimately compile the rule out. */
        readonly exemptStems?: RegExp;
      }
    | { readonly source: RegExp };
}

/** Every shader-side visibility rule. */
export const VISIBILITY_RULES = {
  densityDrop: {
    what: 'the density guard’s hashed thinning (uDensityDrop)',
    glsl: /luxarDensityDropped\(\)(?!\s*\{)/,
    tsl: { codegen: /\* 2146121005u/, helper: /densityDroppedNode\(/ },
  },
  sortedIndexSlot: {
    what: 'the depth-sort draw slot: which ordering buffer addresses the element',
    glsl: /luxarSortedIndex\(\)(?!\s*\{)/,
    tsl: {
      codegen: /int\( aSortedIndex \) \* \( 1 - int\( nodeUniform\d+ \) \)/,
      helper: /sortedIndexNode\(/,
    },
  },
  nearFade: {
    what: 'the perspective near-plane cull and fade (uNearCull)',
    glsl: /perspectiveNearFade\((?!int )/,
    // An orthographic build bakes the fade to 1 (NDC clipping is the only
    // cull there), in the draw and the pick snapshot alike.
    tsl: {
      codegen: /smoothstep\( (\w+), \( \1 \* 2\.0 \), \( - /,
      helper: /perspectiveNearFadeTSL\(/,
      exemptStems: /-variant-ortho$/,
    },
  },
  labelFilter: {
    what: 'the class filter (uLabelFilterIndex) the Layers panel sets',
    glsl: /uLabelFilterIndex > 0/,
    tsl: { source: /uLabelFilterIndex/ },
  },
  reach: {
    what: 'the quad / footprint reach and truncation',
    glsl: /gsplatVisibleMahalSq\((?!float )/,
    tsl: { codegen: /\tnodeVar\d+ = \(.* \* 1\.05 \);/, helper: /gsplatQuadFootprintTSL\(/ },
  },
  alphaCutoff: {
    what: 'the opaque-mode alpha cutout (uAlphaCutoff)',
    glsl: /< uAlphaCutoff/,
    tsl: { source: /uAlphaCutoff/ },
  },
  glassPartition: {
    what: 'the refract_data glass depth partition (uGlassPartition)',
    glsl: /if \(uGlassPartition != 0\) \{/,
    tsl: { source: /glassPartition/i },
  },
} as const satisfies Record<string, RuleSignature>;

export type VisibilityRuleId = keyof typeof VISIBILITY_RULES;

const NO_THINNING = {
  neither: 'A shaded surface is never thinned; neither mesh material declares uDensityDrop.',
};
const NO_SORTED_INDEX = {
  neither: 'A mesh draw order IS its index buffer; nothing addresses an aSortedIndex slot.',
};
const NO_LABEL_FILTER = {
  neither: 'Only gsplats carry a per-element class index the panel filters by.',
};
const NO_CUTOUT = { neither: 'Only a mesh has an opaque alpha cutout.' };
const GLASS = {
  drawOnly:
    'The glass split draws each data fragment in one of two passes; picking must see all ' +
    'the data, so no pick material declares it (MESH_PHYSICAL_MATERIALS_SPEC §3.4).',
};
const NO_REACH_HELPER = {
  neither:
    'The reach-radius helper is the gsplat footprint; this type sizes its quad from its ' +
    'own extent.',
};

/** The table: every rule, for every type and variant. */
export const PICK_VISIBILITY_RULES: Readonly<
  Record<
    GeometryTypeName,
    Readonly<Partial<Record<PrimitiveVariant, Readonly<Record<VisibilityRuleId, RuleVerdict>>>>>
  >
> = {
  points: {
    quad: {
      densityDrop: 'same',
      sortedIndexSlot: 'same',
      nearFade: 'same',
      labelFilter: NO_LABEL_FILTER,
      reach: {
        deliberate:
          'The pick disc is 80% of the drawn radius, so a pick never lands on the faint rim.',
        pickEvidence: /basePointSize \* 0\.8/,
        tslEvidence: [/basePointSize\.mul\(0\.8\)/],
      },
      alphaCutoff: NO_CUTOUT,
      glassPartition: GLASS,
    },
  },
  lines: {
    quad: {
      densityDrop: 'same',
      sortedIndexSlot: 'same',
      nearFade: 'same',
      labelFilter: NO_LABEL_FILTER,
      reach: NO_REACH_HELPER,
      alphaCutoff: NO_CUTOUT,
      glassPartition: GLASS,
    },
    capsule: {
      densityDrop: 'same',
      sortedIndexSlot: 'same',
      nearFade: 'same',
      labelFilter: NO_LABEL_FILTER,
      reach: NO_REACH_HELPER,
      alphaCutoff: NO_CUTOUT,
      glassPartition: GLASS,
    },
  },
  gsplats: {
    quad: {
      densityDrop: 'same',
      sortedIndexSlot: 'same',
      nearFade: 'same',
      labelFilter: 'same',
      // The QUAD reaches as far as the draw (one shared rule), and the pick
      // fragment then truncates tighter, at 1.5σ: pick ⊂ draw.
      reach: {
        deliberate:
          'The pick quad shares the drawn reach radius, but the pick fragment truncates at ' +
          '1.5σ (Mahalanobis² 2.25) instead of the node’s own T.',
        pickEvidence: /if \(mahalSq > uTruncateSq\) discard;/,
        tslEvidence: [
          /Discard\(mahalSq\.greaterThan\(uTruncateSq\)\)/,
          /const truncate = .* \?\? 1\.5;/,
        ],
      },
      alphaCutoff: NO_CUTOUT,
      glassPartition: GLASS,
    },
  },
  mesh: {
    triangle: {
      densityDrop: NO_THINNING,
      sortedIndexSlot: NO_SORTED_INDEX,
      nearFade: 'same',
      labelFilter: NO_LABEL_FILTER,
      reach: {
        neither: 'A triangle covers exactly its own pixels in both passes; there is no reach.',
      },
      alphaCutoff: 'same',
      glassPartition: GLASS,
    },
  },
};
