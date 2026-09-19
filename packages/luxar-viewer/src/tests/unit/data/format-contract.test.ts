import { describe, it, expect } from 'vitest';
import {
  ENCODING_NAMES,
  GSPLATS_FORMAT_VERSION,
  SUPPORTED_GSPLATS_FORMAT_VERSIONS,
  NODE_TYPES,
  NODE_KINDS,
} from '../../../types/format-contract';
import { detectEncoding } from '../../../data/loaders/spatial-query/range-loader/detect-encoding';
import {
  BLENDING_MODES as CONTRACT_BLENDING_MODES,
  BUILTIN_COLORMAP_NAMES as CONTRACT_COLORMAP_NAMES,
  LINE_JOIN_STYLES as CONTRACT_LINE_JOIN_STYLES,
  LINE_TYPES,
  LOD_SELECTORS,
  ND_TRANSFORM_PERMUTATION_KEY,
  TONE_MAPPINGS,
} from '../../../types/format-contract';
import { BLENDING_MODES } from '../../../types/blending';
import { LINE_JOIN_STYLES, DEFAULT_LINE_JOIN } from '../../../types/line-join';
import { isValidLineType } from '../../../types/lines';
import { isPermutation } from '../../../types/zarr';
import { BUILTIN_COLORMAPS } from '../../../rendering/colormap-data';
import { TONE_MAPPING_NAMES } from '../../../rendering/post-processing/tone-mapping';
import { config } from '../../../config';

/**
 * The consumer half of the cross-language format contract. These assert that
 * the TypeScript decoder actually understands every vocabulary term the Python
 * writer can emit (single-sourced from format-contract/contract.yaml). If the
 * contract grows an encoding the decoder does not classify, this fails — the
 * companion Python drift gate is `hatch run check-contract`.
 */
describe('format contract (TS consumer)', () => {
  it('decoder recognizes every contract encoding name', () => {
    for (const name of ENCODING_NAMES) {
      // detectEncoding throws `Unknown encoding name: ...` for a name it cannot
      // route to a decode path — the exact Python-emit <-> TS-decode drift we
      // want to catch. A *validation* error (e.g. missing `n_elements` on this
      // deliberately-minimal metadata) means the name IS recognized, so only
      // the "Unknown encoding name" failure counts.
      try {
        detectEncoding({ encoding: { name } } as never);
      } catch (err) {
        expect(String(err), name).not.toContain('Unknown encoding name');
      }
    }
  });

  it('current gsplats format version is in the supported set', () => {
    expect(SUPPORTED_GSPLATS_FORMAT_VERSIONS).toContain(GSPLATS_FORMAT_VERSION);
  });

  it('shared node vocabulary is non-empty and unique', () => {
    for (const seq of [NODE_TYPES, NODE_KINDS, ENCODING_NAMES]) {
      expect(seq.length).toBeGreaterThan(0);
      expect(new Set(seq).size).toBe(seq.length);
    }
  });
});

/**
 * Vocabularies moved INTO the contract (B1-B10): each hand-written viewer
 * consumer must be the contract's projection, or a declared subset of it. The
 * Python twins live in `typing_utils/tests/test_format_contract.py`.
 */
describe('format contract (vocabularies single-sourced)', () => {
  it('B1: lod selectors are the two known metrics', () => {
    expect(new Set(LOD_SELECTORS)).toEqual(new Set(['coverage', 'screen-area']));
  });

  it('B2: types/blending re-exports the contract list', () => {
    expect(BLENDING_MODES).toBe(CONTRACT_BLENDING_MODES);
    expect(new Set(BLENDING_MODES)).toEqual(
      new Set(['additive', 'volumetric', 'normal', 'max', 'opaque', 'luminous'])
    );
  });

  it('B3: tone-mapping map is exhaustive over the contract, and the default is a member', () => {
    expect(new Set(TONE_MAPPING_NAMES)).toEqual(new Set(TONE_MAPPINGS));
    expect(TONE_MAPPINGS).toContain(config.renderingControls.defaults.toneMapping);
  });

  it('B4: generated colormap LUT keys equal the contract names, in order', () => {
    expect(Object.keys(BUILTIN_COLORMAPS)).toEqual([...CONTRACT_COLORMAP_NAMES]);
  });

  it('B7: line join styles and line types come from the contract', () => {
    expect(LINE_JOIN_STYLES).toBe(CONTRACT_LINE_JOIN_STYLES);
    expect(LINE_JOIN_STYLES).toContain(DEFAULT_LINE_JOIN);
    for (const t of LINE_TYPES) expect(isValidLineType(t)).toBe(true);
    expect(isValidLineType('bogus')).toBe(false);
    expect(isValidLineType(undefined)).toBe(false);
  });

  it('B8: the permutation guard keys on the contract permutation key', () => {
    expect(isPermutation({ [ND_TRANSFORM_PERMUTATION_KEY]: [1, 0] } as never)).toBe(true);
    expect(isPermutation({ scale: 2, offset: 1 })).toBe(false);
    expect(ND_TRANSFORM_PERMUTATION_KEY).toBe('permutation');
  });
});
