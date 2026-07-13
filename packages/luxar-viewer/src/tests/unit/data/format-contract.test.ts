import { describe, it, expect } from 'vitest';
import {
  ENCODING_NAMES,
  GSPLATS_FORMAT_VERSION,
  SUPPORTED_GSPLATS_FORMAT_VERSIONS,
  NODE_TYPES,
  NODE_KINDS,
} from '../../../types/format-contract';
import { detectEncoding } from '../../../data/loaders/spatial-query/range-loader/detect-encoding';

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
