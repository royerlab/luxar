import { describe, expect, it } from 'vitest';

import { ROOT_ATTR_DOCS, rootAttributes } from '../../../types/zarr-documents';

describe('ROOT_ATTR_DOCS', () => {
  it('tries the format-3 document FIRST', () => {
    // Steady state is format-3 datasets, so probing `zarr.json` first makes the
    // second request the exception rather than the rule. A format-2 store pays
    // one 404 per probe; reversing this would make every new dataset pay it.
    expect(ROOT_ATTR_DOCS[0]).toBe('zarr.json');
    expect([...ROOT_ATTR_DOCS]).toEqual(['zarr.json', '.zattrs']);
  });
});

describe('rootAttributes', () => {
  it('returns a format-2 .zattrs document unchanged — it IS the attributes', () => {
    const attrs = { content_hash: 'abc', timestamp: 't' };
    expect(rootAttributes(attrs)).toEqual(attrs);
  });

  it('unwraps a format-3 record to its nested attributes', () => {
    // Reading `content_hash` off the TOP level of a `zarr.json` yields
    // undefined, which the watchdog reads as "identity changed" on every poll
    // and cache validation reads as "no token at all".
    expect(
      rootAttributes({
        zarr_format: 3,
        node_type: 'group',
        attributes: { content_hash: 'abc' },
        consolidated_metadata: { metadata: {} },
      })
    ).toEqual({ content_hash: 'abc' });
  });

  it("never leaks a format-3 record's OWN keys as the node's attributes", () => {
    // `attributes` absent and `attributes: null` must both answer "no
    // attributes". Falling through to the record would surface `zarr_format`
    // and `node_type` as attrs — which compares unequal to the attrs the scene
    // was loaded with on every single poll, i.e. a permanent false "changed".
    for (const record of [
      { zarr_format: 3, node_type: 'group' },
      { zarr_format: 3, node_type: 'group', attributes: null },
      { zarr_format: 3, node_type: 'group', attributes: 'not-an-object' },
    ]) {
      expect(rootAttributes(record)).toEqual({});
    }
  });

  it('answers {} for a non-object body instead of throwing', () => {
    // Both callers parse a network response; a server that returns `null`, a
    // bare string or an array must not take the probe down.
    for (const body of [null, undefined, 'text', 42]) {
      expect(rootAttributes(body)).toEqual({});
    }
  });

  it('does not unwrap a format-2 document that happens to have an attributes key', () => {
    // Only `zarr_format === 3` selects the nested shape. A v2 `.zattrs` whose
    // producer stored a user attr literally named `attributes` is still the
    // attributes object itself.
    const attrs = { attributes: { nested: 1 }, content_hash: 'abc' };
    expect(rootAttributes(attrs)).toEqual(attrs);
  });
});
