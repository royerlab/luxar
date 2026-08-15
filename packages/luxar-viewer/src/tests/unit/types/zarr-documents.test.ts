import { describe, expect, it } from 'vitest';

import { ROOT_ATTR_DOCS, rootAttrDocOf, rootAttributes } from '../../../types/zarr-documents';

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

  it('trusts the DOCUMENT NAME over the content when it is given', () => {
    // The content sniff is a guess, and this is the input it guesses wrong on:
    // a format-2 `.zattrs` whose user attributes happen to carry a
    // `zarr_format` key. Told which document served the bytes, there is nothing
    // to guess — and both probes know, so both pass it.
    const v2WithAConfusingKey = { zarr_format: 3, content_hash: 'abc', mine: 1 };

    expect(rootAttributes(v2WithAConfusingKey)).toEqual({}); // sniffed: wrong
    expect(rootAttributes(v2WithAConfusingKey, '.zattrs')).toEqual(v2WithAConfusingKey);

    // ...and a real format-3 record still unwraps when named.
    const v3 = { zarr_format: 3, node_type: 'group', attributes: { content_hash: 'xyz' } };
    expect(rootAttributes(v3, 'zarr.json')).toEqual({ content_hash: 'xyz' });
  });

  it('the name may DEMOTE but never PROMOTE', () => {
    // A flat, non-v3 body served from a `zarr.json` address stays verbatim.
    // No real server does that, but test fakes and misconfigured proxies do,
    // and answering `{}` there would trade a reachable failure for a silent one.
    const flat = { content_hash: 'abc', timestamp: 't' };
    expect(rootAttributes(flat, 'zarr.json')).toEqual(flat);
  });

  it('does not unwrap a zarr.json-named document that carries no attributes', () => {
    // Named v3 with the member absent is "no attributes", not "fall through to
    // the record" — exposing `zarr_format`/`node_type` AS attributes is what
    // makes the watchdog report every poll as a change.
    expect(rootAttributes({ zarr_format: 3, node_type: 'group' }, 'zarr.json')).toEqual({});
  });
});

describe('rootAttrDocOf', () => {
  it('names the document a probe URL addresses', () => {
    expect(rootAttrDocOf('http://h/d.zarr/zarr.json')).toBe('zarr.json');
    expect(rootAttrDocOf('http://h/d.zarr/.zattrs')).toBe('.zattrs');
  });

  it('ignores a query string or fragment', () => {
    // The cache probe appends cache-busting params; matching the raw string
    // would silently fall back to sniffing on every such request.
    expect(rootAttrDocOf('http://h/d.zarr/zarr.json?v=2')).toBe('zarr.json');
    expect(rootAttrDocOf('http://h/d.zarr/.zattrs#frag')).toBe('.zattrs');
  });

  it('is undefined for a URL that addresses neither document', () => {
    expect(rootAttrDocOf('http://h/d.zarr/points/0.0')).toBeUndefined();
    // A path merely CONTAINING the name is not addressing it.
    expect(rootAttrDocOf('http://h/zarr.json.bak')).toBeUndefined();
  });
});
