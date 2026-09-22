#### Refuse pickled accessions from the downloaded CAFA5 bundle

Both CAFA5 accession readers now use `np.load(..., allow_pickle=False)`. The
published bundle stores a fixed-width string array, so pickle is unnecessary;
an object-dtype replacement is rejected with a clear error rather than being
unpickled. The legacy UMAP-cache repair path uses the same loader.

This closes an unsafe trust boundary without claiming the previous end-to-end
bundle selector reached object-array pickle payloads: that selector already
rejects object dtype while matching the accessions file to the embeddings.

Three smaller items landed in the same pass. The `requests` floor moves to
2.32.4 for GHSA-9hjg-9r4m-mvj7, where a redirect could hand the original host's
`.netrc` credentials to the redirect target. Demo narration no longer selects
the paid OpenAI engine just because `OPENAI_API_KEY` happens to be exported;
the paid path now needs `LUXAR_NARRATION_ENGINE=openai` or an explicit
`engine="openai"`, and the free local engine is preferred when nobody asks.
Finally, a shipped docstring in `cli/serving.py` now uses generic private LAN
addresses rather than values copied from one machine.
