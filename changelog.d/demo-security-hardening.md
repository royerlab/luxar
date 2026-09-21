#### Reading the CAFA5 accessions no longer runs the bundle's code

`demo_protein_embeddings_cafa5.py` read `train_ids.npy` with
`np.load(..., allow_pickle=True)`. That file arrives inside a Kaggle dataset
its owner can replace at any time, and the download is verified by size rather
than content, so unpickling it executes whatever the file says to execute
before a single accession is read. Demonstrated: a crafted `.npy` ran its
payload under the old call and is refused under the new one.

Accessions now load with `allow_pickle=False` first — a fixed-width string
array needs no pickle at all — and an object array falls through to an
unpickler whose `find_class` permits only numpy's array-reconstruction
primitives. A pickle naming `os.system` raises instead of running. This stops
code execution; it does not make an untrusted array trustworthy.

Three smaller items in the same pass. The `requests` floor moves to 2.32.4 for
GHSA-9hjg-9r4m-mvj7, where a redirect could hand the original host's `.netrc`
credentials to the redirect target. Demo narration no longer selects the paid
OpenAI engine just because `OPENAI_API_KEY` happens to be exported — that key
is ambient on many machines for unrelated reasons, and picking it up spent
someone else's money without a decision; the paid path now needs an explicit
`engine="openai"` or the engine environment variable, and the free local engine
is preferred when nobody asked. And a shipped docstring in `cli/serving.py`
recorded two specific private LAN addresses from the author's machine; the
reasoning they illustrate is load-bearing and stays, the addresses do not.
