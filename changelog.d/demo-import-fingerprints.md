#### Narrow demo scene staleness to imported producers

Fingerprint expensive demo scenes from each demo's reachable local Python imports instead of the entire Luxar production tree. Unrelated source edits no longer force every fingerprinted demo scene to rebuild, while shared scene-writing helpers and the Zarr writer environment still invalidate their consumers.
