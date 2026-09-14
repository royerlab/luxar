#### Gallery captures wait for the scene's authored opening slice

The gallery harness now reads `viewer_config.dimensions.current_step` from each
scene and waits until the live debug state reaches that step before framing or
applying a manifest `dimensionNav` override. Visible geometry left over from an
earlier slice can no longer satisfy the readiness gate and produce a tile that
differs from the scene's real opening state. Scenes whose index-zero slice is
empty retain the relaxed pre-navigation element check.
