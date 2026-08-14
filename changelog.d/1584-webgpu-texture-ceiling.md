#### WebGPU line scenes break through the eight-thousand-row ceiling

The WebGPU device was created without requesting the adapter's
`maxTextureDimension2D`, so element-data textures were capped at the
spec-default 8192 rows — for lines that silently clamped every node to
5,586,944 segments while the same GPU's WebGL context rendered 11
million-plus. The renderer now requests the adapter's full texture
limit alongside the buffer limits it already raised, and the per-node
capacity chain picks it up automatically. On adapters exposing a 16384
texture limit (desktop GPUs) a 10 M-segment node now renders whole on
both backends; adapters capped at 8192 keep the previous ceiling.
