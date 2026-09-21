#### Preserve shear in node transforms

The viewer decomposed each node's authored 4×4 into position / quaternion /
scale. That factorisation cannot represent shear, so an ordinary
`compose(rotate, non_uniform_scale)` — which `luxar.transforms` advertises and
`examples/transform_example.py` authors — was silently rendered as different
geometry. The full affine matrix is now installed directly.
