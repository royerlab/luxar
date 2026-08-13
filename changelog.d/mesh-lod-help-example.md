#### `luxar mesh lod --help` advertised a method the command rejects

The examples block showed `--subst-method qem`, and `MESH_SUBSTITUTIVE_METHODS`
is `{"auto", "cluster"}` — so the one example a user is most likely to copy
failed with a validation error. `qem` is a real algorithm the docs discuss as a
possible second decimation tier; it is simply not implemented, and the example
outlived the plan. It now reads `--subst-method cluster`.

A test derives the method names from the command callback's own docstring and
asserts each is valid, so the examples block can grow freely and only an invalid
method fails. Help text is the one place a wrong method name costs a user a round
trip rather than a type error, which is why it gets a test.
