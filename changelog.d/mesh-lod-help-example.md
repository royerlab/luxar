#### Validate mesh LOD methods shown in CLI help

A test derives the method names from the command callback's own docstring and
asserts each is valid, so the examples block can grow freely and only an invalid
method fails. Help text is the one place a wrong method name costs a user a round
trip rather than a type error, which is why it gets a test.
