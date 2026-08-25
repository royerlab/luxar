# DOM Utilities

Pure DOM predicates that are safe to reuse across viewer layers.

- `focus.ts` classifies typing surfaces and determines whether focus is on the
  document body or scene canvas.

Callers pass the active element explicitly, keeping the helpers deterministic
and straightforward to test with jsdom.
