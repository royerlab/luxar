#### Fit parameters now cross the internal pipeline once (#2539, part 1/2)

`fit_gaussian_splats` remains the explicit documented API, but its raw fitting
parameters now enter a `FitParameters` dataclass before crossing
`GaussianSplatFitter.fit` and `prepare_fit_config`. Those two internal hops no
longer repeat 43 defaults apiece, so a new fit knob has one public signature and
one parameter field rather than three independently drifting signatures.

Direct low-level callers now pass `FitParameters(V=..., ...)` to
`GaussianSplatFitter.fit`, and `prepare_fit_config` accepts the same bundle. The
validation order and normalized `FitConfig` output are unchanged; the schema gate
now checks that the bundle covers every entry-point parameter not handled by the
wrapper and that every default still agrees.

#### Fit validation no longer lives in one guard wall (#2539, part 2/2)

`prepare_fit_config` now delegates input normalization and ordered validation to
focused helpers below the complexity ceiling. An all-pairs equivalence harness
pins every recorded validation failure against every later one, so exception
types, messages, and first-error precedence remain unchanged as the checks evolve.
