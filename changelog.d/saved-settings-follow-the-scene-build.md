#### Saved settings follow the scene build; Settings shows the URL's values

The viewer keeps Rendering Controls edits per URL, and a saved document
replaces every authored rendering default at load. A scene rebuilt and served
at the same address, the normal life of an exhibit package, therefore never
showed its new defaults on a machine where anyone had once moved a slider.
Saved documents now record the scene's `content_hash`, and a load that finds
edits from another build resets to the new build's defaults instead. Edits made
on the current build keep winning, as before.

The Settings popover shows the saved machine preferences, which `?workers=`,
`?prefetch=` and `?renderer=` deliberately leave alone, so a launcher's values
were invisible there. A line under Performance now says what the URL set for
the page.

The protein universe kiosk places each structure caption in the turntable's
own units (an offset in `vw` from the clip's centre), so it sits under the
molecule on any screen shape, not only a square one.
