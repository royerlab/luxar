#### Layer edits live in the URL and travel as a JSON file

Re-windowing, recolouring or hiding layers in the Layers panel used to be lost on
reload and impossible to hand to a colleague. The viewer now mirrors those edits
into the page's `#layers=` fragment — only the fields that differ from the scene's
authored defaults, so an untouched scene adds nothing to the address bar — and
applies the fragment again when the scene loads, making the URL a share link.
Switching datasets drops it, and an embedded viewer never touches its host page's
URL (the mirror follows the same `updateBrowserUrl` opt-in as the `?src=` rewrite).

The same document can be copied as formatted JSON, downloaded as
`layer-settings.json`, or loaded from a file through the Layers header's right-click
menu; loading resets every layer to its authored state first, so the result is
exactly the file. Its shape is published as
`packages/luxar-viewer/schemas/layer-settings.v1.schema.json`, and a unit test keeps
the schema and the runtime validator on the same field list.
