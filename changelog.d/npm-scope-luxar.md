#### Publish the viewer as `@luxar/viewer`

The viewer is published under the `@luxar` npm scope instead of
`@royerlab/luxar-viewer`. The scope now matches the project rather than the
lab that hosts it, so the package name stays right if the repository ever
moves.

Nothing about the bundle changes: the entry points are still
`dist/lib/luxar-viewer.js` and `dist/lib/luxar-viewer.css`, and the export map
still offers `.` and `./styles.css`. Only the name a consumer installs and
imports is different, and the release preflight names the new package when it
reports whether a tag will publish.

A guard test greps the tracked tree for the retired name, so a stray
`@royerlab/luxar-viewer`, a `luxar-viewer/...` module specifier, or an
`npm install luxar-viewer` line fails the suite rather than shipping. Because
the source directory `packages/luxar-viewer/` and the Cloudflare Pages project
legitimately keep the old spelling, the guard matches only surfaces where the
bare name can mean the published package and nothing else.
