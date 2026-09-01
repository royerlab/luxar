#### Demo fetches can select exact manifest files

Demo datasets can now resolve an exact subset of manifest files, avoiding
downloads of unused siblings when a demo needs only part of a hosted dataset.
The cell-tracking demo uses this for `--datasets N`, while declared positional
pairs remain atomic so their rows cannot become misaligned.
