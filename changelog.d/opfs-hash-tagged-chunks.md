#### A tab at an old dataset version can no longer seed the disk cache of the new one

Two tabs of the same dataset URL share one disk-cache directory. When the
dataset was republished, the tab that opened the new version cleared that
directory, while a tab still showing the old version kept writing chunks into
it. Those files were not in the new index, so the next session recovered them
as unindexed files and served old bytes as the new dataset. Chunk file names
now carry a tag of the content hash they were written under, so a file of one
version is never read or recovered as another's; the reconcile deletes it. The
on-disk encoding version moves to 3, so existing disk caches start fresh once.
