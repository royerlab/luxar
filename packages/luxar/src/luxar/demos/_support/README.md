# Demo Support Helpers

This private subpackage contains leaf utilities owned exclusively by the demo
package. Modules here support several demos but are not demos themselves and do
not belong in the core runtime or shared `luxar.utils` package.

Broader workflow helpers that compose scenes or datasets remain as flat
`demos/_*_common.py` modules. Small implementation helpers such as UMAP colour
and legend utilities belong here.

Every non-`__init__.py` module in this directory is included in the demo guard
set defined by `demos/tests/_scanned_modules.py`.
