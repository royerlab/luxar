"""``luxar gsplat batch-fit`` — whole-timelapse fitting at scale.

Owns the ``batch-fit`` sub-app and everything behind it: local multi-GPU
``run``, Slurm ``submit``, the shared planning/packing/validation helpers, and
the ``status`` / ``merge`` / ``validate`` / ``cancel`` commands. ``commands.py``
is the registration surface (``app_batch``).
"""
