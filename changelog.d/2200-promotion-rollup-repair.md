#### Repair cancelled push checks after green promotion windows

Scheduled CI now reruns cancelled jobs for any of the five promotion-required contexts
from the push run on the same commit after those scheduled contexts pass. A rejected
rerun is reported without failing the scheduled workflow. This gives GitHub's
protected-branch rollup a best-effort path to converge to the successful verdict instead
of retaining cancelled duplicate checks that block `dev` to `main` promotion.
