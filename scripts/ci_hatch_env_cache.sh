#!/usr/bin/env bash
# LUXAR_CI_HATCH_BASE is a dedicated cache root on one persistent runner.
set -euo pipefail

if [[ -z ${LUXAR_CI_HATCH_BASE:-} ]]; then
  exit 0
fi

python_version=$(python -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
key="py${python_version}-$(date -u +%G-W%V)-${1}"
dir="${LUXAR_CI_HATCH_BASE}/${key}"
mkdir -p "$dir"
touch "$dir"
printf 'HATCH_DATA_DIR=%s\n' "$dir" >> "$GITHUB_ENV"
echo "reusing hatch env cache at ${dir}"

# Only this runner uses this root. Keep recent keys for jobs still using them.
find "$LUXAR_CI_HATCH_BASE" -mindepth 1 -maxdepth 1 -type d \
  -name 'py[0-9]*' -mmin +10080 ! -path "$dir" -exec rm -rf -- {} +
