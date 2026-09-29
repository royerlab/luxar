#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/../../../scripts/ci_hatch_env_cache.sh" "$@"
