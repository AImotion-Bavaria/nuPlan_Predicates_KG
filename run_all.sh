#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stable public families. Internal helper/dependency families are computed as
# needed but are not persisted unless explicitly requested via --categories.
export CATEGORIES="${CATEGORIES:-spatial,motion,temporal,map,interaction,traffic_light,risk}"
export OUTPUT_NAME="${OUTPUT_NAME:-paper_release}"
export DERIVE_SPATIAL="${DERIVE_SPATIAL:-1}"
export DERIVE_SEMANTIC="${DERIVE_SEMANTIC:-1}"

exec bash "${SCRIPT_DIR}/run_predicates.sh" "$@"
