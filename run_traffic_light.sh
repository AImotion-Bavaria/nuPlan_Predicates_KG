#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CATEGORIES="traffic_light"
export OUTPUT_NAME="${OUTPUT_NAME:-paper_traffic_control}"
export DERIVE_SPATIAL="${DERIVE_SPATIAL:-1}"
export DERIVE_SEMANTIC="${DERIVE_SEMANTIC:-1}"
exec bash "${SCRIPT_DIR}/run_predicates.sh" "$@"
