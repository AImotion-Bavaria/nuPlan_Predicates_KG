#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the filter optional for compatibility with the established workflow.
# Without a YAML filter, run_predicates.sh scans the complete nuPlan catalogue.
SCENARIO_FILTER_YAML="${SCENARIO_FILTER_YAML:-}"
export SCENARIO_FILTER_YAML
export OUTPUT_NAME="${OUTPUT_NAME:-paper_interaction}"
export DERIVE_SPATIAL="${DERIVE_SPATIAL:-0}"
export DERIVE_SEMANTIC="${DERIVE_SEMANTIC:-1}"

# The shared runner invokes the paper-release source-tree launcher.
# Explicit default is intentionally visible here for review/tests. A later
# --categories supplied by the user overrides it inside run_predicates.sh.
exec bash "${SCRIPT_DIR}/run_predicates.sh" \
  --categories interaction,map \
  "$@"
