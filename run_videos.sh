#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/batch_config.json}"
WORKERS="${WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: Missing batch configuration: ${CONFIG}" >&2
  exit 1
fi

readarray -t PATHS < <("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["assertion_output_dir"])
print(cfg["export_root"])
PY
)
ASSERTION_OUTPUT_DIR="${PATHS[0]}"
EXPORT_ROOT="${PATHS[1]}"
TOKEN_FILE="${EXPORT_ROOT}/_batch/detected_interaction_tokens.txt"
mkdir -p "$(dirname "${TOKEN_FILE}")"

"${PYTHON_BIN}" - "${ASSERTION_OUTPUT_DIR}" "${TOKEN_FILE}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
if not root.exists():
    raise SystemExit(f"Assertion output directory not found: {root}")

tokens = set()
files = sorted(root.rglob("*.parquet"))
if not files:
    raise SystemExit(f"No Parquet files found under: {root}")

for path in files:
    try:
        frame = pd.read_parquet(path, columns=["predicate_id", "prov_scenario_token"])
    except Exception:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
    if not {"predicate_id", "prov_scenario_token"}.issubset(frame.columns):
        continue
    rows = frame.loc[frame["predicate_id"].astype(str).isin({"np:changesLane", "np:mergesInFrontOf", "np:mergesBehind", "np:crossesInFrontOf", "np:yieldsTo", "np:overtakes"})]
    tokens.update(rows["prov_scenario_token"].dropna().astype(str))

if not tokens:
    raise SystemExit(
        "No np:changesLane, np:mergesInFrontOf, np:mergesBehind, np:crossesInFrontOf, np:yieldsTo, or np:overtakes assertions were found. Run run_interaction.sh first."
    )
out.write_text("\n".join(sorted(tokens)) + "\n", encoding="utf-8")
print(f"Scenarios containing lane-change/merge/crossing/overtake interactions: {len(tokens)}")
print(f"Token file: {out}")
PY

ARGS=(
  --config "${CONFIG}"
  --tokens-file "${TOKEN_FILE}"
  --workers "${WORKERS}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

exec bash "${SCRIPT_DIR}/run_parallel_exports.sh" "${ARGS[@]}"
