#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/batch_config.json"

usage() {
  cat <<'EOF'
Usage:
  bash run_all_predicate_videos.sh \
    --predicate-output /path/to/outputs_python/RUN_NAME \
    [--category NAME | --categories A,B,C] \
    [--max-scenarios N|all] \
    [--workers N] \
    [--export-root DIR] \
    [--config FILE] \
    [--overwrite]

Render every DETECTED predicate separately, using one invocation of
run_category_videos.sh per predicate. Each predicate gets its own directory.

Options:
  --predicate-output DIR   Required predicate extraction directory.
  --category NAME         One category.
  --categories A,B,C      Several categories. If omitted, all detected catalog categories are used.
  --max-scenarios N|all   Maximum unique scenarios per predicate. Default: 10
  --workers N             Parallel scenario workers for each predicate. Default: 8
  --export-root DIR       Optional base visualization directory.
  --config FILE           Base batch_config.json. Default: package batch_config.json
  --overwrite             Re-render completed scenarios.
  -h, --help              Show this help.
EOF
}

PREDICATE_OUTPUT=""
CATEGORY_SPEC=""
WORKERS="8"
MAX_SCENARIOS="10"
CUSTOM_EXPORT_ROOT=""
CONFIG_PATH="$DEFAULT_CONFIG"
OVERWRITE=0

while (($#)); do
  case "$1" in
    --predicate-output)
      [[ $# -ge 2 ]] || { echo "ERROR: --predicate-output requires DIR" >&2; exit 2; }
      PREDICATE_OUTPUT="$2"; shift 2 ;;
    --category|--categories)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      [[ -z "$CATEGORY_SPEC" ]] || { echo "ERROR: use only one of --category or --categories." >&2; exit 2; }
      CATEGORY_SPEC="$2"; shift 2 ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "ERROR: --workers requires N" >&2; exit 2; }
      WORKERS="$2"; shift 2 ;;
    --max-scenarios)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-scenarios requires N or all" >&2; exit 2; }
      MAX_SCENARIOS="$2"; shift 2 ;;
    --export-root)
      [[ $# -ge 2 ]] || { echo "ERROR: --export-root requires DIR" >&2; exit 2; }
      CUSTOM_EXPORT_ROOT="$2"; shift 2 ;;
    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: --config requires FILE" >&2; exit 2; }
      CONFIG_PATH="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$PREDICATE_OUTPUT" ]] || { echo "ERROR: --predicate-output is required." >&2; exit 2; }
[[ -d "$PREDICATE_OUTPUT" ]] || { echo "ERROR: predicate-output directory not found: $PREDICATE_OUTPUT" >&2; exit 2; }
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config not found: $CONFIG_PATH" >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --workers must be a positive integer." >&2; exit 2; }
if [[ "$MAX_SCENARIOS" != "all" ]]; then
  [[ "$MAX_SCENARIOS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --max-scenarios must be a positive integer or all." >&2; exit 2; }
fi

# Do NOT import nuplan_predicates_modular.base here: base.py parses extraction CLI
# arguments at import time. Read make_predicate(...) declarations as text instead.
mapfile -t PREDICATE_ROWS < <(
python - "$PREDICATE_OUTPUT" "$CATEGORY_SPEC" "${SCRIPT_DIR}/nuplan_predicates_modular/base.py" <<'PYSCAN'
from pathlib import Path
import re
import sys
import pandas as pd

output_dir = Path(sys.argv[1]).expanduser().resolve()
category_spec = sys.argv[2].strip()
base_py = Path(sys.argv[3]).resolve()

source = base_py.read_text(encoding="utf-8")
pattern = re.compile(
    r'make_predicate\(\s*["\'](?P<pid>np:[^"\']+)["\']\s*,\s*["\'](?P<category>[^"\']+)["\']',
    re.MULTILINE,
)
catalog = {m.group("pid"): m.group("category") for m in pattern.finditer(source)}
if not catalog:
    raise SystemExit("ERROR: could not parse predicate catalog from base.py")

all_categories = sorted(set(catalog.values()))
if category_spec:
    requested = [x.strip() for x in category_spec.split(",") if x.strip()]
else:
    requested = all_categories
requested = list(dict.fromkeys(requested))
unknown = [x for x in requested if x not in set(all_categories)]
if unknown:
    raise SystemExit(
        "ERROR: unknown category/categories: " + ", ".join(unknown)
        + "\nAvailable: " + ", ".join(all_categories)
    )
requested_set = set(requested)

present = set()
for path in sorted(output_dir.rglob("*.parquet")):
    try:
        df = pd.read_parquet(path, columns=["predicate_id"])
    except Exception:
        continue
    if "predicate_id" in df.columns:
        present.update(df["predicate_id"].dropna().astype(str).unique().tolist())

rows = sorted(
    (catalog[pid], pid)
    for pid in present
    if pid in catalog and catalog[pid] in requested_set
)

for category, predicate_id in rows:
    print(f"{category}\t{predicate_id}")
PYSCAN
)

if [[ "${#PREDICATE_ROWS[@]}" -eq 0 ]]; then
  echo "ERROR: no detected predicates were found for the selected category/categories." >&2
  exit 1
fi

echo
echo "Per-predicate video export"
echo "--------------------------"
echo "Predicate output:  $PREDICATE_OUTPUT"
echo "Categories:        ${CATEGORY_SPEC:-all detected categories}"
echo "Detected predicates: ${#PREDICATE_ROWS[@]}"
echo "Max scenarios/predicate: $MAX_SCENARIOS"
echo "Workers:           $WORKERS"
echo

index=0
for row in "${PREDICATE_ROWS[@]}"; do
  index=$((index + 1))
  category="${row%%$'\t'*}"
  predicate_id="${row#*$'\t'}"
  predicate="${predicate_id#np:}"

  echo "[$index/${#PREDICATE_ROWS[@]}] ${category} :: ${predicate}"

  cmd=(
    bash "${SCRIPT_DIR}/run_category_videos.sh"
    --predicate-output "$PREDICATE_OUTPUT"
    --category "$category"
    --predicate "$predicate"
    --max-scenarios "$MAX_SCENARIOS"
    --workers "$WORKERS"
    --config "$CONFIG_PATH"
  )
  [[ -z "$CUSTOM_EXPORT_ROOT" ]] || cmd+=(--export-root "$CUSTOM_EXPORT_ROOT")
  [[ "$OVERWRITE" -eq 0 ]] || cmd+=(--overwrite)

  "${cmd[@]}"
  echo
done

echo "All detected predicates finished."
