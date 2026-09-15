#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Always prefer the code bundled with this repository.
# This prevents an older installed/editable nuplan_predicates_modular package
# from shadowing the paper-release implementation.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PIPELINE_FILE="${SCRIPT_DIR}/nuplan_predicates_modular/pipeline.py"
if [[ ! -f "${PIPELINE_FILE}" ]]; then
  echo "ERROR: bundled pipeline is missing: ${PIPELINE_FILE}" >&2
  echo "Delete the extracted package directory and unzip a clean copy." >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
DATASET_ROOT="${DATASET_ROOT:-${NUPLAN_DATASET_ROOT:-${SCRIPT_DIR}/data/nuplan}}"
MAP_ROOT="${MAP_ROOT:-${NUPLAN_MAP_ROOT:-${SCRIPT_DIR}/data/maps}}"

OUTPUT_NAME="${OUTPUT_NAME:-paper_release}"
MAX_SCENARIOS="${MAX_SCENARIOS:-all}"
MAX_FRAMES_PER_SCENARIO="${MAX_FRAMES_PER_SCENARIO:-all}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE_SCENARIOS="${BATCH_SIZE_SCENARIOS:-4}"
SCENARIO_FILTER_YAML="${SCENARIO_FILTER_YAML:-}"
CATEGORIES="${CATEGORIES:-spatial,motion,temporal,map,interaction,traffic_light,risk}"
PAIR_DIRECTIONS="${PAIR_DIRECTIONS:-ego-agent,agent-ego,agent-agent}"
PAIR_SELECTION="${PAIR_SELECTION:-relevance}"
PARQUET_COMPRESSION="${PARQUET_COMPRESSION:-zstd}"
MAP_CACHE_RESOLUTION_M="${MAP_CACHE_RESOLUTION_M:-0}"
DERIVE_SPATIAL="${DERIVE_SPATIAL:-1}"
DERIVE_SEMANTIC="${DERIVE_SEMANTIC:-1}"
WRITE_DIAGNOSTICS="${WRITE_DIAGNOSTICS:-0}"
COMPACT_PARQUET="${COMPACT_PARQUET:-1}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
CONFLICT_RISK_HORIZON_S="${CONFLICT_RISK_HORIZON_S:-3.0}"
CONFLICT_RISK_CLEARANCE_M="${CONFLICT_RISK_CLEARANCE_M:-0.5}"
CONFLICT_RISK_MIN_RELATIVE_SPEED_MPS="${CONFLICT_RISK_MIN_RELATIVE_SPEED_MPS:-0.5}"
CONFLICT_RISK_OPTIMIZATION_ITERATIONS="${CONFLICT_RISK_OPTIMIZATION_ITERATIONS:-24}"
CONFLICT_RISK_IMMINENT_BRAKING_DEMAND_MPS2="${CONFLICT_RISK_IMMINENT_BRAKING_DEMAND_MPS2:-5.0}"
CONFLICT_RISK_LANE_CHANGE_MAX_TARGET_DECEL_MPS2="${CONFLICT_RISK_LANE_CHANGE_MAX_TARGET_DECEL_MPS2:-3.0}"
CONFLICT_RISK_LANE_CHANGE_MIN_GAP_TIME_S="${CONFLICT_RISK_LANE_CHANGE_MIN_GAP_TIME_S:-1.0}"

usage() {
  cat <<'USAGE'
Usage:
  bash run_predicates.sh [options]

Main options:
  --yaml FILE, --scenario-filter-yaml FILE
  --no-scenario-filter
  --categories LIST             Default: all seven paper predicate families
  --max-scenarios N|all
  --max-frames N|all
  --workers N
  --output-name NAME

Paper risk model:
  --conflict-risk-horizon-s S        Default: 3.0
  --conflict-risk-clearance-m M      Default: 0.5
  --conflict-risk-min-relative-speed-mps V  Default: 0.5
  --conflict-risk-optimization-iterations N Default: 24
  --conflict-risk-imminent-braking-demand-mps2 A Default: 5.0
  --conflict-risk-lane-change-max-target-decel-mps2 A Default: 3.0
  --conflict-risk-lane-change-min-gap-time-s S Default: 1.0

Pair processing:
  --pair-directions LIST        Default: ego-agent,agent-ego,agent-agent
  --pair-selection VALUE        Default: relevance

Storage/performance:
  --compression zstd|snappy|gzip|none   Default: zstd
  --map-cache-resolution M              Default: 0 (exact proximal queries)
  --diagnostics                         Save large audit/debug CSV files
  --full-parquet                        Keep repeated full provenance/evidence
  --no-clean                            Keep an existing output directory

Derivation:
  --derive-spatial / --no-derive-spatial
  --derive-semantic / --no-derive-semantic

Notes:
  * --categories is a STRICT output selection.
  * Dependency categories may be computed internally but are not saved.
  * Observed-future labels and intent candidates are disabled in this runner.
  * Agent-agent computation is controlled only by --pair-directions.
USAGE
}

while (($#)); do
  case "$1" in
    --scenario-filter-yaml|--yaml)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a file." >&2; exit 2; }
      SCENARIO_FILTER_YAML="$2"; shift 2 ;;
    --no-scenario-filter)
      SCENARIO_FILTER_YAML=""; shift ;;
    --categories)
      [[ $# -ge 2 ]] || { echo "ERROR: --categories requires a value." >&2; exit 2; }
      CATEGORIES="$2"; shift 2 ;;
    --max-scenarios)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-scenarios requires a value." >&2; exit 2; }
      MAX_SCENARIOS="$2"; shift 2 ;;
    --max-frames)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-frames requires a value." >&2; exit 2; }
      MAX_FRAMES_PER_SCENARIO="$2"; shift 2 ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "ERROR: --workers requires a value." >&2; exit 2; }
      NUM_WORKERS="$2"; shift 2 ;;
    --output-name)
      [[ $# -ge 2 ]] || { echo "ERROR: --output-name requires a value." >&2; exit 2; }
      OUTPUT_NAME="$2"; shift 2 ;;
    --conflict-risk-horizon-s)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-horizon-s requires a value." >&2; exit 2; }
      CONFLICT_RISK_HORIZON_S="$2"; shift 2 ;;
    --conflict-risk-clearance-m)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-clearance-m requires a value." >&2; exit 2; }
      CONFLICT_RISK_CLEARANCE_M="$2"; shift 2 ;;
    --conflict-risk-min-relative-speed-mps)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-min-relative-speed-mps requires a value." >&2; exit 2; }
      CONFLICT_RISK_MIN_RELATIVE_SPEED_MPS="$2"; shift 2 ;;
    --conflict-risk-optimization-iterations)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-optimization-iterations requires a value." >&2; exit 2; }
      CONFLICT_RISK_OPTIMIZATION_ITERATIONS="$2"; shift 2 ;;
    --conflict-risk-imminent-braking-demand-mps2)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-imminent-braking-demand-mps2 requires a value." >&2; exit 2; }
      CONFLICT_RISK_IMMINENT_BRAKING_DEMAND_MPS2="$2"; shift 2 ;;
    --conflict-risk-lane-change-max-target-decel-mps2)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-lane-change-max-target-decel-mps2 requires a value." >&2; exit 2; }
      CONFLICT_RISK_LANE_CHANGE_MAX_TARGET_DECEL_MPS2="$2"; shift 2 ;;
    --conflict-risk-lane-change-min-gap-time-s)
      [[ $# -ge 2 ]] || { echo "ERROR: --conflict-risk-lane-change-min-gap-time-s requires a value." >&2; exit 2; }
      CONFLICT_RISK_LANE_CHANGE_MIN_GAP_TIME_S="$2"; shift 2 ;;
    --pair-directions)
      [[ $# -ge 2 ]] || { echo "ERROR: --pair-directions requires a value." >&2; exit 2; }
      PAIR_DIRECTIONS="$2"; shift 2 ;;
    --pair-selection)
      [[ $# -ge 2 ]] || { echo "ERROR: --pair-selection requires a value." >&2; exit 2; }
      PAIR_SELECTION="$2"; shift 2 ;;
    --compression)
      [[ $# -ge 2 ]] || { echo "ERROR: --compression requires a value." >&2; exit 2; }
      PARQUET_COMPRESSION="$2"; shift 2 ;;
    --map-cache-resolution)
      [[ $# -ge 2 ]] || { echo "ERROR: --map-cache-resolution requires a value." >&2; exit 2; }
      MAP_CACHE_RESOLUTION_M="$2"; shift 2 ;;
    --diagnostics)
      WRITE_DIAGNOSTICS=1; shift ;;
    --full-parquet)
      COMPACT_PARQUET=0; shift ;;
    --derive-spatial)
      DERIVE_SPATIAL=1; shift ;;
    --no-derive-spatial)
      DERIVE_SPATIAL=0; shift ;;
    --derive-semantic)
      DERIVE_SEMANTIC=1; shift ;;
    --no-derive-semantic)
      DERIVE_SEMANTIC=0; shift ;;
    --no-clean)
      CLEAN_OUTPUT=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

[[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --workers must be a positive integer." >&2
  exit 2
}
case "${PARQUET_COMPRESSION}" in
  zstd|snappy|gzip|none) ;;
  *) echo "ERROR: compression must be zstd, snappy, gzip, or none." >&2; exit 2 ;;
esac

OUTPUT_DIR="${PROJECT_ROOT}/outputs_python/${OUTPUT_NAME}"
if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
  echo "Cleaning target output directory: ${OUTPUT_DIR}"
  rm -rf -- "${OUTPUT_DIR}"
fi

EXTRA_FILTER_ARGS=()
if [[ -n "${SCENARIO_FILTER_YAML}" ]]; then
  [[ -f "${SCENARIO_FILTER_YAML}" ]] || {
    echo "ERROR: Scenario-filter YAML does not exist: ${SCENARIO_FILTER_YAML}" >&2
    exit 1
  }
  EXTRA_FILTER_ARGS+=(
    --scenario-filter-yaml "${SCENARIO_FILTER_YAML}"
    --filter-random-seed 42
  )
  echo "Scenario selection: YAML filter"
  echo "  ${SCENARIO_FILTER_YAML}"
else
  echo "Scenario selection: complete nuPlan catalogue"
fi

DERIVE_SPATIAL_ARG="--derive-spatial-predicates"
[[ "${DERIVE_SPATIAL}" == "1" ]] || DERIVE_SPATIAL_ARG="--no-derive-spatial-predicates"
DERIVE_SEMANTIC_ARG="--derive-semantic-predicates"
[[ "${DERIVE_SEMANTIC}" == "1" ]] || DERIVE_SEMANTIC_ARG="--no-derive-semantic-predicates"
DIAGNOSTICS_ARG="--no-write-diagnostics"
[[ "${WRITE_DIAGNOSTICS}" == "1" ]] && DIAGNOSTICS_ARG="--write-diagnostics"
COMPACT_ARG="--compact-parquet"
[[ "${COMPACT_PARQUET}" == "1" ]] || COMPACT_ARG="--no-compact-parquet"

echo "Predicate categories written: ${CATEGORIES}"
echo "Dependency categories: internal computation only"
echo "Pair directions: ${PAIR_DIRECTIONS}"
echo "Pair selection: ${PAIR_SELECTION}"
echo "Parquet compression: ${PARQUET_COMPRESSION}"
echo "Compact Parquet: $([[ ${COMPACT_PARQUET} == 1 ]] && echo enabled || echo disabled)"
echo "Detailed diagnostics: $([[ ${WRITE_DIAGNOSTICS} == 1 ]] && echo enabled || echo disabled)"
echo "Proximal map cache resolution: ${MAP_CACHE_RESOLUTION_M} m"
echo "Observed-future labels: disabled"
echo "Intent candidates: disabled"
echo "Maximum scenarios: ${MAX_SCENARIOS}"
echo "Maximum frames/scenario: ${MAX_FRAMES_PER_SCENARIO}"
echo "Workers: ${NUM_WORKERS}"
echo "Conflict risk: horizon=${CONFLICT_RISK_HORIZON_S}s clearance=${CONFLICT_RISK_CLEARANCE_M}m min_relative_speed=${CONFLICT_RISK_MIN_RELATIVE_SPEED_MPS}m/s"
echo "UN R157 safety gate: braking>=${CONFLICT_RISK_IMMINENT_BRAKING_DEMAND_MPS2}m/s^2, lane-change target decel>${CONFLICT_RISK_LANE_CHANGE_MAX_TARGET_DECEL_MPS2}m/s^2, min gap time=${CONFLICT_RISK_LANE_CHANGE_MIN_GAP_TIME_S}s"
echo "Output: ${OUTPUT_DIR}"

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/generate_nuplan_predicates.py" \
  --project-root "${PROJECT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --map-root "${MAP_ROOT}" \
  --map-version nuplan-maps-v1.0 \
  "${EXTRA_FILTER_ARGS[@]}" \
  --output-name "${OUTPUT_NAME}" \
  --categories "${CATEGORIES}" \
  --conflict-risk-horizon-s "${CONFLICT_RISK_HORIZON_S}" \
  --conflict-risk-clearance-m "${CONFLICT_RISK_CLEARANCE_M}" \
  --conflict-risk-min-relative-speed-mps "${CONFLICT_RISK_MIN_RELATIVE_SPEED_MPS}" \
  --conflict-risk-optimization-iterations "${CONFLICT_RISK_OPTIMIZATION_ITERATIONS}" \
  --conflict-risk-imminent-braking-demand-mps2 "${CONFLICT_RISK_IMMINENT_BRAKING_DEMAND_MPS2}" \
  --conflict-risk-lane-change-max-target-decel-mps2 "${CONFLICT_RISK_LANE_CHANGE_MAX_TARGET_DECEL_MPS2}" \
  --conflict-risk-lane-change-min-gap-time-s "${CONFLICT_RISK_LANE_CHANGE_MIN_GAP_TIME_S}" \
  --no-include-category-dependencies \
  --pair-selection "${PAIR_SELECTION}" \
  --pair-directions "${PAIR_DIRECTIONS}" \
  --max-frames-per-scenario "${MAX_FRAMES_PER_SCENARIO}" \
  --max-scenarios "${MAX_SCENARIOS}" \
  --batch-size-scenarios "${BATCH_SIZE_SCENARIOS}" \
  --num-workers "${NUM_WORKERS}" \
  --parquet-rows-per-part 250000 \
  --parquet-compression "${PARQUET_COMPRESSION}" \
  --map-cache \
  --proximal-map-cache-resolution-m "${MAP_CACHE_RESOLUTION_M}" \
  --write-parquet \
  --no-write-jsonl \
  --no-write-rdf \
  "${COMPACT_ARG}" \
  "${DIAGNOSTICS_ARG}" \
  "${DERIVE_SPATIAL_ARG}" \
  "${DERIVE_SEMANTIC_ARG}" \
  --no-derive-observed-future-labels \
  --no-derive-intent-candidates
