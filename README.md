# Predicate KG for Driving-Scene Semantics

Reference repository for the paper **“Grounding Vision–Language Models in Driving Semantics: A Multi-Dataset Predicate Framework for Explainable Reasoning.”**

The repository materialises measurable driving-scene evidence as explicit Predicate Knowledge Graph (Predicate KG) assertions. The semantic vocabulary contains **83 paper predicates** grouped into spatial, motion, temporal, map, interaction, traffic-control, and risk families. Their operational definitions, equations, applicability conditions, and thresholds are documented in [`PREDICATES.md`](PREDICATES.md), following the paper appendix.

> **Scope.** The predicate definitions are dataset-independent and are the definitions used by the paper across nuPlan and nuScenes. The attached source repository supplied for this release contains the **nuPlan reference extraction pipeline**. This release therefore does not claim a separately executable nuScenes adapter that was not present in the supplied source tree.


## Paper predicate inventory

| Family | Count | Representative predicates |
|---|---:|---|
| Spatial | 12 | `inFrontOf`, `leftOf`, `overlapping`, `near` |
| Motion | 17 | `hasVelocity`, `hasSpeed`, `hasClosingSpeedTo` |
| Temporal | 17 | `precedes`, `hasObservedDuration`, `hasCenterDistanceChangeFromPrevious` |
| Map | 25 | `inLane`, `hasPrimaryLane`, `inCrosswalk`, `inSameLaneAs` |
| Interaction | 8 | `follows`, `queuesBehind`, `changesLane`, `mergesInFrontOf`, `mergesBehind`, `crossesInFrontOf`, `yieldsTo`, `overtakes` |
| Traffic-control | 3 | `controls`, `hasSignalState`, `isRelevantSignal` |
| Risk | 1 | `hasConflictRiskWith` |
| **Total** | **83** | |

The graph also contains native/structural assertions needed to represent timestamps, entities, geometry, provenance, and scenario metadata. These support extraction but are not counted among the 83 semantic predicates reported in the paper.

## Repository layout

```text
.
├── PREDICATES.md                 # Appendix-style specification of all 83 predicates
├── README.md
├── pyproject.toml
├── environment.yml
├── generate_nuplan_predicates.py # Main source-tree launcher
├── run_predicates.sh             # Recommended shell runner
├── nuplan_predicate_kg/          # Package, models, CLI, nuPlan adapter
├── nuplan_predicates_modular/    # Predicate extraction implementation
├── configs/                      # Scenario/filter examples
├── tests/                        # Regression and paper-alignment tests
└── ...                           # Export/visualisation utilities
```

## Requirements

The reference nuPlan pipeline targets the nuPlan devkit environment and **Python 3.9 or 3.10**. A reproducible environment snapshot is provided in `environment.yml`.

You need:

- the nuPlan devkit;
- a nuPlan dataset split/cache;
- nuPlan maps (`nuplan-maps-v1.0` for the supplied runner);
- the Python dependencies declared in `pyproject.toml`.

### Option A — use the supplied Conda environment

```bash
conda env create -f environment.yml
conda activate nuplan_predicates
python -m pip install -e . --no-deps
```

### Option B — install into an existing compatible nuPlan environment

```bash
python -m pip install -e .
```

If the nuPlan devkit is already installed in that environment, installing with `--no-deps` is also possible.

## Configure dataset paths

No user-specific absolute paths are embedded in the public runner. Either pass paths explicitly or set:

```bash
export NUPLAN_DATASET_ROOT=/path/to/nuplan/data/cache/mini
export NUPLAN_MAP_ROOT=/path/to/nuplan/maps
```

The fallback locations are `./data/nuplan` and `./data/maps` inside the repository.

## Run all paper predicate families

The recommended command is:

```bash
bash run_predicates.sh \
  --max-scenarios 10 \
  --workers 1 \
  --output-name paper_release
```

By default, the runner requests all seven semantic families:

```text
spatial,motion,temporal,map,interaction,traffic_light,risk
```

For a full catalogue run:

```bash
bash run_predicates.sh --max-scenarios all --max-frames all
```

You can also call the Python launcher directly:

```bash
python generate_nuplan_predicates.py \
  --project-root . \
  --dataset-root "$NUPLAN_DATASET_ROOT" \
  --map-root "$NUPLAN_MAP_ROOT" \
  --output-name paper_release \
  --categories spatial,motion,temporal,map,interaction,traffic_light,risk \
  --derive-spatial-predicates \
  --derive-semantic-predicates
```

Run `python generate_nuplan_predicates.py --help` for all extraction parameters.

## Outputs

Extraction outputs are written under:

```text
outputs_python/<output-name>/
```

The default runner writes compact Parquet output. JSONL, RDF, diagnostics, and other output modes remain available through the Python CLI. Assertions preserve provenance and evidence fields so derived relations can be traced to their source scene and operational rule.

## Predicate specification

See **[`PREDICATES.md`](PREDICATES.md)** for the full specification. It mirrors the appendix structure and includes:

- common notation;
- all 12 spatial predicates;
- all 17 motion predicates;
- all 17 temporal predicates;
- all 25 map predicates;
- all 8 interaction predicates;
- all 3 traffic-control predicates;
- the risk predicate and safety thresholds.

## Validation and tests

Run the repository regression suite with:

```bash
pytest -q
```

The suite includes explicit paper-alignment checks for:

- the exact 83-predicate semantic inventory;
- the UN R157-inspired safety gate used by `hasConflictRiskWith`;
- interaction, map, traffic-light, temporal, and visualization behavior retained from the supplied implementation.

The repository can be unit-tested without the nuPlan dataset because dataset access is only validated when the extraction CLI is actually run.

## Paper results

The paper validates the shared higher-level predicates on 200 scenarios from each dataset, reporting macro F1 of **0.94 on nuPlan** and **0.93 on nuScenes**, with an average cross-dataset F1 difference of **0.02** across the shared predicates. It further evaluates Predicate KG grounding on the nine NuPlanQA subtasks.

Those reported numbers are experimental results from the paper; running this repository on a different scenario selection, dataset release, configuration, or software environment can produce different aggregate counts/results.
