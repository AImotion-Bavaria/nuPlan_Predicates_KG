# ============================================================
# 16. FULL-SCENE CONFIGURABLE MAP-ONLY SEMANTIC VIDEO
# ============================================================
#
# This cell reuses the CURRENT configuration from Cell 15:
#   - SEMANTIC_CATEGORIES;
#   - SEMANTIC_RELATIONS;
#   - SEMANTIC_PREDICATES ("all" or exact predicate list);
#   - AGENT_TYPES;
#   - interested-agent rules and thresholds;
#   - selected-agent edge filtering;
#   - map radius, semantic arrows, labels, edge IDs, and legend limits.
#
# Change the switches in Cell 15, rerun Cell 15 (or at least its
# configuration statements), and then rerun this cell.
#
# This cell:
#   - reads assertion Parquet data because the master Excel contains only
#     one selected frame and cannot represent the complete scene;
#   - iterates through the complete selected nuPlan scene;
#   - renders ONLY the upper map figure;
#   - applies the same category/predicate/relation/agent/selection filters as Cell 15;
#   - saves one browser-compatible H.264 MP4 with yuv420p;
#   - keeps frames with zero enabled semantic relations.
#
# Run all function-definition cells above before running this cell.
# ============================================================

from collections import Counter as _VideoCounter
import shutil as _video_shutil
import subprocess as _video_subprocess
from IPython.display import Video as _NotebookVideo


# ============================================================
# VERIFY THAT CELL 15 CONFIGURATION EXISTS
# ============================================================

_REQUIRED_CELL_15_GLOBALS = [
    "SEMANTIC_CATEGORIES",
    "SEMANTIC_RELATIONS",
    "AGENT_TYPES",
    "INTERESTED_DISTANCE_THRESHOLD_M",
    "INCLUDE_WITHIN_DISTANCE",
    "INCLUDE_FORWARD_CORRIDOR",
    "FORWARD_CORRIDOR_LENGTH_M",
    "FORWARD_CORRIDOR_HALF_WIDTH_M",
    "INCLUDE_SAME_LANE",
    "INCLUDE_PREDICTED_PATH_INTERSECTION",
    "PREDICTION_HORIZON_S",
    "PREDICTION_STEP_S",
    "PATH_INTERSECTION_CLEARANCE_M",
    "INCLUDE_EXISTING_SPATIAL_AGENTS",
    "FILTER_EDGES_TO_SELECTED_AGENTS",
    "MANUAL_HIGHLIGHT_TOKENS",
    "MAP_RADIUS_M",
    "DRAW_SEMANTIC_ARROWS_ON_MAP",
    "SHOW_AGENT_LABELS",
    "SHOW_AGENT_LANE_INFO",
    "SHOW_MAP_STRUCTURE_IDS",
    "SHOW_EDGE_IDS",
    "SHOW_EDGE_LEGEND",
    "MAX_EDGE_LEGEND_ROWS",
    "SHOW_FILTERED_TABLES",
    "MAX_FILTERED_TABLE_ROWS",
]

_missing_cell_15_globals = [
    name
    for name in _REQUIRED_CELL_15_GLOBALS
    if name not in globals()
]

if _missing_cell_15_globals:
    raise RuntimeError(
        "Run the Cell 15 configuration first. Missing variables: "
        + ", ".join(_missing_cell_15_globals)
    )


# ============================================================
# SNAPSHOT THE CURRENT CELL 15 CONFIGURATION
# ============================================================
#
# These copies ensure that every rendered frame uses one consistent
# configuration, even if a notebook variable is edited while rendering.
# ============================================================

VIDEO_SEMANTIC_CATEGORIES = dict(SEMANTIC_CATEGORIES)
VIDEO_SEMANTIC_RELATIONS = dict(SEMANTIC_RELATIONS)
VIDEO_SEMANTIC_PREDICATES = globals().get("SEMANTIC_PREDICATES", "all")
VIDEO_AGENT_TYPES = dict(AGENT_TYPES)

VIDEO_INTERESTED_DISTANCE_THRESHOLD_M = float(
    INTERESTED_DISTANCE_THRESHOLD_M
)
VIDEO_INCLUDE_WITHIN_DISTANCE = bool(INCLUDE_WITHIN_DISTANCE)

VIDEO_INCLUDE_FORWARD_CORRIDOR = bool(INCLUDE_FORWARD_CORRIDOR)
VIDEO_FORWARD_CORRIDOR_LENGTH_M = float(FORWARD_CORRIDOR_LENGTH_M)
VIDEO_FORWARD_CORRIDOR_HALF_WIDTH_M = float(
    FORWARD_CORRIDOR_HALF_WIDTH_M
)

VIDEO_INCLUDE_SAME_LANE = bool(INCLUDE_SAME_LANE)

VIDEO_INCLUDE_PREDICTED_PATH_INTERSECTION = bool(
    INCLUDE_PREDICTED_PATH_INTERSECTION
)
VIDEO_PREDICTION_HORIZON_S = float(PREDICTION_HORIZON_S)
VIDEO_PREDICTION_STEP_S = float(PREDICTION_STEP_S)
VIDEO_PATH_INTERSECTION_CLEARANCE_M = float(
    PATH_INTERSECTION_CLEARANCE_M
)

VIDEO_INCLUDE_EXISTING_SPATIAL_AGENTS = bool(
    INCLUDE_EXISTING_SPATIAL_AGENTS
)
VIDEO_FILTER_EDGES_TO_SELECTED_AGENTS = bool(
    FILTER_EDGES_TO_SELECTED_AGENTS
)
VIDEO_MANUAL_HIGHLIGHT_TOKENS = list(MANUAL_HIGHLIGHT_TOKENS)

VIDEO_MAP_RADIUS_M = float(MAP_RADIUS_M)
VIDEO_DRAW_SEMANTIC_ARROWS_ON_MAP = bool(
    DRAW_SEMANTIC_ARROWS_ON_MAP
)
VIDEO_SHOW_AGENT_LABELS = bool(SHOW_AGENT_LABELS)
VIDEO_SHOW_AGENT_LANE_INFO = bool(SHOW_AGENT_LANE_INFO)
VIDEO_SHOW_MAP_STRUCTURE_IDS = bool(SHOW_MAP_STRUCTURE_IDS)
VIDEO_SHOW_EDGE_IDS = bool(SHOW_EDGE_IDS)

VIDEO_SHOW_EDGE_LEGEND = bool(SHOW_EDGE_LEGEND)
VIDEO_MAX_EDGE_LEGEND_ROWS = MAX_EDGE_LEGEND_ROWS
VIDEO_SHOW_FILTERED_TABLES = bool(SHOW_FILTERED_TABLES)
VIDEO_MAX_FILTERED_TABLE_ROWS = int(MAX_FILTERED_TABLE_ROWS)

_active_video_categories = sorted(
    enabled_names(VIDEO_SEMANTIC_CATEGORIES)
)
_active_video_relations = sorted(
    enabled_names(VIDEO_SEMANTIC_RELATIONS)
)

if not _active_video_categories:
    raise ValueError(
        "Enable at least one SEMANTIC_CATEGORIES entry in Cell 15."
    )

if not _active_video_relations:
    raise ValueError(
        "Enable at least one SEMANTIC_RELATIONS entry in Cell 15."
    )


# ============================================================
# VIDEO-ONLY CONFIGURATION
# ============================================================

VIDEO_EXPORT_ROOT = Path.cwd() / "visual_inspection_semantic_videos"

# Reuse the scenario selected for the master workbook by default.
VIDEO_SCENARIO_SELECTOR = globals().get(
    "MASTER_SCENARIO_SELECTOR",
    "random",
)
VIDEO_SCENARIO_VALUE = globals().get(
    "MASTER_SCENARIO_VALUE",
    0,
)
VIDEO_SCENARIO_SEED = int(
    globals().get("MASTER_SCENARIO_SEED", 42)
)
VIDEO_FILTER_SEED = int(
    globals().get("MASTER_FILTER_SEED", 42)
)

# Full scene by default. END=None means the last available frame.
VIDEO_START_FRAME = 0
VIDEO_END_FRAME = None
VIDEO_FRAME_STEP = 1

# None preserves approximately real-time nuPlan playback.
# Example: 2.0 saves a video that plays twice as fast.
VIDEO_PLAYBACK_SPEED = 1.0
VIDEO_FPS = None

VIDEO_FIGSIZE = (12, 12)
VIDEO_DPI = 120
VIDEO_SHOW_STATUS_BOX = False

# crossesInFrontOf debugging: draw the exact same finite 10 m forward
# segments used by the paper crossing detector for every active crossing.
# Set False to hide these debugging rays from the saved videos.
VIDEO_DRAW_CROSSES_REFERENCE_LINES = True
VIDEO_CROSSES_REFERENCE_RAY_LENGTH_M = 10.0


# Used only to obtain every spatial relation for the optional
# INCLUDE_EXISTING_SPATIAL_AGENTS candidate-selection rule.
_VIDEO_SPATIAL_ONLY_CATEGORIES = {
    name: (name == "spatial")
    for name in ALL_SEMANTIC_CATEGORIES
}
_VIDEO_ALL_RELATIONS = {
    name: True
    for name in ALL_SEMANTIC_RELATIONS
}


def _video_default_fps(frames, frame_indices, playback_speed):
    """Derive playback FPS from the selected nuPlan timestamps."""
    timestamps = np.asarray(
        [int(frames[index].timestamp_us) for index in frame_indices],
        dtype=np.int64,
    )

    if len(timestamps) < 2:
        return max(1.0, float(playback_speed))

    positive_deltas_s = np.diff(timestamps).astype(float) / 1e6
    positive_deltas_s = positive_deltas_s[positive_deltas_s > 0]

    if len(positive_deltas_s) == 0:
        return max(1.0, float(playback_speed))

    source_fps = 1.0 / float(np.median(positive_deltas_s))

    return float(
        np.clip(
            source_fps * float(playback_speed),
            0.25,
            60.0,
        )
    )


def _video_relation_label_counts(plot_edges):
    """Count the displayed semantic labels at one frame."""
    counts = _VideoCounter()

    if plot_edges is None or plot_edges.empty:
        return counts

    for row in plot_edges.itertuples(index=False):
        labels = str(
            getattr(row, "relation_labels", "")
            or "unspecified"
        )
        for label in [part.strip() for part in labels.split(",")]:
            if label:
                counts[label] += 1

    return counts


def _video_displayed_follow_mode_counts(frame_assertions, plot_edges):
    """Count follow modes only for np:follows edges displayed in this frame."""
    counts = _VideoCounter()

    if (
        frame_assertions is None
        or frame_assertions.empty
        or plot_edges is None
        or plot_edges.empty
    ):
        return counts

    displayed_follow_pairs = set()

    for edge in plot_edges.itertuples(index=False):
        predicates = {
            str(predicate)
            for predicate in getattr(edge, "predicates", [])
        }

        if "np:follows" not in predicates:
            continue

        displayed_follow_pairs.add((
            str(getattr(edge, "subject_token", "")),
            str(getattr(edge, "object_token", "")),
        ))

    if not displayed_follow_pairs:
        return counts

    follows = frame_assertions.loc[
        frame_assertions["predicate_id"]
        .astype(str)
        .eq("np:follows")
    ]

    for row in follows.itertuples(index=False):
        subject_token = str(
            entity_id_to_track_token(
                getattr(row, "subject_id", "")
            )
        )
        object_token = str(
            entity_id_to_track_token(
                getattr(row, "object_id", "")
            )
        )

        if (subject_token, object_token) not in displayed_follow_pairs:
            continue

        evidence = _evidence_mapping(
            getattr(row, "evidence_json", None)
        )
        mode = str(
            evidence.get("follow_mode")
            or "unspecified"
        )
        counts[mode] += 1

    return counts



def _video_follow_mode_by_pair(frame_assertions):
    """Return {(subject_token, object_token): follow_mode} for one frame."""
    result = {}

    if frame_assertions is None or frame_assertions.empty:
        return result

    follows = frame_assertions.loc[
        frame_assertions["predicate_id"]
        .astype(str)
        .eq("np:follows")
    ]

    for row in follows.itertuples(index=False):
        subject_token = str(
            entity_id_to_track_token(
                getattr(row, "subject_id", "")
            )
        )
        object_token = str(
            entity_id_to_track_token(
                getattr(row, "object_id", "")
            )
        )
        evidence = _evidence_mapping(
            getattr(row, "evidence_json", None)
        )
        result[(subject_token, object_token)] = str(
            evidence.get("follow_mode")
            or "unspecified"
        )

    return result

def _video_overtake_evidence_by_pair(frame_assertions):
    """Return active full-maneuver overtake evidence by directed pair."""
    result = {}

    if frame_assertions is None or frame_assertions.empty:
        return result

    rows = frame_assertions.loc[
        frame_assertions["predicate_id"]
        .astype(str)
        .eq("np:overtakes")
    ]

    for row in rows.itertuples(index=False):
        subject_token = str(
            entity_id_to_track_token(
                getattr(row, "subject_id", "")
            )
        )
        object_token = str(
            entity_id_to_track_token(
                getattr(row, "object_id", "")
            )
        )
        evidence = _evidence_mapping(
            getattr(row, "evidence_json", None)
        )
        result[(subject_token, object_token)] = evidence

    return result



def _video_restore_overtake_edges(
    semantic_edges_all,
    frame_assertions,
):
    """
    Restore np:overtakes edges that the generic semantic-edge converter may
    omit because np:overtakes is entity-valued and stores its target in
    object_id while value_json is null.

    This mirrors the existing compatibility recovery used for np:follows.
    """
    if semantic_edges_all is None:
        semantic_edges_all = pd.DataFrame()

    if frame_assertions is None or frame_assertions.empty:
        return semantic_edges_all

    if "predicate_id" not in frame_assertions.columns:
        return semantic_edges_all

    overtakes = frame_assertions.loc[
        frame_assertions["predicate_id"]
        .astype(str)
        .eq("np:overtakes")
    ]

    if overtakes.empty:
        return semantic_edges_all

    existing_pairs = set()

    if not semantic_edges_all.empty:
        for row in semantic_edges_all.itertuples(index=False):
            predicates = {
                str(predicate)
                for predicate in _excel_predicate_list(
                    getattr(row, "predicates", [])
                )
            }

            if "np:overtakes" not in predicates:
                continue

            existing_pairs.add((
                str(getattr(row, "subject_id", "")).strip(),
                str(getattr(row, "object_id", "")).strip(),
            ))

    recovered_rows = []

    for row in overtakes.itertuples(index=False):
        subject_id = str(
            getattr(row, "subject_id", "")
        ).strip()
        object_id = str(
            getattr(row, "object_id", "")
        ).strip()

        if not subject_id or not object_id:
            continue

        pair_key = (subject_id, object_id)

        if pair_key in existing_pairs:
            continue

        subject_token = str(
            entity_id_to_track_token(subject_id)
        ).strip()
        object_token = str(
            entity_id_to_track_token(object_id)
        ).strip()

        recovered_rows.append({
            "relation_type": classify_semantic_relation(
                subject_id,
                object_id,
            ),
            "category": "interaction",
            "subject_id": subject_id,
            "object_id": object_id,
            "subject_token": subject_token,
            "object_token": object_token,
            "predicates": ["np:overtakes"],
            "predicate_count": 1,
            "relation_labels": "overtakes",
            "semantic_source_token": subject_token,
            "semantic_target_token": object_token,
            "semantic_statement": (
                f"{short_token(subject_token)} → "
                f"{short_token(object_token)}: overtakes"
            ),
        })

        existing_pairs.add(pair_key)

    if not recovered_rows:
        return semantic_edges_all

    return pd.concat(
        [
            semantic_edges_all,
            pd.DataFrame(recovered_rows),
        ],
        ignore_index=True,
        sort=False,
    )


def _video_restore_merge_edges(semantic_edges_all, frame_assertions):
    """Restore entity-valued merge/crossing relations from current-frame assertions."""
    if semantic_edges_all is None:
        semantic_edges_all = pd.DataFrame()
    if frame_assertions is None or frame_assertions.empty:
        return semantic_edges_all
    if "predicate_id" not in frame_assertions.columns:
        return semantic_edges_all

    supported = {
        "np:mergesInFrontOf": "mergesInFrontOf",
        "np:mergesBehind": "mergesBehind",
        "np:crossesInFrontOf": "crossesInFrontOf",
        "np:yieldsTo": "yieldsTo",
    }
    rows = frame_assertions.loc[
        frame_assertions["predicate_id"].astype(str).isin(set(supported))
    ]
    if rows.empty:
        return semantic_edges_all

    existing = set()
    if not semantic_edges_all.empty:
        for row in semantic_edges_all.itertuples(index=False):
            predicates = {
                str(predicate)
                for predicate in _excel_predicate_list(getattr(row, "predicates", []))
            }
            for predicate in predicates.intersection(supported):
                existing.add((
                    str(getattr(row, "subject_id", "")).strip(),
                    str(getattr(row, "object_id", "")).strip(),
                    predicate,
                ))

    recovered = []
    for row in rows.itertuples(index=False):
        predicate = str(getattr(row, "predicate_id", ""))
        subject_id = str(getattr(row, "subject_id", "")).strip()
        object_id = str(getattr(row, "object_id", "")).strip()
        if not subject_id or not object_id or predicate not in supported:
            continue
        key = (subject_id, object_id, predicate)
        if key in existing:
            continue
        subject_token = str(entity_id_to_track_token(subject_id)).strip()
        object_token = str(entity_id_to_track_token(object_id)).strip()
        label = supported[predicate]
        recovered.append({
            "relation_type": classify_semantic_relation(subject_id, object_id),
            "category": "interaction",
            "subject_id": subject_id,
            "object_id": object_id,
            "subject_token": subject_token,
            "object_token": object_token,
            "predicates": [predicate],
            "predicate_count": 1,
            "relation_labels": label,
            "semantic_source_token": subject_token,
            "semantic_target_token": object_token,
            "semantic_statement": (
                f"{short_token(subject_token)} → {short_token(object_token)}: {label}"
            ),
        })
        existing.add(key)

    if not recovered:
        return semantic_edges_all
    return pd.concat(
        [semantic_edges_all, pd.DataFrame(recovered)],
        ignore_index=True,
        sort=False,
    )


def _video_restore_risk_edges(semantic_edges_all, frame_assertions):
    """Restore exact current-frame risk relations from assertion rows.

    Risk predicates are entity-valued agent/ego relations.  The generic
    semantic-edge conversion should normally retain them, but the video
    pipeline historically needed compatibility restoration for several
    entity-valued predicates.  Keep risk validation robust by reconstructing
    the stored SUBJECT -> OBJECT relation directly from the assertion row when
    it is absent.
    """
    if semantic_edges_all is None:
        semantic_edges_all = pd.DataFrame()
    if frame_assertions is None or frame_assertions.empty:
        return semantic_edges_all
    if "predicate_id" not in frame_assertions.columns:
        return semantic_edges_all

    predicate = "np:hasConflictRiskWith"
    rows = frame_assertions.loc[
        frame_assertions["predicate_id"].astype(str).eq(predicate)
    ]
    if rows.empty:
        return semantic_edges_all

    existing = set()
    if not semantic_edges_all.empty:
        for edge in semantic_edges_all.itertuples(index=False):
            predicates = {
                str(value)
                for value in _excel_predicate_list(
                    getattr(edge, "predicates", [])
                )
            }
            if predicate not in predicates:
                continue
            existing.add((
                str(getattr(edge, "subject_id", "")).strip(),
                str(getattr(edge, "object_id", "")).strip(),
            ))

    recovered = []
    for row in rows.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id", "") or "").strip()
        object_id = str(getattr(row, "object_id", "") or "").strip()
        if not subject_id or not object_id:
            continue
        key = (subject_id, object_id)
        if key in existing:
            continue

        subject_token = str(entity_id_to_track_token(subject_id)).strip()
        object_token = str(entity_id_to_track_token(object_id)).strip()
        recovered.append({
            "relation_type": classify_semantic_relation(subject_id, object_id),
            "category": "risk",
            "subject_id": subject_id,
            "object_id": object_id,
            "subject_token": subject_token,
            "object_token": object_token,
            "predicates": [predicate],
            "predicate_count": 1,
            "relation_labels": "hasConflictRiskWith",
            "semantic_source_token": subject_token,
            "semantic_target_token": object_token,
            "semantic_statement": (
                f"{short_token(subject_token)} → "
                f"{short_token(object_token)}: hasConflictRiskWith"
            ),
        })
        existing.add(key)

    if not recovered:
        return semantic_edges_all

    return pd.concat(
        [semantic_edges_all, pd.DataFrame(recovered)],
        ignore_index=True,
        sort=False,
    )


def _video_active_lane_change_overlays(
    frame_assertions,
    semantic_relations,
    entity_table=None,
):
    """Return active np:changesLane vehicle-to-structure evidence rows."""
    result = []

    if frame_assertions is None or frame_assertions.empty:
        return result
    if "predicate_id" not in frame_assertions.columns:
        return result

    rows = frame_assertions.loc[
        frame_assertions["predicate_id"]
        .astype(str)
        .eq("np:changesLane")
    ]

    enabled_subject_tokens = None
    if entity_table is not None and not entity_table.empty:
        enabled_subject_tokens = set(
            entity_table["track_token"].dropna().astype(str)
        )

    seen_event_ids = set()
    for row in rows.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id", "")).strip()
        object_id = str(getattr(row, "object_id", "")).strip()
        if not subject_id or not object_id:
            continue

        relation_type = str(
            classify_semantic_relation(subject_id, object_id)
        )
        if not bool(semantic_relations.get(relation_type, False)):
            continue

        subject_token = str(
            entity_id_to_track_token(subject_id)
        )
        if (
            enabled_subject_tokens is not None
            and subject_token not in enabled_subject_tokens
        ):
            continue

        evidence = _evidence_mapping(
            getattr(row, "evidence_json", None)
        )
        event_id = str(
            evidence.get("event_id")
            or f"{subject_id}|{object_id}"
        )
        if event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)

        result.append({
            "event_id": event_id,
            "subject_id": subject_id,
            "subject_token": subject_token,
            "object_id": object_id,
            "relation_type": relation_type,
            "evidence": evidence,
        })

    return result


def _video_draw_lane_change_overlays(
    axis,
    frame,
    frame_assertions,
    semantic_relations,
    entity_table=None,
):
    """Draw only the active subject-to-completion-point lane-change arrow."""
    del frame  # Kept in the public signature for notebook compatibility.

    overlays = _video_active_lane_change_overlays(
        frame_assertions,
        semantic_relations,
        entity_table=entity_table,
    )

    for overlay in overlays:
        evidence = overlay["evidence"]
        coordinate_names = (
            "lane_change_arrow_start_x",
            "lane_change_arrow_start_y",
            "lane_change_arrow_end_x",
            "lane_change_arrow_end_y",
        )
        coordinates = []
        for name in coordinate_names:
            try:
                value = float(evidence.get(name))
            except (TypeError, ValueError):
                value = float("nan")
            coordinates.append(value)

        if not all(math.isfinite(value) for value in coordinates):
            continue

        start_x, start_y, end_x, end_y = coordinates
        if math.hypot(end_x - start_x, end_y - start_y) <= 1e-6:
            continue

        axis.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#6a1b9a",
                "linewidth": 4.0,
                "mutation_scale": 18,
                "shrinkA": 0,
                "shrinkB": 0,
                "alpha": 0.95,
            },
            zorder=18,
        )

    return overlays


def _video_safe_filename_fragment(values, fallback):
    """Create a short filename fragment from enabled configuration names."""
    cleaned = []

    for value in values:
        text = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in str(value)
        ).strip("_")

        if text:
            cleaned.append(text)

    if not cleaned:
        return fallback

    joined = "-".join(cleaned)
    return joined[:120]


# ============================================================
# EGO ENDPOINT NORMALIZATION FOR VIDEO EDGES
# ============================================================
#
# The assertion/RDF layer and the frame entity table may represent EGO with
# different identifiers. Normal agents usually retain their track token, while
# EGO can appear as "ego", "EGO", an RDF URI ending in /ego or #ego, or an
# EgoVehicle URI. Normalize edge endpoints before relation-direction filtering
# and before prepare_plot_edges().
# ============================================================


def _video_is_ego_identifier(value):
    """Return True for the common EGO identifiers used by the pipeline."""
    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    lowered = text.lower().rstrip("/")

    if lowered in {
        "ego",
        "np:ego",
        "ego_vehicle",
        "egovehicle",
        "np:egovehicle",
    }:
        return True

    return (
        "egovehicle" in lowered
        or lowered.endswith("/ego")
        or lowered.endswith("#ego")
        or lowered.endswith(":ego")
    )


def _video_canonical_ego_token(entity_table):
    """Return the exact EGO token used by the current frame entity table."""
    if entity_table is None or entity_table.empty:
        return "ego"

    if "is_ego" not in entity_table.columns:
        return "ego"

    ego_rows = entity_table.loc[
        entity_table["is_ego"].fillna(False).astype(bool)
    ]
    if ego_rows.empty:
        return "ego"

    for column in ("track_token", "entity_token", "token"):
        if column in ego_rows.columns:
            value = ego_rows.iloc[0][column]
            if value is not None and str(value).strip():
                return str(value)

    return "ego"


def _video_normalize_endpoint(value, canonical_ego_token):
    """Normalize one semantic-edge endpoint to a frame entity token."""
    if _video_is_ego_identifier(value):
        return str(canonical_ego_token)

    try:
        converted = entity_id_to_track_token(value)
    except Exception:
        converted = value

    if _video_is_ego_identifier(converted):
        return str(canonical_ego_token)

    return str(converted)


def _video_normalize_edge_endpoints(edge_table, entity_table):
    """
    Normalize EGO/agent endpoint identifiers and recompute relation direction.

    This must run before _filter_master_semantic_edges(), because that function
    uses relation_type to apply ego_to_agent, agent_to_ego, and agent_to_agent
    switches.
    """
    if edge_table is None:
        return pd.DataFrame()
    if edge_table.empty:
        return edge_table.copy()

    result = edge_table.copy()
    canonical_ego_token = _video_canonical_ego_token(entity_table)

    subject_columns = [
        column
        for column in ("subject_token", "subject", "subject_id")
        if column in result.columns
    ]
    object_columns = [
        column
        for column in ("object_token", "object", "object_id")
        if column in result.columns
    ]

    for column in subject_columns:
        result[column] = result[column].map(
            lambda value: _video_normalize_endpoint(
                value,
                canonical_ego_token,
            )
        )

    for column in object_columns:
        result[column] = result[column].map(
            lambda value: _video_normalize_endpoint(
                value,
                canonical_ego_token,
            )
        )

    subject_column = next(
        (column for column in ("subject_token", "subject", "subject_id")
         if column in result.columns),
        None,
    )
    object_column = next(
        (column for column in ("object_token", "object", "object_id")
         if column in result.columns),
        None,
    )

    if subject_column is not None and object_column is not None:
        subject_is_ego = result[subject_column].astype(str).eq(
            str(canonical_ego_token)
        )
        object_is_ego = result[object_column].astype(str).eq(
            str(canonical_ego_token)
        )

        if "relation_type" not in result.columns:
            result["relation_type"] = "other"

        entity_tokens = set()
        if "track_token" in entity_table.columns:
            entity_tokens = set(
                entity_table["track_token"].dropna().astype(str)
            )

        subject_is_entity = result[subject_column].astype(str).isin(
            entity_tokens
        )
        object_is_entity = result[object_column].astype(str).isin(
            entity_tokens
        )

        result.loc[
            subject_is_ego & object_is_entity & ~object_is_ego,
            "relation_type",
        ] = "ego_to_agent"
        result.loc[
            ~subject_is_ego & subject_is_entity & object_is_ego,
            "relation_type",
        ] = "agent_to_ego"
        result.loc[
            ~subject_is_ego
            & ~object_is_ego
            & subject_is_entity
            & object_is_entity,
            "relation_type",
        ] = "agent_to_agent"

    return result


def _video_count_ego_edges(edge_table, entity_table):
    """Count rows whose subject or object is EGO."""
    if edge_table is None or edge_table.empty:
        return 0

    canonical_ego_token = _video_canonical_ego_token(entity_table)
    subject_column = next(
        (column for column in ("subject_token", "subject", "subject_id")
         if column in edge_table.columns),
        None,
    )
    object_column = next(
        (column for column in ("object_token", "object", "object_id")
         if column in edge_table.columns),
        None,
    )

    if subject_column is None or object_column is None:
        return 0

    return int(
        (
            edge_table[subject_column].astype(str).eq(canonical_ego_token)
            | edge_table[object_column].astype(str).eq(canonical_ego_token)
        ).sum()
    )


def _video_count_raw_ego_follows(frame_assertions):
    """Count raw np:follows assertions involving EGO before visualization."""
    if frame_assertions is None or frame_assertions.empty:
        return 0

    follows = frame_assertions.loc[
        frame_assertions["predicate_id"].astype(str).eq("np:follows")
    ]
    if follows.empty:
        return 0

    return int(
        (
            follows["subject_id"].map(_video_is_ego_identifier)
            | follows["object_id"].map(_video_is_ego_identifier)
        ).sum()
    )



def _video_convert_mp4v_to_browser_h264(
    temporary_video_path,
    final_video_path,
):
    """Convert OpenCV MP4V output to browser-compatible H.264/yuv420p."""
    temporary_video_path = Path(temporary_video_path)
    final_video_path = Path(final_video_path)

    ffmpeg_executable = _video_shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        raise RuntimeError(
            "FFmpeg is required for browser-compatible H.264 output. "
            "Install it in the active environment, for example with: "
            "conda install -c conda-forge ffmpeg -y"
        )

    if (
        not temporary_video_path.exists()
        or temporary_video_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "OpenCV did not create a valid temporary MP4V video: "
            f"{temporary_video_path}"
        )

    final_video_path.parent.mkdir(parents=True, exist_ok=True)
    final_video_path.unlink(missing_ok=True)

    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temporary_video_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-an",
        str(final_video_path),
    ]

    completed = _video_subprocess.run(
        command,
        stdout=_video_subprocess.PIPE,
        stderr=_video_subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg H.264 conversion failed. The temporary MP4V file "
            f"was kept at {temporary_video_path}.\n"
            f"FFmpeg error:\n{completed.stderr.strip()}"
        )

    if (
        not final_video_path.exists()
        or final_video_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "FFmpeg completed without creating a valid H.264 file: "
            f"{final_video_path}"
        )

    temporary_video_path.unlink(missing_ok=True)
    return final_video_path



def _video_draw_crosses_reference_lines(
    axis,
    plot_edges,
    entity_table,
    *,
    map_radius_m,
):
    """Draw the exact paper 10 m forward segments for active crossings.

    Each segment starts at the current agent position and extends exactly
    5 m in the same current heading direction used by the detector.
    """
    if (
        plot_edges is None
        or plot_edges.empty
        or entity_table is None
        or entity_table.empty
    ):
        return 0

    entity_by_token = entity_table.set_index("track_token", drop=False)
    drawn_pairs = set()
    line_length_m = float(VIDEO_CROSSES_REFERENCE_RAY_LENGTH_M)
    drawn_count = 0

    def _row_for_token(token):
        if token not in entity_by_token.index:
            return None
        row = entity_by_token.loc[token]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _draw_line(row, *, color, linestyle, label_text, zorder):
        try:
            x = float(row["x"])
            y = float(row["y"])
            heading = float(row["heading"])
        except (TypeError, ValueError, KeyError):
            return False

        if not all(math.isfinite(value) for value in (x, y, heading)):
            return False

        dx = line_length_m * math.cos(heading)
        dy = line_length_m * math.sin(heading)
        x0, y0 = x, y
        x1, y1 = x + dx, y + dy

        axis.plot(
            [x0, x1],
            [y0, y1],
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            alpha=0.90,
            zorder=zorder,
        )

        # Put an inline label close to the agent rather than adding a legend.
        label_distance_m = min(3.5, 0.70 * line_length_m)
        label_x = x + label_distance_m * math.cos(heading)
        label_y = y + label_distance_m * math.sin(heading)
        axis.text(
            label_x,
            label_y,
            label_text,
            fontsize=7.0,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
            zorder=zorder + 0.2,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.82,
                "linewidth": 0.5,
            },
        )
        return True

    for edge in plot_edges.itertuples(index=False):
        edge_predicates = {
            str(predicate)
            for predicate in _excel_predicate_list(
                getattr(edge, "predicates", [])
            )
        }
        if "np:crossesInFrontOf" not in edge_predicates:
            continue

        subject_token = str(getattr(edge, "subject_token", ""))
        object_token = str(getattr(edge, "object_token", ""))
        if not subject_token or not object_token:
            continue

        pair_key = (subject_token, object_token)
        if pair_key in drawn_pairs:
            continue
        drawn_pairs.add(pair_key)

        subject_row = _row_for_token(subject_token)
        object_row = _row_for_token(object_token)
        if subject_row is None or object_row is None:
            continue

        subject_drawn = _draw_line(
            subject_row,
            color="#8e24aa",
            linestyle="-.",
            label_text="S forward",
            zorder=8.15,
        )
        object_drawn = _draw_line(
            object_row,
            color="#fb8c00",
            linestyle="--",
            label_text="O forward",
            zorder=8.20,
        )

        if subject_drawn or object_drawn:
            drawn_count += 1

    return drawn_count


# ============================================================
# v9.5.38 TRAFFIC-LIGHT VIDEO OVERLAY
# ============================================================

_VIDEO_TL_SEARCH_RADIUS_M = 20.0
_VIDEO_TL_RELEVANCE_COLOR = "#d81b60"


def _video_tl_signal_connector_id(signal_id):
    text = str(signal_id or "").strip()
    marker = ":control:"
    if text.startswith("traffic_signal:") and marker in text:
        connector_id = text.rsplit(marker, 1)[-1].strip()
        return connector_id or None
    return None


def _video_tl_movement_connector_id(movement_id):
    text = str(movement_id or "").strip()
    marker = ":lane_connector:"
    if text.startswith("controlled_movement:") and marker in text:
        connector_id = text.rsplit(marker, 1)[-1].strip()
        return connector_id or None
    return None


def _video_tl_state_name(state_id):
    text = str(state_id or "").strip()
    prefix = "signal_state:"
    if text.startswith(prefix):
        value = text[len(prefix):].strip().upper()
        if value in {"RED", "YELLOW", "GREEN", "UNKNOWN"}:
            return value
    return "UNKNOWN"


def _video_tl_state_color(state):
    return {
        "RED": "#d32f2f",
        "YELLOW": "#f9a825",
        "GREEN": "#2e7d32",
        "UNKNOWN": "#616161",
    }.get(str(state or "UNKNOWN").upper(), "#616161")


def _video_tl_get_map_object(map_api, object_id, layer):
    if map_api is None:
        return None
    for method_name in ("get_map_object", "get_one_map_object"):
        method = getattr(map_api, method_name, None)
        if method is None:
            continue
        try:
            obj = method(str(object_id), layer)
        except Exception:
            continue
        if obj is not None:
            return obj
    return None


def _video_tl_get_connector(map_api, connector_id):
    if not NUPLAN_MAP_IMPORTS_AVAILABLE:
        return None
    return _video_tl_get_map_object(
        map_api,
        connector_id,
        SemanticMapLayer.LANE_CONNECTOR,
    )


def _video_tl_baseline_points(connector):
    baseline = getattr(connector, "baseline_path", None)
    path = getattr(baseline, "discrete_path", None)
    if path is None:
        return np.empty((0, 2), dtype=float)

    points = []
    for pose in list(path):
        try:
            points.append((float(pose.x), float(pose.y)))
        except Exception:
            continue
    return np.asarray(points, dtype=float)


def _video_tl_connector_polygon(connector):
    polygon = getattr(connector, "polygon", None)
    exterior = getattr(polygon, "exterior", None)
    if exterior is None:
        return np.empty((0, 2), dtype=float)
    try:
        return np.asarray(exterior.coords, dtype=float)
    except Exception:
        return np.empty((0, 2), dtype=float)


def _video_tl_connector_entry(connector):
    points = _video_tl_baseline_points(connector)
    if len(points):
        return np.asarray(points[0], dtype=float)

    polygon = getattr(connector, "polygon", None)
    centroid = getattr(polygon, "centroid", None)
    if centroid is not None:
        try:
            return np.array(
                [float(centroid.x), float(centroid.y)],
                dtype=float,
            )
        except Exception:
            pass
    return None


def _video_tl_connector_center(connector):
    points = _video_tl_baseline_points(connector)
    if len(points):
        return np.mean(points, axis=0)

    polygon = getattr(connector, "polygon", None)
    centroid = getattr(polygon, "centroid", None)
    if centroid is not None:
        try:
            return np.array(
                [float(centroid.x), float(centroid.y)],
                dtype=float,
            )
        except Exception:
            pass
    return None


def _video_tl_geometry(obj):
    for name in ("polygon", "linestring", "line", "geometry"):
        geometry = getattr(obj, name, None)
        if geometry is not None:
            return geometry
    return None


def _video_tl_anchor(obj):
    if obj is None:
        return None

    geometry = _video_tl_geometry(obj)
    if geometry is not None:
        for attr_name in ("representative_point", "centroid"):
            try:
                attr = getattr(geometry, attr_name)
                point = attr() if callable(attr) else attr
                return np.array(
                    [float(point.x), float(point.y)],
                    dtype=float,
                )
            except Exception:
                continue

    for candidate in (
        getattr(obj, "point", None),
        getattr(obj, "center", None),
    ):
        if candidate is None:
            continue
        try:
            return np.array(
                [float(candidate.x), float(candidate.y)],
                dtype=float,
            )
        except Exception:
            continue

    baseline = getattr(obj, "baseline_path", None)
    path = getattr(baseline, "discrete_path", None)
    if path:
        for pose in list(path):
            try:
                return np.array(
                    [float(pose.x), float(pose.y)],
                    dtype=float,
                )
            except Exception:
                continue

    return None


def _video_tl_proximal_objects(map_api, center_xy, radius_m, layer):
    if (
        map_api is None
        or center_xy is None
        or not NUPLAN_MAP_IMPORTS_AVAILABLE
    ):
        return []

    try:
        result = map_api.get_proximal_map_objects(
            Point2D(
                float(center_xy[0]),
                float(center_xy[1]),
            ),
            float(radius_m),
            [layer],
        )
        return list(result.get(layer, []))
    except Exception:
        return []


def _video_tl_select_physical_signal(map_api, connector):
    entry = _video_tl_connector_entry(connector)
    if entry is None or not hasattr(SemanticMapLayer, "TRAFFIC_LIGHT"):
        return None

    candidates = _video_tl_proximal_objects(
        map_api,
        entry,
        _VIDEO_TL_SEARCH_RADIUS_M,
        SemanticMapLayer.TRAFFIC_LIGHT,
    )

    scored = []
    for obj in candidates:
        anchor = _video_tl_anchor(obj)
        if anchor is None:
            continue
        scored.append((
            float(np.linalg.norm(anchor - entry)),
            obj,
        ))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _video_tl_draw_map_object(
    ax,
    obj,
    *,
    facecolor,
    edgecolor,
    alpha=0.55,
    linewidth=2.5,
    zorder=12,
):
    geometry = _video_tl_geometry(obj)

    if geometry is None:
        anchor = _video_tl_anchor(obj)
        if anchor is not None:
            ax.scatter(
                [float(anchor[0])],
                [float(anchor[1])],
                s=150,
                marker="s",
                c=[facecolor],
                edgecolors=edgecolor,
                linewidths=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        return

    geom_type = str(getattr(geometry, "geom_type", ""))

    if geom_type in {"Polygon", "MultiPolygon"}:
        try:
            parts = list(_iter_polygon_parts(geometry))
        except Exception:
            parts = [geometry]

        for polygon in parts:
            try:
                coords = np.asarray(
                    polygon.exterior.coords,
                    dtype=float,
                )
            except Exception:
                continue
            ax.add_patch(
                MplPolygon(
                    coords,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=zorder,
                )
            )
        return

    if geom_type in {"LineString", "LinearRing"}:
        try:
            coords = np.asarray(geometry.coords, dtype=float)
        except Exception:
            coords = np.empty((0, 2), dtype=float)
        if len(coords):
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
        return

    anchor = _video_tl_anchor(obj)
    if anchor is not None:
        ax.scatter(
            [float(anchor[0])],
            [float(anchor[1])],
            s=150,
            marker="s",
            c=[facecolor],
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
        )


def _video_tl_canonical_state_by_signal(frame_assertions):
    result = {}

    if (
        frame_assertions is None
        or frame_assertions.empty
        or "predicate_id" not in frame_assertions.columns
    ):
        return result

    rows = frame_assertions.loc[
        frame_assertions["predicate_id"]
        .astype(str)
        .eq("np:hasSignalState")
    ]

    for row in rows.itertuples(index=False):
        signal_id = str(
            getattr(row, "subject_id", "")
        ).strip()
        state_id = str(
            getattr(row, "object_id", "")
        ).strip()
        if signal_id:
            result[signal_id] = _video_tl_state_name(state_id)

    return result


def _video_tl_native_state_by_connector(frame):
    result = {}
    for light in getattr(frame, "traffic_lights", []) or []:
        connector_id = str(
            getattr(light, "lane_connector_id", "")
        ).strip()
        if not connector_id:
            continue

        raw = getattr(light, "status", "UNKNOWN")
        state = str(getattr(raw, "name", raw)).upper()
        if "." in state:
            state = state.rsplit(".", 1)[-1]
        if state == "AMBER":
            state = "YELLOW"
        if state not in {"RED", "YELLOW", "GREEN"}:
            state = "UNKNOWN"
        result[connector_id] = state
    return result


def _video_tl_selected(predicate_id, semantic_predicates):
    return semantic_predicate_is_selected(
        predicate_id,
        semantic_predicates,
    )


def _video_tl_relation_enabled(relation_type, semantic_relations):
    return bool(
        semantic_relations.get(
            str(relation_type),
            False,
        )
    )



_VIDEO_TL_FRAME_MATCH_TOLERANCE_US = 300_000


def _video_tl_nearest_assertion_snapshot(
    scenario_assertions,
    timestamp_us,
    tolerance_us=_VIDEO_TL_FRAME_MATCH_TOLERANCE_US,
):
    """
    Return the nearest traffic-light assertion snapshot to one nuPlan frame.

    Traffic-light assertion timestamps are not guaranteed to be bit-identical
    to frame.timestamp_us.  The generic video path historically required exact
    equality, which caused valid traffic-light predicates to disappear from
    batch videos and video_edge_events.csv.

    The nearest traffic-light assertion timestamp is accepted only when it is
    within tolerance_us of the displayed frame.
    """
    if (
        scenario_assertions is None
        or scenario_assertions.empty
        or "predicate_id" not in scenario_assertions.columns
        or "valid_time_us" not in scenario_assertions.columns
    ):
        return scenario_assertions.iloc[0:0].copy() if scenario_assertions is not None else None

    traffic_ids = {
        "np:controls",
        "np:hasSignalState",
        "np:isRelevantSignal",
    }

    rows = scenario_assertions.loc[
        scenario_assertions["predicate_id"].astype(str).isin(traffic_ids)
    ].copy()

    if rows.empty:
        return rows

    times = pd.to_numeric(rows["valid_time_us"], errors="coerce")
    good = times.notna()
    if not good.any():
        return rows.iloc[0:0].copy()

    rows = rows.loc[good].copy()
    times = times.loc[good].astype("int64")

    target = int(timestamp_us)
    deltas = (times - target).abs()
    nearest_index = deltas.idxmin()
    nearest_time = int(times.loc[nearest_index])
    nearest_delta = abs(nearest_time - target)

    if nearest_delta > int(tolerance_us):
        return rows.iloc[0:0].copy()

    snapshot = rows.loc[times.eq(nearest_time)].copy()
    snapshot["_video_frame_timestamp_us"] = target
    snapshot["_video_assertion_timestamp_us"] = nearest_time
    snapshot["_video_time_delta_us"] = nearest_delta
    return snapshot


def _video_tl_build_overlay_rows(
    frame,
    frame_assertions,
    entity_table,
    *,
    semantic_categories,
    semantic_relations,
    semantic_predicates,
    scenario_assertions=None,
    timestamp_us=None,
    frame_match_tolerance_us=_VIDEO_TL_FRAME_MATCH_TOLERANCE_US,
):
    """
    Build selected traffic-light events directly from the assertion table.

    This intentionally bypasses semantic_edge_table(), whose generic
    entity-token filtering is designed for tracked actors and historically
    removed canonical traffic_signal / controlled_movement resources.
    """
    if not bool(
        semantic_categories.get("traffic_light", False)
    ):
        return []

    selected_predicates = {
        predicate_id
        for predicate_id in (
            "np:controls",
            "np:hasSignalState",
            "np:isRelevantSignal",
        )
        if _video_tl_selected(
            predicate_id,
            semantic_predicates,
        )
    }

    if not selected_predicates:
        return []

    # Traffic-light assertions need tolerant timestamp alignment.  Prefer a
    # nearest complete traffic-light snapshot from the whole scenario; fall
    # back to the generic exact-frame assertions only when necessary.
    tl_assertions = None

    if (
        scenario_assertions is not None
        and timestamp_us is not None
    ):
        tl_assertions = _video_tl_nearest_assertion_snapshot(
            scenario_assertions,
            timestamp_us,
            tolerance_us=frame_match_tolerance_us,
        )

    if (
        tl_assertions is None
        or tl_assertions.empty
    ):
        if (
            frame_assertions is None
            or frame_assertions.empty
            or "predicate_id" not in frame_assertions.columns
        ):
            return []
        tl_assertions = frame_assertions

    rows = tl_assertions.loc[
        tl_assertions["predicate_id"]
        .astype(str)
        .isin(selected_predicates)
    ]

    if rows.empty:
        return []

    display_lookup = {}
    if (
        entity_table is not None
        and not entity_table.empty
        and "track_token" in entity_table.columns
    ):
        display_lookup = (
            entity_table
            .drop_duplicates("track_token")
            .set_index("track_token")["display_id"]
            .astype(str)
            .to_dict()
        )

    canonical_states = _video_tl_canonical_state_by_signal(
        tl_assertions
    )
    native_states = _video_tl_native_state_by_connector(frame)

    events = []
    seen = set()

    for assertion in rows.itertuples(index=False):
        predicate_id = str(
            getattr(assertion, "predicate_id", "")
        )
        subject_id = str(
            getattr(assertion, "subject_id", "")
        ).strip()
        object_id = str(
            getattr(assertion, "object_id", "")
        ).strip()

        connector_id = None
        relation_type = "structure_to_structure"
        subject_track_token = None

        if predicate_id == "np:isRelevantSignal":
            connector_id = _video_tl_signal_connector_id(
                object_id
            )
            subject_track_token = str(
                entity_id_to_track_token(subject_id)
            )
            relation_type = (
                "ego_to_structure"
                if subject_track_token == "ego"
                else "agent_to_structure"
            )
            signal_id = object_id

        elif predicate_id == "np:controls":
            connector_id = (
                _video_tl_movement_connector_id(object_id)
                or _video_tl_signal_connector_id(subject_id)
            )
            signal_id = subject_id

        elif predicate_id == "np:hasSignalState":
            connector_id = _video_tl_signal_connector_id(
                subject_id
            )
            signal_id = subject_id

        else:
            continue

        if not connector_id:
            continue

        # In exact-predicate video mode, the selected traffic-light predicate
        # is authoritative. Generic relation switches were designed for the
        # agent-only graph renderer and historically suppressed
        # structure_to_structure traffic-light relations such as controls and
        # hasSignalState. Respect relation switches only when displaying all
        # predicates together.
        _exact_tl_selection = normalize_semantic_predicate_selection(
            semantic_predicates
        )
        if (
            _exact_tl_selection is None
            and not _video_tl_relation_enabled(
                relation_type,
                semantic_relations,
            )
        ):
            continue

        state = canonical_states.get(
            signal_id,
            native_states.get(
                connector_id,
                "UNKNOWN",
            ),
        )

        if predicate_id == "np:isRelevantSignal":
            subject_display = display_lookup.get(
                subject_track_token,
                short_token(subject_track_token),
            )
            object_display = f"TrafficSignal:{connector_id}"
        elif predicate_id == "np:controls":
            subject_display = f"TrafficSignal:{connector_id}"
            object_display = f"ControlledMovement:{connector_id}"
        else:
            subject_display = f"TrafficSignal:{connector_id}"
            object_display = state

        key = (
            predicate_id,
            subject_id,
            object_id,
            connector_id,
        )
        if key in seen:
            continue
        seen.add(key)

        events.append({
            "predicate_id": predicate_id,
            "relation_labels": predicate_id.replace("np:", ""),
            "category": "traffic_light",
            "relation_type": relation_type,
            "subject_id": subject_id,
            "object_id": object_id,
            "subject": subject_display,
            "object": object_display,
            "subject_track_token": subject_track_token,
            "signal_id": signal_id,
            "connector_id": connector_id,
            "state": state,
            "assertion_timestamp_us": (
                int(getattr(assertion, "_video_assertion_timestamp_us"))
                if hasattr(assertion, "_video_assertion_timestamp_us")
                else int(getattr(assertion, "valid_time_us", timestamp_us or 0))
            ),
            "frame_time_delta_us": (
                int(getattr(assertion, "_video_time_delta_us"))
                if hasattr(assertion, "_video_time_delta_us")
                else 0
            ),
        })

    return events


def _video_tl_draw_overlays(
    ax,
    frame,
    entity_table,
    events,
    *,
    draw_semantic_arrows_on_map=True,
):
    """
    Draw the traffic-light family on the ordinary ego-centered semantic map.

      Agent --isRelevantSignal--> TrafficSignal
      TrafficSignal --controls--> ControlledMovement
      TrafficSignal [RED/GREEN/YELLOW/UNKNOWN]
    """
    if not events:
        return {
            "event_count": 0,
            "drawn_is_relevant_signal": 0,
            "drawn_controls": 0,
            "drawn_states": 0,
        }

    map_api = getattr(frame, "map_api", None)

    entity_lookup = {}
    if (
        entity_table is not None
        and not entity_table.empty
        and "track_token" in entity_table.columns
    ):
        entity_lookup = (
            entity_table
            .drop_duplicates("track_token")
            .set_index("track_token", drop=False)
        )

    connector_ids = sorted({
        str(event["connector_id"])
        for event in events
        if event.get("connector_id")
    })

    connector_cache = {}
    signal_anchor_cache = {}
    state_by_connector = {}

    for event in events:
        connector_id = str(event["connector_id"])
        state_by_connector.setdefault(
            connector_id,
            str(event.get("state") or "UNKNOWN"),
        )

    # Draw each controlled connector / physical signal once as context.
    for connector_id in connector_ids:
        connector = _video_tl_get_connector(
            map_api,
            connector_id,
        )
        if connector is None:
            continue

        connector_cache[connector_id] = connector
        state = state_by_connector.get(
            connector_id,
            "UNKNOWN",
        )
        state_color = _video_tl_state_color(state)

        polygon = _video_tl_connector_polygon(connector)
        baseline = _video_tl_baseline_points(connector)
        center = _video_tl_connector_center(connector)
        entry = _video_tl_connector_entry(connector)

        if len(polygon):
            ax.add_patch(
                MplPolygon(
                    polygon,
                    closed=True,
                    facecolor=state_color,
                    edgecolor="#006064",
                    alpha=0.16,
                    linewidth=2.2,
                    zorder=6.0,
                )
            )

        if len(baseline):
            ax.plot(
                baseline[:, 0],
                baseline[:, 1],
                color="#006064",
                linewidth=3.0,
                alpha=0.90,
                zorder=7.0,
            )

        physical_signal = _video_tl_select_physical_signal(
            map_api,
            connector,
        )

        if physical_signal is not None:
            _video_tl_draw_map_object(
                ax,
                physical_signal,
                facecolor=state_color,
                edgecolor="#212121",
                alpha=0.72,
                linewidth=2.8,
                zorder=13,
            )
            signal_anchor = _video_tl_anchor(
                physical_signal
            )
        else:
            signal_anchor = entry
            if signal_anchor is not None:
                ax.scatter(
                    [float(signal_anchor[0])],
                    [float(signal_anchor[1])],
                    s=180,
                    marker="s",
                    c=[state_color],
                    edgecolors="#212121",
                    linewidths=2.0,
                    zorder=13,
                )

        if signal_anchor is not None:
            signal_anchor_cache[connector_id] = signal_anchor
            ax.text(
                float(signal_anchor[0]),
                float(signal_anchor[1]) + 1.7,
                f"TrafficSignal\n{state}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                zorder=16,
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": state_color,
                    "alpha": 0.92,
                },
            )

        if center is not None:
            ax.text(
                float(center[0]),
                float(center[1]),
                f"LC:{connector_id}",
                ha="center",
                va="center",
                fontsize=6.5,
                zorder=15,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#006064",
                    "alpha": 0.82,
                },
            )

    diagnostics = {
        "event_count": int(len(events)),
        "drawn_is_relevant_signal": 0,
        "drawn_controls": 0,
        "drawn_states": 0,
    }

    for event in events:
        predicate_id = str(event["predicate_id"])
        connector_id = str(event["connector_id"])
        connector = connector_cache.get(connector_id)
        signal_anchor = signal_anchor_cache.get(
            connector_id
        )

        if predicate_id == "np:hasSignalState":
            # The state is a semantic node without its own map coordinate.
            # Its value is therefore represented by the label on the signal.
            if signal_anchor is not None:
                diagnostics["drawn_states"] += 1
            continue

        if not bool(draw_semantic_arrows_on_map):
            continue

        if (
            predicate_id == "np:controls"
            and connector is not None
            and signal_anchor is not None
        ):
            target = _video_tl_connector_center(
                connector
            )
            if target is not None:
                ax.annotate(
                    "controls",
                    xy=(
                        float(target[0]),
                        float(target[1]),
                    ),
                    xytext=(
                        float(signal_anchor[0]),
                        float(signal_anchor[1]),
                    ),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#c62828",
                        "linewidth": 2.6,
                        "shrinkA": 8,
                        "shrinkB": 7,
                    },
                    color="#c62828",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=17,
                )
                diagnostics["drawn_controls"] += 1
            continue

        if (
            predicate_id == "np:isRelevantSignal"
            and signal_anchor is not None
        ):
            token = str(
                event.get("subject_track_token") or ""
            )
            if token not in entity_lookup.index:
                continue

            source = entity_lookup.loc[token]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]

            source_x = float(source["x"])
            source_y = float(source["y"])

            ax.annotate(
                "isRelevantSignal",
                xy=(
                    float(signal_anchor[0]),
                    float(signal_anchor[1]),
                ),
                xytext=(source_x, source_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": _VIDEO_TL_RELEVANCE_COLOR,
                    "linewidth": 3.3,
                    "alpha": 0.96,
                    "shrinkA": 9,
                    "shrinkB": 9,
                },
                color=_VIDEO_TL_RELEVANCE_COLOR,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=18,
            )

            # Strong outline so the participating road user is obvious.
            try:
                corners = oriented_rectangle_corners(
                    source_x,
                    source_y,
                    float(source["heading"]),
                    source["length"],
                    source["width"],
                )
                ax.add_patch(
                    MplPolygon(
                        corners,
                        closed=True,
                        facecolor="none",
                        edgecolor=_VIDEO_TL_RELEVANCE_COLOR,
                        linewidth=3.0,
                        zorder=17,
                    )
                )
            except Exception:
                pass

            diagnostics["drawn_is_relevant_signal"] += 1

    return diagnostics



# ============================================================
# DIRECT EXACT-PREDICATE VISUALIZATION FALLBACK (v9.5.38 fix2)
# ============================================================
#
# The generic semantic-edge renderer is intentionally graph-oriented. It
# therefore drops unary literal predicates (object_id is null) and historically
# discards agent->map-structure edges because map structure IDs are not tracked
# agent tokens. Exact-predicate validation needs both classes to remain visible
# and to be written to video_edge_events.csv.
#
# This fallback is active only when one or more exact predicates are selected.
# Pairwise predicates already represented by the normal semantic edge table are
# left untouched.
# ============================================================

_VIDEO_DIRECT_MAP_LAYER_NAMES = {
    "lane": "LANE",
    "lane_connector": "LANE_CONNECTOR",
    "roadblock": "ROADBLOCK",
    "roadblock_connector": "ROADBLOCK_CONNECTOR",
    "intersection": "INTERSECTION",
    "crosswalk": "CROSSWALK",
    "stop_line": "STOP_LINE",
    "traffic_light": "TRAFFIC_LIGHT",
    "walkway": "WALKWAYS",
    "carpark": "CARPARK_AREA",
}


def _video_direct_definition_meta(predicate_id):
    if (
        DEFINITIONS is None
        or DEFINITIONS.empty
        or "predicate_id" not in DEFINITIONS.columns
    ):
        return {}
    rows = DEFINITIONS.loc[
        DEFINITIONS["predicate_id"].astype(str).eq(str(predicate_id))
    ]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    result = {}
    for key in ("category", "label", "units", "value_type"):
        if key in rows.columns:
            value = row.get(key)
            if value is not None and str(value).lower() != "nan":
                result[key] = value
    return result


def _video_direct_value(value_json):
    try:
        return _decoded_assertion_value(value_json)
    except Exception:
        pass
    if value_json is None:
        return None
    text = str(value_json).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _video_direct_value_text(value, unit=None):
    if isinstance(value, float):
        if math.isfinite(value):
            text = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
    elif isinstance(value, (list, tuple)):
        text = "[" + ", ".join(map(str, value)) + "]"
    else:
        text = str(value)
    unit_text = ""
    if unit is not None and str(unit).strip() and str(unit).lower() != "nan":
        unit_text = f" {unit}"
    return text + unit_text


def _video_direct_map_object(map_api, entity_id):
    if map_api is None or not NUPLAN_MAP_IMPORTS_AVAILABLE:
        return None
    text_id = str(entity_id or "").strip()
    if ":" not in text_id:
        return None
    kind, native_id = text_id.split(":", 1)
    layer_name = _VIDEO_DIRECT_MAP_LAYER_NAMES.get(kind)
    if not layer_name or not hasattr(SemanticMapLayer, layer_name):
        return None
    layer = getattr(SemanticMapLayer, layer_name)
    return _video_tl_get_map_object(map_api, native_id, layer)


def _video_direct_entity_lookup(entity_table):
    if (
        entity_table is None
        or entity_table.empty
        or "track_token" not in entity_table.columns
    ):
        return {}
    result = {}
    for row in entity_table.to_dict("records"):
        result[str(row.get("track_token"))] = row
    return result


def _video_direct_exact_assertion_events(
    frame,
    frame_assertions,
    entity_table,
    *,
    semantic_categories,
    semantic_predicates,
):
    """Build direct drawable events omitted by the generic edge renderer."""
    selected = normalize_semantic_predicate_selection(semantic_predicates)
    if selected is None:
        return []
    if frame_assertions is None or frame_assertions.empty:
        return []
    if "predicate_id" not in frame_assertions.columns:
        return []

    active_categories = enabled_names(semantic_categories)
    entity_lookup = _video_direct_entity_lookup(entity_table)
    events = []
    seen = set()

    for predicate_id in selected:
        # Traffic-light predicates have their own physically grounded overlay.
        if predicate_id in {
            "np:controls",
            "np:hasSignalState",
            "np:isRelevantSignal",
        }:
            continue

        meta = _video_direct_definition_meta(predicate_id)
        category = str(meta.get("category", ""))
        if category and category not in active_categories:
            continue
        value_type = str(meta.get("value_type", "")).strip().lower()
        unit = meta.get("units")

        rows = frame_assertions.loc[
            frame_assertions["predicate_id"].astype(str).eq(predicate_id)
        ]
        if rows.empty:
            continue

        for assertion in rows.itertuples(index=False):
            subject_id = str(getattr(assertion, "subject_id", "") or "")
            object_raw = getattr(assertion, "object_id", None)
            object_id = "" if object_raw is None else str(object_raw)
            if object_id.lower() in {"none", "nan", "null"}:
                object_id = ""

            subject_kind = _entity_kind(subject_id)
            object_kind = _entity_kind(object_id) if object_id else "other"
            subject_token = str(entity_id_to_track_token(subject_id))

            # Pairwise measurements are already converted to subject->object
            # edges by semantic_edge_table(). Do not duplicate them here.
            if subject_id.startswith("pair:"):
                continue

            # Unary literal/state predicates: attach a value label to the
            # physical subject (ego or tracked agent).
            if value_type != "entity" and subject_kind in {"ego", "agent"}:
                if subject_token not in entity_lookup:
                    continue
                value = _video_direct_value(
                    getattr(assertion, "value_json", None)
                )
                if value is None:
                    continue
                value_text = _video_direct_value_text(value, unit)
                key = (predicate_id, subject_id, value_text)
                if key in seen:
                    continue
                seen.add(key)
                events.append({
                    "visualization_type": "subject_value",
                    "predicate_id": predicate_id,
                    "relation_labels": predicate_id.replace("np:", ""),
                    "category": category,
                    "relation_type": "entity_value",
                    "subject_id": subject_id,
                    "object_id": "",
                    "subject_track_token": subject_token,
                    "value": value,
                    "value_text": value_text,
                    "unit": unit,
                })
                continue

            # Entity-valued agent/ego -> map structure predicates. The normal
            # graph renderer cannot draw these because a lane/crosswalk/etc. is
            # not a tracked-agent token. Draw the map geometry directly.
            if (
                value_type == "entity"
                and subject_kind in {"ego", "agent"}
                and object_kind == "structure"
            ):
                if subject_token not in entity_lookup:
                    continue
                key = (predicate_id, subject_id, object_id)
                if key in seen:
                    continue
                seen.add(key)
                events.append({
                    "visualization_type": "entity_to_map_structure",
                    "predicate_id": predicate_id,
                    "relation_labels": predicate_id.replace("np:", ""),
                    "category": category,
                    "relation_type": (
                        "ego_to_structure"
                        if subject_kind == "ego"
                        else "agent_to_structure"
                    ),
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "subject_track_token": subject_token,
                    "value": None,
                    "value_text": object_id,
                    "unit": None,
                })

    return events


def _video_direct_predicate_agent_tokens(events):
    return {
        str(event.get("subject_track_token"))
        for event in events
        if event.get("subject_track_token")
    }


def _video_direct_draw_events(
    ax,
    frame,
    entity_table,
    events,
    *,
    draw_semantic_arrows_on_map=True,
):
    if not events:
        return {
            "drawn_value_labels": 0,
            "drawn_map_relations": 0,
        }

    entity_lookup = _video_direct_entity_lookup(entity_table)
    map_api = getattr(frame, "map_api", None)
    value_stack = {}
    drawn_value = 0
    drawn_map = 0

    for event in events:
        token = str(event.get("subject_track_token") or "")
        subject = entity_lookup.get(token)
        if subject is None:
            continue
        sx = float(subject.get("x", 0.0))
        sy = float(subject.get("y", 0.0))
        predicate_label = str(event.get("relation_labels") or "predicate")
        vtype = event.get("visualization_type")

        if vtype == "subject_value":
            stack_index = value_stack.get(token, 0)
            value_stack[token] = stack_index + 1
            dy = 1.7 + 1.15 * stack_index
            label = f"{predicate_label}={event.get('value_text', '')}"
            ax.text(
                sx,
                sy + dy,
                label,
                fontsize=7.0,
                ha="center",
                va="bottom",
                zorder=22,
                clip_on=True,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#37474f",
                    "alpha": 0.94,
                    "linewidth": 0.9,
                },
            )
            drawn_value += 1
            continue

        if vtype == "entity_to_map_structure":
            object_id = str(event.get("object_id") or "")
            map_obj = _video_direct_map_object(map_api, object_id)
            anchor = _video_tl_anchor(map_obj) if map_obj is not None else None

            if map_obj is not None:
                _video_tl_draw_map_object(
                    ax,
                    map_obj,
                    facecolor="#ffd54f",
                    edgecolor="#6d4c41",
                    alpha=0.38,
                    linewidth=2.4,
                    zorder=10,
                )

            if anchor is not None:
                tx, ty = float(anchor[0]), float(anchor[1])
                if draw_semantic_arrows_on_map:
                    ax.annotate(
                        predicate_label,
                        xy=(tx, ty),
                        xytext=(sx, sy),
                        arrowprops={
                            "arrowstyle": "-|>",
                            "linewidth": 2.0,
                            "color": "#6a1b9a",
                            "alpha": 0.92,
                            "shrinkA": 8,
                            "shrinkB": 5,
                        },
                        color="#6a1b9a",
                        fontsize=7.2,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=21,
                    )
                ax.text(
                    tx,
                    ty,
                    object_id,
                    fontsize=6.5,
                    ha="center",
                    va="center",
                    zorder=20,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "#6d4c41",
                        "alpha": 0.88,
                    },
                )
            else:
                # Geometry lookup should normally succeed. This fallback keeps
                # the assertion visible even on map API/version mismatches.
                ax.text(
                    sx,
                    sy + 1.7,
                    f"{predicate_label} → {object_id}",
                    fontsize=7.0,
                    ha="center",
                    va="bottom",
                    zorder=22,
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "white",
                        "edgecolor": "#6a1b9a",
                        "alpha": 0.94,
                    },
                )
            drawn_map += 1

    return {
        "drawn_value_labels": int(drawn_value),
        "drawn_map_relations": int(drawn_map),
    }

def render_full_scene_configurable_semantic_video(
    *,
    scenario_selector="random",
    scenario_value=0,
    scenario_seed=42,
    filter_seed=42,
    output_dir=OUTPUT_DIR,
    export_root=VIDEO_EXPORT_ROOT,
    start_frame=0,
    end_frame=None,
    frame_step=1,
    fps=None,
    playback_speed=1.0,
    figsize=(12, 12),
    dpi=120,
    semantic_categories=None,
    semantic_relations=None,
    semantic_predicates="all",
    agent_types=None,
    interested_distance_threshold_m=40.0,
    include_within_distance=True,
    include_forward_corridor=True,
    forward_corridor_length_m=40.0,
    forward_corridor_half_width_m=5.0,
    include_same_lane=True,
    include_predicted_path_intersection=True,
    prediction_horizon_s=5.0,
    prediction_step_s=0.25,
    path_intersection_clearance_m=3.0,
    include_existing_spatial_agents=False,
    filter_edges_to_selected_agents=True,
    manual_highlight_tokens=None,
    map_radius_m=40.0,
    draw_semantic_arrows_on_map=True,
    show_agent_labels=True,
    show_agent_lane_info=False,
    show_map_structure_ids=False,
    show_edge_ids=False,
    show_status_box=True,
    draw_crosses_reference_lines=True,
):
    """
    Render one map-only MP4 using the same filters as Cell 15.

    Only semantic relations whose two endpoints have map coordinates in the
    frame can be drawn as arrows. Literal-valued predicates can still be
    selected in tables, but they do not create a subject-to-object map arrow.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for MP4 writing. Install opencv-python "
            "in the active nuPlan environment."
        ) from exc

    if semantic_categories is None:
        semantic_categories = VIDEO_SEMANTIC_CATEGORIES

    if semantic_relations is None:
        semantic_relations = VIDEO_SEMANTIC_RELATIONS

    if semantic_predicates is None:
        semantic_predicates = "all"

    if agent_types is None:
        agent_types = VIDEO_AGENT_TYPES

    if manual_highlight_tokens is None:
        manual_highlight_tokens = []

    active_categories = sorted(
        enabled_names(semantic_categories)
    )
    active_relations = sorted(
        enabled_names(semantic_relations)
    )
    active_predicate_filter = semantic_predicate_selection_label(
        semantic_predicates
    )

    validate_semantic_predicate_selection(
        semantic_predicates,
        DEFINITIONS,
        active_categories=active_categories,
    )

    if not active_categories:
        raise ValueError(
            "Enable at least one semantic category."
        )

    if not active_relations:
        raise ValueError(
            "Enable at least one semantic relation direction."
        )

    scenario, scenario_catalog_row = select_scenario(
        selector=scenario_selector,
        value=scenario_value,
        random_seed=scenario_seed,
        filter_seed=filter_seed,
    )

    scenario_token = str(scenario.token)
    frames = get_scenario_frames(scenario)

    if not frames:
        raise ValueError(
            "The selected scenario has no frames."
        )

    start_index = max(0, int(start_frame))
    stop_index = (
        len(frames)
        if end_frame is None
        else min(len(frames), int(end_frame) + 1)
    )
    step = max(1, int(frame_step))
    frame_indices = list(
        range(start_index, stop_index, step)
    )

    if not frame_indices:
        raise ValueError(
            "The configured frame range does not contain any frames."
        )

    scenario_assertions = load_assertions_for_scenario(
        str(Path(output_dir)),
        scenario_token,
    )

    if scenario_assertions.empty:
        raise RuntimeError(
            "No assertion rows were found for scenario "
            f"{scenario_token} in {Path(output_dir)}."
        )

    actual_fps = (
        float(fps)
        if fps is not None
        else _video_default_fps(
            frames,
            frame_indices,
            playback_speed,
        )
    )

    if not math.isfinite(actual_fps) or actual_fps <= 0:
        raise ValueError(
            "fps must be a positive finite number."
        )

    export_root = Path(export_root)
    export_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_fragment = _video_safe_filename_fragment(
        active_categories,
        "semantic",
    )
    video_path = export_root / (
        f"{scenario_token}_{category_fragment}_map_only.mp4"
    )
    temporary_video_path = export_root / (
        f"{scenario_token}_{category_fragment}_map_only_mp4v_temporary.mp4"
    )

    # Remove stale files from an interrupted or overwritten run.
    video_path.unlink(missing_ok=True)
    temporary_video_path.unlink(missing_ok=True)

    writer = None
    summary_rows = []
    edge_event_rows = []

    first_timestamp_us = int(
        frames[frame_indices[0]].timestamp_us
    )

    try:
        for rendered_index, frame_index in enumerate(frame_indices):
            frame = frames[frame_index]
            timestamp_us = int(frame.timestamp_us)

            frame_assertions = assertions_at_timestamp(
                scenario_assertions,
                timestamp_us,
                definitions=DEFINITIONS,
            )

            entity_table = add_stable_display_ids(
                build_entity_table(
                    frame,
                    scenario_token,
                ),
                scenario,
            )

            entity_table = filter_entities_by_agent_type(
                entity_table,
                agent_types,
            )

            enabled_agent_tokens = set(
                entity_table.loc[
                    ~entity_table["is_ego"].astype(bool),
                    "track_token",
                ].astype(str)
            )

            # Build the current-frame semantic edges using the normal
            # assertion-to-edge conversion.
            semantic_edges_all = semantic_edge_table(
                frame_assertions,
                DEFINITIONS,
                semantic_categories,
                semantic_relations,
                pair_assertions=scenario_assertions,
            )

            # IMPORTANT: use the exact same np:follows compatibility path as
            # the working single-frame Excel plot.  np:follows is an
            # entity-valued relation whose target is stored in object_id and
            # whose value_json is legitimately null.  Older output/definition
            # combinations may therefore omit it from semantic_edge_table().
            # The one-frame plot recovers it from the Follow evidence sheet;
            # here we build that same table in memory for every timestamp and
            # call the same recovery helper before applying the switches.
            if bool(semantic_categories.get("interaction", False)):
                semantic_edges_all = _restore_follow_edges_from_excel(
                    semantic_edges_all,
                    {
                        "Follow evidence": follow_evidence_table(
                            frame_assertions
                        )
                    },
                )

                # np:overtakes is also an entity-valued interaction relation.
                # Recover it directly from the current-frame assertions for
                # the same reason np:follows needs a compatibility fallback.
                semantic_edges_all = _video_restore_overtake_edges(
                    semantic_edges_all,
                    frame_assertions,
                )

                semantic_edges_all = _video_restore_merge_edges(
                    semantic_edges_all,
                    frame_assertions,
                )

            # Risk is rebuilt independently from the interaction family.  For
            # exact risk validation, restore the entity-valued relation
            # directly from the current-frame assertion if the generic edge
            # converter did not retain it.
            if bool(semantic_categories.get("risk", False)):
                semantic_edges_all = _video_restore_risk_edges(
                    semantic_edges_all,
                    frame_assertions,
                )

            # Normalize EGO and agent endpoints BEFORE relation-direction
            # filtering. This prevents an RDF-style EGO ID from being
            # classified or filtered differently from the frame-table EGO.
            semantic_edges_all = _video_normalize_edge_endpoints(
                semantic_edges_all,
                entity_table,
            )
            restored_ego_edge_count = _video_count_ego_edges(
                semantic_edges_all,
                entity_table,
            )

            # Apply the identical category/relation filtering used by
            # plot_from_master_excel() after follow-edge recovery.
            semantic_edges_all = _filter_master_semantic_edges(
                semantic_edges_all,
                semantic_categories,
                semantic_relations,
            )

            # Second visualization filter: exact predicate(s) inside the
            # enabled semantic family/families. The edge predicate list is
            # trimmed so arrow styling and labels cannot leak another
            # predicate carried by the same S->O edge.
            semantic_edges_all = filter_semantic_edges_by_predicates(
                semantic_edges_all,
                semantic_predicates,
            )

            semantic_edges_all = filter_edges_by_enabled_entity_tokens(
                semantic_edges_all,
                entity_table,
            )
            enabled_ego_edge_count = _video_count_ego_edges(
                semantic_edges_all,
                entity_table,
            )

            # Spatial edges are calculated independently because they may be
            # needed by INCLUDE_EXISTING_SPATIAL_AGENTS even when the spatial
            # category itself is disabled for display.
            all_spatial_edges = semantic_edge_table(
                frame_assertions,
                DEFINITIONS,
                _VIDEO_SPATIAL_ONLY_CATEGORIES,
                _VIDEO_ALL_RELATIONS,
                pair_assertions=scenario_assertions,
            )

            all_spatial_edges = _video_normalize_edge_endpoints(
                all_spatial_edges,
                entity_table,
            )
            all_spatial_edges = filter_edges_by_enabled_entity_tokens(
                all_spatial_edges,
                entity_table,
            )

            raw_follow_assertion_count = int(
                frame_assertions["predicate_id"]
                .astype(str)
                .eq("np:follows")
                .sum()
            )
            raw_overtake_assertion_count = int(
                frame_assertions["predicate_id"]
                .astype(str)
                .eq("np:overtakes")
                .sum()
            )
            raw_lane_change_assertion_count = int(
                frame_assertions["predicate_id"]
                .astype(str)
                .eq("np:changesLane")
                .sum()
            )
            recovered_follow_edge_count = int(
                semantic_edges_all["predicates"].map(
                    lambda values: "np:follows" in set(
                        _excel_predicate_list(values)
                    )
                ).sum()
            ) if (
                semantic_edges_all is not None
                and not semantic_edges_all.empty
                and "predicates" in semantic_edges_all.columns
            ) else 0

            candidate_tokens, selection_audit = (
                select_interesting_agents_for_plot(
                    frame,
                    entity_table,
                    all_spatial_edges,
                    distance_threshold_m=(
                        interested_distance_threshold_m
                    ),
                    include_within_distance=(
                        include_within_distance
                    ),
                    include_forward_corridor=(
                        include_forward_corridor
                    ),
                    forward_corridor_length_m=(
                        forward_corridor_length_m
                    ),
                    forward_corridor_half_width_m=(
                        forward_corridor_half_width_m
                    ),
                    include_same_lane=(
                        include_same_lane
                    ),
                    include_predicted_path_intersection=(
                        include_predicted_path_intersection
                    ),
                    prediction_horizon_s=(
                        prediction_horizon_s
                    ),
                    prediction_step_s=(
                        prediction_step_s
                    ),
                    path_intersection_clearance_m=(
                        path_intersection_clearance_m
                    ),
                    include_existing_spatial_agents=(
                        include_existing_spatial_agents
                    ),
                    manual_tokens=(
                        manual_highlight_tokens
                    ),
                )
            )

            candidate_tokens = (
                set(candidate_tokens)
                & enabled_agent_tokens
            )

            semantic_edges = (
                filter_edges_by_selected_agents(
                    semantic_edges_all,
                    candidate_tokens,
                )
                if filter_edges_to_selected_agents
                else semantic_edges_all.copy()
            )
            selected_ego_edge_count = _video_count_ego_edges(
                semantic_edges,
                entity_table,
            )

            # Normalize once more immediately before plotting so that no helper
            # between filtering and prepare_plot_edges() can reintroduce an
            # assertion-side EGO identifier.
            semantic_edges = _video_normalize_edge_endpoints(
                semantic_edges,
                entity_table,
            )
            plot_edges = prepare_plot_edges(
                semantic_edges,
                entity_table,
            )
            plotted_ego_edge_count = _video_count_ego_edges(
                plot_edges,
                entity_table,
            )

            traffic_light_events = _video_tl_build_overlay_rows(
                frame,
                frame_assertions,
                entity_table,
                semantic_categories=semantic_categories,
                semantic_relations=semantic_relations,
                semantic_predicates=semantic_predicates,
                scenario_assertions=scenario_assertions,
                timestamp_us=timestamp_us,
                frame_match_tolerance_us=_VIDEO_TL_FRAME_MATCH_TOLERANCE_US,
            )

            direct_predicate_events = _video_direct_exact_assertion_events(
                frame,
                frame_assertions,
                entity_table,
                semantic_categories=semantic_categories,
                semantic_predicates=semantic_predicates,
            )

            traffic_light_agent_tokens = {
                str(event["subject_track_token"])
                for event in traffic_light_events
                if (
                    event.get("predicate_id") == "np:isRelevantSignal"
                    and event.get("subject_track_token")
                )
            }
            direct_predicate_agent_tokens = (
                _video_direct_predicate_agent_tokens(
                    direct_predicate_events
                )
            )

            predicate_tokens = (
                _agent_tokens_with_displayed_predicates(
                    plot_edges
                )
                | traffic_light_agent_tokens
                | direct_predicate_agent_tokens
            )

            displayed_selected_tokens = (
                set(candidate_tokens)
                & set(predicate_tokens)
            )

            fig, ax = plt.subplots(
                1,
                1,
                figsize=figsize,
                dpi=dpi,
                constrained_layout=True,
            )

            plot_map_with_agents(
                ax,
                frame,
                entity_table,
                plot_edges,
                frame_assertions=frame_assertions,
                scenario_token=scenario_token,
                frame_index=frame_index,
                selected_tokens=displayed_selected_tokens,
                predicate_tokens=predicate_tokens,
                map_radius_m=float(map_radius_m),
                manual_highlight_tokens=(
                    manual_highlight_tokens
                ),
                draw_semantic_arrows=bool(
                    draw_semantic_arrows_on_map
                ),
                show_agent_labels=bool(
                    show_agent_labels
                ),
                show_agent_lane_info=bool(
                    show_agent_lane_info
                ),
                show_map_structure_ids=bool(
                    show_map_structure_ids
                ),
                show_edge_ids=bool(
                    show_edge_ids
                ),
                agent_type_switches=agent_types,
            )

            crosses_reference_pair_count = 0
            if bool(draw_crosses_reference_lines):
                crosses_reference_pair_count = (
                    _video_draw_crosses_reference_lines(
                        ax,
                        plot_edges,
                        entity_table,
                        map_radius_m=float(map_radius_m),
                    )
                )

            traffic_light_diagnostics = _video_tl_draw_overlays(
                ax,
                frame,
                entity_table,
                traffic_light_events,
                draw_semantic_arrows_on_map=bool(
                    draw_semantic_arrows_on_map
                ),
            )

            direct_predicate_diagnostics = _video_direct_draw_events(
                ax,
                frame,
                entity_table,
                direct_predicate_events,
                draw_semantic_arrows_on_map=bool(
                    draw_semantic_arrows_on_map
                ),
            )

            # Remove upper-right legend.
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

            lane_change_overlays = []
            if (
                bool(semantic_categories.get("interaction", False))
                and bool(draw_semantic_arrows_on_map)
                and semantic_predicate_is_selected(
                    "np:changesLane",
                    semantic_predicates,
                )
            ):
                lane_change_overlays = _video_draw_lane_change_overlays(
                    ax,
                    frame,
                    frame_assertions,
                    semantic_relations,
                    entity_table=entity_table,
                )

            elapsed_s = (
                timestamp_us - first_timestamp_us
            ) / 1e6

            overtake_evidence_by_pair = _video_overtake_evidence_by_pair(
                frame_assertions
            )
            overtake_phase_counts = _VideoCounter(
                str(evidence.get("overtake_phase_label") or "unspecified")
                for evidence in overtake_evidence_by_pair.values()
            )
            overtake_phase_text = ", ".join(
                f"{phase}: {count}"
                for phase, count in sorted(overtake_phase_counts.items())
            )

            lane_change_phase_counts = _VideoCounter(
                str(
                    overlay["evidence"].get("lane_change_phase_label")
                    or "unspecified"
                )
                for overlay in lane_change_overlays
            )
            lane_change_phase_text = ", ".join(
                f"{phase}: {count}"
                for phase, count in sorted(lane_change_phase_counts.items())
            )

            label_counts = _video_relation_label_counts(
                plot_edges
            )
            if lane_change_overlays:
                label_counts["changesLane"] += len(lane_change_overlays)

            for traffic_event in traffic_light_events:
                label_counts[
                    str(traffic_event["relation_labels"])
                ] += 1
            for direct_event in direct_predicate_events:
                label_counts[
                    str(direct_event["relation_labels"])
                ] += 1
            label_text = ", ".join(
                f"{label}: {count}"
                for label, count in sorted(label_counts.items())
            )
            if not label_text:
                label_text = "none"

            follow_mode_counts = (
                _video_displayed_follow_mode_counts(
                    frame_assertions,
                    plot_edges,
                )
                if bool(
                    semantic_categories.get(
                        "interaction",
                        False,
                    )
                )
                else _VideoCounter()
            )
            follow_mode_text = ", ".join(
                f"{mode}: {count}"
                for mode, count in sorted(
                    follow_mode_counts.items()
                )
            )

            category_text = ", ".join(active_categories)
            relation_text = ", ".join(active_relations)

            ax.set_title(
                "Configurable semantic map video\n"
                f"scenario={scenario_token} | frame={frame_index} | "
                f"time={elapsed_s:.1f} s | active relations="
                f"{len(plot_edges) + len(lane_change_overlays) + len(traffic_light_events) + len(direct_predicate_events)}"
            )

            if show_status_box:
                status_lines = [
                    f"Categories: {category_text}",
                    f"Predicate filter: {active_predicate_filter}",
                    f"Relations: {relation_text}",
                    f"Predicates: {label_text}",
                ]

                if follow_mode_text:
                    status_lines.append(
                        f"Follow modes: {follow_mode_text}"
                    )
                if overtake_phase_text:
                    status_lines.append(
                        f"Overtake phase: {overtake_phase_text}"
                    )

                if lane_change_phase_text:
                    status_lines.append(
                        f"Lane-change phase: {lane_change_phase_text}"
                    )

                ax.text(
                    0.015,
                    0.985,
                    "\n".join(status_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    zorder=30,
                    bbox={
                        "boxstyle": "round,pad=0.35",
                        "facecolor": "white",
                        "edgecolor": "#6a1b9a",
                        "alpha": 0.92,
                    },
                )

            fig.canvas.draw()
            rgba = np.asarray(
                fig.canvas.buffer_rgba()
            )
            rgb = np.ascontiguousarray(
                rgba[:, :, :3]
            )
            height, width = rgb.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
                writer = cv2.VideoWriter(
                    str(temporary_video_path),
                    fourcc,
                    actual_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError(
                        "OpenCV could not open the MP4 writer. Confirm "
                        "that the active environment supports mp4v."
                    )

            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )
            writer.write(bgr)
            plt.close(fig)

            raw_ego_follow_assertion_count = (
                _video_count_raw_ego_follows(frame_assertions)
            )

            summary_rows.append({
                "rendered_frame": rendered_index,
                "scene_frame_index": frame_index,
                "timestamp_us": timestamp_us,
                "elapsed_s": elapsed_s,
                "selected_agent_count": int(
                    len(candidate_tokens)
                ),
                "displayed_predicate_agent_count": int(
                    len(predicate_tokens)
                ),
                "semantic_edge_count": int(
                    len(plot_edges)
                ),
                "active_relation_count": int(
                    len(plot_edges)
                    + len(lane_change_overlays)
                    + len(traffic_light_events)
                    + len(direct_predicate_events)
                ),
                "direct_predicate_event_count": int(
                    len(direct_predicate_events)
                ),
                "drawn_direct_value_label_count": int(
                    direct_predicate_diagnostics[
                        "drawn_value_labels"
                    ]
                ),
                "drawn_direct_map_relation_count": int(
                    direct_predicate_diagnostics[
                        "drawn_map_relations"
                    ]
                ),
                "traffic_light_event_count": int(
                    len(traffic_light_events)
                ),
                "drawn_isRelevantSignal_count": int(
                    traffic_light_diagnostics[
                        "drawn_is_relevant_signal"
                    ]
                ),
                "drawn_controls_count": int(
                    traffic_light_diagnostics[
                        "drawn_controls"
                    ]
                ),
                "drawn_signal_state_count": int(
                    traffic_light_diagnostics[
                        "drawn_states"
                    ]
                ),
                "raw_follow_assertion_count": int(
                    raw_follow_assertion_count
                ),
                "raw_overtake_assertion_count": int(
                    raw_overtake_assertion_count
                ),
                "raw_lane_change_assertion_count": int(
                    raw_lane_change_assertion_count
                ),
                "lane_change_overlay_count": int(
                    len(lane_change_overlays)
                ),
                "recovered_follow_edge_count": int(
                    recovered_follow_edge_count
                ),
                "raw_ego_follow_assertion_count": int(
                    raw_ego_follow_assertion_count
                ),
                "restored_ego_edge_count": int(
                    restored_ego_edge_count
                ),
                "enabled_ego_edge_count": int(
                    enabled_ego_edge_count
                ),
                "selected_ego_edge_count": int(
                    selected_ego_edge_count
                ),
                "plotted_ego_edge_count": int(
                    plotted_ego_edge_count
                ),
                "predicate_labels": label_text,
                "crosses_reference_pair_count": int(
                    crosses_reference_pair_count
                ),
                "follow_modes": (
                    follow_mode_text
                    if follow_mode_text
                    else "none"
                ),
                "overtake_phases": (
                    overtake_phase_text
                    if overtake_phase_text
                    else "none"
                ),
                "lane_change_phases": (
                    lane_change_phase_text
                    if lane_change_phase_text
                    else "none"
                ),
            })

            for traffic_event_index, traffic_event in enumerate(
                traffic_light_events,
                start=1,
            ):
                edge_event_rows.append({
                    "scene_frame_index": frame_index,
                    "timestamp_us": timestamp_us,
                    "elapsed_s": elapsed_s,
                    "edge_id_in_frame": (
                        f"TL{traffic_event_index}"
                    ),
                    "subject": traffic_event["subject"],
                    "object": traffic_event["object"],
                    "relation_type": traffic_event[
                        "relation_type"
                    ],
                    "categories": "traffic_light",
                    "relation_labels": traffic_event[
                        "relation_labels"
                    ],
                    "predicate_id": traffic_event[
                        "predicate_id"
                    ],
                    "traffic_signal_id": traffic_event[
                        "signal_id"
                    ],
                    "native_lane_connector_id": traffic_event[
                        "connector_id"
                    ],
                    "signal_state": traffic_event[
                        "state"
                    ],
                    "assertion_timestamp_us": traffic_event.get(
                        "assertion_timestamp_us"
                    ),
                    "frame_time_delta_us": traffic_event.get(
                        "frame_time_delta_us"
                    ),
                    "is_traffic_light_event": True,
                })

            for direct_event_index, direct_event in enumerate(
                direct_predicate_events,
                start=1,
            ):
                subject_token = str(
                    direct_event.get("subject_track_token") or ""
                )
                subject_display = subject_token
                if (
                    entity_table is not None
                    and not entity_table.empty
                    and subject_token
                    and "track_token" in entity_table.columns
                ):
                    _subject_rows = entity_table.loc[
                        entity_table["track_token"].astype(str).eq(
                            subject_token
                        )
                    ]
                    if not _subject_rows.empty:
                        subject_display = str(
                            _subject_rows.iloc[0].get(
                                "display_id",
                                subject_token,
                            )
                        )

                edge_event_rows.append({
                    "scene_frame_index": frame_index,
                    "timestamp_us": timestamp_us,
                    "elapsed_s": elapsed_s,
                    "edge_id_in_frame": f"D{direct_event_index}",
                    "subject": subject_display,
                    "object": (
                        direct_event.get("object_id")
                        or direct_event.get("value_text")
                    ),
                    "relation_type": direct_event.get(
                        "relation_type"
                    ),
                    "categories": direct_event.get("category"),
                    "relation_labels": direct_event.get(
                        "relation_labels"
                    ),
                    "predicate_id": direct_event.get("predicate_id"),
                    "predicate_value": direct_event.get("value"),
                    "predicate_value_text": direct_event.get(
                        "value_text"
                    ),
                    "predicate_unit": direct_event.get("unit"),
                    "direct_visualization_type": direct_event.get(
                        "visualization_type"
                    ),
                    "is_direct_predicate_event": True,
                })

            for overlay in lane_change_overlays:
                evidence = overlay["evidence"]
                edge_event_rows.append({
                    "scene_frame_index": frame_index,
                    "timestamp_us": timestamp_us,
                    "elapsed_s": elapsed_s,
                    "edge_id_in_frame": evidence.get("event_id"),
                    "subject": overlay["subject_token"],
                    "object": evidence.get("target_lane_entity_id") or overlay["object_id"],
                    "relation_type": overlay["relation_type"],
                    "categories": "interaction",
                    "relation_labels": "changesLane",
                    "is_active_lane_change": True,
                    "lane_change_side": evidence.get("lane_change_side"),
                    "lane_change_phase": evidence.get("lane_change_phase"),
                    "lane_change_phase_number": evidence.get("lane_change_phase_number"),
                    "lane_change_phase_label": evidence.get("lane_change_phase_label"),
                    "source_lane_id": evidence.get("source_lane_id"),
                    "target_lane_id": evidence.get("target_lane_id"),
                    "lane_change_arrow_start_x": evidence.get("lane_change_arrow_start_x"),
                    "lane_change_arrow_start_y": evidence.get("lane_change_arrow_start_y"),
                    "lane_change_arrow_end_x": evidence.get("lane_change_arrow_end_x"),
                    "lane_change_arrow_end_y": evidence.get("lane_change_arrow_end_y"),
                })

            follow_mode_by_pair = _video_follow_mode_by_pair(
                frame_assertions
            )

            if plot_edges is not None and not plot_edges.empty:
                for edge in plot_edges.itertuples(index=False):
                    edge_subject_token = str(
                        getattr(edge, "subject_token", "")
                    )
                    edge_object_token = str(
                        getattr(edge, "object_token", "")
                    )
                    edge_predicates = {
                        str(predicate)
                        for predicate in getattr(
                            edge, "predicates", []
                        )
                    }
                    follow_mode = (
                        follow_mode_by_pair.get((
                            edge_subject_token,
                            edge_object_token,
                        ))
                        if "np:follows" in edge_predicates
                        else None
                    )
                    overtake_evidence = (
                        overtake_evidence_by_pair.get((
                            edge_subject_token,
                            edge_object_token,
                        ), {})
                        if "np:overtakes" in edge_predicates
                        else {}
                    )

                    edge_predicate_list = sorted(edge_predicates)
                    edge_event_rows.append({
                        "scene_frame_index": frame_index,
                        "timestamp_us": timestamp_us,
                        "elapsed_s": elapsed_s,
                        "predicate_id": (
                            edge_predicate_list[0]
                            if len(edge_predicate_list) == 1
                            else None
                        ),
                        "predicate_ids": ",".join(edge_predicate_list),
                        "edge_id_in_frame": getattr(
                            edge,
                            "edge_id",
                            None,
                        ),
                        "subject": getattr(
                            edge,
                            "subject_display_id",
                            getattr(edge, "subject_token", None),
                        ),
                        "object": getattr(
                            edge,
                            "object_display_id",
                            getattr(edge, "object_token", None),
                        ),
                        "relation_type": getattr(
                            edge,
                            "relation_type",
                            None,
                        ),
                        "categories": ", ".join(
                            map(
                                str,
                                getattr(edge, "categories", []),
                            )
                        ),
                        "relation_labels": getattr(
                            edge,
                            "relation_labels",
                            None,
                        ),
                        "follow_mode": follow_mode,
                        "is_moving_headway": (
                            follow_mode == "moving_headway"
                        ),
                        "is_queue_or_stop_and_go": (
                            follow_mode
                            == "queue_or_stop_and_go"
                        ),
                        "is_active_overtake": (
                            "np:overtakes" in edge_predicates
                        ),
                        "is_completed_overtake": overtake_evidence.get(
                            "overtake_completed"
                        ),
                        "overtake_phase": overtake_evidence.get(
                            "overtake_phase"
                        ),
                        "overtake_phase_number": overtake_evidence.get(
                            "overtake_phase_number"
                        ),
                        "overtake_phase_label": overtake_evidence.get(
                            "overtake_phase_label"
                        ),
                        "overtake_assignment_scope": overtake_evidence.get(
                            "overtake_assignment_scope"
                        ),
                        "overtake_case": overtake_evidence.get(
                            "overtake_case"
                        ),
                        "overtake_case_number": overtake_evidence.get(
                            "overtake_case_number"
                        ),
                        "overtake_case_label": overtake_evidence.get(
                            "overtake_case_label"
                        ),
                        "overtake_initial_lane_relation": overtake_evidence.get(
                            "overtake_initial_lane_relation"
                        ),
                        "overtake_observation_mode": overtake_evidence.get(
                            "overtake_observation_mode"
                        ),
                        "relation_family": overtake_evidence.get(
                            "relation_family"
                        ),
                        "lane_departure_observed": overtake_evidence.get(
                            "lane_departure_observed"
                        ),
                        "overtake_entry_mode": overtake_evidence.get(
                            "overtake_entry_mode"
                        ),
                        "initial_follow_observed": overtake_evidence.get(
                            "initial_follow_observed"
                        ),
                        "initial_follow_mode": overtake_evidence.get(
                            "initial_follow_mode"
                        ),
                        "overtake_completion_mode": overtake_evidence.get(
                            "overtake_completion_mode"
                        ),
                        "returned_to_original_lane": overtake_evidence.get(
                            "returned_to_original_lane"
                        ),
                        "remained_in_passing_lane": overtake_evidence.get(
                            "remained_in_passing_lane"
                        ),
                        "passing_side": overtake_evidence.get(
                            "passing_side"
                        ),
                        "initial_lane_id": overtake_evidence.get(
                            "initial_lane_id"
                        ),
                        "initial_subject_lane_id": overtake_evidence.get(
                            "initial_subject_lane_id"
                        ),
                        "initial_object_lane_id": overtake_evidence.get(
                            "initial_object_lane_id"
                        ),
                        "passing_lane_id": overtake_evidence.get(
                            "passing_lane_id"
                        ),
                        "final_lane_id": overtake_evidence.get(
                            "final_lane_id"
                        ),
                        "overtake_start_time_us": overtake_evidence.get(
                            "start_time_us"
                        ),
                        "lane_departure_time_us": overtake_evidence.get(
                            "lane_departure_time_us"
                        ),
                        "passing_lane_observation_start_time_us": overtake_evidence.get(
                            "passing_lane_observation_start_time_us"
                        ),
                        "side_by_side_time_us": overtake_evidence.get(
                            "side_by_side_time_us"
                        ),
                        "order_reversal_time_us": overtake_evidence.get(
                            "order_reversal_time_us"
                        ),
                        "clearance_time_us": overtake_evidence.get(
                            "clearance_time_us"
                        ),
                        "return_time_us": overtake_evidence.get(
                            "return_time_us"
                        ),
                        "final_order_stable_start_time_us": overtake_evidence.get(
                            "final_order_stable_start_time_us"
                        ),
                        "completion_time_us": overtake_evidence.get(
                            "completion_time_us"
                        ),
                        "final_order_frame_count": overtake_evidence.get(
                            "final_order_frame_count"
                        ),
                        "final_clearance_gap_m": overtake_evidence.get(
                            "final_clearance_gap_m"
                        ),
                        "maximum_relative_speed_mps": overtake_evidence.get(
                            "maximum_relative_speed_mps"
                        ),
                    })

            if (
                rendered_index == 0
                or (rendered_index + 1) % 10 == 0
                or rendered_index + 1 == len(frame_indices)
            ):
                print(
                    f"Rendered {rendered_index + 1:,}/"
                    f"{len(frame_indices):,} frames; "
                    f"active semantic edges={len(plot_edges):,}; "
                    f"traffic-light events={len(traffic_light_events):,}; "
                    f"isRelevantSignal drawn="
                    f"{traffic_light_diagnostics['drawn_is_relevant_signal']:,}; "
                    f"raw follows={raw_follow_assertion_count:,}; "
                    f"raw overtakes={raw_overtake_assertion_count:,}; "
                    f"recovered follows={recovered_follow_edge_count:,}; "
                    f"raw/restored/enabled/selected/plotted EGO="
                    f"{raw_ego_follow_assertion_count:,}/"
                    f"{restored_ego_edge_count:,}/"
                    f"{enabled_ego_edge_count:,}/"
                    f"{selected_ego_edge_count:,}/"
                    f"{plotted_ego_edge_count:,}"
                )

    finally:
        if writer is not None:
            writer.release()
        plt.close("all")

    if (
        not temporary_video_path.exists()
        or temporary_video_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "The OpenCV video writer did not create a valid temporary "
            f"MP4V file: {temporary_video_path}"
        )

    _video_convert_mp4v_to_browser_h264(
        temporary_video_path,
        video_path,
    )

    summary = pd.DataFrame(summary_rows)
    edge_events = pd.DataFrame(edge_event_rows)

    active_frame_count = (
        int(
            (summary["active_relation_count"] > 0).sum()
        )
        if not summary.empty
        else 0
    )

    print("\nMAP-ONLY SEMANTIC VIDEO SAVED")
    print("Scenario catalog row:", scenario_catalog_row)
    print("Scenario token:", scenario_token)
    print("Categories:", active_categories)
    print("Predicate filter:", active_predicate_filter)
    print("Relations:", active_relations)
    print("Rendered frames:", len(frame_indices))
    print("Frames containing enabled semantic relations:", active_frame_count)
    print("FPS:", actual_fps)
    print("Codec: H.264 (libx264), pixel format: yuv420p")
    print("Web optimization: faststart enabled")
    print("File:", video_path.resolve())

    return {
        "video_path": video_path,
        "scenario": scenario,
        "scenario_token": scenario_token,
        "scenario_catalog_row": scenario_catalog_row,
        "fps": actual_fps,
        "frame_indices": frame_indices,
        "summary": summary,
        "edge_events": edge_events,
        "active_frame_count": active_frame_count,
        "active_categories": active_categories,
        "active_predicates": active_predicate_filter,
        "active_relations": active_relations,
    }


VIDEO_RESULT = render_full_scene_configurable_semantic_video(
    scenario_selector=VIDEO_SCENARIO_SELECTOR,
    scenario_value=VIDEO_SCENARIO_VALUE,
    scenario_seed=VIDEO_SCENARIO_SEED,
    filter_seed=VIDEO_FILTER_SEED,
    output_dir=OUTPUT_DIR,
    export_root=VIDEO_EXPORT_ROOT,
    start_frame=VIDEO_START_FRAME,
    end_frame=VIDEO_END_FRAME,
    frame_step=VIDEO_FRAME_STEP,
    fps=VIDEO_FPS,
    playback_speed=VIDEO_PLAYBACK_SPEED,
    figsize=VIDEO_FIGSIZE,
    dpi=VIDEO_DPI,
    semantic_categories=VIDEO_SEMANTIC_CATEGORIES,
    semantic_relations=VIDEO_SEMANTIC_RELATIONS,
    semantic_predicates=VIDEO_SEMANTIC_PREDICATES,
    agent_types=VIDEO_AGENT_TYPES,
    interested_distance_threshold_m=(
        VIDEO_INTERESTED_DISTANCE_THRESHOLD_M
    ),
    include_within_distance=(
        VIDEO_INCLUDE_WITHIN_DISTANCE
    ),
    include_forward_corridor=(
        VIDEO_INCLUDE_FORWARD_CORRIDOR
    ),
    forward_corridor_length_m=(
        VIDEO_FORWARD_CORRIDOR_LENGTH_M
    ),
    forward_corridor_half_width_m=(
        VIDEO_FORWARD_CORRIDOR_HALF_WIDTH_M
    ),
    include_same_lane=(
        VIDEO_INCLUDE_SAME_LANE
    ),
    include_predicted_path_intersection=(
        VIDEO_INCLUDE_PREDICTED_PATH_INTERSECTION
    ),
    prediction_horizon_s=(
        VIDEO_PREDICTION_HORIZON_S
    ),
    prediction_step_s=(
        VIDEO_PREDICTION_STEP_S
    ),
    path_intersection_clearance_m=(
        VIDEO_PATH_INTERSECTION_CLEARANCE_M
    ),
    include_existing_spatial_agents=(
        VIDEO_INCLUDE_EXISTING_SPATIAL_AGENTS
    ),
    filter_edges_to_selected_agents=(
        VIDEO_FILTER_EDGES_TO_SELECTED_AGENTS
    ),
    manual_highlight_tokens=(
        VIDEO_MANUAL_HIGHLIGHT_TOKENS
    ),
    map_radius_m=VIDEO_MAP_RADIUS_M,
    draw_semantic_arrows_on_map=(
        VIDEO_DRAW_SEMANTIC_ARROWS_ON_MAP
    ),
    show_agent_labels=VIDEO_SHOW_AGENT_LABELS,
    show_agent_lane_info=VIDEO_SHOW_AGENT_LANE_INFO,
    show_map_structure_ids=VIDEO_SHOW_MAP_STRUCTURE_IDS,
    show_edge_ids=VIDEO_SHOW_EDGE_IDS,
    show_status_box=VIDEO_SHOW_STATUS_BOX,
    draw_crosses_reference_lines=(
        VIDEO_DRAW_CROSSES_REFERENCE_LINES
    ),
)


# Display the saved MP4 without embedding all video bytes.
display(
    _NotebookVideo(
        filename=str(VIDEO_RESULT["video_path"]),
        embed=False,
    )
)


# ============================================================
# OPTIONAL OUTPUTS CONTROLLED BY CELL 15 SWITCHES
# ============================================================

active_video_frames = VIDEO_RESULT["summary"].loc[
    VIDEO_RESULT["summary"]["semantic_edge_count"] > 0
].reset_index(drop=True)

if VIDEO_SHOW_FILTERED_TABLES:
    print("\nFRAMES WITH ACTIVE ENABLED SEMANTIC RELATIONS")

    if active_video_frames.empty:
        print(
            "No enabled semantic relation occurred in the rendered "
            "frame range."
        )
    else:
        display(
            active_video_frames.head(
                VIDEO_MAX_FILTERED_TABLE_ROWS
            )
        )

        if (
            len(active_video_frames)
            > VIDEO_MAX_FILTERED_TABLE_ROWS
        ):
            print(
                "Showing "
                f"{VIDEO_MAX_FILTERED_TABLE_ROWS:,} of "
                f"{len(active_video_frames):,} active frames."
            )

if VIDEO_SHOW_EDGE_LEGEND:
    print("\nSEMANTIC RELATIONS OBSERVED IN THE VIDEO")

    edge_events = VIDEO_RESULT["edge_events"]

    if edge_events.empty:
        print(
            "No enabled semantic relation was drawable in the video."
        )
    else:
        video_edge_legend = (
            edge_events.groupby(
                [
                    "subject",
                    "object",
                    "relation_type",
                    "categories",
                    "relation_labels",
                ],
                dropna=False,
            )
            .agg(
                first_frame=("scene_frame_index", "min"),
                last_frame=("scene_frame_index", "max"),
                active_frame_count=("scene_frame_index", "nunique"),
            )
            .reset_index()
            .sort_values(
                [
                    "first_frame",
                    "subject",
                    "object",
                    "relation_labels",
                ]
            )
            .reset_index(drop=True)
        )

        if VIDEO_MAX_EDGE_LEGEND_ROWS is None:
            display(video_edge_legend)
        else:
            display(
                video_edge_legend.head(
                    int(VIDEO_MAX_EDGE_LEGEND_ROWS)
                )
            )
