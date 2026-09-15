"""Top-level scenario loop and safe multi-process orchestration."""
import os
import subprocess
import fcntl
import traceback
import time

from .base import *
from .frame_pipeline import extract_frame_predicates
from .categories.scenario import derive_scenario_level_predicates
from .output import write_batch, _definition_status, _write_csv_records


def _format_duration(seconds):
    """Format seconds as HH:MM:SS, or Dd HH:MM:SS for long runs."""
    if seconds is None:
        return "--:--:--"

    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError, OverflowError):
        return "--:--:--"

    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _scenario_effective_frame_count(scenario, max_frames_per_scenario):
    """Return the number of frames expected to be processed for scheduling."""
    try:
        iterations = max(0, int(scenario.get_number_of_iterations()))
    except Exception:
        iterations = 0

    if max_frames_per_scenario is None:
        return iterations

    try:
        frame_limit = max(0, int(max_frames_per_scenario))
    except (TypeError, ValueError):
        return iterations

    return min(iterations, frame_limit)


def _load_scenario_timing_hints(timing_csv_path):
    """Load optional exact-token and scenario-type runtime hints.

    The expected file is a previous ``scenario_processing_times.csv``.  When a
    sibling ``scenario_catalog.csv`` exists, its scenario types are merged with
    the timing rows.  Missing, malformed, or partial hint files safely fall
    back to frame-count scheduling.
    """
    hints = {
        "path": None,
        "exact_token_elapsed_s": {},
        "scenario_type_elapsed_s": {},
        "global_elapsed_s": None,
    }
    if not timing_csv_path:
        return hints

    path = Path(timing_csv_path).expanduser().resolve()
    if not path.exists():
        return hints

    try:
        timing_df = pd.read_csv(path)
    except Exception:
        return hints

    required_columns = {"scenario_token", "elapsed_s"}
    if not required_columns.issubset(timing_df.columns):
        return hints

    timing_df = timing_df.copy()
    timing_df["scenario_token"] = timing_df["scenario_token"].astype(str)
    timing_df["elapsed_s"] = pd.to_numeric(
        timing_df["elapsed_s"], errors="coerce"
    )
    timing_df = timing_df[
        timing_df["elapsed_s"].notna() & (timing_df["elapsed_s"] >= 0)
    ]
    if "status" in timing_df.columns:
        timing_df = timing_df[timing_df["status"].astype(str) == "completed"]
    if timing_df.empty:
        return hints

    exact = (
        timing_df.groupby("scenario_token", dropna=False)["elapsed_s"]
        .mean()
        .to_dict()
    )
    hints["exact_token_elapsed_s"] = {
        str(token): float(value) for token, value in exact.items()
    }
    hints["global_elapsed_s"] = float(timing_df["elapsed_s"].mean())
    hints["path"] = str(path)

    catalog_path = path.with_name("scenario_catalog.csv")
    if not catalog_path.exists():
        return hints

    try:
        catalog_df = pd.read_csv(catalog_path)
    except Exception:
        return hints

    if "scenario_type" not in catalog_df.columns:
        return hints

    merged = None
    if "scenario_index" in timing_df.columns and "index" in catalog_df.columns:
        merged = timing_df.merge(
            catalog_df[["index", "scenario_type"]],
            left_on="scenario_index",
            right_on="index",
            how="left",
        )
    elif "token" in catalog_df.columns:
        merged = timing_df.merge(
            catalog_df[["token", "scenario_type"]],
            left_on="scenario_token",
            right_on="token",
            how="left",
        )

    if merged is not None and "scenario_type" in merged.columns:
        merged = merged[merged["scenario_type"].notna()]
        if not merged.empty:
            type_means = (
                merged.groupby("scenario_type", dropna=False)["elapsed_s"]
                .mean()
                .to_dict()
            )
            hints["scenario_type_elapsed_s"] = {
                str(scenario_type): float(value)
                for scenario_type, value in type_means.items()
            }

    return hints


def _order_scenarios_longest_first(
    scenarios,
    max_frames_per_scenario,
    timing_csv_path=None,
):
    """Return a deterministic cost-aware longest-processing-time-first order.

    Priority uses, in order:
      1. exact runtime of the same scenario token from a previous run;
      2. mean runtime of the same scenario type from a previous run;
      3. effective frame count as a deterministic fallback.

    This leaves each scenario intact, preserving temporal history while
    reducing the chance that a few expensive scenarios remain at the end.
    """
    timing_hints = _load_scenario_timing_hints(timing_csv_path)
    exact_hints = timing_hints["exact_token_elapsed_s"]
    type_hints = timing_hints["scenario_type_elapsed_s"]
    global_hint = timing_hints["global_elapsed_s"]

    decorated = []
    for original_index, scenario in enumerate(scenarios):
        estimated_frames = _scenario_effective_frame_count(
            scenario, max_frames_per_scenario
        )
        token = str(getattr(scenario, "token", ""))
        log_name = str(getattr(scenario, "log_name", ""))
        scenario_type = str(getattr(scenario, "scenario_type", ""))

        if token in exact_hints:
            predicted_elapsed_s = float(exact_hints[token])
            priority_source = "exact_previous_scenario_time"
            priority_source_rank = 3
        elif scenario_type in type_hints:
            predicted_elapsed_s = float(type_hints[scenario_type])
            priority_source = "previous_scenario_type_mean"
            priority_source_rank = 2
        elif global_hint is not None:
            predicted_elapsed_s = float(global_hint)
            priority_source = "previous_global_mean"
            priority_source_rank = 1
        else:
            predicted_elapsed_s = None
            priority_source = "effective_frame_count"
            priority_source_rank = 0

        decorated.append({
            "original_index": int(original_index),
            "scenario": scenario,
            "estimated_frames": int(estimated_frames),
            "predicted_elapsed_s": predicted_elapsed_s,
            "priority_source": priority_source,
            "priority_source_rank": priority_source_rank,
            "token": token,
            "log_name": log_name,
            "scenario_type": scenario_type,
        })

    decorated.sort(
        key=lambda row: (
            -(
                row["predicted_elapsed_s"]
                if row["predicted_elapsed_s"] is not None
                else float(row["estimated_frames"])
            ),
            -row["estimated_frames"],
            row["log_name"],
            row["token"],
            row["original_index"],
        )
    )

    ordered_scenarios = [row["scenario"] for row in decorated]
    schedule_metadata = [
        {
            "scheduled_index": int(scheduled_index),
            "original_index": int(row["original_index"]),
            "estimated_frames": int(row["estimated_frames"]),
            "predicted_elapsed_s": row["predicted_elapsed_s"],
            "priority_source": row["priority_source"],
            "token": row["token"],
            "log_name": row["log_name"],
            "scenario_type": row["scenario_type"],
        }
        for scheduled_index, row in enumerate(decorated)
    ]
    return ordered_scenarios, schedule_metadata, timing_hints


def _atomic_write_json(path, payload):
    """Atomically publish a small JSON status file for the coordinator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_worker_progress(path):
    """Read a worker status file safely while it may be replaced."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _update_task_queue(queue_path, updater):
    """Atomically update and return the shared dynamic queue state."""
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            state = json.loads(raw) if raw else {}
            updater(state)
            state["updated_at_epoch_s"] = time.time()
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(state, indent=2, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
            return state
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _claim_next_scenario_index(queue_path, total_scenarios, worker_index):
    """Atomically claim one pending scenario for a worker."""
    claimed = {"index": None}

    def updater(state):
        # Workers all build the same filtered catalogue. Publish the actual
        # catalogue size rather than trusting a possibly larger --max-scenarios
        # value that the coordinator used before filtering completed.
        if int(state.get("total_scenarios") or 0) != int(total_scenarios):
            state["total_scenarios"] = int(total_scenarios)
        state.setdefault("next_index", 0)
        state.setdefault("claims", 0)
        state.setdefault("completed_indices", [])
        state.setdefault("failed_indices", [])
        state.setdefault("in_progress", {})

        completed = set(int(x) for x in state["completed_indices"])
        failed = set(int(x) for x in state["failed_indices"])
        in_progress = {int(k): v for k, v in state["in_progress"].items()}

        index = int(state.get("next_index", 0))
        while index < int(total_scenarios) and (
            index in completed or index in failed or index in in_progress
        ):
            index += 1

        if index >= int(total_scenarios):
            state["exhausted"] = True
            return

        claimed["index"] = index
        state["next_index"] = index + 1
        state["claims"] = int(state.get("claims", 0)) + 1
        state["in_progress"][str(index)] = {
            "worker_index": int(worker_index),
            "claimed_at_epoch_s": time.time(),
        }
        state["exhausted"] = state["next_index"] >= int(total_scenarios)

    _update_task_queue(queue_path, updater)
    return claimed["index"]


def _mark_scenario_done(
    queue_path,
    scenario_index,
    worker_index,
    elapsed_s,
    scenario_token=None,
):
    def updater(state):
        state.setdefault("completed_indices", [])
        state.setdefault("failed_indices", [])
        state.setdefault("in_progress", {})
        state["in_progress"].pop(str(int(scenario_index)), None)
        if int(scenario_index) not in state["completed_indices"]:
            state["completed_indices"].append(int(scenario_index))
        state["completed_indices"].sort()
        state.setdefault("scenario_results", {})[str(int(scenario_index))] = {
            "status": "completed",
            "worker_index": int(worker_index),
            "scenario_token": (
                str(scenario_token) if scenario_token is not None else None
            ),
            "elapsed_s": round(float(elapsed_s), 3),
        }
    return _update_task_queue(queue_path, updater)


def _mark_scenario_failed(
    queue_path,
    scenario_index,
    worker_index,
    exc,
    scenario_token=None,
):
    def updater(state):
        state.setdefault("completed_indices", [])
        state.setdefault("failed_indices", [])
        state.setdefault("in_progress", {})
        state["in_progress"].pop(str(int(scenario_index)), None)
        if int(scenario_index) not in state["failed_indices"]:
            state["failed_indices"].append(int(scenario_index))
        state["failed_indices"].sort()
        state.setdefault("scenario_results", {})[str(int(scenario_index))] = {
            "status": "failed",
            "worker_index": int(worker_index),
            "scenario_token": (
                str(scenario_token) if scenario_token is not None else None
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return _update_task_queue(queue_path, updater)

def _replace_cli_option(argv, option, value=None):
    """Return argv with one option replaced, supporting --x value and --x=value."""
    result = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == option:
            if value is None:
                continue
            result.extend([option, str(value)])
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                skip_next = True
            continue
        if item.startswith(option + "="):
            if value is not None:
                result.append(f"{option}={value}")
            continue
        result.append(item)
    if value is not None and not any(
        item == option or item.startswith(option + "=") for item in result
    ):
        result.extend([option, str(value)])
    return result


def _run_parallel_coordinator():
    """Launch isolated workers; never pickle nuPlan scenario/map objects."""
    worker_count = int(ARGS.num_workers)
    coordinator_start = time.time()
    workers_root = OUTPUT_DIR / "workers"
    logs_root = OUTPUT_DIR / "worker_logs"
    workers_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    base_argv = list(sys.argv[1:])
    base_argv = _replace_cli_option(base_argv, "--num-workers", 1)
    base_argv = _replace_cli_option(base_argv, "--worker-count", worker_count)

    task_queue_path = (OUTPUT_DIR / "dynamic_task_queue.json").resolve()
    _atomic_write_json(task_queue_path, {
        "next_index": 0,
        "claims": 0,
        "total_scenarios": ARGS.max_scenarios,
        "completed_indices": [],
        "failed_indices": [],
        "in_progress": {},
        "scenario_results": {},
        "scheduling_strategy": "dynamic_scenario_queue_longest_first",
        "exhausted": False,
        "updated_at_epoch_s": time.time(),
    })

    processes = []
    worker_records = []
    for worker_index in range(worker_count):
        worker_output_name = f"{ARGS.output_name}/workers/worker_{worker_index:03d}"
        child_argv = _replace_cli_option(base_argv, "--worker-index", worker_index)
        child_argv = _replace_cli_option(child_argv, "--output-name", worker_output_name)
        progress_path = (workers_root / f"worker_{worker_index:03d}" / "worker_progress.json").resolve()
        child_argv = _replace_cli_option(child_argv, "--worker-progress-file", progress_path)
        child_argv = _replace_cli_option(child_argv, "--task-queue-file", task_queue_path)
        # Multiple live tqdm bars are unreadable in redirected logs.
        if "--no-progress" not in child_argv:
            child_argv.append("--no-progress")
        command = [sys.executable, str(Path(sys.argv[0]).resolve()), *child_argv]
        log_path = logs_root / f"worker_{worker_index:03d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, env=env
        )
        processes.append((worker_index, process, log_handle, log_path, command))

    print(
        "Scheduling: dynamic one-scenario queue, longest scenarios first "
        "(LPT)"
    )

    # Show one clean coordinator-level progress line. Worker tqdm bars remain
    # disabled because their detailed output is redirected to separate logs.
    #
    # Avg/scenario is the real mean worker time of completed scenarios.
    # ETA and estimated total use observed coordinator wall-clock throughput,
    # which is the relevant measure for a parallel run.
    last_display = None
    last_display_width = 0
    while True:
        exited_workers = sum(
            process.poll() is not None
            for _, process, _, _, _ in processes
        )
        successful_workers = sum(
            process.poll() == 0
            for _, process, _, _, _ in processes
        )
        failed_workers_now = sum(
            process.poll() is not None and process.poll() != 0
            for _, process, _, _, _ in processes
        )

        queue_state = _read_worker_progress(task_queue_path)
        completed_scenarios = len(queue_state.get("completed_indices", []))
        failed_scenarios = len(queue_state.get("failed_indices", []))
        known_total = int(
            queue_state.get("total_scenarios")
            or ARGS.max_scenarios
            or 0
        )
        active_workers = sum(
            process.poll() is None
            for _, process, _, _, _ in processes
        )
        claimed_scenarios = int(queue_state.get("claims", 0) or 0)
        in_progress_scenarios = len(queue_state.get("in_progress", {}) or {})
        pending_unclaimed = (
            max(0, known_total - claimed_scenarios)
            if known_total > 0
            else None
        )

        scenario_results = queue_state.get("scenario_results", {}) or {}
        scenario_elapsed_values = []
        for result in scenario_results.values():
            if result.get("status") != "completed":
                continue
            try:
                elapsed_value = float(result.get("elapsed_s"))
            except (TypeError, ValueError):
                continue
            if elapsed_value >= 0:
                scenario_elapsed_values.append(elapsed_value)

        average_scenario_elapsed_s = (
            sum(scenario_elapsed_values) / len(scenario_elapsed_values)
            if scenario_elapsed_values
            else None
        )

        coordinator_elapsed_s = max(0.0, time.time() - coordinator_start)
        finished_scenarios = completed_scenarios + failed_scenarios
        remaining_scenarios = (
            max(0, known_total - finished_scenarios)
            if known_total > 0
            else None
        )

        effective_scenarios_per_second = (
            finished_scenarios / coordinator_elapsed_s
            if finished_scenarios > 0 and coordinator_elapsed_s > 0
            else 0.0
        )
        estimated_remaining_s = (
            remaining_scenarios / effective_scenarios_per_second
            if (
                remaining_scenarios is not None
                and effective_scenarios_per_second > 0
            )
            else None
        )
        estimated_total_s = (
            coordinator_elapsed_s + estimated_remaining_s
            if estimated_remaining_s is not None
            else None
        )

        total_text = str(known_total) if known_total > 0 else "?"
        average_text = (
            _format_duration(average_scenario_elapsed_s)
            if average_scenario_elapsed_s is not None
            else "calculating..."
        )
        eta_text = (
            _format_duration(estimated_remaining_s)
            if estimated_remaining_s is not None
            else "calculating..."
        )
        estimated_total_text = (
            _format_duration(estimated_total_s)
            if estimated_total_s is not None
            else "calculating..."
        )

        # Include elapsed whole seconds so the display refreshes even while
        # no scenario finishes for a while.
        display = (
            exited_workers,
            successful_workers,
            failed_workers_now,
            completed_scenarios,
            failed_scenarios,
            total_text,
            active_workers,
            pending_unclaimed,
            in_progress_scenarios,
            int(coordinator_elapsed_s),
            (
                int(round(average_scenario_elapsed_s))
                if average_scenario_elapsed_s is not None
                else None
            ),
            (
                int(round(estimated_remaining_s))
                if estimated_remaining_s is not None
                else None
            ),
        )

        if display != last_display:
            message = (
                f"Workers exited: {exited_workers}/{worker_count} "
                f"(ok {successful_workers}, failed {failed_workers_now}) | "
                f"Scenarios: {completed_scenarios}/{total_text} completed, "
                f"{failed_scenarios} failed | "
                f"Active workers: {active_workers} | "
                f"Pending: {pending_unclaimed if pending_unclaimed is not None else '?'} | "
                f"In progress: {in_progress_scenarios} | "
                f"Avg/scenario: {average_text} | "
                f"Elapsed: {_format_duration(coordinator_elapsed_s)} | "
                f"ETA: {eta_text} | "
                f"Est. total: {estimated_total_text}"
            )
            print(
                "\r" + message.ljust(last_display_width),
                end="",
                flush=True,
            )
            last_display_width = max(last_display_width, len(message))
            last_display = display

        if exited_workers == worker_count:
            break
        time.sleep(0.5)
    print()

    failed = []
    for worker_index, process, log_handle, log_path, command in processes:
        return_code = process.wait()
        log_handle.close()
        worker_dir = workers_root / f"worker_{worker_index:03d}"
        summary_path = worker_dir / "run_summary.json"
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record = {
            "worker_index": worker_index,
            "return_code": return_code,
            "output_directory": str(worker_dir),
            "log": str(log_path),
            "summary": summary,
            "progress": _read_worker_progress(worker_dir / "worker_progress.json"),
        }
        worker_records.append(record)
        if return_code != 0:
            failed.append(record)

    # Export one timing record per claimed scenario. This provides the exact
    # scenario-level processing times behind the console average.
    final_queue_state = _read_worker_progress(task_queue_path)
    scenario_timing_rows = []
    for scenario_index_text, result in sorted(
        (final_queue_state.get("scenario_results", {}) or {}).items(),
        key=lambda item: int(item[0]),
    ):
        scenario_timing_rows.append({
            "scenario_index": int(scenario_index_text),
            "scenario_token": result.get("scenario_token"),
            "status": result.get("status"),
            "worker_index": result.get("worker_index"),
            "elapsed_s": result.get("elapsed_s"),
            "error_type": result.get("error_type"),
            "error": result.get("error"),
        })
    pd.DataFrame(
        scenario_timing_rows,
        columns=[
            "scenario_index",
            "scenario_token",
            "status",
            "worker_index",
            "elapsed_s",
            "error_type",
            "error",
        ],
    ).to_csv(OUTPUT_DIR / "scenario_processing_times.csv", index=False)

    completed_timing_values = []
    for row in scenario_timing_rows:
        if row.get("status") != "completed":
            continue
        try:
            elapsed_value = float(row.get("elapsed_s"))
        except (TypeError, ValueError):
            continue
        if elapsed_value >= 0:
            completed_timing_values.append(elapsed_value)

    aggregate_keys = (
        "scenarios_processed", "batches", "assertions_written", "native_assertions",
        "measurement_assertions", "positive_semantic_assertions", "errors",
        "ego_agent_candidates_evaluated", "ego_agent_pairs_selected",
        "ego_agent_pairs_rejected", "critical_ego_agent_pairs_selected",
        "agent_agent_possible_pairs", "agent_agent_candidates_after_prefilter",
        "agent_agent_pairs_fully_evaluated", "agent_agent_pairs_selected",
        "agent_agent_pairs_rejected", "critical_agent_agent_pairs_selected",
        "agent_agent_pairs_removed_by_cap",
    )
    aggregate = {key: 0 for key in aggregate_keys}
    for record in worker_records:
        for key in aggregate_keys:
            aggregate[key] += int((record.get("summary") or {}).get(key, 0))
    aggregate.update({
        "parallel": True,
        "scheduling": "dynamic_scenario_queue_longest_first",
        "num_workers": worker_count,
        "failed_workers": len(failed),
        "elapsed_s": round(time.time() - coordinator_start, 2),
        "average_completed_scenario_elapsed_s": (
            round(
                sum(completed_timing_values) / len(completed_timing_values),
                3,
            )
            if completed_timing_values
            else None
        ),
        "effective_wall_clock_s_per_finished_scenario": (
            round(
                (time.time() - coordinator_start)
                / max(
                    1,
                    len(final_queue_state.get("completed_indices", []))
                    + len(final_queue_state.get("failed_indices", [])),
                ),
                3,
            )
            if (
                len(final_queue_state.get("completed_indices", []))
                + len(final_queue_state.get("failed_indices", []))
            ) > 0
            else None
        ),
        "scenario_processing_times_file": str(
            OUTPUT_DIR / "scenario_processing_times.csv"
        ),
        "worker_outputs": [record["output_directory"] for record in worker_records],
        "worker_logs": [record["log"] for record in worker_records],
    })
    # Merge the small worker summaries into one convenient top-level view.
    predicate_totals = Counter()
    category_totals = Counter()
    first_successful_worker_dir = None
    first_successful_summary = None
    for record in worker_records:
        if record.get("return_code") != 0:
            continue
        worker_dir = Path(record["output_directory"])
        if first_successful_worker_dir is None:
            first_successful_worker_dir = worker_dir
            first_successful_summary = record.get("summary") or {}
        predicate_counts_path = worker_dir / "predicate_counts.csv"
        if predicate_counts_path.exists() and predicate_counts_path.stat().st_size > 0:
            try:
                frame = pd.read_csv(predicate_counts_path)
                for row in frame.to_dict(orient="records"):
                    predicate_totals[str(row.get("predicate_id"))] += int(
                        row.get("count", 0)
                    )
            except pd.errors.EmptyDataError:
                pass
        
        category_counts_path = worker_dir / "category_counts.csv"
        if category_counts_path.exists() and category_counts_path.stat().st_size > 0:
            try:
                frame = pd.read_csv(category_counts_path)
                for row in frame.to_dict(orient="records"):
                    category_totals[str(row.get("category"))] += int(
                        row.get("count", 0)
                    )
            except pd.errors.EmptyDataError:
                pass

    pd.DataFrame([
        {"predicate_id": pid, "count": count}
        for pid, count in predicate_totals.most_common()
    ]).to_csv(OUTPUT_DIR / "predicate_counts.csv", index=False)
    pd.DataFrame([
        {"category": category, "count": count}
        for category, count in category_totals.most_common()
    ]).to_csv(OUTPUT_DIR / "category_counts.csv", index=False)

    if first_successful_summary:
        for key in (
            "requested_categories", "compute_categories", "written_categories",
            "pair_directions", "pair_selection_by_direction", "map_cache",
            "proximal_map_cache_resolution_m", "parquet_compression",
            "compact_parquet", "write_diagnostics",
            "observed_future_labels_enabled", "intent_candidates_enabled",
        ):
            if key in first_successful_summary:
                aggregate[key] = first_successful_summary[key]

    # Keep one copy of category/rule/threshold metadata at the parallel-run root.
    if first_successful_worker_dir is not None:
        for name in (
            "predicate_definitions.csv",
            "rule_definitions.json",
            "active_categories.json",
            "thresholds_used.json",
        ):
            src = first_successful_worker_dir / name
            dst = OUTPUT_DIR / name
            if src.exists():
                dst.write_bytes(src.read_bytes())

    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "parallel_worker_manifest.json").write_text(
        json.dumps(worker_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print("Parallel output directory:", OUTPUT_DIR)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {worker_count} workers failed. "
            f"See {OUTPUT_DIR / 'parallel_worker_manifest.json'} and worker_logs/."
        )


def _main_impl():
    if ARGS.num_workers > 1 and ARGS.worker_count == 1:
        _run_parallel_coordinator()
        return

    config = {
        "dataset_root": str(DATASET_ROOT),
        "map_root": str(MAP_ROOT),
        "map_version": ARGS.map_version,
        "sample_interval_s": ARGS.sample_interval_s,
        "max_scenarios": ARGS.max_scenarios,
        "verbose": True,
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps({**vars(ARGS), **config}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "thresholds_used.json").write_text(
        json.dumps({
            "profile": "route_map_relevance_v6",
            "semantic_built_in_defaults": thresholds_as_dict(),
            "relevance_built_in_defaults": relevance_thresholds_as_dict(),
            "effective_semantic_thresholds": {
                key: value for key, value in vars(ARGS).items() if key in thresholds_as_dict()
            },
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Persist only definitions/rules that belong to categories actually written.
    # Internal dependency definitions remain available in memory for computation.
    written_predicates = [
        predicate for predicate in PREDICATES
        if predicate.category in WRITE_CATEGORIES
    ]
    definition_records = []
    for predicate in written_predicates:
        record = predicate.model_dump()
        record["implementation_status"] = _definition_status(predicate.predicate_id)
        record["positive_only_semantics"] = predicate.predicate_id in SEMANTIC_PREDICATE_IDS
        definition_records.append(record)
    pd.DataFrame(definition_records).to_csv(OUTPUT_DIR / "predicate_definitions.csv", index=False)

    written_rule_ids = {
        predicate.rule_id for predicate in written_predicates
        if getattr(predicate, "rule_id", None)
    }
    pd.DataFrame([
        rule.model_dump() for rule in RULES
        if rule.rule_id in written_rule_ids
    ]).to_json(
        OUTPUT_DIR / "rule_definitions.json", orient="records", indent=2, force_ascii=False
    )
    (OUTPUT_DIR / "active_categories.json").write_text(
        json.dumps({
            "requested": sorted(REQUESTED_CATEGORIES),
            "compute": sorted(COMPUTE_CATEGORIES),
            "write": sorted(WRITE_CATEGORIES),
            "write_dependencies": bool(ARGS.include_category_dependencies),
        }, indent=2), encoding="utf-8"
    )

    report_paths = [OUTPUT_DIR / "predicate_truth_counts.csv"]
    if ARGS.write_diagnostics:
        report_paths.extend([
            OUTPUT_DIR / "semantic_omissions_by_reason.csv",
            OUTPUT_DIR / "temporal_continuity_violations.csv",
            OUTPUT_DIR / "relevance_audit.csv",
            OUTPUT_DIR / "agent_agent_relevance_audit.csv",
        ])
    for path in report_paths:
        if path.exists():
            path.unlink()

    start = time.time()
    scenarios = build_mini_scenarios(config)
    scenarios = filter_scenarios_from_yaml(
        scenarios, ARGS.scenario_filter_yaml, ARGS.filter_random_seed
    )
    if ARGS.max_scenarios is not None:
        scenarios = scenarios[: ARGS.max_scenarios]

    # Cost-aware dynamic scheduling: preserve the selected scenario set, but
    # process scenarios with the most expected frames first.  This is the
    # Longest Processing Time first (LPT) strategy and reduces the end-of-run
    # straggler tail without splitting temporal histories across workers.
    timing_hint_csv = os.environ.get("NUPLAN_SCENARIO_TIMING_HINT_CSV")
    scenarios, schedule_metadata, timing_hints = _order_scenarios_longest_first(
        scenarios,
        ARGS.max_frames_per_scenario,
        timing_csv_path=timing_hint_csv,
    )

    worker_progress_path = (
        Path(ARGS.worker_progress_file).resolve()
        if ARGS.worker_progress_file is not None
        else OUTPUT_DIR / "worker_progress.json"
    )
    task_queue_path = (
        Path(ARGS.task_queue_file).resolve()
        if ARGS.task_queue_file is not None
        else OUTPUT_DIR / "dynamic_task_queue.json"
    )
    _atomic_write_json(worker_progress_path, {
        "worker_index": int(ARGS.worker_index),
        "worker_count": int(ARGS.worker_count),
        "status": "running",
        "total_scenarios": len(scenarios),
        "completed_scenarios": 0,
        "current_scenario_token": None,
        "updated_at_epoch_s": time.time(),
    })

    schedule_by_index = {
        int(row["scheduled_index"]): row for row in schedule_metadata
    }
    catalog = pd.DataFrame([
        {
            "index": index,
            "scheduled_index": index,
            "original_index": schedule_by_index[index]["original_index"],
            "estimated_frames": schedule_by_index[index]["estimated_frames"],
            "predicted_elapsed_s": schedule_by_index[index]["predicted_elapsed_s"],
            "priority_source": schedule_by_index[index]["priority_source"],
            "scheduling_priority": index + 1,
            "token": str(scenario.token),
            "log_name": str(scenario.log_name),
            "scenario_type": str(scenario.scenario_type),
            "iterations": scenario.get_number_of_iterations(),
        }
        for index, scenario in enumerate(scenarios)
    ])
    catalog.to_csv(OUTPUT_DIR / "scenario_catalog.csv", index=False)
    (OUTPUT_DIR / "scheduling_strategy.json").write_text(
        json.dumps({
            "strategy": "dynamic_scenario_queue_longest_first",
            "queue_unit": "one complete scenario",
            "priority_proxy": (
                "previous timing hints, then effective frame count"
            ),
            "timing_hint_csv": timing_hints.get("path"),
            "exact_timing_hint_count": len(
                timing_hints.get("exact_token_elapsed_s", {})
            ),
            "scenario_type_timing_hint_count": len(
                timing_hints.get("scenario_type_elapsed_s", {})
            ),
            "max_frames_per_scenario": ARGS.max_frames_per_scenario,
            "temporal_history_split_across_workers": False,
            "scenario_count": len(scenarios),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    batch_assertions = []
    batch_stats = []
    batch_errors = []
    batch_aux = {
        "semantic_omissions": Counter(),
        "temporal_continuity_violations": [],
        "relevance_audit": [],
        "agent_agent_relevance_audit": [],
    }
    manifest = []
    global_counts = Counter()
    global_category_counts = Counter()
    global_truth_counts = Counter()
    global_semantic_omissions = Counter()
    global_temporal_continuity = []
    global_temporal_continuity_count = 0
    global_relevance_stats = Counter()
    global_agent_agent_stats = Counter()
    relevance_audit_path = OUTPUT_DIR / "relevance_audit.csv"
    agent_agent_relevance_audit_path = OUTPUT_DIR / "agent_agent_relevance_audit.csv"
    if ARGS.write_diagnostics:
        if relevance_audit_path.exists():
            relevance_audit_path.unlink()
        if agent_agent_relevance_audit_path.exists():
            agent_agent_relevance_audit_path.unlink()
    batch_index = 0

    local_completed = 0
    iterator = tqdm(
        desc=f"Worker {ARGS.worker_index} scenarios",
        unit="scenario",
        disable=ARGS.no_progress,
        dynamic_ncols=True,
    )
    while True:
        claimed_index = _claim_next_scenario_index(task_queue_path, len(scenarios), ARGS.worker_index)
        if claimed_index is None:
            break
        scenario = scenarios[claimed_index]
        scenario_number = local_completed + 1
        scenario_start = time.time()
        _atomic_write_json(worker_progress_path, {
            "worker_index": int(ARGS.worker_index),
            "worker_count": int(ARGS.worker_count),
            "status": "processing",
            "total_scenarios": len(scenarios),
            "completed_scenarios": scenario_number - 1,
            "current_scenario_number": scenario_number,
            "current_global_index": claimed_index,
            "last_claimed_global_index": claimed_index,
            "current_scenario_token": str(scenario.token),
            "current_frame": 0,
            "updated_at_epoch_s": time.time(),
        })
        state = {
            "entity_history": {},
            "pair_history": {},
            "snapshots": [],
            "frame_index": -1,
            "semantic_streaks": {},
            "semantic_omissions": Counter(),
            "temporal_continuity_violations": [],
            "relevance_audit": [],
            "agent_agent_relevance_audit": [],
        }
        scenario_assertion_count = 0
        processed_frames = 0
        error_start = len(batch_errors)

        for frame_index, frame in enumerate(iter_frames(scenario)):
            if ARGS.max_frames_per_scenario is not None and frame_index >= ARGS.max_frames_per_scenario:
                break
            state["frame_index"] = frame_index
            try:
                frame_assertions = filter_assertions_by_category(
                    extract_frame_predicates(frame, state)
                )
                batch_assertions.extend(frame_assertions)
                scenario_assertion_count += len(frame_assertions)
            except Exception as exc:
                batch_errors.append({
                    "scenario_token": str(scenario.token),
                    "frame_index": frame_index,
                    "timestamp_us": getattr(frame, "timestamp_us", None),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                if ARGS.fail_on_frame_error:
                    raise
            processed_frames += 1
            # Keep the scenario progress bar visibly alive while a scenario is
            # being expanded. The scenario counter itself is incremented only
            # after the full scenario is completed, but the postfix shows the
            # current frame and accumulated assertion count in real time.
            iterator.set_postfix(
                frame=processed_frames,
                assertions=scenario_assertion_count,
                refresh=True,
            )
            if True:
                _atomic_write_json(worker_progress_path, {
                    "worker_index": int(ARGS.worker_index),
                    "worker_count": int(ARGS.worker_count),
                    "status": "processing",
                    "total_scenarios": len(scenarios),
                    "completed_scenarios": scenario_number - 1,
                    "current_scenario_number": scenario_number,
                    "current_scenario_token": str(scenario.token),
                    "current_frame": processed_frames,
                    "updated_at_epoch_s": time.time(),
                })

        try:
            scenario_level_assertions = filter_assertions_by_category(
                derive_scenario_level_predicates(state)
            )
            batch_assertions.extend(scenario_level_assertions)
            scenario_assertion_count += len(scenario_level_assertions)
        except Exception as exc:
            batch_errors.append({
                "scenario_token": str(scenario.token),
                "frame_index": None,
                "timestamp_us": None,
                "error_type": type(exc).__name__,
                "error": f"scenario-level derivation failed: {exc}",
            })
            if ARGS.fail_on_frame_error:
                raise

        batch_aux["semantic_omissions"].update(state.get("semantic_omissions", {}))
        batch_aux["temporal_continuity_violations"].extend(state.get("temporal_continuity_violations", []))
        batch_aux["relevance_audit"].extend(state.get("relevance_audit", []))
        batch_aux["agent_agent_relevance_audit"].extend(state.get("agent_agent_relevance_audit", []))
        global_agent_agent_stats.update(state.get("agent_agent_relevance_stats", {}))

        selected_relevance = [row for row in state.get("relevance_audit", []) if row.get("selected")]
        batch_stats.append({
            "scenario_token": str(scenario.token),
            "scenario_type": str(scenario.scenario_type),
            "log_name": str(scenario.log_name),
            "processed_frames": processed_frames,
            "assertion_count": scenario_assertion_count,
            "semantic_omission_count": sum(state.get("semantic_omissions", {}).values()),
            "temporal_gap_resets": len(state.get("temporal_continuity_violations", [])),
            "ego_agent_candidates": len(state.get("relevance_audit", [])),
            "ego_agent_pairs_selected": len(selected_relevance),
            "error_count": len(batch_errors) - error_start,
            "elapsed_s": round(time.time() - scenario_start, 2),
        })

        _mark_scenario_done(
            task_queue_path,
            claimed_index,
            ARGS.worker_index,
            time.time() - scenario_start,
            scenario_token=scenario.token,
        )

        if scenario_number % ARGS.batch_size_scenarios == 0:
            batch_index += 1
            result = write_batch(
                batch_index, batch_assertions, batch_stats, batch_errors, auxiliary=batch_aux
            )
            global_counts.update(result.pop("predicate_counts", {}))
            global_category_counts.update(result.pop("category_counts", {}))
            global_truth_counts.update(result.pop("truth_counts"))
            global_semantic_omissions.update(result.pop("semantic_omissions"))
            if ARGS.write_diagnostics:
                global_temporal_continuity.extend(
                    result.pop("temporal_continuity_violations", [])
                )
                relevance_rows = result.pop("relevance_audit", [])
                if relevance_rows:
                    pd.DataFrame(relevance_rows).to_csv(
                        relevance_audit_path, mode="a", index=False,
                        header=not relevance_audit_path.exists(),
                    )
                    global_relevance_stats["candidates"] += len(relevance_rows)
                    global_relevance_stats["selected"] += sum(bool(row.get("selected")) for row in relevance_rows)
                    global_relevance_stats["rejected"] += sum(not bool(row.get("selected")) for row in relevance_rows)
                    global_relevance_stats["critical_selected"] += sum(
                        bool(row.get("selected")) and bool(row.get("critical")) for row in relevance_rows
                    )
                agent_agent_rows = result.pop("agent_agent_relevance_audit", [])
                if agent_agent_rows:
                    pd.DataFrame(agent_agent_rows).to_csv(
                        agent_agent_relevance_audit_path, mode="a", index=False,
                        header=not agent_agent_relevance_audit_path.exists(),
                    )
            else:
                global_temporal_continuity_count += int(
                    result.pop("temporal_continuity_violation_count", 0)
                )
                global_relevance_stats.update(result.pop("relevance_stats", {}))
            manifest.append(result)
            batch_assertions.clear()
            batch_stats.clear()
            batch_errors.clear()
            batch_aux = {
                "semantic_omissions": Counter(),
                "temporal_continuity_violations": [],
                "relevance_audit": [],
                "agent_agent_relevance_audit": [],
            }
            gc.collect()

        local_completed = scenario_number
        iterator.update(1)
        _atomic_write_json(worker_progress_path, {
            "worker_index": int(ARGS.worker_index),
            "worker_count": int(ARGS.worker_count),
            "status": "running",
            "total_scenarios": len(scenarios),
            "completed_scenarios": local_completed,
            "last_claimed_global_index": claimed_index,
            "current_scenario_token": None,
            "current_frame": 0,
            "updated_at_epoch_s": time.time(),
        })

    iterator.close()

    # Flush the final partial worker-local batch after the shared queue is empty.
    if batch_assertions or batch_stats or batch_errors or any(batch_aux.values()):
        batch_index += 1
        result = write_batch(
            batch_index, batch_assertions, batch_stats, batch_errors, auxiliary=batch_aux
        )
        global_counts.update(result.pop("predicate_counts", {}))
        global_category_counts.update(result.pop("category_counts", {}))
        global_truth_counts.update(result.pop("truth_counts"))
        global_semantic_omissions.update(result.pop("semantic_omissions"))
        if ARGS.write_diagnostics:
            global_temporal_continuity.extend(
                result.pop("temporal_continuity_violations", [])
            )
            relevance_rows = result.pop("relevance_audit", [])
            if relevance_rows:
                pd.DataFrame(relevance_rows).to_csv(
                    relevance_audit_path, mode="a", index=False,
                    header=not relevance_audit_path.exists(),
                )
                global_relevance_stats["candidates"] += len(relevance_rows)
                global_relevance_stats["selected"] += sum(bool(row.get("selected")) for row in relevance_rows)
                global_relevance_stats["rejected"] += sum(not bool(row.get("selected")) for row in relevance_rows)
                global_relevance_stats["critical_selected"] += sum(
                    bool(row.get("selected")) and bool(row.get("critical")) for row in relevance_rows
                )
            agent_agent_rows = result.pop("agent_agent_relevance_audit", [])
            if agent_agent_rows:
                pd.DataFrame(agent_agent_rows).to_csv(
                    agent_agent_relevance_audit_path, mode="a", index=False,
                    header=not agent_agent_relevance_audit_path.exists(),
                )
        else:
            global_temporal_continuity_count += int(
                result.pop("temporal_continuity_violation_count", 0)
            )
            global_relevance_stats.update(result.pop("relevance_stats", {}))
        manifest.append(result)
        batch_assertions.clear()
        batch_stats.clear()
        batch_errors.clear()

    _atomic_write_json(worker_progress_path, {
        "worker_index": int(ARGS.worker_index),
        "worker_count": int(ARGS.worker_count),
        "status": "finalizing",
        "total_scenarios": len(scenarios),
        "completed_scenarios": local_completed,
        "current_scenario_token": None,
        "updated_at_epoch_s": time.time(),
    })

    pd.DataFrame(
        [
            {"predicate_id": pid, "count": count}
            for pid, count in global_counts.most_common()
        ],
        columns=["predicate_id", "count"],
    ).to_csv(
        OUTPUT_DIR / "predicate_counts.csv",
        index=False,
    )
    
    pd.DataFrame(
        [
            {"category": category, "count": count}
            for category, count in global_category_counts.most_common()
        ],
        columns=["category", "count"],
    ).to_csv(
        OUTPUT_DIR / "category_counts.csv",
        index=False,
    )

    with (OUTPUT_DIR / "batch_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if ARGS.write_diagnostics:
        _write_csv_records(
            OUTPUT_DIR / "temporal_continuity_violations.csv",
            global_temporal_continuity,
            ["track_key", "previous_timestamp_us", "current_timestamp_us", "previous_frame_index", "current_frame_index", "action"],
        )
        omission_rows = [
            {"predicate_id": pid, "reason": reason, "count": count}
            for (pid, reason), count in global_semantic_omissions.items()
        ]
        _write_csv_records(
            OUTPUT_DIR / "semantic_omissions_by_reason.csv", omission_rows,
            ["predicate_id", "reason", "count"],
        )
    truth_rows = []
    for key, count in global_truth_counts.items():
        if key.startswith("predicate::"):
            truth_rows.append({"scope": "predicate", "name": key.split("::", 1)[1], "count": count})
        else:
            truth_rows.append({"scope": "summary", "name": key, "count": count})
    _write_csv_records(
        OUTPUT_DIR / "predicate_truth_counts.csv", truth_rows,
        ["scope", "name", "count"],
    )

    if ARGS.write_diagnostics and not relevance_audit_path.exists():
        _write_csv_records(
            relevance_audit_path, [],
            [
                "scenario_token", "timestamp_us", "track_token", "selected",
                "mandatory_selection", "relevant_before_cap", "critical",
                "score", "primary_reason", "reasons",
                "generator_selection_reasons", "generator_within_distance",
                "generator_inside_forward_corridor", "generator_same_lane",
                "generator_predicted_path_intersection",
                "ego_longitudinal_m", "ego_lateral_m",
                "prediction_cpa_time_s", "prediction_cpa_center_distance_m",
                "prediction_cpa_clearance_m", "center_distance_m",
                "free_space_distance_m", "signed_path_distance_m",
                "cpa_time_s", "cpa_clearance_m", "directional_allowed",
                "directional_omission_reason",
            ],
        )

    summary = {
        "scenarios_processed": local_completed,
        "parallel_worker_index": ARGS.worker_index,
        "parallel_worker_count": ARGS.worker_count,
        "scheduling": (
            "dynamic_scenario_queue_longest_first"
            if ARGS.worker_count > 1
            else "single_worker"
        ),
        "batches": len(manifest),
        "assertions_written": sum(item["assertions"] for item in manifest),
        "native_assertions": global_truth_counts.get("native_assertions", 0),
        "measurement_assertions": global_truth_counts.get("measurement_assertions", 0),
        "positive_semantic_assertions": global_truth_counts.get("positive_semantic_assertions", 0),
        "semantic_omissions_due_to_insufficient_or_failed_evidence": sum(global_semantic_omissions.values()),
        "errors": sum(item["errors"] for item in manifest),
        "elapsed_s": round(time.time() - start, 2),
        "agent_agent_enabled": "agent-agent" in ARGS.pair_directions,
        "pair_selection": ARGS.pair_selection,
        "pair_selection_by_direction": dict(ARGS.pair_selection_by_direction),
        "pair_directions": list(ARGS.pair_direction_order),
        "requested_categories": sorted(REQUESTED_CATEGORIES),
        "compute_categories": sorted(COMPUTE_CATEGORIES),
        "written_categories": sorted(WRITE_CATEGORIES),
        "map_cache": bool(ARGS.map_cache),
        "proximal_map_cache_resolution_m": ARGS.proximal_map_cache_resolution_m,
        "parquet_rows_per_part": ARGS.parquet_rows_per_part,
        "parquet_compression": ARGS.parquet_compression,
        "compact_parquet": bool(ARGS.compact_parquet),
        "write_diagnostics": bool(ARGS.write_diagnostics),
        "temporal_continuity_violation_count": (
            len(global_temporal_continuity)
            if ARGS.write_diagnostics
            else global_temporal_continuity_count
        ),
        "ego_agent_candidates_evaluated": global_relevance_stats["candidates"],
        "ego_agent_pairs_selected": global_relevance_stats["selected"],
        "ego_agent_pairs_rejected": global_relevance_stats["rejected"],
        "critical_ego_agent_pairs_selected": global_relevance_stats["critical_selected"],
        "agent_agent_possible_pairs": int(global_agent_agent_stats["possible_pairs"]),
        "agent_agent_candidates_after_prefilter": int(global_agent_agent_stats["candidates_after_prefilter"]),
        "agent_agent_pairs_fully_evaluated": int(global_agent_agent_stats["fully_evaluated"]),
        "agent_agent_pairs_selected": int(global_agent_agent_stats["selected"]),
        "agent_agent_pairs_rejected": int(global_agent_agent_stats["rejected"]),
        "critical_agent_agent_pairs_selected": int(global_agent_agent_stats["critical_selected"]),
        "agent_agent_pairs_removed_by_cap": int(global_agent_agent_stats["removed_by_cap"]),
        "positive_only_semantics": True,
        "threshold_profile": "expanded_ego_relevance_v8",
        "spatial_classifier": "map_topology_plus_local_travel_frame_consistency_v8",
        "agent_agent_relevance_selection": {
            "candidate_radius_m": ARGS.agent_agent_candidate_radius_m,
            "max_neighbors_per_agent": ARGS.agent_agent_max_neighbors_per_agent,
            "max_pairs_per_frame": ARGS.agent_agent_max_pairs_per_frame,
        },
        "expanded_relevance_selection": {
            "distance_threshold_m": ARGS.relevance_distance_threshold_m,
            "include_within_distance": ARGS.relevance_include_within_distance,
            "include_forward_corridor": ARGS.relevance_include_forward_corridor,
            "forward_corridor_length_m": ARGS.relevance_forward_corridor_length_m,
            "forward_corridor_half_width_m": ARGS.relevance_forward_corridor_half_width_m,
            "include_same_lane": ARGS.relevance_include_same_lane,
            "include_predicted_path_intersection": ARGS.relevance_include_predicted_path_intersection,
            "prediction_horizon_s": ARGS.relevance_prediction_horizon_s,
            "path_intersection_clearance_m": ARGS.relevance_path_intersection_clearance_m,
        },
        "semantic_thresholds_applied": bool(ARGS.derive_semantic_predicates),
        "observed_future_labels_enabled": bool(ARGS.derive_observed_future_labels),
        "intent_candidates_enabled": bool(ARGS.derive_intent_candidates),
        "maneuver_predicates_enabled": bool(
            ARGS.derive_semantic_predicates and "maneuver" in COMPUTE_CATEGORIES
        ),
        "traffic_control_predicates_enabled": bool(
            ARGS.derive_semantic_predicates and "traffic_light" in COMPUTE_CATEGORIES
        ),
        "geometric_visibility_predicates_enabled": bool(
            ARGS.derive_semantic_predicates and "visibility" in COMPUTE_CATEGORIES
        ),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _atomic_write_json(worker_progress_path, {
        "worker_index": int(ARGS.worker_index),
        "worker_count": int(ARGS.worker_count),
        "status": "completed",
        "total_scenarios": len(scenarios),
        "completed_scenarios": local_completed,
        "current_scenario_token": None,
        "updated_at_epoch_s": time.time(),
    })
    print(json.dumps(summary, indent=2))
    print("Output directory:", OUTPUT_DIR)



def main():
    """Run the pipeline and always publish terminal worker state on failure."""
    try:
        return _main_impl()
    except Exception as exc:
        # Coordinator failures do not have a worker progress file. Worker
        # failures do, and publishing them prevents stale "active" states.
        try:
            if int(getattr(ARGS, "worker_count", 1)) > 1:
                progress_path = (
                    Path(ARGS.worker_progress_file).resolve()
                    if ARGS.worker_progress_file is not None
                    else OUTPUT_DIR / "worker_progress.json"
                )
                previous = _read_worker_progress(progress_path)
                claimed_index = previous.get("last_claimed_global_index")
                if claimed_index is None:
                    claimed_index = previous.get("current_global_index")
                _atomic_write_json(progress_path, {
                    **previous,
                    "worker_index": int(ARGS.worker_index),
                    "worker_count": int(ARGS.worker_count),
                    "status": "failed",
                    "current_frame": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at_epoch_s": time.time(),
                })
                if claimed_index is not None and ARGS.task_queue_file is not None:
                    _mark_scenario_failed(
                        Path(ARGS.task_queue_file).resolve(),
                        int(claimed_index),
                        int(ARGS.worker_index),
                        exc,
                        scenario_token=previous.get("current_scenario_token"),
                    )
        except Exception:
            pass
        raise