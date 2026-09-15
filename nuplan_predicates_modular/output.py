"""Serialization, RDF construction, and batch reports."""
from .base import *

def _assertion_is_positive_semantic(assertion) -> bool:
    if assertion.predicate_id not in SEMANTIC_PREDICATE_IDS:
        return False
    if assertion.value_type == ValueType.BOOLEAN:
        return parse_boolean(assertion.value) is True
    return True


def build_rdf_graph(assertions):
    graph = Graph()
    graph.bind("np", NP)
    graph.bind("prov", PROV)
    graph.bind("owl", OWL)

    for definition in PREDICATES:
        uri = predicate_uri(definition.predicate_id)
        property_type = OWL.ObjectProperty if definition.value_type == ValueType.ENTITY else OWL.DatatypeProperty
        graph.add((uri, RDF.type, property_type))
        graph.add((uri, RDFS.label, Literal(definition.label)))
        graph.add((uri, RDFS.comment, Literal(definition.description)))
        if getattr(definition, "symmetric", False):
            graph.add((uri, RDF.type, OWL.SymmetricProperty))
        inverse_of = getattr(definition, "inverse_of", None)
        if inverse_of:
            graph.add((uri, OWL.inverseOf, predicate_uri(inverse_of)))
        units = getattr(definition, "units", None)
        if units:
            graph.add((uri, NP.units, Literal(units)))
        for subject_type in getattr(definition, "subject_types", []) or []:
            graph.add((uri, RDFS.domain, NP[f"class/{subject_type}"]))
        for object_type in getattr(definition, "object_types", []) or []:
            graph.add((uri, RDFS.range, NP[f"class/{object_type}"]))

    for rule in RULES:
        uri = NP[f"rule/{rule.rule_id}"]
        graph.add((uri, RDF.type, NP.Rule))
        graph.add((uri, RDFS.label, Literal(rule.name)))
        graph.add((uri, NP.expression, Literal(rule.expression)))
        graph.add((uri, NP.ruleVersion, Literal(rule.version)))

    for assertion in assertions:
        # False semantic booleans are never promoted to direct facts.
        if assertion.predicate_id in SEMANTIC_PREDICATE_IDS and assertion.value_type == ValueType.BOOLEAN:
            if parse_boolean(assertion.value) is not True:
                continue
        assertion_uri = NP[f"assertion/{assertion.assertion_id}"]
        subject_uri = entity_uri(assertion.subject_id)
        predicate = predicate_uri(assertion.predicate_id)
        obj = entity_uri(assertion.object_id) if assertion.object_id is not None else Literal(assertion.value)
        graph.add((subject_uri, predicate, obj))
        graph.add((assertion_uri, RDF.type, NP.PredicateAssertion))
        graph.add((assertion_uri, RDF.subject, subject_uri))
        graph.add((assertion_uri, RDF.predicate, predicate))
        graph.add((assertion_uri, RDF.object, obj))
        graph.add((assertion_uri, NP.assertionKind, Literal(str(assertion.assertion_kind))))
        graph.add((assertion_uri, NP.validTimeUs, Literal(int(assertion.valid_time_us))))
        if assertion.end_time_us is not None:
            graph.add((assertion_uri, NP.endTimeUs, Literal(int(assertion.end_time_us))))
        graph.add((assertion_uri, PROV.wasDerivedFrom, Literal(assertion.provenance.scenario_token)))
        if assertion.provenance.rule_id:
            graph.add((assertion_uri, PROV.wasGeneratedBy, NP[f"rule/{assertion.provenance.rule_id}"]))
        if assertion.evidence:
            graph.add((assertion_uri, NP.evidenceJson, Literal(json.dumps(assertion.evidence, ensure_ascii=False, sort_keys=True, default=str))))
            if isinstance(assertion.evidence, dict):
                if "evidence_level" in assertion.evidence:
                    graph.add((assertion_uri, NP.evidenceLevel, Literal(assertion.evidence["evidence_level"])))
                if "confidence" in assertion.evidence:
                    graph.add((assertion_uri, NP.confidence, Literal(float(assertion.evidence["confidence"]))))
    return graph


def _truth_counts(assertions):
    counts = Counter()
    for assertion in assertions:
        if assertion.assertion_kind == NATIVE:
            counts["native_assertions"] += 1
        elif _assertion_is_positive_semantic(assertion):
            counts["positive_semantic_assertions"] += 1
        else:
            counts["measurement_assertions"] += 1
        if assertion.value_type == ValueType.BOOLEAN:
            parsed = parse_boolean(assertion.value)
            counts[f"boolean_{str(parsed).lower()}"] += 1
        counts[f"predicate::{assertion.predicate_id}"] += 1
    return counts


def _write_csv_records(path: Path, records: list[dict[str, Any]], columns: Optional[list[str]] = None):
    if records:
        pd.DataFrame(records).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False)


def write_batch(batch_index, assertions, scenario_stats, errors, auxiliary=None):
    auxiliary = auxiliary or {}
    batch_name = f"batch_{batch_index:06d}"

    # Final defensive persistence boundary: even if an upstream extraction path
    # accidentally returns helper/dependency assertions, only WRITE_CATEGORIES
    # may reach disk. This makes --categories strict at serialization time.
    assertions = filter_assertions_by_category(assertions)

    if ARGS.write_jsonl:
        path = BATCH_DIR / f"{batch_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for assertion in assertions:
                handle.write(assertion.model_dump_json() + "\n")

    if ARGS.write_parquet and assertions:
        # Avoid the large peak caused by flattening the full scenario and then
        # constructing one equally large DataFrame. Independent Parquet parts
        # are supported by the inspection notebook's batch_*.parquet loader.
        rows_per_part = int(ARGS.parquet_rows_per_part)
        compression = None if ARGS.parquet_compression == "none" else ARGS.parquet_compression
        if len(assertions) <= rows_per_part:
            pd.DataFrame([flatten_assertion(a) for a in assertions]).to_parquet(
                BATCH_DIR / f"{batch_name}.parquet", index=False, compression=compression
            )
        else:
            for part_index, start in enumerate(
                range(0, len(assertions), rows_per_part), start=1
            ):
                chunk = assertions[start : start + rows_per_part]
                chunk_path = BATCH_DIR / (
                    f"{batch_name}_part_{part_index:04d}.parquet"
                )
                pd.DataFrame([flatten_assertion(a) for a in chunk]).to_parquet(
                    chunk_path, index=False, compression=compression
                )
                del chunk

    if ARGS.write_rdf and assertions:
        graph = build_rdf_graph(assertions)
        graph.serialize(BATCH_DIR / f"{batch_name}.ttl", format="turtle")
        del graph

    pd.DataFrame(scenario_stats).to_csv(BATCH_DIR / f"{batch_name}_scenario_stats.csv", index=False)
    relevance_rows = list(auxiliary.get("relevance_audit", []))
    agent_agent_relevance_rows = list(auxiliary.get("agent_agent_relevance_audit", []))

    # Detailed audits are valuable for debugging but can be very large and are
    # not required for normal predicate consumption or video export.
    if getattr(ARGS, "write_diagnostics", False):
        _write_csv_records(
            BATCH_DIR / f"{batch_name}_agent_agent_relevance_audit.csv", agent_agent_relevance_rows,
            [
                "scenario_token", "timestamp_us", "subject_token", "object_token",
                "selected_before_cap", "selected", "removed_by_cap", "critical",
                "score", "primary_reason", "reasons", "center_distance_m",
                "free_space_distance_m", "signed_path_distance_m", "cpa_time_s",
                "cpa_clearance_m",
            ],
        )
        _write_csv_records(
            BATCH_DIR / f"{batch_name}_relevance_audit.csv", relevance_rows,
            [
                "scenario_token", "timestamp_us", "track_token", "selected",
                "relevant_before_cap", "critical", "score", "primary_reason",
                "reasons", "center_distance_m", "free_space_distance_m",
                "signed_path_distance_m", "cpa_time_s", "cpa_clearance_m",
                "directional_allowed", "directional_omission_reason",
            ],
        )

    # Keep error details only when something actually failed.
    if errors:
        with (BATCH_DIR / f"{batch_name}_errors.jsonl").open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    truth_counts = _truth_counts(assertions)
    predicate_counts = Counter(a.predicate_id for a in assertions)
    category_counts = Counter(assertion_category(a) for a in assertions)
    result = {
        "batch": batch_name,
        "assertions": len(assertions),
        "scenarios": len(scenario_stats),
        "errors": len(errors),
        "truth_counts": dict(truth_counts),
        "predicate_counts": dict(predicate_counts),
        "category_counts": dict(category_counts),
        "semantic_omissions": dict(auxiliary.get("semantic_omissions", {})),
    }
    if getattr(ARGS, "write_diagnostics", False):
        result["temporal_continuity_violations"] = list(auxiliary.get("temporal_continuity_violations", []))
        result["relevance_audit"] = list(auxiliary.get("relevance_audit", []))
        result["agent_agent_relevance_audit"] = list(auxiliary.get("agent_agent_relevance_audit", []))
    else:
        result["temporal_continuity_violation_count"] = len(
            auxiliary.get("temporal_continuity_violations", [])
        )
        # Preserve only aggregate selection statistics; the row-level audit is
        # intentionally not persisted in compact runs.
        relevance_rows = list(auxiliary.get("relevance_audit", []))
        result["relevance_stats"] = {
            "candidates": len(relevance_rows),
            "selected": sum(bool(row.get("selected")) for row in relevance_rows),
            "rejected": sum(not bool(row.get("selected")) for row in relevance_rows),
            "critical_selected": sum(
                bool(row.get("selected")) and bool(row.get("critical"))
                for row in relevance_rows
            ),
        }
    return result


def _definition_status(predicate_id: str) -> str:
    if predicate_id in {"np:projectsIntoCamera", "np:visibleInCamera", "np:visibleInLidar", "np:partiallyOccluded", "np:merges"}:
        return "catalog_only"
    if predicate_id == "np:beside":
        return "deprecated_nonexclusive_coarse_relation"
    if predicate_id.startswith("np:hasFuture") or predicate_id in {
        "np:hasAvailableFutureDuration", "np:hasFutureSampleCount", "np:hasFutureCoverageRatio",
        "np:hasInsufficientFutureEvidence",
    }:
        return "implemented_disabled_by_default_retrospective"
    if predicate_id in {"np:yieldingTo", "np:waitingFor", "np:creatingGapFor", "np:competingForGapWith"}:
        return "implemented_disabled_by_default_inferred"
    if predicate_id in SEMANTIC_PREDICATE_IDS:
        return "implemented_positive_only_enabled_by_default"
    return "implemented_measurement_or_native"
