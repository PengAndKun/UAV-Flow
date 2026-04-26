from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TARGET_CONDITIONED_STATES = {
    "target_house_not_in_view",
    "target_house_entry_visible",
    "target_house_entry_approachable",
    "target_house_entry_blocked",
    "non_target_house_entry_visible",
    "target_house_geometric_opening_needs_confirmation",
}

TARGET_CONDITIONED_SUBGOALS = {
    "reorient_to_target_house",
    "keep_search_target_house",
    "approach_target_entry",
    "align_target_entry",
    "detour_left_to_target_entry",
    "detour_right_to_target_entry",
    "cross_target_entry",
    "ignore_non_target_entry",
    "backoff_and_reobserve",
}

ACTION_HINTS = {"forward", "yaw_left", "yaw_right", "left", "right", "backward", "hold"}

ENTRY_ASSOCIATIONS = {
    "target_house_entry",
    "non_target_house_entry",
    "window_or_non_entry",
    "uncertain_entry",
    "no_entry",
}

MEMORY_DECISIONS = {
    "reuse_confirmed_best_entry",
    "update_best_entry",
    "continue_observing",
    "shift_search_sector",
    "reject_window",
    "reject_non_target_entry",
    "no_memory_available",
}

DOORLIKE_CLASSES = {"door", "open door", "open_door", "close door", "close_door", "closed door", "closed_door"}
WINDOW_CLASSES = {"window"}
APPROACH_SUBGOALS = {"approach_target_entry", "align_target_entry", "cross_target_entry"}

REQUIRED_FIELDS = [
    "target_conditioned_state",
    "target_conditioned_subgoal",
    "target_conditioned_action_hint",
    "target_candidate_id",
    "entry_association",
    "memory_decision",
    "confidence",
    "reason",
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_class_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def normalize_candidate_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "-1"}:
        return None
    return text


def round_float(value: Any, digits: int = 4) -> float:
    return round(safe_float(value), digits)


def compact_counter(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items()) if key}


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else {}
        except Exception:
            pass
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def load_label(label_path: Path, response_path: Optional[Path] = None) -> Tuple[Dict[str, Any], str]:
    if label_path.exists():
        payload = read_json(label_path)
        if payload:
            return payload, str(label_path)
    if response_path and response_path.exists():
        payload = read_json(response_path)
        if payload:
            for key in ("parsed", "label", "normalized_label"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value, str(response_path)
            content = payload.get("content") or payload.get("text") or payload.get("response")
            if isinstance(content, str):
                parsed = extract_json_object(content)
                if parsed:
                    return parsed, str(response_path)
        text = response_path.read_text(encoding="utf-8", errors="ignore")
        parsed = extract_json_object(text)
        if parsed:
            return parsed, str(response_path)
    return {}, ""


def get_structured_input(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    value = prompt_payload.get("structured_input", {})
    return value if isinstance(value, dict) else {}


def get_memory_evidence(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = get_structured_input(prompt_payload)
    value = structured.get("memory_evidence", {})
    return value if isinstance(value, dict) else {}


def get_depth_evidence(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = get_structured_input(prompt_payload)
    value = structured.get("depth_evidence", {})
    return value if isinstance(value, dict) else {}


def get_target_context(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = get_structured_input(prompt_payload)
    value = structured.get("target_context", {})
    return value if isinstance(value, dict) else {}


def get_rule_reference(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = get_structured_input(prompt_payload)
    value = structured.get("fusion_rule_reference", {})
    return value if isinstance(value, dict) else {}


def candidate_entries(prompt_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    memory = get_memory_evidence(prompt_payload)
    return [item for item in as_list(memory.get("candidate_entries")) if isinstance(item, dict)]


def yolo_detections(prompt_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    structured = get_structured_input(prompt_payload)
    yolo = structured.get("yolo_rgb_evidence", {}) if isinstance(structured.get("yolo_rgb_evidence"), dict) else {}
    return [item for item in as_list(yolo.get("detections")) if isinstance(item, dict)]


def candidate_map(prompt_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in yolo_detections(prompt_payload):
        cid = normalize_candidate_id(item.get("candidate_id"))
        if cid is not None:
            mapping.setdefault(cid, {}).update({"source": "yolo", **item})
    for item in candidate_entries(prompt_payload):
        cid = normalize_candidate_id(item.get("entry_id") or item.get("candidate_id"))
        if cid is not None:
            existing = mapping.get(cid, {})
            merged = {**existing, **item, "source": "memory"}
            mapping[cid] = merged
    memory = get_memory_evidence(prompt_payload)
    best = memory.get("best_entry", {}) if isinstance(memory.get("best_entry"), dict) else {}
    cid = normalize_candidate_id(best.get("entry_id") or best.get("candidate_id") or memory.get("best_entry_id"))
    if cid is not None:
        existing = mapping.get(cid, {})
        mapping[cid] = {**existing, **best, "source": "memory_best"}
    rule = get_rule_reference(prompt_payload)
    cid = normalize_candidate_id(rule.get("best_target_candidate_id"))
    if cid is not None:
        mapping.setdefault(cid, {"candidate_id": cid, "source": "fusion_rule"})
    return mapping


def best_memory_entry(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    memory = get_memory_evidence(prompt_payload)
    best = memory.get("best_entry", {}) if isinstance(memory.get("best_entry"), dict) else {}
    return best


def candidate_type(candidate: Dict[str, Any]) -> str:
    return normalize_class_name(
        candidate.get("entry_type")
        or candidate.get("semantic_class")
        or candidate.get("class_name")
        or candidate.get("type")
    )


def evidence_nonzero(entry: Dict[str, Any]) -> bool:
    evidence = entry.get("association_evidence", {}) if isinstance(entry.get("association_evidence"), dict) else {}
    for key in ("distance_score", "view_consistency_score", "appearance_score", "geometry_score"):
        if safe_float(evidence.get(key), 0.0) > 0.0:
            return True
    return False


def normalize_label(label: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_conditioned_state": normalize_token(label.get("target_conditioned_state")),
        "target_conditioned_subgoal": normalize_token(label.get("target_conditioned_subgoal")),
        "target_conditioned_action_hint": normalize_token(label.get("target_conditioned_action_hint")),
        "target_candidate_id": normalize_candidate_id(label.get("target_candidate_id")),
        "entry_association": normalize_token(label.get("entry_association")),
        "memory_decision": normalize_token(label.get("memory_decision")),
        "confidence": round_float(label.get("confidence"), 4),
        "reason": str(label.get("reason") or "").strip(),
    }


def fallback_label_from_prompt(prompt_payload: Dict[str, Any], *, reason: str = "") -> Dict[str, Any]:
    rule = get_rule_reference(prompt_payload)
    memory = get_memory_evidence(prompt_payload)
    best = best_memory_entry(prompt_payload)
    best_id = normalize_candidate_id(
        rule.get("best_target_candidate_id")
        or best.get("entry_id")
        or best.get("candidate_id")
        or memory.get("best_entry_id")
    )
    best_type = candidate_type(best)
    state = normalize_token(rule.get("rule_target_conditioned_state")) or "target_house_entry_visible"
    subgoal = normalize_token(rule.get("rule_target_conditioned_subgoal")) or "keep_search_target_house"
    action = normalize_token(rule.get("rule_target_conditioned_action_hint")) or "hold"
    association = "target_house_entry" if best_type in DOORLIKE_CLASSES else "uncertain_entry"
    memory_decision = "reuse_confirmed_best_entry" if best_id else "no_memory_available"
    if best_type in WINDOW_CLASSES:
        association = "window_or_non_entry"
        memory_decision = "reject_window"
    confidence = safe_float(best.get("association_confidence"), 0.5) if isinstance(best, dict) else 0.5
    return {
        "target_conditioned_state": state,
        "target_conditioned_subgoal": subgoal,
        "target_conditioned_action_hint": action,
        "target_candidate_id": best_id,
        "entry_association": association,
        "memory_decision": memory_decision,
        "confidence": round(max(0.1, min(0.85, confidence)), 4),
        "reason": reason or str(rule.get("rule_target_conditioned_reason") or "Fallback to rule-based fusion teacher."),
    }


def add_issue(issues: List[Dict[str, str]], severity: str, code: str, message: str) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def validate_label(
    *,
    prompt_payload: Dict[str, Any],
    label: Dict[str, Any],
    label_source: str,
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    if not prompt_payload:
        add_issue(errors, "FAIL", "missing_prompt", "Prompt payload is missing or invalid.")
    if not label:
        add_issue(errors, "FAIL", "missing_label", "LLM teacher label is missing or invalid.")
        fallback = fallback_label_from_prompt(prompt_payload, reason="LLM label is missing; fallback to rule-based fusion teacher.")
        return {
            "valid": False,
            "status": "FAIL",
            "label_source": label_source,
            "normalized_label": {},
            "fallback_label": fallback,
            "errors": errors,
            "warnings": warnings,
        }

    for field in REQUIRED_FIELDS:
        if field not in label:
            add_issue(errors, "FAIL", "missing_required_field", f"Missing required field: {field}")

    normalized = normalize_label(label)
    state = normalized["target_conditioned_state"]
    subgoal = normalized["target_conditioned_subgoal"]
    action = normalized["target_conditioned_action_hint"]
    candidate_id = normalized["target_candidate_id"]
    entry_association = normalized["entry_association"]
    memory_decision = normalized["memory_decision"]
    confidence = safe_float(normalized["confidence"])
    reason = normalized["reason"]

    if state not in TARGET_CONDITIONED_STATES:
        add_issue(errors, "FAIL", "invalid_target_conditioned_state", f"Unsupported target_conditioned_state: {state}")
    if subgoal not in TARGET_CONDITIONED_SUBGOALS:
        add_issue(errors, "FAIL", "invalid_target_conditioned_subgoal", f"Unsupported target_conditioned_subgoal: {subgoal}")
    if action not in ACTION_HINTS:
        add_issue(errors, "FAIL", "invalid_target_conditioned_action_hint", f"Unsupported action hint: {action}")
    if entry_association not in ENTRY_ASSOCIATIONS:
        add_issue(errors, "FAIL", "invalid_entry_association", f"Unsupported entry_association: {entry_association}")
    if memory_decision not in MEMORY_DECISIONS:
        add_issue(errors, "FAIL", "invalid_memory_decision", f"Unsupported memory_decision: {memory_decision}")
    if not (0.0 <= confidence <= 1.0):
        add_issue(errors, "FAIL", "invalid_confidence", f"Confidence must be in [0,1], got {confidence}")

    cmap = candidate_map(prompt_payload)
    selected_candidate = cmap.get(candidate_id or "", {}) if candidate_id is not None else {}
    if candidate_id is not None and candidate_id not in cmap:
        add_issue(errors, "FAIL", "candidate_id_not_found", f"target_candidate_id={candidate_id} is not in provided candidates.")

    selected_type = candidate_type(selected_candidate)
    best = best_memory_entry(prompt_payload)
    best_id = normalize_candidate_id(best.get("entry_id") or best.get("candidate_id"))
    best_type = candidate_type(best)
    depth = get_depth_evidence(prompt_payload)
    target_context = get_target_context(prompt_payload)
    traversable = truthy(depth.get("traversable"))
    crossing_ready = truthy(depth.get("crossing_ready"))
    front_obstacle = truthy(depth.get("front_obstacle_present"))
    target_in_fov = truthy(target_context.get("target_house_in_fov"))

    if selected_type in WINDOW_CLASSES and (
        entry_association == "target_house_entry" or subgoal in APPROACH_SUBGOALS
    ):
        add_issue(errors, "FAIL", "window_selected_as_target_entry", "A window candidate cannot be selected as target entry.")

    if subgoal in APPROACH_SUBGOALS and candidate_id is None:
        add_issue(errors, "FAIL", "approach_without_candidate", f"{subgoal} requires a target_candidate_id.")
    if subgoal in APPROACH_SUBGOALS and selected_type and selected_type not in DOORLIKE_CLASSES:
        add_issue(errors, "FAIL", "approach_candidate_not_doorlike", f"{subgoal} selected non-door candidate type={selected_type}.")
    if subgoal in {"approach_target_entry", "align_target_entry"} and not traversable:
        add_issue(errors, "FAIL", "approach_not_traversable", "Depth evidence is not traversable but label requests approach/align.")
    if subgoal == "cross_target_entry" and not (traversable and crossing_ready):
        add_issue(errors, "FAIL", "cross_not_ready", "cross_target_entry requires traversable and crossing_ready depth evidence.")
    if front_obstacle and action == "forward" and subgoal in {"detour_left_to_target_entry", "detour_right_to_target_entry", "backoff_and_reobserve"}:
        add_issue(errors, "FAIL", "blocked_forward_conflict", "Front obstacle is present but action is forward for a detour/backoff subgoal.")
    if state == "target_house_not_in_view" and subgoal in APPROACH_SUBGOALS:
        add_issue(errors, "FAIL", "target_not_in_view_approach_conflict", "Cannot approach target entry when target house is not in view.")
    if state == "target_house_not_in_view" and target_in_fov:
        add_issue(warnings, "WARN", "target_in_fov_state_conflict", "Prompt says target_house_in_fov=true but label says not in view.")
    if state == "non_target_house_entry_visible" and entry_association == "target_house_entry":
        add_issue(errors, "FAIL", "non_target_state_target_association_conflict", "State says non-target entry but association says target_house_entry.")

    best_strong = (
        best_id is not None
        and best_type in DOORLIKE_CLASSES
        and safe_float(best.get("association_confidence")) >= 0.8
        and safe_float(best.get("observation_count")) >= 3
        and evidence_nonzero(best)
    )
    if best_strong and candidate_id is not None and candidate_id != best_id:
        add_issue(
            warnings,
            "WARN",
            "strong_best_entry_not_selected",
            f"Memory has strong best entry {best_id}, but label selected {candidate_id}.",
        )
    if best_strong and subgoal in {"keep_search_target_house", "ignore_non_target_entry"}:
        add_issue(
            warnings,
            "WARN",
            "strong_best_entry_ignored",
            "Memory has a strong door-like best entry, but label keeps searching or ignores entry.",
        )
    if confidence < 0.45:
        add_issue(warnings, "WARN", "low_confidence", f"LLM teacher confidence is low: {confidence:.2f}")
    if len(reason) < 24:
        add_issue(warnings, "WARN", "reason_too_short", "Reason is too short to explain the decision.")
    if len(reason) > 700:
        add_issue(warnings, "WARN", "reason_too_long", "Reason is unusually long.")

    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    fallback = fallback_label_from_prompt(prompt_payload, reason="LLM label failed validation; fallback to rule-based fusion teacher.")
    return {
        "valid": not errors,
        "status": status,
        "label_source": label_source,
        "normalized_label": normalized,
        "fallback_label": fallback if errors else {},
        "errors": errors,
        "warnings": warnings,
        "context": {
            "candidate_ids": sorted(cmap.keys(), key=lambda item: (len(item), item)),
            "selected_candidate_type": selected_type,
            "best_memory_entry_id": best_id,
            "best_memory_entry_type": best_type,
            "depth_traversable": traversable,
            "depth_crossing_ready": crossing_ready,
            "front_obstacle_present": front_obstacle,
            "target_house_in_fov": target_in_fov,
        },
    }


def validation_status(result: Dict[str, Any]) -> str:
    return str(result.get("status") or "FAIL")


def validate_prompt_and_label(
    *,
    prompt_path: Path,
    label_path: Path,
    response_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    write_validated: bool = False,
) -> Dict[str, Any]:
    prompt_payload = read_json(prompt_path)
    label, label_source = load_label(label_path, response_path)
    result = validate_label(prompt_payload=prompt_payload, label=label, label_source=label_source)
    result.update(
        {
            "prompt_path": str(prompt_path),
            "label_path": str(label_path),
            "response_path": str(response_path) if response_path else "",
            "output_path": str(output_path) if output_path else "",
        }
    )
    if write_validated and output_path:
        payload = {
            "valid": bool(result.get("valid")),
            "status": result.get("status"),
            "label_source": result.get("label_source"),
            "normalized_label": result.get("normalized_label"),
            "fallback_label": result.get("fallback_label"),
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
            "context": result.get("context", {}),
        }
        write_json(output_path, payload)
    return result


def validate_labeling_dir(
    labeling_dir: Path,
    *,
    prompt_name: str = "llm_teacher_prompt.json",
    label_name: str = "llm_teacher_label.json",
    response_name: str = "llm_teacher_response.json",
    output_name: str = "llm_teacher_label_validated.json",
    write_validated: bool = False,
) -> Dict[str, Any]:
    prompt_path = labeling_dir / prompt_name
    label_path = labeling_dir / label_name
    response_path = labeling_dir / response_name
    output_path = labeling_dir / output_name
    result = validate_prompt_and_label(
        prompt_path=prompt_path,
        label_path=label_path,
        response_path=response_path,
        output_path=output_path,
        write_validated=write_validated,
    )
    result["labeling_dir"] = str(labeling_dir)
    return result


def discover_labeling_dirs(session_dir: Path) -> List[Path]:
    captures_root = session_dir / "memory_fusion_captures"
    if not captures_root.exists():
        return []
    output: List[Path] = []
    for capture_dir in sorted(captures_root.iterdir(), key=lambda path: path.name):
        labeling_dir = capture_dir / "labeling"
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(validation_status(item) for item in results)
    issue_counts: Counter = Counter()
    for item in results:
        for issue in as_list(item.get("errors")) + as_list(item.get("warnings")):
            if isinstance(issue, dict):
                issue_counts[str(issue.get("code") or "")] += 1
    return {
        "status": "FAIL" if counts.get("FAIL") else ("WARN" if counts.get("WARN") else "PASS"),
        "count": len(results),
        "status_counts": compact_counter(counts),
        "issue_counts": compact_counter(issue_counts),
        "results": results,
    }


def print_result(result: Dict[str, Any]) -> None:
    print(
        "[llm-label-validator] "
        f"status={result.get('status')} "
        f"valid={int(bool(result.get('valid')))} "
        f"prompt={result.get('prompt_path')}"
    )
    normalized = result.get("normalized_label", {}) if isinstance(result.get("normalized_label"), dict) else {}
    if normalized:
        print(
            "[llm-label-validator] "
            f"state={normalized.get('target_conditioned_state')} "
            f"subgoal={normalized.get('target_conditioned_subgoal')} "
            f"action={normalized.get('target_conditioned_action_hint')} "
            f"candidate={normalized.get('target_candidate_id')} "
            f"assoc={normalized.get('entry_association')} "
            f"memory={normalized.get('memory_decision')}"
        )
    for issue in as_list(result.get("errors")) + as_list(result.get("warnings")):
        if isinstance(issue, dict):
            print(f"  - {issue.get('severity')} {issue.get('code')}: {issue.get('message')}")


def print_batch(summary: Dict[str, Any]) -> None:
    print(
        "[llm-label-validator] "
        f"status={summary.get('status')} "
        f"count={summary.get('count')} "
        f"status_counts={summary.get('status_counts')}"
    )
    if summary.get("issue_counts"):
        print(f"[llm-label-validator] issue_counts={summary.get('issue_counts')}")
    for item in summary.get("results", [])[:20]:
        print(
            "  - "
            f"{item.get('status')} "
            f"{Path(str(item.get('prompt_path'))).parent.name} "
            f"label={Path(str(item.get('label_source') or item.get('label_path'))).name}"
        )
    if len(summary.get("results", [])) > 20:
        print(f"  ... {len(summary['results']) - 20} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate memory-aware LLM teacher labels.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labeling_dir", type=Path, help="Single capture labeling directory.")
    group.add_argument("--session_dir", type=Path, help="memory_episode_* directory.")
    group.add_argument("--prompt_json", type=Path, help="Standalone llm_teacher_prompt.json path.")
    parser.add_argument("--label_json", type=Path, default=None, help="Standalone llm_teacher_label.json path.")
    parser.add_argument("--prompt_name", default="llm_teacher_prompt.json")
    parser.add_argument("--label_name", default="llm_teacher_label.json")
    parser.add_argument("--response_name", default="llm_teacher_response.json")
    parser.add_argument("--output_name", default="llm_teacher_label_validated.json")
    parser.add_argument("--write_validated", action="store_true")
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt_json:
        label_path = args.label_json or args.prompt_json.with_name("llm_teacher_label.json")
        response_path = args.prompt_json.with_name(args.response_name)
        output_path = args.prompt_json.with_name(args.output_name)
        result = validate_prompt_and_label(
            prompt_path=args.prompt_json,
            label_path=label_path,
            response_path=response_path,
            output_path=output_path,
            write_validated=args.write_validated,
        )
        if args.summary_json:
            write_json(args.summary_json, result)
        if not args.quiet:
            print_result(result)
        return 1 if result.get("status") == "FAIL" else 0

    if args.labeling_dir:
        result = validate_labeling_dir(
            args.labeling_dir,
            prompt_name=args.prompt_name,
            label_name=args.label_name,
            response_name=args.response_name,
            output_name=args.output_name,
            write_validated=args.write_validated,
        )
        if args.summary_json:
            write_json(args.summary_json, result)
        if not args.quiet:
            print_result(result)
        return 1 if result.get("status") == "FAIL" else 0

    labeling_dirs = discover_labeling_dirs(args.session_dir)
    results = [
        validate_labeling_dir(
            labeling_dir,
            prompt_name=args.prompt_name,
            label_name=args.label_name,
            response_name=args.response_name,
            output_name=args.output_name,
            write_validated=args.write_validated,
        )
        for labeling_dir in labeling_dirs
    ]
    summary = aggregate_results(results)
    summary["session_dir"] = str(args.session_dir)
    if args.summary_json:
        write_json(args.summary_json, summary)
    if not args.quiet:
        print_batch(summary)
    return 1 if summary.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
