from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are a memory-aware UAV entry-search teacher.

Your job is to produce target-conditioned supervision labels for a lightweight student policy.
You are not directly controlling the UAV.

Use the provided YOLO/RGB evidence, depth traversability evidence, target-house context, and structured memory.

Rules:
1. Do not invent candidate ids. Select target_candidate_id only from provided candidates, or use null.
2. A window must not be selected as a target entry.
3. Prefer a door/open door/close door candidate only if it is associated with the target house.
4. Strong memory evidence includes repeated observations, bbox history, high association confidence, and non-zero distance/view/appearance/geometry evidence.
5. If the target house is not in view, output target_house_not_in_view and reorient_to_target_house.
6. If the best target-house entry is visible but blocked, output target_house_entry_blocked and choose detour/backoff/search, not forward.
7. If the best target-house entry is approachable but not crossing-ready, output approach_target_entry.
8. If the entry is aligned and crossing-ready, output cross_target_entry.
9. If only a non-target entry or window is visible, output non_target_house_entry_visible and ignore_non_target_entry.
10. Return strict JSON only.
"""

OUTPUT_SCHEMA = {
    "target_conditioned_state": "target_house_entry_approachable",
    "target_conditioned_subgoal": "approach_target_entry",
    "target_conditioned_action_hint": "forward",
    "target_candidate_id": "0",
    "entry_association": "target_house_entry",
    "memory_decision": "reuse_confirmed_best_entry",
    "confidence": 0.92,
    "reason": "Short evidence-based explanation.",
}

ALLOWED_TARGET_CONDITIONED_STATES = [
    "target_house_not_in_view",
    "target_house_entry_visible",
    "target_house_entry_approachable",
    "target_house_entry_blocked",
    "non_target_house_entry_visible",
    "target_house_geometric_opening_needs_confirmation",
]

ALLOWED_TARGET_CONDITIONED_SUBGOALS = [
    "reorient_to_target_house",
    "keep_search_target_house",
    "approach_target_entry",
    "align_target_entry",
    "detour_left_to_target_entry",
    "detour_right_to_target_entry",
    "cross_target_entry",
    "ignore_non_target_entry",
    "backoff_and_reobserve",
]

ALLOWED_ACTION_HINTS = [
    "forward",
    "yaw_left",
    "yaw_right",
    "left",
    "right",
    "backward",
    "hold",
]

ALLOWED_ENTRY_ASSOCIATIONS = [
    "target_house_entry",
    "non_target_house_entry",
    "window_or_non_entry",
    "uncertain_entry",
    "no_entry",
]

ALLOWED_MEMORY_DECISIONS = [
    "reuse_confirmed_best_entry",
    "update_best_entry",
    "continue_observing",
    "shift_search_sector",
    "reject_window",
    "reject_non_target_entry",
    "no_memory_available",
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def round_float(value: Any, digits: int = 4) -> float:
    return round(safe_float(value), digits)


def bbox_from_detection(detection: Dict[str, Any]) -> List[float]:
    xyxy = detection.get("xyxy", [])
    if isinstance(xyxy, list) and len(xyxy) == 4:
        return [float(v) for v in xyxy]
    bbox = detection.get("bbox", {})
    if isinstance(bbox, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(set(bbox.keys())):
            return [safe_float(bbox.get("x1")), safe_float(bbox.get("y1")), safe_float(bbox.get("x2")), safe_float(bbox.get("y2"))]
        if {"x", "y", "width", "height"}.issubset(set(bbox.keys())):
            x = safe_float(bbox.get("x"))
            y = safe_float(bbox.get("y"))
            return [x, y, x + safe_float(bbox.get("width")), y + safe_float(bbox.get("height"))]
    return []


def get_fusion(labeling_dir: Path) -> Dict[str, Any]:
    payload = read_json(labeling_dir / "fusion_result.json")
    fusion = payload.get("fusion", {}) if isinstance(payload.get("fusion"), dict) else {}
    return fusion


def get_memory_root(labeling_dir: Path, *, prefer: str = "after") -> Dict[str, Any]:
    filename = "entry_search_memory_snapshot_after.json" if prefer == "after" else "entry_search_memory_snapshot_before.json"
    payload = read_json(labeling_dir / filename)
    memory = payload.get("memory", {}) if isinstance(payload.get("memory"), dict) else {}
    return memory


def get_house_memory(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    memories = memory_root.get("memories", {}) if isinstance(memory_root.get("memories"), dict) else {}
    memory = memories.get(str(house_id or ""))
    return memory if isinstance(memory, dict) else {}


def get_registry_entry(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    registry = memory_root.get("house_registry", {}) if isinstance(memory_root.get("house_registry"), dict) else {}
    entry = registry.get(str(house_id or ""))
    return entry if isinstance(entry, dict) else {}


def get_candidate_entries(memory_root: Dict[str, Any], house_id: str) -> List[Dict[str, Any]]:
    house_memory = get_house_memory(memory_root, house_id)
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    return [entry for entry in as_list(semantic_memory.get("candidate_entries")) if isinstance(entry, dict)]


def choose_best_entry(entries: List[Dict[str, Any]], registry: Dict[str, Any]) -> Dict[str, Any]:
    best_entry_id = str(registry.get("best_entry_id") or "").strip()
    if best_entry_id:
        for entry in entries:
            if str(entry.get("entry_id") or entry.get("candidate_id") or "").strip() == best_entry_id:
                return entry
    for entry in entries:
        if truthy(entry.get("is_best_candidate", False)):
            return entry
    return {}


def summarize_detection(detection: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "candidate_id": str(detection.get("candidate_id", index)),
        "class_name": str(detection.get("class_name") or ""),
        "confidence": round_float(detection.get("confidence")),
        "bbox": bbox_from_detection(detection),
    }


def summarize_yolo(labeling_dir: Path, fusion: Dict[str, Any]) -> Dict[str, Any]:
    yolo = read_json(labeling_dir / "yolo_result.json")
    detections = yolo.get("detections", []) if isinstance(yolo.get("detections"), list) else []
    if not detections:
        detections = fusion.get("candidate_target_scores", []) if isinstance(fusion.get("candidate_target_scores"), list) else []
    return {
        "num_detections": safe_int(yolo.get("num_detections", len(detections))),
        "detections": [summarize_detection(item, idx) for idx, item in enumerate(detections[:8]) if isinstance(item, dict)],
    }


def summarize_depth(labeling_dir: Path, fusion: Dict[str, Any]) -> Dict[str, Any]:
    depth = read_json(labeling_dir / "depth_result.json")
    analysis = depth.get("analysis", depth) if isinstance(depth.get("analysis", depth), dict) else {}
    best = fusion.get("best_target_candidate", {}) if isinstance(fusion.get("best_target_candidate"), dict) else {}
    chosen = fusion.get("chosen_depth_candidate", {}) if isinstance(fusion.get("chosen_depth_candidate"), dict) else {}
    front_obstacle = analysis.get("front_obstacle", {}) if isinstance(analysis.get("front_obstacle"), dict) else {}
    return {
        "best_target_candidate_id": str(best.get("candidate_id") or ""),
        "entry_distance_cm": round_float(best.get("entry_distance_cm", chosen.get("entry_distance_cm"))),
        "opening_width_cm": round_float(best.get("opening_width_cm", chosen.get("opening_width_cm"))),
        "traversable": bool(best.get("traversable", chosen.get("traversable", False))),
        "crossing_ready": bool(best.get("crossing_ready", chosen.get("crossing_ready", False))),
        "front_obstacle_present": bool(front_obstacle.get("present", False)),
        "front_min_depth_cm": front_obstacle.get("front_min_depth_cm"),
        "front_obstacle_severity": str(front_obstacle.get("severity") or ""),
    }


def summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    evidence = entry.get("association_evidence", {}) if isinstance(entry.get("association_evidence"), dict) else {}
    return {
        "entry_id": str(entry.get("entry_id") or entry.get("candidate_id") or ""),
        "entry_type": str(entry.get("entry_type") or entry.get("semantic_class") or ""),
        "status": str(entry.get("status") or entry.get("entry_state") or ""),
        "associated_house_id": str(entry.get("associated_house_id") or ""),
        "observation_count": safe_int(entry.get("observation_count")),
        "source_frame_count": len(as_list(entry.get("source_frames"))),
        "bbox_history_count": len(as_list(entry.get("bbox_history"))),
        "target_match_score": round_float(entry.get("target_match_score")),
        "association_confidence": round_float(entry.get("association_confidence")),
        "association_evidence": {
            "distance_score": round_float(evidence.get("distance_score")),
            "view_consistency_score": round_float(evidence.get("view_consistency_score")),
            "appearance_score": round_float(evidence.get("appearance_score")),
            "language_score": round_float(evidence.get("language_score")),
            "geometry_score": round_float(evidence.get("geometry_score")),
            "memory_similarity_score": round_float(evidence.get("memory_similarity_score")),
        },
        "is_best_candidate": truthy(entry.get("is_best_candidate", False)),
        "is_searched": truthy(entry.get("is_searched", False)),
        "is_entered": truthy(entry.get("is_entered", False)),
    }


def summarize_memory(memory_root: Dict[str, Any], target_house_id: str) -> Dict[str, Any]:
    registry = get_registry_entry(memory_root, target_house_id)
    entries = get_candidate_entries(memory_root, target_house_id)
    best_entry = choose_best_entry(entries, registry)
    house_memory = get_house_memory(memory_root, target_house_id)
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    return {
        "search_status": str(registry.get("search_status") or ""),
        "entry_search_status": str(registry.get("entry_search_status") or ""),
        "candidate_entry_count": len(entries),
        "best_entry_id": str(registry.get("best_entry_id") or ""),
        "best_entry": summarize_entry(best_entry) if best_entry else {},
        "candidate_entries": [summarize_entry(entry) for entry in entries[:8]],
        "perimeter_coverage": semantic_memory.get("perimeter_coverage", {})
        if isinstance(semantic_memory.get("perimeter_coverage"), dict)
        else {},
        "search_completion_evidence": semantic_memory.get("search_completion_evidence", {})
        if isinstance(semantic_memory.get("search_completion_evidence"), dict)
        else {},
        "planner_context": memory_root.get("planner_context", {}) if isinstance(memory_root.get("planner_context"), dict) else {},
    }


def summarize_temporal(labeling_dir: Path) -> Dict[str, Any]:
    temporal = read_json(labeling_dir / "temporal_context.json")
    sample = read_json(labeling_dir / "sample_metadata.json")
    state = read_json(labeling_dir / "state.json")
    actions = temporal.get("actions_since_last_capture")
    if not isinstance(actions, list):
        action_history_path = str(
            temporal.get("action_history_since_last_capture_path")
            or sample.get("action_history_since_last_capture_path")
            or ""
        )
        if action_history_path:
            history = read_json(Path(action_history_path))
            actions = history.get("actions") if isinstance(history.get("actions"), list) else []
    if not isinstance(actions, list):
        actions = []

    def compact_action(item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        pose_before = item.get("pose_before", {}) if isinstance(item.get("pose_before"), dict) else {}
        pose_after = item.get("pose_after", {}) if isinstance(item.get("pose_after"), dict) else {}
        movement = item.get("movement", {}) if isinstance(item.get("movement"), dict) else {}
        return {
            "action_name": str(item.get("action_name") or ""),
            "step_before": safe_int(item.get("step_before")),
            "step_after": safe_int(item.get("step_after")),
            "pose_before": {
                "x": round_float(pose_before.get("x"), 2),
                "y": round_float(pose_before.get("y"), 2),
                "yaw": round_float(pose_before.get("task_yaw", pose_before.get("uav_yaw")), 2),
            },
            "pose_after": {
                "x": round_float(pose_after.get("x"), 2),
                "y": round_float(pose_after.get("y"), 2),
                "yaw": round_float(pose_after.get("task_yaw", pose_after.get("uav_yaw")), 2),
            },
            "movement": {
                "forward_cm": round_float(movement.get("forward_cm"), 2),
                "right_cm": round_float(movement.get("right_cm"), 2),
                "yaw_delta_deg": round_float(movement.get("yaw_delta_deg"), 2),
            },
        }

    return {
        "episode_id": str(sample.get("episode_id") or temporal.get("episode_id") or ""),
        "episode_label": str(sample.get("episode_label") or ""),
        "step_index": safe_int(sample.get("memory_step_index", temporal.get("step_index"))),
        "capture_source": str(sample.get("capture_source") or temporal.get("capture_source") or ""),
        "previous_action": str(temporal.get("previous_action") or state.get("last_action") or ""),
        "action_count_since_last_capture": safe_int(
            sample.get("action_count_since_last_capture", temporal.get("action_count_since_last_capture"))
        ),
        "previous_actions": [
            str(item or "")
            for item in as_list(temporal.get("previous_actions"))[:12]
        ],
        "movement_trajectory": [
            action
            for action in (compact_action(item) for item in actions[-12:])
            if action
        ],
        "target_house_id": str(sample.get("target_house_id") or temporal.get("current_target_house_id") or ""),
        "current_house_id": str(sample.get("current_house_id") or temporal.get("current_house_id") or ""),
    }


def summarize_target_context(fusion: Dict[str, Any], temporal: Dict[str, Any]) -> Dict[str, Any]:
    target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
    return {
        "target_house_id": str(target_context.get("target_house_id") or temporal.get("target_house_id") or ""),
        "current_house_id": str(target_context.get("current_house_id") or temporal.get("current_house_id") or ""),
        "target_house_in_fov": bool(target_context.get("target_house_in_fov", False)),
        "target_house_expected_side": str(target_context.get("target_house_expected_side") or ""),
        "target_house_bearing_deg": target_context.get("target_house_bearing_deg"),
        "target_house_distance_cm": target_context.get("target_house_distance_cm"),
        "uav_pose_world": target_context.get("uav_pose_world", {}),
    }


def summarize_fusion_teacher_reference(fusion: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule_target_conditioned_state": str(fusion.get("target_conditioned_state") or ""),
        "rule_target_conditioned_subgoal": str(fusion.get("target_conditioned_subgoal") or ""),
        "rule_target_conditioned_action_hint": str(fusion.get("target_conditioned_action_hint") or ""),
        "rule_target_conditioned_reason": str(fusion.get("target_conditioned_reason") or ""),
        "best_target_candidate_id": str((fusion.get("best_target_candidate") or {}).get("candidate_id") or "")
        if isinstance(fusion.get("best_target_candidate"), dict)
        else "",
        "memory_guidance": fusion.get("memory_guidance", {}) if isinstance(fusion.get("memory_guidance"), dict) else {},
        "memory_decision_guidance": fusion.get("memory_decision_guidance", {})
        if isinstance(fusion.get("memory_decision_guidance"), dict)
        else {},
    }


def build_structured_input(labeling_dir: Path, *, memory_snapshot: str = "after") -> Dict[str, Any]:
    fusion = get_fusion(labeling_dir)
    temporal = summarize_temporal(labeling_dir)
    target_context = summarize_target_context(fusion, temporal)
    target_house_id = str(target_context.get("target_house_id") or temporal.get("target_house_id") or "")
    memory_root = get_memory_root(labeling_dir, prefer=memory_snapshot)
    return {
        "task": "Find and approach the entrance of the target house safely.",
        "labeling_dir": str(labeling_dir),
        "target_context": target_context,
        "temporal_context": temporal,
        "yolo_rgb_evidence": summarize_yolo(labeling_dir, fusion),
        "depth_evidence": summarize_depth(labeling_dir, fusion),
        "memory_evidence": summarize_memory(memory_root, target_house_id),
        "fusion_rule_reference": summarize_fusion_teacher_reference(fusion),
    }


def build_user_prompt(structured_input: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Task:",
            "Find and approach the entrance of the target house.",
            "",
            "Target context:",
            json.dumps(structured_input["target_context"], indent=2, ensure_ascii=False),
            "",
            "YOLO/RGB evidence:",
            json.dumps(structured_input["yolo_rgb_evidence"], indent=2, ensure_ascii=False),
            "",
            "Depth evidence:",
            json.dumps(structured_input["depth_evidence"], indent=2, ensure_ascii=False),
            "",
            "Memory evidence:",
            json.dumps(structured_input["memory_evidence"], indent=2, ensure_ascii=False),
            "",
            "Previous decision context:",
            json.dumps(structured_input["temporal_context"], indent=2, ensure_ascii=False),
            "",
            "Rule-based fusion reference:",
            json.dumps(structured_input["fusion_rule_reference"], indent=2, ensure_ascii=False),
            "",
            "Return strict JSON with this schema:",
            json.dumps(OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
            "",
            "Allowed target_conditioned_state values:",
            json.dumps(ALLOWED_TARGET_CONDITIONED_STATES, ensure_ascii=False),
            "Allowed target_conditioned_subgoal values:",
            json.dumps(ALLOWED_TARGET_CONDITIONED_SUBGOALS, ensure_ascii=False),
            "Allowed target_conditioned_action_hint values:",
            json.dumps(ALLOWED_ACTION_HINTS, ensure_ascii=False),
            "Allowed entry_association values:",
            json.dumps(ALLOWED_ENTRY_ASSOCIATIONS, ensure_ascii=False),
            "Allowed memory_decision values:",
            json.dumps(ALLOWED_MEMORY_DECISIONS, ensure_ascii=False),
        ]
    )


def build_prompt_payload(labeling_dir: Path, *, memory_snapshot: str = "after") -> Dict[str, Any]:
    structured_input = build_structured_input(labeling_dir, memory_snapshot=memory_snapshot)
    return {
        "version": "memory_aware_llm_teacher_prompt_v1",
        "prompt_type": "memory_aware_target_conditioned_teacher",
        "labeling_dir": str(labeling_dir),
        "memory_snapshot": memory_snapshot,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(structured_input),
        "structured_input": structured_input,
        "output_schema": OUTPUT_SCHEMA,
        "allowed_values": {
            "target_conditioned_state": ALLOWED_TARGET_CONDITIONED_STATES,
            "target_conditioned_subgoal": ALLOWED_TARGET_CONDITIONED_SUBGOALS,
            "target_conditioned_action_hint": ALLOWED_ACTION_HINTS,
            "entry_association": ALLOWED_ENTRY_ASSOCIATIONS,
            "memory_decision": ALLOWED_MEMORY_DECISIONS,
        },
    }


def discover_labeling_dirs(session_dir: Path) -> List[Path]:
    captures_root = session_dir / "memory_fusion_captures"
    if not captures_root.exists():
        return []
    output: List[Path] = []
    for capture_dir in sorted(captures_root.iterdir(), key=lambda p: p.name):
        labeling_dir = capture_dir / "labeling"
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def write_prompt_for_labeling_dir(
    labeling_dir: Path,
    *,
    output_name: str = "llm_teacher_prompt.json",
    memory_snapshot: str = "after",
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_path = labeling_dir / output_name
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "reason": "exists",
            "labeling_dir": str(labeling_dir),
            "output_path": str(output_path),
        }
    payload = build_prompt_payload(labeling_dir, memory_snapshot=memory_snapshot)
    write_json(output_path, payload)
    return {
        "status": "ok",
        "labeling_dir": str(labeling_dir),
        "output_path": str(output_path),
        "target_house_id": payload["structured_input"]["target_context"].get("target_house_id"),
        "best_entry_id": payload["structured_input"]["memory_evidence"].get("best_entry_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build memory-aware LLM teacher prompts for memory capture samples.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labeling_dir", type=Path, help="Single memory capture labeling directory.")
    group.add_argument("--session_dir", type=Path, help="memory_episode_* directory containing memory_fusion_captures.")
    parser.add_argument("--output_name", default="llm_teacher_prompt.json")
    parser.add_argument("--memory_snapshot", default="after", choices=["before", "after"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.labeling_dir:
        results = [
            write_prompt_for_labeling_dir(
                args.labeling_dir,
                output_name=args.output_name,
                memory_snapshot=args.memory_snapshot,
                overwrite=args.overwrite,
            )
        ]
    else:
        labeling_dirs = discover_labeling_dirs(args.session_dir)
        results = [
            write_prompt_for_labeling_dir(
                labeling_dir,
                output_name=args.output_name,
                memory_snapshot=args.memory_snapshot,
                overwrite=args.overwrite,
            )
            for labeling_dir in labeling_dirs
        ]
    summary = {
        "status": "ok",
        "count": len(results),
        "ok": sum(1 for item in results if item.get("status") == "ok"),
        "skipped": sum(1 for item in results if item.get("status") == "skipped"),
        "results": results,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(
        "[llm-prompt-builder] "
        f"count={summary['count']} ok={summary['ok']} skipped={summary['skipped']}"
    )
    for item in results[:20]:
        print(
            "  - "
            f"{item.get('status')} "
            f"target={item.get('target_house_id', 'n/a')} "
            f"best={item.get('best_entry_id', 'n/a')} "
            f"{item.get('output_path')}"
        )
    if len(results) > 20:
        print(f"  ... {len(results) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
