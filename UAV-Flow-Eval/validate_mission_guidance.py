"""
Validate the Phase 4.3 mission/search guidance layer on realistic task prompts.

This script is intentionally lightweight:
- default mode uses the local heuristic planner implementation directly
- it focuses on semantic outputs rather than environment execution
- it provides a quick way to sanity-check search-task prompts before live runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from planner_server import build_heuristic_plan


@dataclass(frozen=True)
class MissionValidationCase:
    task_label: str
    expected_mission_type: str
    expected_search_subgoal: str
    expected_priority_contains: str = ""
    expected_confirm_target: Optional[bool] = None


TASK_SUITE: List[MissionValidationCase] = [
    MissionValidationCase(
        task_label="search the house for people",
        expected_mission_type="person_search",
        expected_search_subgoal="search_house",
        expected_priority_contains="entire house",
        expected_confirm_target=False,
    ),
    MissionValidationCase(
        task_label="search the bedroom first and then check the hallway",
        expected_mission_type="room_search",
        expected_search_subgoal="search_room",
        expected_priority_contains="bedroom",
        expected_confirm_target=False,
    ),
    MissionValidationCase(
        task_label="search the living room for a survivor",
        expected_mission_type="person_search",
        expected_search_subgoal="search_room",
        expected_priority_contains="living room",
        expected_confirm_target=False,
    ),
    MissionValidationCase(
        task_label="approach and verify the suspect region near the bedroom door",
        expected_mission_type="target_verification",
        expected_search_subgoal="approach_suspect_region",
        expected_priority_contains="suspect region",
        expected_confirm_target=True,
    ),
    MissionValidationCase(
        task_label="revisit the bathroom and confirm whether a person is there",
        expected_mission_type="target_verification",
        expected_search_subgoal="confirm_suspect_region",
        expected_priority_contains="bathroom",
        expected_confirm_target=True,
    ),
]


def build_payload(task_label: str, *, step_index: int) -> Dict[str, Any]:
    return {
        "task_label": task_label,
        "frame_id": f"frame_{step_index:06d}",
        "step_index": step_index,
        "pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 120.0,
            "yaw": 0.0,
        },
        "depth": {
            "min_depth": 650.0,
            "max_depth": 1200.0,
        },
    }


def validate_case(case: MissionValidationCase, *, planner_name: str, waypoint_radius_cm: float, step_index: int) -> Dict[str, Any]:
    payload = build_payload(case.task_label, step_index=step_index)
    plan = build_heuristic_plan(payload, planner_name=planner_name, waypoint_radius_cm=waypoint_radius_cm)
    priority_label = str((plan.get("priority_region") or {}).get("region_label", ""))
    checks = {
        "mission_type": plan.get("mission_type") == case.expected_mission_type,
        "search_subgoal": plan.get("search_subgoal") == case.expected_search_subgoal,
        "priority_region": case.expected_priority_contains.lower() in priority_label.lower()
        if case.expected_priority_contains
        else True,
        "confirm_target": bool(plan.get("confirm_target", False)) == case.expected_confirm_target
        if case.expected_confirm_target is not None
        else True,
    }
    passed = all(checks.values())
    return {
        "task_label": case.task_label,
        "expected": {
            "mission_type": case.expected_mission_type,
            "search_subgoal": case.expected_search_subgoal,
            "priority_contains": case.expected_priority_contains,
            "confirm_target": case.expected_confirm_target,
        },
        "actual": {
            "mission_type": plan.get("mission_type"),
            "search_subgoal": plan.get("search_subgoal"),
            "priority_region": priority_label,
            "candidate_regions": [str(region.get("region_label", "")) for region in plan.get("candidate_regions") or []],
            "confirm_target": bool(plan.get("confirm_target", False)),
            "semantic_subgoal": plan.get("semantic_subgoal"),
            "explanation": plan.get("explanation", ""),
        },
        "checks": checks,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 4.3 mission guidance on realistic search prompts.")
    parser.add_argument("--planner_name", default="external_heuristic_planner", help="Planner name for local validation")
    parser.add_argument("--waypoint_radius_cm", type=float, default=60.0, help="Waypoint radius used for local planner builds")
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code if any case fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        validate_case(case, planner_name=args.planner_name, waypoint_radius_cm=args.waypoint_radius_cm, step_index=index + 1)
        for index, case in enumerate(TASK_SUITE)
    ]
    summary = {
        "schema_version": "phase4.mission_guidance_validation.v1",
        "case_count": len(reports),
        "pass_count": sum(1 for report in reports if report["passed"]),
        "fail_count": sum(1 for report in reports if not report["passed"]),
        "reports": reports,
    }
    print("=== Mission Guidance Validation ===")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.strict and summary["fail_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
