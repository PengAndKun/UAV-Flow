"""
Validate the LLM planner adapter on realistic mission/search prompts.

This script is intentionally offline-friendly:
- it reuses the local heuristic planner as a geometry seed
- it validates structured LLM response parsing with a mock client
- it provides a quick smoke-test before live planner-server integration
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_planner_adapter import LLMPlannerAdapterError, build_llm_plan
from planner_server import build_heuristic_plan


@dataclass(frozen=True)
class LLMPlannerValidationCase:
    task_label: str
    llm_response: Dict[str, Any]
    expected_mission_type: str
    expected_search_subgoal: str
    expected_priority_contains: str
    expected_confirm_target: Optional[bool] = None


TASK_SUITE: List[LLMPlannerValidationCase] = [
    LLMPlannerValidationCase(
        task_label="search the house for people",
        llm_response={
            "mission_type": "person_search",
            "search_subgoal": "search_house",
            "priority_region_label": "entire house",
            "candidate_region_labels": ["entire house"],
            "confirm_target": False,
            "should_replan": False,
            "planner_confidence": 0.74,
            "semantic_subgoal": "forward_search",
            "waypoint_strategy": "broader_sweep",
            "explanation": "Start with broad house-level coverage before narrowing to suspect rooms.",
        },
        expected_mission_type="person_search",
        expected_search_subgoal="search_house",
        expected_priority_contains="entire house",
        expected_confirm_target=False,
    ),
    LLMPlannerValidationCase(
        task_label="search the bedroom first and then check the hallway",
        llm_response={
            "mission_type": "room_search",
            "search_subgoal": "search_room",
            "priority_region_label": "bedroom",
            "candidate_region_labels": ["bedroom", "hallway"],
            "confirm_target": False,
            "should_replan": False,
            "planner_confidence": 0.77,
            "semantic_subgoal": "forward_search",
            "waypoint_strategy": "shorter_approach",
            "explanation": "Bedroom should be searched before hallway because the mission explicitly prioritizes it.",
        },
        expected_mission_type="room_search",
        expected_search_subgoal="search_room",
        expected_priority_contains="bedroom",
        expected_confirm_target=False,
    ),
    LLMPlannerValidationCase(
        task_label="search the living room for a survivor",
        llm_response={
            "mission_type": "person_search",
            "search_subgoal": "search_room",
            "priority_region_label": "living room",
            "candidate_region_labels": ["living room"],
            "confirm_target": False,
            "should_replan": False,
            "planner_confidence": 0.79,
            "semantic_subgoal": "forward_search",
            "waypoint_strategy": "use_seed_waypoints",
            "explanation": "The living room is the first room that should be cleared for survivor search.",
        },
        expected_mission_type="person_search",
        expected_search_subgoal="search_room",
        expected_priority_contains="living room",
        expected_confirm_target=False,
    ),
    LLMPlannerValidationCase(
        task_label="approach and verify the suspect region near the bedroom door",
        llm_response={
            "mission_type": "target_verification",
            "search_subgoal": "approach_suspect_region",
            "priority_region_label": "suspect region",
            "candidate_region_labels": ["suspect region", "doorway", "bedroom"],
            "confirm_target": True,
            "should_replan": True,
            "planner_confidence": 0.83,
            "semantic_subgoal": "forward_search",
            "waypoint_strategy": "shorter_approach",
            "explanation": "Approach the suspect region conservatively and prepare to confirm the target.",
        },
        expected_mission_type="target_verification",
        expected_search_subgoal="approach_suspect_region",
        expected_priority_contains="suspect region",
        expected_confirm_target=True,
    ),
    LLMPlannerValidationCase(
        task_label="revisit the bathroom and confirm whether a person is there",
        llm_response={
            "mission_type": "target_verification",
            "search_subgoal": "confirm_suspect_region",
            "priority_region_label": "bathroom",
            "candidate_region_labels": ["bathroom"],
            "confirm_target": True,
            "should_replan": True,
            "planner_confidence": 0.81,
            "semantic_subgoal": "forward_search",
            "waypoint_strategy": "shorter_approach",
            "explanation": "The bathroom must be revisited for explicit confirmation.",
        },
        expected_mission_type="target_verification",
        expected_search_subgoal="confirm_suspect_region",
        expected_priority_contains="bathroom",
        expected_confirm_target=True,
    ),
]


class MockPlannerClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, *, system_prompt: str, user_prompt: str, image_b64: str = "") -> Dict[str, Any]:
        return {
            "text": self.response_text,
            "raw_response": {"mock": True},
            "usage": {"total_tokens": 123},
            "latency_ms": 12.3,
            "model_name": "mock-llm-planner",
            "api_style": "mock",
            "endpoint_path": "/mock",
            "attempt_count": 1,
            "system_prompt_preview": system_prompt[:80],
            "user_prompt_preview": user_prompt[:120],
            "image_included": bool(image_b64),
        }


def build_payload(task_label: str, *, step_index: int) -> Dict[str, Any]:
    mission_type = "person_search" if "people" in task_label or "survivor" in task_label else "room_search"
    if any(keyword in task_label.lower() for keyword in ["verify", "confirm", "approach"]):
        mission_type = "target_verification"
    return {
        "task_label": task_label,
        "frame_id": f"frame_{step_index:06d}",
        "step_index": step_index,
        "pose": {
            "x": 120.0,
            "y": -45.0,
            "z": 180.0,
            "yaw": 15.0,
        },
        "depth": {
            "min_depth": 420.0,
            "max_depth": 1200.0,
        },
        "image_b64": "ZmFrZV9pbWFnZV9iYXNlNjQ=",
        "mission": {
            "mission_type": mission_type,
            "search_scope": "house" if "house" in task_label.lower() else "room",
            "task_label": task_label,
        },
        "search_runtime": {
            "mission_status": "active",
            "current_search_subgoal": "search_frontier",
            "detection_state": "searching",
            "visited_region_count": 2,
            "suspect_region_count": 1 if "suspect" in task_label.lower() else 0,
            "confirmed_region_count": 0,
            "evidence_count": 1 if "suspect" in task_label.lower() else 0,
        },
        "person_evidence_runtime": {
            "evidence_status": "suspect" if "suspect" in task_label.lower() else "searching",
            "suspect_count": 1 if "suspect" in task_label.lower() else 0,
            "confirm_present_count": 0,
            "confirm_absent_count": 0,
            "confidence": 0.58 if "suspect" in task_label.lower() else 0.0,
        },
        "search_result": {
            "result_status": "unknown",
            "person_exists": None,
            "confidence": 0.0,
            "summary": "No final search result yet.",
        },
        "context": {
            "risk_score": 0.12,
            "archive": {
                "current_cell_id": "cell_001",
                "recent_cell_ids": ["cell_001", "cell_000"],
                "top_cells": [{"cell_id": "cell_001"}, {"cell_id": "cell_009"}],
            },
            "reflex_runtime": {
                "mode": "mlp_policy",
                "suggested_action": "forward",
                "policy_confidence": 0.62,
            },
        },
    }


def validate_case(case: LLMPlannerValidationCase, *, planner_name: str, waypoint_radius_cm: float, step_index: int) -> Dict[str, Any]:
    payload = build_payload(case.task_label, step_index=step_index)
    heuristic_seed = build_heuristic_plan(payload, planner_name="heuristic_seed", waypoint_radius_cm=waypoint_radius_cm)
    mock_client = MockPlannerClient(json.dumps(case.llm_response))
    plan = build_llm_plan(
        request_payload=payload,
        heuristic_seed=heuristic_seed,
        client=mock_client,
        planner_name=planner_name,
        waypoint_radius_cm=waypoint_radius_cm,
    )
    priority_label = str((plan.get("priority_region") or {}).get("region_label", ""))
    checks = {
        "mission_type": plan.get("mission_type") == case.expected_mission_type,
        "search_subgoal": plan.get("search_subgoal") == case.expected_search_subgoal,
        "priority_region": case.expected_priority_contains.lower() in priority_label.lower(),
        "confirm_target": bool(plan.get("confirm_target", False)) == case.expected_confirm_target
        if case.expected_confirm_target is not None
        else True,
        "candidate_waypoints": bool(plan.get("candidate_waypoints")),
        "debug_source": str((plan.get("debug") or {}).get("source", "")) == "llm_planner",
    }
    return {
        "task_label": case.task_label,
        "passed": all(checks.values()),
        "checks": checks,
        "actual": {
            "mission_type": plan.get("mission_type"),
            "search_subgoal": plan.get("search_subgoal"),
            "priority_region": priority_label,
            "candidate_regions": [str(region.get("region_label", "")) for region in plan.get("candidate_regions") or []],
            "confirm_target": bool(plan.get("confirm_target", False)),
            "planner_name": plan.get("planner_name"),
            "planner_confidence": plan.get("planner_confidence"),
            "debug": plan.get("debug", {}),
        },
    }


def validate_malformed_response() -> Dict[str, Any]:
    payload = build_payload("search the house for people", step_index=999)
    heuristic_seed = build_heuristic_plan(payload, planner_name="heuristic_seed", waypoint_radius_cm=60.0)
    mock_client = MockPlannerClient("not a json object")
    try:
        build_llm_plan(
            request_payload=payload,
            heuristic_seed=heuristic_seed,
            client=mock_client,
            planner_name="llm_planner",
            waypoint_radius_cm=60.0,
        )
    except LLMPlannerAdapterError as exc:
        return {
            "passed": True,
            "error": str(exc),
        }
    return {
        "passed": False,
        "error": "Malformed response did not raise an adapter error.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the offline LLM planner adapter on realistic search prompts.")
    parser.add_argument("--planner_name", default="external_llm_planner", help="Planner name assigned to generated plans")
    parser.add_argument("--waypoint_radius_cm", type=float, default=60.0, help="Waypoint radius used when seeding geometry")
    parser.add_argument("--strict", action="store_true", help="Return non-zero exit code if any case fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        validate_case(case, planner_name=args.planner_name, waypoint_radius_cm=args.waypoint_radius_cm, step_index=index + 1)
        for index, case in enumerate(TASK_SUITE)
    ]
    malformed = validate_malformed_response()
    summary = {
        "schema_version": "phase45.llm_planner_validation.v1",
        "case_count": len(reports),
        "pass_count": sum(1 for report in reports if report["passed"]),
        "fail_count": sum(1 for report in reports if not report["passed"]),
        "malformed_response_check": malformed,
        "reports": reports,
    }
    print("=== LLM Planner Validation ===")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.strict and (summary["fail_count"] > 0 or not malformed["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
