"""Simple validation script for Phase 5 mission-manual generation."""

from __future__ import annotations

import json

from phase5_mission_manual import build_phase5_mission_manual
from runtime_interfaces import build_mission_state, build_search_region, build_search_runtime_state


def main() -> None:
    mission = build_mission_state(
        mission_id="mission_demo",
        task_label="search the house for people",
        mission_text="search the house for people",
        mission_type="person_search",
        search_scope="house",
        priority_regions=[
            build_search_region(
                region_id="entire_house",
                region_label="entire house",
                region_type="house",
                room_type="house",
                priority=4,
                status="unobserved",
                rationale="Global mission scope requires broad house-level search coverage.",
            )
        ],
        status="active",
    )
    search_runtime = build_search_runtime_state(
        mission_id="mission_demo",
        mission_type="person_search",
        mission_status="active",
        current_search_subgoal="search_house",
        priority_region=mission["priority_regions"][0],
        visited_region_count=0,
        suspect_region_count=0,
        confirmed_region_count=0,
        evidence_count=0,
        detection_state="searching",
        search_status="searching",
        confirm_target=False,
    )
    manual = build_phase5_mission_manual(
        task_label="search the house for people",
        mission=mission,
        search_runtime=search_runtime,
        doorway_runtime={"traversable_candidate_count": 1},
        depth_stats={"min_depth_cm": 80.0, "max_depth_cm": 1200.0},
    )
    print(json.dumps(manual, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
