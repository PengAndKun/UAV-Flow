"""
Launch and hold the Unreal UAV environment, then expose simple HTTP controls.
"""

import argparse
import base64
import copy
import json
import logging
import os
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request
from urllib.parse import urlparse

import cv2
import gym
import gym_unrealcv
import numpy as np

from archive_runtime import ArchiveRuntime
from batch_run_act_all import (
    configure_player_viewport,
    create_obj_if_needed,
    get_follow_preview_cam_id,
    get_policy_cam_id,
    get_third_person_preview_image,
    maybe_override_env_binary,
    set_cam,
    set_free_view_near_pose,
    validate_env_binary_exists,
)
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
from doorway_detection import detect_doorway_runtime
from language_search_memory import LanguageSearchMemory
from lesson4.depth_planar_pipeline import coerce_depth_planar_image, generate_camera_info
from phase5_mission_manual import build_phase5_mission_manual
from runtime_interfaces import (
    build_doorway_runtime_state,
    build_llm_action_request,
    build_llm_action_runtime_state,
    build_mission_state,
    build_plan_request,
    build_plan_state,
    build_planner_executor_runtime_state,
    build_person_evidence_runtime_state,
    build_reflex_request,
    build_reflex_runtime_state,
    build_reflex_sample,
    build_runtime_debug_state,
    build_search_region,
    build_search_result_state,
    build_search_runtime_state,
    build_waypoint,
    coerce_llm_action_runtime_payload,
    coerce_plan_payload,
    coerce_reflex_runtime_payload,
    now_timestamp,
)

logger = logging.getLogger(__name__)

REFLEX_ACTION_ALIASES = {
    "forward(w)": "forward",
    "backward(s)": "backward",
    "left(a)": "left",
    "right(d)": "right",
    "up(r)": "up",
    "down(f)": "down",
    "yaw_left(q)": "yaw_left",
    "yaw_right(e)": "yaw_right",
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "yaw_left": "yaw_left",
    "yaw_right": "yaw_right",
    "hold": "hold_position",
    "hold_position": "hold_position",
    "shield_hold": "shield_hold",
    "idle": "idle",
}

REFLEX_OPPOSITE_ACTIONS = {
    "forward": "backward",
    "backward": "forward",
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
    "yaw_left": "yaw_right",
    "yaw_right": "yaw_left",
}

DEFAULT_TAKEOVER_REASON = "manual_navigation"
MISSION_ROOM_HINTS = [
    (("bedroom", "bed room"), "bedroom", "bedroom"),
    (("kitchen",), "kitchen", "kitchen"),
    (("bathroom", "restroom"), "bathroom", "bathroom"),
    (("living room", "livingroom", "lounge"), "living room", "living_room"),
    (("hallway", "corridor"), "hallway", "hallway"),
    (("stairs", "stair", "staircase"), "stairs", "stairs"),
    (("door", "doorway", "entry"), "doorway", "doorway"),
    (("corner", "occluded corner"), "occluded corner", ""),
]


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def slugify_text(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return slug[:48] or "idle"


def normalize_reflex_action_name(action_name: Any) -> str:
    text = str(action_name or "idle").strip().lower()
    return REFLEX_ACTION_ALIASES.get(text, text or "idle")


def infer_mission_type_from_task_label(task_label: str) -> str:
    text = str(task_label or "").strip().lower()
    if not text:
        return "semantic_navigation"
    if any(keyword in text for keyword in ["confirm", "verify", "approach"]):
        return "target_verification"
    if any(keyword in text for keyword in ["person", "people", "human", "survivor", "victim"]):
        return "person_search"
    if any(keyword in text for keyword in ["search", "inspect", "find", "look for", "check"]):
        return "room_search"
    return "semantic_navigation"


def infer_search_scope_from_task_label(task_label: str) -> str:
    text = str(task_label or "").strip().lower()
    if any(keyword in text for keyword in ["house", "building", "entire", "whole"]):
        return "house"
    if any(keyword in text for keywords, _label, _room_type in MISSION_ROOM_HINTS for keyword in keywords):
        return "room"
    return "local"


def infer_priority_regions_from_task_label(task_label: str, mission_type: str) -> List[Dict[str, Any]]:
    text = str(task_label or "").strip().lower()
    priority_regions_with_pos: List[Tuple[int, Dict[str, Any]]] = []
    if any(keyword in text for keyword in ["house", "building", "entire home", "whole house", "whole home"]):
        priority_regions_with_pos.append(
            (
                text.find("house") if "house" in text else 0,
                build_search_region(
                    region_id="entire_house",
                    region_label="entire house",
                    region_type="house",
                    room_type="house",
                    priority=4,
                    status="unobserved",
                    rationale="Global mission scope derived from the task text.",
                ),
            )
        )
    for keywords, label, room_type in MISSION_ROOM_HINTS:
        matches = [text.find(keyword) for keyword in keywords if keyword in text]
        if matches:
            priority_regions_with_pos.append(
                (
                    min(matches),
                    build_search_region(
                        region_id=f"{slugify_text(label)}_{len(priority_regions_with_pos) + 1}",
                        region_label=label,
                        region_type="room" if room_type else "area",
                        room_type=room_type,
                        priority=0,
                        status="suspect" if mission_type in ("person_search", "target_verification") else "unobserved",
                        rationale=f"Derived from task text keywords={','.join(keywords)}.",
                    ),
                )
            )
    if any(keyword in text for keyword in ["suspect", "possible person", "possible target"]) and not any(
        region.get("region_label") == "suspect region" for _, region in priority_regions_with_pos
    ):
        priority_regions_with_pos.append(
            (
                max(0, text.find("suspect")),
                build_search_region(
                    region_id="suspect_region",
                    region_label="suspect region",
                    region_type="area",
                    priority=0,
                    status="suspect",
                    rationale="Explicit suspect cue from the mission text.",
                ),
            )
        )
    priority_regions_with_pos.sort(key=lambda item: item[0])
    priority_regions: List[Dict[str, Any]] = []
    for index, (_position, region) in enumerate(priority_regions_with_pos):
        region["priority"] = max(1, 4 - index)
        priority_regions.append(region)
    if not priority_regions and mission_type in ("person_search", "room_search", "target_verification"):
        priority_regions.append(
            build_search_region(
                region_id="forward_search_sector",
                region_label="forward search sector",
                region_type="sector",
                priority=2,
                status="unobserved",
                rationale="Default search frontier derived from the mission text.",
            )
        )
    return priority_regions


def build_obj_info(task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "obj_id" not in task_data or "use_obj" not in task_data:
        return None
    if "target_pos" in task_data and isinstance(task_data["target_pos"], list) and len(task_data["target_pos"]) == 6:
        obj_pos = task_data["target_pos"][:3]
        obj_rot = task_data["target_pos"][3:]
    else:
        obj_pos = task_data.get("obj_pos")
        obj_rot = task_data.get("obj_rot", [0, 0, 0])
    if obj_pos is None:
        return None
    return {
        "use_obj": task_data["use_obj"],
        "obj_id": task_data["obj_id"],
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
    }


def encode_image_b64(frame: np.ndarray, jpeg_quality: int = 90) -> str:
    encode_ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not encode_ok:
        raise RuntimeError("Failed to encode frame for planner payload")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def render_depth_preview(
    depth_image: np.ndarray,
    width: int,
    height: int,
    *,
    min_depth_cm: Optional[float] = None,
    max_depth_cm: Optional[float] = None,
    source_mode: str = "depth",
) -> np.ndarray:
    depth = coerce_depth_planar_image(depth_image)
    finite = depth[np.isfinite(depth)]
    canvas = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    canvas[:] = (12, 12, 18)

    if finite.size:
        preview_min = float(np.min(finite)) if min_depth_cm is None else float(min_depth_cm)
        preview_max = float(np.max(finite)) if max_depth_cm is None else float(max_depth_cm)
        if preview_max <= preview_min:
            preview_max = preview_min + 1.0
        valid_mask = np.isfinite(depth) & (depth >= preview_min) & (depth <= preview_max)
        clipped = np.clip(depth, preview_min, preview_max)
        normalized = 1.0 - ((clipped - preview_min) / (preview_max - preview_min))
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        preview_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        canvas = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
        canvas[~valid_mask] = (16, 16, 20)
    else:
        preview_min = 0.0
        preview_max = 0.0

    if width > 0 and height > 0 and (canvas.shape[1] != width or canvas.shape[0] != height):
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.putText(canvas, f"Depth mode: {source_mode}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"range(cm): {preview_min:.1f} -> {preview_max:.1f}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "near=warm  far=cool", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


class UAVControlBackend:
    """Own the Unreal environment and provide thread-safe control methods."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_depth_frame: Optional[np.ndarray] = None
        self.last_depth_summary: Dict[str, Any] = {
            "frame_id": "depth-init",
            "available": False,
            "min_depth": 0.0,
            "max_depth": 0.0,
            "image_width": 0,
            "image_height": 0,
            "fov_deg": 0.0,
            "pipeline": "lesson4_depth_image_proc_style",
            "camera_info": {},
        }
        self.last_capture: Optional[Dict[str, Any]] = None
        self.httpd: Optional[HTTPServer] = None
        self.command_task_yaw_deg: float = 0.0
        self.last_action: str = "idle"
        self.frame_index: int = 0
        self.last_observation_time: str = now_timestamp()
        self.last_observation_id: str = "frame_000000"
        self.current_task_label: str = args.default_task_label
        self.current_plan: Dict[str, Any] = build_plan_state(
            planner_name=args.planner_name,
            semantic_subgoal="idle",
        )
        self.current_mission: Dict[str, Any] = self.build_mission_descriptor(self.current_task_label)
        self.search_runtime: Dict[str, Any] = self.build_search_runtime_snapshot()
        self.doorway_runtime: Dict[str, Any] = build_doorway_runtime_state()
        self.last_plan_request: Dict[str, Any] = {}
        self.plan_execution_state: Dict[str, Any] = {
            "planner_status": "idle",
            "planner_source": "none",
            "planner_source_detail": "none",
            "planner_route_mode": "heuristic_only",
            "planner_route_reason": "",
            "last_trigger": "startup",
            "request_count": 0,
            "auto_request_count": 0,
            "last_latency_ms": 0.0,
            "last_error": "",
            "last_model_name": "",
            "last_api_style": "",
            "last_usage": {},
            "fallback_used": False,
            "fallback_reason": "",
            "step_index": 0,
            "auto_mode": args.planner_auto_mode,
            "auto_interval_steps": int(args.planner_interval_steps),
            "next_auto_trigger_step": 1 if args.planner_auto_mode == "k_step" else 0,
            "last_auto_trigger_step": 0,
        }
        self.runtime_debug: Dict[str, Any] = build_runtime_debug_state()
        self.reflex_runtime: Dict[str, Any] = build_reflex_runtime_state(
            mode="heuristic_stub",
            policy_name=self.args.reflex_policy_name,
            source="local_heuristic",
        )
        self.reflex_execution_state: Dict[str, Any] = {
            "mode": self.args.reflex_execute_mode,
            "last_status": "idle",
            "last_reason": "startup",
            "last_trigger": "startup",
            "last_requested_action": "idle",
            "last_executed_action": "idle",
            "last_source": "none",
            "last_step_index": 0,
            "execution_count": 0,
            "skipped_count": 0,
            "blocked_count": 0,
        }
        self.planner_executor_state: Dict[str, Any] = build_planner_executor_runtime_state(
            mode="manual",
            active=False,
            state="idle",
            mission_id=str(self.current_mission.get("mission_id", "")),
            current_plan_id=str(self.current_plan.get("plan_id", "")),
            current_search_subgoal=str(self.current_plan.get("search_subgoal", "idle")),
        )
        self.takeover_events: List[Dict[str, Any]] = []
        self.takeover_event_counter: int = 0
        self.takeover_state: Dict[str, Any] = {
            "active": False,
            "takeover_id": "",
            "started_at": "",
            "ended_at": "",
            "start_trigger": "",
            "current_reason": "",
            "current_note": "",
            "last_intervention_reason": "",
            "intervention_count": 0,
            "event_count": 0,
            "last_event_id": "",
            "last_event_type": "",
            "last_event_reason": "",
            "last_corrective_action": "",
            "log_path": "",
        }
        self.person_evidence_events: List[Dict[str, Any]] = []
        self.person_evidence_event_counter: int = 0
        self.person_evidence_runtime: Dict[str, Any] = build_person_evidence_runtime_state()
        self.search_result: Dict[str, Any] = build_search_result_state()
        self.archive_runtime = ArchiveRuntime(
            pos_bin_cm=float(self.args.archive_pos_bin_cm),
            yaw_bin_deg=float(self.args.archive_yaw_bin_deg),
            depth_bin_cm=float(self.args.archive_depth_bin_cm),
            recent_limit=int(self.args.archive_recent_limit),
        )
        self.language_search_memory = LanguageSearchMemory(
            recent_limit=max(6, int(self.args.search_recent_limit)),
            region_limit=max(4, int(self.args.archive_retrieval_limit)),
        )
        self.language_memory_runtime: Dict[str, Any] = self.language_search_memory.reset(
            mission_id=str(self.current_mission.get("mission_id", "")),
            mission_type=str(self.current_mission.get("mission_type", "semantic_navigation")),
            task_label=self.current_task_label,
        )
        self.phase5_mission_manual: Dict[str, Any] = build_phase5_mission_manual(
            task_label=self.current_task_label,
            mission=self.current_mission,
            search_runtime=self.search_runtime,
            person_evidence_runtime=self.person_evidence_runtime,
            language_memory_runtime=self.language_memory_runtime,
            doorway_runtime=self.doorway_runtime,
            depth_stats=self.last_depth_summary,
        )
        self.llm_action_runtime: Dict[str, Any] = build_llm_action_runtime_state(
            policy_name=self.args.planner_name,
            source="none",
            status="idle",
        )
        self.last_llm_action_request: Dict[str, Any] = {}
        self.recent_action_history: List[Dict[str, Any]] = []

        os.makedirs(self.args.capture_dir, exist_ok=True)
        os.makedirs(self.args.takeover_log_dir, exist_ok=True)
        os.makedirs(self.args.search_log_dir, exist_ok=True)
        self.takeover_log_path = os.path.join(
            self.args.takeover_log_dir,
            f"takeover_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        self.takeover_state["log_path"] = self.takeover_log_path
        self.search_log_path = os.path.join(
            self.args.search_log_dir,
            f"search_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        self.fixed_spawn_pose_path = self.args.fixed_spawn_pose_file

        maybe_override_env_binary(self.args.env_id, self.args.env_bin_win)
        validate_env_binary_exists(self.args.env_id)

        self.env = gym.make(self.args.env_id)
        if int(self.args.time_dilation) > 0:
            self.env = time_dilation.TimeDilationWrapper(self.env, int(self.args.time_dilation))
        self.env.unwrapped.agents_category = ["drone"]
        self.env = configUE.ConfigUEWrapper(self.env, resolution=(self.args.window_width, self.args.window_height))
        self.env = augmentation.RandomPopulationWrapper(self.env, 2, 2, random_target=False)
        self.env.seed(int(self.args.seed))
        self.env.reset()

        self.player_name = self.env.unwrapped.player_list[0]
        self.env.unwrapped.unrealcv.set_phy(self.player_name, 0)
        logger.info("Active player list after reset: %s", self.env.unwrapped.player_list)
        self.hide_non_primary_agents()

        configure_player_viewport(
            self.env,
            self.args.viewport_mode,
            (self.args.viewport_offset_x, self.args.viewport_offset_y, self.args.viewport_offset_z),
            (self.args.viewport_roll, self.args.viewport_pitch, self.args.viewport_yaw),
        )

        self.policy_cam_id = get_policy_cam_id(self.env)
        self.preview_cam_id = get_follow_preview_cam_id(self.env, self.policy_cam_id)
        self.free_view_offset = (
            self.args.free_view_offset_x,
            self.args.free_view_offset_y,
            self.args.free_view_offset_z,
        )
        self.free_view_rotation = (
            self.args.free_view_roll,
            self.args.free_view_pitch,
            self.args.free_view_yaw,
        )
        self.preview_offset = (
            self.args.preview_offset_x,
            self.args.preview_offset_y,
            self.args.preview_offset_z,
        )
        self.preview_rotation = (
            self.args.preview_roll,
            self.args.preview_pitch,
            self.args.preview_yaw,
        )

        self.apply_initial_task_or_spawn()
        self.command_task_yaw_deg = self.get_task_pose()[3]
        self.position_free_view_once()
        self.refresh_observations()
        self.reset_person_search_state(reset_recent_events=True)
        if self.args.reflex_policy_url:
            try:
                reflex_runtime = self.request_reflex_policy(trigger="startup")
                logger.info(
                    "Startup reflex policy synced policy=%s source=%s suggested=%s",
                    reflex_runtime.get("policy_name", "n/a"),
                    reflex_runtime.get("source", "unknown"),
                    reflex_runtime.get("suggested_action", "idle"),
                )
            except Exception as exc:
                logger.warning("Startup reflex policy sync failed, keep local heuristic state: %s", exc)

    def build_mission_descriptor(self, task_label: str, *, previous_mission_id: str = "") -> Dict[str, Any]:
        normalized_task = str(task_label or "").strip()
        mission_type = infer_mission_type_from_task_label(normalized_task)
        mission_id = previous_mission_id or f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify_text(normalized_task)}"
        priority_regions = infer_priority_regions_from_task_label(normalized_task, mission_type)
        target_type = "search_result" if mission_type in ("person_search", "room_search", "target_verification") else "waypoint"
        success_criteria: List[str]
        if mission_type == "person_search":
            success_criteria = ["decide_if_person_exists", "estimate_person_location", "collect_supporting_evidence"]
        elif mission_type == "target_verification":
            success_criteria = ["approach_suspect_region", "confirm_or_reject_target", "record_confirmation_evidence"]
        elif mission_type == "room_search":
            success_criteria = ["search_requested_region", "mark_region_coverage", "prepare_follow_up_waypoint"]
        else:
            success_criteria = ["reach_semantic_waypoint", "maintain_safe_motion"]
        return build_mission_state(
            mission_id=mission_id,
            task_label=normalized_task,
            mission_text=normalized_task or "idle",
            mission_type=mission_type,
            target_type=target_type,
            search_scope=infer_search_scope_from_task_label(normalized_task),
            priority_regions=priority_regions,
            confirm_target=mission_type == "target_verification",
            success_criteria=success_criteria,
            constraints=["avoid_collision", "respect_takeover", "low_latency_execute"],
            status="active" if normalized_task else "idle",
        )

    def build_search_runtime_snapshot(self, archive_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        archive_payload = archive_state if isinstance(archive_state, dict) else {}
        plan_runtime = self.plan_execution_state if isinstance(getattr(self, "plan_execution_state", None), dict) else {}
        candidate_regions = self.current_plan.get("candidate_regions") if isinstance(self.current_plan.get("candidate_regions"), list) else []
        priority_region = self.current_plan.get("priority_region") if isinstance(self.current_plan.get("priority_region"), dict) else {}
        mission_priority_regions = self.current_mission.get("priority_regions") if isinstance(self.current_mission.get("priority_regions"), list) else []
        person_evidence = (
            getattr(self, "person_evidence_runtime", {})
            if isinstance(getattr(self, "person_evidence_runtime", {}), dict)
            else {}
        )
        search_result = getattr(self, "search_result", {}) if isinstance(getattr(self, "search_result", {}), dict) else {}
        if not candidate_regions and mission_priority_regions:
            candidate_regions = mission_priority_regions
        if not priority_region and candidate_regions:
            priority_region = candidate_regions[0]
        visited_region_count = int(archive_payload.get("cell_count", 0)) if archive_payload else 0
        suspect_region_count = sum(1 for region in candidate_regions if str(region.get("status", "")).lower() == "suspect")
        confirmed_region_count = sum(1 for region in candidate_regions if str(region.get("status", "")).lower() == "confirmed")
        plan_status = str(plan_runtime.get("planner_status", "idle") or "idle")
        evidence_status = str(person_evidence.get("evidence_status", "idle") or "idle")
        result_status = str(search_result.get("result_status", "unknown") or "unknown")
        if result_status == "person_detected":
            detection_state = "confirmed_present"
        elif result_status == "no_person_confirmed":
            detection_state = "confirmed_absent"
        elif evidence_status == "suspect":
            detection_state = "suspect"
        elif self.current_mission.get("mission_type") in ("person_search", "room_search", "target_verification"):
            detection_state = "confirming" if bool(self.current_plan.get("confirm_target", False)) else "searching"
        else:
            detection_state = "navigation"
        return build_search_runtime_state(
            mission_id=str(self.current_mission.get("mission_id", "")),
            mission_type=str(self.current_mission.get("mission_type", "semantic_navigation")),
            mission_status=str(self.current_mission.get("status", "idle")),
            current_search_subgoal=str(
                self.current_plan.get("search_subgoal", self.current_plan.get("semantic_subgoal", "idle")) or "idle"
            ),
            priority_region=priority_region,
            candidate_regions=candidate_regions,
            visited_region_count=visited_region_count,
            suspect_region_count=suspect_region_count,
            confirmed_region_count=confirmed_region_count,
            evidence_count=int(person_evidence.get("evidence_event_count", 0)),
            detection_state=detection_state,
            search_status=detection_state,
            confirm_target=bool(self.current_plan.get("confirm_target", self.current_mission.get("confirm_target", False))),
            estimated_person_position=search_result.get("estimated_person_position", {})
            or person_evidence.get("estimated_person_position", {}),
            last_reasoning=str(self.current_plan.get("explanation", "")),
            replan_count=int(plan_runtime.get("request_count", 0)) if plan_status != "idle" else 0,
        )

    def sync_search_runtime(self, archive_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.search_runtime = self.build_search_runtime_snapshot(archive_state)
        self.sync_language_memory(archive_state)
        self.sync_phase5_mission_manual()
        return self.search_runtime

    def sync_language_memory(self, archive_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        archive_snapshot = archive_state
        if archive_snapshot is None:
            archive_snapshot = self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
        self.language_memory_runtime = self.language_search_memory.sync(
            mission=self.current_mission,
            search_runtime=self.search_runtime,
            person_evidence_runtime=self.person_evidence_runtime,
            search_result=self.search_result,
            archive_state=archive_snapshot,
            doorway_runtime=self.doorway_runtime,
            current_plan=self.current_plan,
        )
        return self.language_memory_runtime

    def sync_doorway_runtime(self) -> Dict[str, Any]:
        self.doorway_runtime = detect_doorway_runtime(
            rgb_frame=self.last_raw_frame,
            depth_frame=self.last_depth_frame,
            depth_summary=self.last_depth_summary,
        )
        return self.doorway_runtime

    def sync_phase5_mission_manual(self) -> Dict[str, Any]:
        self.phase5_mission_manual = build_phase5_mission_manual(
            task_label=self.current_task_label,
            mission=self.current_mission,
            search_runtime=self.search_runtime,
            person_evidence_runtime=self.person_evidence_runtime,
            language_memory_runtime=self.language_memory_runtime,
            doorway_runtime=self.doorway_runtime,
            depth_stats=self.last_depth_summary,
        )
        return self.phase5_mission_manual

    def sync_mission_from_plan(self) -> Dict[str, Any]:
        candidate_regions = self.current_plan.get("candidate_regions") if isinstance(self.current_plan.get("candidate_regions"), list) else []
        if candidate_regions:
            self.current_mission["priority_regions"] = candidate_regions
        self.current_mission["confirm_target"] = bool(self.current_plan.get("confirm_target", self.current_mission.get("confirm_target", False)))
        if self.current_task_label:
            self.current_mission["status"] = "active"
        return self.current_mission

    def build_default_person_evidence_runtime(self) -> Dict[str, Any]:
        mission_type = str(self.current_mission.get("mission_type", "semantic_navigation"))
        initial_status = "searching" if mission_type in ("person_search", "room_search", "target_verification") else "idle"
        return build_person_evidence_runtime_state(
            mission_id=str(self.current_mission.get("mission_id", "")),
            mission_type=mission_type,
            evidence_status=initial_status,
        )

    def build_default_search_result(self) -> Dict[str, Any]:
        mission_type = str(self.current_mission.get("mission_type", "semantic_navigation"))
        if mission_type in ("person_search", "room_search", "target_verification"):
            result_status = "unknown"
            summary = "No person evidence has been confirmed yet."
        else:
            result_status = "not_applicable"
            summary = "Mission type is not person-search oriented."
        return build_search_result_state(
            mission_id=str(self.current_mission.get("mission_id", "")),
            mission_type=mission_type,
            result_status=result_status,
            person_exists=None,
            summary=summary,
        )

    def reset_person_search_state(self, *, reset_recent_events: bool = False) -> None:
        self.person_evidence_runtime = self.build_default_person_evidence_runtime()
        self.search_result = self.build_default_search_result()
        if reset_recent_events:
            self.person_evidence_events = []
        self.sync_search_runtime()

    def resolve_active_region_for_evidence(self) -> Dict[str, Any]:
        runtime_priority = self.search_runtime.get("priority_region") if isinstance(self.search_runtime.get("priority_region"), dict) else {}
        if runtime_priority and (
            str(runtime_priority.get("region_label", "")).strip() or str(runtime_priority.get("region_id", "")).strip()
        ):
            return dict(runtime_priority)
        plan_priority = self.current_plan.get("priority_region") if isinstance(self.current_plan.get("priority_region"), dict) else {}
        if plan_priority and (
            str(plan_priority.get("region_label", "")).strip() or str(plan_priority.get("region_id", "")).strip()
        ):
            return dict(plan_priority)
        mission_regions = self.current_mission.get("priority_regions") if isinstance(self.current_mission.get("priority_regions"), list) else []
        if mission_regions:
            return dict(mission_regions[0])
        return build_search_region(
            region_id="current_view_sector",
            region_label="current view sector",
            region_type="sector",
            priority=1,
            status="unobserved",
            rationale="Fallback evidence region derived from the current local view.",
        )

    def update_region_status(self, region_label: str, status: str, rationale: str) -> None:
        region_label_norm = str(region_label or "").strip().lower()
        if not region_label_norm:
            return
        payloads: List[Any] = [
            self.current_mission.get("priority_regions"),
            self.current_plan.get("candidate_regions"),
        ]
        for regions in payloads:
            if not isinstance(regions, list):
                continue
            for region in regions:
                if not isinstance(region, dict):
                    continue
                label = str(region.get("region_label", "")).strip().lower()
                if label == region_label_norm:
                    region["status"] = status
                    if rationale:
                        region["rationale"] = rationale
        if isinstance(self.current_plan.get("priority_region"), dict):
            plan_priority = self.current_plan["priority_region"]
            if str(plan_priority.get("region_label", "")).strip().lower() == region_label_norm:
                plan_priority["status"] = status
                if rationale:
                    plan_priority["rationale"] = rationale

    def build_estimated_person_position(self) -> Dict[str, Any]:
        task_pose = self.get_task_pose()
        return {
            "x": float(task_pose[0]),
            "y": float(task_pose[1]),
            "z": float(task_pose[2]),
            "yaw": float(task_pose[3]),
            "frame_id": self.last_observation_id,
        }

    def build_person_evidence_snapshot(self) -> Dict[str, Any]:
        task_pose = self.get_task_pose()
        priority_region = self.resolve_active_region_for_evidence()
        return {
            "timestamp": now_timestamp(),
            "step_index": int(self.plan_execution_state.get("step_index", 0)),
            "task_label": self.current_task_label,
            "mission": {
                "mission_id": str(self.current_mission.get("mission_id", "")),
                "mission_type": str(self.current_mission.get("mission_type", "")),
                "status": str(self.current_mission.get("status", "idle")),
            },
            "search_runtime": {
                "current_search_subgoal": str(self.search_runtime.get("current_search_subgoal", "idle")),
                "detection_state": str(self.search_runtime.get("detection_state", "unknown")),
                "priority_region": priority_region,
                "evidence_count": int(self.search_runtime.get("evidence_count", 0)),
            },
            "pose": {
                "x": float(task_pose[0]),
                "y": float(task_pose[1]),
                "z": float(task_pose[2]),
                "task_yaw": float(task_pose[3]),
                "command_yaw": float(self.command_task_yaw_deg),
                "uav_yaw": float(self.get_env_yaw_deg()),
            },
            "depth": {
                "frame_id": str(self.last_depth_summary.get("frame_id", "")),
                "min_depth": float(self.last_depth_summary.get("min_depth", 0.0)),
                "max_depth": float(self.last_depth_summary.get("max_depth", 0.0)),
            },
            "plan": {
                "semantic_subgoal": str(self.current_plan.get("semantic_subgoal", "idle")),
                "search_subgoal": str(self.current_plan.get("search_subgoal", "idle")),
                "planner_confidence": float(self.current_plan.get("planner_confidence", 0.0)),
            },
            "reflex": {
                "suggested_action": str(self.reflex_runtime.get("suggested_action", "idle")),
                "policy_confidence": float(self.reflex_runtime.get("policy_confidence", 0.0)),
                "status": str(self.reflex_runtime.get("status", "idle")),
            },
            "runtime_debug": {
                "risk_score": float(self.runtime_debug.get("risk_score", 0.0)),
                "archive_cell_id": str(self.runtime_debug.get("archive_cell_id", "")),
            },
        }

    def append_person_evidence_event(
        self,
        *,
        event_type: str,
        reason: str,
        note: str,
        confidence: float,
        region: Dict[str, Any],
        capture_label: str = "",
        before_snapshot: Optional[Dict[str, Any]] = None,
        after_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.person_evidence_event_counter += 1
        event = {
            "event_id": f"search_evt_{self.person_evidence_event_counter:05d}",
            "timestamp": now_timestamp(),
            "event_type": str(event_type or ""),
            "reason": str(reason or ""),
            "note": str(note or ""),
            "confidence": float(confidence),
            "task_label": self.current_task_label,
            "mission_id": str(self.current_mission.get("mission_id", "")),
            "mission_type": str(self.current_mission.get("mission_type", "")),
            "step_index": int(self.plan_execution_state.get("step_index", 0)),
            "capture_label": str(capture_label or ""),
            "region": copy.deepcopy(region),
            "before_runtime": copy.deepcopy(before_snapshot),
            "after_runtime": copy.deepcopy(after_snapshot),
            "search_result": copy.deepcopy(self.search_result),
        }
        self.person_evidence_events.append(event)
        limit = max(1, int(self.args.search_recent_limit))
        if len(self.person_evidence_events) > limit:
            self.person_evidence_events = self.person_evidence_events[-limit:]
        with open(self.search_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=True) + "\n")
        return event

    def record_person_evidence(
        self,
        *,
        action: str,
        note: str = "",
        capture_label: str = "",
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {"suspect", "confirm_present", "confirm_absent", "reset"}:
            raise ValueError(f"Unsupported person evidence action: {action}")
        before_snapshot = self.build_person_evidence_snapshot()
        region = self.resolve_active_region_for_evidence()
        region_label = str(region.get("region_label", "current view sector") or "current view sector")
        reason = note or normalized_action
        if normalized_action == "reset":
            self.reset_person_search_state(reset_recent_events=True)
            self.person_evidence_runtime["last_event_type"] = "reset"
            self.person_evidence_runtime["last_reason"] = reason
            self.person_evidence_runtime["last_note"] = str(note or "")
            self.person_evidence_runtime["last_updated_at"] = now_timestamp()
            self.search_result = self.build_default_search_result()
            confidence_value = 0.0
        elif normalized_action == "suspect":
            confidence_value = float(confidence if confidence is not None else 0.55)
            self.person_evidence_runtime["evidence_status"] = "suspect"
            self.person_evidence_runtime["suspect_count"] = int(self.person_evidence_runtime.get("suspect_count", 0)) + 1
            self.person_evidence_runtime["confidence"] = confidence_value
            self.person_evidence_runtime["suspect_region"] = region
            self.person_evidence_runtime["last_event_type"] = "suspect"
            self.person_evidence_runtime["last_reason"] = reason
            self.person_evidence_runtime["last_note"] = str(note or "")
            self.person_evidence_runtime["last_updated_at"] = now_timestamp()
            self.update_region_status(region_label, "suspect", "Marked as suspect by manual evidence annotation.")
            self.search_result["result_status"] = "unknown"
            self.search_result["person_exists"] = None
            self.search_result["confidence"] = max(float(self.search_result.get("confidence", 0.0)), confidence_value)
            self.search_result["summary"] = f"Suspect evidence recorded for {region_label}."
            self.search_result["last_updated_at"] = now_timestamp()
        elif normalized_action == "confirm_present":
            confidence_value = float(confidence if confidence is not None else 0.9)
            estimated_position = self.build_estimated_person_position()
            self.person_evidence_runtime["evidence_status"] = "confirmed_present"
            self.person_evidence_runtime["confirm_present_count"] = int(self.person_evidence_runtime.get("confirm_present_count", 0)) + 1
            self.person_evidence_runtime["confidence"] = confidence_value
            self.person_evidence_runtime["suspect_region"] = region
            self.person_evidence_runtime["estimated_person_position"] = estimated_position
            self.person_evidence_runtime["last_event_type"] = "confirm_present"
            self.person_evidence_runtime["last_reason"] = reason
            self.person_evidence_runtime["last_note"] = str(note or "")
            self.person_evidence_runtime["last_updated_at"] = now_timestamp()
            self.update_region_status(region_label, "confirmed", "Person presence confirmed by manual evidence annotation.")
            self.search_result["result_status"] = "person_detected"
            self.search_result["person_exists"] = True
            self.search_result["confidence"] = confidence_value
            self.search_result["estimated_person_position"] = estimated_position
            self.search_result["summary"] = f"Person confirmed near {region_label}."
            self.search_result["last_updated_at"] = now_timestamp()
        else:
            confidence_value = float(confidence if confidence is not None else 0.85)
            self.person_evidence_runtime["evidence_status"] = "confirmed_absent"
            self.person_evidence_runtime["confirm_absent_count"] = int(self.person_evidence_runtime.get("confirm_absent_count", 0)) + 1
            self.person_evidence_runtime["confidence"] = confidence_value
            self.person_evidence_runtime["suspect_region"] = region
            self.person_evidence_runtime["last_event_type"] = "confirm_absent"
            self.person_evidence_runtime["last_reason"] = reason
            self.person_evidence_runtime["last_note"] = str(note or "")
            self.person_evidence_runtime["last_updated_at"] = now_timestamp()
            self.update_region_status(region_label, "confirmed", "Region cleared by manual no-person confirmation.")
            if str(self.search_result.get("result_status", "unknown")) != "person_detected":
                self.search_result["result_status"] = "no_person_confirmed"
                self.search_result["person_exists"] = False
                self.search_result["confidence"] = confidence_value
                self.search_result["summary"] = f"No person confirmed in {region_label}."
                self.search_result["last_updated_at"] = now_timestamp()
        evidence_capture_ids = list(self.person_evidence_runtime.get("evidence_capture_ids", []))
        supporting_capture_ids = list(self.search_result.get("supporting_capture_ids", []))
        if capture_label:
            evidence_capture_ids.append(str(capture_label))
            supporting_capture_ids.append(str(capture_label))
        self.person_evidence_runtime["evidence_capture_ids"] = evidence_capture_ids[-12:]
        self.search_result["supporting_capture_ids"] = supporting_capture_ids[-12:]
        self.person_evidence_runtime["evidence_event_count"] = int(self.person_evidence_runtime.get("evidence_event_count", 0)) + 1
        self.sync_search_runtime()
        after_snapshot = self.build_person_evidence_snapshot()
        event = self.append_person_evidence_event(
            event_type=normalized_action,
            reason=reason,
            note=str(note or ""),
            confidence=confidence_value,
            region=region,
            capture_label=capture_label,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
        self.language_search_memory.record_evidence(
            event_type=normalized_action,
            note=str(note or ""),
            region=region,
            confidence=confidence_value,
            search_result=self.search_result,
        )
        self.sync_language_memory()
        logger.info(
            "Person evidence action=%s region=%s confidence=%.2f mission=%s",
            normalized_action,
            region_label,
            confidence_value,
            self.current_mission.get("mission_type", "n/a"),
        )
        return {
            "status": "ok",
            "person_evidence_runtime": self.person_evidence_runtime,
            "search_result": self.search_result,
            "person_evidence_recent_events": self.person_evidence_events,
            "event": event,
            "state": self.get_state(),
        }

    def apply_initial_task_or_spawn(self) -> None:
        if self.args.task_json:
            with open(self.args.task_json, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            obj_info = build_obj_info(task_data)
            create_obj_if_needed(self.env, obj_info)
            initial_pos = task_data.get("initial_pos")
            if isinstance(initial_pos, list) and len(initial_pos) >= 5:
                self.set_task_pose(initial_pos[0:3], float(initial_pos[4]))
                logger.info("Loaded initial pose from task JSON: %s", initial_pos[:5])
                return

        if self.args.spawn_x is not None and self.args.spawn_y is not None and self.args.spawn_z is not None:
            spawn_yaw = self.args.spawn_yaw if self.args.spawn_yaw is not None else self.get_task_pose()[3]
            self.set_task_pose(
                [self.args.spawn_x, self.args.spawn_y, self.args.spawn_z],
                spawn_yaw,
            )
            logger.info(
                "Applied manual spawn pose: x=%.3f y=%.3f z=%.3f yaw=%.3f",
                self.args.spawn_x,
                self.args.spawn_y,
                self.args.spawn_z,
                spawn_yaw,
            )
            self.save_fixed_spawn_pose(
                {
                    "x": float(self.args.spawn_x),
                    "y": float(self.args.spawn_y),
                    "z": float(self.args.spawn_z),
                    "yaw": float(spawn_yaw),
                }
            )
            return

        fixed_spawn_pose = self.load_fixed_spawn_pose()
        if fixed_spawn_pose is not None:
            self.set_task_pose(
                [fixed_spawn_pose["x"], fixed_spawn_pose["y"], fixed_spawn_pose["z"]],
                fixed_spawn_pose["yaw"],
            )
            logger.info(
                "Applied fixed spawn pose from file: x=%.3f y=%.3f z=%.3f yaw=%.3f path=%s",
                fixed_spawn_pose["x"],
                fixed_spawn_pose["y"],
                fixed_spawn_pose["z"],
                fixed_spawn_pose["yaw"],
                self.fixed_spawn_pose_path,
            )
            return

        current_pose = self.get_task_pose()
        self.save_fixed_spawn_pose(
            {
                "x": float(current_pose[0]),
                "y": float(current_pose[1]),
                "z": float(current_pose[2]),
                "yaw": float(current_pose[3]),
            }
        )
        logger.info(
            "Initialized fixed spawn pose file from current reset pose: x=%.3f y=%.3f z=%.3f yaw=%.3f path=%s",
            current_pose[0],
            current_pose[1],
            current_pose[2],
            current_pose[3],
            self.fixed_spawn_pose_path,
        )

    def hide_non_primary_agents(self) -> None:
        extra_players = list(self.env.unwrapped.player_list[1:])
        for idx, player_name in enumerate(extra_players, start=1):
            try:
                hide_pos = [0.0, 0.0, -10000.0 - 100.0 * idx]
                self.env.unwrapped.unrealcv.set_phy(player_name, 0)
                self.env.unwrapped.unrealcv.set_obj_location(player_name, hide_pos)
                self.env.unwrapped.unrealcv.set_obj_rotation(player_name, [0.0, 0.0, 0.0])
                logger.info("Hid extra UAV agent %s at %s", player_name, hide_pos)
            except Exception as exc:
                logger.warning("Failed to hide extra UAV agent %s: %s", player_name, exc)

    def should_preserve_reflex_runtime(self) -> bool:
        """Return True when the current reflex state comes from a non-local source."""
        source = str(self.reflex_runtime.get("source", "") or "").strip().lower()
        if not self.args.reflex_policy_url:
            return False
        return bool(source) and source != "local_heuristic"

    def build_takeover_snapshot(self) -> Dict[str, Any]:
        task_pose = self.get_task_pose()
        waypoint = self.runtime_debug.get("current_waypoint")
        return {
            "timestamp": now_timestamp(),
            "step_index": int(self.plan_execution_state.get("step_index", 0)),
            "task_label": self.current_task_label,
            "last_action": self.last_action,
            "pose": {
                "x": float(task_pose[0]),
                "y": float(task_pose[1]),
                "z": float(task_pose[2]),
                "task_yaw": float(task_pose[3]),
                "command_yaw": float(self.command_task_yaw_deg),
                "uav_yaw": float(self.get_env_yaw_deg()),
            },
            "plan": {
                "semantic_subgoal": str(self.current_plan.get("semantic_subgoal", "idle")),
                "planner_name": str(self.current_plan.get("planner_name", "")),
                "planner_confidence": float(self.current_plan.get("planner_confidence", 0.0)),
                "planner_status": str(self.plan_execution_state.get("planner_status", "idle")),
                "planner_source": str(self.plan_execution_state.get("planner_source", "none")),
            },
            "reflex": {
                "mode": str(self.reflex_runtime.get("mode", "")),
                "policy_name": str(self.reflex_runtime.get("policy_name", "")),
                "source": str(self.reflex_runtime.get("source", "")),
                "suggested_action": str(self.reflex_runtime.get("suggested_action", "idle")),
                "should_execute": bool(self.reflex_runtime.get("should_execute", False)),
                "policy_confidence": float(self.reflex_runtime.get("policy_confidence", 0.0)),
                "status": str(self.reflex_runtime.get("status", "idle")),
            },
            "executor": {
                "mode": str(self.reflex_execution_state.get("mode", "manual")),
                "last_status": str(self.reflex_execution_state.get("last_status", "idle")),
                "last_reason": str(self.reflex_execution_state.get("last_reason", "")),
                "last_requested_action": str(self.reflex_execution_state.get("last_requested_action", "idle")),
                "last_executed_action": str(self.reflex_execution_state.get("last_executed_action", "")),
            },
            "runtime_debug": {
                "risk_score": float(self.runtime_debug.get("risk_score", 0.0)),
                "shield_triggered": bool(self.runtime_debug.get("shield_triggered", False)),
                "archive_cell_id": str(self.runtime_debug.get("archive_cell_id", "")),
                "current_waypoint": waypoint if isinstance(waypoint, dict) else None,
            },
        }

    def load_fixed_spawn_pose(self) -> Optional[Dict[str, float]]:
        if not self.fixed_spawn_pose_path:
            return None
        if not os.path.exists(self.fixed_spawn_pose_path):
            return None
        try:
            with open(self.fixed_spawn_pose_path, "r", encoding="utf-8") as pose_file:
                payload = json.load(pose_file)
            if not isinstance(payload, dict):
                return None
            if not all(key in payload for key in ("x", "y", "z", "yaw")):
                return None
            return {
                "x": float(payload["x"]),
                "y": float(payload["y"]),
                "z": float(payload["z"]),
                "yaw": float(payload["yaw"]),
            }
        except Exception as exc:
            logger.warning("Failed to load fixed spawn pose from %s: %s", self.fixed_spawn_pose_path, exc)
            return None

    def save_fixed_spawn_pose(self, pose: Dict[str, float]) -> None:
        if not self.fixed_spawn_pose_path:
            return
        payload = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "yaw": float(pose["yaw"]),
        }
        with open(self.fixed_spawn_pose_path, "w", encoding="utf-8") as pose_file:
            json.dump(payload, pose_file, indent=2)

    def append_takeover_event(
        self,
        *,
        event_type: str,
        reason: str,
        trigger: str,
        corrective_action: str = "",
        note: str = "",
        before_snapshot: Optional[Dict[str, Any]] = None,
        after_snapshot: Optional[Dict[str, Any]] = None,
        post_assist_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.takeover_event_counter += 1
        event = {
            "event_id": f"takeover_evt_{self.takeover_event_counter:05d}",
            "timestamp": now_timestamp(),
            "event_type": str(event_type),
            "trigger": str(trigger),
            "reason": str(reason or DEFAULT_TAKEOVER_REASON),
            "note": str(note or ""),
            "corrective_action": normalize_reflex_action_name(corrective_action),
            "takeover_id": str(self.takeover_state.get("takeover_id", "")),
            "task_label": self.current_task_label,
            "step_index": int(self.plan_execution_state.get("step_index", 0)),
            "before_runtime": before_snapshot,
            "after_runtime": after_snapshot,
        }
        if post_assist_snapshot is not None:
            event["post_assist_runtime"] = post_assist_snapshot
        self.takeover_events.append(event)
        limit = max(1, int(self.args.takeover_recent_limit))
        if len(self.takeover_events) > limit:
            self.takeover_events = self.takeover_events[-limit:]
        with open(self.takeover_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=True) + "\n")
        self.takeover_state["event_count"] = int(self.takeover_state.get("event_count", 0)) + 1
        self.takeover_state["last_event_id"] = event["event_id"]
        self.takeover_state["last_event_type"] = event["event_type"]
        self.takeover_state["last_event_reason"] = event["reason"]
        self.takeover_state["last_corrective_action"] = event["corrective_action"]
        if event_type == "manual_intervention":
            self.takeover_state["last_intervention_reason"] = event["reason"]
        return event

    def infer_manual_intervention_reason(self, action_name: str) -> str:
        manual_action = normalize_reflex_action_name(action_name)
        suggested_action = normalize_reflex_action_name(self.reflex_runtime.get("suggested_action", "idle"))
        last_executor_status = str(self.reflex_execution_state.get("last_status", "idle"))
        last_executor_reason = str(self.reflex_execution_state.get("last_reason", ""))
        if last_executor_status == "blocked" and last_executor_reason:
            return f"post_block:{last_executor_reason}"
        if suggested_action in ("idle", "hold_position", "shield_hold", ""):
            return DEFAULT_TAKEOVER_REASON
        if manual_action == suggested_action:
            return "confirm_reflex_suggestion"
        if REFLEX_OPPOSITE_ACTIONS.get(manual_action) == suggested_action:
            return "override_opposite_reflex"
        return "override_reflex_suggestion"

    def start_takeover(self, *, reason: str, note: str = "", trigger: str = "manual_start") -> Dict[str, Any]:
        if self.takeover_state.get("active", False):
            if note:
                self.takeover_state["current_note"] = str(note)
            return self.takeover_state
        started_at = now_timestamp()
        takeover_id = f"takeover_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(self.plan_execution_state.get('step_index', 0)):04d}"
        self.takeover_state.update(
            {
                "active": True,
                "takeover_id": takeover_id,
                "started_at": started_at,
                "ended_at": "",
                "start_trigger": str(trigger),
                "current_reason": str(reason or DEFAULT_TAKEOVER_REASON),
                "current_note": str(note or ""),
                "intervention_count": 0,
            }
        )
        snapshot = self.build_takeover_snapshot()
        self.append_takeover_event(
            event_type="takeover_start",
            reason=str(reason or DEFAULT_TAKEOVER_REASON),
            trigger=trigger,
            note=note,
            before_snapshot=snapshot,
            after_snapshot=snapshot,
        )
        logger.info("Takeover started id=%s reason=%s trigger=%s", takeover_id, reason or DEFAULT_TAKEOVER_REASON, trigger)
        return self.takeover_state

    def end_takeover(self, *, reason: str = "resolved", note: str = "", trigger: str = "manual_end") -> Dict[str, Any]:
        if not self.takeover_state.get("active", False):
            return self.takeover_state
        snapshot = self.build_takeover_snapshot()
        self.takeover_state["active"] = False
        self.takeover_state["ended_at"] = now_timestamp()
        if reason:
            self.takeover_state["current_reason"] = str(reason)
        if note:
            self.takeover_state["current_note"] = str(note)
        self.append_takeover_event(
            event_type="takeover_end",
            reason=str(reason or "resolved"),
            trigger=trigger,
            note=note,
            before_snapshot=snapshot,
            after_snapshot=snapshot,
        )
        logger.info("Takeover ended id=%s reason=%s trigger=%s", self.takeover_state.get("takeover_id", ""), reason or "resolved", trigger)
        return self.takeover_state

    def record_manual_intervention(
        self,
        *,
        action_name: str,
        before_snapshot: Dict[str, Any],
        after_snapshot: Dict[str, Any],
        post_assist_snapshot: Optional[Dict[str, Any]] = None,
        trigger: str = "manual_action",
    ) -> Dict[str, Any]:
        reason = self.infer_manual_intervention_reason(action_name)
        if not self.takeover_state.get("active", False):
            self.start_takeover(reason=reason, trigger="auto_manual_intervention")
        self.takeover_state["intervention_count"] = int(self.takeover_state.get("intervention_count", 0)) + 1
        event = self.append_takeover_event(
            event_type="manual_intervention",
            reason=reason,
            trigger=trigger,
            corrective_action=action_name,
            note=str(self.takeover_state.get("current_note", "")),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            post_assist_snapshot=post_assist_snapshot,
        )
        logger.info(
            "Takeover intervention action=%s reason=%s step=%s",
            normalize_reflex_action_name(action_name),
            reason,
            self.plan_execution_state.get("step_index", 0),
        )
        return event

    def update_reflex_execution_state(
        self,
        *,
        status: str,
        reason: str,
        trigger: str,
        requested_action: str,
        executed_action: str = "",
        source: str = "",
    ) -> Dict[str, Any]:
        execution_state = self.reflex_execution_state
        execution_state["mode"] = self.args.reflex_execute_mode
        execution_state["last_status"] = str(status)
        execution_state["last_reason"] = str(reason)
        execution_state["last_trigger"] = str(trigger)
        execution_state["last_requested_action"] = str(requested_action or "idle")
        execution_state["last_executed_action"] = str(executed_action or "")
        execution_state["last_source"] = str(source or "")
        execution_state["last_step_index"] = int(self.plan_execution_state.get("step_index", 0))
        if status == "executed":
            execution_state["execution_count"] = int(execution_state.get("execution_count", 0)) + 1
        elif status == "blocked":
            execution_state["blocked_count"] = int(execution_state.get("blocked_count", 0)) + 1
        else:
            execution_state["skipped_count"] = int(execution_state.get("skipped_count", 0)) + 1
        return execution_state

    def build_planner_executor_snapshot(self) -> Dict[str, Any]:
        current_waypoint = (
            self.runtime_debug.get("current_waypoint")
            if isinstance(self.runtime_debug.get("current_waypoint"), dict)
            else {}
        )
        current_state = self.planner_executor_state if isinstance(self.planner_executor_state, dict) else {}
        return build_planner_executor_runtime_state(
            mode=str(current_state.get("mode", "manual") or "manual"),
            active=bool(current_state.get("active", False)),
            state=str(current_state.get("state", "idle") or "idle"),
            run_id=str(current_state.get("run_id", "") or ""),
            mission_id=str(self.current_mission.get("mission_id", current_state.get("mission_id", "")) or ""),
            trigger=str(current_state.get("trigger", "") or ""),
            current_plan_id=str(self.current_plan.get("plan_id", current_state.get("current_plan_id", "")) or ""),
            current_search_subgoal=str(
                self.search_runtime.get(
                    "current_search_subgoal",
                    self.current_plan.get("search_subgoal", current_state.get("current_search_subgoal", "idle")),
                )
                or "idle"
            ),
            target_waypoint=current_waypoint,
            step_budget=int(current_state.get("step_budget", 0)),
            refresh_plan=bool(current_state.get("refresh_plan", False)),
            plan_refresh_interval_steps=int(current_state.get("plan_refresh_interval_steps", 0)),
            steps_executed=int(current_state.get("steps_executed", 0)),
            blocked_count=int(current_state.get("blocked_count", 0)),
            replan_count=int(current_state.get("replan_count", 0)),
            last_action=str(current_state.get("last_action", "idle") or "idle"),
            last_progress_cm=float(current_state.get("last_progress_cm", 0.0)),
            last_stop_reason=str(current_state.get("last_stop_reason", "") or ""),
            last_stop_detail=str(current_state.get("last_stop_detail", "") or ""),
            started_at=str(current_state.get("started_at", "") or now_timestamp()),
            updated_at=now_timestamp(),
        )

    def update_planner_executor_state(self, **updates: Any) -> Dict[str, Any]:
        state = self.build_planner_executor_snapshot()
        state.update(updates)
        self.planner_executor_state = build_planner_executor_runtime_state(
            mode=str(state.get("mode", "manual") or "manual"),
            active=bool(state.get("active", False)),
            state=str(state.get("state", "idle") or "idle"),
            run_id=str(state.get("run_id", "") or ""),
            mission_id=str(state.get("mission_id", self.current_mission.get("mission_id", "")) or ""),
            trigger=str(state.get("trigger", "") or ""),
            current_plan_id=str(state.get("current_plan_id", self.current_plan.get("plan_id", "")) or ""),
            current_search_subgoal=str(
                state.get(
                    "current_search_subgoal",
                    self.current_plan.get("search_subgoal", self.search_runtime.get("current_search_subgoal", "idle")),
                )
                or "idle"
            ),
            target_waypoint=state.get("target_waypoint") if isinstance(state.get("target_waypoint"), dict) else {},
            step_budget=int(state.get("step_budget", 0)),
            refresh_plan=bool(state.get("refresh_plan", False)),
            plan_refresh_interval_steps=int(state.get("plan_refresh_interval_steps", 0)),
            steps_executed=int(state.get("steps_executed", 0)),
            blocked_count=int(state.get("blocked_count", 0)),
            replan_count=int(state.get("replan_count", 0)),
            last_action=str(state.get("last_action", "idle") or "idle"),
            last_progress_cm=float(state.get("last_progress_cm", 0.0)),
            last_stop_reason=str(state.get("last_stop_reason", "") or ""),
            last_stop_detail=str(state.get("last_stop_detail", "") or ""),
            started_at=str(state.get("started_at", "") or now_timestamp()),
            updated_at=now_timestamp(),
        )
        return self.planner_executor_state

    def get_recent_action_history(self, limit: int = 8) -> List[Dict[str, Any]]:
        count = max(1, int(limit))
        return [dict(item) for item in self.recent_action_history[-count:] if isinstance(item, dict)]

    def build_heuristic_llm_action_runtime(self, *, trigger: str, reason: str) -> Dict[str, Any]:
        archive_state = self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
        heuristic_reflex = self.build_heuristic_reflex_runtime(archive_state, trigger=f"{trigger}_heuristic_action")
        suggested_action = normalize_reflex_action_name(heuristic_reflex.get("suggested_action", "hold_position"))
        if suggested_action in ("idle", "", "shield_hold"):
            suggested_action = "hold"
        elif suggested_action == "hold_position":
            suggested_action = "hold"
        return build_llm_action_runtime_state(
            action_id=f"heuristic_llm_action_{self.last_observation_id}",
            mode="llm_action_only",
            policy_name=self.args.planner_name,
            source="local_heuristic",
            status="fallback",
            suggested_action=suggested_action,
            should_execute=suggested_action not in ("hold", "hold_position", "idle"),
            confidence=float(heuristic_reflex.get("policy_confidence", 0.0) or 0.0),
            rationale=str(reason or "Heuristic action fallback"),
            stop_condition="continue_search",
            should_request_plan=False,
            last_trigger=trigger,
            last_latency_ms=0.0,
            risk_score=float(self.runtime_debug.get("risk_score", 0.0) or 0.0),
            model_name=str(self.plan_execution_state.get("last_model_name", "") or ""),
            api_style=str(self.plan_execution_state.get("last_api_style", "") or ""),
            route_mode=str(self.plan_execution_state.get("planner_route_mode", "") or ""),
            usage={},
            attempt_count=0,
            fallback_used=True,
            fallback_reason=str(reason or ""),
            upstream_error=str(reason or ""),
            raw_text="",
            parsed_payload={
                "action": suggested_action,
                "confidence": float(heuristic_reflex.get("policy_confidence", 0.0) or 0.0),
                "rationale": str(reason or "Heuristic action fallback"),
                "stop_condition": "continue_search",
                "should_request_plan": False,
            },
        )

    def request_llm_action(
        self,
        *,
        trigger: str = "manual_request",
        refresh_observations: bool = True,
    ) -> Dict[str, Any]:
        with self.lock:
            if refresh_observations or self.last_raw_frame is None or self.last_depth_frame is None:
                frame = self.refresh_observations()
            else:
                frame = self.last_raw_frame
            pose = self.get_task_pose()
            archive_state = self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
            if not self.should_preserve_reflex_runtime():
                self.sync_reflex_runtime(archive_state, trigger=f"{trigger}_state_sync")
            waypoint_hint = str(self.current_plan.get("semantic_subgoal", "") or self.search_runtime.get("current_search_subgoal", "") or "")
            action_request = build_llm_action_request(
                task_label=self.current_task_label or "idle",
                instruction=self.current_task_label or self.current_mission.get("mission_type", "semantic_navigation"),
                frame_id=self.last_observation_id,
                timestamp=self.last_observation_time,
                pose={
                    "x": pose[0],
                    "y": pose[1],
                    "z": pose[2],
                    "yaw": pose[3],
                },
                depth=self.last_depth_summary,
                camera_info=self.last_depth_summary.get("camera_info", {}),
                image_b64=encode_image_b64(frame, self.args.frame_jpeg_quality),
                planner_name=self.args.planner_name,
                trigger=trigger,
                step_index=int(self.plan_execution_state.get("step_index", 0)),
                mission=self.current_mission,
                search_runtime=self.search_runtime,
                doorway_runtime=self.doorway_runtime,
                phase5_mission_manual=self.phase5_mission_manual,
                person_evidence_runtime=self.person_evidence_runtime,
                search_result=self.search_result,
                language_memory_runtime=self.language_memory_runtime,
                current_plan=self.current_plan,
                reflex_runtime=self.reflex_runtime,
                runtime_debug=self.runtime_debug,
                context={
                    "recent_actions": self.get_recent_action_history(limit=8),
                    "waypoint_hint": waypoint_hint,
                    "archive": self.archive_runtime.get_planner_context(
                        task_label=self.current_task_label or "idle",
                        semantic_subgoal=str(self.current_plan.get("semantic_subgoal", "idle") or "idle"),
                        limit=int(self.args.archive_retrieval_limit),
                    ),
                },
            )
            self.last_llm_action_request = action_request
            if not self.args.planner_url:
                self.llm_action_runtime = self.build_heuristic_llm_action_runtime(trigger=trigger, reason="planner_url_unconfigured")
                return self.llm_action_runtime
            action_started = datetime.now().timestamp()
            req = request.Request(
                f"{self.args.planner_url.rstrip('/')}{self.args.planner_action_endpoint}",
                data=json.dumps(action_request).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.args.planner_timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                payload = raw.get("llm_action_runtime") if isinstance(raw, dict) and isinstance(raw.get("llm_action_runtime"), dict) else raw
                self.llm_action_runtime = coerce_llm_action_runtime_payload(
                    payload,
                    default_policy_name=self.args.planner_name,
                    default_source="external_action",
                )
                self.llm_action_runtime["last_trigger"] = trigger
                self.llm_action_runtime["last_latency_ms"] = float(self.llm_action_runtime.get("last_latency_ms", round((datetime.now().timestamp() - action_started) * 1000.0, 2)) or 0.0)
                return self.llm_action_runtime
            except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                logger.warning("LLM action request failed, fallback to heuristic action runtime: %s", exc)
                self.llm_action_runtime = self.build_heuristic_llm_action_runtime(trigger=trigger, reason=str(exc))
                self.llm_action_runtime["last_latency_ms"] = round((datetime.now().timestamp() - action_started) * 1000.0, 2)
                return self.llm_action_runtime

    def execute_llm_action(
        self,
        *,
        trigger: str = "manual_execute",
        refresh_action: bool = True,
        allow_auto_plan: bool = False,
    ) -> Dict[str, Any]:
        with self.lock:
            llm_action_runtime = self.request_llm_action(trigger=trigger) if refresh_action else self.llm_action_runtime
            requested_action = normalize_reflex_action_name(llm_action_runtime.get("suggested_action", "hold"))
            stop_condition = str(llm_action_runtime.get("stop_condition", "continue_search") or "continue_search")
            if requested_action in ("idle", "hold_position", "hold", "shield_hold"):
                return {
                    "status": "ok",
                    "executed": False,
                    "reason": stop_condition if stop_condition else "hold_action",
                    "requested_action": requested_action,
                    "llm_action_runtime": llm_action_runtime,
                    "state": self.get_state(),
                }
            motion_payload = self.build_motion_payload_for_action(requested_action)
            if motion_payload is None:
                self.llm_action_runtime["status"] = "unsupported_action"
                self.llm_action_runtime["upstream_error"] = f"Unsupported action={requested_action}"
                return {
                    "status": "ok",
                    "executed": False,
                    "reason": "unsupported_action",
                    "requested_action": requested_action,
                    "llm_action_runtime": self.llm_action_runtime,
                    "state": self.get_state(),
                }
            state = self._apply_motion_step(
                forward_cm=float(motion_payload.get("forward_cm", 0.0)),
                right_cm=float(motion_payload.get("right_cm", 0.0)),
                up_cm=float(motion_payload.get("up_cm", 0.0)),
                yaw_delta_deg=float(motion_payload.get("yaw_delta_deg", 0.0)),
                action_name=str(motion_payload.get("action_name", requested_action)),
                action_origin="llm_action",
                trigger_auto_plan=allow_auto_plan and bool(llm_action_runtime.get("should_request_plan", False)),
                trigger_reflex_update=False,
                allow_reflex_assist=False,
            )
            self.llm_action_runtime["status"] = "executed"
            return {
                "status": "ok",
                "executed": True,
                "reason": "ok",
                "requested_action": requested_action,
                "executed_action": str(motion_payload.get("action_name", requested_action)),
                "llm_action_runtime": self.llm_action_runtime,
                "state": state,
            }

    def execute_llm_action_segment(
        self,
        *,
        step_budget: int = 5,
        refresh_plan: bool = True,
        plan_refresh_interval_steps: int = 0,
        trigger: str = "manual_llm_action_segment",
    ) -> Dict[str, Any]:
        with self.lock:
            budget = max(1, min(int(step_budget), 25))
            replan_interval = max(0, int(plan_refresh_interval_steps))
            run_id = f"llm_action_segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(self.plan_execution_state.get('step_index', 0)):04d}"
            self.update_planner_executor_state(
                mode="llm_action_segment",
                active=True,
                state="running",
                run_id=run_id,
                mission_id=str(self.current_mission.get("mission_id", "")),
                trigger=str(trigger or "manual_llm_action_segment"),
                current_plan_id=str(self.current_plan.get("plan_id", "")),
                current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                step_budget=budget,
                refresh_plan=bool(refresh_plan),
                plan_refresh_interval_steps=replan_interval,
                steps_executed=0,
                blocked_count=0,
                replan_count=0,
                last_action="idle",
                last_progress_cm=0.0,
                last_stop_reason="",
                last_stop_detail="",
                started_at=now_timestamp(),
            )
            replan_count = 0
            if refresh_plan:
                self.request_plan(trigger=f"{trigger}_plan")
                replan_count = 1
                self.update_planner_executor_state(
                    replan_count=replan_count,
                    current_plan_id=str(self.current_plan.get("plan_id", "")),
                    current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                    target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                )
            steps_executed = 0
            blocked_count = 0
            stop_reason = "budget_exhausted"
            stop_detail = ""
            last_action = "idle"
            for step_idx in range(budget):
                if bool(self.takeover_state.get("active", False)):
                    stop_reason = "takeover_active"
                    stop_detail = "Manual takeover is active."
                    break
                evidence_status = str(self.person_evidence_runtime.get("evidence_status", "idle") or "idle")
                if evidence_status in ("suspect", "confirmed_present", "confirmed_absent"):
                    stop_reason = "evidence_state"
                    stop_detail = evidence_status
                    break
                if replan_interval > 0 and steps_executed > 0 and (steps_executed % replan_interval) == 0:
                    self.request_plan(trigger=f"{trigger}_replan_{steps_executed}")
                    replan_count += 1
                    self.update_planner_executor_state(
                        replan_count=replan_count,
                        current_plan_id=str(self.current_plan.get("plan_id", "")),
                        current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                        target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                    )
                before_wp_distance = float(self.reflex_runtime.get("waypoint_distance_cm", 0.0) or 0.0)
                llm_action_runtime = self.request_llm_action(trigger=f"{trigger}_step_{step_idx + 1}", refresh_observations=True)
                requested_action = normalize_reflex_action_name(llm_action_runtime.get("suggested_action", "hold"))
                stop_condition = str(llm_action_runtime.get("stop_condition", "continue_search") or "continue_search")
                if requested_action in ("idle", "hold_position", "hold", "shield_hold"):
                    blocked_count += 1
                    stop_reason = stop_condition if stop_condition != "continue_search" else "hold_action"
                    stop_detail = str(llm_action_runtime.get("rationale", "") or "LLM returned hold")
                    break
                motion_payload = self.build_motion_payload_for_action(requested_action)
                if motion_payload is None:
                    blocked_count += 1
                    stop_reason = "unsupported_action"
                    stop_detail = requested_action
                    break
                self._apply_motion_step(
                    forward_cm=float(motion_payload.get("forward_cm", 0.0)),
                    right_cm=float(motion_payload.get("right_cm", 0.0)),
                    up_cm=float(motion_payload.get("up_cm", 0.0)),
                    yaw_delta_deg=float(motion_payload.get("yaw_delta_deg", 0.0)),
                    action_name=str(motion_payload.get("action_name", requested_action)),
                    action_origin="llm_action",
                    trigger_auto_plan=False,
                    trigger_reflex_update=False,
                    allow_reflex_assist=False,
                )
                steps_executed += 1
                last_action = str(motion_payload.get("action_name", requested_action))
                after_wp_distance = float(self.reflex_runtime.get("waypoint_distance_cm", 0.0) or 0.0)
                last_progress_cm = before_wp_distance - after_wp_distance if before_wp_distance > 0.0 and after_wp_distance > 0.0 else 0.0
                self.update_planner_executor_state(
                    steps_executed=steps_executed,
                    blocked_count=blocked_count,
                    replan_count=replan_count,
                    current_plan_id=str(self.current_plan.get("plan_id", "")),
                    current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                    target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                    last_action=last_action,
                    last_progress_cm=last_progress_cm,
                    last_stop_reason="",
                    last_stop_detail=str(llm_action_runtime.get("rationale", "") or ""),
                )
                if bool(llm_action_runtime.get("should_request_plan", False)):
                    self.request_plan(trigger=f"{trigger}_llm_replan_{steps_executed}")
                    replan_count += 1
                    self.update_planner_executor_state(
                        replan_count=replan_count,
                        current_plan_id=str(self.current_plan.get("plan_id", "")),
                        current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                        target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                    )
                if stop_condition in {"need_manual_review", "target_confirmed", "hold_position"}:
                    stop_reason = stop_condition
                    stop_detail = str(llm_action_runtime.get("rationale", "") or stop_condition)
                    break
            if stop_reason == "budget_exhausted" and steps_executed <= 0 and blocked_count > 0:
                stop_reason = "blocked"
            final_state_label = "completed" if stop_reason in ("budget_exhausted", "target_confirmed") else "stopped"
            self.update_planner_executor_state(
                active=False,
                state=final_state_label,
                steps_executed=steps_executed,
                blocked_count=blocked_count,
                replan_count=replan_count,
                last_action=last_action,
                last_stop_reason=stop_reason,
                last_stop_detail=stop_detail,
            )
            return {
                "status": "ok",
                "segment_executed": bool(steps_executed > 0),
                "steps_executed": int(steps_executed),
                "step_budget": int(budget),
                "stop_reason": stop_reason,
                "stop_detail": stop_detail,
                "replan_count": int(replan_count),
                "llm_action_runtime": self.llm_action_runtime,
                "planner_executor_runtime": self.planner_executor_state,
                "state": self.get_state(),
            }

    def build_motion_payload_for_action(self, action_name: str) -> Optional[Dict[str, Any]]:
        action = normalize_reflex_action_name(action_name)
        if action in ("idle", "hold_position", "shield_hold", "", "hold"):
            return None
        if action == "forward":
            return {"forward_cm": float(self.args.move_step_cm), "action_name": "forward"}
        if action == "backward":
            return {"forward_cm": -float(self.args.move_step_cm), "action_name": "backward"}
        if action == "left":
            return {"right_cm": -float(self.args.move_step_cm), "action_name": "left"}
        if action == "right":
            return {"right_cm": float(self.args.move_step_cm), "action_name": "right"}
        if action == "up":
            return {"up_cm": float(self.args.vertical_step_cm), "action_name": "up"}
        if action == "down":
            return {"up_cm": -float(self.args.vertical_step_cm), "action_name": "down"}
        if action == "yaw_left":
            return {"yaw_delta_deg": -float(self.args.yaw_step_deg), "action_name": "yaw_left"}
        if action == "yaw_right":
            return {"yaw_delta_deg": float(self.args.yaw_step_deg), "action_name": "yaw_right"}
        return None

    def get_last_manual_action_name(self) -> str:
        action_info = self.runtime_debug.get("local_policy_action", {})
        if not isinstance(action_info, dict):
            return ""
        if str(action_info.get("action_origin", "") or "") != "manual":
            return ""
        return normalize_reflex_action_name(action_info.get("action_name", ""))

    def evaluate_reflex_execution_gate(self, reflex_runtime: Dict[str, Any]) -> Tuple[bool, str, str]:
        action_name = normalize_reflex_action_name(reflex_runtime.get("suggested_action", "idle"))
        source = str(reflex_runtime.get("source", "") or "")
        if not reflex_runtime.get("should_execute", False):
            return False, "should_execute_false", action_name
        if action_name in ("idle", "hold_position"):
            return False, "hold_action", action_name
        if action_name == "shield_hold":
            return False, "shield_hold", action_name
        if not self.args.reflex_execute_allow_heuristic and str(source).lower() != "external_model":
            return False, "source_not_allowed", action_name
        confidence = float(reflex_runtime.get("policy_confidence", 0.0))
        if str(source).lower() == "external_model" and confidence < float(self.args.reflex_execute_confidence_threshold):
            return False, "low_confidence", action_name
        risk_score = float(reflex_runtime.get("risk_score", self.runtime_debug.get("risk_score", 0.0)))
        if risk_score > float(self.args.reflex_execute_max_risk):
            return False, "high_risk", action_name
        if bool(reflex_runtime.get("shield_triggered", False)):
            return False, "shield_triggered", action_name
        if self.args.reflex_execute_mode == "assist_step":
            last_manual_action = self.get_last_manual_action_name()
            if last_manual_action and REFLEX_OPPOSITE_ACTIONS.get(last_manual_action) == action_name:
                return False, "opposes_manual_action", action_name
        if self.build_motion_payload_for_action(action_name) is None:
            return False, "unsupported_action", action_name
        return True, "ok", action_name

    def build_planner_executor_motion_payload(
        self,
        *,
        trigger: str,
        allow_reflex: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], str, str]:
        archive_state = self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
        if allow_reflex:
            reflex_runtime = self.request_reflex_policy(trigger=trigger, archive_snapshot=archive_state)
            allowed, _reason, action_name = self.evaluate_reflex_execution_gate(reflex_runtime)
            if allowed:
                payload = self.build_motion_payload_for_action(action_name)
                if payload is not None:
                    return payload, action_name, "reflex_runtime"

        heuristic_runtime = self.build_heuristic_reflex_runtime(archive_state, trigger=f"{trigger}_heuristic")
        heuristic_runtime["source"] = "local_heuristic"
        self.reflex_runtime = heuristic_runtime
        heuristic_action = normalize_reflex_action_name(heuristic_runtime.get("suggested_action", "idle"))
        payload = self.build_motion_payload_for_action(heuristic_action)
        if payload is not None:
            return payload, heuristic_action, "local_heuristic"
        return None, "idle", "none"

    def execute_plan_segment(
        self,
        *,
        step_budget: int = 5,
        refresh_plan: bool = True,
        plan_refresh_interval_steps: int = 0,
        allow_reflex: bool = True,
        trigger: str = "manual_segment",
    ) -> Dict[str, Any]:
        with self.lock:
            budget = max(1, min(int(step_budget), 25))
            replan_interval = max(0, int(plan_refresh_interval_steps))
            run_id = f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(self.plan_execution_state.get('step_index', 0)):04d}"
            self.update_planner_executor_state(
                mode="execute_plan_segment",
                active=True,
                state="running",
                run_id=run_id,
                mission_id=str(self.current_mission.get("mission_id", "")),
                trigger=str(trigger or "manual_segment"),
                current_plan_id=str(self.current_plan.get("plan_id", "")),
                current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                step_budget=budget,
                refresh_plan=bool(refresh_plan),
                plan_refresh_interval_steps=replan_interval,
                steps_executed=0,
                blocked_count=0,
                replan_count=0,
                last_action="idle",
                last_progress_cm=0.0,
                last_stop_reason="",
                last_stop_detail="",
                started_at=now_timestamp(),
            )

            replan_count = 0
            if refresh_plan:
                self.request_plan(trigger=f"{trigger}_plan")
                replan_count = 1
                self.update_planner_executor_state(
                    replan_count=replan_count,
                    current_plan_id=str(self.current_plan.get("plan_id", "")),
                    current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                    target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                )

            steps_executed = 0
            blocked_count = 0
            last_action = "idle"
            last_progress_cm = 0.0
            stop_reason = "budget_exhausted"
            stop_detail = ""

            for step_idx in range(budget):
                if bool(self.takeover_state.get("active", False)):
                    stop_reason = "takeover_active"
                    stop_detail = "Manual takeover is active."
                    break

                if replan_interval > 0 and steps_executed > 0 and (steps_executed % replan_interval) == 0:
                    self.request_plan(trigger=f"{trigger}_replan_{steps_executed}")
                    replan_count += 1
                    self.update_planner_executor_state(
                        replan_count=replan_count,
                        current_plan_id=str(self.current_plan.get("plan_id", "")),
                        current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                        target_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {},
                        last_stop_detail=f"replan@step={steps_executed}",
                    )

                evidence_status = str(self.person_evidence_runtime.get("evidence_status", "idle") or "idle")
                if evidence_status in ("suspect", "confirmed_present", "confirmed_absent"):
                    stop_reason = "evidence_state"
                    stop_detail = evidence_status
                    break

                before_wp_distance = float(self.reflex_runtime.get("waypoint_distance_cm", 0.0))
                motion_payload, selected_action, selected_source = self.build_planner_executor_motion_payload(
                    trigger=f"{trigger}_step_{step_idx + 1}",
                    allow_reflex=allow_reflex,
                )
                if motion_payload is None:
                    blocked_count += 1
                    stop_reason = "blocked"
                    stop_detail = "No executable planner action."
                    self.update_planner_executor_state(
                        blocked_count=blocked_count,
                        steps_executed=steps_executed,
                        last_action="idle",
                        last_stop_reason=stop_reason,
                        last_stop_detail=stop_detail,
                    )
                    if blocked_count >= 2:
                        break
                    continue

                self._apply_motion_step(
                    forward_cm=float(motion_payload.get("forward_cm", 0.0)),
                    right_cm=float(motion_payload.get("right_cm", 0.0)),
                    up_cm=float(motion_payload.get("up_cm", 0.0)),
                    yaw_delta_deg=float(motion_payload.get("yaw_delta_deg", 0.0)),
                    action_name=str(motion_payload.get("action_name", selected_action)),
                    action_origin="planner_executor",
                    trigger_auto_plan=False,
                    trigger_reflex_update=False,
                    allow_reflex_assist=False,
                )
                steps_executed += 1
                last_action = str(motion_payload.get("action_name", selected_action))
                after_wp_distance = float(self.reflex_runtime.get("waypoint_distance_cm", 0.0))
                last_progress_cm = before_wp_distance - after_wp_distance if before_wp_distance > 0.0 and after_wp_distance > 0.0 else 0.0
                current_waypoint = self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else {}
                waypoint_radius = float(current_waypoint.get("radius", self.args.default_waypoint_radius_cm)) if current_waypoint else float(self.args.default_waypoint_radius_cm)
                self.update_planner_executor_state(
                    steps_executed=steps_executed,
                    blocked_count=blocked_count,
                    current_plan_id=str(self.current_plan.get("plan_id", "")),
                    current_search_subgoal=str(self.search_runtime.get("current_search_subgoal", "idle")),
                    target_waypoint=current_waypoint,
                    last_action=last_action,
                    last_progress_cm=last_progress_cm,
                    last_stop_reason="",
                    last_stop_detail=f"source={selected_source}",
                )
                if after_wp_distance > 0.0 and after_wp_distance <= waypoint_radius:
                    stop_reason = "waypoint_reached"
                    stop_detail = f"distance_cm={after_wp_distance:.1f}"
                    break

            final_state_label = "completed" if stop_reason in ("waypoint_reached", "budget_exhausted") else "stopped"
            self.update_planner_executor_state(
                active=False,
                state=final_state_label,
                steps_executed=steps_executed,
                blocked_count=blocked_count,
                replan_count=replan_count,
                last_action=last_action,
                last_progress_cm=last_progress_cm,
                last_stop_reason=stop_reason,
                last_stop_detail=stop_detail or self.planner_executor_state.get("last_stop_detail", ""),
            )
            state = self.get_state()
            return {
                "status": "ok",
                "segment_executed": bool(steps_executed > 0),
                "steps_executed": int(steps_executed),
                "step_budget": int(budget),
                "stop_reason": stop_reason,
                "stop_detail": stop_detail,
                "replan_count": int(replan_count),
                "planner_executor_runtime": self.planner_executor_state,
                "state": state,
            }

    def get_env_yaw_deg(self) -> float:
        rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
        if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
            return normalize_angle_deg(float(rotation[1]))
        return 0.0

    def _coerce_fov_deg(self, raw_fov: Any) -> float:
        if isinstance(raw_fov, (int, float)):
            value = float(raw_fov)
            if value > 1e-3:
                return value
        if isinstance(raw_fov, str):
            stripped = raw_fov.strip()
            try:
                value = float(stripped)
                if value > 1e-3:
                    return value
            except ValueError:
                matches = re.findall(r"[-+]?\d*\.?\d+", stripped)
                if len(matches) == 1:
                    value = float(matches[0])
                    if value > 1e-3:
                        return value
        fallback = float(self.last_depth_summary.get("fov_deg", self.args.default_depth_fov_deg))
        logger.warning("Unexpected get_cam_fov response %r, fallback to %.2f deg", raw_fov, fallback)
        return fallback

    def get_task_pose(self) -> List[float]:
        location = self.env.unwrapped.unrealcv.get_obj_location(self.player_name)
        task_yaw = normalize_angle_deg(self.get_env_yaw_deg() + 180.0)
        return [float(location[0]), float(location[1]), float(location[2]), task_yaw]

    def set_task_pose(self, position: List[float], yaw_deg: float) -> None:
        self.env.unwrapped.unrealcv.set_obj_location(self.player_name, list(position))
        self.env.unwrapped.unrealcv.set_rotation(self.player_name, float(yaw_deg) - 180.0)

    def get_control_yaw_deg(self) -> float:
        if self.args.movement_yaw_mode == "task":
            return self.command_task_yaw_deg
        if self.args.movement_yaw_mode == "camera" and self.args.preview_mode == "first_person":
            try:
                set_cam(self.env, self.policy_cam_id)
                cam_rotation = self.env.unwrapped.unrealcv.get_cam_rotation(self.policy_cam_id)
                if isinstance(cam_rotation, (list, tuple)) and len(cam_rotation) > 1:
                    return normalize_angle_deg(float(cam_rotation[1]))
            except Exception as exc:
                logger.debug("Failed to read control camera yaw, fallback to UAV yaw: %s", exc)
        return self.get_env_yaw_deg()

    def position_free_view_once(self) -> None:
        if self.args.viewport_mode != "free" or not self.args.follow_free_view:
            return
        pose = self.get_task_pose()
        focus_pose = [pose[0], pose[1], pose[2], 0.0, pose[3]]
        set_free_view_near_pose(self.env, focus_pose, self.free_view_offset, self.free_view_rotation)

    def get_preview_frame(self) -> np.ndarray:
        if self.args.preview_mode == "third_person":
            return get_third_person_preview_image(
                self.env,
                self.preview_cam_id,
                self.preview_offset,
                self.preview_rotation,
            )
        set_cam(self.env, self.policy_cam_id)
        return self.env.unwrapped.unrealcv.get_image(self.policy_cam_id, "lit")

    def get_depth_observation(self) -> Tuple[np.ndarray, float]:
        set_cam(self.env, self.policy_cam_id)
        depth_image = coerce_depth_planar_image(self.env.unwrapped.unrealcv.get_depth(self.policy_cam_id))
        fov_deg = self._coerce_fov_deg(self.env.unwrapped.unrealcv.get_cam_fov(self.policy_cam_id))
        return depth_image, fov_deg

    def invalidate_cached_observations(self) -> None:
        self.last_raw_frame = None
        self.last_depth_frame = None

    def refresh_preview_only(self) -> np.ndarray:
        self.frame_index += 1
        self.last_observation_time = now_timestamp()
        self.last_observation_id = f"frame_{self.frame_index:06d}"
        frame = self.get_preview_frame()
        self.last_raw_frame = frame
        return frame

    def refresh_depth_only(self) -> Tuple[np.ndarray, float]:
        if self.last_raw_frame is None:
            self.refresh_preview_only()
        depth_image, depth_fov_deg = self.get_depth_observation()
        self.last_depth_frame = depth_image
        camera_info = generate_camera_info(
            int(depth_image.shape[1]),
            int(depth_image.shape[0]),
            float(depth_fov_deg),
            self.args.depth_camera_frame_id,
        )
        finite_depth = depth_image[np.isfinite(depth_image)]
        configured_min_depth = float(self.args.depth_min_cm)
        configured_max_depth = float(self.args.depth_max_cm)
        valid_depth = finite_depth[(finite_depth >= configured_min_depth) & (finite_depth <= configured_max_depth)]
        front_min_depth, front_mean_depth, risk_score, shield_triggered = self.estimate_depth_risk(depth_image)
        self.last_depth_summary = {
            "frame_id": self.last_observation_id,
            "available": bool(finite_depth.size),
            "min_depth": float(np.min(valid_depth)) if valid_depth.size else configured_min_depth,
            "max_depth": float(np.max(valid_depth)) if valid_depth.size else configured_max_depth,
            "raw_min_depth": float(np.min(finite_depth)) if finite_depth.size else 0.0,
            "raw_max_depth": float(np.max(finite_depth)) if finite_depth.size else 0.0,
            "front_min_depth": front_min_depth,
            "front_mean_depth": front_mean_depth,
            "image_width": int(depth_image.shape[1]),
            "image_height": int(depth_image.shape[0]),
            "fov_deg": float(depth_fov_deg),
            "source_mode": "lesson4_depth_planar",
            "pipeline": "lesson4_depth_image_proc_style",
            "camera_id": int(self.policy_cam_id),
            "camera_info": camera_info,
        }
        self.runtime_debug["risk_score"] = risk_score
        self.runtime_debug["shield_triggered"] = shield_triggered
        self.sync_doorway_runtime()
        self.sync_archive_runtime()
        return depth_image, depth_fov_deg

    def refresh_observations(self) -> np.ndarray:
        frame = self.refresh_preview_only()
        self.refresh_depth_only()
        return frame

    def refresh_preview(self) -> np.ndarray:
        return self.refresh_observations()

    def get_depth_frame_jpeg(self) -> bytes:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            if self.last_depth_frame is None:
                raise RuntimeError("No depth frame available")
            image = render_depth_preview(
                self.last_depth_frame,
                self.args.depth_preview_width,
                self.args.depth_preview_height,
                min_depth_cm=float(self.args.depth_min_cm),
                max_depth_cm=float(self.args.depth_max_cm),
                source_mode=self.last_depth_summary.get("source_mode", "policy_depth"),
            )
            encode_ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)],
            )
            if not encode_ok:
                raise RuntimeError("Failed to encode depth preview frame")
            return encoded.tobytes()

    def get_depth_raw_png(self) -> bytes:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            if self.last_depth_frame is None:
                raise RuntimeError("No depth frame available")
            depth_u16 = np.clip(
                np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
                65535.0,
            ).astype(np.uint16)
            encode_ok, encoded = cv2.imencode(".png", depth_u16)
            if not encode_ok:
                raise RuntimeError("Failed to encode raw depth frame")
            return encoded.tobytes()

    def get_camera_info(self) -> Dict[str, Any]:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            return self.last_depth_summary.get("camera_info", {})

    def set_task_label(self, task_label: str) -> Dict[str, Any]:
        with self.lock:
            self.current_task_label = str(task_label or "").strip()
            self.current_mission = self.build_mission_descriptor(self.current_task_label)
            self.language_memory_runtime = self.language_search_memory.reset(
                mission_id=str(self.current_mission.get("mission_id", "")),
                mission_type=str(self.current_mission.get("mission_type", "semantic_navigation")),
                task_label=self.current_task_label,
            )
            self.llm_action_runtime = build_llm_action_runtime_state(
                policy_name=self.args.planner_name,
                source="none",
                status="idle",
            )
            self.last_llm_action_request = {}
            self.reset_person_search_state(reset_recent_events=True)
            self.search_runtime = self.build_search_runtime_snapshot()
            archive_state = self.sync_archive_runtime()
            self.sync_search_runtime(archive_state)
            self.sync_phase5_mission_manual()
            return {
                "status": "ok",
                "task_label": self.current_task_label,
                "mission": self.current_mission,
                "search_runtime": self.search_runtime,
                "doorway_runtime": self.doorway_runtime,
                "person_evidence_runtime": self.person_evidence_runtime,
                "search_result": self.search_result,
                "language_memory_runtime": self.language_memory_runtime,
                "phase5_mission_manual": self.phase5_mission_manual,
            }

    def set_plan_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.current_plan = coerce_plan_payload(
                payload,
                default_plan_id=f"plan_{self.last_observation_id}",
                default_planner_name=self.args.planner_name,
                default_semantic_subgoal="idle",
                default_radius=self.args.default_waypoint_radius_cm,
            )
            candidate_waypoints = self.current_plan.get("candidate_waypoints") or []
            self.runtime_debug["current_waypoint"] = candidate_waypoints[0] if candidate_waypoints else None
            plan_debug = self.current_plan.get("debug") if isinstance(self.current_plan.get("debug"), dict) else {}
            self.plan_execution_state["planner_source_detail"] = str(plan_debug.get("source", self.plan_execution_state.get("planner_source_detail", "none")) or "none")
            self.plan_execution_state["planner_route_mode"] = str(plan_debug.get("route_mode", self.plan_execution_state.get("planner_route_mode", "heuristic_only")) or "heuristic_only")
            self.plan_execution_state["planner_route_reason"] = str(plan_debug.get("route_reason", self.plan_execution_state.get("planner_route_reason", "")) or "")
            plan_model_name = plan_debug.get("model_name", plan_debug.get("llm_model_name", self.plan_execution_state.get("last_model_name", "")))
            plan_api_style = plan_debug.get("api_style", plan_debug.get("llm_api_style", self.plan_execution_state.get("last_api_style", "")))
            plan_usage = plan_debug.get("usage", plan_debug.get("llm_usage", self.plan_execution_state.get("last_usage", {})))
            self.plan_execution_state["last_model_name"] = str(plan_model_name or "")
            self.plan_execution_state["last_api_style"] = str(plan_api_style or "")
            self.plan_execution_state["last_usage"] = plan_usage if isinstance(plan_usage, dict) else {}
            self.plan_execution_state["fallback_used"] = bool(plan_debug.get("fallback_used", self.plan_execution_state.get("fallback_used", False)))
            self.plan_execution_state["fallback_reason"] = str(plan_debug.get("fallback_reason", self.plan_execution_state.get("fallback_reason", "")) or "")
            self.sync_mission_from_plan()
            archive_state = self.sync_archive_runtime()
            self.sync_search_runtime(archive_state)
            self.language_search_memory.record_plan(plan=self.current_plan, search_runtime=self.search_runtime)
            self.sync_language_memory(archive_state)
            return self.current_plan

    def request_plan(self, task_label: Optional[str] = None, trigger: str = "manual_request") -> Dict[str, Any]:
        with self.lock:
            if task_label is not None:
                self.set_task_label(task_label)
            frame = self.refresh_observations()
            pose = self.get_task_pose()
            task_label_value = self.current_task_label or "idle"
            plan_payload: Optional[Dict[str, Any]] = None
            planner_started = datetime.now().timestamp()
            self.plan_execution_state["request_count"] = int(self.plan_execution_state.get("request_count", 0)) + 1
            self.plan_execution_state["last_trigger"] = trigger

            if self.args.planner_url:
                planner_request = build_plan_request(
                    task_label=task_label_value,
                    instruction=task_label_value,
                    frame_id=self.last_observation_id,
                    timestamp=self.last_observation_time,
                    pose={
                        "x": pose[0],
                        "y": pose[1],
                        "z": pose[2],
                        "yaw": pose[3],
                    },
                    depth=self.last_depth_summary,
                    camera_info=self.last_depth_summary.get("camera_info", {}),
                    image_b64=encode_image_b64(frame, self.args.frame_jpeg_quality),
                    planner_name=self.args.planner_name,
                    trigger=trigger,
                    step_index=int(self.plan_execution_state.get("step_index", 0)),
                    mission=self.current_mission,
                    search_runtime=self.search_runtime,
                    doorway_runtime=self.doorway_runtime,
                    phase5_mission_manual=self.phase5_mission_manual,
                    person_evidence_runtime=self.person_evidence_runtime,
                    search_result=self.search_result,
                    language_memory_runtime=self.language_memory_runtime,
                    context={
                        "movement_yaw_mode": self.args.movement_yaw_mode,
                        "preview_mode": self.args.preview_mode,
                        "risk_score": float(self.runtime_debug.get("risk_score", 0.0)),
                        "reflex_runtime": self.reflex_runtime,
                        "archive": self.archive_runtime.get_planner_context(
                            task_label=task_label_value,
                            semantic_subgoal=self.current_plan.get("semantic_subgoal", "idle"),
                            limit=int(self.args.archive_retrieval_limit),
                        ),
                    },
                )
                self.last_plan_request = planner_request
                req = request.Request(
                    f"{self.args.planner_url.rstrip('/')}{self.args.planner_endpoint}",
                    data=json.dumps(planner_request).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with request.urlopen(req, timeout=self.args.planner_timeout_s) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))
                    if isinstance(raw, dict):
                        plan_payload = raw.get("plan") if isinstance(raw.get("plan"), dict) else raw
                        self.plan_execution_state.update(
                            {
                                "planner_status": "ok",
                                "planner_source": "external",
                                "planner_source_detail": "external",
                                "planner_route_mode": "external",
                                "planner_route_reason": "",
                                "last_trigger": trigger,
                                "last_error": "",
                                "fallback_used": False,
                                "fallback_reason": "",
                            }
                        )
                except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    self.plan_execution_state.update(
                        {
                            "planner_status": "fallback",
                            "planner_source": "external_error",
                            "planner_source_detail": "external_error",
                            "planner_route_mode": "external_error",
                            "planner_route_reason": str(exc),
                            "last_trigger": trigger,
                            "last_error": str(exc),
                            "fallback_used": True,
                            "fallback_reason": str(exc),
                        }
                    )
                    logger.warning("Planner request failed, falling back to heuristic plan: %s", exc)

            if plan_payload is None:
                move_yaw = self.get_control_yaw_deg()
                theta = np.radians(move_yaw)
                primary_waypoint = build_waypoint(
                    x=pose[0] + self.args.default_plan_distance_cm * float(np.cos(theta)),
                    y=pose[1] + self.args.default_plan_distance_cm * float(np.sin(theta)),
                    z=pose[2],
                    yaw=move_yaw,
                    radius=self.args.default_waypoint_radius_cm,
                    semantic_label=task_label_value or "forward_search",
                )
                secondary_waypoint = build_waypoint(
                    x=pose[0] + (self.args.default_plan_distance_cm * 0.6) * float(np.cos(theta)),
                    y=pose[1] + (self.args.default_plan_distance_cm * 0.6) * float(np.sin(theta)),
                    z=pose[2],
                    yaw=move_yaw,
                    radius=self.args.default_waypoint_radius_cm,
                    semantic_label="staging_waypoint",
                )
                plan_payload = build_plan_state(
                    plan_id=f"plan_{self.last_observation_id}",
                    planner_name="heuristic_fallback",
                    sector_id=int(round(((move_yaw % 360.0) / 360.0) * self.args.default_sector_count)) % self.args.default_sector_count,
                    candidate_waypoints=[primary_waypoint, secondary_waypoint],
                    semantic_subgoal=task_label_value or "move_forward",
                    planner_confidence=0.35,
                    should_replan=False,
                    mission_type=str(self.current_mission.get("mission_type", "semantic_navigation")),
                    search_subgoal="search_frontier"
                    if str(self.current_mission.get("mission_type", "")) in ("person_search", "room_search", "target_verification")
                    else "advance_to_waypoint",
                    priority_region=(self.current_mission.get("priority_regions") or [{}])[0],
                    candidate_regions=self.current_mission.get("priority_regions") or [],
                    confirm_target=bool(self.current_mission.get("confirm_target", False)),
                    explanation="Local fallback plan reused the current mission descriptor.",
                    debug={
                        "source": "local_heuristic",
                        "planner_interval_steps": self.args.planner_interval_steps,
                        "trigger": trigger,
                        "step_index": int(self.plan_execution_state.get("step_index", 0)),
                        "fallback_used": self.plan_execution_state.get("planner_status") != "ok",
                        "fallback_reason": str(self.plan_execution_state.get("last_error", "")),
                    },
                )
                if self.plan_execution_state.get("planner_status") != "ok":
                    self.plan_execution_state.update(
                        {
                            "planner_status": "fallback",
                            "planner_source": "local_heuristic",
                            "planner_source_detail": "local_heuristic",
                            "planner_route_mode": "heuristic_only",
                            "planner_route_reason": "local_control_server_fallback",
                            "last_trigger": trigger,
                            "last_model_name": "",
                            "last_api_style": "",
                            "last_usage": {},
                        }
                    )

            self.plan_execution_state["last_latency_ms"] = round((datetime.now().timestamp() - planner_started) * 1000.0, 2)
            if trigger != "manual_request":
                self.plan_execution_state["auto_request_count"] = int(self.plan_execution_state.get("auto_request_count", 0)) + 1
                self.plan_execution_state["last_auto_trigger_step"] = int(self.plan_execution_state.get("step_index", 0))
                self.plan_execution_state["next_auto_trigger_step"] = int(self.plan_execution_state.get("step_index", 0)) + max(
                    1, int(self.args.planner_interval_steps)
                )
            self.current_plan = self.set_plan_state(plan_payload)
            return self.current_plan

    def should_auto_request_plan(self) -> bool:
        """Return True if the sparse planner should be auto-triggered now."""
        if self.args.planner_auto_mode != "k_step":
            return False
        interval = max(1, int(self.args.planner_interval_steps))
        next_trigger_step = int(self.plan_execution_state.get("next_auto_trigger_step", 1))
        step_index = int(self.plan_execution_state.get("step_index", 0))
        if next_trigger_step <= 0:
            next_trigger_step = 1
            self.plan_execution_state["next_auto_trigger_step"] = next_trigger_step
        self.plan_execution_state["auto_interval_steps"] = interval
        return step_index >= next_trigger_step

    def update_runtime_debug(
        self,
        *,
        current_waypoint: Optional[Dict[str, Any]] = None,
        local_policy_action: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
        shield_triggered: Optional[bool] = None,
        archive_cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if current_waypoint is not None:
                self.runtime_debug["current_waypoint"] = current_waypoint
            if local_policy_action is not None:
                self.runtime_debug["local_policy_action"] = local_policy_action
            if risk_score is not None:
                self.runtime_debug["risk_score"] = float(risk_score)
            if shield_triggered is not None:
                self.runtime_debug["shield_triggered"] = bool(shield_triggered)
            if archive_cell_id is not None:
                self.runtime_debug["archive_cell_id"] = str(archive_cell_id)
            self.sync_archive_runtime()
            return self.runtime_debug

    def estimate_depth_risk(self, depth_image: np.ndarray) -> Tuple[float, float, float, bool]:
        height, width = depth_image.shape[:2]
        x0 = max(0, int(width * 0.35))
        x1 = min(width, int(width * 0.65))
        y0 = max(0, int(height * 0.30))
        y1 = min(height, int(height * 0.85))
        roi = depth_image[y0:y1, x0:x1]
        finite_depth = roi[np.isfinite(roi)]
        valid_depth = finite_depth[
            (finite_depth >= float(self.args.depth_min_cm)) & (finite_depth <= float(self.args.depth_max_cm))
        ]
        if not valid_depth.size:
            return float(self.args.depth_max_cm), float(self.args.depth_max_cm), 0.0, False
        front_min_depth = float(np.min(valid_depth))
        front_mean_depth = float(np.mean(valid_depth))
        risk_near_cm = max(float(self.args.risk_near_cm), float(self.args.depth_min_cm) + 1.0)
        normalized = 1.0 - min(front_min_depth, risk_near_cm) / risk_near_cm
        risk_score = float(np.clip(normalized, 0.0, 1.0))
        shield_triggered = bool(risk_score >= float(self.args.shield_risk_threshold))
        return front_min_depth, front_mean_depth, risk_score, shield_triggered

    def sync_archive_runtime(self) -> Dict[str, Any]:
        with self.lock:
            pose = self.get_task_pose()
            current_waypoint = self.runtime_debug.get("current_waypoint")
            snapshot = self.archive_runtime.register_observation(
                timestamp=self.last_observation_time,
                frame_id=self.last_observation_id,
                task_label=self.current_task_label or "idle",
                semantic_subgoal=self.current_plan.get("semantic_subgoal", "idle"),
                pose={
                    "x": pose[0],
                    "y": pose[1],
                    "z": pose[2],
                    "yaw": pose[3],
                },
                depth_summary=self.last_depth_summary,
                current_waypoint=current_waypoint if isinstance(current_waypoint, dict) else None,
                action_label=self.last_action,
                risk_score=float(self.runtime_debug.get("risk_score", 0.0)),
            )
            self.runtime_debug["archive_cell_id"] = snapshot["current_cell_id"]
            if not self.should_preserve_reflex_runtime():
                self.sync_reflex_runtime(snapshot, trigger="archive_sync")
            self.sync_search_runtime(snapshot)
            return snapshot

    def build_heuristic_reflex_runtime(self, archive_snapshot: Optional[Dict[str, Any]] = None, *, trigger: str = "") -> Dict[str, Any]:
        pose = self.get_task_pose()
        current_waypoint = self.runtime_debug.get("current_waypoint")
        archive_state = archive_snapshot if isinstance(archive_snapshot, dict) else self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
        active_retrieval = archive_state.get("active_retrieval") if isinstance(archive_state.get("active_retrieval"), dict) else {}

        waypoint_distance_cm = 0.0
        yaw_error_deg = 0.0
        vertical_error_cm = 0.0
        progress_to_waypoint_cm = 0.0
        suggested_action = "hold_position"
        status = "idle"
        should_execute = False

        if isinstance(current_waypoint, dict):
            dx = float(current_waypoint.get("x", pose[0])) - float(pose[0])
            dy = float(current_waypoint.get("y", pose[1])) - float(pose[1])
            dz = float(current_waypoint.get("z", pose[2])) - float(pose[2])
            horizontal_distance = float(np.hypot(dx, dy))
            waypoint_distance_cm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            vertical_error_cm = dz
            desired_yaw = normalize_angle_deg(np.degrees(np.arctan2(dy, dx))) if horizontal_distance > 1e-6 else float(pose[3])
            yaw_error_deg = normalize_angle_deg(desired_yaw - float(pose[3]))
            previous_distance = float(self.reflex_runtime.get("waypoint_distance_cm", waypoint_distance_cm))
            progress_to_waypoint_cm = previous_distance - waypoint_distance_cm
            status = "tracking_waypoint"

            if bool(self.runtime_debug.get("shield_triggered", False)):
                suggested_action = "shield_hold"
                status = "shield_hold"
            elif abs(yaw_error_deg) > max(10.0, float(self.args.yaw_step_deg) * 1.5):
                suggested_action = "yaw_left" if yaw_error_deg < 0.0 else "yaw_right"
                should_execute = True
            elif abs(vertical_error_cm) > max(10.0, float(self.args.vertical_step_cm) * 0.5):
                suggested_action = "down" if vertical_error_cm < 0.0 else "up"
                should_execute = True
            elif waypoint_distance_cm > max(20.0, float(self.args.move_step_cm) * 0.75):
                suggested_action = "forward"
                should_execute = True
            else:
                suggested_action = "hold_position"
                status = "waypoint_arrived"
        elif bool(self.runtime_debug.get("shield_triggered", False)):
            suggested_action = "shield_hold"
            status = "shield_hold"
            should_execute = True

        return build_reflex_runtime_state(
            mode="heuristic_stub",
            policy_name=self.args.reflex_policy_name,
            source="local_heuristic",
            status=status,
            suggested_action=suggested_action,
            should_execute=should_execute,
            last_trigger=trigger,
            last_latency_ms=0.0,
            waypoint_distance_cm=waypoint_distance_cm,
            yaw_error_deg=yaw_error_deg,
            vertical_error_cm=vertical_error_cm,
            progress_to_waypoint_cm=progress_to_waypoint_cm,
            retrieval_cell_id=str(active_retrieval.get("cell_id", "")),
            retrieval_score=float(active_retrieval.get("retrieval_score", 0.0) or 0.0),
            retrieval_semantic_subgoal=str(active_retrieval.get("semantic_subgoal", "")),
            risk_score=float(self.runtime_debug.get("risk_score", 0.0)),
            shield_triggered=bool(self.runtime_debug.get("shield_triggered", False)),
        )

    def request_reflex_policy(self, *, trigger: str = "manual_request", archive_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.lock:
            archive_state = archive_snapshot if isinstance(archive_snapshot, dict) else self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
            pose = self.get_task_pose()
            current_waypoint = self.runtime_debug.get("current_waypoint")
            reflex_started = datetime.now().timestamp()

            if self.args.reflex_policy_url:
                reflex_request = build_reflex_request(
                    policy_name=self.args.reflex_policy_name,
                    frame_id=self.last_observation_id,
                    timestamp=self.last_observation_time,
                    task_label=self.current_task_label or "idle",
                    pose={
                        "x": pose[0],
                        "y": pose[1],
                        "z": pose[2],
                        "yaw": pose[3],
                    },
                    depth=self.last_depth_summary,
                    plan=self.current_plan,
                    current_waypoint=current_waypoint if isinstance(current_waypoint, dict) else None,
                    archive=archive_state,
                    runtime_debug=self.runtime_debug,
                    context={
                        "trigger": trigger,
                        "movement_yaw_mode": self.args.movement_yaw_mode,
                        "preview_mode": self.args.preview_mode,
                    },
                )
                req = request.Request(
                    f"{self.args.reflex_policy_url.rstrip('/')}{self.args.reflex_policy_endpoint}",
                    data=json.dumps(reflex_request).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with request.urlopen(req, timeout=self.args.reflex_policy_timeout_s) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))
                    payload = raw.get("reflex_runtime") if isinstance(raw, dict) and isinstance(raw.get("reflex_runtime"), dict) else raw
                    self.reflex_runtime = coerce_reflex_runtime_payload(
                        payload,
                        default_mode="external_policy_stub",
                        default_policy_name=self.args.reflex_policy_name,
                        default_source="external",
                    )
                    self.reflex_runtime["last_trigger"] = trigger
                    self.reflex_runtime["last_latency_ms"] = round((datetime.now().timestamp() - reflex_started) * 1000.0, 2)
                    return self.reflex_runtime
                except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    logger.warning("Reflex policy request failed, fallback to heuristic reflex runtime: %s", exc)
                    fallback = self.build_heuristic_reflex_runtime(archive_state, trigger=trigger)
                    fallback["source"] = "local_heuristic"
                    fallback["upstream_source"] = "external_error"
                    fallback["upstream_error"] = str(exc)
                    if fallback.get("status") == "idle":
                        fallback["status"] = "external_fallback"
                    fallback["last_latency_ms"] = round((datetime.now().timestamp() - reflex_started) * 1000.0, 2)
                    self.reflex_runtime = fallback
                    return self.reflex_runtime

            self.reflex_runtime = self.build_heuristic_reflex_runtime(archive_state, trigger=trigger)
            return self.reflex_runtime

    def sync_reflex_runtime(self, archive_snapshot: Optional[Dict[str, Any]] = None, *, trigger: str = "sync") -> Dict[str, Any]:
        self.reflex_runtime = self.build_heuristic_reflex_runtime(archive_snapshot, trigger=trigger)
        return self.reflex_runtime

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            task_pose = self.get_task_pose()
            env_yaw = self.get_env_yaw_deg()
            control_yaw = self.get_control_yaw_deg()
            archive_state = self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit))
            self.sync_search_runtime(archive_state)
            if not self.should_preserve_reflex_runtime():
                self.sync_reflex_runtime(archive_state, trigger="state_refresh")
            return {
                "status": "ok",
                "env_id": self.args.env_id,
                "player_name": self.player_name,
                "player_count": 1,
                "internal_player_count": len(self.env.unwrapped.player_list),
                "movement_yaw_mode": self.args.movement_yaw_mode,
                "last_action": self.last_action,
                "task_label": self.current_task_label,
                "observation": {
                    "frame_id": self.last_observation_id,
                    "timestamp": self.last_observation_time,
                },
                "pose": {
                    "x": task_pose[0],
                    "y": task_pose[1],
                    "z": task_pose[2],
                    "yaw": control_yaw,
                    "command_yaw": self.command_task_yaw_deg,
                    "task_yaw": task_pose[3],
                    "uav_yaw": env_yaw,
                },
                "preview_mode": self.args.preview_mode,
                "depth": self.last_depth_summary,
                "camera_info": self.last_depth_summary.get("camera_info", {}),
                "mission": self.current_mission,
                "search_runtime": self.search_runtime,
                "doorway_runtime": self.doorway_runtime,
                "person_evidence_runtime": self.person_evidence_runtime,
                "search_result": self.search_result,
                "language_memory_runtime": self.language_memory_runtime,
                "phase5_mission_manual": self.phase5_mission_manual,
                "plan": self.current_plan,
                "planner_runtime": self.plan_execution_state,
                "archive": archive_state,
                "reflex_runtime": self.reflex_runtime,
                "llm_action_runtime": self.llm_action_runtime,
                "reflex_execution": self.reflex_execution_state,
                "planner_executor_runtime": self.planner_executor_state,
                "takeover_runtime": self.takeover_state,
                "takeover_recent_events": self.takeover_events,
                "person_evidence_recent_events": self.person_evidence_events,
                "last_plan_request": {
                    "schema_version": self.last_plan_request.get("schema_version", ""),
                    "trigger": self.last_plan_request.get("trigger", ""),
                    "step_index": self.last_plan_request.get("step_index", 0),
                    "task_label": self.last_plan_request.get("task_label", ""),
                    "frame_id": self.last_plan_request.get("frame_id", ""),
                },
                "last_llm_action_request": {
                    "schema_version": self.last_llm_action_request.get("schema_version", ""),
                    "trigger": self.last_llm_action_request.get("trigger", ""),
                    "step_index": self.last_llm_action_request.get("step_index", 0),
                    "task_label": self.last_llm_action_request.get("task_label", ""),
                    "frame_id": self.last_llm_action_request.get("frame_id", ""),
                },
                "runtime_debug": self.runtime_debug,
                "last_capture": self.last_capture,
            }

    def _apply_motion_step(
        self,
        *,
        forward_cm: float = 0.0,
        right_cm: float = 0.0,
        up_cm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        action_name: str = "custom",
        action_origin: str = "manual",
        trigger_auto_plan: bool = True,
        trigger_reflex_update: bool = True,
        allow_reflex_assist: bool = False,
    ) -> Dict[str, Any]:
        with self.lock:
            before_takeover_snapshot = self.build_takeover_snapshot() if action_origin == "manual" else None
            manual_after_takeover_snapshot: Optional[Dict[str, Any]] = None
            post_assist_takeover_snapshot: Optional[Dict[str, Any]] = None
            x, y, z, _actual_task_yaw = self.get_task_pose()
            move_yaw_deg = self.get_control_yaw_deg()
            theta = np.radians(move_yaw_deg)
            delta_x = float(forward_cm * np.cos(theta) - right_cm * np.sin(theta))
            delta_y = float(forward_cm * np.sin(theta) + right_cm * np.cos(theta))
            new_pos = [x + delta_x, y + delta_y, z + up_cm]
            self.command_task_yaw_deg = normalize_angle_deg(self.command_task_yaw_deg + yaw_delta_deg)
            self.set_task_pose(new_pos, self.command_task_yaw_deg)
            self.invalidate_cached_observations()
            self.last_action = action_name
            self.plan_execution_state["step_index"] = int(self.plan_execution_state.get("step_index", 0)) + 1
            self.runtime_debug["local_policy_action"] = {
                "action_name": action_name,
                "action_origin": action_origin,
                "forward_cm": float(forward_cm),
                "right_cm": float(right_cm),
                "up_cm": float(up_cm),
                "yaw_delta_deg": float(yaw_delta_deg),
            }
            self.recent_action_history.append(
                {
                    "step_index": int(self.plan_execution_state.get("step_index", 0)),
                    "action_name": str(action_name or "idle"),
                    "action_origin": str(action_origin or "unknown"),
                    "timestamp": now_timestamp(),
                }
            )
            if len(self.recent_action_history) > 20:
                self.recent_action_history = self.recent_action_history[-20:]
            self.sync_archive_runtime()
            auto_plan: Optional[Dict[str, Any]] = None
            if trigger_auto_plan and self.should_auto_request_plan():
                auto_plan = self.request_plan(trigger="step_interval")
                logger.info(
                    "Auto planner triggered at step=%s next_trigger_step=%s planner=%s subgoal=%s",
                    self.plan_execution_state.get("step_index", 0),
                    self.plan_execution_state.get("next_auto_trigger_step", 0),
                    auto_plan.get("planner_name", "n/a") if isinstance(auto_plan, dict) else "n/a",
                    auto_plan.get("semantic_subgoal", "idle") if isinstance(auto_plan, dict) else "idle",
                )
            if trigger_reflex_update and self.args.reflex_auto_mode == "on_move":
                reflex_runtime = self.request_reflex_policy(trigger="on_move")
                logger.info(
                    "Auto reflex policy updated at step=%s policy=%s source=%s suggested=%s",
                    self.plan_execution_state.get("step_index", 0),
                    reflex_runtime.get("policy_name", "n/a"),
                    reflex_runtime.get("source", "unknown"),
                    reflex_runtime.get("suggested_action", "idle"),
                )
                manual_after_takeover_snapshot = self.build_takeover_snapshot() if action_origin == "manual" else None
                if allow_reflex_assist and self.args.reflex_execute_mode == "assist_step":
                    assist_result = self.execute_reflex_action(
                        trigger="assist_step",
                        refresh_policy=False,
                        allow_auto_plan=False,
                        sync_after_execution=True,
                    )
                    assisted_state = assist_result.get("state")
                    post_assist_takeover_snapshot = self.build_takeover_snapshot() if action_origin == "manual" else None
                    if isinstance(assisted_state, dict):
                        final_state = assisted_state
                    else:
                        final_state = self.get_state()
                else:
                    final_state = self.get_state()
            else:
                if action_origin == "manual":
                    manual_after_takeover_snapshot = self.build_takeover_snapshot()
                final_state = self.get_state()
            actual_task_yaw_after = self.get_task_pose()[3]
            actual_uav_yaw_after = self.get_env_yaw_deg()
            logger.info(
                "Remote action=%s local=(fwd=%.1f,right=%.1f,up=%.1f,yaw=%.1f) "
                "world=(dx=%.1f,dy=%.1f,dz=%.1f) pos=(%.1f, %.1f, %.1f) "
                "move_yaw=%.1f command_yaw=%.1f actual_task_yaw=%.1f actual_uav_yaw=%.1f",
                action_name,
                forward_cm,
                right_cm,
                up_cm,
                yaw_delta_deg,
                delta_x,
                delta_y,
                up_cm,
                new_pos[0],
                new_pos[1],
                new_pos[2],
                move_yaw_deg,
                self.command_task_yaw_deg,
                actual_task_yaw_after,
                actual_uav_yaw_after,
            )
            should_record_takeover = action_origin == "manual" and (
                self.args.reflex_execute_mode == "assist_step" or bool(self.takeover_state.get("active", False))
            )
            if should_record_takeover and before_takeover_snapshot is not None:
                self.record_manual_intervention(
                    action_name=action_name,
                    before_snapshot=before_takeover_snapshot,
                    after_snapshot=manual_after_takeover_snapshot or self.build_takeover_snapshot(),
                    post_assist_snapshot=post_assist_takeover_snapshot,
                    trigger="manual_action",
                )
                final_state = self.get_state()
            return final_state

    def execute_reflex_action(
        self,
        *,
        trigger: str = "manual_execute",
        refresh_policy: bool = True,
        allow_auto_plan: bool = True,
        sync_after_execution: bool = True,
    ) -> Dict[str, Any]:
        with self.lock:
            reflex_runtime = self.request_reflex_policy(trigger=trigger) if refresh_policy else self.reflex_runtime
            allowed, reason, requested_action = self.evaluate_reflex_execution_gate(reflex_runtime)
            source = str(reflex_runtime.get("source", "") or "")
            if not allowed:
                execution_state = self.update_reflex_execution_state(
                    status="blocked" if reason in ("source_not_allowed", "low_confidence", "high_risk", "shield_triggered", "opposes_manual_action") else "skipped",
                    reason=reason,
                    trigger=trigger,
                    requested_action=requested_action,
                    source=source,
                )
                return {
                    "status": "ok",
                    "executed": False,
                    "reason": reason,
                    "requested_action": requested_action,
                    "reflex_runtime": self.reflex_runtime,
                    "reflex_execution": execution_state,
                    "state": self.get_state(),
                }

            motion_payload = self.build_motion_payload_for_action(requested_action)
            if motion_payload is None:
                execution_state = self.update_reflex_execution_state(
                    status="skipped",
                    reason="unsupported_action",
                    trigger=trigger,
                    requested_action=requested_action,
                    source=source,
                )
                return {
                    "status": "ok",
                    "executed": False,
                    "reason": "unsupported_action",
                    "requested_action": requested_action,
                    "reflex_runtime": self.reflex_runtime,
                    "reflex_execution": execution_state,
                    "state": self.get_state(),
                }

            state = self._apply_motion_step(
                forward_cm=float(motion_payload.get("forward_cm", 0.0)),
                right_cm=float(motion_payload.get("right_cm", 0.0)),
                up_cm=float(motion_payload.get("up_cm", 0.0)),
                yaw_delta_deg=float(motion_payload.get("yaw_delta_deg", 0.0)),
                action_name=str(motion_payload.get("action_name", requested_action)),
                action_origin="reflex_auto",
                trigger_auto_plan=allow_auto_plan,
                trigger_reflex_update=False,
                allow_reflex_assist=False,
            )
            if sync_after_execution:
                reflex_runtime = self.request_reflex_policy(trigger="auto_execute_sync")
            execution_state = self.update_reflex_execution_state(
                status="executed",
                reason="ok",
                trigger=trigger,
                requested_action=requested_action,
                executed_action=str(motion_payload.get("action_name", requested_action)),
                source=source,
            )
            state = self.get_state()
            logger.info(
                "Reflex auto execution trigger=%s requested=%s executed=%s source=%s",
                trigger,
                requested_action,
                motion_payload.get("action_name", requested_action),
                source or "unknown",
            )
            return {
                "status": "ok",
                "executed": True,
                "reason": "ok",
                "requested_action": requested_action,
                "executed_action": str(motion_payload.get("action_name", requested_action)),
                "reflex_runtime": reflex_runtime,
                "reflex_execution": execution_state,
                "state": state,
            }

    def move_relative(
        self,
        forward_cm: float = 0.0,
        right_cm: float = 0.0,
        up_cm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        action_name: str = "custom",
    ) -> Dict[str, Any]:
        return self._apply_motion_step(
            forward_cm=forward_cm,
            right_cm=right_cm,
            up_cm=up_cm,
            yaw_delta_deg=yaw_delta_deg,
            action_name=action_name,
            action_origin="manual",
            trigger_auto_plan=True,
            trigger_reflex_update=True,
            allow_reflex_assist=True,
        )

    def get_frame_jpeg(self) -> bytes:
        with self.lock:
            if self.last_raw_frame is None:
                frame = self.refresh_preview_only()
            else:
                frame = self.last_raw_frame
            encode_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)],
            )
            if not encode_ok:
                raise RuntimeError("Failed to encode preview frame")
            return encoded.tobytes()

    def capture_frame(self, label: Optional[str] = None, task_label: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            if self.last_raw_frame is None or self.last_depth_frame is None:
                self.refresh_observations()
            if self.last_raw_frame is None:
                raise RuntimeError("No frame available for capture")
            requested_task_label = str(task_label or "").strip()
            if requested_task_label and requested_task_label != self.current_task_label:
                self.set_task_label(requested_task_label)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            capture_id = f"capture_{timestamp}{suffix}"
            image_path = os.path.join(self.args.capture_dir, f"{capture_id}_rgb.png")
            depth_path = os.path.join(self.args.capture_dir, f"{capture_id}_depth_cm.png")
            depth_preview_path = os.path.join(self.args.capture_dir, f"{capture_id}_depth_preview.png")
            camera_info_path = os.path.join(self.args.capture_dir, f"{capture_id}_camera_info.json")
            meta_path = os.path.join(self.args.capture_dir, f"{capture_id}_bundle.json")

            cv2.imwrite(image_path, self.last_raw_frame)
            if self.last_depth_frame is not None:
                depth_to_save = np.clip(
                    np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                    65535.0,
                )
                cv2.imwrite(depth_path, depth_to_save.astype(np.uint16))
                depth_preview = render_depth_preview(
                    self.last_depth_frame,
                    int(self.last_depth_summary.get("image_width", self.args.depth_preview_width)) or self.args.depth_preview_width,
                    int(self.last_depth_summary.get("image_height", self.args.depth_preview_height)) or self.args.depth_preview_height,
                    min_depth_cm=float(self.args.depth_min_cm),
                    max_depth_cm=float(self.args.depth_max_cm),
                    source_mode=self.last_depth_summary.get("source_mode", "policy_depth"),
                )
                cv2.imwrite(depth_preview_path, depth_preview)

            camera_info = self.last_depth_summary.get("camera_info", {})
            with open(camera_info_path, "w", encoding="utf-8") as f:
                json.dump(camera_info, f, indent=2)

            state = self.get_state()
            reflex_sample = build_reflex_sample(
                capture_id=capture_id,
                task_label=self.current_task_label,
                action_label=self.last_action,
                pose=state["pose"],
                current_waypoint=self.runtime_debug.get("current_waypoint") if isinstance(self.runtime_debug.get("current_waypoint"), dict) else None,
                plan=self.current_plan,
                archive=state.get("archive", {}),
                reflex_runtime=self.reflex_runtime,
                runtime_debug=self.runtime_debug,
            )
            metadata = {
                "capture_id": capture_id,
                "capture_time": timestamp,
                "env_id": self.args.env_id,
                "dataset_schema_version": "phase3.capture_bundle.v1",
                "search_capture_schema_version": "phase4.search_capture_bundle.v1",
                "task_label": self.current_task_label,
                "action_label": self.last_action,
                "rgb_image_path": image_path,
                "depth_image_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "metadata_path": meta_path,
                "pose": state["pose"],
                "depth": self.last_depth_summary,
                "camera_info": camera_info,
                "plan": self.current_plan,
                "mission": self.current_mission,
                "search_runtime": self.search_runtime,
                "doorway_runtime": self.doorway_runtime,
                "person_evidence_runtime": self.person_evidence_runtime,
                "search_result": self.search_result,
                "language_memory_runtime": self.language_memory_runtime,
                "phase5_mission_manual": self.phase5_mission_manual,
                "runtime_debug": self.runtime_debug,
                "archive": self.archive_runtime.get_state(limit=int(self.args.archive_recent_limit)),
                "reflex_runtime": self.reflex_runtime,
                "llm_action_runtime": self.llm_action_runtime,
                "reflex_execution": self.reflex_execution_state,
                "planner_executor_runtime": self.planner_executor_state,
                "takeover_runtime": self.takeover_state,
                "takeover_recent_events": self.takeover_events,
                "person_evidence_recent_events": self.person_evidence_events,
                "reflex_sample": reflex_sample,
                "preview_mode": self.args.preview_mode,
                "task_json": self.args.task_json,
                "label": label,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self.last_capture = {
                "capture_id": capture_id,
                "image_path": image_path,
                "depth_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "meta_path": meta_path,
                "capture_time": timestamp,
            }
            logger.info("Captured RGB/depth bundle: %s", meta_path)
            return {
                "status": "ok",
                "capture_id": capture_id,
                "image_path": image_path,
                "depth_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "meta_path": meta_path,
                "task_label": self.current_task_label,
                "pose": metadata["pose"],
                "depth": self.last_depth_summary,
                "camera_info": camera_info,
                "plan": self.current_plan,
                "mission": metadata["mission"],
                "search_runtime": metadata["search_runtime"],
                "doorway_runtime": metadata["doorway_runtime"],
                "person_evidence_runtime": metadata["person_evidence_runtime"],
                "search_result": metadata["search_result"],
                "language_memory_runtime": metadata["language_memory_runtime"],
                "phase5_mission_manual": metadata["phase5_mission_manual"],
                "archive": metadata["archive"],
                "reflex_runtime": metadata["reflex_runtime"],
                "llm_action_runtime": metadata["llm_action_runtime"],
                "reflex_execution": metadata["reflex_execution"],
                "planner_executor_runtime": metadata["planner_executor_runtime"],
                "takeover_runtime": metadata["takeover_runtime"],
                "takeover_recent_events": metadata["takeover_recent_events"],
                "person_evidence_recent_events": metadata["person_evidence_recent_events"],
                "reflex_sample": metadata["reflex_sample"],
            }

    def shutdown(self) -> Dict[str, Any]:
        logger.info("Shutdown requested")
        if self.httpd is not None:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()
        return {"status": "ok", "message": "Shutdown requested"}

    def build_planner_config_url(self) -> str:
        planner_url = str(self.args.planner_url or "").strip()
        if not planner_url:
            raise RuntimeError("No external planner_url configured.")
        return f"{planner_url.rstrip('/')}/config"

    def get_external_planner_config(self) -> Dict[str, Any]:
        req = request.Request(self.build_planner_config_url(), method="GET")
        with request.urlopen(req, timeout=self.args.planner_timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Planner config response must be a JSON object.")
        payload["planner_request_timeout_s"] = float(self.args.planner_timeout_s)
        return payload

    def update_external_planner_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        update_payload = dict(payload or {})
        if "planner_request_timeout_s" in update_payload:
            timeout_value = float(update_payload.pop("planner_request_timeout_s"))
            self.args.planner_timeout_s = max(0.5, timeout_value)
        req = request.Request(
            self.build_planner_config_url(),
            data=json.dumps(update_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.args.planner_timeout_s) as resp:
            response_payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise RuntimeError("Planner config update response must be a JSON object.")
        response_payload["planner_request_timeout_s"] = float(self.args.planner_timeout_s)
        return response_payload

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def make_handler(backend: UAVControlBackend):
    class ControlRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
                logger.warning("Client disconnected before JSON response was sent: %s", exc)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
                logger.warning("Client disconnected before binary response was sent: %s", exc)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/health"):
                    self._send_json({"status": "ok"})
                elif parsed.path == "/state":
                    self._send_json(backend.get_state())
                elif parsed.path == "/frame":
                    self._send_bytes(backend.get_frame_jpeg(), "image/jpeg")
                elif parsed.path in ("/depth", "/depth_frame"):
                    self._send_bytes(backend.get_depth_frame_jpeg(), "image/jpeg")
                elif parsed.path in ("/depth_raw", "/depth_raw.png"):
                    self._send_bytes(backend.get_depth_raw_png(), "image/png")
                elif parsed.path == "/camera_info":
                    self._send_json({"status": "ok", "camera_info": backend.get_camera_info()})
                elif parsed.path == "/plan":
                    self._send_json({"status": "ok", "plan": backend.current_plan})
                elif parsed.path == "/archive":
                    self._send_json({"status": "ok", "archive": backend.archive_runtime.get_state(limit=int(backend.args.archive_recent_limit))})
                elif parsed.path == "/reflex":
                    self._send_json({"status": "ok", "reflex_runtime": backend.reflex_runtime})
                elif parsed.path == "/llm_action":
                    self._send_json({"status": "ok", "llm_action_runtime": backend.llm_action_runtime})
                elif parsed.path == "/reflex_execution":
                    self._send_json({"status": "ok", "reflex_execution": backend.reflex_execution_state})
                elif parsed.path == "/planner_executor":
                    self._send_json({"status": "ok", "planner_executor_runtime": backend.planner_executor_state})
                elif parsed.path == "/takeover":
                    self._send_json(
                        {
                            "status": "ok",
                            "takeover_runtime": backend.takeover_state,
                            "takeover_recent_events": backend.takeover_events,
                        }
                    )
                elif parsed.path == "/person_evidence":
                    self._send_json(
                        {
                            "status": "ok",
                            "person_evidence_runtime": backend.person_evidence_runtime,
                            "search_result": backend.search_result,
                            "person_evidence_recent_events": backend.person_evidence_events,
                        }
                    )
                elif parsed.path == "/planner_config":
                    self._send_json(backend.get_external_planner_config())
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
                logger.warning("GET %s client disconnected: %s", parsed.path, exc)
            except Exception as exc:
                logger.exception("GET %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                data = self._read_json_body()
                if parsed.path == "/move_relative":
                    payload = backend.move_relative(
                        forward_cm=float(data.get("forward_cm", 0.0)),
                        right_cm=float(data.get("right_cm", 0.0)),
                        up_cm=float(data.get("up_cm", 0.0)),
                        yaw_delta_deg=float(data.get("yaw_delta_deg", 0.0)),
                        action_name=str(data.get("action_name", "custom")),
                    )
                    self._send_json(payload)
                elif parsed.path == "/capture":
                    self._send_json(
                        backend.capture_frame(
                            label=data.get("label"),
                            task_label=data.get("task_label"),
                        )
                    )
                elif parsed.path == "/task":
                    self._send_json(backend.set_task_label(str(data.get("task_label", ""))))
                elif parsed.path == "/plan":
                    plan = backend.set_plan_state(data if isinstance(data, dict) else {})
                    self._send_json({"status": "ok", "plan": plan})
                elif parsed.path == "/request_plan":
                    plan = backend.request_plan(task_label=data.get("task_label"))
                    self._send_json({"status": "ok", "plan": plan})
                elif parsed.path == "/request_reflex":
                    reflex_runtime = backend.request_reflex_policy(trigger=str(data.get("trigger", "manual_request")))
                    self._send_json({"status": "ok", "reflex_runtime": reflex_runtime})
                elif parsed.path == "/request_llm_action":
                    llm_action_runtime = backend.request_llm_action(
                        trigger=str(data.get("trigger", "manual_request") or "manual_request"),
                        refresh_observations=bool(data.get("refresh_observations", True)),
                    )
                    self._send_json({"status": "ok", "llm_action_runtime": llm_action_runtime})
                elif parsed.path == "/execute_reflex":
                    self._send_json(
                        backend.execute_reflex_action(
                            trigger=str(data.get("trigger", "manual_execute")),
                            refresh_policy=bool(data.get("refresh_policy", True)),
                            allow_auto_plan=bool(data.get("allow_auto_plan", True)),
                            sync_after_execution=bool(data.get("sync_after_execution", True)),
                        )
                    )
                elif parsed.path == "/execute_llm_action":
                    self._send_json(
                        backend.execute_llm_action(
                            trigger=str(data.get("trigger", "manual_execute") or "manual_execute"),
                            refresh_action=bool(data.get("refresh_action", True)),
                            allow_auto_plan=bool(data.get("allow_auto_plan", False)),
                        )
                    )
                elif parsed.path == "/execute_plan_segment":
                    self._send_json(
                        backend.execute_plan_segment(
                            step_budget=int(data.get("step_budget", 5) or 5),
                            refresh_plan=bool(data.get("refresh_plan", True)),
                            plan_refresh_interval_steps=int(data.get("plan_refresh_interval_steps", 0) or 0),
                            allow_reflex=bool(data.get("allow_reflex", True)),
                            trigger=str(data.get("trigger", "manual_segment") or "manual_segment"),
                        )
                    )
                elif parsed.path == "/execute_llm_action_segment":
                    self._send_json(
                        backend.execute_llm_action_segment(
                            step_budget=int(data.get("step_budget", 5) or 5),
                            refresh_plan=bool(data.get("refresh_plan", True)),
                            plan_refresh_interval_steps=int(data.get("plan_refresh_interval_steps", 0) or 0),
                            trigger=str(data.get("trigger", "manual_llm_action_segment") or "manual_llm_action_segment"),
                        )
                    )
                elif parsed.path == "/takeover":
                    action = str(data.get("action", "start")).strip().lower()
                    reason = str(data.get("reason", "") or "").strip()
                    note = str(data.get("note", "") or "").strip()
                    if action == "start":
                        takeover_state = backend.start_takeover(
                            reason=reason or DEFAULT_TAKEOVER_REASON,
                            note=note,
                            trigger="manual_request",
                        )
                    elif action == "end":
                        takeover_state = backend.end_takeover(
                            reason=reason or "resolved",
                            note=note,
                            trigger="manual_request",
                        )
                    else:
                        raise ValueError(f"Unsupported takeover action: {action}")
                    self._send_json(
                        {
                            "status": "ok",
                            "takeover_runtime": takeover_state,
                            "takeover_recent_events": backend.takeover_events,
                        }
                    )
                elif parsed.path == "/person_evidence":
                    self._send_json(
                        backend.record_person_evidence(
                            action=str(data.get("action", "") or ""),
                            note=str(data.get("note", "") or ""),
                            capture_label=str(data.get("capture_label", "") or ""),
                            confidence=data.get("confidence"),
                        )
                    )
                elif parsed.path == "/planner_config":
                    self._send_json(backend.update_external_planner_config(data if isinstance(data, dict) else {}))
                elif parsed.path == "/runtime_debug":
                    debug_state = backend.update_runtime_debug(
                        current_waypoint=data.get("current_waypoint") if isinstance(data.get("current_waypoint"), dict) else None,
                        local_policy_action=data.get("local_policy_action") if isinstance(data.get("local_policy_action"), dict) else None,
                        risk_score=data.get("risk_score"),
                        shield_triggered=data.get("shield_triggered"),
                        archive_cell_id=data.get("archive_cell_id"),
                    )
                    self._send_json({"status": "ok", "runtime_debug": debug_state})
                elif parsed.path == "/shutdown":
                    self._send_json(backend.shutdown())
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
                logger.warning("POST %s client disconnected: %s", parsed.path, exc)
            except Exception as exc:
                logger.exception("POST %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

    return ControlRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Unreal and expose UAV control endpoints")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5020, help="HTTP port for control")
    parser.add_argument("--env_id", default="UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0", help="Gym environment id")
    parser.add_argument("--env_bin_win", default=None, help="Override env_bin_win for the chosen environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--time_dilation", type=int, default=10, help="Simulator time dilation")
    parser.add_argument("--window_width", type=int, default=640, help="UnrealCV capture width")
    parser.add_argument("--window_height", type=int, default=480, help="UnrealCV capture height")
    parser.add_argument("--viewport_mode", default="free", choices=["first_person", "third_person", "free"], help="Native Unreal viewport mode")
    parser.add_argument("--viewport_offset_x", type=float, default=-220.0, help="Third-person viewport X offset")
    parser.add_argument("--viewport_offset_y", type=float, default=0.0, help="Third-person viewport Y offset")
    parser.add_argument("--viewport_offset_z", type=float, default=90.0, help="Third-person viewport Z offset")
    parser.add_argument("--viewport_roll", type=float, default=0.0, help="Third-person viewport roll")
    parser.add_argument("--viewport_pitch", type=float, default=-12.0, help="Third-person viewport pitch")
    parser.add_argument("--viewport_yaw", type=float, default=0.0, help="Third-person viewport yaw")
    parser.add_argument("--free_view_offset_x", type=float, default=-220.0, help="Free-view follow X offset")
    parser.add_argument("--free_view_offset_y", type=float, default=140.0, help="Free-view follow Y offset")
    parser.add_argument("--free_view_offset_z", type=float, default=50.0, help="Free-view follow Z offset")
    parser.add_argument("--free_view_roll", type=float, default=0.0, help="Free-view roll")
    parser.add_argument("--free_view_pitch", type=float, default=0.0, help="Free-view pitch offset")
    parser.add_argument("--free_view_yaw", type=float, default=0.0, help="Free-view yaw offset")
    parser.add_argument("--follow_free_view", action="store_true", help="Place the native Unreal free view near the UAV once at startup")
    parser.add_argument("--preview_mode", default="first_person", choices=["first_person", "third_person"], help="Frame/capture mode")
    parser.add_argument("--preview_offset_x", type=float, default=-260.0, help="Third-person preview X offset")
    parser.add_argument("--preview_offset_y", type=float, default=0.0, help="Third-person preview Y offset")
    parser.add_argument("--preview_offset_z", type=float, default=120.0, help="Third-person preview Z offset")
    parser.add_argument("--preview_roll", type=float, default=0.0, help="Third-person preview roll")
    parser.add_argument("--preview_pitch", type=float, default=-12.0, help="Third-person preview pitch")
    parser.add_argument("--preview_yaw", type=float, default=0.0, help="Third-person preview yaw offset")
    parser.add_argument("--movement_yaw_mode", default="task", choices=["task", "uav", "camera"], help="Yaw frame used by WASD translation")
    parser.add_argument("--move_step_cm", type=float, default=20.0, help="Reference local translation step used by reflex heuristics")
    parser.add_argument("--vertical_step_cm", type=float, default=20.0, help="Reference vertical step used by reflex heuristics")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="Reference yaw step used by reflex heuristics")
    parser.add_argument("--frame_jpeg_quality", type=int, default=90, help="JPEG quality used by /frame and /depth_frame")
    parser.add_argument("--default_depth_fov_deg", type=float, default=90.0, help="Fallback FOV when UnrealCV returns malformed camera FOV")
    parser.add_argument("--depth_camera_frame_id", default="PX4/CameraDepth_optical", help="Frame id used in generated depth camera_info")
    parser.add_argument("--depth_min_cm", type=float, default=20.0, help="Minimum depth kept in summaries/previews")
    parser.add_argument("--depth_max_cm", type=float, default=1200.0, help="Maximum depth kept in summaries/previews")
    parser.add_argument("--depth_preview_width", type=int, default=480, help="Depth preview render width")
    parser.add_argument("--depth_preview_height", type=int, default=360, help="Depth preview render height")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Directory used for server-side captures")
    parser.add_argument("--takeover_log_dir", default="./phase4_takeover_logs", help="Directory used for takeover/intervention JSONL logs")
    parser.add_argument("--takeover_recent_limit", type=int, default=12, help="How many recent takeover events to expose in /state")
    parser.add_argument("--search_log_dir", default="./phase4_search_logs", help="Directory used for person-evidence/search JSONL logs")
    parser.add_argument("--search_recent_limit", type=int, default=12, help="How many recent person-evidence events to expose in /state")
    parser.add_argument("--fixed_spawn_pose_file", default="./uav_fixed_spawn_pose.json", help="Persistent fixed UAV spawn pose file used when task_json/spawn args are absent")
    parser.add_argument("--default_task_label", default="", help="Default task label used by capture/planner endpoints")
    parser.add_argument("--planner_name", default="phase2-planner", help="Planner name stored in /plan state")
    parser.add_argument("--planner_url", default=None, help="Optional external planner base URL")
    parser.add_argument("--planner_endpoint", default="/plan", help="Planner endpoint path used with planner_url")
    parser.add_argument("--planner_action_endpoint", default="/action", help="Pure LLM action endpoint path used with planner_url")
    parser.add_argument("--planner_timeout_s", type=float, default=5.0, help="Timeout for planner requests")
    parser.add_argument(
        "--planner_auto_mode",
        default="manual",
        choices=["manual", "k_step"],
        help="manual=only request plan on /request_plan, k_step=auto request every K local control steps",
    )
    parser.add_argument("--planner_interval_steps", type=int, default=5, help="Target replan interval for debug/metadata")
    parser.add_argument("--default_plan_distance_cm", type=float, default=300.0, help="Fallback heuristic waypoint distance")
    parser.add_argument("--default_waypoint_radius_cm", type=float, default=60.0, help="Fallback waypoint acceptance radius")
    parser.add_argument("--default_sector_count", type=int, default=8, help="Fallback discrete sector count")
    parser.add_argument("--archive_pos_bin_cm", type=float, default=200.0, help="Quantization bin size for archive x/y/z")
    parser.add_argument("--archive_yaw_bin_deg", type=float, default=30.0, help="Quantization bin size for archive yaw")
    parser.add_argument("--archive_depth_bin_cm", type=float, default=100.0, help="Quantization bin size for archive depth signature")
    parser.add_argument("--archive_recent_limit", type=int, default=6, help="How many recent archive cells to expose")
    parser.add_argument("--archive_retrieval_limit", type=int, default=3, help="How many archive candidates to include in planner context")
    parser.add_argument("--risk_near_cm", type=float, default=250.0, help="Distance threshold used for heuristic collision risk estimation")
    parser.add_argument("--shield_risk_threshold", type=float, default=0.85, help="Heuristic shield trigger threshold for runtime debug")
    parser.add_argument("--reflex_policy_name", default="phase3-reflex", help="Policy name stored in reflex runtime state")
    parser.add_argument("--reflex_policy_url", default=None, help="Optional external reflex policy base URL")
    parser.add_argument("--reflex_policy_endpoint", default="/reflex_policy", help="Reflex policy endpoint path used with reflex_policy_url")
    parser.add_argument("--reflex_policy_timeout_s", type=float, default=3.0, help="Timeout for reflex policy requests")
    parser.add_argument("--reflex_auto_mode", default="manual", choices=["manual", "on_move"], help="manual=only request reflex on /request_reflex, on_move=refresh reflex state after each move")
    parser.add_argument(
        "--reflex_execute_mode",
        default="manual",
        choices=["manual", "assist_step"],
        help="manual=only execute reflex when /execute_reflex is called, assist_step=after each manual move optionally execute one gated reflex step",
    )
    parser.add_argument(
        "--reflex_execute_confidence_threshold",
        type=float,
        default=0.35,
        help="Minimum external-model confidence required for reflex auto execution",
    )
    parser.add_argument(
        "--reflex_execute_max_risk",
        type=float,
        default=0.75,
        help="Maximum allowed risk score for reflex auto execution",
    )
    parser.add_argument(
        "--reflex_execute_allow_heuristic",
        action="store_true",
        help="Allow local/heuristic reflex outputs to be executed by /execute_reflex or assist_step",
    )
    parser.add_argument("--task_json", default=None, help="Optional task JSON used to place the UAV and target")
    parser.add_argument("--spawn_x", type=float, default=None, help="Optional UAV spawn x")
    parser.add_argument("--spawn_y", type=float, default=None, help="Optional UAV spawn y")
    parser.add_argument("--spawn_z", type=float, default=None, help="Optional UAV spawn z")
    parser.add_argument("--spawn_yaw", type=float, default=None, help="Optional UAV spawn yaw in task-json convention")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    backend = UAVControlBackend(args)
    handler = make_handler(backend)
    server = HTTPServer((args.host, args.port), handler)
    backend.httpd = server
    logger.info("UAV control server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
