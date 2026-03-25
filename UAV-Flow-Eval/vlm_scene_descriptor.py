"""
Standalone multimodal scene descriptor for Phase 6 experiments.

This module lets us test the real VLM/LLM API path without going through the
full control stack. It accepts RGB and depth images, builds a composite image,
queries the configured model, and asks for a structured scene interpretation
plus waypoint hints.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from llm_planner_client import LLMPlannerClient, LLMPlannerClientError, LLMPlannerConfig
from runtime_interfaces import build_vlm_scene_runtime_state


class VLMSceneDescriptorError(RuntimeError):
    """Raised when the standalone VLM descriptor cannot produce a valid result."""


def get_default_prompt_log_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "phase6_prompt_logs")


SCENE_STATE_VALUES = ["outside_house", "threshold_zone", "inside_house", "unknown"]
STAGE_VALUES = [
    "outside_localization",
    "target_house_verification",
    "entry_search",
    "approach_entry",
    "cross_entry",
    "indoor_room_search",
    "suspect_verification",
    "mission_report",
]


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_scene_state(*candidates: str) -> str:
    for item in candidates:
        value = str(item or "").strip().lower()
        if value in SCENE_STATE_VALUES:
            return value
    return "unknown"


def _room_type_hint_from_task(task_label: str, search_runtime: Dict[str, Any]) -> str:
    text = str(task_label or "").lower()
    for keyword, label in [
        ("bedroom", "bedroom"),
        ("kitchen", "kitchen"),
        ("bathroom", "bathroom"),
        ("living", "living_room"),
        ("hallway", "hallway"),
        ("corridor", "hallway"),
        ("stairs", "stairs"),
        ("door", "doorway"),
    ]:
        if keyword in text:
            return label
    priority_regions = search_runtime.get("priority_regions")
    if isinstance(priority_regions, list):
        for item in priority_regions:
            if isinstance(item, dict) and str(item.get("room_type", "")).strip():
                return str(item.get("room_type"))
    return ""


def build_vlm_scene_runtime(
    *,
    task_label: str,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    scene_waypoint_runtime: Optional[Dict[str, Any]] = None,
    phase5_mission_manual: Optional[Dict[str, Any]] = None,
    phase6_mission_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
    reference_match_runtime: Optional[Dict[str, Any]] = None,
    current_plan: Optional[Dict[str, Any]] = None,
    depth_summary: Optional[Dict[str, Any]] = None,
    rgb_frame: Any = None,
) -> Dict[str, Any]:
    mission = _coerce_dict(mission)
    search_runtime = _coerce_dict(search_runtime)
    doorway_runtime = _coerce_dict(doorway_runtime)
    scene_waypoint_runtime = _coerce_dict(scene_waypoint_runtime)
    phase5_mission_manual = _coerce_dict(phase5_mission_manual)
    phase6_mission_runtime = _coerce_dict(phase6_mission_runtime)
    person_evidence_runtime = _coerce_dict(person_evidence_runtime)
    search_result = _coerce_dict(search_result)
    reference_match_runtime = _coerce_dict(reference_match_runtime)
    current_plan = _coerce_dict(current_plan)
    depth_summary = _coerce_dict(depth_summary)

    scene_state = _coerce_scene_state(
        str(scene_waypoint_runtime.get("scene_state", "")),
        "outside_house" if doorway_runtime.get("traversable_candidate_count", 0) else "",
        "outside_house" if doorway_runtime.get("candidate_count", 0) else "",
        "inside_house" if search_runtime.get("visited_region_count", 0) else "",
        str(phase6_mission_runtime.get("scene_state", "")),
    )
    entry_visible = bool(
        doorway_runtime.get("candidate_count", 0)
        or scene_waypoint_runtime.get("entry_door_visible", False)
    )
    entry_traversable = bool(
        doorway_runtime.get("traversable_candidate_count", 0)
        or scene_waypoint_runtime.get("entry_door_traversable", False)
    )
    room_type_hint = _room_type_hint_from_task(task_label, search_runtime)
    unexplored_direction = str(scene_waypoint_runtime.get("priority_region", "") or "").strip()
    mission_type = str(mission.get("mission_type", "semantic_navigation") or "semantic_navigation")
    current_subgoal = str(
        search_runtime.get("current_search_subgoal", "")
        or phase6_mission_runtime.get("active_stage_id", "")
        or phase5_mission_manual.get("active_stage_id", "")
        or current_plan.get("semantic_subgoal", "")
        or "idle"
    )
    result_status = str(search_result.get("result_status", "unknown") or "unknown")
    evidence_status = str(person_evidence_runtime.get("evidence_status", "unknown") or "unknown")
    best_label = str(doorway_runtime.get("best_candidate_label", "") or "none")
    best_conf = float(doorway_runtime.get("best_candidate_confidence", 0.0) or 0.0)
    min_depth = float(depth_summary.get("min_depth_cm", 0.0) or 0.0)
    max_depth = float(depth_summary.get("max_depth_cm", 0.0) or 0.0)
    reference_score = float(reference_match_runtime.get("score", 0.0) or 0.0)
    target_visible = bool(reference_match_runtime.get("matched", False))
    scene_tags: List[str] = []
    if scene_state != "unknown":
        scene_tags.append(scene_state)
    if entry_visible:
        scene_tags.append("entry_visible")
    if entry_traversable:
        scene_tags.append("entry_traversable")
    if room_type_hint:
        scene_tags.append(room_type_hint)
    if target_visible:
        scene_tags.append("target_house_visible")
    if evidence_status and evidence_status not in {"idle", "unknown"}:
        scene_tags.append(f"evidence:{evidence_status}")

    if entry_visible:
        scene_description = (
            f"Entry candidate {best_label} is visible with confidence {best_conf:.2f}; "
            f"scene_state={scene_state}; current_subgoal={current_subgoal}."
        )
    else:
        scene_description = (
            f"No clear doorway is visible yet; scene_state={scene_state}; "
            f"current_subgoal={current_subgoal}."
        )
    semantic_text = (
        f"mission={mission_type}; task={task_label or '-'}; scene={scene_state}; "
        f"entry_visible={int(entry_visible)}; entry_traversable={int(entry_traversable)}; "
        f"best_entry={best_label}; target_visible={int(target_visible)}; "
        f"result={result_status}; evidence={evidence_status}; "
        f"depth_range_cm={min_depth:.1f}-{max_depth:.1f}; "
        f"priority={unexplored_direction or 'none'}; "
        f"reference_score={reference_score:.3f}"
    )
    occlusion_summary = "open_threshold" if entry_traversable else ("partial_occlusion" if entry_visible else "unknown")
    confidence = max(best_conf, float(scene_waypoint_runtime.get("confidence", 0.0) or 0.0))

    return build_vlm_scene_runtime_state(
        mission_id=str(mission.get("mission_id", "") or ""),
        mission_type=mission_type,
        task_label=task_label,
        status="ok",
        source="heuristic_scene_fusion",
        model_name="heuristic_scene_fusion_v0",
        scene_description=scene_description,
        semantic_text=semantic_text,
        scene_tags=scene_tags,
        scene_state=scene_state,
        entry_visible=entry_visible,
        entry_traversable=entry_traversable,
        room_type_hint=room_type_hint,
        occlusion_summary=occlusion_summary,
        unexplored_direction=unexplored_direction,
        confidence=confidence,
    )


VLM_SCENE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_state": {"type": "string", "enum": SCENE_STATE_VALUES},
        "active_stage": {"type": "string", "enum": STAGE_VALUES},
        "target_house_visible": {"type": "boolean"},
        "entry_door_visible": {"type": "boolean"},
        "entry_door_traversable": {"type": "boolean"},
        "scene_description": {"type": "string"},
        "likely_unexplored_regions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "x_hint": {"type": "number"},
                    "y_hint": {"type": "number"},
                    "z_offset_cm": {"type": "number"},
                    "yaw_hint_deg": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["label", "x_hint", "y_hint", "z_offset_cm", "yaw_hint_deg", "rationale"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
    },
    "required": [
        "scene_state",
        "active_stage",
        "target_house_visible",
        "entry_door_visible",
        "entry_door_traversable",
        "scene_description",
        "likely_unexplored_regions",
        "next_waypoints",
        "confidence",
    ],
    "additionalProperties": False,
}


def _truncate(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _strip_fenced_json(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return raw


def _sanitize_json_candidate(text: str) -> str:
    raw = str(text or "").replace("\ufeff", "").strip()
    if not raw:
        return raw
    parts: List[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if escaped:
            parts.append(char)
            escaped = False
            continue
        if char == "\\":
            parts.append(char)
            escaped = True
            continue
        if char == '"':
            parts.append(char)
            in_string = not in_string
            continue
        if in_string:
            if char == "\n":
                parts.append("\\n")
                continue
            if char == "\r":
                parts.append("\\r")
                continue
            if char == "\t":
                parts.append("\\t")
                continue
        parts.append(char)
    sanitized = "".join(parts)
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    if sanitized.count('"') % 2 == 1:
        sanitized += '"'
    bracket_delta = sanitized.count("[") - sanitized.count("]")
    if bracket_delta > 0:
        sanitized += "]" * bracket_delta
    brace_delta = sanitized.count("{") - sanitized.count("}")
    if brace_delta > 0:
        sanitized += "}" * brace_delta
    return sanitized


def parse_vlm_scene_json_response(text: str) -> Dict[str, Any]:
    candidate = _strip_fenced_json(text)
    if not candidate:
        raise VLMSceneDescriptorError("VLM scene descriptor returned empty text.")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        repaired_candidate = _sanitize_json_candidate(candidate)
        try:
            payload = json.loads(repaired_candidate)
        except json.JSONDecodeError as exc:
            raise VLMSceneDescriptorError(f"VLM scene descriptor returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VLMSceneDescriptorError("VLM scene descriptor JSON response must be an object.")
    return payload


def normalize_descriptor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    scene_state = str(payload.get("scene_state", "unknown") or "unknown").strip().lower()
    if scene_state not in SCENE_STATE_VALUES:
        scene_state = "unknown"

    active_stage = str(payload.get("active_stage", "outside_localization") or "outside_localization").strip()
    if active_stage not in STAGE_VALUES:
        active_stage = "outside_localization"

    likely_unexplored_regions = [
        str(item).strip()
        for item in (payload.get("likely_unexplored_regions") or [])
        if str(item).strip()
    ]

    normalized_waypoints: List[Dict[str, Any]] = []
    for item in payload.get("next_waypoints") or []:
        if not isinstance(item, dict):
            continue
        try:
            x_hint = float(item.get("x_hint", 0.5))
            y_hint = float(item.get("y_hint", 0.5))
            z_offset_cm = float(item.get("z_offset_cm", 0.0))
            yaw_hint_deg = float(item.get("yaw_hint_deg", 0.0))
        except (TypeError, ValueError):
            continue
        normalized_waypoints.append(
            {
                "label": str(item.get("label", "waypoint") or "waypoint").strip(),
                "x_hint": max(0.0, min(1.0, x_hint)),
                "y_hint": max(0.0, min(1.0, y_hint)),
                "z_offset_cm": max(-200.0, min(200.0, z_offset_cm)),
                "yaw_hint_deg": max(-180.0, min(180.0, yaw_hint_deg)),
                "rationale": str(item.get("rationale", "") or "").strip(),
            }
        )

    return {
        "scene_state": scene_state,
        "active_stage": active_stage,
        "target_house_visible": bool(payload.get("target_house_visible", False)),
        "entry_door_visible": bool(payload.get("entry_door_visible", False)),
        "entry_door_traversable": bool(payload.get("entry_door_traversable", False)),
        "scene_description": str(payload.get("scene_description", "") or "").strip(),
        "likely_unexplored_regions": likely_unexplored_regions,
        "next_waypoints": normalized_waypoints,
        "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
    }


def load_image_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise VLMSceneDescriptorError(f"Failed to load image: {path}")
    return image


def _draw_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (max(150, 12 * len(label)), 30), (20, 20, 20), thickness=-1)
    cv2.putText(output, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def build_composite_image(
    *,
    rgb_path: str,
    depth_path: str,
    reference_path: str = "",
) -> np.ndarray:
    rgb = _draw_label(load_image_bgr(rgb_path), "RGB")
    depth = _draw_label(load_image_bgr(depth_path), "DEPTH")
    panels = [rgb, depth]
    if str(reference_path or "").strip():
        panels.append(_draw_label(load_image_bgr(reference_path), "TARGET_REF"))

    target_height = max(panel.shape[0] for panel in panels)
    resized_panels: List[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] != target_height:
            scale = target_height / float(panel.shape[0])
            panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        resized_panels.append(panel)
    return cv2.hconcat(resized_panels)


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise VLMSceneDescriptorError("Failed to encode composite image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_scene_descriptor_prompt(
    *,
    task_label: str,
    has_reference: bool,
) -> Dict[str, str]:
    system_prompt = (
        "You are a multimodal UAV scene interpreter for house-entry and person-search tasks.\n"
        "You receive one composite image containing the current RGB view and the aligned depth visualization.\n"
        "Sometimes a third panel contains a target-house reference image.\n"
        "Return one JSON object only.\n"
        "Infer whether the UAV is outside the house, at the threshold, or already inside.\n"
        "Identify whether an entry door is visible and traversable.\n"
        "Generate 2-5 short waypoint hints in normalized image coordinates to guide the UAV.\n"
        "Use doorway evidence from RGB and depth jointly.\n"
        "Do not output markdown."
    )
    user_prompt = (
        f"task_label={task_label or 'search the house for people'}\n"
        f"has_target_house_reference={int(bool(has_reference))}\n"
        "Interpret the scene and return JSON with:\n"
        "- scene_state\n"
        "- active_stage\n"
        "- target_house_visible\n"
        "- entry_door_visible\n"
        "- entry_door_traversable\n"
        "- scene_description\n"
        "- likely_unexplored_regions\n"
        "- next_waypoints\n"
        "- confidence\n"
        "Important:\n"
        "- If the UAV is outside and sees an open front door, prefer active_stage=approach_entry or cross_entry.\n"
        "- If depth shows a clear deep opening behind the door, set entry_door_traversable=true.\n"
        "- next_waypoints should be continuous waypoint hints for the next short segment."
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


@dataclass
class StandaloneSceneDescriptorResult:
    parsed: Dict[str, Any]
    raw_text: str
    usage: Dict[str, Any]
    latency_ms: float
    api_style: str
    model_name: str
    composite_image_path: str
    system_prompt: str
    user_prompt: str


def resolve_api_key(explicit_key: str, api_key_env: str) -> str:
    if str(explicit_key or "").strip():
        return str(explicit_key).strip()
    env_name = str(api_key_env or "").strip()
    if env_name:
        return str(os.environ.get(env_name, "") or "").strip()
    return ""


def describe_scene_with_vlm(
    *,
    api_style: str,
    base_url: str,
    model_name: str,
    api_key: str,
    task_label: str,
    rgb_path: str,
    depth_path: str,
    reference_path: str = "",
    timeout_s: float = 30.0,
    max_output_tokens: int = 900,
) -> StandaloneSceneDescriptorResult:
    composite = build_composite_image(rgb_path=rgb_path, depth_path=depth_path, reference_path=reference_path)
    composite_path = os.path.splitext(rgb_path)[0] + "_vlm_composite.jpg"
    cv2.imwrite(composite_path, composite)
    image_b64 = encode_image_to_base64(composite)
    prompts = build_scene_descriptor_prompt(task_label=task_label, has_reference=bool(reference_path))

    client = LLMPlannerClient(
        LLMPlannerConfig(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            api_style=api_style,
            timeout_s=float(timeout_s),
            include_images=True,
            force_json=True,
            max_output_tokens=int(max_output_tokens),
            temperature=0.1,
        )
    )
    response = client.generate(
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
        image_b64=image_b64,
        json_schema=VLM_SCENE_RESPONSE_SCHEMA,
    )
    parsed = normalize_descriptor_payload(parse_vlm_scene_json_response(str(response.get("text", "") or "")))
    return StandaloneSceneDescriptorResult(
        parsed=parsed,
        raw_text=str(response.get("text", "") or ""),
        usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
        latency_ms=float(response.get("latency_ms", 0.0) or 0.0),
        api_style=str(response.get("api_style", api_style) or api_style),
        model_name=str(response.get("model_name", model_name) or model_name),
        composite_image_path=composite_path,
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
    )


def save_prompt_log(
    *,
    prompt_log_dir: str,
    api_style: str,
    model_name: str,
    task_label: str,
    rgb_path: str,
    depth_path: str,
    reference_path: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
) -> str:
    os.makedirs(prompt_log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = str(model_name or "unknown").replace("/", "_").replace(":", "_")
    output_path = os.path.join(prompt_log_dir, f"{timestamp}_{api_style}_{safe_model}_prompt.json")
    payload = {
        "timestamp": timestamp,
        "api_style": api_style,
        "model_name": model_name,
        "task_label": task_label,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "reference_path": reference_path,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "json_schema": json_schema,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone RGB+depth VLM scene descriptor")
    parser.add_argument("--api_style", default="google_genai_sdk")
    parser.add_argument("--base_url", default="google-genai-sdk")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--task_label", default="search the house for people")
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--reference_path", default="")
    parser.add_argument("--timeout_s", type=float, default=30.0)
    parser.add_argument("--max_output_tokens", type=int, default=900)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--prompt_log_dir", default=get_default_prompt_log_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key, args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"API key is empty. Provide --api_key or set environment variable {args.api_key_env or 'GEMINI_API_KEY'}."
        )

    result = describe_scene_with_vlm(
        api_style=args.api_style,
        base_url=args.base_url,
        model_name=args.model,
        api_key=api_key,
        task_label=args.task_label,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        reference_path=args.reference_path,
        timeout_s=args.timeout_s,
        max_output_tokens=args.max_output_tokens,
    )
    prompt_log_path = save_prompt_log(
        prompt_log_dir=str(args.prompt_log_dir or get_default_prompt_log_dir()),
        api_style=result.api_style,
        model_name=result.model_name,
        task_label=args.task_label,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        reference_path=args.reference_path,
        system_prompt=result.system_prompt,
        user_prompt=result.user_prompt,
        json_schema=VLM_SCENE_RESPONSE_SCHEMA,
    )

    output_payload = {
        "api_style": result.api_style,
        "model_name": result.model_name,
        "latency_ms": result.latency_ms,
        "usage": result.usage,
        "composite_image_path": result.composite_image_path,
        "prompt_log_path": prompt_log_path,
        "parsed": result.parsed,
        "raw_text_preview": _truncate(result.raw_text, 600),
    }

    print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    if str(args.output_json or "").strip():
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
