"""
Prompt building and structured response parsing for pure LLM action control.

This adapter asks the LLM for a single bounded macro-action instead of a
high-level waypoint plan. It is intended as an explicit experimental baseline
for "LLM action only" exploration.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from llm_planner_client import LLMPlannerClient
from runtime_interfaces import build_llm_action_runtime_state


class LLMActionAdapterError(RuntimeError):
    """Raised when the LLM action adapter cannot produce a valid action."""


ALLOWED_ACTIONS = {
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
    "yaw_left",
    "yaw_right",
    "hold",
}

ALLOWED_STOP_CONDITIONS = {
    "continue_search",
    "replan_after_step",
    "hold_position",
    "need_manual_review",
    "target_confirmed",
}

LLM_ACTION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "stop_condition": {"type": "string", "enum": sorted(ALLOWED_STOP_CONDITIONS)},
        "should_request_plan": {"type": "boolean"},
    },
    "required": ["action", "confidence", "rationale", "stop_condition", "should_request_plan"],
    "additionalProperties": False,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def _truncate_debug_text(text: str, limit: int) -> str:
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


def _extract_partial_action_fields(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    extracted: Dict[str, Any] = {}

    def capture_string(key: str) -> Optional[str]:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*"([^"\r\n}}]*)"',
            rf'"{re.escape(key)}"\s*:\s*"([^"\r\n]*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return str(match.group(1)).strip()
        return None

    def capture_bool(key: str) -> Optional[bool]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).lower() == "true"
        return None

    def capture_number(key: str) -> Optional[float]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', raw, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    for key in ("action", "rationale", "stop_condition"):
        value = capture_string(key)
        if value:
            extracted[key] = value
    bool_value = capture_bool("should_request_plan")
    if bool_value is not None:
        extracted["should_request_plan"] = bool_value
    confidence = capture_number("confidence")
    if confidence is not None:
        extracted["confidence"] = confidence
    return extracted


def parse_llm_action_json_response(text: str) -> Dict[str, Any]:
    candidate = _strip_fenced_json(text)
    if not candidate:
        raise LLMActionAdapterError("LLM action policy returned empty text.")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        repaired_candidate = _sanitize_json_candidate(candidate)
        if repaired_candidate != candidate:
            try:
                payload = json.loads(repaired_candidate)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        if not payload:
            payload = _extract_partial_action_fields(candidate)
        if not payload:
            raise LLMActionAdapterError(f"LLM action policy returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMActionAdapterError("LLM action policy JSON response must be an object.")
    return payload


def normalize_llm_action_name(action_name: Any) -> str:
    text = str(action_name or "hold").strip().lower()
    if text in ("idle", "hold_position", "hold", ""):
        return "hold"
    if text in ALLOWED_ACTIONS:
        return text
    return "hold"


def build_llm_action_prompt(request_payload: Dict[str, Any]) -> Dict[str, str]:
    task_label = str(request_payload.get("task_label", "")).strip() or "idle"
    mission = request_payload.get("mission") if isinstance(request_payload.get("mission"), dict) else {}
    search_runtime = request_payload.get("search_runtime") if isinstance(request_payload.get("search_runtime"), dict) else {}
    person_evidence_runtime = (
        request_payload.get("person_evidence_runtime")
        if isinstance(request_payload.get("person_evidence_runtime"), dict)
        else {}
    )
    search_result = request_payload.get("search_result") if isinstance(request_payload.get("search_result"), dict) else {}
    language_memory_runtime = (
        request_payload.get("language_memory_runtime")
        if isinstance(request_payload.get("language_memory_runtime"), dict)
        else {}
    )
    current_plan = request_payload.get("current_plan") if isinstance(request_payload.get("current_plan"), dict) else {}
    reflex_runtime = request_payload.get("reflex_runtime") if isinstance(request_payload.get("reflex_runtime"), dict) else {}
    runtime_debug = request_payload.get("runtime_debug") if isinstance(request_payload.get("runtime_debug"), dict) else {}
    pose = request_payload.get("pose") if isinstance(request_payload.get("pose"), dict) else {}
    depth = request_payload.get("depth") if isinstance(request_payload.get("depth"), dict) else {}
    context = request_payload.get("context") if isinstance(request_payload.get("context"), dict) else {}

    pose_summary = {
        "x": round(float(pose.get("x", 0.0)), 1),
        "y": round(float(pose.get("y", 0.0)), 1),
        "z": round(float(pose.get("z", 0.0)), 1),
        "yaw": round(float(pose.get("yaw", 0.0)), 1),
    }
    depth_summary = {
        "min_depth_cm": round(float(depth.get("min_depth", 0.0)), 1),
        "max_depth_cm": round(float(depth.get("max_depth", 0.0)), 1),
    }
    search_summary = {
        "mission_status": str(search_runtime.get("mission_status", "idle")),
        "current_search_subgoal": str(search_runtime.get("current_search_subgoal", "idle")),
        "detection_state": str(search_runtime.get("detection_state", "unknown")),
        "visited_region_count": int(search_runtime.get("visited_region_count", 0)),
        "suspect_region_count": int(search_runtime.get("suspect_region_count", 0)),
        "confirmed_region_count": int(search_runtime.get("confirmed_region_count", 0)),
    }
    evidence_summary = {
        "evidence_status": str(person_evidence_runtime.get("evidence_status", "idle")),
        "suspect_count": int(person_evidence_runtime.get("suspect_count", 0)),
        "confirm_present_count": int(person_evidence_runtime.get("confirm_present_count", 0)),
        "confirm_absent_count": int(person_evidence_runtime.get("confirm_absent_count", 0)),
        "confidence": float(person_evidence_runtime.get("confidence", 0.0)),
        "suspect_region": person_evidence_runtime.get("suspect_region", {})
        if isinstance(person_evidence_runtime.get("suspect_region"), dict)
        else {},
    }
    result_summary = {
        "result_status": str(search_result.get("result_status", "unknown")),
        "person_exists": search_result.get("person_exists"),
        "confidence": float(search_result.get("confidence", 0.0)),
        "summary": str(search_result.get("summary", "")),
    }
    language_memory_summary = {
        "global_summary": str(language_memory_runtime.get("global_summary", "")),
        "current_focus_summary": str(language_memory_runtime.get("current_focus_summary", "")),
        "region_notes": [
            {
                "region_label": str(note.get("region_label", "")),
                "status": str(note.get("status", "")),
                "summary": str(note.get("summary", "")),
            }
            for note in (language_memory_runtime.get("region_notes") or [])[:4]
            if isinstance(note, dict)
        ],
    }
    current_plan_summary = {
        "semantic_subgoal": str(current_plan.get("semantic_subgoal", "")),
        "search_subgoal": str(current_plan.get("search_subgoal", "")),
        "priority_region": current_plan.get("priority_region", {})
        if isinstance(current_plan.get("priority_region"), dict)
        else {},
        "explanation": str(current_plan.get("explanation", "")),
    }
    reflex_summary = {
        "suggested_action": str(reflex_runtime.get("suggested_action", "")),
        "confidence": float(reflex_runtime.get("policy_confidence", 0.0)),
        "source": str(reflex_runtime.get("source", "")),
    }
    execution_summary = {
        "risk_score": float(runtime_debug.get("risk_score", 0.0) or 0.0),
        "current_waypoint": runtime_debug.get("current_waypoint", {})
        if isinstance(runtime_debug.get("current_waypoint"), dict)
        else {},
        "recent_actions": context.get("recent_actions", []) if isinstance(context.get("recent_actions"), list) else [],
        "waypoint_hint": str(context.get("waypoint_hint", "")),
    }

    system_prompt = (
        "You are a UAV indoor search action policy. "
        "You must output one JSON object only. "
        "Do not output markdown, code fences, or explanations outside JSON. "
        "You are controlling exactly one bounded macro-step. "
        "Choose exactly one action from: forward, backward, left, right, up, down, yaw_left, yaw_right, hold. "
        "Prefer safe exploration, target search, and verification behavior. "
        "Avoid oscillation and avoid repeating the opposite of the most recent action unless safety requires it. "
        "If evidence is confirmed_present or confirmed_absent, prefer hold or cautious reorientation. "
        "If risk is high, prefer yaw changes, vertical adjustment, or hold over aggressive translation. "
        "All string values must be single-line plain strings without embedded newlines."
    )
    user_prompt = (
        "Return JSON with keys: action, confidence, rationale, stop_condition, should_request_plan.\n"
        f"task_label={task_label}\n"
        f"mission={_compact_json(mission)}\n"
        f"search_runtime={_compact_json(search_summary)}\n"
        f"person_evidence={_compact_json(evidence_summary)}\n"
        f"search_result={_compact_json(result_summary)}\n"
        f"language_memory={_compact_json(language_memory_summary)}\n"
        f"current_plan={_compact_json(current_plan_summary)}\n"
        f"pose={_compact_json(pose_summary)}\n"
        f"depth={_compact_json(depth_summary)}\n"
        f"reflex_hint={_compact_json(reflex_summary)}\n"
        f"execution_context={_compact_json(execution_summary)}\n"
        "Rules:\n"
        "- confidence must be between 0.0 and 1.0.\n"
        "- stop_condition must be one of continue_search, replan_after_step, hold_position, need_manual_review, target_confirmed.\n"
        "- should_request_plan should be true when the next macro-step should ask the high-level planner again.\n"
        "- keep rationale short and single-line.\n"
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_b64": str(request_payload.get("image_b64", "") or ""),
    }


def build_llm_action(
    *,
    request_payload: Dict[str, Any],
    client: LLMPlannerClient,
    policy_name: str = "external_llm_action",
) -> Dict[str, Any]:
    prompt = build_llm_action_prompt(request_payload)
    response = client.generate(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        image_b64=prompt["image_b64"],
        json_schema=LLM_ACTION_RESPONSE_SCHEMA,
    )
    payload = parse_llm_action_json_response(response["text"])
    action_name = normalize_llm_action_name(payload.get("action", "hold"))
    confidence = _clamp(payload.get("confidence", 0.0), 0.0, 1.0)
    rationale = str(payload.get("rationale", "") or "").replace("\n", " ").strip()
    stop_condition = str(payload.get("stop_condition", "hold_position") or "hold_position").strip()
    if stop_condition not in ALLOWED_STOP_CONDITIONS:
        stop_condition = "hold_position"
    should_request_plan = bool(payload.get("should_request_plan", stop_condition in {"replan_after_step", "target_confirmed"}))
    return build_llm_action_runtime_state(
        action_id=f"llm_action_{request_payload.get('frame_id', 'unknown')}",
        mode="llm_action_only",
        policy_name=policy_name,
        source="llm_action",
        status="ok",
        suggested_action=action_name,
        should_execute=action_name != "hold",
        confidence=confidence,
        rationale=rationale,
        stop_condition=stop_condition,
        should_request_plan=should_request_plan,
        last_trigger=str(request_payload.get("trigger", "manual_request") or "manual_request"),
        last_latency_ms=float(response.get("latency_ms", 0.0) or 0.0),
        risk_score=float((request_payload.get("runtime_debug") or {}).get("risk_score", 0.0) or 0.0),
        model_name=str(response.get("model_name", "") or ""),
        api_style=str(response.get("api_style", "") or ""),
        route_mode="llm_only",
        usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
        attempt_count=int(response.get("attempt_count", 0) or 0),
        fallback_used=False,
        fallback_reason="",
        upstream_error="",
        raw_text=str(response.get("text", "") or ""),
        parsed_payload={
            "action": action_name,
            "confidence": confidence,
            "rationale": rationale,
            "stop_condition": stop_condition,
            "should_request_plan": should_request_plan,
        },
        system_prompt_excerpt=_truncate_debug_text(prompt["system_prompt"], 1800),
        user_prompt_excerpt=_truncate_debug_text(prompt["user_prompt"], 4000),
    )
