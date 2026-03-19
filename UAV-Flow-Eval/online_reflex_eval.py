"""
Online evaluation recorder for Phase 3 reflex policy experiments.

This tool passively polls the running UAV control server, writes a JSONL trace,
and summarizes the online reflex behavior for a session. It is intended for
Phase 3 closed-loop validation after a trained local policy is connected to the
live control stack.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error, request


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ServerClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def get_json(self, path: str) -> Dict[str, Any]:
        with request.urlopen(f"{self.base_url}{path}", timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def summarize_series(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "initial": 0.0,
            "final": 0.0,
            "delta_final_minus_initial": 0.0,
            "improvement_initial_minus_final": 0.0,
        }
    avg = float(sum(values) / len(values))
    initial = float(values[0])
    final = float(values[-1])
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": avg,
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "initial": initial,
        "final": final,
        "delta_final_minus_initial": float(final - initial),
        "improvement_initial_minus_final": float(initial - final),
    }


def build_trace_record(state: Dict[str, Any], *, wall_time_s: float) -> Dict[str, Any]:
    observation = state.get("observation", {}) if isinstance(state.get("observation"), dict) else {}
    pose = state.get("pose", {}) if isinstance(state.get("pose"), dict) else {}
    depth = state.get("depth", {}) if isinstance(state.get("depth"), dict) else {}
    plan = state.get("plan", {}) if isinstance(state.get("plan"), dict) else {}
    planner_runtime = state.get("planner_runtime", {}) if isinstance(state.get("planner_runtime"), dict) else {}
    archive = state.get("archive", {}) if isinstance(state.get("archive"), dict) else {}
    reflex = state.get("reflex_runtime", {}) if isinstance(state.get("reflex_runtime"), dict) else {}
    runtime_debug = state.get("runtime_debug", {}) if isinstance(state.get("runtime_debug"), dict) else {}
    active_retrieval = archive.get("active_retrieval", {}) if isinstance(archive.get("active_retrieval"), dict) else {}

    return {
        "schema_version": "phase3.online_reflex_trace.v1",
        "captured_at": datetime.now().isoformat(timespec="milliseconds"),
        "wall_time_s": float(wall_time_s),
        "frame_id": str(observation.get("frame_id", "")),
        "observation_timestamp": str(observation.get("timestamp", "")),
        "task_label": str(state.get("task_label", "") or "idle"),
        "last_action": str(state.get("last_action", "") or "idle"),
        "step_index": coerce_int(planner_runtime.get("step_index", 0)),
        "pose": {
            "x": coerce_float(pose.get("x", 0.0)),
            "y": coerce_float(pose.get("y", 0.0)),
            "z": coerce_float(pose.get("z", 0.0)),
            "yaw": coerce_float(pose.get("yaw", 0.0)),
            "command_yaw": coerce_float(pose.get("command_yaw", 0.0)),
            "task_yaw": coerce_float(pose.get("task_yaw", 0.0)),
            "uav_yaw": coerce_float(pose.get("uav_yaw", 0.0)),
        },
        "depth": {
            "min_depth": coerce_float(depth.get("min_depth", 0.0)),
            "max_depth": coerce_float(depth.get("max_depth", 0.0)),
            "front_min_depth": coerce_float(depth.get("front_min_depth", 0.0)),
            "front_mean_depth": coerce_float(depth.get("front_mean_depth", 0.0)),
        },
        "plan": {
            "planner_name": str(plan.get("planner_name", "")),
            "semantic_subgoal": str(plan.get("semantic_subgoal", "")),
            "planner_confidence": coerce_float(plan.get("planner_confidence", 0.0)),
            "sector_id": str(plan.get("sector_id", "")),
        },
        "archive": {
            "current_cell_id": str(archive.get("current_cell_id", "")),
            "visit_count": coerce_int(archive.get("visit_count", 0)),
            "cell_count": coerce_int(archive.get("cell_count", 0)),
            "transition_count": coerce_int(archive.get("transition_count", 0)),
            "active_retrieval_cell_id": str(active_retrieval.get("cell_id", "")),
            "active_retrieval_score": coerce_float(active_retrieval.get("retrieval_score", 0.0)),
            "active_retrieval_subgoal": str(active_retrieval.get("semantic_subgoal", "")),
        },
        "reflex": {
            "mode": str(reflex.get("mode", "")),
            "policy_name": str(reflex.get("policy_name", "")),
            "source": str(reflex.get("source", "")),
            "status": str(reflex.get("status", "")),
            "suggested_action": str(reflex.get("suggested_action", "idle")),
            "should_execute": bool(reflex.get("should_execute", False)),
            "last_trigger": str(reflex.get("last_trigger", "")),
            "last_latency_ms": coerce_float(reflex.get("last_latency_ms", 0.0)),
            "policy_confidence": coerce_float(reflex.get("policy_confidence", 0.0)),
            "waypoint_distance_cm": coerce_float(reflex.get("waypoint_distance_cm", 0.0)),
            "yaw_error_deg": coerce_float(reflex.get("yaw_error_deg", 0.0)),
            "vertical_error_cm": coerce_float(reflex.get("vertical_error_cm", 0.0)),
            "progress_to_waypoint_cm": coerce_float(reflex.get("progress_to_waypoint_cm", 0.0)),
            "retrieval_cell_id": str(reflex.get("retrieval_cell_id", "")),
            "retrieval_score": coerce_float(reflex.get("retrieval_score", 0.0)),
            "risk_score": coerce_float(reflex.get("risk_score", 0.0)),
            "shield_triggered": bool(reflex.get("shield_triggered", False)),
        },
        "runtime_debug": {
            "risk_score": coerce_float(runtime_debug.get("risk_score", 0.0)),
            "archive_cell_id": str(runtime_debug.get("archive_cell_id", "")),
            "current_waypoint": runtime_debug.get("current_waypoint", {}),
        },
    }


def record_signature(record: Dict[str, Any]) -> Tuple[Any, ...]:
    pose = record.get("pose", {}) if isinstance(record.get("pose"), dict) else {}
    reflex = record.get("reflex", {}) if isinstance(record.get("reflex"), dict) else {}
    return (
        record.get("frame_id", ""),
        record.get("step_index", 0),
        record.get("last_action", ""),
        reflex.get("suggested_action", ""),
        round(coerce_float(pose.get("x", 0.0)), 1),
        round(coerce_float(pose.get("y", 0.0)), 1),
        round(coerce_float(pose.get("z", 0.0)), 1),
        round(coerce_float(pose.get("yaw", 0.0)), 1),
    )


def build_summary(
    records: List[Dict[str, Any]],
    *,
    session_id: str,
    server_url: str,
    poll_interval_s: float,
    request_reflex_on_poll: bool,
    error_count: int,
) -> Dict[str, Any]:
    task_counts: Counter[str] = Counter()
    executed_action_counts: Counter[str] = Counter()
    suggested_action_counts: Counter[str] = Counter()
    planner_subgoal_counts: Counter[str] = Counter()
    policy_mode_counts: Counter[str] = Counter()
    policy_name_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    switch_by_subgoal: Counter[str] = Counter()

    confidence_values: List[float] = []
    waypoint_values: List[float] = []
    yaw_error_abs_values: List[float] = []
    progress_values: List[float] = []
    latency_values: List[float] = []
    risk_values: List[float] = []

    retrieval_hits = 0
    switch_count = 0
    repeated_action_count = 0

    previous_action: Optional[str] = None
    for record in records:
        task_label = str(record.get("task_label", "") or "idle")
        last_action = str(record.get("last_action", "") or "idle")
        plan = record.get("plan", {}) if isinstance(record.get("plan"), dict) else {}
        reflex = record.get("reflex", {}) if isinstance(record.get("reflex"), dict) else {}

        suggested_action = str(reflex.get("suggested_action", "idle"))
        planner_subgoal = str(plan.get("semantic_subgoal", "") or "idle")
        policy_mode = str(reflex.get("mode", ""))
        policy_name = str(reflex.get("policy_name", ""))
        source = str(reflex.get("source", ""))

        task_counts[task_label] += 1
        executed_action_counts[last_action] += 1
        suggested_action_counts[suggested_action] += 1
        planner_subgoal_counts[planner_subgoal] += 1
        policy_mode_counts[policy_mode] += 1
        policy_name_counts[policy_name] += 1
        source_counts[source] += 1

        confidence_values.append(coerce_float(reflex.get("policy_confidence", 0.0)))
        waypoint_values.append(coerce_float(reflex.get("waypoint_distance_cm", 0.0)))
        yaw_error_abs_values.append(abs(coerce_float(reflex.get("yaw_error_deg", 0.0))))
        progress_values.append(coerce_float(reflex.get("progress_to_waypoint_cm", 0.0)))
        latency_values.append(coerce_float(reflex.get("last_latency_ms", 0.0)))
        risk_values.append(coerce_float(reflex.get("risk_score", 0.0)))

        if str(reflex.get("retrieval_cell_id", "")):
            retrieval_hits += 1

        if previous_action is not None:
            if suggested_action != previous_action:
                switch_count += 1
                switch_by_subgoal[planner_subgoal] += 1
            else:
                repeated_action_count += 1
        previous_action = suggested_action

    first_time = coerce_float(records[0].get("wall_time_s", 0.0)) if records else 0.0
    last_time = coerce_float(records[-1].get("wall_time_s", 0.0)) if records else 0.0
    sample_count = len(records)
    transition_count = max(0, sample_count - 1)

    return {
        "schema_version": "phase3.online_reflex_eval.v1",
        "session_id": session_id,
        "server_url": server_url,
        "poll_interval_s": float(poll_interval_s),
        "request_reflex_on_poll": bool(request_reflex_on_poll),
        "sample_count": sample_count,
        "duration_s": float(max(0.0, last_time - first_time)),
        "error_count": int(error_count),
        "task_counts": dict(task_counts),
        "executed_action_counts": dict(executed_action_counts),
        "suggested_action_counts": dict(suggested_action_counts),
        "planner_subgoal_counts": dict(planner_subgoal_counts),
        "policy_mode_counts": dict(policy_mode_counts),
        "policy_name_counts": dict(policy_name_counts),
        "source_counts": dict(source_counts),
        "action_switch_count": int(switch_count),
        "action_repeat_count": int(repeated_action_count),
        "action_switch_rate_per_transition": float(switch_count / transition_count) if transition_count else 0.0,
        "switch_by_subgoal": dict(switch_by_subgoal),
        "retrieval_hit_rate": float(retrieval_hits / sample_count) if sample_count else 0.0,
        "zero_confidence_fraction": (
            float(sum(1 for value in confidence_values if value <= 1e-6) / len(confidence_values))
            if confidence_values
            else 0.0
        ),
        "confidence_stats": summarize_series(confidence_values),
        "waypoint_distance_stats_cm": summarize_series(waypoint_values),
        "yaw_error_abs_stats_deg": summarize_series(yaw_error_abs_values),
        "progress_stats_cm": summarize_series(progress_values),
        "latency_stats_ms": summarize_series(latency_values),
        "risk_stats": summarize_series(risk_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record and summarize online Phase 3 reflex behavior from /state")
    parser.add_argument("--server_url", default="http://127.0.0.1:5020", help="Base URL for the running UAV control server")
    parser.add_argument("--output_dir", default="./phase3_online_eval", help="Directory for summary JSON and trace JSONL")
    parser.add_argument("--session_name", default="", help="Optional custom session name prefix")
    parser.add_argument("--duration_s", type=float, default=45.0, help="How long to record before stopping")
    parser.add_argument("--poll_interval_s", type=float, default=0.5, help="Polling interval for GET /state")
    parser.add_argument("--timeout_s", type=float, default=2.5, help="HTTP timeout per request")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional sample cap; 0 means unlimited until duration")
    parser.add_argument("--request_reflex_on_poll", action="store_true", help="Also POST /request_reflex on every poll before reading /state")
    parser.add_argument("--only_on_change", action="store_true", help="Only keep a sample when state signature changes")
    parser.add_argument("--max_errors", type=int, default=8, help="Abort if repeated server errors exceed this count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"{args.session_name}_{now_timestamp()}" if args.session_name else f"online_reflex_eval_{now_timestamp()}"
    trace_path = output_dir / f"{session_id}_trace.jsonl"
    summary_path = output_dir / f"{session_id}_summary.json"

    client = ServerClient(args.server_url, timeout_s=float(args.timeout_s))
    records: List[Dict[str, Any]] = []
    seen_signature: Optional[Tuple[Any, ...]] = None
    error_count = 0
    started = time.time()

    try:
        with trace_path.open("w", encoding="utf-8") as trace_file:
            while True:
                now_s = time.time()
                elapsed = float(now_s - started)
                if elapsed >= float(args.duration_s):
                    break
                if int(args.max_samples) > 0 and len(records) >= int(args.max_samples):
                    break

                try:
                    if args.request_reflex_on_poll:
                        client.post_json("/request_reflex", {"trigger": "online_eval"})
                    state = client.get_json("/state")
                    record = build_trace_record(state, wall_time_s=elapsed)
                    signature = record_signature(record)
                    if args.only_on_change and signature == seen_signature:
                        time.sleep(float(args.poll_interval_s))
                        continue
                    seen_signature = signature
                    records.append(record)
                    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    trace_file.flush()
                    error_count = 0
                except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    error_count += 1
                    if error_count >= int(args.max_errors):
                        raise RuntimeError(f"Exceeded max_errors while polling server: {exc}") from exc
                time.sleep(float(args.poll_interval_s))
    except KeyboardInterrupt:
        pass

    summary = build_summary(
        records,
        session_id=session_id,
        server_url=args.server_url,
        poll_interval_s=float(args.poll_interval_s),
        request_reflex_on_poll=bool(args.request_reflex_on_poll),
        error_count=error_count,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Online Reflex Evaluation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {summary_path}")
    print(f"Saved trace to: {trace_path}")


if __name__ == "__main__":
    main()
