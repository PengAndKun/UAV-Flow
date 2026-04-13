from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "UAV-Flow-Eval"
if str(EVAL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(EVAL_DIR))

from vlm_scene_descriptor import (  # type: ignore
    VLM_SCENE_RESPONSE_SCHEMA,
    get_default_prompt_log_dir,
    save_prompt_log,
)


FUSION_ENTERABLE_STATES = {"enterable_open_door", "enterable_door"}
FUSION_BLOCKED_STATES = {"visible_but_blocked_entry", "front_blocked_detour"}
FUSION_SEARCH_STATES = {"window_visible_keep_search", "geometric_opening_needs_confirmation", "no_entry_confirmed"}


def _safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return value.strip("._-") or "result"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_labeling_summary(path: Path) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not path.exists():
        return parsed
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()

    top_yolo = str(parsed.get("top_yolo", "")).strip()
    match = re.match(r"^(.*?)\s+\(([\d.]+)\)$", top_yolo)
    if match:
        parsed["top_yolo_class"] = match.group(1).strip().lower()
        try:
            parsed["top_yolo_confidence"] = float(match.group(2))
        except ValueError:
            pass

    depth = str(parsed.get("depth", "")).strip()
    depth_match = re.search(
        r"front_obstacle=(\d+)\s+min=([A-Za-z0-9.\-]+)cm\s+severity=([A-Za-z_]+)",
        depth,
    )
    if depth_match:
        parsed["front_obstacle_present"] = depth_match.group(1) == "1"
        try:
            parsed["front_min_depth_cm"] = float(depth_match.group(2))
        except ValueError:
            parsed["front_min_depth_cm"] = None
        parsed["front_obstacle_severity"] = depth_match.group(3)

    best_entry = str(parsed.get("best_entry", "")).strip()
    best_entry_match = re.search(
        r"dist=([A-Za-z0-9.\-]+)cm\s+width=([A-Za-z0-9.\-]+)cm\s+trav=(\d+)",
        best_entry,
    )
    if best_entry_match:
        for src_key, dst_key in [
            (best_entry_match.group(1), "best_entry_distance_cm"),
            (best_entry_match.group(2), "best_entry_width_cm"),
        ]:
            try:
                parsed[dst_key] = float(src_key)
            except ValueError:
                parsed[dst_key] = None
        parsed["best_entry_traversable"] = best_entry_match.group(3) == "1"

    parsed["fusion_state"] = str(parsed.get("fusion_state", "")).strip()
    parsed["fusion_reason"] = str(parsed.get("fusion_reason", "")).strip()
    parsed["task_label"] = str(parsed.get("task_label", "")).strip()
    parsed["sample_id"] = str(parsed.get("sample_id", path.parent.parent.name)).strip()
    return parsed


def _infer_expected_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    fusion_state = str(summary.get("fusion_state", "")).strip()
    top_yolo_class = str(summary.get("top_yolo_class", "")).strip()
    expected_entry_visible = fusion_state in (FUSION_ENTERABLE_STATES | FUSION_BLOCKED_STATES)
    if fusion_state == "window_visible_keep_search" or top_yolo_class == "window":
        expected_entry_visible = False

    expected_entry_traversable = fusion_state in FUSION_ENTERABLE_STATES
    expected_stage_group = "search"
    if fusion_state in FUSION_ENTERABLE_STATES:
        expected_stage_group = "entry"
    elif fusion_state in FUSION_BLOCKED_STATES:
        expected_stage_group = "blocked"
    elif fusion_state in FUSION_SEARCH_STATES:
        expected_stage_group = "search"

    return {
        "fusion_state": fusion_state,
        "top_yolo_class": top_yolo_class,
        "expected_entry_visible": expected_entry_visible,
        "expected_entry_traversable": expected_entry_traversable,
        "expected_stage_group": expected_stage_group,
    }


def _stage_group(active_stage: str) -> str:
    stage = str(active_stage or "").strip().lower()
    if stage in {"approach_entry", "cross_entry"}:
        return "entry"
    if stage in {"entry_search", "outside_localization", "target_house_verification"}:
        return "search"
    if stage in {"inside_house", "indoor_room_search", "mission_report"}:
        return "other"
    return "other"


def _compare_summary_and_llm(summary: Dict[str, Any], llm_payload: Dict[str, Any]) -> Dict[str, Any]:
    parsed = llm_payload.get("parsed", {}) if isinstance(llm_payload.get("parsed"), dict) else {}
    expected = _infer_expected_from_summary(summary)
    actual_stage_group = _stage_group(str(parsed.get("active_stage", "")))

    entry_visible_match = bool(parsed.get("entry_door_visible", False)) == bool(expected["expected_entry_visible"])
    entry_traversable_match = bool(parsed.get("entry_door_traversable", False)) == bool(
        expected["expected_entry_traversable"]
    )

    stage_match = False
    if expected["expected_stage_group"] == "entry":
        stage_match = actual_stage_group == "entry"
    elif expected["expected_stage_group"] == "search":
        stage_match = actual_stage_group == "search"
    elif expected["expected_stage_group"] == "blocked":
        stage_match = actual_stage_group != "entry"

    issues: List[str] = []
    if not entry_visible_match:
        issues.append("entry_visible_mismatch")
    if not entry_traversable_match:
        issues.append("entry_traversable_mismatch")
    if not stage_match:
        issues.append("stage_group_mismatch")

    overall = "consistent"
    if issues and len(issues) == 1:
        overall = "partially_consistent"
    elif len(issues) >= 2:
        overall = "conflict"

    return {
        "expected": expected,
        "actual": {
            "scene_state": parsed.get("scene_state"),
            "active_stage": parsed.get("active_stage"),
            "entry_door_visible": parsed.get("entry_door_visible"),
            "entry_door_traversable": parsed.get("entry_door_traversable"),
            "confidence": parsed.get("confidence"),
        },
        "agreement": {
            "entry_visible_match": entry_visible_match,
            "entry_traversable_match": entry_traversable_match,
            "stage_group_match": stage_match,
            "overall": overall,
            "issues": issues,
        },
    }


def _discover_labeling_dirs(results_root: Path) -> List[Path]:
    output: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        labeling_dir = child / "labeling"
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def _resolve_task_label(summary: Dict[str, Any], state_excerpt: Optional[Dict[str, Any]]) -> str:
    task_label = str(summary.get("task_label", "")).strip()
    if task_label:
        return task_label
    if isinstance(state_excerpt, dict):
        task_label = str(state_excerpt.get("task_label", "")).strip()
        if task_label:
            return task_label
    return "search the house for people"


def _find_existing_outputs(labeling_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    result_candidates = sorted(
        [
            path
            for path in labeling_dir.glob("anthropic*_scene_result.json")
            if "_vs_labeling_compare" not in path.name
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    compare_candidates = sorted(
        list(labeling_dir.glob("anthropic*_vs_labeling_compare.json")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    result_path = result_candidates[0] if result_candidates else None
    compare_path = compare_candidates[0] if compare_candidates else None
    return result_path, compare_path


def _call_anthropic_with_retries(
    *,
    base_url: str,
    auth_token: str,
    model_name: str,
    task_label: str,
    rgb_path: str,
    depth_path: str,
    reference_path: str,
    timeout_s: float,
    max_output_tokens: int,
    max_attempts: int,
    retry_delay_s: float,
    sample_name: str,
) -> Any:
    from anthropic_vlm_scene_descriptor import describe_scene_with_anthropic  # type: ignore

    last_error: Optional[Exception] = None
    attempt_count = max(1, int(max_attempts))
    for attempt in range(1, attempt_count + 1):
        try:
            if attempt > 1:
                print(f"    [retry {attempt}/{attempt_count}] -> {sample_name}")
            return describe_scene_with_anthropic(
                base_url=base_url,
                auth_token=auth_token,
                model_name=model_name,
                task_label=task_label,
                rgb_path=rgb_path,
                depth_path=depth_path,
                reference_path=reference_path,
                timeout_s=timeout_s,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= attempt_count:
                break
            print(f"    [retry-wait] {sample_name}: {exc}")
            time.sleep(max(0.0, float(retry_delay_s)))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Anthropic request failed without an explicit error for sample {sample_name}")


def process_labeling_dir(
    *,
    labeling_dir: Path,
    base_url: str,
    auth_token: str,
    model_name: str,
    timeout_s: float,
    max_output_tokens: int,
    reference_path: str,
    prompt_log_dir: str,
    force: bool,
    retry_attempts: int,
    retry_delay_s: float,
) -> Tuple[Path, Path, str]:
    rgb_path = labeling_dir / "rgb.png"
    depth_preview_path = labeling_dir / "depth_preview.png"
    summary_path = labeling_dir / "labeling_summary.txt"
    state_excerpt_path = labeling_dir / "state_excerpt.json"

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing rgb.png in {labeling_dir}")
    if not depth_preview_path.exists():
        raise FileNotFoundError(f"Missing depth_preview.png in {labeling_dir}")

    summary = _parse_labeling_summary(summary_path)
    state_excerpt = _read_json(state_excerpt_path) if state_excerpt_path.exists() else None
    task_label = _resolve_task_label(summary, state_excerpt)

    result_name = f"anthropic_{_safe_slug(model_name)}_scene_result.json"
    compare_name = f"anthropic_{_safe_slug(model_name)}_vs_labeling_compare.json"
    result_path = labeling_dir / result_name
    compare_path = labeling_dir / compare_name

    if not force:
        existing_result, existing_compare = _find_existing_outputs(labeling_dir)
        if existing_result and existing_compare:
            return existing_result, existing_compare, "skipped_existing"

    if force or not result_path.exists():
        result = _call_anthropic_with_retries(
            base_url=base_url,
            auth_token=auth_token,
            model_name=model_name,
            task_label=task_label,
            rgb_path=str(rgb_path),
            depth_path=str(depth_preview_path),
            reference_path=reference_path,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
            max_attempts=retry_attempts,
            retry_delay_s=retry_delay_s,
            sample_name=labeling_dir.parent.name,
        )
        prompt_log_path = save_prompt_log(
            prompt_log_dir=str(prompt_log_dir or get_default_prompt_log_dir()),
            api_style="anthropic_sdk",
            model_name=result.model_name,
            task_label=task_label,
            rgb_path=str(rgb_path),
            depth_path=str(depth_preview_path),
            reference_path=reference_path,
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
            json_schema=VLM_SCENE_RESPONSE_SCHEMA,
        )
        payload = {
            "api_style": "anthropic_sdk",
            "base_url": result.base_url,
            "model_name": result.model_name,
            "latency_ms": result.latency_ms,
            "usage": result.usage,
            "composite_image_path": result.composite_image_path,
            "prompt_log_path": prompt_log_path,
            "parsed": result.parsed,
            "raw_text_preview": result.raw_text[:600] + ("..." if len(result.raw_text) > 600 else ""),
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    llm_payload = _read_json(result_path)
    compare_payload = {
        "sample_id": summary.get("sample_id", labeling_dir.parent.name),
        "labeling_dir": str(labeling_dir),
        "task_label": task_label,
        "model_name": llm_payload.get("model_name", model_name),
        "summary_reference": summary,
        "llm_reference": {
            "result_json": str(result_path),
            "prompt_log_path": llm_payload.get("prompt_log_path", ""),
        },
        "comparison": _compare_summary_and_llm(summary, llm_payload),
    }
    compare_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if force:
        mode = "rerun_existing"
    elif result_path.exists() and compare_path.exists():
        mode = "processed"
    else:
        mode = "processed"
    return result_path, compare_path, mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run Pure LLM baseline on Phase2 fusion labeling folders")
    parser.add_argument(
        "--results_root",
        default=str(REPO_ROOT / "phase2_multimodal_fusion_analysis" / "results"),
        help="Directory containing fusion_xxx subdirectories",
    )
    parser.add_argument("--base_url", default="")
    parser.add_argument("--base_url_env", default="ANTHROPIC_BASE_URL")
    parser.add_argument("--auth_token", default="")
    parser.add_argument("--auth_token_env", default="ANTHROPIC_AUTH_TOKEN")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--reference_path", default="")
    parser.add_argument("--timeout_s", type=float, default=30.0)
    parser.add_argument("--max_output_tokens", type=int, default=900)
    parser.add_argument("--prompt_log_dir", default=str(REPO_ROOT / "phase6_prompt_logs"))
    parser.add_argument("--only_dir", default="", help="Optional single fusion directory name, e.g. fusion_20260408_145511")
    parser.add_argument("--force", action="store_true", help="Re-run LLM even if output json already exists")
    parser.add_argument(
        "--rerun_existing",
        action="store_true",
        help="Alias of --force. Revisit labeling folders that already have LLM outputs.",
    )
    parser.add_argument("--retry_attempts", type=int, default=3, help="Retry count when API access fails")
    parser.add_argument("--retry_delay_s", type=float, default=3.0, help="Wait time between API retries")
    return parser.parse_args()


def _resolve_value(explicit_value: str, env_name: str, default_env_name: str) -> str:
    if str(explicit_value or "").strip():
        return str(explicit_value).strip()
    resolved_env = str(env_name or "").strip() or default_env_name
    return str(os.environ.get(resolved_env, "") or "").strip()


def main() -> None:
    args = parse_args()
    base_url = _resolve_value(args.base_url, args.base_url_env, "ANTHROPIC_BASE_URL")
    auth_token = _resolve_value(args.auth_token, args.auth_token_env, "ANTHROPIC_AUTH_TOKEN")
    if not base_url:
        raise SystemExit("Anthropic base URL is empty. Provide --base_url or set ANTHROPIC_BASE_URL.")
    if not auth_token:
        raise SystemExit("Anthropic auth token is empty. Provide --auth_token or set ANTHROPIC_AUTH_TOKEN.")

    results_root = Path(args.results_root).resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results_root does not exist: {results_root}")

    labeling_dirs = _discover_labeling_dirs(results_root)
    if str(args.only_dir or "").strip():
        target = results_root / str(args.only_dir).strip() / "labeling"
        labeling_dirs = [target] if target.is_dir() else []
    if not labeling_dirs:
        raise SystemExit("No labeling directories found.")

    force = bool(args.force or args.rerun_existing)
    summary_rows: List[Dict[str, Any]] = []
    total = len(labeling_dirs)
    print(f"[batch] discovered {total} labeling directories")
    print(f"[batch] mode -> {'rerun_existing' if force else 'skip_existing'}")
    for index, labeling_dir in enumerate(labeling_dirs, start=1):
        sample_name = labeling_dir.parent.name
        print(f"[{index}/{total}] start -> {sample_name}")
        try:
            result_path, compare_path, mode = process_labeling_dir(
                labeling_dir=labeling_dir,
                base_url=base_url,
                auth_token=auth_token,
                model_name=args.model,
                timeout_s=float(args.timeout_s),
                max_output_tokens=int(args.max_output_tokens),
                reference_path=str(args.reference_path or ""),
                prompt_log_dir=str(args.prompt_log_dir or ""),
                force=force,
                retry_attempts=int(args.retry_attempts),
                retry_delay_s=float(args.retry_delay_s),
            )
            compare_payload = _read_json(compare_path)
            summary_rows.append(
                {
                    "sample_id": compare_payload.get("sample_id", labeling_dir.parent.name),
                    "labeling_dir": str(labeling_dir),
                    "result_json": str(result_path),
                    "compare_json": str(compare_path),
                    "mode": mode,
                    "overall": compare_payload.get("comparison", {}).get("agreement", {}).get("overall", "unknown"),
                    "issues": compare_payload.get("comparison", {}).get("agreement", {}).get("issues", []),
                }
            )
            overall = compare_payload.get("comparison", {}).get("agreement", {}).get("overall", "unknown")
            if mode == "skipped_existing":
                print(f"[{index}/{total}] skip -> {sample_name} (existing_result)")
            else:
                print(f"[{index}/{total}] done -> {sample_name} ({overall})")
        except Exception as exc:
            summary_rows.append(
                {
                    "sample_id": labeling_dir.parent.name,
                    "labeling_dir": str(labeling_dir),
                    "mode": "error",
                    "overall": "error",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{total}] error -> {sample_name}: {exc}")

    batch_summary_path = results_root / f"batch_pure_llm_summary_{_safe_slug(args.model)}.json"
    batch_summary_path.write_text(json.dumps({"model_name": args.model, "rows": summary_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_count = sum(1 for row in summary_rows if row.get("overall") != "error")
    err_count = sum(1 for row in summary_rows if row.get("overall") == "error")
    skipped_count = sum(1 for row in summary_rows if row.get("mode") == "skipped_existing")
    print(f"[batch] finished: ok={ok_count} error={err_count}")
    print(
        json.dumps(
            {
                "status": "ok",
                "summary_json": str(batch_summary_path),
                "count": len(summary_rows),
                "skipped_existing": skipped_count,
                "rerun_existing": force,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
