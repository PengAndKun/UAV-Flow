from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_aware_llm_teacher_label_validator import (
    extract_json_object,
    validate_labeling_dir,
    write_json,
)


DEFAULT_PROMPT_NAME = "llm_teacher_prompt.json"
DEFAULT_RESPONSE_NAME = "llm_teacher_response.json"
DEFAULT_LABEL_NAME = "llm_teacher_label.json"
DEFAULT_VALIDATED_NAME = "llm_teacher_label_validated.json"


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def resolve_value(explicit_value: str, env_name: str, default_env_name: str) -> str:
    if str(explicit_value or "").strip():
        return str(explicit_value).strip()
    resolved_env = str(env_name or "").strip() or default_env_name
    return str(os.environ.get(resolved_env, "") or "").strip()


def discover_labeling_dirs(session_dir: Path) -> List[Path]:
    output: List[Path] = []
    if session_dir.name == "labeling" and session_dir.is_dir():
        return [session_dir]
    captures_root = session_dir / "memory_fusion_captures"
    if captures_root.is_dir():
        for capture_dir in sorted(captures_root.iterdir(), key=lambda path: path.name):
            labeling_dir = capture_dir / "labeling"
            if labeling_dir.is_dir():
                output.append(labeling_dir)
        return output
    if (session_dir / "labeling").is_dir():
        return [session_dir / "labeling"]
    for labeling_dir in sorted(session_dir.glob("memory_episode*/memory_fusion_captures/*/labeling")):
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def prompt_paths_from_args(args: argparse.Namespace) -> List[Path]:
    if args.prompt_json:
        return [args.prompt_json.resolve()]
    if args.labeling_dir:
        return [(args.labeling_dir / args.prompt_name).resolve()]
    labeling_dirs = discover_labeling_dirs(args.session_dir)
    return [(labeling_dir / args.prompt_name).resolve() for labeling_dir in labeling_dirs]


def extract_anthropic_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


def extract_anthropic_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    output: Dict[str, Any] = {}
    total = 0
    for attr in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            output[attr] = value
            total += value
    if total:
        output["total_token_count"] = total
    return output


def call_anthropic_teacher(
    *,
    base_url: str,
    auth_token: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    timeout_s: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic(api_key=auth_token, base_url=base_url, timeout=float(timeout_s))
    user_text = (
        user_prompt
        + "\n\nReturn only one JSON object. No markdown. No commentary."
        + "\nExpected JSON shape:\n"
        + json.dumps(output_schema, ensure_ascii=False, indent=2)
    )
    start_time = time.time()
    response = client.messages.create(
        model=model_name,
        max_tokens=int(max_output_tokens),
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        ],
    )
    latency_ms = (time.time() - start_time) * 1000.0
    raw_text = extract_anthropic_text(response)
    parsed = extract_json_object(raw_text)
    return {
        "api_style": "anthropic_sdk_text",
        "model_name": model_name,
        "base_url": base_url,
        "latency_ms": round(float(latency_ms), 3),
        "usage": extract_anthropic_usage(response),
        "raw_text": raw_text,
        "parsed": parsed,
    }


def should_skip(
    labeling_dir: Path,
    *,
    response_name: str,
    label_name: str,
    validated_name: str,
    skip_existing: bool,
) -> Tuple[bool, str]:
    if not skip_existing:
        return False, ""
    validated_path = labeling_dir / validated_name
    label_path = labeling_dir / label_name
    response_path = labeling_dir / response_name
    if validated_path.exists():
        return True, "existing_validated"
    if label_path.exists() and response_path.exists():
        return True, "existing_label_and_response"
    return False, ""


def process_prompt(
    prompt_path: Path,
    *,
    base_url: str,
    auth_token: str,
    model_name: str,
    timeout_s: float,
    max_output_tokens: int,
    max_attempts: int,
    retry_delay_s: float,
    response_name: str,
    label_name: str,
    validated_name: str,
    skip_existing: bool,
    validate: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    labeling_dir = prompt_path.parent
    sample_name = labeling_dir.parent.name
    if not prompt_path.exists():
        return {"sample": sample_name, "status": "skip", "reason": "missing_prompt"}
    skip, reason = should_skip(
        labeling_dir,
        response_name=response_name,
        label_name=label_name,
        validated_name=validated_name,
        skip_existing=skip_existing,
    )
    if skip:
        return {"sample": sample_name, "status": "skip", "reason": reason}
    if dry_run:
        return {"sample": sample_name, "status": "dry_run", "reason": "would_call_llm"}

    prompt_payload = read_json(prompt_path)
    system_prompt = str(prompt_payload.get("system_prompt") or "").strip()
    user_prompt = str(prompt_payload.get("user_prompt") or "").strip()
    output_schema = prompt_payload.get("output_schema", {})
    if not system_prompt or not user_prompt:
        return {"sample": sample_name, "status": "error", "reason": "empty_prompt"}
    if not isinstance(output_schema, dict):
        output_schema = {}

    last_error = ""
    result: Dict[str, Any] = {}
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            result = call_anthropic_teacher(
                base_url=base_url,
                auth_token=auth_token,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                timeout_s=float(timeout_s),
                max_output_tokens=int(max_output_tokens),
            )
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < int(max_attempts):
                time.sleep(max(0.0, float(retry_delay_s)))
    if not result:
        return {"sample": sample_name, "status": "error", "reason": f"llm_call_failed:{last_error}"}

    response_path = labeling_dir / response_name
    label_path = labeling_dir / label_name
    parsed = result.get("parsed", {}) if isinstance(result.get("parsed"), dict) else {}
    response_payload = {
        **result,
        "prompt_path": str(prompt_path),
        "labeling_dir": str(labeling_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(response_path, response_payload)
    write_json(label_path, parsed)

    validation_status = ""
    if validate:
        validation = validate_labeling_dir(
            labeling_dir,
            prompt_name=prompt_path.name,
            label_name=label_name,
            response_name=response_name,
            output_name=validated_name,
            write_validated=True,
        )
        validation_status = str(validation.get("status") or "")

    return {
        "sample": sample_name,
        "status": "ok",
        "reason": validation_status or "llm_label_written",
        "response_path": str(response_path),
        "label_path": str(label_path),
        "validated_path": str(labeling_dir / validated_name) if validate else "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch call memory-aware LLM teacher prompts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session_dir", type=Path)
    group.add_argument("--labeling_dir", type=Path)
    group.add_argument("--prompt_json", type=Path)
    parser.add_argument("--prompt_name", default=DEFAULT_PROMPT_NAME)
    parser.add_argument("--response_name", default=DEFAULT_RESPONSE_NAME)
    parser.add_argument("--label_name", default=DEFAULT_LABEL_NAME)
    parser.add_argument("--validated_name", default=DEFAULT_VALIDATED_NAME)
    parser.add_argument("--base_url", default="")
    parser.add_argument("--base_url_env", default="ANTHROPIC_BASE_URL")
    parser.add_argument("--auth_token", default="")
    parser.add_argument("--auth_token_env", default="ANTHROPIC_AUTH_TOKEN")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout_s", type=float, default=45.0)
    parser.add_argument("--max_output_tokens", type=int, default=700)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--retry_delay_s", type=float, default=2.0)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompt_paths = prompt_paths_from_args(args)
    if not prompt_paths:
        print("[llm-teacher-batch] no prompts discovered")
        return 1

    base_url = resolve_value(args.base_url, args.base_url_env, "ANTHROPIC_BASE_URL")
    auth_token = resolve_value(args.auth_token, args.auth_token_env, "ANTHROPIC_AUTH_TOKEN")
    if not args.dry_run:
        if not base_url:
            raise SystemExit(f"Base URL is empty. Provide --base_url or set {args.base_url_env}.")
        if not auth_token:
            raise SystemExit(f"Auth token is empty. Provide --auth_token or set {args.auth_token_env}.")

    print(f"[llm-teacher-batch] prompts={len(prompt_paths)} dry_run={int(bool(args.dry_run))}")
    results: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for idx, prompt_path in enumerate(prompt_paths, start=1):
        result = process_prompt(
            prompt_path,
            base_url=base_url,
            auth_token=auth_token,
            model_name=str(args.model),
            timeout_s=float(args.timeout_s),
            max_output_tokens=int(args.max_output_tokens),
            max_attempts=int(args.max_attempts),
            retry_delay_s=float(args.retry_delay_s),
            response_name=str(args.response_name),
            label_name=str(args.label_name),
            validated_name=str(args.validated_name),
            skip_existing=bool(args.skip_existing),
            validate=bool(args.validate),
            dry_run=bool(args.dry_run),
        )
        results.append(result)
        counts[str(result.get("status") or "unknown")] += 1
        print(f"[{idx}/{len(prompt_paths)}] {result.get('status')} -> {result.get('sample')} ({result.get('reason')})")

    summary = {
        "prompt_count": len(prompt_paths),
        "status_counts": dict(sorted(counts.items(), key=lambda pair: pair[0])),
        "results": results,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(f"[llm-teacher-batch] done status_counts={summary['status_counts']}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
