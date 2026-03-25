"""
Standalone multimodal scene descriptor using the official Anthropic SDK.

This module mirrors `vlm_scene_descriptor.py`, but it talks to Anthropic-
compatible backends through `import anthropic` and supports:

- ANTHROPIC_BASE_URL
- ANTHROPIC_AUTH_TOKEN

It is intended for direct API validation with models such as:
- qwen3-coder-next
- claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

from anthropic import Anthropic

from vlm_scene_descriptor import (
    VLM_SCENE_RESPONSE_SCHEMA,
    VLMSceneDescriptorError,
    _truncate,
    build_composite_image,
    build_scene_descriptor_prompt,
    encode_image_to_base64,
    get_default_prompt_log_dir,
    normalize_descriptor_payload,
    parse_vlm_scene_json_response,
    save_prompt_log,
)


def _resolve_value(explicit_value: str, env_name: str, default_env_name: str) -> str:
    if str(explicit_value or "").strip():
        return str(explicit_value).strip()
    resolved_env = str(env_name or "").strip() or default_env_name
    return str(os.environ.get(resolved_env, "") or "").strip()


def _extract_text_from_anthropic_response(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_usage_from_anthropic_response(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", None)
    cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", None)
    total_tokens = 0
    for value in (
        input_tokens,
        output_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
    ):
        if isinstance(value, int):
            total_tokens += value
    result: Dict[str, Any] = {}
    if isinstance(input_tokens, int):
        result["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        result["output_tokens"] = output_tokens
    if isinstance(cache_creation_input_tokens, int):
        result["cache_creation_input_tokens"] = cache_creation_input_tokens
    if isinstance(cache_read_input_tokens, int):
        result["cache_read_input_tokens"] = cache_read_input_tokens
    if total_tokens:
        result["total_token_count"] = total_tokens
    return result


@dataclass
class AnthropicSceneDescriptorResult:
    parsed: Dict[str, Any]
    raw_text: str
    usage: Dict[str, Any]
    latency_ms: float
    model_name: str
    base_url: str
    composite_image_path: str
    system_prompt: str
    user_prompt: str


def describe_scene_with_anthropic(
    *,
    base_url: str,
    auth_token: str,
    model_name: str,
    task_label: str,
    rgb_path: str,
    depth_path: str,
    reference_path: str = "",
    timeout_s: float = 30.0,
    max_output_tokens: int = 900,
) -> AnthropicSceneDescriptorResult:
    composite = build_composite_image(
        rgb_path=rgb_path,
        depth_path=depth_path,
        reference_path=reference_path,
    )
    composite_path = os.path.splitext(rgb_path)[0] + "_anthropic_vlm_composite.jpg"
    import cv2

    cv2.imwrite(composite_path, composite)
    image_b64 = encode_image_to_base64(composite)
    prompts = build_scene_descriptor_prompt(
        task_label=task_label,
        has_reference=bool(reference_path),
    )

    client = Anthropic(
        api_key=auth_token,
        base_url=base_url,
        timeout=timeout_s,
    )

    start_time = time.time()
    response = client.messages.create(
        model=model_name,
        max_tokens=int(max_output_tokens),
        system=prompts["system_prompt"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            prompts["user_prompt"]
                            + "\nReturn only one JSON object. No markdown. No commentary."
                            + "\nJSON schema reference:\n"
                            + json.dumps(VLM_SCENE_RESPONSE_SCHEMA, ensure_ascii=False)
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    )
    latency_ms = (time.time() - start_time) * 1000.0
    raw_text = _extract_text_from_anthropic_response(response)
    if not raw_text:
        raise VLMSceneDescriptorError("Anthropic scene descriptor returned empty text.")
    parsed = normalize_descriptor_payload(parse_vlm_scene_json_response(raw_text))
    return AnthropicSceneDescriptorResult(
        parsed=parsed,
        raw_text=raw_text,
        usage=_extract_usage_from_anthropic_response(response),
        latency_ms=latency_ms,
        model_name=model_name,
        base_url=base_url,
        composite_image_path=composite_path,
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Anthropic RGB+depth scene descriptor")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--base_url_env", default="ANTHROPIC_BASE_URL")
    parser.add_argument("--auth_token", default="")
    parser.add_argument("--auth_token_env", default="ANTHROPIC_AUTH_TOKEN")
    parser.add_argument("--model", required=True)
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
    base_url = _resolve_value(args.base_url, args.base_url_env, "ANTHROPIC_BASE_URL")
    auth_token = _resolve_value(args.auth_token, args.auth_token_env, "ANTHROPIC_AUTH_TOKEN")
    if not base_url:
        raise SystemExit(
            f"Anthropic base URL is empty. Provide --base_url or set {args.base_url_env or 'ANTHROPIC_BASE_URL'}."
        )
    if not auth_token:
        raise SystemExit(
            "Anthropic auth token is empty. Provide --auth_token or set "
            f"{args.auth_token_env or 'ANTHROPIC_AUTH_TOKEN'}."
        )

    result = describe_scene_with_anthropic(
        base_url=base_url,
        auth_token=auth_token,
        model_name=args.model,
        task_label=args.task_label,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        reference_path=args.reference_path,
        timeout_s=args.timeout_s,
        max_output_tokens=args.max_output_tokens,
    )
    prompt_log_path = save_prompt_log(
        prompt_log_dir=str(args.prompt_log_dir or get_default_prompt_log_dir()),
        api_style="anthropic_sdk",
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
        "api_style": "anthropic_sdk",
        "base_url": result.base_url,
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
