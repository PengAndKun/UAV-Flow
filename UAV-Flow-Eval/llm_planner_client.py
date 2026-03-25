"""
HTTP client for LLM-backed planner calls.

This module intentionally keeps dependencies minimal:
- uses urllib from the standard library
- supports OpenAI-compatible chat-completions style endpoints
- supports OpenAI responses-style endpoints
- supports Anthropic Messages API style endpoints
- supports the official Anthropic Python SDK
- supports Google Gemini generateContent-style endpoints
- supports the official Google GenAI Python SDK
- keeps request/response parsing tolerant across vendors
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, parse, request


class LLMPlannerClientError(RuntimeError):
    """Raised when the LLM planner client cannot complete a request."""


@dataclass(frozen=True)
class LLMPlannerConfig:
    """Runtime configuration for the LLM planner client."""

    base_url: str
    model_name: str
    api_key: str = ""
    api_style: str = "openai_chat"
    endpoint_path: str = ""
    timeout_s: float = 30.0
    max_retries: int = 1
    temperature: float = 0.1
    max_output_tokens: int = 800
    include_images: bool = True
    force_json: bool = True
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    anthropic_version: str = "2023-06-01"
    extra_headers: Dict[str, str] = field(default_factory=dict)


def _build_data_uri(image_b64: str, mime_type: str = "image/jpeg") -> str:
    return f"data:{mime_type};base64,{image_b64}"


def _extract_anthropic_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")) == "text" and isinstance(item.get("text"), str):
            parts.append(str(item.get("text")))
    return "\n".join(part for part in parts if part)


def _extract_anthropic_sdk_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for item in content:
        text_value = getattr(item, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            parts.append(text_value)
    return "\n".join(part for part in parts if part)


def _extract_google_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    parts: List[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        candidate_parts = content.get("parts")
        if not isinstance(candidate_parts, list):
            continue
        for item in candidate_parts:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item.get("text")))
    return "\n".join(part for part in parts if part)


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                elif isinstance(item.get("output_text"), str):
                    parts.append(str(item.get("output_text")))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "output_text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def extract_text_from_response(payload: Dict[str, Any]) -> str:
    """Try several common OpenAI-compatible, Anthropic, and Gemini response shapes."""
    anthropic_text = _extract_anthropic_text(payload.get("content"))
    if anthropic_text:
        return anthropic_text

    google_text = _extract_google_text(payload)
    if google_text:
        return google_text

    if isinstance(payload.get("output_text"), str):
        return str(payload.get("output_text"))

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
        text = _extract_message_text(message.get("content"))
        if text:
            return text
        if isinstance(first_choice.get("text"), str):
            return str(first_choice.get("text"))

    output = payload.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("content"), list):
                parts.append(_extract_message_text(item.get("content")))
            elif isinstance(item.get("text"), str):
                parts.append(str(item.get("text")))
        text = "\n".join(part for part in parts if part)
        if text:
            return text

    if isinstance(payload.get("content"), list):
        text = _extract_message_text(payload.get("content"))
        if text:
            return text

    if isinstance(payload.get("text"), str):
        return str(payload.get("text"))

    raise LLMPlannerClientError("Unable to extract text from LLM response payload.")


def extract_usage_from_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return usage
    if isinstance(payload.get("usageMetadata"), dict):
        return payload.get("usageMetadata") or {}
    if isinstance(payload.get("response_metadata"), dict):
        metadata = payload.get("response_metadata") or {}
        if isinstance(metadata.get("usage"), dict):
            return metadata.get("usage") or {}
    return {}


def _coerce_object_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        result: Dict[str, Any] = {}
        for key, item in value.__dict__.items():
            if str(key).startswith("_"):
                continue
            if isinstance(item, (str, int, float, bool, list, dict)) or item is None:
                result[str(key)] = item
            else:
                nested = _coerce_object_to_dict(item)
                result[str(key)] = nested if nested else str(item)
        return result
    return {}


def _extract_anthropic_sdk_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", None)
    cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", None)
    result: Dict[str, Any] = {}
    total_tokens = 0
    for key, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cache_creation_input_tokens", cache_creation_input_tokens),
        ("cache_read_input_tokens", cache_read_input_tokens),
    ):
        if isinstance(value, int):
            result[key] = value
            total_tokens += value
    if total_tokens:
        result["total_token_count"] = total_tokens
    return result


class LLMPlannerClient:
    """Thin HTTP client that talks to multiple planner API styles."""

    def __init__(self, config: LLMPlannerConfig) -> None:
        self.config = config
        if self.config.api_style not in {"google_genai_sdk"} and not str(config.base_url or "").strip():
            raise LLMPlannerClientError("LLM planner base_url is required.")
        if not str(config.model_name or "").strip():
            raise LLMPlannerClientError("LLM planner model_name is required.")

    def resolve_endpoint_path(self) -> str:
        if str(self.config.endpoint_path or "").strip():
            return str(self.config.endpoint_path).strip()
        if self.config.api_style == "openai_responses":
            return "/v1/responses"
        if self.config.api_style == "anthropic_messages":
            return "/v1/messages"
        if self.config.api_style == "anthropic_sdk":
            return ""
        if self.config.api_style == "google_gemini":
            encoded_model = parse.quote(str(self.config.model_name), safe="")
            return f"/v1beta/models/{encoded_model}:generateContent"
        if self.config.api_style == "google_genai_sdk":
            return ""
        return "/v1/chat/completions"

    def build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        api_key = str(self.config.api_key or "").strip()
        if api_key:
            if self.config.api_style == "anthropic_messages":
                header_name = str(self.config.auth_header or "").strip() or "x-api-key"
                if header_name.lower() == "authorization":
                    header_name = "x-api-key"
                headers[header_name] = api_key
                headers.setdefault("anthropic-version", str(self.config.anthropic_version or "2023-06-01"))
            elif self.config.api_style == "google_gemini":
                header_name = str(self.config.auth_header or "").strip() or "x-goog-api-key"
                if header_name.lower() == "authorization":
                    header_name = "x-goog-api-key"
                headers[header_name] = api_key
            else:
                value = api_key
                if str(self.config.auth_scheme or "").strip():
                    value = f"{self.config.auth_scheme.strip()} {value}"
                headers[str(self.config.auth_header or "Authorization")] = value
        headers.update({str(key): str(value) for key, value in self.config.extra_headers.items()})
        return headers

    def build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_b64: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        include_image = bool(self.config.include_images and image_b64)
        if self.config.api_style == "anthropic_messages":
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            if include_image:
                user_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    }
                )
            return {
                "model": self.config.model_name,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_content}],
                "temperature": float(self.config.temperature),
                "max_tokens": int(self.config.max_output_tokens),
            }

        if self.config.api_style == "google_gemini":
            user_parts: List[Dict[str, Any]] = [{"text": user_prompt}]
            if include_image:
                user_parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    }
                )
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": user_parts,
                    }
                ],
                "generationConfig": {
                    "temperature": float(self.config.temperature),
                    "maxOutputTokens": int(self.config.max_output_tokens),
                },
            }
            if self.config.force_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"
                if isinstance(json_schema, dict) and json_schema:
                    payload["generationConfig"]["responseJsonSchema"] = json_schema
            return payload

        if self.config.api_style == "openai_responses":
            user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
            if include_image:
                user_content.append({"type": "input_image", "image_url": _build_data_uri(image_b64)})
            payload: Dict[str, Any] = {
                "model": self.config.model_name,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                "temperature": float(self.config.temperature),
                "max_output_tokens": int(self.config.max_output_tokens),
            }
            if self.config.force_json:
                payload["text"] = {"format": {"type": "json_object"}}
            return payload

        user_content_chat: Any
        if include_image:
            user_content_chat = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": _build_data_uri(image_b64), "detail": "low"}},
            ]
        else:
            user_content_chat = user_prompt

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_chat},
            ],
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_output_tokens),
        }
        if self.config.force_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_b64: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.config.api_style == "google_genai_sdk":
            return self.generate_with_google_sdk(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_b64=image_b64,
                json_schema=json_schema,
            )
        if self.config.api_style == "anthropic_sdk":
            return self.generate_with_anthropic_sdk(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_b64=image_b64,
                json_schema=json_schema,
            )
        endpoint_path = self.resolve_endpoint_path()
        url = f"{self.config.base_url.rstrip('/')}{endpoint_path}"
        if self.config.api_style == "google_gemini":
            api_key = str(self.config.api_key or "").strip()
            if api_key:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}key={parse.quote(api_key, safe='')}"
        payload = self.build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_b64=image_b64,
            json_schema=json_schema,
        )
        body = json.dumps(payload).encode("utf-8")
        headers = self.build_headers()
        last_error: Optional[Exception] = None

        for attempt in range(max(1, int(self.config.max_retries))):
            request_started = time.time()
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=float(self.config.timeout_s)) as resp:
                    response_payload = json.loads(resp.read().decode("utf-8"))
                text = extract_text_from_response(response_payload)
                usage = extract_usage_from_response(response_payload)
                latency_ms = round((time.time() - request_started) * 1000.0, 2)
                return {
                    "text": text,
                    "raw_response": response_payload,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "model_name": self.config.model_name,
                    "api_style": self.config.api_style,
                    "endpoint_path": endpoint_path,
                    "attempt_count": attempt + 1,
                }
            except error.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                last_error = LLMPlannerClientError(
                    f"HTTP {getattr(exc, 'code', 'error')} from planner API: {error_body or exc.reason}"
                )
            except (error.URLError, TimeoutError, json.JSONDecodeError, LLMPlannerClientError) as exc:
                last_error = exc

        raise LLMPlannerClientError(f"LLM planner request failed after retries: {last_error}")

    def generate_with_google_sdk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_b64: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError as exc:
            raise LLMPlannerClientError(
                "google-genai is not installed. Install it with `pip install google-genai` to use api_style=google_genai_sdk."
            ) from exc

        request_started = time.time()
        api_key = str(self.config.api_key or "").strip()
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        try:
            client = genai.Client(**client_kwargs)
        except Exception as exc:
            raise LLMPlannerClientError(f"Failed to initialize google-genai client: {exc}") from exc

        include_image = bool(self.config.include_images and image_b64)
        contents: Any = user_prompt
        if include_image:
            try:
                image_part = genai_types.Part.from_bytes(
                    data=base64.b64decode(image_b64),
                    mime_type="image/jpeg",
                )
                contents = [user_prompt, image_part]
            except Exception:
                contents = user_prompt

        config_kwargs: Dict[str, Any] = {
            "temperature": float(self.config.temperature),
            "max_output_tokens": int(self.config.max_output_tokens),
        }
        if str(system_prompt or "").strip():
            config_kwargs["system_instruction"] = system_prompt
        if self.config.force_json:
            config_kwargs["response_mime_type"] = "application/json"
            if isinstance(json_schema, dict) and json_schema:
                config_kwargs["response_json_schema"] = json_schema

        config_obj: Any = config_kwargs
        try:
            config_obj = genai_types.GenerateContentConfig(**config_kwargs)
        except Exception:
            config_obj = config_kwargs

        try:
            response = client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=config_obj,
            )
        except Exception as exc:
            raise LLMPlannerClientError(f"google-genai request failed: {exc}") from exc

        text = str(getattr(response, "text", "") or "").strip()
        raw_response = _coerce_object_to_dict(response)
        if not text:
            try:
                text = extract_text_from_response(raw_response)
            except Exception as exc:
                raise LLMPlannerClientError(f"google-genai response did not contain usable text: {exc}") from exc

        usage = _coerce_object_to_dict(getattr(response, "usage_metadata", None))
        latency_ms = round((time.time() - request_started) * 1000.0, 2)
        return {
            "text": text,
            "raw_response": raw_response,
            "usage": usage,
            "latency_ms": latency_ms,
            "model_name": self.config.model_name,
            "api_style": self.config.api_style,
            "endpoint_path": "google.genai.Client.models.generate_content",
            "attempt_count": 1,
        }

    def generate_with_anthropic_sdk(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_b64: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:
            raise LLMPlannerClientError(
                "anthropic is not installed. Install it with `pip install anthropic` to use api_style=anthropic_sdk."
            ) from exc

        request_started = time.time()
        client_kwargs: Dict[str, Any] = {
            "api_key": str(self.config.api_key or "").strip(),
            "timeout": float(self.config.timeout_s),
        }
        base_url = str(self.config.base_url or "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url
        try:
            client = Anthropic(**client_kwargs)
        except Exception as exc:
            raise LLMPlannerClientError(f"Failed to initialize anthropic client: {exc}") from exc

        include_image = bool(self.config.include_images and image_b64)
        content_parts: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": user_prompt
                + (
                    "\nReturn only one JSON object. No markdown. No commentary."
                    if self.config.force_json
                    else ""
                ),
            }
        ]
        if self.config.force_json and isinstance(json_schema, dict) and json_schema:
            content_parts[0]["text"] += "\nJSON schema reference:\n" + json.dumps(json_schema, ensure_ascii=False)
        if include_image:
            content_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                }
            )

        try:
            response = client.messages.create(
                model=self.config.model_name,
                max_tokens=int(self.config.max_output_tokens),
                system=system_prompt,
                messages=[{"role": "user", "content": content_parts}],
            )
        except Exception as exc:
            raise LLMPlannerClientError(f"anthropic SDK request failed: {exc}") from exc

        text = _extract_anthropic_sdk_text(getattr(response, "content", None))
        raw_response = _coerce_object_to_dict(response)
        if not text:
            try:
                text = extract_text_from_response(raw_response)
            except Exception as exc:
                raise LLMPlannerClientError(f"anthropic SDK response did not contain usable text: {exc}") from exc

        usage = _extract_anthropic_sdk_usage(response)
        latency_ms = round((time.time() - request_started) * 1000.0, 2)
        return {
            "text": text,
            "raw_response": raw_response,
            "usage": usage,
            "latency_ms": latency_ms,
            "model_name": self.config.model_name,
            "api_style": self.config.api_style,
            "endpoint_path": "anthropic.Anthropic.messages.create",
            "attempt_count": 1,
        }
