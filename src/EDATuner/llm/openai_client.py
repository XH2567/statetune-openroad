# src/openai_client.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from src.EDATuner.llm.cot_parser import strip_cot, parse_cot_response, extract_json_from_cot

import requests


# ── Reasoning model (thinking) support ─────────────────────────────────────
# DeepSeek R1 / deepseek-reasoner returns a separate `reasoning_content` field
# in choices[0].message. Some proxy providers instead embed the CoT inside
# <think>...</think> tags within the `content` field.
# We handle BOTH cases transparently.

_RE_THINK_BLOCK = re.compile(
    r"<think>[\s\S]*?</think>",
    re.IGNORECASE,
)

# Model names that are known reasoning/thinking models.
# For these models we skip unsupported params (temperature, response_format, etc.)
_REASONING_MODEL_PATTERNS = re.compile(
    r"(deepseek-reasoner|deepseek-r1|deepseek.*r1|r1-|reasoning|thinking|qwen3.*thinking)",
    re.IGNORECASE,
)

# ── Qwen model detection ──────────────────────────────────────────────────
# Detect Qwen-series models so requests can be routed to the official DashScope API
_QWEN_MODEL_PATTERN = re.compile(
    r"(qwen|qwen2|qwen3|qwen-|qwq)",
    re.IGNORECASE,
)

# Default DashScope API endpoint (OpenAI-compatible format)
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_CHAT_PATH = "/chat/completions"


def _is_reasoning_model(model: str) -> bool:
    """Check if the model name indicates a reasoning/thinking model."""
    return bool(_REASONING_MODEL_PATTERNS.search(str(model or "")))


def _is_qwen_model(model: str) -> bool:
    """Check if the model name indicates a Qwen series model."""
    return bool(_QWEN_MODEL_PATTERN.search(str(model or "")))


def _strip_thinking_tags(s: str) -> str:
    """Remove <think>...</think> blocks that some providers embed in content."""
    return _RE_THINK_BLOCK.sub("", s).strip()


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # First strip any thinking blocks
    s = _strip_thinking_tags(s)
    if s.startswith("```"):
        # remove opening fence line
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        # remove closing fence
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_loose(text: str) -> Any:
    """
    Best-effort JSON parse:
    - strip thinking tags
    - strip markdown fences
    - try direct json.loads
    - try substring from first '{'..last '}' or first '['..last ']'
    """
    s = _strip_code_fences(text)
    if not s:
        raise ValueError("empty response")

    # 1) direct
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) object substring
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        sub = s[l : r + 1].strip()
        try:
            return json.loads(sub)
        except Exception:
            pass

    # 3) array substring
    l = s.find("[")
    r = s.rfind("]")
    if l != -1 and r != -1 and r > l:
        sub = s[l : r + 1].strip()
        try:
            return json.loads(sub)
        except Exception:
            pass

    # 4) regex fallback (first JSON object)
    m = re.search(r"(\{[\s\S]*\})", s)
    if m:
        sub = m.group(1).strip()
        try:
            return json.loads(sub)
        except Exception:
            pass

    raise ValueError("cannot parse json: " + s[:200])


# ── Token usage tracking (lazy import to avoid circular imports) ──────────
_token_tracker = None

def _get_token_tracker():
    """Lazily retrieve the global token tracker singleton."""
    global _token_tracker
    if _token_tracker is None:
        try:
            from src.EDATuner.llm.token_tracker import token_tracker
            _token_tracker = token_tracker
        except ImportError:
            _token_tracker = False  # Mark as unavailable so we do not retry
    return _token_tracker if _token_tracker is not False else None


def _record_token_usage(raw: Dict[str, Any], model: str) -> None:
    """Extract the usage field from an API response and record it in the global tracker."""
    tracker = _get_token_tracker()
    if tracker is None:
        return
    try:
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return
        tracker.record(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            model=model,
        )
    except Exception:
        pass


class ProviderChatCompletionsClient:
    """
    Provider chat/completions client (requests-based).

    Compatible with providers that implement an OpenAI-compatible
    `/v1/chat/completions` endpoint and return:
      raw["choices"][0]["message"]["content"]

    Also handles DeepSeek reasoning models that return:
      raw["choices"][0]["message"]["reasoning_content"]  (CoT / thinking)
      raw["choices"][0]["message"]["content"]              (final answer)

    Multi-provider routing:
      - Qwen models -> official DashScope API (DASHSCOPE_API_KEY)
      - Other models (for example DeepSeek) -> the configured provider API (PROVIDER_API_KEY)
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.apiyi.com",
        path: str = "/v1/chat/completions",
        timeout_sec: int = 60,
    ) -> None:
        # ── Credentials for the configured provider (for example DeepSeek) ──
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.path = path if path.startswith("/") else ("/" + path)
        self.timeout_sec = int(timeout_sec)

        # ── Credentials for official DashScope (Qwen) ──
        self.dashscope_api_key = (
            os.getenv("DASHSCOPE_API_KEY", "")
        ).strip()
        self.dashscope_base_url = os.getenv(
            "DASHSCOPE_BASE_URL", _DASHSCOPE_BASE_URL
        ).rstrip("/")
        self.dashscope_path = _DASHSCOPE_CHAT_PATH

        if self.dashscope_api_key:
            print(f"  [LLM] DashScope (Qwen official) API key loaded ✓")

        # ── Detect official DeepSeek API credentials ──
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if self.deepseek_api_key:
            print(f"  [LLM] DeepSeek official API key loaded ✓")


    def _headers(self, *, for_qwen: bool = False) -> Dict[str, str]:
        """Return request headers for the selected model family."""
        if for_qwen:
            key = self.dashscope_api_key
            if not key:
                raise RuntimeError(
                    "DASHSCOPE_API_KEY is not set. "
                    "Please set the DASHSCOPE_API_KEY environment variable "
                    "to use official Qwen models."
                )
        else:
            key = self.api_key
            if not key:
                raise RuntimeError("api_key is empty.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def _resolve_endpoint(self, model: str) -> Tuple[str, bool]:
        """
        Resolve the API endpoint based on the model name.

        Returns:
            (url, is_qwen) - the full request URL and whether the model is a Qwen model
        """
        if _is_qwen_model(model) and self.dashscope_api_key:
            url = f"{self.dashscope_base_url}{self.dashscope_path}"
            return url, True
        else:
            url = f"{self.base_url}{self.path}"
            return url, False

    def create_text(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        # Route to the appropriate API endpoint based on the model name
        url, is_qwen = self._resolve_endpoint(model)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        # ★ Reasoning models don't support temperature / top_p.
        if not _is_reasoning_model(model):
            payload["temperature"] = float(temperature)

        if extra_payload:
            payload.update(extra_payload)

        # ★ Reasoning/thinking models need more time for CoT generation
        effective_timeout = self.timeout_sec * 2 if _is_reasoning_model(model) else self.timeout_sec

        r = requests.post(
            url,
            headers=self._headers(for_qwen=is_qwen),
            json=payload,
            timeout=effective_timeout,
        )

        if r.status_code >= 400:
            provider_tag = "DashScope" if is_qwen else "Provider"
            raise RuntimeError(
                f"{provider_tag} API error: status={r.status_code} body={r.text[:2000]}"
            )

        raw = r.json()
        text = self._extract_content(raw)
        if not text:
            raise RuntimeError("Empty assistant content.")

        # Token usage tracking
        _record_token_usage(raw, model)

        return text, raw

    def _extract_content(self, raw: Dict[str, Any]) -> str:
        """
        Extract the final answer content, ignoring reasoning/thinking content.

        Handles three scenarios:
        1) Official DeepSeek API: reasoning_content is a separate field,
           content has only the final answer → use content directly.
        2) Some proxies embed <think>...</think> inside content →
           strip those tags to get the clean answer.
        3) Standard OpenAI format: no reasoning_content, no think tags →
           use content as-is.
        """
        choices = raw.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return ""
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = c0.get("message")

        if isinstance(msg, dict) and "content" in msg:
            content = str(msg.get("content") or "").strip()
            # ★ Strip <think>...</think> blocks that some proxies embed
            content = _strip_thinking_tags(content)
            return content

        if "text" in c0:
            text = str(c0.get("text") or "").strip()
            return _strip_thinking_tags(text)

        return ""

    def create_structured_json_best_effort(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 1200,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Best-effort structured output:
        - For non-reasoning models, first try response_format(json_schema)
        - For reasoning models, go directly to the fallback path
          because response_format is not supported
        - If that is not supported, force JSON via the system prompt and
          parse the response with a fallback strategy
        """
        # ★ Reasoning models don't support response_format / json_schema.
        #   Skip the first attempt entirely to save a round-trip.
        if not _is_reasoning_model(model):
            # 1) try response_format
            try_payload: Dict[str, Any] = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "strict": True, "schema": json_schema},
                }
            }
            if extra_payload:
                try_payload.update(extra_payload)

            try:
                text, raw = self.create_text(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_payload=try_payload,
                )
                parsed = _parse_json_loose(text)
                if not isinstance(parsed, dict):
                    raise ValueError("structured output is not an object")
                return parsed, raw
            except Exception:
                pass

        # 2) fallback: force JSON only (and do NOT send provider-unknown fields)
        sys_msg = (
            "You MUST output ONLY valid JSON (no markdown, no extra text). "
            "Start your response with '{' and end with '}'. "
            "Do NOT include any thinking, explanation, or commentary outside the JSON. "
            "The JSON MUST conform to this JSON Schema:\n"
            + json.dumps(json_schema, ensure_ascii=False)
        )
        forced = [{"role": "system", "content": sys_msg}] + messages
        text, raw = self.create_text(
            model=model,
            messages=forced,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_payload=None,
        )
        parsed = _parse_json_loose(text)
        if not isinstance(parsed, dict):
            raise ValueError("structured output is not an object")
        return parsed, raw


class OpenAIResponsesClient:
    """
    Compatibility wrapper used by src/ka_agent.py.

    Internally calls ProviderChatCompletionsClient (/v1/chat/completions).

    Env vars:
      - PROVIDER_API_KEY (preferred) or OPENAI_API_KEY (fallback)  -> provider models such as DeepSeek
      - PROVIDER_BASE_URL (default https://api.apiyi.com)          -> provider models such as DeepSeek
      - DASHSCOPE_API_KEY                                          -> official Qwen access
      - DASHSCOPE_BASE_URL (optional, default https://dashscope.aliyuncs.com/compatible-mode/v1)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
        timeout_sec: int = 60,
    ) -> None:
        key = (api_key or os.getenv("PROVIDER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        b = (base_url or os.getenv("PROVIDER_BASE_URL") or "https://api.apiyi.com").strip()
        p = (path or os.getenv("PROVIDER_CHAT_PATH") or "/v1/chat/completions").strip()
        self._inner = ProviderChatCompletionsClient(api_key=key, base_url=b, path=p, timeout_sec=timeout_sec)

    def is_ready(self) -> bool:
        return bool(getattr(self._inner, "api_key", ""))

    def create_text(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        # max_output_tokens is an alias some code uses
        mt = int(max_tokens if max_tokens is not None else (max_output_tokens if max_output_tokens is not None else 4000))
        # Safety cap: keep max_tokens bounded for reasoning models because reasoning tokens consume budget
        if _is_reasoning_model(model) and mt > 16000:
            mt = 16000
        # NOTE: metadata is ignored by default to avoid provider rejecting unknown fields
        return self._inner.create_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=mt,
            extra_payload=extra_payload,
        )
    
    def create_structured_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mt = int(max_tokens if max_tokens is not None else (max_output_tokens if max_output_tokens is not None else 4000))
        # NOTE: metadata is ignored by default to avoid provider rejecting unknown fields
        return self._inner.create_structured_json_best_effort(
            model=model,
            messages=messages,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=mt,
            extra_payload=extra_payload,
        )
