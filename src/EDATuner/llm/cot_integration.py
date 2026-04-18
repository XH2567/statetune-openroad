#!/usr/bin/env python3
"""
src/EDATuner/llm/cot_integration.py
=====================================
Integration layer between the CoT parser and the existing OpenAI client.

This module provides drop-in wrappers that make the existing LLM client
transparently handle CoT/reasoning models (DeepSeek-R1, DeepSeek-V3.2, etc.)
without modifying the rest of the codebase.

Integration points:
  1. openai_client.py  → wrap call_chat / call_completion to auto-strip thinking
  2. ka_agent.py       → wrap candidate extraction to handle CoT output
  3. llm_queue_client  → wrap queued responses

Usage in openai_client.py:
--------------------------
Add these lines at the top of your openai_client.py:

    from src.EDATuner.llm.cot_parser import strip_cot, extract_json_from_cot, CoTStreamBuffer

Then wrap your response extraction. For example, if you have:

    # BEFORE (old code):
    content = response.choices[0].message.content
    return content

Change to:

    # AFTER (CoT-aware):
    raw_content = response.choices[0].message.content or ""
    # Check for separate reasoning_content field (some APIs)
    if hasattr(response.choices[0].message, 'reasoning_content'):
        reasoning = response.choices[0].message.reasoning_content
        if reasoning:
            logger.debug(f"CoT reasoning ({len(reasoning)} chars): {reasoning[:200]}...")
    content = strip_cot(raw_content)
    return content

Usage in ka_agent.py:
---------------------
For extracting structured candidates from CoT responses:

    from src.EDATuner.llm.cot_parser import extract_candidates_from_cot

    # BEFORE:
    raw = self.llm.call(prompt)
    candidates = json.loads(raw)

    # AFTER:
    raw = self.llm.call(prompt)
    candidates = extract_candidates_from_cot(raw)

Usage in streaming:
-------------------
    from src.EDATuner.llm.cot_parser import CoTStreamBuffer

    buf = CoTStreamBuffer()
    for chunk in client.chat.completions.create(stream=True, ...):
        delta = chunk.choices[0].delta.content or ""
        clean = buf.feed(delta)
        if clean:
            yield clean
    final = buf.flush()
    if final:
        yield final
"""
from __future__ import annotations

import json
import logging
import functools
from typing import Any, Callable, Dict, List, Optional

from src.EDATuner.llm.cot_parser import (
    strip_cot,
    parse_cot_response,
    extract_json_from_cot,
    extract_candidates_from_cot,
    extract_content_from_api_response,
    CoTStreamBuffer,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Generic LLM call wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class CoTAwareLLMClient:
    """
    Wraps any LLM client to transparently handle CoT responses.

    This is designed to be a drop-in wrapper around your existing client.

    Example:
        # Your existing client
        from src.EDATuner.llm.openai_client import OpenAIClient
        raw_client = OpenAIClient(cfg)

        # Wrap it
        client = CoTAwareLLMClient(raw_client)

        # Use exactly as before — CoT stripping is automatic
        answer = client.call(messages)
    """

    def __init__(self, inner_client: Any, log_thinking: bool = True):
        """
        Args:
            inner_client: The original LLM client (must have .call() method)
            log_thinking: If True, log thinking content at DEBUG level
        """
        self._inner = inner_client
        self._log_thinking = log_thinking
        self._total_thinking_tokens = 0
        self._total_answer_tokens = 0

    def call(self, *args, **kwargs) -> str:
        """
        Call the inner LLM client and strip CoT thinking from the response.
        """
        raw_response = self._inner.call(*args, **kwargs)

        if not isinstance(raw_response, str):
            # If response is a dict (raw API response), extract content
            if isinstance(raw_response, dict):
                return extract_content_from_api_response(raw_response)
            raw_response = str(raw_response)

        result = parse_cot_response(raw_response)

        if result.has_thinking and self._log_thinking:
            thinking_preview = (result.thinking or "")[:300]
            logger.debug(
                f"[CoT] Stripped {result.thinking_tokens_approx} thinking tokens "
                f"(tag={result.thinking_tag}, truncated={result.was_truncated}): "
                f"{thinking_preview}..."
            )
            self._total_thinking_tokens += result.thinking_tokens_approx
            self._total_answer_tokens += result.answer_tokens_approx

        return result.answer

    def call_json(self, *args, **kwargs) -> Any:
        """
        Call the inner LLM and extract JSON from the response (CoT-stripped).
        """
        raw_response = self._inner.call(*args, **kwargs)
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        parsed = extract_json_from_cot(raw_response)
        if parsed is None:
            # Fallback: try the answer text directly
            answer = strip_cot(raw_response)
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"[CoT] Failed to extract JSON from response: {answer[:200]}...")
                raise ValueError(f"Could not parse JSON from LLM response")

        return parsed

    def call_candidates(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Call the inner LLM and extract parameter candidates from the response.
        """
        raw_response = self._inner.call(*args, **kwargs)
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        candidates = extract_candidates_from_cot(raw_response)
        if not candidates:
            logger.warning(f"[CoT] No candidates extracted from response: {raw_response[:200]}...")

        return candidates

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_thinking_tokens": self._total_thinking_tokens,
            "total_answer_tokens": self._total_answer_tokens,
        }

    # Proxy all other attributes to the inner client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Decorator for wrapping existing call functions
# ═══════════════════════════════════════════════════════════════════════════════

def cot_aware(func: Callable) -> Callable:
    """
    Decorator that makes any LLM call function CoT-aware.

    Usage:
        @cot_aware
        def call_llm(messages, **kwargs):
            response = openai.ChatCompletion.create(messages=messages, **kwargs)
            return response.choices[0].message.content

        # Now call_llm() automatically strips <think> blocks
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> str:
        result = func(*args, **kwargs)
        if isinstance(result, str):
            return strip_cot(result)
        if isinstance(result, dict):
            return extract_content_from_api_response(result)
        return str(result)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Patch helper: monkey-patch an existing client in-place
# ═══════════════════════════════════════════════════════════════════════════════

def patch_client_for_cot(client: Any, method_name: str = "call") -> None:
    """
    Monkey-patch an existing LLM client's call method to strip CoT.

    Usage:
        from src.EDATuner.llm.openai_client import OpenAIClient
        client = OpenAIClient(cfg)
        patch_client_for_cot(client)
        # client.call() now auto-strips thinking blocks
    """
    original_method = getattr(client, method_name)

    @functools.wraps(original_method)
    def patched(*args, **kwargs):
        result = original_method(*args, **kwargs)
        if isinstance(result, str):
            return strip_cot(result)
        if isinstance(result, dict):
            return extract_content_from_api_response(result)
        return result

    setattr(client, method_name, patched)
    logger.info(f"[CoT] Patched {type(client).__name__}.{method_name} for CoT stripping")


# ═══════════════════════════════════════════════════════════════════════════════
# Queue client integration
# ═══════════════════════════════════════════════════════════════════════════════

def strip_cot_from_queue_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a queued LLM response to strip CoT thinking.
    Modifies the response dict in-place and returns it.

    Usage in llm_queue_client.py:
        result = queue.get_result(task_id)
        result = strip_cot_from_queue_response(result)
    """
    if not isinstance(response, dict):
        return response

    # Handle "content" field
    if "content" in response and isinstance(response["content"], str):
        parsed = parse_cot_response(response["content"])
        response["content"] = parsed.answer
        if parsed.thinking:
            response["_thinking"] = parsed.thinking
            response["_thinking_tokens"] = parsed.thinking_tokens_approx

    # Handle "text" field
    if "text" in response and isinstance(response["text"], str):
        response["text"] = strip_cot(response["text"])

    # Handle nested choices
    if "choices" in response:
        for choice in response.get("choices", []):
            msg = choice.get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = strip_cot(str(msg.get("content", "")))

    return response
