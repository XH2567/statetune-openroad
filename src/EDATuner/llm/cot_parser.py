#!/usr/bin/env python3
"""
src/EDATuner/llm/cot_parser.py
================================
Chain-of-Thought (CoT) response parser for reasoning models.

Many reasoning/thinking models (DeepSeek-R1, QwQ, o1-mini, etc.) wrap their
internal reasoning in special tags like <think>...</think> before producing
the actual answer. This module provides utilities to:

1. Strip thinking blocks, returning only the final answer
2. Optionally extract and log the thinking process for debugging
3. Handle various tag formats across different model families
4. Robustly parse partial/malformed thinking blocks

Supported formats:
  - <think>...</think>           (DeepSeek-R1, DeepSeek-V3)
  - <thinking>...</thinking>     (Anthropic-style)
  - <reasoning>...</reasoning>   (generic)
  - <reflection>...</reflection> (generic)
  - <|thinking|>...<|/thinking|> (some open models)

Usage:
    from src.EDATuner.llm.cot_parser import strip_cot, parse_cot_response

    # Simple: just get the answer
    answer = strip_cot(raw_response)

    # Detailed: get both thinking and answer
    result = parse_cot_response(raw_response)
    print(result.answer)       # The final answer only
    print(result.thinking)     # The thinking process (or None)
    print(result.has_thinking) # Whether thinking was detected
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Tag patterns for various reasoning models
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered by priority. Each entry: (open_pattern, close_pattern, name)
_COT_TAG_PATTERNS: List[Tuple[str, str, str]] = [
    # DeepSeek-R1 / DeepSeek-V3 (most common in this project)
    (r"<think>", r"</think>", "think"),
    # Anthropic-style
    (r"<thinking>", r"</thinking>", "thinking"),
    # Generic reasoning tags
    (r"<reasoning>", r"</reasoning>", "reasoning"),
    (r"<reflection>", r"</reflection>", "reflection"),
    # Special token variants (some open-source models)
    (r"<\|thinking\|>", r"<\|/thinking\|>", "thinking_special"),
    (r"<\|think\|>", r"<\|/think\|>", "think_special"),
]

# Compiled regex: matches any thinking block (greedy across newlines)
# Uses re.DOTALL so '.' matches newlines
_COT_STRIP_PATTERNS = []
for open_pat, close_pat, name in _COT_TAG_PATTERNS:
    # Match the full block including tags
    pattern = re.compile(
        rf"{open_pat}(.*?){close_pat}",
        re.DOTALL | re.IGNORECASE,
    )
    _COT_STRIP_PATTERNS.append((pattern, name))

# Pattern for unclosed thinking blocks (model stopped mid-thought)
_UNCLOSED_PATTERNS = []
for open_pat, close_pat, name in _COT_TAG_PATTERNS:
    pattern = re.compile(
        rf"{open_pat}(.*)$",
        re.DOTALL | re.IGNORECASE,
    )
    _UNCLOSED_PATTERNS.append((pattern, name))


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoTParseResult:
    """Result of parsing a CoT response."""
    answer: str                          # The final answer (thinking stripped)
    thinking: Optional[str] = None       # The thinking content (if found)
    has_thinking: bool = False           # Whether any thinking block was detected
    thinking_tag: Optional[str] = None   # Which tag type was found
    raw_response: str = ""               # Original unmodified response
    thinking_tokens_approx: int = 0      # Approximate token count of thinking
    answer_tokens_approx: int = 0        # Approximate token count of answer
    was_truncated: bool = False          # Whether thinking was unclosed (truncated)


# ═══════════════════════════════════════════════════════════════════════════════
# Core parsing functions
# ═══════════════════════════════════════════════════════════════════════════════

def strip_cot(text: str) -> str:
    """
    Strip all CoT thinking blocks from text, returning only the final answer.

    This is the simplest interface — use when you only need the answer.

    Examples:
        >>> strip_cot("<think>Let me reason about this...</think>The answer is 42.")
        'The answer is 42.'

        >>> strip_cot("No thinking tags here.")
        'No thinking tags here.'

        >>> strip_cot("<think>Incomplete thinking block without close tag")
        ''
    """
    if not text:
        return ""

    result = text

    # Strip closed thinking blocks
    for pattern, name in _COT_STRIP_PATTERNS:
        result = pattern.sub("", result)

    # Strip unclosed thinking blocks (model truncated mid-thought)
    for pattern, name in _UNCLOSED_PATTERNS:
        if pattern.search(result):
            result = pattern.sub("", result)

    return result.strip()


def parse_cot_response(text: str) -> CoTParseResult:
    """
    Parse a CoT response, extracting both thinking and answer components.

    Returns a CoTParseResult with:
      - answer: the clean answer text
      - thinking: the thinking content (if any)
      - has_thinking: whether thinking was detected
      - thinking_tag: which tag format was used

    Examples:
        >>> r = parse_cot_response("<think>Step 1: ...</think>Answer: 42")
        >>> r.answer
        'Answer: 42'
        >>> r.thinking
        'Step 1: ...'
        >>> r.has_thinking
        True
    """
    if not text:
        return CoTParseResult(answer="", raw_response="")

    raw = text
    thinking_parts: List[str] = []
    thinking_tag: Optional[str] = None
    was_truncated = False
    remaining = text

    # Extract closed thinking blocks
    for pattern, name in _COT_STRIP_PATTERNS:
        matches = pattern.findall(remaining)
        if matches:
            thinking_tag = thinking_tag or name
            thinking_parts.extend(matches)
            remaining = pattern.sub("", remaining)

    # Check for unclosed thinking block
    if not thinking_parts:
        for pattern, name in _UNCLOSED_PATTERNS:
            match = pattern.search(remaining)
            if match:
                thinking_tag = name
                thinking_parts.append(match.group(1))
                remaining = pattern.sub("", remaining)
                was_truncated = True
                break

    answer = remaining.strip()
    thinking = "\n\n".join(t.strip() for t in thinking_parts if t.strip()) or None

    # Approximate token counts (rough: 1 token ≈ 4 chars for English, 2 chars for Chinese)
    def _approx_tokens(s: str) -> int:
        if not s:
            return 0
        # Mixed CJK/Latin heuristic
        cjk = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
        other = len(s) - cjk
        return int(cjk / 1.5 + other / 4)

    return CoTParseResult(
        answer=answer,
        thinking=thinking,
        has_thinking=bool(thinking_parts),
        thinking_tag=thinking_tag,
        raw_response=raw,
        thinking_tokens_approx=_approx_tokens(thinking or ""),
        answer_tokens_approx=_approx_tokens(answer),
        was_truncated=was_truncated,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# JSON extraction from CoT responses
# ═══════════════════════════════════════════════════════════════════════════════

def extract_json_from_cot(text: str) -> Optional[Any]:
    """
    Strip CoT thinking, then extract JSON from the remaining answer.

    Handles common patterns:
      - JSON in ```json ... ``` code blocks
      - Raw JSON object/array
      - JSON embedded in text

    Returns parsed JSON object, or None if extraction fails.
    """
    import json

    answer = strip_cot(text)
    if not answer:
        return None

    # Try 1: Direct JSON parse
    try:
        return json.loads(answer)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try 2: Extract from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", answer, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 3: Find first JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = answer.find(start_char)
        if start_idx == -1:
            continue
        # Find matching close bracket (handle nesting)
        depth = 0
        for i in range(start_idx, len(answer)):
            if answer[i] == start_char:
                depth += 1
            elif answer[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(answer[start_idx:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
                    break

    return None


def extract_candidates_from_cot(text: str) -> List[Dict[str, Any]]:
    """
    Extract parameter candidate dicts from a CoT response.

    This is specifically designed for the KA agent's output format,
    where the model produces a list of parameter configurations.

    Handles:
      - JSON array of dicts
      - Multiple JSON objects separated by text
      - YAML-like key-value blocks
    """
    import json

    answer = strip_cot(text)
    if not answer:
        return []

    # Try extracting a JSON array
    parsed = extract_json_from_cot(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        # Single candidate wrapped in a dict
        candidates = parsed.get("candidates", parsed.get("params", parsed.get("configurations", None)))
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, dict)]
        return [parsed]

    # Fallback: find all JSON objects in text
    candidates = []
    for match in re.finditer(r"\{[^{}]+\}", answer):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                candidates.append(obj)
        except (json.JSONDecodeError, ValueError):
            continue

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible API response handling
# ═══════════════════════════════════════════════════════════════════════════════

def extract_content_from_api_response(response: Dict[str, Any]) -> str:
    """
    Extract the text content from an OpenAI-compatible API response,
    handling both standard and reasoning model response formats.

    Handles:
      - Standard: response.choices[0].message.content
      - With reasoning: response.choices[0].message.reasoning_content + content
      - DeepSeek: response.choices[0].message.content (with <think> tags inline)

    Returns the ANSWER ONLY (thinking stripped).
    """
    if not isinstance(response, dict):
        return str(response)

    # Handle choices array
    choices = response.get("choices", [])
    if not choices:
        # Direct content field (some APIs)
        content = response.get("content", "")
        if isinstance(content, str):
            return strip_cot(content)
        return ""

    choice = choices[0]
    message = choice.get("message", {}) or {}

    # Some APIs separate reasoning_content from content
    reasoning = message.get("reasoning_content", "")
    content = message.get("content", "")

    if reasoning and content:
        # Model provides separate fields; content is already the answer
        return str(content).strip()

    if content:
        # Content may contain inline <think> tags
        return strip_cot(str(content))

    # Fallback: try text field
    text = message.get("text", choice.get("text", ""))
    if text:
        return strip_cot(str(text))

    return ""


def extract_reasoning_from_api_response(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract the reasoning/thinking content from an API response.
    Returns None if no reasoning content is found.
    """
    if not isinstance(response, dict):
        return None

    choices = response.get("choices", [])
    if not choices:
        content = response.get("content", "")
        if isinstance(content, str):
            result = parse_cot_response(content)
            return result.thinking
        return None

    choice = choices[0]
    message = choice.get("message", {}) or {}

    # Explicit reasoning_content field
    reasoning = message.get("reasoning_content", "")
    if reasoning:
        return str(reasoning).strip()

    # Extract from inline tags
    content = message.get("content", "")
    if content:
        result = parse_cot_response(str(content))
        return result.thinking

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming support
# ═══════════════════════════════════════════════════════════════════════════════

class CoTStreamBuffer:
    """
    Buffer for streaming CoT responses that strips thinking in real-time.

    Usage:
        buf = CoTStreamBuffer()
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            clean = buf.feed(text)
            if clean:
                print(clean, end="")
        # Get any remaining buffered text
        final = buf.flush()
        if final:
            print(final, end="")
    """

    def __init__(self):
        self._buffer: str = ""
        self._in_thinking: bool = False
        self._thinking_tag: Optional[str] = None
        self._output_started: bool = False

    def feed(self, chunk: str) -> str:
        """
        Feed a chunk of streamed text. Returns the portion that should
        be shown to the user (thinking filtered out).
        """
        if not chunk:
            return ""

        self._buffer += chunk
        output = ""

        while self._buffer:
            if self._in_thinking:
                # Look for close tag
                close_idx = -1
                close_tag = ""
                for _, close_pat, name in _COT_TAG_PATTERNS:
                    # Use literal string search for close tags
                    literal = close_pat.replace(r"\|", "|")
                    idx = self._buffer.find(literal)
                    if idx >= 0 and (close_idx < 0 or idx < close_idx):
                        close_idx = idx
                        close_tag = literal

                if close_idx >= 0:
                    # Found close tag — discard thinking content
                    self._buffer = self._buffer[close_idx + len(close_tag):]
                    self._in_thinking = False
                else:
                    # Still inside thinking block — buffer everything
                    break
            else:
                # Look for open tag
                open_idx = -1
                open_tag = ""
                tag_name = ""
                for open_pat, _, name in _COT_TAG_PATTERNS:
                    literal = open_pat.replace(r"\|", "|")
                    idx = self._buffer.find(literal)
                    if idx >= 0 and (open_idx < 0 or idx < open_idx):
                        open_idx = idx
                        open_tag = literal
                        tag_name = name

                if open_idx >= 0:
                    # Output text before the tag
                    output += self._buffer[:open_idx]
                    self._buffer = self._buffer[open_idx + len(open_tag):]
                    self._in_thinking = True
                    self._thinking_tag = tag_name
                else:
                    # No tag found — check if buffer might contain partial tag
                    # Keep last few chars in case a tag is being split across chunks
                    safe_len = max(0, len(self._buffer) - 15)
                    if safe_len > 0:
                        output += self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                    break

        return output

    def flush(self) -> str:
        """Flush remaining buffer (call at end of stream)."""
        if self._in_thinking:
            # Thinking was never closed (truncated)
            self._buffer = ""
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining

    @property
    def in_thinking(self) -> bool:
        return self._in_thinking


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: patch an existing OpenAI client call
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_chat_completion(original_func):
    """
    Decorator that wraps an OpenAI chat completion function to automatically
    strip CoT thinking from responses.

    Usage:
        @wrap_chat_completion
        def call_llm(messages, **kwargs):
            return client.chat.completions.create(messages=messages, **kwargs)

        # Now call_llm automatically strips <think> blocks
        answer = call_llm([{"role": "user", "content": "..."}])
    """
    import functools

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        response = original_func(*args, **kwargs)

        # Handle dict response (raw API)
        if isinstance(response, dict):
            return extract_content_from_api_response(response)

        # Handle OpenAI SDK response object
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""
            stripped = strip_cot(content)
            # Mutate in place for compatibility
            response.choices[0].message.content = stripped
            return response

        # Handle string
        if isinstance(response, str):
            return strip_cot(response)

        return response

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Run basic self-tests."""
    # Test 1: Basic strip
    assert strip_cot("<think>reasoning here</think>The answer is 42.") == "The answer is 42."
    print("✓ Test 1: Basic strip")

    # Test 2: No thinking
    assert strip_cot("Just a normal response.") == "Just a normal response."
    print("✓ Test 2: No thinking")

    # Test 3: Multiline thinking
    text = "<think>\nStep 1: Consider X\nStep 2: Compute Y\n</think>\nResult: Z"
    assert strip_cot(text) == "Result: Z"
    print("✓ Test 3: Multiline thinking")

    # Test 4: Unclosed thinking (truncated)
    assert strip_cot("<think>partial thought without close") == ""
    print("✓ Test 4: Unclosed thinking")

    # Test 5: Multiple thinking blocks
    text = "<think>first</think>A<think>second</think>B"
    assert strip_cot(text) == "AB"
    print("✓ Test 5: Multiple blocks")

    # Test 6: JSON extraction
    text = '<think>Let me figure this out...</think>```json\n{"a": 1}\n```'
    result = extract_json_from_cot(text)
    assert result == {"a": 1}
    print("✓ Test 6: JSON extraction from CoT")

    # Test 7: Candidate extraction
    text = '<think>Analyzing param space...</think>[{"CORE_UTIL": 0.7}, {"CORE_UTIL": 0.6}]'
    cands = extract_candidates_from_cot(text)
    assert len(cands) == 2
    assert cands[0]["CORE_UTIL"] == 0.7
    print("✓ Test 7: Candidate extraction")

    # Test 8: API response
    resp = {
        "choices": [{
            "message": {
                "content": "<think>thinking...</think>Final answer here."
            }
        }]
    }
    assert extract_content_from_api_response(resp) == "Final answer here."
    print("✓ Test 8: API response extraction")

    # Test 9: Separate reasoning_content field
    resp2 = {
        "choices": [{
            "message": {
                "reasoning_content": "My reasoning process...",
                "content": "The answer is 42."
            }
        }]
    }
    assert extract_content_from_api_response(resp2) == "The answer is 42."
    assert extract_reasoning_from_api_response(resp2) == "My reasoning process..."
    print("✓ Test 9: Separate reasoning field")

    # Test 10: Parse result details
    text = "<think>Deep thought here</think>42"
    r = parse_cot_response(text)
    assert r.answer == "42"
    assert r.thinking == "Deep thought here"
    assert r.has_thinking is True
    assert r.thinking_tag == "think"
    print("✓ Test 10: Parse result details")

    # Test 11: Streaming buffer
    buf = CoTStreamBuffer()
    chunks = ["<thi", "nk>reason", "ing here</", "think>The an", "swer."]
    output = ""
    for c in chunks:
        output += buf.feed(c)
    output += buf.flush()
    assert output.strip() == "The answer."
    print("✓ Test 11: Streaming buffer")

    # Test 12: Non-English content
    text = "<think>Let me analyze the parameter space...</think>Recommended config: CORE_UTIL=0.7"
    assert "Recommended config" in strip_cot(text)
    assert "analyze the parameter space" not in strip_cot(text)
    print("✓ Test 12: Non-English content")

    print("\n✅ All self-tests passed!")


if __name__ == "__main__":
    _self_test()
