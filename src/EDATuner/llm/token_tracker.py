# src/EDATuner/llm/token_tracker.py
"""
Thread-safe global token usage tracker.

All LLM calls sent through openai_client.py are accumulated automatically.
Supports:
  - Global totals across the full experiment lifecycle
  - Per-run segmentation by run_id
  - Per-model aggregation

Usage:
    from src.EDATuner.llm.token_tracker import token_tracker

    # Start a new run segment
    token_tracker.begin_run("run_0042")

    # ... LLM calls are accumulated automatically ...

    # End the run and retrieve token usage for that segment
    run_usage = token_tracker.end_run()
    # => {"prompt_tokens": 1234, "completion_tokens": 567, "total_tokens": 1801}

    # Retrieve global statistics
    stats = token_tracker.get_global_stats()
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class _TokenUsageTracker:
    """Thread-safe global token usage tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # ── Global totals ──
        self._global_prompt_tokens: int = 0
        self._global_completion_tokens: int = 0
        self._global_total_tokens: int = 0
        self._global_call_count: int = 0

        # ── Per-model aggregation ──
        self._by_model: Dict[str, Dict[str, int]] = {}
        # { "qwen3-235b-...": {"prompt": N, "completion": N, "total": N, "calls": N} }

        # ── Per-run segmentation ──
        self._current_run_id: Optional[str] = None
        self._run_prompt_tokens: int = 0
        self._run_completion_tokens: int = 0
        self._run_total_tokens: int = 0
        self._run_call_count: int = 0

        # ── Timestamps ──
        self._start_time: float = time.time()

    def record(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
    ) -> None:
        """
        Record token usage for a single LLM call.

        This is called automatically by openai_client.py create_text().
        """
        # If the API only returns total_tokens, keep the split fields at zero
        if total_tokens > 0 and prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = 0
            completion_tokens = 0
        # If the split fields are present but total_tokens is missing
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

        with self._lock:
            # Global totals
            self._global_prompt_tokens += prompt_tokens
            self._global_completion_tokens += completion_tokens
            self._global_total_tokens += total_tokens
            self._global_call_count += 1

            # Per-model totals
            if model:
                if model not in self._by_model:
                    self._by_model[model] = {
                        "prompt": 0, "completion": 0, "total": 0, "calls": 0,
                    }
                m = self._by_model[model]
                m["prompt"] += prompt_tokens
                m["completion"] += completion_tokens
                m["total"] += total_tokens
                m["calls"] += 1

            # Current run segment
            if self._current_run_id is not None:
                self._run_prompt_tokens += prompt_tokens
                self._run_completion_tokens += completion_tokens
                self._run_total_tokens += total_tokens
                self._run_call_count += 1

# ── Run segment management ─────────────────────────────────

    def begin_run(self, run_id: str) -> None:
        """Start a new run segment and reset its counters."""
        with self._lock:
            self._current_run_id = run_id
            self._run_prompt_tokens = 0
            self._run_completion_tokens = 0
            self._run_total_tokens = 0
            self._run_call_count = 0

    def end_run(self) -> Dict[str, Any]:
        """
        End the current run segment and return its token usage.

        Returns:
            {
                "run_id": "...",
                "llm_prompt_tokens": N,
                "llm_completion_tokens": N,
                "llm_total_tokens": N,
                "llm_call_count": N,
            }
        """
        with self._lock:
            result = {
                "run_id": self._current_run_id or "",
                "llm_prompt_tokens": self._run_prompt_tokens,
                "llm_completion_tokens": self._run_completion_tokens,
                "llm_total_tokens": self._run_total_tokens,
                "llm_call_count": self._run_call_count,
            }
            self._current_run_id = None
            self._run_prompt_tokens = 0
            self._run_completion_tokens = 0
            self._run_total_tokens = 0
            self._run_call_count = 0
            return result

    def get_run_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the current run segment without resetting it."""
        with self._lock:
            return {
                "run_id": self._current_run_id or "",
                "llm_prompt_tokens": self._run_prompt_tokens,
                "llm_completion_tokens": self._run_completion_tokens,
                "llm_total_tokens": self._run_total_tokens,
                "llm_call_count": self._run_call_count,
            }

# ── Global statistics ─────────────────────────────────────

    def get_global_stats(self) -> Dict[str, Any]:
        """Return global token statistics."""
        with self._lock:
            elapsed = time.time() - self._start_time
            return {
                "llm_total_prompt_tokens": self._global_prompt_tokens,
                "llm_total_completion_tokens": self._global_completion_tokens,
                "llm_total_tokens": self._global_total_tokens,
                "llm_total_calls": self._global_call_count,
                "llm_avg_tokens_per_call": (
                    round(self._global_total_tokens / self._global_call_count, 1)
                    if self._global_call_count > 0 else 0
                ),
                "llm_elapsed_sec": round(elapsed, 1),
                "llm_by_model": dict(self._by_model),
            }

    def reset(self) -> None:
        """Fully reset the tracker (generally only needed for tests)."""
        with self._lock:
            self._global_prompt_tokens = 0
            self._global_completion_tokens = 0
            self._global_total_tokens = 0
            self._global_call_count = 0
            self._by_model.clear()
            self._current_run_id = None
            self._run_prompt_tokens = 0
            self._run_completion_tokens = 0
            self._run_total_tokens = 0
            self._run_call_count = 0
            self._start_time = time.time()


# ── Global singleton ───────────────────────────────────────
token_tracker = _TokenUsageTracker()
