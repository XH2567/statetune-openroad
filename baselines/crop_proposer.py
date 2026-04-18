#!/usr/bin/env python3
"""
baselines/crop_proposer.py
==========================
CROP-like Baseline: Circuit Retrieval and Optimization with Parameter guidance.

Reproduces the core algorithm spirit from:
  "CROP: Circuit Retrieval and Optimization with Parameter Guidance using LLMs"
  (Pan et al., Duke University & Synopsys, 2025)

Key components:
  1. RAG-based circuit/design knowledge retrieval
  2. Two-stage CoT LLM pipeline:
       Stage 1 — Reasoning model produces strategy digest (explore/exploit/repair)
       Stage 2 — Generation model outputs structured parameter JSON
  3. History-conditioned iterative refinement (feedback loop)
  4. NO Bayesian Optimization / EHVI / knowledge persistence / sensitivity analysis
     — pure LLM + RAG approach for fair baseline comparison

Interface matches RandomProposer / BOProposer / OptunaProposer / RankTunerProposer.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── LLM Client (reuse existing infrastructure) ──
try:
    from src.EDATuner.llm.openai_client import ProviderChatCompletionsClient
except ImportError:
    ProviderChatCompletionsClient = None

# ── RAG Service (reuse existing RAG infrastructure) ──
try:
    from src.EDATuner.knowledge.rag_service import get_rag_service, RAGService
except ImportError:
    get_rag_service = None
    RAGService = None


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: CoT Strategy Digest ──
_COT_SYSTEM = """\
You are a senior EDA/PD tuning strategy advisor.
Based on the CONTEXT below (param space, history, Pareto front, domain knowledge), \
output a SHORT strategy digest.

Requirements:
- 3-8 sentences ONLY. Be extremely concise and actionable.
- Do NOT output JSON, candidates, or parameter values.
- Focus on DIRECTIONS: which param region to explore/exploit/avoid and why.
- Mention specific parameter names and interactions when relevant.
- If the search is stuck (same WNS range, area fixed), say what fundamental change is needed.

Output format: plain text paragraph, no headers, no bullets, no JSON.
"""

_COT_USER = """\
Analyze the following tuning context and provide your strategy digest.
Do NOT propose any candidate configurations. Only analyze and recommend directions.

{context}
"""

# ── Stage 2: Structured Generation ──
_GEN_SYSTEM = """\
You are an expert EDA/VLSI physical design engineer. Your task is to propose \
EDA tool parameter configurations that optimize QoR: minimize Area, maximize WNS \
(Worst Negative Slack — higher/less-negative is better), and minimize Power.

A strategy advisor has analyzed the current state and recommends:
--- STRATEGY DIGEST ---
{cot_digest}
--- END DIGEST ---

Follow the advisor's recommendations. Output ONLY valid JSON:
{{
  "candidates": [
    {{"<param_name>": <value>, ...}},
    ...
  ]
}}

Output {n_candidates} candidates. Each must include ALL parameters. \
Respect ranges and valid choices. Do NOT repeat configurations from history.
Start with '{{' and end with '}}'. No extra text.
"""

_GEN_USER = """\
=== PARAMETER SPACE ===
{param_space_desc}

=== DEFAULT PARAMETERS ===
{default_params_json}

=== DOMAIN KNOWLEDGE (RAG) ===
{rag_context}

=== RECENT RUN HISTORY ({n_history} runs) ===
{history_summary}

Propose {n_candidates} new parameter configurations following the strategy digest.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _format_param_space(param_space: List[Dict[str, Any]]) -> str:
    lines = []
    for p in param_space:
        name = p.get("name", "?")
        ptype = p.get("type", "float")
        if "choices" in p:
            choices_str = ", ".join(str(c) for c in p["choices"])
            lines.append(f"- {name} (categorical): choices=[{choices_str}]")
        elif "range" in p:
            lo, hi = p["range"]
            lines.append(f"- {name} ({ptype}): range=[{lo}, {hi}]")
        else:
            lines.append(f"- {name} ({ptype})")
    return "\n".join(lines)


def _format_history(history: List[Dict[str, Any]], max_items: int = 12) -> str:
    """Compact history: metrics + key params only (no full param dump)."""
    valid = [h for h in history if isinstance(h.get("metrics"), dict)]
    if not valid:
        return "(no history yet)"

    def _sort_key(h):
        m = h.get("metrics", {})
        is_valid = (
            math.isfinite(float(m.get("area", float("inf"))))
            and float(m.get("wns", -100)) > -99
        )
        return (0 if is_valid else 1, -float(h.get("runtime_sec", 0)))

    sorted_hist = sorted(valid[-max_items * 2:], key=_sort_key)[:max_items]
    lines = []
    for h in sorted_hist:
        m = h.get("metrics", {})
        area = m.get("area", "inf")
        wns = m.get("wns", "NA")
        power = m.get("power", "inf")
        status = "OK" if h.get("success", False) else "FAIL"
        # Token optimization: show only placement-effective key parameters instead of the full parameter set.
        p = h.get("params", {})
        key_parts = []
        for k in ("CORE_UTIL", "PLACE_CONG_EFFORT", "PLACE_TIMING_EFFORT",
                   "TD_PLACE", "CORE_MARGIN", "PLACE_MAX_DENSITY"):
            if k in p:
                key_parts.append(f"{k}={p[k]}")
        key_str = ",".join(key_parts) if key_parts else ""
        lines.append(f"  [{status}] A={area} W={wns} P={power} {key_str}")
    return "\n".join(lines)


def _build_context_block(
    param_space: List[Dict[str, Any]],
    default_params: Dict[str, Any],
    history: List[Dict[str, Any]],
    rag_context: str,
    max_history: int = 20,
) -> str:
    """Build the unified context block used by both CoT and generation stages."""
    parts = [
        "=== PARAMETER SPACE ===",
        _format_param_space(param_space),
        "",
        "=== DEFAULT PARAMS ===",
        json.dumps(default_params or {}, ensure_ascii=False),  # Keep compact to reduce prompt size.
        "",
        "=== DOMAIN KNOWLEDGE ===",
        rag_context,
        "",
        f"=== HISTORY ({min(len(history), max_history)} runs) ===",
        _format_history(history, max_history),
    ]
    return "\n".join(parts)


def _build_rag_context(
    rag_service,
    param_space: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    max_chars: int = 4000,
) -> str:
    """Build RAG context by querying relevant domain knowledge."""
    if rag_service is None or not getattr(rag_service, "is_ready", False):
        return _FALLBACK_KNOWLEDGE

    queries = []
    param_names = [p.get("name", "") for p in param_space]
    for name in param_names[:8]:
        queries.append(f"EDA parameter {name} optimization tuning")

    recent_fails = [h for h in history[-10:] if not h.get("success", False)]
    if recent_fails:
        queries.append("fix EDA tool failure crash timeout common causes")
    queries.append("multi-objective optimization area timing power tradeoff")

    try:
        results = rag_service.retrieve_multi(queries, top_k=5)
        if results:
            chunks = []
            total_chars = 0
            for chunk, score in results:
                text = getattr(chunk, "content", str(chunk))
                if total_chars + len(text) > max_chars:
                    break
                chunks.append(text)
                total_chars += len(text)
            if chunks:
                return "\n---\n".join(chunks)
    except Exception:
        pass

    return _FALLBACK_KNOWLEDGE


_FALLBACK_KNOWLEDGE = """\
=== EDA Parameter Tuning Domain Knowledge (built-in) ===
- CORE_UTIL: 0.55-0.75 is safe. >0.75 causes congestion. Lower helps timing.
- PLACE_TIMING_EFFORT "high": improves WNS by ~10-20%, +30% runtime.
- PLACE_CONG_EFFORT "high": reduces congestion but may hurt timing.
- CTS_TARGET_SKEW 0.05-0.10: typical. Lower = better timing but harder to close.
- ROUTE_TIMING_DRIVEN true: critical for meeting timing after routing.
- ROUTE_SI_DRIVEN true: helps with crosstalk but adds runtime.
- PRECTS_SETUP_OPT 1: pre-CTS setup optimization, recommended for tight timing.
- TD_PLACE 1: timing-driven placement, strongly recommended.
- PLACE_UNIFORM_DENSITY true: helps with congestion-sensitive designs.
- DRV_EFFORT "medium" or "high": fixes design rule violations.
- ASPECT_RATIO 1.0: square die is usually optimal. Deviation adds wirelength.
- CORE_MARGIN: larger margin helps IO placement but wastes area.
- Key tradeoff: CORE_UTIL up → Area down but WNS down and congestion up.
- Key tradeoff: PLACE_TIMING_EFFORT "high" + TD_PLACE=1 → best timing but slowest.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CROPProposer
# ═══════════════════════════════════════════════════════════════════════════════

class CROPProposer:
    """
    CROP-like Proposer: Two-stage CoT LLM + RAG for EDA parameter tuning.

    Pipeline (faithful to CROP paper):
      Stage 1 — CoT reasoning model analyses history + RAG context,
                produces a strategy digest (explore / exploit / repair)
      Stage 2 — Generation model outputs structured parameter JSON
                conditioned on the CoT summary + full context

    Differences from EDATuner full system:
      - NO knowledge persistence (hard/soft rules)
      - NO EHVI-guided candidate scoring for proposal
      - NO sensitivity / failure / prior refinement loops
    """

    def __init__(
        self,
        param_space: List[Dict[str, Any]],
        *,
        # Two-stage CoT models
        cot_model: Optional[str] = "qwen3-235b-a22b-thinking-2507",
        generation_model: str = "deepseek-v3.2",
        # Temperature
        llm_temperature: float = 0.6,
        cot_temperature: float = 0.4,
        # Token limits
        llm_max_tokens: int = 8192,
        cot_max_tokens: int = 800,
        # Timeout
        llm_timeout_sec: int = 300,
        # RAG settings
        rag_cfg: Optional[Dict[str, Any]] = None,
        # Prompt sizing
        max_history_in_prompt: int = 12,
        max_rag_context_chars: int = 2500,
        # Token optimization: bound the reasoning-token budget for the CoT model.
        cot_thinking_budget: int = 4096,
        # Fallback
        fallback_to_random: bool = True,
    ):
        self.param_space = param_space
        self.param_names = [p["name"] for p in param_space]

        # Two-stage CoT models
        self.cot_model = cot_model or None
        self.generation_model = generation_model
        self.llm_temperature = llm_temperature
        self.cot_temperature = cot_temperature
        self.llm_max_tokens = llm_max_tokens
        self.cot_max_tokens = cot_max_tokens
        self.llm_timeout_sec = llm_timeout_sec

        # RAG
        self.rag_cfg = rag_cfg or {}
        self.max_rag_context_chars = max_rag_context_chars
        self.max_history_in_prompt = max_history_in_prompt
        self.fallback_to_random = fallback_to_random
        self.cot_thinking_budget = cot_thinking_budget

        # Internal state
        self._client: Optional[Any] = None
        self._rag_service: Optional[Any] = None
        self._init_attempted = False
        self._rng = random.Random()
        self._consecutive_llm_failures: int = 0
        self._last_cot_digest: str = ""
        # Token optimization: maintain a candidate buffer to avoid calling the LLM on every propose().
        self._candidate_buffer: List[Dict[str, Any]] = []

        if self.cot_model:
            print(f"  [CROP] Dual-model: CoT={self.cot_model}, Gen={self.generation_model}")
        else:
            print(f"  [CROP] Single-model: Gen={self.generation_model}")

    # ── Lazy initialization ──────────────────────────────────────────────────
    def _ensure_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True

        api_key = os.getenv("PROVIDER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        base_url = os.getenv("PROVIDER_BASE_URL") or None

        if api_key and ProviderChatCompletionsClient is not None:
            try:
                self._client = ProviderChatCompletionsClient(
                    api_key=api_key,
                    base_url=base_url,
                    timeout_sec=self.llm_timeout_sec,
                )
                print(f"  [CROP] LLM client initialized")
            except Exception as e:
                print(f"  [CROP] LLM client init failed: {e}")
                self._client = None
        else:
            print("  [CROP] No API key found, LLM disabled")

        if get_rag_service is not None and self.rag_cfg.get("enable", False):
            try:
                self._rag_service = get_rag_service(self.rag_cfg)
                if self._rag_service and self._rag_service.is_ready:
                    print(f"  [CROP] RAG service ready")
                else:
                    self._rag_service = None
            except Exception as e:
                print(f"  [CROP] RAG init failed: {e}")
                self._rag_service = None

    def is_ready(self) -> bool:
        self._ensure_init()
        return self._client is not None

    # ── Random fallback ──────────────────────────────────────────────────────
    def _random_sample(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for p in self.param_space:
            name = p["name"]
            if "range" in p:
                lo, hi = p["range"]
                if p.get("type") in ("int", "integer"):
                    params[name] = self._rng.randint(int(lo), int(hi))
                else:
                    params[name] = round(self._rng.uniform(float(lo), float(hi)), 4)
            elif "choices" in p:
                params[name] = self._rng.choice(p["choices"])
            else:
                params[name] = 0.0
        return params

    # ── Stage 1: CoT Strategy Digest ─────────────────────────────────────────
    def _cot_strategy_digest(self, context: str) -> str:
        """
        Stage 1: CoT reasoning model → pure-text strategy digest (3-8 sentences).
        Returns empty string if cot_model is not configured or call fails.
        """
        if not self.cot_model or not self._client:
            return ""

        # Token optimization: cap the hidden reasoning budget to control token usage.
        extra = None
        if self.cot_thinking_budget and self.cot_thinking_budget > 0:
            extra = {"thinking": {"budget_tokens": self.cot_thinking_budget}}

        try:
            text, _raw = self._client.create_text(
                model=self.cot_model,
                messages=[
                    {"role": "system", "content": _COT_SYSTEM},
                    {"role": "user", "content": _COT_USER.format(context=context)},
                ],
                temperature=self.cot_temperature,
                max_tokens=self.cot_max_tokens,
                extra_payload=extra,
            )
            digest = text.strip()
            if digest:
                print(f"  [CROP] CoT digest ({len(digest)} chars): {digest[:120]}...")
                self._last_cot_digest = digest
                return digest
        except Exception as e:
            print(f"  [CROP] CoT stage failed: {e}, skipping to generation")

        return ""

    # ── Stage 2: Structured Generation ───────────────────────────────────────
    def _generate_candidates(
        self,
        cot_digest: str,
        default_params: Dict[str, Any],
        history: List[Dict[str, Any]],
        rag_context: str,
        n_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Stage 2: Generation model → structured parameter JSON."""
        if not self._client:
            return []

        if not cot_digest:
            cot_digest = (
                "No strategy advisor available. Explore diverse configurations "
                "while avoiding recently failed parameter combinations."
            )

        sys_prompt = _GEN_SYSTEM.format(
            cot_digest=cot_digest,
            n_candidates=n_candidates,
        )

        user_prompt = _GEN_USER.format(
            param_space_desc=_format_param_space(self.param_space),
            default_params_json=json.dumps(default_params or {}, ensure_ascii=False),  # ★ no indent
            rag_context="(see strategy digest above)",  # RAG information is already summarized in the digest.
            n_history=min(len(history), 8),  # Stage 2 only needs a short history window for deduplication.
            history_summary=_format_history(history, 8),
            n_candidates=n_candidates,
        )

        text, _raw = self._client.create_text(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )

        return self._parse_llm_response(text)

    # ── Parse LLM JSON output ────────────────────────────────────────────────
    def _parse_llm_response(self, text: str) -> List[Dict[str, Any]]:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        parsed = None
        for fn in [
            lambda t: json.loads(t),
            lambda t: json.loads(t[t.find("{"):t.rfind("}") + 1]),
            lambda t: json.loads(t[t.find("["):t.rfind("]") + 1]),
        ]:
            try:
                parsed = fn(text)
                break
            except Exception:
                continue

        if parsed is None:
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    return []
            else:
                return []

        candidates = []
        if isinstance(parsed, dict):
            if "candidates" in parsed and isinstance(parsed["candidates"], list):
                candidates = parsed["candidates"]
            elif any(name in parsed for name in self.param_names):
                candidates = [parsed]
        elif isinstance(parsed, list):
            candidates = parsed

        cleaned = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            known_keys = set(cand.keys()) & set(self.param_names)
            if len(known_keys) < max(1, len(self.param_names) // 3):
                continue
            valid_cand = {}
            for p in self.param_space:
                name = p["name"]
                if name not in cand:
                    continue
                val = cand[name]
                if "choices" in p:
                    choices = p["choices"]
                    str_val = str(val).lower()
                    matched = [c for c in choices if str(c).lower() == str_val]
                    valid_cand[name] = matched[0] if matched else self._rng.choice(choices)
                elif "range" in p:
                    try:
                        lo, hi = p["range"]
                        fval = max(float(lo), min(float(val), float(hi)))
                        if p.get("type") in ("int", "integer"):
                            valid_cand[name] = int(round(fval))
                        else:
                            valid_cand[name] = round(fval, 4)
                    except (TypeError, ValueError):
                        lo, hi = p["range"]
                        mid = (float(lo) + float(hi)) / 2
                        valid_cand[name] = int(round(mid)) if p.get("type") in ("int", "integer") else round(mid, 4)
            if valid_cand:
                cleaned.append(valid_cand)
        return cleaned

    # ── Main propose interface ───────────────────────────────────────────────
    def propose(
        self,
        history: List[Dict],
        n: int = 1,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Propose n candidates using two-stage CoT LLM + RAG.

        Token optimization: use an internal candidate buffer. The LLM generates
        16 candidates at a time, and later calls consume from the buffer until it
        is empty. Callers should generally use n=1 so that multiple scheduling
        rounds can reuse the buffered candidates. When parsing fails, populate the
        buffer with random fallbacks to avoid repeated failing LLM calls.

        Stage 1: CoT model → strategy digest (plain text)
        Stage 2: Generation model → structured param JSON
        """
        self._ensure_init()

        if not self.is_ready():
            return [self._random_sample() for _ in range(n)] if self.fallback_to_random else []

        # Serve buffered candidates first.
        if self._candidate_buffer:
            result = self._candidate_buffer[:n]
            self._candidate_buffer = self._candidate_buffer[n:]
            while len(result) < n:
                result.append(self._random_sample())
            print(f"  [CROP] Served {len(result)} from buffer ({len(self._candidate_buffer)} remaining)")
            return result

        if self._consecutive_llm_failures >= 3:
            self._consecutive_llm_failures = max(0, self._consecutive_llm_failures - 1)
            print(f"  [CROP] LLM cooling down ({self._consecutive_llm_failures} failures)")
            return [self._random_sample() for _ in range(n)]

        # Request 16 candidates at once and buffer the remainder.
        # In a multi-worker setup this can cover several scheduling rounds while
        # significantly reducing LLM call frequency.
        n_request = 16

        try:
            # Build RAG context
            rag_ctx = _build_rag_context(
                self._rag_service, self.param_space, history,
                max_chars=self.max_rag_context_chars,
            )
            # Build shared context for CoT
            context = _build_context_block(
                self.param_space, default_params or {}, history,
                rag_ctx, self.max_history_in_prompt,
            )

            # ── Stage 1: CoT strategy digest ──
            cot_digest = self._cot_strategy_digest(context)

            # ── Stage 2: Structured generation ──
            candidates = self._generate_candidates(
                cot_digest, default_params or {},
                history, rag_ctx, n_request,
            )

            self._consecutive_llm_failures = 0

            if candidates:
                print(f"  [CROP] Generated {len(candidates)} valid candidates")
                # Return the first n candidates and buffer the remainder.
                result = candidates[:n]
                self._candidate_buffer = candidates[n:]
                if self._candidate_buffer:
                    print(f"  [CROP] Buffered {len(self._candidate_buffer)} candidates for next calls")
                while len(result) < n:
                    result.append(self._random_sample())
                return result
            else:
                # Fallback: fill the buffer with random candidates on parse failure
                # so the next propose() call does not immediately trigger another
                # failing LLM request.
                print(f"  [CROP] No parseable candidates, filling buffer with random fallbacks")
                n_fallback = n_request
                fallbacks = [self._random_sample() for _ in range(n_fallback)]
                result = fallbacks[:n]
                self._candidate_buffer = fallbacks[n:]
                return result

        except Exception as e:
            self._consecutive_llm_failures += 1
            # Fallback: also fill the buffer on exceptions to avoid repeated retries.
            print(f"  [CROP] LLM pipeline failed: {e}, filling buffer with random fallbacks")
            n_fallback = n_request
            fallbacks = [self._random_sample() for _ in range(n_fallback)]
            result = fallbacks[:n]
            self._candidate_buffer = fallbacks[n:]
            return result

    @property
    def strategy_name(self) -> str:
        return "crop_llm_rag"

    # ── State serialization ──────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        return {
            "consecutive_llm_failures": self._consecutive_llm_failures,
            "last_cot_digest": self._last_cot_digest[:500] if self._last_cot_digest else "",
            "candidate_buffer": self._candidate_buffer[:16],  # Persist a small slice of the buffer.
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state and isinstance(state, dict):
            self._consecutive_llm_failures = int(state.get("consecutive_llm_failures", 0))
            self._last_cot_digest = str(state.get("last_cot_digest", ""))
            buf = state.get("candidate_buffer", [])
            self._candidate_buffer = buf if isinstance(buf, list) else []


# ═══════════════════════════════════════════════════════════════════════════════
# Resume helper
# ═══════════════════════════════════════════════════════════════════════════════

def _restore_crop_proposer(
    proposer: CROPProposer,
    history: List[Dict[str, Any]],
    saved_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """Restore CROPProposer state after a resume."""
    if saved_state and isinstance(saved_state, dict):
        proposer.set_state(saved_state)
        print(f"  [Resume] CROP: restored state")
        return True
    print(f"  [Resume] CROP: stateless, {len(history)} history points available")
    return True

