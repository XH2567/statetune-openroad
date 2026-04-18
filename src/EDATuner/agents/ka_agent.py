# src/ka_agent.py
from __future__ import annotations

import random
import time
import re
import json
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..llm.openai_client import OpenAIResponsesClient
from src.EDATuner.llm.cot_parser import extract_candidates_from_cot, strip_cot

from ..knowledge.knowledge_rules import (
    append_jsonl_unique,
    load_jsonl_unique,
    check_hard_rules as eval_hard_rules,
    hard_rule_key,
    soft_heuristic_key,
    render_hard_rule_text,
    render_soft_heuristic_text,
    soft_preferences_from_heuristics,
)

# Keywords used to infer whether an LLM-generated rule expresses a ban or a requirement
_BAN_HINT = re.compile(r"\b(ban|banned|avoid)\b", re.IGNORECASE)
_REQUIRE_HINT = re.compile(r"\b(require|required|must use|requires)\b", re.IGNORECASE)

# Rules that are only warnings are not treated as hard rules by default (this can be adjusted if needed)
_WARNING_HINT = re.compile(r"\bwarning\b", re.IGNORECASE)
_SEVERE_HINT = re.compile(r"\b(crash|failed|failure|invalid|timeout)\b", re.IGNORECASE)

_VALID_OPS = {"==", "!=", "<", "<=", ">", ">=", "in", "not_in"}


def _is_valid_cond(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and isinstance(x.get("param"), str)
        and x.get("param", "").strip() != ""
        and isinstance(x.get("op"), str)
        and x.get("op", "").strip() in _VALID_OPS
        and "value" in x
    )

def compute_strategy(
    history: List[Dict[str, Any]],
    history_full_valid: List[Dict[str, Any]],
    *,
    stagnation_window: int = 10,
    failure_threshold: float = 0.30,
    diversity_threshold: float = 0.15,
) -> Tuple[str, Dict[str, Any]]:
    n = len(history)
    if n < 5:
        return "explore", {"reason": "too_few_runs", "n": n}

    # 1) Failure rate
    recent = history[-stagnation_window:]
    fail_count = sum(1 for h in recent if not h.get("success", False))
    fail_rate = fail_count / len(recent)
    if fail_rate >= failure_threshold:
        return "repair", {
            "reason": "high_failure_rate",
            "recent_fail_rate": round(fail_rate, 3),
            "fail_count": fail_count,
            "window": len(recent),
        }

    # 2) Hypervolume stagnation detection (global)
    hv_values = [
        h.get("cum_hv", 0.0) for h in history
        if h.get("cum_hv") is not None
    ]
    stagnant = False
    relative_change = 0.0
    if len(hv_values) >= stagnation_window:
        recent_hvs = hv_values[-stagnation_window:]
        hv_range = max(recent_hvs) - min(recent_hvs)
        hv_max = max(abs(v) for v in recent_hvs) if recent_hvs else 1.0
        relative_change = hv_range / max(hv_max, 1e-10)
        stagnant = relative_change < 0.01

    # 3) Quick-stage signal stagnation detection
    #    If the main metrics of the last N quick runs have extremely low variance, the LLM is likely repeating ineffective configurations.
    recent_quick = [
        h for h in history[-stagnation_window * 2:]
        if h.get("fidelity") == "quick" and h.get("success", False)
    ]
    quick_stagnant = False
    quick_diag = {}
    if len(recent_quick) >= 8:
        import numpy as np
        wns_vals = []
        area_vals = []
        for h in recent_quick[-stagnation_window:]:
            m = h.get("metrics", {}) or {}
            try:
                wns_vals.append(float(m.get("wns", 0)))
                area_vals.append(float(m.get("area", 0)))
            except (ValueError, TypeError):
                pass

        if wns_vals and area_vals:
            wns_arr = np.array(wns_vals)
            area_arr = np.array(area_vals)

            # Area is effectively fixed, which suggests the die size is constrained
            area_cv = np.std(area_arr) / max(abs(np.mean(area_arr)), 1e-10)
            # Coefficient of variation for WNS
            wns_cv = np.std(wns_arr) / max(abs(np.mean(wns_arr)), 1e-10)

            quick_stagnant = (area_cv < 0.001) and (wns_cv < 0.05)
            quick_diag = {
                "area_cv": round(float(area_cv), 6),
                "wns_cv": round(float(wns_cv), 4),
                "wns_range": [round(float(wns_arr.min()), 2), round(float(wns_arr.max()), 2)],
                "area_fixed": bool(area_cv < 0.001),
                "n_quick_checked": len(wns_vals),
            }

    if quick_stagnant:
        return "diversify", {
            "reason": "quick_stage_signal_saturated",
            "detail": "Area is fixed and WNS has very low variance at quick stage. "
                      "Many parameters (routing, DRV) have no effect at placement. "
                      "Need to explore fundamentally different configurations or "
                      "focus perturbations only on placement-effective parameters.",
            **quick_diag,
        }

    # 4) Parameter-space coverage
    if history_full_valid and len(history_full_valid) >= 5:
        import numpy as np
        param_keys = sorted(set(
            k for h in history_full_valid
            for k in (h.get("params") or {}).keys()
        ))
        vectors = []
        for h in history_full_valid:
            p = h.get("params", {})
            vec = []
            for k in param_keys:
                try:
                    vec.append(float(p.get(k, 0)))
                except (ValueError, TypeError):
                    vec.append(0.0)
            vectors.append(vec)
        if vectors:
            arr = np.array(vectors)
            ranges = arr.max(axis=0) - arr.min(axis=0)
            ranges[ranges == 0] = 1.0
            normed = (arr - arr.min(axis=0)) / ranges
            mean_std = np.mean(np.std(normed, axis=0))
        else:
            mean_std = 0.5
    else:
        mean_std = 0.5

    if stagnant and mean_std < diversity_threshold:
        return "diversify", {
            "reason": "stagnant_low_diversity",
            "hv_relative_change": round(relative_change, 5),
            "param_diversity": round(float(mean_std), 4),
        }

    if stagnant:
        return "diversify", {
            "reason": "hv_stagnant",
            "hv_relative_change": round(relative_change, 5),
            "param_diversity": round(float(mean_std), 4),
        }

    if len(hv_values) >= 3:
        recent_delta = hv_values[-1] - hv_values[-3]
        if recent_delta > 0:
            return "exploit", {
                "reason": "hv_improving",
                "recent_hv_delta": round(recent_delta, 4),
            }

    return "explore", {
        "reason": "default_explore",
        "param_diversity": round(float(mean_std), 4),
    }


@dataclass
class KASuggestion:
    agent_name: str
    model: str
    strategy: str
    rationale: str
    candidates: List[Dict[str, Any]]  # each: {"params":..., "fidelity":..., "tcl_overrides":..., ...}
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class KnowledgeAgent:
    """
    KA = Knowledge Agent (LLM-driven candidate proposer + lightweight "rules" summarizer)

    Input:
      - param_space (list[dict]): [{"name":..., "type":"int/float", "range":[a,b] or "choices":[...]}, ...]
      - default_params (dict): baseline make vars
      - history (list[dict]): finished trials from KB (must contain params/metrics/runtime/fidelity/success)
      - pareto (optional): list of pareto points (same schema as history items)
      - remaining_sec (optional)
      - stage hint ("quick"/"full"/"promote"/...): used for prompt + fidelity suggestion

    Output:
      KASuggestion with ordered candidates, plus strategy/rationale for console.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        llm_cfg = cfg.get("llm", {}) or {}
        ka_cfg = cfg.get("ka", {}) or {}

        self.enabled = bool(llm_cfg.get("enable", False) or ka_cfg.get("enable", False))
        self.temperature = float(llm_cfg.get("temperature", ka_cfg.get("temperature", 0.2)))
        self.max_output_tokens = int(llm_cfg.get("max_output_tokens", ka_cfg.get("max_output_tokens", 4000)))
        self.timeout_sec = int(llm_cfg.get("timeout_sec", ka_cfg.get("timeout_sec", 45)))
        self.max_candidates_per_call = int(
            llm_cfg.get("max_candidates_per_call", ka_cfg.get("max_candidates_per_call", 8))
        )

        # Dual-model configuration
        # cot_model: thinking/reasoning model used only for strategy summarization (optional; None falls back to single-model mode)
        # generation_model: non-thinking model used for structured JSON candidate generation (required)
        self.cot_model = llm_cfg.get("cot_model", None) or None
        self.generation_model = str(
            llm_cfg.get("generation_model", None)
            or llm_cfg.get("model", ka_cfg.get("model", "deepseek-v3.2"))
        )
        # Compatibility: self.model points to the actual candidate-generation model
        self.model = self.generation_model
        self.cot_max_output_tokens = int(llm_cfg.get("cot_max_output_tokens", 2000))
        self.cot_temperature = float(llm_cfg.get("cot_temperature", 0.4))

        if self.cot_model:
            print(f"[KA] Dual-model mode: CoT={self.cot_model}, Gen={self.generation_model}")
        else:
            print(f"[KA] Single-model mode: Gen={self.generation_model}")

        self.agent_name = str(ka_cfg.get("agent_name", "KA"))
        self.design_prior_text = ""   # Injected by main.py

        # LLM backend selection:
        # Offline/file-queue mode was removed. We always use the online
        # OpenAI-compatible chat-completions API client here.
        requested_backend = str(llm_cfg.get("backend", llm_cfg.get("mode", "online"))).strip().lower()
        if requested_backend and requested_backend not in ("online", "network", "api"):
            print(
                f"[KA] llm.backend={requested_backend!r} is no longer supported; "
                "falling back to online API mode."
            )
        self.llm_backend = "online"
        self._client = OpenAIResponsesClient(timeout_sec=self.timeout_sec)

        # Official DeepSeek fallback (automatically used on timeout)
        _ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if _ds_key:
            self._deepseek_client = OpenAIResponsesClient(
                api_key=_ds_key,
                base_url="https://api.deepseek.com",
                path="/v1/chat/completions",
                timeout_sec=self.timeout_sec,
            )
            print("[KA] DeepSeek official fallback client loaded ✓")
        else:
            self._deepseek_client = None
            print("[KA] DEEPSEEK_API_KEY not set, no fallback")


        # Persistent heuristic rules (append-only)
        rules_cfg = ka_cfg.get("rules", {}) or {}
        base = Path(cfg.get("work_dir", ".")).resolve()

        def _resolve_path(rel_or_abs: str) -> Path:
            p = Path(str(rel_or_abs))
            return p.resolve() if p.is_absolute() else (base / p).resolve()

        self.hard_rules_path = _resolve_path(str(rules_cfg.get("hard_path", "knowledge/hard_rules.jsonl")))
        self.soft_heuristics_path = _resolve_path(str(rules_cfg.get("soft_path", "knowledge/soft_heuristics.jsonl")))

        # Backward-compat legacy text heuristics (old rules.txt)
        self.legacy_soft_rules_path = _resolve_path(str(rules_cfg.get("path", "knowledge/rules.txt")))

        self.max_hard_rules_in_prompt = int(rules_cfg.get("max_hard_in_prompt", 40))
        self.max_soft_heuristics_in_prompt = int(rules_cfg.get("max_soft_in_prompt", 80))

        # Context vars for rules (optional)
        orfs_cfg = cfg.get("orfs", {}) or {}
        cad_cfg = cfg.get("cadence", {}) or {}
        self.platform = str(orfs_cfg.get("platform") or cad_cfg.get("platform") or "").strip()

        self.dump_bad_llm = bool(ka_cfg.get("dump_bad_llm", True))
        self.dump_bad_llm_max_chars = int(ka_cfg.get("dump_bad_llm_max_chars", 8000))
        self.design_prior_text = ""       
        self.failure_analysis_text = "" 
        self.sensitivity_text = ""

        self.min_full_valid_for_hard_rules = int(ka_cfg.get("min_full_valid_for_hard_rules", 12))
        self.max_active_hard_rules = int(ka_cfg.get("max_active_hard_rules", 8))
        self.hard_rule_trim_interval = int(ka_cfg.get("hard_rule_trim_interval", 50))
        self.hard_rule_mode = str(ka_cfg.get("hard_rule_mode", "soft")).strip().lower()
        self.max_active_soft_heuristics = int(ka_cfg.get("max_active_soft_heuristics", 30))
        self._current_full_valid_count = 0   # Updated by main.py on each iteration
        self._feedback_stats = {
            "total_proposed": 0,
            "total_evaluated": 0,
            "success_count": 0,
            "pareto_hit_count": 0,
            "hv_improvement_sum": 0.0,
            "recent_proposed_ids": [],  # run_id values from the most recent N proposals
        }
    
    def update_feedback(self, run_result: Dict[str, Any], is_pareto: bool) -> None:
        """Called by main.py after each KA-proposed run completes."""
        self._feedback_stats["total_evaluated"] += 1
        if run_result.get("success", False):
            self._feedback_stats["success_count"] += 1
        if is_pareto:
            self._feedback_stats["pareto_hit_count"] += 1
        hv_delta = float(run_result.get("cum_hv_delta", 0.0))
        if hv_delta > 0:
            self._feedback_stats["hv_improvement_sum"] += hv_delta

    def record_proposed(self, run_ids: List[str]) -> None:
        """Record the run_id values proposed in the current round."""
        self._feedback_stats["total_proposed"] += len(run_ids)
        self._feedback_stats["recent_proposed_ids"].extend(run_ids)
        # Keep only the most recent 50 entries
        self._feedback_stats["recent_proposed_ids"] = \
            self._feedback_stats["recent_proposed_ids"][-50:]

    def _format_feedback_block(self) -> str:
        """Build the feedback summary injected into the prompt."""
        stats = self._feedback_stats
        total_eval = stats["total_evaluated"]
        if total_eval < 3:
            return ""
        
        success_rate = stats["success_count"] / total_eval
        pareto_rate = stats["pareto_hit_count"] / total_eval
        avg_hv_gain = stats["hv_improvement_sum"] / total_eval

        lines = [
            "=== YOUR PAST PERFORMANCE (KA self-reflection) ===",
            f"Total proposals evaluated: {total_eval}",
            f"Success rate: {success_rate:.1%}",
            f"Pareto-front hit rate: {pareto_rate:.1%}",
            f"Average HV improvement per proposal: {avg_hv_gain:.4f}",
        ]

        if success_rate < 0.5:
            lines.append("⚠ Your success rate is LOW. Propose safer configurations.")
        if pareto_rate < 0.1 and total_eval >= 10:
            lines.append("⚠ Your Pareto hit rate is LOW. Analyze which params of Pareto points "
                         "differ from your proposals and adjust accordingly.")
        if pareto_rate > 0.3:
            lines.append("✓ Good Pareto hit rate. Continue current direction with refinements.")

        lines.append("=== END PERFORMANCE ===")
        return "\n".join(lines) + "\n"
    
    def _dbg(self, msg: str) -> None:
        print(msg)

    def _raw_to_text(self, raw: dict) -> str:
        # Compatible with multiple response shapes:
        # 1) Provider/OpenAI chat-completions style: choices[0].message.content
        # 2) Some wrappers may return {'content': '...'} or {'text': '...'}
        try:
            if isinstance(raw, dict):
                for k in ("content", "text", "assistant", "output_text"):
                    v = raw.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                msg = raw.get("message")
                if isinstance(msg, dict):
                    v = msg.get("content")
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass

        # provider chat/completions: choices[0].message.content
        try:
            choices = raw.get("choices") or []
            if choices and isinstance(choices[0], dict):
                c0 = choices[0]
                msg = c0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "").strip()
                if "text" in c0:
                    return str(c0.get("text") or "").strip()
        except Exception:
            pass
        return ""

    def is_ready(self) -> bool:
        return self.enabled and self._client.is_ready()

    def _normalize_hard_rule(self, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rule_type = str(r.get("rule_type", "")).strip().lower()
        when = r.get("when", []) or []
        must = r.get("must", []) or []
        message = str(r.get("message", "")).strip()

        if rule_type not in ("ban", "require"):
            return None

        when = [c for c in when if _is_valid_cond(c)]
        must = [c for c in must if _is_valid_cond(c)]
        if not when:
            return None

        # Hard rules derived only from warnings can over-constrain the search space, so they are rejected by default here.
        if _WARNING_HINT.search(message) and not _SEVERE_HINT.search(message):
            return None

        # Core guardrail: the combination of ban + must is high-risk because it is easy to invert the intended meaning of require.
        if rule_type == "ban" and must:
            if _BAN_HINT.search(message):
                # If it looks more like forbidding a combination, merge must into when to form a stricter ban condition.
                when = when + must
                must = []
            elif _REQUIRE_HINT.search(message):
                # If it looks more like a mandatory requirement, convert it to require
                rule_type = "require"
                if not must:
                    return None
            else:
                # If the intent cannot be determined safely, reject the rule
                return None

        if rule_type == "require" and not must:
            return None

        # Guardrail: a ban rule cannot be based only on contextual keys such as __stage__, __fidelity__, or __platform__
        if rule_type == "ban":
            has_real_param = any(
                isinstance(c.get("param"), str) and not c["param"].startswith("__")
                for c in when
            )
            if not has_real_param:
                return None

        return {
            "rule_type": rule_type,
            "when": when,
            "must": must,
            "message": message,
        }

    # ---------------------------
    # CoT Strategy Digest (Phase 1 of dual-model pipeline)
    # ---------------------------
    def _build_slim_cot_context(
        self,
        *,
        history: List[Dict[str, Any]],
        pareto: Optional[List[Dict[str, Any]]],
        stage: str,
        remaining_sec: Optional[float],
        param_space: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a compact context block for the CoT model.
        It contains only the key information needed for strategy decisions, without the full parameter space or detailed parameters for every run.
        This saves roughly 50% of the input tokens compared with the full user prompt.
        """
        import math

        lines: List[str] = []

        # 1) Basic state
        n_total = len(history)
        n_success = sum(1 for h in history if h.get("success", False))
        n_fail = n_total - n_success
        fv = [h for h in history if h.get("metrics_valid", False)]
        lines.append(f"Stage: {stage} | Total runs: {n_total} | Success: {n_success} | Fail: {n_fail} | Full-valid: {len(fv)}")
        if remaining_sec is not None:
            lines.append(f"Remaining time: {remaining_sec:.0f}s ({remaining_sec/3600:.1f}h)")

        # 1.5) Parameter bounds (to prevent CoT suggestions outside valid ranges)
        if param_space:
            lines.append("\nPARAM BOUNDS (strictly enforced, do NOT suggest values outside these):")
            for p in param_space:
                name = str(p.get("name", "")).strip()
                if not name:
                    continue
                if "range" in p and p["range"] is not None:
                    lines.append(f"  {name}: [{p['range'][0]}, {p['range'][1]}]")
                elif "choices" in p and p["choices"] is not None:
                    lines.append(f"  {name}: choices={p['choices']}")

        # 2) Strategy diagnostics (output of compute_strategy)
        strategy, diag = compute_strategy(history, fv)
        lines.append(f"Computed strategy: {strategy} | Reason: {diag.get('reason', '?')}")
        for k, v in diag.items():
            if k != "reason":
                lines.append(f"  {k}: {v}")

        # 3) Pareto summary (metrics only, without params)
        if pareto:
            lines.append(f"\nPareto front ({len(pareto)} points):")
            for p in pareto[:5]:
                m = p.get("metrics", {}) or {}
                lines.append(f"  WNS={m.get('wns','?')} Area={m.get('area','?')} Power={m.get('power','?')}")

        # 4) Recent failure summary (symptoms only, without params or log tails)
        recent_fails = [h for h in history[-15:] if not h.get("success", False)]
        if recent_fails:
            lines.append(f"\nRecent failures ({len(recent_fails)} in last 15):")
            for h in recent_fails[-5:]:
                ls = h.get("log_summary", {}) or {}
                syms = ls.get("symptoms", []) if isinstance(ls, dict) else []
                brief = (ls.get("brief", "") if isinstance(ls, dict) else "")[:100]
                lines.append(f"  {h.get('run_id','?')}: symptoms={syms} brief={brief}")

        # 5) Inject existing sensitivity and failure-analysis text (truncated)
        if getattr(self, "sensitivity_text", ""):
            sens = str(self.sensitivity_text).strip()
            if len(sens) > 600:
                sens = sens[:600] + "..."
            lines.append(f"\nSensitivity: {sens}")

        if getattr(self, "failure_analysis_text", ""):
            fa = str(self.failure_analysis_text).strip()
            if len(fa) > 600:
                fa = fa[:600] + "..."
            lines.append(f"\nFailure patterns: {fa}")

        # 6) EHVI regions (if available)
        regions = getattr(self, "ehvi_promising_regions", [])
        if regions:
            lines.append("\nEHVI promising regions:")
            for i, r in enumerate(regions[:3]):
                lines.append(f"  #{i+1}: EHVI={r.get('ehvi_score',0):.4f} params={r.get('params',{})}")

        return "\n".join(lines)

    def _cot_strategy_digest(
        self,
        *,
        user_context: str,
        history: List[Dict[str, Any]],
        stage: str,
        pareto: Optional[List[Dict[str, Any]]] = None,
        remaining_sec: Optional[float] = None,
        param_space: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Ask the thinking/reasoning model to output only a strategy digest (plain text, 3-8 sentences).
        No structured JSON is required.
        Use the compact context instead of the full user prompt.
        """
        if not self.cot_model:
            return ""

        digest_system = (
            "You are a senior EDA/PD tuning strategy advisor.\n"
            "Based on the CONTEXT below (run stats, Pareto front, "
            "failure patterns, sensitivity data), output a SHORT strategy digest.\n"
            "\n"
            "Requirements:\n"
            "- 3-8 sentences ONLY. Be extremely concise and actionable.\n"
            "- Do NOT output JSON, candidates, or parameter values.\n"
            "- Focus on DIRECTIONS: which param region to explore/exploit/avoid and why.\n"
            "- Mention specific parameter names and interactions when relevant.\n"
            "- If you notice the search is stuck (e.g., same WNS range, area fixed), "
            "explicitly say what fundamental change is needed.\n"
            "- If quick-stage parameters are exhausted, say 'promote existing configs to full'.\n"
            "\n"
            "Output format: plain text paragraph, no headers, no bullets, no JSON.\n"
        )

        # Use the compact context instead of the full user prompt
        slim_ctx = self._build_slim_cot_context(
            history=history,
            pareto=pareto,
            stage=stage,
            remaining_sec=remaining_sec,
            param_space=param_space,
        )

        digest_user = (
            "Analyze the following tuning context and provide your strategy digest.\n"
            "Do NOT propose any candidate configurations. "
            "Only analyze and recommend directions.\n"
            "\n"
            + slim_ctx
        )

        text, _ = self._client.create_text(
            model=self.cot_model,
            messages=[
                {"role": "system", "content": digest_system},
                {"role": "user", "content": digest_user},
            ],
            temperature=self.cot_temperature,
            max_output_tokens=self.cot_max_output_tokens,
        )

        # Strip <think> tags
        cleaned = strip_cot(text).strip()

        # Safety fallback in case the CoT model unexpectedly returns JSON
        if cleaned.startswith("{") or cleaned.startswith("["):
            print("[KA] CoT digest returned JSON instead of text, truncating.")
            cleaned = cleaned[:800]

        # Truncate overly long digests
        if len(cleaned) > 1500:
            cleaned = cleaned[:1500] + "..."

        return cleaned

    # ---------------------------
    # Public API
    # ---------------------------
    def propose_candidates(
        self,
        *,
        param_space: List[Dict[str, Any]],
        default_params: Dict[str, Any],
        history: List[Dict[str, Any]],
        pareto: Optional[List[Dict[str, Any]]] = None,
        remaining_sec: Optional[float] = None,
        stage: str = "quick",
        k: int = 4,
        seed: Optional[int] = None,
    ) -> KASuggestion:
        if seed is not None:
            random.seed(int(seed))

        k = max(1, min(int(k), self.max_candidates_per_call))

        if not self.is_ready():
            cands = [self._random_candidate(param_space, default_params, stage=stage) for _ in range(k)]
            return KASuggestion(
                agent_name=self.agent_name,
                model=self.model,
                strategy="fallback_random",
                rationale="LLM disabled or online API not ready (set PROVIDER_API_KEY or OPENAI_API_KEY). Fallback to random sampling.",
                candidates=cands,
                raw=None,
                error=None,
            )

        schema = self._build_json_schema(max_k=k)

        system_msg = self._system_prompt()
        user_msg = self._user_prompt(
            param_space=param_space,
            default_params=default_params,
            history=history,
            pareto=pareto,
            remaining_sec=remaining_sec,
            stage=stage,
            k=k,
        )

        # Stage 1: CoT strategy digest (only used when cot_model is configured)
        # Pass pareto/remaining_sec so the CoT stage can use the compact context internally
        strategy_digest = ""
        if self.cot_model:
            _cot_max_retries = 3
            for _cot_attempt in range(1, _cot_max_retries + 1):
                try:
                    strategy_digest = self._cot_strategy_digest(
                        user_context=user_msg,
                        history=history,
                        stage=stage,
                        pareto=pareto,
                        remaining_sec=remaining_sec,
                        param_space=param_space,
                    )
                    print(f"[KA] CoT digest ({len(strategy_digest)} chars): "
                          f"{strategy_digest[:300]}...")
                    break
                except Exception as e:
                    if _cot_attempt < _cot_max_retries:
                        _cot_wait = min(30 * (2 ** (_cot_attempt - 1)), 120) + random.randint(0, 10)
                        print(f"[KA] CoT connection error (attempt {_cot_attempt}/{_cot_max_retries}): {e}")
                        print(f"[KA] Retrying CoT in {_cot_wait}s ...")
                        time.sleep(_cot_wait)
                    else:
                        print(f"[KA] CoT failed after {_cot_max_retries} attempts ({e}), proceeding without CoT.")
                        strategy_digest = ""

        # If a CoT digest is available, inject it into the system prompt
        if strategy_digest:
            system_msg += (
                "\n\n=== AI STRATEGY ADVISOR ANALYSIS ===\n"
                + strategy_digest.strip()
                + "\n=== END ADVISOR ANALYSIS ===\n"
                "\nUse the above analysis to guide your candidate selection. "
                "It provides high-level direction — you MUST still output valid JSON "
                "matching the schema.\n"
            )

        adaptive_temp = self.compute_adaptive_temperature(
            history,
            base_temperature=self.temperature,
        )

        # Stage 2: structured candidate generation with a non-thinking model
        try:
            parsed, raw = self._client.create_structured_json(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                json_schema=schema,
                temperature=adaptive_temp,
                max_output_tokens=max(1, min(int(self.max_output_tokens), 8192)),
                metadata={
                    "agent": self.agent_name,
                    "ts": str(int(time.time())),
                    "adaptive_temperature": str(adaptive_temp),
                    "cot_model": str(self.cot_model or "none"),
                    "cot_digest_len": str(len(strategy_digest)),
                },
            )

            strategy = str(parsed.get("strategy", "unknown"))
            rationale = str(parsed.get("rationale", "")).strip()

            # 1) Persist KA knowledge
            hard_rules_new = parsed.get("hard_rules", []) or []
            soft_heuristics_new = parsed.get("soft_heuristics", []) or []

            legacy_rule = str(parsed.get("heuristic_rule", "")).strip()
            if legacy_rule:
                soft_heuristics_new = list(soft_heuristics_new) + [{"trigger": "other", "preferences": [], "notes": legacy_rule}]

            self._append_hard_rules(hard_rules_new)
            self._append_soft_heuristics(soft_heuristics_new)
            # 2) Build a rule set for filtering (load ALL to be safe)
            hard_all = self._load_hard_rules(limit=None)

            # Determine the filtering mode
            _hr_mode = getattr(self, "hard_rule_mode", "soft")

            # 3) Validate + clamp candidates + hard-rule filtering
            raw_candidates = parsed.get("candidates", []) or []
            candidates: List[Dict[str, Any]] = []
            rejected = []  # debug only

            for item in raw_candidates:
                ok, fixed = self._validate_and_fix_candidate(
                    item=item,
                    param_space=param_space,
                    default_params=default_params,
                    stage=stage,
                )
                if not ok:
                    rejected.append({"reason": "validate_fail", "item": item})
                    continue

                fixed["proposed_strategy"] = strategy
                fixed["proposed_model"] = self.model

                viol = self.hard_rule_violations(
                    fixed.get("params", {}) or {},
                    stage=stage,
                    fidelity=str(fixed.get("fidelity", "quick")),
                    hard_rules=hard_all,
                )

                if viol:
                    if _hr_mode == "hard":
                        # Original hard-rejection logic
                        rejected.append({"reason": "hard_rule", "violations": viol, "item": fixed})
                        continue
                    else:
                        # Soft mode: record the number of violations instead of discarding the candidate
                        fixed["_hard_rule_penalty"] = len(viol)
                        fixed["_hard_rule_violations"] = viol
                        print(f"[KA] candidate soft-penalized: {len(viol)} violations")
                else:
                    fixed["_hard_rule_penalty"] = 0

                candidates.append(fixed)

            # In soft mode, sort by the number of violations and prioritize candidates with none
            if _hr_mode != "hard" and candidates:
                candidates.sort(key=lambda c: c.get("_hard_rule_penalty", 0))

            candidates = candidates[:k]

            if not candidates:
                self._dbg("[KA-DEBUG] LLM returned 0 usable candidates after validation/filtering.")
                txt = self._raw_to_text(raw or {})
                if txt:
                    self._dbg("[KA-DEBUG] assistant_text:\n" + txt[: self.dump_bad_llm_max_chars])

                try:
                    self._dbg("[KA-DEBUG] parsed:\n" + json.dumps(parsed, ensure_ascii=False, indent=2)[: self.dump_bad_llm_max_chars])
                except Exception:
                    self._dbg("[KA-DEBUG] parsed=<unprintable>")

                try:
                    self._dbg("[KA-DEBUG] rejected:\n" + json.dumps(rejected, ensure_ascii=False, indent=2)[: self.dump_bad_llm_max_chars])
                except Exception:
                    self._dbg("[KA-DEBUG] rejected=<unprintable>")

                try:
                    self._dbg("[KA-DEBUG] raw:\n" + json.dumps(raw, ensure_ascii=False)[: self.dump_bad_llm_max_chars])
                except Exception:
                    self._dbg("[KA-DEBUG] raw=<unprintable>")

                candidates = []
                strategy = "empty_after_invalid"
                rationale = ((rationale + "\n") if rationale else "") + "LLM output invalid/empty after validation/hard-rule filtering; returning empty (no random fallback)."
            
            self.param_space_cache = param_space
            candidates = self.rerank_candidates_with_rules(candidates)

            return KASuggestion(
                agent_name=self.agent_name,
                model=self.model,
                strategy=strategy,
                rationale=rationale,
                candidates=candidates,
                raw=raw,
                error=None,
            )

        except Exception as e:
            _gen_max_retries = 5
            print(f"[KA] LLM connection error (attempt 1): {e}")

            # Timeout detection: switch to the official DeepSeek endpoint immediately on timeout
            _err_str = str(e).lower()
            # Official API model-name mapping: deepseek-v3.2 -> deepseek-chat
            _ds_model = "deepseek-chat" if "v3" in self.generation_model.lower() else self.generation_model
            _is_timeout = any(kw in _err_str for kw in (
                "timeout", "timed out", "read timed out", "connect timed out",
            ))
            if _is_timeout and self._deepseek_client is not None:
                print(f"[KA] ⚡ Timeout detected! Switching to DeepSeek official")
                _active_client = self._deepseek_client
            else:
                _active_client = self._client

            _gen_succeeded = False
            for _gen_attempt in range(2, _gen_max_retries + 1):
                _gen_wait = min(30 * (2 ** (_gen_attempt - 2)), 120) + random.randint(0, 10)
                _via = "DeepSeek official" if _active_client is self._deepseek_client else "primary"
                print(f"[KA] Retrying in {_gen_wait}s via {_via} ... (attempt {_gen_attempt}/{_gen_max_retries})")
                time.sleep(_gen_wait)
                try:
                    parsed, raw = _active_client.create_structured_json(
                        model=_ds_model if _active_client is self._deepseek_client else self.generation_model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        json_schema=schema,
                        temperature=adaptive_temp,
                        max_output_tokens=max(1, min(int(self.max_output_tokens), 8192)),
                        metadata={
                            "agent": self.agent_name,
                            "ts": str(int(time.time())),
                            "adaptive_temperature": str(adaptive_temp),
                            "retry_attempt": str(_gen_attempt),
                            "via": _via,
                        },
                    )
                    _gen_succeeded = True
                    print(f"[KA] LLM reconnected on attempt {_gen_attempt} via {_via}")
                    break
                except Exception as e2:
                    print(f"[KA] LLM connection error (attempt {_gen_attempt}): {e2}")
                    # Switch to DeepSeek for the remaining retries if retries against the primary backend fail
                    if self._deepseek_client is not None and _active_client is self._client:
                        print(f"[KA] ⚡ Switching to DeepSeek official for remaining retries")
                        _active_client = self._deepseek_client
                    e = e2
                    continue


            if not _gen_succeeded:
                print(f"[KA ERROR] LLM failed after {_gen_max_retries} attempts, returning empty (no random fallback)")
                return KASuggestion(
                    agent_name=self.agent_name,
                    model=self.generation_model,
                    strategy="empty_after_connection_failure",
                    rationale=f"LLM call failed after {_gen_max_retries} attempts ({e}); returning empty.",
                    candidates=[],
                    raw=None,
                    error=str(e),
                )

            # ---- retry succeeded, run full processing (same as main try block) ----
            strategy = str(parsed.get("strategy", "unknown"))
            rationale = str(parsed.get("rationale", "")).strip()

            hard_rules_new = parsed.get("hard_rules", []) or []
            soft_heuristics_new = parsed.get("soft_heuristics", []) or []
            legacy_rule = str(parsed.get("heuristic_rule", "")).strip()
            if legacy_rule:
                soft_heuristics_new = list(soft_heuristics_new) + [{"trigger": "other", "preferences": [], "notes": legacy_rule}]
            self._append_hard_rules(hard_rules_new)
            self._append_soft_heuristics(soft_heuristics_new)
            hard_all = self._load_hard_rules(limit=None)
            _hr_mode = getattr(self, "hard_rule_mode", "soft")

            raw_candidates = parsed.get("candidates", []) or []
            candidates: List[Dict[str, Any]] = []
            rejected = []
            for item in raw_candidates:
                ok, fixed = self._validate_and_fix_candidate(
                    item=item, param_space=param_space,
                    default_params=default_params, stage=stage,
                )
                if not ok:
                    rejected.append({"reason": "validate_fail", "item": item})
                    continue
                fixed["proposed_strategy"] = strategy
                fixed["proposed_model"] = self.model
                viol = self.hard_rule_violations(
                    fixed.get("params", {}) or {}, stage=stage,
                    fidelity=str(fixed.get("fidelity", "quick")), hard_rules=hard_all,
                )
                if viol:
                    if _hr_mode == "hard":
                        rejected.append({"reason": "hard_rule", "violations": viol, "item": fixed})
                        continue
                    else:
                        fixed["_hard_rule_penalty"] = len(viol)
                        fixed["_hard_rule_violations"] = viol
                else:
                    fixed["_hard_rule_penalty"] = 0
                candidates.append(fixed)

            if not candidates:
                candidates = []
                strategy = "empty_after_invalid"
                rationale = ((rationale + "\n") if rationale else "") + "LLM output invalid/empty after retry; returning empty (no random fallback)."

            if _hr_mode != "hard" and candidates:
                candidates.sort(key=lambda c: c.get("_hard_rule_penalty", 0))
            candidates = candidates[:k]

            self.param_space_cache = param_space
            candidates = self.rerank_candidates_with_rules(candidates)
            return KASuggestion(
                agent_name=self.agent_name,
                model=self.model,
                strategy=strategy,
                rationale=rationale,
                candidates=candidates,
                raw=raw,
                error=None,
            )

    # ---------------------------
    # Persistent knowledge: hard rules + soft heuristics
    # ---------------------------
    def _strip_rule_timestamp(self, line: str) -> str:
        """Remove leading timestamp like [YYYY-MM-DD HH:MM:SS] ..."""
        s = (line or "").strip()
        s = re.sub(r"^\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*", "", s)
        return s.strip()

    def _clean_rule_text(self, rule: str) -> str:
        """Legacy text rule cleaning (for rules.txt compatibility)."""
        s = self._strip_rule_timestamp(rule)
        if not s:
            return ""
        s = (
            s.replace("≤", "<=")
            .replace("≥", ">=")
            .replace("–", "-")
            .replace("—", "-")
            .replace("＝", "=")
        )
        s = " ".join(s.split()).strip()
        while s.endswith((".", "。", ";", "；")):
            s = s[:-1].rstrip()
        return s

    def _load_legacy_soft_rules_text(self, *, limit: Optional[int]) -> List[str]:
        """
        Load legacy soft heuristics from old rules.txt (if exists).
        Returns oldest->newest.
        """
        try:
            p = Path(self.legacy_soft_rules_path)
            if not p.exists():
                return []
            raw_lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

            seen = set()
            uniq_recent: List[str] = []
            for ln in reversed(raw_lines):
                txt = self._clean_rule_text(ln)
                if not txt:
                    continue
                k = txt.casefold()
                if k in seen:
                    continue
                seen.add(k)
                uniq_recent.append(txt)
                if limit is not None and len(uniq_recent) >= int(limit):
                    break
            return list(reversed(uniq_recent))
        except Exception:
            return []

    def _load_hard_rules(self, *, limit: Optional[int]) -> List[Dict[str, Any]]:
        """
        IMPORTANT:
        - We normalize + filter again on load to avoid "old bad rules" polluting future runs.
        - De-dup by normalized hard_rule_key.
        """
        if not getattr(self, "_persist_rules_enabled", True):
            return []
        try:
            raw = load_jsonl_unique(Path(self.hard_rules_path), key_func=hard_rule_key, limit=limit)
            out: List[Dict[str, Any]] = []
            seen = set()
            for r in raw or []:
                if not isinstance(r, dict):
                    continue
                rr = self._normalize_hard_rule(r)
                if rr is None:
                    continue
                k = hard_rule_key(rr)
                if k in seen:
                    continue
                seen.add(k)
                out.append(rr)
            return out
        except Exception:
            return []

    def _load_soft_heuristics(self, *, limit: Optional[int]) -> List[Dict[str, Any]]:
        # If rule persistence is disabled, do not load rules from disk
        if not getattr(self, "_persist_rules_enabled", True):
            return []
        try:
            return load_jsonl_unique(Path(self.soft_heuristics_path), key_func=soft_heuristic_key, limit=limit)
        except Exception:
            print(f"[KA ERROR] failed to load soft heuristics. Error: {Exception}")
            return []

    def _append_hard_rules(self, hard_rules: List[Dict[str, Any]]) -> None:
        if not getattr(self, "_persist_rules_enabled", True):
            return   # ABLATION: skip rule persistence

        # Evidence threshold: do not write hard rules when there is not enough data
        min_ev = getattr(self, "min_full_valid_for_hard_rules", 12)
        cur_fv = getattr(self, "_current_full_valid_count", 0)
        if cur_fv < min_ev:
            if hard_rules:
                print(f"[KA] defer {len(hard_rules)} hard_rules: "
                      f"only {cur_fv} full-valid (need {min_ev})")
            return

        items: List[Dict[str, Any]] = []
        for r in hard_rules or []:
            if not isinstance(r, dict):
                continue

            rr = self._normalize_hard_rule(r)
            if rr is None:
                print(f"[KA] drop hard_rule: {r}")
                continue

            items.append(rr)

        if not items:
            return

        # Check the rule-count limit before writing; evict the oldest rules if the budget is full
        max_active = getattr(self, "max_active_hard_rules", 8)
        try:
            existing = load_jsonl_unique(Path(self.hard_rules_path),
                                         key_func=hard_rule_key, limit=None) or []
        except Exception:
            existing = []

        budget = max(0, max_active - len(existing))
        if budget <= 0:
            # LRU-style eviction: remove the oldest rules to make room for new ones
            n_to_evict = min(len(items), max(1, max_active // 4))  # Evict at most 25% at a time
            if len(existing) > n_to_evict:
                surviving = existing[n_to_evict:]  # Keep the newer entries
                evicted_msgs = [r.get("message", "?")[:60] for r in existing[:n_to_evict]]
                print(f"[KA] hard_rules full ({len(existing)}/{max_active}), "
                      f"evicting {n_to_evict} oldest: {evicted_msgs}")
                # Rewrite the file and keep only the surviving entries
                try:
                    import json as _json
                    Path(self.hard_rules_path).write_text(
                        "\n".join(_json.dumps(r, ensure_ascii=False) for r in surviving) + "\n",
                        encoding="utf-8",
                    )
                    existing = surviving
                    budget = max(0, max_active - len(existing))
                except Exception as _e:
                    print(f"[KA] eviction write failed: {_e}")
                    budget = 0
            else:
                # If the existing count is <= n_to_evict, clear the file completely
                try:
                    Path(self.hard_rules_path).write_text("", encoding="utf-8")
                    existing = []
                    budget = max_active
                except Exception:
                    budget = 0

        if budget <= 0:
            print(f"[KA] hard_rules still full after eviction, skip")
            return
        items = items[:budget]

        try:
            append_jsonl_unique(Path(self.hard_rules_path), items, key_func=hard_rule_key)
            print(f"[KA] appended {len(items)} hard_rules (total ~{len(existing) + len(items)})")
        except Exception:
            return
        
    def _append_soft_heuristics(self, soft_heuristics: List[Dict[str, Any]]) -> None:
        if not getattr(self, "_persist_rules_enabled", True):
            return   # ABLATION: skip rule persistence
        items: List[Dict[str, Any]] = []
        for h in soft_heuristics or []:
            if not isinstance(h, dict):
                continue
            hh = {
                "trigger": str(h.get("trigger", "")).strip(),
                "preferences": h.get("preferences", []) or [],
                "notes": str(h.get("notes", "")).strip(),
            }
            if not hh["notes"]:
                continue
            items.append(hh)
        if not items:
            return

        # Maximum number of soft heuristics
        max_soft = getattr(self, "max_active_soft_heuristics", 30)
        try:
            existing = load_jsonl_unique(Path(self.soft_heuristics_path),
                                         key_func=soft_heuristic_key, limit=None) or []
        except Exception:
            existing = []
        budget = max(0, max_soft - len(existing))
        if budget <= 0:
            return
        items = items[:budget]

        try:
            append_jsonl_unique(Path(self.soft_heuristics_path), items, key_func=soft_heuristic_key)
        except Exception as e:
            print(f"[KA ERROR] failed to append soft heuristics. Error: {e}")
            return
    
    def trim_hard_rules(self, *, keep_latest: Optional[int] = None) -> int:
        """
        ★ NEW: Periodically trim excessive hard rules and keep only the latest N entries.
        Return the number of removed rules.
        """
        if not getattr(self, "_persist_rules_enabled", True):
            return 0
        if keep_latest is None:
            keep_latest = getattr(self, "max_active_hard_rules", 8)
        keep_latest = max(1, int(keep_latest))
        try:
            all_rules = load_jsonl_unique(Path(self.hard_rules_path),
                                          key_func=hard_rule_key, limit=None) or []
            if len(all_rules) <= keep_latest:
                return 0
            trimmed = all_rules[-keep_latest:]
            removed = len(all_rules) - len(trimmed)
            # Rewrite the file
            Path(self.hard_rules_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.hard_rules_path).write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in trimmed) + "\n",
                encoding="utf-8",
            )
            print(f"[KA] trimmed hard_rules: {len(all_rules)} -> {len(trimmed)} (removed {removed})")
            return removed
        except Exception as e:
            print(f"[KA] trim_hard_rules failed: {e}")
            return 0

    def trim_soft_heuristics(self, *, keep_latest: Optional[int] = None) -> int:
        """
        ★ NEW: Periodically trim excessive soft heuristics and keep only the latest N entries.
        """
        if not getattr(self, "_persist_rules_enabled", True):
            return 0
        if keep_latest is None:
            keep_latest = getattr(self, "max_active_soft_heuristics", 30)
        keep_latest = max(1, int(keep_latest))
        try:
            all_h = load_jsonl_unique(Path(self.soft_heuristics_path),
                                      key_func=soft_heuristic_key, limit=None) or []
            if len(all_h) <= keep_latest:
                return 0
            trimmed = all_h[-keep_latest:]
            removed = len(all_h) - len(trimmed)
            Path(self.soft_heuristics_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.soft_heuristics_path).write_text(
                "\n".join(json.dumps(h, ensure_ascii=False) for h in trimmed) + "\n",
                encoding="utf-8",
            )
            print(f"[KA] trimmed soft_heuristics: {len(all_h)} -> {len(trimmed)} (removed {removed})")
            return removed
        except Exception as e:
            print(f"[KA] trim_soft_heuristics failed: {e}")
            return 0

    def hard_rule_violations(self, params: Dict[str, Any], *, stage: str, fidelity: str, hard_rules: List[Dict[str, Any]]) -> List[str]:
        """
        Evaluate hard rules using params + context keys.
        Context keys supported in rules:
          - __stage__  : 'quick'/'full'/...
          - __fidelity__: 'quick'/'full'
          - __platform__: ORFS platform (if available)
        """
        if not getattr(self, "_persist_rules_enabled", True):
            return []   # ABLATION: no hard rule filtering
        ctx = {"__stage__": str(stage), "__fidelity__": str(fidelity)}
        if getattr(self, "platform", ""):
            ctx["__platform__"] = str(self.platform)
        merged = dict(params or {})
        merged.update(ctx)
        try:
            return eval_hard_rules(merged, hard_rules or [])
        except Exception:
            return []

    def get_soft_preferences(self) -> Dict[str, int]:
        """Expose soft heuristic preference signs for DA."""
        soft = self._load_soft_heuristics(limit=None)
        try:
            return soft_preferences_from_heuristics(soft)
        except Exception:
            return {}

    # ---------------------------
    # Prompting
    # ---------------------------
    def _system_prompt(self) -> str:
        # Load persistent knowledge (skip if persistence disabled)
        if getattr(self, "_persist_rules_enabled", True):
            hard = self._load_hard_rules(limit=self.max_hard_rules_in_prompt)
            soft = self._load_soft_heuristics(limit=self.max_soft_heuristics_in_prompt)
            legacy_soft = self._load_legacy_soft_rules_text(limit=self.max_soft_heuristics_in_prompt)
        else:
            hard = []
            soft = []
            legacy_soft = []

        hard_txt = ""
        if hard:
            hard_txt = (
                "\n\nPersistent HARD rules (strict constraints; candidates violating them will be filtered):\n"
                + "\n".join([f"- {render_hard_rule_text(r)}" for r in hard])
            )

        soft_lines: List[str] = []
        seen_soft = set()

        for h in soft:
            s = render_soft_heuristic_text(h)
            k = s.casefold()
            if k not in seen_soft:
                seen_soft.add(k)
                soft_lines.append(s)

        for s in legacy_soft:
            k = s.casefold()
            if k not in seen_soft:
                seen_soft.add(k)
                soft_lines.append(s)

        soft_txt = ""
        if soft_lines:
            soft_txt = (
                "\n\nPersistent SOFT heuristics (preferences; use them as priors, not hard constraints):\n"
                + "\n".join([f"- {s}" for s in soft_lines[: int(self.max_soft_heuristics_in_prompt)]])
            )

        # 3.2 Failure-pattern analysis
        # Truncate failure_analysis to 1500 characters
        failure_txt = ""
        if getattr(self, "failure_analysis_text", ""):
            _fa = str(self.failure_analysis_text).strip()
            if len(_fa) > 1500:
                _fa = _fa[:1500] + "\n... [truncated]"
            failure_txt = (
                "\n\n=== FAILURE PATTERN ANALYSIS ===\n"
                + _fa
                + "\n=== END FAILURE ANALYSIS ===\n"
            )

        # Truncate design_prior to 2000 characters
        design_txt = ""
        if getattr(self, "design_prior_text", ""):
            _dp = str(self.design_prior_text).strip()
            if len(_dp) > 2000:
                _dp = _dp[:2000] + "\n... [truncated]"
            design_txt = (
                "\n\n=== DESIGN PRIOR KNOWLEDGE ===\n"
                + _dp
                + "\n=== END DESIGN PRIOR ===\n"
            )

        feedback_txt = self._format_feedback_block()

        return (
            "You are KA (Knowledge Agent), an expert EDA/PD engineer for Cadence Genus+Innovus parameter tuning.\n"
            "\n"
            "=== YOUR DOMAIN EXPERTISE ===\n"
            "You understand:\n"
            "- Cadence Innovus place/CTS/route stages and how parameters affect each stage\n"
            "- UTIL controls core utilization (higher = denser = shorter wires but harder to route/close timing)\n"
            "- MARGIN controls floorplan margin (higher = more whitespace = easier routing but larger area)\n"
            "- TD_PLACE=1 enables timing-driven placement (critical for designs with long combinational paths)\n"
            "- RUN_CTS=1 enables clock tree synthesis (needed for realistic timing but adds runtime)\n"
            "- WNS is in the SAME unit as clock period (typically ps for advanced nodes like ASAP7)\n"
            "- Negative WNS = timing violation; WNS closer to 0 or positive = better\n"
            "- Area and Power should be minimized; WNS should be maximized (less negative)\n"
            "\n"
            "=== OBJECTIVE ===\n"
            "Multi-objective optimization: maximize WNS, minimize Area, minimize Power.\n"
            "You must output STRICT JSON matching the provided JSON Schema.\n"
            "\n"
            "\n"
            "=== STRATEGY EXECUTION GUIDE ===\n"
            "The strategy has been PRE-SELECTED by the system based on quantitative analysis.\n"
            "You MUST follow the MANDATORY STRATEGY in the user message.\n"
            "\n"
            "Strategy behaviors:\n"
            "- explore: sample diverse corners of param space, maximize coverage\n"
            "- exploit: refine around best-known Pareto points with small perturbations (±5-15% of range)\n"
            "- diversify: try parameter combinations far from anything in history, break out of local optima\n"
            "- repair: diagnose failures and propose safe configurations, avoid parameters that correlate with failure\n"
            "- promote: pick candidates with best quick-stage metrics for full evaluation\n"
            "\n"
            "=== CANDIDATE RULES ===\n"
            "1) Only use parameters from PARAM_SPACE.\n"
            "2) Respect type/range/choices strictly.\n"
            "3) Return candidates in PRIORITY ORDER (best first).\n"
            "4) Format: {params:{...}, notes:'why this config helps'}.\n"
            "5) Each candidate SHOULD differ meaningfully from others (avoid near-duplicates).\n"
            "6) Use DESIGN CONTEXT below to make informed decisions about parameter ranges.\n"
            "\n"
            "=== RULES OUTPUT ===\n"
            "hard_rules (0-3): strict ban/require learned from failures. Only output if confident.\n"
            "  Schema: {rule_type:'ban'|'require', when:[{param,op,value}], must:[{...}], message:'...'}\n"
            "soft_heuristics (0-5): directional preferences learned from history.\n"
            "  Schema: {trigger:'congestion|overflow|setup|hold|drc|power|runtime|crash|other',\n"
            "           preferences:[{param,direction:'increase|decrease|keep_low|keep_high|toggle'}],\n"
            "           notes:'...'}\n"
            "\n"
            "⚠ RULE EVIDENCE REQUIREMENT:\n"
            "  Derive rules ONLY from [full] fidelity results (post-route metrics).\n"
            "  [quick] results are placement/CTS-level approximations — their WNS, area,\n"
            "  and power are NOT reliable for drawing conclusions about parameter effects.\n"
            "  If no [full] results are available, output 0 hard_rules and 0 soft_heuristics.\n"
            + design_txt
            + failure_txt
            + hard_txt
            + soft_txt
            + ("\n" + feedback_txt if feedback_txt else "")
        )

    def _user_prompt(self, *, param_space, default_params, history, pareto,
                    remaining_sec, stage, k) -> str:
        ps = self._format_param_space(param_space, default_params)
        hist = self._format_history(history, max_rows=12, param_space=param_space)
        pareto_txt = self._format_history(pareto or [], max_rows=8, param_space=param_space, title="PARETO_SNAPSHOT")
        remain_txt = "unknown" if remaining_sec is None else ("%.1f" % float(remaining_sec))

        # Quantitative strategy selection
        strategy, diag = compute_strategy(
            history,
            [h for h in history if h.get("metrics_valid", False)],
        )
        strategy_directive = (
            f"MANDATORY STRATEGY: {strategy}\n"
            f"Reason: {diag.get('reason', 'unknown')}\n"
            f"Diagnostics: {json.dumps(diag, ensure_ascii=False)}\n"
            f"You MUST follow this strategy. Do NOT override it.\n"
        )

        # 3.3 Parameter sensitivity
        # Truncate to 1000 characters
        sens_txt = ""
        if getattr(self, "sensitivity_text", ""):
            _st = str(self.sensitivity_text).strip()
            if len(_st) > 1000:
                _st = _st[:1000] + "\n... [truncated]"
            sens_txt = "\n" + _st + "\n"

        # Tell the LLM which parameters are actually effective at the current stage
        stage_param_hint = ""
        if "quick" in str(stage).lower() or "place" in str(stage).lower():
            stage_param_hint = (
                "\n⚠ CRITICAL: Current stage is PLACEMENT ONLY (quick/place).\n"
                "The following parameters have NO EFFECT at placement stage and "
                "MUST NOT be your differentiation strategy:\n"
                "  - ROUTE_EFFORT, ROUTE_LAYER_EFFORT, ROUTE_SI_DRIVEN, "
                "ROUTE_TIMING_DRIVEN (routing not executed)\n"
                "  - DRV_EFFORT (DRV fixing not executed)\n"
                "\n"
                "Parameters that ACTUALLY MATTER at placement:\n"
                "  - CORE_UTIL, CORE_MARGIN, ASPECT_RATIO (floorplan geometry)\n"
                "  - PLACE_MAX_DENSITY, PLACE_CONG_EFFORT, PLACE_TIMING_EFFORT, "
                "PLACE_UNIFORM_DENSITY (placement quality)\n"
                "  - TD_PLACE, PLACE_IO_PINS_AWARE (placement mode)\n"
                "  - CTS_TARGET_SKEW, RUN_CTS, CTS_POST_OPT, "
                "PRECTS_SETUP_OPT, PRECTS_HOLD_OPT (CTS, if CTS runs in quick)\n"
                "\n"
                "Focus your perturbations on placement-effective parameters ONLY.\n"
                "If you have already exhausted the placement-effective parameter space, "
                "say so in your rationale and suggest promoting existing configs to full.\n"
            )

        # EHVI promising-region guidance
        region_txt = ""
        regions = getattr(self, "ehvi_promising_regions", [])
        if regions:
            region_lines = ["EHVI-PROMISING REGIONS (BO model predicts highest improvement here):"]
            for i, r in enumerate(regions[:3]):
                p = r.get("params", {})
                score = r.get("ehvi_score", 0)
                region_lines.append(
                    f"  Region #{i+1} (EHVI={score:.4f}): {json.dumps(p, ensure_ascii=False)}"
                )
            region_lines.append(
                "Use these as ANCHOR POINTS: at least 1-2 of your candidates should be "
                "near these regions (within ±10% of each parameter). "
                "Other candidates can explore elsewhere for diversity."
            )
            region_txt = "\n".join(region_lines) + "\n"
        
        return (
            "TASK:\n"
            "Propose %d candidate configs for next trials.\n"
            "\n"
            "%s\n"
            "%s\n"
            "\n"
            "CONTEXT:\n"
            "- stage: %s\n"
            "- remaining_sec: %s\n"
            "\n"
            "PARAM_SPACE:\n%s\n"
            "\n"
            "DEFAULT_PARAMS (baseline):\n%s\n"
            "\n"
            "RECENT_HISTORY (newest last):\n%s\n"
            "\n"
            "%s\n"
            "\n"
            "%s\n"         # sens_txt
            "\n"
            "%s\n"         # region_txt  <- additional placeholder
            "\n"
            "OUTPUT:\n"
            "- Use the MANDATORY STRATEGY above (do NOT choose a different one).\n"
            "- Provide rationale referencing the diagnostics.\n"
            "- Return %d candidates.\n"
            % (
                k,
                strategy_directive,
                stage_param_hint,
                stage,
                remain_txt,
                ps,
                json.dumps(default_params, ensure_ascii=False),
                hist,
                pareto_txt,
                sens_txt,
                region_txt,          # passed separately
                k,
            )
        )



    # ---------------------------
    # JSON schema (structured outputs)
    # ---------------------------
    def _build_json_schema(self, max_k: int) -> Dict[str, Any]:
        cond_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["param", "op", "value"],
            "properties": {
                "param": {"type": "string"},
                "op": {"type": "string", "enum": ["==", "!=", "<", "<=", ">", ">=", "in", "not_in"]},
                "value": {
                    "type": ["number", "string", "array"],
                    "items": {"type": ["number", "string"]},
                },
            },
        }

        hard_rule_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["rule_type", "when", "must", "message"],
            "properties": {
                "rule_type": {"type": "string", "enum": ["ban", "require"]},
                "when": {"type": "array", "minItems": 1, "maxItems": 4, "items": cond_schema},
                "must": {"type": "array", "minItems": 0, "maxItems": 4, "items": cond_schema},
                "message": {"type": "string"},
            },
        }

        pref_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["param", "direction"],
            "properties": {
                "param": {"type": "string"},
                "direction": {"type": "string", "enum": ["increase", "decrease", "keep_low", "keep_high", "toggle"]},
                "strength": {"type": "string", "enum": ["weak", "medium", "strong"]},
            },
        }

        soft_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["trigger", "preferences", "notes"],
            "properties": {
                "trigger": {"type": "string"},
                "preferences": {"type": "array", "minItems": 0, "maxItems": 4, "items": pref_schema},
                "notes": {"type": "string"},
            },
        }

        cand_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["params", "notes"],
            "properties": {
                "params": {"type": "object", "additionalProperties": True},
                "notes": {"type": "string"},
            },
        }

        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["strategy", "rationale", "hard_rules", "soft_heuristics", "candidates"],
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["explore", "exploit", "diversify", "promote", "repair", "fallback_random"],
                },
                "rationale": {"type": "string"},
                "hard_rules": {"type": "array", "minItems": 0, "maxItems": 3, "items": hard_rule_schema},
                "soft_heuristics": {"type": "array", "minItems": 0, "maxItems": 5, "items": soft_schema},
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": int(max_k),
                    "items": cand_schema,
                },
            },
        }

    # ---------------------------
    # Validation / clamp
    # ---------------------------
    def _validate_and_fix_candidate(
        self,
        *,
        item: Dict[str, Any],
        param_space: List[Dict[str, Any]],
        default_params: Dict[str, Any],
        stage: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(item, dict):
            return False, {}

        param_names = {
            str(p.get("name", "")).strip()
            for p in param_space
            if isinstance(p, dict) and str(p.get("name", "")).strip()
        }

        # Accept both formats:
        # 1) nested: {"params": {...}, "notes": "..."}  <-- NEW (schema enforced)
        # 2) flat:   {"place_density": 0.52, ..., "notes": "..."}  <-- compatibility
        params_in: Dict[str, Any] = {}

        if isinstance(item.get("params"), dict):
            for k, v in (item.get("params") or {}).items():
                if k in param_names:
                    params_in[k] = v

        if not params_in:
            params_in = {k: v for k, v in item.items() if k in param_names}

        if not params_in:
            return False, {}

        fixed_params = self._fix_params(params_in, param_space, default_params)

        fidelity = "quick" if "quick" in str(stage).lower() else "full"

        out = {
            "params": fixed_params,
            "fidelity": fidelity,
            "tcl_overrides": [],
            "parent_id": None,
            "notes": str(item.get("notes") or item.get("note") or "").strip()[:500],
            "proposed_by": self.agent_name,
            "proposed_strategy": str(item.get("strategy", "")).strip(),
            "stage_hint": stage,
        }

        if not out["params"]:
            return False, {}

        if not out["notes"]:
            out["notes"] = "KA proposal."

        return True, out

    def _fix_params(
        self,
        params_in: Dict[str, Any],
        param_space: List[Dict[str, Any]],
        default_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enforce: only allow param_space keys, cast types, clamp to range, snap to choices.
        Fill missing keys with defaults ONLY if defaults provide them; otherwise keep sparse.
        """
        spec_by_name: Dict[str, Dict[str, Any]] = {}
        for p in param_space:
            name = str(p.get("name", "")).strip()
            if name:
                spec_by_name[name] = p

        fixed: Dict[str, Any] = {}

        for name, val in params_in.items():
            if name not in spec_by_name:
                continue
            spec = spec_by_name[name]
            v = self._cast_and_clamp(val, spec)
            if v is not None:
                fixed[name] = v

        for name, spec in spec_by_name.items():
            if name in fixed:
                continue
            if name in default_params:
                v = self._cast_and_clamp(default_params.get(name), spec)
                if v is not None:
                    fixed[name] = v

        return fixed

    def _cast_and_clamp(self, val: Any, spec: Dict[str, Any]) -> Optional[Any]:
        typ = str(spec.get("type", "")).lower()
        if "choices" in spec and spec["choices"] is not None:
            choices = list(spec["choices"])
            if not choices:
                return None
            try:
                if typ == "int":
                    x = int(float(val))
                elif typ == "float":
                    x = float(val)
                else:
                    x = str(val)
            except Exception:
                x = choices[0]

            if isinstance(choices[0], (int, float)):
                best = choices[0]
                best_d = None
                for c in choices:
                    d = abs(float(c) - float(x))
                    if best_d is None or d < best_d:
                        best_d = d
                        best = c
                return int(best) if typ == "int" else float(best)
            else:
                sx = str(x)
                return sx if sx in choices else str(choices[0])

        if "range" in spec and spec["range"] is not None:
            r = spec["range"]
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                return None
            lo, hi = float(r[0]), float(r[1])
            if lo > hi:
                lo, hi = hi, lo
            try:
                x = float(val)
            except Exception:
                return None
            x = max(lo, min(hi, x))
            if typ == "int":
                return int(round(x))
            return float(x)

        try:
            if typ == "int":
                return int(float(val))
            if typ == "float":
                return float(val)
        except Exception:
            return None
        return val

    # ---------------------------
    # Formatting helpers
    # ---------------------------
    def _format_param_space(self, param_space: List[Dict[str, Any]], default_params: Dict[str, Any]) -> str:
        lines: List[str] = []
        for p in param_space:
            name = str(p.get("name", "")).strip()
            typ = str(p.get("type", "")).strip()
            if not name:
                continue
            s = "- %s: %s" % (name, typ)
            if "choices" in p and p["choices"] is not None:
                s += " choices=%s" % (p["choices"],)
            if "range" in p and p["range"] is not None:
                s += " range=%s" % (p["range"],)
            if name in default_params:
                s += " default=%s" % (default_params.get(name),)
            lines.append(s)
        return "\n".join(lines)

    def _format_history(
        self,
        history: List[Dict[str, Any]],
        *,
        max_rows: int,
        param_space: List[Dict[str, Any]],
        title: str = "HISTORY",
    ) -> str:
        """
        Compact the history format.
        - Successful runs: keep only run_id, metrics, and params (no log_summary).
        - Failed runs: show symptoms + brief only for the most recent 3 entries (no fail_log_tail).
        - Earlier failed runs: keep a single line with run_id + symptoms.
        This typically saves about 2000-4000 input tokens per call.
        """
        if not history:
            return "%s: (empty)" % title

        names = [str(p.get("name")) for p in param_space if p.get("name")]
        rows = history[-max_rows:]
        out_lines: List[str] = ["%s:" % title]

        # Track recent failed runs and show details only for the most recent 3
        fail_indices = [i for i, h in enumerate(rows) if not h.get("success", h.get("ok", True))]
        recent_fail_detail_set = set(fail_indices[-3:]) if fail_indices else set()

        for idx, h in enumerate(rows):
            if not isinstance(h, dict):
                continue
            run_id = h.get("run_id", "")
            fidelity = h.get("fidelity", "")
            success = h.get("success", h.get("ok", None))
            rt = h.get("runtime_sec", None)
            m = h.get("metrics", {}) or {}
            params = h.get("params", {}) or {}

            # Keep only parameters defined in param_space
            p2 = {}
            if isinstance(params, dict):
                for n in names:
                    if n in params:
                        p2[n] = params[n]

            if success:
                # Successful run: compact format without log_summary
                out_lines.append(
                    "  - %s [%s] ok %.0fs metrics=%s params=%s"
                    % (run_id, fidelity, float(rt) if rt else 0, self._safe_json(m), self._safe_json(p2))
                )
            elif idx in recent_fail_detail_set:
                # Most recent 3 failed runs: show symptoms + brief without fail_log_tail
                ls = h.get("log_summary", {}) or {}
                syms = ls.get("symptoms", []) if isinstance(ls, dict) else []
                brief = (ls.get("brief", "") if isinstance(ls, dict) else "")[:120]
                out_lines.append(
                    "  - %s [%s] FAIL %.0fs metrics=%s params=%s symptoms=%s brief=%s"
                    % (run_id, fidelity, float(rt) if rt else 0, self._safe_json(m),
                       self._safe_json(p2), self._safe_json(syms), brief)
                )
            else:
                # Earlier failed runs: one-line summary
                ls = h.get("log_summary", {}) or {}
                syms = ls.get("symptoms", []) if isinstance(ls, dict) else []
                out_lines.append(
                    "  - %s [%s] FAIL symptoms=%s"
                    % (run_id, fidelity, self._safe_json(syms))
                )

        return "\n".join(out_lines)

    def _safe_json(self, obj: Any) -> str:
        try:
            return str(obj) if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    # ---------------------------
    # Fallback random
    # ---------------------------
    def _random_candidate(self, param_space, default_params, *, stage):
        # If KA is disabled or rule persistence is disabled, do not load rules
        if not self.is_ready() or not getattr(self, "_persist_rules_enabled", True):
            hard_all = []
        else:
            hard_all = self._load_hard_rules(limit=None)

        def _sample_once() -> Dict[str, Any]:
            params: Dict[str, Any] = {}
            for p in param_space:
                name = str(p.get("name", "")).strip()
                if not name:
                    continue
                typ = str(p.get("type", "")).lower()
                if "choices" in p and p["choices"] is not None:
                    choices = list(p["choices"])
                    if choices:
                        params[name] = random.choice(choices)
                    continue
                if "range" in p and p["range"] is not None:
                    lo, hi = float(p["range"][0]), float(p["range"][1])
                    if typ == "int":
                        params[name] = int(random.randint(int(round(lo)), int(round(hi))))
                    else:
                        params[name] = float(lo + (hi - lo) * random.random())
                    continue
                if name in default_params:
                    params[name] = default_params[name]
            return self._fix_params(params, param_space, default_params)

        picked = None
        for _ in range(60):
            p = _sample_once()
            viol = self.hard_rule_violations(
                p,
                stage=stage,
                fidelity=("quick" if stage != "full" else "full"),
                hard_rules=hard_all,
            )
            if not viol:
                picked = p
                break
        if picked is None:
            picked = _sample_once()

        return {
            "params": picked,
            "fidelity": "quick" if stage != "full" else "full",
            "tcl_overrides": [],
            "parent_id": None,
            "notes": "Top-up random candidate (LLM insufficient/duplicate).",
            "proposed_by": self.agent_name,
            "proposed_strategy": "fallback_random",
            "proposed_model": self.generation_model,
            "stage_hint": stage,
        }

    def score_candidate_with_soft_rules(
        self,
        candidate_params: Dict[str, Any],
        soft_heuristics: List[Dict[str, Any]],
    ) -> float:
        """
        Score candidates based on soft heuristics.
        Each matched heuristic contributes +/- confidence points.
        
        Returns: bonus score (can be positive or negative)
        """
        bonus = 0.0
        for rule in soft_heuristics:
            confidence = float(rule.get("confidence", 0.5))
            prefs = rule.get("preferences", [])
            if not prefs:
                continue
            
            matches = 0
            for pref in prefs:
                pname = pref.get("param", "")
                direction = pref.get("direction", "")
                val = candidate_params.get(pname)
                if val is None:
                    continue
                
                # Check whether the candidate matches the preferred direction
                # Simple heuristic: if direction=increase and the value is in the upper half, treat it as a match
                ps = next((p for p in self.param_space_cache if p.get("name") == pname), None)
                if ps and "range" in ps:
                    lo, hi = float(ps["range"][0]), float(ps["range"][1])
                    mid = (lo + hi) / 2.0
                    try:
                        v = float(val)
                        if direction == "increase" and v > mid:
                            matches += 1
                        elif direction == "decrease" and v < mid:
                            matches += 1
                        elif direction in ("keep_low",) and v < mid:
                            matches += 1
                        elif direction in ("keep_high",) and v > mid:
                            matches += 1
                    except (ValueError, TypeError):
                        pass

            if matches > 0:
                bonus += confidence * matches / max(len(prefs), 1)

        return bonus

    def rerank_candidates_with_rules(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not getattr(self, "_persist_rules_enabled", True):
            return candidates

        soft = self._load_soft_heuristics(limit=None)
        if not soft:
            return candidates

        # Defensive check: skip reranking if param_space_cache is empty
        if not getattr(self, "param_space_cache", None):
            return candidates

        scored = []
        for i, c in enumerate(candidates):
            params = c.get("params", {})
            bonus = self.score_candidate_with_soft_rules(params, soft)
            base_score = len(candidates) - i
            scored.append((base_score + bonus * 2.0, c))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]


    def compute_adaptive_temperature(
        self,
        history: List[Dict[str, Any]],
        base_temperature: float = 0.6,
        stagnation_window: int = 10,
    ) -> float:
        """
        Adjust the LLM temperature dynamically based on the search state.
        Stagnation -> raise the temperature (max 1.0)
        Improvement -> lower the temperature (min 0.15)
        """
        hv_values = [
            h.get("cum_hv", 0.0) for h in history
            if h.get("cum_hv") is not None
        ]

        if len(hv_values) < stagnation_window:
            return base_temperature

        recent = hv_values[-stagnation_window:]
        hv_range = max(recent) - min(recent)
        hv_max = max(abs(v) for v in recent) if recent else 1.0
        relative_change = hv_range / max(hv_max, 1e-10)

        if relative_change < 0.005:
            # Severe stagnation -> high temperature
            return min(1.0, base_temperature + 0.3)
        elif relative_change < 0.02:
            # Mild stagnation -> slightly raise the temperature
            return min(0.9, base_temperature + 0.15)
        elif relative_change > 0.05:
            # Active improvement -> lower the temperature
            return max(0.15, base_temperature - 0.2)
        else:
            return base_temperature
