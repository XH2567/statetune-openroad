# main.py
from __future__ import annotations

import json
import os
import random
import time
import math
import gc
import torch

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Tuple
# from botorch.utils.multi_objective.hypervolume import Hypervolume


from src.EDATuner.utils.utils import load_config
# from src.EDATuner.runners.openroad_runner import OpenRoadFlow
from src.EDATuner.agents.mo_ehvi_agent import MOEHVIAgent
from src.EDATuner.knowledge.kb import KB
from src.EDATuner.optimization.runtime_model import MultiFidelityRuntimePredictor, IterationBudgetEstimator
from src.EDATuner.agents.ka_agent import KnowledgeAgent
from src.EDATuner.knowledge.log_summary import analyze_log_summary
from src.EDATuner.runners.eda_interface import EDARunner
from src.EDATuner.runners.runner_factory import create_runner
from src.EDATuner.knowledge.design_prior import DesignPriorBuilder
from src.EDATuner.knowledge.rag_service import get_rag_service
from src.EDATuner.knowledge.failure_analyzer import (
    analyze_failure_patterns,
    format_failure_analysis_for_prompt,
)
from src.EDATuner.knowledge.sensitivity import (
    compute_parameter_sensitivity,
    format_sensitivity_for_prompt,
    detect_threshold_effects, 
    format_threshold_effects_for_prompt
)

_MOEHVIAgent = None
_Hypervolume = None

def _get_ehvi_imports():
    """Lazily import MOEHVIAgent and Hypervolume on first use."""
    global _MOEHVIAgent, _Hypervolume
    if _MOEHVIAgent is None:
        from src.EDATuner.agents.mo_ehvi_agent import MOEHVIAgent
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        _MOEHVIAgent = MOEHVIAgent
        _Hypervolume = Hypervolume
    return _MOEHVIAgent, _Hypervolume

# --- Resilience imports ---
try:
    from src.EDATuner.runners.remote_cadence_support import (
        RemoteCommError, RemoteSessionLostError,
        RemoteCommandTimeoutError, SchedulerBackpressureError,
        ControlledStop, is_backpressure_active,
        slugify,
    )

    _FATAL_REMOTE_EXCEPTIONS = (
        RemoteCommError, RemoteSessionLostError,
        SchedulerBackpressureError, ControlledStop,
    )
except ImportError:
    _FATAL_REMOTE_EXCEPTIONS = ()
    is_backpressure_active = lambda: False
    slugify = lambda s: str(s).strip().replace(" ", "_").lower()


from src.EDATuner.llm.token_tracker import token_tracker


_PLACEMENT_EFFECTIVE_PARAMS = {
    "CORE_UTIL", "ASPECT_RATIO", "CORE_MARGIN",
    "PLACE_CONG_EFFORT", "PLACE_TIMING_EFFORT",
    "PLACE_UNIFORM_DENSITY", "PLACE_IO_PINS_AWARE",
    "TD_PLACE", "PLACE_MAX_DENSITY",
    "PRECTS_SETUP_OPT", "PRECTS_HOLD_OPT",
    "RUN_CTS", "CTS_TARGET_SKEW", "CTS_POST_OPT",
}

def _needs_direct_full(params: dict, quick_target: str) -> bool:
    """
    Return whether this configuration should skip quick and go directly to full.
    When quick_target=cts and RUN_CTS=0, the quick CTS stage is effectively a no-op,
    so the metrics remain placement-level and add little value for GP/EHVI.
    Going straight to full saves quick budget and still explores the RUN_CTS=0 subspace.
    """
    if quick_target in ("cts",):
        run_cts = str(params.get("RUN_CTS", "1")).lower()
        if run_cts in ("0", "false"):
            return True
    return False

def _fingerprint_for_stage(params: Dict[str, Any], fidelity: str) -> str:
    """
    For the quick stage, fingerprint only placement-effective parameters,
    so configurations that differ only in routing knobs are not treated as new points.
    Use the full fingerprint for the full stage.
    """
    if fidelity == "quick":
        filtered = {k: v for k, v in params.items() if k in _PLACEMENT_EFFECTIVE_PARAMS}
        return json.dumps(filtered, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _fingerprint(params)

def _fingerprint(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _default_params_to_friendly(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dp = cfg.get("default_params", {}) or {}
    if not isinstance(dp, dict):
        return {}

    mapping = {
        "SYNTH_FLATTEN": "synth_flatten",
        "GPL_ROUTABILITY_DRIVEN": "gp_routability_driven",
        "CELL_PAD_IN_SITES_GLOBAL_PLACEMENT": "place_padding",
        "GPL_OVERFLOW": "gp_overflow",
        "SYNTH_MAX_FANOUT": "synth_max_fanout",
        "CTS_CLUSTER_SIZE": "cts_cluster_size",
        "GR_OVERFLOW_ITER": "gr_overflow_iter",
        "GPL_TIMING_DRIVEN": "gp_timing_driven",
        "SYNTH_RETIME": "synth_retime",
    }

    out: Dict[str, Any] = {}
    for k, v in dp.items():
        kk = str(k).strip()
        if kk in mapping:
            out[mapping[kk]] = v
        else:
            # keep unknown ORFS vars as-is
            out[kk] = v
    return out


def _get_param_space(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(cfg.get("param_space", None), list):
        return cfg["param_space"]
    tuning = cfg.get("tuning", {}) or {}
    if isinstance(tuning.get("param_space", None), list):
        return tuning["param_space"]

    # fallback demo param space
    return [
        {"name": "synth_flatten", "type": "int", "choices": [0, 1]},
        {"name": "place_density", "type": "float", "range": [0.30, 0.70]},
        {"name": "place_padding", "type": "int", "range": [0, 1]},
        {"name": "gp_routability_driven", "type": "int", "choices": [0, 1]},
        {"name": "gp_overflow", "type": "float", "range": [0.1, 0.9]},
    ]


def _compute_hv(history_full_valid: list, ref_point_list=None) -> float:
    """
    Compute the hypervolume of the current full-valid history.
    Objective direction: maximize [-area, wns, -power].
    FIX: ref_point must be provided externally as a global fixed value.
    """
    pts = []
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    for r in history_full_valid:
        m = r.get("metrics", {}) or {}
        try:
            a = float(m["area"])
            w = float(m["wns"])
            p = float(m["power"])
        except (KeyError, TypeError, ValueError):
            continue
        import math
        if not (math.isfinite(a) and math.isfinite(w) and math.isfinite(p)):
            continue
        if w == -100.0 or (not math.isfinite(w)):
            continue
        pts.append([-a, w, -p])

    if len(pts) < 1:
        return 0.0

    Y = torch.tensor(pts, dtype=torch.double)

    if ref_point_list is not None:
        ref = torch.tensor(ref_point_list, dtype=torch.double)
    else:
        # Fixed fallback reference point (no longer adaptive).
        ref = torch.tensor([-5e6, -10.0, -5.0], dtype=torch.double)

    try:
        from botorch.utils.multi_objective.pareto import is_non_dominated
        nd_mask = is_non_dominated(Y)
        pareto_Y = Y[nd_mask]
        if pareto_Y.shape[0] == 0:
            return 0.0
        hv = float(Hypervolume(ref).compute(pareto_Y))
        return hv
    except Exception:
        return 0.0


def compute_global_ref_point(history_full_valid: list) -> list:
    """Compute a fixed reference point from all valid points before the main loop starts."""
    pts = []
    for r in history_full_valid:
        m = r.get("metrics", {}) or {}
        try:
            a = float(m["area"])
            w = float(m["wns"])
            p = float(m["power"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(a) and math.isfinite(w) and math.isfinite(p)):
            continue
        if w == -100.0 or (not math.isfinite(w)):
            continue
        pts.append([-a, w, -p])
    if len(pts) < 1:
        return [-5e6, -1e5, -5e3]
    Y = torch.tensor(pts, dtype=torch.double)
    y_min = Y.min(dim=0).values
    slack = 0.1 * Y.abs().mean(dim=0).clamp(min=1.0)
    ref = (y_min - slack).tolist()
    return ref

def _classify_failure(res: dict) -> str:
    """
    Classify a failure into a discrete category for visualization.
    Returns: "ok" | "crash" | "timeout" | "metrics_invalid" | "drc" |
          "overflow" | "setup" | "infra" | "other_fail"
    FIX: Add an "infra" category for network/SSH/LSF infrastructure failures.
    """
    if res.get("success") and res.get("metrics_valid"):
        return "ok"
    if res.get("success") and res.get("metrics_usable"):
        return "ok_partial"

    # NEW: Detect infrastructure errors first (they are not parameter-related).
    error_text = str(res.get("error", "")).lower()
    fail_tail = str(res.get("fail_log_tail", "")).lower()
    _infra_combined = error_text + " " + fail_tail
    _INFRA_MARKERS = (
        "ssh", "paramiko", "connection refused", "connection reset",
        "socket is closed", "session not active", "transport is not active",
        "remotecommerror", "remotesessionlosterror", "schedulerbackpressure",
        "controlledstop", "lsf", "channel closed",
        "winerror 10051", "winerror 10054",
        "no route to host", "network is unreachable",
    )
    if any(m in _infra_combined for m in _INFRA_MARKERS):
        return "infra"

    # Extract signals from log_summary.symptoms.
    ls = res.get("log_summary", {}) or {}
    symptoms = ls.get("symptoms", []) if isinstance(ls, dict) else []
    if not isinstance(symptoms, list):
        symptoms = [str(symptoms)]
    sym_str = ",".join([str(s).lower() for s in symptoms])

    if "crash" in sym_str:
        return "crash"
    if "timeout" in sym_str:
        return "timeout"
    if "drc" in sym_str:
        return "drc"
    if "overflow" in sym_str:
        return "overflow"
    if "setup" in sym_str:
        return "setup"

    # Fall back to metrics-based classification.
    ms = str(res.get("metrics_status", "")).lower()
    if ms == "invalid":
        return "metrics_invalid"

    if not res.get("success"):
        return "other_fail"

    return "ok"


def _extract_error_warning_tail(log_path: str, *, max_lines: int = 10, max_bytes: int = 200_000) -> str:
    """
    Extract the last N lines containing 'Error' or 'Warning' from a log file.
    Best-effort: reads only the tail part of the file.
    """
    try:
        p = Path(str(log_path))
        if not p.exists() or not p.is_file():
            return ""
        size = p.stat().st_size
        # Read tail bytes (up to max_bytes)
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        txt = data.decode("utf-8", errors="ignore")
        lines = txt.splitlines()
        hit = []
        for ln in lines:
            if ("error" in ln.lower()) or ("warning" in ln.lower()):
                hit.append(ln)
        if not hit:
            return ""
        tail = hit[-int(max_lines):]
        # Keep it compact
        return "\n".join(tail).strip()
    except Exception:
        return ""


def generate_llm_report(
    *,
    ka: "KnowledgeAgent",
    cfg: Dict[str, Any],
    pareto_json_path: str,
    history_all: List[Dict[str, Any]],
    history_full_valid: List[Dict[str, Any]],
) -> str:
    """
    Generate a human-friendly Markdown report (final_report.md) by calling LLM.
    Returns the report path.
    """
    work_dir = Path(cfg.get("work_dir", ".")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "final_report.md"

    def _safe_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _short_text(x: Any, max_len: int = 1200) -> str:
        s = str(x or "").strip()
        if len(s) <= max_len:
            return s
        return s[:max_len] + "\n... [truncated]"

    # ------------------------------------------------------------------
    # Load pareto.json
    # Supports BOTH:
    #   1) {"ts": ..., "num_pareto": ..., "pareto": [...]}
    #   2) [...]
    # ------------------------------------------------------------------
    pareto: List[Dict[str, Any]] = []
    pareto_meta: Dict[str, Any] = {}
    pareto_load_error: str = ""

    try:
        payload = json.loads(Path(pareto_json_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            pareto_meta = dict(payload)
            raw_pareto = payload.get("pareto", [])
            if isinstance(raw_pareto, list):
                pareto = [p for p in raw_pareto if isinstance(p, dict)]
            else:
                pareto = []
        elif isinstance(payload, list):
            pareto = [p for p in payload if isinstance(p, dict)]
        else:
            pareto = []
    except Exception as e:
        pareto = []
        pareto_load_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Normalize history
    # ------------------------------------------------------------------
    records = [h for h in history_all if isinstance(h, dict)]

    def _is_job_success(h: Dict[str, Any]) -> bool:
        return bool(h.get("success", False))

    def _is_usable(h: Dict[str, Any]) -> bool:
        return bool(h.get("metrics_usable", False))

    def _is_full_valid(h: Dict[str, Any]) -> bool:
        if bool(h.get("metrics_valid", False)):
            return True
        m = h.get("metrics", {}) or {}
        area = _safe_float(m.get("area"), float("inf"))
        wns = _safe_float(m.get("wns"), -100.0)
        power = _safe_float(m.get("power"), float("inf"))
        return math.isfinite(area) and math.isfinite(wns) and math.isfinite(power) and (wns > -99.0)

    job_succ = [h for h in records if _is_job_success(h)]
    job_fail = [h for h in records if not _is_job_success(h)]

    usable = [h for h in records if _is_usable(h)]
    unusable = [h for h in records if not _is_usable(h)]

    # Prefer the caller-provided full-valid list; fall back to recomputing if needed
    full_valid = [h for h in history_full_valid if isinstance(h, dict)]
    if not full_valid:
        full_valid = [h for h in records if _is_full_valid(h)]

    # ------------------------------------------------------------------
    # Best points: compute from USABLE points, not only job_success points
    # ------------------------------------------------------------------
    def _best_by(pool: List[Dict[str, Any]], key: str, *, higher: bool) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_v: Optional[float] = None

        for h in pool:
            m = h.get("metrics", {}) or {}
            v = _safe_float(m.get(key), float("nan"))
            if not math.isfinite(v):
                continue

            if best is None:
                best = h
                best_v = v
                continue

            if best_v is None:
                best = h
                best_v = v
                continue

            if (higher and v > best_v) or ((not higher) and v < best_v):
                best = h
                best_v = v

        return best

    best_wns = _best_by(usable, "wns", higher=True)
    best_area = _best_by(usable, "area", higher=False)
    best_power = _best_by(usable, "power", higher=False)

    # ------------------------------------------------------------------
    # Small summaries for failure analysis
    # ------------------------------------------------------------------
    stage_counter: Dict[str, int] = {}
    fidelity_counter: Dict[str, int] = {}
    metrics_status_counter: Dict[str, int] = {}

    for h in records:
        st = str(h.get("stage", "unknown") or "unknown")
        fd = str(h.get("fidelity", "unknown") or "unknown")
        ms = str(h.get("metrics_status", "unknown") or "unknown")

        stage_counter[st] = stage_counter.get(st, 0) + 1
        fidelity_counter[fd] = fidelity_counter.get(fd, 0) + 1
        metrics_status_counter[ms] = metrics_status_counter.get(ms, 0) + 1

    # Cap payload sizes for LLM
    pareto_cap = pareto[:50]
    usable_cap = usable[-80:]
    unusable_cap = unusable[-40:]
    full_valid_cap = full_valid[-50:]

    summary = {
        "project_name": cfg.get("project_name", ""),
        "work_dir": str(work_dir),
        "counts": {
            "all": len(records),
            "job_success": len(job_succ),
            "job_fail": len(job_fail),
            "usable": len(usable),
            "unusable": len(unusable),
            "full_valid": len(full_valid),
            "pareto_size": len(pareto),
        },
        "distributions": {
            "stage": stage_counter,
            "fidelity": fidelity_counter,
            "metrics_status": metrics_status_counter,
        },
        "pareto_meta": {
            "num_full_valid": pareto_meta.get("num_full_valid"),
            "num_pareto": pareto_meta.get("num_pareto"),
            "load_error": pareto_load_error or None,
        },
        "best": {
            "best_wns": {
                "run_id": best_wns.get("run_id") if best_wns else None,
                "metrics": best_wns.get("metrics") if best_wns else None,
                "params": best_wns.get("params") if best_wns else None,
            },
            "best_area": {
                "run_id": best_area.get("run_id") if best_area else None,
                "metrics": best_area.get("metrics") if best_area else None,
                "params": best_area.get("params") if best_area else None,
            },
            "best_power": {
                "run_id": best_power.get("run_id") if best_power else None,
                "metrics": best_power.get("metrics") if best_power else None,
                "params": best_power.get("params") if best_power else None,
            },
        },
    }

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    sys_msg = (
        "You are an expert EDA/PD optimization analyst. "
        "You are analyzing a multi-objective design-space exploration (DSE) run. "
        "Objectives: maximize WNS, minimize Area and Power. "
        "Return ONLY Markdown, no code fences. "
        "Important interpretation rules: "
        "job_success means the full tool flow returned success; "
        "usable means metrics are still usable for analysis even if the job failed; "
        "full_valid means the point has complete valid metrics for Pareto-style comparison. "
        "Do not confuse job failure with lack of usable metrics. "
        "Your tasks: "
        "1) Summarize the real search quality and data quality separately. "
        "2) Identify parameters that appear to affect WNS/Area/Power. "
        "3) Explain trade-offs on the Pareto frontier. "
        "4) Diagnose common failure modes using failure snippets. "
        "5) Give 5-10 concrete next-step recommendations. "
        "Use this exact structure: "
        "# Final DSE Report\n"
        "## Executive Summary\n"
        "## Search Quality and Data Quality\n"
        "## Pareto Frontier Insights\n"
        "## Parameter Sensitivity\n"
        "## Failure Analysis\n"
        "## Recommended Next Experiments\n"
        "## Appendix: Top Pareto Points\n"
    )

    design_prior = getattr(ka, "design_prior_text", "")
    if isinstance(design_prior, str) and design_prior.strip():
        sys_msg += "\nDESIGN CONTEXT:\n" + design_prior.strip() + "\n"

    user_payload = {
        "summary": summary,
        "pareto_points": pareto_cap,
        "recent_usable_history": [
            {
                "run_id": h.get("run_id"),
                "fidelity": h.get("fidelity"),
                "stage": h.get("stage"),
                "params": h.get("params"),
                "metrics": h.get("metrics"),
                "runtime_sec": h.get("runtime_sec"),
                "success": h.get("success"),
                "metrics_status": h.get("metrics_status"),
                "metrics_valid": h.get("metrics_valid"),
                "log_summary": h.get("log_summary", {}),
            }
            for h in usable_cap
        ],
        "recent_unusable_history": [
            {
                "run_id": h.get("run_id"),
                "fidelity": h.get("fidelity"),
                "stage": h.get("stage"),
                "params": h.get("params"),
                "metrics": h.get("metrics"),
                "runtime_sec": h.get("runtime_sec"),
                "success": h.get("success"),
                "metrics_status": h.get("metrics_status"),
                "log_summary": h.get("log_summary", {}),
                "fail_log_tail": _short_text(h.get("fail_log_tail") or h.get("error_tail") or h.get("error") or ""),
            }
            for h in unusable_cap
        ],
        "recent_full_valid_history": [
            {
                "run_id": h.get("run_id"),
                "fidelity": h.get("fidelity"),
                "stage": h.get("stage"),
                "params": h.get("params"),
                "metrics": h.get("metrics"),
                "runtime_sec": h.get("runtime_sec"),
            }
            for h in full_valid_cap
        ],
    }

    user_msg = (
        "Here is the DSE data in JSON. "
        "Analyze it carefully and produce the final report.\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )

    # ------------------------------------------------------------------
    # Try LLM
    # ------------------------------------------------------------------
    if getattr(ka, "is_ready", lambda: False)():
        try:
            text, _raw = ka._client.create_text(
                model=getattr(ka, "model", "gpt-4o-2024-08-06"),
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_output_tokens=getattr(ka, "max_output_tokens", 100000),
            )
            if isinstance(text, str) and text.strip():
                out_path.write_text(text.strip() + "\n", encoding="utf-8")
                return str(out_path)
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
    else:
        llm_error = "LLM not ready"

    # ------------------------------------------------------------------
    # Fallback Markdown report
    # ------------------------------------------------------------------
    md_lines: List[str] = []
    md_lines.append("# Final DSE Report")
    md_lines.append("")
    md_lines.append("## Executive Summary")
    md_lines.append(f"- Total trials: {summary['counts']['all']}")
    md_lines.append(f"- Job success: {summary['counts']['job_success']} | Job fail: {summary['counts']['job_fail']}")
    md_lines.append(f"- Usable points: {summary['counts']['usable']} | Unusable points: {summary['counts']['unusable']}")
    md_lines.append(f"- Full-valid points: {summary['counts']['full_valid']} | Pareto size: {summary['counts']['pareto_size']}")
    md_lines.append("")

    md_lines.append("## Search Quality and Data Quality")
    md_lines.append("- `job_success` means the full run completed successfully.")
    md_lines.append("- `usable` means the metrics are still usable for analysis even if the job failed.")
    md_lines.append("- `full_valid` means complete finite metrics (WNS/Area/Power) suitable for Pareto comparison.")
    md_lines.append("")
    md_lines.append(f"- Stage distribution: `{json.dumps(stage_counter, ensure_ascii=False)}`")
    md_lines.append(f"- Fidelity distribution: `{json.dumps(fidelity_counter, ensure_ascii=False)}`")
    md_lines.append(f"- Metrics status distribution: `{json.dumps(metrics_status_counter, ensure_ascii=False)}`")
    if pareto_load_error:
        md_lines.append(f"- Pareto load warning: `{pareto_load_error}`")
    md_lines.append("")

    md_lines.append("## Best Points")
    for k, v in summary["best"].items():
        md_lines.append(f"- **{k}**: run_id={v.get('run_id')} metrics={v.get('metrics')} params={v.get('params')}")
    md_lines.append("")

    md_lines.append("## Failure Analysis")
    if not unusable_cap:
        md_lines.append("- No unusable failures captured in the capped history.")
    else:
        for h in unusable_cap[-10:]:
            md_lines.append(
                f"- **{h.get('run_id')}** "
                f"(fidelity={h.get('fidelity')}, stage={h.get('stage')}, metrics_status={h.get('metrics_status')}, success={h.get('success')}):"
            )
            tail = _short_text(h.get("fail_log_tail") or h.get("error_tail") or h.get("error") or "", max_len=800)
            if tail:
                md_lines.append("")
                md_lines.append("```text")
                md_lines.append(tail)
                md_lines.append("```")
            else:
                md_lines.append("  - no error tail available")
    md_lines.append("")

    md_lines.append("## Appendix: Top Pareto Points")
    md_lines.append("| run_id | wns | area | power | params |")
    md_lines.append("|---|---:|---:|---:|---|")
    for p in pareto_cap:
        rid = p.get("run_id", "")
        m = p.get("metrics", {}) or {}
        md_lines.append(
            f"| {rid} | {m.get('wns')} | {m.get('area')} | {m.get('power')} | {json.dumps(p.get('params', {}), ensure_ascii=False)} |"
        )

    md_lines.append("")
    md_lines.append("## Report Generation Notes")
    md_lines.append(f"- LLM status: `{llm_error}`")

    out_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return str(out_path)


def _metrics_str(m: Dict[str, Any]) -> str:
    """
    Pretty-print metrics for logs.
    Keeps backward compatibility (area/wns/power) and adds wl/ecp/clock_period if present.

    Expected keys (float-like):
      - area, wns, power, wl, ecp, clock_period
    Also may contain:
      - __proxy__ / __missing__ / __reason__ / proxy_* fields
    """
    import math

    def f(key: str, default: float) -> float:
        try:
            v = m.get(key, default)
            return float(v)
        except Exception:
            return default

    area = f("area", float("inf"))
    wns = f("wns", -100.0)
    power = f("power", float("inf"))
    wl = f("wl", float("inf"))
    ecp = f("ecp", float("inf"))
    clk = f("clock_period", float("inf"))

    def fmt(x: float, prec: int = 3) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return f"{x:.{prec}f}"

    # Keep your legacy order first
    parts = [
        f"Area={fmt(area, 1)}",
        f"WNS={fmt(wns, 4)}",
        f"Power={fmt(power, 4)}",
    ]

    # Add new metrics if available (not NA)
    if math.isfinite(wl):
        parts.append(f"WL={fmt(wl, 1)}")
    if math.isfinite(clk):
        parts.append(f"CLK={fmt(clk, 4)}")
    if math.isfinite(ecp):
        parts.append(f"ECP={fmt(ecp, 4)}")

    # Optional: show proxy marker if present
    if m.get("__proxy__", False):
        # keep short to avoid clutter
        kind = m.get("__wl_kind__", None)
        if kind:
            parts.append(f"proxy(wl={kind})")
        else:
            parts.append("proxy")

    return " ".join(parts)


def _slim_history_entry_main(res: dict) -> None:
    """
    MEM FIX: Shrink history entries in place to control memory growth.
    Each fail_log_tail can reach 200 KB, and each log_summary can be tens of KB,
    so thousands of entries can accumulate into tens of GB.
    """
    tail = res.get("fail_log_tail", "")
    if isinstance(tail, str) and len(tail) > 2048:
        res["fail_log_tail"] = tail[-2048:]

    ls = res.get("log_summary")
    if isinstance(ls, dict):
        res["log_summary"] = {
            "symptoms": ls.get("symptoms", []),
            "stage": ls.get("stage"),
            "final_status": ls.get("final_status"),
        }

    # Truncate oversized content in the error field.
    err = res.get("error", "")
    if isinstance(err, str) and len(err) > 1024:
        res["error"] = err[:1024]





def _classify_metrics(metrics: Dict[str, Any], *, fidelity: str) -> Dict[str, Any]:
    """
    Classify metrics completeness/quality.
    - success: task finished and returned a metrics dict (maybe proxy)
    - metrics_usable: usable for quick exploration / promotion prefilter
    - metrics_valid: FULL-valid for Pareto/EHVI (requires finite area/wns/power)
    """
    m = metrics or {}

    # ★ FIX: if any core metric was a sentinel (extraction failed), reject early
    if any(m.get(f"__{k}_sentinel__", False) for k in ("area", "wns", "power")):
        return {"metrics_status": "sentinel", "metrics_usable": False, "metrics_valid": False}

    try:
        area = float(m.get("area", float("inf")))
    except Exception:
        area = float("inf")
    try:
        wns = float(m.get("wns", -100.0))
    except Exception:
        wns = -100.0
    try:
        power = float(m.get("power", float("inf")))
    except Exception:
        power = float("inf")

    # basic sanity for any "usable" point
    basic_ok = (math.isfinite(area) and math.isfinite(wns) and (wns != -100.0))

    if not basic_ok:
        return {"metrics_status": "invalid", "metrics_usable": False, "metrics_valid": False}

    # full must have power
    if str(fidelity).lower().strip() == "full":
        if math.isfinite(power):
            return {"metrics_status": "full", "metrics_usable": True, "metrics_valid": True}
        return {"metrics_status": "invalid", "metrics_usable": False, "metrics_valid": False}

    # quick: power is allowed to be inf
    if bool(m.get("__proxy__", False)):
        return {"metrics_status": "proxy", "metrics_usable": True, "metrics_valid": False}
    return {"metrics_status": "partial", "metrics_usable": True, "metrics_valid": False}


def _dominates(ma: Dict[str, Any], mb: Dict[str, Any]) -> bool:
    # minimize area/power, maximize wns
    try:
        a_area, a_power, a_wns = float(ma["area"]), float(ma["power"]), float(ma["wns"])
        b_area, b_power, b_wns = float(mb["area"]), float(mb["power"]), float(mb["wns"])
    except Exception:
        return False

    cond = (b_area <= a_area) and (b_power <= a_power) and (b_wns >= a_wns)
    strict = (b_area < a_area) or (b_power < a_power) or (b_wns > a_wns)
    return bool(cond and strict)


def _pareto_snapshot(full_valid: List[Dict[str, Any]], max_items: int = 8) -> List[Dict[str, Any]]:
    valid = []
    for r in full_valid:
        if not r.get("metrics_valid", False):
            continue
        m = r.get("metrics", {}) or {}
        # Detect sentinel markers instead of relying on a hard-coded WNS threshold.
        if any(m.get(f"__{k}_sentinel__", False) for k in ("area", "wns", "power")):
            continue
        area = float(m.get("area", float("inf")))
        power = float(m.get("power", float("inf")))
        wns = float(m.get("wns", -100.0))
        # Exclude only the exact sentinel default values.
        if area == float("inf") or power == float("inf") or wns == -100.0:
            continue
        valid.append(r)

    pareto = []
    for a in valid:
        dominated = False
        for b in valid:
            if a["run_id"] == b["run_id"]:
                continue
            if _dominates(a["metrics"], b["metrics"]):
                dominated = True
                break
        if not dominated:
            pareto.append(a)

    # sort: higher wns first, then lower area, lower power
    pareto.sort(key=lambda x: (-float(x["metrics"]["wns"]), float(x["metrics"]["area"]), float(x["metrics"]["power"])))
    return pareto[: int(max_items)]

@dataclass
class TaskSpec:
    run_id: str
    fidelity: str                  # "quick" | "full"
    stage: str                     # "warmup_full" | "explore_quick" | "promote_full"
    params: Dict[str, Any]
    make_target: Optional[str]
    timeout_sec: int
    allow_missing_metrics: bool

    # (LLM / advanced) direct Tcl script override, consistent with EDARunner.run(..., script_override=...)
    script_override: Optional[str] = None

    parent_run_id: Optional[str] = None

    # attribution for console/KB
    proposed_by: str = "unknown"
    proposed_strategy: str = "unknown"
    notes: str = ""
    model: str = ""
    launch_wall_sec: float = 0.0     # wall-clock sec since experiment start
    ehvi_score: float = 0.0          # EHVI acquisition score (promotion only)
    ehvi_time_ucb: float = 0.0       # predicted runtime UCB

def _evaluate_one(runner: EDARunner, spec: TaskSpec) -> Dict[str, Any]:
    start = time.time()

    # Save params fingerprint into checkpoint for auto-resume dedup
    _ckpt_dir = Path(runner.work_dir if hasattr(runner, 'work_dir') else '.') / str(spec.run_id)
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        _fp_ckpt_path = _ckpt_dir / "checkpoint.json"
        if not _fp_ckpt_path.exists():
            _fp_ckpt_path.write_text(json.dumps({
                "run_id": spec.run_id,
                "fidelity": spec.fidelity,
                "params_fingerprint": _fingerprint(spec.params),
                "current_phase": "pre_launch",
                "timestamp": time.time(),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    # OpenRoadFlow.run(params, run_id, script_override=None, **kwargs)
    ok, log_path = runner.run(
        spec.params,
        spec.run_id,
        script_override=getattr(spec, "script_override", None),
        make_target=spec.make_target,
        timeout_sec=spec.timeout_sec,
        allow_missing_metrics=spec.allow_missing_metrics,
    )
    runtime_sec = time.time() - start

    # IMPORTANT: get_metrics expects log_path (not run_id)
    metrics = runner.get_metrics(log_path)

    # classify metrics completeness/validity (keep your current logic)
    cls = _classify_metrics(metrics, fidelity=spec.fidelity)
    metrics_status = cls["metrics_status"]
    metrics_usable = cls["metrics_usable"]
    metrics_valid = cls["metrics_valid"]

    # attach error tail if failed OR unusable
    fail_tail = ""
    if (not ok) or (not metrics_usable):
        try:
            fail_tail = _extract_error_warning_tail(log_path, max_lines=12, max_bytes=200_000)
        except Exception:
            fail_tail = ""

    # log summary (original signature requires metrics/runtime_sec)
    success_for_log = bool(ok) and (not bool((metrics or {}).get("__placeholder__", False)))
    log_summary = analyze_log_summary(
        log_path=str(log_path),
        metrics=(metrics or {}),
        runtime_sec=float(runtime_sec),
        success=bool(success_for_log),
        stage=str(spec.stage),
        fidelity=str(spec.fidelity),
        make_target=spec.make_target,
        fail_log_tail=str(fail_tail or ""),
    )

    # enrich proxy metrics from log_summary numbers
    try:
        nums = (log_summary.get("numbers", {}) or {}) if isinstance(log_summary, dict) else {}
        if metrics_status in ("proxy", "partial") and isinstance(nums, dict):
            if "tns" in nums and nums["tns"] is not None:
                metrics.setdefault("proxy_tns", nums["tns"])
            if "drc_count" in nums and nums["drc_count"] is not None:
                metrics.setdefault("proxy_drc_count", nums["drc_count"])
            if "route_overflow" in nums and nums["route_overflow"] is not None:
                metrics.setdefault("proxy_route_overflow", nums["route_overflow"])
    except Exception:
        pass

    return {
        "run_id": spec.run_id,
        "parent_run_id": spec.parent_run_id,
        "fidelity": spec.fidelity,
        "stage": spec.stage,
        "params": spec.params,
        "metrics": metrics,
        "success": bool(ok),
        "metrics_status": metrics_status,
        "metrics_usable": bool(metrics_usable),
        "metrics_valid": bool(metrics_valid),
        "log_path": log_path,
        "runtime_sec": float(runtime_sec),
        "proposed_by": spec.proposed_by,
        "proposed_strategy": spec.proposed_strategy,
        "proposed_notes": spec.notes,
        "proposed_model": spec.model,
        "log_summary": log_summary,
        "fail_log_tail": fail_tail,
    }


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    tuning = cfg.get("tuning", {}) or {}
    fid = cfg.get("fidelity", {}) or {}

    # ------------ budgets ------------
    MAX_CONCURRENCY = int(tuning.get("max_concurrency", 4))
    TOTAL_SECONDS = int(tuning.get("total_seconds", 10 * 3600))
    MAX_TRIALS = int(tuning.get("max_trials", 100000))
    CLEANUP_EVERY = int(tuning.get("cleanup_every", 20))

    # ---- Innovus-only mode ----
    TUNING_MODE = str(tuning.get("mode", "full")).strip().lower()
    MAX_ITERATIONS = int(tuning.get("max_iterations", 200))
    EST_INNOVUS_SEC = float(tuning.get("estimated_innovus_sec", 1800))
    SHARED_GENUS_RUN_ID = str(tuning.get("shared_genus_run_id", "") or "").strip()
    IS_INNOVUS_ONLY = (TUNING_MODE == "innovus_only")


    STOP_STARTING_SEC = int(fid.get("stop_starting_sec", 15 * 60))
    RISK_BETA = float(fid.get("risk_beta", 2.0))
    MARGIN_SEC = float(fid.get("margin_sec", 0.0))

    # ------------ fidelity ------------
    QUICK_TARGET = str(fid.get("quick_target", "cts")).strip()
    FULL_TARGET = str(fid.get("full_target", "")).strip()
    QUICK_TIMEOUT = int(fid.get("quick_timeout_sec", 1800))
    FULL_TIMEOUT = int(fid.get("full_timeout_sec", 7200))

    PROMOTE_PROB = float(fid.get("promote_prob", 0.35))
    PROMO_BATCH = int(fid.get("promo_batch", 64))
    WARMUP_QUICK = int(fid.get("warmup_quick", max(16, PROMO_BATCH // 2)))
    MAX_PROMO_POOL = int(fid.get("max_promo_pool", 2000))

    TOPK_EVAL = int(tuning.get("topk_eval", 256))

    # ------------ ablation flags ------------
    USE_EHVI = bool(tuning.get("use_ehvi", True))
    ka_cfg_section = cfg.get("ka", {}) or {}
    PERSIST_RULES = bool(ka_cfg_section.get("persist_rules", True))

    FILTER_NO_CTS = bool(fid.get("filter_no_cts", False))

    # NEW: Safety guardrails for Knowledge Persistence.
    KP_MIN_FV_FOR_HARD_RULES = int(ka_cfg_section.get("min_full_valid_for_hard_rules", 12))
    KP_MAX_ACTIVE_HARD_RULES = int(ka_cfg_section.get("max_active_hard_rules", 8))
    KP_HARD_RULE_TRIM_INTERVAL = int(ka_cfg_section.get("hard_rule_trim_interval", 50))

    if not USE_EHVI:
        print("[ABLATION] EHVI disabled: promotions are sampled uniformly at random")
    if not PERSIST_RULES:
        print("[ABLATION] Knowledge Persistence disabled: no rule learning, no periodic analysis, no prior refinement")
    else:
        print(f"[KP] safety guards: min_fv_for_hard_rules={KP_MIN_FV_FOR_HARD_RULES}, "
              f"max_active_hard_rules={KP_MAX_ACTIVE_HARD_RULES}, "
              f"trim_interval={KP_HARD_RULE_TRIM_INTERVAL}")

    # ------------ core objects ------------
    param_space = _get_param_space(cfg)
    default_params = _default_params_to_friendly(cfg)

    runner = create_runner(cfg)
    agent = MOEHVIAgent(param_space)
    agent.warmup_trials = int(cfg.get("tuning", {}).get("ehvi_warmup_trials", agent.warmup_trials))
    agent.candidate_pool_size = int(cfg.get("tuning", {}).get("ehvi_candidate_pool_size", agent.candidate_pool_size))
    agent.max_gp_obs = int(cfg.get("tuning", {}).get("ehvi_max_gp_obs", agent.max_gp_obs))
    rt = MultiFidelityRuntimePredictor(param_space, min_samples_quick=10, min_samples_full=10)
    # ---- Iteration budget estimator (innovus-only mode) ----
    ibe = IterationBudgetEstimator(
        estimated_sec_per_iter=EST_INNOVUS_SEC,
        max_concurrency=MAX_CONCURRENCY,
    ) if IS_INNOVUS_ONLY else None


    # KA (LLM)
    ka = KnowledgeAgent(cfg)

    # ABLATION: disable knowledge persistence if configured
    if not PERSIST_RULES:
        ka._persist_rules_enabled = False
    else:
        ka._persist_rules_enabled = True

    # FIX: If KA is disabled, clear leftover knowledge files.
    if not ka.is_ready():
        _knowledge_dir = Path(cfg["work_dir"]) / "knowledge"
        if _knowledge_dir.is_dir():
            for _kf in ("hard_rules.jsonl", "soft_heuristics.jsonl"):
                _kf_path = _knowledge_dir / _kf
                if _kf_path.is_file() and _kf_path.stat().st_size > 0:
                    print(f"[ABLATION WARN] KA disabled but found residual {_kf}, "
                          f"renaming to {_kf}.bak to prevent leakage")
                    _kf_path.rename(_kf_path.with_suffix(_kf_path.suffix + ".bak"))

    prior_builder = DesignPriorBuilder(cfg, llm_client=ka._client, model=ka.model)
    if ka.is_ready():
        design_prior_text = prior_builder.get_or_build()
    else:
        design_prior_text = ""
        print("[ABLATION] KA disabled: skipping design prior to prevent info leakage")
    ka.design_prior_text = design_prior_text

    # ------------ RAG Service (global) ------------
    rag = get_rag_service(cfg)  # Builds the index automatically; no-op when rag.enable=false.
    if rag.is_ready:
        print(f"[RAG] Service ready, vectors={rag._index._index.ntotal}")
        print(f"[RAG] load_summary={rag.load_summary}")
        print(f"[RAG] has_custom_corpus={rag.has_custom_corpus} has_transfer_corpus={rag.has_transfer_corpus}")
    if rag.is_ready:
        print(f"[RAG] Service ready, vectors={rag._index._index.ntotal}")
        # Use RAG to enrich the design prior.
        rag_prior = rag.context_for_design_prior(
            current_params=default_params,
            metrics=None,
        )
        if rag_prior and ka.design_prior_text:
            ka.design_prior_text = ka.design_prior_text + "\n\n" + rag_prior
        elif rag_prior:
            ka.design_prior_text = rag_prior

    # ------------ KB ------------
    kb_dir = Path(cfg["work_dir"]) / "kb"
    kb = KB(kb_dir)
    kb.write_meta(
        {
            "project_name": cfg.get("project_name", ""),
            "created_ts": time.time(),
            "max_concurrency": MAX_CONCURRENCY,
            "total_seconds": TOTAL_SECONDS,
            "stop_starting_sec": STOP_STARTING_SEC,
            "quick_target": QUICK_TARGET,
            "full_target": FULL_TARGET,
            "quick_timeout_sec": QUICK_TIMEOUT,
            "full_timeout_sec": FULL_TIMEOUT,
            "promote_prob": PROMOTE_PROB,
            "promo_batch": PROMO_BATCH,
            "max_promo_pool": MAX_PROMO_POOL,
            "risk_beta": RISK_BETA,
            "margin_sec": MARGIN_SEC,
            "param_space": param_space,
            "default_params": cfg.get("default_params", {}),
            "llm_enabled": bool(ka.is_ready()),
        }
    )

    # ------------ state ------------
    history_all: List[Dict[str, Any]] = []
    history_full_valid: List[Dict[str, Any]] = []  # only full + valid metrics

    promo_pool: Dict[str, Tuple[Dict[str, Any], str]] = {}  # fp -> (params, quick_run_id)
    warmup_quick_done: bool = (WARMUP_QUICK <= 0)
    warmup_quick_success: int = 0   # counts usable quick runs

    seen_quick: set[str] = set()
    seen_full: set[str] = set()

    _direct_full_count: int = 0       # Number of direct-to-full launches.
    _DIRECT_FULL_MAX_FRAC = 0.15      # Cap direct-to-full launches at 15% of all launches.

    # KA candidate buffer
    ka_buffer: List[Dict[str, Any]] = []  # each item is KA candidate dict

    # ---- Periodic analysis for Sections 3.1/3.2/3.3 ----
    # TOKEN-OPT: Increase intervals to reduce LLM call frequency.
    ANALYSIS_INTERVAL = max(25, CLEANUP_EVERY * 2)             # TOKEN-OPT: 15 -> 25
    ANALYSIS_MIN_HISTORY = 30                                  # TOKEN-OPT: 25 -> 30
    ANALYSIS_MIN_FULL_VALID = 10                               # TOKEN-OPT: 8 -> 10
    PRIOR_REFINE_INTERVAL = max(80, CLEANUP_EVERY * 6)         # TOKEN-OPT: 40 -> 80
    PRIOR_REFINE_MIN_FULL_VALID = 15                           # TOKEN-OPT: 10 -> 15
    last_analysis_count = 0
    last_refine_count = 0
    last_rule_trim_count = 0                                   # NEW: Rule cleanup counter.

    start_wall = time.time()
    if IS_INNOVUS_ONLY:
        # In innovus-only mode, terminate by iteration count rather than wall-clock deadline.
        # Use a very large deadline to avoid tripping any remaining time-based checks.
        deadline = start_wall + 365 * 24 * 3600  # 1 year
    else:
        deadline = start_wall + TOTAL_SECONDS

    launched = 0
    _resume_elapsed_base = 0.0   # NEW: Accumulated elapsed time from previous sessions.
    num_quick_done = 0
    inflight = {}  # future -> TaskSpec
    _resume_inflight_runs = []  # NEW: Inflight runs restored from resume_state.

    # ============================================================
    # Resume from resume_state.json (generated by tools/recover_state.py)
    # ============================================================
    resume_path = Path(cfg["work_dir"]) / "resume_state.json"
    if resume_path.is_file():
        try:
            _rs = json.loads(resume_path.read_text(encoding="utf-8"))
            if isinstance(_rs, dict) and _rs.get("version"):
                # Restore history
                _restored_all = _rs.get("history_all", [])
                if isinstance(_restored_all, list) and _restored_all:
                    history_all = [r for r in _restored_all if isinstance(r, dict)]

                _restored_fv = _rs.get("history_full_valid", [])
                if isinstance(_restored_fv, list) and _restored_fv:
                    history_full_valid = [r for r in _restored_fv if isinstance(r, dict)]

                # Restore promo_pool
                _pp = _rs.get("promo_pool", {})
                if isinstance(_pp, dict):
                    for fp, info in _pp.items():
                        if isinstance(info, dict) and "params" in info:
                            promo_pool[str(fp)] = (info["params"], str(info.get("parent_run_id") or info.get("quick_run_id") or ""), info.get("quick_metrics", {}))



                # Restore dedup sets
                _sq = _rs.get("seen_quick", [])
                if isinstance(_sq, list):
                    seen_quick = set(str(x) for x in _sq)
                _sf = _rs.get("seen_full", [])
                if isinstance(_sf, list):
                    seen_full = set(str(x) for x in _sf)

                # Restore counters
                launched = int(_rs.get("launched", 0))
                warmup_quick_success = int(_rs.get("warmup_quick_success", 0))
                if warmup_quick_success >= WARMUP_QUICK:
                    warmup_quick_done = True

                # Adjust deadline: inherit previously consumed time
                prev_elapsed = float(_rs.get("total_elapsed_sec", 0.0))
                if prev_elapsed > 0:
                    _resume_elapsed_base = prev_elapsed   # NEW: Preserve previously accumulated time.
                    deadline = start_wall + max(0, TOTAL_SECONDS - prev_elapsed)
                    remaining_h = max(0, TOTAL_SECONDS - prev_elapsed) / 3600.0
                    print("         time inherited: %.1fh used, %.1fh remaining (of %.1fh total)"
                          % (prev_elapsed / 3600.0, remaining_h, TOTAL_SECONDS / 3600.0))

                # Feed runtime surrogate with historical data (deferred until fill_params is defined)
                _resume_rt_data = []
                for _hr in history_all:
                    if _hr.get("success", False):
                        _hfid = str(_hr.get("fidelity", "quick"))
                        _hparams = _hr.get("params", {})
                        _hrt = float(_hr.get("runtime_sec", 0.0))
                        if _hparams and _hrt > 0:
                            _resume_rt_data.append((_hfid, _hparams, _hrt))

                print("[RESUME] loaded resume_state.json:")
                print("         history_all=%d  history_full_valid=%d  promo_pool=%d"
                      % (len(history_all), len(history_full_valid), len(promo_pool)))
                print("         seen_quick=%d  seen_full=%d  launched=%d  warmup_quick_success=%d"
                      % (len(seen_quick), len(seen_full), launched, warmup_quick_success))
                print("         warmup_quick_done=%s  next_run_id=iter_%d"
                      % (str(warmup_quick_done), launched))

                # NEW: Save inflight runs that need to be re-attached.
                _resume_inflight_runs = _rs.get("inflight_runs", [])
                if isinstance(_resume_inflight_runs, list) and _resume_inflight_runs:
                    print("         inflight_runs=%d (will re-attach after executor starts)"
                          % len(_resume_inflight_runs))
                else:
                    _resume_inflight_runs = []

                # Rename resume file so it's not loaded again on next restart
                # (the state is now in memory; KB will accumulate new records)
                _backup = resume_path.with_suffix(".json.used")
                try:
                    resume_path.rename(_backup)
                    print("         resume_state.json renamed to resume_state.json.used")
                except Exception:
                    pass

        except Exception as _e:
            print("[WARN] failed to load resume_state.json: %s" % str(_e))

    # ============================================================
    # Auto-detect existing runs (safety net when no resume_state.json)
    # Prevents re-launching iter_0..N when restarting without recovery
    # ============================================================
    if launched == 0:
        # Scan for existing iter_* directories
        _work_path = Path(cfg["work_dir"])
        _max_existing_iter = -1
        _auto_seen_quick = set()
        _auto_seen_full = set()
        if _work_path.is_dir():
            for _d in _work_path.iterdir():
                if _d.is_dir() and _d.name.startswith("iter_"):
                    try:
                        _n = int(_d.name.replace("iter_", ""))
                        _max_existing_iter = max(_max_existing_iter, _n)
                    except ValueError:
                        pass
                    # Try to extract params fingerprint from checkpoint
                    _ckpt_file = _d / "checkpoint.json"
                    if _ckpt_file.is_file():
                        try:
                            _ckpt_data = json.loads(_ckpt_file.read_text(encoding="utf-8"))
                            _ckpt_fp = _ckpt_data.get("params_fingerprint")
                            _ckpt_fid = _ckpt_data.get("fidelity", "quick")
                            if _ckpt_fp:
                                if _ckpt_fid == "full":
                                    _auto_seen_full.add(_ckpt_fp)
                                else:
                                    _auto_seen_quick.add(_ckpt_fp)
                        except Exception:
                            pass

        if _max_existing_iter >= 0:
            launched = _max_existing_iter + 1
            seen_quick.update(_auto_seen_quick)
            seen_full.update(_auto_seen_full)
            print("[AUTO-RESUME] detected existing runs up to iter_%d, starting from iter_%d"
                  % (_max_existing_iter, launched))
            print("             auto-recovered seen_quick=%d seen_full=%d"
                  % (len(_auto_seen_quick), len(_auto_seen_full)))

    # ---- Global stop / pause state ----
    fatal_stop_requested: bool = False
    fatal_stop_reason: str = ""

    def _check_stop_file() -> bool:
        nonlocal fatal_stop_requested, fatal_stop_reason
        stop_path = Path(cfg["work_dir"]) / "STOP"
        if stop_path.exists():
            if not fatal_stop_requested:
                print("[STOP] stop flag detected at %s" % str(stop_path))
                fatal_stop_requested = True
                fatal_stop_reason = "STOP file detected"
            return True
        return False

    def _trigger_fatal_stop(reason: str) -> None:
        nonlocal fatal_stop_requested, fatal_stop_reason
        if not fatal_stop_requested:
            fatal_stop_requested = True
            fatal_stop_reason = reason
            print("[FATAL] %s" % reason)
    
    # NEW: Guardrail for repeated connection failures.
    _consecutive_conn_failures = 0
    _MAX_CONSECUTIVE_CONN_FAILURES = 5        # Start pausing after 5 consecutive failures.
    _CONN_FAILURE_WAIT_SEC = 120              # Wait 120 seconds per round.
    _MAX_CONN_FAILURE_WAIT_ROUNDS = 5         # Wait at most 5 rounds (~10 minutes) before going fatal.

    def _on_run_result(success: bool, is_conn_error: bool) -> None:
        """Update the consecutive connection failure counter after each run."""
        nonlocal _consecutive_conn_failures
        if is_conn_error:
            _consecutive_conn_failures += 1
            if _consecutive_conn_failures >= _MAX_CONSECUTIVE_CONN_FAILURES:
                wait_round = _consecutive_conn_failures - _MAX_CONSECUTIVE_CONN_FAILURES + 1
                if wait_round > _MAX_CONN_FAILURE_WAIT_ROUNDS:
                    _trigger_fatal_stop(
                        f"Network unreachable: {_consecutive_conn_failures} consecutive connection failures "
                        f"over {wait_round * _CONN_FAILURE_WAIT_SEC / 60:.0f} minutes"
                    )
                else:
                    print(
                        f"[WARN] {_consecutive_conn_failures} consecutive connection failures, "
                        f"waiting {_CONN_FAILURE_WAIT_SEC}s before retrying "
                        f"(round {wait_round}/{_MAX_CONN_FAILURE_WAIT_ROUNDS}) ..."
                    )
                    time.sleep(_CONN_FAILURE_WAIT_SEC)
        else:
            if _consecutive_conn_failures > 0:
                print(
                    f"[INFO] Connection restored after "
                    f"{_consecutive_conn_failures} consecutive failures"
                )
            _consecutive_conn_failures = 0

    def remaining_sec() -> float:
        if IS_INNOVUS_ONLY:
            remaining_iters = max(0, MAX_ITERATIONS - launched)
            return ibe.remaining_sec(remaining_iters)
        return max(0.0, deadline - time.time())

    def compute_budget_remaining() -> float:
        if IS_INNOVUS_ONLY:
            remaining_iters = max(0, MAX_ITERATIONS - launched)
            return ibe.budget_remaining(remaining_iters)
        elapsed = TOTAL_SECONDS - remaining_sec()
        return max(0.0, TOTAL_SECONDS * MAX_CONCURRENCY - elapsed)

    def can_launch_new() -> bool:
        if fatal_stop_requested:
            return False
        if is_backpressure_active():
            return False
        if launched >= MAX_TRIALS:
            return False
        if IS_INNOVUS_ONLY:
            return launched < MAX_ITERATIONS
        # Do not stop launching early based on STOP_STARTING_SEC; keep launching until the deadline.
        # Allow the last inflight batch to run out naturally; the main loop waits for inflight tasks.
        return time.time() < deadline






    def fill_params(p: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill missing/None values in params using:
        1) cfg default_params (already mapped to friendly names)
        2) param_space defaults
        3) param_space range midpoint / first choice

        This ensures runtime_model and dedup see a full vector without None.
        """
        out: Dict[str, Any] = dict(default_params)
        if isinstance(p, dict):
            out.update(p)

        for ps in param_space:
            if not isinstance(ps, dict):
                continue
            name = ps.get("name", None)
            if not name:
                continue

            v = out.get(name, None)
            if v is not None:
                continue

            # 1) explicit default in param_space
            if "default" in ps and ps.get("default") is not None:
                out[name] = ps.get("default")
                continue

            # 2) range midpoint
            if "range" in ps and isinstance(ps.get("range"), (list, tuple)) and len(ps.get("range")) == 2:
                try:
                    lo, hi = ps["range"]
                    lo_f = float(lo)
                    hi_f = float(hi)
                    mid = (lo_f + hi_f) / 2.0
                    typ = str(ps.get("type", "")).lower().strip()
                    if typ in ("int", "integer"):
                        out[name] = int(round(mid))
                    else:
                        out[name] = float(mid)
                    continue
                except Exception:
                    pass

            # 3) choices first item
            if "choices" in ps and isinstance(ps.get("choices"), (list, tuple)) and len(ps.get("choices")) > 0:
                out[name] = ps["choices"][0]
                continue

        return out

    def merge_defaults(p: Dict[str, Any]) -> Dict[str, Any]:
        # Backward-compat alias used by scoring code.
        return fill_params(p)

    # ---- Deferred: feed runtime model with resumed history ----
    try:
        _rt_data = _resume_rt_data  # set in resume section above
        for _fid, _params, _runtime in _rt_data:
            try:
                rt.observe(_fid, fill_params(_params), _runtime)
            except Exception:
                pass
        try:
            rt.fit("quick")
            rt.fit("full")
        except Exception:
            pass
        del _resume_rt_data
        print("[RESUME] runtime model fed with %d historical observations" % len(_rt_data))
    except NameError:
        pass  # no resume data

    def _print_launch(spec: TaskSpec) -> None:
        phase = "1(quick)" if spec.fidelity == "quick" else "2(full)"
        promo = "YES" if spec.stage == "promote_full" else "NO"
        parent = spec.parent_run_id or "-"
        print(
            "[LAUNCH] %s phase=%s stage=%s promotion=%s parent=%s target=%s timeout=%ss by=%s strategy=%s model=%s"
            % (
                spec.run_id,
                phase,
                spec.stage,
                promo,
                parent,
                (spec.make_target if spec.make_target else "<default>"),
                str(spec.timeout_sec),
                spec.proposed_by,
                spec.proposed_strategy,
                (spec.model or "-"),
            )
        )
        print("         params=%s" % (json.dumps(spec.params, ensure_ascii=False),))
        if spec.notes:
            print("         notes=%s" % (spec.notes[:240],))

    def _print_done(res: Dict[str, Any]) -> None:
        phase = "1(quick)" if res.get("fidelity") == "quick" else "2(full)"
        print(
            "[DONE]  %s phase=%s stage=%s success=%s metrics_status=%s usable=%s valid=%s runtime=%.1fs %s by=%s strategy=%s"
            % (
                res.get("run_id"),
                phase,
                res.get("stage", ""),
                str(res.get("success")),
                str(res.get("metrics_status", "?")),
                str(res.get("metrics_usable", False)),
                str(res.get("metrics_valid", False)),
                float(res.get("runtime_sec", 0.0)),
                _metrics_str(res.get("metrics", {}) or {}),
                res.get("proposed_by", "unknown"),
                res.get("proposed_strategy", "unknown"),
            )
        )

    def _refill_ka_buffer(stage: str, need: int) -> None:
        """Ask KA for *M* candidates and push them into ka_buffer (no EHVI scoring here).

        EHVI scoring is done at selection time so it always uses the latest history.
        """
        if USE_EHVI and len(history_full_valid) >= agent.warmup_trials:
            try:
                regions = agent.get_promising_regions(history_full_valid, n_regions=3)
                if regions:
                    ka.ehvi_promising_regions = regions
            except Exception as e:
                print(f"[EHVI→KA] region extraction failed: {e}")
        
        nonlocal ka_buffer
        need = max(1, int(need))
        M = max(1, int((cfg.get("ka", {}) or {}).get("max_candidates_per_call", 8)))

        # NEW: Sync the current full_valid count to KA for evidence thresholds.
        ka._current_full_valid_count = len(history_full_valid)

        # Use full pareto if we have it; otherwise fall back to all history so KA still gets signal.
        pareto_hist = history_full_valid if history_full_valid else history_all
        pareto = _pareto_snapshot(pareto_hist, max_items=8)

        # FIX: Pass full_valid to KA so rules are based only on complete evaluations.
        # Fall back to history_all when full_valid is still too small during cold start.
        _ka_history = history_full_valid if len(history_full_valid) >= 6 else history_all
        suggest = ka.propose_candidates(
            param_space=param_space,
            default_params=default_params,
            history=_ka_history,
            pareto=pareto,
            remaining_sec=remaining_sec(),
            stage=stage,
            # KA internally clamps to max_candidates_per_call; we request at least M.
            k=max(M, need),
        )

        # print KA summary (once per refill)
        msg = "[KA] enabled=%s by=%s model=%s strategy=%s" % (
            str(ka.is_ready()),
            suggest.agent_name,
            suggest.model,
            suggest.strategy,
        )
        print(msg)
        if suggest.error:
            print("[KA] error=%s" % suggest.error)
        if suggest.rationale:
            print("[KA] rationale=%s" % (suggest.rationale,))

        # Dedup (by fingerprint) when extending buffer
        existing = set()
        for c in ka_buffer:
            p0 = c.get("params", {}) or {}
            if isinstance(p0, dict) and p0:
                existing.add(_fingerprint(fill_params(merge_defaults(p0))))

        added = 0
        for c in suggest.candidates:
            p0 = c.get("params", {}) or {}
            if not isinstance(p0, dict) or not p0:
                continue
            p = merge_defaults(p0)
            fp = _fingerprint(fill_params(p))
            if fp in existing:
                continue
            c2 = dict(c)
            c2["params"] = p
            ka_buffer.append(c2)
            existing.add(fp)
            added += 1
            

        # Keep buffer bounded so we don't keep stale candidates forever.
        max_buf = max(32, 3 * M)
        if len(ka_buffer) > max_buf:
            ka_buffer = ka_buffer[:max_buf]
            

    def _maybe_refill_ka_buffer(*, stage: str, min_buf: int, target_buf: int) -> None:
        """
        Only refill when buffer is below min_buf, and only top-up to target_buf.
        This avoids calling KA/LLM too often.
        """
        nonlocal ka_buffer

        min_buf = int(max(0, min_buf))
        target_buf = int(max(min_buf, target_buf))

        if len(ka_buffer) >= min_buf:
            return

        need = max(1, target_buf - len(ka_buffer))
        _refill_ka_buffer(stage=stage, need=need)
        print(f"[KA] refill stage={stage} len={len(ka_buffer)} -> need={need} (min={min_buf}, target={target_buf})")

    def _pick_next_quick_from_ka(*, stage: str = "explore_quick") -> Optional[TaskSpec]:
        nonlocal ka_buffer, launched, _direct_full_count

        # NEW: Budget control for direct-to-full launches.
        _allow_direct_full = (
            launched == 0  # Avoid division by zero.
            or (_direct_full_count / max(1, launched)) < _DIRECT_FULL_MAX_FRAC
        )

        # NEW: With 25% probability, use EHVI surrogate or random sampling to break LLM locality.
        EXPLORE_RANDOM_PROB = 0.25
        if random.random() < EXPLORE_RANDOM_PROB:
            p = None
            _by = "random"
            _strategy = "random_explore"
            if USE_EHVI and len(history_full_valid) >= 6:
                try:
                    regions = agent.get_promising_regions(
                        history_full_valid, n_regions=8, pool_size=2048,  # Increase candidate count.
                    )
                    if regions:
                        # NEW [Change 5b]: Pick the candidate farthest from the current Pareto front.
                        import numpy as np
                        pareto_snap = _pareto_snapshot(history_full_valid, max_items=9999)
                        if len(pareto_snap) >= 3:
                            pareto_X = []
                            for ph in pareto_snap:
                                try:
                                    pareto_X.append(
                                        agent._to_tensor(ph["params"]).squeeze().tolist()
                                    )
                                except Exception:
                                    pass
                            if pareto_X:
                                pareto_arr = np.array(pareto_X)
                                best_dist = -1.0
                                best_p = None
                                for r in regions:
                                    try:
                                        rx = np.array(
                                            agent._to_tensor(r["params"]).squeeze().tolist()
                                        )
                                        min_d = np.linalg.norm(pareto_arr - rx, axis=1).min()
                                        if min_d > best_dist:
                                            best_dist = min_d
                                            best_p = r["params"]
                                    except Exception:
                                        pass
                                if best_p is not None:
                                    p = fill_params(best_p)
                                    _by = "EHVI_diverse_explore"
                                    _strategy = "surrogate_guided_quick"
                        if p is None and regions:
                            p = fill_params(regions[0]["params"])
                    if p is not None and _by == "random":
                        _by = "EHVI_explore"
                        _strategy = "surrogate_guided_quick"
                except Exception:
                    p = None
            if p is None:
                p = fill_params(agent._random_sample())
                _by = "random"
                _strategy = "random_explore"

            p = fill_params(p)

            # NEW: Send RUN_CTS=0 cases directly to full without wasting quick budget.
            if _allow_direct_full and _needs_direct_full(p, QUICK_TARGET):
                _direct_full_count += 1
                fp_full = _fingerprint(p)
                if fp_full not in seen_full:
                    seen_full.add(fp_full)
                    run_id = "iter_%d" % launched
                    return TaskSpec(
                        run_id=run_id, fidelity="full", stage="direct_full_no_cts",
                        params=p,
                        make_target=(None if FULL_TARGET == "" else FULL_TARGET),
                        timeout_sec=FULL_TIMEOUT, allow_missing_metrics=False,
                        parent_run_id=None,
                        proposed_by=_by, proposed_strategy=_strategy + "_direct_full",
                        notes="RUN_CTS=0: skip quick (CTS no-op), direct to full.",
                    )
                # Collision: fall through.
            else:
                fp = _fingerprint_for_stage(p, "quick")
                if fp not in seen_quick:
                    seen_quick.add(fp)
                    run_id = "iter_%d" % launched
                    return TaskSpec(
                        run_id=run_id, fidelity="quick", stage=stage,
                        params=p, make_target=QUICK_TARGET,
                        timeout_sec=QUICK_TIMEOUT, allow_missing_metrics=True,
                        parent_run_id=None,
                        proposed_by=_by, proposed_strategy=_strategy,
                        notes="Surrogate/random exploration for diversity",
                    )
            # On collision, fall through to KA.

        # Original KA path (taken 75% of the time).
        KA_M = int((cfg.get("ka", {}) or {}).get("max_candidates_per_call", 6))
        min_buf = max(1, KA_M // 2)    # Changed from 1/4 to 1/2: refill when the buffer is half empty.
        target_buf = KA_M              # Refill only up to KA_M each time.

        _maybe_refill_ka_buffer(stage="quick", min_buf=min_buf, target_buf=target_buf)

        while ka_buffer:
            c = ka_buffer.pop(0)
            p = c.get("params", {}) or {}
            if not isinstance(p, dict) or not p:
                continue

            p = fill_params(p)

            run_id = "iter_%d" % launched
            proposed_by = str(c.get("proposed_by", "KA"))
            proposed_strategy = str(c.get("proposed_strategy", "KA"))
            notes = str(c.get("notes", ""))

            # NEW: Send RUN_CTS=0 cases directly to full.
            if _allow_direct_full and _needs_direct_full(p, QUICK_TARGET):
                fp_full = _fingerprint(p)
                if fp_full in seen_full:
                    continue
                seen_full.add(fp_full)
                return TaskSpec(
                    run_id=run_id, fidelity="full", stage="direct_full_no_cts",
                    params=p,
                    make_target=(None if FULL_TARGET == "" else FULL_TARGET),
                    timeout_sec=FULL_TIMEOUT, allow_missing_metrics=False,
                    parent_run_id=None,
                    proposed_by=proposed_by,
                    proposed_strategy=proposed_strategy + "_direct_full",
                    notes="RUN_CTS=0: " + notes,
                    model=str(c.get("proposed_model", "")) if c.get("proposed_model") else "",
                )
            else:
                fp = _fingerprint_for_stage(p, "quick")
                if fp in seen_quick:
                    continue
                seen_quick.add(fp)
                return TaskSpec(
                    run_id=run_id, fidelity="quick", stage=stage,
                    params=p, make_target=QUICK_TARGET,
                    timeout_sec=QUICK_TIMEOUT, allow_missing_metrics=True,
                    parent_run_id=None,
                    proposed_by=proposed_by,
                    proposed_strategy=proposed_strategy,
                    notes=notes,
                    model=str(c.get("proposed_model", "")) if c.get("proposed_model") else "",
                )

        return None

    # Sobol engine for warmup space filling.
    _warmup_sobol_engine = torch.quasirandom.SobolEngine(
        dimension=len(param_space), scramble=True
    )
    _warmup_sobol_drawn: int = 0

    def _sobol_to_params() -> Dict[str, Any]:
        """Generate one parameter set from the Sobol sequence."""
        nonlocal _warmup_sobol_drawn
        u = _warmup_sobol_engine.draw(1).squeeze().tolist()
        _warmup_sobol_drawn += 1
        p: Dict[str, Any] = {}
        for i, ps in enumerate(param_space):
            name = ps["name"]
            if "range" in ps:
                lo, hi = float(ps["range"][0]), float(ps["range"][1])
                val = lo + u[i] * (hi - lo)
                if ps.get("type") == "int":
                    val = int(round(val))
                else:
                    val = round(val, 4)
                p[name] = val
            elif "choices" in ps:
                choices = list(ps["choices"])
                idx = int(u[i] * len(choices)) % len(choices)
                p[name] = choices[idx]
        return fill_params(p)

    def _pick_warmup_quick() -> Optional[TaskSpec]:
        nonlocal launched, _direct_full_count

        # Budget control for direct-to-full during Sobol warmup.
        _warmup_allow_direct_full = (
            launched == 0
            or (_direct_full_count / max(1, launched)) < _DIRECT_FULL_MAX_FRAC
        )

        # Use Sobol for the first 50% of warmup, then KA for the remaining 50%.
        sobol_budget = max(8, WARMUP_QUICK // 2)
        if warmup_quick_success < sobol_budget:
            # Try up to 20 times to avoid collisions.
            for _ in range(20):
                p = _sobol_to_params()
                # FIX: Apply budget control to Sobol RUN_CTS=0 points; resample if the ratio is exceeded.
                if _needs_direct_full(p, QUICK_TARGET):
                    if not _warmup_allow_direct_full:
                        continue  # Budget exceeded; skip this RUN_CTS=0 point and resample.
                    fp_full = _fingerprint(p)
                    if fp_full not in seen_full:
                        seen_full.add(fp_full)
                        _direct_full_count += 1
                        run_id = "iter_%d" % launched
                        return TaskSpec(
                            run_id=run_id, fidelity="full",
                            stage="warmup_direct_full_no_cts",
                            params=p,
                            make_target=(None if FULL_TARGET == "" else FULL_TARGET),
                            timeout_sec=FULL_TIMEOUT, allow_missing_metrics=False,
                            parent_run_id=None,
                            proposed_by="Sobol", proposed_strategy="space_filling_direct_full",
                            notes="Sobol warmup, RUN_CTS=0: direct to full.",
                        )
                else:
                    fp = _fingerprint_for_stage(p, "quick")
                    if fp not in seen_quick:
                        seen_quick.add(fp)
                        run_id = "iter_%d" % launched
                        return TaskSpec(
                            run_id=run_id, fidelity="quick", stage="warmup_quick",
                            params=p,
                            make_target=(None if QUICK_TARGET == "" else QUICK_TARGET),
                            timeout_sec=QUICK_TIMEOUT, allow_missing_metrics=True,
                            parent_run_id=None,
                            proposed_by="Sobol", proposed_strategy="space_filling",
                            notes="Sobol warmup for initial space coverage",
                        )
            # Collision retries exhausted; fall through to KA.

        # Use KA for the second half of warmup (original logic).
        KA_M = int((cfg.get("ka", {}) or {}).get("max_candidates_per_call", 8))
        min_buf = max(1, KA_M // 4)
        target_buf = max(4, KA_M)
        _maybe_refill_ka_buffer(stage="quick", min_buf=min_buf, target_buf=target_buf)

        while ka_buffer:
            c = ka_buffer.pop(0)
            p = c.get("params", {}) or {}
            if not isinstance(p, dict) or not p:
                continue
            p = fill_params(p)

            run_id = "iter_%d" % launched
            proposed_by = str(c.get("proposed_by", "KA"))
            proposed_strategy = str(c.get("proposed_strategy", "KA"))
            notes = str(c.get("notes", ""))

            # FIX: Handle RUN_CTS=0 in the warmup KA path as well.
            if _warmup_allow_direct_full and _needs_direct_full(p, QUICK_TARGET):
                fp_full = _fingerprint(p)
                if fp_full in seen_full:
                    continue
                seen_full.add(fp_full)
                _direct_full_count += 1
                return TaskSpec(
                    run_id=run_id, fidelity="full", stage="warmup_direct_full_no_cts",
                    params=p,
                    make_target=(None if FULL_TARGET == "" else FULL_TARGET),
                    timeout_sec=FULL_TIMEOUT, allow_missing_metrics=False,
                    parent_run_id=None,
                    proposed_by=proposed_by,
                    proposed_strategy=proposed_strategy + "_direct_full",
                    notes="RUN_CTS=0: " + notes,
                    model=str(c.get("proposed_model", "")) if c.get("proposed_model") else "",
                )
            else:
                fp = _fingerprint_for_stage(p, "quick")
                if fp in seen_quick:
                    continue
                seen_quick.add(fp)
                return TaskSpec(
                    run_id=run_id, fidelity="quick", stage="warmup_quick",
                    params=p,
                    make_target=(None if QUICK_TARGET == "" else QUICK_TARGET),
                    timeout_sec=QUICK_TIMEOUT, allow_missing_metrics=True,
                    parent_run_id=None,
                    proposed_by=proposed_by,
                    proposed_strategy=proposed_strategy,
                    notes=notes,
                    model=str(c.get("proposed_model", "")) if c.get("proposed_model") else "",
                )
        return None

    _promo_ranked_cache: List[Tuple[str, Dict[str, Any], str, float, float]] = []
    #                          fp,   params,          parent_id, ehvi, time_ucb
    ehvi_agent = None
    def _pick_promotion_full() -> Optional[TaskSpec]:
        nonlocal launched

        if not promo_pool:
            return None

        cand = []
        rem = max(1.0, compute_budget_remaining())                  # Changed: use compute-budget remaining.
        for fp, (params, parent_id, _quick_m) in promo_pool.items():
            fp_full = _fingerprint(params)
            if fp_full in seen_full:
                continue
            mean_t, std_t = rt.full.predict(params)
            time_ucb = float(max(1.0, mean_t + RISK_BETA * std_t))
            time_ucb = min(time_ucb, float(FULL_TIMEOUT))           # Added: cap the UCB.
            if time_ucb > rem:
                continue
            cand.append((fp, params, parent_id, time_ucb))

        if not cand:
            return None

        if not USE_EHVI:
            # EHVI ablation: retain the common runtime-budget feasibility gate
            # above, then promote uniformly at random from the whole feasible
            # quick pool.  Return before constructing or calling an EHVI agent.
            fp, params, parent_id, time_ucb = random.choice(cand)
            promo_pool.pop(fp, None)

            fp_full = _fingerprint(params)
            if fp_full in seen_full:
                return None
            seen_full.add(fp_full)

            run_id = f"iter_{launched}"
            return TaskSpec(
                run_id=run_id,
                fidelity="full",
                stage="promote_full_random",
                params=params,
                make_target=(None if FULL_TARGET == "" else FULL_TARGET),
                timeout_sec=FULL_TIMEOUT,
                allow_missing_metrics=False,
                parent_run_id=parent_id,
                proposed_by="promote_random",
                proposed_strategy="Random promotion (EHVI ablation).",
                notes="Uniform random promotion from the runtime-feasible quick pool; EHVI disabled.",
                ehvi_score=0.0,
                ehvi_time_ucb=time_ucb,
            )

        cand.sort(key=lambda x: x[3])
        subset_size = max(1, min(len(cand), PROMO_BATCH * 4))
        subset = cand[:subset_size]
        params_list = [it[1] for it in subset]

        # MEMORY FIX: Lazily create ehvi_agent (import BoTorch on first promotion).
        nonlocal ehvi_agent
        if ehvi_agent is None:
            MOEHVIAgent, _ = _get_ehvi_imports()
            ehvi_agent = MOEHVIAgent(param_space)
            ehvi_agent.warmup_trials = 0
            print("[LAZY] Created MOEHVIAgent for promotion scoring")

        # Use EHVI scoring for promotion (fair comparison — all baselines use the same)
        scored = ehvi_agent.score_params_cost_aware(
            history_full_valid,
            params_list,
            rt.full,
            remaining_sec=compute_budget_remaining(),               # Changed: use compute-budget remaining.
            margin_sec=MARGIN_SEC,
            risk_beta=RISK_BETA,
        )

        best_i, best_score = None, -1e30
        for i, s in enumerate(scored):
            sc = float(s.get("score", -1e30))
            if sc > best_score:
                best_score = sc
                best_i = i

        if best_i is None or best_score <= -1e20:
            return None

        fp, params, parent_id, _ = subset[best_i]
        promo_pool.pop(fp, None)

        fp_full = _fingerprint(params)
        if fp_full in seen_full:
            return None
        seen_full.add(fp_full)

        _best_scored = scored[best_i] if best_i is not None else {}
        _ehvi_val = float(_best_scored.get("ehvi", 0.0))
        _time_ucb_val = float(_best_scored.get("time_ucb", 0.0))

        run_id = f"iter_{launched}"
        return TaskSpec(
            run_id=run_id,
            fidelity="full",
            stage="promote_full",
            params=params,
            make_target=(None if FULL_TARGET == "" else FULL_TARGET),
            timeout_sec=FULL_TIMEOUT,
            allow_missing_metrics=False,
            parent_run_id=parent_id,
            proposed_by="promote",
            proposed_strategy="Promoted from quick pool using EHVI/time scoring.",
            notes="Promoted from quick pool using EHVI/time scoring.",
            ehvi_score=_ehvi_val,
            ehvi_time_ucb=_time_ucb_val,
        )


    # ------------ main scheduling loop ------------
    print("=== EDATuner ===")
    print("DASHSCOPE_API_KEY set: %s | PROVIDER_API_KEY set: %s | KA ready: %s" % (str(bool(os.getenv("DASHSCOPE_API_KEY"))), str(bool(os.getenv("PROVIDER_API_KEY"))), str(ka.is_ready())))
    print("quick_target=%s full_target=%s" % (QUICK_TARGET, (FULL_TARGET if FULL_TARGET else "<default>")))
    print("max_concurrency=%d total_hours=%.2f stop_starting_sec=%d" % (MAX_CONCURRENCY, TOTAL_SECONDS / 3600.0, STOP_STARTING_SEC))
    print("tuning_mode=%s" % TUNING_MODE)
    if IS_INNOVUS_ONLY:
        print("  innovus_only: max_iterations=%d est_innovus_sec=%.0f" % (MAX_ITERATIONS, EST_INNOVUS_SEC))


    # Initialize the global fixed HV reference point.
    if history_full_valid:
        _global_ref_point = compute_global_ref_point(history_full_valid)
    else:
        _global_ref_point = [-5e6, -10.0, -5.0]
    print(f"[HV] Global reference point: {_global_ref_point}")
    # ============================================================
    # Innovus-only mode: run initial Genus (or reuse existing)
    # ============================================================
    _innovus_only_genus_ok = False
    if IS_INNOVUS_ONLY:
        print("=== INNOVUS-ONLY MODE ===")
        print(f"max_iterations={MAX_ITERATIONS} estimated_innovus_sec={EST_INNOVUS_SEC}")

        if SHARED_GENUS_RUN_ID:
            # Reuse existing genus result
            _existing_genus_remote_dir = "/".join([
                runner.remote_root_base.rstrip("/"),
                runner.project_slug,
                slugify(SHARED_GENUS_RUN_ID),
                "outbox/cadence/genus",
            ]) if hasattr(runner, 'remote_root_base') else ""
            if _existing_genus_remote_dir:
                runner.set_shared_genus_dir(_existing_genus_remote_dir)
                _innovus_only_genus_ok = True
                print(f"[INNOVUS-ONLY] Reusing genus from run_id={SHARED_GENUS_RUN_ID}")
            else:
                print("[INNOVUS-ONLY][ERROR] Cannot determine remote genus dir for shared_genus_run_id")
        else:
            # Check if genus_shared already succeeded in a previous session
            _genus_already_done = any(
                r.get("run_id") == "genus_shared" and r.get("success")
                for r in history_all
            )
            if _genus_already_done:
                _existing_dir = "/".join([
                    runner.remote_root_base.rstrip("/"),
                    runner.project_slug,
                    "genus_shared",
                    "outbox/cadence/genus",
                ]) if hasattr(runner, 'remote_root_base') else ""
                if _existing_dir:
                    runner.set_shared_genus_dir(_existing_dir)
                    _innovus_only_genus_ok = True
                    print(f"[INNOVUS-ONLY] Genus already done in previous session, reusing: {_existing_dir}")
                else:
                    print("[INNOVUS-ONLY][WARN] genus_shared in history but cannot determine remote dir, re-running...")
                    _genus_already_done = False

            if not _genus_already_done:
                # Run Genus once with default params
                print("[INNOVUS-ONLY] Running initial Genus with default params...")
                try:
                    _genus_ok, _genus_log, _genus_remote_dir = runner.run_initial_genus(
                        params=default_params,
                        run_id="genus_shared",
                        timeout_sec=FULL_TIMEOUT,
                    )
                    _innovus_only_genus_ok = _genus_ok
                    if _genus_ok:
                        print(f"[INNOVUS-ONLY] Initial Genus succeeded: {_genus_remote_dir}")
                        # Treat the initial full run as a warmup data point
                        _genus_metrics = runner.get_metrics(_genus_log)
                        _genus_cls = _classify_metrics(_genus_metrics, fidelity="full")
                        _genus_res = {
                            "run_id": "genus_shared",
                            "parent_run_id": None,
                            "fidelity": "full",
                            "stage": "innovus_only_genus_warmup",
                            "params": default_params,
                            "metrics": _genus_metrics,
                            "success": True,
                            "metrics_status": _genus_cls["metrics_status"],
                            "metrics_usable": _genus_cls["metrics_usable"],
                            "metrics_valid": _genus_cls["metrics_valid"],
                            "log_path": _genus_log,
                            "runtime_sec": 0.0,
                            "proposed_by": "initial_genus",
                            "proposed_strategy": "innovus_only_warmup",
                            "proposed_notes": "Initial Genus+Innovus run for innovus-only mode",
                            "proposed_model": "",
                            "log_summary": {},
                            "fail_log_tail": "",
                        }
                        _slim_history_entry_main(_genus_res)
                        history_all.append(_genus_res)
                        kb.append_run(_genus_res, include_log_tail=False)
                        if _genus_cls["metrics_valid"]:
                            history_full_valid.append(_genus_res)
                        print(f"[INNOVUS-ONLY] Warmup point added: valid={_genus_cls['metrics_valid']}")
                    else:
                        print("[INNOVUS-ONLY][FATAL] Initial Genus failed, cannot proceed")
                        _trigger_fatal_stop("Initial Genus failed in innovus-only mode")
                except Exception as _ge:
                    print(f"[INNOVUS-ONLY][FATAL] Initial Genus exception: {_ge}")
                    _trigger_fatal_stop(f"Initial Genus exception: {_ge}")

        if IS_INNOVUS_ONLY and not _innovus_only_genus_ok:
            print("[INNOVUS-ONLY] Genus not available, falling back to full mode for this session")
            # Don't actually change IS_INNOVUS_ONLY — just let fatal_stop prevent launches


    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:

        # ============================================================
        # NEW: Re-attach inflight runs from resume_state.
        # runner.run() will resume automatically from the polling phase if it finds a local checkpoint.
        # ============================================================
        if _resume_inflight_runs:
            print("[RESUME-INFLIGHT] Re-attaching %d inflight runs to thread pool ..."
                  % len(_resume_inflight_runs))
            for _ifr in _resume_inflight_runs:
                if not isinstance(_ifr, dict):
                    continue
                _if_rid = str(_ifr.get("run_id", ""))
                _if_params = _ifr.get("params", {})
                _if_fidelity = str(_ifr.get("fidelity", "quick"))
                _if_stage = str(_ifr.get("stage", "explore_quick"))
                if not _if_rid:
                    print("  [WARN] skipping inflight run with missing run_id")
                    continue
                # If params are missing, try to recover them from the local checkpoint.
                if not _if_params:
                    _ifr_ckpt_path = Path(cfg["work_dir"]) / _if_rid / "checkpoint.json"
                    if _ifr_ckpt_path.is_file():
                        try:
                            _ifr_ckpt = json.loads(_ifr_ckpt_path.read_text(encoding="utf-8"))
                            _if_params = _ifr_ckpt.get("params", {})
                        except Exception:
                            pass
                if not _if_params:
                    print("  [WARN] skipping %s: no params available" % _if_rid)
                    continue

                _if_spec = TaskSpec(
                    run_id=_if_rid,
                    fidelity=_if_fidelity,
                    stage=_if_stage,
                    params=_if_params,
                    make_target=(None if _if_fidelity == "quick"
                                 else (None if FULL_TARGET == "" else FULL_TARGET)),
                    timeout_sec=(QUICK_TIMEOUT if _if_fidelity == "quick" else FULL_TIMEOUT),
                    allow_missing_metrics=(_if_fidelity == "quick"),
                    proposed_by="resume_inflight",
                    proposed_strategy="re-attach",
                    notes=("re-attached from checkpoint phase=%s"
                           % _ifr.get("checkpoint_phase", "?")),
                )
                _if_spec.launch_wall_sec = time.time() - start_wall
                # runner.run() detects a jobid in the local checkpoint and resumes polling automatically.
                _if_fut = ex.submit(_evaluate_one, runner, _if_spec)
                inflight[_if_fut] = _if_spec
                # Update the deduplication sets.
                _if_fp = _ifr.get("params_fingerprint", "")
                if _if_fp:
                    if _if_fidelity == "full":
                        seen_full.add(_if_fp)
                    else:
                        seen_quick.add(_if_fp)
                print("  [OK] re-attached %s (fidelity=%s stage=%s "
                      "genus_job=%s innovus_job=%s)"
                      % (_if_rid, _if_fidelity, _if_stage,
                         _ifr.get("genus_jobid", "n/a"),
                         _ifr.get("innovus_jobid", "n/a")))
            print("[RESUME-INFLIGHT] %d runs re-attached, %d slots free"
                  % (len(inflight), MAX_CONCURRENCY - len(inflight)))
            _resume_inflight_runs = []  # Clear after use to avoid duplicate re-attachment.

        while inflight or can_launch_new():
            _check_stop_file()
            # fill workers
            while len(inflight) < MAX_CONCURRENCY and can_launch_new():
                # Decide which task to launch

                # (A) warmup_quick: only run CTS/quick first
                if (not warmup_quick_done) and (warmup_quick_success < WARMUP_QUICK):
                    spec = _pick_warmup_quick()

                    if spec is None:
                        break

                else:
                    warmup_quick_done = True

                    # (B) warmup_full: use Sobol space filling for early full runs,
                    # without relying on promo_pool, because GP data is still insufficient and promotion is blind.
                    if len(history_full_valid) < agent.warmup_trials:
                        # NEW [Change 5a]: use Sobol for full runs in early warmup.
                        if len(history_full_valid) < max(6, agent.warmup_trials // 2):
                            # Generate full points directly from Sobol.
                            for _sobol_try in range(20):
                                _p = _sobol_to_params()
                                _fp_f = _fingerprint(_p)
                                if _fp_f not in seen_full:
                                    seen_full.add(_fp_f)
                                    run_id = "iter_%d" % launched
                                    spec = TaskSpec(
                                        run_id=run_id, fidelity="full",
                                        stage="warmup_full_sobol",
                                        params=_p,
                                        make_target=(None if FULL_TARGET == "" else FULL_TARGET),
                                        timeout_sec=FULL_TIMEOUT,
                                        allow_missing_metrics=False,
                                        parent_run_id=None,
                                        proposed_by="Sobol",
                                        proposed_strategy="space_filling",
                                        notes="Sobol warmup full (avoids blind promotion)",
                                    )
                                    break
                            else:
                                spec = None
                        else:
                            # In later warmup, GP has initial data, so promotion can be used.
                            spec = _pick_promotion_full()

                        if spec is None:
                            spec = _pick_next_quick_from_ka()

                    # (C) normal phase: mix promote/full and quick
                    else:
                        _effective_promote_prob = PROMOTE_PROB
                        if len(promo_pool) > 3 * max(len(history_full_valid), 1):
                            _effective_promote_prob = min(0.7, PROMOTE_PROB + 0.2)
                        elif len(promo_pool) > 2 * max(len(history_full_valid), 1):
                            _effective_promote_prob = min(0.6, PROMOTE_PROB + 0.1)
                        do_promote = (len(promo_pool) > 0) and (random.random() < _effective_promote_prob)
                        if do_promote:
                            spec = _pick_promotion_full()
                            if spec is None:
                                spec = _pick_next_quick_from_ka()
                        else:
                            spec = _pick_next_quick_from_ka()

                    if spec is None:
                        break
                _print_launch(spec)

                spec.launch_wall_sec = time.time() - start_wall   # NEW
                token_tracker.begin_run(spec.run_id)
                fut = ex.submit(_evaluate_one, runner, spec)
                inflight[fut] = spec
                launched += 1
                if ka.is_ready() and ("KA" in str(spec.proposed_by) or "KA" in str(spec.proposed_strategy)):
                    ka.record_proposed([spec.run_id])

            if not inflight:
                break

            if time.time() > deadline and not IS_INNOVUS_ONLY:
                print("[OVERTIME] deadline passed, waiting for %d inflight tasks to finish..." % len(inflight))

            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)


            for fut in done:
                spec = inflight.pop(fut)
                try:
                    res = fut.result()
                    # print(f"[DEBUG] res = {res}")
                except _FATAL_REMOTE_EXCEPTIONS as e:
                    # FIX: Distinguish recoverable connection issues from true scheduler failures.
                    if isinstance(e, (SchedulerBackpressureError, ControlledStop)):
                        # Scheduler failures or user-triggered stops remain fatal.
                        _trigger_fatal_stop(f"{type(e).__name__}: {str(e)[:200]}")
                    else:
                        # RemoteCommError / RemoteSessionLostError are not fatal; mark only this run as failed.
                        # The system keeps running; the next run will establish a new connection.
                        print(f"[WARN] Run {spec.run_id} lost connection ({type(e).__name__}), "
                              f"marking as failed but NOT stopping experiment. "
                              f"Will retry with new connection on next iteration.")
                    print("[ERROR] Run %s failed with %s: %s" % (spec.run_id, type(e).__name__, str(e)[:200]))
                    res = {
                        "run_id": spec.run_id, "parent_run_id": spec.parent_run_id,
                        "fidelity": spec.fidelity, "stage": spec.stage, "params": spec.params,
                        "metrics": {
                            "area": float("inf"), "wns": -100.0, "power": float("inf"),
                            "__area_sentinel__": True,
                            "__wns_sentinel__": True,
                            "__power_sentinel__": True,
                        },
                        "success": False,
                        "metrics_status": "sentinel",
                        "metrics_usable": False,
                        "metrics_valid": False,
                        "log_path": "", "runtime_sec": 0.0,
                        "proposed_by": spec.proposed_by, "proposed_strategy": spec.proposed_strategy,
                        "proposed_notes": spec.notes, "proposed_model": spec.model,
                        "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                    }
                except Exception as e:
                    print("[ERROR] Run %s failed with exception: %s" % (spec.run_id, str(e)))
                    res = {
                        "run_id": spec.run_id, "parent_run_id": spec.parent_run_id,
                        "fidelity": spec.fidelity, "stage": spec.stage, "params": spec.params,
                        "metrics": {
                            "area": float("inf"), "wns": -100.0, "power": float("inf"),
                            "__area_sentinel__": True,
                            "__wns_sentinel__": True,
                            "__power_sentinel__": True,
                        },
                        "success": False,
                        "metrics_status": "sentinel",
                        "metrics_usable": False,
                        "metrics_valid": False,
                        "log_path": "", "runtime_sec": 0.0,
                        "proposed_by": spec.proposed_by, "proposed_strategy": spec.proposed_strategy,
                        "proposed_notes": spec.notes, "proposed_model": spec.model,
                        "error": "%s: %s" % (type(e).__name__, str(e)),
                    }
                # NEW: Guardrail for repeated connection failures.
                _is_conn_err = (
                    "RemoteCommError" in str(res.get("error", ""))
                    or "RemoteSessionLostError" in str(res.get("error", ""))
                )
                _on_run_result(
                    success=res.get("success", False),
                    is_conn_error=_is_conn_err,
                )

                # ---- QoR delta vs parent (for promotion/full, or any parent-linked runs) ----
                try:
                    pid = res.get("parent_run_id", None)
                    if pid:
                        parent = next((h for h in history_all if h.get("run_id") == pid), None)
                        if parent and parent.get("success", False) and res.get("success", False):
                            pm = parent.get("metrics", {}) or {}
                            cm = res.get("metrics", {}) or {}

                            # only if metrics are sane
                            pa, pw, pp = float(pm.get("area", float("inf"))), float(pm.get("wns", -100.0)), float(pm.get("power", float("inf")))
                            ca, cw, cp = float(cm.get("area", float("inf"))), float(cm.get("wns", -100.0)), float(cm.get("power", float("inf")))

                            # FIX: Also check sentinel markers, not just value-based checks.
                            _pm_sentinel = any(pm.get(f"__{k}_sentinel__", False) for k in ("area", "wns", "power"))
                            _cm_sentinel = any(cm.get(f"__{k}_sentinel__", False) for k in ("area", "wns", "power"))
                            if (not _pm_sentinel and not _cm_sentinel
                                    and pa != float("inf") and ca != float("inf")
                                    and pw > -99.0 and cw > -99.0
                                    and pp != float("inf") and cp != float("inf")):
                                delta = {"wns": cw - pw, "area": ca - pa, "power": cp - pp}
                                regressed = (delta["wns"] < 0.0) or (delta["area"] > 0.0) or (delta["power"] > 0.0)

                                ls = res.get("log_summary", {}) or {}
                                if isinstance(ls, dict):
                                    st = ls.get("stage", {}) or {}
                                    if isinstance(st, dict):
                                        st["qor_vs_parent"] = delta
                                        st["qor_regressed"] = bool(regressed)
                                        ls["stage"] = st
                                        res["log_summary"] = ls
                except Exception:
                    pass

                _end_wall = time.time()

                # (a) Wall-clock timing
                res["wall_start_sec"] = spec.launch_wall_sec
                res["wall_end_sec"]   = _end_wall - start_wall
                res["experiment_ts"]  = start_wall       # absolute epoch of experiment start

                # (b) Ablation / experiment tag
                res["experiment_id"]  = cfg.get("project_name", "unknown")

                # (c) Cumulative counts (BEFORE appending this run)
                res["cum_launched"]       = launched
                res["cum_history_all"]    = len(history_all) + 1  # +1 counting self
                res["cum_full_valid"]     = len(history_full_valid)
                res["cum_quick_usable"]   = warmup_quick_success
                res["cum_promo_pool_size"] = len(promo_pool)

                # (d) Cumulative success/fail counts
                _cum_success = sum(1 for h in history_all if h.get("success", False))
                _cum_fail    = sum(1 for h in history_all if not h.get("success", False))
                if res.get("success", False):
                    _cum_success += 1
                else:
                    _cum_fail += 1
                res["cum_success_count"] = _cum_success
                res["cum_fail_count"]    = _cum_fail

                # (e) Failure category
                res["failure_category"] = _classify_failure(res)

                # (f) EHVI score (from TaskSpec, only meaningful for promotions)
                res["ehvi_score"]    = spec.ehvi_score
                res["ehvi_time_ucb"] = spec.ehvi_time_ucb

                # (g) Hypervolume tracking
                #     Temporarily append to full_valid to compute HV, then pop it back out.
                _tmp_fv = list(history_full_valid)
                if spec.fidelity == "full" and res.get("metrics_valid", False):
                    _tmp_fv.append(res)

                _hv_after = _compute_hv(_tmp_fv, ref_point_list=_global_ref_point)
                _hv_before = _compute_hv(history_full_valid, ref_point_list=_global_ref_point)
                res["cum_hv"]       = _hv_after
                res["cum_hv_delta"] = _hv_after - _hv_before

                # (h) Pareto status
                _pareto_snap_now = _pareto_snapshot(
                    _tmp_fv if _tmp_fv else history_all, max_items=9999
                )
                _pareto_ids_now = set(p.get("run_id") for p in _pareto_snap_now)
                res["is_pareto"]   = (res.get("run_id") in _pareto_ids_now)
                res["pareto_size"] = len(_pareto_snap_now)
                res["cum_full_valid"] = len(_tmp_fv)   # update with self if applicable

                # (i) Knowledge state snapshot
                try:
                    _n_hard = len(ka._load_hard_rules(limit=None)) if getattr(ka, '_persist_rules_enabled', True) else 0
                except Exception:
                    _n_hard = 0
                try:
                    _n_soft = len(ka._load_soft_heuristics(limit=None)) if getattr(ka, '_persist_rules_enabled', True) else 0
                except Exception:
                    _n_soft = 0
                res["num_hard_rules"]       = _n_hard
                res["num_soft_heuristics"]  = _n_soft
                res["ka_buffer_size"]       = len(ka_buffer)
                res["design_prior_len"]     = len(getattr(ka, "design_prior_text", "") or "")
                res["failure_analysis_len"] = len(getattr(ka, "failure_analysis_text", "") or "")
                res["sensitivity_text_len"] = len(getattr(ka, "sensitivity_text", "") or "")

                # (j) Runtime model confidence (how well we predict runtimes)
                try:
                    _rt_mean, _rt_std = rt.quick.predict(fill_params(spec.params)) if spec.fidelity == "quick" else rt.full.predict(fill_params(spec.params))
                    res["predicted_runtime_mean"] = float(_rt_mean)
                    res["predicted_runtime_std"]  = float(_rt_std)
                    _actual_rt = float(res.get("runtime_sec", 0.0))
                    if _actual_rt > 0 and _rt_mean > 0:
                        res["runtime_prediction_error"] = abs(_actual_rt - _rt_mean) / _actual_rt
                    else:
                        res["runtime_prediction_error"] = None
                except Exception:
                    res["predicted_runtime_mean"] = None
                    res["predicted_runtime_std"]  = None
                    res["runtime_prediction_error"] = None

                _slim_history_entry_main(res)  # MEM FIX
                history_all.append(res)
                try:
                    _token_stats = token_tracker.get_global_stats()
                    res["llm_total_tokens_so_far"] = _token_stats.get("llm_total_tokens", 0)
                    res["llm_total_calls_so_far"] = _token_stats.get("llm_total_calls", 0)
                except Exception:
                    pass
                kb.append_run(res, include_log_tail=True)
                # TOKEN-OPT: Call update_feedback only once (it used to be called three times).
                if ka.is_ready() and ("KA" in str(res.get("proposed_by", "")) or "KA" in str(res.get("proposed_strategy", ""))):
                    ka.update_feedback(res, is_pareto=bool(res.get("is_pareto", False)))

                # update runtime surrogate
                if res.get("success", False):
                    rt.observe(spec.fidelity, fill_params(spec.params), res.get("runtime_sec", 0.0))
                    rt.fit(spec.fidelity)

                    # ---- Innovus-only: feed actual runtime to adaptive estimator ----
                    if IS_INNOVUS_ONLY and ibe is not None:
                        _actual_rt = float(res.get("runtime_sec", 0.0))
                        if _actual_rt > 0:
                            ibe.observe(_actual_rt)


                # ---- quick usable -> promotion pool ----
                # FIX [Change 4a]: Use _fingerprint() consistently when inserting into and removing from the pool.
                if spec.fidelity == "quick" and res.get("metrics_usable", False):
                    warmup_quick_success += 1

                    # FIX: Store quick results in promo_pool (this block was previously misplaced under an elif branch).
                    filled_p = fill_params(spec.params)
                    _skip_promo = False

                    fp = _fingerprint(filled_p)
                    if (not _skip_promo
                            and fp not in promo_pool
                            and fp not in seen_full
                            and len(promo_pool) < MAX_PROMO_POOL):
                        quick_m = res.get("metrics", {}) or {}
                        promo_pool[fp] = (filled_p, spec.run_id, quick_m)

                # FIX: Count direct-to-full toward warmup progress as well.
                if (spec.fidelity == "full"
                      and "direct_full" in str(spec.stage)
                      and not warmup_quick_done):
                    warmup_quick_success += 1
                    # Do not insert direct_full results into promo_pool because they are already full evaluations.

                # full success + valid metrics -> QoR history
                if spec.fidelity == "full" and res.get("metrics_valid", False):
                    history_full_valid.append(res)
                    _promo_ranked_cache.clear()

                    # NEW: If HV improves or a new Pareto point appears, flush the KA buffer so the next round uses fresh context.
                    _hv_improved = (res.get("cum_hv_delta", 0) > 0)
                    _is_new_pareto = res.get("is_pareto", False)
                    if _hv_improved or _is_new_pareto:
                        _stale_count = len(ka_buffer)
                        if _stale_count > 0:
                            ka_buffer.clear()
                            print(f"[KA] Buffer flushed ({_stale_count} stale candidates) — "
                                  f"HV improved={_hv_improved}, new_pareto={_is_new_pareto}")

                    # NEW [Change 6a]: Record calibration pairs (quick -> full).
                    # FIX: Do not rely on parent_run_id (older resumed entries may have an empty parent).
                    #        Match by fingerprint instead and find the quick result with the same parameters.
                    if USE_EHVI:
                        _full_fp = _fingerprint(fill_params(spec.params))
                        _found_quick_parent = False

                        # Method 1: exact lookup via parent_run_id.
                        if spec.parent_run_id:
                            for _ph in reversed(history_all):
                                if _ph.get("run_id") == spec.parent_run_id:
                                    _parent_m = _ph.get("metrics", {}) or {}
                                    _full_m = res.get("metrics", {}) or {}
                                    if _parent_m and _full_m:
                                        agent.record_calibration_pair(
                                            params_fp=_full_fp,
                                            quick_metrics=_parent_m,
                                            full_metrics=_full_m,
                                        )
                                        _found_quick_parent = True
                                    break

                        # Method 2: if parent_run_id is empty, match the most recent quick run by fingerprint.
                        if not _found_quick_parent:
                            for _ph in reversed(history_all):
                                if (_ph.get("fidelity") == "quick"
                                        and _ph.get("metrics_usable", False)
                                        and isinstance(_ph.get("params"), dict)
                                        and _ph["params"]):
                                    _ph_fp = _fingerprint(fill_params(_ph["params"]))
                                    if _ph_fp == _full_fp:
                                        _parent_m = _ph.get("metrics", {}) or {}
                                        _full_m = res.get("metrics", {}) or {}
                                        if _parent_m and _full_m:
                                            agent.record_calibration_pair(
                                                params_fp=_full_fp,
                                                quick_metrics=_parent_m,
                                                full_metrics=_full_m,
                                            )
                                        break


                _print_done(res)

                print(
                    "        rem=%.2fh inflight=%d launched=%d promo_pool=%d full_valid=%d"
                    % (
                        remaining_sec() / 3600.0,
                        len(inflight),
                        launched,
                        len(promo_pool),
                        len(history_full_valid),
                    )
                )

                # ---- Innovus-only: iteration progress ----
                if IS_INNOVUS_ONLY and ibe is not None:
                    _iter_pct = launched / max(1, MAX_ITERATIONS) * 100
                    _est_remain = ibe.remaining_sec(max(0, MAX_ITERATIONS - launched))
                    print("        [iter] %d/%d (%.1f%%) est_remain=%.0fs (%.1fh) eff_est=%.0fs/iter"
                          % (launched, MAX_ITERATIONS, _iter_pct,
                             _est_remain, _est_remain / (3600.0 * max(1, MAX_CONCURRENCY)),
                             ibe.effective_est_sec))


                # ---- Periodic analysis ----
                # FIX: Raise the activation threshold to avoid noisy analysis on sparse data.
                if (PERSIST_RULES and ka.is_ready()
                        and len(history_all) - last_analysis_count >= ANALYSIS_INTERVAL
                        and len(history_all) >= ANALYSIS_MIN_HISTORY
                        and len(history_full_valid) >= ANALYSIS_MIN_FULL_VALID):
                    last_analysis_count = len(history_all)
                    try:
                        fa = analyze_failure_patterns(history_all, param_space)
                        fa_text = format_failure_analysis_for_prompt(fa)
                        if fa_text:
                            ka.failure_analysis_text = fa_text
                            # RAG-enhanced failure analysis.
                            if rag.is_ready:
                                symptoms = list(fa.get("pattern_counts", {}).keys())[:5]
                                rag_fa = rag.context_for_failure_analysis(symptoms)
                                if rag_fa:
                                    ka.failure_analysis_text += "\n\n" + rag_fa
                            print("[Analysis] failure patterns: %d patterns, %d insights"
                                  % (len(fa.get("pattern_counts", {})),
                                     len(fa.get("top_insights", []))))
                        

                        sa = compute_parameter_sensitivity(history_full_valid, param_space)
                        te = detect_threshold_effects(history_all, param_space)
                        te_text = format_threshold_effects_for_prompt(te)
                        if te_text:
                            # Append after sensitivity_text.
                            ka.sensitivity_text = (ka.sensitivity_text or "") + "\n" + te_text
                            print("[Analysis] threshold effects: %d detected" % len(te.get("threshold_effects", {})))
                        sa_text = format_sensitivity_for_prompt(sa)
                        if sa_text:
                            ka.sensitivity_text = sa_text
                            # RAG-enhanced parameter sensitivity.
                            if rag.is_ready:
                                top_params = sa.get("ranking", [])[:5]
                                rag_sens = rag.context_for_sensitivity(*top_params)
                                if rag_sens:
                                    ka.sensitivity_text += "\n\n" + rag_sens
                            print("[Analysis] sensitivity ranking=%s"
                                  % str(sa.get("ranking", [])))
                    except Exception as e:
                        print("[Analysis] periodic analysis failed: %s" % str(e))

                # ---- Periodic prior refinement (Section 3.1) ----
                # FIX: Raise the threshold and require at least PRIOR_REFINE_MIN_FULL_VALID full-valid points.
                if (PERSIST_RULES and ka.is_ready()
                        and len(history_all) - last_refine_count >= PRIOR_REFINE_INTERVAL
                        and len(history_full_valid) >= PRIOR_REFINE_MIN_FULL_VALID):
                    last_refine_count = len(history_all)
                    try:
                        pareto_snap = _pareto_snapshot(
                            history_full_valid if history_full_valid else history_all,
                            max_items=8,
                        )
                        updated_prior = prior_builder.refine_with_history(
                            current_prior=ka.design_prior_text,
                            history_recent=history_all[-30:],
                            pareto=pareto_snap,
                            failure_summary=getattr(ka, "failure_analysis_text", ""),
                            sensitivity_summary=getattr(ka, "sensitivity_text", ""),
                        )
                        ka.design_prior_text = updated_prior
                        # RAG-enhance the updated prior.
                        if rag.is_ready:
                            latest_metrics = history_full_valid[-1].get("metrics", {}) if history_full_valid else {}
                            rag_ctx = rag.context_for_design_prior(
                                current_params=default_params,
                                metrics=latest_metrics,
                            )
                            if rag_ctx:
                                ka.design_prior_text += "\n\n" + rag_ctx
                        print("[Analysis] prior refined (%d chars)" % len(updated_prior))
                    except Exception as e:
                        print("[Analysis] prior refine failed: %s" % str(e))

                # ---- NEW: Periodic rule cleanup ----
                if (PERSIST_RULES and ka.is_ready()
                        and len(history_all) - last_rule_trim_count >= KP_HARD_RULE_TRIM_INTERVAL):
                    last_rule_trim_count = len(history_all)
                    try:
                        ka.trim_hard_rules(keep_latest=KP_MAX_ACTIVE_HARD_RULES)
                        ka.trim_soft_heuristics()
                    except Exception as e:
                        print(f"[KP] rule trim failed: {e}")

                if CLEANUP_EVERY > 0 and (len(history_all) % CLEANUP_EVERY == 0):
                    try:
                        runner.cleanup(history_all)
                    except Exception:
                        pass
                    # MEM FIX: Periodically force GC.
                    gc.collect()

    if fatal_stop_requested:
        print("[FATAL] scheduling stopped: %s" % fatal_stop_reason)

    # finalize exports
    try:
        runner.cleanup(history_all)
    except Exception:
        pass

    pareto_path = kb.export_pareto(history_full_valid)
    csv_path = kb.export_csv(history_all)

    report_path = generate_llm_report(
        ka=ka,
        cfg=cfg,
        pareto_json_path=pareto_path,
        history_all=history_all,
        history_full_valid=history_full_valid,
    )

    _final_hv = _compute_hv(history_full_valid, ref_point_list=_global_ref_point)
    _final_pareto = _pareto_snapshot(history_full_valid if history_full_valid else history_all, max_items=9999)
    _total_wall_sec = time.time() - start_wall
    _total_success = sum(1 for h in history_all if h.get("success", False))
    _total_fail = len(history_all) - _total_success

    experiment_summary = {
        "experiment_id": cfg.get("project_name", "unknown"),
        "config_path": config_path,
        "total_wall_sec": _total_wall_sec,
        "total_wall_hours": _total_wall_sec / 3600.0,
        "total_launched": launched,
        "total_history": len(history_all),
        "total_full_valid": len(history_full_valid),
        "total_success": _total_success,
        "total_fail": _total_fail,
        "final_failure_rate": _total_fail / max(1, len(history_all)),
        "final_hv": _final_hv,
        "final_pareto_size": len(_final_pareto),
        "final_promo_pool_size": len(promo_pool),
        # knowledge state
        "final_hard_rules": len(ka._load_hard_rules(limit=None)) if getattr(ka, '_persist_rules_enabled', True) else 0,
        "final_soft_heuristics": len(ka._load_soft_heuristics(limit=None)) if getattr(ka, '_persist_rules_enabled', True) else 0,
        "final_design_prior_len": len(getattr(ka, "design_prior_text", "") or ""),
        # ablation flags
        "use_ehvi": USE_EHVI,
        "ka_enabled": bool(ka.is_ready()),
        "persist_rules": PERSIST_RULES,
        # innovus-only mode info
        "tuning_mode": TUNING_MODE,
        "max_iterations": MAX_ITERATIONS if IS_INNOVUS_ONLY else None,
        "estimated_innovus_sec": EST_INNOVUS_SEC if IS_INNOVUS_ONLY else None,
        "final_effective_est_sec": ibe.effective_est_sec if (IS_INNOVUS_ONLY and ibe) else None,
        "ibe_observed_count": len(ibe._observed) if (IS_INNOVUS_ONLY and ibe) else None,

        # best points
        "best_wns": max((float(h["metrics"].get("wns", -1e9)) for h in history_full_valid if h.get("metrics")), default=None),
        "best_area": min((float(h["metrics"].get("area", 1e9)) for h in history_full_valid if h.get("metrics")), default=None),
        "best_power": min((float(h["metrics"].get("power", 1e9)) for h in history_full_valid if h.get("metrics")), default=None),
        # pareto points detail
        "pareto_points": [
            {
                "run_id": p.get("run_id"),
                "area": p.get("metrics", {}).get("area"),
                "wns": p.get("metrics", {}).get("wns"),
                "power": p.get("metrics", {}).get("power"),
            }
            for p in _final_pareto
        ],
        # HV trajectory (for quick re-plotting without parsing full jsonl)
        "hv_trajectory": [
            {
                "wall_end_sec": h.get("wall_end_sec"),
                "cum_hv": h.get("cum_hv"),
                "run_id": h.get("run_id"),
            }
            for h in history_all
            if h.get("cum_hv") is not None and h.get("wall_end_sec") is not None
        ],
        **token_tracker.get_global_stats(),
    }

    _summary_path = Path(cfg["work_dir"]) / "experiment_summary.json"
    try:
        _summary_path.write_text(
            json.dumps(experiment_summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8"
        )
        print("Experiment summary: %s" % str(_summary_path))
    except Exception as _e:
        print("[WARN] failed to write experiment_summary.json: %s" % str(_e))

    print("\n=== Finished ===")
    print("All finished: %d | Full valid: %d | Promo left: %d" % (len(history_all), len(history_full_valid), len(promo_pool)))
    print("Pareto(full): %s" % pareto_path)
    print("Summary(all): %s" % csv_path)
    print("Final report: %s" % report_path)
    print("KB: %s" % str(kb.runs_jsonl))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        "-c",
        default="configs/config.yaml",
        help="Path to YAML config (e.g. configs/gdb_config.yaml)",
    )
    args = ap.parse_args()
    main(args.config)
