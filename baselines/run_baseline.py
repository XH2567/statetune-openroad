#!/usr/bin/env python3
"""
baselines/run_baseline.py
=========================
Baseline experiment runner for comparing against EDATuner (LLM-Agent system).
Supports three strategies:
  1. bo        — Multi-objective Bayesian Optimization (qEHVI, same surrogate as EDATuner
                 but without LLM-guided candidate proposal or knowledge persistence)
  2. optuna    — Optuna TPE multi-objective sampler
  3. random    — Pure random search

All baselines share the **same** evaluation pipeline (remote Cadence runner,
multi-fidelity quick→full promotion, KB logging, HV tracking) so the comparison
is fair: only the candidate-proposal mechanism differs.

Usage:
    python -m baselines.run_baseline --config configs/baseline_bo.yaml
    python -m baselines.run_baseline --config configs/baseline_optuna.yaml
    python -m baselines.run_baseline --config configs/baseline_random.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Tuple

import gc
import torch

# ── Shared EDATuner infrastructure (we reuse everything except the LLM agent) ──
from src.EDATuner.utils.utils import load_config
from src.EDATuner.knowledge.kb import KB
from src.EDATuner.optimization.runtime_model import MultiFidelityRuntimePredictor, IterationBudgetEstimator
from src.EDATuner.knowledge.log_summary import analyze_log_summary
from src.EDATuner.runners.runner_factory import create_runner
from src.EDATuner.runners.eda_interface import EDARunner

# Lazily import large components to avoid a multi-GB startup memory spike.

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


# Resilience (optional)
try:
    from src.EDATuner.runners.remote_cadence_support import (
        RemoteCommError, RemoteSessionLostError,
        RemoteCommandTimeoutError, SchedulerBackpressureError,
        ControlledStop, is_backpressure_active,
    )
    _FATAL_REMOTE_EXCEPTIONS = (
        RemoteCommError, RemoteSessionLostError,
        SchedulerBackpressureError, ControlledStop,
    )
except ImportError:
    _FATAL_REMOTE_EXCEPTIONS = ()
    is_backpressure_active = lambda: False


# Connection resilience: pause after repeated connection failures instead of terminating immediately.
_consecutive_conn_failures = 0
_MAX_CONSECUTIVE_CONN_FAILURES = 5        # Start waiting after 5 consecutive connection failures.
_CONN_FAILURE_WAIT_SEC = 300              # Wait 5 minutes before retrying.
_MAX_CONN_FAILURE_WAIT_ROUNDS = 12        # Up to 12 wait rounds (about 1 hour).

# ── Optuna (lazy import; only needed for "optuna" strategy) ──
_optuna = None

def _get_optuna():
    global _optuna
    if _optuna is None:
        import optuna
        _optuna = optuna
    return _optuna


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions (copied from main.py to keep baseline self-contained)
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


_PLACEMENT_EFFECTIVE_PARAMS = {
    "CORE_UTIL", "ASPECT_RATIO", "CORE_MARGIN",
    "PLACE_CONG_EFFORT", "PLACE_TIMING_EFFORT",
    "PLACE_UNIFORM_DENSITY", "PLACE_IO_PINS_AWARE",
    "TD_PLACE", "PLACE_MAX_DENSITY",
    "PRECTS_SETUP_OPT", "PRECTS_HOLD_OPT",
    "RUN_CTS", "CTS_TARGET_SKEW", "CTS_POST_OPT",
}


def _needs_direct_full(params: dict, quick_target: str) -> bool:
    if quick_target in ("cts",):
        run_cts = str(params.get("RUN_CTS", "1")).lower()
        if run_cts in ("0", "false"):
            return True
    return False


def _fingerprint_for_stage(params: Dict[str, Any], fidelity: str) -> str:
    if fidelity == "quick":
        filtered = {k: v for k, v in params.items() if k in _PLACEMENT_EFFECTIVE_PARAMS}
        return json.dumps(filtered, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _fingerprint(params)


def _infer_shared_genus_dir(cfg: Dict[str, Any], runner: Any, run_id: str) -> str:
    run_id = str(run_id or "").strip()
    if not run_id:
        return ""
    remote_root_base = getattr(runner, "remote_root_base", "") or ""
    project_slug = getattr(runner, "project_slug", "") or ""
    if not remote_root_base or not project_slug:
        return ""
    try:
        from src.EDATuner.runners.remote_cadence_support import slugify as _slugify
    except Exception:
        _slugify = lambda s: str(s).strip().replace(" ", "_").lower()
    return "/".join([
        str(remote_root_base).rstrip("/"),
        str(project_slug),
        _slugify(run_id),
        "outbox/cadence/genus",
    ])


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
        out[mapping.get(kk, kk)] = v
    return out


def _get_param_space(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(cfg.get("param_space", None), list):
        return cfg["param_space"]
    tuning = cfg.get("tuning", {}) or {}
    if isinstance(tuning.get("param_space", None), list):
        return tuning["param_space"]
    raise ValueError("No param_space found in config")


def _compute_hv(history_full_valid: list, ref_point_list=None) -> float:
    pts = []
    for r in history_full_valid:
        m = r.get("metrics", {}) or {}
        try:
            a, w, p = float(m["area"]), float(m["wns"]), float(m["power"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(a) and math.isfinite(w) and math.isfinite(p)):
            continue
        if w <= -99.0:
            continue
        pts.append([-a, w, -p])
    if len(pts) < 1:
        return 0.0
    Y = torch.tensor(pts, dtype=torch.double)
    ref = torch.tensor(ref_point_list or [-5e6, -10.0, -5.0], dtype=torch.double)
    try:
        from botorch.utils.multi_objective.pareto import is_non_dominated
        _, Hypervolume = _get_ehvi_imports()
        nd_mask = is_non_dominated(Y)
        pareto_Y = Y[nd_mask]
        if pareto_Y.shape[0] == 0:
            return 0.0
        return float(Hypervolume(ref).compute(pareto_Y))
    except Exception:
        return 0.0


def compute_global_ref_point(history_full_valid: list) -> list:
    pts = []
    for r in history_full_valid:
        m = r.get("metrics", {}) or {}
        try:
            a, w, p = float(m["area"]), float(m["wns"]), float(m["power"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(a) and math.isfinite(w) and math.isfinite(p)):
            continue
        if w <= -99.0:
            continue
        pts.append([-a, w, -p])
    if len(pts) < 1:
        return [-5e6, -10.0, -5.0]
    Y = torch.tensor(pts, dtype=torch.double)
    y_min = Y.min(dim=0).values
    slack = 0.1 * Y.abs().mean(dim=0).clamp(min=1.0)
    return (y_min - slack).tolist()


def _classify_metrics(metrics: Dict[str, Any], *, fidelity: str) -> Dict[str, Any]:
    m = metrics or {}
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
    basic_ok = math.isfinite(area) and math.isfinite(wns) and (wns != -100.0)
    if not basic_ok:
        return {"metrics_status": "invalid", "metrics_usable": False, "metrics_valid": False}
    if str(fidelity).lower().strip() == "full":
        if math.isfinite(power):
            return {"metrics_status": "full", "metrics_usable": True, "metrics_valid": True}
        return {"metrics_status": "invalid", "metrics_usable": False, "metrics_valid": False}
    if bool(m.get("__proxy__", False)):
        return {"metrics_status": "proxy", "metrics_usable": True, "metrics_valid": False}
    return {"metrics_status": "partial", "metrics_usable": True, "metrics_valid": False}


def _dominates(ma: Dict[str, Any], mb: Dict[str, Any]) -> bool:
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
        area = float(m.get("area", float("inf")))
        power = float(m.get("power", float("inf")))
        wns = float(m.get("wns", -100.0))
        if area == float("inf") or power == float("inf") or wns <= -99.0:
            continue
        valid.append(r)
    pareto = []
    for a in valid:
        dominated = any(
            a["run_id"] != b["run_id"] and _dominates(a["metrics"], b["metrics"])
            for b in valid
        )
        if not dominated:
            pareto.append(a)
    pareto.sort(key=lambda x: (-float(x["metrics"]["wns"]),
                                float(x["metrics"]["area"]),
                                float(x["metrics"]["power"])))
    return pareto[:int(max_items)]


def _metrics_str(m: Dict[str, Any]) -> str:
    def f(key, default):
        try:
            return float(m.get(key, default))
        except Exception:
            return default

    area, wns, power = f("area", float("inf")), f("wns", -100.0), f("power", float("inf"))

    def fmt(x, prec=3):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return f"{x:.{prec}f}"

    return f"Area={fmt(area,1)} WNS={fmt(wns,4)} Power={fmt(power,4)}"


def _slim_history_entry(res: dict) -> None:
    """
    Memory optimization: shrink history entries in place by dropping large fields.
    Large log tails and summaries can accumulate into gigabytes across long runs.
    """
    # Truncate fail_log_tail to 2 KB.
    tail = res.get("fail_log_tail", "")
    if isinstance(tail, str) and len(tail) > 2048:
        res["fail_log_tail"] = tail[-2048:]

    # Keep only key fields from log_summary.
    ls = res.get("log_summary")
    if isinstance(ls, dict):
        slim_ls = {
            "symptoms": ls.get("symptoms", []),
            "stage": ls.get("stage"),
            "final_status": ls.get("final_status"),
        }
        res["log_summary"] = slim_ls


def _slim_for_resume(h: dict) -> dict:
    """
    Memory optimization: further trim resume-state serialization by removing
    large fields that are not needed for recovery.
    """
    skip_keys = {
        "fail_log_tail", "log_summary", "error",
        "log_path", "proposed_notes",
    }
    return {k: v for k, v in h.items() if k not in skip_keys}








def _classify_failure(res: dict) -> str:
    if res.get("success") and res.get("metrics_valid"):
        return "ok"
    if res.get("success") and res.get("metrics_usable"):
        return "ok_partial"
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
    return "other_fail"




# ── Baseline resume helpers ───────────────────────────────────────────────────

def _resume_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["work_dir"]) / "resume_state.baseline.json"


def _legacy_resume_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["work_dir"]) / "resume_state.json"


def _sort_history_by_run_id(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(item: Dict[str, Any]):
        rid = str(item.get("run_id", ""))
        try:
            if rid.startswith("iter_"):
                return int(rid.split("_", 1)[1])
        except Exception:
            pass
        return 10**12
    return sorted(history, key=_key)


def _compute_elapsed_from_history(history_all: List[Dict[str, Any]]) -> float:
    """Accumulate elapsed time by session (experiment_ts) to preserve timing across multiple resumes."""
    session_max_wall: Dict[int, float] = {}
    runtime_sum = 0.0
    for h in history_all:
        try:
            we = h.get("wall_end_sec")
            ts = h.get("experiment_ts")
            if we is not None and ts is not None:
                we_f = float(we)
                ts_key = round(float(ts))
                if we_f > 0:
                    session_max_wall[ts_key] = max(session_max_wall.get(ts_key, 0.0), we_f)
        except Exception:
            pass
        try:
            runtime_sum += max(0.0, float(h.get("runtime_sec", 0.0)))
        except Exception:
            pass
    if session_max_wall:
        return sum(session_max_wall.values())
    return runtime_sum


def _build_baseline_resume_state(
    cfg: Dict[str, Any],
    *,
    strategy: str,
    launched: int,
    start_wall: float,
    resume_elapsed_base: float = 0.0,
    warmup_quick_success: int,
    history_all: List[Dict[str, Any]],
    history_full_valid: List[Dict[str, Any]],
    seen_quick: set,
    seen_full: set,
    promo_pool: Dict[str, Tuple[Dict[str, Any], str]],
    global_ref_point: List[float],
    proposer_state: Optional[Dict[str, Any]] = None,
    tuning_mode: str = "full",
    max_iterations: int = 0,
    estimated_innovus_sec: float = 0.0,
    shared_genus_run_id: str = "",
    innovus_only_genus_ok: bool = False,
    innovus_only_shared_genus_dir: str = "",
) -> Dict[str, Any]:
    promo_pool_json: Dict[str, Dict[str, Any]] = {}
    for fp, value in (promo_pool or {}).items():
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        params, parent_run_id = value[0], value[1]
        promo_pool_json[str(fp)] = {
            "params": dict(params or {}),
            "parent_run_id": str(parent_run_id or ""),
        }

    return {
        "version": 1,
        "runner": "baseline",
        "baseline_strategy": str(strategy),
        "project_name": cfg.get("project_name", ""),
        "config_path": cfg.get("_loaded_from", ""),
        "created_ts": time.time(),
        "work_dir": str(cfg["work_dir"]),
        "launched": int(launched),
        "total_elapsed_sec": float(resume_elapsed_base + max(0.0, time.time() - start_wall)),
        "warmup_quick_success": int(warmup_quick_success),
        "history_all": [_slim_for_resume(h) for h in (history_all or [])],  # Trim before serialization.
        "history_full_valid": [_slim_for_resume(h) for h in (history_full_valid or [])],  # Trim before serialization.
        "seen_quick": sorted(str(x) for x in (seen_quick or set())),
        "seen_full": sorted(str(x) for x in (seen_full or set())),
        "promo_pool": promo_pool_json,
        "global_ref_point": list(global_ref_point or [-5e6, -10.0, -5.0]),
        "proposer_state": proposer_state,
        "tuning_mode": str(tuning_mode or "full"),
        "max_iterations": int(max_iterations or 0),
        "estimated_innovus_sec": float(estimated_innovus_sec or 0.0),
        "shared_genus_run_id": str(shared_genus_run_id or ""),
        "innovus_only_genus_ok": bool(innovus_only_genus_ok),
        "innovus_only_shared_genus_dir": str(innovus_only_shared_genus_dir or ""),
    }


def _save_baseline_resume_state(path: Path, state: Dict[str, Any]) -> None:
    """Atomic save with fsync — prevents corruption on crash/kill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    data = json.dumps(state, indent=2, ensure_ascii=False, default=str) + "\n"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)


def _load_baseline_resume_state(cfg: Dict[str, Any], strategy: str) -> Optional[Dict[str, Any]]:
    candidates = [_resume_path(cfg), _legacy_resume_path(cfg)]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] failed to load resume file {path}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("runner", "baseline")).lower() not in ("baseline", ""):
            continue
        stored_strategy = str(data.get("baseline_strategy", strategy)).strip().lower()
        if stored_strategy and stored_strategy != str(strategy).strip().lower():
            print(f"[WARN] resume strategy mismatch in {path.name}: {stored_strategy} != {strategy}, ignore.")
            continue
        return data
    return None


def _auto_resume_from_dirs(work_dir: Path) -> Tuple[int, set, set]:
    max_iter = -1
    seen_q, seen_f = set(), set()
    if not work_dir.is_dir():
        return max_iter, seen_q, seen_f

    for d in work_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("iter_"):
            continue
        try:
            idx = int(d.name.split("_", 1)[1])
            max_iter = max(max_iter, idx)
        except Exception:
            pass

        ckpt = d / "checkpoint.json"
        if not ckpt.is_file():
            continue
        try:
            info = json.loads(ckpt.read_text(encoding="utf-8"))
            fp = info.get("params_fingerprint")
            fidelity = str(info.get("fidelity", "quick"))
            if fp:
                if fidelity == "full":
                    seen_f.add(str(fp))
                else:
                    seen_q.add(str(fp))
        except Exception:
            continue

    return max_iter, seen_q, seen_f


def _rebuild_runtime_predictor(
    rt: MultiFidelityRuntimePredictor,
    history_all: List[Dict[str, Any]],
    fill_params,
) -> None:
    quick_n = 0
    full_n = 0
    for h in history_all:
        if not h.get("success", False):
            continue
        params = h.get("params", {}) or {}
        try:
            runtime_sec = float(h.get("runtime_sec", 0.0))
        except Exception:
            runtime_sec = 0.0
        if runtime_sec <= 0:
            continue
        fidelity = str(h.get("fidelity", "quick"))
        try:
            rt.observe(fidelity, fill_params(params), runtime_sec)
            if fidelity == "full":
                full_n += 1
            else:
                quick_n += 1
        except Exception:
            continue

    try:
        if quick_n > 0:
            rt.fit("quick")
    except Exception:
        pass
    try:
        if full_n > 0:
            rt.fit("full")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# Resume: Proposer State Restoration
# (merged from baseline_resume_patch.py — no external file needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _align_params_to_distributions(params_raw: Dict[str, Any],
                                    param_space: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Align historical parameter values to the categorical choices defined in param_space."""
    aligned = {}
    choices_map = {p["name"]: p["choices"] for p in param_space if "choices" in p}
    for k, v in params_raw.items():
        if k in choices_map:
            str_v = str(v).lower() if isinstance(v, (bool, str)) else str(v)
            matched = [c for c in choices_map[k] if str(c).lower() == str_v]
            if matched:
                aligned[k] = matched[0]
            else:
                continue  # Skip unmatchable parameters to avoid invalid trials.
        else:
            aligned[k] = v
    return aligned


def _restore_optuna_study(proposer, history: List[Dict[str, Any]]) -> bool:
    """Replay completed trials into OptunaProposer's study.

    The study has three objectives (area, -wns, power), so each trial must use
    a 3-element values list.
    """
    if not hasattr(proposer, "study") or proposer.study is None:
        return False
    import optuna
    restored = 0
    # Use proposer's own distributions for consistency
    dists = proposer._build_distributions() if hasattr(proposer, "_build_distributions") else {}
    for rec in history:
        params_raw = rec.get("params") or rec.get("config", {}).get("params")
        if params_raw is None:
            continue
        # Extract area, WNS, and power from metrics rather than a scalar score.
        m = rec.get("metrics") or {}
        try:
            area = float(m["area"])
            wns = float(m["wns"])
            power = float(m["power"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(area) and math.isfinite(wns) and math.isfinite(power)):
            continue
        try:
            trial = optuna.trial.create_trial(
                params=_align_params_to_distributions(
                    {k: v for k, v in params_raw.items() if k in (dists or params_raw)},
                    proposer.param_space,
                ),

                distributions=dists if dists else {
                    k: optuna.distributions.CategoricalDistribution([v])
                    for k, v in params_raw.items()
                },
                values=[area, -wns, power],  # Three values matching the three study directions.
            )
            proposer.study.add_trial(trial)
            restored += 1
        except Exception:
            # If create_trial fails, fall back to a manual restore path.
            try:
                _restore_optuna_study_manual(proposer, [rec])
            except Exception:
                pass
    if restored:
        print(f"  [Resume] Restored {restored} trials to Optuna study")
    return restored > 0


def _restore_optuna_study_manual(proposer, history: List[Dict[str, Any]]) -> int:
    """Manual fallback: enqueue known-good params so Optuna doesn't re-explore."""
    if not hasattr(proposer, "study"):
        return 0
    count = 0
    for rec in history:
        params_raw = rec.get("params") or rec.get("config", {}).get("params")
        if params_raw:
            try:
                proposer.study.enqueue_trial(
                    _align_params_to_distributions(params_raw, proposer.param_space)
                )

                count += 1
            except Exception:
                pass
    return count


def _warm_start_bo_proposer(proposer, history: List[Dict[str, Any]]) -> bool:
    """Log warm-start info for BOProposer (BO reads history_all directly)."""
    if not hasattr(proposer, "_bo_model"):
        return False
    valid = [r for r in history if r.get("score") is not None]
    print(f"  [Resume] BO warm-start: {len(valid)} valid history points available")
    return len(valid) > 0


def _restore_proposer_on_resume(proposer, strategy: str, history: List[Dict[str, Any]], saved_proposer_state: Any = None) -> None:
    """One-call entry: restore proposer internal state after resume."""
    if not history:
        return
    strategy = str(strategy).strip().lower()
    print(f"  [Resume] Restoring proposer state for strategy='{strategy}', history={len(history)}")

    if strategy == "optuna":
        if not _restore_optuna_study(proposer, history):
            n = _restore_optuna_study_manual(proposer, history)
            if n:
                print(f"  [Resume] Optuna manual enqueue: {n} trials")
    elif strategy == "bo":
        _warm_start_bo_proposer(proposer, history)
    elif strategy == "random":
        print(f"  [Resume] RandomProposer: no state to restore (stateless)")
    elif strategy in ("ranktuner", "rank_tuner", "rank"):
        from baselines.ranktuner_proposer import _restore_ranktuner_proposer  # Lazy import.
        _restore_ranktuner_proposer(
            proposer, history,
            saved_state=saved_proposer_state if saved_proposer_state else None,
        )
    elif strategy in ("crop", "crop_llm", "crop_rag"):
        from baselines.crop_proposer import _restore_crop_proposer  # Lazy import.
        _restore_crop_proposer(
            proposer, history,
            saved_state=saved_proposer_state if saved_proposer_state else None,
        )
    else:
        print(f"  [Resume] Unknown strategy '{strategy}', skip proposer restore")


def _save_iteration_checkpoint(work_dir: Path, result: Dict[str, Any],
                                spec: Optional[Dict[str, Any]] = None) -> None:
    """Save a per-iteration checkpoint with params fingerprint for dedup."""
    import hashlib
    params = result.get("params") or result.get("config", {}).get("params", {})
    fidelity = result.get("fidelity", "quick")
    fp = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

    iteration = result.get("iteration", result.get("iter", -1))
    ckpt_dir = work_dir / f"iter_{iteration}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "iteration": iteration,
        "params_fingerprint": fp,
        "fidelity": str(fidelity),
        "score": result.get("score"),
        "status": result.get("status", "ok"),
        "params": params,
    }
    if spec:
        ckpt["spec_summary"] = {k: v for k, v in spec.items()
                                 if k in ("make_target", "fidelity", "timeout")}

    ckpt_path = ckpt_dir / "checkpoint.json"
    try:
        _save_baseline_resume_state(ckpt_path, ckpt)
    except Exception as e:
        print(f"  [WARN] checkpoint save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate Proposer Strategies
# ═══════════════════════════════════════════════════════════════════════════════

class RandomProposer:
    """Pure random sampling from param_space."""

    def __init__(self, param_space: List[Dict[str, Any]]):
        self.param_space = param_space

    def propose(self, history: List[Dict], n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        for _ in range(n):
            params: Dict[str, Any] = {}
            for p in self.param_space:
                name = p["name"]
                if "range" in p:
                    lo, hi = p["range"]
                    if p.get("type") == "int":
                        params[name] = random.randint(int(lo), int(hi))
                    else:
                        params[name] = round(random.uniform(float(lo), float(hi)), 4)
                elif "choices" in p:
                    params[name] = random.choice(p["choices"])
                else:
                    params[name] = 0.0
            candidates.append(params)
        return candidates

    @property
    def strategy_name(self) -> str:
        return "random"


class BOProposer:
    """
    Multi-objective BO proposer using qEHVI acquisition.
    Reuses MOEHVIAgent from EDATuner but without any LLM/KA guidance.
    Falls back to random when insufficient data for GP fitting.
    """

    def __init__(self, param_space: List[Dict[str, Any]]):
        self.param_space = param_space
        MOEHVIAgent, _ = _get_ehvi_imports()
        self.agent = MOEHVIAgent(param_space)
        self.agent.warmup_trials = 0  # we handle warmup externally
        self._random = RandomProposer(param_space)

    def propose(self, history: List[Dict], n: int = 1) -> List[Dict[str, Any]]:
        # Need at least 6 valid points for GP fitting
        valid = [
            h for h in history
            if self.agent._is_valid_metrics(h.get("metrics", {}))
            and isinstance(h.get("params", None), dict)
            and len(h.get("params", {})) > 0
        ]

        if len(valid) < 6:
            return self._random.propose(history, n)

        candidates = []
        for _ in range(n):
            try:
                X = torch.cat([self.agent._to_tensor(h.get("params", {})) for h in valid], dim=0).to(torch.double)
                Y = torch.stack([self.agent._to_objectives_max(h["metrics"]) for h in valid], dim=0).to(torch.double)
                ref_point = self.agent._choose_ref_point(Y)
                self.agent._update_trust_region(X, Y, ref_point)
                model = self.agent._fit_qor_model(X, Y)

                from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
                from botorch.sampling.normal import SobolQMCNormalSampler
                from botorch.acquisition.multi_objective.logei  import qLogExpectedHypervolumeImprovement

                partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y)
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
                acq = qLogExpectedHypervolumeImprovement(
                    model=model, ref_point=ref_point,
                    partitioning=partitioning, sampler=sampler,
                )

                cand_X = self.agent._sample_candidate_pool(512)
                model.eval()
                acq.eval()
                with torch.no_grad():
                    ehvi = acq(cand_X.unsqueeze(1)).view(-1)
                best_idx = int(torch.argmax(ehvi).item())
                params = self.agent._to_dict(cand_X[best_idx:best_idx + 1])
                candidates.append(params)
            except Exception as e:
                print(f"  [BO] GP/EHVI failed: {e}, fallback random")
                candidates.extend(self._random.propose(history, 1))
        return candidates

    @property
    def strategy_name(self) -> str:
        return "bo_ehvi"


class OptunaProposer:
    """
    Optuna TPE multi-objective proposer.
    Uses NSGAIISampler or TPESampler with multi-objective support.
    Maintains an internal Optuna study that gets told about completed trials.
    """

    def __init__(self, param_space: List[Dict[str, Any]], study_name: str = "eda_optuna"):
        optuna = _get_optuna()
        self.param_space = param_space

        # Create multi-objective study: minimize area, maximize wns, minimize power
        # Optuna always minimizes, so we negate wns
        self.study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "minimize", "minimize"],  # area, -wns, power
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                n_startup_trials=10,
            ),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._reported_ids: set = set()

    def _sync_history(self, history: List[Dict]):
        """Report completed trials to Optuna study that haven't been reported yet."""
        optuna = _get_optuna()
        for h in history:
            rid = h.get("run_id", "")
            if rid in self._reported_ids:
                continue
            if not h.get("metrics_valid", False):
                continue
            m = h.get("metrics", {}) or {}
            try:
                area = float(m["area"])
                wns = float(m["wns"])
                power = float(m["power"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(area) and math.isfinite(wns) and math.isfinite(power)):
                continue

            # Build param dict for Optuna
            params = h.get("params", {}) or {}
            # Align categorical values to the legal entries defined in choices.
            aligned_params = {}
            for p in self.param_space:
                name = p["name"]
                if name not in params:
                    continue
                val = params[name]
                if "choices" in p:
                    # Map booleans or case-variant strings back to valid choices.
                    str_val = str(val).lower() if isinstance(val, (bool, str)) else str(val)
                    matched = [c for c in p["choices"] if str(c).lower() == str_val]
                    if matched:
                        val = matched[0]
                    # If it still cannot be matched, skip the parameter.
                    elif str_val not in [str(c) for c in p["choices"]]:
                        continue
                aligned_params[name] = val

            trial = optuna.trial.create_trial(
                params=aligned_params,
                distributions=self._build_distributions(),
                values=[area, -wns, power],
            )

            self.study.add_trial(trial)
            self._reported_ids.add(rid)

    def _build_distributions(self) -> dict:
        optuna = _get_optuna()
        dists = {}
        for p in self.param_space:
            name = p["name"]
            if "range" in p:
                lo, hi = p["range"]
                if p.get("type") == "int":
                    dists[name] = optuna.distributions.IntDistribution(int(lo), int(hi))
                else:
                    dists[name] = optuna.distributions.FloatDistribution(float(lo), float(hi))
            elif "choices" in p:
                dists[name] = optuna.distributions.CategoricalDistribution(list(p["choices"]))
        return dists

    def propose(self, history: List[Dict], n: int = 1) -> List[Dict[str, Any]]:
        optuna = _get_optuna()
        self._sync_history(history)

        candidates = []
        for _ in range(n):
            trial = self.study.ask(self._build_distributions())
            params: Dict[str, Any] = {}
            for p in self.param_space:
                name = p["name"]
                if "range" in p:
                    lo, hi = p["range"]
                    if p.get("type") == "int":
                        params[name] = trial.suggest_int(name, int(lo), int(hi))
                    else:
                        params[name] = round(trial.suggest_float(name, float(lo), float(hi)), 4)
                elif "choices" in p:
                    params[name] = trial.suggest_categorical(name, list(p["choices"]))
            # Tell Optuna we won't complete this trial through the normal flow
            # (we'll report results via _sync_history on next call)
            try:
                # Mark the placeholder ask() result as pruned with the correct objective shape.
                self.study.tell(trial, values=[0.0, 0.0, 0.0],
                                state=optuna.trial.TrialState.PRUNED)
            except Exception:
                pass
            candidates.append(params)
        return candidates

    @property
    def strategy_name(self) -> str:
        return "optuna_tpe"


def create_proposer(strategy: str, param_space: List[Dict[str, Any]],
                    study_name: str = "eda", cfg: Dict[str, Any] = None):
    strategy = strategy.strip().lower()
    if strategy == "random":
        return RandomProposer(param_space)
    elif strategy in ("bo", "bo_ehvi", "bayesian"):
        return BOProposer(param_space)
    elif strategy in ("optuna", "tpe", "optuna_tpe"):
        return OptunaProposer(param_space, study_name=study_name)
    elif strategy in ("ranktuner", "rank_tuner", "rank"):
        from baselines.ranktuner_proposer import RankTunerProposer  # Lazy import to reduce startup memory use.
        # Read RankTuner hyperparameters from the config when provided.
        rt_cfg = (cfg or {}).get("ranktuner", {}) or {}
        return RankTunerProposer(
            param_space,
            noise_sigma=float(rt_cfg.get("noise_sigma", 1.0)),
            lengthscale=float(rt_cfg.get("lengthscale", 1.0)),    # Public default is 1.0.
            outputscale=float(rt_cfg.get("outputscale", 1.0)),
            n_candidate_pool=int(rt_cfg.get("n_candidate_pool", 512)),
            min_data_for_gp=int(rt_cfg.get("min_data_for_gp", 6)),
            embed_dim=int(rt_cfg.get("embed_dim", 0)) or None,     # 0 means auto.
            embed_seed=int(rt_cfg.get("embed_seed", 42)),
            tr_init_length=float(rt_cfg.get("tr_init_length", 0.8)),
            tr_min_length=float(rt_cfg.get("tr_min_length", 0.05)),
            tr_max_length=float(rt_cfg.get("tr_max_length", 1.0)),
            tr_shrink=float(rt_cfg.get("tr_shrink", 0.7)),
            tr_grow=float(rt_cfg.get("tr_grow", 1.3)),
            optimize_hyperparams=bool(rt_cfg.get("optimize_hyperparams", True)),
            hyperparam_opt_interval=int(rt_cfg.get("hyperparam_opt_interval", 5)),
        )
    elif strategy in ("crop", "crop_llm", "crop_rag"):
        from baselines.crop_proposer import CROPProposer  # Lazy import to reduce startup memory use.
        crop_cfg = (cfg or {}).get("crop", {}) or {}
        rag_cfg = (cfg or {}).get("rag", {}) or {}
        llm_cfg = (cfg or {}).get("llm", {}) or {}
        return CROPProposer(
            param_space,
            # ── Two-stage CoT models ──
            cot_model=str(crop_cfg.get("cot_model",
                          llm_cfg.get("cot_model", "")) or "") or None,
            generation_model=str(crop_cfg.get("generation_model",
                                llm_cfg.get("generation_model", "deepseek-v3.2"))),
            # ── Temperature ──
            llm_temperature=float(crop_cfg.get("llm_temperature",
                                  llm_cfg.get("temperature", 0.6))),
            cot_temperature=float(crop_cfg.get("cot_temperature",
                                  llm_cfg.get("cot_temperature", 0.4))),
            # ── Token limits ──
            llm_max_tokens=int(crop_cfg.get("llm_max_tokens",
                               llm_cfg.get("max_output_tokens", 8192))),
            cot_max_tokens=int(crop_cfg.get("cot_max_tokens",
                               llm_cfg.get("cot_max_output_tokens", 2000))),
            # ── Timeout ──
            llm_timeout_sec=int(crop_cfg.get("llm_timeout_sec",
                                llm_cfg.get("timeout_sec", 300))),
            # ── RAG & prompt ──
            rag_cfg=rag_cfg,
            max_history_in_prompt=int(crop_cfg.get("max_history_in_prompt", 20)),
            max_rag_context_chars=int(crop_cfg.get("max_rag_context_chars", 4000)),
            fallback_to_random=bool(crop_cfg.get("fallback_to_random", True)),
        )
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy!r}. "
                            f"Choose from: random, bo, optuna, ranktuner, crop")


# ═══════════════════════════════════════════════════════════════════════════════
# TaskSpec & Evaluation (mirror main.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskSpec:
    run_id: str
    fidelity: str
    stage: str
    params: Dict[str, Any]
    make_target: Optional[str]
    timeout_sec: int
    allow_missing_metrics: bool
    parent_run_id: Optional[str] = None
    proposed_by: str = "unknown"
    proposed_strategy: str = "unknown"
    notes: str = ""
    model: str = ""
    launch_wall_sec: float = 0.0
    ehvi_score: float = 0.0
    ehvi_time_ucb: float = 0.0


def _extract_error_warning_tail(log_path, max_lines=12, max_bytes=200_000) -> str:
    """Extract last few lines from log for error diagnosis."""
    try:
        p = Path(str(log_path))
        if not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_bytes:
            text = text[-max_bytes:]
        lines = text.strip().splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _evaluate_one(runner: EDARunner, spec: TaskSpec) -> Dict[str, Any]:
    start = time.time()

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

    # Per-run retry: do not discard a parameter set just because the connection dropped.
    _MAX_RUN_RETRIES = 5
    _last_exc = None
    ok, log_path = False, ""
    for _run_attempt in range(_MAX_RUN_RETRIES):
        try:
            ok, log_path = runner.run(
                spec.params,
                spec.run_id,
                script_override=None,
                make_target=spec.make_target,
                timeout_sec=spec.timeout_sec,
                allow_missing_metrics=spec.allow_missing_metrics,
            )
            _last_exc = None
            break  # A completed run attempt is enough, regardless of success status.
        except Exception as _run_exc:
            _last_exc = _run_exc
            _exc_name = type(_run_exc).__name__
            # Retry only connection-related errors; re-raise everything else.
            if "RemoteComm" not in _exc_name and "RemoteSessionLost" not in _exc_name:
                raise
            _wait = min(60 * (2 ** _run_attempt), 600) + random.randint(0, 30)
            print(f"[RETRY] Run {spec.run_id} attempt {_run_attempt+1}/{_MAX_RUN_RETRIES} "
                  f"failed ({_exc_name}), retrying in {_wait}s ...")
            time.sleep(_wait)

    if _last_exc is not None:
        # All retries failed; re-raise so the main loop can handle it.
        raise _last_exc

    runtime_sec = time.time() - start


    metrics = runner.get_metrics(log_path)

    cls = _classify_metrics(metrics, fidelity=spec.fidelity)
    metrics_status = cls["metrics_status"]
    metrics_usable = cls["metrics_usable"]
    metrics_valid = cls["metrics_valid"]

    fail_tail = ""
    if (not ok) or (not metrics_usable):
        try:
            fail_tail = _extract_error_warning_tail(log_path, max_lines=12, max_bytes=200_000)
        except Exception:
            fail_tail = ""

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


# ═══════════════════════════════════════════════════════════════════════════════
# Main Baseline Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main(config_path: str = "configs/baseline_random.yaml") -> None:
    cfg = load_config(config_path)
    tuning = cfg.get("tuning", {}) or {}
    fid = cfg.get("fidelity", {}) or {}

    # ── Strategy ──
    STRATEGY = str(cfg.get("baseline_strategy", "random")).strip().lower()
    print(f"[BASELINE] Strategy: {STRATEGY}")

    # ── Budgets ──
    MAX_CONCURRENCY = int(tuning.get("max_concurrency", 1))
    TOTAL_SECONDS = int(tuning.get("total_seconds", 10 * 3600))
    TUNING_MODE = str(tuning.get("mode", "full")).strip().lower()
    MAX_ITERATIONS = int(tuning.get("max_iterations", 200))
    EST_INNOVUS_SEC = float(tuning.get("estimated_innovus_sec", 1800))
    SHARED_GENUS_RUN_ID = str(tuning.get("shared_genus_run_id", "") or "").strip()
    IS_INNOVUS_ONLY = (TUNING_MODE == "innovus_only")
    MAX_TRIALS = int(tuning.get("max_trials", 100000))
    CLEANUP_EVERY = int(tuning.get("cleanup_every", 20))
    STOP_STARTING_SEC = int(fid.get("stop_starting_sec", 15 * 60))
    RISK_BETA = float(fid.get("risk_beta", 2.0))
    MARGIN_SEC = float(fid.get("margin_sec", 0.0))

    # ── Fidelity ──
    QUICK_TARGET = str(fid.get("quick_target", "place")).strip()
    FULL_TARGET = str(fid.get("full_target", "route")).strip()
    QUICK_TIMEOUT = int(fid.get("quick_timeout_sec", 3600))
    FULL_TIMEOUT = int(fid.get("full_timeout_sec", 14400))
    PROMOTE_PROB = float(fid.get("promote_prob", 0.35))
    PROMO_BATCH = int(fid.get("promo_batch", 64))
    WARMUP_QUICK = int(fid.get("warmup_quick", max(16, 2 * PROMO_BATCH)))
    MAX_PROMO_POOL = int(fid.get("max_promo_pool", 2000))

    # ── Core objects ──
    param_space = _get_param_space(cfg)
    default_params = _default_params_to_friendly(cfg)

    runner = create_runner(cfg)
    proposer = create_proposer(
        STRATEGY, param_space,
        study_name=cfg.get("project_name", "baseline"),
        cfg=cfg,
    )

    # Lazily instantiate the EHVI agent on first promotion to avoid a heavy
    # BoTorch startup cost when promotion is never used.
    ehvi_agent = None
    rt = MultiFidelityRuntimePredictor(param_space, min_samples_quick=10, min_samples_full=10)
    ibe = IterationBudgetEstimator(
        estimated_sec_per_iter=EST_INNOVUS_SEC,
        max_concurrency=MAX_CONCURRENCY,
    ) if IS_INNOVUS_ONLY else None

    # ── KB ──
    kb_dir = Path(cfg["work_dir"]) / "kb"
    kb = KB(kb_dir)
    kb.write_meta({
        "project_name": cfg.get("project_name", ""),
        "baseline_strategy": STRATEGY,
        "created_ts": time.time(),
        "max_concurrency": MAX_CONCURRENCY,
        "total_seconds": TOTAL_SECONDS,
        "quick_target": QUICK_TARGET,
        "full_target": FULL_TARGET,
        "quick_timeout_sec": QUICK_TIMEOUT,
        "full_timeout_sec": FULL_TIMEOUT,
        "promote_prob": PROMOTE_PROB,
        "param_space": param_space,
        "default_params": cfg.get("default_params", {}),
        "llm_enabled": False,
    })

    # ── State ──
    history_all: List[Dict[str, Any]] = []
    history_full_valid: List[Dict[str, Any]] = []
    promo_pool: Dict[str, Tuple[Dict[str, Any], str]] = {}
    warmup_quick_done: bool = (WARMUP_QUICK <= 0)
    warmup_quick_success: int = 0
    seen_quick: set = set()
    seen_full: set = set()

    start_wall = time.time()
    if IS_INNOVUS_ONLY:
        deadline = start_wall + 365 * 24 * 3600
    else:
        deadline = start_wall + TOTAL_SECONDS
    launched = 0
    _global_ref_point = [-5e6, -10.0, -5.0]
    _innovus_only_genus_ok = False
    _innovus_only_shared_genus_dir = ""

    fatal_stop_requested = False
    fatal_stop_reason = ""

    def remaining_sec() -> float:
        if IS_INNOVUS_ONLY:
            remaining_iters = max(0, MAX_ITERATIONS - launched)
            return ibe.remaining_sec(remaining_iters) if ibe is not None else 0.0
        return max(0.0, deadline - time.time())

    def compute_budget_remaining() -> float:
        if IS_INNOVUS_ONLY:
            remaining_iters = max(0, MAX_ITERATIONS - launched)
            return ibe.budget_remaining(remaining_iters) if ibe is not None else 0.0
        elapsed = TOTAL_SECONDS - remaining_sec()
        return max(0.0, TOTAL_SECONDS * MAX_CONCURRENCY - elapsed)

    def can_launch_new() -> bool:
        if fatal_stop_requested:
            return False
        # Do not short-circuit backpressure in the main loop. Workers are allowed
        # to block at submission time so the scheduler can recover naturally
        # without deadlocking the launch path.
        if launched >= MAX_TRIALS:
            return False
        if IS_INNOVUS_ONLY:
            return launched < MAX_ITERATIONS
        return time.time() < deadline



    def _check_stop_file() -> bool:
        nonlocal fatal_stop_requested, fatal_stop_reason
        stop_path = Path(cfg["work_dir"]) / "STOP"
        if stop_path.exists():
            if not fatal_stop_requested:
                print("[STOP] stop flag detected")
                fatal_stop_requested = True
                fatal_stop_reason = "STOP file detected"
            return True
        return False

    def _trigger_fatal_stop(reason: str) -> None:
        nonlocal fatal_stop_requested, fatal_stop_reason
        if not fatal_stop_requested:
            fatal_stop_requested = True
            fatal_stop_reason = reason
            print(f"[FATAL] {reason}")



    def _on_run_result(success: bool, is_conn_error: bool) -> None:
        """Update the consecutive connection-failure counter after each run."""
        global _consecutive_conn_failures
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


    def fill_params(p: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values from defaults and param_space midpoints."""
        out = dict(default_params)
        if isinstance(p, dict):
            out.update(p)
        for ps in param_space:
            if not isinstance(ps, dict):
                continue
            name = ps.get("name")
            if not name or out.get(name) is not None:
                continue
            if "default" in ps and ps["default"] is not None:
                out[name] = ps["default"]
            elif "range" in ps and isinstance(ps["range"], (list, tuple)) and len(ps["range"]) == 2:
                lo, hi = ps["range"]
                mid = (float(lo) + float(hi)) / 2.0
                if str(ps.get("type", "")).lower() in ("int", "integer"):
                    out[name] = int(round(mid))
                else:
                    out[name] = float(mid)
            elif "choices" in ps and isinstance(ps["choices"], (list, tuple)) and ps["choices"]:
                out[name] = ps["choices"][0]
        return out

    # ── Resume / restore baseline state ──
    resume_elapsed_base = 0.0
    rs = _load_baseline_resume_state(cfg, STRATEGY)
    if rs:
        history_all = _sort_history_by_run_id([
            h for h in (rs.get("history_all", []) or [])
            if isinstance(h, dict)
        ])
        history_full_valid = _sort_history_by_run_id([
            h for h in (rs.get("history_full_valid", []) or [])
            if isinstance(h, dict)
        ])

        promo_pool = {}
        for fp, entry in (rs.get("promo_pool", {}) or {}).items():
            if isinstance(entry, dict):
                params = fill_params(entry.get("params", {}) or {})
                parent_run_id = str(entry.get("parent_run_id") or entry.get("run_id") or "")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                params = fill_params(entry[0] or {})
                parent_run_id = str(entry[1] or "")
            else:
                continue
            promo_pool[str(fp)] = (params, parent_run_id)

        seen_quick = set(str(x) for x in (rs.get("seen_quick", []) or []))
        seen_full = set(str(x) for x in (rs.get("seen_full", []) or []))
        launched = int(rs.get("launched", 0))
        warmup_quick_success = int(rs.get("warmup_quick_success", 0))
        warmup_quick_done = (warmup_quick_success >= WARMUP_QUICK)

        for h in history_all:
            params = fill_params(h.get("params", {}) or {})
            fp = _fingerprint(params)
            if str(h.get("fidelity", "quick")) == "full":
                seen_full.add(fp)
            else:
                seen_quick.add(fp)

        prev_elapsed = float(rs.get("total_elapsed_sec", _compute_elapsed_from_history(history_all)))
        resume_elapsed_base = max(0.0, prev_elapsed)
        if not IS_INNOVUS_ONLY:
            deadline = start_wall + max(0.0, TOTAL_SECONDS - resume_elapsed_base)

        grp = rs.get("global_ref_point")
        if isinstance(grp, list) and len(grp) == 3:
            try:
                _global_ref_point = [float(grp[0]), float(grp[1]), float(grp[2])]
            except Exception:
                _global_ref_point = [-5e6, -10.0, -5.0]

        _rebuild_runtime_predictor(rt, history_all, fill_params)

        if IS_INNOVUS_ONLY:
            _resume_shared_dir = str(rs.get("innovus_only_shared_genus_dir", "") or "").strip()
            _resume_genus_ok = bool(rs.get("innovus_only_genus_ok", False))
            if (not _resume_shared_dir) and SHARED_GENUS_RUN_ID:
                _resume_shared_dir = _infer_shared_genus_dir(cfg, runner, SHARED_GENUS_RUN_ID)
            if (not _resume_shared_dir) and any(r.get("run_id") == "genus_shared" and r.get("success") for r in history_all):
                _resume_shared_dir = _infer_shared_genus_dir(cfg, runner, "genus_shared")
            if _resume_genus_ok and _resume_shared_dir and hasattr(runner, "set_shared_genus_dir"):
                runner.set_shared_genus_dir(_resume_shared_dir)
                _innovus_only_genus_ok = True
                _innovus_only_shared_genus_dir = _resume_shared_dir
                print(f"[INNOVUS-ONLY][RESUME] shared genus restored: {_innovus_only_shared_genus_dir}")

        print(
            f"[RESUME] loaded {_resume_path(cfg).name if _resume_path(cfg).is_file() else _legacy_resume_path(cfg).name} "
            f"launched={launched} warmup_quick_success={warmup_quick_success} "
            f"history_all={len(history_all)} history_full_valid={len(history_full_valid)} "
            f"promo_pool={len(promo_pool)}"
        )

        # ── Restore proposer internal state (Optuna study / BO warm-start / RankTuner TR) ──
        _saved_proposer_state = rs.get("proposer_state")
        _restore_proposer_on_resume(proposer, STRATEGY, history_all,
                                    saved_proposer_state=_saved_proposer_state)
    else:
        max_iter, auto_seen_quick, auto_seen_full = _auto_resume_from_dirs(Path(cfg["work_dir"]))
        if max_iter >= 0:
            launched = max_iter + 1
            seen_quick.update(auto_seen_quick)
            seen_full.update(auto_seen_full)
            print(
                f"[RESUME] no resume_state found; detected existing iter_* up to iter_{max_iter}, "
                f"next launched={launched}"
            )

    def _persist_resume_state() -> None:
        try:
            # Capture proposer-internal state (e.g. RankTuner Trust Region)
            _proposer_state = None
            if hasattr(proposer, 'get_state'):
                try:
                    _proposer_state = proposer.get_state()
                except Exception:
                    pass

            state = _build_baseline_resume_state(
                cfg,
                strategy=STRATEGY,
                launched=launched,
                start_wall=start_wall,
                resume_elapsed_base=resume_elapsed_base,
                warmup_quick_success=warmup_quick_success,
                history_all=history_all,
                history_full_valid=history_full_valid,
                seen_quick=seen_quick,
                seen_full=seen_full,
                promo_pool=promo_pool,
                global_ref_point=_global_ref_point,
                proposer_state=_proposer_state,
                tuning_mode=TUNING_MODE,
                max_iterations=MAX_ITERATIONS,
                estimated_innovus_sec=EST_INNOVUS_SEC,
                shared_genus_run_id=SHARED_GENUS_RUN_ID,
                innovus_only_genus_ok=_innovus_only_genus_ok,
                innovus_only_shared_genus_dir=_innovus_only_shared_genus_dir,
            )
            state["total_elapsed_sec"] = float(resume_elapsed_base + max(0.0, time.time() - start_wall))
            _save_baseline_resume_state(_resume_path(cfg), state)
        except Exception as e:
            print(f"[WARN] failed to persist baseline resume state: {e}")

    # ── Candidate proposal ──
    def _propose_quick(stage: str = "explore_quick") -> Optional[TaskSpec]:
        nonlocal launched

        # Request one candidate at a time so internal proposer buffers can be
        # used efficiently. If a sampled candidate is already in seen_quick,
        # retry and fetch the next buffered proposal.
        _MAX_DEDUP_RETRIES = 20
        for _attempt in range(_MAX_DEDUP_RETRIES):
            try:
                candidates = proposer.propose(history_all, n=1, default_params=default_params)
            except TypeError:
                candidates = proposer.propose(history_all, n=1)
            if not candidates:
                break
            p = fill_params(candidates[0])
            fp = _fingerprint(p)
            if fp in seen_quick:
                continue
            seen_quick.add(fp)

            run_id = f"iter_{launched}"
            return TaskSpec(
                run_id=run_id,
                fidelity="quick",
                stage=stage,
                params=p,
                make_target=(None if QUICK_TARGET == "" else QUICK_TARGET),
                timeout_sec=QUICK_TIMEOUT,
                allow_missing_metrics=True,
                proposed_by=f"baseline({STRATEGY})",
                proposed_strategy=proposer.strategy_name,
                notes=f"Baseline {STRATEGY} candidate",
            )
        return None

    def _pick_promotion_full() -> Optional[TaskSpec]:
        nonlocal launched

        if not promo_pool:
            return None

        cand = []
        rem = max(1.0, remaining_sec() - float(MARGIN_SEC))
        for fp, (params, parent_id) in promo_pool.items():
            fp_full = _fingerprint(params)
            if fp_full in seen_full:
                continue
            mean_t, std_t = rt.full.predict(params)
            time_ucb = float(max(1.0, mean_t + RISK_BETA * std_t))
            time_ucb = min(time_ucb, float(FULL_TIMEOUT))
            if time_ucb > rem:
                continue
            cand.append((fp, params, parent_id, time_ucb))

        if not cand:
            return None

        cand.sort(key=lambda x: x[3])
        subset_size = max(1, min(len(cand), PROMO_BATCH * 4))
        subset = cand[:subset_size]
        params_list = [it[1] for it in subset]

        # Lazily create ehvi_agent on the first promotion request.
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
            remaining_sec=compute_budget_remaining(),
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
            proposed_by=f"baseline({STRATEGY})",
            proposed_strategy=f"{proposer.strategy_name}_promotion",
            notes="Promoted from quick pool using EHVI/time scoring.",
            ehvi_score=_ehvi_val,
            ehvi_time_ucb=_time_ucb_val,
        )

    # ── HV reference point ──

    # ── Main scheduling loop ──
    print("=" * 60)
    print(f"=== Baseline: {STRATEGY} ===")
    print(f"quick_target={QUICK_TARGET} full_target={FULL_TARGET}")
    print(f"max_concurrency={MAX_CONCURRENCY} total_hours={TOTAL_SECONDS/3600:.2f}")
    print(f"warmup_quick={WARMUP_QUICK} promote_prob={PROMOTE_PROB}")
    print(f"tuning_mode={TUNING_MODE}")
    if IS_INNOVUS_ONLY:
        print(f"  innovus_only: max_iterations={MAX_ITERATIONS} est_innovus_sec={EST_INNOVUS_SEC:.0f}")
    print("=" * 60)

    if history_full_valid:
        _global_ref_point = compute_global_ref_point(history_full_valid)
    else:
        _global_ref_point = [-5e6, -10.0, -5.0]
    print(f"[HV] Global reference point: {_global_ref_point}")

    if IS_INNOVUS_ONLY and not _innovus_only_genus_ok:
        print("=== INNOVUS-ONLY MODE ===")
        print(f"max_iterations={MAX_ITERATIONS} estimated_innovus_sec={EST_INNOVUS_SEC}")
        _shared_dir = ""
        if SHARED_GENUS_RUN_ID:
            _shared_dir = _infer_shared_genus_dir(cfg, runner, SHARED_GENUS_RUN_ID)
            if _shared_dir and hasattr(runner, "set_shared_genus_dir"):
                runner.set_shared_genus_dir(_shared_dir)
                _innovus_only_genus_ok = True
                _innovus_only_shared_genus_dir = _shared_dir
                print(f"[INNOVUS-ONLY] Reusing genus from run_id={SHARED_GENUS_RUN_ID}")
        if (not _innovus_only_genus_ok) and any(r.get("run_id") == "genus_shared" and r.get("success") for r in history_all):
            _shared_dir = _infer_shared_genus_dir(cfg, runner, "genus_shared")
            if _shared_dir and hasattr(runner, "set_shared_genus_dir"):
                runner.set_shared_genus_dir(_shared_dir)
                _innovus_only_genus_ok = True
                _innovus_only_shared_genus_dir = _shared_dir
                print(f"[INNOVUS-ONLY] Genus already done in previous session, reusing: {_shared_dir}")
        if not _innovus_only_genus_ok:
            print("[INNOVUS-ONLY] Running initial Genus with default params...")
            try:
                _genus_ok, _genus_log, _genus_remote_dir = runner.run_initial_genus(
                    params=default_params,
                    run_id="genus_shared",
                    timeout_sec=FULL_TIMEOUT,
                )
                _innovus_only_genus_ok = bool(_genus_ok)
                _innovus_only_shared_genus_dir = str(_genus_remote_dir or "")
                if _genus_ok:
                    print(f"[INNOVUS-ONLY] Initial Genus succeeded: {_genus_remote_dir}")
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
                    _slim_history_entry(_genus_res)
                    history_all.append(_genus_res)
                    kb.append_run(_genus_res, include_log_tail=False)
                    if _genus_cls.get("metrics_valid"):
                        history_full_valid.append(_genus_res)
                    _persist_resume_state()
                    print(f"[INNOVUS-ONLY] Warmup point added: valid={_genus_cls['metrics_valid']}")
                else:
                    print("[INNOVUS-ONLY][FATAL] Initial Genus failed, cannot proceed")
                    _trigger_fatal_stop("Initial Genus failed in innovus-only mode")
            except Exception as _ge:
                print(f"[INNOVUS-ONLY][FATAL] Initial Genus exception: {_ge}")
                _trigger_fatal_stop(f"Initial Genus exception: {_ge}")

    inflight: Dict = {}

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        while inflight or can_launch_new():
            _check_stop_file()

            while len(inflight) < MAX_CONCURRENCY and can_launch_new():
                # (A) warmup_quick
                if not warmup_quick_done and warmup_quick_success < WARMUP_QUICK:
                    spec = _propose_quick(stage="warmup_quick")
                    if spec is None:
                        break
                else:
                    warmup_quick_done = True

                    # (B) warmup full via promotion
                    if len(history_full_valid) < 6:
                        spec = _pick_promotion_full()
                        if spec is None:
                            spec = _propose_quick()
                    # (C) normal: mix promote and quick
                    else:
                        do_promote = (len(promo_pool) > 0) and (random.random() < PROMOTE_PROB)
                        if do_promote:
                            spec = _pick_promotion_full()
                            if spec is None:
                                spec = _propose_quick()
                        else:
                            spec = _propose_quick()

                    if spec is None:
                        break

                # Launch
                print(
                    f"[LAUNCH] {spec.run_id} fidelity={spec.fidelity} stage={spec.stage} "
                    f"by={spec.proposed_by} strategy={spec.proposed_strategy}"
                )
                print(f"         params={json.dumps(spec.params, ensure_ascii=False)}")
                spec.launch_wall_sec = time.time() - start_wall
                fut = ex.submit(_evaluate_one, runner, spec)
                inflight[fut] = spec
                launched += 1

            if not inflight:
                break

            if time.time() > deadline:
                print("[OVERTIME] deadline passed, waiting for %d inflight tasks to finish..." % len(inflight))

            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)


            for fut in done:
                spec = inflight.pop(fut)
                try:
                    res = fut.result()
                except _FATAL_REMOTE_EXCEPTIONS as e:
                    # Distinguish recoverable connection issues from scheduler failures.
                    if isinstance(e, (SchedulerBackpressureError, ControlledStop)):
                        # Scheduler-level failures or explicit stops remain fatal.
                        _trigger_fatal_stop(f"{type(e).__name__}: {str(e)[:200]}")
                    else:
                        # Remote connection failures are not fatal. Mark this run as failed
                        # and allow the next run to establish a fresh connection.
                        print(f"[WARN] Run {spec.run_id} lost connection ({type(e).__name__}), "
                              f"marking as failed but NOT stopping experiment. "
                              f"Will retry with new connection on next iteration.")
                    res = {
                        "run_id": spec.run_id, "parent_run_id": spec.parent_run_id,
                        "fidelity": spec.fidelity, "stage": spec.stage, "params": spec.params,
                        "metrics": {"area": float("inf"), "wns": -100.0, "power": float("inf"),
                                    "__area_sentinel__": True, "__wns_sentinel__": True, "__power_sentinel__": True},
                        "success": False, "metrics_status": "sentinel",
                        "metrics_usable": False, "metrics_valid": False,
                        "log_path": "", "runtime_sec": 0.0,
                        "proposed_by": spec.proposed_by, "proposed_strategy": spec.proposed_strategy,
                        "proposed_notes": spec.notes, "proposed_model": "",
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                    }
                except Exception as e:
                    print(f"[ERROR] Run {spec.run_id} failed: {e}")
                    res = {
                        "run_id": spec.run_id, "parent_run_id": spec.parent_run_id,
                        "fidelity": spec.fidelity, "stage": spec.stage, "params": spec.params,
                        "metrics": {"area": float("inf"), "wns": -100.0, "power": float("inf"),
                                    "__area_sentinel__": True, "__wns_sentinel__": True, "__power_sentinel__": True},
                        "success": False, "metrics_status": "sentinel",
                        "metrics_usable": False, "metrics_valid": False,
                        "log_path": "", "runtime_sec": 0.0,
                        "proposed_by": spec.proposed_by, "proposed_strategy": spec.proposed_strategy,
                        "proposed_notes": spec.notes, "proposed_model": "",
                        "error": f"{type(e).__name__}: {str(e)}",
                    }
                # Repeated connection-failure protection
                _is_conn_err = "RemoteCommError" in str(res.get("error", "")) or "RemoteSessionLostError" in str(res.get("error", ""))
                _on_run_result(success=res.get("success", False), is_conn_error=_is_conn_err)
                _end_wall = time.time()

                # ── Enrich result (same as main.py) ──
                res["wall_start_sec"] = spec.launch_wall_sec
                res["wall_end_sec"] = _end_wall - start_wall
                res["experiment_ts"] = start_wall
                res["experiment_id"] = cfg.get("project_name", "unknown")
                res["cum_launched"] = launched
                res["cum_history_all"] = len(history_all) + 1
                res["cum_full_valid"] = len(history_full_valid)
                res["failure_category"] = _classify_failure(res)
                res["ehvi_score"] = spec.ehvi_score
                res["ehvi_time_ucb"] = spec.ehvi_time_ucb

                # HV tracking
                _tmp_fv = list(history_full_valid)
                if spec.fidelity == "full" and res.get("metrics_valid", False):
                    _tmp_fv.append(res)
                _hv_after = _compute_hv(_tmp_fv, ref_point_list=_global_ref_point)
                _hv_before = _compute_hv(history_full_valid, ref_point_list=_global_ref_point)
                res["cum_hv"] = _hv_after
                res["cum_hv_delta"] = _hv_after - _hv_before

                # Pareto status
                _pareto_snap = _pareto_snapshot(_tmp_fv if _tmp_fv else history_all, max_items=9999)
                _pareto_ids = set(p.get("run_id") for p in _pareto_snap)
                res["is_pareto"] = (res.get("run_id") in _pareto_ids)
                res["pareto_size"] = len(_pareto_snap)
                res["cum_full_valid"] = len(_tmp_fv)

                # Baseline-specific: no knowledge state
                res["num_hard_rules"] = 0
                res["num_soft_heuristics"] = 0
                res["ka_buffer_size"] = 0
                res["design_prior_len"] = 0
                res["failure_analysis_len"] = 0
                res["sensitivity_text_len"] = 0

                # Runtime prediction
                try:
                    if spec.fidelity == "quick":
                        _rt_mean, _rt_std = rt.quick.predict(fill_params(spec.params))
                    else:
                        _rt_mean, _rt_std = rt.full.predict(fill_params(spec.params))
                    res["predicted_runtime_mean"] = float(_rt_mean)
                    res["predicted_runtime_std"] = float(_rt_std)
                    _actual_rt = float(res.get("runtime_sec", 0.0))
                    if _actual_rt > 0 and _rt_mean > 0:
                        res["runtime_prediction_error"] = abs(_actual_rt - _rt_mean) / _actual_rt
                except Exception:
                    res["predicted_runtime_mean"] = None
                    res["predicted_runtime_std"] = None
                    res["runtime_prediction_error"] = None

                # Slim down history entries to avoid GB-scale memory growth.
                _slim_history_entry(res)
                history_all.append(res)
                kb.append_run(res, include_log_tail=True)

                # ── Per-iteration checkpoint (for dedup on resume) ──
                _save_iteration_checkpoint(Path(cfg["work_dir"]), res, spec.__dict__ if hasattr(spec, '__dict__') else None)

                # Update runtime surrogate
                if res.get("success", False):
                    rt.observe(spec.fidelity, fill_params(spec.params), res.get("runtime_sec", 0.0))
                    rt.fit(spec.fidelity)

                # Quick usable → promo pool
                if spec.fidelity == "quick" and res.get("metrics_usable", False):
                    warmup_quick_success += 1
                    fp = _fingerprint(fill_params(spec.params))
                    if fp not in promo_pool and len(promo_pool) < MAX_PROMO_POOL:
                        promo_pool[fp] = (fill_params(spec.params), spec.run_id)

                # Full valid → QoR history
                if spec.fidelity == "full" and res.get("metrics_valid", False):
                    history_full_valid.append(res)

                # Update global ref point after first few full valid
                if len(history_full_valid) >= 3 and _global_ref_point == [-5e6, -10.0, -5.0]:
                    _global_ref_point[:] = compute_global_ref_point(history_full_valid)
                    print(f"[HV] Updated global ref point: {_global_ref_point}")

                # Print
                phase = "1(quick)" if res.get("fidelity") == "quick" else "2(full)"
                print(
                    f"[DONE]  {res.get('run_id')} phase={phase} stage={res.get('stage','')} "
                    f"success={res.get('success')} valid={res.get('metrics_valid',False)} "
                    f"runtime={float(res.get('runtime_sec',0)):.1f}s "
                    f"{_metrics_str(res.get('metrics',{}) or {})} "
                    f"strategy={res.get('proposed_strategy','')}"
                )
                print(
                    "        rem=%.2fh inflight=%d launched=%d promo_pool=%d full_valid=%d"
                    % (remaining_sec()/3600.0, len(inflight), launched,
                       len(promo_pool), len(history_full_valid))
                )
                if IS_INNOVUS_ONLY and ibe is not None:
                    _pct = launched / max(1, MAX_ITERATIONS) * 100
                    _er = ibe.remaining_sec(max(0, MAX_ITERATIONS - launched))
                    print("        [iter] %d/%d (%.1f%%) est_remain=%.0fs (%.1fh) eff_est=%.0fs/iter"
                          % (launched, MAX_ITERATIONS, _pct, _er,
                             _er/(3600.0*max(1,MAX_CONCURRENCY)), ibe.effective_est_sec))

                _persist_resume_state()

                if CLEANUP_EVERY > 0 and (len(history_all) % CLEANUP_EVERY == 0):
                    try:
                        runner.cleanup(history_all)
                    except Exception:
                        pass
                    # Periodically force garbage collection to reclaim GP and BoTorch temporaries.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


    # ── Finalize ──
    if fatal_stop_requested:
        print(f"[FATAL] scheduling stopped: {fatal_stop_reason}")

    try:
        runner.cleanup(history_all)
    except Exception:
        pass

    _persist_resume_state()

    pareto_path = kb.export_pareto(history_full_valid)
    csv_path = kb.export_csv(history_all)

    _final_hv = _compute_hv(history_full_valid, ref_point_list=_global_ref_point)
    _final_pareto = _pareto_snapshot(history_full_valid if history_full_valid else history_all, max_items=9999)
    _total_wall_sec = time.time() - start_wall
    _total_success = sum(1 for h in history_all if h.get("success", False))
    _total_fail = len(history_all) - _total_success

    experiment_summary = {
        "experiment_id": cfg.get("project_name", "unknown"),
        "baseline_strategy": STRATEGY,
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
        # No knowledge state for baselines
        "final_hard_rules": 0,
        "final_soft_heuristics": 0,
        "final_design_prior_len": 0,
        # Ablation flags
        "use_ehvi": True,
        "ka_enabled": False,
        "persist_rules": False,
        # Best points
        "best_wns": max((float(h["metrics"].get("wns", -1e9)) for h in history_full_valid if h.get("metrics")), default=None),
        "best_area": min((float(h["metrics"].get("area", 1e9)) for h in history_full_valid if h.get("metrics")), default=None),
        "best_power": min((float(h["metrics"].get("power", 1e9)) for h in history_full_valid if h.get("metrics")), default=None),
        # Pareto points
        "pareto_points": [
            {"run_id": p.get("run_id"), "area": p["metrics"].get("area"),
             "wns": p["metrics"].get("wns"), "power": p["metrics"].get("power")}
            for p in _final_pareto
        ],
        # HV trajectory
        "hv_trajectory": [
            {"wall_end_sec": h.get("wall_end_sec"), "cum_hv": h.get("cum_hv"), "run_id": h.get("run_id")}
            for h in history_all
            if h.get("cum_hv") is not None and h.get("wall_end_sec") is not None
        ],
    }

    _summary_path = Path(cfg["work_dir"]) / "experiment_summary.json"
    try:
        _summary_path.write_text(
            json.dumps(experiment_summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8"
        )
        print(f"Experiment summary: {_summary_path}")
    except Exception as e:
        print(f"[WARN] failed to write experiment_summary.json: {e}")

    print(f"\n=== Baseline {STRATEGY} Finished ===")
    print(f"All: {len(history_all)} | Full valid: {len(history_full_valid)} | Promo left: {len(promo_pool)}")
    print(f"Final HV: {_final_hv:.2f}")
    print(f"Final Pareto size: {len(_final_pareto)}")
    print(f"Pareto: {pareto_path}")
    print(f"CSV: {csv_path}")
    print(f"KB: {kb.runs_jsonl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run baseline experiments for EDATuner comparison")
    ap.add_argument("--config", "-c", required=True, help="Path to baseline YAML config")
    args = ap.parse_args()
    main(args.config)
