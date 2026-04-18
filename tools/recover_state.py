#!/usr/bin/env python3
"""
tools/recover_state.py
=======================
Rebuild EDATuner (main system / ablation) resume state from on-disk artifacts
when resume_state.json is missing, corrupted, or out-of-date.

This handles the MAIN EDATuner runs (with LLM/KA), not baselines.
For baselines, use tools/recover_baseline_state.py.

Recovery sources (priority order):
  1. kb/runs.jsonl              — per-run records (metrics, params, wall times)
  2. iter_*/checkpoint.json     — fingerprint + fidelity + phase info
  3. iter_*/metrics.json        — fallback metrics
  4. kb/hard_rules.json         — persisted knowledge rules
  5. kb/soft_heuristics.json    — soft heuristic rules
  6. kb/design_prior.json       — design prior state
  7. kb/failure_analysis.json   — accumulated failure analysis
  8. kb/sensitivity.json        — parameter sensitivity data
  9. experiment_summary.json    — global ref point, HV, metadata
 10. [NEW] Remote server        — fetch most recent max_concurrency runs via SSH

Rebuilds:
  - history_all / history_full_valid
  - seen_quick / seen_full (fingerprint dedup)
  - promo_pool (quick-usable → full promotion candidates)
  - warmup_quick_success
  - global_ref_point
  - launched counter
  - total_elapsed_sec
  - knowledge state (hard_rules, soft_heuristics, design_prior, etc.)
  - Optuna study state (if the run used an Optuna-based component)
  - Innovus-only mode state (tuning_mode, IBE observations, genus info)

Usage:
    # Basic recovery
    python -m tools.recover_state --work-dir ./runs/ablation_full

    # With config
    python -m tools.recover_state --work-dir ./runs/ablation_full --config configs/ablation_full.yaml

    # Dry run
    python -m tools.recover_state --work-dir ./runs/ablation_full --dry-run

    # Batch recovery for all ablation runs
    python -m tools.recover_state --batch-dir ./runs --prefix ablation_

    # Verify existing state
    python -m tools.recover_state --work-dir ./runs/ablation_full --verify

    # ★ Fetch recent runs from remote server then recover
    python -m tools.recover_state --work-dir ./runs/exp1 --config configs/exp1.yaml --fetch-remote
    python -m tools.recover_state --work-dir ./runs/exp1 --fetch-remote --cred-file ~/.ssh/creds.json
    python -m tools.recover_state --work-dir ./runs/exp1 --fetch-remote --max-concurrency 16 --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.EDATuner.utils.utils import load_config
except ImportError:
    load_config = None

# Optional dependencies for remote fetching.
_HAS_REMOTE = False
try:
    from src.EDATuner.runners.remote_cadence_support import (
        RemoteConnection, load_credentials, shell_quote as _remote_shell_quote,
    )
    _HAS_REMOTE = True
except ImportError:
    try:
        from src.remote_cadence_support import (
            RemoteConnection, load_credentials, shell_quote as _remote_shell_quote,
        )
        _HAS_REMOTE = True
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _run_id_index(run_id: str) -> int:
    try:
        if run_id.startswith("iter_"):
            return int(run_id.split("_", 1)[1])
    except Exception:
        pass
    return -1


def _sort_by_run_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort records: non-iter_* (e.g. genus_shared) first, then iter_* by index."""
    def _sort_key(r):
        rid = str(r.get("run_id", ""))
        idx = _run_id_index(rid)
        if idx >= 0:
            return (1, idx)       # iter_* sorted by index
        return (0, rid)           # non-iter_* sorted alphabetically, placed first
    return sorted(records, key=_sort_key)


def _classify_metrics(metrics: Dict[str, Any], fidelity: str) -> Dict[str, Any]:
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


def _compute_global_ref_point(history_full_valid: List[Dict[str, Any]]) -> List[float]:
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
    import statistics
    y_min = [min(p[i] for p in pts) for i in range(3)]
    y_abs_mean = [statistics.mean(abs(p[i]) for p in pts) for i in range(3)]
    slack = [0.1 * max(y_abs_mean[i], 1.0) for i in range(3)]
    return [y_min[i] - slack[i] for i in range(3)]


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue
    return records


def _load_json_file(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_runs_jsonl(work_dir: Path) -> List[Dict[str, Any]]:
    candidates = [
        work_dir / "kb" / "runs.jsonl",
        work_dir / "runs.jsonl",
    ]
    for p in candidates:
        records = _load_jsonl(p)
        if records:
            print(f"  [OK] loaded {len(records)} records from {p}")
            return records
    return []


# Additional non-iter_* run directory prefixes to scan in innovus-only mode.
_EXTRA_RUN_DIR_PREFIXES = ("genus_shared",)


def _scan_iter_dirs(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan iter_* directories AND known non-iter_* run dirs (e.g. genus_shared)."""
    found = {}
    if not work_dir.is_dir():
        return found
    for d in sorted(work_dir.iterdir()):
        if not d.is_dir():
            continue
        # Also scan non-iter_* directories such as genus_shared.
        is_iter = d.name.startswith("iter_")
        is_extra = any(d.name.startswith(pfx) for pfx in _EXTRA_RUN_DIR_PREFIXES)
        if not is_iter and not is_extra:
            continue

        run_id = d.name
        entry: Dict[str, Any] = {"run_id": run_id, "_dir": str(d)}

        for fname in ("checkpoint.json", "result.json", "metrics.json"):
            fpath = d / fname
            if fpath.is_file():
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        if fname == "metrics.json" and "metrics" not in entry:
                            entry["metrics"] = data
                        else:
                            for k, v in data.items():
                                if k.startswith("_"):
                                    continue
                                if k not in entry or entry[k] is None:
                                    entry[k] = v
                except Exception:
                    pass
        found[run_id] = entry
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge state recovery
# ═══════════════════════════════════════════════════════════════════════════════

def _recover_knowledge_state(work_dir: Path) -> Dict[str, Any]:
    """
    Recover accumulated knowledge state from KB directory.
    Returns dict with hard_rules, soft_heuristics, design_prior, etc.
    """
    kb_dir = work_dir / "kb"
    knowledge = {
        "hard_rules": [],
        "soft_heuristics": [],
        "design_prior": {},
        "failure_analysis": {},
        "sensitivity": {},
        "num_hard_rules": 0,
        "num_soft_heuristics": 0,
        "design_prior_len": 0,
        "failure_analysis_len": 0,
        "sensitivity_text_len": 0,
    }

    if not kb_dir.is_dir():
        return knowledge

    # Hard rules
    hr = _load_json_file(kb_dir / "hard_rules.json")
    if isinstance(hr, list):
        knowledge["hard_rules"] = hr
        knowledge["num_hard_rules"] = len(hr)
    elif isinstance(hr, dict):
        rules = hr.get("rules", [])
        if isinstance(rules, list):
            knowledge["hard_rules"] = rules
            knowledge["num_hard_rules"] = len(rules)

    # Soft heuristics
    sh = _load_json_file(kb_dir / "soft_heuristics.json")
    if isinstance(sh, list):
        knowledge["soft_heuristics"] = sh
        knowledge["num_soft_heuristics"] = len(sh)
    elif isinstance(sh, dict):
        heurs = sh.get("heuristics", [])
        if isinstance(heurs, list):
            knowledge["soft_heuristics"] = heurs
            knowledge["num_soft_heuristics"] = len(heurs)

    # Design prior
    dp = _load_json_file(kb_dir / "design_prior.json")
    if isinstance(dp, dict):
        knowledge["design_prior"] = dp
        knowledge["design_prior_len"] = len(json.dumps(dp, default=str))

    # Failure analysis
    fa = _load_json_file(kb_dir / "failure_analysis.json")
    if isinstance(fa, dict):
        knowledge["failure_analysis"] = fa
        knowledge["failure_analysis_len"] = len(json.dumps(fa, default=str))

    # Sensitivity
    sens = _load_json_file(kb_dir / "sensitivity.json")
    if isinstance(sens, dict):
        knowledge["sensitivity"] = sens
        knowledge["sensitivity_text_len"] = len(json.dumps(sens, default=str))
    # Also check for text-based sensitivity
    sens_txt = kb_dir / "sensitivity.txt"
    if sens_txt.is_file():
        try:
            txt = sens_txt.read_text(encoding="utf-8")
            knowledge["sensitivity_text_len"] = max(
                knowledge["sensitivity_text_len"], len(txt)
            )
        except Exception:
            pass

    return knowledge


# ═══════════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cfg(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path and load_config:
        try:
            return load_config(config_path)
        except Exception as e:
            print(f"  [WARN] load_config failed: {e}")
    if config_path:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"  [WARN] yaml load failed: {e}")
    return {}


def _get_param_space(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(cfg.get("param_space"), list):
        return cfg["param_space"]
    tuning = cfg.get("tuning", {}) or {}
    if isinstance(tuning.get("param_space"), list):
        return tuning["param_space"]
    return []


def _get_default_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dp = cfg.get("default_params", {}) or {}
    return dict(dp) if isinstance(dp, dict) else {}


def _fill_params(p: Dict[str, Any], default_params: Dict[str, Any],
                 param_space: List[Dict[str, Any]]) -> Dict[str, Any]:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Detect run type (main vs ablation variant)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_run_type(work_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Detect ablation flags and tuning mode from config or experiment summary."""
    info = {
        "project_name": cfg.get("project_name", work_dir.name),
        "ka_enabled": True,
        "use_ehvi": True,
        "persist_rules": True,
        "llm_enabled": True,
        # ★ INNOVUS-ONLY: tuning mode fields
        "tuning_mode": "full",
        "max_iterations": 200,
        "estimated_innovus_sec": 1800.0,
        "shared_genus_run_id": "",
    }

    # From config
    ka_cfg = cfg.get("ka", {}) or {}
    llm_cfg = cfg.get("llm", {}) or {}
    tuning_cfg = cfg.get("tuning", {}) or {}

    if "enable" in ka_cfg:
        info["ka_enabled"] = bool(ka_cfg["enable"])
    if "enable" in llm_cfg:
        info["llm_enabled"] = bool(llm_cfg["enable"])
    if "persist_rules" in ka_cfg:
        info["persist_rules"] = bool(ka_cfg["persist_rules"])
    if "use_ehvi" in tuning_cfg:
        info["use_ehvi"] = bool(tuning_cfg["use_ehvi"])

    # ★ INNOVUS-ONLY: read tuning mode from config
    if "mode" in tuning_cfg:
        info["tuning_mode"] = str(tuning_cfg["mode"]).strip().lower()
    if "max_iterations" in tuning_cfg:
        info["max_iterations"] = int(tuning_cfg["max_iterations"])
    if "estimated_innovus_sec" in tuning_cfg:
        info["estimated_innovus_sec"] = float(tuning_cfg["estimated_innovus_sec"])
    if "shared_genus_run_id" in tuning_cfg:
        info["shared_genus_run_id"] = str(tuning_cfg["shared_genus_run_id"] or "").strip()

    # From experiment summary (overrides)
    summary = _load_json_file(work_dir / "experiment_summary.json")
    if isinstance(summary, dict):
        for k in ("ka_enabled", "use_ehvi", "persist_rules", "llm_enabled"):
            if k in summary:
                info[k] = bool(summary[k])
        # ★ INNOVUS-ONLY: override from experiment_summary
        if "tuning_mode" in summary:
            info["tuning_mode"] = str(summary["tuning_mode"]).strip().lower()
        if "max_iterations" in summary and summary["max_iterations"] is not None:
            info["max_iterations"] = int(summary["max_iterations"])
        if "estimated_innovus_sec" in summary and summary["estimated_innovus_sec"] is not None:
            info["estimated_innovus_sec"] = float(summary["estimated_innovus_sec"])

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# IBE (IterationBudgetEstimator) state recovery
# ═══════════════════════════════════════════════════════════════════════════════

def _recover_ibe_observations(history_all: List[Dict[str, Any]]) -> List[float]:
    """
    Extract successful run runtimes for IBE cold-start recovery.
    In innovus_only mode, IBE tracks actual runtimes to adaptively estimate
    time-per-iteration. We recover this from history.
    """
    observations: List[float] = []
    for h in history_all:
        if not h.get("success", False):
            continue
        rt = _safe_float(h.get("runtime_sec"), 0.0)
        if rt > 0 and math.isfinite(rt):
            observations.append(rt)
    return observations


# ═══════════════════════════════════════════════════════════════════════════════
# Remote run fetching (Phase 2.5)
# ═══════════════════════════════════════════════════════════════════════════════

# Only download these lightweight files during remote recovery.
_REMOTE_FETCH_FILENAMES = (
    "checkpoint.json", "metrics.json", "result.json",
)
# Optionally download report files with these suffixes as well.
_REMOTE_FETCH_SUFFIXES = {".rpt", ".log", ".txt"}

# checkpoint.current_phase values indicating that the run may still be active on the remote host.
_INFLIGHT_PHASES = frozenset({
    "connect",
    "genus_submit", "genus_done",
    "submit",             # Innovus was submitted and is waiting to be polled.
    "poll",               # Polling is in progress.
    "session_lost",       # SSH session was lost, but the LSF job may still be alive.
})

# LSF states that indicate the job is still alive.
_LSF_ALIVE_STATUSES = frozenset({"RUN", "PEND", "SSUSP", "USUSP", "PSUSP"})


def _check_lsf_job_alive(
    conn: "RemoteConnection",
    jobid: Optional[str],
) -> Optional[str]:
    """
    Check whether an LSF job is still alive.
    Return the status string (RUN/PEND/DONE/EXIT/NOT_FOUND/...) or None if the job id is invalid.
    """
    if not jobid or str(jobid).lower() in ("reused", "none", ""):
        return None
    try:
        cmd = f"bjobs -o 'stat' -noheader {_remote_shell_quote(jobid)} 2>&1"
        rc, out, err = conn.exec_cmd_resilient(cmd, timeout=60)
        combined = (out + err).strip()
        if "not found" in combined.lower() or "not exist" in combined.lower() or "illegal" in combined.lower():
            return "NOT_FOUND"
        if not combined:
            return "UNKNOWN"
        status = combined.splitlines()[-1].strip().upper()
        return status
    except Exception:
        return "UNKNOWN"


def _detect_inflight_status(
    conn: "RemoteConnection",
    local_run_dir: Path,
) -> Dict[str, Any]:
    """
    Detect whether a run is still active on the remote host.
    Return a dict with:
      - inflight: bool
      - genus_jobid, innovus_jobid: str or None
      - genus_status, innovus_status: str or None
      - phase: str  (checkpoint current_phase)
      - params: dict (restored from the checkpoint when available)
      - fidelity: str
      - stage: str
    """
    info: Dict[str, Any] = {
        "inflight": False,
        "genus_jobid": None,
        "innovus_jobid": None,
        "genus_status": None,
        "innovus_status": None,
        "phase": "",
        "params": {},
        "fidelity": "quick",
        "stage": "",
        "params_fingerprint": "",
    }

    ckpt = _load_json_file(local_run_dir / "checkpoint.json")
    if not isinstance(ckpt, dict):
        return info

    phase = str(ckpt.get("current_phase", ""))
    info["phase"] = phase
    info["genus_jobid"] = ckpt.get("genus_jobid")
    info["innovus_jobid"] = ckpt.get("innovus_jobid")
    info["fidelity"] = str(ckpt.get("fidelity", "quick"))
    info["stage"] = str(ckpt.get("stage", ""))
    info["params_fingerprint"] = str(ckpt.get("params_fingerprint", ""))

    # Try to restore params from the checkpoint.
    if isinstance(ckpt.get("params"), dict):
        info["params"] = ckpt["params"]

    if phase not in _INFLIGHT_PHASES:
        return info

    # If the phase suggests an inflight run, check the LSF job status.
    any_alive = False

    genus_jid = info["genus_jobid"]
    innovus_jid = info["innovus_jobid"]

    if genus_jid and str(genus_jid).lower() != "reused":
        gs = _check_lsf_job_alive(conn, genus_jid)
        info["genus_status"] = gs
        if gs in _LSF_ALIVE_STATUSES:
            any_alive = True

    if innovus_jid:
        ivs = _check_lsf_job_alive(conn, innovus_jid)
        info["innovus_status"] = ivs
        if ivs in _LSF_ALIVE_STATUSES:
            any_alive = True

    # Special case: session_lost but the job has already finished -> not truly inflight,
    # but the checkpoint still says session_lost, so the runner will repoll on resume and detect completion.
    if any_alive:
        info["inflight"] = True
    elif phase == "session_lost":
        # The job has already finished, but the local checkpoint was never updated to done.
        # Still mark it as needing re-attachment; the runner will resume polling and immediately observe completion.
        info["inflight"] = True

    return info


def _get_remote_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract remote-related settings from the config."""
    remote_cfg = cfg.get("remote", {}) or {}
    cadence_cfg = cfg.get("cadence", {}) or {}
    cadence_remote = cadence_cfg.get("remote", {}) or {}
    tuning_cfg = cfg.get("tuning", {}) or {}

    # max_concurrency priority: remote > tuning > cadence.remote > default 8.
    max_conc = (
        remote_cfg.get("max_concurrency")
        or tuning_cfg.get("max_concurrency")
        or cadence_remote.get("max_concurrency")
        or 8
    )
    # Remote base directory (parent directory of each iter_* run).
    remote_base = (
        remote_cfg.get("remote_base_dir")
        or cadence_remote.get("remote_base_dir")
        or remote_cfg.get("base_dir")
        or ""
    )
    # Credential file.
    cred_file = (
        remote_cfg.get("cred_file")
        or cadence_remote.get("cred_file")
        or remote_cfg.get("credentials")
        or ""
    )

    return {
        "max_concurrency": int(max_conc),
        "remote_base_dir": str(remote_base).strip(),
        "cred_file": str(cred_file).strip(),
    }


def _list_remote_iter_dirs(
    conn: "RemoteConnection",
    remote_base: str,
) -> List[Tuple[str, int]]:
    """List iter_* directories under remote_base and return [(dir_name, index), ...] sorted by descending index."""
    # Use ls to list directories and filter for iter_* names.
    cmd = f"ls -1d {_remote_shell_quote(remote_base)}/iter_*/ 2>/dev/null || true"
    rc, out, err = conn.exec_cmd_resilient(cmd, timeout=60)
    results: List[Tuple[str, int]] = []
    for line in (out or "").strip().splitlines():
        line = line.strip().rstrip("/")
        if not line:
            continue
        import posixpath as _pp
        dirname = _pp.basename(line)
        idx = _run_id_index(dirname)
        if idx >= 0:
            results.append((dirname, idx))
    # Sort by descending index so the newest runs come first.
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _download_run_metadata(
    conn: "RemoteConnection",
    remote_base: str,
    run_dir_name: str,
    local_dest: Path,
) -> bool:
    """
    Download metadata files for a single run directory from the remote host to the local machine.
    Return True if at least one valid file was downloaded.
    """
    import posixpath as _pp
    remote_run_dir = _pp.join(remote_base, run_dir_name)
    local_run_dir = local_dest / run_dir_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    downloaded_any = False

    # 1) Download core metadata files.
    for fname in _REMOTE_FETCH_FILENAMES:
        remote_path = _pp.join(remote_run_dir, fname)
        local_path = local_run_dir / fname
        if local_path.is_file():
            # Skip files that already exist locally.
            downloaded_any = True
            continue
        try:
            if conn.path_exists(remote_path):
                conn.get_file(remote_path, str(local_path))
                downloaded_any = True
        except Exception as e:
            print(f"    [WARN] failed to download {remote_path}: {e}")

    # 2) Optionally download report files under outbox/cadence/ via a lightweight scan.
    for subdir in ("outbox/cadence/innovus", "outbox/cadence/genus"):
        remote_sub = _pp.join(remote_run_dir, subdir)
        local_sub = local_run_dir / subdir
        try:
            # Use find to list report files and avoid recursively downloading large directories.
            find_cmd = (
                f"find {_remote_shell_quote(remote_sub)} -maxdepth 1 -type f "
                f"\\( -name '*.rpt' -o -name '*.txt' \\) 2>/dev/null || true"
            )
            rc, out, _err = conn.exec_cmd_resilient(find_cmd, timeout=30)
            for fline in (out or "").strip().splitlines():
                fline = fline.strip()
                if not fline:
                    continue
                fname = _pp.basename(fline)
                local_file = local_sub / fname
                if local_file.is_file():
                    continue
                try:
                    local_sub.mkdir(parents=True, exist_ok=True)
                    conn.get_file(fline, str(local_file))
                except Exception:
                    pass
        except Exception:
            pass

    return downloaded_any


def _validate_remote_run(local_run_dir: Path) -> Dict[str, Any]:
    """
    Validate whether a remotely downloaded run directory contains usable data.
    Return a validation dict containing fields such as valid, reason, and metrics.
    Note: inflight status is determined separately by _detect_inflight_status.
    """
    result = {
        "valid": False,
        "reason": "",
        "has_checkpoint": False,
        "has_metrics": False,
        "has_result": False,
        "checkpoint_phase": "",
        "possibly_inflight": False,
    }

    # Check checkpoint.json.
    ckpt = _load_json_file(local_run_dir / "checkpoint.json")
    if isinstance(ckpt, dict):
        result["has_checkpoint"] = True
        result["checkpoint_phase"] = str(ckpt.get("current_phase", ""))

        # A checkpoint counts as complete only when current_phase is "done".
        if result["checkpoint_phase"] != "done":
            # The run may still be active.
            if result["checkpoint_phase"] in _INFLIGHT_PHASES:
                result["possibly_inflight"] = True
                result["reason"] = f"checkpoint phase={result['checkpoint_phase']} (possibly inflight)"
            else:
                result["reason"] = f"checkpoint phase={result['checkpoint_phase']} (not done)"
            return result

    # Check metrics.json.
    metrics_data = _load_json_file(local_run_dir / "metrics.json")
    if isinstance(metrics_data, dict) and metrics_data:
        result["has_metrics"] = True

    # Check result.json.
    result_data = _load_json_file(local_run_dir / "result.json")
    if isinstance(result_data, dict):
        result["has_result"] = True

    # Consider the run valid if it has either a done checkpoint or usable metrics.
    if result["has_checkpoint"] and result["checkpoint_phase"] == "done":
        result["valid"] = True
        result["reason"] = "checkpoint done"
    elif result["has_metrics"] and result["has_result"]:
        result["valid"] = True
        result["reason"] = "metrics+result present"
    elif result["has_metrics"]:
        result["valid"] = True
        result["reason"] = "metrics present (no checkpoint)"
    else:
        result["reason"] = "no valid metadata found"

    return result


def _fetch_remote_recent_runs(
    work_dir: Path,
    cfg: Dict[str, Any],
    *,
    cred_file_override: Optional[str] = None,
    max_concurrency_override: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Fetch metadata for the most recent max_concurrency runs from the remote server.

    Only fetch runs that are not already present locally to avoid duplicate downloads.
    Also detect which runs are still active on the remote host by checking LSF job state.

    Returns:
        (fetched_valid_count, inflight_runs)
        - fetched_valid_count: number of completed and valid runs.
        - inflight_runs: list of runs still active on the remote host, each containing:
            run_id, params, fidelity, stage, genus_jobid, innovus_jobid,
            genus_status, innovus_status, checkpoint_phase, params_fingerprint
    """
    if not _HAS_REMOTE:
        print("  [SKIP] remote_cadence_support not available, cannot fetch from remote")
        return 0, []

    remote_cfg = _get_remote_cfg(cfg)
    max_conc = max_concurrency_override or remote_cfg["max_concurrency"]
    remote_base = remote_cfg["remote_base_dir"]
    cred_path = cred_file_override or remote_cfg["cred_file"]

    if not remote_base:
        print("  [SKIP] remote_base_dir not configured, cannot fetch from remote")
        return 0, []
    if not cred_path:
        print("  [SKIP] cred_file not configured, cannot fetch from remote")
        return 0, []

    print(f"  remote_base_dir = {remote_base}")
    print(f"  max_concurrency = {max_conc}")
    print(f"  cred_file       = {cred_path}")

    # Load credentials.
    try:
        creds = load_credentials(cred_path)
    except Exception as e:
        print(f"  [ERROR] failed to load credentials: {e}")
        return 0, []

    # Connect to the remote host.
    conn = RemoteConnection(creds, timeout_sec=30, keepalive_sec=30)
    try:
        conn.connect()
        print(f"  [OK] connected to remote server")
    except Exception as e:
        print(f"  [ERROR] failed to connect to remote: {e}")
        return 0, []

    fetched_valid = 0
    inflight_runs: List[Dict[str, Any]] = []

    try:
        # List remote iter_* directories.
        remote_dirs = _list_remote_iter_dirs(conn, remote_base)
        print(f"  [OK] found {len(remote_dirs)} iter_* directories on remote")

        if not remote_dirs:
            return 0, []

        # Take the most recent max_concurrency runs.
        recent = remote_dirs[:max_conc]
        print(f"  [INFO] checking most recent {len(recent)} runs: "
              f"iter_{recent[0][1]}..iter_{recent[-1][1]}")

        # Find runs that already have a local done checkpoint so they can be skipped.
        local_done = set()
        local_has_ckpt = set()
        for d in work_dir.iterdir():
            if d.is_dir() and d.name.startswith("iter_"):
                ckpt_file = d / "checkpoint.json"
                if ckpt_file.is_file():
                    local_has_ckpt.add(d.name)
                    ckpt_data = _load_json_file(ckpt_file)
                    if isinstance(ckpt_data, dict) and ckpt_data.get("current_phase") == "done":
                        local_done.add(d.name)

        if dry_run:
            to_check = [(n, i) for n, i in recent if n not in local_done]
            print(f"  [DRY RUN] would fetch/check {len(to_check)} runs "
                  f"(skipping {len(recent) - len(to_check)} locally done)")
            return 0, []

        # Process runs one by one.
        for dir_name, idx in recent:
            # Runs already completed locally can be counted immediately.
            if dir_name in local_done:
                validation = _validate_remote_run(work_dir / dir_name)
                if validation["valid"]:
                    fetched_valid += 1
                continue

            # Download metadata.
            print(f"    fetching {dir_name} ...", end=" ", flush=True)
            ok = _download_run_metadata(conn, remote_base, dir_name, work_dir)
            if not ok:
                print("no files found")
                continue

            validation = _validate_remote_run(work_dir / dir_name)

            if validation["valid"]:
                print(f"OK ({validation['reason']})")
                fetched_valid += 1
                continue

            # If the run is not done and may still be inflight, query the LSF job status.
            if validation["possibly_inflight"]:
                print(f"checking LSF ...", end=" ", flush=True)
                inflight_info = _detect_inflight_status(conn, work_dir / dir_name)
                if inflight_info["inflight"]:
                    g_st = inflight_info["genus_status"] or "n/a"
                    i_st = inflight_info["innovus_status"] or "n/a"
                    print(f"INFLIGHT (phase={inflight_info['phase']} "
                          f"genus={g_st} innovus={i_st})")
                    inflight_runs.append({
                        "run_id": dir_name,
                        "params": inflight_info["params"],
                        "fidelity": inflight_info["fidelity"],
                        "stage": inflight_info["stage"],
                        "genus_jobid": inflight_info["genus_jobid"],
                        "innovus_jobid": inflight_info["innovus_jobid"],
                        "genus_status": inflight_info["genus_status"],
                        "innovus_status": inflight_info["innovus_status"],
                        "checkpoint_phase": inflight_info["phase"],
                        "params_fingerprint": inflight_info["params_fingerprint"],
                    })
                else:
                    print(f"DEAD (phase={inflight_info['phase']} "
                          f"jobs not alive)")
            else:
                print(f"INVALID ({validation['reason']})")

    except Exception as e:
        print(f"  [ERROR] remote fetch failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

    if inflight_runs:
        print(f"  [INFO] detected {len(inflight_runs)} inflight runs on remote server")

    return fetched_valid, inflight_runs


# ═══════════════════════════════════════════════════════════════════════════════
# Main recovery
# ═══════════════════════════════════════════════════════════════════════════════

def recover_state(
    work_dir: str | Path,
    config_path: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    fetch_remote: bool = False,
    cred_file: Optional[str] = None,
    max_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Recover EDATuner main/ablation resume state from on-disk artifacts.

    Args:
        fetch_remote: whether to pull recent run artifacts from the remote server.
        cred_file: path to the SSH credential file (overrides the config).
        max_concurrency: maximum number of runs to fetch (overrides the config).
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        raise FileNotFoundError(f"work_dir does not exist: {work_dir}")

    resume_path = work_dir / "resume_state.json"
    if resume_path.is_file() and not force:
        print(f"[WARN] {resume_path} already exists. Use --force to overwrite.")

    # ── Load config ──
    cfg = _load_cfg(config_path)
    param_space = _get_param_space(cfg)
    default_params = _get_default_params(cfg)
    fid_cfg = cfg.get("fidelity", {}) or {}
    warmup_quick_target = int(fid_cfg.get("warmup_quick", 8))
    max_promo_pool = int(fid_cfg.get("max_promo_pool", 2000))

    run_info = _detect_run_type(work_dir, cfg)
    is_innovus_only = (run_info["tuning_mode"] == "innovus_only")

    print(f"\n{'='*60}")
    print(f"Recovering EDATuner state: {work_dir}")
    print(f"Project: {run_info['project_name']}")
    print(f"KA={run_info['ka_enabled']} EHVI={run_info['use_ehvi']} "
          f"Persist={run_info['persist_rules']} LLM={run_info['llm_enabled']}")
    if is_innovus_only:
        print(f"Mode: INNOVUS_ONLY  max_iterations={run_info['max_iterations']} "
              f"est_innovus_sec={run_info['estimated_innovus_sec']}")
    else:
        print(f"Mode: {run_info['tuning_mode']}")
    print(f"Config: {config_path or '(none)'}")
    print(f"{'='*60}\n")

    # ── Phase 1: Load runs.jsonl ──
    print("[Phase 1] Loading runs.jsonl ...")
    jsonl_records = _load_runs_jsonl(work_dir)

    # ── Phase 2: Scan iter_* directories (and genus_shared, etc.) ──
    print("[Phase 2] Scanning run directories ...")
    dir_entries = _scan_iter_dirs(work_dir)
    _extra_count = sum(1 for k in dir_entries if not k.startswith("iter_"))
    print(f"  [OK] found {len(dir_entries)} run directories "
          f"({len(dir_entries) - _extra_count} iter_* + {_extra_count} extra)")

    # ── Phase 2.5: Fetch recent runs from remote server ──
    remote_fetched_count = 0
    inflight_runs: List[Dict[str, Any]] = []
    if fetch_remote:
        print("[Phase 2.5] Fetching recent runs from remote server ...")
        remote_fetched_count, inflight_runs = _fetch_remote_recent_runs(
            work_dir,
            cfg,
            cred_file_override=cred_file,
            max_concurrency_override=max_concurrency,
            dry_run=dry_run,
        )
        if remote_fetched_count > 0 or inflight_runs:
            # Rescan local directories so newly downloaded runs are included.
            print(f"  [OK] fetched {remote_fetched_count} valid + "
                  f"{len(inflight_runs)} inflight, re-scanning ...")
            dir_entries = _scan_iter_dirs(work_dir)
            _extra_count = sum(1 for k in dir_entries if not k.startswith("iter_"))
            print(f"  [OK] now {len(dir_entries)} run directories "
                  f"({len(dir_entries) - _extra_count} iter_* + {_extra_count} extra)")
        else:
            print(f"  [INFO] no new valid runs fetched from remote")

    # ── Phase 3: Merge ──
    print("[Phase 3] Merging data sources ...")
    jsonl_by_id = {str(r.get("run_id", "")): r for r in jsonl_records}
    all_run_ids = set(jsonl_by_id.keys()) | set(dir_entries.keys())
    merged_records: List[Dict[str, Any]] = []

    for rid in sorted(all_run_ids, key=lambda x: (_run_id_index(x) if _run_id_index(x) >= 0 else -1, x)):
        jrec = jsonl_by_id.get(rid, {})
        drec = dir_entries.get(rid, {})

        if jrec:
            record = dict(jrec)
            for k, v in drec.items():
                if k.startswith("_"):
                    continue
                if k not in record or record[k] is None:
                    record[k] = v
        elif drec:
            record = dict(drec)
            record.setdefault("run_id", rid)
            record.setdefault("fidelity", drec.get("fidelity", "quick"))
            record.setdefault("success", True)
            metrics = record.get("metrics", {})
            if metrics:
                cls = _classify_metrics(metrics, record.get("fidelity", "quick"))
                record.update(cls)
        else:
            continue

        merged_records.append(record)

    merged_records = _sort_by_run_id(merged_records)
    print(f"  [OK] merged {len(merged_records)} total records")

    # ── Phase 4: Rebuild run state ──
    print("[Phase 4] Rebuilding run state ...")

    history_all: List[Dict[str, Any]] = []
    history_full_valid: List[Dict[str, Any]] = []
    seen_quick: Set[str] = set()
    seen_full: Set[str] = set()
    promo_pool: Dict[str, Dict[str, Any]] = {}
    warmup_quick_success: int = 0
    max_launched_idx: int = -1

    # Track KA buffer (knowledge agent pending observations)
    ka_buffer: List[Dict[str, Any]] = []

    # ★ INNOVUS-ONLY: track genus_shared run_id for reference
    genus_shared_found: bool = False

    for record in merged_records:
        rid = str(record.get("run_id", ""))
        fidelity = str(record.get("fidelity", "quick")).strip().lower()
        params = record.get("params", {}) or {}

        if param_space:
            filled = _fill_params(params, default_params, param_space)
        else:
            filled = dict(params) if params else {}

        fp = _fingerprint(filled)

        metrics = record.get("metrics", {}) or {}
        if "metrics_usable" not in record and metrics:
            cls = _classify_metrics(metrics, fidelity)
            record.update(cls)

        idx = _run_id_index(rid)
        if idx >= 0:
            max_launched_idx = max(max_launched_idx, idx)

        # ★ INNOVUS-ONLY: detect genus_shared
        if rid.startswith("genus_shared"):
            genus_shared_found = True
            # genus_shared is always full fidelity
            if not fidelity or fidelity == "quick":
                fidelity = "full"
                record["fidelity"] = "full"

        if fidelity == "full":
            seen_full.add(fp)
        else:
            seen_quick.add(fp)

        if fidelity == "quick" and record.get("metrics_usable", False):
            warmup_quick_success += 1
            if fp not in seen_full and len(promo_pool) < max_promo_pool:
                promo_pool[fp] = {
                    "params": filled,
                    "parent_run_id": rid,
                }

        if fidelity == "full" and record.get("metrics_valid", False):
            history_full_valid.append(record)

        # KA buffer: recent full-valid records that the knowledge agent
        # hasn't processed yet. We conservatively add the last N.
        if record.get("metrics_valid", False) or record.get("metrics_usable", False):
            ka_buffer.append({
                "run_id": rid,
                "params": filled,
                "metrics": metrics,
                "fidelity": fidelity,
                "success": record.get("success", False),
            })
        # Ensure each record has complete params; resumed MF-GP reconstruction depends on them.
        if filled and (not record.get("params")):
            record["params"] = filled

        history_all.append(record)

    # Remove from promo_pool anything already in seen_full
    for fp in list(promo_pool.keys()):
        if fp in seen_full:
            del promo_pool[fp]

    launched = max_launched_idx + 1 if max_launched_idx >= 0 else len(merged_records)

    # Group by session (experiment_ts) and accumulate each session's max(wall_end_sec).
    # Previous bug: taking max(wall_ends) kept only the largest single-session value and lost earlier resumed-session time.
    session_max_wall: Dict[int, float] = {}
    runtime_sum = 0.0
    for h in history_all:
        we = _safe_float(h.get("wall_end_sec"))
        ts = h.get("experiment_ts")
        if ts is not None and math.isfinite(we) and we > 0:
            ts_key = round(float(ts))
            session_max_wall[ts_key] = max(session_max_wall.get(ts_key, 0.0), we)
        rt = _safe_float(h.get("runtime_sec"), 0.0)
        if math.isfinite(rt) and rt > 0:
            runtime_sum += rt

    if session_max_wall:
        total_elapsed_sec = sum(session_max_wall.values())
        print(f"  [TIME] {len(session_max_wall)} session(s) detected: "
              + ", ".join(f"{v/3600:.2f}h" for v in sorted(session_max_wall.values(), reverse=True)))
    else:
        total_elapsed_sec = runtime_sum
        print(f"  [TIME] no wall_end_sec found, using runtime_sum={runtime_sum/3600:.2f}h (approximate)")

    # Global ref point
    if len(history_full_valid) >= 3:
        global_ref_point = _compute_global_ref_point(history_full_valid)
    else:
        summary = _load_json_file(work_dir / "experiment_summary.json")
        if isinstance(summary, dict) and isinstance(summary.get("global_ref_point"), list):
            global_ref_point = [float(x) for x in summary["global_ref_point"]]
        else:
            global_ref_point = [-5e6, -10.0, -5.0]

    # ── Phase 5: Recover knowledge state ──
    print("[Phase 5] Recovering knowledge state ...")
    knowledge = _recover_knowledge_state(work_dir)
    print(f"  hard_rules={knowledge['num_hard_rules']} "
          f"soft_heuristics={knowledge['num_soft_heuristics']} "
          f"design_prior_len={knowledge['design_prior_len']}")

    # ── Phase 6: Recover IBE state (innovus_only) ──
    ibe_observations: List[float] = []
    if is_innovus_only:
        print("[Phase 6] Recovering IBE state (innovus_only) ...")
        ibe_observations = _recover_ibe_observations(history_all)
        print(f"  [OK] {len(ibe_observations)} IBE runtime observations recovered")
        if ibe_observations:
            _ibe_mean = sum(ibe_observations) / len(ibe_observations)
            print(f"  [OK] mean observed runtime: {_ibe_mean:.1f}s ({_ibe_mean/3600:.2f}h)")
        if genus_shared_found:
            print(f"  [OK] genus_shared run detected in history")
        else:
            print(f"  [WARN] genus_shared run NOT found in history — "
                   "main.py may re-run genus on resume")

    # ── Build resume state ──
    # Use the same schema as main.py's _build_resume_state / _persist_resume_state
    promo_pool_json: Dict[str, Dict[str, Any]] = {}
    for fp, value in promo_pool.items():
        if isinstance(value, dict):
            promo_pool_json[fp] = value
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            promo_pool_json[fp] = {
                "params": value[0],
                "parent_run_id": str(value[1]),
            }

    state = {
        "version": 2,
        "runner": "edatuner",
        "project_name": run_info["project_name"],
        "config_path": config_path or "",
        "created_ts": time.time(),
        "recovered_ts": time.time(),
        "recovered": True,
        "work_dir": str(work_dir),
        "launched": launched,
        "total_elapsed_sec": total_elapsed_sec,
        "warmup_quick_success": warmup_quick_success,
        "history_all": history_all,
        "history_full_valid": history_full_valid,
        "seen_quick": sorted(seen_quick),
        "seen_full": sorted(seen_full),
        "promo_pool": promo_pool_json,
        "global_ref_point": global_ref_point,
        # Ablation flags
        "ka_enabled": run_info["ka_enabled"],
        "use_ehvi": run_info["use_ehvi"],
        "persist_rules": run_info["persist_rules"],
        "llm_enabled": run_info["llm_enabled"],
        # Knowledge state
        "num_hard_rules": knowledge["num_hard_rules"],
        "num_soft_heuristics": knowledge["num_soft_heuristics"],
        "design_prior_len": knowledge["design_prior_len"],
        "failure_analysis_len": knowledge["failure_analysis_len"],
        "sensitivity_text_len": knowledge["sensitivity_text_len"],
        # KA buffer (last 20 observations for knowledge agent to process)
        "ka_buffer": ka_buffer[-20:] if ka_buffer else [],
        "ka_buffer_size": min(len(ka_buffer), 20),
        # ★ INNOVUS-ONLY: tuning mode and IBE state
        "tuning_mode": run_info["tuning_mode"],
        "max_iterations": run_info["max_iterations"] if is_innovus_only else None,
        "estimated_innovus_sec": run_info["estimated_innovus_sec"] if is_innovus_only else None,
        "shared_genus_run_id": run_info["shared_genus_run_id"] if is_innovus_only else None,
        "genus_shared_found": genus_shared_found,
        # IBE observations: list of actual runtimes for IterationBudgetEstimator warm-start
        "ibe_observations": ibe_observations if is_innovus_only else [],
        "ibe_observed_count": len(ibe_observations) if is_innovus_only else 0,
        # Remote-fetch statistics.
        "remote_fetched_count": remote_fetched_count,
        # Runs still active on the remote host; main.py must re-attach them to the thread pool.
        "inflight_runs": inflight_runs,
    }

    # ── Report ──
    print(f"\n{'─'*50}")
    print(f"Recovery Summary:")
    print(f"  Project:             {run_info['project_name']}")
    print(f"  Tuning mode:         {run_info['tuning_mode']}")
    print(f"  Total records:       {len(history_all)}")
    print(f"  Full valid:          {len(history_full_valid)}")
    print(f"  Quick usable:        {warmup_quick_success}")
    print(f"  Promo pool size:     {len(promo_pool)}")
    print(f"  Launched (next idx): {launched}")
    print(f"  Seen quick FPs:      {len(seen_quick)}")
    print(f"  Seen full FPs:       {len(seen_full)}")
    print(f"  Total elapsed:       {total_elapsed_sec/3600:.2f} hours")
    print(f"  Global ref point:    {global_ref_point}")
    print(f"  KA buffer:           {len(state['ka_buffer'])} items")
    print(f"  Knowledge rules:     {knowledge['num_hard_rules']} hard, "
          f"{knowledge['num_soft_heuristics']} soft")

    quick_count = sum(1 for r in history_all if r.get("fidelity") == "quick")
    full_count = sum(1 for r in history_all if r.get("fidelity") == "full")
    success_count = sum(1 for r in history_all if r.get("success", False))
    print(f"  Quick/Full:          {quick_count}/{full_count}")
    print(f"  Success/Fail:        {success_count}/{len(history_all) - success_count}")
    if remote_fetched_count > 0:
        print(f"  Remote fetched:      {remote_fetched_count} valid runs")
    if inflight_runs:
        print(f"  Remote inflight:     {len(inflight_runs)} runs still running on server")
        for _ifr in inflight_runs:
            print(f"    - {_ifr['run_id']}: phase={_ifr['checkpoint_phase']} "
                  f"genus={_ifr.get('genus_status', 'n/a')} "
                  f"innovus={_ifr.get('innovus_status', 'n/a')}")

    if is_innovus_only:
        print(f"  [INNOVUS-ONLY]")
        print(f"    max_iterations:    {run_info['max_iterations']}")
        print(f"    est_innovus_sec:   {run_info['estimated_innovus_sec']}")
        print(f"    IBE observations:  {len(ibe_observations)}")
        print(f"    genus_shared:      {'found' if genus_shared_found else 'NOT found'}")

    if history_full_valid:
        best_wns = max((_safe_float(r["metrics"].get("wns"), -1e9) for r in history_full_valid), default=float("nan"))
        best_area = min((_safe_float(r["metrics"].get("area"), 1e9) for r in history_full_valid), default=float("nan"))
        best_power = min((_safe_float(r["metrics"].get("power"), 1e9) for r in history_full_valid), default=float("nan"))
        print(f"  Best WNS:            {best_wns:.4f}")
        print(f"  Best Area:           {best_area:.1f}")
        print(f"  Best Power:          {best_power:.4f}")
    print(f"{'─'*50}\n")

    # ── Write ──
    if dry_run:
        print("[DRY RUN] Not writing resume state.")
    else:
        if resume_path.is_file() and not force:
            backup = resume_path.with_suffix(f".bak.{int(time.time())}.json")
            resume_path.rename(backup)
            print(f"  [BACKUP] existing state → {backup.name}")

        resume_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(resume_path) + ".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        tmp.replace(resume_path)
        print(f"  [SAVED] {resume_path}")

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Batch recovery
# ═══════════════════════════════════════════════════════════════════════════════

def recover_batch(
    parent_dir: str | Path,
    prefix: str = "ablation_",
    config_map: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    parent = Path(parent_dir)
    config_map = config_map or {}
    results = {}

    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue

        has_iters = any((d / f"iter_{i}").is_dir() for i in range(5))
        has_jsonl = (d / "kb" / "runs.jsonl").is_file()
        if not has_iters and not has_jsonl:
            continue

        cfg_path = config_map.get(d.name)
        # Auto-detect config
        if not cfg_path:
            candidate = Path("configs") / f"{d.name}.yaml"
            if candidate.is_file():
                cfg_path = str(candidate)

        print(f"\n{'═'*60}")
        print(f"Recovering: {d.name}")
        try:
            state = recover_state(d, config_path=cfg_path, dry_run=dry_run, force=force)
            results[d.name] = state
        except Exception as e:
            print(f"[ERROR] Failed to recover {d.name}: {e}")
            import traceback
            traceback.print_exc()
            results[d.name] = {"error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_state(work_dir: str | Path) -> None:
    work_dir = Path(work_dir)
    existing_path = work_dir / "resume_state.json"
    if not existing_path.is_file():
        print("[SKIP] No existing resume_state.json to verify against.")
        return

    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    recovered = recover_state(work_dir, dry_run=True)

    print(f"\n{'─'*50}")
    print("Verification (existing vs recovered):")
    for field in ("launched", "warmup_quick_success"):
        ev = existing.get(field, 0)
        rv = recovered.get(field, 0)
        match = "✓" if ev == rv else "✗"
        print(f"  {match} {field}: existing={ev} recovered={rv}")

    for field in ("history_all", "history_full_valid", "seen_quick", "seen_full"):
        ev = len(existing.get(field, []))
        rv = len(recovered.get(field, []))
        match = "✓" if ev == rv else "✗"
        print(f"  {match} len({field}): existing={ev} recovered={rv}")

    promo_ev = len(existing.get("promo_pool", {}))
    promo_rv = len(recovered.get("promo_pool", {}))
    match = "✓" if promo_ev == promo_rv else "~"
    print(f"  {match} promo_pool: existing={promo_ev} recovered={promo_rv}")

    # ★ INNOVUS-ONLY: verify additional fields
    for field in ("tuning_mode", "genus_shared_found", "ibe_observed_count"):
        ev = existing.get(field)
        rv = recovered.get(field)
        match = "✓" if ev == rv else "~"
        print(f"  {match} {field}: existing={ev} recovered={rv}")

    print(f"{'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Recover EDATuner main/ablation resume state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.recover_state --work-dir ./runs/ablation_full --config configs/ablation_full.yaml
  python -m tools.recover_state --work-dir ./runs/ablation_full --force
  python -m tools.recover_state --batch-dir ./runs --prefix ablation_
  python -m tools.recover_state --work-dir ./runs/ablation_full --verify --dry-run

  # Fetch the most recent N runs from the remote server before recovery.
  python -m tools.recover_state --work-dir ./runs/exp1 --config configs/exp1.yaml --fetch-remote
  python -m tools.recover_state --work-dir ./runs/exp1 --fetch-remote --cred-file ~/.ssh/creds.json
  python -m tools.recover_state --work-dir ./runs/exp1 --fetch-remote --max-concurrency 16
        """,
    )
    ap.add_argument("--work-dir", "-w", type=str, help="Work directory of a single run")
    ap.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config")
    ap.add_argument("--batch-dir", "-b", type=str, default=None, help="Parent dir for batch recovery")
    ap.add_argument("--prefix", type=str, default="ablation_", help="Dir prefix for batch mode")
    ap.add_argument("--dry-run", action="store_true", help="Report only")
    ap.add_argument("--force", "-f", action="store_true", help="Overwrite existing state")
    ap.add_argument("--verify", action="store_true", help="Compare existing vs recovered")

    # Remote-fetch options.
    ap.add_argument("--fetch-remote", action="store_true",
                    help="Fetch metadata for the most recent max_concurrency runs from the remote server.")
    ap.add_argument("--cred-file", type=str, default=None,
                    help="SSH credential file path (overrides the config).")
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="Maximum number of runs to fetch (overrides the config).")

    args = ap.parse_args()

    if args.batch_dir:
        recover_batch(args.batch_dir, prefix=args.prefix, dry_run=args.dry_run, force=args.force)
    elif args.work_dir:
        if args.verify:
            verify_state(args.work_dir)
        else:
            recover_state(
                args.work_dir,
                config_path=args.config,
                dry_run=args.dry_run,
                force=args.force,
                fetch_remote=args.fetch_remote,
                cred_file=args.cred_file,
                max_concurrency=args.max_concurrency,
            )
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
