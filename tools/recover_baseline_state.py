#!/usr/bin/env python3
"""
tools/recover_baseline_state.py
================================
Rebuild baseline resume state from on-disk artifacts when resume_state.baseline.json
is missing, corrupted, or out-of-date.

Recovery sources (in priority order):
  1. kb/runs.jsonl            — richest per-run data (metrics, fidelity, params, wall times)
  2. iter_*/checkpoint.json   — fingerprint + fidelity for dedup sets
  3. iter_*/metrics.json      — fallback metrics if runs.jsonl is incomplete
  4. experiment_summary.json  — global ref point, HV, metadata

The script rebuilds:
  - history_all / history_full_valid  (sorted by run_id numeric index)
  - seen_quick / seen_full            (fingerprint dedup sets)
  - promo_pool                        (quick-usable params eligible for full promotion)
  - warmup_quick_success counter
  - global_ref_point
  - launched counter
  - total_elapsed_sec

Usage:
    # Auto-detect strategy from config or existing state
    python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo

    # Explicit strategy + config
    python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo --strategy bo

    # With config for param_space / default_params
    python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo --config configs/baseline_bo.yaml

    # Dry-run (don't write, just report)
    python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo --dry-run

    # Force overwrite existing resume state
    python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo --force
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

# ---------------------------------------------------------------------------
# Add project root to path so we can import EDATuner utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.EDATuner.utils.utils import load_config
except ImportError:
    load_config = None


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers (self-contained, no external deps beyond stdlib)
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(params: Dict[str, Any]) -> str:
    """Deterministic JSON fingerprint of a param dict."""
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _infer_shared_genus_dir_from_workdir(work_dir: Path, run_id: str) -> str:
    run_id = str(run_id or "").strip()
    if not run_id:
        return ""
    candidate = work_dir / run_id / "remote_artifacts" / "outbox" / "cadence" / "genus"
    if candidate.is_dir():
        return str(candidate)
    return ""


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _run_id_index(run_id: str) -> int:
    """Extract numeric index from 'iter_42' style run_id."""
    try:
        if run_id.startswith("iter_"):
            return int(run_id.split("_", 1)[1])
    except Exception:
        pass
    return -1


def _sort_by_run_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: _run_id_index(str(r.get("run_id", ""))))


def _classify_metrics(metrics: Dict[str, Any], fidelity: str) -> Dict[str, Any]:
    """Classify metrics quality (mirrors run_baseline.py logic)."""
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
    """Compute global HV reference point from full-valid history."""
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
# RankTuner: Rebuild proposer state (Trust Region) from history
# ═══════════════════════════════════════════════════════════════════════════════

def _rebuild_ranktuner_proposer_state(
    history_all: List[Dict[str, Any]],
    param_space: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Rebuild RankTuner proposer state (Trust Region) from history.

    This re-derives the TR center/length/prev_best_f by:
      1. Constructing pairwise Pareto-dominance comparisons from history
      2. Fitting a PairwiseGP on the comparisons
      3. Extracting TR state from the best latent value

    If insufficient data or any failure, returns None (safe — run_baseline
    will fall back to _restore_ranktuner_proposer on resume).
    """
    if not param_space:
        return None

    # Filter valid records with metrics
    valid_records = []
    for h in history_all:
        m = h.get("metrics", {}) or {}
        try:
            area = float(m.get("area", float("inf")))
            wns = float(m.get("wns", -100.0))
            power = float(m.get("power", float("inf")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(area) and math.isfinite(wns) and wns > -99.0 and math.isfinite(power)):
            continue
        params = h.get("params", {})
        if not params:
            continue
        valid_records.append(h)

    if len(valid_records) < 6:
        print(f"  [RankTuner] insufficient valid records ({len(valid_records)}) for TR rebuild")
        return None

    # Try to import RankTuner components
    try:
        from baselines.ranktuner_proposer import (
            RankTunerProposer,
            generate_comparisons_from_history,
            PairwiseGP,
        )
        import torch
    except ImportError as e:
        print(f"  [RankTuner] cannot import proposer components: {e}")
        return None

    # Build comparisons
    valid_for_comp, comparisons = generate_comparisons_from_history(valid_records)
    if len(comparisons) < 3:
        print(f"  [RankTuner] insufficient comparisons ({len(comparisons)}) for TR rebuild")
        return None

    # Create a temporary proposer to use its encoding methods
    # Create a temporary proposer to use its encoding methods
    try:
        proposer = RankTunerProposer(param_space)

        # First construct the historical design matrix in the raw parameter space.
        X_full = torch.cat(
            [proposer._to_tensor(v["params"]) for v in valid_for_comp],
            dim=0,
        ).to(torch.double)

        # Then map it into the embedding space.
        X_embed = proposer._embed(X_full)
        if X_embed is None or X_embed.numel() == 0:
            print("  [RankTuner] empty embedded design matrix")
            return None
        X_embed = X_embed.to(torch.double)

        # PairwiseGP.fit expects comparisons as List[Tuple[int, int]].
        comparisons_list = [(int(i), int(j)) for i, j in comparisons]

        gp = PairwiseGP(
            dim=int(proposer.embed_dim or X_embed.shape[-1]),
            noise_sigma=proposer.noise_sigma,
            lengthscale=proposer.lengthscale,
            outputscale=proposer.outputscale,
        )
        gp.fit(X_embed, comparisons_list, max_iter=50)

        # If a latent f_map is available, restore the trust region in embedding space.
        if gp.f_map is not None and gp.f_map.numel() > 0:
            proposer._update_trust_region(X_embed, gp.f_map)

        # Store the GP on the proposer when supported; ignore it otherwise.
        try:
            proposer.gp = gp
        except Exception:
            pass

        state = proposer.get_state()
        print(f"  [RankTuner] rebuilt TR state: length={state.get('tr_length', '?'):.4f}")
        return state

    except Exception as e:
        print(f"  [RankTuner] TR rebuild failed: {e}")



    except Exception as e:
        print(f"  [RankTuner] TR rebuild failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Recovery: load from runs.jsonl
# ═══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all valid JSON objects from a .jsonl file."""
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


def _load_runs_jsonl(work_dir: Path) -> List[Dict[str, Any]]:
    """Try multiple known locations for runs.jsonl."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Recovery: scan iter_* directories
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_iter_dirs(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Scan iter_* directories for checkpoint.json + metrics.json.
    Returns {run_id: {checkpoint_data + metrics_data}}.
    """
    found = {}
    if not work_dir.is_dir():
        return found

    for d in sorted(work_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("iter_"):
            continue
        run_id = d.name
        entry: Dict[str, Any] = {"run_id": run_id, "_dir": str(d)}

        # checkpoint.json
        ckpt = d / "checkpoint.json"
        if ckpt.is_file():
            try:
                ckpt_data = json.loads(ckpt.read_text(encoding="utf-8"))
                entry.update(ckpt_data)
            except Exception as e:
                print(f"  [WARN] failed to parse {ckpt}: {e}")

        # metrics.json (fallback)
        metrics_file = d / "metrics.json"
        if metrics_file.is_file():
            try:
                mdata = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(mdata, dict) and "metrics" not in entry:
                    entry["metrics"] = mdata
            except Exception:
                pass

        # Also check for result.json (some runners write this)
        result_file = d / "result.json"
        if result_file.is_file():
            try:
                rdata = json.loads(result_file.read_text(encoding="utf-8"))
                if isinstance(rdata, dict):
                    for k in ("metrics", "success", "runtime_sec", "fidelity",
                              "params", "stage", "log_path"):
                        if k in rdata and k not in entry:
                            entry[k] = rdata[k]
            except Exception:
                pass

        found[run_id] = entry

    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Recovery: load experiment_summary.json
# ═══════════════════════════════════════════════════════════════════════════════

def _load_experiment_summary(work_dir: Path) -> Optional[Dict[str, Any]]:
    p = work_dir / "experiment_summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cfg(config_path: Optional[str]) -> Dict[str, Any]:
    """Load config YAML, falling back to a minimal dict."""
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


def _fill_params(p: Dict[str, Any], default_params: Dict[str, Any],
                 param_space: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill missing parameter values from defaults and param_space midpoints."""
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
# Detect strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_strategy(work_dir: Path, cfg: Dict[str, Any], explicit: Optional[str]) -> str:
    """Detect baseline strategy from multiple sources."""
    if explicit:
        return explicit.strip().lower()

    # From config
    s = cfg.get("baseline_strategy", "")
    if s:
        return str(s).strip().lower()

    # From existing resume state
    for fname in ("resume_state.baseline.json", "resume_state.json"):
        p = work_dir / fname
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                s = data.get("baseline_strategy", "")
                if s:
                    return str(s).strip().lower()
            except Exception:
                pass

    # From experiment_summary.json
    summary = _load_experiment_summary(work_dir)
    if summary:
        s = summary.get("baseline_strategy", "")
        if s:
            return str(s).strip().lower()

    # From runs.jsonl (check proposed_strategy field)
    records = _load_runs_jsonl(work_dir)
    for r in records[:5]:
        ps = str(r.get("proposed_strategy", "")).lower()
        if "optuna" in ps or "tpe" in ps:
            return "optuna"
        if "bo_ehvi" in ps or "bayesian" in ps:
            return "bo"
        if "crop" in ps:
            return "crop"
        if "random" in ps:
            return "random"
        if "ranktuner" in ps or "rank_tuner" in ps:
            return "ranktuner"

    # From work_dir name
    dirname = work_dir.name.lower()
    if "ranktuner" in dirname or "rank_tuner" in dirname:
        return "ranktuner"
    if "optuna" in dirname:
        return "optuna"
    if "crop" in dirname:
        return "crop"
    if "bo" in dirname:
        return "bo"
    if "random" in dirname:
        return "random"

    print("  [WARN] could not detect strategy, defaulting to 'random'")
    return "random"


# ═══════════════════════════════════════════════════════════════════════════════
# Main recovery logic
# ═══════════════════════════════════════════════════════════════════════════════

def recover_baseline_state(
    work_dir: str | Path,
    config_path: Optional[str] = None,
    strategy: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Recover baseline resume state from on-disk artifacts.

    Returns the rebuilt state dict.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        raise FileNotFoundError(f"work_dir does not exist: {work_dir}")

    resume_path = work_dir / "resume_state.baseline.json"
    if resume_path.is_file() and not force:
        print(f"[WARN] {resume_path} already exists. Use --force to overwrite.")
        print(f"       Loading existing state for verification...")

    # ── Load config ──
    cfg = _load_cfg(config_path)
    param_space = _get_param_space(cfg)
    default_params = _get_default_params(cfg)
    fid_cfg = cfg.get("fidelity", {}) or {}
    tuning_cfg = cfg.get("tuning", {}) or {}
    warmup_quick_target = int(fid_cfg.get("warmup_quick", 20))
    max_promo_pool = int(fid_cfg.get("max_promo_pool", 2000))
    tuning_mode = str(tuning_cfg.get("mode", "full")).strip().lower()
    max_iterations = int(tuning_cfg.get("max_iterations", 200))
    estimated_innovus_sec = float(tuning_cfg.get("estimated_innovus_sec", 1800))
    shared_genus_run_id = str(tuning_cfg.get("shared_genus_run_id", "") or "").strip()
    is_innovus_only = (tuning_mode == "innovus_only")

    STRATEGY = _detect_strategy(work_dir, cfg, strategy)
    print(f"\n{'='*60}")
    print(f"Recovering baseline state: {work_dir}")
    print(f"Strategy: {STRATEGY}")
    print(f"Config: {config_path or '(none)'}")
    print(f"{'='*60}\n")

    # ── Phase 1: Load runs.jsonl (primary source) ──
    print("[Phase 1] Loading runs.jsonl ...")
    jsonl_records = _load_runs_jsonl(work_dir)

    # ── Phase 2: Scan iter_* directories (secondary/fallback) ──
    print("[Phase 2] Scanning iter_* directories ...")
    dir_entries = _scan_iter_dirs(work_dir)
    print(f"  [OK] found {len(dir_entries)} iter_* directories")

    # ── Phase 3: Merge sources ──
    print("[Phase 3] Merging data sources ...")

    # Index jsonl records by run_id
    jsonl_by_id: Dict[str, Dict[str, Any]] = {}
    for r in jsonl_records:
        rid = str(r.get("run_id", ""))
        if rid:
            jsonl_by_id[rid] = r

    # Merge: prefer jsonl data, fill gaps from directory scans
    all_run_ids = set(jsonl_by_id.keys()) | set(dir_entries.keys())
    merged_records: List[Dict[str, Any]] = []

    for rid in sorted(all_run_ids, key=_run_id_index):
        jrec = jsonl_by_id.get(rid, {})
        drec = dir_entries.get(rid, {})

        if jrec:
            # jsonl is authoritative; only fill missing fields from dir scan
            record = dict(jrec)
            for k, v in drec.items():
                if k.startswith("_"):
                    continue
                if k not in record or record[k] is None:
                    record[k] = v
        elif drec:
            # Only have dir scan data — reconstruct what we can
            record = dict(drec)
            # Ensure basic fields
            record.setdefault("run_id", rid)
            record.setdefault("fidelity", drec.get("fidelity", "quick"))
            record.setdefault("success", True)  # assume success if checkpoint exists

            # Classify metrics if we have them
            metrics = record.get("metrics", {})
            if metrics:
                cls = _classify_metrics(metrics, record.get("fidelity", "quick"))
                record.update(cls)
        else:
            continue

        merged_records.append(record)

    merged_records = _sort_by_run_id(merged_records)
    print(f"  [OK] merged {len(merged_records)} total records")

    # ── Phase 4: Rebuild state ──
    print("[Phase 4] Rebuilding state ...")

    history_all: List[Dict[str, Any]] = []
    history_full_valid: List[Dict[str, Any]] = []
    seen_quick: Set[str] = set()
    seen_full: Set[str] = set()
    promo_pool: Dict[str, Dict[str, Any]] = {}
    warmup_quick_success: int = 0
    max_launched_idx: int = -1

    for record in merged_records:
        rid = str(record.get("run_id", ""))
        fidelity = str(record.get("fidelity", "quick")).strip().lower()
        params = record.get("params", {}) or {}

        # Fill params from defaults if config is available
        if param_space:
            filled = _fill_params(params, default_params, param_space)
        else:
            filled = dict(params) if params else {}

        fp = _fingerprint(filled)

        # Ensure metrics classification
        metrics = record.get("metrics", {}) or {}
        if "metrics_usable" not in record and metrics:
            cls = _classify_metrics(metrics, fidelity)
            record.update(cls)

        # Track max launched index
        idx = _run_id_index(rid)
        if idx >= 0:
            max_launched_idx = max(max_launched_idx, idx)

        # Update seen sets
        if fidelity == "full":
            seen_full.add(fp)
        else:
            seen_quick.add(fp)

        # Quick usable → promo pool candidate
        if fidelity == "quick" and record.get("metrics_usable", False):
            warmup_quick_success += 1
            if fp not in seen_full and len(promo_pool) < max_promo_pool:
                promo_pool[fp] = {
                    "params": filled,
                    "parent_run_id": rid,
                }

        # Full valid → history_full_valid
        if fidelity == "full" and record.get("metrics_valid", False):
            history_full_valid.append(record)

        history_all.append(record)

    # Remove from promo_pool anything already seen in full
    for fp in list(promo_pool.keys()):
        if fp in seen_full:
            del promo_pool[fp]

    launched = max_launched_idx + 1 if max_launched_idx >= 0 else len(merged_records)
    if is_innovus_only:
        launched = max(launched, len([r for r in history_all if str(r.get("run_id", "")).startswith("iter_")]))

    # ── Compute elapsed time (real wall-clock time) ──
    session_wall: Dict[str, Dict[str, float]] = {}
    orphan_starts: List[float] = []
    orphan_ends: List[float] = []
    runtime_sum = 0.0

    for h in history_all:
        ws = _safe_float(h.get("wall_start_sec"))
        we = _safe_float(h.get("wall_end_sec"))
        ts = h.get("experiment_ts")
        rt = _safe_float(h.get("runtime_sec"), 0.0)

        if math.isfinite(rt) and rt > 0:
            runtime_sum += rt

        has_ws = math.isfinite(ws) and ws >= 0.0
        has_we = math.isfinite(we) and we > 0.0

        if ts is not None and (has_ws or has_we):
            ts_key = str(ts)
            info = session_wall.setdefault(ts_key, {"start": float("inf"), "end": 0.0})

            if has_ws:
                info["start"] = min(info["start"], ws)
            if has_we:
                info["end"] = max(info["end"], we)

        elif has_ws or has_we:
            if has_ws:
                orphan_starts.append(ws)
            if has_we:
                orphan_ends.append(we)

    total_elapsed_sec = 0.0

    if session_wall:
        for info in session_wall.values():
            start = info["start"]
            end = info["end"]

            if math.isfinite(start) and end > 0.0 and end >= start:
                total_elapsed_sec += (end - start)
            elif end > 0.0:
                total_elapsed_sec += end

    # Treat records without experiment_ts as a fallback timeline.
    if orphan_ends:
        orphan_start = min(orphan_starts) if orphan_starts else 0.0
        orphan_end = max(orphan_ends)
        if orphan_end >= orphan_start:
            total_elapsed_sec += (orphan_end - orphan_start)
        else:
            total_elapsed_sec += orphan_end

    # Final fallback: if wall-clock data are completely unavailable, fall back to runtime_sum.
    if total_elapsed_sec <= 0.0:
        total_elapsed_sec = runtime_sum



    innovus_only_shared_genus_dir = ""
    innovus_only_genus_ok = False
    if is_innovus_only:
        if shared_genus_run_id:
            innovus_only_shared_genus_dir = _infer_shared_genus_dir_from_workdir(work_dir, shared_genus_run_id)
        if not innovus_only_shared_genus_dir:
            innovus_only_shared_genus_dir = _infer_shared_genus_dir_from_workdir(work_dir, "genus_shared")
        innovus_only_genus_ok = bool(innovus_only_shared_genus_dir) or any(
            str(r.get("run_id", "")) == "genus_shared" and bool(r.get("success", False))
            for r in history_all
        )

    # ── Compute global ref point ──
    if len(history_full_valid) >= 3:
        global_ref_point = _compute_global_ref_point(history_full_valid)
    else:
        # Try from experiment_summary
        summary = _load_experiment_summary(work_dir)
        if summary and isinstance(summary.get("global_ref_point"), list):
            global_ref_point = [float(x) for x in summary["global_ref_point"]]
        else:
            global_ref_point = [-5e6, -10.0, -5.0]

    # ── Build resume state ──
    state = {
        "version": 1,
        "runner": "baseline",
        "baseline_strategy": STRATEGY,
        "project_name": cfg.get("project_name", work_dir.name),
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
        "promo_pool": promo_pool,
        "global_ref_point": global_ref_point,
        "tuning_mode": tuning_mode,
        "max_iterations": max_iterations,
        "estimated_innovus_sec": estimated_innovus_sec,
        "shared_genus_run_id": shared_genus_run_id,
        "innovus_only_genus_ok": innovus_only_genus_ok,
        "innovus_only_shared_genus_dir": innovus_only_shared_genus_dir,
    }

    # ── RankTuner: rebuild proposer_state (Trust Region) ──
    if STRATEGY in ("ranktuner", "rank_tuner", "rank"):
        rt_state = _rebuild_ranktuner_proposer_state(history_all, param_space)
        if rt_state:
            state["proposer_state"] = rt_state
        else:
            state["proposer_state"] = None
    elif STRATEGY == "crop":
        # CROP proposer state: candidate buffer is transient (repopulated on first propose)
        # No heavy state to rebuild — just mark it as recovered so run_baseline knows
        state["proposer_state"] = {"_candidate_buffer": [], "_recovered": True}
        print("  [CROP] proposer_state reset (candidate buffer will refill on first propose)")
    else:
        state["proposer_state"] = None

    # ── Phase 5: Report ──
    print(f"\n{'─'*50}")
    print(f"Recovery Summary:")
    print(f"  Strategy:            {STRATEGY}")
    print(f"  Total records:       {len(history_all)}")
    print(f"  Full valid:          {len(history_full_valid)}")
    print(f"  Quick usable:        {warmup_quick_success}")
    print(f"  Promo pool size:     {len(promo_pool)}")
    print(f"  Launched (next idx): {launched}")
    print(f"  Seen quick FPs:      {len(seen_quick)}")
    print(f"  Seen full FPs:       {len(seen_full)}")
    print(f"  Total elapsed:       {total_elapsed_sec/3600:.2f} hours")
    print(f"  Global ref point:    {global_ref_point}")
    if is_innovus_only:
        print(f"  Tuning mode:         {tuning_mode}")
        print(f"  Max iterations:      {max_iterations}")
        print(f"  Est innovus sec:     {estimated_innovus_sec}")
        print(f"  Genus ready:         {innovus_only_genus_ok}")
        print(f"  Shared genus dir:    {innovus_only_shared_genus_dir or '-'}")
    print(f"{'─'*50}")

    # Fidelity breakdown
    quick_count = sum(1 for r in history_all if r.get("fidelity") == "quick")
    full_count = sum(1 for r in history_all if r.get("fidelity") == "full")
    success_count = sum(1 for r in history_all if r.get("success", False))
    fail_count = len(history_all) - success_count
    print(f"  Quick runs:          {quick_count}")
    print(f"  Full runs:           {full_count}")
    print(f"  Successes:           {success_count}")
    print(f"  Failures:            {fail_count}")

    # Best metrics
    if history_full_valid:
        best_wns = max(
            (_safe_float(r["metrics"].get("wns"), -1e9) for r in history_full_valid),
            default=float("nan")
        )
        best_area = min(
            (_safe_float(r["metrics"].get("area"), 1e9) for r in history_full_valid),
            default=float("nan")
        )
        best_power = min(
            (_safe_float(r["metrics"].get("power"), 1e9) for r in history_full_valid),
            default=float("nan")
        )
        print(f"  Best WNS:            {best_wns:.4f}")
        print(f"  Best Area:           {best_area:.1f}")
        print(f"  Best Power:          {best_power:.4f}")
    print(f"{'─'*50}\n")

    # ── Phase 6: Write ──
    if dry_run:
        print("[DRY RUN] Not writing resume state.")
    else:
        if resume_path.is_file() and not force:
            # Backup existing
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
# Batch recovery (recover all baselines in a parent directory)
# ═══════════════════════════════════════════════════════════════════════════════

def recover_all_baselines(
    parent_dir: str | Path,
    config_map: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan parent_dir for baseline_* or baseline-named subdirectories
    and recover each one.
    """
    parent = Path(parent_dir)
    config_map = config_map or {}
    results = {}

    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if not (name.startswith("baseline_") or "baseline" in name):
            continue

        # Check if it has iter_* dirs or kb/runs.jsonl
        has_iters = any(
            (d / f"iter_{i}").is_dir() for i in range(5)
        )
        has_jsonl = (d / "kb" / "runs.jsonl").is_file()
        if not has_iters and not has_jsonl:
            continue

        cfg_path = config_map.get(d.name, config_map.get(name, None))
        print(f"\n{'═'*60}")
        print(f"Recovering: {d.name}")
        try:
            state = recover_baseline_state(
                d, config_path=cfg_path,
                dry_run=dry_run, force=force,
            )
            results[d.name] = state
        except Exception as e:
            print(f"[ERROR] Failed to recover {d.name}: {e}")
            results[d.name] = {"error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Resume Consistency Check
# (merged from baseline_resume_patch.py)
# ═══════════════════════════════════════════════════════════════════════════════

def check_resume_consistency(
    resume_state: Dict[str, Any],
    work_dir: str | Path,
    *,
    fix: bool = False,
) -> Dict[str, Any]:
    """
    Validate a loaded resume_state against the actual iter_* directories on disk.

    Checks:
      1. launched count matches actual iter_* dirs
      2. history entries have matching result files
      3. seen_quick / seen_full fingerprints are consistent

    Args:
        resume_state: The loaded resume dict.
        work_dir: The baseline work directory.
        fix: If True, auto-correct mismatches and return the fixed state.

    Returns:
        Dict with keys: {"valid": bool, "issues": List[str], "fixed_state": Dict | None}
    """
    work_dir = Path(work_dir)
    issues: List[str] = []

    # 1) Check launched vs actual dirs
    launched = resume_state.get("launched", 0)
    actual_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda d: d.name,
    ) if work_dir.is_dir() else []
    actual_count = len(actual_dirs)

    if launched != actual_count:
        issues.append(
            f"launched={launched} but found {actual_count} iter_* directories"
        )

    # 2) Check history length vs dirs with results
    history = resume_state.get("history_all", [])
    dirs_with_results = 0
    for d in actual_dirs:
        if (d / "result.json").is_file() or (d / "checkpoint.json").is_file():
            dirs_with_results += 1

    if len(history) != dirs_with_results:
        issues.append(
            f"history_all has {len(history)} entries but {dirs_with_results} dirs have results"
        )

    # 3) Check fingerprint sets
    seen_q = set(resume_state.get("seen_quick", []))
    seen_f = set(resume_state.get("seen_full", []))
    disk_q, disk_f = set(), set()
    for d in actual_dirs:
        ckpt = d / "checkpoint.json"
        if not ckpt.is_file():
            continue
        try:
            info = json.loads(ckpt.read_text(encoding="utf-8"))
            fp = info.get("params_fingerprint")
            fid = str(info.get("fidelity", "quick"))
            if fp:
                if fid == "full":
                    disk_f.add(str(fp))
                else:
                    disk_q.add(str(fp))
        except Exception:
            continue

    missing_q = disk_q - seen_q
    missing_f = disk_f - seen_f
    if missing_q:
        issues.append(f"seen_quick missing {len(missing_q)} fingerprints from disk")
    if missing_f:
        issues.append(f"seen_full missing {len(missing_f)} fingerprints from disk")

    result = {
        "valid": len(issues) == 0,
        "issues": issues,
        "fixed_state": None,
    }

    if fix and issues:
        fixed = dict(resume_state)
        fixed["launched"] = max(launched, actual_count)
        fixed["seen_quick"] = list(seen_q | disk_q)
        fixed["seen_full"] = list(seen_f | disk_f)
        result["fixed_state"] = fixed
        print(f"[Resume Check] Auto-fixed {len(issues)} issues")
        for iss in issues:
            print(f"  - {iss}")

    elif issues:
        print(f"[Resume Check] Found {len(issues)} consistency issues:")
        for iss in issues:
            print(f"  - {iss}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Verification: compare recovered state with original
# ═══════════════════════════════════════════════════════════════════════════════

def verify_state(work_dir: str | Path) -> None:
    """Compare existing resume_state with what recovery would produce."""
    work_dir = Path(work_dir)
    existing_path = work_dir / "resume_state.baseline.json"
    if not existing_path.is_file():
        print("[SKIP] No existing resume_state.baseline.json to verify against.")
        return

    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    recovered = recover_baseline_state(work_dir, dry_run=True)

    print(f"\n{'─'*50}")
    print("Verification (existing vs recovered):")

    fields = [
        ("launched", int),
        ("warmup_quick_success", int),
        ("total_elapsed_sec", float),
    ]
    for field, typ in fields:
        ev = typ(existing.get(field, 0))
        rv = typ(recovered.get(field, 0))
        match = "✓" if ev == rv else "✗"
        print(f"  {match} {field}: existing={ev} recovered={rv}")

    # Compare list lengths
    for field in ("history_all", "history_full_valid", "seen_quick", "seen_full"):
        ev = len(existing.get(field, []))
        rv = len(recovered.get(field, []))
        match = "✓" if ev == rv else "✗"
        print(f"  {match} len({field}): existing={ev} recovered={rv}")

    promo_ev = len(existing.get("promo_pool", {}))
    promo_rv = len(recovered.get("promo_pool", {}))
    match = "✓" if promo_ev == promo_rv else "~"
    print(f"  {match} promo_pool: existing={promo_ev} recovered={promo_rv}")
    print(f"{'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Recover baseline resume state from on-disk artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recover single baseline
  python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo --config configs/baseline_bo.yaml

  # Recover with explicit strategy
  python -m tools.recover_baseline_state --work-dir ./runs/baseline_optuna --strategy optuna

  # Recover all baselines under runs/
  python -m tools.recover_baseline_state --batch-dir ./runs

  # Dry run (report only)
  python -m tools.recover_baseline_state --work-dir ./runs/baseline_random --dry-run

  # Verify existing state
  python -m tools.recover_baseline_state --work-dir ./runs/baseline_bo --verify
        """,
    )
    ap.add_argument("--work-dir", "-w", type=str, help="Work directory of a single baseline run")
    ap.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
    ap.add_argument("--strategy", "-s", type=str, default=None,
                    choices=["random", "bo", "optuna", "ranktuner", "crop"],
                    help="Baseline strategy (auto-detected if omitted)")
    ap.add_argument("--batch-dir", "-b", type=str, default=None,
                    help="Parent directory to recover all baseline_* subdirs")
    ap.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    ap.add_argument("--force", "-f", action="store_true", help="Overwrite existing resume state")
    ap.add_argument("--verify", action="store_true", help="Compare existing vs recovered state")

    args = ap.parse_args()

    if args.batch_dir:
        recover_all_baselines(args.batch_dir, dry_run=args.dry_run, force=args.force)
    elif args.work_dir:
        if args.verify:
            verify_state(args.work_dir)
        else:
            recover_baseline_state(
                args.work_dir,
                config_path=args.config,
                strategy=args.strategy,
                dry_run=args.dry_run,
                force=args.force,
            )
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
