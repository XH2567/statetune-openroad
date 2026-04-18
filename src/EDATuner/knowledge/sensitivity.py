# src/EDATuner/knowledge/sensitivity.py
"""
3.3 Parameter Sensitivity Analysis
====================================
Compute the sensitivity of each parameter to WNS / Area / Power from historical runs,
and generate a compact statistical summary for injection into the KA user prompt.

Core methods:
  - Pearson correlation coefficient (continuous parameters)
  - Mean differences across groups (discrete parameters / choices)
  - Single-parameter scan detection (delta analysis when only one parameter changes within a batch)
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# -- Utility functions --------------------------------------------------------

def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Compute the Pearson correlation coefficient. Return 0 when samples are insufficient or variance is zero."""
    n = len(xs)
    if n < 4:
        return 0.0

    mx = sum(xs) / n
    my = sum(ys) / n

    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))

    denom = math.sqrt(sxx * syy)
    if denom < 1e-15:
        return 0.0
    return sxy / denom


def _rank_transform(vals: List[float]) -> List[float]:
    """Rank transform for Spearman correlation, using average rank for ties."""
    n = len(vals)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n and vals[indexed[j]] == vals[indexed[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation coefficient. More robust for non-linear relationships."""
    if len(xs) < 4:
        return 0.0
    rx = _rank_transform(xs)
    ry = _rank_transform(ys)
    return _pearson(rx, ry)


# -- Main analysis functions --------------------------------------------------

def compute_parameter_sensitivity(
    history: List[Dict[str, Any]],
    param_space: List[Dict[str, Any]],
    *,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """
    Compute the sensitivity of each parameter to each objective.

    Returns:
    {
        "n_valid": int,
        "sensitivities": {
            "UTIL": {
                "wns_corr": -0.82,
                "area_corr": 0.91,
                "power_corr": 0.45,
                "rank": 1,             # Overall importance rank
                "impact_score": 0.73,  # Combined impact score (0-1)
                "direction_hint": "higher UTIL -> worse WNS, higher Area",
                "type": "float",
                "range_used": [0.55, 0.75],
            },
            ...
        },
        "ranking": ["UTIL", "TD_PLACE", "MARGIN", "RUN_CTS"],  # Sorted by impact
    }
    """
    param_names = []
    param_types = {}
    for p in param_space:
        name = str(p.get("name", "")).strip()
        if name:
            param_names.append(name)
            param_types[name] = str(p.get("type", "float")).lower()

    # Keep only valid successful records
    valid_records: List[Dict[str, Any]] = []
    for r in history:
        if not isinstance(r, dict):
            continue
        if not r.get("success", False):
            continue
        m = r.get("metrics", {}) or {}
        area = _safe_float(m.get("area"))
        wns = _safe_float(m.get("wns"))
        power = _safe_float(m.get("power"))

        # Require at least area and wns
        if not math.isfinite(area) or not math.isfinite(wns):
            continue
        # WNS cannot be the default placeholder value (-100.0)
        if wns == -100.0:
            continue

        valid_records.append(r)

    result: Dict[str, Any] = {
        "n_valid": len(valid_records),
        "sensitivities": {},
        "ranking": [],
    }

    if len(valid_records) < min_samples:
        return result

    # Extract objective vectors
    areas = [_safe_float(r["metrics"]["area"]) for r in valid_records]
    wnss = [_safe_float(r["metrics"]["wns"]) for r in valid_records]
    powers = [_safe_float(r["metrics"].get("power", float("inf"))) for r in valid_records]

    has_power = sum(1 for p in powers if math.isfinite(p)) >= min_samples

    sensitivities: Dict[str, Dict[str, Any]] = {}

    for pname in param_names:
        # Extract values for this parameter
        param_vals: List[float] = []
        area_vals: List[float] = []
        wns_vals: List[float] = []
        power_vals: List[float] = []

        for i, r in enumerate(valid_records):
            p = r.get("params", {}) or {}
            if pname not in p:
                continue
            try:
                pv = float(p[pname])
            except (ValueError, TypeError):
                continue

            param_vals.append(pv)
            area_vals.append(areas[i])
            wns_vals.append(wnss[i])
            if has_power and math.isfinite(powers[i]):
                power_vals.append(powers[i])

        if len(param_vals) < min_samples:
            continue

        # Compute correlations (Spearman is more robust here)
        wns_corr = _spearman(param_vals, wns_vals)
        area_corr = _spearman(param_vals, area_vals)
        power_corr = _spearman(param_vals, power_vals) if len(power_vals) >= min_samples else 0.0

        # Combined impact score: weighted average of absolute correlations
        # WNS gets higher weight because it is often the bottleneck
        impact_score = (
            0.45 * abs(wns_corr)
            + 0.30 * abs(area_corr)
            + 0.25 * abs(power_corr)
        )

        # Generate a direction hint
        hints: List[str] = []
        if abs(wns_corr) > 0.15:
            hints.append(f"{'better' if wns_corr > 0 else 'worse'} WNS")
        if abs(area_corr) > 0.15:
            hints.append(f"{'higher' if area_corr > 0 else 'lower'} Area")
        if abs(power_corr) > 0.15:
            hints.append(f"{'higher' if power_corr > 0 else 'lower'} Power")

        direction_hint = ""
        if hints:
            direction_hint = f"higher {pname} -> " + ", ".join(hints)

        sensitivities[pname] = {
            "wns_corr": round(wns_corr, 3),
            "area_corr": round(area_corr, 3),
            "power_corr": round(power_corr, 3),
            "impact_score": round(impact_score, 3),
            "direction_hint": direction_hint,
            "type": param_types.get(pname, "float"),
            "n_samples": len(param_vals),
            "range_used": [round(min(param_vals), 4), round(max(param_vals), 4)],
        }

    # Rank parameters
    ranked = sorted(sensitivities.items(), key=lambda kv: -kv[1]["impact_score"])
    for rank_idx, (pname, s) in enumerate(ranked):
        s["rank"] = rank_idx + 1

    result["sensitivities"] = sensitivities
    result["ranking"] = [name for name, _ in ranked]

    return result


def format_sensitivity_for_prompt(
    analysis: Dict[str, Any],
    *,
    max_chars: int = 800,
) -> str:
    """
    Format sensitivity analysis results as text for injection into the user prompt.
    """
    n = analysis.get("n_valid", 0)
    if n < 5:
        return ""

    sensitivities = analysis.get("sensitivities", {})
    ranking = analysis.get("ranking", [])

    if not ranking:
        return ""

    lines: List[str] = []
    lines.append(f"PARAMETER SENSITIVITY (from {n} successful runs):")

    for pname in ranking:
        s = sensitivities[pname]
        wc = s["wns_corr"]
        ac = s["area_corr"]
        pc = s["power_corr"]
        imp = s["impact_score"]
        hint = s.get("direction_hint", "")

        # Compact format: one parameter per line
        line = f"  #{s['rank']} {pname}: WNS_corr={wc:+.2f} Area_corr={ac:+.2f} Power_corr={pc:+.2f} impact={imp:.2f}"
        if hint:
            line += f"  ({hint})"
        lines.append(line)

    # Add a short summary
    if ranking:
        top = ranking[0]
        top_s = sensitivities[top]
        lines.append(f"Most impactful parameter: {top} (impact={top_s['impact_score']:.2f})")
        if len(ranking) >= 2:
            bot = ranking[-1]
            bot_s = sensitivities[bot]
            lines.append(f"Least impactful parameter: {bot} (impact={bot_s['impact_score']:.2f})")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text

def detect_threshold_effects(
    history: List[Dict[str, Any]],
    param_space: List[Dict[str, Any]],
    *,
    metric_keys: Tuple[str, ...] = ("wns", "area", "power"),
    n_bins: int = 3,
    min_samples_per_bin: int = 3,
) -> Dict[str, Any]:
    """
    Detect threshold and other non-linear parameter effects.
    Split each continuous parameter range into ``n_bins`` segments and compare
    the mean metric value in each segment. If the difference between adjacent
    segments is much larger than the within-segment variance, flag it as a
    threshold effect.

    Returns:
        {
            "threshold_effects": {
                "CORE_UTIL": {
                    "metric": "wns",
                    "threshold": 0.70,
                    "below_mean": -0.5,
                    "above_mean": -3.2,
                    "effect_size": 2.7,
                    "description": "WNS degrades sharply when CORE_UTIL > 0.70"
                }, ...
            }
        }
    """
    import numpy as np

    valid = []
    for h in history:
        if not h.get("success", False):
            continue
        m = h.get("metrics", {}) or {}
        try:
            vals = {k: float(m[k]) for k in metric_keys}
            if all(math.isfinite(v) for v in vals.values()):
                valid.append(h)
        except (KeyError, TypeError, ValueError):
            continue

    if len(valid) < n_bins * min_samples_per_bin:
        return {"threshold_effects": {}}

    param_info = {p["name"]: p for p in param_space if "range" in p}
    effects = {}

    for pname, pdef in param_info.items():
        lo, hi = float(pdef["range"][0]), float(pdef["range"][1])
        if hi <= lo:
            continue

        # Collect (param_val, metric_dict) pairs
        data = []
        for h in valid:
            p = h.get("params", {}) or {}
            if pname not in p:
                continue
            try:
                pval = float(p[pname])
            except (ValueError, TypeError):
                continue
            m = h.get("metrics", {})
            data.append((pval, {k: float(m[k]) for k in metric_keys}))

        if len(data) < n_bins * min_samples_per_bin:
            continue

        # Partition into bins
        sorted_data = sorted(data, key=lambda x: x[0])
        bin_size = len(sorted_data) // n_bins
        bins = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(sorted_data)
            bins.append(sorted_data[start:end])

        # Find the largest jump between adjacent bins
        for mk in metric_keys:
            bin_means = []
            for b in bins:
                vals = [d[1][mk] for d in b]
                bin_means.append(np.mean(vals))

            for i in range(len(bin_means) - 1):
                jump = abs(bin_means[i+1] - bin_means[i])
                # Within-bin standard deviation
                within_std = np.mean([
                    np.std([d[1][mk] for d in bins[j]]) 
                    for j in [i, i+1]
                ])
                
                if within_std > 0 and jump > 2.0 * within_std:
                    threshold = bins[i][-1][0]  # Boundary point
                    direction = "increases" if bin_means[i+1] > bin_means[i] else "decreases"
                    
                    # For WNS, increase is good; for area/power, decrease is good
                    if mk == "wns":
                        quality = "improves" if direction == "increases" else "degrades"
                    else:
                        quality = "improves" if direction == "decreases" else "degrades"
                    
                    effect_key = f"{pname}__{mk}"
                    if effect_key not in effects or jump > effects[effect_key]["effect_size"]:
                        effects[effect_key] = {
                            "param": pname,
                            "metric": mk,
                            "threshold": round(threshold, 4),
                            "below_mean": round(float(bin_means[i]), 4),
                            "above_mean": round(float(bin_means[i+1]), 4),
                            "effect_size": round(float(jump), 4),
                            "description": (
                                f"{mk} {quality} sharply when {pname} > {threshold:.3f} "
                                f"(below: {bin_means[i]:.3f}, above: {bin_means[i+1]:.3f})"
                            ),
                        }

    return {"threshold_effects": effects}


def format_threshold_effects_for_prompt(
    analysis: Dict[str, Any],
    *,
    max_effects: int = 5,
) -> str:
    """Format threshold effects as prompt text."""
    effects = analysis.get("threshold_effects", {})
    if not effects:
        return ""

    # Sort by effect_size descending
    sorted_effects = sorted(effects.values(), key=lambda x: -x["effect_size"])[:max_effects]
    
    lines = ["NON-LINEAR THRESHOLD EFFECTS (critical for avoiding bad regions):"]
    for e in sorted_effects:
        lines.append(f"  - {e['description']}")
    
    return "\n".join(lines) + "\n"
