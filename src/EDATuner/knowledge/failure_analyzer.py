# src/EDATuner/knowledge/failure_analyzer.py
"""
3.2 Failure Pattern Mining
==========================
Extract structured failure patterns from historical failed runs, learn associations
between parameter combinations and failure modes, and generate a compact summary
for injection into the KA prompt.

Instead of sending raw log tails directly to the LLM, this module first performs
pattern matching and statistics, then passes only structured conclusions forward.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# -- Failure pattern definitions ------------------------------------------------

FAILURE_PATTERNS: Dict[str, re.Pattern] = {
    # Common Innovus / OpenROAD failures
    "placement_overflow":  re.compile(r"cannot\s+place|placement.*overflow|place_design\s+failed", re.I),
    "timing_catastrophic": re.compile(r"Slack\s+Time\s+-\d{4,}", re.I),              # WNS < -1000
    "cts_failure":         re.compile(r"ccopt_design\s+failed|clock\s+tree.*fail", re.I),
    "route_congestion":    re.compile(r"overflow.*exceed|congestion.*limit|routeDesign\s+failed", re.I),
    "drc_overflow":        re.compile(r"DRC.*violation|Total\s+DRC\s+violations\s*[:=]\s*[1-9]", re.I),
    "init_design_fail":    re.compile(r"init_design\s+failed|floorPlan.*fail", re.I),
    "lef_lib_missing":     re.compile(r"cannot\s+find.*LEF|missing.*site|cannot\s+find.*lib", re.I),
    "memory_oom":          re.compile(r"out\s+of\s+memory|cannot\s+allocate|MemoryError", re.I),
    "timeout":             re.compile(r"\btimeout\b|\btimed\s+out\b|\btime\s+limit\b", re.I),
    "crash":               re.compile(r"segmentation\s+fault|core\s+dumped|SIGKILL|SIGSEGV|fatal\s+error", re.I),
    "genus_synth_fail":    re.compile(r"genus.*fail|synthesis.*fail|elaborate.*fail", re.I),
    "ssh_connection":      re.compile(r"ssh.*timeout|connection.*refused|paramiko.*error", re.I),
    "lsf_job_exit":        re.compile(r"EXIT|job.*killed|TERM_RUNLIMIT", re.I),
}


def classify_failure(record: Dict[str, Any]) -> List[str]:
    """
    Classify a single failure record.
    Return the list of all matching failure tags.
    """
    tags: List[str] = []

    # 1) Fast extraction from log_summary.symptoms
    ls = record.get("log_summary", {}) or {}
    if isinstance(ls, dict):
        symptoms = ls.get("symptoms", [])
        if isinstance(symptoms, list):
            for s in symptoms:
                s_lower = str(s).lower()
                if "timeout" in s_lower:
                    tags.append("timeout")
                if "crash" in s_lower:
                    tags.append("crash")
                if "overflow" in s_lower:
                    tags.append("route_congestion")
                if "drc" in s_lower:
                    tags.append("drc_overflow")

    # 2) Regex matching on fail_log_tail / error fields
    text_sources = [
        str(record.get("fail_log_tail", "")),
        str(record.get("error_tail", "")),
        str(record.get("error", "")),
        str((ls.get("brief", "") if isinstance(ls, dict) else "")),
    ]
    combined_text = "\n".join(text_sources)

    for pattern_name, regex in FAILURE_PATTERNS.items():
        if regex.search(combined_text):
            if pattern_name not in tags:
                tags.append(pattern_name)

    # 3) Infer from metrics
    m = record.get("metrics", {}) or {}
    try:
        wns = float(m.get("wns", -100.0))
        if wns < -1000.0 and wns != -100.0:
            # Catastrophic WNS, but not just the default sentinel value
            if "timing_catastrophic" not in tags:
                tags.append("timing_catastrophic")
    except Exception:
        pass

    # 4) Infer from _meta (Cadence remote-specific)
    meta = m.get("_meta", {}) or {}
    if isinstance(meta, dict):
        genus_status = str(meta.get("genus_status", "")).upper()
        innovus_status = str(meta.get("innovus_status", "")).upper()
        if genus_status == "EXIT":
            if "genus_synth_fail" not in tags:
                tags.append("genus_synth_fail")
        if innovus_status == "EXIT":
            if "lsf_job_exit" not in tags:
                tags.append("lsf_job_exit")
        meta_error = str(meta.get("error", ""))
        if meta_error:
            for pn, rx in FAILURE_PATTERNS.items():
                if rx.search(meta_error) and pn not in tags:
                    tags.append(pn)

    if not tags:
        tags.append("unknown")

    return tags


# -- Failure-pattern / parameter correlation analysis ---------------------------

def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        return float(val)
    except Exception:
        return default


# Infrastructure failure tags: exclude these from parameter/failure correlation
_INFRA_FAILURE_TAGS = frozenset({
    "ssh_connection", "lsf_job_exit", "memory_oom",
})

def _is_infra_failure(record: Dict[str, Any]) -> bool:
    """
    Determine whether a failure record is an infrastructure/network issue.
    These failures are unrelated to parameter choice and should not affect
    parameter/failure correlation analysis.
    """
    tags = classify_failure(record)
    # If every tag is infrastructure-only, treat it as a pure infrastructure failure
    real_tags = [t for t in tags if t not in _INFRA_FAILURE_TAGS and t != "unknown"]
    return len(real_tags) == 0 and len(tags) > 0


def analyze_failure_patterns(
    history: List[Dict[str, Any]],
    param_space: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze failed runs and return structured failure-pattern statistics.
    NOTE: Pure infrastructure/network errors (ssh_connection, lsf_job_exit, memory_oom)
    are excluded from parameter/failure correlation analysis, but they are still counted
    in ``pattern_counts``.

    Returns:
    {
        "total_runs": int,
        "total_failures": int,
        "total_infra_failures": int,
        "failure_rate": float,
        "pattern_counts": {"timeout": 5, "placement_overflow": 3, ...},
        "pattern_param_stats": { ... },
        "param_failure_correlation": { ... },
        "top_insights": [...],
    }
    """
    param_names = [str(p.get("name", "")) for p in param_space if p.get("name")]

    all_fails = [r for r in history if isinstance(r, dict) and not r.get("success", False)]
    successes = [r for r in history if isinstance(r, dict) and r.get("success", False)]

    # Separate infrastructure failures
    infra_fails = [r for r in all_fails if _is_infra_failure(r)]
    fails = [r for r in all_fails if not _is_infra_failure(r)]  # Only parameter-related failures

    result: Dict[str, Any] = {
        "total_runs": len(history),
        "total_failures": len(all_fails),
        "total_infra_failures": len(infra_fails),       # ★ NEW
        "param_related_failures": len(fails),            # ★ NEW
        "failure_rate": len(fails) / max(1, len(history)),  # Only parameter-related failures count here
        "pattern_counts": {},
        "pattern_param_stats": {},
        "param_failure_correlation": {},
        "top_insights": [],
    }

    if not fails and not infra_fails:
        return result

    # Even if no parameter-related failures remain, still count all patterns (including infra)
    # Use only parameter-related failures for correlation statistics

    # Step 1: classify each failed record
    # pattern_counts includes all failures (including infra) for monitoring
    pattern_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pattern_counter: Counter = Counter()

    for r in all_fails:
        tags = classify_failure(r)
        for tag in tags:
            pattern_counter[tag] += 1
            pattern_records[tag].append(r)

    result["pattern_counts"] = dict(pattern_counter.most_common())

    # Step 2: parameter statistics for each failure mode
    pattern_param_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pattern, records in pattern_records.items():
        if pattern == "unknown":
            continue
        param_stats: Dict[str, Dict[str, Any]] = {}

        for pname in param_names:
            values = []
            for r in records:
                p = (r.get("params", {}) or {})
                if pname in p:
                    try:
                        values.append(float(p[pname]))
                    except (ValueError, TypeError):
                        pass

            if values:
                param_stats[pname] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        if param_stats:
            pattern_param_stats[pattern] = param_stats

    result["pattern_param_stats"] = pattern_param_stats

    # Step 3: parameter -> failure-rate correlation (simple threshold analysis)
    param_correlations: Dict[str, Dict[str, Any]] = {}

    for pname in param_names:
        # Collect this parameter from both success and failure records
        fail_values = []
        succ_values = []

        for r in fails:
            p = r.get("params", {}) or {}
            if pname in p:
                try:
                    fail_values.append(float(p[pname]))
                except (ValueError, TypeError):
                    pass

        for r in successes:
            p = r.get("params", {}) or {}
            if pname in p:
                try:
                    succ_values.append(float(p[pname]))
                except (ValueError, TypeError):
                    pass

        if len(fail_values) < 2 or len(succ_values) < 2:
            continue

        fail_mean = sum(fail_values) / len(fail_values)
        succ_mean = sum(succ_values) / len(succ_values)

        # Use the mean-based threshold to compare failure rates above/below the split
        all_vals = fail_values + succ_values
        threshold = (fail_mean + succ_mean) / 2.0

        high_fail = sum(1 for v in fail_values if v >= threshold)
        high_succ = sum(1 for v in succ_values if v >= threshold)
        low_fail = sum(1 for v in fail_values if v < threshold)
        low_succ = sum(1 for v in succ_values if v < threshold)

        high_total = high_fail + high_succ
        low_total = low_fail + low_succ

        if high_total > 0 and low_total > 0:
            rate_high = high_fail / high_total
            rate_low = low_fail / low_total

            # Record only parameters with a clear gap (>20%) in failure rate
            if abs(rate_high - rate_low) > 0.20:
                direction = "higher=worse" if rate_high > rate_low else "lower=worse"
                param_correlations[pname] = {
                    "fail_mean": round(fail_mean, 4),
                    "succ_mean": round(succ_mean, 4),
                    "threshold": round(threshold, 4),
                    "failure_rate_above": round(rate_high, 3),
                    "failure_rate_below": round(rate_low, 3),
                    "direction": direction,
                }

    result["param_failure_correlation"] = param_correlations

    # Step 4: generate top insights
    insights: List[str] = []

    # 4a: frequent failure patterns
    for pattern, count in pattern_counter.most_common(5):
        if pattern == "unknown":
            continue
        rate = count / max(1, len(fails))
        if rate > 0.15:
            insights.append(
                f"{pattern}: {count}/{len(fails)} failures ({rate:.0%})"
            )

    # 4b: parameter/failure correlation
    for pname, corr in param_correlations.items():
        direction = corr["direction"]
        if direction == "higher=worse":
            insights.append(
                f"{pname} > {corr['threshold']:.3f} → {corr['failure_rate_above']:.0%} fail rate "
                f"(vs {corr['failure_rate_below']:.0%} below)"
            )
        else:
            insights.append(
                f"{pname} < {corr['threshold']:.3f} → {corr['failure_rate_below']:.0%} fail rate "
                f"(vs {corr['failure_rate_above']:.0%} above)"
            )

    # 4c: parameter trends within each specific pattern
    for pattern, pstats in pattern_param_stats.items():
        if pattern_counter[pattern] < 3:
            continue
        for pname, stats in pstats.items():
            if pname in param_correlations:
                continue  # Already reported in 4b
            # Check whether the parameter distribution is clearly biased to one side
            all_succ_vals = []
            for r in successes:
                p = r.get("params", {}) or {}
                if pname in p:
                    try:
                        all_succ_vals.append(float(p[pname]))
                    except (ValueError, TypeError):
                        pass
            if all_succ_vals:
                succ_mean = sum(all_succ_vals) / len(all_succ_vals)
                diff = stats["mean"] - succ_mean
                if abs(diff) > 0.01 * max(abs(succ_mean), 1.0):
                    dir_word = "higher" if diff > 0 else "lower"
                    insights.append(
                        f"{pattern} runs have {dir_word} {pname} "
                        f"(fail_avg={stats['mean']:.3f} vs succ_avg={succ_mean:.3f})"
                    )

    result["top_insights"] = insights[:10]

    return result


def format_failure_analysis_for_prompt(
    analysis: Dict[str, Any],
    *,
    max_chars: int = 1200,
) -> str:
    """
    Format ``analyze_failure_patterns()`` output as prompt-ready text.
    """
    total = analysis.get("total_runs", 0)
    fails = analysis.get("total_failures", 0)
    rate = analysis.get("failure_rate", 0)

    if fails == 0:
        return ""

    lines: List[str] = []
    lines.append(f"Failure rate: {fails}/{total} ({rate:.0%})")

    # Pattern distribution
    pc = analysis.get("pattern_counts", {})
    if pc:
        top_patterns = list(pc.items())[:5]
        lines.append("Failure patterns: " + ", ".join(
            f"{p}={c}" for p, c in top_patterns
        ))

    # Parameter correlation
    corr = analysis.get("param_failure_correlation", {})
    if corr:
        lines.append("Parameter-failure correlations:")
        for pname, c in list(corr.items())[:5]:
            lines.append(
                f"  {pname}: {c['direction']} "
                f"(threshold={c['threshold']:.3f}, "
                f"above={c['failure_rate_above']:.0%}, below={c['failure_rate_below']:.0%})"
            )

    # Top insights
    insights = analysis.get("top_insights", [])
    if insights:
        lines.append("Key findings:")
        for ins in insights[:6]:
            lines.append(f"  - {ins}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text
