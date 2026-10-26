#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight metrics generator for OpenROAD-flow-scripts forks that do not ship util/genMetrics.py.

Extracts (best-effort):
  - area (um^2)
  - wns (time units from reports; typically ps for ASAP7, ns for SKY130HD, depending on libs)
  - power (W)
  - wl  (Post-DR routed wirelength, unit usually "um" as reported by DRT)
  - ecp (Effective Clock Period = clock_period - worst_slack)  [per ORFS-agent paper]

Sources (searched in order of preference):
  1) JSON files under reports dir (if any)
  2) Text reports/logs under reports/logs/results dirs via regex heuristics
  3) Optional run log (--run_log) to capture WL/clock_period when ORFS doesn't export them elsewhere

Output:
  a JSON file with keys:
    area, wns, power, wl, clock_period, ecp
  plus metadata & source annotations.

NOTE:
  - "wl" is DRWL (post-detailed-route total wire length), as reported by OpenROAD DRT:
      "Total wire length = <N> um."
  - "ecp" is computed only if both clock_period and wns are finite:
      ecp = clock_period - wns
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


_DEBUG = bool(int(os.getenv("GENMETRICS_DEBUG", "0")))


def dprint(msg: str) -> None:
    if _DEBUG:
        print(msg, file=sys.stderr)


def _public_path(path: Any, *, root: Optional[Path] = None) -> Optional[str]:
    """
    Convert a path to a privacy-safe label for logs and JSON metadata.

    Preference order:
      1) relative path under the provided root
      2) file name only
      3) string form as a last resort
    """
    if path is None:
        return None

    try:
        p = Path(path)
    except Exception:
        return str(path)

    if root is not None:
        try:
            return str(p.resolve().relative_to(root.resolve()))
        except Exception:
            pass

    try:
        name = p.name
        if name:
            return name
    except Exception:
        pass

    return str(path)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        m = re.search(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", s, flags=re.IGNORECASE)
        if not m:
            return None
        return float(m.group(1))
    except Exception:
        return None


def flatten_json(obj: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}__{k}" if prefix else str(k)
            flatten_json(v, p, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}__{i}" if prefix else str(i)
            flatten_json(v, p, out)
    else:
        out[prefix] = obj
    return out


@dataclass
class MetricPick:
    value: Optional[float] = None
    source: Optional[str] = None
    score: int = -10**9

    def consider(self, v: Optional[float], p: Path, score: int) -> None:
        if v is None:
            return
        if math.isnan(v):
            return
        if score > self.score:
            self.value = v
            self.source = _public_path(p)
            self.score = score


def file_priority_score(p: Path) -> int:
    name = p.name.lower()
    score = 0
    if "final" in name or "finish" in name or "signoff" in name:
        score += 100
    if any(tag in name for tag in ["6_", "7_", "8_", "9_", "10_", "11_", "12_"]):
        score += 30
    if "metrics" in name or "metadata" in name or "summary" in name:
        score += 40
    if any(k in name for k in ["route", "drt", "detailedroute", "detailed_route"]):
        score += 15
    if p.suffix.lower() == ".json":
        score += 20
    # ORFS per-stage files (1_synth.json ... 6_report.json): later stage wins
    m = re.match(r"^(\d+)_", name)
    if m:
        score += 5 * int(m.group(1))
    return score


def _read_text_best_effort(p: Path, max_bytes: int = 8 * 1024 * 1024) -> str:
    try:
        b = p.read_bytes()
        if len(b) > max_bytes:
            b = b[-max_bytes:]
        return b.decode("utf-8", errors="replace")
    except Exception:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def _last_match_float(regexes: List[re.Pattern], text: str) -> Optional[float]:
    last: Optional[float] = None
    for rgx in regexes:
        for m in rgx.finditer(text):
            last = safe_float(m.group(1))
    return last


JSON_KEYS = {
    "area": [
        "design__instance__area",
        "finish__design_area",
        "finish__design__instance__area",
        "finish__design__instance__area__stdcell",
        "synth__design__instance__area__stdcell",
        "finish__design__area",
        "finish__design__area__total",
    ],
    "wns": [
        "timing__setup__ws",
        "timing__setup__wns",
        "finish__timing__setup__ws",
        "finish__timing__setup__wns",
        "finish__timing__setup__worst_slack",
    ],
    "power": [
        "power__total",
        "finish__power__total",
        "finish__power__total__internal",
        "power__total__internal",
        "finish__power__total__switching",
        "finish__power__total__leakage",
    ],
    "wl": [
        "route__wirelength",
        "route__wire_length",
        "finish__route__wirelength",
        "finish__route__wire_length",
        "finish__routing__wirelength",
        "finish__routing__wire_length",
        "finish__dr__wirelength",
        "finish__dr__wire_length",
        "finish__drt__wirelength",
        "finish__drt__wire_length",
    ],
    "clock_period": [
        "constraints__clock_period",
        "finish__constraints__clock_period",
        "sdc__clock_period",
        "finish__sdc__clock_period",
        "clock_period",
    ],
}


def extract_from_json(p: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None, None, None, None, None

    flat = flatten_json(raw)

    def pick(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in flat:
                v = safe_float(flat[k])
                if v is not None and not math.isnan(v):
                    return v
        return None

    def pick_stage(suffix: str) -> Optional[float]:
        # Newer ORFS prefixes metrics with the stage name, e.g. "cts__timing__setup__ws";
        # take the last such key (files hold one stage each, so this is that stage's value).
        val: Optional[float] = None
        for k, v in flat.items():
            if k.endswith(suffix) and k.count("__") == suffix.count("__"):
                fv = safe_float(v)
                if fv is not None and not math.isnan(fv):
                    val = fv
        return val

    area = pick(JSON_KEYS["area"])
    if area is None:
        area = pick_stage("__design__instance__area")
    wns = pick(JSON_KEYS["wns"])
    if wns is None:
        wns = pick_stage("__timing__setup__ws")
    power = pick(JSON_KEYS["power"])
    if power is None:
        power = pick_stage("__power__total")
    wl = pick(JSON_KEYS["wl"])
    clk = pick(JSON_KEYS["clock_period"])
    return area, wns, power, wl, clk


RE_WNS = [
    re.compile(r"\btiming[_\s-]*worst[_\s-]*slack\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bWNS\s*\(ns\)\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bWNS\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bworst\s+slack\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bsetup\s+wns\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
]

RE_AREA = [
    re.compile(r"\barea\b\s*[:=]?\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"(?:Standard\s+cells\s+area|Placed\s+Cell\s+Area)\s*[:=]\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"\btotal\s+area\b\s*[:=]?\s*([0-9.]+)", re.IGNORECASE),
]

RE_POWER = [
    re.compile(r"\btotal\s+power\b\s*[:=]?\s*([0-9.eE+-]+)", re.IGNORECASE),
    re.compile(r"\bpower\b\s*[:=]?\s*([0-9.eE+-]+)", re.IGNORECASE),
    re.compile(r"\btotal\s+dynamic\s+power\b\s*[:=]?\s*([0-9.eE+-]+)", re.IGNORECASE),
]

RE_WL = [
    re.compile(r"Total\s+wire\s+length\s*=\s*([0-9]+(?:\.\d+)?)\s*um\b", re.IGNORECASE),
    re.compile(r"\bpost[-\s]*dr\s+wirelength\b\s*[:=]?\s*([0-9]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\brouted\s+wirelength\b\s*[:=]?\s*([0-9]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bwirelength\b\s*[:=]?\s*([0-9]+(?:\.\d+)?)", re.IGNORECASE),
]

RE_CLOCK = [
    re.compile(r"\bclock\s+period\s+to\s*([0-9]+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bclock\s+period\b\s*[:=]\s*([0-9]+(?:\.\d+)?)\b", re.IGNORECASE),
]

_RE_CLK_FILE = re.compile(r"clock\s+period\s+from\s+SDC\s+file:\s*([^\s]+clock_period\.txt)", re.IGNORECASE)


def extract_from_text(p: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    txt = _read_text_best_effort(p)
    if not txt:
        return None, None, None, None, None

    area = _last_match_float(RE_AREA, txt)
    wns = _last_match_float(RE_WNS, txt)
    power = _last_match_float(RE_POWER, txt)
    wl = _last_match_float(RE_WL, txt)
    clk = _last_match_float(RE_CLOCK, txt)
    return area, wns, power, wl, clk


def _read_single_number_file(p: Path) -> Optional[float]:
    try:
        s = p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None
    return safe_float(s)


def extract_wl_clk_from_run_log(run_log: Path, flow_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if not run_log.exists():
        return None, None, None, None

    txt = _read_text_best_effort(run_log, max_bytes=16 * 1024 * 1024)
    if not txt:
        return None, None, None, None

    wl = _last_match_float(RE_WL, txt)
    wl_src = _public_path(run_log, root=flow_dir) if wl is not None else None

    clk: Optional[float] = None
    clk_src: Optional[str] = None

    m = None
    for mm in _RE_CLK_FILE.finditer(txt):
        m = mm
    if m:
        hinted = m.group(1)
        hinted_path = Path(hinted)
        if not hinted_path.is_absolute():
            hinted_path = (flow_dir / hinted_path).resolve()
        clk = _read_single_number_file(hinted_path)
        if clk is not None:
            clk_src = _public_path(hinted_path, root=flow_dir)

    if clk is None:
        clk = _last_match_float(RE_CLOCK, txt)
        if clk is not None:
            clk_src = _public_path(run_log, root=flow_dir)

    return wl, clk, wl_src, clk_src


def generate_metrics(
    flow_dir: Path,
    platform: str,
    design: str,
    variant: str,
    out_path: Path,
    *,
    run_log: Optional[Path] = None,
) -> Dict[str, Any]:
    dprint(f"[genMetrics] flow_dir={flow_dir}")
    dprint(f"[genMetrics] platform={platform} design={design} variant={variant}")

    reports_dir = flow_dir / "reports" / platform / design / variant
    logs_dir = flow_dir / "logs" / platform / design / variant
    results_dir = flow_dir / "results" / platform / design / variant

    candidates: List[Path] = []
    for d in [reports_dir, logs_dir, results_dir]:
        if d.exists():
            candidates += [
                p for p in d.rglob("*")
                if p.is_file() and p.suffix.lower() in [".json", ".rpt", ".log", ".txt"]
            ]

    candidates.sort(key=lambda p: (file_priority_score(p), p.stat().st_mtime))

    area_pick = MetricPick()
    wns_pick = MetricPick()
    power_pick = MetricPick()
    wl_pick = MetricPick()
    clk_pick = MetricPick()

    for p in candidates:
        score = file_priority_score(p)
        if p.suffix.lower() == ".json":
            a, w, pw, wl, clk = extract_from_json(p)
        else:
            a, w, pw, wl, clk = extract_from_text(p)

        area_pick.consider(a, p, score)
        wns_pick.consider(w, p, score)
        power_pick.consider(pw, p, score)
        wl_pick.consider(wl, p, score)
        clk_pick.consider(clk, p, score)

    # Optional run log overrides (WL/clock_period).
    if run_log is not None:
        wl_from_run, clk_from_run, wl_src, clk_src = extract_wl_clk_from_run_log(run_log, flow_dir)
        if wl_from_run is not None:
            wl_pick.value, wl_pick.source, wl_pick.score = wl_from_run, wl_src, 10**9
        if clk_from_run is not None:
            clk_pick.value, clk_pick.source, clk_pick.score = clk_from_run, clk_src, 10**9

    area = area_pick.value if area_pick.value is not None else float("inf")
    wns = wns_pick.value if wns_pick.value is not None else -100.0
    power = power_pick.value if power_pick.value is not None else float("inf")
    wl = wl_pick.value if wl_pick.value is not None else float("inf")
    clock_period = clk_pick.value if clk_pick.value is not None else float("inf")

    ecp = float("inf")
    if math.isfinite(clock_period) and math.isfinite(wns) and (wns > -1e9):
        ecp = float(clock_period) - float(wns)

    result: Dict[str, Any] = {
        "platform": platform,
        "design": design,
        "variant": variant,
        "area": area,
        "wns": wns,
        "power": power,
        "wl": wl,
        "clock_period": clock_period,
        "ecp": ecp,
        "sources": {
            "area": area_pick.source,
            "wns": wns_pick.source,
            "power": power_pick.source,
            "wl": wl_pick.source,
            "clock_period": clk_pick.source,
            "ecp": "derived(clock_period - wns)",
        },
        "units_hint": {
            "area": "um^2",
            "power": "W",
            "wl": "um (from DRT 'Total wire length')",
            "time": "depends on tech/libs (ps or ns); keep consistent between clock_period and wns",
        },
        "searched_dirs": {
            "reports_dir": _public_path(reports_dir, root=flow_dir),
            "logs_dir": _public_path(logs_dir, root=flow_dir),
            "results_dir": _public_path(results_dir, root=flow_dir),
        },
        "run_log": _public_path(run_log, root=flow_dir) if run_log is not None else None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_dir", required=True, help="ORFS flow directory (contains reports/ logs/ results/)")
    ap.add_argument("--platform", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True, help="Output metrics.json path")
    ap.add_argument("--run_log", default=None, help="Optional: path to the run's master log (e.g., work_dir/<run_id>/orfs.log)")
    args = ap.parse_args(argv)

    flow_dir = Path(args.flow_dir).resolve()
    out_path = Path(args.out).resolve()
    run_log = Path(args.run_log).resolve() if args.run_log else None

    metrics = generate_metrics(flow_dir, args.platform, args.design, args.variant, out_path, run_log=run_log)

    # Keep legacy behavior: return non-zero if core metrics are missing.
    if (not math.isfinite(metrics["area"])) or (not math.isfinite(metrics["power"])) or metrics["wns"] <= -99.0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
