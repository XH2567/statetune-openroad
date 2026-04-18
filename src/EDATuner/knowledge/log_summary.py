# src/log_summary.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- regex patterns (best-effort) ----
_RE_WNS_1 = re.compile(r"\bWNS\b[^0-9+\-]*([+\-]?[0-9.]+(?:e[+\-]?\d+)?)", re.IGNORECASE)
_RE_TNS_1 = re.compile(r"\bTNS\b[^0-9+\-]*([+\-]?[0-9.]+(?:e[+\-]?\d+)?)", re.IGNORECASE)
_RE_WORST_SLACK = re.compile(r"worst\s+slack\s*[:=]?\s*([+\-]?[0-9.]+(?:e[+\-]?\d+)?)", re.IGNORECASE)

_RE_OVERFLOW = re.compile(r"\bOverflow\b[^0-9]*([0-9.]+(?:e[+\-]?\d+)?)", re.IGNORECASE)

_RE_DRC_PATTERNS = [
    re.compile(r"Total\s+DRC\s+violations\s*[:=]\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bDRC\b.*?\bviolations?\b[^0-9]*(\d+)", re.IGNORECASE),
    re.compile(r"violation(?:s)?\s+count\s*[:=]\s*(\d+)", re.IGNORECASE),
]

_RE_HOLD_SLACK = re.compile(r"\bhold\b.*?\bslack\b[^0-9+\-]*([+\-]?[0-9.]+(?:e[+\-]?\d+)?)", re.IGNORECASE)

_RE_TIMEOUT = re.compile(r"\btimeout\b|\btimed out\b|\btime limit\b", re.IGNORECASE)
_RE_CRASH = re.compile(r"\bsegmentation fault\b|\bcore dumped\b|\bassert\b|\bfatal\b|\bexception\b", re.IGNORECASE)

_STAGE_HINTS: List[Tuple[str, List[str]]] = [
    (
        "route",
        [
            # OpenROAD
            "detailed route",
            "detailed routing",
            "tritonroute",
            "routing",
            "global route",
            "fastroute",
            "grt",
            "drt",
            # Cadence Innovus
            "innovus",
            "routedesign",
            "nanoroute",
        ],
    ),
    (
        "cts",
        [
            "cts",
            "clock tree",
            "clock_tree",
            # Cadence Innovus
            "clockdesign",
            "ccopt",
        ],
    ),
    (
        "place",
        [
            "placement",
            "global placement",
            "detailed placement",
            "gpl",
            "dpl",
            "place",
            # Cadence Innovus
            "place_design",
            "placedesign",
        ],
    ),
    (
        "floorplan",
        [
            "floorplan",
            "init_floorplan",
            # Cadence Innovus
            "init_design",
            "initdesign",
            "floorplan",
        ],
    ),
    (
        "synth",
        [
            "synthesis",
            "yosys",
            "abc",
            # Cadence Genus
            "genus",
            "synthesize",
        ],
    ),
    ("drc", ["drc", "violation", "antenna"]),
    (
        "power",
        [
            "ir drop",
            "power",
            "pdn",
            "rail",
            # Cadence Innovus
            "report_power",
        ],
    ),
]


def _read_tail_bytes(path: str, *, max_bytes: int = 200_000) -> str:
    p = Path(str(path))
    if (not p.exists()) or (not p.is_file()):
        return ""
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _last_float(regex: re.Pattern, text: str) -> Optional[float]:
    last = None
    for m in regex.finditer(text):
        try:
            last = float(m.group(1))
        except Exception:
            continue
    return last


def _last_int(regex: re.Pattern, text: str) -> Optional[int]:
    last = None
    for m in regex.finditer(text):
        try:
            last = int(m.group(1))
        except Exception:
            continue
    return last


def _infer_fail_stage(text: str, *, make_target: Optional[str], default_stage: str = "") -> str:
    t = (text or "").lower()
    for stage, keys in _STAGE_HINTS:
        for k in keys:
            if k in t:
                return stage
    if make_target:
        return str(make_target)
    return default_stage


def analyze_log_summary(
    *,
    log_path: str,
    metrics: Dict[str, Any],
    runtime_sec: float,
    success: bool,
    stage: str,
    fidelity: str,
    make_target: Optional[str],
    fail_log_tail: str = "",
) -> Dict[str, Any]:
    """
    Structured log summary:
      {
        "symptoms": [...],
        "numbers": {"wns":..., "tns":..., "drc_count":..., "route_overflow":..., "runtime_sec":...},
        "stage": {"experiment_stage":..., "fidelity":..., "make_target":..., "fail_stage":...},
        "brief": "..."
      }
    """
    tail = _read_tail_bytes(log_path, max_bytes=200_000)
    text = tail if tail.strip() else (fail_log_tail or "")

    numbers: Dict[str, Any] = {"runtime_sec": float(runtime_sec)}
    try:
        numbers["wns"] = float(metrics.get("wns")) if metrics is not None else None
    except Exception:
        numbers["wns"] = None

    wns2 = _last_float(_RE_WNS_1, text) or _last_float(_RE_WORST_SLACK, text)
    tns2 = _last_float(_RE_TNS_1, text)
    if numbers.get("wns") is None and wns2 is not None:
        numbers["wns"] = float(wns2)
    if tns2 is not None:
        numbers["tns"] = float(tns2)

    overflow = _last_float(_RE_OVERFLOW, text)
    if overflow is not None:
        numbers["route_overflow"] = float(overflow)

    drc = None
    for rgx in _RE_DRC_PATTERNS:
        x = _last_int(rgx, text)
        if x is not None:
            drc = x
    if drc is not None:
        numbers["drc_count"] = int(drc)

    hold_slack = _last_float(_RE_HOLD_SLACK, text)
    if hold_slack is not None:
        numbers["hold_slack"] = float(hold_slack)

    symptoms: List[str] = []
    if (not success) and _RE_TIMEOUT.search(text or ""):
        symptoms.append("timeout")
    if (not success) and _RE_CRASH.search(text or ""):
        symptoms.append("crash")

    if overflow is not None and overflow > 0.0:
        symptoms.append("overflow")
        if overflow >= 0.10:
            symptoms.append("congestion")

    if drc is not None and drc > 0:
        symptoms.append("drc")

    try:
        wns_v = numbers.get("wns", None)
        if wns_v is not None and float(wns_v) < 0.0:
            symptoms.append("setup")
    except Exception:
        pass

    if hold_slack is not None and hold_slack < 0.0:
        symptoms.append("hold")

    inferred_fail_stage = ""
    if not success:
        inferred_fail_stage = _infer_fail_stage(text, make_target=make_target, default_stage=str(stage))

    stage_info = {
        "experiment_stage": str(stage),
        "fidelity": str(fidelity),
        "make_target": ("" if make_target is None else str(make_target)),
        "fail_stage": inferred_fail_stage,
    }

    brief = ""
    if text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines[-200:]):
            ll = ln.lower()
            if ("error" in ll) or ("fatal" in ll) or ("exception" in ll) or ("warning" in ll) or ("assert" in ll):
                brief = ln[:240]
                break
        if not brief and lines:
            brief = lines[-1][:240]

    # de-dup symptoms keep order
    seen = set()
    sym2: List[str] = []
    for s in symptoms:
        if s in seen:
            continue
        seen.add(s)
        sym2.append(s)

    return {"symptoms": sym2, "numbers": numbers, "stage": stage_info, "brief": brief}
