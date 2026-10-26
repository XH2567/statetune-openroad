#!/usr/bin/env python3
"""
Summarize StateTune/OpenROAD runs into the markdown table used in docs/OPENROAD_PORT.md.

    python tools/summarize_results.py                                  # results/openroad_2026-10-24
    python tools/summarize_results.py runs                             # live run dirs (runs/openroad_*)

Per case, over FULL-valid evaluations only (complete flow to `finish`, finite WNS/area/power):
  best WNS / best area / best power  -> each is the best value of that metric, possibly from different points
  Pareto  -> size of the non-dominated set (max WNS, min area, min power)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [  # (PDK, design, run name, WNS unit, clock period)
    ("NanGate45", "AES", "aes_ng45", "ns", 0.82),
    ("NanGate45", "JPEG", "jpeg_ng45", "ns", 1.0),
    ("NanGate45", "IBEX", "ibex_ng45", "ns", 2.2),
    ("ASAP7", "AES", "aes_asap7", "ps", 380),
    ("ASAP7", "JPEG", "jpeg_asap7", "ps", 680),
    ("ASAP7", "IBEX", "ibex_asap7", "ps", 1000),
]


def load(src: Path, name: str) -> list[dict]:
    gz = src / name / "runs.jsonl.gz"           # results/ layout
    if gz.exists():
        return [json.loads(l) for l in gzip.open(gz, "rt", encoding="utf-8")]
    return [json.loads(l) for l in open(src / f"openroad_{name}" / "kb" / "runs.jsonl", encoding="utf-8")]  # runs/ layout


def dominates(a: dict, b: dict) -> bool:
    ge = a["wns"] >= b["wns"] and a["area"] <= b["area"] and a["power"] <= b["power"]
    gt = a["wns"] > b["wns"] or a["area"] < b["area"] or a["power"] < b["power"]
    return ge and gt


def main(argv: list[str]) -> int:
    src = ROOT / (argv[0] if argv else "results/openroad_2026-10-24")
    print("| PDK | Design | Clock | Best WNS | Best Area (µm²) | Best Power (mW) | Pareto | Full evals | All evals |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for pdk, design, name, unit, clk in CASES:
        rs = load(src, name)
        fv = [r["metrics"] for r in rs if r.get("metrics_valid")]
        pareto = [m for m in fv if not any(dominates(o, m) for o in fv)]
        print(f"| {pdk} | {design} | {clk} {unit} | {max(m['wns'] for m in fv):+.4g} {unit} "
              f"| {min(m['area'] for m in fv):.1f} | {min(m['power'] for m in fv) * 1e3:.2f} "
              f"| {len(pareto)} | {len(fv)} | {len(rs)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
