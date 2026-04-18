# src/kb.py
from __future__ import annotations

import csv
import json
import threading
import time
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.ranktuner_utils import is_pareto_dominated
    


class KB:
    """
    Minimal KB:
      - meta.json
      - runs.jsonl (append-only)
      - summary.csv
      - pareto.json

    Compatible with being constructed with either:
      KB("./runs/kb")  -> directory
      KB("./runs/kb/runs.jsonl") -> file
    """

    def __init__(self, kb_path: Any):
        p = Path(kb_path).resolve()
        if p.suffix.lower() == ".jsonl":
            self.kb_dir = p.parent
            self.runs_jsonl = p
        else:
            self.kb_dir = p
            self.runs_jsonl = self.kb_dir / "runs.jsonl"

        self.meta_json = self.kb_dir / "meta.json"
        self.summary_csv = self.kb_dir / "summary.csv"
        self.pareto_json = self.kb_dir / "pareto.json"
        self.out_dir = self.kb_dir

        self.kb_dir.mkdir(parents=True, exist_ok=True)
        if not self.runs_jsonl.exists():
            self.runs_jsonl.write_text("", encoding="utf-8")

        self._lock = threading.Lock()

    @staticmethod
    def _sanitize_path_for_export(path: Any) -> str:
        """Return a privacy-safe path string for persisted metadata."""
        try:
            s = str(path or "").strip()
            if not s:
                return ""
            p = Path(s)
            return p.name if p.is_absolute() else s
        except Exception:
            return ""

    def write_meta(self, meta: Dict[str, Any]) -> None:
        meta = dict(meta)
        meta.setdefault("ts", time.time())
        with self._lock:
            self.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _tail_lines_best_effort(self, path: str, *, tail_lines: int = 60, max_bytes: int = 200_000) -> str:
        """
        Best-effort tail reader for a growing log file.
        - reads only last max_bytes
        - tolerant to concurrent writes / partial utf-8
        - retries a few times on transient OS errors
        """
        p = Path(str(path))
        if (not p.exists()) or (not p.is_file()):
            return ""

        for _ in range(3):
            try:
                size = p.stat().st_size
                with p.open("rb") as f:
                    if size > max_bytes:
                        f.seek(size - max_bytes)
                    data = f.read()

                txt = data.decode("utf-8", errors="ignore")
                lines = txt.splitlines()

                # If the read starts from the middle of the file, the first line may be partial; drop it.
                if size > max_bytes and lines:
                    lines = lines[1:]

                return "\n".join(lines[-tail_lines:]).strip()
            except (OSError, PermissionError):
                time.sleep(0.05)
            except Exception:
                return ""
        return ""

    def append_run(self, record: Dict[str, Any], *, include_log_tail: bool = True, tail_lines: int = 60) -> None:
        rec = dict(record)
        rec.setdefault("ts", time.time())

        if include_log_tail:
            lp = rec.get("log_path", "")
            if lp:
                try:
                    Path(lp).read_text(encoding="utf-8", errors="replace").splitlines()
                    rec["log_tail"] = self._tail_lines_best_effort(lp, tail_lines=tail_lines)
                except Exception:
                    pass
            rec["log_path"] = self._sanitize_path_for_export(lp)

        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with self.runs_jsonl.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _iter_lines(self) -> Iterable[Dict[str, Any]]:
        if not self.runs_jsonl.exists():
            return
        with self.runs_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def export_csv(self, history: List[Dict[str, Any]]) -> str:
        """
        Export all run records to kb/summary.csv.
        Writes to self.summary_csv (defined in __init__).
        """
        out = self.summary_csv
        out.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in history:
            m = r.get("metrics", {}) or {}

            ls = r.get("log_summary", {}) or {}
            nums = (ls.get("numbers", {}) or {}) if isinstance(ls, dict) else {}
            stg = (ls.get("stage", {}) or {}) if isinstance(ls, dict) else {}
            symptoms = ls.get("symptoms", []) if isinstance(ls, dict) else []
            if not isinstance(symptoms, list):
                symptoms = []

            delta = stg.get("qor_vs_parent", {}) if isinstance(stg, dict) else {}
            if not isinstance(delta, dict):
                delta = {}

            rows.append(
                {
                    "run_id": r.get("run_id", ""),
                    "fidelity": r.get("fidelity", ""),
                    "stage": r.get("stage", ""),
                    "parent_run_id": r.get("parent_run_id", ""),
                    "success": r.get("success", False),
                    "metrics_status": r.get("metrics_status", ""),
                    "metrics_usable": r.get("metrics_usable", False),
                    "metrics_valid": r.get("metrics_valid", False),
                    "runtime_sec": r.get("runtime_sec", 0.0),
                    "area": m.get("area", float("inf")),
                    "wns": m.get("wns", -100.0),
                    "power": m.get("power", float("inf")),
                    "proxy_tns": m.get("proxy_tns", ""),
                    "proxy_drc_count": m.get("proxy_drc_count", ""),
                    "proxy_route_overflow": m.get("proxy_route_overflow", ""),
                    # structured log summary fields
                    "symptoms": ",".join([str(x) for x in symptoms]),
                    "tns": nums.get("tns", ""),
                    "drc_count": nums.get("drc_count", ""),
                    "route_overflow": nums.get("route_overflow", ""),
                    "fail_stage": stg.get("fail_stage", ""),
                    "qor_regressed_vs_parent": stg.get("qor_regressed", ""),
                    "delta_wns_vs_parent": delta.get("wns", ""),
                    "delta_area_vs_parent": delta.get("area", ""),
                    "delta_power_vs_parent": delta.get("power", ""),
                    "brief": (ls.get("brief", "") if isinstance(ls, dict) else "")[:160],
                    "wall_start_sec": r.get("wall_start_sec", ""),
                    "wall_end_sec": r.get("wall_end_sec", ""),
                    "experiment_id": r.get("experiment_id", ""),
                    "cum_hv": r.get("cum_hv", ""),
                    "cum_hv_delta": r.get("cum_hv_delta", ""),
                    "is_pareto": r.get("is_pareto", ""),
                    "pareto_size": r.get("pareto_size", ""),
                    "cum_launched": r.get("cum_launched", ""),
                    "cum_full_valid": r.get("cum_full_valid", ""),
                    "cum_success_count": r.get("cum_success_count", ""),
                    "cum_fail_count": r.get("cum_fail_count", ""),
                    "failure_category": r.get("failure_category", ""),
                    "ehvi_score": r.get("ehvi_score", ""),
                    "num_hard_rules": r.get("num_hard_rules", ""),
                    "num_soft_heuristics": r.get("num_soft_heuristics", ""),
                    "predicted_runtime_mean": r.get("predicted_runtime_mean", ""),
                    "runtime_prediction_error": r.get("runtime_prediction_error", ""),
                    "log_path": self._sanitize_path_for_export(r.get("log_path", "")),
                }
            )

        fieldnames = list(rows[0].keys()) if rows else [
            "run_id","fidelity","stage","parent_run_id",
            "success","metrics_status","metrics_usable","metrics_valid",
            "runtime_sec","area","wns","power",
            "proxy_tns","proxy_drc_count","proxy_route_overflow",
            "symptoms","tns","drc_count","route_overflow","fail_stage",
            "qor_regressed_vs_parent","delta_wns_vs_parent","delta_area_vs_parent","delta_power_vs_parent",
            "brief","log_path"
        ]

        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)

        return str(out)

    def export_pareto(self, full_history: List[Dict[str, Any]]) -> str:
        """
        Export Pareto frontier from FULL history into pareto.json.

        Objective:
        - minimize area
        - minimize power
        - maximize wns
        """
        from pathlib import Path
        import json
        import math
        import time

        def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            # a dominates b if it's no worse in all objectives and better in at least one
            aa = float(a.get("area", float("inf")))
            ap = float(a.get("power", float("inf")))
            aw = float(a.get("wns", -100.0))

            ba = float(b.get("area", float("inf")))
            bp = float(b.get("power", float("inf")))
            bw = float(b.get("wns", -100.0))

            if (not math.isfinite(aa)) or (not math.isfinite(ap)) or (aw <= -99.0):
                return False
            if (not math.isfinite(ba)) or (not math.isfinite(bp)) or (bw <= -99.0):
                return False

            no_worse = (aa <= ba) and (ap <= bp) and (aw >= bw)
            strictly_better = (aa < ba) or (ap < bp) or (aw > bw)
            return no_worse and strictly_better

        # collect valid full points
        valid = []
        for r in full_history:
            if not r.get("success", False):
                continue
            m = r.get("metrics", {}) or {}
            area = float(m.get("area", float("inf")))
            power = float(m.get("power", float("inf")))
            wns = float(m.get("wns", -100.0))
            if (not math.isfinite(area)) or (not math.isfinite(power)) or (wns <= -99.0):
                continue
            valid.append(r)

        pareto = []
        for a in valid:
            dominated = False
            for b in valid:
                if a.get("run_id") == b.get("run_id"):
                    continue
                if _dominates(b.get("metrics", {}) or {}, a.get("metrics", {}) or {}):
                    dominated = True
                    break
            if not dominated:
                pareto.append(a)

        payload = {
            "ts": time.time(),
            "num_full_valid": len(valid),
            "num_pareto": len(pareto),
            "pareto": pareto,
        }

        out_path = getattr(self, "pareto_json", None)
        if out_path is None:
            kb_dir = Path(getattr(self, "kb_dir", "."))
            out_path = kb_dir / "pareto.json"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path)
