# src/openroad_runner.py
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import signal
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .eda_interface import EDARunner
from ..utils.ranktuner_utils import is_pareto_dominated


_RE_STD_AREA    = re.compile(r"(?:Standard cells area|Placed Cell Area)\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
_RE_UTIL        = re.compile(r"Utilization\s*([0-9.]+)\s*%", re.IGNORECASE)
_RE_WORST_SL    = re.compile(r"worst slack\s*([+-]?[0-9.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
_RE_OVERFLOW    = re.compile(r"Overflow\s*([0-9.]+)", re.IGNORECASE)
_RE_HPWL        = re.compile(r"HPWL\s*([0-9.]+)", re.IGNORECASE)
_RE_DRT_WL      = re.compile(r"Total\s+wire\s+length\s*=\s*([0-9]+(?:\.\d+)?)\s*um\b", re.IGNORECASE)
_RE_HPW_LIKE    = re.compile(r"\bHPWL\b[^0-9]*([0-9]+(?:\.\d+)?)", re.IGNORECASE)
_RE_GR_WL_LIKE  = re.compile(r"\b(global|groute|global\s*route).{0,60}\bwire\s*length\b[^0-9]*([0-9]+(?:\.\d+)?)", re.IGNORECASE)

def _read_text_best_effort(p: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
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

def extract_wl_proxy_from_orfs_log(log_file: Path) -> Dict[str, Any]:
    """
    Returns:
      {
        "wl": <float> or inf,
        "__wl_kind__": "drwl" | "hpwl" | "grwl" | "missing",
        "__wl_src__": "<log path>" or None
      }
    """
    txt = _read_text_best_effort(log_file)
    if not txt:
        return {"wl": float("inf"), "__wl_kind__": "missing", "__wl_src__": None}

    # 1) Prefer DRWL from DRT
    m = None
    for mm in _RE_DRT_WL.finditer(txt):
        m = mm
    if m:
        return {"wl": float(m.group(1)), "__wl_kind__": "drwl", "__wl_src__": str(log_file)}

    # 2) Fallback: HPWL (proxy)
    m = None
    for mm in _RE_HPW_LIKE.finditer(txt):
        m = mm
    if m:
        return {"wl": float(m.group(1)), "__wl_kind__": "hpwl", "__wl_src__": str(log_file)}

    # 3) Fallback: global-route wirelength (if log prints something similar)
    m = None
    for mm in _RE_GR_WL_LIKE.finditer(txt):
        m = mm
    if m:
        return {"wl": float(m.group(2)), "__wl_kind__": "grwl", "__wl_src__": str(log_file)}

    return {"wl": float("inf"), "__wl_kind__": "missing", "__wl_src__": str(log_file)}


class OpenRoadFlow(EDARunner):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.orfs_root = Path(self.cfg["orfs"]["root"]).resolve()
        self.flow_dir = (self.orfs_root / self.cfg["orfs"].get("flow_dir", "flow")).resolve()
        self.work_dir = Path(self.cfg["work_dir"]).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Debug toggles:
        # - export RANKTUNER_DEBUG=1
        # - or set config.debug=true / config.orfs.debug=true
        self.debug = bool(int(os.getenv("RANKTUNER_DEBUG", "0"))) or bool(self.cfg.get("debug", False)) or bool(self.cfg.get("orfs", {}).get("debug", False))

        mk = self.flow_dir / "Makefile"
        if not mk.exists():
            raise FileNotFoundError(f"ORFS Makefile not found: {mk}")

    # -------- required by EDARunner (abstract) --------
    def run(
        self,
        params: Dict[str, Any],
        run_id: str,
        script_override: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        make_target = kwargs.get("make_target", None)
        timeout_sec = int(kwargs.get("timeout_sec", 7200))
        allow_missing_metrics = bool(kwargs.get("allow_missing_metrics", False))

        try:
            asyncio.get_running_loop()
            raise RuntimeError("OpenRoadFlow.run() called inside event loop; use await run_async().")
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                return asyncio.run(
                    self.run_async(
                        params,
                        run_id,
                        script_override=script_override,
                        make_target=make_target,
                        timeout_sec=timeout_sec,
                        allow_missing_metrics=allow_missing_metrics,
                    )
                )
            raise

    # -------- helpers --------
    def _bash_quote(self, s: Any) -> str:
        s = str(s)
        return "'" + s.replace("'", "'\"'\"'") + "'"

    async def _run_cmd_logged(
        self,
        argv: List[str],
        *,
        cwd: Path,
        log_file: Path,
        timeout_sec: Optional[int],
        append: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        with open(log_file, mode, encoding="utf-8", errors="replace") as log:
            # Key point: make this command a new session / process-group leader
            # This allows timeout handling to kill the entire process group (wrapper + make + flow.sh + openroad)
            popen_kwargs = dict(
                stdout=log,
                stderr=log,
                cwd=str(cwd),
                env=(env if env is not None else None),
            )
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True

            proc = await asyncio.create_subprocess_exec(*argv, **popen_kwargs)

            async def _kill_tree() -> None:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        return
                    # Give it a short grace period for clean shutdown
                    await asyncio.sleep(2)
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    # Fallback for non-POSIX platforms
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass

            try:
                if timeout_sec and timeout_sec > 0:
                    await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
                else:
                    await proc.wait()
            except asyncio.TimeoutError:
                try:
                    await _kill_tree()
                finally:
                    await proc.wait()
                return 124

            return int(proc.returncode or 0)

    def _run_dir(self, run_id: str) -> Path:
        return self.work_dir / run_id

    def _metrics_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "metrics.json"

    def _dbg(self, msg: str) -> None:
        if getattr(self, "debug", False):
            print(msg)

    def _copy_metrics_from_orfs_outputs(self, run_id: str, metrics_out: Path, log_file: Path) -> bool:
        """If a metrics.json exists under reports/logs/results but not in run_dir, copy it back."""
        platform = self.cfg["orfs"]["platform"]
        design = self.cfg["design"]["name"]
        candidates = [
            self.flow_dir / "reports" / platform / design / run_id / "metrics.json",
            self.flow_dir / "logs" / platform / design / run_id / "metrics.json",
            self.flow_dir / "results" / platform / design / run_id / "metrics.json",
        ]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            return False
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        src_p = existing[0]
        try:
            metrics_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(src_p), str(metrics_out))
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n# Copied metrics.json from {src_p} -> {metrics_out}\n")
            self._dbg(f"    [METRICS] copied from {src_p}")
            return True
        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n# Failed copying metrics from {src_p}: {type(e).__name__}: {e}\n")
            return False

    def _get_results_dir(self, run_id: str) -> Path:
        design = self.cfg["design"]["name"]
        platform = self.cfg["orfs"]["platform"]
        return self.flow_dir / "results" / platform / design / run_id

    def _get_reports_dir(self, run_id: str) -> Path:
        design = self.cfg["design"]["name"]
        platform = self.cfg["orfs"]["platform"]
        return self.flow_dir / "reports" / platform / design / run_id

    def _get_logs_dir(self, run_id: str) -> Path:
        design = self.cfg["design"]["name"]
        platform = self.cfg["orfs"]["platform"]
        return self.flow_dir / "logs" / platform / design / run_id

    # -------- override hook --------
    def _prepare_extra_vars_from_script_override(self, run_dir: Path, script_override: Optional[Any]) -> Dict[str, Any]:
        if script_override is None:
            return {}
        hook_var = "FASTROUTE_TCL"
        content = script_override
        if isinstance(script_override, dict):
            hook_var = str(script_override.get("hook_var", hook_var)).strip().upper()
            content = str(script_override.get("content", ""))
        tcl_path = run_dir / f"override_{hook_var.lower()}.tcl"
        tcl_path.write_text(str(content) + "\n", encoding="utf-8")
        return {hook_var: str(tcl_path.resolve())}

    # -------- config.mk (export VAR = value) --------
    def _generate_config_mk(
        self,
        run_dir: Path,
        params: Dict[str, Any],
        *,
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Dict[str, str]]:
        design_name = self.cfg["design"]["name"]
        platform = self.cfg["orfs"]["platform"]

        verilog_cfg = self.cfg["design"]["verilog"]
        if isinstance(verilog_cfg, str):
            verilog_cfg = [verilog_cfg]
        verilog_files = [str(Path(v).resolve()) for v in verilog_cfg]
        sdc_file = str(Path(self.cfg["design"]["sdc"]).resolve())

        inc_cfg = self.cfg["design"].get("verilog_include_dirs", [])
        if isinstance(inc_cfg, str):
            inc_cfg = [inc_cfg]
        verilog_inc_dirs = [str(Path(d).resolve()) for d in inc_cfg]

        vars_dict: Dict[str, str] = {
            "DESIGN_NAME": design_name,
            "PLATFORM": platform,
            "VERILOG_FILES": " ".join(verilog_files),
            "SDC_FILE": sdc_file,
        }

        # Export include directories (common ORFS variable names)
        if verilog_inc_dirs:
            vars_dict["VERILOG_INCLUDE_DIRS"] = " ".join(verilog_inc_dirs)
            # Some environments/scripts use this name, so export it as a compatibility alias as well
            vars_dict.setdefault("SYNTH_VERILOG_INCLUDE_DIRS", vars_dict["VERILOG_INCLUDE_DIRS"])


        param_mapping = {
            "place_density": "PLACE_DENSITY",
            "place_padding": "CELL_PAD_IN_SITES_GLOBAL_PLACEMENT",
            "gp_timing_driven": "GPL_TIMING_DRIVEN",
            "gp_routability_driven": "GPL_ROUTABILITY_DRIVEN",
            "gp_overflow": "GPL_OVERFLOW",
            "synth_flatten": "SYNTH_FLATTEN",
            "synth_retime": "SYNTH_RETIME",

            # floorplan knobs (optional)
            "fp_sizing": "FP_SIZING",
            "core_utilization": "CORE_UTILIZATION",
            "core_aspect_ratio": "CORE_ASPECT_RATIO",
            "core_margin": "CORE_MARGIN",
            "die_area": "DIE_AREA",
            "core_area": "CORE_AREA",
            "fp_core_util": "FP_CORE_UTIL",
            "fp_aspect_ratio": "FP_ASPECT_RATIO",
        }

        for k, v in (params or {}).items():
            k = str(k)
            mk_var = param_mapping.get(k, k.upper())
            if isinstance(v, bool):
                val = "1" if v else "0"
            elif isinstance(v, float):
                val = f"{v:.6f}"
            elif isinstance(v, (list, tuple)):
                val = " ".join(map(str, v))
            else:
                val = str(v)
            vars_dict[mk_var] = val

        if extra_vars:
            for k, v in extra_vars.items():
                vars_dict[str(k)] = str(v)

        # floorplan defaults (avoid ORFS error)
        has_abs = ("DIE_AREA" in vars_dict) or ("CORE_AREA" in vars_dict)
        vars_dict.setdefault("FP_SIZING", "absolute" if has_abs else "relative")
        if vars_dict["FP_SIZING"].strip().lower() == "absolute":
            if "DIE_AREA" not in vars_dict or "CORE_AREA" not in vars_dict:
                vars_dict["FP_SIZING"] = "relative"
        if vars_dict["FP_SIZING"].strip().lower() == "relative":
            vars_dict.setdefault("CORE_UTILIZATION", "50")
            vars_dict.setdefault("CORE_ASPECT_RATIO", "1.0")
            vars_dict.setdefault("CORE_MARGIN", "2")
            vars_dict.setdefault("FP_CORE_UTIL", vars_dict["CORE_UTILIZATION"])
            vars_dict.setdefault("FP_ASPECT_RATIO", "1")

        mk_path = run_dir / "config.mk"
        lines = [f"# Generated config for {run_dir.name}"]
        for k, v in vars_dict.items():
            lines.append(f"export {k} = {v}")
        mk_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return mk_path, vars_dict

    def _debug_metrics_log(self, run_id: str, log_file: Path) -> None:
        """Append debug info about where metrics.log might be and a small head snippet."""
        platform = self.cfg["orfs"]["platform"]
        design = self.cfg["design"]["name"]
        reports_dir = self.flow_dir / "reports" / platform / design / run_id
        logs_dir = self.flow_dir / "logs" / platform / design / run_id
        results_dir = self.flow_dir / "results" / platform / design / run_id
        candidates = [
            reports_dir / "metrics.log",
            logs_dir / "metrics.log",
            results_dir / "metrics.log",
            self._run_dir(run_id) / "metrics.log",
        ]
        with open(log_file, "a", encoding="utf-8", errors="replace") as log:
            log.write("\n\n# --- metrics.log debug ---\n")
            log.write(f"# reports_dir={reports_dir} exists={reports_dir.exists()}\n")
            log.write(f"# logs_dir={logs_dir} exists={logs_dir.exists()}\n")
            log.write(f"# results_dir={results_dir} exists={results_dir.exists()}\n")
            for p in candidates:
                if p.exists():
                    try:
                        txt = p.read_text(encoding="utf-8", errors="replace")
                        head = "\n".join(txt.splitlines()[:40])
                    except Exception as e:
                        head = f"<failed reading: {type(e).__name__}: {e}>"
                    log.write(f"# FOUND {p} size={p.stat().st_size}\n")
                    log.write("# --- head ---\n")
                    log.write(head + "\n")
                    log.write("# --- end head ---\n")
                else:
                    log.write(f"# MISSING {p}\n")

    def _write_placeholder_metrics(self, run_id: str, reason: str) -> None:
        p = self._metrics_path(run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "area": float("inf"),
            "wns": -100.0,
            "power": float("inf"),
            "__missing__": True,
            "__reason__": reason,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_tail_text(self, p: Path, max_bytes: int = 200_000) -> str:
        """Safe tail-read (avoid huge logs; tolerate partial writes)."""
        try:
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                sz = f.tell()
                f.seek(max(0, sz - int(max_bytes)), os.SEEK_SET)
                b = f.read()
            return b.decode("utf-8", errors="replace")
        except Exception:
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""

    def _last_float(self, text: str, rgx: re.Pattern) -> Optional[float]:
        m_last = None
        for m in rgx.finditer(text):
            m_last = m
        if not m_last:
            return None
        try:
            return float(m_last.group(1))
        except Exception:
            return None

    def _extract_proxy_metrics_from_orfs_log(self, log_file: Path) -> Dict[str, float]:
        """
        Extract academic-style proxies from orfs.log:
        - area proxy: Standard cells area / Placed Cell Area
        - wns proxy: worst slack (heuristic unit convert)
        - congestion proxy: Overflow
        - wirelength proxy: WL (prefer DRWL 'Total wire length', fallback HPWL)
        - util proxy: Utilization
        """
        txt = self._read_tail_text(log_file)

        area = self._last_float(txt, _RE_STD_AREA)
        util = self._last_float(txt, _RE_UTIL)
        slack = self._last_float(txt, _RE_WORST_SL)
        overflow = self._last_float(txt, _RE_OVERFLOW)
        hpwl = self._last_float(txt, _RE_HPWL)

        # NEW: routed WL from DRT if present: "Total wire length = <N> um."
        # You need to define _RE_DRT_WL somewhere (see below).
        drwl = self._last_float(txt, _RE_DRT_WL)

        wns = None
        if slack is not None and math.isfinite(slack):
            # OpenROAD slack is often in seconds (e.g. 1e-11); convert to ns by heuristic.
            # If it's already a "normal" sized number, keep as-is.
            wns = slack * 1e9 if abs(slack) < 1e-3 else slack

        out: Dict[str, float] = {}
        if area is not None and math.isfinite(area):
            out["area"] = float(area)
        if wns is not None and math.isfinite(wns):
            out["wns"] = float(wns)

        # keep these as extra fields for ranking/debug (optional)
        if overflow is not None and math.isfinite(overflow):
            out["proxy_overflow"] = float(overflow)

        # Keep hpwl as extra debug field
        if hpwl is not None and math.isfinite(hpwl):
            out["proxy_hpwl"] = float(hpwl)

        if util is not None and math.isfinite(util):
            out["proxy_util"] = float(util)

        # NEW: unified WL proxy key used by downstream (prefer DRWL, else HPWL)
        wl = None
        if drwl is not None and math.isfinite(drwl):
            wl = float(drwl)
        elif hpwl is not None and math.isfinite(hpwl):
            wl = float(hpwl)

        if wl is not None and math.isfinite(wl):
            out["wl"] = float(wl)

        return out

    def _write_proxy_metrics(self, metrics_path: Path, proxy: Dict[str, Any], reason: str = "proxy") -> None:
        """
        Write proxy metrics to metrics.json, keeping markers for downstream classification.
        """
        out: Dict[str, Any] = {}

        out["area"] = float(proxy.get("area", float("inf")))
        out["wns"] = float(proxy.get("wns", -100.0))
        out["power"] = float(proxy.get("power", float("inf")))
        out["wl"] = float(proxy.get("wl", float("inf")))
        out["__wl_kind__"] = proxy.get("__wl_kind__", "missing")
        out["__wl_src__"] = proxy.get("__wl_src__", None)
        
        for k, v in proxy.items():
            if k in out:
                continue
            out[k] = v

        out["__proxy__"] = True
        out["__missing__"] = True
        out["__reason__"] = reason

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


    async def run_async(
        self,
        params: Dict[str, Any],
        run_id: str,
        script_override: Optional[Any] = None,
        *,
        make_target: Optional[str] = None,
        timeout_sec: int = 7200,
        allow_missing_metrics: bool = False,
    ) -> Tuple[bool, str]:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "orfs.log"

        extra_vars = self._prepare_extra_vars_from_script_override(run_dir, script_override)
        config_mk_path, vars_dict = self._generate_config_mk(run_dir, params, extra_vars=extra_vars)

        yosys_bin = self.cfg.get("tools", {}).get("yosys", "yosys")
        openroad_bin = self.cfg.get("tools", {}).get("openroad", "")

        platform = self.cfg["orfs"]["platform"]
        design = self.cfg["design"]["name"]

        target = (make_target or "").strip()
        make_cmd = "make" if not target else f"make {target}"

        wrapper = run_dir / "run_orfs.sh"
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            f.write("set -euo pipefail\n\n")
            f.write(f"""echo {self._bash_quote(f'Starting ORFS run for {run_id}, target={target or "<default>"}')}\n""")
            f.write(f"export FLOW_VARIANT={self._bash_quote(run_id)}\n")
            f.write(f"export TAG={self._bash_quote(run_id)}\n")
            f.write(f"export DESIGN_CONFIG={self._bash_quote(str(config_mk_path.resolve()))}\n")
            f.write(f"export YOSYS_EXE={self._bash_quote(yosys_bin)}\n")
            if openroad_bin:
                f.write(f"export OPENROAD_EXE={self._bash_quote(openroad_bin)}\n")
            for k, v in extra_vars.items():
                f.write(f"export {k}={self._bash_quote(str(v))}\n")

            # extra safety: export vars into env too
            f.write("\n# Export config vars for Tcl scripts (extra safety)\n")
            for k, v in vars_dict.items():
                f.write(f"export {k}={self._bash_quote(v)}\n")

            f.write(f"\ncd {self._bash_quote(str(self.flow_dir.resolve()))}\n")

            f.write("\n# Thread caps (avoid OpenROAD using 80 threads)\n")
            f.write("export OMP_NUM_THREADS=32\n")
            f.write("export OPENROAD_NUM_THREADS=32\n")
            f.write("export TBB_NUM_THREADS=32\n")
            f.write("export MKL_NUM_THREADS=1\n")
            f.write("export MAKEFLAGS=-j1\n")

            f.write("\n# Clean per-run dirs\n")
            f.write(f"rm -rf results/{platform}/{design}/{run_id} \\\n")
            f.write(f"       reports/{platform}/{design}/{run_id} \\\n")
            f.write(f"       logs/{platform}/{design}/{run_id}\n")

            f.write("\n# Execute Make\n")
            f.write(
                f"{make_cmd} "
                f"FLOW_VARIANT={self._bash_quote(run_id)} "
                f"TAG={self._bash_quote(run_id)} "
                f"DESIGN_CONFIG={self._bash_quote(str(config_mk_path.resolve()))}\n"
            )

        wrapper.chmod(0o755)

        print(f"    [ORFS] {run_id} target={(target if target else '<default>')} timeout={timeout_sec}s")
        make_rc = await self._run_cmd_logged(
            [str(wrapper.resolve())],
            cwd=run_dir,
            log_file=log_file,
            timeout_sec=int(timeout_sec),
        )

        # Don't hard-fail yet; sometimes ORFS can still produce parsable metrics.
        if make_rc != 0:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n# NOTE: make failed rc={make_rc}; will still try to collect metrics.\n")

        # Debug: confirm metrics.log existence and show head snippet
        try:
            self._debug_metrics_log(run_id, log_file)
        except Exception:
            pass

        # metrics generation
        metrics_out = self._metrics_path(run_id)

        # Prefer local lightweight generator (this repo)
        local_candidates = [
            (Path(__file__).resolve().parent / "genMetrics.py"),
            (Path(__file__).resolve().parents[1] / "genMetrics.py"),
        ]
        local_gen = next((p for p in local_candidates if p.exists()), None)

        orfs_gen = (self.flow_dir / "util" / "genMetrics.py")
        if not orfs_gen.exists():
            orfs_gen = None

        with open(log_file, "a", encoding="utf-8") as log:
            log.write("\n\n# --- genMetrics ---\n")
            log.write(f"# local_gen={local_gen}\n")
            log.write(f"# orfs_gen={orfs_gen}\n")
            log.write(f"# expected_out={metrics_out}\n")

        last_rc: int = 127

        # 0) local genMetrics (recommended)
        if local_gen is not None:
            cmd_local = [
                sys.executable,
                str(local_gen),
                "--flow_dir", str(self.flow_dir.resolve()),
                "--platform", platform,
                "--design", design,
                "--variant", run_id,
                "--out", str(Path(metrics_out).resolve()),
                "--run_log", str(Path(log_file).resolve()),
            ]

            self._dbg(f"    [METRICS] try=local_lightweight out={metrics_out}")
            last_rc = await self._run_cmd_logged(
                cmd_local,
                cwd=self.flow_dir,
                log_file=log_file,
                timeout_sec=600,
                append=True,
            )

        # 1) ORFS util/genMetrics as fallback (set env)
        # print(f"[DEBUG] metrics_out.exists()={metrics_out.exists()}")
        if (not metrics_out.exists()) and (orfs_gen is not None):
            env = os.environ.copy()
            if getattr(self, "debug", False):
                env["GENMETRICS_DEBUG"] = "1"
            env.setdefault("PLATFORM_DIR", str((self.flow_dir / "platforms" / platform).resolve()))
            env.setdefault("FLOW_VARIANT", run_id)
            env.setdefault("TAG", run_id)

            cmd_orfs = [
                sys.executable,
                str(orfs_gen),
                "--design", design,
                "--platform", platform,
                "--flowVariant", run_id,
                "--logs", str((self.flow_dir / "logs").resolve()),
                "--reports", str((self.flow_dir / "reports").resolve()),
                "--results", str((self.flow_dir / "results").resolve()),
            ]
            self._dbg(f"    [METRICS] try=orfs_util out={metrics_out}")
            last_rc = await self._run_cmd_logged(
                cmd_orfs,
                cwd=self.flow_dir,
                log_file=log_file,
                timeout_sec=600,
                append=True,
                env=env,
            )

            if not metrics_out.exists():
                _ = self._copy_metrics_from_orfs_outputs(run_id, metrics_out, log_file)

        if not metrics_out.exists():
            if allow_missing_metrics:
                proxy = self._extract_proxy_metrics_from_orfs_log(log_file)
                if proxy.get("area") is not None or proxy.get("wns") is not None:
                    self._write_proxy_metrics(run_id, proxy, "metrics.json missing; used proxies from orfs.log (allowed)")
                else:
                    self._write_placeholder_metrics(run_id, "metrics.json missing after genMetrics (allowed)")
                return True, str(log_file)
            return False, str(log_file)

        # Validate JSON + values
        import math
        try:
            # print(f"[DEBUG] Reading metrics_out={metrics_out}")
            data = json.loads(metrics_out.read_text(encoding="utf-8", errors="replace"))
            area = float(data.get("area", float("inf")))
            power = float(data.get("power", float("inf")))
            wns = float(data.get("wns", -100.0))
            # print(f"[DEBUG] metrics: area={area} power={power} wns={wns}")
            valid = (math.isfinite(area) and math.isfinite(power) and (wns > -99.0))
            # 2) metrics invalid
            if not valid:
                if allow_missing_metrics:
                    proxy = self._extract_proxy_metrics_from_orfs_log(log_file)
                    if proxy.get("area") is not None or proxy.get("wns") is not None:
                        self._write_proxy_metrics(run_id, proxy, "metrics invalid; used proxies from orfs.log (allowed)")
                    else:
                        self._write_placeholder_metrics(run_id, "metrics invalid (allowed)")
                    return True, str(log_file)
                return False, str(log_file)

        # 3) metrics unreadable
        except Exception as e:
            if allow_missing_metrics:
                proxy = self._extract_proxy_metrics_from_orfs_log(log_file)
                if proxy.get("area") is not None or proxy.get("wns") is not None:
                    self._write_proxy_metrics(run_id, proxy, f"metrics.json unreadable; used proxies from orfs.log (allowed)")
                else:
                    self._write_placeholder_metrics(run_id, f"metrics.json unreadable: {type(e).__name__}: {e} (allowed)")
                return True, str(log_file)
            return False, str(log_file)

        # If make failed but metrics valid, accept with warning.
        if make_rc != 0:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n# NOTE: make_rc={make_rc} but metrics are valid; accepting run as success.\n")

        if int(last_rc) != 0:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n# genMetrics rc={int(last_rc)} but metrics are valid; accepting.\n")

        return True, str(log_file)


    def get_metrics(self, log_path: str) -> Dict[str, Any]:
        run_id = Path(log_path).parent.name
        p = self._metrics_path(run_id)

        default_fail: Dict[str, Any] = {
            "area": float("inf"),
            "wns": -100.0,
            "power": float("inf"),
            "wl": float("inf"),
            "clock_period": float("inf"),
            "ecp": float("inf"),
        }

        if not p.exists():
            if getattr(self, "debug", False):
                self._dbg(f"[METRICS-DBG] metrics.json missing for {run_id}: {p}")
                try:
                    self._debug_metrics_log(run_id, log_file=(self._run_dir(run_id) / "orfs.log"))
                except Exception:
                    pass
            return default_fail

        try:
            if getattr(self, "debug", False):
                txt = p.read_text(encoding="utf-8", errors="replace")
                self._dbg(f"[METRICS-DBG] metrics.json path={p} size={p.stat().st_size}\n{txt[:800]}")

            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                return default_fail

            # keep all fields, but normalize numeric keys we care about
            def _f(key: str, fallback: float) -> float:
                try:
                    return float(data.get(key, fallback))
                except Exception:
                    return fallback

            data["area"] = _f("area", float("inf"))
            data["wns"] = _f("wns", -100.0)
            data["power"] = _f("power", float("inf"))
            data["wl"] = _f("wl", float("inf"))
            data["clock_period"] = _f("clock_period", float("inf"))
            data["ecp"] = _f("ecp", float("inf"))

            return data
        except Exception:
            return default_fail


    def cleanup(self, history: List[Dict[str, Any]]) -> None:
        if not history:
            return
        valid = [h for h in history if h.get("metrics", {}).get("area") != float("inf")]
        pareto_ids = set()
        to_delete = set()

        for i, h1 in enumerate(valid):
            dominated = False
            for j, h2 in enumerate(valid):
                if i == j:
                    continue
                if is_pareto_dominated(h1["metrics"], h2["metrics"]):
                    dominated = True
                    break
            if dominated:
                to_delete.add(h1["run_id"])
            else:
                pareto_ids.add(h1["run_id"])

        all_ids = {h["run_id"] for h in history}
        ok_ids = {h["run_id"] for h in valid}
        to_delete.update(all_ids - ok_ids)

        if to_delete:
            print(f"  [Cleanup] deleting {len(to_delete)} runs, keeping pareto={len(pareto_ids)}")
            for rid in to_delete:
                for d in [self._get_results_dir(rid), self._get_reports_dir(rid), self._get_logs_dir(rid)]:
                    if d.exists():
                        shutil.rmtree(d, ignore_errors=True)
                        # pass # for safety, avoid deleting outputs [CHANGE]
                wd = self._run_dir(rid)
                if wd.exists():
                    shutil.rmtree(wd, ignore_errors=True)
                    # pass # for safety, avoid deleting outputs [CHANGE]
    