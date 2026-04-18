# src/cadence_runner.py
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import posixpath

from .eda_interface import EDARunner
from .remote_cadence_support import (
    CadenceRemoteJobRunner,
    CadenceRemoteResilienceConfig,
    CadenceRemoteRunSpec,
    RemoteCommError,
    RemoteSessionLostError,
    RemoteCommandTimeoutError,
    SchedulerBackpressureError,
    ControlledStop,
    RemoteConnection,
    shell_quote,
    format_lsf_wallclock,
    load_credentials,
    slugify,
    _load_checkpoint,
    _save_checkpoint,
)
from ..utils.ranktuner_utils import is_pareto_dominated
from ..utils.utils import prepare_run_dir

_RE_FLOAT = r"[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?"
_ITER_RUN_RE = re.compile(r"^iter_\d+$")


def _read_text_best_effort(path: Path, max_chars: int = 2_000_000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
        if len(txt) <= max_chars:
            return txt
        return txt[-max_chars:]
    except Exception:
        return ""


def _parse_first_float(text: str, patterns: List[re.Pattern]) -> Optional[float]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def _parse_area_from_reports(*report_texts: str) -> Optional[float]:
    pats = [
        # ★ NEW: alloc_area from Innovus placement log — this IS the die area
        # Format: "alloc_area 424278 sites (6186 um^2)"
        re.compile(rf"\balloc_area\b[^(]*\(\s*({_RE_FLOAT})\s*(?:um\^2|um2)\s*\)", re.IGNORECASE),

        # Die area from report_area_die_*.rpt (Tcl-generated)
        re.compile(rf"\bDie\s+area\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bDesign\s+area\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bTotal\s+area\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),

        # ★ DEMOTED: Total cell area is synthesis-constant, only use as last resort
        re.compile(rf"\bTotal\s+cell\s+area\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bcell\s+area\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
    ]
    for txt in report_texts:
        v = _parse_first_float(txt, pats)
        if v is not None:
            return v
    return None


def _detect_time_unit_scale(text: str) -> float:
    """
    Detect the time unit used in an Innovus report and return the
    conversion factor to ps.
    - "Time Units = 1e-12 secs" -> 1.0  (already in ps)
    - "Time Units = 1e-09 secs" -> 1000.0 (ns -> ps)
    - "Time Units = 1ns"        -> 1000.0
    - "WNS (ns)" (QoR inline)   -> 1000.0
    - Default assumption: ps (ASAP7 commonly uses ps)
    """
    # Method 1: "Time Units = 1e-12 secs" header
    m = re.search(r"Time\s+Units?\s*[:=]\s*([\d.eE+\-]+)\s*(?:secs?|seconds?)", text, re.IGNORECASE)
    if m:
        try:
            tu = float(m.group(1))
            # tu means "1 time unit = tu seconds"
            # ps = 1e-12s, so scale = tu / 1e-12
            scale = tu / 1e-12
            return scale
        except Exception:
            pass

    # Method 2: "Time Units = 1ns" / "Time Units = 1ps" header
    m2 = re.search(r"Time\s+Units?\s*[:=]\s*\d+\s*(ps|ns|us|ms|s)\b", text, re.IGNORECASE)
    if m2:
        unit_map = {"ps": 1.0, "ns": 1000.0, "us": 1e6, "ms": 1e9, "s": 1e12}
        return unit_map.get(m2.group(1).lower(), 1.0)

    # Method 3: inline unit markers in QoR reports
    # report_qor may emit lines like "WNS (ns): -2.718" or "Setup WNS (ns): -2.718"
    # These reports may not include a "Time Units" header, but they are emitted in ns
    if re.search(r"\bWNS\s*\(ns\)", text, re.IGNORECASE):
        return 1000.0   # ns → ps
    if re.search(r"\bWNS\s*\(ps\)", text, re.IGNORECASE):
        return 1.0
    if re.search(r"\bTNS\s*\(ns\)", text, re.IGNORECASE):
        return 1000.0   # Likewise, a TNS ns marker implies the file is using ns

    # Method 4: detect Innovus report_qor / summaryReport output
    # Innovus report_qor and summaryReport typically emit timing values (WNS/TNS) in ns,
    # even when the technology library uses ps (for example ASAP7). We distinguish them via the Command header.
    # report_timing emits Slack Time in the library-native unit (often ps), so no conversion is needed.
    if re.search(r"#\s*Command:.*\breport_qor\b", text, re.IGNORECASE):
        return 1000.0  # ns → ps
    if re.search(r"#\s*Command:.*\bsummaryReport\b", text, re.IGNORECASE):
        return 1000.0  # ns → ps

    # Method 5: detect report_qor-style output when no Command header is present
    # report_qor output typically contains "Timing Path Type" and "Clock Group" lines
    if re.search(r"Timing\s+Path\s+Type\s*:", text) and re.search(r"Clock\s+Group\s*:", text):
        return 1000.0  # ns → ps

    return 1.0  # Default to ps (appropriate for report_timing and similar reports)


def _parse_wns_from_reports(*report_texts: str) -> Optional[float]:
    """
    Parse WNS and normalize the return value to ps.
    """
    pats = [
        re.compile(rf"\bWNS\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bWorst\s+Negative\s+Slack\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"=\s*Slack\s+Time\s+({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bSlack\s*\([^)]*\)\s*:\s*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bSetup\s+WNS\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bworst\s+slack\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
    ]

    for txt in report_texts:
        v = _parse_first_float(txt, pats)
        if v is not None:
            # Detect the time unit used in this text
            scale = _detect_time_unit_scale(txt)
            return v * scale   # Normalize to ps

    return None


def _parse_power_from_reports(*report_texts: str) -> Optional[float]:
    pats = [
        re.compile(rf"\bTotal\s+Power\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bTotal\s+power\b[^0-9+\-]*({_RE_FLOAT})", re.IGNORECASE),
        re.compile(rf"\bTotal\b\s*[:=]\s*({_RE_FLOAT})\s*(?:W|mW|uW|nW)\b", re.IGNORECASE),
    ]
    for txt in report_texts:
        v = _parse_first_float(txt, pats)
        if v is not None:
            return v
    return None


class CadenceFlow(EDARunner):
    """
    Remote Cadence Genus + Innovus flow runner.

    Differences from the previous local runner:
      - no local subprocess execution
      - always submits remote jobs through LSF/bjobs
      - always uses password-based SSH login through the jump host
      - each run_id gets an isolated remote work directory to support concurrency
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.work_dir = Path(str(config.get("work_dir", "./runs_cadence"))).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        cad = config.get("cadence", {}) or {}
        remote = cad.get("remote", {}) or {}
        design = config.get("design", {}) or {}
        default_params = config.get("default_params", {}) or {}

        self.remote_cfg = remote
        self.design_cfg = design
        self.default_params = dict(default_params)

        cred_file = remote.get("credentials_file") or remote.get("cred_file") or "credentials.json"
        self.credentials_file = Path(str(cred_file)).expanduser().resolve()

        project_name = str(config.get("project_name") or design.get("name") or "edatuner_cadence")
        self.project_slug = slugify(project_name)
        self.remote_root_base = str(remote.get("remote_root") or remote.get("root") or "/remote/edatuner_runs")
        self.queue = str(remote.get("queue") or "normal")

        self.poll_interval_sec = int(remote.get("poll_interval_sec", 60))
        self.max_poll_attempts = int(remote.get("max_poll_attempts", 120))
        self.cleanup_remote_after_download = bool(remote.get("cleanup_remote_after_download", False))
        self.cleanup_remote_batch = bool(remote.get("cleanup_remote_batch", True))

        self.remote_rtl_dir = str(
            remote.get("rtl_dir")
            or design.get("remote_rtl_dir")
            or self._fallback_remote_rtl_dir(design)
            or "./benchmark/JPEG"
        )
        self.remote_rtl_include_dirs = list(
            remote.get("rtl_include_dirs")
            or design.get("remote_rtl_include_dirs")
            or design.get("verilog_include_dirs")
            or []
        )
        self.remote_plat_dir = str(remote.get("plat_dir") or remote.get("platform_dir") or "./platform/asap7")
        self.remote_sdc_file = str(remote.get("sdc_file") or design.get("remote_sdc") or design.get("sdc") or "")
        self.lib_model = str(remote.get("lib_model") or "NLDM")
        self.lib_corner = str(remote.get("lib_corner") or "RVT_TT")

        self.tool_env_script = str(remote.get("tool_env_script") or "")
        self.genus_bin = str(remote.get("genus_bin") or "")
        self.innovus_bin = str(remote.get("innovus_bin") or "")

        # ── Thread / CPU control ───────────────────────────────────────
        # threads_per_task is the unified default; genus_ncpu / innovus_ncpu
        # override it per-tool if explicitly specified.
        threads_per_task = int(remote.get("threads_per_task", 32))
        self.genus_ncpu = int(remote.get("genus_ncpu", threads_per_task))
        # ── stage-aware Innovus CPU count ──
        # Config may specify innovus_ncpu_quick / innovus_ncpu_full separately.
        # Falls back to innovus_ncpu -> threads_per_task.
        _inno_default = int(remote.get("innovus_ncpu", threads_per_task))
        self.innovus_ncpu_quick = int(remote.get("innovus_ncpu_quick", _inno_default))
        self.innovus_ncpu_full  = int(remote.get("innovus_ncpu_full",  _inno_default))
        # Keep a unified attribute for backward compat (used as full default)
        self.innovus_ncpu = self.innovus_ncpu_full


        self.genus_time = str(remote.get("genus_time") or "")
        self.innovus_time = str(remote.get("innovus_time") or "")

        self.stage_env_var = str(cad.get("stage_env_var") or "RUN_STAGE")
        self.override_env_var = str(cad.get("override_env_var") or "OVERRIDE_TCL")

        self.debug = bool(cad.get("debug", False) or config.get("debug", False))

        # ── Resilience config ─────────────────────────────────────
        self.resilience = CadenceRemoteResilienceConfig(
            stop_on_remote_error=bool(remote.get("stop_on_remote_error", True)),
            resume_on_restart=bool(remote.get("resume_on_restart", True)),
            keepalive_sec=int(remote.get("keepalive_sec", 30)),
            reconnect_retries=int(remote.get("reconnect_retries", 2)),
            reconnect_retry_interval_sec=int(remote.get("reconnect_retry_interval_sec", 5)),
            max_submit_backpressure_wait_sec=int(remote.get("max_submit_backpressure_wait_sec", 1800)),
            submit_retry_interval_sec=int(remote.get("submit_retry_interval_sec", 60)),

            # ---- NEW ----
            lsf_pending_jobs_soft_limit=int(remote.get("lsf_pending_jobs_soft_limit", 24)),
            lsf_running_jobs_soft_limit=int(remote.get("lsf_running_jobs_soft_limit", 32)),
            lsf_running_slots_soft_limit=int(remote.get("lsf_running_slots_soft_limit", 256)),
            lsf_check_interval_sec=int(remote.get("lsf_check_interval_sec", 45)),
            lsf_check_jitter_sec=int(remote.get("lsf_check_jitter_sec", 10)),
        )

        # ---- Innovus-only mode state ----
        _tuning = config.get("tuning", {}) or {}
        self._innovus_only_mode = str(_tuning.get("mode", "full")).strip().lower() == "innovus_only"
        self._shared_genus_dir: Optional[str] = None   # set after initial genus run


    def _fallback_remote_rtl_dir(self, design: Dict[str, Any]) -> Optional[str]:
        verilog = design.get("verilog") or []
        if isinstance(verilog, str):
            verilog = [verilog]
        if not verilog:
            return None

        first = str(verilog[0]).strip()
        if not first:
            return None

        try:
            p = Path(first)
            if "*" in first or "?" in first or "[" in first:
                return str(p.parent) if str(p.parent) else "."
            return str(p.parent) if str(p.parent) else "."
        except Exception:
            return None

    def _run_dir(self, run_id: str) -> Path:
        return self.work_dir / str(run_id)

    def _metrics_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "metrics.json"

    def _log_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "cadence.log"

    def _remote_run_root(self, run_id: str) -> str:
        return "/".join(
            [
                self.remote_root_base.rstrip("/"),
                self.project_slug,
                slugify(run_id),
            ]
        )

    def _normalize_stage(self, make_target: Optional[str]) -> str:
        s = str(make_target or "").strip().lower()
        if not s:
            return "route"
        if s in ("full", "all", "route", "postroute", "drt", "detailed_route"):
            return "route"
        if "place" in s:
            return "place"
        if "cts" in s:
            return "cts"
        return s

    def _stage_defaults(self, stage: str) -> Dict[str, str]:
        out: Dict[str, str] = {
            "TOP": str(self.design_cfg.get("name") or "design"),
            "RTL_DIR": self.remote_rtl_dir,
            "PLAT_DIR": self.remote_plat_dir,
            "LIB_MODEL": self.lib_model,
            "LIB_CORNER": self.lib_corner,
            "REMOTE_SDC_FILE": self.remote_sdc_file,
            self.stage_env_var: stage,
        }
        if self.remote_rtl_include_dirs:
            include_str = " ".join([str(x) for x in self.remote_rtl_include_dirs if str(x).strip()])
            out["RTL_INCLUDE_DIRS"] = include_str
            out["VERILOG_INCLUDE_DIRS"] = include_str
            out["SYNTH_VERILOG_INCLUDE_DIRS"] = include_str
        return out
    
    def _is_cleanup_eligible_run_id(self, run_id: str) -> bool:
        rid = str(run_id or "").strip()
        return bool(_ITER_RUN_RE.fullmatch(rid))

    # ── Innovus params that require enum true/false (not 0/1) ──────
    _INNOVUS_BOOL_PARAMS = frozenset({
        "PLACE_UNIFORM_DENSITY",
        "PLACE_IO_PINS_AWARE",
        "ROUTE_TIMING_DRIVEN",
        "ROUTE_SI_DRIVEN",
    })

    @staticmethod
    def _to_tcl_bool(value) -> str:
        """Convert Python bool / int / str to Tcl enum 'true' or 'false'.

        Handles: True/False, 1/0, "1"/"0", "true"/"false", "yes"/"no", "on"/"off".
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return "true" if value else "false"
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return "true"
        if s in ("0", "false", "no", "off"):
            return "false"
        # Unknown → keep original and let Innovus report the error explicitly
        return str(value)

    def _canonicalize_params(self, params: Dict[str, Any], *, stage: str) -> Dict[str, str]:
        merged: Dict[str, Any] = dict(self.default_params)
        merged.update(params or {})

        alias_map = {
            "util": "UTIL",
            "margin": "MARGIN",
            "td_place": "TD_PLACE",
            "run_cts": "RUN_CTS",
        }

        env = self._stage_defaults(stage)
        for key, value in merged.items():
            k = str(key)
            # ★ FIX IMPTCM-23: Innovus enum params must be "true"/"false"
            if k in self._INNOVUS_BOOL_PARAMS:
                env[k] = self._to_tcl_bool(value)
            else:
                env[k] = str(value)
            alias = alias_map.get(k.lower())
            if alias:
                env[alias] = env[k]

        if "RUN_CTS" not in env:
            env["RUN_CTS"] = "1" if stage in ("cts", "route") else "0"

        return {str(k): str(v) for k, v in env.items()}

    def _build_remote_spec(
        self,
        *,
        run_id: str,
        params: Dict[str, Any],
        stage: str,
        timeout_sec: int,
    ) -> CadenceRemoteRunSpec:
        # _env = self._canonicalize_params(params, stage=stage)
        # _env_rs = _env.get("RUN_STAGE", "<MISSING>")
        # print(f"[DIAG-2] run_id={run_id} stage={stage!r} env_RUN_STAGE={_env_rs!r} "
        #       f"RUN_STAGE_in_params={'RUN_STAGE' in (params or {})} "
        #       f"RUN_STAGE_in_defaults={'RUN_STAGE' in self.default_params}")
        _skip_genus = bool(self._innovus_only_mode and self._shared_genus_dir)
        return CadenceRemoteRunSpec(
            run_id=str(run_id),
            project_name=str(self.cfg.get("project_name") or self.design_cfg.get("name") or "edatuner_cadence"),
            top=str(self.design_cfg.get("name") or "design"),
            stage=str(stage),
            remote_root=self._remote_run_root(run_id),
            queue=self.queue,
            genus_ncpu=self.genus_ncpu,
            innovus_ncpu=self.innovus_ncpu_quick if stage in ("place", "cts") else self.innovus_ncpu_full,
            genus_time=format_lsf_wallclock(timeout_sec // 2 if timeout_sec > 0 else None, explicit_value=self.genus_time, fallback="00:30"),
            innovus_time=format_lsf_wallclock(timeout_sec if timeout_sec > 0 else None, explicit_value=self.innovus_time, fallback="01:00"),
            poll_interval_sec=self.poll_interval_sec,
            max_poll_attempts=self.max_poll_attempts,
            cleanup_remote_after_download=self.cleanup_remote_after_download,
            tool_env_script=self.tool_env_script,
            genus_bin=self.genus_bin,
            innovus_bin=self.innovus_bin,
            remote_sdc=self.design_cfg.get("sdc", "") or "",
            env=self._canonicalize_params(params, stage=stage),
            skip_genus=_skip_genus,
            shared_genus_dir=str(self._shared_genus_dir or ""),
        )


    def _extract_override_text(self, script_override: Optional[Any]) -> Optional[str]:
        if script_override is None:
            return None
        if isinstance(script_override, str):
            return script_override
        if isinstance(script_override, dict):
            tcl = script_override.get("tcl")
            if isinstance(tcl, str):
                return tcl
        return str(script_override)

    @staticmethod
    def _detect_actual_stage(innovus_root: Path) -> str:
        """Detect the actual completed Innovus stage based on report file presence."""
        for stage in ["route", "cts", "prects", "place"]:
            if (innovus_root / f"report_timing_{stage}.rpt").exists():
                return stage
        return "unknown"

    def _collect_report_texts(self, root: Path) -> Dict[str, List[str]]:
        cadence_root = root / "outbox" / "cadence"
        genus_root = cadence_root / "genus"
        innovus_root = cadence_root / "innovus"
        logs_root = cadence_root / "logs"

        def _stage_priority(path: Path) -> int:
            """Sort by stage priority: route first (0), earlier stages later.
            _parse_wns_from_reports uses first-match-wins semantics, so route must come first
            to ensure we pick the final-stage data (route > cts > prects > place > post_init)."""
            name = path.name.lower()
            if "route" in name:
                return 0
            if "cts" in name and "prects" not in name:
                return 1
            if "prects" in name:
                return 2
            if "place" in name:
                return 3
            if "post_init" in name:
                return 4
            # Files without an explicit stage name (for example report_area_final.rpt) go last
            return 5

        def read_many(base_dir: Path, patterns: List[str]) -> List[str]:
            texts: List[str] = []
            if not base_dir.is_dir():
                return texts
            for pattern in patterns:
                # Sort by stage priority instead of alphabetical order
                # This ensures report_qor_route.rpt is processed before report_qor_cts.rpt
                for p in sorted(base_dir.glob(pattern), key=_stage_priority):
                    if p.is_file():
                        txt = _read_text_best_effort(p)
                        if txt:
                            texts.append(txt)
            return texts

        genus_texts = read_many(genus_root, [
            "report_area*.rpt", "report_power*.rpt",
            "report_qor*.rpt", "report_timing*.rpt",
        ])

        innovus_texts = read_many(innovus_root, [
            "report_area*.rpt", "report_power*.rpt",
            "report_qor*.rpt", "report_timing*.rpt",
            "report_route*.rpt", "summary*.rpt",
            "timing_*/*.rpt",
        ])

        # Keep Innovus logs separate; they are only used for area extraction
        innovus_log_texts = []
        if logs_root.is_dir():
            for p in sorted(logs_root.glob("innovus*.log")):
                if p.is_file():
                    txt = _read_text_best_effort(p, max_chars=500_000)
                    if txt:
                        innovus_log_texts.append(txt)

        return {
            "genus": genus_texts,
            "innovus": innovus_texts,           # Report files
            "innovus_log": innovus_log_texts,    # Logs kept separate
        }


    def _extract_metrics_from_artifacts(
        self,
        artifact_root: Path,
        *,
        expected_stage: str = "route",
    ) -> Dict[str, Any]:
        """Extract metrics from downloaded reports.

        When expected_stage="route", only route-stage reports are used for WNS/Power.
        If route reports are missing, mark sentinel values and record the reason.
        """
        innovus_root = artifact_root / "outbox" / "cadence" / "innovus"
        actual_stage = self._detect_actual_stage(innovus_root)

        reports = self._collect_report_texts(artifact_root)
        genus_txt = reports["genus"]
        innovus_txt = reports["innovus"]
        innovus_log_txt = reports.get("innovus_log", [])

        # Area: search logs (alloc_area) + reports + Genus output (area is stage-independent)
        area = _parse_area_from_reports(*(innovus_log_txt + innovus_txt + genus_txt))

        # WNS / Power: use report files only
        wns = _parse_wns_from_reports(*(innovus_txt + genus_txt))
        power = _parse_power_from_reports(*(innovus_txt + genus_txt))

        if area is None:
            print(f"[ERROR] Failed to parse area from reports under {artifact_root}")
        if wns is None:
            print(f"[ERROR] Failed to parse WNS from reports under {artifact_root}")
        if power is None:
            print(f"[ERROR] Failed to parse power from reports under {artifact_root}")

        result: Dict[str, Any] = {}
        result["area"] = float(area) if area is not None else float("inf")
        result["wns"] = float(wns) if wns is not None else -100.0
        result["power"] = float(power) if power is not None else float("inf")

        if area is None:
            result["__area_sentinel__"] = True
        if wns is None:
            result["__wns_sentinel__"] = True
        if power is None:
            result["__power_sentinel__"] = True

        # Stage metadata
        result["__expected_stage__"] = expected_stage
        result["__actual_stage__"] = actual_stage

        # Core fix: expected route but route did not complete -> force sentinel values
        if expected_stage == "route" and actual_stage != "route":
            error_reason = (
                f"Stage mismatch: expected route but Innovus only completed "
                f"'{actual_stage}'. Possible causes: LSF timeout, OOM, crash, "
                f"or RUN_STAGE env var was incorrectly set."
            )
            result["__wns_sentinel__"] = True
            result["__power_sentinel__"] = True
            result["wns"] = -100.0
            result["power"] = float("inf")
            result["__stage_mismatch__"] = True
            result["__stage_mismatch_reason__"] = error_reason
            print(f"[WARN] {error_reason}")

        return result


    def _write_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        path = self._metrics_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    def set_shared_genus_dir(self, remote_genus_dir: str) -> None:
        """Set the shared genus output directory for innovus-only mode."""
        self._shared_genus_dir = str(remote_genus_dir)
        print(f"[INNOVUS-ONLY] shared genus dir set: {self._shared_genus_dir}")

    def run_initial_genus(self, params: Dict[str, Any], run_id: str = "genus_shared",
                          timeout_sec: int = 3600) -> Tuple[bool, str, str]:
        """
        Run Genus once and return (success, log_path, remote_genus_dir).
        After success, call set_shared_genus_dir(remote_genus_dir) so that
        subsequent runs skip Genus and reuse this output.
        """
        stage = "route"  # genus needs to produce full netlist
        log_path = self._log_path(run_id)

        remote_root = self._remote_run_root(run_id)
        remote_genus_dir = posixpath.join(remote_root, "outbox/cadence/genus")

        # ---- Fast path: check BEFORE prepare_run_dir (which may wipe the directory) ----
        local_genus_dir = Path(self.work_dir) / str(run_id) / "remote_artifacts" / "outbox" / "cadence" / "genus"
        if local_genus_dir.is_dir() and any(f.name.endswith(".rpt") for f in local_genus_dir.iterdir()):
            print(f"[INNOVUS-ONLY] Genus output already exists locally: {local_genus_dir}")
            self.set_shared_genus_dir(remote_genus_dir)
            return True, str(log_path), remote_genus_dir

        run_dir = prepare_run_dir(self.work_dir, str(run_id))

        # ---- Normal path: run the full Genus→Innovus flow ----
        creds = load_credentials(str(self.credentials_file))
        spec = CadenceRemoteRunSpec(
            run_id=str(run_id),
            project_name=str(self.cfg.get("project_name") or self.design_cfg.get("name") or "edatuner_cadence"),
            top=str(self.design_cfg.get("name") or "design"),
            stage=str(stage),
            remote_root=remote_root,
            queue=self.queue,
            genus_ncpu=self.genus_ncpu,
            innovus_ncpu=self.innovus_ncpu_full,
            genus_time=format_lsf_wallclock(timeout_sec if timeout_sec > 0 else None,
                                            explicit_value=self.genus_time, fallback="02:00"),
            innovus_time=format_lsf_wallclock(timeout_sec if timeout_sec > 0 else None,
                                              explicit_value=self.innovus_time, fallback="01:00"),
            poll_interval_sec=self.poll_interval_sec,
            max_poll_attempts=self.max_poll_attempts,
            cleanup_remote_after_download=self.cleanup_remote_after_download,
            tool_env_script=self.tool_env_script,
            genus_bin=self.genus_bin,
            innovus_bin=self.innovus_bin,
            remote_sdc=self.design_cfg.get("sdc", "") or "",
            env=self._canonicalize_params(params, stage=stage),
            skip_genus=False,
            shared_genus_dir="",
            genus_only=True,
        )

        runner = CadenceRemoteJobRunner(spec, creds, resilience=self.resilience)
        remote_result = runner.run(run_dir)

        local_gd = run_dir / "remote_artifacts" / "outbox" / "cadence" / "genus"
        genus_ok = local_gd.is_dir() and any(f.name.endswith(".rpt") for f in local_gd.iterdir())

        if genus_ok:
            self.set_shared_genus_dir(remote_genus_dir)

        # Extract metrics (best-effort warmup point)
        try:
            metrics = self._extract_metrics_from_artifacts(
                run_dir / "remote_artifacts", expected_stage=stage,
            )
            self._write_metrics(run_id, metrics)
        except Exception as e:
            print(f"[INNOVUS-ONLY] Metrics extraction failed (genus ok={genus_ok}): {e}")

        return genus_ok, str(log_path), remote_genus_dir



    def run(
        self,
        params: Dict[str, Any],
        run_id: str,
        script_override: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        make_target = kwargs.get("make_target")
        timeout_sec = int(kwargs.get("timeout_sec", 3600))
        allow_missing_metrics = bool(kwargs.get("allow_missing_metrics", False))
        stage = self._normalize_stage(make_target)
        # print(f"[DIAG-1] run_id={run_id} make_target={make_target!r} → stage={stage!r}")

        # Resume-safe: keep existing run_dir if checkpoint exists
        run_dir = self._run_dir(run_id)
        ckpt_path = run_dir / "checkpoint.json"
        if run_dir.exists() and ckpt_path.exists() and self.resilience.resume_on_restart:
            pass  # preserve existing dir for resume
        else:
            run_dir = prepare_run_dir(self.work_dir, str(run_id))
        log_path = self._log_path(run_id)

        creds = load_credentials(str(self.credentials_file))
        spec = self._build_remote_spec(run_id=run_id, params=params, stage=stage, timeout_sec=timeout_sec)
        runner = CadenceRemoteJobRunner(spec, creds, resilience=self.resilience)

        remote_result = runner.run(run_dir, script_override=self._extract_override_text(script_override))
        metrics = self._extract_metrics_from_artifacts(
            run_dir / "remote_artifacts",
            expected_stage=stage,
        )

        metrics["_meta"] = {
            "stage": stage,
            "remote_root": remote_result.remote_root,
            "ok_submit": bool(remote_result.ok_submit),
            "genus_jobid": remote_result.genus_jobid,
            "innovus_jobid": remote_result.innovus_jobid,
            "genus_status": remote_result.genus_status,
            "innovus_status": remote_result.innovus_status,
            "error": remote_result.error,
        }
        self._write_metrics(run_id, metrics)

        jobs_ok = (
            bool(remote_result.ok_submit)
            and str(remote_result.genus_status or "").upper() == "DONE"
            and str(remote_result.innovus_status or "").upper() == "DONE"
        )
        # ★ FIX: use sentinel markers (from _extract_metrics_from_artifacts) when available;
        #   fall back to value-based check for backward compatibility.
        #   Previous: wns > -99.0 — this rejected legitimate place-stage WNS like -35000.
        #   Now: only reject if it's the exact sentinel value OR marked as sentinel.
        _area = float(metrics.get("area", float("inf")))
        _wns = float(metrics.get("wns", -100.0))
        _power = float(metrics.get("power", float("inf")))

        _area_ok = math.isfinite(_area) and not metrics.get("__area_sentinel__", False)
        _wns_ok = math.isfinite(_wns) and (_wns != -100.0) and not metrics.get("__wns_sentinel__", False)
        _power_ok = math.isfinite(_power) and not metrics.get("__power_sentinel__", False)

        core_ok = _area_ok and _wns_ok
        full_ok = core_ok and _power_ok

        # Reject on stage mismatch
        if metrics.get("__stage_mismatch__"):
            with open(log_path, "a", encoding="utf-8", errors="replace") as fp:
                fp.write(
                    f"\\n# REJECTED: {metrics.get('__stage_mismatch_reason__', 'stage mismatch')}\\n"
                )
            return False, str(log_path)

        if jobs_ok:
            if full_ok or (allow_missing_metrics and core_ok):
                return True, str(log_path)
            return False, str(log_path)

        if full_ok or (allow_missing_metrics and core_ok):
            with open(log_path, "a", encoding="utf-8", errors="replace") as fp:
                fp.write(
                    "\\n# NOTE: remote job status was not fully DONE, "
                    "but usable metrics were recovered from downloaded reports; accepting run.\\n"
                )
            return True, str(log_path)

        return False, str(log_path)

    def get_metrics(self, log_path: str) -> Dict[str, Any]:
        run_id = Path(str(log_path)).parent.name
        metrics_path = self._metrics_path(run_id)
        if metrics_path.exists():
            try:
                return json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        # ★ FIX: fallback is explicitly invalid — no more Inf/-100 sentinels
        return {"__valid__": False, "__missing__": True, "__reason__": "metrics_file_not_found"}

    def cleanup(self, history: List[Dict[str, Any]]) -> None:
        """
        Clean up Pareto-dominated run directories (local + remote).

        Logic:
          1. Find all Pareto-dominated run_ids outside the recent keep_last window.
          2. Remove the matching local run_dir entries.
          3. If cleanup_remote_batch is enabled, batch-delete remote directories over SSH.
        """
        try:
            keep_last = 10
            if len(history) <= keep_last:
                return

            valid = [h for h in history if isinstance(h.get("metrics"), dict)]
            valid = [h for h in valid if float(h["metrics"].get("area", float("inf"))) != float("inf")]
            if len(valid) < 2:
                return

            dominated_ids: List[str] = []
            for i in range(len(valid)):
                for j in range(len(valid)):
                    if i == j:
                        continue
                    if is_pareto_dominated(valid[i]["metrics"], valid[j]["metrics"]):
                        rid = str(valid[i].get("run_id") or "")
                        if rid:
                            dominated_ids.append(rid)
                        break

            recent_ids = set([str(h.get("run_id")) for h in history[-keep_last:]])

            # Collect run_ids to clean up
            to_cleanup_ids: List[str] = []
            for rid in dominated_ids:
                rid = str(rid or "").strip()
                if rid in recent_ids:
                    continue
                if not self._is_cleanup_eligible_run_id(rid):
                    continue
                to_cleanup_ids.append(rid)


            if not to_cleanup_ids:
                return

            # Local cleanup (same behavior as before)
            local_deleted = 0
            for rid in to_cleanup_ids:
                run_dir = self._run_dir(rid)
                if run_dir.exists() and run_dir.is_dir():
                    shutil.rmtree(run_dir, ignore_errors=True)
                    local_deleted += 1

            # Remote cleanup
            remote_deleted = 0
            if getattr(self, "cleanup_remote_batch", True):
                remote_deleted = self._cleanup_remote_dirs(to_cleanup_ids)

            if local_deleted > 0 or remote_deleted > 0:
                print(f"  [Cleanup] local={local_deleted} remote={remote_deleted} dominated runs cleaned")

        except Exception as exc:
            try:
                print(f"  [Cleanup][WARN] cleanup error (ignored): {exc}")
            except Exception:
                pass
            return

    def _cleanup_remote_dirs(self, run_ids: List[str]) -> int:
        """
        Batch-delete Pareto-dominated run directories on the remote server over SSH.

        Path construction reuses self._remote_run_root(rid).
        To avoid repeatedly opening SSH sessions, one connection deletes all target directories.

        Returns:
            Number of remote directories successfully deleted.
        """
        if not run_ids:
            return 0

        # Collect remote paths
        remote_paths: List[str] = []
        for rid in run_ids:
            try:
                rpath = self._remote_run_root(rid)
                if rpath and rpath.strip():
                    remote_paths.append(rpath)
            except Exception:
                continue

        if not remote_paths:
            return 0

        # Load credentials
        try:
            creds = load_credentials(str(self.credentials_file))
        except Exception as exc:
            print(f"  [Cleanup][WARN] cannot load credentials for remote cleanup: {exc}")
            return 0

        # Establish the SSH connection
        conn = RemoteConnection(
            creds,
            keepalive_sec=getattr(self.resilience, "keepalive_sec", 30),
            reconnect_retries=getattr(self.resilience, "reconnect_retries", 2),
            reconnect_retry_interval_sec=getattr(self.resilience, "reconnect_retry_interval_sec", 5),
        )
        total_deleted = 0
        try:
            conn.connect()

            # Build rm -rf commands in batches (up to 20 directories each to avoid long command lines)
            batch_size = 20
            for i in range(0, len(remote_paths), batch_size):
                batch = remote_paths[i : i + batch_size]
                quoted_paths = " ".join(shell_quote(p) for p in batch)
                cmd = f"rm -rf -- {quoted_paths}"
                try:
                    rc, _out, err = conn.exec_cmd_resilient(cmd, timeout=120)
                    if rc == 0:
                        total_deleted += len(batch)
                    else:
                        print(f"  [Cleanup][WARN] remote rm batch failed: rc={rc} err={err[:200]}")
                except Exception as cmd_exc:
                    print(f"  [Cleanup][WARN] remote rm command error: {cmd_exc}")
                    break  # Exit early if the SSH session is lost

        except Exception as exc:
            print(f"  [Cleanup][WARN] remote cleanup SSH error (ignored): {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

        return total_deleted

