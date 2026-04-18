# src/remote_cadence_support.py
from __future__ import annotations

import json
import posixpath
import shlex
import stat
import textwrap
import time
import random
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import paramiko
except ImportError as exc:  # pragma: no cover
    raise ImportError("paramiko is required for remote Cadence execution") from exc



# ============================================================
# Custom exception types
# ============================================================

class RemoteCommError(Exception):
    """Initial connection failure, network unreachable, DNS, SSH handshake."""

class RemoteSessionLostError(Exception):
    """SSH transport/session became inactive after connection."""

class RemoteCommandTimeoutError(Exception):
    """Remote command did not return exit status within timeout (command slow, NOT session lost)."""

class SchedulerBackpressureError(Exception):
    """LSF 'Pending job threshold reached' and wait exceeded max."""

class ControlledStop(Exception):
    """System performing controlled stop (checkpoint saved)."""

# ============================================================
# Checkpoint helpers
# ============================================================

def _save_checkpoint(ckpt_path, data):
    data["timestamp"] = time.time()
    data["timestamp_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = Path(str(ckpt_path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(ckpt_path)
    # print(f"[CHECKPOINT] saved: {ckpt_path} phase={data.get('current_phase', '?')}")

def _load_checkpoint(ckpt_path):
    p = Path(ckpt_path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

# ============================================================
# Session-loss / network-error detection
# ============================================================

_SESSION_LOST_MARKERS = (
    "ssh session not active", "socket is closed", "no existing session",
    "channel closed", "transport is not active",
    # Windows Chinese messages
    "10054", "远程主机强迫关闭", "现有的连接",
    "an existing connection was forcibly closed",
)
_NETWORK_UNREACHABLE_MARKERS = (
    "winerror 10051", "no route to host", "network is unreachable",
    "connection refused", "connection timed out", "connection reset",
    "name or service not known", "temporary failure in name resolution",
    "由于目标计算机积极拒绝", "连接尝试失败",
)
_CHANNEL_EXHAUSTION_MARKERS = (
    "administratively prohibited", "channelexception",
    "too many sessions", "too many channels",
)

_BACKPRESSURE_MARKER = "pending job threshold reached"

def _is_session_lost(exc):
    msg = str(exc).lower()
    return any(m in msg for m in _SESSION_LOST_MARKERS)

def _is_network_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in _NETWORK_UNREACHABLE_MARKERS)

def _is_channel_exhaustion(exc):
    msg = str(exc).lower()
    return any(m in msg for m in _CHANNEL_EXHAUSTION_MARKERS)

# ============================================================
# Global backpressure signaling (shared across threads)
# ============================================================
# When ANY worker thread detects LSF backpressure, it sets this event.
# The main scheduling loop checks it before launching new runs.
# Cleared when a submit eventually succeeds.
_global_backpressure = threading.Event()   # is_set() == True means "LSF is overloaded"

# Throttle SSH connection setup: limit how many threads perform the SSH handshake at once to avoid overloading the jump host
# With 24-32 concurrent workers, keeping only a few concurrent handshakes avoids "Error reading SSH protocol banner"
_connect_semaphore = threading.Semaphore(6)

def is_backpressure_active() -> bool:
    return _global_backpressure.is_set()

def signal_backpressure():
    if not _global_backpressure.is_set():
        print("[BACKPRESSURE] signaling: LSF queue overloaded, pausing new launches")
    _global_backpressure.set()

def clear_backpressure():
    if _global_backpressure.is_set():
        print("[BACKPRESSURE] cleared: LSF queue accepting jobs again")
    _global_backpressure.clear()


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def format_lsf_wallclock(
    timeout_sec: Optional[int],
    *,
    explicit_value: Optional[str] = None,
    fallback: str = "01:00",
) -> str:
    if explicit_value:
        s = str(explicit_value).strip()
        if s:
            return s

    if timeout_sec is None:
        return fallback

    try:
        sec = int(timeout_sec)
    except Exception:
        return fallback

    if sec <= 0:
        return fallback

    total_minutes = max(1, (sec + 59) // 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def slugify(value: str) -> str:
    import re

    s = str(value or "").strip()
    if not s:
        return "edatuner"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._-") or "edatuner"


def load_credentials(cred_file: str) -> Dict[str, str]:
    path = Path(str(cred_file)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"credentials file not found: {path}. "
            "Create a JSON file with jump_host/jump_user/jump_pass/target_host/target_user/target_pass."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "jump_host",
        "jump_user",
        "jump_pass",
        "target_host",
        "target_user",
        "target_pass",
    )
    missing = [k for k in required if k not in data or str(data.get(k, "")).strip() == ""]
    if missing:
        raise ValueError(f"credentials file missing required keys: {', '.join(missing)}")
    return {k: str(data[k]) for k in required}


class RemoteConnection:
    """SSH connection manager using password auth via jump host."""

    def __init__(self, creds: Dict[str, str], *, timeout_sec: int = 30,
                 keepalive_sec: int = 30, reconnect_retries: int = 2,
                 reconnect_retry_interval_sec: int = 5):
        self.creds = dict(creds)
        self.timeout_sec = int(timeout_sec)
        self.keepalive_sec = int(keepalive_sec)
        self.reconnect_retries = int(reconnect_retries)
        self.reconnect_retry_interval_sec = int(reconnect_retry_interval_sec)
        self.jump_client: Optional[paramiko.SSHClient] = None
        self.target_client: Optional[paramiko.SSHClient] = None

    def connect(self) -> None:
        self.jump_client = paramiko.SSHClient()
        self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.jump_client.connect(
            hostname=self.creds["jump_host"],
            username=self.creds["jump_user"],
            password=self.creds["jump_pass"],
            timeout=self.timeout_sec,
            allow_agent=False,
            look_for_keys=False,
        )

        jump_transport = self.jump_client.get_transport()
        if jump_transport is None:
            raise RemoteCommError("failed to create SSH transport to jump host")

        channel = jump_transport.open_channel(
            "direct-tcpip",
            (self.creds["target_host"], 22),
            ("127.0.0.1", 0),
        )

        self.target_client = paramiko.SSHClient()
        self.target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.target_client.connect(
            hostname=self.creds["target_host"],
            username=self.creds["target_user"],
            password=self.creds["target_pass"],
            sock=channel,
            timeout=self.timeout_sec,
            allow_agent=False,
            look_for_keys=False,
        )

        try:
            self.jump_client.get_transport().set_keepalive(self.keepalive_sec)  # type: ignore[union-attr]
            self.target_client.get_transport().set_keepalive(self.keepalive_sec)  # type: ignore[union-attr]
        except Exception:
            pass

    def close(self) -> None:
        if self.target_client is not None:
            try:
                self.target_client.close()
            except Exception:
                pass
            self.target_client = None

        if self.jump_client is not None:
            try:
                self.jump_client.close()
            except Exception:
                pass
            self.jump_client = None


    # ---- Health check ----
    def is_alive(self) -> bool:
        try:
            if self.jump_client is None or self.target_client is None:
                return False
            jt = self.jump_client.get_transport()
            tt = self.target_client.get_transport()
            if jt is None or tt is None:
                return False
            return jt.is_active() and tt.is_active()
        except Exception:
            return False

    # ---- Reconnect ----
    def reconnect(self) -> None:
        print("[WARN] SSH session lost, attempting reconnect ...")
        self.close()
        try:
            self.connect()
            print("[INFO] reconnect succeeded")
        except Exception as exc:
            raise RemoteCommError(f"reconnect failed: {exc}") from exc

    # ---- Resilient exec_cmd ----
    def exec_cmd_resilient(self, cmd: str, *, timeout: Optional[int] = 120,
                           retry_on_session_loss: int = 2) -> Tuple[int, str, str]:
        for attempt in range(1 + retry_on_session_loss):
            # Pre-flight: check session alive
            if not self.is_alive():
                if attempt > 0 and attempt >= retry_on_session_loss:
                    raise RemoteSessionLostError("SSH session not active after all reconnect attempts")
                # Exponential backoff for reconnects: 5s -> 10s -> 20s -> 40s -> ...
                base_interval = self.reconnect_retry_interval_sec
                retries_left = self.reconnect_retries
                retry_idx = 0
                while retries_left > 0:
                    wait_sec = min(base_interval * (2 ** retry_idx), 120) + random.randint(0, 10)
                    try:
                        self.reconnect()
                        break
                    except RemoteCommError:
                        retries_left -= 1
                        retry_idx += 1
                        if retries_left <= 0:
                            raise RemoteSessionLostError("unable to restore SSH session")
                        print(f"[WARN] reconnect failed, retrying in {wait_sec}s ({retries_left} left) ...")
                        time.sleep(wait_sec)
            try:
                return self.exec_cmd(cmd, timeout=timeout)
            except RemoteCommandTimeoutError:
                raise  # command is slow, NOT session lost - don't reconnect
            except paramiko.ssh_exception.SSHException as exc:
                if _is_channel_exhaustion(exc):
                    # SSH server refusing new channels - wait and retry
                    wait_sec = 10 + random.randint(0, 20)
                    print(f"[WARN] SSH channel exhaustion ({exc}), waiting {wait_sec}s ...")
                    time.sleep(wait_sec)
                    try:
                        self.reconnect()
                    except Exception:
                        pass
                    continue
                if _is_session_lost(exc):
                    label = f"({attempt + 1}/{1 + retry_on_session_loss})"
                    print(f"[WARN] SSH session lost during exec_cmd, trying reconnect {label} ...")
                    try:
                        self.reconnect()
                    except Exception as re_exc:
                        raise RemoteSessionLostError(f"session lost and reconnect failed: {re_exc}") from exc
                    continue
                if _is_network_error(exc):
                    raise RemoteCommError(f"network error: {exc}") from exc
                raise RemoteSessionLostError(f"SSH error: {exc}") from exc
            except (OSError, IOError, EOFError) as exc:
                if _is_network_error(exc):
                    raise RemoteCommError(f"network error: {exc}") from exc
                raise RemoteSessionLostError(f"connection error: {exc}") from exc
        raise RemoteSessionLostError("exec_cmd_resilient: retries exhausted")

    def _require_target(self) -> paramiko.SSHClient:
        if self.target_client is None:
            raise RuntimeError("target SSH connection is not established")
        return self.target_client

    def exec_cmd(self, cmd: str, *, timeout: Optional[int] = 120) -> Tuple[int, str, str]:
        client = self._require_target()
        effective_timeout = timeout or 300
        stdin_, stdout_, stderr_ = client.exec_command(cmd, timeout=effective_timeout)
        channel = stdout_.channel
        # Wait for exit status WITH timeout (paramiko default blocks forever)
        if not channel.status_event.wait(timeout=effective_timeout):
            # Command timed out - try to collect partial output
            partial_out = ""
            partial_err = ""
            try:
                channel.settimeout(2)
                if channel.recv_ready():
                    partial_out = channel.recv(65536).decode("utf-8", errors="replace")
                if channel.recv_stderr_ready():
                    partial_err = channel.recv_stderr(65536).decode("utf-8", errors="replace")
            except Exception:
                pass
            channel.close()
            raise RemoteCommandTimeoutError(
                f"remote command timed out after {effective_timeout}s (no exit status). "
                f"cmd: {cmd[:200]}. partial_stdout: {partial_out[:200]}. partial_stderr: {partial_err[:500]}"
            )
        exit_code = channel.recv_exit_status()
        # Add a timeout while reading stdout/stderr to prevent a half-dead channel from blocking forever
        channel.settimeout(120)   # Read timeout: 120 seconds
        try:
            out = stdout_.read().decode("utf-8", errors="replace")
            err = stderr_.read().decode("utf-8", errors="replace")
        except Exception as e:
            # If reads stall, close the channel proactively and raise a retryable exception
            channel.close()
            raise RemoteCommandTimeoutError(
                f"stdout/stderr read stalled after exit_status: {e}"
            ) from e
        return int(exit_code), out, err

    def ensure_remote_dir(self, remote_dir: str) -> None:
        rc, _out, err = self.exec_cmd_resilient(f"mkdir -p -- {shell_quote(remote_dir)}", timeout=60)
        if rc != 0:
            raise RuntimeError(f"failed to create remote dir {remote_dir}: {err.strip()}")

    def put_content(self, content: str, remote_path: str) -> None:
        client = self._require_target()
        parent = posixpath.dirname(str(remote_path))
        if parent:
            self.ensure_remote_dir(parent)

        sftp = client.open_sftp()
        try:
            with sftp.file(str(remote_path), "w") as fp:
                fp.write(content)
        finally:
            sftp.close()

    def get_file(self, remote_path: str, local_path: str) -> None:
        client = self._require_target()
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)

        sftp = client.open_sftp()
        try:
            sftp.get(str(remote_path), str(local))
        finally:
            sftp.close()

    def path_exists(self, remote_path: str) -> bool:
        client = self._require_target()
        sftp = client.open_sftp()
        try:
            sftp.stat(str(remote_path))
            return True
        except FileNotFoundError:
            return False
        finally:
            sftp.close()

    def get_dir_recursive(self, remote_dir: str, local_dir: str) -> None:
        client = self._require_target()
        sftp = client.open_sftp()
        try:
            self._get_dir_recursive_impl(sftp, str(remote_dir), Path(local_dir))
        finally:
            sftp.close()

    def _get_dir_recursive_impl(self, sftp: Any, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for item in sftp.listdir_attr(remote_dir):
            remote_path = posixpath.join(remote_dir, item.filename)
            local_path = local_dir / item.filename
            if stat.S_ISDIR(item.st_mode):
                self._get_dir_recursive_impl(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, str(local_path))


def _csh_quote(value: Any) -> str:
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _env_or_default_tcl_proc() -> str:
    return textwrap.dedent(
        """\
        proc env_or_default {name default_value} {
          if {[info exists ::env($name)] && $::env($name) ne ""} {
            return $::env($name)
          }
          return $default_value
        }
        """
    )


@dataclass
class CadenceRemoteRunSpec:
    run_id: str
    project_name: str
    top: str
    stage: str
    remote_root: str
    queue: str
    genus_ncpu: int
    innovus_ncpu: int
    genus_time: str
    innovus_time: str
    poll_interval_sec: int
    max_poll_attempts: int
    cleanup_remote_after_download: bool
    tool_env_script: str
    genus_bin: str
    innovus_bin: str
    remote_sdc: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    # ---- Innovus-only mode ----
    skip_genus: bool = False
    shared_genus_dir: str = ""       # remote path to reuse genus output
    genus_only: bool = False  # Treat Genus DONE as success and skip the entire Innovus phase




@dataclass
class CadenceRemoteExecutionResult:
    ok_submit: bool
    remote_root: str
    local_artifact_root: str
    master_log_path: str
    genus_jobid: Optional[str] = None
    innovus_jobid: Optional[str] = None
    genus_status: Optional[str] = None
    innovus_status: Optional[str] = None
    error: Optional[str] = None



@dataclass
class CadenceRemoteResilienceConfig:
    """Resilience parameters from cadence.remote config."""
    stop_on_remote_error: bool = True
    resume_on_restart: bool = True
    keepalive_sec: int = 30
    reconnect_retries: int = 2
    reconnect_retry_interval_sec: int = 5
    max_submit_backpressure_wait_sec: int = 1800
    submit_retry_interval_sec: int = 60

    # ---- NEW: proactive LSF gating ----
    # Job-count limits derived from bu output
    lsf_pending_jobs_soft_limit: int = 24
    lsf_running_jobs_soft_limit: int = 32

    # Total RUN-slot limit derived from bjobs output
    lsf_running_slots_soft_limit: int = 256

    # Polling intervals
    lsf_check_interval_sec: int = 45
    lsf_check_jitter_sec: int = 10



class CadenceRemoteScriptBuilder:
    def __init__(self, spec: CadenceRemoteRunSpec):
        self.spec = spec

    def _merged_env(self) -> "Dict[str, str]":
        env = dict(self.spec.env)
        if self.spec.remote_sdc and str(self.spec.remote_sdc).strip():
            env["REMOTE_SDC_FILE"] = str(self.spec.remote_sdc).strip()
        # Inject CPU counts into the environment so the Tcl scripts can read them
        env["GENUS_NCPU"] = str(self.spec.genus_ncpu)
        env["INNOVUS_NCPU"] = str(self.spec.innovus_ncpu)
        return env
    
    def build_run_env_csh(self) -> str:
        merged_env = self._merged_env()
        # _diag_rs_in_env = "RUN_STAGE" in merged_env
        # _diag_rs_val = merged_env.get("RUN_STAGE", "<MISSING>")
        # print(f"[DIAG-3] spec.stage={self.spec.stage!r} "
        #       f"RUN_STAGE_in_merged_env={_diag_rs_in_env} "
        #       f"merged_env_RUN_STAGE={_diag_rs_val!r} "
        #       f"spec.run_id={self.spec.run_id!r}")
        lines = [
            "#!/bin/csh -f",
            f"setenv TOP {_csh_quote(self.spec.top)}",
            f"setenv RUN_STAGE {_csh_quote(self.spec.stage)}",
        ]
        for key in sorted(merged_env.keys()):
            lines.append(f"setenv {key} {_csh_quote(merged_env[key])}")
        return "\n".join(lines) + "\n"

    def build_run_env_sh(self) -> str:
        merged_env = self._merged_env()
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"export TOP={shell_quote(self.spec.top)}",
            f"export RUN_STAGE={shell_quote(self.spec.stage)}",
        ]
        for key in sorted(merged_env.keys()):
            lines.append(f"export {key}={shell_quote(merged_env[key])}")
        return "\n".join(lines) + "\n"

    def build_genus_tcl(self) -> str:
        remote_root = self.spec.remote_root
        benchmark_root = "/remote/benchmark"
        # Read PLAT_DIR from the environment instead of hardcoding ASAP7
        platform_root = self.spec.env.get("PLAT_DIR", "/remote/platform/asap7")

        return textwrap.dedent(
            f"""\
            # Auto-generated Genus TCL (ASAP7 + NanGate45 dual-platform)
            {_env_or_default_tcl_proc()}
            set ROOT      {json.dumps(remote_root)}
            set OUT_DIR   {json.dumps(posixpath.join(remote_root, "outbox/cadence/genus"))}
            set LOG_DIR   {json.dumps(posixpath.join(remote_root, "outbox/cadence/logs"))}
            file mkdir $OUT_DIR
            file mkdir $LOG_DIR

            set TOP          [env_or_default "TOP" {json.dumps(self.spec.top)}]
            set RTL_DIR      [env_or_default "RTL_DIR" {json.dumps(posixpath.join(benchmark_root, "jpeg"))}]
            set PLAT_DIR     [env_or_default "PLAT_DIR" {json.dumps(platform_root)}]
            set LIB_MODEL    [string toupper [env_or_default "LIB_MODEL" "NLDM"]]
            set LIB_CORNER   [string toupper [env_or_default "LIB_CORNER" "RVT_TT"]]
            set REMOTE_SDC   [env_or_default "REMOTE_SDC_FILE" ""]
            set OVERRIDE_TCL [env_or_default "OVERRIDE_TCL" ""]

            # ── Platform detection ──
            set _is_nangate [string match "*nangate*" [string tolower $PLAT_DIR]]
            if {{$_is_nangate}} {{
                puts "INFO: ★ NanGate45 platform detected"
            }} else {{
                puts "INFO: ★ ASAP7 platform detected"
            }}

            # Split LIB_CORNER like RVT_TT -> VT=RVT, PVT=TT (ASAP7 only, harmless for NanGate)
            set corner_parts [split $LIB_CORNER "_"]
            if {{[llength $corner_parts] != 2}} {{
                if {{!$_is_nangate}} {{
                    puts "ERROR: LIB_CORNER must look like RVT_TT / LVT_FF / SLVT_SS, got $LIB_CORNER"
                    exit 4
                }}
                # NanGate45 does not need corner splitting, so use placeholder values
                set LIB_VT  "NOM"
                set LIB_PVT "TYP"
            }} else {{
                set LIB_VT  [lindex $corner_parts 0]
                set LIB_PVT [lindex $corner_parts 1]
            }}
            set model_l [string tolower $LIB_MODEL]

            puts "INFO: ROOT=$ROOT"
            puts "INFO: TOP=$TOP"
            puts "INFO: RTL_DIR=$RTL_DIR"
            puts "INFO: PLAT_DIR=$PLAT_DIR"
            puts "INFO: OUT_DIR=$OUT_DIR"
            puts "INFO: LIB_MODEL=$LIB_MODEL"
            puts "INFO: LIB_CORNER=$LIB_CORNER"
            puts "INFO: LIB_VT=$LIB_VT LIB_PVT=$LIB_PVT"
            puts "INFO: REMOTE_SDC=$REMOTE_SDC"

            set NCPUS [env_or_default "GENUS_NCPU" "4"]
            puts "INFO: GENUS_NCPU=$NCPUS"
            catch {{ set_db max_cpus_per_server $NCPUS }}

            if {{![file isdirectory $RTL_DIR]}} {{
                puts "ERROR: RTL_DIR not found: $RTL_DIR"
                exit 3
            }}

            set rtl_cmd "find \\"$RTL_DIR\\" -type f | grep -E '\\\\.(v|sv)$' | sort -u"
            if {{[catch {{ set rtl_out [exec sh -c $rtl_cmd] }} err]}} {{
                puts "ERROR: failed to enumerate RTL files: $err"
                puts "ERROR: rtl_cmd=$rtl_cmd"
                exit 3
            }}

            set rtl_files {{}}
            foreach f [split $rtl_out "\\n"] {{
                set f [string trim $f]
                if {{$f ne ""}} {{ lappend rtl_files $f }}
            }}

            if {{[llength $rtl_files] == 0}} {{
                puts "ERROR: no RTL files found under $RTL_DIR"
                exit 3
            }}

            puts "INFO: RTL files count = [llength $rtl_files]"

            # ── Sort: package/header files first (e.g. ibex_pkg.sv, prim_assert.sv) ──
            set pkg_files {{}}
            set non_pkg_files {{}}
            foreach f $rtl_files {{
                set fname [file tail $f]
                if {{[string match "*_pkg*" $fname] || [string match "*_package*" $fname] || [string match "prim_assert*" $fname]}} {{
                    lappend pkg_files $f
                }} else {{
                    lappend non_pkg_files $f
                }}
            }}
            set rtl_files [concat $pkg_files $non_pkg_files]
            if {{[llength $pkg_files] > 0}} {{
                puts "INFO: Package/header files (loaded first): $pkg_files"
            }}

            set incdirs [list $RTL_DIR]

            set EXTRA_INCS [env_or_default "VERILOG_INCLUDE_DIRS" ""]
            if {{$EXTRA_INCS eq ""}} {{ set EXTRA_INCS [env_or_default "SYNTH_VERILOG_INCLUDE_DIRS" ""] }}
            if {{$EXTRA_INCS eq ""}} {{ set EXTRA_INCS [env_or_default "RTL_INCLUDE_DIRS" ""] }}

            if {{$EXTRA_INCS ne ""}} {{
                foreach d [split $EXTRA_INCS " "] {{
                    set d [string trim $d]
                    if {{$d ne ""}} {{ lappend incdirs $d }}
                }}
            }} else {{
                foreach d [list "$RTL_DIR/include" "$RTL_DIR/inc"] {{
                    if {{[file isdirectory $d]}} {{ lappend incdirs $d }}
                }}
            }}
            set incdirs [lsort -unique $incdirs]

            set_db init_hdl_search_path $incdirs
            puts "INFO: INCDIRS=$incdirs"

            # ================================================================
            # ★ Platform-aware lib discovery
            # ================================================================
            set LIBCACHE "$OUT_DIR/libcache"
            file mkdir $LIBCACHE

            if {{$_is_nangate}} {{
                # NanGate45: libraries live directly under $PLAT_DIR/lib/
                set LIB_DIR "$PLAT_DIR/lib"
                if {{![file isdirectory $LIB_DIR]}} {{
                    puts "ERROR: LIB_DIR not found: $LIB_DIR"
                    exit 4
                }}
                set lib_candidates {{}}
                # Find all .lib files while excluding fakeram
                set ng_cmd "find \\"$LIB_DIR\\" -maxdepth 1 -type f -name '*.lib' | grep -vi 'fakeram' | sort"
                if {{![catch {{ set ng_out [exec sh -c $ng_cmd] }} ng_err]}} {{
                    foreach f [split $ng_out "\\n"] {{
                        set f [string trim $f]
                        if {{$f ne ""}} {{ lappend lib_candidates $f }}
                    }}
                }}
                if {{[llength $lib_candidates] == 0}} {{
                    puts "ERROR: cannot find any stdcell .lib under $LIB_DIR (NanGate45)"
                    exit 4
                }}
                set lib_candidates [lsort -unique $lib_candidates]
                puts "INFO: NanGate45 libs: $lib_candidates"

            }} else {{
                # ASAP7: libraries live under $PLAT_DIR/lib/NLDM/ in 5 groups
                set LIB_DIR "$PLAT_DIR/lib/$LIB_MODEL"
                if {{![file isdirectory $LIB_DIR]}} {{
                    puts "ERROR: LIB_DIR not found: $LIB_DIR"
                    exit 4
                }}

                set groups {{AO INVBUF OA SEQ SIMPLE}}

                proc pick_one_stdcell_lib {{lib_dir group lib_vt lib_pvt model_l}} {{
                    set pats [list \\
                        "asap7sc7p5t_${{group}}_${{lib_vt}}_${{lib_pvt}}_${{model_l}}_*.lib" \\
                        "asap7sc7p5t_${{group}}_${{lib_vt}}_${{lib_pvt}}_${{model_l}}_*.lib.gz" \\
                    ]
                    foreach p $pats {{
                        set cmd "find \\"$lib_dir\\" -maxdepth 1 -type f -iname \\"$p\\" | grep -v 'FAKE\\\\.lib$' | grep -vi 'fakeram' | grep -vi 'fakeregfile' | sort | head -n 1"
                        if {{![catch {{ set out [exec sh -c $cmd] }} err]}} {{
                            set out [string trim $out]
                            if {{$out ne ""}} {{
                                return $out
                            }}
                        }}
                    }}
                    return ""
                }}

                set lib_candidates {{}}
                foreach g $groups {{
                    set found [pick_one_stdcell_lib $LIB_DIR $g $LIB_VT $LIB_PVT $model_l]
                    if {{$found eq ""}} {{
                        puts "ERROR: cannot find stdcell lib for group=$g VT=$LIB_VT PVT=$LIB_PVT MODEL=$model_l under $LIB_DIR"
                        exit 4
                    }}

                    if {{[string match "*.gz" $found]}} {{
                        set outlib "$LIBCACHE/[file rootname [file tail $found]]"
                        if {{![file exists $outlib]}} {{
                            set gunzip_cmd "gunzip -c \\"$found\\" > \\"$outlib\\""
                            if {{[catch {{ exec sh -c $gunzip_cmd }} gz_err]}} {{
                                puts "ERROR: failed to gunzip $found: $gz_err"
                                exit 4
                            }}
                        }}
                        lappend lib_candidates $outlib
                    }} else {{
                        lappend lib_candidates $found
                    }}
                }}

                set lib_candidates [lsort -unique $lib_candidates]
            }}

            puts "INFO: Selected libs=$lib_candidates"

            set_db common_ui true
            set_db init_lib_search_path [list $LIB_DIR $LIBCACHE $PLAT_DIR]
            read_libs $lib_candidates
            set_db library $lib_candidates

            set SDC ""
            if {{$REMOTE_SDC ne "" && [file exists $REMOTE_SDC]}} {{
                set SDC $REMOTE_SDC
            }} else {{
                foreach sdc_try [list \\
                    "{posixpath.join(remote_root, "constraints")}/${{TOP}}.sdc" \\
                    "{posixpath.join(remote_root, "constraints")}/top.sdc" \\
                    "$RTL_DIR/${{TOP}}.sdc" \\
                    "$RTL_DIR/constraints.sdc" \\
                    "{posixpath.join(remote_root, "outbox/cadence/scripts")}/${{TOP}}.sdc" \\
                    "$PLAT_DIR/constraints.sdc" \\
                ] {{
                    if {{[file exists $sdc_try]}} {{
                        set SDC $sdc_try
                        break
                    }}
                }}
            }}
            puts "INFO: SDC final candidate=$SDC"

            if {{[catch {{ read_hdl -sv $rtl_files }} err]}} {{
                puts "ERROR: read_hdl failed: $err"
                exit 5
            }}

            if {{[catch {{ elaborate $TOP }} err]}} {{
                puts "ERROR: elaborate failed: $err"
                exit 6
            }}

            catch {{ set_db information_level 2 }}
            # catch {{ set_db message:truncate false }}

            if {{$SDC ne ""}} {{
                puts "INFO: Using SDC=$SDC"
                if {{[catch {{ read_sdc $SDC }} err]}} {{
                    puts "ERROR: read_sdc failed: $err"
                    exit 7
                }}
            }} else {{
                puts "WARNING: no SDC found; continue without SDC"
            }}

            catch {{ set_db delete_unloaded_insts false }}
            catch {{ set_db delete_unloaded_seqs false }}
            catch {{ set_db optimize_constant_0_flops false }}
            catch {{ set_db optimize_constant_1_flops false }}

            catch {{ check_design  > "$OUT_DIR/check_design_pre.rpt" }}
            catch {{ report_design > "$OUT_DIR/report_design_pre_synth.rpt" }}
            catch {{ report_clocks > "$OUT_DIR/report_clocks_pre_synth.rpt" }}
            catch {{ check_timing  > "$OUT_DIR/check_timing_pre_synth.rpt" }}
            catch {{ report_qor    > "$OUT_DIR/report_qor_pre_synth.rpt" }}
            catch {{ report_timing > "$OUT_DIR/report_timing_pre_synth.rpt" }}

            if {{$OVERRIDE_TCL ne "" && [file exists $OVERRIDE_TCL]}} {{
                source $OVERRIDE_TCL
            }}

            syn_generic
            catch {{ report_design > "$OUT_DIR/report_design_post_generic.rpt" }}
            catch {{ report_clocks > "$OUT_DIR/report_clocks_post_generic.rpt" }}
            catch {{ check_timing  > "$OUT_DIR/check_timing_post_generic.rpt" }}
            catch {{ report_qor    > "$OUT_DIR/report_qor_post_generic.rpt" }}
            catch {{ report_timing > "$OUT_DIR/report_timing_post_generic.rpt" }}

            syn_map
            catch {{ report_design > "$OUT_DIR/report_design_post_map.rpt" }}
            catch {{ report_clocks > "$OUT_DIR/report_clocks_post_map.rpt" }}
            catch {{ check_timing  > "$OUT_DIR/check_timing_post_map.rpt" }}
            catch {{ report_qor    > "$OUT_DIR/report_qor_post_map.rpt" }}
            catch {{ report_timing > "$OUT_DIR/report_timing_post_map.rpt" }}

            syn_opt
            catch {{ report_design > "$OUT_DIR/report_design_post_opt.rpt" }}
            catch {{ report_clocks > "$OUT_DIR/report_clocks_post_opt.rpt" }}
            catch {{ check_timing  > "$OUT_DIR/check_timing_post_opt.rpt" }}
            catch {{ report_qor    > "$OUT_DIR/report_qor_post_opt.rpt" }}
            catch {{ report_timing > "$OUT_DIR/report_timing_post_opt.rpt" }}

            set out_v "$OUT_DIR/${{TOP}}_mapped.v"
            set out_s "$OUT_DIR/${{TOP}}_mapped.sdc"
            write_hdl -mapped > $out_v
            write_sdc > $out_s

            catch {{ report_area   > "$OUT_DIR/report_area_final.rpt" }}
            catch {{ report_power  > "$OUT_DIR/report_power_final.rpt" }}
            catch {{ report_clocks > "$OUT_DIR/report_clocks_final.rpt" }}
            catch {{ check_timing  > "$OUT_DIR/check_timing_final.rpt" }}
            catch {{ report_qor    > "$OUT_DIR/report_qor_final.rpt" }}
            catch {{ report_timing > "$OUT_DIR/report_timing_final.rpt" }}

            puts "INFO: Genus done."
            exit
            """
        )

    def build_innovus_tcl(self) -> str:
        remote_root = self.spec.remote_root
        # Read PLAT_DIR from the environment instead of hardcoding ASAP7
        platform_root = self.spec.env.get("PLAT_DIR", "/remote/platform/asap7")
        # ---- Innovus-only mode: reuse shared genus output ----
        if self.spec.skip_genus and self.spec.shared_genus_dir:
            genus_dir = self.spec.shared_genus_dir
        else:
            genus_dir = posixpath.join(remote_root, "outbox/cadence/genus")
        innovus_dir = posixpath.join(remote_root, "outbox/cadence/innovus")
        logs_dir = posixpath.join(remote_root, "outbox/cadence/logs")


        return textwrap.dedent(
            f"""\
            # Auto-generated Innovus TCL (ASAP7 + NanGate45 dual-platform)
            {_env_or_default_tcl_proc()}
            set ROOT         {json.dumps(remote_root)}
            set PLAT_DIR     [env_or_default "PLAT_DIR" {json.dumps(platform_root)}]
            set GENUS_DIR    {json.dumps(genus_dir)}
            set OUT_DIR      {json.dumps(innovus_dir)}
            set LOG_DIR      {json.dumps(logs_dir)}
            set TOP          [env_or_default "TOP" {json.dumps(self.spec.top)}]
            set RUN_STAGE    [string tolower [env_or_default "RUN_STAGE" {json.dumps(self.spec.stage)}]]
            # ---- Floorplan parameters ----
            set CORE_UTIL           [expr {{double([env_or_default "CORE_UTIL" "0.65"])}}]
            set ASPECT_RATIO        [expr {{double([env_or_default "ASPECT_RATIO" "1.00"])}}]
            set CORE_MARGIN         [expr {{int([env_or_default "CORE_MARGIN" "8"])}}]
            # ★ FIX IMPTCM-48: PLACE_DENSITY var kept for INFO logging but no longer
            # passed to setPlaceMode (the old -place_detail_density option does not exist).
            # Use PLACE_MAX_DENSITY to control global placement density instead.
            set PLACE_DENSITY       [expr {{double([env_or_default "PLACE_DENSITY" "0.70"])}}]

            # ---- Global placement parameters ----
            set PLACE_CONG_EFFORT   [env_or_default "PLACE_CONG_EFFORT" "medium"]
            set PLACE_TIMING_EFFORT [env_or_default "PLACE_TIMING_EFFORT" "high"]
            # ★ FIX IMPTCM-23: Innovus requires enum "true"/"false", not int 1/0
            set PLACE_UNIFORM_DENSITY [env_or_default "PLACE_UNIFORM_DENSITY" "true"]
            set PLACE_IO_PINS_AWARE   [env_or_default "PLACE_IO_PINS_AWARE"   "true"]
            set TD_PLACE            [expr {{int([env_or_default "TD_PLACE" "1"])}}]
            set PLACE_MAX_DENSITY   [expr {{double([env_or_default "PLACE_MAX_DENSITY" "0.85"])}}]

            # ---- Pre-CTS optimization parameters ----
            set PRECTS_SETUP_OPT    [expr {{int([env_or_default "PRECTS_SETUP_OPT" "1"])}}]
            set PRECTS_HOLD_OPT     [expr {{int([env_or_default "PRECTS_HOLD_OPT" "0"])}}]
            set DRV_EFFORT          [env_or_default "DRV_EFFORT" "medium"]

            # ---- CTS parameters ----
            set RUN_CTS             [expr {{int([env_or_default "RUN_CTS" "1"])}}]
            set CTS_TARGET_SKEW     [expr {{double([env_or_default "CTS_TARGET_SKEW" "0.08"])}}]
            set CTS_POST_OPT        [expr {{int([env_or_default "CTS_POST_OPT" "1"])}}]

            # ---- Routing parameters ----
            # ★ FIX IMPTCM-23: Innovus requires enum "true"/"false", not int 1/0
            set ROUTE_TIMING_DRIVEN [env_or_default "ROUTE_TIMING_DRIVEN" "true"]
            set ROUTE_SI_DRIVEN     [env_or_default "ROUTE_SI_DRIVEN"     "false"]
            set ROUTE_EFFORT        [env_or_default "ROUTE_EFFORT" "medium"]
            set ROUTE_LAYER_EFFORT  [env_or_default "ROUTE_LAYER_EFFORT" "medium"]

            # ---- Non-DSE parameters (unchanged) ----
            set LIB_MODEL    [string toupper [env_or_default "LIB_MODEL" "NLDM"]]
            set LIB_CORNER   [string toupper [env_or_default "LIB_CORNER" "RVT_TT"]]
            set OVERRIDE_TCL [env_or_default "OVERRIDE_TCL" ""]

            # ── Platform detection ──
            set _is_nangate [string match "*nangate*" [string tolower $PLAT_DIR]]
            if {{$_is_nangate}} {{
                puts "INFO: ★ NanGate45 platform detected (Innovus)"
            }} else {{
                puts "INFO: ★ ASAP7 platform detected (Innovus)"
            }}

            set corner_parts [split $LIB_CORNER "_"]
            if {{[llength $corner_parts] != 2}} {{
                if {{!$_is_nangate}} {{
                    puts "ERROR: LIB_CORNER must look like RVT_TT / LVT_FF / SLVT_SS, got $LIB_CORNER"
                    exit 21
                }}
                set LIB_VT  "NOM"
                set LIB_PVT "TYP"
            }} else {{
                set LIB_VT  [lindex $corner_parts 0]
                set LIB_PVT [lindex $corner_parts 1]
            }}
            set model_l [string tolower $LIB_MODEL]

            file mkdir $OUT_DIR
            file mkdir $LOG_DIR

            # ================================================================
            # ★ Platform-aware message suppression
            # ================================================================
            if {{!$_is_nangate}} {{
                # ASAP7 specific LEF error suppression
                suppressMessage IMPLF-40
                suppressMessage IMPLF-63
                suppressMessage IMPLF-45
            }}

            puts "INFO: ROOT=$ROOT  TOP=$TOP  RUN_STAGE=$RUN_STAGE"
            puts "INFO: Floorplan: CORE_UTIL=$CORE_UTIL ASPECT_RATIO=$ASPECT_RATIO CORE_MARGIN=$CORE_MARGIN PLACE_DENSITY=$PLACE_DENSITY"
            puts "INFO: Placement: TD_PLACE=$TD_PLACE CONG_EFFORT=$PLACE_CONG_EFFORT TIMING_EFFORT=$PLACE_TIMING_EFFORT UNIFORM=$PLACE_UNIFORM_DENSITY IO_PINS=$PLACE_IO_PINS_AWARE MAX_DENSITY=$PLACE_MAX_DENSITY"
            puts "INFO: Pre-CTS: SETUP_OPT=$PRECTS_SETUP_OPT HOLD_OPT=$PRECTS_HOLD_OPT DRV_EFFORT=$DRV_EFFORT"
            puts "INFO: CTS: RUN_CTS=$RUN_CTS TARGET_SKEW=$CTS_TARGET_SKEW POST_OPT=$CTS_POST_OPT"
            puts "INFO: Routing: TIMING_DRIVEN=$ROUTE_TIMING_DRIVEN SI_DRIVEN=$ROUTE_SI_DRIVEN EFFORT=$ROUTE_EFFORT LAYER_EFFORT=$ROUTE_LAYER_EFFORT"
            puts "INFO: LIB_MODEL=$LIB_MODEL LIB_CORNER=$LIB_CORNER"

            set NETLIST "$GENUS_DIR/${{TOP}}_mapped.v"
            set SDC_G   "$GENUS_DIR/${{TOP}}_mapped.sdc"
            if {{![file exists $NETLIST]}} {{
                puts "ERROR: missing netlist $NETLIST (run Genus first)"
                exit 2
            }}

            set SDC $SDC_G
            set PLAT_SDC [env_or_default "REMOTE_SDC_FILE" ""]
            if {{$PLAT_SDC ne "" && [file exists $PLAT_SDC]}} {{
                set SDC $PLAT_SDC
            }}
            puts "INFO: Innovus final SDC=$SDC"

            set NCPUS [env_or_default "INNOVUS_NCPU" "8"]
            puts "INFO: INNOVUS_NCPU=$NCPUS"
            catch {{ setMultiCpuUsage -localCpu $NCPUS }}

            # ================================================================
            # ★ Platform-aware LEF discovery
            # ================================================================
            set lefs_all [lsort -unique [concat [glob -nocomplain "$PLAT_DIR/lef/*.lef"] [glob -nocomplain "$PLAT_DIR/lef/**/*.lef"]]]
            if {{[llength $lefs_all] == 0}} {{
                puts "ERROR: no LEF found under $PLAT_DIR/lef"
                exit 2
            }}

            # Tech LEF: search multiple patterns
            set tech_candidates [lsort -unique [concat \\
                [glob -nocomplain "$PLAT_DIR/lef/asap7_tech*.lef"] \\
                [glob -nocomplain "$PLAT_DIR/lef/*tech*.lef"] \\
                [glob -nocomplain "$PLAT_DIR/lef/*Tech*.lef"] \\
            ]]
            if {{[llength $tech_candidates] == 0}} {{
                puts "ERROR: cannot find tech LEF"
                exit 3
            }}
            set tech_lef [lindex $tech_candidates 0]
            puts "INFO: tech_lef=$tech_lef"

            # Separate tech LEF from cell LEFs
            set lefs_rest {{}}
            foreach f $lefs_all {{
                if {{[file normalize $f] ne [file normalize $tech_lef]}} {{
                    lappend lefs_rest $f
                }}
            }}
            set init_lef_file [concat [list $tech_lef] $lefs_rest]

            # ================================================================
            # ★ Platform-aware SITE name
            # ================================================================
            set SITE_NAME ""
            if {{[file exists $tech_lef]}} {{
                set fh [open $tech_lef r]
                while {{[gets $fh line] >= 0}} {{
                    if {{[regexp {{^\\s*SITE\\s+([A-Za-z0-9_]+)}} $line -> s]}} {{
                        set SITE_NAME $s
                        break
                    }}
                }}
                close $fh
            }}
            if {{$SITE_NAME eq ""}} {{
                if {{$_is_nangate}} {{
                    set SITE_NAME "FreePDK45_38x28_10R_NP_162NW_34O"
                }} else {{
                    set SITE_NAME "asap7sc7p5t"
                }}
            }}
            puts "INFO: SITE_NAME=$SITE_NAME"

            set init_top_cell $TOP
            set init_verilog $NETLIST
            if {{[file exists $SDC]}} {{
                set init_sdc_file $SDC
            }}

            # ================================================================
            # ★ Platform-aware timing lib discovery
            # ================================================================
            set LIBCACHE "$GENUS_DIR/libcache"
            set timing_libs {{}}

            if {{$_is_nangate}} {{
                # NanGate45: libraries live directly under $PLAT_DIR/lib/
                set LIB_DIR "$PLAT_DIR/lib"
                # Check Genus libcache first (Genus may have already decompressed them), then fall back to the platform lib directory
                set ng_cmd "find \\"$LIBCACHE\\" \\"$LIB_DIR\\" -maxdepth 1 -type f -name '*.lib' 2>/dev/null | grep -vi 'fakeram' | sort -u"
                if {{![catch {{ set ng_out [exec sh -c $ng_cmd] }} ng_err]}} {{
                    foreach f [split $ng_out "\\n"] {{
                        set f [string trim $f]
                        if {{$f ne "" && ![string match "*.gz" $f]}} {{
                            lappend timing_libs $f
                        }}
                    }}
                }}
                set timing_libs [lsort -unique $timing_libs]
                if {{[llength $timing_libs] == 0}} {{
                    puts "ERROR: cannot find any timing lib for NanGate45"
                    exit 21
                }}
                puts "INFO: NanGate45 timing libs ([llength $timing_libs]): $timing_libs"

            }} else {{
                # ── ASAP7: 5 groups in $PLAT_DIR/lib/NLDM/ ──
                set LIB_DIR "$PLAT_DIR/lib/$LIB_MODEL"
                set groups {{AO INVBUF OA SEQ SIMPLE}}

                proc pick_innovus_stdcell_lib {{libcache libdir group lib_vt lib_pvt model_l}} {{
                    set pats [list \\
                        "$libcache/asap7sc7p5t_${{group}}_${{lib_vt}}_${{lib_pvt}}_${{model_l}}_*.lib" \\
                        "$libdir/asap7sc7p5t_${{group}}_${{lib_vt}}_${{lib_pvt}}_${{model_l}}_*.lib" \\
                    ]
                    foreach p $pats {{
                        set cands [lsort -unique [glob -nocomplain $p]]
                        foreach c $cands {{
                            if {{![regexp {{FAKE\\.lib$}} $c] &&
                                ![regexp {{fakeram}} $c] &&
                                ![regexp {{fakeregfile}} $c] &&
                                ![regexp {{SRAM}} $c]}} {{
                                return $c
                            }}
                        }}
                    }}
                    return ""
                }}

                foreach g $groups {{
                    set found [pick_innovus_stdcell_lib $LIBCACHE $LIB_DIR $g $LIB_VT $LIB_PVT $model_l]
                    if {{$found eq ""}} {{
                        puts "ERROR: cannot find timing lib for group=$g VT=$LIB_VT PVT=$LIB_PVT MODEL=$model_l"
                        exit 21
                    }}
                    if {{[string match "*.gz" $found]}} {{
                        puts "ERROR: Innovus requires decompressed .lib, but got: $found"
                        exit 21
                    }}
                    lappend timing_libs $found
                }}

                set timing_libs [lsort -unique $timing_libs]
                if {{[llength $timing_libs] != 5}} {{
                    puts "ERROR: expected 5 stdcell timing libs, got [llength $timing_libs]"
                    puts "ERROR: timing_libs=$timing_libs"
                    exit 21
                }}
            }}

            # ================================================================
            # MMMC setup (same for both platforms)
            # ================================================================
            set MMMC_FILE "$OUT_DIR/mmmc.tcl"
            set fh [open $MMMC_FILE w]
            puts $fh "create_library_set -name LS_TYP -timing {{ [join $timing_libs {{ }}] }}"
            puts $fh "create_rc_corner -name RC_TYP"
            puts $fh "create_delay_corner -name DC_TYP -library_set LS_TYP -rc_corner RC_TYP"
            if {{[file exists $SDC]}} {{
                set SDC_N [file normalize $SDC]
                puts $fh "create_constraint_mode -name CM_TYP -sdc_files {{ $SDC_N }}"
            }} else {{
                puts $fh "create_constraint_mode -name CM_TYP"
            }}
            puts $fh "create_analysis_view -name AV_TYP -constraint_mode CM_TYP -delay_corner DC_TYP"
            puts $fh "set_analysis_view -setup AV_TYP -hold AV_TYP"
            close $fh
            set init_mmmc_file $MMMC_FILE

            proc write_stage_reports {{out_dir stage_name}} {{
                file mkdir $out_dir
                catch {{ report_area > "$out_dir/report_area_${{stage_name}}.rpt" }}
                catch {{ report_power > "$out_dir/report_power_${{stage_name}}.rpt" }}
                catch {{ report_qor -format text -out_file "$out_dir/report_qor_${{stage_name}}.rpt" }}
                catch {{ redirect -file "$out_dir/report_timing_${{stage_name}}.rpt" {{ report_timing }} }}
                catch {{ summaryReport -outFile "$out_dir/summary_${{stage_name}}.rpt" }}
                # ★ FIX: output die area
                catch {{
                    set _bbox ""
                    set _got_bbox 0

                    if {{!$_got_bbox}} {{
                        if {{![catch {{set _bbox [get_db current_design .bbox]}}]}} {{
                            if {{$_bbox ne "" && $_bbox ne "{{}}"}} {{
                                set _got_bbox 1
                                puts "INFO: die bbox from get_db: $_bbox"
                            }}
                        }}
                    }}

                    if {{!$_got_bbox}} {{
                        if {{![catch {{set _bbox [dbGet top.fPlan.box]}}]}} {{
                            if {{$_bbox ne "" && $_bbox ne "{{}}"}} {{
                                set _got_bbox 1
                                puts "INFO: die bbox from dbGet: $_bbox"
                            }}
                        }}
                    }}

                    if {{!$_got_bbox}} {{
                        if {{![catch {{
                            set _rows [dbGet top.fPlan.numRows]
                            set _row_h [dbGet [lindex [dbGet top.fPlan.rows] 0] .site.size_y]
                            set _row_w [dbGet top.fPlan.box_size_x]
                        }}]}} {{
                        }}
                    }}

                    if {{!$_got_bbox}} {{
                        puts "WARN: cannot get die bbox, skipping report_area_die"
                        return
                    }}

                    set _flat [join $_bbox]
                    if {{[llength $_flat] == 4}} {{
                        set _x0 [lindex $_flat 0]
                        set _y0 [lindex $_flat 1]
                        set _x1 [lindex $_flat 2]
                        set _y1 [lindex $_flat 3]
                    }} else {{
                        set _x0 [lindex [lindex $_bbox 0] 0]
                        set _y0 [lindex [lindex $_bbox 0] 1]
                        set _x1 [lindex [lindex $_bbox 1] 0]
                        set _y1 [lindex [lindex $_bbox 1] 1]
                    }}

                    set _die_area [expr {{($_x1 - $_x0) * ($_y1 - $_y0)}}]

                    set _die_rpt [open "$out_dir/report_area_die_${{stage_name}}.rpt" w]
                    puts $_die_rpt "Die area : $_die_area"
                    puts $_die_rpt "Die bbox : $_bbox"
                    puts $_die_rpt "Die W x H: [expr {{$_x1 - $_x0}}] x [expr {{$_y1 - $_y0}}]"
                    close $_die_rpt
                    puts "INFO: wrote die area = $_die_area to report_area_die_${{stage_name}}.rpt"
                }}
            }}

            if {{[catch {{ init_design }} init_err]}} {{
                puts "ERROR: init_design failed: $init_err"
                exit 24
            }}

            catch {{ redirect -file "$OUT_DIR/report_analysis_view_post_init.rpt" {{ report_analysis_view }} }}
            catch {{ redirect -file "$OUT_DIR/report_clocks_post_init.rpt" {{ report_clocks }} }}
            catch {{ redirect -file "$OUT_DIR/check_timing_post_init.rpt" {{ check_timing }} }}
            catch {{ redirect -file "$OUT_DIR/report_timing_post_init.rpt" {{ report_timing }} }}

            set clk_list {{}}
            if {{[catch {{ set clk_list [get_clocks *] }} clk_err]}} {{
                puts "ERROR: get_clocks failed after init_design: $clk_err"
                exit 25
            }}
            if {{[llength $clk_list] == 0}} {{
                puts "ERROR: no clocks found after init_design"
                exit 25
            }}

            # ★ Source user override FIRST, so our layer cap always wins afterwards
            if {{$OVERRIDE_TCL ne "" && [file exists $OVERRIDE_TCL]}} {{
                puts "INFO: sourcing OVERRIDE_TCL=$OVERRIDE_TCL"
                source $OVERRIDE_TCL
            }}

            # ================================================================
            # ★ Platform-aware routing layer setup
            # ================================================================
            if {{$_is_nangate}} {{
                # NanGate45: metal1-metal9 (metal10 exists but reserved for power)
                set _BOT_LAYER "metal2"
                set _TOP_LAYER "metal9"
                set _TOP_NUM 9
            }} else {{
                # ASAP7: M1-M9
                set _BOT_LAYER "M2"
                set _TOP_LAYER "M9"
                set _TOP_NUM 9
            }}
            puts "INFO: forcing routing layers to $_BOT_LAYER..$_TOP_LAYER"
            if {{[catch {{ setDesignMode -bottomRoutingLayer $_BOT_LAYER }} _err]}} {{
                puts "WARN: setDesignMode -bottomRoutingLayer $_BOT_LAYER failed ($_err), trying number 2"
                catch {{ setDesignMode -bottomRoutingLayer 2 }}
            }}
            if {{[catch {{ setDesignMode -topRoutingLayer $_TOP_LAYER }} _err]}} {{
                puts "WARN: setDesignMode -topRoutingLayer $_TOP_LAYER failed ($_err), trying number $_TOP_NUM"
                catch {{ setDesignMode -topRoutingLayer $_TOP_NUM }}
            }}
            foreach {{_opt _val}} [list -routeBottomRoutingLayer 2 -routeTopRoutingLayer $_TOP_NUM] {{
                if {{[catch {{ setNanoRouteMode $_opt $_val }} _err]}} {{
                    puts "WARN: setNanoRouteMode $_opt $_val failed: $_err"
                }}
            }}
            catch {{ redirect -file "$OUT_DIR/design_mode_check.rpt" {{ report_design_mode }} }}

            # ================================================================
            # STAGE 1: Floorplan  (params: CORE_UTIL, ASPECT_RATIO, CORE_MARGIN)
            # ================================================================
            floorPlan -site $SITE_NAME \\
                -r $ASPECT_RATIO $CORE_UTIL \\
                $CORE_MARGIN $CORE_MARGIN $CORE_MARGIN $CORE_MARGIN

            # ★ re-assert routing layer cap after floorPlan
            if {{[catch {{ setDesignMode -bottomRoutingLayer $_BOT_LAYER }} _err]}} {{
                catch {{ setDesignMode -bottomRoutingLayer 2 }}
            }}
            if {{[catch {{ setDesignMode -topRoutingLayer $_TOP_LAYER }} _err]}} {{
                catch {{ setDesignMode -topRoutingLayer $_TOP_NUM }}
            }}
            foreach {{_opt _val}} [list -routeBottomRoutingLayer 2 -routeTopRoutingLayer $_TOP_NUM] {{
                if {{[catch {{ setNanoRouteMode $_opt $_val }} _err]}} {{
                    puts "WARN: setNanoRouteMode $_opt $_val failed: $_err"
                }}
            }}


            # ================================================================
            # STAGE 2: Placement
            # ================================================================
            if {{$TD_PLACE}} {{
                catch {{ setPlaceMode -timingDriven true }}
            }} else {{
                catch {{ setPlaceMode -timingDriven false }}
            }}

            if {{[catch {{ setPlaceMode -place_global_cong_effort $PLACE_CONG_EFFORT }} _err]}} {{
                puts "WARN: setPlaceMode -place_global_cong_effort failed: $_err"
            }}
            if {{[catch {{ setPlaceMode -place_global_timing_effort $PLACE_TIMING_EFFORT }} _err]}} {{
                puts "WARN: setPlaceMode -place_global_timing_effort failed: $_err"
            }}
            if {{[catch {{ setPlaceMode -place_global_uniform_density $PLACE_UNIFORM_DENSITY }} _err]}} {{
                puts "WARN: setPlaceMode -place_global_uniform_density failed: $_err"
            }}
            if {{[catch {{ setPlaceMode -place_global_place_io_pins $PLACE_IO_PINS_AWARE }} _err]}} {{
                puts "WARN: setPlaceMode -place_global_place_io_pins failed: $_err"
            }}
            if {{[catch {{ setPlaceMode -place_global_max_density $PLACE_MAX_DENSITY }} _err]}} {{
                puts "WARN: setPlaceMode -place_global_max_density failed: $_err"
            }}

            if {{[catch {{ place_design }} err]}} {{
                puts "ERROR: place_design failed: $err"
                exit 40
            }}

            write_stage_reports $OUT_DIR "place"
            if {{$RUN_STAGE eq "place"}} {{
                catch {{ saveDesign "$OUT_DIR/${{TOP}}_place.enc" }}
                puts "INFO: place stage finished."
                exit
            }}

            # ================================================================
            # STAGE 3: Pre-CTS Optimization
            # ================================================================
            catch {{ setOptMode -drvEffort $DRV_EFFORT }}

            if {{$PRECTS_SETUP_OPT}} {{
                puts "INFO: Running pre-CTS setup optimization..."
                if {{[catch {{ optDesign -preCTS -setup }} err]}} {{
                    puts "WARN: optDesign -preCTS -setup failed: $err"
                }}
            }}
            if {{$PRECTS_HOLD_OPT}} {{
                puts "INFO: Running pre-CTS hold optimization..."
                if {{[catch {{ optDesign -preCTS -hold }} err]}} {{
                    puts "WARN: optDesign -preCTS -hold failed: $err"
                }}
            }}

            write_stage_reports $OUT_DIR "prects"

            # ================================================================
            # STAGE 4: CTS
            # ================================================================
            if {{$RUN_CTS}} {{
                puts "INFO: Running CTS with target_skew=$CTS_TARGET_SKEW"
                catch {{ set_ccopt_property target_skew $CTS_TARGET_SKEW }}
                if {{[catch {{ ccopt_design }} err]}} {{
                    puts "ERROR: ccopt_design failed: $err"
                    exit 30
                }}
                if {{$CTS_POST_OPT}} {{
                    puts "INFO: Running post-CTS optimization..."
                    catch {{ optDesign -postCTS -setup -hold }}
                }}
            }} else {{
                puts "INFO: RUN_CTS=0 -> skip CTS"
            }}

            write_stage_reports $OUT_DIR "cts"
            if {{$RUN_STAGE eq "cts"}} {{
                catch {{ saveDesign "$OUT_DIR/${{TOP}}_cts.enc" }}
                puts "INFO: cts stage finished."
                exit
            }}

            # ================================================================
            # STAGE 5: Routing
            # ================================================================
            if {{[catch {{ setNanoRouteMode -routeWithTimingDriven $ROUTE_TIMING_DRIVEN }} _err]}} {{
                puts "WARN: setNanoRouteMode -routeWithTimingDriven failed: $_err"
            }}
            if {{[catch {{ setNanoRouteMode -routeWithSiDriven $ROUTE_SI_DRIVEN }} _err]}} {{
                puts "WARN: setNanoRouteMode -routeWithSiDriven failed: $_err"
            }}

            # Map ROUTE_EFFORT to drouteEndIteration
            switch $ROUTE_EFFORT {{
                "low"    {{ catch {{ setNanoRouteMode -drouteEndIteration 3  }} }}
                "medium" {{ catch {{ setNanoRouteMode -drouteEndIteration 10 }} }}
                "high"   {{ catch {{ setNanoRouteMode -drouteEndIteration 30 }} }}
            }}

            # ★ Platform-aware ROUTE_LAYER_EFFORT mapping
            if {{$_is_nangate}} {{
                switch $ROUTE_LAYER_EFFORT {{
                    "low"    {{ catch {{ setDesignMode -topRoutingLayer metal6 }} }}
                    "medium" {{ catch {{ setDesignMode -topRoutingLayer metal8 }} }}
                    "high"   {{ catch {{ setDesignMode -topRoutingLayer metal9 }} }}
                }}
            }} else {{
                switch $ROUTE_LAYER_EFFORT {{
                    "low"    {{ catch {{ setDesignMode -topRoutingLayer M7 }} }}
                    "medium" {{ catch {{ setDesignMode -topRoutingLayer M8 }} }}
                    "high"   {{ catch {{ setDesignMode -topRoutingLayer M9 }} }}
                }}
            }}

            if {{[catch {{ routeDesign -globalDetail }} err]}} {{
                puts "ERROR: routeDesign failed: $err"
                exit 50
            }}

            # Post-route optimization (always run for best QoR)
            puts "INFO: Running post-route optimization..."
            catch {{ setOptMode -postRouteOpt true }}
            catch {{ optDesign -postRoute -setup -hold }}

            write_stage_reports $OUT_DIR "route"
            catch {{ saveDesign "$OUT_DIR/${{TOP}}.enc" }}
            puts "INFO: route stage finished."
            exit
            """
        )

    def _build_job_wrapper_setup(self) -> str:
        tool_env = self.spec.tool_env_script
        genus_bin = self.spec.genus_bin
        innovus_bin = self.spec.innovus_bin

        return textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail

            ROOT={shell_quote(self.spec.remote_root)}
            LOGDIR="$ROOT/outbox/cadence/logs"
            SCRIPTSDIR="$ROOT/outbox/cadence/scripts"
            QUEUE={shell_quote(self.spec.queue)}
            TOP={shell_quote(self.spec.top)}
            TOOL_ENV_SCRIPT={shell_quote(tool_env)}
            GENUS_CMD={shell_quote(genus_bin)}
            INNOVUS_CMD={shell_quote(innovus_bin)}

            mkdir -p "$LOGDIR"

            cat > "$LOGDIR/run_genus_job.csh" <<'GENEOF'
            #!/bin/csh -f
            set ROOT={_csh_quote(self.spec.remote_root)}
            set LOGDIR="$ROOT/outbox/cadence/logs"
            set SCRIPTSDIR="$ROOT/outbox/cadence/scripts"
            set TOOL_ENV_SCRIPT={_csh_quote(tool_env)}
            set GENUS_CMD={_csh_quote(genus_bin)}
            set APPS_BASE="/apps"
            if ( ! -d "$APPS_BASE" ) set APPS_BASE="/share/apps"
            if ( "$TOOL_ENV_SCRIPT" == "" ) set TOOL_ENV_SCRIPT="$APPS_BASE/EDAs/cshrc/cshrc_cds618"
            if ( "$GENUS_CMD" == "" ) set GENUS_CMD="$APPS_BASE/EDAs/cadence/GENUS21.1/bin/genus"

            if ( ! -d "$ROOT" ) then
            echo "[ERROR] ROOT not found: $ROOT"
            exit 2
            endif
            if ( ! -f "$TOOL_ENV_SCRIPT" ) then
            echo "[ERROR] Cadence env not found: $TOOL_ENV_SCRIPT"
            exit 3
            endif
            if ( ! -x "$GENUS_CMD" ) then
            echo "[ERROR] Genus binary not found: $GENUS_CMD"
            exit 4
            endif
            if ( ! -f "$SCRIPTSDIR/run_genus.tcl" ) then
            echo "[ERROR] missing $SCRIPTSDIR/run_genus.tcl"
            exit 10
            endif
            if ( ! -f "$SCRIPTSDIR/run_env.csh" ) then
            echo "[ERROR] missing $SCRIPTSDIR/run_env.csh"
            exit 11
            endif

            cd "$ROOT"
            source "$TOOL_ENV_SCRIPT"
            source "$SCRIPTSDIR/run_env.csh"
            echo "[INFO] HOST=`hostname`"
            "$GENUS_CMD" -no_gui -files "$SCRIPTSDIR/run_genus.tcl" -log "$LOGDIR/genus.${{LSB_JOBID}}.log"
            GENEOF
            chmod +x "$LOGDIR/run_genus_job.csh"

            cat > "$LOGDIR/run_innovus_job.csh" <<'INNOEOF'
            #!/bin/csh -f
            set ROOT={_csh_quote(self.spec.remote_root)}
            set LOGDIR="$ROOT/outbox/cadence/logs"
            set SCRIPTSDIR="$ROOT/outbox/cadence/scripts"
            set TOOL_ENV_SCRIPT={_csh_quote(tool_env)}
            set INNOVUS_CMD={_csh_quote(innovus_bin)}
            set APPS_BASE="/apps"
            if ( ! -d "$APPS_BASE" ) set APPS_BASE="/share/apps"
            if ( "$TOOL_ENV_SCRIPT" == "" ) set TOOL_ENV_SCRIPT="$APPS_BASE/EDAs/cshrc/cshrc_cds618"
            if ( "$INNOVUS_CMD" == "" ) set INNOVUS_CMD="$APPS_BASE/EDAs/cadence/INNOVUS21.1/bin/innovus"

            if ( ! -d "$ROOT" ) then
            echo "[ERROR] ROOT not found: $ROOT"
            exit 2
            endif
            if ( ! -f "$TOOL_ENV_SCRIPT" ) then
            echo "[ERROR] Cadence env not found: $TOOL_ENV_SCRIPT"
            exit 3
            endif
            if ( ! -x "$INNOVUS_CMD" ) then
            echo "[ERROR] Innovus binary not found: $INNOVUS_CMD"
            exit 5
            endif
            if ( ! -f "$SCRIPTSDIR/run_innovus.tcl" ) then
            echo "[ERROR] missing $SCRIPTSDIR/run_innovus.tcl"
            exit 10
            endif
            if ( ! -f "$SCRIPTSDIR/run_env.csh" ) then
            echo "[ERROR] missing $SCRIPTSDIR/run_env.csh"
            exit 11
            endif

            cd "$ROOT"
            source "$TOOL_ENV_SCRIPT"
            source "$SCRIPTSDIR/run_env.csh"
            echo "[INFO] HOST=`hostname`"
            "$INNOVUS_CMD" -no_gui -files "$SCRIPTSDIR/run_innovus.tcl" -log "$LOGDIR/innovus.${{LSB_JOBID}}.log"
            INNOEOF
            chmod +x "$LOGDIR/run_innovus_job.csh"
            """
        )

    def build_submit_genus_sh(self) -> str:
        return self._build_job_wrapper_setup() + textwrap.dedent(
            f"""\
            GENUS_JOBNAME="genus_{slugify(self.spec.top)}_{slugify(self.spec.run_id)}"

            rm -f "$LOGDIR/genus_jobid.txt"

            J1=$(bsub -q "$QUEUE" -J "$GENUS_JOBNAME" -n {int(self.spec.genus_ncpu)} -R "span[hosts=1]" -W {shell_quote(self.spec.genus_time)} \\
            -o "$LOGDIR/${{GENUS_JOBNAME}}.%J.out" -e "$LOGDIR/${{GENUS_JOBNAME}}.%J.err" \\
            csh -f "$LOGDIR/run_genus_job.csh" | awk -F'[<>]' '/Job/ {{print $2}}')

            if [ -z "$J1" ]; then
            echo "[ERROR] failed to get Genus job id"
            exit 40
            fi

            echo "$J1" > "$LOGDIR/genus_jobid.txt"
            echo "[INFO] Genus jobid=$J1"
            """
        )

    def build_submit_innovus_sh(self) -> str:
        return self._build_job_wrapper_setup() + textwrap.dedent(
            f"""\
            INNO_JOBNAME="innovus_{slugify(self.spec.top)}_{slugify(self.spec.run_id)}"

            rm -f "$LOGDIR/innovus_jobid.txt"

            J2=$(bsub -q "$QUEUE" -J "$INNO_JOBNAME" -n {int(self.spec.innovus_ncpu)} -R "span[hosts=1]" -W {shell_quote(self.spec.innovus_time)} \\
            -o "$LOGDIR/${{INNO_JOBNAME}}.%J.out" -e "$LOGDIR/${{INNO_JOBNAME}}.%J.err" \\
            csh -f "$LOGDIR/run_innovus_job.csh" | awk -F'[<>]' '/Job/ {{print $2}}')

            if [ -z "$J2" ]; then
            echo "[ERROR] failed to get Innovus job id"
            exit 41
            fi

            echo "$J2" > "$LOGDIR/innovus_jobid.txt"
            echo "[INFO] Innovus jobid=$J2"
            """
        )


class CadenceRemoteJobRunner:
    def __init__(self, spec: CadenceRemoteRunSpec, creds: Dict[str, str],
                 resilience: "Optional[CadenceRemoteResilienceConfig]" = None):
        self.spec = spec
        self.creds = dict(creds)
        self.resilience = resilience or CadenceRemoteResilienceConfig()

    def run(self, local_run_dir: Path, *, script_override: Optional[str] = None) -> CadenceRemoteExecutionResult:
        local_run_dir.mkdir(parents=True, exist_ok=True)
        master_log = local_run_dir / "cadence.log"
        artifact_root = local_run_dir / "remote_artifacts"

        result = CadenceRemoteExecutionResult(
            ok_submit=False,
            remote_root=self.spec.remote_root,
            local_artifact_root=str(artifact_root),
            master_log_path=str(master_log),
        )

        # print(f"[DIAG-4] run_id={self.spec.run_id} spec.stage={self.spec.stage!r}")
        # --- Resume from checkpoint ---
        ckpt_path = local_run_dir / "checkpoint.json"
        ckpt = _load_checkpoint(ckpt_path)
        resume_phase = None
        if ckpt and self.resilience.resume_on_restart:
            phase = ckpt.get("current_phase", "")
            if phase == "done":
                # Check stage consistency to prevent quick-stage results from being reused as full-stage results
                ckpt_stage = str(ckpt.get("stage", "")).strip().lower()
                current_stage = str(self.spec.stage).strip().lower()
                if ckpt_stage and ckpt_stage != current_stage:
                    print(f"[RESUME] run_id={self.spec.run_id} stage mismatch: "
                          f"checkpoint={ckpt_stage} current={current_stage}, "
                          f"discarding old checkpoint and re-running")
                    # Delete the old checkpoint and force a rerun
                    try:
                        ckpt_path.unlink()
                    except Exception:
                        pass
                    ckpt = None
                else:
                    print(f"[RESUME] run_id={self.spec.run_id} already done, returning cached result")
                    result.ok_submit = bool(ckpt.get("ok_submit", False))
                    result.genus_jobid = ckpt.get("genus_jobid")
                    result.innovus_jobid = ckpt.get("innovus_jobid")
                    result.genus_status = ckpt.get("genus_status")
                    result.innovus_status = ckpt.get("innovus_status")
                    return result
            elif phase in ("poll", "download"):
                resume_phase = phase
                result.genus_jobid = ckpt.get("genus_jobid")
                result.innovus_jobid = ckpt.get("innovus_jobid")
                result.ok_submit = bool(ckpt.get("ok_submit", False))
                print(f"[RESUME] run_id={self.spec.run_id} from phase={phase} "
                      f"genus_jobid={result.genus_jobid} innovus_jobid={result.innovus_jobid}")

        conn = RemoteConnection(
            self.creds,
            keepalive_sec=self.resilience.keepalive_sec,
            reconnect_retries=self.resilience.reconnect_retries,
            reconnect_retry_interval_sec=self.resilience.reconnect_retry_interval_sec,
        )
        try:
            self._append_log(master_log, f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] connect")


            # Add retries and semaphore-based throttling during connect,
            # to avoid overwhelming the jump-host SSH service when 24+ threads handshake at once
            _connect_max_retries = 5
            _connect_last_exc = None

            # Randomize startup offsets to reduce handshake bursts
            _stagger = random.uniform(0, 3.0)
            time.sleep(_stagger)

            for _ci in range(_connect_max_retries):
                # The semaphore ensures that at most N threads perform the SSH handshake simultaneously
                with _connect_semaphore:
                    try:
                        conn.connect()
                        _connect_last_exc = None
                        break
                    except Exception as _conn_exc:
                        _connect_last_exc = _conn_exc
                if _connect_last_exc is not None and _ci < _connect_max_retries - 1:
                    _cw = min(30 * (2 ** _ci), 300) + random.randint(0, 15)
                    self._append_log(master_log,
                        f"[WARN] connect attempt {_ci+1}/{_connect_max_retries} "
                        f"failed: {_connect_last_exc}, retrying in {_cw}s ...")
                    print(f"[WARN] connect attempt {_ci+1}/{_connect_max_retries} "
                          f"failed: {_connect_last_exc}, retrying in {_cw}s ...")
                    time.sleep(_cw)
            if _connect_last_exc is not None:
                err_msg = (f"RemoteCommError: connect failed after "
                           f"{_connect_max_retries} attempts: {_connect_last_exc}")
                result.error = err_msg
                self._append_log(master_log, f"[FATAL] {err_msg}")
                _save_checkpoint(ckpt_path, {"run_id": self.spec.run_id, "current_phase": "fatal_stop",
                    "stage": self.spec.stage,
                    "last_error_type": "RemoteCommError",
                    "last_error_message": str(_connect_last_exc)})
                raise RemoteCommError(err_msg) from _connect_last_exc
            _save_checkpoint(ckpt_path, {"run_id": self.spec.run_id, "current_phase": "connect",
                "stage": self.spec.stage})

            self._prepare_remote_layout(conn)
            self._upload_payload(conn, script_override=script_override)

            logs_dir = posixpath.join(self.spec.remote_root, "outbox/cadence/logs")

            # ---- Innovus-only mode: skip Genus entirely ----
            if self.spec.skip_genus:
                self._append_log(master_log, "# [INNOVUS-ONLY] Genus skipped, reusing shared genus output")
                result.genus_jobid = "reused"
                result.genus_status = "DONE"
                _save_checkpoint(ckpt_path, {
                    "run_id": self.spec.run_id,
                    "current_phase": "genus_done",
                    "stage": self.spec.stage,
                    "ok_submit": True,
                    "genus_jobid": result.genus_jobid,
                    "genus_status": result.genus_status,
                })

                # Create the remote genus -> shared_genus symlink
                if self.spec.shared_genus_dir:
                    local_genus_dir = posixpath.join(self.spec.remote_root, "outbox/cadence/genus")
                    link_cmd = (
                        f"mkdir -p {shell_quote(posixpath.dirname(local_genus_dir))} && "
                        f"ln -sfn {shell_quote(self.spec.shared_genus_dir)} {shell_quote(local_genus_dir)}"
                    )
                    rc_ln, out_ln, err_ln = conn.exec_cmd_resilient(link_cmd, timeout=30)
                    self._append_exec(master_log, "symlink_genus", rc_ln, out_ln, err_ln)
                    if rc_ln != 0:
                        self._append_log(master_log, f"[WARN] genus symlink failed: rc={rc_ln} err={err_ln[:200]}")

            else:
                # ---- Original Genus submission logic ----
                submit_genus_cmd = f"cd {shell_quote(self.spec.remote_root)} && bash submit_genus.sh"
                rc, out, err = self._submit_single_with_backpressure_retry(
                    conn,
                    master_log,
                    submit_genus_cmd,
                    stage_label="Genus",
                    jobid_filename="genus_jobid.txt",
                    run_slots_needed=max(1, int(self.spec.genus_ncpu)),
                )
                self._append_exec(master_log, "submit_genus", rc, out, err)
                if rc != 0:
                    result.error = f"submit_genus failed with rc={rc}"
                    return result

                result.genus_jobid = self._read_remote_text(conn, posixpath.join(logs_dir, "genus_jobid.txt"))
                if not result.genus_jobid:
                    result.genus_jobid = self._extract_jobid(out, key="Genus")

                self._append_log(master_log, f"# genus_jobid={result.genus_jobid}")
                _save_checkpoint(ckpt_path, {
                    "run_id": self.spec.run_id,
                    "current_phase": "genus_submit",
                    "stage": self.spec.stage,
                    "ok_submit": True,
                    "genus_jobid": result.genus_jobid,
                })

                result.genus_status, _ = self._poll_until_done(
                    conn,
                    master_log,
                    genus_jobid=result.genus_jobid,
                    innovus_jobid=None,
                )

                _save_checkpoint(ckpt_path, {
                    "run_id": self.spec.run_id,
                    "current_phase": "genus_done",
                    "stage": self.spec.stage,
                    "ok_submit": True,
                    "genus_jobid": result.genus_jobid,
                    "genus_status": result.genus_status,
                })

                # genus_only must be handled before the DONE check: even if Genus is interrupted by wallclock limits,
                # still download the generated .rpt files so the caller can decide whether they are usable
                if self.spec.genus_only:
                    self._download_remote_tree(conn, artifact_root)
                    self._merge_downloaded_logs(master_log, artifact_root)
                    result.ok_submit = True
                    result.innovus_status = "SKIPPED"
                    if str(result.genus_status or "").upper() != "DONE":
                        result.error = f"Genus not DONE in genus_only: {result.genus_status}"
                    return result

                if str(result.genus_status or "").upper() != "DONE":
                    result.error = f"Genus not DONE: {result.genus_status}"
                    self._download_remote_tree(conn, artifact_root)
                    self._merge_downloaded_logs(master_log, artifact_root)
                    return result

            submit_innovus_cmd = f"cd {shell_quote(self.spec.remote_root)} && bash submit_innovus.sh"
            rc, out, err = self._submit_single_with_backpressure_retry(
                conn,
                master_log,
                submit_innovus_cmd,
                stage_label="Innovus",
                jobid_filename="innovus_jobid.txt",
                run_slots_needed=max(1, int(self.spec.innovus_ncpu)),
            )
            self._append_exec(master_log, "submit_innovus", rc, out, err)
            if rc != 0:
                result.error = f"submit_innovus failed with rc={rc}"
                self._download_remote_tree(conn, artifact_root)
                self._merge_downloaded_logs(master_log, artifact_root)
                return result

            result.innovus_jobid = self._read_remote_text(conn, posixpath.join(logs_dir, "innovus_jobid.txt"))
            if not result.innovus_jobid:
                result.innovus_jobid = self._extract_jobid(out, key="Innovus")

            self._append_log(master_log, f"# innovus_jobid={result.innovus_jobid}")

            if not result.genus_jobid:
                result.genus_jobid = self._extract_jobid(out, key="Genus")
            if not result.innovus_jobid:
                result.innovus_jobid = self._extract_jobid(out, key="Innovus")

            self._append_log(master_log, f"# genus_jobid={result.genus_jobid} innovus_jobid={result.innovus_jobid}")
            _save_checkpoint(ckpt_path, {
                "run_id": self.spec.run_id,
                "current_phase": "submit",
                "stage": self.spec.stage,
                "ok_submit": True,
                "genus_jobid": result.genus_jobid,
                "genus_status": result.genus_status,
                "innovus_jobid": result.innovus_jobid,
            })

            try:
                result.genus_status, result.innovus_status = self._poll_until_done(
                    conn,
                    master_log,
                    genus_jobid=result.genus_jobid,
                    innovus_jobid=result.innovus_jobid,
                )
                _save_checkpoint(ckpt_path, {
                    "run_id": self.spec.run_id,
                    "current_phase": "poll",
                    "stage": self.spec.stage,
                    "ok_submit": True,
                    "genus_jobid": result.genus_jobid,
                    "innovus_jobid": result.innovus_jobid,
                    "genus_status": result.genus_status,
                    "innovus_status": result.innovus_status,
                })
            except RemoteSessionLostError as exc:
                self._append_log(master_log, f"[WARN] session lost during poll: {exc}")
                reconnected = False
                max_retries = max(self.resilience.reconnect_retries, 10)
                base_interval = self.resilience.reconnect_retry_interval_sec
                for i in range(max_retries):
                    wait_sec = min(base_interval * (2 ** min(i, 5)), 300) + random.randint(0, 15)
                    print(f"[WARN] SSH session lost, trying reconnect ({i+1}/{max_retries}) in {wait_sec}s ...")
                    try:
                        time.sleep(wait_sec)
                        conn.reconnect()
                        reconnected = True
                        print("[INFO] reconnect succeeded, continue polling existing jobs")
                        break
                    except RemoteCommError as re_exc:
                        print(f"[WARN] reconnect attempt {i+1} failed: {re_exc}")
                        continue
                if reconnected:
                    result.genus_status, result.innovus_status = self._poll_until_done(
                        conn,
                        master_log,
                        genus_jobid=result.genus_jobid,
                        innovus_jobid=result.innovus_jobid,
                    )
                else:
                    err_msg = f"SSH session lost during poll, reconnect failed after {self.resilience.reconnect_retries} attempts"
                    result.error = err_msg
                    self._append_log(master_log, f"[FATAL] {err_msg}")
                    _save_checkpoint(ckpt_path, {
                        "run_id": self.spec.run_id,
                        "current_phase": "session_lost",
                        "stage": self.spec.stage,
                        "ok_submit": True,
                        "genus_jobid": result.genus_jobid,
                        "innovus_jobid": result.innovus_jobid,
                        "resume_from_phase": "poll",
                    })
                    raise RemoteSessionLostError(err_msg) from exc

            except RemoteSessionLostError as exc:
                self._append_log(master_log, f"[WARN] session lost during poll: {exc}")
                reconnected = False
                max_retries = max(self.resilience.reconnect_retries, 10)  # At least 10 attempts
                base_interval = self.resilience.reconnect_retry_interval_sec
                for i in range(max_retries):
                    wait_sec = min(base_interval * (2 ** min(i, 5)), 300) + random.randint(0, 15)
                    print(f"[WARN] SSH session lost, trying reconnect ({i+1}/{max_retries}) in {wait_sec}s ...")
                    try:
                        time.sleep(wait_sec)
                        conn.reconnect()
                        reconnected = True
                        print("[INFO] reconnect succeeded, continue polling existing jobs")
                        break
                    except RemoteCommError as re_exc:
                        print(f"[WARN] reconnect attempt {i+1} failed: {re_exc}")
                        continue
                if reconnected:
                    result.genus_status, result.innovus_status = self._poll_until_done(
                        conn, master_log,
                        genus_jobid=result.genus_jobid, innovus_jobid=result.innovus_jobid,
                    )
                else:
                    err_msg = f"SSH session lost during poll, reconnect failed after {self.resilience.reconnect_retries} attempts"
                    result.error = err_msg
                    self._append_log(master_log, f"[FATAL] {err_msg}")
                    _save_checkpoint(ckpt_path, {"run_id": self.spec.run_id, "current_phase": "session_lost",
                        "stage": self.spec.stage,
                        "ok_submit": True, "genus_jobid": result.genus_jobid, "innovus_jobid": result.innovus_jobid,
                        "resume_from_phase": "poll"})
                    raise RemoteSessionLostError(err_msg) from exc

            self._download_remote_tree(conn, artifact_root)
            self._merge_downloaded_logs(master_log, artifact_root)

            # Save a "done" checkpoint so resume logic can correctly skip completed runs
            _save_checkpoint(ckpt_path, {
                "run_id": self.spec.run_id,
                "current_phase": "done",
                "stage": self.spec.stage,
                "ok_submit": True,
                "genus_jobid": result.genus_jobid,
                "innovus_jobid": result.innovus_jobid,
                "genus_status": result.genus_status,
                "innovus_status": result.innovus_status,
            })

            if self.spec.cleanup_remote_after_download:
                cleanup_cmd = f"rm -rf -- {shell_quote(self.spec.remote_root)}"
                rc, out, err = conn.exec_cmd_resilient(cleanup_cmd, timeout=120)
                self._append_exec(master_log, "cleanup_remote", rc, out, err)

            return result

        except (RemoteCommError, RemoteSessionLostError, SchedulerBackpressureError, ControlledStop) as exc:
            # Log to this run's cadence.log BEFORE re-raising to main.py
            self._append_log(master_log, f"\n[FATAL] {type(exc).__name__}: {exc}")
            self._append_log(master_log, traceback.format_exc())
            raise
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            self._append_log(master_log, traceback.format_exc())
            return result
        finally:
            conn.close()
            self._append_log(master_log, f"# [{time.strftime('%Y-%m-%d %H:%M:%S')}] connection closed")




    def _query_current_user_bu_summary(self, conn: RemoteConnection) -> Tuple[Optional[int], Optional[int]]:
        # Use a single-line command to avoid nested single quotes
        cmd = 'u=$(whoami); bu 2>/dev/null | awk -v u="$u" \'$1==u {print $5,$6; exit}\''
        try:
            rc, out, err = conn.exec_cmd_resilient(f"bash -c {shell_quote(cmd)}", timeout=30)
            txt = (out or "").strip()
            if not txt:
                return None, None
            parts = txt.split()
            if len(parts) < 2:
                return None, None
            pend_jobs = int(parts[0])
            run_jobs = int(parts[1])
            return pend_jobs, run_jobs
        except Exception:
            return None, None

    def _query_current_user_running_slots(self, conn: RemoteConnection) -> Optional[int]:
        cmd = 'bjobs -u "$USER" -noheader -o "stat slots" 2>/dev/null | awk \'$1=="RUN" && $2~/^[0-9]+$/ {s+=$2} END{print s+0}\''
        try:
            rc, out, err = conn.exec_cmd_resilient(f"bash -c {shell_quote(cmd)}", timeout=30)
            txt = (out or "").strip()
            if not txt:
                return None
            return int(txt.splitlines()[-1].strip())
        except Exception:
            return None

    def _wait_for_lsf_budget(
        self,
        conn: RemoteConnection,
        master_log: Path,
        *,
        stage_label: str,
        pend_jobs_needed: int = 1,
        run_jobs_needed: int = 1,
        run_slots_needed: int = 1,
    ) -> None:
        pend_limit = max(1, int(self.resilience.lsf_pending_jobs_soft_limit))
        run_jobs_limit = max(1, int(self.resilience.lsf_running_jobs_soft_limit))
        run_slots_limit = max(1, int(self.resilience.lsf_running_slots_soft_limit))
        interval = max(1, int(self.resilience.lsf_check_interval_sec))
        jitter_max = max(0, int(self.resilience.lsf_check_jitter_sec))
        max_wait = max(60, int(self.resilience.max_submit_backpressure_wait_sec))
        waited = 0

        while True:
            pend_jobs, run_jobs = self._query_current_user_bu_summary(conn)
            run_slots = self._query_current_user_running_slots(conn)

            if pend_jobs is None or run_jobs is None or run_slots is None:
                self._append_log(
                    master_log,
                    f"[LSF_GATE] {stage_label}: "
                    f"pend_jobs={pend_jobs}, run_jobs={run_jobs}, run_slots={run_slots}, "
                    f"query_incomplete -> allow submit"
                )
                clear_backpressure()
                return

            pend_ok = (pend_jobs + max(1, int(pend_jobs_needed)) <= pend_limit)
            run_jobs_ok = (run_jobs + max(1, int(run_jobs_needed)) <= run_jobs_limit)
            run_slots_ok = (run_slots + max(1, int(run_slots_needed)) <= run_slots_limit)

            if pend_ok and run_jobs_ok and run_slots_ok:
                clear_backpressure()
                return

            signal_backpressure()

            sleep_sec = interval + random.randint(0, jitter_max)
            waited += sleep_sec

            msg = (
                f"[LSF_GATE] {stage_label}: "
                f"pend_jobs={pend_jobs}/{pend_limit} (+{pend_jobs_needed}), "
                f"run_jobs={run_jobs}/{run_jobs_limit} (+{run_jobs_needed}), "
                f"run_slots={run_slots}/{run_slots_limit} (+{run_slots_needed}), "
                f"sleep={sleep_sec}s"
            )
            self._append_log(master_log, msg)
            print(msg)

            if waited > max_wait:
                fatal_msg = (
                    f"{stage_label} waiting for LSF budget timed out after {waited}s "
                    f"(pend_limit={pend_limit}, run_jobs_limit={run_jobs_limit}, run_slots_limit={run_slots_limit})"
                )
                self._append_log(master_log, f"[FATAL] {fatal_msg}")
                raise SchedulerBackpressureError(fatal_msg)

            time.sleep(sleep_sec)

    def _submit_single_with_backpressure_retry(
        self,
        conn: RemoteConnection,
        master_log: Path,
        submit_cmd: str,
        *,
        stage_label: str,
        jobid_filename: str,
        run_slots_needed: int,
    ) -> Tuple[int, str, str]:
        max_wait = max(60, int(self.resilience.max_submit_backpressure_wait_sec))
        interval = max(1, int(self.resilience.submit_retry_interval_sec))
        waited = 0
        logs_dir = posixpath.join(self.spec.remote_root, "outbox/cadence/logs")
        jobid_path = posixpath.join(logs_dir, jobid_filename)

        while True:
            try:
                if conn.path_exists(jobid_path):
                    clear_backpressure()
                    print(f"[INFO] {stage_label} already submitted for {self.spec.run_id}, skip re-submit")
                    return 0, "", ""
            except Exception:
                pass

            self._wait_for_lsf_budget(
                conn,
                master_log,
                stage_label=stage_label,
                pend_jobs_needed=1,
                run_jobs_needed=1,
                run_slots_needed=max(1, int(run_slots_needed)),
            )

            try:
                rc, out, err = conn.exec_cmd_resilient(submit_cmd, timeout=600)
            except RemoteCommandTimeoutError as timeout_exc:
                signal_backpressure()
                sleep_sec = interval + random.randint(0, 30)
                waited += sleep_sec
                if waited > max_wait:
                    msg = f"{stage_label} submit timeout after {waited}s (max={max_wait}s)"
                    self._append_log(master_log, f"[FATAL] {msg}")
                    raise SchedulerBackpressureError(msg) from timeout_exc
                self._append_log(master_log, f"[WARN] {stage_label} submit timeout: {timeout_exc}")
                print(f"[WARN] {stage_label} submit timeout, retry in {sleep_sec}s")
                time.sleep(sleep_sec)
                continue

            combined = (out + err).lower()

            if _BACKPRESSURE_MARKER not in combined:
                if rc == 0:
                    clear_backpressure()
                return rc, out, err

            signal_backpressure()
            sleep_sec = interval + random.randint(0, 30)
            waited += sleep_sec
            if waited > max_wait:
                msg = f"{stage_label} backpressure timeout after {waited}s (max={max_wait}s)"
                self._append_log(master_log, f"[FATAL] {msg}")
                raise SchedulerBackpressureError(msg)

            self._append_log(master_log, f"[WARN] {stage_label} backpressure retry in {sleep_sec}s")
            print(f"[WARN] {stage_label} backpressure, retry in {sleep_sec}s")
            time.sleep(sleep_sec)


    def _prepare_remote_layout(self, conn: RemoteConnection) -> None:
        # Clean old scripts and Innovus reports first to avoid interference from stale files
        for clean_rel in ("outbox/cadence/scripts", "outbox/cadence/innovus"):
            clean_dir = posixpath.join(self.spec.remote_root, clean_rel)
            try:
                conn.exec_cmd_resilient(f"rm -rf -- {shell_quote(clean_dir)}", timeout=30)
            except Exception:
                pass

        for rel in (
            "",
            "outbox",
            "outbox/cadence",
            "outbox/cadence/scripts",
            "outbox/cadence/logs",
            "outbox/cadence/genus",
            "outbox/cadence/innovus",
        ):
            remote_dir = self.spec.remote_root if rel == "" else posixpath.join(self.spec.remote_root, rel)
            conn.ensure_remote_dir(remote_dir)

    def _upload_payload(self, conn: RemoteConnection, *, script_override: Optional[str]) -> None:
        scripts_dir = posixpath.join(self.spec.remote_root, "outbox/cadence/scripts")
        builder = CadenceRemoteScriptBuilder(self.spec)
        conn.put_content(builder.build_genus_tcl(), posixpath.join(scripts_dir, "run_genus.tcl"))
        conn.put_content(builder.build_innovus_tcl(), posixpath.join(scripts_dir, "run_innovus.tcl"))
        conn.put_content(builder.build_run_env_csh(), posixpath.join(scripts_dir, "run_env.csh"))
        conn.put_content(builder.build_run_env_sh(), posixpath.join(scripts_dir, "run_env.sh"))

        conn.put_content(builder.build_submit_genus_sh(), posixpath.join(self.spec.remote_root, "submit_genus.sh"))
        conn.put_content(builder.build_submit_innovus_sh(), posixpath.join(self.spec.remote_root, "submit_innovus.sh"))

        conn.exec_cmd_resilient(
            f"chmod +x -- "
            f"{shell_quote(posixpath.join(self.spec.remote_root, 'submit_genus.sh'))} "
            f"{shell_quote(posixpath.join(self.spec.remote_root, 'submit_innovus.sh'))}",
            timeout=30,
        )
        conn.exec_cmd_resilient(
            f"chmod +x -- {shell_quote(posixpath.join(scripts_dir, 'run_env.sh'))} "
            f"{shell_quote(posixpath.join(scripts_dir, 'run_env.csh'))}",
            timeout=30,
        )

        if script_override and str(script_override).strip():
            conn.put_content(str(script_override).strip() + "\n", posixpath.join(scripts_dir, "override.tcl"))


        if script_override and str(script_override).strip():
            conn.put_content(str(script_override).strip() + "\n", posixpath.join(scripts_dir, "override.tcl"))

    def _poll_until_done(
        self,
        conn: RemoteConnection,
        master_log: Path,
        *,
        genus_jobid: Optional[str],
        innovus_jobid: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        pending: Dict[str, str] = {}
        if genus_jobid:
            pending[str(genus_jobid)] = "Genus"
        if innovus_jobid:
            pending[str(innovus_jobid)] = "Innovus"

        last_status: Dict[str, str] = {}

        for attempt in range(1, max(1, int(self.spec.max_poll_attempts)) + 1):
            if not pending:
                break

            # Print in real time for easier polling diagnostics
            # print(f"[POLL] {self.spec.run_id} attempt={attempt} pending={list(pending.keys())}")

            now = time.strftime("%Y-%m-%d %H:%M:%S")
            self._append_log(master_log, f"# [poll {attempt}] {now}")

            done_ids: List[str] = []
            for jobid, name in list(pending.items()):
                status = self._check_job_status(conn, jobid)
                last_status[jobid] = status
                self._append_log(master_log, f"#   {name}(jobid={jobid}) status={status}")
                if status in ("DONE", "EXIT", "NOT_FOUND"):
                    done_ids.append(jobid)

            for jobid in done_ids:
                pending.pop(jobid, None)

            if pending:
                time.sleep(max(1, int(self.spec.poll_interval_sec)))

        if pending:
            self._append_log(master_log, f"# poll timeout; unfinished jobs={json.dumps(pending, ensure_ascii=False)}")

        genus_status = last_status.get(str(genus_jobid), None) if genus_jobid else None
        innovus_status = last_status.get(str(innovus_jobid), None) if innovus_jobid else None
        return genus_status, innovus_status

    def _check_job_status(self, conn: RemoteConnection, jobid: str) -> str:
        # Handle the reused placeholder and return a terminal status directly
        if str(jobid).lower() == "reused":
            return "DONE"
        
        rc, out, err = conn.exec_cmd_resilient(f"bjobs -o 'stat' -noheader {shell_quote(jobid)} 2>&1", timeout=60)
        combined = (out + err).strip()
        # Add detection for "illegal job id"
        if "not found" in combined.lower() or "not exist" in combined.lower() or "illegal job id" in combined.lower():
            return "NOT_FOUND"
        if not combined:
            return "UNKNOWN"
        status = combined.splitlines()[-1].strip().upper()
        if status in ("RUN", "PEND", "DONE", "EXIT", "SSUSP", "USUSP", "PSUSP"):
            return status
        return status[:120]

    def _download_remote_tree(self, conn: RemoteConnection, artifact_root: Path) -> None:
        """Download only .rpt / .log / .txt / .out / .err files under outbox/cadence."""
        artifact_root.mkdir(parents=True, exist_ok=True)

        ALLOWED_SUFFIXES = {".rpt", ".log", ".txt", ".out", ".err"}

        subdirs_to_download = [
            "outbox/cadence/genus",
            "outbox/cadence/innovus",
            "outbox/cadence/logs",
            "outbox/cadence/scripts",   # Also download scripts for postmortem diagnostics
        ]

        downloaded_any = False
        for subdir in subdirs_to_download:
            remote_path = posixpath.join(self.spec.remote_root, subdir)
            local_path = artifact_root / subdir
            try:
                if conn.path_exists(remote_path):
                    self._download_filtered(conn, remote_path, local_path, ALLOWED_SUFFIXES)
                    downloaded_any = True
                else:
                    print(f"[WARN] remote dir not found, skip: {remote_path}")
            except Exception as e:
                print(f"[WARN] failed to download {remote_path}: {e}")

        if not downloaded_any:
            print(f"[ERROR] No result files found on remote server under "
                f"{self.spec.remote_root}/outbox/cadence/")

    def _download_filtered(
        self,
        conn: RemoteConnection,
        remote_dir: str,
        local_dir: Path,
        allowed_suffixes: set,
    ) -> None:
        """Recursively download a remote directory, but only fetch files whose suffix is in allowed_suffixes."""
        client = conn._require_target()
        sftp = client.open_sftp()
        try:
            self._download_filtered_impl(sftp, remote_dir, local_dir, allowed_suffixes)
        finally:
            sftp.close()

    @staticmethod
    def _download_filtered_impl(
        sftp: Any,
        remote_dir: str,
        local_dir: Path,
        allowed_suffixes: set,
    ) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            items = sftp.listdir_attr(remote_dir)
        except Exception:
            return
        for item in items:
            remote_path = posixpath.join(remote_dir, item.filename)
            local_path = local_dir / item.filename
            if stat.S_ISDIR(item.st_mode):
                # Skip directories such as .enc.dat
                if item.filename.endswith(".enc.dat"):
                    continue
                CadenceRemoteJobRunner._download_filtered_impl(
                    sftp, remote_path, local_path, allowed_suffixes
                )
            else:
                # Download only allowed suffixes
                suffix = posixpath.splitext(item.filename)[1].lower()
                if suffix in allowed_suffixes:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_path, str(local_path))


    def _merge_downloaded_logs(self, master_log: Path, artifact_root: Path) -> None:
        log_dir = artifact_root / "outbox" / "cadence" / "logs"
        if not log_dir.is_dir():
            return

        parts: List[Path] = []
        for pattern in ("*.log", "*.out", "*.err", "*.txt", "*.csh"):
            parts.extend(sorted(log_dir.glob(pattern)))
        if not parts:
            return

        with master_log.open("a", encoding="utf-8", errors="replace") as fp:
            for path in parts:
                fp.write("\n\n")
                fp.write(f"# ===== downloaded: {path.relative_to(artifact_root)} =====\n")
                try:
                    fp.write(path.read_text(encoding="utf-8", errors="replace"))
                except Exception as exc:
                    fp.write(f"[read error] {type(exc).__name__}: {exc}\n")

    def _read_remote_text(self, conn: RemoteConnection, remote_path: str) -> Optional[str]:
        rc, out, err = conn.exec_cmd_resilient(f"cat {shell_quote(remote_path)} 2>/dev/null", timeout=30)
        if rc != 0:
            return None
        text = (out or "").strip()
        return text or None

    def _extract_jobid(self, text: str, *, key: str) -> Optional[str]:
        import re

        pattern = re.compile(rf"{key}\s+jobid\s*=\s*(\d+)", re.IGNORECASE)
        match = pattern.search(text or "")
        return match.group(1) if match else None

    def _append_exec(self, log_path: Path, title: str, rc: int, out: str, err: str) -> None:
        text = (
            f"\n# ===== {title} =====\n"
            f"# rc={rc}\n"
            f"[stdout]\n{out}\n"
            f"[stderr]\n{err}\n"
        )
        self._append_log(log_path, text)

    def _append_log(self, log_path: Path, text: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as fp:
            fp.write(str(text))
            if not str(text).endswith("\n"):
                fp.write("\n")
